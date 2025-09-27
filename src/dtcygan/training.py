"""Training utilities for the temporal CycleGAN on synthetic datasets."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import yaml
import json
from datetime import datetime

from dtcygan.synthetic import generate_dataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@dataclass
class Config:
    seq_len: int = 8
    batch_size: int = 32
    epochs: int = 200
    lr_g: float = 4.0e-5
    lr_d: float = 5.0e-5
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cyc: float = 20.0
    lambda_id: float = 30.0
    g_hidden: int = 256
    d_hidden: int = 128
    num_layers: int = 3
    seed: int = 42
    device: Optional[str] = None
    adv_loss: str = "lsgan"
    criterion: str = "l1"
    clinical_features: Dict[str, Any] = field(default_factory=dict)
    treatment_features: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    allowed = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return Config(**filtered)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_adversarial_loss(name: str) -> nn.Module:
    key = (name or "bce").lower()
    if key in {"bce", "binary_cross_entropy", "bcewithlogits"}:
        return nn.BCELoss()
    if key in {"l2", "mse", "lsgan"}:
        return nn.MSELoss()
    raise ValueError(f"Unsupported adversarial loss: {name}")


def get_criterion(name: str) -> nn.Module:
    key = (name or "l1").lower()
    if key in {"l1", "mae"}:
        return nn.L1Loss()
    if key in {"l2", "mse"}:
        return nn.MSELoss()
    if key in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported reconstruction loss: {name}")


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Return a safe torch.device, falling back to CPU when CUDA is unsuitable."""
    def _cuda_supported(dev: torch.device) -> bool:
        try:
            torch.zeros(1, device=dev)
            test_lstm = nn.LSTM(1, 1).to(dev)
            test_inp = torch.zeros(1, 1, 1, device=dev)
            test_lstm(test_inp)
            return True
        except RuntimeError as exc:
            print(f"[dtcygan] CUDA device test failed ({exc}); falling back to CPU.")
            return False

    if preferred:
        try:
            device = torch.device(preferred)
            if device.type == "cuda":
                if not torch.cuda.is_available() or not _cuda_supported(device):
                    return torch.device("cpu")
            return device
        except Exception as exc:  # noqa: BLE001
            print(f"[dtcygan] Unable to honour preferred device '{preferred}': {exc}; using CPU.")
            return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if _cuda_supported(device):
            return device
    return torch.device("cpu")


class LSTMGenerator(nn.Module):
    """LSTM-based generator with mask-aware conditioning."""

    def __init__(self, input_dim: int, cond_input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        combined_dim = (input_dim + cond_input_dim) * 2
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_aug = torch.cat([x, m_x], dim=-1)
        cond_aug = torch.cat([cond, m_cond], dim=-1)
        combined = torch.cat([x_aug, cond_aug], dim=-1)

        if lengths is not None:
            packed = pack_padded_sequence(
                combined, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, h = self.lstm(packed, hidden)
            lstm_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=combined.size(1)
            )
        else:
            lstm_out, h = self.lstm(combined, hidden)

        out = self.fc(lstm_out)
        return out, h


class LSTMDiscriminator(nn.Module):
    """Mask-aware LSTM discriminator returning per-sequence scores."""

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        total_dim = (input_dim + cond_dim) * 2
        self.lstm = nn.LSTM(
            input_size=total_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_aug = torch.cat([x, m_x], dim=-1)
        cond_aug = torch.cat([cond, m_cond], dim=-1)
        combined = torch.cat([x_aug, cond_aug], dim=-1)

        if lengths is not None:
            packed = pack_padded_sequence(
                combined, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(combined)

        last_hidden = h_n[-1]
        score = self.classifier(last_hidden).squeeze(-1)
        return score


class SyntheticSequenceDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, Any],
        seq_len: int,
        spec_clinical: Optional[Dict[str, Any]] = None,
        spec_treatment: Optional[Dict[str, Any]] = None,
    ):
        patients = data.get("patients", [])
        if not isinstance(patients, list) or not patients:
            raise ValueError("Synthetic dataset must contain a non-empty 'patients' list.")

        self.seq_len = seq_len
        self.patients = patients
        self.spec_clinical = spec_clinical or {}
        self.spec_treatment = spec_treatment or {}
        self.clin_columns, self.clin_categorical, self.clin_categories, self.clin_ord_maps = self._configure_columns(
            "clinical", self.spec_clinical
        )
        self.treat_columns, self.treat_categorical, self.treat_categories, self.treat_ord_maps = self._configure_columns(
            "treatment", self.spec_treatment
        )
        self.cond_columns = self._infer_condition_columns()
        self.clin_dim = len(self.clin_columns)
        self.treat_dim = len(self.treat_columns)
        self.cond_dim = len(self.cond_columns)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, sort_keys=True)
            except TypeError:
                return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def _collect_values(self, key: str) -> Dict[str, set[Any]]:
        raw_values: Dict[str, set[Any]] = {}
        for patient in self.patients:
            records = patient.get(key, [])
            df = pd.DataFrame(records)
            df = df.drop(columns=["_id", "timestep"], errors="ignore")
            for col in df.columns:
                values = (df[col].dropna().apply(self._stringify)).tolist()
                raw_values.setdefault(col, set()).update(values)
        return raw_values

    def _configure_columns(
        self, key: str, spec: Dict[str, Any]
    ) -> Tuple[List[str], set[str], Dict[str, List[str]], Dict[str, Dict[str, float]]]:
        raw_values = self._collect_values(key)
        ordinal_maps: Dict[str, Dict[str, float]] = {}

        if spec:
            numeric = list(spec.get("numeric", []))
            one_hot = list(spec.get("one_hot", []))
            ordinal_spec = spec.get("ordinal", {}) or {}
            ordinal_cols = list(ordinal_spec.keys())

            columns: List[str] = []
            seen: set[str] = set()
            for col in numeric + one_hot + ordinal_cols:
                if col not in seen:
                    columns.append(col)
                    seen.add(col)

            categorical = set(one_hot)
            categories: Dict[str, List[str]] = {}
            for col in one_hot:
                values = raw_values.get(col, set())
                categories[col] = sorted({self._stringify(v) for v in values})

            for col, mapping in ordinal_spec.items():
                ordinal_maps[col] = {self._stringify(k): float(v) for k, v in mapping.items()}

            return columns, categorical, categories, ordinal_maps

        # Fallback to inferred schema
        columns = sorted(raw_values.keys())
        categorical: set[str] = set()
        categories: Dict[str, List[str]] = {}
        for col, values in raw_values.items():
            if any(isinstance(v, (str, bool, dict, list, tuple, set)) for v in values):
                categorical.add(col)
                categories[col] = sorted({self._stringify(v) for v in values})
        return columns, categorical, categories, ordinal_maps

    def _infer_condition_columns(self) -> List[str]:
        columns: set[str] = set()
        for patient in self.patients:
            records = patient.get("actual_treatment", [])
            df = pd.DataFrame(records)
            df = df.drop(columns=["_id", "timestep"], errors="ignore")
            columns.update(df.columns.tolist())
        ordered = sorted(columns)
        if not ordered:
            raise ValueError("Conditioning data must include at least one column.")
        return ordered

    @staticmethod
    def _pad(array: np.ndarray, seq_len: int) -> np.ndarray:
        if array.shape[0] >= seq_len:
            return array[:seq_len]
        pad = np.zeros((seq_len - array.shape[0], array.shape[1]), dtype=array.dtype)
        return np.vstack([array, pad])

    def _encode_records(
        self,
        records: Iterable[Dict[str, Any]],
        columns: List[str],
        categorical: set[str],
        categories: Dict[str, List[str]],
        ordinal_maps: Dict[str, Dict[str, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame(records)
        df = df.drop(columns=["_id", "timestep"], errors="ignore")
        df = df.reindex(columns=columns)
        if df.empty:
            return np.zeros((0, len(columns)), dtype=np.float32), np.zeros((0, len(columns)), dtype=np.float32)

        values = np.zeros((len(df), len(columns)), dtype=np.float32)
        mask = np.zeros_like(values)
        for idx, col in enumerate(columns):
            series = df[col]
            if series is None:
                continue
            present = series.notna().to_numpy()
            mask[present, idx] = 1.0
            if col in ordinal_maps:
                normalized = series.apply(self._stringify)
                mapped = normalized.map(ordinal_maps[col]).fillna(0.0).astype(np.float32)
                values[:, idx] = mapped.to_numpy()
                continue
            if col in categorical:
                normalized = series.apply(self._stringify)
                cat = pd.Categorical(normalized, categories=categories.get(col, []))
                codes = cat.codes.astype(np.float32)
                codes[codes < 0] = 0.0
                values[:, idx] = codes
            else:
                numeric = pd.to_numeric(series, errors="coerce").astype(np.float32)
                numeric = np.nan_to_num(numeric, nan=0.0)
                values[:, idx] = numeric
        return values, mask

    def _encode_actual(self, records: Iterable[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame(records)
        df = df.drop(columns=["_id", "timestep"], errors="ignore")
        df = df.reindex(columns=self.cond_columns)
        if df.empty:
            arr = np.zeros((0, self.cond_dim), dtype=np.float32)
            return arr, arr
        mask = (~df.isna()).astype(np.float32).to_numpy()
        arr = df.fillna(0.0).astype(np.float32).to_numpy()
        return arr, mask

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient = self.patients[idx]
        clin_vals, clin_mask = self._encode_records(
            patient.get("clinical", []), self.clin_columns, self.clin_categorical, self.clin_categories, self.clin_ord_maps
        )
        treat_vals, treat_mask = self._encode_records(
            patient.get("treatment", []), self.treat_columns, self.treat_categorical, self.treat_categories, self.treat_ord_maps
        )
        actual_vals, actual_mask = self._encode_actual(patient.get("actual_treatment", []))
        raw_len = actual_vals.shape[0]
        if raw_len:
            base_vals = actual_vals[:1]
            base_mask = actual_mask[:1]
            actual_vals = np.repeat(base_vals, raw_len, axis=0)
            actual_mask = np.repeat(base_mask, raw_len, axis=0)

        clin_vals = self._pad(clin_vals, self.seq_len)
        treat_vals = self._pad(treat_vals, self.seq_len)
        actual_vals = self._pad(actual_vals, self.seq_len)
        actual_mask = self._pad(actual_mask, self.seq_len)
        clin_mask = self._pad(clin_mask, self.seq_len)
        treat_mask = self._pad(treat_mask, self.seq_len)

        return {
            "x_clin": torch.from_numpy(clin_vals),
            "x_treat": torch.from_numpy(treat_vals),
            "actual_treatment": torch.from_numpy(actual_vals),
            "mask_clin": torch.from_numpy(clin_mask),
            "mask_treat": torch.from_numpy(treat_mask),
            "mask_actual": torch.from_numpy(actual_mask),
        }


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    mask = mask.expand_as(pred)
    if loss_type == "l1":
        loss = torch.abs(pred - target) * mask
    else:
        loss = ((pred - target) ** 2) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def randomize_multiple_one_hot(
    cond: torch.Tensor,
    mask: torch.Tensor,
    extra_ones: int = 1,
) -> torch.Tensor:
    if cond.shape != mask.shape:
        raise ValueError("Condition tensor and mask must share the same shape.")

    randomized = cond.clone()
    batch, steps, features = cond.shape
    device = cond.device

    for _ in range(max(1, extra_ones)):
        idx = torch.randint(features, (batch, steps, 1), device=device)
        additions = torch.zeros_like(randomized).scatter(2, idx, 1.0)
        randomized = torch.clamp(randomized + additions, max=1.0)

    randomized = randomized * mask
    valid = mask.sum(dim=2, keepdim=True) > 0
    zero_mask = valid & (randomized.sum(dim=2, keepdim=True) == 0)
    if zero_mask.any():
        idx = torch.randint(features, (batch, steps, 1), device=device)
        additions = torch.zeros_like(randomized).scatter(2, idx, 1.0)
        randomized = torch.where(zero_mask, additions, randomized)
        randomized = randomized * mask

    return randomized


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> Tuple[Dataset, Dataset]:
    val_size = max(1, int(len(dataset) * val_fraction))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def create_dataloaders(
    dataset: SyntheticSequenceDataset, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    train_subset, val_subset = split_dataset(dataset, val_fraction=0.2, seed=seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    if hasattr(train_subset, "indices"):
        train_indices = list(train_subset.indices)
    else:
        train_indices = list(range(len(train_subset)))
    if hasattr(val_subset, "indices"):
        val_indices = list(val_subset.indices)
    else:
        val_indices = list(range(len(val_subset)))
    return train_loader, val_loader, train_indices, val_indices


def train_epoch(
    Gx: nn.Module,
    Gy: nn.Module,
    Dx: nn.Module,
    Dy: nn.Module,
    loader: DataLoader,
    optimizer_Gx: torch.optim.Optimizer,
    optimizer_Gy: torch.optim.Optimizer,
    optimizer_Dx: torch.optim.Optimizer,
    optimizer_Dy: torch.optim.Optimizer,
    criterion_adv: nn.Module,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    Gx.train()
    Gy.train()
    Dx.train()
    Dy.train()

    totals = {k: 0.0 for k in [
        "g_loss",
        "d_loss",
        "cycle_loss",
        "identity_loss",
        "gx_adv",
        "gy_adv",
        "dx_loss",
        "dy_loss",
    ]}

    for batch in loader:
        cl_real = batch["x_clin"].to(device)
        tr_real = batch["x_treat"].to(device)
        tr_actual = batch["actual_treatment"].to(device)
        mask_clin = batch["mask_clin"].to(device)
        mask_treat = batch["mask_treat"].to(device)
        mask_actual = batch["mask_actual"].to(device)

        step_mask = ((mask_clin.sum(dim=2, keepdim=True) + mask_treat.sum(dim=2, keepdim=True)) > 0).float()
        lengths = step_mask.squeeze(-1).sum(dim=1).clamp(min=1).long()

        cl_real = cl_real * mask_clin
        tr_real = tr_real * mask_treat
        tr_actual = tr_actual * mask_actual
        cond_mask = torch.ones_like(tr_actual)
        tr_counter = randomize_multiple_one_hot(tr_actual, mask_actual, extra_ones=1)

        optimizer_Dx.zero_grad()
        optimizer_Dy.zero_grad()

        fake_treat_detached, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
        fake_treat_detached = fake_treat_detached * mask_treat
        rec_treat_detached, _ = Gy(fake_treat_detached, mask_treat, tr_actual, cond_mask, lengths)
        rec_treat_detached = rec_treat_detached * mask_treat

        main_real = torch.cat((cl_real, tr_real), dim=2)
        main_mask = torch.cat((mask_clin, mask_treat), dim=2)
        main_fake_x = torch.cat((cl_real, fake_treat_detached.detach()), dim=2)
        main_fake_y = torch.cat((cl_real, rec_treat_detached.detach()), dim=2)

        pred_real_x = Dx(main_real, main_mask, tr_actual, cond_mask, lengths)
        pred_fake_x = Dx(main_fake_x, main_mask, tr_counter, cond_mask, lengths)
        pred_real_y = Dy(main_real, main_mask, tr_actual, cond_mask, lengths)
        pred_fake_y = Dy(main_fake_y, main_mask, tr_actual, cond_mask, lengths)

        ones_seq = torch.ones_like(pred_real_x)
        zeros_seq = torch.zeros_like(pred_real_x)

        loss_Dx = 0.5 * (
            criterion_adv(pred_real_x, ones_seq) + criterion_adv(pred_fake_x, zeros_seq)
        )
        loss_Dy = 0.5 * (
            criterion_adv(pred_real_y, ones_seq) + criterion_adv(pred_fake_y, zeros_seq)
        )
        (loss_Dx + loss_Dy).backward()
        optimizer_Dx.step()
        optimizer_Dy.step()

        for p in Dx.parameters():
            p.requires_grad_(False)
        for p in Dy.parameters():
            p.requires_grad_(False)

        optimizer_Gx.zero_grad()
        optimizer_Gy.zero_grad()

        fake_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
        fake_treat = fake_treat * mask_treat
        rec_treat, _ = Gy(fake_treat, mask_treat, tr_actual, cond_mask, lengths)
        rec_treat = rec_treat * mask_treat

        main_fake_x = torch.cat((cl_real, fake_treat), dim=2)
        main_fake_y = torch.cat((cl_real, rec_treat), dim=2)

        pred_fake_x = Dx(main_fake_x, main_mask, tr_counter, cond_mask, lengths)
        pred_fake_y = Dy(main_fake_y, main_mask, tr_actual, cond_mask, lengths)

        loss_Gx_adv = criterion_adv(pred_fake_x, ones_seq)
        loss_Gy_adv = criterion_adv(pred_fake_y, ones_seq)

        cycle_loss = masked_reconstruction_loss(rec_treat, tr_real, mask_treat, cfg.criterion)

        id_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
        id_treat = id_treat * mask_treat
        id_rec, _ = Gy(id_treat, mask_treat, tr_actual, cond_mask, lengths)
        id_rec = id_rec * mask_treat
        identity_loss = 0.5 * masked_reconstruction_loss(id_rec, tr_real, mask_treat, cfg.criterion)

        g_loss = loss_Gx_adv + loss_Gy_adv + cfg.lambda_cyc * cycle_loss + cfg.lambda_id * identity_loss
        g_loss.backward()
        optimizer_Gx.step()
        optimizer_Gy.step()

        for p in Dx.parameters():
            p.requires_grad_(True)
        for p in Dy.parameters():
            p.requires_grad_(True)

        totals["g_loss"] += g_loss.item()
        totals["d_loss"] += (loss_Dx + loss_Dy).item()
        totals["cycle_loss"] += cycle_loss.item()
        totals["identity_loss"] += identity_loss.item()
        totals["gx_adv"] += loss_Gx_adv.item()
        totals["gy_adv"] += loss_Gy_adv.item()
        totals["dx_loss"] += loss_Dx.item()
        totals["dy_loss"] += loss_Dy.item()

    batches = max(1, len(loader))
    return {k: v / batches for k, v in totals.items()}


def evaluate_epoch(
    Gx: nn.Module,
    Gy: nn.Module,
    Dx: nn.Module,
    Dy: nn.Module,
    loader: DataLoader,
    criterion_adv: nn.Module,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    Gx.eval()
    Gy.eval()
    Dx.eval()
    Dy.eval()

    totals = {k: 0.0 for k in [
        "g_loss",
        "d_loss",
        "cycle_loss",
        "identity_loss",
        "gx_adv",
        "gy_adv",
        "dx_loss",
        "dy_loss",
    ]}

    with torch.no_grad():
        for batch in loader:
            cl_real = batch["x_clin"].to(device)
            tr_real = batch["x_treat"].to(device)
            tr_actual = batch["actual_treatment"].to(device)
            mask_clin = batch["mask_clin"].to(device)
            mask_treat = batch["mask_treat"].to(device)
            mask_actual = batch["mask_actual"].to(device)

            step_mask = ((mask_clin.sum(dim=2, keepdim=True) + mask_treat.sum(dim=2, keepdim=True)) > 0).float()
            lengths = step_mask.squeeze(-1).sum(dim=1).clamp(min=1).long()

            cl_real = cl_real * mask_clin
            tr_real = tr_real * mask_treat
            tr_actual = tr_actual * mask_actual
            cond_mask = torch.ones_like(tr_actual)
            tr_counter = randomize_multiple_one_hot(tr_actual, mask_actual, extra_ones=1)

            fake_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
            fake_treat = fake_treat * mask_treat
            rec_treat, _ = Gy(fake_treat, mask_treat, tr_actual, cond_mask, lengths)
            rec_treat = rec_treat * mask_treat

            main_real = torch.cat((cl_real, tr_real), dim=2)
            main_mask = torch.cat((mask_clin, mask_treat), dim=2)
            main_fake_x = torch.cat((cl_real, fake_treat), dim=2)
            main_fake_y = torch.cat((cl_real, rec_treat), dim=2)

            pred_fake_x = Dx(main_fake_x, main_mask, tr_counter, cond_mask, lengths)
            pred_fake_y = Dy(main_fake_y, main_mask, tr_actual, cond_mask, lengths)
            pred_real_x = Dx(main_real, main_mask, tr_actual, cond_mask, lengths)
            pred_real_y = Dy(main_real, main_mask, tr_actual, cond_mask, lengths)

            ones_seq = torch.ones_like(pred_real_x)
            zeros_seq = torch.zeros_like(pred_real_x)

            loss_Gx_adv = criterion_adv(pred_fake_x, ones_seq)
            loss_Gy_adv = criterion_adv(pred_fake_y, ones_seq)
            cycle_loss = masked_reconstruction_loss(rec_treat, tr_real, mask_treat, cfg.criterion)
            id_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
            id_treat = id_treat * mask_treat
            id_rec, _ = Gy(id_treat, mask_treat, tr_actual, cond_mask, lengths)
            id_rec = id_rec * mask_treat
            identity_loss = 0.5 * masked_reconstruction_loss(id_rec, tr_real, mask_treat, cfg.criterion)

            loss_Dx = 0.5 * (
                criterion_adv(pred_real_x, ones_seq) + criterion_adv(pred_fake_x, zeros_seq)
            )
            loss_Dy = 0.5 * (
                criterion_adv(pred_real_y, ones_seq) + criterion_adv(pred_fake_y, zeros_seq)
            )

            totals["g_loss"] += (loss_Gx_adv + loss_Gy_adv + cfg.lambda_cyc * cycle_loss + cfg.lambda_id * identity_loss).item()
            totals["d_loss"] += (loss_Dx + loss_Dy).item()
            totals["cycle_loss"] += cycle_loss.item()
            totals["identity_loss"] += identity_loss.item()
            totals["gx_adv"] += loss_Gx_adv.item()
            totals["gy_adv"] += loss_Gy_adv.item()
            totals["dx_loss"] += loss_Dx.item()
            totals["dy_loss"] += loss_Dy.item()

    batches = max(1, len(loader))
    return {k: v / batches for k, v in totals.items()}


__all__ = [
    "Config",
    "load_config",
    "set_seed",
    "get_adversarial_loss",
    "get_criterion",
    "resolve_device",
    "SyntheticSequenceDataset",
    "create_dataloaders",
    "LSTMGenerator",
    "LSTMDiscriminator",
    "masked_reconstruction_loss",
    "randomize_multiple_one_hot",
    "train_epoch",
    "evaluate_epoch",
    "train",
]


def train(
    cfg: Config,
    *,
    synthetic_data: Optional[str] = None,
    patients: int = 64,
    timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    schema_path: Optional[str] = None,
    checkpoint_dir: str | Path = "data/models",
) -> Path:
    run_seed = seed if seed is not None else cfg.seed
    set_seed(run_seed)
    device = resolve_device(cfg.device)

    if synthetic_data:
        with open(synthetic_data, "r", encoding="utf-8") as fh:
            dataset_dict = json.load(fh)
    else:
        dataset_dict = generate_dataset(
            num_patients=patients,
            timesteps=timesteps or cfg.seq_len,
            seed=run_seed,
            schema_path=schema_path,
        )

    dataset = SyntheticSequenceDataset(
        dataset_dict,
        cfg.seq_len,
        spec_clinical=cfg.clinical_features,
        spec_treatment=cfg.treatment_features,
    )
    train_loader, val_loader, train_indices, val_indices = create_dataloaders(
        dataset, cfg.batch_size, run_seed
    )

    cond_dim = dataset.cond_dim
    clin_dim = dataset.clin_dim
    treat_dim = dataset.treat_dim

    Gx = LSTMGenerator(clin_dim, cond_dim, cfg.g_hidden, treat_dim, cfg.num_layers).to(device)
    Gy = LSTMGenerator(treat_dim, cond_dim, cfg.g_hidden, treat_dim, cfg.num_layers).to(device)
    Dx = LSTMDiscriminator(clin_dim + treat_dim, cond_dim, cfg.d_hidden, cfg.num_layers).to(device)
    Dy = LSTMDiscriminator(clin_dim + treat_dim, cond_dim, cfg.d_hidden, cfg.num_layers).to(device)

    optimizer_Gx = torch.optim.Adam(Gx.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    optimizer_Gy = torch.optim.Adam(Gy.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    optimizer_Dx = torch.optim.Adam(Dx.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    optimizer_Dy = torch.optim.Adam(Dy.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    criterion_adv = get_adversarial_loss(cfg.adv_loss)

    for epoch in range(cfg.epochs):
        train_metrics = train_epoch(
            Gx,
            Gy,
            Dx,
            Dy,
            train_loader,
            optimizer_Gx,
            optimizer_Gy,
            optimizer_Dx,
            optimizer_Dy,
            criterion_adv,
            cfg,
            device,
        )
        val_metrics = evaluate_epoch(
            Gx,
            Gy,
            Dx,
            Dy,
            val_loader,
            criterion_adv,
            cfg,
            device,
        )
        print(
            "Epoch {}/{} "
            "train[G={:.4f} D={:.4f} cycle={:.4f} id={:.4f} Gx_adv={:.4f} Gy_adv={:.4f} Dx={:.4f} Dy={:.4f}] "
            "val[G={:.4f} D={:.4f} cycle={:.4f} id={:.4f} Gx_adv={:.4f} Gy_adv={:.4f} Dx={:.4f} Dy={:.4f}]".format(
                epoch + 1,
                cfg.epochs,
                train_metrics["g_loss"],
                train_metrics["d_loss"],
                train_metrics["cycle_loss"],
                train_metrics["identity_loss"],
                train_metrics["gx_adv"],
                train_metrics["gy_adv"],
                train_metrics["dx_loss"],
                train_metrics["dy_loss"],
                val_metrics["g_loss"],
                val_metrics["d_loss"],
                val_metrics["cycle_loss"],
                val_metrics["identity_loss"],
                val_metrics["gx_adv"],
                val_metrics["gy_adv"],
                val_metrics["dx_loss"],
                val_metrics["dy_loss"],
            )
        )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"dtcygan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(
        {
            "config": cfg.__dict__,
            "metadata": {
                "clin_dim": clin_dim,
                "treat_dim": treat_dim,
                "cond_dim": cond_dim,
                "clinical_feature_spec": cfg.clinical_features,
                "treatment_feature_spec": cfg.treatment_features,
                "validation_patient_ids": [
                    dataset.patients[idx]["patient_id"] for idx in val_indices
                ],
            },
            "Gx": Gx.state_dict(),
            "Gy": Gy.state_dict(),
            "Dx": Dx.state_dict(),
            "Dy": Dy.state_dict(),
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path
