"""Training utilities for the temporal CycleGAN on synthetic datasets."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
import json
from datetime import datetime

from ditcygan.synthetic import generate_dataset

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


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


class LSTMGenerator(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim + cond_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, cond], dim=-1)
        out, _ = self.lstm(inp)
        return self.proj(out)


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.score(out)


class SyntheticSequenceDataset(Dataset):
    def __init__(self, data: Dict[str, Any], seq_len: int):
        patients = data.get("patients", [])
        if not isinstance(patients, list) or not patients:
            raise ValueError("Synthetic dataset must contain a non-empty 'patients' list.")

        self.seq_len = seq_len
        self.patients = patients
        self.clin_columns, self.clin_categorical, self.clin_categories = self._analyse_columns("clinical")
        self.treat_columns, self.treat_categorical, self.treat_categories = self._analyse_columns("treatment")
        self.cond_columns = self._infer_condition_columns()
        self.clin_dim = len(self.clin_columns)
        self.treat_dim = len(self.treat_columns)
        self.cond_dim = len(self.cond_columns)

    def _analyse_columns(self, key: str) -> Tuple[List[str], set[str], Dict[str, List[str]]]:
        columns: set[str] = set()
        raw_values: Dict[str, set] = {}
        for patient in self.patients:
            records = patient.get(key, [])
            df = pd.DataFrame(records)
            df = df.drop(columns=["_id", "timestep"], errors="ignore")
            for col in df.columns:
                columns.add(col)
                raw_values.setdefault(col, set()).update(df[col].dropna().tolist())
        ordered = sorted(columns)
        categorical: set[str] = set()
        categories: Dict[str, List[str]] = {}
        for col in ordered:
            values = raw_values.get(col, set())
            if any(isinstance(v, str) for v in values):
                categorical.add(col)
                categories[col] = sorted({str(v) for v in values})
        return ordered, categorical, categories

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
            if col in categorical:
                cat = pd.Categorical(series.astype(str), categories=categories.get(col, []))
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
            patient.get("clinical", []), self.clin_columns, self.clin_categorical, self.clin_categories
        )
        treat_vals, treat_mask = self._encode_records(
            patient.get("treatment", []), self.treat_columns, self.treat_categorical, self.treat_categories
        )
        actual_vals, _ = self._encode_actual(patient.get("actual_treatment", []))

        clin_vals = self._pad(clin_vals, self.seq_len)
        treat_vals = self._pad(treat_vals, self.seq_len)
        actual_vals = self._pad(actual_vals, self.seq_len)
        clin_mask = self._pad(clin_mask, self.seq_len)
        treat_mask = self._pad(treat_mask, self.seq_len)

        return {
            "x_clin": torch.from_numpy(clin_vals),
            "x_treat": torch.from_numpy(treat_vals),
            "actual_treatment": torch.from_numpy(actual_vals),
            "mask_clin": torch.from_numpy(clin_mask),
            "mask_treat": torch.from_numpy(treat_mask),
        }


def masked_adversarial_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    mask = mask.expand_as(predictions)
    if loss_type == "lsgan":
        loss = ((predictions - targets) ** 2) * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return nn.functional.binary_cross_entropy_with_logits(
        predictions,
        targets,
        weight=mask,
        reduction="sum",
    ) / mask.sum().clamp_min(1.0)


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    mask = mask.expand_as(pred)
    if loss_type == "l1":
        loss = torch.abs(pred - target) * mask
    else:
        loss = ((pred - target) ** 2) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def random_condition(cond: torch.Tensor) -> torch.Tensor:
    batch, steps, num_classes = cond.shape
    indices = torch.randint(0, num_classes, (batch, steps), device=cond.device)
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> Tuple[Dataset, Dataset]:
    val_size = max(1, int(len(dataset) * val_fraction))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def create_dataloaders(dataset: SyntheticSequenceDataset, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_subset, val_subset = split_dataset(dataset, val_fraction=0.2, seed=seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


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
        x_clin = batch["x_clin"].to(device)
        x_treat = batch["x_treat"].to(device)
        cond = batch["actual_treatment"].to(device)
        mask_clin = batch["mask_clin"].to(device)
        mask_treat = batch["mask_treat"].to(device)

        step_mask = ((mask_clin.sum(dim=2, keepdim=True) + mask_treat.sum(dim=2, keepdim=True)) > 0).float()
        x_clin = x_clin * step_mask
        x_treat = x_treat * step_mask
        cond_random = random_condition(cond)

        ones = torch.ones_like(step_mask)
        zeros = torch.zeros_like(step_mask)

        optimizer_Dx.zero_grad()
        optimizer_Dy.zero_grad()

        fake_treat_detached = Gx(x_clin, cond_random).detach() * step_mask
        cycle_clin_detached = Gy(x_treat, cond).detach() * step_mask

        dx_real = Dx(torch.cat([x_clin, x_treat, cond], dim=-1))
        dx_fake = Dx(torch.cat([x_clin, fake_treat_detached, cond_random], dim=-1))
        dy_real = Dy(torch.cat([x_clin, x_treat, cond], dim=-1))
        dy_fake = Dy(torch.cat([cycle_clin_detached, x_treat, cond], dim=-1))

        loss_Dx = 0.5 * (
            masked_adversarial_loss(dx_real, ones, step_mask, cfg.adv_loss)
            + masked_adversarial_loss(dx_fake, zeros, step_mask, cfg.adv_loss)
        )
        loss_Dy = 0.5 * (
            masked_adversarial_loss(dy_real, ones, step_mask, cfg.adv_loss)
            + masked_adversarial_loss(dy_fake, zeros, step_mask, cfg.adv_loss)
        )
        (loss_Dx + loss_Dy).backward()
        optimizer_Dx.step()
        optimizer_Dy.step()

        optimizer_Gx.zero_grad()
        optimizer_Gy.zero_grad()

        fake_treat = Gx(x_clin, cond_random)
        cycle_clin = Gy(fake_treat, cond)
        cycle_treat = Gx(x_treat, cond)

        dx_pred = Dx(torch.cat([x_clin, fake_treat, cond_random], dim=-1))
        dy_pred = Dy(torch.cat([cycle_clin, x_treat, cond], dim=-1))

        loss_Gx_adv = masked_adversarial_loss(dx_pred, ones, step_mask, cfg.adv_loss)
        loss_Gy_adv = masked_adversarial_loss(dy_pred, ones, step_mask, cfg.adv_loss)

        cycle_loss = masked_reconstruction_loss(cycle_clin, x_clin, step_mask, cfg.criterion)
        cycle_loss += masked_reconstruction_loss(cycle_treat, x_treat, step_mask, cfg.criterion)

        id_treat = masked_reconstruction_loss(Gx(x_clin, cond), x_treat, step_mask, cfg.criterion)
        id_clin = masked_reconstruction_loss(Gy(x_treat, cond), x_clin, step_mask, cfg.criterion)
        identity_loss = 0.5 * (id_treat + id_clin)

        g_loss = loss_Gx_adv + loss_Gy_adv + cfg.lambda_cyc * cycle_loss + cfg.lambda_id * identity_loss
        g_loss.backward()
        optimizer_Gx.step()
        optimizer_Gy.step()

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
            x_clin = batch["x_clin"].to(device)
            x_treat = batch["x_treat"].to(device)
            cond = batch["actual_treatment"].to(device)
            mask_clin = batch["mask_clin"].to(device)
            mask_treat = batch["mask_treat"].to(device)

            step_mask = ((mask_clin.sum(dim=2, keepdim=True) + mask_treat.sum(dim=2, keepdim=True)) > 0).float()
            x_clin = x_clin * step_mask
            x_treat = x_treat * step_mask
            cond_random = random_condition(cond)

            ones = torch.ones_like(step_mask)
            zeros = torch.zeros_like(step_mask)

            fake_treat = Gx(x_clin, cond_random)
            cycle_clin = Gy(fake_treat, cond)
            cycle_treat = Gx(x_treat, cond)

            dx_pred = Dx(torch.cat([x_clin, fake_treat, cond_random], dim=-1))
            dy_pred = Dy(torch.cat([cycle_clin, x_treat, cond], dim=-1))
            dx_real = Dx(torch.cat([x_clin, x_treat, cond], dim=-1))
            dy_real = Dy(torch.cat([x_clin, x_treat, cond], dim=-1))

            loss_Gx_adv = masked_adversarial_loss(dx_pred, ones, step_mask, cfg.adv_loss)
            loss_Gy_adv = masked_adversarial_loss(dy_pred, ones, step_mask, cfg.adv_loss)
            cycle_loss = masked_reconstruction_loss(cycle_clin, x_clin, step_mask, cfg.criterion)
            cycle_loss += masked_reconstruction_loss(cycle_treat, x_treat, step_mask, cfg.criterion)
            id_treat = masked_reconstruction_loss(Gx(x_clin, cond), x_treat, step_mask, cfg.criterion)
            id_clin = masked_reconstruction_loss(Gy(x_treat, cond), x_clin, step_mask, cfg.criterion)
            identity_loss = 0.5 * (id_treat + id_clin)

            loss_Dx = 0.5 * (
                masked_adversarial_loss(dx_real, ones, step_mask, cfg.adv_loss)
                + masked_adversarial_loss(dx_pred, zeros, step_mask, cfg.adv_loss)
            )
            loss_Dy = 0.5 * (
                masked_adversarial_loss(dy_real, ones, step_mask, cfg.adv_loss)
                + masked_adversarial_loss(dy_pred, zeros, step_mask, cfg.adv_loss)
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
    "SyntheticSequenceDataset",
    "create_dataloaders",
    "LSTMGenerator",
    "LSTMDiscriminator",
    "masked_adversarial_loss",
    "masked_reconstruction_loss",
    "random_condition",
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
    checkpoint_dir: str | Path = "checkpoints",
) -> Path:
    run_seed = seed if seed is not None else cfg.seed
    set_seed(run_seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

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

    dataset = SyntheticSequenceDataset(dataset_dict, cfg.seq_len)
    train_loader, val_loader = create_dataloaders(dataset, cfg.batch_size, run_seed)

    cond_dim = dataset.cond_dim
    clin_dim = dataset.clin_dim
    treat_dim = dataset.treat_dim

    Gx = LSTMGenerator(clin_dim, cond_dim, cfg.g_hidden, treat_dim, cfg.num_layers).to(device)
    Gy = LSTMGenerator(treat_dim, cond_dim, cfg.g_hidden, clin_dim, cfg.num_layers).to(device)
    Dx = LSTMDiscriminator(clin_dim + treat_dim + cond_dim, cfg.d_hidden, cfg.num_layers).to(device)
    Dy = LSTMDiscriminator(clin_dim + treat_dim + cond_dim, cfg.d_hidden, cfg.num_layers).to(device)

    optimizer_Gx = torch.optim.Adam(Gx.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    optimizer_Gy = torch.optim.Adam(Gy.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    optimizer_Dx = torch.optim.Adam(Dx.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    optimizer_Dy = torch.optim.Adam(Dy.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

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
            cfg,
            device,
        )
        val_metrics = evaluate_epoch(Gx, Gy, Dx, Dy, val_loader, cfg, device)
        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs}: "
            f"train_G={train_metrics['g_loss']:.4f} train_D={train_metrics['d_loss']:.4f} "
            f"val_G={val_metrics['g_loss']:.4f} val_D={val_metrics['d_loss']:.4f}"
        )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"ditcygan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(
        {
            "config": cfg.__dict__,
            "metadata": {
                "clin_dim": clin_dim,
                "treat_dim": treat_dim,
                "cond_dim": cond_dim,
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
