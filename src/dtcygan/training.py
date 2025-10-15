"""Training utilities for the temporal CycleGAN on synthetic datasets."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
    '''
    Dataclass carrying all hyperparameters required for CycleGAN training.

    args:
    - seq_len: sequence length per patient [int]
    - batch_size: samples per mini-batch [int]
    - epochs: number of training epochs [int]
    - lr_g: generator learning rate [float]
    - lr_d: discriminator learning rate [float]
    - beta1: first momentum coefficient [float]
    - beta2: second momentum coefficient [float]
    - opt_gen: optimizer identifier for generators [str]
    - opt_disc: optimizer identifier for discriminators [str]
    - lambda_cyc: cycle-consistency weight [float]
    - lambda_id: identity penalty weight [float]
    - g_hidden: generator hidden dimension [int]
    - d_hidden: discriminator hidden dimension [int]
    - num_layers: LSTM depth [int]
    - seed: random seed [int]
    - device: preferred torch device [str | None]
    - adv_loss: adversarial loss key [str]
    - criterion: reconstruction loss key [str]
    - clinical_features: clinical feature spec [Dict]
    - treatment_features: treatment feature spec [Dict]

    return:
    - Config: populated configuration object [Config]
    '''
    seq_len: int = 8
    batch_size: int = 32
    epochs: int = 200
    lr_g: float = 4.0e-5
    lr_d: float = 5.0e-5
    beta1: float = 0.5
    beta2: float = 0.999
    opt_gen: str = "adam"
    opt_disc: str = "adam"
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
    '''
    Load YAML configuration file and build a Config instance.

    args:
    - path: path to YAML file [str | Path]

    return:
    - cfg: parsed configuration [Config]
    '''
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    allowed = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return Config(**filtered)


def set_seed(seed: int) -> None:
    '''
    Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    args:
    - seed: random seed value [int]

    return:
    - None: function mutates global RNG state [None]
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adversarial_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Binary cross-entropy adversarial loss operating on logits.

    args:
    - output: discriminator predictions [torch.Tensor]
    - target: label tensor of ones or zeros [torch.Tensor]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    return F.binary_cross_entropy_with_logits(output, target)


def least_squares_gan_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Least-squares GAN criterion using mean-squared error.

    args:
    - output: discriminator predictions [torch.Tensor]
    - target: regression targets [torch.Tensor]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    return F.mse_loss(output, target)


def hinge_loss(output: torch.Tensor, is_real: bool) -> torch.Tensor:
    '''
    Compute hinge GAN loss given discriminator logits and target flag.

    args:
    - output: discriminator predictions [torch.Tensor]
    - is_real: whether samples are real [bool]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    if is_real:
        return torch.relu(1.0 - output).mean()
    return torch.relu(1.0 + output).mean()


def wasserstein_loss(output: torch.Tensor, is_real: bool) -> torch.Tensor:
    '''
    Compute Wasserstein loss treating logits as critic scores.

    args:
    - output: discriminator predictions [torch.Tensor]
    - is_real: whether samples are real [bool]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    return -output.mean() if is_real else output.mean()


def _resolve_is_real(target: torch.Tensor) -> bool:
    '''
    Infer real/fake flag by inspecting averaged target tensor.

    args:
    - target: tensor of labels [torch.Tensor]

    return:
    - is_real: True if targets indicate real samples [bool]
    '''
    return target.float().mean().item() > 0.5


adversarial_loss_map: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "bce": adversarial_loss,
    "binary_cross_entropy": adversarial_loss,
    "bcewithlogits": adversarial_loss,
    "lsgan": least_squares_gan_loss,
    "least_squares": least_squares_gan_loss,
    "hinge": lambda output, target: hinge_loss(output, _resolve_is_real(target)),
    "wasserstein": lambda output, target: wasserstein_loss(output, _resolve_is_real(target)),
}


def get_adversarial_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    '''
    Retrieve adversarial loss callable by string key.

    args:
    - name: loss identifier [str]

    return:
    - loss_fn: mapped loss function [Callable]
    '''
    key = (name or "bce").lower()
    if key not in adversarial_loss_map:
        raise ValueError(f"Unknown adversarial loss: {name}")
    return adversarial_loss_map[key]


def get_criterion(name: str) -> nn.Module:
    '''
    Return reconstruction loss module with no internal reduction.

    args:
    - name: criterion identifier [str]

    return:
    - loss_fn: torch loss module [nn.Module]
    '''
    name = (name or "l1").lower()
    if name in {"bce", "bcelogits", "binary_cross_entropy"}:
        return nn.BCEWithLogitsLoss(reduction="none")
    if name in {"mse", "mse_loss", "l2"}:
        return nn.MSELoss(reduction="none")
    if name in {"l1", "mae"}:
        return nn.L1Loss(reduction="none")
    if name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss(reduction="none")
    if name in {"cosine", "cosine_embedding"}:
        return nn.CosineEmbeddingLoss(reduction="none")
    raise ValueError(f"Unsupported criterion: {name}")


def get_optimizer(
    name: Optional[str],
    params: Iterable[torch.nn.Parameter],
    lr: float,
    betas: Optional[Tuple[float, float]] = None,
) -> torch.optim.Optimizer:
    '''
    Construct optimizer for provided parameters according to key.

    args:
    - name: optimizer identifier [str | None]
    - params: iterable of parameters to update [Iterable]
    - lr: learning rate [float]
    - betas: optional beta/momentum tuple [Tuple | None]

    return:
    - optimizer: initialized optimizer instance [torch.optim.Optimizer]
    '''
    key = (name or "adam").lower()
    if key == "adam":
        beta1, beta2 = betas if betas is not None else (0.9, 0.999)
        return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
    if key == "adamw":
        beta1, beta2 = betas if betas is not None else (0.9, 0.999)
        return torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2))
    if key in {"rmsprop", "rms"}:
        alpha = betas[0] if betas is not None else 0.99
        return torch.optim.RMSprop(params, lr=lr, alpha=alpha)
    if key == "sgd":
        momentum = betas[0] if betas is not None else 0.0
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if key == "radam":
        beta1, beta2 = betas if betas is not None else (0.9, 0.999)
        return torch.optim.RAdam(params, lr=lr, betas=(beta1, beta2))
    raise ValueError(f"Unsupported optimizer: {name}")


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    '''
    Choose execution device, testing CUDA availability if requested.

    args:
    - preferred: optional device string [str | None]

    return:
    - device: resolved torch device [torch.device]
    '''
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
    '''
    LSTM-based generator that maps sequences to treatment features.

    args:
    - input_dim: dimension of main sequence [int]
    - cond_input_dim: conditioning vector dimension [int]
    - hidden_dim: LSTM hidden size [int]
    - output_dim: generated feature dimension [int]
    - num_layers: number of LSTM layers [int]

    return:
    - LSTMGenerator: initialized generator module [LSTMGenerator]
    '''

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
    '''
    Mask-aware LSTM discriminator that scores complete sequences.

    args:
    - input_dim: dimensionality of concatenated main sequence [int]
    - cond_dim: dimension of conditioning features [int]
    - hidden_dim: LSTM hidden size [int]
    - num_layers: number of LSTM layers [int]

    return:
    - LSTMDiscriminator: initialized discriminator module [LSTMDiscriminator]
    '''

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        total_dim = (input_dim + cond_dim) * 2
        self.lstm = nn.LSTM(
            input_size=total_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

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
    '''
    Dataset wrapping synthetic patient timelines with masking metadata.

    args:
    - data: synthetic dataset dictionary [Dict]
    - seq_len: desired sequence length [int]
    - spec_clinical: clinical feature spec [Dict | None]
    - spec_treatment: treatment feature spec [Dict | None]

    return:
    - SyntheticSequenceDataset: prepared dataset instance [SyntheticSequenceDataset]
    '''
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
        '''
        Convert arbitrary values to comparable string representations.

        args:
        - value: input object to stringify [Any]

        return:
        - text: normalized string representation [str]
        '''
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, sort_keys=True)
            except TypeError:
                return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def _collect_values(self, key: str) -> Dict[str, set[Any]]:
        '''
        Gather observed values per column across all patients.

        args:
        - key: section of the patient record (clinical/treatment) [str]

        return:
        - value_map: mapping of column -> observed values [Dict]
        '''
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
        '''
        Build ordered column lists and categorical metadata.

        args:
        - key: data section name [str]
        - spec: user-provided feature specification [Dict]

        return:
        - columns: ordered list of columns [List]
        - categorical: categorical feature names [set]
        - categories: mapping to category lists [Dict]
        - ordinal_maps: ordinal encoding maps [Dict]
        '''
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
        '''
        Determine conditioning feature order for actual treatment.

        args:
        - None: operates on dataset state [None]

        return:
        - columns: sorted conditioning column names [List]
        '''
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
    @staticmethod
    def _pad(array: np.ndarray, seq_len: int) -> np.ndarray:
        '''
        Trim or zero-pad arrays to the configured sequence length.

        args:
        - array: time-major feature matrix [np.ndarray]
        - seq_len: target sequence length [int]

        return:
        - padded: length-adjusted array [np.ndarray]
        '''
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
        '''
        Encode patient records into numeric arrays with masks.

        args:
        - records: list of timestep dictionaries [Iterable]
        - columns: desired column order [List]
        - categorical: categorical feature names [set]
        - categories: known category values [Dict]
        - ordinal_maps: ordinal mapping dictionaries [Dict]

        return:
        - values: encoded feature matrix [np.ndarray]
        - mask: presence mask matrix [np.ndarray]
        '''
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
        '''
        Encode actual treatment records with simple imputation.

        args:
        - records: iterable of treatment entries [Iterable]

        return:
        - values: encoded conditioning matrix [np.ndarray]
        - mask: presence mask matrix [np.ndarray]
        '''
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
        '''
        Report number of patients available in dataset.

        args:
        - None: required by Dataset interface [None]

        return:
        - count: dataset length [int]
        '''
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        '''
        Fetch padded tensors and masks for a specific patient.

        args:
        - idx: patient index [int]

        return:
        - sample: dictionary of tensors for model input [Dict]
        '''
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
def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    '''
    Apply a masked reduction around elementwise reconstruction losses.

    args:
    - pred: model predictions [torch.Tensor]
    - target: reference values [torch.Tensor]
    - mask: per-element visibility mask [torch.Tensor]
    - loss_fn: reduction-free loss module [nn.Module]

    return:
    - loss: scalar masked reconstruction loss [torch.Tensor]
    '''
    if isinstance(loss_fn, nn.CosineEmbeddingLoss):
        mask_steps = (mask.sum(dim=2) > 0).to(pred.dtype)
        if mask_steps.numel() == 0:
            return pred.new_tensor(0.0)
        flat_pred = pred.reshape(-1, pred.size(-1))
        flat_target = target.reshape(-1, target.size(-1))
        cos_target = torch.ones(flat_pred.size(0), device=pred.device, dtype=pred.dtype)
        losses = loss_fn(flat_pred, flat_target, cos_target)
        losses = losses.view(pred.size(0), pred.size(1))
        losses = losses * mask_steps
        denom = mask_steps.sum().clamp_min(1.0)
        return losses.sum() / denom

    mask = mask.expand_as(pred).to(dtype=pred.dtype)
    losses = loss_fn(pred, target)
    if losses.shape != pred.shape:
        losses = losses.view_as(pred)
    losses = losses * mask
    denom = mask.sum().clamp_min(1.0)
    return losses.sum() / denom


def randomize_multiple_one_hot(
    cond: torch.Tensor,
    mask: torch.Tensor,
    extra_ones: int = 1,
) -> torch.Tensor:
    '''
    Inject random positive entries into masked one-hot tensors.

    args:
    - cond: conditioning tensor to modify [torch.Tensor]
    - mask: feature availability mask [torch.Tensor]
    - extra_ones: number of random insertions [int]

    return:
    - randomized: perturbed conditioning tensor [torch.Tensor]
    '''
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
    '''
    Split dataset into train and validation subsets with a fixed seed.

    args:
    - dataset: source dataset to partition [Dataset]
    - val_fraction: fraction assigned to validation [float]
    - seed: RNG seed for randomness [int]

    return:
    - train_subset: training dataset subset [Dataset]
    - val_subset: validation dataset subset [Dataset]
    '''
    val_size = max(1, int(len(dataset) * val_fraction))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def create_dataloaders(
    dataset: SyntheticSequenceDataset, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    '''
    Create loaders plus index bookkeeping for train/validation splits.

    args:
    - dataset: dataset instance to load from [SyntheticSequenceDataset]
    - batch_size: loader batch size [int]
    - seed: RNG seed for splitting [int]

    return:
    - train_loader: training DataLoader [DataLoader]
    - val_loader: validation DataLoader [DataLoader]
    - train_indices: patient indices for train split [List]
    - val_indices: patient indices for validation split [List]
    '''
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
    optimizer_G: torch.optim.Optimizer,
    optimizer_Dx: torch.optim.Optimizer,
    optimizer_Dy: torch.optim.Optimizer,
    criterion_adv: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    criterion_cycle: nn.Module,
    criterion_id: nn.Module,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    '''
    Execute one training epoch over the provided dataloader.

    args:
    - Gx: clinical-to-treatment generator [nn.Module]
    - Gy: treatment autoencoder generator [nn.Module]
    - Dx: discriminator scoring fake treatments [nn.Module]
    - Dy: discriminator scoring reconstructions [nn.Module]
    - loader: training dataloader [DataLoader]
    - optimizer_G: shared generator optimizer [torch.optim.Optimizer]
    - optimizer_Dx: optimizer for Dx [torch.optim.Optimizer]
    - optimizer_Dy: optimizer for Dy [torch.optim.Optimizer]
    - criterion_adv: adversarial loss function [Callable]
    - criterion_cycle: cycle-consistency loss [nn.Module]
    - criterion_id: identity loss [nn.Module]
    - cfg: training configuration [Config]
    - device: torch device in use [torch.device]

    return:
    - metrics: dictionary of averaged losses [Dict]
    '''
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

        # Update discriminators on detached generator outputs
        optimizer_Dx.zero_grad()
        optimizer_Dy.zero_grad()

        fake_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
        fake_treat = fake_treat * mask_treat
        rec_treat, _ = Gy(fake_treat, mask_treat, tr_actual, cond_mask, lengths)
        rec_treat = rec_treat * mask_treat

        main_real = torch.cat((cl_real, tr_real), dim=2)
        main_mask = torch.cat((mask_clin, mask_treat), dim=2)
        main_fake_x = torch.cat((cl_real, fake_treat.detach()), dim=2)
        main_fake_y = torch.cat((cl_real, rec_treat.detach()), dim=2)

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

        # Update generators with frozen discriminators
        optimizer_G.zero_grad()

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

        cycle_loss = masked_reconstruction_loss(rec_treat, tr_real, mask_treat, criterion_cycle)

        id_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
        id_treat = id_treat * mask_treat
        id_rec, _ = Gy(id_treat, mask_treat, tr_actual, cond_mask, lengths)
        id_rec = id_rec * mask_treat
        identity_loss = 0.5 * masked_reconstruction_loss(id_rec, tr_real, mask_treat, criterion_id)

        g_loss = loss_Gx_adv + loss_Gy_adv + cfg.lambda_cyc * cycle_loss + cfg.lambda_id * identity_loss
        g_loss.backward()
        optimizer_G.step()

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
    criterion_adv: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    criterion_cycle: nn.Module,
    criterion_id: nn.Module,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    '''
    Run evaluation pass without gradient updates for validation data.

    args:
    - Gx: clinical-to-treatment generator [nn.Module]
    - Gy: treatment autoencoder generator [nn.Module]
    - Dx: discriminator scoring fake treatments [nn.Module]
    - Dy: discriminator scoring reconstructions [nn.Module]
    - loader: validation dataloader [DataLoader]
    - criterion_adv: adversarial loss function [Callable]
    - criterion_cycle: cycle-consistency loss [nn.Module]
    - criterion_id: identity loss [nn.Module]
    - cfg: training configuration [Config]
    - device: torch device in use [torch.device]

    return:
    - metrics: dictionary of averaged losses [Dict]
    '''
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
            cycle_loss = masked_reconstruction_loss(rec_treat, tr_real, mask_treat, criterion_cycle)
            id_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
            id_treat = id_treat * mask_treat
            id_rec, _ = Gy(id_treat, mask_treat, tr_actual, cond_mask, lengths)
            id_rec = id_rec * mask_treat
            identity_loss = 0.5 * masked_reconstruction_loss(id_rec, tr_real, mask_treat, criterion_id)

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
    "get_optimizer",
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
    '''
    Top-level training routine handling data, models, and checkpoints.

    args:
    - cfg: parsed training configuration [Config]
    - synthetic_data: optional path to pre-generated JSON [str | None]
    - patients: number of patients to generate when needed [int]
    - timesteps: timesteps for synthetic generation [int | None]
    - seed: optional override for dataset RNG [int | None]
    - schema_path: path to schema YAML [str | None]
    - checkpoint_dir: directory for saving checkpoints [str | Path]

    return:
    - ckpt_path: path to written checkpoint file [Path]
    '''
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

    optimizer_G = get_optimizer(
        cfg.opt_gen,
        list(Gx.parameters()) + list(Gy.parameters()),
        cfg.lr_g,
        (cfg.beta1, cfg.beta2),
    )
    optimizer_Dx = get_optimizer(cfg.opt_disc, Dx.parameters(), cfg.lr_d, (cfg.beta1, cfg.beta2))
    optimizer_Dy = get_optimizer(cfg.opt_disc, Dy.parameters(), cfg.lr_d, (cfg.beta1, cfg.beta2))

    criterion_adv = get_adversarial_loss(cfg.adv_loss)
    criterion_cycle = get_criterion(cfg.criterion)
    criterion_id = get_criterion(cfg.criterion)

    for epoch in range(cfg.epochs):
        train_metrics = train_epoch(
            Gx,
            Gy,
            Dx,
            Dy,
            train_loader,
            optimizer_G,
            optimizer_Dx,
            optimizer_Dy,
            criterion_adv,
            criterion_cycle,
            criterion_id,
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
            criterion_cycle,
            criterion_id,
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
