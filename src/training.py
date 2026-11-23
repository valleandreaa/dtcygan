"""DTCyGAN training used by the paper reproduction workflows."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

from src.causal_weights import (
    CausalWeightSpec,
    load_training_weights,
    validate_longitudinal_training_weights,
)
from src.surgery_rt_training_data import (
    SarcomaSequenceDataset,
    build_surgery_rt_preop_two_arm_training_data,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@dataclass
class Config:
    seq_len: int = 8
    batch_size: int = 32
    epochs: int = 200
    generator_lr: float = 4.0e-5
    discriminator_lr: float = 5.0e-5
    beta1: float = 0.5
    beta2: float = 0.999
    opt_gen: str = "adam"
    opt_disc: str = "adam"
    cycle_weight: float = 20.0
    identity_weight: float = 30.0
    generator_hidden: int = 256
    discriminator_hidden: int = 128
    layers: int = 3
    seed: int = 42
    device: Optional[str] = None
    adv_loss: str = "lsgan"
    criterion: str = "l1"
    clinical_features: Dict[str, Any] = field(default_factory=dict)
    treatment_features: Dict[str, Any] = field(default_factory=dict)
    causal_weighting: Dict[str, Any] = field(default_factory=dict)
    data_contract: str = "synthetic"
    feature_groups: Dict[str, Any] = field(default_factory=dict)
    cohort_contract: Dict[str, Any] = field(default_factory=dict)
    audit_only: List[str] = field(default_factory=list)
    counterfactual_mode: str = "temporal_step_flip"
    lambda_outcome: float = 1.0
    factual_weight: float = 0.0
    condition_weight: float = 0.0
    lambda_regimen_delivery: float = 0.0
    lambda_generated_outcome: float = 0.0
    lambda_rt_feasibility: float = 0.0
    conditioning_feature: Optional[str] = None
    regimen_delivery_features: Dict[str, str] = field(default_factory=dict)
    regimen_delivery_feature_indices: Dict[int, int] = field(default_factory=dict)
    outcome_positive_weight: Optional[float] = 1.0
    outcome_positive_weight_cap: float = 20.0
    rt_feasibility: Dict[str, Any] = field(default_factory=dict)
    rt_feasibility_activation_indices: List[int] = field(default_factory=list)
    rt_feasibility_parameter_bounds: Dict[int, List[float]] = field(
        default_factory=dict
    )
    conditioning_contract: str = "time_local"
    broadcast_actual_treatment: bool = True
    require_explicit_splits: bool = False
    split_seed: Optional[int] = None
    validation_fraction: float = 0.20
    test_fraction: Optional[float] = None
    outcome_hidden: int = 128
    architecture: str = "lstm"
    include_missingness_indicators: bool = True
    missingness_indicator_encoding: str = "raw"
    treatment_prediction: bool = True
    temporal_alignment: Dict[str, Any] = field(default_factory=dict)
    recursive_state_generation: bool = False
    lambda_state: float = 0.0
    state_teacher_forcing_ratio: float = 0.0


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return Config(**yaml.safe_load(handle))


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


def weighted_mean(
    losses: torch.Tensor,
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce a tensor through patient-level weighted means."""

    if losses.ndim == 0:
        if patient_weights is not None:
            raise ValueError("Weighted reduction requires an unreduced loss tensor")
        return losses
    per_patient = losses.reshape(losses.shape[0], -1).mean(dim=1)
    if patient_weights is None:
        return per_patient.mean()
    weights = patient_weights.to(device=losses.device, dtype=losses.dtype)
    if weights.ndim == 2:
        if losses.ndim < 2 or weights.shape != losses.shape[:2]:
            raise ValueError("Patient-time weights must match the first two loss dimensions")
        per_step = losses.reshape(losses.shape[0], losses.shape[1], -1).mean(dim=2)
        return weighted_time_mean(per_step, weights, weights > 0)
    weights = weights.reshape(-1)
    if len(weights) != len(per_patient):
        raise ValueError(
            "Patient weights must have one value per batch row; "
            f"observed {len(weights)} weights for {len(per_patient)} rows"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("Patient weights must all be finite")
    if torch.any(weights < 0):
        raise ValueError("Patient weights cannot be negative")
    return (per_patient * weights).sum() / weights.sum().clamp_min(1e-8)


def weighted_time_mean(
    losses: torch.Tensor,
    weights: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    """Average weighted patient contributions within time, then across time."""

    if losses.ndim != 2 or weights.shape != losses.shape or eligible.shape != losses.shape:
        raise ValueError("losses, weights, and eligible must have shape [batch, time]")
    weights = weights.to(device=losses.device, dtype=losses.dtype)
    active = eligible.to(device=losses.device, dtype=losses.dtype) * weights
    if not torch.isfinite(active).all() or torch.any(active < 0):
        raise ValueError("Patient-time weights must be finite and non-negative")
    denominator = active.sum(dim=0)
    valid_times = denominator > 0
    if not torch.any(valid_times):
        return losses.sum() * 0.0
    time_losses = (losses * active).sum(dim=0)[valid_times] / denominator[valid_times]
    return time_losses.mean()


def adversarial_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Binary cross-entropy adversarial loss operating on logits.

    args:
    - output: discriminator predictions [torch.Tensor]
    - target: label tensor of ones or zeros [torch.Tensor]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    losses = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    return weighted_mean(losses, patient_weights)


def least_squares_gan_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Least-squares GAN criterion using mean-squared error.

    args:
    - output: discriminator predictions [torch.Tensor]
    - target: regression targets [torch.Tensor]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    losses = F.mse_loss(output, target, reduction="none")
    return weighted_mean(losses, patient_weights)


def hinge_loss(
    output: torch.Tensor,
    is_real: bool,
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Compute hinge GAN loss given discriminator logits and target flag.

    args:
    - output: discriminator predictions [torch.Tensor]
    - is_real: whether samples are real [bool]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    if is_real:
        return weighted_mean(torch.relu(1.0 - output), patient_weights)
    return weighted_mean(torch.relu(1.0 + output), patient_weights)


def wasserstein_loss(
    output: torch.Tensor,
    is_real: bool,
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Compute Wasserstein loss treating logits as critic scores.

    args:
    - output: discriminator predictions [torch.Tensor]
    - is_real: whether samples are real [bool]

    return:
    - loss: scalar loss value [torch.Tensor]
    '''
    losses = -output if is_real else output
    return weighted_mean(losses, patient_weights)


def _resolve_is_real(target: torch.Tensor) -> bool:
    '''
    Infer real/fake flag by inspecting averaged target tensor.

    args:
    - target: tensor of labels [torch.Tensor]

    return:
    - is_real: True if targets indicate real samples [bool]
    '''
    return target.float().mean().item() > 0.5


AdversarialLoss = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    torch.Tensor,
]


adversarial_loss_map: Dict[str, AdversarialLoss] = {
    "bce": adversarial_loss,
    "binary_cross_entropy": adversarial_loss,
    "bcewithlogits": adversarial_loss,
    "lsgan": least_squares_gan_loss,
    "least_squares": least_squares_gan_loss,
    "hinge": lambda output, target, weights=None: hinge_loss(
        output, _resolve_is_real(target), weights
    ),
    "wasserstein": lambda output, target, weights=None: wasserstein_loss(
        output, _resolve_is_real(target), weights
    ),
}


def get_adversarial_loss(name: str) -> AdversarialLoss:
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

def _encode_missingness_indicator(
    mask: torch.Tensor,
    encoding: str,
) -> torch.Tensor:
    """Encode observation masks without changing checkpoint tensor shapes."""

    if encoding == "raw":
        return mask
    if encoding == "centered":
        return mask - 1.0
    raise ValueError(
        "missingness_indicator_encoding must be 'raw' or 'centered'"
    )


class LSTMGenerator(nn.Module):
    '''
    LSTM-based generator that maps sequences to treatment features.

    args:
    - input_dim: dimension of main sequence [int]
    - cond_input_dim: conditioning vector dimension [int]
    - hidden_dim: LSTM hidden size [int]
    - output_dim: generated feature dimension [int]
    - layers: number of LSTM layers [int]

    return:
    - LSTMGenerator: initialized generator module [LSTMGenerator]
    '''

    def __init__(self, input_dim: int, cond_input_dim: int, hidden_dim: int, output_dim: int, layers: int, *, include_missingness_indicators: bool = True, missingness_indicator_encoding: str = "raw"):
        super().__init__()
        self.include_missingness_indicators = include_missingness_indicators
        self.missingness_indicator_encoding = missingness_indicator_encoding
        combined_dim = (input_dim + cond_input_dim) * (2 if include_missingness_indicators else 1)
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
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
        x_aug = torch.cat(
            [x, _encode_missingness_indicator(m_x, self.missingness_indicator_encoding)],
            dim=-1,
        ) if self.include_missingness_indicators else x
        cond_aug = torch.cat(
            [cond, _encode_missingness_indicator(m_cond, self.missingness_indicator_encoding)],
            dim=-1,
        ) if self.include_missingness_indicators else cond
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


class RecursiveStateGenerator(LSTMGenerator):
    """Causal generator that rolls generated clinical states through a forecast.

    The ordinary treatment/output head is retained for checkpoint compatibility.
    A second head predicts the temporal clinical state at the next timestamp.
    Observed clinical inputs are used only before ``forecast_start``; every later
    temporal state is replaced by the preceding prediction.
    """

    def __init__(
        self,
        input_dim: int,
        cond_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        state_feature_indices: Sequence[int],
        forecast_start: int,
        include_missingness_indicators: bool = True,
        missingness_indicator_encoding: str = "raw",
    ) -> None:
        super().__init__(
            input_dim,
            cond_input_dim,
            hidden_dim,
            output_dim,
            layers,
            include_missingness_indicators=include_missingness_indicators,
            missingness_indicator_encoding=missingness_indicator_encoding,
        )
        indices = tuple(int(index) for index in state_feature_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("state_feature_indices must be unique and non-empty")
        if min(indices) < 0 or max(indices) >= input_dim:
            raise ValueError("state_feature_indices fall outside the clinical input")
        if forecast_start < 0:
            raise ValueError("forecast_start cannot be negative")
        self.state_feature_indices = indices
        self.forecast_start = int(forecast_start)
        self.initial_state_head = nn.Sequential(
            nn.Linear(input_dim + cond_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(indices)),
        )
        self.state_head = nn.Linear(hidden_dim + output_dim, len(indices))
        self.last_state_predictions: Optional[torch.Tensor] = None
        self.last_generated_states: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        state_targets: Optional[torch.Tensor] = None,
        state_target_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 3 or m_x.shape != x.shape:
            raise ValueError("x and m_x must have shape [batch, time, clinical]")
        if cond.ndim != 3 or m_cond.shape != cond.shape or cond.shape[:2] != x.shape[:2]:
            raise ValueError("condition tensors must align with x in batch and time")
        if self.forecast_start >= x.size(1):
            raise ValueError("forecast_start must lie inside the supplied sequence")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must lie in [0, 1]")
        state_dim = len(self.state_feature_indices)
        expected_state_shape = (x.size(0), x.size(1), state_dim)
        if state_targets is not None and tuple(state_targets.shape) != expected_state_shape:
            raise ValueError(
                f"state_targets must have shape {expected_state_shape}; "
                f"got {tuple(state_targets.shape)}"
            )
        if state_target_mask is not None and (
            state_targets is None or state_target_mask.shape != state_targets.shape
        ):
            raise ValueError("state_target_mask must match state_targets")

        outputs: List[torch.Tensor] = []
        state_predictions = x.new_zeros(expected_state_shape)
        generated_states = x.new_zeros(expected_state_shape)
        predicted_state: Optional[torch.Tensor] = None
        recurrent_hidden = hidden
        state_indices = list(self.state_feature_indices)

        for timestamp in range(x.size(1)):
            x_step = x[:, timestamp : timestamp + 1].clone()
            mask_step = m_x[:, timestamp : timestamp + 1].clone()
            if timestamp < self.forecast_start:
                generated_states[:, timestamp] = x_step[:, 0, state_indices]
            else:
                if timestamp == 0 and predicted_state is None:
                    initial_x = x_step[:, 0].clone()
                    initial_x[:, state_indices] = 0.0
                    predicted_state = self.initial_state_head(
                        torch.cat((initial_x, cond[:, 0]), dim=-1)
                    )
                if predicted_state is None:
                    raise RuntimeError("Recursive forecast lacks a preceding state prediction")
                state_predictions[:, timestamp] = predicted_state
                state_input = predicted_state
                if state_targets is not None and teacher_forcing_ratio > 0:
                    target = state_targets[:, timestamp]
                    target_mask = (
                        state_target_mask[:, timestamp] > 0
                        if state_target_mask is not None
                        else torch.ones_like(target, dtype=torch.bool)
                    )
                    if teacher_forcing_ratio >= 1.0:
                        use_teacher = torch.ones(
                            (x.size(0), 1), device=x.device, dtype=torch.bool
                        )
                    else:
                        use_teacher = (
                            torch.rand((x.size(0), 1), device=x.device)
                            < teacher_forcing_ratio
                        )
                    state_input = torch.where(
                        use_teacher & target_mask,
                        target,
                        predicted_state,
                    )
                generated_states[:, timestamp] = state_input
                x_step[:, 0, state_indices] = state_input
                mask_step[:, 0, state_indices] = 1.0

            x_aug = (
                torch.cat(
                    [
                        x_step,
                        _encode_missingness_indicator(
                            mask_step, self.missingness_indicator_encoding
                        ),
                    ],
                    dim=-1,
                )
                if self.include_missingness_indicators
                else x_step
            )
            cond_step = cond[:, timestamp : timestamp + 1]
            cond_mask_step = m_cond[:, timestamp : timestamp + 1]
            cond_aug = (
                torch.cat(
                    [
                        cond_step,
                        _encode_missingness_indicator(
                            cond_mask_step, self.missingness_indicator_encoding
                        ),
                    ],
                    dim=-1,
                )
                if self.include_missingness_indicators
                else cond_step
            )
            recurrent, recurrent_hidden = self.lstm(
                torch.cat([x_aug, cond_aug], dim=-1), recurrent_hidden
            )
            generated_output = self.fc(recurrent)
            outputs.append(generated_output)
            predicted_state = self.state_head(
                torch.cat((recurrent[:, 0], generated_output[:, 0]), dim=-1)
            )

        if recurrent_hidden is None:
            raise RuntimeError("Recursive generator did not produce a hidden state")
        self.last_state_predictions = state_predictions
        self.last_generated_states = generated_states
        return torch.cat(outputs, dim=1), recurrent_hidden


class LSTMDiscriminator(nn.Module):
    '''
    Mask-aware LSTM discriminator that scores complete sequences.

    args:
    - input_dim: dimensionality of concatenated main sequence [int]
    - cond_dim: dimension of conditioning features [int]
    - hidden_dim: LSTM hidden size [int]
    - layers: number of LSTM layers [int]

    return:
    - LSTMDiscriminator: initialized discriminator module [LSTMDiscriminator]
    '''

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int, layers: int, *, include_missingness_indicators: bool = True, missingness_indicator_encoding: str = "raw"):
        super().__init__()
        self.include_missingness_indicators = include_missingness_indicators
        self.missingness_indicator_encoding = missingness_indicator_encoding
        total_dim = (input_dim + cond_dim) * (2 if include_missingness_indicators else 1)
        self.lstm = nn.LSTM(
            input_size=total_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        time_resolved: bool = False,
    ) -> torch.Tensor:
        x_aug = torch.cat(
            [x, _encode_missingness_indicator(m_x, self.missingness_indicator_encoding)],
            dim=-1,
        ) if self.include_missingness_indicators else x
        cond_aug = torch.cat(
            [cond, _encode_missingness_indicator(m_cond, self.missingness_indicator_encoding)],
            dim=-1,
        ) if self.include_missingness_indicators else cond
        combined = torch.cat([x_aug, cond_aug], dim=-1)

        if lengths is not None:
            packed = pack_padded_sequence(
                combined, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, _) = self.lstm(packed)
            recurrent, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=combined.size(1)
            )
        else:
            recurrent, (h_n, _) = self.lstm(combined)

        if time_resolved:
            return self.classifier(recurrent).squeeze(-1)

        last_hidden = h_n[-1]
        score = self.classifier(last_hidden).squeeze(-1)
        return score


class LSTMOutcomePredictor(nn.Module):
    """Mask-aware longitudinal predictor for interval-level binary outcomes."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        include_missingness_indicators: bool = True,
        missingness_indicator_encoding: str = "raw",
    ) -> None:
        super().__init__()
        self.include_missingness_indicators = include_missingness_indicators
        self.missingness_indicator_encoding = missingness_indicator_encoding
        total_dim = (input_dim + cond_dim) * (2 if include_missingness_indicators else 1)
        self.lstm = nn.LSTM(
            input_size=total_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
        )
        if self.lstm.bidirectional:
            raise ValueError(
                "The local-recurrence network must be forward-causal"
            )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_aug = torch.cat(
            [x, _encode_missingness_indicator(m_x, self.missingness_indicator_encoding)],
            dim=-1,
        ) if self.include_missingness_indicators else x
        cond_aug = torch.cat(
            [cond, _encode_missingness_indicator(m_cond, self.missingness_indicator_encoding)],
            dim=-1,
        ) if self.include_missingness_indicators else cond
        combined = torch.cat([x_aug, cond_aug], dim=-1)
        if lengths is not None:
            packed = pack_padded_sequence(
                combined, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            recurrent, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=combined.size(1)
            )
        else:
            recurrent, _ = self.lstm(combined)
        return self.classifier(recurrent)


def build_generator(
    cfg: Config,
    input_dim: int,
    cond_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Module:
    """Build the configured generator while keeping checkpoint loading uniform."""

    kwargs = {
        "include_missingness_indicators": cfg.include_missingness_indicators,
        "missingness_indicator_encoding": cfg.missingness_indicator_encoding,
    }
    if cfg.architecture == "lstm":
        return LSTMGenerator(
            input_dim, cond_dim, hidden_dim, output_dim, cfg.layers, **kwargs
        )
    if cfg.architecture == "temporal_mlp":
        from src.architectures import TemporalMLPGenerator

        return TemporalMLPGenerator(
            input_dim, cond_dim, hidden_dim, output_dim, cfg.layers, **kwargs
        )
    raise ValueError(f"Unsupported architecture: {cfg.architecture!r}")


def build_clinical_generator(
    cfg: Config,
    dataset: Any,
) -> nn.Module:
    """Build Gx, including the opt-in recursive clinical-state head."""

    if not cfg.recursive_state_generation:
        return build_generator(
            cfg,
            dataset.clin_dim,
            dataset.cond_dim,
            cfg.generator_hidden,
            dataset.treat_dim,
        )
    if cfg.architecture != "lstm":
        raise ValueError("Recursive state generation currently requires architecture='lstm'")
    supported_contracts = {"synthetic", "surgery_rt_preop_two_arm"}
    if cfg.data_contract not in supported_contracts:
        raise ValueError(
            "Recursive state generation currently supports synthetic and "
            "surgery_rt_preop_two_arm data"
        )
    if int(getattr(dataset, "state_dim", 0)) <= 0:
        raise ValueError("Recursive state generation requires separate state targets")
    forecast_start = getattr(dataset, "forecast_start", None)
    if forecast_start is None:
        raise ValueError("Recursive state generation requires a forecast start")
    if cfg.data_contract == "synthetic":
        configured_start = cfg.temporal_alignment.get("evaluation_start")
        if configured_start is None:
            raise ValueError(
                "Synthetic recursive generation requires forecast alignment"
            )
        if int(forecast_start) != int(configured_start):
            raise ValueError("Dataset and config forecast starts differ")
    elif int(forecast_start) != 0:
        raise ValueError("Sarcoma baseline-only rollout must generate from timestamp 0")
    return RecursiveStateGenerator(
        dataset.clin_dim,
        dataset.cond_dim,
        cfg.generator_hidden,
        dataset.treat_dim,
        cfg.layers,
        state_feature_indices=dataset.state_feature_indices,
        forecast_start=int(forecast_start),
        include_missingness_indicators=cfg.include_missingness_indicators,
        missingness_indicator_encoding=cfg.missingness_indicator_encoding,
    )


def build_discriminator(
    cfg: Config,
    input_dim: int,
    cond_dim: int,
    hidden_dim: int,
) -> nn.Module:
    kwargs = {
        "include_missingness_indicators": cfg.include_missingness_indicators,
        "missingness_indicator_encoding": cfg.missingness_indicator_encoding,
    }
    if cfg.architecture == "lstm":
        return LSTMDiscriminator(
            input_dim, cond_dim, hidden_dim, cfg.layers, **kwargs
        )
    if cfg.architecture == "temporal_mlp":
        from src.architectures import TemporalMLPDiscriminator

        return TemporalMLPDiscriminator(
            input_dim, cond_dim, hidden_dim, cfg.layers, **kwargs
        )
    raise ValueError(f"Unsupported architecture: {cfg.architecture!r}")


def build_outcome_predictor(
    cfg: Config,
    input_dim: int,
    cond_dim: int,
    output_dim: int,
) -> nn.Module:
    kwargs = {
        "include_missingness_indicators": cfg.include_missingness_indicators,
        "missingness_indicator_encoding": cfg.missingness_indicator_encoding,
    }
    if cfg.architecture == "lstm":
        return LSTMOutcomePredictor(
            input_dim,
            cond_dim,
            cfg.outcome_hidden,
            output_dim,
            cfg.layers,
            **kwargs,
        )
    if cfg.architecture == "temporal_mlp":
        from src.architectures import TemporalMLPOutcomePredictor

        return TemporalMLPOutcomePredictor(
            input_dim,
            cond_dim,
            cfg.outcome_hidden,
            output_dim,
            cfg.layers,
            **kwargs,
        )
    raise ValueError(f"Unsupported architecture: {cfg.architecture!r}")


class SyntheticSequenceDataset(Dataset):
    '''
    Dataset wrapping synthetic patient timelines with masking metadata.

    args:
    - data: synthetic dataset dictionary [Dict]
    - seq_len: desired sequence length [int]
    - spec_clinical: clinical feature spec [Dict | None]
    - spec_treatment: treatment feature spec [Dict | None]
    - patient_weights: optional patient-ID to causal-weight mapping [Mapping | None]

    return:
    - SyntheticSequenceDataset: prepared dataset instance [SyntheticSequenceDataset]
    '''
    def __init__(
        self,
        data: Dict[str, Any],
        seq_len: int,
        spec_clinical: Optional[Dict[str, Any]] = None,
        spec_treatment: Optional[Dict[str, Any]] = None,
        patient_weights: Optional[Mapping[str, float]] = None,
        broadcast_actual_treatment: bool = True,
    ):
        patients = data.get("patients", [])
        if not isinstance(patients, list) or not patients:
            raise ValueError("Synthetic dataset must contain a non-empty 'patients' list.")

        self.seq_len = seq_len
        self.broadcast_actual_treatment = bool(broadcast_actual_treatment)
        self.patients = patients
        self.patient_ids = [str(patient.get("patient_id")) for patient in patients]
        if any(patient.get("patient_id") is None for patient in patients):
            raise ValueError("Every training patient must have a patient_id")
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("Training patient identifiers must be unique")
        if patient_weights is None:
            self.patient_weights = np.ones(len(self.patient_ids), dtype=np.float32)
            self.causal_weighting_enabled = False
        else:
            normalized_mapping = {str(key): float(value) for key, value in patient_weights.items()}
            expected = set(self.patient_ids)
            observed = set(normalized_mapping)
            missing = sorted(expected - observed)[:5]
            unexpected = sorted(observed - expected)[:5]
            if missing or unexpected:
                raise ValueError(
                    "Causal weights do not match training patients; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            values = np.asarray(
                [normalized_mapping[patient_id] for patient_id in self.patient_ids],
                dtype=np.float32,
            )
            if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
                raise ValueError("Causal weights must be finite, non-negative, and have positive sum")
            self.patient_weights = values
            self.causal_weighting_enabled = True
        self.spec_clinical = spec_clinical or {}
        self.spec_treatment = spec_treatment or {}
        self.clin_columns, self.clin_categorical, self.clin_categories, self.clin_ord_maps = self._configure_columns(
            "clinical", self.spec_clinical
        )
        self.treat_columns, self.treat_categorical, self.treat_categories, self.treat_ord_maps = self._configure_columns(
            "treatment", self.spec_treatment
        )
        metadata = data.get("metadata", {}) or {}
        self.state_columns = [
            str(name) for name in metadata.get("state_target_features", [])
        ]
        unknown_state_columns = sorted(set(self.state_columns) - set(self.clin_columns))
        if unknown_state_columns:
            raise ValueError(
                "State-target features are absent from the clinical schema: "
                f"{unknown_state_columns}"
            )
        self.state_feature_indices = [
            self.clin_columns.index(name) for name in self.state_columns
        ]
        self.state_dim = len(self.state_columns)
        alignment = metadata.get("forecast_alignment") or {}
        self.forecast_start = (
            int(alignment["evaluation_start"])
            if alignment.get("evaluation_start") is not None
            else None
        )
        self.cond_columns = self._infer_condition_columns()
        self.clin_dim = len(self.clin_columns)
        self.treat_dim = len(self.treat_columns)
        self.cond_dim = len(self.cond_columns)
        self.outcome_dim = 0
        self.outcome_columns: List[str] = []
        self.spec_outcome: Dict[str, Any] = {}

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
        state_vals: Optional[np.ndarray] = None
        state_mask: Optional[np.ndarray] = None
        if self.state_columns:
            state_vals, state_mask = self._encode_records(
                patient.get("state_target", []),
                self.state_columns,
                set(),
                {},
                {},
            )
        raw_len = actual_vals.shape[0]
        if raw_len and self.broadcast_actual_treatment:
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
        if state_vals is not None and state_mask is not None:
            state_vals = self._pad(state_vals, self.seq_len)
            state_mask = self._pad(state_mask, self.seq_len)

        sample = {
            "x_clin": torch.from_numpy(clin_vals),
            "x_treat": torch.from_numpy(treat_vals),
            "actual_treatment": torch.from_numpy(np.array(actual_vals, copy=True)),
            "mask_clin": torch.from_numpy(clin_mask),
            "mask_treat": torch.from_numpy(treat_mask),
            "mask_actual": torch.from_numpy(np.array(actual_mask, copy=True)),
            "causal_weight": torch.tensor(self.patient_weights[idx], dtype=torch.float32),
        }
        if state_vals is not None and state_mask is not None:
            sample["state_target"] = torch.from_numpy(state_vals)
            sample["state_target_mask"] = torch.from_numpy(state_mask)
        return sample
def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Apply a masked reduction around elementwise reconstruction losses.

    args:
    - pred: model predictions [torch.Tensor]
    - target: reference values [torch.Tensor]
    - mask: per-element visibility mask [torch.Tensor]
    - loss_fn: reduction-free loss module [nn.Module]
    - patient_weights: optional scalar weight for each patient [torch.Tensor | None]

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
        weights = (
            torch.ones(pred.size(0), device=pred.device, dtype=pred.dtype)
            if patient_weights is None
            else patient_weights.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        )
        if len(weights) != pred.size(0):
            raise ValueError("Patient weights must have one value per batch row")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Patient weights must be finite and non-negative")
        weighted_mask = mask_steps * weights[:, None]
        denom = weighted_mask.sum().clamp_min(1.0)
        return (losses * weighted_mask).sum() / denom

    mask = mask.expand_as(pred).to(dtype=pred.dtype)
    losses = loss_fn(pred, target)
    if losses.shape != pred.shape:
        losses = losses.view_as(pred)
    if patient_weights is not None and patient_weights.ndim == 2:
        observed = mask.sum(dim=2)
        per_step = (losses * mask).sum(dim=2) / observed.clamp_min(1.0)
        return weighted_time_mean(
            per_step,
            patient_weights.to(device=pred.device, dtype=pred.dtype),
            observed > 0,
        )
    weights = (
        torch.ones(pred.size(0), device=pred.device, dtype=pred.dtype)
        if patient_weights is None
        else patient_weights.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    )
    if len(weights) != pred.size(0):
        raise ValueError("Patient weights must have one value per batch row")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("Patient weights must be finite and non-negative")
    weighted_mask = mask * weights[:, None, None]
    denom = weighted_mask.sum().clamp_min(1.0)
    return (losses * weighted_mask).sum() / denom


def masked_binary_outcome_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patient_weights: Optional[torch.Tensor] = None,
    positive_weight: Optional[float] = None,
) -> torch.Tensor:
    """Masked BCE whose risk-set contributions use patient-level causal weights."""

    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError("Outcome logits, target, and mask must share the same shape")
    pos_weight = (
        None
        if positive_weight is None
        else torch.as_tensor(
            float(positive_weight), device=logits.device, dtype=logits.dtype
        )
    )
    if pos_weight is not None and (
        not torch.isfinite(pos_weight) or float(pos_weight) <= 0
    ):
        raise ValueError("positive_weight must be finite and positive")
    losses = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    if patient_weights is not None and patient_weights.ndim == 2:
        observed = mask.sum(dim=2)
        per_step = (losses * mask).sum(dim=2) / observed.clamp_min(1.0)
        return weighted_time_mean(
            per_step,
            patient_weights.to(device=logits.device, dtype=logits.dtype),
            observed > 0,
        )
    weights = (
        torch.ones(logits.size(0), device=logits.device, dtype=logits.dtype)
        if patient_weights is None
        else patient_weights.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    )
    if len(weights) != logits.size(0):
        raise ValueError("Patient weights must have one value per batch row")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("Patient weights must be finite and non-negative")
    weighted_mask = mask * weights[:, None, None]
    denominator = weighted_mask.sum()
    if denominator <= 0:
        return logits.sum() * 0.0
    return (losses * weighted_mask).sum() / denominator


def make_counterfactual_condition(
    cond: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Construct the configured counterfactual treatment assignment."""

    if mode == "two_arm_alternative":
        if cond.shape != mask.shape or cond.size(-1) != 2:
            raise ValueError(
                "two_arm_alternative requires two masked one-hot strategy features"
            )
        observed = cond[mask.bool()]
        if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
            raise ValueError("two_arm_alternative requires binary one-hot values")
        if torch.any(mask[:, 0, :].sum(dim=1) != 2):
            raise ValueError("Both strategy features must be available at baseline")
        base = cond[:, 0, :]
        if torch.any(base.sum(dim=1) != 1):
            raise ValueError("Each patient must have exactly one active strategy")
        valid_rows = mask.sum(dim=2) > 0
        if torch.any(
            (cond - base[:, None, :]).abs().sum(dim=2)[valid_rows] > 0
        ):
            raise ValueError("Two-arm strategy assignment must be static over time")
        alternative = 1 - base.argmax(dim=1)
        changed = F.one_hot(alternative, num_classes=2).to(cond.dtype)
        return changed[:, None, :].repeat(1, cond.size(1), 1) * mask
    if mode == "temporal_step_flip":
        if cond.shape != mask.shape or cond.size(-1) != 1:
            raise ValueError(
                "temporal_step_flip requires one binary treatment value per timestamp"
            )
        observed = cond[mask.bool()]
        if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
            raise ValueError("temporal_step_flip requires binary conditioning values")
        available = mask[:, :, 0].float()
        if torch.any(available.sum(dim=1) <= 0):
            raise ValueError(
                "temporal_step_flip requires an available timestamp per patient"
            )
        timestamp = torch.multinomial(available, num_samples=1)
        flip = torch.zeros_like(cond).scatter(1, timestamp.unsqueeze(-1), 1.0)
        return torch.where(flip.bool(), 1.0 - cond, cond) * mask
    raise ValueError(f"Unsupported counterfactual mode: {mode}")


def regimen_conditioning_loss(
    prediction: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    treatment_mask: torch.Tensor,
    feature_index: int,
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize disagreement with a time-indexed binary treatment regimen."""

    if condition.shape != condition_mask.shape:
        raise ValueError("Condition tensor and mask must share the same shape")
    if prediction.ndim != 3 or treatment_mask.shape != prediction.shape:
        raise ValueError("Prediction and treatment mask must be matching 3D tensors")
    if feature_index < 0 or feature_index >= prediction.size(2):
        raise ValueError("conditioning feature index is outside the prediction tensor")
    if condition.size(2) == 1:
        target = condition[:, :, 0]
        target_mask = condition_mask[:, :, 0]
    elif condition.size(2) == prediction.size(1):
        target = condition[:, 0, :]
        target_mask = condition_mask[:, 0, :]
    else:
        raise ValueError(
            "Conditioning must be time-local or contain one regimen bit per timestamp"
        )
    observed = target[target_mask.bool()]
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("Regimen conditioning targets must be binary")
    predicted_treatment = prediction[:, :, feature_index]
    prediction_mask = treatment_mask[:, :, feature_index]
    mask = target_mask * prediction_mask
    losses = torch.square(predicted_treatment - target)
    if patient_weights is None:
        weights = torch.ones_like(mask)
    else:
        weights = patient_weights.to(device=prediction.device, dtype=prediction.dtype)
        if weights.ndim == 1:
            if len(weights) != prediction.size(0):
                raise ValueError("Patient weights must have one value per prediction row")
            weights = weights[:, None].expand_as(mask)
        elif weights.ndim == 2:
            if weights.shape != mask.shape:
                raise ValueError(
                    "Patient-time weights must match the conditioning loss shape"
                )
        else:
            raise ValueError("Conditioning weights must be patient or patient-time values")
    weighted_mask = mask * weights
    denominator = weighted_mask.sum()
    if denominator <= 0:
        return prediction.sum() * 0.0
    return (losses * weighted_mask).sum() / denominator


def regimen_delivery_loss(
    prediction: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    treatment_mask: torch.Tensor,
    feature_indices: Mapping[int, int],
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Match regimen-arm assignments to any delivery of their RT component.

    ``feature_indices`` maps a condition column to its corresponding generated
    treatment feature. The reserved condition index ``-1`` means the feature
    is required in every strategy (surgery in v6). Surgery-only is represented
    by both RT component targets being zero.
    """

    if prediction.ndim != 3 or prediction.shape != treatment_mask.shape:
        raise ValueError("Prediction and treatment mask must be matching 3D tensors")
    if condition.shape != condition_mask.shape:
        raise ValueError("Condition tensor and mask must share the same shape")
    if not feature_indices:
        raise ValueError("At least one regimen delivery feature is required")
    if torch.any(condition[:, 0, :].sum(dim=1) != 1):
        raise ValueError("Static regimen conditions must be one-hot")

    component_losses: list[torch.Tensor] = []
    for raw_condition_index, raw_feature_index in feature_indices.items():
        condition_index = int(raw_condition_index)
        feature_index = int(raw_feature_index)
        if condition_index != -1 and not 0 <= condition_index < condition.size(2):
            raise ValueError("Regimen condition index is outside the condition tensor")
        if not 0 <= feature_index < prediction.size(2):
            raise ValueError("Regimen delivery feature is outside the prediction tensor")
        target = (
            torch.ones(condition.size(0), device=condition.device, dtype=condition.dtype)
            if condition_index == -1
            else condition[:, 0, condition_index]
        )
        active = treatment_mask[:, :, feature_index].bool()
        logits = 8.0 * (prediction[:, :, feature_index] - 0.5)
        logits = logits.masked_fill(~active, -1.0e4)
        losses = F.binary_cross_entropy_with_logits(
            logits.max(dim=1).values, target, reduction="none"
        )
        condition_valid = (
            condition_mask[:, 0, :].max(dim=1).values
            if condition_index == -1
            else condition_mask[:, 0, condition_index]
        )
        valid = active.any(dim=1).to(losses.dtype) * condition_valid
        weights = (
            torch.ones_like(losses)
            if patient_weights is None
            else patient_weights.to(losses.device, losses.dtype).reshape(-1)
        )
        weighted = valid * weights
        denominator = weighted.sum()
        component_losses.append(
            prediction.sum() * 0.0
            if denominator <= 0
            else (losses * weighted).sum() / denominator
        )
    return torch.stack(component_losses).mean()


def rt_parameter_feasibility_loss(
    prediction: torch.Tensor,
    treatment_mask: torch.Tensor,
    activation_indices: Sequence[int],
    parameter_bounds: Dict[int, Sequence[float]],
    patient_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize dose/fraction values outside protocol ranges when RT is active."""

    if prediction.ndim != 3 or prediction.shape != treatment_mask.shape:
        raise ValueError("Prediction and treatment mask must be matching 3D tensors")
    if not activation_indices or not parameter_bounds:
        raise ValueError("RT feasibility requires activation and parameter features")
    width = prediction.size(2)
    if any(index < 0 or index >= width for index in activation_indices):
        raise ValueError("RT activation feature index is outside the prediction tensor")
    if any(index < 0 or index >= width for index in parameter_bounds):
        raise ValueError("RT parameter feature index is outside the prediction tensor")

    activation = torch.stack(
        [prediction[:, :, index] for index in activation_indices], dim=2
    ).max(dim=2).values
    activation_probability = torch.sigmoid(8.0 * (activation - 0.5))
    activation_mask = torch.stack(
        [treatment_mask[:, :, index] for index in activation_indices], dim=2
    ).max(dim=2).values
    weights = (
        torch.ones(prediction.size(0), device=prediction.device, dtype=prediction.dtype)
        if patient_weights is None
        else patient_weights.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
    )
    if len(weights) != prediction.size(0):
        raise ValueError("Patient weights must have one value per prediction row")

    numerator = prediction.sum() * 0.0
    denominator = prediction.sum() * 0.0
    for raw_index, raw_bounds in parameter_bounds.items():
        index = int(raw_index)
        if len(raw_bounds) != 2:
            raise ValueError("Each RT parameter range must contain lower and upper bounds")
        lower, upper = (float(raw_bounds[0]), float(raw_bounds[1]))
        if not lower < upper:
            raise ValueError("RT parameter lower bound must be below upper bound")
        values = prediction[:, :, index]
        violation = F.relu(lower - values) + F.relu(values - upper)
        active_weight = (
            activation_probability
            * activation_mask
            * treatment_mask[:, :, index]
            * weights[:, None]
        )
        numerator = numerator + (torch.square(violation) * active_weight).sum()
        denominator = denominator + active_weight.sum()
    if denominator <= 0:
        return prediction.sum() * 0.0
    return numerator / denominator


def build_outcome_model_inputs(
    clinical: torch.Tensor,
    clinical_mask: torch.Tensor,
    treatment: torch.Tensor,
    treatment_mask: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    history_only: bool = False,
    history_clinical: Optional[torch.Tensor] = None,
    history_clinical_mask: Optional[torch.Tensor] = None,
    history_treatment: Optional[torch.Tensor] = None,
    history_treatment_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build recurrence inputs at either the legacy or interval-start landmark.

    The chronology-safe preoperative contract uses a dataset-prepared clinical
    prefix and a one-interval-lagged treatment trajectory. The retrospective
    completed-regimen label remains a generator instruction and is withheld
    from the recurrence network.
    """

    if not history_only:
        main = torch.cat((clinical, treatment), dim=2)
        main_mask = torch.cat((clinical_mask, treatment_mask), dim=2)
        return main * main_mask, main_mask, condition, condition_mask

    if history_clinical is None or history_clinical_mask is None:
        raise ValueError(
            "History-only recurrence inputs require interval-start clinical prefixes"
        )

    if history_treatment is None or history_treatment_mask is None:
        history_treatment = torch.zeros_like(treatment)
        history_treatment_mask = torch.zeros_like(treatment_mask)
        if treatment.size(1) > 1:
            history_treatment[:, 1:] = (
                treatment[:, :-1] * treatment_mask[:, :-1]
            )
            history_treatment_mask[:, 1:] = treatment_mask[:, :-1]

    main = torch.cat((history_clinical, history_treatment), dim=2)
    main_mask = torch.cat(
        (history_clinical_mask, history_treatment_mask), dim=2
    )
    safe_condition = torch.zeros_like(condition)
    safe_condition_mask = torch.zeros_like(condition_mask)
    return main * main_mask, main_mask, safe_condition, safe_condition_mask


def configured_feature_index(cfg: Config, feature_name: str) -> int:
    """Resolve the encoded treatment-column index from the configuration."""

    spec = cfg.treatment_features or {}
    ordered: List[str] = []
    for name in [
        *list(spec.get("numeric", []) or []),
        *list(spec.get("one_hot", []) or []),
        *list((spec.get("ordinal", {}) or {}).keys()),
    ]:
        if str(name) not in ordered:
            ordered.append(str(name))
    if feature_name not in ordered:
        raise ValueError(
            f"conditioning_feature {feature_name!r} is not in treatment_features"
        )
    return ordered.index(feature_name)


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
    dataset: SyntheticSequenceDataset,
    batch_size: int,
    seed: int,
    require_explicit_splits: bool = False,
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
    split_values = [patient.get("split") for patient in dataset.patients]
    has_explicit_splits = any(value is not None for value in split_values)
    if has_explicit_splits:
        if any(value is None for value in split_values):
            raise ValueError("Explicit patient splits must be present for every patient")
        labels = {str(value) for value in split_values}
        if labels.issubset({"dev", "test"}):
            train_indices = [
                index for index, value in enumerate(split_values) if value == "dev"
            ]
            val_indices = [
                index for index, value in enumerate(split_values) if value == "test"
            ]
            if not train_indices or not val_indices:
                raise ValueError("Explicit dev/test splits must both be non-empty")
        else:
            invalid = sorted(labels - {"train", "validation", "test"})
            if invalid:
                raise ValueError(f"Unsupported explicit patient splits: {invalid}")
            train_indices = [
                index for index, value in enumerate(split_values) if value == "train"
            ]
            val_indices = [
                index
                for index, value in enumerate(split_values)
                if value == "validation"
            ]
            test_indices = [
                index for index, value in enumerate(split_values) if value == "test"
            ]
            if not train_indices or not val_indices:
                raise ValueError(
                    "Explicit splits require non-empty train and validation sets"
                )
            if require_explicit_splits and not test_indices:
                raise ValueError("Leakage-safe training requires a non-empty test split")
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
    else:
        if require_explicit_splits:
            raise ValueError("Training configuration requires explicit patient splits")
        train_subset, val_subset = split_dataset(
            dataset, val_fraction=0.2, seed=seed
        )
        train_indices = list(getattr(train_subset, "indices", range(len(train_subset))))
        val_indices = list(getattr(val_subset, "indices", range(len(val_subset))))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_indices, val_indices


class LongitudinalWeightDataset(Dataset):
    """Attach aligned patient-time weights to an existing sequence dataset."""

    def __init__(self, dataset: Dataset, weights: pd.DataFrame, seq_len: int) -> None:
        self.dataset = dataset
        self.seq_len = int(seq_len)
        patient_ids = [str(value) for value in getattr(dataset, "patient_ids")]
        observed_ids = set(weights["patient_id"].astype(str))
        missing = sorted(set(patient_ids) - observed_ids)[:5]
        unexpected = sorted(observed_ids - set(patient_ids))[:5]
        if missing or unexpected:
            raise ValueError(
                "Patient-time weights do not match training patients; "
                f"missing={missing}, unexpected={unexpected}"
            )
        self._weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for patient_id, group in weights.groupby("patient_id", sort=False):
            ordered = group.sort_values("interval", kind="stable")
            values = np.zeros(self.seq_len, dtype=np.float32)
            eligible = np.zeros(self.seq_len, dtype=np.float32)
            if len(ordered) > self.seq_len:
                raise ValueError(f"Patient {patient_id} has more weights than sequence steps")
            intervals = ordered["interval"].to_numpy(dtype=int)
            if len(np.unique(intervals)) != len(intervals):
                raise ValueError(f"Patient {patient_id} has duplicate weight intervals")
            if np.any(intervals < 0) or np.any(intervals >= self.seq_len):
                raise ValueError(
                    f"Patient {patient_id} has a weight interval outside [0, {self.seq_len})"
                )
            values[intervals] = ordered["causal_weight"].to_numpy(dtype=np.float32)
            eligible[intervals] = ordered["weight_eligible"].to_numpy(dtype=np.float32)
            self._weights[str(patient_id)] = (values, eligible)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = dict(self.dataset[index])
        patient_id = str(getattr(self.dataset, "patient_ids")[index])
        values, eligible = self._weights[patient_id]
        sample["causal_weight_time"] = torch.from_numpy(values.copy())
        sample["weight_eligible_mask"] = torch.from_numpy(eligible.copy())
        return sample

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def train_epoch(
    Gx: nn.Module,
    Gy: nn.Module,
    Dx: nn.Module,
    Dy: nn.Module,
    loader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_Dx: torch.optim.Optimizer,
    optimizer_Dy: torch.optim.Optimizer,
    criterion_adv: AdversarialLoss,
    criterion_cycle: nn.Module,
    criterion_id: nn.Module,
    cfg: Config,
    device: torch.device,
    outcome_model: Optional[nn.Module] = None,
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
    if outcome_model is not None:
        outcome_model.train()

    totals = {k: 0.0 for k in [
        "g_loss",
        "d_loss",
        "cycle_loss",
        "identity_loss",
        "factual_loss",
        "condition_loss",
        "regimen_delivery_loss",
        "rt_feasibility_loss",
        "generated_outcome_loss",
        "state_loss",
        "gx_adv",
        "gy_adv",
        "dx_loss",
        "dy_loss",
        "dx_loss_unweighted",
        "dy_loss_unweighted",
        "outcome_loss",
    ]}

    for batch in loader:
        cl_real = batch["x_clin"].to(device)
        tr_real = batch["x_treat"].to(device)
        tr_actual = batch["actual_treatment"].to(device)
        mask_clin = batch["mask_clin"].to(device)
        mask_treat = batch["mask_treat"].to(device)
        if not cfg.treatment_prediction:
            if not cfg.conditioning_feature:
                raise ValueError(
                    "treatment_prediction=False requires conditioning_feature"
                )
            mask_treat = mask_treat.clone()
            mask_treat[:, :, configured_feature_index(cfg, cfg.conditioning_feature)] = 0.0
        mask_actual = batch["mask_actual"].to(device)
        state_target = batch.get("state_target")
        state_target_mask = batch.get("state_target_mask")
        if state_target is not None:
            state_target = state_target.to(device)
            state_target_mask = state_target_mask.to(device)
        patient_weights = batch["causal_weight"].to(device).reshape(-1)
        time_weights = batch.get("causal_weight_time")
        weight_eligible = batch.get("weight_eligible_mask")
        if time_weights is not None:
            time_weights = time_weights.to(device)
            weight_eligible = weight_eligible.to(device)
            loss_weights = time_weights * weight_eligible
            unweighted_loss_weights = weight_eligible
        else:
            loss_weights = patient_weights
            unweighted_loss_weights = None

        step_mask = batch.get("sequence_mask")
        if step_mask is None:
            step_mask = (
                (mask_clin.sum(dim=2, keepdim=True) + mask_treat.sum(dim=2, keepdim=True))
                > 0
            ).float()
        else:
            step_mask = step_mask.to(device)
        lengths = step_mask.squeeze(-1).sum(dim=1).clamp(min=1).long()

        cl_real = cl_real * mask_clin
        tr_real = tr_real * mask_treat
        tr_actual = tr_actual * mask_actual
        cond_mask = mask_actual
        history_only_outcome = bool(
            cfg.data_contract == "surgery_rt_preop_two_arm"
            and int(cfg.temporal_alignment.get("contract_version", 1)) >= 2
        )
        history_clinical = (
            batch["x_outcome_clin"].to(device)
            if history_only_outcome
            else None
        )
        history_clinical_mask = (
            batch["mask_outcome_clin"].to(device)
            if history_only_outcome
            else None
        )
        history_treatment = (
            batch["x_outcome_treat"].to(device)
            if history_only_outcome
            else None
        )
        history_treatment_mask = (
            batch["mask_outcome_treat"].to(device)
            if history_only_outcome
            else None
        )
        tr_counter = make_counterfactual_condition(
            tr_actual, mask_actual, cfg.counterfactual_mode
        )

        # Update discriminators on detached generator outputs
        optimizer_Dx.zero_grad()
        optimizer_Dy.zero_grad()

        fake_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
        fake_treat = fake_treat * mask_treat
        gy_cycle_input = torch.cat((cl_real, fake_treat), dim=2)
        gy_cycle_mask = torch.cat((mask_clin, mask_treat), dim=2)
        rec_treat, _ = Gy(
            gy_cycle_input, gy_cycle_mask, tr_actual, cond_mask, lengths
        )
        rec_treat = rec_treat * mask_treat

        main_real = torch.cat((cl_real, tr_real), dim=2)
        main_mask = torch.cat((mask_clin, mask_treat), dim=2)
        main_fake_x = torch.cat((cl_real, fake_treat.detach()), dim=2)
        main_fake_y = torch.cat((cl_real, rec_treat.detach()), dim=2)

        time_resolved = time_weights is not None
        pred_real_x = Dx(main_real, main_mask, tr_actual, cond_mask, lengths, time_resolved)
        pred_fake_x = Dx(main_fake_x, main_mask, tr_counter, cond_mask, lengths, time_resolved)
        pred_real_y = Dy(main_real, main_mask, tr_actual, cond_mask, lengths, time_resolved)
        pred_fake_y = Dy(main_fake_y, main_mask, tr_actual, cond_mask, lengths, time_resolved)

        ones_seq = torch.ones_like(pred_real_x)
        zeros_seq = torch.zeros_like(pred_real_x)

        loss_Dx = 0.5 * (
            criterion_adv(pred_real_x, ones_seq, loss_weights)
            + criterion_adv(pred_fake_x, zeros_seq, loss_weights)
        )
        loss_Dy = 0.5 * (
            criterion_adv(pred_real_y, ones_seq, loss_weights)
            + criterion_adv(pred_fake_y, zeros_seq, loss_weights)
        )
        loss_Dx_unweighted = 0.5 * (
            criterion_adv(pred_real_x, ones_seq, unweighted_loss_weights)
            + criterion_adv(pred_fake_x, zeros_seq, unweighted_loss_weights)
        )
        loss_Dy_unweighted = 0.5 * (
            criterion_adv(pred_real_y, ones_seq, unweighted_loss_weights)
            + criterion_adv(pred_fake_y, zeros_seq, unweighted_loss_weights)
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
        gy_cycle_input = torch.cat((cl_real, fake_treat), dim=2)
        gy_cycle_mask = torch.cat((mask_clin, mask_treat), dim=2)
        rec_treat, _ = Gy(
            gy_cycle_input, gy_cycle_mask, tr_actual, cond_mask, lengths
        )
        rec_treat = rec_treat * mask_treat

        main_fake_x = torch.cat((cl_real, fake_treat), dim=2)
        main_fake_y = torch.cat((cl_real, rec_treat), dim=2)

        pred_fake_x = Dx(main_fake_x, main_mask, tr_counter, cond_mask, lengths, time_resolved)
        pred_fake_y = Dy(main_fake_y, main_mask, tr_actual, cond_mask, lengths, time_resolved)

        loss_Gx_adv = criterion_adv(pred_fake_x, ones_seq, loss_weights)
        loss_Gy_adv = criterion_adv(pred_fake_y, ones_seq, loss_weights)

        cycle_loss = masked_reconstruction_loss(
            rec_treat,
            tr_real,
            mask_treat,
            criterion_cycle,
            loss_weights,
        )

        id_cf, _ = Gx(cl_real, mask_clin, tr_actual, cond_mask, lengths)
        id_cf = id_cf * mask_treat
        gy_identity_input = torch.cat((cl_real, tr_real), dim=2)
        gy_identity_mask = torch.cat((mask_clin, mask_treat), dim=2)
        id_y, _ = Gy(
            gy_identity_input, gy_identity_mask, tr_actual, cond_mask, lengths
        )
        id_y = id_y * mask_treat
        identity_loss = 0.5 * (
            masked_reconstruction_loss(
                id_cf, tr_real, mask_treat, criterion_id, loss_weights
            )
            + masked_reconstruction_loss(
                id_y, tr_real, mask_treat, criterion_id, loss_weights
            )
        )

        outcome_loss = cl_real.new_tensor(0.0)
        if outcome_model is not None:
            outcome = batch["outcome"].to(device)
            outcome_mask = batch["outcome_mask"].to(device)
            (
                outcome_main,
                outcome_main_mask,
                outcome_condition,
                outcome_condition_mask,
            ) = build_outcome_model_inputs(
                cl_real,
                mask_clin,
                tr_real,
                mask_treat,
                tr_actual,
                cond_mask,
                history_only=history_only_outcome,
                history_clinical=history_clinical,
                history_clinical_mask=history_clinical_mask,
                history_treatment=history_treatment,
                history_treatment_mask=history_treatment_mask,
            )
            outcome_logits = outcome_model(
                outcome_main,
                outcome_main_mask,
                outcome_condition,
                outcome_condition_mask,
                lengths,
            )
            outcome_loss = masked_binary_outcome_loss(
                outcome_logits,
                outcome,
                outcome_mask,
                loss_weights,
                cfg.outcome_positive_weight,
            )

        factual_loss = cl_real.new_tensor(0.0)
        generated_outcome_loss = cl_real.new_tensor(0.0)
        state_loss = cl_real.new_tensor(0.0)
        pred_factual: Optional[torch.Tensor] = None
        if (
            cfg.factual_weight > 0
            or cfg.lambda_state > 0
            or cfg.lambda_generated_outcome > 0
            or cfg.lambda_regimen_delivery > 0
            or cfg.lambda_rt_feasibility > 0
        ):
            if isinstance(Gx, RecursiveStateGenerator):
                pred_factual, _ = Gx(
                    cl_real,
                    mask_clin,
                    tr_actual,
                    cond_mask,
                    lengths,
                    state_targets=state_target,
                    state_target_mask=state_target_mask,
                    teacher_forcing_ratio=cfg.state_teacher_forcing_ratio,
                )
            else:
                pred_factual, _ = Gx(
                    cl_real, mask_clin, tr_actual, cond_mask, lengths
                )
            pred_factual = pred_factual * mask_treat
        if cfg.lambda_state > 0:
            if (
                not isinstance(Gx, RecursiveStateGenerator)
                or Gx.last_state_predictions is None
                or state_target is None
                or state_target_mask is None
            ):
                raise ValueError("State loss requires recursive predictions and targets")
            forecast_mask = state_target_mask.clone()
            forecast_mask[:, : Gx.forecast_start] = 0.0
            state_loss = masked_reconstruction_loss(
                Gx.last_state_predictions,
                state_target,
                forecast_mask,
                criterion_cycle,
                loss_weights,
            )
        if cfg.factual_weight > 0 and pred_factual is not None:
            factual_loss = masked_reconstruction_loss(
                pred_factual,
                tr_real,
                mask_treat,
                criterion_cycle,
                loss_weights,
            )
        if (
            cfg.lambda_generated_outcome > 0
            and outcome_model is not None
            and pred_factual is not None
        ):
            (
                generated_main,
                generated_main_mask,
                generated_condition,
                generated_condition_mask,
            ) = build_outcome_model_inputs(
                cl_real,
                mask_clin,
                pred_factual,
                mask_treat,
                tr_actual,
                cond_mask,
                history_only=history_only_outcome,
                history_clinical=history_clinical,
                history_clinical_mask=history_clinical_mask,
            )
            generated_logits = outcome_model(
                generated_main,
                generated_main_mask,
                generated_condition,
                generated_condition_mask,
                lengths,
            )
            generated_outcome_loss = masked_binary_outcome_loss(
                generated_logits,
                outcome,
                outcome_mask,
                loss_weights,
                cfg.outcome_positive_weight,
            )

        condition_loss = cl_real.new_tensor(0.0)
        if cfg.condition_weight > 0:
            if not cfg.conditioning_feature:
                raise ValueError(
                    "condition_weight > 0 requires conditioning_feature"
                )
            condition_loss = regimen_conditioning_loss(
                fake_treat,
                tr_counter,
                cond_mask,
                mask_treat,
                configured_feature_index(cfg, cfg.conditioning_feature),
                loss_weights,
            )

        regimen_loss = cl_real.new_tensor(0.0)
        if cfg.lambda_regimen_delivery > 0:
            if not cfg.regimen_delivery_feature_indices:
                raise ValueError(
                    "Regimen delivery loss requires resolved delivery features"
                )
            counter_delivery = regimen_delivery_loss(
                fake_treat,
                tr_counter,
                cond_mask,
                mask_treat,
                cfg.regimen_delivery_feature_indices,
                patient_weights,
            )
            if pred_factual is None:
                raise ValueError("Regimen delivery loss requires factual predictions")
            factual_delivery = regimen_delivery_loss(
                pred_factual,
                tr_actual,
                cond_mask,
                mask_treat,
                cfg.regimen_delivery_feature_indices,
                patient_weights,
            )
            regimen_loss = 0.5 * (
                counter_delivery + factual_delivery
            )

        rt_feasibility_loss = cl_real.new_tensor(0.0)
        if cfg.lambda_rt_feasibility > 0:
            counter_feasibility = rt_parameter_feasibility_loss(
                fake_treat,
                mask_treat,
                cfg.rt_feasibility_activation_indices,
                cfg.rt_feasibility_parameter_bounds,
                patient_weights,
            )
            if pred_factual is None:
                raise ValueError("RT feasibility loss requires factual predictions")
            factual_feasibility = rt_parameter_feasibility_loss(
                pred_factual,
                mask_treat,
                cfg.rt_feasibility_activation_indices,
                cfg.rt_feasibility_parameter_bounds,
                patient_weights,
            )
            rt_feasibility_loss = 0.5 * (
                counter_feasibility + factual_feasibility
            )

        g_loss = (
            loss_Gx_adv
            + loss_Gy_adv
            + cfg.cycle_weight * cycle_loss
            + cfg.identity_weight * identity_loss
            + cfg.lambda_outcome * outcome_loss
            + cfg.factual_weight * factual_loss
            + cfg.condition_weight * condition_loss
            + cfg.lambda_regimen_delivery * regimen_loss
            + cfg.lambda_rt_feasibility * rt_feasibility_loss
            + cfg.lambda_generated_outcome * generated_outcome_loss
            + cfg.lambda_state * state_loss
        )
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
        totals["factual_loss"] += factual_loss.item()
        totals["condition_loss"] += condition_loss.item()
        totals["regimen_delivery_loss"] += regimen_loss.item()
        totals["rt_feasibility_loss"] += rt_feasibility_loss.item()
        totals["generated_outcome_loss"] += generated_outcome_loss.item()
        totals["state_loss"] += state_loss.item()
        totals["gx_adv"] += loss_Gx_adv.item()
        totals["gy_adv"] += loss_Gy_adv.item()
        totals["dx_loss"] += loss_Dx.item()
        totals["dy_loss"] += loss_Dy.item()
        totals["dx_loss_unweighted"] += loss_Dx_unweighted.item()
        totals["dy_loss_unweighted"] += loss_Dy_unweighted.item()
        totals["outcome_loss"] += outcome_loss.item()

    batches = max(1, len(loader))
    return {k: v / batches for k, v in totals.items()}


def evaluate_epoch(
    Gx: nn.Module,
    Gy: nn.Module,
    Dx: nn.Module,
    Dy: nn.Module,
    loader: DataLoader,
    criterion_adv: AdversarialLoss,
    criterion_cycle: nn.Module,
    criterion_id: nn.Module,
    cfg: Config,
    device: torch.device,
    outcome_model: Optional[nn.Module] = None,
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
    if outcome_model is not None:
        outcome_model.eval()

    totals = {k: 0.0 for k in [
        "g_loss",
        "d_loss",
        "cycle_loss",
        "identity_loss",
        "factual_loss",
        "condition_loss",
        "regimen_delivery_loss",
        "rt_feasibility_loss",
        "generated_outcome_loss",
        "state_loss",
        "gx_adv",
        "gy_adv",
        "dx_loss",
        "dy_loss",
        "dx_loss_unweighted",
        "dy_loss_unweighted",
        "outcome_loss",
    ]}

    with torch.no_grad():
        for batch in loader:
            cl_real = batch["x_clin"].to(device)
            tr_real = batch["x_treat"].to(device)
            tr_actual = batch["actual_treatment"].to(device)
            mask_clin = batch["mask_clin"].to(device)
            mask_treat = batch["mask_treat"].to(device)
            if not cfg.treatment_prediction:
                if not cfg.conditioning_feature:
                    raise ValueError(
                        "treatment_prediction=False requires conditioning_feature"
                    )
                mask_treat = mask_treat.clone()
                mask_treat[:, :, configured_feature_index(cfg, cfg.conditioning_feature)] = 0.0
            mask_actual = batch["mask_actual"].to(device)
            state_target = batch.get("state_target")
            state_target_mask = batch.get("state_target_mask")
            if state_target is not None:
                state_target = state_target.to(device)
                state_target_mask = state_target_mask.to(device)
            patient_weights = batch["causal_weight"].to(device).reshape(-1)
            time_weights = batch.get("causal_weight_time")
            weight_eligible = batch.get("weight_eligible_mask")
            if time_weights is not None:
                time_weights = time_weights.to(device)
                weight_eligible = weight_eligible.to(device)
                loss_weights = time_weights * weight_eligible
                unweighted_loss_weights = weight_eligible
            else:
                loss_weights = patient_weights
                unweighted_loss_weights = None

            step_mask = batch.get("sequence_mask")
            if step_mask is None:
                step_mask = (
                    (
                        mask_clin.sum(dim=2, keepdim=True)
                        + mask_treat.sum(dim=2, keepdim=True)
                    )
                    > 0
                ).float()
            else:
                step_mask = step_mask.to(device)
            lengths = step_mask.squeeze(-1).sum(dim=1).clamp(min=1).long()

            cl_real = cl_real * mask_clin
            tr_real = tr_real * mask_treat
            tr_actual = tr_actual * mask_actual
            cond_mask = mask_actual
            history_only_outcome = bool(
                cfg.data_contract == "surgery_rt_preop_two_arm"
                and int(cfg.temporal_alignment.get("contract_version", 1)) >= 2
            )
            history_clinical = (
                batch["x_outcome_clin"].to(device)
                if history_only_outcome
                else None
            )
            history_clinical_mask = (
                batch["mask_outcome_clin"].to(device)
                if history_only_outcome
                else None
            )
            history_treatment = (
                batch["x_outcome_treat"].to(device)
                if history_only_outcome
                else None
            )
            history_treatment_mask = (
                batch["mask_outcome_treat"].to(device)
                if history_only_outcome
                else None
            )
            tr_counter = make_counterfactual_condition(
                tr_actual, mask_actual, cfg.counterfactual_mode
            )

            fake_treat, _ = Gx(cl_real, mask_clin, tr_counter, cond_mask, lengths)
            fake_treat = fake_treat * mask_treat
            gy_cycle_input = torch.cat((cl_real, fake_treat), dim=2)
            gy_cycle_mask = torch.cat((mask_clin, mask_treat), dim=2)
            rec_treat, _ = Gy(
                gy_cycle_input, gy_cycle_mask, tr_actual, cond_mask, lengths
            )
            rec_treat = rec_treat * mask_treat

            main_real = torch.cat((cl_real, tr_real), dim=2)
            main_mask = torch.cat((mask_clin, mask_treat), dim=2)
            main_fake_x = torch.cat((cl_real, fake_treat), dim=2)
            main_fake_y = torch.cat((cl_real, rec_treat), dim=2)

            time_resolved = time_weights is not None
            pred_fake_x = Dx(main_fake_x, main_mask, tr_counter, cond_mask, lengths, time_resolved)
            pred_fake_y = Dy(main_fake_y, main_mask, tr_actual, cond_mask, lengths, time_resolved)
            pred_real_x = Dx(main_real, main_mask, tr_actual, cond_mask, lengths, time_resolved)
            pred_real_y = Dy(main_real, main_mask, tr_actual, cond_mask, lengths, time_resolved)

            ones_seq = torch.ones_like(pred_real_x)
            zeros_seq = torch.zeros_like(pred_real_x)

            loss_Gx_adv = criterion_adv(pred_fake_x, ones_seq, loss_weights)
            loss_Gy_adv = criterion_adv(pred_fake_y, ones_seq, loss_weights)
            cycle_loss = masked_reconstruction_loss(
                rec_treat,
                tr_real,
                mask_treat,
                criterion_cycle,
                loss_weights,
            )
            id_cf, _ = Gx(cl_real, mask_clin, tr_actual, cond_mask, lengths)
            id_cf = id_cf * mask_treat
            gy_identity_input = torch.cat((cl_real, tr_real), dim=2)
            gy_identity_mask = torch.cat((mask_clin, mask_treat), dim=2)
            id_y, _ = Gy(
                gy_identity_input, gy_identity_mask, tr_actual, cond_mask, lengths
            )
            id_y = id_y * mask_treat
            identity_loss = 0.5 * (
                masked_reconstruction_loss(
                    id_cf, tr_real, mask_treat, criterion_id, loss_weights
                )
                + masked_reconstruction_loss(
                    id_y, tr_real, mask_treat, criterion_id, loss_weights
                )
            )

            loss_Dx = 0.5 * (
                criterion_adv(pred_real_x, ones_seq, loss_weights)
                + criterion_adv(pred_fake_x, zeros_seq, loss_weights)
            )
            loss_Dy = 0.5 * (
                criterion_adv(pred_real_y, ones_seq, loss_weights)
                + criterion_adv(pred_fake_y, zeros_seq, loss_weights)
            )
            loss_Dx_unweighted = 0.5 * (
                criterion_adv(pred_real_x, ones_seq, unweighted_loss_weights)
                + criterion_adv(pred_fake_x, zeros_seq, unweighted_loss_weights)
            )
            loss_Dy_unweighted = 0.5 * (
                criterion_adv(pred_real_y, ones_seq, unweighted_loss_weights)
                + criterion_adv(pred_fake_y, zeros_seq, unweighted_loss_weights)
            )

            outcome_loss = cl_real.new_tensor(0.0)
            if outcome_model is not None:
                outcome = batch["outcome"].to(device)
                outcome_mask = batch["outcome_mask"].to(device)
                (
                    outcome_main,
                    outcome_main_mask,
                    outcome_condition,
                    outcome_condition_mask,
                ) = build_outcome_model_inputs(
                    cl_real,
                    mask_clin,
                    tr_real,
                    mask_treat,
                    tr_actual,
                    cond_mask,
                    history_only=history_only_outcome,
                    history_clinical=history_clinical,
                    history_clinical_mask=history_clinical_mask,
                    history_treatment=history_treatment,
                    history_treatment_mask=history_treatment_mask,
                )
                outcome_logits = outcome_model(
                    outcome_main,
                    outcome_main_mask,
                    outcome_condition,
                    outcome_condition_mask,
                    lengths,
                )
                outcome_loss = masked_binary_outcome_loss(
                    outcome_logits,
                    outcome,
                    outcome_mask,
                    loss_weights,
                    cfg.outcome_positive_weight,
                )

            factual_loss = cl_real.new_tensor(0.0)
            generated_outcome_loss = cl_real.new_tensor(0.0)
            state_loss = cl_real.new_tensor(0.0)
            pred_factual: Optional[torch.Tensor] = None
            if (
                cfg.factual_weight > 0
                or cfg.lambda_state > 0
                or cfg.lambda_generated_outcome > 0
                or cfg.lambda_regimen_delivery > 0
                or cfg.lambda_rt_feasibility > 0
            ):
                if isinstance(Gx, RecursiveStateGenerator):
                    pred_factual, _ = Gx(
                        cl_real,
                        mask_clin,
                        tr_actual,
                        cond_mask,
                        lengths,
                        state_targets=state_target,
                        state_target_mask=state_target_mask,
                        teacher_forcing_ratio=0.0,
                    )
                else:
                    pred_factual, _ = Gx(
                        cl_real, mask_clin, tr_actual, cond_mask, lengths
                    )
                pred_factual = pred_factual * mask_treat
            if cfg.lambda_state > 0:
                if (
                    not isinstance(Gx, RecursiveStateGenerator)
                    or Gx.last_state_predictions is None
                    or state_target is None
                    or state_target_mask is None
                ):
                    raise ValueError("State loss requires recursive predictions and targets")
                forecast_mask = state_target_mask.clone()
                forecast_mask[:, : Gx.forecast_start] = 0.0
                state_loss = masked_reconstruction_loss(
                    Gx.last_state_predictions,
                    state_target,
                    forecast_mask,
                    criterion_cycle,
                    loss_weights,
                )
            if cfg.factual_weight > 0 and pred_factual is not None:
                factual_loss = masked_reconstruction_loss(
                    pred_factual,
                    tr_real,
                    mask_treat,
                    criterion_cycle,
                    loss_weights,
                )
            if (
                cfg.lambda_generated_outcome > 0
                and outcome_model is not None
                and pred_factual is not None
            ):
                (
                    generated_main,
                    generated_main_mask,
                    generated_condition,
                    generated_condition_mask,
                ) = build_outcome_model_inputs(
                    cl_real,
                    mask_clin,
                    pred_factual,
                    mask_treat,
                    tr_actual,
                    cond_mask,
                    history_only=history_only_outcome,
                    history_clinical=history_clinical,
                    history_clinical_mask=history_clinical_mask,
                )
                generated_logits = outcome_model(
                    generated_main,
                    generated_main_mask,
                    generated_condition,
                    generated_condition_mask,
                    lengths,
                )
                generated_outcome_loss = masked_binary_outcome_loss(
                    generated_logits,
                    outcome,
                    outcome_mask,
                    loss_weights,
                    cfg.outcome_positive_weight,
                )

            condition_loss = cl_real.new_tensor(0.0)
            if cfg.condition_weight > 0:
                if not cfg.conditioning_feature:
                    raise ValueError(
                        "condition_weight > 0 requires conditioning_feature"
                    )
                condition_loss = regimen_conditioning_loss(
                    fake_treat,
                    tr_counter,
                    cond_mask,
                    mask_treat,
                    configured_feature_index(cfg, cfg.conditioning_feature),
                    loss_weights,
                )

            regimen_loss = cl_real.new_tensor(0.0)
            if cfg.lambda_regimen_delivery > 0:
                if not cfg.regimen_delivery_feature_indices:
                    raise ValueError(
                        "Regimen delivery loss requires resolved delivery features"
                    )
                counter_delivery = regimen_delivery_loss(
                    fake_treat,
                    tr_counter,
                    cond_mask,
                    mask_treat,
                    cfg.regimen_delivery_feature_indices,
                    patient_weights,
                )
                if pred_factual is None:
                    raise ValueError(
                        "Regimen delivery loss requires factual predictions"
                    )
                factual_delivery = regimen_delivery_loss(
                    pred_factual,
                    tr_actual,
                    cond_mask,
                    mask_treat,
                    cfg.regimen_delivery_feature_indices,
                    patient_weights,
                )
                regimen_loss = 0.5 * (
                    counter_delivery + factual_delivery
                )

            rt_feasibility_loss = cl_real.new_tensor(0.0)
            if cfg.lambda_rt_feasibility > 0:
                counter_feasibility = rt_parameter_feasibility_loss(
                    fake_treat,
                    mask_treat,
                    cfg.rt_feasibility_activation_indices,
                    cfg.rt_feasibility_parameter_bounds,
                    patient_weights,
                )
                if pred_factual is None:
                    raise ValueError("RT feasibility loss requires factual predictions")
                factual_feasibility = rt_parameter_feasibility_loss(
                    pred_factual,
                    mask_treat,
                    cfg.rt_feasibility_activation_indices,
                    cfg.rt_feasibility_parameter_bounds,
                    patient_weights,
                )
                rt_feasibility_loss = 0.5 * (
                    counter_feasibility + factual_feasibility
                )

            totals["g_loss"] += (
                loss_Gx_adv
                + loss_Gy_adv
                + cfg.cycle_weight * cycle_loss
                + cfg.identity_weight * identity_loss
                + cfg.lambda_outcome * outcome_loss
                + cfg.factual_weight * factual_loss
                + cfg.condition_weight * condition_loss
                + cfg.lambda_regimen_delivery * regimen_loss
                + cfg.lambda_rt_feasibility * rt_feasibility_loss
                + cfg.lambda_generated_outcome * generated_outcome_loss
                + cfg.lambda_state * state_loss
            ).item()
            totals["d_loss"] += (loss_Dx + loss_Dy).item()
            totals["cycle_loss"] += cycle_loss.item()
            totals["identity_loss"] += identity_loss.item()
            totals["factual_loss"] += factual_loss.item()
            totals["condition_loss"] += condition_loss.item()
            totals["regimen_delivery_loss"] += regimen_loss.item()
            totals["rt_feasibility_loss"] += rt_feasibility_loss.item()
            totals["generated_outcome_loss"] += generated_outcome_loss.item()
            totals["state_loss"] += state_loss.item()
            totals["gx_adv"] += loss_Gx_adv.item()
            totals["gy_adv"] += loss_Gy_adv.item()
            totals["dx_loss"] += loss_Dx.item()
            totals["dy_loss"] += loss_Dy.item()
            totals["dx_loss_unweighted"] += loss_Dx_unweighted.item()
            totals["dy_loss_unweighted"] += loss_Dy_unweighted.item()
            totals["outcome_loss"] += outcome_loss.item()

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
    "RecursiveStateGenerator",
    "LSTMDiscriminator",
    "LSTMOutcomePredictor",
    "build_generator",
    "build_clinical_generator",
    "build_discriminator",
    "build_outcome_predictor",
    "weighted_mean",
    "weighted_time_mean",
    "masked_reconstruction_loss",
    "masked_binary_outcome_loss",
    "make_counterfactual_condition",
    "regimen_conditioning_loss",
    "regimen_delivery_loss",
    "rt_parameter_feasibility_loss",
    "configured_feature_index",
    "train_epoch",
    "evaluate_epoch",
    "train",
]


def train(
    cfg: Config,
    synthetic_data: Optional[str] = None,
    person_period_data: Optional[str | Path] = None,
    seed: Optional[int] = None,
    checkpoint_dir: str | Path = "data/models",
    causal_weights_path: Optional[str | Path] = None,
    causal_weight_metadata_path: Optional[str | Path] = None,
    longitudinal_weights_path: Optional[str | Path] = None,
    longitudinal_weight_metadata_path: Optional[str | Path] = None,
    split_assignments: Optional[Mapping[str, str]] = None,
) -> Path:
    '''
    Top-level training routine handling data, models, and checkpoints.

    args:
    - cfg: parsed training configuration [Config]
    - synthetic_data: path to a prepared JSON dataset [str | None]
    - person_period_data: strict-D person-period CSV for sarcoma contracts [str | Path | None]
    - seed: optional override for dataset RNG [int | None]
    - checkpoint_dir: directory for saving checkpoints [str | Path]
    - causal_weights_path: validated patient-level causal-weight CSV [str | Path | None]
    - causal_weight_metadata_path: support/provenance metadata JSON [str | Path | None]

    return:
    - ckpt_path: path to written checkpoint file [Path]
    '''
    run_seed = seed if seed is not None else cfg.seed
    set_seed(run_seed)
    device = resolve_device(cfg.device)

    # Load either synthetic sequences or the fixed sarcoma cohort contract.
    if cfg.data_contract == "surgery_rt_preop_two_arm":
        if person_period_data is None:
            raise ValueError(
                "surgery_rt_preop_two_arm training requires --person-period-data"
            )
        if synthetic_data is not None:
            raise ValueError(
                "--synthetic-data cannot be combined with "
                "surgery_rt_preop_two_arm training"
            )
        dataset_dict = build_surgery_rt_preop_two_arm_training_data(
            person_period_data,
            cfg.feature_groups,
            seq_len=cfg.seq_len,
            cohort_contract=cfg.cohort_contract,
            split_seed=cfg.split_seed,
            validation_fraction=cfg.validation_fraction,
            test_fraction=cfg.test_fraction,
            split_assignments=split_assignments,
        )
    elif cfg.data_contract == "synthetic" and synthetic_data is not None:
        with open(synthetic_data, "r", encoding="utf-8") as fh:
            dataset_dict = json.load(fh)
    elif cfg.data_contract == "synthetic":
        raise ValueError(
            "Synthetic training requires a prepared dataset via --synthetic-data"
        )
    else:
        raise ValueError(f"Unsupported data contract: {cfg.data_contract}")

    dataset_patients = dataset_dict.get("patients", [])
    if not isinstance(dataset_patients, list) or not dataset_patients:
        raise ValueError("Training data must contain a non-empty 'patients' list")
    patient_ids = [str(patient.get("patient_id")) for patient in dataset_patients]
    if any(patient.get("patient_id") is None for patient in dataset_patients):
        raise ValueError("Every training patient must have a patient_id")

    causal_settings = cfg.causal_weighting or {}
    weighting_enabled = bool(
        causal_settings.get("enabled", False)
        or causal_weights_path
        or longitudinal_weights_path
    )
    causal_weight_table = None
    longitudinal_weight_frame: Optional[pd.DataFrame] = None
    longitudinal_weight_metadata: dict[str, Any] = {}
    weight_by_patient: Optional[dict[str, float]] = None
    if weighting_enabled:
        if causal_weights_path is None and longitudinal_weights_path is None:
            raise ValueError(
                "Causal weighting is enabled, but no causal-weight artifact was provided"
            )
        required_components = {
            "discriminator",
            "generator_adversarial",
            "cycle",
            "identity",
        }
        if cfg.data_contract == "surgery_rt_preop_two_arm":
            required_components.add("outcome")
        configured_components = set(
            causal_settings.get("apply_to", sorted(required_components))
        )
        if configured_components != required_components:
            raise ValueError(
                "Causal weights must be applied consistently to all losses; "
                f"expected {sorted(required_components)}, observed {sorted(configured_components)}"
            )
        if causal_weights_path is not None:
            weight_spec = CausalWeightSpec(
                method=str(causal_settings.get("method", "primary_overlap")),
                stage=str(causal_settings.get("stage", "preoperative_rt")),
                ineligible_policy=str(causal_settings.get("ineligible_policy", "exclude")),
                normalization=str(causal_settings.get("normalization", "mean_one")),
                require_supported=bool(causal_settings.get("require_supported", True)),
            )
            causal_weight_table = load_training_weights(
                causal_weights_path,
                spec=weight_spec,
                metadata_path=causal_weight_metadata_path,
                expected_patient_ids=patient_ids,
            )
            expected_target = causal_settings.get("target_population")
            observed_target = causal_weight_table.metadata.get("weighting_target")
            if expected_target is not None and observed_target != expected_target:
                raise ValueError(
                    "Causal-weight target population does not match training configuration; "
                    f"expected {expected_target!r}, observed {observed_target!r}"
                )
            weight_by_patient = dict(
                zip(
                    causal_weight_table.frame[weight_spec.patient_id_col].astype(str),
                    causal_weight_table.frame[weight_spec.weight_col].astype(float),
                )
            )
        if longitudinal_weights_path is not None:
            longitudinal_weight_frame = validate_longitudinal_training_weights(
                pd.read_csv(longitudinal_weights_path, dtype={"patient_id": "string"})
            )
            metadata_source = (
                Path(longitudinal_weight_metadata_path)
                if longitudinal_weight_metadata_path is not None
                else Path(longitudinal_weights_path).with_name(
                    f"{Path(longitudinal_weights_path).stem}_metadata.json"
                )
            )
            if metadata_source.exists():
                longitudinal_weight_metadata = json.loads(
                    metadata_source.read_text(encoding="utf-8")
                )

    if cfg.data_contract == "surgery_rt_preop_two_arm":
        dataset = SarcomaSequenceDataset(
            dataset_dict,
            cfg.seq_len,
            feature_groups=cfg.feature_groups,
            patient_weights=weight_by_patient,
        )
    else:
        dataset = SyntheticSequenceDataset(
            dataset_dict,
            cfg.seq_len,
            spec_clinical=cfg.clinical_features,
            spec_treatment=cfg.treatment_features,
            patient_weights=weight_by_patient,
            broadcast_actual_treatment=cfg.broadcast_actual_treatment,
        )
    if longitudinal_weight_frame is not None:
        dataset = LongitudinalWeightDataset(
            dataset, longitudinal_weight_frame, cfg.seq_len
        )
    train_loader, val_loader, train_indices, val_indices = create_dataloaders(
        dataset,
        cfg.batch_size,
        run_seed,
        require_explicit_splits=cfg.require_explicit_splits,
    )
    split_labels = {
        patient.get("split") for patient in getattr(dataset, "patients", [])
    }
    sarcoma_dev_test = cfg.data_contract.startswith("surgery_rt_") and split_labels == {
        "dev",
        "test",
    }
    test_indices = (
        list(val_indices)
        if sarcoma_dev_test
        else [
            index
            for index, patient in enumerate(getattr(dataset, "patients", []))
            if patient.get("split") == "test"
        ]
    )
    epoch_evaluation_loader = (
        DataLoader(
            torch.utils.data.Subset(dataset, train_indices),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        if sarcoma_dev_test
        else val_loader
    )

    cond_dim = dataset.cond_dim
    clin_dim = dataset.clin_dim
    treat_dim = dataset.treat_dim

    # Resolve optional paper losses before constructing the networks.
    if min(
        cfg.factual_weight,
        cfg.lambda_state,
        cfg.condition_weight,
        cfg.lambda_regimen_delivery,
        cfg.lambda_generated_outcome,
        cfg.lambda_rt_feasibility,
    ) < 0:
        raise ValueError("All auxiliary loss weights must be non-negative")
    if not 0.0 <= cfg.state_teacher_forcing_ratio <= 1.0:
        raise ValueError("state_teacher_forcing_ratio must lie in [0, 1]")
    if cfg.recursive_state_generation and cfg.lambda_state <= 0:
        raise ValueError("Recursive state generation requires lambda_state > 0")
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if cfg.test_fraction is not None and not 0.0 < cfg.test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")
    if cfg.conditioning_contract == "two_arm_static":
        if cfg.data_contract != "surgery_rt_preop_two_arm":
            raise ValueError(
                "two_arm_static conditioning requires surgery_rt_preop_two_arm data"
            )
        if cfg.counterfactual_mode != "two_arm_alternative":
            raise ValueError(
                "two_arm_static conditioning requires two_arm_alternative"
            )
        if cond_dim != 2:
            raise ValueError("two_arm_static conditioning requires two strategies")
        if not cfg.broadcast_actual_treatment:
            raise ValueError("two_arm_static must repeat the baseline strategy")
    if cfg.lambda_regimen_delivery > 0:
        if cfg.conditioning_contract != "two_arm_static":
            raise ValueError(
                "Regimen delivery consistency requires static strategy conditioning"
            )
        if not cfg.regimen_delivery_features:
            raise ValueError(
                "lambda_regimen_delivery > 0 requires regimen_delivery_features"
            )
        resolved: Dict[int, int] = {}
        for condition_name, feature_name in cfg.regimen_delivery_features.items():
            if (
                condition_name != "all_strategies"
                and condition_name not in getattr(dataset, "cond_columns", [])
            ):
                raise ValueError(
                    f"Unknown regimen condition {condition_name!r}"
                )
            if feature_name not in getattr(dataset, "treat_columns", []):
                raise ValueError(
                    f"Regimen delivery feature {feature_name!r} is not encoded"
                )
            condition_index = (
                -1
                if condition_name == "all_strategies"
                else dataset.cond_columns.index(condition_name)
            )
            resolved[condition_index] = (
                dataset.treat_columns.index(feature_name)
            )
        cfg.regimen_delivery_feature_indices = resolved
    if cfg.lambda_rt_feasibility > 0:
        if cfg.data_contract != "surgery_rt_preop_two_arm":
            raise ValueError("RT feasibility loss is reserved for the sarcoma contract")
        activation_features = list(
            cfg.rt_feasibility.get("activation_features", []) or []
        )
        parameter_ranges = dict(
            cfg.rt_feasibility.get("parameter_ranges", {}) or {}
        )
        if not activation_features or not parameter_ranges:
            raise ValueError(
                "lambda_rt_feasibility > 0 requires activation_features and parameter_ranges"
            )
        missing = [
            name
            for name in [*activation_features, *parameter_ranges]
            if name not in getattr(dataset, "treat_columns", [])
        ]
        if missing:
            raise ValueError(f"RT feasibility features are not encoded: {missing}")
        cfg.rt_feasibility_activation_indices = [
            dataset.treat_columns.index(name) for name in activation_features
        ]
        normalized_bounds: Dict[int, List[float]] = {}
        numeric_stats = getattr(dataset.treatment_encoder, "numeric_stats", {})
        for feature, raw_bounds in parameter_ranges.items():
            if feature not in numeric_stats:
                raise ValueError(
                    f"RT feasibility parameter {feature!r} is not numeric"
                )
            if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 2:
                raise ValueError(
                    f"RT feasibility bounds for {feature!r} must be [lower, upper]"
                )
            lower, upper = float(raw_bounds[0]), float(raw_bounds[1])
            if not lower < upper:
                raise ValueError(
                    f"RT feasibility bounds for {feature!r} are not increasing"
                )
            mean, std = numeric_stats[feature]
            normalized_bounds[dataset.treat_columns.index(feature)] = [
                (lower - mean) / std,
                (upper - mean) / std,
            ]
        cfg.rt_feasibility_parameter_bounds = normalized_bounds
    if cfg.condition_weight > 0:
        if not cfg.conditioning_feature:
            raise ValueError("condition_weight > 0 requires conditioning_feature")
        configured_feature_index(cfg, cfg.conditioning_feature)
        if cfg.conditioning_contract == "time_local":
            if cfg.counterfactual_mode != "temporal_step_flip":
                raise ValueError(
                    "time_local conditioning requires counterfactual_mode='temporal_step_flip'"
                )
            if cfg.broadcast_actual_treatment:
                raise ValueError(
                    "time_local conditioning cannot broadcast the first treatment row"
                )
            if cond_dim != 1:
                raise ValueError(
                    "time_local conditioning requires one treatment condition feature"
                )
            if cfg.architecture not in {"lstm", "temporal_mlp"}:
                raise ValueError(
                    "time_local leakage-safe conditioning requires a causal LSTM or "
                    "a per-timestamp temporal MLP"
                )
        elif cfg.conditioning_contract == "two_arm_static":
            raise ValueError(
                "Static strategy conditioning uses lambda_regimen_delivery; "
                "condition_weight must be 0"
            )
        else:
            raise ValueError(
                f"Unsupported conditioning_contract: {cfg.conditioning_contract!r}"
            )

    temporal_version = int(cfg.temporal_alignment.get("contract_version", 1))
    if temporal_version >= 2:
        if cfg.data_contract != "surgery_rt_preop_two_arm":
            raise ValueError(
                "Temporal contract v2 is restricted to preoperative two-arm training"
            )
        expected_temporal = {
            "treatment_scope": "preoperative_only",
            "postoperative_policy": "exclude_patient",
            "prediction_landmark": "interval_start",
            "history_lag_intervals": 1,
            "outcome_uses_static_regimen": False,
            "recurrence_network": "forward_causal",
        }
        mismatches = {
            key: (cfg.temporal_alignment.get(key), expected)
            for key, expected in expected_temporal.items()
            if cfg.temporal_alignment.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Invalid chronology-safe temporal contract settings: "
                f"{mismatches}"
            )
        if cfg.architecture != "lstm":
            raise ValueError("Temporal contract v2 requires a forward LSTM")
        if not getattr(dataset, "history_only_outcome", False):
            raise ValueError(
                "Temporal contract v2 requires history-only recurrence tensors"
            )

    if cfg.architecture not in {"lstm", "temporal_mlp"}:
        raise ValueError("architecture must be 'lstm' or 'temporal_mlp'")

    # Build the paired generators and discriminators.
    Gx = build_clinical_generator(cfg, dataset).to(device)
    Gy = build_generator(
        cfg, clin_dim + treat_dim, cond_dim, cfg.generator_hidden, treat_dim
    ).to(device)
    Dx = build_discriminator(
        cfg, clin_dim + treat_dim, cond_dim, cfg.discriminator_hidden
    ).to(device)
    Dy = build_discriminator(
        cfg, clin_dim + treat_dim, cond_dim, cfg.discriminator_hidden
    ).to(device)
    outcome_dim = int(getattr(dataset, "outcome_dim", 0))
    if cfg.lambda_generated_outcome > 0 and outcome_dim == 0:
        raise ValueError("Generated-outcome loss requires an outcome model")
    if cfg.outcome_positive_weight_cap <= 0:
        raise ValueError("outcome_positive_weight_cap must be positive")
    if outcome_dim and cfg.outcome_positive_weight is None:
        observed_outcomes = 0.0
        positive_outcomes = 0.0
        for index in train_indices:
            sample = dataset[index]
            target = sample["outcome"].float()
            mask = sample["outcome_mask"].float()
            observed_outcomes += float(mask.sum())
            positive_outcomes += float((target * mask).sum())
        negative_outcomes = observed_outcomes - positive_outcomes
        if positive_outcomes <= 0 or negative_outcomes <= 0:
            raise ValueError(
                "Training split must contain positive and negative outcome rows"
            )
        cfg.outcome_positive_weight = min(
            negative_outcomes / positive_outcomes,
            float(cfg.outcome_positive_weight_cap),
        )
    outcome_model: Optional[nn.Module] = None
    if outcome_dim:
        outcome_model = build_outcome_predictor(
            cfg, clin_dim + treat_dim, cond_dim, outcome_dim
        ).to(device)

    generator_parameters = list(Gx.parameters()) + list(Gy.parameters())
    if outcome_model is not None:
        generator_parameters.extend(outcome_model.parameters())

    # Generators share one optimizer; discriminators update separately.
    optimizer_G = get_optimizer(
        cfg.opt_gen,
        generator_parameters,
        cfg.generator_lr,
        (cfg.beta1, cfg.beta2),
    )
    optimizer_Dx = get_optimizer(
        cfg.opt_disc,
        Dx.parameters(),
        cfg.discriminator_lr,
        (cfg.beta1, cfg.beta2),
    )
    optimizer_Dy = get_optimizer(
        cfg.opt_disc,
        Dy.parameters(),
        cfg.discriminator_lr,
        (cfg.beta1, cfg.beta2),
    )

    criterion_adv = get_adversarial_loss(cfg.adv_loss)
    criterion_cycle = get_criterion(cfg.criterion)
    criterion_id = get_criterion(cfg.criterion)

    # Retain every epoch for the paper diagnostics.
    training_history: List[Dict[str, Any]] = []
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
            outcome_model=outcome_model,
        )
        val_metrics = evaluate_epoch(
            Gx,
            Gy,
            Dx,
            Dy,
            epoch_evaluation_loader,
            criterion_adv,
            criterion_cycle,
            criterion_id,
            cfg,
            device,
            outcome_model=outcome_model,
        )
        history_entry = {
            "epoch": epoch + 1,
            "train": {key: float(value) for key, value in train_metrics.items()},
            ("development" if sarcoma_dev_test else "validation"): {
                key: float(value) for key, value in val_metrics.items()
            },
        }
        training_history.append(history_entry)
        evaluation_label = "dev" if sarcoma_dev_test else "val"
        print(
            "Epoch {}/{} "
            "train[G={:.4f} D={:.4f} cycle={:.4f} id={:.4f} factual={:.4f} condition={:.4f} regimen={:.4f} rt={:.4f} outcome={:.4f} gen_outcome={:.4f} Gx_adv={:.4f} Gy_adv={:.4f} Dx={:.4f} Dy={:.4f}] "
            "{}[G={:.4f} D={:.4f} cycle={:.4f} id={:.4f} factual={:.4f} condition={:.4f} regimen={:.4f} rt={:.4f} outcome={:.4f} gen_outcome={:.4f} Gx_adv={:.4f} Gy_adv={:.4f} Dx={:.4f} Dy={:.4f}]".format(
                epoch + 1,
                cfg.epochs,
                train_metrics["g_loss"],
                train_metrics["d_loss"],
                train_metrics["cycle_loss"],
                train_metrics["identity_loss"],
                train_metrics["factual_loss"],
                train_metrics["condition_loss"],
                train_metrics["regimen_delivery_loss"],
                train_metrics["rt_feasibility_loss"],
                train_metrics["outcome_loss"],
                train_metrics["generated_outcome_loss"],
                train_metrics["gx_adv"],
                train_metrics["gy_adv"],
                train_metrics["dx_loss"],
                train_metrics["dy_loss"],
                evaluation_label,
                val_metrics["g_loss"],
                val_metrics["d_loss"],
                val_metrics["cycle_loss"],
                val_metrics["identity_loss"],
                val_metrics["factual_loss"],
                val_metrics["condition_loss"],
                val_metrics["regimen_delivery_loss"],
                val_metrics["rt_feasibility_loss"],
                val_metrics["outcome_loss"],
                val_metrics["generated_outcome_loss"],
                val_metrics["gx_adv"],
                val_metrics["gy_adv"],
                val_metrics["dx_loss"],
                val_metrics["dy_loss"],
            )
        )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save the model with its split and complete training history.
    ckpt_path = checkpoint_dir / f"dtcygan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    def split_weight_summary(indices: List[int]) -> Dict[str, float | int]:
        values = np.asarray(dataset.patient_weights[indices], dtype=float)
        denominator = float(np.square(values).sum())
        effective_sample_size = (
            float(values.sum() ** 2 / denominator) if denominator > 0 else 0.0
        )
        return {
            "n": int(len(values)),
            "sum": float(values.sum()),
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "effective_sample_size": effective_sample_size,
        }

    causal_checkpoint_metadata: Dict[str, Any] = {"enabled": weighting_enabled}
    if causal_weight_table is not None:
        resolved_weight_path = Path(causal_weights_path).resolve()
        resolved_metadata_path = (
            Path(causal_weight_metadata_path).resolve()
            if causal_weight_metadata_path is not None
            else resolved_weight_path.with_name(
                f"{resolved_weight_path.stem}_metadata.json"
            )
        )
        causal_checkpoint_metadata.update(
            {
                "method": causal_weight_table.metadata.get("method"),
                "stage": causal_weight_table.metadata.get("stage"),
                "target_population": causal_weight_table.metadata.get("weighting_target"),
                "adjustment_supported": causal_weight_table.metadata.get(
                    "adjustment_supported"
                ),
                "n_patients": int(len(causal_weight_table.frame)),
                "weight_summary": causal_weight_table.metadata.get("weight_summary"),
                "gate_metrics": causal_weight_table.metadata.get("gate_metrics"),
                "source_csv": str(resolved_weight_path),
                "source_metadata": str(resolved_metadata_path),
                ("dev_split" if sarcoma_dev_test else "train_split"): split_weight_summary(
                    train_indices
                ),
                ("test_split" if sarcoma_dev_test else "validation_split"): split_weight_summary(
                    val_indices
                ),
            }
        )
    if longitudinal_weight_frame is not None:
        causal_checkpoint_metadata.update(
            {
                "table_type": "patient_time",
                "scheme": longitudinal_weight_metadata.get("scheme"),
                "truncation_target": longitudinal_weight_metadata.get(
                    "truncation_target"
                ),
                "truncation_threshold": longitudinal_weight_metadata.get(
                    "truncation_threshold"
                ),
                "original_distribution": longitudinal_weight_metadata.get(
                    "original_distribution"
                ),
                "normalized_distribution": longitudinal_weight_metadata.get(
                    "normalized_distribution"
                ),
                "n_truncated": longitudinal_weight_metadata.get("n_truncated", 0),
                "source_longitudinal_csv": str(
                    Path(longitudinal_weights_path).resolve()
                ),
            }
        )

    split_provenance = (
        {
            "dev_patient_ids": [
                dataset.patients[idx]["patient_id"] for idx in train_indices
            ],
            "test_patient_ids": [
                dataset.patients[idx]["patient_id"] for idx in test_indices
            ],
            "test_evaluated_during_training": False,
        }
        if sarcoma_dev_test
        else {
            "validation_patient_ids": [
                dataset.patients[idx]["patient_id"] for idx in val_indices
            ],
            "train_patient_ids": [
                dataset.patients[idx]["patient_id"] for idx in train_indices
            ],
            "test_patient_ids": [
                dataset.patients[idx]["patient_id"] for idx in test_indices
            ],
        }
    )

    torch.save(
        {
            "config": cfg.__dict__,
            "metadata": {
                "clin_dim": clin_dim,
                "treat_dim": treat_dim,
                "cond_dim": cond_dim,
                "outcome_dim": outcome_dim,
                "data_contract": cfg.data_contract,
                "clinical_feature_spec": getattr(dataset, "spec_clinical", {}),
                "treatment_feature_spec": getattr(dataset, "spec_treatment", {}),
                "outcome_feature_spec": getattr(dataset, "spec_outcome", {}),
                "clinical_feature_names": getattr(dataset, "clin_columns", []),
                "treatment_feature_names": getattr(dataset, "treat_columns", []),
                "outcome_feature_names": getattr(dataset, "outcome_columns", []),
                "cohort_contract": cfg.cohort_contract,
                "audit_only": cfg.audit_only,
                **split_provenance,
                "split_counts": dataset_dict.get("split_counts"),
                "conditioning_contract": cfg.conditioning_contract,
                "broadcast_actual_treatment": cfg.broadcast_actual_treatment,
                "outcome_positive_weight": cfg.outcome_positive_weight,
                "regimen_delivery_features": cfg.regimen_delivery_features,
                "regimen_delivery_feature_indices": cfg.regimen_delivery_feature_indices,
                "condition_feature_names": getattr(dataset, "cond_columns", []),
                "state_feature_names": getattr(dataset, "state_columns", []),
                "state_feature_indices": getattr(dataset, "state_feature_indices", []),
                "state_dim": int(getattr(dataset, "state_dim", 0)),
                "forecast_alignment": dataset_dict.get("metadata", {}).get(
                    "forecast_alignment"
                ),
                "recursive_state_generation": cfg.recursive_state_generation,
                "temporal_contract": dataset_dict.get("temporal_contract"),
                "cohort": dataset_dict.get("cohort"),
                "rt_feasibility": cfg.rt_feasibility,
                "rt_feasibility_activation_indices": cfg.rt_feasibility_activation_indices,
                "rt_feasibility_parameter_bounds": cfg.rt_feasibility_parameter_bounds,
                "causal_weighting": causal_checkpoint_metadata,
            },
            "Gx": Gx.state_dict(),
            "Gy": Gy.state_dict(),
            "Dx": Dx.state_dict(),
            "Dy": Dy.state_dict(),
            "outcome_model": (
                outcome_model.state_dict() if outcome_model is not None else None
            ),
            "training_history": training_history,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path
