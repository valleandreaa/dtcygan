"""Evaluate a trained DTCyGAN against fully synthetic temporal ground truth.

The evaluator is specific to the JSON contract produced by
``export.py``. Counterfactual truth is used only here,
after training, and evaluation defaults to the checkpoint's held-out patients.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from src.training import (
    Config,
    SyntheticSequenceDataset,
    build_clinical_generator,
    resolve_device,
)


def resolve_checkpoint(
    checkpoint: str | Path | None, checkpoint_dir: str | Path | None
) -> Path:
    """Resolve an explicit checkpoint or the newest checkpoint in a directory."""

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    if checkpoint_dir is None:
        raise ValueError("Provide --checkpoint or --checkpoint-dir")
    directory = Path(checkpoint_dir)
    candidates = sorted(directory.glob("dtcygan_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No dtcygan_*.pt checkpoints found in {directory}")
    return candidates[-1]


def _load_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("patients"), list) or not payload["patients"]:
        raise ValueError("Dataset must contain a non-empty patients list")
    return payload


def _load_external_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = _load_payload(path)
    result: dict[str, dict[str, Any]] = {}
    for patient in payload["patients"]:
        patient_id = str(patient.get("patient_id"))
        if not patient_id or patient_id == "None":
            raise ValueError("Every oracle-truth patient requires a patient_id")
        if patient_id in result:
            raise ValueError(f"Duplicate oracle-truth patient ID: {patient_id}")
        truth = patient.get("counterfactual_truth")
        if not isinstance(truth, list) or not truth:
            raise ValueError(f"Oracle truth is missing for {patient_id}")
        result[patient_id] = patient
    return result


def _load_bundle(
    checkpoint_path: Path,
    payload: dict[str, Any],
    desired_device: str | None,
) -> tuple[Config, dict[str, Any], dict[str, Any], SyntheticSequenceDataset, torch.nn.Module, torch.device]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"])
    metadata = checkpoint.get("metadata") or {}
    dataset = SyntheticSequenceDataset(
        payload,
        config.seq_len,
        spec_clinical=metadata.get("clinical_feature_spec"),
        spec_treatment=metadata.get("treatment_feature_spec"),
        broadcast_actual_treatment=config.broadcast_actual_treatment,
    )
    observed_dimensions = (dataset.clin_dim, dataset.treat_dim, dataset.cond_dim)
    stored_dimensions = tuple(
        int(metadata.get(name, -1)) for name in ("clin_dim", "treat_dim", "cond_dim")
    )
    if observed_dimensions != stored_dimensions:
        raise ValueError(
            f"Checkpoint/data dimension mismatch: checkpoint={stored_dimensions}, "
            f"dataset={observed_dimensions}"
        )
    if list(metadata.get("clinical_feature_names", [])) != dataset.clin_columns:
        raise ValueError("Checkpoint clinical feature order differs from the dataset")
    if list(metadata.get("treatment_feature_names", [])) != dataset.treat_columns:
        raise ValueError("Checkpoint treatment feature order differs from the dataset")

    device = resolve_device(desired_device or config.device)
    generator = build_clinical_generator(config, dataset).to(device)
    generator.load_state_dict(checkpoint["Gx"])
    generator.eval()
    return config, metadata, checkpoint, dataset, generator, device


def build_feature_contract(
    config: Config,
    dataset: SyntheticSequenceDataset,
) -> dict[str, Any]:
    """Describe the exact feature routing and tensor widths used by DTCyGAN."""

    multiplier = 2 if config.include_missingness_indicators else 1
    clinical = list(dataset.clin_columns)
    treatment = list(dataset.treat_columns)
    condition = list(dataset.cond_columns)
    temporal = [name for name in clinical if name.startswith("x_")]
    static = [name for name in clinical if name.startswith("static_")]
    return {
        "sequence_length": int(dataset.seq_len),
        "include_missingness_indicators": bool(
            config.include_missingness_indicators
        ),
        "features": {
            "clinical_temporal": temporal,
            "clinical_static_repeated_each_step": static,
            "clinical_other": [
                name for name in clinical if name not in {*temporal, *static}
            ],
            "treatment_domain_targets": treatment,
            "conditioning": condition,
        },
        "tensors": {
            "x_clin": ["batch", dataset.seq_len, dataset.clin_dim],
            "mask_clin": ["batch", dataset.seq_len, dataset.clin_dim],
            "x_treat": ["batch", dataset.seq_len, dataset.treat_dim],
            "mask_treat": ["batch", dataset.seq_len, dataset.treat_dim],
            "actual_treatment": ["batch", dataset.seq_len, dataset.cond_dim],
            "mask_actual": ["batch", dataset.seq_len, dataset.cond_dim],
        },
        "components": {
            "Gx": {
                "role": "clinical-to-treatment/outcome generator",
                "value_inputs": [*clinical, *condition],
                "value_inputs_by_tensor": {
                    "x_clin": clinical,
                    "actual_treatment": condition,
                },
                "mask_inputs": bool(config.include_missingness_indicators),
                "effective_input_channels_per_step": int(
                    (dataset.clin_dim + dataset.cond_dim) * multiplier
                ),
                "outputs": treatment,
                "output_channels_per_step": int(dataset.treat_dim),
                "recursively_generates_clinical_state": bool(
                    config.recursive_state_generation
                ),
                "generated_state_features": list(dataset.state_columns),
                "forecast_start": dataset.forecast_start,
            },
            "Gy": {
                "role": "treatment-domain reconstruction generator",
                "value_inputs": [
                    *[f"generated::{name}" for name in treatment],
                    *condition,
                ],
                "value_inputs_by_tensor": {
                    "generated_x_treat": treatment,
                    "actual_treatment": condition,
                },
                "mask_inputs": bool(config.include_missingness_indicators),
                "effective_input_channels_per_step": int(
                    (dataset.treat_dim + dataset.cond_dim) * multiplier
                ),
                "outputs": [f"reconstructed::{name}" for name in treatment],
                "output_channels_per_step": int(dataset.treat_dim),
                "does_not_reconstruct_clinical_features": True,
            },
            "Dx": {
                "role": "real-vs-counterfactual treatment-domain discriminator",
                "value_inputs": [*clinical, *treatment, *condition],
                "value_inputs_by_tensor": {
                    "x_clin": clinical,
                    "real_or_generated_x_treat": treatment,
                    "actual_or_counterfactual_treatment": condition,
                },
                "mask_inputs": bool(config.include_missingness_indicators),
                "effective_input_channels_per_step": int(
                    (dataset.clin_dim + dataset.treat_dim + dataset.cond_dim)
                    * multiplier
                ),
                "output": "one patient-sequence score",
            },
            "Dy": {
                "role": "real-vs-reconstructed treatment-domain discriminator",
                "value_inputs": [*clinical, *treatment, *condition],
                "value_inputs_by_tensor": {
                    "x_clin": clinical,
                    "real_or_reconstructed_x_treat": treatment,
                    "actual_treatment": condition,
                },
                "mask_inputs": bool(config.include_missingness_indicators),
                "effective_input_channels_per_step": int(
                    (dataset.clin_dim + dataset.treat_dim + dataset.cond_dim)
                    * multiplier
                ),
                "output": "one patient-sequence score",
            },
        },
        "loss_feature_routing": {
            "factual": treatment,
            "conditioning": [config.conditioning_feature]
            if config.conditioning_feature
            else [],
            "cycle": treatment,
            "identity": treatment,
            "adversarial": [*clinical, *treatment, *condition],
        },
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    residual = prediction - truth
    mse = float(np.mean(np.square(residual)))
    variance = float(np.var(truth))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": math.sqrt(mse),
        "nrmse": math.sqrt(mse) / max(float(np.std(truth)), 1e-8),
        "r2": float(1.0 - mse / variance) if variance > 0 else float("nan"),
        "correlation": _safe_correlation(truth, prediction),
        "wasserstein": float(wasserstein_distance(truth, prediction)),
        "truth_mean": float(np.mean(truth)),
        "prediction_mean": float(np.mean(prediction)),
        "truth_std": float(np.std(truth)),
        "prediction_std": float(np.std(prediction)),
    }


def _binary_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    clipped = np.clip(score, 0.0, 1.0)
    return {
        "accuracy": float(np.mean((clipped >= 0.5) == truth)),
        "brier": float(np.mean(np.square(clipped - truth))),
        "wasserstein": float(wasserstein_distance(truth.astype(float), score)),
        "prediction_out_of_range_fraction": float(np.mean((score < 0.0) | (score > 1.0))),
        "truth_rate": float(np.mean(truth)),
        "prediction_mean": float(np.mean(score)),
    }


def _batched(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def evaluate_split(
    name: str,
    indices: Sequence[int],
    dataset: SyntheticSequenceDataset,
    generator: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    truth_by_patient: Mapping[str, Mapping[str, Any]] | None = None,
    intervention_chunk_size: int = 1,
    evaluation_start: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate factual prediction and point-treatment potential outcomes."""

    if not indices:
        raise ValueError(f"The {name} split is empty")
    if intervention_chunk_size <= 0:
        raise ValueError("intervention_chunk_size must be positive")
    if not 0 <= evaluation_start < dataset.seq_len:
        raise ValueError("evaluation_start must be within the sequence")
    try:
        treatment_index = dataset.treat_columns.index("temporal_treatment")
        outcome_index = dataset.treat_columns.index("outcome_factual")
    except ValueError as exc:
        raise ValueError(
            "Expected temporal_treatment and outcome_factual treatment features"
        ) from exc

    time_local_condition = dataset.cond_columns == ["temporal_treatment"]
    regimen_indices = []
    if not time_local_condition:
        for timestamp in range(dataset.seq_len):
            expected = f"temporal_treatment_t{timestamp:03d}"
            if expected not in dataset.cond_columns:
                raise ValueError(f"Conditioning data is missing {expected}")
            regimen_indices.append(dataset.cond_columns.index(expected))

    accumulated: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "factual_treatment",
            "factual_outcome",
            "pred_factual_treatment",
            "pred_factual_outcome",
            "truth_y0",
            "truth_y1",
            "pred_y0",
            "pred_y1",
            "pred_treatment_a0",
            "pred_treatment_a1",
            "future_prediction_delta",
            "future_ite_delta",
        )
    }

    with torch.inference_mode():
        for batch_indices in _batched(list(indices), batch_size):
            samples = [dataset[index] for index in batch_indices]
            clinical = torch.stack([sample["x_clin"] for sample in samples]).to(device)
            clinical_mask = torch.stack([sample["mask_clin"] for sample in samples]).to(device)
            treatment = torch.stack([sample["x_treat"] for sample in samples]).to(device)
            treatment_mask = torch.stack([sample["mask_treat"] for sample in samples]).to(device)
            condition = torch.stack([sample["actual_treatment"] for sample in samples]).to(device)
            condition_mask = torch.stack([sample["mask_actual"] for sample in samples]).to(device)
            step_mask = (
                (clinical_mask.sum(dim=2) + treatment_mask.sum(dim=2)) > 0
            )
            lengths = step_mask.sum(dim=1).clamp(min=1).long()

            factual_prediction, _ = generator(
                clinical * clinical_mask,
                clinical_mask,
                condition * condition_mask,
                condition_mask,
                lengths,
            )
            accumulated["factual_treatment"].append(
                treatment[:, :, treatment_index].cpu().numpy()
            )
            accumulated["factual_outcome"].append(
                treatment[:, :, outcome_index].cpu().numpy()
            )
            accumulated["pred_factual_treatment"].append(
                factual_prediction[:, :, treatment_index].cpu().numpy()
            )
            accumulated["pred_factual_outcome"].append(
                factual_prediction[:, :, outcome_index].cpu().numpy()
            )

            batch_y0 = np.zeros((len(batch_indices), dataset.seq_len), dtype=float)
            batch_y1 = np.zeros_like(batch_y0)
            for row, dataset_index in enumerate(batch_indices):
                patient_id = dataset.patient_ids[dataset_index]
                if truth_by_patient is None:
                    truth = dataset.patients[dataset_index].get(
                        "counterfactual_truth", []
                    )
                else:
                    truth_patient = truth_by_patient.get(patient_id)
                    if truth_patient is None:
                        raise ValueError(f"External oracle truth lacks {patient_id}")
                    expected_split = dataset.patients[dataset_index].get("split")
                    if (
                        truth_patient.get("split") is not None
                        and truth_patient.get("split") != expected_split
                    ):
                        raise ValueError(
                            f"Oracle split mismatch for patient {patient_id}"
                        )
                    truth = truth_patient.get("counterfactual_truth", [])
                truth_by_time = {int(record["timestep"]): record for record in truth}
                for timestamp in range(dataset.seq_len):
                    if timestamp not in truth_by_time:
                        raise ValueError(
                            f"Patient {dataset.patient_ids[dataset_index]} lacks truth at t={timestamp}"
                        )
                    batch_y0[row, timestamp] = float(truth_by_time[timestamp]["y0"])
                    batch_y1[row, timestamp] = float(truth_by_time[timestamp]["y1"])
            accumulated["truth_y0"].append(batch_y0)
            accumulated["truth_y1"].append(batch_y1)

            predicted_y0 = np.zeros_like(batch_y0)
            predicted_y1 = np.zeros_like(batch_y1)
            predicted_treatment_a0 = np.zeros_like(batch_y0)
            predicted_treatment_a1 = np.zeros_like(batch_y1)
            future_prediction_delta = np.zeros_like(batch_y0)
            future_ite_delta = np.zeros_like(batch_y0)
            for chunk_start in range(
                0, dataset.seq_len, intervention_chunk_size
            ):
                timestamps = list(
                    range(
                        chunk_start,
                        min(
                            dataset.seq_len,
                            chunk_start + intervention_chunk_size,
                        ),
                    )
                )
                variants = len(timestamps)
                # Variant-major layout allows several point interventions to
                # share one generator call without changing sequence order.
                base_condition = condition.unsqueeze(0).repeat(variants, 1, 1, 1)
                condition_a0 = base_condition.clone()
                condition_a1 = base_condition.clone()
                for variant, timestamp in enumerate(timestamps):
                    if time_local_condition:
                        condition_a0[variant, :, timestamp, 0] = 0.0
                        condition_a1[variant, :, timestamp, 0] = 1.0
                    else:
                        condition_index = regimen_indices[timestamp]
                        condition_a0[variant, :, :, condition_index] = 0.0
                        condition_a1[variant, :, :, condition_index] = 1.0

                expanded_clinical = clinical.unsqueeze(0).expand(
                    variants, -1, -1, -1
                )
                expanded_clinical_mask = clinical_mask.unsqueeze(0).expand(
                    variants, -1, -1, -1
                )
                expanded_condition_mask = condition_mask.unsqueeze(0).expand(
                    variants, -1, -1, -1
                )
                expanded_lengths = lengths.unsqueeze(0).expand(variants, -1)
                flat_batch = variants * len(batch_indices)

                def generate_for(condition_variants: torch.Tensor) -> torch.Tensor:
                    output, _ = generator(
                        (expanded_clinical * expanded_clinical_mask).reshape(
                            flat_batch, dataset.seq_len, dataset.clin_dim
                        ),
                        expanded_clinical_mask.reshape(
                            flat_batch, dataset.seq_len, dataset.clin_dim
                        ),
                        (condition_variants * expanded_condition_mask).reshape(
                            flat_batch, dataset.seq_len, dataset.cond_dim
                        ),
                        expanded_condition_mask.reshape(
                            flat_batch, dataset.seq_len, dataset.cond_dim
                        ),
                        expanded_lengths.reshape(flat_batch),
                    )
                    return output.reshape(
                        variants,
                        len(batch_indices),
                        dataset.seq_len,
                        dataset.treat_dim,
                    )

                output_a0 = generate_for(condition_a0)
                output_a1 = generate_for(condition_a1)
                for variant, timestamp in enumerate(timestamps):
                    predicted_y0[:, timestamp] = (
                        output_a0[variant, :, timestamp, outcome_index]
                        .cpu()
                        .numpy()
                    )
                    predicted_y1[:, timestamp] = (
                        output_a1[variant, :, timestamp, outcome_index]
                        .cpu()
                        .numpy()
                    )
                    predicted_treatment_a0[:, timestamp] = (
                        output_a0[variant, :, timestamp, treatment_index]
                        .cpu()
                        .numpy()
                    )
                    predicted_treatment_a1[:, timestamp] = (
                        output_a1[variant, :, timestamp, treatment_index]
                        .cpu()
                        .numpy()
                    )

                future_condition = base_condition.clone()
                future_a0 = condition_a0.clone()
                future_a1 = condition_a1.clone()
                has_future = False
                for variant, timestamp in enumerate(timestamps):
                    if timestamp + 1 >= dataset.seq_len:
                        continue
                    has_future = True
                    if time_local_condition:
                        for tensor in (future_condition, future_a0, future_a1):
                            tensor[variant, :, timestamp + 1 :, 0] = (
                                1.0 - tensor[variant, :, timestamp + 1 :, 0]
                            )
                    else:
                        for tensor in (future_condition, future_a0, future_a1):
                            tensor[variant, :, :, timestamp + 1 :] = (
                                1.0 - tensor[variant, :, :, timestamp + 1 :]
                            )
                if has_future:
                    changed_factual = generate_for(future_condition)
                    changed_a0 = generate_for(future_a0)
                    changed_a1 = generate_for(future_a1)
                    for variant, timestamp in enumerate(timestamps):
                        if timestamp + 1 >= dataset.seq_len:
                            continue
                        if time_local_condition:
                            factual_assignment = condition[:, timestamp, 0]
                        else:
                            factual_assignment = condition[
                                :, 0, regimen_indices[timestamp]
                            ]
                        factual_branch = torch.where(
                            factual_assignment >= 0.5,
                            output_a1[variant, :, timestamp, outcome_index],
                            output_a0[variant, :, timestamp, outcome_index],
                        )
                        future_prediction_delta[:, timestamp] = (
                            changed_factual[variant, :, timestamp, outcome_index]
                            - factual_branch
                        ).abs().cpu().numpy()
                        future_ite_delta[:, timestamp] = (
                            (
                                changed_a1[variant, :, timestamp, outcome_index]
                                - changed_a0[variant, :, timestamp, outcome_index]
                            )
                            - (
                                output_a1[variant, :, timestamp, outcome_index]
                                - output_a0[variant, :, timestamp, outcome_index]
                            )
                        ).abs().cpu().numpy()
            accumulated["pred_y0"].append(predicted_y0)
            accumulated["pred_y1"].append(predicted_y1)
            accumulated["pred_treatment_a0"].append(predicted_treatment_a0)
            accumulated["pred_treatment_a1"].append(predicted_treatment_a1)
            accumulated["future_prediction_delta"].append(future_prediction_delta)
            accumulated["future_ite_delta"].append(future_ite_delta)

    arrays = {key: np.concatenate(parts, axis=0) for key, parts in accumulated.items()}
    arrays = {key: value[:, evaluation_start:] for key, value in arrays.items()}
    truth_ite = arrays["truth_y1"] - arrays["truth_y0"]
    predicted_ite = arrays["pred_y1"] - arrays["pred_y0"]
    factual_metrics = _regression_metrics(
        arrays["factual_outcome"], arrays["pred_factual_outcome"]
    )
    potential_metrics = _regression_metrics(
        np.concatenate((arrays["truth_y0"].ravel(), arrays["truth_y1"].ravel())),
        np.concatenate((arrays["pred_y0"].ravel(), arrays["pred_y1"].ravel())),
    )
    treatment_metrics = _binary_metrics(
        arrays["factual_treatment"], arrays["pred_factual_treatment"]
    )
    pehe = float(np.sqrt(np.mean(np.square(predicted_ite - truth_ite))))
    causal_metrics = {
        "pehe": pehe,
        "normalized_pehe": pehe / max(float(np.std(truth_ite)), 1e-8),
        "true_ate": float(np.mean(truth_ite)),
        "predicted_ate": float(np.mean(predicted_ite)),
        "ate_absolute_error": float(abs(np.mean(predicted_ite) - np.mean(truth_ite))),
        "ite_correlation": _safe_correlation(truth_ite, predicted_ite),
        "effect_sign_accuracy": float(
            np.mean(np.sign(predicted_ite) == np.sign(truth_ite))
        ),
        "mean_absolute_conditioning_response": float(np.mean(np.abs(predicted_ite))),
    }
    conditioning_metrics = {
        "a0_adherence": float(np.mean(arrays["pred_treatment_a0"] < 0.5)),
        "a1_adherence": float(np.mean(arrays["pred_treatment_a1"] >= 0.5)),
        "balanced_adherence": float(
            0.5
            * (
                np.mean(arrays["pred_treatment_a0"] < 0.5)
                + np.mean(arrays["pred_treatment_a1"] >= 0.5)
            )
        ),
        "mean_treatment_separation": float(
            np.mean(arrays["pred_treatment_a1"] - arrays["pred_treatment_a0"])
        ),
    }
    temporal_metrics = {
        "truth_outcome_lag1_correlation": _safe_correlation(
            arrays["factual_outcome"][:, :-1], arrays["factual_outcome"][:, 1:]
        ),
        "predicted_outcome_lag1_correlation": _safe_correlation(
            arrays["pred_factual_outcome"][:, :-1],
            arrays["pred_factual_outcome"][:, 1:],
        ),
    }
    temporal_metrics["lag1_correlation_absolute_error"] = abs(
        temporal_metrics["predicted_outcome_lag1_correlation"]
        - temporal_metrics["truth_outcome_lag1_correlation"]
    )
    future_prediction = arrays["future_prediction_delta"][:, :-1]
    future_ite = arrays["future_ite_delta"][:, :-1]
    leakage_metrics = {
        "future_treatment_prediction_mae": float(np.mean(future_prediction)),
        "future_treatment_prediction_max": float(np.max(future_prediction)),
        "future_treatment_ite_mae": float(np.mean(future_ite)),
        "future_treatment_ite_max": float(np.max(future_ite)),
        "threshold": 1.0e-7,
    }
    leakage_metrics["passes_no_anticipation"] = bool(
        leakage_metrics["future_treatment_prediction_mae"]
        <= leakage_metrics["threshold"]
        and leakage_metrics["future_treatment_ite_mae"]
        <= leakage_metrics["threshold"]
    )

    timestep_rows: list[dict[str, Any]] = []
    for offset, timestamp in enumerate(range(evaluation_start, dataset.seq_len)):
        factual_at_t = _regression_metrics(
            arrays["factual_outcome"][:, offset],
            arrays["pred_factual_outcome"][:, offset],
        )
        truth_ite_t = truth_ite[:, offset]
        predicted_ite_t = predicted_ite[:, offset]
        timestep_rows.append(
            {
                "split": name,
                "timestep": timestamp,
                "factual_outcome_rmse": factual_at_t["rmse"],
                "factual_outcome_r2": factual_at_t["r2"],
                "pehe": float(
                    np.sqrt(np.mean(np.square(predicted_ite_t - truth_ite_t)))
                ),
                "true_ate": float(np.mean(truth_ite_t)),
                "predicted_ate": float(np.mean(predicted_ite_t)),
                "ate_absolute_error": float(
                    abs(np.mean(predicted_ite_t) - np.mean(truth_ite_t))
                ),
                "effect_sign_accuracy": float(
                    np.mean(np.sign(predicted_ite_t) == np.sign(truth_ite_t))
                ),
            }
        )

    result = {
        "n_patients": len(indices),
        "n_patient_timestamps": int(len(indices) * (dataset.seq_len - evaluation_start)),
        "factual_outcome": factual_metrics,
        "factual_treatment": treatment_metrics,
        "potential_outcomes": potential_metrics,
        "causal_effect": causal_metrics,
        "conditioning": conditioning_metrics,
        "temporal_fidelity": temporal_metrics,
        "leakage_audit": leakage_metrics,
    }
    arrays["truth_ite"] = truth_ite
    arrays["predicted_ite"] = predicted_ite
    return result, timestep_rows, arrays


def _history_summary(history: Any) -> dict[str, Any]:
    if not isinstance(history, list) or not history:
        return {"available": False, "epochs_recorded": 0}
    result: dict[str, Any] = {
        "available": True,
        "epochs_recorded": len(history),
        "first": history[0],
        "last": history[-1],
    }
    if len(history) >= 6:
        recent = history[-5:]
        previous = history[-6:-1]
        for metric in ("g_loss", "d_loss", "cycle_loss", "identity_loss"):
            recent_mean = float(np.mean([row["validation"][metric] for row in recent]))
            previous_mean = float(np.mean([row["validation"][metric] for row in previous]))
            result[f"recent_validation_{metric}_relative_change"] = (
                (recent_mean - previous_mean) / max(abs(previous_mean), 1e-8)
            )
    return result


def _recommendations(
    config: Config,
    split_metrics: Mapping[str, Mapping[str, Any]],
    history: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    validation = (
        split_metrics.get("test")
        or split_metrics.get("validation")
        or next(iter(split_metrics.values()))
    )
    recommendations: list[dict[str, Any]] = []
    recommended_overrides: dict[str, Any] = {}

    nrmse = float(validation["factual_outcome"]["nrmse"])
    pehe = float(validation["causal_effect"]["normalized_pehe"])
    adherence = float(validation["conditioning"]["balanced_adherence"])
    factual_accuracy = float(validation["factual_treatment"]["accuracy"])
    lag_error = float(validation["temporal_fidelity"]["lag1_correlation_absolute_error"])
    leakage_passes = bool(validation["leakage_audit"]["passes_no_anticipation"])

    if not leakage_passes:
        recommendations.append(
            {
                "severity": "high",
                "component": "future-treatment leakage",
                "evidence": (
                    "future-treatment prediction MAE="
                    f"{validation['leakage_audit']['future_treatment_prediction_mae']:.6g}, "
                    "ITE MAE="
                    f"{validation['leakage_audit']['future_treatment_ite_mae']:.6g}"
                ),
                "action": "Do not make a causal claim until the no-anticipation gate passes.",
            }
        )

    if nrmse > 0.75:
        recommendations.append(
            {
                "severity": "high",
                "component": "factual outcome generator",
                "evidence": f"validation outcome NRMSE={nrmse:.3f} (>0.75)",
                "action": (
                    "Complete the configured training schedule and compare held-out NRMSE; "
                    "if the factual loss has plateaued, tune factual_weight on validation "
                    "patients before opening the test result."
                ),
            }
        )
        recommended_overrides["epochs"] = max(100, int(config.epochs))
    if pehe > 0.50:
        recommendations.append(
            {
                "severity": "high",
                "component": "causal effect recovery",
                "evidence": f"validation normalized PEHE={pehe:.3f} (>0.50)",
                "action": (
                    "Do not claim counterfactual accuracy from this checkpoint. Complete "
                    "training and, if validation PEHE remains high, add time-varying "
                    "propensity adjustment without using test truth for tuning."
                ),
            }
        )
    if adherence < 0.80:
        recommendations.append(
            {
                "severity": "high",
                "component": "treatment conditioning",
                "evidence": f"balanced intervention adherence={adherence:.3f} (<0.80)",
                "action": (
                    "Complete training with the configured conditioning-consistency loss; "
                    "if validation adherence remains low, increase condition_weight before "
                    "evaluating the untouched test split."
                ),
            }
        )
    if factual_accuracy < 0.80:
        recommendations.append(
            {
                "severity": "medium",
                "component": "treatment generator",
                "evidence": f"factual treatment accuracy={factual_accuracy:.3f} (<0.80)",
                "action": "Add a binary treatment reconstruction term and tune its weight on validation data.",
            }
        )
    if lag_error > 0.20:
        recommendations.append(
            {
                "severity": "medium",
                "component": "temporal dynamics",
                "evidence": f"lag-1 correlation absolute error={lag_error:.3f} (>0.20)",
                "action": (
                    "Compare 2 versus 3 LSTM layers and hidden sizes 128/256/384; choose by "
                    "validation temporal error and PEHE, not training loss."
                ),
            }
        )

    train = split_metrics.get("train")
    if train:
        generalization_gap = nrmse - float(train["factual_outcome"]["nrmse"])
        if generalization_gap > 0.15:
            recommendations.append(
                {
                    "severity": "medium",
                    "component": "generalization",
                    "evidence": f"train-to-validation NRMSE gap={generalization_gap:.3f}",
                    "action": "Reduce generator_hidden to 128 and layers to 2, or add dropout before extending training.",
                }
            )
            recommended_overrides.update({"generator_hidden": 128, "layers": 2})

    if not history.get("available"):
        recommendations.append(
            {
                "severity": "information",
                "component": "training provenance",
                "evidence": "checkpoint predates persisted training_history",
                "action": "Future checkpoints now store every train/validation epoch for plateau and GAN-balance diagnosis.",
            }
        )
    else:
        last_validation = history["last"]["validation"]
        discriminator_loss = float(last_validation["d_loss"])
        if discriminator_loss < 0.15:
            recommendations.append(
                {
                    "severity": "medium",
                    "component": "GAN balance",
                    "evidence": f"final validation discriminator loss={discriminator_loss:.3f}",
                    "action": "Try discriminator_lr=2.5e-5 while retaining generator_lr=4e-5; the discriminator appears too strong.",
                }
            )
            recommended_overrides["discriminator_lr"] = 2.5e-5
        elif discriminator_loss > 0.75:
            recommendations.append(
                {
                    "severity": "medium",
                    "component": "GAN balance",
                    "evidence": f"final validation discriminator loss={discriminator_loss:.3f}",
                    "action": "Try discriminator_lr=1e-4; the discriminator appears too weak under LSGAN.",
                }
            )
            recommended_overrides["discriminator_lr"] = 1.0e-4

    high_count = sum(row["severity"] == "high" for row in recommendations)
    medium_count = sum(row["severity"] == "medium" for row in recommendations)
    status = (
        "fail"
        if not leakage_passes or high_count >= 2
        else "needs_improvement"
        if high_count or medium_count
        else "pass"
    )
    grid = {
        "selection_order": [
            "test.leakage_audit.passes_no_anticipation",
            "test.normalized_pehe",
            "test.factual_outcome.nrmse",
            "test.temporal_fidelity.lag1_correlation_absolute_error",
        ],
        "parameters": {
            "epochs": sorted({int(config.epochs), 100}),
            "generator_hidden": [128, 256, 384],
            "discriminator_hidden": [64, 128],
            "layers": [2, 3],
            "generator_lr": [2.0e-5, 4.0e-5, 8.0e-5],
            "discriminator_lr": [2.5e-5, 5.0e-5, 1.0e-4],
            "cycle_weight": [10.0, 20.0],
            "identity_weight": [10.0, 30.0],
            "seed": [42, 43, 44],
        },
        "recommended_overrides": recommended_overrides,
        "warning": "Change one parameter family at a time and rank only on held-out patients.",
    }
    return status, recommendations, grid


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    timestep_rows: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    try:
        import matplotlib

        # This evaluator writes files only; an interactive GUI backend (for
        # example TkAgg) is unnecessary and can fail in headless sessions.
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[str] = []
    rng = np.random.default_rng(42)
    factual_truth = arrays["factual_outcome"].ravel()
    factual_prediction = arrays["pred_factual_outcome"].ravel()
    ite_truth = arrays["truth_ite"].ravel()
    ite_prediction = arrays["predicted_ite"].ravel()
    selected = rng.choice(
        len(factual_truth), size=min(3000, len(factual_truth)), replace=False
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(factual_truth[selected], factual_prediction[selected], s=5, alpha=0.25)
    axes[0].set(title="Factual outcome", xlabel="Ground truth", ylabel="Prediction")
    axes[1].scatter(ite_truth[selected], ite_prediction[selected], s=5, alpha=0.25)
    axes[1].set(title="Conditional ITE", xlabel="Ground truth", ylabel="Prediction")
    figure.tight_layout()
    path = output_dir / "prediction_diagnostics.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(str(path))

    primary_split = "test" if any(row["split"] == "test" for row in timestep_rows) else "validation"
    validation_rows = [row for row in timestep_rows if row["split"] == primary_split]
    if validation_rows:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        times = [row["timestep"] for row in validation_rows]
        axis.plot(times, [row["factual_outcome_rmse"] for row in validation_rows], marker="o", label="Factual RMSE")
        axis.plot(times, [row["pehe"] for row in validation_rows], marker="o", label="PEHE")
        axis.set(xlabel="Timestamp", ylabel="Error", title="Held-out error by timestamp")
        axis.legend()
        figure.tight_layout()
        path = output_dir / "metrics_by_timestep.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(str(path))

    if history:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        epochs = [row["epoch"] for row in history]
        for metric in ("g_loss", "d_loss", "cycle_loss", "identity_loss"):
            axis.plot(
                epochs,
                [row["validation"][metric] for row in history],
                label=metric,
            )
        axis.set(xlabel="Epoch", ylabel="Validation loss", title="Training history")
        axis.legend()
        figure.tight_layout()
        path = output_dir / "training_history.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(str(path))
    return written


def _markdown_summary(report: Mapping[str, Any]) -> str:
    validation = (
        report["splits"].get("test")
        or report["splits"].get("validation")
        or next(iter(report["splits"].values()))
    )
    primary_name = "test" if "test" in report["splits"] else "validation"
    lines = [
        "# DTCyGAN temporal synthetic post-training report",
        "",
        f"Overall assessment: **{report['assessment'].upper()}**",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        f"Primary split: {primary_name} ({validation['n_patients']} patients)",
        "",
        "## Primary held-out metrics",
        "",
        f"- Factual outcome RMSE: {validation['factual_outcome']['rmse']:.4f}",
        f"- Factual outcome normalized RMSE: {validation['factual_outcome']['nrmse']:.4f}",
        f"- Factual outcome R²: {validation['factual_outcome']['r2']:.4f}",
        f"- Factual treatment accuracy: {validation['factual_treatment']['accuracy']:.4f}",
        f"- PEHE: {validation['causal_effect']['pehe']:.4f}",
        f"- Normalized PEHE: {validation['causal_effect']['normalized_pehe']:.4f}",
        f"- ATE absolute error: {validation['causal_effect']['ate_absolute_error']:.4f}",
        f"- Effect-sign accuracy: {validation['causal_effect']['effect_sign_accuracy']:.4f}",
        f"- Balanced conditioning adherence: {validation['conditioning']['balanced_adherence']:.4f}",
        f"- Lag-1 correlation error: {validation['temporal_fidelity']['lag1_correlation_absolute_error']:.4f}",
        f"- Future-treatment prediction MAE: {validation['leakage_audit']['future_treatment_prediction_mae']:.8f}",
        f"- Future-treatment ITE MAE: {validation['leakage_audit']['future_treatment_ite_mae']:.8f}",
        f"- No-anticipation gate: {validation['leakage_audit']['passes_no_anticipation']}",
        "",
        "## Recommendations",
        "",
    ]
    for row in report["recommendations"]:
        lines.append(
            f"- **{row['severity']} — {row['component']}**: {row['evidence']}. {row['action']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Potential outcomes are conditional one-step interventions at each observed history; they are not full alternative dynamic-regime trajectories. Counterfactual truth is evaluation-only and is never supplied to DTCyGAN training.",
            "",
        ]
    )
    return "\n".join(lines)


def run_postprocess(
    dataset_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    batch_size: int = 128,
    max_patients: int | None = None,
    device: str | None = None,
    truth_path: str | Path | None = None,
    evaluate_test: bool = False,
    intervention_chunk_size: int = 1,
    evaluation_start: int = 0,
) -> dict[str, Any]:
    resolved_checkpoint = resolve_checkpoint(checkpoint_path, checkpoint_dir)
    payload = _load_payload(dataset_path)
    config, metadata, checkpoint, dataset, generator, target_device = _load_bundle(
        resolved_checkpoint, payload, device
    )
    truth_by_patient = (
        _load_external_truth(truth_path) if truth_path is not None else None
    )
    embedded_truth_count = sum(
        "counterfactual_truth" in patient for patient in payload["patients"]
    )
    if config.require_explicit_splits:
        if truth_by_patient is None:
            raise ValueError("Leakage-safe evaluation requires --truth")
        if embedded_truth_count:
            raise ValueError(
                "Leakage-safe training dataset must not embed counterfactual truth"
            )
    validation_ids = {
        str(value) for value in metadata.get("validation_patient_ids", [])
    }
    test_ids = {str(value) for value in metadata.get("test_patient_ids", [])}
    has_explicit_splits = all(
        patient.get("split") in {"train", "validation", "test"}
        for patient in dataset.patients
    )
    if has_explicit_splits:
        train_indices = [
            index
            for index, patient in enumerate(dataset.patients)
            if patient["split"] == "train"
        ]
        validation_indices = [
            index
            for index, patient in enumerate(dataset.patients)
            if patient["split"] == "validation"
        ]
        test_indices = [
            index
            for index, patient in enumerate(dataset.patients)
            if patient["split"] == "test"
        ]
        observed_validation_ids = {dataset.patient_ids[index] for index in validation_indices}
        observed_test_ids = {dataset.patient_ids[index] for index in test_indices}
        if observed_validation_ids != validation_ids:
            raise ValueError("Checkpoint validation IDs do not match explicit dataset split")
        if observed_test_ids != test_ids:
            raise ValueError("Checkpoint test IDs do not match explicit dataset split")
    else:
        validation_indices = [
            index
            for index, patient_id in enumerate(dataset.patient_ids)
            if patient_id in validation_ids
        ]
        train_indices = [
            index
            for index, patient_id in enumerate(dataset.patient_ids)
            if patient_id not in validation_ids
        ]
        test_indices = []
    if not validation_indices:
        raise ValueError("Checkpoint contains no validation IDs matching the dataset")
    if config.require_explicit_splits and not test_indices:
        raise ValueError("Leakage-safe evaluation requires an untouched test split")
    if truth_by_patient is not None:
        dataset_ids = set(dataset.patient_ids)
        if set(truth_by_patient) != dataset_ids:
            raise ValueError("Oracle truth IDs do not exactly match the training dataset")
    if max_patients is not None:
        if max_patients <= 0:
            raise ValueError("max_patients must be positive")
        train_indices = train_indices[:max_patients]
        validation_indices = validation_indices[:max_patients]
        test_indices = test_indices[:max_patients]

    split_metrics: dict[str, Any] = {}
    timestep_rows: list[dict[str, Any]] = []
    primary_arrays: dict[str, np.ndarray] | None = None
    split_definitions = [
        ("train", train_indices),
        ("validation", validation_indices),
    ]
    if test_indices and evaluate_test:
        split_definitions.append(("test", test_indices))
    primary_name = "test" if test_indices and evaluate_test else "validation"
    for split_name, indices in split_definitions:
        metrics, rows, arrays = evaluate_split(
            split_name,
            indices,
            dataset,
            generator,
            target_device,
            batch_size=batch_size,
            truth_by_patient=truth_by_patient,
            intervention_chunk_size=intervention_chunk_size,
            evaluation_start=evaluation_start,
        )
        split_metrics[split_name] = metrics
        timestep_rows.extend(rows)
        if split_name == primary_name:
            primary_arrays = arrays

    history_raw = checkpoint.get("training_history")
    history = _history_summary(history_raw)
    assessment, recommendations, grid = _recommendations(
        config, split_metrics, history
    )
    parameter_count = int(sum(parameter.numel() for parameter in generator.parameters()))
    report: dict[str, Any] = {
        "assessment": assessment,
        "checkpoint": str(resolved_checkpoint.resolve()),
        "dataset": str(Path(dataset_path).resolve()),
        "oracle_truth": str(Path(truth_path).resolve()) if truth_path else None,
        "device": str(target_device),
        "model": {
            "generator_parameters": parameter_count,
            "config": dict(config.__dict__),
            "dimensions": {
                "clinical": dataset.clin_dim,
                "treatment": dataset.treat_dim,
                "conditioning": dataset.cond_dim,
                "sequence_length": dataset.seq_len,
            },
            "feature_contract": build_feature_contract(config, dataset),
        },
        "training_history": history,
        "splits": split_metrics,
        "recommendations": recommendations,
        "methodology": {
            "primary_split": primary_name,
            "untouched_test_for_primary_conclusions": bool(
                test_indices and evaluate_test
            ),
            "counterfactual_truth_used_during_training": False,
            "training_dataset_truth_field_count": embedded_truth_count,
            "patient_split_overlap": 0,
            "estimand": "conditional point-treatment effect Y_t(1)-Y_t(0) at observed history",
            "intervention_chunk_size": int(intervention_chunk_size),
            "evaluation_start": int(evaluation_start),
            "forecast_alignment": payload.get("metadata", {}).get(
                "forecast_alignment"
            ),
            "causal_weighting": metadata.get("causal_weighting", {}),
        },
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with (destination / "recommended_config.json").open("w", encoding="utf-8") as handle:
        json.dump(grid, handle, indent=2, allow_nan=False)
    with (destination / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(_markdown_summary(report))
    split_rows = []
    for split_name, metrics in split_metrics.items():
        split_rows.append(
            {
                "split": split_name,
                "n_patients": metrics["n_patients"],
                "factual_outcome_rmse": metrics["factual_outcome"]["rmse"],
                "factual_outcome_nrmse": metrics["factual_outcome"]["nrmse"],
                "factual_outcome_r2": metrics["factual_outcome"]["r2"],
                "factual_treatment_accuracy": metrics["factual_treatment"]["accuracy"],
                "pehe": metrics["causal_effect"]["pehe"],
                "normalized_pehe": metrics["causal_effect"]["normalized_pehe"],
                "ate_absolute_error": metrics["causal_effect"]["ate_absolute_error"],
                "effect_sign_accuracy": metrics["causal_effect"]["effect_sign_accuracy"],
                "conditioning_adherence": metrics["conditioning"]["balanced_adherence"],
                "temporal_lag1_error": metrics["temporal_fidelity"]["lag1_correlation_absolute_error"],
                "future_treatment_prediction_mae": metrics["leakage_audit"]["future_treatment_prediction_mae"],
                "future_treatment_ite_mae": metrics["leakage_audit"]["future_treatment_ite_mae"],
                "passes_no_anticipation": metrics["leakage_audit"]["passes_no_anticipation"],
            }
        )
    _write_csv(destination / "metrics_by_split.csv", split_rows)
    _write_csv(destination / "metrics_by_timestep.csv", timestep_rows)
    feature_rows = []
    for split_name, metrics in split_metrics.items():
        outcome = metrics["factual_outcome"]
        treatment = metrics["factual_treatment"]
        potential = metrics["potential_outcomes"]
        feature_rows.extend(
            [
                {
                    "split": split_name,
                    "feature": "outcome_factual",
                    "wasserstein": outcome["wasserstein"],
                    "truth_mean": outcome["truth_mean"],
                    "prediction_mean": outcome["prediction_mean"],
                    "truth_std": outcome["truth_std"],
                    "prediction_std": outcome["prediction_std"],
                },
                {
                    "split": split_name,
                    "feature": "temporal_treatment",
                    "wasserstein": treatment["wasserstein"],
                    "truth_mean": treatment["truth_rate"],
                    "prediction_mean": treatment["prediction_mean"],
                    "truth_std": float("nan"),
                    "prediction_std": float("nan"),
                },
                {
                    "split": split_name,
                    "feature": "potential_outcomes_y0_y1",
                    "wasserstein": potential["wasserstein"],
                    "truth_mean": potential["truth_mean"],
                    "prediction_mean": potential["prediction_mean"],
                    "truth_std": potential["truth_std"],
                    "prediction_std": potential["prediction_std"],
                },
            ]
        )
    _write_csv(destination / "feature_fidelity.csv", feature_rows)
    if primary_arrays is not None:
        report["plots"] = _write_plots(
            destination,
            primary_arrays,
            timestep_rows,
            history_raw if isinstance(history_raw, list) else None,
        )
        with (destination / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    return report


