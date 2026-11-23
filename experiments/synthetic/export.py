"""Export the fully synthetic temporal T4 benchmark for DTCyGAN training."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TREATMENT_FEATURES = ("temporal_treatment", "outcome_factual")


def _json_yaml_compatible(value: Any, *, indent: int | None = None) -> str:
    serialized = json.dumps(value, indent=indent, allow_nan=False)
    return re.sub(r"(?<![\w.])(-?\d+)([eE][+-]\d+)", r"\1.0\2", serialized)


def _finite_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{field} must have {ndim} dimensions; got {result.shape}")
    result = result.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} contains a non-finite value")
    return result


def _explicit_split_labels(
    n_patients: int,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> list[str]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train and validation fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Explicit splits must leave a non-empty test fraction")
    if n_patients < 3:
        raise ValueError("Explicit train/validation/test splitting needs at least 3 patients")
    counts = {
        "train": int(n_patients * train_fraction),
        "validation": int(n_patients * validation_fraction),
    }
    counts["test"] = n_patients - counts["train"] - counts["validation"]
    if min(counts.values()) <= 0:
        raise ValueError("Every explicit split must contain at least one patient")
    labels = np.empty(n_patients, dtype=object)
    permutation = np.random.default_rng(seed).permutation(n_patients)
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    labels[permutation[:train_end]] = "train"
    labels[permutation[train_end:validation_end]] = "validation"
    labels[permutation[validation_end:]] = "test"
    return [str(value) for value in labels]


def build_dtcygan_dataset(
    data: Mapping[Any, Mapping[str, Any]],
    timesteps: int = 8,
    conditioning_contract: str = "time_local",
    split_seed: int | None = None,
    forecast_start: int | None = None,
) -> dict[str, Any]:
    """Convert fully synthetic arrays into DTCyGAN timestamped records."""

    if not data:
        raise ValueError("The fully synthetic pickle contains no patients")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if forecast_start is not None and not 1 <= forecast_start < timesteps:
        raise ValueError("forecast_start must lie strictly inside the sequence")

    patients: list[dict[str, Any]] = []
    temporal_names: list[str] | None = None
    static_names: list[str] | None = None
    split_labels = (
        _explicit_split_labels(len(data), seed=split_seed)
        if split_seed is not None
        else [None] * len(data)
    )

    for patient_index, datum in enumerate(data.values(), start=1):
        missing = [name for name in ("x", "x_static", "a", "y0", "y1") if name not in datum]
        if missing:
            raise ValueError(f"Synthetic patient {patient_index} is missing: {missing}")

        x = _finite_array(datum["x"], field="x", ndim=2)
        x_static = _finite_array(datum["x_static"], field="x_static", ndim=1)
        treatment_raw = _finite_array(datum["a"], field="a", ndim=1)
        treatment = treatment_raw.astype(int)
        if not np.array_equal(treatment_raw, treatment) or not np.isin(treatment, [0, 1]).all():
            raise ValueError("a must be a binary treatment sequence")
        y0 = _finite_array(datum["y0"], field="y0", ndim=1)
        y1 = _finite_array(datum["y1"], field="y1", ndim=1)

        available = x.shape[1]
        lengths = {available, len(treatment), len(y0), len(y1)}
        if len(lengths) != 1:
            raise ValueError(
                f"Synthetic patient {patient_index} has inconsistent lengths: {sorted(lengths)}"
            )
        length = min(available, timesteps)

        if temporal_names is None:
            temporal_names = [f"x_{index:02d}" for index in range(x.shape[0])]
            static_names = [f"static_{index:02d}" for index in range(len(x_static))]
        elif len(temporal_names) != x.shape[0] or len(static_names or []) != len(x_static):
            raise ValueError("All synthetic patients must have the same feature dimensions")

        propensity = datum.get("propensity")
        if propensity is not None:
            propensity = _finite_array(propensity, field="propensity", ndim=1)
            if len(propensity) != available:
                raise ValueError("propensity must have one value per timestamp")
        state_mask = datum.get("x_mask")
        if state_mask is not None:
            state_mask = _finite_array(state_mask, field="x_mask", ndim=2)
            if state_mask.shape != x.shape or not np.isin(state_mask, [0, 1]).all():
                raise ValueError("x_mask must be a binary array matching x")

        regime_y0 = datum.get("regime_y0")
        regime_y1 = datum.get("regime_y1")
        if (regime_y0 is None) != (regime_y1 is None):
            raise ValueError("regime_y0 and regime_y1 must be provided together")
        if regime_y0 is not None:
            regime_y0 = _finite_array(regime_y0, field="regime_y0", ndim=1)
            regime_y1 = _finite_array(regime_y1, field="regime_y1", ndim=1)
            if len(regime_y0) != available or len(regime_y1) != available:
                raise ValueError("regime outcomes must have one value per timestamp")

        patient_id = f"SYNTHETIC_{patient_index:06d}"
        static_record = {
            name: float(x_static[index])
            for index, name in enumerate(static_names or [])
        }
        clinical_records: list[dict[str, Any]] = []
        state_target_records: list[dict[str, Any]] = []
        treatment_records: list[dict[str, Any]] = []
        actual_treatment_records: list[dict[str, Any]] = []
        truth_records: list[dict[str, Any]] = []
        for timestamp in range(length):
            factual = y1[timestamp] if treatment[timestamp] else y0[timestamp]
            state_values = {
                name: (
                    float(x[index, timestamp])
                    if state_mask is None or state_mask[index, timestamp] == 1
                    else None
                )
                for index, name in enumerate(temporal_names or [])
            }
            clinical_records.append(
                {
                    **{
                        name: (
                            None
                            if forecast_start is not None
                            and timestamp >= forecast_start
                            else value
                        )
                        for name, value in state_values.items()
                    },
                    **static_record,
                    "_id": patient_id,
                    "timestep": timestamp,
                }
            )
            if forecast_start is not None:
                state_target_records.append(
                    {
                        **state_values,
                        "_id": patient_id,
                        "timestep": timestamp,
                    }
                )
            treatment_records.append(
                {
                    "temporal_treatment": int(treatment[timestamp]),
                    "outcome_factual": float(factual),
                    "_id": patient_id,
                    "timestep": timestamp,
                }
            )
            actual_treatment_records.append(
                {
                    "temporal_treatment": int(treatment[timestamp]),
                    "_id": patient_id,
                    "timestep": timestamp,
                }
            )
            truth = {
                "timestep": timestamp,
                "treatment": int(treatment[timestamp]),
                "y0": float(y0[timestamp]),
                "y1": float(y1[timestamp]),
                "ite": float(y1[timestamp] - y0[timestamp]),
            }
            if propensity is not None:
                truth["propensity"] = float(propensity[timestamp])
            if regime_y0 is not None:
                truth["regime_y0"] = float(regime_y0[timestamp])
                truth["regime_y1"] = float(regime_y1[timestamp])
                truth["regime_effect"] = float(
                    regime_y1[timestamp] - regime_y0[timestamp]
                )
            truth_records.append(truth)

        patient_record = {
                "patient_id": patient_id,
                "clinical": clinical_records,
                "treatment": treatment_records,
                "actual_treatment": actual_treatment_records,
                "counterfactual_truth": truth_records,
            }
        if forecast_start is not None:
            patient_record["state_target"] = state_target_records
        split_label = split_labels[patient_index - 1]
        if split_label is not None:
            patient_record["split"] = split_label
        patients.append(patient_record)

    assert temporal_names is not None and static_names is not None
    metadata = {
            "format": "dtcygan-temporal-v1",
            "source": "T4 fully synthetic longitudinal causal DGP",
            "n_patients": len(patients),
            "sequence_length": timesteps,
            "clinical_features": [*temporal_names, *static_names],
            "training_treatment_features": list(TREATMENT_FEATURES),
            "conditioning": "time_local",
            "truth_field": "patients[].counterfactual_truth",
            "truth_definition": "conditional ITE at each observed history: Y_t(1)-Y_t(0)",
            "truth_excluded_from_training": True,
            "state_target_features": list(temporal_names),
            "forecast_alignment": (
                {
                    "contract": "t4_baseline_forecast",
                    "baseline_timestamps": list(range(forecast_start)),
                    "evaluation_timestamps": list(range(forecast_start, timesteps)),
                    "baseline_end": forecast_start - 1,
                    "evaluation_start": forecast_start,
                    "evaluation_end": timesteps - 1,
                    "followup_temporal_covariates_available": False,
                    "preserved_static_features": list(static_names),
                    "withheld_temporal_features": list(temporal_names),
                    "state_targets_separate_from_inputs": True,
                }
                if forecast_start is not None
                else None
            ),
            "split_counts": {
                name: sum(patient.get("split") == name for patient in patients)
                for name in ("train", "validation", "test")
            }
            if split_seed is not None
            else None,
        }
    return {
        "metadata": metadata,
        "patients": patients,
    }


def build_dtcygan_config(
    clinical_features: Sequence[str],
    sequence_length: int,
    seed: int = 42,
    batch_size: int = 32,
    forecast_start: int | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if forecast_start is not None and not 1 <= forecast_start < sequence_length:
        raise ValueError("forecast_start must lie strictly inside the sequence")
    config = {
        "seq_len": int(sequence_length),
        "batch_size": int(batch_size),
        "epochs": 100,
        "generator_lr": 4.0e-5,
        "discriminator_lr": 5.0e-5,
        "beta1": 0.5,
        "beta2": 0.999,
        "cycle_weight": 20.0,
        "identity_weight": 30.0,
        "generator_hidden": 256,
        "discriminator_hidden": 128,
        "layers": 3,
        "seed": int(seed),
        "device": None,
        "adv_loss": "lsgan",
        "criterion": "l1",
        "counterfactual_mode": "temporal_step_flip",
        "factual_weight": 10.0,
        "condition_weight": 5.0,
        "conditioning_feature": "temporal_treatment",
        "conditioning_contract": "time_local",
        "broadcast_actual_treatment": False,
        "require_explicit_splits": True,
        "causal_weighting": {"enabled": False},
        "clinical_features": {
            "numeric": list(clinical_features),
            "one_hot": [],
            "ordinal": {},
        },
        "treatment_features": {
            "numeric": list(TREATMENT_FEATURES),
            "one_hot": [],
            "ordinal": {},
        },
    }
    if forecast_start is not None:
        config.update(
            {
                "recursive_state_generation": True,
                "lambda_state": 10.0,
                "state_teacher_forcing_ratio": 0.5,
                "temporal_alignment": {
                    "contract": "t4_baseline_forecast",
                    "baseline_end": int(forecast_start - 1),
                    "evaluation_start": int(forecast_start),
                    "evaluation_end": int(sequence_length - 1),
                    "followup_temporal_covariates_available": False,
                },
            }
        )
    return config


def export_files(
    input_path: str | Path,
    output_path: str | Path,
    config_output_path: str | Path,
    timesteps: int = 8,
    seed: int = 42,
    split_seed: int = 2027,
    truth_output_path: str | Path | None = None,
    batch_size: int = 32,
    forecast_start: int | None = None,
) -> dict[str, Any]:
    with Path(input_path).open("rb") as handle:
        data = pickle.load(handle)
    dataset = build_dtcygan_dataset(
        data,
        timesteps=timesteps,
        conditioning_contract="time_local",
        split_seed=split_seed,
        forecast_start=forecast_start,
    )

    truth_patients = []
    for patient in dataset["patients"]:
        truth_patients.append(
            {
                "patient_id": patient["patient_id"],
                "split": patient["split"],
                "counterfactual_truth": patient.pop("counterfactual_truth"),
            }
        )
    truth_payload = {
        "metadata": {
            "source": dataset["metadata"]["source"],
            "n_patients": dataset["metadata"]["n_patients"],
            "sequence_length": dataset["metadata"]["sequence_length"],
            "split_counts": dataset["metadata"]["split_counts"],
            "training_dataset_contains_truth": False,
        },
        "patients": truth_patients,
    }
    truth_output = Path(truth_output_path)
    truth_output.parent.mkdir(parents=True, exist_ok=True)
    with truth_output.open("w", encoding="utf-8") as handle:
        json.dump(truth_payload, handle, separators=(",", ":"), allow_nan=False)
    dataset["metadata"]["truth_field"] = None
    dataset["metadata"]["truth_definition"] = None
    dataset["metadata"]["truth_location"] = "external evaluation artifact"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, separators=(",", ":"), allow_nan=False)

    metadata = dataset["metadata"]
    config = build_dtcygan_config(
        metadata["clinical_features"],
        metadata["sequence_length"],
        seed=seed,
        batch_size=batch_size,
        forecast_start=forecast_start,
    )
    config_output = Path(config_output_path)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    with config_output.open("w", encoding="utf-8") as handle:
        handle.write(_json_yaml_compatible(config, indent=2))
    return metadata


