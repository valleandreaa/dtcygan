"""Export T4 semi-synthetic MIMIC sequences for unmodified DTCyGAN training.

The source pickle is produced by ``gen_synthetic_mimic.py``. Observed MIMIC
covariates, time-varying treatment assignments, and factual outcomes become
DTCyGAN training records. The known potential outcomes are retained in a
separate field that DTCyGAN's dataset loader ignores.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEMOGRAPHICS = ("agegroup", "heightgroup", "weightgroup", "gender")
TRAINING_TREATMENT_FEATURES = ("temporal_treatment", "outcome_factual")


def _json_yaml_compatible(value: Any, *, indent: int | None = None) -> str:
    """Serialize JSON numbers in a form that PyYAML also types as numeric."""

    serialized = json.dumps(value, indent=indent, allow_nan=False)
    # PyYAML 1.1 treats JSON's valid ``4e-05`` spelling as a string. Adding a
    # decimal point keeps valid JSON and makes yaml.safe_load return a float.
    return re.sub(
        r"(?<![\w.])(-?\d+)([eE][+-]\d+)",
        r"\1.0\2",
        serialized,
    )


def _finite_number(value: Any, *, field: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{field!r} must be scalar; observed shape {array.shape}")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{field!r} must be finite")
    return result


def _sequence(value: Any, *, field: str) -> list[float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not len(array):
        raise ValueError(f"{field!r} must contain at least one timestamp")
    if not np.isfinite(array).all():
        raise ValueError(f"{field!r} contains a non-finite value")
    return [float(item) for item in array]


def _potential_outcomes(
    treatment: Sequence[int],
    factual: Sequence[float],
    counterfactual: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Recover Y(0) and Y(1) from T4's factual/counterfactual storage."""

    y0: list[float] = []
    y1: list[float] = []
    for assigned, observed, alternative in zip(treatment, factual, counterfactual):
        if assigned == 0:
            y0.append(float(observed))
            y1.append(float(alternative))
        elif assigned == 1:
            y0.append(float(alternative))
            y1.append(float(observed))
        else:
            raise ValueError(f"Treatment must be binary, observed {assigned!r}")
    return y0, y1


def _explicit_split_labels(
    n_patients: int,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> list[str]:
    """Create deterministic, non-overlapping patient-level splits."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train and validation fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Explicit splits must leave a non-empty test fraction")
    if n_patients < 3:
        raise ValueError("Explicit train/validation/test splitting needs at least 3 patients")

    train_count = int(n_patients * train_fraction)
    validation_count = int(n_patients * validation_fraction)
    test_count = n_patients - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Every explicit split must contain at least one patient")

    labels = np.empty(n_patients, dtype=object)
    permutation = np.random.default_rng(seed).permutation(n_patients)
    validation_end = train_count + validation_count
    labels[permutation[:train_count]] = "train"
    labels[permutation[train_count:validation_end]] = "validation"
    labels[permutation[validation_end:]] = "test"
    return [str(label) for label in labels]


def build_dtcygan_dataset(
    data: Mapping[Any, Mapping[str, Any]],
    features: Sequence[str],
    timesteps: int | None = 8,
    demographics: Sequence[str] = DEMOGRAPHICS,
    conditioning_contract: str = "time_local",
    split_seed: int | None = None,
    external_truth: Mapping[str, Mapping[str, Any]] | None = None,
    split_assignments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Convert T4 MIMIC records to DTCyGAN's temporal patient JSON format."""

    if not data:
        raise ValueError("The T4 semi-synthetic pickle contains no patients")
    feature_names = [str(name) for name in features]
    demographic_names = [str(name) for name in demographics]
    if not feature_names:
        raise ValueError("header.pkl contains no temporal features")
    if len(set([*feature_names, *demographic_names])) != len(
        [*feature_names, *demographic_names]
    ):
        raise ValueError("Temporal feature and demographic names must be unique")
    if timesteps is not None and timesteps <= 0:
        raise ValueError("timesteps must be positive when provided")

    if split_assignments is not None:
        source_ids = [str(value) for value in data]
        if set(split_assignments) != set(source_ids):
            raise ValueError("Split manifest IDs do not exactly match the source pickle")
        split_labels = [str(split_assignments[source_id]) for source_id in source_ids]
        if set(split_labels) != {"train", "validation", "test"}:
            raise ValueError("Split manifest must contain train, validation, and test")
    else:
        split_labels = (
            _explicit_split_labels(len(data), seed=split_seed)
            if split_seed is not None
            else [None] * len(data)
        )

    patients: list[dict[str, Any]] = []
    exported_lengths: list[int] = []
    for patient_index, (source_patient_id, datum) in enumerate(data.items(), start=1):
        required = [
            *feature_names,
            *demographic_names,
            "treatment",
            "outcome",
        ]
        missing = [name for name in required if name not in datum]
        if missing:
            raise ValueError(f"T4 patient {patient_index} is missing fields: {missing}")

        treatment_raw = _sequence(datum["treatment"], field="treatment")
        treatment = [int(value) for value in treatment_raw]
        if any(value not in (0, 1) for value in treatment):
            raise ValueError("T4 treatment sequence must be binary")

        outcome = np.asarray(datum["outcome"], dtype=float)
        if outcome.ndim == 2 and outcome.shape[0] == 2:
            factual = _sequence(outcome[0], field="outcome factual")
            counterfactual = _sequence(outcome[1], field="outcome counterfactual")
            y0, y1 = _potential_outcomes(treatment, factual, counterfactual)
        elif outcome.ndim == 1 and external_truth is not None:
            source_key = str(source_patient_id)
            if source_key not in external_truth:
                raise ValueError(f"External truth is missing source patient {source_key}")
            factual = _sequence(outcome, field="outcome factual")
            y0 = _sequence(external_truth[source_key]["y0"], field="truth y0")
            y1 = _sequence(external_truth[source_key]["y1"], field="truth y1")
            counterfactual = [
                y0[index] if assigned == 1 else y1[index]
                for index, assigned in enumerate(treatment)
            ]
        else:
            raise ValueError(
                "T4 outcome must be legacy (2, timestamps), or factual-only with external_truth"
            )

        feature_sequences = {
            name: _sequence(datum[name], field=name) for name in feature_names
        }
        available_lengths = {
            len(treatment),
            len(factual),
            len(counterfactual),
            *(len(values) for values in feature_sequences.values()),
        }
        if len(available_lengths) != 1:
            raise ValueError(
                f"T4 patient {patient_index} has inconsistent temporal lengths: "
                f"{sorted(available_lengths)}"
            )
        available = available_lengths.pop()
        length = min(available, timesteps) if timesteps is not None else available
        exported_lengths.append(length)
        treatment = treatment[:length]
        factual = factual[:length]
        counterfactual = counterfactual[:length]
        y0 = y0[:length]
        y1 = y1[:length]

        patient_id = f"MIMIC_SS_{patient_index:06d}"
        static = {
            name: _finite_number(datum[name], field=name)
            for name in demographic_names
        }
        clinical_records: list[dict[str, Any]] = []
        treatment_records: list[dict[str, Any]] = []
        actual_treatment_records: list[dict[str, Any]] = []
        truth_records: list[dict[str, Any]] = []
        for timestamp in range(length):
            clinical_records.append(
                {
                    **{
                        name: feature_sequences[name][timestamp]
                        for name in feature_names
                    },
                    **static,
                    "_id": patient_id,
                    "timestep": timestamp,
                }
            )
            treatment_records.append(
                {
                    "temporal_treatment": treatment[timestamp],
                    "outcome_factual": factual[timestamp],
                    "_id": patient_id,
                    "timestep": timestamp,
                }
            )
            actual_treatment_records.append(
                {
                    "temporal_treatment": treatment[timestamp],
                    "_id": patient_id,
                    "timestep": timestamp,
                }
            )
            truth_records.append(
                {
                    "timestep": timestamp,
                    "treatment": treatment[timestamp],
                    "y0": y0[timestamp],
                    "y1": y1[timestamp],
                    "ite": y1[timestamp] - y0[timestamp],
                }
            )

        patient_record = {
            "patient_id": patient_id,
            "clinical": clinical_records,
            "treatment": treatment_records,
            "actual_treatment": actual_treatment_records,
            # Removed into a separate artifact in leakage-free export mode.
            "counterfactual_truth": truth_records,
        }
        split_label = split_labels[patient_index - 1]
        if split_label is not None:
            patient_record["split"] = split_label
        patients.append(patient_record)

    return {
        "metadata": {
            "format": "dtcygan-temporal-v1",
            "source": "T4 semi-synthetic MIMIC",
            "n_patients": len(patients),
            "sequence_length": max(exported_lengths),
            "min_sequence_length": min(exported_lengths),
            "requested_timesteps": timesteps,
            "clinical_features": [*feature_names, *demographic_names],
            "training_treatment_features": list(TRAINING_TREATMENT_FEATURES),
            "conditioning": "time_local",
            "truth_field": "patients[].counterfactual_truth",
            "truth_definition": "ite = Y(1) - Y(0)",
            "truth_excluded_from_training": True,
            "patient_identifiers": "sequential export identifiers; source IDs omitted",
            "split_counts": {
                name: sum(patient.get("split") == name for patient in patients)
                for name in ("train", "validation", "test")
            }
            if split_seed is not None or split_assignments is not None
            else None,
        },
        "patients": patients,
    }


def build_dtcygan_config(
    features: Sequence[str],
    sequence_length: int,
    demographics: Sequence[str] = DEMOGRAPHICS,
    seed: int = 42,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Return a DTCyGAN config specialized to the exported MIMIC features."""

    return {
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
            "numeric": [
                *[str(name) for name in features],
                *[str(name) for name in demographics],
            ],
            "one_hot": [],
            "ordinal": {},
        },
        "treatment_features": {
            "numeric": list(TRAINING_TREATMENT_FEATURES),
            "one_hot": [],
            "ordinal": {},
        },
    }


def export_files(
    input_path: str | Path,
    header_path: str | Path,
    output_path: str | Path,
    config_output_path: str | Path,
    timesteps: int | None = 8,
    seed: int = 42,
    split_seed: int = 62042,
    truth_output_path: str | Path | None = None,
    batch_size: int = 32,
    source_truth_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    with Path(input_path).open("rb") as handle:
        data = pickle.load(handle)
    with Path(header_path).open("rb") as handle:
        features = pickle.load(handle)
    external_truth = None
    if source_truth_path is not None:
        source_truth_payload = json.loads(Path(source_truth_path).read_text())
        external_truth = {
            str(row["source_patient_id"]): row
            for row in source_truth_payload["patients"]
        }
    split_assignments = None
    if split_manifest_path is not None:
        split_payload = json.loads(Path(split_manifest_path).read_text())
        split_assignments = {
            str(row["source_patient_id"]): str(row["split"])
            for row in split_payload["patients"]
        }

    dataset = build_dtcygan_dataset(
        data,
        features,
        timesteps=timesteps,
        conditioning_contract="time_local",
        split_seed=split_seed if split_assignments is None else None,
        external_truth=external_truth,
        split_assignments=split_assignments,
    )

    truth_patients = [
        {
            "patient_id": patient["patient_id"],
            "split": patient["split"],
            "counterfactual_truth": patient.pop("counterfactual_truth"),
        }
        for patient in dataset["patients"]
    ]
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

    config = build_dtcygan_config(
        features,
        int(dataset["metadata"]["sequence_length"]),
        seed=seed,
        batch_size=batch_size,
    )
    config_output = Path(config_output_path)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML, so DTCyGAN's YAML loader accepts this directly while
    # T4 does not need an additional PyYAML dependency.
    with config_output.open("w", encoding="utf-8") as handle:
        handle.write(_json_yaml_compatible(config, indent=2))

    return dataset["metadata"]


