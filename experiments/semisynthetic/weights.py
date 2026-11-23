#!/usr/bin/env python
"""Build training-fitted patient-time propensity weights for aligned MIMIC."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STATIC_FEATURES = ("agegroup", "heightgroup", "weightgroup", "gender")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


def _absolute_smd(
    features: np.ndarray, treatment: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    result = np.zeros(values.shape[1], dtype=float)
    for column in range(values.shape[1]):
        x = values[:, column]
        summaries = []
        for arm in (0, 1):
            active = treatment == arm
            arm_weights = weights[active]
            arm_x = x[active]
            total = float(arm_weights.sum())
            if not active.any() or total <= 0:
                summaries.append((0.0, 0.0))
                continue
            mean = float(np.sum(arm_weights * arm_x) / total)
            variance = float(np.sum(arm_weights * np.square(arm_x - mean)) / total)
            summaries.append((mean, variance))
        pooled = np.sqrt(max((summaries[0][1] + summaries[1][1]) / 2.0, 0.0))
        result[column] = (
            abs(summaries[1][0] - summaries[0][0]) / pooled if pooled > 0 else 0.0
        )
    return result


def _extract_arrays(
    payload: dict[str, Any], baseline_end: int
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[str]]:
    metadata = payload.get("metadata", {})
    patients = payload.get("patients")
    if not isinstance(patients, list) or not patients:
        raise ValueError("dataset must contain patients")
    alignment = metadata.get("forecast_alignment")
    if not isinstance(alignment, dict) or alignment.get("contract") != "t4_baseline_forecast":
        raise ValueError("propensity weights require a T4-aligned forecast dataset")
    features = [str(value) for value in metadata.get("clinical_features", [])]
    temporal = [value for value in features if value not in STATIC_FEATURES]
    if not temporal:
        raise ValueError("aligned dataset has no temporal features")
    feature_names = [
        f"{feature}_t{timestamp}"
        for timestamp in range(baseline_end + 1)
        for feature in temporal
    ] + list(STATIC_FEATURES)
    patient_ids: list[str] = []
    split_labels: list[str] = []
    baseline_rows: list[list[float]] = []
    treatments: list[list[int]] = []
    for patient in patients:
        patient_ids.append(str(patient["patient_id"]))
        split_labels.append(str(patient["split"]))
        records = sorted(patient["clinical"], key=lambda row: int(row["timestep"]))
        if len(records) <= baseline_end:
            raise ValueError("patient clinical sequence is shorter than the baseline")
        row = [
            float(records[timestamp][feature])
            if records[timestamp].get(feature) is not None
            else float("nan")
            for timestamp in range(baseline_end + 1)
            for feature in temporal
        ]
        row.extend(float(records[0][feature]) for feature in STATIC_FEATURES)
        baseline_rows.append(row)
        actual = sorted(
            patient["actual_treatment"], key=lambda item: int(item["timestep"])
        )
        treatments.append([int(item["temporal_treatment"]) for item in actual])
    return (
        patient_ids,
        split_labels,
        np.asarray(baseline_rows, dtype=float),
        np.asarray(treatments, dtype=int),
        feature_names,
    )


def build_weights(
    payload: dict[str, Any],
    baseline_end: int = 8,
    evaluation_start: int = 9,
    evaluation_end: int = 15,
    probability_clip: float = 0.01,
    truncation_quantiles: tuple[float, float] = (0.01, 0.99),
    source_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 <= baseline_end < evaluation_start <= evaluation_end:
        raise ValueError("invalid baseline/follow-up timestamp contract")
    alignment = payload.get("metadata", {}).get("forecast_alignment", {})
    if alignment.get("baseline_timestamps") != list(range(baseline_end + 1)):
        raise ValueError("weight baseline does not match the aligned dataset")
    if alignment.get("evaluation_timestamps") != list(
        range(evaluation_start, evaluation_end + 1)
    ):
        raise ValueError("weight timestamps do not match the aligned dataset")
    patient_ids, splits, baseline, treatment, feature_names = _extract_arrays(
        payload, baseline_end
    )
    split_array = np.asarray(splits)
    train_mask = split_array == "train"
    if not train_mask.any() or not np.any(split_array == "validation"):
        raise ValueError("explicit train and validation splits are required")
    sequence_length = treatment.shape[1]
    if evaluation_end >= sequence_length:
        raise ValueError("evaluation timestamp exceeds the treatment sequence")
    if not 0 < probability_clip < 0.5:
        raise ValueError("probability_clip must lie in (0, 0.5)")
    lower_q, upper_q = truncation_quantiles
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("invalid truncation quantiles")

    rows = [
        {
            "patient_id": patient_id,
            "interval": timestamp,
            "causal_weight": 0.0,
            "weight_eligible": False,
        }
        for patient_id in patient_ids
        for timestamp in range(sequence_length)
    ]
    row_lookup = {
        (row["patient_id"], row["interval"]): row
        for row in rows
    }
    diagnostics: dict[str, Any] = {}
    all_weights: list[float] = []
    for timestamp in range(evaluation_start, evaluation_end + 1):
        prior_treatment = treatment[:, evaluation_start:timestamp]
        design = (
            np.column_stack((baseline, prior_treatment))
            if prior_treatment.shape[1]
            else baseline.copy()
        )
        target = treatment[:, timestamp]
        train_target = target[train_mask]
        if len(np.unique(train_target)) != 2:
            raise ValueError(f"timestamp {timestamp} lacks both treatment arms in training")
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        solver="lbfgs", max_iter=2000, random_state=0
                    ),
                ),
            ]
        )
        model.fit(design[train_mask], train_target)
        propensity = np.clip(
            model.predict_proba(design)[:, 1], probability_clip, 1.0 - probability_clip
        )
        marginal = float(train_target.mean())
        numerator = np.where(target == 1, marginal, 1.0 - marginal)
        denominator = np.where(target == 1, propensity, 1.0 - propensity)
        raw = numerator / denominator
        lower, upper = np.quantile(raw[train_mask], [lower_q, upper_q])
        truncated = np.clip(raw, lower, upper)
        normalization = float(truncated[train_mask].mean())
        final = truncated / normalization
        transformed = model[:-1].transform(design[train_mask])
        before = _absolute_smd(
            transformed, train_target, np.ones(int(train_mask.sum()), dtype=float)
        )
        after = _absolute_smd(transformed, train_target, final[train_mask])
        diagnostics[str(timestamp)] = {
            "train_treatment_rate": marginal,
            "train_propensity_min": float(propensity[train_mask].min()),
            "train_propensity_max": float(propensity[train_mask].max()),
            "truncation_bounds": [float(lower), float(upper)],
            "normalization_scale": normalization,
            "train_effective_sample_size": _effective_sample_size(final[train_mask]),
            "train_max_abs_smd_before": float(before.max()),
            "train_max_abs_smd_after": float(after.max()),
            "train_mean_abs_smd_before": float(before.mean()),
            "train_mean_abs_smd_after": float(after.mean()),
        }
        for patient_id, weight in zip(patient_ids, final):
            row = row_lookup[(patient_id, timestamp)]
            row["causal_weight"] = float(weight)
            row["weight_eligible"] = True
        all_weights.extend(final[train_mask].tolist())

    train_weights = np.asarray(all_weights, dtype=float)
    metadata: dict[str, Any] = {
        "table_type": "patient_time",
        "scheme": "stabilized_time_local_iptw",
        "estimand": "conditional point-treatment effect at baseline-aligned history",
        "fit_split": "train",
        "n_fit_patients": int(train_mask.sum()),
        "baseline_timestamps": list(range(baseline_end + 1)),
        "eligible_timestamps": list(range(evaluation_start, evaluation_end + 1)),
        "propensity_features": feature_names,
        "prior_treatment_history_included": True,
        "probability_clip": probability_clip,
        "truncation_quantiles": [lower_q, upper_q],
        "normalization": "training_mean_one_per_timestamp",
        "threshold_fitted_on_training_only": True,
        "normalized_distribution": {
            "mean": float(train_weights.mean()),
            "minimum": float(train_weights.min()),
            "maximum": float(train_weights.max()),
            "effective_sample_size": _effective_sample_size(train_weights),
        },
        "diagnostics_by_timestamp": diagnostics,
    }
    if source_sha256 is not None:
        metadata["source_sha256"] = source_sha256
    return rows, metadata


def write_weights(
    dataset: Path,
    output_csv: Path,
    output_metadata: Path,
    baseline_end: int = 8,
    evaluation_start: int = 9,
    evaluation_end: int = 15,
    probability_clip: float = 0.01,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    with dataset.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows, metadata = build_weights(
        payload,
        baseline_end=baseline_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        probability_clip=probability_clip,
        truncation_quantiles=(lower_quantile, upper_quantile),
        source_sha256=_sha256(dataset),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("patient_id", "interval", "causal_weight", "weight_eligible"),
        )
        writer.writeheader()
        writer.writerows(rows)
    output_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


