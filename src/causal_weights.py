"""Validated patient-level causal-weight handoff utilities.

This module deliberately stops at the boundary between causal estimation and
model training.  It creates, validates, saves, and loads weight tables, but it
does not change a dataset, loss function, optimizer, or checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalWeightSpec:
    """Contract for a patient-level causal-weight table."""

    method: str = "primary_overlap"
    stage: str = "preoperative_rt"
    patient_id_col: str = "patient_id"
    weight_col: str = "causal_weight"
    ineligible_policy: str = "unit_weight"
    normalization: str = "mean_one"
    require_supported: bool = True

    def __post_init__(self) -> None:
        if self.ineligible_policy not in {"unit_weight", "exclude"}:
            raise ValueError("ineligible_policy must be 'unit_weight' or 'exclude'")
        if self.normalization not in {"mean_one", "none"}:
            raise ValueError("normalization must be 'mean_one' or 'none'")
        if not self.patient_id_col:
            raise ValueError("patient_id_col cannot be empty")
        if not self.weight_col:
            raise ValueError("weight_col cannot be empty")


@dataclass
class CausalWeightTable:
    """A validated weight frame plus its causal and normalization metadata."""

    frame: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LongitudinalWeightSpec:
    """Configuration for a patient-time training-weight experiment."""

    scheme: str = "unweighted"
    truncation_target: str = "none"
    upper_quantile: float | None = None
    normalized_cap: float | None = None
    normalization: str = "mean_one"

    def __post_init__(self) -> None:
        if self.scheme not in {"unweighted", "ow", "ipcw", "ow_ipcw"}:
            raise ValueError("scheme must be unweighted, ow, ipcw, or ow_ipcw")
        if self.truncation_target not in {"none", "combined", "ipcw"}:
            raise ValueError("truncation_target must be none, combined, or ipcw")
        if self.truncation_target != "none":
            if self.upper_quantile is None or not 0.0 < self.upper_quantile < 1.0:
                raise ValueError("A truncation quantile strictly between zero and one is required")
        elif self.upper_quantile is not None:
            raise ValueError("upper_quantile requires a truncation target")
        if self.normalized_cap is not None and self.normalized_cap <= 0:
            raise ValueError("normalized_cap must be positive")
        if self.normalization not in {"mean_one", "none"}:
            raise ValueError("normalization must be 'mean_one' or 'none'")


def _weight_distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    denominator = float(np.square(array).sum())
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p97_5": float(np.quantile(array, 0.975)),
        "p99": float(np.quantile(array, 0.99)),
        "effective_sample_size": (
            float(array.sum() ** 2 / denominator) if denominator > 0 else 0.0
        ),
    }


def compose_longitudinal_training_weights(
    person_time: pd.DataFrame,
    spec: LongitudinalWeightSpec,
    patient_id_col: str = "patient_id",
    interval_col: str = "interval",
    overlap_col: str = "overlap_weight",
    ipcw_col: str = "ipcw",
    eligible_col: str = "weight_eligible",
    training_col: str = "is_training",
) -> CausalWeightTable:
    """Compose and normalize weights using eligible training person-time only."""

    required = {patient_id_col, interval_col, eligible_col, training_col}
    if spec.scheme in {"ow", "ow_ipcw"}:
        required.add(overlap_col)
    if spec.scheme in {"ipcw", "ow_ipcw"}:
        required.add(ipcw_col)
    missing = required - set(person_time)
    if missing:
        raise ValueError(f"Longitudinal weight input is missing columns: {sorted(missing)}")
    if person_time.empty:
        raise ValueError("Longitudinal weight input cannot be empty")

    frame = person_time.copy()
    if frame.duplicated([patient_id_col, interval_col]).any():
        raise ValueError("Patient-time weight keys must be unique")
    eligible = frame[eligible_col].astype(bool)
    training = frame[training_col].astype(bool)
    fitting = eligible & training
    if not fitting.any():
        raise ValueError("At least one eligible training person-time row is required")

    ones = pd.Series(np.ones(len(frame)), index=frame.index, dtype=float)
    overlap = pd.to_numeric(frame.get(overlap_col, ones), errors="coerce")
    ipcw = pd.to_numeric(frame.get(ipcw_col, ones), errors="coerce")
    for label, values in ((overlap_col, overlap), (ipcw_col, ipcw)):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{label} must contain finite non-negative values")

    threshold: float | None = None
    truncated_count = 0
    if spec.truncation_target == "ipcw":
        threshold = float(ipcw.loc[fitting].quantile(float(spec.upper_quantile)))
        truncated_count = int((ipcw.loc[eligible] > threshold).sum())
        ipcw = ipcw.clip(upper=threshold)

    raw = {
        "unweighted": ones,
        "ow": overlap,
        "ipcw": ipcw,
        "ow_ipcw": overlap * ipcw,
    }[spec.scheme].astype(float)
    if spec.truncation_target == "combined":
        threshold = float(raw.loc[fitting].quantile(float(spec.upper_quantile)))
        truncated_count = int((raw.loc[eligible] > threshold).sum())
        raw = raw.clip(upper=threshold)

    pre_normalization = raw.copy()
    if spec.normalization == "mean_one":
        scale = float(raw.loc[fitting].mean())
        if scale <= 0:
            raise ValueError("Eligible training weights must have a positive mean")
        final = raw / scale
    else:
        scale = 1.0
        final = raw.copy()
    if spec.normalized_cap is not None:
        truncated_count += int((final.loc[eligible] > spec.normalized_cap).sum())
        final = final.clip(upper=float(spec.normalized_cap))

    frame[patient_id_col] = frame[patient_id_col].astype(str)
    frame[interval_col] = pd.to_numeric(frame[interval_col], errors="raise").astype(int)
    frame["overlap_weight"] = overlap.to_numpy(dtype=float)
    frame["ipcw"] = ipcw.to_numpy(dtype=float)
    frame["causal_weight_raw"] = pre_normalization.to_numpy(dtype=float)
    frame["causal_weight"] = final.to_numpy(dtype=float)
    frame["weight_eligible"] = eligible.to_numpy(dtype=bool)
    eligible_count = int(eligible.sum())
    metadata = {
        "table_type": "patient_time",
        "scheme": spec.scheme,
        "truncation_target": spec.truncation_target,
        "upper_quantile": spec.upper_quantile,
        "normalized_cap": spec.normalized_cap,
        "normalization": spec.normalization,
        "normalization_scale": scale,
        "threshold_fitted_on_training_only": True,
        "truncation_threshold": threshold,
        "n_truncated": int(truncated_count),
        "percent_truncated": 100.0 * truncated_count / max(1, eligible_count),
        "original_distribution": _weight_distribution(pre_normalization.loc[fitting]),
        "normalized_distribution": _weight_distribution(final.loc[fitting]),
        "patient_id_col": patient_id_col,
        "interval_col": interval_col,
        "weight_col": "causal_weight",
        "eligible_col": "weight_eligible",
    }
    return CausalWeightTable(frame=frame.reset_index(drop=True), metadata=metadata)


def validate_longitudinal_training_weights(
    frame: pd.DataFrame,
    patient_id_col: str = "patient_id",
    interval_col: str = "interval",
    weight_col: str = "causal_weight",
    eligible_col: str = "weight_eligible",
) -> pd.DataFrame:
    """Validate a patient-time weight artifact without changing row order."""

    required = {patient_id_col, interval_col, weight_col, eligible_col}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Patient-time weights are missing columns: {sorted(missing)}")
    result = frame.copy()
    if result.empty or result.duplicated([patient_id_col, interval_col]).any():
        raise ValueError("Patient-time weights must be non-empty with unique keys")
    result[patient_id_col] = result[patient_id_col].astype(str)
    result[interval_col] = pd.to_numeric(result[interval_col], errors="raise").astype(int)
    result[weight_col] = pd.to_numeric(result[weight_col], errors="coerce")
    active = result[eligible_col].astype(bool)
    values = result.loc[active, weight_col]
    if values.empty or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Eligible patient-time weights must be finite, non-negative, and nonzero")
    result[eligible_col] = active
    return result


def normalize_training_weights(
    weights: Iterable[float],
    normalization: str = "mean_one",
) -> np.ndarray:
    """Validate and optionally rescale non-negative weights to mean one."""

    values = np.asarray(list(weights), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Causal weights must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("Causal weights must all be finite")
    if (values < 0).any():
        raise ValueError("Causal weights cannot be negative")
    if float(values.sum()) <= 0:
        raise ValueError("Causal weights must have a positive sum")
    if normalization == "none":
        return values.copy()
    if normalization != "mean_one":
        raise ValueError("normalization must be 'mean_one' or 'none'")
    return values / float(values.mean())


def validate_training_weights(
    frame: pd.DataFrame,
    patient_id_col: str = "patient_id",
    weight_col: str = "causal_weight",
    expected_patient_ids: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Return a defensive copy after enforcing the patient/weight contract."""

    missing_columns = {patient_id_col, weight_col} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Weight table is missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Weight table cannot be empty")
    if frame[patient_id_col].isna().any():
        raise ValueError("Weight table contains missing patient identifiers")
    duplicated = frame.loc[frame[patient_id_col].duplicated(keep=False), patient_id_col]
    if not duplicated.empty:
        preview = duplicated.drop_duplicates().head(5).tolist()
        raise ValueError(f"Weight table has duplicate patient identifiers: {preview}")

    result = frame.copy()
    result[weight_col] = pd.to_numeric(result[weight_col], errors="coerce")
    normalize_training_weights(result[weight_col], normalization="none")

    if expected_patient_ids is not None:
        expected = pd.Index(list(expected_patient_ids))
        if expected.hasnans:
            raise ValueError("Expected patient identifiers contain missing values")
        if expected.duplicated().any():
            raise ValueError("Expected patient identifiers must be unique")
        observed = pd.Index(result[patient_id_col])
        missing = expected.difference(observed)
        unexpected = observed.difference(expected)
        if len(missing) or len(unexpected):
            raise ValueError(
                "Weight-table patient identifiers do not match the expected cohort; "
                f"missing={missing[:5].tolist()}, unexpected={unexpected[:5].tolist()}"
            )
        order = pd.Series(np.arange(len(expected)), index=expected)
        result["_expected_order"] = result[patient_id_col].map(order)
        result = (
            result.sort_values("_expected_order", kind="stable")
            .drop(columns="_expected_order")
            .reset_index(drop=True)
        )
    return result


def save_training_weights(
    table: CausalWeightTable,
    csv_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Validate and persist a weight handoff as CSV plus JSON metadata."""

    csv_target = Path(csv_path)
    metadata_target = (
        Path(metadata_path)
        if metadata_path is not None
        else csv_target.with_name(f"{csv_target.stem}_metadata.json")
    )
    patient_id_col = str(table.metadata.get("patient_id_col", "patient_id"))
    weight_col = str(table.metadata.get("weight_col", "causal_weight"))
    validated = validate_training_weights(
        table.frame,
        patient_id_col=patient_id_col,
        weight_col=weight_col,
    )
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(csv_target, index=False)
    with metadata_target.open("w", encoding="utf-8") as handle:
        json.dump(table.metadata, handle, indent=2, allow_nan=False)
    return csv_target, metadata_target


def load_training_weights(
    csv_path: str | Path,
    spec: CausalWeightSpec | None = None,
    metadata_path: str | Path | None = None,
    expected_patient_ids: Iterable[Any] | None = None,
) -> CausalWeightTable:
    """Load a persisted handoff and revalidate IDs and numeric weights."""

    contract = spec or CausalWeightSpec()
    csv_source = Path(csv_path)
    metadata_source = (
        Path(metadata_path)
        if metadata_path is not None
        else csv_source.with_name(f"{csv_source.stem}_metadata.json")
    )
    # Patient identifiers are labels, not measurements.  Reading them as
    # strings preserves leading zeroes and gives downstream joins one stable
    # representation even when all IDs happen to look numeric.
    frame = pd.read_csv(csv_source, dtype={contract.patient_id_col: "string"})
    expected_ids = (
        None if expected_patient_ids is None else [str(value) for value in expected_patient_ids]
    )
    frame = validate_training_weights(
        frame,
        patient_id_col=contract.patient_id_col,
        weight_col=contract.weight_col,
        expected_patient_ids=expected_ids,
    )
    metadata: dict[str, Any] = {}
    if metadata_source.exists():
        with metadata_source.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    required_metadata = {
        "method",
        "stage",
        "weighting_target",
        "ineligible_policy",
        "normalization",
        "adjustment_supported",
        "patient_id_col",
        "weight_col",
    }
    missing_metadata = required_metadata - set(metadata)
    if missing_metadata:
        raise ValueError(
            "Weight metadata is missing required fields: "
            f"{sorted(missing_metadata)}"
        )
    for field, expected in (
        ("method", contract.method),
        ("stage", contract.stage),
        ("ineligible_policy", contract.ineligible_policy),
        ("normalization", contract.normalization),
        ("patient_id_col", contract.patient_id_col),
        ("weight_col", contract.weight_col),
    ):
        observed = metadata.get(field)
        if observed != expected:
            raise ValueError(
                f"Weight metadata {field!r} is {observed!r}, expected {expected!r}"
            )
    if contract.require_supported and metadata.get("adjustment_supported") is not True:
        raise ValueError("Weight table is not marked as supported by the adjustment gate")
    if contract.normalization == "mean_one":
        observed_mean = float(frame[contract.weight_col].mean())
        if not np.isclose(observed_mean, 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError(
                "Weight table declares mean-one normalization, but its observed "
                f"mean is {observed_mean:.8g}"
            )
    return CausalWeightTable(frame=frame, metadata=metadata)


__all__ = [
    "CausalWeightSpec",
    "CausalWeightTable",
    "LongitudinalWeightSpec",
    "compose_longitudinal_training_weights",
    "load_training_weights",
    "normalize_training_weights",
    "save_training_weights",
    "validate_longitudinal_training_weights",
    "validate_training_weights",
]
