"""Diagnostics for longitudinal causal weights used during DTCyGAN training."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .causal_diagnostics import standardized_mean_differences


def effective_sample_size(weights: Sequence[float]) -> float:
    values = np.asarray(weights, dtype=float)
    denominator = float(np.square(values).sum())
    return float(values.sum() ** 2 / denominator) if denominator > 0 else 0.0


def summarize_weight_distribution(weights: Sequence[float]) -> dict[str, float]:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("weights must be a non-empty finite vector")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "p95": float(np.quantile(values, 0.95)),
        "p97_5": float(np.quantile(values, 0.975)),
        "p99": float(np.quantile(values, 0.99)),
        "effective_sample_size": effective_sample_size(values),
    }


def longitudinal_weight_diagnostics(
    frame: pd.DataFrame,
    weight_col: str = "causal_weight",
    eligible_col: str = "weight_eligible",
    interval_col: str = "interval",
    treatment_col: str = "treatment",
) -> dict[str, Any]:
    """Return overall, arm-specific, and time-specific weight diagnostics."""

    required = {weight_col, eligible_col, interval_col}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Weight diagnostics are missing columns: {sorted(missing)}")
    eligible = frame.loc[frame[eligible_col].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("No eligible person-time rows are available")
    eligible[weight_col] = pd.to_numeric(eligible[weight_col], errors="raise")

    def grouped(column: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for value, group in eligible.groupby(column, sort=True, dropna=False):
            rows.append(
                {
                    column: value.item() if hasattr(value, "item") else value,
                    "n": int(len(group)),
                    **summarize_weight_distribution(group[weight_col]),
                }
            )
        return rows

    result: dict[str, Any] = {
        "n_eligible_person_time": int(len(eligible)),
        "overall": summarize_weight_distribution(eligible[weight_col]),
        "by_time": grouped(interval_col),
    }
    if treatment_col in eligible:
        result["by_treatment_arm"] = grouped(treatment_col)
    if "causal_weight_raw" in eligible:
        result["original"] = summarize_weight_distribution(
            eligible["causal_weight_raw"]
        )
    return result


def longitudinal_balance_diagnostics(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    weight_col: str = "causal_weight",
    eligible_col: str = "weight_eligible",
    interval_col: str = "interval",
) -> list[dict[str, Any]]:
    """Calculate weighted baseline-covariate balance within later risk sets."""

    available = [column for column in covariates if column in frame]
    if not available:
        return []
    rows: list[dict[str, Any]] = []
    eligible = frame.loc[frame[eligible_col].astype(bool)]
    for interval, group in eligible.groupby(interval_col, sort=True):
        if group[treatment_col].nunique() < 2:
            continue
        table = standardized_mean_differences(
            group,
            treatment_col,
            available,
            weights=group[weight_col].to_numpy(dtype=float),
        )
        for record in table.to_dict(orient="records"):
            record[interval_col] = int(interval)
            rows.append(record)
    return rows


__all__ = [
    "effective_sample_size",
    "longitudinal_weight_diagnostics",
    "longitudinal_balance_diagnostics",
    "summarize_weight_distribution",
]
