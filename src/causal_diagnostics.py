"""Diagnostics for the surgery versus surgery-plus-radiotherapy analysis.

The helpers in this module are deliberately estimator agnostic.  They operate on
patient-level baseline records and make every exclusion from the adjustment set
auditable.  In particular, fields that are recognisably downstream outcomes or
post-treatment summaries are never silently accepted as baseline confounders.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LEAKAGE_TOKENS = (
    "status_at_last_follow_up",
    "last_status",
    "final_status",
    "death",
    "dod",
    "local_recurrence",
    "metastasis_at_follow_up",
    "treatment_metastasis",
    "treatment_of_metastasis",
    "reoperation",
    "margin_status",
    "margin_width",
    "radiotherapy_dose",
    "fraction",
    "toxicity",
    "treatment_response",
    "chemo_response",
    "hyperthermia",
    "interventional_radiology",
    "surgery_type",
    "prom_score",
    "outcome",
    "endpoint",
    "fake_episodes",
    "counterfactual",
)


def effective_sample_size(weights: Sequence[float] | pd.Series) -> float:
    """Return Kish's effective sample size for non-negative weights."""

    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return 0.0
    denominator = float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def _looks_like_leakage(column: str, leakage_tokens: Iterable[str]) -> bool:
    lowered = column.lower().replace(" ", "_")
    return any(token.lower() in lowered for token in leakage_tokens)


def assess_covariates(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    required: Sequence[str] = (),
    extra_candidates: Sequence[str] = (),
    forbidden: Sequence[str] = (),
    leakage_tokens: Sequence[str] = DEFAULT_LEAKAGE_TOKENS,
    min_coverage: float = 0.05,
    allow_leakage_named: Sequence[str] = (),
) -> tuple[list[str], pd.DataFrame]:
    """Select usable baseline covariates and return a decision table.

    ``extra_candidates`` are evaluated by exactly the same rules as the primary
    covariates.  This is how newly available data can be considered without
    automatically treating every new ETL field as a confounder.
    """

    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must lie between zero and one")

    required_set = set(required)
    forbidden_set = set(forbidden)
    allow_leakage_set = set(allow_leakage_named)
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, values in (("configured", candidates), ("extra", extra_candidates)):
        for column in values:
            if column not in seen:
                ordered.append((column, source))
                seen.add(column)

    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    missing_required: list[str] = []
    for column, source in ordered:
        exists = column in frame.columns
        coverage = float(frame[column].notna().mean()) if exists and len(frame) else 0.0
        unique = int(frame[column].nunique(dropna=True)) if exists else 0
        reason = "included"
        include = True
        if not exists:
            include = False
            reason = "missing"
        elif column in forbidden_set:
            include = False
            reason = "explicitly_forbidden"
        elif column not in allow_leakage_set and _looks_like_leakage(column, leakage_tokens):
            include = False
            reason = "possible_post_treatment_or_outcome_leakage"
        elif coverage < min_coverage:
            include = False
            reason = "coverage_below_threshold"
        elif unique <= 1:
            include = False
            reason = "constant_or_all_missing"

        if include:
            selected.append(column)
        elif column in required_set:
            missing_required.append(f"{column} ({reason})")

        rows.append(
            {
                "variable": column,
                "source": source,
                "required": column in required_set,
                "exists": exists,
                "coverage": coverage,
                "n_unique": unique,
                "selected": include,
                "reason": reason,
            }
        )

    if missing_required:
        raise ValueError("Required baseline covariates are unusable: " + ", ".join(missing_required))
    return selected, pd.DataFrame(rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or weights[valid].sum() <= 0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or weights[valid].sum() <= 0:
        return float("nan")
    x = values[valid]
    w = weights[valid]
    mean = np.average(x, weights=w)
    return float(np.average(np.square(x - mean), weights=w))


def _smd_numeric(values: pd.Series, treatment: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    if pd.isna(median):
        return float("nan")
    x = numeric.fillna(median).to_numpy(dtype=float)
    a = treatment.to_numpy(dtype=int)
    w = weights.to_numpy(dtype=float)
    treated = a == 1
    control = a == 0
    mean_1 = _weighted_mean(x[treated], w[treated])
    mean_0 = _weighted_mean(x[control], w[control])
    var_1 = _weighted_variance(x[treated], w[treated])
    var_0 = _weighted_variance(x[control], w[control])
    pooled = np.sqrt((var_1 + var_0) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0:
        return 0.0 if np.isclose(mean_1, mean_0, equal_nan=False) else float("nan")
    return float((mean_1 - mean_0) / pooled)


def standardized_mean_differences(
    frame: pd.DataFrame,
    treatment_col: str,
    covariates: Sequence[str],
    weights: Sequence[float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Compute numeric and level-specific categorical standardized differences."""

    if treatment_col not in frame:
        raise KeyError(f"Treatment column not found: {treatment_col}")
    treatment = pd.to_numeric(frame[treatment_col], errors="coerce")
    if not set(treatment.dropna().unique()).issubset({0, 1}):
        raise ValueError("Treatment must be binary for standardized differences")
    weight_series = (
        pd.Series(np.ones(len(frame)), index=frame.index, dtype=float)
        if weights is None
        else pd.Series(np.asarray(weights, dtype=float), index=frame.index)
    )

    rows: list[dict[str, Any]] = []
    for column in covariates:
        if column not in frame:
            continue
        series = frame[column]
        is_numeric = pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 2
        if is_numeric:
            rows.append(
                {
                    "variable": column,
                    "term": column,
                    "kind": "numeric",
                    "smd": _smd_numeric(series, treatment, weight_series),
                }
            )
            if series.isna().any():
                rows.append(
                    {
                        "variable": column,
                        "term": f"{column}=missing",
                        "kind": "missing_indicator",
                        "smd": _smd_numeric(series.isna().astype(float), treatment, weight_series),
                    }
                )
            continue

        categorical = series.astype("string").fillna("__MISSING__")
        for level in sorted(categorical.unique().tolist(), key=str):
            indicator = (categorical == level).astype(float)
            rows.append(
                {
                    "variable": column,
                    "term": f"{column}={level}",
                    "kind": "categorical_level",
                    "smd": _smd_numeric(indicator, treatment, weight_series),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_smd"] = result["smd"].abs()
    return result


def propensity_overlap_summary(
    treatment: Sequence[int] | pd.Series,
    propensity: Sequence[float] | pd.Series,
) -> dict[str, Any]:
    """Summarize empirical propensity overlap between the two treatment arms."""

    a = np.asarray(treatment, dtype=int)
    e = np.asarray(propensity, dtype=float)
    if a.shape != e.shape:
        raise ValueError("Treatment and propensity arrays must have the same shape")
    treated = e[a == 1]
    control = e[a == 0]
    if treated.size == 0 or control.size == 0:
        return {"has_both_arms": False}
    lower = float(max(treated.min(), control.min()))
    upper = float(min(treated.max(), control.max()))
    inside = (e >= lower) & (e <= upper) if lower <= upper else np.zeros_like(e, dtype=bool)
    return {
        "has_both_arms": True,
        "treated_n": int(treated.size),
        "control_n": int(control.size),
        "treated_range": [float(treated.min()), float(treated.max())],
        "control_range": [float(control.min()), float(control.max())],
        "common_support": [lower, upper] if lower <= upper else None,
        "outside_common_support_n": int((~inside).sum()),
        "outside_common_support_fraction": float((~inside).mean()),
        "quantiles": {
            "treated": np.quantile(treated, [0.01, 0.25, 0.5, 0.75, 0.99]).tolist(),
            "control": np.quantile(control, [0.01, 0.25, 0.5, 0.75, 0.99]).tolist(),
        },
    }


def weight_summary(
    treatment: Sequence[int] | pd.Series,
    weights: Sequence[float] | pd.Series,
) -> dict[str, Any]:
    """Return distribution and effective-size summaries overall and by arm."""

    a = np.asarray(treatment, dtype=int)
    w = np.asarray(weights, dtype=float)
    if a.shape != w.shape:
        raise ValueError("Treatment and weight arrays must have the same shape")

    def summarize(values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"n": 0, "effective_sample_size": 0.0}
        return {
            "n": int(finite.size),
            "min": float(finite.min()),
            "median": float(np.median(finite)),
            "mean": float(finite.mean()),
            "p95": float(np.quantile(finite, 0.95)),
            "p99": float(np.quantile(finite, 0.99)),
            "max": float(finite.max()),
            "effective_sample_size": effective_sample_size(finite),
        }

    return {
        "overall": summarize(w),
        "surgery": summarize(w[a == 0]),
        "surgery_radiotherapy": summarize(w[a == 1]),
    }


__all__ = [
    "DEFAULT_LEAKAGE_TOKENS",
    "assess_covariates",
    "effective_sample_size",
    "propensity_overlap_summary",
    "standardized_mean_differences",
    "weight_summary",
]
