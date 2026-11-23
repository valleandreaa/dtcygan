"""Weights and diagnostics for preoperative RT plus surgery versus surgery."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .causal_diagnostics import (
    assess_covariates,
    propensity_overlap_summary,
    standardized_mean_differences,
    weight_summary,
)


ARM_LABELS = {0: "S", 1: "S+RT"}


class DynamicTreatmentNotSupported(ValueError):
    """Raised when a baseline-strategy estimator receives dynamic treatment."""


@dataclass
class SurgeryRTEffectConfig:
    patient_id_col: str = "patient_id"
    interval_col: str = "interval"
    treatment_col: str = "treatment_arm"
    at_risk_col: str = "at_risk"
    observed_col: str = "observed"
    outcome_columns: dict[str, str] = field(
        default_factory=lambda: {
            "local_recurrence": "local_recurrence",
            "dod": "dod",
        }
    )
    estimand_outcomes: list[str] = field(
        default_factory=lambda: ["local_recurrence"]
    )
    baseline_covariates: list[str] = field(default_factory=list)
    required_covariates: list[str] = field(default_factory=list)
    extra_baseline_covariates: list[str] = field(default_factory=list)
    allow_leakage_named: list[str] = field(default_factory=list)
    time_varying_censoring_covariates: list[str] = field(default_factory=list)
    estimands: list[str] = field(
        default_factory=lambda: ["ate", "att", "overlap"]
    )
    censoring_weighting_enabled: bool = True
    censoring_treatment_mode: str = "auto"
    min_covariate_coverage: float = 0.05
    propensity_clip: float = 0.01
    weight_truncation_quantiles: tuple[float, float] | None = (0.01, 0.99)
    censoring_weight_truncation_quantiles: tuple[float, float] | None = (0.01, 0.99)
    interval_months: int = 6
    horizons_months: list[int] = field(default_factory=lambda: [12, 24, 48])
    bootstrap_samples: int = 0
    seed: int = 42


@dataclass
class PropensityResult:
    patient_weights: pd.DataFrame
    covariates: list[str]
    availability: pd.DataFrame
    balance: pd.DataFrame
    diagnostics: dict[str, Any]
    model: Any = None


@dataclass
class SurgeryRTAnalysisResult:
    effects: pd.DataFrame
    risk_curves: pd.DataFrame
    propensity_scores: pd.DataFrame
    covariate_balance: pd.DataFrame
    covariate_availability: pd.DataFrame
    bootstrap_effects: pd.DataFrame
    diagnostics: dict[str, Any]


def _binary_value(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Treatment contains a missing value")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric in {0.0, 1.0}:
            return int(numeric)
    normalized = str(value).strip().lower().replace(" ", "").replace("_", "")
    control = {"0", "s", "surgery", "surgeryonly", "surgeryalone", "no", "false"}
    treated = {
        "1",
        "s+rt",
        "surgery+rt",
        "surgery+radiotherapy",
        "surgeryradiotherapy",
        "rt",
        "yes",
        "true",
    }
    if normalized in control:
        return 0
    if normalized in treated:
        return 1
    raise ValueError(
        f"Unsupported treatment value {value!r}; the analysis only accepts S and S+RT"
    )


def _coerce_binary(series: pd.Series, name: str, *, allow_missing: bool = False) -> pd.Series:
    replacements = {
        True: 1,
        False: 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "Yes": 1,
        "No": 0,
    }
    converted = pd.to_numeric(series.replace(replacements), errors="coerce")
    if not allow_missing and converted.isna().any():
        raise ValueError(f"{name} contains values that cannot be interpreted as binary")
    observed = set(converted.dropna().unique().tolist())
    if not observed.issubset({0, 1}):
        raise ValueError(f"{name} must contain only zero and one; observed {sorted(observed)}")
    return converted.astype("Int64" if allow_missing else int)


def _validate_and_prepare_longitudinal(
    data: pd.DataFrame,
    config: SurgeryRTEffectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {config.patient_id_col, config.interval_col, config.treatment_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Person-period data are missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Person-period data are empty")

    work = data.copy()
    if work[config.patient_id_col].isna().any():
        raise ValueError("Patient identifiers cannot be missing")
    work[config.interval_col] = pd.to_numeric(work[config.interval_col], errors="coerce")
    if work[config.interval_col].isna().any():
        raise ValueError("Interval values must be numeric")
    if work.duplicated([config.patient_id_col, config.interval_col]).any():
        raise ValueError("There must be at most one row per patient and interval")
    work["_treatment"] = work[config.treatment_col].map(_binary_value)

    treatment_counts = work.groupby(config.patient_id_col, sort=False)["_treatment"].nunique()
    dynamic_ids = treatment_counts[treatment_counts > 1].index.tolist()
    if dynamic_ids:
        preview = dynamic_ids[:5]
        raise DynamicTreatmentNotSupported(
            "The selected treatment column changes within patient. This implementation "
            "estimates a baseline S versus S+RT strategy and cannot estimate sequential "
            f"radiotherapy effects. Example patient IDs: {preview}"
        )

    work = work.sort_values([config.patient_id_col, config.interval_col]).reset_index(drop=True)
    if config.at_risk_col in work:
        work["_at_risk"] = _coerce_binary(work[config.at_risk_col], config.at_risk_col)
    else:
        work["_at_risk"] = 1
    if config.observed_col in work:
        work["_observed"] = _coerce_binary(work[config.observed_col], config.observed_col)
    else:
        work["_observed"] = 1

    available_outcomes = 0
    for endpoint, column in config.outcome_columns.items():
        internal = f"_outcome_{endpoint}"
        if column in work:
            work[internal] = _coerce_binary(work[column], column, allow_missing=True).fillna(0).astype(int)
            available_outcomes += 1
        else:
            work[internal] = 0
    if available_outcomes == 0:
        raise ValueError("None of the configured outcome columns is present")

    baseline = work.groupby(config.patient_id_col, sort=False).first().reset_index()
    if baseline["_treatment"].nunique() < 2:
        counts = baseline["_treatment"].value_counts().to_dict()
        raise ValueError(f"Both S and S+RT patients are required; observed arm counts: {counts}")
    return work, baseline


def _preprocessor(frame: pd.DataFrame, columns: Sequence[str], *, scale_numeric: bool = True) -> ColumnTransformer:
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in columns if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ]
        if scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        # Preserve informative missingness as its own level;
                        # modal imputation hides the exact imbalance that the
                        # causal diagnostics are intended to detect.
                        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _fit_probability_model(
    frame: pd.DataFrame,
    target: pd.Series,
    columns: Sequence[str],
    sample_weight: Sequence[float] | None = None,
) -> tuple[np.ndarray, Any]:
    y = np.asarray(target, dtype=int)
    if np.unique(y).size < 2:
        probability = float(np.average(y, weights=sample_weight)) if y.size else 0.0
        return np.full(len(frame), probability, dtype=float), {"constant_probability": probability}
    if not columns:
        probability = float(np.average(y, weights=sample_weight))
        return np.full(len(frame), probability, dtype=float), {"constant_probability": probability}
    preprocessor = _preprocessor(frame, columns)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            # Propensity/censoring models need enough flexibility to balance
            # strongly separated clinical groups. A weak penalty avoids the
            # substantial shrinkage (and residual imbalance) of sklearn's
            # default C=1 while retaining numerical stability.
            ("model", LogisticRegression(max_iter=4000, solver="lbfgs", C=100.0)),
        ]
    )
    fit_kwargs = {"model__sample_weight": np.asarray(sample_weight, dtype=float)} if sample_weight is not None else {}
    pipeline.fit(frame[list(columns)], y, **fit_kwargs)
    probability = pipeline.predict_proba(frame[list(columns)])[:, 1]
    return probability.astype(float), pipeline


def _truncate_weights(weights: np.ndarray, quantiles: tuple[float, float] | None) -> tuple[np.ndarray, list[float] | None]:
    if quantiles is None:
        return weights.copy(), None
    lower_q, upper_q = quantiles
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("Weight truncation quantiles must satisfy 0 <= lower < upper <= 1")
    lower, upper = np.quantile(weights, [lower_q, upper_q])
    return np.clip(weights, lower, upper), [float(lower), float(upper)]


def fit_propensity_weights(
    baseline: pd.DataFrame,
    config: SurgeryRTEffectConfig,
) -> PropensityResult:
    """Fit baseline propensity scores and construct ATE and ATT weights."""

    forbidden = [
        config.patient_id_col,
        config.interval_col,
        config.treatment_col,
        config.at_risk_col,
        config.observed_col,
        *config.outcome_columns.values(),
    ]
    covariates, availability = assess_covariates(
        baseline,
        config.baseline_covariates,
        required=config.required_covariates,
        extra_candidates=config.extra_baseline_covariates,
        forbidden=forbidden,
        min_coverage=config.min_covariate_coverage,
        allow_leakage_named=config.allow_leakage_named,
    )
    treatment = baseline["_treatment"].astype(int).to_numpy()
    prevalence = float(treatment.mean())
    if not 0 < prevalence < 1:
        raise ValueError("Both treatment arms are required to estimate propensity weights")
    propensity, model = _fit_probability_model(baseline, baseline["_treatment"], covariates)
    clip = float(config.propensity_clip)
    if not 0 < clip < 0.5:
        raise ValueError("propensity_clip must be between zero and 0.5")
    propensity = np.clip(propensity, clip, 1.0 - clip)

    ate_raw = np.where(treatment == 1, prevalence / propensity, (1.0 - prevalence) / (1.0 - propensity))
    att_raw = np.where(treatment == 1, 1.0, propensity / (1.0 - propensity))
    overlap = np.where(treatment == 1, 1.0 - propensity, propensity)
    ate, ate_bounds = _truncate_weights(ate_raw, config.weight_truncation_quantiles)
    att, att_bounds = _truncate_weights(att_raw, config.weight_truncation_quantiles)

    patient_weights = baseline[[config.patient_id_col, "_treatment"]].copy()
    patient_weights["propensity_score"] = propensity
    patient_weights["weight_ate_raw"] = ate_raw
    patient_weights["weight_ate"] = ate
    patient_weights["weight_att_raw"] = att_raw
    patient_weights["weight_att"] = att
    patient_weights["weight_overlap"] = overlap

    before = standardized_mean_differences(baseline, "_treatment", covariates)
    balance_frames: list[pd.DataFrame] = []
    if not before.empty:
        before = before.assign(estimand="unweighted")
        balance_frames.append(before)
    for estimand, weights in (("ate", ate), ("att", att), ("overlap", overlap)):
        table = standardized_mean_differences(
            baseline, "_treatment", covariates, weights=weights
        )
        if not table.empty:
            balance_frames.append(table.assign(estimand=estimand))
    balance = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()

    diagnostics = {
        "model": "baseline_logistic_propensity",
        "treatment_prevalence": prevalence,
        "n_patients": int(len(baseline)),
        "arm_counts": {
            ARM_LABELS[key]: int(value)
            for key, value in baseline["_treatment"].value_counts().sort_index().items()
        },
        "selected_covariates": covariates,
        "propensity_clip": clip,
        "overlap": propensity_overlap_summary(treatment, propensity),
        "weight_truncation_quantiles": list(config.weight_truncation_quantiles)
        if config.weight_truncation_quantiles
        else None,
        "ate_truncation_bounds": ate_bounds,
        "att_truncation_bounds": att_bounds,
        "ate_weights": weight_summary(treatment, ate),
        "att_weights": weight_summary(treatment, att),
        "overlap_weights": weight_summary(treatment, overlap),
    }
    return PropensityResult(
        patient_weights=patient_weights,
        covariates=covariates,
        availability=availability,
        balance=balance,
        diagnostics=diagnostics,
        model=model,
    )


def _censoring_weights(
    longitudinal: pd.DataFrame,
    baseline_covariates: Sequence[str],
    time_varying_covariates: Sequence[str],
    config: SurgeryRTEffectConfig,
) -> tuple[pd.Series, dict[str, Any]]:
    """Estimate stabilized inverse-probability-of-observation weights.

    Rows after a patient is first unobserved do not contribute.  If observation
    is complete, unit weights are returned without fitting a meaningless model.
    """

    result = pd.Series(np.ones(len(longitudinal)), index=longitudinal.index, dtype=float)
    eligible = longitudinal["_at_risk"].eq(1)
    fit_frame = longitudinal.loc[eligible].copy()
    if fit_frame.empty or fit_frame["_observed"].nunique() < 2:
        return result, {
            "applied": False,
            "reason": "complete_observation_or_no_eligible_rows",
            "n_unobserved": int((fit_frame.get("_observed", pd.Series(dtype=int)) == 0).sum()),
            "uses_future_pathway_label": False,
        }

    # Censoring may use only treatment observed before the current interval.
    lagged_treatment_history = [
        column
        for column in (
            "lag_preoperative_radiotherapy",
        )
        if column in fit_frame
    ]
    treatment_mode = str(config.censoring_treatment_mode).lower()
    if treatment_mode not in {"auto", "baseline", "lagged"}:
        raise ValueError(
            "censoring_treatment_mode must be one of: auto, baseline, lagged"
        )
    use_baseline_treatment = treatment_mode == "baseline" or (
        treatment_mode == "auto" and not lagged_treatment_history
    )
    if treatment_mode == "lagged" and not lagged_treatment_history:
        raise ValueError(
            "censoring_treatment_mode='lagged' requires lagged treatment columns"
        )
    treatment_columns = ["_treatment"] if use_baseline_treatment else lagged_treatment_history
    numerator_columns = [*treatment_columns, config.interval_col]
    denominator_candidates = [
        *numerator_columns,
        *[column for column in baseline_covariates if column in fit_frame],
        *[column for column in time_varying_covariates if column in fit_frame],
    ]
    denominator_columns = list(dict.fromkeys(denominator_candidates))
    numerator, _ = _fit_probability_model(
        fit_frame, fit_frame["_observed"], numerator_columns
    )
    denominator, _ = _fit_probability_model(
        fit_frame, fit_frame["_observed"], denominator_columns
    )
    clip = config.propensity_clip
    numerator = np.clip(numerator, clip, 1.0 - clip)
    denominator = np.clip(denominator, clip, 1.0 - clip)
    observed_values = fit_frame["_observed"].to_numpy(dtype=float)
    fit_frame["_ipcw_increment"] = np.where(
        fit_frame["_observed"].to_numpy() == 1,
        numerator / denominator,
        (1.0 - numerator) / (1.0 - denominator),
    )
    fit_frame["_ipcw"] = fit_frame.groupby(config.patient_id_col, sort=False)[
        "_ipcw_increment"
    ].cumprod()
    raw_ipcw = fit_frame["_ipcw"].to_numpy(dtype=float)
    truncated_ipcw, truncation_bounds = _truncate_weights(
        raw_ipcw, config.censoring_weight_truncation_quantiles
    )
    fit_frame["_ipcw"] = truncated_ipcw
    result.loc[fit_frame.index] = fit_frame["_ipcw"].to_numpy()
    return result, {
        "applied": True,
        "n_model_rows": int(len(fit_frame)),
        "n_unobserved": int((fit_frame["_observed"] == 0).sum()),
        "denominator_covariates": denominator_columns,
        "numerator_covariates": numerator_columns,
        "treatment_history_mode": "baseline" if use_baseline_treatment else "lagged",
        "uses_future_pathway_label": False,
        "numerator_probability_range": [float(numerator.min()), float(numerator.max())],
        "denominator_probability_range": [
            float(denominator.min()),
            float(denominator.max()),
        ],
        "numerator_brier_score": float(np.mean(np.square(observed_values - numerator))),
        "denominator_brier_score": float(
            np.mean(np.square(observed_values - denominator))
        ),
        "observed_fraction": float(observed_values.mean()),
        "mean_predicted_observation_numerator": float(numerator.mean()),
        "mean_predicted_observation_denominator": float(denominator.mean()),
        "truncation_bounds": truncation_bounds,
        "raw_min": float(raw_ipcw.min()),
        "raw_median": float(np.median(raw_ipcw)),
        "raw_max": float(raw_ipcw.max()),
        "min": float(fit_frame["_ipcw"].min()),
        "median": float(fit_frame["_ipcw"].median()),
        "max": float(fit_frame["_ipcw"].max()),
    }


def estimate_training_ipcw(
    data: pd.DataFrame,
    config: SurgeryRTEffectConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate untruncated chronology-safe IPCW for training person-time.

    Truncation is applied later using the fixed training-weight specification.
    """

    untruncated = replace(config, censoring_weight_truncation_quantiles=None)
    longitudinal, _ = _validate_and_prepare_longitudinal(data, untruncated)
    ipcw, diagnostics = _censoring_weights(
        longitudinal,
        untruncated.baseline_covariates,
        untruncated.time_varying_censoring_covariates,
        untruncated,
    )
    eligible = longitudinal["_at_risk"].eq(1) & longitudinal["_observed"].eq(1)
    if "at_risk_local_recurrence" in longitudinal:
        eligible &= longitudinal["at_risk_local_recurrence"].eq(1)
    result = longitudinal[
        [untruncated.patient_id_col, untruncated.interval_col]
    ].copy()
    result = result.rename(
        columns={
            untruncated.patient_id_col: "patient_id",
            untruncated.interval_col: "interval",
        }
    )
    result["patient_id"] = result["patient_id"].astype(str)
    result["ipcw"] = ipcw.to_numpy(dtype=float)
    result["weight_eligible"] = eligible.to_numpy(dtype=bool)
    if "split" in longitudinal:
        result["is_training"] = longitudinal["split"].isin({"dev", "train"}).to_numpy()
    diagnostics = {
        **diagnostics,
        "training_export": True,
        "post_event_rows_eligible": False,
        "disease_related_death_is_competing": True,
        "truncation_deferred_to_experiment": True,
    }
    return result.reset_index(drop=True), diagnostics


def build_training_weight_components(
    data: pd.DataFrame,
    config: SurgeryRTEffectConfig,
    training_patient_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build aligned OW and raw IPCW components for sensitivity experiments."""

    longitudinal, baseline = _validate_and_prepare_longitudinal(data, config)
    propensity = fit_propensity_weights(baseline, config)
    patient = propensity.patient_weights.rename(
        columns={
            config.patient_id_col: "patient_id",
            "_treatment": "treatment",
            "weight_overlap": "overlap_weight",
        }
    )[["patient_id", "treatment", "overlap_weight"]]
    patient["patient_id"] = patient["patient_id"].astype(str)
    ipcw, ipcw_diagnostics = estimate_training_ipcw(data, config)
    components = ipcw.merge(patient, on="patient_id", how="left", validate="many_to_one")
    if components["overlap_weight"].isna().any():
        raise ValueError("Overlap weights could not be aligned to every patient-time row")
    available_covariates = [
        column for column in config.baseline_covariates if column in baseline
    ]
    if available_covariates:
        covariates = baseline[
            [config.patient_id_col, *available_covariates]
        ].rename(columns={config.patient_id_col: "patient_id"})
        covariates["patient_id"] = covariates["patient_id"].astype(str)
        components = components.merge(
            covariates, on="patient_id", how="left", validate="many_to_one"
        )
    if training_patient_ids is not None:
        training_ids = {str(value) for value in training_patient_ids}
        components["is_training"] = components["patient_id"].isin(training_ids)
    elif "split" in longitudinal:
        split = longitudinal[
            [config.patient_id_col, config.interval_col, "split"]
        ].rename(
            columns={config.patient_id_col: "patient_id", config.interval_col: "interval"}
        )
        split["patient_id"] = split["patient_id"].astype(str)
        components = components.drop(columns="is_training", errors="ignore").merge(
            split, on=["patient_id", "interval"], how="left", validate="one_to_one"
        )
        split_labels = components.pop("split")
        components["is_training"] = split_labels.isin({"dev", "train"})
    else:
        raise ValueError(
            "Training patient IDs are required when person-period data has no split column"
        )
    return components, {
        "ipcw": ipcw_diagnostics,
        "overlap": propensity.diagnostics,
        "baseline_balance": json.loads(
            propensity.balance.to_json(orient="records")
        ),
        "uses_outcomes_for_weight_selection": False,
    }


def _first_event_risk_set(
    data: pd.DataFrame,
    endpoint: str,
    endpoint_col: str,
    competing_col: str | None,
    config: SurgeryRTEffectConfig,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    endpoint_risk_col = f"at_risk_{endpoint}"
    clinical_risk_col = endpoint_risk_col if endpoint_risk_col in data else config.at_risk_col
    for _, patient in data.groupby(config.patient_id_col, sort=False):
        active = True
        for _, row in patient.sort_values(config.interval_col).iterrows():
            if not active:
                break
            if int(row[clinical_risk_col] if clinical_risk_col in row else row["_at_risk"]) != 1:
                break
            if int(row["_observed"]) != 1:
                break
            record = row.copy()
            event = int(record[endpoint_col])
            competing = int(record[competing_col]) if competing_col else 0
            # With six-month bins exact within-bin ordering may be unavailable.
            # Conservatively count a recorded endpoint before a same-bin death.
            if event == 1:
                competing = 0
            record["_event"] = event
            record["_competing"] = competing
            rows.append(record)
            if event == 1 or competing == 1:
                active = False
    if not rows:
        return pd.DataFrame(columns=[*data.columns, "_event", "_competing"])
    return pd.DataFrame(rows).reset_index(drop=True)


def _hazard_design(frame: pd.DataFrame, intervals: Sequence[float]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["treatment"] = frame["_treatment"].astype(float)
    for interval in intervals[1:]:
        label = str(interval).replace(".", "_")
        dummy = frame["_analysis_interval"].eq(interval).astype(float)
        result[f"time_{label}"] = dummy
        result[f"treatment_x_time_{label}"] = dummy * result["treatment"]
    return result


@dataclass
class _HazardModel:
    intervals: list[float]
    model: Any
    constant: float | None = None

    def predict(self, treatment: int) -> np.ndarray:
        frame = pd.DataFrame(
            {
                "_treatment": [treatment] * len(self.intervals),
                "_analysis_interval": self.intervals,
            }
        )
        if self.constant is not None:
            return np.full(len(frame), self.constant, dtype=float)
        design = _hazard_design(frame, self.intervals)
        return self.model.predict_proba(design)[:, 1]


def _fit_hazard(
    risk_set: pd.DataFrame,
    target: str,
    intervals: Sequence[float],
) -> _HazardModel:
    y = risk_set[target].astype(int).to_numpy()
    if np.unique(y).size < 2:
        constant = float(np.average(y, weights=risk_set["_analysis_weight"])) if len(y) else 0.0
        return _HazardModel(list(intervals), model=None, constant=constant)
    design = _hazard_design(risk_set, intervals)
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(design, y, sample_weight=risk_set["_analysis_weight"].to_numpy(dtype=float))
    return _HazardModel(list(intervals), model=model)


def _estimate_endpoint_curve(
    longitudinal: pd.DataFrame,
    endpoint: str,
    endpoint_col: str,
    competing_col: str | None,
    estimand: str,
    config: SurgeryRTEffectConfig,
) -> pd.DataFrame:
    risk_set = _first_event_risk_set(longitudinal, endpoint, endpoint_col, competing_col, config)
    if risk_set.empty:
        raise ValueError(f"No observable at-risk rows for outcome '{endpoint}'")
    if risk_set["_event"].nunique() < 2:
        raise ValueError(
            f"Outcome '{endpoint}' has one observed class in its risk set; "
            "an effect cannot be estimated"
        )
    intervals = sorted(risk_set[config.interval_col].dropna().unique().tolist())
    risk_set["_analysis_interval"] = risk_set[config.interval_col]
    risk_set["_analysis_weight"] = (
        risk_set[f"weight_{estimand}"].to_numpy(dtype=float)
        * risk_set["_ipcw"].to_numpy(dtype=float)
    )
    event_model = _fit_hazard(risk_set, "_event", intervals)
    competing_model = _fit_hazard(risk_set, "_competing", intervals) if competing_col else None

    rows: list[dict[str, Any]] = []
    for arm in (0, 1):
        event_hazard = np.clip(event_model.predict(arm), 0.0, 1.0)
        competing_hazard = (
            np.clip(competing_model.predict(arm), 0.0, 1.0)
            if competing_model is not None
            else np.zeros_like(event_hazard)
        )
        survival = 1.0
        cumulative = 0.0
        competing_cumulative = 0.0
        for interval, event_h, competing_h in zip(intervals, event_hazard, competing_hazard):
            total_hazard = float(event_h + competing_h)
            if total_hazard > 1.0:
                event_h = float(event_h / total_hazard)
                competing_h = float(competing_h / total_hazard)
            cumulative += survival * float(event_h)
            competing_cumulative += survival * float(competing_h)
            survival *= max(0.0, 1.0 - float(event_h) - float(competing_h))
            rows.append(
                {
                    "estimand": estimand,
                    "outcome": endpoint,
                    "arm": ARM_LABELS[arm],
                    "treatment": arm,
                    "interval": interval,
                    "months": (int(interval) + 1) * config.interval_months,
                    "hazard": float(event_h),
                    "competing_hazard": float(competing_h),
                    "risk": float(cumulative),
                    "competing_risk": float(competing_cumulative),
                    "event_free_survival": float(survival),
                }
            )
    return pd.DataFrame(rows)


def _effects_from_curves(curves: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (estimand, outcome), group in curves.groupby(["estimand", "outcome"], sort=False):
        max_month = int(group["months"].max())
        for horizon in horizons:
            arm_rows = group[group["months"] == horizon].set_index("arm")
            available = horizon <= max_month and {"S", "S+RT"}.issubset(arm_rows.index)
            if available:
                risk_s = float(arm_rows.loc["S", "risk"])
                risk_rt = float(arm_rows.loc["S+RT", "risk"])
                difference = risk_rt - risk_s
                ratio = risk_rt / risk_s if risk_s > 0 else float("nan")
            else:
                risk_s = risk_rt = difference = ratio = float("nan")
            rows.append(
                {
                    "estimand": estimand,
                    "outcome": outcome,
                    "horizon_months": int(horizon),
                    "available": bool(available),
                    "risk_surgery": risk_s,
                    "risk_surgery_radiotherapy": risk_rt,
                    "risk_difference": difference,
                    "risk_ratio": ratio,
                    "max_observed_month": max_month,
                }
            )
    return pd.DataFrame(rows)


def _analyze_once(
    data: pd.DataFrame,
    config: SurgeryRTEffectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, PropensityResult, dict[str, Any]]:
    longitudinal, baseline = _validate_and_prepare_longitudinal(data, config)
    propensity = fit_propensity_weights(baseline, config)
    longitudinal = longitudinal.merge(
        propensity.patient_weights,
        on=[config.patient_id_col, "_treatment"],
        how="left",
        validate="many_to_one",
    )
    longitudinal["weight_unadjusted"] = 1.0
    requested_estimands = list(dict.fromkeys(config.estimands))
    supported_estimands = {"unadjusted", "ate", "att", "overlap"}
    invalid_estimands = sorted(set(requested_estimands) - supported_estimands)
    if invalid_estimands:
        raise ValueError(f"Unsupported estimands: {invalid_estimands}")
    if not requested_estimands:
        raise ValueError("At least one estimand must be requested")
    if config.censoring_weighting_enabled:
        ipcw, censoring_diagnostics = _censoring_weights(
            longitudinal,
            propensity.covariates,
            config.time_varying_censoring_covariates,
            config,
        )
    else:
        ipcw = pd.Series(np.ones(len(longitudinal)), index=longitudinal.index, dtype=float)
        eligible = longitudinal["_at_risk"].eq(1)
        censoring_diagnostics = {
            "applied": False,
            "reason": "disabled_for_analysis_variant",
            "n_unobserved": int(
                longitudinal.loc[eligible, "_observed"].eq(0).sum()
            ),
            "uses_future_pathway_label": False,
        }
    longitudinal["_ipcw"] = ipcw

    curve_frames: list[pd.DataFrame] = []
    outcome_errors: dict[str, str] = {}
    dod_internal = "_outcome_dod" if config.outcome_columns.get("dod") in data.columns else None
    for estimand in requested_estimands:
        for endpoint in config.estimand_outcomes:
            configured_column = config.outcome_columns.get(endpoint)
            if not configured_column or configured_column not in data.columns:
                outcome_errors[endpoint] = "configured outcome column is unavailable"
                continue
            internal = f"_outcome_{endpoint}"
            competing = dod_internal if endpoint in {"local_recurrence", "metastasis"} else None
            try:
                curve_frames.append(
                    _estimate_endpoint_curve(
                        longitudinal,
                        endpoint,
                        internal,
                        competing,
                        estimand,
                        config,
                    )
                )
            except ValueError as exc:
                outcome_errors[endpoint] = str(exc)
    if not curve_frames:
        raise ValueError(
            "No configured outcome could be estimated. "
            + "; ".join(f"{key}: {value}" for key, value in outcome_errors.items())
        )
    curves = pd.concat(curve_frames, ignore_index=True)
    effects = _effects_from_curves(curves, config.horizons_months)
    max_abs_smd = (
        propensity.balance.groupby("estimand")["abs_smd"].max().dropna().to_dict()
        if not propensity.balance.empty
        else {}
    )
    combined_weight_diagnostics: dict[str, Any] = {}
    eligible_weight_rows = longitudinal["_at_risk"].eq(1) & longitudinal["_observed"].eq(1)
    for estimand in requested_estimands:
        values = (
            longitudinal.loc[eligible_weight_rows, f"weight_{estimand}"].to_numpy(dtype=float)
            * longitudinal.loc[eligible_weight_rows, "_ipcw"].to_numpy(dtype=float)
        )
        treatment = longitudinal.loc[eligible_weight_rows, "_treatment"].to_numpy(dtype=int)
        combined_weight_diagnostics[estimand] = weight_summary(treatment, values)
    diagnostics = {
        "analysis_type": "baseline_strategy_weighted_pooled_logistic",
        "treatment_contrast": "S+RT versus S",
        "dynamic_treatment_supported": False,
        "interval_months": config.interval_months,
        "observed_intervals": sorted(longitudinal[config.interval_col].unique().tolist()),
        "propensity": propensity.diagnostics,
        "balance": {
            "maximum_absolute_smd": {key: float(value) for key, value in max_abs_smd.items()},
            "threshold": 0.10,
            "passes_threshold": {
                key: bool(value <= 0.10) for key, value in max_abs_smd.items() if key != "unweighted"
            },
        },
        "censoring": censoring_diagnostics,
        "censoring_weighting_enabled": bool(config.censoring_weighting_enabled),
        "requested_estimands": requested_estimands,
        "combined_person_period_weights": combined_weight_diagnostics,
        "outcome_errors": outcome_errors,
        "same_interval_event_death_rule": "endpoint_before_competing_death",
    }
    return effects, curves, propensity, diagnostics


def _bootstrap(
    data: pd.DataFrame,
    config: SurgeryRTEffectConfig,
) -> pd.DataFrame:
    if config.bootstrap_samples <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(config.seed)
    patient_ids = data[config.patient_id_col].drop_duplicates().to_numpy()
    rows: list[pd.DataFrame] = []
    for sample_index in range(config.bootstrap_samples):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        pieces: list[pd.DataFrame] = []
        for draw_index, patient_id in enumerate(sampled):
            piece = data[data[config.patient_id_col] == patient_id].copy()
            piece[config.patient_id_col] = f"bootstrap_{sample_index}_{draw_index}"
            pieces.append(piece)
        bootstrap_data = pd.concat(pieces, ignore_index=True)
        bootstrap_config = SurgeryRTEffectConfig(**{**config.__dict__, "bootstrap_samples": 0})
        try:
            effects, _, _, _ = _analyze_once(bootstrap_data, bootstrap_config)
        except (ValueError, np.linalg.LinAlgError):
            continue
        effect_rows = effects[effects["available"]].copy()
        effect_rows["bootstrap_sample"] = sample_index
        rows.append(effect_rows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _attach_bootstrap_intervals(
    effects: pd.DataFrame,
    bootstrap: pd.DataFrame,
    requested: int,
) -> pd.DataFrame:
    result = effects.copy()
    for metric in ("risk_surgery", "risk_surgery_radiotherapy", "risk_difference", "risk_ratio"):
        result[f"{metric}_ci_lower"] = np.nan
        result[f"{metric}_ci_upper"] = np.nan
    result["bootstrap_samples_requested"] = int(requested)
    result["bootstrap_samples_used"] = 0
    if bootstrap.empty:
        return result
    key_columns = ["estimand", "outcome", "horizon_months"]
    for index, row in result.iterrows():
        mask = np.ones(len(bootstrap), dtype=bool)
        for key in key_columns:
            mask &= bootstrap[key].eq(row[key]).to_numpy()
        group = bootstrap.loc[mask]
        result.at[index, "bootstrap_samples_used"] = int(group["bootstrap_sample"].nunique())
        for metric in ("risk_surgery", "risk_surgery_radiotherapy", "risk_difference", "risk_ratio"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not values.empty:
                lower, upper = np.quantile(values, [0.025, 0.975])
                result.at[index, f"{metric}_ci_lower"] = float(lower)
                result.at[index, f"{metric}_ci_upper"] = float(upper)
    return result


def analyze_surgery_rt(
    data: pd.DataFrame,
    config: SurgeryRTEffectConfig | None = None,
) -> SurgeryRTAnalysisResult:
    """Run the complete S versus S+RT analysis on person-period data."""

    config = config or SurgeryRTEffectConfig()
    effects, curves, propensity, diagnostics = _analyze_once(data, config)
    bootstrap = _bootstrap(data, config)
    effects = _attach_bootstrap_intervals(effects, bootstrap, config.bootstrap_samples)
    diagnostics["bootstrap"] = {
        "requested": int(config.bootstrap_samples),
        "successful": int(bootstrap["bootstrap_sample"].nunique()) if not bootstrap.empty else 0,
        "resampling_unit": "patient",
    }
    return SurgeryRTAnalysisResult(
        effects=effects,
        risk_curves=curves,
        propensity_scores=propensity.patient_weights,
        covariate_balance=propensity.balance,
        covariate_availability=propensity.availability,
        bootstrap_effects=bootstrap,
        diagnostics=diagnostics,
    )


def config_from_mapping(payload: Mapping[str, Any]) -> SurgeryRTEffectConfig:
    """Construct an estimator configuration from a flexible project YAML mapping."""

    analysis = payload.get("analysis", payload.get("causal_analysis", {})) or {}
    roles = payload.get("roles", payload.get("causal_roles", {})) or {}
    variables = payload.get("variables", {}) or {}

    baseline = list(analysis.get("baseline_covariates", roles.get("baseline_confounders", [])) or [])
    extras = list(analysis.get("extra_baseline_covariates", []) or [])
    required = list(analysis.get("required_covariates", []) or [])
    if variables:
        for semantic_name, definition in variables.items():
            if not isinstance(definition, Mapping):
                continue
            if definition.get("role") != "baseline_confounder":
                continue
            resolved = definition.get("resolved") or definition.get("column")
            if resolved:
                (baseline if definition.get("primary", True) else extras).append(str(resolved))
                if definition.get("required"):
                    required.append(str(resolved))

    configured_outcomes = analysis.get("outcome_columns")
    outcome_columns = (
        dict(configured_outcomes)
        if configured_outcomes is not None
        else {
            "local_recurrence": "local_recurrence",
            "dod": "dod",
        }
    )
    truncation = analysis.get("weight_truncation_quantiles", [0.01, 0.99])
    return SurgeryRTEffectConfig(
        patient_id_col=analysis.get("patient_id_col", "patient_id"),
        interval_col=analysis.get("interval_col", "interval"),
        treatment_col=analysis.get("treatment_col", "treatment_arm"),
        at_risk_col=analysis.get("at_risk_col", "at_risk"),
        observed_col=analysis.get("observed_col", "observed"),
        outcome_columns=outcome_columns,
        estimand_outcomes=list(
            analysis.get("estimand_outcomes", list(outcome_columns)) or []
        ),
        baseline_covariates=list(dict.fromkeys(baseline)),
        required_covariates=list(dict.fromkeys(required)),
        extra_baseline_covariates=list(dict.fromkeys(extras)),
        allow_leakage_named=list(analysis.get("allow_leakage_named", []) or []),
        time_varying_censoring_covariates=list(
            analysis.get("time_varying_censoring_covariates", roles.get("time_varying_confounders", []))
            or []
        ),
        estimands=list(analysis.get("estimands", ["ate", "att", "overlap"]) or []),
        censoring_weighting_enabled=bool(
            analysis.get("censoring_weighting_enabled", True)
        ),
        censoring_treatment_mode=str(
            analysis.get("censoring_treatment_mode", "auto")
        ),
        min_covariate_coverage=float(analysis.get("min_covariate_coverage", 0.05)),
        propensity_clip=float(analysis.get("propensity_clip", 0.01)),
        weight_truncation_quantiles=tuple(truncation) if truncation is not None else None,
        censoring_weight_truncation_quantiles=(
            tuple(analysis.get("censoring_weight_truncation_quantiles", truncation))
            if analysis.get("censoring_weight_truncation_quantiles", truncation) is not None
            else None
        ),
        interval_months=int(analysis.get("interval_months", payload.get("time", {}).get("interval_months", 6))),
        horizons_months=list(analysis.get("horizons_months", [12, 24, 48])),
        bootstrap_samples=int(analysis.get("bootstrap_samples", 0)),
        seed=int(analysis.get("seed", 42)),
    )


__all__ = [
    "DynamicTreatmentNotSupported",
    "PropensityResult",
    "SurgeryRTAnalysisResult",
    "SurgeryRTEffectConfig",
    "analyze_surgery_rt",
    "build_training_weight_components",
    "config_from_mapping",
    "estimate_training_ipcw",
    "fit_propensity_weights",
]
