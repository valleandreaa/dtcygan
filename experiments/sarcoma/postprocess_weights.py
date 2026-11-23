#!/usr/bin/env python3
"""Build propensity-score overlap weights for the focused preoperative-RT cohort.

Postoperative-only and combined pre/postoperative radiotherapy patients are
excluded. For chronology-safe configurations, the retained patients are assigned
to the preoperative-RT arm only when their RT is valid for LR prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]

from src.causal_diagnostics import propensity_overlap_summary, weight_summary
from src.causal_weights import CausalWeightTable, save_training_weights
from src.surgery_rt_effects import SurgeryRTEffectConfig, fit_propensity_weights


GATE_CRITERIA = {
    "maximum_absolute_smd": 0.10,
    "minimum_treatment_group_ess": 20.0,
    "maximum_nonoverlap_fraction": 0.05,
}

POSITIVITY_THRESHOLDS = (0.01, 0.025, 0.05, 0.10)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_propensity_covariates(config_path: str | Path) -> list[str]:
    with Path(config_path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    weighting = payload.get("causal_weighting") or {}
    covariates = [
        str(value) for value in weighting.get("adjustment_covariates", []) or []
    ]
    if not covariates:
        raise ValueError("Training configuration declares no propensity covariates")
    if len(covariates) != len(set(covariates)):
        raise ValueError("Training configuration repeats a propensity covariate")
    return covariates


def select_preop_two_arm_baseline(
    data: pd.DataFrame,
    patient_id_col: str = "patient_id",
    interval_col: str = "interval",
    chronology_safe: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one baseline row per two-arm patient without eligibility filtering."""

    required = {
        patient_id_col,
        interval_col,
        "preoperative_radiotherapy",
        "postoperative_radiotherapy",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Person-period data are missing columns: {sorted(missing)}")
    if chronology_safe:
        temporal_columns = {
            "preoperative_rt_valid_for_lr",
            "preoperative_rt_order_resolved",
        }
        temporal_missing = temporal_columns - set(data.columns)
        if temporal_missing:
            raise ValueError(
                "Chronology-safe propensity assignment requires temporal-order "
                f"columns: {sorted(temporal_missing)}. Rebuild the person-period "
                "cohort with the v9 cohort builder."
            )
    work = data.copy()
    work[patient_id_col] = work[patient_id_col].astype("string")
    work[interval_col] = pd.to_numeric(work[interval_col], errors="raise")
    if work.duplicated([patient_id_col, interval_col]).any():
        raise ValueError("There must be at most one row per patient and interval")
    work = work.sort_values([patient_id_col, interval_col], kind="stable")
    baseline = work.groupby(patient_id_col, sort=False, as_index=False).first()
    for column in ("preoperative_radiotherapy", "postoperative_radiotherapy"):
        baseline[column] = pd.to_numeric(baseline[column], errors="raise").astype(int)
        if not baseline[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only zero and one")

    preop = baseline["preoperative_radiotherapy"]
    postop = baseline["postoperative_radiotherapy"]
    surgery_only = preop.eq(0) & postop.eq(0)
    preop_only = preop.eq(1) & postop.eq(0)
    postop_only = preop.eq(0) & postop.eq(1)
    combined = preop.eq(1) & postop.eq(1)
    selected = baseline.loc[surgery_only | preop_only].copy()
    if chronology_safe:
        for column in (
            "preoperative_rt_valid_for_lr",
            "preoperative_rt_order_resolved",
        ):
            work[column] = pd.to_numeric(work[column], errors="raise").astype(int)
            if not work[column].isin([0, 1]).all():
                raise ValueError(f"{column} must contain only zero and one")
        if (
            work["preoperative_rt_valid_for_lr"]
            > work["preoperative_radiotherapy"]
        ).any():
            raise ValueError(
                "preoperative_rt_valid_for_lr cannot be one when the raw "
                "preoperative_radiotherapy indicator is zero"
            )
        aligned_assignment = work.groupby(patient_id_col, sort=False)[
            "preoperative_rt_valid_for_lr"
        ].max()
        selected["_treatment"] = (
            selected[patient_id_col].map(aligned_assignment).astype(int)
        )
        # The propensity fitter reads this configured treatment column.
        selected["preoperative_radiotherapy"] = selected["_treatment"]
    else:
        selected["_treatment"] = selected["preoperative_radiotherapy"].astype(int)
    counts = {
        "n_parent": int(len(baseline)),
        "n": int(len(selected)),
        "surgery_only": int(selected["_treatment"].eq(0).sum()),
        "preoperative_rt_surgery": int(selected["_treatment"].eq(1).sum()),
        "excluded_postoperative_rt": int(postop_only.sum()),
        "excluded_combined_rt": int(combined.sum()),
    }
    return selected.reset_index(drop=True), counts


def build_preop_two_arm_weights(
    data: pd.DataFrame,
    baseline_covariates: list[str],
    propensity_clip: float = 0.01,
    expected_counts: dict[str, int] | None = None,
    chronology_safe: bool = False,
) -> tuple[CausalWeightTable, Any, dict[str, Any]]:
    """Fit the focused propensity model and return mean-one overlap weights."""

    baseline, counts = select_preop_two_arm_baseline(
        data, chronology_safe=chronology_safe
    )
    for key, expected in (expected_counts or {}).items():
        if key in counts and counts[key] != int(expected):
            raise ValueError(
                f"Two-arm cohort contract failed for {key}: "
                f"expected {expected}, observed {counts[key]}"
            )
    missing = set(baseline_covariates) - set(baseline.columns)
    if missing:
        raise ValueError(f"Baseline propensity covariates are missing: {sorted(missing)}")

    config = SurgeryRTEffectConfig(
        treatment_col="preoperative_radiotherapy",
        baseline_covariates=list(baseline_covariates),
        propensity_clip=float(propensity_clip),
        weight_truncation_quantiles=None,
    )
    propensity = fit_propensity_weights(baseline, config)
    rows = propensity.patient_weights.copy()
    raw_weights = rows["weight_overlap"].to_numpy(dtype=float)
    causal_weights = raw_weights / float(raw_weights.mean())
    treatment = rows["_treatment"].to_numpy(dtype=int)
    scores = rows["propensity_score"].to_numpy(dtype=float)
    prevalence = float(treatment.mean())
    stabilized_iptw_raw = np.where(
        treatment == 1,
        prevalence / scores,
        (1.0 - prevalence) / (1.0 - scores),
    )
    iptw_lower, iptw_upper = np.quantile(stabilized_iptw_raw, [0.01, 0.99])
    stabilized_iptw_q01_q99 = np.clip(
        stabilized_iptw_raw, iptw_lower, iptw_upper
    )
    rows["weight_iptw_stabilized_raw"] = stabilized_iptw_raw
    rows["weight_iptw_stabilized_q01_q99"] = stabilized_iptw_q01_q99
    propensity.patient_weights["weight_iptw_stabilized_raw"] = stabilized_iptw_raw
    propensity.patient_weights[
        "weight_iptw_stabilized_q01_q99"
    ] = stabilized_iptw_q01_q99

    overlap = propensity_overlap_summary(treatment, scores)
    # Match the parent adjustment gate: count treated patients with very low
    # treatment propensity and controls with very high treatment propensity.
    # Empirical range intersection remains a stricter descriptive diagnostic,
    # not a trimming rule.
    treated_tail = float(np.mean(scores[treatment == 1] < 0.05))
    control_tail = float(np.mean(scores[treatment == 0] > 0.95))
    nonoverlap_fraction = max(treated_tail, control_tail)
    balance = propensity.balance
    weighted_balance = balance.loc[balance["estimand"].eq("overlap")]
    maximum_smd = (
        float(weighted_balance["abs_smd"].max()) if not weighted_balance.empty else np.nan
    )
    weight_diagnostics = weight_summary(treatment, causal_weights)
    arm_ess = [
        float(weight_diagnostics[label]["effective_sample_size"])
        for label in ("surgery", "surgery_radiotherapy")
    ]
    minimum_group_ess = min(arm_ess)
    supported = bool(
        np.isfinite(maximum_smd)
        and maximum_smd <= GATE_CRITERIA["maximum_absolute_smd"]
        and minimum_group_ess >= GATE_CRITERIA["minimum_treatment_group_ess"]
        and nonoverlap_fraction <= GATE_CRITERIA["maximum_nonoverlap_fraction"]
    )

    handoff = rows[["patient_id", "_treatment", "propensity_score", "weight_overlap"]].copy()
    handoff = handoff.rename(
        columns={
            "_treatment": "treatment",
            "weight_overlap": "causal_weight_raw",
        }
    )
    handoff["causal_weight"] = causal_weights
    handoff["method"] = "primary_overlap"
    handoff["stage"] = "preoperative_rt"
    handoff["target_population"] = "overlap_population"
    handoff["adjustment_supported"] = supported
    handoff["ineligible_policy"] = "exclude"

    metadata = {
        "method": "primary_overlap",
        "stage": "preoperative_rt",
        "stage_label": "baseline_preoperative_rt_two_arm",
        "weighting_target": "overlap_population",
        "changes_primary_estimand": True,
        "patient_id_col": "patient_id",
        "weight_col": "causal_weight",
        "ineligible_policy": "exclude",
        "normalization": "mean_one",
        "normalization_scope": "full_two_arm_cohort",
        "eligibility_filter_applied": False,
        "eligibility_columns_read": [],
        "postoperative_rt_excluded": True,
        "combined_rt_excluded": True,
        "selection_uses_effect_estimates": False,
        "chronology_safe_treatment_assignment": chronology_safe,
        "adjustment_supported": supported,
        "cohort_counts": counts,
        "selected_covariates": propensity.covariates,
        "propensity_model": "baseline_logistic_regression",
        "propensity_clip": float(propensity_clip),
        "gate_criteria": dict(GATE_CRITERIA),
        "gate_metrics": {
            "maximum_absolute_smd": maximum_smd,
            "minimum_treatment_group_ess": minimum_group_ess,
            "nonoverlap_fraction": nonoverlap_fraction,
        },
        "propensity_overlap": overlap,
        "positivity": {
            "tail_thresholds": positivity_diagnostic_table(
                treatment, scores
            ).to_dict(orient="records"),
            "empirical_common_support": overlap,
            "interpretation": (
                "Tail non-overlap and empirical range-intersection support are "
                "distinct diagnostics and are both reported."
            ),
        },
        "stabilized_iptw_sensitivity": {
            "truncation_quantiles": [0.01, 0.99],
            "truncation_bounds": [float(iptw_lower), float(iptw_upper)],
            "raw": weight_summary(treatment, stabilized_iptw_raw),
            "q01_q99": weight_summary(treatment, stabilized_iptw_q01_q99),
        },
        "weight_summary": {
            "mean": float(causal_weights.mean()),
            "minimum": float(causal_weights.min()),
            "maximum": float(causal_weights.max()),
            "effective_sample_size": float(
                causal_weights.sum() ** 2 / np.square(causal_weights).sum()
            ),
            "by_treatment": {
                "0": weight_diagnostics["surgery"],
                "1": weight_diagnostics["surgery_radiotherapy"],
            },
        },
    }
    return CausalWeightTable(handoff, metadata), propensity, metadata


def positivity_diagnostic_table(
    treatment: np.ndarray | pd.Series,
    propensity: np.ndarray | pd.Series,
    thresholds: tuple[float, ...] = POSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Return prespecified arm-specific propensity-tail diagnostics."""

    a = np.asarray(treatment, dtype=int)
    e = np.asarray(propensity, dtype=float)
    if a.shape != e.shape:
        raise ValueError("Treatment and propensity arrays must have the same shape")
    if not set(np.unique(a)).issubset({0, 1}):
        raise ValueError("Treatment must be binary")
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if not 0 < threshold < 0.5:
            raise ValueError("Positivity thresholds must lie between zero and 0.5")
        treated_low = (a == 1) & (e < threshold)
        control_high = (a == 0) & (e > 1.0 - threshold)
        violation = treated_low | control_high
        treated_n = int((a == 1).sum())
        control_n = int((a == 0).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "treated_below_n": int(treated_low.sum()),
                "treated_below_fraction": float(treated_low.sum() / treated_n),
                "control_above_n": int(control_high.sum()),
                "control_above_fraction": float(control_high.sum() / control_n),
                "either_tail_n": int(violation.sum()),
                "either_tail_fraction": float(violation.mean()),
                "maximum_arm_tail_fraction": float(
                    max(treated_low.sum() / treated_n, control_high.sum() / control_n)
                ),
            }
        )
    return pd.DataFrame(rows)


def weight_diagnostic_table(
    treatment: np.ndarray | pd.Series,
    propensity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize overlap and stabilized-IPTW weights overall and by arm."""

    a = np.asarray(treatment, dtype=int)
    specifications = {
        "overlap_raw": propensity_rows["weight_overlap"].to_numpy(dtype=float),
        "iptw_stabilized_raw": propensity_rows[
            "weight_iptw_stabilized_raw"
        ].to_numpy(dtype=float),
        "iptw_stabilized_q01_q99": propensity_rows[
            "weight_iptw_stabilized_q01_q99"
        ].to_numpy(dtype=float),
    }
    rows: list[dict[str, Any]] = []
    for method, values in specifications.items():
        summary = weight_summary(a, values)
        for arm in ("overall", "surgery", "surgery_radiotherapy"):
            rows.append({"method": method, "arm": arm, **summary[arm]})
    return pd.DataFrame(rows)


def write_propensity_figures(
    propensity_rows: pd.DataFrame,
    balance: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write propensity-distribution and Love plots for reviewer diagnostics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "propensity": output / "preop_two_arm_propensity_overlap.png",
        "love_plot": output / "preop_two_arm_love_plot.png",
    }

    figure, axis = plt.subplots(figsize=(8, 5))
    for arm, label, color in (
        (0, "Surgery only", "#4C78A8"),
        (1, "Preoperative RT + surgery", "#E45756"),
    ):
        scores = propensity_rows.loc[
            propensity_rows["_treatment"].eq(arm), "propensity_score"
        ]
        axis.hist(scores, bins=20, alpha=0.55, density=True, label=label, color=color)
    axis.set(xlabel="Estimated probability of preoperative RT", ylabel="Density")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(paths["propensity"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    selected = balance[balance["estimand"].isin(["unweighted", "overlap"])].copy()
    pivot = selected.pivot_table(index="term", columns="estimand", values="abs_smd")
    pivot = pivot.dropna(subset=["unweighted", "overlap"], how="all")
    pivot["maximum"] = pivot[["unweighted", "overlap"]].max(axis=1)
    pivot = pivot.nlargest(min(30, len(pivot)), "maximum").sort_values("maximum")
    height = max(5.0, 0.24 * len(pivot) + 1.5)
    figure, axis = plt.subplots(figsize=(9, height))
    positions = np.arange(len(pivot))
    axis.scatter(pivot["unweighted"], positions, label="Unweighted", color="#9D755D")
    axis.scatter(pivot["overlap"], positions, label="Overlap weighted", color="#59A14F")
    axis.axvline(0.10, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(positions, labels=pivot.index)
    axis.set(xlabel="Absolute standardized mean difference", ylabel="")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(paths["love_plot"], dpi=180, bbox_inches="tight")
    plt.close(figure)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build propensity overlap weights for the preoperative two-arm cohort."
    )
    parser.add_argument("--data", required=True, help="Strict-D person-period CSV.")
    parser.add_argument("--output-dir", required=True, help="Weight/diagnostic output directory.")
    parser.add_argument(
        "--training-config",
        default=str(ROOT / "config" / "sarcoma.yaml"),
        help="Training YAML used to obtain baseline covariates and cohort constraints.",
    )
    parser.add_argument("--propensity-clip", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    with Path(args.training_config).open(encoding="utf-8") as handle:
        training_config = yaml.safe_load(handle) or {}
    cohort = training_config.get("cohort_contract") or {}
    expectation_keys = {
        "n": "expected_n",
        "surgery_only": "expected_surgery_only",
        "preoperative_rt_surgery": "expected_preoperative_rt_surgery",
        "excluded_postoperative_rt": "expected_excluded_postoperative_rt",
        "excluded_combined_rt": "expected_excluded_combined_rt",
    }
    expected = {
        observed_key: int(cohort[config_key])
        for observed_key, config_key in expectation_keys.items()
        if cohort.get(config_key) is not None
    }
    temporal_alignment = training_config.get("temporal_alignment") or {}
    chronology_safe = int(temporal_alignment.get("contract_version", 1)) >= 2
    data = pd.read_csv(args.data, dtype={"patient_id": "string"})
    table, propensity, metadata = build_preop_two_arm_weights(
        data,
        baseline_covariates=_load_propensity_covariates(args.training_config),
        propensity_clip=args.propensity_clip,
        expected_counts=expected,
        chronology_safe=chronology_safe,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "preop_two_arm_overlap_weights.csv"
    metadata_path = output / "preop_two_arm_overlap_weights_metadata.json"
    save_training_weights(table, weights_path, metadata_path)
    propensity.patient_weights.to_csv(output / "preop_two_arm_propensity_scores.csv", index=False)
    propensity.balance.to_csv(output / "preop_two_arm_covariate_balance.csv", index=False)
    propensity.availability.to_csv(output / "preop_two_arm_covariate_availability.csv", index=False)
    treatment = propensity.patient_weights["_treatment"].to_numpy(dtype=int)
    scores = propensity.patient_weights["propensity_score"].to_numpy(dtype=float)
    positivity_diagnostic_table(treatment, scores).to_csv(
        output / "preop_two_arm_positivity_diagnostics.csv", index=False
    )
    weight_diagnostic_table(treatment, propensity.patient_weights).to_csv(
        output / "preop_two_arm_weight_diagnostics.csv", index=False
    )
    figure_paths = write_propensity_figures(
        propensity.patient_weights, propensity.balance, output / "figures"
    )
    with (output / "preop_two_arm_propensity_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_json_default, allow_nan=False)
    if not metadata["adjustment_supported"]:
        metrics = metadata["gate_metrics"]
        raise SystemExit(
            "Focused two-arm propensity adjustment failed its diagnostic gate: "
            f"max |SMD|={metrics['maximum_absolute_smd']:.4f}, "
            f"minimum group ESS={metrics['minimum_treatment_group_ess']:.2f}, "
            f"nonoverlap={metrics['nonoverlap_fraction']:.4f}"
        )
    print(f"Propensity weights: {weights_path}")
    print(f"Metadata: {metadata_path}")
    for label, path in figure_paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
