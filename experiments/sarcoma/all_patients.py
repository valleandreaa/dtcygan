#!/usr/bin/env python3
"""All-patient counterfactual model-plausibility diagnostics for sarcoma v8.

This analysis intentionally includes training patients. Its estimands describe
the fitted model over the observed 433-patient two-arm cohort; they are not an
out-of-sample causal validation and no uncertainty interval is calculated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from evaluate import (  # noqa: E402
    adherence,
    factual_metrics,
    final_risk_table,
    generate_trajectories,
    load_bundle,
)


ARM_LABELS = {
    "surgery_only": "Surgery alone",
    "preoperative_rt_surgery": "Preoperative RT + surgery",
}
COLORS = {
    "surgery_only": "#4C78A8",
    "preoperative_rt_surgery": "#E45756",
}


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.average(x, weights=w))


def _weighted_sd(values: Sequence[float], weights: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = np.average(x, weights=w)
    return float(np.sqrt(np.average(np.square(x - mean), weights=w)))


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantile: float
) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    positions = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(float(quantile), positions, x))


def _effective_sample_size(weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def summarize_estimand(
    effects: pd.DataFrame,
    estimand: str,
    target_population: str,
    mask: pd.Series | np.ndarray | None = None,
    weight_column: str | None = None,
) -> dict[str, Any]:
    """Summarize model-predicted potential outcomes for one target population."""

    selected = effects.loc[np.ones(len(effects), dtype=bool) if mask is None else mask].copy()
    if selected.empty:
        raise ValueError(f"Estimand {estimand} has an empty target population")
    weights = (
        np.ones(len(selected), dtype=float)
        if weight_column is None
        else selected[weight_column].to_numpy(dtype=float)
    )
    risk_0 = selected["risk_surgery_only"].to_numpy(dtype=float)
    risk_1 = selected["risk_preoperative_rt_surgery"].to_numpy(dtype=float)
    ite = selected["individual_risk_difference"].to_numpy(dtype=float)
    mean_0 = _weighted_mean(risk_0, weights)
    mean_1 = _weighted_mean(risk_1, weights)
    risk_difference = _weighted_mean(ite, weights)
    risk_ratio = mean_1 / mean_0 if mean_0 > 0 else None
    odds_0 = mean_0 / (1.0 - mean_0) if 0 < mean_0 < 1 else None
    odds_1 = mean_1 / (1.0 - mean_1) if 0 < mean_1 < 1 else None
    odds_ratio = odds_1 / odds_0 if odds_0 and odds_1 is not None else None
    material_threshold = 0.001
    return {
        "estimand": estimand,
        "target_population": target_population,
        "n_patients": int(len(selected)),
        "effective_sample_size": _effective_sample_size(weights),
        "risk_surgery_only": mean_0,
        "risk_preoperative_rt_surgery": mean_1,
        "risk_difference": risk_difference,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "ite_sd": _weighted_sd(ite, weights),
        "ite_p05": _weighted_quantile(ite, weights, 0.05),
        "ite_p25": _weighted_quantile(ite, weights, 0.25),
        "ite_median": _weighted_quantile(ite, weights, 0.50),
        "ite_p75": _weighted_quantile(ite, weights, 0.75),
        "ite_p95": _weighted_quantile(ite, weights, 0.95),
        "predicted_benefit_fraction": _weighted_mean(ite < 0, weights),
        "predicted_harm_fraction": _weighted_mean(ite > 0, weights),
        "material_benefit_fraction": _weighted_mean(
            ite < -material_threshold, weights
        ),
        "material_harm_fraction": _weighted_mean(ite > material_threshold, weights),
        "material_threshold_absolute_risk": material_threshold,
        "uncertainty_interval": "not_computed",
    }


def build_estimand_table(effects: pd.DataFrame) -> pd.DataFrame:
    treated = effects["observed_treatment"].eq(1)
    control = effects["observed_treatment"].eq(0)
    rows = [
        summarize_estimand(
            effects,
            estimand="ATE",
            target_population="all_433_empirical_patients",
        ),
        summarize_estimand(
            effects,
            estimand="ATT",
            target_population="observed_preoperative_rt_patients",
            mask=treated,
        ),
        summarize_estimand(
            effects,
            estimand="ATC",
            target_population="observed_surgery_only_patients",
            mask=control,
        ),
        summarize_estimand(
            effects,
            estimand="overlap_weighted",
            target_population="propensity_overlap_population",
            weight_column="causal_weight",
        ),
    ]
    return pd.DataFrame(rows)


def build_split_comparison(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in effects.groupby("split", sort=False):
        for estimand, weight_column in (
            ("ATE", None),
            ("overlap_weighted", "causal_weight"),
        ):
            row = summarize_estimand(
                group,
                estimand=estimand,
                target_population=f"{split}_patients",
                weight_column=weight_column,
            )
            row["split"] = split
            rows.append(row)
    return pd.DataFrame(rows)


def build_histology_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for histology, group in effects.groupby("histology", sort=False):
        ate = summarize_estimand(
            group,
            estimand="ATE",
            target_population=str(histology),
        )
        overlap_weighted = summarize_estimand(
            group,
            estimand="overlap_weighted",
            target_population=str(histology),
            weight_column="causal_weight",
        )
        rows.append(
            {
                "histology": histology,
                "n": int(len(group)),
                "overlap_ess": overlap_weighted["effective_sample_size"],
                "ate_risk_difference": ate["risk_difference"],
                "overlap_weighted_risk_difference": overlap_weighted["risk_difference"],
                "overlap_weighted_ite_sd": overlap_weighted["ite_sd"],
                "descriptive_minimum_met": bool(
                    len(group) >= 20
                    and overlap_weighted["effective_sample_size"] >= 20
                ),
                "causal_subgroup_supported": False,
                "support_reason": "all_patient_in_sample_model_diagnostic",
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def build_risk_curves(
    trajectories: pd.DataFrame, effects: pd.DataFrame
) -> pd.DataFrame:
    membership = effects.set_index("patient_id")
    targets = [
        ("ATE", effects.index, None),
        ("ATT", effects.index[effects["observed_treatment"].eq(1)], None),
        ("ATC", effects.index[effects["observed_treatment"].eq(0)], None),
        ("overlap_weighted", effects.index, "causal_weight"),
    ]
    rows: list[dict[str, Any]] = []
    for estimand, effect_indices, weight_column in targets:
        ids = set(effects.loc[effect_indices, "patient_id"].astype(str))
        target = trajectories.loc[trajectories["patient_id"].astype(str).isin(ids)]
        for (regimen, interval), group in target.groupby(
            ["regimen", "interval"], sort=False
        ):
            weights = (
                np.ones(len(group), dtype=float)
                if weight_column is None
                else group[weight_column].to_numpy(dtype=float)
            )
            rows.append(
                {
                    "estimand": estimand,
                    "regimen": regimen,
                    "interval": int(interval),
                    "month": int(interval + 1) * 3,
                    "mean_cumulative_risk": _weighted_mean(
                        group["cumulative_risk"], weights
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_outputs(
    output: Path,
    estimands: pd.DataFrame,
    split_comparison: pd.DataFrame,
    effects: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        estimands["estimand"],
        100 * estimands["risk_difference"],
        color=["#4C78A8", "#F58518", "#72B7B2", "#B279A2"],
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        ylabel="Model-predicted risk difference (percentage points)",
        title="All-patient model-based estimands (no uncertainty intervals)",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "all_patient_estimands.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for split, color in (("dev", "#4C78A8"), ("test", "#E45756")):
        group = effects.loc[effects["split"].eq(split)]
        axis.hist(
            100 * group["individual_risk_difference"],
            bins=20,
            alpha=0.45,
            label=split.title(),
            color=color,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set(
        xlabel="Individual risk difference (percentage points)",
        ylabel="Patients",
        title="Development-versus-test individual model effects",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "ite_distribution_by_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, estimand in zip(axes, ("ATE", "overlap_weighted")):
        for regimen in ARM_LABELS:
            group = curves.loc[
                curves["estimand"].eq(estimand) & curves["regimen"].eq(regimen)
            ]
            axis.plot(
                group["month"],
                100 * group["mean_cumulative_risk"],
                marker="o",
                color=COLORS[regimen],
                label=ARM_LABELS[regimen],
            )
        axis.set(
            xlabel="Months",
            ylabel="Mean cumulative local-recurrence risk (%)",
            title=f"{estimand} target population",
        )
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(figures / "all_patient_risk_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = split_comparison.pivot(
        index="estimand", columns="split", values="risk_difference"
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.35
    axis.bar(x - width / 2, 100 * pivot["dev"], width, label="Development")
    axis.bar(x + width / 2, 100 * pivot["test"], width, label="Untouched test")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, pivot.index)
    axis.set_ylabel("Risk difference (percentage points)")
    axis.set_title("In-sample versus held-out model effects")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "estimand_dev_test_comparison.png", dpi=180)
    plt.close(fig)


def run_analysis(
    checkpoint: str | Path,
    person_period_data: str | Path,
    causal_weights: str | Path,
    causal_weight_metadata: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    make_figures: bool = True,
) -> dict[str, Any]:
    bundle = load_bundle(
        checkpoint,
        person_period_data,
        weights_path=causal_weights,
        weight_metadata_path=causal_weight_metadata,
        device=device,
    )
    all_indices = list(range(len(bundle.dataset)))
    all_patient_factual = factual_metrics(bundle, indices=all_indices)
    trajectories = generate_trajectories(bundle, indices=all_indices)
    split_by_id = {
        bundle.dataset.patient_ids[index]: "dev" for index in bundle.dev_indices
    }
    split_by_id.update(
        {
            bundle.dataset.patient_ids[index]: "test"
            for index in bundle.test_indices
        }
    )
    observed_by_id = {
        bundle.dataset.patient_ids[index]: int(
            bundle.dataset.patients[index]["regimen_index"]
        )
        for index in all_indices
    }
    trajectories["patient_id"] = trajectories["patient_id"].astype(str)
    trajectories["split"] = trajectories["patient_id"].map(split_by_id)
    trajectories["observed_treatment"] = trajectories["patient_id"].map(observed_by_id)

    final = final_risk_table(trajectories).rename(
        columns={
            "surgery_only": "risk_surgery_only",
            "preoperative_rt_surgery": "risk_preoperative_rt_surgery",
        }
    )
    final["patient_id"] = final["patient_id"].astype(str)
    final["individual_risk_difference"] = (
        final["risk_preoperative_rt_surgery"] - final["risk_surgery_only"]
    )
    final["split"] = final["patient_id"].map(split_by_id)
    final["observed_treatment"] = final["patient_id"].map(observed_by_id).astype(int)
    final["observed_regimen"] = final["observed_treatment"].map(
        {0: "surgery_only", 1: "preoperative_rt_surgery"}
    )
    if bundle.weight_table is None:
        raise ValueError("All-patient v8 diagnostics require propensity weights")
    weight_frame = bundle.weight_table.frame.copy()
    weight_frame["patient_id"] = weight_frame["patient_id"].astype(str)
    extra_columns = [
        column
        for column in ("patient_id", "propensity_score", "causal_weight_raw")
        if column in weight_frame
    ]
    final = final.merge(weight_frame[extra_columns], on="patient_id", how="left")
    if final[["causal_weight", "propensity_score"]].isna().any().any():
        raise ValueError("All-patient effects are missing propensity data")

    estimands = build_estimand_table(final)
    split_comparison = build_split_comparison(final)
    histology = build_histology_summary(final)
    curves = build_risk_curves(trajectories, final)
    monotone = (
        trajectories.sort_values("interval")
        .groupby(["patient_id", "regimen"])["cumulative_risk"]
        .apply(lambda values: bool((values.diff().dropna() >= -1.0e-8).all()))
    )
    adherence_result = adherence(trajectories)
    dev_overlap_weighted = split_comparison.loc[
        split_comparison["split"].eq("dev")
        & split_comparison["estimand"].eq("overlap_weighted"),
        "risk_difference",
    ].iloc[0]
    test_overlap_weighted = split_comparison.loc[
        split_comparison["split"].eq("test")
        & split_comparison["estimand"].eq("overlap_weighted"),
        "risk_difference",
    ].iloc[0]
    plausibility = {
        "n_patients": int(len(final)),
        "n_development_patients": int(final["split"].eq("dev").sum()),
        "n_test_patients": int(final["split"].eq("test").sum()),
        "all_risks_finite": bool(
            np.isfinite(
                final[["risk_surgery_only", "risk_preoperative_rt_surgery"]]
            ).all().all()
        ),
        "all_risks_in_unit_interval": bool(
            final[["risk_surgery_only", "risk_preoperative_rt_surgery"]]
            .apply(lambda column: column.between(0, 1).all())
            .all()
        ),
        "monotone_cumulative_risk_fraction": float(monotone.mean()),
        "strategy_adherence": adherence_result,
        "minimum_individual_risk_difference": float(
            final["individual_risk_difference"].min()
        ),
        "maximum_individual_risk_difference": float(
            final["individual_risk_difference"].max()
        ),
        "propensity_ite_spearman_correlation": float(
            final[["propensity_score", "individual_risk_difference"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "dev_test_overlap_weighted_difference": float(
            dev_overlap_weighted - test_overlap_weighted
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output / "all_patient_counterfactual_trajectories.csv", index=False)
    final.to_csv(output / "all_patient_individual_effects.csv", index=False)
    estimands.to_csv(output / "all_patient_estimands.csv", index=False)
    split_comparison.to_csv(output / "dev_test_estimand_comparison.csv", index=False)
    histology.to_csv(output / "all_patient_histology_effects.csv", index=False)
    curves.to_csv(output / "all_patient_risk_curves.csv", index=False)
    report = {
        "assessment": "all_patient_model_plausibility_diagnostic",
        "checkpoint": str(bundle.checkpoint_path),
        "cohort": bundle.metadata.get("cohort"),
        "all_patient_factual_performance": all_patient_factual,
        "estimands": estimands.to_dict(orient="records"),
        "split_comparison": split_comparison.to_dict(orient="records"),
        "plausibility": plausibility,
        "interpretation_boundary": (
            "The ATE, ATT, ATC, and overlap-weighted result are fitted-model "
            "summaries. Because 346 of "
            "433 patients contributed to model training, they are not unbiased "
            "out-of-sample causal estimates. The overlap-weighted result matches the model's "
            "propensity-overlap training target. No uncertainty interval was computed."
        ),
    }
    (output / "all_patient_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    if make_figures:
        _plot_outputs(output, estimands, split_comparison, final, curves)
    return report


