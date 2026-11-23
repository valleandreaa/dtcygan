#!/usr/bin/env python3
"""Generate plots for the preoperative-RT versus surgery model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
COLORS = {"surgery_only": "#4C78A8", "preoperative_rt_surgery": "#E45756"}
LABELS = {
    "surgery_only": "Surgery alone",
    "preoperative_rt_surgery": "Preoperative RT + surgery",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    x = values.to_numpy(float)
    w = weights.to_numpy(float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    positions = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, positions, x))


def _weighted_gini_mean_difference(values: pd.Series, weights: pd.Series) -> float:
    x = values.to_numpy(float)
    w = weights.to_numpy(float)
    return float(np.sum(np.abs(x[:, None] - x[None, :]) * w[:, None] * w[None, :]) / (w.sum() ** 2))


def _bernoulli_kl(observed: float, predicted: float) -> float:
    eps = 1.0e-8
    p = float(np.clip(observed, eps, 1 - eps))
    q = float(np.clip(predicted, eps, 1 - eps))
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


def run(results: Path, all_dir: Path, person_period: Path, output: Path) -> dict:
    required = [
        results / "report.json",
        results / "counterfactual_trajectories.csv",
        results / "test_covariate_balance.csv",
        results / "histology_effect_summary.csv",
        all_dir / "all_patient_estimands.csv",
        all_dir / "all_patient_individual_effects.csv",
        all_dir / "dev_test_estimand_comparison.csv",
        person_period,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sarcoma postprocessing inputs: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report = json.loads((results / "report.json").read_text(encoding="utf-8"))
    trajectories = pd.read_csv(results / "counterfactual_trajectories.csv", dtype={"patient_id": "string"})
    balance = pd.read_csv(results / "test_covariate_balance.csv")
    histology = pd.read_csv(results / "histology_effect_summary.csv")
    estimands = pd.read_csv(all_dir / "all_patient_estimands.csv")
    individual = pd.read_csv(all_dir / "all_patient_individual_effects.csv", dtype={"patient_id": "string"})
    split_estimands = pd.read_csv(all_dir / "dev_test_estimand_comparison.csv")
    person = pd.read_csv(person_period, dtype={"patient_id": "string"})

    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    interval_end_month = (
        person[["interval", "interval_end_month"]]
        .drop_duplicates("interval")
        .set_index("interval")["interval_end_month"]
        .to_dict()
    )
    history_rows = []
    for row in checkpoint.get("training_history", []):
        development = row.get("development", row.get("validation", {}))
        history_rows.append({
            "epoch": row["epoch"],
            **{f"train_{key}": value for key, value in row["train"].items()},
            **{f"development_{key}": value for key, value in development.items()},
        })
    history = pd.DataFrame(history_rows)
    history.to_csv(output / "training_history.csv", index=False)

    # Training history.
    metrics = ["g_loss", "d_loss", "outcome_loss", "factual_loss", "regimen_delivery_loss", "rt_feasibility_loss"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        axis.plot(history["epoch"], history[f"train_{metric}"], label="Train", color="#4C78A8")
        axis.plot(history["epoch"], history[f"development_{metric}"], label="Development", color="#F58518")
        axis.set(title=metric.replace("_", " ").title(), xlabel="Epoch")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    _save(fig, figures / "training_diagnostics")

    # Held-out test balance.
    before = balance.loc[balance["estimand"].eq("unweighted")].set_index("term")["abs_smd"]
    after = balance.loc[balance["estimand"].eq("overlap_weighted")].copy()
    after["unweighted_abs_smd"] = after["term"].map(before)
    plotted = after.nlargest(20, "abs_smd").sort_values("abs_smd")
    fig, axis = plt.subplots(figsize=(10, 8))
    y = np.arange(len(plotted))
    axis.scatter(plotted["unweighted_abs_smd"], y, label="Unweighted", color="#E45756")
    axis.scatter(plotted["abs_smd"], y, label="Overlap weighted", color="#4C78A8")
    axis.axvline(0.10, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, plotted["term"])
    axis.set(xlabel="Absolute standardized mean difference", title="Untouched-test covariate balance")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.25)
    _save(fig, figures / "test_covariate_balance")

    # Test cumulative-risk trajectories at the true three-month spacing.
    trajectory_summary = []
    for (regimen, interval), group in trajectories.groupby(["regimen", "interval"], sort=True):
        trajectory_summary.append({
            "regimen": regimen,
            "interval": int(interval),
            "month": int(interval_end_month[int(interval)]),
            "weighted_mean_risk": _weighted_mean(group["cumulative_risk"], group["causal_weight"]),
            "weighted_median_risk": _weighted_quantile(group["cumulative_risk"], group["causal_weight"], 0.5),
            "weighted_gini_mean_difference": _weighted_gini_mean_difference(group["cumulative_risk"], group["causal_weight"]),
        })
    trajectory_summary = pd.DataFrame(trajectory_summary)
    trajectory_summary.to_csv(output / "trajectory_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for regimen, group in trajectory_summary.groupby("regimen"):
        axes[0].plot(group["month"], 100 * group["weighted_median_risk"], marker="o", label=LABELS[regimen], color=COLORS[regimen])
        axes[1].plot(group["month"], 100 * group["weighted_gini_mean_difference"], marker="o", label=LABELS[regimen], color=COLORS[regimen])
    axes[0].set(title="Cumulative local-recurrence risk", xlabel="Months", ylabel="Weighted median risk (%)")
    axes[1].set(title="Trajectory dispersion", xlabel="Months", ylabel="Weighted Gini mean difference (pp)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    _save(fig, figures / "median_gini_trajectories")

    # Individual effects and waterfall.
    individual["paper_delta"] = individual["individual_risk_difference"]
    individual["predicted_benefit"] = -individual["paper_delta"]
    individual.to_csv(output / "individualized_treatment_effects.csv", index=False)
    ordered = individual.sort_values("paper_delta").reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(np.arange(len(ordered)), 100 * ordered["paper_delta"], color=np.where(ordered["paper_delta"] <= 0, "#4C78A8", "#E45756"), width=1)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Patients ranked by individual effect", ylabel="Risk difference (percentage points)", title="Individual-effect waterfall")
    _save(fig, figures / "ite_waterfall")

    # Fitted-model summaries and development/test distributions.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(estimands["estimand"], 100 * estimands["risk_difference"], color=["#4C78A8", "#F58518", "#72B7B2", "#B279A2"])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="All-patient model-based estimands", ylabel="Risk difference (percentage points)")
    for split, color in (("dev", "#4C78A8"), ("test", "#E45756")):
        group = individual.loc[individual["split"].eq(split)]
        axes[1].hist(100 * group["paper_delta"], bins=20, alpha=0.45, color=color, label=split.upper())
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(title="Development versus untouched test", xlabel="Individual risk difference (pp)", ylabel="Patients")
    axes[1].legend(frameon=False)
    _save(fig, figures / "estimands_and_split_effects")

    # Histology and grade summaries. Retain every test-set histology, including
    # sparse strata, and label sample size/support instead of silently filtering.
    baseline = person.sort_values(["patient_id", "interval"]).drop_duplicates("patient_id")[["patient_id", "biopsy_grading"]]
    subgroup_data = individual.merge(baseline, on="patient_id", how="left")
    grade_rows = []
    for grade, group in subgroup_data.groupby("biopsy_grading", dropna=False):
        if len(group) < 5:
            continue
        grade_rows.append({"grade": str(grade), "n": len(group), "weighted_mean_risk_difference": _weighted_mean(group["paper_delta"], group["causal_weight"])})
    grade = pd.DataFrame(grade_rows)
    histology.to_csv(output / "histology_effects.csv", index=False)
    grade.to_csv(output / "grade_effects.csv", index=False)
    histology_order = histology.sort_values(
        ["n", "histology"], ascending=[False, True]
    )["histology"].tolist()
    hist_plot = histology.set_index("histology").loc[histology_order].reset_index()
    hist_plot["plot_label"] = hist_plot.apply(
        lambda row: (
            f"{str(row['histology']).replace('_', ' ')} "
            f"(n={int(row['n'])}, ESS={row['effective_sample_size']:.1f})"
        ),
        axis=1,
    )
    hist_colors = np.where(
        hist_plot["descriptive_minimum_met"], "#4C78A8", "#B8B8B8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].barh(
        hist_plot["plot_label"],
        100 * hist_plot["weighted_mean_risk_difference"],
        color=hist_colors,
    )
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].invert_yaxis()
    axes[0].set(
        title="All test-set histologies (descriptive)",
        xlabel="Risk difference (percentage points)",
    )
    axes[1].bar(grade["grade"], 100 * grade["weighted_mean_risk_difference"], color="#72B7B2")
    axes[1].set(title="Grade effects (descriptive)", ylabel="Risk difference (pp)")
    _save(fig, figures / "histology_and_grade_effects")

    # Histology-specific counterfactual trajectories for all eight test strata.
    histology_trajectory_rows = []
    for (histology_name, regimen, interval), group in trajectories.groupby(
        ["histology", "regimen", "interval"], sort=True
    ):
        histology_trajectory_rows.append({
            "histology": histology_name,
            "regimen": regimen,
            "interval": int(interval),
            "month": int(interval_end_month[int(interval)]),
            "n": int(group["patient_id"].nunique()),
            "effective_sample_size": float(
                group["causal_weight"].sum() ** 2
                / np.square(group["causal_weight"]).sum()
            ),
            "weighted_mean_risk": _weighted_mean(
                group["cumulative_risk"], group["causal_weight"]
            ),
            "weighted_median_risk": _weighted_quantile(
                group["cumulative_risk"], group["causal_weight"], 0.5
            ),
        })
    histology_trajectories = pd.DataFrame(histology_trajectory_rows)
    histology_trajectories.to_csv(
        output / "histology_trajectory_summary.csv", index=False
    )
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 9), sharex=True, sharey=True, constrained_layout=True
    )
    histology_lookup = histology.set_index("histology")
    for axis, histology_name in zip(axes.flat, histology_order):
        subset = histology_trajectories.loc[
            histology_trajectories["histology"].eq(histology_name)
        ]
        for regimen, group in subset.groupby("regimen", sort=False):
            axis.plot(
                group["month"],
                100 * group["weighted_median_risk"],
                marker="o",
                linewidth=1.8,
                color=COLORS[regimen],
                label=LABELS[regimen],
            )
        row = histology_lookup.loc[histology_name]
        sparse = not bool(row["descriptive_minimum_met"])
        axis.set_title(
            f"{str(histology_name).replace('_', ' ').title()}\n"
            f"n={int(row['n'])}, ESS={row['effective_sample_size']:.1f}"
            + (" — sparse" if sparse else ""),
            color="#8C3B3B" if sparse else "black",
            fontsize=10,
        )
        axis.set_xticks(sorted(interval_end_month.values())[1::2])
        axis.grid(alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Weighted median risk (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Months")
    axes.flat[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        "Counterfactual cumulative-risk trajectories for every test-set histology",
        y=1.02,
    )
    _save(fig, figures / "all_histology_risk_trajectories")

    # Patient-level test effects by histology. Points are preferable to violin
    # summaries because several explicitly retained strata contain only 1–2 patients.
    test_individual = individual.loc[individual["split"].eq("test")].copy()
    test_individual.to_csv(output / "test_histology_individual_effects.csv", index=False)
    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    rng = np.random.default_rng(42)
    for position, histology_name in enumerate(histology_order):
        group = test_individual.loc[test_individual["histology"].eq(histology_name)]
        jitter = rng.uniform(-0.14, 0.14, size=len(group))
        axis.scatter(
            100 * group["paper_delta"],
            position + jitter,
            s=28,
            alpha=0.75,
            color="#4C78A8",
        )
        row = histology_lookup.loc[histology_name]
        axis.scatter(
            100 * row["weighted_mean_risk_difference"],
            position,
            marker="D",
            s=60,
            color="#E45756",
            zorder=3,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(np.arange(len(hist_plot)), hist_plot["plot_label"])
    axis.invert_yaxis()
    axis.set(
        title="Individual effects for every test-set histology",
        xlabel="Preoperative RT + surgery minus surgery risk (percentage points)",
        ylabel="Histology",
    )
    axis.grid(axis="x", alpha=0.25)
    axis.scatter([], [], color="#4C78A8", label="Individual patient")
    axis.scatter([], [], marker="D", color="#E45756", label="Weighted mean")
    axis.legend(frameon=False)
    _save(fig, figures / "all_histology_individual_effects")

    # Factual trajectory alignment and factual-risk diagnostic.
    observed_cols = ["patient_id", "interval", "local_recurrence", "observed", "at_risk_local_recurrence", "surgery_this_interval", "preoperative_rt_this_interval"]
    observed = person[observed_cols].copy()
    factual = trajectories.merge(individual[["patient_id", "observed_regimen"]], on="patient_id", how="left")
    factual = factual.loc[factual["regimen"].eq(factual["observed_regimen"])].merge(observed, on=["patient_id", "interval"], how="left")
    factual = factual.loc[factual["observed"].eq(1) & factual["at_risk_local_recurrence"].eq(1)].copy()
    factual["interval_risk_clipped"] = factual["interval_risk"].clip(1.0e-8, 1 - 1.0e-8)
    factual["binary_cross_entropy"] = -(factual["local_recurrence"] * np.log(factual["interval_risk_clipped"]) + (1 - factual["local_recurrence"]) * np.log(1 - factual["interval_risk_clipped"]))
    patient_risk = factual.groupby(["patient_id", "causal_weight"], as_index=False)["binary_cross_entropy"].mean()
    plugin = _weighted_mean(patient_risk["binary_cross_entropy"], patient_risk["causal_weight"])
    if_diagnostic = pd.DataFrame([{
        "n": len(patient_risk),
        "weighted_factual_binary_cross_entropy": plugin,
        "uncertainty_interval": "not_computed",
        "note": "No valid influence-function or bootstrap uncertainty was computed for this longitudinal model.",
    }])
    if_diagnostic.to_csv(output / "influence_function_factual_risk.csv", index=False)

    alignment_rows = []
    for feature, generated in (("surgery_this_interval", "generated_surgery_this_interval"), ("preoperative_rt_this_interval", "generated_preoperative_rt_this_interval")):
        x = factual[feature].dropna().to_numpy(float)
        y = factual[generated].dropna().to_numpy(float)
        alignment_rows.append({"feature": feature, "n_observed": len(x), "n_generated": len(y), "wasserstein_1": float(wasserstein_distance(x, y)), "ks_statistic": float(ks_2samp(x, y).statistic)})
    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(output / "distribution_alignment.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, row in zip(axes, alignment_rows):
        feature = row["feature"]
        generated = f"generated_{feature}"
        axis.hist(factual[feature].dropna(), bins=20, alpha=0.5, density=True, label="Observed", color="#4C78A8")
        axis.hist(factual[generated].dropna(), bins=20, alpha=0.5, density=True, label="Generated", color="#E45756")
        axis.set(title=f"{feature}\nWasserstein={row['wasserstein_1']:.3f}; KS={row['ks_statistic']:.3f}")
    axes[0].legend(frameon=False)
    _save(fig, figures / "distribution_alignment")

    kl_rows = []
    for regimen, group in factual.groupby("regimen"):
        observed_rate = _weighted_mean(group["local_recurrence"], group["causal_weight"])
        predicted_rate = _weighted_mean(group["interval_risk_clipped"], group["causal_weight"])
        kl_rows.append({"regimen": regimen, "observed_event_rate": observed_rate, "predicted_event_rate": predicted_rate, "bernoulli_kl_observed_to_predicted": _bernoulli_kl(observed_rate, predicted_rate)})
    kl = pd.DataFrame(kl_rows)
    kl.to_csv(output / "kl_alignment.csv", index=False)

    factual_metrics = pd.DataFrame([report["test"]["factual"]])
    factual_metrics.to_csv(output / "test_factual_metrics.csv", index=False)
    split_estimands.to_csv(output / "dev_test_estimand_comparison.csv", index=False)
    summary = {
        "assessment": "paper_style_w6_descriptive_model_based",
        "checkpoint": report["checkpoint"],
        "cohort": report["design"]["cohort"],
        "splits": report["design"]["splits"],
        "interval_end_months": [int(interval_end_month[key]) for key in sorted(interval_end_month)],
        "interval_resolution": "three_month_intervals_derived_from_person_period_data",
        "test_balance_supported": report["causal_interpretation"]["test_effect_balance"]["supported"],
        "test_maximum_weighted_smd": report["causal_interpretation"]["test_effect_balance"]["maximum_absolute_smd_weighted"],
        "effect_summary": report["effect_summary"],
        "bootstrap": "disabled_by_w6_analysis_contract",
        "interpretation_boundary": "All counterfactual effects are descriptive and model-based; the untouched test balance gate failed and no uncertainty interval was computed.",
    }
    (output / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    manifest = {"report": "report.json", "artifacts": artifacts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


