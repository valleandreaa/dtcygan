#!/usr/bin/env python3
"""Comprehensive validation suite for temporal CycleGAN counterfactuals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

CONSTRAINT_LIMITS: Dict[str, tuple[float, float]] = {
    "tumour_size_before_treatment_mm": (0.0, 400.0),
    "radiotherapy_total_dose_gy": (0.0, 120.0),
    "radiotherapy_fraction_count": (0.0, 40.0),
    "radiotherapy_dose_per_fraction_gy": (0.0, 8.0),
    "chemotherapy_duration_days": (0.0, 1800.0),
    "chemotherapy_cycles": (0.0, 20.0),
    "number_of_reoperations": (0.0, 10.0),
    "patient_reported_outcome_score": (0.0, 1.0),
}

GUIDELINE_RANGES: Dict[str, tuple[float, float]] = {
    "radiotherapy_total_dose_gy": (36.0, 66.0),
    "radiotherapy_fraction_count": (25.0, 33.0),
}

DISTRIBUTION_FEATURES = [
    "tumour_size_before_treatment_mm",
    "radiotherapy_total_dose_gy",
    "radiotherapy_fraction_count",
    "patient_reported_outcome_score",
]

OUTCOME_LABELS = ["DOD", "NED", "AWD"]
TREATMENT_SCENARIOS = ["S", "S+CT", "S+RT", "S+RT+CT"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate synthetic counterfactual trajectories against clinical constraints and empirical data."
    )
    parser.add_argument("--counterfactual", required=True, help="Synthetic counterfactual dataset (JSON).")
    parser.add_argument("--reference", required=True, help="Empirical reference dataset (JSON).")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint containing validation_patient_ids metadata to restrict analysis to the validation fold.",
    )
    parser.add_argument("--output", help="Optional path to write a JSON validation report.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for risk estimates (default: 1000).",
    )
    parser.add_argument(
        "--risk-feature",
        default="patient_reported_outcome_score",
        help="Feature to use when computing scenario risks (default: patient_reported_outcome_score).",
    )
    parser.add_argument("--seed", type=int, help="Optional RNG seed for bootstrapping.")
    return parser


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    patients = payload.get("patients", [])
    if not isinstance(patients, list):
        raise ValueError(f"Dataset at {path} does not contain a 'patients' list.")
    return patients


def load_validation_ids(checkpoint_path: Optional[str | Path]) -> Optional[set[str]]:
    if not checkpoint_path:
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    metadata = ckpt.get("metadata", {})
    ids = metadata.get("validation_patient_ids")
    if not ids:
        return None
    return set(ids)


def filter_patients(patients: List[Dict[str, Any]], keep_ids: Optional[set[str]]) -> List[Dict[str, Any]]:
    if keep_ids is None:
        return patients
    return [patient for patient in patients if patient.get("patient_id") in keep_ids]


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scenario_label(actual: Dict[str, Any]) -> Optional[str]:
    if not actual:
        return None
    surgery = bool(to_float(actual.get("episodes.surgery")) or 0)
    chemo = bool(to_float(actual.get("episodes.chemotherapy")) or 0)
    radio = bool(to_float(actual.get("episodes.radiotherapy")) or 0)
    mapping = {
        (True, False, False): "S",
        (True, True, False): "S+CT",
        (True, False, True): "S+RT",
        (True, True, True): "S+RT+CT",
    }
    return mapping.get((surgery, chemo, radio), None)


def final_record(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    return records[-1] if records else {}


def evaluate_constraints(patients: List[Dict[str, Any]]) -> Dict[str, Any]:
    violation_counts: Dict[str, int] = defaultdict(int)
    total_steps = 0
    invalid_steps = 0
    patient_invalid = 0

    for patient in patients:
        patient_flag = False
        for record in patient.get("treatment", []):
            total_steps += 1
            for feature, (lo, hi) in CONSTRAINT_LIMITS.items():
                value = to_float(record.get(feature))
                if value is None:
                    continue
                if value < lo or value > hi:
                    violation_counts[feature] += 1
                    invalid_steps += 1
                    patient_flag = True
            for feature, (lo, hi) in GUIDELINE_RANGES.items():
                value = to_float(record.get(feature))
                if value is None:
                    continue
                if value < lo or value > hi:
                    violation_counts[f"guideline::{feature}"] += 1
                    invalid_steps += 1
                    patient_flag = True
        if patient_flag:
            patient_invalid += 1

    return {
        "total_timesteps": total_steps,
        "invalid_timesteps": invalid_steps,
        "invalid_fraction": (invalid_steps / total_steps) if total_steps else 0.0,
        "invalid_patients": patient_invalid,
        "invalid_patient_fraction": (patient_invalid / len(patients)) if patients else 0.0,
        "violations": dict(sorted(violation_counts.items(), key=lambda kv: kv[0])),
    }


def extract_feature(values: List[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def iter_feature_values(patients: List[Dict[str, Any]], feature: str, section: str = "treatment") -> List[float]:
    values: List[float] = []
    for patient in patients:
        for record in patient.get(section, []):
            value = to_float(record.get(feature))
            if value is not None:
                values.append(value)
    return values


def wasserstein_distance(arr1: np.ndarray, arr2: np.ndarray, bins: int = 512) -> float:
    if arr1.size == 0 or arr2.size == 0:
        return float("nan")
    min_v = min(arr1.min(), arr2.min())
    max_v = max(arr1.max(), arr2.max())
    if math.isclose(min_v, max_v):
        return 0.0
    grid = np.linspace(min_v, max_v, bins)
    cdf1 = np.searchsorted(np.sort(arr1), grid, side="right") / arr1.size
    cdf2 = np.searchsorted(np.sort(arr2), grid, side="right") / arr2.size
    return float(np.trapz(np.abs(cdf1 - cdf2), grid))


def ks_2sample(arr1: np.ndarray, arr2: np.ndarray) -> tuple[float, float]:
    if arr1.size == 0 or arr2.size == 0:
        return float("nan"), float("nan")
    data_all = np.concatenate([arr1, arr2])
    data_all.sort()
    cdf1 = np.searchsorted(np.sort(arr1), data_all, side="right") / arr1.size
    cdf2 = np.searchsorted(np.sort(arr2), data_all, side="right") / arr2.size
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    n1, n2 = arr1.size, arr2.size
    n_eff = math.sqrt(n1 * n2 / (n1 + n2))
    p_val = min(1.0, 2.0 * math.exp(-2.0 * (d_stat * n_eff) ** 2))
    return d_stat, p_val


def compute_distribution_metrics(
    synthetic_patients: List[Dict[str, Any]],
    reference_patients: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for feature in DISTRIBUTION_FEATURES:
        syn_array = extract_feature(iter_feature_values(synthetic_patients, feature))
        ref_array = extract_feature(iter_feature_values(reference_patients, feature))
        if syn_array.size == 0 or ref_array.size == 0:
            metrics[feature] = {"w1": float("nan"), "ks_stat": float("nan"), "ks_pvalue": float("nan")}
            continue
        w1 = wasserstein_distance(syn_array, ref_array)
        ks_stat, ks_pvalue = ks_2sample(syn_array, ref_array)
        metrics[feature] = {"w1": w1, "ks_stat": ks_stat, "ks_pvalue": ks_pvalue}
    return metrics


def gather_scenario_samples(
    patients: List[Dict[str, Any]],
) -> Dict[str, Dict[str, np.ndarray]]:
    scenario_outcomes: Dict[str, Dict[str, List[float]]] = {
        scen: {out: [] for out in OUTCOME_LABELS} for scen in TREATMENT_SCENARIOS
    }
    for patient in patients:
        actual_final = final_record(patient.get("actual_treatment", []))
        scenario = scenario_label(actual_final)
        if scenario not in scenario_outcomes:
            continue
        last_treatment = final_record(patient.get("treatment", []))
        status = last_treatment.get("status_at_last_follow_up")
        if status not in OUTCOME_LABELS:
            continue
        for outcome in OUTCOME_LABELS:
            scenario_outcomes[scenario][outcome].append(1.0 if status == outcome else 0.0)
    return {
        scen: {out: extract_feature(vals) for out, vals in outcomes.items()}
        for scen, outcomes in scenario_outcomes.items()
    }


def compute_scenario_metrics(
    synthetic_patients: List[Dict[str, Any]],
    reference_patients: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    syn_samples = gather_scenario_samples(synthetic_patients)
    ref_samples = gather_scenario_samples(reference_patients)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario in TREATMENT_SCENARIOS:
        scenario_metrics: Dict[str, Dict[str, float]] = {}
        for outcome in OUTCOME_LABELS:
            syn_arr = syn_samples.get(scenario, {}).get(outcome, np.array([]))
            ref_arr = ref_samples.get(scenario, {}).get(outcome, np.array([]))
            if syn_arr.size == 0 or ref_arr.size == 0:
                scenario_metrics[outcome] = {"w1": float("nan"), "ks_stat": float("nan"), "ks_pvalue": float("nan")}
                continue
            w1 = abs(syn_arr.mean() - ref_arr.mean())
            ks_stat, ks_pvalue = ks_2sample(syn_arr, ref_arr)
            scenario_metrics[outcome] = {"w1": w1, "ks_stat": ks_stat, "ks_pvalue": ks_pvalue}
        results[scenario] = scenario_metrics
    return results


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def compute_kl_metrics(
    synthetic_patients: List[Dict[str, Any]],
    reference_patients: List[Dict[str, Any]],
) -> Dict[str, float]:
    def joint_counts(patients: List[Dict[str, Any]]) -> np.ndarray:
        counts = np.zeros((len(TREATMENT_SCENARIOS), len(OUTCOME_LABELS)), dtype=float)
        for patient in patients:
            scenario = scenario_label(final_record(patient.get("actual_treatment", [])))
            if scenario not in TREATMENT_SCENARIOS:
                continue
            status = final_record(patient.get("treatment", [])).get("status_at_last_follow_up")
            if status not in OUTCOME_LABELS:
                continue
            i = TREATMENT_SCENARIOS.index(scenario)
            j = OUTCOME_LABELS.index(status)
            counts[i, j] += 1
        if counts.sum() == 0:
            return counts
        return counts / counts.sum()

    syn_joint = joint_counts(synthetic_patients)
    ref_joint = joint_counts(reference_patients)
    joint_kl = kl_divergence(syn_joint.flatten(), ref_joint.flatten()) if syn_joint.sum() and ref_joint.sum() else float("nan")

    syn_marginal = syn_joint.sum(axis=0)
    ref_marginal = ref_joint.sum(axis=0)
    marginal_kl = kl_divergence(syn_marginal, ref_marginal) if syn_marginal.sum() and ref_marginal.sum() else float("nan")

    return {"joint": joint_kl, "marginal_outcome": marginal_kl}


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: Optional[int]) -> tuple[float, tuple[float, float], float, float]:
    if values.size == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    baseline = float(values.mean())
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, values.size, size=values.size)
        samples[i] = values[idx].mean()
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    ci_width = (ci_high - ci_low)
    ci_width_pct = (ci_width / baseline * 100.0) if baseline else float("nan")
    se = float(samples.std(ddof=1))
    return baseline, (float(ci_low), float(ci_high)), ci_width_pct, se


def compute_scenario_risks(
    patients: List[Dict[str, Any]],
    feature: str,
    n_boot: int,
    seed: Optional[int],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for scenario in TREATMENT_SCENARIOS:
        values: List[float] = []
        for patient in patients:
            if scenario_label(final_record(patient.get("actual_treatment", []))) != scenario:
                continue
            last = final_record(patient.get("treatment", []))
            value = to_float(last.get(feature))
            if value is not None:
                values.append(value)
        arr = extract_feature(values)
        mean_val, (ci_low, ci_high), ci_width_pct, se = bootstrap_mean(arr, n_boot, seed)
        results[scenario] = {
            "risk": mean_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width_percent": ci_width_pct,
            "se": se,
            "sample_size": int(arr.size),
        }
    return results


def generate_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    constraints = summary["constraints"]
    lines.append("Hard constraint violations:")
    lines.append(
        f"  Invalid timesteps: {constraints['invalid_timesteps']} / {constraints['total_timesteps']} "
        f"({constraints['invalid_fraction']*100:.2f}%)"
    )
    lines.append(
        f"  Invalid patients: {constraints['invalid_patients']} / {summary['counts']['patients']} "
        f"({constraints['invalid_patient_fraction']*100:.2f}%)"
    )
    if constraints["violations"]:
        for feature, count in constraints["violations"].items():
            lines.append(f"    - {feature}: {count}")

    lines.append("")
    lines.append("Distributional alignment (W1 distance, KS statistic, KS p-value):")
    for feature, stats in summary["distribution_metrics"].items():
        lines.append(
            f"  {feature}: W1={stats['w1']:.4f}, KS={stats['ks_stat']:.4f}, p={stats['ks_pvalue']:.3e}"
        )

    lines.append("")
    lines.append("Scenario-outcome comparison (|Δmean| ~ W1 surrogate, KS):")
    for scenario, outcomes in summary["scenario_metrics"].items():
        lines.append(f"  {scenario}:")
        for outcome, stats in outcomes.items():
            lines.append(
                f"    {outcome}: |Δ|={stats['w1']:.4f}, KS={stats['ks_stat']:.4f}, p={stats['ks_pvalue']:.3e}"
            )

    lines.append("")
    lines.append(
        "KL divergence: joint={joint:.4f}, marginal_outcome={marginal:.4f}".format(
            joint=summary["kl_metrics"]["joint"],
            marginal=summary["kl_metrics"]["marginal_outcome"],
        )
    )

    lines.append("")
    lines.append("Scenario risk summary (bootstrap means):")
    for scenario, stats in summary["scenario_risks"].items():
        lines.append(
            f"  {scenario}: risk={stats['risk']:.4f} CI=[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}] "
            f"width={stats['ci_width_percent']:.1f}% SE={stats['se']:.4f} N={stats['sample_size']}"
        )

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    cf_patients = load_dataset(args.counterfactual)
    ref_patients = load_dataset(args.reference)

    validation_ids = load_validation_ids(args.checkpoint)
    cf_patients = filter_patients(cf_patients, validation_ids)
    ref_patients = filter_patients(ref_patients, validation_ids)

    summary: Dict[str, Any] = {
        "counts": {
            "patients": len(cf_patients),
            "reference_patients": len(ref_patients),
        },
        "constraints": evaluate_constraints(cf_patients),
        "distribution_metrics": compute_distribution_metrics(cf_patients, ref_patients),
        "scenario_metrics": compute_scenario_metrics(cf_patients, ref_patients),
        "kl_metrics": compute_kl_metrics(cf_patients, ref_patients),
        "scenario_risks": compute_scenario_risks(
            cf_patients, args.risk_feature, args.bootstrap, args.seed
        ),
    }

    report = generate_report(summary)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nValidation report written to {out_path}")


if __name__ == "__main__":
    main()
