#!/usr/bin/env python3
"""Comprehensive validation suite for temporal CycleGAN counterfactuals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch



from dtcygan.training import (
    Config,
    SyntheticSequenceDataset,
    LSTMGenerator,
    resolve_device,
    set_seed,
)

from dtcygan.eval_utils import (
    load_dataset_payload,
    load_dataset,
    filter_patients,
    ensure_probability_metadata,
    load_checkpoint_bundle,
    build_counterfactual_patients,
    bootstrap_two_sample_ci,
    wasserstein_1d,
    ks_two_sample,
    final_record,
)

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


SCENARIO_CONDITIONS: Dict[str, Dict[str, int]] = {
    "S": {
        "episodes.surgery": 1,
        "episodes.chemotherapy": 0,
        "episodes.radiotherapy": 0,
    },
    "S+CT": {
        "episodes.surgery": 1,
        "episodes.chemotherapy": 1,
        "episodes.radiotherapy": 0,
    },
    "S+RT": {
        "episodes.surgery": 1,
        "episodes.chemotherapy": 0,
        "episodes.radiotherapy": 1,
    },
    "S+RT+CT": {
        "episodes.surgery": 1,
        "episodes.chemotherapy": 1,
        "episodes.radiotherapy": 1,
    },
}


PROBABILITY_ONLY_FEATURES = {
    "status_at_last_follow_up",
    "treatment_endpoint_category",
}


def build_arg_parser() -> argparse.ArgumentParser:
    '''
    Build command line parser for validation CLI.

    args:
    - None

    return:
    - parser: configured argument parser [argparse.ArgumentParser]
    '''
    parser = argparse.ArgumentParser(
        description="Validate synthetic counterfactual trajectories against clinical.4"
    )
    parser.add_argument(
        "--dataset",
        default=str(Path("data") / "syntects.json"),
        help="Dataset JSON used to rebuild counterfactuals (default: data/syntects.json).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path("data") / "models" / "dtcygan_20250928_135339.pt"),
        help="Checkpoint containing generator weights and validation metadata (default: data/models/dtcygan_20250928_135339.pt).",
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
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Stochastic counterfactual samples per patient when generating internally (default: 32).",
    )
    parser.add_argument(
        "--natural-positive-label",
        default="ALL",
        help="Outcome label for natural experiment analysis (use ALL to evaluate every available label).",
    )
    parser.add_argument(
        "--natural-match-cols",
        nargs="*",
        help="Optional columns used to match cohorts in natural experiment analysis; defaults to a predefined set.",
    )
    parser.add_argument("--seed", type=int, help="Optional RNG seed for bootstrapping.")
    return parser


def load_dataset_payload(path: str | Path) -> Dict[str, Any]:
    '''
    Load dataset JSON payload and verify structure.

    args:
    - path: dataset JSON path [str | Path]

    return:
    - payload: parsed dataset dictionary [Dict[str, Any]]
    '''
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or "patients" not in payload:
        raise ValueError(f"Dataset at {path} must contain a top-level 'patients' key.")
    return payload


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    '''
    Extract patient list from dataset payload.

    args:
    - path: dataset JSON path [str | Path]

    return:
    - patients: list of patient records [List[Dict[str, Any]]]
    '''
    payload = load_dataset_payload(path)
    patients = payload.get("patients", [])
    if not isinstance(patients, list):
        raise ValueError(f"Dataset at {path} does not contain a 'patients' list.")
    return patients


def filter_patients(patients: List[Dict[str, Any]], keep_ids: Optional[set[str]]) -> List[Dict[str, Any]]:
    '''
    Filter patients by requested identifiers.

    args:
    - patients: collection of patient dicts [List[Dict[str, Any]]]
    - keep_ids: identifiers to retain [Optional[set[str]]]

    return:
    - filtered: filtered patient records [List[Dict[str, Any]]]
    '''
    if keep_ids is None:
        return patients
    keep_ids = {str(pid) for pid in keep_ids}
    return [patient for patient in patients if str(patient.get("patient_id")) in keep_ids]

def _format_float(value: float, precision: int = 4) -> str:
    '''
    Render numeric value with consistent precision.

    args:
    - value: numeric input to format [float]
    - precision: decimal places to include [int]

    return:
    - text: formatted representation [str]
    '''
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def _format_ci(ci: Optional[Tuple[float, float]], precision: int = 4) -> str:
    '''
    Format confidence interval values for display.

    args:
    - ci: low and high bounds tuple [Optional[Tuple[float, float]]]
    - precision: decimal places to include [int]

    return:
    - text: formatted bounds string [str]
    '''
    if not ci:
        return "[nan, nan]"
    low, high = ci
    if not np.isfinite(low) or not np.isfinite(high):
        return "[nan, nan]"
    return f"[{low:.{precision}f}, {high:.{precision}f}]"




def extract_validation_ids(metadata: Dict[str, Any]) -> Optional[set[str]]:
    '''
    Pull validation patient identifiers from metadata.

    args:
    - metadata: checkpoint metadata dictionary [Dict[str, Any]]

    return:
    - ids: normalized identifier set [Optional[set[str]]]
    '''
    ids = metadata.get("validation_patient_ids") if metadata else None
    if not ids:
        return None
    return {str(pid) for pid in ids}


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
) -> tuple[Config, Dict[str, Any], LSTMGenerator, Optional[set[str]], torch.device]:
    '''
    Load training checkpoint and rebuild generator bundle.

    args:
    - checkpoint_path: path to checkpoint file [str | Path]

    return:
    - bundle: config, metadata, generator, ids, device tuple [tuple[Config, Dict[str, Any], LSTMGenerator, Optional[set[str]], torch.device]]
    '''
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    metadata: Dict[str, Any] = ckpt.get("metadata") or {}
    validation_ids = extract_validation_ids(metadata)

    clin_dim = metadata.get("clin_dim")
    treat_dim = metadata.get("treat_dim")
    cond_dim = metadata.get("cond_dim")
    if None in {clin_dim, treat_dim, cond_dim}:
        raise ValueError("Checkpoint metadata must include clin_dim, treat_dim, and cond_dim.")

    device = resolve_device(cfg.device)
    generator = LSTMGenerator(clin_dim, cond_dim, cfg.g_hidden, treat_dim, cfg.num_layers).to(device)
    generator.load_state_dict(ckpt["Gx"])
    generator.eval()
    return cfg, metadata, generator, validation_ids, device


def build_counterfactual_patients(
    dataset: SyntheticSequenceDataset,
    generator: LSTMGenerator,
    device: torch.device,
    validation_ids: Optional[set[str]],
    extra_ones: int = 1,
    *,
    status_labels: Optional[List[str]] = None,
    endpoint_labels: Optional[List[str]] = None,
    samples_per_patient: int = 32,
) -> List[Dict[str, Any]]:
    '''
    Generate counterfactual patient trajectories for each scenario.

    args:
    - dataset: dataset wrapper used for sampling [SyntheticSequenceDataset]
    - generator: trained generator network [LSTMGenerator]
    - device: computation device for sampling [torch.device]
    - validation_ids: subset of patient identifiers to include [Optional[set[str]]]
    - extra_ones: number of one-hot perturbations for conditioning [int]
    - status_labels: optional outcome labels override [Optional[List[str]]]
    - endpoint_labels: optional endpoint labels override [Optional[List[str]]]
    - samples_per_patient: stochastic samples per patient [int]

    return:
    - patients: synthetic patient payloads [List[Dict[str, Any]]]
    '''
    generator.eval()

    categorical_features = dataset.treat_categorical
    categorical_map = dataset.treat_categories
    ordinal_lookup = {
        feature: [(float(value), label) for label, value in mapping.items()]
        for feature, mapping in dataset.treat_ord_maps.items()
    }

    status_label_list = (
        list(status_labels)
        if status_labels is not None
        else list(dataset.treat_categories.get("status_at_last_follow_up", OUTCOME_LABELS))
    )
    endpoint_label_list = (
        list(endpoint_labels)
        if endpoint_labels is not None
        else list(dataset.treat_categories.get("treatment_endpoint_category", []))
    )

    def decode_generated(feature: str, raw: float) -> tuple[Any, float]:
        raw_value = float(raw)
        if feature in PROBABILITY_ONLY_FEATURES:
            return raw_value, raw_value
        if feature in categorical_features:
            categories = categorical_map.get(feature, [])
            if categories:
                code = int(round(raw_value))
                code = max(0, min(code, len(categories) - 1))
                return categories[code], raw_value
            return int(round(raw_value)), raw_value
        if feature in ordinal_lookup:
            candidates = ordinal_lookup[feature]
            if candidates:
                closest_value, closest_label = min(candidates, key=lambda item: abs(item[0] - raw_value))
                return closest_label, raw_value
        return raw_value, raw_value

    def logits_to_probabilities(feature: str, raw_value: float) -> Dict[str, float]:
        categories = categorical_map.get(feature, [])
        if not categories:
            return {}
        if len(categories) == 1:
            return {categories[0]: 1.0}
        if len(categories) == 2:
            prob_positive = float(np.clip(raw_value, 0.0, 1.0))
            return {
                categories[0]: float(1.0 - prob_positive),
                categories[1]: prob_positive,
            }
        positions = np.arange(len(categories), dtype=float)
        scores = -np.square(raw_value - positions)
        scores -= scores.max()
        weights = np.exp(scores)
        denom = weights.sum()
        if denom <= 0:
            return {cat: 0.0 for cat in categories}
        weights /= denom
        return {cat: float(w) for cat, w in zip(categories, weights)}

    results: List[Dict[str, Any]] = []

    for idx, patient in enumerate(dataset.patients):
        patient_id = str(patient.get("patient_id", f"P{idx:05d}"))
        if validation_ids is not None and patient_id not in validation_ids:
            continue

        sample = dataset[idx]
        x_clin = sample["x_clin"].unsqueeze(0).to(device)
        mask_clin = sample["mask_clin"].unsqueeze(0).to(device)
        tr_actual = sample["actual_treatment"].unsqueeze(0).to(device)
        mask_actual = sample["mask_actual"].unsqueeze(0).to(device)
        mask_treat = sample["mask_treat"].unsqueeze(0).to(device)

        step_mask = ((mask_clin.sum(dim=2, keepdim=True) + mask_treat.sum(dim=2, keepdim=True)) > 0).float()
        lengths = step_mask.squeeze(-1).sum(dim=1).clamp(min=1).long()

        x_clin = x_clin * mask_clin
        mask_clin = mask_clin * step_mask
        mask_treat = mask_treat * step_mask
        tr_actual = tr_actual * mask_actual

        actual_np = tr_actual.squeeze(0).cpu().numpy() if tr_actual.numel() else np.zeros((0, 0))
        actual_mask_np = mask_actual.squeeze(0).cpu().numpy() if mask_actual.numel() else np.zeros((0, 0))

        base_cond_map: Dict[str, float] = {feature: 0.0 for feature in dataset.cond_columns}
        if actual_np.size and actual_mask_np.size:
            valid_rows = actual_mask_np.sum(axis=1) > 0
            if valid_rows.any():
                base_idx = int(np.argmax(valid_rows))
                row_vals = actual_np[base_idx]
                row_mask = actual_mask_np[base_idx]
                for feat_idx, feature in enumerate(dataset.cond_columns):
                    if feat_idx < row_vals.shape[0] and row_mask[feat_idx] > 0:
                        base_cond_map[feature] = float(row_vals[feat_idx])

        observed_flag_map = {
            "actual_surgery": base_cond_map.get("episodes.surgery", 0.0),
            "actual_chemotherapy": base_cond_map.get("episodes.chemotherapy", 0.0),
            "actual_radiotherapy": base_cond_map.get("episodes.radiotherapy", 0.0),
        }

        seq_len = x_clin.size(1)
        step_mask_np = step_mask.squeeze(0).squeeze(-1).cpu().numpy().astype(bool)

        for scenario_name, overrides in SCENARIO_CONDITIONS.items():
            scenario_cond_map = base_cond_map.copy()
            scenario_cond_map.update({k: float(v) for k, v in overrides.items() if k in scenario_cond_map})

            cond_vector = torch.tensor(
                [float(scenario_cond_map.get(feature, 0.0)) for feature in dataset.cond_columns],
                dtype=torch.float32,
                device=device,
            )
            cond_tensor = cond_vector.view(1, 1, -1).repeat(1, seq_len, 1)
            cond_tensor = cond_tensor * step_mask
            cond_mask = torch.ones_like(cond_tensor) * step_mask

            with torch.no_grad():
                fake_treat, _ = generator(x_clin, mask_clin, cond_tensor, cond_mask, lengths)
                fake_treat = fake_treat * mask_treat

            fake_np = fake_treat.squeeze(0).cpu().numpy()
            mask_np = mask_treat.squeeze(0).cpu().numpy()

            treatment_records: List[Dict[str, Any]] = []
            scenario_logits: Dict[str, float] = {}
            for step_idx, step_valid in enumerate(step_mask_np):
                if not bool(step_valid):
                    continue
                step_record: Dict[str, Any] = {}
                for feat_idx, feature in enumerate(dataset.treat_columns):
                    if mask_np[step_idx, feat_idx] <= 0:
                        continue
                    decoded_value, raw_value = decode_generated(feature, fake_np[step_idx, feat_idx])
                    if feature in PROBABILITY_ONLY_FEATURES:
                        scenario_logits[feature] = raw_value
                        step_record[f"{feature}_logit"] = raw_value
                        continue
                    step_record[feature] = decoded_value
                step_record.update(observed_flag_map)
                step_record.update(
                    {
                        "scenario_surgery": scenario_cond_map.get("episodes.surgery", 0.0),
                        "scenario_chemotherapy": scenario_cond_map.get("episodes.chemotherapy", 0.0),
                        "scenario_radiotherapy": scenario_cond_map.get("episodes.radiotherapy", 0.0),
                        "treatment_scenario": scenario_name,
                    }
                )
                step_record["_id"] = patient_id
                step_record["timestep"] = step_idx
                treatment_records.append(step_record)

            actual_records: List[Dict[str, Any]] = []
            for step_idx, step_valid in enumerate(step_mask_np):
                if not bool(step_valid):
                    continue
                cond_record = {
                    feature: float(scenario_cond_map.get(feature, 0.0))
                    for feature in dataset.cond_columns
                }
                cond_record["treatment_scenario"] = scenario_name
                cond_record["_id"] = patient_id
                cond_record["timestep"] = step_idx
                actual_records.append(cond_record)

            status_probs: Dict[str, float] = {label: 0.0 for label in status_label_list}
            status_logit = scenario_logits.get("status_at_last_follow_up")
            if status_logit is not None:
                derived = logits_to_probabilities("status_at_last_follow_up", status_logit)
                for label, prob in derived.items():
                    status_probs[label] = prob

            endpoint_probs: Dict[str, float] = {label: 0.0 for label in endpoint_label_list}
            endpoint_logit = scenario_logits.get("treatment_endpoint_category")
            if endpoint_logit is not None:
                derived = logits_to_probabilities("treatment_endpoint_category", endpoint_logit)
                for label, prob in derived.items():
                    endpoint_probs[label] = prob

            results.append(
                {
                    "patient_id": patient_id,
                    "scenario": scenario_name,
                    "clinical": patient.get("clinical", []),
                    "treatment": treatment_records,
                    "actual_treatment": actual_records,
                    "status_probabilities": status_probs,
                    "endpoint_probabilities": endpoint_probs,
                    "samples_per_patient": 1,
                }
            )

    return results


def to_float(value: Any) -> Optional[float]:
    '''
    Convert arbitrary value into float when possible.

    args:
    - value: object to convert [Any]

    return:
    - numeric: parsed float or None [Optional[float]]
    '''
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scenario_label(actual: Dict[str, Any]) -> Optional[str]:
    '''
    Map actual treatment flags to scenario label.

    args:
    - actual: conditioning record with episode flags [Dict[str, Any]]

    return:
    - label: scenario identifier [Optional[str]]
    '''
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
    '''
    Return final element from treatment record sequence.

    args:
    - records: iterable of treatment dictionaries [Iterable[Dict[str, Any]]]

    return:
    - record: last record or empty dict [Dict[str, Any]]
    '''
    records = list(records)
    return records[-1] if records else {}


def evaluate_constraints(patients: List[Dict[str, Any]]) -> Dict[str, Any]:
    '''
    Count hard constraint violations across generated treatments.

    args:
    - patients: synthetic patient payloads to analyse [List[Dict[str, Any]]]

    return:
    - summary: aggregate violation metrics [Dict[str, Any]]
    '''
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
    '''
    Convert list of numbers into finite numpy vector.

    args:
    - values: collected scalar measurements [List[float]]

    return:
    - array: finite numpy array of values [np.ndarray]
    '''
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def iter_feature_values(patients: List[Dict[str, Any]], feature: str, section: str = "treatment") -> List[float]:
    '''
    Collect numeric feature values from patient records.

    args:
    - patients: sequence of patient dictionaries [List[Dict[str, Any]]]
    - feature: feature key to extract [str]
    - section: record section to traverse [str]

    return:
    - values: extracted feature values [List[float]]
    '''
    values: List[float] = []
    for patient in patients:
        for record in patient.get(section, []):
            value = to_float(record.get(feature))
            if value is not None:
                values.append(value)
    return values


def _wasserstein_distance_numpy(arr1: np.ndarray, arr2: np.ndarray, bins: int = 512) -> float:
    '''
    Approximate 1D Wasserstein distance via discretised CDFs.

    args:
    - arr1: first sample array [np.ndarray]
    - arr2: second sample array [np.ndarray]
    - bins: discretisation bins for integration [int]

    return:
    - distance: estimated Wasserstein distance [float]
    '''
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


def _ks_2sample_numpy(arr1: np.ndarray, arr2: np.ndarray) -> tuple[float, float]:
    '''
    Compute Kolmogorov-Smirnov statistic without SciPy.

    args:
    - arr1: first sample array [np.ndarray]
    - arr2: second sample array [np.ndarray]

    return:
    - result: KS statistic and p-value [tuple[float, float]]
    '''
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
    n_boot: int,
    seed: Optional[int],
) -> Dict[str, Dict[str, float]]:
    '''
    Compare feature distributions between synthetic and reference cohorts.

    args:
    - synthetic_patients: generated patient records [List[Dict[str, Any]]]
    - reference_patients: clinical patient records [List[Dict[str, Any]]]
    - n_boot: bootstrap iterations for intervals [int]
    - seed: optional RNG seed [Optional[int]]

    return:
    - metrics: distribution alignment statistics [Dict[str, Dict[str, float]]]
    '''
    def collect_values(
        patients: List[Dict[str, Any]],
        feature: str,
        scenario: str,
        *,
        synthetic: bool,
    ) -> np.ndarray:
        values: List[float] = []
        for patient in patients:
            if synthetic:
                patient_scenario = patient.get("scenario")
            else:
                patient_scenario = scenario_label(final_record(patient.get("actual_treatment", [])))
            if patient_scenario != scenario:
                continue
            for record in patient.get("treatment", []):
                value = to_float(record.get(feature))
                if value is not None:
                    values.append(value)
        return extract_feature(values)

    metrics: Dict[str, Dict[str, float]] = {}
    for idx, feature in enumerate(DISTRIBUTION_FEATURES):
        syn_chunks: List[np.ndarray] = []
        ref_chunks: List[np.ndarray] = []
        for scenario in TREATMENT_SCENARIOS:
            syn_vals = collect_values(synthetic_patients, feature, scenario, synthetic=True)
            ref_vals = collect_values(reference_patients, feature, scenario, synthetic=False)
            if syn_vals.size == 0 or ref_vals.size == 0:
                continue
            syn_chunks.append(syn_vals)
            ref_chunks.append(ref_vals)

        syn_array = np.concatenate(syn_chunks) if syn_chunks else np.array([], dtype=float)
        ref_array = np.concatenate(ref_chunks) if ref_chunks else np.array([], dtype=float)
        if syn_array.size == 0 or ref_array.size == 0:
            metrics[feature] = {
                "w1": float("nan"),
                "w1_ci": (float("nan"), float("nan")),
                "ks_stat": float("nan"),
                "ks_pvalue": float("nan"),
            }
            continue

        w1 = wasserstein_1d(syn_array, ref_array)
        w1_ci = bootstrap_two_sample_ci(
            syn_array,
            ref_array,
            n_boot,
            None if seed is None else seed + idx * 4,
            lambda a, b: wasserstein_1d(a, b),
        )

        ks_stat, ks_pvalue = ks_two_sample(syn_array, ref_array)

        metrics[feature] = {
            "w1": w1,
            "w1_ci": w1_ci,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
        }
    return metrics


def gather_scenario_samples(
    patients: List[Dict[str, Any]],
    labels: List[str],
    probability_key: str,
    field_name: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    '''
    Aggregate scenario-specific outcome samples for metrics.

    args:
    - patients: patient records to examine [List[Dict[str, Any]]]
    - labels: outcome labels of interest [List[str]]
    - probability_key: key for stored probabilities [str]
    - field_name: fallback deterministic field [str]

    return:
    - samples: scenario to label sample mapping [Dict[str, Dict[str, np.ndarray]]]
    '''
    scenario_outcomes: Dict[str, Dict[str, List[float]]] = {
        scen: {label: [] for label in labels} for scen in TREATMENT_SCENARIOS
    }
    for patient in patients:
        actual_final = final_record(patient.get("actual_treatment", []))
        scenario = scenario_label(actual_final)
        if scenario not in scenario_outcomes:
            continue

        probs = patient.get(probability_key)
        if isinstance(probs, dict) and probs:
            for label in labels:
                scenario_outcomes[scenario][label].append(float(probs.get(label, 0.0)))
            continue

        last_treatment = final_record(patient.get("treatment", []))
        observed_label = last_treatment.get(field_name)
        for label in labels:
            scenario_outcomes[scenario][label].append(1.0 if observed_label == label else 0.0)

    return {
        scen: {label: extract_feature(vals) for label, vals in outcomes.items()}
        for scen, outcomes in scenario_outcomes.items()
    }


def compute_scenario_metrics(
    synthetic_patients: List[Dict[str, Any]],
    reference_patients: List[Dict[str, Any]],
    labels: List[str],
    probability_key: str,
    field_name: str,
    n_boot: int,
    seed: Optional[int],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    '''
    Evaluate per-scenario outcome alignment metrics.

    args:
    - synthetic_patients: generated patient set [List[Dict[str, Any]]]
    - reference_patients: observed patient set [List[Dict[str, Any]]]
    - labels: outcome labels to score [List[str]]
    - probability_key: key for probability metadata [str]
    - field_name: deterministic fallback field [str]
    - n_boot: bootstrap iterations for CIs [int]
    - seed: optional RNG seed [Optional[int]]

    return:
    - metrics: nested scenario metrics dictionary [Dict[str, Dict[str, Dict[str, Any]]]]
    '''
    syn_samples = gather_scenario_samples(synthetic_patients, labels, probability_key, field_name)
    ref_samples = gather_scenario_samples(reference_patients, labels, probability_key, field_name)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scen_idx, scenario in enumerate(TREATMENT_SCENARIOS):
        scenario_metrics: Dict[str, Dict[str, Any]] = {}
        for out_idx, label in enumerate(labels):
            syn_arr = syn_samples.get(scenario, {}).get(label, np.array([]))
            ref_arr = ref_samples.get(scenario, {}).get(label, np.array([]))
            entry: Dict[str, Any] = {
                "sample_sizes": {
                    "synthetic": int(syn_arr.size),
                    "reference": int(ref_arr.size),
                }
            }
            if syn_arr.size == 0 or ref_arr.size == 0:
                entry.update(
                    {
                        "delta_mean": float("nan"),
                        "delta_ci": (float("nan"), float("nan")),
                        "ks_stat": float("nan"),
                        "ks_pvalue": float("nan"),
                    }
                )
                scenario_metrics[label] = entry
                continue

            delta = float(abs(syn_arr.mean() - ref_arr.mean()))
            delta_ci = bootstrap_two_sample_ci(
                syn_arr,
                ref_arr,
                n_boot,
                None if seed is None else seed + scen_idx * 64 + out_idx * 4,
                lambda a, b: float(abs(a.mean() - b.mean())),
            )

            ks_stat, ks_pvalue = ks_two_sample(syn_arr, ref_arr)

            entry.update(
                {
                    "delta_mean": delta,
                    "delta_ci": delta_ci,
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pvalue,
                }
            )
            scenario_metrics[label] = entry
        results[scenario] = scenario_metrics
    return results


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    '''
    Compute KL divergence between two discrete distributions.

    args:
    - p: baseline probability mass [np.ndarray]
    - q: comparison probability mass [np.ndarray]

    return:
    - value: KL divergence estimate [float]
    '''
    eps = 1e-12
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def compute_kl_metrics(
    synthetic_patients: List[Dict[str, Any]],
    reference_patients: List[Dict[str, Any]],
    n_boot: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    '''
    Measure KL divergence between synthetic and reference joint outcomes.

    args:
    - synthetic_patients: generated patient records [List[Dict[str, Any]]]
    - reference_patients: observed patient records [List[Dict[str, Any]]]
    - n_boot: bootstrap iterations for confidence intervals [int]
    - seed: optional RNG seed [Optional[int]]

    return:
    - metrics: joint and marginal KL statistics [Dict[str, Any]]
    '''
    def joint_counts(patients: List[Dict[str, Any]]) -> np.ndarray:
        counts = np.zeros((len(TREATMENT_SCENARIOS), len(OUTCOME_LABELS)), dtype=float)
        for patient in patients:
            scenario = scenario_label(final_record(patient.get("actual_treatment", [])))
            if scenario not in TREATMENT_SCENARIOS:
                continue
            
            # Check if this is a synthetic patient with probabilities
            status_probs = patient.get("status_probabilities")
            if status_probs and isinstance(status_probs, dict):
                # For synthetic patients, use probabilistic outcomes
                i = TREATMENT_SCENARIOS.index(scenario)
                for status_label, prob in status_probs.items():
                    if status_label in OUTCOME_LABELS:
                        j = OUTCOME_LABELS.index(status_label)
                        counts[i, j] += prob
            else:
                # For reference patients, use deterministic outcomes
                status = final_record(patient.get("treatment", [])).get("status_at_last_follow_up")
                if status not in OUTCOME_LABELS:
                    continue
                i = TREATMENT_SCENARIOS.index(scenario)
                j = OUTCOME_LABELS.index(status)
                counts[i, j] += 1
        if counts.sum() == 0:
            return counts
        return counts / counts.sum()

    def compute_kl_pair(
        syn_subset: Sequence[Dict[str, Any]],
        ref_subset: Sequence[Dict[str, Any]],
    ) -> tuple[float, float]:
        syn_joint = joint_counts(list(syn_subset))
        ref_joint = joint_counts(list(ref_subset))
        joint_val = (
            kl_divergence(syn_joint.flatten(), ref_joint.flatten())
            if syn_joint.sum() and ref_joint.sum()
            else float("nan")
        )
        syn_marginal = syn_joint.sum(axis=0)
        ref_marginal = ref_joint.sum(axis=0)
        marginal_val = (
            kl_divergence(syn_marginal, ref_marginal)
            if syn_marginal.sum() and ref_marginal.sum()
            else float("nan")
        )
        return joint_val, marginal_val

    joint_kl, marginal_kl = compute_kl_pair(synthetic_patients, reference_patients)

    joint_ci: tuple[float, float] = (float("nan"), float("nan"))
    marginal_ci: tuple[float, float] = (float("nan"), float("nan"))

    if n_boot > 0 and synthetic_patients and reference_patients:
        syn_count = len(synthetic_patients)
        ref_count = len(reference_patients)
        rng = np.random.default_rng(seed)
        joint_samples = np.full(n_boot, float("nan"), dtype=float)
        marginal_samples = np.full(n_boot, float("nan"), dtype=float)

        for i in range(n_boot):
            syn_indices = rng.integers(0, syn_count, size=syn_count)
            ref_indices = rng.integers(0, ref_count, size=ref_count)
            syn_subset = [synthetic_patients[idx] for idx in syn_indices]
            ref_subset = [reference_patients[idx] for idx in ref_indices]
            joint_val, marginal_val = compute_kl_pair(syn_subset, ref_subset)
            joint_samples[i] = joint_val
            marginal_samples[i] = marginal_val

        joint_valid = joint_samples[np.isfinite(joint_samples)]
        marginal_valid = marginal_samples[np.isfinite(marginal_samples)]
        if joint_valid.size:
            joint_low, joint_high = np.percentile(joint_valid, [2.5, 97.5])
            joint_ci = (float(joint_low), float(joint_high))
        if marginal_valid.size:
            marginal_low, marginal_high = np.percentile(marginal_valid, [2.5, 97.5])
            marginal_ci = (float(marginal_low), float(marginal_high))

    return {
        "joint": joint_kl,
        "joint_ci": joint_ci,
        "marginal_outcome": marginal_kl,
        "marginal_outcome_ci": marginal_ci,
    }


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: Optional[int]) -> tuple[float, tuple[float, float], float, float]:
    '''
    Estimate mean, confidence interval, width, and SE via bootstrap.

    args:
    - values: sample array to resample [np.ndarray]
    - n_boot: bootstrap sample count [int]
    - seed: optional RNG seed [Optional[int]]

    return:
    - stats: mean, CI, CI width %, and standard error [tuple[float, tuple[float, float], float, float]]
    '''
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
    '''
    Summarise scenario-specific risk estimates with bootstrap intervals.

    args:
    - patients: patient records contributing to risk [List[Dict[str, Any]]]
    - feature: treatment outcome feature to analyse [str]
    - n_boot: bootstrap iterations [int]
    - seed: optional RNG seed [Optional[int]]

    return:
    - risks: per-scenario risk statistics [Dict[str, Dict[str, float]]]
    '''
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
    '''
    Build human-readable validation summary report.

    args:
    - summary: aggregated validation outputs [Dict[str, Any]]

    return:
    - text: multi-line report string [str]
    '''
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
        p_val = stats.get("ks_pvalue")
        p_str = f"{p_val:.3e}" if p_val is not None and np.isfinite(p_val) else "nan"
        lines.append(
            "  {feature}: W1={w1} CI={w1_ci}, KS={ks}, p={p}".format(
                feature=feature,
                w1=_format_float(stats.get("w1")),
                w1_ci=_format_ci(stats.get("w1_ci")),
                ks=_format_float(stats.get("ks_stat")),
                p=p_str,
            )
        )

    scenario_section = summary.get("scenario_metrics", {})
    status_labels = summary.get("status_labels", OUTCOME_LABELS)
    status_metrics = scenario_section.get("status", {})

    lines.append("")
    lines.append("Scenario status probability comparison (|Δmean|, KS):")
    for scenario in TREATMENT_SCENARIOS:
        lines.append(f"  {scenario}:")
        metrics = status_metrics.get(scenario, {})
        for label in status_labels:
            stats = metrics.get(label, {})
            p_val = stats.get("ks_pvalue")
            p_str = f"{p_val:.3e}" if p_val is not None and np.isfinite(p_val) else "nan"
            sizes = stats.get("sample_sizes", {})
            n_syn = sizes.get("synthetic")
            n_ref = sizes.get("reference")
            size_str = (
                f" N_syn={n_syn} N_ref={n_ref}"
                if n_syn is not None and n_ref is not None
                else ""
            )
            lines.append(
                "    {label}: |Δ|={delta} CI={delta_ci}, KS={ks}, p={p}{sizes}".format(
                    label=label,
                    delta=_format_float(stats.get("delta_mean")),
                    delta_ci=_format_ci(stats.get("delta_ci")),
                    ks=_format_float(stats.get("ks_stat")),
                    p=p_str,
                    sizes=size_str,
                )
            )

    endpoint_labels = summary.get("endpoint_labels", [])
    endpoint_metrics = scenario_section.get("endpoint", {}) if endpoint_labels else {}
    if endpoint_labels and endpoint_metrics:
        lines.append("")
        lines.append("Scenario treatment endpoint probability comparison (|Δmean|, KS):")
        for scenario in TREATMENT_SCENARIOS:
            lines.append(f"  {scenario}:")
            metrics = endpoint_metrics.get(scenario, {})
            for label in endpoint_labels:
                stats = metrics.get(label, {})
                p_val = stats.get("ks_pvalue")
                p_str = f"{p_val:.3e}" if p_val is not None and np.isfinite(p_val) else "nan"
                sizes = stats.get("sample_sizes", {})
                n_syn = sizes.get("synthetic")
                n_ref = sizes.get("reference")
                size_str = (
                    f" N_syn={n_syn} N_ref={n_ref}"
                    if n_syn is not None and n_ref is not None
                    else ""
                )
                lines.append(
                    "    {label}: |Δ|={delta} CI={delta_ci}, KS={ks}, p={p}{sizes}".format(
                        label=label,
                        delta=_format_float(stats.get("delta_mean")),
                        delta_ci=_format_ci(stats.get("delta_ci")),
                        ks=_format_float(stats.get("ks_stat")),
                        p=p_str,
                        sizes=size_str,
                    )
                )

    lines.append("")
    kl_stats = summary["kl_metrics"]
    lines.append(
        "KL divergence: joint={joint} CI={joint_ci}, marginal_outcome={marginal} CI={marg_ci}".format(
            joint=_format_float(kl_stats.get("joint")),
            joint_ci=_format_ci(kl_stats.get("joint_ci")),
            marginal=_format_float(kl_stats.get("marginal_outcome")),
            marg_ci=_format_ci(kl_stats.get("marginal_outcome_ci")),
        )
    )

    lines.append("")
    lines.append("Scenario risk summary (bootstrap means):")
    for scenario, stats in summary["scenario_risks"].items():
        lines.append(
            f"  {scenario}: risk={stats['risk']:.4f} CI=[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}] "
            f"width={stats['ci_width_percent']:.1f}% SE={stats['se']:.4f} N={stats['sample_size']}"
        )

    natural_metrics = summary.get("natural_experiments") or {}
    if natural_metrics:
        lines.append("")
        lines.append("Natural experiment outcome alignment (|Δmean|, KS p-value) for outcome labels:")
        for scenario, scenario_stats in natural_metrics.items():
            lines.append(f"  {scenario}:")
            for outcome_label in OUTCOME_LABELS:  # ["DOD", "NED", "AWD"]
                stats = scenario_stats.get(outcome_label, {})
                p_val = stats.get("ks_pvalue")
                p_str = f"{p_val:.3e}" if p_val is not None and np.isfinite(p_val) else "nan"
                sizes = stats.get("sample_sizes", {})
                n_syn = sizes.get("synthetic", 0)
                n_ref = sizes.get("reference", 0)
                lines.append(
                    "    {label}: |Δ|={delta} CI={ci}, KS={ks}, p={p} N_syn={n_syn} N_ref={n_ref}".format(
                        label=outcome_label,
                        delta=_format_float(stats.get("delta_mean")),
                        ci=_format_ci(stats.get("delta_ci")),
                        ks=_format_float(stats.get("ks_stat")),
                        p=p_str,
                        n_syn=n_syn,
                        n_ref=n_ref,
                    )
                )

    return "\n".join(lines)

def main(argv: Optional[List[str]] = None) -> None:
    '''
    Execute validation workflow from command line arguments.

    args:
    - argv: optional argument vector override [Optional[List[str]]]

    return:
    - None
    '''
    args = build_arg_parser().parse_args(argv)

    # retrieve model
    cfg, metadata, generator, validation_ids, device = load_checkpoint_bundle(args.checkpoint)
    
    # seed definition
    seed_value = args.seed if args.seed is not None else cfg.seed
    if seed_value is not None:
        set_seed(seed_value)

    dataset_status_labels: List[str] = []
    dataset_endpoint_labels: List[str] = []

    dataset_payload = load_dataset_payload(args.dataset)
    dataset = SyntheticSequenceDataset(
        dataset_payload,
        cfg.seq_len,
        spec_clinical=metadata.get("clinical_feature_spec"),
        spec_treatment=metadata.get("treatment_feature_spec"),
    )
    dataset_status_labels = list(dataset.treat_categories.get("status_at_last_follow_up", OUTCOME_LABELS))
    dataset_endpoint_labels = list(dataset.treat_categories.get("treatment_endpoint_category", []))
    
    cf_patients = build_counterfactual_patients(
        dataset,
        generator,
        device,
        validation_ids,
        status_labels=dataset_status_labels,
        endpoint_labels=dataset_endpoint_labels,
        samples_per_patient=max(1, args.samples),
    )
    ref_patients = filter_patients(dataset_payload["patients"], validation_ids)

    actual_scenarios: Dict[str, str] = {}
    aligned_ref_patients: List[Dict[str, Any]] = []
    for patient in ref_patients:
        scenario = scenario_label(final_record(patient.get("actual_treatment", [])))
        if scenario not in TREATMENT_SCENARIOS:
            continue
        patient_id = str(patient.get("patient_id"))
        actual_scenarios[patient_id] = scenario
        aligned_ref_patients.append(patient)
    ref_patients = aligned_ref_patients

    aligned_cf_patients: List[Dict[str, Any]] = []
    for patient in cf_patients:
        patient_id = str(patient.get("patient_id"))
        scenario = patient.get("scenario")
        if scenario not in TREATMENT_SCENARIOS:
            continue
        if actual_scenarios.get(patient_id) != scenario:
            continue
        aligned_cf_patients.append(patient)
    cf_patients = aligned_cf_patients

    status_labels = sorted(dataset_status_labels or OUTCOME_LABELS)
    endpoint_labels = sorted(dataset_endpoint_labels)

    ensure_probability_metadata(cf_patients, status_labels, endpoint_labels)
    ensure_probability_metadata(ref_patients, status_labels, endpoint_labels)

    natural_match_cols = args.natural_match_cols or []

    summary: Dict[str, Any] = {
        "counts": {
            "patients": len(cf_patients),
            "reference_patients": len(ref_patients),
        },
        "constraints": evaluate_constraints(cf_patients),
        "status_labels": status_labels,
        "endpoint_labels": endpoint_labels,
        "distribution_metrics": compute_distribution_metrics(
            cf_patients, ref_patients, args.bootstrap, seed_value
        ),
        "scenario_metrics": {
            "status": compute_scenario_metrics(
                cf_patients,
                ref_patients,
                status_labels,
                "status_probabilities",
                "status_at_last_follow_up",
                args.bootstrap,
                seed_value,
            ),
            "endpoint": compute_scenario_metrics(
                cf_patients,
                ref_patients,
                endpoint_labels,
                "endpoint_probabilities",
                "treatment_endpoint_category",
                args.bootstrap,
                seed_value,
            ) if endpoint_labels else {},
        },
        "kl_metrics": compute_kl_metrics(
            cf_patients,
            ref_patients,
            args.bootstrap,
            seed_value,
        ),
        "scenario_risks": compute_scenario_risks(
            cf_patients, args.risk_feature, args.bootstrap, seed_value
        ),
        "natural_experiments": compute_scenario_metrics(
            cf_patients,
            ref_patients,
            OUTCOME_LABELS, 
            "status_probabilities",
            "status_at_last_follow_up",
            args.bootstrap,
            seed_value,
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
