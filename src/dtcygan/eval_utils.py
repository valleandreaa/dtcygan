"""Shared evaluation utilities for counterfactual generation and statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import pandas as pd

from dtcygan.training import (
    Config,
    SyntheticSequenceDataset,
    LSTMGenerator,
    randomize_multiple_one_hot,
    resolve_device,
)

try:  # Optional SciPy acceleration for statistical tests
    from scipy.stats import ks_2samp as _scipy_ks_2samp
    from scipy.stats import wasserstein_distance as _scipy_wasserstein_distance
except Exception:  # pragma: no cover - SciPy might be unavailable
    _scipy_ks_2samp = None
    _scipy_wasserstein_distance = None


def load_dataset_payload(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or "patients" not in payload:
        raise ValueError(f"Dataset at {path} must contain a top-level 'patients' key.")
    return payload


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    payload = load_dataset_payload(path)
    patients = payload.get("patients", [])
    if not isinstance(patients, list):
        raise ValueError(f"Dataset at {path} does not contain a 'patients' list.")
    return patients


def filter_patients(patients: List[Dict[str, Any]], keep_ids: Optional[set[str]]) -> List[Dict[str, Any]]:
    if keep_ids is None:
        return patients
    keep_ids = {str(pid) for pid in keep_ids}
    return [patient for patient in patients if str(patient.get("patient_id")) in keep_ids]


def bootstrap_two_sample_ci(
    arr1: np.ndarray,
    arr2: np.ndarray,
    n_boot: int,
    seed: Optional[int],
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Tuple[float, float]:
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if n_boot <= 0 or arr1.size == 0 or arr2.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample1 = arr1[rng.integers(0, arr1.size, size=arr1.size)]
        sample2 = arr2[rng.integers(0, arr2.size, size=arr2.size)]
        stats[i] = stat_fn(sample1, sample2)
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def _wasserstein_distance_numpy(arr1: np.ndarray, arr2: np.ndarray, bins: int = 512) -> float:
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
    arr1 = np.sort(arr1)
    arr2 = np.sort(arr2)
    n1 = arr1.size
    n2 = arr2.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    data_all = np.concatenate([arr1, arr2])
    data_all.sort()
    cdf1 = np.searchsorted(arr1, data_all, side="right") / n1
    cdf2 = np.searchsorted(arr2, data_all, side="right") / n2
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    n_eff = math.sqrt(n1 * n2 / (n1 + n2))
    p_val = min(1.0, 2.0 * math.exp(-2.0 * (d_stat * n_eff) ** 2))
    return d_stat, p_val


def wasserstein_1d(arr1: np.ndarray, arr2: np.ndarray) -> float:
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if arr1.size == 0 or arr2.size == 0:
        return float("nan")
    if _scipy_wasserstein_distance is not None:
        return float(_scipy_wasserstein_distance(arr1, arr2))
    return _wasserstein_distance_numpy(arr1, arr2)


def ks_two_sample(arr1: np.ndarray, arr2: np.ndarray) -> tuple[float, float]:
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if arr1.size == 0 or arr2.size == 0:
        return float("nan"), float("nan")
    if _scipy_ks_2samp is not None:
        stat, p_val = _scipy_ks_2samp(arr1, arr2, alternative="two-sided", mode="auto")
        return float(stat), float(p_val)
    return _ks_2sample_numpy(arr1, arr2)


def collect_labels_from_patients(
    patients: List[Dict[str, Any]],
    field_name: str,
    prob_key: str,
) -> List[str]:
    labels: set[str] = set()
    for patient in patients:
        probs = patient.get(prob_key)
        if isinstance(probs, dict):
            labels.update(str(label) for label in probs.keys())
        treatment_records = patient.get("treatment", [])
        if treatment_records:
            label = final_record(treatment_records).get(field_name)
            if label is not None:
                labels.add(str(label))
    return sorted(labels)


def normalize_probability_dict(values: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0:
        return {k: 0.0 for k in values}
    return {k: float(v) / total for k, v in values.items()}


def ensure_probability_metadata(
    patients: List[Dict[str, Any]],
    status_labels: Sequence[str],
    endpoint_labels: Sequence[str],
) -> None:
    status_set = [str(label) for label in status_labels]
    endpoint_set = [str(label) for label in endpoint_labels]

    for patient in patients:
        treatment_records = patient.get("treatment", [])
        final_rec = final_record(treatment_records) if treatment_records else {}

        status_probs = patient.get("status_probabilities") or {}
        status_probs = {str(k): float(v) for k, v in status_probs.items() if v is not None}
        for label in status_set:
            status_probs.setdefault(label, 0.0)
        if status_set:
            label = str(final_rec.get("status_at_last_follow_up")) if final_rec else None
            if label in status_probs and all(v == 0.0 for v in status_probs.values()):
                status_probs[label] = 1.0
            patient["status_probabilities"] = normalize_probability_dict(status_probs)

        endpoint_probs = patient.get("endpoint_probabilities") or {}
        endpoint_probs = {str(k): float(v) for k, v in endpoint_probs.items() if v is not None}
        for label in endpoint_set:
            endpoint_probs.setdefault(label, 0.0)
        if endpoint_set:
            label = str(final_rec.get("treatment_endpoint_category")) if final_rec else None
            if label in endpoint_probs and all(v == 0.0 for v in endpoint_probs.values()):
                endpoint_probs[label] = 1.0
            patient["endpoint_probabilities"] = normalize_probability_dict(endpoint_probs)


def final_record(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    return records[-1] if records else {}


def decode_treatment_value(
    feature: str,
    raw: float,
    categorical_features: set[str],
    categorical_map: Dict[str, List[str]],
    ordinal_lookup: Dict[str, List[Tuple[float, str]]],
) -> Any:
    if feature in categorical_features:
        categories = categorical_map.get(feature, [])
        if categories:
            code = int(round(raw))
            code = max(0, min(code, len(categories) - 1))
            return categories[code]
        return int(round(raw))
    if feature in ordinal_lookup:
        candidates = ordinal_lookup[feature]
        if candidates:
            closest_value, closest_label = min(candidates, key=lambda item: abs(item[0] - raw))
            return closest_label
    return float(raw)


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    desired_device: Optional[str] = None,
) -> tuple[Config, Dict[str, Any], LSTMGenerator, Optional[set[str]], torch.device]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    metadata: Dict[str, Any] = ckpt.get("metadata") or {}
    validation_ids = metadata.get("validation_patient_ids")
    if validation_ids is not None:
        validation_ids = {str(pid) for pid in validation_ids}

    clin_dim = metadata.get("clin_dim")
    treat_dim = metadata.get("treat_dim")
    cond_dim = metadata.get("cond_dim")
    if None in {clin_dim, treat_dim, cond_dim}:
        raise ValueError("Checkpoint metadata must include clin_dim, treat_dim, and cond_dim.")

    device = resolve_device(desired_device or cfg.device)
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
    results: List[Dict[str, Any]] = []
    generator.eval()

    categorical_features = dataset.treat_categorical
    categorical_map = dataset.treat_categories
    ordinal_lookup = {
        feature: [(float(value), label) for label, value in mapping.items()]
        for feature, mapping in dataset.treat_ord_maps.items()
    }

    status_labels = (
        status_labels
        or list(dataset.treat_categories.get("status_at_last_follow_up", []))
        or ["AWD", "DOD", "NED"]
    )
    endpoint_labels = (
        endpoint_labels
        or list(dataset.treat_categories.get("treatment_endpoint_category", []))
    )

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
        tr_actual = tr_actual * mask_actual
        base_actual = tr_actual[:, :1, :]
        base_mask = mask_actual[:, :1, :]

        representative_records: Optional[List[Dict[str, Any]]] = None
        conditioning_records: List[Dict[str, Any]] = []

        status_counts: Dict[str, float] = {label: 0.0 for label in status_labels}
        endpoint_counts: Dict[str, float] = {label: 0.0 for label in endpoint_labels}

        samples = max(1, samples_per_patient)

        for sample_idx in range(samples):
            with torch.no_grad():
                cond_random = randomize_multiple_one_hot(base_actual, base_mask, extra_ones=extra_ones)

            seq_len = x_clin.size(1)
            cond_random = cond_random.repeat(1, seq_len, 1)
            cond_mask = base_mask.repeat(1, seq_len, 1)
            cond_random = cond_random * step_mask
            cond_mask = cond_mask * step_mask

            with torch.no_grad():
                fake_treat, _ = generator(x_clin, mask_clin, cond_random, cond_mask, lengths)
                fake_treat = fake_treat * mask_treat

            fake_np = fake_treat.squeeze(0).cpu().numpy()
            mask_np = mask_treat.squeeze(0).cpu().numpy()
            cond_np = cond_random.squeeze(0).cpu().numpy()
            cond_mask_np = cond_mask.squeeze(0).cpu().numpy()
            valid_steps = mask_np.sum(axis=1) > 0
            step_mask_np = step_mask.squeeze(0).squeeze(-1).cpu().numpy().astype(bool)

            treatment_records_sample: List[Dict[str, Any]] = []
            for step_idx, valid in enumerate(valid_steps):
                if not bool(valid):
                    continue
                step_record: Dict[str, Any] = {
                    feature: decode_treatment_value(
                        feature,
                        fake_np[step_idx, feat_idx],
                        categorical_features,
                        categorical_map,
                        ordinal_lookup,
                    )
                    for feat_idx, feature in enumerate(dataset.treat_columns)
                    if mask_np[step_idx, feat_idx] > 0
                }
                step_record["_id"] = patient_id
                step_record["timestep"] = step_idx
                treatment_records_sample.append(step_record)

            if representative_records is None:
                representative_records = [dict(rec) for rec in treatment_records_sample]

            if treatment_records_sample:
                final_record_sample = treatment_records_sample[-1]
                status_label = final_record_sample.get("status_at_last_follow_up")
                if status_label in status_counts:
                    status_counts[status_label] += 1.0
                endpoint_label = final_record_sample.get("treatment_endpoint_category")
                if endpoint_label in endpoint_counts:
                    endpoint_counts[endpoint_label] += 1.0

            if sample_idx == 0:
                conditioning_records.clear()
                for step_idx, valid in enumerate(step_mask_np):
                    if not bool(valid):
                        continue
                    cond_record: Dict[str, Any] = {
                        feature: int(round(cond_np[step_idx, feat_idx]))
                        for feat_idx, feature in enumerate(dataset.cond_columns)
                        if cond_mask_np[step_idx, feat_idx] > 0
                    }
                    cond_record["_id"] = patient_id
                    cond_record["timestep"] = step_idx
                    conditioning_records.append(cond_record)

        total_samples = float(samples)
        status_probs = {
            label: (status_counts.get(label, 0.0) / total_samples)
            for label in status_labels
        }
        endpoint_probs = {
            label: (endpoint_counts.get(label, 0.0) / total_samples)
            for label in endpoint_labels
        }

        if representative_records is None:
            representative_records = []
        if not representative_records:
            representative_records = patient.get("treatment", [])

        results.append(
            {
                "patient_id": patient_id,
                "clinical": patient.get("clinical", []),
                "treatment": representative_records,
                "actual_treatment": conditioning_records or patient.get("actual_treatment", []),
                "status_probabilities": normalize_probability_dict(status_probs) if status_probs else {},
                "endpoint_probabilities": normalize_probability_dict(endpoint_probs) if endpoint_probs else {},
                "samples_per_patient": samples,
            }
        )

    return results


def patients_to_dataframe(
    patients: List[Dict[str, Any]],
    *,
    status_labels: Sequence[str],
    endpoint_labels: Sequence[str],
    timestep_key: str = "timestep",
    include_timesteps: bool = True,
    include_probabilities: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for patient in patients:
        pid = patient.get("patient_id")
        treatment_records = patient.get("treatment", []) or []
        conditioning_records = patient.get("actual_treatment", []) or []

        # Derive flags from first conditioning record if available
        surgery_flag = chemo_flag = radio_flag = None
        if conditioning_records:
            first_cond = conditioning_records[0]
            surgery_flag = int(first_cond.get("episodes.surgery", 0))
            chemo_flag = int(first_cond.get("episodes.chemotherapy", 0))
            radio_flag = int(first_cond.get("episodes.radiotherapy", 0))

        status_probs = patient.get("status_probabilities", {})
        endpoint_probs = patient.get("endpoint_probabilities", {})

        # Clinical attributes from first clinical record (if any)
        clinical_records = patient.get("clinical", []) or []
        clinical_base = clinical_records[0] if clinical_records else {}

        probability_cols = (
            _probability_columns(status_labels, endpoint_labels, status_probs, endpoint_probs)
            if include_probabilities
            else {}
        )

        if not treatment_records or not include_timesteps:
            row = {
                "patient_id": pid,
                timestep_key: treatment_records[-1].get(timestep_key, 0) if treatment_records else 0,
                "has_clinical_data": True,
            }
            if surgery_flag is not None:
                row.update(
                    {
                        "surgery": surgery_flag,
                        "radiotherapy": radio_flag,
                        "chemotherapy": chemo_flag,
                    }
                )
            row.update({k: clinical_base.get(k) for k in clinical_base})
            if treatment_records:
                row.update(treatment_records[-1])
            row.update(probability_cols)
            rows.append(row)
            continue

        for record in treatment_records:
            timestep = record.get(timestep_key, 0)
            row = {
                "patient_id": pid,
                timestep_key: timestep,
                "has_clinical_data": True,
            }
            row.update(
                {
                    "surgery": surgery_flag,
                    "radiotherapy": radio_flag,
                    "chemotherapy": chemo_flag,
                }
            )
            row.update({k: clinical_base.get(k) for k in clinical_base})
            row.update(record)
            row.update(probability_cols)
            rows.append(row)

    return pd.DataFrame(rows)


def _probability_columns(
    status_labels: Sequence[str],
    endpoint_labels: Sequence[str],
    status_probs: Dict[str, float],
    endpoint_probs: Dict[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label in status_labels:
        prob = float(status_probs.get(label, 0.0))
        key = f"fake_episodes.diagnosis.fields.status_{label}_prob"
        out[key] = prob
        out[f"status_{label}_prob"] = prob
    endpoint_map = {
        "LR": ("local_recurrence_prob", "fake_episodes.treatments.fields.endpoint_LR_prob"),
        "MET": ("metastasis_prob", "fake_episodes.treatments.fields.endpoint_MET_prob"),
        "DOD": ("death_of_disease_prob", "fake_episodes.treatments.fields.endpoint_DOD_prob"),
    }
    for idx, label in enumerate(endpoint_labels):
        prob = float(endpoint_probs.get(label, 0.0))
        cols = endpoint_map.get(label, (f"endpoint_{label}_prob", f"fake_episodes.treatments.fields.endpoint_{label}_prob"))
        out[cols[0]] = prob
        out[cols[1]] = prob
        numeric_key = f"fake_episodes.treatments.fields.endpoint_{idx + 1}.0_prob"
        out[numeric_key] = prob
    return out
