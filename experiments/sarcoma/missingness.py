#!/usr/bin/env python
"""Create protected, reproducible missingness datasets for sarcoma DTCyGAN."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml


PROTECTED_COLUMNS = {
    "any_whoops",
    "patient_id",
    "split",
    "treatment_arm",
    "eligible_preoperative_rt",
    "local_recurrence",
    "local_recurrence_date",
    "observed",
    "censored",
    "censoring_at_risk",
    "at_risk_local_recurrence",
    "loss_to_followup",
    "interval",
}


def _feature_columns(group: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for kind in ("numeric", "binary", "categorical"):
        columns.extend(str(value) for value in group.get(kind, []) or [])
    ordinal = group.get("ordinal", {}) or {}
    columns.extend(str(value) for value in ordinal)
    return list(dict.fromkeys(columns))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inject_missingness(
    frame: pd.DataFrame,
    baseline_columns: Sequence[str],
    state_columns: Sequence[str],
    rate: float,
    seed: int,
    patient_id_col: str = "patient_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0.0 < rate < 1.0:
        raise ValueError("Missingness rate must be strictly between zero and one")
    if patient_id_col not in frame:
        raise ValueError(f"Dataset is missing patient identifier {patient_id_col!r}")

    result = frame.copy()
    rng = np.random.default_rng(seed)
    details: dict[str, Any] = {}

    for column in baseline_columns:
        if column not in result or column in PROTECTED_COLUMNS:
            continue
        patient_values = (
            result[[patient_id_col, column]]
            .dropna(subset=[column])
            .drop_duplicates(subset=[patient_id_col])
        )
        patient_ids = patient_values[patient_id_col].astype(str).to_numpy()
        n_selected = min(len(patient_ids), int(round(len(patient_ids) * rate)))
        selected = set(rng.choice(patient_ids, size=n_selected, replace=False).tolist())
        before = int(result[column].isna().sum())
        mask = result[patient_id_col].astype(str).isin(selected) & result[column].notna()
        result.loc[mask, column] = np.nan
        details[column] = {
            "scope": "patient",
            "eligible_patients": int(len(patient_ids)),
            "selected_patients": int(n_selected),
            "new_missing_cells": int(result[column].isna().sum() - before),
        }

    for column in state_columns:
        if column not in result or column in PROTECTED_COLUMNS:
            continue
        candidates = result.index[result[column].notna()].to_numpy()
        n_selected = min(len(candidates), int(round(len(candidates) * rate)))
        selected = rng.choice(candidates, size=n_selected, replace=False)
        before = int(result[column].isna().sum())
        result.loc[selected, column] = np.nan
        details[column] = {
            "scope": "cell",
            "eligible_cells": int(len(candidates)),
            "selected_cells": int(n_selected),
            "new_missing_cells": int(result[column].isna().sum() - before),
        }

    for column in PROTECTED_COLUMNS & set(frame.columns):
        if not result[column].equals(frame[column]):
            raise AssertionError(f"Protected column changed: {column}")

    return result, details


def prepare(
    source: Path,
    config_path: Path,
    output_root: Path,
    rates: Sequence[float],
    seed: int,
) -> list[dict[str, Any]]:
    source = source.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    groups = config.get("feature_groups", {}) or {}
    baseline_columns = _feature_columns(groups.get("baseline", {}) or {})
    state_columns = _feature_columns(groups.get("state", {}) or {})
    overlap = set(baseline_columns) & set(state_columns)
    if overlap:
        raise ValueError(f"Features occur in baseline and state groups: {sorted(overlap)}")
    frame = pd.read_csv(source, dtype={"patient_id": "string"})
    output_root.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    manifest: list[dict[str, Any]] = []
    for rate in rates:
        label = f"missing_{int(round(rate * 100)):02d}pct"
        output_dir = output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)
        output_frame, details = inject_missingness(
            frame,
            baseline_columns=baseline_columns,
            state_columns=state_columns,
            rate=rate,
            seed=seed,
        )
        dataset_path = output_dir / "person_period.csv"
        output_frame.to_csv(dataset_path, index=False)
        provenance = {
            "label": label,
            "source": str(source),
            "source_sha256": source_hash,
            "config": str(config_path),
            "rate": rate,
            "seed": seed,
            "patient_id_column": "patient_id",
            "protected_columns": sorted(PROTECTED_COLUMNS),
            "protected_model_features": sorted(
                (set(baseline_columns) | set(state_columns)) & PROTECTED_COLUMNS
            ),
            "baseline_columns": baseline_columns,
            "state_columns": state_columns,
            "columns": details,
            "output": str(dataset_path.resolve()),
        }
        provenance_path = output_dir / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        manifest.append(provenance)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


