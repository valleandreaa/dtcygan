#!/usr/bin/env python
"""Withhold MIMIC follow-up covariates for a T4-aligned forecast contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


STATIC_FEATURES = frozenset({"agegroup", "heightgroup", "weightgroup", "gender"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_dataset(
    payload: dict[str, Any],
    baseline_end: int = 8,
    evaluation_start: int = 9,
    evaluation_end: int = 15,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Null follow-up temporal covariates without changing targets or splits."""
    metadata = payload.get("metadata")
    patients = payload.get("patients")
    if not isinstance(metadata, dict) or not isinstance(patients, list) or not patients:
        raise ValueError("dataset must contain metadata and a non-empty patient list")
    sequence_length = int(metadata.get("sequence_length", 0))
    if not (0 <= baseline_end < evaluation_start <= evaluation_end < sequence_length):
        raise ValueError("invalid baseline/follow-up timestamp contract")
    clinical_features = metadata.get("clinical_features")
    if not isinstance(clinical_features, list) or not clinical_features:
        raise ValueError("metadata.clinical_features must be a non-empty list")
    temporal_features = [
        str(feature) for feature in clinical_features if str(feature) not in STATIC_FEATURES
    ]
    if not temporal_features:
        raise ValueError("no temporal clinical features are available to align")

    withheld = 0
    for patient in patients:
        records = patient.get("clinical")
        if not isinstance(records, list) or len(records) != sequence_length:
            raise ValueError("every patient must have a complete clinical sequence")
        for record_index, record in enumerate(records):
            timestamp = int(record.get("timestep", record_index))
            if evaluation_start <= timestamp <= evaluation_end:
                for feature in temporal_features:
                    if feature not in record:
                        raise ValueError(f"clinical record is missing feature {feature!r}")
                    if record[feature] is not None:
                        withheld += 1
                    record[feature] = None
        if len(patient.get("treatment", [])) != sequence_length:
            raise ValueError("alignment must not change the treatment/outcome sequence")
        if len(patient.get("actual_treatment", [])) != sequence_length:
            raise ValueError("alignment must not change the treatment condition sequence")

    contract = {
        "contract": "t4_baseline_forecast",
        "baseline_timestamps": list(range(0, baseline_end + 1)),
        "evaluation_timestamps": list(range(evaluation_start, evaluation_end + 1)),
        "followup_temporal_covariates_available": False,
        "preserved_static_features": sorted(STATIC_FEATURES),
        "withheld_temporal_features": temporal_features,
        "withheld_values": int(withheld),
    }
    if source_sha256 is not None:
        contract["source_sha256"] = source_sha256
    metadata["forecast_alignment"] = contract
    return contract


def write_aligned_dataset(
    source: Path,
    output: Path,
    baseline_end: int = 8,
    evaluation_start: int = 9,
    evaluation_end: int = 15,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    source_sha256 = _sha256(source)
    if output.exists() and not overwrite:
        with output.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        contract = existing.get("metadata", {}).get("forecast_alignment")
        expected_baseline = list(range(0, baseline_end + 1))
        expected_evaluation = list(range(evaluation_start, evaluation_end + 1))
        if (
            not isinstance(contract, dict)
            or contract.get("source_sha256") != source_sha256
            or contract.get("baseline_timestamps") != expected_baseline
            or contract.get("evaluation_timestamps") != expected_evaluation
            or contract.get("followup_temporal_covariates_available") is not False
        ):
            raise ValueError(f"existing aligned dataset is incompatible: {output}")
        return contract

    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    contract = align_dataset(
        payload,
        baseline_end=baseline_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        source_sha256=source_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return contract


