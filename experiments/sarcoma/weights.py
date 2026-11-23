#!/usr/bin/env python
"""Build the fixed 3-month sarcoma cohort and its training weights."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent

from src.surgery_rt_cohort import (  # noqa: E402
    build_surgery_rt_cohort,
    configure_preoperative_rt,
    load_causal_config,
    load_patient_records,
    save_surgery_rt_cohort,
)
from weighting import prepare_training_weights  # noqa: E402


def _get_path(record: Mapping[str, Any], path: str) -> Any:
    if path in record:
        return record[path]
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _date(value: Any) -> pd.Timestamp:
    if isinstance(value, Mapping) and "$date" in value:
        value = value["$date"]
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None)


def _event_date(event: Mapping[str, Any]) -> pd.Timestamp:
    for name in (
        "date_field",
        "date_pathology_report",
        "radiotherapy_start_date",
        "date_index_surgery",
        "cycle_start_date",
        "date_local_recurrence",
        "date_death",
        "interventional_radiology_date",
    ):
        parsed = _date(event.get(name))
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def _time_zero(patient: Mapping[str, Any], config: Mapping[str, Any]) -> pd.Timestamp:
    candidates = config.get("time", {}).get("time_zero_candidates", [])
    for candidate in candidates:
        parsed = _date(_get_path(patient, str(candidate)))
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def _source_observation_ranges(patient: Mapping[str, Any]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    episodes = patient.get("episodes", [])
    if not isinstance(episodes, list):
        return ranges
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        start = _date(episode.get("start_date"))
        end = _date(episode.get("end_date"))
        if pd.notna(start) and pd.notna(end):
            ranges.append((start, end))
    return ranges


def _is_observed(
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    source_ranges: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    return any(source_start < end_exclusive and source_end >= start for source_start, source_end in source_ranges)


def rebin_patient_to_three_months(
    patient: Mapping[str, Any],
    config: Mapping[str, Any],
    interval_months: int = 3,
    n_intervals: int = 8,
) -> dict[str, Any]:
    """Create observed three-month episodes from exact-dated timeline events."""

    result = deepcopy(dict(patient))
    time_zero = _time_zero(patient, config)
    if pd.isna(time_zero):
        # Preserve the original record so the cohort builder applies its normal
        # missing-time-zero exclusion rather than inventing a date.
        return result
    timeline = patient.get("diagnosis_treatment_sequence", [])
    events = [dict(item) for item in timeline if isinstance(item, Mapping)] if isinstance(timeline, list) else []
    source_ranges = _source_observation_ranges(patient)
    episodes: list[dict[str, Any]] = []
    for interval in range(n_intervals):
        start = time_zero + pd.DateOffset(months=interval * interval_months)
        end_exclusive = time_zero + pd.DateOffset(months=(interval + 1) * interval_months)
        if not _is_observed(start, end_exclusive, source_ranges):
            continue
        in_interval = [
            event
            for event in events
            if pd.notna(_event_date(event)) and start <= _event_date(event) < end_exclusive
        ]
        # The source timeline uses sections such as ``follow_up`` for disease
        # status records. The downstream state extractor intentionally reads
        # ``last_status`` from the diagnosis/status container, so classification
        # must follow content as well as the literal section label.
        diagnosis = [
            event
            for event in in_interval
            if str(event.get("section", "")).lower() == "diagnosis"
            or str(event.get("last_status", "")).strip()
        ]
        diagnosis_ids = {id(event) for event in diagnosis}
        treatments = [event for event in in_interval if id(event) not in diagnosis_ids]
        sections = {str(event.get("section", "")).strip().lower().replace("_", "-") for event in treatments}
        episodes.append(
            {
                "episode_id": interval + 1,
                "start_date": start.isoformat(),
                "end_date": (end_exclusive - pd.Timedelta(days=1)).isoformat(),
                "duration_days": int((end_exclusive - start).days),
                "n_interventions": len(in_interval),
                "treatments": treatments,
                "diagnosis": diagnosis,
                "surgery": int("surgery" in sections),
                "chemotherapy": int("systemic-therapy" in sections or "chemotherapy" in sections),
                "radiotherapy": int("radiotherapy" in sections),
                "interventional_radiology": int("interventional-radiology" in sections),
                "metastasis": int("metastasis" in sections),
                "local_recurrence": int("local-recurrence" in sections),
                "final_status": int(any(str(event.get("last_status", "")).strip() for event in diagnosis)),
            }
        )
    result["episodes"] = episodes
    result["analysis_episode_contract"] = {
        "interval_months": interval_months,
        "n_intervals": n_intervals,
        "horizon_months": interval_months * n_intervals,
        "source": "exact_dated_diagnosis_treatment_sequence",
    }
    return result


def make_3m24_config(base: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(base))
    config.setdefault("time", {}).update(
        {"interval_months": 3, "n_intervals": 8, "horizon_months": 24}
    )
    config.setdefault("analysis", {})["horizons_months"] = [12, 24]
    config.setdefault("target_trial", {})["follow_up_months"] = 24
    return config


def _reset_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_experiment(
    source_path: Path,
    causal_config_path: Path,
    training_config_path: Path,
    cohort_output: Path,
    weight_output: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    _reset_output(cohort_output, overwrite=overwrite)
    _reset_output(weight_output, overwrite=overwrite)
    base_causal = load_causal_config(causal_config_path)
    causal_3m24 = make_3m24_config(base_causal)
    configured = configure_preoperative_rt(causal_3m24)
    records = load_patient_records(source_path)
    rebinned = [rebin_patient_to_three_months(record, configured) for record in records]
    cohort = build_surgery_rt_cohort(rebinned, configured)
    save_surgery_rt_cohort(cohort, cohort_output, configured)
    return prepare_training_weights(
        cohort_output / "person_period.csv",
        training_config_path,
        causal_config_path,
        weight_output,
    )


