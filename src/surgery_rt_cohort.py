"""Build an auditable surgery-versus-radiotherapy longitudinal cohort.

The builder deliberately separates eligibility, treatment assignment, and
post-treatment variables.  It consumes either the nested patient records
produced by the Causal-Sarcoma-DT ETL or a legacy flat table.  Flat records are
accepted conservatively: an ``ever radiotherapy`` flag without timing or an
eligible indication is never interpreted as perioperative radiotherapy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math

import numpy as np
import pandas as pd
import yaml


_MISSING_STRINGS = {"", "-", "--", "na", "n/a", "none", "null", "nan", "unknown"}
_DEATH_STATES = {
    "dead",
    "deceased",
    "dod",
    "dead of disease",
    "death of disease",
    "dead of other cause",
    "death of other cause",
}


@dataclass
class SurgeryRTCohort:
    """Outputs of :func:`build_surgery_rt_cohort`."""

    person_period: pd.DataFrame
    patient_summary: pd.DataFrame
    exclusions: pd.DataFrame
    cohort_flow: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TemporalValidity:
    """Binary temporal validity plus an explicit ordering-resolution flag."""

    valid: float
    resolved: float


def _administration_before_recurrence(
    administration_date: pd.Timestamp,
    recurrence_date: pd.Timestamp,
    date_resolved: bool = True,
) -> TemporalValidity:
    """Return whether a dated administration precedes first local recurrence.

    An unavailable or proxy administration date is unresolved and therefore
    fails closed. Equality is invalid because treatment must strictly precede
    recurrence to be a pre-recurrence exposure.
    """

    if not date_resolved or pd.isna(administration_date):
        return TemporalValidity(valid=0.0, resolved=0.0)
    if pd.isna(recurrence_date):
        return TemporalValidity(valid=1.0, resolved=1.0)
    return TemporalValidity(
        valid=float(administration_date < recurrence_date),
        resolved=1.0,
    )


def _available_at_prediction_time(
    availability_date: pd.Timestamp,
    prediction_time: pd.Timestamp,
    date_resolved: bool = True,
) -> TemporalValidity:
    """Return availability at a fixed landmark without consulting outcome time."""

    if (
        not date_resolved
        or pd.isna(availability_date)
        or pd.isna(prediction_time)
    ):
        return TemporalValidity(valid=0.0, resolved=0.0)
    return TemporalValidity(
        valid=float(availability_date <= prediction_time),
        resolved=1.0,
    )


def load_causal_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def configure_preoperative_rt(config: Mapping[str, Any]) -> dict[str, Any]:
    """Configure preoperative RT plus surgery versus surgery alone."""

    import copy

    configured = copy.deepcopy(dict(config))
    rt = configured.setdefault("treatment", {}).setdefault("radiotherapy", {})
    rt["eligible_indications"] = ["preoperative"]
    rt["allowed_nonexposure_indications"] = ["postoperative"]
    rt["assignment_rule"] = "window_before_outcome"
    configured.setdefault("analysis", {})["name"] = "preoperative_rt_vs_surgery"
    return configured


def load_patient_records(path: str | Path) -> list[dict[str, Any]]:
    """Load nested JSON/JSONL or flat CSV records."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source).replace({np.nan: None}).to_dict(orient="records")
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("patients", "records", "data"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError("JSON input must be a patient list or contain patients/records/data.")
    raise ValueError(f"Unsupported cohort input format: {source.suffix}")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _MISSING_STRINGS
    if isinstance(value, (list, tuple, set)):
        return any(_present(item) for item in value)
    return True


def _get_path(record: Mapping[str, Any], path: str, default: Any = None) -> Any:
    # CSV and json_normalize outputs can contain literal dotted column names.
    if path in record:
        return record[path]
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _first(record: Mapping[str, Any], candidates: Sequence[str], default: Any = None) -> Any:
    for path in candidates:
        value = _get_path(record, str(path), None)
        if _present(value):
            return value
    return default


def _as_number(value: Any, default: float | None = None) -> float | None:
    if not _present(value):
        return default
    if isinstance(value, (list, tuple)):
        numbers = [_as_number(item) for item in value]
        numbers = [item for item in numbers if item is not None]
        return max(numbers) if numbers else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_flag(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return int(any(_as_flag(item) for item in value))
    if isinstance(value, bool):
        return int(value)
    number = _as_number(value)
    if number is not None:
        return int(number > 0)
    return int(str(value).strip().lower() in {"yes", "y", "true", "present", "positive", "dod"})


def _date(value: Any) -> pd.Timestamp:
    if isinstance(value, Mapping) and "$date" in value:
        value = value["$date"]
    if isinstance(value, (list, tuple)):
        parsed = [_date(item) for item in value]
        parsed = [item for item in parsed if pd.notna(item)]
        return min(parsed) if parsed else pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None)


def _values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        result: list[Any] = []
        for item in value:
            result.extend(_values(item))
        return result
    return [] if not _present(value) else [value]


def _normalise_label(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", "-").split())


def _event_date(event: Mapping[str, Any]) -> pd.Timestamp:
    candidates = (
        "date_field",
        "radiotherapy_start_date",
        "date_index_surgery",
        "cycle_start_date",
        "date_local_recurrence",
        "date_death",
        "interventional_radiology_date",
    )
    return _date(_first(event, candidates))


def _episodes(patient: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = patient.get("episodes", [])
    episodes = [dict(item) for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []
    return sorted(
        episodes,
        key=lambda item: (
            _as_number(item.get("episode_id"), float("inf")),
            _date(item.get("start_date")),
        ),
    )


def _events(patient: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the full dated timeline, falling back to episode-contained events.

    Prediagnosis WHOOPS surgery and other baseline events can be intentionally
    absent from follow-up episodes. The full sequence is therefore authoritative
    for eligibility and treatment classification, while episodes remain the
    source of interval-specific covariates.
    """

    events: list[dict[str, Any]] = []
    timeline = patient.get("diagnosis_treatment_sequence", [])
    if isinstance(timeline, list) and any(isinstance(event, Mapping) for event in timeline):
        return [dict(event) for event in timeline if isinstance(event, Mapping)]

    episodes = _episodes(patient)
    if episodes:
        for episode in episodes:
            episode_id = int(_as_number(episode.get("episode_id"), 0) or 0)
            for container in ("treatments", "diagnosis"):
                values = episode.get(container, [])
                if not isinstance(values, list):
                    continue
                for value in values:
                    if isinstance(value, Mapping):
                        event = dict(value)
                        event["_episode_id"] = episode_id
                        events.append(event)
        return events

    return []


def _histology_in_scope(value: Any, population: Mapping[str, Any]) -> tuple[bool, str]:
    if not _present(value):
        return False, "missing_histology"
    normal = _normalise_label(value)
    groups = {_normalise_label(item) for item in population.get("manuscript_histology_groups", [])}
    labels = {_normalise_label(item) for item in population.get("manuscript_histology_labels", [])}
    if normal in groups:
        return True, "grouped_code"
    if normal in labels:
        return True, "original_label"
    return False, "outside_manuscript_histology_scope"


def _definitive_surgery(
    events: Sequence[Mapping[str, Any]], population: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    eligible_types = {
        _normalise_label(item) for item in population.get("primary_surgery_types", ["index", "whoops"])
    }
    excluded_indications = {
        _normalise_label(item) for item in population.get("excluded_surgery_indications", [])
    }
    candidates = []

    def canonical_type(value: Any) -> str:
        label = _normalise_label(value)
        if "whoops" in label:
            return "whoops"
        if "index" in label:
            return "index"
        return label

    for event in events:
        if _normalise_label(event.get("section")) != "surgery":
            continue
        types = [canonical_type(item) for item in _values(event.get("surgery_types"))]
        if _present(event.get("surgery_type")):
            types.append(canonical_type(event.get("surgery_type")))
        indications = {
            _normalise_label(item)
            for item in (
                _values(event.get("surgery_indication"))
                + _values(event.get("surgery_indications"))
            )
        }
        if eligible_types and not any(item in eligible_types for item in types):
            continue
        if indications & excluded_indications:
            continue
        event_date = _event_date(event)
        if pd.notna(event_date):
            candidates.append((event_date, event))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _rt_indications(event: Mapping[str, Any]) -> set[str]:
    raw = _values(event.get("radiotherapy_indication")) + _values(event.get("radiotherapy_indications"))
    labels = set()
    for value in raw:
        normal = _normalise_label(value)
        if "preop" in normal or "neoadjuvant" in normal:
            labels.add("preoperative")
        elif "postop" in normal or "adjuvant" in normal:
            labels.add("postoperative")
        elif "palliat" in normal:
            labels.add("palliative")
        elif "definit" in normal:
            labels.add("definitive")
        elif "stereot" in normal:
            labels.add("stereotactic")
        elif "recurr" in normal:
            labels.add("recurrence")
        elif "metasta" in normal:
            labels.add("metastasis")
        elif normal:
            labels.add(normal)
    return labels


def _classify_radiotherapy(
    patient: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    surgery_date: pd.Timestamp,
    config: Mapping[str, Any],
    first_outcome_date: pd.Timestamp,
) -> dict[str, Any]:
    rt_config = config.get("treatment", {}).get("radiotherapy", {})
    eligible = {_normalise_label(item) for item in rt_config.get("eligible_indications", [])}
    excluded = {_normalise_label(item) for item in rt_config.get("excluded_indications", [])}
    allowed_nonexposure = {
        _normalise_label(item) for item in rt_config.get("allowed_nonexposure_indications", [])
    }
    assignment_rule = str(rt_config.get("assignment_rule", "window_before_outcome"))
    explicit_any_time = assignment_rule == "explicit_indication_any_time"
    before = int(rt_config.get("days_before_surgery", 365))
    after = int(rt_config.get("days_after_surgery", 180))
    lower = surgery_date - pd.Timedelta(days=before)
    upper = surgery_date + pd.Timedelta(days=after)
    top_flag = _as_flag(_first(patient, rt_config.get("top_level_preoperative_flag_candidates", []), 0))

    qualifying: list[tuple[pd.Timestamp, Mapping[str, Any], str]] = []
    ambiguous_perioperative = False
    has_nonqualifying = False
    nonqualifying: list[tuple[pd.Timestamp, Mapping[str, Any], str]] = []
    observed_nonexposure_indications: set[str] = set()
    all_rt = []
    for event in events:
        if _normalise_label(event.get("section")) != "radiotherapy":
            continue
        when = _event_date(event)
        indications = _rt_indications(event)
        all_rt.append((when, event, indications))
        in_window = explicit_any_time or (pd.notna(when) and lower <= when <= upper)
        before_outcome = explicit_any_time or pd.isna(first_outcome_date) or (
            pd.notna(when) and when < first_outcome_date
        )
        # Treatment observed after the outcome is downstream of the endpoint.
        # It must neither define baseline exposure nor exclude an otherwise
        # eligible patient from the target trial.
        if not before_outcome:
            continue
        if indications & excluded:
            has_nonqualifying = True
            nonqualifying.extend(
                (when, event, indication) for indication in sorted(indications & excluded)
            )
            continue
        matched = sorted(indications & eligible)
        if matched and in_window and before_outcome:
            qualifying.extend((when, event, indication) for indication in matched)
        elif indications & allowed_nonexposure and in_window and before_outcome:
            observed_nonexposure_indications.update(indications & allowed_nonexposure)
        elif in_window and before_outcome and not indications:
            if explicit_any_time:
                has_nonqualifying = True
                nonqualifying.append((when, event, "missing_indication"))
            else:
                ambiguous_perioperative = True
        elif indications or pd.notna(when):
            has_nonqualifying = True
            label = sorted(indications)[0] if indications else "missing_indication"
            nonqualifying.append((when, event, label))

    # The ETL's explicit preoperative flag is permitted to resolve an otherwise
    # indication-missing preoperative course, but it does not convert palliative
    # or definitive courses into eligible exposure.
    if not qualifying and top_flag:
        plausible = [item for item in all_rt if pd.notna(item[0]) and lower <= item[0] <= surgery_date and not (item[2] & excluded)]
        if plausible:
            when, event, _ = min(plausible, key=lambda item: abs((surgery_date - item[0]).days))
            qualifying.append((when, event, "preoperative_top_level_flag"))
            ambiguous_perioperative = False
        elif not all_rt:
            qualifying.append((surgery_date, {}, "preoperative_top_level_flag_date_unavailable"))

    if qualifying:
        qualifying.sort(key=lambda item: (pd.isna(item[0]), item[0] if pd.notna(item[0]) else pd.Timestamp.max))
        when, event, indication = qualifying[0]
        return {
            "treated": 1,
            "date": when,
            "indication": indication,
            "event": event,
            "ambiguous": False,
            "any_rt": bool(all_rt or top_flag),
            "has_nonqualifying": has_nonqualifying,
            "qualifying_courses": qualifying,
            "nonqualifying_courses": nonqualifying,
            "allowed_nonexposure_indications": sorted(observed_nonexposure_indications),
        }
    return {
        "treated": 0,
        "date": pd.NaT,
        "indication": None,
        "event": {},
        "ambiguous": bool(ambiguous_perioperative),
        "any_rt": bool(all_rt or top_flag),
        "has_nonqualifying": has_nonqualifying,
        "qualifying_courses": [],
        "nonqualifying_courses": nonqualifying,
        "allowed_nonexposure_indications": sorted(observed_nonexposure_indications),
    }


def _first_event_date(events: Sequence[Mapping[str, Any]], sections: set[str], *, dod: bool = False) -> pd.Timestamp:
    dates = []
    for event in events:
        explicit_death_date = _date(event.get("date_death")) if dod else pd.NaT
        if dod and pd.notna(explicit_death_date):
            dates.append(explicit_death_date)
            continue
        section = _normalise_label(event.get("section"))
        if section not in sections:
            continue
        # A follow-up record can carry both a visit date and the actual death
        # date. Any explicit death date is an all-cause competing event. A
        # status-only record is accepted only when it has a known death label.
        when = _event_date(event)
        if dod:
            if _normalise_label(event.get("last_status")) not in _DEATH_STATES:
                continue
        if pd.notna(when):
            dates.append(when)
    return min(dates) if dates else pd.NaT


def _interval_index(when: pd.Timestamp, episodes: Sequence[Mapping[str, Any]], time_zero: pd.Timestamp, n: int) -> int | None:
    if pd.isna(when) or when < time_zero:
        return None
    for fallback, episode in enumerate(episodes[:n]):
        start = _date(episode.get("start_date"))
        end = _date(episode.get("end_date"))
        idx = int(_as_number(episode.get("episode_id"), fallback + 1) or fallback + 1) - 1
        if pd.notna(start) and pd.notna(end) and start <= when <= end and 0 <= idx < n:
            return idx
    # Calendar approximation is only a fallback when explicit episode bounds
    # are unavailable.
    idx = int((when - time_zero).days // 183)
    return idx if 0 <= idx < n else None


def _episode_features(episode: Mapping[str, Any]) -> dict[str, Any]:
    treatments = episode.get("treatments", [])
    treatments = treatments if isinstance(treatments, list) else []
    diagnosis = episode.get("diagnosis", [])
    diagnosis = diagnosis if isinstance(diagnosis, list) else []

    margin = None
    reoperations = 0.0
    hyperthermia = 0
    rt_dose = None
    rt_dose_raw = None
    rt_fractions = None
    rt_num_fractions = None
    rt_dose_per_fraction = None
    rt_indication = None
    rt_type = None
    rt_interrupted = 0
    surgery_type = None
    current_tumour_size = None
    status = None
    for event in treatments:
        if not isinstance(event, Mapping):
            continue
        section = _normalise_label(event.get("section"))
        if section == "surgery":
            margin = _first(event, ("pathologist_margin_judgement", "pathologist_margin_judgment"), margin)
            reoperations = max(reoperations, _as_number(event.get("number_of_reoperations"), 0.0) or 0.0)
            current_tumour_size = _as_number(event.get("tumor_max_size_before_surgery"), current_tumour_size)
            surgery_type = _first(event, ("surgery_type", "surgery_types"), surgery_type)
        elif section == "radiotherapy":
            hyperthermia = max(hyperthermia, _as_flag(event.get("hyperthermia_status")))
            rt_indication = _first(event, ("radiotherapy_indication", "radiotherapy_indications"), rt_indication)
            rt_type = _first(event, ("radiotherapy_type",), rt_type)
            rt_dose_raw = _as_number(event.get("radiotherapy_total_dose"), rt_dose_raw)
            rt_dose = _as_number(event.get("radiotherapy_total_dose_gy"), rt_dose)
            rt_num_fractions = _as_number(event.get("radiotherapy_num_fractions"), rt_num_fractions)
            rt_dose_per_fraction = _as_number(event.get("radiotherapy_dose_per_fraction_gy"), rt_dose_per_fraction)
            rt_fractions = _as_number(event.get("radiotherapy_fraction_count"), rt_fractions)
            rt_interrupted = max(rt_interrupted, _as_flag(event.get("radiotherapy_interrupted")))
    for event in diagnosis:
        if isinstance(event, Mapping) and _present(event.get("last_status")):
            status = event.get("last_status")
    return {
        "margin_status": margin,
        "reoperation_count": reoperations,
        "hyperthermia": hyperthermia,
        "surgery_type": surgery_type,
        "pathologist_margin_judgment": margin,
        "number_of_reoperations": reoperations,
        "hyperthermia_status": hyperthermia,
        "radiotherapy_indication": rt_indication,
        "radiotherapy_type": rt_type,
        "radiotherapy_total_dose": rt_dose_raw,
        "interventional_radiology": _as_flag(episode.get("interventional_radiology")),
        "radiotherapy_total_dose_gy": rt_dose,
        "radiotherapy_num_fractions": rt_num_fractions,
        "radiotherapy_dose_per_fraction_gy": rt_dose_per_fraction,
        "radiotherapy_fraction_count": rt_fractions,
        "radiotherapy_interrupted": rt_interrupted,
        "tumour_size_before_surgery": current_tumour_size,
        "end_of_interval_status": status,
    }


def _baseline_values(patient: Mapping[str, Any], variables: Mapping[str, Any]) -> dict[str, Any]:
    values = {}
    excluded_semantics = {
        "patient_id", "surgery", "perioperative_radiotherapy", "margin_status", "reoperation_count",
        "hyperthermia", "interventional_radiology", "local_recurrence", "metastasis",
        "death_of_disease", "chemotherapy",
    }
    for semantic, spec in variables.items():
        if semantic in excluded_semantics or not isinstance(spec, Mapping):
            continue
        if spec.get("role") != "baseline_confounder":
            continue
        values[semantic] = _first(patient, spec.get("candidates", []))
    return values


def _flat_total(patient: Mapping[str, Any], candidates: Sequence[str]) -> int:
    return _as_flag(_first(patient, candidates, 0))


def _cohort_flow(total: int, exclusions: pd.DataFrame, included: int) -> pd.DataFrame:
    rows = [{"stage": "input_records", "n": total}]
    if not exclusions.empty:
        for reason, count in exclusions["reason"].value_counts(sort=False).items():
            rows.append({"stage": f"excluded:{reason}", "n": int(count)})
    rows.append({"stage": "included_surgery_rt_cohort", "n": included})
    return pd.DataFrame(rows)


def build_surgery_rt_cohort(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    config: Mapping[str, Any],
) -> SurgeryRTCohort:
    """Construct the restricted eight-interval surgery/RT cohort.

    Patients are assigned to ``S+RT`` only for explicitly preoperative or
    postoperative radiotherapy in the configured perioperative window.  An
    initial systemic-therapy course is an exclusion; a later course creates a
    strategy-deviation censoring event.
    """

    if isinstance(records, pd.DataFrame):
        patient_records = records.replace({np.nan: None}).to_dict(orient="records")
    else:
        patient_records = [dict(record) for record in records]

    variables = config.get("variables", {})
    population = config.get("population", {})
    time_config = config.get("time", {})
    n_intervals = int(time_config.get("n_intervals", 8))
    time_zero_candidates = time_config.get("time_zero_candidates", [])
    require_time_zero = bool(time_config.get("require_time_zero", True))

    person_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    for row_number, patient in enumerate(patient_records):
        patient_id = _first(patient, variables.get("patient_id", {}).get("candidates", []), row_number)
        histology = _first(patient, population.get("histology_candidates", []))
        in_scope, scope_source = _histology_in_scope(histology, population)
        if population.get("require_manuscript_histology", True) and not in_scope:
            exclusion_rows.append({"patient_id": patient_id, "reason": scope_source})
            continue
        if not in_scope:
            scope_source = "strict_malignant_soft_tissue_definition"

        time_zero = _date(_first(patient, time_zero_candidates))
        episodes = _episodes(patient)
        events = _events(patient)
        if pd.isna(time_zero) and episodes and time_config.get("allow_episode_time_zero_fallback", True):
            time_zero = _date(episodes[0].get("start_date"))

        if population.get("require_malignant_soft_tissue", False):
            grade = _normalise_label(
                _first(patient, variables.get("biopsy_grading", {}).get("candidates", []))
            )
            region = _normalise_label(
                _first(patient, variables.get("anatomic_region_grouping", {}).get("candidates", []))
            )
            eligible_grades = {
                _normalise_label(item) for item in population.get("malignant_grades", [])
            }
            eligible_regions = {
                _normalise_label(item) for item in population.get("soft_tissue_regions", [])
            }
            if grade not in eligible_grades or region not in eligible_regions:
                exclusion_rows.append({"patient_id": patient_id, "reason": "not_malignant_soft_tissue"})
                continue

        baseline_metastasis = _first(
            patient, variables.get("baseline_metastasis", {}).get("candidates", [])
        )
        prediagnosis_metastasis = _first(
            patient,
            ("treatments.pre_diagnosis_metastasis", "pre_diagnosis_metastasis"),
            0,
        )
        if population.get("require_localized_at_baseline", False):
            if _as_number(baseline_metastasis) != 0 or _as_flag(prediagnosis_metastasis):
                exclusion_rows.append({"patient_id": patient_id, "reason": "not_localized_at_baseline"})
                continue

        surgery_event = _definitive_surgery(events, population)
        flat = not bool(episodes or events)
        if surgery_event is None and flat:
            surgery_spec = variables.get("surgery", {}).get("candidates", [])
            if _flat_total(patient, surgery_spec):
                surgery_event = {"section": "surgery", "date_field": time_zero, "surgery_type": "flat_unspecified"}
        if surgery_event is None:
            exclusion_rows.append({"patient_id": patient_id, "reason": "no_definitive_surgery"})
            continue
        surgery_date = _event_date(surgery_event)
        if pd.isna(surgery_date):
            exclusion_rows.append({"patient_id": patient_id, "reason": "missing_definitive_surgery_date"})
            continue
        if pd.isna(time_zero) and require_time_zero:
            exclusion_rows.append({"patient_id": patient_id, "reason": "missing_time_zero"})
            continue

        lr_date = _first_event_date(events, {"local-recurrence", "local recurrence"})
        # Normalisation turns underscores into hyphens.
        if pd.isna(lr_date):
            lr_date = _first_event_date(events, {"local-recurrence"})
        if population.get("exclude_baseline_local_recurrence", False):
            prediagnosis_lr = _first(
                patient,
                variables.get("prediagnosis_local_recurrence", {}).get("candidates", []),
                0,
            )
            if _as_flag(prediagnosis_lr) or (pd.notna(lr_date) and lr_date <= time_zero):
                exclusion_rows.append({"patient_id": patient_id, "reason": "baseline_local_recurrence"})
                continue
        dod_date = _first_event_date(events, {"follow-up", "follow up"}, dod=True)
        # The analysis endpoint is local recurrence; all-cause death is its
        # competing event. Metastasis is not part of the modeled outcome
        # process for this restricted analysis.
        outcome_dates = [item for item in (lr_date, dod_date) if pd.notna(item)]
        first_outcome = min(outcome_dates) if outcome_dates else pd.NaT

        rt = _classify_radiotherapy(patient, events, surgery_date, config, first_outcome)
        rt_exclusion_reason = None
        if flat:
            any_rt = _flat_total(patient, variables.get("perioperative_radiotherapy", {}).get("candidates", []))
            indication = _normalise_label(_first(patient, ("radiotherapy_indication", "redonc_indication")))
            if any_rt and indication in {"preoperative", "postoperative"}:
                rt.update({"treated": 1, "date": surgery_date, "indication": indication, "ambiguous": False, "any_rt": True})
            elif any_rt and not rt["treated"]:
                rt_exclusion_reason = "ambiguous_radiotherapy_timing"
        elif rt["ambiguous"]:
            rt_exclusion_reason = "ambiguous_perioperative_radiotherapy"
        nonqualifying_policy = config.get("treatment", {}).get("radiotherapy", {}).get(
            "nonqualifying_radiotherapy_policy", "keep_as_s"
        )
        exclude_nonqualifying = rt["has_nonqualifying"] and (
            nonqualifying_policy == "exclude"
            or (nonqualifying_policy == "exclude_without_qualifying" and not rt["treated"])
        )
        if exclude_nonqualifying:
            rt_exclusion_reason = "nonqualifying_radiotherapy"

        chemotherapy_events = [event for event in events if _normalise_label(event.get("section")) == "systemic-therapy"]
        chemo_dates = sorted(_event_date(event) for event in chemotherapy_events if pd.notna(_event_date(event)))
        chemotherapy_policy = population.get("chemotherapy_policy", "exclude_any")
        if chemotherapy_policy == "exclude_any":
            # Strict D excludes any recorded systemic therapy, including an
            # event whose exact date is unavailable.
            initial_chemo = chemo_dates or ([surgery_date] if chemotherapy_events else [])
            later_chemo = []
        else:
            initial_cutoff = surgery_date + pd.Timedelta(
                days=int(population.get("initial_chemotherapy_window_days_after_surgery", 180))
            )
            initial_chemo = [when for when in chemo_dates if when <= initial_cutoff]
            later_chemo = [when for when in chemo_dates if when > initial_cutoff]
        if flat:
            any_chemo = _flat_total(patient, variables.get("chemotherapy", {}).get("candidates", []))
            if any_chemo and chemotherapy_policy == "exclude_any":
                initial_chemo = [surgery_date]
        if initial_chemo:
            exclusion_rows.append({"patient_id": patient_id, "reason": "chemotherapy_outside_strategies"})
            continue
        if rt_exclusion_reason is not None:
            exclusion_rows.append({"patient_id": patient_id, "reason": rt_exclusion_reason})
            continue

        treatment_arm = "S+RT" if rt["treated"] else "S"
        baseline = _baseline_values(patient, variables)
        qualifying_indications = {
            indication for _when, _event, indication in rt["qualifying_courses"]
        }
        has_preoperative_rt = any(
            indication.startswith("preoperative") for indication in qualifying_indications
        )
        has_postoperative_rt = "postoperative" in qualifying_indications
        preoperative_dates = sorted(
            when
            for when, _event, indication in rt["qualifying_courses"]
            if indication.startswith("preoperative") and pd.notna(when)
        )
        postoperative_dates = sorted(
            when
            for when, _event, indication in rt["qualifying_courses"]
            if indication == "postoperative" and pd.notna(when)
        )
        preoperative_rt_date = preoperative_dates[0] if preoperative_dates else pd.NaT
        postoperative_rt_date = postoperative_dates[0] if postoperative_dates else pd.NaT
        if has_preoperative_rt and has_postoperative_rt:
            rt_timing = "preoperative + postoperative"
        elif has_preoperative_rt:
            rt_timing = "preoperative"
        elif has_postoperative_rt:
            rt_timing = "postoperative"
        else:
            rt_timing = "no RT"
        primary_surgery_type = _first(surgery_event, ("surgery_type", "surgery_types"))
        primary_surgery_margin = _first(
            surgery_event,
            ("pathologist_margin_judgement", "pathologist_margin_judgment"),
        )
        primary_surgery_reoperations = _as_number(
            surgery_event.get("number_of_reoperations"), 0.0
        )
        historical_preoperative_rt = int(
            pd.notna(preoperative_rt_date) and preoperative_rt_date < time_zero
        )
        historical_postoperative_rt = int(
            pd.notna(postoperative_rt_date) and postoperative_rt_date < time_zero
        )
        eligible_preoperative_rt = int(
            surgery_date >= time_zero and not historical_preoperative_rt
        )
        histology_scope_approximate = int(scope_source == "grouped_code")
        episode_by_idx: dict[int, Mapping[str, Any]] = {}
        for fallback, episode in enumerate(episodes[:n_intervals]):
            idx = int(_as_number(episode.get("episode_id"), fallback + 1) or fallback + 1) - 1
            if 0 <= idx < n_intervals:
                episode_by_idx[idx] = episode

        lr_idx = _interval_index(lr_date, episodes, time_zero, n_intervals)
        dod_idx = _interval_index(dod_date, episodes, time_zero, n_intervals)
        qualifying_course_indices = [
            (_interval_index(when, episodes, time_zero, n_intervals), indication)
            for when, _event, indication in rt["qualifying_courses"]
        ]
        observed_preoperative_courses = [
            (_event_date(event), "preoperative_observed")
            for event in events
            if _normalise_label(event.get("section")) == "radiotherapy"
            and "preoperative" in _rt_indications(event)
        ]
        proxy_preoperative_courses = [
            (when, indication)
            for when, _event, indication in rt["qualifying_courses"]
            if indication == "preoperative_top_level_flag_date_unavailable"
        ]
        preoperative_courses_all = [
            (when, indication)
            for when, indication in (
                observed_preoperative_courses + proxy_preoperative_courses
            )
            if pd.notna(when)
        ]
        preoperative_courses_by_interval: dict[
            int, list[tuple[pd.Timestamp, str]]
        ] = {}
        for when, indication in preoperative_courses_all:
            course_interval = _interval_index(
                when, episodes, time_zero, n_intervals
            )
            if course_interval is not None:
                preoperative_courses_by_interval.setdefault(
                    course_interval, []
                ).append((when, indication))
        preoperative_rt_indices = set(preoperative_courses_by_interval)
        postoperative_rt_indices = {
            idx for idx, indication in qualifying_course_indices
            if idx is not None and indication == "postoperative"
        }
        qualifying_rt_indices = preoperative_rt_indices | postoperative_rt_indices
        nonqualifying_rt_indices = {
            idx
            for when, _event, _indication in rt["nonqualifying_courses"]
            if (idx := _interval_index(when, episodes, time_zero, n_intervals)) is not None
        }
        surgery_idx = _interval_index(surgery_date, episodes, time_zero, n_intervals)
        later_chemo_date = min(later_chemo) if later_chemo else pd.NaT
        deviation_idx = _interval_index(later_chemo_date, episodes, time_zero, n_intervals)

        prior_lr = int(pd.notna(lr_date) and lr_date < time_zero)
        prior_dod = int(pd.notna(dod_date) and dod_date < time_zero)
        prior_rt = 0
        prior_preoperative_rt = 0
        prior_margin = None
        cumulative_reoperations = 0.0
        prior_status = None
        first_unobserved_marked = False
        censoring_failure_seen = False
        for interval in range(n_intervals):
            episode = episode_by_idx.get(interval)
            interval_start_date = _date(episode.get("start_date")) if episode else pd.NaT
            interval_end_date = _date(episode.get("end_date")) if episode else pd.NaT
            if pd.isna(interval_start_date):
                interval_start_date = time_zero + pd.DateOffset(
                    months=interval * int(time_config.get("interval_months", 6))
                )
            if pd.isna(interval_end_date):
                interval_end_date = time_zero + pd.DateOffset(
                    months=(interval + 1) * int(time_config.get("interval_months", 6))
                ) - pd.Timedelta(days=1)
            observed = int(episode is not None)
            after_strategy_deviation = deviation_idx is not None and interval > deviation_idx
            if after_strategy_deviation or censoring_failure_seen:
                observed = 0
            # An event dated after the 48-month analysis horizon is correctly
            # event-free in every included interval. The original date remains
            # available in patient_summary for auditing.
            incident_lr = int(lr_idx == interval) if lr_idx is not None else 0
            incident_dod = int(dod_idx == interval) if dod_idx is not None else 0
            episode_values = _episode_features(episode or {})
            strategy_deviation = int(deviation_idx == interval)
            loss_to_followup = int(not observed and not after_strategy_deviation and not first_unobserved_marked)
            if loss_to_followup:
                first_unobserved_marked = True
            censoring_at_risk = int(not censoring_failure_seen)
            current_preoperative_rt = int(interval in preoperative_rt_indices)
            current_postoperative_rt = int(interval in postoperative_rt_indices)
            current_rt = int(interval in qualifying_rt_indices)
            current_nonqualifying_rt = int(interval in nonqualifying_rt_indices)
            current_surgery = int(surgery_idx == interval) if surgery_idx is not None else 0

            preoperative_courses = preoperative_courses_by_interval.get(interval, [])
            preoperative_validities = [
                _administration_before_recurrence(
                    when,
                    lr_date,
                    date_resolved=(
                        indication != "preoperative_top_level_flag_date_unavailable"
                    ),
                )
                for when, indication in preoperative_courses
            ]
            preoperative_availabilities = [
                _available_at_prediction_time(
                    when,
                    interval_start_date,
                    date_resolved=(
                        indication != "preoperative_top_level_flag_date_unavailable"
                    ),
                )
                for when, indication in preoperative_courses
            ]
            preoperative_rt_valid_for_lr = int(
                any(item.valid > 0.5 for item in preoperative_validities)
            )
            preoperative_rt_order_resolved = int(
                bool(preoperative_validities)
                and all(item.resolved > 0.5 for item in preoperative_validities)
            )
            preoperative_rt_available_at_interval_start = int(
                any(item.valid > 0.5 for item in preoperative_availabilities)
            )
            preoperative_rt_availability_resolved = int(
                bool(preoperative_availabilities)
                and all(item.resolved > 0.5 for item in preoperative_availabilities)
            )

            record = {
                "patient_id": patient_id,
                "interval": interval,
                "interval_start_month": interval * int(time_config.get("interval_months", 6)),
                "interval_end_month": (interval + 1) * int(time_config.get("interval_months", 6)),
                "interval_start_date": interval_start_date,
                "interval_end_date": interval_end_date,
                "time_zero": time_zero,
                "preoperative_rt_decision_date": time_zero,
                "eligible_preoperative_rt": eligible_preoperative_rt,
                "historical_preoperative_rt": historical_preoperative_rt,
                "historical_postoperative_rt": historical_postoperative_rt,
                "preoperative_rt_date": preoperative_rt_date,
                "postoperative_rt_date": postoperative_rt_date,
                "local_recurrence_date": lr_date,
                "dod_date": dod_date,
                "treatment_arm": treatment_arm,
                "perioperative_radiotherapy": int(rt["treated"]),
                "preoperative_radiotherapy": int(has_preoperative_rt),
                "postoperative_radiotherapy": int(has_postoperative_rt),
                "primary_surgery_type": primary_surgery_type,
                "primary_surgery_margin": primary_surgery_margin,
                "primary_surgery_reoperations": primary_surgery_reoperations,
                "radiotherapy_this_interval": current_rt,
                "preoperative_rt_this_interval": current_preoperative_rt,
                "preoperative_rt_valid_for_lr": preoperative_rt_valid_for_lr,
                "preoperative_rt_order_resolved": preoperative_rt_order_resolved,
                "preoperative_rt_available_at_interval_start": (
                    preoperative_rt_available_at_interval_start
                ),
                "preoperative_rt_availability_resolved": (
                    preoperative_rt_availability_resolved
                ),
                "postoperative_rt_this_interval": current_postoperative_rt,
                "nonqualifying_rt_this_interval": current_nonqualifying_rt,
                "surgery_this_interval": current_surgery,
                "observed": observed,
                "censoring_at_risk": censoring_at_risk,
                "at_risk_local_recurrence": int(not prior_lr and not prior_dod),
                "at_risk_dod": int(not prior_dod),
                "local_recurrence": incident_lr,
                "dod": incident_dod,
                "lag_local_recurrence": prior_lr,
                "lag_dod": prior_dod,
                "lag_radiotherapy": prior_rt,
                "lag_preoperative_radiotherapy": prior_preoperative_rt,
                "lag_margin_status": prior_margin,
                "lag_reoperation_count": cumulative_reoperations,
                "lag_disease_status": prior_status,
                "strategy_deviation_chemotherapy": strategy_deviation,
                "strategy_deviation_nonqualifying_rt": current_nonqualifying_rt,
                "censored": int(strategy_deviation or loss_to_followup),
                "loss_to_followup": loss_to_followup,
                "same_interval_event_order_ambiguous": int(
                    strategy_deviation and any(value == 1 for value in (incident_lr, incident_dod))
                ),
                "histology_scope_approximate": histology_scope_approximate,
                **baseline,
                **episode_values,
            }
            person_rows.append(record)

            if incident_lr == 1:
                prior_lr = 1
            if incident_dod == 1:
                prior_dod = 1
            if observed == 0:
                censoring_failure_seen = True
            prior_rt = int(prior_rt or current_rt)
            prior_preoperative_rt = int(prior_preoperative_rt or current_preoperative_rt)
            if _present(episode_values["margin_status"]):
                prior_margin = episode_values["margin_status"]
            cumulative_reoperations = max(cumulative_reoperations, episode_values["reoperation_count"] or 0.0)
            if _present(episode_values["end_of_interval_status"]):
                prior_status = episode_values["end_of_interval_status"]

        summaries.append(
            {
                "patient_id": patient_id,
                "histology": histology,
                "histology_scope_source": scope_source,
                "time_zero": time_zero,
                "definitive_surgery_date": surgery_date,
                "preoperative_rt_decision_date": time_zero,
                "eligible_preoperative_rt": eligible_preoperative_rt,
                "historical_preoperative_rt": historical_preoperative_rt,
                "historical_postoperative_rt": historical_postoperative_rt,
                "preoperative_rt_date": preoperative_rt_date,
                "postoperative_rt_date": postoperative_rt_date,
                "treatment_arm": treatment_arm,
                "perioperative_radiotherapy": int(rt["treated"]),
                "preoperative_radiotherapy": int(has_preoperative_rt),
                "postoperative_radiotherapy": int(has_postoperative_rt),
                "rt_timing": rt_timing,
                "primary_surgery_type": primary_surgery_type,
                "primary_surgery_margin": primary_surgery_margin,
                "primary_surgery_reoperations": primary_surgery_reoperations,
                "n_qualifying_rt_courses": len(rt["qualifying_courses"]),
                "n_nonqualifying_rt_courses": len(rt["nonqualifying_courses"]),
                "mixed_rt_history": int(bool(rt["treated"] and rt["has_nonqualifying"])),
                "perioperative_rt_date": rt["date"],
                "perioperative_rt_indication": rt["indication"],
                "allowed_downstream_rt_indications": " | ".join(
                    rt["allowed_nonexposure_indications"]
                ),
                "any_nonqualifying_radiotherapy": int(rt["has_nonqualifying"]),
                "later_chemotherapy_date": later_chemo_date,
                "strategy_deviation_chemotherapy": int(pd.notna(later_chemo_date)),
                "local_recurrence_date": lr_date,
                "dod_date": dod_date,
                "local_recurrence_48m": int(lr_idx is not None),
                "dod_48m": int(dod_idx is not None),
                "n_source_episodes": len(episodes),
                "n_analysis_intervals": n_intervals,
                **baseline,
            }
        )

    exclusions = pd.DataFrame(exclusion_rows, columns=["patient_id", "reason"])
    summary = pd.DataFrame(summaries)
    person_period = pd.DataFrame(person_rows)
    flow = _cohort_flow(len(patient_records), exclusions, len(summary))
    rt_config = config.get("treatment", {}).get("radiotherapy", {})
    eligible_indications = list(rt_config.get("eligible_indications", []))
    allowed_nonexposure = list(rt_config.get("allowed_nonexposure_indications", []))
    metadata = {
        "n_input": len(patient_records),
        "n_included": len(summary),
        "n_excluded": len(exclusions),
        "n_intervals": n_intervals,
        "treatment_contrast": "preoperative RT plus surgery versus surgery alone",
        "rt_definition": "explicit preoperative RT before surgery and outcome",
        "assignment_rule": rt_config.get("assignment_rule", "window_before_outcome"),
        "eligible_rt_indications": eligible_indications,
        "allowed_downstream_rt_indications": allowed_nonexposure,
        "postoperative_policy": "exclude postoperative-only and combined RT",
    }
    return SurgeryRTCohort(person_period, summary, exclusions, flow, metadata)


def save_surgery_rt_cohort(cohort: SurgeryRTCohort, output_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Path]:
    """Write cohort artifacts without modifying source data."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = config.get("outputs", {})
    paths = {
        "person_period": output / names.get("person_period", "person_period.csv"),
        "patient_summary": output / names.get("patient_summary", "patient_summary.csv"),
        "exclusions": output / names.get("exclusions", "exclusions.csv"),
        "cohort_flow": output / names.get("cohort_flow", "cohort_flow.csv"),
        "metadata": output / "cohort_metadata.json",
    }
    cohort.person_period.to_csv(paths["person_period"], index=False)
    cohort.patient_summary.to_csv(paths["patient_summary"], index=False)
    cohort.exclusions.to_csv(paths["exclusions"], index=False)
    cohort.cohort_flow.to_csv(paths["cohort_flow"], index=False)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(cohort.metadata, handle, indent=2, default=str)
    return paths


__all__ = [
    "SurgeryRTCohort",
    "build_surgery_rt_cohort",
    "configure_preoperative_rt",
    "load_causal_config",
    "load_patient_records",
    "save_surgery_rt_cohort",
]
