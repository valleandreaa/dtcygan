"""Strict-D parent-cohort training adapter with explicit causal feature groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FORBIDDEN_PREDICTORS = {
    "local_recurrence",
    "dod",
    "observed",
    "censored",
    "loss_to_followup",
    "at_risk_local_recurrence",
    "at_risk_dod",
    "treatment_arm",
}

PREOP_TWO_ARM_REGIMENS = {
    "surgery_only": (0, 0),
    "preoperative_rt_surgery": (1, 0),
}
PREOP_TWO_ARM_CONDITION_COLUMNS = list(PREOP_TWO_ARM_REGIMENS)

FORBIDDEN_PREOP_MODEL_FEATURES = {
    "postoperative_radiotherapy",
    "postoperative_rt_this_interval",
    "postoperative_rt_date",
    "lag_postoperative_radiotherapy",
}


def _as_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, dict):
        return [dict(value)]
    return []


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_names(spec: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("numeric", "binary", "categorical"):
        names.extend(str(value) for value in spec.get(key, []) or [])
    names.extend(str(value) for value in (spec.get("ordinal", {}) or {}))
    return list(dict.fromkeys(names))


def _trajectory_resolution(frame: pd.DataFrame) -> str:
    """Describe the observed person-period width without assuming six months."""

    required = {"interval_start_month", "interval_end_month"}
    if not required.issubset(frame.columns):
        return "time_indexed_intervals"
    widths = (
        pd.to_numeric(frame["interval_end_month"], errors="coerce")
        - pd.to_numeric(frame["interval_start_month"], errors="coerce")
    ).dropna().unique()
    if len(widths) != 1 or float(widths[0]) <= 0:
        return "variable_month_intervals"
    width = float(widths[0])
    if not width.is_integer():
        return f"{width:g}_month_intervals"
    names = {3: "three", 6: "six"}
    integer_width = int(width)
    prefix = names.get(integer_width, str(integer_width))
    return f"{prefix}_month_intervals"


def _assign_two_way_splits(
    patients: Sequence[dict[str, Any]],
    seed: int,
    test_fraction: float,
) -> None:
    """Assign deterministic patient-level dev/test labels by causal strata."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")
    if len(patients) < 2:
        raise ValueError("Two-way splitting requires at least two patients")

    strata: dict[tuple[int, int, int], list[int]] = {}
    for index, patient in enumerate(patients):
        if "regimen_index" in patient:
            treatment_stratum = int(patient["regimen_index"])
        else:
            treatment_stratum = int(
                patient["actual_treatment"][0]["preoperative_radiotherapy"]
            )
        eligible = int(patient["eligible_preoperative_rt"])
        recurrence = int(
            any(float(record.get("local_recurrence", 0.0)) > 0.5 for record in patient["outcome"])
        )
        strata.setdefault((treatment_stratum, eligible, recurrence), []).append(index)

    test_target = max(1, min(len(patients) - 1, round(len(patients) * test_fraction)))
    allocations: dict[tuple[int, int, int], int] = {}
    remainders: list[tuple[float, tuple[int, int, int]]] = []
    for stratum, indices in strata.items():
        exact = len(indices) * test_fraction
        maximum = len(indices) - 1 if len(indices) > 1 else 0
        allocated = min(int(np.floor(exact)), maximum)
        allocations[stratum] = allocated
        remainders.append((exact - np.floor(exact), stratum))

    rng = np.random.default_rng(seed)
    tie_breakers = {stratum: float(rng.random()) for stratum in strata}
    ordered = [
        stratum
        for _, stratum in sorted(
            remainders,
            key=lambda item: (-item[0], tie_breakers[item[1]], item[1]),
        )
    ]
    while sum(allocations.values()) < test_target:
        changed = False
        for stratum in ordered:
            maximum = len(strata[stratum]) - 1 if len(strata[stratum]) > 1 else 0
            if allocations[stratum] < maximum:
                allocations[stratum] += 1
                changed = True
                if sum(allocations.values()) == test_target:
                    break
        if not changed:
            raise ValueError("Causal strata cannot support the requested test fraction")

    for patient in patients:
        patient["split"] = "dev"
    for stratum, indices in strata.items():
        shuffled = np.asarray(indices, dtype=int)
        rng.shuffle(shuffled)
        for index in shuffled[: allocations[stratum]]:
            patients[int(index)]["split"] = "test"


def _sarcoma_test_fraction(
    validation_fraction: float,
    test_fraction: float | None,
) -> float:
    """Resolve the new name while accepting older programmatic callers."""

    return float(validation_fraction if test_fraction is None else test_fraction)


def _normalize_sarcoma_split_label(label: Any) -> str:
    normalized = str(label)
    aliases = {"train": "dev", "validation": "test"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"dev", "test"}:
        raise ValueError(f"Unsupported explicit split label: {label!r}")
    return normalized


def build_sarcoma_sequence_records(
    data: pd.DataFrame | str | Path,
    feature_groups: Mapping[str, Any],
    seq_len: int = 8,
    cohort_contract: Mapping[str, Any] | None = None,
    split_seed: int | None = None,
    validation_fraction: float = 0.20,
    test_fraction: float | None = None,
) -> dict[str, Any]:
    """Convert strict-D person-period rows to causally separated patient sequences.

    The default contract retains the complete parent cohort.  Set
    ``eligible_preoperative_rt_only`` restricts the preoperative-RT risk set.
    """

    if isinstance(data, (str, Path)):
        frame = pd.read_csv(data, dtype={"patient_id": "string"})
    else:
        frame = data.copy()
        frame["patient_id"] = frame["patient_id"].astype("string")
    if frame.empty:
        raise ValueError("Strict-D person-period data are empty")

    group_columns = {
        group: _feature_names(feature_groups.get(group, {}) or {})
        for group in ("baseline", "state", "treatment", "outcome")
    }
    if group_columns["outcome"] != ["local_recurrence"]:
        raise ValueError(
            "The strict-D surgery/RT training outcome must be local_recurrence only"
        )
    forbidden = FORBIDDEN_PREDICTORS.intersection(
        set(group_columns["baseline"])
        | set(group_columns["state"])
        | set(group_columns["treatment"])
    )
    if forbidden:
        raise ValueError(
            "Outcome, observation, or future-summary variables cannot be predictors: "
            f"{sorted(forbidden)}"
        )
    required = {
        "patient_id",
        "interval",
        "eligible_preoperative_rt",
        "preoperative_radiotherapy",
        "preoperative_rt_this_interval",
        "postoperative_rt_this_interval",
        "observed",
        "censoring_at_risk",
        "at_risk_local_recurrence",
        "strategy_deviation_chemotherapy",
        "strategy_deviation_nonqualifying_rt",
        *group_columns["baseline"],
        *group_columns["state"],
        *group_columns["treatment"],
        *group_columns["outcome"],
    }
    required.discard("interval_scaled")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training person-period data are missing: {sorted(missing)}")
    if frame.duplicated(["patient_id", "interval"]).any():
        raise ValueError("Training data must contain at most one row per patient and interval")

    contract = dict(cohort_contract or {})
    if bool(contract.get("eligible_preoperative_rt_only", False)):
        eligibility = frame.groupby("patient_id", sort=False)["eligible_preoperative_rt"].first()
        eligible_ids = eligibility[eligibility.eq(1)].index
        frame = frame.loc[frame["patient_id"].isin(eligible_ids)].copy()
    frame["interval"] = pd.to_numeric(frame["interval"], errors="raise").astype(int)
    frame = frame.sort_values(["patient_id", "interval"]).reset_index(drop=True)

    patients: list[dict[str, Any]] = []
    for patient_id, patient in frame.groupby("patient_id", sort=False):
        intervals = patient["interval"].tolist()
        if intervals != list(range(seq_len)):
            raise ValueError(
                f"Patient {patient_id!r} must have intervals 0..{seq_len - 1}; "
                f"observed {intervals}"
            )
        for column in group_columns["baseline"]:
            if patient[column].dropna().nunique() > 1:
                raise ValueError(
                    f"Baseline variable {column!r} changes within patient {patient_id!r}"
                )
        if patient["preoperative_radiotherapy"].nunique(dropna=False) != 1:
            raise ValueError(f"Preoperative RT changes within patient {patient_id!r}")

        first = patient.iloc[0]
        baseline_record = {
            column: _json_value(first[column]) for column in group_columns["baseline"]
        }
        if "whoops_margin_status" in baseline_record:
            if int(first.get("any_whoops", 0) or 0) == 0:
                baseline_record["whoops_margin_status"] = "no_whoops"
            elif baseline_record["whoops_margin_status"] is None:
                baseline_record["whoops_margin_status"] = "unknown_whoops_margin"

        state_records: list[dict[str, Any]] = []
        treatment_records: list[dict[str, Any]] = []
        assignment_records: list[dict[str, Any]] = []
        outcome_records: list[dict[str, Any]] = []
        outcome_mask_records: list[dict[str, Any]] = []
        sequence_mask: list[float] = []
        strategy_active = True
        preoperative_rt = int(first["preoperative_radiotherapy"])
        for _, row in patient.iterrows():
            deviates = bool(
                int(row["strategy_deviation_chemotherapy"])
                or int(row["strategy_deviation_nonqualifying_rt"])
            )
            if deviates:
                strategy_active = False
            observed = bool(int(row["observed"])) and strategy_active
            sequence_mask.append(float(observed))

            state_record = {
                column: (
                    float(row["interval"]) / max(1, seq_len - 1)
                    if column == "interval_scaled"
                    else _json_value(row[column])
                )
                for column in group_columns["state"]
            }
            preoperative_value = pd.to_numeric(
                pd.Series([row["preoperative_rt_this_interval"]]),
                errors="coerce",
            ).iloc[0]
            postoperative_value = pd.to_numeric(
                pd.Series([row["postoperative_rt_this_interval"]]),
                errors="coerce",
            ).iloc[0]
            qualifying_rt = bool(
                (pd.notna(preoperative_value) and int(preoperative_value))
                or (pd.notna(postoperative_value) and int(postoperative_value))
            )
            treatment_record = {
                column: _json_value(row[column])
                for column in group_columns["treatment"]
            }
            version_features = set(
                feature_groups.get("treatment", {}).get("version_features", []) or []
            )
            if not qualifying_rt:
                for column in version_features:
                    if column in treatment_record:
                        treatment_record[column] = None

            outcome_record: dict[str, float] = {}
            for column in group_columns["outcome"]:
                value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                outcome_record[column] = float(value) if pd.notna(value) else 0.0
            mask_record: dict[str, float] = {}
            for column in group_columns["outcome"]:
                if column == "local_recurrence":
                    at_risk = bool(int(row["at_risk_local_recurrence"]))
                elif column == "dod":
                    at_risk = bool(int(row["at_risk_dod"]))
                else:
                    at_risk = True
                mask_record[column] = float(
                    observed and bool(int(row["censoring_at_risk"])) and at_risk
                )

            state_records.append(state_record)
            treatment_records.append(treatment_record)
            assignment_records.append({"preoperative_radiotherapy": float(preoperative_rt)})
            outcome_records.append(outcome_record)
            outcome_mask_records.append(mask_record)

        patients.append(
            {
                "patient_id": str(patient_id),
                "eligible_preoperative_rt": int(first["eligible_preoperative_rt"]),
                "baseline": baseline_record,
                "clinical_state": state_records,
                "treatment": treatment_records,
                "actual_treatment": assignment_records,
                "outcome": outcome_records,
                "outcome_mask": outcome_mask_records,
                "sequence_mask": sequence_mask,
            }
        )

    if split_seed is not None:
        _assign_two_way_splits(
            patients,
            seed=int(split_seed),
            test_fraction=_sarcoma_test_fraction(validation_fraction, test_fraction),
        )

    treated = sum(
        int(patient["actual_treatment"][0]["preoperative_radiotherapy"])
        for patient in patients
    )
    observed_counts = {
        "n": len(patients),
        "treated": treated,
        "untreated": len(patients) - treated,
    }
    for key, expected_key in (
        ("n", "expected_n"),
        ("treated", "expected_treated"),
        ("untreated", "expected_untreated"),
    ):
        expected = contract.get(expected_key)
        if expected is not None and observed_counts[key] != int(expected):
            raise ValueError(
                f"Preoperative RT training cohort contract failed for {key}: "
                f"expected {expected}, observed {observed_counts[key]}"
            )
    return {
        "data_contract": "sarcoma_records",
        "patients": patients,
        "cohort": observed_counts,
        "seq_len": seq_len,
        "split_counts": (
            {
                name: sum(patient.get("split") == name for patient in patients)
                for name in ("dev", "test")
            }
            if split_seed is not None
            else None
        ),
    }


def build_surgery_rt_preop_two_arm_training_data(
    data: pd.DataFrame | str | Path,
    feature_groups: Mapping[str, Any],
    seq_len: int = 8,
    cohort_contract: Mapping[str, Any] | None = None,
    split_seed: int | None = None,
    validation_fraction: float = 0.20,
    test_fraction: float | None = None,
    split_assignments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the preoperative-RT-plus-surgery versus surgery-only cohort.

    Postoperative-only and combined pre/postoperative RT patients are excluded
    before patient-level splitting. The generated trajectories remain time
    indexed and retain the strict-D censoring masks from the parent adapter.
    """

    if isinstance(data, (str, Path)):
        frame = pd.read_csv(data, dtype={"patient_id": "string"})
    else:
        frame = data.copy()
        frame["patient_id"] = frame["patient_id"].astype("string")
    configured_features = set()
    for group in ("baseline", "state", "treatment", "outcome"):
        configured_features.update(
            _feature_names(feature_groups.get(group, {}) or {})
        )
    postoperative_features = configured_features & FORBIDDEN_PREOP_MODEL_FEATURES
    if postoperative_features:
        raise ValueError(
            "The preoperative two-arm contract cannot model postoperative RT "
            f"features: {sorted(postoperative_features)}"
        )

    contract = dict(cohort_contract or {})
    temporal_columns = {
        "preoperative_rt_valid_for_lr",
        "preoperative_rt_order_resolved",
        "preoperative_rt_available_at_interval_start",
        "preoperative_rt_availability_resolved",
    }
    if bool(contract.get("require_temporal_order_fields", False)):
        missing_temporal = temporal_columns - set(frame.columns)
        if missing_temporal:
            raise ValueError(
                "Chronology-safe preoperative training requires rebuilt "
                "person-period data with temporal fields: "
                f"{sorted(missing_temporal)}"
            )
    required = {
        "patient_id",
        "preoperative_radiotherapy",
        "postoperative_radiotherapy",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Two-arm training data are missing: {sorted(missing)}")

    assignment = (
        frame.groupby("patient_id", sort=False)[
            ["preoperative_radiotherapy", "postoperative_radiotherapy"]
        ]
        .first()
        .apply(pd.to_numeric, errors="raise")
        .astype(int)
    )
    if not assignment.isin([0, 1]).all().all():
        raise ValueError("Preoperative and postoperative RT assignments must be binary")
    combined_mask = assignment["preoperative_radiotherapy"].eq(1) & assignment[
        "postoperative_radiotherapy"
    ].eq(1)
    postoperative_mask = assignment["preoperative_radiotherapy"].eq(0) & assignment[
        "postoperative_radiotherapy"
    ].eq(1)
    combined_ids = assignment.index[combined_mask]
    postoperative_ids = assignment.index[postoperative_mask]
    excluded_ids = combined_ids.union(postoperative_ids)
    filtered = frame.loc[~frame["patient_id"].isin(excluded_ids)].copy()

    if temporal_columns.issubset(filtered.columns):
        raw_preoperative = pd.to_numeric(
            filtered["preoperative_rt_this_interval"], errors="raise"
        ).astype(int)
        valid_preoperative = pd.to_numeric(
            filtered["preoperative_rt_valid_for_lr"], errors="raise"
        ).astype(int)
        resolved_preoperative = pd.to_numeric(
            filtered["preoperative_rt_order_resolved"], errors="raise"
        ).astype(int)
        if not raw_preoperative.isin([0, 1]).all():
            raise ValueError("Preoperative administrations must be binary")
        if not valid_preoperative.isin([0, 1]).all():
            raise ValueError("Event-aligned preoperative administrations must be binary")
        if not resolved_preoperative.isin([0, 1]).all():
            raise ValueError("Preoperative ordering resolution must be binary")
        if ((valid_preoperative == 1) & (raw_preoperative == 0)).any():
            raise ValueError(
                "A valid preoperative administration requires a raw administration"
            )

        # The raw administration remains in the person-period artifact.  The
        # model-facing copy fails closed for post-recurrence or unresolved
        # administrations and removes their associated regimen parameters.
        invalid_preoperative = raw_preoperative.eq(1) & (
            valid_preoperative.eq(0) | resolved_preoperative.eq(0)
        )
        filtered["preoperative_rt_this_interval"] = valid_preoperative
        filtered.loc[
            invalid_preoperative, "preoperative_rt_this_interval"
        ] = np.nan
        aligned_preoperative_assignment = (
            filtered.assign(_valid_preoperative=valid_preoperative)
            .groupby("patient_id", sort=False)["_valid_preoperative"]
            .max()
            .astype(int)
        )
        filtered["preoperative_radiotherapy"] = filtered["patient_id"].map(
            aligned_preoperative_assignment
        )
        for column in (
            feature_groups.get("treatment", {}).get("version_features", []) or []
        ):
            if column in filtered.columns:
                filtered.loc[invalid_preoperative, column] = np.nan

    base = build_sarcoma_sequence_records(
        filtered,
        feature_groups,
        seq_len=seq_len,
        cohort_contract={"eligible_preoperative_rt_only": False},
        split_seed=None,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    counts = {name: 0 for name in PREOP_TWO_ARM_CONDITION_COLUMNS}
    for patient in base["patients"]:
        preoperative = int(
            patient["actual_treatment"][0]["preoperative_radiotherapy"]
        )
        regimen = (
            "preoperative_rt_surgery" if preoperative else "surgery_only"
        )
        regimen_index = PREOP_TWO_ARM_CONDITION_COLUMNS.index(regimen)
        one_hot = {
            name: float(index == regimen_index)
            for index, name in enumerate(PREOP_TWO_ARM_CONDITION_COLUMNS)
        }
        patient["regimen"] = regimen
        patient["regimen_index"] = regimen_index
        patient["actual_treatment"] = [dict(one_hot) for _ in range(seq_len)]
        counts[regimen] += 1

    if split_assignments is not None:
        normalized_assignments = {
            str(patient_id): _normalize_sarcoma_split_label(label)
            for patient_id, label in split_assignments.items()
        }
        patient_ids = {str(patient["patient_id"]) for patient in base["patients"]}
        if set(normalized_assignments) != patient_ids:
            missing_ids = sorted(patient_ids - set(normalized_assignments))[:5]
            unexpected_ids = sorted(set(normalized_assignments) - patient_ids)[:5]
            raise ValueError(
                "Explicit split assignments do not match the two-arm cohort; "
                f"missing={missing_ids}, unexpected={unexpected_ids}"
            )
        for patient in base["patients"]:
            patient["split"] = normalized_assignments[str(patient["patient_id"])]
        split_sizes = {
            label: sum(value == label for value in normalized_assignments.values())
            for label in ("dev", "test")
        }
        if min(split_sizes.values()) <= 0:
            raise ValueError("Explicit splits require non-empty dev and test sets")
    elif split_seed is not None:
        _assign_two_way_splits(
            base["patients"],
            seed=int(split_seed),
            test_fraction=_sarcoma_test_fraction(validation_fraction, test_fraction),
        )

    observed = {
        "n": len(base["patients"]),
        **counts,
        "excluded_postoperative_rt": int(len(postoperative_ids)),
        "excluded_combined_rt": int(len(combined_ids)),
    }
    expectations = {
        "n": "expected_n",
        "surgery_only": "expected_surgery_only",
        "preoperative_rt_surgery": "expected_preoperative_rt_surgery",
        "excluded_postoperative_rt": "expected_excluded_postoperative_rt",
        "excluded_combined_rt": "expected_excluded_combined_rt",
    }
    for key, expected_key in expectations.items():
        expected = contract.get(expected_key)
        if expected is not None and observed[key] != int(expected):
            raise ValueError(
                f"Two-arm cohort contract failed for {key}: "
                f"expected {expected}, observed {observed[key]}"
            )

    return {
        "data_contract": "surgery_rt_preop_two_arm",
        "patients": base["patients"],
        "cohort": observed,
        "seq_len": seq_len,
        "temporal_contract": (
            {
                "contract_version": 2,
                "strategy_assignment": "event_aligned_preoperative_administration",
                "trajectory_resolution": _trajectory_resolution(filtered),
                "censoring": "strict_d_time_indexed_masks",
                "treatment_scope": "preoperative_only",
                "postoperative_policy": "exclude_patient",
                "prediction_landmark": "interval_start",
                "history_lag_intervals": 1,
                "within_interval_ordering": "timestamp_resolved_or_masked",
                "outcome_uses_static_regimen": False,
            }
            if bool(contract.get("require_temporal_order_fields", False))
            else {
                "contract_version": 1,
                "strategy_assignment": "baseline",
                "trajectory_resolution": _trajectory_resolution(filtered),
                "censoring": "strict_d_time_indexed_masks",
                "within_interval_ordering": "not_identified",
            }
        ),
        "split_counts": (
            {
                name: sum(patient.get("split") == name for patient in base["patients"])
                for name in ("dev", "test")
            }
            if split_seed is not None or split_assignments is not None
            else None
        ),
    }


@dataclass
class _GroupEncoder:
    key: str
    spec: Mapping[str, Any]
    categories: dict[str, list[str]]
    numeric_stats: dict[str, tuple[float, float]]

    @classmethod
    def fit(
        cls,
        patients: Sequence[Mapping[str, Any]],
        key: str,
        spec: Mapping[str, Any],
    ) -> "_GroupEncoder":
        categories: dict[str, list[str]] = {}
        numeric_stats: dict[str, tuple[float, float]] = {}
        for column in spec.get("numeric", []) or []:
            values: list[float] = []
            for patient in patients:
                for record in _as_records(patient.get(key)):
                    value = pd.to_numeric(
                        pd.Series([record.get(column)]), errors="coerce"
                    ).iloc[0]
                    if pd.notna(value):
                        values.append(float(value))
            mean = float(np.mean(values)) if values else 0.0
            std = float(np.std(values)) if len(values) > 1 else 1.0
            numeric_stats[str(column)] = (mean, std if std > 0 else 1.0)
        for column in spec.get("categorical", []) or []:
            values: set[str] = set()
            for patient in patients:
                for record in _as_records(patient.get(key)):
                    value = record.get(column)
                    if value is not None and not pd.isna(value):
                        values.add(str(value))
            categories[str(column)] = sorted(values)
        return cls(
            key=key,
            spec=spec,
            categories=categories,
            numeric_stats=numeric_stats,
        )

    @property
    def feature_names(self) -> list[str]:
        names = [
            *[str(value) for value in self.spec.get("numeric", []) or []],
            *[str(value) for value in self.spec.get("binary", []) or []],
            *[str(value) for value in (self.spec.get("ordinal", {}) or {})],
        ]
        for column in self.spec.get("categorical", []) or []:
            names.extend(f"{column}={level}" for level in self.categories[str(column)])
        return names

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def encode_record(self, record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        values: list[float] = []
        masks: list[float] = []
        for column in self.spec.get("numeric", []) or []:
            value = pd.to_numeric(pd.Series([record.get(column)]), errors="coerce").iloc[0]
            present = bool(pd.notna(value))
            mean, std = self.numeric_stats[str(column)]
            values.append((float(value) - mean) / std if present else 0.0)
            masks.append(float(present))
        for column in self.spec.get("binary", []) or []:
            value = pd.to_numeric(pd.Series([record.get(column)]), errors="coerce").iloc[0]
            present = bool(pd.notna(value))
            values.append(float(value) if present else 0.0)
            masks.append(float(present))
        for column, mapping in (self.spec.get("ordinal", {}) or {}).items():
            value = record.get(column)
            normalized = None if value is None or pd.isna(value) else str(value)
            mapped = mapping.get(normalized) if normalized is not None else None
            values.append(float(mapped) if mapped is not None else 0.0)
            masks.append(float(mapped is not None))
        for column in self.spec.get("categorical", []) or []:
            value = record.get(column)
            normalized = None if value is None or pd.isna(value) else str(value)
            levels = self.categories[str(column)]
            present = normalized in levels
            values.extend(float(present and normalized == level) for level in levels)
            masks.extend(float(present) for _ in levels)
        return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=np.float32)

    def encode_sequence(
        self,
        records: Sequence[Mapping[str, Any]],
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((seq_len, self.dim), dtype=np.float32)
        masks = np.zeros_like(values)
        for index, record in enumerate(records[:seq_len]):
            values[index], masks[index] = self.encode_record(record)
        return values, masks


class SarcomaSequenceDataset(Dataset):
    """Dataset for surgery versus preoperative RT plus surgery."""

    def __init__(
        self,
        data: Mapping[str, Any],
        seq_len: int,
        feature_groups: Mapping[str, Any],
        patient_weights: Mapping[str, float] | None = None,
    ) -> None:
        patients = data.get("patients", [])
        if data.get("data_contract") != "surgery_rt_preop_two_arm" or not patients:
            raise ValueError("Expected a non-empty two-arm sarcoma training bundle")
        self.patients = list(patients)
        self.patient_ids = [str(patient["patient_id"]) for patient in self.patients]
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("Training patient identifiers must be unique")
        self.seq_len = int(seq_len)
        self.history_only_outcome = True
        split_values = [patient.get("split") for patient in self.patients]
        has_splits = any(value is not None for value in split_values)
        if has_splits:
            if any(value not in {"dev", "test"} for value in split_values):
                raise ValueError(
                    "Sarcoma patients must all use dev/test split labels"
                )
            encoder_patients = [
                patient for patient in self.patients if patient["split"] == "dev"
            ]
            if not encoder_patients or len(encoder_patients) == len(self.patients):
                raise ValueError("Sarcoma splits require non-empty dev and test sets")
        else:
            encoder_patients = self.patients
        self.baseline_encoder = _GroupEncoder.fit(
            encoder_patients, "baseline", feature_groups.get("baseline", {}) or {}
        )
        self.state_encoder = _GroupEncoder.fit(
            encoder_patients, "clinical_state", feature_groups.get("state", {}) or {}
        )
        self.treatment_encoder = _GroupEncoder.fit(
            encoder_patients, "treatment", feature_groups.get("treatment", {}) or {}
        )
        self.outcome_encoder = _GroupEncoder.fit(
            encoder_patients, "outcome", feature_groups.get("outcome", {}) or {}
        )
        self.baseline_columns = self.baseline_encoder.feature_names
        self.state_columns = self.state_encoder.feature_names
        self.treat_columns = self.treatment_encoder.feature_names
        self.outcome_columns = self.outcome_encoder.feature_names
        self.cond_columns = list(PREOP_TWO_ARM_CONDITION_COLUMNS)
        self.baseline_dim = self.baseline_encoder.dim
        self.state_dim = self.state_encoder.dim
        self.clin_dim = self.baseline_dim + self.state_dim
        self.treat_dim = self.treatment_encoder.dim
        self.outcome_dim = self.outcome_encoder.dim
        self.cond_dim = len(self.cond_columns)
        self.clin_columns = [
            *[f"baseline::{name}" for name in self.baseline_columns],
            *[f"state::{name}" for name in self.state_columns],
        ]
        self.state_feature_indices = list(
            range(self.baseline_dim, self.baseline_dim + self.state_dim)
        )
        # Baseline-only counterfactual inference generates the first state and
        # every subsequent state; no observed longitudinal prefix is required.
        self.forecast_start = 0
        self.spec_clinical = {
            "baseline": dict(feature_groups.get("baseline", {}) or {}),
            "state": dict(feature_groups.get("state", {}) or {}),
        }
        self.spec_treatment = dict(feature_groups.get("treatment", {}) or {})
        self.spec_outcome = dict(feature_groups.get("outcome", {}) or {})

        if patient_weights is None:
            self.patient_weights = np.ones(len(self.patients), dtype=np.float32)
            self.causal_weighting_enabled = False
        else:
            mapping = {str(key): float(value) for key, value in patient_weights.items()}
            missing = sorted(set(self.patient_ids) - set(mapping))[:5]
            unexpected = sorted(set(mapping) - set(self.patient_ids))[:5]
            if missing or unexpected:
                raise ValueError(
                    "Causal weights do not match training patients; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            self.patient_weights = np.asarray(
                [mapping[patient_id] for patient_id in self.patient_ids], dtype=np.float32
            )
            if (
                not np.isfinite(self.patient_weights).all()
                or (self.patient_weights < 0).any()
                or self.patient_weights.sum() <= 0
            ):
                raise ValueError("Causal weights must be finite, non-negative, and nonzero")
            self.causal_weighting_enabled = True

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patient = self.patients[index]
        baseline_values, baseline_mask = self.baseline_encoder.encode_record(
            patient["baseline"]
        )
        state_values, state_mask = self.state_encoder.encode_sequence(
            patient["clinical_state"], self.seq_len
        )
        treatment_values, treatment_mask = self.treatment_encoder.encode_sequence(
            patient["treatment"], self.seq_len
        )
        outcome_values, _ = self.outcome_encoder.encode_sequence(
            patient["outcome"], self.seq_len
        )
        outcome_mask, _ = self.outcome_encoder.encode_sequence(
            patient["outcome_mask"], self.seq_len
        )
        sequence_mask = np.asarray(patient["sequence_mask"], dtype=np.float32)[: self.seq_len]
        sequence_mask = np.pad(
            sequence_mask,
            (0, max(0, self.seq_len - len(sequence_mask))),
        )[: self.seq_len]
        state_mask *= sequence_mask[:, None]
        treatment_mask *= sequence_mask[:, None]
        baseline_sequence = np.repeat(baseline_values[None, :], self.seq_len, axis=0)
        baseline_mask_sequence = np.repeat(baseline_mask[None, :], self.seq_len, axis=0)
        clinical_values = np.concatenate([baseline_sequence, state_values], axis=1)
        clinical_mask = np.concatenate([baseline_mask_sequence, state_mask], axis=1)

        outcome_state_values = state_values.copy()
        outcome_state_mask = state_mask.copy()
        outcome_treatment_values = treatment_values.copy()
        outcome_treatment_mask = treatment_mask.copy()
        if self.history_only_outcome:
            outcome_state_values = np.zeros_like(state_values)
            outcome_state_mask = np.zeros_like(state_mask)
            outcome_treatment_values = np.zeros_like(treatment_values)
            outcome_treatment_mask = np.zeros_like(treatment_mask)
            if self.seq_len > 1:
                outcome_state_values[1:] = state_values[:-1] * state_mask[:-1]
                outcome_state_mask[1:] = state_mask[:-1]
                outcome_treatment_values[1:] = (
                    treatment_values[:-1] * treatment_mask[:-1]
                )
                outcome_treatment_mask[1:] = treatment_mask[:-1]

            # Time itself is known at the interval-start landmark and is not a
            # patient measurement that needs lagging.
            for feature_index, feature_name in enumerate(self.state_columns):
                if feature_name == "interval_scaled":
                    outcome_state_values[:, feature_index] = state_values[
                        :, feature_index
                    ]
                    outcome_state_mask[:, feature_index] = state_mask[
                        :, feature_index
                    ]

        outcome_clinical_values = np.concatenate(
            [baseline_sequence, outcome_state_values], axis=1
        )
        outcome_clinical_mask = np.concatenate(
            [baseline_mask_sequence, outcome_state_mask], axis=1
        )
        assignment = np.asarray(
            [
                [float(record[column]) for column in self.cond_columns]
                for record in patient["actual_treatment"][: self.seq_len]
            ],
            dtype=np.float32,
        )
        if assignment.shape != (self.seq_len, self.cond_dim):
            raise ValueError("Treatment assignment does not match sequence dimensions")
        assignment_mask = np.repeat(sequence_mask[:, None], self.cond_dim, axis=1)
        return {
            "x_baseline": torch.from_numpy(baseline_values),
            "mask_baseline": torch.from_numpy(baseline_mask),
            "x_state": torch.from_numpy(state_values),
            "mask_state": torch.from_numpy(state_mask),
            "state_target": torch.from_numpy(state_values),
            "state_target_mask": torch.from_numpy(state_mask),
            "x_clin": torch.from_numpy(clinical_values),
            "mask_clin": torch.from_numpy(clinical_mask),
            "x_treat": torch.from_numpy(treatment_values),
            "mask_treat": torch.from_numpy(treatment_mask),
            "x_outcome_clin": torch.from_numpy(outcome_clinical_values),
            "mask_outcome_clin": torch.from_numpy(outcome_clinical_mask),
            "x_outcome_treat": torch.from_numpy(outcome_treatment_values),
            "mask_outcome_treat": torch.from_numpy(outcome_treatment_mask),
            "actual_treatment": torch.from_numpy(assignment),
            "mask_actual": torch.from_numpy(assignment_mask),
            "outcome": torch.from_numpy(outcome_values),
            "outcome_mask": torch.from_numpy(outcome_mask),
            "sequence_mask": torch.from_numpy(sequence_mask[:, None]),
            "causal_weight": torch.tensor(
                self.patient_weights[index], dtype=torch.float32
            ),
        }


__all__ = [
    "FORBIDDEN_PREDICTORS",
    "PREOP_TWO_ARM_REGIMENS",
    "PREOP_TWO_ARM_CONDITION_COLUMNS",
    "SarcomaSequenceDataset",
    "build_surgery_rt_preop_two_arm_training_data",
]
