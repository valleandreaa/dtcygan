"""Reconstruct the MIMIC-III cohort described in the T4 paper.

This pipeline deliberately keeps the paper-cohort artifacts separate from the
broader ``mimic3.pkl`` dataset.  It uses the official MIT-LCP hourly SOFA
concept, then applies the cohort rules stated by Liu et al.:

* adult ICU stays lasting 9 hours to 20 days;
* suspected infection from one of the paper's 18 antibiotics paired with a
  culture (culture up to 72 hours before or 24 hours after antibiotics);
* a SOFA increase of at least two points within six hours; and
* the SOFA event occurring from 24 hours before through 12 hours after the
  suspected-infection time.

The upstream paper does not publish cohort identifiers or preprocessing code.
Consequently this is a versioned, auditable reconstruction, not a claim that
the unpublished author cohort can be recovered byte-for-byte.

The command is resumable by stage::

    python preprocessing/build_t4_paper_cohort.py prepare-db ...
    python preprocessing/build_t4_paper_cohort.py build-concepts ...
    python preprocessing/build_t4_paper_cohort.py select-cohort ...
    python preprocessing/build_t4_paper_cohort.py build-pickles ...
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import mimic_data as base


MIMIC_CODE_COMMIT = "303d26c623dcc9c49cc0f204468d4acc2f063797"
PAPER_PATIENT_COUNT = 10_840
PAPER_TIMESTEPS = 16
PAPER_HOURS_PER_TIMESTEP = 3.0

PAPER_ANTIBIOTICS = (
    "cefazolin",
    "cefepime",
    "ceftazidime",
    "ciprofloxacin",
    "clindamycin",
    "erythromycin",
    "gentamicin",
    "levofloxacin",
    "metronidazole",
    "moxifloxacin",
    "piperacillin",
    "rifampin",
    "tobramycin",
    "vancomycin",
    "amikacin",
    "ampicillin",
    "azithromycin",
    "aztreonam",
)
ANTIBIOTIC_PATTERN = "|".join(re.escape(name) for name in PAPER_ANTIBIOTICS)

# Only tables used by the official concepts and the final feature extraction
# are imported.  Explicit names prevent repaired backups such as
# CHARTEVENTS.corrupt-oversized.csv.gz from being loaded accidentally.
RAW_TABLES = (
    "ADMISSIONS",
    "CHARTEVENTS",
    "ICUSTAYS",
    "INPUTEVENTS_CV",
    "INPUTEVENTS_MV",
    "LABEVENTS",
    "MICROBIOLOGYEVENTS",
    "NOTEEVENTS",
    "OUTPUTEVENTS",
    "PATIENTS",
    "PRESCRIPTIONS",
    "PROCEDUREEVENTS_MV",
)

CONCEPT_FILES = (
    "echo_data.sql",
    "durations/weight_durations.sql",
    "demographics/icustay_times.sql",
    "demographics/icustay_hours.sql",
    "durations/dobutamine_dose.sql",
    "durations/dopamine_dose.sql",
    "durations/epinephrine_dose.sql",
    "durations/norepinephrine_dose.sql",
    "durations/ventilation_classification.sql",
    "durations/ventilation_durations.sql",
    "pivot/pivoted_bg.sql",
    "pivot/pivoted_gcs.sql",
    "pivot/pivoted_lab.sql",
    "pivot/pivoted_uo.sql",
    "pivot/pivoted_bg_art.sql",
    "pivot/pivoted_sofa.sql",
    "treatment/abx_prescriptions_list.sql",
    "treatment/suspicion_of_infection.sql",
)


def _duckdb_module():
    try:
        import duckdb  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "DuckDB is required for the official hourly-SOFA pipeline. Install "
            "it in the active environment with: python -m pip install duckdb"
        ) from error
    return duckdb


def _connect(database: Path):
    duckdb = _duckdb_module()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    connection.execute("PRAGMA enable_progress_bar")
    connection.execute("CREATE SCHEMA IF NOT EXISTS mimiciii")
    connection.execute("CREATE SCHEMA IF NOT EXISTS mimiciii_derived")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS main.t4_build_state (
            stage VARCHAR PRIMARY KEY,
            completed_at TIMESTAMP,
            detail VARCHAR
        )
        """
    )
    return connection


def _stage_complete(connection: Any, stage: str) -> bool:
    count = connection.execute(
        "SELECT COUNT(*) FROM main.t4_build_state WHERE stage = ?", [stage]
    ).fetchone()[0]
    return bool(count)


def _mark_stage(connection: Any, stage: str, detail: dict[str, Any]) -> None:
    connection.execute("DELETE FROM main.t4_build_state WHERE stage = ?", [stage])
    connection.execute(
        "INSERT INTO main.t4_build_state VALUES (?, CURRENT_TIMESTAMP, ?)",
        [stage, json.dumps(detail, sort_keys=True)],
    )


def _canonical_table_path(input_dir: Path, table: str) -> Path:
    return base._table_path(input_dir, f"{table}.csv")


def _sql_literal_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _checkout_commit(mimic_code_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(mimic_code_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_mimic_code(mimic_code_dir: Path) -> tuple[Path, str | None]:
    concepts = mimic_code_dir / "mimic-iii" / "concepts_duckdb"
    ddl = (
        mimic_code_dir
        / "mimic-iii"
        / "buildmimic"
        / "duckdb"
        / "duckdb_add_tables.sql"
    )
    missing = [str(ddl)] if not ddl.is_file() else []
    missing.extend(str(concepts / name) for name in CONCEPT_FILES if not (concepts / name).is_file())
    if missing:
        raise FileNotFoundError("Missing MIT-LCP files: " + ", ".join(missing))
    commit = _checkout_commit(mimic_code_dir)
    if commit is not None and commit != MIMIC_CODE_COMMIT:
        raise RuntimeError(
            f"mimic-code checkout is {commit}, expected pinned commit "
            f"{MIMIC_CODE_COMMIT}. Check out the pinned commit before running."
        )
    return concepts, commit


def prepare_database(input_dir: Path, database: Path, mimic_code_dir: Path) -> None:
    """Load canonical raw MIMIC tables into a resumable DuckDB database."""

    _, commit = validate_mimic_code(mimic_code_dir)
    connection = _connect(database)
    try:
        ddl_stage = "raw_schema"
        if not _stage_complete(connection, ddl_stage):
            ddl_path = (
                mimic_code_dir
                / "mimic-iii"
                / "buildmimic"
                / "duckdb"
                / "duckdb_add_tables.sql"
            )
            connection.execute("USE mimiciii")
            connection.execute(ddl_path.read_text())
            connection.execute("USE main")
            _mark_stage(connection, ddl_stage, {"mimic_code_commit": commit})

        for table in RAW_TABLES:
            stage = f"raw:{table.lower()}"
            if _stage_complete(connection, stage):
                print(f"[skip] {table} already imported", flush=True)
                continue
            source = _canonical_table_path(input_dir, table)
            print(f"[load] {table} <- {source}", flush=True)
            connection.execute(f"DELETE FROM mimiciii.{table.lower()}")
            connection.execute(
                f"COPY mimiciii.{table.lower()} FROM '{_sql_literal_path(source)}' "
                "(HEADER, DELIM ',', QUOTE '\"', ESCAPE '\"')"
            )
            rows = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM mimiciii.{table.lower()}"
                ).fetchone()[0]
            )
            _mark_stage(
                connection,
                stage,
                {"path": str(source.resolve()), "bytes": source.stat().st_size, "rows": rows},
            )
            print(f"[done] {table}: {rows:,} rows", flush=True)
    finally:
        connection.close()


def build_concepts(database: Path, mimic_code_dir: Path) -> None:
    """Run the minimal official MIT-LCP dependency graph for hourly SOFA."""

    concepts, commit = validate_mimic_code(mimic_code_dir)
    connection = _connect(database)
    try:
        missing_raw = [
            table
            for table in RAW_TABLES
            if not _stage_complete(connection, f"raw:{table.lower()}")
        ]
        if missing_raw:
            raise RuntimeError(
                "Raw import is incomplete; missing stages: " + ", ".join(missing_raw)
            )
        for relative_name in CONCEPT_FILES:
            stage = f"concept:{relative_name}"
            if _stage_complete(connection, stage):
                print(f"[skip] {relative_name}", flush=True)
                continue
            sql_path = concepts / relative_name
            print(f"[concept] {relative_name}", flush=True)
            connection.execute(sql_path.read_text())
            table_name = sql_path.stem
            rows = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM mimiciii_derived.{table_name}"
                ).fetchone()[0]
            )
            _mark_stage(
                connection,
                stage,
                {"mimic_code_commit": commit, "rows": rows},
            )
            print(f"[done] {relative_name}: {rows:,} rows", flush=True)
    finally:
        connection.close()


def _cohort_selection_sql() -> str:
    """Return the paper-rule cohort query applied to official concepts."""

    return f"""
    CREATE OR REPLACE TABLE main.t4_paper_cohort_candidates AS
    WITH ranked_stays AS (
        SELECT
            ie.subject_id,
            ie.hadm_id,
            ie.icustay_id,
            ie.intime,
            ie.outtime,
            CASE
                WHEN DATE_DIFF('year', p.dob, ie.intime) > 89 THEN 90.0
                ELSE DATE_DIFF('second', p.dob, ie.intime) / (365.2425 * 86400.0)
            END AS age,
            p.gender,
            a.hospital_expire_flag,
            ROW_NUMBER() OVER (
                PARTITION BY ie.subject_id
                ORDER BY ie.intime, ie.icustay_id
            ) AS subject_icu_rank
        FROM mimiciii.icustays ie
        INNER JOIN mimiciii.patients p USING (subject_id)
        INNER JOIN mimiciii.admissions a USING (subject_id, hadm_id)
    ), eligible_stays AS (
        SELECT * EXCLUDE (subject_icu_rank)
        FROM ranked_stays
        WHERE subject_icu_rank = 1
          AND DATE_DIFF('second', intime, outtime) / 3600.0
              BETWEEN 9.0 AND 480.0
    ), adults AS (
        SELECT * FROM eligible_stays WHERE age >= 18.0
    ), paper_suspicion AS (
        SELECT
            soi.icustay_id,
            soi.antibiotic_name,
            soi.antibiotic_time,
            soi.suspected_infection_time AS culture_time,
            LEAST(soi.antibiotic_time, soi.suspected_infection_time) AS t_susp
        FROM mimiciii_derived.suspicion_of_infection soi
        WHERE REGEXP_MATCHES(LOWER(soi.antibiotic_name), '{ANTIBIOTIC_PATTERN}')
          AND soi.suspected_infection_time IS NOT NULL
          AND soi.antibiotic_time IS NOT NULL
    ), sofa_with_baseline AS (
        SELECT
            icustay_id,
            hr,
            endtime AS t_sofa,
            sofa_24hours,
            MIN(sofa_24hours) OVER (
                PARTITION BY icustay_id ORDER BY hr
                ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
            ) AS sofa_baseline_6h
        FROM mimiciii_derived.pivoted_sofa
    ), deterioration AS (
        SELECT *, sofa_24hours - sofa_baseline_6h AS sofa_delta_6h
        FROM sofa_with_baseline
        WHERE sofa_baseline_6h IS NOT NULL
          AND sofa_24hours - sofa_baseline_6h >= 2
    ), matched AS (
        SELECT
            a.*,
            s.antibiotic_name,
            s.antibiotic_time,
            s.culture_time,
            s.t_susp,
            d.t_sofa,
            d.sofa_baseline_6h,
            d.sofa_24hours,
            d.sofa_delta_6h,
            ROW_NUMBER() OVER (
                PARTITION BY a.icustay_id
                ORDER BY s.t_susp, d.t_sofa, s.antibiotic_time
            ) AS stay_event_rank
        FROM adults a
        INNER JOIN paper_suspicion s USING (icustay_id)
        INNER JOIN deterioration d USING (icustay_id)
        WHERE d.t_sofa BETWEEN s.t_susp - INTERVAL '24' HOUR
                           AND s.t_susp + INTERVAL '12' HOUR
    ), one_event_per_stay AS (
        SELECT * EXCLUDE (stay_event_rank)
        FROM matched WHERE stay_event_rank = 1
    ), one_stay_per_patient AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY subject_id ORDER BY t_susp, intime, icustay_id
        ) AS subject_stay_rank
        FROM one_event_per_stay
    )
    SELECT * EXCLUDE (subject_stay_rank)
    FROM one_stay_per_patient
    WHERE subject_stay_rank = 1
    ORDER BY subject_id
    """


def _funnel_counts(connection: Any) -> dict[str, int]:
    pattern = ANTIBIOTIC_PATTERN.replace("'", "''")
    queries = {
        "all_icu_stays": "SELECT COUNT(*) FROM mimiciii.icustays",
        "first_icu_stays": "SELECT COUNT(DISTINCT subject_id) FROM mimiciii.icustays",
        "first_icu_duration_9h_to_20d": """
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY subject_id ORDER BY intime, icustay_id
                ) AS subject_icu_rank
                FROM mimiciii.icustays
            )
            SELECT COUNT(*) FROM ranked
            WHERE subject_icu_rank = 1
              AND DATE_DIFF('second', intime, outtime) / 3600.0 BETWEEN 9 AND 480
        """,
        "first_icu_adult_duration_eligible": """
            WITH ranked AS (
                SELECT ie.*, ROW_NUMBER() OVER (
                    PARTITION BY ie.subject_id ORDER BY ie.intime, ie.icustay_id
                ) AS subject_icu_rank
                FROM mimiciii.icustays ie
            )
            SELECT COUNT(*)
            FROM ranked ie JOIN mimiciii.patients p USING (subject_id)
            WHERE ie.subject_icu_rank = 1
              AND DATE_DIFF('second', ie.intime, ie.outtime) / 3600.0 BETWEEN 9 AND 480
              AND CASE WHEN DATE_DIFF('year', p.dob, ie.intime) > 89 THEN 90.0
                       ELSE DATE_DIFF('second', p.dob, ie.intime) / (365.2425 * 86400.0)
                  END >= 18
        """,
        "paper_antibiotic_culture_pairs": f"""
            SELECT COUNT(*) FROM mimiciii_derived.suspicion_of_infection
            WHERE REGEXP_MATCHES(LOWER(antibiotic_name), '{pattern}')
              AND suspected_infection_time IS NOT NULL
        """,
        "selected_unique_patients": "SELECT COUNT(*) FROM main.t4_paper_cohort_candidates",
    }
    return {
        name: int(connection.execute(query).fetchone()[0])
        for name, query in queries.items()
    }


def select_cohort(
    database: Path,
    output_dir: Path,
    count_tolerance_fraction: float,
) -> dict[str, Any]:
    """Select the paper-rule cohort, write its manifest, and enforce the gate."""

    if not 0.0 <= count_tolerance_fraction <= 1.0:
        raise ValueError("count_tolerance_fraction must be between 0 and 1")
    connection = _connect(database)
    try:
        if not _stage_complete(connection, "concept:pivot/pivoted_sofa.sql"):
            raise RuntimeError("Official hourly SOFA concept has not been built")
        if not _stage_complete(
            connection, "concept:treatment/suspicion_of_infection.sql"
        ):
            raise RuntimeError("Suspicion-of-infection concept has not been built")
        connection.execute(_cohort_selection_sql())
        funnel = _funnel_counts(connection)
        actual = funnel["selected_unique_patients"]
        relative_difference = abs(actual - PAPER_PATIENT_COUNT) / PAPER_PATIENT_COUNT
        within_tolerance = relative_difference <= count_tolerance_fraction
        cohort_csv = output_dir / "t4_paper_cohort_rows.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"COPY main.t4_paper_cohort_candidates TO '{_sql_literal_path(cohort_csv)}' "
            "(HEADER, DELIMITER ',')"
        )
        manifest = {
            "status": "cohort_gate_passed" if within_tolerance else "cohort_count_mismatch",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_reference_count": PAPER_PATIENT_COUNT,
            "reconstructed_count": actual,
            "relative_count_difference": relative_difference,
            "count_tolerance_fraction": count_tolerance_fraction,
            "within_count_tolerance": within_tolerance,
            "funnel": funnel,
            "mimic_code_commit": MIMIC_CODE_COMMIT,
            "paper_rules": {
                "minimum_age_years": 18,
                "minimum_icu_hours": 9,
                "maximum_icu_days": 20,
                "sofa_increase_points": 2,
                "sofa_deterioration_window_hours": 6,
                "sofa_relative_to_suspicion_hours": [-24, 12],
                "culture_before_antibiotic_hours": 72,
                "culture_after_antibiotic_hours": 24,
                "antibiotics": list(PAPER_ANTIBIOTICS),
            },
            "sequence_contract": {
                "anchor": "ICU admission",
                "timesteps": PAPER_TIMESTEPS,
                "hours_per_timestep": PAPER_HOURS_PER_TIMESTEP,
                "observation_hours": 48,
            },
            "identity_caveat": (
                "The authors did not publish cohort IDs or preprocessing code; "
                "matching the published count does not prove patient-level identity."
            ),
            "cohort_rows": str(cohort_csv.resolve()),
        }
        manifest_path = output_dir / "t4_paper_cohort_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _mark_stage(connection, "paper_cohort", manifest)
    finally:
        connection.close()

    if not manifest["within_count_tolerance"]:
        raise RuntimeError(
            f"Cohort gate failed: reconstructed {actual:,} patients; paper reports "
            f"{PAPER_PATIENT_COUNT:,}. Inspect {manifest_path} before building pickles."
        )
    print(
        f"Cohort gate passed: {actual:,} reconstructed patients "
        f"(paper reference {PAPER_PATIENT_COUNT:,})"
    )
    return manifest


def _load_cohort_rows(path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(path)
    cohort.columns = [str(column).casefold() for column in cohort.columns]
    date_columns = (
        "intime",
        "outtime",
        "antibiotic_time",
        "culture_time",
        "t_susp",
        "t_sofa",
    )
    missing_dates = set(date_columns).difference(cohort.columns)
    if missing_dates:
        raise ValueError(
            "Cohort CSV is missing date columns: "
            + ", ".join(sorted(missing_dates))
        )
    for column in date_columns:
        cohort[column] = pd.to_datetime(cohort[column], errors="raise")
    required = {
        "subject_id",
        "hadm_id",
        "icustay_id",
        "intime",
        "age",
        "gender",
        "hospital_expire_flag",
    }
    missing = required.difference(cohort.columns)
    if missing:
        raise ValueError("Cohort CSV is missing columns: " + ", ".join(sorted(missing)))
    if cohort["subject_id"].duplicated().any():
        raise ValueError("Cohort CSV must contain one ICU stay per subject")
    cohort = cohort.sort_values("subject_id").reset_index(drop=True)
    cohort["patient_index"] = np.arange(len(cohort), dtype=int)
    return cohort


def _feature_summary(raw_temporal: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean(raw_temporal[:, :, index])),
            "std": float(np.std(raw_temporal[:, :, index])),
        }
        for index, name in enumerate(base.TEMPORAL_FEATURES)
    }


def build_data_for_cohort(
    input_dir: Path,
    cohort: pd.DataFrame,
    chunksize: int,
) -> tuple[dict[int, dict[str, object]], list[str], dict[str, Any]]:
    """Extract the T4 16-by-22 feature contract for selected ICU stays."""

    n_patients = len(cohort)
    n_features = len(base.TEMPORAL_FEATURES)
    sums = np.zeros((n_patients, PAPER_TIMESTEPS, n_features), dtype=float)
    counts = np.zeros((n_patients, PAPER_TIMESTEPS, n_features), dtype=np.int64)
    static_sums = np.zeros((n_patients, 2), dtype=float)
    static_counts = np.zeros((n_patients, 2), dtype=np.int64)
    intime_by_patient = cohort["intime"].to_numpy()
    admission_to_patient = dict(
        zip(cohort["hadm_id"].astype(int), cohort["patient_index"].astype(int))
    )
    stay_to_patient = dict(
        zip(cohort["icustay_id"].astype(int), cohort["patient_index"].astype(int))
    )

    lab_ids = frozenset(base.LAB_ITEM_TO_FEATURE)
    for chunk in base._chunks(
        base._table_path(input_dir, "LABEVENTS.csv"),
        ("hadm_id", "itemid", "charttime", "valuenum"),
        chunksize,
    ):
        chunk = chunk[chunk["itemid"].isin(lab_ids)]
        base._accumulate_temporal(
            chunk,
            id_column="hadm_id",
            id_to_patient=admission_to_patient,
            intime_by_patient=intime_by_patient,
            item_to_feature=base.LAB_ITEM_TO_FEATURE,
            sums=sums,
            counts=counts,
            timesteps=PAPER_TIMESTEPS,
            hours_per_timestep=PAPER_HOURS_PER_TIMESTEP,
        )

    chart_ids = frozenset(base.CHART_ITEM_TO_FEATURE) | (
        base.WEIGHT_KG_ITEM_IDS
        | base.WEIGHT_LB_ITEM_IDS
        | base.WEIGHT_OZ_ITEM_IDS
        | base.HEIGHT_IN_ITEM_IDS
        | base.HEIGHT_CM_ITEM_IDS
    )
    for chunk in base._chunks(
        base._table_path(input_dir, "CHARTEVENTS.csv"),
        ("icustay_id", "itemid", "charttime", "valuenum"),
        chunksize,
    ):
        chunk = chunk[chunk["itemid"].isin(chart_ids)]
        base._accumulate_temporal(
            chunk[chunk["itemid"].isin(base.CHART_ITEM_TO_FEATURE)],
            id_column="icustay_id",
            id_to_patient=stay_to_patient,
            intime_by_patient=intime_by_patient,
            item_to_feature=base.CHART_ITEM_TO_FEATURE,
            sums=sums,
            counts=counts,
            timesteps=PAPER_TIMESTEPS,
            hours_per_timestep=PAPER_HOURS_PER_TIMESTEP,
            convert_temperatures=True,
        )
        base._accumulate_static(
            chunk,
            stay_to_patient=stay_to_patient,
            intime_by_patient=intime_by_patient,
            static_sums=static_sums,
            static_counts=static_counts,
            observation_hours=48.0,
        )

    temporal = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=temporal, where=counts > 0)
    temporal = base._forward_fill_then_mean(temporal)
    raw_summary = _feature_summary(temporal)
    temporal = base._standardize_temporal(temporal)

    static = np.full_like(static_sums, np.nan)
    np.divide(static_sums, static_counts, out=static, where=static_counts > 0)
    static_labels = ("weight_kg", "height_cm")
    static_summary: dict[str, dict[str, float]] = {}
    for column, label in enumerate(static_labels):
        observed = static[np.isfinite(static[:, column]), column]
        fallback = 80.0 if column == 0 else 170.0
        median = float(np.median(observed)) if observed.size else fallback
        static[~np.isfinite(static[:, column]), column] = median
        static_summary[label] = {
            "mean": float(np.mean(static[:, column])),
            "std": float(np.std(static[:, column])),
            "imputation_median": median,
            "observed_fraction": float(observed.size / max(n_patients, 1)),
        }

    result: dict[int, dict[str, object]] = {}
    for row in cohort.itertuples(index=False):
        index = int(row.patient_index)
        weight, height = static[index]
        datum: dict[str, object] = {
            name: temporal[index, :, feature_index].tolist()
            for feature_index, name in enumerate(base.TEMPORAL_FEATURES)
        }
        datum.update(
            {
                "agegroup": base._representative_group(
                    float(row.age),
                    [40, 50, 60, 70, 80],
                    [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
                ),
                "heightgroup": base._representative_group(
                    float(height),
                    [160, 170, 180, 190],
                    [-1.0, -0.5, 0.0, 0.5, 1.0],
                ),
                "weightgroup": base._representative_group(
                    float(weight),
                    [60, 70, 80, 90, 100, 110],
                    [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                ),
                "gender": float(str(row.gender).upper() == "M"),
                "death": int(row.hospital_expire_flag),
            }
        )
        result[int(row.subject_id)] = datum

    summary = {
        "age_years": {
            "mean": float(cohort["age"].mean()),
            "std": float(cohort["age"].std(ddof=0)),
        },
        "female_fraction": float((cohort["gender"].str.upper() == "F").mean()),
        "temporal_before_standardization": raw_summary,
        "static": static_summary,
    }
    return result, list(base.TEMPORAL_FEATURES), summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_paper_pickles(
    input_dir: Path,
    output_dir: Path,
    chunksize: int,
) -> tuple[Path, Path]:
    manifest_path = output_dir / "t4_paper_cohort_manifest.json"
    cohort_path = output_dir / "t4_paper_cohort_rows.csv"
    if not manifest_path.is_file() or not cohort_path.is_file():
        raise FileNotFoundError("Run select-cohort before build-pickles")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("within_count_tolerance"):
        raise RuntimeError("Cohort count gate has not passed")
    cohort = _load_cohort_rows(cohort_path)
    if len(cohort) != int(manifest["reconstructed_count"]):
        raise ValueError("Cohort CSV count does not match its manifest")
    data, header, summary = build_data_for_cohort(
        input_dir, cohort, chunksize=chunksize
    )
    data_path = output_dir / "t4_paper_mimic3.pkl"
    header_path = output_dir / "t4_paper_header.pkl"
    with data_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with header_path.open("wb") as handle:
        pickle.dump(header, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest["status"] = "paper_pickles_built"
    manifest["feature_summary"] = summary
    manifest["outputs"] = {
        "data": str(data_path.resolve()),
        "header": str(header_path.resolve()),
        "data_sha256": _sha256(data_path),
        "header_sha256": _sha256(header_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(data):,} patients, {len(header)} features, and "
        f"{PAPER_TIMESTEPS} timestamps to {data_path}"
    )
    return data_path, header_path


