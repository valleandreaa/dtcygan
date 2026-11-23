"""Build the MIMIC-III pickle inputs expected by T4's semi-synthetic DGP.

The upstream T4 repository publishes the data contract but not the raw
MIMIC-III preprocessing code.  This module constructs that contract from the
official MIMIC-III CSV files.  It keeps the first adult ICU stay with enough
observed time, averages measurements inside fixed-width bins, and carries the
last measurement forward.  Leading missing values are filled with the cohort
mean for that feature. Temporal covariates are then standardized feature-wise;
ordered demographic groups are encoded on [-1, 1]. Keeping the DGP inputs on
a comparable scale prevents T4's logistic treatment mechanism from saturating.

For comparison with the eight-period sarcoma dataset, the default is eight
six-hour bins over the first 48 hours.  T4's paper used three-hour bins; use
``--timesteps 16 --hours-per-timestep 3`` to reproduce that resolution.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEMPORAL_FEATURES = (
    "anion_gap",
    "bicarbonate",
    "bilirubin",
    "creatinine",
    "chloride",
    "glucose",
    "hematocrit",
    "hemoglobin",
    "lactate",
    "platelet",
    "potassium",
    "pt",
    "sodium",
    "bun",
    "wbc",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "respiratory_rate",
    "temperature",
    "spo2",
)

# MIMIC-III D_LABITEMS identifiers. Blood-gas variants are included where
# their units match the corresponding chemistry/haematology measurement.
LAB_ITEM_TO_FEATURE = {
    50868: "anion_gap",
    50882: "bicarbonate",
    50885: "bilirubin",
    50912: "creatinine",
    50902: "chloride",
    50809: "glucose",
    50931: "glucose",
    50810: "hematocrit",
    51221: "hematocrit",
    50811: "hemoglobin",
    51222: "hemoglobin",
    50813: "lactate",
    51265: "platelet",
    50822: "potassium",
    50971: "potassium",
    51274: "pt",
    50824: "sodium",
    50983: "sodium",
    51006: "bun",
    51300: "wbc",
    51301: "wbc",
}

# MIMIC-III D_ITEMS identifiers used by the MIT-LCP/ricu concepts. Invasive,
# non-invasive, and manual blood-pressure measurements are pooled.
CHART_ITEM_TO_FEATURE = {
    211: "heart_rate",
    220045: "heart_rate",
    51: "systolic_bp",
    442: "systolic_bp",
    455: "systolic_bp",
    6701: "systolic_bp",
    220050: "systolic_bp",
    220179: "systolic_bp",
    224167: "systolic_bp",
    227243: "systolic_bp",
    8368: "diastolic_bp",
    8440: "diastolic_bp",
    8441: "diastolic_bp",
    8555: "diastolic_bp",
    220051: "diastolic_bp",
    220180: "diastolic_bp",
    224643: "diastolic_bp",
    227242: "diastolic_bp",
    52: "mean_bp",
    443: "mean_bp",
    456: "mean_bp",
    6072: "mean_bp",
    6702: "mean_bp",
    220052: "mean_bp",
    220181: "mean_bp",
    225312: "mean_bp",
    615: "respiratory_rate",
    618: "respiratory_rate",
    220210: "respiratory_rate",
    224688: "respiratory_rate",
    224689: "respiratory_rate",
    224690: "respiratory_rate",
    676: "temperature",
    677: "temperature",
    678: "temperature",
    679: "temperature",
    223761: "temperature",
    223762: "temperature",
    646: "spo2",
    220277: "spo2",
    226253: "spo2",
}

TEMPERATURE_F_ITEM_IDS = frozenset((678, 679, 223761))

WEIGHT_KG_ITEM_IDS = frozenset((762, 763, 3580, 3693, 224639, 226512))
WEIGHT_LB_ITEM_IDS = frozenset((3581, 226531))
WEIGHT_OZ_ITEM_IDS = frozenset((3582,))
HEIGHT_IN_ITEM_IDS = frozenset((920, 1394, 3485, 3486, 4187, 4188, 226707))
HEIGHT_CM_ITEM_IDS = frozenset((226730,))

# Broad physiological bounds remove unit errors and impossible chart entries.
VALID_RANGES = {
    "anion_gap": (0.0, 100.0),
    "bicarbonate": (0.0, 80.0),
    "bilirubin": (0.0, 100.0),
    "creatinine": (0.0, 50.0),
    "chloride": (50.0, 200.0),
    "glucose": (1.0, 2000.0),
    "hematocrit": (1.0, 80.0),
    "hemoglobin": (1.0, 30.0),
    "lactate": (0.0, 50.0),
    "platelet": (1.0, 2000.0),
    "potassium": (1.0, 15.0),
    "pt": (5.0, 150.0),
    "sodium": (80.0, 220.0),
    "bun": (0.0, 300.0),
    "wbc": (0.01, 500.0),
    "heart_rate": (1.0, 300.0),
    "systolic_bp": (1.0, 300.0),
    "diastolic_bp": (1.0, 200.0),
    "mean_bp": (1.0, 250.0),
    "respiratory_rate": (1.0, 120.0),
    "temperature": (25.0, 45.0),
    "spo2": (1.0, 100.0),
}

REQUIRED_FILES = (
    "PATIENTS.csv",
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "LABEVENTS.csv",
    "CHARTEVENTS.csv",
)


def _table_path(input_dir: Path, filename: str) -> Path:
    """Return an uncompressed or gzip-compressed MIMIC-III table path."""

    candidates = (input_dir / filename, input_dir / f"{filename}.gz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing required MIMIC-III table in {input_dir}: "
        f"{filename} or {filename}.gz"
    )


def _require_files(input_dir: Path) -> None:
    missing = []
    for filename in REQUIRED_FILES:
        try:
            _table_path(input_dir, filename)
        except FileNotFoundError:
            missing.append(f"{filename}[.gz]")
    if missing:
        raise FileNotFoundError(
            f"Missing required MIMIC-III files in {input_dir}: {', '.join(missing)}"
        )


def _normalize_columns(frame: pd.DataFrame, usecols: Iterable[str], path: Path) -> pd.DataFrame:
    frame.columns = [str(column).casefold() for column in frame.columns]
    expected = {column.casefold() for column in usecols}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    return frame


def _read_table(input_dir: Path, filename: str, usecols: Iterable[str]) -> pd.DataFrame:
    """Read selected columns regardless of PhysioNet header capitalization."""

    path = _table_path(input_dir, filename)
    requested = tuple(usecols)
    wanted = {column.casefold() for column in requested}
    frame = pd.read_csv(
        path,
        usecols=lambda column: str(column).casefold() in wanted,
    )
    return _normalize_columns(frame, requested, path)


def _load_cohort(
    input_dir: Path,
    timesteps: int,
    hours_per_timestep: float,
) -> pd.DataFrame:
    patients = _read_table(
        input_dir,
        "PATIENTS.csv",
        ("subject_id", "gender", "dob"),
    )
    admissions = _read_table(
        input_dir,
        "ADMISSIONS.csv",
        ("subject_id", "hadm_id", "hospital_expire_flag"),
    )
    stays = _read_table(
        input_dir,
        "ICUSTAYS.csv",
        ("subject_id", "hadm_id", "icustay_id", "intime", "outtime"),
    )
    patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")

    stays = stays.sort_values(["subject_id", "intime", "icustay_id"])
    stays = stays.drop_duplicates("subject_id", keep="first")
    stays = stays.merge(patients, on="subject_id", how="inner", validate="one_to_one")
    stays = stays.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="many_to_one",
    )
    stays["age"] = (stays["intime"] - stays["dob"]).dt.total_seconds() / (
        365.2425 * 24 * 3600
    )
    # MIMIC-III shifts dates for patients older than 89, producing ages near
    # 300. Conventionally these records are represented as age 90.
    stays.loc[stays["age"] > 89.0, "age"] = 90.0
    required_hours = float(timesteps) * float(hours_per_timestep)
    duration_hours = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600
    stays = stays[(stays["age"] >= 18.0) & (duration_hours >= required_hours)].copy()
    stays = stays.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    if stays.empty:
        raise ValueError(
            "No adult first ICU stays are long enough for "
            f"{timesteps} x {hours_per_timestep:g}-hour bins"
        )
    stays["patient_index"] = np.arange(len(stays), dtype=int)
    return stays


def _chunks(path: Path, usecols: Iterable[str], chunksize: int) -> Iterable[pd.DataFrame]:
    requested = tuple(usecols)
    wanted = {column.casefold() for column in requested}
    reader = pd.read_csv(
        path,
        usecols=lambda column: str(column).casefold() in wanted,
        chunksize=chunksize,
    )
    for frame in reader:
        yield _normalize_columns(frame, requested, path)


def _accumulate_temporal(
    frame: pd.DataFrame,
    id_column: str,
    id_to_patient: dict[int, int],
    intime_by_patient: np.ndarray,
    item_to_feature: dict[int, str],
    sums: np.ndarray,
    counts: np.ndarray,
    timesteps: int,
    hours_per_timestep: float,
    convert_temperatures: bool = False,
) -> None:
    frame["patient_index"] = frame[id_column].map(id_to_patient)
    frame["feature"] = frame["itemid"].map(item_to_feature)
    frame["valuenum"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    frame["charttime"] = pd.to_datetime(frame["charttime"], errors="coerce")
    frame = frame.dropna(
        subset=["patient_index", "feature", "valuenum", "charttime"]
    ).copy()
    if frame.empty:
        return

    if convert_temperatures:
        fahrenheit = frame["itemid"].isin(TEMPERATURE_F_ITEM_IDS)
        frame.loc[fahrenheit, "valuenum"] = (
            frame.loc[fahrenheit, "valuenum"] - 32.0
        ) * (5.0 / 9.0)

    patient_indices = frame["patient_index"].astype(int).to_numpy()
    intimes = pd.to_datetime(intime_by_patient[patient_indices])
    elapsed_hours = (
        frame["charttime"].reset_index(drop=True) - pd.Series(intimes)
    ).dt.total_seconds().to_numpy() / 3600.0
    frame = frame.reset_index(drop=True)
    frame["bin"] = np.floor(elapsed_hours / hours_per_timestep)
    feature_indices = frame["feature"].map(
        {name: index for index, name in enumerate(TEMPORAL_FEATURES)}
    )
    frame["feature_index"] = feature_indices

    valid_time = frame["bin"].between(0, timesteps - 1)
    valid_range = np.zeros(len(frame), dtype=bool)
    for feature, (lower, upper) in VALID_RANGES.items():
        feature_rows = frame["feature"].eq(feature)
        valid_range |= feature_rows & frame["valuenum"].between(lower, upper)
    frame = frame[valid_time & valid_range]
    if frame.empty:
        return

    coordinates = (
        frame["patient_index"].astype(int).to_numpy(),
        frame["bin"].astype(int).to_numpy(),
        frame["feature_index"].astype(int).to_numpy(),
    )
    np.add.at(sums, coordinates, frame["valuenum"].to_numpy(dtype=float))
    np.add.at(counts, coordinates, 1)


def _accumulate_static(
    frame: pd.DataFrame,
    stay_to_patient: dict[int, int],
    intime_by_patient: np.ndarray,
    static_sums: np.ndarray,
    static_counts: np.ndarray,
    observation_hours: float,
) -> None:
    static_ids = (
        WEIGHT_KG_ITEM_IDS
        | WEIGHT_LB_ITEM_IDS
        | WEIGHT_OZ_ITEM_IDS
        | HEIGHT_IN_ITEM_IDS
        | HEIGHT_CM_ITEM_IDS
    )
    frame = frame[frame["itemid"].isin(static_ids)].copy()
    if frame.empty:
        return
    frame["patient_index"] = frame["icustay_id"].map(stay_to_patient)
    frame["valuenum"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    frame["charttime"] = pd.to_datetime(frame["charttime"], errors="coerce")
    frame = frame.dropna(subset=["patient_index", "valuenum", "charttime"]).copy()
    if frame.empty:
        return

    patient_indices = frame["patient_index"].astype(int).to_numpy()
    intimes = pd.to_datetime(intime_by_patient[patient_indices])
    elapsed_hours = (
        frame["charttime"].reset_index(drop=True) - pd.Series(intimes)
    ).dt.total_seconds().to_numpy() / 3600.0
    frame = frame.reset_index(drop=True)
    frame["elapsed_hours"] = elapsed_hours
    frame = frame[frame["elapsed_hours"].between(0, observation_hours, inclusive="left")]
    if frame.empty:
        return

    values = frame["valuenum"].astype(float).copy()
    values.loc[frame["itemid"].isin(WEIGHT_LB_ITEM_IDS)] /= 2.2046226218
    values.loc[frame["itemid"].isin(WEIGHT_OZ_ITEM_IDS)] /= 35.27396195
    values.loc[frame["itemid"].isin(HEIGHT_IN_ITEM_IDS)] *= 2.54
    is_weight = frame["itemid"].isin(
        WEIGHT_KG_ITEM_IDS | WEIGHT_LB_ITEM_IDS | WEIGHT_OZ_ITEM_IDS
    )
    is_height = frame["itemid"].isin(HEIGHT_IN_ITEM_IDS | HEIGHT_CM_ITEM_IDS)
    valid = (is_weight & values.between(20.0, 400.0)) | (
        is_height & values.between(100.0, 230.0)
    )
    frame = frame[valid]
    values = values[valid]
    if frame.empty:
        return

    static_kind = np.where(is_weight[valid], 0, 1)
    coordinates = (frame["patient_index"].astype(int).to_numpy(), static_kind)
    np.add.at(static_sums, coordinates, values.to_numpy(dtype=float))
    np.add.at(static_counts, coordinates, 1)


def _forward_fill_then_mean(matrix: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    for patient in range(result.shape[0]):
        for feature in range(result.shape[2]):
            last = np.nan
            for timestamp in range(result.shape[1]):
                if np.isfinite(result[patient, timestamp, feature]):
                    last = result[patient, timestamp, feature]
                elif np.isfinite(last):
                    result[patient, timestamp, feature] = last

    finite = np.isfinite(result)
    feature_counts = finite.sum(axis=(0, 1))
    feature_means = np.full(result.shape[2], np.nan, dtype=float)
    np.divide(
        np.nansum(result, axis=(0, 1)),
        feature_counts,
        out=feature_means,
        where=feature_counts > 0,
    )
    missing_features = [
        TEMPORAL_FEATURES[index]
        for index, value in enumerate(feature_means)
        if not np.isfinite(value)
    ]
    if missing_features:
        raise ValueError(
            "No valid observations were found for required features: "
            + ", ".join(missing_features)
        )
    missing = np.where(~np.isfinite(result))
    result[missing] = feature_means[missing[2]]
    return result


def _standardize_temporal(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=(0, 1))
    standard_deviations = matrix.std(axis=(0, 1))
    standard_deviations[standard_deviations < 1.0e-8] = 1.0
    return (matrix - means) / standard_deviations


def _representative_group(value: float, boundaries: list[float], values: list[float]) -> float:
    return float(values[int(np.digitize(value, boundaries, right=False))])


def build_mimic3_data(
    input_dir: str | Path,
    timesteps: int = 8,
    hours_per_timestep: float = 6.0,
    chunksize: int = 250_000,
) -> tuple[dict[int, dict[str, object]], list[str]]:
    """Return the raw-MIMIC data in the pickle contract consumed by T4."""

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if hours_per_timestep <= 0:
        raise ValueError("hours_per_timestep must be positive")
    input_path = Path(input_dir)
    _require_files(input_path)
    cohort = _load_cohort(
        input_path,
        timesteps=timesteps,
        hours_per_timestep=hours_per_timestep,
    )
    n_patients = len(cohort)
    n_features = len(TEMPORAL_FEATURES)
    sums = np.zeros((n_patients, timesteps, n_features), dtype=float)
    counts = np.zeros((n_patients, timesteps, n_features), dtype=np.int64)
    static_sums = np.zeros((n_patients, 2), dtype=float)  # weight, height
    static_counts = np.zeros((n_patients, 2), dtype=np.int64)
    intime_by_patient = cohort["intime"].to_numpy()
    admission_to_patient = dict(
        zip(cohort["hadm_id"].astype(int), cohort["patient_index"].astype(int))
    )
    stay_to_patient = dict(
        zip(cohort["icustay_id"].astype(int), cohort["patient_index"].astype(int))
    )

    lab_ids = frozenset(LAB_ITEM_TO_FEATURE)
    for chunk in _chunks(
        _table_path(input_path, "LABEVENTS.csv"),
        ("hadm_id", "itemid", "charttime", "valuenum"),
        chunksize,
    ):
        chunk = chunk[chunk["itemid"].isin(lab_ids)]
        _accumulate_temporal(
            chunk,
            id_column="hadm_id",
            id_to_patient=admission_to_patient,
            intime_by_patient=intime_by_patient,
            item_to_feature=LAB_ITEM_TO_FEATURE,
            sums=sums,
            counts=counts,
            timesteps=timesteps,
            hours_per_timestep=hours_per_timestep,
        )

    chart_ids = frozenset(CHART_ITEM_TO_FEATURE) | (
        WEIGHT_KG_ITEM_IDS
        | WEIGHT_LB_ITEM_IDS
        | WEIGHT_OZ_ITEM_IDS
        | HEIGHT_IN_ITEM_IDS
        | HEIGHT_CM_ITEM_IDS
    )
    for chunk in _chunks(
        _table_path(input_path, "CHARTEVENTS.csv"),
        ("icustay_id", "itemid", "charttime", "valuenum"),
        chunksize,
    ):
        chunk = chunk[chunk["itemid"].isin(chart_ids)]
        _accumulate_temporal(
            chunk[chunk["itemid"].isin(CHART_ITEM_TO_FEATURE)],
            id_column="icustay_id",
            id_to_patient=stay_to_patient,
            intime_by_patient=intime_by_patient,
            item_to_feature=CHART_ITEM_TO_FEATURE,
            sums=sums,
            counts=counts,
            timesteps=timesteps,
            hours_per_timestep=hours_per_timestep,
            convert_temperatures=True,
        )
        _accumulate_static(
            chunk,
            stay_to_patient=stay_to_patient,
            intime_by_patient=intime_by_patient,
            static_sums=static_sums,
            static_counts=static_counts,
            observation_hours=timesteps * hours_per_timestep,
        )

    temporal = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=temporal, where=counts > 0)
    temporal = _forward_fill_then_mean(temporal)
    temporal = _standardize_temporal(temporal)

    static = np.full_like(static_sums, np.nan)
    np.divide(static_sums, static_counts, out=static, where=static_counts > 0)
    for column in range(static.shape[1]):
        observed = static[np.isfinite(static[:, column]), column]
        if observed.size:
            median = float(np.median(observed))
        else:
            # T4 requires both fields. These population values are used only
            # when the selected demo cohort has no valid anthropometry.
            median = 80.0 if column == 0 else 170.0
        static[~np.isfinite(static[:, column]), column] = median

    result: dict[int, dict[str, object]] = {}
    for row in cohort.itertuples(index=False):
        patient_index = int(row.patient_index)
        weight, height = static[patient_index]
        datum: dict[str, object] = {
            feature: temporal[patient_index, :, feature_index].tolist()
            for feature_index, feature in enumerate(TEMPORAL_FEATURES)
        }
        # Ordered groups follow the public AmsterdamUMCdb categories used
        # alongside MIMIC-III by T4, encoded evenly on [-1, 1].
        datum.update(
            {
                "agegroup": _representative_group(
                    float(row.age),
                    [40, 50, 60, 70, 80],
                    [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
                ),
                "heightgroup": _representative_group(
                    float(height),
                    [160, 170, 180, 190],
                    [-1.0, -0.5, 0.0, 0.5, 1.0],
                ),
                "weightgroup": _representative_group(
                    float(weight),
                    [60, 70, 80, 90, 100, 110],
                    [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                ),
                "gender": float(str(row.gender).upper() == "M"),
                "death": int(row.hospital_expire_flag),
            }
        )
        result[int(row.subject_id)] = datum
    return result, list(TEMPORAL_FEATURES)


def write_pickles(
    input_dir: str | Path,
    output_dir: str | Path,
    timesteps: int = 8,
    hours_per_timestep: float = 6.0,
    chunksize: int = 250_000,
) -> tuple[Path, Path]:
    data, header = build_mimic3_data(
        input_dir,
        timesteps=timesteps,
        hours_per_timestep=hours_per_timestep,
        chunksize=chunksize,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = output_path / "mimic3.pkl"
    header_path = output_path / "header.pkl"
    with data_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with header_path.open("wb") as handle:
        pickle.dump(header, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Wrote {len(data)} patients with {len(header)} temporal features "
        f"and {timesteps} timestamps to {data_path}"
    )
    return data_path, header_path


