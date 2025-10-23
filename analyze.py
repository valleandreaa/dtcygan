#!/usr/bin/env python
"""Compact analysis CLI for cohort, histology, and FNCLCC summaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtcygan.visualization import (
    DEFAULT_TREATMENT_GROUPS,
    as_prob,
    compute_individualized_treatment_effects,
    load_plot_config,
    plot_endpoint_waterfall_fan_grid,
    plot_three_endpoint_trajectories_with_bands,
)
from dtcygan.training import SyntheticSequenceDataset, set_seed
from dtcygan.eval_utils import (
    load_dataset_payload,
    ensure_probability_metadata,
    load_checkpoint_bundle,
    patients_to_dataframe,
)
from validate import build_counterfactual_patients as build_counterfactual_patients_full

DEFAULT_IMG_DIR = "imgs/analysis"
EPS = 1e-12
OUTCOME_LABELS = ["DOD", "NED", "AWD"]

# ------------------------------------------------------------------ #
# Column name candidates (customise if the dataset uses different ids)
# ------------------------------------------------------------------ #
LOCAL_RECURRENCE_CANDIDATES = [
    "fake_episodes.treatments.fields.endpoint_1.0_prob",
    "endpoint_1_prob",
    "local_recurrence_prob",
    "local_recurrence",
]
METASTATIC_CANDIDATES = [
    "fake_episodes.treatments.fields.endpoint_2.0_prob",
    "endpoint_2_prob",
    "metastasis_prob",
    "metastasis",
]
ENDPOINT_DOD_CANDIDATES = [
    "fake_episodes.treatments.fields.endpoint_3.0_prob",
    "endpoint_3_prob",
    "death_of_disease_prob",
    "death_of_disease",
]
DEAD_OF_DISEASE_CANDIDATES = [
    "fake_episodes.diagnosis.fields.status_DOD_prob",
    "status_DOD_prob",
    "dod_prob",
]
STATUS_AWD_CANDIDATES = [
    "fake_episodes.diagnosis.fields.status_AWD_prob",
    "status_AWD_prob",
]
STATUS_NED_CANDIDATES = [
    "fake_episodes.diagnosis.fields.status_NED_prob",
    "status_NED_prob",
]
SURGERY_FLAG_CANDIDATES = [
    "surgery",
    "surgery_performed",
    "episodes.surgery",
]
RADIOTHERAPY_FLAG_CANDIDATES = [
    "radiotherapy",
    "radiotherapy_administered",
    "episodes.radiotherapy",
]
CHEMOTHERAPY_FLAG_CANDIDATES = [
    "chemotherapy",
    "chemotherapy_administered",
    "episodes.chemotherapy",
]
TIMESTEP_CANDIDATES = ["timestep", "time", "step"]
HISTOLOGY_CANDIDATES = [
    "tumor_characteristics.histological_diagnosis",
    "histological_subtype",
]
FNCLCC_CANDIDATES = [
    "tumor_characteristics.grading_fnclcc",
    "fnclcc_grade",
]

# Resolved column names (initialised to the first candidate; updated at runtime)
LOCAL_RECURRENCE_COL = LOCAL_RECURRENCE_CANDIDATES[0]
METASTATIC_COL = METASTATIC_CANDIDATES[0]
ENDPOINT_DOD_COL = ENDPOINT_DOD_CANDIDATES[0]
DEAD_OF_DISEASE = DEAD_OF_DISEASE_CANDIDATES[0]
STATUS_AWD_COL = STATUS_AWD_CANDIDATES[0]
STATUS_NED_COL = STATUS_NED_CANDIDATES[0]
SURGERY_FLAG_COL = SURGERY_FLAG_CANDIDATES[0]
RADIOTHERAPY_FLAG_COL = RADIOTHERAPY_FLAG_CANDIDATES[0]
CHEMOTHERAPY_FLAG_COL = CHEMOTHERAPY_FLAG_CANDIDATES[0]
TIMESTEP_COL = TIMESTEP_CANDIDATES[0]
HISTOLOGICAL_DIAGNOSIS_COL = HISTOLOGY_CANDIDATES[0]
FNCLCC_GRADING_COL = FNCLCC_CANDIDATES[0]


def _slugify(value: object) -> str:
    '''
    Convert a raw value into a lowercase alphanumeric slug.

    args:
    - value: object to be converted into a slug [object]

    return:
    - slug: slugified representation suitable for file names [str]
    '''
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def _resolve_column_aliases(df: pd.DataFrame) -> None:
    '''
    Align known column aliases with the columns present in the dataframe.

    args:
    - df: dataset whose columns require alias resolution [pd.DataFrame]

    return:
    - None: updates global alias variables in place [None]
    '''
    global LOCAL_RECURRENCE_COL, METASTATIC_COL, ENDPOINT_DOD_COL, DEAD_OF_DISEASE
    global STATUS_AWD_COL, STATUS_NED_COL
    global SURGERY_FLAG_COL, RADIOTHERAPY_FLAG_COL, CHEMOTHERAPY_FLAG_COL
    global TIMESTEP_COL, HISTOLOGICAL_DIAGNOSIS_COL, FNCLCC_GRADING_COL

    def pick(candidates: List[str], current: str) -> str:
        '''
        Select the first matching column name from the provided candidates.

        args:
        - candidates: possible column names ordered by preference [List[str]]
        - current: fallback column name when no candidates match [str]

        return:
        - column: resolved column name to use downstream [str]
        '''
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return current

    LOCAL_RECURRENCE_COL = pick(LOCAL_RECURRENCE_CANDIDATES, LOCAL_RECURRENCE_COL)
    METASTATIC_COL = pick(METASTATIC_CANDIDATES, METASTATIC_COL)
    ENDPOINT_DOD_COL = pick(ENDPOINT_DOD_CANDIDATES, ENDPOINT_DOD_COL)
    DEAD_OF_DISEASE = pick(DEAD_OF_DISEASE_CANDIDATES, DEAD_OF_DISEASE)
    STATUS_AWD_COL = pick(STATUS_AWD_CANDIDATES, STATUS_AWD_COL)
    STATUS_NED_COL = pick(STATUS_NED_CANDIDATES, STATUS_NED_COL)
    SURGERY_FLAG_COL = pick(SURGERY_FLAG_CANDIDATES, SURGERY_FLAG_COL)
    RADIOTHERAPY_FLAG_COL = pick(RADIOTHERAPY_FLAG_CANDIDATES, RADIOTHERAPY_FLAG_COL)
    CHEMOTHERAPY_FLAG_COL = pick(CHEMOTHERAPY_FLAG_CANDIDATES, CHEMOTHERAPY_FLAG_COL)
    TIMESTEP_COL = pick(TIMESTEP_CANDIDATES, TIMESTEP_COL)
    HISTOLOGICAL_DIAGNOSIS_COL = pick(HISTOLOGY_CANDIDATES, HISTOLOGICAL_DIAGNOSIS_COL)
    FNCLCC_GRADING_COL = pick(FNCLCC_CANDIDATES, FNCLCC_GRADING_COL)


def min_max_normalize(series: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    '''
    Normalize a numeric series using quantile clipping for robustness.

    args:
    - series: data to normalise [pd.Series]
    - q_low: lower quantile used for clipping [float]
    - q_high: upper quantile used for clipping [float]

    return:
    - normalized: normalized values bounded within [0, 1] [pd.Series]
    '''
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    q_min = s.quantile(q_low)
    q_max = s.quantile(q_high)
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max == q_min:
        return pd.Series(np.zeros_like(s, dtype=float), index=s.index)
    clipped = s.clip(lower=q_min, upper=q_max)
    out = (clipped - q_min) / (q_max - q_min)
    return out


def load_and_prepare(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    '''
    Load patient data and prepare standardized probability features.

    args:
    - source: CSV path or dataframe containing patient rows [str | Path | pd.DataFrame]

    return:
    - df_prepared: processed dataframe ready for downstream analysis [pd.DataFrame]
    '''
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    _resolve_column_aliases(df)

    def _minmax_inplace(frame: pd.DataFrame, cols: List[str]) -> None:
        '''
        Apply min-max normalization to the provided numeric columns.

        args:
        - frame: dataframe that will be mutated in place [pd.DataFrame]
        - cols: list of column names to normalise [List[str]]

        return:
        - None: updates the dataframe columns directly [None]
        '''
        present = [c for c in cols if c in frame.columns]
        if not present:
            return
        for col in present:
            frame[col] = min_max_normalize(frame[col])

    _minmax_inplace(df, [LOCAL_RECURRENCE_COL, METASTATIC_COL, ENDPOINT_DOD_COL])
    _minmax_inplace(df, [STATUS_AWD_COL, DEAD_OF_DISEASE, STATUS_NED_COL])

    def _assign_probability_column(target: str, source: str, *, force: bool = False) -> None:
        '''
        Create a probability-style column if source data is available.

        args:
        - target: destination column name for the normalized probabilities [str]
        - source: raw probability column to transform [str]
        - force: when True, ensure the column exists even if data missing [bool]

        return:
        - None: writes the computed column or NaNs when data is absent [None]
        '''
        if source not in df.columns:
            if force and target not in df.columns:
                df[target] = np.nan
            return
        raw = pd.to_numeric(df[source], errors="coerce")
        if raw.notna().any():
            df[target] = as_prob(raw)
        elif force and target not in df.columns:
            df[target] = np.nan

    probability_sources = {
        "LR_prob_norm_global": (LOCAL_RECURRENCE_COL, False),
        "MET_prob_norm_global": (METASTATIC_COL, False),
        "DOD_prob_norm_global": (DEAD_OF_DISEASE, True),
    }
    for target, (source, force) in probability_sources.items():
        _assign_probability_column(target, source, force=force)

    df["prob_norm"] = df.get(
        "LR_prob_norm_global",
        pd.Series(np.nan, index=df.index, dtype=float),
    )
    df["group"] = _make_group_label(df)
    return df

def _gmd_norm(values: np.ndarray) -> float:
    '''
    Compute the Gini mean difference for a set of probability values.

    args:
    - values: array of probability values [np.ndarray]

    return:
    - gmd: twice the Gini mean difference used for dispersion scoring [float]
    '''
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n < 2:
        return 0.0
    w = (2 * np.arange(1, n + 1) - n - 1)
    gmd = (2.0 / (n * (n - 1))) * float(np.sum(w * x))
    return float(2.0 * gmd)


def _auc_trapz(t: np.ndarray, y: np.ndarray) -> float:
    '''
    Integrate a curve using the trapezoidal rule with float output.

    args:
    - t: monotonic time or step vector [np.ndarray]
    - y: values sampled at the same resolution as t [np.ndarray]

    return:
    - auc: area under the curve defined by t and y [float]
    '''
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0 or y.size == 0:
        return float("nan")
    return float(np.trapezoid(y, t))


def _arm_curves(time_to_probs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Build median and GMD curves for an arm across timesteps.

    args:
    - time_to_probs: mapping from timestep to probability samples [Dict[int, np.ndarray]]

    return:
    - curves: times, median values, and dispersion estimates [Tuple[np.ndarray, np.ndarray, np.ndarray]]
    '''
    times = np.array(sorted(time_to_probs.keys()), dtype=float)
    med, gmd = [], []
    for tt in times:
        p = np.asarray(time_to_probs[tt], dtype=float)
        p = np.clip(p, 0.0, 1.0)
        if p.size == 0:
            med.append(np.nan)
            gmd.append(np.nan)
        else:
            med.append(float(np.median(p)))
            gmd.append(_gmd_norm(p))
    return times, np.array(med, dtype=float), np.array(gmd, dtype=float)


def _arm_score(time_to_probs: Dict[int, np.ndarray], lam_gmd: float = 0.0) -> Dict[str, Any]:
    '''
    Aggregate arm performance into summary scores and curves.

    args:
    - time_to_probs: probability samples over time for one arm [Dict[int, np.ndarray]]
    - lam_gmd: weight applied to dispersion when computing the score [float]

    return:
    - metrics: computed AUC metrics and supporting curves [Dict[str, Any]]
    '''
    t, med, gmd = _arm_curves(time_to_probs)
    auc_m = _auc_trapz(t, med)
    auc_g = _auc_trapz(t, gmd)
    return {
        "AUC_median": auc_m,
        "AUC_GMD*": auc_g,
        "risk_averse_score": (auc_m + lam_gmd * auc_g),
        "times": t,
        "median_curve": med,
        "gmd_curve": gmd,
    }


def _compare_arms(
    time_to_probs_A: Dict[int, np.ndarray],
    time_to_probs_B: Dict[int, np.ndarray],
    *,
    lam_gmd: float = 0.0,
    n_boot: int = 2000,
    seed: Optional[int] = 42,
    paired: bool = True,
) -> Dict[str, Any]:
    '''
    Compare two treatment arms with bootstrap uncertainty estimates.

    args:
    - time_to_probs_A: probabilities over time for baseline arm [Dict[int, np.ndarray]]
    - time_to_probs_B: probabilities over time for comparative arm [Dict[int, np.ndarray]]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - n_boot: number of bootstrap replicates [int]
    - seed: optional random seed for reproducibility [Optional[int]]
    - paired: whether to resample paired observations [bool]

    return:
    - summary: percent change and bootstrap confidence interval [Dict[str, Any]]
    '''
    rng = np.random.default_rng(seed)

    sA = _arm_score(time_to_probs_A, lam_gmd)["risk_averse_score"]
    sB = _arm_score(time_to_probs_B, lam_gmd)["risk_averse_score"]

    def _percent_change(a: float, b: float) -> float:
        '''
        Compute percent change while guarding against divide-by-zero.

        args:
        - a: baseline score [float]
        - b: comparative score [float]

        return:
        - delta: percent change from a to b [float]
        '''
        if not np.isfinite(a) or abs(a) < EPS:
            return float("nan")
        return float((b - a) / a * 100.0)

    delta = _percent_change(sA, sB)

    t = sorted(set(time_to_probs_A.keys()) & set(time_to_probs_B.keys()))
    A_arr = {tt: np.asarray(time_to_probs_A[tt], dtype=float) for tt in t}
    B_arr = {tt: np.asarray(time_to_probs_B[tt], dtype=float) for tt in t}
    n_by_t = {tt: min(A_arr[tt].size, B_arr[tt].size) for tt in t}

    def resample_once() -> float:
        '''
        Draw one bootstrap replicate and compute the percent change.

        args:
        - None: uses closure state for probability arrays [None]

        return:
        - delta_boot: percent change for the bootstrap sample [float]
        '''
        A_s, B_s = {}, {}
        for tt in t:
            n = n_by_t[tt]
            if n == 0:
                A_s[tt] = np.array([])
                B_s[tt] = np.array([])
                continue
            if paired:
                idx = rng.integers(0, n, size=n)
                A_s[tt] = A_arr[tt][:n][idx]
                B_s[tt] = B_arr[tt][:n][idx]
            else:
                A_s[tt] = A_arr[tt][rng.integers(0, A_arr[tt].size, size=n)]
                B_s[tt] = B_arr[tt][rng.integers(0, B_arr[tt].size, size=n)]
        score_A = _arm_score(A_s, lam_gmd)["risk_averse_score"]
        score_B = _arm_score(B_s, lam_gmd)["risk_averse_score"]
        return _percent_change(score_A, score_B)

    boots = np.array([resample_once() for _ in range(n_boot)], dtype=float)
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        lo = hi = p_two = float("nan")
    else:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_two = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {
        "percent_change": float(delta),
        "CI95": (float(lo), float(hi)),
        "p~": float(p_two),
    }


def _make_group_label(df: pd.DataFrame) -> pd.Series:
    '''
    Derive treatment group labels based on modality indicator columns.

    args:
    - df: dataframe containing treatment indicators [pd.DataFrame]

    return:
    - labels: categorical group labels aligned with df index [pd.Series]
    '''
    conditions = [
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 0)
        & (df[CHEMOTHERAPY_FLAG_COL] == 0),
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 1)
        & (df[CHEMOTHERAPY_FLAG_COL] == 0),
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 0)
        & (df[CHEMOTHERAPY_FLAG_COL] == 1),
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 1)
        & (df[CHEMOTHERAPY_FLAG_COL] == 1),
    ]
    choices = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    return np.select(conditions, choices, default="Other")


def _build_time_to_probs_from_df(df: pd.DataFrame, endpoint_col: str, arm_label: str) -> Dict[int, np.ndarray]:
    '''
    Gather probability trajectories for a single treatment arm.

    args:
    - df: dataframe containing patient trajectories [pd.DataFrame]
    - endpoint_col: column with endpoint probabilities [str]
    - arm_label: desired treatment arm label [str]

    return:
    - time_map: mapping from timestep to probability samples [Dict[int, np.ndarray]]
    '''
    sub = df[df["group"] == arm_label]
    cols = [TIMESTEP_COL, endpoint_col]
    has_patient = "patient_id" in sub.columns
    if has_patient:
        cols.append("patient_id")
    sub = sub[cols].copy()
    sub[endpoint_col] = pd.to_numeric(sub[endpoint_col], errors="coerce").astype(float).clip(0.0, 1.0)
    sub = sub.dropna(subset=[endpoint_col, TIMESTEP_COL])
    if sub.empty:
        return {}
    group_cols = [TIMESTEP_COL] + (["patient_id"] if has_patient else [])
    grouped = sub.groupby(group_cols)[endpoint_col].mean().reset_index()
    return {
        int(timestep): group[endpoint_col].to_numpy(dtype=float)
        for timestep, group in grouped.groupby(TIMESTEP_COL)
    }


def compare_two_arms_df(
    df: pd.DataFrame,
    endpoint_col: str,
    arm_A: str = "Surgery only",
    arm_B: str = "Surgery + RT",
    *,
    lam_gmd: float = 0.0,
    n_boot: int = 2000,
    paired: bool = True,
    seed: Optional[int] = 42,
) -> Optional[Dict[str, Any]]:
    '''
    Compare two arms directly from a dataframe using bootstrap sampling.

    args:
    - df: dataframe that includes treatment groups and endpoints [pd.DataFrame]
    - endpoint_col: probability column to analyse [str]
    - arm_A: baseline arm name [str]
    - arm_B: comparative arm name [str]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - n_boot: number of bootstrap replicates [int]
    - paired: whether to use paired bootstrap resampling [bool]
    - seed: optional random seed [Optional[int]]

    return:
    - summary: comparison summary or None when data missing [Optional[Dict[str, Any]]]
    '''
    df = df.copy()
    df["group"] = _make_group_label(df)

    A = _build_time_to_probs_from_df(df, endpoint_col, arm_A)
    B = _build_time_to_probs_from_df(df, endpoint_col, arm_B)
    if not A or not B:
        return None
    return _compare_arms(A, B, lam_gmd=lam_gmd, n_boot=n_boot, seed=seed, paired=paired)


def risk_averse_comparison_grid(
    df: pd.DataFrame,
    *,
    lam_gmd: float = 0.0,
    n_boot: int = 200,
    paired: bool = True,
    seed: Optional[int] = 42,
    baseline: str = "Surgery only",
    arms: Optional[List[str]] = None,
    endpoints: Optional[List[str]] = None,
) -> pd.DataFrame:
    '''
    Generate a grid of risk-averse comparisons across arms and endpoints.

    args:
    - df: dataframe containing patient trajectories [pd.DataFrame]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - n_boot: number of bootstrap replicates [int]
    - paired: whether to bootstrap paired observations [bool]
    - seed: optional random seed for reproducibility [Optional[int]]
    - baseline: reference treatment arm [str]
    - arms: additional arms to compare against the baseline [Optional[List[str]]]
    - endpoints: endpoint columns considered in the comparison [Optional[List[str]]]

    return:
    - grid: tidy dataframe summarising arm comparisons [pd.DataFrame]
    '''
    df = df.copy()
    df["group"] = _make_group_label(df)
    if endpoints is None:
        endpoints = [
            "LR_prob_norm_global" if "LR_prob_norm_global" in df.columns else LOCAL_RECURRENCE_COL,
            "MET_prob_norm_global" if "MET_prob_norm_global" in df.columns else METASTATIC_COL,
            "DOD_prob_norm_global" if "DOD_prob_norm_global" in df.columns else DEAD_OF_DISEASE,
        ]
    if arms is None:
        arms = ["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    rows: List[Dict[str, Any]] = [
        {
            "endpoint": ep,
            "baseline": baseline,
            "arm": arm,
            "percent_change": res["percent_change"],
            "ci_low": float(res["CI95"][0]),
            "ci_high": float(res["CI95"][1]),
            "p_tilde": res["p~"],
        }
        for ep in endpoints
        for arm in arms
        for res in [
            compare_two_arms_df(
                df,
                endpoint_col=ep,
                arm_A=baseline,
                arm_B=arm,
                lam_gmd=lam_gmd,
                n_boot=n_boot,
                paired=paired,
                seed=seed,
            )
        ]
        if res
    ]
    return pd.DataFrame(rows)


def compute_category_distribution(
    df: pd.DataFrame,
    column: str,
    *,
    label_column: str,
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    '''
    Summarise patient counts and percentages for a categorical variable.

    args:
    - df: dataframe containing category information [pd.DataFrame]
    - column: column whose distribution is calculated [str]
    - label_column: friendly label for the output column [str]
    - patient_col: identifier used to deduplicate patients [str]

    return:
    - table: table containing counts and percentages per category [pd.DataFrame]
    '''
    counts = (
        df.groupby(column)[patient_col].nunique()
        if patient_col in df.columns
        else df[column].value_counts(dropna=False)
    ).sort_values(ascending=False)
    table = counts.rename("nr_patients").reset_index().rename(columns={column: label_column})
    table[label_column] = table[label_column].fillna("Unknown").astype(str)
    total = table["nr_patients"].sum()
    table["percent"] = table["nr_patients"].div(total).mul(100.0) if total else np.nan
    return table


def _iter_category_subsets(df: pd.DataFrame, column: str) -> List[Tuple[str, pd.DataFrame]]:
    '''
    Yield labelled subsets for each category value including missing data.

    args:
    - df: dataframe to partition [pd.DataFrame]
    - column: column whose unique values define the subsets [str]

    return:
    - subsets: list of (label, subset dataframe) pairs [List[Tuple[str, pd.DataFrame]]]
    '''
    series = df[column]
    known = sorted(series.dropna().unique(), key=lambda x: str(x).lower())
    subsets = [(str(val), df.loc[series == val].copy()) for val in known]
    if series.isna().any():
        subsets.append(("Unknown", df.loc[series.isna()].copy()))
    return subsets


def _subset_paths(output_dir: Path, slug: str) -> Dict[str, Path]:
    '''
    Construct output paths for artefacts derived from a subset analysis.

    args:
    - output_dir: base directory where artefacts are stored [Path]
    - slug: identifier appended to file names [str]

    return:
    - paths: mapping of artefact type to filesystem path [Dict[str, Path]]
    '''
    paths = {
        "ite": output_dir / f"{slug}_ite.csv",
        "waterfall": output_dir / f"{slug}_waterfall_fan.png",
        "trajectories": output_dir / f"{slug}_trajectories.png",
        "risk": output_dir / f"{slug}_risk_averse.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def _endpoint_specs_for_work(work: pd.DataFrame, endpoint_map: Mapping[str, str]) -> List[Tuple[str, str]]:
    '''
    Resolve endpoint display names and source columns for plotting.

    args:
    - work: dataframe subset under consideration [pd.DataFrame]
    - endpoint_map: mapping of logical endpoint names to column aliases [Mapping[str, str]]

    return:
    - endpoints: list of (display name, column) pairs [List[Tuple[str, str]]]
    '''
    return [
        (
            "Local Recurrence",
            "LR_prob_norm_global" if "LR_prob_norm_global" in work.columns else endpoint_map["local_recurrence"],
        ),
        (
            "Metastasis",
            "MET_prob_norm_global" if "MET_prob_norm_global" in work.columns else endpoint_map["metastasis"],
        ),
        (
            "DOD",
            "DOD_prob_norm_global" if "DOD_prob_norm_global" in work.columns else endpoint_map["death_of_disease"],
        ),
    ]


def _write_ite_artifacts(
    work: pd.DataFrame,
    *,
    label: str,
    prefix: str,
    paths: Dict[str, Path],
    bootstrap: int,
    seed: Optional[int],
    endpoint_map: Mapping[str, str],
    plot_config: Mapping[str, Any],
) -> None:
    '''
    Persist ITE tables and waterfall plots for a dataframe subset.

    args:
    - work: dataframe subset with probabilities and groups [pd.DataFrame]
    - label: human-readable subset label [str]
    - prefix: prefix applied to output artefact names [str]
    - paths: precomputed output paths per artefact type [Dict[str, Path]]
    - bootstrap: number of bootstrap samples for plotting [int]
    - seed: optional random seed [Optional[int]]
    - endpoint_map: endpoint column mapping for modelling utilities [Mapping[str, str]]
    - plot_config: configuration dictionary for plots [Mapping[str, Any]]

    return:
    - None: writes files to disk and prints progress [None]
    '''
    ite_df = compute_individualized_treatment_effects(
        work,
        groups=DEFAULT_TREATMENT_GROUPS,
        reference_group="Surgery only",
        patient_col="patient_id",
        time_col=TIMESTEP_COL,
        endpoint_cols=endpoint_map,
        group_col="group",
    )
    if ite_df.empty:
        print(f"{prefix}={label}: no ITE rows")
        return
    ite_df.to_csv(paths["ite"], index=False)
    plot_endpoint_waterfall_fan_grid(
        ite_df,
        histology=label,
        out_path=paths["waterfall"],
        n_boot=bootstrap,
        seed=seed,
        config=plot_config.get("waterfall", {}),
    )
    print(f"{prefix}={label}: wrote {paths['ite']} / {paths['waterfall']}")


def _plot_trajectories_artifacts(
    work: pd.DataFrame,
    *,
    label: str,
    prefix: str,
    path: Path,
    bootstrap: int,
    seed: Optional[int],
    endpoint_map: Mapping[str, str],
    plot_config: Mapping[str, Any],
) -> None:
    '''
    Produce treatment trajectory plots for a dataframe subset.

    args:
    - work: dataframe subset with probability trajectories [pd.DataFrame]
    - label: human-readable subset label [str]
    - prefix: prefix applied to output artefact names [str]
    - path: output path for the trajectories figure [Path]
    - bootstrap: number of bootstrap samples for uncertainty bands [int]
    - seed: optional random seed [Optional[int]]
    - endpoint_map: mapping of endpoint names to columns [Mapping[str, str]]
    - plot_config: configuration dictionary for plots [Mapping[str, Any]]

    return:
    - None: writes a plot to disk and logs progress [None]
    '''
    trajectories_cfg = plot_config.get("trajectories", {})
    out_path = plot_three_endpoint_trajectories_with_bands(
        work,
        endpoints=_endpoint_specs_for_work(work, endpoint_map),
        histology=label,
        out_path=path,
        n_boot=bootstrap,
        seed=seed,
        groups=DEFAULT_TREATMENT_GROUPS,
        time_col=TIMESTEP_COL,
        patient_col="patient_id",
        group_col="group",
        histology_col=HISTOLOGICAL_DIAGNOSIS_COL if HISTOLOGICAL_DIAGNOSIS_COL in work.columns else None,
        agg=trajectories_cfg.get("agg", "median"),
        band_method=trajectories_cfg.get("band_method", "rcqe"),
        config=trajectories_cfg,
    )
    print(f"{prefix}={label}: wrote {out_path}")


def _write_risk_grid(
    work: pd.DataFrame,
    *,
    label: str,
    prefix: str,
    path: Path,
    lam_gmd: float,
    n_boot: int,
    seed: Optional[int],
) -> None:
    '''
    Generate the risk-averse comparison grid for a dataframe subset.

    args:
    - work: dataframe subset with treatment probabilities [pd.DataFrame]
    - label: human-readable subset label [str]
    - prefix: prefix applied to output artefact names [str]
    - path: destination CSV path [Path]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - n_boot: number of bootstrap replicates [int]
    - seed: optional random seed [Optional[int]]

    return:
    - None: writes a CSV grid summarising risk-averse comparisons [None]
    '''
    grid = risk_averse_comparison_grid(
        work,
        lam_gmd=lam_gmd,
        n_boot=n_boot,
        paired=True,
        seed=seed,
        baseline="Surgery only",
    )
    if grid.empty:
        print(f"{prefix}={label}: no risk grid")
        return
    grid.to_csv(path, index=False)
    print(f"{prefix}={label}: wrote {path}")


def _analyze_subset(
    df: pd.DataFrame,
    *,
    label: str,
    prefix: str,
    output_dir: Path,
    bootstrap: int,
    seed: Optional[int],
    lam_gmd: float,
    plot_config: Mapping[str, Any],
) -> None:
    '''
    Run all analyses for a dataframe subset and persist artefacts.

    args:
    - df: dataframe or subset to analyse [pd.DataFrame]
    - label: descriptive label for logs and artefacts [str]
    - prefix: prefix applied to output artefact names [str]
    - output_dir: directory where artefacts are written [Path]
    - bootstrap: number of bootstrap replicates [int]
    - seed: optional random seed [Optional[int]]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - plot_config: configuration dictionary for plots [Mapping[str, Any]]

    return:
    - None: writes artefacts to disk and prints status [None]
    '''
    if df.empty:
        print(f"{prefix}={label}: no rows")
        return

    work = df.copy()
    if "group" not in work.columns:
        work["group"] = _make_group_label(work)

    slug = f"{prefix}_{_slugify(label)}"
    paths = _subset_paths(output_dir, slug)
    endpoint_map = {
        "local_recurrence": LOCAL_RECURRENCE_COL,
        "metastasis": METASTATIC_COL,
        "death_of_disease": DEAD_OF_DISEASE,
    }

    _write_ite_artifacts(
        work,
        label=label,
        prefix=prefix,
        paths=paths,
        bootstrap=bootstrap,
        seed=seed,
        endpoint_map=endpoint_map,
        plot_config=plot_config,
    )
    _plot_trajectories_artifacts(
        work,
        label=label,
        prefix=prefix,
        path=paths["trajectories"],
        bootstrap=bootstrap,
        seed=seed,
        endpoint_map=endpoint_map,
        plot_config=plot_config,
    )
    _write_risk_grid(
        work,
        label=label,
        prefix=prefix,
        path=paths["risk"],
        lam_gmd=lam_gmd,
        n_boot=bootstrap,
        seed=seed,
    )


def run_category_analysis(
    df: pd.DataFrame,
    *,
    column: str,
    label_column: str,
    prefix: str,
    output_dir: Path,
    bootstrap: int,
    seed: Optional[int],
    lam_gmd: float,
    plot_config: Mapping[str, Any],
) -> None:
    '''
    Execute categorical subgroup analyses and persist their artefacts.

    args:
    - df: dataframe that includes the subgroup column [pd.DataFrame]
    - column: column used to define categories [str]
    - label_column: friendly name for the exported column [str]
    - prefix: prefix applied to output artefact names [str]
    - output_dir: directory where artefacts are written [Path]
    - bootstrap: number of bootstrap replicates [int]
    - seed: optional random seed [Optional[int]]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - plot_config: configuration dictionary for plots [Mapping[str, Any]]

    return:
    - None: saves distribution tables and subgroup artefacts [None]
    '''
    if column not in df.columns:
        print(f"Column '{column}' not found; skipping {prefix} analysis.")
        return

    table = compute_category_distribution(df, column, label_column=label_column)
    if not table.empty:
        table_path = output_dir / f"{prefix}_distribution.csv"
        table.to_csv(table_path, index=False)
        print(f"Saved {prefix} distribution to {table_path}")

    for label, subset in _iter_category_subsets(df, column):
        _analyze_subset(
            subset,
            label=label,
            prefix=prefix,
            output_dir=output_dir,
            bootstrap=bootstrap,
            seed=seed,
            lam_gmd=lam_gmd,
            plot_config=plot_config,
        )


def run_histology_analysis(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    bootstrap: int,
    seed: Optional[int],
    lam_gmd: float,
    skip: bool,
    plot_config: Mapping[str, Any],
) -> None:
    '''
    Orchestrate per-histology analyses unless explicitly skipped.

    args:
    - df: dataframe with histology information [pd.DataFrame]
    - output_dir: directory where artefacts are written [Path]
    - bootstrap: number of bootstrap replicates [int]
    - seed: optional random seed [Optional[int]]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - skip: flag to skip histology analysis entirely [bool]
    - plot_config: configuration dictionary for plots [Mapping[str, Any]]

    return:
    - None: conditionally performs histology subgroup analyses [None]
    '''
    if skip:
        print("Skipping histology aggregation as requested.")
        return
    run_category_analysis(
        df,
        column=HISTOLOGICAL_DIAGNOSIS_COL,
        label_column="histology",
        prefix="histology",
        output_dir=output_dir,
        bootstrap=bootstrap,
        seed=seed,
        lam_gmd=lam_gmd,
        plot_config=plot_config,
    )


def run_grade_analysis(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    bootstrap: int,
    seed: Optional[int],
    lam_gmd: float,
    skip: bool,
    plot_config: Mapping[str, Any],
) -> None:
    '''
    Orchestrate per-grade analyses unless explicitly skipped.

    args:
    - df: dataframe with FNCLCC grading information [pd.DataFrame]
    - output_dir: directory where artefacts are written [Path]
    - bootstrap: number of bootstrap replicates [int]
    - seed: optional random seed [Optional[int]]
    - lam_gmd: dispersion weight for risk-averse scoring [float]
    - skip: flag to skip grade analysis entirely [bool]
    - plot_config: configuration dictionary for plots [Mapping[str, Any]]

    return:
    - None: conditionally performs grade subgroup analyses [None]
    '''
    if skip:
        print("Skipping FNCLCC grade aggregation as requested.")
        return
    run_category_analysis(
        df,
        column=FNCLCC_GRADING_COL,
        label_column="fnclcc_grade",
        prefix="fnclcc",
        output_dir=output_dir,
        bootstrap=bootstrap,
        seed=seed,
        lam_gmd=lam_gmd,
        plot_config=plot_config,
    )


def generate_counterfactual_dataframe(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: Optional[int],
) -> Tuple[pd.DataFrame, Optional[int]]:
    '''
    Rebuild counterfactual trajectories and prepare them for analysis.

    args:
    - dataset_path: dataset payload path [str | Path]
    - checkpoint_path: checkpoint containing model weights [str | Path]
    - seed: optional RNG seed overriding the checkpoint [Optional[int]]

    return:
    - df_and_seed: prepared dataframe and effective seed [Tuple[pd.DataFrame, Optional[int]]]
    '''
    cfg, metadata, generator, validation_ids, device = load_checkpoint_bundle(checkpoint_path)
    seed_value = seed if seed is not None else cfg.seed
    if seed_value is not None:
        set_seed(seed_value)

    dataset_payload = load_dataset_payload(dataset_path)
    dataset = SyntheticSequenceDataset(
        dataset_payload,
        cfg.seq_len,
        spec_clinical=metadata.get("clinical_feature_spec"),
        spec_treatment=metadata.get("treatment_feature_spec"),
    )

    status_labels = list(dataset.treat_categories.get("status_at_last_follow_up", OUTCOME_LABELS))
    endpoint_labels = list(dataset.treat_categories.get("treatment_endpoint_category", []))

    cf_patients = build_counterfactual_patients_full(
        dataset,
        generator,
        device,
        validation_ids=None,
        status_labels=status_labels,
        endpoint_labels=endpoint_labels,
        samples_per_patient=1,
    )

    status_labels = sorted(status_labels)
    endpoint_labels = sorted(endpoint_labels)

    ensure_probability_metadata(cf_patients, status_labels, endpoint_labels)

    df_raw = patients_to_dataframe(
        cf_patients,
        status_labels=status_labels,
        endpoint_labels=endpoint_labels,
        include_timesteps=True,
    )

    return load_and_prepare(df_raw), seed_value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate histology and FNCLCC analyses from counterfactual trajectories.",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path("data") / "syntects.json"),
        help="Dataset JSON used to rebuild counterfactuals (default: data/syntects.json).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path("data") / "models" / "dtcygan_20250928_135339.pt"),
        help="Checkpoint containing generator weights (default: data/models/dtcygan_20250928_135339.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_IMG_DIR,
        help="Directory for generated tables and figures (default: imgs/analysis).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap replicates for uncertainty estimates (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed overriding the checkpoint configuration.",
    )
    parser.add_argument(
        "--lambda-gmd",
        type=float,
        default=0.0,
        help="Weight applied to the dispersion term in risk-averse scoring (default: 0.0).",
    )
    parser.add_argument(
        "--skip-histology",
        action="store_true",
        help="Skip per-histology aggregation and plots.",
    )
    parser.add_argument(
        "--skip-grade",
        action="store_true",
        help="Skip per-grade aggregation and plots.",
    )
    parser.add_argument(
        "--plot-config",
        type=str,
        help="Optional YAML file overriding plot defaults (defaults to packaged settings).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global DEFAULT_IMG_DIR
    DEFAULT_IMG_DIR = str(output_dir)

    df, seed_value = generate_counterfactual_dataframe(
        args.dataset,
        args.checkpoint,
        seed=args.seed,
    )
    plot_config = load_plot_config(args.plot_config)

    _analyze_subset(
        df,
        label="All patients",
        prefix="cohort",
        output_dir=output_dir,
        bootstrap=args.bootstrap,
        seed=seed_value,
        lam_gmd=args.lambda_gmd,
        plot_config=plot_config,
    )

    run_histology_analysis(
        df,
        output_dir=output_dir,
        bootstrap=args.bootstrap,
        seed=seed_value,
        lam_gmd=args.lambda_gmd,
        skip=args.skip_histology,
        plot_config=plot_config,
    )

    run_grade_analysis(
        df,
        output_dir=output_dir,
        bootstrap=args.bootstrap,
        seed=seed_value,
        lam_gmd=args.lambda_gmd,
        skip=args.skip_grade,
        plot_config=plot_config,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
