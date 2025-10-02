#!/usr/bin/env python
"""Extended analysis CLI replicating histology and grade results."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

from dtcygan.training import SyntheticSequenceDataset, set_seed
from dtcygan.eval_utils import (
    load_dataset_payload,
    collect_labels_from_patients,
    ensure_probability_metadata,
    load_checkpoint_bundle,
    build_counterfactual_patients,
    patients_to_dataframe,
)


DEFAULT_IMG_DIR = "imgs/analysis"
EPS = 1e-12

# ------------------------------------------------------------------ #
# Column names (edit here if your CSV uses different labels)
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
LESION_SITE_CANDIDATES = [
    "tumor_characteristics.lesion_site",
    "anatomical_site",
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
LESION_SITE_COL = LESION_SITE_CANDIDATES[0]
FNCLCC_GRADING_COL = FNCLCC_CANDIDATES[0]

EXTREMITY_SITE_SET: set[str] = {
    "femur",
    "tibia",
    "knee",
    "Acetabulum",
    "Axilla/scapula",
}


def _map_extremity_site(val: object) -> str:
    if pd.isna(val):
        return "Unknown"
    s = str(val)
    return "Extremity" if s in EXTREMITY_SITE_SET else "Non-extremity"


def as_prob(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.clip(0.0, 1.0)


def min_max_normalize(series: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
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
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    def _minmax_inplace(frame: pd.DataFrame, cols: List[str]) -> None:
        present = [c for c in cols if c in frame.columns]
        if not present:
            return
        for col in present:
            frame[col] = min_max_normalize(frame[col])

    _minmax_inplace(df, [LOCAL_RECURRENCE_COL, METASTATIC_COL, ENDPOINT_DOD_COL])
    _minmax_inplace(df, [STATUS_AWD_COL, DEAD_OF_DISEASE, STATUS_NED_COL])

    if LOCAL_RECURRENCE_COL in df.columns:
        df["LR_prob_norm_global"] = as_prob(df[LOCAL_RECURRENCE_COL])
    if METASTATIC_COL in df.columns:
        df["MET_prob_norm_global"] = as_prob(df[METASTATIC_COL])
    base_dod = pd.to_numeric(df.get(DEAD_OF_DISEASE, pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["DOD_prob_norm_global"] = as_prob(base_dod) if base_dod.notna().any() else np.nan

    if "LR_prob_norm_global" in df.columns:
        df["prob_norm"] = df["LR_prob_norm_global"]
    return df


def _resolve_out_path(out_path: str | Path, default_dir: str | Path = DEFAULT_IMG_DIR) -> Path:
    target = Path(out_path)
    if target.parent == Path("") or str(target.parent) in {"", "."}:
        target = Path(default_dir) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _gmd_norm(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n < 2:
        return 0.0
    w = (2 * np.arange(1, n + 1) - n - 1)
    gmd = (2.0 / (n * (n - 1))) * float(np.sum(w * x))
    return float(2.0 * gmd)


def _auc_trapz(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0 or y.size == 0:
        return float("nan")
    return float(np.trapz(y, t))


def _arm_curves(time_to_probs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    rng = np.random.default_rng(seed)

    sA = _arm_score(time_to_probs_A, lam_gmd)["risk_averse_score"]
    sB = _arm_score(time_to_probs_B, lam_gmd)["risk_averse_score"]

    def _percent_change(a: float, b: float) -> float:
        if not np.isfinite(a) or abs(a) < EPS:
            return float("nan")
        return float((b - a) / a * 100.0)

    delta = _percent_change(sA, sB)

    t = sorted(set(time_to_probs_A.keys()) & set(time_to_probs_B.keys()))
    A_arr = {tt: np.asarray(time_to_probs_A[tt], dtype=float) for tt in t}
    B_arr = {tt: np.asarray(time_to_probs_B[tt], dtype=float) for tt in t}
    n_by_t = {tt: min(A_arr[tt].size, B_arr[tt].size) for tt in t}

    def resample_once() -> float:
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


def _build_time_to_probs_from_df(df: pd.DataFrame, endpoint_col: str, arm_label: str) -> Dict[int, np.ndarray]:
    sub = df[df["group"] == arm_label]
    if sub.empty:
        return {}
    cols = [TIMESTEP_COL, endpoint_col]
    if "patient_id" in sub.columns:
        cols.append("patient_id")
    sub = sub[cols].copy()
    sub[endpoint_col] = pd.to_numeric(sub[endpoint_col], errors="coerce").astype(float).clip(0.0, 1.0)
    sub = sub.dropna(subset=[endpoint_col, TIMESTEP_COL])
    if sub.empty:
        return {}
    if "patient_id" in sub.columns:
        agg = sub.groupby([TIMESTEP_COL, "patient_id"], as_index=False)[endpoint_col].mean()
        grouped = agg.groupby(TIMESTEP_COL)
    else:
        grouped = sub.groupby(TIMESTEP_COL)
    return {int(t): g[endpoint_col].to_numpy(dtype=float) for t, g in grouped}


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
) -> Dict[str, Any]:
    df = df.copy()
    df["group"] = _make_group_label(df)
    if endpoint_col not in df.columns:
        fallback = {
            LOCAL_RECURRENCE_COL: "LR_prob_norm_global",
            METASTATIC_COL: "MET_prob_norm_global",
            DEAD_OF_DISEASE: "DOD_prob_norm_global",
        }.get(endpoint_col)
        if fallback and fallback in df.columns:
            endpoint_col = fallback
        else:
            raise ValueError(f"Missing endpoint column: {endpoint_col}")

    A = _build_time_to_probs_from_df(df, endpoint_col, arm_A)
    B = _build_time_to_probs_from_df(df, endpoint_col, arm_B)
    if not A or not B:
        raise ValueError("Selected arms or endpoint have no data.")
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
    rows: List[Dict[str, Any]] = []
    for ep in endpoints:
        if ep not in df.columns and ep not in (LOCAL_RECURRENCE_COL, METASTATIC_COL, DEAD_OF_DISEASE):
            continue
        for arm in arms:
            try:
                res = compare_two_arms_df(
                    df,
                    endpoint_col=ep,
                    arm_A=baseline,
                    arm_B=arm,
                    lam_gmd=lam_gmd,
                    n_boot=n_boot,
                    paired=paired,
                    seed=seed,
                )
                rows.append(
                    {
                        "endpoint": ep,
                        "baseline": baseline,
                        "arm": arm,
                        "percent_change": res["percent_change"],
                        "ci_low": float(res["CI95"][0]),
                        "ci_high": float(res["CI95"][1]),
                        "p_tilde": res["p~"],
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows)


def _cluster_bootstrap_stat(
    values: pd.Series,
    *,
    pids: Optional[pd.Series] = None,
    agg: Literal["mean", "median"] = "mean",
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> Tuple[float, Tuple[float, float]]:
    s = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if s.empty:
        return np.nan, (np.nan, np.nan)
    s = s.clip(0.0, 1.0)

    if agg == "mean":
        point = float(s.mean())
        stat_fn = np.mean
    else:
        point = float(s.median())
        stat_fn = np.median

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)

    if pids is None or pids.isna().all():
        m = s.shape[0]
        for b in range(n_boot):
            idx = rng.integers(0, m, size=m)
            boot[b] = float(stat_fn(s.iloc[idx])) if idx.size else np.nan
    else:
        p = pids.loc[s.index]
        codes, _ = pd.factorize(p, sort=False)
        clusters = [np.flatnonzero(codes == k) for k in range(len(np.unique(codes)))]
        n_clusters = len(clusters)
        if n_clusters == 0:
            return point, (np.nan, np.nan)
        for b in range(n_boot):
            draw = rng.integers(0, n_clusters, size=n_clusters)
            sel = np.concatenate([clusters[d] for d in draw], axis=0)
            boot[b] = float(stat_fn(s.iloc[sel])) if sel.size else np.nan

    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return point, (np.nan, np.nan)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return point, ci


def _make_group_label(df: pd.DataFrame) -> pd.Series:
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


def compare_groups_overall(
    df: pd.DataFrame,
    *,
    header: str = "Overall comparison",
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    surgery_only_mask = (
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 0)
        & (df[CHEMOTHERAPY_FLAG_COL] == 0)
    )
    surgery_rt_mask = (
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 1)
        & (df[CHEMOTHERAPY_FLAG_COL] == 0)
    )
    surgery_ct_mask = (
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 0)
        & (df[CHEMOTHERAPY_FLAG_COL] == 1)
    )
    surgery_rt_ct_mask = (
        (df[SURGERY_FLAG_COL] == 1)
        & (df[RADIOTHERAPY_FLAG_COL] == 1)
        & (df[CHEMOTHERAPY_FLAG_COL] == 1)
    )

    grp_only = df.loc[surgery_only_mask, "prob_norm"]
    grp_rt = df.loc[surgery_rt_mask, "prob_norm"]
    grp_ct = df.loc[surgery_ct_mask, "prob_norm"]
    grp_rt_ct = df.loc[surgery_rt_ct_mask, "prob_norm"]

    _print_summary_stats_all_groups(
        grp_only,
        grp_rt,
        grp_ct,
        grp_rt_ct,
        header=header,
        n_boot=n_boot,
        seed=seed,
        patient_ids=df.get("patient_id"),
    )

    return grp_only, grp_rt, grp_ct, grp_rt_ct


def compare_groups_by_timestep(
    df: pd.DataFrame,
    *,
    header: str = "Per-timestep comparison",
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    df = df.copy()
    df["group"] = _make_group_label(df)
    df = df[df["group"].isin(["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"])]

    results = []
    for ts, chunk in df.groupby(TIMESTEP_COL):
        grp_only = chunk.loc[chunk["group"] == "Surgery only", "prob_norm"]
        grp_rt = chunk.loc[chunk["group"] == "Surgery + RT", "prob_norm"]
        grp_ct = chunk.loc[chunk["group"] == "Surgery + CT", "prob_norm"]
        grp_rt_ct = chunk.loc[chunk["group"] == "Surgery + RT + CT", "prob_norm"]

        groups_with_data = [g for g in [grp_only, grp_rt, grp_ct, grp_rt_ct] if not g.empty]
        if len(groups_with_data) < 2:
            continue

        pvals: Dict[str, Optional[float]] = {"p_only_rt": None, "p_only_ct": None, "p_only_rt_ct": None, "p_rt_ct": None, "p_rt_rt_ct": None, "p_ct_rt_ct": None}
        if not grp_only.empty and not grp_rt.empty and (grp_only.nunique() > 1 or grp_rt.nunique() > 1):
            _, pvals["p_only_rt"] = mannwhitneyu(grp_only, grp_rt, alternative="two-sided")
        if not grp_only.empty and not grp_ct.empty and (grp_only.nunique() > 1 or grp_ct.nunique() > 1):
            _, pvals["p_only_ct"] = mannwhitneyu(grp_only, grp_ct, alternative="two-sided")
        if not grp_only.empty and not grp_rt_ct.empty and (grp_only.nunique() > 1 or grp_rt_ct.nunique() > 1):
            _, pvals["p_only_rt_ct"] = mannwhitneyu(grp_only, grp_rt_ct, alternative="two-sided")
        if not grp_rt.empty and not grp_ct.empty and (grp_rt.nunique() > 1 or grp_ct.nunique() > 1):
            _, pvals["p_rt_ct"] = mannwhitneyu(grp_rt, grp_ct, alternative="two-sided")
        if not grp_rt.empty and not grp_rt_ct.empty and (grp_rt.nunique() > 1 or grp_rt_ct.nunique() > 1):
            _, pvals["p_rt_rt_ct"] = mannwhitneyu(grp_rt, grp_rt_ct, alternative="two-sided")
        if not grp_ct.empty and not grp_rt_ct.empty and (grp_ct.nunique() > 1 or grp_rt_ct.nunique() > 1):
            _, pvals["p_ct_rt_ct"] = mannwhitneyu(grp_ct, grp_rt_ct, alternative="two-sided")

        pid_col = chunk["patient_id"] if "patient_id" in chunk.columns else None

        def _boot_ci(series: pd.Series) -> Tuple[float, float]:
            if pid_col is None or series.empty:
                return (np.nan, np.nan)
            _, ci = _cluster_bootstrap_stat(series, pids=pid_col.loc[series.index], agg="mean", n_boot=n_boot, seed=seed)
            return ci

        mean_only_ci = _boot_ci(grp_only)
        mean_rt_ci = _boot_ci(grp_rt)
        mean_ct_ci = _boot_ci(grp_ct)
        mean_rt_ct_ci = _boot_ci(grp_rt_ct)

        results.append(
            {
                "timestep": ts,
                "n_only": len(grp_only),
                "n_rt": len(grp_rt),
                "n_ct": len(grp_ct),
                "n_rt_ct": len(grp_rt_ct),
                "mean_only": grp_only.mean() if not grp_only.empty else np.nan,
                "mean_rt": grp_rt.mean() if not grp_rt.empty else np.nan,
                "mean_ct": grp_ct.mean() if not grp_ct.empty else np.nan,
                "mean_rt_ct": grp_rt_ct.mean() if not grp_rt_ct.empty else np.nan,
                "mean_only_ci_low": mean_only_ci[0],
                "mean_only_ci_high": mean_only_ci[1],
                "mean_rt_ci_low": mean_rt_ci[0],
                "mean_rt_ci_high": mean_rt_ci[1],
                "mean_ct_ci_low": mean_ct_ci[0],
                "mean_ct_ci_high": mean_ct_ci[1],
                "mean_rt_ct_ci_low": mean_rt_ct_ci[0],
                "mean_rt_ct_ci_high": mean_rt_ct_ci[1],
                "std_only": grp_only.std() if not grp_only.empty else np.nan,
                "std_rt": grp_rt.std() if not grp_rt.empty else np.nan,
                "std_ct": grp_ct.std() if not grp_ct.empty else np.nan,
                "std_rt_ct": grp_rt_ct.std() if not grp_rt_ct.empty else np.nan,
                "median_only": grp_only.median() if not grp_only.empty else np.nan,
                "median_rt": grp_rt.median() if not grp_rt.empty else np.nan,
                "median_ct": grp_ct.median() if not grp_ct.empty else np.nan,
                "median_rt_ct": grp_rt_ct.median() if not grp_rt_ct.empty else np.nan,
                **pvals,
            }
        )

    res_df = pd.DataFrame(results).sort_values("timestep").reset_index(drop=True)

    print(f"\n=== {header} ===")
    if res_df.empty:
        print("No timesteps have treatment groups with sufficient data.")
    else:
        print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return res_df


def compute_individualized_treatment_effects(
    df: pd.DataFrame,
    *,
    reference_group: str = "Surgery only",
    patient_col: str = "patient_id",
    time_col: str = TIMESTEP_COL,
    endpoint_cols: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_boot: int = 0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    df = df.copy()
    df["group"] = _make_group_label(df)

    groups = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    df = df[df["group"].isin(groups)]

    if endpoint_cols is None:
        endpoint_cols = {
            "local_recurrence": LOCAL_RECURRENCE_COL,
            "metastasis": METASTATIC_COL,
            "death_of_disease": DEAD_OF_DISEASE,
        }

    results: List[pd.DataFrame] = []

    for endpoint, col in endpoint_cols.items():
        if col not in df.columns:
            continue
        pivot = df.pivot_table(index=[patient_col, time_col], columns="group", values=col)
        if reference_group not in pivot.columns:
            continue
        ref_vals = pivot[reference_group]
        for treatment in groups:
            if treatment == reference_group or treatment not in pivot.columns:
                continue
            delta = pivot[treatment] - ref_vals
            delta = delta.dropna().reset_index(name="delta")

            def _integrate(group: pd.DataFrame) -> float:
                times = group[time_col].values
                vals = group["delta"].values
                if len(vals) == 1:
                    return float(vals[0])
                denom = times[-1] - times[0]
                if denom == 0:
                    return float(vals.mean())
                return float(np.trapz(vals, x=times) / denom)

            per_patient = delta.groupby(patient_col).apply(_integrate).reset_index(name="ite")
            per_patient["treatment"] = treatment
            per_patient["endpoint"] = endpoint
            results.append(per_patient)

    if not results:
        return pd.DataFrame(columns=[patient_col, "treatment", "endpoint", "ite"])

    result = pd.concat(results, ignore_index=True)

    if weights:
        weight_series = pd.Series(weights, dtype=float)
        composite = (
            result.pivot_table(index=[patient_col, "treatment"], columns="endpoint", values="ite")
            .reindex(columns=weight_series.index)
            .mul(weight_series, axis=1)
            .sum(axis=1)
            .reset_index(name="composite_ite")
        )
        result = result.merge(composite, on=[patient_col, "treatment"], how="left")

    if weights and n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        weight_series = pd.Series(weights, dtype=float)

        dwork = df.copy()
        dwork["group"] = _make_group_label(dwork)

        comp_ci_rows: List[Dict[str, Any]] = []
        groups = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]

        ep_cols = {
            k: endpoint_cols[k] if endpoint_cols and k in endpoint_cols else v
            for k, v in {
                "local_recurrence": LOCAL_RECURRENCE_COL,
                "metastasis": METASTATIC_COL,
                "death_of_disease": DEAD_OF_DISEASE,
            }.items()
            if (endpoint_cols or True)
        }

        for (pid, treat), _ in (
            result.dropna(subset=["composite_ite"]).drop_duplicates([patient_col, "treatment"]).groupby([patient_col, "treatment"])
        ):
            if treat == reference_group:
                continue
            ref = dwork[(dwork[patient_col] == pid) & (dwork["group"] == reference_group)]
            trt = dwork[(dwork[patient_col] == pid) & (dwork["group"] == treat)]
            if ref.empty or trt.empty:
                continue
            comp_df: Optional[pd.DataFrame] = None
            for ep_name, col in ep_cols.items():
                if col not in dwork.columns or ep_name not in weight_series.index:
                    continue
                r = ref[[time_col, col]].rename(columns={col: "ref"})
                t = trt[[time_col, col]].rename(columns={col: "trt"})
                m = pd.merge(r, t, on=time_col, how="inner")
                if m.empty:
                    continue
                m[ep_name] = (m["trt"].astype(float) - m["ref"].astype(float)) * float(weight_series[ep_name])
                m = m[[time_col, ep_name]]
                comp_df = m if comp_df is None else pd.merge(comp_df, m, on=time_col, how="inner")
            if comp_df is None or comp_df.empty:
                continue
            comp_df = comp_df.sort_values(time_col)
            comp_df["comp"] = comp_df.drop(columns=[time_col]).sum(axis=1)
            times = comp_df[time_col].to_numpy()
            vals = comp_df["comp"].to_numpy()
            if len(vals) < 1:
                continue
            point = float(vals[0]) if len(vals) == 1 else float(np.trapz(vals, x=times) / (times[-1] - times[0])) if times[-1] != times[0] else float(vals.mean())
            boots = []
            if len(vals) >= 2:
                m = len(vals)
                for _ in range(n_boot):
                    idx = rng.integers(0, m, size=m)
                    tb = times[idx]
                    vb = vals[idx]
                    order = np.argsort(tb)
                    tb = tb[order]
                    vb = vb[order]
                    area = float(np.trapz(vb, x=tb) / (tb[-1] - tb[0])) if tb[-1] != tb[0] else float(vb.mean())
                    boots.append(area)
            if boots:
                low, high = np.percentile(boots, [2.5, 97.5])
            else:
                low = high = np.nan
            comp_ci_rows.append(
                {
                    patient_col: pid,
                    "treatment": treat,
                    "composite_ci_low": low,
                    "composite_ci_high": high,
                }
            )

        if comp_ci_rows:
            comp_ci_df = pd.DataFrame(comp_ci_rows)
            result = result.merge(
                comp_ci_df[[patient_col, "treatment", "composite_ci_low", "composite_ci_high"]],
                on=[patient_col, "treatment"],
                how="left",
            )

    return result


def plot_composite_benefit_waterfall(
    ite_df: pd.DataFrame,
    *,
    treatment: str,
    histology: Optional[str] = None,
    out_path: str = "composite_benefit_waterfall.png",
    patient_col: str = "patient_id",
) -> None:
    data = ite_df[ite_df["treatment"] == treatment]
    if "composite_ite" not in data.columns or data.empty:
        print("plot_composite_benefit_waterfall: no composite_ite values to plot.")
        return

    keep_cols = [c for c in [patient_col, "composite_ite", "composite_ci_low", "composite_ci_high"] if c in data.columns]
    data = data[keep_cols].drop_duplicates(subset=[patient_col]).sort_values("composite_ite", ascending=False)

    colors = data["composite_ite"].apply(lambda x: "#2ca02c" if x > 0 else "#d62728")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(data))
    ax.bar(x, data["composite_ite"], color=colors, edgecolor="black")
    if "composite_ci_low" in data.columns and "composite_ci_high" in data.columns:
        y = data["composite_ite"].to_numpy(dtype=float)
        yerr_low = np.clip(y - data["composite_ci_low"].to_numpy(dtype=float), 0, None)
        yerr_high = np.clip(data["composite_ci_high"].to_numpy(dtype=float) - y, 0, None)
        ax.errorbar(x, y, yerr=np.vstack([yerr_low, yerr_high]), fmt="none", ecolor="black", elinewidth=0.8, capsize=2, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Patients (sorted)")
    ax.set_ylabel("Composite benefit")
    title = f"Composite benefit: {treatment}"
    if histology:
        title += f" – {histology}"
    ax.set_title(title)

    n_pos = (data["composite_ite"] > 0).sum()
    n_total = len(data)
    frac = n_pos / n_total if n_total else 0.0
    ax.text(0.98, 0.95, f"{n_pos}/{n_total} ({frac:.0%}) above zero", transform=ax.transAxes, ha="right", va="top", fontsize=10)

    plt.tight_layout()
    resolved = _resolve_out_path(out_path)
    plt.savefig(resolved, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nComposite benefit waterfall saved to {resolved}")


def plot_composite_benefit_fan_envelope(
    ite_df: pd.DataFrame,
    *,
    treatment: str,
    histology: Optional[str] = None,
    out_path: str = "composite_benefit_fan_envelope.png",
    n_boot: int = 200,
    seed: Optional[int] = 42,
    patient_col: str = "patient_id",
    sort_desc: bool = False,
) -> None:
    data = ite_df[ite_df["treatment"] == treatment]
    if "composite_ite" not in data.columns or data.empty:
        print("plot_composite_benefit_fan_envelope: no composite_ite values to plot.")
        return

    vals = (
        data[[patient_col, "composite_ite"]].drop_duplicates(subset=[patient_col])["composite_ite"].astype(float).dropna().to_numpy()
    )
    N = vals.size
    if N == 0:
        print("plot_composite_benefit_fan_envelope: empty after filtering.")
        return

    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, N), dtype=float)
    for b in range(n_boot):
        smp = rng.choice(vals, size=N, replace=True)
        smp.sort()
        if sort_desc:
            smp = smp[::-1]
        reps[b, :] = smp

    q2p5 = np.percentile(reps, 2.5, axis=0)
    q50 = np.percentile(reps, 50, axis=0)
    q97p5 = np.percentile(reps, 97.5, axis=0)

    ranks = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    ax.fill_between(ranks, q2p5, q97p5, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.plot(ranks, q50, color="#1f77b4", lw=2)
    ax.axhline(0, color="black", lw=1, ls=":")
    ax.set_xlim(1, N)
    ax.set_xlabel("Rank (1 = best, N = worst)")
    ax.set_ylabel("Composite ITE (ordered)")
    title = f"Rank-consistent envelope: {treatment}"
    if histology:
        title += f" – {histology}"
    ax.set_title(title)
    n_benef = int(np.sum(vals < 0))
    ax.text(0.98, 0.95, f"{n_benef}/{N} ({n_benef / N:.0%}) beneficial", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    resolved = _resolve_out_path(out_path)
    fig.savefig(resolved, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nComposite benefit fan envelope (95% CI) saved to {resolved}")


def _rank_consistent_envelope(
    values: np.ndarray,
    *,
    n_boot: int = 200,
    seed: Optional[int] = 42,
    sort_desc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    N = vals.size
    if N == 0:
        return np.array([]), np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, N), dtype=float)
    for b in range(n_boot):
        smp = rng.choice(vals, size=N, replace=True)
        smp.sort()
        if sort_desc:
            smp = smp[::-1]
        reps[b, :] = smp
    q2p5 = np.percentile(reps, 2.5, axis=0)
    q50 = np.percentile(reps, 50, axis=0)
    q97p5 = np.percentile(reps, 97.5, axis=0)
    return q2p5, q50, q97p5


def plot_endpoint_waterfall_fan_grid(
    ite_df: pd.DataFrame,
    *,
    treatments: Optional[List[str]] = None,
    endpoints: Optional[List[str]] = None,
    histology: Optional[str] = None,
    out_path: str = "waterfall_fan_grid.png",
    n_boot: int = 200,
    seed: Optional[int] = 42,
    patient_col: str = "patient_id",
    sort_desc: bool = False,
) -> None:
    if endpoints is None:
        endpoints = ["local_recurrence", "metastasis", "death_of_disease"]
    if treatments is None:
        treatments = ["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]

    t_label = {
        "Surgery + RT": "S+RT",
        "Surgery + CT": "S+CT",
        "Surgery + RT + CT": "S+RT+CT",
    }
    e_label = {
        "local_recurrence": "Local Recurrence",
        "metastasis": "Metastasis",
        "death_of_disease": "DOD",
    }

    base = ite_df[[patient_col, "treatment", "endpoint", "ite"]].copy()
    if base.empty:
        print("plot_endpoint_waterfall_fan_grid: empty ite_df")
        return

    n_rows = len(endpoints)
    n_cols = len(treatments)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows), sharex=False, sharey=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    ylims: Dict[str, Tuple[float, float]] = {}
    for ep in endpoints:
        vals_ep = base.loc[base["endpoint"] == ep, "ite"].astype(float)
        if vals_ep.dropna().empty:
            ylims[ep] = (-0.01, 0.01)
        else:
            v = vals_ep.to_numpy()
            lo, hi = np.nanpercentile(v, [1, 99])
            rng_val = max(abs(lo), abs(hi))
            ylims[ep] = (-rng_val, rng_val)

    for r, ep in enumerate(endpoints):
        for c, tr in enumerate(treatments):
            ax = axes[r, c]
            vals = (
                base[(base["endpoint"] == ep) & (base["treatment"] == tr)][[patient_col, "ite"]]
                .drop_duplicates(subset=[patient_col])["ite"].astype(float).dropna().to_numpy()
            )
            if vals.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            q5, q50, q95 = _rank_consistent_envelope(vals, n_boot=n_boot, seed=seed, sort_desc=sort_desc)
            N = q50.size
            ranks = np.arange(1, N + 1)
            ax.fill_between(ranks, q5, q95, color="#1f77b4", alpha=0.2, linewidth=0)
            ax.plot(ranks, q50, color="#1f77b4", lw=1.8)
            ax.axhline(0, color="black", lw=0.8, ls=":")
            ax.set_xlim(1, N)
            ax.set_ylim(*ylims[ep])
            if r == n_rows - 1:
                ax.set_xlabel("Rank")
            if c == 0:
                ax.set_ylabel(e_label.get(ep, ep))
            ax.set_title(t_label.get(tr, tr), fontsize=10)
            n_benef = int(np.sum(vals < 0))
            ax.text(0.98, 0.92, f"{n_benef}/{vals.size} ({(n_benef/vals.size):.0%})", transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("" if histology is None else str(histology), y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    resolved = _resolve_out_path(out_path)
    fig.savefig(resolved, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"waterfall fan grid saved to {resolved}")


def plot_boxplot(
    grp_only: pd.Series,
    grp_rt: pd.Series,
    grp_ct: pd.Series,
    grp_rt_ct: pd.Series,
    *,
    out_path: str = "local_recurrence_boxplot.png",
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "figure.titlesize": 16,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.3,
        }
    )

    plt.figure(figsize=(8, 6))

    groups: List[pd.Series] = []
    labels: List[str] = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    group_colors: List[str] = []

    if not grp_only.empty:
        groups.append(grp_only)
        labels.append("Surgery\nonly")
        group_colors.append(colors[0])
    if not grp_rt.empty:
        groups.append(grp_rt)
        labels.append("Surgery\n+ RT")
        group_colors.append(colors[1])
    if not grp_ct.empty:
        groups.append(grp_ct)
        labels.append("Surgery\n+ CT")
        group_colors.append(colors[2])
    if not grp_rt_ct.empty:
        groups.append(grp_rt_ct)
        labels.append("Surgery\n+ RT + CT")
        group_colors.append(colors[3])

    if groups:
        bp = plt.boxplot(
            groups,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops={"color": "white", "linewidth": 1.5},
            meanprops={"color": "black", "linewidth": 1.5, "linestyle": "--"},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        )

        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)

        plt.ylabel("Normalized Local Recurrence Probability", fontweight="bold")
        plt.title("Local Recurrence Probability by Treatment Modality", fontweight="bold", pad=20)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        plt.gca().set_axisbelow(True)
        plt.ylim(-0.05, 1.05)

        plt.tight_layout()
        resolved = _resolve_out_path(out_path)
        plt.savefig(resolved, dpi=300, bbox_inches="tight", facecolor="white")

        print(f"\nBox-plot saved to {resolved}")
    else:
        print("No data available for box plot")

    plt.close("all")


def _bootstrap_timecourse_by_group(
    df: pd.DataFrame,
    *,
    value_col: str,
    time_col: str = TIMESTEP_COL,
    patient_col: str = "patient_id",
    n_boot: int = 200,
    seed: Optional[int] = 42,
    agg: Literal["median", "mean"] = "median",
    band_method: Literal["pointwise", "rcqe"] = "rcqe",
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
    work = df[[time_col, value_col] + ([patient_col] if patient_col in df.columns else [])].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").clip(0.0, 1.0)
    work = work.dropna(subset=[value_col, time_col])
    if work.empty:
        return np.array([]), np.array([]), (np.array([]), np.array([])), np.array([])

    times = np.array(sorted(work[time_col].dropna().unique()))
    agg_fn = np.nanmedian if agg == "median" else np.nanmean

    counts = []
    if patient_col in work.columns:
        for t in times:
            counts.append(int(work.loc[work[time_col] == t, patient_col].dropna().nunique()))
    else:
        for t in times:
            counts.append(int(work.loc[work[time_col] == t, value_col].notna().sum()))
    counts = np.array(counts, dtype=int)

    rng = np.random.default_rng(seed)

    use_cluster = patient_col in work.columns and work[patient_col].notna().any()
    if use_cluster:
        codes, uniques = pd.factorize(work[patient_col], sort=False)
        clusters = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(clusters)
        if n_clusters == 0:
            use_cluster = False

    boots = np.empty((n_boot, times.size), dtype=float)
    for b in range(n_boot):
        if use_cluster:
            draw = rng.integers(0, len(clusters), size=len(clusters))
            sel_idx = np.concatenate([clusters[d] for d in draw], axis=0)
            sample = work.iloc[sel_idx]
        else:
            m = work.shape[0]
            sel_idx = rng.integers(0, m, size=m)
            sample = work.iloc[sel_idx]

        for i, t in enumerate(times):
            vals = pd.to_numeric(sample.loc[sample[time_col] == t, value_col], errors="coerce").to_numpy()
            vals = vals[(~np.isnan(vals)) & (vals >= 0.0) & (vals <= 1.0)]
            boots[b, i] = agg_fn(vals) if vals.size else np.nan

    if band_method == "pointwise":
        low = np.nanpercentile(boots, 2.5, axis=0)
        high = np.nanpercentile(boots, 97.5, axis=0)
    else:
        curve_score = np.nanmean(boots, axis=1)
        valid = np.isfinite(curve_score)
        if not np.any(valid):
            low = np.nanpercentile(boots, 2.5, axis=0)
            high = np.nanpercentile(boots, 97.5, axis=0)
        else:
            idx = np.argsort(curve_score[valid])
            boots_valid = boots[valid]
            n = boots_valid.shape[0]
            q = 0.025
            low_i = int(np.floor(q * (n - 1)))
            high_i = int(np.ceil((1.0 - q) * (n - 1)))
            low = boots_valid[idx[low_i], :]
            high = boots_valid[idx[high_i], :]
    center = np.nanmedian(boots, axis=0) if agg == "median" else np.nanmean(boots, axis=0)

    return times, center, (low, high), counts


def plot_three_endpoint_trajectories_with_bands(
    df: pd.DataFrame,
    *,
    histology: Optional[str] = None,
    out_path: str = "three_endpoint_trajectories_bands.png",
    n_boot: int = 200,
    seed: Optional[int] = 42,
    agg: Literal["median", "mean"] = "median",
    band_method: Literal["pointwise", "rcqe"] = "rcqe",
    use_color: bool = True,
    layout: Literal["rows", "cols"] = "rows",
    show_counts: bool = True,
    show_errorbars: bool = True,
    table_outside: bool = True,
    table_mode: Literal["counts", "summary", "both"] = "summary",
    side_titles: bool = True,
    annotate_arm_dispersion: bool = True,
    row_height: float = 5.0,
    panel_hspace: float = 0.85,
    table_gap: float = 0.10,
    legend_y: float = 1.4,
    tight_top: float = 0.92,
) -> None:
    if df.empty:
        print("plot_three_endpoint_trajectories_with_bands: Empty DataFrame; nothing to plot.")
        return

    work = df.copy()
    if "LR_prob_norm_global" not in work.columns and LOCAL_RECURRENCE_COL in work.columns:
        work["LR_prob_norm_global"] = as_prob(work[LOCAL_RECURRENCE_COL])
    if "MET_prob_norm_global" not in work.columns and METASTATIC_COL in work.columns:
        work["MET_prob_norm_global"] = as_prob(work[METASTATIC_COL])
    if "DOD_prob_norm_global" not in work.columns and DEAD_OF_DISEASE in work.columns:
        work["DOD_prob_norm_global"] = as_prob(work[DEAD_OF_DISEASE])

    if histology is not None and HISTOLOGICAL_DIAGNOSIS_COL in work.columns:
        work = work[work[HISTOLOGICAL_DIAGNOSIS_COL] == histology]

    work["group"] = _make_group_label(work)
    arms = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(arms)]
    if work.empty:
        print("plot_three_endpoint_trajectories_with_bands: No rows for the specified groups/histology.")
        return

    endpoints = [
        ("Local Recurrence", "LR_prob_norm_global" if "LR_prob_norm_global" in work.columns else LOCAL_RECURRENCE_COL),
        ("Metastasis", "MET_prob_norm_global" if "MET_prob_norm_global" in work.columns else METASTATIC_COL),
        ("DOD", "DOD_prob_norm_global" if "DOD_prob_norm_global" in work.columns else DEAD_OF_DISEASE),
    ]

    color_map = {
        "Surgery only": "#1f77b4",
        "Surgery + RT": "#ff7f0e",
        "Surgery + CT": "#2ca02c",
        "Surgery + RT + CT": "#d62728",
    }
    gray_line_map = {
        "Surgery only": "#111111",
        "Surgery + RT": "#333333",
        "Surgery + CT": "#555555",
        "Surgery + RT + CT": "#000000",
    }
    linestyles = {
        "Surgery only": "-",
        "Surgery + RT": "--",
        "Surgery + CT": "-.",
        "Surgery + RT + CT": ":",
    }

    all_times = sorted(work[TIMESTEP_COL].dropna().unique()) if TIMESTEP_COL in work.columns else []
    n_steps = len(all_times)
    if layout == "rows":
        fig_width = max(18.0, min(32.0, 0.7 * max(1, n_steps)))
        base_row_h = float(row_height)
        extra_bottom = 1.6 if (show_counts and table_outside and (table_mode in ("summary", "both"))) else (1.2 if (show_counts and table_outside) else 0.5)
        extra_top = 1.8
        fig_height = base_row_h * 3 + extra_bottom + extra_top
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    else:
        fig_width = max(16.0, min(24.0, 4.5 + 0.35 * max(1, n_steps)))
        fig_height = 5.0
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.25,
        }
    )

    any_drawn = False
    for ax_idx, (ax, (ep_name, ep_col)) in enumerate(zip(np.ravel(axes), endpoints)):
        if ep_col not in work.columns:
            ax.set_visible(False)
            continue

        trajectories: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        counts_map: Dict[str, np.ndarray] = {}
        times_ref: Optional[np.ndarray] = None
        for arm in arms:
            sub = work[work["group"] == arm]
            if sub.empty or sub[ep_col].dropna().empty:
                continue
            times, center, (low, high), counts = _bootstrap_timecourse_by_group(
                sub,
                value_col=ep_col,
                time_col=TIMESTEP_COL,
                patient_col="patient_id",
                n_boot=n_boot,
                seed=seed,
                agg=agg,
                band_method=band_method,
            )
            if times.size == 0:
                continue
            if times_ref is None:
                times_ref = times
            else:
                common = np.intersect1d(times_ref, times)
                if common.size == 0:
                    continue
                idx_ref = np.nonzero(np.isin(times_ref, common))[0]
                idx_t = np.nonzero(np.isin(times, common))[0]
                times_ref = common
                for key in list(trajectories.keys()):
                    c_prev, l_prev, h_prev = trajectories[key]
                    trajectories[key] = (c_prev[idx_ref], l_prev[idx_ref], h_prev[idx_ref])
                    counts_map[key] = counts_map[key][idx_ref]
                center, low, high, counts = center[idx_t], low[idx_t], high[idx_t], counts[idx_t]

            trajectories[arm] = (center, low, high)
            counts_map[arm] = counts

        if times_ref is None:
            ax.set_visible(False)
            continue

        for arm in arms:
            if arm not in trajectories:
                continue
            center, low, high = trajectories[arm]
            base_c = color_map[arm] if use_color else gray_line_map[arm]
            ax.fill_between(times_ref, low, high, color=base_c, alpha=0.18, linewidth=0)

        step = max(1, int(len(times_ref) / 10))
        markers = {"Surgery only": "o", "Surgery + RT": "s", "Surgery + CT": "^", "Surgery + RT + CT": "D"}
        for arm in arms:
            if arm not in trajectories:
                continue
            center, low, high = trajectories[arm]
            line_c = color_map[arm] if use_color else gray_line_map[arm]
            ax.plot(times_ref, center, linestyle=linestyles[arm], color=line_c, linewidth=2.2, label=arm, zorder=3)
            ax.plot(times_ref[::step], center[::step], linestyle="None", marker=markers[arm], color=line_c, markersize=4, alpha=0.8, zorder=4)
            if show_errorbars:
                yerr_low = np.maximum(0, center[::step] - low[::step])
                yerr_high = np.maximum(0, high[::step] - center[::step])
                ax.errorbar(times_ref[::step], center[::step], yerr=[yerr_low, yerr_high], fmt="none", ecolor=line_c, elinewidth=1.0, capsize=2, alpha=0.9, zorder=5)

        if side_titles:
            ax.text(-0.05, 0.5, ep_name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=12)
        else:
            ax.set_title(ep_name, pad=10)

        if layout == "rows":
            ax.xaxis.tick_top()
            if times_ref is not None and len(times_ref) > 0:
                x_labels = [str(i + 1) for i in range(len(times_ref))]
                ax.set_xticks(times_ref)
                ax.set_xticklabels(x_labels)
                ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False)
            if ax_idx == 0:
                ax.xaxis.set_label_position("top")
                ax.set_xlabel("Time")
                ax.xaxis.labelpad = 2
            else:
                ax.set_xlabel("")

        ax.grid(True, alpha=0.25, linestyle="-")
        ax.set_ylim(0.0, 1.0)
        any_drawn = True

        if annotate_arm_dispersion and TIMESTEP_COL in work.columns:
            nbins = 20
            bins = np.linspace(0.0, 1.0, nbins + 1)
            I_vals: List[float] = []
            dG_vals: List[float] = []
            for t in times_ref:
                values_by_arm: Dict[str, np.ndarray] = {}
                ns: Dict[str, int] = {}
                total_n = 0
                for arm in arms:
                    s = pd.to_numeric(work.loc[(work["group"] == arm) & (work[TIMESTEP_COL] == t), ep_col], errors="coerce").dropna()
                    s = s[(s >= 0.0) & (s <= 1.0)]
                    if s.size >= 5:
                        arr = s.to_numpy()
                        values_by_arm[arm] = arr
                        ns[arm] = int(arr.size)
                        total_n += int(arr.size)
                if len(values_by_arm) < 2 or total_n < 10:
                    I_vals.append(np.nan)
                    dG_vals.append(np.nan)
                    continue
                eps = 1e-12
                P: Dict[str, np.ndarray] = {}
                for arm, arr in values_by_arm.items():
                    cnt, _ = np.histogram(arr, bins=bins)
                    cnt = cnt.astype(float) + eps
                    P[arm] = cnt / cnt.sum()
                weights = {arm: ns[arm] / total_n for arm in values_by_arm}
                p_mix = np.zeros(nbins, dtype=float)
                for arm, p in P.items():
                    p_mix += weights[arm] * p
                H_mix = -float(np.sum(p_mix * np.log(p_mix)))
                H_within = 0.0
                for arm, p in P.items():
                    H_within += weights[arm] * (-float(np.sum(p * np.log(p))))
                I = H_mix - H_within
                I_norm = I / np.log(nbins)
                def gini(p: np.ndarray) -> float:
                    return 1.0 - float(np.sum(p ** 2))
                G_mix = gini(p_mix)
                G_within = 0.0
                for arm, p in P.items():
                    G_within += weights[arm] * gini(p)
                denom = 1.0 - 1.0 / nbins
                dG_norm = (G_mix - G_within) / denom
                I_vals.append(float(I_norm))
                dG_vals.append(float(dG_norm))

            I_vals = np.asarray(I_vals, dtype=float)
            dG_vals = np.asarray(dG_vals, dtype=float)
            valid = np.where(np.isfinite(I_vals) & np.isfinite(dG_vals))[0]
            if valid.size >= 1:
                i0, i1 = int(valid[0]), int(valid[-1])
                txt = f"Separation H {I_vals[i0]:.2f}→{I_vals[i1]:.2f}\nΔG {dG_vals[i0]:.2f}→{dG_vals[i1]:.2f}"
                ax.text(
                    0.02,
                    0.98,
                    txt,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65, edgecolor="none"),
                    zorder=6,
                )

        if show_counts:
            shown_arms = [arm for arm in arms if arm in counts_map]
            if shown_arms:
                col_labels = []
                for i, _ in enumerate(times_ref):
                    if len(times_ref) <= 16 or i % max(1, len(times_ref) // 16) == 0:
                        col_labels.append(str(i + 1))
                    else:
                        col_labels.append("")
                row_map = {
                    "Surgery only": "S",
                    "Surgery + RT": "S+RT",
                    "Surgery + CT": "S+CT",
                    "Surgery + RT + CT": "S+RT+CT",
                }
                disp_rows = [row_map[a] for a in shown_arms]
                if table_mode == "counts":
                    cell_text = [[str(int(n)) for n in counts_map[a]] for a in shown_arms]
                elif table_mode in ("summary", "both"):
                    fmt = lambda c, l, h: "–" if (np.isnan(c) or np.isnan(l) or np.isnan(h)) else f"{c:.2f} [{l:.2f}–{h:.2f}]"
                    summary_rows = []
                    for a in shown_arms:
                        c_arr, l_arr, h_arr = trajectories[a]
                        summary_rows.append([fmt(c_arr[i], l_arr[i], h_arr[i]) for i in range(len(times_ref))])
                    if table_mode == "summary":
                        cell_text = summary_rows
                    else:
                        cell_text = [[f"n={int(counts_map[a][i])}\n{summary_rows[j][i]}" for i in range(len(times_ref))] for j, a in enumerate(shown_arms)]

                if table_outside:
                    bbox_height = 0.48 if table_mode in ("summary", "both") else 0.34
                    tbl = ax.table(
                        cellText=cell_text,
                        rowLabels=disp_rows,
                        colLabels=col_labels,
                        cellLoc="center",
                        bbox=[0.0, -bbox_height - float(table_gap), 1.0, bbox_height],
                    )
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(9)
                else:
                    tbl = ax.table(cellText=cell_text, rowLabels=disp_rows, colLabels=col_labels, loc="bottom", cellLoc="center")
                    tbl.scale(1.0, 0.45)
                    ax.set_xlabel("Time (t)")
                    pos = ax.get_position()
                    ax.set_position([pos.x0, pos.y0 + 0.12, pos.width, pos.height - 0.12])

    if any_drawn:
        legend_colors = color_map if use_color else gray_line_map
        label_map = {
            "Surgery only": "S",
            "Surgery + RT": "S+RT",
            "Surgery + CT": "S+CT",
            "Surgery + RT + CT": "S+RT+CT",
        }
        order = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
        legend_handles = [
            Line2D([0], [0], color=legend_colors[a], linestyle=linestyles[a], linewidth=2.2, label=label_map[a])
            for a in order
        ]
        if layout == "rows":
            axes[0].legend(
                legend_handles,
                [h.get_label() for h in legend_handles],
                loc="upper center",
                ncol=4,
                frameon=True,
                bbox_to_anchor=(0.5, float(legend_y)),
                fancybox=True,
                shadow=False,
                fontsize=12,
            )
            need_extra = show_counts and table_outside
            has_summary = table_mode in ("summary", "both")
            bottom_margin = 0.14 if (need_extra and has_summary) else (0.10 if need_extra else (0.08 if show_counts else 0.05))
            left_margin = 0.08 if side_titles else 0.05
            plt.tight_layout(rect=[left_margin, bottom_margin, 1, float(tight_top)])
            plt.subplots_adjust(hspace=float(panel_hspace))
        else:
            mid_ax = np.ravel(axes)[1] if len(np.ravel(axes)) >= 2 else np.ravel(axes)[0]
            mid_ax.legend(
                legend_handles,
                [h.get_label() for h in legend_handles],
                loc="upper center",
                ncol=4,
                frameon=True,
                bbox_to_anchor=(0.5, 1.70),
                fancybox=True,
                shadow=False,
                fontsize=11,
            )
            need_extra = show_counts and table_outside
            has_summary = table_mode in ("summary", "both")
            bottom_margin = 0.36 if (need_extra and has_summary) else (0.26 if need_extra else (0.14 if show_counts else 0.06))
            left_margin = 0.14 if side_titles else 0.06
            plt.tight_layout(rect=[left_margin, bottom_margin, 1, 0.68])
            plt.subplots_adjust(wspace=0.25)
        resolved = _resolve_out_path(out_path)
        fig.savefig(resolved, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"three-endpoint trajectories with bands saved to {resolved}")
    else:
        plt.close(fig)
        print("plot_three_endpoint_trajectories_with_bands: nothing drawn (missing columns/data).")


def _print_summary_stats_all_groups(
    grp_only: pd.Series,
    grp_rt: pd.Series,
    grp_ct: pd.Series,
    grp_rt_ct: pd.Series,
    *,
    header: str = "",
    n_boot: int = 200,
    seed: Optional[int] = 42,
    patient_ids: Optional[pd.Series] = None,
) -> None:
    if header:
        print(f"\n=== {header} ===")

    groups_info = [
        (grp_only, "Surgery only"),
        (grp_rt, "Surgery + RT"),
        (grp_ct, "Surgery + CT"),
        (grp_rt_ct, "Surgery + RT + CT"),
    ]

    valid_groups = [(grp, name) for grp, name in groups_info if not grp.empty]

    if not valid_groups:
        print("No valid groups for comparison.")
        return

    summary_data = []
    for grp, name in valid_groups:
        summary_data.append(
            {
                "group": name,
                "n": len(grp),
                "mean": grp.mean(),
                "std": grp.std(),
                "median": grp.median(),
                "IQR": grp.quantile(0.75) - grp.quantile(0.25),
            }
        )

    summary = pd.DataFrame(summary_data)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if patient_ids is not None:
        pid = patient_ids
        for grp, name in valid_groups:
            if grp.empty:
                continue
            mean_pt, mean_ci = _cluster_bootstrap_stat(grp, pids=pid.loc[grp.index], agg="mean", n_boot=n_boot, seed=seed)
            med_pt, med_ci = _cluster_bootstrap_stat(grp, pids=pid.loc[grp.index], agg="median", n_boot=n_boot, seed=seed)
            print(
                f"  {name}: mean={mean_pt:.4f} [95% CI {mean_ci[0]:.4f}, {mean_ci[1]:.4f}]  "
                f"median={med_pt:.4f} [95% CI {med_ci[0]:.4f}, {med_ci[1]:.4f}]"
            )

    print("\nPairwise Mann-Whitney U tests:")
    for i in range(len(valid_groups)):
        for j in range(i + 1, len(valid_groups)):
            grp1, name1 = valid_groups[i]
            grp2, name2 = valid_groups[j]
            u_stat, p_val = mannwhitneyu(grp1, grp2, alternative="two-sided")
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {name1} vs {name2}: U = {u_stat:.2f}, p = {p_val:.6f} {significance}")


def analyze_diagnosis(
    df: pd.DataFrame,
    *,
    diagnosis_name: str,
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> None:
    header = f"Overall comparison for {diagnosis_name}"
    grp_only, grp_rt, grp_ct, grp_rt_ct = compare_groups_overall(df, header=header, n_boot=n_boot, seed=seed)
    compare_groups_by_timestep(df, header=f"Per-timestep comparison for {diagnosis_name}", n_boot=n_boot, seed=seed)
    if any(not g.empty for g in [grp_only, grp_rt, grp_ct, grp_rt_ct]):
        diagnosis_str = "".join(c if c.isalnum() else "_" for c in str(diagnosis_name))
        out_path = f"local_recurrence_boxplot_{diagnosis_str}.png"
        plot_boxplot(grp_only, grp_rt, grp_ct, grp_rt_ct, out_path=out_path)


def _binary_entropy(p: pd.Series, eps: float = 1e-12) -> pd.Series:
    p = p.clip(eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _gini_impurity_binary(p: pd.Series) -> pd.Series:
    return 2 * p * (1 - p)


def compute_timeseries_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work["group"] = _make_group_label(work)
    keep_groups = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(keep_groups)]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows = []
    for (ts, grp), chunk in work.groupby([TIMESTEP_COL, "group"], sort=True):
        s = chunk["prob_norm"].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        mean = s.mean()
        std = s.std(ddof=1)
        cv = (std / mean) if (np.isfinite(mean) and mean != 0) else np.nan
        rows.append(
            {
                "timestep": ts,
                "group": grp,
                "n": int(s.size),
                "mean": mean,
                "median": s.median(),
                "std": std,
                "q1": q1,
                "q3": q3,
                "IQR": iqr,
                "CV": cv,
                "entropy_mean": _binary_entropy(s).mean(),
                "gini_mean": _gini_impurity_binary(s).mean(),
            }
        )
    stats_df = pd.DataFrame(rows).sort_values(["timestep", "group"]).reset_index(drop=True)

    shifts_rows = []
    if not stats_df.empty:
        for ts, sub in stats_df.groupby("timestep"):
            base_row = sub[sub["group"] == "Surgery only"]
            base = base_row.iloc[0] if not base_row.empty else None
            for _, r in sub.iterrows():
                if base is None or r["group"] == "Surgery only":
                    mean_diff = np.nan
                    median_diff = np.nan
                    std_ratio = np.nan
                    iqr_ratio = np.nan
                else:
                    mean_diff = r["mean"] - base["mean"]
                    median_diff = r["median"] - base["median"]
                    std_ratio = (r["std"] / base["std"]) if (np.isfinite(base["std"]) and base["std"] > 0) else np.nan
                    iqr_ratio = (r["IQR"] / base["IQR"]) if (np.isfinite(base["IQR"]) and base["IQR"] > 0) else np.nan
                shifts_rows.append(
                    {
                        "timestep": ts,
                        "group": r["group"],
                        "mean_diff_vs_S": mean_diff,
                        "median_diff_vs_S": median_diff,
                        "std_ratio_vs_S": std_ratio,
                        "IQR_ratio_vs_S": iqr_ratio,
                    }
                )
    shifts_df = pd.DataFrame(shifts_rows).sort_values(["timestep", "group"]).reset_index(drop=True)

    deltas_rows = []
    if not stats_df.empty:
        ordered_ts = sorted(stats_df["timestep"].unique())
        if ordered_ts:
            t0 = ordered_ts[0]
            base_df = stats_df[stats_df["timestep"] == t0].set_index("group")
            for _, r in stats_df.iterrows():
                grp = r["group"]
                if grp in base_df.index:
                    base = base_df.loc[grp]
                    mean_delta = r["mean"] - base["mean"]
                    median_delta = r["median"] - base["median"]
                    iqr_ratio = (r["IQR"] / base["IQR"]) if (np.isfinite(base["IQR"]) and base["IQR"] > 0) else np.nan
                else:
                    mean_delta = np.nan
                    median_delta = np.nan
                    iqr_ratio = np.nan
                deltas_rows.append(
                    {
                        "timestep": r["timestep"],
                        "group": grp,
                        "mean_delta_from_t1": mean_delta,
                        "median_delta_from_t1": median_delta,
                        "IQR_ratio_vs_t1": iqr_ratio,
                    }
                )
    deltas_df = pd.DataFrame(deltas_rows).sort_values(["group", "timestep"]).reset_index(drop=True)

    return stats_df, shifts_df, deltas_df


def save_timeseries_statistics_csv(
    prefix: str,
    stats_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_IMG_DIR,
) -> None:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    if not stats_df.empty:
        stats_df.to_csv(base / f"{prefix}_timeseries_stats.csv", index=False)
    if not shifts_df.empty:
        shifts_df.to_csv(base / f"{prefix}_timeseries_shifts_vs_S.csv", index=False)
    if not deltas_df.empty:
        deltas_df.to_csv(base / f"{prefix}_timeseries_deltas_from_t1.csv", index=False)


def compute_histology_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    if HISTOLOGICAL_DIAGNOSIS_COL not in df.columns:
        return pd.DataFrame()
    if "patient_id" in df.columns:
        counts = df.groupby(HISTOLOGICAL_DIAGNOSIS_COL)["patient_id"].nunique()
    else:
        counts = df.groupby(HISTOLOGICAL_DIAGNOSIS_COL).size()
    counts = counts.sort_values(ascending=False)
    total = counts.sum()
    if total == 0:
        return pd.DataFrame()
    percent = counts / total * 100.0
    return (
        pd.DataFrame(
            {
                "histology": counts.index,
                "nr_patients": counts.values.astype(int),
                "percent": percent.values,
            }
        )
        .sort_values("nr_patients", ascending=False)
        .reset_index(drop=True)
    )


def run_histology_analysis(
    *,
    df: pd.DataFrame,
    bootstrap: int,
    seed: Optional[int],
    lam_gmd: float,
    output_dir: Path,
    skip_histology: bool = False,
    skip_grade: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_table = compute_histology_distribution_table(df)
    if not hist_table.empty:
        table_path = output_dir / "histology_distribution.csv"
        hist_table.to_csv(table_path, index=False)
        tex_path = output_dir / "histology_distribution.tex"
        hist_table.rename(
            columns={"histology": "Histology", "nr_patients": "Nr Patients", "percent": "Percent"}
        ).to_latex(tex_path, index=False, float_format=lambda x: f"{x:.1f}%")
        print(f"Saved histology distribution to {table_path}")

    try:
        res_lr = compare_two_arms_df(
            df,
            endpoint_col="LR_prob_norm_global" if "LR_prob_norm_global" in df.columns else LOCAL_RECURRENCE_COL,
            arm_A="Surgery only",
            arm_B="Surgery + RT",
            lam_gmd=lam_gmd,
            n_boot=bootstrap,
            paired=True,
            seed=seed,
        )
        print(
            "\nRisk-averse %Δscore (B vs A) on LR [B=S+RT, A=S]:",
            f"{res_lr['percent_change']:.2f}%  CI95={res_lr['CI95']}  p~={res_lr['p~']:.4f}",
        )
    except Exception as exc:
        print(f"Risk-averse arm comparison (LR) skipped: {exc}")

    if not skip_histology and HISTOLOGICAL_DIAGNOSIS_COL in df.columns:
        for hist in df[HISTOLOGICAL_DIAGNOSIS_COL].dropna().unique():
            hist_df = df[df[HISTOLOGICAL_DIAGNOSIS_COL] == hist]
            if hist_df.empty:
                continue
            weights = {
                "local_recurrence": 1 / 3,
                "metastasis": 1 / 3,
                "death_of_disease": 1 / 3,
            }
            ite_res = compute_individualized_treatment_effects(
                hist_df,
                weights=weights,
                n_boot=bootstrap,
                seed=seed,
            )

            safe_hist = str(hist).replace(" ", "_").replace("/", "_")
            try:
                plot_three_endpoint_trajectories_with_bands(
                    hist_df,
                    histology=str(hist),
                    out_path=f"three_endpoint_trajectories_bands_{safe_hist}.png",
                    n_boot=bootstrap,
                    seed=seed,
                    agg="median",
                )
                grid = risk_averse_comparison_grid(
                    hist_df,
                    lam_gmd=lam_gmd,
                    n_boot=bootstrap,
                    paired=True,
                    seed=seed,
                    baseline="Surgery only",
                )
                if not grid.empty:
                    out_csv = output_dir / f"risk_averse_scores_{safe_hist}.csv"
                    grid.to_csv(out_csv, index=False)
                    print(f"Saved risk-averse arm comparisons to {out_csv}")
            except Exception as exc:
                print(f"three_endpoint_trajectories_bands (histology={hist}) skipped: {exc}")

            if ite_res.empty:
                continue
            for treat in ["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]:
                plot_composite_benefit_waterfall(
                    ite_res,
                    treatment=treat,
                    histology=str(hist),
                    out_path=(
                        "composite_benefit_waterfall_"
                        f"{str(hist).replace(' ', '_')}_"
                        f"{treat.replace(' ', '_').replace('+', '')}.png"
                    ),
                )
                plot_composite_benefit_fan_envelope(
                    ite_res,
                    treatment=treat,
                    histology=str(hist),
                    out_path=(
                        "composite_benefit_fan_envelope_"
                        f"{str(hist).replace(' ', '_')}_"
                        f"{treat.replace(' ', '_').replace('+', '')}.png"
                    ),
                    n_boot=bootstrap,
                    seed=seed,
                )

            plot_endpoint_waterfall_fan_grid(
                ite_res,
                treatments=["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"],
                endpoints=["local_recurrence", "metastasis", "death_of_disease"],
                histology=str(hist),
                out_path=f"waterfall_fan_grid_{str(hist).replace(' ', '_')}.png",
                n_boot=bootstrap,
                seed=seed,
            )

    if not skip_grade and FNCLCC_GRADING_COL in df.columns:
        unique_grades = df[FNCLCC_GRADING_COL].unique()
        print(f"\nFound {len(unique_grades)} unique FNCLCC grades. Generating tables and plots for each.")
        for grade in unique_grades:
            if pd.isna(grade):
                filtered_df = df[df[FNCLCC_GRADING_COL].isna()]
                grade_str = "unknown_fnclcc_grade"
                print_grade = "Unknown"
            else:
                filtered_df = df[df[FNCLCC_GRADING_COL] == grade]
                grade_str = "".join(c if c.isalnum() else "_" for c in str(grade))
                print_grade = grade

            if filtered_df.empty:
                print(f"\n--- No data for FNCLCC grade: {print_grade} ---")
                continue

            print(f"\n--- Analysis for FNCLCC Grade: {print_grade} ---")
            analyze_diagnosis(filtered_df, diagnosis_name=str(print_grade), n_boot=bootstrap, seed=seed)
            s_df, sh_df, d_df = compute_timeseries_statistics(filtered_df)
            save_timeseries_statistics_csv(f"fnclcc_{grade_str}", s_df, sh_df, d_df, output_dir=output_dir)
            try:
                plot_three_endpoint_trajectories_with_bands(
                    filtered_df,
                    histology=str(print_grade),
                    out_path=f"three_endpoint_trajectories_bands_fnclcc_{grade_str}.png",
                    n_boot=bootstrap,
                    seed=seed,
                    agg="median",
                )
            except Exception as exc:
                print(f"three_endpoint_trajectories_bands (FNCLCC={print_grade}) skipped: {exc}")
            try:
                grid = risk_averse_comparison_grid(
                    filtered_df,
                    lam_gmd=lam_gmd,
                    n_boot=bootstrap,
                    paired=True,
                    seed=seed,
                    baseline="Surgery only",
                )
                if not grid.empty:
                    out_csv = output_dir / f"risk_averse_scores_fnclcc_{grade_str}.csv"
                    grid.to_csv(out_csv, index=False)
                    print(
                        "Saved risk-averse arm comparisons "
                        f"(FNCLCC {print_grade}) to {out_csv}"
                    )
            except Exception as exc:
                print(f"Risk-averse FNCLCC comparison (grade={print_grade}) skipped: {exc}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce histology and grade analyses for counterfactual scenarios.")
    parser.add_argument(
        "--input",
        help="Existing counterfactual scenarios CSV. If omitted, counterfactuals are generated from the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path("data") / "models" / "dtcygan_20250928_135339.pt"),
        help="Checkpoint containing generator weights (default: data/models/dtcygan_20250928_135339.pt).",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path("data") / "syntects.json"),
        help="Dataset JSON used when generating counterfactuals locally (default: data/syntects.json).",
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMG_DIR, help="Directory to store generated figures and tables.")
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap replicates used for uncertainty estimates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping.")
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Stochastic counterfactual samples per patient when generating locally (default: 32).",
    )
    parser.add_argument("--lambda-gmd", type=float, default=0.0, help="Weight applied to the Gini dispersion term in risk-averse scoring.")
    parser.add_argument("--skip-histology", action="store_true", help="Skip per-histology analyses.")
    parser.add_argument("--skip-grade", action="store_true", help="Skip per-grade analyses.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    global DEFAULT_IMG_DIR
    DEFAULT_IMG_DIR = args.image_dir

    seed_value = args.seed

    if args.input:
        df = load_and_prepare(args.input)
    else:
        cfg, metadata, generator, validation_ids, device = load_checkpoint_bundle(args.checkpoint)
        seed_value = args.seed if args.seed is not None else cfg.seed
        if seed_value is not None:
            set_seed(seed_value)

        dataset_payload = load_dataset_payload(args.dataset)
        dataset = SyntheticSequenceDataset(
            dataset_payload,
            cfg.seq_len,
            spec_clinical=metadata.get("clinical_feature_spec"),
            spec_treatment=metadata.get("treatment_feature_spec"),
        )

        cf_patients = build_counterfactual_patients(
            dataset,
            generator,
            device,
            validation_ids=None,
            status_labels=list(dataset.treat_categories.get("status_at_last_follow_up", OUTCOME_LABELS)),
            endpoint_labels=list(dataset.treat_categories.get("treatment_endpoint_category", [])),
            samples_per_patient=max(1, args.samples),
        )

        status_labels = sorted(
            set(dataset.treat_categories.get("status_at_last_follow_up", OUTCOME_LABELS))
            | set(collect_labels_from_patients(cf_patients, "status_at_last_follow_up", "status_probabilities"))
        )
        endpoint_labels = sorted(
            set(dataset.treat_categories.get("treatment_endpoint_category", []))
            | set(collect_labels_from_patients(cf_patients, "treatment_endpoint_category", "endpoint_probabilities"))
        )

        ensure_probability_metadata(cf_patients, status_labels, endpoint_labels)

        df_raw = patients_to_dataframe(
            cf_patients,
            status_labels=status_labels,
            endpoint_labels=endpoint_labels,
            include_timesteps=True,
        )

        df = load_and_prepare(df_raw)
    output_dir = Path(DEFAULT_IMG_DIR)

    run_histology_analysis(
        df=df,
        bootstrap=args.bootstrap,
        seed=seed_value,
        lam_gmd=args.lambda_gmd,
        output_dir=output_dir,
        skip_histology=args.skip_histology,
        skip_grade=args.skip_grade,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
