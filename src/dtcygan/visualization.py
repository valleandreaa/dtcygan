"""Plotting and individualized treatment effect helpers used by the analysis CLI."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "analysis_plots.yaml"

DEFAULT_TREATMENT_GROUPS = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
DEFAULT_ENDPOINT_KEYS = ["local_recurrence", "metastasis", "death_of_disease"]


def as_prob(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    clipped = values.clip(lower=0.0, upper=1.0)
    return clipped.astype(float)


def load_plot_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load analysis plotting defaults from YAML."""

    if path is None:
        return copy.deepcopy(_default_plot_config())
    return _read_plot_config(Path(path))


@lru_cache()
def _default_plot_config() -> Dict[str, Any]:
    return _read_plot_config(DEFAULT_CONFIG_PATH)


def _read_plot_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return dict(data)


def _prepare_waterfall_base(ite_df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    base = ite_df[[patient_col, "treatment", "endpoint", "ite"]].copy()
    base["ite"] = base["ite"].astype(float)
    return base


def _create_waterfall_axes(n_rows: int, n_cols: int) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 2.8 * n_rows),
        sharex=False,
        sharey=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes_arr = np.array([[axes]])
    elif n_rows == 1:
        axes_arr = np.array([axes])
    elif n_cols == 1:
        axes_arr = axes.reshape(-1, 1)
    else:
        axes_arr = axes
    return fig, axes_arr


def _endpoint_ylim(base: pd.DataFrame, endpoint: str) -> Tuple[float, float]:
    vals = base.loc[base["endpoint"] == endpoint, "ite"].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return (-0.01, 0.01)
    lo, hi = np.nanpercentile(vals, [1, 99])
    rng_val = max(abs(lo), abs(hi))
    if not np.isfinite(rng_val) or rng_val == 0:
        rng_val = 0.01
    return (-rng_val, rng_val)


def _panel_values(
    base: pd.DataFrame,
    endpoint: str,
    treatment: str,
    patient_col: str,
) -> np.ndarray:
    vals = (
        base[(base["endpoint"] == endpoint) & (base["treatment"] == treatment)][[patient_col, "ite"]]
        .drop_duplicates(subset=[patient_col])["ite"]
        .dropna()
        .to_numpy(dtype=float)
    )
    return vals


def _render_waterfall_panel(
    ax: Axes,
    vals: np.ndarray,
    y_limits: Tuple[float, float],
    *,
    n_boot: int,
    seed: Optional[int],
    sort_desc: bool,
) -> bool:
    if vals.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    q5, q50, q95 = _rank_consistent_envelope(vals, n_boot=n_boot, seed=seed, sort_desc=sort_desc)
    ranks = np.arange(1, q50.size + 1)
    ax.fill_between(ranks, q5, q95, color="#1f77b4", alpha=0.2, linewidth=0)
    ax.plot(ranks, q50, color="#1f77b4", lw=1.8)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlim(1, ranks[-1])
    ax.set_ylim(*y_limits)
    n_benef = int(np.sum(vals < 0))
    ax.text(
        0.98,
        0.92,
        f"{n_benef}/{vals.size} ({(n_benef / vals.size):.0%})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.25)
    return True


def plot_endpoint_waterfall_fan_grid(
    ite_df: pd.DataFrame,
    *,
    out_path: str | Path,
    treatments: Optional[Sequence[str]] = None,
    endpoints: Optional[Sequence[str]] = None,
    histology: Optional[str] = None,
    n_boot: int = 200,
    seed: Optional[int] = 42,
    patient_col: str = "patient_id",
    sort_desc: Optional[bool] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Path:
    cfg = dict(config or {})
    endpoints = list(endpoints or cfg.get("endpoints", DEFAULT_ENDPOINT_KEYS))
    treatments = list(treatments or cfg.get("treatments", DEFAULT_TREATMENT_GROUPS[1:]))
    treatment_labels = cfg.get(
        "treatment_labels",
        {
            "Surgery + RT": "S+RT",
            "Surgery + CT": "S+CT",
            "Surgery + RT + CT": "S+RT+CT",
        },
    )
    endpoint_labels = cfg.get(
        "endpoint_labels",
        {
            "local_recurrence": "Local Recurrence",
            "metastasis": "Metastasis",
            "death_of_disease": "DOD",
        },
    )
    sort_desc = bool(cfg.get("sort_desc", False) if sort_desc is None else sort_desc)

    base = _prepare_waterfall_base(ite_df, patient_col)
    if base.empty:
        print("plot_endpoint_waterfall_fan_grid: empty ite_df")
        return Path(out_path)

    n_rows, n_cols = len(endpoints), len(treatments)
    fig, axes = _create_waterfall_axes(n_rows, n_cols)
    ylims = {ep: _endpoint_ylim(base, ep) for ep in endpoints}

    for r, ep in enumerate(endpoints):
        for c, tr in enumerate(treatments):
            ax = axes[r, c]
            vals = _panel_values(base, ep, tr, patient_col)
            drew_panel = _render_waterfall_panel(
                ax,
                vals,
                y_limits=ylims[ep],
                n_boot=n_boot,
                seed=seed,
                sort_desc=sort_desc,
            )
            if not drew_panel:
                continue
            if r == n_rows - 1:
                ax.set_xlabel("Rank")
            if c == 0:
                ax.set_ylabel(endpoint_labels.get(ep, ep))
            ax.set_title(treatment_labels.get(tr, tr), fontsize=10)

    fig.suptitle("" if histology is None else str(histology), y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"waterfall fan grid saved to {out_path}")
    return out_path


def _prepare_grouped_dataframe(df: pd.DataFrame, groups: Sequence[str], group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not present in dataframe.")
    return df[df[group_col].isin(groups)].copy()


def _resolve_endpoint_columns(endpoint_cols: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if endpoint_cols is not None:
        return dict(endpoint_cols)
    return {
        "local_recurrence": "local_recurrence",
        "metastasis": "metastasis",
        "death_of_disease": "death_of_disease",
    }


def _normalized_trapezoid(times: np.ndarray, values: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return float(values[0])
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    span = times[-1] - times[0]
    if span == 0:
        return float(np.nanmean(values))
    return float(np.trapezoid(values, x=times) / span)


def _compute_patient_ites(
    delta: pd.DataFrame,
    patient_col: str,
    time_col: str,
) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame(columns=[patient_col, "ite"])

    def _integrate(group: pd.DataFrame) -> float:
        return _normalized_trapezoid(group[time_col].to_numpy(), group["delta"].to_numpy())

    return (
        delta.groupby(patient_col)
        .apply(_integrate, include_groups=False)
        .to_frame(name="ite")
        .reset_index()
    )


def _treatment_effects_for_endpoint(
    df: pd.DataFrame,
    endpoint_label: str,
    endpoint_col: str,
    *,
    patient_col: str,
    time_col: str,
    reference_group: str,
    groups: Sequence[str],
    group_col: str,
) -> List[pd.DataFrame]:
    pivot = df.pivot_table(index=[patient_col, time_col], columns=group_col, values=endpoint_col)
    if reference_group not in pivot.columns:
        return []

    frames: List[pd.DataFrame] = []
    ref_vals = pivot[reference_group]
    for treatment in groups:
        if treatment == reference_group or treatment not in pivot.columns:
            continue
        delta = (pivot[treatment] - ref_vals).dropna().reset_index(name="delta")
        per_patient = _compute_patient_ites(delta, patient_col, time_col)
        if per_patient.empty:
            continue
        per_patient["treatment"] = treatment
        per_patient["endpoint"] = endpoint_label
        frames.append(per_patient)
    return frames


def _append_composite_scores(
    result: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    patient_col: str,
) -> pd.DataFrame:
    weight_series = pd.Series(weights, dtype=float)
    composite = (
        result.pivot_table(index=[patient_col, "treatment"], columns="endpoint", values="ite")
        .reindex(columns=weight_series.index)
        .mul(weight_series, axis=1)
        .sum(axis=1)
        .reset_index(name="composite_ite")
    )
    return result.merge(composite, on=[patient_col, "treatment"], how="left")


def _build_weighted_timecourse(
    ref: pd.DataFrame,
    trt: pd.DataFrame,
    ep_cols: Mapping[str, str],
    weight_series: pd.Series,
    *,
    time_col: str,
) -> Optional[pd.DataFrame]:
    comp_df: Optional[pd.DataFrame] = None
    for ep_name, col in ep_cols.items():
        if col not in ref.columns or col not in trt.columns or ep_name not in weight_series.index:
            continue
        ref_ep = ref[[time_col, col]].rename(columns={col: "ref"})
        trt_ep = trt[[time_col, col]].rename(columns={col: "trt"})
        merged = pd.merge(ref_ep, trt_ep, on=time_col, how="inner")
        if merged.empty:
            continue
        merged[ep_name] = (merged["trt"].astype(float) - merged["ref"].astype(float)) * float(weight_series[ep_name])
        merged = merged[[time_col, ep_name]]
        comp_df = merged if comp_df is None else pd.merge(comp_df, merged, on=time_col, how="inner")
    if comp_df is None or comp_df.empty:
        return None
    comp_df = comp_df.sort_values(time_col)
    comp_df["comp"] = comp_df.drop(columns=[time_col]).sum(axis=1)
    return comp_df[[time_col, "comp"]]


def _bootstrap_composite_intervals(
    comp_df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_boot: int,
    time_col: str,
) -> Tuple[float, float]:
    if comp_df.empty or n_boot <= 0:
        return (np.nan, np.nan)
    times = comp_df[time_col].to_numpy()
    vals = comp_df["comp"].to_numpy()
    if vals.size < 2:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, vals.size, size=vals.size)
        tb = times[idx]
        vb = vals[idx]
        order = np.argsort(tb)
        tb = tb[order]
        vb = vb[order]
        boots.append(_normalized_trapezoid(tb, vb))
    if not boots:
        return (np.nan, np.nan)
    low, high = np.percentile(boots, [2.5, 97.5])
    return (float(low), float(high))


def _append_composite_ci(
    result: pd.DataFrame,
    df: pd.DataFrame,
    *,
    weights: Mapping[str, float],
    endpoint_cols: Mapping[str, str],
    patient_col: str,
    time_col: str,
    reference_group: str,
    n_boot: int,
    seed: Optional[int],
    groups: Sequence[str],
    group_col: str,
) -> pd.DataFrame:
    if n_boot <= 0:
        return result

    rng = np.random.default_rng(seed)
    weight_series = pd.Series(weights, dtype=float)

    ep_cols = {
        k: endpoint_cols.get(k, v)
        for k, v in {
            "local_recurrence": endpoint_cols.get("local_recurrence", "local_recurrence"),
            "metastasis": endpoint_cols.get("metastasis", "metastasis"),
            "death_of_disease": endpoint_cols.get("death_of_disease", "death_of_disease"),
        }.items()
    }

    comp_ci_rows: List[Dict[str, Any]] = []
    iterable = (
        result.dropna(subset=["composite_ite"])
        .drop_duplicates([patient_col, "treatment"])
        .groupby([patient_col, "treatment"])
    )
    for (pid, treat), _ in iterable:
        if treat == reference_group or treat not in groups:
            continue
        ref = df[(df[patient_col] == pid) & (df[group_col] == reference_group)]
        trt = df[(df[patient_col] == pid) & (df[group_col] == treat)]
        if ref.empty or trt.empty:
            continue
        comp_df = _build_weighted_timecourse(ref, trt, ep_cols, weight_series, time_col=time_col)
        if comp_df is None:
            continue
        low, high = _bootstrap_composite_intervals(comp_df, rng=rng, n_boot=n_boot, time_col=time_col)
        comp_ci_rows.append(
            {
                patient_col: pid,
                "treatment": treat,
                "composite_ci_low": low,
                "composite_ci_high": high,
            }
        )

    if not comp_ci_rows:
        return result

    comp_ci_df = pd.DataFrame(comp_ci_rows)
    return result.merge(
        comp_ci_df[[patient_col, "treatment", "composite_ci_low", "composite_ci_high"]],
        on=[patient_col, "treatment"],
        how="left",
    )


def compute_individualized_treatment_effects(
    df: pd.DataFrame,
    *,
    groups: Sequence[str] = DEFAULT_TREATMENT_GROUPS,
    reference_group: str = "Surgery only",
    patient_col: str = "patient_id",
    time_col: str = "timestep",
    endpoint_cols: Optional[Mapping[str, str]] = None,
    group_col: str = "group",
    weights: Optional[Mapping[str, float]] = None,
    n_boot: int = 0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    df_grouped = _prepare_grouped_dataframe(df, groups, group_col=group_col)
    endpoint_map = _resolve_endpoint_columns(endpoint_cols)

    frames: List[pd.DataFrame] = []
    for endpoint, col in endpoint_map.items():
        if col not in df_grouped.columns:
            continue
        frames.extend(
            _treatment_effects_for_endpoint(
                df_grouped,
                endpoint,
                col,
                patient_col=patient_col,
                time_col=time_col,
                reference_group=reference_group,
                groups=groups,
                group_col=group_col,
            )
        )

    if not frames:
        return pd.DataFrame(columns=[patient_col, "treatment", "endpoint", "ite"])

    result = pd.concat(frames, ignore_index=True)

    if not weights:
        return result

    result = _append_composite_scores(result, weights, patient_col=patient_col)
    if n_boot > 0:
        result = _append_composite_ci(
            result,
            df_grouped,
            weights=weights,
            endpoint_cols=endpoint_map,
            patient_col=patient_col,
            time_col=time_col,
            reference_group=reference_group,
            n_boot=n_boot,
            seed=seed,
            groups=groups,
            group_col=group_col,
        )

    return result


def _bootstrap_timecourse_by_group(
    df: pd.DataFrame,
    *,
    value_col: str,
    time_col: str,
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


def plot_three_endpoint_trajectories_with_bands(
    df: pd.DataFrame,
    *,
    endpoints: Sequence[Tuple[str, str]],
    out_path: str | Path,
    groups: Sequence[str] = DEFAULT_TREATMENT_GROUPS,
    time_col: str = "timestep",
    patient_col: str = "patient_id",
    group_col: str = "group",
    histology_col: Optional[str] = None,
    histology: Optional[str] = None,
    n_boot: int = 200,
    seed: Optional[int] = 42,
    agg: Literal["median", "mean"] = "median",
    band_method: Literal["pointwise", "rcqe"] = "rcqe",
    config: Optional[Mapping[str, Any]] = None,
) -> Path:
    if df.empty:
        print("plot_three_endpoint_trajectories_with_bands: Empty DataFrame; nothing to plot.")
        return Path(out_path)

    cfg = dict(config or {})
    use_color = bool(cfg.get("use_color", True))
    layout = cfg.get("layout", "rows")
    show_counts = bool(cfg.get("show_counts", True))
    show_errorbars = bool(cfg.get("show_errorbars", True))
    table_outside = bool(cfg.get("table_outside", True))
    table_mode = cfg.get("table_mode", "summary")
    side_titles = bool(cfg.get("side_titles", True))
    annotate_arm_dispersion = bool(cfg.get("annotate_arm_dispersion", True))
    row_height = float(cfg.get("row_height", 5.0))
    panel_hspace = float(cfg.get("panel_hspace", 0.85))
    table_gap = float(cfg.get("table_gap", 0.10))
    legend_y = float(cfg.get("legend_y", 1.4))
    tight_top = float(cfg.get("tight_top", 0.92))

    color_map = cfg.get(
        "color_map",
        {
            "Surgery only": "#1f77b4",
            "Surgery + RT": "#ff7f0e",
            "Surgery + CT": "#2ca02c",
            "Surgery + RT + CT": "#d62728",
        },
    )
    gray_line_map = cfg.get(
        "gray_line_map",
        {
            "Surgery only": "#111111",
            "Surgery + RT": "#333333",
            "Surgery + CT": "#555555",
            "Surgery + RT + CT": "#000000",
        },
    )
    linestyles = cfg.get(
        "linestyles",
        {
            "Surgery only": "-",
            "Surgery + RT": "--",
            "Surgery + CT": "-.",
            "Surgery + RT + CT": ":",
        },
    )
    markers = cfg.get(
        "markers",
        {"Surgery only": "o", "Surgery + RT": "s", "Surgery + CT": "^", "Surgery + RT + CT": "D"},
    )

    work = df.copy()
    if histology is not None and histology_col and histology_col in work.columns:
        work = work[work[histology_col] == histology]

    if group_col not in work.columns:
        raise KeyError(f"Column '{group_col}' not present in dataframe.")
    work = work[work[group_col].isin(groups)]
    if work.empty:
        print("plot_three_endpoint_trajectories_with_bands: No rows for the specified groups/histology.")
        return Path(out_path)

    endpoints = list(endpoints)
    all_times = sorted(work[time_col].dropna().unique()) if time_col in work.columns else []
    n_steps = len(all_times)
    if layout == "rows":
        fig_width = max(18.0, min(32.0, 0.7 * max(1, n_steps)))
        extra_bottom = 1.6 if (show_counts and table_outside and (table_mode in ("summary", "both"))) else (
            1.2 if (show_counts and table_outside) else 0.5
        )
        extra_top = 1.8
        fig_height = row_height * len(endpoints) + extra_bottom + extra_top
        fig, axes = plt.subplots(len(endpoints), 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    else:
        fig_width = max(16.0, min(24.0, 4.5 + 0.35 * max(1, n_steps)))
        fig_height = 5.0
        fig, axes = plt.subplots(1, len(endpoints), figsize=(fig_width, fig_height), sharey=True)
    axes = np.atleast_1d(axes)

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
    counts_map: Dict[str, np.ndarray] = {}
    trajectories: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    times_ref: Optional[np.ndarray] = None

    for ax_idx, (ax, (ep_name, ep_col)) in enumerate(zip(np.ravel(axes), endpoints)):
        if ep_col not in work.columns:
            ax.set_visible(False)
            continue

        trajectories.clear()
        counts_map.clear()
        times_ref = None
        for arm in groups:
            sub = work[work[group_col] == arm]
            if sub.empty or sub[ep_col].dropna().empty:
                continue
            times, center, (low, high), counts = _bootstrap_timecourse_by_group(
                sub,
                value_col=ep_col,
                time_col=time_col,
                patient_col=patient_col,
                n_boot=n_boot,
                seed=seed,
                agg=agg,
                band_method=band_method,
            )
            if times.size == 0:
                continue
            if times_ref is None:
                times_ref = times
            trajectories[arm] = (center, low, high)
            counts_map[arm] = counts

        if times_ref is None:
            ax.set_visible(False)
            continue

        for arm in groups:
            if arm not in trajectories:
                continue
            center, low, high = trajectories[arm]
            base_c = color_map.get(arm, "#1f77b4") if use_color else gray_line_map.get(arm, "#333333")
            ax.fill_between(times_ref, low, high, color=base_c, alpha=0.18, linewidth=0)

        if len(times_ref) > 0:
            step = max(1, int(len(times_ref) / 10))
        else:
            step = 1

        for arm in groups:
            if arm not in trajectories:
                continue
            center, low, high = trajectories[arm]
            line_c = color_map.get(arm, "#1f77b4") if use_color else gray_line_map.get(arm, "#333333")
            ax.plot(times_ref, center, linestyle=linestyles.get(arm, "-"), color=line_c, linewidth=2.2, label=arm, zorder=3)
            marker_style = markers.get(arm, "o")
            ax.plot(times_ref[::step], center[::step], linestyle="None", marker=marker_style, color=line_c, markersize=4, alpha=0.8, zorder=4)
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

        if annotate_arm_dispersion and time_col in work.columns:
            nbins = 20
            bins = np.linspace(0.0, 1.0, nbins + 1)
            I_vals: List[float] = []
            dG_vals: List[float] = []
            for t in times_ref:
                values_by_arm: Dict[str, np.ndarray] = {}
                ns: Dict[str, int] = {}
                total_n = 0
                for arm in groups:
                    s = pd.to_numeric(work.loc[(work[group_col] == arm) & (work[time_col] == t), ep_col], errors="coerce").dropna()
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
                    return 1.0 - float(np.sum(p**2))

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
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.65, edgecolor="none"),
                )

        if show_counts and times_ref is not None:
            shown_arms = [arm for arm in groups if arm in trajectories]
            if shown_arms:
                col_labels = [f"t={int(t)}" for t in times_ref]
                row_map = {
                    "Surgery only": "S",
                    "Surgery + RT": "S+RT",
                    "Surgery + CT": "S+CT",
                    "Surgery + RT + CT": "S+RT+CT",
                }
                disp_rows = [row_map.get(a, a) for a in shown_arms]
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
                        cell_text = [
                            [f"n={int(counts_map[a][i])}\n{summary_rows[j][i]}" for i in range(len(times_ref))]
                            for j, a in enumerate(shown_arms)
                        ]
                else:
                    cell_text = [[str(int(n)) for n in counts_map[a]] for a in shown_arms]

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

    if any_drawn and times_ref is not None:
        legend_colors = color_map if use_color else gray_line_map
        label_map = {
            "Surgery only": "S",
            "Surgery + RT": "S+RT",
            "Surgery + CT": "S+CT",
            "Surgery + RT + CT": "S+RT+CT",
        }
        legend_handles = [
            Line2D([0], [0], color=legend_colors.get(arm, "#1f77b4"), linestyle=linestyles.get(arm, "-"), linewidth=2.2, label=label_map.get(arm, arm))
            for arm in groups
        ]
        labels = [h.get_label() for h in legend_handles]
        if layout == "rows":
            axes[0].legend(
                legend_handles,
                labels,
                loc="upper center",
                ncol=len(groups),
                frameon=True,
                bbox_to_anchor=(0.5, legend_y),
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
                labels,
                loc="upper center",
                ncol=len(groups),
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
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"three-endpoint trajectories with bands saved to {out_path}")
        return out_path

    plt.close(fig)
    print("plot_three_endpoint_trajectories_with_bands: nothing drawn (missing columns/data).")
    return Path(out_path)


__all__ = [
    "DEFAULT_TREATMENT_GROUPS",
    "as_prob",
    "load_plot_config",
    "compute_individualized_treatment_effects",
    "plot_endpoint_waterfall_fan_grid",
    "plot_three_endpoint_trajectories_with_bands",
]
