"""Visualization and summary helpers for synthetic counterfactual outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SummaryStats:
    column: str
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def summarize_numeric(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> List[SummaryStats]:
    if columns is None:
        columns = df.select_dtypes(include="number").columns
    summaries: List[SummaryStats] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        summaries.append(
            SummaryStats(
                column=col,
                mean=float(series.mean()),
                std=float(series.std(ddof=0)),
                median=float(series.median()),
                minimum=float(series.min()),
                maximum=float(series.max()),
            )
        )
    return summaries


def save_summary(summaries: List[SummaryStats], path: str | Path) -> None:
    rows = [s.__dict__ for s in summaries]
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_histograms(
    df: pd.DataFrame,
    columns: Iterable[str],
    output_dir: str | Path,
    bins: int = 30,
) -> List[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=bins, color="#4C72B0", alpha=0.85)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        out_path = out_dir / f"hist_{col}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def plot_grouped_boxplots(
    df: pd.DataFrame,
    value_columns: Iterable[str],
    group_column: str,
    output_dir: str | Path,
) -> List[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for col in value_columns:
        local = df[[group_column, col]].dropna()
        if local.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        local.boxplot(column=col, by=group_column, ax=ax, grid=False)
        ax.set_title(f"{col} by {group_column}")
        ax.set_xlabel(group_column)
        ax.set_ylabel(col)
        fig.suptitle("")
        out_path = out_dir / f"boxplot_{col}_by_{group_column}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def plot_timeseries_mean(
    df: pd.DataFrame,
    time_column: str,
    value_columns: Iterable[str],
    group_column: Optional[str],
    output_dir: str | Path,
) -> List[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    value_columns = list(value_columns)
    if not value_columns:
        value_columns = df.select_dtypes(include="number").columns.tolist()
    group_values = df[group_column].unique() if group_column else [None]
    for value_col in value_columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in group_values:
            subset = df if group is None else df[df[group_column] == group]
            subset = subset[[time_column, value_col]].dropna()
            if subset.empty:
                continue
            grouped = subset.groupby(time_column)[value_col].mean()
            label = str(group) if group is not None else value_col
            ax.plot(grouped.index, grouped.values, marker="o", label=label)
        ax.set_title(f"Mean {value_col} over {time_column}")
        ax.set_xlabel(time_column)
        ax.set_ylabel(value_col)
        if group_column:
            ax.legend(title=group_column)
        out_path = out_dir / f"timeseries_{value_col}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


__all__ = [
    "SummaryStats",
    "load_table",
    "summarize_numeric",
    "save_summary",
    "plot_histograms",
    "plot_grouped_boxplots",
    "plot_timeseries_mean",
]
