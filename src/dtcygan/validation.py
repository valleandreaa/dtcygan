"""Bootstrap validation utilities for synthetic counterfactual predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


@dataclass
class BootstrapResult:
    baseline: float
    samples: np.ndarray

    def confidence_interval(self, alpha: float = 0.95) -> tuple[float, float]:
        lower = (1 - alpha) / 2 * 100
        upper = (1 + alpha) / 2 * 100
        low, high = np.percentile(self.samples, [lower, upper])
        return float(low), float(high)


def load_predictions(path: str | Path, label_column: str, score_column: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    labels = pd.to_numeric(df[label_column], errors="coerce").to_numpy()
    scores = pd.to_numeric(df[score_column], errors="coerce").to_numpy()
    mask = ~np.isnan(labels) & ~np.isnan(scores)
    labels = labels[mask]
    scores = scores[mask]
    return labels.astype(float), scores.astype(float)


def bootstrap_statistic(
    labels: np.ndarray,
    scores: np.ndarray,
    func,
    n_boot: int = 2000,
    seed: Optional[int] = None,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    baseline = func(labels, scores)
    samples = np.empty(n_boot, dtype=float)
    n = labels.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = func(labels[idx], scores[idx])
    return BootstrapResult(baseline=baseline, samples=samples)


def score_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    if np.all(labels == labels[0]):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_binary_predictions(
    csv_path: str | Path,
    label_column: str,
    score_column: str,
    n_boot: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    labels, scores = load_predictions(csv_path, label_column, score_column)
    result = bootstrap_statistic(labels, scores, score_auc, n_boot=n_boot, seed=seed)
    ci_low, ci_high = result.confidence_interval()
    return {
        "auc": result.baseline,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str | Path,
    title: str = "ROC curve",
) -> Path:
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pr_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str | Path,
    title: str = "Precision-Recall curve",
) -> Path:
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_bootstrap_distribution(result: BootstrapResult, out_path: str | Path, title: str = "AUC bootstrap") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(result.samples, bins=40, color="#55A868", alpha=0.85)
    ax.axvline(result.baseline, color="black", linestyle="--", label=f"Baseline {result.baseline:.3f}")
    ci_low, ci_high = result.confidence_interval()
    ax.axvline(ci_low, color="red", linestyle=":", label=f"CI [{ci_low:.3f}, {ci_high:.3f}]")
    ax.axvline(ci_high, color="red", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Frequency")
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = [
    "BootstrapResult",
    "load_predictions",
    "bootstrap_statistic",
    "evaluate_binary_predictions",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_bootstrap_distribution",
]
