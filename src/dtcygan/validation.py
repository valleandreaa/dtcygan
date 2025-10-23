"""Bootstrap validation utilities for synthetic counterfactual predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


@dataclass
class BootstrapResult:
    baseline: float
    samples: np.ndarray

    def confidence_interval(self, alpha: float = 0.95) -> tuple[float, float]:
        ''' args:
        - alpha: confidence mass to retain around the baseline [float]

        return:
        - interval: tuple containing low and high percentile estimates [tuple[float, float]]
        '''
        lower = (1 - alpha) / 2 * 100
        upper = (1 + alpha) / 2 * 100
        low, high = np.percentile(self.samples, [lower, upper])
        return float(low), float(high)


def load_predictions(path: str | Path, label_column: str, score_column: str) -> tuple[np.ndarray, np.ndarray]:
    ''' args:
    - path: CSV file path containing predictions [str | Path]
    - label_column: column name holding binary labels [str]
    - score_column: column name holding prediction scores [str]

    return:
    - labels_scores: tuple of cleaned label and score arrays [tuple[np.ndarray, np.ndarray]]
    '''
    df = pd.read_csv(path)
    labels = pd.to_numeric(df[label_column], errors="coerce").to_numpy()
    scores = pd.to_numeric(df[score_column], errors="coerce").to_numpy()
    mask = (~np.isnan(labels)) & (~np.isnan(scores))
    labels, scores = (arr[mask].astype(float) for arr in (labels, scores))
    return labels, scores


def bootstrap_statistic(
    labels: np.ndarray,
    scores: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: Optional[int] = None,
) -> BootstrapResult:
    ''' args:
    - labels: ground-truth binary labels [np.ndarray]
    - scores: prediction scores aligned with labels [np.ndarray]
    - func: statistic function applied to labels and scores [Callable]
    - n_boot: number of bootstrap resamples [int]
    - seed: optional random seed for reproducibility [Optional[int]]

    return:
    - result: bootstrap result containing baseline and samples [BootstrapResult]
    '''
    rng = np.random.default_rng(seed)
    baseline = func(labels, scores)
    samples = np.empty(n_boot, dtype=float)
    n = labels.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = func(labels[idx], scores[idx])
    return BootstrapResult(baseline=baseline, samples=samples)


def score_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    ''' args:
    - labels: binary labels used for AUC computation [np.ndarray]
    - scores: prediction scores corresponding to labels [np.ndarray]

    return:
    - auc: area under curve value or NaN if undefined [float]
    '''
    if labels.size == 0 or np.all(labels == labels[0]):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_binary_predictions(
    csv_path: str | Path,
    label_column: str,
    score_column: str,
    n_boot: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    ''' args:
    - csv_path: CSV file containing labels and scores [str | Path]
    - label_column: name of the binary label column [str]
    - score_column: name of the prediction score column [str]
    - n_boot: number of bootstrap samples for confidence interval [int]
    - seed: optional random seed for reproducibility [Optional[int]]

    return:
    - metrics: dictionary with AUC baseline and confidence interval [dict]
    '''
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
    ''' args:
    - labels: binary labels for ROC computation [np.ndarray]
    - scores: prediction scores aligned to labels [np.ndarray]
    - out_path: destination path for the ROC figure [str | Path]
    - title: plot title displayed on the figure [str]

    return:
    - figure_path: resolved path to the saved ROC curve [Path]
    '''
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
    ''' args:
    - labels: binary labels for PR computation [np.ndarray]
    - scores: prediction scores aligned to labels [np.ndarray]
    - out_path: destination path for the PR figure [str | Path]
    - title: plot title displayed on the figure [str]

    return:
    - figure_path: resolved path to the saved PR curve [Path]
    '''
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
    ''' args:
    - result: bootstrap statistics comprising baseline and samples [BootstrapResult]
    - out_path: destination path for the histogram figure [str | Path]
    - title: plot title displayed on the figure [str]

    return:
    - figure_path: resolved path to the saved bootstrap histogram [Path]
    '''
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
