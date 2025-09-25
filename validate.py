#!/usr/bin/env python
"""CLI for bootstrap validation of binary predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dtcygan.validation import (
    score_binary_classifier,
    load_predictions,
    bootstrap_statistic,
    score_auc,
    plot_roc_curve,
    plot_pr_curve,
    plot_bootstrap_distribution,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap AUC evaluation for binary predictions.")
    parser.add_argument("--csv", required=True, help="CSV file containing labels and prediction scores.")
    parser.add_argument("--label-column", required=True, help="Column name holding binary labels (0/1).")
    parser.add_argument("--score-column", required=True, help="Column name holding prediction scores.")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap samples (default: 1000).")
    parser.add_argument("--seed", type=int, help="Optional random seed for bootstrapping.")
    parser.add_argument(
        "--plot-dir",
        type=str,
        help="Optional directory to store ROC, PR, and bootstrap histogram plots.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def execute_validation(args: argparse.Namespace) -> None:
    result = score_binary_classifier(
        csv_path=args.csv,
        label_column=args.label_column,
        score_column=args.score_column,
        n_boot=args.bootstraps,
        seed=args.seed,
    )
    print("AUC: {auc:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])".format(**result))
    if args.plot_dir:
        labels, scores = load_predictions(args.csv, args.label_column, args.score_column)
        stats = bootstrap_statistic(labels, scores, score_auc, n_boot=args.bootstraps, seed=args.seed)
        plot_dir = Path(args.plot_dir)
        roc_path = plot_roc_curve(labels, scores, plot_dir / "roc_curve.png")
        pr_path = plot_pr_curve(labels, scores, plot_dir / "pr_curve.png")
        hist_path = plot_bootstrap_distribution(stats, plot_dir / "auc_bootstrap.png")
        print("Plots saved:")
        print(f"  ROC: {roc_path}")
        print(f"  PR: {pr_path}")
        print(f"  Bootstrap: {hist_path}")


def main(argv: list[str] | None = None) -> None:
    execute_validation(parse_args(argv))


if __name__ == "__main__":
    main()
