#!/usr/bin/env python
"""CLI for basic analysis of synthetic counterfactual datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ditcygan.analysis import (
    load_table,
    summarize_numeric,
    save_summary,
    plot_histograms,
    plot_grouped_boxplots,
    plot_timeseries_mean,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize numeric columns in a dataset.")
    parser.add_argument("--input", required=True, help="CSV or JSON file to analyse.")
    parser.add_argument("--output", required=True, help="Where to save the summary CSV.")
    parser.add_argument("--summary-columns", nargs="*", help="Optional columns for summary statistics (defaults to numeric columns)")
    parser.add_argument("--hist-columns", nargs="*", help="Columns to plot as histograms")
    parser.add_argument("--box-columns", nargs="*", help="Value columns to plot as grouped boxplots")
    parser.add_argument("--group-column", help="Grouping column used for boxplots and timeseries plots")
    parser.add_argument("--timeseries-columns", nargs="*", help="Value columns for timeseries mean plots")
    parser.add_argument("--time-column", help="Time axis column for timeseries plots (required if timeseries columns set)")
    parser.add_argument("--plot-dir", type=str, help="Directory to store generated plots (required when any plot option is used)")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def run_analysis(args: argparse.Namespace) -> None:
    df = load_table(args.input)
    summaries = summarize_numeric(df, args.summary_columns)
    save_summary(summaries, args.output)
    print(f"Summary saved to {args.output}")

    if any([args.hist_columns, args.box_columns, args.timeseries_columns]):
        if not args.plot_dir:
            raise SystemExit("--plot-dir is required when requesting plots")

    if args.hist_columns:
        paths = plot_histograms(df, args.hist_columns, args.plot_dir)
        print("Histogram plots:")
        for path in paths:
            print(f"  {path}")

    if args.box_columns:
        if not args.group_column:
            raise SystemExit("--group-column is required for boxplots")
        paths = plot_grouped_boxplots(df, args.box_columns, args.group_column, args.plot_dir)
        print("Boxplots:")
        for path in paths:
            print(f"  {path}")

    if args.timeseries_columns:
        if not args.time_column:
            raise SystemExit("--time-column is required for timeseries plots")
        paths = plot_timeseries_mean(
            df,
            args.time_column,
            args.timeseries_columns,
            args.group_column,
            args.plot_dir,
        )
        print("Timeseries plots:")
        for path in paths:
            print(f"  {path}")


def main(argv: list[str] | None = None) -> None:
    run_analysis(parse_args(argv))


if __name__ == "__main__":
    main()
