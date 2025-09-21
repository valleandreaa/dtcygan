#!/usr/bin/env python
"""CLI for generating counterfactual treatments from a checkpoint and synthetic dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ditcygan.inference import generate_counterfactuals


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate counterfactual treatments using a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--data", required=True, help="Synthetic dataset (JSON) to evaluate.")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSON file for generated counterfactual summaries.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help="Optional names for additional scenarios (defaults to ['actual', 'random']).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def run_inference(args: argparse.Namespace) -> None:
    generate_counterfactuals(
        checkpoint_path=args.checkpoint,
        dataset_path=args.data,
        output_path=args.output,
        scenario_names=args.scenarios,
    )
    print(f"Counterfactual summaries written to {args.output}")


def main(argv: list[str] | None = None) -> None:
    run_inference(parse_args(argv))


if __name__ == "__main__":
    main()
