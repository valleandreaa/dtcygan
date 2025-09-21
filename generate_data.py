#!/usr/bin/env python
"""CLI to write synthetic clinical/treatment JSON payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ditcygan.synthetic import generate_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical/treatment sequences for demos."
    )
    parser.add_argument("--patients", type=int, default=25, help="Number of patients to simulate (default: 25)")
    parser.add_argument("--timesteps", type=int, default=5, help="Sequence length per patient (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", required=True, help="Path to the JSON file to create")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for pretty-printing JSON (default: 2; use 0 for compact)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(SRC_DIR / "config" / "synthetic.yaml"),
        help="Schema file describing feature distributions (default: synthetic.yaml).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def write_synthetic_dataset(args: argparse.Namespace) -> None:
    payload = generate_dataset(
        num_patients=args.patients,
        timesteps=args.timesteps,
        seed=args.seed,
        schema_path=args.schema,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indent = None if args.indent <= 0 else args.indent
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent)

    print(f"Synthetic dataset written to {output_path.resolve()}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    write_synthetic_dataset(args)


if __name__ == "__main__":
    main()
