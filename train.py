#!/usr/bin/env python
"""CLI entrypoint for training the temporal CycleGAN on synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dtcygan.training import Config, load_config, train


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the temporal CycleGAN on synthetic sequences.")
    parser.add_argument("--config", default=str(SRC_DIR / "config" / "training.yaml"), help="Path to YAML configuration file")
    parser.add_argument("--synthetic-data", type=str, help="Optional path to a synthetic JSON dataset. If omitted a fresh dataset is generated.")
    parser.add_argument("--schema", type=str, default=str(SRC_DIR / "config" / "synthetic.yaml"), help="Schema file used when generating synthetic data on the fly.")
    parser.add_argument("--patients", type=int, default=64, help="Number of synthetic patients when generating data (default: 64)")
    parser.add_argument("--timesteps", type=int, default=8, help="Sequence length for generated patients (default: 8)")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for dataset generation (overrides config seed)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory where model checkpoints are written")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def run_training(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    return train(
        cfg,
        synthetic_data=args.synthetic_data,
        patients=args.patients,
        timesteps=args.timesteps,
        seed=args.seed,
        schema_path=args.schema,
        checkpoint_dir=args.checkpoint_dir,
    )


def main(argv: list[str] | None = None) -> None:
    ckpt_path = run_training(parse_args(argv))
    print(f"Training finished. Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
