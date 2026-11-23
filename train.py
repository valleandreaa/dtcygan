#!/usr/bin/env python
"""Internal training entry point used by the frozen paper workflows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

ROOT = Path(__file__).resolve().parent

from src.training import load_config, train


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--synthetic-data")
    dataset.add_argument("--person-period-data")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--causal-weights")
    parser.add_argument("--causal-weight-metadata")
    parser.add_argument("--longitudinal-weights")
    parser.add_argument("--longitudinal-weight-metadata")
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


def run_training(args):
    # Every weight file is interpreted together with its metadata.
    if bool(args.causal_weights) != bool(args.causal_weight_metadata):
        raise ValueError("Patient-level weights require their metadata file")
    if bool(args.longitudinal_weights) != bool(args.longitudinal_weight_metadata):
        raise ValueError("Longitudinal weights require their metadata file")
    # YAML is the single source of model settings.
    cfg = load_config(args.config)

    return train(
        cfg,
        synthetic_data=args.synthetic_data,
        person_period_data=args.person_period_data,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        causal_weights_path=args.causal_weights,
        causal_weight_metadata_path=args.causal_weight_metadata,
        longitudinal_weights_path=args.longitudinal_weights,
        longitudinal_weight_metadata_path=args.longitudinal_weight_metadata,
    )


def main(argv=None):
    ckpt_path = run_training(parse_args(argv))
    print(f"Training finished. Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
