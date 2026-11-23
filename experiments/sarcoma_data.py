#!/usr/bin/env python3
"""Prepare the fixed preoperative-RT versus surgery analysis inputs."""

import argparse
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "sarcoma"
CONFIG = PROJECT / "config"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from weights import run_experiment  # noqa: E402
from postprocess_weights import main as build_postprocess_weights  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.timeline.is_file():
        raise FileNotFoundError(args.timeline)

    cohort = args.output_dir / "cohort"
    weights = args.output_dir / "weights"

    # Build the fixed cohort and selected training weights.
    run_experiment(
        args.timeline,
        CONFIG / "sarcoma_cohort.yaml",
        CONFIG / "sarcoma.yaml",
        cohort,
        weights,
        overwrite=True,
    )

    # Export patient-level overlap weights for effect summaries.
    build_postprocess_weights(
        [
            "--data", str(cohort / "person_period.csv"),
            "--training-config", str(weights / "config.yaml"),
            "--output-dir", str(args.output_dir / "postprocess_weights"),
        ]
    )


if __name__ == "__main__":
    main()
