#!/usr/bin/env python3
"""Reproduce the fully synthetic results reported in the paper."""

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "synthetic"
OUTPUT = PROJECT / "output" / "paper" / "synthetic"
CONFIG = PROJECT / "config" / "synthetic.yaml"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT))

from pipeline import run_grid  # noqa: E402
from ablation import run_ablation  # noqa: E402


SEED = 2027
EPOCHS = 100
EVALUATION_START = 43
SCENARIOS = {
    "reference",
    "sign_changing_effect",
    "missing_10pct",
    "missing_30pct",
    "missing_50pct",
    "missing_70pct",
}


def main():
    import yaml

    settings = yaml.safe_load(CONFIG.read_text())

    # Run every synthetic scenario used in the paper.
    run_grid(
        CONFIG,
        OUTPUT,
        run_dtcygan=True,
        epochs=EPOCHS,
        max_runs=None,
        scenario_names=SCENARIOS,
        seeds={SEED},
    )

    # Ablations reuse the reference dataset and its external oracle truth.
    reference = OUTPUT / "reference" / f"seed_{SEED}"
    required = (
        reference / "dtcygan.json",
        reference / "oracle_truth.json",
        reference / "dtcygan.yaml",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Synthetic preparation is incomplete: {missing}")

    run_ablation(
        required[0],
        required[1],
        required[2],
        {
            "seeds": [settings["seed"]],
            "variants": settings["ablations"],
        },
        OUTPUT / "ablation",
        epochs=EPOCHS,
        evaluate_test=True,
        evaluation_start=EVALUATION_START,
    )


if __name__ == "__main__":
    main()
