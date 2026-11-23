#!/usr/bin/env python3
"""Reproduce the preoperative-RT versus surgery sarcoma experiment."""

import argparse
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "sarcoma"
OUTPUT = PROJECT / "output" / "paper" / "sarcoma"
CONFIG = PROJECT / "config" / "sarcoma.yaml"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from all_patients import run_analysis  # noqa: E402
from missingness import prepare as prepare_missingness  # noqa: E402
from report import run as write_report  # noqa: E402
from training import run_study  # noqa: E402


SEED = 42
MISSINGNESS_RATES = (0.10, 0.30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    person_period = args.data_dir / "cohort" / "person_period.csv"
    weights = args.data_dir / "weights"
    config = weights / "config.yaml"
    longitudinal_weights = weights / "training_weights.csv"
    longitudinal_metadata = weights / "training_weights_metadata.json"
    postprocess = args.data_dir / "postprocess_weights"
    patient_weights = postprocess / "preop_two_arm_overlap_weights.csv"
    patient_metadata = (
        postprocess / "preop_two_arm_overlap_weights_metadata.json"
    )

    # Train the selected preoperative-RT model.
    reference = OUTPUT / "reference"
    rows = run_study(
        person_period,
        config,
        reference,
        longitudinal_weights,
        longitudinal_metadata,
        patient_weights,
        patient_metadata,
        epochs=100,
    )
    if len(rows) != 1:
        raise RuntimeError("Sarcoma training did not produce exactly one result")

    # Repeat the model after injecting 10% and 30% missing values.
    missing_data = OUTPUT / "missingness_data"
    prepare_missingness(
        person_period,
        config,
        missing_data,
        rates=MISSINGNESS_RATES,
        seed=SEED,
    )
    for rate in MISSINGNESS_RATES:
        label = f"missing_{int(rate * 100)}pct"
        run_study(
            missing_data / label / "person_period.csv",
            config,
            OUTPUT / label,
            longitudinal_weights,
            longitudinal_metadata,
            patient_weights,
            patient_metadata,
            epochs=100,
            result_label=label,
            comparison_summary=OUTPUT / "missingness_summary.csv",
        )

    # Create the final all-patient diagnostics and paper figures.
    run_dir = reference / "full" / "seed_42"
    checkpoint_dir = run_dir / "checkpoints"
    report_dir = run_dir / "postprocessing"
    all_patients = OUTPUT / "all_patient_diagnostics"
    run_analysis(
        checkpoint=checkpoint_dir,
        person_period_data=person_period,
        causal_weights=patient_weights,
        causal_weight_metadata=patient_metadata,
        output_dir=all_patients,
        device=None,
        make_figures=True,
    )
    write_report(report_dir, all_patients, person_period, OUTPUT / "report")


if __name__ == "__main__":
    main()
