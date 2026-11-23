#!/usr/bin/env python
"""Train and evaluate DTCyGAN component ablations on synthetic ground truth."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

SYNTHETIC = Path(__file__).resolve().parent / "synthetic"
if str(SYNTHETIC) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC))
from evaluate import run_postprocess


ROOT = Path(__file__).resolve().parents[1]


def run_command(command, cwd, env=None):
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def run_ablation(
    dataset,
    truth,
    base_config,
    matrix,
    output_dir,
    epochs,
    max_runs=None,
    evaluate_test=False,
    evaluation_batch_size=16,
    intervention_chunk_size=4,
    longitudinal_weights=None,
    longitudinal_weight_metadata=None,
    evaluation_start=0,
):
    dataset = dataset.resolve()
    truth = truth.resolve() if truth is not None else None
    base_config = base_config.resolve()
    output_dir = output_dir.resolve()
    longitudinal_weights = (
        longitudinal_weights.resolve() if longitudinal_weights is not None else None
    )
    longitudinal_weight_metadata = (
        longitudinal_weight_metadata.resolve()
        if longitudinal_weight_metadata is not None
        else None
    )
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    specification = matrix
    if isinstance(matrix, Path):
        specification = yaml.safe_load(matrix.read_text(encoding="utf-8"))
    jobs = [
        (name, overrides or {}, int(seed))
        for seed in specification.get("seeds", [42])
        for name, overrides in specification["variants"].items()
    ]
    if max_runs is not None:
        jobs = jobs[:max_runs]
    rows: list[dict[str, Any]] = []
    for name, overrides, seed in jobs:
        run_dir = output_dir / name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {**base, **overrides, "seed": seed, "epochs": epochs}
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        checkpoint_dir = run_dir / "checkpoints"
        report_dir = run_dir / "report"
        checkpoints = sorted(checkpoint_dir.glob("dtcygan_*.pt"))
        if not checkpoints:
            command = [
                sys.executable,
                "train.py",
                "--config",
                str(config_path),
                "--synthetic-data",
                str(dataset),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--seed",
                str(seed),
            ]
            if longitudinal_weights is not None:
                command.extend(("--longitudinal-weights", str(longitudinal_weights)))
            if longitudinal_weight_metadata is not None:
                command.extend(
                    ("--longitudinal-weight-metadata", str(longitudinal_weight_metadata))
                )
            run_command(
                command,
                cwd=ROOT,
            )
        else:
            print(f"REUSE {checkpoints[-1]}", flush=True)
        report_path = report_dir / "report.json"
        report_ready = report_path.is_file()
        if report_ready and evaluate_test:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            report_ready = "test" in existing_report.get("splits", {})
        if not report_ready:
            run_postprocess(
                dataset_path=dataset,
                output_dir=report_dir,
                checkpoint_dir=checkpoint_dir,
                batch_size=evaluation_batch_size,
                truth_path=truth,
                evaluate_test=evaluate_test,
                intervention_chunk_size=intervention_chunk_size,
                evaluation_start=evaluation_start,
            )
        else:
            print(f"REUSE {report_path}", flush=True)
        report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        split_name = "test" if evaluate_test else "validation"
        validation = report["splits"][split_name]
        rows.append(
            {
                "variant": name,
                "seed": seed,
                "evaluation_split": split_name,
                "architecture": config.get("architecture", "lstm"),
                "cycle_weight": config.get("cycle_weight", 0.0),
                "identity_weight": config.get("identity_weight", 0.0),
                "factual_weight": config.get("factual_weight", 0.0),
                "condition_weight": config.get("condition_weight", 0.0),
                "treatment_prediction": config.get(
                    "treatment_prediction", True
                ),
                "include_missingness_indicators": config.get(
                    "include_missingness_indicators", True
                ),
                "factual_outcome_rmse": validation["factual_outcome"]["rmse"],
                "factual_outcome_wasserstein": validation["factual_outcome"]["wasserstein"],
                "factual_outcome_r2": validation["factual_outcome"]["r2"],
                "potential_outcome_rmse": validation["potential_outcomes"]["rmse"],
                "potential_outcome_wasserstein": validation["potential_outcomes"][
                    "wasserstein"
                ],
                "trajectory_lag1_error": validation["temporal_fidelity"][
                    "lag1_correlation_absolute_error"
                ],
                "treatment_accuracy": validation["factual_treatment"]["accuracy"],
                "conditioning_adherence": validation["conditioning"][
                    "balanced_adherence"
                ],
                "pehe": validation["causal_effect"]["pehe"],
                "normalized_pehe": validation["causal_effect"]["normalized_pehe"],
                "ate_absolute_error": validation["causal_effect"]["ate_absolute_error"],
                "ite_correlation": validation["causal_effect"]["ite_correlation"],
                "effect_sign_accuracy": validation["causal_effect"][
                    "effect_sign_accuracy"
                ],
                "future_treatment_prediction_mae": validation["leakage_audit"][
                    "future_treatment_prediction_mae"
                ],
                "future_treatment_ite_mae": validation["leakage_audit"][
                    "future_treatment_ite_mae"
                ],
                "passes_no_anticipation": validation["leakage_audit"][
                    "passes_no_anticipation"
                ],
                "assessment": report["assessment"],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "ablation_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset.resolve()),
                "truth": str(truth.resolve()) if truth is not None else None,
                "base_config": str(base_config.resolve()),
                "matrix": str(matrix.resolve()),
                "epochs": epochs,
                "runs": len(rows),
                "evaluation_split": "test" if evaluate_test else "validation",
                "evaluation_batch_size": evaluation_batch_size,
                "intervention_chunk_size": intervention_chunk_size,
                "evaluation_start": evaluation_start,
                "longitudinal_weights": (
                    str(longitudinal_weights) if longitudinal_weights is not None else None
                ),
                "longitudinal_weight_metadata": (
                    str(longitudinal_weight_metadata)
                    if longitudinal_weight_metadata is not None
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


