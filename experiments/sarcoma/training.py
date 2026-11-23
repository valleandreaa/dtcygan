#!/usr/bin/env python
"""Train and postprocess resumable real-sarcoma DTCyGAN ablations."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from evaluate import run_postprocess


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration mappings; lists remain explicit replacements."""

    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _report_row(name: str, seed: int, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = report.get("test", report.get("validation", {}))
    factual = evaluation.get("factual", {})
    effects = report.get("effect_summary", [])
    effect = effects[0] if effects else {}
    balance = (
        report.get("causal_interpretation", {})
        .get(
            "test_effect_balance",
            report.get("causal_interpretation", {}).get(
                "validation_effect_balance", {}
            ),
        )
    )
    return {
        "variant": name,
        "seed": seed,
        "assessment": report.get("assessment"),
        "checkpoint": report.get("checkpoint"),
        "test_auroc": factual.get("auroc"),
        "test_average_precision": factual.get("average_precision"),
        "test_brier": factual.get("brier"),
        "test_predicted_rate": factual.get("predicted_rate"),
        "test_observed_rate": factual.get("observed_rate"),
        "risk_surgery_only": effect.get("risk_surgery_only"),
        "risk_preoperative_rt_surgery": effect.get("risk_preoperative_rt_surgery"),
        "risk_difference": effect.get("risk_difference"),
        "effect_support": effect.get("effect_support"),
        "test_maximum_absolute_smd_weighted": balance.get(
            "maximum_absolute_smd_weighted"
        ),
        "passes_t_time_censoring": report.get("leakage_audit", {}).get(
            "passes_t_time_censoring"
        ),
    }


def run_study(
    dataset: Path,
    base_config: Path,
    output_dir: Path,
    longitudinal_weights: Path,
    longitudinal_weight_metadata: Path,
    postprocess_weights: Path,
    postprocess_weight_metadata: Path,
    epochs: int | None = None,
    device: str | None = None,
    result_label: str | None = None,
    comparison_summary: Path | None = None,
) -> list[dict[str, Any]]:
    paths = [
        dataset,
        base_config,
        longitudinal_weights,
        longitudinal_weight_metadata,
        postprocess_weights,
        postprocess_weight_metadata,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    base = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    jobs = [("full", {}, int(base.get("seed", 42)))]

    rows: list[dict[str, Any]] = []
    for name, overrides, seed in jobs:
        run_dir = output_dir / name / f"seed_{seed}"
        checkpoint_dir = run_dir / "checkpoints"
        report_dir = run_dir / "postprocessing"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _deep_merge(base, overrides)
        config["seed"] = seed
        if epochs is not None:
            config["epochs"] = epochs
        temporal_contract = config.get("temporal_alignment", {}) or {}
        if (
            temporal_contract.get("contract_version") == 2
            and config.get("architecture", "lstm") != "lstm"
        ):
            raise ValueError(
                f"Variant {name!r} violates temporal contract v2: "
                "only the forward LSTM architecture is permitted"
            )
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        checkpoints = sorted(checkpoint_dir.glob("dtcygan_*.pt"))
        if not checkpoints:
            command = [
                sys.executable,
                "train.py",
                "--config",
                str(config_path.resolve()),
                "--person-period-data",
                str(dataset.resolve()),
                "--longitudinal-weights",
                str(longitudinal_weights.resolve()),
                "--longitudinal-weight-metadata",
                str(longitudinal_weight_metadata.resolve()),
                "--checkpoint-dir",
                str(checkpoint_dir.resolve()),
                "--seed",
                str(seed),
            ]
            _run(command)
        else:
            print(f"REUSE {checkpoints[-1]}", flush=True)

        report_path = report_dir / "report.json"
        if not report_path.is_file():
            run_postprocess(
                checkpoint_dir,
                dataset,
                report_dir,
                postprocess_weights,
                postprocess_weight_metadata,
                0,
                device,
            )
        else:
            print(f"REUSE {report_path}", flush=True)
        rows.append(_report_row(name, seed, report_path))

    if result_label is not None:
        for row in rows:
            row["scenario"] = result_label

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ablation_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "dataset": str(dataset.resolve()),
        "base_config": str(base_config.resolve()),
        "longitudinal_training_weights": str(longitudinal_weights.resolve()),
        "longitudinal_training_weight_metadata": str(
            longitudinal_weight_metadata.resolve()
        ),
        "patient_level_postprocess_weights": str(postprocess_weights.resolve()),
        "patient_level_postprocess_weight_metadata": str(
            postprocess_weight_metadata.resolve()
        ),
        "requested_variants": ["full"],
        "completed_runs": len(rows),
        "summary": str(summary_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if comparison_summary is not None:
        comparison_summary.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]] = []
        if comparison_summary.is_file():
            with comparison_summary.open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        if result_label is not None:
            existing = [row for row in existing if row.get("scenario") != result_label]
        combined = existing + rows
        if combined:
            fieldnames = list(dict.fromkeys(key for row in combined for key in row))
            with comparison_summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(combined)
    return rows


