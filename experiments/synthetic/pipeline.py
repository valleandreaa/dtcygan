"""Run the fully synthetic reviewer robustness grid.

By default this executes data generation and longitudinal g-computation. Pass
``--run-dtcygan`` to train and evaluate DTCyGAN for every scenario/seed.
Patient-level data and checkpoints remain below the ignored results directory.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from export import export_files
from gcomp import explicit_split_indices, fit_g_computation
from generate import generate_temporal_data, save_temporal_data
from evaluate import run_postprocess


ROOT = Path(__file__).resolve().parent
DTCYGAN = ROOT.parents[1]


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, output)
    else:
        output[prefix] = value


def _effect_metrics(truth: list[float], prediction: list[float]) -> dict[str, Any]:
    if not truth or len(truth) != len(prediction):
        raise ValueError("Effect truth and prediction must be nonempty and aligned")
    errors = [estimate - oracle for oracle, estimate in zip(truth, prediction)]
    truth_mean = sum(truth) / len(truth)
    prediction_mean = sum(prediction) / len(prediction)
    truth_variance = sum((value - truth_mean) ** 2 for value in truth) / len(truth)
    prediction_variance = sum(
        (value - prediction_mean) ** 2 for value in prediction
    ) / len(prediction)
    if truth_variance > 0 and prediction_variance > 0:
        covariance = sum(
            (oracle - truth_mean) * (estimate - prediction_mean)
            for oracle, estimate in zip(truth, prediction)
        ) / len(truth)
        correlation = covariance / math.sqrt(truth_variance * prediction_variance)
    else:
        correlation = None
    return {
        "pehe": math.sqrt(sum(error**2 for error in errors) / len(errors)),
        "true_ate": truth_mean,
        "predicted_ate": prediction_mean,
        "ate_absolute_error": abs(sum(errors) / len(errors)),
        "ite_correlation": correlation,
        "effect_sign_accuracy": sum(
            (oracle > 0) == (estimate > 0) and (oracle < 0) == (estimate < 0)
            for oracle, estimate in zip(truth, prediction)
        )
        / len(truth),
    }


def diagnostic_baselines(
    data: dict[int, dict[str, Any]],
    train_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    timesteps: int,
) -> dict[str, Any]:
    train_by_timestep = [
        [float(data[int(index)]["ite"][timestamp]) for index in train_indices]
        for timestamp in range(timesteps)
    ]
    train_effects = [value for values in train_by_timestep for value in values]
    evaluation_effects = [
        float(data[int(index)]["ite"][timestamp])
        for index in evaluation_indices
        for timestamp in range(timesteps)
    ]
    evaluation_treatments = [
        int(data[int(index)]["a"][timestamp])
        for index in evaluation_indices
        for timestamp in range(timesteps)
    ]
    evaluation_propensities = sorted(
        float(data[int(index)]["propensity"][timestamp])
        for index in evaluation_indices
        for timestamp in range(timesteps)
    )
    observed_mask = [
        float(value)
        for index in evaluation_indices
        for row in data[int(index)].get("x_mask", [])
        for value in row[:timesteps]
    ]

    def quantile(values: list[float], probability: float) -> float:
        return values[round((len(values) - 1) * probability)]

    global_mean = sum(train_effects) / len(train_effects)
    timestep_means = [sum(values) / len(values) for values in train_by_timestep]
    timestep_prediction = [
        timestep_means[timestamp]
        for _ in evaluation_indices
        for timestamp in range(timesteps)
    ]
    negative_fraction = sum(value < 0 for value in evaluation_effects) / len(
        evaluation_effects
    )
    return {
        "n_evaluation_effects": len(evaluation_effects),
        "oracle_negative_fraction": negative_fraction,
        "oracle_positive_fraction": sum(value > 0 for value in evaluation_effects)
        / len(evaluation_effects),
        "observed_treatment_rate": sum(evaluation_treatments)
        / len(evaluation_treatments),
        "propensity_min": evaluation_propensities[0],
        "propensity_p05": quantile(evaluation_propensities, 0.05),
        "propensity_p95": quantile(evaluation_propensities, 0.95),
        "propensity_max": evaluation_propensities[-1],
        "observed_missing_fraction": (
            1.0 - sum(observed_mask) / len(observed_mask) if observed_mask else 0.0
        ),
        "zero_effect": _effect_metrics(
            evaluation_effects, [0.0] * len(evaluation_effects)
        ),
        "training_global_mean": _effect_metrics(
            evaluation_effects, [global_mean] * len(evaluation_effects)
        ),
        "training_timestep_mean": _effect_metrics(
            evaluation_effects, timestep_prediction
        ),
    }


def run_grid(
    config_path: Path,
    output_dir: Path,
    run_dtcygan: bool,
    epochs: int,
    max_runs: int | None,
    scenario_names: set[str] | None = None,
    seeds: set[int] | None = None,
) -> list[dict[str, Any]]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = {
        "patients": config["patients"],
        "timesteps": config["timesteps"],
        "temporal_features": config["temporal_features"],
        "static_features": config["static_features"],
        "missingness_rate": 0.0,
        **config["data"],
    }
    training_defaults = {
        "batch_size": config["training"]["batch_size"],
        "include_missingness_indicators": config["training"]["missingness_indicators"],
    }
    scenarios = [
        {"name": name, **values}
        for name, values in config["scenarios"].items()
    ]
    split_seed_offset = int(config.get("split_seed_offset", 60000))
    jobs = [
        (scenario, int(seed))
        for seed in [config["seed"]]
        for scenario in scenarios
        if (scenario_names is None or str(scenario["name"]) in scenario_names)
        and (seeds is None or int(seed) in seeds)
    ]
    if scenario_names is not None:
        configured = {str(row["name"]) for row in scenarios}
        unknown = scenario_names - configured
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    if max_runs is not None:
        jobs = jobs[:max_runs]
    rows: list[dict[str, Any]] = []
    for scenario, seed in jobs:
        # Each scenario gets an isolated, reproducible output directory.
        settings = {**defaults, **scenario}
        name = str(settings.pop("name"))
        training_overrides = {
            **training_defaults,
            **dict(settings.pop("training_overrides", {}) or {}),
        }
        run_dir = output_dir / name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pickle_path = run_dir / "synthetic.pkl"
        dataset_path = run_dir / "dtcygan.json"
        truth_path = run_dir / "oracle_truth.json"
        training_config = run_dir / "dtcygan.yaml"
        required_data_files = (pickle_path, dataset_path, truth_path, training_config)
        if all(path.is_file() for path in required_data_files):
            print(f"REUSE {pickle_path}", flush=True)
            with pickle_path.open("rb") as handle:
                data = pickle.load(handle)
        else:
            # Generate factual training data and external oracle truth together.
            data = generate_temporal_data(
                n_patients=int(settings["patients"]),
                timesteps=int(settings["timesteps"]),
                n_temporal=int(settings["temporal_features"]),
                n_static=int(settings["static_features"]),
                seed=seed,
                treatment_intercept=float(settings["treatment_intercept"]),
                confounding_strength=float(settings["confounding_strength"]),
                overlap_floor=float(settings["overlap_floor"]),
                missingness_rate=float(settings["missingness_rate"]),
                effect_intercept=float(settings["effect_intercept"]),
                effect_state_scale=float(settings["effect_state_scale"]),
                effect_static_scale=float(settings["effect_static_scale"]),
                effect_previous_scale=float(settings["effect_previous_scale"]),
            )
            save_temporal_data(data, pickle_path)
            export_files(
                pickle_path,
                dataset_path,
                training_config,
                timesteps=int(settings["timesteps"]),
                seed=seed,
                split_seed=split_seed_offset + seed,
                truth_output_path=truth_path,
                batch_size=int(training_overrides.get("batch_size", 8)),
            )

        # Fit simple benchmarks on exactly the same held-out patients.
        train_indices, test_indices = explicit_split_indices(
            dataset_path,
            n_patients=len(data),
            evaluation_split="test",
        )
        g_result = fit_g_computation(
            data,
            seed=seed,
            n_estimators=int(config.get("g_computation_estimators", 120)),
            train_indices=train_indices,
            evaluation_indices=test_indices,
            timesteps=int(settings["timesteps"]),
            evaluation_timestamps=range(
                int(config["evaluation_start"]),
                int(settings["timesteps"]),
            ),
            evaluation_split="test",
        )
        (run_dir / "g_computation.json").write_text(
            json.dumps(g_result.summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        baselines = diagnostic_baselines(
            data,
            train_indices,
            test_indices,
            timesteps=int(settings["timesteps"]),
        )
        (run_dir / "diagnostic_baselines.json").write_text(
            json.dumps(baselines, indent=2, allow_nan=False), encoding="utf-8"
        )
        row: dict[str, Any] = {
            "scenario": name,
            "seed": seed,
            "split_seed": split_seed_offset + seed,
            **settings,
        }
        _flatten("g_computation", g_result.summary, row)
        _flatten("diagnostic", baselines, row)

        if run_dtcygan:
            # Freeze scenario-specific settings before launching training.
            training_payload = yaml.safe_load(
                training_config.read_text(encoding="utf-8")
            )
            training_payload["epochs"] = int(epochs)
            training_payload.update(training_overrides)
            training_config.write_text(
                yaml.safe_dump(training_payload, sort_keys=False), encoding="utf-8"
            )
            checkpoint_dir = run_dir / "checkpoints"
            report_dir = run_dir / "dtcygan_report"
            checkpoints = sorted(checkpoint_dir.glob("dtcygan_*.pt"))
            if not checkpoints:
                _run(
                    [
                        sys.executable,
                        "train.py",
                        "--config",
                        str(training_config),
                        "--synthetic-data",
                        str(dataset_path),
                        "--checkpoint-dir",
                        str(checkpoint_dir),
                        "--seed",
                        str(seed),
                    ],
                    cwd=DTCYGAN,
                )
            else:
                print(f"REUSE {checkpoints[-1]}", flush=True)
            if not (report_dir / "report.json").is_file():
                run_postprocess(
                    dataset_path=dataset_path,
                    output_dir=report_dir,
                    checkpoint_dir=checkpoint_dir,
                    batch_size=config.get("evaluation_batch_size", 16),
                    truth_path=truth_path,
                    evaluate_test=True,
                    intervention_chunk_size=config.get("intervention_chunk_size", 4),
                    evaluation_start=config["evaluation_start"],
                )
            else:
                print(f"REUSE {report_dir / 'report.json'}", flush=True)

            report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
            _flatten("dtcygan", report["splits"]["test"], row)
            row["dtcygan.assessment"] = report["assessment"]
        rows.append(row)

    # Collect the scenario results in one paper-level summary.
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        columns = sorted({key for row in rows for key in row})
        with (output_dir / "robustness_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "run_dtcygan": run_dtcygan,
                "epochs": epochs if run_dtcygan else None,
                "runs": len(rows),
                "scenario_filter": sorted(scenario_names) if scenario_names else None,
                "seed_filter": sorted(seeds) if seeds else None,
                "evaluation_split": "test",
                "training_dataset_contains_truth": False,
                "external_oracle_truth": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


