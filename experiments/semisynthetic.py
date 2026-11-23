#!/usr/bin/env python3
"""Reproduce the aligned MIMIC-III semi-synthetic paper results."""

import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import yaml


PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "semisynthetic"
OUTPUT = PROJECT / "output" / "paper" / "semisynthetic"
CONFIG = PROJECT / "config" / "semisynthetic.yaml"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT))

from align import write_aligned_dataset  # noqa: E402
from export import export_files  # noqa: E402
from generate import simulate_mimic  # noqa: E402
from weights import write_weights  # noqa: E402
from gcomp import run as run_g_computation  # noqa: E402
from ablation import run_ablation  # noqa: E402


SEED = 101
SOURCE_DGP_SEED = 42
SPLIT_SEED = 62042
EPOCHS = 100
BASELINE_END = 8
EVALUATION_START = 9
EVALUATION_END = 15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(CONFIG.read_text())

    cohort = args.cohort_dir / "t4_paper_mimic3.pkl"
    header = args.cohort_dir / "t4_paper_header.pkl"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Generate factual outcomes and keep oracle outcomes in a separate file.
    simulated = OUTPUT / "cohort.pkl"
    local_header = OUTPUT / "header.pkl"
    source_truth = OUTPUT / "source_truth.json"
    split = OUTPUT / "split.json"
    simulate_mimic(
        args.cohort_dir,
        "t4_paper_mimic3",
        simulated,
        seed=SOURCE_DGP_SEED,
        header_path=header,
        truth_output=source_truth,
        split_output=split,
        split_seed=SPLIT_SEED,
    )
    shutil.copyfile(header, local_header)

    # Convert the cohort to the DTCyGAN sequence format.
    dataset = OUTPUT / "dtcygan.json"
    generated_config = OUTPUT / "generated.yaml"
    truth = OUTPUT / "truth.json"
    export_files(
        simulated,
        header,
        dataset,
        generated_config,
        timesteps=16,
        seed=SEED,
        split_seed=SPLIT_SEED,
        truth_output_path=truth,
        batch_size=32,
        source_truth_path=source_truth,
        split_manifest_path=split,
    )

    # Match the baseline and forecast windows used by T4.
    aligned = OUTPUT / "dtcygan_aligned.json"
    write_aligned_dataset(
        dataset,
        aligned,
        baseline_end=BASELINE_END,
        evaluation_start=EVALUATION_START,
        evaluation_end=EVALUATION_END,
        overwrite=True,
    )

    # Estimate longitudinal weights on the aligned training data.
    weights = OUTPUT / "weights.csv"
    weight_metadata = OUTPUT / "weights.json"
    write_weights(
        aligned,
        weights,
        weight_metadata,
        baseline_end=BASELINE_END,
        evaluation_start=EVALUATION_START,
        evaluation_end=EVALUATION_END,
        probability_clip=0.01,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )

    training = settings["training"]
    config = OUTPUT / "config.yaml"
    model_config = {
        "seq_len": settings["timesteps"],
        "batch_size": training["batch_size"],
        "epochs": training["epochs"],
        "generator_lr": training["generator_lr"],
        "discriminator_lr": training["discriminator_lr"],
        "beta1": training["beta1"],
        "beta2": training["beta2"],
        "cycle_weight": training["cycle_weight"],
        "identity_weight": training["identity_weight"],
        "factual_weight": training["factual_weight"],
        "condition_weight": training["condition_weight"],
        "generator_hidden": training["generator_hidden"],
        "discriminator_hidden": training["discriminator_hidden"],
        "layers": training["layers"],
        "seed": settings["training_seed"],
        "counterfactual_mode": "temporal_step_flip",
        "conditioning_feature": "temporal_treatment",
        "conditioning_contract": "time_local",
        "broadcast_actual_treatment": False,
        "require_explicit_splits": True,
        "include_missingness_indicators": training["missingness_indicators"],
        "causal_weighting": {
            "enabled": True,
            "method": settings["weighting"]["method"],
            "stage": "timestamps_9_15",
            "target_population": "paper_aligned_mimic3",
            "ineligible_policy": "exclude",
            "normalization": "mean_one",
            "require_supported": False,
            "apply_to": ["discriminator", "generator_adversarial", "cycle", "identity"],
        },
        "temporal_alignment": {
            "contract_version": 1,
            "contract": "t4_baseline_forecast",
            "baseline_end": settings["baseline_end"],
            "evaluation_start": settings["evaluation_start"],
            "evaluation_end": settings["evaluation_end"],
        },
        "clinical_features": {
            "numeric": settings["features"]["clinical"],
            "one_hot": [],
            "ordinal": {},
        },
        "treatment_features": {
            "numeric": settings["features"]["treatment"],
            "one_hot": [],
            "ordinal": {},
        },
    }
    config.write_text(yaml.safe_dump(model_config, sort_keys=False))

    # Train the paper model and its fixed ablations.
    run_ablation(
        aligned,
        truth,
        config,
        {
            "seeds": [settings["training_seed"]],
            "variants": settings["ablations"],
        },
        OUTPUT / "ablation",
        epochs=EPOCHS,
        evaluate_test=True,
        longitudinal_weights=weights,
        longitudinal_weight_metadata=weight_metadata,
        evaluation_start=EVALUATION_START,
    )

    # Fit the g-computation benchmark on the same split.
    run_g_computation(
        SimpleNamespace(
            input=str(simulated),
            header=str(header),
            truth=str(source_truth),
            split_manifest=str(split),
            output_dir=str(OUTPUT / "g_computation"),
            split="test",
            baseline_end=BASELINE_END,
            evaluation_start=EVALUATION_START,
            evaluation_end=EVALUATION_END,
            history_mode="baseline",
            estimators=120,
            min_samples_leaf=5,
            n_jobs=-1,
            seed=SOURCE_DGP_SEED,
            static_features=["agegroup", "heightgroup", "weightgroup", "gender"],
        )
    )


if __name__ == "__main__":
    main()
