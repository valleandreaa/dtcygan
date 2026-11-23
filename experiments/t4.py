#!/usr/bin/env python3
"""Clone the official T4 release and prepare the paper-specific working copy."""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def command(parts, directory=ROOT):
    print(" ".join(str(part) for part in parts), flush=True)
    subprocess.run([str(part) for part in parts], cwd=directory, check=True)


def prepare_t4(config):
    source = ROOT / config["directory"]
    work = ROOT / config["work_directory"]

    if not source.exists():
        # Keep the untouched official repository outside the paper outputs.
        command(["git", "clone", config["repository"], source])

    command(["git", "checkout", config["commit"]], source)
    command(["git", "status", "--porcelain"], source)

    # Apply paper adaptations only to a disposable working copy.
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(source, work, ignore=shutil.ignore_patterns(".git"))
    command(["git", "apply", ROOT / config["patch"]], work)
    return work


def train_t4(work, settings, data, header, split, output):
    output.mkdir(parents=True, exist_ok=True)
    pretrain = output / "pretrain.pt"
    model = output / "t4.pt"
    common = [
        "--epochs", settings["epochs"],
        "--max_stay", settings["max_stay"],
        "--pre_window", settings["prediction_window"],
        "--learning_rate", settings["learning_rate"],
        "--hid_size", settings["hidden_size"],
        "--batch_size", settings["batch_size"],
        "--optimizer", settings["optimizer"],
        "--data_dir", f"{data}/",
        "--header_path", header,
        "--split_manifest", split,
        "--dataset_name", settings["dataset"],
        "--seed", settings["seed"],
    ]
    if settings.get("static_features"):
        common += ["--static_features", *settings["static_features"]]

    # Reuse completed stages when a run is resumed.
    if not pretrain.exists():
        command([
            sys.executable, "pre_train.py", *common,
            "--save_model", pretrain,
            "--log_file", output / "pretrain.log",
        ], work / "model")

    if not model.exists():
        command([
            sys.executable, "main.py", *common,
            "--aug_ratio", settings["augmentation_ratio"],
            "--pretrained_model", pretrain,
            "--save_model", model,
            "--log_file", output / "main.log",
            "--external_dataset_name", "",
            "--output_mimic", output / "diagnostic",
            "--output_amsterdamdb", output / "diagnostic",
        ], work / "model")
    return model


def run_synthetic(work, settings):
    source = ROOT / "output/paper/synthetic/reference/seed_2027"
    data = ROOT / "output/t4/synthetic/data"
    output = ROOT / "output/t4/synthetic"
    # Translate the reference synthetic run into T4's input format.
    command([
        sys.executable, work / "simulation/prepare_t4_t50_benchmark.py",
        "--source-pickle", source / "synthetic.pkl",
        "--source-dataset", source / "dtcygan.json",
        "--source-truth", source / "oracle_truth.json",
        "--output-dir", data,
        "--dataset-name", settings["dataset"],
    ])
    # Train T4 and evaluate only on the held-out test patients.
    model = train_t4(
        work, settings, data, data / "header.pkl",
        data / "split_manifest.json", output,
    )
    command([
        sys.executable, work / "simulation/postprocess_t4_semisynthetic.py",
        "--data-dir", data,
        "--dataset-name", settings["dataset"],
        "--header", data / "header.pkl",
        "--static-features", *settings["static_features"],
        "--truth", data / "oracle_truth.json",
        "--split-manifest", data / "split_manifest.json",
        "--checkpoint", model,
        "--output-dir", output / "test",
        "--split", "test",
        "--max-stay", settings["max_stay"],
        "--pre-window", settings["prediction_window"],
    ])


def run_semisynthetic(work, settings):
    data = ROOT / "output/paper/semisynthetic"
    output = ROOT / "output/t4/semisynthetic"
    # Use the same split manifest as the DTCyGAN comparison.
    model = train_t4(
        work, settings, data, data / "header.pkl", data / "split.json", output,
    )
    command([
        sys.executable, work / "simulation/postprocess_t4_semisynthetic.py",
        "--data-dir", data,
        "--dataset-name", settings["dataset"],
        "--header", data / "header.pkl",
        "--truth", data / "source_truth.json",
        "--split-manifest", data / "split.json",
        "--checkpoint", model,
        "--output-dir", output / "test",
        "--split", "test",
        "--max-stay", settings["max_stay"],
        "--pre-window", settings["prediction_window"],
    ])


def main():
    config = yaml.safe_load((ROOT / "config/t4.yaml").read_text())
    work = prepare_t4(config)
    experiment = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if experiment not in ("synthetic", "semisynthetic"):
        raise SystemExit("usage: python experiments/t4.py [synthetic|semisynthetic]")

    if experiment == "synthetic":
        run_synthetic(work, config[experiment])
    else:
        run_semisynthetic(work, config[experiment])


if __name__ == "__main__":
    main()
