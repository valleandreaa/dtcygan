# dtcygan

## Setup
- Target Python version: 3.12.
- Create a Conda environment:
  ```bash
  conda create -n dtcygan-env python=3.12
  conda activate dtcygan-env
  ```
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```

## Generate synthetic data
The `generate_data.py` CLI writes a JSON dataset using the schema in `src/config/synthetic.yaml`.

Example (matches the VS Code launch configuration):

```bash
python generate_data.py \
  --patients 25 \
  --timesteps 5 \
  --output data/syntects.json \
  --schema src/config/synthetic.yaml
```

Arguments:

- `--patients`: number of patient records to simulate (default `25`).
- `--timesteps`: sequence length per patient (default `5`).
- `--seed`: RNG seed for reproducibility (default `42`).
- `--output`: destination JSON path (parent folders created, file overwritten).
- `--indent`: pretty-printing indent (`2` by default, use `0` for compact).
- `--schema`: path to the synthetic schema; omit to use the package default.

The script overwrites the output file if it already exists. Ensure you run it from the project root (or set `PYTHONPATH=src`) so the `dtcygan` package is importable.

## Train the temporal CycleGAN
Training expects both the schema (`synthetic.yaml`) and the feature specifications defined in `src/config/training.yaml` (numeric/one-hot/ordinal mappings for clinical and treatment features).

Example training run (matches the VS Code launch configuration):

```bash
python train.py \
  --config src/config/training.yaml \
  --synthetic-data data/syntects.json \
  --schema src/config/synthetic.yaml \
  --patients 64 \
  --timesteps 8 \
  --checkpoint-dir data/models
```

- `--config` points to the YAML containing model hyperparameters and feature specs.
- `--synthetic-data` can be omitted to regenerate data on the fly; provide it to reuse an existing dataset.
- `--schema` is only needed when the dataset is regenerated (default matches above).
- `--patients` / `--timesteps` govern on-the-fly data generation size.
- `--checkpoint-dir` defaults to `data/models`; adjust if you want checkpoints elsewhere.

Each epoch prints expanded loss diagnostics (generator/discriminator, cycle, identity, component adversarial terms). Checkpoint metadata records the feature spec so inference can apply identical preprocessing.

## Run the analysis suite
`analyze.py` rebuilds counterfactual trajectories from a checkpoint and produces CSV/PNG outputs (ITE tables, waterfall fans, three-endpoint trajectories, risk-averse summaries).

Basic invocation (mirrors the launch configuration):

```bash
python analyze.py \
  --dataset data/syntects.json \
  --checkpoint data/models/dtcygan_20250928_135339.pt \
  --output-dir imgs/analysis \
  --bootstrap 200 \
  --lambda-gmd 0.0
```

Key flags:

- `--dataset`: path to the synthetic cohort JSON (from `generate_data.py`).
- `--checkpoint`: generator checkpoint saved by `train.py`.
- `--output-dir`: destination folder for tables/figures (created if missing).
- `--bootstrap`: number of resamples for uncertainty bands; set `0` to disable.
- `--lambda-gmd`: dispersion weight for risk-averse scoring.
- `--seed`: override RNG seed used for counterfactual generation (defaults to the checkpoint seed).
- `--skip-histology` / `--skip-grade`: disable subgroup reporting.
- `--plot-config`: optional YAML overriding default plot styling; omit to use the builtin `src/config/analysis_plots.yaml`.

Run the command from the project root (or export `PYTHONPATH=src`) so the `dtcygan` package resolves correctly.

## Validate a checkpoint
`validate.py` rebuilds counterfactual trajectories for a compact subset of patients and emits evaluation CSVs/plots (calibration, endpoint metrics, etc.).

Typical usage:

```bash
python validate.py \
  --dataset path/to/dataset.json \
  --checkpoint path/to/checkpoint.pt \
  --output-dir imgs/validation
```

Important options:

- `--dataset`: dataset JSON aligned with the model feature specification.
- `--checkpoint`: generator checkpoint to score.
- `--output-dir`: directory for validation artifacts (created automatically).
- `--patients`: how many patients to sample for counterfactual evaluation (default `25`).
- `--timesteps`: sequence length used during validation (defaults to the training setting).
- `--seed`: seed controlling patient sampling and bootstrap draws.
- `--bootstrap`: number of resamples for confidence intervals (set `0` to skip).

As with the other CLIs, execute it from the project root (or set `PYTHONPATH=src`) so Python can import the `dtcygan` package.
