# DTCyGAN

## Project Description

DTCyGAN is a dynamic treatment counterfactual generator tailored to longitudinal oncology cohorts. Unlike prior temporal GANs that treat therapies as static labels, DTCyGAN embeds the full treatment parameter vector at each time step, enabling multi‑arm counterfactual mapping within a single cycle. Cycle‑consistency is enforced jointly over treatment specifications and outcomes, compelling the generator to reproduce the unobserved factual path and regularizing counterfactual rollouts. Collectively, these components empower DTCyGAN to generate treatment trajectories that remain faithful to observed physiology while exploring plausible therapeutic alternatives—an essential capability for treatment‑effect prediction.

### Innovations

* **Treatment‑aware temporal modeling.** DTCyGAN conditions on the complete treatment vector at every time step, supporting many‑to‑many mappings between therapeutic regimens and physiological responses within a single cycle.
* **Joint cycle‑consistency.** Treatment specifications and outcomes are tied together during training, forcing the generator to reconstruct the observed trajectory from counterfactual predictions and improving realism in unobserved scenarios.

### Individualized Treatment Effects

We compute patient‑level effects across multiple treatment arms learned during training. For each patient and time horizon, we compare predicted outcome probabilities under a candidate arm versus a reference arm and report the difference as the individualized treatment effect (positive means lower adverse‑event risk than the reference). Optionally, we summarize effects over time using trapezoidal aggregation. All metrics are computed per‑patient and per‑arm.

### Validation Strategy

Evaluating counterfactual generators is difficult because alternative trajectories are never observed once a patient receives a therapy. DTCyGAN adopts a dual‑track validation scheme that blends empirical consistency checks grounded in observable data with a counterfactual risk estimator based on influence functions. Together, these diagnostics deliver qualitative sanity checks and a single scalar error metric that can be compared across models.

#### Empirical Consistency Checks

1. **Physiological and protocol feasibility.** Generated sequences are screened for violations of physiological limits (e.g., negative tumor volume, hematological parameters outside viable ranges) and dosing schedules that contradict sarcoma guidelines. The proportion of failing sequences establishes a hard plausibility bound.
2. **Natural experiments.** When the data include patients who actually received an alternative therapy, we compare their outcomes with counterfactual predictions for matched patients who received the original therapy. Wasserstein‑1 distance and two‑sample Kolmogorov–Smirnov p‑values quantify concordance.
3. **Marginal distribution alignment.** We measure Kullback–Leibler divergence between synthetic treatment–outcome pairs and the observed treatment–outcome pairs. Large shifts flag calibration issues even when individual trajectories appear realistic.

#### Influence‑Function Risk Estimation

Empirical checks alone lack a unified scalar score and can fail when alternative arms are sparse. We therefore report an influence‑function–based risk estimate that adjusts the empirical risk with a first‑order correction derived from each training point’s sensitivity. Under mild assumptions, this estimate is consistent for the true counterfactual risk while requiring no unobserved outcomes. We summarize the score with nonparametric bootstrap 95% confidence intervals.

## Getting Started

### Setup

* Target Python version: 3.12.
* Create and activate a Conda environment:

```bash
conda create -n dtcygan-env python=3.12
conda activate dtcygan-env
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

### Generate Synthetic Data

`generate_data.py` writes a JSON dataset using the schema in `src/config/synthetic.yaml`.

Example:

```bash
python generate_data.py \
  --patients 25 \
  --timesteps 5 \
  --output data/synthetic.json \
  --schema src/config/synthetic.yaml
```

Key arguments:

* `--patients`: number of patient records to simulate (default `25`).
* `--timesteps`: sequence length per patient (default `5`).
* `--seed`: RNG seed for reproducibility (default `42`).
* `--output`: destination JSON path (parent folders created, file overwritten).
* `--indent`: pretty‑printing indent (`2` by default, use `0` for compact output).
* `--schema`: path to the synthetic schema; omit to use the package default.

Run the command from the project root (or set `PYTHONPATH=src`) so the `dtcygan` package is importable.

### Train the Temporal CycleGAN

Training expects both the schema (`synthetic.yaml`) and the feature specifications defined in `src/config/training.yaml`.

Example training run:

```bash
python train.py \
  --config src/config/training.yaml \
  --synthetic-data data/synthetic.json \
  --schema src/config/synthetic.yaml \
  --patients 64 \
  --timesteps 8 \
  --checkpoint-dir data/models
```

Important flags:

* `--config`: YAML file containing model hyperparameters and feature specs.
* `--synthetic-data`: reuse an existing dataset; omit to regenerate data on the fly.
* `--schema`: required when regenerating data (default matches above).
* `--patients` / `--timesteps`: control on‑the‑fly data generation size.
* `--checkpoint-dir`: defaults to `data/models`; adjust for alternative checkpoint locations.

Each epoch prints expanded loss diagnostics, and checkpoint metadata records the feature spec for consistent preprocessing during inference.

### Run the Analysis Suite

`analyze.py` rebuilds counterfactual trajectories from a checkpoint and produces CSV/PNG outputs (ITE tables, waterfall fans, multi‑endpoint trajectories, risk‑averse summaries).

Basic invocation:

```bash
python analyze.py \
  --dataset data/synthetic.json \
  --checkpoint data/models/dtcygan_20250928_135339.pt \
  --output-dir imgs/analysis \
  --bootstrap 200 \
  --lambda-gmd 0.0
```

Key flags:

* `--dataset`: path to the synthetic cohort JSON.
* `--checkpoint`: generator checkpoint saved by `train.py`.
* `--output-dir`: destination folder for tables and figures (created if missing).
* `--bootstrap`: number of resamples for uncertainty bands; set `0` to disable.
* `--lambda-gmd`: dispersion weight for risk‑averse scoring.
* `--seed`: override the RNG seed used for counterfactual generation (defaults to the checkpoint seed).
* `--skip-histology` / `--skip-grade`: disable subgroup reporting.
* `--plot-config`: optional YAML overriding default plot styling; omit to use `src/config/analysis_plots.yaml`.

Run the command from the project root (or export `PYTHONPATH=src`) so the `dtcygan` package resolves correctly.

### Validate a Checkpoint

`validate.py` rebuilds counterfactual trajectories for a compact subset of patients and emits evaluation CSVs and plots.

Example usage:

```bash
python validate.py \
  --dataset path/to/dataset.json \
  --checkpoint path/to/checkpoint.pt \
  --output-dir imgs/validation
```

Useful options:

* `--dataset`: dataset JSON aligned with the model feature specification.
* `--checkpoint`: generator checkpoint to score.
* `--output-dir`: directory for validation artifacts (created automatically).
* `--patients`: number of patients sampled for counterfactual evaluation (default `25`).
* `--timesteps`: sequence length used during validation (defaults to the training setting).
* `--seed`: seed controlling patient sampling and bootstrap draws.
* `--bootstrap`: number of resamples for confidence intervals (set `0` to skip).

As with the other CLIs, execute the command from the project root (or set `PYTHONPATH=src`) so Python can import the `dtcygan` package.
