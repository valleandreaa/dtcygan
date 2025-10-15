# DTCyGAN

## Project Description

DTCyGAN is a dynamic treatment counterfactual generator tailored to longitudinal oncology cohorts. Unlike prior temporal GANs that treat therapies as static labels, DTCyGAN embeds the full treatment parameter vector at each time step, enabling multi‑arm counterfactual mapping within a single cycle. Cycle‑consistency is enforced jointly over treatment specifications and outcomes, compelling the generator to reproduce the unobserved factual path and regularizing counterfactual rollouts. Collectively, these components empower DTCyGAN to generate treatment trajectories that remain faithful to observed physiology while exploring plausible therapeutic alternatives—an essential capability for treatment‑effect prediction.

### Innovations

* **Treatment‑aware temporal modeling.** DTCyGAN conditions on the complete treatment vector at every time step, supporting many‑to‑many mappings between therapeutic regimens and physiological responses within a single cycle.
* **Joint cycle‑consistency.** Treatment specifications and outcomes are tied together during training, forcing the generator to reconstruct the observed trajectory from counterfactual predictions and improving realism in unobserved scenarios.

### Individualized Treatment Effects

Following dynamic individualized treatment effect (ITE) formulations for time‑series health data, DTCyGAN computes patient‑level effects across multiple arms learned during training. For patient $i$, endpoint $e$, horizon $t$, and treatment arms $s$ and reference $r$:

$$
\Delta_{i,e,t}(s,\mathrm{vs},r) = \hat{p}^{(s)}*{i,e,t} - \hat{p}^{(r)}*{i,e,t},\qquad
B_{i,e,t}(s,\mathrm{vs},r) = \Delta_{i,e,t}(s,\mathrm{vs},r).
$$

A reduction in adverse‑event risk relative to the reference yields a positive effect. Over discrete horizons $\mathcal{T}$, an optional time‑aggregated ITE can be reported via the trapezoidal rule. All quantities are computed per patient and per arm.

### Validation Strategy

Evaluating counterfactual generators is difficult because alternative trajectories are never observed once a patient receives a therapy. DTCyGAN adopts a dual‑track validation scheme that blends empirical consistency checks grounded in observable data with a counterfactual risk estimator based on influence functions. Together, these diagnostics deliver qualitative sanity checks and a single scalar error metric that can be compared across models.

#### Empirical Consistency Checks

1. **Physiological and protocol feasibility.** Generated sequences are screened for violations of physiological limits (e.g., negative tumor volume, hematological parameters outside viable ranges) and dosing schedules that contradict sarcoma guidelines. The proportion of failing sequences establishes a hard plausibility bound.
2. **Natural experiments.** When observational data include patients who received an alternative therapy $\tilde{\mathbf{T}}$, their outcomes are compared with counterfactual predictions for matched patients on $\mathbf{T}$. Wasserstein‑1 distance and two‑sample Kolmogorov–Smirnov $p$‑values quantify concordance.
3. **Marginal distribution alignment.** Kullback–Leibler divergence is measured between DTCyGAN’s synthetic scenarios $(\tilde{\mathbf{T}},\tilde{\mathbf{Y}})$ and the empirical distribution of $(\mathbf{T},\mathbf{Y})$. Large shifts flag calibration issues even when individual trajectories appear realistic.

#### Influence‑Function Risk Estimation

Empirical checks alone lack a unified scalar score and can fail when alternative treatment arms are sparse. Influence‑function (IF) validation addresses this by linearizing the risk around the empirical distribution $\hat{P}_{n}$:

$$
\hat{R}*{n}(\theta) = \frac{1}{n}\sum*{i=1}^{n} \ell(\theta; z_{i}),\qquad
\hat{R}^{\mathrm{IF}}*{n}(\theta) = \hat{R}*{n}(\theta) + \frac{1}{n}\sum_{i=1}^{n} \psi_{\theta}(z_{i}),
$$
with influence term
$$
\psi_{\theta}(z) = \mathbb{E}*{\hat{P}*{n}}\big[\nabla_{z},\ell(\theta; z)\big]^{\top},(z - \hat{\mu}).
$$

Under mild regularity conditions, $\hat{R}^{\mathrm{IF}}*{n}$ is $\sqrt{n}$‑consistent for the true counterfactual risk while requiring no unobserved outcomes. Reported metrics include $\hat{R}^{\mathrm{IF}}*{n}$ with nonparametric bootstrap 95% confidence intervals.

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
