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
