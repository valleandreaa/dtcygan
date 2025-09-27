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
