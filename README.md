# DTCyGAN paper reproduction

This repository contains the minimum workflows used to reproduce the final
experiments in the associated study. Scientific settings are frozen in
version-controlled YAML; incomplete or mismatched inputs fail immediately.

## Install

Python 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root.

## Fully synthetic

This runs the 5,000-patient, 50-timestep experiment, the sign-changing effect,
10/30/50/70% missingness, ablations, and g-computation.

```bash
python experiments/synthetic.py
```

Output: `output/paper/synthetic/`.

Run the fixed T4 comparator after the synthetic workflow has completed:

```bash
python experiments/t4.py synthetic
```

T4 output: `output/t4/synthetic/test/`.

## MIMIC-III semi-synthetic

MIMIC-III is restricted and is not distributed here. Build the fixed cohort
from your local MIMIC-III installation and the official MIMIC code repository:

```bash
python experiments/mimic.py \
  --mimic-dir /path/to/mimic-iii-clinical-database \
  --mimic-code-dir /path/to/mimic-code \
  --output-dir data/paper/mimic

python experiments/semisynthetic.py \
  --cohort-dir data/paper/mimic
```

Output: `output/paper/semisynthetic/`.

Run the aligned T4 test comparator after that workflow completes:

```bash
python experiments/t4.py semisynthetic
```

T4 output: `output/t4/semisynthetic/test/`.

## Sarcoma

The clinical timeline is controlled data and is not distributed. Authorized
users can build the 433-patient cohort and selected training weights, then run
the preoperative-RT plus surgery versus surgery model and its 10% and 30%
missingness analyses:

```bash
python experiments/sarcoma_data.py \
  --timeline /path/to/sarcoma_timeline.csv \
  --output-dir data/paper/sarcoma

python experiments/sarcoma.py --data-dir data/paper/sarcoma
```

Output: `output/paper/sarcoma/`.

The reproduction commands expose only paths for restricted inputs. All model
and experiment settings are stored under `config/` so scientific changes stay
visible and version controlled. T4 is cloned from its official repository at
the commit recorded in `config/t4.yaml`; the paper compatibility changes are
kept separately in `patches/t4-paper.patch`.
