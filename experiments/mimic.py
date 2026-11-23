#!/usr/bin/env python3
"""Build the paper-aligned MIMIC-III cohort from restricted source data."""

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent / "semisynthetic"
sys.path.insert(0, str(HERE))

from cohort import (  # noqa: E402
    build_concepts,
    prepare_database,
    select_cohort,
    write_paper_pickles,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", required=True, type=Path)
    parser.add_argument("--mimic-code-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    # Build only the concepts required by the paper cohort.
    database = args.output_dir / "mimic3.duckdb"
    prepare_database(args.mimic_dir, database, args.mimic_code_dir)
    build_concepts(database, args.mimic_code_dir)

    # Select the cohort and export the two files used downstream.
    select_cohort(database, args.output_dir, count_tolerance_fraction=0.05)
    write_paper_pickles(args.mimic_dir, args.output_dir, chunksize=250_000)


if __name__ == "__main__":
    main()
