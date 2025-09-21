"""Synthetic data generation utilities driven by external schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import random

import numpy as np
import yaml

Number = float | int

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_SCHEMA_PATH = CONFIG_DIR / "synthetic.yaml"


@dataclass
class NumericSpec:
    name: str
    minimum: Number
    maximum: Number
    decimals: int = 2
    missing_rate: float = 0.0

    def sample(self, rng: random.Random, size: int) -> List[Optional[float]]:
        values = []
        for _ in range(size):
            if rng.random() < self.missing_rate:
                values.append(None)
                continue
            raw = rng.uniform(self.minimum, self.maximum)
            values.append(round(raw, self.decimals))
        return values


@dataclass
class IntegerSpec:
    name: str
    minimum: int
    maximum: int
    missing_rate: float = 0.0

    def sample(self, rng: random.Random, size: int) -> List[Optional[int]]:
        values = []
        for _ in range(size):
            if rng.random() < self.missing_rate:
                values.append(None)
                continue
            values.append(rng.randint(self.minimum, self.maximum))
        return values


@dataclass
class CategoricalSpec:
    name: str
    categories: List[str]
    missing_rate: float = 0.0

    def sample(self, rng: random.Random, size: int) -> List[Optional[str]]:
        values = []
        for _ in range(size):
            if rng.random() < self.missing_rate:
                values.append(None)
                continue
            values.append(rng.choice(self.categories))
        return values


@dataclass
class BernoulliSpec:
    name: str
    probability: float
    missing_rate: float = 0.0

    def sample(self, rng: random.Random, size: int) -> List[Optional[int]]:
        values = []
        for _ in range(size):
            if rng.random() < self.missing_rate:
                values.append(None)
                continue
            values.append(int(rng.random() < self.probability))
        return values


FEATURE_TYPES = {
    "numeric": NumericSpec,
    "integer": IntegerSpec,
    "categorical": CategoricalSpec,
    "bernoulli": BernoulliSpec,
}


def load_schema(path: str | Path | None = None) -> Dict[str, List[Any]]:
    schema_path = Path(path) if path else DEFAULT_SCHEMA_PATH
    with open(schema_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    def build_feature(entry: Dict[str, Any]) -> Any:
        kind = entry.get("kind")
        spec_cls = FEATURE_TYPES.get(kind)
        if spec_cls is None:
            raise ValueError(f"Unsupported feature kind: {kind}")
        params = {k: v for k, v in entry.items() if k != "kind"}
        return spec_cls(**params)

    return {
        "clinical": [build_feature(item) for item in raw.get("clinical", [])],
        "treatment": [build_feature(item) for item in raw.get("treatment", [])],
        "actual_treatment": [build_feature(item) for item in raw.get("actual_treatment", [])],
    }


def _sample_feature_matrix(features: List[Any], size: int, rng: random.Random) -> Dict[str, List[Optional[Any]]]:
    columns: Dict[str, List[Optional[Any]]] = {}
    for feat in features:
        columns[feat.name] = feat.sample(rng, size)
    return columns


def _build_patient_records(
    features: List[Any],
    patient_id: str,
    timesteps: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    columns = _sample_feature_matrix(features, timesteps, rng)
    records = []
    for t in range(timesteps):
        entry = {name: values[t] for name, values in columns.items()}
        entry["_id"] = patient_id
        entry["timestep"] = t
        records.append(entry)
    return records


def generate_patient(schema: Dict[str, List[Any]], patient_id: str, timesteps: int, rng: random.Random) -> Dict[str, Any]:
    clinical = _build_patient_records(schema.get("clinical", []), patient_id, timesteps, rng)
    treatment = _build_patient_records(schema.get("treatment", []), patient_id, timesteps, rng)
    actual = _build_patient_records(schema.get("actual_treatment", []), patient_id, timesteps, rng)
    return {
        "patient_id": patient_id,
        "clinical": clinical,
        "treatment": treatment,
        "actual_treatment": actual,
    }


def generate_dataset(
    num_patients: int,
    timesteps: int,
    seed: int | None = None,
    schema_path: str | Path | None = None,
    schema: Dict[str, List[Any]] | None = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    schema_def = schema or load_schema(schema_path)
    patients = [generate_patient(schema_def, f"P{idx:05d}", timesteps, rng) for idx in range(1, num_patients + 1)]
    return {"patients": patients}


def save_dataset(dataset: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)
