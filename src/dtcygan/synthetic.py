"""Synthetic data generation utilities driven by external schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import random

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
    probabilities: Optional[List[float]] = None
    missing_rate: float = 0.0

    def __post_init__(self) -> None:
        self._normalized_probs: Optional[List[float]] = None
        if self.probabilities is None:
            return

        if len(self.probabilities) != len(self.categories):
            raise ValueError(
                f"Categorical feature '{self.name}' expected {len(self.categories)}"
                f" probabilities but received {len(self.probabilities)}"
            )

        total = float(sum(self.probabilities))
        if total <= 0:
            raise ValueError(
                f"Categorical feature '{self.name}' requires a positive sum of probabilities"
            )

        self._normalized_probs = [p / total for p in self.probabilities]

    def sample(self, rng: random.Random, size: int) -> List[Optional[str]]:
        values = []
        for _ in range(size):
            if rng.random() < self.missing_rate:
                values.append(None)
                continue
            if self._normalized_probs is None:
                values.append(rng.choice(self.categories))
            else:
                values.append(rng.choices(self.categories, weights=self._normalized_probs, k=1)[0])
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

_MODALITY_RULES = (
    {
        "prefix": "chemotherapy_",
        "admin_key": "chemotherapy_administered",
        "episode_key": "episodes.chemotherapy",
    },
    {
        "prefix": "radiotherapy_",
        "admin_key": "radiotherapy_administered",
        "episode_key": "episodes.radiotherapy",
    },
    {
        "prefix": "surgery_",
        "admin_key": "surgery_performed",
        "episode_key": "episodes.surgery",
    },
)


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
    *,
    static: bool = False,
) -> List[Dict[str, Any]]:
    if not features:
        return []

    sample_size = 1 if static else timesteps
    columns = _sample_feature_matrix(features, sample_size, rng)
    records = []
    for t in range(timesteps):
        if static:
            entry = {name: values[0] for name, values in columns.items()}
        else:
            entry = {name: values[t] for name, values in columns.items()}
        entry["_id"] = patient_id
        entry["timestep"] = t
        records.append(entry)
    return records


def _apply_treatment_dependencies(
    treatment_records: List[Dict[str, Any]],
    actual_records: List[Dict[str, Any]],
) -> None:
    for treatment_entry, actual_entry in zip(treatment_records, actual_records):
        for rule in _MODALITY_RULES:
            admin_value = treatment_entry.get(rule["admin_key"])

            if admin_value is None:
                episode_value: Optional[int] = None
                clear_dependents = True
            elif bool(admin_value):
                episode_value = 1
                clear_dependents = False
            else:
                episode_value = 0
                clear_dependents = True

            episode_key = rule["episode_key"]
            if episode_key in actual_entry:
                actual_entry[episode_key] = episode_value

            if clear_dependents:
                prefix = rule["prefix"]
                admin_key = rule["admin_key"]
                for key in list(treatment_entry.keys()):
                    if key.startswith(prefix) and key != admin_key:
                        treatment_entry[key] = None


def generate_patient(schema: Dict[str, List[Any]], patient_id: str, timesteps: int, rng: random.Random) -> Dict[str, Any]:
    clinical = _build_patient_records(
        schema.get("clinical", []), patient_id, timesteps, rng, static=True
    )
    treatment = _build_patient_records(schema.get("treatment", []), patient_id, timesteps, rng)
    actual = _build_patient_records(schema.get("actual_treatment", []), patient_id, timesteps, rng)
    if treatment and actual:
        _apply_treatment_dependencies(treatment, actual)
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
