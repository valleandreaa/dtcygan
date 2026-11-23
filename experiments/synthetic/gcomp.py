"""Longitudinal g-computation reference for the fully synthetic T4 benchmark.

The estimator sees only factual treatment/outcome records and the observed
clinical-state mask. Potential outcomes and complete-regime trajectories are
used after fitting, exclusively for held-out evaluation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class GComputationResult:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


def _patients(data: Mapping[Any, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    patients = list(data.values())
    if not patients:
        raise ValueError("Synthetic benchmark is empty")
    return patients


def patient_split(n_patients: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if n_patients < 5:
        raise ValueError("At least five patients are required")
    n_validation = max(1, int(0.20 * n_patients))
    n_validation = min(n_validation, n_patients - 1)
    order = torch.randperm(
        n_patients, generator=torch.Generator().manual_seed(seed)
    ).numpy()
    n_training = n_patients - n_validation
    return order[:n_training], order[n_training:]


def explicit_split_indices(dataset_path, n_patients, evaluation_split="test"):
    """Read training and evaluation rows from the exported split labels."""

    payload = json.loads(Path(dataset_path).read_text())
    patients = payload["patients"]
    if len(patients) != n_patients:
        raise ValueError("Exported split labels do not match the synthetic cohort")

    evaluation = np.asarray(
        [
            index
            for index, patient in enumerate(patients)
            if patient["split"] == evaluation_split
        ]
    )
    training = np.asarray(
        [
            index
            for index, patient in enumerate(patients)
            if patient["split"] != evaluation_split
        ]
    )
    return training, evaluation


def _design_rows(
    patients: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    timesteps: int | None = None,
    timestamps: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    outcomes: list[float] = []
    next_states: list[np.ndarray] = []
    next_masks: list[np.ndarray] = []
    row_keys: list[tuple[int, int]] = []
    selected_timestamps = None if timestamps is None else tuple(timestamps)
    for patient_index in indices:
        patient = patients[int(patient_index)]
        state = np.asarray(patient["x"], dtype=float)
        mask = np.asarray(patient.get("x_mask", np.ones_like(state)), dtype=float)
        static = np.asarray(patient["x_static"], dtype=float)
        treatment = np.asarray(patient["a"], dtype=int)
        factual = np.asarray(patient["y"], dtype=float)[0]
        horizon = state.shape[1] if timesteps is None else int(timesteps)
        if horizon <= 0 or horizon > state.shape[1]:
            raise ValueError(
                f"timesteps must lie in [1, {state.shape[1]}]; got {horizon}"
            )
        if len(treatment) < horizon or len(factual) < horizon:
            raise ValueError("Treatment and outcome arrays must cover the requested horizon")
        timepoints = range(horizon) if selected_timestamps is None else selected_timestamps
        if any(timestamp < 0 or timestamp >= horizon for timestamp in timepoints):
            raise ValueError(f"timestamps must lie in [0, {horizon - 1}]")
        for timestamp in timepoints:
            observed_state = np.where(mask[:, timestamp] > 0, state[:, timestamp], np.nan)
            previous = float(treatment[timestamp - 1]) if timestamp else 0.0
            history = np.asarray(
                [previous, treatment[:timestamp].sum(), timestamp / max(state.shape[1] - 1, 1)],
                dtype=float,
            )
            features.append(np.concatenate((observed_state, static, history)))
            outcomes.append(float(factual[timestamp]))
            row_keys.append((int(patient_index), timestamp))
            if timestamp + 1 < horizon:
                next_states.append(state[:, timestamp + 1])
                next_masks.append(mask[:, timestamp + 1])
            else:
                next_states.append(np.full(state.shape[0], np.nan))
                next_masks.append(np.zeros(state.shape[0]))
    return (
        np.asarray(features),
        np.asarray(outcomes),
        np.asarray(next_states),
        np.asarray(next_masks),
        np.asarray(row_keys, dtype=int),
    )


def _effect_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(truth)
    truth_flat = np.asarray(truth).reshape(-1)
    prediction_flat = np.asarray(prediction).reshape(-1)
    pehe = float(np.sqrt(np.mean(np.square(error))))
    truth_std = float(np.std(truth_flat))
    correlation = (
        float(np.corrcoef(truth_flat, prediction_flat)[0, 1])
        if np.std(truth_flat) > 0 and np.std(prediction_flat) > 0
        else float("nan")
    )
    return {
        "pehe": pehe,
        "normalized_pehe": pehe / truth_std if truth_std > 0 else 0.0,
        "true_ate": float(np.mean(truth)),
        "predicted_ate": float(np.mean(prediction)),
        "ate_absolute_error": float(abs(np.mean(error))),
        "ite_correlation": correlation,
        "effect_sign_accuracy": float(
            np.mean(np.sign(truth_flat) == np.sign(prediction_flat))
        ),
    }


def fit_g_computation(
    data: Mapping[Any, Mapping[str, Any]],
    seed: int = 42,
    n_estimators: int = 120,
    train_indices: Sequence[int] | None = None,
    evaluation_indices: Sequence[int] | None = None,
    timesteps: int | None = None,
    evaluation_timestamps: Sequence[int] | None = None,
    evaluation_split: str = "validation",
) -> GComputationResult:
    patients = _patients(data)
    if (train_indices is None) != (evaluation_indices is None):
        raise ValueError(
            "train_indices and evaluation_indices must be supplied together"
        )
    if train_indices is None:
        train_indices_array, evaluation_indices_array = patient_split(
            len(patients), seed
        )
    else:
        train_indices_array = np.asarray(train_indices, dtype=int)
        evaluation_indices_array = np.asarray(evaluation_indices, dtype=int)
        if len(train_indices_array) == 0 or len(evaluation_indices_array) == 0:
            raise ValueError("Explicit training and evaluation splits cannot be empty")
        all_indices = np.concatenate(
            (train_indices_array, evaluation_indices_array)
        )
        if np.any(all_indices < 0) or np.any(all_indices >= len(patients)):
            raise ValueError("Explicit split contains an out-of-range patient index")
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("Training and evaluation patient splits must not overlap")

    train_x, train_y, train_next, train_next_mask, train_keys = _design_rows(
        patients, train_indices_array, timesteps=timesteps
    )
    evaluation_x, _, _, _, evaluation_keys = _design_rows(
        patients,
        evaluation_indices_array,
        timesteps=timesteps,
        timestamps=evaluation_timestamps,
    )
    # Current treatment is appended only for the outcome model. Treatment is
    # deliberately absent from the history design until this point.
    train_treatment = np.asarray(
        [patients[i]["a"][t] for i, t in train_keys],
        dtype=float,
    )
    outcome_model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    outcome_model.fit(np.column_stack((train_x, train_treatment)), train_y)

    x0 = np.column_stack((evaluation_x, np.zeros(len(evaluation_x))))
    x1 = np.column_stack((evaluation_x, np.ones(len(evaluation_x))))
    predicted_y0 = outcome_model.predict(x0)
    predicted_y1 = outcome_model.predict(x1)
    truth_y0 = np.asarray(
        [patients[i]["y0"][t] for i, t in evaluation_keys], dtype=float
    )
    truth_y1 = np.asarray(
        [patients[i]["y1"][t] for i, t in evaluation_keys], dtype=float
    )
    point_truth = truth_y1 - truth_y0
    point_prediction = predicted_y1 - predicted_y0

    transition_models = []
    transition_training_x = np.column_stack((train_x, train_treatment))
    for feature_index in range(train_next.shape[1]):
        observed = train_next_mask[:, feature_index] > 0
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(
                n_estimators=max(40, n_estimators // 2),
                min_samples_leaf=5,
                random_state=seed + feature_index + 1,
                n_jobs=-1,
            ),
        )
        model.fit(
            transition_training_x[observed],
            train_next[observed, feature_index],
        )
        transition_models.append(model)

    evaluation_patients = [patients[int(index)] for index in evaluation_indices_array]
    statics = np.stack(
        [np.asarray(patient["x_static"], dtype=float) for patient in evaluation_patients]
    )
    initial_states = np.stack(
        [np.asarray(patient["x"], dtype=float)[:, 0] for patient in evaluation_patients]
    )
    available_timesteps = np.asarray(evaluation_patients[0]["x"]).shape[1]
    evaluation_timesteps = (
        available_timesteps if timesteps is None else int(timesteps)
    )
    scoring_timestamps = (
        np.arange(evaluation_timesteps, dtype=int)
        if evaluation_timestamps is None
        else np.asarray(evaluation_timestamps, dtype=int)
    )
    if (
        scoring_timestamps.size == 0
        or np.any(scoring_timestamps < 0)
        or np.any(scoring_timestamps >= evaluation_timesteps)
    ):
        raise ValueError(
            f"evaluation_timestamps must be non-empty and lie in [0, {evaluation_timesteps - 1}]"
        )
    regime_predictions: dict[int, np.ndarray] = {}
    regime_truth: dict[int, np.ndarray] = {}
    for regime in (0, 1):
        state = initial_states.copy()
        previous = np.zeros(len(evaluation_patients), dtype=float)
        cumulative = np.zeros(len(evaluation_patients), dtype=float)
        predictions = np.zeros(
            (len(evaluation_patients), evaluation_timesteps), dtype=float
        )
        for timestamp in range(evaluation_timesteps):
            history = np.column_stack(
                (
                    previous,
                    cumulative,
                    np.full(
                        len(evaluation_patients),
                        timestamp / max(available_timesteps - 1, 1),
                    ),
                )
            )
            design = np.column_stack((state, statics, history))
            treated_design = np.column_stack(
                (design, np.full(len(evaluation_patients), regime))
            )
            predictions[:, timestamp] = outcome_model.predict(treated_design)
            if timestamp + 1 < evaluation_timesteps:
                state = np.column_stack(
                    [model.predict(treated_design) for model in transition_models]
                )
            previous.fill(float(regime))
            cumulative += float(regime)
        regime_predictions[regime] = predictions
        regime_truth[regime] = np.stack(
            [
                np.asarray(patient[f"regime_y{regime}"], dtype=float)
                for patient in evaluation_patients
            ]
        )[:, :evaluation_timesteps]

    predicted_regime_effect = regime_predictions[1] - regime_predictions[0]
    truth_regime_effect = regime_truth[1] - regime_truth[0]
    point_metrics = _effect_metrics(point_truth, point_prediction)
    regime_metrics = _effect_metrics(
        truth_regime_effect[:, scoring_timestamps],
        predicted_regime_effect[:, scoring_timestamps],
    )
    rows = [
        {
            "patient_index": int(patient_index),
            "timestamp": int(timestamp),
            "truth_ite": float(truth),
            "predicted_ite": float(prediction),
        }
        for (patient_index, timestamp), truth, prediction in zip(
            evaluation_keys, point_truth, point_prediction
        )
    ]
    return GComputationResult(
        summary={
            "method": "longitudinal_g_computation",
            "seed": seed,
            "evaluation_split": str(evaluation_split),
            "timesteps": int(evaluation_timesteps),
            "evaluation_timestamps": [int(timestamp) for timestamp in scoring_timestamps],
            "n_train_patients": int(len(train_indices_array)),
            "n_evaluation_patients": int(len(evaluation_indices_array)),
            "n_validation_patients": int(len(evaluation_indices_array))
            if evaluation_split == "validation"
            else 0,
            "n_test_patients": int(len(evaluation_indices_array))
            if evaluation_split == "test"
            else 0,
            "explicit_patient_split": bool(train_indices is not None),
            "point_treatment": point_metrics,
            "dynamic_regime": regime_metrics,
            "truth_used_for_training": False,
        },
        rows=rows,
    )


