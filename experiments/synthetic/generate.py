"""Generate a reproducible temporal causal dataset with known potential outcomes.

This generator is intentionally independent of MIMIC.  It provides a larger
fully synthetic benchmark while access to the clinical database is pending.
The time-varying state is affected by past treatment, treatment assignment is
confounded by the observed history, and both current-treatment potential
outcomes are retained at every timestamp.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def generate_temporal_data(
    n_patients: int = 5000,
    timesteps: int = 50,
    n_temporal: int = 20,
    n_static: int = 5,
    seed: int = 42,
    treatment_intercept: float = -0.20,
    confounding_strength: float = 1.0,
    overlap_floor: float = 0.05,
    missingness_rate: float = 0.0,
    effect_intercept: float = -0.45,
    effect_state_scale: float = 0.16,
    effect_static_scale: float = 0.10,
    effect_previous_scale: float = -0.08,
) -> dict[int, dict[str, Any]]:
    """Return patient records from a longitudinal structural data generator."""

    if n_patients <= 0 or timesteps <= 0:
        raise ValueError("n_patients and timesteps must be positive")
    if n_temporal < 4:
        raise ValueError("n_temporal must be at least 4")
    if n_static < 2:
        raise ValueError("n_static must be at least 2")
    if confounding_strength < 0:
        raise ValueError("confounding_strength must be non-negative")
    if not 0.0 < overlap_floor < 0.5:
        raise ValueError("overlap_floor must lie strictly between 0 and 0.5")
    if not 0.0 <= missingness_rate < 1.0:
        raise ValueError("missingness_rate must lie in [0, 1)")
    effect_parameters = np.asarray(
        [
            effect_intercept,
            effect_state_scale,
            effect_static_scale,
            effect_previous_scale,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(effect_parameters)):
        raise ValueError("Treatment-effect parameters must be finite")

    rng = np.random.default_rng(seed)
    static = rng.normal(size=(n_patients, n_static))

    static_to_state = rng.normal(0.0, 0.18, size=(n_static, n_temporal))
    treatment_to_state = rng.normal(0.0, 0.08, size=n_temporal)
    treatment_to_state[:4] += np.asarray([0.22, -0.18, 0.14, -0.10])
    outcome_weights = rng.normal(0.0, 0.12, size=n_temporal)
    outcome_weights[:4] += np.asarray([0.30, -0.22, 0.18, 0.12])
    static_outcome_weights = rng.normal(0.0, 0.15, size=n_static)

    states = np.zeros((n_patients, n_temporal, timesteps), dtype=np.float32)
    treatment = np.zeros((n_patients, timesteps), dtype=np.int8)
    propensity = np.zeros((n_patients, timesteps), dtype=np.float32)
    y0 = np.zeros((n_patients, timesteps), dtype=np.float32)
    y1 = np.zeros((n_patients, timesteps), dtype=np.float32)
    state_mask = np.ones((n_patients, n_temporal, timesteps), dtype=np.int8)

    states[:, :, 0] = (
        static @ static_to_state
        + rng.normal(0.0, 0.65, size=(n_patients, n_temporal))
    ).astype(np.float32)

    innovations = rng.normal(
        0.0, 0.30, size=(n_patients, n_temporal, max(timesteps - 1, 0))
    )
    outcome_noise = rng.normal(0.0, 0.18, size=(n_patients, timesteps))

    def treatment_probability(
        state: np.ndarray,
        previous: np.ndarray,
        time_fraction: float,
    ) -> np.ndarray:
        score = confounding_strength * (
            0.55 * state[:, 0]
            - 0.40 * state[:, 1]
            + 0.28 * state[:, 2]
            + 0.30 * static[:, 0]
            - 0.22 * static[:, 1]
            + 0.35 * previous
        )
        logit = treatment_intercept + score + 0.15 * np.sin(
            2.0 * np.pi * time_fraction
        )
        return np.clip(_sigmoid(logit), overlap_floor, 1.0 - overlap_floor)

    def potential_outcomes(
        state: np.ndarray,
        previous: np.ndarray,
        cumulative: np.ndarray,
        time_fraction: float,
        noise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        untreated = (
            state @ outcome_weights
            + static @ static_outcome_weights
            + 0.18 * cumulative
            + 0.25 * time_fraction
            + 0.12 * state[:, 0] * static[:, 0]
            + noise
        )
        treatment_effect = (
            effect_intercept
            + effect_state_scale * np.tanh(state[:, 0])
            + effect_static_scale * static[:, 0]
            + effect_previous_scale * previous
        )
        return untreated, untreated + treatment_effect

    def next_state(
        state: np.ndarray,
        assigned: np.ndarray,
        time_fraction: float,
        innovation: np.ndarray,
    ) -> np.ndarray:
        return (
            0.72 * state
            + 0.12 * (static @ static_to_state)
            + assigned[:, None] * treatment_to_state[None, :]
            + 0.05 * np.sin(2.0 * np.pi * time_fraction)
            + innovation
        )

    previous_treatment = np.zeros(n_patients, dtype=np.float32)
    cumulative_treatment = np.zeros(n_patients, dtype=np.float32)
    for timestamp in range(timesteps):
        state = states[:, :, timestamp].astype(np.float64)
        time_fraction = timestamp / max(timesteps - 1, 1)

        probability = treatment_probability(state, previous_treatment, time_fraction)
        assigned = rng.binomial(1, probability).astype(np.int8)
        treatment[:, timestamp] = assigned
        propensity[:, timestamp] = probability.astype(np.float32)

        untreated, treated = potential_outcomes(
            state,
            previous_treatment,
            cumulative_treatment,
            time_fraction,
            outcome_noise[:, timestamp],
        )
        y0[:, timestamp] = untreated.astype(np.float32)
        y1[:, timestamp] = treated.astype(np.float32)

        if timestamp + 1 < timesteps:
            states[:, :, timestamp + 1] = next_state(
                state,
                assigned,
                time_fraction,
                innovations[:, :, timestamp],
            ).astype(np.float32)

        previous_treatment = assigned.astype(np.float32)
        cumulative_treatment += previous_treatment

    factual = np.where(treatment == 1, y1, y0).astype(np.float32)
    counterfactual = np.where(treatment == 1, y0, y1).astype(np.float32)

    # Evaluate two complete dynamic regimes with the same baseline state and
    # exogenous noise. These trajectories provide ground truth for sequential
    # rollout evaluation; the point-treatment y0/y1 values above condition on
    # the observed history and answer a different estimand.
    regime_outcomes: dict[str, np.ndarray] = {}
    regime_states: dict[str, np.ndarray] = {}
    for label, assigned_value in (("always_0", 0), ("always_1", 1)):
        simulated_states = np.zeros_like(states)
        simulated_states[:, :, 0] = states[:, :, 0]
        simulated_outcomes = np.zeros((n_patients, timesteps), dtype=np.float32)
        previous = np.zeros(n_patients, dtype=np.float32)
        cumulative = np.zeros(n_patients, dtype=np.float32)
        assigned = np.full(n_patients, assigned_value, dtype=np.float32)
        for timestamp in range(timesteps):
            time_fraction = timestamp / max(timesteps - 1, 1)
            state = simulated_states[:, :, timestamp].astype(np.float64)
            untreated, treated = potential_outcomes(
                state,
                previous,
                cumulative,
                time_fraction,
                outcome_noise[:, timestamp],
            )
            simulated_outcomes[:, timestamp] = (
                treated if assigned_value else untreated
            ).astype(np.float32)
            if timestamp + 1 < timesteps:
                simulated_states[:, :, timestamp + 1] = next_state(
                    state,
                    assigned,
                    time_fraction,
                    innovations[:, :, timestamp],
                ).astype(np.float32)
            previous = assigned.copy()
            cumulative += assigned
        regime_outcomes[label] = simulated_outcomes
        regime_states[label] = simulated_states

    if missingness_rate > 0:
        # State-dependent missingness is introduced only in the observed input
        # mask. The complete latent state remains available as oracle truth.
        logits = -0.4 + 0.35 * np.abs(states[:, 0, :])
        base_probability = _sigmoid(logits)
        scaled_probability = np.clip(
            missingness_rate * base_probability / max(float(base_probability.mean()), 1e-8),
            0.0,
            0.95,
        )
        missing = rng.random(size=states.shape) < scaled_probability[:, None, :]
        state_mask[missing] = 0

    data: dict[int, dict[str, Any]] = {}
    for patient in range(n_patients):
        data[patient] = {
            "x": states[patient],
            "x_static": static[patient].astype(np.float32),
            "a": treatment[patient],
            "y": np.vstack((factual[patient], counterfactual[patient])),
            "y0": y0[patient],
            "y1": y1[patient],
            "ite": y1[patient] - y0[patient],
            "propensity": propensity[patient],
            "x_mask": state_mask[patient],
            "regime_y0": regime_outcomes["always_0"][patient],
            "regime_y1": regime_outcomes["always_1"][patient],
            "regime_x0": regime_states["always_0"][patient],
            "regime_x1": regime_states["always_1"][patient],
        }
    return data


def save_temporal_data(data: dict[int, dict[str, Any]], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


