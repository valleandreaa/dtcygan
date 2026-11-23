"""Retrospective prefix updating without future factual-state leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


@dataclass
class TransitionBundle:
    model: Any
    clinical_dim: int


@dataclass
class SarcomaTransitionBundle:
    """Training-only models for baseline-to-state and subsequent state transitions."""

    initial_model: Any
    transition_model: Any
    baseline_dim: int
    state_dim: int
    treatment_dim: int
    condition_dim: int = 1


def undo_positive_class_weight(
    probability: torch.Tensor,
    positive_weight: float | None,
) -> torch.Tensor:
    """Map weighted-BCE probabilities back to the unweighted event scale.

    For a positive-class weight ``w``, the weighted-log-loss optimum is
    ``q = w*p / (1 - p + w*p)``.  This function applies its inverse.  Ranking
    is unchanged, while calibration-sensitive risks and effects return to the
    ordinary event-probability scale.
    """

    weight = 1.0 if positive_weight is None else float(positive_weight)
    if weight <= 0:
        raise ValueError("positive_weight must be positive")
    clipped = probability.clamp(1.0e-7, 1.0 - 1.0e-7)
    if np.isclose(weight, 1.0):
        return clipped
    return clipped / (weight - (weight - 1.0) * clipped)


def _extra_trees_pipeline(seed: int, n_estimators: int) -> Any:
    return make_pipeline(
        SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        ),
        ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
    )


def fit_sarcoma_transition_model(
    dataset: Any,
    train_indices: Sequence[int],
    seed: int = 42,
    n_estimators: int = 200,
) -> SarcomaTransitionBundle:
    """Fit post-assignment state evolution using training patients only.

    The initial model maps baseline covariates and the strategy to the first
    state. The transition model maps baseline, current state, generated care,
    treatment assignment, and time to the next state. Outcomes are never inputs.
    """

    initial_design: list[np.ndarray] = []
    initial_targets: list[np.ndarray] = []
    transition_design: list[np.ndarray] = []
    transition_targets: list[np.ndarray] = []
    for raw_index in train_indices:
        sample = dataset[int(raw_index)]
        baseline = sample["x_baseline"].numpy()
        baseline_mask = sample["mask_baseline"].numpy()
        states = sample["x_state"].numpy()
        state_mask = sample["mask_state"].numpy()
        treatments = sample["x_treat"].numpy()
        treatment_mask = sample["mask_treat"].numpy()
        sequence_mask = sample["sequence_mask"].squeeze(-1).numpy()
        assignment = sample["actual_treatment"][0].numpy().astype(float)
        baseline_observed = np.where(baseline_mask > 0, baseline, np.nan)

        if sequence_mask[0] > 0 and state_mask[0].sum() > 0:
            initial_design.append(
                np.concatenate((baseline_observed, assignment))
            )
            initial_targets.append(states[0])

        for timestamp in range(dataset.seq_len - 1):
            if (
                sequence_mask[timestamp] <= 0
                or sequence_mask[timestamp + 1] <= 0
                or state_mask[timestamp].sum() == 0
                or state_mask[timestamp + 1].sum() == 0
            ):
                continue
            current_state = np.where(
                state_mask[timestamp] > 0, states[timestamp], np.nan
            )
            current_treatment = np.where(
                treatment_mask[timestamp] > 0, treatments[timestamp], np.nan
            )
            transition_design.append(
                np.concatenate(
                    (
                        baseline_observed,
                        current_state,
                        current_treatment,
                        np.asarray(
                            [*assignment, timestamp / max(dataset.seq_len - 1, 1)],
                            dtype=float,
                        ),
                    )
                )
            )
            transition_targets.append(states[timestamp + 1])

    if not initial_design or not transition_design:
        raise ValueError("Training split has insufficient observed sarcoma transitions")
    initial_model = _extra_trees_pipeline(seed, n_estimators)
    transition_model = _extra_trees_pipeline(seed + 1, n_estimators)
    initial_model.fit(np.asarray(initial_design), np.asarray(initial_targets))
    transition_model.fit(np.asarray(transition_design), np.asarray(transition_targets))
    return SarcomaTransitionBundle(
        initial_model=initial_model,
        transition_model=transition_model,
        baseline_dim=int(dataset.baseline_dim),
        state_dim=int(dataset.state_dim),
        treatment_dim=int(dataset.treat_dim),
        condition_dim=int(dataset.cond_dim),
    )


def rollout_sarcoma_preop_two_arm_from_prefix(
    generator: torch.nn.Module,
    outcome_model: torch.nn.Module,
    dataset: Any,
    transition: SarcomaTransitionBundle,
    sample: dict[str, torch.Tensor],
    prefix_length: int,
    regimen_index: int,
    device: torch.device,
    outcome_positive_weight: float | None = 1.0,
    history_only_outcome: bool = False,
) -> dict[str, np.ndarray]:
    """Roll surgery alone or preoperative RT plus surgery."""

    if int(regimen_index) not in {0, 1}:
        raise ValueError("regimen_index must be 0 or 1")
    assignment = np.zeros(2, dtype=np.float32)
    assignment[int(regimen_index)] = 1.0
    return _rollout_sarcoma_static_condition(
        generator,
        outcome_model,
        dataset,
        transition,
        sample,
        prefix_length=prefix_length,
        assignment=assignment,
        device=device,
        outcome_positive_weight=outcome_positive_weight,
        history_only_outcome=history_only_outcome,
    )


def rollout_sarcoma_two_arm_from_baseline(
    generator: torch.nn.Module,
    outcome_model: torch.nn.Module,
    dataset: Any,
    sample: dict[str, torch.Tensor],
    regimen_index: int,
    device: torch.device,
    outcome_positive_weight: float | None = 1.0,
    history_only_outcome: bool = True,
) -> dict[str, np.ndarray]:
    """Generate a complete sarcoma trajectory inside DTCyGAN from baseline.

    No observed longitudinal state is supplied. The recursive generator emits
    the first clinical state, treatment parameters, and every later state. The
    outcome model then consumes only the generated, chronology-safe history.
    """

    if int(regimen_index) not in {0, 1}:
        raise ValueError("regimen_index must be 0 or 1")
    if not hasattr(generator, "last_generated_states"):
        raise TypeError("Integrated rollout requires a recursive state generator")

    assignment = np.zeros(2, dtype=np.float32)
    assignment[int(regimen_index)] = 1.0
    baseline = sample["x_baseline"].numpy().astype(np.float32, copy=True)
    baseline_mask = sample["mask_baseline"].numpy().astype(np.float32, copy=True)
    baseline_sequence = np.repeat(baseline[None, :], dataset.seq_len, axis=0)
    baseline_mask_sequence = np.repeat(
        baseline_mask[None, :], dataset.seq_len, axis=0
    )
    empty_states = np.zeros((dataset.seq_len, dataset.state_dim), dtype=np.float32)
    empty_state_mask = np.zeros_like(empty_states)
    clinical = np.concatenate((baseline_sequence, empty_states), axis=1)
    clinical_mask = np.concatenate(
        (baseline_mask_sequence, empty_state_mask), axis=1
    )
    condition_values = np.repeat(assignment[None, :], dataset.seq_len, axis=0)
    condition = torch.from_numpy(condition_values).unsqueeze(0).to(device)
    condition_mask = torch.ones_like(condition)
    lengths = torch.asarray([dataset.seq_len], dtype=torch.long, device=device)

    with torch.inference_mode():
        generated_treatment, _ = generator(
            torch.from_numpy(clinical).unsqueeze(0).to(device),
            torch.from_numpy(clinical_mask).unsqueeze(0).to(device),
            condition,
            condition_mask,
            lengths,
            teacher_forcing_ratio=0.0,
        )
    generated_states_tensor = getattr(generator, "last_generated_states", None)
    if generated_states_tensor is None:
        raise RuntimeError("Recursive generator did not expose generated states")
    states = generated_states_tensor[0].detach().cpu().numpy().astype(np.float32)
    treatments = generated_treatment[0].detach().cpu().numpy().astype(np.float32)
    state_mask = np.ones_like(states, dtype=np.float32)
    treatment_mask = np.ones_like(treatments, dtype=np.float32)

    outcome_condition = condition
    outcome_condition_mask = condition_mask
    if history_only_outcome:
        history_states = np.zeros_like(states)
        history_state_mask = np.zeros_like(state_mask)
        history_treatments = np.zeros_like(treatments)
        history_treatment_mask = np.zeros_like(treatment_mask)
        if dataset.seq_len > 1:
            history_states[1:] = states[:-1]
            history_state_mask[1:] = state_mask[:-1]
            history_treatments[1:] = treatments[:-1]
            history_treatment_mask[1:] = treatment_mask[:-1]
        for feature_index, feature_name in enumerate(dataset.state_columns):
            if feature_name == "interval_scaled":
                history_states[:, feature_index] = states[:, feature_index]
                history_state_mask[:, feature_index] = 1.0
        outcome_clinical = np.concatenate(
            (baseline_sequence, history_states), axis=1
        )
        outcome_clinical_mask = np.concatenate(
            (baseline_mask_sequence, history_state_mask), axis=1
        )
        main = np.concatenate((outcome_clinical, history_treatments), axis=1)
        main_mask = np.concatenate(
            (outcome_clinical_mask, history_treatment_mask), axis=1
        )
        outcome_condition = torch.zeros_like(condition)
        outcome_condition_mask = torch.zeros_like(condition_mask)
    else:
        generated_clinical = np.concatenate((baseline_sequence, states), axis=1)
        generated_clinical_mask = np.concatenate(
            (baseline_mask_sequence, state_mask), axis=1
        )
        main = np.concatenate((generated_clinical, treatments), axis=1)
        main_mask = np.concatenate(
            (generated_clinical_mask, treatment_mask), axis=1
        )

    with torch.inference_mode():
        logits = outcome_model(
            torch.from_numpy(main * main_mask).unsqueeze(0).to(device),
            torch.from_numpy(main_mask).unsqueeze(0).to(device),
            outcome_condition,
            outcome_condition_mask,
            lengths,
        )
    weighted_probability = torch.sigmoid(logits[0, :, 0])
    risks = undo_positive_class_weight(
        weighted_probability, outcome_positive_weight
    ).detach().cpu().numpy().astype(np.float32)
    cumulative = 1.0 - np.cumprod(1.0 - np.clip(risks, 0.0, 1.0))
    return {
        "state": states,
        "treatment": treatments,
        "interval_risk": risks,
        "cumulative_risk": cumulative.astype(np.float32),
    }


def rollout_sarcoma_observed_clinical(
    generator: torch.nn.Module,
    outcome_model: torch.nn.Module,
    dataset: Any,
    sample: dict[str, torch.Tensor],
    regimen_index: int,
    device: torch.device,
    outcome_positive_weight: float | None = 1.0,
    history_only_outcome: bool = True,
) -> dict[str, np.ndarray]:
    """Condition a plain LSTM rollout on recorded, masked clinical updates.

    Clinical values are never imputed or generated.  An unavailable value is
    zeroed by its mask, while the recurrent hidden state carries the observed
    patient history forward.  This is a treatment-conditioned simulation given
    the factual clinical record, not a fully intervened state trajectory.
    """

    if int(regimen_index) not in {0, 1}:
        raise ValueError("regimen_index must be 0 or 1")
    if hasattr(generator, "last_generated_states"):
        raise TypeError("Observed-clinical rollout requires a plain LSTM generator")

    assignment = np.zeros(2, dtype=np.float32)
    assignment[int(regimen_index)] = 1.0
    clinical = sample["x_clin"].to(dtype=torch.float32, device=device)
    clinical_mask = sample["mask_clin"].to(dtype=torch.float32, device=device)
    sequence_mask = sample["sequence_mask"].to(
        dtype=torch.float32, device=device
    ).reshape(dataset.seq_len, 1)
    active_length = int(sequence_mask.sum().clamp(min=1).item())
    lengths = torch.asarray([active_length], dtype=torch.long, device=device)

    condition = torch.from_numpy(
        np.repeat(assignment[None, :], dataset.seq_len, axis=0)
    ).unsqueeze(0).to(device)
    condition_mask = sequence_mask.repeat(1, dataset.cond_dim).unsqueeze(0)
    treatment_mask = sequence_mask.repeat(1, dataset.treat_dim)

    with torch.inference_mode():
        generated_treatment, _ = generator(
            (clinical * clinical_mask).unsqueeze(0),
            clinical_mask.unsqueeze(0),
            condition,
            condition_mask,
            lengths,
        )
    generated_treatment = generated_treatment[0] * treatment_mask

    outcome_condition = condition
    outcome_condition_mask = condition_mask
    if history_only_outcome:
        outcome_clinical = sample["x_outcome_clin"].to(
            dtype=torch.float32, device=device
        )
        outcome_clinical_mask = sample["mask_outcome_clin"].to(
            dtype=torch.float32, device=device
        )
        history_treatment = torch.zeros_like(generated_treatment)
        history_treatment_mask = torch.zeros_like(treatment_mask)
        if dataset.seq_len > 1:
            history_treatment[1:] = generated_treatment[:-1]
            history_treatment_mask[1:] = treatment_mask[:-1]
        main = torch.cat(
            (outcome_clinical * outcome_clinical_mask, history_treatment), dim=1
        )
        main_mask = torch.cat(
            (outcome_clinical_mask, history_treatment_mask), dim=1
        )
        outcome_condition = torch.zeros_like(condition)
        outcome_condition_mask = torch.zeros_like(condition_mask)
    else:
        main = torch.cat(
            (clinical * clinical_mask, generated_treatment), dim=1
        )
        main_mask = torch.cat((clinical_mask, treatment_mask), dim=1)

    with torch.inference_mode():
        logits = outcome_model(
            main.unsqueeze(0),
            main_mask.unsqueeze(0),
            outcome_condition,
            outcome_condition_mask,
            lengths,
        )
    weighted_probability = torch.sigmoid(logits[0, :, 0])
    risks = undo_positive_class_weight(
        weighted_probability, outcome_positive_weight
    ) * sequence_mask[:, 0]
    risks_np = risks.detach().cpu().numpy().astype(np.float32)
    treatments_np = (
        generated_treatment.detach().cpu().numpy().astype(np.float32)
    )
    cumulative = 1.0 - np.cumprod(1.0 - np.clip(risks_np, 0.0, 1.0))
    return {
        "treatment": treatments_np,
        "interval_risk": risks_np,
        "cumulative_risk": cumulative.astype(np.float32),
    }


def mask_sarcoma_sample_at_landmark(
    sample: dict[str, torch.Tensor],
    dataset: Any,
    observed_prefix_length: int,
) -> dict[str, torch.Tensor]:
    """Return a copy containing only clinical observations known at a landmark.

    ``observed_prefix_length`` is the number of completed three-month intervals
    whose clinical states are available. Baseline values and calendar time stay
    available throughout the forecast. The chronology-safe outcome input is
    lagged by one interval, so state ``t`` remains available in outcome row
    ``t + 1``.
    """

    prefix = int(observed_prefix_length)
    if not 0 <= prefix <= int(dataset.seq_len):
        raise ValueError("observed_prefix_length must lie in [0, sequence length]")
    if not getattr(dataset, "history_only_outcome", False):
        raise ValueError("Landmark masking requires the chronology-safe outcome contract")

    landmark = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    state_start = int(dataset.baseline_dim)
    state_stop = state_start + int(dataset.state_dim)
    state_slice = slice(state_start, state_stop)

    landmark["x_clin"][prefix:, state_slice] = 0.0
    landmark["mask_clin"][prefix:, state_slice] = 0.0
    outcome_unknown_start = min(prefix + 1, int(dataset.seq_len))
    landmark["x_outcome_clin"][outcome_unknown_start:, state_slice] = 0.0
    landmark["mask_outcome_clin"][outcome_unknown_start:, state_slice] = 0.0

    if "interval_scaled" in dataset.state_columns:
        time_index = state_start + dataset.state_columns.index("interval_scaled")
        for values_key, mask_key in (
            ("x_clin", "mask_clin"),
            ("x_outcome_clin", "mask_outcome_clin"),
        ):
            landmark[values_key][:, time_index] = sample[values_key][:, time_index]
            landmark[mask_key][:, time_index] = sample[mask_key][:, time_index]
    return landmark


def extract_sarcoma_landmark_representation(
    generator: torch.nn.Module,
    dataset: Any,
    sample: dict[str, torch.Tensor],
    regimen_index: int,
    representation_length: int,
    device: torch.device,
) -> np.ndarray:
    """Extract the top-layer LSTM hidden/cell representation at a landmark."""

    if int(regimen_index) not in {0, 1}:
        raise ValueError("regimen_index must be 0 or 1")
    if hasattr(generator, "last_generated_states"):
        raise TypeError("Landmark representations require a plain LSTM generator")
    length = max(1, int(representation_length))
    if length > int(dataset.seq_len):
        raise ValueError("representation_length exceeds the sequence length")

    assignment = torch.zeros(
        (1, length, int(dataset.cond_dim)), dtype=torch.float32, device=device
    )
    assignment[:, :, int(regimen_index)] = 1.0
    sequence_mask = sample["sequence_mask"][:length].to(
        dtype=torch.float32, device=device
    )
    condition_mask = sequence_mask.unsqueeze(0).repeat(1, 1, int(dataset.cond_dim))
    clinical = sample["x_clin"][:length].to(dtype=torch.float32, device=device)
    clinical_mask = sample["mask_clin"][:length].to(
        dtype=torch.float32, device=device
    )
    lengths = torch.asarray([length], dtype=torch.long, device=device)
    with torch.inference_mode():
        _, recurrent_state = generator(
            (clinical * clinical_mask).unsqueeze(0),
            clinical_mask.unsqueeze(0),
            assignment,
            condition_mask,
            lengths,
        )
    if recurrent_state is None:
        raise TypeError("Generator did not return an LSTM hidden representation")
    hidden, cell = recurrent_state
    representation = torch.cat((hidden[-1, 0], cell[-1, 0]), dim=0)
    return representation.detach().cpu().numpy().astype(np.float32)


def rollout_sarcoma_observed_clinical_at_landmark(
    generator: torch.nn.Module,
    outcome_model: torch.nn.Module,
    dataset: Any,
    sample: dict[str, torch.Tensor],
    regimen_index: int,
    observed_prefix_length: int,
    device: torch.device,
    outcome_positive_weight: float | None = 1.0,
    history_only_outcome: bool = True,
    representation_length: int | None = None,
) -> dict[str, np.ndarray]:
    """Run an observed-clinical forecast from a retrospective landmark."""

    landmark = mask_sarcoma_sample_at_landmark(
        sample,
        dataset,
        observed_prefix_length=observed_prefix_length,
    )
    rollout = rollout_sarcoma_observed_clinical(
        generator,
        outcome_model,
        dataset,
        landmark,
        regimen_index=regimen_index,
        device=device,
        outcome_positive_weight=outcome_positive_weight,
        history_only_outcome=history_only_outcome,
    )
    rollout["representation"] = extract_sarcoma_landmark_representation(
        generator,
        dataset,
        landmark,
        regimen_index=regimen_index,
        representation_length=(
            max(1, int(observed_prefix_length))
            if representation_length is None
            else int(representation_length)
        ),
        device=device,
    )
    return rollout


def _rollout_sarcoma_static_condition(
    generator: torch.nn.Module,
    outcome_model: torch.nn.Module,
    dataset: Any,
    transition: SarcomaTransitionBundle,
    sample: dict[str, torch.Tensor],
    prefix_length: int,
    assignment: np.ndarray,
    device: torch.device,
    outcome_positive_weight: float | None,
    history_only_outcome: bool = False,
) -> dict[str, np.ndarray]:
    """Shared leakage-safe rollout for a baseline-assigned static strategy."""

    if not 0 <= prefix_length <= dataset.seq_len:
        raise ValueError("prefix_length must lie in [0, sequence length]")
    assignment = np.asarray(assignment, dtype=np.float32).reshape(-1)
    if len(assignment) != int(dataset.cond_dim):
        raise ValueError("Strategy assignment does not match dataset condition width")
    if (
        transition.baseline_dim != dataset.baseline_dim
        or transition.state_dim != dataset.state_dim
        or transition.treatment_dim != dataset.treat_dim
        or transition.condition_dim != dataset.cond_dim
    ):
        raise ValueError("Sarcoma transition model dimensions do not match dataset")

    baseline = sample["x_baseline"].numpy().copy()
    baseline_mask = sample["mask_baseline"].numpy().copy()
    baseline_observed = np.where(baseline_mask > 0, baseline, np.nan)
    states = np.zeros((dataset.seq_len, dataset.state_dim), dtype=np.float32)
    state_mask = np.zeros_like(states)
    if prefix_length:
        observed_states = sample["x_state"].numpy()
        observed_mask = sample["mask_state"].numpy()
        states[:prefix_length] = observed_states[:prefix_length]
        state_mask[:prefix_length] = observed_mask[:prefix_length]

    treatments = np.zeros((dataset.seq_len, dataset.treat_dim), dtype=np.float32)
    risks = np.zeros(dataset.seq_len, dtype=np.float32)
    condition = torch.from_numpy(
        np.repeat(assignment[None, :], dataset.seq_len, axis=0)
    ).unsqueeze(0).to(device)

    for timestamp in range(dataset.seq_len):
        if timestamp >= prefix_length:
            if timestamp == 0:
                design = np.concatenate(
                    (baseline_observed, assignment.astype(float))
                )[None, :]
                states[timestamp] = transition.initial_model.predict(design)[0]
            else:
                design = np.concatenate(
                    (
                        baseline_observed,
                        states[timestamp - 1],
                        treatments[timestamp - 1],
                        np.asarray(
                            [
                                *assignment.astype(float),
                                (timestamp - 1) / max(dataset.seq_len - 1, 1),
                            ],
                            dtype=float,
                        ),
                    )
                )[None, :]
                states[timestamp] = transition.transition_model.predict(design)[0]
            state_mask[timestamp] = 1.0

        baseline_sequence = np.repeat(baseline[None, :], dataset.seq_len, axis=0)
        baseline_mask_sequence = np.repeat(
            baseline_mask[None, :], dataset.seq_len, axis=0
        )
        clinical = np.concatenate((baseline_sequence, states), axis=1)
        clinical_mask = np.concatenate((baseline_mask_sequence, state_mask), axis=1)
        active_length = timestamp + 1
        lengths = torch.asarray([active_length], dtype=torch.long, device=device)
        clinical_tensor = torch.from_numpy(clinical).unsqueeze(0).to(device)
        clinical_mask_tensor = torch.from_numpy(clinical_mask).unsqueeze(0).to(device)
        condition_mask = torch.zeros_like(condition)
        condition_mask[:, :active_length] = 1.0
        with torch.inference_mode():
            generated, _ = generator(
                clinical_tensor * clinical_mask_tensor,
                clinical_mask_tensor,
                condition,
                condition_mask,
                lengths,
            )
        treatments[timestamp] = generated[0, timestamp].cpu().numpy()

        treatment_mask = np.zeros_like(treatments)
        treatment_mask[:active_length] = 1.0
        outcome_condition = condition
        outcome_condition_mask = condition_mask
        if history_only_outcome:
            history_states = np.zeros_like(states)
            history_state_mask = np.zeros_like(state_mask)
            history_treatments = np.zeros_like(treatments)
            history_treatment_mask = np.zeros_like(treatment_mask)
            if dataset.seq_len > 1:
                history_states[1:] = states[:-1] * state_mask[:-1]
                history_state_mask[1:] = state_mask[:-1]
                history_treatments[1:] = treatments[:-1] * treatment_mask[:-1]
                history_treatment_mask[1:] = treatment_mask[:-1]
            for feature_index, feature_name in enumerate(dataset.state_columns):
                if feature_name == "interval_scaled":
                    history_states[:, feature_index] = states[:, feature_index]
                    history_state_mask[:, feature_index] = state_mask[:, feature_index]
            outcome_clinical = np.concatenate(
                (baseline_sequence, history_states), axis=1
            )
            outcome_clinical_mask = np.concatenate(
                (baseline_mask_sequence, history_state_mask), axis=1
            )
            main = np.concatenate(
                (outcome_clinical, history_treatments), axis=1
            )
            main_mask = np.concatenate(
                (outcome_clinical_mask, history_treatment_mask), axis=1
            )
            outcome_condition = torch.zeros_like(condition)
            outcome_condition_mask = torch.zeros_like(condition_mask)
        else:
            main = np.concatenate((clinical, treatments), axis=1)
            main_mask = np.concatenate((clinical_mask, treatment_mask), axis=1)
        with torch.inference_mode():
            logits = outcome_model(
                torch.from_numpy(main * main_mask).unsqueeze(0).to(device),
                torch.from_numpy(main_mask).unsqueeze(0).to(device),
                outcome_condition,
                outcome_condition_mask,
                lengths,
            )
        weighted_probability = torch.sigmoid(logits[0, timestamp, 0])
        risks[timestamp] = undo_positive_class_weight(
            weighted_probability, outcome_positive_weight
        ).item()

    cumulative = 1.0 - np.cumprod(1.0 - np.clip(risks, 0.0, 1.0))
    return {
        "state": states,
        "treatment": treatments,
        "interval_risk": risks,
        "cumulative_risk": cumulative.astype(np.float32),
    }


def fit_transition_model(
    dataset: Any,
    train_indices: Sequence[int],
    seed: int = 42,
    n_estimators: int = 100,
) -> TransitionBundle:
    """Fit X(t+1) from observed X(t), A(t), and normalized time."""

    treatment_index = dataset.treat_columns.index("temporal_treatment")
    design: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in train_indices:
        sample = dataset[int(index)]
        clinical = sample["x_clin"].numpy()
        clinical_mask = sample["mask_clin"].numpy()
        treatment = sample["x_treat"].numpy()[:, treatment_index]
        for timestamp in range(dataset.seq_len - 1):
            if clinical_mask[timestamp].sum() == 0 or clinical_mask[timestamp + 1].sum() == 0:
                continue
            observed = np.where(
                clinical_mask[timestamp] > 0, clinical[timestamp], np.nan
            )
            design.append(
                np.concatenate(
                    (
                        observed,
                        np.asarray(
                            [
                                treatment[timestamp],
                                timestamp / max(dataset.seq_len - 1, 1),
                            ]
                        ),
                    )
                )
            )
            targets.append(clinical[timestamp + 1])
    if not design:
        raise ValueError("No observed state transitions are available")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    model.fit(np.asarray(design), np.asarray(targets))
    return TransitionBundle(model=model, clinical_dim=dataset.clin_dim)


def _condition_for_actions(dataset: Any, actions: np.ndarray) -> torch.Tensor:
    condition = np.zeros((dataset.seq_len, dataset.cond_dim), dtype=np.float32)
    for timestamp, action in enumerate(actions):
        column = f"temporal_treatment_t{timestamp:03d}"
        condition[:, dataset.cond_columns.index(column)] = float(action)
    return torch.from_numpy(condition)


def rollout_from_prefix(
    generator: torch.nn.Module,
    dataset: Any,
    transition: TransitionBundle,
    sample: dict[str, torch.Tensor],
    prefix_length: int,
    actions: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    """Predict a trajectory after replacing all post-prefix states by rollouts."""

    if not 1 <= prefix_length <= dataset.seq_len:
        raise ValueError("prefix_length must lie within the sequence")
    action_array = np.asarray(actions, dtype=int)
    if action_array.shape != (dataset.seq_len,) or not np.isin(action_array, [0, 1]).all():
        raise ValueError("actions must be a binary sequence matching dataset.seq_len")
    treatment_index = dataset.treat_columns.index("temporal_treatment")
    outcome_index = dataset.treat_columns.index("outcome_factual")
    clinical = sample["x_clin"].clone().numpy()
    mask = sample["mask_clin"].clone().numpy()
    condition = _condition_for_actions(dataset, action_array)
    predictions = np.full(dataset.seq_len, np.nan, dtype=float)

    for timestamp in range(prefix_length, dataset.seq_len):
        previous = timestamp - 1
        previous_state = np.where(mask[previous] > 0, clinical[previous], np.nan)
        design = np.concatenate(
            (
                previous_state,
                np.asarray(
                    [
                        action_array[previous],
                        previous / max(dataset.seq_len - 1, 1),
                    ]
                ),
            )
        )[None, :]
        clinical[timestamp] = transition.model.predict(design)[0]
        mask[timestamp] = 1.0

    clinical_tensor = torch.from_numpy(clinical).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    condition_tensor = condition.unsqueeze(0).to(device)
    condition_mask = torch.ones_like(condition_tensor)
    lengths = torch.asarray([dataset.seq_len], device=device)
    with torch.inference_mode():
        generated, _ = generator(
            clinical_tensor * mask_tensor,
            mask_tensor,
            condition_tensor,
            condition_mask,
            lengths,
        )
    generated_np = generated.squeeze(0).cpu().numpy()
    predictions[:] = generated_np[:, outcome_index]
    # Retain the generated treatment channel only as an internal audit; the
    # intervention itself is always read from ``actions``.
    _ = generated_np[:, treatment_index]
    return predictions


def rollout_batch_from_prefix(
    generator: torch.nn.Module,
    dataset: Any,
    transition: TransitionBundle,
    samples: Sequence[dict[str, torch.Tensor]],
    prefix_length: int,
    actions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Vectorized counterpart of :func:`rollout_from_prefix`."""

    if not samples:
        raise ValueError("samples cannot be empty")
    action_array = np.asarray(actions, dtype=int)
    if action_array.shape != (len(samples), dataset.seq_len):
        raise ValueError("actions must have shape (patients, seq_len)")
    clinical = np.stack([sample["x_clin"].numpy() for sample in samples])
    mask = np.stack([sample["mask_clin"].numpy() for sample in samples])
    for timestamp in range(prefix_length, dataset.seq_len):
        previous = timestamp - 1
        previous_state = np.where(mask[:, previous] > 0, clinical[:, previous], np.nan)
        design = np.column_stack(
            (
                previous_state,
                action_array[:, previous],
                np.full(
                    len(samples), previous / max(dataset.seq_len - 1, 1)
                ),
            )
        )
        clinical[:, timestamp] = transition.model.predict(design)
        mask[:, timestamp] = 1.0
    condition = np.zeros(
        (len(samples), dataset.seq_len, dataset.cond_dim), dtype=np.float32
    )
    for timestamp in range(dataset.seq_len):
        column = dataset.cond_columns.index(
            f"temporal_treatment_t{timestamp:03d}"
        )
        condition[:, :, column] = action_array[:, timestamp, None]
    clinical_tensor = torch.from_numpy(clinical).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)
    condition_tensor = torch.from_numpy(condition).to(device)
    lengths = torch.full(
        (len(samples),), dataset.seq_len, dtype=torch.long, device=device
    )
    with torch.inference_mode():
        generated, _ = generator(
            clinical_tensor * mask_tensor,
            mask_tensor,
            condition_tensor,
            torch.ones_like(condition_tensor),
            lengths,
        )
    outcome_index = dataset.treat_columns.index("outcome_factual")
    return generated[:, :, outcome_index].cpu().numpy()
