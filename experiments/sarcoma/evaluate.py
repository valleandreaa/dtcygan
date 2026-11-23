#!/usr/bin/env python
"""Evaluate the focused preoperative-RT versus surgery-only sarcoma model."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

from src.sequential_rollout import (  # noqa: E402
    rollout_sarcoma_two_arm_from_baseline,
    rollout_sarcoma_observed_clinical,
    undo_positive_class_weight,
)
from src.causal_weights import (  # noqa: E402
    CausalWeightSpec,
    CausalWeightTable,
    load_training_weights,
)
from src.causal_diagnostics import standardized_mean_differences  # noqa: E402
from src.surgery_rt_training_data import (  # noqa: E402
    PREOP_TWO_ARM_CONDITION_COLUMNS,
    SarcomaSequenceDataset,
    build_surgery_rt_preop_two_arm_training_data,
)
from src.training import (  # noqa: E402
    Config,
    LSTMOutcomePredictor,
    build_clinical_generator,
    build_outcome_model_inputs,
    create_dataloaders,
    resolve_device,
)


@dataclass
class TwoArmBundle:
    config: Config
    metadata: dict[str, Any]
    dataset: SarcomaSequenceDataset
    generator: torch.nn.Module
    outcome_model: LSTMOutcomePredictor
    dev_indices: list[int]
    test_indices: list[int]
    device: torch.device
    checkpoint_path: Path
    weight_table: CausalWeightTable | None


def resolve_checkpoint(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    checkpoints = sorted(candidate.glob("dtcygan_*.pt"), key=lambda item: item.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No DTCyGAN checkpoint found under {candidate}")
    return checkpoints[-1]


def load_bundle(
    checkpoint_path: str | Path,
    person_period_path: str | Path,
    weights_path: str | Path | None = None,
    weight_metadata_path: str | Path | None = None,
    device: str | None = None,
    allow_unweighted_reconstruction: bool = False,
) -> TwoArmBundle:
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"])
    if config.data_contract != "surgery_rt_preop_two_arm":
        raise ValueError("This postprocessor requires surgery_rt_preop_two_arm")
    data = build_surgery_rt_preop_two_arm_training_data(
        person_period_path,
        config.feature_groups,
        seq_len=config.seq_len,
        cohort_contract=config.cohort_contract,
        split_seed=config.split_seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    patient_ids = [str(patient["patient_id"]) for patient in data["patients"]]
    causal_settings = config.causal_weighting or {}
    weighting_enabled = bool(causal_settings.get("enabled", False))
    weight_table: CausalWeightTable | None = None
    weight_by_patient: dict[str, float] | None = None
    if weighting_enabled:
        if weights_path is None or weight_metadata_path is None:
            if not allow_unweighted_reconstruction:
                raise ValueError(
                    "Weighted two-arm postprocessing requires --causal-weights and "
                    "--causal-weight-metadata"
                )
        else:
            spec = CausalWeightSpec(
                method=str(causal_settings.get("method", "primary_overlap")),
                stage=str(causal_settings.get("stage", "preoperative_rt")),
                ineligible_policy=str(causal_settings.get("ineligible_policy", "exclude")),
                normalization=str(causal_settings.get("normalization", "mean_one")),
                require_supported=bool(causal_settings.get("require_supported", True)),
            )
            weight_table = load_training_weights(
                weights_path,
                spec=spec,
                metadata_path=weight_metadata_path,
                expected_patient_ids=patient_ids,
            )
            expected_target = causal_settings.get("target_population")
            observed_target = weight_table.metadata.get("weighting_target")
            if expected_target is not None and observed_target != expected_target:
                raise ValueError(
                    "Causal-weight target differs from checkpoint configuration: "
                    f"{observed_target!r} versus {expected_target!r}"
                )
            weight_by_patient = dict(
                zip(
                    weight_table.frame[spec.patient_id_col].astype(str),
                    weight_table.frame[spec.weight_col].astype(float),
                )
            )
    dataset = SarcomaSequenceDataset(
        data,
        config.seq_len,
        config.feature_groups,
        patient_weights=weight_by_patient,
    )
    _, _, dev_indices, test_indices = create_dataloaders(
        dataset, config.batch_size, config.seed
    )
    metadata = checkpoint.get("metadata") or {}
    temporal_version = int(
        config.temporal_alignment.get("contract_version", 1)
    )
    if temporal_version >= 2:
        checkpoint_temporal = metadata.get("temporal_contract") or {}
        if int(checkpoint_temporal.get("contract_version", 0)) != temporal_version:
            raise ValueError(
                "Checkpoint temporal metadata does not match the chronology-safe "
                "preoperative configuration"
            )
        rebuilt_temporal = data.get("temporal_contract") or {}
        checkpoint_comparable = dict(checkpoint_temporal)
        legacy_resolution_corrected = (
            checkpoint_comparable.get("trajectory_resolution")
            == "six_month_intervals"
            and rebuilt_temporal.get("trajectory_resolution")
            == "three_month_intervals"
            and int(config.seq_len) == 8
        )
        if legacy_resolution_corrected:
            checkpoint_comparable["trajectory_resolution"] = (
                rebuilt_temporal["trajectory_resolution"]
            )
        if rebuilt_temporal != checkpoint_comparable:
            raise ValueError(
                "Rebuilt preoperative temporal contract does not match checkpoint "
                "provenance"
            )
        if legacy_resolution_corrected:
            metadata = dict(metadata)
            metadata["temporal_contract"] = dict(rebuilt_temporal)
            metadata["legacy_checkpoint_trajectory_resolution"] = (
                "six_month_intervals"
            )
    rebuilt_dev = [dataset.patient_ids[index] for index in dev_indices]
    rebuilt_test = [dataset.patient_ids[index] for index in test_indices]
    if metadata.get("dev_patient_ids") != rebuilt_dev:
        raise ValueError("Rebuilt dev split does not match checkpoint provenance")
    if metadata.get("test_patient_ids") != rebuilt_test:
        raise ValueError("Rebuilt test split does not match checkpoint provenance")
    if metadata.get("test_evaluated_during_training") is not False:
        raise ValueError("Checkpoint does not prove that test was held out during training")
    observed_dimensions = (
        dataset.clin_dim,
        dataset.treat_dim,
        dataset.cond_dim,
        dataset.outcome_dim,
    )
    checkpoint_dimensions = tuple(
        int(metadata.get(name, -1))
        for name in ("clin_dim", "treat_dim", "cond_dim", "outcome_dim")
    )
    if checkpoint_dimensions != observed_dimensions:
        raise ValueError(
            f"Checkpoint/data dimensions differ: {checkpoint_dimensions} vs "
            f"{observed_dimensions}"
        )

    target_device = resolve_device(device or config.device)
    generator = build_clinical_generator(config, dataset).to(target_device)
    generator.load_state_dict(checkpoint["Gx"])
    generator.eval()
    outcome_model = LSTMOutcomePredictor(
        dataset.clin_dim + dataset.treat_dim,
        dataset.cond_dim,
        config.outcome_hidden,
        dataset.outcome_dim,
        config.layers,
    ).to(target_device)
    outcome_model.load_state_dict(checkpoint["outcome_model"])
    outcome_model.eval()
    return TwoArmBundle(
        config=config,
        metadata=metadata,
        dataset=dataset,
        generator=generator,
        outcome_model=outcome_model,
        dev_indices=dev_indices,
        test_indices=test_indices,
        device=target_device,
        checkpoint_path=checkpoint_path.resolve(),
        weight_table=weight_table,
    )


def factual_metrics(
    bundle: TwoArmBundle, indices: Sequence[int] | None = None
) -> dict[str, Any]:
    """Compute factual outcome metrics for a specified patient set.

    The default remains the untouched test set.  Supplying ``indices`` is for
    explicitly labelled in-sample diagnostic analyses only.
    """
    truth: list[np.ndarray] = []
    probability: list[np.ndarray] = []
    row_weights: list[np.ndarray] = []
    selected_indices = bundle.test_indices if indices is None else list(indices)
    for index in selected_indices:
        sample = bundle.dataset[index]
        clinical = sample["x_clin"].unsqueeze(0).to(bundle.device)
        clinical_mask = sample["mask_clin"].unsqueeze(0).to(bundle.device)
        treatment = sample["x_treat"].unsqueeze(0).to(bundle.device)
        treatment_mask = sample["mask_treat"].unsqueeze(0).to(bundle.device)
        condition = sample["actual_treatment"].unsqueeze(0).to(bundle.device)
        condition_mask = sample["mask_actual"].unsqueeze(0).to(bundle.device)
        outcome = sample["outcome"].unsqueeze(0).to(bundle.device)
        outcome_mask = sample["outcome_mask"].unsqueeze(0).to(bundle.device)
        lengths = sample["sequence_mask"].sum().clamp(min=1).long().reshape(1).to(
            bundle.device
        )
        history_only = int(
            bundle.config.temporal_alignment.get("contract_version", 1)
        ) >= 2
        main, main_mask, outcome_condition, outcome_condition_mask = (
            build_outcome_model_inputs(
                clinical,
                clinical_mask,
                treatment,
                treatment_mask,
                condition,
                condition_mask,
                history_only=history_only,
                history_clinical=(
                    sample["x_outcome_clin"].unsqueeze(0).to(bundle.device)
                    if history_only
                    else None
                ),
                history_clinical_mask=(
                    sample["mask_outcome_clin"].unsqueeze(0).to(bundle.device)
                    if history_only
                    else None
                ),
                history_treatment=(
                    sample["x_outcome_treat"].unsqueeze(0).to(bundle.device)
                    if history_only
                    else None
                ),
                history_treatment_mask=(
                    sample["mask_outcome_treat"].unsqueeze(0).to(bundle.device)
                    if history_only
                    else None
                ),
            )
        )
        with torch.inference_mode():
            logits = bundle.outcome_model(
                main,
                main_mask,
                outcome_condition,
                outcome_condition_mask,
                lengths,
            )
        active = outcome_mask.bool().cpu().numpy().reshape(-1)
        truth.append(outcome.cpu().numpy().reshape(-1)[active])
        corrected = undo_positive_class_weight(
            torch.sigmoid(logits), bundle.config.outcome_positive_weight
        )
        probability.append(corrected.cpu().numpy().reshape(-1)[active])
        patient_weight = float(sample["causal_weight"])
        row_weights.append(np.full(int(active.sum()), patient_weight, dtype=float))
    y_true = np.concatenate(truth)
    y_probability = np.concatenate(probability)
    weights = np.concatenate(row_weights)
    has_both_outcomes = len(np.unique(y_true)) == 2
    unweighted = {
        "auroc": float(roc_auc_score(y_true, y_probability)) if has_both_outcomes else None,
        "average_precision": (
            float(average_precision_score(y_true, y_probability))
            if has_both_outcomes
            else None
        ),
        "brier": float(brier_score_loss(y_true, y_probability)),
        "predicted_rate": float(y_probability.mean()),
        "observed_rate": float(y_true.mean()),
    }
    return {
        "n_rows": int(len(y_true)),
        "event_rows": int(y_true.sum()),
        "auroc": (
            float(roc_auc_score(y_true, y_probability, sample_weight=weights))
            if has_both_outcomes
            else None
        ),
        "average_precision": (
            float(average_precision_score(y_true, y_probability, sample_weight=weights))
            if has_both_outcomes
            else None
        ),
        "brier": float(brier_score_loss(y_true, y_probability, sample_weight=weights)),
        "predicted_rate": float(np.average(y_probability, weights=weights)),
        "observed_rate": float(np.average(y_true, weights=weights)),
        "weight_sum": float(weights.sum()),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        "metric_scale": (
            "propensity_overlap_weighted"
            if bundle.weight_table is not None
            else "unweighted"
        ),
        "unweighted_sensitivity": unweighted,
    }


def generate_trajectories(
    bundle: TwoArmBundle,
    indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Generate both strategy trajectories for the requested patient indices.

    The default is the held-out test split. Passing explicit indices
    is reserved for clearly labelled model-diagnostic analyses.
    """

    integrated = bool(bundle.config.recursive_state_generation)
    feature_indices = {
        name: bundle.dataset.treat_columns.index(name)
        for name in ("surgery_this_interval", "preoperative_rt_this_interval")
    }
    rows: list[dict[str, Any]] = []
    selected_indices = bundle.test_indices if indices is None else list(indices)
    for index in selected_indices:
        sample = bundle.dataset[index]
        patient = bundle.dataset.patients[index]
        patient_id = bundle.dataset.patient_ids[index]
        histology = patient["baseline"].get("who_diagnosis_code") or "unknown"
        for regimen_index, regimen in enumerate(PREOP_TWO_ARM_CONDITION_COLUMNS):
            rollout_kwargs = {
                "regimen_index": regimen_index,
                "device": bundle.device,
                "outcome_positive_weight": bundle.config.outcome_positive_weight,
                "history_only_outcome": int(
                    bundle.config.temporal_alignment.get("contract_version", 1)
                ) >= 2,
            }
            if integrated:
                rollout = rollout_sarcoma_two_arm_from_baseline(
                    bundle.generator,
                    bundle.outcome_model,
                    bundle.dataset,
                    sample,
                    **rollout_kwargs,
                )
            else:
                rollout = rollout_sarcoma_observed_clinical(
                    bundle.generator,
                    bundle.outcome_model,
                    bundle.dataset,
                    sample,
                    **rollout_kwargs,
                )
            for interval in range(bundle.config.seq_len):
                row = {
                    "patient_id": patient_id,
                    "histology": str(histology),
                    "causal_weight": float(sample["causal_weight"]),
                    "regimen": regimen,
                    "regimen_index": regimen_index,
                    "interval": interval,
                    "interval_risk": float(rollout["interval_risk"][interval]),
                    "cumulative_risk": float(rollout["cumulative_risk"][interval]),
                }
                for feature, feature_index in feature_indices.items():
                    row[f"generated_{feature}"] = float(
                        rollout["treatment"][interval, feature_index]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def final_risk_table(trajectories: pd.DataFrame) -> pd.DataFrame:
    return (
        trajectories.sort_values("interval")
        .groupby(
            ["patient_id", "histology", "causal_weight", "regimen"], sort=False
        )[
            "cumulative_risk"
        ]
        .last()
        .unstack("regimen")
        .reset_index()
    )


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def _weighted_standard_deviation(values: pd.Series, weights: pd.Series) -> float:
    x = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    mean = float(np.average(x, weights=w))
    return float(np.sqrt(np.average(np.square(x - mean), weights=w)))


def _effective_sample_size(weights: pd.Series) -> float:
    w = weights.to_numpy(dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def summarize_effects(
    trajectories: pd.DataFrame, *, estimand: str = "propensity_overlap_population"
) -> pd.DataFrame:
    final = final_risk_table(trajectories)
    difference = final["preoperative_rt_surgery"] - final["surgery_only"]
    weights = final["causal_weight"]
    return pd.DataFrame(
        [
            {
                "contrast": "preoperative_rt_surgery - surgery_only",
                "risk_surgery_only": _weighted_mean(final["surgery_only"], weights),
                "risk_preoperative_rt_surgery": _weighted_mean(
                    final["preoperative_rt_surgery"], weights
                ),
                "risk_difference": _weighted_mean(difference, weights),
                "unweighted_risk_difference_sensitivity": float(difference.mean()),
                "n_test_patients": int(len(difference)),
                "effective_sample_size": _effective_sample_size(weights),
                "estimand": estimand,
                "uncertainty_interval": "not_computed",
            }
        ]
    )


def summarize_risk_distribution(trajectories: pd.DataFrame) -> pd.DataFrame:
    final = (
        trajectories.sort_values("interval")
        .groupby(["patient_id", "causal_weight", "regimen"], sort=False)[
            "cumulative_risk"
        ]
        .last()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for regimen, group in final.groupby("regimen", sort=False):
        rows.append(
            {
                "regimen": regimen,
                "n": int(len(group)),
                "effective_sample_size": _effective_sample_size(group["causal_weight"]),
                "weighted_mean": _weighted_mean(
                    group["cumulative_risk"], group["causal_weight"]
                ),
                "weighted_std": _weighted_standard_deviation(
                    group["cumulative_risk"], group["causal_weight"]
                ),
                "minimum": float(group["cumulative_risk"].min()),
                "maximum": float(group["cumulative_risk"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_histology_effects(trajectories: pd.DataFrame) -> pd.DataFrame:
    final = final_risk_table(trajectories)
    final["risk_difference"] = (
        final["preoperative_rt_surgery"] - final["surgery_only"]
    )
    rows: list[dict[str, Any]] = []
    for histology, group in final.groupby("histology", sort=False):
        rows.append(
            {
                "histology": histology,
                "n": int(len(group)),
                "effective_sample_size": _effective_sample_size(group["causal_weight"]),
                "weighted_mean_risk_difference": _weighted_mean(
                    group["risk_difference"], group["causal_weight"]
                ),
                "weighted_sd_risk_difference": _weighted_standard_deviation(
                    group["risk_difference"], group["causal_weight"]
                ),
                "unweighted_mean_risk_difference_sensitivity": float(
                    group["risk_difference"].mean()
                ),
                "descriptive_minimum_met": bool(
                    len(group) >= 10
                    and _effective_sample_size(group["causal_weight"]) >= 10
                ),
                "causal_subgroup_supported": False,
                "support_reason": "stratum_specific_balance_not_established",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["descriptive_minimum_met", "n"], ascending=[False, False]
    )


def test_covariate_balance(
    bundle: TwoArmBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit whether the held-out effect-summary subset remains balanced."""

    if bundle.weight_table is None:
        return pd.DataFrame(), {
            "evaluated": False,
            "supported": False,
            "reason": "causal_weighting_disabled",
        }
    covariates = [
        str(value)
        for value in bundle.weight_table.metadata.get("selected_covariates", [])
    ]
    rows: list[dict[str, Any]] = []
    for index in bundle.test_indices:
        patient = bundle.dataset.patients[index]
        sample = bundle.dataset[index]
        row = {name: patient["baseline"].get(name) for name in covariates}
        row["treatment"] = int(patient["regimen_index"])
        row["causal_weight"] = float(sample["causal_weight"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    unweighted = standardized_mean_differences(frame, "treatment", covariates)
    weighted = standardized_mean_differences(
        frame,
        "treatment",
        covariates,
        weights=frame["causal_weight"],
    )
    balance = pd.concat(
        [
            unweighted.assign(estimand="unweighted"),
            weighted.assign(estimand="overlap_weighted"),
        ],
        ignore_index=True,
    )
    maximum_unweighted = (
        float(unweighted["abs_smd"].max()) if not unweighted.empty else None
    )
    maximum_weighted = float(weighted["abs_smd"].max()) if not weighted.empty else None
    arm_ess = {
        str(arm): _effective_sample_size(group["causal_weight"])
        for arm, group in frame.groupby("treatment")
    }
    minimum_arm_ess = min(arm_ess.values()) if len(arm_ess) == 2 else 0.0
    supported = bool(
        maximum_weighted is not None
        and maximum_weighted <= 0.10
        and minimum_arm_ess >= 20.0
    )
    return balance, {
        "evaluated": True,
        "n_test_patients": int(len(frame)),
        "arm_counts": {
            str(key): int(value)
            for key, value in frame["treatment"].value_counts().sort_index().items()
        },
        "maximum_absolute_smd_unweighted": maximum_unweighted,
        "maximum_absolute_smd_weighted": maximum_weighted,
        "arm_effective_sample_size": arm_ess,
        "minimum_arm_effective_sample_size": minimum_arm_ess,
        "maximum_absolute_smd_threshold": 0.10,
        "minimum_arm_effective_sample_size_threshold": 20.0,
        "supported": supported,
        "reason": (
            "passes_balance_and_effective_sample_size_gate"
            if supported
            else "held_out_test_subset_fails_balance_or_effective_sample_size_gate"
        ),
    }


def adherence(trajectories: pd.DataFrame) -> dict[str, float]:
    delivered = (
        trajectories.groupby(["patient_id", "regimen"])[
            [
                "generated_surgery_this_interval",
                "generated_preoperative_rt_this_interval",
            ]
        ]
        .max()
        .ge(0.5)
    )
    surgery = delivered.xs("surgery_only", level="regimen")
    preoperative = delivered.xs("preoperative_rt_surgery", level="regimen")
    return {
        "surgery_only": float(
            (
                surgery["generated_surgery_this_interval"]
                & ~surgery["generated_preoperative_rt_this_interval"]
            ).mean()
        ),
        "preoperative_rt_surgery": float(
            (
                preoperative["generated_surgery_this_interval"]
                & preoperative["generated_preoperative_rt_this_interval"]
            ).mean()
        ),
    }


def leakage_audit(bundle: TwoArmBundle, maximum_patients: int = 16) -> dict[str, Any]:
    integrated = bool(bundle.config.recursive_state_generation)
    differences: list[float] = []
    for index in bundle.test_indices[:maximum_patients]:
        sample = {key: value.clone() for key, value in bundle.dataset[index].items()}
        perturbed = {key: value.clone() for key, value in sample.items()}
        if integrated:
            perturbed["x_state"] = torch.randn_like(perturbed["x_state"]) * 100.0
            perturbed["x_clin"][:, bundle.dataset.baseline_dim :] = torch.randn_like(
                perturbed["x_clin"][:, bundle.dataset.baseline_dim :]
            ) * 100.0
        else:
            for value_key, mask_key in (
                ("x_clin", "mask_clin"),
                ("x_outcome_clin", "mask_outcome_clin"),
            ):
                random_values = torch.randn_like(perturbed[value_key]) * 100.0
                perturbed[value_key] = torch.where(
                    perturbed[mask_key] > 0,
                    perturbed[value_key],
                    random_values,
                )
        for regimen_index in range(2):
            rollout_kwargs = {
                "regimen_index": regimen_index,
                "device": bundle.device,
                "outcome_positive_weight": bundle.config.outcome_positive_weight,
                "history_only_outcome": int(
                    bundle.config.temporal_alignment.get("contract_version", 1)
                ) >= 2,
            }
            if integrated:
                original = rollout_sarcoma_two_arm_from_baseline(
                    bundle.generator, bundle.outcome_model, bundle.dataset,
                    sample, **rollout_kwargs,
                )
                changed = rollout_sarcoma_two_arm_from_baseline(
                    bundle.generator, bundle.outcome_model, bundle.dataset,
                    perturbed, **rollout_kwargs,
                )
            else:
                original = rollout_sarcoma_observed_clinical(
                    bundle.generator, bundle.outcome_model, bundle.dataset,
                    sample, **rollout_kwargs,
                )
                changed = rollout_sarcoma_observed_clinical(
                    bundle.generator, bundle.outcome_model, bundle.dataset,
                    perturbed, **rollout_kwargs,
                )
            differences.append(
                float(
                    np.max(
                        np.abs(
                            original["cumulative_risk"]
                            - changed["cumulative_risk"]
                        )
                    )
                )
            )
    maximum = max(differences, default=0.0)
    audit = {
        "audited_test_patients": min(
            maximum_patients, len(bundle.test_indices)
        ),
        "threshold": 1.0e-7,
    }
    if integrated:
        audit.update(
            {
                "audit_type": "post_t_factual_state_invariance",
                "post_t_factual_state_sensitivity_max": maximum,
                "passes_t_time_censoring": bool(maximum <= 1.0e-7),
            }
        )
    else:
        audit.update(
            {
                "audit_type": "masked_clinical_value_invariance",
                "masked_clinical_value_sensitivity_max": maximum,
                "passes_missingness_masking": bool(maximum <= 1.0e-7),
                "passes_t_time_censoring": None,
                "factual_clinical_updates_used": True,
            }
        )
    return audit


def run_postprocess(
    checkpoint: str | Path,
    person_period_data: str | Path,
    output_dir: str | Path,
    causal_weights: str | Path | None = None,
    causal_weight_metadata: str | Path | None = None,
    bootstrap: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    if bootstrap != 0:
        raise ValueError(
            "Bootstrap uncertainty is disabled for the v8 analysis; use --bootstrap 0"
        )
    bundle = load_bundle(
        checkpoint,
        person_period_data,
        weights_path=causal_weights,
        weight_metadata_path=causal_weight_metadata,
        device=device,
    )
    dev_ids = {bundle.dataset.patient_ids[index] for index in bundle.dev_indices}
    test_ids = {
        bundle.dataset.patient_ids[index] for index in bundle.test_indices
    }
    if dev_ids & test_ids:
        raise ValueError("Dev and test patient sets overlap")
    trajectories = generate_trajectories(bundle)
    weighting_enabled = bundle.weight_table is not None
    effects = summarize_effects(
        trajectories,
        estimand=(
            "propensity_overlap_population"
            if weighting_enabled
            else "unweighted_test_population"
        ),
    )
    risk_distribution = summarize_risk_distribution(trajectories)
    histology_effects = summarize_histology_effects(trajectories)
    factual = factual_metrics(bundle)
    conditioning = adherence(trajectories)
    leakage = leakage_audit(bundle)
    test_balance, test_balance_diagnostics = test_covariate_balance(
        bundle
    )
    test_effect_supported = bool(
        test_balance_diagnostics.get("supported", False)
    )
    effects["effect_support"] = (
        "balance_supported_model_based"
        if test_effect_supported
        else "descriptive_only_test_balance_failed"
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output / "counterfactual_trajectories.csv", index=False)
    effects.to_csv(output / "population_effect_summary.csv", index=False)
    risk_distribution.to_csv(output / "final_risk_distribution.csv", index=False)
    histology_effects.to_csv(output / "histology_effect_summary.csv", index=False)
    test_balance.to_csv(
        output / "test_covariate_balance.csv", index=False
    )
    causal_metadata = bundle.metadata.get("causal_weighting") or {}
    report = {
        "assessment": (
            "propensity_weighted_model_based_test_balance_supported"
            if weighting_enabled and test_effect_supported
            else "propensity_weighted_model_based_test_balance_failed"
            if weighting_enabled
            else "exploratory_unweighted"
        ),
        "checkpoint": str(bundle.checkpoint_path),
        "design": {
            "strategies": PREOP_TWO_ARM_CONDITION_COLUMNS,
            "cohort": bundle.metadata.get("cohort"),
            "splits": {
                "dev": len(bundle.dev_indices),
                "test": len(bundle.test_indices),
            },
            "test_split_present": True,
            "test_evaluated_during_training": False,
            "patient_overlap": 0,
            "temporal_contract": bundle.metadata.get("temporal_contract"),
            "clinical_history_policy": (
                "generated_recursive_state"
                if bundle.config.recursive_state_generation
                else "recorded_factual_updates_with_missingness_mask"
            ),
            "estimand": (
                "propensity_overlap_population"
                if weighting_enabled
                else "unweighted_test_population"
            ),
        },
        "test": {
            "factual": factual,
            "strategy_adherence": conditioning,
        },
        "leakage_audit": leakage,
        "state_transition_engine": (
            "integrated_recursive_dtcygan"
            if bundle.config.recursive_state_generation
            else "observed_masked_lstm"
        ),
        "effect_summary": effects.to_dict(orient="records"),
        "risk_distribution": risk_distribution.to_dict(orient="records"),
        "causal_interpretation": {
            "weighting_enabled": weighting_enabled,
            "trajectory_scope": (
                "fully_recursive_generated_state"
                if bundle.config.recursive_state_generation
                else "treatment_conditioned_on_recorded_factual_clinical_history"
            ),
            "fully_intervened_longitudinal_trajectory": bool(
                bundle.config.recursive_state_generation
            ),
            "method": causal_metadata.get("method"),
            "target_population": causal_metadata.get("target_population"),
            "full_cohort_adjustment_supported": causal_metadata.get(
                "adjustment_supported", False
            ),
            "gate_metrics": causal_metadata.get("gate_metrics"),
            "training_adjustment_supported": causal_metadata.get(
                "adjustment_supported", False
            ),
            "test_effect_balance": test_balance_diagnostics,
            "test_effect_adjustment_supported": test_effect_supported,
            "uncertainty_intervals": "not_computed",
            "boundary": (
                "Full-cohort propensity weighting supports measured baseline balance "
                "for development, but the held-out test subset fails the balance "
                "gate. Its risk difference is descriptive and model-based, not a "
                "balance-supported causal effect."
                if weighting_enabled and not test_effect_supported
                else "Propensity weighting supports measured baseline balance in the "
                "overlap population. The contrast remains model-based because "
                "individual counterfactual outcomes are generated. The independent "
                "test set remains a model-based evaluation set."
                if weighting_enabled
                else "This is an unweighted model-based exploratory contrast."
            ),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        (
            "# Sarcoma preoperative-RT two-arm DTCyGAN v8"
            if weighting_enabled
            else "# Sarcoma preoperative-RT two-arm DTCyGAN"
        ),
        "",
        (
            "Assessment: **PROPENSITY-WEIGHTED DEVELOPMENT; TEST EFFECT "
            "DESCRIPTIVE ONLY**"
            if weighting_enabled and not test_effect_supported
            else "Assessment: **PROPENSITY-OVERLAP-WEIGHTED, MODEL-BASED**"
            if weighting_enabled
            else "Assessment: **EXPLORATORY, UNWEIGHTED**"
        ),
        f"Checkpoint: `{bundle.checkpoint_path}`",
        (
            f"Splits: {len(bundle.dev_indices)} dev / "
            f"{len(bundle.test_indices)} test"
        ),
        (
            f"T-time censoring gate: {leakage['passes_t_time_censoring']}"
            if bundle.config.recursive_state_generation
            else "Clinical-state policy: recorded masked updates; unavailable "
            "values are carried through the LSTM hidden state"
        ),
        (
            ""
            if bundle.config.recursive_state_generation
            else f"Missingness masking gate: {leakage['passes_missingness_masking']}"
        ),
        "",
        "Postoperative-only and combined-RT patients were excluded before splitting.",
        (
            "Point estimates target the propensity-overlap population; no bootstrap "
            "or confidence interval was computed."
            if weighting_enabled
            else "No propensity adjustment was applied."
        ),
        (
            "Development-cohort balance passed, but the test subset failed: "
            f"maximum weighted |SMD| = "
            f"{test_balance_diagnostics.get('maximum_absolute_smd_weighted'):.3f}. "
            "Treat its risk difference as descriptive only."
            if weighting_enabled and not test_effect_supported
            else "The test-subset weighting gate passed."
            if weighting_enabled
            else ""
        ),
        (
            "The generated-counterfactual contrast remains model-based and was "
            "evaluated once on the independent held-out test cohort."
            if bundle.config.recursive_state_generation
            else "The treatment-conditioned contrast uses recorded factual clinical "
            "updates and is not a fully intervened longitudinal trajectory. It was "
            "evaluated once on the independent held-out test cohort."
        ),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


