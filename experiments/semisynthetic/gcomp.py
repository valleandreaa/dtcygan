"""Leakage-safe G-computation baseline for T4's semi-synthetic MIMIC cohort.

The outcome model is trained exclusively on factual outcomes.  Oracle potential
outcomes are loaded only after fitting to score conditional point-treatment
effects on the shared validation or test split.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


DEFAULT_STATIC_FEATURES = ("agegroup", "heightgroup", "weightgroup", "gender")


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    truth_flat = np.asarray(truth, dtype=float).reshape(-1)
    prediction_flat = np.asarray(prediction, dtype=float).reshape(-1)
    truth_std = float(np.std(truth_flat))
    correlation = (
        float(np.corrcoef(truth_flat, prediction_flat)[0, 1])
        if np.std(truth_flat) > 0 and np.std(prediction_flat) > 0
        else 0.0
    )
    pehe = float(np.sqrt(np.mean(np.square(error))))
    return {
        "pehe": pehe,
        "normalized_pehe": pehe / truth_std if truth_std > 0 else 0.0,
        "true_ate": float(np.mean(truth_flat)),
        "predicted_ate": float(np.mean(prediction_flat)),
        "ate_absolute_error": float(abs(np.mean(error))),
        "ite_correlation": correlation,
        "effect_sign_accuracy": float(
            np.mean(np.sign(truth_flat) == np.sign(prediction_flat))
        ),
    }


def _load_split(path: str | Path, source_ids: Sequence[str]) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = {
        str(row["source_patient_id"]): str(row["split"])
        for row in payload["patients"]
    }
    if set(labels) != set(source_ids):
        raise ValueError("Split manifest IDs do not exactly match the input pickle")
    if set(labels.values()) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation, and test")
    return labels


def _load_truth(path: str | Path, source_ids: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    truth = {str(row["source_patient_id"]): row for row in payload["patients"]}
    if set(truth) != set(source_ids):
        raise ValueError("Oracle truth IDs do not exactly match the input pickle")
    return truth


def _design_rows(
    data: Mapping[str, Mapping[str, Any]],
    patient_ids: Sequence[str],
    temporal_features: Sequence[str],
    static_features: Sequence[str],
    baseline_end: int,
    evaluation_end: int,
    history_mode: str,
    timestamps: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Return baseline-only designs, factual outcomes, treatments, and keys."""

    designs: list[np.ndarray] = []
    outcomes: list[float] = []
    treatments: list[float] = []
    keys: list[tuple[str, int]] = []
    for patient_id in patient_ids:
        patient = data[patient_id]
        clinical = np.column_stack(
            [np.asarray(patient[name], dtype=float) for name in temporal_features]
        )
        treatment = np.asarray(patient["treatment"], dtype=float)
        outcome = np.asarray(patient["outcome"], dtype=float)
        if clinical.shape[0] <= max(timestamps) or len(treatment) != len(outcome):
            raise ValueError(f"Patient {patient_id} does not cover the requested horizon")
        if history_mode not in {"baseline", "observed"}:
            raise ValueError("history_mode must be 'baseline' or 'observed'")
        static = np.asarray([patient[name] for name in static_features], dtype=float)
        for timestamp in timestamps:
            clinical_end = baseline_end if history_mode == "baseline" else timestamp
            clinical_history = clinical[: clinical_end + 1].reshape(-1)
            clinical_slots = (evaluation_end + 1) * clinical.shape[1]
            padded_clinical = np.full(clinical_slots, np.nan, dtype=float)
            padded_clinical[: clinical_history.size] = clinical_history
            # Treatment before the evaluated timestamp is observed history;
            # contemporaneous treatment is appended only at model prediction.
            treatment_start = baseline_end + 1 if history_mode == "baseline" else 0
            treatment_history = treatment[treatment_start:timestamp]
            history_slots = evaluation_end - treatment_start
            padded_history = np.full(history_slots, np.nan, dtype=float)
            padded_history[: len(treatment_history)] = treatment_history
            designs.append(
                np.concatenate(
                    (padded_clinical, static, padded_history, [float(timestamp)])
                )
            )
            outcomes.append(float(outcome[timestamp]))
            treatments.append(float(treatment[timestamp]))
            keys.append((patient_id, int(timestamp)))
    return np.asarray(designs), np.asarray(outcomes), np.asarray(treatments), keys


def run(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.input).open("rb") as handle:
        raw = pickle.load(handle)
    data = {str(patient_id): datum for patient_id, datum in raw.items()}
    with Path(args.header).open("rb") as handle:
        temporal_features = [str(name) for name in pickle.load(handle)]
    source_ids = list(data)
    labels = _load_split(args.split_manifest, source_ids)
    timestamps = list(range(args.evaluation_start, args.evaluation_end + 1))
    if args.baseline_end >= args.evaluation_start:
        raise ValueError("baseline_end must precede evaluation_start")
    train_ids = [patient_id for patient_id in source_ids if labels[patient_id] == "train"]
    evaluation_ids = [patient_id for patient_id in source_ids if labels[patient_id] == args.split]
    train_x, train_y, train_a, _ = _design_rows(
        data, train_ids, temporal_features=temporal_features,
        static_features=args.static_features, baseline_end=args.baseline_end,
        evaluation_end=args.evaluation_end,
        history_mode=args.history_mode,
        timestamps=timestamps,
    )
    evaluation_x, factual, _, evaluation_keys = _design_rows(
        data, evaluation_ids, temporal_features=temporal_features,
        static_features=args.static_features, baseline_end=args.baseline_end,
        evaluation_end=args.evaluation_end,
        history_mode=args.history_mode,
        timestamps=timestamps,
    )
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesRegressor(
            n_estimators=args.estimators,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        ),
    )
    model.fit(np.column_stack((train_x, train_a)), train_y)

    # Oracle truth is deliberately loaded only after fitting the factual model.
    truth = _load_truth(args.truth, source_ids)
    pred_y0 = model.predict(np.column_stack((evaluation_x, np.zeros(len(evaluation_x)))))
    pred_y1 = model.predict(np.column_stack((evaluation_x, np.ones(len(evaluation_x)))))
    predicted_ite = pred_y1 - pred_y0
    truth_ite = np.asarray(
        [
            float(truth[patient_id]["y1"][timestamp])
            - float(truth[patient_id]["y0"][timestamp])
            for patient_id, timestamp in evaluation_keys
        ]
    )
    causal = _metrics(truth_ite, predicted_ite)
    rows = []
    for timestamp in timestamps:
        selected = np.asarray([key[1] == timestamp for key in evaluation_keys])
        rows.append({"timestamp": timestamp, **_metrics(truth_ite[selected], predicted_ite[selected])})
    report = {
        "model": "G-computation (ExtraTrees outcome regression)",
        "split": args.split,
        "n_patients": len(evaluation_ids),
        "timestamps": timestamps,
        "causal_effect": causal,
        "factual_outcome": {
            "rmse": float(np.sqrt(np.mean(np.square(model.predict(np.column_stack((evaluation_x, np.asarray([data[p]["treatment"][t] for p, t in evaluation_keys], dtype=float)))) - factual)))),
        },
        "methodology": {
            "counterfactual_truth_used_during_training": False,
            "shared_split_manifest": str(Path(args.split_manifest).resolve()),
            "estimand": "conditional point-treatment effect Y_t(1)-Y_t(0) at observed history",
            "baseline_end": args.baseline_end,
            "evaluation_start": args.evaluation_start,
            "evaluation_end": args.evaluation_end,
            "followup_temporal_covariates_available": args.history_mode == "observed",
            "history_mode": args.history_mode,
            "static_features": list(args.static_features),
            "outcome_model": "ExtraTreesRegressor",
            "n_estimators": args.estimators,
            "min_samples_leaf": args.min_samples_leaf,
        },
    }
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "metrics_by_timestep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.md").write_text(
        f"# G-computation semi-synthetic {args.split} evaluation\n\n"
        f"- Patients: {report['n_patients']}\n"
        f"- PEHE: {causal['pehe']:.6f}\n"
        f"- Normalized PEHE: {causal['normalized_pehe']:.6f}\n"
        f"- ATE absolute error: {causal['ate_absolute_error']:.6f}\n"
        f"- ITE correlation: {causal['ite_correlation']:.6f}\n",
        encoding="utf-8",
    )
    return report


