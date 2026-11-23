"""Prepare the fixed training weights used by the sarcoma experiment."""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.causal_weights import LongitudinalWeightSpec, compose_longitudinal_training_weights
from src.surgery_rt_effects import build_training_weight_components, config_from_mapping
from src.surgery_rt_training_data import build_surgery_rt_preop_two_arm_training_data
from src.weighting_diagnostics import longitudinal_balance_diagnostics, longitudinal_weight_diagnostics


def prepare_training_weights(person_period_path, model_config_path, causal_config_path, output_dir):
    model_config = yaml.safe_load(Path(model_config_path).read_text())
    causal_config = yaml.safe_load(Path(causal_config_path).read_text())

    # Fit weights on the same development patients used for model training.
    training_data = build_surgery_rt_preop_two_arm_training_data(
        person_period_path,
        model_config["feature_groups"],
        seq_len=model_config["seq_len"],
        cohort_contract=model_config["cohort_contract"],
        split_seed=model_config["split_seed"],
        validation_fraction=model_config.get("validation_fraction", 0.2),
        test_fraction=model_config["test_fraction"],
    )
    included_ids = {str(patient["patient_id"]) for patient in training_data["patients"]}
    development_ids = [
        str(patient["patient_id"])
        for patient in training_data["patients"]
        if patient["split"] == "dev"
    ]

    person_period = pd.read_csv(person_period_path)
    patient_column = causal_config["surgery_rt_analysis"].get("patient_id_col", "patient_id")
    person_period = person_period.loc[
        person_period[patient_column].astype(str).isin(included_ids)
    ].copy()

    effect_config = config_from_mapping(causal_config)
    components, component_metadata = build_training_weight_components(
        person_period,
        effect_config,
        training_patient_ids=development_ids,
    )

    # Fixed selected specification: overlap x IPCW, truncated at 95%.
    specification = LongitudinalWeightSpec(
        scheme="ow_ipcw",
        truncation_target="combined",
        upper_quantile=0.95,
    )
    table = compose_longitudinal_training_weights(components, specification)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "training_weights.csv"
    metadata_path = output / "training_weights_metadata.json"
    diagnostics_path = output / "training_weight_diagnostics.json"
    config_path = output / "config.yaml"

    table.frame.to_csv(weights_path, index=False)
    metadata = {**table.metadata, "components": component_metadata}
    metadata_path.write_text(json.dumps(metadata, indent=2))

    diagnostics = longitudinal_weight_diagnostics(table.frame)
    diagnostics["balance_by_time"] = longitudinal_balance_diagnostics(
        table.frame, effect_config.baseline_covariates
    )
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    config_path.write_text(yaml.safe_dump(model_config, sort_keys=False))

    return config_path, weights_path, metadata_path
