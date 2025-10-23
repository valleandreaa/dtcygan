"""Inference utilities operating on synthetic datasets and trained checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from dtcygan.training import (
    Config,
    SyntheticSequenceDataset,
    LSTMGenerator,
    randomize_multiple_one_hot,
    resolve_device,
)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Config, dict, torch.nn.Module]:
    ''' args:
    - path: filesystem location of the trained checkpoint [str | Path]
    - device: target torch device for loading weights [torch.device]

    return:
    - checkpoint_data: tuple containing config, metadata, and initialized generator [tuple[Config, dict, torch.nn.Module]]
    '''
    checkpoint = torch.load(path, map_location=device)
    cfg = Config(**checkpoint["config"])
    metadata = checkpoint.get("metadata") or {}
    dims = {key: metadata.get(key) for key in ("clin_dim", "treat_dim", "cond_dim")}
    if None in dims.values():
        raise ValueError("Checkpoint metadata must include clin_dim, treat_dim, and cond_dim.")

    clin_dim, treat_dim, cond_dim = (int(dims[key]) for key in ("clin_dim", "treat_dim", "cond_dim"))
    generator = LSTMGenerator(clin_dim, cond_dim, cfg.g_hidden, treat_dim, cfg.num_layers).to(device)
    generator.load_state_dict(checkpoint["Gx"])
    generator.eval()
    return cfg, metadata, generator


def load_dataset(
    path: str | Path,
    seq_len: int,
    spec_clinical: Optional[Dict[str, Any]] = None,
    spec_treatment: Optional[Dict[str, Any]] = None,
) -> SyntheticSequenceDataset:
    ''' args:
    - path: JSON dataset path containing synthetic patients [str | Path]
    - seq_len: maximum sequence length expected by the model [int]
    - spec_clinical: optional clinical feature specification [Optional[Dict[str, Any]]]
    - spec_treatment: optional treatment feature specification [Optional[Dict[str, Any]]]

    return:
    - dataset: synthetic sequence dataset aligned with model expectations [SyntheticSequenceDataset]
    '''
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return SyntheticSequenceDataset(
        data,
        seq_len,
        spec_clinical=spec_clinical,
        spec_treatment=spec_treatment,
    )


def summarize_sequence(tensor: torch.Tensor, mask: torch.Tensor) -> Dict[str, List[float]]:
    ''' args:
    - tensor: generated sequence tensor with shape [B, T, F] [torch.Tensor]
    - mask: binary mask indicating valid entries in the sequence [torch.Tensor]

    return:
    - summary: dictionary with mean and last-step vectors per batch item [Dict[str, List[float]]]
    '''
    array = tensor.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    mask_sums = mask_np.sum(axis=1, keepdims=True).clip(min=1.0)
    mean = (array * mask_np).sum(axis=1) / mask_sums

    valid_steps = mask_np.sum(axis=2) > 0
    last_vectors = [
        seq[np.flatnonzero(flags)[-1]].tolist() if np.flatnonzero(flags).size else seq[-1].tolist()
        for seq, flags in zip(array, valid_steps)
    ]

    return {"mean": mean.tolist(), "last": last_vectors}


def generate_counterfactuals(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    scenario_names: Optional[List[str]] = None,
) -> Path:
    ''' args:
    - checkpoint_path: trained model checkpoint to load [str | Path]
    - dataset_path: synthetic dataset to draw patients from [str | Path]
    - output_path: destination JSON file for generated counterfactuals [str | Path]
    - scenario_names: optional additional scenario labels to include [Optional[List[str]]]

    return:
    - result_path: resolved path to the saved counterfactual summaries [Path]
    '''
    device = resolve_device()
    cfg, metadata, generator = load_checkpoint(checkpoint_path, device)
    dataset = load_dataset(
        dataset_path,
        cfg.seq_len,
        spec_clinical=metadata.get("clinical_feature_spec"),
        spec_treatment=metadata.get("treatment_feature_spec"),
    )

    records: List[Dict[str, object]] = []
    for idx in range(len(dataset)):
        patient = dataset.patients[idx]
        data = dataset[idx]
        x_clin = data["x_clin"].unsqueeze(0).to(device)
        mask_clin = data["mask_clin"].unsqueeze(0).to(device)
        mask_treat = data["mask_treat"].unsqueeze(0).to(device)
        cond_actual = data["actual_treatment"].unsqueeze(0).to(device)
        mask_actual = data["mask_actual"].unsqueeze(0).to(device)

        step_mask = ((mask_clin.sum(dim=2, keepdim=True) + mask_treat.sum(dim=2, keepdim=True)) > 0).float()
        lengths = step_mask.squeeze(-1).sum(dim=1).clamp(min=1).long()

        x_clin, mask_clin, cond_actual, mask_actual = [
            tensor * step_mask for tensor in (x_clin, mask_clin, cond_actual, mask_actual)
        ]

        scenarios = {"actual": cond_actual}
        cond_random = randomize_multiple_one_hot(cond_actual, mask_actual, extra_ones=1) * step_mask
        scenarios["random"] = cond_random
        if scenario_names:
            for name in scenario_names:
                scenarios.setdefault(name, cond_random)

        for scenario, cond_tensor in scenarios.items():
            with torch.no_grad():
                fake_treat, _ = generator(x_clin, mask_clin, cond_tensor, mask_actual, lengths)
                fake_treat = fake_treat * mask_treat
            summary = summarize_sequence(fake_treat, mask_treat)
            records.append(
                {
                    "patient_id": patient.get("patient_id", f"P{idx:05d}"),
                    "scenario": scenario,
                    "mean": summary["mean"],
                    "last": summary["last"],
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    return output_path
