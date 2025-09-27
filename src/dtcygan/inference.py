"""Inference utilities operating on synthetic datasets and trained checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from dtcygan.training import (
    Config,
    SyntheticSequenceDataset,
    LSTMGenerator,
    randomize_multiple_one_hot,
    resolve_device,
)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Config, dict, torch.nn.Module]:
    checkpoint = torch.load(path, map_location=device)
    cfg = Config(**checkpoint["config"])
    metadata = checkpoint.get("metadata") or {}
    clin_dim = metadata.get("clin_dim")
    treat_dim = metadata.get("treat_dim")
    cond_dim = metadata.get("cond_dim")
    if None in {clin_dim, treat_dim, cond_dim}:
        raise ValueError("Checkpoint metadata must include clin_dim, treat_dim, and cond_dim.")

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
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return SyntheticSequenceDataset(
        data,
        seq_len,
        spec_clinical=spec_clinical,
        spec_treatment=spec_treatment,
    )


def summarize_sequence(tensor: torch.Tensor, mask: torch.Tensor) -> Dict[str, List[float]]:
    array = tensor.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    mask_sums = mask_np.sum(axis=1, keepdims=True).clip(min=1.0)
    mean = (array * mask_np).sum(axis=1) / mask_sums

    last_vectors: List[List[float]] = []
    valid_steps = mask_np.sum(axis=2) > 0
    for seq, val_flags in zip(array, valid_steps):
        if val_flags.any():
            last_idx = int(val_flags.nonzero()[0][-1])
            last_vectors.append(seq[last_idx].tolist())
        else:
            last_vectors.append(seq[-1].tolist())

    return {"mean": mean.tolist(), "last": last_vectors}


def generate_counterfactuals(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    scenario_names: Optional[List[str]] = None,
) -> Path:
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

        x_clin = x_clin * step_mask
        mask_clin = mask_clin * step_mask
        cond_actual = cond_actual * step_mask
        mask_actual = mask_actual * step_mask

        scenarios = {"actual": cond_actual}
        cond_random = randomize_multiple_one_hot(cond_actual, mask_actual, extra_ones=1) * step_mask
        scenarios["random"] = cond_random
        if scenario_names:
            for name in scenario_names:
                if name not in scenarios:
                    scenarios[name] = cond_random

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
