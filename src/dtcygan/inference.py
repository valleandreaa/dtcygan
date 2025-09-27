"""Inference utilities operating on synthetic datasets and trained checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from dtcygan.training import (
    Config,
    SyntheticSequenceDataset,
    LSTMGenerator,
    random_condition,
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


def load_dataset(path: str | Path, seq_len: int) -> SyntheticSequenceDataset:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return SyntheticSequenceDataset(data, seq_len)


def summarize_sequence(tensor: torch.Tensor) -> Dict[str, List[float]]:
    array = tensor.detach().cpu().numpy()
    summary: Dict[str, List[float]] = {
        "mean": array.mean(axis=1).tolist(),
        "last": array[:, -1, :].tolist(),
    }
    return summary


def generate_counterfactuals(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    scenario_names: Optional[List[str]] = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, _, generator = load_checkpoint(checkpoint_path, device)
    dataset = load_dataset(dataset_path, cfg.seq_len)

    records: List[Dict[str, object]] = []
    for idx in range(len(dataset)):
        patient = dataset.patients[idx]
        data = dataset[idx]
        x_clin = data["x_clin"].unsqueeze(0).to(device)
        cond_actual = data["actual_treatment"].unsqueeze(0).to(device)
        mask = data["mask_clin"].unsqueeze(0).to(device)
        step_mask = (mask.sum(dim=3, keepdim=True) > 0).float()
        cond_actual = cond_actual * step_mask

        scenarios = {"actual": cond_actual}
        cond_random = random_condition(cond_actual) * step_mask
        scenarios["random"] = cond_random
        if scenario_names:
            for name in scenario_names:
                if name not in scenarios:
                    scenarios[name] = cond_random

        for scenario, cond_tensor in scenarios.items():
            with torch.no_grad():
                fake_treat = generator(x_clin, cond_tensor) * step_mask
            summary = summarize_sequence(fake_treat)
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
