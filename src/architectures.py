"""Non-recurrent architecture used by the DTCyGAN ablation study."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _augment(
    value: torch.Tensor,
    mask: torch.Tensor,
    include_missingness_indicators: bool,
    missingness_indicator_encoding: str,
) -> torch.Tensor:
    if missingness_indicator_encoding == "raw":
        indicator = mask
    elif missingness_indicator_encoding == "centered":
        indicator = mask - 1.0
    else:
        raise ValueError(
            "missingness_indicator_encoding must be 'raw' or 'centered'"
        )
    return (
        torch.cat((value, indicator), dim=-1)
        if include_missingness_indicators
        else value
    )


class TemporalMLPGenerator(nn.Module):
    """Time-local generator with no recurrent state."""

    def __init__(
        self,
        input_dim: int,
        cond_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        include_missingness_indicators: bool = True,
        missingness_indicator_encoding: str = "raw",
    ) -> None:
        super().__init__()
        self.include_missingness_indicators = include_missingness_indicators
        self.missingness_indicator_encoding = missingness_indicator_encoding
        multiplier = 2 if include_missingness_indicators else 1
        width = max(1, int(hidden_dim))
        depth = max(1, int(layers))
        layers: list[nn.Module] = [nn.Linear((input_dim + cond_input_dim) * multiplier, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, None]:
        del lengths, hidden
        combined = torch.cat(
            (
                _augment(x, m_x, include_missingness_indicators=self.include_missingness_indicators, missingness_indicator_encoding=self.missingness_indicator_encoding),
                _augment(cond, m_cond, include_missingness_indicators=self.include_missingness_indicators, missingness_indicator_encoding=self.missingness_indicator_encoding),
            ),
            dim=-1,
        )
        return self.network(combined), None


class TemporalMLPDiscriminator(nn.Module):
    """Time-local encoder with masked temporal mean pooling."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int,
        layers: int,
        include_missingness_indicators: bool = True,
        missingness_indicator_encoding: str = "raw",
    ) -> None:
        super().__init__()
        self.include_missingness_indicators = include_missingness_indicators
        self.missingness_indicator_encoding = missingness_indicator_encoding
        multiplier = 2 if include_missingness_indicators else 1
        width = max(1, int(hidden_dim))
        depth = max(1, int(layers))
        layers: list[nn.Module] = [nn.Linear((input_dim + cond_dim) * multiplier, width), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.LeakyReLU(0.2)))
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, 1)

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        time_resolved: bool = False,
    ) -> torch.Tensor:
        combined = torch.cat(
            (
                _augment(x, m_x, include_missingness_indicators=self.include_missingness_indicators, missingness_indicator_encoding=self.missingness_indicator_encoding),
                _augment(cond, m_cond, include_missingness_indicators=self.include_missingness_indicators, missingness_indicator_encoding=self.missingness_indicator_encoding),
            ),
            dim=-1,
        )
        encoded = self.encoder(combined)
        if time_resolved:
            return self.classifier(encoded).squeeze(-1)
        if lengths is None:
            pooled = encoded.mean(dim=1)
        else:
            steps = torch.arange(encoded.shape[1], device=encoded.device)[None, :]
            active = (steps < lengths[:, None]).to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled).squeeze(-1)


class TemporalMLPOutcomePredictor(TemporalMLPGenerator):
    """Non-recurrent interval-level outcome head."""

    def forward(
        self,
        x: torch.Tensor,
        m_x: torch.Tensor,
        cond: torch.Tensor,
        m_cond: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _ = super().forward(x, m_x, cond, m_cond, lengths)
        return output
