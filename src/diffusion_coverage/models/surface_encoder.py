from __future__ import annotations

import torch
from torch import nn


class PointNetSurfaceEncoder(nn.Module):
    """PointNet-style surface encoder returning local tokens and a global code."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, surface: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if surface.ndim != 3 or surface.shape[-1] != 6:
            raise ValueError("surface must have shape [B, P, 6]")
        tokens = self.point_mlp(surface)
        pooled = torch.cat((tokens.amax(dim=1), tokens.mean(dim=1)), dim=-1)
        return tokens, self.global_mlp(pooled)
