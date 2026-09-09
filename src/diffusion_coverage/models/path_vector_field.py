from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from diffusion_coverage.models.surface_encoder import PointNetSurfaceEncoder


@dataclass(frozen=True)
class PathVectorFieldConfig:
    path_dim: int = 3
    condition_dim: int = 2
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    time_embedding_dim: int = 32
    position_embedding_dim: int = 16
    dropout: float = 0.0
    path_self_attention: bool = False

    def __post_init__(self) -> None:
        if self.path_dim < 1:
            raise ValueError("path_dim must be positive")
        if self.condition_dim < 2:
            raise ValueError("condition_dim must be at least two")
        if self.hidden_dim < 16 or self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("invalid model dimensions")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.time_embedding_dim % 2 or self.position_embedding_dim % 2:
            raise ValueError("Fourier embedding dimensions must be even")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class FourierEmbedding(nn.Module):
    def __init__(self, dimension: int, max_frequency: float = 1000.0) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(0.0, math.log(max_frequency), dimension // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        angles = values[..., None] * self.frequencies * (2.0 * math.pi)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class PathConditioningBlock(nn.Module):
    def __init__(self, config: PathVectorFieldConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.path_self_attention = config.path_self_attention
        if self.path_self_attention:
            self.self_norm = nn.LayerNorm(hidden)
            self.self_attention = nn.MultiheadAttention(
                hidden, config.num_heads, dropout=config.dropout, batch_first=True
            )
        self.local_norm = nn.LayerNorm(hidden)
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden, 2 * hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(2 * hidden, hidden, kernel_size=3, padding=1),
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(
            hidden, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * hidden, hidden),
        )

    def forward(
        self,
        path_tokens: torch.Tensor,
        surface_tokens: torch.Tensor,
        path_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if path_mask is not None:
            path_tokens = path_tokens.masked_fill(~path_mask[..., None], 0.0)
        if self.path_self_attention:
            query = self.self_norm(path_tokens)
            attended, _ = self.self_attention(
                query,
                query,
                query,
                key_padding_mask=None if path_mask is None else ~path_mask,
                need_weights=False,
            )
            path_tokens = path_tokens + attended
            if path_mask is not None:
                path_tokens = path_tokens.masked_fill(~path_mask[..., None], 0.0)
        local = self.local_norm(path_tokens).transpose(1, 2)
        local = self.local_conv[1](self.local_conv[0](local))
        if path_mask is not None:
            local = local.masked_fill(~path_mask[:, None, :], 0.0)
        path_tokens = path_tokens + self.local_conv[2](local).transpose(1, 2)
        if path_mask is not None:
            path_tokens = path_tokens.masked_fill(~path_mask[..., None], 0.0)
        query = self.cross_norm(path_tokens)
        attended, _ = self.cross_attention(query, surface_tokens, surface_tokens, need_weights=False)
        path_tokens = path_tokens + attended
        path_tokens = path_tokens + self.feed_forward(self.feed_forward_norm(path_tokens))
        if path_mask is not None:
            path_tokens = path_tokens.masked_fill(~path_mask[..., None], 0.0)
        return path_tokens


class PathVectorField(nn.Module):
    """Conditional velocity field over padded path or configuration tokens."""

    def __init__(self, config: PathVectorFieldConfig | None = None) -> None:
        super().__init__()
        self.config = PathVectorFieldConfig() if config is None else config
        hidden = self.config.hidden_dim
        self.surface_encoder = PointNetSurfaceEncoder(hidden_dim=hidden)
        self.time_embedding = FourierEmbedding(self.config.time_embedding_dim)
        self.position_embedding = FourierEmbedding(self.config.position_embedding_dim)
        self.path_input = nn.Linear(self.config.path_dim + self.config.position_embedding_dim, hidden)
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.config.time_embedding_dim + self.config.condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            PathConditioningBlock(self.config) for _ in range(self.config.num_layers)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.config.path_dim))

    def forward(
        self,
        path: torch.Tensor,
        time: torch.Tensor,
        surface: torch.Tensor,
        condition: torch.Tensor,
        path_mask: torch.Tensor | None = None,
        path_arclength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if path.ndim != 3 or path.shape[-1] != self.config.path_dim:
            raise ValueError(f"path must have shape [B, M, {self.config.path_dim}]")
        batch_size, num_waypoints, _ = path.shape
        if time.shape not in {(batch_size,), (batch_size, 1)}:
            raise ValueError("time must have shape [B] or [B, 1]")
        if condition.shape != (batch_size, self.config.condition_dim):
            raise ValueError(
                f"condition must have shape [B, {self.config.condition_dim}]"
            )
        if path_mask is not None and path_mask.shape != (batch_size, num_waypoints):
            raise ValueError("path_mask must have shape [B, M]")
        if path_arclength is not None and path_arclength.shape != (batch_size, num_waypoints):
            raise ValueError("path_arclength must have shape [B, M]")
        surface_tokens, surface_global = self.surface_encoder(surface)
        if path_arclength is None:
            positions = torch.linspace(0.0, 1.0, num_waypoints, device=path.device, dtype=path.dtype)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            positions = path_arclength.to(device=path.device, dtype=path.dtype)
        position_code = self.position_embedding(positions)
        path_tokens = self.path_input(torch.cat((path, position_code), dim=-1))
        time_code = self.time_embedding(time.reshape(batch_size))
        global_condition = surface_global + self.condition_mlp(torch.cat((time_code, condition), dim=-1))
        path_tokens = path_tokens + global_condition[:, None, :]
        surface_tokens = surface_tokens + global_condition[:, None, :]
        for block in self.blocks:
            path_tokens = block(path_tokens, surface_tokens, path_mask)
        velocity = self.output(path_tokens)
        if path_mask is not None:
            velocity = velocity.masked_fill(~path_mask[..., None], 0.0)
        return velocity
