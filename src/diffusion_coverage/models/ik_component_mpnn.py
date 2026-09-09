from __future__ import annotations

import torch
from torch import nn


class IKComponentMPNN(nn.Module):
    """Scores numerical IK components after candidate-level message passing."""

    def __init__(self, input_dim: int = 17, hidden_dim: int = 128, layers: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.component_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        edge_index: torch.Tensor,
        component_index: torch.Tensor,
        num_components: int,
        candidate_graph_index: torch.Tensor,
        component_graph_index: torch.Tensor,
        num_graphs: int,
        component_budget: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(candidate_features)
        source, target = edge_index
        for message, norm in zip(self.messages, self.norms):
            aggregate = torch.zeros_like(hidden)
            degree = torch.zeros(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)
            if source.numel():
                aggregate.index_add_(0, target, hidden[source])
                aggregate.index_add_(0, source, hidden[target])
                ones = torch.ones(source.shape[0], 1, device=hidden.device, dtype=hidden.dtype)
                degree.index_add_(0, target, ones)
                degree.index_add_(0, source, ones)
            aggregate = aggregate / degree.clamp_min(1.0)
            hidden = norm(hidden + message(torch.cat((hidden, aggregate), dim=-1)))

        pooled = torch.zeros(num_components, hidden.shape[-1], device=hidden.device)
        counts = torch.zeros(num_components, 1, device=hidden.device)
        pooled.index_add_(0, component_index, hidden)
        counts.index_add_(
            0,
            component_index,
            torch.ones(hidden.shape[0], 1, device=hidden.device),
        )
        pooled = pooled / counts.clamp_min(1.0)
        graph_codes = torch.zeros(num_graphs, hidden.shape[-1], device=hidden.device)
        graph_counts = torch.zeros(num_graphs, 1, device=hidden.device)
        graph_codes.index_add_(0, candidate_graph_index, hidden)
        graph_counts.index_add_(
            0,
            candidate_graph_index,
            torch.ones(hidden.shape[0], 1, device=hidden.device),
        )
        graph_codes = graph_codes / graph_counts.clamp_min(1.0)
        graph_code = graph_codes[component_graph_index]
        budget_code = component_budget[:, None]
        return self.component_head(torch.cat((pooled, graph_code, budget_code), dim=-1)).squeeze(-1)
