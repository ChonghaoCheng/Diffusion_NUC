from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraphInstance:
    """Maximal-continuity graph with node-local IK-sheet colour candidates."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    valid_colour_mask: np.ndarray
    boundary_mask: np.ndarray
    q_candidates: np.ndarray
    colour_compatibility: np.ndarray | None = None
    manipulability: np.ndarray | None = None
    joint_limit_margin: np.ndarray | None = None
    graph_id: str | None = None
    robot_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        n = self.node_features.shape[0]
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.valid_colour_mask.shape[0] != n:
            raise ValueError("valid_colour_mask must have shape [N, C_max]")
        if self.boundary_mask.shape != (n,):
            raise ValueError("boundary_mask must have shape [N]")
        if self.q_candidates.shape[:2] != self.valid_colour_mask.shape:
            raise ValueError("q_candidates must have shape [N, C_max, nq]")
        if self.edge_features.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_features must have shape [E, F_e]")
        if self.colour_compatibility is not None:
            expected = (self.edge_index.shape[1], self.num_colours, self.num_colours)
            if self.colour_compatibility.shape != expected:
                raise ValueError("colour_compatibility must have shape [E, C_max, C_max]")
        if np.any(self.valid_colour_mask.sum(axis=1) == 0):
            raise ValueError("every node must have at least one valid colour")

    @property
    def num_nodes(self) -> int:
        return int(self.valid_colour_mask.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_colours(self) -> int:
        return int(self.valid_colour_mask.shape[1])

    @property
    def nq(self) -> int:
        return int(self.q_candidates.shape[2])

    @property
    def boundary_nodes(self) -> np.ndarray:
        return np.flatnonzero(self.boundary_mask)

    def valid_colours(self, node: int) -> list[int]:
        return np.flatnonzero(self.valid_colour_mask[node]).astype(int).tolist()
