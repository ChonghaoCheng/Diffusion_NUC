from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GraphSolution:
    colours: np.ndarray
    feasible: bool
    num_lift_offs: int
    joint_motion_cost: float
    manipulability_cost: float = 0.0
    joint_limit_cost: float = 0.0
    solve_time: float = 0.0
    search_nodes: int = 0
    evaluated_assignments: int = 0
    certified_optimal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def objective(self) -> tuple[int, float]:
        return objective_tuple(self)

    def better_than(self, other: GraphSolution | None) -> bool:
        if not self.feasible:
            return False
        if other is None or not other.feasible:
            return True
        return self.objective < other.objective

    @classmethod
    def infeasible(cls, num_nodes: int, **kwargs: Any) -> GraphSolution:
        return cls(
            colours=np.full(num_nodes, -1, dtype=np.int64),
            feasible=False,
            num_lift_offs=10**9,
            joint_motion_cost=float("inf"),
            **kwargs,
        )


def objective_tuple(solution: GraphSolution) -> tuple[int, float]:
    return (solution.num_lift_offs, solution.joint_motion_cost)


def lexicographic_less(a: tuple[int, float], b: tuple[int, float]) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])
