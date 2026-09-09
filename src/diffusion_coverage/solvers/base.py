from __future__ import annotations

from abc import ABC, abstractmethod

from diffusion_coverage.problem import GraphInstance, GraphSolution


class BaseGraphSolver(ABC):
    @abstractmethod
    def solve(self, instance: GraphInstance, **kwargs) -> GraphSolution:
        raise NotImplementedError
