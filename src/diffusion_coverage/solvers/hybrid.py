from __future__ import annotations

from time import perf_counter

from diffusion_coverage.problem import GraphInstance, GraphSolution
from diffusion_coverage.solvers.base import BaseGraphSolver
from diffusion_coverage.solvers.exact import ExactSolver
from diffusion_coverage.solvers.greedy import GreedySolver


class GreedyGuidedExactSolver(BaseGraphSolver):
    """Greedy incumbent followed by the same exact branch-and-bound solver."""

    def __init__(
        self,
        greedy_solver: GreedySolver | None = None,
        exact_solver: ExactSolver | None = None,
    ) -> None:
        self.greedy_solver = greedy_solver or GreedySolver()
        self.exact_solver = exact_solver or ExactSolver()

    def solve(self, instance: GraphInstance, **kwargs) -> GraphSolution:
        del kwargs
        total_start = perf_counter()
        greedy = self.greedy_solver.solve(instance)
        proposal_time = perf_counter() - total_start
        incumbent = greedy if greedy.feasible else None
        exact = self.exact_solver.solve(instance, incumbent=incumbent)
        total_solve_time = perf_counter() - total_start

        initial_obj = incumbent.objective if incumbent is not None else None
        exact.metadata = {
            **exact.metadata,
            "initial_incumbent_objective": initial_obj,
            "final_objective": exact.objective if exact.feasible else None,
            "proposal_time": proposal_time,
            "exact_search_time": exact.solve_time,
            "total_solve_time": total_solve_time,
            "greedy_feasible": greedy.feasible,
            "greedy_num_lift_offs": greedy.num_lift_offs,
            "greedy_joint_motion_cost": greedy.joint_motion_cost,
        }
        exact.solve_time = total_solve_time
        return exact
