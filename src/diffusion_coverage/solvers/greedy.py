from __future__ import annotations

from time import perf_counter

import numpy as np

from diffusion_coverage.graph.feasibility import edge_is_continuous, evaluate_assignment
from diffusion_coverage.problem import GraphInstance, GraphSolution
from diffusion_coverage.solvers.base import BaseGraphSolver


class GreedySolver(BaseGraphSolver):
    """Sequential heuristic: preserve assigned-neighbour continuity, then minimize q motion."""

    def solve(self, instance: GraphInstance, **kwargs) -> GraphSolution:
        del kwargs
        start = perf_counter()
        colours = np.full(instance.num_nodes, -1, dtype=np.int64)
        order = self._node_order(instance)
        adjacency = self._adjacency(instance)

        for node in order:
            best_colour = None
            best_score: tuple[int, float, int] | None = None
            for colour in instance.valid_colours(int(node)):
                continuity_breaks = 0
                local_cost = 0.0
                for edge_id, nbr, reverse in adjacency[int(node)]:
                    nbr_colour = colours[nbr]
                    if nbr_colour < 0:
                        continue
                    c_src, c_dst = (nbr_colour, colour) if reverse else (colour, nbr_colour)
                    if not edge_is_continuous(instance, edge_id, int(c_src), int(c_dst)):
                        continuity_breaks += 1
                    delta = instance.q_candidates[nbr, nbr_colour] - instance.q_candidates[node, colour]
                    local_cost += float(delta @ delta)
                score = (continuity_breaks, local_cost, int(colour))
                if best_score is None or score < best_score:
                    best_score = score
                    best_colour = int(colour)
            colours[int(node)] = int(best_colour)

        elapsed = perf_counter() - start
        return evaluate_assignment(
            instance,
            colours,
            solve_time=elapsed,
            search_nodes=0,
            evaluated_assignments=1,
            certified_optimal=False,
            metadata={"time_to_first_feasible": elapsed, "time_to_best": elapsed, "time_to_certified": None},
        )

    def _node_order(self, instance: GraphInstance) -> list[int]:
        degrees = np.bincount(instance.edge_index.reshape(-1), minlength=instance.num_nodes)
        return sorted(range(instance.num_nodes), key=lambda i: (not instance.boundary_mask[i], -degrees[i], i))

    def _adjacency(self, instance: GraphInstance) -> list[list[tuple[int, int, bool]]]:
        adjacency: list[list[tuple[int, int, bool]]] = [[] for _ in range(instance.num_nodes)]
        for edge_id, (src, dst) in enumerate(instance.edge_index.T):
            adjacency[int(src)].append((edge_id, int(dst), False))
            adjacency[int(dst)].append((edge_id, int(src), True))
        return adjacency
