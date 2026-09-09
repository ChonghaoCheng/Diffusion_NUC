from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from diffusion_coverage.graph.feasibility import edge_is_continuous, evaluate_assignment, is_valid_assignment
from diffusion_coverage.problem import GraphInstance, GraphSolution
from diffusion_coverage.problem.graph_solution import lexicographic_less
from diffusion_coverage.solvers.base import BaseGraphSolver


@dataclass
class ExactSearchStats:
    search_nodes: int = 0
    pruned_search_nodes: int = 0
    evaluated_assignments: int = 0
    time_to_first_feasible: float | None = None
    time_to_best: float | None = None
    time_to_certified: float | None = None


class ExactSolver(BaseGraphSolver):
    """Exhaustive lexicographic solver with branch-order hooks for learned priors."""

    def __init__(self, search_boundary_only: bool = True) -> None:
        self.search_boundary_only = search_boundary_only
        self.stats = ExactSearchStats()

    def solve(
        self,
        instance: GraphInstance,
        incumbent: GraphSolution | None = None,
        branch_prior: np.ndarray | None = None,
    ) -> GraphSolution:
        start = perf_counter()
        self.stats = ExactSearchStats()
        best = self._validated_incumbent(instance, incumbent)
        incumbent_accepted = best is not None
        if best is not None:
            self.stats.time_to_first_feasible = 0.0
            self.stats.time_to_best = 0.0

        order = self._node_order(instance)
        colours = np.full(instance.num_nodes, -1, dtype=np.int64)
        completion_nodes = [i for i in range(instance.num_nodes) if i not in set(order)]

        def dfs(depth: int) -> None:
            nonlocal best
            lower_bound = self._partial_lower_bound(instance, colours)
            if best is not None and not lexicographic_less(lower_bound, best.objective):
                self.stats.pruned_search_nodes += 1
                return

            self.stats.search_nodes += 1
            if depth == len(order):
                for completed in self._complete_assignments(instance, colours, completion_nodes):
                    self.stats.evaluated_assignments += 1
                    sol = evaluate_assignment(instance, completed)
                    if sol.better_than(best):
                        elapsed = perf_counter() - start
                        if best is None:
                            self.stats.time_to_first_feasible = elapsed
                        self.stats.time_to_best = elapsed
                        best = sol
                return

            node = int(order[depth])
            for colour in self._colour_order(instance, node, branch_prior):
                colours[node] = colour
                dfs(depth + 1)
                colours[node] = -1

        dfs(0)
        elapsed = perf_counter() - start
        self.stats.time_to_certified = elapsed
        if best is None:
            return GraphSolution.infeasible(
                instance.num_nodes,
                solve_time=elapsed,
                search_nodes=self.stats.search_nodes,
                evaluated_assignments=self.stats.evaluated_assignments,
                certified_optimal=True,
                metadata=self._stats_metadata(incumbent_accepted=incumbent_accepted),
            )

        best.solve_time = elapsed
        best.search_nodes = self.stats.search_nodes
        best.evaluated_assignments = self.stats.evaluated_assignments
        best.certified_optimal = True
        best.metadata = {
            **best.metadata,
            **self._stats_metadata(incumbent_accepted=incumbent_accepted),
        }
        return best

    def _node_order(self, instance: GraphInstance) -> list[int]:
        boundary = instance.boundary_nodes.tolist() if self.search_boundary_only else list(range(instance.num_nodes))
        boundary_set = set(boundary)
        internal = [i for i in range(instance.num_nodes) if i not in boundary_set]
        degrees = np.bincount(instance.edge_index.reshape(-1), minlength=instance.num_nodes)
        boundary = sorted(boundary, key=lambda i: (-degrees[i], i))
        internal = sorted(internal, key=lambda i: (-degrees[i], i))
        return boundary if self.search_boundary_only else boundary + internal

    def _colour_order(
        self,
        instance: GraphInstance,
        node: int,
        branch_prior: np.ndarray | None,
    ) -> list[int]:
        valid = instance.valid_colours(node)
        if branch_prior is None:
            return valid
        scores = np.asarray(branch_prior[node], dtype=float)
        return sorted(valid, key=lambda c: (-scores[c], c))

    def _validated_incumbent(
        self,
        instance: GraphInstance,
        incumbent: GraphSolution | None,
    ) -> GraphSolution | None:
        if incumbent is None or not incumbent.feasible:
            return None
        colours = np.asarray(incumbent.colours, dtype=np.int64)
        if not is_valid_assignment(instance, colours):
            return None
        return evaluate_assignment(instance, colours, metadata={"source": "incumbent"})

    def _complete_assignments(
        self,
        instance: GraphInstance,
        partial_colours: np.ndarray,
        completion_nodes: list[int],
    ):
        if not completion_nodes:
            yield partial_colours.copy()
            return

        def complete(depth: int):
            if depth == len(completion_nodes):
                yield partial_colours.copy()
                return
            node = completion_nodes[depth]
            for colour in instance.valid_colours(node):
                partial_colours[node] = colour
                yield from complete(depth + 1)
                partial_colours[node] = -1

        yield from complete(0)

    def _partial_lower_bound(
        self,
        instance: GraphInstance,
        colours: np.ndarray,
    ) -> tuple[int, float]:
        potential_adj: list[list[int]] = [[] for _ in range(instance.num_nodes)]
        min_joint_cost = 0.0
        for edge_id, (src_raw, dst_raw) in enumerate(instance.edge_index.T):
            src = int(src_raw)
            dst = int(dst_raw)
            c_src = int(colours[src])
            c_dst = int(colours[dst])
            if c_src >= 0 and c_dst >= 0:
                can_connect = edge_is_continuous(instance, edge_id, c_src, c_dst)
            else:
                can_connect = self._edge_can_be_continuous(instance, edge_id, src, dst, c_src, c_dst)
            min_joint_cost += self._edge_min_joint_cost(instance, src, dst, c_src, c_dst)
            if can_connect:
                potential_adj[src].append(dst)
                potential_adj[dst].append(src)
        components = _count_components(potential_adj)
        return (max(0, components - 1), min_joint_cost)

    def _edge_can_be_continuous(
        self,
        instance: GraphInstance,
        edge_id: int,
        src: int,
        dst: int,
        c_src: int,
        c_dst: int,
    ) -> bool:
        src_colours = [c_src] if c_src >= 0 else instance.valid_colours(src)
        dst_colours = [c_dst] if c_dst >= 0 else instance.valid_colours(dst)
        for src_colour in src_colours:
            for dst_colour in dst_colours:
                if edge_is_continuous(instance, edge_id, int(src_colour), int(dst_colour)):
                    return True
        return False

    def _edge_min_joint_cost(
        self,
        instance: GraphInstance,
        src: int,
        dst: int,
        c_src: int,
        c_dst: int,
    ) -> float:
        src_colours = [c_src] if c_src >= 0 else instance.valid_colours(src)
        dst_colours = [c_dst] if c_dst >= 0 else instance.valid_colours(dst)
        best = float("inf")
        for src_colour in src_colours:
            for dst_colour in dst_colours:
                delta = instance.q_candidates[dst, dst_colour] - instance.q_candidates[src, src_colour]
                best = min(best, float(delta @ delta))
        return best

    def _stats_metadata(self, *, incumbent_accepted: bool) -> dict[str, float | int | bool | None]:
        return {
            "pruned_search_nodes": self.stats.pruned_search_nodes,
            "time_to_first_feasible": self.stats.time_to_first_feasible,
            "time_to_best": self.stats.time_to_best,
            "time_to_certified": self.stats.time_to_certified,
            "exact_search_time": self.stats.time_to_certified,
            "total_solve_time": self.stats.time_to_certified,
            "incumbent_accepted": incumbent_accepted,
        }


def _count_components(adj: list[list[int]]) -> int:
    seen = [False] * len(adj)
    components = 0
    for start in range(len(adj)):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nbr in adj[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(nbr)
    return components
