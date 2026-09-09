from __future__ import annotations

import numpy as np

from diffusion_coverage.problem import GraphInstance, GraphSolution


def edge_is_continuous(instance: GraphInstance, edge_id: int, c_src: int, c_dst: int) -> bool:
    if instance.colour_compatibility is None:
        return c_src == c_dst
    return bool(instance.colour_compatibility[edge_id, c_src, c_dst])


def is_valid_assignment(instance: GraphInstance, colours: np.ndarray) -> bool:
    if colours.shape != (instance.num_nodes,):
        return False
    for i, c in enumerate(colours):
        if c < 0 or c >= instance.num_colours or not instance.valid_colour_mask[i, c]:
            return False
    return True


def evaluate_assignment(
    instance: GraphInstance,
    colours: np.ndarray,
    *,
    solve_time: float = 0.0,
    search_nodes: int = 0,
    evaluated_assignments: int = 0,
    certified_optimal: bool = False,
    metadata: dict | None = None,
) -> GraphSolution:
    colours = np.asarray(colours, dtype=np.int64)
    if not is_valid_assignment(instance, colours):
        return GraphSolution.infeasible(
            instance.num_nodes,
            solve_time=solve_time,
            search_nodes=search_nodes,
            evaluated_assignments=evaluated_assignments,
            certified_optimal=certified_optimal,
            metadata=metadata or {},
        )

    continuity_adj: list[list[int]] = [[] for _ in range(instance.num_nodes)]
    joint_motion_cost = 0.0
    for e, (src, dst) in enumerate(instance.edge_index.T):
        c_src = int(colours[src])
        c_dst = int(colours[dst])
        delta = instance.q_candidates[dst, c_dst] - instance.q_candidates[src, c_src]
        joint_motion_cost += float(delta @ delta)
        if edge_is_continuous(instance, e, c_src, c_dst):
            continuity_adj[int(src)].append(int(dst))
            continuity_adj[int(dst)].append(int(src))

    components = _count_components(continuity_adj)
    num_lift_offs = max(0, components - 1)

    manip_cost = 0.0
    if instance.manipulability is not None:
        vals = instance.manipulability[np.arange(instance.num_nodes), colours]
        manip_cost = float(np.sum(1.0 / np.maximum(vals, 1e-6)))

    limit_cost = 0.0
    if instance.joint_limit_margin is not None:
        margins = instance.joint_limit_margin[np.arange(instance.num_nodes), colours]
        limit_cost = float(np.sum(1.0 / np.maximum(margins, 1e-6)))

    return GraphSolution(
        colours=colours.copy(),
        feasible=True,
        num_lift_offs=num_lift_offs,
        joint_motion_cost=joint_motion_cost,
        manipulability_cost=manip_cost,
        joint_limit_cost=limit_cost,
        solve_time=solve_time,
        search_nodes=search_nodes,
        evaluated_assignments=evaluated_assignments,
        certified_optimal=certified_optimal,
        metadata=metadata or {},
    )


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
