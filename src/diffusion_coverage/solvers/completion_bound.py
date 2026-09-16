from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import inf
from typing import Iterable

import numpy as np

from diffusion_coverage.coverage.episode_summary import EpisodeEdgeSummary


@dataclass(frozen=True)
class CompletionEdge:
    edge_id: int
    start: int
    end: int
    summary: EpisodeEdgeSummary
    joint_cost: float = 0.0


@dataclass(frozen=True)
class CompletionBoundResult:
    future_lower_bound: float
    total_lower_bound: float
    target_distances: np.ndarray
    reachable_uncovered_weight: float


def optimistic_edge_repeat_cost(
    edge: CompletionEdge, covered: np.ndarray, weights: np.ndarray
) -> float:
    d = edge.summary.episode_counts.astype(np.int64) - edge.summary.start_membership.astype(np.int64)
    if np.any(d < 0):
        raise ValueError("edge episode count must include its active start membership")
    value = float(np.dot(np.asarray(weights, dtype=np.float64)[covered], d[covered]))
    # Downward rounding protects admissibility in the presence of floating point noise.
    return max(0.0, float(np.nextafter(value / float(np.sum(weights)), -np.inf)))


def completion_repeat_lower_bound(
    *,
    node: int,
    covered: np.ndarray,
    repeat_error: float,
    used_on_segments: int,
    maximum_on_segments: int,
    weights: np.ndarray,
    edges: Iterable[CompletionEdge],
    missed_tolerance: float,
) -> CompletionBoundResult:
    area = np.asarray(weights, dtype=np.float64)
    seen = np.asarray(covered, dtype=bool)
    if seen.shape != area.shape:
        raise ValueError("covered and weights must have matching shape")
    if not 0.0 <= missed_tolerance < 1.0:
        raise ValueError("missed_tolerance must lie in [0, 1)")
    edge_list = tuple(edges)
    outgoing: dict[int, list[CompletionEdge]] = {}
    for edge in edge_list:
        outgoing.setdefault(edge.start, []).append(edge)

    start_state = (int(node), int(used_on_segments))
    distance: dict[tuple[int, int], float] = {start_state: 0.0}
    queue: list[tuple[float, int, int]] = [(0.0, *start_state)]
    target = np.full(len(area), np.inf, dtype=np.float64)
    while queue:
        value, current, used = heapq.heappop(queue)
        if value != distance.get((current, used)):
            continue
        for edge in outgoing.get(current, ()):
            new_used = used + edge.summary.off_to_on_count
            if new_used > maximum_on_segments:
                continue
            cost = optimistic_edge_repeat_cost(edge, seen, area)
            candidate = value + cost
            hit = edge.summary.footprint & ~seen
            target[hit] = np.minimum(target[hit], candidate)
            key = (edge.end, new_used)
            if candidate < distance.get(key, inf):
                distance[key] = candidate
                heapq.heappush(queue, (candidate, edge.end, new_used))

    total = float(area.sum())
    already = float(area[seen].sum())
    required = (1.0 - missed_tolerance) * total
    finite_new = (~seen) & np.isfinite(target)
    reachable = already + float(area[finite_new].sum())
    if reachable + 1e-15 * total < required:
        future = inf
    elif already >= required - 1e-15 * total:
        future = 0.0
    else:
        order = np.argsort(target, kind="stable")
        cumulative = already
        future = inf
        for index in order:
            if seen[index] or not np.isfinite(target[index]):
                continue
            cumulative += float(area[index])
            if cumulative + 1e-15 * total >= required:
                future = max(0.0, float(np.nextafter(target[index], -np.inf)))
                break
    return CompletionBoundResult(
        future_lower_bound=future,
        total_lower_bound=float(repeat_error + future),
        target_distances=target,
        reachable_uncovered_weight=float(area[finite_new].sum()),
    )
