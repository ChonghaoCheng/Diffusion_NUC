from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from time import perf_counter
from typing import Any

import numpy as np

from diffusion_coverage.coverage.episode_summary import EpisodeState, apply_edge_summary, initial_episode_state
from diffusion_coverage.solvers.completion_bound import CompletionEdge, completion_repeat_lower_bound


@dataclass(frozen=True)
class SearchGraph:
    node_membership: tuple[np.ndarray, ...]
    edges: tuple[CompletionEdge, ...]
    weights: np.ndarray
    graph_hash: str


@dataclass(frozen=True)
class SearchLabel:
    node: int
    covered: np.ndarray
    membership: np.ndarray
    repeat_error: float
    joint_cost: float
    used_on_segments: int
    path: tuple[int, ...] = ()


@dataclass
class SearchMetrics:
    expanded: int = 0
    generated: int = 0
    dominance_pruned: int = 0
    repeat_pruned: int = 0
    segment_pruned: int = 0
    reachability_pruned: int = 0
    completion_bound_pruned: int = 0
    completion_bound_positive: int = 0
    completion_bound_stronger: int = 0
    completion_bound_calls: int = 0
    completion_bound_cache_hits: int = 0
    completion_bound_seconds: float = 0.0


@dataclass(frozen=True)
class SearchResult:
    incumbent: SearchLabel | None
    optimality_proved: bool
    termination: str
    elapsed_seconds: float
    metrics: SearchMetrics
    checkpoints: tuple[dict[str, Any], ...]
    first_solution_seconds: float | None
    mechanism_sample: dict[str, Any] | None


def search_history_graph(
    graph: SearchGraph,
    *,
    start_node: int,
    maximum_on_segments: int,
    missed_tolerance: float,
    repeat_tolerance: float,
    use_completion_bound: bool,
    wall_time_s: float,
    expanded_limit: int,
    checkpoint_times: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0, 300.0),
    tolerance: float = 1e-12,
) -> SearchResult:
    start_time = perf_counter()
    metrics = SearchMetrics()
    initial = initial_episode_state(graph.node_membership[start_node])
    first = SearchLabel(start_node, initial.covered, initial.membership, 0.0, 0.0, 1)
    serial = 0
    queue: list[tuple[int, float, int, SearchLabel]] = [(0, 0.0, serial, first)]
    outgoing: dict[int, list[CompletionEdge]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.start, []).append(edge)
    for values in outgoing.values():
        values.sort(key=lambda edge: (edge.joint_cost, edge.edge_id))

    pareto: dict[tuple[int, bytes, bytes, int], list[tuple[float, float]]] = {}
    _admit_pareto(first, pareto, tolerance)
    incumbent: SearchLabel | None = None
    first_solution_seconds: float | None = None
    mechanism_sample: dict[str, Any] | None = None
    cache: dict[tuple[int, bytes, int, str], Any] = {}
    checkpoints: list[dict[str, Any]] = []
    checkpoint_index = 0
    termination = "queue_exhausted"

    while queue:
        elapsed = perf_counter() - start_time
        while checkpoint_index < len(checkpoint_times) and elapsed >= checkpoint_times[checkpoint_index]:
            checkpoints.append(_checkpoint(checkpoint_times[checkpoint_index], incumbent, metrics))
            checkpoint_index += 1
        if elapsed >= wall_time_s:
            termination = "wall_time"
            break
        if metrics.expanded >= expanded_limit:
            termination = "expanded_limit"
            break
        _, _, _, label = heapq.heappop(queue)
        if incumbent is not None and _objective(label) >= _objective(incumbent):
            continue
        if _is_goal(label, graph.weights, missed_tolerance, repeat_tolerance, tolerance):
            if first_solution_seconds is None:
                first_solution_seconds = perf_counter() - start_time
            incumbent = label
            continue
        metrics.expanded += 1

        if not _reachable_area_sufficient(label, outgoing, graph.weights, missed_tolerance):
            metrics.reachability_pruned += 1
            continue
        if use_completion_bound:
            key = (label.node, np.packbits(label.covered).tobytes(), maximum_on_segments - label.used_on_segments, graph.graph_hash)
            bound = cache.get(key)
            if bound is None:
                bound_start = perf_counter()
                bound = completion_repeat_lower_bound(
                    node=label.node,
                    covered=label.covered,
                    repeat_error=0.0,
                    used_on_segments=label.used_on_segments,
                    maximum_on_segments=maximum_on_segments,
                    weights=graph.weights,
                    edges=graph.edges,
                    missed_tolerance=missed_tolerance,
                )
                metrics.completion_bound_seconds += perf_counter() - bound_start
                metrics.completion_bound_calls += 1
                cache[key] = bound
            else:
                metrics.completion_bound_cache_hits += 1
            if bound.future_lower_bound > tolerance:
                metrics.completion_bound_positive += 1
                metrics.completion_bound_stronger += 1
            if label.repeat_error + bound.future_lower_bound > repeat_tolerance + tolerance:
                metrics.completion_bound_pruned += 1
                if mechanism_sample is None and label.repeat_error <= repeat_tolerance + tolerance:
                    mechanism_sample = {
                        "node": label.node,
                        "covered_weight": float(graph.weights[label.covered].sum()),
                        "total_weight": float(graph.weights.sum()),
                        "repeat_error": label.repeat_error,
                        "future_lower_bound": bound.future_lower_bound,
                        "total_lower_bound": label.repeat_error + bound.future_lower_bound,
                        "finite_uncovered_targets": int(np.count_nonzero(~label.covered & np.isfinite(bound.target_distances))),
                        "path": list(label.path),
                    }
                continue

        for edge in outgoing.get(label.node, ()):
            metrics.generated += 1
            new_used = label.used_on_segments + edge.summary.off_to_on_count
            if new_used > maximum_on_segments:
                metrics.segment_pruned += 1
                continue
            try:
                state = apply_edge_summary(
                    EpisodeState(label.covered, label.membership, label.repeat_error),
                    edge.summary,
                    graph.weights,
                    atol=tolerance,
                )
            except ValueError:
                continue
            if state.repeat_error > repeat_tolerance + tolerance:
                metrics.repeat_pruned += 1
                continue
            child = SearchLabel(
                node=edge.end,
                covered=state.covered,
                membership=state.membership,
                repeat_error=state.repeat_error,
                joint_cost=label.joint_cost + edge.joint_cost,
                used_on_segments=new_used,
                path=label.path + (edge.edge_id,),
            )
            if incumbent is not None and _objective(child) >= _objective(incumbent):
                continue
            if not _admit_pareto(child, pareto, tolerance):
                metrics.dominance_pruned += 1
                continue
            serial += 1
            heapq.heappush(queue, (child.used_on_segments - 1, child.joint_cost, serial, child))

    elapsed = perf_counter() - start_time
    while checkpoint_index < len(checkpoint_times) and checkpoint_times[checkpoint_index] <= min(elapsed, wall_time_s):
        checkpoints.append(_checkpoint(checkpoint_times[checkpoint_index], incumbent, metrics))
        checkpoint_index += 1
    return SearchResult(
        incumbent=incumbent,
        optimality_proved=not queue and termination == "queue_exhausted",
        termination=termination,
        elapsed_seconds=elapsed,
        metrics=metrics,
        checkpoints=tuple(checkpoints),
        first_solution_seconds=first_solution_seconds,
        mechanism_sample=mechanism_sample,
    )


def _objective(label: SearchLabel) -> tuple[int, float]:
    return label.used_on_segments - 1, label.joint_cost


def _is_goal(label: SearchLabel, weights: np.ndarray, missed: float, repeated: float, tolerance: float) -> bool:
    miss = float(np.asarray(weights)[~label.covered].sum() / np.asarray(weights).sum())
    return miss <= missed + tolerance and label.repeat_error <= repeated + tolerance


def _admit_pareto(label: SearchLabel, table: dict, tolerance: float) -> bool:
    key = (label.node, np.packbits(label.covered).tobytes(), np.packbits(label.membership).tobytes(), label.used_on_segments)
    values = table.setdefault(key, [])
    for repeated, cost in values:
        if repeated <= label.repeat_error + tolerance and cost <= label.joint_cost + tolerance:
            return False
    table[key] = [
        (repeated, cost)
        for repeated, cost in values
        if not (label.repeat_error <= repeated + tolerance and label.joint_cost <= cost + tolerance)
    ] + [(label.repeat_error, label.joint_cost)]
    return True


def _reachable_area_sufficient(label: SearchLabel, outgoing: dict[int, list[CompletionEdge]], weights: np.ndarray, missed: float) -> bool:
    # Deliberately optimistic ordinary reachability: ignore costs and ON-segment use.
    reachable_nodes = {label.node}
    stack = [label.node]
    footprint = label.covered.copy()
    while stack:
        node = stack.pop()
        for edge in outgoing.get(node, ()):
            footprint |= edge.summary.footprint
            if edge.end not in reachable_nodes:
                reachable_nodes.add(edge.end)
                stack.append(edge.end)
    return float(np.asarray(weights)[footprint].sum()) >= (1.0 - missed) * float(np.asarray(weights).sum()) - 1e-15


def _checkpoint(seconds: float, incumbent: SearchLabel | None, metrics: SearchMetrics) -> dict[str, Any]:
    return {
        "seconds": float(seconds),
        "found": incumbent is not None,
        "used_reconfigurations": None if incumbent is None else incumbent.used_on_segments - 1,
        "joint_cost": None if incumbent is None else incumbent.joint_cost,
        "expanded": metrics.expanded,
        "generated": metrics.generated,
    }
