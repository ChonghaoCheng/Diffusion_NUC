from __future__ import annotations

import numpy as np
import heapq

from diffusion_coverage.coverage.episode_summary import summarize_ordered_membership
from diffusion_coverage.solvers.completion_bound import CompletionEdge
from diffusion_coverage.solvers.history_search import (
    SearchGraph,
    require_real_graph_readiness,
    search_history_graph,
)


def _off_reconfiguration_graph() -> SearchGraph:
    weights = np.ones(2, dtype=np.float64)
    membership = np.asarray(
        [
            [True, False, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    summary = summarize_ordered_membership(
        membership,
        weights,
        active=np.asarray([True, False, True]),
    )
    edge = CompletionEdge(0, 0, 1, summary, 1.0)
    return SearchGraph(
        node_membership=(membership[:, 0], membership[:, -1]),
        edges=(edge,),
        weights=weights,
        graph_hash="off-reconfiguration-test",
    )


def _search(graph: SearchGraph, *, maximum_on_segments: int, use_bound: bool):
    return search_history_graph(
        graph,
        start_node=0,
        maximum_on_segments=maximum_on_segments,
        missed_tolerance=0.0,
        repeat_tolerance=0.0,
        use_completion_bound=use_bound,
        wall_time_s=1.0,
        expanded_limit=100,
        checkpoint_times=(),
    )


def test_both_arms_apply_segment_budget_to_ordinary_reachability():
    graph = _off_reconfiguration_graph()
    for use_bound in (False, True):
        result = _search(graph, maximum_on_segments=1, use_bound=use_bound)
        assert result.incumbent is None
        assert result.metrics.segment_budget_reachability_pruned == 1
        assert result.metrics.reachability_pruned == 0
        assert result.metrics.completion_bound_pruned == 0
        assert result.metrics.completion_bound_calls == 0


def test_segment_budget_reachability_allows_same_edge_with_second_on_segment():
    graph = _off_reconfiguration_graph()
    for use_bound in (False, True):
        result = _search(graph, maximum_on_segments=2, use_bound=use_bound)
        assert result.incumbent is not None
        assert result.incumbent.used_on_segments == 2
        assert result.metrics.segment_budget_reachability_pruned == 0


def test_repeat_bound_obstruction_is_separate_from_shared_reachability():
    weights = np.ones(2)
    membership = np.asarray([[True, False, True], [False, False, True]])
    edge = CompletionEdge(0, 0, 1, summarize_ordered_membership(membership, weights), 1.0)
    graph = SearchGraph(
        node_membership=(membership[:, 0], membership[:, -1]),
        edges=(edge,),
        weights=weights,
        graph_hash="positive-repeat-obstruction",
    )
    s0 = _search(graph, maximum_on_segments=1, use_bound=False)
    s1 = _search(graph, maximum_on_segments=1, use_bound=True)
    assert s0.metrics.repeat_pruned == 1
    assert s0.metrics.segment_budget_reachability_pruned == 0
    assert s1.metrics.completion_bound_pruned == 1
    assert s1.metrics.segment_budget_reachability_pruned == 0
    assert s1.metrics.completion_bound_positive == 1


def _continuous_edge(edge_id: int, start: int, end: int, start_cell: int, end_cell: int, weights: np.ndarray):
    columns = np.zeros((3, len(weights)), dtype=bool)
    columns[0, start_cell] = True
    columns[1, [start_cell, end_cell]] = True
    columns[2, end_cell] = True
    return CompletionEdge(edge_id, start, end, summarize_ordered_membership(columns.T, weights), 1.0)


def _full_count_oracle(graph: SearchGraph, repeat_tolerance: float):
    start_counts = graph.node_membership[0].astype(np.int64)
    queue = [(0.0, 0, start_counts, 1)]
    best = {}
    outgoing = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.start, []).append(edge)
    total = graph.weights.sum()
    while queue:
        cost, node, counts, used = heapq.heappop(queue)
        key = (node, tuple(counts), used)
        if best.get(key, np.inf) <= cost:
            continue
        best[key] = cost
        repeat = float(np.dot(graph.weights, np.maximum(counts - 1, 0)) / total)
        if np.all(counts > 0) and repeat <= repeat_tolerance:
            return used, cost
        for edge in outgoing.get(node, ()):
            new_used = used + edge.summary.off_to_on_count
            new_counts = counts + edge.summary.episode_counts - edge.summary.start_membership.astype(np.int64)
            new_repeat = float(np.dot(graph.weights, np.maximum(new_counts - 1, 0)) / total)
            if new_used <= 1 and new_repeat <= repeat_tolerance:
                heapq.heappush(queue, (cost + edge.joint_cost, edge.end, new_counts, new_used))
    return None


def test_cyclic_graph_matches_bounded_independent_full_count_oracle():
    weights = np.ones(3)
    memberships = tuple(np.eye(3, dtype=bool))
    edges = (
        _continuous_edge(0, 0, 1, 0, 1, weights),
        _continuous_edge(1, 1, 0, 1, 0, weights),
        _continuous_edge(2, 1, 2, 1, 2, weights),
        _continuous_edge(3, 2, 1, 2, 1, weights),
    )
    graph = SearchGraph(memberships, edges, weights, "cyclic-full-count-oracle")
    oracle = _full_count_oracle(graph, repeat_tolerance=1.0)
    assert oracle == (1, 2.0)
    for use_bound in (False, True):
        result = search_history_graph(
            graph,
            start_node=0,
            maximum_on_segments=1,
            missed_tolerance=0.0,
            repeat_tolerance=1.0,
            use_completion_bound=use_bound,
            wall_time_s=1.0,
            expanded_limit=100,
            checkpoint_times=(),
        )
        assert result.optimality_proved
        assert result.incumbent is not None
        assert (result.incumbent.used_on_segments, result.incumbent.joint_cost) == oracle


def test_real_graph_readiness_guard_requires_on_connections_and_activity():
    try:
        require_real_graph_readiness(
            {
                "task_preserving_on_connections": False,
                "explicit_node_activity": True,
                "recomputed_endpoint_membership_checked": True,
            }
        )
    except RuntimeError as exc:
        assert "task_preserving_on_connections" in str(exc)
    else:
        raise AssertionError("missing ON connection capability was accepted")
    require_real_graph_readiness(
        {
            "task_preserving_on_connections": True,
            "explicit_node_activity": True,
            "recomputed_endpoint_membership_checked": True,
        }
    )
