from __future__ import annotations

import numpy as np
import pytest

from diffusion_coverage.coverage.episode_summary import (
    EpisodeState,
    apply_edge_summary,
    initial_episode_state,
    summarize_ordered_membership,
)
from diffusion_coverage.solvers.completion_bound import CompletionEdge
from diffusion_coverage.solvers.history_search import SearchGraph, SearchLabel
from diffusion_coverage.solvers.structured_routing import (
    SourceRunCache,
    compose_episode_summaries,
    fixed_route_initialize,
    replay_edge_sequence,
    structured_anytime_search,
)


def _edge(eid, start, end, membership, weights, cost=1.0, active=None):
    summary = summarize_ordered_membership(np.asarray(membership, dtype=bool), weights, active=active)
    return CompletionEdge(eid, start, end, summary, cost)


def _data(edges, memberships, weights, *, routes=None, geom=None, kinds=None):
    graph = SearchGraph(tuple(np.asarray(x, dtype=bool) for x in memberships), tuple(edges), np.asarray(weights, dtype=float), "tiny")
    n = len(edges)
    geom = list(range(n)) if geom is None else geom
    kinds = ["source"] * n if kinds is None else kinds
    meta = [{"kind": kinds[i], "geom_arc_id": geom[i], "family": "f", "forward": True} for i in range(n)]
    return {"graph": graph, "edge_meta": meta, "routes": routes or {"f/forward": tuple(geom)}, "arc_start": np.arange(n), "arc_end": np.arange(1, n + 1), "node_ports": np.arange(len(memberships))}


def _config(**search_updates):
    search = {"run_lengths": [2, 4, 8, 16], "run_branch_cap": 8, "run_cache_mib": 1, "online_screen_limit": 4, "candidate_reservoir": 8, "bucket_live_limit": 32, "endpoint_diversity_limit": 2, "expanded_label_limit": 1000, "retained_record_limit": 30000, "private_memory_gib": 6, "checkpoints_s": []}
    search.update(search_updates)
    return {"coverage": {"missed_tolerance": 0.0, "repeat_tolerance": 1.0}, "search": search}


def test_composed_summary_matches_raw_concatenation_with_multiple_visits():
    weights = np.array([1.0, 2.0, 3.0])
    first = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]], bool)
    second = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1]], bool)
    a = summarize_ordered_membership(first, weights)
    b = summarize_ordered_membership(second, weights)
    composed = compose_episode_summaries((a, b), weights)
    raw = summarize_ordered_membership(np.concatenate((first, second[:, 1:]), axis=1), weights)
    np.testing.assert_array_equal(composed.episode_counts, raw.episode_counts)
    np.testing.assert_array_equal(composed.footprint, raw.footprint)
    assert composed.weighted_episode_mass == raw.weighted_episode_mass


def test_composed_summary_requires_exact_boundary_membership():
    w = np.ones(1)
    a = summarize_ordered_membership(np.array([[1, 1]], bool), w)
    b = summarize_ordered_membership(np.array([[0, 1]], bool), w)
    with pytest.raises(ValueError, match="endpoint membership"):
        compose_episode_summaries((a, b), w)


def test_run_update_equals_sequential_atomic_update():
    w = np.array([1.0, 3.0])
    m0 = np.array([[1, 1], [0, 0]], bool)
    m1 = np.array([[1, 0, 0], [0, 1, 1]], bool)
    a = summarize_ordered_membership(m0, w)
    b = summarize_ordered_membership(m1, w)
    state = initial_episode_state(m0[:, 0])
    sequential = apply_edge_summary(apply_edge_summary(state, a, w), b, w)
    block = apply_edge_summary(state, compose_episode_summaries((a, b), w), w)
    np.testing.assert_array_equal(block.covered, sequential.covered)
    assert block.repeat_error == sequential.repeat_error


def test_source_run_catalog_preserves_every_atomic_edge():
    w = np.ones(3)
    e0 = _edge(0, 0, 1, [[1, 1], [0, 1], [0, 0]], w)
    e1 = _edge(1, 1, 2, [[1, 0], [1, 1], [0, 1]], w)
    data = _data([e0, e1], [[1, 0, 0], [1, 1, 0], [0, 1, 1]], w)
    cache = SourceRunCache(data, (2, 4), 8, 1024 * 1024)
    from diffusion_coverage.solvers.structured_routing import AnytimeMetrics
    actions = cache.actions(0, AnytimeMetrics())
    assert any(a.edge_ids == (0, 1) for a in actions)
    assert [e.edge_id for e in data["graph"].edges] == [0, 1]


def test_replay_is_exact_and_rejects_discontinuous_sequences():
    w = np.ones(2)
    e0 = _edge(0, 0, 1, [[1, 1], [0, 1]], w)
    e1 = _edge(1, 1, 2, [[1, 0], [1, 1]], w)
    data = _data([e0, e1], [[1, 0], [1, 1], [0, 1]], w)
    replay = replay_edge_sequence(data["graph"], 0, (0, 1))
    assert replay.node == 2 and replay.path == (0, 1) and replay.joint_cost == 2.0
    with pytest.raises(ValueError, match="discontinuity"):
        replay_edge_sequence(data["graph"], 0, (1,))


def test_failed_initializer_does_not_block_root_global_search():
    w = np.ones(2)
    e = _edge(0, 0, 1, [[1, 1], [0, 1]], w)
    data = _data([e], [[1, 0], [1, 1]], w)
    result = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1)
    assert result.candidates and result.candidates[0].edge_ids == (0,)


def test_validated_fallback_is_retained_when_budget_is_zero():
    w = np.ones(1)
    graph = SearchGraph((np.array([True]),), (), w, "empty")
    data = {"graph": graph, "edge_meta": [], "routes": {}, "arc_start": np.array([], int), "arc_end": np.array([], int), "node_ports": np.array([0])}
    fallback = SearchLabel(0, np.array([True]), np.array([True]), 0.0, 2.0, 1, ())
    result = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=fallback, use_source_runs=False, config=_config(), wall_time_s=0)
    assert result.fallback_retained and result.termination == "wall_time"


def test_q3_failed_goal_does_not_stop_alternative_search():
    w = np.ones(2)
    a = _edge(0, 0, 1, [[1, 1], [0, 1]], w, 1.0)
    b = _edge(1, 0, 2, [[1, 1], [0, 1]], w, 2.0)
    data = _data([a, b], [[1, 0], [1, 1], [1, 1]], w, routes={"x": (0,)}, geom=[0, 1])
    calls = []
    def screen(path):
        calls.append(path)
        return {"status": "Q3_FAIL" if path == (0,) else "Q3_PASS", "E_miss": 0.0, "E_rep": 0.0}
    result = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1, screen=screen)
    assert (0,) in calls and (1,) in calls
    assert result.candidates[0].edge_ids == (1,)


def test_repeat_and_cost_are_both_required_for_dominance():
    w = np.ones(2)
    # Both paths end at the same exact state/history; one has lower cost, the
    # other lower repeat.  Neither may delete the other.
    s = summarize_ordered_membership(np.array([[1, 0, 1], [0, 1, 1]], bool), w)
    t = summarize_ordered_membership(np.array([[1, 1], [0, 1]], bool), w)
    e0 = CompletionEdge(0, 0, 1, s, 1.0)
    e1 = CompletionEdge(1, 0, 1, t, 2.0)
    data = _data([e0, e1], [[1, 0], [1, 1]], w, geom=[0, 1])
    result = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1)
    assert result.metrics.dominance_pruned == 0


def test_live_frontier_limit_is_not_cumulative_admission_limit():
    w = np.ones(4)
    edges = []
    memberships = [np.array([1, 0, 0, 0], bool)]
    for i in range(3):
        m0 = memberships[-1]
        m1 = m0.copy(); m1[i + 1] = True
        edges.append(_edge(i, i, i + 1, np.column_stack((m0, m1)), w))
        memberships.append(m1)
    data = _data(edges, memberships, w)
    cfg = _config(bucket_live_limit=1, retained_record_limit=100)
    result = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=cfg, wall_time_s=1)
    assert result.metrics.peak_open <= 1
    assert result.termination == "beam_exhausted"


def test_two_segment_bucket_is_serviced_with_one_segment_backlog():
    w = np.ones(2)
    on = _edge(0, 0, 1, [[1, 1], [0, 0]], w)
    off_membership = np.array([[1, 0, 0], [0, 0, 1]], bool)
    off = _edge(1, 0, 2, off_membership, w, active=np.array([1, 0, 1], bool))
    data = _data([on, off], [[1, 0], [1, 0], [0, 1]], w, geom=[0, 1], kinds=["source", "off_reconfiguration"])
    result = structured_anytime_search(data, start_node=0, maximum_on_segments=2, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1)
    assert result.candidates and result.candidates[0].used_on_segments == 2


def test_source_runs_and_atomic_search_agree_without_truncation():
    w = np.ones(3)
    e0 = _edge(0, 0, 1, [[1, 1], [0, 1], [0, 0]], w)
    e1 = _edge(1, 1, 2, [[1, 0], [1, 1], [0, 1]], w)
    data = _data([e0, e1], [[1, 0, 0], [1, 1, 0], [0, 1, 1]], w)
    kwargs=dict(start_node=0,maximum_on_segments=1,initial_prefixes=(),validated_fallback=None,config=_config(),wall_time_s=1)
    atomic=structured_anytime_search(data,use_source_runs=False,**kwargs)
    grouped=structured_anytime_search(data,use_source_runs=True,**kwargs)
    assert atomic.candidates[0].joint_cost == grouped.candidates[0].joint_cost == 2.0


def test_fixed_archive_replays_exact_prefix_states():
    w = np.ones(2)
    e = _edge(0, 0, 1, [[1, 1], [0, 1]], w)
    data = _data([e], [[1, 0], [1, 1]], w)
    result, archive = fixed_route_initialize(data, 0, 1, _config(), wall_time=1, expanded_limit=10)
    assert result.incumbent is not None
    for item in archive:
        replay_edge_sequence(data["graph"], 0, item["path"])
