from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from diffusion_coverage.coverage.episode_summary import summarize_ordered_membership
from diffusion_coverage.solvers.completion_bound import CompletionEdge
from diffusion_coverage.solvers.history_search import SearchGraph, SearchLabel
from diffusion_coverage.solvers.structured_routing import (
    greedy_prefix_completion,
    replay_edge_sequence,
    structured_anytime_search,
)


def _edge(eid, start, end, membership, weights, cost=1.0, active=None):
    return CompletionEdge(eid, start, end, summarize_ordered_membership(np.asarray(membership, bool), weights, active=active), cost)


def _data(edges, memberships, weights):
    graph = SearchGraph(tuple(np.asarray(x, bool) for x in memberships), tuple(edges), np.asarray(weights, float), "e11-tiny")
    return {"graph": graph, "edge_meta": [{"kind": "cross_port", "geom_arc_id": i} for i in range(len(edges))], "routes": {}, "arc_start": np.array([], int), "arc_end": np.array([], int), "node_ports": np.arange(len(memberships))}


def _config(**updates):
    search = {"online_screen_limit": 4, "candidate_reservoir": 8, "bucket_live_limit": 32, "endpoint_diversity_limit": 2, "expanded_label_limit": 1000, "retained_record_limit": 30000, "private_memory_gib": 6, "checkpoints_s": [], "run_lengths": [2, 4, 8, 16], "run_branch_cap": 8, "run_cache_mib": 1}
    search.update(updates)
    return {"coverage": {"missed_tolerance": 0.0, "repeat_tolerance": 1.0}, "search": search}


def test_event_logging_preserves_untimed_atomic_choice():
    w = np.ones(2); e = _edge(0, 0, 1, [[1, 1], [0, 1]], w)
    data = _data([e], [[1, 0], [1, 1]], w)
    base = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1)
    logged = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1, screen=lambda _: {"status": "Q3_PASS", "E_miss": 0.0, "E_rep": 0.0})
    assert base.candidates[0].edge_ids == logged.candidates[0].edge_ids == (0,)
    assert len(logged.screen_events) == logged.metrics.screen_calls == 1


def test_screen_event_survives_reservoir_discard():
    w = np.ones(2); edges = [_edge(i, 0, i + 1, [[1, 1], [0, 1]], w, i + 1) for i in range(10)]
    data = _data(edges, [np.array([1, 0], bool)] + [np.array([1, 1], bool)] * 10, w)
    cfg = _config(online_screen_limit=10)
    result = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=cfg, wall_time_s=1, screen=lambda _: {"status": "Q3_PASS", "E_miss": 0.0, "E_rep": 0.0})
    assert len(result.screen_events) == result.metrics.screen_calls == 10
    assert any(not x["reservoir_retained"] for x in result.screen_events)


def test_greedy_uses_all_atomic_edge_kinds_and_local_rank():
    w = np.ones(3)
    source = _edge(0, 0, 1, [[1, 1], [0, 1], [0, 0]], w, 0.1)
    cross = _edge(1, 0, 2, [[1, 1], [0, 1], [0, 1]], w, 3.0)
    data = _data([source, cross], [[1, 0, 0], [1, 1, 0], [1, 1, 1]], w)
    result = greedy_prefix_completion(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, config=_config(), wall_time_s=1)
    assert result.candidates and result.candidates[0].edge_ids == (1,)
    assert result.metrics.generated_actions == 2


def test_greedy_keeps_one_continuation_per_rollout():
    w = np.ones(2); edges = [_edge(i, 0, i + 1, [[1, 1], [0, 1]], w, i + 1) for i in range(3)]
    data = _data(edges, [[1, 0]] + [[1, 1]] * 3, w)
    result = greedy_prefix_completion(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, config=_config(), wall_time_s=1)
    assert result.run_catalog["one_continuation_per_rollout"] is True
    assert result.run_catalog["rollouts"] == 1


def test_greedy_loop_terminates_by_exact_state_dominance():
    w = np.ones(1)
    a = _edge(0, 0, 1, [[1, 1]], w)
    b = _edge(1, 1, 0, [[1, 1]], w)
    data = _data([a, b], [[1], [1]], w)
    result = greedy_prefix_completion(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, config=_config(), wall_time_s=1)
    assert result.termination == "greedy_exhausted"
    assert result.metrics.dominance_pruned >= 1


def test_prefix_replay_and_root_ablation_only_change_seed_set():
    w = np.ones(3)
    e0 = _edge(0, 0, 1, [[1, 1], [0, 1], [0, 0]], w)
    e1 = _edge(1, 1, 2, [[1, 1], [1, 1], [0, 1]], w)
    data = _data([e0, e1], [[1, 0, 0], [1, 1, 0], [1, 1, 1]], w)
    seeded = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=((0,),), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1)
    root = structured_anytime_search(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=None, use_source_runs=False, config=_config(), wall_time_s=1)
    assert replay_edge_sequence(data["graph"], 0, (0,)).path == (0,)
    assert seeded.candidates[0].edge_ids == root.candidates[0].edge_ids == (0, 1)


def test_valid_fallback_survives_greedy_timeout():
    w = np.ones(1); data = _data([], [[1]], w)
    fallback = SearchLabel(0, np.array([True]), np.array([True]), 0.0, 1.0, 1, ())
    result = greedy_prefix_completion(data, start_node=0, maximum_on_segments=1, initial_prefixes=(), validated_fallback=fallback, config=_config(), wall_time_s=0)
    assert result.fallback_retained and result.termination == "wall_time"


def test_registered_transfer_transform_hashes_and_formula():
    root = Path(__file__).resolve().parents[1]
    doc = json.loads((root / "configs/e11_transfer_scenes_v1.json").read_text())
    assert [x["scene_id"] for x in doc["scenes"]] == [f"H{i:02d}" for i in range(6)]
    for row in doc["scenes"]:
        matrix = np.asarray(row["transform_base_from_surface"], np.float64)
        assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-14)
        assert np.linalg.det(matrix[:3, :3]) > 0.999999999999
        assert hashlib.sha256(matrix.tobytes()).hexdigest() == row["transform_sha256"]


def test_e11_scene_adapter_uses_candidate_id_without_legacy_placements_key():
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from e11_runner_support import _scene
    config = json.loads((root / "configs/e11_mechanism_placement_transfer_v1.json").read_text())
    assert "placements" not in config
    assert _scene(root, config, "T30")["candidate_id"] == "T30"
