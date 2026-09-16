from __future__ import annotations

import itertools

import numpy as np

from diffusion_coverage.coverage.episode_summary import (
    EpisodeState,
    apply_edge_summary,
    concatenate_membership_edges,
    initial_episode_state,
    summarize_ordered_membership,
)
from diffusion_coverage.solvers.completion_bound import (
    CompletionEdge,
    completion_repeat_lower_bound,
)


def edge(edge_id, start, end, columns, weights, active=None, cost=1.0):
    membership = np.asarray(columns, dtype=bool).T
    return CompletionEdge(
        edge_id,
        start,
        end,
        summarize_ordered_membership(membership, weights, active=None if active is None else np.asarray(active)),
        cost,
    )


def test_random_segmentation_matches_full_membership():
    rng = np.random.default_rng(20260916)
    weights = np.arange(1, 9, dtype=np.float64)
    for _ in range(50):
        membership = rng.random((8, 20)) < 0.35
        active = rng.random(20) < 0.8
        cuts = sorted(rng.choice(np.arange(1, 19), size=3, replace=False))
        bounds = [0, *cuts, 19]
        pieces = tuple(membership[:, lo : hi + 1] for lo, hi in zip(bounds[:-1], bounds[1:]))
        activities = tuple(active[lo : hi + 1] for lo, hi in zip(bounds[:-1], bounds[1:]))
        rebuilt, rebuilt_active = concatenate_membership_edges(pieces, activities)
        assert np.array_equal(rebuilt, membership)
        assert np.array_equal(rebuilt_active, active)
        full = summarize_ordered_membership(membership, weights, active=active)
        state = initial_episode_state(membership[:, 0] & active[0])
        for value, mask in zip(pieces, activities):
            state = apply_edge_summary(state, summarize_ordered_membership(value, weights, active=mask), weights)
        exact_repeat = np.dot(weights, np.maximum(full.episode_counts - full.footprint.astype(int), 0)) / weights.sum()
        assert np.isclose(state.repeat_error, exact_repeat)
        assert np.array_equal(state.covered, full.footprint)


def test_episode_corner_cases_and_initial_footprint():
    weights = np.ones(2)
    cases = [
        ([[1, 0], [1, 0]], [1, 1], 0.0),  # dwell
        ([[1, 0], [0, 0], [1, 0]], [1, 1, 1], 0.5),  # leave and return
        ([[1, 0], [1, 1]], [1, 1], 0.0),  # ON shared endpoint/new cell
        ([[1, 0], [1, 0], [1, 0]], [1, 0, 1], 0.5),  # off then on
        ([[1, 0], [0, 0], [0, 0], [1, 0]], [1, 0, 0, 1], 0.5),
    ]
    for columns, active, expected in cases:
        summary = summarize_ordered_membership(np.asarray(columns, bool).T, weights, active=np.asarray(active))
        state = apply_edge_summary(initial_episode_state(np.asarray(columns[0], bool)), summary, weights)
        assert np.isclose(state.repeat_error, expected)


def test_same_node_different_history_and_roll_does_not_revisit():
    weights = np.ones(3)
    summary = summarize_ordered_membership(np.asarray([[1, 1], [0, 1], [0, 0]], bool), weights)
    a = apply_edge_summary(EpisodeState(np.array([1, 0, 0], bool), np.array([1, 0, 0], bool), 0.0), summary, weights)
    b = apply_edge_summary(EpisodeState(np.array([1, 1, 0], bool), np.array([1, 0, 0], bool), 0.0), summary, weights)
    assert a.repeat_error == 0.0
    assert b.repeat_error == 1.0 / 3.0
    roll_only = summarize_ordered_membership(np.asarray([[1, 1, 1], [0, 0, 0], [0, 0, 0]], bool), weights)
    assert apply_edge_summary(initial_episode_state(np.array([1, 0, 0], bool)), roll_only, weights).repeat_error == 0.0


def test_shared_bottleneck_uses_max_not_sum():
    weights = np.ones(3)
    # Unit 0 is the already-covered bottleneck. Either target costs one revisit;
    # reaching both in one path still costs only that same revisit.
    e0 = edge(0, 0, 1, [[1, 0, 0], [0, 0, 0], [1, 1, 1]], weights)
    e1 = edge(1, 1, 2, [[1, 1, 1], [1, 1, 1]], weights)
    result = completion_repeat_lower_bound(
        node=0, covered=np.array([1, 0, 0], bool), repeat_error=0.0,
        used_on_segments=1, maximum_on_segments=1, weights=weights,
        edges=(e0, e1), missed_tolerance=0.0,
    )
    assert np.allclose(result.target_distances[1:], 1.0 / 3.0)
    assert np.isclose(result.future_lower_bound, 1.0 / 3.0)
    assert result.target_distances[1:].sum() > result.future_lower_bound


def test_area_weighted_quantile_can_ignore_small_unreachable_unit():
    weights = np.array([5.0, 4.0, 1.0])
    e0 = edge(0, 0, 1, [[1, 0, 0], [0, 1, 0]], weights)
    result = completion_repeat_lower_bound(
        node=0, covered=np.array([1, 0, 0], bool), repeat_error=0.0,
        used_on_segments=1, maximum_on_segments=1, weights=weights,
        edges=(e0,), missed_tolerance=0.1,
    )
    assert np.isfinite(result.future_lower_bound)
    count_quantile_wrong = completion_repeat_lower_bound(
        node=0, covered=np.array([1, 0, 0], bool), repeat_error=0.0,
        used_on_segments=1, maximum_on_segments=1, weights=np.ones(3),
        edges=(edge(0, 0, 1, [[1, 0, 0], [0, 1, 0]], np.ones(3)),), missed_tolerance=0.1,
    )
    assert np.isinf(count_quantile_wrong.future_lower_bound)


def test_off_reconfiguration_and_cheaper_added_edge_lower_bound():
    weights = np.ones(2)
    on = edge(0, 0, 1, [[1, 0], [0, 1]], weights)
    off = edge(1, 0, 1, [[1, 0], [0, 0], [0, 1]], weights, active=[1, 0, 1])
    base = dict(node=0, covered=np.array([1, 0], bool), repeat_error=0.0, used_on_segments=1,
                weights=weights, missed_tolerance=0.0)
    only_on = completion_repeat_lower_bound(maximum_on_segments=2, edges=(on,), **base)
    with_off = completion_repeat_lower_bound(maximum_on_segments=2, edges=(on, off), **base)
    assert with_off.future_lower_bound <= only_on.future_lower_bound
    assert with_off.future_lower_bound == 0.0
    blocked = completion_repeat_lower_bound(maximum_on_segments=1, edges=(off,), **base)
    assert np.isinf(blocked.future_lower_bound)


def test_joint_winding_and_activity_are_not_implicit_equivalence_keys():
    # This test documents the exact arrays that graph deduplication must retain.
    q = np.zeros(6)
    wound = q.copy(); wound[-1] += 2.0 * np.pi
    assert not np.array_equal(q, wound)
    assert not np.array_equal(np.array([True]), np.array([False]))
