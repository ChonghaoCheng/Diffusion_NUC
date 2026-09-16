from __future__ import annotations

import itertools

import numpy as np

from diffusion_coverage.coverage.episode_summary import (
    apply_edge_summary,
    initial_episode_state,
    summarize_ordered_membership,
)
from diffusion_coverage.coverage.evaluator import _length_and_sources
from diffusion_coverage.coverage.nuc_evaluator import _ordered_footprint_membership
from diffusion_coverage.coverage.ordered_trace_evaluator import (
    evaluate_ordered_trace_euclidean,
    resample_prescribed_path,
    saddle_pairwise_distance_bounds,
    sphere_pairwise_distances,
    uncertain_episode_bounds,
)
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def _two_triangle_square(flipped: bool = False) -> SurfaceInstance:
    vertices = 1e-3 * np.asarray([[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0]], float)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]] if not flipped else [[0, 1, 3], [1, 2, 3]])
    return SurfaceInstance.from_mesh(vertices, faces, samples_per_face=1, surface_id="plane", metadata={"width": 0.02, "height": 0.02})


def test_two_triangle_regression_preserves_straight_motion_and_exposes_legacy_detour():
    surface = _two_triangle_square()
    endpoints = 1e-3 * np.asarray([[9, 11, 0], [11, 9, 0]], float)
    center = 1e-3 * np.asarray([[10, 10, 0]], float)
    metrics = evaluate_ordered_trace_euclidean(center, np.ones(1), (endpoints,), footprint_radius=0.008)
    assert np.isclose(metrics.path_length, np.sqrt(8.0) * 1e-3, atol=1e-15)
    assert metrics.episode_min[0] == 1
    projection = project_points(surface, endpoints)
    legacy_length, legacy_sources = _length_and_sources(surface, projection, max_spacing=0.001)
    legacy_membership = _ordered_footprint_membership(surface, legacy_sources, footprint_radius=0.008)
    assert legacy_length > 0.02
    assert np.count_nonzero(legacy_membership[0, 1:] & ~legacy_membership[0, :-1]) >= 1


def test_planar_reference_is_invariant_to_diagonal_and_vertex_relabeling():
    endpoints = 1e-3 * np.asarray([[9, 11, 0], [11, 9, 0]], float)
    center = 1e-3 * np.asarray([[10, 10, 0]], float)
    baseline = evaluate_ordered_trace_euclidean(center, np.ones(1), (endpoints,), footprint_radius=0.008)
    for surface in (_two_triangle_square(), _two_triangle_square(flipped=True)):
        projected = project_points(surface, endpoints).points
        result = evaluate_ordered_trace_euclidean(center, np.ones(1), (projected,), footprint_radius=0.008)
        assert result.path_length == baseline.path_length
        assert np.array_equal(result.episode_min, baseline.episode_min)


def test_stationary_leave_return_and_separate_on_segments():
    sample = np.zeros((1, 3))
    weights = np.ones(1)
    stationary = np.zeros((3, 3))
    away = np.asarray([[0, 0, 0], [0.02, 0, 0], [0, 0, 0]], float)
    assert evaluate_ordered_trace_euclidean(sample, weights, (stationary,), footprint_radius=0.008).episode_min[0] == 1
    assert evaluate_ordered_trace_euclidean(sample, weights, (away,), footprint_radius=0.008).episode_min[0] == 2
    assert evaluate_ordered_trace_euclidean(sample, weights, (stationary[:1], stationary[:1]), footprint_radius=0.008).episode_min[0] == 2


def test_shared_active_endpoint_summary_does_not_duplicate_episode():
    weights = np.ones(1)
    first = summarize_ordered_membership(np.asarray([[True, True]]), weights)
    second = summarize_ordered_membership(np.asarray([[True, True]]), weights)
    state = initial_episode_state(np.asarray([True]))
    state = apply_edge_summary(state, first, weights)
    state = apply_edge_summary(state, second, weights)
    assert state.repeat_error == 0.0


def test_sphere_distance_handles_seam_and_pole():
    radius = 0.14
    eps = 1e-8
    seam = radius * np.asarray([[np.cos(eps), np.sin(eps), 0], [np.cos(eps), -np.sin(eps), 0]])
    distance = sphere_pairwise_distances(seam[:1], seam[1:], radius=radius)[0, 0]
    assert np.isclose(distance, 2.0 * eps * radius, rtol=1e-7)
    pole = np.asarray([[0.0, 0.0, radius]])
    near = radius * np.asarray([[np.sin(eps), 0.0, np.cos(eps)]])
    assert np.isclose(sphere_pairwise_distances(pole, near, radius=radius)[0, 0], eps * radius, rtol=1e-7)


def test_saddle_bounds_enclose_dense_chart_segment_length():
    curvature = 1.5
    starts = np.asarray([[-0.08, -0.03, curvature * (0.08**2 - 0.03**2)]])
    ends = np.asarray([[0.09, 0.07, curvature * (0.09**2 - 0.07**2)]])
    lower, upper = saddle_pairwise_distance_bounds(starts, ends, curvature=curvature)
    t = np.linspace(0.0, 1.0, 200001)
    x = starts[0, 0] + t * (ends[0, 0] - starts[0, 0])
    y = starts[0, 1] + t * (ends[0, 1] - starts[0, 1])
    points = np.column_stack((x, y, curvature * (x * x - y * y)))
    numerical = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    assert lower[0, 0] <= numerical <= upper[0, 0]
    assert upper[0, 0] - numerical < 1e-10


def test_uncertain_episode_dp_matches_bruteforce_and_gap_bridge():
    for length in range(1, 7):
        for states in itertools.product((0, 1, 2), repeat=length):
            zero = np.asarray([[value != 1 for value in states]])
            one = np.asarray([[value != 0 for value in states]])
            lo, hi = uncertain_episode_bounds(zero, one)
            values = []
            for sequence in itertools.product((0, 1), repeat=length):
                if all((bit == 0 and zero[0, i]) or (bit == 1 and one[0, i]) for i, bit in enumerate(sequence)):
                    values.append(sequence[0] + sum((not sequence[i - 1]) and sequence[i] for i in range(1, length)))
            assert lo[0] == min(values)
            assert hi[0] == max(values)
    zero = np.asarray([[False, True, False]])
    one = np.asarray([[True, True, True]])
    lo, hi = uncertain_episode_bounds(zero, one)
    assert (lo[0], hi[0]) == (1, 2)


def test_resolutions_start_from_same_original_intervals():
    metadata = {"width": 0.24, "height": 0.24, "curvature": 1.5}
    x = np.asarray([-0.1, 0.02, 0.1])
    y = np.asarray([-0.04, 0.05, 0.08])
    original = np.column_stack((x, y, 1.5 * (x * x - y * y)))
    coarse = resample_prescribed_path(original, surface_id="saddle", surface_metadata=metadata, maximum_step=0.01)
    fine = resample_prescribed_path(original, surface_id="saddle", surface_metadata=metadata, maximum_step=0.005)
    for waypoint in original:
        assert np.min(np.linalg.norm(coarse - waypoint, axis=1)) < 1e-14
        assert np.min(np.linalg.norm(fine - waypoint, axis=1)) < 1e-14
