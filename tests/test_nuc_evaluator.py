from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage import CoveragePlan, evaluate_nuc_coverage
from diffusion_coverage.surface import SurfaceInstance, make_plane


def one_sample_surface() -> SurfaceInstance:
    vertices = np.asarray([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]])
    return SurfaceInstance.from_mesh(vertices, np.asarray([[0, 1, 2]]), samples_per_face=1)


def plan_from_paths(paths: list[np.ndarray]) -> CoveragePlan:
    width = max(len(path) for path in paths)
    waypoints = np.zeros((len(paths), width, 3), dtype=np.float64)
    mask = np.zeros((len(paths), width), dtype=bool)
    for index, path in enumerate(paths):
        waypoints[index, : len(path)] = path
        mask[index, : len(path)] = True
    return CoveragePlan(waypoints, waypoint_mask=mask)


def test_one_continuous_pass_is_one_visit():
    surface = one_sample_surface()
    point = surface.sample_points[0]
    path = np.stack((point + [-0.5, 0.0, 0.0], point + [0.5, 0.0, 0.0]))
    metrics = evaluate_nuc_coverage(surface, CoveragePlan(path), footprint_radius=0.15, path_sample_spacing=0.03)
    assert metrics.visit_counts[0] == 1


def test_repeated_identical_samples_in_one_segment_are_one_visit():
    surface = one_sample_surface()
    point = surface.sample_points[0]
    path = np.repeat(point[None, :], 4, axis=0)
    metrics = evaluate_nuc_coverage(surface, CoveragePlan(path), footprint_radius=0.1)
    assert metrics.visit_counts[0] == 1


def test_leave_and_return_in_one_segment_are_two_visits():
    surface = one_sample_surface()
    point = surface.sample_points[0]
    path = np.stack((point, point + [0.8, 0.0, 0.0], point))
    metrics = evaluate_nuc_coverage(surface, CoveragePlan(path), footprint_radius=0.1, path_sample_spacing=0.025)
    assert metrics.visit_counts[0] == 2
    assert metrics.repeat_error == 1.0


def test_same_region_in_two_active_segments_is_two_visits():
    surface = one_sample_surface()
    point = surface.sample_points[0]
    paths = [np.stack((point, point + [0.05, 0.0, 0.0]))] * 2
    metrics = evaluate_nuc_coverage(surface, plan_from_paths(paths), footprint_radius=0.1)
    assert metrics.visit_counts[0] == 2


def test_never_visited_sample_has_zero_visits():
    surface = one_sample_surface()
    point = surface.sample_points[0]
    path = np.stack((point + [0.8, 0.0, 0.0], point + [1.0, 0.0, 0.0]))
    metrics = evaluate_nuc_coverage(surface, CoveragePlan(path), footprint_radius=0.05)
    assert metrics.visit_counts[0] == 0
    assert metrics.missed_error == 1.0


def test_area_weighting_uses_surface_quadrature_weights():
    vertices = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0],
        [10.0, 0.0, 0.0], [12.0, 0.0, 0.0], [10.0, 2.0, 0.0],
    ])
    surface = SurfaceInstance.from_mesh(vertices, np.asarray([[0, 1, 2], [3, 4, 5]]), samples_per_face=1)
    point = surface.sample_points[0]
    metrics = evaluate_nuc_coverage(
        surface, CoveragePlan(np.stack((point, point + [0.01, 0.0, 0.0]))), footprint_radius=0.05
    )
    assert np.isclose(metrics.missed_error, 2.0 / 3.0)
    assert np.isclose(metrics.single_coverage_fraction, 1.0 / 3.0)


def test_input_path_densification_does_not_change_visit_count():
    surface = one_sample_surface()
    point = surface.sample_points[0]
    sparse = np.stack((point, point + [0.8, 0.0, 0.0], point))
    dense = np.concatenate((
        np.linspace(sparse[0], sparse[1], 20, endpoint=False),
        np.linspace(sparse[1], sparse[2], 21),
    ))
    sparse_metrics = evaluate_nuc_coverage(surface, CoveragePlan(sparse), footprint_radius=0.1, path_sample_spacing=0.02)
    dense_metrics = evaluate_nuc_coverage(surface, CoveragePlan(dense), footprint_radius=0.1, path_sample_spacing=0.02)
    assert np.array_equal(sparse_metrics.visit_counts, dense_metrics.visit_counts)


def test_nuc_metrics_converge_with_denser_path_and_surface_sampling():
    path = np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    coarse = make_plane(width=1.0, height=1.0, nx=12, ny=12, samples_per_face=1)
    dense = make_plane(width=1.0, height=1.0, nx=24, ny=24, samples_per_face=2)
    coarse_metrics = evaluate_nuc_coverage(coarse, CoveragePlan(path), footprint_radius=0.2, path_sample_spacing=0.05)
    dense_metrics = evaluate_nuc_coverage(dense, CoveragePlan(path), footprint_radius=0.2, path_sample_spacing=0.025)
    assert abs(coarse_metrics.nuc_error - dense_metrics.nuc_error) < 0.08
    assert abs(coarse_metrics.legacy_missed_fraction - dense_metrics.legacy_missed_fraction) < 0.08
