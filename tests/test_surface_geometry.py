from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage import CoveragePlan, evaluate_coverage
from diffusion_coverage.surface import (
    geodesic_polyline_length,
    make_cylinder,
    make_freeform_patch,
    make_hemisphere,
    make_plane,
    make_saddle,
    make_torus,
    project_points,
    shortest_surface_polyline,
    surface_sample_distances,
    vertex_geodesic_distances,
)
from diffusion_coverage.surface.geodesic import resample_projected_polyline


def test_plane_area_samples_and_projection():
    surface = make_plane(width=2.0, height=3.0, nx=4, ny=6, samples_per_face=3)
    assert np.isclose(surface.total_area, 6.0)
    assert np.isclose(surface.area_weights.sum(), 6.0)
    assert np.allclose(surface.sample_normals, np.array([0.0, 0.0, 1.0]))

    query = np.array([[0.25, -0.5, 2.0], [3.0, 0.0, -1.0]])
    projection = project_points(surface, query)
    assert np.allclose(projection.points[0], np.array([0.25, -0.5, 0.0]))
    assert np.isclose(projection.distances[0], 2.0)
    assert np.allclose(projection.points[1], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(projection.barycentric.sum(axis=1), 1.0)


def test_vectorized_projection_matches_dense_planar_queries():
    surface = make_plane(width=2.0, height=3.0, nx=8, ny=7, samples_per_face=1)
    rng = np.random.default_rng(14)
    query = rng.uniform([-2.0, -2.5, -1.0], [2.0, 2.5, 1.0], size=(100, 3))
    projection = project_points(surface, query)
    assert np.allclose(projection.points[:, 2], 0.0)
    assert np.all(projection.points[:, 0] >= -1.0 - 1e-12)
    assert np.all(projection.points[:, 0] <= 1.0 + 1e-12)
    assert np.all(projection.points[:, 1] >= -1.5 - 1e-12)
    assert np.all(projection.points[:, 1] <= 1.5 + 1e-12)
    assert np.all(projection.barycentric >= -1e-12)
    assert np.allclose(projection.barycentric.sum(axis=1), 1.0)


def test_projection_chunking_and_allowed_faces_do_not_change_results():
    surface = make_cylinder(n_azimuth=14, n_height=4, samples_per_face=1)
    query = np.random.default_rng(25).normal(size=(73, 3))
    allowed = np.arange(0, surface.num_faces, 2)
    scalar_chunks = project_points(surface, query, allowed_faces=allowed, chunk_size=1)
    vector_chunks = project_points(surface, query, allowed_faces=allowed, chunk_size=64)
    assert np.allclose(scalar_chunks.points, vector_chunks.points)
    assert np.array_equal(scalar_chunks.face_indices, vector_chunks.face_indices)
    assert np.allclose(scalar_chunks.barycentric, vector_chunks.barycentric)
    assert np.allclose(scalar_chunks.distances, vector_chunks.distances)


def test_plane_geodesic_matches_diagonal():
    surface = make_plane(width=1.0, height=1.0, nx=1, ny=1, samples_per_face=1)
    length = geodesic_polyline_length(surface, np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]))
    assert np.isclose(length, np.sqrt(2.0))


def test_cylinder_graph_geodesic_converges_to_half_circumference():
    radius = 1.3
    n_azimuth = 96
    surface = make_cylinder(radius=radius, height=1.0, n_azimuth=n_azimuth, n_height=1, samples_per_face=1)
    distances = vertex_geodesic_distances(surface, [0])
    opposite_vertex = n_azimuth // 2
    assert np.isclose(distances[opposite_vertex], np.pi * radius, rtol=5e-4)


def test_cylinder_area_and_geodesic_errors_decrease_with_resolution():
    radius = 1.0
    height = 1.5
    exact_area = 2.0 * np.pi * radius * height
    exact_distance = np.pi * radius
    errors = []
    for n_azimuth in (12, 24, 48):
        surface = make_cylinder(
            radius=radius,
            height=height,
            n_azimuth=n_azimuth,
            n_height=1,
            samples_per_face=1,
        )
        distance = vertex_geodesic_distances(surface, [0])[n_azimuth // 2]
        errors.append(
            (
                abs(surface.total_area - exact_area) / exact_area,
                abs(distance - exact_distance) / exact_distance,
            )
        )
    assert all(errors[i + 1][0] < errors[i][0] for i in range(len(errors) - 1))
    assert all(errors[i + 1][1] < errors[i][1] for i in range(len(errors) - 1))


def test_cylinder_surface_polyline_does_not_cross_interior():
    radius = 1.0
    surface = make_cylinder(radius=radius, height=1.0, n_azimuth=32, n_height=2, samples_per_face=1)
    path = shortest_surface_polyline(surface, surface.vertices[0], surface.vertices[16])
    radial_distance = np.linalg.norm(path[:, :2], axis=1)
    assert np.allclose(radial_distance, radius)
    assert np.isclose(np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), np.pi * radius, rtol=5e-3)


def test_finite_footprint_coverage_on_plane():
    surface = make_plane(width=1.0, height=1.0, nx=12, ny=12, samples_per_face=4)
    plan = CoveragePlan(np.array([[-0.5, 0.0, 0.2], [0.5, 0.0, 0.2]]))
    metrics = evaluate_coverage(surface, plan, footprint_radius=0.8, path_sample_spacing=0.05)
    assert metrics.missed_fraction == 0.0
    assert np.isclose(metrics.path_length, 1.0)
    assert np.isclose(metrics.max_projection_distance, 0.2)
    assert metrics.metadata["num_path_sources"] >= 20


def test_coverage_checker_reuses_geodesic_without_changing_metrics():
    surface = make_cylinder(radius=1.0, height=1.0, n_azimuth=16, n_height=3, samples_per_face=2)
    path = np.asarray([surface.vertices[0], surface.vertices[5], surface.vertices[10]])
    spacing = 0.07
    metrics = evaluate_coverage(
        surface, CoveragePlan(path), footprint_radius=0.2, path_sample_spacing=spacing
    )
    old_length = geodesic_polyline_length(surface, path)
    old_sources = resample_projected_polyline(surface, path, max_spacing=spacing)
    old_distances = surface_sample_distances(surface, old_sources)
    old_missed = 1.0 - surface.area_weights[old_distances <= 0.2 + 1e-12].sum() / surface.total_area
    assert np.isclose(metrics.path_length, old_length)
    assert np.isclose(metrics.missed_fraction, old_missed)
    assert metrics.metadata["num_path_sources"] == len(old_sources)


def test_coverage_plan_masks_are_enforced():
    waypoints = np.zeros((2, 3, 3), dtype=np.float64)
    plan = CoveragePlan(
        waypoints,
        segment_mask=np.array([True, False]),
        waypoint_mask=np.array([[True, True, False], [False, False, False]]),
    )
    assert plan.num_segments == 1
    assert plan.active_paths()[0].shape == (2, 3)


def test_all_initial_surface_families_are_valid():
    surfaces = [
        make_cylinder(n_azimuth=12, n_height=3, samples_per_face=1),
        make_hemisphere(n_azimuth=12, n_polar=4, samples_per_face=1),
        make_saddle(nx=4, ny=4, samples_per_face=1),
        make_torus(n_major=12, n_minor=6, samples_per_face=1),
        make_freeform_patch(nx=4, ny=4, samples_per_face=1),
    ]
    for surface in surfaces:
        assert surface.num_vertices > 0
        assert surface.num_faces > 0
        assert surface.total_area > 0.0
        assert np.isclose(surface.area_weights.sum(), surface.total_area)
        assert np.all(np.isfinite(surface.face_normals))
