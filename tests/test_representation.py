from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage import generate_pattern_proposals
from diffusion_coverage.evaluation.representation_audit import (
    audit_fixed_control_bspline,
    symmetric_modified_hausdorff,
)
from diffusion_coverage.representation import (
    canonicalize_surface_path,
    pad_canonical_paths,
    suggested_token_count,
)
from diffusion_coverage.surface import make_cylinder, make_plane


def test_canonicalization_is_invariant_to_open_path_reversal():
    surface = make_plane(nx=8, ny=8, samples_per_face=1)
    path = np.column_stack((np.linspace(-1.0, 1.0, 31), 0.2 * np.sin(np.linspace(0, 4, 31)), np.zeros(31)))
    forward = canonicalize_surface_path(surface, path, num_tokens=64)
    backward = canonicalize_surface_path(surface, path[::-1], num_tokens=64)
    assert np.allclose(forward.points, backward.points, atol=1e-10)
    assert forward.reversed != backward.reversed


def test_token_count_increases_with_area_and_smaller_footprint():
    assert suggested_token_count(4.0, 0.1) > suggested_token_count(4.0, 0.2)
    assert suggested_token_count(8.0, 0.2) > suggested_token_count(4.0, 0.2)


def test_variable_token_batch_padding_has_explicit_mask():
    surface = make_plane(nx=6, ny=6, samples_per_face=1)
    line = np.column_stack((np.linspace(-1.0, 1.0, 20), np.zeros(20), np.zeros(20)))
    short = canonicalize_surface_path(surface, line, num_tokens=32)
    long = canonicalize_surface_path(surface, line, num_tokens=73)
    batch = pad_canonical_paths([short, long])
    assert batch.points.shape == (2, 73, 3)
    assert batch.mask.sum(axis=1).tolist() == [32, 73]
    assert not batch.mask[0, 32:].any()


def test_preserve_source_waypoints_uses_source_length_and_surface_projection():
    surface = make_plane(nx=6, ny=6, samples_per_face=1)
    path = np.array(
        [[-0.5, 0.0, 1e-8], [0.0, 0.25, -1e-8], [0.5, 0.0, 1e-8]],
        dtype=np.float64,
    )
    canonical = canonicalize_surface_path(
        surface,
        path,
        num_tokens=2,
        preserve_source_waypoints=True,
    )
    assert len(canonical.points) == len(path)
    np.testing.assert_allclose(canonical.points[:, :2], path[:, :2], atol=1e-12)
    np.testing.assert_allclose(canonical.points[:, 2], 0.0, atol=1e-12)
    assert canonical.normalized_arclength[1] > 0.0
    assert canonical.normalized_arclength[-1] == 1.0


def test_preserving_resample_keeps_source_turns():
    surface = make_plane(nx=6, ny=6, samples_per_face=1)
    path = np.array([[-0.4, -0.4, 0.0], [0.0, 0.4, 0.0], [0.4, -0.4, 0.0]])
    canonical = canonicalize_surface_path(
        surface, path, num_tokens=17, preserve_source_waypoints=True
    )
    assert len(canonical.points) == 17
    assert np.min(np.linalg.norm(canonical.points - path[1], axis=1)) < 1e-12


def test_fixed_control_bspline_audit_is_finite_on_cylinder():
    surface = make_cylinder(n_azimuth=16, n_height=6, samples_per_face=1)
    proposal = generate_pattern_proposals(surface, footprint_radius=0.28, overlap=0.6)[0]
    result = audit_fixed_control_bspline(
        surface,
        proposal.plan.active_paths()[0],
        footprint_radius=0.28,
        num_reference_tokens=96,
        num_control_points=16,
    )
    assert np.isfinite(result.geometry_error)
    assert np.isfinite(result.delta_missed_fraction)
    assert np.isfinite(result.relative_path_length_error)
    assert symmetric_modified_hausdorff(np.zeros((2, 3)), np.zeros((2, 3))) == 0.0
