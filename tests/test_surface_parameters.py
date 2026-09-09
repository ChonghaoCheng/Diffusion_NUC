from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage import (
    densify_parameter_polyline,
    decode_raster_parameter_controls,
    inverse_surface_parameters,
    map_surface_parameters,
    simplify_parameter_polyline,
)
from diffusion_coverage.surface import (
    make_cylinder,
    make_freeform_patch,
    make_hemisphere,
    make_plane,
    make_saddle,
    make_torus,
)


def test_analytic_surface_parameter_roundtrip_preserves_ordered_points():
    surfaces = [
        make_plane(),
        make_cylinder(),
        make_hemisphere(),
        make_saddle(),
        make_freeform_patch(),
        make_torus(),
    ]
    for surface in surfaces:
        if surface.surface_id in {"cylinder", "hemisphere", "torus"}:
            u = np.linspace(0.8, 1.2, 31)
        else:
            u = np.linspace(0.1, 0.9, 31)
        v = np.linspace(0.15, 0.85, 31)
        points = map_surface_parameters(surface, u, v)
        recovered = inverse_surface_parameters(surface, points, unwrap_periodic=True)
        restored = map_surface_parameters(surface, recovered[:, 0], recovered[:, 1])
        assert np.allclose(restored, points, atol=1e-9)
        if surface.surface_id in {"cylinder", "hemisphere", "torus"}:
            assert np.max(np.abs(np.diff(recovered[:, 0]))) < 0.1


def test_parameter_polyline_simplification_preserves_raster_turns_and_winding():
    track_a = np.column_stack((np.linspace(0.0, 1.0, 41), np.full(41, 0.25)))
    connector = np.column_stack((np.ones(5), np.linspace(0.25, 0.75, 5)))
    track_b = np.column_stack((np.linspace(1.0, 0.0, 41), np.full(41, 0.75)))
    dense = np.concatenate((track_a, connector[1:], track_b[1:]))
    controls = simplify_parameter_polyline(dense)
    restored = densify_parameter_polyline(controls, maximum_step=0.025)
    assert controls.shape == (4, 2)
    assert np.allclose(
        controls,
        [[0.0, 0.25], [1.0, 0.25], [1.0, 0.75], [0.0, 0.75]],
    )
    assert np.max(np.linalg.norm(np.diff(restored, axis=0), axis=1)) <= 0.025 + 1e-12
    decoded = decode_raster_parameter_controls(
        make_cylinder(radius=1.0, height=1.0),
        controls,
        footprint_radius=0.1,
    )
    assert len(decoded) > len(controls)
    assert np.array_equal(decoded[-1], controls[-1])
