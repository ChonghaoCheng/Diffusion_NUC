import numpy as np

from diffusion_coverage.planning.e09_geometry import (
    build_geometry_bank,
    shortest_sphere_arc,
    spherical_distance,
)


def test_short_sphere_arc_has_fixed_endpoints_and_domain():
    radius = 0.14
    start = np.asarray([0.0, 0.0, radius])
    end = np.asarray([radius, 0.0, 0.0])
    arc = shortest_sphere_arc(start, end, radius, 0.01)
    assert np.allclose(arc[0], start)
    assert np.allclose(arc[-1], end)
    assert np.allclose(np.linalg.norm(arc, axis=1), radius)
    assert np.all(arc[:, 2] >= -1e-14)
    assert np.isclose(spherical_distance(start, end, radius), 0.5 * np.pi * radius)


def test_geometry_bank_preserves_both_route_directions():
    radius = 0.14
    path = shortest_sphere_arc(np.asarray([0.0, 0.0, radius]), np.asarray([radius, 0.0, 0.0]), radius, 0.02)
    bank = build_geometry_bank((path,), ("sample",), radius=radius, macro_length=0.05, connector_radius=0.032, nearest_count=6, port_tolerance=1e-9)
    forward = bank.route_arc_ids["sample/forward"]
    reverse = bank.route_arc_ids["sample/reverse"]
    assert len(forward) == len(reverse)
    assert bank.arcs[forward[0]].start_port == bank.arcs[reverse[-1]].end_port
