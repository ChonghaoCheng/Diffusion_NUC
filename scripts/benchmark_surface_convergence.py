#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion_coverage.coverage import CoveragePlan, evaluate_coverage
from diffusion_coverage.surface import make_cylinder, make_plane, vertex_geodesic_distances


def main() -> None:
    print("Cylinder convergence")
    print("n_azimuth   area rel.err   half-geodesic rel.err")
    print("------------------------------------------------")
    radius = 1.3
    height = 1.7
    exact_area = 2.0 * np.pi * radius * height
    exact_half_geodesic = np.pi * radius
    for n_azimuth in (12, 24, 48, 96):
        surface = make_cylinder(
            radius=radius,
            height=height,
            n_azimuth=n_azimuth,
            n_height=2,
            samples_per_face=1,
        )
        half_geodesic = vertex_geodesic_distances(surface, [0])[n_azimuth // 2]
        area_error = abs(surface.total_area - exact_area) / exact_area
        distance_error = abs(half_geodesic - exact_half_geodesic) / exact_half_geodesic
        print(f"{n_azimuth:10d} {area_error:14.6e} {distance_error:23.6e}")

    print("\nPlane finite-footprint convergence")
    print("grid       covered fraction   abs.error")
    print("----------------------------------------")
    exact_covered_fraction = 0.4
    plan = CoveragePlan(np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    for resolution in (8, 16, 32):
        surface = make_plane(nx=resolution, ny=resolution, samples_per_face=8)
        metrics = evaluate_coverage(
            surface,
            plan,
            footprint_radius=0.2,
            path_sample_spacing=0.025,
        )
        covered_fraction = 1.0 - metrics.missed_fraction
        print(
            f"{resolution:4d}x{resolution:<4d} {covered_fraction:17.6f} "
            f"{abs(covered_fraction - exact_covered_fraction):11.6f}"
        )


if __name__ == "__main__":
    main()
