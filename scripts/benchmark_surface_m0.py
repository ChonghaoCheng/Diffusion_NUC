#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion_coverage.coverage import CoveragePlan, evaluate_coverage
from diffusion_coverage.surface import make_cylinder, make_freeform_patch, make_hemisphere, make_saddle, make_torus


def main() -> None:
    surfaces = [
        make_cylinder(n_azimuth=24, n_height=8),
        make_hemisphere(n_azimuth=24, n_polar=8),
        make_saddle(nx=16, ny=16),
        make_torus(n_major=24, n_minor=10),
        make_freeform_patch(nx=16, ny=16),
    ]
    print("Surface             V      F   Samples   Missed    Length    Eval time")
    print("------------------------------------------------------------------------")
    for surface in surfaces:
        first = surface.vertices[0]
        last = surface.vertices[len(surface.vertices) // 2]
        midpoint = 0.5 * (first + last)
        plan = CoveragePlan(np.stack((first, midpoint, last), axis=0))
        start = perf_counter()
        metrics = evaluate_coverage(surface, plan, footprint_radius=0.2, path_sample_spacing=0.1)
        elapsed = perf_counter() - start
        print(
            f"{surface.surface_id:<18} {surface.num_vertices:5d} {surface.num_faces:6d} "
            f"{surface.num_samples:9d} {metrics.missed_fraction:8.3f} "
            f"{metrics.path_length:9.3f} {elapsed:10.4f}s"
        )


if __name__ == "__main__":
    main()
