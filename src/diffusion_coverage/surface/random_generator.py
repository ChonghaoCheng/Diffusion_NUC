from __future__ import annotations

import numpy as np

from diffusion_coverage.surface.primitives import (
    make_cylinder,
    make_freeform_patch,
    make_hemisphere,
    make_saddle,
    make_torus,
)
from diffusion_coverage.surface.surface_instance import SurfaceInstance


SURFACE_TYPES = ("cylinder", "hemisphere", "saddle", "torus", "freeform_patch")


def generate_random_surface(
    surface_type: str,
    *,
    seed: int,
    resolution: int = 12,
    samples_per_face: int = 4,
) -> SurfaceInstance:
    if surface_type not in SURFACE_TYPES:
        raise ValueError(f"unsupported surface_type={surface_type!r}")
    if resolution < 4:
        raise ValueError("resolution must be at least four")
    rng = np.random.default_rng(seed)
    if surface_type == "cylinder":
        return make_cylinder(
            radius=float(rng.uniform(0.8, 1.2)),
            height=float(rng.uniform(1.5, 2.2)),
            n_azimuth=2 * resolution,
            n_height=resolution,
            samples_per_face=samples_per_face,
        )
    if surface_type == "hemisphere":
        return make_hemisphere(
            radius=float(rng.uniform(0.8, 1.2)),
            n_azimuth=2 * resolution,
            n_polar=resolution,
            samples_per_face=samples_per_face,
        )
    if surface_type == "saddle":
        return make_saddle(
            width=float(rng.uniform(1.6, 2.2)),
            height=float(rng.uniform(1.6, 2.2)),
            curvature=float(rng.uniform(0.15, 0.35)),
            nx=resolution,
            ny=resolution,
            samples_per_face=samples_per_face,
        )
    if surface_type == "torus":
        return make_torus(
            major_radius=float(rng.uniform(0.8, 1.2)),
            minor_radius=float(rng.uniform(0.22, 0.34)),
            n_major=2 * resolution,
            n_minor=resolution,
            samples_per_face=samples_per_face,
        )
    return make_freeform_patch(
        width=float(rng.uniform(1.6, 2.2)),
        height=float(rng.uniform(1.6, 2.2)),
        amplitude=float(rng.uniform(0.12, 0.28)),
        nx=resolution,
        ny=resolution,
        samples_per_face=samples_per_face,
    )
