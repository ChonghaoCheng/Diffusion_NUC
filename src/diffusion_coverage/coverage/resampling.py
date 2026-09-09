from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoveragePlan
from diffusion_coverage.surface.geodesic import shortest_surface_polyline_projected
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def resample_surface_path(
    surface: SurfaceInstance,
    waypoints: np.ndarray,
    *,
    num_waypoints: int,
) -> np.ndarray:
    """Resample a path by arclength after expanding it along mesh topology."""

    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("waypoints must have shape [L, 3] with L >= 2")
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least two")

    result, _ = resample_surface_path_with_length(
        surface, points, num_waypoints=num_waypoints
    )
    return result


def resample_surface_path_with_length(
    surface: SurfaceInstance,
    waypoints: np.ndarray,
    *,
    num_waypoints: int,
) -> tuple[np.ndarray, float]:
    """Resample and return the source polyline's expanded mesh-geodesic length."""

    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("waypoints must have shape [L, 3] with L >= 2")
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least two")
    projection = project_points(surface, points)
    projected = projection.points
    expanded: list[np.ndarray] = [projected[0]]
    for index, (start, end) in enumerate(zip(projected[:-1], projected[1:])):
        local_path = shortest_surface_polyline_projected(
            surface,
            start,
            int(projection.face_indices[index]),
            end,
            int(projection.face_indices[index + 1]),
        )
        expanded.extend(local_path[1:])
    polyline = np.asarray(expanded, dtype=np.float64)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1e-12))
    polyline = polyline[keep]
    if len(polyline) < 2:
        return np.repeat(polyline[:1], num_waypoints, axis=0), 0.0

    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))))
    targets = np.linspace(0.0, cumulative[-1], num_waypoints)
    result = np.column_stack(
        [np.interp(targets, cumulative, polyline[:, dimension]) for dimension in range(3)]
    )
    # Interpolation along mesh edges remains on the mesh; this projection also
    # removes small floating-point drift at endpoints and same-face segments.
    return project_points(surface, result).points, float(cumulative[-1])


def resample_plan_fixed_waypoints(
    surface: SurfaceInstance,
    plan: CoveragePlan,
    *,
    num_waypoints: int,
) -> CoveragePlan:
    """Convert a single-segment teacher plan to the fixed-M M2 representation."""

    paths = plan.active_paths()
    if len(paths) != 1:
        raise ValueError("fixed-M M2 currently supports exactly one active segment")
    resampled = resample_surface_path(surface, paths[0], num_waypoints=num_waypoints)
    return CoveragePlan(resampled, metadata={**plan.metadata, "fixed_num_waypoints": num_waypoints})
