from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoverageMetrics, CoveragePlan
from diffusion_coverage.surface.geodesic import (
    shortest_surface_polyline_projected,
    surface_sample_distances,
)
from diffusion_coverage.surface.projection import ProjectionResult, project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def evaluate_coverage(
    surface: SurfaceInstance,
    plan: CoveragePlan,
    *,
    footprint_radius: float,
    path_sample_spacing: float | None = None,
) -> CoverageMetrics:
    """Evaluate finite-footprint coverage using area-weighted surface quadrature."""

    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius must be positive")
    spacing = 0.5 * footprint_radius if path_sample_spacing is None else path_sample_spacing
    if spacing <= 0.0:
        raise ValueError("path_sample_spacing must be positive")

    path_length = 0.0
    path_sources: list[np.ndarray] = []
    projection_distances: list[np.ndarray] = []
    for path in plan.active_paths():
        projection = project_points(surface, path)
        projection_distances.append(projection.distances)
        local_length, local_sources = _length_and_sources(
            surface, projection, max_spacing=spacing
        )
        path_length += local_length
        path_sources.append(local_sources)
    sources = np.concatenate(path_sources, axis=0)
    distances = surface_sample_distances(surface, sources)
    covered = distances <= footprint_radius + 1e-12
    covered_area = float(surface.area_weights[covered].sum())
    total_area = float(surface.area_weights.sum())
    missed_fraction = max(0.0, min(1.0, 1.0 - covered_area / total_area))
    denominator = 2.0 * footprint_radius * path_length + plan.num_segments * np.pi * footprint_radius**2
    coverage_efficiency = covered_area / denominator
    all_projection_distances = np.concatenate(projection_distances)
    return CoverageMetrics(
        missed_fraction=missed_fraction,
        covered_area=covered_area,
        total_area=total_area,
        path_length=path_length,
        coverage_efficiency=float(coverage_efficiency),
        num_segments=plan.num_segments,
        max_projection_distance=float(all_projection_distances.max()),
        mean_projection_distance=float(all_projection_distances.mean()),
        metadata={
            "geodesic_backend": "mesh_edge_dijkstra",
            "path_sample_spacing": float(spacing),
            "num_path_sources": int(len(sources)),
            "num_surface_samples": surface.num_samples,
        },
    )


def _length_and_sources(
    surface: SurfaceInstance,
    projection: ProjectionResult,
    *,
    max_spacing: float,
) -> tuple[float, np.ndarray]:
    """Compute path length and footprint sources from one geodesic traversal."""

    samples: list[np.ndarray] = [projection.points[0]]
    total_length = 0.0
    for index, (start, end) in enumerate(zip(projection.points[:-1], projection.points[1:])):
        surface_polyline = shortest_surface_polyline_projected(
            surface,
            start,
            int(projection.face_indices[index]),
            end,
            int(projection.face_indices[index + 1]),
        )
        for edge_start, edge_end in zip(surface_polyline[:-1], surface_polyline[1:]):
            edge_length = float(np.linalg.norm(edge_end - edge_start))
            total_length += edge_length
            intervals = max(1, int(np.ceil(edge_length / max_spacing)))
            interpolation = np.linspace(0.0, 1.0, intervals + 1)[1:, None]
            samples.extend(edge_start[None, :] + interpolation * (edge_end - edge_start)[None, :])
    return total_length, np.asarray(samples, dtype=np.float64)
