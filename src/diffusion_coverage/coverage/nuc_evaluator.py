from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoveragePlan
from diffusion_coverage.coverage.evaluator import _length_and_sources, evaluate_coverage
from diffusion_coverage.surface.geodesic import _dijkstra
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class NUCCoverageMetrics:
    """Area-weighted temporal revisit metrics for finite-footprint coverage."""

    missed_error: float
    repeat_error: float
    nuc_error: float
    single_coverage_fraction: float
    overlap_area_fraction: float
    max_visit_count: int
    visit_counts: np.ndarray
    legacy_missed_fraction: float
    legacy_coverage_efficiency: float
    path_length: float
    num_segments: int
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_nuc_coverage(
    surface: SurfaceInstance,
    plan: CoveragePlan,
    *,
    footprint_radius: float,
    path_sample_spacing: float | None = None,
) -> NUCCoverageMetrics:
    """Count distinct footprint episodes without changing the legacy evaluator.

    Each active ``CoveragePlan`` segment is a separate tool-action episode. Within a
    segment, consecutive in-footprint samples form one visit; leaving and returning
    starts another visit.
    """

    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius must be positive")
    spacing = 0.5 * footprint_radius if path_sample_spacing is None else path_sample_spacing
    if spacing <= 0.0:
        raise ValueError("path_sample_spacing must be positive")

    visit_counts = np.zeros(surface.num_samples, dtype=np.int64)
    segment_source_counts: list[int] = []
    for path in plan.active_paths():
        projection = project_points(surface, path)
        _, sources = _length_and_sources(surface, projection, max_spacing=spacing)
        membership = _ordered_footprint_membership(
            surface, sources, footprint_radius=footprint_radius
        )
        starts = membership[:, :1]
        if membership.shape[1] > 1:
            starts = np.concatenate(
                (starts, membership[:, 1:] & ~membership[:, :-1]), axis=1
            )
        visit_counts += starts.sum(axis=1, dtype=np.int64)
        segment_source_counts.append(int(len(sources)))

    weights = surface.area_weights
    total_area = float(weights.sum())
    missed_error = float(weights[visit_counts == 0].sum() / total_area)
    repeat_error = float(
        np.dot(weights, np.maximum(visit_counts - 1, 0)) / total_area
    )
    single_fraction = float(weights[visit_counts == 1].sum() / total_area)
    overlap_fraction = float(weights[visit_counts >= 2].sum() / total_area)
    legacy = evaluate_coverage(
        surface,
        plan,
        footprint_radius=footprint_radius,
        path_sample_spacing=spacing,
    )
    return NUCCoverageMetrics(
        missed_error=missed_error,
        repeat_error=repeat_error,
        nuc_error=missed_error + repeat_error,
        single_coverage_fraction=single_fraction,
        overlap_area_fraction=overlap_fraction,
        max_visit_count=int(visit_counts.max(initial=0)),
        visit_counts=visit_counts,
        legacy_missed_fraction=legacy.missed_fraction,
        legacy_coverage_efficiency=legacy.coverage_efficiency,
        path_length=legacy.path_length,
        num_segments=plan.num_segments,
        metadata={
            "geodesic_backend": legacy.metadata["geodesic_backend"],
            "path_sample_spacing": float(spacing),
            "segment_source_counts": segment_source_counts,
            "episode_definition": "connected sampled-time intervals per active segment",
        },
    )


def _ordered_footprint_membership(
    surface: SurfaceInstance,
    ordered_sources: np.ndarray,
    *,
    footprint_radius: float,
) -> np.ndarray:
    distances = _surface_sample_to_ordered_source_distances(surface, ordered_sources)
    return distances <= footprint_radius + 1e-12


def _surface_sample_to_ordered_source_distances(
    surface: SurfaceInstance, ordered_sources: np.ndarray
) -> np.ndarray:
    """Pairwise version of the legacy mesh-edge geodesic approximation."""

    sources = np.asarray(ordered_sources, dtype=np.float64)
    if sources.ndim != 2 or sources.shape[1] != 3 or len(sources) < 1:
        raise ValueError("ordered_sources must have shape [T, 3] with T >= 1")
    source_projection = project_points(surface, sources)
    source_triangles = surface.faces[source_projection.face_indices]
    source_vertices = surface.vertices[source_triangles]
    source_to_vertices = np.linalg.norm(
        source_vertices - source_projection.points[:, None, :], axis=2
    )
    distances = np.empty((surface.num_samples, len(sources)), dtype=np.float64)
    for sample_index, (sample, face_index) in enumerate(
        zip(surface.sample_points, surface.sample_face_indices)
    ):
        initial = np.full(surface.num_vertices, np.inf, dtype=np.float64)
        face_vertices = surface.faces[int(face_index)]
        initial[face_vertices] = np.linalg.norm(
            surface.vertices[face_vertices] - sample, axis=1
        )
        vertex_distances = _dijkstra(surface, initial)
        row = np.min(vertex_distances[source_triangles] + source_to_vertices, axis=1)
        same_face = source_projection.face_indices == face_index
        if np.any(same_face):
            row[same_face] = np.minimum(
                row[same_face],
                np.linalg.norm(source_projection.points[same_face] - sample, axis=1),
            )
        distances[sample_index] = row
    return distances
