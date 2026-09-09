from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from diffusion_coverage.coverage.resampling import (
    resample_surface_path,
    resample_surface_path_with_length,
)
from diffusion_coverage.surface.geodesic import geodesic_polyline_length
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.geodesic import shortest_surface_polyline_projected
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class CanonicalPath:
    points: np.ndarray
    normalized_arclength: np.ndarray
    intrinsic_length: float
    reversed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        arclength = np.asarray(self.normalized_arclength, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("points must have shape [M, 3] with M >= 2")
        if arclength.shape != (len(points),):
            raise ValueError("normalized_arclength must have shape [M]")
        if not np.all(np.diff(arclength) >= 0.0) or not np.isclose(arclength[0], 0.0) or not np.isclose(arclength[-1], 1.0):
            raise ValueError("normalized_arclength must increase from zero to one")
        if self.intrinsic_length <= 0.0:
            raise ValueError("intrinsic_length must be positive")
        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "normalized_arclength", np.ascontiguousarray(arclength))


@dataclass(frozen=True)
class PaddedPathBatch:
    points: np.ndarray
    mask: np.ndarray
    normalized_arclength: np.ndarray
    intrinsic_lengths: np.ndarray


def pad_canonical_paths(paths: list[CanonicalPath] | tuple[CanonicalPath, ...]) -> PaddedPathBatch:
    """Pad variable-resolution paths while keeping every padded token explicitly masked."""

    if not paths:
        raise ValueError("at least one path is required")
    batch_size = len(paths)
    maximum_tokens = max(len(path.points) for path in paths)
    points = np.zeros((batch_size, maximum_tokens, 3), dtype=np.float32)
    mask = np.zeros((batch_size, maximum_tokens), dtype=bool)
    arclength = np.zeros((batch_size, maximum_tokens), dtype=np.float32)
    lengths = np.empty(batch_size, dtype=np.float32)
    for index, path in enumerate(paths):
        count = len(path.points)
        points[index, :count] = path.points
        mask[index, :count] = True
        arclength[index, :count] = path.normalized_arclength
        lengths[index] = path.intrinsic_length
    return PaddedPathBatch(points, mask, arclength, lengths)


def suggested_token_count(
    surface_area: float,
    footprint_radius: float,
    *,
    tokens_per_footprint_area: float = 1.0,
    minimum: int = 32,
    maximum: int = 2048,
) -> int:
    """Choose path bandwidth from known task scale rather than predicting token count."""

    if surface_area <= 0.0 or footprint_radius <= 0.0:
        raise ValueError("surface_area and footprint_radius must be positive")
    if tokens_per_footprint_area <= 0.0 or minimum < 2 or maximum < minimum:
        raise ValueError("invalid token-count parameters")
    estimate = int(np.ceil(tokens_per_footprint_area * surface_area / footprint_radius**2))
    return min(maximum, max(minimum, estimate))


def canonicalize_surface_path(
    surface: SurfaceInstance,
    waypoints: np.ndarray,
    *,
    num_tokens: int,
    anchor: np.ndarray | None = None,
    direction_axis: np.ndarray | None = None,
    closed_tolerance: float = 1e-6,
    topology_safe_source: bool = False,
    preserve_source_waypoints: bool = False,
) -> CanonicalPath:
    """Remove open-path endpoint, direction, and sampling gauges without merging real modes."""

    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("waypoints must have shape [L, 3] with L >= 2")
    if num_tokens < 2:
        raise ValueError("num_tokens must be at least two")
    surface_extent = float(np.linalg.norm(np.ptp(surface.vertices, axis=0)))
    is_closed = float(np.linalg.norm(points[0] - points[-1])) <= closed_tolerance * max(surface_extent, 1.0)
    reversed_path = False
    if not is_closed:
        task_anchor = np.min(surface.vertices, axis=0) if anchor is None else np.asarray(anchor, dtype=np.float64)
        axis = np.asarray((1.0, 0.0, 0.0) if direction_axis is None else direction_axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if task_anchor.shape != (3,) or axis.shape != (3,) or axis_norm <= 1e-12:
            raise ValueError("anchor and direction_axis must be valid 3-vectors")
        start_distance = float(np.linalg.norm(points[0] - task_anchor))
        end_distance = float(np.linalg.norm(points[-1] - task_anchor))
        tolerance = 1e-10 * max(surface_extent, 1.0)
        if end_distance < start_distance - tolerance:
            reversed_path = True
        elif abs(end_distance - start_distance) <= tolerance:
            start_tangent = points[1] - points[0]
            reverse_tangent = points[-2] - points[-1]
            reversed_path = float(np.dot(reverse_tangent, axis)) > float(np.dot(start_tangent, axis))
        if reversed_path:
            points = points[::-1]

    if preserve_source_waypoints:
        resampled, normalized_arclength, intrinsic_length = _resample_preserving_waypoints(
            surface, points, max(num_tokens, len(points))
        )
    elif topology_safe_source:
        resampled, intrinsic_length = _resample_known_surface_polyline(
            surface, points, num_tokens
        )
    else:
        resampled, intrinsic_length = resample_surface_path_with_length(
            surface, points, num_waypoints=num_tokens
        )
    if not preserve_source_waypoints:
        normalized_arclength = np.linspace(0.0, 1.0, num_tokens)
    return CanonicalPath(
        points=resampled,
        normalized_arclength=normalized_arclength,
        intrinsic_length=intrinsic_length,
        reversed=reversed_path,
        metadata={"closed": is_closed, "gauge": "task_anchor_then_tangent"},
    )


def _resample_preserving_waypoints(
    surface: SurfaceInstance,
    points: np.ndarray,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    projection = project_points(surface, points)
    local_paths: list[np.ndarray] = []
    segment_lengths = np.empty(len(points) - 1, dtype=np.float64)
    for index in range(len(points) - 1):
        local = shortest_surface_polyline_projected(
            surface,
            projection.points[index],
            int(projection.face_indices[index]),
            projection.points[index + 1],
            int(projection.face_indices[index + 1]),
        )
        local_paths.append(local)
        segment_lengths[index] = np.linalg.norm(np.diff(local, axis=0), axis=1).sum()
    intrinsic_length = float(segment_lengths.sum())
    if intrinsic_length <= 1e-12:
        raise ValueError("source path has zero intrinsic length")

    # Every source segment receives at least one interval. Remaining bandwidth
    # is distributed by intrinsic segment length, so source turns cannot be
    # shortcut by a global resampling operation.
    intervals = np.ones(len(segment_lengths), dtype=np.int64)
    remaining = num_tokens - 1 - len(segment_lengths)
    if remaining > 0:
        quotas = remaining * segment_lengths / intrinsic_length
        extras = np.floor(quotas).astype(np.int64)
        intervals += extras
        leftover = remaining - int(extras.sum())
        if leftover:
            order = np.argsort(-(quotas - extras), kind="stable")
            intervals[order[:leftover]] += 1

    output_points: list[np.ndarray] = [projection.points[0]]
    output_arclength: list[float] = [0.0]
    offset = 0.0
    for local, length, count in zip(local_paths, segment_lengths, intervals):
        local_cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1)))
        )
        targets = np.linspace(0.0, length, int(count) + 1)[1:]
        interpolated = np.column_stack(
            [np.interp(targets, local_cumulative, local[:, dimension]) for dimension in range(3)]
        )
        output_points.extend(interpolated)
        output_arclength.extend(offset + targets)
        offset += length
    result = project_points(surface, np.asarray(output_points)).points
    return result, np.asarray(output_arclength) / intrinsic_length, intrinsic_length


def _resample_known_surface_polyline(
    surface: SurfaceInstance,
    points: np.ndarray,
    num_tokens: int,
) -> tuple[np.ndarray, float]:
    projected = project_points(surface, points).points
    segment_lengths = np.linalg.norm(np.diff(projected, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-12))
    projected = projected[keep]
    if len(projected) < 2:
        raise ValueError("topology-safe source path has zero length")
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(projected, axis=0), axis=1)))
    )
    targets = np.linspace(0.0, cumulative[-1], num_tokens)
    resampled = np.column_stack(
        [np.interp(targets, cumulative, projected[:, dimension]) for dimension in range(3)]
    )
    return project_points(surface, resampled).points, float(cumulative[-1])
