from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.geometry.robot_surface_metric import smooth_surface_normal, surface_vertex_normals
from diffusion_coverage.surface.projection import _closest_points_on_triangles, project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class SurfaceCurve:
    points: np.ndarray
    normals: np.ndarray
    intrinsic_length: float
    requested_length: float
    face_indices: np.ndarray


def trace_surface_curve(
    surface: SurfaceInstance,
    start: np.ndarray,
    initial_direction: np.ndarray,
    length: float,
    *,
    maximum_step: float = 2.5e-4,
) -> SurfaceCurve:
    """Trace a short mesh curve by tangent advance and parallel projection.

    Each step starts from the previous admitted surface point. The tangent is
    projected into the newly reached tangent plane; points are never produced by
    independently projecting samples of one ambient chord.
    """

    if length <= 0.0 or maximum_step <= 0.0:
        raise ValueError("length and maximum_step must be positive")
    vertex_normals = surface_vertex_normals(surface)
    projected, normals = smooth_surface_normal(surface, np.asarray(start)[None], vertex_normals=vertex_normals)
    point = projected[0]
    normal = normals[0]
    direction = np.asarray(initial_direction, dtype=np.float64)
    direction -= normal * np.dot(normal, direction)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("initial direction must have a tangent component")
    direction /= norm
    points = [point.copy()]
    output_normals = [normal.copy()]
    faces = [int(project_points(surface, point[None]).face_indices[0])]
    accumulated = 0.0
    while accumulated < length - 1e-12:
        requested_step = min(maximum_step, length - accumulated)
        trial = point + requested_step * direction
        next_point, next_face = _forward_surface_projection(
            surface, point, trial, direction, requested_step
        )
        next_projection = project_points(surface, next_point[None], allowed_faces=[next_face])
        _, next_normals = smooth_surface_normal(surface, next_point[None], vertex_normals=vertex_normals)
        next_normal = next_normals[0]
        displacement = next_point - point
        step = float(np.linalg.norm(displacement))
        if step <= 1e-10:
            if length - accumulated <= 1e-8:
                break
            raise ValueError("surface trace stalled")
        if accumulated + step > length:
            fraction = (length - accumulated) / step
            next_point = point + fraction * displacement
            next_projection = project_points(surface, next_point[None])
            next_point = next_projection.points[0]
            _, next_normals = smooth_surface_normal(surface, next_point[None], vertex_normals=vertex_normals)
            next_normal = next_normals[0]
            step = float(np.linalg.norm(next_point - point))
        points.append(next_point.copy()); output_normals.append(next_normal.copy())
        faces.append(int(next_projection.face_indices[0])); accumulated += step
        transported = direction - next_normal * np.dot(next_normal, direction)
        transported_norm = float(np.linalg.norm(transported))
        if transported_norm <= 1e-10:
            raise ValueError("parallel tangent update became degenerate")
        direction = transported / transported_norm
        point, normal = next_point, next_normal
        if len(points) > 10000:
            raise RuntimeError("surface trace exceeded safety iteration limit")
    return SurfaceCurve(
        np.asarray(points), np.asarray(output_normals), float(accumulated),
        float(length), np.asarray(faces, dtype=np.int64),
    )


def _forward_surface_projection(
    surface: SurfaceInstance,
    current: np.ndarray,
    trial: np.ndarray,
    direction: np.ndarray,
    requested_step: float,
) -> tuple[np.ndarray, int]:
    """Choose a local face projection with positive progress at mesh creases."""

    triangles = surface.vertices[surface.faces]
    candidates, _ = _closest_points_on_triangles(trial, triangles)
    displacement = candidates - current
    distances = np.linalg.norm(displacement, axis=1)
    forward = displacement @ direction
    trial_error = np.linalg.norm(candidates - trial, axis=1)
    valid = (
        (forward > 1e-10)
        & (distances <= 1.5 * requested_step)
        & (trial_error <= 1.5 * requested_step)
    )
    if not np.any(valid):
        projection = project_points(surface, trial[None])
        return projection.points[0], int(projection.face_indices[0])
    score = np.where(valid, forward - 0.05 * trial_error, -np.inf)
    face = int(np.argmax(score))
    return candidates[face], face
