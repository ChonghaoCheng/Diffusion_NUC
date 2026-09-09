from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class ProjectionResult:
    points: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray
    distances: np.ndarray


def project_points(
    surface: SurfaceInstance,
    points: np.ndarray,
    *,
    allowed_faces: Sequence[int] | None = None,
    chunk_size: int = 256,
) -> ProjectionResult:
    """Project points onto closest mesh triangles using an exact triangle query."""

    query_points = np.asarray(points, dtype=np.float64)
    if query_points.ndim == 1:
        query_points = query_points[None, :]
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("points must have shape [P, 3]")
    if not np.all(np.isfinite(query_points)):
        raise ValueError("points must be finite")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    if allowed_faces is None:
        face_indices = np.arange(surface.num_faces, dtype=np.int64)
    else:
        face_indices = np.asarray(allowed_faces, dtype=np.int64)
        if face_indices.ndim != 1 or len(face_indices) == 0:
            raise ValueError("allowed_faces must be a nonempty one-dimensional sequence")
        if np.any(face_indices < 0) or np.any(face_indices >= surface.num_faces):
            raise ValueError("allowed_faces contain an out-of-range face index")

    triangles = surface.vertices[surface.faces[face_indices]]
    projected = np.empty_like(query_points)
    projected_faces = np.empty(len(query_points), dtype=np.int64)
    barycentric = np.empty((len(query_points), 3), dtype=np.float64)
    distances = np.empty(len(query_points), dtype=np.float64)
    num_faces = len(triangles)
    for start in range(0, len(query_points), chunk_size):
        stop = min(len(query_points), start + chunk_size)
        points_chunk = query_points[start:stop]
        count = len(points_chunk)
        pair_points = np.broadcast_to(points_chunk[:, None, :], (count, num_faces, 3)).reshape(-1, 3)
        pair_triangles = np.broadcast_to(
            triangles[None, :, :, :], (count, num_faces, 3, 3)
        ).reshape(-1, 3, 3)
        candidates, candidate_barycentric = _closest_points_on_triangle_pairs(
            pair_points, pair_triangles
        )
        candidates = candidates.reshape(count, num_faces, 3)
        candidate_barycentric = candidate_barycentric.reshape(count, num_faces, 3)
        differences = candidates - points_chunk[:, None, :]
        squared_distances = np.einsum("pfi,pfi->pf", differences, differences)
        local_faces = np.argmin(squared_distances, axis=1)
        row = np.arange(count)
        projected[start:stop] = candidates[row, local_faces]
        projected_faces[start:stop] = face_indices[local_faces]
        barycentric[start:stop] = candidate_barycentric[row, local_faces]
        distances[start:stop] = np.sqrt(squared_distances[row, local_faces])
    return ProjectionResult(projected, projected_faces, barycentric, distances)


def _closest_points_on_triangles(point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Ericson closest-point regions for one point and F triangles."""

    points = np.broadcast_to(np.asarray(point)[None, :], (len(triangles), 3))
    return _closest_points_on_triangle_pairs(points, triangles)


def _closest_points_on_triangle_pairs(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ericson closest points for aligned point-triangle pairs."""

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = np.einsum("fi,fi->f", ab, ap)
    d2 = np.einsum("fi,fi->f", ac, ap)
    num_faces = len(triangles)
    closest = np.empty((num_faces, 3), dtype=np.float64)
    barycentric = np.empty((num_faces, 3), dtype=np.float64)
    unassigned = np.ones(num_faces, dtype=bool)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    closest[mask] = a[mask]
    barycentric[mask] = np.array([1.0, 0.0, 0.0])
    unassigned &= ~mask

    bp = points - b
    d3 = np.einsum("fi,fi->f", ab, bp)
    d4 = np.einsum("fi,fi->f", ac, bp)
    mask = unassigned & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b[mask]
    barycentric[mask] = np.array([0.0, 1.0, 0.0])
    unassigned &= ~mask

    vc = d1 * d4 - d3 * d2
    mask = unassigned & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    edge_v = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=np.abs(d1 - d3) > 1e-15)
    closest[mask] = a[mask] + edge_v[mask, None] * ab[mask]
    barycentric[mask] = np.column_stack((1.0 - edge_v[mask], edge_v[mask], np.zeros(mask.sum())))
    unassigned &= ~mask

    cp = points - c
    d5 = np.einsum("fi,fi->f", ab, cp)
    d6 = np.einsum("fi,fi->f", ac, cp)
    mask = unassigned & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c[mask]
    barycentric[mask] = np.array([0.0, 0.0, 1.0])
    unassigned &= ~mask

    vb = d5 * d2 - d1 * d6
    mask = unassigned & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    edge_w = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=np.abs(d2 - d6) > 1e-15)
    closest[mask] = a[mask] + edge_w[mask, None] * ac[mask]
    barycentric[mask] = np.column_stack((1.0 - edge_w[mask], np.zeros(mask.sum()), edge_w[mask]))
    unassigned &= ~mask

    va = d3 * d6 - d5 * d4
    bc_numerator = d4 - d3
    bc_denominator = bc_numerator + d5 - d6
    edge_bc = np.divide(
        bc_numerator,
        bc_denominator,
        out=np.zeros_like(bc_numerator),
        where=np.abs(bc_denominator) > 1e-15,
    )
    mask = unassigned & (va <= 0.0) & (bc_numerator >= 0.0) & ((d5 - d6) >= 0.0)
    closest[mask] = b[mask] + edge_bc[mask, None] * (c[mask] - b[mask])
    barycentric[mask] = np.column_stack((np.zeros(mask.sum()), 1.0 - edge_bc[mask], edge_bc[mask]))
    unassigned &= ~mask

    denominator = va + vb + vc
    face_v = np.divide(vb, denominator, out=np.zeros_like(vb), where=np.abs(denominator) > 1e-15)
    face_w = np.divide(vc, denominator, out=np.zeros_like(vc), where=np.abs(denominator) > 1e-15)
    closest[unassigned] = (
        a[unassigned]
        + ab[unassigned] * face_v[unassigned, None]
        + ac[unassigned] * face_w[unassigned, None]
    )
    barycentric[unassigned] = np.column_stack(
        (1.0 - face_v[unassigned] - face_w[unassigned], face_v[unassigned], face_w[unassigned])
    )
    return closest, barycentric
