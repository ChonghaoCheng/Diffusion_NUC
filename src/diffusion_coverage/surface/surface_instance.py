from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurfaceInstance:
    """Triangular surface mesh with deterministic area-weighted quadrature samples."""

    vertices: np.ndarray
    faces: np.ndarray
    sample_points: np.ndarray
    sample_normals: np.ndarray
    area_weights: np.ndarray
    sample_face_indices: np.ndarray
    sample_barycentric: np.ndarray
    surface_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    face_normals: np.ndarray = field(init=False, repr=False)
    face_areas: np.ndarray = field(init=False, repr=False)
    edge_index: np.ndarray = field(init=False, repr=False)
    edge_lengths: np.ndarray = field(init=False, repr=False)
    adjacency: tuple[tuple[tuple[int, float], ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        sample_points = np.asarray(self.sample_points, dtype=np.float64)
        sample_normals = np.asarray(self.sample_normals, dtype=np.float64)
        area_weights = np.asarray(self.area_weights, dtype=np.float64)
        sample_face_indices = np.asarray(self.sample_face_indices, dtype=np.int64)
        sample_barycentric = np.asarray(self.sample_barycentric, dtype=np.float64)

        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("vertices must have shape [V, 3] with V >= 3")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
            raise ValueError("faces must have shape [F, 3] with F >= 1")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces contain an out-of-range vertex index")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")

        triangles = vertices[faces]
        area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        double_areas = np.linalg.norm(area_vectors, axis=1)
        if np.any(double_areas <= 1e-14):
            raise ValueError("mesh contains a degenerate triangle")
        face_areas = 0.5 * double_areas
        face_normals = area_vectors / double_areas[:, None]

        num_samples = len(sample_points)
        if sample_points.shape != (num_samples, 3) or num_samples < 1:
            raise ValueError("sample_points must have shape [P, 3] with P >= 1")
        if sample_normals.shape != (num_samples, 3):
            raise ValueError("sample_normals must have shape [P, 3]")
        if area_weights.shape != (num_samples,):
            raise ValueError("area_weights must have shape [P]")
        if sample_face_indices.shape != (num_samples,):
            raise ValueError("sample_face_indices must have shape [P]")
        if sample_barycentric.shape != (num_samples, 3):
            raise ValueError("sample_barycentric must have shape [P, 3]")
        if np.any(sample_face_indices < 0) or np.any(sample_face_indices >= len(faces)):
            raise ValueError("sample_face_indices contain an out-of-range face index")
        if np.any(area_weights <= 0.0) or not np.all(np.isfinite(area_weights)):
            raise ValueError("area_weights must be positive and finite")
        if not np.isclose(area_weights.sum(), face_areas.sum(), rtol=1e-8, atol=1e-12):
            raise ValueError("area_weights must sum to the mesh surface area")
        if not np.allclose(sample_barycentric.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("sample barycentric coordinates must sum to one")
        if np.any(sample_barycentric < -1e-10):
            raise ValueError("sample barycentric coordinates must be nonnegative")
        if not np.all(np.isfinite(sample_points)) or not np.all(np.isfinite(sample_normals)):
            raise ValueError("surface samples must be finite")
        expected_sample_points = np.einsum(
            "pi,pij->pj", sample_barycentric, triangles[sample_face_indices]
        )
        if not np.allclose(sample_points, expected_sample_points, rtol=1e-8, atol=1e-10):
            raise ValueError("sample_points do not match their face and barycentric coordinates")
        if not np.allclose(np.linalg.norm(sample_normals, axis=1), 1.0, atol=1e-8):
            raise ValueError("sample_normals must be unit length")

        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
        edges.sort(axis=1)
        edge_index = np.unique(edges, axis=0)
        edge_lengths = np.linalg.norm(vertices[edge_index[:, 1]] - vertices[edge_index[:, 0]], axis=1)
        adjacency_lists: list[list[tuple[int, float]]] = [[] for _ in range(len(vertices))]
        for (source, target), length in zip(edge_index, edge_lengths):
            adjacency_lists[int(source)].append((int(target), float(length)))
            adjacency_lists[int(target)].append((int(source), float(length)))
        adjacency = tuple(tuple(neighbours) for neighbours in adjacency_lists)

        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "faces", np.ascontiguousarray(faces))
        object.__setattr__(self, "sample_points", np.ascontiguousarray(sample_points))
        object.__setattr__(self, "sample_normals", np.ascontiguousarray(sample_normals))
        object.__setattr__(self, "area_weights", np.ascontiguousarray(area_weights))
        object.__setattr__(self, "sample_face_indices", np.ascontiguousarray(sample_face_indices))
        object.__setattr__(self, "sample_barycentric", np.ascontiguousarray(sample_barycentric))
        object.__setattr__(self, "face_normals", np.ascontiguousarray(face_normals))
        object.__setattr__(self, "face_areas", np.ascontiguousarray(face_areas))
        object.__setattr__(self, "edge_index", np.ascontiguousarray(edge_index.T))
        object.__setattr__(self, "edge_lengths", np.ascontiguousarray(edge_lengths))
        object.__setattr__(self, "adjacency", adjacency)

    @classmethod
    def from_mesh(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        samples_per_face: int = 4,
        surface_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SurfaceInstance:
        if samples_per_face < 1:
            raise ValueError("samples_per_face must be positive")
        vertices_array = np.asarray(vertices, dtype=np.float64)
        faces_array = np.asarray(faces, dtype=np.int64)
        triangles = vertices_array[faces_array]
        area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        double_areas = np.linalg.norm(area_vectors, axis=1)
        if np.any(double_areas <= 1e-14):
            raise ValueError("mesh contains a degenerate triangle")
        face_areas = 0.5 * double_areas
        face_normals = area_vectors / double_areas[:, None]

        barycentric = _deterministic_barycentric_samples(samples_per_face)
        num_faces = len(faces_array)
        sample_barycentric = np.tile(barycentric, (num_faces, 1))
        sample_face_indices = np.repeat(np.arange(num_faces, dtype=np.int64), samples_per_face)
        sampled_triangles = triangles[sample_face_indices]
        sample_points = np.einsum("pi,pij->pj", sample_barycentric, sampled_triangles)
        sample_normals = face_normals[sample_face_indices]
        area_weights = np.repeat(face_areas / samples_per_face, samples_per_face)
        return cls(
            vertices=vertices_array,
            faces=faces_array,
            sample_points=sample_points,
            sample_normals=sample_normals,
            area_weights=area_weights,
            sample_face_indices=sample_face_indices,
            sample_barycentric=sample_barycentric,
            surface_id=surface_id,
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def num_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def num_faces(self) -> int:
        return int(len(self.faces))

    @property
    def num_samples(self) -> int:
        return int(len(self.sample_points))

    @property
    def total_area(self) -> float:
        return float(self.face_areas.sum())


def _deterministic_barycentric_samples(count: int) -> np.ndarray:
    """Equal-weight low-discrepancy points on the reference triangle."""

    index = np.arange(count, dtype=np.float64)
    u = (index + 0.5) / count
    v = np.mod((index + 0.5) * 0.6180339887498949, 1.0)
    sqrt_u = np.sqrt(u)
    return np.column_stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v))
