from __future__ import annotations

import heapq

import numpy as np

from diffusion_coverage.surface.projection import ProjectionResult, project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def vertex_geodesic_distances(surface: SurfaceInstance, source_vertices: np.ndarray | list[int]) -> np.ndarray:
    sources = np.asarray(source_vertices, dtype=np.int64).reshape(-1)
    if len(sources) == 0:
        raise ValueError("at least one source vertex is required")
    if np.any(sources < 0) or np.any(sources >= surface.num_vertices):
        raise ValueError("source vertex is out of range")
    initial = np.full(surface.num_vertices, np.inf, dtype=np.float64)
    initial[sources] = 0.0
    return _dijkstra(surface, initial)


def surface_sample_distances(surface: SurfaceInstance, source_points: np.ndarray) -> np.ndarray:
    """Approximate intrinsic distance from mesh samples to arbitrary surface sources."""

    source_projection = project_points(surface, source_points)
    vertex_distances = _distances_from_projected_sources(surface, source_projection)
    sample_triangles = surface.faces[surface.sample_face_indices]
    sample_vertex_positions = surface.vertices[sample_triangles]
    via_vertices = vertex_distances[sample_triangles] + np.linalg.norm(
        sample_vertex_positions - surface.sample_points[:, None, :], axis=2
    )
    sample_distances = via_vertices.min(axis=1)

    # Same-face paths are straight in a triangle and should not detour through a vertex.
    for source_point, source_face in zip(source_projection.points, source_projection.face_indices):
        same_face = surface.sample_face_indices == source_face
        if np.any(same_face):
            direct = np.linalg.norm(surface.sample_points[same_face] - source_point, axis=1)
            sample_distances[same_face] = np.minimum(sample_distances[same_face], direct)
    return sample_distances


def geodesic_distance(surface: SurfaceInstance, source: np.ndarray, target: np.ndarray) -> float:
    polyline = shortest_surface_polyline(surface, source, target)
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())


def shortest_surface_polyline(
    surface: SurfaceInstance,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return a topology-preserving mesh-edge approximation to a surface geodesic."""

    source_projection = project_points(surface, np.asarray(source, dtype=np.float64).reshape(1, 3))
    target_projection = project_points(surface, np.asarray(target, dtype=np.float64).reshape(1, 3))
    return shortest_surface_polyline_projected(
        surface,
        source_projection.points[0],
        int(source_projection.face_indices[0]),
        target_projection.points[0],
        int(target_projection.face_indices[0]),
    )


def shortest_surface_polyline_projected(
    surface: SurfaceInstance,
    source: np.ndarray,
    source_face_index: int,
    target: np.ndarray,
    target_face_index: int,
) -> np.ndarray:
    """Mesh-edge shortest polyline when endpoint projections are already known."""

    if source_face_index == target_face_index:
        return np.stack((source, target), axis=0)
    initial = np.full(surface.num_vertices, np.inf, dtype=np.float64)
    source_face = surface.faces[source_face_index]
    initial[source_face] = np.linalg.norm(surface.vertices[source_face] - source, axis=1)
    vertex_distances, predecessors = _dijkstra_with_predecessors(surface, initial)
    target_face = surface.faces[target_face_index]
    target_vertex_cost = np.linalg.norm(surface.vertices[target_face] - target, axis=1)
    target_costs = vertex_distances[target_face] + target_vertex_cost
    if not np.any(np.isfinite(target_costs)):
        raise ValueError("source and target lie on disconnected mesh components")
    final_vertex = int(target_face[np.argmin(target_costs)])

    vertex_path = [final_vertex]
    while predecessors[vertex_path[-1]] >= 0:
        vertex_path.append(int(predecessors[vertex_path[-1]]))
    vertex_path.reverse()
    points = np.vstack(
        (
            source,
            surface.vertices[np.asarray(vertex_path, dtype=np.int64)],
            target,
        )
    )
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-14))
    return points[keep]


def geodesic_polyline_length(surface: SurfaceInstance, waypoints: np.ndarray) -> float:
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("waypoints must have shape [M, 3] with M >= 2")
    return float(sum(geodesic_distance(surface, points[i], points[i + 1]) for i in range(len(points) - 1)))


def resample_projected_polyline(
    surface: SurfaceInstance,
    waypoints: np.ndarray,
    *,
    max_spacing: float,
) -> np.ndarray:
    if max_spacing <= 0.0:
        raise ValueError("max_spacing must be positive")
    projected_waypoints = project_points(surface, waypoints).points
    samples: list[np.ndarray] = [projected_waypoints[0]]
    for start, end in zip(projected_waypoints[:-1], projected_waypoints[1:]):
        surface_polyline = shortest_surface_polyline(surface, start, end)
        for edge_start, edge_end in zip(surface_polyline[:-1], surface_polyline[1:]):
            edge_length = float(np.linalg.norm(edge_end - edge_start))
            intervals = max(1, int(np.ceil(edge_length / max_spacing)))
            interpolation = np.linspace(0.0, 1.0, intervals + 1)[1:, None]
            edge_samples = edge_start[None, :] + interpolation * (edge_end - edge_start)[None, :]
            samples.extend(edge_samples)
    return np.asarray(samples, dtype=np.float64)


def _distances_from_projected_sources(
    surface: SurfaceInstance,
    source_projection: ProjectionResult,
) -> np.ndarray:
    initial = _initial_vertex_distances(surface, source_projection)
    return _dijkstra(surface, initial)


def _initial_vertex_distances(
    surface: SurfaceInstance,
    source_projection: ProjectionResult,
) -> np.ndarray:
    initial = np.full(surface.num_vertices, np.inf, dtype=np.float64)
    for source_point, face_index in zip(source_projection.points, source_projection.face_indices):
        face_vertices = surface.faces[face_index]
        local_costs = np.linalg.norm(surface.vertices[face_vertices] - source_point, axis=1)
        initial[face_vertices] = np.minimum(initial[face_vertices], local_costs)
    return initial


def _dijkstra(surface: SurfaceInstance, initial_distances: np.ndarray) -> np.ndarray:
    distances, _ = _dijkstra_with_predecessors(surface, initial_distances)
    return distances


def _dijkstra_with_predecessors(
    surface: SurfaceInstance,
    initial_distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(initial_distances, dtype=np.float64).copy()
    predecessors = np.full(surface.num_vertices, -1, dtype=np.int64)
    queue = [(float(distance), int(vertex)) for vertex, distance in enumerate(distances) if np.isfinite(distance)]
    heapq.heapify(queue)
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance > distances[vertex]:
            continue
        for neighbour, edge_length in surface.adjacency[vertex]:
            candidate = distance + edge_length
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                predecessors[neighbour] = vertex
                heapq.heappush(queue, (candidate, neighbour))
    return distances, predecessors
