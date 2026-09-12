from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from diffusion_coverage.coverage.nuc_evaluator import NUCCoverageMetrics, _ordered_footprint_membership
from diffusion_coverage.surface.primitives import make_hemisphere
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class CanonicalSymmetryPath:
    surface_id: str
    points: np.ndarray
    normals: np.ndarray
    source_q_start: np.ndarray
    source_path: str
    source_sample_count: int
    source_max_projection_m: float


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def rotation_z(angle_radians: float) -> np.ndarray:
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def saddle_symmetry(name: str) -> np.ndarray:
    matrices = {
        "identity": np.diag((1.0, 1.0, 1.0)),
        "reflect_x": np.diag((-1.0, 1.0, 1.0)),
        "reflect_y": np.diag((1.0, -1.0, 1.0)),
        "rotate_180": np.diag((-1.0, -1.0, 1.0)),
    }
    if name not in matrices:
        raise ValueError(f"unsupported saddle symmetry: {name}")
    return matrices[name]


def analytical_points_and_normals(
    surface_id: str, points: np.ndarray, surface_config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    if surface_id == "hemisphere":
        radius = float(surface_config["radius"])
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("hemisphere path contains the sphere centre")
        unit = values / norms
        projected = radius * unit
        if np.any(projected[:, 2] < -1e-10):
            raise ValueError("path leaves the bounded hemisphere")
        return projected, unit
    if surface_id == "saddle":
        width = float(surface_config["width"])
        height = float(surface_config["height"])
        curvature = float(surface_config["curvature"])
        if np.any(np.abs(values[:, 0]) > width / 2 + 1e-10) or np.any(
            np.abs(values[:, 1]) > height / 2 + 1e-10
        ):
            raise ValueError("path leaves the bounded saddle")
        projected = values.copy()
        projected[:, 2] = curvature * (projected[:, 0] ** 2 - projected[:, 1] ** 2)
        normals = np.column_stack(
            (-2.0 * curvature * projected[:, 0], 2.0 * curvature * projected[:, 1], np.ones(len(projected)))
        )
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return projected, normals
    raise ValueError(f"unsupported surface: {surface_id}")


def load_canonical_path(
    root: Path, archived: dict[str, Any], surface_id: str, surface_config: dict[str, Any], *, maximum_spacing: float
) -> CanonicalSymmetryPath:
    source = root / "results/nuc_robot_skeleton_coupling_v1/witnesses" / f"{surface_id}_P_easy_S00.npz"
    witness = np.load(source)
    transform = np.asarray(
        archived["placements"]["surfaces"][surface_id]["selected"]["P_easy"]["transform_base_from_surface"],
        dtype=np.float64,
    )
    raw = (np.asarray(witness["desired_positions"], dtype=np.float64) - transform[:3, 3]) @ transform[:3, :3]
    projected, _ = analytical_points_and_normals(surface_id, raw, surface_config)
    points = _densify_analytical(surface_id, projected, surface_config, maximum_spacing)
    points, normals = analytical_points_and_normals(surface_id, points, surface_config)
    return CanonicalSymmetryPath(
        surface_id=surface_id,
        points=np.ascontiguousarray(points),
        normals=np.ascontiguousarray(normals),
        source_q_start=np.asarray(witness["q"][0], dtype=np.float64),
        source_path=str(source.relative_to(root)),
        source_sample_count=int(len(raw)),
        source_max_projection_m=float(np.max(np.linalg.norm(projected - raw, axis=1))),
    )


def symmetry_specs(surface_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if surface_id == "hemisphere":
        count = int(config["surfaces"][surface_id]["rotation_count"])
        step = float(config["surfaces"][surface_id]["rotation_step_degrees"])
        return [
            {
                "symmetry_id": f"H{index:02d}",
                "angle_degrees": step * index,
                "matrix": rotation_z(np.deg2rad(step * index)),
            }
            for index in range(count)
        ]
    return [
        {"symmetry_id": f"S{index:02d}", "name": name, "angle_degrees": 0.0 if name == "identity" else None,
         "matrix": saddle_symmetry(name)}
        for index, name in enumerate(config["surfaces"][surface_id]["symmetries"])
    ]


def transform_path(
    canonical: CanonicalSymmetryPath, spec: dict[str, Any], surface_config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(spec["matrix"], dtype=np.float64)
    if np.array_equal(matrix, np.eye(3, dtype=np.float64)):
        return canonical.points.copy(), canonical.normals.copy()
    points = canonical.points @ matrix.T
    expected_points, expected_normals = analytical_points_and_normals(
        canonical.surface_id, points, surface_config
    )
    transported_normals = canonical.normals @ matrix.T
    transported_normals /= np.linalg.norm(transported_normals, axis=1, keepdims=True)
    if not np.allclose(points, expected_points, atol=1e-12, rtol=0):
        raise ValueError("symmetry does not preserve the analytical surface")
    if not np.allclose(transported_normals, expected_normals, atol=1e-12, rtol=0):
        raise ValueError("symmetry does not transport analytical normals")
    return np.ascontiguousarray(points), np.ascontiguousarray(expected_normals)


def make_symmetry_reference_surface(surface_id: str, config: dict[str, Any]) -> SurfaceInstance:
    cfg = config["surfaces"][surface_id]
    if surface_id == "hemisphere":
        base = make_hemisphere(
            radius=cfg["radius"], n_azimuth=cfg["reference_n_azimuth"],
            n_polar=cfg["reference_n_polar"], samples_per_face=1,
        )
        return _with_symmetric_quadrature(base.vertices, base.faces, surface_id, cfg, config)
    if surface_id == "saddle":
        vertices, faces = _symmetric_saddle_mesh(cfg)
        return _with_symmetric_quadrature(vertices, faces, surface_id, cfg, config)
    raise ValueError(f"unsupported surface: {surface_id}")


def evaluate_symmetry_path(
    surface: SurfaceInstance, points: np.ndarray, config: dict[str, Any]
) -> NUCCoverageMetrics:
    radius = float(config["coverage"]["footprint_radius_m"])
    membership = _ordered_footprint_membership(surface, np.asarray(points), footprint_radius=radius)
    starts = np.concatenate((membership[:, :1], membership[:, 1:] & ~membership[:, :-1]), axis=1)
    visits = starts.sum(axis=1, dtype=np.int64)
    weights = surface.area_weights
    total_area = float(weights.sum())
    missed = float(weights[visits == 0].sum() / total_area)
    repeated = float(np.dot(weights, np.maximum(visits - 1, 0)) / total_area)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    covered_area = float(weights[visits > 0].sum())
    efficiency = covered_area / (2.0 * radius * length + np.pi * radius**2)
    return NUCCoverageMetrics(
        missed_error=missed,
        repeat_error=repeated,
        nuc_error=missed + repeated,
        single_coverage_fraction=float(weights[visits == 1].sum() / total_area),
        overlap_area_fraction=float(weights[visits >= 2].sum() / total_area),
        max_visit_count=int(visits.max(initial=0)),
        visit_counts=visits,
        legacy_missed_fraction=missed,
        legacy_coverage_efficiency=float(efficiency),
        path_length=length,
        num_segments=1,
        metadata={
            "geodesic_backend": "mesh_edge_dijkstra_direct_ordered_sources",
            "footprint_radius_m": radius,
            "path_sample_spacing": float(config["coverage"]["path_sample_spacing_m"]),
            "segment_source_counts": [int(len(points))],
            "episode_definition": "connected sampled-time intervals per active segment",
        },
    )


def symmetry_sample_permutation(surface: SurfaceInstance, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    transformed = surface.sample_points @ np.asarray(matrix, dtype=np.float64).T
    distances, indices = cKDTree(surface.sample_points).query(transformed, k=1)
    if len(np.unique(indices)) != surface.num_samples:
        raise RuntimeError("surface symmetry does not induce a bijection of evaluation samples")
    maximum = float(np.max(distances, initial=0.0))
    if maximum > 1e-12:
        raise RuntimeError("surface symmetry does not preserve the common evaluation sample set")
    if not np.allclose(surface.area_weights, surface.area_weights[indices], atol=1e-14, rtol=1e-12):
        raise RuntimeError("surface symmetry does not preserve quadrature weights")
    return np.asarray(indices, dtype=np.int64), maximum


def verify_mesh_automorphism(surface: SurfaceInstance, matrix: np.ndarray) -> float:
    transformed = surface.vertices @ np.asarray(matrix, dtype=np.float64).T
    distances, permutation = cKDTree(surface.vertices).query(transformed, k=1)
    if len(np.unique(permutation)) != surface.num_vertices:
        raise RuntimeError("surface symmetry is not a vertex bijection")
    edges = np.sort(surface.edge_index.T, axis=1)
    edge_set = {tuple(edge) for edge in edges}
    mapped = np.sort(permutation[edges], axis=1)
    if any(tuple(edge) not in edge_set for edge in mapped):
        raise RuntimeError("surface symmetry does not preserve mesh adjacency")
    old_lengths = {tuple(edge): length for edge, length in zip(edges, surface.edge_lengths)}
    length_error = max(
        abs(float(length) - float(old_lengths[tuple(edge)]))
        for edge, length in zip(mapped, surface.edge_lengths)
    )
    return max(float(np.max(distances, initial=0.0)), float(length_error))


def transport_coverage_metrics(
    baseline: NUCCoverageMetrics,
    surface: SurfaceInstance,
    matrix: np.ndarray,
    transformed_points: np.ndarray,
) -> tuple[NUCCoverageMetrics, float]:
    permutation, sample_error = symmetry_sample_permutation(surface, matrix)
    mesh_error = verify_mesh_automorphism(surface, matrix)
    visits = np.empty_like(baseline.visit_counts)
    visits[permutation] = baseline.visit_counts
    weights = surface.area_weights
    total_area = float(weights.sum())
    missed = float(weights[visits == 0].sum() / total_area)
    repeated = float(np.dot(weights, np.maximum(visits - 1, 0)) / total_area)
    length = float(np.linalg.norm(np.diff(transformed_points, axis=0), axis=1).sum())
    covered_area = float(weights[visits > 0].sum())
    radius = float(baseline.metadata["footprint_radius_m"])
    metrics = NUCCoverageMetrics(
        missed_error=missed,
        repeat_error=repeated,
        nuc_error=missed + repeated,
        single_coverage_fraction=float(weights[visits == 1].sum() / total_area),
        overlap_area_fraction=float(weights[visits >= 2].sum() / total_area),
        max_visit_count=int(visits.max(initial=0)),
        visit_counts=visits,
        legacy_missed_fraction=missed,
        legacy_coverage_efficiency=float(covered_area / (2.0 * radius * length + np.pi * radius**2)),
        path_length=length,
        num_segments=1,
        metadata={**baseline.metadata, "symmetry_transport": True},
    )
    return metrics, max(sample_error, mesh_error)


def intrinsic_segment_lengths(surface: SurfaceInstance, points: np.ndarray) -> np.ndarray:
    del surface
    # The archived trace is already densely sampled. This discrete curve-length
    # sequence is preserved exactly by every ambient rigid self-isometry.
    return np.linalg.norm(np.diff(np.asarray(points, dtype=np.float64), axis=0), axis=1)


def choose_invariance_tolerance(records: list[dict[str, Any]], hard_ceiling: float) -> float:
    keys = ("abs_E_miss_error", "abs_E_rep_error", "abs_E_NUC_error", "abs_L_S_error", "max_segment_length_error")
    observed = max(float(row[key]) for row in records for key in keys)
    tolerance = max(1e-12, 10.0 * observed)
    if tolerance > hard_ceiling:
        raise RuntimeError(
            f"symmetry invariance error {observed:.3e} requires tolerance {tolerance:.3e} above ceiling"
        )
    return tolerance


def assert_invariance(records: list[dict[str, Any]], tolerance: float) -> None:
    keys = ("abs_E_miss_error", "abs_E_rep_error", "abs_E_NUC_error", "abs_L_S_error", "max_segment_length_error")
    for row in records:
        if any(float(row[key]) > tolerance for key in keys):
            raise RuntimeError(f"symmetry invariance failed for {row['surface_id']}/{row['symmetry_id']}")
        if not row["sample_order_preserved"] or not row["activity_preserved"] or not row["normal_covariance_pass"]:
            raise RuntimeError(f"symmetry structural contract failed for {row['surface_id']}/{row['symmetry_id']}")


def require_capacity_for_riemannian(summary: dict[str, Any]) -> None:
    if summary.get("decision") != "GO" or not summary.get("riemannian_authorized", False):
        raise RuntimeError("Riemannian symmetry analysis is blocked because E06-G2 capacity did not pass")


def scene_seed(config: dict[str, Any], surface_id: str, placement_level: str, *, strong: bool) -> int:
    surface_index = ("saddle", "hemisphere").index(surface_id)
    placement_index = tuple(config["placement_levels"]).index(placement_level)
    return int(config["seed"]) + 10000 * surface_index + 100 * placement_index + (1 if strong else 0)


def transition_cost_decomposition(q_path: np.ndarray, surface_points: np.ndarray) -> dict[str, float]:
    q = np.asarray(q_path, dtype=np.float64)
    points = np.asarray(surface_points, dtype=np.float64)
    if q.shape[0] != points.shape[0] or q.ndim != 2 or points.shape[1] != 3:
        raise ValueError("q_path and surface_points must share a valid sample axis")
    joint_steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    surface_steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_surface = float(surface_steps.sum())
    progress = (np.cumsum(surface_steps) - 0.5 * surface_steps) / max(total_surface, 1e-15)
    first = progress < 0.05
    last = progress >= 0.95
    middle = ~(first | last)
    return {
        "J_q_first_5pct": float(joint_steps[first].sum()),
        "J_q_middle_90pct": float(joint_steps[middle].sum()),
        "J_q_last_5pct": float(joint_steps[last].sum()),
    }


def _densify_analytical(
    surface_id: str, points: np.ndarray, surface_config: dict[str, Any], maximum_spacing: float
) -> np.ndarray:
    if maximum_spacing <= 0:
        raise ValueError("maximum_spacing must be positive")
    result = [np.asarray(points[0], dtype=np.float64)]
    for start, end in zip(points[:-1], points[1:]):
        intervals = max(1, int(np.ceil(np.linalg.norm(end - start) / maximum_spacing)))
        for fraction in np.linspace(0.0, 1.0, intervals + 1)[1:]:
            candidate = start + fraction * (end - start)
            projected, _ = analytical_points_and_normals(surface_id, candidate[None], surface_config)
            result.append(projected[0])
    return np.asarray(result, dtype=np.float64)


def _with_symmetric_quadrature(
    vertices: np.ndarray, faces: np.ndarray, surface_id: str, metadata: dict[str, Any], config: dict[str, Any]
) -> SurfaceInstance:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(area_vectors, axis=1)
    face_areas = 0.5 * double_areas
    face_normals = area_vectors / double_areas[:, None]
    seeds = ((0.6, 0.3, 0.1),) * int(config["coverage"]["symmetric_barycentric_orbits"])
    barycentric = np.asarray(sorted(set(permutation for seed in seeds for permutation in permutations(seed))), dtype=np.float64)
    sample_barycentric = np.tile(barycentric, (len(faces), 1))
    sample_face_indices = np.repeat(np.arange(len(faces), dtype=np.int64), len(barycentric))
    sample_points = np.einsum("pi,pij->pj", sample_barycentric, triangles[sample_face_indices])
    sample_normals = face_normals[sample_face_indices]
    area_weights = np.repeat(face_areas / len(barycentric), len(barycentric))
    return SurfaceInstance(
        vertices=vertices, faces=faces, sample_points=sample_points, sample_normals=sample_normals,
        area_weights=area_weights, sample_face_indices=sample_face_indices,
        sample_barycentric=sample_barycentric, surface_id=surface_id,
        metadata={**metadata, "symmetry_compatible_quadrature": True},
    )


def _symmetric_saddle_mesh(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = int(cfg["reference_nx"]), int(cfg["reference_ny"])
    x = np.linspace(-float(cfg["width"]) / 2, float(cfg["width"]) / 2, nx + 1)
    y = np.linspace(-float(cfg["height"]) / 2, float(cfg["height"]) / 2, ny + 1)
    curvature = float(cfg["curvature"])
    vertices = [[xi, yi, curvature * (xi * xi - yi * yi)] for yi in y for xi in x]
    faces = []
    width = nx + 1
    for j in range(ny):
        for i in range(nx):
            a, b = j * width + i, j * width + i + 1
            c, d = (j + 1) * width + i, (j + 1) * width + i + 1
            cx, cy = 0.5 * (x[i] + x[i + 1]), 0.5 * (y[j] + y[j + 1])
            centre = len(vertices)
            vertices.append([cx, cy, curvature * (cx * cx - cy * cy)])
            faces.extend(((a, b, centre), (b, d, centre), (d, c, centre), (c, a, centre)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
