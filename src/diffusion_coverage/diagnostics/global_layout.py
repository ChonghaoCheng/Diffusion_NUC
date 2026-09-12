from __future__ import annotations

from dataclasses import replace
import hashlib
from time import perf_counter
from typing import Any

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoveragePlan
from diffusion_coverage.coverage.nuc_evaluator import NUCCoverageMetrics, evaluate_nuc_coverage
from diffusion_coverage.nuc.adapter import NUCSkeleton, generate_nuc_skeleton, validate_nuc_skeleton
from diffusion_coverage.nuc.robot_lift import NUCIKCatalog, NUCLiftResult, minimum_cost_nuc_lift
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import (
    UR5eKinematics,
    interpolate_vertex_normals,
    transform_surface_pose_path,
)
from diffusion_coverage.surface.geodesic import resample_projected_polyline, surface_sample_distances
from diffusion_coverage.surface.primitives import make_hemisphere, make_saddle
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def make_reference_surface(surface_id: str, config: dict[str, Any]) -> SurfaceInstance:
    cfg = config["surfaces"][surface_id]
    if surface_id == "saddle":
        return make_saddle(
            width=cfg["width"], height=cfg["height"], curvature=cfg["curvature"],
            nx=cfg["reference_nx"], ny=cfg["reference_ny"],
            samples_per_face=cfg["reference_samples_per_face"],
        )
    if surface_id == "hemisphere":
        return make_hemisphere(
            radius=cfg["radius"], n_azimuth=cfg["reference_n_azimuth"],
            n_polar=cfg["reference_n_polar"],
            samples_per_face=cfg["reference_samples_per_face"],
        )
    raise ValueError(f"unsupported E06-G surface: {surface_id}")


def make_planning_remesh(
    surface_id: str,
    remesh_id: str,
    config: dict[str, Any],
    *,
    samples_per_face: int = 1,
) -> SurfaceInstance:
    if remesh_id not in config["remesh_ids"]:
        raise ValueError(f"unknown remesh: {remesh_id}")
    cfg = config["surfaces"][surface_id]
    if surface_id == "saddle":
        canonical = make_saddle(
            width=cfg["width"], height=cfg["height"], curvature=cfg["curvature"],
            nx=cfg["planning_nx"], ny=cfg["planning_ny"], samples_per_face=1,
        )
        if remesh_id == "M00":
            faces = canonical.faces.copy()
        else:
            pattern = {"M01": "opposite", "M02": "checker", "M03": "x_stripe"}[remesh_id]
            faces = _grid_faces_pattern(cfg["planning_nx"], cfg["planning_ny"], pattern)
        return SurfaceInstance.from_mesh(
            canonical.vertices, faces, samples_per_face=samples_per_face,
            surface_id=surface_id, metadata={**canonical.metadata, "remesh_id": remesh_id},
        )
    if surface_id == "hemisphere":
        phase = 0.0 if remesh_id in {"M00", "M02"} else 0.5
        pattern = {"M00": "canonical", "M01": "canonical", "M02": "opposite", "M03": "checker"}[remesh_id]
        vertices, faces = _hemisphere_mesh(
            cfg["radius"], cfg["planning_n_azimuth"], cfg["planning_n_polar"],
            azimuth_phase=phase, diagonal_pattern=pattern,
        )
        return SurfaceInstance.from_mesh(
            vertices, faces, samples_per_face=samples_per_face, surface_id=surface_id,
            metadata={"radius": cfg["radius"], "remesh_id": remesh_id, "azimuth_phase_cells": phase},
        )
    raise ValueError(f"unsupported E06-G surface: {surface_id}")


def surface_hash(surface: SurfaceInstance) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(surface.vertices, dtype="<f8").tobytes())
    digest.update(np.asarray(surface.faces, dtype="<i8").tobytes())
    digest.update(np.asarray(surface.sample_points, dtype="<f8").tobytes())
    return digest.hexdigest()


def remesh_statistics(
    surface_id: str,
    remesh: SurfaceInstance,
    reference: SurfaceInstance,
    config: dict[str, Any],
) -> dict[str, Any]:
    projection = project_points(remesh, reference.sample_points)
    boundary_vertices = _boundary_vertices(remesh.faces)
    boundary = remesh.vertices[boundary_vertices]
    cfg = config["surfaces"][surface_id]
    if surface_id == "saddle":
        analytic = cfg["curvature"] * (boundary[:, 0] ** 2 - boundary[:, 1] ** 2)
        boundary_residual = np.max(np.abs(boundary[:, 2] - analytic), initial=0.0)
    else:
        radial = np.linalg.norm(boundary[:, :2], axis=1)
        boundary_residual = max(
            np.max(np.abs(radial - cfg["radius"]), initial=0.0),
            np.max(np.abs(boundary[:, 2]), initial=0.0),
        )
    return {
        "vertex_count": remesh.num_vertices,
        "face_count": remesh.num_faces,
        "mean_edge_length_m": float(np.mean(remesh.edge_lengths)),
        "edge_length_cv": float(np.std(remesh.edge_lengths) / np.mean(remesh.edge_lengths)),
        "maximum_reference_projection_error_m": float(np.max(projection.distances)),
        "mean_reference_projection_error_m": float(np.mean(projection.distances)),
        "boundary_analytic_residual_m": float(boundary_residual),
        "reference_surface_hash": surface_hash(reference),
    }


def validate_remesh_library(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    limits = config["remesh_admission"]
    canonical = next(row for row in records if row["remesh_id"] == "M00")
    for row in records:
        if row["vertex_count"] != canonical["vertex_count"] or row["face_count"] != canonical["face_count"]:
            raise ValueError("remesh resolution counts differ from M00")
        relative_mean = abs(row["mean_edge_length_m"] - canonical["mean_edge_length_m"]) / canonical["mean_edge_length_m"]
        if relative_mean > limits["maximum_mean_edge_relative_change"] + 1e-12:
            raise ValueError("remesh mean edge length differs from M00")
        if abs(row["edge_length_cv"] - canonical["edge_length_cv"]) > limits["maximum_edge_cv_absolute_change"] + 1e-12:
            raise ValueError("remesh edge-length CV differs from M00")
        if row["maximum_reference_projection_error_m"] > limits["maximum_reference_projection_error_m"] + 1e-12:
            raise ValueError("remesh exceeds physical-reference approximation tolerance")
        if row["boundary_analytic_residual_m"] > limits["maximum_boundary_analytic_residual_m"] + 1e-12:
            raise ValueError("remesh boundary differs from analytic physical boundary")


def generate_physical_roots(
    reference: SurfaceInstance,
    canonical: SurfaceInstance,
    count: int,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("root count must be positive")
    root0 = project_points(reference, canonical.vertices[canonical.faces[0]].mean(axis=0)[None]).points[0]
    roots = [root0]
    minimum = surface_sample_distances(reference, np.asarray(roots))
    sample_indices = [None]
    while len(roots) < count:
        selected = int(np.argmax(minimum))
        roots.append(reference.sample_points[selected].copy())
        sample_indices.append(selected)
        minimum = np.minimum(minimum, surface_sample_distances(reference, roots[-1][None]))
    return [
        {"root_id": f"R{index:02d}", "position": point.tolist(), "reference_sample_index": sample_indices[index]}
        for index, point in enumerate(roots)
    ]


def map_physical_root(remesh: SurfaceInstance, root_position: np.ndarray) -> dict[str, Any]:
    projection = project_points(remesh, np.asarray(root_position, dtype=np.float64)[None])
    return {
        "mapped_face": int(projection.face_indices[0]),
        "mapping_distance_m": float(projection.distances[0]),
        "mapped_position": projection.points[0].tolist(),
    }


def generate_layout(
    reference: SurfaceInstance,
    remesh: SurfaceInstance,
    root: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], NUCSkeleton, NUCCoverageMetrics]:
    mapping = map_physical_root(remesh, np.asarray(root["position"], dtype=np.float64))
    if mapping["mapping_distance_m"] > config["root_mapping_tolerance_m"] + 1e-12:
        raise ValueError("physical root mapping exceeds tolerance")
    policy = config["nuc"]["expansion_policy"]
    generation_start = perf_counter()
    skeleton = generate_nuc_skeleton(remesh, policy=policy, root_face=mapping["mapped_face"])
    generation_time = perf_counter() - generation_start
    validate_nuc_skeleton(remesh.vertices, remesh.faces, skeleton)
    points = skeleton.waypoints.copy()
    refinement_start = perf_counter()
    for _ in range(config["nuc"]["local_refinement_iterations"]):
        points = project_points(remesh, points).points
    refinement_time = perf_counter() - refinement_start
    mapped = project_points(reference, points).points
    skeleton = replace(
        skeleton, waypoints=mapped,
        metadata={**skeleton.metadata, "evaluation_surface_hash": surface_hash(reference)},
    )
    metrics = evaluate_nuc_coverage(
        reference, CoveragePlan(mapped),
        footprint_radius=config["coverage"]["footprint_radius_m"],
        path_sample_spacing=config["coverage"]["path_sample_spacing_m"],
    )
    fingerprint = hashlib.sha256(
        skeleton.topological_path.astype("<i8").tobytes() + skeleton.tree_edges.astype("<i8").tobytes()
    ).hexdigest()
    record = {
        **mapping,
        "root_id": root["root_id"],
        "physical_root_position": list(root["position"]),
        "expansion_policy": policy,
        "skeleton_fingerprint": fingerprint,
        "topological_path": skeleton.topological_path.tolist(),
        "tree_edges": skeleton.tree_edges.tolist(),
        "ordered_physical_path": mapped.tolist(),
        "evaluation_surface_hash": surface_hash(reference),
        "NUC_generation_time": generation_time,
        "local_refinement_time": refinement_time,
        "E_miss": metrics.missed_error,
        "E_rep": metrics.repeat_error,
        "E_NUC": metrics.nuc_error,
        "single_coverage_fraction": metrics.single_coverage_fraction,
        "overlap_area_fraction": metrics.overlap_area_fraction,
        "max_visit_count": metrics.max_visit_count,
        "L_S": metrics.path_length,
    }
    return record, skeleton, metrics


def select_geometry_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {"E_NUC", "L_S", "root_id", "remesh_id"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("geometry baseline rows are incomplete")
    return min(rows, key=lambda row: (row["E_NUC"], row["L_S"], row["root_id"], row["remesh_id"]))


def coverage_equivalent(rows: list[dict[str, Any]], baseline: dict[str, Any], delta_nuc: float) -> list[dict[str, Any]]:
    limit = float(baseline["E_NUC"]) + float(delta_nuc)
    return [row for row in rows if float(row["E_NUC"]) <= limit + 1e-12]


def finite_verified_oracle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("overall_pass") is True and row.get("J_q") is not None]
    return None if not eligible else min(eligible, key=lambda row: (float(row["J_q"]), row["layout_id"]))


def code_points_from_layout(layout: dict[str, Any]) -> np.ndarray:
    codes = np.asarray(layout["topological_path"], dtype=np.int64)
    ordered = np.asarray(layout["ordered_physical_path"], dtype=np.float64)
    points = np.empty_like(ordered)
    points[codes] = ordered
    return points


def build_ordered_path_ik_catalog(
    robot: UR5eKinematics,
    reference: SurfaceInstance,
    code_points: np.ndarray,
    transform: np.ndarray,
    *,
    axis_tolerance: float,
    characteristic_length: float,
    sigma_safe: float,
    random_restarts: int,
    max_candidates: int,
    orientation_cone_samples: int,
    seed: int,
) -> NUCIKCatalog:
    projection = project_points(reference, code_points)
    normals = interpolate_vertex_normals(
        reference.vertices, reference.faces, reference.face_normals, reference.face_areas,
        projection.face_indices, projection.barycentric,
    )
    positions, axes = transform_surface_pose_path(projection.points, normals, transform)
    rng = np.random.default_rng(seed)
    layers = []
    from time import perf_counter
    start = perf_counter()
    for position, axis in zip(positions, axes):
        candidates = robot.enumerate_ik(
            position, axis, random_restarts=random_restarts, rng=rng,
            axis_tolerance=axis_tolerance, minimum_manipulability=0.0,
            max_candidates=max_candidates, orientation_cone_samples=orientation_cone_samples,
        )
        layers.append(tuple(
            candidate for candidate in candidates
            if evaluate_task_kinematics_5d(robot, candidate.q, characteristic_length=characteristic_length).sigma_min_5 >= sigma_safe
        ))
    return NUCIKCatalog(
        positions, axes, tuple(layers), perf_counter() - start,
        {"random_restarts": random_restarts, "max_candidates": max_candidates,
         "orientation_cone_samples": orientation_cone_samples, "sigma_safe": sigma_safe,
         "characteristic_length": characteristic_length, "surface_positions": projection.points},
    )


def lift_ordered_layout(
    robot: UR5eKinematics,
    reference: SurfaceInstance,
    layout: dict[str, Any],
    catalog: NUCIKCatalog,
    transform: np.ndarray,
    transition_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    *,
    axis_tolerance: float,
    characteristic_length: float,
    sigma_safe: float,
    task_edge_samples: int,
    surface_path_spacing: float,
    maximum_joint_step: float,
    position_tolerance: float,
    max_active_branches: int,
) -> NUCLiftResult:
    proxy = NUCSkeleton(
        topological_path=np.asarray(layout["topological_path"], dtype=np.int64),
        waypoints=np.asarray(layout["ordered_physical_path"], dtype=np.float64),
        visited_faces=np.empty(0, dtype=np.int64), tree_edges=np.empty((0, 3), dtype=np.int64),
        policy="upstream_first", seed=None, expansion_decisions=(), root_face=int(layout["mapped_face"]),
        metadata={"ordered_path_adapter": True},
    )
    return minimum_cost_nuc_lift(
        robot, reference, proxy, catalog, transform, transition_cache,
        axis_tolerance=axis_tolerance, characteristic_length=characteristic_length,
        sigma_safe=sigma_safe, task_edge_samples=task_edge_samples,
        surface_path_spacing=surface_path_spacing, maximum_joint_step=maximum_joint_step,
        position_tolerance=position_tolerance, max_active_branches=max_active_branches,
    )


def aligned_ordered_path(reference: SurfaceInstance, points: np.ndarray, count: int) -> np.ndarray:
    dense = resample_projected_polyline(reference, points, max_spacing=0.002)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-14))
    dense, cumulative = dense[keep], cumulative[keep]
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([np.interp(targets, cumulative, dense[:, axis]) for axis in range(3)])


def pairwise_layout_distance(
    reference: SurfaceInstance,
    first: dict[str, Any],
    second: dict[str, Any],
    count: int,
) -> dict[str, float]:
    a = aligned_ordered_path(reference, np.asarray(first["ordered_physical_path"]), count)
    b = aligned_ordered_path(reference, np.asarray(second["ordered_physical_path"]), count)
    ta = np.diff(a, axis=0); tb = np.diff(b, axis=0)
    ta /= np.maximum(np.linalg.norm(ta, axis=1, keepdims=True), 1e-15)
    tb /= np.maximum(np.linalg.norm(tb, axis=1, keepdims=True), 1e-15)
    angles = np.arccos(np.clip(np.sum(ta * tb, axis=1), -1.0, 1.0))
    return {
        "ordered_path_distance_m": float(np.mean(np.linalg.norm(a - b, axis=1))),
        "tangent_disagreement_rad": float(np.mean(angles)),
    }


def _grid_faces_pattern(nx: int, ny: int, pattern: str) -> np.ndarray:
    faces = []
    width = nx + 1
    for y in range(ny):
        for x in range(nx):
            a = y * width + x; b = a + 1; c = (y + 1) * width + x; d = c + 1
            opposite = pattern == "opposite" or (pattern == "checker" and (x + y) % 2 == 1) or (pattern == "x_stripe" and x % 2 == 1)
            faces.extend(([a, b, c], [b, d, c]) if opposite else ([a, b, d], [a, d, c]))
    return np.asarray(faces, dtype=np.int64)


def _hemisphere_mesh(radius: float, n_azimuth: int, n_polar: int, *, azimuth_phase: float, diagonal_pattern: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = [[0.0, 0.0, radius]]
    phase = azimuth_phase * 2.0 * np.pi / n_azimuth
    for polar_index in range(1, n_polar + 1):
        polar = 0.5 * np.pi * polar_index / n_polar
        for azimuth_index in range(n_azimuth):
            azimuth = 2.0 * np.pi * azimuth_index / n_azimuth + phase
            vertices.append([radius * np.sin(polar) * np.cos(azimuth), radius * np.sin(polar) * np.sin(azimuth), radius * np.cos(polar)])
    faces = []
    for azimuth in range(n_azimuth):
        faces.append([0, 1 + azimuth, 1 + (azimuth + 1) % n_azimuth])
    for polar in range(n_polar - 1):
        start = 1 + polar * n_azimuth; following_start = start + n_azimuth
        for azimuth in range(n_azimuth):
            nxt = (azimuth + 1) % n_azimuth
            a, b = start + azimuth, start + nxt
            c, d = following_start + azimuth, following_start + nxt
            opposite = diagonal_pattern == "opposite" or (diagonal_pattern == "checker" and (polar + azimuth) % 2 == 1)
            faces.extend(([a, c, b], [b, c, d]) if opposite else ([a, c, d], [a, d, b]))
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    return vertices_array, _orient_faces(vertices_array, faces_array, "hemisphere", None)


def _orient_faces(vertices: np.ndarray, faces: np.ndarray, surface_id: str, curvature: float | None) -> np.ndarray:
    result = faces.copy()
    for index, face in enumerate(result):
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        center = triangle.mean(axis=0)
        expected = center if surface_id == "hemisphere" else np.asarray([-2 * curvature * center[0], 2 * curvature * center[1], 1.0])
        if np.dot(normal, expected) < 0:
            result[index, [1, 2]] = result[index, [2, 1]]
    return result


def _boundary_vertices(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])
