from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from diffusion_coverage.coverage.evaluator import _length_and_sources
from diffusion_coverage.nuc.adapter import NUCSkeleton, generate_nuc_skeleton
from diffusion_coverage.robot.ur5e_mujoco import interpolate_vertex_normals, transform_surface_pose_path
from diffusion_coverage.surface import make_hemisphere, make_saddle
from diffusion_coverage.surface.projection import project_points


def load_e06_contract(root: Path) -> tuple[dict, dict, dict]:
    result = root / "results/nuc_robot_skeleton_coupling_v1"
    archived = json.loads((result / "config.json").read_text())
    rows = [json.loads(line) for line in (result / "candidate_results.jsonl").read_text().splitlines() if line]
    scenes = [json.loads(line) for line in (result / "scene_results.jsonl").read_text().splitlines() if line]
    return archived, rows, scenes


def make_e06_surface(config: dict, surface_id: str, samples_per_face: int):
    maker = make_saddle if surface_id == "saddle" else make_hemisphere
    return maker(**config["surfaces"][surface_id], samples_per_face=samples_per_face)


def regenerate_e06_variants(surface, count: int, seed: int, refinement_iterations: int) -> tuple[NUCSkeleton, ...]:
    variants: list[NUCSkeleton] = []
    fingerprints: set[bytes] = set()
    jobs: list[tuple[str, int | None]] = [("upstream_first", None), ("reverse_order", None)]
    next_seed = seed
    random_attempts = 0
    while len(variants) < count:
        if jobs:
            policy, value = jobs.pop(0)
        else:
            policy = "seeded_random" if random_attempts < 1000 else "frontier_random"
            value = next_seed
            next_seed += 1
            random_attempts += 1
        skeleton = generate_nuc_skeleton(surface, policy=policy, seed=value)
        fingerprint = skeleton.topological_path.tobytes()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        points = skeleton.waypoints.copy()
        for _ in range(refinement_iterations):
            points = project_points(surface, points).points
        variants.append(replace(skeleton, waypoints=points))
    return tuple(variants)


def canonical_task_poses(surface, reference: NUCSkeleton, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = 3 * surface.num_faces
    points = np.empty((count, 3), dtype=np.float64)
    points[reference.topological_path] = reference.waypoints
    projection = project_points(surface, points)
    normals = interpolate_vertex_normals(
        surface.vertices, surface.faces, surface.face_normals, surface.face_areas,
        projection.face_indices, projection.barycentric,
    )
    return transform_surface_pose_path(points, normals, transform)


def dense_geodesic_path(surface, skeleton: NUCSkeleton, spacing: float) -> np.ndarray:
    projection = project_points(surface, skeleton.waypoints)
    return _length_and_sources(surface, projection, max_spacing=spacing)[1]


def reconstructed_transition_poses(
    surface,
    skeleton: NUCSkeleton,
    canonical_positions: np.ndarray,
    transform: np.ndarray,
    spacing: float,
    task_edge_samples: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    from diffusion_coverage.nuc.robot_lift import NUCIKCatalog, _projected_edge

    _, axes = canonical_task_poses(surface, skeleton, transform)
    catalog = NUCIKCatalog(canonical_positions, axes, tuple(() for _ in range(len(axes))), 0.0)
    result = []
    for source, target in zip(skeleton.topological_path[:-1], skeleton.topological_path[1:]):
        chord = float(np.linalg.norm(canonical_positions[source] - canonical_positions[target]))
        edge_spacing = min(spacing, chord / max(task_edge_samples - 1, 1))
        result.append(_projected_edge(surface, int(source), int(target), catalog, transform, edge_spacing))
    return tuple(result)
