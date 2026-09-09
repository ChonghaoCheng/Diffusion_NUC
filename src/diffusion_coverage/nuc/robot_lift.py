from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from diffusion_coverage.nuc.adapter import NUCSkeleton
from diffusion_coverage.coverage.evaluator import _length_and_sources
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import (
    IKCandidate,
    UR5eKinematics,
    interpolate_vertex_normals,
    transform_surface_pose_path,
)
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class NUCTransitionWitness:
    q: np.ndarray
    desired_positions: np.ndarray
    desired_axes: np.ndarray
    joint_length: float


@dataclass(frozen=True)
class NUCIKCatalog:
    positions: np.ndarray
    axes: np.ndarray
    candidates: tuple[tuple[IKCandidate, ...], ...]
    enumeration_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NUCLiftResult:
    found: bool
    q_path: np.ndarray | None
    desired_positions: np.ndarray | None
    desired_axes: np.ndarray | None
    joint_length: float | None
    failure_reason: str | None
    evaluated_transitions: int
    continuation_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


def build_nuc_ik_catalog(
    robot: UR5eKinematics,
    surface: SurfaceInstance,
    reference_skeleton: NUCSkeleton,
    transform_base_from_surface: np.ndarray,
    *,
    axis_tolerance: float,
    characteristic_length: float,
    sigma_safe: float,
    random_restarts: int,
    max_candidates: int,
    orientation_cone_samples: int,
    seed: int,
) -> NUCIKCatalog:
    """Enumerate one shared safe IK layer per NUC subfacet code."""

    code_count = 3 * surface.num_faces
    points = np.empty((code_count, 3), dtype=np.float64)
    points[reference_skeleton.topological_path] = reference_skeleton.waypoints
    projection = project_points(surface, points)
    normals = interpolate_vertex_normals(
        surface.vertices, surface.faces, surface.face_normals, surface.face_areas,
        projection.face_indices, projection.barycentric,
    )
    positions, axes = transform_surface_pose_path(points, normals, transform_base_from_surface)
    rng = np.random.default_rng(seed)
    layers = []
    start = perf_counter()
    for position, axis in zip(positions, axes):
        raw = robot.enumerate_ik(
            position, axis, random_restarts=random_restarts, rng=rng,
            axis_tolerance=axis_tolerance, minimum_manipulability=0.0,
            max_candidates=max_candidates, orientation_cone_samples=orientation_cone_samples,
        )
        layers.append(tuple(
            candidate for candidate in raw
            if evaluate_task_kinematics_5d(
                robot, candidate.q, characteristic_length=characteristic_length
            ).sigma_min_5 >= sigma_safe
        ))
    return NUCIKCatalog(
        positions, axes, tuple(layers), perf_counter() - start,
        {"random_restarts": random_restarts, "max_candidates": max_candidates,
         "orientation_cone_samples": orientation_cone_samples, "sigma_safe": sigma_safe},
    )


def minimum_cost_nuc_lift(
    robot: UR5eKinematics,
    surface: SurfaceInstance,
    skeleton: NUCSkeleton,
    catalog: NUCIKCatalog,
    transform_base_from_surface: np.ndarray,
    transition_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    *,
    axis_tolerance: float,
    characteristic_length: float,
    sigma_safe: float,
    task_edge_samples: int,
    surface_path_spacing: float,
    maximum_joint_step: float,
    position_tolerance: float,
    max_target_matches: int = 3,
    max_active_branches: int = 6,
) -> NUCLiftResult:
    """Minimize witness L_q over a layered graph of propagated IK branches.

    Independently enumerated endpoint IK samples are deliberately not forced as
    transition endpoints: doing so creates numerical disconnections when the valid
    continuation lands between endpoint samples. Every admitted initial branch is
    propagated, deduplicated, and retained under one fixed beam budget.
    """

    codes = np.asarray(skeleton.topological_path, dtype=np.int64)
    if len(codes) < 2 or not catalog.candidates[int(codes[0])]:
        return _failed_lift(0, 0.0)
    states = [
        (candidate.q.copy(), 0.0, [])
        for candidate in catalog.candidates[int(codes[0])]
    ][:max_active_branches]
    evaluated = 0
    elapsed = 0.0
    for source_code, target_code in zip(codes[:-1], codes[1:]):
        key = (int(source_code), int(target_code))
        if key not in transition_cache:
            chord = float(np.linalg.norm(catalog.positions[int(source_code)] - catalog.positions[int(target_code)]))
            edge_spacing = min(
                surface_path_spacing,
                chord / max(task_edge_samples - 1, 1),
            )
            transition_cache[key] = _projected_edge(
                surface, *key, catalog, transform_base_from_surface, edge_spacing
            )
        positions, axes = transition_cache[key]
        next_states = []
        start = perf_counter()
        for source_q, source_cost, history in states:
            transition = robot.continue_task_transition(
                source_q,
                positions,
                axes,
                maximum_joint_step=maximum_joint_step,
                minimum_manipulability=0.0,
                position_tolerance=position_tolerance,
                axis_tolerance=axis_tolerance,
            )
            evaluated += 1
            if not transition.feasible:
                continue
            if any(
                evaluate_task_kinematics_5d(
                    robot, value, characteristic_length=characteristic_length
                ).sigma_min_5 < sigma_safe
                for value in transition.q_path
            ):
                continue
            witness = NUCTransitionWitness(
                q=transition.q_path.astype(np.float64, copy=False),
                desired_positions=positions,
                desired_axes=axes,
                joint_length=compute_joint_execution_cost(
                    (transition.q_path,)
                ).weighted_joint_length,
            )
            next_states.append(
                (transition.q_path[-1].copy(), source_cost + witness.joint_length, history + [witness])
            )
        elapsed += perf_counter() - start
        states = _deduplicate_states(next_states, max_active_branches)
        if not states:
            return _failed_lift(
                evaluated, elapsed,
                failed_transition=[int(source_code), int(target_code)],
            )
    _, _, selected = min(states, key=lambda item: item[1])
    q_path = np.concatenate([selected[0].q] + [item.q[1:] for item in selected[1:]])
    positions = np.concatenate(
        [selected[0].desired_positions] + [item.desired_positions[1:] for item in selected[1:]]
    )
    axes = np.concatenate(
        [selected[0].desired_axes] + [item.desired_axes[1:] for item in selected[1:]]
    )
    joint_length = compute_joint_execution_cost((q_path,)).weighted_joint_length
    return NUCLiftResult(True, q_path, positions, axes, joint_length, None, evaluated, elapsed)


def _deduplicate_states(
    states: list[tuple[np.ndarray, float, list[NUCTransitionWitness]]],
    maximum: int,
) -> list[tuple[np.ndarray, float, list[NUCTransitionWitness]]]:
    retained = []
    for state in sorted(states, key=lambda item: item[1]):
        if any(np.linalg.norm(state[0] - other[0]) < 5e-2 for other in retained):
            continue
        retained.append(state)
        if len(retained) == maximum:
            break
    return retained


def _failed_lift(evaluated: int, elapsed: float, **metadata: Any) -> NUCLiftResult:
    return NUCLiftResult(
        False, None, None, None, None,
        "continuous_lift_not_found_under_budget", evaluated, elapsed, metadata,
    )


def _projected_edge(
    surface: SurfaceInstance,
    source_code: int,
    target_code: int,
    catalog: NUCIKCatalog,
    transform: np.ndarray,
    max_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    endpoints_surface = (catalog.positions[[source_code, target_code]] - translation) @ rotation
    endpoint_projection = project_points(surface, endpoints_surface)
    _, path = _length_and_sources(surface, endpoint_projection, max_spacing=max_spacing)
    projection = project_points(surface, path)
    normals = interpolate_vertex_normals(
        surface.vertices, surface.faces, surface.face_normals, surface.face_areas,
        projection.face_indices, projection.barycentric,
    )
    return transform_surface_pose_path(projection.points, normals, transform)
