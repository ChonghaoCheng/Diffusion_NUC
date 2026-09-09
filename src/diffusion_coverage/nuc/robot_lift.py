from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from diffusion_coverage.nuc.adapter import NUCSkeleton
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
    transition_cache: dict[tuple[int, int], dict[tuple[int, int], NUCTransitionWitness]],
    *,
    axis_tolerance: float,
    characteristic_length: float,
    sigma_safe: float,
    task_edge_samples: int,
    maximum_joint_step: float,
    position_tolerance: float,
    max_target_matches: int = 3,
) -> NUCLiftResult:
    """Minimize witness L_q over the enumerated layered candidate graph."""

    codes = np.asarray(skeleton.topological_path, dtype=np.int64)
    if len(codes) < 2 or not catalog.candidates[int(codes[0])]:
        return _failed_lift(0, 0.0)
    costs = np.zeros(len(catalog.candidates[int(codes[0])]), dtype=np.float64)
    parents: list[np.ndarray] = []
    selected_by_target: list[list[NUCTransitionWitness | None]] = []
    evaluated = 0
    elapsed = 0.0
    for source_code, target_code in zip(codes[:-1], codes[1:]):
        key = (int(source_code), int(target_code))
        if key not in transition_cache:
            start = perf_counter()
            transition_cache[key], edge_evaluated = _enumerate_transition(
                robot, surface, catalog, *key, transform_base_from_surface,
                axis_tolerance=axis_tolerance,
                characteristic_length=characteristic_length,
                sigma_safe=sigma_safe,
                task_edge_samples=task_edge_samples,
                maximum_joint_step=maximum_joint_step,
                position_tolerance=position_tolerance,
                max_target_matches=max_target_matches,
            )
            elapsed += perf_counter() - start
            evaluated += edge_evaluated
        mapping = transition_cache[key]
        target_count = len(catalog.candidates[int(target_code)])
        next_costs = np.full(target_count, np.inf)
        next_parents = np.full(target_count, -1, dtype=np.int64)
        next_selected: list[NUCTransitionWitness | None] = [None] * target_count
        for (source_index, target_index), witness in mapping.items():
            value = costs[source_index] + witness.joint_length
            if value < next_costs[target_index]:
                next_costs[target_index] = value
                next_parents[target_index] = source_index
                next_selected[target_index] = witness
        if not np.any(np.isfinite(next_costs)):
            return _failed_lift(
                evaluated, elapsed,
                failed_transition=[int(source_code), int(target_code)],
            )
        costs = next_costs
        parents.append(next_parents)
        selected_by_target.append(next_selected)
    target_index = int(np.argmin(costs))
    selected = []
    for layer in range(len(parents) - 1, -1, -1):
        witness = selected_by_target[layer][target_index]
        if witness is None:
            raise RuntimeError("finite DP cost has no continuation witness")
        selected.append(witness)
        target_index = int(parents[layer][target_index])
    selected.reverse()
    q_path = np.concatenate([selected[0].q] + [item.q[1:] for item in selected[1:]])
    positions = np.concatenate(
        [selected[0].desired_positions] + [item.desired_positions[1:] for item in selected[1:]]
    )
    axes = np.concatenate(
        [selected[0].desired_axes] + [item.desired_axes[1:] for item in selected[1:]]
    )
    joint_length = compute_joint_execution_cost((q_path,)).weighted_joint_length
    return NUCLiftResult(True, q_path, positions, axes, joint_length, None, evaluated, elapsed)


def _failed_lift(evaluated: int, elapsed: float, **metadata: Any) -> NUCLiftResult:
    return NUCLiftResult(
        False, None, None, None, None,
        "continuous_lift_not_found_under_budget", evaluated, elapsed, metadata,
    )


def _enumerate_transition(
    robot: UR5eKinematics,
    surface: SurfaceInstance,
    catalog: NUCIKCatalog,
    source_code: int,
    target_code: int,
    transform: np.ndarray,
    *,
    axis_tolerance: float,
    characteristic_length: float,
    sigma_safe: float,
    task_edge_samples: int,
    maximum_joint_step: float,
    position_tolerance: float,
    max_target_matches: int,
) -> tuple[dict[tuple[int, int], NUCTransitionWitness], int]:
    positions, axes = _projected_edge(
        surface, source_code, target_code, catalog, transform, task_edge_samples
    )
    result = {}
    evaluated = 0
    targets = catalog.candidates[target_code]
    for source_index, source in enumerate(catalog.candidates[source_code]):
        prefix = robot.continue_task_transition(
            source.q, positions[:-1], axes[:-1], maximum_joint_step=maximum_joint_step,
            minimum_manipulability=0.0, position_tolerance=position_tolerance,
            axis_tolerance=axis_tolerance,
        )
        evaluated += 1
        if not prefix.feasible:
            continue
        distances = np.asarray([
            np.max(np.abs(target.q - prefix.q_path[-1])) for target in targets
        ])
        for target_index in np.argsort(distances, kind="stable")[:max_target_matches]:
            evaluated += 1
            fractions = np.linspace(0.0, 1.0, 3)
            final_positions = (1.0 - fractions[:, None]) * positions[-2] + fractions[:, None] * positions[-1]
            final_axes = (1.0 - fractions[:, None]) * axes[-2] + fractions[:, None] * axes[-1]
            final_axes /= np.linalg.norm(final_axes, axis=1, keepdims=True)
            final = robot.check_task_transition(
                prefix.q_path[-1], targets[int(target_index)].q,
                final_positions, final_axes,
                maximum_joint_step=maximum_joint_step, minimum_manipulability=0.0,
                position_tolerance=position_tolerance, axis_tolerance=axis_tolerance,
                allow_equivalent_end=False, check_endpoints=False,
            )
            if not final.feasible:
                continue
            q = np.concatenate((prefix.q_path, final.q_path[1:]))
            desired_positions = np.concatenate((positions[:-1], final_positions[1:]))
            desired_axes = np.concatenate((axes[:-1], final_axes[1:]))
            if any(
                evaluate_task_kinematics_5d(
                    robot, value, characteristic_length=characteristic_length
                ).sigma_min_5 < sigma_safe
                for value in q
            ):
                continue
            result[(source_index, int(target_index))] = NUCTransitionWitness(
                q=q.astype(np.float64, copy=False),
                desired_positions=desired_positions,
                desired_axes=desired_axes,
                joint_length=compute_joint_execution_cost((q,)).weighted_joint_length,
            )
    return result, evaluated


def _projected_edge(
    surface: SurfaceInstance,
    source_code: int,
    target_code: int,
    catalog: NUCIKCatalog,
    transform: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    endpoints_surface = (catalog.positions[[source_code, target_code]] - translation) @ rotation
    fractions = np.linspace(0.0, 1.0, samples)[:, None]
    raw = (1.0 - fractions) * endpoints_surface[0] + fractions * endpoints_surface[1]
    projection = project_points(surface, raw)
    normals = interpolate_vertex_normals(
        surface.vertices, surface.faces, surface.face_normals, surface.face_areas,
        projection.face_indices, projection.barycentric,
    )
    return transform_surface_pose_path(projection.points, normals, transform)
