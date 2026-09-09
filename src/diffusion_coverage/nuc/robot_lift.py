from __future__ import annotations

import copy
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
    raw_candidate_counts: tuple[int, ...] = ()
    collision_safe_candidate_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class NUCContinuationLayerTrace:
    pose_index: int
    source_code: int | None
    target_code: int
    normalized_progress: float
    candidate_count_before_safety: int | None
    candidate_count_after_joint_limits: int | None
    candidate_count_after_collision: int | None
    candidate_count_after_sigma: int
    propagated_incoming_edges: int
    valid_outgoing_edges: int
    beam_width_before_pruning: int
    beam_width_after_pruning: int
    minimum_sigma_min_5: float | None
    maximum_sigma_min_5: float | None
    minimum_joint_limit_margin: float | None
    surface_location: tuple[float, float, float]
    desired_tool_axis: tuple[float, float, float]


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
    collect_diagnostics: bool = False,
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
    raw_counts: list[int] = []
    collision_safe_counts: list[int] = []
    start = perf_counter()
    for position, axis in zip(positions, axes):
        raw_rng = None
        if collect_diagnostics:
            raw_rng = np.random.default_rng()
            raw_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        collision_safe = robot.enumerate_ik(
            position, axis, random_restarts=random_restarts, rng=rng,
            axis_tolerance=axis_tolerance, minimum_manipulability=0.0,
            max_candidates=max_candidates, orientation_cone_samples=orientation_cone_samples,
        )
        if raw_rng is not None:
            raw = robot.enumerate_ik(
                position, axis, random_restarts=random_restarts, rng=raw_rng,
                axis_tolerance=axis_tolerance, minimum_manipulability=0.0,
                require_collision_free=False, max_candidates=max_candidates,
                orientation_cone_samples=orientation_cone_samples,
            )
            raw_counts.append(len(raw))
            collision_safe_counts.append(len(collision_safe))
        layers.append(tuple(
            candidate for candidate in collision_safe
            if evaluate_task_kinematics_5d(
                robot, candidate.q, characteristic_length=characteristic_length
            ).sigma_min_5 >= sigma_safe
        ))
    return NUCIKCatalog(
        positions, axes, tuple(layers), perf_counter() - start,
        {"random_restarts": random_restarts, "max_candidates": max_candidates,
         "orientation_cone_samples": orientation_cone_samples, "sigma_safe": sigma_safe,
         "characteristic_length": characteristic_length,
         "surface_positions": points,
         "collect_diagnostics": collect_diagnostics},
        tuple(raw_counts), tuple(collision_safe_counts),
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
    collect_trace: bool = False,
) -> NUCLiftResult:
    """Minimize witness L_q over a layered graph of propagated IK branches.

    Independently enumerated endpoint IK samples are deliberately not forced as
    transition endpoints: doing so creates numerical disconnections when the valid
    continuation lands between endpoint samples. Every admitted initial branch is
    propagated, deduplicated, and retained under one fixed beam budget.
    """

    codes = np.asarray(skeleton.topological_path, dtype=np.int64)
    traces: list[NUCContinuationLayerTrace] = []
    if len(codes) < 2 or not catalog.candidates[int(codes[0])]:
        code = int(codes[0]) if len(codes) else -1
        if collect_trace and len(codes):
            traces.append(_catalog_layer_trace(catalog, code, 0, len(codes), None, 0, 0, 0, 0, robot))
        return _failed_lift(0, 0.0, failed_pose_index=0, layer_trace=traces)
    states = [
        (candidate.q.copy(), 0.0, [])
        for candidate in catalog.candidates[int(codes[0])]
    ][:max_active_branches]
    evaluated = 0
    elapsed = 0.0
    for transition_index, (source_code, target_code) in enumerate(zip(codes[:-1], codes[1:])):
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
        retained = _deduplicate_states(next_states, max_active_branches)
        if collect_trace:
            traces.append(_catalog_layer_trace(
                catalog, int(target_code), transition_index + 1, len(codes), int(source_code),
                len(states), len(next_states), len(next_states), len(retained), robot,
                propagated_states=retained,
            ))
        states = retained
        if not states:
            return _failed_lift(
                evaluated, elapsed,
                failed_transition=[int(source_code), int(target_code)],
                failed_pose_index=transition_index + 1,
                layer_trace=traces,
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
    return NUCLiftResult(
        True, q_path, positions, axes, joint_length, None, evaluated, elapsed,
        {"layer_trace": traces} if collect_trace else {},
    )


def _catalog_layer_trace(
    catalog: NUCIKCatalog,
    target_code: int,
    pose_index: int,
    pose_count: int,
    source_code: int | None,
    incoming: int,
    outgoing: int,
    beam_before: int,
    beam_after: int,
    robot: UR5eKinematics,
    *,
    propagated_states: list[tuple[np.ndarray, float, list[NUCTransitionWitness]]] | None = None,
) -> NUCContinuationLayerTrace:
    safe = catalog.candidates[target_code]
    raw = None if not catalog.raw_candidate_counts else catalog.raw_candidate_counts[target_code]
    collision = None if not catalog.collision_safe_candidate_counts else catalog.collision_safe_candidate_counts[target_code]
    q_values = [state[0] for state in propagated_states] if propagated_states else [item.q for item in safe]
    task = [evaluate_task_kinematics_5d(robot, q, characteristic_length=float(catalog.metadata["characteristic_length"])) for q in q_values]
    configurations = [robot.evaluate_configuration(q) for q in q_values]
    return NUCContinuationLayerTrace(
        pose_index=pose_index,
        source_code=source_code,
        target_code=target_code,
        normalized_progress=pose_index / max(pose_count - 1, 1),
        candidate_count_before_safety=raw,
        candidate_count_after_joint_limits=raw,
        candidate_count_after_collision=collision,
        candidate_count_after_sigma=len(safe),
        propagated_incoming_edges=incoming,
        valid_outgoing_edges=outgoing,
        beam_width_before_pruning=beam_before,
        beam_width_after_pruning=beam_after,
        minimum_sigma_min_5=None if not task else min(item.sigma_min_5 for item in task),
        maximum_sigma_min_5=None if not task else max(item.sigma_min_5 for item in task),
        minimum_joint_limit_margin=None if not configurations else min(item.joint_limit_margin for item in configurations),
        surface_location=tuple(float(value) for value in catalog.metadata["surface_positions"][target_code]),
        desired_tool_axis=tuple(float(value) for value in catalog.axes[target_code]),
    )


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
