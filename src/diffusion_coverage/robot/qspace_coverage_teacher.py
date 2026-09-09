from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any

import numpy as np

from diffusion_coverage.coverage import CoverageMetrics, CoveragePlan, evaluate_coverage
from diffusion_coverage.robot.surface_ik_graph import SurfaceIKGraph, surface_edge_target_poses
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.surface import SurfaceInstance


@dataclass(frozen=True)
class QSpaceCoverageTeacherResult:
    feasible: bool
    q_segments: tuple[np.ndarray, ...]
    target_position_segments: tuple[np.ndarray, ...]
    target_axis_segments: tuple[np.ndarray, ...]
    node_routes: tuple[np.ndarray, ...]
    selected_components: np.ndarray
    covered_node_mask: np.ndarray
    joint_travel: float
    squared_joint_motion: float
    task_path_length: float
    repeated_node_visits: int
    coverage_metrics: CoverageMetrics | None
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QSpacePlanCheck:
    feasible: bool
    failure_reason: str | None
    max_position_error: float
    max_axis_error: float
    minimum_manipulability: float
    minimum_joint_limit_margin: float


def resample_qspace_segment(
    q_values: np.ndarray,
    target_positions: np.ndarray,
    target_axes: np.ndarray,
    *,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one continuous segment uniformly in cumulative joint travel."""

    q = np.asarray(q_values, dtype=np.float64)
    positions = np.asarray(target_positions, dtype=np.float64)
    axes = np.asarray(target_axes, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6 or positions.shape != axes.shape:
        raise ValueError("invalid q-space segment shapes")
    if positions.shape != (len(q), 3) or len(q) < 2:
        raise ValueError("segments require aligned q, position, and axis samples")
    if num_tokens < 2:
        raise ValueError("num_tokens must be at least two")
    travel = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(travel) > 1e-12))
    travel = travel[keep]
    q = q[keep]
    positions = positions[keep]
    axes = axes[keep]
    if len(travel) == 1:
        return (
            np.repeat(q, 2, axis=0),
            np.repeat(positions, 2, axis=0),
            np.repeat(axes, 2, axis=0),
        )
    count = min(num_tokens, len(travel))
    targets = np.linspace(0.0, travel[-1], count)
    q_resampled = np.column_stack(
        [np.interp(targets, travel, q[:, dimension]) for dimension in range(6)]
    )
    position_resampled = np.column_stack(
        [np.interp(targets, travel, positions[:, dimension]) for dimension in range(3)]
    )
    axis_resampled = np.column_stack(
        [np.interp(targets, travel, axes[:, dimension]) for dimension in range(3)]
    )
    axis_resampled /= np.maximum(
        np.linalg.norm(axis_resampled, axis=1, keepdims=True), 1e-12
    )
    return q_resampled, position_resampled, axis_resampled


def simplify_qspace_segment(
    q_values: np.ndarray,
    target_positions: np.ndarray,
    target_axes: np.ndarray,
    *,
    maximum_joint_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep original samples needed to approximate q along joint arclength."""

    q = np.asarray(q_values, dtype=np.float64)
    positions = np.asarray(target_positions, dtype=np.float64)
    axes = np.asarray(target_axes, dtype=np.float64)
    if positions.shape != axes.shape or positions.shape != (len(q), 3):
        raise ValueError("invalid q-space segment shapes")
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2:
        raise ValueError("q_values must have shape [M, 6] with M >= 2")
    if maximum_joint_error <= 0.0:
        raise ValueError("maximum_joint_error must be positive")
    travel = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))))
    selected = {0, len(q) - 1}
    stack = [(0, len(q) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1 or travel[end] <= travel[start] + 1e-12:
            continue
        alpha = ((travel[start + 1 : end] - travel[start]) / (travel[end] - travel[start]))[:, None]
        interpolated = (1.0 - alpha) * q[start] + alpha * q[end]
        errors = np.linalg.norm(q[start + 1 : end] - interpolated, axis=1)
        worst_local = int(np.argmax(errors))
        if errors[worst_local] > maximum_joint_error:
            split = start + 1 + worst_local
            selected.add(split)
            stack.extend(((start, split), (split, end)))
    indices = np.asarray(sorted(selected), dtype=np.int64)
    return q[indices], positions[indices], axes[indices]


def densify_qspace_segment(
    q_values: np.ndarray,
    target_positions: np.ndarray,
    target_axes: np.ndarray,
    *,
    maximum_joint_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate aligned task and joint samples for continuous-path checking."""

    q = np.asarray(q_values, dtype=np.float64)
    positions = np.asarray(target_positions, dtype=np.float64)
    axes = np.asarray(target_axes, dtype=np.float64)
    if positions.shape != axes.shape or positions.shape != (len(q), 3):
        raise ValueError("invalid q-space segment shapes")
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2:
        raise ValueError("q_values must have shape [M, 6] with M >= 2")
    if maximum_joint_step <= 0.0:
        raise ValueError("maximum_joint_step must be positive")
    q_result = [q[0]]
    position_result = [positions[0]]
    axis_result = [axes[0]]
    for start, end, start_position, end_position, start_axis, end_axis in zip(
        q[:-1], q[1:], positions[:-1], positions[1:], axes[:-1], axes[1:]
    ):
        intervals = max(1, int(np.ceil(np.max(np.abs(end - start)) / maximum_joint_step)))
        for fraction in np.linspace(0.0, 1.0, intervals + 1)[1:]:
            q_result.append((1.0 - fraction) * start + fraction * end)
            position_result.append((1.0 - fraction) * start_position + fraction * end_position)
            axis = (1.0 - fraction) * start_axis + fraction * end_axis
            axis_result.append(axis / max(np.linalg.norm(axis), 1e-12))
    return np.asarray(q_result), np.asarray(position_result), np.asarray(axis_result)


def select_components_marginal(
    graph: SurfaceIKGraph, max_segments: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    coverage = component_node_coverage(graph)
    covered = np.zeros(graph.num_nodes, dtype=bool)
    selected = []
    for _ in range(min(max_segments, graph.num_components)):
        gains = np.sum(coverage & ~covered[:, None], axis=0)
        if selected:
            gains[np.asarray(selected)] = -1
        choice = int(np.argmax(gains))
        if gains[choice] <= 0:
            break
        selected.append(choice)
        covered |= coverage[:, choice]
    return np.asarray(selected, dtype=np.int64), covered


def component_node_coverage(graph: SurfaceIKGraph) -> np.ndarray:
    coverage = np.zeros((graph.num_nodes, graph.num_components), dtype=bool)
    for node, labels in enumerate(graph.component_labels):
        if len(labels):
            coverage[node, np.unique(labels)] = True
    return coverage


def plan_candidate_routes(
    graph: SurfaceIKGraph,
    selected_components: np.ndarray,
    *,
    route_objective: str = "surface_then_joint",
) -> tuple[np.ndarray, ...]:
    if route_objective not in {"surface_then_joint", "joint_then_surface"}:
        raise ValueError("unknown route objective")
    selected = np.asarray(selected_components, dtype=np.int64)
    coverage = component_node_coverage(graph)
    assigned = np.zeros(graph.num_nodes, dtype=bool)
    flat = _flatten_graph(graph)
    routes = []
    for component in selected:
        targets = coverage[:, component] & ~assigned
        if not targets.any():
            continue
        route = _route_component_targets(
            graph, flat, int(component), targets, route_objective
        )
        routes.append(route)
        assigned[np.unique(flat["node"][route])] = True
    return tuple(routes)


def build_qspace_coverage_teacher(
    robot: UR5eKinematics,
    surface: SurfaceInstance,
    transform_base_from_surface: np.ndarray,
    graph: SurfaceIKGraph,
    *,
    max_segments: int,
    footprint_radius: float,
    route_objective: str = "surface_then_joint",
    hard_position_tolerance: float | None = None,
) -> QSpaceCoverageTeacherResult:
    selected, covered = select_components_marginal(graph, max_segments)
    candidate_routes = plan_candidate_routes(
        graph, selected, route_objective=route_objective
    )
    flat = _flatten_graph(graph)
    edge_lookup = _surface_edge_lookup(graph)
    q_segments = []
    position_segments = []
    axis_segments = []
    node_routes = []
    for route in candidate_routes:
        start_flat = int(route[0])
        start_node = int(flat["node"][start_flat])
        start_local = int(flat["local"][start_flat])
        q_values = [graph.candidates[start_node][start_local].q]
        target_positions = [graph.positions[start_node]]
        target_axes = [graph.axes[start_node]]
        route_nodes = [start_node]
        for source_flat, target_flat in zip(route[:-1], route[1:]):
            source_node = int(flat["node"][source_flat])
            source_local = int(flat["local"][source_flat])
            target_node = int(flat["node"][target_flat])
            target_local = int(flat["local"][target_flat])
            edge_id, reverse_edge = edge_lookup[(source_node, target_node)]
            edge_positions, edge_axes = surface_edge_target_poses(
                surface,
                transform_base_from_surface,
                graph.uv[source_node],
                graph.uv[target_node],
                samples=int(graph.metadata["task_edge_samples"]),
                periodic_u=bool(graph.metadata["periodic_u"]),
            )
            target_q = graph.candidates[target_node][target_local].q
            witness_key = (
                (target_local, source_local)
                if reverse_edge
                else (source_local, target_local)
            )
            witness = graph.edge_witnesses[edge_id].get(witness_key)
            edge_target_positions, edge_target_axes = _target_sequence_for_transition(
                edge_positions, edge_axes
            )
            transition = None
            transition_q = None if witness is None else (
                witness[::-1].copy() if reverse_edge else witness.copy()
            )
            if transition_q is None:
                transition = robot.continue_task_transition_to_configuration(
                    q_values[-1],
                    target_q,
                    edge_positions,
                    edge_axes,
                    maximum_joint_step=float(graph.metadata["maximum_joint_step"]),
                    minimum_manipulability=float(graph.metadata["minimum_manipulability"]),
                    position_tolerance=float(graph.metadata["task_position_tolerance"]),
                    axis_tolerance=np.deg2rad(float(graph.metadata["axis_tolerance_degrees"])),
                )
            if transition is not None and not transition.feasible:
                reverse_transition = robot.continue_task_transition_to_configuration(
                    target_q,
                    q_values[-1],
                    edge_positions[::-1],
                    edge_axes[::-1],
                    maximum_joint_step=float(graph.metadata["maximum_joint_step"]),
                    minimum_manipulability=float(graph.metadata["minimum_manipulability"]),
                    position_tolerance=float(graph.metadata["task_position_tolerance"]),
                    axis_tolerance=np.deg2rad(float(graph.metadata["axis_tolerance_degrees"])),
                )
                if reverse_transition.feasible:
                    transition = type(reverse_transition)(
                        True,
                        reverse_transition.q_path[::-1],
                        None,
                        reverse_transition.max_position_error,
                        reverse_transition.max_axis_error,
                    )
                    reverse_positions, reverse_axes = _target_sequence_for_transition(
                        edge_positions[::-1], edge_axes[::-1]
                    )
                    edge_target_positions = reverse_positions[::-1]
                    edge_target_axes = reverse_axes[::-1]
            if transition is not None and not transition.feasible:
                return _failed_result(
                    graph,
                    selected,
                    covered,
                    f"edge_{edge_id}:{transition.failure_reason}",
                )
            if transition_q is None:
                assert transition is not None
                transition_q = transition.q_path
            q_values.extend(transition_q[1:])
            target_positions.extend(edge_target_positions[1:])
            target_axes.extend(edge_target_axes[1:])
            route_nodes.append(target_node)
        if len(q_values) == 1:
            q_values.append(q_values[0].copy())
            target_positions.append(np.asarray(target_positions[0]).copy())
            target_axes.append(np.asarray(target_axes[0]).copy())
        q_segments.append(np.asarray(q_values))
        position_segments.append(np.asarray(target_positions))
        axis_segments.append(np.asarray(target_axes))
        node_routes.append(np.asarray(route_nodes, dtype=np.int64))

    check = hard_check_qspace_plan(
        robot,
        tuple(q_segments),
        tuple(position_segments),
        tuple(axis_segments),
        position_tolerance=(
            float(graph.metadata["task_position_tolerance"])
            if hard_position_tolerance is None
            else hard_position_tolerance
        ),
        axis_tolerance=np.deg2rad(float(graph.metadata["axis_tolerance_degrees"])),
        minimum_manipulability=float(graph.metadata["minimum_manipulability"]),
    )
    if not check.feasible:
        return _failed_result(
            graph,
            selected,
            covered,
            check.failure_reason or "hard_check",
            metadata={
                "max_position_error": check.max_position_error,
                "max_axis_error": check.max_axis_error,
                "minimum_manipulability": check.minimum_manipulability,
                "minimum_joint_limit_margin": check.minimum_joint_limit_margin,
            },
        )
    workspace_plan = workspace_plan_from_q(
        robot, tuple(q_segments), transform_base_from_surface
    )
    coverage_metrics = evaluate_coverage(
        surface, workspace_plan, footprint_radius=footprint_radius
    )
    differences = np.concatenate(
        [np.diff(segment, axis=0) for segment in q_segments if len(segment) > 1], axis=0
    )
    node_visits = sum(len(route) for route in node_routes)
    unique_visits = sum(len(np.unique(route)) for route in node_routes)
    return QSpaceCoverageTeacherResult(
        feasible=True,
        q_segments=tuple(q_segments),
        target_position_segments=tuple(position_segments),
        target_axis_segments=tuple(axis_segments),
        node_routes=tuple(node_routes),
        selected_components=selected,
        covered_node_mask=covered,
        joint_travel=float(np.linalg.norm(differences, axis=1).sum()),
        squared_joint_motion=float(np.einsum("ij,ij->", differences, differences)),
        task_path_length=float(
            sum(
                np.linalg.norm(
                    np.diff(np.asarray([robot.forward(q)[0] for q in segment]), axis=0),
                    axis=1,
                ).sum()
                for segment in q_segments
            )
        ),
        repeated_node_visits=node_visits - unique_visits,
        coverage_metrics=coverage_metrics,
        metadata={
            "minimum_manipulability": check.minimum_manipulability,
            "minimum_joint_limit_margin": check.minimum_joint_limit_margin,
            "max_position_error": check.max_position_error,
            "max_axis_error": check.max_axis_error,
            "route_objective": route_objective,
        },
    )


def hard_check_qspace_plan(
    robot: UR5eKinematics,
    q_segments: tuple[np.ndarray, ...],
    target_position_segments: tuple[np.ndarray, ...],
    target_axis_segments: tuple[np.ndarray, ...],
    *,
    position_tolerance: float,
    axis_tolerance: float,
    minimum_manipulability: float,
    interpolation_joint_step: float | None = 0.05,
) -> QSpacePlanCheck:
    max_position_error = 0.0
    max_axis_error = 0.0
    min_manipulability = np.inf
    min_margin = np.inf
    for q_values, positions, axes in zip(
        q_segments, target_position_segments, target_axis_segments
    ):
        if interpolation_joint_step is not None:
            q_values, positions, axes = densify_qspace_segment(
                q_values,
                positions,
                axes,
                maximum_joint_step=interpolation_joint_step,
            )
        if q_values.shape[0] != len(positions) or positions.shape != axes.shape:
            return QSpacePlanCheck(False, "shape", np.inf, np.inf, 0.0, 0.0)
        for q, target_position, target_axis in zip(q_values, positions, axes):
            if np.any(q < robot.lower_limits) or np.any(q > robot.upper_limits):
                return QSpacePlanCheck(False, "joint_limit", np.inf, np.inf, 0.0, 0.0)
            achieved_position, achieved_axis = robot.forward(q)
            position_error = float(np.linalg.norm(achieved_position - target_position))
            axis_error = float(
                np.arccos(np.clip(np.dot(achieved_axis, target_axis), -1.0, 1.0))
            )
            candidate = robot.evaluate_configuration(q)
            max_position_error = max(max_position_error, position_error)
            max_axis_error = max(max_axis_error, axis_error)
            min_manipulability = min(min_manipulability, candidate.manipulability)
            min_margin = min(min_margin, candidate.joint_limit_margin)
            if position_error > position_tolerance:
                return QSpacePlanCheck(
                    False, "surface_tracking", max_position_error, max_axis_error,
                    min_manipulability, min_margin,
                )
            if axis_error > axis_tolerance:
                return QSpacePlanCheck(
                    False, "axis_tracking", max_position_error, max_axis_error,
                    min_manipulability, min_margin,
                )
            if not candidate.collision_free:
                return QSpacePlanCheck(
                    False, "collision", max_position_error, max_axis_error,
                    min_manipulability, min_margin,
                )
            if candidate.manipulability < minimum_manipulability:
                return QSpacePlanCheck(
                    False, "singularity", max_position_error, max_axis_error,
                    min_manipulability, min_margin,
                )
    return QSpacePlanCheck(
        True, None, max_position_error, max_axis_error,
        float(min_manipulability), float(min_margin),
    )


def _flatten_graph(graph: SurfaceIKGraph) -> dict[str, Any]:
    counts = [len(layer) for layer in graph.candidates]
    offsets = np.cumsum((0, *counts))
    nodes = np.repeat(np.arange(graph.num_nodes), counts)
    local = np.concatenate([np.arange(count) for count in counts])
    q = np.asarray(
        [candidate.q for layer in graph.candidates for candidate in layer],
        dtype=np.float64,
    ).reshape(-1, 6)
    components = np.concatenate(graph.component_labels)
    adjacency: list[list[tuple[int, float, float]]] = [[] for _ in range(len(nodes))]
    for edge_id, compatibility in enumerate(graph.edge_compatibility):
        source, target = (int(value) for value in graph.edge_index[:, edge_id])
        for source_local, target_local in np.argwhere(compatibility):
            source_flat = int(offsets[source] + source_local)
            target_flat = int(offsets[target] + target_local)
            difference = (q[target_flat] - q[source_flat] + np.pi) % (2.0 * np.pi) - np.pi
            joint_weight = float(np.linalg.norm(difference))
            surface_weight = float(
                np.linalg.norm(graph.positions[target] - graph.positions[source])
            )
            adjacency[source_flat].append((target_flat, surface_weight, joint_weight))
            adjacency[target_flat].append((source_flat, surface_weight, joint_weight))
    return {
        "offsets": offsets,
        "node": nodes,
        "local": local,
        "q": q,
        "component": components,
        "adjacency": adjacency,
    }


def _route_component_targets(
    graph: SurfaceIKGraph,
    flat: dict[str, Any],
    component: int,
    target_nodes: np.ndarray,
    route_objective: str,
) -> np.ndarray:
    candidate_ids = np.flatnonzero(flat["component"] == component)
    target_ids = candidate_ids[target_nodes[flat["node"][candidate_ids]]]
    start = max(target_ids.tolist(), key=lambda index: (len(flat["adjacency"][index]), -index))
    route = [start]
    visited_nodes = np.zeros(graph.num_nodes, dtype=bool)
    visited_nodes[int(flat["node"][start])] = True
    while np.any(target_nodes & ~visited_nodes):
        destination, parents = _nearest_unvisited_target(
            route[-1], flat, component, target_nodes & ~visited_nodes, route_objective
        )
        extension = [destination]
        while extension[-1] != route[-1]:
            extension.append(parents[extension[-1]])
        extension.reverse()
        route.extend(extension[1:])
        visited_nodes[flat["node"][extension]] = True
    return np.asarray(route, dtype=np.int64)


def _nearest_unvisited_target(
    start: int,
    flat: dict[str, Any],
    component: int,
    target_nodes: np.ndarray,
    route_objective: str,
) -> tuple[int, dict[int, int]]:
    queue = [(0.0, 0.0, start)]
    distances = {start: (0.0, 0.0)}
    parents: dict[int, int] = {}
    while queue:
        surface_distance, joint_distance, node = heapq.heappop(queue)
        if (surface_distance, joint_distance) != distances[node]:
            continue
        if target_nodes[int(flat["node"][node])]:
            return node, parents
        for neighbour, surface_weight, joint_weight in flat["adjacency"][node]:
            if int(flat["component"][neighbour]) != component:
                continue
            edge_cost = (
                (surface_weight, joint_weight)
                if route_objective == "surface_then_joint"
                else (joint_weight, surface_weight)
            )
            candidate = (
                surface_distance + edge_cost[0],
                joint_distance + edge_cost[1],
            )
            if candidate < distances.get(neighbour, (np.inf, np.inf)):
                distances[neighbour] = candidate
                parents[neighbour] = node
                heapq.heappush(queue, (*candidate, neighbour))
    raise RuntimeError("component-labelled target is disconnected in the candidate graph")


def _surface_edge_lookup(graph: SurfaceIKGraph) -> dict[tuple[int, int], tuple[int, bool]]:
    lookup = {}
    for edge_id, (source, target) in enumerate(graph.edge_index.T):
        lookup[(int(source), int(target))] = (edge_id, False)
        lookup[(int(target), int(source))] = (edge_id, True)
    return lookup


def _target_sequence_for_transition(
    positions: np.ndarray, axes: np.ndarray, final_samples: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    fractions = np.linspace(0.0, 1.0, final_samples)
    final_positions = (
        (1.0 - fractions[:, None]) * positions[-2]
        + fractions[:, None] * positions[-1]
    )
    final_axes = (
        (1.0 - fractions[:, None]) * axes[-2]
        + fractions[:, None] * axes[-1]
    )
    final_axes /= np.maximum(np.linalg.norm(final_axes, axis=1, keepdims=True), 1e-12)
    return (
        np.concatenate((positions[:-1], final_positions[1:]), axis=0),
        np.concatenate((axes[:-1], final_axes[1:]), axis=0),
    )


def workspace_plan_from_q(
    robot: UR5eKinematics,
    q_segments: tuple[np.ndarray, ...],
    transform_base_from_surface: np.ndarray,
) -> CoveragePlan:
    inverse = np.linalg.inv(np.asarray(transform_base_from_surface, dtype=np.float64))
    paths = []
    for q_values in q_segments:
        base_positions = np.asarray([robot.forward(q)[0] for q in q_values])
        homogeneous = np.column_stack((base_positions, np.ones(len(base_positions))))
        paths.append((inverse @ homogeneous.T).T[:, :3])
    maximum = max(len(path) for path in paths)
    waypoints = np.zeros((len(paths), maximum, 3), dtype=np.float64)
    mask = np.zeros((len(paths), maximum), dtype=bool)
    for index, path in enumerate(paths):
        waypoints[index, : len(path)] = path
        mask[index, : len(path)] = True
    return CoveragePlan(waypoints, waypoint_mask=mask)


def _failed_result(
    graph: SurfaceIKGraph,
    selected: np.ndarray,
    covered: np.ndarray,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> QSpaceCoverageTeacherResult:
    return QSpaceCoverageTeacherResult(
        feasible=False,
        q_segments=(),
        target_position_segments=(),
        target_axis_segments=(),
        node_routes=(),
        selected_components=selected,
        covered_node_mask=covered,
        joint_travel=np.inf,
        squared_joint_motion=np.inf,
        task_path_length=np.inf,
        repeated_node_visits=0,
        coverage_metrics=None,
        failure_reason=reason,
        metadata={} if metadata is None else metadata,
    )
