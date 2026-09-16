from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from diffusion_coverage.robot.ur5e_mujoco import TaskTransitionResult, UR5eKinematics


CurveEvaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class SynchronizedMotionTrace:
    """A joint witness and declared task sampled at one shared path parameter."""

    q: np.ndarray
    u: np.ndarray
    target_position: np.ndarray
    target_axis: np.ndarray
    activity: np.ndarray
    geometry_arc_id: int
    original_interval: tuple[float, float]
    start_node: int
    end_node: int

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=np.float64)
        u = np.asarray(self.u, dtype=np.float64)
        p = np.asarray(self.target_position, dtype=np.float64)
        a = np.asarray(self.target_axis, dtype=np.float64)
        active = np.asarray(self.activity, dtype=bool)
        n = len(q)
        if q.ndim != 2 or q.shape[1] != 6:
            raise ValueError("q must have shape [N, 6]")
        if u.shape != (n,) or p.shape != (n, 3) or a.shape != (n, 3) or active.shape != (n,):
            raise ValueError("synchronized motion arrays have unequal or invalid shapes")
        if n < 2 or not all(np.all(np.isfinite(x)) for x in (q, u, p, a)):
            raise ValueError("synchronized motion arrays must be finite and contain two samples")
        if np.any(np.diff(u) < -1e-14):
            raise ValueError("motion parameter must be nondecreasing")
        if np.any(np.linalg.norm(a, axis=1) <= 1e-12):
            raise ValueError("target axes must be nonzero")

    def reversed(self) -> "SynchronizedMotionTrace":
        lo, hi = self.original_interval
        u = lo + hi - self.u[::-1]
        return SynchronizedMotionTrace(
            self.q[::-1].copy(), u.copy(), self.target_position[::-1].copy(),
            self.target_axis[::-1].copy(), self.activity[::-1].copy(),
            self.geometry_arc_id, (lo, hi), self.end_node, self.start_node,
        )


def spherical_polyline_parameter(points: np.ndarray, radius: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("curve points must have shape [N,3], N>=2")
    unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    angles = np.arctan2(np.linalg.norm(np.cross(unit[:-1], unit[1:]), axis=1), np.sum(unit[:-1] * unit[1:], axis=1))
    lengths = radius * angles
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-15:
        return np.linspace(0.0, 1.0, len(values))
    return cumulative / cumulative[-1]


def spherical_polyline_evaluator(
    surface_points: np.ndarray,
    transform_base_from_surface: np.ndarray,
    *,
    radius: float,
    axis_opposes_normal: bool = True,
    parameter: np.ndarray | None = None,
) -> tuple[np.ndarray, CurveEvaluator]:
    points = np.asarray(surface_points, dtype=np.float64)
    knots = spherical_polyline_parameter(points, radius) if parameter is None else np.asarray(parameter, dtype=np.float64)
    if knots.shape != (len(points),) or np.any(np.diff(knots) < 0.0):
        raise ValueError("invalid curve parameter")
    transform = np.asarray(transform_base_from_surface, dtype=np.float64)
    rotation, translation = transform[:3, :3], transform[:3, 3]

    def evaluate(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float64)
        if np.any(query < knots[0] - 1e-14) or np.any(query > knots[-1] + 1e-14):
            raise ValueError("curve parameter outside registered interval")
        surface = np.empty((len(query), 3), dtype=np.float64)
        for out_index, value in enumerate(query):
            index = min(int(np.searchsorted(knots, value, side="right")) - 1, len(points) - 2)
            index = max(index, 0)
            span = knots[index + 1] - knots[index]
            fraction = 0.0 if span <= 1e-15 else float((value - knots[index]) / span)
            x = points[index] / np.linalg.norm(points[index])
            y = points[index + 1] / np.linalg.norm(points[index + 1])
            angle = float(np.arctan2(np.linalg.norm(np.cross(x, y)), np.dot(x, y)))
            if angle <= 1e-14:
                point = (1.0 - fraction) * points[index] + fraction * points[index + 1]
                point = radius * point / np.linalg.norm(point)
            else:
                point = radius * (np.sin((1.0 - fraction) * angle) * x + np.sin(fraction * angle) * y) / np.sin(angle)
            surface[out_index] = point
        positions = surface @ rotation.T + translation
        axes = surface / np.linalg.norm(surface, axis=1, keepdims=True)
        if axis_opposes_normal:
            axes = -axes
        axes = axes @ rotation.T
        return positions, axes

    return knots, evaluate


def continue_to_configuration_synchronized(
    robot: UR5eKinematics,
    start_q: np.ndarray,
    end_q: np.ndarray,
    target_u: np.ndarray,
    curve: CurveEvaluator,
    *,
    geometry_arc_id: int,
    start_node: int,
    end_node: int,
    final_tracking_samples: int = 3,
    maximum_joint_step: float = 0.8,
    minimum_manipulability: float = 0.0,
    position_tolerance: float = 1e-4,
    axis_tolerance: float = np.deg2rad(0.1),
    backend: str = "task5",
) -> tuple[TaskTransitionResult, SynchronizedMotionTrace | None]:
    """Continue to an exact stored q while retaining the tail's task parameter rows."""

    u = np.asarray(target_u, dtype=np.float64)
    if u.ndim != 1 or len(u) < 3 or np.any(np.diff(u) <= 0.0):
        raise ValueError("target_u must contain at least three strictly increasing values")
    if final_tracking_samples < 2:
        raise ValueError("final_tracking_samples must be at least two")
    positions, axes = curve(u)
    prefix = robot.continue_task_transition(
        start_q, positions[:-1], axes[:-1], maximum_joint_step=maximum_joint_step,
        minimum_manipulability=minimum_manipulability, position_tolerance=position_tolerance,
        axis_tolerance=axis_tolerance, backend=backend,
    )
    if not prefix.feasible:
        return prefix, None
    tail_u = np.linspace(u[-2], u[-1], final_tracking_samples)
    tail_positions, tail_axes = curve(tail_u)
    final = robot.check_task_transition(
        prefix.q_path[-1], end_q, tail_positions, tail_axes,
        maximum_joint_step=maximum_joint_step, minimum_manipulability=minimum_manipulability,
        position_tolerance=position_tolerance, axis_tolerance=axis_tolerance,
        allow_equivalent_end=False,
    )
    q = np.concatenate((prefix.q_path, final.q_path[1:]), axis=0)
    combined_u = np.concatenate((u[:-1], tail_u[1:]))
    combined_positions, combined_axes = curve(combined_u)
    result = TaskTransitionResult(
        final.feasible, q, final.failure_reason,
        max(prefix.max_position_error, final.max_position_error),
        max(prefix.max_axis_error, final.max_axis_error),
    )
    if not result.feasible:
        return result, None
    trace = SynchronizedMotionTrace(
        q, combined_u, combined_positions, combined_axes, np.ones(len(q), dtype=bool),
        geometry_arc_id, (float(u[0]), float(u[-1])), start_node, end_node,
    )
    return result, trace


def densify_synchronized_trace(
    trace: SynchronizedMotionTrace,
    curve: CurveEvaluator,
    *,
    maximum_joint_step: float,
    maximum_parameter_step: float,
) -> SynchronizedMotionTrace:
    """Densify the stored piecewise-linear q(u) without silently pairing unequal arrays."""

    trace.__post_init__()
    qout = [trace.q[0]]
    uout = [trace.u[0]]
    aout = [bool(trace.activity[0])]
    for index in range(len(trace.q) - 1):
        qa, qb = trace.q[index], trace.q[index + 1]
        ua, ub = float(trace.u[index]), float(trace.u[index + 1])
        count = max(1, int(np.ceil(max(np.max(np.abs(qb - qa)) / maximum_joint_step, abs(ub - ua) / maximum_parameter_step))))
        for step in range(1, count + 1):
            fraction = step / count
            qout.append((1.0 - fraction) * qa + fraction * qb)
            uout.append((1.0 - fraction) * ua + fraction * ub)
            aout.append(bool(trace.activity[index + 1]) if step == count else bool(trace.activity[index] and trace.activity[index + 1]))
    positions, axes = curve(np.asarray(uout))
    return SynchronizedMotionTrace(
        np.asarray(qout), np.asarray(uout), positions, axes, np.asarray(aout),
        trace.geometry_arc_id, trace.original_interval, trace.start_node, trace.end_node,
    )
