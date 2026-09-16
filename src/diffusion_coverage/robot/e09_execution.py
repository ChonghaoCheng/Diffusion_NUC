from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.coverage.episode_summary import EpisodeEdgeSummary, summarize_ordered_membership
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.robot.synchronized_motion import SynchronizedMotionTrace


@dataclass(frozen=True)
class E09TraceCheck:
    summary: EpisodeEdgeSummary
    surface_points: np.ndarray
    projection_residuals: np.ndarray
    max_position_error: float
    max_axis_error: float
    min_sigma5: float
    min_joint_margin: float
    collision_free: bool
    sigma5: np.ndarray


def sphere_membership_stream(
    sample_points: np.ndarray,
    trace_points: np.ndarray,
    *,
    radius: float,
    footprint_radius: float,
    chunk_size: int = 256,
) -> np.ndarray:
    """Exact intrinsic-sphere membership without mesh path reconstruction."""

    samples = np.asarray(sample_points, dtype=np.float64)
    trace = np.asarray(trace_points, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError("sphere points must have shape [N, 3]")
    unit_samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    unit_trace = trace / np.linalg.norm(trace, axis=1, keepdims=True)
    threshold = float(np.cos(footprint_radius / radius))
    output = np.empty((len(samples), len(trace)), dtype=bool)
    for begin in range(0, len(samples), chunk_size):
        end = min(begin + chunk_size, len(samples))
        output[begin:end] = unit_samples[begin:end] @ unit_trace.T >= threshold - 2e-15
    return output


def evaluate_e09_fk_trace(
    robot: UR5eKinematics,
    q_values: np.ndarray,
    activity: np.ndarray,
    declared_surface_points: np.ndarray,
    transform_base_from_surface: np.ndarray,
    quadrature_points: np.ndarray,
    quadrature_weights: np.ndarray,
    *,
    sphere_radius: float,
    footprint_radius: float,
    characteristic_length: float,
) -> E09TraceCheck:
    """Validate one ordered E09 witness and summarize its achieved FK footprint."""

    q = np.asarray(q_values, dtype=np.float64)
    active = np.asarray(activity, dtype=bool)
    target = np.asarray(declared_surface_points, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6 or active.shape != (len(q),) or target.shape != (len(q), 3):
        raise ValueError("incompatible E09 trace arrays")
    transform = np.asarray(transform_base_from_surface, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    achieved_surface = np.empty((len(q), 3), dtype=np.float64)
    residuals = np.empty(len(q), dtype=np.float64)
    position_errors = np.empty(len(q), dtype=np.float64)
    axis_errors = np.empty(len(q), dtype=np.float64)
    sigma = np.empty(len(q), dtype=np.float64)
    margin = np.empty(len(q), dtype=np.float64)
    collisions = np.zeros(len(q), dtype=bool)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    for index, value in enumerate(q):
        task = evaluate_task_kinematics_5d(robot, value, characteristic_length=characteristic_length)
        raw = (task.position - translation) @ rotation
        norm = float(np.linalg.norm(raw))
        projected = sphere_radius * raw / norm
        achieved_surface[index] = projected
        residuals[index] = abs(norm - sphere_radius)
        desired_base = target[index] @ rotation.T + translation
        desired_axis = -(target[index] / np.linalg.norm(target[index])) @ rotation.T
        position_errors[index] = np.linalg.norm(task.position - desired_base)
        axis_errors[index] = np.arccos(np.clip(np.dot(task.tool_axis, desired_axis), -1.0, 1.0))
        sigma[index] = task.sigma_min_5
        checked = robot.evaluate_configuration(value)
        margin[index] = checked.joint_limit_margin
        collisions[index] = not checked.collision_free
    membership = sphere_membership_stream(
        quadrature_points,
        achieved_surface,
        radius=sphere_radius,
        footprint_radius=footprint_radius,
    )
    membership[:, ~active] = False
    summary = summarize_ordered_membership(membership, quadrature_weights, active=active)
    return E09TraceCheck(
        summary=summary,
        surface_points=achieved_surface,
        projection_residuals=residuals,
        max_position_error=float(position_errors[active].max(initial=0.0)),
        max_axis_error=float(axis_errors[active].max(initial=0.0)),
        min_sigma5=float(sigma[active].min(initial=np.inf)),
        min_joint_margin=float(margin.min(initial=np.inf)),
        collision_free=not bool(collisions.any()),
        sigma5=sigma.copy(),
    )


def evaluate_synchronized_fk_trace(
    robot: UR5eKinematics,
    trace: SynchronizedMotionTrace,
    transform_base_from_surface: np.ndarray,
    quadrature_points: np.ndarray,
    quadrature_weights: np.ndarray,
    *,
    sphere_radius: float,
    footprint_radius: float,
    characteristic_length: float,
) -> E09TraceCheck:
    """Validate a synchronized trace whose declared task positions/axes are in robot base."""

    trace.__post_init__()
    transform = np.asarray(transform_base_from_surface, dtype=np.float64)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    achieved_surface = np.empty((len(trace.q), 3), dtype=np.float64)
    residuals = np.empty(len(trace.q), dtype=np.float64)
    position_errors = np.empty(len(trace.q), dtype=np.float64)
    axis_errors = np.empty(len(trace.q), dtype=np.float64)
    sigma = np.empty(len(trace.q), dtype=np.float64)
    margin = np.empty(len(trace.q), dtype=np.float64)
    collisions = np.zeros(len(trace.q), dtype=bool)
    for index, value in enumerate(trace.q):
        task = evaluate_task_kinematics_5d(robot, value, characteristic_length=characteristic_length)
        raw = (task.position - translation) @ rotation
        norm = float(np.linalg.norm(raw))
        if norm <= 1e-14:
            raise ValueError("TCP cannot be projected to the declared sphere")
        achieved_surface[index] = sphere_radius * raw / norm
        residuals[index] = abs(norm - sphere_radius)
        position_errors[index] = np.linalg.norm(task.position - trace.target_position[index])
        desired_axis = trace.target_axis[index] / np.linalg.norm(trace.target_axis[index])
        axis_errors[index] = np.arccos(np.clip(np.dot(task.tool_axis, desired_axis), -1.0, 1.0))
        sigma[index] = task.sigma_min_5
        checked = robot.evaluate_configuration(value)
        margin[index] = checked.joint_limit_margin
        collisions[index] = not checked.collision_free
    membership = sphere_membership_stream(
        quadrature_points, achieved_surface, radius=sphere_radius,
        footprint_radius=footprint_radius,
    )
    membership[:, ~trace.activity] = False
    summary = summarize_ordered_membership(membership, quadrature_weights, active=trace.activity)
    on = trace.activity
    return E09TraceCheck(
        summary=summary,
        surface_points=achieved_surface,
        projection_residuals=residuals,
        max_position_error=float(position_errors[on].max(initial=0.0)),
        max_axis_error=float(axis_errors[on].max(initial=0.0)),
        min_sigma5=float(sigma[on].min(initial=np.inf)),
        min_joint_margin=float(margin.min(initial=np.inf)),
        collision_free=not bool(collisions.any()),
        sigma5=sigma.copy(),
    )


def sphere_episode_counts_indexed(
    sample_points: np.ndarray,
    trace_points: np.ndarray,
    activity: np.ndarray,
    *,
    radius: float,
    footprint_radius: float,
) -> np.ndarray:
    """Count sampled episodes without allocating a surface-by-time matrix."""

    from scipy.spatial import cKDTree

    samples = np.asarray(sample_points, dtype=np.float64)
    trace = np.asarray(trace_points, dtype=np.float64)
    active = np.asarray(activity, dtype=bool)
    if samples.ndim != 2 or samples.shape[1] != 3 or trace.ndim != 2 or trace.shape[1] != 3 or active.shape != (len(trace),):
        raise ValueError("invalid indexed episode inputs")
    unit_samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    unit_trace = trace / np.linalg.norm(trace, axis=1, keepdims=True)
    threshold = float(np.cos(footprint_radius / radius))
    chord = float(np.sqrt(max(0.0, 2.0 - 2.0 * threshold))) + 4e-15
    tree = cKDTree(unit_samples)
    last = np.full(len(samples), -2, dtype=np.int64)
    counts = np.zeros(len(samples), dtype=np.int64)
    for time_index, point in enumerate(unit_trace):
        if not active[time_index]:
            continue
        candidates = np.asarray(tree.query_ball_point(point, chord), dtype=np.int64)
        if len(candidates) == 0:
            continue
        hits = candidates[unit_samples[candidates] @ point >= threshold - 2e-15]
        counts[hits[last[hits] != time_index - 1]] += 1
        last[hits] = time_index
    return counts
