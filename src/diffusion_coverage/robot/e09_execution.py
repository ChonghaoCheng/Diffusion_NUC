from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.coverage.episode_summary import EpisodeEdgeSummary, summarize_ordered_membership
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


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
    )
