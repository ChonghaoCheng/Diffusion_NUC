from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.geometry.robot_surface_metric import (
    compute_robot_surface_metric,
    estimate_surface_contact_differential,
)
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class ProbeContinuation:
    feasible: bool
    q_path: np.ndarray
    failure_reason: str | None
    max_position_error: float
    max_axis_error: float


def require_frozen_experiment_inputs(scenes: dict, anchors: dict | None = None) -> None:
    if not scenes.get("frozen_before_R1", False):
        raise RuntimeError("R1 scenes were not frozen before directional probes")
    if anchors is not None and not anchors.get("frozen_before_probe_results", False):
        raise RuntimeError("R1 anchors were not frozen before directional probes")


def select_uniform_indices(
    cumulative_arclength: np.ndarray,
    eligible_indices: list[int],
    maximum_count: int,
) -> list[int]:
    if maximum_count < 1:
        raise ValueError("maximum_count must be positive")
    if len(eligible_indices) <= maximum_count:
        return sorted(eligible_indices)
    cumulative = np.asarray(cumulative_arclength, dtype=np.float64)
    eligible = np.asarray(sorted(eligible_indices), dtype=np.int64)
    values = cumulative[eligible]
    targets = np.linspace(values[0], values[-1], maximum_count)
    selected = []
    for target in targets:
        ordered = np.argsort(np.abs(values - target), kind="stable")
        index = next(int(eligible[item]) for item in ordered if int(eligible[item]) not in selected)
        selected.append(index)
    return sorted(selected)


def absolute_joint_limit_margin(robot: UR5eKinematics, q_path: np.ndarray) -> float:
    values = np.asarray(q_path, dtype=np.float64)
    margin = np.minimum(values - robot.lower_limits, robot.upper_limits - values)
    return float(np.min(margin))


def metric_along_witness(
    robot: UR5eKinematics,
    surface: SurfaceInstance,
    transform: np.ndarray,
    q_path: np.ndarray,
    desired_positions: np.ndarray,
    *,
    characteristic_length: float,
    sigma_safe: float,
    sample_count: int,
    finite_difference_step: float,
) -> list[dict[str, float | int]]:
    q_values = np.asarray(q_path, dtype=np.float64)
    positions = np.asarray(desired_positions, dtype=np.float64)
    if len(q_values) != len(positions):
        raise ValueError("witness q and desired positions must have equal length")
    transform_value = np.asarray(transform, dtype=np.float64)
    inverse_rotation = transform_value[:3, :3].T
    indices = np.unique(np.linspace(0, len(q_values) - 1, min(sample_count, len(q_values)), dtype=int))
    rows = []
    for index in indices:
        task = evaluate_task_kinematics_5d(robot, q_values[index], characteristic_length=characteristic_length)
        if task.sigma_min_5 < sigma_safe - 1e-12:
            raise ValueError("witness sample violates sigma_safe")
        point_surface = inverse_rotation @ (positions[index] - transform_value[:3, 3])
        contact = estimate_surface_contact_differential(
            surface, point_surface, transform_value, task.axis_basis,
            characteristic_length=characteristic_length,
            finite_difference_step=finite_difference_step,
        )
        metric = compute_robot_surface_metric(task, contact.task_differential, minimum_singular_value=sigma_safe)
        rows.append({
            "witness_index": int(index), "sigma_min_5": task.sigma_min_5,
            "lambda_min": float(metric.eigenvalues[0]), "lambda_max": float(metric.eigenvalues[1]),
            "kappa_R": metric.kappa_R, "log_kappa_R": metric.log_kappa_R,
            "R_G": metric.R_G, "sqrt_lambda_min": metric.sqrt_lambda_min,
            "sqrt_lambda_max": metric.sqrt_lambda_max,
        })
    return rows


def continue_probe_from_shared_q(
    robot: UR5eKinematics,
    q_start: np.ndarray,
    desired_positions: np.ndarray,
    desired_axes: np.ndarray,
    *,
    maximum_joint_step: float,
    position_tolerance: float,
    axis_tolerance: float,
) -> ProbeContinuation:
    start = np.asarray(q_start, dtype=np.float64).copy()
    result = robot.continue_task_transition(
        start, desired_positions, desired_axes,
        maximum_joint_step=maximum_joint_step,
        minimum_manipulability=0.0,
        position_tolerance=position_tolerance,
        axis_tolerance=axis_tolerance,
    )
    if len(result.q_path) and not np.array_equal(result.q_path[0], start):
        raise RuntimeError("probe continuation did not preserve the shared initial q")
    return ProbeContinuation(
        result.feasible, np.asarray(result.q_path, dtype=np.float64), result.failure_reason,
        result.max_position_error, result.max_axis_error,
    )
