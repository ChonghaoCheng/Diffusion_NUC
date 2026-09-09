from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.coverage.nuc_evaluator import NUCCoverageMetrics, evaluate_nuc_coverage
from diffusion_coverage.robot.execution_cost import JointExecutionCost, compute_joint_execution_cost
from diffusion_coverage.robot.qspace_coverage_teacher import densify_qspace_segment, workspace_plan_from_q
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.surface.surface_instance import SurfaceInstance


STRICT_FAILURE_REASONS = {
    "joint_limit",
    "position_tracking",
    "axis_tracking",
    "task_singularity",
    "robot_collision",
    "missing_surface_path",
    "coverage_miss",
    "coverage_repeat",
    "coverage_nuc",
    "numerical_check_failure",
}


@dataclass(frozen=True)
class StrictCoverageExecutionResult:
    kinematics_pass: bool
    coverage_pass: bool
    timing_pass: bool | None
    overall_pass: bool
    failure_reason: str | None
    failure_reasons: tuple[str, ...]
    max_position_error: float
    max_axis_error: float
    min_sigma_min_5: float
    min_mu_bar: float
    min_joint_limit_margin: float
    manipulability_6d_legacy: float
    execution_cost: JointExecutionCost | None
    coverage_metrics: NUCCoverageMetrics | None
    checked_q_segments: tuple[np.ndarray, ...]


def check_strict_coverage_execution(
    robot: UR5eKinematics,
    q_segments: tuple[np.ndarray, ...],
    desired_position_segments: tuple[np.ndarray, ...],
    desired_axis_segments: tuple[np.ndarray, ...],
    surface: SurfaceInstance,
    transform_base_from_surface: np.ndarray,
    *,
    footprint_radius: float,
    position_tolerance: float,
    axis_tolerance: float,
    characteristic_length: float,
    sigma_safe: float,
    missed_tolerance: float,
    repeat_tolerance: float,
    interpolation_joint_step: float,
    nuc_error_tolerance: float | None = None,
    coverage_path_sample_spacing: float | None = None,
    weight_matrix: np.ndarray | None = None,
) -> StrictCoverageExecutionResult:
    """Apply one dense final admission contract to any numerical q witness."""

    failures: list[str] = []
    checked_q: list[np.ndarray] = []
    max_position_error = 0.0
    max_axis_error = 0.0
    min_sigma = np.inf
    min_mu = np.inf
    min_margin = np.inf
    min_legacy = np.inf

    if (
        not q_segments
        or len(q_segments) != len(desired_position_segments)
        or len(q_segments) != len(desired_axis_segments)
    ):
        failures.append("missing_surface_path")
    else:
        for q_values, positions, axes in zip(
            q_segments, desired_position_segments, desired_axis_segments
        ):
            try:
                q_dense, positions_dense, axes_dense = densify_qspace_segment(
                    q_values,
                    positions,
                    axes,
                    maximum_joint_step=interpolation_joint_step,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                failures.append("numerical_check_failure")
                continue
            checked_q.append(q_dense)
            for q, desired_position, desired_axis in zip(q_dense, positions_dense, axes_dense):
                try:
                    if np.any(q < robot.lower_limits) or np.any(q > robot.upper_limits):
                        failures.append("joint_limit")
                        continue
                    task = evaluate_task_kinematics_5d(
                        robot, q, characteristic_length=characteristic_length
                    )
                    position_error = float(np.linalg.norm(task.position - desired_position))
                    axis_error = float(
                        np.arccos(
                            np.clip(np.dot(task.tool_axis, desired_axis), -1.0, 1.0)
                        )
                    )
                    candidate = robot.evaluate_configuration(q)
                    max_position_error = max(max_position_error, position_error)
                    max_axis_error = max(max_axis_error, axis_error)
                    min_sigma = min(min_sigma, task.sigma_min_5)
                    min_mu = min(min_mu, task.mu_bar)
                    min_margin = min(min_margin, candidate.joint_limit_margin)
                    min_legacy = min(min_legacy, candidate.manipulability)
                    if position_error > position_tolerance:
                        failures.append("position_tracking")
                    if axis_error > axis_tolerance:
                        failures.append("axis_tracking")
                    if task.sigma_min_5 < sigma_safe:
                        failures.append("task_singularity")
                    if not candidate.collision_free:
                        failures.append("robot_collision")
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    failures.append("numerical_check_failure")

    failures = list(dict.fromkeys(failures))
    kinematics_failures = set(failures) & {
        "joint_limit",
        "position_tracking",
        "axis_tracking",
        "task_singularity",
        "robot_collision",
        "missing_surface_path",
        "numerical_check_failure",
    }
    kinematics_pass = not kinematics_failures
    coverage_metrics = None
    execution_cost = None
    if checked_q and len(checked_q) == len(q_segments):
        try:
            checked_tuple = tuple(checked_q)
            execution_cost = compute_joint_execution_cost(
                checked_tuple, weight_matrix=weight_matrix
            )
            workspace_plan = workspace_plan_from_q(
                robot, checked_tuple, transform_base_from_surface
            )
            coverage_metrics = evaluate_nuc_coverage(
                surface,
                workspace_plan,
                footprint_radius=footprint_radius,
                path_sample_spacing=coverage_path_sample_spacing,
            )
            if coverage_metrics.missed_error > missed_tolerance:
                failures.append("coverage_miss")
            if coverage_metrics.repeat_error > repeat_tolerance:
                failures.append("coverage_repeat")
            if (
                nuc_error_tolerance is not None
                and coverage_metrics.nuc_error > nuc_error_tolerance
            ):
                failures.append("coverage_nuc")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            failures.append("numerical_check_failure")
    else:
        failures.append("missing_surface_path")

    failures = list(dict.fromkeys(failures))
    coverage_pass = coverage_metrics is not None and not (
        {"coverage_miss", "coverage_repeat", "coverage_nuc", "missing_surface_path", "numerical_check_failure"}
        & set(failures)
    )
    timing_pass = None
    overall_pass = kinematics_pass and coverage_pass
    checked_tuple = tuple(checked_q)
    return StrictCoverageExecutionResult(
        kinematics_pass=kinematics_pass,
        coverage_pass=coverage_pass,
        timing_pass=timing_pass,
        overall_pass=overall_pass,
        failure_reason=None if not failures else failures[0],
        failure_reasons=tuple(failures),
        max_position_error=float(max_position_error),
        max_axis_error=float(max_axis_error),
        min_sigma_min_5=float(min_sigma),
        min_mu_bar=float(min_mu),
        min_joint_limit_margin=float(min_margin),
        manipulability_6d_legacy=float(min_legacy),
        execution_cost=execution_cost,
        coverage_metrics=coverage_metrics,
        checked_q_segments=checked_tuple,
    )
