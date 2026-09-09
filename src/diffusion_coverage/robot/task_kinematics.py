from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


@dataclass(frozen=True)
class TaskKinematics5D:
    position: np.ndarray
    tool_axis: np.ndarray
    axis_basis: np.ndarray
    jacobian_5: np.ndarray
    normalized_jacobian_5: np.ndarray
    singular_values: np.ndarray
    sigma_min_5: float
    mu_bar: float
    characteristic_length: float


def orthonormal_axis_basis(axis: np.ndarray) -> np.ndarray:
    """Return a deterministic right-handed basis for the plane normal to ``axis``."""

    value = np.asarray(axis, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("axis must have shape [3]")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("axis cannot be zero")
    value = value / norm
    reference = np.eye(3)[int(np.argmin(np.abs(value)))]
    u = np.cross(reference, value)
    u /= np.linalg.norm(u)
    v = np.cross(value, u)
    return np.column_stack((u, v))


def task_jacobian_5d(
    jacobian_position: np.ndarray,
    jacobian_rotation: np.ndarray,
    tool_axis: np.ndarray,
    *,
    axis_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    jp = np.asarray(jacobian_position, dtype=np.float64)
    jw = np.asarray(jacobian_rotation, dtype=np.float64)
    if jp.shape != jw.shape or jp.ndim != 2 or jp.shape[0] != 3:
        raise ValueError("site Jacobians must have matching shape [3, nq]")
    basis = orthonormal_axis_basis(tool_axis) if axis_basis is None else np.asarray(axis_basis, dtype=np.float64)
    if basis.shape != (3, 2):
        raise ValueError("axis_basis must have shape [3, 2]")
    axis = np.asarray(tool_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    if not np.allclose(basis.T @ basis, np.eye(2), atol=1e-9) or not np.allclose(
        basis.T @ axis, 0.0, atol=1e-9
    ):
        raise ValueError("axis_basis must be orthonormal and perpendicular to tool_axis")
    return np.vstack((jp, basis[:, 0] @ jw, basis[:, 1] @ jw)), basis


def normalize_task_jacobian(jacobian_5: np.ndarray, characteristic_length: float) -> np.ndarray:
    value = np.asarray(jacobian_5, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != 5:
        raise ValueError("jacobian_5 must have shape [5, nq]")
    if characteristic_length <= 0.0:
        raise ValueError("characteristic_length must be positive")
    normalized = value.copy()
    normalized[:3] /= characteristic_length
    return normalized


def task_singularity_metrics(
    jacobian_position: np.ndarray,
    jacobian_rotation: np.ndarray,
    tool_axis: np.ndarray,
    *,
    characteristic_length: float,
    axis_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    jacobian_5, basis = task_jacobian_5d(
        jacobian_position,
        jacobian_rotation,
        tool_axis,
        axis_basis=axis_basis,
    )
    normalized = normalize_task_jacobian(jacobian_5, characteristic_length)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    return (
        basis,
        jacobian_5,
        singular_values,
        float(singular_values[-1]),
        float(np.prod(singular_values)),
    )


def evaluate_task_kinematics_5d(
    robot: UR5eKinematics,
    q: np.ndarray,
    *,
    characteristic_length: float,
) -> TaskKinematics5D:
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (robot.model.nq,):
        raise ValueError("q has the wrong shape for the robot model")
    robot.data.qpos[:] = value
    mujoco.mj_forward(robot.model, robot.data)
    rotation = robot.data.site_xmat[robot.site_id].reshape(3, 3)
    axis = robot.tool_axis_sign * rotation[:, robot.tool_axis_index]
    jacobian_position = np.zeros((3, robot.model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, robot.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        robot.model,
        robot.data,
        jacobian_position,
        jacobian_rotation,
        robot.site_id,
    )
    basis, jacobian_5, singular_values, sigma_min, mu_bar = task_singularity_metrics(
        jacobian_position,
        jacobian_rotation,
        axis,
        characteristic_length=characteristic_length,
    )
    return TaskKinematics5D(
        position=robot.data.site_xpos[robot.site_id].copy(),
        tool_axis=axis.copy(),
        axis_basis=basis,
        jacobian_5=jacobian_5,
        normalized_jacobian_5=normalize_task_jacobian(jacobian_5, characteristic_length),
        singular_values=singular_values,
        sigma_min_5=sigma_min,
        mu_bar=mu_bar,
        characteristic_length=float(characteristic_length),
    )
