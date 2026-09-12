from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.robot.task_kinematics import TaskKinematics5D
from diffusion_coverage.robot.ur5e_mujoco import interpolate_vertex_normals
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class SurfaceContactDifferential:
    point_surface: np.ndarray
    point_base: np.ndarray
    surface_normal_base: np.ndarray
    tool_axis: np.ndarray
    tangent_basis: np.ndarray
    tool_axis_jacobian: np.ndarray
    task_differential: np.ndarray


@dataclass(frozen=True)
class RobotSurfaceMetric:
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    kappa_R: float
    log_kappa_R: float
    R_G: float
    sqrt_lambda_min: float
    sqrt_lambda_max: float


def orthonormal_surface_tangent(normal: np.ndarray) -> np.ndarray:
    value = np.asarray(normal, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("normal must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("normal cannot be zero")
    value = value / norm
    reference = np.eye(3)[int(np.argmin(np.abs(value)))]
    first = np.cross(reference, value)
    first /= np.linalg.norm(first)
    second = np.cross(value, first)
    return np.column_stack((first, second))


def surface_vertex_normals(surface: SurfaceInstance) -> np.ndarray:
    values = np.zeros_like(surface.vertices)
    for face, normal, area in zip(surface.faces, surface.face_normals, surface.face_areas):
        values[face] += area * normal
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("surface contains a vertex without a valid normal")
    return values / norms


def smooth_surface_normal(
    surface: SurfaceInstance,
    points: np.ndarray,
    *,
    vertex_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    projection = project_points(surface, points)
    normals = surface_vertex_normals(surface) if vertex_normals is None else np.asarray(vertex_normals)
    selected = normals[surface.faces[projection.face_indices]]
    interpolated = np.einsum("pi,pij->pj", projection.barycentric, selected)
    interpolated /= np.linalg.norm(interpolated, axis=1, keepdims=True)
    return projection.points, interpolated


def task_differential_matrix(
    tangent_basis: np.ndarray,
    tool_axis: np.ndarray,
    tool_axis_jacobian: np.ndarray,
    axis_basis: np.ndarray,
    *,
    characteristic_length: float,
) -> np.ndarray:
    tangent = np.asarray(tangent_basis, dtype=np.float64)
    axis = np.asarray(tool_axis, dtype=np.float64)
    axis_derivative = np.asarray(tool_axis_jacobian, dtype=np.float64)
    perpendicular = np.asarray(axis_basis, dtype=np.float64)
    if tangent.shape != (3, 2) or axis_derivative.shape != (3, 2) or perpendicular.shape != (3, 2):
        raise ValueError("surface and axis differentials must use [3,2] bases")
    if axis.shape != (3,) or characteristic_length <= 0.0:
        raise ValueError("invalid tool axis or characteristic length")
    axis = axis / np.linalg.norm(axis)
    angular = np.column_stack((np.cross(axis, axis_derivative[:, 0]), np.cross(axis, axis_derivative[:, 1])))
    return np.vstack((tangent / characteristic_length, perpendicular.T @ angular))


def estimate_surface_contact_differential(
    surface: SurfaceInstance,
    point_surface: np.ndarray,
    transform_base_from_surface: np.ndarray,
    axis_basis: np.ndarray,
    *,
    characteristic_length: float,
    finite_difference_step: float = 1e-5,
    tangent_basis_surface: np.ndarray | None = None,
) -> SurfaceContactDifferential:
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    transform = np.asarray(transform_base_from_surface, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape [4,4]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError("transform rotation must be orthonormal")
    vertex_normals = surface_vertex_normals(surface)
    projected, normals = smooth_surface_normal(surface, np.asarray(point_surface)[None], vertex_normals=vertex_normals)
    point = projected[0]
    normal_surface = normals[0]
    tangent_surface = orthonormal_surface_tangent(normal_surface) if tangent_basis_surface is None else np.asarray(tangent_basis_surface, dtype=np.float64)
    if tangent_surface.shape != (3, 2) or not np.allclose(tangent_surface.T @ tangent_surface, np.eye(2), atol=1e-8):
        raise ValueError("tangent_basis_surface must be orthonormal")
    offsets = []
    for column in range(2):
        direction = tangent_surface[:, column]
        _, plus_normal = smooth_surface_normal(surface, (point + finite_difference_step * direction)[None], vertex_normals=vertex_normals)
        _, minus_normal = smooth_surface_normal(surface, (point - finite_difference_step * direction)[None], vertex_normals=vertex_normals)
        offsets.append(-(rotation @ (plus_normal[0] - minus_normal[0])) / (2.0 * finite_difference_step))
    tool_axis_jacobian = np.column_stack(offsets)
    tangent_base = rotation @ tangent_surface
    normal_base = rotation @ normal_surface
    tool_axis = -normal_base
    point_base = rotation @ point + transform[:3, 3]
    differential = task_differential_matrix(
        tangent_base, tool_axis, tool_axis_jacobian, axis_basis,
        characteristic_length=characteristic_length,
    )
    return SurfaceContactDifferential(
        point, point_base, normal_base, tool_axis, tangent_base,
        tool_axis_jacobian, differential,
    )


def compute_robot_surface_metric(
    task: TaskKinematics5D,
    task_differential: np.ndarray,
    *,
    minimum_singular_value: float = 0.0,
    symmetry_tolerance: float = 1e-9,
) -> RobotSurfaceMetric:
    jacobian = np.asarray(task.normalized_jacobian_5, dtype=np.float64)
    differential = np.asarray(task_differential, dtype=np.float64)
    if jacobian.shape[0] != 5 or differential.shape != (5, 2):
        raise ValueError("expected a [5,nq] Jacobian and [5,2] task differential")
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] + 1e-12 < minimum_singular_value:
        raise ValueError("task Jacobian is below the admitted threshold")
    gram = jacobian @ jacobian.T
    try:
        factor = np.linalg.cholesky(gram)
        solved = np.linalg.solve(factor.T, np.linalg.solve(factor, differential))
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(gram, differential, rcond=None)[0]
    matrix = differential.T @ solved
    matrix = 0.5 * (matrix + matrix.T)
    if not np.allclose(matrix, matrix.T, atol=symmetry_tolerance):
        raise ValueError("robot surface metric is not symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise ValueError("robot surface metric must be positive definite")
    kappa = float(eigenvalues[1] / eigenvalues[0])
    return RobotSurfaceMetric(
        matrix=matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kappa_R=kappa,
        log_kappa_R=float(np.log(kappa)),
        R_G=float(np.sqrt(kappa)),
        sqrt_lambda_min=float(np.sqrt(eigenvalues[0])),
        sqrt_lambda_max=float(np.sqrt(eigenvalues[1])),
    )
