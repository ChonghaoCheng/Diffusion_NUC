from __future__ import annotations

import numpy as np


def minimal_axis_rotation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine >= 0.0:
            return np.zeros(3, dtype=np.float64)
        reference = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, reference)
        axis /= np.linalg.norm(axis)
        return np.pi * axis
    return np.arctan2(sine, cosine) * cross / sine


def local_task_increment(
    position_first: np.ndarray,
    axis_first: np.ndarray,
    position_second: np.ndarray,
    axis_second: np.ndarray,
    axis_basis: np.ndarray,
    *,
    characteristic_length: float,
) -> np.ndarray:
    if characteristic_length <= 0.0:
        raise ValueError("characteristic_length must be positive")
    basis = np.asarray(axis_basis, dtype=np.float64)
    if basis.shape != (3, 2):
        raise ValueError("axis_basis must have shape [3, 2]")
    rotation = minimal_axis_rotation(axis_first, axis_second)
    return np.concatenate((
        (np.asarray(position_second) - np.asarray(position_first)) / characteristic_length,
        basis.T @ rotation,
    ))


def predicted_local_execution_length(
    normalized_jacobian_5: np.ndarray,
    delta_y_bar: np.ndarray,
    *,
    minimum_singular_value: float = 0.0,
) -> float:
    jacobian = np.asarray(normalized_jacobian_5, dtype=np.float64)
    delta = np.asarray(delta_y_bar, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != 5 or delta.shape != (5,):
        raise ValueError("expected a [5, nq] Jacobian and a five-vector increment")
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] + 1e-12 < minimum_singular_value:
        raise ValueError("task Jacobian is below the admitted singular-value threshold")
    gram = jacobian @ jacobian.T
    try:
        factor = np.linalg.cholesky(gram)
        solved = np.linalg.solve(factor, delta)
        value = float(np.dot(solved, solved))
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(jacobian, delta, rcond=None)[0]
        value = float(np.dot(solved, solved))
    return float(np.sqrt(max(value, 0.0)))
