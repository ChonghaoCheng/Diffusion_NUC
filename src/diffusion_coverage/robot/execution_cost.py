from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointExecutionCost:
    weighted_joint_length: float
    unweighted_joint_travel: float
    per_joint_travel: np.ndarray
    max_per_joint_travel: float
    q_sample_count: int


def compute_joint_execution_cost(
    q_segments: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    weight_matrix: np.ndarray | None = None,
) -> JointExecutionCost:
    """Integrate joint travel along validated witnesses without angle wrapping."""

    if not q_segments:
        raise ValueError("at least one q segment is required")
    segments = [np.asarray(segment, dtype=np.float64) for segment in q_segments]
    nq = segments[0].shape[1] if segments[0].ndim == 2 else 0
    if nq < 1 or any(segment.ndim != 2 or segment.shape[1] != nq or len(segment) < 2 for segment in segments):
        raise ValueError("every q segment must have shape [T, nq] with T >= 2")
    weight = np.eye(nq, dtype=np.float64) if weight_matrix is None else np.asarray(weight_matrix, dtype=np.float64)
    if weight.shape != (nq, nq) or not np.allclose(weight, weight.T, atol=1e-12):
        raise ValueError("weight_matrix must be symmetric with shape [nq, nq]")
    if np.linalg.eigvalsh(weight).min() <= 0.0:
        raise ValueError("weight_matrix must be positive definite")

    differences = np.concatenate([np.diff(segment, axis=0) for segment in segments], axis=0)
    weighted_steps = np.sqrt(np.einsum("ti,ij,tj->t", differences, weight, differences))
    per_joint = np.abs(differences).sum(axis=0)
    return JointExecutionCost(
        weighted_joint_length=float(weighted_steps.sum()),
        unweighted_joint_travel=float(np.linalg.norm(differences, axis=1).sum()),
        per_joint_travel=per_joint,
        max_per_joint_travel=float(per_joint.max(initial=0.0)),
        q_sample_count=int(sum(len(segment) for segment in segments)),
    )
