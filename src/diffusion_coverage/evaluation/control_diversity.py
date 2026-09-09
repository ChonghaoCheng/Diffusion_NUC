from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ControlClusterResult:
    assignments: np.ndarray
    centers: tuple[np.ndarray, ...]
    normalized_entropy: float

    @property
    def num_clusters(self) -> int:
        return len(self.centers)


def control_rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    """RMS waypoint distance for controls already expressed in physical units."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 2:
        raise ValueError("control arrays must have the same [M, D] shape")
    return float(np.sqrt(np.mean(np.sum((left_array - right_array) ** 2, axis=1))))


def cluster_controls(
    controls: list[np.ndarray],
    *,
    threshold: float,
) -> ControlClusterResult:
    """Deterministic leader clustering under RMS physical control distance."""
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if not controls:
        return ControlClusterResult(np.empty(0, dtype=np.int64), (), 0.0)
    centers: list[np.ndarray] = []
    assignments: list[int] = []
    for control in controls:
        value = np.asarray(control, dtype=np.float64)
        if not centers:
            centers.append(value)
            assignments.append(0)
            continue
        distances = [control_rms_distance(value, center) for center in centers]
        nearest = int(np.argmin(distances))
        if distances[nearest] > threshold:
            centers.append(value)
            assignments.append(len(centers) - 1)
        else:
            assignments.append(nearest)
    assignment_array = np.asarray(assignments, dtype=np.int64)
    counts = np.bincount(assignment_array, minlength=len(centers)).astype(np.float64)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized = entropy / math.log(len(centers)) if len(centers) > 1 else 0.0
    return ControlClusterResult(assignment_array, tuple(centers), normalized)


def expert_control_coverage(
    generated: list[np.ndarray],
    experts: list[np.ndarray],
    *,
    threshold: float,
) -> tuple[float, float | None]:
    """Return covered expert fraction and mean nearest generated distance."""
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if not experts:
        raise ValueError("at least one expert control is required")
    if not generated:
        return 0.0, None
    nearest = [
        min(control_rms_distance(expert, sample) for sample in generated)
        for expert in experts
    ]
    return float(np.mean(np.asarray(nearest) <= threshold)), float(np.mean(nearest))


def leave_one_out_control_coverage(
    controls: list[np.ndarray],
    *,
    threshold: float,
) -> float:
    """Coverage of each control by another sample from the same finite set."""
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if len(controls) < 2:
        return 0.0
    covered = []
    for index, control in enumerate(controls):
        nearest = min(
            control_rms_distance(control, other)
            for other_index, other in enumerate(controls)
            if other_index != index
        )
        covered.append(nearest <= threshold)
    return float(np.mean(covered))


def control_energy_distance(left: list[np.ndarray], right: list[np.ndarray]) -> float | None:
    """Biased finite-sample energy distance under RMS control geometry."""
    if not left or not right:
        return None
    cross = np.mean([
        control_rms_distance(a, b) for a in left for b in right
    ])
    within_left = np.mean([
        control_rms_distance(a, b) for a in left for b in left
    ])
    within_right = np.mean([
        control_rms_distance(a, b) for a in right for b in right
    ])
    return float(max(0.0, 2.0 * cross - within_left - within_right))
