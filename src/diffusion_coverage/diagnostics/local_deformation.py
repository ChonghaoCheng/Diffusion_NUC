from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from diffusion_coverage.geometry.robot_surface_metric import (
    compute_robot_surface_metric,
    estimate_surface_contact_differential,
    orthonormal_surface_tangent,
    smooth_surface_normal,
    surface_vertex_normals,
)
from diffusion_coverage.geometry.surface_curve import trace_surface_curve
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class DeformedSurfacePath:
    points: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray
    surface_length: float
    maximum_turn: float
    topology_preserved: bool


@dataclass(frozen=True)
class AdmissionLimits:
    maximum_relative_surface_length_change: float
    delta_nuc: float
    maximum_terminal_q_mismatch: float
    sigma_safe: float


def cumulative_length(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("points must have shape [N,3] with N >= 2")
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))


def window_indices(cumulative: np.ndarray, centre: int, length: float) -> tuple[int, int] | None:
    arc = np.asarray(cumulative, dtype=np.float64)
    if not 0 <= centre < len(arc) or length <= 0.0:
        raise ValueError("invalid centre or window length")
    half = 0.5 * length
    if arc[centre] - half < arc[0] or arc[centre] + half > arc[-1]:
        return None
    start = int(np.argmin(np.abs(arc - (arc[centre] - half))))
    stop = int(np.argmin(np.abs(arc - (arc[centre] + half))))
    return None if stop <= start + 6 else (start, stop)


def control_indices(cumulative: np.ndarray, start: int, stop: int, count: int = 7) -> np.ndarray:
    if count < 2 or stop <= start:
        raise ValueError("invalid control request")
    arc = np.asarray(cumulative, dtype=np.float64)
    targets = np.linspace(arc[start], arc[stop], count)
    indices = np.asarray([start + np.argmin(np.abs(arc[start : stop + 1] - value)) for value in targets], dtype=np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError("window does not contain enough distinct control samples")
    return indices


def deterministic_window_seed(base_seed: int, window_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{window_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def proposal_schedule(evaluations: int, radii: list[float], per_radius: int, seed: int) -> list[tuple[int, float, float]]:
    if evaluations < 1 or per_radius < 1 or not radii:
        raise ValueError("invalid proposal schedule")
    base = [(axis, sign) for axis in range(6) for sign in (-1.0, 1.0)]
    rng = np.random.default_rng(seed)
    output: list[tuple[int, float, float]] = []
    for evaluation in range(evaluations):
        if evaluation % len(base) == 0:
            rng.shuffle(base)
        axis, sign = base[evaluation % len(base)]
        radius = radii[min(evaluation // per_radius, len(radii) - 1)]
        output.append((axis, sign, float(radius)))
    return output


def propose_parameters(parameters: np.ndarray, proposal: tuple[int, float, float], maximum_norm: float) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64).reshape(3, 2).copy()
    axis, sign, radius = proposal
    values.flat[axis] += sign * radius
    norm = float(np.linalg.norm(values[axis // 2]))
    if norm > maximum_norm:
        values[axis // 2] *= maximum_norm / norm
    return values


def deform_surface_path(
    surface: SurfaceInstance,
    baseline_points: np.ndarray,
    controls: np.ndarray,
    parameters: np.ndarray,
    *,
    maximum_displacement: float,
    maximum_retraction_step: float,
) -> DeformedSurfacePath:
    points = np.asarray(baseline_points, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.int64)
    values = np.asarray(parameters, dtype=np.float64).reshape(3, 2)
    if len(controls) != 7 or not np.array_equal(controls, np.sort(controls)):
        raise ValueError("seven ordered controls are required")
    if controls[0] != 0 or controls[-1] != len(points) - 1:
        raise ValueError("controls must include both path endpoints")
    if np.any(np.linalg.norm(values, axis=1) > maximum_displacement + 1e-12):
        raise ValueError("free-control displacement exceeds frozen bound")

    vertex_normals = surface_vertex_normals(surface)
    projected, normals = smooth_surface_normal(surface, points, vertex_normals=vertex_normals)
    baseline_projection = project_points(surface, points)
    control_offsets = np.zeros((7, 3), dtype=np.float64)
    for row, control in enumerate((2, 3, 4)):
        basis = orthonormal_surface_tangent(normals[controls[control]])
        control_offsets[control] = basis @ values[row]

    arc = cumulative_length(projected)
    control_arc = arc[controls]
    deformed = projected.copy()
    for index in range(1, len(points) - 1):
        segment = min(int(np.searchsorted(control_arc, arc[index], side="right") - 1), 5)
        segment = max(segment, 0)
        denominator = max(control_arc[segment + 1] - control_arc[segment], 1e-15)
        fraction = (arc[index] - control_arc[segment]) / denominator
        offset = (1.0 - fraction) * control_offsets[segment] + fraction * control_offsets[segment + 1]
        offset -= normals[index] * float(np.dot(normals[index], offset))
        distance = float(np.linalg.norm(offset))
        if distance > 1e-12:
            curve = trace_surface_curve(
                surface, projected[index], offset / distance, distance,
                maximum_step=maximum_retraction_step,
            )
            deformed[index] = curve.points[-1]
    # Fixed controls are restored exactly after numerical retraction bookkeeping.
    deformed[controls[[0, 1, 5, 6]]] = projected[controls[[0, 1, 5, 6]]]
    deformed, deformed_normals = smooth_surface_normal(surface, deformed, vertex_normals=vertex_normals)
    fixed = controls[[0, 1, 5, 6]]
    deformed[fixed] = points[fixed]
    _, fixed_normals = smooth_surface_normal(surface, points[fixed], vertex_normals=vertex_normals)
    deformed_normals[fixed] = fixed_normals
    projection = project_points(surface, deformed)
    adjacency = _face_adjacency(surface)
    corridor_ok = all(
        _within_face_hops(int(reference), int(candidate), adjacency, 2)
        for candidate, reference in zip(projection.face_indices, baseline_projection.face_indices)
    )
    connected = all(
        _within_face_hops(int(a), int(b), adjacency, 2)
        for a, b in zip(projection.face_indices[:-1], projection.face_indices[1:])
    )
    return DeformedSurfacePath(
        points=deformed,
        normals=deformed_normals,
        face_indices=projection.face_indices,
        surface_length=float(cumulative_length(deformed)[-1]),
        maximum_turn=maximum_turn_angle(deformed),
        topology_preserved=bool(corridor_ok and connected),
    )


def maximum_turn_angle(points: np.ndarray) -> float:
    first = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    norms = np.linalg.norm(first, axis=1)
    valid = (norms[:-1] > 1e-12) & (norms[1:] > 1e-12)
    if not np.any(valid):
        return 0.0
    cosine = np.sum(first[:-1] * first[1:], axis=1) / np.maximum(norms[:-1] * norms[1:], 1e-15)
    return float(np.max(np.arccos(np.clip(cosine[valid], -1.0, 1.0))))


def terminal_q_mismatch(q_end: np.ndarray, baseline_q_end: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(q_end, dtype=np.float64) - np.asarray(baseline_q_end, dtype=np.float64))))


def hard_admission(
    *,
    surface_length: float,
    baseline_surface_length: float,
    nuc_error: float,
    missed_error: float,
    repeat_error: float,
    baseline_nuc_error: float,
    baseline_missed_error: float,
    baseline_repeat_error: float,
    terminal_mismatch: float,
    strict_pass: bool,
    topology_preserved: bool,
    limits: AdmissionLimits,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if not topology_preserved:
        reasons.append("topology")
    if abs(surface_length - baseline_surface_length) / baseline_surface_length > limits.maximum_relative_surface_length_change:
        reasons.append("surface_length")
    if nuc_error > baseline_nuc_error + limits.delta_nuc:
        reasons.append("coverage_nuc")
    if missed_error > baseline_missed_error + limits.delta_nuc:
        reasons.append("coverage_miss")
    if repeat_error > baseline_repeat_error + limits.delta_nuc:
        reasons.append("coverage_repeat")
    if terminal_mismatch > limits.maximum_terminal_q_mismatch:
        reasons.append("terminal_q")
    if not strict_pass:
        reasons.append("strict_robot")
    return not reasons, tuple(reasons)


def integrated_robot_metric_length(
    robot,
    surface: SurfaceInstance,
    transform: np.ndarray,
    q_path: np.ndarray,
    points_surface: np.ndarray,
    *,
    characteristic_length: float,
    sigma_safe: float,
    finite_difference_step: float,
) -> float:
    q = np.asarray(q_path, dtype=np.float64)
    points = np.asarray(points_surface, dtype=np.float64)
    if len(q) != len(points):
        raise ValueError("q witness and surface path must have equal lengths")
    total = 0.0
    for index in range(len(q) - 1):
        task = evaluate_task_kinematics_5d(robot, q[index], characteristic_length=characteristic_length)
        contact = estimate_surface_contact_differential(
            surface, points[index], transform, task.axis_basis,
            characteristic_length=characteristic_length,
            finite_difference_step=finite_difference_step,
        )
        metric = compute_robot_surface_metric(task, contact.task_differential, minimum_singular_value=sigma_safe)
        delta_base = transform[:3, :3] @ (points[index + 1] - points[index])
        delta_xi = contact.tangent_basis.T @ delta_base
        total += float(np.sqrt(max(delta_xi @ metric.matrix @ delta_xi, 0.0)))
    return total


def euclidean_objective(surface_length: float, metric_callback=None) -> float:
    """M1 objective; metric_callback is deliberately ignored by contract."""
    return float(surface_length)


def accept_incumbent_update(
    parameters: np.ndarray,
    objective: float,
    candidate_parameters: np.ndarray,
    candidate_objective: float | None,
    *,
    admitted: bool,
    accepted_count: int,
    maximum_accepted: int,
    improvement_tolerance: float,
) -> tuple[np.ndarray, float, bool]:
    accept = bool(
        admitted
        and candidate_objective is not None
        and accepted_count < maximum_accepted
        and candidate_objective < objective - improvement_tolerance
    )
    if not accept:
        return np.asarray(parameters, dtype=np.float64).copy(), float(objective), False
    return np.asarray(candidate_parameters, dtype=np.float64).copy(), float(candidate_objective), True


def _face_adjacency(surface: SurfaceInstance) -> list[set[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(surface.faces):
        for first, second in zip(face, np.roll(face, -1)):
            edge_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)
    adjacency = [set() for _ in range(surface.num_faces)]
    for faces in edge_faces.values():
        for first in faces:
            adjacency[first].update(second for second in faces if second != first)
    return adjacency


def _within_face_hops(source: int, target: int, adjacency: list[set[int]], maximum_hops: int) -> bool:
    if source == target:
        return True
    frontier = {source}
    visited = {source}
    for _ in range(maximum_hops):
        frontier = {neighbor for face in frontier for neighbor in adjacency[face]} - visited
        if target in frontier:
            return True
        visited.update(frontier)
    return False
