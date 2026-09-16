from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np


UR5E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class IKCandidate:
    q: np.ndarray
    position_error: float
    axis_error: float
    manipulability: float
    joint_limit_margin: float
    collision_free: bool


@dataclass(frozen=True)
class LiftabilityResult:
    feasible: bool
    q_path: np.ndarray | None
    failure_reason: str | None
    candidate_counts: tuple[int, ...]
    search_edges: int
    min_manipulability: float | None
    min_joint_limit_margin: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskTransitionResult:
    feasible: bool
    q_path: np.ndarray
    failure_reason: str | None
    max_position_error: float
    max_axis_error: float


class UR5eKinematics:
    """MuJoCo-backed UR5e IK and continuous-lift checker with no rendering dependency."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        site_name: str = "attachment_site",
        tool_axis_index: int = 2,
        tool_axis_sign: float = 1.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self.site_id < 0 or self.model.nq != 6 or self.model.nv != 6:
            raise ValueError("model must expose a six-DOF UR5e and the requested tool site")
        if tool_axis_index not in (0, 1, 2) or tool_axis_sign not in (-1.0, 1.0):
            raise ValueError("invalid tool-axis convention")
        self.tool_axis_index = tool_axis_index
        self.tool_axis_sign = tool_axis_sign
        self.lower_limits = self.model.jnt_range[:, 0].copy()
        self.upper_limits = self.model.jnt_range[:, 1].copy()
        self.home = np.asarray(
            (-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0),
            dtype=np.float64,
        )

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        axis = self.tool_axis_sign * rotation[:, self.tool_axis_index]
        return self.data.site_xpos[self.site_id].copy(), axis.copy()

    def evaluate_configuration(self, q: np.ndarray) -> IKCandidate:
        """Return collision, conditioning, and joint-margin metrics at one configuration."""

        value = np.asarray(q, dtype=np.float64)
        if value.shape != (6,):
            raise ValueError("configuration must have shape [6]")
        self.forward(value)
        return self._candidate(value, 0.0, 0.0)

    def solve_ik(
        self,
        target_position: np.ndarray,
        target_axis: np.ndarray,
        initial_q: np.ndarray,
        *,
        position_tolerance: float = 1e-3,
        axis_tolerance: float = np.deg2rad(5.0),
        max_iterations: int = 120,
        damping: float = 2e-3,
        max_update: float = 0.25,
        constrain_limits: bool = True,
        backend: str = "legacy6",
    ) -> IKCandidate | None:
        """Solve the free-roll position/tool-axis task.

        ``legacy6`` preserves the historical implementation and its default.  ``task5``
        projects angular velocity and the cross-product residual onto the tangent plane
        of the current tool axis, so rotation about that axis is not penalized.
        """
        if backend not in {"legacy6", "task5"}:
            raise ValueError("backend must be 'legacy6' or 'task5'")
        target = np.asarray(target_position, dtype=np.float64)
        axis = np.asarray(target_axis, dtype=np.float64)
        if target.shape != (3,) or axis.shape != (3,):
            raise ValueError("target position and axis must be 3-vectors")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            raise ValueError("target axis cannot be zero")
        axis = axis / axis_norm
        q = np.asarray(initial_q, dtype=np.float64).copy()
        if q.shape != (6,):
            raise ValueError("initial_q must have shape [6]")
        if constrain_limits:
            q = np.clip(q, self.lower_limits, self.upper_limits)

        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(max_iterations):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            position = self.data.site_xpos[self.site_id]
            rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            current_axis = self.tool_axis_sign * rotation[:, self.tool_axis_index]
            position_error_vector = target - position
            axis_error_vector = np.cross(current_axis, axis)
            position_error = float(np.linalg.norm(position_error_vector))
            axis_error = float(np.arccos(np.clip(np.dot(current_axis, axis), -1.0, 1.0)))
            if position_error <= position_tolerance and axis_error <= axis_tolerance:
                return self._candidate(q, position_error, axis_error)
            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                self.site_id,
            )
            if backend == "legacy6":
                jacobian = np.vstack((jacobian_position, jacobian_rotation))
                residual = np.concatenate((position_error_vector, axis_error_vector))
            else:
                # Import locally to avoid a module-level cycle: task_kinematics imports
                # UR5eKinematics for its public evaluation helper.
                from diffusion_coverage.robot.task_kinematics import orthonormal_axis_basis

                basis = orthonormal_axis_basis(current_axis)
                jacobian = np.vstack((jacobian_position, basis.T @ jacobian_rotation))
                residual = np.concatenate((position_error_vector, basis.T @ axis_error_vector))
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0]), residual
            )
            update_norm = float(np.linalg.norm(update))
            if update_norm > max_update:
                update *= max_update / update_norm
            q += update
            if constrain_limits:
                q = np.clip(q, self.lower_limits, self.upper_limits)
        return None

    def solve_position_ik(
        self,
        target_position: np.ndarray,
        initial_q: np.ndarray,
        *,
        position_tolerance: float = 1e-3,
        max_iterations: int = 120,
        damping: float = 2e-3,
        constrain_limits: bool = True,
    ) -> np.ndarray | None:
        target = np.asarray(target_position, dtype=np.float64)
        q = np.asarray(initial_q, dtype=np.float64).copy()
        if target.shape != (3,) or q.shape != (6,):
            raise ValueError("target_position and initial_q have invalid shapes")
        if constrain_limits:
            q = np.clip(q, self.lower_limits, self.upper_limits)
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(max_iterations):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            residual = target - self.data.site_xpos[self.site_id]
            if float(np.linalg.norm(residual)) <= position_tolerance:
                return q
            mujoco.mj_jacSite(
                self.model, self.data, jacobian_position, jacobian_rotation, self.site_id
            )
            update = jacobian_position.T @ np.linalg.solve(
                jacobian_position @ jacobian_position.T + damping**2 * np.eye(3), residual
            )
            update_norm = float(np.linalg.norm(update))
            if update_norm > 0.25:
                update *= 0.25 / update_norm
            q += update
            if constrain_limits:
                q = np.clip(q, self.lower_limits, self.upper_limits)
        return None

    def enumerate_ik(
        self,
        target_position: np.ndarray,
        target_axis: np.ndarray,
        *,
        seed_configurations: Sequence[np.ndarray] = (),
        random_restarts: int = 12,
        rng: np.random.Generator | None = None,
        axis_tolerance: float = np.deg2rad(5.0),
        minimum_manipulability: float = 1e-5,
        require_collision_free: bool = True,
        max_candidates: int = 64,
        orientation_cone_samples: int = 1,
        inner_cone_tolerance: float | None = None,
        inner_cone_samples: int = 1,
        inner_max_candidates: int | None = None,
    ) -> list[IKCandidate]:
        generator = np.random.default_rng() if rng is None else rng
        if orientation_cone_samples < 1:
            raise ValueError("orientation_cone_samples must be positive")
        if inner_cone_tolerance is not None:
            if not 0.0 <= inner_cone_tolerance < axis_tolerance:
                raise ValueError("inner cone tolerance must be inside the outer cone")
            if inner_cone_samples < 1:
                raise ValueError("inner_cone_samples must be positive")
            inner_limit = (
                max_candidates if inner_max_candidates is None else inner_max_candidates
            )
            if not 1 <= inner_limit <= max_candidates:
                raise ValueError("inner_max_candidates must be within the total budget")
            inner = self.enumerate_ik(
                target_position,
                target_axis,
                seed_configurations=seed_configurations,
                random_restarts=random_restarts,
                rng=generator,
                axis_tolerance=inner_cone_tolerance,
                minimum_manipulability=minimum_manipulability,
                require_collision_free=require_collision_free,
                max_candidates=inner_limit,
                orientation_cone_samples=inner_cone_samples,
            )
            outer = self.enumerate_ik(
                target_position,
                target_axis,
                seed_configurations=seed_configurations,
                random_restarts=random_restarts,
                rng=generator,
                axis_tolerance=axis_tolerance,
                minimum_manipulability=minimum_manipulability,
                require_collision_free=require_collision_free,
                max_candidates=max_candidates,
                orientation_cone_samples=orientation_cone_samples,
            )
            combined = list(inner)
            for candidate in outer:
                if any(
                    _configuration_distance(candidate.q, existing.q) < 1.5e-1
                    for existing in combined
                ):
                    continue
                combined.append(candidate)
                if len(combined) == max_candidates:
                    break
            return combined
        central_axis = np.asarray(target_axis, dtype=np.float64)
        central_axis /= np.linalg.norm(central_axis)
        sampled_axes = sample_axis_cone(
            central_axis, axis_tolerance, orientation_cone_samples
        )
        jobs = [(central_axis, np.asarray(seed)) for seed in seed_configurations]
        jobs.extend((axis, self.home) for axis in sampled_axes)
        jobs.extend(
            (
                sampled_axes[restart % len(sampled_axes)],
                generator.uniform(self.lower_limits, self.upper_limits),
            )
            for restart in range(random_restarts)
        )
        candidates: list[IKCandidate] = []
        convergence_tolerance = (
            axis_tolerance
            if orientation_cone_samples == 1
            else min(axis_tolerance, np.deg2rad(1.0))
        )
        for sampled_axis, seed in jobs:
            candidate = self.solve_ik(
                target_position,
                sampled_axis,
                seed,
                axis_tolerance=convergence_tolerance,
            )
            if candidate is None:
                continue
            position, achieved_axis = self.forward(candidate.q)
            original_axis_error = float(
                np.arccos(np.clip(np.dot(achieved_axis, central_axis), -1.0, 1.0))
            )
            if original_axis_error > axis_tolerance + 1e-12:
                continue
            candidate = self._candidate(
                candidate.q,
                float(np.linalg.norm(position - target_position)),
                original_axis_error,
            )
            if require_collision_free and not candidate.collision_free:
                continue
            if candidate.manipulability < minimum_manipulability:
                continue
            if any(_configuration_distance(candidate.q, existing.q) < 1.5e-1 for existing in candidates):
                continue
            candidates.append(candidate)
        if len(candidates) <= max_candidates:
            return candidates
        return _farthest_configuration_subset(candidates, max_candidates)

    def check_continuous_lift(
        self,
        target_positions: np.ndarray,
        target_axes: np.ndarray,
        *,
        axis_tolerance: float = np.deg2rad(5.0),
        random_restarts: int = 12,
        maximum_joint_step: float = 0.7,
        minimum_manipulability: float = 1e-5,
        max_active_branches: int = 12,
        rng: np.random.Generator | None = None,
    ) -> LiftabilityResult:
        positions = np.asarray(target_positions, dtype=np.float64)
        axes = np.asarray(target_axes, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or axes.shape != positions.shape:
            raise ValueError("target positions and axes must have shape [T, 3]")
        generator = np.random.default_rng() if rng is None else rng
        if max_active_branches < 1:
            raise ValueError("max_active_branches must be positive")
        layers: list[list[IKCandidate]] = []
        predecessors: list[np.ndarray] = []
        search_edges = 0
        for waypoint_index, (position, axis) in enumerate(zip(positions, axes)):
            if not layers:
                candidates = self.enumerate_ik(
                    position, axis, random_restarts=random_restarts, rng=generator,
                    axis_tolerance=axis_tolerance,
                    minimum_manipulability=minimum_manipulability,
                )
                if not candidates:
                    reason = self._diagnose_pose_failure(position, axis, axis_tolerance, generator)
                    return LiftabilityResult(
                        False, None, reason, (0,), search_edges, None, None,
                        {"failed_waypoint": waypoint_index},
                    )
                parent = np.full(len(candidates), -1, dtype=np.int64)
            else:
                previous = layers[-1]
                candidates = []
                parent_list: list[int] = []
                minimum_joint_jump = np.inf
                rejected_collision = 0
                rejected_singularity = 0
                rejected_joint_step = 0
                # Preserve branch identity by propagating every reachable parent first.
                for prior_index, prior in enumerate(previous):
                    candidate = self.solve_ik(
                        position, axis, prior.q, axis_tolerance=axis_tolerance
                    )
                    if candidate is None or not candidate.collision_free or candidate.manipulability < minimum_manipulability:
                        continue
                    search_edges += 1
                    adjusted = _nearest_equivalent(
                        candidate.q, prior.q, self.lower_limits, self.upper_limits
                    )
                    joint_jump = float(np.max(np.abs(adjusted - prior.q)))
                    minimum_joint_jump = min(minimum_joint_jump, joint_jump)
                    if joint_jump > maximum_joint_step:
                        rejected_joint_step += 1
                        continue
                    edge_reason = self._edge_invalid_reason(
                        prior.q, adjusted, minimum_manipulability
                    )
                    if edge_reason == "collision":
                        rejected_collision += 1
                        continue
                    if edge_reason == "singularity":
                        rejected_singularity += 1
                        continue
                    propagated = IKCandidate(
                        adjusted, candidate.position_error, candidate.axis_error,
                        candidate.manipulability, candidate.joint_limit_margin,
                        candidate.collision_free,
                    )
                    if any(_configuration_distance(propagated.q, existing.q) < 5e-2 for existing in candidates):
                        continue
                    candidates.append(propagated)
                    parent_list.append(prior_index)

                # Independent restarts can recover branches a local continuation missed.
                independent = (
                    self.enumerate_ik(
                        position, axis,
                        random_restarts=max(2, random_restarts // 4), rng=generator,
                        axis_tolerance=axis_tolerance,
                        minimum_manipulability=minimum_manipulability,
                    )
                    if len(candidates) < 4
                    else []
                )
                for candidate in independent:
                    if any(_configuration_distance(candidate.q, existing.q) < 1.5e-1 for existing in candidates):
                        continue
                    best_edge: tuple[float, int, np.ndarray] | None = None
                    for prior_index, prior in enumerate(previous):
                        search_edges += 1
                        adjusted = _nearest_equivalent(
                            candidate.q, prior.q, self.lower_limits, self.upper_limits
                        )
                        joint_jump = float(np.max(np.abs(adjusted - prior.q)))
                        minimum_joint_jump = min(minimum_joint_jump, joint_jump)
                        if joint_jump > maximum_joint_step:
                            rejected_joint_step += 1
                            continue
                        edge_reason = self._edge_invalid_reason(
                            prior.q, adjusted, minimum_manipulability
                        )
                        if edge_reason == "collision":
                            rejected_collision += 1
                            continue
                        if edge_reason == "singularity":
                            rejected_singularity += 1
                            continue
                        if best_edge is None or joint_jump < best_edge[0]:
                            best_edge = (joint_jump, prior_index, adjusted)
                    if best_edge is not None:
                        _, prior_index, adjusted = best_edge
                        candidates.append(
                            IKCandidate(
                                adjusted, candidate.position_error, candidate.axis_error,
                                candidate.manipulability, candidate.joint_limit_margin,
                                candidate.collision_free,
                            )
                        )
                        parent_list.append(prior_index)
                parent = np.asarray(parent_list, dtype=np.int64)

            if not candidates:
                if independent:
                    # A finite continuation search cannot certify that the
                    # underlying configuration-space components are disconnected.
                    reason = "continuous_ik_search_failure"
                else:
                    reason = self._diagnose_pose_failure(
                        position, axis, axis_tolerance, generator
                    )
                return LiftabilityResult(
                    False, None, reason,
                    tuple(len(layer) for layer in layers) + (len(independent),),
                    search_edges, None, None, {
                        "failed_waypoint": waypoint_index,
                        "minimum_joint_jump": float(minimum_joint_jump),
                        "rejected_joint_step": rejected_joint_step,
                        "rejected_collision": rejected_collision,
                        "rejected_singularity": rejected_singularity,
                    },
                )
            if len(candidates) > max_active_branches:
                selected = _farthest_configuration_indices(
                    candidates, max_active_branches
                )
                candidates = [candidates[index] for index in selected]
                parent = parent[selected]
            layers.append(candidates)
            predecessors.append(parent)

        final_index = 0
        selected = [final_index]
        for layer_index in range(len(layers) - 1, 0, -1):
            selected.append(int(predecessors[layer_index][selected[-1]]))
        selected.reverse()
        q_path = np.asarray([layer[index].q for layer, index in zip(layers, selected)])
        selected_candidates = [layer[index] for layer, index in zip(layers, selected)]
        return LiftabilityResult(
            True,
            q_path,
            None,
            tuple(len(layer) for layer in layers),
            search_edges,
            min(candidate.manipulability for candidate in selected_candidates),
            min(candidate.joint_limit_margin for candidate in selected_candidates),
            {},
        )

    def check_transition(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        maximum_joint_step: float = 0.8,
        minimum_manipulability: float = 1e-5,
        allow_equivalent_end: bool = True,
        check_endpoints: bool = True,
    ) -> tuple[bool, np.ndarray, str | None]:
        """Check one local IK-candidate edge using the continuation checker contract."""

        start_q = np.asarray(start, dtype=np.float64)
        end_q = np.asarray(end, dtype=np.float64)
        if start_q.shape != (6,) or end_q.shape != (6,):
            raise ValueError("transition configurations must have shape [6]")
        adjusted = (
            _nearest_equivalent(end_q, start_q, self.lower_limits, self.upper_limits)
            if allow_equivalent_end
            else end_q.copy()
        )
        if float(np.max(np.abs(adjusted - start_q))) > maximum_joint_step:
            return False, adjusted, "joint_step"
        if check_endpoints:
            for endpoint in (start_q, adjusted):
                candidate = self.evaluate_configuration(endpoint)
                if not candidate.collision_free:
                    return False, adjusted, "collision"
                if candidate.manipulability < minimum_manipulability:
                    return False, adjusted, "singularity"
        reason = self._edge_invalid_reason(
            start_q, adjusted, minimum_manipulability
        )
        return reason is None, adjusted, reason

    def check_task_transition(
        self,
        start: np.ndarray,
        end: np.ndarray,
        target_positions: np.ndarray,
        target_axes: np.ndarray,
        *,
        maximum_joint_step: float = 0.8,
        minimum_manipulability: float = 1e-5,
        position_tolerance: float = 3e-3,
        axis_tolerance: float = np.deg2rad(3.0),
        allow_equivalent_end: bool = True,
        check_endpoints: bool = True,
    ) -> TaskTransitionResult:
        """Require a local q interpolation to track a sampled surface task edge."""

        positions = np.asarray(target_positions, dtype=np.float64)
        axes = np.asarray(target_axes, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or axes.shape != positions.shape:
            raise ValueError("target positions and axes must have shape [T, 3]")
        if len(positions) < 2:
            raise ValueError("a task transition requires at least two target poses")
        axis_norms = np.linalg.norm(axes, axis=1)
        if np.any(axis_norms <= 1e-12):
            raise ValueError("target axes must be nonzero")
        axes = axes / axis_norms[:, None]
        valid, adjusted, reason = self.check_transition(
            start,
            end,
            maximum_joint_step=maximum_joint_step,
            minimum_manipulability=minimum_manipulability,
            allow_equivalent_end=allow_equivalent_end,
            check_endpoints=check_endpoints,
        )
        fractions = np.linspace(0.0, 1.0, len(positions))
        start_q = np.asarray(start, dtype=np.float64)
        q_path = (1.0 - fractions[:, None]) * start_q + fractions[:, None] * adjusted
        if not valid:
            return TaskTransitionResult(False, q_path, reason, np.inf, np.inf)

        position_errors = np.empty(len(q_path), dtype=np.float64)
        axis_errors = np.empty(len(q_path), dtype=np.float64)
        for index, q in enumerate(q_path):
            achieved_position, achieved_axis = self.forward(q)
            position_errors[index] = np.linalg.norm(achieved_position - positions[index])
            axis_errors[index] = np.arccos(
                np.clip(np.dot(achieved_axis, axes[index]), -1.0, 1.0)
            )
        max_position_error = float(position_errors.max())
        max_axis_error = float(axis_errors.max())
        if max_position_error > position_tolerance:
            reason = "surface_tracking"
        elif max_axis_error > axis_tolerance:
            reason = "axis_tracking"
        else:
            reason = None
        return TaskTransitionResult(
            reason is None,
            q_path,
            reason,
            max_position_error,
            max_axis_error,
        )

    def continue_task_transition(
        self,
        start: np.ndarray,
        target_positions: np.ndarray,
        target_axes: np.ndarray,
        *,
        maximum_joint_step: float = 0.8,
        minimum_manipulability: float = 1e-5,
        position_tolerance: float = 3e-3,
        axis_tolerance: float = np.deg2rad(3.0),
        backend: str = "legacy6",
    ) -> TaskTransitionResult:
        """Track sampled task poses by warm-started IK continuation from one branch."""

        positions = np.asarray(target_positions, dtype=np.float64)
        axes = np.asarray(target_axes, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or axes.shape != positions.shape:
            raise ValueError("target positions and axes must have shape [T, 3]")
        if len(positions) < 2:
            raise ValueError("a task transition requires at least two target poses")
        q_values = [np.asarray(start, dtype=np.float64).copy()]
        if q_values[0].shape != (6,):
            raise ValueError("start configuration must have shape [6]")
        for position, axis in zip(positions[1:], axes[1:]):
            candidate = self.solve_ik(
                position,
                axis,
                q_values[-1],
                position_tolerance=min(position_tolerance, 1e-3),
                axis_tolerance=axis_tolerance,
                backend=backend,
            )
            if candidate is None:
                return TaskTransitionResult(
                    False, np.asarray(q_values), "ik_continuation", np.inf, np.inf
                )
            adjusted = _nearest_equivalent(
                candidate.q, q_values[-1], self.lower_limits, self.upper_limits
            )
            valid, adjusted, reason = self.check_transition(
                q_values[-1],
                adjusted,
                maximum_joint_step=maximum_joint_step,
                minimum_manipulability=minimum_manipulability,
                allow_equivalent_end=False,
            )
            if not valid:
                return TaskTransitionResult(
                    False, np.asarray(q_values), reason, np.inf, np.inf
                )
            q_values.append(adjusted)

        q_path = np.asarray(q_values)
        position_errors = np.empty(len(q_path), dtype=np.float64)
        axis_errors = np.empty(len(q_path), dtype=np.float64)
        normalized_axes = axes / np.maximum(
            np.linalg.norm(axes, axis=1, keepdims=True), 1e-12
        )
        for index, q in enumerate(q_path):
            achieved_position, achieved_axis = self.forward(q)
            position_errors[index] = np.linalg.norm(achieved_position - positions[index])
            axis_errors[index] = np.arccos(
                np.clip(np.dot(achieved_axis, normalized_axes[index]), -1.0, 1.0)
            )
        max_position_error = float(position_errors.max())
        max_axis_error = float(axis_errors.max())
        if max_position_error > position_tolerance:
            reason = "surface_tracking"
        elif max_axis_error > axis_tolerance:
            reason = "axis_tracking"
        else:
            reason = None
        return TaskTransitionResult(
            reason is None,
            q_path,
            reason,
            max_position_error,
            max_axis_error,
        )

    def continue_task_transition_to_configuration(
        self,
        start: np.ndarray,
        end: np.ndarray,
        target_positions: np.ndarray,
        target_axes: np.ndarray,
        *,
        maximum_joint_step: float = 0.8,
        minimum_manipulability: float = 1e-5,
        position_tolerance: float = 3e-3,
        axis_tolerance: float = np.deg2rad(3.0),
        final_tracking_samples: int = 3,
        backend: str = "legacy6",
    ) -> TaskTransitionResult:
        """Continue along a task edge and finish exactly at a target-layer candidate."""

        positions = np.asarray(target_positions, dtype=np.float64)
        axes = np.asarray(target_axes, dtype=np.float64)
        if len(positions) < 3 or final_tracking_samples < 2:
            raise ValueError("candidate-targeted continuation needs at least three task poses")
        prefix = self.continue_task_transition(
            start,
            positions[:-1],
            axes[:-1],
            maximum_joint_step=maximum_joint_step,
            minimum_manipulability=minimum_manipulability,
            position_tolerance=position_tolerance,
            axis_tolerance=axis_tolerance,
            backend=backend,
        )
        if not prefix.feasible:
            return prefix
        fractions = np.linspace(0.0, 1.0, final_tracking_samples)
        final_positions = (
            (1.0 - fractions[:, None]) * positions[-2]
            + fractions[:, None] * positions[-1]
        )
        final_axes = (
            (1.0 - fractions[:, None]) * axes[-2]
            + fractions[:, None] * axes[-1]
        )
        final_axes /= np.maximum(np.linalg.norm(final_axes, axis=1, keepdims=True), 1e-12)
        final = self.check_task_transition(
            prefix.q_path[-1],
            end,
            final_positions,
            final_axes,
            maximum_joint_step=maximum_joint_step,
            minimum_manipulability=minimum_manipulability,
            position_tolerance=position_tolerance,
            axis_tolerance=axis_tolerance,
            allow_equivalent_end=False,
        )
        q_path = np.concatenate((prefix.q_path, final.q_path[1:]), axis=0)
        return TaskTransitionResult(
            final.feasible,
            q_path,
            final.failure_reason,
            max(prefix.max_position_error, final.max_position_error),
            max(prefix.max_axis_error, final.max_axis_error),
        )

    def _candidate(self, q: np.ndarray, position_error: float, axis_error: float) -> IKCandidate:
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacobian_position, jacobian_rotation, self.site_id
        )
        jacobian = np.vstack((jacobian_position, jacobian_rotation))
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        manipulability = float(np.prod(singular_values))
        margin = np.minimum(q - self.lower_limits, self.upper_limits - q)
        normalized_margin = float(np.min(margin / (self.upper_limits - self.lower_limits)))
        return IKCandidate(
            q=q.copy(),
            position_error=position_error,
            axis_error=axis_error,
            manipulability=manipulability,
            joint_limit_margin=normalized_margin,
            collision_free=self.data.ncon == 0,
        )

    def _edge_invalid_reason(
        self, start: np.ndarray, end: np.ndarray, minimum_manipulability: float
    ) -> str | None:
        for fraction in np.linspace(0.0, 1.0, 7)[1:-1]:
            q = (1.0 - fraction) * start + fraction * end
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            if self.data.ncon > 0:
                return "collision"
            jacobian_position = np.zeros((3, self.model.nv))
            jacobian_rotation = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(
                self.model, self.data, jacobian_position, jacobian_rotation, self.site_id
            )
            if float(np.prod(np.linalg.svd(np.vstack((jacobian_position, jacobian_rotation)), compute_uv=False))) < minimum_manipulability:
                return "singularity"
        return None

    def _diagnose_pose_failure(
        self,
        position: np.ndarray,
        axis: np.ndarray,
        axis_tolerance: float,
        rng: np.random.Generator,
    ) -> str:
        # Wider axis tolerance separates positional reachability from orientation feasibility.
        position_seeds = [self.home]
        position_seeds.extend(
            rng.uniform(self.lower_limits, self.upper_limits) for _ in range(20)
        )
        position_solutions = [
            solution for seed in position_seeds
            if (solution := self.solve_position_ik(position, seed)) is not None
        ]
        if not position_solutions:
            unconstrained = self.solve_position_ik(
                position, self.home, constrain_limits=False
            )
            return "joint_limit" if unconstrained is not None else "position_unreachable"
        relaxed = self.enumerate_ik(
            position, axis, seed_configurations=position_solutions,
            random_restarts=8, rng=rng, axis_tolerance=axis_tolerance,
            minimum_manipulability=0.0, require_collision_free=False,
        )
        if not relaxed:
            return "orientation_tolerance"
        if all(not candidate.collision_free for candidate in relaxed):
            return "collision"
        if all(candidate.manipulability < 1e-5 for candidate in relaxed):
            return "singularity"
        return "orientation_tolerance"


def transform_surface_pose_path(
    points: np.ndarray,
    normals: np.ndarray,
    transform_base_from_surface: np.ndarray,
    *,
    axis_opposes_normal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform_base_from_surface, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape [4, 4]")
    positions = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    axes = np.asarray(normals) @ transform[:3, :3].T
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    if axis_opposes_normal:
        axes = -axes
    return positions, axes


def interpolate_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    projected_face_indices: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals * face_areas[:, None])
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= np.maximum(norms, 1e-12)
    triangle_normals = vertex_normals[faces[projected_face_indices]]
    interpolated = np.einsum("pi,pij->pj", barycentric, triangle_normals)
    interpolated /= np.maximum(np.linalg.norm(interpolated, axis=1, keepdims=True), 1e-12)
    return interpolated


def _configuration_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = (first - second + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(difference))


def sample_axis_cone(
    central_axis: np.ndarray,
    axis_tolerance: float,
    count: int,
) -> np.ndarray:
    """Deterministically sample directions inside an angular tolerance cone."""

    axis = np.asarray(central_axis, dtype=np.float64)
    if axis.shape != (3,) or float(np.linalg.norm(axis)) <= 1e-12:
        raise ValueError("central_axis must be a nonzero 3-vector")
    if axis_tolerance < 0.0 or count < 1:
        raise ValueError("axis tolerance and sample count are invalid")
    axis /= np.linalg.norm(axis)
    if count == 1 or axis_tolerance == 0.0:
        return axis[None, :]
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    tangent_u = np.cross(axis, reference)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(axis, tangent_u)
    directions = [axis]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(count - 1):
        radial_fraction = np.sqrt((index + 0.5) / (count - 1))
        angle = axis_tolerance * radial_fraction
        azimuth = golden_angle * index
        tangent = np.cos(azimuth) * tangent_u + np.sin(azimuth) * tangent_v
        directions.append(np.cos(angle) * axis + np.sin(angle) * tangent)
    return np.asarray(directions)


def _nearest_equivalent(
    q: np.ndarray,
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    result = q.copy()
    for joint in range(len(result)):
        options = result[joint] + 2.0 * np.pi * np.arange(-2, 3)
        valid = options[(options >= lower[joint]) & (options <= upper[joint])]
        if len(valid):
            result[joint] = valid[np.argmin(np.abs(valid - reference[joint]))]
    return result


def _farthest_configuration_subset(
    candidates: list[IKCandidate], count: int
) -> list[IKCandidate]:
    # Start from the strongest-conditioned solution, then cover distinct IK/roll families.
    return [candidates[index] for index in _farthest_configuration_indices(candidates, count)]


def _farthest_configuration_indices(
    candidates: list[IKCandidate], count: int
) -> list[int]:
    first = int(np.argmax([candidate.manipulability for candidate in candidates]))
    selected = [first]
    minimum_distances = np.asarray(
        [_configuration_distance(candidate.q, candidates[first].q) for candidate in candidates]
    )
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distances))
        selected.append(next_index)
        distances = np.asarray(
            [_configuration_distance(candidate.q, candidates[next_index].q) for candidate in candidates]
        )
        minimum_distances = np.minimum(minimum_distances, distances)
        minimum_distances[selected] = -np.inf
    return selected
