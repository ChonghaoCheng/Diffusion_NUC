from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from diffusion_coverage.planning.e12_programs import DecodedProgram, GeometryLibrary, GeometryProgram, decode_program
from diffusion_coverage.robot.synchronized_motion import SynchronizedMotionTrace
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


@dataclass(frozen=True)
class ProgramLiftResult:
    status: str
    trace: SynchronizedMotionTrace | None
    decoded: DecodedProgram | None
    ik_calls: int
    spacing_m: float
    halvings: int
    elapsed_s: float
    failure_index: int | None
    failure_reason: str | None
    corrections: tuple[dict[str, Any], ...]


def _targets(surface: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation, translation = transform[:3, :3], transform[:3, 3]
    positions = surface @ rotation.T + translation
    axes = -(surface / np.linalg.norm(surface, axis=1, keepdims=True)) @ rotation.T
    return positions, axes


def lift_program(
    program: GeometryProgram,
    library: GeometryLibrary,
    transform_base_from_surface: np.ndarray,
    q0: np.ndarray,
    robot: UR5eKinematics,
    config: dict[str, Any],
    *,
    start_surface_point: np.ndarray,
) -> ProgramLiftResult:
    """Continuously lift a geometry-only program from q0 without graph or future q access."""

    began = perf_counter()
    settings = config["lifter"]
    rsettings = config["robot"]
    deadline = began + float(settings["deadline_s"])
    maximum_calls = int(settings["max_ik_calls"])
    initial_spacing = float(settings["initial_spacing_m"])
    calls = 0
    last_reason = None
    last_index = None
    transform = np.asarray(transform_base_from_surface, dtype=np.float64)
    initial = np.asarray(q0, dtype=np.float64)
    if initial.shape != (6,) or not np.all(np.isfinite(initial)):
        raise ValueError("q0 must be a finite six-vector")
    q6 = float(initial[5])

    for halving in range(int(settings["maximum_halvings"]) + 1):
        spacing = initial_spacing / (2**halving)
        if spacing < float(settings["minimum_spacing_m"]) - 1e-15:
            break
        decoded = decode_program(program, library, start_surface_point, maximum_step=spacing, maximum_tokens=int(config["program"]["maximum_tokens"]))
        positions, axes = _targets(decoded.surface_points, transform)
        q_values = [initial.copy()]
        last_reason = None; last_index = None
        for index in range(1, len(positions)):
            if perf_counter() >= deadline:
                return ProgramLiftResult("lift_budget_limited", None, decoded, calls, spacing, halving, perf_counter()-began, index, "deadline", decoded.corrections)
            if calls >= maximum_calls:
                return ProgramLiftResult("lift_budget_limited", None, decoded, calls, spacing, halving, perf_counter()-began, index, "ik_call_limit", decoded.corrections)
            calls += 1
            solved = robot.solve_ik(
                positions[index], axes[index], q_values[-1],
                position_tolerance=float(rsettings["ik_position_tolerance_m"]),
                axis_tolerance=np.deg2rad(float(rsettings["ik_axis_tolerance_degrees"])),
                max_iterations=int(rsettings["ik_max_iterations"]), damping=float(rsettings["ik_damping"]),
                max_update=float(rsettings["ik_max_update_rad"]), backend="task5",
            )
            if solved is None:
                last_reason = "ik_not_found"; last_index = index; break
            q = np.asarray(solved.q, dtype=np.float64)
            if abs(float(q[5]) - q6) > 1e-10:
                last_reason = "constant_q6_violation"; last_index = index; break
            if np.max(np.abs(q - q_values[-1])) > float(settings["dense_joint_step_rad"]) + 1e-12:
                last_reason = "joint_step_violation"; last_index = index; break
            task = evaluate_task_kinematics_5d(robot, q, characteristic_length=float(rsettings["characteristic_length_m"]))
            checked = robot.evaluate_configuration(q)
            pe = float(np.linalg.norm(task.position - positions[index]))
            ae = float(np.arccos(np.clip(np.dot(task.tool_axis, axes[index]), -1.0, 1.0)))
            if pe > float(rsettings["position_tolerance_m"]) + 1e-12:
                last_reason = "position_contract"; last_index = index; break
            if ae > np.deg2rad(float(rsettings["axis_tolerance_degrees"])) + 1e-12:
                last_reason = "axis_contract"; last_index = index; break
            if task.sigma_min_5 < float(rsettings["sigma_safe"]) - 1e-12:
                last_reason = "sigma_contract"; last_index = index; break
            if checked.joint_limit_margin < -1e-12:
                last_reason = "joint_limit"; last_index = index; break
            if not checked.collision_free:
                last_reason = "modeled_collision"; last_index = index; break
            q_values.append(q.copy())
        if last_reason is None:
            q = np.asarray(q_values, dtype=np.float64)
            trace = SynchronizedMotionTrace(
                q=q, u=decoded.parameter.copy(), target_position=positions, target_axis=axes,
                activity=np.ones(len(q), dtype=bool), geometry_arc_id=-12,
                original_interval=(0.0, 1.0), start_node=-1, end_node=-1,
            )
            return ProgramLiftResult("lifted", trace, decoded, calls, spacing, halving, perf_counter()-began, None, None, decoded.corrections)
    return ProgramLiftResult("lift_failed", None, decoded if 'decoded' in locals() else None, calls, spacing if 'spacing' in locals() else initial_spacing, halving if 'halving' in locals() else 0, perf_counter()-began, last_index, last_reason, tuple() if 'decoded' not in locals() else decoded.corrections)


def densify_program_trace(trace: SynchronizedMotionTrace, surface_points: np.ndarray, transform: np.ndarray, radius: float, *, joint_step: float, surface_step: float) -> SynchronizedMotionTrace:
    """Densify the original q/declared-sphere intervals without changing the q curve."""
    trace.__post_init__()
    surface = np.asarray(surface_points, dtype=np.float64)
    if surface.shape != (len(trace.q), 3):
        raise ValueError("surface points and synchronized trace must have identical length")
    qout=[trace.q[0]]; sout=[surface[0]]; uout=[trace.u[0]]
    for qa,qb,sa,sb,ua,ub in zip(trace.q[:-1],trace.q[1:],surface[:-1],surface[1:],trace.u[:-1],trace.u[1:]):
        x=sa/radius; y=sb/radius
        angle=float(np.arctan2(np.linalg.norm(np.cross(x,y)),np.dot(x,y)))
        distance=radius*angle
        count=max(1,int(np.ceil(max(np.max(np.abs(qb-qa))/joint_step,distance/surface_step))))
        for j in range(1,count+1):
            f=j/count
            qout.append((1-f)*qa+f*qb);uout.append((1-f)*ua+f*ub)
            if angle<=1e-14:
                v=(1-f)*x+f*y;v/=np.linalg.norm(v)
            else:v=(np.sin((1-f)*angle)*x+np.sin(f*angle)*y)/np.sin(angle)
            sout.append(radius*v)
    surface_dense=np.asarray(sout);positions,axes=_targets(surface_dense,np.asarray(transform,dtype=np.float64))
    return SynchronizedMotionTrace(np.asarray(qout),np.asarray(uout),positions,axes,np.ones(len(qout),bool),trace.geometry_arc_id,trace.original_interval,-1,-1)
