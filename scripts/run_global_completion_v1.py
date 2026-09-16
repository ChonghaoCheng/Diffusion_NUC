#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import tracemalloc
from typing import Any
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np

from diffusion_coverage.coverage.episode_summary import summarize_ordered_membership
from diffusion_coverage.coverage.nuc_evaluator import _ordered_footprint_membership, evaluate_nuc_coverage
from diffusion_coverage.coverage.patterns import raster_pattern, spiral_pattern
from diffusion_coverage.diagnostics.symmetry_layout import analytical_points_and_normals
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics, transform_surface_pose_path
from diffusion_coverage.solvers.completion_bound import CompletionEdge, completion_repeat_lower_bound
from diffusion_coverage.solvers.history_search import SearchGraph, require_real_graph_readiness, search_history_graph
from diffusion_coverage.surface.primitives import make_hemisphere, make_saddle


STAGES = ("audit", "qualify", "validate-bound", "build-graph", "compare", "verify")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen E08 completion-bound prototype")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/global_completion_bound_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/global_completion_bound_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    functions = {
        "audit": stage_audit,
        "qualify": stage_qualify,
        "validate-bound": stage_validate_bound,
        "build-graph": stage_build_graph,
        "compare": stage_compare,
        "verify": stage_verify,
    }
    functions[args.stage](config, output)


def robot_from_config(config: dict[str, Any]) -> UR5eKinematics:
    return UR5eKinematics(
        config["inputs"]["robot_model"],
        site_name=config["robot"]["site_name"],
        tool_axis_index=int(config["robot"]["tool_axis_index"]),
        tool_axis_sign=float(config["robot"]["tool_axis_sign"]),
    )


def make_surface(surface_id: str, config: dict[str, Any], *, verification: bool = False):
    historical = json.loads((ROOT / config["inputs"]["historical_config"]).read_text())
    cfg = historical["surfaces"][surface_id]
    samples = config["coverage"]["verification_samples_per_face" if verification else "main_samples_per_face"]
    if surface_id == "saddle":
        return make_saddle(
            width=cfg["width"], height=cfg["height"], curvature=cfg["curvature"],
            nx=12, ny=12, samples_per_face=int(samples),
        )
    return make_hemisphere(
        radius=cfg["radius"], n_azimuth=24, n_polar=10, samples_per_face=int(samples)
    )


def scene_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    placements = json.loads((ROOT / config["inputs"]["placements"]).read_text())["selected"]
    records = []
    for surface_id in ("saddle", "hemisphere"):
        for level, value in placements[surface_id].items():
            records.append({"surface_id": surface_id, "level": level, **value})
    return sorted(records, key=lambda item: (item["surface_id"], item["candidate_id"]))


def stage_audit(config: dict[str, Any], output: Path) -> None:
    robot = robot_from_config(config)
    arrays = np.load(ROOT / config["inputs"]["symmetry_arrays"])
    dependencies = model_dependencies(Path(config["inputs"]["robot_model"]))
    preflight = {
        "timestamp": config["registered_at"],
        "code": git_snapshot(ROOT),
        "ara": git_snapshot(Path("/data/chocheng/worktrees/global-completion-bound-v1-ara")),
        "expected_code_baseline": config["code_baseline"],
        "expected_ara_baseline": config["ara_baseline"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "model_dependencies": dependencies,
        "model_scope": inspect_model(robot, config),
        "task_freedom_audit": audit_task_freedom(robot, arrays, config),
        "reverse_axis_cross_residual": audit_reverse_axis(robot, config),
        "e01_descriptive_pool": "missing_not_rerun",
    }
    write_json(output / "preflight.json", preflight)
    rows = compare_ik_backends(robot, arrays, config)
    write_csv(output / "ik_comparison.csv", rows)
    checkpoint(output, "audit", {"complete": True, "rows": len(rows)})
    print(json.dumps({"stage": "audit", "ik_rows": len(rows), "model_files": len(dependencies)}, indent=2))


def git_snapshot(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=path, text=True, capture_output=True, check=True).stdout.strip()
    return {
        "path": str(path), "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"), "status": run("status", "--short", "--branch"),
        "remotes": run("remote", "-v"),
    }


def model_dependencies(model_path: Path) -> list[dict[str, Any]]:
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited or not path.exists():
            return
        visited.add(path)
        if path.suffix.lower() != ".xml":
            return
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        meshdir = "" if compiler is None else compiler.attrib.get("meshdir", "")
        texturedir = "" if compiler is None else compiler.attrib.get("texturedir", "")
        for element in root.iter():
            for attribute in ("file",):
                value = element.attrib.get(attribute)
                if not value:
                    continue
                candidates = [(path.parent / value).resolve()]
                if element.tag == "mesh" and meshdir:
                    candidates.insert(0, (path.parent / meshdir / value).resolve())
                if element.tag == "texture" and texturedir:
                    candidates.insert(0, (path.parent / texturedir / value).resolve())
                for candidate in candidates:
                    if candidate.exists():
                        visit(candidate); break

    visit(model_path)
    return [{"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size} for path in sorted(visited)]


def inspect_model(robot: UR5eKinematics, config: dict[str, Any]) -> dict[str, Any]:
    model = robot.model
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    site_body = int(model.site_bodyid[robot.site_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, site_body)
    excluded = []
    for index in range(model.nexclude):
        signature = int(model.exclude_signature[index])
        body1, body2 = divmod(signature, model.nbody)
        excluded.append([
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
        ])
    return {
        "nq": model.nq, "nv": model.nv, "joint_names": joint_names,
        "joint_ranges_rad": model.jnt_range.tolist(),
        "tcp_site": config["robot"]["site_name"], "tcp_body": body_name,
        "tcp_local_position_m": model.site_pos[robot.site_id].tolist(),
        "tool_axis_index": robot.tool_axis_index, "tool_axis_sign": robot.tool_axis_sign,
        "q6_axis_in_parent": model.jnt_axis[-1].tolist(),
        "collision_geometries": int(model.ngeom), "explicit_excluded_body_pairs": excluded,
        "claim_scope": config["process_contract"]["collision_claim"],
        "h_c_definition": "minimum singular value of [Jp/Lc; U(a)^T Jw] minus sigma_safe",
    }


def audit_task_freedom(robot: UR5eKinematics, arrays: Any, config: dict[str, Any]) -> dict[str, Any]:
    q_values: list[np.ndarray] = []
    witness_dir = ROOT / "results/symmetry_preserving_global_layout_v1/witnesses"
    for path in sorted(witness_dir.glob("*.npz")):
        q = np.asarray(np.load(path)["q"], dtype=np.float64)
        indices = np.linspace(0, len(q) - 1, min(3, len(q)), dtype=int)
        q_values.extend(q[indices])
        if len(q_values) >= int(config["ik_comparison"]["witness_samples"]):
            break
    q_values = q_values[: int(config["ik_comparison"]["witness_samples"])]
    rng = np.random.default_rng(int(config["seed"]))
    while len(q_values) < int(config["ik_comparison"]["witness_samples"]) + int(config["ik_comparison"]["additional_legal_samples"]):
        q = rng.uniform(robot.lower_limits, robot.upper_limits)
        if robot.evaluate_configuration(q).collision_free:
            q_values.append(q)
    records = []
    for q in q_values:
        task = evaluate_task_kinematics_5d(robot, q, characteristic_length=config["robot"]["characteristic_length_m"])
        robot.data.qpos[:] = q; mujoco.mj_forward(robot.model, robot.data)
        jp = np.zeros((3, 6)); jw = np.zeros((3, 6))
        mujoco.mj_jacSite(robot.model, robot.data, jp, jw, robot.site_id)
        _, _, vh = np.linalg.svd(task.normalized_jacobian_5, full_matrices=True)
        kernel = vh[-1]
        if kernel[-1] < 0: kernel = -kernel
        eps = 1e-6
        qp = q.copy(); qm = q.copy(); qp[-1] = min(robot.upper_limits[-1], qp[-1] + eps); qm[-1] = max(robot.lower_limits[-1], qm[-1] - eps)
        pp, ap = robot.forward(qp); pm, am = robot.forward(qm)
        delta = min(0.1, robot.upper_limits[-1] - q[-1], q[-1] - robot.lower_limits[-1])
        qr = q.copy(); qr[-1] += max(delta, 0.0)
        pr, ar = robot.forward(qr)
        tr = evaluate_task_kinematics_5d(robot, qr, characteristic_length=config["robot"]["characteristic_length_m"])
        records.append({
            "Jp_q6_norm": float(np.linalg.norm(jp[:, -1])),
            "projected_Jw_q6_norm": float(np.linalg.norm(task.axis_basis.T @ jw[:, -1])),
            "kernel_abs_dot_e6": float(abs(kernel[-1]) / np.linalg.norm(kernel)),
            "kernel_angle_to_e6_deg": float(np.rad2deg(np.arccos(np.clip(abs(kernel[-1]) / np.linalg.norm(kernel), -1, 1)))),
            "fd_position_derivative_error": float(np.linalg.norm((pp - pm) / (qp[-1] - qm[-1]) - jp[:, -1])),
            "fd_axis_derivative_norm": float(np.linalg.norm((ap - am) / (qp[-1] - qm[-1]))),
            "q6_perturb_position_change_m": float(np.linalg.norm(pr - task.position)),
            "q6_perturb_axis_change_rad": float(np.arccos(np.clip(np.dot(ar, task.tool_axis), -1, 1))),
            "q6_perturb_sigma_change": float(abs(tr.sigma_min_5 - task.sigma_min_5)),
            "q6_perturb_collision_change": bool(robot.evaluate_configuration(qr).collision_free != robot.evaluate_configuration(q).collision_free),
        })
    keys = records[0]
    return {
        "samples": len(records),
        "maxima": {key: max(float(record[key]) for record in records) for key in keys if key != "q6_perturb_collision_change"},
        "collision_changes": sum(record["q6_perturb_collision_change"] for record in records),
        "pure_q6_elimination_allowed": bool(
            max(record["q6_perturb_position_change_m"] for record in records) < 1e-9
            and max(record["q6_perturb_axis_change_rad"] for record in records) < 1e-6
            and not any(record["q6_perturb_collision_change"] for record in records)
        ),
        "records": records,
    }


def audit_reverse_axis(robot: UR5eKinematics, config: dict[str, Any]) -> dict[str, Any]:
    q = robot.home.copy()
    position, axis = robot.forward(q)
    cross_norm = float(np.linalg.norm(np.cross(axis, -axis)))
    outcomes = {}
    for backend in config["ik_comparison"]["backends"]:
        result = robot.solve_ik(position, -axis, q, backend=backend, **ik_kwargs(config))
        outcomes[backend] = result is not None
    return {"cross_residual_norm": cross_norm, "true_axis_error_degrees": 180.0, "converged": outcomes}


def compare_ik_backends(robot: UR5eKinematics, arrays: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    placements = json.loads((ROOT / config["inputs"]["placements"]).read_text())["selected"]
    for surface_id, level in (("saddle", "P_low"), ("saddle", "P_mid"), ("saddle", "P_high"),
                              ("hemisphere", "P_low"), ("hemisphere", "P_mid"), ("hemisphere", "P_high")):
        scene = placements[surface_id][level]
        transform = np.asarray(scene["transform_base_from_surface"], dtype=np.float64)
        points = np.asarray(arrays[f"{surface_id}_canonical_points"], dtype=np.float64)
        normals = np.asarray(arrays[f"{surface_id}_canonical_normals"], dtype=np.float64)
        positions, axes = transform_surface_pose_path(points, normals, transform)
        source = np.asarray(arrays[f"{surface_id}_source_q_start"], dtype=np.float64)
        common = robot.solve_ik(positions[0], axes[0], source, backend="task5", **ik_kwargs(config))
        starts = [] if common is None else [common.q]
        for backend in config["ik_comparison"]["backends"]:
            start = perf_counter()
            result = lift_targets(robot, positions, axes, starts, backend, config)
            q = result.get("q")
            rows.append({
                "surface_id": surface_id, "scene_id": scene["candidate_id"], "placement_level": level,
                "backend": backend, "common_start_count": len(starts), "complete_lift": q is not None,
                "longest_valid_prefix": result["prefix"], "first_failure_index": result["failure_index"],
                "failure_label": result["failure"], "ik_calls": result["calls"],
                "runtime_s": perf_counter() - start,
                "J_q": None if q is None else compute_joint_execution_cost((q,)).weighted_joint_length,
                "q6_cumulative_motion": None if q is None else float(np.abs(np.diff(q[:, -1])).sum()),
                "min_joint_margin": result.get("min_joint_margin"), "min_sigma_min_5": result.get("min_sigma"),
                "max_position_error_m": result.get("max_position_error"),
                "max_axis_error_degrees": None if result.get("max_axis_error") is None else np.rad2deg(result["max_axis_error"]),
            })
    return rows


def ik_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "position_tolerance": float(config["robot"]["ik_position_tolerance_m"]),
        "axis_tolerance": np.deg2rad(config["robot"]["ik_axis_tolerance_degrees"]),
        "max_iterations": int(config["robot"]["ik_max_iterations"]),
        "damping": float(config["robot"]["ik_damping"]),
        "max_update": float(config["robot"]["ik_max_update_rad"]),
    }


def lift_targets(robot: UR5eKinematics, positions: np.ndarray, axes: np.ndarray, starts: list[np.ndarray], backend: str, config: dict[str, Any]) -> dict[str, Any]:
    if not starts:
        return {"q": None, "prefix": 0, "failure_index": 0, "failure": "no_common_start", "calls": 0}
    calls = 0; best_prefix = 0; best_failure = "ik_nonconvergence"; best_index = 0
    for initial in starts:
        q_values = [np.asarray(initial, dtype=np.float64)]
        for index in range(1, len(positions)):
            calls += 1
            candidate = robot.solve_ik(positions[index], axes[index], q_values[-1], backend=backend, **ik_kwargs(config))
            if candidate is None:
                if index > best_prefix: best_prefix, best_index = index, index
                break
            if not candidate.collision_free:
                best_failure = "collision"; best_prefix, best_index = max(best_prefix, index), index; break
            q_values.append(candidate.q)
        if len(q_values) == len(positions):
            q = np.asarray(q_values)
            stats = path_task_stats(robot, q, positions, axes, config)
            return {"q": q, "prefix": len(q), "failure_index": None, "failure": None, "calls": calls, **stats}
    return {"q": None, "prefix": best_prefix, "failure_index": best_index, "failure": best_failure, "calls": calls}


def path_task_stats(robot: UR5eKinematics, q: np.ndarray, positions: np.ndarray, axes: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    position_errors=[]; axis_errors=[]; sigma=[]; margins=[]
    for value, position, axis in zip(q, positions, axes):
        task = evaluate_task_kinematics_5d(robot, value, characteristic_length=config["robot"]["characteristic_length_m"])
        position_errors.append(np.linalg.norm(task.position-position)); axis_errors.append(np.arccos(np.clip(np.dot(task.tool_axis,axis),-1,1)))
        sigma.append(task.sigma_min_5); margins.append(robot.evaluate_configuration(value).joint_limit_margin)
    return {"max_position_error": float(max(position_errors)), "max_axis_error": float(max(axis_errors)),
            "min_sigma": float(min(sigma)), "min_joint_margin": float(min(margins))}


def stage_qualify(config: dict[str, Any], output: Path) -> None:
    require_checkpoint(output, "audit")
    robot = robot_from_config(config)
    arrays = np.load(ROOT / config["inputs"]["symmetry_arrays"])
    rows: list[dict[str, Any]] = []
    witness_dir = output / "qualification_witnesses"
    witness_dir.mkdir(exist_ok=True)
    for scene in scene_records(config):
        surface_id = scene["surface_id"]
        surface = make_surface(surface_id, config)
        transform = np.asarray(scene["transform_base_from_surface"], dtype=np.float64)
        source_q = np.asarray(arrays[f"{surface_id}_source_q_start"], dtype=np.float64)
        for family_index, family in enumerate(config["candidate_families"][surface_id]):
            start = perf_counter()
            proposal = make_registered_proposal(surface, family, config)
            intended = evaluate_nuc_coverage(
                surface, proposal.plan, footprint_radius=config["coverage"]["footprint_radius_m"],
                path_sample_spacing=config["coverage"]["main_path_sample_spacing_m"],
            )
            paths = proposal.plan.active_paths()
            target_segments = [
                analytical_points_and_normals(
                    surface_id,
                    densify_surface_path(path, surface_id, surface.metadata, config["coverage"]["main_path_sample_spacing_m"]),
                    surface.metadata,
                )
                for path in paths
            ]
            position_segments=[]; axis_segments=[]
            for points, normals in target_segments:
                positions, axes = transform_surface_pose_path(points, normals, transform)
                position_segments.append(positions); axis_segments.append(axes)
            geometric_pass = (
                intended.missed_error <= config["coverage"]["missed_tolerance"]
                and intended.repeat_error <= config["coverage"]["repeat_tolerance"]
            )
            lifted = (
                lift_plan_segments(robot, position_segments, axis_segments, source_q, config)
                if geometric_pass else
                {"q_segments": None, "off_q": np.empty((0, 6)), "off_connection_pass": False,
                 "failure": "intended_coverage_failure", "ik_calls": 0}
            )
            q_segments = lifted.get("q_segments")
            strict = None
            if q_segments is not None:
                strict = check_strict_coverage_execution(
                    robot, tuple(q_segments), tuple(position_segments), tuple(axis_segments), surface, transform,
                    footprint_radius=config["coverage"]["footprint_radius_m"],
                    position_tolerance=config["robot"]["position_tolerance_m"],
                    axis_tolerance=np.deg2rad(config["robot"]["axis_tolerance_degrees"]),
                    characteristic_length=config["robot"]["characteristic_length_m"],
                    sigma_safe=config["robot"]["sigma_safe"],
                    missed_tolerance=config["coverage"]["missed_tolerance"],
                    repeat_tolerance=config["coverage"]["repeat_tolerance"],
                    interpolation_joint_step=config["robot"]["dense_q_step_rad"],
                    coverage_path_sample_spacing=config["coverage"]["main_path_sample_spacing_m"],
                )
            passed = bool(strict is not None and strict.overall_pass and lifted["off_connection_pass"])
            witness = None
            if passed:
                witness = witness_dir / f"{surface_id}_{scene['candidate_id']}_{family_index}.npz"
                payload: dict[str, Any] = {
                    "transform_base_from_surface": transform,
                    "family": np.asarray(family["name"]),
                    "surface_id": np.asarray(surface_id),
                    "off_q": lifted["off_q"],
                }
                for index, (q, positions, axes, target) in enumerate(zip(q_segments, position_segments, axis_segments, target_segments)):
                    payload[f"q_{index}"] = q; payload[f"positions_{index}"] = positions; payload[f"axes_{index}"] = axes
                    payload[f"surface_points_{index}"] = target[0]; payload[f"surface_normals_{index}"] = target[1]
                np.savez_compressed(witness, **payload)
            row = {
                "surface_id": surface_id, "scene_id": scene["candidate_id"], "placement_level": scene["level"],
                "family_index": family_index, "family": family["name"], "on_segments": len(paths),
                "intended_E_miss": intended.missed_error, "intended_E_rep": intended.repeat_error,
                "intended_E_NUC": intended.nuc_error, "geometry_projection_max_m": max_projection_error(surface_id, paths, surface.metadata),
                "reference_found": passed, "failure_reason": qualification_failure(lifted, strict),
                "missing_legal_reconfiguration": len(paths) > 1 and not lifted["off_connection_pass"],
                "actual_E_miss": None if strict is None or strict.coverage_metrics is None else strict.coverage_metrics.missed_error,
                "actual_E_rep": None if strict is None or strict.coverage_metrics is None else strict.coverage_metrics.repeat_error,
                "actual_E_NUC": None if strict is None or strict.coverage_metrics is None else strict.coverage_metrics.nuc_error,
                "J_q": lifted.get("J_q"), "max_position_error_m": None if strict is None else strict.max_position_error,
                "max_axis_error_degrees": None if strict is None else np.rad2deg(strict.max_axis_error),
                "min_sigma_min_5": None if strict is None else strict.min_sigma_min_5,
                "min_joint_margin": None if strict is None else strict.min_joint_limit_margin,
                "collision_scope": config["process_contract"]["collision_claim"],
                "witness_file": None if witness is None else str(witness.relative_to(ROOT)),
                "ik_calls": lifted["ik_calls"], "runtime_s": perf_counter() - start,
            }
            rows.append(row)
            print(surface_id, scene["candidate_id"], family["name"], "pass", passed, row["failure_reason"], flush=True)
    write_csv(output / "canonical_qualification.csv", rows)
    scene_pass = {(row["surface_id"], row["scene_id"]) for row in rows if row["reference_found"]}
    checkpoint(output, "qualify", {
        "complete": True, "candidate_rows": len(rows), "qualified_scenes": sorted([list(value) for value in scene_pass]),
        "saddle_qualified": any(surface == "saddle" for surface, _ in scene_pass),
        "hemisphere_qualified": any(surface == "hemisphere" for surface, _ in scene_pass),
    })
    print(json.dumps(json.loads((output / "qualify.checkpoint.json").read_text()), indent=2))


def make_registered_proposal(surface, family: dict[str, Any], config: dict[str, Any]):
    name = family["name"].removesuffix("_k2")
    common = {
        "footprint_radius": config["coverage"]["footprint_radius_m"],
        "max_segments": int(family["max_segments"]), "overlap": float(family["overlap"]),
        "waypoint_spacing": 0.75 * config["coverage"]["footprint_radius_m"],
    }
    if name.startswith("raster_"):
        mode, phase = name.split("_phase_")
        return raster_pattern(surface, sweep_axis=mode[-1], phase=float(phase), **common)
    return spiral_pattern(surface, phase=float(name.split("_phase_")[1]), **common)


def lift_plan_segments(robot: UR5eKinematics, position_segments: list[np.ndarray], axis_segments: list[np.ndarray], source_q: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    q_segments=[]; off_paths=[]; current=np.asarray(source_q, dtype=np.float64); calls=0
    for segment_index, (positions, axes) in enumerate(zip(position_segments, axis_segments)):
        calls += 1
        initial = robot.solve_ik(positions[0], axes[0], current, backend="task5", **ik_kwargs(config))
        if initial is None:
            return {"q_segments": None, "off_q": np.empty((0,6)), "off_connection_pass": False,
                    "failure": f"segment_{segment_index}_start_ik", "ik_calls": calls}
        if segment_index:
            off = interpolate_q(current, initial.q, config["robot"]["dense_q_step_rad"])
            if not all(robot.evaluate_configuration(q).collision_free for q in off):
                return {"q_segments": None, "off_q": np.asarray(off_paths, dtype=object), "off_connection_pass": False,
                        "failure": f"segment_{segment_index}_off_collision", "ik_calls": calls}
            off_paths.append(off)
        result = lift_targets(robot, positions, axes, [initial.q], "task5", config)
        calls += result["calls"]
        if result["q"] is None:
            return {"q_segments": None, "off_q": np.asarray(off_paths, dtype=object), "off_connection_pass": False,
                    "failure": f"segment_{segment_index}_{result['failure']}_at_{result['failure_index']}", "ik_calls": calls}
        q_segments.append(result["q"]); current=result["q"][-1]
    on_cost = compute_joint_execution_cost(tuple(q_segments)).weighted_joint_length
    off_cost = sum(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) for path in off_paths)
    return {"q_segments": q_segments, "off_q": np.asarray(off_paths, dtype=object), "off_connection_pass": True,
            "failure": None, "ik_calls": calls, "J_q": float(on_cost + off_cost)}


def interpolate_q(start: np.ndarray, end: np.ndarray, maximum_step: float) -> np.ndarray:
    intervals=max(1, int(np.ceil(np.max(np.abs(np.asarray(end)-np.asarray(start))) / maximum_step)))
    return np.linspace(start, end, intervals+1)


def max_projection_error(surface_id: str, paths: list[np.ndarray], metadata: dict[str, Any]) -> float:
    errors=[]
    for path in paths:
        projected, _ = analytical_points_and_normals(surface_id, path, metadata)
        errors.append(np.max(np.linalg.norm(projected-path, axis=1)))
    return float(max(errors, default=0.0))


def densify_surface_path(path:np.ndarray,surface_id:str,metadata:dict[str,Any],spacing:float)->np.ndarray:
    values=[np.asarray(path[0],dtype=float)]
    for start,end in zip(path[:-1],path[1:]):
        intervals=max(1,int(np.ceil(np.linalg.norm(end-start)/spacing)))
        for fraction in np.linspace(0,1,intervals+1)[1:]:
            provisional=((1-fraction)*start+fraction*end)[None,:]
            projected,_=analytical_points_and_normals(surface_id,provisional,metadata)
            values.append(projected[0])
    return np.asarray(values)


def qualification_failure(lifted: dict[str, Any], strict: Any) -> str | None:
    if lifted.get("q_segments") is None:
        return lifted.get("failure")
    if not lifted.get("off_connection_pass", False):
        return lifted.get("failure", "off_connection_failure")
    if strict is None:
        return "strict_checker_not_run"
    return strict.failure_reason


def stage_validate_bound(config: dict[str, Any], output: Path) -> None:
    require_checkpoint(output, "audit")
    validation = validate_random_graphs(config)
    validation["named_tests"] = {
        "random_segmentation_summary": "pytest:test_random_segmentation_matches_full_membership",
        "episode_corner_cases": "pytest:test_episode_corner_cases_and_initial_footprint",
        "history_dependence": "pytest:test_same_node_different_history_and_roll_does_not_revisit",
        "shared_bottleneck_not_summed": "pytest:test_shared_bottleneck_uses_max_not_sum",
        "area_weighted_quantile": "pytest:test_area_weighted_quantile_can_ignore_small_unreachable_unit",
        "off_reconfiguration": "pytest:test_off_reconfiguration_and_cheaper_added_edge_lower_bound",
        "graph_monotonicity": "pytest:test_off_reconfiguration_and_cheaper_added_edge_lower_bound",
        "endpoint_activity_and_winding": "pytest:test_joint_winding_and_activity_are_not_implicit_equivalence_keys",
    }
    write_json(output / "bound_validation.json", validation)
    if validation["oracle_state_mismatches"] or validation["inadmissible_prefixes"] or validation["false_prunes"] or validation["search_mismatches"]:
        checkpoint(output, "validate-bound", {"complete": False, "reason": "correctness_failure", **validation})
        raise RuntimeError("completion-bound validation failed; main experiment is forbidden")
    manifest = freeze_manifest(config, output)
    write_exclusive_json(output / "manifest.json", manifest)
    checkpoint(output, "validate-bound", {"complete": True, **validation, "manifest_sha256": file_hash(output / "manifest.json")})
    print(json.dumps(validation, indent=2))


def validate_random_graphs(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["seed"]) + 17)
    graph_count = int(config["bound_validation"]["random_graphs"])
    prefix_count=0; completion_prefixes=0; state_mismatches=0; inadmissible=0; false_prunes=0; search_mismatches=0
    graph_records=[]
    for graph_index in range(graph_count):
        units=int(rng.integers(3, int(config["bound_validation"]["max_units"])+1))
        nodes=int(rng.integers(3, int(config["bound_validation"]["max_nodes"])+1))
        weights=rng.integers(1, int(config["bound_validation"]["integer_area_weight_max"])+1, size=units).astype(float)
        memberships=[]
        for node in range(nodes):
            member=np.zeros(units,dtype=bool); member[node % units]=True
            if rng.random()<0.3: member[int(rng.integers(units))]=True
            memberships.append(member)
        raw=[]; edge_id=0
        pairs=[(i,i+1) for i in range(nodes-1)]
        pairs += [(i,j) for i in range(nodes-2) for j in range(i+2,nodes) if rng.random()<0.22]
        for start,end in pairs:
            mid_count=int(rng.integers(0,3)); columns=[memberships[start]]
            columns += [rng.random(units)<0.3 for _ in range(mid_count)]; columns += [memberships[end]]
            active=np.ones(len(columns),dtype=bool)
            if len(columns)>=3 and rng.random()<0.2: active[1:-1]=False
            summary=summarize_ordered_membership(np.asarray(columns,bool).T,weights,active=active)
            raw.append(CompletionEdge(edge_id,start,end,summary,float(rng.integers(1,8)))); edge_id+=1
        graph_hash=hash_graph_arrays(tuple(memberships),tuple(raw),weights)
        graph=SearchGraph(tuple(memberships),tuple(raw),weights,graph_hash)
        paths=enumerate_dag_paths(raw,0,nodes)
        oracle_goals=[]
        for path in paths:
            counts=memberships[0].astype(np.int64); state_covered=memberships[0].copy(); state_membership=memberships[0].copy(); repeat=0.0; used=1; cost=0.0
            prefixes=[(0,counts.copy(),state_covered.copy(),state_membership.copy(),repeat,used,cost,())]
            for edge in path:
                counts += edge.summary.episode_counts - edge.summary.start_membership.astype(np.int64)
                from diffusion_coverage.coverage.episode_summary import EpisodeState, apply_edge_summary
                state=apply_edge_summary(EpisodeState(state_covered,state_membership,repeat),edge.summary,weights)
                state_covered,state_membership,repeat=state.covered,state.membership,state.repeat_error
                used += edge.summary.off_to_on_count; cost += edge.joint_cost
                prefixes.append((edge.end,counts.copy(),state_covered.copy(),state_membership.copy(),repeat,used,cost,tuple(e.edge_id for e in path[:len(prefixes)])))
            for node,counts,covered,membership,repeat,used,cost,prefix_path in prefixes:
                prefix_count+=1
                exact_repeat=float(np.dot(weights,np.maximum(counts-1,0))/weights.sum())
                if not np.array_equal(counts>0,covered) or abs(exact_repeat-repeat)>1e-12: state_mismatches+=1
                completions=oracle_completions(node,counts,used,cost,raw,weights,maximum_on_segments=2,missed_tolerance=0.25)
                if completions:
                    completion_prefixes+=1; optimal_repeat=min(item[0] for item in completions)
                    bound=completion_repeat_lower_bound(node=node,covered=covered,repeat_error=repeat,used_on_segments=used,
                        maximum_on_segments=2,weights=weights,edges=raw,missed_tolerance=0.25)
                    if bound.total_lower_bound>optimal_repeat+1e-12: inadmissible+=1
                    if bound.total_lower_bound>0.4+1e-12 and optimal_repeat<=0.4+1e-12: false_prunes+=1
                miss=float(weights[counts==0].sum()/weights.sum())
                if miss<=0.25 and used<=2: oracle_goals.append((used-1,cost))
        oracle=min(oracle_goals) if oracle_goals else None
        outcomes=[]
        for enabled in (False,True):
            result=search_history_graph(graph,start_node=0,maximum_on_segments=2,missed_tolerance=0.25,repeat_tolerance=10.0,
                use_completion_bound=enabled,wall_time_s=30.0,expanded_limit=200000,checkpoint_times=())
            objective=None if result.incumbent is None else (result.incumbent.used_on_segments-1,result.incumbent.joint_cost)
            outcomes.append(objective)
            if objective!=oracle: search_mismatches+=1
        graph_records.append({"graph":graph_index,"units":units,"nodes":nodes,"edges":len(raw),"paths":len(paths),"oracle":oracle,"search":outcomes})
    return {"random_graphs":graph_count,"prefixes":prefix_count,"completable_prefixes":completion_prefixes,
            "oracle_state_mismatches":state_mismatches,"inadmissible_prefixes":inadmissible,"false_prunes":false_prunes,
            "search_mismatches":search_mismatches,"graphs":graph_records}


def enumerate_dag_paths(edges: list[CompletionEdge], start: int, nodes: int) -> list[tuple[CompletionEdge,...]]:
    outgoing={i:[] for i in range(nodes)}
    for edge in edges: outgoing[edge.start].append(edge)
    paths=[]
    def walk(node: int, prefix: tuple[CompletionEdge,...]):
        paths.append(prefix)
        for edge in outgoing[node]: walk(edge.end,prefix+(edge,))
    walk(start,())
    return paths


def oracle_completions(node: int, counts: np.ndarray, used: int, cost: float, edges: list[CompletionEdge], weights: np.ndarray,
                       *, maximum_on_segments: int, missed_tolerance: float) -> list[tuple[float,float,int]]:
    outgoing={}
    for edge in edges: outgoing.setdefault(edge.start,[]).append(edge)
    results=[]
    def walk(current: int, visits: np.ndarray, segments: int, joint_cost: float):
        miss=float(weights[visits==0].sum()/weights.sum())
        if miss<=missed_tolerance:
            repeated=float(np.dot(weights,np.maximum(visits-1,0))/weights.sum())
            results.append((repeated,joint_cost,segments)); return
        for edge in outgoing.get(current,[]):
            new_segments=segments+edge.summary.off_to_on_count
            if new_segments<=maximum_on_segments:
                walk(edge.end,visits+edge.summary.episode_counts-edge.summary.start_membership.astype(np.int64),new_segments,joint_cost+edge.joint_cost)
    walk(node,counts.copy(),used,cost)
    return results


def freeze_manifest(config: dict[str, Any], output: Path) -> dict[str, Any]:
    input_paths=[]
    for value in config["inputs"].values():
        path=Path(value); path=path if path.is_absolute() else ROOT/path
        if path.exists() and path.is_file(): input_paths.append(path)
    for item in model_dependencies(Path(config["inputs"]["robot_model"])):
        input_paths.append(Path(item["path"]))
    diff=subprocess.run(["git","diff","--binary","--no-ext-diff"],cwd=ROOT,capture_output=True,check=True).stdout
    code_state_sha256,code_files=code_state_snapshot()
    return {
        "frozen_before_main_results": True, "registered_at": config["registered_at"],
        "code_head": git_snapshot(ROOT)["head"], "ara_head": git_snapshot(Path("/data/chocheng/worktrees/global-completion-bound-v1-ara"))["head"],
        "config_sha256": file_hash(ROOT/"configs/global_completion_bound_v1.json"),
        "plan_sha256": file_hash(Path("/data/chocheng/worktrees/global-completion-bound-v1-ara/staging/global_completion_bound_v1_plan.md")),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(), "code_diff_bytes": len(diff),
        "code_state_sha256":code_state_sha256,"changed_code_files":code_files,
        "inputs": [{"path":str(path),"sha256":file_hash(path),"bytes":path.stat().st_size} for path in sorted(set(input_paths))],
        "thresholds": {"robot":config["robot"],"coverage":config["coverage"],"on_segment_budgets":config["on_segment_budgets"],
                       "graph":config["graph"],"search":config["search"]},
    }


def stage_build_graph(config: dict[str, Any], output: Path) -> None:
    require_real_graph_readiness(config.get("graph_capabilities", {}))
    require_checkpoint(output, "qualify")
    require_checkpoint(output, "validate-bound")
    verify_frozen_diff(config, output)
    rows=read_csv(output/"canonical_qualification.csv")
    passed=[row for row in rows if as_bool(row["reference_found"])]
    grouped={}
    for row in passed: grouped.setdefault((row["surface_id"],row["scene_id"]),[]).append(row)
    graph_dir=output/"graphs"; graph_dir.mkdir(exist_ok=True)
    manifest_rows=[]
    for scene in scene_records(config):
        key=(scene["surface_id"],scene["candidate_id"]); candidates=sorted(grouped.get(key,[]),key=lambda row:int(row["family_index"]))
        if not candidates:
            manifest_rows.append({"surface_id":key[0],"scene_id":key[1],"status":"not_qualified","graph_hash":None,
                                  "nodes":0,"edges":0,"connection_attempts":0,"connection_edges":0,"build_s":0.0})
            continue
        start=perf_counter()
        graph_file=graph_dir/f"{key[0]}_{key[1]}.npz"
        built=build_scene_graph(candidates[:4],scene,config,graph_file)
        manifest_rows.append({"surface_id":key[0],"scene_id":key[1],"status":built["status"],"graph_file":str(graph_file.relative_to(ROOT)),
                              "graph_hash":built["graph_hash"],"nodes":built["nodes"],"edges":built["edges"],
                              "connection_attempts":built["connection_attempts"],"connection_edges":built["connection_edges"],
                              "ik_calls":0,"stop_reason":built["stop_reason"],"reference_edge_ids":built["reference_edge_ids"],
                              "build_s":perf_counter()-start})
        print(key,built["nodes"],built["edges"],built["connection_edges"],flush=True)
    document={"frozen_before_search":True,"graphs":manifest_rows,"common_manifest_sha256":file_hash(output/"manifest.json")}
    write_json(output/"graph_manifest.json",document)
    checkpoint(output,"build-graph",{"complete":True,"graphs":sum(row["status"]=="ready" for row in manifest_rows)})


def build_scene_graph(candidates: list[dict[str,str]], scene: dict[str,Any], config: dict[str,Any], graph_file: Path) -> dict[str,Any]:
    robot=robot_from_config(config); surface=make_surface(scene["surface_id"],config)
    transform=np.asarray(scene["transform_base_from_surface"],dtype=np.float64); inverse=np.linalg.inv(transform)
    weights=surface.area_weights.copy(); nodes=[]; node_memberships=[]; node_points=[]; edges=[]; q_witness=[]; active_witness=[]; target_witness=[]; kinds=[]
    reference_edge_ids=[]; family_chains=[]
    def add_node(q: np.ndarray, membership: np.ndarray, point: np.ndarray) -> int:
        if len(nodes)>=int(config["graph"]["max_configuration_nodes"]): raise RuntimeError("node_limit")
        nodes.append(np.asarray(q,dtype=np.float64)); node_memberships.append(np.asarray(membership,dtype=bool)); node_points.append(np.asarray(point,dtype=np.float64)); return len(nodes)-1
    def add_edge(start_node:int,end_node:int,q_values:np.ndarray,active:np.ndarray,targets:np.ndarray,kind:str) -> int:
        membership=edge_membership_from_q(robot,surface,inverse,q_values,active,config,node_memberships[start_node],node_memberships[end_node])
        summary=summarize_ordered_membership(membership,weights,active=active)
        if not np.array_equal(summary.start_membership,node_memberships[start_node]) or not np.array_equal(summary.end_membership,node_memberships[end_node]):
            raise RuntimeError("edge_endpoint_membership_mismatch")
        edge_id=len(edges); cost=float(np.linalg.norm(np.diff(q_values,axis=0),axis=1).sum())
        edges.append(CompletionEdge(edge_id,start_node,end_node,summary,cost)); q_witness.append(q_values); active_witness.append(active); target_witness.append(targets); kinds.append(kind); return edge_id
    try:
        for family_position,row in enumerate(candidates):
            data=np.load(ROOT/row["witness_file"],allow_pickle=True)
            chain=[]
            segment_indices=sorted(int(key.split("_")[1]) for key in data.files if key.startswith("q_") and key.split("_")[1].isdigit())
            for segment_index in segment_indices:
                q=np.asarray(data[f"q_{segment_index}"],dtype=np.float64); targets=np.asarray(data[f"surface_points_{segment_index}"],dtype=np.float64)
                bounds=macro_boundaries(targets,float(config["graph"]["macro_target_length_m"]))
                previous_node=None
                for lo,hi in zip(bounds[:-1],bounds[1:]):
                    dense_q,dense_targets=densify_graph_macro(q[lo:hi+1],targets[lo:hi+1],robot,config)
                    membership=edge_membership_from_q(robot,surface,inverse,dense_q,np.ones(len(dense_q),bool),config)
                    if previous_node is None:
                        previous_node=add_node(dense_q[0],membership[:,0],dense_targets[0])
                    end_node=add_node(dense_q[-1],membership[:,-1],dense_targets[-1])
                    edge_id=add_edge(previous_node,end_node,dense_q,np.ones(len(dense_q),bool),dense_targets,"on")
                    reverse_id=add_edge(end_node,previous_node,dense_q[::-1].copy(),np.ones(len(dense_q),bool),dense_targets[::-1].copy(),"on_reverse")
                    chain.append(edge_id)
                    if family_position==0: reference_edge_ids.append(edge_id)
                    previous_node=end_node
            family_chains.append(chain)
    except RuntimeError as exc:
        if str(exc)=="node_limit":
            return {"status":"limit","stop_reason":"node_limit","nodes":len(nodes),"edges":len(edges),"connection_attempts":0,"connection_edges":0,"graph_hash":None,"reference_edge_ids":reference_edge_ids}
        raise
    pairs=[]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i==j: continue
            point_distance=float(np.linalg.norm(node_points[i]-node_points[j])); joint_distance=float(np.linalg.norm(nodes[i]-nodes[j]))
            if point_distance<=4.0*config["coverage"]["footprint_radius_m"] and joint_distance<=2.0:
                pairs.append((joint_distance,point_distance,i,j))
    pairs.sort()
    attempts=0; connections=0; cap=int(config["graph"]["max_connection_attempts"])
    existing={(edge.start,edge.end) for edge in edges}
    for _,_,i,j in pairs:
        if attempts>=cap: break
        if (i,j) in existing: continue
        attempts+=1; q=interpolate_q(nodes[i],nodes[j],config["robot"]["dense_q_step_rad"])
        if len(q)<3: q=np.linspace(nodes[i],nodes[j],3)
        if not all(robot.evaluate_configuration(value).collision_free for value in q): continue
        active=np.zeros(len(q),dtype=bool); active[[0,-1]]=True
        targets=np.vstack((node_points[i],np.full((len(q)-2,3),np.nan),node_points[j]))
        add_edge(i,j,q,active,targets,"off_reconfiguration"); existing.add((i,j)); connections+=1
    graph_hash=hash_graph_arrays(tuple(node_memberships),tuple(edges),weights,tuple(nodes),tuple(q_witness),tuple(active_witness))
    save_graph(graph_file,nodes,node_memberships,node_points,edges,weights,q_witness,active_witness,target_witness,kinds,graph_hash,reference_edge_ids)
    return {"status":"ready","stop_reason":"connection_attempt_limit" if attempts>=cap else "candidate_pairs_exhausted",
            "nodes":len(nodes),"edges":len(edges),"connection_attempts":attempts,"connection_edges":connections,
            "graph_hash":graph_hash,"reference_edge_ids":reference_edge_ids}


def macro_boundaries(points: np.ndarray, target_length: float) -> list[int]:
    boundaries=[0]; accumulated=0.0
    for index,length in enumerate(np.linalg.norm(np.diff(points,axis=0),axis=1),start=1):
        accumulated+=float(length)
        if accumulated>=target_length and index<len(points)-1: boundaries.append(index); accumulated=0.0
    if boundaries[-1]!=len(points)-1: boundaries.append(len(points)-1)
    return boundaries


def densify_graph_macro(q: np.ndarray, targets: np.ndarray, robot: UR5eKinematics, config: dict[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    q_out=[q[0]]; target_out=[targets[0]]
    for q0,q1,p0,p1 in zip(q[:-1],q[1:],targets[:-1],targets[1:]):
        intervals=max(1,int(np.ceil(np.linalg.norm(p1-p0)/config["coverage"]["main_path_sample_spacing_m"])),
                      int(np.ceil(np.max(np.abs(q1-q0))/config["robot"]["dense_q_step_rad"])))
        for value in np.linspace(0,1,intervals+1)[1:]:
            q_out.append((1-value)*q0+value*q1); target_out.append((1-value)*p0+value*p1)
    return np.asarray(q_out),np.asarray(target_out)


def edge_membership_from_q(robot:UR5eKinematics,surface,inverse:np.ndarray,q:np.ndarray,active:np.ndarray,config:dict[str,Any],start=None,end=None)->np.ndarray:
    base=np.asarray([robot.forward(value)[0] for value in q])
    homogeneous=np.column_stack((base,np.ones(len(base))))
    surface_points=(homogeneous@inverse.T)[:,:3]
    membership=_ordered_footprint_membership(surface,surface_points,footprint_radius=config["coverage"]["footprint_radius_m"])
    membership[:,~np.asarray(active,dtype=bool)]=False
    if start is not None and not np.array_equal(membership[:,0],np.asarray(start,dtype=bool)):
        raise RuntimeError("recomputed_start_membership_mismatch")
    if end is not None and not np.array_equal(membership[:,-1],np.asarray(end,dtype=bool)):
        raise RuntimeError("recomputed_end_membership_mismatch")
    return membership


def save_graph(path:Path,nodes,node_memberships,node_points,edges,weights,q_witness,active_witness,target_witness,kinds,graph_hash,reference_edge_ids):
    np.savez_compressed(path,nodes=np.asarray(nodes),node_membership=np.asarray(node_memberships),node_points=np.asarray(node_points),weights=weights,
        edge_start=np.asarray([e.start for e in edges]),edge_end=np.asarray([e.end for e in edges]),edge_cost=np.asarray([e.joint_cost for e in edges]),
        edge_footprint=np.asarray([e.summary.footprint for e in edges]),edge_counts=np.asarray([e.summary.episode_counts for e in edges]),
        edge_start_membership=np.asarray([e.summary.start_membership for e in edges]),edge_end_membership=np.asarray([e.summary.end_membership for e in edges]),
        edge_mass=np.asarray([e.summary.weighted_episode_mass for e in edges]),edge_off_to_on=np.asarray([e.summary.off_to_on_count for e in edges]),
        q_witness=np.asarray(q_witness,dtype=object),active_witness=np.asarray(active_witness,dtype=object),target_witness=np.asarray(target_witness,dtype=object),
        kinds=np.asarray(kinds),graph_hash=np.asarray(graph_hash),reference_edge_ids=np.asarray(reference_edge_ids,dtype=np.int64))


def load_graph(path:Path)->tuple[SearchGraph,dict[str,Any]]:
    data=np.load(path,allow_pickle=True); weights=np.asarray(data["weights"],dtype=float); edges=[]
    from diffusion_coverage.coverage.episode_summary import EpisodeEdgeSummary
    for i in range(len(data["edge_start"])):
        summary=EpisodeEdgeSummary(np.asarray(data["edge_footprint"][i],bool),np.asarray(data["edge_counts"][i],np.int64),
            np.asarray(data["edge_start_membership"][i],bool),np.asarray(data["edge_end_membership"][i],bool),float(data["edge_mass"][i]),int(data["edge_off_to_on"][i]))
        edges.append(CompletionEdge(i,int(data["edge_start"][i]),int(data["edge_end"][i]),summary,float(data["edge_cost"][i])))
    graph=SearchGraph(tuple(np.asarray(x,bool) for x in data["node_membership"]),tuple(edges),weights,str(data["graph_hash"].item()))
    extras={key:data[key] for key in ("nodes","node_points","q_witness","active_witness","target_witness","kinds","reference_edge_ids")}
    return graph,extras


def stage_compare(config: dict[str,Any], output: Path) -> None:
    require_checkpoint(output,"build-graph"); verify_frozen_diff(config,output)
    graph_manifest=json.loads((output/"graph_manifest.json").read_text())["graphs"]
    ready=[row for row in graph_manifest if row["status"]=="ready"]
    result_rows=[]; anytime_rows=[]; prune_rows=[]; mechanism=None
    witness_dir=output/"selected_plan_witnesses"; witness_dir.mkdir(exist_ok=True)
    for scene_index,row in enumerate(ready):
        graph,extras=load_graph(ROOT/row["graph_file"])
        if graph.graph_hash!=row["graph_hash"]: raise RuntimeError("graph hash changed before comparison")
        connection_limited=int(row["connection_edges"])==0
        for k in config["on_segment_budgets"]:
            outcomes={}
            arm_order=("S0","S1") if (scene_index+int(k))%2==0 else ("S1","S0")
            for arm in arm_order:
                tracemalloc.start(); start=perf_counter()
                result=search_history_graph(graph,start_node=0,maximum_on_segments=int(k),
                    missed_tolerance=config["coverage"]["missed_tolerance"],repeat_tolerance=config["coverage"]["repeat_tolerance"],
                    use_completion_bound=arm=="S1",wall_time_s=config["search"]["wall_time_s"],
                    expanded_limit=config["search"]["expanded_label_limit"],checkpoint_times=tuple(config["search"]["checkpoints_s"]),
                    tolerance=config["search"]["conservative_float_tolerance"])
                total=perf_counter()-start; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); outcomes[arm]=result
                incumbent=result.incumbent
                miss=None if incumbent is None else float(graph.weights[~incumbent.covered].sum()/graph.weights.sum())
                plan_file=None
                if incumbent is not None:
                    plan_file=witness_dir/f"{row['surface_id']}_{row['scene_id']}_k{k}_{arm}.npz"
                    save_selected_plan(plan_file,incumbent.path,extras,graph,row,arm,k)
                result_rows.append({
                    "surface_id":row["surface_id"],"scene_id":row["scene_id"],"k":k,"arm":arm,"graph_hash":graph.graph_hash,
                    "found":incumbent is not None,"first_solution_s":result.first_solution_seconds,"total_search_s":total,
                    "termination":result.termination,"optimality_proved":result.optimality_proved,
                    "on_segments":None if incumbent is None else incumbent.used_on_segments,
                    "E_miss":miss,"E_rep":None if incumbent is None else incumbent.repeat_error,
                    "E_NUC":None if incumbent is None else miss+incumbent.repeat_error,"J_q":None if incumbent is None else incumbent.joint_cost,
                    "expanded":result.metrics.expanded,"generated":result.metrics.generated,"peak_memory_bytes":peak,
                    "connection_capability_limited":bool(k==2 and connection_limited),
                    "selected_plan_witness":None if plan_file is None else str(plan_file.relative_to(ROOT)),
                })
                prune_rows.append({"surface_id":row["surface_id"],"scene_id":row["scene_id"],"k":k,"arm":arm,
                    "dominance":result.metrics.dominance_pruned,"repeat_budget":result.metrics.repeat_pruned,
                    "segment_budget":result.metrics.segment_pruned,
                    "segment_budget_reachable_area":result.metrics.segment_budget_reachability_pruned,
                    "reachable_area":result.metrics.reachability_pruned,
                    "completion_bound":result.metrics.completion_bound_pruned,"bound_calls":result.metrics.completion_bound_calls,
                    "bound_cache_hits":result.metrics.completion_bound_cache_hits,"bound_time_s":result.metrics.completion_bound_seconds,
                    "bound_nonzero_fraction":None if not result.metrics.completion_bound_calls else result.metrics.completion_bound_positive/result.metrics.completion_bound_calls,
                    "bound_stronger_fraction":None if not result.metrics.completion_bound_calls else result.metrics.completion_bound_stronger/result.metrics.completion_bound_calls})
                anytime_rows.extend(make_anytime_rows(row,k,arm,result,config))
                if mechanism is None and result.mechanism_sample is not None:
                    mechanism={"surface_id":row["surface_id"],"scene_id":row["scene_id"],"k":k,"arm":arm,"graph_hash":graph.graph_hash,
                               **result.mechanism_sample,"interpretation":"prefix remains within repeat budget and has graph-reachable uncovered targets, but the area-weighted completion bound exceeds the remaining repeat budget"}
                print(row["scene_id"],"k",k,arm,result.termination,"found",incumbent is not None,"expanded",result.metrics.expanded,flush=True)
            a,b=outcomes["S0"],outcomes["S1"]
            if a.optimality_proved and b.optimality_proved:
                ao=None if a.incumbent is None else (a.incumbent.used_on_segments,a.incumbent.joint_cost)
                bo=None if b.incumbent is None else (b.incumbent.used_on_segments,b.incumbent.joint_cost)
                if ao!=bo: raise RuntimeError("S0/S1 disagree on a completely searched graph")
    write_csv(output/"search_results.csv",result_rows); write_csv(output/"anytime.csv",anytime_rows); write_csv(output/"prune_breakdown.csv",prune_rows)
    write_json(output/"mechanism_example.json", mechanism or {"found":False,"reason":"no natural qualifying prefix occurred"})
    checkpoint(output,"compare",{"complete":True,"runs":len(result_rows),"mechanism_example":mechanism is not None})


def save_selected_plan(path:Path,edge_ids:tuple[int,...],extras:dict[str,Any],graph:SearchGraph,manifest_row:dict[str,Any],arm:str,k:int)->None:
    q_parts=[]; active_parts=[]; target_parts=[]
    for position,edge_id in enumerate(edge_ids):
        q=np.asarray(extras["q_witness"][edge_id],dtype=float); active=np.asarray(extras["active_witness"][edge_id],dtype=bool); target=np.asarray(extras["target_witness"][edge_id],dtype=float)
        if position: q=q[1:]; active=active[1:]; target=target[1:]
        q_parts.append(q); active_parts.append(active); target_parts.append(target)
    np.savez_compressed(path,q=np.concatenate(q_parts),active=np.concatenate(active_parts),surface_targets=np.concatenate(target_parts),
        edge_ids=np.asarray(edge_ids),graph_hash=np.asarray(graph.graph_hash),surface_id=np.asarray(manifest_row["surface_id"]),scene_id=np.asarray(manifest_row["scene_id"]),
        arm=np.asarray(arm),k=np.asarray(k))


def make_anytime_rows(row:dict[str,Any],k:int,arm:str,result,config:dict[str,Any])->list[dict[str,Any]]:
    values=[]; checkpoints={float(item["seconds"]):item for item in result.checkpoints}
    for seconds in config["search"]["checkpoints_s"]:
        item=checkpoints.get(float(seconds))
        if item is None and result.optimality_proved and result.elapsed_seconds<=seconds:
            incumbent=result.incumbent; item={"found":incumbent is not None,"used_reconfigurations":None if incumbent is None else incumbent.used_on_segments-1,
                "joint_cost":None if incumbent is None else incumbent.joint_cost,"expanded":result.metrics.expanded,"generated":result.metrics.generated}
        values.append({"surface_id":row["surface_id"],"scene_id":row["scene_id"],"k":k,"arm":arm,"seconds":seconds,
            "observed":item is not None,"found":None if item is None else item["found"],"used_reconfigurations":None if item is None else item["used_reconfigurations"],
            "J_q":None if item is None else item["joint_cost"],"expanded":None if item is None else item["expanded"],"generated":None if item is None else item["generated"]})
    return values


def stage_verify(config:dict[str,Any],output:Path)->None:
    require_checkpoint(output,"compare"); verify_frozen_diff(config,output)
    rows=read_csv(output/"search_results.csv"); scenes={(item["surface_id"],item["candidate_id"]):item for item in scene_records(config)}
    verified=[]
    for row in rows:
        if not as_bool(row["found"]): continue
        witness=ROOT/row["selected_plan_witness"]; data=np.load(witness)
        q=np.asarray(data["q"],float); active=np.asarray(data["active"],bool); targets=np.asarray(data["surface_targets"],float)
        scene=scenes[(row["surface_id"],row["scene_id"])]
        main=verify_plan_arrays(q,active,targets,scene,config,verification=False)
        dense=verify_plan_arrays(q,active,targets,scene,config,verification=True)
        sensitive=main["overall_pass"]!=dense["overall_pass"]
        record={"surface_id":row["surface_id"],"scene_id":row["scene_id"],"k":row["k"],"arm":row["arm"],"graph_hash":row["graph_hash"],
            **{f"main_{key}":value for key,value in main.items()},**{f"verification_{key}":value for key,value in dense.items()},
            "numerical_resolution_sensitive":sensitive,"certificate_scope":"accepted under sampled checker; no continuous interval certificate",
            "collision_scope":config["process_contract"]["collision_claim"]}
        verified.append(record); write_json(witness.with_suffix(".verification.json"),record)
    write_csv(output/"final_verification.csv",verified)
    commands="\n".join([f"python scripts/run_global_completion_v1.py --stage {stage}" for stage in STAGES]+["python scripts/summarize_global_completion_v1.py"])+"\n"
    (output/"reproduction_commands.txt").write_text(commands)
    checkpoint(output,"verify",{"complete":True,"plans":len(verified),"resolution_sensitive":sum(row["numerical_resolution_sensitive"] for row in verified)})


def verify_plan_arrays(q:np.ndarray,active:np.ndarray,targets:np.ndarray,scene:dict[str,Any],config:dict[str,Any],*,verification:bool)->dict[str,Any]:
    robot=robot_from_config(config); surface=make_surface(scene["surface_id"],config,verification=verification)
    transform=np.asarray(scene["transform_base_from_surface"],float); segments=[]; target_segments=[]; axis_segments=[]
    spacing=config["coverage"]["verification_path_sample_spacing_m" if verification else "main_path_sample_spacing_m"]
    q_step=config["robot"]["verification_q_step_rad" if verification else "dense_q_step_rad"]
    for lo,hi in active_runs(active):
        if hi-lo<1: continue
        q_dense,target_dense=densify_analytic_segment(q[lo:hi+1],targets[lo:hi+1],scene["surface_id"],surface.metadata,spacing,q_step)
        points,normals=analytical_points_and_normals(scene["surface_id"],target_dense,surface.metadata)
        positions,axes=transform_surface_pose_path(points,normals,transform)
        segments.append(q_dense); target_segments.append(positions); axis_segments.append(axes)
    strict=check_strict_coverage_execution(robot,tuple(segments),tuple(target_segments),tuple(axis_segments),surface,transform,
        footprint_radius=config["coverage"]["footprint_radius_m"],position_tolerance=config["robot"]["position_tolerance_m"],
        axis_tolerance=np.deg2rad(config["robot"]["axis_tolerance_degrees"]),characteristic_length=config["robot"]["characteristic_length_m"],
        sigma_safe=config["robot"]["sigma_safe"],missed_tolerance=config["coverage"]["missed_tolerance"],repeat_tolerance=config["coverage"]["repeat_tolerance"],
        interpolation_joint_step=q_step,coverage_path_sample_spacing=spacing)
    all_collision=all(robot.evaluate_configuration(value).collision_free for value in q)
    total_j=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
    coverage=strict.coverage_metrics
    return {"overall_pass":bool(strict.overall_pass and all_collision),"failure_reasons":";".join(strict.failure_reasons),
        "E_miss":None if coverage is None else coverage.missed_error,"E_rep":None if coverage is None else coverage.repeat_error,
        "E_NUC":None if coverage is None else coverage.nuc_error,"J_q_including_off":total_j,"on_segments":len(segments),
        "max_position_error_m":strict.max_position_error,"max_axis_error_degrees":float(np.rad2deg(strict.max_axis_error)),
        "min_sigma_min_5":strict.min_sigma_min_5,"min_joint_margin":strict.min_joint_limit_margin,"collision_pass":all_collision,
        "trajectory_spacing_m":spacing,"surface_samples":surface.num_samples,"q_step_rad":q_step}


def active_runs(active:np.ndarray)->list[tuple[int,int]]:
    runs=[]; start=None
    for index,value in enumerate(active):
        if value and start is None: start=index
        if start is not None and (not value or index==len(active)-1):
            end=index if value else index-1
            if end>=start: runs.append((start,end))
            start=None
    return runs


def densify_analytic_segment(q:np.ndarray,targets:np.ndarray,surface_id:str,metadata:dict[str,Any],spacing:float,q_step:float)->tuple[np.ndarray,np.ndarray]:
    q_out=[q[0]]; targets_out=[targets[0]]
    for q0,q1,p0,p1 in zip(q[:-1],q[1:],targets[:-1],targets[1:]):
        intervals=max(1,int(np.ceil(np.linalg.norm(p1-p0)/spacing)),int(np.ceil(np.max(np.abs(q1-q0))/q_step)))
        for value in np.linspace(0,1,intervals+1)[1:]:
            q_out.append((1-value)*q0+value*q1); provisional=((1-value)*p0+value*p1)[None,:]
            projected,_=analytical_points_and_normals(surface_id,provisional,metadata); targets_out.append(projected[0])
    return np.asarray(q_out),np.asarray(targets_out)


def hash_graph_arrays(node_memberships:tuple[np.ndarray,...],edges:tuple[CompletionEdge,...],weights:np.ndarray,*extra_groups)->str:
    digest=hashlib.sha256()
    def update(value):
        array=np.ascontiguousarray(value); digest.update(str(array.dtype).encode()); digest.update(np.asarray(array.shape,dtype="<i8").tobytes()); digest.update(array.tobytes())
    update(weights)
    for membership in node_memberships: update(membership)
    for edge in edges:
        update(np.asarray([edge.edge_id,edge.start,edge.end,edge.summary.off_to_on_count],dtype=np.int64)); update(np.asarray([edge.joint_cost,edge.summary.weighted_episode_mass]));
        update(edge.summary.footprint); update(edge.summary.episode_counts); update(edge.summary.start_membership); update(edge.summary.end_membership)
    for group in extra_groups:
        for value in group: update(value)
    return digest.hexdigest()


def verify_frozen_diff(config:dict[str,Any],output:Path)->None:
    manifest=json.loads((output/"manifest.json").read_text())
    revisions=sorted(output.glob("manifest_revision_*.json"))
    latest=json.loads(revisions[-1].read_text()) if revisions else manifest
    expected=latest.get("code_state_sha256")
    actual,_=code_state_snapshot()
    if expected is None or actual!=expected:
        raise RuntimeError("code diff changed after manifest freeze; create a recorded manifest revision before continuing")


def code_state_snapshot()->tuple[str,list[dict[str,Any]]]:
    status=subprocess.run(["git","status","--porcelain=v1","--untracked-files=all"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()
    files=[]; digest=hashlib.sha256()
    for line in sorted(status):
        state=line[:2]; relative=line[3:]; path=ROOT/relative
        if not path.is_file(): continue
        value=file_hash(path); record={"status":state,"path":relative,"sha256":value,"bytes":path.stat().st_size}; files.append(record)
        digest.update(json.dumps(record,sort_keys=True,separators=(",",":")).encode())
    return digest.hexdigest(),files


def checkpoint(output:Path,stage:str,payload:dict[str,Any])->None:
    write_json(output/f"{stage}.checkpoint.json",{"stage":stage,**payload})


def require_checkpoint(output:Path,stage:str)->None:
    path=output/f"{stage}.checkpoint.json"
    if not path.exists() or not json.loads(path.read_text()).get("complete",False): raise RuntimeError(f"required stage {stage!r} is incomplete")


def write_json(path:Path,value:Any)->None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def write_exclusive_json(path:Path,value:Any)->None:
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"w") as handle: json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False); handle.write("\n")


def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def read_csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="") as handle:return list(csv.DictReader(handle))


def as_bool(value:Any)->bool:
    return value is True or str(value).lower()=="true"


def file_hash(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
