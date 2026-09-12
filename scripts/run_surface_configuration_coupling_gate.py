#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import numpy as np

from diffusion_coverage.diagnostics.anisotropy import absolute_joint_limit_margin, continue_probe_from_shared_q
from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract, make_e06_surface
from diffusion_coverage.diagnostics.local_deformation import (
    AdmissionLimits,
    cumulative_length,
    deform_surface_path,
    hard_admission,
    maximum_turn_angle,
    terminal_q_mismatch,
)
from diffusion_coverage.diagnostics.surface_configuration_coupling import (
    frozen_surface_candidate_bank,
    minimum_cost_layered_lift,
    select_verified_incumbent,
    tangent_metric_diagnostics,
)
from diffusion_coverage.geometry.robot_surface_metric import smooth_surface_normal
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


def parse_args():
    parser = argparse.ArgumentParser(description="E06-J coupling capacity gate")
    parser.add_argument("--stage", choices=("freeze-candidates", "run"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surface_configuration_coupling_gate_v1.json")
    parser.add_argument("--source-windows", type=Path, default=ROOT / "results/riemannian_local_deformation_v1/frozen_windows.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/surface_configuration_coupling_gate_v1")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--window-id", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args(); config = json.loads(args.config.read_text()); source = json.loads(args.source_windows.read_text())
    if source.get("content_hash") != config["source_windows_hash"]:
        raise RuntimeError("source E06-R2 window hash does not match preregistration")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze-candidates":
        freeze_candidates(config, source, args.output); return
    frozen_path = args.output / "frozen_surface_candidates.json"
    if not frozen_path.exists():
        raise RuntimeError("freeze and commit surface candidates before F3 execution")
    frozen = json.loads(frozen_path.read_text())
    if not frozen.get("frozen_before_F3"):
        raise RuntimeError("candidate bank was not frozen before F3")
    archived, _, _ = load_e06_contract(ROOT)
    write_json(args.output / "config.json", config)
    run_all(config, archived, source, frozen, args.output, args.jobs, set(args.window_id))


def freeze_candidates(config, source, output):
    cfg = config["surface_candidates"]
    bank = frozen_surface_candidate_bank(cfg["nonzero_candidates"], cfg["maximum_control_displacement_m"], config["seed"])
    payload = {
        "experiment": config["experiment"], "frozen_before_F3": True,
        "source_windows_hash": source["content_hash"], "sampler": cfg["sampler"], "seed": config["seed"],
        "parameters": bank.tolist(),
    }
    payload["content_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    write_json(output / "frozen_surface_candidates.json", payload)
    write_json(output / "frozen_windows.json", source)
    write_json(output / "config.json", config)
    print(json.dumps({"candidate_count": len(bank), "content_hash": payload["content_hash"]}, indent=2))


def make_robot(archived):
    cfg = archived["config"]["robot"]
    return UR5eKinematics(cfg["model"], site_name=cfg["site_name"], tool_axis_index=cfg["tool_axis_index"], tool_axis_sign=cfg["tool_axis_sign"])


def run_all(config, archived, windows, frozen, output, jobs, requested):
    items = [item for group in windows["windows_by_scene"].values() for item in group]
    if requested:
        items = [item for item in items if item["window_id"] in requested]
    arguments = [(config, archived, frozen, item) for item in items]
    completed = []
    if jobs <= 1:
        completed = [run_window(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(run_window, argument): argument[-1]["window_id"] for argument in arguments}
            for future in as_completed(futures):
                completed.append(future.result()); print(f"completed {futures[future]}", flush=True)
    completed.sort(key=lambda result: result["F0"]["window_id"])
    for formulation in ("F0", "F1", "F2", "F3"):
        write_csv(output / f"{formulation}_results.csv", [result[formulation] for result in completed])
    diagnostics = [row for result in completed for row in result["diagnostics"]]
    write_csv(output / "solver_diagnostics.csv", diagnostics)


def run_window(arguments):
    config, archived, frozen, window = arguments
    robot = make_robot(archived)
    surface = make_e06_surface(archived["config"], window["surface_id"], int(archived["frozen_contract"]["coverage_samples_per_face"]))
    witness = np.load(ROOT / window["witness_file"])
    start, stop = window["start_index"], window["stop_index"]
    q0 = np.asarray(witness["q"][start : stop + 1], dtype=np.float64)
    positions0 = np.asarray(witness["desired_positions"][start : stop + 1], dtype=np.float64)
    axes0 = np.asarray(witness["desired_axes"][start : stop + 1], dtype=np.float64)
    transform = np.asarray(window["transform_base_from_surface"], dtype=np.float64)
    points0 = (positions0 - transform[:3, 3]) @ transform[:3, :3]
    points0, normals0 = smooth_surface_normal(surface, points0)
    controls = np.asarray(window["control_indices_local"], dtype=np.int64)

    F0 = baseline_result(robot, surface, transform, q0, positions0, axes0, points0, config, archived, window)
    diagnostics = []
    F1, diagnostic = configuration_only(robot, surface, transform, q0, positions0, axes0, points0, F0, config, archived, window)
    diagnostics.append(diagnostic)
    F2 = dict(F0); F2.update({"formulation": "F2", "optimized": False, "solver": "single_warm_continuation_surface_bank", "surface_candidate_id": "baseline"})
    F3 = dict(F0); F3.update({"formulation": "F3", "optimized": False, "solver": "joint_surface_plus_layered_q_beam", "surface_candidate_id": "baseline"})
    for candidate_id, parameters in enumerate(np.asarray(frozen["parameters"], dtype=np.float64)):
        geometry = construct_surface_candidate(surface, transform, points0, controls, parameters, window, config)
        if geometry["failure_reason"] is not None:
            for formulation in ("F2", "F3"):
                diagnostics.append(diag_row(window, formulation, candidate_id, False, geometry["failure_reason"], 0, 0, None))
            continue
        curve, positions, axes = geometry["curve"], geometry["positions"], geometry["axes"]
        candidate2, diagnostic2 = surface_only_candidate(robot, surface, transform, q0, curve, positions, axes, F0, config, archived, window, candidate_id)
        F2, accepted2 = select_verified_incumbent(F2, candidate2); diagnostic2["selected_update"] = accepted2; diagnostics.append(diagnostic2)
        candidate3, diagnostic3 = coupled_candidate(robot, surface, transform, q0, curve, positions, axes, F0, config, archived, window, candidate_id)
        F3, accepted3 = select_verified_incumbent(F3, candidate3); diagnostic3["selected_update"] = accepted3; diagnostics.append(diagnostic3)
    F2 = finalize_diagnostics(robot, surface, transform, F2, config)
    F3 = finalize_diagnostics(robot, surface, transform, F3, config)
    return {"F0": strip_arrays(F0), "F1": strip_arrays(F1), "F2": strip_arrays(F2), "F3": strip_arrays(F3), "diagnostics": diagnostics}


def baseline_result(robot, surface, transform, q, positions, axes, points, config, archived, window):
    strict = strict_check(robot, surface, transform, q, positions, axes, config, archived)
    if not strict.overall_pass:
        raise RuntimeError(f"F0 strict replay failed for {window['window_id']}: {strict.failure_reasons}")
    direct = compute_joint_execution_cost((q,)).weighted_joint_length
    if abs(direct - strict.execution_cost.weighted_joint_length) > 1e-12:
        raise RuntimeError("F0 archived replay changed J_q")
    result = common_result(window, "F0", q, points, strict, float(cumulative_length(points)[-1]), 0.0, "fixed_archived_witness", "baseline")
    result.update({"optimized": False, "baseline_reconstruction_error": abs(direct - strict.execution_cost.weighted_joint_length), "J_q": direct})
    return finalize_diagnostics(robot, surface, transform, result, config)


def configuration_only(robot, surface, transform, q0, positions, axes, points, baseline, config, archived, window):
    search = layered_search(robot, positions, axes, q0, config, reference=q0)
    diagnostic = diag_row(window, "F1", "fixed_surface", search.found, search.failure_reason, search.expanded_states, search.generated_candidates, search.solve_time)
    incumbent = dict(baseline); incumbent.update({"formulation": "F1", "solver": "fixed_surface_layered_q_beam", "optimized": False, "surface_candidate_id": "fixed"})
    if not search.found:
        return incumbent, diagnostic
    strict = strict_check(robot, surface, transform, search.q_path, positions, axes, config, archived)
    candidate = common_result(window, "F1", search.q_path, points, strict, baseline["surface_length"], 0.0, "fixed_surface_layered_q_beam", "fixed")
    candidate["optimized"] = bool(candidate["admitted"] and candidate["J_q"] < baseline["J_q"] - 1e-12)
    selected, accepted = select_verified_incumbent(incumbent, candidate)
    diagnostic.update({"strict_pass": strict.overall_pass, "admitted": candidate["admitted"], "selected_update": accepted, "J_q": candidate["J_q"]})
    return finalize_diagnostics(robot, surface, transform, selected, config), diagnostic


def construct_surface_candidate(surface, transform, points0, controls, parameters, window, config):
    try:
        curve = deform_surface_path(surface, points0, controls, parameters, maximum_displacement=config["surface_candidates"]["maximum_control_displacement_m"], maximum_retraction_step=config["surface_candidates"]["retraction_maximum_step_m"])
        turn_limit = max(np.deg2rad(45.0), window["baseline_maximum_turn_rad"] + np.deg2rad(15.0))
        if not curve.topology_preserved or curve.maximum_turn > turn_limit:
            return {"failure_reason": "surface_topology_or_turn"}
        relative = abs(curve.surface_length - float(cumulative_length(points0)[-1])) / float(cumulative_length(points0)[-1])
        if relative > config["admission"]["maximum_relative_surface_length_change"]:
            return {"failure_reason": "surface_length"}
        positions = curve.points @ transform[:3, :3].T + transform[:3, 3]
        axes = -(curve.normals @ transform[:3, :3].T); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        return {"failure_reason": None, "curve": curve, "positions": positions, "axes": axes}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return {"failure_reason": "surface_numerical_failure"}


def surface_only_candidate(robot, surface, transform, q0, curve, positions, axes, baseline, config, archived, window, candidate_id):
    continuation = continue_probe_from_shared_q(robot, q0[0], positions, axes, maximum_joint_step=config["robot_contract"]["maximum_continuation_joint_step_rad"], position_tolerance=config["robot_contract"]["construction_position_tolerance_m"], axis_tolerance=np.deg2rad(config["robot_contract"]["axis_tolerance_degrees"]))
    diagnostic = diag_row(window, "F2", candidate_id, continuation.feasible, continuation.failure_reason, 0, len(positions), None)
    if not continuation.feasible or len(continuation.q_path) != len(positions):
        return rejected_candidate(window, "F2", candidate_id, continuation.failure_reason), diagnostic
    candidate = verified_surface_candidate(robot, surface, transform, continuation.q_path, curve, positions, axes, baseline, config, archived, window, "F2", candidate_id, "single_warm_continuation_surface_bank")
    diagnostic.update({"strict_pass": candidate.get("strict_pass"), "admitted": candidate["admitted"], "J_q": candidate.get("J_q")})
    return candidate, diagnostic


def coupled_candidate(robot, surface, transform, q0, curve, positions, axes, baseline, config, archived, window, candidate_id):
    search = layered_search(robot, positions, axes, q0, config, reference=None)
    diagnostic = diag_row(window, "F3", candidate_id, search.found, search.failure_reason, search.expanded_states, search.generated_candidates, search.solve_time)
    if not search.found:
        return rejected_candidate(window, "F3", candidate_id, search.failure_reason), diagnostic
    candidate = verified_surface_candidate(robot, surface, transform, search.q_path, curve, positions, axes, baseline, config, archived, window, "F3", candidate_id, "joint_surface_plus_layered_q_beam")
    diagnostic.update({"strict_pass": candidate.get("strict_pass"), "admitted": candidate["admitted"], "J_q": candidate.get("J_q")})
    return candidate, diagnostic


def layered_search(robot, positions, axes, q0, config, reference):
    cfg = config["q_search"]; contract = config["robot_contract"]
    return minimum_cost_layered_lift(
        robot, positions, axes, q0[0], q0[-1], characteristic_length=contract["characteristic_length_m"], sigma_safe=contract["sigma_safe"],
        axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), position_tolerance=contract["construction_position_tolerance_m"],
        maximum_joint_step=contract["maximum_continuation_joint_step_rad"], beam_width=cfg["beam_width"], null_offsets=tuple(cfg["null_offsets_rad"]),
        deduplication_radius=cfg["state_deduplication_rad"], ik_max_iterations=cfg["ik_max_iterations"], reference_q_path=reference,
    )


def verified_surface_candidate(robot, surface, transform, q, curve, positions, axes, baseline, config, archived, window, formulation, candidate_id, solver):
    strict = strict_check(robot, surface, transform, q, positions, axes, config, archived)
    coverage = strict.coverage_metrics; mismatch = terminal_q_mismatch(q[-1], baseline["q_path"][-1])
    limits = AdmissionLimits(config["admission"]["maximum_relative_surface_length_change"], config["admission"]["delta_NUC"], config["robot_contract"]["maximum_terminal_q_mismatch_rad"], config["robot_contract"]["sigma_safe"])
    admitted, reasons = hard_admission(surface_length=curve.surface_length, baseline_surface_length=baseline["surface_length"], nuc_error=np.inf if coverage is None else coverage.nuc_error, missed_error=np.inf if coverage is None else coverage.missed_error, repeat_error=np.inf if coverage is None else coverage.repeat_error, baseline_nuc_error=baseline["E_NUC"], baseline_missed_error=baseline["E_miss"], baseline_repeat_error=baseline["E_rep"], terminal_mismatch=mismatch, strict_pass=strict.overall_pass, topology_preserved=curve.topology_preserved, limits=limits)
    result = common_result(window, formulation, q, curve.points, strict, curve.surface_length, mismatch, solver, candidate_id)
    result.update({"admitted": admitted, "admission_reasons": ";".join(reasons), "strict_pass": strict.overall_pass, "optimized": admitted and result["J_q"] < baseline["J_q"] - 1e-12})
    return result


def common_result(window, formulation, q, points, strict, surface_length, mismatch, solver, candidate_id):
    coverage = strict.coverage_metrics
    return {
        "window_id": window["window_id"], "surface_id": window["surface_id"], "anisotropy_level": window["anisotropy_level"], "placement_id": window["placement_id"],
        "formulation": formulation, "solver": solver, "surface_candidate_id": candidate_id, "admitted": bool(strict.overall_pass), "admission_reasons": "" if strict.overall_pass else ";".join(strict.failure_reasons),
        "J_q": None if strict.execution_cost is None else strict.execution_cost.weighted_joint_length, "surface_length": surface_length,
        "E_NUC": None if coverage is None else coverage.nuc_error, "E_miss": None if coverage is None else coverage.missed_error, "E_rep": None if coverage is None else coverage.repeat_error,
        "min_sigma_min_5": strict.min_sigma_min_5, "min_joint_limit_margin": strict.min_joint_limit_margin, "minimum_absolute_joint_margin_rad": np.nan,
        "terminal_q_mismatch": mismatch, "max_position_error": strict.max_position_error, "max_axis_error": strict.max_axis_error,
        "q_sample_count": len(q), "q_path": np.asarray(q, dtype=np.float64), "points_surface": np.asarray(points, dtype=np.float64),
    }


def finalize_diagnostics(robot, surface, transform, result, config):
    if result.get("q_path") is None:
        return result
    result = dict(result)
    result["minimum_absolute_joint_margin_rad"] = absolute_joint_limit_margin(robot, result["q_path"])
    result.update(tangent_metric_diagnostics(robot, surface, transform, result["q_path"], result["points_surface"], characteristic_length=config["robot_contract"]["characteristic_length_m"], sigma_safe=config["robot_contract"]["sigma_safe"], finite_difference_step=config["metric_diagnostics"]["finite_difference_step_m"], sample_count=config["metric_diagnostics"]["samples"]))
    return result


def strict_check(robot, surface, transform, q, positions, axes, config, archived):
    c = config["robot_contract"]
    return check_strict_coverage_execution(robot, (q,), (positions,), (axes,), surface, transform, footprint_radius=archived["config"]["coverage"]["footprint_radius_m"], position_tolerance=c["position_tolerance_m"], axis_tolerance=np.deg2rad(c["axis_tolerance_degrees"]), characteristic_length=c["characteristic_length_m"], sigma_safe=c["sigma_safe"], missed_tolerance=1.0, repeat_tolerance=10.0, interpolation_joint_step=c["maximum_dense_q_step_rad"], coverage_path_sample_spacing=archived["frozen_contract"]["coverage_path_sample_spacing_m"])


def rejected_candidate(window, formulation, candidate_id, reason):
    return {"window_id": window["window_id"], "formulation": formulation, "surface_candidate_id": candidate_id, "admitted": False, "admission_reasons": reason or "solver_failure", "J_q": None}


def diag_row(window, formulation, candidate_id, solved, reason, expanded, generated, solve_time):
    return {"window_id": window["window_id"], "surface_id": window["surface_id"], "anisotropy_level": window["anisotropy_level"], "formulation": formulation, "surface_candidate_id": candidate_id, "solver_solved": solved, "failure_reason": reason, "expanded_states": expanded, "generated_candidates": generated, "solve_time": solve_time, "strict_pass": None, "admitted": None, "selected_update": False, "J_q": None}


def strip_arrays(result):
    return {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}


def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
