#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import numpy as np

from diffusion_coverage.diagnostics.anisotropy import absolute_joint_limit_margin, continue_probe_from_shared_q, select_uniform_indices
from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract, make_e06_surface
from diffusion_coverage.diagnostics.local_deformation import (
    AdmissionLimits,
    accept_incumbent_update,
    control_indices,
    cumulative_length,
    deform_surface_path,
    deterministic_window_seed,
    euclidean_objective,
    hard_admission,
    integrated_robot_metric_length,
    maximum_turn_angle,
    proposal_schedule,
    propose_parameters,
    terminal_q_mismatch,
    window_indices,
)
from diffusion_coverage.geometry.robot_surface_metric import smooth_surface_normal
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze or run E06-R2 local deformation")
    parser.add_argument("--stage", choices=("freeze-windows", "run"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/riemannian_local_deformation_v1.json")
    parser.add_argument("--scenes", type=Path, default=ROOT / "configs/riemannian_anisotropy_scenes_v1.json")
    parser.add_argument("--anchors", type=Path, default=ROOT / "results/riemannian_anisotropy_utility_v1/frozen_anchors.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/riemannian_local_deformation_v1")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--window-id", action="append", default=[])
    return parser.parse_args()


def make_robot(archived: dict) -> UR5eKinematics:
    cfg = archived["config"]["robot"]
    return UR5eKinematics(cfg["model"], site_name=cfg["site_name"], tool_axis_index=cfg["tool_axis_index"], tool_axis_sign=cfg["tool_axis_sign"])


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    scenes = json.loads(args.scenes.read_text())
    anchors = json.loads(args.anchors.read_text())
    if anchors.get("content_hash") != config["source_anchor_hash"]:
        raise RuntimeError("E06-R anchor hash differs from the preregistered source")
    archived, _, _ = load_e06_contract(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze-windows":
        frozen = freeze_windows(config, scenes, anchors, archived)
        write_json(args.output / "frozen_windows.json", frozen)
        print(json.dumps({key: len(value) for key, value in frozen["windows_by_scene"].items()}, indent=2))
        return
    frozen_path = args.output / "frozen_windows.json"
    if not frozen_path.exists():
        raise RuntimeError("freeze and commit frozen_windows.json before running deformation")
    frozen = json.loads(frozen_path.read_text())
    if not frozen.get("frozen_before_deformation_results"):
        raise RuntimeError("windows were not frozen before deformation")
    write_json(args.output / "config.json", config)
    run_all(config, scenes, frozen, archived, args.output, args.jobs, set(args.window_id))


def freeze_windows(config: dict, scenes: dict, anchors: dict, archived: dict) -> dict:
    robot = make_robot(archived)
    window_cfg = config["window"]
    contract = config["robot_contract"]
    windows_by_scene: dict[str, list[dict]] = {}
    rejected: dict[str, list[dict]] = {}
    for surface_id in ("saddle", "hemisphere"):
        surface = make_e06_surface(archived["config"], surface_id, 1)
        for level in ("P_low", "P_mid", "P_high"):
            key = f"{surface_id}/{level}"
            scene = scenes["selected"][surface_id][level]
            if scene["candidate_id"] != config["scenes"][surface_id][level]:
                raise RuntimeError(f"scene mismatch for {key}")
            witness = np.load(ROOT / scene["witness_file"])
            q = witness["q"]
            positions = witness["desired_positions"]
            transform = np.asarray(scene["transform_base_from_surface"], dtype=np.float64)
            points_surface = (positions - transform[:3, 3]) @ transform[:3, :3]
            points_surface, _ = smooth_surface_normal(surface, points_surface)
            cumulative = cumulative_length(points_surface)
            eligible: list[tuple[int, dict]] = []
            rejects: list[dict] = []
            for anchor in anchors["anchors"][key]:
                interval = window_indices(cumulative, int(anchor["witness_index"]), window_cfg["length_m"])
                reason = None
                if interval is None:
                    reason = "insufficient_path_extent"
                else:
                    start, stop = interval
                    try:
                        controls = control_indices(cumulative, start, stop, window_cfg["num_control_points"])
                    except ValueError:
                        reason = "insufficient_control_samples"
                    if reason is None and absolute_joint_limit_margin(robot, q[start : stop + 1]) <= 0.0:
                        reason = "nonpositive_joint_margin"
                    if reason is None:
                        sigmas = [
                            evaluate_task_kinematics_5d(robot, q[index], characteristic_length=contract["characteristic_length_m"]).sigma_min_5
                            for index in np.unique(np.linspace(start, stop, min(21, stop - start + 1), dtype=int))
                        ]
                        if min(sigmas) < contract["sigma_safe"]:
                            reason = "sigma_below_frozen_threshold"
                if reason is not None:
                    rejects.append({"anchor_id": anchor["anchor_id"], "witness_index": anchor["witness_index"], "reason": reason})
                    continue
                local_controls = controls - start
                baseline_points = points_surface[start : stop + 1]
                eligible.append((int(anchor["witness_index"]), {
                    "anchor_id": anchor["anchor_id"],
                    "anchor_witness_index": int(anchor["witness_index"]),
                    "start_index": int(start),
                    "stop_index": int(stop),
                    "control_indices_global": controls.tolist(),
                    "control_indices_local": local_controls.tolist(),
                    "baseline_surface_length": float(cumulative[stop] - cumulative[start]),
                    "baseline_maximum_turn_rad": maximum_turn_angle(baseline_points),
                    "minimum_sampled_sigma_min_5": float(min(sigmas)),
                    "minimum_absolute_joint_margin_rad": absolute_joint_limit_margin(robot, q[start : stop + 1]),
                }))
            selected_indices = select_uniform_indices(
                cumulative,
                [item[0] for item in eligible],
                int(window_cfg["count_per_scene"]),
            )
            by_anchor = {index: record for index, record in eligible}
            selected = []
            for ordinal, anchor_index in enumerate(selected_indices):
                record = dict(by_anchor[anchor_index])
                record.update({
                    "window_id": f"{surface_id}_{level}_W{ordinal:02d}",
                    "surface_id": surface_id,
                    "anisotropy_level": level,
                    "placement_id": scene["candidate_id"],
                    "witness_file": scene["witness_file"],
                    "transform_base_from_surface": scene["transform_base_from_surface"],
                    "seed": deterministic_window_seed(config["seed"], f"{surface_id}/{level}/{ordinal}"),
                })
                selected.append(record)
            if len(selected) != window_cfg["count_per_scene"]:
                raise RuntimeError(f"{key} has only {len(selected)} eligible frozen windows")
            windows_by_scene[key] = selected
            rejected[key] = rejects + [
                {"anchor_id": record["anchor_id"], "witness_index": index, "reason": "uniform_subselection"}
                for index, record in eligible if index not in selected_indices
            ]
    payload = {
        "experiment": config["experiment"],
        "frozen_before_deformation_results": True,
        "source_anchor_hash": anchors["content_hash"],
        "selection_rule": "eligible E06-R anchors followed by uniform archived baseline arclength selection",
        "windows_by_scene": windows_by_scene,
        "rejected_anchors": rejected,
    }
    payload["content_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def run_all(config, scenes, frozen, archived, output: Path, jobs: int, requested: set[str]):
    windows = [window for group in frozen["windows_by_scene"].values() for window in group]
    if requested:
        windows = [window for window in windows if window["window_id"] in requested]
    arguments = [(config, scenes, archived, window) for window in windows]
    results = []
    if jobs <= 1:
        for argument in arguments:
            results.append(run_window(argument))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(run_window, argument): argument[-1]["window_id"] for argument in arguments}
            for future in as_completed(futures):
                results.append(future.result())
                print(f"completed {futures[future]}", flush=True)
    results.sort(key=lambda item: item["final"][0]["window_id"])
    histories = [row for item in results for row in item["history"]]
    finals = [row for item in results for row in item["final"]]
    baselines = [item["baseline"] for item in results]
    write_csv(output / "window_baselines.csv", baselines)
    write_csv(output / "final_window_results.csv", finals)
    with gzip.open(output / "candidate_history.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in histories for key in row}))
        writer.writeheader(); writer.writerows(histories)


def run_window(arguments):
    config, scenes, archived, window = arguments
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
    baseline = evaluate_baseline(robot, surface, transform, q0, positions0, axes0, points0, config, archived, window)
    history = []
    finals = []
    for method in ("M1", "M2"):
        final, rows = optimize_method(robot, surface, transform, q0, points0, normals0, controls, baseline, config, archived, window, method)
        finals.append(final); history.extend(rows)
    return {"baseline": baseline, "final": finals, "history": history}


def evaluate_baseline(robot, surface, transform, q, positions, axes, points_surface, config, archived, window):
    strict = strict_check(robot, surface, transform, q, positions, axes, config, archived)
    if not strict.overall_pass:
        raise RuntimeError(f"frozen baseline {window['window_id']} failed strict replay: {strict.failure_reasons}")
    direct_lq = compute_joint_execution_cost((q,)).weighted_joint_length
    if abs(direct_lq - strict.execution_cost.weighted_joint_length) > 1e-9:
        raise RuntimeError("baseline witness reconstruction changed L_q")
    metric_length = integrated_robot_metric_length(
        robot, surface, transform, q, points_surface,
        characteristic_length=config["robot_contract"]["characteristic_length_m"],
        sigma_safe=config["robot_contract"]["sigma_safe"],
        finite_difference_step=config["metric"]["finite_difference_step_m"],
    )
    metric_samples = []
    for index in np.unique(np.linspace(0, len(q) - 1, min(21, len(q)), dtype=int)):
        task = evaluate_task_kinematics_5d(robot, q[index], characteristic_length=config["robot_contract"]["characteristic_length_m"])
        from diffusion_coverage.geometry.robot_surface_metric import compute_robot_surface_metric, estimate_surface_contact_differential
        contact = estimate_surface_contact_differential(surface, points_surface[index], transform, task.axis_basis, characteristic_length=config["robot_contract"]["characteristic_length_m"], finite_difference_step=config["metric"]["finite_difference_step_m"])
        metric_samples.append(compute_robot_surface_metric(task, contact.task_differential, minimum_singular_value=config["robot_contract"]["sigma_safe"]).log_kappa_R)
    coverage = strict.coverage_metrics
    return {
        "window_id": window["window_id"], "surface_id": window["surface_id"], "anisotropy_level": window["anisotropy_level"],
        "placement_id": window["placement_id"], "L_q0": direct_lq,
        "L_surface0": float(cumulative_length(points_surface)[-1]), "L_G0": metric_length,
        "E_NUC0": coverage.nuc_error, "E_miss0": coverage.missed_error, "E_rep0": coverage.repeat_error,
        "min_sigma0": strict.min_sigma_min_5, "joint_margin0": strict.min_joint_limit_margin,
        "absolute_joint_margin0": absolute_joint_limit_margin(robot, q), "q_sample_count0": len(q),
        "A_window": float(np.median(metric_samples)), "baseline_reconstruction_error": abs(direct_lq - strict.execution_cost.weighted_joint_length),
    }


def optimize_method(robot, surface, transform, q0, points0, normals0, controls, baseline, config, archived, window, method):
    optimizer = config["optimizer"]; window_cfg = config["window"]
    parameters = np.zeros((3, 2), dtype=np.float64)
    incumbent_objective = baseline["L_surface0"] if method == "M1" else baseline["L_G0"]
    incumbent = baseline_candidate(method, baseline, parameters)
    best_actual = baseline["L_q0"]; accepted = 0; rows = []
    schedule = proposal_schedule(optimizer["candidate_evaluations"], optimizer["proposal_radii_m"], optimizer["evaluations_per_radius"], window["seed"])
    for evaluation, proposal in enumerate(schedule, start=1):
        candidate_parameters = propose_parameters(parameters, proposal, window_cfg["maximum_control_displacement_m"])
        started = perf_counter()
        row = evaluate_candidate(robot, surface, transform, q0, points0, controls, candidate_parameters, baseline, config, archived, window, method)
        objective = row.get("objective")
        next_parameters, next_objective, can_accept = accept_incumbent_update(
            parameters, incumbent_objective, candidate_parameters, objective,
            admitted=row["admitted"], accepted_count=accepted,
            maximum_accepted=optimizer["maximum_accepted_iterations"],
            improvement_tolerance=optimizer["improvement_tolerance"],
        )
        if can_accept:
            parameters = next_parameters
            incumbent_objective = next_objective
            incumbent = dict(row)
            accepted += 1
        if row.get("L_q") is not None:
            best_actual = min(best_actual, float(row["L_q"]))
        row.update({
            "evaluation": evaluation, "accepted": can_accept, "accepted_count": accepted,
            "best_feasible_actual_L_q": best_actual, "elapsed_s": perf_counter() - started,
            "proposal_axis": proposal[0], "proposal_sign": proposal[1], "proposal_radius_m": proposal[2],
        })
        rows.append(row)
    if method == "M1" and incumbent.get("q_path") is not None:
        incumbent["L_G_posthoc"] = integrated_robot_metric_length(
            robot, surface, transform, incumbent.pop("q_path"), incumbent.pop("points_surface"),
            characteristic_length=config["robot_contract"]["characteristic_length_m"], sigma_safe=config["robot_contract"]["sigma_safe"],
            finite_difference_step=config["metric"]["finite_difference_step_m"],
        )
    else:
        incumbent.pop("q_path", None); incumbent.pop("points_surface", None)
    final = {key: value for key, value in incumbent.items() if not isinstance(value, np.ndarray)}
    final.update({"accepted_iterations": accepted, "candidate_evaluations": len(schedule), "best_feasible_actual_L_q": best_actual})
    return final, [{key: value for key, value in row.items() if not isinstance(value, np.ndarray)} for row in rows]


def baseline_candidate(method, baseline, parameters):
    return {
        "window_id": baseline["window_id"], "surface_id": baseline["surface_id"], "anisotropy_level": baseline["anisotropy_level"],
        "placement_id": baseline["placement_id"], "method": method, "parameters": parameters.tolist(), "admitted": True,
        "objective": baseline["L_surface0"] if method == "M1" else baseline["L_G0"], "L_q": baseline["L_q0"],
        "L_surface": baseline["L_surface0"], "L_G": baseline["L_G0"], "L_E": baseline["L_surface0"],
        "E_NUC": baseline["E_NUC0"], "E_miss": baseline["E_miss0"], "E_rep": baseline["E_rep0"],
        "min_sigma_min_5": baseline["min_sigma0"], "min_joint_limit_margin": baseline["joint_margin0"],
        "terminal_q_mismatch": 0.0, "relative_surface_length_change": 0.0, "admission_reasons": "",
        "Delta": 0.0, "C_q": baseline["L_q0"] / baseline["L_surface0"], "q_path": None, "points_surface": None,
    }


def evaluate_candidate(robot, surface, transform, q0, points0, controls, parameters, baseline, config, archived, window, method):
    common = {"window_id": window["window_id"], "surface_id": window["surface_id"], "anisotropy_level": window["anisotropy_level"], "placement_id": window["placement_id"], "method": method, "parameters": parameters.tolist()}
    try:
        curve = deform_surface_path(surface, points0, controls, parameters, maximum_displacement=config["window"]["maximum_control_displacement_m"], maximum_retraction_step=config["window"]["retraction_maximum_step_m"])
        turn_limit = max(np.deg2rad(config["window"]["maximum_turn_degrees"]), window["baseline_maximum_turn_rad"] + np.deg2rad(config["window"]["turn_increase_allowance_degrees"]))
        topology = curve.topology_preserved and curve.maximum_turn <= turn_limit
        positions = curve.points @ transform[:3, :3].T + transform[:3, 3]
        axes = -(curve.normals @ transform[:3, :3].T); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        continuation = continue_probe_from_shared_q(robot, q0[0], positions, axes, maximum_joint_step=config["robot_contract"]["maximum_continuation_joint_step_rad"], position_tolerance=config["robot_contract"]["construction_position_tolerance_m"], axis_tolerance=np.deg2rad(config["robot_contract"]["axis_tolerance_degrees"]))
        if not continuation.feasible or len(continuation.q_path) != len(positions):
            return {**common, "admitted": False, "admission_reasons": continuation.failure_reason or "continuation", "L_surface": curve.surface_length, "objective": None, "L_q": None}
        strict = strict_check(robot, surface, transform, continuation.q_path, positions, axes, config, archived)
        coverage = strict.coverage_metrics
        mismatch = terminal_q_mismatch(continuation.q_path[-1], q0[-1])
        limits = AdmissionLimits(config["admission"]["maximum_relative_surface_length_change"], config["admission"]["delta_NUC"], config["robot_contract"]["maximum_terminal_q_mismatch_rad"], config["robot_contract"]["sigma_safe"])
        admitted, reasons = hard_admission(
            surface_length=curve.surface_length, baseline_surface_length=baseline["L_surface0"],
            nuc_error=np.inf if coverage is None else coverage.nuc_error, missed_error=np.inf if coverage is None else coverage.missed_error,
            repeat_error=np.inf if coverage is None else coverage.repeat_error, baseline_nuc_error=baseline["E_NUC0"],
            baseline_missed_error=baseline["E_miss0"], baseline_repeat_error=baseline["E_rep0"], terminal_mismatch=mismatch,
            strict_pass=strict.overall_pass, topology_preserved=topology, limits=limits,
        )
        lq = None if strict.execution_cost is None else strict.execution_cost.weighted_joint_length
        lg = None
        objective = None
        if admitted:
            if method == "M1":
                objective = euclidean_objective(curve.surface_length)
            else:
                lg = integrated_robot_metric_length(robot, surface, transform, continuation.q_path, curve.points, characteristic_length=config["robot_contract"]["characteristic_length_m"], sigma_safe=config["robot_contract"]["sigma_safe"], finite_difference_step=config["metric"]["finite_difference_step_m"])
                objective = lg
        return {
            **common, "admitted": admitted, "admission_reasons": ";".join(reasons), "objective": objective,
            "L_q": lq, "L_surface": curve.surface_length, "L_E": curve.surface_length, "L_G": lg,
            "E_NUC": None if coverage is None else coverage.nuc_error, "E_miss": None if coverage is None else coverage.missed_error,
            "E_rep": None if coverage is None else coverage.repeat_error, "min_sigma_min_5": strict.min_sigma_min_5,
            "min_joint_limit_margin": strict.min_joint_limit_margin, "terminal_q_mismatch": mismatch,
            "relative_surface_length_change": abs(curve.surface_length - baseline["L_surface0"]) / baseline["L_surface0"],
            "Delta": None if lq is None else (baseline["L_q0"] - lq) / baseline["L_q0"],
            "C_q": None if lq is None else lq / curve.surface_length, "q_path": continuation.q_path, "points_surface": curve.points,
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as error:
        return {**common, "admitted": False, "admission_reasons": f"numerical:{type(error).__name__}", "objective": None, "L_q": None, "L_surface": None}


def strict_check(robot, surface, transform, q, positions, axes, config, archived):
    contract = config["robot_contract"]
    return check_strict_coverage_execution(
        robot, (q,), (positions,), (axes,), surface, transform,
        footprint_radius=archived["config"]["coverage"]["footprint_radius_m"], position_tolerance=contract["position_tolerance_m"],
        axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), characteristic_length=contract["characteristic_length_m"], sigma_safe=contract["sigma_safe"],
        missed_tolerance=1.0, repeat_tolerance=10.0, interpolation_joint_step=contract["maximum_dense_q_step_rad"],
        coverage_path_sample_spacing=archived["frozen_contract"]["coverage_path_sample_spacing_m"],
    )


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
