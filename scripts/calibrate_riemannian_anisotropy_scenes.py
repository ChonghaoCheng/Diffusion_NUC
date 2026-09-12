#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.diagnostics.anisotropy import absolute_joint_limit_margin, metric_along_witness
from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract, make_e06_surface
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


def parse_args():
    parser = argparse.ArgumentParser(description="E06-R R0 fully-liftable anisotropy scene calibration")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/riemannian_anisotropy_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/riemannian_anisotropy_utility_v1/r0_scene_calibration")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args(); config = json.loads(args.config.read_text()); archived, _, _ = load_e06_contract(ROOT)
    assert_contract(config, archived)
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "witnesses").mkdir(exist_ok=True)
    placements = make_candidate_placements(config, archived)
    jobs = [(surface, row) for surface in ("saddle", "hemisphere") for row in placements[surface]]
    if args.limit is not None: jobs = jobs[:args.limit]
    existing = {row["job_id"]: row for row in load_jsonl(args.output / "candidate_placements.partial.jsonl")}
    jobs = [job for job in jobs if f"{job[0]}/{job[1]['candidate_id']}" not in existing]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_candidate, config, archived, surface, row, str(args.output)): (surface, row) for surface, row in jobs}
        for future in as_completed(futures):
            row = future.result(); existing[row["job_id"]] = row
            with (args.output / "candidate_placements.partial.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(row["job_id"], "admitted", row["admitted"], "A", row.get("median_log_kappa_R"), flush=True)
    rows = sorted(existing.values(), key=lambda row: row["job_id"])
    expected = 2 * int(config["candidate_placements_per_surface"])
    if args.limit is not None or len(rows) != expected:
        write_tables(args.output, rows); print(f"partial R0: {len(rows)}/{expected}"); return
    selected = select_scenes(rows)
    frozen = {
        "experiment": config["experiment"], "frozen_before_R1": True,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "selection_rule": config["r0"]["selection_rule"], "all_candidate_placements": rows,
        "selected": selected,
    }
    scenes_path = ROOT / "configs/riemannian_anisotropy_scenes_v1.json"
    scenes_path.write_text(json.dumps(frozen, indent=2) + "\n")
    (args.output.parent / "frozen_scenes.json").write_text(json.dumps(frozen, indent=2) + "\n")
    write_tables(args.output, rows); plot_calibration(args.output, rows, selected)
    print(json.dumps(selected, indent=2))


def assert_contract(config, archived):
    frozen = archived["frozen_contract"]; robot = archived["config"]["robot"]; contract = config["robot_contract"]
    checks = [
        (contract["characteristic_length_m"], robot["characteristic_length_m"]),
        (contract["sigma_safe"], frozen["sigma_safe"]),
        (contract["maximum_dense_q_step_rad"], frozen["q_interpolation_step_rad"]),
        (contract["axis_tolerance_degrees"], robot["axis_tolerance_degrees"]),
        (contract["position_tolerance_m"], robot["position_tolerance_m"]),
    ]
    if not all(np.isclose(a, b) for a, b in checks): raise RuntimeError("E06-R contract differs from E06-D")


def make_candidate_placements(config, archived):
    rng = np.random.default_rng(int(config["seed"])); count = int(config["candidate_placements_per_surface"])
    perturb = config["placement_perturbation"]; result = {}
    for surface in ("saddle", "hemisphere"):
        base = np.asarray(archived["placements"]["surfaces"][surface]["selected"]["P_easy"]["transform_base_from_surface"], dtype=float)
        rows = [{"candidate_id": "T00", "transform_base_from_surface": base.tolist(), "translation_delta_m": [0,0,0], "rotation_delta_degrees": [0,0,0]}]
        for index in range(1, count):
            translation = rng.uniform(-float(perturb["translation_max_m"]), float(perturb["translation_max_m"]), 3)
            angles = np.deg2rad([rng.uniform(-perturb["roll_pitch_max_degrees"], perturb["roll_pitch_max_degrees"]), rng.uniform(-perturb["roll_pitch_max_degrees"], perturb["roll_pitch_max_degrees"]), rng.uniform(-perturb["yaw_max_degrees"], perturb["yaw_max_degrees"])])
            rotation = rotation_z(angles[2]) @ rotation_y(angles[1]) @ rotation_x(angles[0])
            transform = np.eye(4); transform[:3,:3] = rotation @ base[:3,:3]; transform[:3,3] = base[:3,3] + translation
            rows.append({"candidate_id": f"T{index:02d}", "transform_base_from_surface": transform.tolist(), "translation_delta_m": translation.tolist(), "rotation_delta_degrees": np.rad2deg(angles).tolist()})
        result[surface] = rows
    return result


def evaluate_candidate(config, archived, surface_id, placement, output_string):
    start_time = perf_counter(); robot_cfg = archived["config"]["robot"]; frozen = archived["frozen_contract"]; contract = config["robot_contract"]
    robot = UR5eKinematics(robot_cfg["model"], site_name=robot_cfg["site_name"], tool_axis_index=robot_cfg["tool_axis_index"], tool_axis_sign=robot_cfg["tool_axis_sign"])
    surface = make_e06_surface(archived["config"], surface_id, int(frozen["coverage_samples_per_face"]))
    old_transform = np.asarray(archived["placements"]["surfaces"][surface_id]["selected"]["P_easy"]["transform_base_from_surface"], dtype=float)
    transform = np.asarray(placement["transform_base_from_surface"], dtype=float)
    source = np.load(ROOT / f"results/nuc_robot_skeleton_coupling_v1/witnesses/{surface_id}_P_easy_S00.npz")
    local_positions = (source["desired_positions"] - old_transform[:3,3]) @ old_transform[:3,:3]
    local_axes = source["desired_axes"] @ old_transform[:3,:3]
    positions = local_positions @ transform[:3,:3].T + transform[:3,3]
    axes = local_axes @ transform[:3,:3].T; axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    q_path = None; continuation_time = 0.0
    if placement["candidate_id"] == "T00":
        q_path = source["q"].astype(np.float64, copy=True)
    else:
        candidates = robot.enumerate_ik(
            positions[0], axes[0], seed_configurations=[source["q"][0]], random_restarts=int(config["r0"]["random_restarts"]),
            rng=np.random.default_rng(int(config["seed"]) + 1009 * int(placement["candidate_id"][1:]) + (0 if surface_id == "saddle" else 100000)),
            axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), minimum_manipulability=0.0,
            max_candidates=int(config["r0"]["max_candidates"]), orientation_cone_samples=int(config["r0"]["orientation_cone_samples"]),
        )
        successful = []
        for candidate in candidates:
            tick = perf_counter(); result = robot.continue_task_transition(
                candidate.q, positions, axes, maximum_joint_step=contract["maximum_continuation_joint_step_rad"],
                minimum_manipulability=0.0, position_tolerance=contract["construction_position_tolerance_m"],
                axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]),
            ); continuation_time += perf_counter() - tick
            if result.feasible and len(result.q_path) == len(positions): successful.append(result.q_path)
        if successful:
            q_path = min(successful, key=lambda q: compute_joint_execution_cost((q,)).weighted_joint_length)
    base = {"job_id": f"{surface_id}/{placement['candidate_id']}", "surface_id": surface_id, **placement, "full_lift_found": q_path is not None, "continuation_runtime": continuation_time}
    if q_path is None: return {**base, "admitted": False, "failure_reason": "continuous_lift_not_found_under_budget", "total_runtime": perf_counter()-start_time}
    strict = check_strict_coverage_execution(
        robot, (q_path,), (positions,), (axes,), surface, transform,
        footprint_radius=float(archived["config"]["coverage"]["footprint_radius_m"]), position_tolerance=contract["position_tolerance_m"],
        axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), characteristic_length=contract["characteristic_length_m"], sigma_safe=contract["sigma_safe"],
        missed_tolerance=1.0, repeat_tolerance=10.0, interpolation_joint_step=contract["maximum_dense_q_step_rad"],
        coverage_path_sample_spacing=float(frozen["coverage_path_sample_spacing_m"]),
    )
    absolute_margin = absolute_joint_limit_margin(robot, q_path)
    admitted = bool(strict.overall_pass and strict.min_sigma_min_5 >= contract["sigma_safe"] and absolute_margin > contract["minimum_absolute_joint_margin_rad"])
    if not admitted: return {**base, "admitted": False, "failure_reason": strict.failure_reason or "joint_margin_interior_gate", "min_sigma_min_5": strict.min_sigma_min_5, "minimum_absolute_joint_margin_rad": absolute_margin, "total_runtime": perf_counter()-start_time}
    metric_rows = metric_along_witness(
        robot, surface, transform, q_path, positions, characteristic_length=contract["characteristic_length_m"], sigma_safe=contract["sigma_safe"],
        sample_count=int(config["metric"]["path_samples"]), finite_difference_step=float(config["metric"]["finite_difference_step_m"]),
    )
    logs = np.asarray([row["log_kappa_R"] for row in metric_rows]); ratios = np.asarray([row["R_G"] for row in metric_rows])
    witness_path = Path(output_string) / "witnesses" / f"{surface_id}_{placement['candidate_id']}.npz"
    np.savez_compressed(witness_path, q=np.asarray(q_path, dtype=np.float64), desired_positions=positions, desired_axes=axes)
    return {**base, "admitted": True, "failure_reason": None, "strict_kinematics_pass": strict.kinematics_pass,
        "min_sigma_min_5": strict.min_sigma_min_5, "min_joint_limit_margin_normalized": strict.min_joint_limit_margin,
        "minimum_absolute_joint_margin_rad": absolute_margin, "L_q": strict.execution_cost.weighted_joint_length,
        "median_log_kappa_R": float(np.median(logs)), "p75_log_kappa_R": float(np.percentile(logs,75)), "p90_log_kappa_R": float(np.percentile(logs,90)), "max_log_kappa_R": float(np.max(logs)),
        **{f"fraction_R_G_ge_{str(threshold).replace('.','p')}": float(np.mean(ratios >= threshold)) for threshold in (1.1,1.2,1.5,2.0)},
        "metric_samples": len(metric_rows), "witness_file": str(witness_path.relative_to(ROOT)), "total_runtime": perf_counter()-start_time}


def select_scenes(rows):
    selected = {}
    for surface in ("saddle", "hemisphere"):
        eligible = sorted((row for row in rows if row["surface_id"] == surface and row["admitted"]), key=lambda row: (row["median_log_kappa_R"], row["candidate_id"]))
        if len(eligible) < 3: raise RuntimeError(f"only {len(eligible)} admissible placements for {surface}")
        choices = {"P_low": eligible[0], "P_mid": eligible[(len(eligible)-1)//2], "P_high": eligible[-1]}
        selected[surface] = {name: {key: row[key] for key in row} for name, row in choices.items()}
    return selected


def write_tables(output, rows):
    (output / "candidate_placements.jsonl").write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows))
    fields = sorted({key for row in rows for key in row})
    with (output / "candidate_placements.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def plot_calibration(output, rows, selected):
    figure, axes = plt.subplots(1,2,figsize=(10,4)); source=[]
    for axis, surface in zip(axes,("saddle","hemisphere")):
        values=[row["median_log_kappa_R"] for row in rows if row["surface_id"]==surface and row["admitted"]]
        axis.hist(values,bins=min(12,len(values)),alpha=.75); axis.set(title=surface,xlabel="median log(kappa_R)",ylabel="admissible placements")
        for level,color in zip(("P_low","P_mid","P_high"),("#2ca02c","#ffbf00","#d62728")):
            value=selected[surface][level]["median_log_kappa_R"]; axis.axvline(value,color=color,label=level); source.append({"surface_id":surface,"level":level,"median_log_kappa_R":value})
        axis.legend()
    figure.tight_layout(); figure.savefig(output/"selected_placement_anisotropy.png",dpi=180); plt.close(figure)
    with (output/"selected_placement_anisotropy_source.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=source[0].keys()); writer.writeheader(); writer.writerows(source)


def load_jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rotation_x(a):
    c,s=np.cos(a),np.sin(a); return np.array(((1,0,0),(0,c,-s),(0,s,c)))
def rotation_y(a):
    c,s=np.cos(a),np.sin(a); return np.array(((c,0,s),(0,1,0),(-s,0,c)))
def rotation_z(a):
    c,s=np.cos(a),np.sin(a); return np.array(((c,-s,0),(s,c,0),(0,0,1)))


if __name__ == "__main__": main()
