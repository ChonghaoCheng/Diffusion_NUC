#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from diffusion_coverage.diagnostics.anisotropy import absolute_joint_limit_margin
from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract
from diffusion_coverage.diagnostics.symmetry_layout import (
    array_hash,
    make_symmetry_reference_surface,
    scene_seed,
    transition_cost_decomposition,
)
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics, transform_surface_pose_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run E06-G2 symmetry-orbit full-path continuation")
    parser.add_argument("--stage", choices=("default", "strong"), default="default")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/symmetry_preserving_global_layout_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/symmetry_preserving_global_layout_v1")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args(); args.output = args.output.resolve(); config = json.loads(args.config.read_text())
    frozen = json.loads((args.output / "frozen_symmetry_orbits.json").read_text())
    if not frozen.get("frozen_before_robot_execution"):
        raise RuntimeError("symmetry orbits must be frozen before robot execution")
    archived, _, _ = load_e06_contract(ROOT)
    placements = json.loads((ROOT / config["placements_file"]).read_text())["selected"]
    geometry = read_csv(args.output / "geometry_invariance.csv")
    geometry_lookup = {(row["surface_id"], row["symmetry_id"]): row for row in geometry}
    if args.stage == "default":
        jobs = [
            make_job(config, frozen, placements, geometry_lookup, surface_id, level, orbit, False, args.output)
            for surface_id in ("saddle", "hemisphere")
            for level in config["placement_levels"]
            for orbit in frozen["orbits"] if orbit["surface_id"] == surface_id
        ]
        output_file = args.output / "execution_results.csv"
        partial = args.output / "execution_results.partial.jsonl"
    else:
        defaults = read_csv(args.output / "execution_results.csv")
        mixed = {
            (surface, level)
            for surface in ("saddle", "hemisphere") for level in config["placement_levels"]
            if _is_mixed([row for row in defaults if row["surface_id"] == surface and row["placement_level"] == level])
        }
        failed = [row for row in defaults if (row["surface_id"], row["placement_level"]) in mixed and not as_bool(row["overall_pass"])]
        orbit_lookup = {(row["surface_id"], row["symmetry_id"]): row for row in frozen["orbits"]}
        jobs = [
            make_job(config, frozen, placements, geometry_lookup, row["surface_id"], row["placement_level"],
                     orbit_lookup[(row["surface_id"], row["symmetry_id"])], True, args.output,
                     default_failure_reason=row["failure_reason"])
            for row in failed
        ]
        output_file = args.output / "liftability_sensitivity.csv"
        partial = args.output / "liftability_sensitivity.partial.jsonl"
        if not jobs:
            print("no default mixed-success scenes; strong search not required")
            return
    if args.limit is not None: jobs = jobs[:args.limit]
    existing = {row["job_id"]: row for row in load_jsonl(partial)}
    pending = [job for job in jobs if job["job_id"] not in existing]
    arguments = [(config, archived, job) for job in pending]
    if args.jobs == 1:
        generated = [run_job(argument) for argument in arguments]
    else:
        generated = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_job, argument): argument[2]["job_id"] for argument in arguments}
            for future in as_completed(futures):
                row = future.result(); generated.append(row); existing[row["job_id"]] = row
                with partial.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(row["job_id"], "pass", row["overall_pass"], flush=True)
    if args.jobs == 1:
        with partial.open("a") as handle:
            for row in generated: existing[row["job_id"]] = row; handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = sorted(existing.values(), key=lambda row: row["job_id"])
    if args.limit is not None or len(rows) != len(jobs):
        print(f"partial {args.stage}: {len(rows)}/{len(jobs)}")
        return
    write_csv(output_file, rows)
    partial.unlink(missing_ok=True)
    print(json.dumps({"stage": args.stage, "jobs": len(rows), "verified": sum(row["overall_pass"] for row in rows)}, indent=2))


def make_job(config, frozen, placements, geometry, surface_id, level, orbit, strong, output, default_failure_reason=None):
    return {
        "job_id": f"{surface_id}/{level}/{orbit['symmetry_id']}/{'strong' if strong else 'default'}",
        "surface_id": surface_id, "placement_level": level,
        "placement": placements[surface_id][level], "orbit": orbit,
        "geometry": geometry[(surface_id, orbit["symmetry_id"])],
        "array_file": str((output / frozen["array_file"]).resolve()),
        "output": str(output), "strong": strong, "default_failure_reason": default_failure_reason,
    }


def run_job(arguments):
    config, archived, job = arguments
    start_time = perf_counter(); strong = bool(job["strong"]); budget = config["strong_search" if strong else "default_search"]
    robot_cfg = archived["config"]["robot"]; contract = config["robot_contract"]
    robot = UR5eKinematics(robot_cfg["model"], site_name=robot_cfg["site_name"], tool_axis_index=robot_cfg["tool_axis_index"], tool_axis_sign=robot_cfg["tool_axis_sign"])
    arrays = np.load(job["array_file"]); orbit = job["orbit"]
    points = np.asarray(arrays[orbit["array_keys"]["points"]], dtype=np.float64)
    normals = np.asarray(arrays[orbit["array_keys"]["normals"]], dtype=np.float64)
    source_q = np.asarray(arrays[f"{job['surface_id']}_source_q_start"], dtype=np.float64)
    transform = np.asarray(job["placement"]["transform_base_from_surface"], dtype=np.float64)
    positions, axes = transform_surface_pose_path(points, normals, transform)
    seed = scene_seed(config, job["surface_id"], job["placement_level"], strong=strong)
    candidates = robot.enumerate_ik(
        positions[0], axes[0], seed_configurations=[source_q], random_restarts=int(budget["random_restarts"]),
        rng=np.random.default_rng(seed), axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]),
        minimum_manipulability=0.0, max_candidates=int(budget["max_candidates"]),
        orientation_cone_samples=int(budget["orientation_cone_samples"]),
    )
    continuation_start = perf_counter(); continued = []
    for candidate in candidates:
        result = robot.continue_task_transition(
            candidate.q, positions, axes,
            maximum_joint_step=contract["maximum_continuation_joint_step_rad"], minimum_manipulability=0.0,
            position_tolerance=contract["construction_position_tolerance_m"],
            axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]),
        )
        if result.feasible and len(result.q_path) == len(positions):
            q = np.asarray(result.q_path, dtype=np.float64)
            continued.append((compute_joint_execution_cost((q,)).weighted_joint_length, q))
    continuation_time = perf_counter() - continuation_start
    surface = make_symmetry_reference_surface(job["surface_id"], config)
    selected = None; strict = None
    for _, q in sorted(continued, key=lambda item: item[0]):
        checked = check_strict_coverage_execution(
            robot, (q,), (positions,), (axes,), surface, transform,
            footprint_radius=config["coverage"]["footprint_radius_m"], position_tolerance=contract["position_tolerance_m"],
            axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), characteristic_length=contract["characteristic_length_m"],
            sigma_safe=contract["sigma_safe"], missed_tolerance=1.0, repeat_tolerance=10.0,
            interpolation_joint_step=contract["maximum_dense_q_step_rad"],
            coverage_path_sample_spacing=config["coverage"]["path_sample_spacing_m"],
        )
        if checked.overall_pass:
            selected, strict = q, checked
            break
    witness_path = None
    if selected is not None:
        directory = Path(job["output"]) / ("strong_witnesses" if strong else "witnesses"); directory.mkdir(exist_ok=True)
        witness_path = directory / f"{job['surface_id']}_{job['placement_level']}_{orbit['symmetry_id']}.npz"
        np.savez_compressed(witness_path, q=selected, desired_positions=positions, desired_axes=axes, surface_points=points, surface_normals=normals)
    geometry = job["geometry"]
    base = {
        "job_id": job["job_id"], "surface_id": job["surface_id"], "placement_level": job["placement_level"],
        "placement_id": job["placement"]["candidate_id"], "symmetry_id": orbit["symmetry_id"],
        "symmetry_name": orbit["symmetry_name"], "angle_degrees": orbit["angle_degrees"],
        "search_budget": "strong" if strong else "default", "search_seed": seed,
        "random_restarts": budget["random_restarts"], "max_candidates": budget["max_candidates"],
        "orientation_cone_samples": budget["orientation_cone_samples"], "neighbor_warm_start": False,
        "start_seed_hash": array_hash(source_q), "candidate_count": len(candidates), "continued_candidate_count": len(continued),
        "E_miss": float(geometry["E_miss"]), "E_rep": float(geometry["E_rep"]), "E_NUC": float(geometry["E_NUC"]),
        "L_S": float(geometry["L_S"]), "lift_found": selected is not None,
        "strict_kinematics_pass": False if strict is None else strict.kinematics_pass,
        "overall_pass": selected is not None and strict.overall_pass,
        "failure_reason": None if selected is not None else "continuous_lift_not_found_under_numerical_budget",
        "J_q": None if strict is None else strict.execution_cost.weighted_joint_length,
        "C_q": None if strict is None else strict.execution_cost.weighted_joint_length / float(geometry["L_S"]),
        "min_sigma_min_5": None if strict is None else strict.min_sigma_min_5,
        "minimum_absolute_joint_margin_rad": None if selected is None else absolute_joint_limit_margin(robot, selected),
        "max_position_error": None if strict is None else strict.max_position_error,
        "max_axis_error": None if strict is None else strict.max_axis_error,
        "collision_pass": False if strict is None else "robot_collision" not in strict.failure_reasons,
        "continuation_runtime": continuation_time, "total_runtime": perf_counter()-start_time,
        "witness_file": None if witness_path is None else str(witness_path.relative_to(ROOT)),
        "default_failure_reason": job.get("default_failure_reason"),
    }
    if selected is not None: base.update(transition_cost_decomposition(selected, points))
    return base


def _is_mixed(rows):
    values = [as_bool(row["overall_pass"]) for row in rows]
    return any(values) and not all(values)
def as_bool(value): return value is True or str(value).lower() == "true"
def load_jsonl(path): return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines() if line]
def read_csv(path):
    with path.open(newline="") as handle: return list(csv.DictReader(handle))
def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
