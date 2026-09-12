#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import numpy as np

from diffusion_coverage.diagnostics.anisotropy import absolute_joint_limit_margin
from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract
from diffusion_coverage.diagnostics.global_layout import (
    build_ordered_path_ik_catalog,
    code_points_from_layout,
    lift_ordered_layout,
    make_reference_surface,
)
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


def parse_args():
    parser = argparse.ArgumentParser(description="Run E06-G full global-layout continuation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/global_layout_capacity_gate_v1.json")
    parser.add_argument("--library", type=Path, default=ROOT / "results/global_layout_capacity_gate_v1/layout_library.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/global_layout_capacity_gate_v1")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--limit-jobs", type=int)
    return parser.parse_args()


def main():
    args = parse_args(); config = json.loads(args.config.read_text()); library = json.loads(args.library.read_text())
    if not library.get("frozen_before_G0") or len(library.get("layouts", [])) != 64:
        raise RuntimeError("complete 64-layout library must be frozen before G0")
    archived, _, _ = load_e06_contract(ROOT)
    placements = json.loads((ROOT / config["placements_file"]).read_text())["selected"]
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "witnesses").mkdir(exist_ok=True)
    groups = make_groups(library["layouts"], placements, config)
    if args.limit_jobs is not None: groups = groups[:args.limit_jobs]
    partial_path = args.output / "execution_results.partial.jsonl"
    existing = load_jsonl(partial_path)
    completed = {(row["surface_id"], row["remesh_id"], row["anisotropy_level"]) for row in existing}
    pending = [group for group in groups if group["job_key"] not in completed]
    arguments = [(config, archived, group, str(args.output)) for group in pending]
    if args.jobs == 1:
        generated = [run_group(argument) for argument in arguments]
    else:
        generated = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_group, argument): argument[2]["job_key"] for argument in arguments}
            for future in as_completed(futures):
                rows = future.result(); generated.append(rows)
                with partial_path.open("a") as handle:
                    for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
                print("completed", "/".join(futures[future]), flush=True)
    if args.jobs == 1:
        with partial_path.open("a") as handle:
            for rows in generated:
                for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = sorted(load_jsonl(partial_path), key=lambda row: (row["surface_id"], row["anisotropy_level"], row["layout_id"]))
    expected = 192 if args.limit_jobs is None else 8 * len(groups)
    if len(rows) != expected:
        print(f"partial G0: {len(rows)}/{expected}")
        return
    write_csv(args.output / "execution_results.csv", rows)
    (args.output / "execution_results.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    partial_path.unlink(missing_ok=True)
    print(json.dumps({"executions": len(rows), "verified": sum(row["overall_pass"] for row in rows), "failures": sum(not row["overall_pass"] for row in rows)}, indent=2))


def make_groups(layouts, placements, config):
    groups = []
    for surface_id in ("saddle", "hemisphere"):
        for remesh_id in config["remesh_ids"]:
            selected = sorted((row for row in layouts if row["surface_id"] == surface_id and row["remesh_id"] == remesh_id), key=lambda row: row["root_id"])
            if len(selected) != config["root_count"]: raise RuntimeError("incomplete root/remesh factorial library")
            for level in config["placement_levels"]:
                groups.append({"job_key": (surface_id, remesh_id, level), "surface_id": surface_id, "remesh_id": remesh_id, "anisotropy_level": level, "placement": placements[surface_id][level], "layouts": selected})
    return groups


def run_group(arguments):
    config, archived, group, output_string = arguments
    robot_cfg = archived["config"]["robot"]
    robot = UR5eKinematics(robot_cfg["model"], site_name=robot_cfg["site_name"], tool_axis_index=robot_cfg["tool_axis_index"], tool_axis_sign=robot_cfg["tool_axis_sign"])
    reference = make_reference_surface(group["surface_id"], config)
    transform = np.asarray(group["placement"]["transform_base_from_surface"], dtype=np.float64)
    code_points = code_points_from_layout(group["layouts"][0])
    for layout in group["layouts"][1:]:
        if not np.allclose(code_points_from_layout(layout), code_points, atol=1e-12, rtol=0):
            raise RuntimeError("root variants do not share a canonical code-point catalog")
    search = config["default_search"]; contract = config["robot_contract"]
    surface_number = 0 if group["surface_id"] == "saddle" else 1
    remesh_number = int(group["remesh_id"][1:]); placement_number = config["placement_levels"].index(group["anisotropy_level"])
    seed = int(config["seed"]) + 100000 * surface_number + 1000 * remesh_number + 100 * placement_number
    catalog = build_ordered_path_ik_catalog(
        robot, reference, code_points, transform,
        axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), characteristic_length=contract["characteristic_length_m"], sigma_safe=contract["sigma_safe"],
        random_restarts=search["random_restarts"], max_candidates=search["max_candidates"], orientation_cone_samples=search["orientation_cone_samples"], seed=seed,
    )
    cache = {}; rows = []
    for layout in group["layouts"]:
        start = perf_counter()
        lift = lift_ordered_layout(
            robot, reference, layout, catalog, transform, cache,
            axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), characteristic_length=contract["characteristic_length_m"], sigma_safe=contract["sigma_safe"],
            task_edge_samples=search["task_edge_samples"], surface_path_spacing=config["coverage"]["path_sample_spacing_m"],
            maximum_joint_step=contract["maximum_continuation_joint_step_rad"], position_tolerance=contract["construction_position_tolerance_m"], max_active_branches=search["max_active_branches"],
        )
        strict = None
        if lift.found:
            strict = check_strict_coverage_execution(
                robot, (lift.q_path,), (lift.desired_positions,), (lift.desired_axes,), reference, transform,
                footprint_radius=config["coverage"]["footprint_radius_m"], position_tolerance=contract["position_tolerance_m"],
                axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]), characteristic_length=contract["characteristic_length_m"], sigma_safe=contract["sigma_safe"],
                missed_tolerance=1.0, repeat_tolerance=10.0, interpolation_joint_step=contract["maximum_dense_q_step_rad"],
                coverage_path_sample_spacing=config["coverage"]["path_sample_spacing_m"],
            )
        witness_path = None
        if lift.found:
            witness_path = Path(output_string) / "witnesses" / f"{group['surface_id']}_{group['anisotropy_level']}_{layout['layout_id']}.npz"
            np.savez_compressed(witness_path, q=np.asarray(lift.q_path, dtype=np.float64), desired_positions=lift.desired_positions, desired_axes=lift.desired_axes)
        failure = lift.failure_reason if not lift.found else strict.failure_reason
        execution = None if strict is None else strict.execution_cost
        row = {
            "surface_id": group["surface_id"], "anisotropy_level": group["anisotropy_level"], "placement_id": group["placement"]["candidate_id"],
            "remesh_id": group["remesh_id"], "root_id": layout["root_id"], "layout_id": layout["layout_id"],
            "geometry_baseline": layout["geometry_baseline"], "canonical_layout": layout["canonical_layout"],
            "E_miss": layout["E_miss"], "E_rep": layout["E_rep"], "E_NUC": layout["E_NUC"], "L_S": layout["L_S"],
            "lift_found": lift.found, "strict_kinematics_pass": False if strict is None else strict.kinematics_pass,
            "strict_coverage_pass": False if strict is None else strict.coverage_pass, "overall_pass": False if strict is None else strict.overall_pass,
            "failure_reason": failure, "J_q": None if execution is None else execution.weighted_joint_length,
            "C_q": None if execution is None else execution.weighted_joint_length / layout["L_S"],
            "min_sigma_min_5": None if strict is None else strict.min_sigma_min_5,
            "minimum_absolute_joint_margin_rad": None if not lift.found else absolute_joint_limit_margin(robot, lift.q_path),
            "max_position_error": None if strict is None else strict.max_position_error, "max_axis_error": None if strict is None else strict.max_axis_error,
            "collision_pass": False if strict is None else "robot_collision" not in strict.failure_reasons,
            "q_sample_count": 0 if execution is None else execution.q_sample_count,
            "evaluated_transitions": lift.evaluated_transitions, "IK_enumeration_time_shared": catalog.enumeration_time,
            "continuation_time": lift.continuation_time, "total_execution_evaluation_time": perf_counter() - start,
            "witness_file": None if witness_path is None else str(witness_path.relative_to(ROOT)), "search_budget": "default",
        }
        rows.append(row)
    return rows


def load_jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
