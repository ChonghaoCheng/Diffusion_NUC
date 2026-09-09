#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.coverage import CoveragePlan, evaluate_nuc_coverage
from diffusion_coverage.nuc import (
    build_nuc_ik_catalog,
    generate_nuc_skeleton,
    minimum_cost_nuc_lift,
    validate_nuc_skeleton,
)
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.surface import make_hemisphere, make_saddle
from diffusion_coverage.surface.projection import project_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E06 NUC skeleton/robot coupling mechanism gate")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nuc_robot_coupling_v1.json")
    parser.add_argument("--placements", type=Path, default=ROOT / "configs/nuc_robot_coupling_v1_placements.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "results/nuc_robot_contract_calibration_v1/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nuc_robot_skeleton_coupling_v1")
    parser.add_argument("--max-candidates", type=int, default=None, help="Smoke-test override; not valid for the registered E06 result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    placements = json.loads(args.placements.read_text())
    calibration = json.loads(args.calibration.read_text())
    expected = int(config["e06"]["num_skeleton_candidates"])
    candidate_count = expected if args.max_candidates is None else min(args.max_candidates, expected)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "witnesses").mkdir(exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config_hash = hashlib.sha256(
        json.dumps({"config": config, "placements": placements, "calibration": calibration["frozen_contract"]}, sort_keys=True).encode()
    ).hexdigest()
    robot_cfg = config["robot"]
    e06 = config["e06"]
    frozen = calibration["frozen_contract"]
    robot = UR5eKinematics(
        robot_cfg["model"], site_name=robot_cfg["site_name"],
        tool_axis_index=robot_cfg["tool_axis_index"], tool_axis_sign=robot_cfg["tool_axis_sign"],
    )
    all_rows = []
    scene_rows = []
    experiment_start = perf_counter()
    for surface_number, surface_id in enumerate(config["surfaces"]):
        surface = make_surface(config, surface_id, int(frozen["coverage_samples_per_face"]))
        variants, generation_times = generate_variants(
            surface, candidate_count, int(config["seed"]) + 100000 * surface_number
        )
        refined = []
        refinement_times = []
        for skeleton in variants:
            start = perf_counter()
            value = projection_refine(surface, skeleton, int(e06["local_refinement_iterations"]))
            refinement_times.append(perf_counter() - start)
            validate_nuc_skeleton(surface.vertices, surface.faces, value)
            refined.append(value)
        geometry = []
        for skeleton in refined:
            start = perf_counter()
            metrics = evaluate_nuc_coverage(
                surface, CoveragePlan(skeleton.waypoints),
                footprint_radius=float(config["coverage"]["footprint_radius_m"]),
                path_sample_spacing=float(frozen["coverage_path_sample_spacing_m"]),
            )
            geometry.append((metrics, perf_counter() - start))
        geometry_index = min(
            range(candidate_count),
            key=lambda index: (
                geometry[index][0].nuc_error,
                geometry[index][0].missed_error,
                geometry[index][0].path_length,
            ),
        )
        nuc_limit = geometry[geometry_index][0].nuc_error + float(frozen["delta_NUC"])
        for placement_number, placement_id in enumerate(("P_easy", "P_mid", "P_hard")):
            selected_placement = placements["surfaces"][surface_id]["selected"][placement_id]
            transform = np.asarray(selected_placement["transform_base_from_surface"], dtype=np.float64)
            catalog = build_nuc_ik_catalog(
                robot, surface, refined[0], transform,
                axis_tolerance=np.deg2rad(float(robot_cfg["axis_tolerance_degrees"])),
                characteristic_length=float(robot_cfg["characteristic_length_m"]),
                sigma_safe=float(frozen["sigma_safe"]),
                random_restarts=int(e06["ik_random_restarts"]),
                max_candidates=int(e06["ik_max_candidates"]),
                orientation_cone_samples=5,
                seed=int(config["seed"]) + 1009 * surface_number + 9176 * placement_number,
            )
            edge_pose_cache = {}
            scene_candidate_rows = []
            for index, skeleton in enumerate(refined):
                planner_start = perf_counter()
                lift = minimum_cost_nuc_lift(
                    robot, surface, skeleton, catalog, transform, edge_pose_cache,
                    axis_tolerance=np.deg2rad(float(robot_cfg["axis_tolerance_degrees"])),
                    characteristic_length=float(robot_cfg["characteristic_length_m"]),
                    sigma_safe=float(frozen["sigma_safe"]),
                    task_edge_samples=int(e06["task_edge_samples"]),
                    surface_path_spacing=float(frozen["coverage_path_sample_spacing_m"]),
                    maximum_joint_step=float(e06["maximum_joint_step_rad"]),
                    position_tolerance=float(robot_cfg["construction_position_tolerance_m"]),
                    max_active_branches=int(e06["ik_max_candidates"]),
                )
                strict = None
                if lift.found:
                    strict = check_strict_coverage_execution(
                        robot, (lift.q_path,), (lift.desired_positions,), (lift.desired_axes,),
                        surface, transform,
                        footprint_radius=float(config["coverage"]["footprint_radius_m"]),
                        position_tolerance=float(robot_cfg["position_tolerance_m"]),
                        axis_tolerance=np.deg2rad(float(robot_cfg["axis_tolerance_degrees"])),
                        characteristic_length=float(robot_cfg["characteristic_length_m"]),
                        sigma_safe=float(frozen["sigma_safe"]),
                        missed_tolerance=1.0,
                        repeat_tolerance=10.0,
                        nuc_error_tolerance=nuc_limit,
                        interpolation_joint_step=float(frozen["q_interpolation_step_rad"]),
                        coverage_path_sample_spacing=float(frozen["coverage_path_sample_spacing_m"]),
                    )
                    save_witness(args.output / "witnesses" / f"{surface_id}_{placement_id}_S{index:02d}.npz", lift)
                row = make_row(
                    surface_id, placement_id, index, skeleton, geometry[index], geometry_index,
                    generation_times[index], refinement_times[index], catalog.enumeration_time,
                    lift, strict, commit, config_hash, planner_start,
                )
                scene_candidate_rows.append(row)
                all_rows.append(row)
                print(
                    f"{surface_id:<10} {placement_id:<6} S{index:02d} "
                    f"NUC={row['E_NUC']:.4f} lift={row['lift_found']} "
                    f"strict={row['strict_kinematics_pass']} Lq={row['L_q']}", flush=True,
                )
            scene = summarize_scene(
                surface_id, placement_id, scene_candidate_rows, geometry_index,
                nuc_limit, float(frozen["delta_NUC"]),
            )
            scene_rows.append(scene)
    registered = candidate_count == expected
    summary = summarize_experiment(
        all_rows, scene_rows, config, frozen, commit, config_hash,
        perf_counter() - experiment_start, registered,
    )
    write_jsonl(args.output / "candidate_results.jsonl", all_rows)
    write_csv(args.output / "candidate_results.csv", all_rows)
    write_jsonl(args.output / "scene_results.jsonl", scene_rows)
    write_csv(args.output / "scene_results.csv", scene_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps({
        "code_commit": commit, "config_hash": config_hash, "registered_run": registered,
        "config": config, "placements": placements, "frozen_contract": frozen,
    }, indent=2) + "\n")
    make_plots(all_rows, scene_rows, args.output)
    (args.output / "README.md").write_text(render_report(summary, scene_rows))
    print(json.dumps(summary, indent=2))


def make_surface(config: dict, surface_id: str, samples_per_face: int):
    maker = make_saddle if surface_id == "saddle" else make_hemisphere
    return maker(**config["surfaces"][surface_id], samples_per_face=samples_per_face)


def generate_variants(surface, count: int, seed: int):
    variants, times, fingerprints = [], [], set()
    jobs = [("upstream_first", None), ("reverse_order", None)]
    next_seed = seed
    while len(variants) < count:
        policy, value = jobs.pop(0) if jobs else ("seeded_random", next_seed)
        if policy == "seeded_random":
            next_seed += 1
        start = perf_counter()
        candidate = generate_nuc_skeleton(surface, policy=policy, seed=value)
        elapsed = perf_counter() - start
        fingerprint = candidate.topological_path.tobytes()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        variants.append(candidate)
        times.append(elapsed)
    return variants, times


def projection_refine(surface, skeleton, iterations: int):
    points = skeleton.waypoints.copy()
    for _ in range(iterations):
        points = project_points(surface, points).points
    return replace(
        skeleton, waypoints=points,
        metadata={**skeleton.metadata, "local_refinement": "equal-budget surface projection", "iterations": iterations},
    )


def make_row(
    surface_id, placement_id, index, skeleton, geometry_record, geometry_index,
    generation_time, refinement_time, ik_time, lift, strict, commit, config_hash,
    planner_start,
):
    metrics, coverage_time = geometry_record
    execution = None if strict is None else strict.execution_cost
    executed = None if strict is None else strict.coverage_metrics
    failure = lift.failure_reason if not lift.found else strict.failure_reason
    return {
        "surface_id": surface_id,
        "placement_id": placement_id,
        "skeleton_id": f"S{index:02d}",
        "skeleton_policy": skeleton.policy,
        "seed": skeleton.seed,
        "repository_commit": commit,
        "experiment_config_hash": config_hash,
        "geometry_baseline": index == geometry_index,
        "E_miss": metrics.missed_error,
        "E_rep": metrics.repeat_error,
        "E_NUC": metrics.nuc_error,
        "single_coverage_fraction": metrics.single_coverage_fraction,
        "overlap_area_fraction": metrics.overlap_area_fraction,
        "max_visit_count": metrics.max_visit_count,
        "surface_path_length": metrics.path_length,
        "legacy_missed_fraction": metrics.legacy_missed_fraction,
        "legacy_coverage_efficiency": metrics.legacy_coverage_efficiency,
        "executed_E_miss": None if executed is None else executed.missed_error,
        "executed_E_rep": None if executed is None else executed.repeat_error,
        "executed_E_NUC": None if executed is None else executed.nuc_error,
        "lift_found": lift.found,
        "strict_kinematics_pass": False if strict is None else strict.kinematics_pass,
        "strict_coverage_pass": False if strict is None else strict.coverage_pass,
        "overall_pass": False if strict is None else strict.overall_pass,
        "failure_reason": failure,
        "L_q": None if execution is None else execution.weighted_joint_length,
        "joint_travel": None if execution is None else execution.unweighted_joint_travel,
        "max_joint_travel": None if execution is None else execution.max_per_joint_travel,
        "min_sigma_min_5": None if strict is None else strict.min_sigma_min_5,
        "min_mu_bar": None if strict is None else strict.min_mu_bar,
        "min_joint_limit_margin": None if strict is None else strict.min_joint_limit_margin,
        "max_position_error": None if strict is None else strict.max_position_error,
        "max_axis_error": None if strict is None else strict.max_axis_error,
        "q_sample_count": 0 if execution is None else execution.q_sample_count,
        "NUC_generation_time": generation_time,
        "geometric_refinement_time": refinement_time,
        "coverage_evaluation_time": coverage_time,
        "IK_enumeration_time": ik_time,
        "continuation_check_time": lift.continuation_time,
        "total_planner_evaluation_time": perf_counter() - planner_start + generation_time + refinement_time + coverage_time + ik_time,
    }


def summarize_scene(surface_id, placement_id, rows, geometry_index, nuc_limit, delta_nuc):
    baseline = rows[geometry_index]
    equivalent = [
        row for row in rows
        if float(row["E_NUC"]) <= nuc_limit + 1e-12 and row["overall_pass"] and row["L_q"] is not None
    ]
    oracle = min(equivalent, key=lambda row: float(row["L_q"])) if equivalent else None
    values = np.asarray([float(row["L_q"]) for row in equivalent])
    spread = None if len(values) < 2 else float((values.max() - values.min()) / values.min())
    reduction = None
    if baseline["overall_pass"] and oracle is not None:
        reduction = 1.0 - float(oracle["L_q"]) / float(baseline["L_q"])
    return {
        "surface_id": surface_id,
        "placement_id": placement_id,
        "geometry_skeleton_id": baseline["skeleton_id"],
        "geometry_L_q": baseline["L_q"],
        "geometry_overall_pass": baseline["overall_pass"],
        "oracle_skeleton_id": None if oracle is None else oracle["skeleton_id"],
        "oracle_L_q": None if oracle is None else oracle["L_q"],
        "oracle_E_NUC": None if oracle is None else oracle["E_NUC"],
        "L_q_reduction": reduction,
        "NUC_equivalent_feasible_candidates": len(equivalent),
        "L_q_relative_spread": spread,
        "nuc_limit": nuc_limit,
        "delta_NUC": delta_nuc,
    }


def summarize_experiment(rows, scenes, config, frozen, commit, config_hash, elapsed, registered):
    spread_hits = sum(
        scene["L_q_relative_spread"] is not None and scene["L_q_relative_spread"] >= 0.1
        for scene in scenes
    )
    reductions = [scene["L_q_reduction"] for scene in scenes if scene["L_q_reduction"] is not None]
    median_reduction = None if not reductions else float(np.median(reductions))
    gate_a = spread_hits >= int(config["e06"]["gate_min_scenes_with_10pct_spread"])
    gate_b = median_reduction is not None and median_reduction >= float(config["e06"]["gate_median_lq_reduction"])
    return {
        "experiment": "E06: NUC skeleton-robot coupling motivation gate",
        "registered_run": registered,
        "branch": "exp/nuc-robot-coupling-v1",
        "code_commit": commit,
        "config_hash": config_hash,
        "candidate_executions": len(rows),
        "scenes": len(scenes),
        "elapsed_seconds": elapsed,
        "frozen_contract": frozen,
        "gate": {
            "scenes_with_at_least_10pct_spread": spread_hits,
            "required_scenes": int(config["e06"]["gate_min_scenes_with_10pct_spread"]),
            "median_paired_L_q_reduction": median_reduction,
            "required_median_reduction": float(config["e06"]["gate_median_lq_reduction"]),
            "condition_A": gate_a,
            "condition_B": gate_b,
            "decision": "GO" if registered and gate_a and gate_b else "NO-GO",
        },
        "failure_counts": dict(Counter(str(row["failure_reason"]) for row in rows if row["failure_reason"])),
        "interpretation": "Optimum only over the finite numerical candidate/continuation graph; no continuous C-space global-optimality claim.",
    }


def save_witness(path: Path, lift) -> None:
    np.savez_compressed(
        path,
        q=np.asarray(lift.q_path, dtype=np.float64),
        desired_positions=lift.desired_positions,
        desired_axes=lift.desired_axes,
    )


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def make_plots(rows, scenes, output: Path) -> None:
    placements = ("P_easy", "P_mid", "P_hard")
    for surface in sorted({row["surface_id"] for row in rows}):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        for axis, placement in zip(axes, placements):
            selected = [row for row in rows if row["surface_id"] == surface and row["placement_id"] == placement]
            for row in selected:
                if row["L_q"] is not None:
                    axis.scatter(row["E_NUC"], row["L_q"], c=row["min_sigma_min_5"], cmap="viridis", vmin=0.0, vmax=1.0, marker="o" if row["overall_pass"] else "x")
                else:
                    axis.scatter(row["E_NUC"], 0, color="red", marker="x")
            axis.set_title(placement); axis.set_xlabel("E_NUC"); axis.set_ylabel("Witness L_q"); axis.grid(alpha=0.25)
        figure.tight_layout(); figure.savefig(output / f"scatter_{surface}.png", dpi=180); plt.close(figure)
    labels = [f"{row['surface_id']}\n{row['placement_id']}" for row in scenes]
    x = np.arange(len(scenes)); width = 0.36
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(x - width / 2, [np.nan if row["geometry_L_q"] is None else row["geometry_L_q"] for row in scenes], width, label="NUC-Geometry")
    axis.bar(x + width / 2, [np.nan if row["oracle_L_q"] is None else row["oracle_L_q"] for row in scenes], width, label="NUC-Execution-Oracle")
    axis.set_xticks(x, labels); axis.set_ylabel("Witness L_q"); axis.legend(); axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(output / "geometry_vs_execution_oracle.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5))
    for surface in sorted({row["surface_id"] for row in rows}):
        for placement in placements:
            selected = [row for row in rows if row["surface_id"] == surface and row["placement_id"] == placement]
            values = sorted((float(row["L_q"]), row["skeleton_id"]) for row in selected if row["L_q"] is not None)
            ranks = {name: rank + 1 for rank, (_, name) in enumerate(values)}
            axis.plot(range(len(selected)), [ranks.get(f"S{i:02d}", np.nan) for i in range(len(selected))], marker=".", label=f"{surface}-{placement}")
    axis.set_xlabel("Skeleton ID"); axis.set_ylabel("L_q rank"); axis.legend(ncol=2, fontsize=7); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output / "placement_candidate_ranking.png", dpi=180); plt.close(figure)
    selected_rows = []
    for scene in scenes:
        for role, key in (("Geometry", "geometry_skeleton_id"), ("Oracle", "oracle_skeleton_id")):
            match = next((row for row in rows if row["surface_id"] == scene["surface_id"] and row["placement_id"] == scene["placement_id"] and row["skeleton_id"] == scene[key]), None)
            if match is not None:
                selected_rows.append((f"{scene['surface_id']}-{scene['placement_id']}-{role}", match))
    figure, axis = plt.subplots(figsize=(12, 4))
    xx = np.arange(len(selected_rows))
    axis.bar(xx, [row["E_miss"] for _, row in selected_rows], label="E_miss")
    axis.bar(xx, [row["E_rep"] for _, row in selected_rows], bottom=[row["E_miss"] for _, row in selected_rows], label="E_rep")
    axis.set_xticks(xx, [name for name, _ in selected_rows], rotation=45, ha="right"); axis.legend(); axis.set_ylabel("NUC decomposition")
    figure.tight_layout(); figure.savefig(output / "selected_coverage_decomposition.png", dpi=180); plt.close(figure)
    reasons = Counter(str(row["failure_reason"]) for row in rows if row["failure_reason"])
    figure, axis = plt.subplots(figsize=(8, 4)); axis.bar(list(reasons), list(reasons.values())); axis.tick_params(axis="x", rotation=35); axis.set_ylabel("Candidates")
    figure.tight_layout(); figure.savefig(output / "failure_reasons.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 4))
    for scene in scenes:
        values = [float(row["L_q"]) for row in rows if row["surface_id"] == scene["surface_id"] and row["placement_id"] == scene["placement_id"] and row["overall_pass"] and float(row["E_NUC"]) <= scene["nuc_limit"]]
        if values:
            axis.hist(values, bins=min(10, len(values)), alpha=0.35, label=f"{scene['surface_id']}-{scene['placement_id']}")
    axis.set_xlabel("Witness L_q"); axis.set_ylabel("Count"); axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output / "equivalent_Lq_distributions.png", dpi=180); plt.close(figure)


def render_report(summary, scenes) -> str:
    lines = [
        "# E06 NUC skeleton-robot coupling", "",
        f"Registered run: `{summary['registered_run']}`  ",
        f"Code commit: `{summary['code_commit']}`  ",
        f"Decision: **{summary['gate']['decision']}**", "",
        "| Surface | Placement | Geometry L_q | Oracle L_q | Reduction | Equivalent | Spread |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in scenes:
        fmt = lambda value: "NA" if value is None else f"{value:.4f}"
        lines.append(f"| {row['surface_id']} | {row['placement_id']} | {fmt(row['geometry_L_q'])} | {fmt(row['oracle_L_q'])} | {fmt(row['L_q_reduction'])} | {row['NUC_equivalent_feasible_candidates']} | {fmt(row['L_q_relative_spread'])} |")
    lines += ["", "The oracle is optimal only over the finite numerical candidate/continuation graph. This experiment does not support global C-space optimality, topological-disconnection, learned-planner, timing, or physical-execution claims."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
