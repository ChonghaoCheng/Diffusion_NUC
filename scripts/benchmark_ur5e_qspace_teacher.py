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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.coverage import load_teacher_instance, surface_from_teacher_archive
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.robot.qspace_coverage_teacher import build_qspace_coverage_teacher
from diffusion_coverage.robot.surface_ik_graph import (
    build_surface_ik_graph,
    load_surface_ik_graph,
    save_surface_ik_graph,
)
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


DEFAULT_MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)
_ROBOT: UR5eKinematics | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark strict component-internal UR5e q-space teachers")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-cache", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--surfaces", nargs="+", default=["cylinder", "hemisphere"])
    parser.add_argument("--max-instances-per-surface", type=int, default=2)
    parser.add_argument("--grid-shape", nargs=2, type=int, default=[5, 5])
    parser.add_argument("--tau-degrees", type=float, default=10.0)
    parser.add_argument(
        "--transform-tau-degrees",
        type=float,
        help="Tolerance row used to select a fixed placement; defaults to --tau-degrees",
    )
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--segment-budgets", nargs="+", type=int)
    parser.add_argument("--random-restarts", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--orientation-cone-samples", type=int, default=9)
    parser.add_argument("--inner-cone-degrees", type=float, default=3.0)
    parser.add_argument("--inner-cone-samples", type=int, default=9)
    parser.add_argument("--inner-max-candidates", type=int, default=12)
    parser.add_argument("--task-edge-samples", type=int, default=7)
    parser.add_argument("--task-position-tolerance", type=float, default=0.003)
    parser.add_argument("--hard-position-tolerance", type=float, default=0.003)
    parser.add_argument("--candidate-match-tolerance", type=float, default=0.8)
    parser.add_argument("--max-target-matches", type=int, default=4)
    parser.add_argument(
        "--route-objective",
        choices=("surface_then_joint", "joint_then_surface"),
        default="surface_then_joint",
    )
    parser.add_argument("--maximum-joint-step", type=float, default=0.8)
    parser.add_argument("--minimum-manipulability", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "instances").mkdir(exist_ok=True)
    (args.output / "graphs").mkdir(exist_ok=True)
    transform_tau = (
        args.tau_degrees
        if args.transform_tau_degrees is None
        else args.transform_tau_degrees
    )
    transforms = load_transforms(args.fixed_results, transform_tau)
    grouped = {surface: [] for surface in args.surfaces}
    for row in load_manifest(args.dataset):
        instance_id = str(row["instance_id"])
        surface_id = str(row["surface_id"])
        if surface_id in grouped and instance_id in transforms:
            grouped[surface_id].append(row)
    selected = []
    for surface_id in args.surfaces:
        selected.extend(sorted(grouped[surface_id], key=lambda row: str(row["instance_id"]))[: args.max_instances_per_surface])
    settings = vars(args).copy()
    for key in ("dataset", "fixed_results", "output", "model", "graph_cache"):
        settings[key] = None if settings[key] is None else str(settings[key])
    tasks = [
        (row, transforms[str(row["instance_id"])].tolist(), settings, index)
        for index, row in enumerate(selected)
    ]
    start = perf_counter()
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.model),),
    ) as executor:
        futures = [executor.submit(evaluate_instance, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            instance_rows = future.result()
            rows.extend(instance_rows)
            for row in instance_rows:
                print(
                    f"[{completed:03d}/{len(tasks):03d}] {row['instance_id']:<24} "
                    f"k={row['max_segments']:<2d} teacher={row['feasible']} "
                    f"nodes={row['covered_node_fraction']:.1%} miss={row['missed_fraction']}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["instance_id"], row["max_segments"]))
    summary = summarize(rows, settings, perf_counter() - start)
    with (args.output / "instances.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def initialize_worker(model_path: str) -> None:
    global _ROBOT
    _ROBOT = UR5eKinematics(model_path)


def evaluate_instance(task) -> list[dict]:
    row, transform_raw, settings, instance_number = task
    if _ROBOT is None:
        raise RuntimeError("worker robot is unavailable")
    dataset = Path(settings["dataset"])
    output = Path(settings["output"])
    archive = load_teacher_instance(dataset / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    metadata = dict(archive["metadata"])
    transform = np.asarray(transform_raw, dtype=np.float64)
    surface_scale = float(np.mean(np.linalg.svd(transform[:3, :3], compute_uv=False)))
    instance_id = str(row["instance_id"])
    start = perf_counter()
    cached_path = (
        None
        if settings["graph_cache"] is None
        else Path(settings["graph_cache"]) / f"{instance_id}.npz"
    )
    if cached_path is not None and cached_path.exists():
        graph = load_surface_ik_graph(cached_path)
        validate_cached_graph(graph, settings)
    else:
        graph = build_surface_ik_graph(
            _ROBOT,
            surface,
            transform,
            grid_shape=tuple(settings["grid_shape"]),
            axis_tolerance=np.deg2rad(settings["tau_degrees"]),
            random_restarts=settings["random_restarts"],
            max_candidates=settings["max_candidates"],
            orientation_cone_samples=settings["orientation_cone_samples"],
            inner_cone_tolerance=np.deg2rad(settings["inner_cone_degrees"]),
            inner_cone_samples=settings["inner_cone_samples"],
            inner_max_candidates=settings["inner_max_candidates"],
            maximum_joint_step=settings["maximum_joint_step"],
            minimum_manipulability=settings["minimum_manipulability"],
            task_edge_samples=settings["task_edge_samples"],
            task_position_tolerance=settings["task_position_tolerance"],
            task_transition_mode="ik_continuation",
            candidate_match_tolerance=settings["candidate_match_tolerance"],
            max_target_matches=settings["max_target_matches"],
            seed=settings["seed"] + 104729 * instance_number,
        )
    save_surface_ik_graph(output / "graphs" / f"{instance_id}.npz", graph)
    graph_time = perf_counter() - start
    budgets = settings["segment_budgets"] or [settings["max_segments"]]
    result_rows = []
    footprint_radius = float(metadata["teacher_config"]["footprint_radius"])
    for budget in budgets:
        teacher_start = perf_counter()
        result = build_qspace_coverage_teacher(
            _ROBOT,
            surface,
            transform,
            graph,
            max_segments=budget,
            footprint_radius=footprint_radius,
            route_objective=settings["route_objective"],
            hard_position_tolerance=settings["hard_position_tolerance"],
        )
        save_result(output / "instances" / f"{instance_id}_k{budget}.npz", result)
        metrics = result.coverage_metrics
        physical_radius = footprint_radius * surface_scale
        physical_covered_area = (
            None if metrics is None else metrics.covered_area * surface_scale**2
        )
        end_cap_area = len(result.q_segments) * np.pi * physical_radius**2
        planar_tube_bound = (
            None
            if physical_covered_area is None
            else max(0.0, (physical_covered_area - end_cap_area) / (2.0 * physical_radius))
        )
        fk_efficiency = (
            None
            if physical_covered_area is None
            else physical_covered_area
            / (2.0 * physical_radius * result.task_path_length + end_cap_area)
        )
        result_rows.append({
            "instance_id": instance_id,
            "surface_id": str(row["surface_id"]),
            "max_segments": budget,
            "feasible": result.feasible,
            "failure_reason": result.failure_reason,
            "max_position_error": result.metadata.get("max_position_error"),
            "max_axis_error": result.metadata.get("max_axis_error"),
            "reachable_fraction": graph.summary()["reachable_fraction"],
            "largest_component_fraction": graph.summary()["largest_component_fraction"],
            "covered_node_fraction": float(result.covered_node_mask.mean()),
            "selected_components": len(result.selected_components),
            "q_samples": int(sum(len(segment) for segment in result.q_segments)),
            "joint_travel": None if not result.feasible else result.joint_travel,
            "squared_joint_motion": None if not result.feasible else result.squared_joint_motion,
            "task_path_length": None if not result.feasible else result.task_path_length,
            "repeated_node_visits": result.repeated_node_visits,
            "missed_fraction": None if metrics is None else metrics.missed_fraction,
            "path_length": None if metrics is None else metrics.path_length,
            "mesh_path_length_physical": None
            if metrics is None
            else metrics.path_length * surface_scale,
            "fk_coverage_efficiency": fk_efficiency,
            "planar_tube_bound_length": planar_tube_bound,
            "planar_tube_bound_length_ratio": None
            if planar_tube_bound is None or planar_tube_bound <= 0.0
            else result.task_path_length / planar_tube_bound,
            "coverage_efficiency": None if metrics is None else metrics.coverage_efficiency,
            "graph_time": graph_time,
            "teacher_time": perf_counter() - teacher_start,
        })
    return result_rows


def validate_cached_graph(graph, settings: dict) -> None:
    expected = {
        "grid_shape": list(settings["grid_shape"]),
        "axis_tolerance_degrees": float(settings["tau_degrees"]),
        "task_edge_samples": int(settings["task_edge_samples"]),
        "task_position_tolerance": float(settings["task_position_tolerance"]),
        "candidate_match_tolerance": float(settings["candidate_match_tolerance"]),
        "max_target_matches": int(settings["max_target_matches"]),
    }
    actual = {
        "grid_shape": list(graph.grid_shape),
        "axis_tolerance_degrees": float(graph.metadata["axis_tolerance_degrees"]),
        "task_edge_samples": int(graph.metadata["task_edge_samples"]),
        "task_position_tolerance": float(graph.metadata["task_position_tolerance"]),
        "candidate_match_tolerance": float(graph.metadata["candidate_match_tolerance"]),
        "max_target_matches": int(graph.metadata["max_target_matches"]),
    }
    for key, value in expected.items():
        if isinstance(value, float):
            matches = np.isclose(actual[key], value)
        else:
            matches = actual[key] == value
        if not matches:
            raise ValueError(f"cached graph {key}={actual[key]} does not match requested {value}")


def save_result(path: Path, result) -> None:
    maximum = max((len(segment) for segment in result.q_segments), default=0)
    q = np.zeros((len(result.q_segments), maximum, 6), dtype=np.float64)
    positions = np.zeros((len(result.q_segments), maximum, 3), dtype=np.float64)
    axes = np.zeros_like(positions)
    mask = np.zeros((len(result.q_segments), maximum), dtype=bool)
    for index, (q_values, target_positions, target_axes) in enumerate(
        zip(result.q_segments, result.target_position_segments, result.target_axis_segments)
    ):
        q[index, : len(q_values)] = q_values
        positions[index, : len(q_values)] = target_positions
        axes[index, : len(q_values)] = target_axes
        mask[index, : len(q_values)] = True
    np.savez_compressed(
        path,
        q=q,
        target_positions=positions,
        target_axes=axes,
        mask=mask,
        selected_components=result.selected_components,
        covered_node_mask=result.covered_node_mask,
        feasible=np.asarray(result.feasible),
        failure_reason=np.asarray("" if result.failure_reason is None else result.failure_reason),
    )


def load_transforms(path: Path, tau_degrees: float) -> dict[str, np.ndarray]:
    transforms = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["tau_degrees"]) - tau_degrees) <= 1e-9:
                transforms[row["instance_id"]] = np.asarray(
                    json.loads(row["transform_base_from_surface"]), dtype=np.float64
                )
    return transforms


def summarize(rows: list[dict], settings: dict, elapsed: float) -> dict:
    budgets = sorted({row["max_segments"] for row in rows})
    return {
        "settings": settings,
        "instances": len(rows),
        "elapsed_seconds": elapsed,
        "by_budget": {
            str(budget): aggregate([row for row in rows if row["max_segments"] == budget])
            for budget in budgets
        },
    }


def aggregate(rows: list[dict]) -> dict:
    feasible = [row for row in rows if row["feasible"]]
    return {
        "instances": len(rows),
        "feasible_rate": len(feasible) / max(len(rows), 1),
        "mean_covered_node_fraction": float(np.mean([row["covered_node_fraction"] for row in rows])),
        "mean_missed_fraction": None if not feasible else float(np.mean([row["missed_fraction"] for row in feasible])),
        "mean_joint_travel": None if not feasible else float(np.mean([row["joint_travel"] for row in feasible])),
        "mean_task_path_length": None
        if not feasible
        else float(np.mean([row["task_path_length"] for row in feasible])),
        "mean_mesh_path_length": None
        if not feasible
        else float(np.mean([row["path_length"] for row in feasible])),
        "mean_mesh_path_length_physical": None
        if not feasible
        else float(np.mean([row["mesh_path_length_physical"] for row in feasible])),
        "mean_fk_coverage_efficiency": None
        if not feasible
        else float(np.mean([row["fk_coverage_efficiency"] for row in feasible])),
        "mean_planar_tube_bound_length_ratio": None
        if not feasible
        else float(
            np.mean([row["planar_tube_bound_length_ratio"] for row in feasible])
        ),
        "failure_counts": {
            reason: sum(row["failure_reason"] == reason for row in rows)
            for reason in sorted({row["failure_reason"] for row in rows if row["failure_reason"]})
        },
    }


if __name__ == "__main__":
    main()
