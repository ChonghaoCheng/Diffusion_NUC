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
from diffusion_coverage.robot.surface_ik_graph import build_surface_ik_graph
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


DEFAULT_MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)
_ROBOT: UR5eKinematics | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate coarse real UR5e IK components over fixed 3D surface placements"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--surfaces", nargs="+", default=["cylinder", "hemisphere"])
    parser.add_argument("--max-instances-per-surface", type=int, default=1)
    parser.add_argument("--grid-shape", nargs=2, type=int, default=[6, 6])
    parser.add_argument("--tau-degrees", type=float, default=3.0)
    parser.add_argument("--random-restarts", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--orientation-cone-samples", type=int, default=1)
    parser.add_argument("--inner-cone-degrees", type=float)
    parser.add_argument("--inner-cone-samples", type=int, default=1)
    parser.add_argument("--inner-max-candidates", type=int)
    parser.add_argument("--maximum-joint-step", type=float, default=0.8)
    parser.add_argument("--minimum-manipulability", type=float, default=1e-5)
    parser.add_argument("--task-edge-samples", type=int, default=5)
    parser.add_argument("--task-position-tolerance", type=float, default=0.003)
    parser.add_argument(
        "--task-transition-mode",
        choices=("linear_tracking", "ik_continuation"),
        default="ik_continuation",
    )
    parser.add_argument("--candidate-match-tolerance", type=float, default=0.25)
    parser.add_argument("--max-target-matches", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_instances_per_surface < 1 or args.workers < 1:
        raise ValueError("instance and worker counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "instances").mkdir(exist_ok=True)
    transforms = load_transforms(args.fixed_results, args.tau_degrees)
    grouped = {surface: [] for surface in args.surfaces}
    for row in load_manifest(args.dataset):
        instance_id = str(row["instance_id"])
        surface = str(row["surface_id"])
        if surface in grouped and instance_id in transforms:
            grouped[surface].append(row)
    selected = []
    for surface in args.surfaces:
        selected.extend(
            sorted(grouped[surface], key=lambda row: str(row["instance_id"]))[
                : args.max_instances_per_surface
            ]
        )
    settings = {
        "grid_shape": list(args.grid_shape),
        "tau_degrees": args.tau_degrees,
        "random_restarts": args.random_restarts,
        "max_candidates": args.max_candidates,
        "orientation_cone_samples": args.orientation_cone_samples,
        "inner_cone_degrees": args.inner_cone_degrees,
        "inner_cone_samples": args.inner_cone_samples,
        "inner_max_candidates": args.inner_max_candidates,
        "maximum_joint_step": args.maximum_joint_step,
        "minimum_manipulability": args.minimum_manipulability,
        "task_edge_samples": args.task_edge_samples,
        "task_position_tolerance": args.task_position_tolerance,
        "task_transition_mode": args.task_transition_mode,
        "candidate_match_tolerance": args.candidate_match_tolerance,
        "max_target_matches": args.max_target_matches,
        "seed": args.seed,
    }
    tasks = [
        (
            str(args.dataset),
            str(args.output),
            row,
            transforms[str(row["instance_id"])].tolist(),
            settings,
            index,
        )
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
            row = future.result()
            rows.append(row)
            print(
                f"[{completed:03d}/{len(tasks):03d}] {row['instance_id']:<24} "
                f"reach={row['reachable_fraction']:.1%} "
                f"components={row['components']:<3d} "
                f"largest={row['largest_component_fraction']:.1%}",
                flush=True,
            )
    rows.sort(key=lambda row: str(row["instance_id"]))
    summary = summarize(rows, settings, perf_counter() - start)
    with (args.output / "instances.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2))


def initialize_worker(model_path: str) -> None:
    global _ROBOT
    _ROBOT = UR5eKinematics(Path(model_path))


def evaluate_instance(task):
    dataset_raw, output_raw, row, transform_raw, settings, instance_number = task
    if _ROBOT is None:
        raise RuntimeError("worker robot was not initialized")
    dataset, output = Path(dataset_raw), Path(output_raw)
    archive = load_teacher_instance(dataset / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    start = perf_counter()
    graph = build_surface_ik_graph(
        _ROBOT,
        surface,
        np.asarray(transform_raw, dtype=np.float64),
        grid_shape=tuple(int(value) for value in settings["grid_shape"]),
        axis_tolerance=np.deg2rad(float(settings["tau_degrees"])),
        random_restarts=int(settings["random_restarts"]),
        max_candidates=int(settings["max_candidates"]),
        orientation_cone_samples=int(settings["orientation_cone_samples"]),
        inner_cone_tolerance=(
            None
            if settings["inner_cone_degrees"] is None
            else np.deg2rad(float(settings["inner_cone_degrees"]))
        ),
        inner_cone_samples=int(settings["inner_cone_samples"]),
        inner_max_candidates=(
            None
            if settings["inner_max_candidates"] is None
            else int(settings["inner_max_candidates"])
        ),
        maximum_joint_step=float(settings["maximum_joint_step"]),
        minimum_manipulability=float(settings["minimum_manipulability"]),
        task_edge_samples=int(settings["task_edge_samples"]),
        task_position_tolerance=float(settings["task_position_tolerance"]),
        task_transition_mode=str(settings["task_transition_mode"]),
        candidate_match_tolerance=float(settings["candidate_match_tolerance"]),
        max_target_matches=int(settings["max_target_matches"]),
        seed=int(settings["seed"]) + 104729 * instance_number,
    )
    instance_id = str(row["instance_id"])
    save_graph(output / "instances" / f"{instance_id}.npz", graph)
    summary = graph.summary()
    summary.update({"instance_id": instance_id, "solve_time": perf_counter() - start})
    return summary


def save_graph(path: Path, graph) -> None:
    maximum_candidates = max((len(layer) for layer in graph.candidates), default=0)
    q = np.zeros((graph.num_nodes, maximum_candidates, 6), dtype=np.float32)
    candidate_mask = np.zeros((graph.num_nodes, maximum_candidates), dtype=bool)
    components = np.full((graph.num_nodes, maximum_candidates), -1, dtype=np.int32)
    manipulability = np.zeros((graph.num_nodes, maximum_candidates), dtype=np.float32)
    joint_margin = np.zeros((graph.num_nodes, maximum_candidates), dtype=np.float32)
    for node, (layer, labels) in enumerate(zip(graph.candidates, graph.component_labels)):
        for candidate, value in enumerate(layer):
            q[node, candidate] = value.q
            candidate_mask[node, candidate] = True
            components[node, candidate] = labels[candidate]
            manipulability[node, candidate] = value.manipulability
            joint_margin[node, candidate] = value.joint_limit_margin
    edge_compatibility = np.zeros(
        (graph.num_edges, maximum_candidates, maximum_candidates), dtype=bool
    )
    for edge, values in enumerate(graph.edge_compatibility):
        edge_compatibility[edge, : values.shape[0], : values.shape[1]] = values
    np.savez_compressed(
        path,
        uv=graph.uv,
        positions=graph.positions,
        axes=graph.axes,
        q_candidates=q,
        candidate_mask=candidate_mask,
        component_labels=components,
        manipulability=manipulability,
        joint_limit_margin=joint_margin,
        edge_index=graph.edge_index,
        edge_compatibility=edge_compatibility,
        grid_shape=np.asarray(graph.grid_shape),
        metadata_json=np.asarray(json.dumps(graph.metadata, sort_keys=True)),
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


def summarize(rows, settings, elapsed):
    surfaces = sorted({str(row["surface_id"]) for row in rows})
    return {
        "settings": settings,
        "instances": len(rows),
        "elapsed_seconds": elapsed,
        "overall": aggregate(rows),
        "by_surface": {
            surface: aggregate([row for row in rows if row["surface_id"] == surface])
            for surface in surfaces
        },
    }


def aggregate(rows):
    return {
        "instances": len(rows),
        "mean_reachable_fraction": float(np.mean([row["reachable_fraction"] for row in rows])),
        "mean_components": float(np.mean([row["components"] for row in rows])),
        "mean_multi_node_components": float(
            np.mean([row["components_spanning_multiple_nodes"] for row in rows])
        ),
        "mean_largest_component_fraction": float(
            np.mean([row["largest_component_fraction"] for row in rows])
        ),
        "mean_candidates_per_reachable_node": float(
            np.mean([row["mean_candidates_per_reachable_node"] for row in rows])
        ),
        "mean_solve_time": float(np.mean([row["solve_time"] for row in rows])),
    }


def render_report(summary) -> str:
    lines = [
        "# Coarse UR5e surface IK-component graph",
        "",
        "| Surface | N | Reachable nodes | Components | Multi-node components | Largest component |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for surface, row in summary["by_surface"].items():
        lines.append(
            f"| {surface} | {row['instances']} | {row['mean_reachable_fraction']:.2%} | "
            f"{row['mean_components']:.2f} | {row['mean_multi_node_components']:.2f} | "
            f"{row['mean_largest_component_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Components are exact only over the numerically enumerated IK candidates and coarse grid edges.",
            "They are not topology certificates for the continuous configuration space.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
