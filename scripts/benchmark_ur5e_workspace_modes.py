#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.coverage import (
    CoveragePlan,
    constrained_coverage_key,
    load_teacher_instance,
    surface_from_teacher_archive,
)
from diffusion_coverage.coverage.resampling import resample_surface_path
from diffusion_coverage.learning import canonical_candidate_name, load_manifest
from diffusion_coverage.robot.ur5e_mujoco import (
    UR5eKinematics,
    interpolate_vertex_normals,
    transform_surface_pose_path,
)
from diffusion_coverage.surface.projection import project_points


DEFAULT_MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)
_ROBOT: UR5eKinematics | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-placement UR5e audit of geometry-best versus best-per-mode workspace paths"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--surfaces", nargs="+", default=["cylinder", "hemisphere"])
    parser.add_argument("--max-instances-per-surface", type=int, default=None)
    parser.add_argument("--tau-degrees", type=float, default=3.0)
    parser.add_argument("--pose-spacing", type=float, default=0.005)
    parser.add_argument("--maximum-pose-samples", type=int, default=None)
    parser.add_argument("--random-restarts", type=int, default=16)
    parser.add_argument("--maximum-joint-step", type=float, default=0.8)
    parser.add_argument("--max-active-branches", type=int, default=12)
    parser.add_argument("--target-extent", type=float, default=0.32)
    parser.add_argument(
        "--candidate-policy",
        choices=("best_per_mode", "all"),
        default="best_per_mode",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument(
        "--verify-rescues-from",
        type=Path,
        default=None,
        help="Candidate-results JSONL from a prior run; reevaluate each rescue and its geometry baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pose_spacing <= 0.0 or args.target_extent <= 0.0:
        raise ValueError("pose spacing and target extent must be positive")
    if args.workers < 1 or args.random_restarts < 1 or args.max_active_branches < 1:
        raise ValueError("worker and search counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    transforms, old_results = load_fixed_results(
        args.fixed_results, tau_degrees=args.tau_degrees
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in load_manifest(args.dataset):
        surface = str(row["surface_id"])
        instance_id = str(row["instance_id"])
        if surface in args.surfaces and instance_id in transforms:
            grouped[surface].append(row)
    rescue_selection = (
        load_rescue_selection(args.verify_rescues_from)
        if args.verify_rescues_from is not None
        else None
    )
    selected = []
    for surface in args.surfaces:
        rows = sorted(grouped[surface], key=lambda row: str(row["instance_id"]))
        if args.max_instances_per_surface is not None:
            rows = rows[: args.max_instances_per_surface]
        selected.extend(
            row
            for row in rows
            if rescue_selection is None
            or str(row["instance_id"]) in rescue_selection
        )
    if not selected:
        raise ValueError("no dataset rows have matching fixed transforms")

    settings = {
        "dataset": str(args.dataset),
        "model": str(args.model),
        "tau_degrees": args.tau_degrees,
        "pose_spacing": args.pose_spacing,
        "maximum_pose_samples": args.maximum_pose_samples,
        "random_restarts": args.random_restarts,
        "maximum_joint_step": args.maximum_joint_step,
        "max_active_branches": args.max_active_branches,
        "target_extent": args.target_extent,
        "candidate_policy": args.candidate_policy,
        "candidate_indices": rescue_selection,
        "seed": args.seed,
    }
    tasks = [
        (
            str(args.dataset),
            row,
            transforms[str(row["instance_id"])].tolist(),
            old_results[str(row["instance_id"])],
            settings,
            index,
        )
        for index, row in enumerate(selected)
    ]
    start = perf_counter()
    candidate_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.model),),
    ) as executor:
        futures = [executor.submit(evaluate_instance, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            candidates, instance = future.result()
            candidate_rows.extend(candidates)
            instance_rows.append(instance)
            print(
                f"[{completed:03d}/{len(tasks):03d}] {instance['instance_id']:<24} "
                f"base={str(instance['geometry_best_liftable']):<5} "
                f"any={str(instance['best_of_mode_liftable']):<5} "
                f"rescue={str(instance['mode_rescue']):<5}",
                flush=True,
            )
    candidate_rows.sort(key=lambda row: (str(row["instance_id"]), str(row["mode_name"])))
    instance_rows.sort(key=lambda row: str(row["instance_id"]))
    elapsed = perf_counter() - start
    summary = summarize(instance_rows, candidate_rows, settings, elapsed)
    write_jsonl(args.output / "candidate_results.jsonl", candidate_rows)
    write_jsonl(args.output / "instance_results.jsonl", instance_rows)
    write_csv(args.output / "candidate_results.csv", candidate_rows)
    write_csv(args.output / "instance_results.csv", instance_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "report.md").write_text(render_report(summary))
    plot_summary(summary, args.output / "workspace_mode_rescue.png")
    print(json.dumps(summary, indent=2))


def initialize_worker(model_path: str) -> None:
    global _ROBOT
    _ROBOT = UR5eKinematics(Path(model_path))


def evaluate_instance(task):
    dataset_raw, row, transform_raw, old_result, settings, instance_number = task
    if _ROBOT is None:
        raise RuntimeError("worker robot was not initialized")
    dataset = Path(dataset_raw)
    archive = load_teacher_instance(dataset / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    metadata = dict(archive["metadata"])
    config = dict(metadata["teacher_config"])
    tolerance = float(config["missed_tolerance"])
    metrics = np.asarray(archive["candidate_metrics"])
    names = [canonical_candidate_name(str(name)) for name in archive["proposal_names"]]
    best_per_mode = select_best_per_mode(metrics, names, tolerance)
    requested = settings.get("candidate_indices")
    if requested is not None:
        selected = [int(index) for index in requested[str(row["instance_id"])]]
    else:
        selected = (
            sorted(best_per_mode.values())
            if settings["candidate_policy"] == "best_per_mode"
            else [
                index
                for index in range(len(metrics))
                if float(metrics[index, 0]) <= tolerance + 1e-12
            ]
        )
    geometry_best = min(
        selected,
        key=lambda index: constrained_coverage_key(
            float(metrics[index, 0]), float(metrics[index, 1]), tolerance
        ),
    )
    transform = np.asarray(transform_raw, dtype=np.float64)
    scale = similarity_scale(surface, float(settings["target_extent"]))
    candidate_rows = []
    for candidate_index in sorted(selected, key=lambda index: (names[index], index)):
        mode_name = names[candidate_index]
        plan = CoveragePlan(
            archive["candidate_waypoints"][candidate_index],
            archive["candidate_segment_mask"][candidate_index],
            archive["candidate_waypoint_mask"][candidate_index],
        )
        physical_length = float(metrics[candidate_index, 1]) * scale
        pose_samples = max(
            2, int(np.ceil(physical_length / float(settings["pose_spacing"]))) + 1
        )
        maximum_pose_samples = settings["maximum_pose_samples"]
        if maximum_pose_samples is not None:
            pose_samples = min(pose_samples, int(maximum_pose_samples))
        path = resample_surface_path(
            surface, plan.active_paths()[0], num_waypoints=pose_samples
        )
        projection = project_points(surface, path)
        normals = interpolate_vertex_normals(
            surface.vertices,
            surface.faces,
            surface.face_normals,
            surface.face_areas,
            projection.face_indices,
            projection.barycentric,
        )
        positions, axes = transform_surface_pose_path(path, normals, transform)
        # Use identical restart draws for every mode of one instance so the
        # paired comparison changes only the workspace path.
        seed = stable_seed(int(settings["seed"]), str(row["instance_id"]))
        start = perf_counter()
        result = _ROBOT.check_continuous_lift(
            positions,
            axes,
            axis_tolerance=np.deg2rad(float(settings["tau_degrees"])),
            random_restarts=int(settings["random_restarts"]),
            maximum_joint_step=float(settings["maximum_joint_step"]),
            max_active_branches=int(settings["max_active_branches"]),
            rng=np.random.default_rng(seed),
        )
        candidate_rows.append(
            {
                "instance_id": str(row["instance_id"]),
                "surface_id": str(row["surface_id"]),
                "candidate_index": candidate_index,
                "mode_name": mode_name,
                "geometry_best": candidate_index == geometry_best,
                "liftable": result.feasible,
                "failure_reason": result.failure_reason,
                "failed_waypoint": result.metadata.get("failed_waypoint"),
                "physical_path_length": physical_length,
                "missed_fraction": float(metrics[candidate_index, 0]),
                "pose_samples": pose_samples,
                "min_candidates": min(result.candidate_counts) if result.candidate_counts else 0,
                "mean_candidates": (
                    float(np.mean(result.candidate_counts)) if result.candidate_counts else 0.0
                ),
                "search_edges": result.search_edges,
                "solve_time": perf_counter() - start,
            }
        )
    baseline = next(row for row in candidate_rows if row["geometry_best"])
    feasible = [row for row in candidate_rows if row["liftable"]]
    best_liftable = (
        min(feasible, key=lambda row: float(row["physical_path_length"]))
        if feasible
        else None
    )
    instance_row = {
        "instance_id": str(row["instance_id"]),
        "surface_id": str(row["surface_id"]),
        "modes_evaluated": len(candidate_rows),
        "old_geometry_best_liftable": bool(old_result["liftable"]),
        "geometry_best_liftable": bool(baseline["liftable"]),
        "best_of_mode_liftable": bool(feasible),
        "mode_rescue": not bool(baseline["liftable"]) and bool(feasible),
        "mode_regression": bool(baseline["liftable"]) and not bool(feasible),
        "geometry_best_mode": str(baseline["mode_name"]),
        "selected_liftable_mode": None if best_liftable is None else str(best_liftable["mode_name"]),
        "geometry_best_length": float(baseline["physical_path_length"]),
        "selected_liftable_length": (
            None if best_liftable is None else float(best_liftable["physical_path_length"])
        ),
        "selected_to_geometry_length_ratio": (
            None
            if best_liftable is None
            else float(best_liftable["physical_path_length"])
            / float(baseline["physical_path_length"])
        ),
    }
    return candidate_rows, instance_row


def select_best_per_mode(
    metrics: np.ndarray, names: list[str], tolerance: float
) -> dict[str, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, mode_name in enumerate(names):
        if float(metrics[index, 0]) <= tolerance + 1e-12:
            groups[mode_name].append(index)
    if not groups:
        raise ValueError("instance has no hard-feasible workspace candidates")
    return {
        mode_name: min(
            indices,
            key=lambda index: constrained_coverage_key(
                float(metrics[index, 0]), float(metrics[index, 1]), tolerance
            ),
        )
        for mode_name, indices in groups.items()
    }


def load_fixed_results(
    path: Path, *, tau_degrees: float
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    transforms = {}
    results = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["tau_degrees"]) - tau_degrees) > 1e-9:
                continue
            instance_id = row["instance_id"]
            transforms[instance_id] = np.asarray(
                json.loads(row["transform_base_from_surface"]), dtype=np.float64
            )
            results[instance_id] = {
                "liftable": row["liftable"].lower() == "true",
                "failure_reason": row["failure_reason"],
            }
    if not transforms:
        raise ValueError(f"no fixed results found for tau={tau_degrees}")
    return transforms, results


def load_rescue_selection(path: Path) -> dict[str, list[int]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["instance_id"])].append(row)
    selected = {}
    for instance_id, candidates in grouped.items():
        baseline = next(row for row in candidates if row["geometry_best"])
        feasible = [row for row in candidates if row["liftable"]]
        if baseline["liftable"] or not feasible:
            continue
        best_liftable = min(feasible, key=lambda row: float(row["physical_path_length"]))
        selected[instance_id] = sorted(
            {int(baseline["candidate_index"]), int(best_liftable["candidate_index"])}
        )
    if not selected:
        raise ValueError("source results contain no paired workspace rescues")
    return selected


def summarize(instance_rows, candidate_rows, settings, elapsed):
    surfaces = sorted({str(row["surface_id"]) for row in instance_rows})
    return {
        "settings": settings,
        "instances": len(instance_rows),
        "candidate_evaluations": len(candidate_rows),
        "elapsed_seconds": elapsed,
        "overall": summarize_instances(instance_rows, candidate_rows),
        "by_surface": {
            surface: summarize_instances(
                [row for row in instance_rows if row["surface_id"] == surface],
                [row for row in candidate_rows if row["surface_id"] == surface],
            )
            for surface in surfaces
        },
    }


def summarize_instances(instance_rows, candidate_rows):
    ratios = [
        float(row["selected_to_geometry_length_ratio"])
        for row in instance_rows
        if row["selected_to_geometry_length_ratio"] is not None
    ]
    return {
        "instances": len(instance_rows),
        "old_geometry_best_success_rate": float(
            np.mean([row["old_geometry_best_liftable"] for row in instance_rows])
        ),
        "geometry_best_success_rate": float(
            np.mean([row["geometry_best_liftable"] for row in instance_rows])
        ),
        "best_of_mode_success_rate": float(
            np.mean([row["best_of_mode_liftable"] for row in instance_rows])
        ),
        "paired_mode_rescue_rate": float(np.mean([row["mode_rescue"] for row in instance_rows])),
        "paired_mode_regression_rate": float(
            np.mean([row["mode_regression"] for row in instance_rows])
        ),
        "mean_selected_length_ratio": None if not ratios else float(np.mean(ratios)),
        "median_selected_length_ratio": None if not ratios else float(np.median(ratios)),
        "selected_mode_counts": dict(
            Counter(
                str(row["selected_liftable_mode"])
                for row in instance_rows
                if row["selected_liftable_mode"] is not None
            )
        ),
        "candidate_failure_counts": dict(Counter(str(row["failure_reason"]) for row in candidate_rows)),
        "mean_candidate_solve_time": float(np.mean([row["solve_time"] for row in candidate_rows])),
    }


def render_report(summary) -> str:
    lines = [
        "# Fixed-placement UR5e workspace-mode audit",
        "",
        "All methods reuse the recorded base transform and unchanged finite-beam continuation checker.",
        "",
        "| Surface | N | Old baseline | v10 geometry-best | Best-of-mode | Paired rescue | Length ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for surface, row in summary["by_surface"].items():
        ratio = row["mean_selected_length_ratio"]
        lines.append(
            f"| {surface} | {row['instances']} | {row['old_geometry_best_success_rate']:.2%} | "
            f"{row['geometry_best_success_rate']:.2%} | {row['best_of_mode_success_rate']:.2%} | "
            f"{row['paired_mode_rescue_rate']:.2%} | {'n/a' if ratio is None else f'{ratio:.4f}'} |"
        )
    row = summary["overall"]
    ratio = row["mean_selected_length_ratio"]
    lines.extend(
        [
            f"| **Overall** | **{row['instances']}** | **{row['old_geometry_best_success_rate']:.2%}** | "
            f"**{row['geometry_best_success_rate']:.2%}** | **{row['best_of_mode_success_rate']:.2%}** | "
            f"**{row['paired_mode_rescue_rate']:.2%}** | **{'n/a' if ratio is None else f'{ratio:.4f}'}** |",
            "",
            "These are numerical continuation results, not C-space topology certificates. Workpiece collision is not modeled.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_summary(summary, output: Path) -> None:
    surfaces = list(summary["by_surface"])
    x = np.arange(len(surfaces))
    width = 0.35
    geometry = [summary["by_surface"][surface]["geometry_best_success_rate"] for surface in surfaces]
    best = [summary["by_surface"][surface]["best_of_mode_success_rate"] for surface in surfaces]
    figure, axis = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    axis.bar(x - width / 2, geometry, width, label="Geometry-best", color="#4b5563")
    axis.bar(x + width / 2, best, width, label="Best-of-mode", color="#0f766e")
    axis.set_xticks(x, surfaces)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Continuous-lift success")
    axis.set_title("Fixed-placement UR5e workspace proposal control")
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def similarity_scale(surface, target_extent: float) -> float:
    return target_extent / float(np.max(np.ptp(surface.vertices, axis=0)))


def stable_seed(base: int, instance_id: str) -> int:
    digest = hashlib.sha256(instance_id.encode()).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
