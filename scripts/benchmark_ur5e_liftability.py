#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.robot.ur5e_mujoco import (
    UR5eKinematics,
    interpolate_vertex_normals,
    transform_surface_pose_path,
)
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


DEFAULT_MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-FM UR5e workspace-first liftability audit")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--pose-samples", type=int, default=None)
    parser.add_argument("--pose-spacing", type=float, default=0.025)
    parser.add_argument("--max-pose-samples", type=int, default=256)
    parser.add_argument("--tau-degrees", nargs="+", type=float, default=[5.0])
    parser.add_argument("--random-restarts", type=int, default=8)
    parser.add_argument("--target-extent", type=float, default=0.32)
    parser.add_argument("--center", nargs=3, type=float, default=[0.0, 0.58, 0.42])
    parser.add_argument("--translation-jitter", type=float, default=0.035)
    parser.add_argument("--tilt-degrees", type=float, default=25.0)
    parser.add_argument("--maximum-joint-step", type=float, default=0.8)
    parser.add_argument("--max-active-branches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--calibration-placements", type=int, default=6)
    parser.add_argument("--calibration-pose-samples", type=int, default=8)
    parser.add_argument(
        "--fixed-transforms", type=Path, default=None,
        help="Raw-results CSV whose per-instance base transforms are reused",
    )
    parser.add_argument("--motivation-lift-threshold", type=float, default=0.8)
    parser.add_argument("--motivation-continuity-threshold", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.pose_samples is not None and args.pose_samples < 2) or args.pose_spacing <= 0.0 or args.target_extent <= 0.0:
        raise ValueError("pose samples and target extent must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    robot = UR5eKinematics(args.model)
    manifest = load_manifest(args.dataset)
    if args.max_instances is not None:
        manifest = manifest[: args.max_instances]
    records: list[dict[str, object]] = []
    fixed_transforms = load_fixed_transforms(args.fixed_transforms)
    for instance_index, row in enumerate(manifest):
        archive = load_teacher_instance(args.dataset / str(row["path"]))
        metadata = archive["metadata"]
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        candidate_metrics = np.asarray(archive["candidate_metrics"])
        tolerance = float(metadata["teacher_config"]["missed_tolerance"])
        candidate_index = min(
            range(len(candidate_metrics)),
            key=lambda index: constrained_coverage_key(
                float(candidate_metrics[index, 0]), float(candidate_metrics[index, 1]), tolerance
            ),
        )
        plan = CoveragePlan(
            archive["candidate_waypoints"][candidate_index],
            archive["candidate_segment_mask"][candidate_index],
            archive["candidate_waypoint_mask"][candidate_index],
        )
        physical_path_length = float(candidate_metrics[candidate_index, 1]) * similarity_scale(surface, args.target_extent)
        pose_samples = args.pose_samples or min(
            args.max_pose_samples, max(2, int(np.ceil(physical_path_length / args.pose_spacing)) + 1)
        )
        path = resample_surface_path(surface, plan.active_paths()[0], num_waypoints=pose_samples)
        projection = project_points(surface, path)
        normals = interpolate_vertex_normals(
            surface.vertices, surface.faces, surface.face_normals, surface.face_areas,
            projection.face_indices, projection.barycentric,
        )
        rng = np.random.default_rng(args.seed + instance_index)
        fixed = fixed_transforms.get(str(row["instance_id"]))
        if fixed is None:
            transform, calibration_fraction = calibrate_base_placement(
                robot, surface, path, normals, rng,
                num_placements=args.calibration_placements,
                calibration_pose_samples=args.calibration_pose_samples,
                random_restarts=max(2, args.random_restarts // 2),
                target_extent=args.target_extent, center=np.asarray(args.center),
                translation_jitter=args.translation_jitter, tilt_degrees=args.tilt_degrees,
            )
        else:
            transform, calibration_fraction = fixed
        positions, axes = transform_surface_pose_path(path, normals, transform)
        previous_success = None
        previous_success_tau = None
        for tau_degrees in sorted(args.tau_degrees):
            inherited = previous_success is not None
            start = perf_counter()
            if inherited:
                # Feasibility is monotone in orientation tolerance: a path
                # satisfying a tighter cone is a certificate for every wider cone.
                result = previous_success
            else:
                result = robot.check_continuous_lift(
                    positions,
                    axes,
                    axis_tolerance=np.deg2rad(tau_degrees),
                    random_restarts=args.random_restarts,
                    maximum_joint_step=args.maximum_joint_step,
                    max_active_branches=args.max_active_branches,
                    rng=np.random.default_rng(args.seed + 100003 * instance_index),
                )
                if result.feasible:
                    previous_success = result
                    previous_success_tau = tau_degrees
            record: dict[str, object] = {
                "instance_id": str(row["instance_id"]),
                "surface_id": str(row["surface_id"]),
                "tau_degrees": tau_degrees,
                "liftable": result.feasible,
                "failure_reason": result.failure_reason,
                "failed_waypoint": result.metadata.get("failed_waypoint"),
                "minimum_joint_jump": result.metadata.get("minimum_joint_jump"),
                "rejected_joint_step": result.metadata.get("rejected_joint_step", 0),
                "rejected_collision": result.metadata.get("rejected_collision", 0),
                "rejected_singularity": result.metadata.get("rejected_singularity", 0),
                "pose_samples": pose_samples,
                "physical_path_length": physical_path_length,
                "calibration_posewise_ik_fraction": calibration_fraction,
                "min_candidates": min(result.candidate_counts) if result.candidate_counts else 0,
                "mean_candidates": float(np.mean(result.candidate_counts)) if result.candidate_counts else 0.0,
                "search_edges": result.search_edges,
                "min_manipulability": result.min_manipulability,
                "min_joint_limit_margin": result.min_joint_limit_margin,
                "solve_time": perf_counter() - start,
                "solver_invoked": not inherited,
                "inherited_from_tau": previous_success_tau if inherited else None,
                "surface_coverage_feasible": bool(candidate_metrics[candidate_index, 0] <= float(metadata["teacher_config"]["missed_tolerance"]) + 1e-12),
                "missed_fraction": float(candidate_metrics[candidate_index, 0]),
                "object_scale": similarity_scale(surface, args.target_extent),
                "transform_base_from_surface": transform.tolist(),
            }
            records.append(record)
            print(
                f"{record['instance_id']:<24} tau={tau_degrees:4.1f} "
                f"lift={str(result.feasible):<5} reason={str(result.failure_reason):<32} "
                f"time={record['solve_time']:.2f}s",
                flush=True,
            )
        write_csv(args.output / "raw_results.csv", records)
    write_csv(args.output / "raw_results.csv", records)
    summary = summarize(records, args)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_failures(summary, args.output / "liftability_failures.png")
    print(json.dumps(summary, indent=2))


def similarity_scale(surface: SurfaceInstance, target_extent: float) -> float:
    return target_extent / float(np.max(np.ptp(surface.vertices, axis=0)))


def sample_base_placement(
    surface: SurfaceInstance,
    rng: np.random.Generator,
    *,
    target_extent: float,
    center: np.ndarray,
    translation_jitter: float,
    tilt_degrees: float,
) -> np.ndarray:
    scale = similarity_scale(surface, target_extent)
    yaw = rng.uniform(-np.pi, np.pi)
    tilt_x, tilt_y = rng.uniform(-np.deg2rad(tilt_degrees), np.deg2rad(tilt_degrees), size=2)
    rotation = rotation_z(yaw) @ rotation_y(tilt_y) @ rotation_x(tilt_x)
    linear = scale * rotation
    object_center = 0.5 * (surface.vertices.min(axis=0) + surface.vertices.max(axis=0))
    target_center = center + rng.uniform(-translation_jitter, translation_jitter, size=3)
    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = target_center - linear @ object_center
    return transform


def calibrate_base_placement(
    robot: UR5eKinematics,
    surface: SurfaceInstance,
    path: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
    *,
    num_placements: int,
    calibration_pose_samples: int,
    random_restarts: int,
    target_extent: float,
    center: np.ndarray,
    translation_jitter: float,
    tilt_degrees: float,
) -> tuple[np.ndarray, float]:
    if num_placements < 1:
        raise ValueError("num_placements must be positive")
    indices = np.linspace(0, len(path) - 1, min(calibration_pose_samples, len(path)), dtype=int)
    best_transform: np.ndarray | None = None
    best_fraction = -1.0
    for _ in range(num_placements):
        transform = sample_base_placement(
            surface, rng, target_extent=target_extent, center=center,
            translation_jitter=translation_jitter, tilt_degrees=tilt_degrees,
        )
        positions, axes = transform_surface_pose_path(path[indices], normals[indices], transform)
        feasible = 0
        for position, axis in zip(positions, axes):
            candidates = robot.enumerate_ik(
                position, axis, random_restarts=random_restarts, rng=rng,
                axis_tolerance=np.deg2rad(5.0), max_candidates=1,
            )
            feasible += bool(candidates)
        fraction = feasible / len(indices)
        if fraction > best_fraction:
            best_fraction = fraction
            best_transform = transform
        if fraction == 1.0:
            break
    assert best_transform is not None
    return best_transform, best_fraction


def rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)), dtype=float)


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)), dtype=float)


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)), dtype=float)


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    flattened = [{**record, "transform_base_from_surface": json.dumps(record["transform_base_from_surface"])} for record in records]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def load_fixed_transforms(path: Path | None) -> dict[str, tuple[np.ndarray, float]]:
    if path is None:
        return {}
    with path.open(newline="") as source:
        rows = csv.DictReader(source)
        return {
            str(row["instance_id"]): (
                np.asarray(json.loads(row["transform_base_from_surface"]), dtype=float),
                float(row["calibration_posewise_ik_fraction"]),
            )
            for row in rows
        }


def summarize(records: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    by_tau: dict[str, object] = {}
    for tau in sorted({float(record["tau_degrees"]) for record in records}):
        selected = [record for record in records if float(record["tau_degrees"]) == tau]
        failure_counts: dict[str, int] = {}
        for record in selected:
            reason = "success" if record["liftable"] else str(record["failure_reason"])
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        lift_rate = float(np.mean([bool(record["liftable"]) for record in selected]))
        continuity_rate = failure_counts.get("continuous_ik_search_failure", 0) / len(selected)
        by_tau[str(tau)] = {
            "instances": len(selected),
            "lift_success_rate": lift_rate,
            "failure_counts": failure_counts,
            "mean_solve_time": float(np.mean([record["solve_time"] for record in selected])),
            "motivation_gate": {
                "workspace_first_gap_material": lift_rate < args.motivation_lift_threshold,
                "continuity_failure_material": continuity_rate >= args.motivation_continuity_threshold,
                "passed": lift_rate < args.motivation_lift_threshold and continuity_rate >= args.motivation_continuity_threshold,
            },
        }
    return {
        "model": str(args.model),
        "collision_scope": "UR5e self-collision only; workpiece collision not yet modeled",
        "continuation_scope": "finite-beam numerical search; failures are not topology certificates",
        "max_active_branches": args.max_active_branches,
        "by_tau_degrees": by_tau,
    }


def plot_failures(summary: dict[str, object], output: Path) -> None:
    by_tau = summary["by_tau_degrees"]
    labels = list(by_tau)
    reasons = sorted({reason for result in by_tau.values() for reason in result["failure_counts"]})
    bottom = np.zeros(len(labels))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for reason in reasons:
        values = np.asarray([
            by_tau[label]["failure_counts"].get(reason, 0) / by_tau[label]["instances"]
            for label in labels
        ])
        axis.bar(labels, values, bottom=bottom, label=reason)
        bottom += values
    axis.set(xlabel="Orientation tolerance (degrees)", ylabel="Fraction of plans", ylim=(0, 1))
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
