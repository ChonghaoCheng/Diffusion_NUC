#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.coverage import (
    evaluate_coverage,
    load_teacher_instance,
    surface_from_teacher_archive,
)
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.robot.qspace_coverage_teacher import (
    densify_qspace_segment,
    hard_check_qspace_plan,
    resample_qspace_segment,
    simplify_qspace_segment,
    workspace_plan_from_q,
)
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


DEFAULT_MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit q-space teacher token compression")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--teacher-results", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--segments", type=int, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--q-tolerances", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--tau-degrees", type=float, default=10.0)
    parser.add_argument("--check-joint-step", type=float, default=0.05)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {str(row["instance_id"]): row for row in load_manifest(args.dataset)}
    row = manifest[args.instance_id]
    archive = load_teacher_instance(args.dataset / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    transform = load_transform(args.fixed_results, args.instance_id, args.tau_degrees)
    result_path = args.teacher_results / "instances" / f"{args.instance_id}_k{args.segments}.npz"
    with np.load(result_path, allow_pickle=False) as result:
        mask = result["mask"]
        q_segments = tuple(result["q"][index, mask[index]] for index in range(len(mask)))
        position_segments = tuple(
            result["target_positions"][index, mask[index]] for index in range(len(mask))
        )
        axis_segments = tuple(
            result["target_axes"][index, mask[index]] for index in range(len(mask))
        )
    robot = UR5eKinematics(args.model)
    footprint_radius = float(dict(archive["metadata"])["teacher_config"]["footprint_radius"])
    experiments = [
        (
            "original",
            0.0,
            list(zip(q_segments, position_segments, axis_segments)),
        )
    ]
    for tokens in args.tokens:
        experiments.append(("uniform", float(tokens), [
            resample_qspace_segment(q, positions, axes, num_tokens=tokens)
            for q, positions, axes in zip(q_segments, position_segments, axis_segments)
        ]))
    for tolerance in args.q_tolerances:
        experiments.append(("adaptive", tolerance, [
            simplify_qspace_segment(
                q, positions, axes, maximum_joint_error=tolerance
            )
            for q, positions, axes in zip(q_segments, position_segments, axis_segments)
        ]))
    rows = []
    for mode, parameter, resampled in experiments:
        q_new = tuple(item[0] for item in resampled)
        dense = [
            densify_qspace_segment(*item, maximum_joint_step=args.check_joint_step)
            for item in resampled
        ]
        q_checked = tuple(item[0] for item in dense)
        positions_checked = tuple(item[1] for item in dense)
        axes_checked = tuple(item[2] for item in dense)
        check = hard_check_qspace_plan(
            robot,
            q_checked,
            positions_checked,
            axes_checked,
            position_tolerance=0.003,
            axis_tolerance=np.deg2rad(args.tau_degrees),
            minimum_manipulability=1e-5,
            interpolation_joint_step=None,
        )
        metrics = None
        if check.feasible:
            plan = workspace_plan_from_q(robot, q_checked, transform)
            metrics = evaluate_coverage(surface, plan, footprint_radius=footprint_radius)
            surface_scale = float(
                np.mean(np.linalg.svd(transform[:3, :3], compute_uv=False))
            )
            if metrics.max_projection_distance * surface_scale > 0.003:
                check = type(check)(
                    False,
                    "surface_projection",
                    metrics.max_projection_distance * surface_scale,
                    check.max_axis_error,
                    check.minimum_manipulability,
                    check.minimum_joint_limit_margin,
                )
                metrics = None
        original_count = sum(len(q) for q in q_segments)
        resampled_count = sum(len(q) for q in q_new)
        rows.append(
            {
                "mode": mode,
                "tokens_per_segment": int(parameter) if mode == "uniform" else None,
                "maximum_joint_error": parameter if mode == "adaptive" else None,
                "original_samples": original_count,
                "resampled_samples": resampled_count,
                "checked_samples": int(sum(len(q) for q in q_checked)),
                "compression_ratio": original_count / resampled_count,
                "feasible": check.feasible,
                "failure_reason": check.failure_reason,
                "max_position_error": check.max_position_error,
                "max_axis_error_degrees": float(np.rad2deg(check.max_axis_error)),
                "missed_fraction": None if metrics is None else metrics.missed_fraction,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


def load_transform(path: Path, instance_id: str, tau_degrees: float) -> np.ndarray:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["instance_id"] == instance_id and np.isclose(
                float(row["tau_degrees"]), tau_degrees
            ):
                return np.asarray(json.loads(row["transform_base_from_surface"]), dtype=np.float64)
    raise KeyError(f"missing fixed transform for {instance_id}")


if __name__ == "__main__":
    main()
