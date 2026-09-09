#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

import numpy as np

from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics, transform_surface_pose_path
from diffusion_coverage.surface import make_hemisphere, make_saddle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze neutral E06 robot/workpiece placements")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nuc_robot_coupling_v1.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "results/nuc_robot_contract_calibration_v1/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/nuc_robot_coupling_v1_placements.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    calibration = json.loads(args.calibration.read_text())
    placement = config["placement_calibration"]
    robot_cfg = config["robot"]
    robot = UR5eKinematics(
        robot_cfg["model"],
        site_name=robot_cfg["site_name"],
        tool_axis_index=robot_cfg["tool_axis_index"],
        tool_axis_sign=robot_cfg["tool_axis_sign"],
    )
    sigma_safe = float(calibration["frozen_contract"]["sigma_safe"])
    candidate_specs = make_candidate_specs(placement, int(config["seed"]))
    result = {
        "experiment": config["experiment"],
        "branch": "exp/nuc-robot-coupling-v1",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "random_seed": int(config["seed"]),
        "selection_rule": placement["selection_rule"],
        "neutral_trajectory": "evenly spaced triangle centroids with no skeleton ordering",
        "sigma_safe": sigma_safe,
        "candidate_specs": candidate_specs,
        "surfaces": {},
    }
    for surface_id in config["surfaces"]:
        surface = make_surface(config, surface_id)
        points, normals = neutral_surface_samples(surface, int(placement["neutral_surface_samples"]))
        rows = []
        for index, spec in enumerate(candidate_specs):
            transform = rigid_surface_transform(surface, spec)
            start = perf_counter()
            score = score_transform(
                robot,
                points,
                normals,
                transform,
                sigma_safe=sigma_safe,
                characteristic_length=float(robot_cfg["characteristic_length_m"]),
                axis_tolerance=np.deg2rad(float(robot_cfg["axis_tolerance_degrees"])),
                random_restarts=int(placement["random_restarts"]),
                max_candidates=int(placement["max_candidates"]),
                orientation_cone_samples=int(placement["orientation_cone_samples"]),
                seed=int(config["seed"]) + 1009 * index,
            )
            rows.append({
                "candidate_id": f"T{index:02d}",
                "transform_base_from_surface": transform.tolist(),
                "elapsed_seconds": perf_counter() - start,
                **score,
            })
            print(surface_id, rows[-1]["candidate_id"], f"reach={score['reachable_fraction']:.3f}", flush=True)
        selected = select_placements(rows, float(placement["minimum_reachable_fraction"]))
        result["surfaces"][surface_id] = {
            "candidate_statistics": rows,
            "selected": selected,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({surface: value["selected"] for surface, value in result["surfaces"].items()}, indent=2))


def make_surface(config: dict, surface_id: str):
    values = config["surfaces"][surface_id]
    maker = make_saddle if surface_id == "saddle" else make_hemisphere
    return maker(**values, samples_per_face=1)


def make_candidate_specs(config: dict, seed: int) -> list[dict[str, object]]:
    count = int(config["num_candidate_transforms"])
    rng = np.random.default_rng(seed)
    center = np.asarray(config["nominal_target_center_m"], dtype=float)
    jitter = np.asarray(config["translation_jitter_m"], dtype=float)
    specs = [{"target_center_m": center.tolist(), "roll_degrees": 0.0, "pitch_degrees": 0.0, "yaw_degrees": 0.0}]
    for _ in range(count - 1):
        specs.append({
            "target_center_m": (center + rng.uniform(-jitter, jitter)).tolist(),
            "roll_degrees": float(rng.uniform(-config["maximum_tilt_degrees"], config["maximum_tilt_degrees"])),
            "pitch_degrees": float(rng.uniform(-config["maximum_tilt_degrees"], config["maximum_tilt_degrees"])),
            "yaw_degrees": float(rng.uniform(-config["maximum_yaw_degrees"], config["maximum_yaw_degrees"])),
        })
    return specs


def rigid_surface_transform(surface, spec: dict[str, object]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([
        spec["roll_degrees"], spec["pitch_degrees"], spec["yaw_degrees"]
    ])
    rotation = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    object_center = 0.5 * (surface.vertices.min(axis=0) + surface.vertices.max(axis=0))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(spec["target_center_m"], dtype=float) - rotation @ object_center
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12):
        raise AssertionError("placement transform is not rigid")
    return transform


def neutral_surface_samples(surface, count: int) -> tuple[np.ndarray, np.ndarray]:
    face_indices = np.linspace(0, surface.num_faces - 1, min(count, surface.num_faces), dtype=int)
    triangles = surface.vertices[surface.faces[face_indices]]
    return triangles.mean(axis=1), surface.face_normals[face_indices]


def score_transform(
    robot: UR5eKinematics,
    points: np.ndarray,
    normals: np.ndarray,
    transform: np.ndarray,
    *,
    sigma_safe: float,
    characteristic_length: float,
    axis_tolerance: float,
    random_restarts: int,
    max_candidates: int,
    orientation_cone_samples: int,
    seed: int,
) -> dict[str, float]:
    positions, axes = transform_surface_pose_path(points, normals, transform)
    rng = np.random.default_rng(seed)
    safe_counts = []
    best_sigmas = []
    best_margins = []
    for position, axis in zip(positions, axes):
        candidates = robot.enumerate_ik(
            position,
            axis,
            random_restarts=random_restarts,
            rng=rng,
            axis_tolerance=axis_tolerance,
            minimum_manipulability=0.0,
            max_candidates=max_candidates,
            orientation_cone_samples=orientation_cone_samples,
        )
        metrics = [
            (evaluate_task_kinematics_5d(robot, candidate.q, characteristic_length=characteristic_length).sigma_min_5, candidate.joint_limit_margin)
            for candidate in candidates
        ]
        safe = [value for value in metrics if value[0] >= sigma_safe]
        safe_counts.append(len(safe))
        if safe:
            best = max(safe)
            best_sigmas.append(best[0])
            best_margins.append(best[1])
    reachable = np.asarray(safe_counts) > 0
    return {
        "reachable_fraction": float(np.mean(reachable)),
        "mean_safe_branches": float(np.mean(safe_counts)),
        "median_best_sigma_min_5": float(np.median(best_sigmas)) if best_sigmas else 0.0,
        "median_best_joint_limit_margin": float(np.median(best_margins)) if best_margins else 0.0,
        "neutral_samples": int(len(points)),
    }


def placement_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["reachable_fraction"]),
        float(row["median_best_sigma_min_5"]),
        float(row["mean_safe_branches"]),
        float(row["median_best_joint_limit_margin"]),
    )


def select_placements(rows: list[dict[str, object]], minimum_reachable_fraction: float) -> dict[str, dict[str, object]]:
    eligible = sorted(
        [row for row in rows if float(row["reachable_fraction"]) >= minimum_reachable_fraction],
        key=placement_key,
    )
    if len(eligible) < 3:
        raise RuntimeError(f"only {len(eligible)} placement candidates meet neutral reachability")
    choices = {"P_hard": eligible[0], "P_mid": eligible[len(eligible) // 2], "P_easy": eligible[-1]}
    return {
        name: {
            "candidate_id": row["candidate_id"],
            "transform_base_from_surface": row["transform_base_from_surface"],
            "calibration_statistics": {key: row[key] for key in (
                "reachable_fraction", "mean_safe_branches", "median_best_sigma_min_5", "median_best_joint_limit_margin"
            )},
        }
        for name, row in choices.items()
    }


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)))


def rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)))


if __name__ == "__main__":
    main()
