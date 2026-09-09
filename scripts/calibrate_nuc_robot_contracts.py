#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.coverage import CoveragePlan, evaluate_nuc_coverage
from diffusion_coverage.nuc import generate_nuc_skeleton
from diffusion_coverage.robot.qspace_coverage_teacher import densify_qspace_segment
from diffusion_coverage.robot.surface_ik_graph import SurfaceIKGraph, load_surface_ik_graph, save_surface_ik_graph
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d, task_singularity_metrics
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.surface import make_hemisphere, make_saddle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze NUC/robot evaluation contracts before E06")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nuc_robot_coupling_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nuc_robot_contract_calibration_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    robot_cfg = config["robot"]
    coverage_cfg = config["coverage"]
    robot = UR5eKinematics(
        robot_cfg["model"],
        site_name=robot_cfg["site_name"],
        tool_axis_index=robot_cfg["tool_axis_index"],
        tool_axis_sign=robot_cfg["tool_axis_sign"],
    )
    rows: list[dict[str, object]] = []
    coverage_rows, delta_eval = calibrate_coverage(config)
    rows.extend(coverage_rows)
    q_rows, sigma_safe, selected_q_step = calibrate_q_sampling(robot, config)
    rows.extend(q_rows)
    consistency = calibrate_task_metric(robot, config)
    rows.extend(consistency["rows"])
    witness = calibrate_witness_roundtrip(robot, args.output)
    rows.append(witness)
    delta_nuc = 2.0 * delta_eval
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    frozen = {
        **config,
        "branch": "exp/nuc-robot-coupling-v1",
        "code_commit": code_commit,
        "frozen_contract": {
            "coverage_samples_per_face": coverage_cfg["normal_samples_per_face"],
            "coverage_path_sample_spacing_m": coverage_cfg["normal_path_sample_spacing_m"],
            "q_interpolation_step_rad": selected_q_step,
            "characteristic_length_m": robot_cfg["characteristic_length_m"],
            "sigma_safe": sigma_safe,
            "delta_eval": delta_eval,
            "delta_NUC": delta_nuc,
            "timing_enabled": False,
        },
    }
    summary = {
        "branch": frozen["branch"],
        "code_commit": code_commit,
        "result_directory": str(args.output.relative_to(ROOT)),
        "frozen_contract": frozen["frozen_contract"],
        "task_metric_consistency": consistency["summary"],
        "witness_roundtrip": witness,
        "coverage_resolution_rows": len(coverage_rows),
        "q_resolution_rows": len(q_rows),
    }
    (args.output / "config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output / "raw.csv", rows)
    plot_coverage(
        coverage_rows,
        args.output / "coverage_resolution.png",
        normal_samples_per_face=coverage_cfg["normal_samples_per_face"],
        reference_path_spacing=coverage_cfg["reference_path_sample_spacing_m"],
    )
    plot_q(q_rows, args.output / "q_resolution.png")
    (args.output / "README.md").write_text(
        "# NUC robot contract calibration v1\n\n"
        f"Code commit: `{code_commit}`  \n"
        "Branch: `exp/nuc-robot-coupling-v1`\n\n"
        "This pre-method calibration freezes temporal NUC evaluation resolution, q-space "
        "interpolation density, normalized 5D singularity admission, and NUC equivalence "
        "tolerance. It does not compare skeleton-selection methods. The collision check is "
        "limited to contacts represented by the loaded standalone UR5e Menagerie model.\n"
    )
    print(json.dumps(summary, indent=2))


def make_surface(config: dict, surface_id: str, samples_per_face: int):
    values = config["surfaces"][surface_id]
    if surface_id == "saddle":
        return make_saddle(**values, samples_per_face=samples_per_face)
    if surface_id == "hemisphere":
        return make_hemisphere(**values, samples_per_face=samples_per_face)
    raise ValueError(surface_id)


def calibrate_coverage(config: dict) -> tuple[list[dict[str, object]], float]:
    calibration = config["calibration"]
    coverage = config["coverage"]
    rows = []
    comparisons = []
    for surface_id in config["surfaces"]:
        base = make_surface(config, surface_id, coverage["normal_samples_per_face"])
        skeleton = generate_nuc_skeleton(base, policy="upstream_first")
        plan = CoveragePlan(skeleton.waypoints)
        metrics_by_resolution = {}
        for samples_per_face in calibration["coverage_samples_per_face"]:
            surface = make_surface(config, surface_id, int(samples_per_face))
            for spacing in calibration["coverage_path_spacings_m"]:
                metrics = evaluate_nuc_coverage(
                    surface,
                    plan,
                    footprint_radius=coverage["footprint_radius_m"],
                    path_sample_spacing=float(spacing),
                )
                key = (int(samples_per_face), float(spacing))
                metrics_by_resolution[key] = metrics
                rows.append({
                    "kind": "coverage_resolution",
                    "surface_id": surface_id,
                    "samples_per_face": samples_per_face,
                    "path_sample_spacing": spacing,
                    "E_miss": metrics.missed_error,
                    "E_rep": metrics.repeat_error,
                    "E_NUC": metrics.nuc_error,
                    "path_length": metrics.path_length,
                })
        normal = metrics_by_resolution[(coverage["normal_samples_per_face"], coverage["normal_path_sample_spacing_m"])]
        reference = metrics_by_resolution[(coverage["reference_samples_per_face"], coverage["reference_path_sample_spacing_m"])]
        comparisons.append(abs(normal.nuc_error - reference.nuc_error))
    return rows, float(max(comparisons))


def calibrate_q_sampling(robot: UR5eKinematics, config: dict) -> tuple[list[dict[str, object]], float, float]:
    characteristic_length = config["robot"]["characteristic_length_m"]
    t = np.linspace(0.0, 1.0, 9)
    amplitudes = np.asarray([0.18, -0.14, 0.12, 0.10, -0.08, 0.16])
    q = robot.home[None, :] + np.sin(np.pi * t)[:, None] * amplitudes[None, :]
    poses = [robot.forward(value) for value in q]
    positions = np.asarray([pose[0] for pose in poses])
    axes = np.asarray([pose[1] for pose in poses])
    rows = []
    all_sigmas = []
    for step in config["calibration"]["q_interpolation_steps_rad"]:
        q_dense, positions_dense, axes_dense = densify_qspace_segment(
            q, positions, axes, maximum_joint_step=float(step)
        )
        position_errors = []
        axis_errors = []
        sigmas = []
        for q_value, desired_position, desired_axis in zip(q_dense, positions_dense, axes_dense):
            task = evaluate_task_kinematics_5d(
                robot, q_value, characteristic_length=characteristic_length
            )
            position_errors.append(np.linalg.norm(task.position - desired_position))
            axis_errors.append(np.arccos(np.clip(task.tool_axis @ desired_axis, -1.0, 1.0)))
            sigmas.append(task.sigma_min_5)
        all_sigmas.extend(sigmas)
        rows.append({
            "kind": "q_resolution",
            "q_interpolation_step": step,
            "q_samples": len(q_dense),
            "max_position_error": max(position_errors),
            "max_axis_error": max(axis_errors),
            "min_sigma_min_5": min(sigmas),
        })
    reference = rows[-1]
    tolerances = config["calibration"]
    stable = [
        row for row in rows
        if abs(float(row["max_position_error"]) - float(reference["max_position_error"])) <= tolerances["q_position_stability_m"]
        and abs(float(row["max_axis_error"]) - float(reference["max_axis_error"])) <= tolerances["q_axis_stability_rad"]
        and abs(float(row["min_sigma_min_5"]) - float(reference["min_sigma_min_5"])) <= tolerances["q_sigma_stability"]
    ]
    selected_step = float(max(float(row["q_interpolation_step"]) for row in stable))
    sigma_safe = float(max(0.02, 0.10 * np.median(all_sigmas)))
    return rows, sigma_safe, selected_step


def calibrate_task_metric(robot: UR5eKinematics, config: dict) -> dict[str, object]:
    length = config["robot"]["characteristic_length_m"]
    q = robot.home + np.asarray([0.11, -0.09, 0.07, 0.05, -0.04, 0.03])
    task = evaluate_task_kinematics_5d(robot, q, characteristic_length=length)
    epsilon = 1e-6
    position_fd = np.column_stack([
        (robot.forward(q + epsilon * np.eye(6)[j])[0] - robot.forward(q - epsilon * np.eye(6)[j])[0]) / (2 * epsilon)
        for j in range(6)
    ])
    axis_fd = np.column_stack([
        (robot.forward(q + epsilon * np.eye(6)[j])[1] - robot.forward(q - epsilon * np.eye(6)[j])[1]) / (2 * epsilon)
        for j in range(6)
    ])
    u, v = task.axis_basis.T
    angular_fd = np.vstack((-v @ axis_fd, u @ axis_fd))
    angle = 0.41
    plane_rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = task_singularity_metrics(
        task.jacobian_5[:3],
        task.axis_basis @ task.jacobian_5[3:],
        task.tool_axis,
        characteristic_length=length,
        axis_basis=task.axis_basis @ plane_rotation,
    )[2]
    scaled = task_singularity_metrics(
        1000.0 * task.jacobian_5[:3],
        task.axis_basis @ task.jacobian_5[3:],
        task.tool_axis,
        characteristic_length=1000.0 * length,
    )[2]
    summary = {
        "position_finite_difference_max_error": float(np.max(np.abs(task.jacobian_5[:3] - position_fd))),
        "axis_finite_difference_max_error": float(np.max(np.abs(task.jacobian_5[3:] - angular_fd))),
        "basis_rotation_singular_value_max_error": float(np.max(np.abs(task.singular_values - rotated))),
        "unit_scaling_singular_value_max_error": float(np.max(np.abs(task.singular_values - scaled))),
    }
    return {
        "summary": summary,
        "rows": [{"kind": "task_metric_consistency", **summary}],
    }


def calibrate_witness_roundtrip(robot: UR5eKinematics, output: Path) -> dict[str, object]:
    first = robot.evaluate_configuration(robot.home)
    second = robot.evaluate_configuration(robot.home + np.asarray([0.02, 0, 0, 0, 0, 0]))
    witness = np.linspace(first.q, second.q, 7)
    graph = SurfaceIKGraph(
        uv=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        positions=np.asarray([robot.forward(first.q)[0], robot.forward(second.q)[0]]),
        axes=np.asarray([robot.forward(first.q)[1], robot.forward(second.q)[1]]),
        candidates=((first,), (second,)),
        edge_index=np.asarray([[0], [1]]),
        edge_compatibility=(np.ones((1, 1), dtype=bool),),
        component_labels=(np.asarray([0]), np.asarray([0])),
        grid_shape=(2, 1),
        edge_witnesses=({(0, 0): witness},),
        metadata={"calibration": True},
    )
    path = output / "witness_roundtrip.npz"
    save_surface_ik_graph(path, graph)
    restored = load_surface_ik_graph(path)
    exact = np.array_equal(restored.edge_witnesses[0][(0, 0)], witness)
    return {
        "kind": "witness_roundtrip",
        "float64": restored.edge_witnesses[0][(0, 0)].dtype == np.float64,
        "exact": bool(exact),
        "samples": len(witness),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_coverage(
    rows: list[dict[str, object]],
    path: Path,
    *,
    normal_samples_per_face: int,
    reference_path_spacing: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for surface in sorted({str(row["surface_id"]) for row in rows}):
        selected = [
            row for row in rows
            if row["surface_id"] == surface
            and int(row["samples_per_face"]) == normal_samples_per_face
        ]
        selected.sort(key=lambda row: float(row["path_sample_spacing"]), reverse=True)
        axes[0].plot([row["path_sample_spacing"] for row in selected], [row["E_NUC"] for row in selected], marker="o", label=surface)
        selected = [
            row for row in rows
            if row["surface_id"] == surface
            and np.isclose(float(row["path_sample_spacing"]), reference_path_spacing)
        ]
        selected.sort(key=lambda row: int(row["samples_per_face"]))
        axes[1].plot([row["samples_per_face"] for row in selected], [row["E_NUC"] for row in selected], marker="o", label=surface)
    axes[0].set_xlabel("Path sample spacing [m]")
    axes[0].set_ylabel("E_NUC")
    axes[1].set_xlabel("Surface samples per face")
    axes[1].set_ylabel("E_NUC")
    for axis in axes: axis.grid(True, alpha=0.3); axis.legend()
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def plot_q(rows: list[dict[str, object]], path: Path) -> None:
    steps = [row["q_interpolation_step"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, key, label in zip(
        axes,
        ("max_position_error", "max_axis_error", "min_sigma_min_5"),
        ("Max position error [m]", "Max axis error [rad]", "Min sigma_min_5"),
    ):
        axis.plot(steps, [row[key] for row in rows], marker="o")
        axis.set_xscale("log"); axis.set_xlabel("Max q interpolation step [rad]"); axis.set_ylabel(label); axis.grid(True, alpha=0.3)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
