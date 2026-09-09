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

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.coverage import load_teacher_instance, surface_from_teacher_archive
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


DEFAULT_MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a strict UR5e q-space coverage teacher")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--teacher-results", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--segments", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tau-degrees", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {str(row["instance_id"]): row for row in load_manifest(args.dataset)}
    row = manifest[args.instance_id]
    archive = load_teacher_instance(args.dataset / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    transform = load_transform(args.fixed_results, args.instance_id, args.tau_degrees)
    inverse = np.linalg.inv(transform)
    result_path = args.teacher_results / "instances" / f"{args.instance_id}_k{args.segments}.npz"
    with np.load(result_path, allow_pickle=False) as result:
        q = result["q"]
        mask = result["mask"]
    robot = UR5eKinematics(args.model)
    surface_paths = []
    q_paths = []
    for segment in range(len(q)):
        q_path = np.asarray(q[segment, mask[segment]], dtype=np.float64)
        if not len(q_path):
            continue
        base_positions = np.asarray([robot.forward(value)[0] for value in q_path])
        homogeneous = np.column_stack((base_positions, np.ones(len(base_positions))))
        surface_paths.append((inverse @ homogeneous.T).T[:, :3])
        q_paths.append(q_path)

    metrics = None
    raw_path = args.teacher_results / "instances.jsonl"
    if raw_path.exists():
        metrics = next(
            (
                json.loads(line)
                for line in raw_path.read_text().splitlines()
                if json.loads(line)["instance_id"] == args.instance_id
                and json.loads(line)["max_segments"] == args.segments
            ),
            None,
        )
    fig = plt.figure(figsize=(12.0, 5.0))
    surface_ax = fig.add_subplot(1, 2, 1, projection="3d")
    triangles = surface.vertices[surface.faces]
    surface_ax.plot_trisurf(
        surface.vertices[:, 0],
        surface.vertices[:, 1],
        surface.vertices[:, 2],
        triangles=surface.faces,
        color="#d9dde2",
        alpha=0.35,
        linewidth=0.1,
    )
    colours = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(surface_paths), 1)))
    for index, path in enumerate(surface_paths):
        surface_ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colours[index], linewidth=1.5)
        surface_ax.scatter(*path[0], color=colours[index], s=18)
    surface_ax.set_title(f"{args.instance_id}: actual FK paths")
    surface_ax.set_box_aspect(np.ptp(surface.vertices, axis=0))
    surface_ax.set_xlabel("x")
    surface_ax.set_ylabel("y")
    surface_ax.set_zlabel("z")

    joint_ax = fig.add_subplot(1, 2, 2)
    offset = 0
    for segment, q_path in enumerate(q_paths):
        x = np.arange(len(q_path)) + offset
        for joint in range(6):
            joint_ax.plot(
                x,
                q_path[:, joint],
                linewidth=0.9,
                label=f"q{joint + 1}" if segment == 0 else None,
            )
        offset = int(x[-1]) + 2
        joint_ax.axvline(offset - 1, color="#999999", linewidth=0.6)
    joint_ax.set_xlabel("trajectory sample")
    joint_ax.set_ylabel("joint position [rad]")
    joint_ax.set_title("Configuration-space trajectory")
    joint_ax.grid(alpha=0.2)
    joint_ax.legend(ncol=3, fontsize=8)
    if metrics is not None:
        fig.suptitle(
            f"k={args.segments}, missed={metrics['missed_fraction']}, "
            f"joint travel={metrics['joint_travel']}",
            fontsize=10,
        )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


def load_transform(path: Path, instance_id: str, tau_degrees: float) -> np.ndarray:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["instance_id"] == instance_id and abs(float(row["tau_degrees"]) - tau_degrees) <= 1e-9:
                return np.asarray(json.loads(row["transform_base_from_surface"]), dtype=np.float64)
    raise KeyError(f"missing fixed transform for {instance_id}")


if __name__ == "__main__":
    main()
