#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from diffusion_coverage.evaluation import control_energy_distance, control_rms_distance
from diffusion_coverage.learning import TeacherPathDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare generated-to-teacher and teacher self-coverage in residual space"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15, 0.25, 0.50, 1.00),
    )
    return parser.parse_args()


def nearest_distances(
    queries: list[np.ndarray], references: list[np.ndarray]
) -> np.ndarray:
    if not queries or not references:
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        [min(control_rms_distance(query, ref) for ref in references) for query in queries],
        dtype=np.float64,
    )


def leave_one_out_distances(controls: list[np.ndarray]) -> np.ndarray:
    if len(controls) < 2:
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        [
            min(
                control_rms_distance(control, other)
                for other_index, other in enumerate(controls)
                if other_index != index
            )
            for index, control in enumerate(controls)
        ],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    if any(threshold <= 0.0 for threshold in args.thresholds):
        raise ValueError("thresholds must be positive")

    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    dataset = TeacherPathDataset(
        args.dataset,
        instance_ids=checkpoint["validation_instance_ids"],
        num_surface_points=int(checkpoint["num_surface_points"]),
        num_path_waypoints=None,
        minimum_path_tokens=int(checkpoint["minimum_path_tokens"]),
        maximum_path_tokens=int(checkpoint["maximum_path_tokens"]),
        candidate_policy="all",
        allow_repaired_candidates=False,
        path_coordinate_system="analytic_uv_structured_residual",
        include_mode_conditioning=True,
        preserve_source_waypoints=True,
        seed=int(checkpoint.get("data_seed", checkpoint["seed"])),
    )

    teacher_groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for index, reference in enumerate(dataset.sample_index):
        row = dataset.rows[reference.instance_index]
        sample = dataset[index]
        key = (str(row["instance_id"]), str(sample["mode_name"]))
        teacher_groups[key].append(sample["path"].numpy().astype(np.float64))

    generated_groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    candidate_path = args.run / "eval_validation_k8" / "candidates.jsonl"
    for line in candidate_path.read_text().splitlines():
        row = json.loads(line)
        if row["feasible"]:
            key = (str(row["instance_id"]), str(row["mode"]))
            generated_groups[key].append(np.asarray(row["residual_controls"], dtype=np.float64))

    mode_rows = []
    for key, teachers in teacher_groups.items():
        generated = generated_groups.get(key, [])
        generated_to_teacher = nearest_distances(teachers, generated)
        teacher_self = leave_one_out_distances(teachers)
        mode_rows.append(
            {
                "instance_id": key[0],
                "mode": key[1],
                "num_generated": len(generated),
                "num_teachers": len(teachers),
                "mean_teacher_to_generated_distance": (
                    float(np.mean(generated_to_teacher)) if generated_to_teacher.size else None
                ),
                "mean_teacher_leave_one_out_distance": (
                    float(np.mean(teacher_self)) if teacher_self.size else None
                ),
                "energy_distance": control_energy_distance(generated, teachers),
                "thresholds": {
                    str(threshold): {
                        "generated_teacher_coverage": (
                            float(np.mean(generated_to_teacher <= threshold))
                            if generated_to_teacher.size
                            else 0.0
                        ),
                        "teacher_leave_one_out_coverage": (
                            float(np.mean(teacher_self <= threshold))
                            if teacher_self.size
                            else None
                        ),
                    }
                    for threshold in args.thresholds
                },
            }
        )

    threshold_summary = []
    for threshold in args.thresholds:
        key = str(threshold)
        generated_coverage = [
            row["thresholds"][key]["generated_teacher_coverage"] for row in mode_rows
        ]
        teacher_coverage = [
            row["thresholds"][key]["teacher_leave_one_out_coverage"]
            for row in mode_rows
            if row["thresholds"][key]["teacher_leave_one_out_coverage"] is not None
        ]
        threshold_summary.append(
            {
                "threshold_radius": threshold,
                "mean_generated_teacher_coverage": float(np.mean(generated_coverage)),
                "mean_teacher_leave_one_out_coverage": (
                    float(np.mean(teacher_coverage)) if teacher_coverage else None
                ),
            }
        )

    teacher_to_generated = [
        row["mean_teacher_to_generated_distance"]
        for row in mode_rows
        if row["mean_teacher_to_generated_distance"] is not None
    ]
    teacher_self = [
        row["mean_teacher_leave_one_out_distance"]
        for row in mode_rows
        if row["mean_teacher_leave_one_out_distance"] is not None
    ]
    energy = [row["energy_distance"] for row in mode_rows if row["energy_distance"] is not None]
    document = {
        "protocol": {
            "distance": "RMS residual-control distance in footprint-radius units",
            "candidate_source": str(candidate_path),
            "thresholds": args.thresholds,
        },
        "summary": {
            "instance_modes": len(mode_rows),
            "mean_teacher_to_generated_distance": float(np.mean(teacher_to_generated)),
            "mean_teacher_leave_one_out_distance": float(np.mean(teacher_self)),
            "mean_energy_distance": float(np.mean(energy)),
            "threshold_curve": threshold_summary,
        },
        "instance_modes": mode_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "control_diversity.json").write_text(json.dumps(document, indent=2) + "\n")

    lines = [
        "# Residual-control diversity diagnosis",
        "",
        "Distances are RMS residual-control distances in footprint-radius units.",
        "",
        "| Threshold | Generated -> teacher | Teacher leave-one-out |",
        "|---:|---:|---:|",
    ]
    for row in threshold_summary:
        lines.append(
            f"| {row['threshold_radius']:.2f} | "
            f"{row['mean_generated_teacher_coverage']:.2%} | "
            f"{row['mean_teacher_leave_one_out_coverage']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Mean teacher-to-generated nearest distance: {document['summary']['mean_teacher_to_generated_distance']:.4f}r",
            f"Mean teacher leave-one-out nearest distance: {document['summary']['mean_teacher_leave_one_out_distance']:.4f}r",
            f"Mean generated/teacher energy distance: {document['summary']['mean_energy_distance']:.4f}r",
        ]
    )
    (args.output / "control_diversity.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
