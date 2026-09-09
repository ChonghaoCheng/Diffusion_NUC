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

from diffusion_coverage.coverage import (
    CoveragePlan,
    constrained_coverage_key,
    load_teacher_instance,
    surface_from_teacher_archive,
)
from diffusion_coverage.evaluation.representation_audit import audit_fixed_control_bsplines
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.representation import suggested_token_count
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fixed B-spline bandwidth against variable-token paths")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--raw-inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-points", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--max-reference-tokens", type=int, default=1024)
    parser.add_argument("--geometry-threshold-r", type=float, default=0.25)
    parser.add_argument("--coverage-threshold", type=float, default=0.01)
    parser.add_argument("--length-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.raw_inputs:
        rows = []
        for raw_input in args.raw_inputs:
            with raw_input.open(newline="") as source:
                rows.extend(dict(row) for row in csv.DictReader(source))
    else:
        rows = audit_dataset(args)
    write_csv(args.output / "raw_results.csv", rows)
    summary = summarize(rows, args)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_summary(rows, args.output / "representation_audit.png")
    print(json.dumps(summary["decision"], indent=2))


def audit_dataset(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    assert args.dataset is not None
    rows: list[dict[str, float | int | str]] = []
    for manifest_row in load_manifest(args.dataset):
        archive = load_teacher_instance(args.dataset / str(manifest_row["path"]))
        metadata = archive["metadata"]
        surface = surface_from_teacher_archive(
            archive, surface_id=str(manifest_row["surface_id"])
        )
        radius = float(metadata["teacher_config"]["footprint_radius"])
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
        path = plan.active_paths()[0]
        reference_tokens = suggested_token_count(
            surface.total_area, radius, tokens_per_footprint_area=0.5,
            minimum=max(args.control_points), maximum=args.max_reference_tokens,
        )
        control_counts = [count for count in args.control_points if count <= reference_tokens]
        results = audit_fixed_control_bsplines(
            surface,
            path,
            footprint_radius=radius,
            num_reference_tokens=reference_tokens,
            control_point_counts=control_counts,
        )
        for result in results:
            rows.append({
                "instance_id": str(manifest_row["instance_id"]),
                "surface_id": str(manifest_row["surface_id"]),
                "radius": radius,
                "surface_area": surface.total_area,
                "reference_tokens": reference_tokens,
                "control_points": result.num_control_points,
                "geometry_error": result.geometry_error,
                "geometry_error_over_radius": result.geometry_error / radius,
                "delta_missed_fraction": result.delta_missed_fraction,
                "delta_path_length": result.delta_path_length,
                "relative_path_length_error": result.relative_path_length_error,
            })
        print(f"audited {manifest_row['instance_id']} M={reference_tokens}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]], args: argparse.Namespace) -> dict[str, object]:
    grouped: dict[str, dict[str, float | int | bool]] = {}
    for control_points in sorted({int(row["control_points"]) for row in rows}):
        selected = [row for row in rows if int(row["control_points"]) == control_points]
        geometry = np.asarray([row["geometry_error_over_radius"] for row in selected], dtype=float)
        coverage = np.abs(np.asarray([row["delta_missed_fraction"] for row in selected], dtype=float))
        length = np.abs(np.asarray([row["relative_path_length_error"] for row in selected], dtype=float))
        grouped[str(control_points)] = {
            "instances": len(selected),
            "median_geometry_error_over_radius": float(np.median(geometry)),
            "p95_geometry_error_over_radius": float(np.quantile(geometry, 0.95)),
            "median_abs_delta_missed_fraction": float(np.median(coverage)),
            "p95_abs_delta_missed_fraction": float(np.quantile(coverage, 0.95)),
            "median_abs_relative_length_error": float(np.median(length)),
            "p95_abs_relative_length_error": float(np.quantile(length, 0.95)),
        }
    acceptable = [
        int(key) for key, value in grouped.items()
        if value["p95_geometry_error_over_radius"] <= args.geometry_threshold_r
        and value["p95_abs_delta_missed_fraction"] <= args.coverage_threshold
        and value["p95_abs_relative_length_error"] <= args.length_threshold
    ]
    return {
        "num_instances": len({str(row["instance_id"]) for row in rows}),
        "thresholds": {
            "p95_geometry_error_over_radius": args.geometry_threshold_r,
            "p95_abs_delta_missed_fraction": args.coverage_threshold,
            "p95_abs_relative_length_error": args.length_threshold,
        },
        "by_control_points": grouped,
        "decision": {
            "fixed_bspline_acceptable": bool(acceptable),
            "minimum_acceptable_control_points": min(acceptable) if acceptable else None,
            "default_representation": "variable_token",
        },
    }


def plot_summary(rows: list[dict[str, float | int | str]], output: Path) -> None:
    radii = np.asarray([row["radius"] for row in rows], dtype=float)
    boundaries = np.quantile(radii, [0.0, 0.33, 0.66, 1.0])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = (
        ("geometry_error_over_radius", "Geometry error / r"),
        ("delta_missed_fraction", "Absolute delta missed"),
        ("relative_path_length_error", "Absolute relative length error"),
    )
    for bin_index in range(3):
        low, high = boundaries[bin_index], boundaries[bin_index + 1]
        for axis, (field, label) in zip(axes, metrics):
            controls = sorted({int(row["control_points"]) for row in rows})
            medians = []
            for control in controls:
                values = [
                    abs(float(row[field])) for row in rows
                    if int(row["control_points"]) == control
                    and low <= float(row["radius"]) <= high + 1e-12
                ]
                medians.append(float(np.median(values)))
            axis.plot(controls, medians, marker="o", label=f"r={low:.3f}-{high:.3f}")
            axis.set_xscale("log", base=2)
            axis.set_xlabel("B-spline control points")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
