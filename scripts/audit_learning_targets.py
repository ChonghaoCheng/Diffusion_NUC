#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from diffusion_coverage.coverage import (
    CoveragePlan,
    densify_parameter_polyline,
    decode_raster_parameter_controls,
    decode_structured_parameter_controls,
    structured_residual_to_controls,
    evaluate_coverage,
    load_teacher_instance,
    surface_from_teacher_archive,
    map_surface_parameters,
)
from diffusion_coverage.learning import TeacherPathDataset, load_manifest
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-check canonical learning targets")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("train", "validation", "all", "dataset"),
        default="validation",
    )
    parser.add_argument(
        "--candidate-policy", choices=("best", "all"), default="best"
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--tokens-per-footprint-area", type=float, default=None)
    parser.add_argument("--maximum-path-tokens", type=int, default=None)
    parser.add_argument("--preserve-source-waypoints", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--path-coordinate-system",
        choices=(
            "xyz", "analytic_uv", "analytic_uv_control", "analytic_uv_structured",
            "analytic_uv_structured_residual",
        ),
        default=None,
        help="Override the checkpoint coordinate system for target-contract audits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.split == "train":
        ids = checkpoint["train_instance_ids"]
    elif args.split == "validation":
        ids = checkpoint["validation_instance_ids"]
    elif args.split == "all":
        ids = checkpoint["train_instance_ids"] + checkpoint["validation_instance_ids"]
    else:
        ids = None
    if args.max_instances is not None:
        if ids is None:
            ids = [
                str(row["instance_id"])
                for row in load_manifest(args.dataset)[: args.max_instances]
            ]
        else:
            ids = ids[: args.max_instances]
    coordinate_system = (
        checkpoint.get("path_coordinate_system", "xyz")
        if args.path_coordinate_system is None
        else args.path_coordinate_system
    )
    dataset = TeacherPathDataset(
        args.dataset,
        instance_ids=ids,
        num_surface_points=int(checkpoint["num_surface_points"]),
        num_path_waypoints=checkpoint["num_path_waypoints"],
        tokens_per_footprint_area=(
            float(checkpoint["tokens_per_footprint_area"])
            if args.tokens_per_footprint_area is None
            else args.tokens_per_footprint_area
        ),
        minimum_path_tokens=int(checkpoint["minimum_path_tokens"]),
        maximum_path_tokens=(
            int(checkpoint["maximum_path_tokens"])
            if args.maximum_path_tokens is None
            else args.maximum_path_tokens
        ),
        candidate_policy=args.candidate_policy,
        candidate_name=checkpoint.get("candidate_name"),
        allow_repaired_candidates=bool(
            checkpoint.get("allow_repaired_candidates", True)
        ),
        path_coordinate_system=coordinate_system,
        include_mode_conditioning=bool(checkpoint.get("include_mode_conditioning", False)),
        preserve_source_waypoints=args.preserve_source_waypoints,
        seed=int(checkpoint.get("data_seed", checkpoint["seed"])),
    )
    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        row = dataset.rows[dataset.sample_index[index].instance_index]
        archive = load_teacher_instance(args.dataset / str(row["path"]))
        metadata = archive["metadata"]
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        radius = float(metadata["teacher_config"]["footprint_radius"])
        epsilon = float(metadata["teacher_config"]["missed_tolerance"])
        model_path = sample["path"].numpy().astype(np.float64)
        if coordinate_system in {
            "analytic_uv", "analytic_uv_control", "analytic_uv_structured",
            "analytic_uv_structured_residual",
        }:
            parameters = (
                decode_raster_parameter_controls(
                    surface,
                    model_path,
                    footprint_radius=radius,
                    sweep_axis="u",
                )
                if coordinate_system == "analytic_uv_control"
                else (
                    decode_structured_parameter_controls(
                        surface,
                        (
                            structured_residual_to_controls(
                                surface,
                                model_path,
                                footprint_radius=radius,
                                overlap=float(metadata["teacher_config"]["overlap"]),
                                mode_name=str(sample["mode_name"]),
                            )
                            if coordinate_system == "analytic_uv_structured_residual"
                            else model_path
                        ),
                        footprint_radius=radius,
                        mode_name=str(sample["mode_name"]),
                    )
                    if coordinate_system in {
                        "analytic_uv_structured", "analytic_uv_structured_residual"
                    }
                    else model_path
                )
            )
            path = map_surface_parameters(surface, parameters[:, 0], parameters[:, 1])
        else:
            path = (
                model_path * float(sample["scale"])
                + sample["center"].numpy().astype(np.float64)
            )
        start = perf_counter()
        metrics = evaluate_coverage(surface, CoveragePlan(path), footprint_radius=radius)
        source_metrics = np.asarray(archive["candidate_metrics"])[
            int(sample["candidate_index"])
        ]
        record = {
            "instance_id": str(row["instance_id"]),
            "surface_id": str(row["surface_id"]),
            "candidate_index": int(sample["candidate_index"]),
            "mode_name": str(sample.get("mode_name", "unknown")),
            "num_tokens": len(path),
            "feasible": metrics.missed_fraction <= epsilon + 1e-12,
            "missed_fraction": metrics.missed_fraction,
            "source_missed_fraction": float(source_metrics[0]),
            "path_length": metrics.path_length,
            "source_path_length": float(source_metrics[1]),
            "check_time": perf_counter() - start,
        }
        records.append(record)
        if not args.quiet:
            print(
                f"{record['instance_id']:<24} feasible={record['feasible']} "
                f"miss={record['missed_fraction']:.4f} source={record['source_missed_fraction']:.4f}",
                flush=True,
            )
    summary = summarize(records)
    summary["unique_instances"] = len(
        {str(record["instance_id"]) for record in records}
    )
    summary["by_surface"] = {
        surface: summarize([record for record in records if record["surface_id"] == surface])
        for surface in sorted({record["surface_id"] for record in records})
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "instances": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "instances": len(records),
        "feasible_rate": float(np.mean([record["feasible"] for record in records])),
        "mean_absolute_missed_change": float(np.mean([
            abs(record["missed_fraction"] - record["source_missed_fraction"]) for record in records
        ])),
        "maximum_absolute_missed_change": float(max(
            abs(record["missed_fraction"] - record["source_missed_fraction"]) for record in records
        )),
        "mean_length_ratio": float(np.mean([
            record["path_length"] / record["source_path_length"] for record in records
        ])),
        "mean_check_time": float(np.mean([record["check_time"] for record in records])),
    }


if __name__ == "__main__":
    main()
