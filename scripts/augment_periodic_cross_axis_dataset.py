#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.coverage import (
    load_teacher_instance,
    MultiStartStructuredTeacher,
    StructuredTeacherConfig,
    surface_from_teacher_archive,
)
from diffusion_coverage.coverage.patterns import raster_pattern
from diffusion_coverage.coverage.teacher import plan_smoothness
from diffusion_coverage.learning import load_manifest


PERIODIC_SURFACES = {"cylinder", "hemisphere"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append audited raster-v multi-start candidates to periodic surfaces"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--initial-sigma-radius", type=float, default=0.30)
    parser.add_argument("--final-sigma-radius", type=float, default=0.03)
    parser.add_argument("--minimum-diversity-radius", type=float, default=0.05)
    parser.add_argument("--maximum-length-ratio", type=float, default=1.15)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    (args.output / "instances").mkdir(parents=True)
    rows = load_manifest(args.input)
    settings = {
        "restarts": args.restarts,
        "steps": args.steps,
        "initial_sigma_radius": args.initial_sigma_radius,
        "final_sigma_radius": args.final_sigma_radius,
        "minimum_diversity_radius": args.minimum_diversity_radius,
        "maximum_length_ratio": args.maximum_length_ratio,
        "seed": args.seed,
    }
    start = perf_counter()
    tasks = [
        (str(args.input), str(args.output), row, index, settings)
        for index, row in enumerate(rows)
    ]
    output_rows = []
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(augment_instance, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            output_row, record = future.result()
            output_rows.append(output_row)
            records.append(record)
            print(
                f"[{completed:03d}/{len(rows):03d}] {record['instance_id']:<24} "
                f"added={record['added_candidates']}",
                flush=True,
            )
    output_rows.sort(key=lambda row: str(row["instance_id"]))
    records.sort(key=lambda row: str(row["instance_id"]))
    with (args.output / "manifest.jsonl").open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.output / "augmentation.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "instances": len(rows),
        "augmented_instances": sum(record["added_candidates"] > 0 for record in records),
        "input_candidates": int(sum(record["input_candidates"] for record in records)),
        "added_candidates": int(sum(record["added_candidates"] for record in records)),
        "output_candidates": int(sum(record["output_candidates"] for record in records)),
        "evaluated_assignments": int(sum(record["evaluated_assignments"] for record in records)),
        "settings": settings,
        "elapsed_seconds": perf_counter() - start,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def augment_instance(task):
    input_raw, output_raw, row, instance_index, settings = task
    input_root = Path(input_raw)
    output_root = Path(output_raw)
    input_path = input_root / str(row["path"])
    output_path = output_root / "instances" / input_path.name
    if str(row["surface_id"]) not in PERIODIC_SURFACES:
        shutil.copy2(input_path, output_path)
        output_row = dict(row)
        output_row["path"] = str(output_path.relative_to(output_root))
        count = int(row["num_candidates"])
        return output_row, {
            "instance_id": str(row["instance_id"]),
            "surface_id": str(row["surface_id"]),
            "input_candidates": count,
            "added_candidates": 0,
            "output_candidates": count,
            "evaluated_assignments": 0,
        }

    archive = load_teacher_instance(input_path)
    existing_names = {str(name) for name in archive["proposal_names"]}
    if "raster_v_phase_0.00" in existing_names:
        raise ValueError(f"{row['instance_id']} already contains raster_v_phase_0.00")
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    metadata = dict(archive["metadata"])
    config = metadata["teacher_config"]
    radius = float(config["footprint_radius"])
    proposal = raster_pattern(
        surface,
        footprint_radius=radius,
        max_segments=1,
        overlap=float(config["overlap"]),
        waypoint_spacing=(
            None if config.get("waypoint_spacing") is None else float(config["waypoint_spacing"])
        ),
        sweep_axis="v",
        phase=0.0,
    )
    teacher = MultiStartStructuredTeacher(
        StructuredTeacherConfig(
            footprint_radius=radius,
            missed_tolerance=float(config["missed_tolerance"]),
            restarts=int(settings["restarts"]),
            steps_per_restart=int(settings["steps"]),
            initial_sigma_radius=float(settings["initial_sigma_radius"]),
            final_sigma_radius=float(settings["final_sigma_radius"]),
            minimum_diversity_radius=float(settings["minimum_diversity_radius"]),
            maximum_length_ratio=float(settings["maximum_length_ratio"]),
            seed=int(settings["seed"]) + 1009 * instance_index,
        )
    ).solve(surface, proposal)
    if not teacher.candidates:
        shutil.copy2(input_path, output_path)
        output_row = dict(row)
        output_row["path"] = str(output_path.relative_to(output_root))
        count = len(archive["proposal_names"])
        return output_row, {
            "instance_id": str(row["instance_id"]),
            "surface_id": str(row["surface_id"]),
            "input_candidates": count,
            "added_candidates": 0,
            "output_candidates": count,
            "evaluated_assignments": teacher.evaluated_assignments,
            "rejected_infeasible_template": not teacher.template.feasible,
        }
    new_arrays = pack_candidates(teacher.candidates)
    merged = merge_archive_candidates(archive, new_arrays, proposal.name)
    metadata["periodic_cross_axis_augmentation"] = {
        **settings,
        "mode": proposal.name,
        "evaluated_assignments": teacher.evaluated_assignments,
        "rejected_infeasible_template": not teacher.template.feasible,
    }
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            vertices=np.asarray(archive["vertices"]),
            faces=np.asarray(archive["faces"]),
            sample_points=np.asarray(archive["sample_points"]),
            sample_normals=np.asarray(archive["sample_normals"]),
            area_weights=np.asarray(archive["area_weights"]),
            sample_face_indices=np.asarray(archive["sample_face_indices"]),
            sample_barycentric=np.asarray(archive["sample_barycentric"]),
            candidate_waypoints=merged["waypoints"],
            candidate_segment_mask=merged["segment_mask"],
            candidate_waypoint_mask=merged["waypoint_mask"],
            candidate_metrics=merged["metrics"],
            candidate_controls=merged["controls"],
            candidate_control_mask=merged["control_mask"],
            proposal_names=merged["proposal_names"],
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    temporary.replace(output_path)
    output_row = dict(row)
    output_row["path"] = str(output_path.relative_to(output_root))
    output_row["num_candidates"] = len(merged["proposal_names"])
    output_row["num_feasible_candidates"] = len(merged["proposal_names"])
    return output_row, {
        "instance_id": str(row["instance_id"]),
        "surface_id": str(row["surface_id"]),
        "input_candidates": len(archive["proposal_names"]),
        "added_candidates": len(teacher.candidates),
        "output_candidates": len(merged["proposal_names"]),
        "evaluated_assignments": teacher.evaluated_assignments,
    }


def pack_candidates(candidates) -> dict[str, np.ndarray]:
    count = len(candidates)
    max_waypoints = max(len(candidate.plan.active_paths()[0]) for candidate in candidates)
    max_controls = max(len(candidate.controls) for candidate in candidates)
    waypoints = np.zeros((count, 1, max_waypoints, 3), dtype=np.float64)
    segment_mask = np.ones((count, 1), dtype=bool)
    waypoint_mask = np.zeros((count, 1, max_waypoints), dtype=bool)
    metrics = np.zeros((count, 5), dtype=np.float64)
    controls = np.zeros((count, max_controls, 2), dtype=np.float64)
    control_mask = np.zeros((count, max_controls), dtype=bool)
    for index, candidate in enumerate(candidates):
        path = candidate.plan.active_paths()[0]
        waypoints[index, 0, : len(path)] = path
        waypoint_mask[index, 0, : len(path)] = True
        metrics[index] = (
            candidate.metrics.missed_fraction,
            candidate.metrics.path_length,
            candidate.metrics.coverage_efficiency,
            plan_smoothness(candidate.plan),
            float(candidate.feasible),
        )
        controls[index, : len(candidate.controls)] = candidate.controls
        control_mask[index, : len(candidate.controls)] = True
    return {
        "waypoints": waypoints,
        "segment_mask": segment_mask,
        "waypoint_mask": waypoint_mask,
        "metrics": metrics,
        "controls": controls,
        "control_mask": control_mask,
    }


def merge_archive_candidates(archive, new, mode_name: str) -> dict[str, np.ndarray]:
    old_waypoints = np.asarray(archive["candidate_waypoints"])
    max_waypoints = max(old_waypoints.shape[2], new["waypoints"].shape[2])
    old_controls = np.asarray(archive["candidate_controls"])
    max_controls = max(old_controls.shape[1], new["controls"].shape[1])

    def pad(array, shape):
        output = np.zeros(shape, dtype=array.dtype)
        slices = tuple(slice(0, size) for size in array.shape)
        output[slices] = array
        return output

    old_count = len(old_waypoints)
    new_count = len(new["waypoints"])
    return {
        "waypoints": np.concatenate(
            (
                pad(old_waypoints, (old_count, 1, max_waypoints, 3)),
                pad(new["waypoints"], (new_count, 1, max_waypoints, 3)),
            )
        ),
        "segment_mask": np.concatenate(
            (np.asarray(archive["candidate_segment_mask"]), new["segment_mask"])
        ),
        "waypoint_mask": np.concatenate(
            (
                pad(
                    np.asarray(archive["candidate_waypoint_mask"]),
                    (old_count, 1, max_waypoints),
                ),
                pad(new["waypoint_mask"], (new_count, 1, max_waypoints)),
            )
        ),
        "metrics": np.concatenate((np.asarray(archive["candidate_metrics"]), new["metrics"])),
        "controls": np.concatenate(
            (
                pad(old_controls, (old_count, max_controls, 2)),
                pad(new["controls"], (new_count, max_controls, 2)),
            )
        ),
        "control_mask": np.concatenate(
            (
                pad(
                    np.asarray(archive["candidate_control_mask"]),
                    (old_count, max_controls),
                ),
                pad(new["control_mask"], (new_count, max_controls)),
            )
        ),
        "proposal_names": np.concatenate(
            (
                np.asarray(archive["proposal_names"]),
                np.asarray([mode_name] * new_count, dtype=np.str_),
            )
        ),
    }


if __name__ == "__main__":
    main()
