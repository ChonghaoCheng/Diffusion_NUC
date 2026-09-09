#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.coverage import (
    generate_pattern_proposals,
    load_teacher_instance,
    MultiStartStructuredTeacher,
    StructuredTeacherConfig,
    surface_from_teacher_archive,
)
from diffusion_coverage.coverage.teacher import plan_smoothness
from diffusion_coverage.learning import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hard-feasible multi-start structured teacher dataset"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances-per-surface", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--initial-sigma-radius", type=float, default=0.30)
    parser.add_argument("--final-sigma-radius", type=float, default=0.03)
    parser.add_argument("--minimum-diversity-radius", type=float, default=0.05)
    parser.add_argument("--maximum-length-ratio", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    instances_dir = args.output / "instances"
    instances_dir.mkdir(parents=True)
    selected = select_rows(
        load_manifest(args.input), args.instances_per_surface
    )
    output_rows: list[dict[str, object]] = []
    mode_counts: Counter[str] = Counter()
    alternatives_by_mode: dict[str, list[int]] = defaultdict(list)
    distance_by_mode: dict[str, list[float]] = defaultdict(list)
    length_ratio_by_mode: dict[str, list[float]] = defaultdict(list)
    total_evaluations = 0
    build_start = perf_counter()
    settings = {
        "restarts": args.restarts,
        "steps": args.steps,
        "initial_sigma_radius": args.initial_sigma_radius,
        "final_sigma_radius": args.final_sigma_radius,
        "minimum_diversity_radius": args.minimum_diversity_radius,
        "maximum_length_ratio": args.maximum_length_ratio,
        "seed": args.seed,
    }
    tasks = [
        (str(args.input), str(args.output), row, instance_number, settings)
        for instance_number, row in enumerate(selected, start=1)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_instance, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            output_row, mode_records = future.result()
            completed += 1
            output_rows.append(output_row)
            for record in mode_records:
                mode = str(record["mode"])
                mode_counts[mode] += int(record["candidates"])
                alternatives_by_mode[mode].append(int(record["alternatives"]))
                distance_by_mode[mode].extend(record["distances"])
                length_ratio_by_mode[mode].extend(record["length_ratios"])
                total_evaluations += int(record["evaluations"])
            print(
                f"[{completed:03d}/{len(selected):03d}] {output_row['instance_id']:<24} "
                f"candidates={output_row['num_candidates']} evaluations={total_evaluations}",
                flush=True,
            )
    output_rows.sort(key=lambda row: str(row["instance_id"]))
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "instances": len(output_rows),
        "candidates": int(sum(mode_counts.values())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "mean_alternatives_per_instance_mode": {
            mode: float(np.mean(values))
            for mode, values in sorted(alternatives_by_mode.items())
        },
        "mean_alternative_distance_radius": {
            mode: float(np.mean(values))
            for mode, values in sorted(distance_by_mode.items())
        },
        "mean_alternative_length_ratio": {
            mode: float(np.mean(values))
            for mode, values in sorted(length_ratio_by_mode.items())
        },
        "evaluated_assignments": total_evaluations,
        "build_time": perf_counter() - build_start,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def select_rows(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        surface_id = str(row["surface_id"])
        if counts[surface_id] < count:
            selected.append(row)
            counts[surface_id] += 1
    return selected


def build_instance(task):
    input_dir, output_dir, row, instance_number, settings = task
    input_path = Path(input_dir)
    output_path_root = Path(output_dir)
    archive = load_teacher_instance(input_path / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    metadata = dict(archive["metadata"])
    config = metadata["teacher_config"]
    radius = float(config["footprint_radius"])
    available_modes = {str(name) for name in archive["proposal_names"]}
    proposals = generate_pattern_proposals(
        surface,
        footprint_radius=radius,
        max_segments=int(config["max_segments"]),
        overlap=float(config["overlap"]),
        waypoint_spacing=(
            None
            if config.get("waypoint_spacing") is None
            else float(config["waypoint_spacing"])
        ),
    )
    candidates = []
    names = []
    mode_records = []
    for mode_number, proposal in enumerate(proposals):
        if proposal.name not in available_modes:
            continue
        result = MultiStartStructuredTeacher(
            StructuredTeacherConfig(
                footprint_radius=radius,
                missed_tolerance=float(config["missed_tolerance"]),
                restarts=int(settings["restarts"]),
                steps_per_restart=int(settings["steps"]),
                initial_sigma_radius=float(settings["initial_sigma_radius"]),
                final_sigma_radius=float(settings["final_sigma_radius"]),
                minimum_diversity_radius=float(settings["minimum_diversity_radius"]),
                maximum_length_ratio=float(settings["maximum_length_ratio"]),
                seed=int(settings["seed"]) + 1009 * instance_number + mode_number,
            )
        ).solve(surface, proposal)
        alternatives = result.candidates[1:]
        mode_records.append(
            {
                "mode": proposal.name,
                "candidates": len(result.candidates),
                "alternatives": len(alternatives),
                "distances": [
                    candidate.distance_from_template_radius
                    for candidate in alternatives
                ],
                "length_ratios": [
                    candidate.metrics.path_length / result.template.metrics.path_length
                    for candidate in alternatives
                ],
                "evaluations": result.evaluated_assignments,
            }
        )
        for candidate in result.candidates:
            candidates.append(candidate)
            names.append(proposal.name)
    if not candidates:
        raise RuntimeError(f"no candidates for {row['instance_id']}")
    arrays = pack_candidates(candidates)
    metadata["multistart_structured_teacher"] = {
        **settings,
        "steps_per_restart": settings["steps"],
        "includes_template_per_mode": True,
    }
    output_path = output_path_root / "instances" / f"{row['instance_id']}.npz"
    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            vertices=np.asarray(archive["vertices"]),
            faces=np.asarray(archive["faces"]),
            sample_points=np.asarray(archive["sample_points"]),
            sample_normals=np.asarray(archive["sample_normals"]),
            area_weights=np.asarray(archive["area_weights"]),
            sample_face_indices=np.asarray(archive["sample_face_indices"]),
            sample_barycentric=np.asarray(archive["sample_barycentric"]),
            candidate_waypoints=arrays["waypoints"],
            candidate_segment_mask=arrays["segment_mask"],
            candidate_waypoint_mask=arrays["waypoint_mask"],
            candidate_metrics=arrays["metrics"],
            candidate_controls=arrays["controls"],
            candidate_control_mask=arrays["control_mask"],
            proposal_names=np.asarray(names, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    temporary_path.replace(output_path)
    output_row = dict(row)
    output_row["path"] = str(output_path.relative_to(output_path_root))
    output_row["num_candidates"] = len(candidates)
    output_row["num_feasible_candidates"] = len(candidates)
    return output_row, mode_records


def pack_candidates(candidates) -> dict[str, np.ndarray]:
    maximum_waypoints = max(
        len(candidate.plan.active_paths()[0]) for candidate in candidates
    )
    count = len(candidates)
    waypoints = np.zeros((count, 1, maximum_waypoints, 3), dtype=np.float64)
    segment_mask = np.ones((count, 1), dtype=bool)
    waypoint_mask = np.zeros((count, 1, maximum_waypoints), dtype=bool)
    metrics = np.zeros((count, 5), dtype=np.float64)
    maximum_controls = max(len(candidate.controls) for candidate in candidates)
    controls = np.zeros((count, maximum_controls, 2), dtype=np.float64)
    control_mask = np.zeros((count, maximum_controls), dtype=bool)
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


if __name__ == "__main__":
    main()
