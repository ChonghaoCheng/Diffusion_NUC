#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
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

from diffusion_coverage.coverage import evaluate_coverage, load_teacher_instance, surface_from_teacher_archive
from diffusion_coverage.coverage.patterns import raster_pattern
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.liftability import (
    SyntheticColourField,
    SyntheticColourFieldConfig,
    evaluate_colour_lift,
)


OPERATING_POINTS = {"easy": 4, "medium": 16, "hard": 32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit missing raster-v templates on cylinder and hemisphere tasks"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--liftability-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = [
        row for row in load_manifest(args.dataset) if row["surface_id"] in {"cylinder", "hemisphere"}
    ]
    frontier = [
        json.loads(line)
        for line in (args.liftability_results / "frontier_results.jsonl").read_text().splitlines()
        if line
    ]
    fields_by_instance: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frontier:
        if row["instance_id"] not in {item["instance_id"] for item in manifest}:
            continue
        if int(row["budget"]) == OPERATING_POINTS[str(row["difficulty"])]:
            fields_by_instance[str(row["instance_id"])].append(row)
    start = perf_counter()
    tasks = [(str(args.dataset), row, fields_by_instance[str(row["instance_id"])]) for row in manifest]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_instance, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            instance_rows = future.result()
            results.extend(instance_rows)
            print(
                f"[{completed:03d}/{len(tasks):03d}] {instance_rows[0]['instance_id']:<24} "
                f"coverage={instance_rows[0]['cross_axis_coverage_feasible']}",
                flush=True,
            )
    results.sort(key=lambda row: (row["instance_id"], row["difficulty"], row["field_index"]))
    with (args.output / "task_results.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(results, perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = format_report(summary)
    (args.output / "report.md").write_text(report)
    print(report)


def evaluate_instance(task):
    dataset_raw, manifest_row, field_rows = task
    dataset = Path(dataset_raw)
    archive = load_teacher_instance(dataset / str(manifest_row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(manifest_row["surface_id"]))
    config = dict(archive["metadata"])["teacher_config"]
    radius = float(config["footprint_radius"])
    tolerance = float(config["missed_tolerance"])
    proposal = raster_pattern(
        surface,
        footprint_radius=radius,
        max_segments=1,
        overlap=float(config["overlap"]),
        waypoint_spacing=(
            None if config.get("waypoint_spacing") is None else float(config["waypoint_spacing"])
        ),
        sweep_axis="v",
    )
    coverage = evaluate_coverage(surface, proposal.plan, footprint_radius=radius)
    coverage_feasible = coverage.missed_fraction <= tolerance + 1e-12
    rows = []
    for field_row in field_rows:
        values = dict(field_row["field_config"])
        values["global_colours"] = tuple(values.get("global_colours", ()))
        lift = evaluate_colour_lift(
            surface,
            proposal.plan,
            SyntheticColourField(SyntheticColourFieldConfig(**values)),
        )
        cross_axis_success = coverage_feasible and lift.within_budget(int(field_row["budget"]))
        baseline_success = bool(field_row["colour_aware_success"])
        rows.append(
            {
                "instance_id": str(field_row["instance_id"]),
                "surface_id": str(field_row["surface_id"]),
                "difficulty": str(field_row["difficulty"]),
                "field_index": int(field_row["field_index"]),
                "budget": int(field_row["budget"]),
                "baseline_success": baseline_success,
                "cross_axis_coverage_feasible": coverage_feasible,
                "cross_axis_missed_fraction": coverage.missed_fraction,
                "cross_axis_length": coverage.path_length,
                "cross_axis_segments": lift.min_segments,
                "cross_axis_success": cross_axis_success,
                "augmented_success": baseline_success or cross_axis_success,
                "cross_axis_rescue": cross_axis_success and not baseline_success,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]], elapsed: float) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["surface_id"]), str(row["difficulty"]))].append(row)
    by_surface = {}
    for (surface, difficulty), group in groups.items():
        by_surface.setdefault(surface, {})[difficulty] = {
            "tasks": len(group),
            "cross_axis_coverage_feasible_rate": float(
                np.mean([row["cross_axis_coverage_feasible"] for row in group])
            ),
            "baseline_success_rate": float(np.mean([row["baseline_success"] for row in group])),
            "augmented_success_rate": float(np.mean([row["augmented_success"] for row in group])),
            "cross_axis_rescue_rate": float(np.mean([row["cross_axis_rescue"] for row in group])),
        }
    return {
        "instances": len({row["instance_id"] for row in rows}),
        "tasks": len(rows),
        "elapsed_seconds": elapsed,
        "by_surface": by_surface,
    }


def format_report(summary: dict[str, object]) -> str:
    lines = [
        "# Periodic-surface cross-axis template audit",
        "",
        "| Surface | Difficulty | Tasks | raster-v coverage feasible | Baseline success | Augmented success | Rescue |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for surface, difficulties in summary["by_surface"].items():
        for difficulty in OPERATING_POINTS:
            row = difficulties[difficulty]
            lines.append(
                f"| {surface} | {difficulty} | {row['tasks']} | "
                f"{row['cross_axis_coverage_feasible_rate']:.2%} | "
                f"{row['baseline_success_rate']:.2%} | {row['augmented_success_rate']:.2%} | "
                f"{row['cross_axis_rescue_rate']:.2%} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
