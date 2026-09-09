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

from diffusion_coverage.coverage import (
    generate_pattern_proposals,
    load_teacher_instance,
    surface_from_teacher_archive,
)
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.liftability import (
    ColourAwareStructuredTeacher,
    ColourAwareTeacherConfig,
    SyntheticColourField,
    SyntheticColourFieldConfig,
    colour_aware_objective,
)


OPERATING_POINTS = {"easy": 4, "medium": 16, "hard": 32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched-budget workspace and colour-aware multi-start teachers"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--liftability-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances-per-surface", type=int, default=5)
    parser.add_argument("--fields-per-difficulty", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--initial-sigma-radius", type=float, default=0.30)
    parser.add_argument("--final-sigma-radius", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only-task",
        action="append",
        default=[],
        metavar="INSTANCE:DIFFICULTY:FIELD",
        help="Restrict execution to explicit operating-point tasks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances_per_surface < 1 or args.fields_per_difficulty < 1:
        raise ValueError("instance and field counts must be positive")
    if args.restarts < 1 or args.steps < 1 or args.workers < 1:
        raise ValueError("search and worker counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(args.dataset)
    selected_ids = select_instance_ids(manifest, args.instances_per_surface)
    frontier = [
        json.loads(line)
        for line in (args.liftability_results / "frontier_results.jsonl").read_text().splitlines()
        if line
    ]
    selected_rows = [
        row
        for row in frontier
        if row["instance_id"] in selected_ids
        and int(row["budget"]) == OPERATING_POINTS[str(row["difficulty"])]
        and int(row["field_index"]) < args.fields_per_difficulty
    ]
    if args.only_task:
        requested = {parse_task_key(value) for value in args.only_task}
        selected_rows = [
            row
            for row in selected_rows
            if (str(row["instance_id"]), str(row["difficulty"]), int(row["field_index"]))
            in requested
        ]
        expected = len(requested)
    else:
        expected = len(selected_ids) * len(OPERATING_POINTS) * args.fields_per_difficulty
    if len(selected_rows) != expected:
        raise ValueError(f"expected {expected} operating-point rows, found {len(selected_rows)}")
    tasks = [
        (
            str(args.dataset),
            next(row for row in manifest if row["instance_id"] == frontier_row["instance_id"]),
            frontier_row,
            {
                "restarts": args.restarts,
                "steps": args.steps,
                "initial_sigma_radius": args.initial_sigma_radius,
                "final_sigma_radius": args.final_sigma_radius,
                "seed": args.seed + 100003 * index,
            },
        )
        for index, frontier_row in enumerate(selected_rows)
    ]
    start = perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed:03d}/{len(tasks):03d}] {result['instance_id']:<24} "
                f"{result['difficulty']:<6} baseline={result['baseline_success']} "
                f"direct={result['direct_success']} augmented={result['augmented_success']}",
                flush=True,
            )
    results.sort(key=lambda row: (row["instance_id"], row["difficulty"], row["field_index"]))
    with (args.output / "task_results.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(results, elapsed=perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = format_report(summary)
    (args.output / "report.md").write_text(report)
    print(report)


def parse_task_key(value: str) -> tuple[str, str, int]:
    parts = value.split(":")
    if len(parts) != 3 or parts[1] not in OPERATING_POINTS:
        raise ValueError(f"invalid task key: {value!r}")
    return parts[0], parts[1], int(parts[2])


def select_instance_ids(rows: list[dict[str, object]], count: int) -> set[str]:
    selected: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        surface = str(row["surface_id"])
        if counts[surface] < count:
            selected.add(str(row["instance_id"]))
            counts[surface] += 1
    return selected


def evaluate_task(task):
    dataset_raw, manifest_row, frontier_row, settings = task
    dataset = Path(dataset_raw)
    archive = load_teacher_instance(dataset / str(manifest_row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(manifest_row["surface_id"]))
    metadata = dict(archive["metadata"])
    teacher_config = metadata["teacher_config"]
    radius = float(teacher_config["footprint_radius"])
    tolerance = float(teacher_config["missed_tolerance"])
    overlap = float(teacher_config["overlap"])
    available_modes = {str(name) for name in archive["proposal_names"]}
    field_values = dict(frontier_row["field_config"])
    field_values["global_colours"] = tuple(field_values.get("global_colours", ()))
    field = SyntheticColourField(SyntheticColourFieldConfig(**field_values))
    proposals = [
        proposal
        for proposal in generate_pattern_proposals(
            surface,
            footprint_radius=radius,
            max_segments=1,
            overlap=overlap,
            waypoint_spacing=(
                None
                if teacher_config.get("waypoint_spacing") is None
                else float(teacher_config["waypoint_spacing"])
            ),
        )
        if proposal.name in available_modes
    ]
    mode_results = []
    direct_candidates = []
    total_evaluations = 0
    for mode_index, proposal in enumerate(proposals):
        config = ColourAwareTeacherConfig(
            footprint_radius=radius,
            missed_tolerance=tolerance,
            max_segments=int(frontier_row["budget"]),
            restarts=int(settings["restarts"]),
            steps_per_restart=int(settings["steps"]),
            initial_sigma_radius=float(settings["initial_sigma_radius"]),
            final_sigma_radius=float(settings["final_sigma_radius"]),
            seed=int(settings["seed"]) + mode_index,
        )
        result = ColourAwareStructuredTeacher(config).solve(surface, proposal, field)
        total_evaluations += result.evaluated_assignments
        candidate = result.best
        direct_candidates.append((candidate, config))
        mode_results.append(
            {
                "mode": proposal.name,
                "evaluated_assignments": result.evaluated_assignments,
                "template_segments": result.template.lift.min_segments,
                "template_length": result.template.coverage.path_length,
                "best_coverage_feasible": candidate.coverage_feasible,
                "best_lift_feasible": candidate.lift_feasible,
                "best_segments": candidate.lift.min_segments,
                "best_length": candidate.coverage.path_length,
                "best_missed_fraction": candidate.coverage.missed_fraction,
            }
        )
    best_candidate, best_config = min(
        direct_candidates,
        key=lambda item: colour_aware_objective(
            missed_fraction=item[0].coverage.missed_fraction,
            min_segments=item[0].lift.min_segments,
            path_length=item[0].coverage.path_length,
            missed_tolerance=item[1].missed_tolerance,
            max_segments=item[1].max_segments,
        ),
    )
    direct_success = best_candidate.feasible
    baseline_success = bool(frontier_row["colour_aware_success"])
    baseline_length = (
        float(frontier_row["colour_aware_length"]) if baseline_success else None
    )
    direct_length = best_candidate.coverage.path_length if direct_success else None
    augmented_success = baseline_success or direct_success
    feasible_lengths = [value for value in (baseline_length, direct_length) if value is not None]
    augmented_length = min(feasible_lengths) if feasible_lengths else None
    return {
        "instance_id": str(frontier_row["instance_id"]),
        "surface_id": str(frontier_row["surface_id"]),
        "difficulty": str(frontier_row["difficulty"]),
        "field_index": int(frontier_row["field_index"]),
        "field_config": frontier_row["field_config"],
        "budget": int(frontier_row["budget"]),
        "baseline_success": baseline_success,
        "baseline_length": baseline_length,
        "baseline_segments": frontier_row["colour_aware_min_segments"],
        "direct_success": direct_success,
        "direct_mode": best_candidate.plan.metadata["mode_name"],
        "direct_length": direct_length,
        "direct_segments": best_candidate.lift.min_segments,
        "direct_missed_fraction": best_candidate.coverage.missed_fraction,
        "direct_rescue": direct_success and not baseline_success,
        "direct_regression": baseline_success and not direct_success,
        "augmented_success": augmented_success,
        "augmented_length": augmented_length,
        "augmented_rescue": augmented_success and not baseline_success,
        "evaluated_assignments": total_evaluations,
        "mode_results": mode_results,
    }


def summarize(rows: list[dict[str, object]], *, elapsed: float) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["difficulty"])].append(row)
    by_difficulty = {}
    for difficulty, group in groups.items():
        both = [row for row in group if row["baseline_success"] and row["direct_success"]]
        augmented_ratios = [
            float(row["augmented_length"]) / float(row["baseline_length"])
            for row in group
            if row["baseline_success"] and row["augmented_length"] is not None
        ]
        by_difficulty[difficulty] = {
            "tasks": len(group),
            "baseline_success_rate": float(np.mean([row["baseline_success"] for row in group])),
            "direct_success_rate": float(np.mean([row["direct_success"] for row in group])),
            "direct_rescue_rate": float(np.mean([row["direct_rescue"] for row in group])),
            "direct_regression_rate": float(np.mean([row["direct_regression"] for row in group])),
            "augmented_success_rate": float(np.mean([row["augmented_success"] for row in group])),
            "augmented_rescue_rate": float(np.mean([row["augmented_rescue"] for row in group])),
            "mean_direct_to_baseline_length_ratio_when_both_feasible": (
                float(np.mean([float(row["direct_length"]) / float(row["baseline_length"]) for row in both]))
                if both
                else None
            ),
            "mean_augmented_to_baseline_length_ratio": (
                float(np.mean(augmented_ratios)) if augmented_ratios else None
            ),
            "mean_evaluated_assignments": float(
                np.mean([row["evaluated_assignments"] for row in group])
            ),
        }
    return {
        "tasks": len(rows),
        "instances": len({row["instance_id"] for row in rows}),
        "elapsed_seconds": elapsed,
        "by_difficulty": by_difficulty,
    }


def format_report(summary: dict[str, object]) -> str:
    lines = [
        "# Matched-budget colour-aware teacher benchmark",
        "",
        "| Difficulty | Tasks | Baseline success | Direct success | Direct rescue | Direct regression | Augmented success | Augmented rescue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for difficulty in OPERATING_POINTS:
        if difficulty not in summary["by_difficulty"]:
            continue
        row = summary["by_difficulty"][difficulty]
        lines.append(
            f"| {difficulty} | {row['tasks']} | {row['baseline_success_rate']:.2%} | "
            f"{row['direct_success_rate']:.2%} | {row['direct_rescue_rate']:.2%} | "
            f"{row['direct_regression_rate']:.2%} | {row['augmented_success_rate']:.2%} | "
            f"{row['augmented_rescue_rate']:.2%} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
