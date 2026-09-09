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

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.coverage import CoveragePlan, load_teacher_instance, surface_from_teacher_archive
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.liftability import (
    SyntheticColourField,
    SyntheticColourFieldConfig,
    evaluate_colour_lift,
)


DIFFICULTIES = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark post-hoc and colour-aware liftability on 3D teacher paths"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances-per-surface", type=int, default=40)
    parser.add_argument("--fields-per-difficulty", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32, 64, 128))
    parser.add_argument("--maximum-parameter-step", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances_per_surface < 1 or args.fields_per_difficulty < 1:
        raise ValueError("instance and field counts must be positive")
    if args.workers < 1 or any(budget < 1 for budget in args.budgets):
        raise ValueError("workers and budgets must be positive")
    budgets = sorted(set(args.budgets))
    rows = select_rows(load_manifest(args.dataset), args.instances_per_surface)
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    tasks = [
        (
            str(args.dataset),
            row,
            args.fields_per_difficulty,
            budgets,
            args.maximum_parameter_step,
            args.seed + 104729 * index,
        )
        for index, row in enumerate(rows)
    ]
    candidate_rows: list[dict[str, object]] = []
    frontier_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_instance, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            candidates, frontiers = future.result()
            candidate_rows.extend(candidates)
            frontier_rows.extend(frontiers)
            print(
                f"[{completed:03d}/{len(rows):03d}] {frontiers[0]['instance_id']:<24} "
                f"fields={len(frontiers) // len(budgets)}",
                flush=True,
            )

    candidate_rows.sort(
        key=lambda row: (
            row["instance_id"], row["difficulty"], row["field_index"], row["candidate_index"]
        )
    )
    frontier_rows.sort(
        key=lambda row: (row["instance_id"], row["difficulty"], row["field_index"], row["budget"])
    )
    write_jsonl(args.output / "candidate_results.jsonl", candidate_rows)
    write_jsonl(args.output / "frontier_results.jsonl", frontier_rows)
    summary = summarize(frontier_rows, candidate_rows, elapsed=perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = format_report(summary)
    (args.output / "report.md").write_text(report)
    plot_summary(summary, args.output / "synthetic_liftability.png")
    print(report)


def select_rows(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    selected = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        surface = str(row["surface_id"])
        if counts[surface] < count:
            selected.append(row)
            counts[surface] += 1
    return selected


def evaluate_instance(task):
    dataset_raw, row, fields_per_difficulty, budgets, maximum_parameter_step, seed = task
    dataset = Path(dataset_raw)
    archive = load_teacher_instance(dataset / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    plans = []
    for index in range(len(archive["proposal_names"])):
        plans.append(
            CoveragePlan(
                np.asarray(archive["candidate_waypoints"])[index],
                np.asarray(archive["candidate_segment_mask"])[index],
                np.asarray(archive["candidate_waypoint_mask"])[index],
            )
        )
    lengths = np.asarray(archive["candidate_metrics"], dtype=np.float64)[:, 1]
    names = [str(name) for name in archive["proposal_names"]]
    workspace_index = int(np.argmin(lengths))
    rng = np.random.default_rng(seed)
    candidate_rows = []
    frontier_rows = []
    for difficulty in DIFFICULTIES:
        for field_index in range(fields_per_difficulty):
            field_config = sample_field_config(difficulty, rng)
            field = SyntheticColourField(field_config)
            segment_counts = []
            sampled_points = []
            for candidate_index, plan in enumerate(plans):
                result = evaluate_colour_lift(
                    surface,
                    plan,
                    field,
                    maximum_parameter_step=maximum_parameter_step,
                )
                segment_counts.append(result.min_segments)
                sampled_points.append(result.sampled_points)
                candidate_rows.append(
                    {
                        "instance_id": str(row["instance_id"]),
                        "surface_id": str(row["surface_id"]),
                        "difficulty": difficulty,
                        "field_index": field_index,
                        "field_config": field_config.__dict__,
                        "candidate_index": candidate_index,
                        "mode": names[candidate_index],
                        "path_length": float(lengths[candidate_index]),
                        "min_segments": result.min_segments,
                        "sampled_points": result.sampled_points,
                        "valid_colour_fraction": result.valid_colour_fraction,
                        "workspace_best": candidate_index == workspace_index,
                    }
                )
            finite_indices = [index for index, count in enumerate(segment_counts) if count is not None]
            if not finite_indices:
                raise RuntimeError("synthetic field unexpectedly made every candidate unreachable")
            previous_aware = False
            previous_length = float("inf")
            for budget in budgets:
                feasible_indices = [
                    index
                    for index in finite_indices
                    if int(segment_counts[index]) <= budget
                ]
                aware_index = (
                    min(feasible_indices, key=lambda index: float(lengths[index]))
                    if feasible_indices
                    else None
                )
                aware_feasible = aware_index is not None
                aware_length = float(lengths[aware_index]) if aware_index is not None else None
                if previous_aware and not aware_feasible:
                    raise AssertionError("colour-aware feasibility must be monotone in segment budget")
                if aware_length is not None and aware_length > previous_length + 1e-10:
                    raise AssertionError("colour-aware length frontier must be nonincreasing")
                previous_aware = aware_feasible
                if aware_length is not None:
                    previous_length = aware_length
                workspace_success = int(segment_counts[workspace_index]) <= budget
                frontier_rows.append(
                    {
                        "instance_id": str(row["instance_id"]),
                        "surface_id": str(row["surface_id"]),
                        "difficulty": difficulty,
                        "field_index": field_index,
                        "field_config": field_config.__dict__,
                        "budget": budget,
                        "workspace_mode": names[workspace_index],
                        "workspace_length": float(lengths[workspace_index]),
                        "workspace_min_segments": int(segment_counts[workspace_index]),
                        "workspace_success": workspace_success,
                        "colour_aware_success": aware_feasible,
                        "colour_aware_mode": names[aware_index] if aware_index is not None else None,
                        "colour_aware_length": aware_length,
                        "colour_aware_min_segments": (
                            int(segment_counts[aware_index]) if aware_index is not None else None
                        ),
                        "rescued": aware_feasible and not workspace_success,
                    }
                )
    return candidate_rows, frontier_rows


def sample_field_config(
    difficulty: str, rng: np.random.Generator
) -> SyntheticColourFieldConfig:
    if difficulty == "easy":
        frequencies = [(1, 0), (0, 1)]
        overlap = 0.25
        warp = 0.02
        global_colours = (0,) if rng.random() < 0.25 else ()
    elif difficulty == "medium":
        frequencies = [(1, 1), (2, 0), (0, 2)]
        overlap = 0.15
        warp = 0.06
        global_colours = ()
    elif difficulty == "hard":
        frequencies = [(2, 1), (1, 2), (3, 1), (1, 3)]
        overlap = 0.07
        warp = 0.12
        global_colours = ()
    else:
        raise ValueError(f"unknown difficulty: {difficulty}")
    frequency_u, frequency_v = frequencies[int(rng.integers(len(frequencies)))]
    return SyntheticColourFieldConfig(
        num_colours=3,
        frequency_u=frequency_u,
        frequency_v=frequency_v,
        overlap_fraction=overlap,
        warp_amplitude=warp,
        phase=float(rng.uniform()),
        global_colours=global_colours,
    )


def summarize(
    frontier_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    *,
    elapsed: float,
) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in frontier_rows:
        grouped[(str(row["difficulty"]), int(row["budget"]))].append(row)
    curves = {}
    for (difficulty, budget), rows in grouped.items():
        aware_lengths = [
            float(row["colour_aware_length"])
            for row in rows
            if row["colour_aware_length"] is not None
        ]
        ratios = [
            float(row["colour_aware_length"]) / float(row["workspace_length"])
            for row in rows
            if row["colour_aware_length"] is not None
        ]
        curves.setdefault(difficulty, {})[str(budget)] = {
            "tasks": len(rows),
            "workspace_success_rate": float(np.mean([row["workspace_success"] for row in rows])),
            "colour_aware_success_rate": float(
                np.mean([row["colour_aware_success"] for row in rows])
            ),
            "rescue_rate": float(np.mean([row["rescued"] for row in rows])),
            "mean_colour_aware_length": float(np.mean(aware_lengths)) if aware_lengths else None,
            "mean_colour_aware_to_workspace_length_ratio": (
                float(np.mean(ratios)) if ratios else None
            ),
        }
    by_surface = {}
    surface_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in frontier_rows:
        surface_groups[(str(row["surface_id"]), str(row["difficulty"]), int(row["budget"]))].append(row)
    for (surface, difficulty, budget), rows in surface_groups.items():
        by_surface.setdefault(surface, {}).setdefault(difficulty, {})[str(budget)] = {
            "tasks": len(rows),
            "workspace_success_rate": float(np.mean([row["workspace_success"] for row in rows])),
            "colour_aware_success_rate": float(
                np.mean([row["colour_aware_success"] for row in rows])
            ),
            "rescue_rate": float(np.mean([row["rescued"] for row in rows])),
        }
    field_keys = {
        (row["instance_id"], row["difficulty"], row["field_index"])
        for row in frontier_rows
    }
    workspace_counts = [
        int(row["min_segments"])
        for row in candidate_rows
        if row["workspace_best"] and row["min_segments"] is not None
    ]
    return {
        "instances": len({row["instance_id"] for row in frontier_rows}),
        "surface_types": sorted({row["surface_id"] for row in frontier_rows}),
        "colour_fields": len(field_keys),
        "candidate_evaluations": len(candidate_rows),
        "elapsed_seconds": elapsed,
        "median_workspace_min_segments": float(np.median(workspace_counts)),
        "mean_workspace_min_segments": float(np.mean(workspace_counts)),
        "curves": curves,
        "by_surface": by_surface,
    }


def format_report(summary: dict[str, object]) -> str:
    lines = [
        "# Synthetic 3D colour-liftability benchmark",
        "",
        f"Instances: {summary['instances']}; colour fields: {summary['colour_fields']}; "
        f"candidate evaluations: {summary['candidate_evaluations']}.",
        "",
        "| Difficulty | k | Workspace-first success | Colour-aware success | Rescue rate | Aware/workspace length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for difficulty in DIFFICULTIES:
        for budget, row in sorted(
            summary["curves"][difficulty].items(), key=lambda item: int(item[0])
        ):
            ratio = row["mean_colour_aware_to_workspace_length_ratio"]
            lines.append(
                f"| {difficulty} | {budget} | {row['workspace_success_rate']:.2%} | "
                f"{row['colour_aware_success_rate']:.2%} | {row['rescue_rate']:.2%} | "
                f"{ratio:.4f} |" if ratio is not None else
                f"| {difficulty} | {budget} | {row['workspace_success_rate']:.2%} | "
                f"{row['colour_aware_success_rate']:.2%} | {row['rescue_rate']:.2%} | n/a |"
            )
    return "\n".join(lines) + "\n"


def plot_summary(summary: dict[str, object], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colours = {"easy": "#15803d", "medium": "#2563eb", "hard": "#b91c1c"}
    for difficulty in DIFFICULTIES:
        curve = summary["curves"][difficulty]
        budgets = sorted(int(value) for value in curve)
        workspace = [curve[str(k)]["workspace_success_rate"] for k in budgets]
        aware = [curve[str(k)]["colour_aware_success_rate"] for k in budgets]
        rescue = [curve[str(k)]["rescue_rate"] for k in budgets]
        axes[0].plot(budgets, workspace, marker="o", color=colours[difficulty], label=difficulty)
        axes[1].plot(budgets, aware, marker="o", color=colours[difficulty], label=difficulty)
        axes[2].plot(budgets, rescue, marker="o", color=colours[difficulty], label=difficulty)
    for axis, title in zip(
        axes,
        ("Workspace-first lift success", "Colour-aware candidate success", "Colour-aware rescue rate"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Segment budget k")
        axis.set_ylabel("Rate")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.savefig(output, dpi=180)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
