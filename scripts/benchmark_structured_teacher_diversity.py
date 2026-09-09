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

from diffusion_coverage.coverage import (
    generate_pattern_proposals,
    load_teacher_instance,
    MultiStartStructuredTeacher,
    StructuredTeacherConfig,
    surface_from_teacher_archive,
)
from diffusion_coverage.learning import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot multi-start structured teacher diversity"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances-per-surface", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--initial-sigma-radius", type=float, default=0.30)
    parser.add_argument("--final-sigma-radius", type=float, default=0.03)
    parser.add_argument("--minimum-diversity-radius", type=float, default=0.05)
    parser.add_argument("--maximum-length-ratio", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = select_rows(load_manifest(args.dataset), args.instances_per_surface)
    records: list[dict[str, object]] = []
    for instance_number, row in enumerate(selected, start=1):
        archive = load_teacher_instance(args.dataset / str(row["path"]))
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        config = archive["metadata"]["teacher_config"]
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
        for mode_number, proposal in enumerate(proposals):
            if proposal.name not in available_modes:
                continue
            teacher = MultiStartStructuredTeacher(
                StructuredTeacherConfig(
                    footprint_radius=radius,
                    missed_tolerance=float(config["missed_tolerance"]),
                    restarts=args.restarts,
                    steps_per_restart=args.steps,
                    initial_sigma_radius=args.initial_sigma_radius,
                    final_sigma_radius=args.final_sigma_radius,
                    minimum_diversity_radius=args.minimum_diversity_radius,
                    maximum_length_ratio=args.maximum_length_ratio,
                    seed=args.seed + 1009 * instance_number + mode_number,
                )
            )
            result = teacher.solve(surface, proposal)
            alternatives = result.candidates[1:]
            records.append(
                {
                    "instance_id": str(row["instance_id"]),
                    "surface_id": str(row["surface_id"]),
                    "mode": proposal.name,
                    "evaluated_assignments": result.evaluated_assignments,
                    "template_feasible": result.template.feasible,
                    "retained_candidates": len(result.candidates),
                    "retained_alternatives": len(alternatives),
                    "found_diverse_alternative": bool(alternatives),
                    "maximum_alternative_distance_radius": (
                        max(
                            candidate.distance_from_template_radius
                            for candidate in alternatives
                        )
                        if alternatives
                        else 0.0
                    ),
                    "best_alternative_length_ratio": (
                        min(
                            candidate.metrics.path_length
                            / result.template.metrics.path_length
                            for candidate in alternatives
                        )
                        if alternatives
                        else None
                    ),
                    "minimum_alternative_missed_fraction": (
                        min(candidate.metrics.missed_fraction for candidate in alternatives)
                        if alternatives
                        else None
                    ),
                }
            )
            print(
                f"[{instance_number:02d}/{len(selected):02d}] {row['instance_id']:<24} "
                f"{proposal.name:<24} alternatives={len(alternatives)}",
                flush=True,
            )
    summary = summarize(records)
    summary["by_mode"] = {
        mode: summarize([record for record in records if record["mode"] == mode])
        for mode in sorted({str(record["mode"]) for record in records})
    }
    summary["by_surface"] = {
        surface_id: summarize(
            [record for record in records if record["surface_id"] == surface_id]
        )
        for surface_id in sorted({str(record["surface_id"]) for record in records})
    }
    with (args.output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    protocol = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "metrics.json").write_text(
        json.dumps({"protocol": protocol, "summary": summary}, indent=2) + "\n"
    )
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


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    alternatives = [int(record["retained_alternatives"]) for record in records]
    distances = [
        float(record["maximum_alternative_distance_radius"])
        for record in records
        if record["found_diverse_alternative"]
    ]
    length_ratios = [
        float(record["best_alternative_length_ratio"])
        for record in records
        if record["best_alternative_length_ratio"] is not None
    ]
    return {
        "instance_modes": len(records),
        "modes_with_diverse_alternative": int(
            sum(bool(record["found_diverse_alternative"]) for record in records)
        ),
        "diverse_alternative_rate": float(
            np.mean([record["found_diverse_alternative"] for record in records])
        ),
        "mean_retained_alternatives": float(np.mean(alternatives)),
        "mean_maximum_alternative_distance_radius": (
            float(np.mean(distances)) if distances else None
        ),
        "mean_best_alternative_length_ratio": (
            float(np.mean(length_ratios)) if length_ratios else None
        ),
        "evaluated_assignments": int(
            sum(int(record["evaluated_assignments"]) for record in records)
        ),
    }


if __name__ == "__main__":
    main()
