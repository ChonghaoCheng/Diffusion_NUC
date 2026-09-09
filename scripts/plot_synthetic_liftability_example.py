#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    inverse_surface_parameters,
    load_teacher_instance,
    surface_from_teacher_archive,
)
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.liftability import SyntheticColourField, SyntheticColourFieldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one rescued 3D synthetic liftability task")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--surface", default="freeform_patch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frontier = [
        json.loads(line)
        for line in (args.results / "frontier_results.jsonl").read_text().splitlines()
        if line
    ]
    examples = [
        row
        for row in frontier
        if row["difficulty"] == args.difficulty
        and int(row["budget"]) == args.budget
        and row["surface_id"] == args.surface
        and row["rescued"]
    ]
    if not examples:
        raise ValueError("no rescued task matches the requested filters")
    example = examples[0]
    candidate_rows = [
        json.loads(line)
        for line in (args.results / "candidate_results.jsonl").read_text().splitlines()
        if line
    ]
    matching = [
        row
        for row in candidate_rows
        if row["instance_id"] == example["instance_id"]
        and row["difficulty"] == example["difficulty"]
        and row["field_index"] == example["field_index"]
    ]
    workspace = next(row for row in matching if row["workspace_best"])
    aware = min(
        (row for row in matching if int(row["min_segments"]) <= args.budget),
        key=lambda row: float(row["path_length"]),
    )

    manifest = {str(row["instance_id"]): row for row in load_manifest(args.dataset)}
    archive = load_teacher_instance(args.dataset / str(manifest[example["instance_id"]]["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(example["surface_id"]))
    config_values = dict(example["field_config"])
    config_values["global_colours"] = tuple(config_values.get("global_colours", ()))
    field = SyntheticColourField(SyntheticColourFieldConfig(**config_values))
    uv = inverse_surface_parameters(surface, surface.sample_points, unwrap_periodic=False)
    colour_mask = field.valid_colours(uv)
    display_colour = np.argmax(colour_mask, axis=1)
    palette = np.asarray(["#2563eb", "#dc2626", "#16a34a", "#7c3aed"])

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    for panel, (row, title) in enumerate(
        ((workspace, "Workspace-shortest: not liftable"), (aware, "Colour-aware selection: rescued")),
        start=1,
    ):
        axis = fig.add_subplot(1, 2, panel, projection="3d")
        axis.scatter(
            surface.sample_points[:, 0],
            surface.sample_points[:, 1],
            surface.sample_points[:, 2],
            c=palette[display_colour],
            s=5,
            alpha=0.32,
            depthshade=False,
        )
        index = int(row["candidate_index"])
        plan = CoveragePlan(
            np.asarray(archive["candidate_waypoints"])[index],
            np.asarray(archive["candidate_segment_mask"])[index],
            np.asarray(archive["candidate_waypoint_mask"])[index],
        )
        for path in plan.active_paths():
            axis.plot(path[:, 0], path[:, 1], path[:, 2], color="black", linewidth=1.4)
        axis.set_title(
            f"{title}\n{row['mode']}, segments={row['min_segments']}, length={row['path_length']:.2f}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_box_aspect(np.ptp(surface.vertices, axis=0))
        axis.view_init(elev=28, azim=-55)
    fig.suptitle(
        f"{example['instance_id']} | {args.difficulty} colour field | segment budget k={args.budget}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    metadata = {"frontier": example, "workspace": workspace, "colour_aware": aware}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
