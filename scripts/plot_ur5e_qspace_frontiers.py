#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot strict UR5e q-space teacher frontiers")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Curves formatted as LABEL=RESULT_DIRECTORY",
    )
    parser.add_argument("--surface", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves = []
    for specification in args.results:
        label, separator, raw_path = specification.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"invalid result specification: {specification}")
        path = Path(raw_path) / "instances.jsonl"
        all_rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [row for row in all_rows if row["surface_id"] == args.surface]
        if not rows:
            raise ValueError(f"no {args.surface} rows in {path}")
        curves.append((label, sorted(rows, key=lambda row: row["max_segments"])))

    panels = (
        ("covered_node_fraction", "Covered graph nodes", 100.0),
        ("missed_fraction", "Missed surface area", 100.0),
        ("task_path_length", "Actual FK path length [m]", 1.0),
        ("joint_travel", "Cumulative joint travel [rad]", 1.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    for axis, (field, title, scale) in zip(axes.ravel(), panels):
        for label, rows in curves:
            axis.plot(
                [row["max_segments"] for row in rows],
                [row[field] * scale for row in rows],
                marker="o",
                linewidth=1.6,
                label=label,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if scale == 100.0:
            axis.set_ylabel("percent")
    for axis in axes[-1]:
        axis.set_xlabel("segment budget k")
    axes[0, 0].legend()
    figure.suptitle(f"Strict q-space teacher frontier: {args.surface}")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
