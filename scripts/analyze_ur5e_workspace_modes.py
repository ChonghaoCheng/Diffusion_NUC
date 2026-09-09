#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified instance-bootstrap analysis for the UR5e workspace-mode audit"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rows = [
        json.loads(line)
        for line in (args.results / "instance_results.jsonl").read_text().splitlines()
        if line
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["surface_id"])].append(row)
    rng = np.random.default_rng(args.seed)
    document = {
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "resampling_unit": "complete instance, stratified by surface for overall estimates",
        "overall": bootstrap_summary(grouped, args.bootstrap_samples, rng),
        "by_surface": {
            surface: bootstrap_summary(
                {surface: surface_rows}, args.bootstrap_samples, rng
            )
            for surface, surface_rows in sorted(grouped.items())
        },
    }
    (args.results / "bootstrap.json").write_text(json.dumps(document, indent=2) + "\n")
    (args.results / "bootstrap.md").write_text(render_markdown(document))
    print((args.results / "bootstrap.md").read_text())


def bootstrap_summary(grouped, samples: int, rng: np.random.Generator):
    observed = [row for rows in grouped.values() for row in rows]
    draws = {name: [] for name in metric_values(observed)}
    for _ in range(samples):
        resampled = []
        for rows in grouped.values():
            indices = rng.integers(0, len(rows), size=len(rows))
            resampled.extend(rows[int(index)] for index in indices)
        for name, value in metric_values(resampled).items():
            draws[name].append(value)
    return {
        "instances": len(observed),
        **{
            name: {
                "estimate": value,
                "ci95": [
                    float(np.quantile(draws[name], 0.025)),
                    float(np.quantile(draws[name], 0.975)),
                ],
            }
            for name, value in metric_values(observed).items()
        },
    }


def metric_values(rows):
    return {
        "geometry_best_success_rate": float(
            np.mean([row["geometry_best_liftable"] for row in rows])
        ),
        "best_of_mode_success_rate": float(
            np.mean([row["best_of_mode_liftable"] for row in rows])
        ),
        "paired_mode_rescue_rate": float(np.mean([row["mode_rescue"] for row in rows])),
    }


def render_markdown(document) -> str:
    lines = [
        "# Fixed-placement UR5e workspace-mode bootstrap",
        "",
        "Intervals resample complete instances and preserve surface sample counts.",
        "",
        "| Group | N | Geometry-best | Best-of-mode | Paired rescue |",
        "|---|---:|---:|---:|---:|",
    ]
    groups = {"overall": document["overall"], **document["by_surface"]}
    for name, row in groups.items():
        values = []
        for key in (
            "geometry_best_success_rate",
            "best_of_mode_success_rate",
            "paired_mode_rescue_rate",
        ):
            metric = row[key]
            values.append(
                f"{metric['estimate']:.2%} [{metric['ci95'][0]:.2%}, {metric['ci95'][1]:.2%}]"
            )
        lines.append(f"| {name} | {row['instances']} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
