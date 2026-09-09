#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-bootstrap synthetic liftability results by 3D instance"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rows = [
        json.loads(line)
        for line in (args.results / "frontier_results.jsonl").read_text().splitlines()
        if line
    ]
    rng = np.random.default_rng(args.seed)
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["difficulty"]), int(row["budget"]))].append(row)

    analysis = {}
    for (difficulty, budget), group in sorted(groups.items()):
        instance_values = aggregate_instances(group)
        workspace = np.asarray([value["workspace_success"] for value in instance_values])
        aware = np.asarray([value["colour_aware_success"] for value in instance_values])
        rescue = aware - workspace
        ratios = np.asarray([value["length_ratio"] for value in instance_values])
        analysis.setdefault(difficulty, {})[str(budget)] = {
            "instances": len(instance_values),
            "workspace_success_rate": estimate(workspace, rng, args.bootstrap_samples),
            "colour_aware_success_rate": estimate(aware, rng, args.bootstrap_samples),
            "paired_rescue_rate": estimate(rescue, rng, args.bootstrap_samples),
            "aware_to_workspace_length_ratio": estimate(
                ratios[np.isfinite(ratios)], rng, args.bootstrap_samples
            ),
        }

    selected = (("easy", 4), ("medium", 16), ("hard", 32))
    transitions = {}
    for difficulty, budget in selected:
        rescued = [
            row
            for row in groups[(difficulty, budget)]
            if bool(row["rescued"])
        ]
        transitions[f"{difficulty}_k{budget}"] = {
            "rescued_tasks": len(rescued),
            "mode_transitions": dict(
                Counter(
                    f"{row['workspace_mode']} -> {row['colour_aware_mode']}"
                    for row in rescued
                ).most_common()
            ),
        }

    document = {
        "protocol": {
            "bootstrap_unit": "3D surface instance",
            "bootstrap_samples": args.bootstrap_samples,
            "interval": "percentile 95% cluster bootstrap",
            "seed": args.seed,
        },
        "curves": analysis,
        "selected_mode_transitions": transitions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cluster_bootstrap.json").write_text(json.dumps(document, indent=2) + "\n")
    report = format_report(document, selected)
    (args.output / "cluster_bootstrap.md").write_text(report)
    print(report)


def aggregate_instances(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["instance_id"])].append(row)
    values = []
    for instance_rows in grouped.values():
        feasible_ratios = [
            float(row["colour_aware_length"]) / float(row["workspace_length"])
            for row in instance_rows
            if row["colour_aware_length"] is not None
        ]
        values.append(
            {
                "workspace_success": float(
                    np.mean([row["workspace_success"] for row in instance_rows])
                ),
                "colour_aware_success": float(
                    np.mean([row["colour_aware_success"] for row in instance_rows])
                ),
                "length_ratio": (
                    float(np.mean(feasible_ratios)) if feasible_ratios else float("nan")
                ),
            }
        )
    return values


def estimate(values: np.ndarray, rng: np.random.Generator, samples: int) -> dict[str, object]:
    if len(values) == 0:
        return {"mean": None, "bootstrap_95": [None, None]}
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "bootstrap_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def format_report(document: dict[str, object], selected) -> str:
    lines = [
        "# Synthetic liftability cluster-bootstrap diagnosis",
        "",
        "Intervals resample complete 3D instances, retaining their repeated colour fields.",
        "",
        "| Difficulty | k | Workspace success | Colour-aware success | Paired rescue | Length ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for difficulty, budget in selected:
        row = document["curves"][difficulty][str(budget)]
        fields = []
        for name in (
            "workspace_success_rate",
            "colour_aware_success_rate",
            "paired_rescue_rate",
            "aware_to_workspace_length_ratio",
        ):
            estimate_row = row[name]
            fields.append(
                f"{estimate_row['mean']:.4f} [{estimate_row['bootstrap_95'][0]:.4f}, "
                f"{estimate_row['bootstrap_95'][1]:.4f}]"
            )
        lines.append(
            f"| {difficulty} | {budget} | {fields[0]} | {fields[1]} | "
            f"{fields[2]} | {fields[3]} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
