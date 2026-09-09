#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact segment-budget frontiers from UR5e IK components"
    )
    parser.add_argument("--graphs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def component_coverage(component_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(component_labels, dtype=np.int64)
    reachable = np.any(labels >= 0, axis=1)
    valid = labels[labels >= 0]
    num_components = 0 if not len(valid) else int(valid.max()) + 1
    coverage = np.zeros((labels.shape[0], num_components), dtype=np.float64)
    for node, layer in enumerate(labels):
        coverage[node, np.unique(layer[layer >= 0])] = 1.0
    return coverage, reachable


def maximum_covered_nodes(coverage: np.ndarray, reachable: np.ndarray, budget: int) -> int:
    _, covered = optimal_component_selection(coverage, reachable, budget)
    return int(covered.sum())


def optimal_component_selection(
    coverage: np.ndarray, reachable: np.ndarray, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    matrix = coverage[reachable]
    num_nodes, num_components = matrix.shape
    if num_nodes == 0 or num_components == 0:
        return np.zeros(num_components, dtype=bool), np.zeros(coverage.shape[0], dtype=bool)
    # Variables are component selections y followed by covered-node indicators z.
    objective = np.concatenate((np.zeros(num_components), -np.ones(num_nodes)))
    link = np.column_stack((-matrix, np.eye(num_nodes)))
    budget_row = np.concatenate((np.ones(num_components), np.zeros(num_nodes)))[None, :]
    constraints = [
        LinearConstraint(link, -np.inf, 0.0),
        LinearConstraint(budget_row, -np.inf, float(budget)),
    ]
    result = milp(
        objective,
        integrality=np.ones(num_components + num_nodes),
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"maximum-coverage MILP failed: {result.message}")
    selected = result.x[:num_components] > 0.5
    covered = np.zeros(coverage.shape[0], dtype=bool)
    covered[reachable] = result.x[num_components:] > 0.5
    return selected, covered


def minimum_cover_segments(coverage: np.ndarray, reachable: np.ndarray) -> int | None:
    matrix = coverage[reachable]
    if not matrix.shape[0]:
        return 0
    if not matrix.shape[1]:
        return None
    result = milp(
        np.ones(matrix.shape[1]),
        integrality=np.ones(matrix.shape[1]),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, 1.0, np.inf),
        options={"presolve": True},
    )
    if not result.success:
        return None
    return int(round(result.fun))


def analyze_directory(path: Path, label: str, max_segments: int) -> list[dict]:
    rows = []
    for archive_path in sorted((path / "instances").glob("*.npz")):
        with np.load(archive_path, allow_pickle=False) as archive:
            coverage, reachable = component_coverage(archive["component_labels"])
            metadata = json.loads(str(archive["metadata_json"]))
        denominator = int(reachable.sum())
        row = {
            "condition": label,
            "instance_id": archive_path.stem,
            "surface_id": metadata["surface_id"],
            "reachable_nodes": denominator,
            "components": int(coverage.shape[1]),
            "minimum_cover_segments": minimum_cover_segments(coverage, reachable),
        }
        for budget in range(1, max_segments + 1):
            covered = maximum_covered_nodes(coverage, reachable, budget)
            row[f"covered_k{budget}"] = covered
            row[f"fraction_k{budget}"] = 0.0 if denominator == 0 else covered / denominator
        rows.append(row)
    return rows


def summarize(rows: list[dict], max_segments: int) -> dict:
    result = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        result[condition] = {
            "instances": len(selected),
            "mean_minimum_cover_segments": float(
                np.mean([row["minimum_cover_segments"] for row in selected])
            ),
            "mean_coverage_frontier": {
                str(budget): float(np.mean([row[f"fraction_k{budget}"] for row in selected]))
                for budget in range(1, max_segments + 1)
            },
        }
    return result


def save_plot(summary: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for condition, values in summary.items():
        x = np.asarray([int(key) for key in values["mean_coverage_frontier"]])
        y = np.asarray(list(values["mean_coverage_frontier"].values()))
        ax.plot(x, y, marker="o", label=condition)
    ax.set(xlabel="Segment budget k", ylabel="Mean reachable-node coverage", ylim=(0.0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.graphs) != len(args.labels):
        raise ValueError("--graphs and --labels must have the same length")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, label in zip(args.graphs, args.labels):
        rows.extend(analyze_directory(path, label, args.max_segments))
    summary = summarize(rows, args.max_segments)
    fields = list(rows[0]) if rows else []
    with (args.output / "frontier.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_plot(summary, args.output / "component_frontier.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
