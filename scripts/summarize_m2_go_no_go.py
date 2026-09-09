#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fixed-r, k=1 M2 go/no-go runs")
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run1 = json.loads((args.run1 / "eval_k8" / "metrics.json").read_text())
    run2 = json.loads((args.run2 / "eval_k8_refined" / "metrics.json").read_text())
    history1 = json.loads((args.run1 / "history.json").read_text())
    history2 = json.loads((args.run2 / "history.json").read_text())

    rows = [
        summarize("FM iid base", run1["instances"], raw=False),
        summarize("FM smooth base", run2["instances"], raw=True),
        summarize("FM smooth + repair", run2["instances"], raw=False),
        summarize_teacher(run2["instances"]),
    ]
    payload = {"decision": "NO-GO", "epsilon": 0.05, "methods": rows}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(args.output / "report.md", rows)
    plot_losses(history1, history2, args.output / "training_loss.png")
    plot_surface_misses(run2["instances"], args.output / "missed_fraction_by_surface.png")
    print_table(rows)
    print(f"\nM2 decision: NO-GO\nReport: {args.output / 'report.md'}")


def summarize(name: str, records: list[dict[str, object]], *, raw: bool) -> dict[str, object]:
    prefix = "raw_" if raw else ""
    missed = np.asarray([record[f"{prefix}best_missed_fraction"] for record in records], dtype=float)
    lengths = np.asarray([record[f"{prefix}best_path_length"] for record in records], dtype=float)
    return {
        "method": name,
        "instances": len(records),
        "feasible_rate": float(np.mean(missed <= 0.05 + 1e-12)),
        "mean_missed_fraction": float(missed.mean()),
        "median_missed_fraction": float(np.median(missed)),
        "mean_path_length": float(lengths.mean()),
        "median_path_length": float(np.median(lengths)),
    }


def summarize_teacher(records: list[dict[str, object]]) -> dict[str, object]:
    missed = np.asarray([record["teacher_missed_fraction"] for record in records], dtype=float)
    lengths = np.asarray([record["teacher_path_length"] for record in records], dtype=float)
    return {
        "method": "Classical teacher",
        "instances": len(records),
        "feasible_rate": float(np.mean(missed <= 0.05 + 1e-12)),
        "mean_missed_fraction": float(missed.mean()),
        "median_missed_fraction": float(np.median(missed)),
        "mean_path_length": float(lengths.mean()),
        "median_path_length": float(np.median(lengths)),
    }


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# M2 fixed-r, k=1 go/no-go\n",
        "Validation uses 10 held-out instances, stratified as two per 3D surface family. "
        "Each neural row uses top-8 Flow Matching samples and the same exact mesh projection "
        "and geodesic finite-footprint evaluator as the teacher.\n",
        "| Method | Feasible | Mean missed | Median missed | Mean length |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {100 * float(row['feasible_rate']):.1f}% | "
            f"{float(row['mean_missed_fraction']):.4f} | "
            f"{float(row['median_missed_fraction']):.4f} | {float(row['mean_path_length']):.2f} |"
        )
    lines.extend(
        [
            "\n## Decision\n",
            "**NO-GO.** Raw FM feasibility is 10%. A 12-step coverage repair raises it to 60%, "
            "but cylinder and torus remain infeasible. Do not proceed to radius/segment conditioning "
            "or manipulator integration until sequence alignment and closed-surface modes are fixed.\n",
            "## Diagnosis\n",
            "Teacher targets for one instance mix spiral, orthogonal raster, and phase-shifted paths. "
            "Fixed waypoint-index regression therefore has severe correspondence ambiguity. Smooth base "
            "noise removes high-frequency jaggedness but does not recover missing coverage bands.\n",
        ]
    )
    path.write_text("\n".join(lines))


def plot_losses(history1: list[dict[str, float]], history2: list[dict[str, float]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot([x["epoch"] for x in history1], [x["validation_loss"] for x in history1], label="iid base")
    axis.plot([x["epoch"] for x in history2], [x["validation_loss"] for x in history2], label="smooth base")
    axis.set(xlabel="Epoch", ylabel="Validation CFM loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_surface_misses(records: list[dict[str, object]], output: Path) -> None:
    labels = [str(record["instance_id"]) for record in records]
    raw = [float(record["raw_best_missed_fraction"]) for record in records]
    refined = [float(record["best_missed_fraction"]) for record in records]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(x - 0.2, raw, width=0.4, label="Raw FM")
    axis.bar(x + 0.2, refined, width=0.4, label="After repair")
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="epsilon=0.05")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set_ylabel("Missed coverage fraction")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def print_table(rows: list[dict[str, object]]) -> None:
    print("Method                    Feasible   Mean missed   Mean length")
    print("---------------------------------------------------------------")
    for row in rows:
        print(
            f"{str(row['method']):<25} {100 * float(row['feasible_rate']):7.1f}% "
            f"{float(row['mean_missed_fraction']):13.4f} {float(row['mean_path_length']):13.2f}"
        )


if __name__ == "__main__":
    main()
