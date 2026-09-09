#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot expanded Stage 2 multi-seed results")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    metrics = [
        load_json(run / "eval_validation_k8" / "metrics.json")["summary"]
        for run in args.run
    ]
    diversity = [
        load_json(run / "eval_validation_k8" / "control_diversity.json")["summary"]
        for run in args.run
    ]
    labels = [f"Seed {index}" for index in range(len(args.run))]
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axes[0, 0].bar(
        positions,
        [row["candidate_feasible_rate"] for row in metrics],
        color="#2563eb",
    )
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel("Hard-feasible candidate rate")
    axes[0, 0].set_title("Single-sample reliability")

    ks = [1, 2, 4, 8]
    top_k = np.asarray(
        [
            [row["top_k_mode_recovery"][str(k)]["mode_recovery_rate"] for k in ks]
            for row in metrics
        ]
    )
    for label, values in zip(labels, top_k):
        axes[0, 1].plot(ks, values, marker="o", alpha=0.45, label=label)
    axes[0, 1].plot(ks, top_k.mean(axis=0), color="black", marker="o", linewidth=2, label="Mean")
    axes[0, 1].set_xticks(ks)
    axes[0, 1].set_ylim(0.7, 1.01)
    axes[0, 1].set_xlabel("K samples per conditioned mode")
    axes[0, 1].set_ylabel("Mode recovery")
    axes[0, 1].set_title("Best-of-K recovery")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(
        positions,
        [row["mean_feasible_length_ratio"] for row in metrics],
        color="#15803d",
    )
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].set_ylim(0.98, max(row["mean_feasible_length_ratio"] for row in metrics) + 0.02)
    axes[1, 0].set_ylabel("Generated / best teacher length")
    axes[1, 0].set_title("Hard-feasible path quality")

    thresholds = np.asarray(
        [row["threshold_radius"] for row in diversity[0]["threshold_curve"]]
    )
    generated = np.asarray(
        [
            [row["mean_generated_teacher_coverage"] for row in summary["threshold_curve"]]
            for summary in diversity
        ]
    )
    teacher = np.asarray(
        [
            [row["mean_teacher_leave_one_out_coverage"] for row in summary["threshold_curve"]]
            for summary in diversity
        ]
    )
    axes[1, 1].plot(
        thresholds,
        generated.mean(axis=0),
        color="#7c3aed",
        marker="o",
        label="Generated -> teacher",
    )
    axes[1, 1].fill_between(
        thresholds, generated.min(axis=0), generated.max(axis=0), color="#7c3aed", alpha=0.15
    )
    axes[1, 1].plot(
        thresholds,
        teacher.mean(axis=0),
        color="#4b5563",
        marker="s",
        label="Teacher leave-one-out",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xticks(thresholds, [f"{value:g}" for value in thresholds])
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_xlabel("Residual-control threshold (footprint radii)")
    axes[1, 1].set_ylabel("Coverage")
    axes[1, 1].set_title("Finite-teacher distribution diagnostic")
    axes[1, 1].legend(fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
