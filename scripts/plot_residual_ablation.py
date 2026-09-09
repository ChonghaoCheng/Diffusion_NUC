#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot absolute-control versus residual ablation")
    parser.add_argument("--absolute", type=Path, action="append", required=True)
    parser.add_argument("--residual", type=Path, action="append", required=True)
    parser.add_argument("--residual-diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(run: Path) -> dict:
    document = json.loads((run / "eval_validation_k8" / "metrics.json").read_text())
    return document.get("summary", document)


def main() -> None:
    args = parse_args()
    absolute = [load_metrics(path) for path in args.absolute]
    residual = [load_metrics(path) for path in args.residual]
    diagnosis = json.loads(args.residual_diagnosis.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    groups = [absolute, residual]
    labels = ["Absolute UV", "Template residual"]
    colors = ["#6b7280", "#15803d"]
    for index, (metrics, label, color) in enumerate(zip(groups, labels, colors)):
        values = [row["candidate_feasible_rate"] for row in metrics]
        axes[0, 0].bar(index, np.mean(values), color=color, width=0.6)
        axes[0, 0].scatter([index] * len(values), values, color="black", zorder=3)
    axes[0, 0].set_xticks(range(2), labels)
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel("Candidate hard-feasible rate")
    axes[0, 0].set_title("Three-seed feasibility")

    for metrics, label, color in zip(groups, labels, colors):
        ks = [1, 2, 4, 8]
        means = [
            np.mean([row["top_k_mode_recovery"][str(k)]["mode_recovery_rate"] for row in metrics])
            for k in ks
        ]
        axes[0, 1].plot(ks, means, marker="o", label=label, color=color)
    axes[0, 1].set_xticks([1, 2, 4, 8])
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_xlabel("K samples per mode")
    axes[0, 1].set_ylabel("Mode recovery")
    axes[0, 1].set_title("Top-K recovery")
    axes[0, 1].legend()

    modes = list(residual[0]["by_mode"])
    positions = np.arange(len(modes))
    width = 0.36
    for offset, metrics, label, color in zip((-width / 2, width / 2), groups, labels, colors):
        means = [
            np.mean([row["by_mode"][mode]["mode_recovery_rate"] for row in metrics])
            for mode in modes
        ]
        axes[1, 0].bar(positions + offset, means, width=width, label=label, color=color)
    axes[1, 0].set_xticks(positions, [mode.replace("_phase_", "\n") for mode in modes], rotation=20)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel("Mean K=8 recovery")
    axes[1, 0].set_title("Recovery by structured mode")
    axes[1, 0].legend()

    ratios = [row["mean_best_generated_to_template_length_ratio"] for row in diagnosis["runs"]]
    axes[1, 1].bar(range(len(ratios)), ratios, color="#15803d", width=0.6)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Template length")
    axes[1, 1].set_xticks(range(len(ratios)), [f"Seed {i}" for i in range(len(ratios))])
    axes[1, 1].set_ylim(min(0.9, min(ratios) - 0.02), 1.02)
    axes[1, 1].set_ylabel("Best generated / template length")
    axes[1, 1].set_title("Residual proposals shorten templates")
    axes[1, 1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
