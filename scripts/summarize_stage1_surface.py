#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 1 variable-token FM diagnostics")
    parser.add_argument("--local-run", required=True, type=Path)
    parser.add_argument("--attention-run", required=True, type=Path)
    parser.add_argument("--best-target-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {
        "Local, all targets": load_run(args.local_run),
        "Self-attention, all targets": load_run(args.attention_run),
        "Self-attention, best target": load_run(args.best_target_run),
    }
    summary = build_summary(runs)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "report.md").write_text(render_report(summary))
    plot_diagnostics(runs, args.output / "stage1_diagnostics.png")
    print(render_report(summary))


def load_run(path: Path) -> dict[str, object]:
    history = json.loads((path / "history.json").read_text())
    metrics_path = next(
        candidate for candidate in (
            path / "eval_k8_anytime" / "metrics.json",
            path / "eval_k8_raw_scaling" / "metrics.json",
            path / "eval_k8_raw" / "metrics.json",
        )
        if candidate.exists()
    )
    metrics = json.loads(metrics_path.read_text())
    return {"history": history, "metrics": metrics, "metrics_path": str(metrics_path)}


def build_summary(runs: dict[str, dict[str, object]]) -> dict[str, object]:
    comparison = {}
    for name, run in runs.items():
        metrics = run["metrics"]["summary"]
        instances = run["metrics"]["instances"]
        length_ratio = metrics.get(
            "mean_length_ratio_to_teacher",
            float(np.mean([
                row["best_path_length"] / row["teacher_path_length"] for row in instances
            ])),
        )
        comparison[name] = {
            "best_validation_loss": min(row["validation_loss"] for row in run["history"]),
            "raw_feasible_rate": metrics["raw_feasible_rate"],
            "mean_missed_fraction": metrics["mean_missed_fraction"],
            "mean_length_ratio_to_teacher": length_ratio,
            "metrics_path": run["metrics_path"],
        }
    best = runs["Self-attention, best target"]["metrics"]["summary"]
    return {
        "comparison": comparison,
        "best_model_anytime": best["anytime_by_samples"],
        "best_model_by_surface": best["by_surface"],
        "classical_teacher": {
            "feasible_rate": 1.0,
            "mean_total_solve_time": best["mean_teacher_total_solve_time"],
            "median_total_solve_time": best["median_teacher_total_solve_time"],
        },
        "gate": {
            "passed": False,
            "reason": (
                "Best raw FM reaches 30% feasibility in 6.27 s on average, while the "
                "classical teacher supplies feasible plans in 4.67 s on average."
            ),
            "supported_regime": "hemisphere",
        },
    }


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# Stage 1 Variable-Token Surface FM",
        "",
        "## Model Ablations",
        "",
        "| Model | Best val loss | Raw feasible | Mean missed | Length / teacher |",
        "|:---|---:|---:|---:|---:|",
    ]
    for name, row in summary["comparison"].items():
        lines.append(
            f"| {name} | {row['best_validation_loss']:.4f} | {row['raw_feasible_rate']:.1%} "
            f"| {row['mean_missed_fraction']:.4f} | {row['mean_length_ratio_to_teacher']:.3f} |"
        )
    lines.extend([
        "",
        "## Best-of-M Anytime",
        "",
        "| Samples | Feasible | Mean missed | Mean time | Median time |",
        "|---:|---:|---:|---:|---:|",
    ])
    for samples, row in summary["best_model_anytime"].items():
        lines.append(
            f"| {samples} | {row['feasible_rate']:.1%} | {row['mean_best_missed_fraction']:.4f} "
            f"| {row['mean_elapsed_time']:.3f}s | {row['median_elapsed_time']:.3f}s |"
        )
    lines.extend([
        "",
        "## Surface Breakdown",
        "",
        "| Surface | Feasible | Mean missed | r-length corr. | Teacher corr. |",
        "|:---|---:|---:|---:|---:|",
    ])
    for surface, row in summary["best_model_by_surface"].items():
        lines.append(
            f"| {surface} | {row['feasible_rate']:.1%} | {row['mean_missed_fraction']:.4f} "
            f"| {row['log_radius_log_length_correlation']:.3f} "
            f"| {row['teacher_log_radius_log_length_correlation']:.3f} |"
        )
    teacher = summary["classical_teacher"]
    lines.extend([
        "",
        f"Classical teacher: 100% feasible, mean solve time {teacher['mean_total_solve_time']:.3f}s, "
        f"median {teacher['median_total_solve_time']:.3f}s.",
        "",
        "## Gate",
        "",
        f"**NO-GO.** {summary['gate']['reason']} Hemisphere is the only currently supported amortization regime.",
    ])
    return "\n".join(lines) + "\n"


def plot_diagnostics(runs: dict[str, dict[str, object]], output: Path) -> None:
    best_metrics = runs["Self-attention, best target"]["metrics"]
    summary = best_metrics["summary"]
    records = best_metrics["instances"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for name, run in runs.items():
        history = run["history"]
        axes[0, 0].plot(
            [row["global_step"] for row in history],
            [row["validation_loss"] for row in history],
            label=name,
        )
    axes[0, 0].set(title="Validation loss", xlabel="Optimizer steps", ylabel="Flow loss")
    axes[0, 0].legend(fontsize=8)

    anytime = summary["anytime_by_samples"]
    sample_counts = np.asarray([int(value) for value in anytime])
    axes[0, 1].plot(
        [anytime[str(value)]["mean_elapsed_time"] for value in sample_counts],
        [anytime[str(value)]["feasible_rate"] for value in sample_counts],
        marker="o", label="FM + hard checker",
    )
    axes[0, 1].scatter(
        [summary["mean_teacher_total_solve_time"]], [1.0], marker="x", s=90,
        label="Classical teacher (final)",
    )
    axes[0, 1].set(title="Feasible rate vs wall time", xlabel="Mean time (s)", ylabel="Feasible rate", ylim=(0, 1.05))
    axes[0, 1].legend(fontsize=8)

    surfaces = list(summary["by_surface"])
    axes[1, 0].bar(surfaces, [summary["by_surface"][surface]["feasible_rate"] for surface in surfaces])
    axes[1, 0].set(title="Best-of-8 feasibility", ylabel="Feasible rate", ylim=(0, 1.05))
    axes[1, 0].tick_params(axis="x", rotation=20)

    for surface in surfaces:
        selected = [row for row in records if row["surface_id"] == surface]
        axes[1, 1].scatter(
            [row["footprint_radius"] for row in selected],
            [row["best_path_length"] for row in selected],
            label=surface,
        )
    axes[1, 1].set(title="Footprint conditioning", xlabel="Footprint radius", ylabel="Generated path length")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
