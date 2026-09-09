#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize contract-corrected Stage 1 FM runs")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--long-train", required=True, type=Path)
    parser.add_argument("--smooth-base", required=True, type=Path)
    parser.add_argument("--representation-audit", required=True, type=Path)
    parser.add_argument("--target-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_run(path: Path) -> dict[str, object]:
    return {
        "history": json.loads((path / "history.json").read_text()),
        "metrics": json.loads((path / "eval_k8_anytime" / "metrics.json").read_text()),
        "path": str(path),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {
        "Segment-preserving, 3.2k": load_run(args.baseline),
        "Long training": load_run(args.long_train),
        "Smooth base (10%)": load_run(args.smooth_base),
    }
    representation = json.loads(args.representation_audit.read_text())
    target_audit = json.loads(args.target_audit.read_text())
    summary = build_summary(runs, representation, target_audit)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = render_report(summary)
    (args.output / "report.md").write_text(report)
    plot_summary(runs, args.output / "stage1_corrected_diagnostics.png")
    print(report)


def build_summary(runs, representation, target_audit):
    comparison = {}
    for name, run in runs.items():
        metrics = run["metrics"]["summary"]
        comparison[name] = {
            "best_validation_loss": min(row["validation_loss"] for row in run["history"]),
            "feasible_rate": metrics["feasible_rate"],
            "mean_missed_fraction": metrics["mean_missed_fraction"],
            "mean_length_ratio_to_teacher": metrics["mean_length_ratio_to_teacher"],
            "mean_total_pipeline_time": metrics["mean_total_pipeline_time"],
            "path": run["path"],
        }
    selected_name = "Smooth base (10%)"
    selected = runs[selected_name]["metrics"]["summary"]
    feasible_gate = 0.9
    length_gate = 1.1
    passed = (
        selected["feasible_rate"] >= feasible_gate
        and selected["mean_length_ratio_to_teacher"] <= length_gate
    )
    return {
        "contract": {
            "teacher_roundtrip_feasible_rate": target_audit["summary"]["feasible_rate"],
            "representation": representation["decision"],
            "old_stage1_runs": "diagnostic_only_due_to_archive_quadrature_and_float32_contract",
        },
        "comparison": comparison,
        "selected_run": selected_name,
        "anytime": selected["anytime_by_samples"],
        "by_surface": selected["by_surface"],
        "teacher": {
            "feasible_rate": 1.0,
            "mean_solve_time": selected["mean_teacher_total_solve_time"],
            "median_solve_time": selected["median_teacher_total_solve_time"],
        },
        "gate": {
            "passed": passed,
            "minimum_feasible_rate": feasible_gate,
            "maximum_mean_length_ratio": length_gate,
            "observed_feasible_rate": selected["feasible_rate"],
            "observed_mean_length_ratio": selected["mean_length_ratio_to_teacher"],
            "diagnosis": (
                "The model learns footprint scaling and recognizable sweep structure, but the "
                "current PointNet/MPNN-style vector field underfits exact sweep spacing and connectors. "
                "Smoother base paths improve feasibility by trading toward substantial over-travel."
            ),
        },
    }


def render_report(summary) -> str:
    lines = [
        "# Contract-Corrected Stage 1 Surface Flow Matching",
        "",
        "The archived surface quadrature is restored exactly, targets are stored in float64, and the "
        "segment-preserving variable-token roundtrip is hard-checked before training.",
        "",
        "## Representation And Data Gates",
        "",
        f"- Teacher target roundtrip feasibility: {summary['contract']['teacher_roundtrip_feasible_rate']:.1%}",
        f"- Default representation: {summary['contract']['representation']['default_representation']}",
        "- Earlier Stage 1 runs are diagnostic-only because they used an invalid archive reconstruction contract.",
        "",
        "## Corrected K=8 Results",
        "",
        "| Run | Best val loss | Feasible | Mean missed | Length / teacher | Mean time |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["comparison"].items():
        lines.append(
            f"| {name} | {row['best_validation_loss']:.4f} | {row['feasible_rate']:.1%} | "
            f"{row['mean_missed_fraction']:.4f} | {row['mean_length_ratio_to_teacher']:.3f} | "
            f"{row['mean_total_pipeline_time']:.3f}s |"
        )
    lines.extend([
        "",
        "## Best-Of-K Anytime (Smooth Base)",
        "",
        "| K | Feasible | Mean missed | Mean time | Median time |",
        "|---:|---:|---:|---:|---:|",
    ])
    for count, row in summary["anytime"].items():
        lines.append(
            f"| {count} | {row['feasible_rate']:.1%} | {row['mean_best_missed_fraction']:.4f} | "
            f"{row['mean_elapsed_time']:.3f}s | {row['median_elapsed_time']:.3f}s |"
        )
    lines.extend([
        "",
        "## Surface Breakdown (Smooth Base, K=8)",
        "",
        "| Surface | Feasible | Mean missed | Length / teacher |",
        "|:---|---:|---:|---:|",
    ])
    for surface, row in summary["by_surface"].items():
        lines.append(
            f"| {surface} | {row['feasible_rate']:.1%} | {row['mean_missed_fraction']:.4f} | "
            f"{row['mean_length_ratio_to_teacher']:.3f} |"
        )
    gate = summary["gate"]
    teacher = summary["teacher"]
    lines.extend([
        "",
        f"Classical teacher: 100% feasible, mean {teacher['mean_solve_time']:.3f}s, "
        f"median {teacher['median_solve_time']:.3f}s.",
        "",
        "## Gate",
        "",
        f"**{'PASS' if gate['passed'] else 'NO-GO'}.** Required at least "
        f"{gate['minimum_feasible_rate']:.0%} feasibility and at most "
        f"{gate['maximum_mean_length_ratio']:.2f}x teacher length; observed "
        f"{gate['observed_feasible_rate']:.1%} and {gate['observed_mean_length_ratio']:.3f}x.",
        "",
        gate["diagnosis"],
        "",
        "Do not proceed to Stage 2 or C-space FM under the current model/data configuration.",
    ])
    return "\n".join(lines) + "\n"


def plot_summary(runs, output: Path) -> None:
    names = list(runs)
    summaries = [runs[name]["metrics"]["summary"] for name in names]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for name, run in runs.items():
        history = run["history"]
        axes[0, 0].plot(
            [row["global_step"] for row in history],
            [row["validation_loss"] for row in history],
            label=name,
            alpha=0.85,
        )
    axes[0, 0].set(title="Validation flow loss", xlabel="Continuation steps", ylabel="Loss")
    axes[0, 0].legend(fontsize=8)

    x = np.arange(len(names))
    axes[0, 1].bar(x, [row["feasible_rate"] for row in summaries], color="#3973ac")
    axes[0, 1].axhline(0.9, color="#b33a3a", linestyle="--", label="gate")
    axes[0, 1].set(title="Best-of-8 hard feasibility", ylabel="Feasible rate", ylim=(0, 1.05))
    axes[0, 1].set_xticks(x, names, rotation=16, ha="right")
    axes[0, 1].legend()

    axes[1, 0].bar(x, [row["mean_length_ratio_to_teacher"] for row in summaries], color="#4b9560")
    axes[1, 0].axhline(1.1, color="#b33a3a", linestyle="--", label="gate")
    axes[1, 0].set(title="Path length ratio", ylabel="Generated / teacher")
    axes[1, 0].set_xticks(x, names, rotation=16, ha="right")
    axes[1, 0].legend()

    selected = summaries[-1]
    anytime = selected["anytime_by_samples"]
    counts = [int(value) for value in anytime]
    axes[1, 1].plot(
        counts, [anytime[str(value)]["feasible_rate"] for value in counts], marker="o"
    )
    axes[1, 1].set(title="Smooth-base anytime", xlabel="Samples checked", ylabel="Feasible rate", ylim=(0, 1.05))
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
