#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 1 single-family ablations")
    parser.add_argument("--target-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    return parser.parse_args()


def label(path: Path) -> str:
    names = {
        "flow_matching_stage1_single_family_raster_u_run2_roundtrip": "Velocity only",
        "flow_matching_stage1_single_family_objective_loss_run1": "Coverage-heavy",
        "flow_matching_stage1_single_family_objective_loss_run2_balanced": "Coverage + length",
        "flow_matching_stage1_single_family_tangent_run1": "+ tangent",
        "flow_matching_stage1_single_family_surface_consistency_run1": "+ surface",
    }
    return names.get(path.name, path.name)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.target_audit.read_text())["summary"]
    runs = {}
    for path in args.runs:
        metrics = json.loads((path / "eval_k8_anytime" / "metrics.json").read_text())["summary"]
        history = json.loads((path / "history.json").read_text())
        runs[label(path)] = {
            "path": str(path),
            "best_validation_loss": min(row["validation_loss"] for row in history),
            **metrics,
        }
    summary = {
        "candidate_name": "raster_u_phase_0.00",
        "curated_instances": 178,
        "validation_target_audit": audit,
        "runs": runs,
        "gate": {"minimum_feasible_rate": 0.9, "maximum_length_ratio": 1.1},
    }
    summary["gate"]["passed"] = any(
        row["feasible_rate"] >= 0.9 and row["mean_length_ratio_to_teacher"] <= 1.1
        for row in runs.values()
    )
    report = render_report(summary)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "report.md").write_text(report)
    plot(summary, args.output / "single_family_objective_ablation.png")
    print(report)


def render_report(summary: dict[str, object]) -> str:
    audit = summary["validation_target_audit"]
    lines = [
        "# Stage 1 Canonical Single-Family Ablation",
        "",
        "This experiment fixes the exact expert proposal to `raster_u_phase_0.00` before "
        "the instance split. Every retained candidate passes the complete variable-token "
        "model-input roundtrip.",
        "",
        "## Data Gate",
        "",
        f"- Curated instances: {summary['curated_instances']}",
        f"- Validation targets: {audit['instances']}",
        f"- Validation target hard feasibility: {audit['feasible_rate']:.1%}",
        f"- Mean target roundtrip length ratio: {audit['mean_length_ratio']:.4f}x",
        "",
        "## Best-Of-8 Hard Results",
        "",
        "| Run | Feasible | Mean missed | Length / same-family teacher | Mean time |",
        "|:---|---:|---:|---:|---:|",
    ]
    for name, row in summary["runs"].items():
        lines.append(
            f"| {name} | {row['feasible_rate']:.1%} | {row['mean_missed_fraction']:.4f} | "
            f"{row['mean_length_ratio_to_teacher']:.3f} | {row['mean_total_pipeline_time']:.3f}s |"
        )
    lines.extend([
        "",
        "## Diagnosis",
        "",
        "Fixing path family and phase improves the velocity-only model relative to mixed targets, "
        "but does not pass the gate. A coverage-heavy surrogate increases feasibility by adding "
        "travel. A calibrated excess-length term recovers much of that travel while retaining "
        "feasibility. Tangent and local surface-consistency terms move the same tradeoff but do not "
        "resolve the repeated cylinder and hemisphere failures.",
        "",
        "The remaining bottleneck is structural: raw dense XYZ tokens do not reliably preserve "
        "ordered sweep/stroke topology. Further scalar loss-weight scans are not justified.",
        "",
        "## Gate",
        "",
        f"**{'PASS' if summary['gate']['passed'] else 'NO-GO'}.** The gate requires at least "
        f"{summary['gate']['minimum_feasible_rate']:.0%} feasibility and no more than "
        f"{summary['gate']['maximum_length_ratio']:.2f}x teacher length.",
        "",
        "Stage 2 and C-space Flow Matching remain stopped.",
    ])
    return "\n".join(lines) + "\n"


def plot(summary: dict[str, object], output: Path) -> None:
    names = list(summary["runs"])
    rows = [summary["runs"][name] for name in names]
    short = ["Velocity", "Coverage", "Balanced", "Tangent", "Surface"]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(x, [row["feasible_rate"] for row in rows], color="#3973ac")
    axes[0].axhline(0.9, color="#b33a3a", linestyle="--")
    axes[0].set(title="Best-of-8 hard feasibility", ylabel="Rate", ylim=(0, 1.05))
    axes[1].bar(x, [row["mean_length_ratio_to_teacher"] for row in rows], color="#4b9560")
    axes[1].axhline(1.1, color="#b33a3a", linestyle="--")
    axes[1].set(title="Path length", ylabel="Generated / teacher")
    for name, row in zip(names, rows):
        anytime = row["anytime_by_samples"]
        counts = [int(value) for value in anytime]
        axes[2].plot(
            counts,
            [anytime[str(value)]["feasible_rate"] for value in counts],
            marker="o",
            label=name,
        )
    axes[2].set(title="Anytime feasibility", xlabel="Samples", ylabel="Rate", ylim=(0, 1.05))
    axes[2].legend(fontsize=7)
    for axis in axes[:2]:
        axis.set_xticks(x, short, rotation=15)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
