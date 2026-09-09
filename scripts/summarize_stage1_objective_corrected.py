#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion_coverage.coverage import load_teacher_instance
from diffusion_coverage.learning import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize objective-corrected Stage 1 runs")
    parser.add_argument("--old-dataset", required=True, type=Path)
    parser.add_argument("--new-dataset", required=True, type=Path)
    parser.add_argument("--old-run", required=True, type=Path)
    parser.add_argument("--old-eval-dir", default="eval_objective_correct_k8_anytime")
    parser.add_argument("--fresh-run", required=True, type=Path)
    parser.add_argument("--warm-run", required=True, type=Path)
    parser.add_argument("--target-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_run(path: Path, eval_dir: str = "eval_k8_anytime") -> dict[str, object]:
    history = json.loads((path / "history.json").read_text())
    metrics = json.loads((path / eval_dir / "metrics.json").read_text())["summary"]
    return {
        "path": str(path),
        "best_validation_loss": min(row["validation_loss"] for row in history),
        "history": history,
        "metrics": metrics,
    }


def selected_targets(dataset: Path) -> dict[str, tuple[str, float, str]]:
    result = {}
    for row in load_manifest(dataset):
        archive = load_teacher_instance(dataset / str(row["path"]))
        result[str(row["instance_id"])] = (
            str(row["surface_id"]),
            float(archive["candidate_metrics"][0, 1]),
            str(archive["proposal_names"][0]),
        )
    return result


def target_bias(old_dataset: Path, new_dataset: Path) -> dict[str, object]:
    old = selected_targets(old_dataset)
    new = selected_targets(new_dataset)
    ratios = []
    changed = 0
    by_surface: dict[str, list[float]] = defaultdict(list)
    for instance_id, (surface, new_length, new_name) in new.items():
        _, old_length, old_name = old[instance_id]
        ratio = old_length / new_length
        ratios.append(ratio)
        by_surface[surface].append(ratio)
        changed += old_name != new_name or abs(old_length - new_length) > 1e-9
    return {
        "instances": len(ratios),
        "changed_targets": changed,
        "mean_old_to_new_length_ratio": float(np.mean(ratios)),
        "median_old_to_new_length_ratio": float(np.median(ratios)),
        "p95_old_to_new_length_ratio": float(np.quantile(ratios, 0.95)),
        "by_surface": {
            surface: float(np.mean(values)) for surface, values in sorted(by_surface.items())
        },
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {
        "Old-target warm model": load_run(args.old_run, args.old_eval_dir),
        "Correct target, fresh": load_run(args.fresh_run),
        "Correct target, warm": load_run(args.warm_run),
    }
    audit = json.loads(args.target_audit.read_text())["summary"]
    summary = {
        "objective_contract": "minimize path length subject to missed_fraction <= epsilon",
        "target_bias": target_bias(args.old_dataset, args.new_dataset),
        "target_roundtrip": audit,
        "runs": {
            name: {
                "path": run["path"],
                "best_validation_loss": run["best_validation_loss"],
                **run["metrics"],
            }
            for name, run in runs.items()
        },
        "gate": {
            "minimum_feasible_rate": 0.9,
            "maximum_mean_length_ratio": 1.1,
        },
    }
    selected = summary["runs"]["Correct target, warm"]
    summary["gate"]["passed"] = bool(
        selected["feasible_rate"] >= 0.9
        and selected["mean_length_ratio_to_teacher"] <= 1.1
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "report.md").write_text(render_report(summary))
    plot_runs(runs, args.output / "stage1_objective_corrected_diagnostics.png")
    print(render_report(summary))


def render_report(summary: dict[str, object]) -> str:
    bias = summary["target_bias"]
    audit = summary["target_roundtrip"]
    runs = summary["runs"]
    selected = runs["Correct target, warm"]
    lines = [
        "# Objective-Corrected Stage 1 Surface Flow Matching",
        "",
        "The shared objective now ranks all hard-feasible candidates by path length. "
        "Missed coverage is a constraint, not a secondary minimization target.",
        "",
        "## Data Contract",
        "",
        f"- Changed selected targets: {bias['changed_targets']}/{bias['instances']}",
        f"- Mean old/new target length: {bias['mean_old_to_new_length_ratio']:.4f}x",
        f"- Hard-feasible learning-target roundtrip: {audit['feasible_rate']:.1%}",
        f"- Mean roundtrip length ratio: {audit['mean_length_ratio']:.4f}x",
        "",
        "## Best-Of-8 Results",
        "",
        "| Run | Best val loss | Feasible | Mean missed | Length / teacher | Mean time |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for name, row in runs.items():
        lines.append(
            f"| {name} | {row['best_validation_loss']:.4f} | {row['feasible_rate']:.1%} | "
            f"{row['mean_missed_fraction']:.4f} | {row['mean_length_ratio_to_teacher']:.3f} | "
            f"{row['mean_total_pipeline_time']:.3f}s |"
        )
    lines.extend([
        "",
        "## Correct-Target Warm Run By Surface",
        "",
        "| Surface | Feasible | Mean missed | Length / teacher |",
        "|:---|---:|---:|---:|",
    ])
    for surface, row in selected["by_surface"].items():
        lines.append(
            f"| {surface} | {row['feasible_rate']:.1%} | {row['mean_missed_fraction']:.4f} | "
            f"{row['mean_length_ratio_to_teacher']:.3f} |"
        )
    lines.extend([
        "",
        "## Anytime Feasibility",
        "",
        "| K | Feasible | Mean missed | Mean time | Median time |",
        "|---:|---:|---:|---:|---:|",
    ])
    for count, row in selected["anytime_by_samples"].items():
        lines.append(
            f"| {count} | {row['feasible_rate']:.1%} | {row['mean_best_missed_fraction']:.4f} | "
            f"{row['mean_elapsed_time']:.3f}s | {row['median_elapsed_time']:.3f}s |"
        )
    gate = summary["gate"]
    lines.extend([
        "",
        "## Gate",
        "",
        f"**{'PASS' if gate['passed'] else 'NO-GO'}.** Required at least "
        f"{gate['minimum_feasible_rate']:.0%} feasibility and no more than "
        f"{gate['maximum_mean_length_ratio']:.2f}x teacher length; the strongest corrected-target "
        f"run reached {selected['feasible_rate']:.1%} and {selected['mean_length_ratio_to_teacher']:.3f}x.",
        "",
        "Warm-starting recovers feasibility by generating denser, longer paths; fresh training produces "
        "shorter paths but misses more surface. The current vector field therefore exposes a coverage-"
        "versus-length tradeoff rather than learning the constrained shortest-path frontier.",
        "",
        "Stage 2 and C-space Flow Matching remain stopped at this gate.",
    ])
    return "\n".join(lines) + "\n"


def plot_runs(runs: dict[str, dict[str, object]], output: Path) -> None:
    names = list(runs)
    display_names = ["Old", "Fresh", "Warm"]
    metrics = [runs[name]["metrics"] for name in names]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for name, run in runs.items():
        history = run["history"]
        axes[0, 0].plot(
            [row["global_step"] for row in history],
            [row["validation_loss"] for row in history],
            label=name,
        )
    axes[0, 0].set(title="Validation flow loss", xlabel="Training step", ylabel="Loss")
    axes[0, 0].legend(fontsize=8)
    x = np.arange(len(names))
    axes[0, 1].bar(x, [row["feasible_rate"] for row in metrics], color="#3973ac")
    axes[0, 1].axhline(0.9, color="#b33a3a", linestyle="--")
    axes[0, 1].set(title="Best-of-8 hard feasibility", ylabel="Rate", ylim=(0, 1.05))
    axes[1, 0].bar(x, [row["mean_length_ratio_to_teacher"] for row in metrics], color="#4b9560")
    axes[1, 0].axhline(1.1, color="#b33a3a", linestyle="--")
    axes[1, 0].set(title="Path length ratio", ylabel="Generated / teacher")
    anytime = metrics[-1]["anytime_by_samples"]
    counts = [int(value) for value in anytime]
    axes[1, 1].plot(counts, [anytime[str(value)]["feasible_rate"] for value in counts], marker="o")
    axes[1, 1].set(title="Correct-target warm anytime", xlabel="Samples", ylabel="Feasible rate", ylim=(0, 1.05))
    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, display_names)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
