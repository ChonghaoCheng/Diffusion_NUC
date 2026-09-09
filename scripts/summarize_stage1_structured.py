#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 1 representation ablations")
    parser.add_argument(
        "--single-family-summary",
        type=Path,
        default=ROOT / "results/stage1_single_family_summary/summary.json",
    )
    parser.add_argument(
        "--dense-uv",
        type=Path,
        default=ROOT / "results/flow_matching_stage1_single_family_uv_run1/eval_k8/metrics.json",
    )
    parser.add_argument(
        "--length-uv",
        type=Path,
        default=ROOT / "results/flow_matching_stage1_single_family_uv_length_tangent_run1/eval_k8/metrics.json",
    )
    parser.add_argument(
        "--control-uv",
        type=Path,
        default=ROOT / "results/flow_matching_stage1_raster_control_run1/eval_k8/metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage1_structured_summary",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count**2)) / denominator
    return float(center - radius), float(center + radius)


def row_from_metrics(name: str, payload: dict[str, object]) -> dict[str, object]:
    summary = payload["summary"]
    instances = payload["instances"]
    feasible = [row for row in instances if row["generated_feasible"]]
    successes = len(feasible)
    count = len(instances)
    lower, upper = wilson_interval(successes, count)
    return {
        "method": name,
        "instances": count,
        "feasible_rate": successes / count,
        "feasible_rate_wilson_95": [lower, upper],
        "mean_missed_fraction": summary["mean_missed_fraction"],
        "mean_length_ratio": summary["mean_length_ratio_to_teacher"],
        "feasible_only_length_ratio": float(np.mean([
            row["best_path_length"] / row["teacher_path_length"] for row in feasible
        ])) if feasible else None,
        "mean_path_tokens": summary["mean_path_tokens"],
        "mean_total_pipeline_time": summary["mean_total_pipeline_time"],
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xyz = json.loads(args.single_family_summary.read_text())
    rows = []
    for name, run in xyz["runs"].items():
        rows.append({
            "method": f"XYZ: {name}",
            "instances": run["num_instances"],
            "feasible_rate": run["feasible_rate"],
            "feasible_rate_wilson_95": list(wilson_interval(
                round(run["feasible_rate"] * run["num_instances"]), run["num_instances"]
            )),
            "mean_missed_fraction": run["mean_missed_fraction"],
            "mean_length_ratio": run["mean_length_ratio_to_teacher"],
            "feasible_only_length_ratio": None,
            "mean_path_tokens": run["mean_path_tokens"],
            "mean_total_pipeline_time": run["mean_total_pipeline_time"],
        })
    rows.extend((
        row_from_metrics("Dense analytic UV", load_metrics(args.dense_uv)),
        row_from_metrics("UV + length/tangent", load_metrics(args.length_uv)),
        row_from_metrics("Raster control tokens", load_metrics(args.control_uv)),
    ))
    output = {
        "gate": {"minimum_feasible_rate": 0.9, "maximum_length_ratio": 1.1},
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "# Stage 1 Structured Representation Summary",
        "",
        "Registered gate: hard feasibility >= 90% and mean length / teacher <= 1.10.",
        "",
        "| Method | N | Feasible | 95% Wilson CI | Length / teacher | Feasible-only length | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["feasible_rate_wilson_95"]
        feasible_only = row["feasible_only_length_ratio"]
        lines.append(
            f"| {row['method']} | {row['instances']} | {100.0 * row['feasible_rate']:.1f}% | "
            f"[{100.0 * low:.1f}%, {100.0 * high:.1f}%] | {row['mean_length_ratio']:.3f}x | "
            f"{'n/a' if feasible_only is None else f'{feasible_only:.3f}x'} | {row['mean_path_tokens']:.1f} |"
        )
    lines.extend((
        "",
        "The control-token run passes the point-estimate gate on 31/33 held-out instances. "
        "Its 95% Wilson lower bound remains below 90%, so larger-cohort replication is required.",
        "",
        "Scope: the passing run uses one unrepaired `raster_u_phase_0.00` family. It validates "
        "structured stroke generation, not multimodal planning.",
    ))
    (args.output / "report.md").write_text("\n".join(lines) + "\n")

    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    for row in rows:
        axis.scatter(row["mean_length_ratio"], row["feasible_rate"], s=65)
        axis.annotate(
            row["method"].replace("XYZ: ", ""),
            (row["mean_length_ratio"], row["feasible_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0.9, color="#B42318", linestyle="--", linewidth=1.2, label="Feasibility gate")
    axis.axvline(1.1, color="#175CD3", linestyle="--", linewidth=1.2, label="Length gate")
    axis.set_xlabel("Mean path length / teacher")
    axis.set_ylabel("Hard feasibility rate")
    axis.set_ylim(0.0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(args.output / "stage1_feasibility_length.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
