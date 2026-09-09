#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-start Stage 2 seeds")
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--roundtrip-audit", type=Path, required=True)
    parser.add_argument("--template-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluations = [json.loads(path.read_text()) for path in args.evaluations]
    summaries = [evaluation["summary"] for evaluation in evaluations]
    dataset = json.loads(args.dataset_summary.read_text())
    roundtrip = json.loads(args.roundtrip_audit.read_text())["summary"]
    template = json.loads(args.template_audit.read_text())["summary"]
    aggregate = {
        "seeds": len(summaries),
        "validation_instances_per_seed": [summary["instances"] for summary in summaries],
        "instance_modes_per_seed": [summary["instance_modes"] for summary in summaries],
        "candidate_feasible_rate": stats(
            [summary["candidate_feasible_rate"] for summary in summaries]
        ),
        "mode_recovery_rate_k8": stats(
            [summary["mode_recovery_rate"] for summary in summaries]
        ),
        "all_modes_recovered_rate": stats(
            [summary["all_modes_recovered_rate"] for summary in summaries]
        ),
        "feasible_length_ratio": stats(
            [summary["mean_feasible_length_ratio"] for summary in summaries]
        ),
        "top_k_mode_recovery": {
            str(k): stats(
                [summary["top_k_mode_recovery"][str(k)]["mode_recovery_rate"] for summary in summaries]
            )
            for k in (1, 2, 4, 8)
        },
        "deterministic_template_feasible_rate": template[
            "deterministic_template_feasible_rate"
        ],
        "multistart_dataset_candidates": dataset["candidates"],
        "multistart_dataset_roundtrip_feasible_rate": roundtrip["feasible_rate"],
        "multistart_dataset_roundtrip_maximum_missed_change": roundtrip[
            "maximum_absolute_missed_change"
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    plot_summary(summaries, args.output / "stage2_multistart_seed_summary.png")
    (args.output / "report.md").write_text(report(aggregate, summaries) + "\n")
    print(json.dumps(aggregate, indent=2))


def stats(values) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def plot_summary(summaries, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = [f"seed {index}" for index in range(len(summaries))]
    x = np.arange(len(summaries))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [summary["candidate_feasible_rate"] for summary in summaries],
        width,
        label="per-sample feasible",
    )
    axes[0].bar(
        x + width / 2,
        [summary["mode_recovery_rate"] for summary in summaries],
        width,
        label="K=8 mode recovery",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", label="template baseline")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    sample_counts = [1, 2, 4, 8]
    for index, summary in enumerate(summaries):
        axes[1].plot(
            sample_counts,
            [
                summary["top_k_mode_recovery"][str(k)]["mode_recovery_rate"]
                for k in sample_counts
            ],
            marker="o",
            label=labels[index],
        )
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_xticks(sample_counts)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Samples per conditioned mode (K)")
    axes[1].set_ylabel("Mode recovery rate")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def report(aggregate, summaries) -> str:
    candidate = aggregate["candidate_feasible_rate"]
    recovery = aggregate["mode_recovery_rate_k8"]
    length = aggregate["feasible_length_ratio"]
    rows = [
        "# Stage 2 Multi-Start Teacher and Flow Matching Replication",
        "",
        "## Data contract",
        "",
        f"- Multi-start dataset: 40 instances, {aggregate['multistart_dataset_candidates']} candidates.",
        f"- Model-input roundtrip feasibility: {aggregate['multistart_dataset_roundtrip_feasible_rate']:.2%}.",
        f"- Maximum roundtrip missed-fraction change: {aggregate['multistart_dataset_roundtrip_maximum_missed_change']:.6g}.",
        "- Every instance-mode includes a deterministic template plus hard-feasible, distance-filtered multi-start alternatives.",
        "",
        "## Three-seed held-out result",
        "",
        "| Seed | Per-sample feasible | K=8 mode recovery | All modes recovered | Feasible length ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, summary in enumerate(summaries):
        rows.append(
            f"| {index} | {summary['candidate_feasible_rate']:.2%} | "
            f"{summary['mode_recovery_rate']:.2%} | "
            f"{summary['all_modes_recovered_rate']:.2%} | "
            f"{summary['mean_feasible_length_ratio']:.4f} |"
        )
    rows.extend(
        [
            "",
            f"Mean per-sample feasibility is {candidate['mean']:.2%} "
            f"(sample SD {candidate['sample_std']:.2%}). Mean K=8 mode recovery is "
            f"{recovery['mean']:.2%} (sample SD {recovery['sample_std']:.2%}). "
            f"Mean feasible length ratio is {length['mean']:.4f}.",
            "",
            "## Diagnosis",
            "",
            "The previous structured dataset was an analytic-template replay and is invalid as evidence for generative multimodality. On the corrected multi-start distribution, standard independently coupled conditional Flow Matching is seed-sensitive and remains below the 100% feasible deterministic template baseline. It therefore fails the current Stage 2 gate.",
            "",
            "The next model experiment should change the coupling or conditional representation, not select the best random seed. Direct C-space expansion remains deferred.",
        ]
    )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
