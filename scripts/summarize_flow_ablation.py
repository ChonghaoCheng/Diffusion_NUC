#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize comparable Flow Matching runs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="Repeat for every run; RUN_DIR must contain best.pt and eval metrics",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for specification in args.run:
        if "=" not in specification:
            raise ValueError(f"invalid run specification: {specification}")
        label, raw_path = specification.split("=", 1)
        run = Path(raw_path)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        metrics_document = json.loads(
            (run / "eval_validation_k8" / "metrics.json").read_text()
        )
        metrics = metrics_document.get("summary", metrics_document)
        history = json.loads((run / "history.json").read_text())
        row = {
            "label": label,
            "run": str(run),
            "seed": checkpoint["seed"],
            "batch_size": history[-1].get("batch_size"),
            "global_steps": history[-1]["global_step"],
            "checkpoint_step": checkpoint["global_step"],
            "coupling": checkpoint.get("coupling", "independent"),
            "training_seconds": history[-1]["elapsed_seconds"],
            "candidate_feasible_rate": metrics["candidate_feasible_rate"],
            "mode_recovery_rate": metrics["mode_recovery_rate"],
            "all_modes_recovered_rate": metrics["all_modes_recovered_rate"],
            "mean_feasible_length_ratio": metrics["mean_feasible_length_ratio"],
            "by_mode": metrics["by_mode"],
            "by_surface": metrics["by_surface"],
        }
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    lines = [
        "# Flow Matching ablation summary",
        "",
        "| Run | Batch | Steps | Coupling | Train s | Candidate feasible | K=8 recovery | All modes | Length ratio |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        length = row["mean_feasible_length_ratio"]
        lines.append(
            f"| {row['label']} | {row['batch_size']} | {row['global_steps']} | "
            f"{row['coupling']} | {row['training_seconds']:.2f} | "
            f"{row['candidate_feasible_rate']:.2%} | {row['mode_recovery_rate']:.2%} | "
            f"{row['all_modes_recovered_rate']:.2%} | "
            f"{'n/a' if length is None else f'{length:.4f}'} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
