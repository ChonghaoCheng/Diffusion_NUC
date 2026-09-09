#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from diffusion_coverage.learning import TeacherPathDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose residual generation against templates")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_rows = []
    template_summary = None
    for specification in args.run:
        label, raw_run = specification.split("=", 1)
        run = Path(raw_run)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        dataset = TeacherPathDataset(
            args.dataset,
            instance_ids=checkpoint["validation_instance_ids"],
            num_surface_points=int(checkpoint["num_surface_points"]),
            num_path_waypoints=None,
            minimum_path_tokens=int(checkpoint["minimum_path_tokens"]),
            maximum_path_tokens=int(checkpoint["maximum_path_tokens"]),
            candidate_policy="all",
            allow_repaired_candidates=False,
            path_coordinate_system="analytic_uv_structured_residual",
            include_mode_conditioning=True,
            preserve_source_waypoints=True,
            seed=int(checkpoint.get("data_seed", checkpoint["seed"])),
        )
        groups = defaultdict(list)
        for index, reference in enumerate(dataset.sample_index):
            row = dataset.rows[reference.instance_index]
            sample = dataset[index]
            groups[(str(row["instance_id"]), str(sample["mode_name"]))].append(sample)
        templates = {}
        teacher_best = {}
        for key, samples in groups.items():
            template = min(samples, key=lambda sample: float(np.linalg.norm(sample["path"].numpy())))
            templates[key] = float(template["teacher_metrics"][1])
            teacher_best[key] = min(float(sample["teacher_metrics"][1]) for sample in samples)
        if template_summary is None:
            ratios = [templates[key] / teacher_best[key] for key in groups]
            template_summary = {
                "instance_modes": len(groups),
                "mean_template_to_best_teacher_length_ratio": float(np.mean(ratios)),
                "median_template_to_best_teacher_length_ratio": float(np.median(ratios)),
                "template_modes_strictly_longer_than_best_teacher": int(
                    sum(ratio > 1.0 + 1e-9 for ratio in ratios)
                ),
            }
        mode_rows = [
            json.loads(line)
            for line in (run / "eval_validation_k8" / "instance_modes.jsonl").read_text().splitlines()
            if line
        ]
        metrics_document = json.loads(
            (run / "eval_validation_k8" / "metrics.json").read_text()
        )
        metrics = metrics_document.get("summary", metrics_document)
        feasible = [row for row in mode_rows if row["mode_recovered"]]
        generated_to_template = [
            float(row["best_path_length"]) / templates[(row["instance_id"], row["mode"])]
            for row in feasible
        ]
        output_rows.append({
            "label": label,
            "run": str(run),
            "recovered_modes": len(feasible),
            "instance_modes": len(mode_rows),
            "candidate_feasible_rate": metrics["candidate_feasible_rate"],
            "mode_recovery_rate": metrics["mode_recovery_rate"],
            "all_modes_recovered_rate": metrics["all_modes_recovered_rate"],
            "mean_feasible_length_ratio": metrics["mean_feasible_length_ratio"],
            "mean_best_generated_to_template_length_ratio": float(np.mean(generated_to_template)),
            "median_best_generated_to_template_length_ratio": float(np.median(generated_to_template)),
            "modes_shorter_than_template": int(sum(ratio < 1.0 - 1e-9 for ratio in generated_to_template)),
            "modes_within_one_percent_of_best_teacher": int(sum(
                float(row["best_path_length"]) / teacher_best[(row["instance_id"], row["mode"])] <= 1.01
                for row in feasible
            )),
        })
    document = {"template_baseline": template_summary, "runs": output_rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "residual_diagnosis.json").write_text(json.dumps(document, indent=2) + "\n")
    lines = [
        "# Stage 2 template-residual diagnosis",
        "",
        f"The deterministic template is {template_summary['mean_template_to_best_teacher_length_ratio']:.4f}x "
        "the same-mode best teacher length on average.",
        "",
        "| Run | Candidate feasible | K=8 recovery | All modes | Generated / best teacher | Generated / template | Shorter modes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['label']} | {row['candidate_feasible_rate']:.2%} | "
            f"{row['mode_recovery_rate']:.2%} | {row['all_modes_recovered_rate']:.2%} | "
            f"{row['mean_feasible_length_ratio']:.4f} | "
            f"{row['mean_best_generated_to_template_length_ratio']:.4f} | "
            f"{row['modes_shorter_than_template']}/{row['instance_modes']} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
