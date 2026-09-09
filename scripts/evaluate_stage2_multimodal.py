#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_coverage.coverage import (
    CoveragePlan,
    constrained_coverage_key,
    decode_structured_parameter_controls,
    evaluate_coverage,
    load_teacher_instance,
    map_surface_parameters,
    structured_control_token_count,
    structured_residual_to_controls,
    structured_controls_to_residual,
    surface_from_teacher_archive,
)
from diffusion_coverage.learning import TeacherPathDataset
from diffusion_coverage.evaluation import (
    cluster_controls,
    control_energy_distance,
    expert_control_coverage,
    leave_one_out_control_coverage,
)
from diffusion_coverage.models import PathVectorField, PathVectorFieldConfig, heun_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mode-conditioned structured flow matching"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--samples-per-mode", type=int, default=8)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cuda-device-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--diversity-threshold-radius", type=float, default=0.05)
    parser.add_argument("--expert-match-threshold-radius", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_mode < 1 or args.ode_steps < 1:
        raise ValueError("sample and ODE step counts must be positive")
    if args.diversity_threshold_radius <= 0.0 or args.expert_match_threshold_radius <= 0.0:
        raise ValueError("diversity thresholds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cuda_device_index is not None and args.cuda_device_index < 0:
        raise ValueError("cuda_device_index must be nonnegative")
    if args.cuda_device_index is not None and args.device == "cpu":
        raise ValueError("cuda_device_index cannot be used with --device cpu")
    cuda_name = (
        "cuda" if args.cuda_device_index is None else f"cuda:{args.cuda_device_index}"
    )
    device = (
        torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(cuda_name if args.device == "cuda" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    coordinate_system = checkpoint.get("path_coordinate_system")
    if coordinate_system not in {
        "analytic_uv_structured", "analytic_uv_structured_residual"
    }:
        raise ValueError("this evaluator requires structured analytic targets")
    if not checkpoint.get("include_mode_conditioning", False):
        raise ValueError("this evaluator requires a mode-conditioned checkpoint")
    model = PathVectorField(PathVectorFieldConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    if args.split == "validation":
        instance_ids = list(checkpoint["validation_instance_ids"])
    elif args.split == "train":
        instance_ids = list(checkpoint["train_instance_ids"])
    else:
        instance_ids = list(checkpoint["train_instance_ids"]) + list(
            checkpoint["validation_instance_ids"]
        )
    if args.max_instances is not None:
        instance_ids = instance_ids[: args.max_instances]
    dataset = TeacherPathDataset(
        args.dataset,
        instance_ids=instance_ids,
        num_surface_points=int(checkpoint["num_surface_points"]),
        num_path_waypoints=None,
        minimum_path_tokens=int(checkpoint["minimum_path_tokens"]),
        maximum_path_tokens=int(checkpoint["maximum_path_tokens"]),
        candidate_policy="all",
        candidate_name=None,
        allow_repaired_candidates=False,
        path_coordinate_system=coordinate_system,
        include_mode_conditioning=True,
        preserve_source_waypoints=True,
        seed=int(checkpoint.get("data_seed", checkpoint["seed"])),
    )
    sample_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for sample_index, reference in enumerate(dataset.sample_index):
        row = dataset.rows[reference.instance_index]
        sample = dataset[sample_index]
        sample_groups[(str(row["instance_id"]), str(sample["mode_name"]))].append(
            sample_index
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    candidate_records: list[dict[str, object]] = []
    mode_records: list[dict[str, object]] = []
    for group_number, ((instance_id, mode_name), sample_indices) in enumerate(
        sorted(sample_groups.items()), start=1
    ):
        sample = dataset[sample_indices[0]]
        reference = dataset.sample_index[sample_indices[0]]
        row = dataset.rows[reference.instance_index]
        archive = load_teacher_instance(args.dataset / str(row["path"]))
        metadata = archive["metadata"]
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        config = metadata["teacher_config"]
        radius = float(config["footprint_radius"])
        epsilon = float(config["missed_tolerance"])
        overlap = float(config["overlap"])
        num_tokens = structured_control_token_count(
            surface,
            footprint_radius=radius,
            overlap=overlap,
            mode_name=mode_name,
        )
        surface_tensor = sample["surface"].unsqueeze(0).repeat(
            args.samples_per_mode, 1, 1
        ).to(device)
        condition = sample["condition"].unsqueeze(0).repeat(
            args.samples_per_mode, 1
        ).to(device)
        path_mask = torch.ones(
            args.samples_per_mode, num_tokens, dtype=torch.bool, device=device
        )
        path_arclength = torch.linspace(
            0.0, 1.0, num_tokens, device=device
        ).unsqueeze(0).repeat(args.samples_per_mode, 1)
        generation_start = perf_counter()
        generated = heun_sample(
            model,
            surface_tensor,
            condition,
            num_waypoints=num_tokens,
            num_steps=args.ode_steps,
            noise_smoothing_sigma=float(checkpoint.get("noise_smoothing_sigma", 0.0)),
            noise_smoothing_fraction=float(
                checkpoint.get("noise_smoothing_fraction", 0.0)
            ),
            path_mask=path_mask,
            path_arclength=path_arclength,
        ).cpu().numpy().astype(np.float64)
        generation_time = perf_counter() - generation_start

        group_candidates: list[dict[str, object]] = []
        generated_residuals: list[np.ndarray] = []
        for sample_number, model_output in enumerate(generated, start=1):
            controls = (
                structured_residual_to_controls(
                    surface,
                    model_output,
                    footprint_radius=radius,
                    overlap=overlap,
                    mode_name=mode_name,
                )
                if coordinate_system == "analytic_uv_structured_residual"
                else model_output
            )
            residual = (
                model_output
                if coordinate_system == "analytic_uv_structured_residual"
                else structured_controls_to_residual(
                    surface,
                    controls,
                    footprint_radius=radius,
                    overlap=overlap,
                    mode_name=mode_name,
                )
            )
            generated_residuals.append(np.asarray(residual, dtype=np.float64))
            parameters = decode_structured_parameter_controls(
                surface,
                controls,
                footprint_radius=radius,
                mode_name=mode_name,
            )
            path = map_surface_parameters(
                surface, parameters[:, 0], parameters[:, 1]
            )
            check_start = perf_counter()
            metrics = evaluate_coverage(
                surface, CoveragePlan(path), footprint_radius=radius
            )
            check_time = perf_counter() - check_start
            record = {
                "instance_id": instance_id,
                "surface_id": str(row["surface_id"]),
                "mode": mode_name,
                "sample": sample_number,
                "num_control_tokens": num_tokens,
                "feasible": bool(metrics.missed_fraction <= epsilon + 1e-12),
                "missed_fraction": metrics.missed_fraction,
                "path_length": metrics.path_length,
                "generation_time_share": generation_time / args.samples_per_mode,
                "hard_checker_time": check_time,
                "residual_controls": np.asarray(residual, dtype=np.float64).tolist(),
            }
            candidate_records.append(record)
            group_candidates.append(record)

        best = min(
            group_candidates,
            key=lambda item: constrained_coverage_key(
                float(item["missed_fraction"]), float(item["path_length"]), epsilon
            ),
        )
        teacher_candidates = [dataset[index] for index in sample_indices]
        teacher_best = min(
            teacher_candidates,
            key=lambda item: constrained_coverage_key(
                float(item["teacher_metrics"][0]),
                float(item["teacher_metrics"][1]),
                epsilon,
            ),
        )
        feasible_count = sum(bool(item["feasible"]) for item in group_candidates)
        feasible_residuals = [
            residual
            for residual, item in zip(generated_residuals, group_candidates)
            if bool(item["feasible"])
        ]
        clustered = cluster_controls(
            feasible_residuals, threshold=args.diversity_threshold_radius
        )
        expert_residuals = []
        for teacher_sample in teacher_candidates:
            teacher_path = teacher_sample["path"].numpy().astype(np.float64)
            expert_residuals.append(
                teacher_path
                if coordinate_system == "analytic_uv_structured_residual"
                else structured_controls_to_residual(
                    surface,
                    teacher_path,
                    footprint_radius=radius,
                    overlap=overlap,
                    mode_name=mode_name,
                ).astype(np.float64)
            )
        expert_coverage, expert_nearest_distance = expert_control_coverage(
            feasible_residuals,
            expert_residuals,
            threshold=args.expert_match_threshold_radius,
        )
        teacher_self_coverage = leave_one_out_control_coverage(
            expert_residuals, threshold=args.expert_match_threshold_radius
        )
        energy_distance = control_energy_distance(feasible_residuals, expert_residuals)
        mode_record = {
            "instance_id": instance_id,
            "surface_id": str(row["surface_id"]),
            "mode": mode_name,
            "samples": args.samples_per_mode,
            "num_control_tokens": num_tokens,
            "feasible_samples": feasible_count,
            "sample_feasible_rate": feasible_count / args.samples_per_mode,
            "mode_recovered": feasible_count > 0,
            "feasible_control_clusters": clustered.num_clusters,
            "normalized_control_cluster_entropy": clustered.normalized_entropy,
            "expert_control_coverage": expert_coverage,
            "teacher_leave_one_out_control_coverage": teacher_self_coverage,
            "generated_teacher_control_energy_distance": energy_distance,
            "mean_expert_nearest_generated_distance_radius": expert_nearest_distance,
            "best_missed_fraction": float(best["missed_fraction"]),
            "best_path_length": float(best["path_length"]),
            "teacher_missed_fraction": float(teacher_best["teacher_metrics"][0]),
            "teacher_path_length": float(teacher_best["teacher_metrics"][1]),
            "feasible_length_ratio": (
                float(best["path_length"]) / float(teacher_best["teacher_metrics"][1])
                if bool(best["feasible"])
                else None
            ),
            "generation_time": generation_time,
            "hard_checker_time": sum(
                float(item["hard_checker_time"]) for item in group_candidates
            ),
        }
        mode_records.append(mode_record)
        print(
            f"[{group_number:03d}/{len(sample_groups):03d}] {instance_id:<24} "
            f"{mode_name:<24} recovered={mode_record['mode_recovered']} "
            f"feasible={feasible_count}/{args.samples_per_mode} "
            f"miss={mode_record['best_missed_fraction']:.4f}",
            flush=True,
        )

    instance_records = summarize_instances(mode_records)
    summary = summarize(mode_records, candidate_records, instance_records)
    summary["by_mode"] = {
        mode: summarize_mode([row for row in mode_records if row["mode"] == mode])
        for mode in sorted({str(row["mode"]) for row in mode_records})
    }
    summary["by_surface"] = {
        surface_id: summarize_mode(
            [row for row in mode_records if row["surface_id"] == surface_id]
        )
        for surface_id in sorted({str(row["surface_id"]) for row in mode_records})
    }
    write_jsonl(args.output / "candidates.jsonl", candidate_records)
    write_jsonl(args.output / "instance_modes.jsonl", mode_records)
    write_jsonl(args.output / "instances.jsonl", instance_records)
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "split": args.split,
                    "samples_per_mode": args.samples_per_mode,
                    "ode_steps": args.ode_steps,
                    "seed": args.seed,
                    "mode_sampling": "explicit enumeration of expert-supported modes",
                    "selection": "best-of-K under hard constrained objective",
                    "diversity_threshold_radius": args.diversity_threshold_radius,
                    "expert_match_threshold_radius": args.expert_match_threshold_radius,
                },
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    plot_summary(summary, args.output / "stage2_mode_recovery.png")
    print(json.dumps(summary, indent=2))


def summarize_instances(mode_records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in mode_records:
        groups[str(row["instance_id"])].append(row)
    records = []
    for instance_id, rows in sorted(groups.items()):
        recovered = sum(bool(row["mode_recovered"]) for row in rows)
        feasible_samples = np.asarray(
            [int(row["feasible_samples"]) for row in rows], dtype=np.float64
        )
        probabilities = feasible_samples / feasible_samples.sum() if feasible_samples.sum() else feasible_samples
        entropy = float(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])))
        records.append(
            {
                "instance_id": instance_id,
                "surface_id": str(rows[0]["surface_id"]),
                "expert_modes": len(rows),
                "recovered_modes": recovered,
                "mode_coverage": recovered / len(rows),
                "all_modes_recovered": recovered == len(rows),
                "feasible_sample_mode_entropy": entropy,
                "normalized_feasible_sample_mode_entropy": (
                    entropy / math.log(len(rows)) if len(rows) > 1 else 1.0
                ),
            }
        )
    return records


def summarize_mode(rows: list[dict[str, object]]) -> dict[str, object]:
    feasible_ratios = [
        float(row["feasible_length_ratio"])
        for row in rows
        if row["feasible_length_ratio"] is not None
    ]
    return {
        "instance_modes": len(rows),
        "mode_recovery_rate": float(np.mean([row["mode_recovered"] for row in rows])),
        "sample_feasible_rate": float(np.mean([row["sample_feasible_rate"] for row in rows])),
        "mean_feasible_control_clusters": float(
            np.mean([row["feasible_control_clusters"] for row in rows])
        ),
        "mean_expert_control_coverage": float(
            np.mean([row["expert_control_coverage"] for row in rows])
        ),
        "mean_teacher_leave_one_out_control_coverage": float(
            np.mean([row["teacher_leave_one_out_control_coverage"] for row in rows])
        ),
        "mean_feasible_length_ratio": (
            float(np.mean(feasible_ratios)) if feasible_ratios else None
        ),
        "median_feasible_length_ratio": (
            float(np.median(feasible_ratios)) if feasible_ratios else None
        ),
    }


def summarize(
    mode_records: list[dict[str, object]],
    candidate_records: list[dict[str, object]],
    instance_records: list[dict[str, object]],
) -> dict[str, object]:
    feasible_ratios = [
        float(row["feasible_length_ratio"])
        for row in mode_records
        if row["feasible_length_ratio"] is not None
    ]
    mode_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in candidate_records:
        mode_groups[(str(row["instance_id"]), str(row["mode"]))].append(row)
    top_k = {}
    for sample_count in (1, 2, 4, 8):
        if sample_count > max(len(rows) for rows in mode_groups.values()):
            continue
        recovered = [
            any(bool(row["feasible"]) for row in rows[:sample_count])
            for rows in mode_groups.values()
        ]
        top_k[str(sample_count)] = {
            "mode_recovery_rate": float(np.mean(recovered)),
            "recovered_instance_modes": int(sum(recovered)),
            "instance_modes": len(recovered),
        }
    mode_recovery_rate = float(
        np.mean([row["mode_recovered"] for row in mode_records])
    )
    all_modes_recovered_rate = float(
        np.mean([row["all_modes_recovered"] for row in instance_records])
    )
    return {
        "instances": len(instance_records),
        "instance_modes": len(mode_records),
        "generated_candidates": len(candidate_records),
        "candidate_feasible_rate": float(
            np.mean([row["feasible"] for row in candidate_records])
        ),
        "mode_recovery_rate": mode_recovery_rate,
        "mode_recovery_wilson_95": wilson_interval(
            int(sum(bool(row["mode_recovered"]) for row in mode_records)),
            len(mode_records),
        ),
        "mean_instance_mode_coverage": float(
            np.mean([row["mode_coverage"] for row in instance_records])
        ),
        "all_modes_recovered_rate": all_modes_recovered_rate,
        "all_modes_recovered_wilson_95": wilson_interval(
            int(sum(bool(row["all_modes_recovered"]) for row in instance_records)),
            len(instance_records),
        ),
        "mean_recovered_modes": float(
            np.mean([row["recovered_modes"] for row in instance_records])
        ),
        "mean_expert_modes": float(
            np.mean([row["expert_modes"] for row in instance_records])
        ),
        "mean_normalized_feasible_sample_mode_entropy": float(
            np.mean(
                [row["normalized_feasible_sample_mode_entropy"] for row in instance_records]
            )
        ),
        "mean_feasible_control_clusters": float(
            np.mean([row["feasible_control_clusters"] for row in mode_records])
        ),
        "mean_normalized_control_cluster_entropy": float(
            np.mean([row["normalized_control_cluster_entropy"] for row in mode_records])
        ),
        "mean_expert_control_coverage": float(
            np.mean([row["expert_control_coverage"] for row in mode_records])
        ),
        "mean_teacher_leave_one_out_control_coverage": float(
            np.mean([
                row["teacher_leave_one_out_control_coverage"] for row in mode_records
            ])
        ),
        "mean_generated_teacher_control_energy_distance": float(
            np.mean([
                row["generated_teacher_control_energy_distance"]
                for row in mode_records
                if row["generated_teacher_control_energy_distance"] is not None
            ])
        ),
        "mean_feasible_length_ratio": (
            float(np.mean(feasible_ratios)) if feasible_ratios else None
        ),
        "median_feasible_length_ratio": (
            float(np.median(feasible_ratios)) if feasible_ratios else None
        ),
        "mean_generation_time_per_mode": float(
            np.mean([row["generation_time"] for row in mode_records])
        ),
        "mean_hard_checker_time_per_mode": float(
            np.mean([row["hard_checker_time"] for row in mode_records])
        ),
        "top_k_mode_recovery": top_k,
    }


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count < 1:
        raise ValueError("count must be positive")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [center - margin, center + margin]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def plot_summary(summary: dict[str, object], output: Path) -> None:
    by_mode = summary["by_mode"]
    assert isinstance(by_mode, dict)
    modes = list(by_mode)
    recovery = [float(by_mode[mode]["mode_recovery_rate"]) for mode in modes]
    sample_feasible = [float(by_mode[mode]["sample_feasible_rate"]) for mode in modes]
    x = np.arange(len(modes))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axis = axes[0]
    axis.bar(x - width / 2, recovery, width, label="best-of-K mode recovery")
    axis.bar(x + width / 2, sample_feasible, width, label="per-sample feasibility")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_xticks(x, [mode.replace("_phase_", "\nphase ") for mode in modes])
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    top_k = summary["top_k_mode_recovery"]
    assert isinstance(top_k, dict)
    sample_counts = [int(value) for value in top_k]
    axes[1].plot(
        sample_counts,
        [float(top_k[str(value)]["mode_recovery_rate"]) for value in sample_counts],
        marker="o",
        color="#2f6f8f",
    )
    axes[1].set_xticks(sample_counts)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Samples per conditioned mode (K)")
    axes[1].set_ylabel("Mode recovery rate")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
