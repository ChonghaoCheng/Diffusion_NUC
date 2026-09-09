#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    densify_parameter_polyline,
    decode_raster_parameter_controls,
    decode_structured_parameter_controls,
    evaluate_coverage,
    load_teacher_instance,
    refine_coverage_by_insertion,
    refine_coverage_by_shortcutting,
    surface_from_teacher_archive,
    constrained_coverage_key,
    map_surface_parameters,
    raster_control_token_count,
    structured_control_token_count,
    structured_residual_to_controls,
)
from diffusion_coverage.learning import TeacherPathDataset, candidate_indices_by_name
from diffusion_coverage.models import PathVectorField, PathVectorFieldConfig, heun_sample
from diffusion_coverage.representation import suggested_token_count
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated paths with exact mesh projection")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "flow_matching_m2_eval")
    parser.add_argument("--samples-per-instance", type=int, default=8)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--split", choices=("validation", "train", "all"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refine-top-k", type=int, default=0)
    parser.add_argument("--refinement-steps", type=int, default=16)
    parser.add_argument("--shortcut-passes", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--tokens-per-footprint-area", type=float, default=None)
    parser.add_argument("--minimum-path-tokens", type=int, default=None)
    parser.add_argument("--maximum-path-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PathVectorField(PathVectorFieldConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    if args.split == "validation":
        instance_ids = checkpoint["validation_instance_ids"]
    elif args.split == "train":
        instance_ids = checkpoint["train_instance_ids"]
    else:
        instance_ids = checkpoint["train_instance_ids"] + checkpoint["validation_instance_ids"]
    if not instance_ids:
        raise ValueError(f"checkpoint has no instances for split {args.split}")
    dataset = TeacherPathDataset(
        args.dataset,
        instance_ids=instance_ids,
        num_surface_points=int(checkpoint["num_surface_points"]),
        num_path_waypoints=(
            None
            if checkpoint.get("path_representation", "fixed") == "variable"
            else int(checkpoint["num_path_waypoints"])
        ),
        tokens_per_footprint_area=float(checkpoint.get("tokens_per_footprint_area", 1.0)),
        minimum_path_tokens=int(checkpoint.get("minimum_path_tokens", 32)),
        maximum_path_tokens=int(checkpoint.get("maximum_path_tokens", 2048)),
        candidate_policy=str(checkpoint.get("candidate_policy", "all")),
        candidate_name=checkpoint.get("candidate_name"),
        allow_repaired_candidates=bool(
            checkpoint.get("allow_repaired_candidates", True)
        ),
        path_coordinate_system=checkpoint.get("path_coordinate_system", "xyz"),
        include_mode_conditioning=bool(checkpoint.get("include_mode_conditioning", False)),
        preserve_source_waypoints=bool(checkpoint.get("preserve_source_waypoints", False)),
        seed=int(checkpoint.get("data_seed", checkpoint["seed"])),
    )
    row_by_id = {str(row["instance_id"]): row for row in dataset.rows}
    first_sample_by_id: dict[str, int] = {}
    for index, sample_ref in enumerate(dataset.sample_index):
        instance_id = str(dataset.rows[sample_ref.instance_index]["instance_id"])
        first_sample_by_id.setdefault(instance_id, index)

    torch.manual_seed(args.seed)
    records: list[dict[str, object]] = []
    selected_samples = list(first_sample_by_id.items())
    if args.max_instances is not None:
        selected_samples = selected_samples[: args.max_instances]
    for instance_id, sample_index in selected_samples:
        sample = dataset[sample_index]
        row = row_by_id[instance_id]
        archive = load_teacher_instance(args.dataset / str(row["path"]))
        metadata = archive["metadata"]
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        radius = float(metadata["teacher_config"]["footprint_radius"])
        epsilon = float(metadata["teacher_config"]["missed_tolerance"])
        surface_tensor = sample["surface"].unsqueeze(0).repeat(args.samples_per_instance, 1, 1).to(device)
        condition = sample["condition"].unsqueeze(0).repeat(args.samples_per_instance, 1).to(device)
        coordinate_system = checkpoint.get("path_coordinate_system", "xyz")
        if checkpoint.get("path_representation", "fixed") == "variable":
            if coordinate_system == "analytic_uv_control":
                num_waypoints = raster_control_token_count(
                    surface,
                    footprint_radius=radius,
                    overlap=float(metadata["teacher_config"]["overlap"]),
                    sweep_axis="u",
                )
            elif coordinate_system in {
                "analytic_uv_structured", "analytic_uv_structured_residual"
            }:
                num_waypoints = structured_control_token_count(
                    surface,
                    footprint_radius=radius,
                    overlap=float(metadata["teacher_config"]["overlap"]),
                    mode_name=str(sample["mode_name"]),
                )
            else:
                num_waypoints = suggested_token_count(
                    surface.total_area,
                    radius,
                    tokens_per_footprint_area=(
                        float(checkpoint.get("tokens_per_footprint_area", 1.0))
                        if args.tokens_per_footprint_area is None
                        else args.tokens_per_footprint_area
                    ),
                    minimum=(
                        int(checkpoint.get("minimum_path_tokens", 32))
                        if args.minimum_path_tokens is None
                        else args.minimum_path_tokens
                    ),
                    maximum=(
                        int(checkpoint.get("maximum_path_tokens", 2048))
                        if args.maximum_path_tokens is None
                        else args.maximum_path_tokens
                    ),
                )
            path_mask = torch.ones(
                args.samples_per_instance, num_waypoints, dtype=torch.bool, device=device
            )
            path_arclength = torch.linspace(
                0.0, 1.0, num_waypoints, device=device
            ).unsqueeze(0).repeat(args.samples_per_instance, 1)
        else:
            num_waypoints = int(sample["path"].shape[0])
            path_mask = sample["path_mask"].unsqueeze(0).repeat(args.samples_per_instance, 1).to(device)
            path_arclength = sample["path_arclength"].unsqueeze(0).repeat(args.samples_per_instance, 1).to(device)
        pipeline_start = perf_counter()
        start = pipeline_start
        generated = heun_sample(
            model,
            surface_tensor,
            condition,
            num_waypoints=num_waypoints,
            num_steps=args.ode_steps,
            noise_smoothing_sigma=float(checkpoint.get("noise_smoothing_sigma", 0.0)),
            noise_smoothing_fraction=float(checkpoint.get("noise_smoothing_fraction", 0.0)),
            path_mask=path_mask,
            path_arclength=path_arclength,
        )
        generation_time = perf_counter() - start
        generated_paths = generated.cpu().numpy().astype(np.float64)
        if coordinate_system in {
            "analytic_uv", "analytic_uv_control", "analytic_uv_structured",
            "analytic_uv_structured_residual",
        }:
            world_paths = []
            for path in generated_paths:
                parameters = (
                    decode_raster_parameter_controls(
                        surface,
                        path,
                        footprint_radius=radius,
                        sweep_axis="u",
                    )
                    if coordinate_system == "analytic_uv_control"
                    else (
                        decode_structured_parameter_controls(
                            surface,
                            (
                                structured_residual_to_controls(
                                    surface,
                                    path,
                                    footprint_radius=radius,
                                    overlap=float(metadata["teacher_config"]["overlap"]),
                                    mode_name=str(sample["mode_name"]),
                                )
                                if coordinate_system == "analytic_uv_structured_residual"
                                else path
                            ),
                            footprint_radius=radius,
                            mode_name=str(sample["mode_name"]),
                        )
                        if coordinate_system in {
                            "analytic_uv_structured", "analytic_uv_structured_residual"
                        }
                        else path
                    )
                )
                world_paths.append(
                    map_surface_parameters(surface, parameters[:, 0], parameters[:, 1])
                )
        else:
            center = sample["center"].numpy().astype(np.float64)
            scale = float(sample["scale"])
            world_paths = generated_paths * scale + center
        checker_start = perf_counter()
        candidates: list[tuple[float, float, np.ndarray]] = []
        anytime_trace: list[dict[str, float | int | bool]] = []
        best_so_far: tuple[float, float] | None = None
        for candidate_number, path in enumerate(world_paths, start=1):
            projected = project_points(surface, path).points
            metrics = evaluate_coverage(surface, CoveragePlan(projected), footprint_radius=radius)
            candidates.append((metrics.missed_fraction, metrics.path_length, projected))
            objective = (metrics.missed_fraction, metrics.path_length)
            if best_so_far is None or proposal_key(*objective, epsilon) < proposal_key(*best_so_far, epsilon):
                best_so_far = objective
            assert best_so_far is not None
            anytime_trace.append({
                "samples_checked": candidate_number,
                "elapsed_time": perf_counter() - pipeline_start,
                "best_feasible": best_so_far[0] <= epsilon + 1e-12,
                "best_missed_fraction": best_so_far[0],
                "best_path_length": best_so_far[1],
            })
        candidates.sort(key=lambda item: (int(item[0] > epsilon), max(0.0, item[0] - epsilon), item[1]))
        raw_best_missed, raw_best_length, raw_best_path = candidates[0]
        refined_candidates: list[tuple[float, float, np.ndarray, int, int]] = []
        for missed, length, path in candidates[:args.refine_top_k]:
            refined = refine_coverage_by_insertion(
                surface,
                CoveragePlan(path),
                footprint_radius=radius,
                missed_tolerance=epsilon,
                max_steps=args.refinement_steps,
            )
            shortened = refine_coverage_by_shortcutting(
                surface,
                refined.plan,
                footprint_radius=radius,
                missed_tolerance=epsilon,
                max_passes=args.shortcut_passes,
            )
            refined_candidates.append(
                (
                    shortened.metrics.missed_fraction,
                    shortened.metrics.path_length,
                    shortened.plan.active_paths()[0],
                    refined.steps,
                    shortened.steps,
                )
            )
        if refined_candidates:
            refined_candidates.sort(
                key=lambda item: (int(item[0] > epsilon), max(0.0, item[0] - epsilon), item[1])
            )
            (
                best_missed,
                best_length,
                best_path,
                refinement_steps,
                shortcut_passes,
            ) = refined_candidates[0]
        else:
            best_missed, best_length, best_path = raw_best_missed, raw_best_length, raw_best_path
            refinement_steps = 0
            shortcut_passes = 0
        checker_time = perf_counter() - checker_start
        total_pipeline_time = perf_counter() - pipeline_start
        teacher_metrics = np.asarray(archive["candidate_metrics"])
        teacher_indices = candidate_indices_by_name(
            archive,
            checkpoint.get("candidate_name"),
            allow_repaired=bool(checkpoint.get("allow_repaired_candidates", True)),
        )
        teacher_best_index = min(
            teacher_indices,
            key=lambda i: constrained_coverage_key(
                float(teacher_metrics[i, 0]), float(teacher_metrics[i, 1]), epsilon
            ),
        )
        record = {
            "instance_id": instance_id,
            "surface_id": str(row["surface_id"]),
            "footprint_radius": radius,
            "surface_area": surface.total_area,
            "samples": args.samples_per_instance,
            "num_path_tokens": num_waypoints,
            "generated_feasible": bool(best_missed <= epsilon + 1e-12),
            "raw_generated_feasible": bool(raw_best_missed <= epsilon + 1e-12),
            "raw_best_missed_fraction": raw_best_missed,
            "raw_best_path_length": raw_best_length,
            "best_missed_fraction": best_missed,
            "best_path_length": best_length,
            "refinement_steps": refinement_steps,
            "shortcut_passes": shortcut_passes,
            "teacher_missed_fraction": float(teacher_metrics[teacher_best_index, 0]),
            "teacher_path_length": float(teacher_metrics[teacher_best_index, 1]),
            "generation_time": generation_time,
            "hard_checker_time": checker_time,
            "total_pipeline_time": total_pipeline_time,
            "anytime_trace": anytime_trace,
            "teacher_total_solve_time": float(metadata["teacher_runtime"]["total_solve_time"]),
        }
        records.append(record)
        plot_path(surface, best_path, args.output / f"{instance_id}.png")
        print(
            f"{instance_id:<24} feasible={record['generated_feasible']} "
            f"raw={raw_best_missed:.4f} refined={best_missed:.4f} length={best_length:.3f}"
        )
    summary = summarize_records(records)
    summary["by_surface"] = {
        surface_id: summarize_records(
            [record for record in records if record["surface_id"] == surface_id]
        )
        for surface_id in sorted({str(record["surface_id"]) for record in records})
    }
    (args.output / "metrics.json").write_text(
        json.dumps({"summary": summary, "instances": records}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def proposal_key(missed: float, length: float, epsilon: float) -> tuple[int, float, float]:
    return constrained_coverage_key(missed, length, epsilon)


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize empty records")
    radii = np.asarray([record["footprint_radius"] for record in records], dtype=float)
    generated_lengths = np.asarray([record["best_path_length"] for record in records], dtype=float)
    teacher_lengths = np.asarray([record["teacher_path_length"] for record in records], dtype=float)
    areas = np.asarray([record["surface_area"] for record in records], dtype=float)
    maximum_samples = min(len(record["anytime_trace"]) for record in records)
    checkpoints = sorted({1, maximum_samples, *[value for value in (2, 4, 8, 16, 32) if value <= maximum_samples]})
    anytime = {}
    for samples in checkpoints:
        traces = [record["anytime_trace"][samples - 1] for record in records]
        anytime[str(samples)] = {
            "feasible_rate": float(np.mean([trace["best_feasible"] for trace in traces])),
            "mean_best_missed_fraction": float(np.mean([trace["best_missed_fraction"] for trace in traces])),
            "mean_elapsed_time": float(np.mean([trace["elapsed_time"] for trace in traces])),
            "median_elapsed_time": float(np.median([trace["elapsed_time"] for trace in traces])),
        }
    return {
        "num_instances": len(records),
        "feasible_rate": float(np.mean([record["generated_feasible"] for record in records])),
        "raw_feasible_rate": float(np.mean([record["raw_generated_feasible"] for record in records])),
        "mean_missed_fraction": float(np.mean([record["best_missed_fraction"] for record in records])),
        "mean_generation_time": float(np.mean([record["generation_time"] for record in records])),
        "mean_hard_checker_time": float(np.mean([record["hard_checker_time"] for record in records])),
        "mean_total_pipeline_time": float(np.mean([record["total_pipeline_time"] for record in records])),
        "median_total_pipeline_time": float(np.median([record["total_pipeline_time"] for record in records])),
        "minimum_path_tokens": int(min(record["num_path_tokens"] for record in records)),
        "maximum_path_tokens": int(max(record["num_path_tokens"] for record in records)),
        "mean_path_tokens": float(np.mean([record["num_path_tokens"] for record in records])),
        "mean_generated_path_length": float(np.mean([record["best_path_length"] for record in records])),
        "mean_teacher_path_length": float(np.mean([record["teacher_path_length"] for record in records])),
        "mean_length_ratio_to_teacher": float(np.mean([
            record["best_path_length"] / record["teacher_path_length"] for record in records
        ])),
        "mean_normalized_footprint_length": float(np.mean(radii * generated_lengths / areas)),
        "mean_teacher_normalized_footprint_length": float(np.mean(radii * teacher_lengths / areas)),
        "log_radius_log_length_correlation": safe_correlation(np.log(radii), np.log(generated_lengths)),
        "teacher_log_radius_log_length_correlation": safe_correlation(np.log(radii), np.log(teacher_lengths)),
        "mean_teacher_total_solve_time": float(np.mean([record["teacher_total_solve_time"] for record in records])),
        "median_teacher_total_solve_time": float(np.median([record["teacher_total_solve_time"] for record in records])),
        "anytime_by_samples": anytime,
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def plot_path(surface: SurfaceInstance, path: np.ndarray, output: Path) -> None:
    figure = plt.figure(figsize=(6, 5))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_trisurf(
        surface.vertices[:, 0], surface.vertices[:, 1], surface.faces, surface.vertices[:, 2],
        color="#d9dde2", edgecolor="#8a929b", linewidth=0.15, alpha=0.55,
    )
    axis.plot(path[:, 0], path[:, 1], path[:, 2], color="#c43b35", linewidth=1.2)
    axis.set_box_aspect(np.ptp(surface.vertices, axis=0) + 1e-6)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
