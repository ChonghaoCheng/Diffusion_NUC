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

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.coverage import (
    CoveragePlan,
    evaluate_coverage,
    extract_structured_parameter_controls,
    generate_pattern_proposals,
    load_teacher_instance,
    parse_pattern_mode,
    surface_from_teacher_archive,
)
from diffusion_coverage.coverage.patterns import _intrinsic_extents, _periodicity
from diffusion_coverage.learning import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit learned freedom beyond deterministic structured templates"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    rows = load_manifest(args.dataset)
    for row_number, row in enumerate(rows, start=1):
        archive = load_teacher_instance(args.dataset / str(row["path"]))
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        metadata = archive["metadata"]
        config = metadata["teacher_config"]
        radius = float(config["footprint_radius"])
        epsilon = float(config["missed_tolerance"])
        overlap = float(config["overlap"])
        proposals = {
            proposal.name: proposal
            for proposal in generate_pattern_proposals(
                surface,
                footprint_radius=radius,
                max_segments=int(config["max_segments"]),
                overlap=overlap,
                waypoint_spacing=(
                    None
                    if config.get("waypoint_spacing") is None
                    else float(config["waypoint_spacing"])
                ),
            )
        }
        for candidate_index, raw_name in enumerate(archive["proposal_names"]):
            mode = parse_pattern_mode(str(raw_name))
            target_plan = CoveragePlan(
                archive["candidate_waypoints"][candidate_index],
                archive["candidate_segment_mask"][candidate_index],
                archive["candidate_waypoint_mask"][candidate_index],
            )
            template_plan = proposals[mode.name].plan
            target_controls = extract_structured_parameter_controls(
                surface, target_plan.active_paths()[0], mode_name=mode.name
            )
            template_controls = extract_structured_parameter_controls(
                surface, template_plan.active_paths()[0], mode_name=mode.name
            )
            if target_controls.shape != template_controls.shape:
                raise ValueError(
                    f"control shape mismatch for {row['instance_id']} {mode.name}: "
                    f"{target_controls.shape} != {template_controls.shape}"
                )
            residual = aligned_parameter_residual(
                target_controls, template_controls, surface
            )
            u_extent, v_extent = _intrinsic_extents(surface)
            physical_residual = residual * np.asarray([u_extent, v_extent])
            template_metrics = evaluate_coverage(
                surface, template_plan, footprint_radius=radius
            )
            source_metrics = np.asarray(archive["candidate_metrics"], dtype=np.float64)[
                candidate_index
            ]
            rms = float(np.sqrt(np.mean(np.sum(physical_residual**2, axis=1))))
            maximum = float(np.max(np.linalg.norm(physical_residual, axis=1)))
            records.append(
                {
                    "instance_id": str(row["instance_id"]),
                    "surface_id": str(row["surface_id"]),
                    "mode": mode.name,
                    "refinement_iterations": int(config["refinement_iterations"]),
                    "num_controls": len(target_controls),
                    "control_rms": rms,
                    "control_max": maximum,
                    "control_rms_per_radius": rms / radius,
                    "control_max_per_radius": maximum / radius,
                    "fraction_controls_above_0_01_radius": float(
                        np.mean(np.linalg.norm(physical_residual, axis=1) > 0.01 * radius)
                    ),
                    "fraction_controls_above_0_10_radius": float(
                        np.mean(np.linalg.norm(physical_residual, axis=1) > 0.10 * radius)
                    ),
                    "teacher_feasible": bool(float(source_metrics[0]) <= epsilon + 1e-12),
                    "teacher_missed_fraction": float(source_metrics[0]),
                    "teacher_path_length": float(source_metrics[1]),
                    "template_feasible": bool(
                        template_metrics.missed_fraction <= epsilon + 1e-12
                    ),
                    "template_missed_fraction": template_metrics.missed_fraction,
                    "template_path_length": template_metrics.path_length,
                    "template_teacher_length_ratio": (
                        template_metrics.path_length / float(source_metrics[1])
                    ),
                }
            )
        print(
            f"[{row_number:03d}/{len(rows):03d}] {row['instance_id']}", flush=True
        )

    summary = summarize(records)
    summary["by_mode"] = {
        mode: summarize([record for record in records if record["mode"] == mode])
        for mode in sorted({str(record["mode"]) for record in records})
    }
    summary["by_surface"] = {
        surface_id: summarize(
            [record for record in records if record["surface_id"] == surface_id]
        )
        for surface_id in sorted({str(record["surface_id"]) for record in records})
    }
    with (args.output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output / "metrics.json").write_text(
        json.dumps({"summary": summary}, indent=2) + "\n"
    )
    plot_results(records, summary, args.output / "structured_mode_freedom.png")
    print(json.dumps(summary, indent=2))


def aligned_parameter_residual(
    target: np.ndarray,
    template: np.ndarray,
    surface,
) -> np.ndarray:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        template, dtype=np.float64
    )
    periodic_u, periodic_v = _periodicity(surface)
    for axis, periodic in enumerate((periodic_u, periodic_v)):
        if periodic:
            residual[:, axis] -= np.round(np.median(residual[:, axis]))
    return residual


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    rms = np.asarray([record["control_rms_per_radius"] for record in records], dtype=float)
    maximum = np.asarray(
        [record["control_max_per_radius"] for record in records], dtype=float
    )
    ratios = np.asarray(
        [record["template_teacher_length_ratio"] for record in records], dtype=float
    )
    exact = np.asarray(
        [
            float(record["control_max_per_radius"]) <= 1e-6
            for record in records
        ],
        dtype=bool,
    )
    return {
        "candidates": len(records),
        "instances": len({str(record["instance_id"]) for record in records}),
        "all_refinement_iterations_zero": all(
            int(record["refinement_iterations"]) == 0 for record in records
        ),
        "exact_template_match_rate": float(np.mean(exact)),
        "mean_control_rms_per_radius": float(np.mean(rms)),
        "median_control_rms_per_radius": float(np.median(rms)),
        "p95_control_rms_per_radius": float(np.quantile(rms, 0.95)),
        "maximum_control_error_per_radius": float(np.max(maximum)),
        "mean_fraction_controls_above_0_01_radius": float(
            np.mean(
                [record["fraction_controls_above_0_01_radius"] for record in records]
            )
        ),
        "mean_fraction_controls_above_0_10_radius": float(
            np.mean(
                [record["fraction_controls_above_0_10_radius"] for record in records]
            )
        ),
        "teacher_feasible_rate": float(
            np.mean([record["teacher_feasible"] for record in records])
        ),
        "deterministic_template_feasible_rate": float(
            np.mean([record["template_feasible"] for record in records])
        ),
        "mean_template_teacher_length_ratio": float(np.mean(ratios)),
        "median_template_teacher_length_ratio": float(np.median(ratios)),
        "maximum_template_teacher_length_ratio": float(np.max(ratios)),
    }


def plot_results(
    records: list[dict[str, object]],
    summary: dict[str, object],
    output: Path,
) -> None:
    by_mode = summary["by_mode"]
    assert isinstance(by_mode, dict)
    modes = list(by_mode)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    values = np.asarray(
        [record["control_rms_per_radius"] for record in records], dtype=float
    )
    axes[0].hist(values, bins=30, color="#2f6f8f")
    axes[0].set_xlabel("Control RMS / footprint radius")
    axes[0].set_ylabel("Candidates")
    axes[0].set_yscale("log")
    x = np.arange(len(modes))
    axes[1].bar(
        x,
        [float(by_mode[mode]["deterministic_template_feasible_rate"]) for mode in modes],
        color="#b44c43",
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Deterministic template feasible rate")
    axes[1].set_xticks(
        x, [mode.replace("_phase_", "\nphase ") for mode in modes]
    )
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
