#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.coverage import (
    CoveragePlan,
    decode_structured_parameter_controls,
    evaluate_coverage,
    load_teacher_instance,
    map_surface_parameters,
    structured_controls_to_residual,
    structured_residual_to_controls,
    surface_from_teacher_archive,
)
from diffusion_coverage.learning import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel full-corpus float32 structured-residual contract audit"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.dataset)
    start = perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(audit_instance, (str(args.dataset), row)) for row in rows
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            instance_records = future.result()
            records.extend(instance_records)
            feasible = sum(record["feasible"] for record in instance_records)
            print(
                f"[{completed:03d}/{len(rows):03d}] {instance_records[0]['instance_id']:<24} "
                f"feasible={feasible}/{len(instance_records)}",
                flush=True,
            )
    records.sort(key=lambda row: (row["instance_id"], row["candidate_index"]))
    summary = summarize(records)
    summary["unique_instances"] = len({record["instance_id"] for record in records})
    summary["elapsed_seconds"] = perf_counter() - start
    summary["by_surface"] = {
        surface: summarize([record for record in records if record["surface_id"] == surface])
        for surface in sorted({record["surface_id"] for record in records})
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "candidates": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def audit_instance(task):
    dataset_raw, row = task
    dataset = Path(dataset_raw)
    archive = load_teacher_instance(dataset / str(row["path"]))
    surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
    metadata = dict(archive["metadata"])
    config = metadata["teacher_config"]
    radius = float(config["footprint_radius"])
    overlap = float(config["overlap"])
    tolerance = float(config["missed_tolerance"])
    records = []
    for candidate_index, mode_raw in enumerate(archive["proposal_names"]):
        mode = str(mode_raw)
        control_mask = np.asarray(archive["candidate_control_mask"])[candidate_index]
        controls = np.asarray(archive["candidate_controls"])[candidate_index][control_mask]
        residual = structured_controls_to_residual(
            surface,
            controls,
            footprint_radius=radius,
            overlap=overlap,
            mode_name=mode,
        )
        reconstructed = structured_residual_to_controls(
            surface,
            residual.astype(np.float32),
            footprint_radius=radius,
            overlap=overlap,
            mode_name=mode,
        )
        parameters = decode_structured_parameter_controls(
            surface,
            reconstructed,
            footprint_radius=radius,
            mode_name=mode,
        )
        world = map_surface_parameters(surface, parameters[:, 0], parameters[:, 1])
        metrics = evaluate_coverage(surface, CoveragePlan(world), footprint_radius=radius)
        source = np.asarray(archive["candidate_metrics"])[candidate_index]
        records.append(
            {
                "instance_id": str(row["instance_id"]),
                "surface_id": str(row["surface_id"]),
                "candidate_index": candidate_index,
                "mode_name": mode,
                "num_controls": len(controls),
                "num_path_points": len(world),
                "feasible": metrics.missed_fraction <= tolerance + 1e-12,
                "missed_fraction": metrics.missed_fraction,
                "source_missed_fraction": float(source[0]),
                "path_length": metrics.path_length,
                "source_path_length": float(source[1]),
            }
        )
    return records


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    absolute_missed = [
        abs(float(record["missed_fraction"]) - float(record["source_missed_fraction"]))
        for record in records
    ]
    return {
        "candidates": len(records),
        "feasible_candidates": int(sum(record["feasible"] for record in records)),
        "feasible_rate": float(np.mean([record["feasible"] for record in records])),
        "mean_absolute_missed_change": float(np.mean(absolute_missed)),
        "maximum_absolute_missed_change": float(np.max(absolute_missed)),
        "mean_length_ratio": float(
            np.mean(
                [
                    float(record["path_length"]) / float(record["source_path_length"])
                    for record in records
                ]
            )
        ),
        "maximum_controls": int(max(record["num_controls"] for record in records)),
    }


if __name__ == "__main__":
    main()
