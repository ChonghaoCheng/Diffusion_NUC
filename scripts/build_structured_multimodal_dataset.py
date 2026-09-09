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

from collections import Counter, defaultdict

import numpy as np

from diffusion_coverage.coverage import (
    CoveragePlan,
    decode_structured_parameter_controls,
    evaluate_coverage,
    extract_structured_parameter_controls,
    load_teacher_instance,
    map_surface_parameters,
    parse_pattern_mode,
    structured_control_token_count,
    surface_from_teacher_archive,
)
from diffusion_coverage.learning import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-checked structured multimodal targets")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances_dir = args.output / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"output manifest already exists: {manifest_path}")

    admitted = Counter()
    rejected = Counter()
    length_ratios: dict[str, list[float]] = defaultdict(list)
    output_rows = []
    for row in load_manifest(args.input):
        archive = load_teacher_instance(args.input / str(row["path"]))
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        metadata = dict(archive["metadata"])
        config = metadata["teacher_config"]
        radius = float(config["footprint_radius"])
        epsilon = float(config["missed_tolerance"])
        overlap = float(config["overlap"])
        keep = []
        for index, raw_name in enumerate(archive["proposal_names"]):
            raw_name = str(raw_name)
            if raw_name.endswith("_robustness_repair"):
                rejected["repaired"] += 1
                continue
            mode = parse_pattern_mode(raw_name)
            plan = CoveragePlan(
                archive["candidate_waypoints"][index],
                archive["candidate_segment_mask"][index],
                archive["candidate_waypoint_mask"][index],
            )
            controls = extract_structured_parameter_controls(
                surface,
                plan.active_paths()[0],
                mode_name=mode.name,
            ).astype(np.float32).astype(np.float64)
            expected = structured_control_token_count(
                surface,
                footprint_radius=radius,
                overlap=overlap,
                mode_name=mode.name,
            )
            if len(controls) != expected:
                rejected[f"{mode.name}:token_count"] += 1
                continue
            parameters = decode_structured_parameter_controls(
                surface,
                controls,
                footprint_radius=radius,
                mode_name=mode.name,
            )
            decoded = map_surface_parameters(
                surface, parameters[:, 0], parameters[:, 1]
            )
            metrics = evaluate_coverage(
                surface, CoveragePlan(decoded), footprint_radius=radius
            )
            if metrics.missed_fraction > epsilon + 1e-12:
                rejected[f"{mode.name}:hard_infeasible"] += 1
                continue
            source_length = float(np.asarray(archive["candidate_metrics"])[index, 1])
            length_ratios[mode.name].append(metrics.path_length / source_length)
            admitted[mode.name] += 1
            keep.append(index)
        if not keep:
            rejected["empty_instance"] += 1
            continue

        keep_array = np.asarray(keep, dtype=np.int64)
        output_path = instances_dir / f"{row['instance_id']}.npz"
        metadata["structured_multimodal_contract"] = {
            "coordinate_system": "analytic_uv_structured",
            "allow_repaired_candidates": False,
            "hard_roundtrip_checked": True,
        }
        np.savez_compressed(
            output_path,
            vertices=np.asarray(archive["vertices"]),
            faces=np.asarray(archive["faces"]),
            sample_points=np.asarray(archive["sample_points"]),
            sample_normals=np.asarray(archive["sample_normals"]),
            area_weights=np.asarray(archive["area_weights"]),
            sample_face_indices=np.asarray(archive["sample_face_indices"]),
            sample_barycentric=np.asarray(archive["sample_barycentric"]),
            candidate_waypoints=np.asarray(archive["candidate_waypoints"])[keep_array],
            candidate_segment_mask=np.asarray(archive["candidate_segment_mask"])[keep_array],
            candidate_waypoint_mask=np.asarray(archive["candidate_waypoint_mask"])[keep_array],
            candidate_metrics=np.asarray(archive["candidate_metrics"])[keep_array],
            proposal_names=np.asarray(archive["proposal_names"])[keep_array],
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
        output_row = dict(row)
        output_row["path"] = str(output_path.relative_to(args.output))
        output_row["num_candidates"] = len(keep)
        output_row["num_feasible_candidates"] = len(keep)
        output_rows.append(output_row)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "input_instances": len(load_manifest(args.input)),
        "output_instances": len(output_rows),
        "admitted_candidates": dict(sorted(admitted.items())),
        "rejected": dict(sorted(rejected.items())),
        "mean_length_ratio": {
            mode: float(np.mean(values)) for mode, values in sorted(length_ratios.items())
        },
        "maximum_length_ratio": {
            mode: float(np.max(values)) for mode, values in sorted(length_ratios.items())
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
