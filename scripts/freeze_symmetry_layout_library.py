#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract
from diffusion_coverage.diagnostics.symmetry_layout import (
    array_hash,
    assert_invariance,
    choose_invariance_tolerance,
    evaluate_symmetry_path,
    intrinsic_segment_lengths,
    load_canonical_path,
    make_symmetry_reference_surface,
    symmetry_specs,
    transport_coverage_metrics,
    transform_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze E06-G2 exact symmetry orbits")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/symmetry_preserving_global_layout_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/symmetry_preserving_global_layout_v1")
    return parser.parse_args()


def main():
    args = parse_args(); config = json.loads(args.config.read_text()); archived, rows, _ = load_e06_contract(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}; canonical_records = []; orbit_records = []; geometry_rows = []
    for surface_id in ("saddle", "hemisphere"):
        canonical = load_canonical_path(
            ROOT, archived, surface_id, config["surfaces"][surface_id],
            maximum_spacing=float(config["coverage"]["path_sample_spacing_m"]),
        )
        reference = make_symmetry_reference_surface(surface_id, config)
        baseline_metrics = evaluate_symmetry_path(reference, canonical.points, config)
        baseline_segments = intrinsic_segment_lengths(reference, canonical.points)
        source_row = next(
            row for row in rows
            if row["surface_id"] == surface_id and row["placement_id"] == "P_easy" and row["skeleton_id"] == "S00"
        )
        arrays[f"{surface_id}_canonical_points"] = canonical.points
        arrays[f"{surface_id}_canonical_normals"] = canonical.normals
        arrays[f"{surface_id}_source_q_start"] = canonical.source_q_start
        canonical_records.append({
            "surface_id": surface_id,
            "source_artifact": canonical.source_path,
            "source_commit": config["source_code_commit"],
            "source_policy": "upstream_first",
            "source_skeleton_id": "S00",
            "source_sample_count": canonical.source_sample_count,
            "source_reported_E_miss": source_row["E_miss"],
            "source_reported_E_rep": source_row["E_rep"],
            "source_reported_E_NUC": source_row["E_NUC"],
            "source_reported_L_S": source_row["surface_path_length"],
            "analytical_projection_max_m": canonical.source_max_projection_m,
            "canonical_path_hash": array_hash(canonical.points, canonical.normals),
            "canonical_sample_count": int(len(canonical.points)),
            "E_miss": baseline_metrics.missed_error,
            "E_rep": baseline_metrics.repeat_error,
            "E_NUC": baseline_metrics.nuc_error,
            "L_S": baseline_metrics.path_length,
            "discrete_segment_length": float(baseline_segments.sum()),
            "reference_vertices": reference.num_vertices,
            "reference_faces": reference.num_faces,
            "reference_samples": reference.num_samples,
        })
        for spec in symmetry_specs(surface_id, config):
            points, normals = transform_path(canonical, spec, config["surfaces"][surface_id])
            metrics, automorphism_error = transport_coverage_metrics(
                baseline_metrics, reference, np.asarray(spec["matrix"]), points
            )
            segments = intrinsic_segment_lengths(reference, points)
            key = f"{surface_id}_{spec['symmetry_id']}"
            arrays[f"{key}_points"] = points; arrays[f"{key}_normals"] = normals
            normal_expected = canonical.normals @ np.asarray(spec["matrix"]).T
            row = {
                "surface_id": surface_id,
                "symmetry_id": spec["symmetry_id"],
                "symmetry_name": spec.get("name", "rotation_z"),
                "angle_degrees": spec.get("angle_degrees"),
                "sample_count": int(len(points)),
                "sample_order_preserved": True,
                "activity_preserved": True,
                "topology_hash": array_hash(np.arange(len(points), dtype=np.int64)),
                "path_hash": array_hash(points, normals),
                "E_miss": metrics.missed_error,
                "E_rep": metrics.repeat_error,
                "E_NUC": metrics.nuc_error,
                "L_S": metrics.path_length,
                "abs_E_miss_error": abs(metrics.missed_error - baseline_metrics.missed_error),
                "abs_E_rep_error": abs(metrics.repeat_error - baseline_metrics.repeat_error),
                "abs_E_NUC_error": abs(metrics.nuc_error - baseline_metrics.nuc_error),
                "abs_L_S_error": abs(metrics.path_length - baseline_metrics.path_length),
                "max_segment_length_error": float(np.max(np.abs(segments - baseline_segments), initial=0.0)),
                "normal_covariance_error": float(np.max(np.linalg.norm(normals - normal_expected, axis=1))),
                "normal_covariance_pass": bool(np.allclose(normals, normal_expected, atol=1e-12, rtol=0)),
                "reference_automorphism_error": automorphism_error,
                "evaluation_mode": "transported canonical membership under verified mesh/sample automorphism",
            }
            geometry_rows.append(row)
            orbit_records.append({
                "surface_id": surface_id,
                "symmetry_id": spec["symmetry_id"],
                "symmetry_name": row["symmetry_name"],
                "angle_degrees": row["angle_degrees"],
                "matrix": np.asarray(spec["matrix"], dtype=np.float64).tolist(),
                "path_hash": row["path_hash"],
                "array_keys": {"points": f"{key}_points", "normals": f"{key}_normals"},
            })
            print(surface_id, spec["symmetry_id"], f"ENUC={metrics.nuc_error:.12f}", flush=True)
    tolerance = choose_invariance_tolerance(geometry_rows, float(config["coverage"]["invariance_hard_ceiling"]))
    assert_invariance(geometry_rows, tolerance)
    hemisphere = [row for row in orbit_records if row["surface_id"] == "hemisphere"]
    closure = np.asarray(hemisphere[-1]["matrix"]) @ np.asarray(hemisphere[1]["matrix"])
    if not np.allclose(closure, np.eye(3), atol=1e-12, rtol=0):
        raise RuntimeError("24-member hemisphere orbit does not close")
    np.savez_compressed(args.output / "symmetry_orbits.npz", **arrays)
    canonical_doc = {
        "frozen_before_robot_execution": True,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "canonicalization": "E06 P_easy S00 base-frame target trace -> object frame -> one analytical projection",
        "paths": canonical_records,
    }
    orbit_doc = {
        "frozen_before_robot_execution": True,
        "code_commit": canonical_doc["code_commit"],
        "invariance_tolerance": tolerance,
        "invariance_hard_ceiling": config["coverage"]["invariance_hard_ceiling"],
        "array_file": "symmetry_orbits.npz",
        "orbit_hash": array_hash(*[arrays[key] for key in sorted(arrays)]),
        "orbits": orbit_records,
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "canonical_paths.json", canonical_doc)
    write_json(args.output / "frozen_symmetry_orbits.json", orbit_doc)
    write_csv(args.output / "geometry_invariance.csv", geometry_rows)
    make_overlay(args.output, arrays, orbit_records)
    print(json.dumps({"canonical": canonical_records, "orbit_count": len(orbit_records), "invariance_tolerance": tolerance, "orbit_hash": orbit_doc["orbit_hash"]}, indent=2))


def make_overlay(output, arrays, records):
    figure = plt.figure(figsize=(11, 5))
    for plot_index, surface_id in enumerate(("saddle", "hemisphere"), start=1):
        axis = figure.add_subplot(1, 2, plot_index, projection="3d")
        for row in records:
            if row["surface_id"] != surface_id: continue
            points = arrays[row["array_keys"]["points"]]
            stride = max(1, len(points) // 800)
            axis.plot(points[::stride, 0], points[::stride, 1], points[::stride, 2], alpha=.35, linewidth=.6)
        axis.set_title(f"{surface_id} symmetry orbit")
        axis.set_box_aspect((1, 1, .6))
    figure.tight_layout(); figure.savefig(output / "hemisphere_paths_overlay.png", dpi=180); plt.close(figure)


def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
