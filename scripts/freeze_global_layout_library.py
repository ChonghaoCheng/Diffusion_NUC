#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.diagnostics.global_layout import (
    aligned_ordered_path,
    generate_layout,
    generate_physical_roots,
    make_planning_remesh,
    make_reference_surface,
    map_physical_root,
    remesh_statistics,
    select_geometry_baseline,
    surface_hash,
    validate_remesh_library,
)
from diffusion_coverage.nuc.adapter import generate_nuc_skeleton
from diffusion_coverage.surface.geodesic import geodesic_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze E06-G roots, remeshes, and layouts")
    parser.add_argument("--stage", required=True, choices=("smoke", "freeze"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/global_layout_capacity_gate_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/global_layout_capacity_gate_v1")
    return parser.parse_args()


def main():
    args = parse_args(); config = json.loads(args.config.read_text()); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        run_smoke(config, args.output)
    else:
        smoke_path = args.output / "stage_a_smoke.json"
        if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get("passed"):
            raise RuntimeError("Stage A smoke must pass before freezing E06-G")
        freeze(config, args.output)


def run_smoke(config, output):
    surface_id = "saddle"; reference = make_reference_surface(surface_id, config)
    remeshes = {name: make_planning_remesh(surface_id, name, config) for name in ("M00", "M01")}
    records = [{"surface_id": surface_id, "remesh_id": name, **remesh_statistics(surface_id, mesh, reference, config)} for name, mesh in remeshes.items()]
    validate_remesh_library(records, config)
    roots = generate_physical_roots(reference, remeshes["M00"], config["root_count"])
    layouts = []
    for remesh_id, mesh in remeshes.items():
        for root in roots[:2]:
            row, _, _ = generate_layout(reference, mesh, root, config)
            layouts.append({"surface_id": surface_id, "remesh_id": remesh_id, "layout_id": f"{root['root_id']}_{remesh_id}", **row})
    canonical_default = generate_nuc_skeleton(remeshes["M00"], policy="upstream_first")
    canonical_root = generate_nuc_skeleton(remeshes["M00"], policy="upstream_first", root_face=layouts[0]["mapped_face"])
    checks = {
        "canonical_root_maps_face_zero": layouts[0]["mapped_face"] == 0,
        "canonical_topology_exact": np.array_equal(canonical_default.topological_path, canonical_root.topological_path),
        "canonical_geometry_exact": np.array_equal(canonical_default.waypoints, canonical_root.waypoints),
        "policy_fixed": all(row["expansion_policy"] == "upstream_first" for row in layouts),
        "common_evaluation_surface": len({row["evaluation_surface_hash"] for row in layouts}) == 1,
        "root_mapping_within_tolerance": all(row["mapping_distance_m"] <= config["root_mapping_tolerance_m"] for row in layouts),
        "root_coordinates_remesh_independent": all(
            len({tuple(row["physical_root_position"]) for row in layouts if row["root_id"] == root["root_id"]}) == 1
            for root in roots[:2]
        ),
        "layout_count": len(layouts) == 4,
    }
    payload = {"passed": all(checks.values()), "checks": checks, "remeshes": records, "roots": roots[:2], "layouts": layouts}
    write_json(output / "stage_a_smoke.json", payload)
    write_csv(output / "stage_a_smoke_geometry.csv", layouts)
    print(json.dumps({"passed": payload["passed"], "checks": checks, "E_NUC": {r["layout_id"]: r["E_NUC"] for r in layouts}}, indent=2))
    if not payload["passed"]:
        raise RuntimeError("Stage A physical/evaluation contract failed")


def freeze(config, output):
    all_roots = {}; all_remeshes = {}; layouts = []; diversity = []
    aligned_cache = {}
    for surface_id in ("saddle", "hemisphere"):
        reference = make_reference_surface(surface_id, config)
        remeshes = {name: make_planning_remesh(surface_id, name, config) for name in config["remesh_ids"]}
        remesh_records = [{"surface_id": surface_id, "remesh_id": name, **remesh_statistics(surface_id, mesh, reference, config)} for name, mesh in remeshes.items()]
        validate_remesh_library(remesh_records, config)
        roots = generate_physical_roots(reference, remeshes["M00"], config["root_count"])
        all_roots[surface_id] = {"reference_surface_hash": surface_hash(reference), "roots": roots}
        all_remeshes[surface_id] = remesh_records
        surface_layouts = []
        for remesh_id, remesh in remeshes.items():
            for root in roots:
                row, _, _ = generate_layout(reference, remesh, root, config)
                row.update({"surface_id": surface_id, "remesh_id": remesh_id, "layout_id": f"{root['root_id']}_{remesh_id}"})
                surface_layouts.append(row)
                aligned_cache[(surface_id, row["layout_id"])] = aligned_ordered_path(reference, np.asarray(row["ordered_physical_path"]), config["diversity"]["aligned_samples"])
        baseline = select_geometry_baseline(surface_layouts)
        for row in surface_layouts:
            row["geometry_baseline"] = row["layout_id"] == baseline["layout_id"]
            row["canonical_layout"] = row["layout_id"] == "R00_M00"
        layouts.extend(surface_layouts)
        for i, first in enumerate(surface_layouts):
            for second in surface_layouts[i + 1:]:
                a = aligned_cache[(surface_id, first["layout_id"])]; b = aligned_cache[(surface_id, second["layout_id"])]
                ta = np.diff(a, axis=0); tb = np.diff(b, axis=0)
                ta /= np.maximum(np.linalg.norm(ta, axis=1, keepdims=True), 1e-15); tb /= np.maximum(np.linalg.norm(tb, axis=1, keepdims=True), 1e-15)
                same_remesh = first["remesh_id"] == second["remesh_id"]
                tree_jaccard = None
                if same_remesh:
                    ea = {tuple(edge[:2]) for edge in first["tree_edges"]}; eb = {tuple(edge[:2]) for edge in second["tree_edges"]}
                    tree_jaccard = len(ea & eb) / len(ea | eb)
                diversity.append({
                    "surface_id": surface_id, "layout_a": first["layout_id"], "layout_b": second["layout_id"],
                    "same_root": first["root_id"] == second["root_id"], "same_remesh": same_remesh,
                    "root_separation_m": geodesic_distance(reference, np.asarray(all_roots[surface_id]["roots"][int(first["root_id"][1:])]["position"]), np.asarray(all_roots[surface_id]["roots"][int(second["root_id"][1:])]["position"])),
                    "ordered_path_distance_m": float(np.mean(np.linalg.norm(a - b, axis=1))),
                    "tangent_disagreement_rad": float(np.mean(np.arccos(np.clip(np.sum(ta * tb, axis=1), -1, 1)))),
                    "tree_edge_jaccard": tree_jaccard,
                })
        plot_layouts(reference, surface_layouts, output / "figures" / f"layouts_{surface_id}.png")
    diversity_status = {}
    for surface_id in ("saddle", "hemisphere"):
        rows = [row for row in diversity if row["surface_id"] == surface_id]
        median_distance = float(np.median([row["ordered_path_distance_m"] for row in rows]))
        median_angle = float(np.median([row["tangent_disagreement_rad"] for row in rows]))
        failed = median_distance < config["diversity"]["failure_median_ordered_distance_m"] and np.rad2deg(median_angle) < config["diversity"]["failure_median_tangent_disagreement_degrees"]
        diversity_status[surface_id] = {"median_ordered_path_distance_m": median_distance, "median_tangent_disagreement_degrees": float(np.rad2deg(median_angle)), "candidate_family_diversity_failure": failed}
    if any(row["candidate_family_diversity_failure"] for row in diversity_status.values()):
        raise RuntimeError(f"candidate-family diversity failure: {diversity_status}")
    roots_payload = {"frozen_before_G0": True, "surfaces": all_roots}
    remesh_payload = {"frozen_before_G0": True, "surfaces": all_remeshes}
    library_payload = {"frozen_before_G0": True, "layouts": layouts, "diversity_status": diversity_status}
    for payload in (roots_payload, remesh_payload, library_payload):
        payload["content_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    write_json(output / "frozen_roots.json", roots_payload); write_json(output / "frozen_remeshes.json", remesh_payload)
    write_json(output / "layout_library.json", library_payload); write_json(output / "config.json", config)
    write_csv(output / "geometry_results.csv", [{k: v for k, v in row.items() if k not in {"topological_path", "tree_edges", "ordered_physical_path"}} for row in layouts])
    write_csv(output / "layout_diversity.csv", diversity)
    write_json(output / "freeze_summary.json", {"layout_count": len(layouts), "diversity": diversity_status, "root_hash": roots_payload["content_hash"], "remesh_hash": remesh_payload["content_hash"], "library_hash": library_payload["content_hash"]})
    print(json.dumps({"layouts": len(layouts), "diversity": diversity_status, "library_hash": library_payload["content_hash"]}, indent=2))


def plot_layouts(reference, rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 8, figsize=(20, 10), subplot_kw={"projection": "3d"})
    for axis, row in zip(axes.ravel(), sorted(rows, key=lambda item: (item["remesh_id"], item["root_id"]))):
        points = np.asarray(row["ordered_physical_path"])
        axis.plot(points[:, 0], points[:, 1], points[:, 2], lw=.45)
        axis.scatter(*points[0], s=8, c="red"); axis.set_title(row["layout_id"], fontsize=8); axis.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
