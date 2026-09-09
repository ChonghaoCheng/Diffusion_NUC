#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.diagnostics.cost_decomposition import classify_transition_commonality, decompose_transition_costs
from diffusion_coverage.diagnostics.e06_artifacts import canonical_task_poses, load_e06_contract, make_e06_surface, reconstructed_transition_poses, regenerate_e06_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="E06-D D2 witness cost decomposition")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nuc_robot_coupling_diagnosis_v1/d2_cost_decomposition")
    args = parser.parse_args()
    archived, rows, _ = load_e06_contract(ROOT)
    config, frozen, placements = archived["config"], archived["frozen_contract"], archived["placements"]
    args.output.mkdir(parents=True, exist_ok=True)
    transition_rows, candidate_rows, summary = [], [], {}
    for surface_number, surface_id in enumerate(config["surfaces"]):
        surface = make_e06_surface(config, surface_id, int(frozen["coverage_samples_per_face"]))
        variants = regenerate_e06_variants(surface, 20, int(config["seed"]) + 100000 * surface_number, int(config["e06"]["local_refinement_iterations"]))
        frequency, common, variable = classify_transition_commonality([item.topological_path for item in variants])
        surface_candidates = []
        for row in rows:
            if row["surface_id"] != surface_id or row["L_q"] is None:
                continue
            index = int(row["skeleton_id"][1:])
            skeleton = variants[index]
            transform = np.asarray(placements["surfaces"][surface_id]["selected"][row["placement_id"]]["transform_base_from_surface"], dtype=np.float64)
            canonical_positions, _ = canonical_task_poses(surface, variants[0], transform)
            edge_poses = reconstructed_transition_poses(
                surface, skeleton, canonical_positions, transform,
                float(frozen["coverage_path_sample_spacing_m"]), int(config["e06"]["task_edge_samples"]),
            )
            witness_path = ROOT / "results/nuc_robot_skeleton_coupling_v1/witnesses" / f"{surface_id}_{row['placement_id']}_{row['skeleton_id']}.npz"
            witness = np.load(witness_path)
            expected_positions = np.concatenate([edge_poses[0][0]] + [item[0][1:] for item in edge_poses[1:]])
            expected_axes = np.concatenate([edge_poses[0][1]] + [item[1][1:] for item in edge_poses[1:]])
            if not np.allclose(expected_positions, witness["desired_positions"], atol=1e-12) or not np.allclose(expected_axes, witness["desired_axes"], atol=1e-12):
                raise RuntimeError(f"{surface_id} {row['placement_id']} {row['skeleton_id']} transition replay mismatch")
            costs = decompose_transition_costs(
                skeleton.topological_path, witness["q"],
                np.asarray([len(item[0]) for item in edge_poses]), expected_total=float(row["L_q"]),
            )
            aggregates = {"within": 0.0, "cross": 0.0, "common": 0.0, "variable": 0.0}
            threshold_cost = {threshold: 0.0 for threshold in (0.50, 0.75, 0.90, 1.00)}
            for item in costs:
                edge = (item.source_code, item.target_code)
                within = item.source_code // 3 == item.target_code // 3
                is_common = edge in common
                aggregates["within" if within else "cross"] += item.joint_length
                aggregates["common" if is_common else "variable"] += item.joint_length
                for threshold in threshold_cost:
                    if frequency[edge] >= threshold:
                        threshold_cost[threshold] += item.joint_length
                transition_rows.append({
                    "surface_id": surface_id, "placement_id": row["placement_id"], "skeleton_id": row["skeleton_id"],
                    **item.__dict__, "within_parent": within, "library_common": is_common,
                    "transition_frequency": frequency[edge],
                })
            total = float(row["L_q"])
            record = {
                "surface_id": surface_id, "placement_id": row["placement_id"], "skeleton_id": row["skeleton_id"],
                "overall_pass": row["overall_pass"], "L_q_total": total,
                "L_q_within_parent": aggregates["within"], "L_q_cross_parent": aggregates["cross"],
                "L_q_common": aggregates["common"], "L_q_variable": aggregates["variable"],
                "eta_cross": aggregates["cross"] / total, "eta_var": aggregates["variable"] / total,
                "archive_abs_error": abs(sum(item.joint_length for item in costs) - total),
            }
            for threshold, value in threshold_cost.items():
                record[f"C_{threshold:.2f}"] = value / total
            candidate_rows.append(record); surface_candidates.append(record)
        summary[surface_id] = {
            "common_directed_transitions": len(common), "variable_directed_transitions": len(variable),
            "median_eta_var": float(np.median([row["eta_var"] for row in surface_candidates])),
            "median_eta_cross": float(np.median([row["eta_cross"] for row in surface_candidates])),
            "max_archive_abs_error": float(max(row["archive_abs_error"] for row in surface_candidates)),
            **{f"median_C_{threshold:.2f}": float(np.median([row[f"C_{threshold:.2f}"] for row in surface_candidates])) for threshold in (0.50, 0.75, 0.90, 1.00)},
            "scene_component_spreads": {
                placement: {
                    key: float((max(row[key] for row in surface_candidates if row["placement_id"] == placement) - min(row[key] for row in surface_candidates if row["placement_id"] == placement)) / min(row[key] for row in surface_candidates if row["placement_id"] == placement))
                    for key in ("L_q_total", "L_q_common", "L_q_variable", "L_q_cross_parent")
                }
                for placement in sorted({row["placement_id"] for row in surface_candidates})
            },
        }
        plot_surface_commonality(args.output, surface_id, surface, variants[0], canonical_task_poses(surface, variants[0], np.eye(4))[0], common, variable)
    write_rows(args.output / "transition_costs.jsonl", transition_rows)
    write_csv(args.output / "transition_costs.csv", transition_rows)
    write_rows(args.output / "candidate_decomposition.jsonl", candidate_rows)
    write_csv(args.output / "candidate_decomposition.csv", candidate_rows)
    make_plots(args.output, candidate_rows, transition_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def plot_surface_commonality(output, surface_id, surface, skeleton, points, common, variable):
    figure = plt.figure(figsize=(8, 6)); axis = figure.add_subplot(111, projection="3d")
    axis.plot_trisurf(*surface.vertices.T, triangles=surface.faces, color="#d9d9d9", alpha=0.15, linewidth=0.1)
    for edge in sorted(common | variable):
        values = points[list(edge)]
        axis.plot(*values.T, color="#222222" if edge in common else "#d62728", alpha=0.8 if edge in common else 0.25, linewidth=1.2)
    axis.set_title(f"{surface_id}: common (black) / variable (red) directed transitions")
    figure.tight_layout(); figure.savefig(output / f"{surface_id}_common_variable_transitions.png", dpi=180); plt.close(figure)


def make_plots(output, candidates, transitions):
    scenes = sorted({(row["surface_id"], row["placement_id"]) for row in candidates})
    figure, axes = plt.subplots(len(scenes), 1, figsize=(11, 2.6 * len(scenes)), squeeze=False)
    for axis, scene in zip(axes[:, 0], scenes):
        selected = sorted((r for r in candidates if (r["surface_id"], r["placement_id"]) == scene), key=lambda r: r["skeleton_id"])
        x = np.arange(len(selected)); common = np.asarray([r["L_q_common"] for r in selected]); variable = np.asarray([r["L_q_variable"] for r in selected])
        axis.bar(x, common, label="library-common"); axis.bar(x, variable, bottom=common, label="library-variable")
        axis.set_title(" / ".join(scene)); axis.set_ylabel("Witness L_q"); axis.set_xticks(x, [r["skeleton_id"] for r in selected], rotation=45); axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output / "stacked_Lq_decomposition.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 4)); data=[[r["eta_var"] for r in candidates if (r["surface_id"],r["placement_id"])==scene] for scene in scenes]
    axis.boxplot(data, labels=["\n".join(scene) for scene in scenes]); axis.set_ylabel("eta_var"); figure.tight_layout(); figure.savefig(output / "eta_var_by_scene.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    for surface in sorted({r["surface_id"] for r in candidates}):
        selected=[r for r in candidates if r["surface_id"]==surface]
        axis.plot([.5,.75,.9,1.0],[np.median([r[f"C_{v:.2f}"] for r in selected]) for v in (.5,.75,.9,1.0)],marker="o",label=surface)
    axis.set_xlabel("Transition frequency threshold"); axis.set_ylabel("Median accumulated L_q fraction"); axis.legend(); axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(output / "frequency_vs_Lq_contribution.png",dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    for scene in scenes:
        selected=[r for r in candidates if (r["surface_id"],r["placement_id"])==scene]
        spread=(max(r["L_q_total"] for r in selected)-min(r["L_q_total"] for r in selected))/min(r["L_q_total"] for r in selected)
        axis.scatter(np.median([r["eta_var"] for r in selected]),spread,label="-".join(scene))
    axis.set_xlabel("Median eta_var"); axis.set_ylabel("Actual L_q relative spread"); axis.legend(fontsize=7); axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(output / "Lq_spread_vs_eta_var.png",dpi=180); plt.close(figure)


def write_rows(path, rows): path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows))
def write_csv(path, rows):
    fields=sorted({k for row in rows for k in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
