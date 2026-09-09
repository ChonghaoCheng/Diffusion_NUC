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

from diffusion_coverage.diagnostics.e06_artifacts import dense_geodesic_path, load_e06_contract, make_e06_surface, regenerate_e06_variants
from diffusion_coverage.diagnostics.structure import compare_skeletons, transition_parent_types


def main() -> None:
    parser = argparse.ArgumentParser(description="E06-D D1 skeleton structure diagnosis")
    parser.add_argument("--e06-root", type=Path, default=ROOT / "results/nuc_robot_skeleton_coupling_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nuc_robot_coupling_diagnosis_v1/d1_structure")
    args = parser.parse_args()
    archived, rows, _ = load_e06_contract(ROOT)
    config, frozen = archived["config"], archived["frozen_contract"]
    args.output.mkdir(parents=True, exist_ok=True)
    pair_rows, candidate_rows, summaries = [], [], {}
    for surface_number, surface_id in enumerate(config["surfaces"]):
        surface = make_e06_surface(config, surface_id, int(frozen["coverage_samples_per_face"]))
        variants = regenerate_e06_variants(
            surface, int(config["e06"]["num_skeleton_candidates"]),
            int(config["seed"]) + 100000 * surface_number,
            int(config["e06"]["local_refinement_iterations"]),
        )
        reference_rows = sorted(
            (row for row in rows if row["surface_id"] == surface_id and row["placement_id"] == "P_easy"),
            key=lambda row: row["skeleton_id"],
        )
        for index, (skeleton, reference) in enumerate(zip(variants, reference_rows)):
            if skeleton.policy != reference["skeleton_policy"] or skeleton.seed != reference["seed"]:
                raise RuntimeError(f"regenerated {surface_id} S{index:02d} identity differs from E06")
            within, cross = transition_parent_types(skeleton.topological_path)
            candidate_rows.append({
                "surface_id": surface_id, "skeleton_id": f"S{index:02d}",
                "policy": skeleton.policy, "seed": skeleton.seed,
                "num_waypoints": len(skeleton.topological_path),
                "num_directed_transitions": len(skeleton.topological_path) - 1,
                "within_parent_fraction": float(np.mean(within)),
                "cross_parent_fraction": float(np.mean(cross)),
                "expansion_tree_edges": len(skeleton.tree_edges),
                "expansion_order": [decision.parent_face for decision in skeleton.expansion_decisions],
            })
        dense = [dense_geodesic_path(surface, item, float(frozen["coverage_path_sample_spacing_m"])) for item in variants]
        matrices = {
            "directed_transition_jaccard": np.eye(len(variants)),
            "geometric_path_distance": np.zeros((len(variants), len(variants))),
        }
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                metric = compare_skeletons(variants[i], variants[j], first_path=dense[i], second_path=dense[j])
                record = {"surface_id": surface_id, "skeleton_a": f"S{i:02d}", "skeleton_b": f"S{j:02d}", **metric.__dict__}
                pair_rows.append(record)
                for name in matrices:
                    matrices[name][i, j] = matrices[name][j, i] = getattr(metric, name)
        selected = [row for row in pair_rows if row["surface_id"] == surface_id]
        summaries[surface_id] = {
            name: {"mean": float(np.mean([row[name] for row in selected])), "median": float(np.median([row[name] for row in selected])), "min": float(np.min([row[name] for row in selected])), "max": float(np.max([row[name] for row in selected]))}
            for name in ("directed_transition_jaccard", "undirected_transition_jaccard", "tree_edge_jaccard", "normalized_sequence_distance", "geometric_path_distance", "tangent_disagreement")
        }
        for name, matrix in matrices.items():
            figure, axis = plt.subplots(figsize=(7, 6))
            image = axis.imshow(matrix, cmap="viridis", aspect="equal")
            axis.set_title(f"{surface_id}: {name.replace('_', ' ')}")
            axis.set_xlabel("Skeleton ID"); axis.set_ylabel("Skeleton ID")
            axis.set_xticks(range(0, 20, 2), [f"S{i:02d}" for i in range(0, 20, 2)], rotation=45)
            axis.set_yticks(range(0, 20, 2), [f"S{i:02d}" for i in range(0, 20, 2)])
            figure.colorbar(image, ax=axis); figure.tight_layout()
            figure.savefig(args.output / f"{surface_id}_{name}_heatmap.png", dpi=180); plt.close(figure)
    write_rows(args.output / "pairwise_structure.jsonl", pair_rows)
    write_csv(args.output / "pairwise_structure.csv", pair_rows)
    write_rows(args.output / "candidate_structure.jsonl", candidate_rows)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
