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

from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract
from diffusion_coverage.diagnostics.execution_metric import local_task_increment, predicted_local_execution_length
from diffusion_coverage.diagnostics.statistics import pearson, relative_spread, spearman
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


def main() -> None:
    parser = argparse.ArgumentParser(description="E06-D D3 normalized 5D execution metric diagnosis")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nuc_robot_coupling_diagnosis_v1/d3_execution_metric")
    args = parser.parse_args()
    archived, rows, _ = load_e06_contract(ROOT)
    config, frozen = archived["config"], archived["frozen_contract"]
    robot_cfg = config["robot"]
    robot = UR5eKinematics(robot_cfg["model"], site_name=robot_cfg["site_name"], tool_axis_index=robot_cfg["tool_axis_index"], tool_axis_sign=robot_cfg["tool_axis_sign"])
    args.output.mkdir(parents=True, exist_ok=True)
    local_rows, candidate_rows = [], []
    for row in rows:
        if row["L_q"] is None:
            continue
        witness = np.load(ROOT / "results/nuc_robot_skeleton_coupling_v1/witnesses" / f"{row['surface_id']}_{row['placement_id']}_{row['skeleton_id']}.npz")
        q, positions, axes = witness["q"], witness["desired_positions"], witness["desired_axes"]
        predicted, actual = [], []
        for index in range(len(q) - 1):
            task = evaluate_task_kinematics_5d(robot, q[index], characteristic_length=float(robot_cfg["characteristic_length_m"]))
            if task.sigma_min_5 < float(frozen["sigma_safe"]) - 1e-12:
                raise RuntimeError("archived E06 witness violates frozen sigma_safe")
            delta = local_task_increment(positions[index], axes[index], positions[index + 1], axes[index + 1], task.axis_basis, characteristic_length=float(robot_cfg["characteristic_length_m"]))
            metric_length = predicted_local_execution_length(task.normalized_jacobian_5, delta, minimum_singular_value=float(frozen["sigma_safe"]))
            actual_length = float(np.linalg.norm(q[index + 1] - q[index]))
            candidate = robot.evaluate_configuration(q[index])
            axis_change = float(np.arccos(np.clip(np.dot(axes[index] / np.linalg.norm(axes[index]), axes[index + 1] / np.linalg.norm(axes[index + 1])), -1.0, 1.0)))
            predicted.append(metric_length); actual.append(actual_length)
            local_rows.append({
                "surface_id": row["surface_id"], "placement_id": row["placement_id"], "skeleton_id": row["skeleton_id"], "interval_index": index,
                "delta_L_G": metric_length, "delta_L_q_actual": actual_length,
                "residual": actual_length - metric_length, "sigma_min_5": task.sigma_min_5,
                "joint_limit_margin": candidate.joint_limit_margin, "normal_change_rad": axis_change,
            })
        record = {
            "surface_id": row["surface_id"], "placement_id": row["placement_id"], "skeleton_id": row["skeleton_id"],
            "overall_pass": row["overall_pass"], "E_NUC": row["E_NUC"], "L_q": float(row["L_q"]), "L_G": float(np.sum(predicted)),
            "L_q_over_L_G": float(np.sum(actual) / np.sum(predicted)),
            "local_pearson": pearson(np.asarray(predicted), np.asarray(actual)),
            "local_spearman": spearman(np.asarray(predicted), np.asarray(actual)),
        }
        if not np.isclose(record["L_q"], np.sum(actual), atol=1e-9, rtol=1e-9):
            raise RuntimeError("D3 local actual increments do not reproduce archived L_q")
        candidate_rows.append(record)
        print(f"{row['surface_id']} {row['placement_id']} {row['skeleton_id']} LG={record['L_G']:.5f} Lq={record['L_q']:.5f}", flush=True)
    summaries = {}
    for scene in sorted({(row["surface_id"], row["placement_id"]) for row in candidate_rows}):
        selected = [row for row in candidate_rows if (row["surface_id"], row["placement_id"]) == scene and row["overall_pass"]]
        lg=np.asarray([row["L_G"] for row in selected]); lq=np.asarray([row["L_q"] for row in selected])
        local=[row for row in local_rows if (row["surface_id"],row["placement_id"])==scene]
        summaries["/".join(scene)] = {
            "candidate_count": len(selected), "L_G_relative_spread": relative_spread(lg), "L_q_relative_spread": relative_spread(lq),
            "candidate_pearson": pearson(lg,lq), "candidate_spearman": spearman(lg,lq),
            "per_scene_normalized_pearson": pearson(lg/np.mean(lg),lq/np.mean(lq)),
            "local_pearson": pearson(np.asarray([r["delta_L_G"] for r in local]),np.asarray([r["delta_L_q_actual"] for r in local])),
            "local_spearman": spearman(np.asarray([r["delta_L_G"] for r in local]),np.asarray([r["delta_L_q_actual"] for r in local])),
            "residual_sigma_pearson": pearson(np.asarray([r["residual"] for r in local]),np.asarray([r["sigma_min_5"] for r in local])),
            "residual_margin_pearson": pearson(np.asarray([r["residual"] for r in local]),np.asarray([r["joint_limit_margin"] for r in local])),
            "median_L_q_over_L_G": float(np.median([row["L_q_over_L_G"] for row in selected])),
        }
    write_rows(args.output / "candidate_metric.jsonl", candidate_rows); write_csv(args.output / "candidate_metric.csv", candidate_rows)
    write_rows(args.output / "local_metric.jsonl", local_rows); write_csv(args.output / "local_metric.csv", local_rows)
    (args.output / "summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    make_plots(args.output,candidate_rows,local_rows)
    print(json.dumps(summaries,indent=2))


def make_plots(output,candidates,local):
    scenes=sorted({(r["surface_id"],r["placement_id"]) for r in candidates})
    figure,axes=plt.subplots(2,2,figsize=(11,9))
    for scene in scenes:
        selected=[r for r in candidates if (r["surface_id"],r["placement_id"])==scene and r["overall_pass"]]
        if selected: axes[0,0].scatter([r["L_G"] for r in selected],[r["L_q"] for r in selected],label="-".join(scene))
    axes[0,0].set(xlabel="L_G",ylabel="L_q",title="Candidate-level prediction"); axes[0,0].legend(fontsize=7)
    stride=max(1,len(local)//30000); sample=local[::stride]
    axes[0,1].scatter([r["delta_L_G"] for r in sample],[r["delta_L_q_actual"] for r in sample],s=2,alpha=.2)
    axes[0,1].set(xlabel="delta L_G",ylabel="delta L_q actual",title="Local intervals")
    axes[1,0].scatter([r["sigma_min_5"] for r in sample],[r["residual"] for r in sample],s=2,alpha=.2); axes[1,0].set(xlabel="sigma_min_5",ylabel="Lq-LG residual")
    axes[1,1].scatter([r["joint_limit_margin"] for r in sample],[r["residual"] for r in sample],s=2,alpha=.2); axes[1,1].set(xlabel="Joint-limit margin",ylabel="Lq-LG residual")
    for axis in axes.ravel(): axis.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(output/"execution_metric_diagnostics.png",dpi=180); plt.close(figure)


def write_rows(path,rows): path.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
def write_csv(path,rows):
    fields=sorted({k for row in rows for k in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
