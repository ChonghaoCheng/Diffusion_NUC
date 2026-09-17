#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.robot.e09_execution import sphere_episode_counts_indexed
from e11_runner_support import quadrature


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"results/e11_mechanism_placement_transfer_v1");args=parser.parse_args()
    output = args.output.resolve()
    config = json.loads((ROOT / "configs/e11_mechanism_placement_transfer_v1.json").read_text())
    scene_docs = json.loads((ROOT / config["inputs"]["transfer_scenes"]).read_text())["scenes"]
    old = json.loads((ROOT / config["inputs"]["placements"]).read_text())["selected"]
    scenes = {x["scene_id"]: x for x in scene_docs}
    for surface in old.values():
        for value in surface.values():
            scenes[value["candidate_id"]] = value
    selected = rows(output / "final_validation.csv")
    chosen = {}
    for row in selected:
        if row.get("selected_plan_file"):
            chosen.setdefault(row["witness_hash"], row)
    robot = UR5eKinematics(config["inputs"]["robot_model"], site_name=config["robot"]["site_name"], tool_axis_index=int(config["robot"]["tool_axis_index"]), tool_axis_sign=float(config["robot"]["tool_axis_sign"]))
    qpoints, _ = quadrature(config, "Q2", float(config["surface"]["radius_m"]), ROOT)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    sources = output / "plot_source_data"; sources.mkdir(exist_ok=True)
    index = []
    for witness_hash, row in sorted(chosen.items()):
        plan = np.load(ROOT / row["selected_plan_file"], allow_pickle=False)
        transform = np.asarray(scenes[row["scene_id"]]["transform_base_from_surface"], float)
        rotation, translation = transform[:3, :3], transform[:3, 3]
        surface = np.empty((len(plan["q"]), 3)); sigma = np.empty(len(plan["q"]))
        for i, q in enumerate(plan["q"]):
            task = evaluate_task_kinematics_5d(robot, q, characteristic_length=float(config["robot"]["characteristic_length_m"]))
            raw = (task.position - translation) @ rotation
            surface[i] = float(config["surface"]["radius_m"]) * raw / np.linalg.norm(raw)
            sigma[i] = task.sigma_min_5
        counts = sphere_episode_counts_indexed(qpoints, surface, plan["activity"], radius=float(config["surface"]["radius_m"]), footprint_radius=float(config["coverage"]["footprint_radius_m"]))
        np.savez_compressed(sources / f"{witness_hash[:12]}.npz", achieved_surface=surface, activity=plan["activity"], sigma5=sigma, quadrature_points=qpoints, episode_counts=counts)
        unit = qpoints / np.linalg.norm(qpoints, axis=1, keepdims=True); az = np.arctan2(unit[:, 1], unit[:, 0]); polar = np.arccos(np.clip(unit[:, 2], -1, 1))
        fig = plt.figure(figsize=(12, 4)); ax = fig.add_subplot(131, projection="3d"); active = plan["activity"].astype(bool); ax.plot(surface[active,0], surface[active,1], surface[active,2], lw=.35); ax.scatter(surface[~active,0],surface[~active,1],surface[~active,2],s=2,c="orange"); ax.set_title(f"{row['scene_id']} actual FK")
        ax2=fig.add_subplot(132); sc=ax2.scatter(az,polar,c=np.minimum(counts,2),s=.35,cmap="viridis",vmin=0,vmax=2,rasterized=True); ax2.set(xlabel="azimuth",ylabel="polar",title="Q2 episode count");fig.colorbar(sc,ax=ax2)
        ax3=fig.add_subplot(133);ax3.plot(sigma,lw=.4);ax3.axhline(float(config["robot"]["sigma_safe"]),c="r",ls="--");ax3.set(title="sampled sigma5",xlabel="stored sample")
        fig.tight_layout(); figure=figures/f"{witness_hash[:12]}_whole_surface.png";fig.savefig(figure,dpi=150);plt.close(fig)
        index.append({"witness_hash":witness_hash,"scene_id":row["scene_id"],"source":str((sources/f"{witness_hash[:12]}.npz").relative_to(ROOT)),"figure":str(figure.relative_to(ROOT)),"note":"actual FK at stored graph-witness samples; verdict uses denser validation tables"})
    (output/"figure_index.json").write_text(json.dumps(index,indent=2)+"\n")


if __name__ == "__main__":
    main()
