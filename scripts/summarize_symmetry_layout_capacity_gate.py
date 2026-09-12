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


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize E06-G2 symmetry capacity")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/symmetry_preserving_global_layout_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/symmetry_preserving_global_layout_v1")
    return parser.parse_args()


def main():
    args = parse_args(); config = json.loads(args.config.read_text()); rows = read_csv(args.output / "execution_results.csv")
    for row in rows: normalize(row)
    strong = read_csv(args.output / "liftability_sensitivity.csv") if (args.output / "liftability_sensitivity.csv").exists() else []
    for row in strong: normalize(row)
    scene_rows = summarize_scenes(rows, strong, config)
    mixed = [row for row in scene_rows if row["default_mixed_success"]]
    strong_pending = bool(mixed and not strong)
    hemisphere_spread_hits = sum(
        row["surface_id"] == "hemisphere" and row["S_sym"] is not None
        and row["S_sym"] >= config["gate"]["minimum_scene_spread"]
        for row in scene_rows
    )
    hemisphere_gains = [row["Delta_sym"] for row in scene_rows if row["surface_id"] == "hemisphere" and row["Delta_sym"] is not None]
    median_gain = None if not hemisphere_gains else float(np.median(hemisphere_gains))
    cost_go = (
        hemisphere_spread_hits >= config["gate"]["minimum_hemisphere_scenes_with_10pct_spread"]
        and median_gain is not None and median_gain >= config["gate"]["minimum_median_canonical_gain"]
    )
    robust_mixed = sum(row["robust_mixed_success"] is True for row in scene_rows)
    feasibility_go = not cost_go and robust_mixed >= config["gate"]["minimum_robust_mixed_success_scenes"]
    decision = "PENDING_STRONG_SENSITIVITY" if strong_pending else ("GO" if cost_go or feasibility_go else "NO-GO")
    summary = {
        "experiment": config["experiment"], "orbit_executions": len(rows),
        "verified_default": sum(row["overall_pass"] for row in rows),
        "failed_default": sum(not row["overall_pass"] for row in rows),
        "hemisphere_scenes_with_S_sym_ge_10pct": hemisphere_spread_hits,
        "median_hemisphere_Delta_sym": median_gain, "cost_GO": cost_go,
        "default_mixed_success_scenes": len(mixed), "strong_cases": len(strong),
        "strong_recoveries": sum(row["overall_pass"] for row in strong),
        "robust_mixed_success_scenes": robust_mixed, "feasibility_GO": feasibility_go,
        "strong_sensitivity_pending": strong_pending, "decision": decision,
        "riemannian_authorized": decision == "GO",
        "provisional_outcome": "D" if feasibility_go else ("B_or_C_pending_Riemannian" if cost_go else (None if strong_pending else "A")),
    }
    write_csv(args.output / "scene_summary.csv", scene_rows)
    write_json(args.output / "capacity_summary.json", summary)
    (args.output / "capacity_summary.md").write_text(render_markdown(summary, scene_rows))
    make_figures(args.output, rows, scene_rows)
    (args.output / "reproduction_commands.txt").write_text("\n".join((
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/freeze_symmetry_layout_library.py",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_symmetry_layout_capacity_gate.py --stage default --jobs 12",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/summarize_symmetry_layout_capacity_gate.py",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_symmetry_layout_capacity_gate.py --stage strong --jobs 12  # only for mixed scenes",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/summarize_symmetry_layout_capacity_gate.py",
        "/data/chocheng/.venvs/coverage-fm/bin/python -m pytest -q",
    )) + "\n")
    print(json.dumps(summary, indent=2))


def summarize_scenes(rows, strong, config):
    results = []
    for surface in ("saddle", "hemisphere"):
        for level in config["placement_levels"]:
            scene = [row for row in rows if row["surface_id"] == surface and row["placement_level"] == level]
            verified = [row for row in scene if row["overall_pass"] and row["J_q"] is not None]
            q = np.asarray([row["J_q"] for row in verified], dtype=float)
            canonical = next(row for row in scene if row["symmetry_id"] in {"S00", "H00"})
            mixed = bool(verified and len(verified) < len(scene))
            strong_scene = [row for row in strong if row["surface_id"] == surface and row["placement_level"] == level]
            final_success = {row["symmetry_id"]: row["overall_pass"] for row in scene}
            for row in strong_scene: final_success[row["symmetry_id"]] = row["overall_pass"]
            robust = None if mixed and not strong_scene else bool(any(final_success.values()) and not all(final_success.values()))
            best = None if not verified else min(verified, key=lambda row: (row["J_q"], row["symmetry_id"]))
            results.append({
                "surface_id": surface, "placement_level": level, "placement_id": scene[0]["placement_id"],
                "orbit_size": len(scene), "verified_count": len(verified), "failed_count": len(scene)-len(verified),
                "J_min": None if not len(q) else float(q.min()), "J_max": None if not len(q) else float(q.max()),
                "J_median": None if not len(q) else float(np.median(q)),
                "S_sym": None if len(q) < 2 else float((q.max()-q.min())/q.min()),
                "canonical_verified": canonical["overall_pass"], "canonical_J_q": canonical["J_q"],
                "best_symmetry_id": None if best is None else best["symmetry_id"],
                "best_angle_degrees": None if best is None else best["angle_degrees"],
                "Delta_sym": None if not canonical["overall_pass"] or best is None else (canonical["J_q"]-best["J_q"])/canonical["J_q"],
                "default_mixed_success": mixed, "strong_failed_cases": len(strong_scene),
                "strong_recoveries": sum(row["overall_pass"] for row in strong_scene),
                "robust_mixed_success": robust,
                "median_first_fraction": None if not verified else float(np.median([row["J_q_first_5pct"]/row["J_q"] for row in verified])),
                "median_middle_fraction": None if not verified else float(np.median([row["J_q_middle_90pct"]/row["J_q"] for row in verified])),
                "median_last_fraction": None if not verified else float(np.median([row["J_q_last_5pct"]/row["J_q"] for row in verified])),
            })
    return results


def make_figures(output, rows, scenes):
    figures = output / "figures"; sources = output / "figure_sources"; figures.mkdir(exist_ok=True); sources.mkdir(exist_ok=True)
    write_csv(sources / "execution_plot_source.csv", rows); write_csv(sources / "scene_plot_source.csv", scenes)
    for level in ("P_low", "P_mid", "P_high"):
        part = [row for row in rows if row["surface_id"] == "hemisphere" and row["placement_level"] == level]
        fig, ax = plt.subplots(figsize=(8, 4)); passed = [row for row in part if row["overall_pass"]]
        ax.plot([row["angle_degrees"] for row in passed], [row["J_q"] for row in passed], "o-")
        ax.set(xlabel="Object-frame rotation (deg)", ylabel="Verified J_q", title=f"Hemisphere {level}"); ax.grid(alpha=.25)
        save(fig, figures / f"hemisphere_Jq_vs_theta_{level}.png")
        fig, ax = plt.subplots(figsize=(8, 2.8)); ax.scatter([row["angle_degrees"] for row in part], [int(row["overall_pass"]) for row in part])
        ax.set(xlabel="Object-frame rotation (deg)", ylabel="Strict lift", yticks=(0,1), title=f"Hemisphere {level}")
        save(fig, figures / f"hemisphere_liftability_vs_theta_{level}.png")
    labels=[f"{row['surface_id']}\n{row['placement_level']}" for row in scenes]
    fig,ax=plt.subplots(figsize=(9,4)); ax.bar(range(len(scenes)),[np.nan if row["S_sym"] is None else row["S_sym"] for row in scenes]);ax.axhline(.1,color="red",ls="--");ax.set_xticks(range(len(labels)),labels);ax.set_ylabel("S_sym");save(fig,figures/"symmetry_Jq_spread.png")
    verified=[row for row in rows if row["overall_pass"]]
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(verified));ax.stackplot(x,[row["J_q_first_5pct"] for row in verified],[row["J_q_middle_90pct"] for row in verified],[row["J_q_last_5pct"] for row in verified],labels=("first 5%","middle 90%","last 5%"));ax.set(xlabel="Verified orbit member",ylabel="J_q contribution");ax.legend();save(fig,figures/"endpoint_vs_interior_cost.png")


def render_markdown(summary, scenes):
    lines=["# E06-G2 capacity summary","",f"Decision: **{summary['decision']}**","",f"Default strict witnesses: **{summary['verified_default']}/{summary['orbit_executions']}**.","","| Scene | Verified | S_sym | Delta_sym | Best | Default mixed | Strong recoveries |","|---|---:|---:|---:|---|---|---:|"]
    fmt=lambda value:"n/a" if value is None else f"{value:.3%}"
    for row in scenes: lines.append(f"| {row['surface_id']} {row['placement_level']} | {row['verified_count']}/{row['orbit_size']} | {fmt(row['S_sym'])} | {fmt(row['Delta_sym'])} | {row['best_symmetry_id'] or 'n/a'} | {row['default_mixed_success']} | {row['strong_recoveries']} |")
    lines.extend(["",f"Cost GO: `{summary['cost_GO']}`; feasibility GO: `{summary['feasibility_GO']}`; Riemannian diagnostic authorized: `{summary['riemannian_authorized']}`.","","This is a finite exact-symmetry orbit under finite numerical continuation budgets, not a planner or global optimum.",""])
    return "\n".join(lines)


def normalize(row):
    for key in ("E_miss","E_rep","E_NUC","L_S","J_q","C_q","angle_degrees","J_q_first_5pct","J_q_middle_90pct","J_q_last_5pct","min_sigma_min_5","minimum_absolute_joint_margin_rad"):
        row[key] = None if row.get(key) in (None,"","None") else float(row[key])
    for key in ("lift_found","strict_kinematics_pass","overall_pass","collision_pass","neighbor_warm_start"):
        row[key] = as_bool(row.get(key))
def as_bool(value): return value is True or str(value).lower() == "true"
def read_csv(path):
    with path.open(newline="") as handle: return list(csv.DictReader(handle))
def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def write_csv(path,rows):
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def save(fig,path): fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


if __name__ == "__main__": main()
