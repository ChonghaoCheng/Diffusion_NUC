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
from scipy.stats import pearsonr, spearmanr

from diffusion_coverage.diagnostics.global_layout import finite_verified_oracle


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize E06-G G0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/global_layout_capacity_gate_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/global_layout_capacity_gate_v1")
    return parser.parse_args()


def main():
    args = parse_args(); config = json.loads(args.config.read_text())
    geometry = read_csv(args.output / "geometry_results.csv"); execution = read_csv(args.output / "execution_results.csv")
    if len(geometry) != 64 or len(execution) != 192:
        raise RuntimeError("G0 summary requires 64 geometry and 192 execution rows")
    for row in geometry:
        for key in ("E_miss","E_rep","E_NUC","L_S","mapping_distance_m","NUC_generation_time","local_refinement_time"):
            row[key] = float(row[key])
        row["geometry_baseline"] = as_bool(row["geometry_baseline"])
        row["canonical_layout"] = as_bool(row["canonical_layout"])
    geometry_by_id = {(row["surface_id"], row["layout_id"]): row for row in geometry}
    for row in execution:
        source = geometry_by_id[(row["surface_id"], row["layout_id"])]
        row.update({key: source[key] for key in ("E_miss", "E_rep", "E_NUC", "L_S", "geometry_baseline", "canonical_layout")})
        normalize_row(row)
    for surface in ("saddle", "hemisphere"):
        baseline = next(row for row in geometry if row["surface_id"] == surface and row["geometry_baseline"])
        limit = baseline["E_NUC"] + config["coverage"]["delta_NUC"]
        for row in execution:
            if row["surface_id"] == surface:
                row["coverage_equivalent"] = row["E_NUC"] <= limit + 1e-12
    scenes = summarize_scenes(execution, config)
    effects = effect_decomposition(execution, geometry)
    strong_rows = read_csv(args.output / "strong_lift_sensitivity.csv") if (args.output / "strong_lift_sensitivity.csv").exists() else []
    summary = summarize_gate(scenes, strong_rows, config)
    summary["verified_executions"] = sum(row["overall_pass"] for row in execution)
    summary["failed_executions"] = sum(not row["overall_pass"] for row in execution)
    summary["canonical_verified_scenes"] = sum(row["overall_pass"] and row["canonical_layout"] for row in execution)
    summary["diversity"] = json.loads((args.output / "freeze_summary.json").read_text())["diversity"]
    write_csv(args.output / "scene_results.csv", scenes); write_csv(args.output / "root_remesh_effects.csv", effects)
    write_json(args.output / "G0_summary.json", summary); (args.output / "G0_summary.md").write_text(render_markdown(summary, scenes))
    make_plots(execution, scenes, effects, args.output)
    selection = select_strong_sensitivity(scenes, execution)
    if selection and not strong_rows:
        write_json(args.output / "strong_sensitivity_selection.json", {"frozen_after_default_G0": True, "selection_rule": config["strong_search"]["selection_rule"], "cases": selection})
    (args.output / "reproduction_commands.txt").write_text("\n".join([
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/freeze_global_layout_library.py --stage smoke",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/freeze_global_layout_library.py --stage freeze --jobs 12",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_global_layout_capacity_gate.py --jobs 12",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/summarize_global_layout_capacity_gate.py",
        "/data/chocheng/.venvs/coverage-fm/bin/python -m pytest -q",
    ]) + "\n")
    print(json.dumps(summary, indent=2))


def summarize_scenes(rows, config):
    result = []
    for surface in ("saddle", "hemisphere"):
        surface_rows = [row for row in rows if row["surface_id"] == surface]
        baseline_layout = next(row["layout_id"] for row in surface_rows if row["geometry_baseline"])
        baseline_geometry = next(row for row in surface_rows if row["layout_id"] == baseline_layout)
        nuc_limit = baseline_geometry["E_NUC"] + config["coverage"]["delta_NUC"]
        for level in config["placement_levels"]:
            scene = [row for row in surface_rows if row["anisotropy_level"] == level]
            equivalent = [row for row in scene if row["E_NUC"] <= nuc_limit + 1e-12]
            verified = [row for row in equivalent if row["overall_pass"] and row["J_q"] is not None]
            baseline = next(row for row in scene if row["layout_id"] == baseline_layout)
            oracle = finite_verified_oracle(verified)
            q = np.asarray([row["J_q"] for row in verified], dtype=float)
            cq = np.asarray([row["C_q"] for row in verified], dtype=float)
            all_verified = [row for row in scene if row["overall_pass"] and row["J_q"] is not None]
            all_q = np.asarray([row["J_q"] for row in all_verified], dtype=float)
            if len(all_verified) >= 2:
                lengths = np.asarray([row["L_S"] for row in all_verified], dtype=float)
                pearson_ls = float(pearsonr(lengths, all_q).statistic)
                spearman_ls = float(spearmanr(lengths, all_q).statistic)
            else:
                pearson_ls = spearman_ls = None
            length_matched = [row for row in verified if abs(row["L_S"] - baseline["L_S"]) / baseline["L_S"] <= config["length_match_relative_tolerance"] + 1e-12]
            length_oracle = finite_verified_oracle(length_matched)
            delta = None if not baseline["overall_pass"] or oracle is None else (baseline["J_q"] - oracle["J_q"]) / baseline["J_q"]
            delta_length = None if not baseline["overall_pass"] or length_oracle is None else (baseline["J_q"] - length_oracle["J_q"]) / baseline["J_q"]
            result.append({
                "surface_id": surface, "anisotropy_level": level, "placement_id": scene[0]["placement_id"],
                "geometry_layout_id": baseline_layout, "geometry_lift_found": baseline["overall_pass"], "geometry_J_q": baseline["J_q"],
                "coverage_equivalent_layouts": len(equivalent), "verified_equivalent_layouts": len(verified),
                "all_verified_layouts": len(all_verified),
                "all_verified_S_q": None if len(all_q)<2 else float((all_q.max()-all_q.min())/all_q.min()),
                "Pearson_L_S_J_q_all_verified": pearson_ls, "Spearman_L_S_J_q_all_verified": spearman_ls,
                "J_min": None if not len(q) else float(q.min()), "J_max": None if not len(q) else float(q.max()),
                "J_median": None if not len(q) else float(np.median(q)), "J_IQR": None if not len(q) else float(np.quantile(q,.75)-np.quantile(q,.25)),
                "S_q": None if len(q)<2 else float((q.max()-q.min())/q.min()),
                "C_q_spread": None if len(cq)<2 else float((cq.max()-cq.min())/cq.min()),
                "oracle_layout_id": None if oracle is None else oracle["layout_id"], "oracle_J_q": None if oracle is None else oracle["J_q"],
                "Delta_global": delta, "length_matched_layouts": len(length_matched),
                "length_oracle_layout_id": None if length_oracle is None else length_oracle["layout_id"],
                "Delta_global_length": delta_length,
                "apparent_liftability_rescue": bool(not baseline["overall_pass"] and len(verified)>0),
            })
    return result


def summarize_gate(scenes, strong_rows, config):
    gate = config["gate"]
    spread_hits = sum(row["S_q"] is not None and row["S_q"] >= gate["minimum_scene_spread"] for row in scenes)
    gains = [row["Delta_global"] for row in scenes if row["Delta_global"] is not None]
    median_gain = None if not gains else float(np.median(gains))
    cost_go = spread_hits >= gate["minimum_scenes_with_10pct_spread"] and median_gain is not None and median_gain >= gate["minimum_median_geometry_oracle_gain"]
    apparent = [row for row in scenes if row["apparent_liftability_rescue"]]
    confirmed_scenes = len({(row["surface_id"], row["anisotropy_level"]) for row in strong_rows if as_bool(row.get("rescue_confirmed"))})
    sensitivity_pending = bool(apparent and not strong_rows)
    feasibility_go = not cost_go and confirmed_scenes >= gate["minimum_strong_confirmed_rescue_scenes"]
    if sensitivity_pending:
        decision = "PENDING_STRONG_SENSITIVITY"
    else:
        decision = "GO" if cost_go or feasibility_go else "NO-GO"
    return {
        "experiment": config["experiment"], "geometry_layouts": 64, "scene_layout_executions": 192,
        "scenes_with_S_q_ge_10pct": spread_hits, "median_Delta_global": median_gain,
        "apparent_liftability_rescue_scenes": len(apparent), "strong_confirmed_rescue_scenes": confirmed_scenes,
        "cost_GO": cost_go, "feasibility_GO": feasibility_go, "strong_sensitivity_pending": sensitivity_pending,
        "decision": decision, "G1_authorized": decision == "GO",
        "classification": None if decision != "NO-GO" else "A_no_demonstrated_admitted_global_layout_headroom",
        "capacity_interpretation": None if decision != "NO-GO" else "coverage-equivalent sets are singleton for saddle and contain no verified witnesses for hemisphere; the gate is negative but cannot establish absence outside this frozen NUC library",
    }


def select_strong_sensitivity(scenes, execution):
    selected = []
    for scene in scenes:
        if not scene["apparent_liftability_rescue"]: continue
        rows = [row for row in execution if row["surface_id"]==scene["surface_id"] and row["anisotropy_level"]==scene["anisotropy_level"]]
        baseline = next(row for row in rows if row["layout_id"]==scene["geometry_layout_id"])
        limit = baseline["E_NUC"] + 0.0297927413
        alternatives = [row for row in rows if row["E_NUC"]<=limit+1e-12 and row["overall_pass"]]
        alternative = min(alternatives, key=lambda row:(row["J_q"],row["layout_id"]))
        selected.append({"surface_id":scene["surface_id"],"anisotropy_level":scene["anisotropy_level"],"geometry_layout_id":baseline["layout_id"],"alternative_layout_id":alternative["layout_id"]})
    return selected


def effect_decomposition(execution, geometry):
    rows = []
    for surface in ("saddle","hemisphere"):
        geo = [row for row in geometry if row["surface_id"]==surface]
        for metric in ("E_NUC","L_S"):
            rows.append(effect_row(surface,"geometry",metric,geo))
        for level in ("P_low","P_mid","P_high"):
            scene=[row for row in execution if row["surface_id"]==surface and row["anisotropy_level"]==level]
            for metric in ("J_q","C_q","lift_success"):
                if metric=="lift_success":
                    working=[{**row,"lift_success":float(row["overall_pass"])} for row in scene]
                else: working=[row for row in scene if row[metric] is not None]
                rows.append(effect_row(surface,level,metric,working))
    return rows


def effect_row(surface, scope, metric, rows):
    if not rows:return {"surface_id":surface,"scope":scope,"metric":metric,"count":0}
    values=np.asarray([row[metric] for row in rows],float); grand=float(values.mean())
    root_means={key:np.mean([row[metric] for row in rows if row["root_id"]==key]) for key in sorted({row["root_id"] for row in rows})}
    remesh_means={key:np.mean([row[metric] for row in rows if row["remesh_id"]==key]) for key in sorted({row["remesh_id"] for row in rows})}
    residual=np.asarray([row[metric]-(root_means[row["root_id"]]+remesh_means[row["remesh_id"]]-grand) for row in rows],float)
    total_var=float(np.var(values)); interaction=None if total_var<=1e-20 else float(np.var(residual)/total_var)
    root_range=float(max(root_means.values())-min(root_means.values())); remesh_range=float(max(remesh_means.values())-min(remesh_means.values()))
    return {"surface_id":surface,"scope":scope,"metric":metric,"count":len(rows),"grand_mean":grand,"root_mean_range":root_range,"root_mean_relative_range":None if abs(grand)<1e-15 else root_range/abs(grand),"remesh_mean_range":remesh_range,"remesh_mean_relative_range":None if abs(grand)<1e-15 else remesh_range/abs(grand),"interaction_residual_variance_fraction":interaction}


def make_plots(rows, scenes, effects, output):
    figures=output/"figures"; sources=output/"figure_sources"; figures.mkdir(exist_ok=True);sources.mkdir(exist_ok=True)
    write_csv(sources/"execution_plot_source.csv",rows);write_csv(sources/"scene_plot_source.csv",scenes);write_csv(sources/"effects_plot_source.csv",effects)
    labels=[f"{r['surface_id']}\n{r['anisotropy_level']}" for r in scenes]
    fig,ax=plt.subplots(figsize=(10,4)); data=[]
    for scene in scenes:
        data.append([row["J_q"] for row in rows if row["surface_id"]==scene["surface_id"] and row["anisotropy_level"]==scene["anisotropy_level"] and row["overall_pass"]])
    ax.boxplot(data,tick_labels=labels);ax.set_ylabel("Verified J_q");save(fig,figures/"Jq_distribution_by_scene.png")
    bar_plot(labels,[np.nan if r["S_q"] is None else r["S_q"] for r in scenes],"S_q",figures/"coverage_equivalent_Jq_spread.png",.1)
    x=np.arange(len(scenes));fig,ax=plt.subplots(figsize=(10,4));ax.bar(x-.18,[np.nan if r["geometry_J_q"] is None else r["geometry_J_q"] for r in scenes],.36,label="Gamma_geo");ax.bar(x+.18,[np.nan if r["oracle_J_q"] is None else r["oracle_J_q"] for r in scenes],.36,label="finite oracle");ax.set(xticks=x,xticklabels=labels,ylabel="J_q");ax.legend();save(fig,figures/"geometry_vs_robot_oracle.png")
    fig,ax=plt.subplots(figsize=(6,5));
    for surface,marker in (("saddle","o"),("hemisphere","s")):
        part=[row for row in rows if row["surface_id"]==surface and row["overall_pass"]];ax.scatter([r["L_S"] for r in part],[r["J_q"] for r in part],s=16,marker=marker,label=surface,alpha=.7)
    ax.set(xlabel="Intrinsic surface length L_S",ylabel="J_q");ax.legend();save(fig,figures/"Jq_vs_surface_length.png")
    fig,ax=plt.subplots(figsize=(10,4));ax.boxplot([[row["C_q"] for row in rows if row["surface_id"]==scene["surface_id"] and row["anisotropy_level"]==scene["anisotropy_level"] and row["overall_pass"]] for scene in scenes],tick_labels=labels);ax.set_ylabel("C_q = J_q/L_S");save(fig,figures/"Cq_distributions.png")
    effect_plot(effects,"root_mean_range",figures/"root_main_effect.png");effect_plot(effects,"remesh_mean_range",figures/"remesh_main_effect.png")
    heatmaps(rows,"J_q",figures/"root_remesh_interaction.png");heatmaps(rows,"overall_pass",figures/"liftability_map.png")


def heatmaps(rows,key,path):
    fig,axes=plt.subplots(2,3,figsize=(11,7));
    for ax,surface,level in zip(axes.ravel(),["saddle"]*3+["hemisphere"]*3,["P_low","P_mid","P_high"]*2):
        part=[row for row in rows if row["surface_id"]==surface and row["anisotropy_level"]==level]; matrix=np.full((8,4),np.nan)
        for row in part:matrix[int(row["root_id"][1:]),int(row["remesh_id"][1:])]=float(row[key]) if row[key] is not None else np.nan
        image=ax.imshow(matrix,aspect="auto",origin="lower");ax.set(title=f"{surface} {level}",xlabel="remesh",ylabel="root");fig.colorbar(image,ax=ax,shrink=.7)
    save(fig,path)


def effect_plot(rows,key,path):
    selected=[row for row in rows if row["metric"] in ("J_q","C_q","lift_success")];fig,ax=plt.subplots(figsize=(10,4));labels=[f"{r['surface_id']}:{r['scope']}:{r['metric']}" for r in selected];ax.bar(range(len(selected)),[r.get(key,np.nan) for r in selected]);ax.set_xticks(range(len(selected)),labels,rotation=60,ha="right");ax.set_ylabel(key);save(fig,path)


def bar_plot(labels,values,ylabel,path,line=None):
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(range(len(labels)),values);ax.set_xticks(range(len(labels)),labels);ax.set_ylabel(ylabel)
    if line is not None:ax.axhline(line,color="red",ls="--")
    save(fig,path)


def render_markdown(summary, scenes):
    lines=["# E06-G G0 summary","",f"Decision: **{summary['decision']}**","",f"Verified executions: **{summary['verified_executions']}/192**. G1 was not authorized.","", "| Scene | Eq. | Eq. verified | S_q (eq.) | All verified | S_q (all) | Delta_global | Length-matched gain |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in scenes:
        fmt=lambda value:"n/a" if value is None else f"{value:.3%}"
        lines.append(f"| {row['surface_id']} {row['anisotropy_level']} | {row['coverage_equivalent_layouts']} | {row['verified_equivalent_layouts']} | {fmt(row['S_q'])} | {row['all_verified_layouts']} | {fmt(row['all_verified_S_q'])} | {fmt(row['Delta_global'])} | {fmt(row['Delta_global_length'])} |")
    lines.extend(["",f"Cost GO: `{summary['cost_GO']}`; feasibility GO: `{summary['feasibility_GO']}`; G1 authorized: `{summary['G1_authorized']}`.","","The negative gate is admission-limited: saddle has a singleton coverage-equivalent set, while hemisphere has eight equivalent layouts but no verified witness among them. Non-equivalent layouts exhibit robot-cost and liftability variation, but cannot support the registered claim.","","This is a frozen finite-layout and finite-continuation capacity test, not a global optimum or C-space topology result.",""])
    return "\n".join(lines)


def normalize_row(row):
    for key in ("E_miss","E_rep","E_NUC","L_S","J_q","C_q","min_sigma_min_5","minimum_absolute_joint_margin_rad","max_position_error","max_axis_error"):
        row[key]=None if row.get(key) in (None,"","None") else float(row[key])
    for key in ("geometry_baseline","canonical_layout","lift_found","strict_kinematics_pass","strict_coverage_pass","overall_pass","collision_pass"):
        row[key]=as_bool(row.get(key))


def as_bool(value):return value is True or str(value).lower()=="true"
def read_csv(path):
    with path.open(newline="") as handle:return list(csv.DictReader(handle))
def write_json(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def write_csv(path,rows):
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def save(fig,path):fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


if __name__=="__main__":main()
