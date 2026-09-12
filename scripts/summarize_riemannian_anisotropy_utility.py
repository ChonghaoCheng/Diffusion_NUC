#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); os.environ.setdefault("MPLCONFIGDIR","/data/chocheng/.cache/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr


OUTPUT=ROOT/"results/riemannian_anisotropy_utility_v1"


def main():
    config=json.loads((ROOT/"configs/riemannian_anisotropy_v1.json").read_text()); scenes=json.loads((ROOT/"configs/riemannian_anisotropy_scenes_v1.json").read_text()); anchors=json.loads((OUTPUT/"frozen_anchors.json").read_text())
    (OUTPUT/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    probes=load_jsonl(OUTPUT/"probe_results.jsonl"); pairs=make_pairs(probes,config); anchor_rows=aggregate_anchors(pairs)
    write_jsonl(OUTPUT/"paired_anchor_results.jsonl",pairs); write_csv(OUTPUT/"paired_anchor_results.csv",pairs)
    write_jsonl(OUTPUT/"anchor_results.jsonl",anchor_rows); write_csv(OUTPUT/"anchor_results.csv",anchor_rows)
    summary=summarize(pairs,anchor_rows,probes,scenes,anchors,config)
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); (OUTPUT/"summary.md").write_text(render_summary(summary))
    make_plots(pairs,probes,scenes,anchors)
    shutil.copy2(OUTPUT/"r0_scene_calibration/selected_placement_anisotropy.png",OUTPUT/"figures/selected_placement_anisotropy.png")
    shutil.copy2(OUTPUT/"r0_scene_calibration/selected_placement_anisotropy_source.csv",OUTPUT/"figures/selected_placement_anisotropy_source.csv")
    (OUTPUT/"reproduction_commands.txt").write_text(
        "PYTHONPATH=src python scripts/calibrate_riemannian_anisotropy_scenes.py --workers 8\n"
        "PYTHONPATH=src python scripts/run_riemannian_directional_probe.py --stage freeze-anchors\n"
        "# append frozen anchor IDs/hash to ARA before the next command\n"
        "PYTHONPATH=src python scripts/run_riemannian_directional_probe.py --stage run\n"
        "PYTHONPATH=src python scripts/summarize_riemannian_anisotropy_utility.py\npytest -q\n"
    )
    print(json.dumps(summary,indent=2))


def make_pairs(probes,config):
    grouped={}
    for row in probes: grouped.setdefault((row["surface_id"],row["anisotropy_level"],row["anchor_id"],row["probe_length_requested"]),[]).append(row)
    pairs=[]
    for key,rows in sorted(grouped.items()):
        lookup={(row["eigen_direction"],int(row["sign"])):row for row in rows}; required=[("min",-1),("min",1),("max",-1),("max",1)]
        valid=all(item in lookup and lookup[item]["strict_kinematics_pass"] for item in required)
        base={"surface_id":key[0],"anisotropy_level":key[1],"anchor_id":key[2],"probe_length_requested":key[3],"R_G0":rows[0]["R_G0"],"kappa_R":rows[0]["kappa_R"],"paired_strict_pass":valid}
        if len({row["shared_q_hash"] for row in rows}) != 1: raise RuntimeError("a probe pair did not share exactly one initial q")
        if not valid: pairs.append({**base,"R_q":None,"failure_count":sum(not row["strict_kinematics_pass"] for row in rows)}); continue
        costs={item:lookup[item]["L_q_actual"]/lookup[item]["surface_length_actual"] for item in required}
        cmin=.5*(costs[("min",-1)]+costs[("min",1)]); cmax=.5*(costs[("max",-1)]+costs[("max",1)])
        lengths=np.asarray([lookup[item]["surface_length_actual"] for item in required]); mismatch=float((lengths.max()-lengths.min())/key[3])
        asym_min=abs(costs[("min",1)]-costs[("min",-1)])/max(cmin,1e-12); asym_max=abs(costs[("max",1)]-costs[("max",-1)])/max(cmax,1e-12)
        pairs.append({**base,"R_q":cmax/cmin,"cbar_q_min":cmin,"cbar_q_max":cmax,"expensive_direction_wins":cmax>cmin,
            "relative_paired_length_mismatch":mismatch,"sign_asymmetry_min":asym_min,"sign_asymmetry_max":asym_max,"mean_sign_asymmetry":.5*(asym_min+asym_max),
            "min_sigma_min_5":min(row["min_sigma_min_5"] for row in rows),"min_joint_limit_margin":min(row["min_joint_limit_margin"] for row in rows),
            "minimum_absolute_joint_margin_rad":min(row["minimum_absolute_joint_margin_rad"] for row in rows),"failure_count":0})
    return pairs


def aggregate_anchors(pairs):
    grouped={}
    for row in pairs: grouped.setdefault((row["surface_id"],row["anisotropy_level"],row["anchor_id"]),[]).append(row)
    output=[]
    for key,rows in sorted(grouped.items()):
        valid=[row for row in rows if row["R_q"] is not None]
        if not valid: continue
        output.append({"surface_id":key[0],"anisotropy_level":key[1],"anchor_id":key[2],"R_G0":valid[0]["R_G0"],"R_q":float(np.exp(np.mean(np.log([r["R_q"] for r in valid])))),"expensive_direction_wins":float(np.exp(np.mean(np.log([r["R_q"] for r in valid]))))>1,"valid_probe_lengths":len(valid),"min_sigma_min_5":min(r["min_sigma_min_5"] for r in valid),"minimum_absolute_joint_margin_rad":min(r["minimum_absolute_joint_margin_rad"] for r in valid),"mean_sign_asymmetry":float(np.mean([r["mean_sign_asymmetry"] for r in valid])),"maximum_relative_paired_length_mismatch":max(r["relative_paired_length_mismatch"] for r in valid)})
    return output


def summarize(pairs,anchor_rows,probes,scenes,anchors,config):
    valid=anchor_rows; x=np.log([row["R_G0"] for row in valid]); y=np.log([row["R_q"] for row in valid])
    regression=linregress(x,y) if len(valid)>=3 and np.ptp(x)>0 else None; rho=float(spearmanr(x,y).statistic) if len(valid)>=2 else float("nan")
    by_level={}
    for level in ("P_low","P_mid","P_high"):
        selected=[row for row in valid if row["anisotropy_level"]==level]; values=np.asarray([row["R_q"] for row in selected])
        by_level[level]=distribution(values)
    medium_high=[row for row in valid if row["anisotropy_level"] in {"P_mid","P_high"}]
    sigma_cut=float(np.percentile([row["min_sigma_min_5"] for row in valid],25)) if valid else float("nan")
    robust=[row for row in valid if row["min_sigma_min_5"]>sigma_cut]
    robust_rho=float(spearmanr(np.log([r["R_G0"] for r in robust]),np.log([r["R_q"] for r in robust])).statistic) if len(robust)>=2 else float("nan")
    max_mismatch=max((row["maximum_relative_paired_length_mismatch"] for row in valid),default=float("inf"))
    residual=y-x; sigmas=np.asarray([row["min_sigma_min_5"] for row in valid]); margins=np.asarray([row["minimum_absolute_joint_margin_rad"] for row in valid])
    successful_probes=[row for row in probes if row["strict_kinematics_pass"]]
    integrated=np.asarray([row["L_G_integrated"] for row in successful_probes]); actual=np.asarray([row["L_q_actual"] for row in successful_probes])
    gate=config["gate"]; conditions={
        "pooled_spearman":rho>=gate["minimum_pooled_spearman"],
        "medium_high_win_fraction":mean_bool(medium_high,"expensive_direction_wins")>=gate["minimum_medium_high_win_fraction"],
        "high_median_R_q":by_level["P_high"]["median"] is not None and by_level["P_high"]["median"]>=gate["minimum_high_median_R_q"],
        "intrinsic_length_match":max_mismatch<=gate["maximum_relative_paired_length_mismatch"],
        "lowest_sigma_quartile_robustness":robust_rho>=gate["minimum_pooled_spearman"],
    }
    probe_success=sum(row["strict_kinematics_pass"] for row in probes)
    return {"experiment":config["experiment"],"R0":{"candidate_count":sum(len(scenes["all_candidate_placements"]) for _ in [0]),"admissible_count":sum(row["admitted"] for row in scenes["all_candidate_placements"]),"selected":{s:{l:{"candidate_id":v["candidate_id"],"median_log_kappa_R":v["median_log_kappa_R"],"median_R_G":float(np.exp(.5*v["median_log_kappa_R"])),"min_sigma_min_5":v["min_sigma_min_5"]} for l,v in levels.items()} for s,levels in scenes["selected"].items()}},
        "R1":{"frozen_anchor_count":sum(len(v) for v in anchors["anchors"].values()),"probe_count":len(probes),"successful_probes":probe_success,"eligible_anchor_length_pairs":sum(r["R_q"] is not None for r in pairs),"eligible_anchors":len(valid),"directional_win_fraction":mean_bool(valid,"expensive_direction_wins"),"medium_high_win_fraction":mean_bool(medium_high,"expensive_direction_wins"),"pooled_spearman_log":rho,"by_level":by_level,"by_scene":{f"{surface}/{level}":distribution(np.asarray([r["R_q"] for r in valid if r["surface_id"]==surface and r["anisotropy_level"]==level])) for surface in ("saddle","hemisphere") for level in ("P_low","P_mid","P_high")},"maximum_relative_paired_length_mismatch":max_mismatch,"median_sign_asymmetry":float(np.median([r["mean_sign_asymmetry"] for r in valid])) if valid else None,"sigma_low_quartile_cut":sigma_cut,"spearman_after_low_sigma_removal":robust_rho,"controls":{"log_ratio_residual_vs_sigma_pearson":correlation(residual,sigmas),"log_ratio_residual_vs_sigma_spearman":rank_correlation(residual,sigmas),"log_ratio_residual_vs_joint_margin_pearson":correlation(residual,margins),"log_ratio_residual_vs_joint_margin_spearman":rank_correlation(residual,margins),"integrated_LG_vs_actual_Lq_pearson":correlation(integrated,actual),"integrated_LG_vs_actual_Lq_spearman":rank_correlation(integrated,actual),"median_actual_over_integrated_LG":float(np.median(actual/integrated))},"probe_length":{str(length):distribution(np.asarray([r["R_q"] for r in pairs if r["R_q"] is not None and np.isclose(r["probe_length_requested"],length)])) for length in config["r1"]["probe_lengths_m"]},"regression":None if regression is None else {"a":float(regression.intercept),"b":float(regression.slope),"a_95ci":[float(regression.intercept-1.96*regression.intercept_stderr),float(regression.intercept+1.96*regression.intercept_stderr)],"b_95ci":[float(regression.slope-1.96*regression.stderr),float(regression.slope+1.96*regression.stderr)],"R2":float(regression.rvalue**2)}},
        "gate":{"conditions":conditions,"decision":"GO" if all(conditions.values()) else "NO-GO","thresholds":gate},
        "interpretation":"E06-R measures local directional utility only; no full-path optimization or deformation was performed."}


def distribution(values):
    if len(values)==0:return {"count":0,"median":None,"q25":None,"q75":None,"fraction_gt_1":None,"fraction_gt_1p1":None,"fraction_gt_1p2":None}
    return {"count":len(values),"median":float(np.median(values)),"q25":float(np.percentile(values,25)),"q75":float(np.percentile(values,75)),"fraction_gt_1":float(np.mean(values>1)),"fraction_gt_1p1":float(np.mean(values>1.1)),"fraction_gt_1p2":float(np.mean(values>1.2))}
def mean_bool(rows,key): return float(np.mean([row[key] for row in rows])) if rows else float("nan")
def correlation(x,y): return float(np.corrcoef(x,y)[0,1]) if len(x)>=2 and np.std(x)>0 and np.std(y)>0 else float("nan")
def rank_correlation(x,y): return float(spearmanr(x,y).statistic) if len(x)>=2 else float("nan")


def make_plots(pairs,probes,scenes,anchors):
    figdir=OUTPUT/"figures"; figdir.mkdir(exist_ok=True); valid=[r for r in pairs if r["R_q"] is not None]
    # Anchor/eigenvector visualization, source is frozen_anchors.json.
    figure=plt.figure(figsize=(11,5))
    for panel,surface in enumerate(("saddle","hemisphere"),1):
        axis=figure.add_subplot(1,2,panel,projection="3d")
        for level,color in zip(("P_low","P_mid","P_high"),("#2ca02c","#ffbf00","#d62728")):
            rows=anchors["anchors"][f"{surface}/{level}"]; points=np.asarray([r["point_surface"] for r in rows]); axis.scatter(*points.T,s=12,color=color,label=level)
            for row in rows:
                tangent=np.asarray(row["tangent_basis_surface"]); vectors=np.asarray(row["eigenvectors"]); point=np.asarray(row["point_surface"])
                for index,width in ((0,1.0),(1,2.0)):
                    direction=tangent@vectors[:,index]; axis.quiver(*point,*direction,length=.012,color=color,linewidth=width)
        axis.set_title(surface); axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(figdir/"anchors_and_eigenvectors.png",dpi=180); plt.close(figure)
    x=np.log([r["R_G0"] for r in valid]); y=np.log([r["R_q"] for r in valid]); figure,axis=plt.subplots(figsize=(6,5)); axis.scatter(x,y,c=[{"P_low":"#2ca02c","P_mid":"#ffbf00","P_high":"#d62728"}[r["anisotropy_level"]] for r in valid],alpha=.75); bounds=[min(np.min(x),np.min(y)),max(np.max(x),np.max(y))]; axis.plot(bounds,bounds,"k--"); axis.set(xlabel="log R_G0",ylabel="log R_q"); axis.grid(alpha=.2); save(figure,figdir/"predicted_vs_actual_ratio.png")
    grouped=[[r["R_q"] for r in valid if r["anisotropy_level"]==level] for level in ("P_low","P_mid","P_high")]; figure,axis=plt.subplots(figsize=(6,4)); axis.boxplot(grouped,tick_labels=["low","mid","high"]); axis.axhline(1,color="k",ls="--"); axis.set(ylabel="R_q"); save(figure,figdir/"Rq_by_anisotropy_level.png")
    rates=[np.mean([r["expensive_direction_wins"] for r in valid if r["anisotropy_level"]==level]) for level in ("P_low","P_mid","P_high")]; figure,axis=plt.subplots(figsize=(6,4)); axis.bar(["low","mid","high"],rates); axis.set(ylim=(0,1),ylabel="P(R_q > 1)"); save(figure,figdir/"expensive_direction_win_rate.png")
    for key,label,name in (("min_sigma_min_5","min sigma_min_5","Rq_vs_sigma.png"),("minimum_absolute_joint_margin_rad","min joint margin [rad]","Rq_vs_joint_margin.png")):
        figure,axis=plt.subplots(figsize=(6,4)); axis.scatter([r[key] for r in valid],[r["R_q"] for r in valid],alpha=.7); axis.axhline(1,color="k",ls="--"); axis.set(xlabel=label,ylabel="R_q"); axis.grid(alpha=.2); save(figure,figdir/name)
    lengths=sorted({r["probe_length_requested"] for r in valid}); by_anchor={}
    for row in valid: by_anchor.setdefault((row["surface_id"],row["anisotropy_level"],row["anchor_id"]),{})[row["probe_length_requested"]]=row["R_q"]
    matched=[v for v in by_anchor.values() if all(length in v for length in lengths)]; figure,axis=plt.subplots(figsize=(5,5)); axis.scatter([v[lengths[0]] for v in matched],[v[lengths[1]] for v in matched]); bounds=axis.get_xlim(); axis.plot(bounds,bounds,"k--"); axis.set(xlabel=f"R_q at {lengths[0]:.3f} m",ylabel=f"R_q at {lengths[1]:.3f} m"); save(figure,figdir/"probe_length_sensitivity.png")
    figure,axis=plt.subplots(figsize=(6,4)); axis.hist([r["mean_sign_asymmetry"] for r in valid],bins=15); axis.set(xlabel="mean sign asymmetry",ylabel="paired anchors"); save(figure,figdir/"sign_asymmetry.png")
    good=[r for r in probes if r["strict_kinematics_pass"]]; figure,axis=plt.subplots(figsize=(6,5)); axis.scatter([r["L_G_integrated"] for r in good],[r["L_q_actual"] for r in good],s=12,alpha=.5); axis.set(xlabel="integrated L_G",ylabel="actual L_q"); axis.grid(alpha=.2); save(figure,figdir/"integrated_LG_vs_actual_Lq.png")
    # The R0 selected-placement distribution is generated with its candidate source table by the calibration script.


def render_summary(s):
    lines=["# E06-R Riemannian anisotropy planning-utility gate","",f"Decision: **{s['gate']['decision']}**","","| Level | n | median R_q | P(R_q>1) |","|---|---:|---:|---:|"]
    for level,row in s["R1"]["by_level"].items(): lines.append(f"| {level} | {row['count']} | {fmt(row['median'])} | {fmt(row['fraction_gt_1'])} |")
    lines += ["",f"Pooled Spearman(log R_G0, log R_q): {fmt(s['R1']['pooled_spearman_log'])}",f"Medium/high win fraction: {fmt(s['R1']['medium_high_win_fraction'])}",f"Valid anchors: {s['R1']['eligible_anchors']}","","No full path was optimized and E07 was not run."]
    return "\n".join(lines)+"\n"
def fmt(v): return "NA" if v is None else f"{v:.4f}"
def save(fig,path): fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
def load_jsonl(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def write_jsonl(path,rows): path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r});
    with path.open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__=="__main__": main()
