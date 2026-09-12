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
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize E06-J")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surface_configuration_coupling_gate_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/surface_configuration_coupling_gate_v1")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="") as handle: return list(csv.DictReader(handle))


def num(row, key):
    value = row.get(key, "")
    return np.nan if value in (None, "", "None", "nan") else float(value)


def main():
    args=parse_args(); config=json.loads(args.config.read_text())
    tables={name:{row["window_id"]:row for row in read_csv(args.output/f"{name}_results.csv")} for name in ("F0","F1","F2","F3")}
    ids=sorted(tables["F0"])
    if len(ids)!=60 or any(set(table)!=set(ids) for table in tables.values()): raise RuntimeError("summary requires 60 complete F0-F3 windows")
    paired=[]
    for window_id in ids:
        f0,f1,f2,f3=(tables[name][window_id] for name in ("F0","F1","F2","F3")); j0,j1,j2,j3=(num(row,"J_q") for row in (f0,f1,f2,f3))
        paired.append({
            "window_id":window_id,"surface_id":f0["surface_id"],"anisotropy_level":f0["anisotropy_level"],"placement_id":f0["placement_id"],
            "J0":j0,"J1":j1,"J2":j2,"J3":j3,"Delta_conf":(j0-j1)/j0,"Delta_surface":(j0-j2)/j0,"Delta_joint":(j0-j3)/j0,
            "B_joint":(min(j1,j2)-j3)/j0,"F3_beats_F1":j3<j1-1e-12,"F3_beats_F2":j3<j2-1e-12,"F3_beats_both":j3<min(j1,j2)-1e-12,
            "surface0":num(f0,"surface_length"),"surface1":num(f1,"surface_length"),"surface2":num(f2,"surface_length"),"surface3":num(f3,"surface_length"),
            "terminal1":num(f1,"terminal_q_mismatch"),"terminal2":num(f2,"terminal_q_mismatch"),"terminal3":num(f3,"terminal_q_mismatch"),
            "sigma0":num(f0,"min_sigma_min_5"),"sigma1":num(f1,"min_sigma_min_5"),"sigma2":num(f2,"min_sigma_min_5"),"sigma3":num(f3,"min_sigma_min_5"),
            "margin0":num(f0,"minimum_absolute_joint_margin_rad"),"margin1":num(f1,"minimum_absolute_joint_margin_rad"),"margin2":num(f2,"minimum_absolute_joint_margin_rad"),"margin3":num(f3,"minimum_absolute_joint_margin_rad"),
            "E_NUC0":num(f0,"E_NUC"),"E_NUC1":num(f1,"E_NUC"),"E_NUC2":num(f2,"E_NUC"),"E_NUC3":num(f3,"E_NUC"),
            "E_miss0":num(f0,"E_miss"),"E_miss1":num(f1,"E_miss"),"E_miss2":num(f2,"E_miss"),"E_miss3":num(f3,"E_miss"),
            "E_rep0":num(f0,"E_rep"),"E_rep1":num(f1,"E_rep"),"E_rep2":num(f2,"E_rep"),"E_rep3":num(f3,"E_rep"),
            "LG0":num(f0,"integrated_L_G_diagnostic"),"LG1":num(f1,"integrated_L_G_diagnostic"),"LG2":num(f2,"integrated_L_G_diagnostic"),"LG3":num(f3,"integrated_L_G_diagnostic"),
            "kappa0":num(f0,"median_log_kappa_R"),"kappa1":num(f1,"median_log_kappa_R"),"kappa2":num(f2,"median_log_kappa_R"),"kappa3":num(f3,"median_log_kappa_R"),
            "cheap0":num(f0,"mean_abs_alignment_cheap"),"cheap1":num(f1,"mean_abs_alignment_cheap"),"cheap2":num(f2,"mean_abs_alignment_cheap"),"cheap3":num(f3,"mean_abs_alignment_cheap"),
            "optimized1":f1["optimized"]=="True","optimized2":f2["optimized"]=="True","optimized3":f3["optimized"]=="True",
            "surface_candidate2":f2["surface_candidate_id"],"surface_candidate3":f3["surface_candidate_id"],
            "F0_replay_error":num(f0,"baseline_reconstruction_error"),
        })
    write_csv(args.output/"paired_formulation_results.csv",paired)
    diagnostics=read_csv(args.output/"solver_diagnostics.csv")
    summary=summarize(paired,diagnostics,config)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    make_plots(paired,diagnostics,args.output)
    (args.output/"summary.md").write_text(markdown(summary))
    (args.output/"reproduction_commands.txt").write_text("\n".join([
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_surface_configuration_coupling_gate.py --stage freeze-candidates",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_surface_configuration_coupling_gate.py --stage run --jobs 16",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/summarize_surface_configuration_coupling_gate.py",
        "/data/chocheng/.venvs/coverage-fm/bin/python -m pytest -q",
    ])+"\n")
    print(json.dumps({"decision":summary["decision"],"outcome":summary["outcome"],"medium_high":summary["primary"]["medium_high"]},indent=2))


def vals(rows,key): return np.asarray([float(row[key]) for row in rows],dtype=np.float64)


def bootstrap_ci(data,config):
    values=np.asarray(data); rng=np.random.default_rng(config["statistics"]["bootstrap_seed"]); n=config["statistics"]["bootstrap_samples"]
    med=np.median(rng.choice(values,size=(n,len(values)),replace=True),axis=1)
    return [float(np.quantile(med,.025)),float(np.quantile(med,.975))]


def group_summary(rows,config):
    if not rows:return {"count":0}
    return {"count":len(rows),"median_Delta_conf":float(np.median(vals(rows,"Delta_conf"))),"median_Delta_surface":float(np.median(vals(rows,"Delta_surface"))),"median_Delta_joint":float(np.median(vals(rows,"Delta_joint"))),"median_B_joint":float(np.median(vals(rows,"B_joint"))),"B_joint_IQR":[float(np.quantile(vals(rows,"B_joint"),.25)),float(np.quantile(vals(rows,"B_joint"),.75))],"B_joint_bootstrap_95ci":bootstrap_ci(vals(rows,"B_joint"),config),"F3_beats_F1_fraction":float(np.mean([row["F3_beats_F1"] for row in rows])),"F3_beats_F2_fraction":float(np.mean([row["F3_beats_F2"] for row in rows])),"F3_beats_both_fraction":float(np.mean([row["F3_beats_both"] for row in rows]))}


def summarize(rows,diagnostics,config):
    mh=[row for row in rows if row["anisotropy_level"] in ("P_mid","P_high")]; primary=group_summary(mh,config); gate=config["gate"]
    sigma_cut=float(np.quantile(vals(mh,"sigma0"),.25))
    subsets={
        "remove_lowest_sigma_quartile":[row for row in mh if row["sigma0"]>sigma_cut],
        "terminal_le_0p025":[row for row in mh if max(row["terminal1"],row["terminal2"],row["terminal3"])<=config["admission"]["maximum_terminal_q_mismatch_sensitivity_rad"]],
        "surface_change_le_1pct":[row for row in mh if max(abs(row["surface2"]-row["surface0"])/row["surface0"],abs(row["surface3"]-row["surface0"])/row["surface0"])<=config["admission"]["maximum_relative_surface_length_change_sensitivity"]],
    }
    sensitivity={name:group_summary(part,config) for name,part in subsets.items()}
    persistence=all(value.get("F3_beats_both_fraction",0)>=.5 and value.get("median_B_joint",-1)>0 for value in sensitivity.values())
    checks={"F3_beats_both":primary["F3_beats_both_fraction"]>=gate["minimum_medium_high_F3_beats_both_fraction"],"median_B_joint":primary["median_B_joint"]>=gate["minimum_medium_high_median_B_joint"],"median_Delta_joint":primary["median_Delta_joint"]>=gate["minimum_medium_high_median_Delta_joint"],"sensitivity_persistence":persistence}
    if all(checks.values()): outcome="C_coupling_dominated"
    elif primary["median_Delta_conf"]>=.05 and abs(primary["median_Delta_joint"]-primary["median_Delta_conf"])<.02 and primary["median_B_joint"]<.02: outcome="A_configuration_dominated"
    elif primary["median_Delta_surface"]>=.05 and abs(primary["median_Delta_joint"]-primary["median_Delta_surface"])<.02 and primary["median_B_joint"]<.02: outcome="B_surface_dominated"
    else: outcome="D_low_local_headroom"
    solved={}
    for formulation in ("F1","F2","F3"):
        selected=[row for row in diagnostics if row["formulation"]==formulation]
        runtimes=np.asarray([num(row,"solve_time") for row in selected],dtype=np.float64)
        runtimes=runtimes[np.isfinite(runtimes)]
        failure_counts={}
        for row in selected:
            reason=row["failure_reason"] or "solver_success"
            failure_counts[reason]=failure_counts.get(reason,0)+1
        optimized=sum(row[f"optimized{formulation[-1]}"] for row in rows)
        solved[formulation]={
            "attempts":len(selected),
            "solver_solved":sum(row["solver_solved"]=="True" for row in selected),
            "strict_admitted":sum(row["admitted"]=="True" for row in selected),
            "selected_updates":sum(row["selected_update"]=="True" for row in selected),
            "final_optimized_windows":optimized,
            "final_baseline_fallback_windows":len(rows)-optimized,
            "median_solver_time_s":None if len(runtimes)==0 else float(np.median(runtimes)),
            "failure_counts":failure_counts,
        }
    controls={"max_F0_replay_error":float(max(row["F0_replay_error"] for row in rows)),"max_surface_change":float(max(max(abs(row["surface2"]-row["surface0"]),abs(row["surface3"]-row["surface0"]))/row["surface0"] for row in rows)),"max_terminal_mismatch":float(max(max(row["terminal1"],row["terminal2"],row["terminal3"]) for row in rows)),"max_E_NUC_change":float(max(abs(row[f"E_NUC{i}"]-row["E_NUC0"]) for row in rows for i in (1,2,3))),"max_E_miss_change":float(max(abs(row[f"E_miss{i}"]-row["E_miss0"]) for row in rows for i in (1,2,3))),"max_E_rep_change":float(max(abs(row[f"E_rep{i}"]-row["E_rep0"]) for row in rows for i in (1,2,3))),"min_sigma":float(min(row[f"sigma{i}"] for row in rows for i in (0,1,2,3))),"min_absolute_joint_margin":float(min(row[f"margin{i}"] for row in rows for i in (0,1,2,3)))}
    metric={"median_LG_change_F1":float(np.median((vals(rows,"LG0")-vals(rows,"LG1"))/vals(rows,"LG0"))),"median_LG_change_F2":float(np.median((vals(rows,"LG0")-vals(rows,"LG2"))/vals(rows,"LG0"))),"median_LG_change_F3":float(np.median((vals(rows,"LG0")-vals(rows,"LG3"))/vals(rows,"LG0"))),"spearman_Delta_LG3_Delta_joint":float(spearmanr((vals(rows,"LG0")-vals(rows,"LG3"))/vals(rows,"LG0"),vals(rows,"Delta_joint")).statistic),"median_cheap_alignment_F0":float(np.median(vals(rows,"cheap0"))),"median_cheap_alignment_F3":float(np.median(vals(rows,"cheap3")))}
    selected_surface_match=sum(row["surface_candidate2"]==row["surface_candidate3"] for row in rows)
    return {"experiment":config["experiment"],"windows":len(rows),"primary":{"all":group_summary(rows,config),"medium_high":primary},"by_level":{level:group_summary([row for row in rows if row["anisotropy_level"]==level],config) for level in ("P_low","P_mid","P_high")},"by_surface":{surface:group_summary([row for row in rows if row["surface_id"]==surface],config) for surface in ("saddle","hemisphere")},"solver":solved,"controls":controls,"metric_diagnostics":metric,"F2_F3_same_selected_surface_candidate_windows":selected_surface_match,"sensitivity":sensitivity,"gate_checks":checks,"decision":"GO" if all(checks.values()) else "NO-GO","outcome":outcome}


def make_plots(rows,diagnostics,output):
    figures=output/"figures"; sources=output/"figure_sources"; figures.mkdir(exist_ok=True); sources.mkdir(exist_ok=True); write_csv(sources/"paired_source.csv",rows); write_csv(sources/"solver_source.csv",diagnostics)
    series=[vals(rows,key) for key in ("J0","J1","J2","J3")]; paired_plot(series,["F0","F1","F2","F3"],"Verified $J_q$",figures/"paired_Jq.png")
    series=[vals(rows,key) for key in ("Delta_conf","Delta_surface","Delta_joint")]; box_plot(series,["F1 config","F2 surface","F3 joint"],"Improvement from F0",figures/"formulation_improvements.png")
    fig,ax=plt.subplots(figsize=(6,4)); ax.hist(vals(rows,"B_joint"),bins=16); ax.axvline(0,color="k",lw=1); ax.set(xlabel="$B_{joint}$",ylabel="Windows"); save(fig,figures/"coupling_benefit.png")
    box_plot([vals([row for row in rows if row["anisotropy_level"]==level],"B_joint") for level in ("P_low","P_mid","P_high")],["low","mid","high"],"$B_{joint}$",figures/"coupling_by_anisotropy.png")
    win_source=[]
    for level in ("P_low","P_mid","P_high"):
        part=[row for row in rows if row["anisotropy_level"]==level]
        for key,label in (("F3_beats_F1","F3<F1"),("F3_beats_F2","F3<F2"),("F3_beats_both","F3<both")):win_source.append({"level":level,"comparison":label,"fraction":float(np.mean([row[key] for row in part]))})
    write_csv(sources/"win_rates_source.csv",win_source); fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(3); width=.25
    for i,label in enumerate(("F3<F1","F3<F2","F3<both")):ax.bar(x+(i-1)*width,[row["fraction"] for row in win_source if row["comparison"]==label],width,label=label)
    ax.set(xticks=x,xticklabels=["low","mid","high"],ylim=(0,1),ylabel="Paired win fraction");ax.legend();save(fig,figures/"F3_win_rates.png")
    box_plot([np.abs(vals(rows,f"surface{i}")-vals(rows,"surface0"))/vals(rows,"surface0") for i in (1,2,3)],["F1","F2","F3"],"Relative surface-length change",figures/"surface_length_control.png")
    box_plot([[row[f"E_NUC{i}"]-row["E_NUC0"] for row in rows] for i in (1,2,3)],["F1","F2","F3"],"E_NUC change",figures/"NUC_control.png")
    box_plot([vals(rows,f"terminal{i}") for i in (1,2,3)],["F1","F2","F3"],"Terminal q mismatch [rad]",figures/"terminal_control.png")
    box_plot([vals(rows,f"sigma{i}") for i in (0,1,2,3)],["F0","F1","F2","F3"],"Minimum sigma_min_5",figures/"sigma_control.png")
    fig,ax=plt.subplots(figsize=(6,4));
    for i,label in ((1,"F1"),(2,"F2"),(3,"F3")):ax.scatter((vals(rows,"LG0")-vals(rows,f"LG{i}"))/vals(rows,"LG0"),vals(rows,f"Delta_{'conf' if i==1 else 'surface' if i==2 else 'joint'}"),s=14,label=label,alpha=.7)
    ax.set(xlabel="Diagnostic integrated L_G reduction",ylabel="Actual J_q reduction");ax.legend();save(fig,figures/"LG_vs_Jq.png")
    box_plot([vals(rows,f"cheap{i}") for i in (0,1,2,3)],["F0","F1","F2","F3"],"Mean cheap-eigendirection alignment",figures/"cheap_alignment.png")
    failure={}
    for row in diagnostics:
        key=(row["formulation"],row["failure_reason"] or "success"); failure[key]=failure.get(key,0)+1
    failure_rows=[{"formulation":key[0],"reason":key[1],"count":value} for key,value in failure.items()];write_csv(sources/"failure_source.csv",failure_rows)
    fig,ax=plt.subplots(figsize=(9,4)); labels=[f"{r['formulation']}:{r['reason']}" for r in failure_rows];ax.bar(np.arange(len(labels)),[r["count"] for r in failure_rows]);ax.set_xticks(np.arange(len(labels)),labels,rotation=45,ha="right");ax.set(ylabel="Attempts");save(fig,figures/"solver_outcomes.png")


def paired_plot(series,labels,ylabel,path):
    fig,ax=plt.subplots(figsize=(8,4));
    for row in zip(*series):ax.plot(range(len(series)),row,color="0.8",lw=.5)
    for i,(data,label) in enumerate(zip(series,labels)):ax.scatter(np.full(len(data),i),data,s=9,label=label)
    ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel=ylabel);save(fig,path)


def box_plot(series,labels,ylabel,path):
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot(series,tick_labels=labels);ax.axhline(0,color="0.6",lw=.8);ax.set(ylabel=ylabel);save(fig,path)


def save(fig,path):fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
def write_csv(path,rows):
    fields=sorted({key for row in rows for key in row})
    with Path(path).open("w",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def markdown(summary):
    p=summary["primary"]["medium_high"]
    levels="\n".join(
        f"| {level} | {value['median_Delta_conf']:.3%} | {value['median_Delta_surface']:.3%} | "
        f"{value['median_Delta_joint']:.3%} | {value['median_B_joint']:.3%} | {value['F3_beats_both_fraction']:.1%} |"
        for level,value in summary["by_level"].items()
    )
    controls=summary["controls"]
    return f"""# E06-J summary

Decision: **{summary['decision']}**  
Outcome: **{summary['outcome']}**

| Metric | Medium/high |
|---|---:|
| Median Delta_conf | {p['median_Delta_conf']:.4%} |
| Median Delta_surface | {p['median_Delta_surface']:.4%} |
| Median Delta_joint | {p['median_Delta_joint']:.4%} |
| Median B_joint | {p['median_B_joint']:.4%} |
| F3 beats F1 | {p['F3_beats_F1_fraction']:.2%} |
| F3 beats F2 | {p['F3_beats_F2_fraction']:.2%} |
| F3 beats both | {p['F3_beats_both_fraction']:.2%} |

| Level | Delta_conf | Delta_surface | Delta_joint | B_joint | F3 beats both |
|---|---:|---:|---:|---:|---:|
{levels}

All 60 F0 witnesses replayed exactly. F2 and F3 found a non-baseline admitted solution in
{summary['solver']['F2']['final_optimized_windows']} and {summary['solver']['F3']['final_optimized_windows']}
windows, respectively, and selected the same surface candidate in
{summary['F2_F3_same_selected_surface_candidate_windows']}/60 windows.

The final controls remained inside the frozen contract: maximum relative surface-length change
was {controls['max_surface_change']:.3%}, maximum terminal mismatch was
{controls['max_terminal_mismatch']:.6f} rad, maximum absolute E_NUC change was
{controls['max_E_NUC_change']:.6f}, minimum sigma_min_5 was {controls['min_sigma']:.6f}, and
minimum absolute joint-limit margin was {controls['min_absolute_joint_margin']:.6f} rad.

E06-J is a local numerical formulation-capacity gate. It does not establish global optimality,
full-path NUC planning benefit, hardware performance, C-space topology, or learned-planner benefit.
"""


if __name__=="__main__":main()
