#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def read_jsonl(path): return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    output=ROOT/"results/nuc_robot_coupling_diagnosis_v1"
    e06=ROOT/"results/nuc_robot_skeleton_coupling_v1"
    d1=json.loads((output/"d1_structure/summary.json").read_text())
    d2=read_jsonl(output/"d2_cost_decomposition/candidate_decomposition.jsonl")
    d3=json.loads((output/"d3_execution_metric/summary.json").read_text())
    scenes=read_jsonl(e06/"scene_results.jsonl")
    default=read_jsonl(output/"d4_continuation/default_results.jsonl")
    strong=read_jsonl(output/"d4_continuation/strong_comparison.jsonl")
    table=[]
    for scene in scenes:
        if scene["L_q_relative_spread"] is None: continue
        key=f"{scene['surface_id']}/{scene['placement_id']}"
        selected=[r for r in d2 if r["surface_id"]==scene["surface_id"] and r["placement_id"]==scene["placement_id"] and r["overall_pass"]]
        table.append({
            "surface":scene["surface_id"],"placement":scene["placement_id"],"E06_L_q_spread":scene["L_q_relative_spread"],
            "median_eta_var":float(np.median([r["eta_var"] for r in selected])),"median_eta_cross":float(np.median([r["eta_cross"] for r in selected])),
            "L_G_spread":d3[key]["L_G_relative_spread"],"Pearson_L_G_L_q":d3[key]["candidate_pearson"],"Spearman_L_G_L_q":d3[key]["candidate_spearman"],
        })
    d4=[]
    for placement in ("P_mid","P_hard"):
        selected=[r for r in default if r["placement_id"]==placement]
        selected_strong=[r for r in strong if r["placement_id"]==placement]
        categories=[]
        strong_lookup={r["skeleton_id"]:r for r in selected_strong}
        for row in selected:
            categories.append(strong_lookup.get(row["skeleton_id"],{}).get("final_failure_category",row["provisional_failure_category"]))
        d4.append({
            "placement":placement,"number_failed":len(selected),"dominant_failure_category":Counter(categories).most_common(1)[0][0],
            "median_first_failure_progress":float(np.median([r["failure_progress"] for r in selected])),"strong_search_cases":len(selected_strong),"strong_search_recoveries":sum(r["recovered"] for r in selected_strong),
        })
    metric_high=all(row["Pearson_L_G_L_q"] is not None and row["Pearson_L_G_L_q"]>=.8 for row in table)
    spreads_small=all(row["L_G_spread"] is not None and row["L_G_spread"]<.02 and row["E06_L_q_spread"]<.02 for row in table)
    total_recoveries=sum(row["strong_search_recoveries"] for row in d4); total_cases=sum(row["strong_search_cases"] for row in d4)
    primary="Outcome 2" if metric_high and spreads_small else "Outcome 3" if any(row["L_G_spread"]>=.1 for row in table) else "Outcome 1"
    hemisphere="Outcome 5" if total_recoveries>=total_cases/2 else "Outcome 4" if total_recoveries==0 else "mixed Outcome 4/5"
    summary={
        "experiment":"E06-D: NUC skeleton coupling mechanism diagnosis","parent_code_commit":"78876d52d5313c0e99978700ff3cb7de02e2d0a5",
        "archived_E06_reproduced":True,"D1":d1,"valid_scene_table":table,"D4":d4,
        "interpretation":{"primary":primary,"hemisphere":hemisphere,"strong_recoveries":total_recoveries,"strong_cases":total_cases,
                          "riemannian_line":"narrowed to a diagnostic; deformation remains blocked"},
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    write_csv(output/"valid_scene_summary.csv",table); write_csv(output/"hemisphere_failure_summary.csv",d4)
    (output/"summary.md").write_text(render(summary))
    commands="""cd /data/chocheng/Code/diffusion_coverage
MPLCONFIGDIR=/data/chocheng/.cache/matplotlib /data/chocheng/.venvs/coverage-fm/bin/python scripts/diagnose_nuc_skeleton_structure.py
MPLCONFIGDIR=/data/chocheng/.cache/matplotlib /data/chocheng/.venvs/coverage-fm/bin/python scripts/diagnose_nuc_joint_cost_decomposition.py
MPLCONFIGDIR=/data/chocheng/.cache/matplotlib /data/chocheng/.venvs/coverage-fm/bin/python scripts/diagnose_nuc_execution_metric.py
MPLCONFIGDIR=/data/chocheng/.cache/matplotlib /data/chocheng/.venvs/coverage-fm/bin/python scripts/diagnose_hemisphere_continuation.py --stage default
# Freeze strong_search_selection.json IDs in the ARA and set ara_frozen=true before continuing.
MPLCONFIGDIR=/data/chocheng/.cache/matplotlib /data/chocheng/.venvs/coverage-fm/bin/python scripts/diagnose_hemisphere_continuation.py --stage strong
/data/chocheng/.venvs/coverage-fm/bin/python scripts/summarize_nuc_coupling_diagnosis.py
/data/chocheng/.venvs/coverage-fm/bin/python -m pytest -q
"""
    (output/"reproduction_commands.txt").write_text(commands)
    print(json.dumps(summary["interpretation"],indent=2))


def render(summary):
    lines=["# E06-D NUC skeleton coupling diagnosis","",f"Primary classification: **{summary['interpretation']['primary']}**  ",f"Hemisphere classification: **{summary['interpretation']['hemisphere']}**  ",f"Riemannian line: {summary['interpretation']['riemannian_line']}","","| Surface | Placement | E06 Lq spread | eta_var | eta_cross | LG spread | Pearson | Spearman |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["valid_scene_table"]: lines.append(f"| {r['surface']} | {r['placement']} | {r['E06_L_q_spread']:.4f} | {r['median_eta_var']:.4f} | {r['median_eta_cross']:.4f} | {r['L_G_spread']:.4f} | {r['Pearson_L_G_L_q']:.4f} | {r['Spearman_L_G_L_q']:.4f} |")
    lines += ["","| Placement | Failed | Dominant category | Median progress | Strong cases | Recoveries |","|---|---:|---|---:|---:|---:|"]
    for r in summary["D4"]: lines.append(f"| {r['placement']} | {r['number_failed']} | {r['dominant_failure_category']} | {r['median_first_failure_progress']:.4f} | {r['strong_search_cases']} | {r['strong_search_recoveries']} |")
    lines += ["","This diagnosis does not revise E06, authorize E07, prove C-space disconnection, or establish deformation, learned-planner, global-optimality, or physical-execution benefit."]
    return "\n".join(lines)+"\n"


def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__=="__main__": main()
