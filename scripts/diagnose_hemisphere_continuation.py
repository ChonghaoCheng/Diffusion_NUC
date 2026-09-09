#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.diagnostics.continuation import classify_continuation_failure, require_unchanged_admission_contract
from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract, make_e06_surface, regenerate_e06_variants
from diffusion_coverage.nuc import build_nuc_ik_catalog, minimum_cost_nuc_lift
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


DEFAULT_BUDGET = {"random_restarts": 8, "max_candidates": 6, "orientation_cone_samples": 5, "max_active_branches": 6, "task_edge_samples": 7}
STRONG_BUDGET = {"random_restarts": 32, "max_candidates": 24, "orientation_cone_samples": 15, "max_active_branches": 24, "task_edge_samples": 15}


def main() -> None:
    parser=argparse.ArgumentParser(description="E06-D D4 hemisphere continuation localization")
    parser.add_argument("--stage",choices=("default","strong"),required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"results/nuc_robot_coupling_diagnosis_v1/d4_continuation")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    archived,rows,_=load_e06_contract(ROOT); config,frozen,placements=archived["config"],archived["frozen_contract"],archived["placements"]
    assert_default_contract(config,frozen)
    admission={"characteristic_length_m":float(config["robot"]["characteristic_length_m"]),"sigma_safe":float(frozen["sigma_safe"]),"delta_NUC":float(frozen["delta_NUC"]),"q_interpolation_step_rad":float(frozen["q_interpolation_step_rad"]),"axis_tolerance_degrees":float(config["robot"]["axis_tolerance_degrees"]),"position_tolerance_m":float(config["robot"]["position_tolerance_m"]),"coverage_path_sample_spacing_m":float(frozen["coverage_path_sample_spacing_m"])}
    require_unchanged_admission_contract(admission,dict(admission))
    surface_id="hemisphere"; surface_number=list(config["surfaces"]).index(surface_id)
    surface=make_e06_surface(config,surface_id,int(frozen["coverage_samples_per_face"]))
    variants=regenerate_e06_variants(surface,20,int(config["seed"])+100000*surface_number,int(config["e06"]["local_refinement_iterations"]))
    robot_cfg=config["robot"]; robot=UR5eKinematics(robot_cfg["model"],site_name=robot_cfg["site_name"],tool_axis_index=robot_cfg["tool_axis_index"],tool_axis_sign=robot_cfg["tool_axis_sign"])
    if args.stage == "default":
        selected_ids=None; budget=DEFAULT_BUDGET
    else:
        selection_path=args.output/"strong_search_selection.json"
        if not selection_path.exists(): raise RuntimeError("run --stage default and freeze selected IDs in the ARA before strong search")
        selection=json.loads(selection_path.read_text())
        if not selection.get("ara_frozen",False): raise RuntimeError("strong-search IDs have not been marked ARA-frozen")
        selected_ids={placement:set(ids) for placement,ids in selection["selected_ids"].items()}; budget=STRONG_BUDGET
    result_rows=[]; layer_rows=[]
    for placement_number,placement_id in enumerate(("P_mid","P_hard"),start=1):
        transform=np.asarray(placements["surfaces"][surface_id]["selected"][placement_id]["transform_base_from_surface"],dtype=np.float64)
        catalog_start=perf_counter()
        catalog=build_nuc_ik_catalog(
            robot,surface,variants[0],transform,
            axis_tolerance=np.deg2rad(float(robot_cfg["axis_tolerance_degrees"])),characteristic_length=float(robot_cfg["characteristic_length_m"]),sigma_safe=float(frozen["sigma_safe"]),
            random_restarts=budget["random_restarts"],max_candidates=budget["max_candidates"],orientation_cone_samples=budget["orientation_cone_samples"],
            seed=int(config["seed"])+1009*surface_number+9176*placement_number,collect_diagnostics=True,
        )
        catalog_wall=perf_counter()-catalog_start; cache={}
        indices=range(20) if selected_ids is None else sorted(int(value[1:]) for value in selected_ids[placement_id])
        for index in indices:
            start=perf_counter()
            lift=minimum_cost_nuc_lift(
                robot,surface,variants[index],catalog,transform,cache,
                axis_tolerance=np.deg2rad(float(robot_cfg["axis_tolerance_degrees"])),characteristic_length=float(robot_cfg["characteristic_length_m"]),sigma_safe=float(frozen["sigma_safe"]),
                task_edge_samples=budget["task_edge_samples"],surface_path_spacing=float(frozen["coverage_path_sample_spacing_m"]),maximum_joint_step=float(config["e06"]["maximum_joint_step_rad"]),position_tolerance=float(robot_cfg["construction_position_tolerance_m"]),max_active_branches=budget["max_active_branches"],collect_trace=True,
            )
            wall=perf_counter()-start; trace=lift.metadata.get("layer_trace",()); final=trace[-1] if trace else None
            strict_pass=None
            if lift.found:
                strict=check_strict_coverage_execution(
                    robot,(lift.q_path,),(lift.desired_positions,),(lift.desired_axes,),surface,transform,
                    footprint_radius=float(config["coverage"]["footprint_radius_m"]),position_tolerance=float(robot_cfg["position_tolerance_m"]),axis_tolerance=np.deg2rad(float(robot_cfg["axis_tolerance_degrees"])),characteristic_length=float(robot_cfg["characteristic_length_m"]),sigma_safe=float(frozen["sigma_safe"]),missed_tolerance=1.0,repeat_tolerance=10.0,nuc_error_tolerance=None,interpolation_joint_step=float(frozen["q_interpolation_step_rad"]),coverage_path_sample_spacing=float(frozen["coverage_path_sample_spacing_m"]),
                ); strict_pass=strict.kinematics_pass
                np.savez_compressed(args.output/f"strong_witness_{placement_id}_S{index:02d}.npz",q=lift.q_path,desired_positions=lift.desired_positions,desired_axes=lift.desired_axes)
            record={
                "stage":args.stage,"placement_id":placement_id,"skeleton_id":f"S{index:02d}","lift_found":lift.found,"strict_kinematics_pass":strict_pass,
                "failure_pose_index":None if lift.found or final is None else final.pose_index,"failure_progress":1.0 if lift.found else (None if final is None else final.normalized_progress),
                "provisional_failure_category":None if lift.found or final is None else classify_continuation_failure(final),
                "catalog_raw_candidates_total":sum(catalog.raw_candidate_counts),"catalog_safe_candidates_total":sum(len(x) for x in catalog.candidates),
                "catalog_wall_time":catalog_wall,"continuation_time":lift.continuation_time,"wall_time":wall,"evaluated_transitions":lift.evaluated_transitions,
                "random_restarts":budget["random_restarts"],"max_candidates":budget["max_candidates"],"orientation_cone_samples":budget["orientation_cone_samples"],"max_active_branches":budget["max_active_branches"],"task_edge_samples":budget["task_edge_samples"],
                "sigma_safe":float(frozen["sigma_safe"]),"axis_tolerance_degrees":float(robot_cfg["axis_tolerance_degrees"]),"q_interpolation_step_rad":float(frozen["q_interpolation_step_rad"]),
            }
            result_rows.append(record)
            for item in trace: layer_rows.append({"stage":args.stage,"placement_id":placement_id,"skeleton_id":f"S{index:02d}",**asdict(item)})
            print(f"{args.stage} {placement_id} S{index:02d} found={lift.found} progress={record['failure_progress']}",flush=True)
    write_rows(args.output/f"{args.stage}_results.jsonl",result_rows); write_csv(args.output/f"{args.stage}_results.csv",result_rows)
    write_rows(args.output/f"{args.stage}_layers.jsonl",layer_rows); write_csv(args.output/f"{args.stage}_layers.csv",layer_rows)
    if args.stage=="default":
        verify_default_replay(rows,result_rows); selection=select_strong_subset(rows,result_rows)
        (args.output/"strong_search_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
        make_default_plots(args.output,result_rows,layer_rows,surface)
    else:
        summarize_strong(args.output,result_rows)


def assert_default_contract(config,frozen):
    assert np.isclose(config["robot"]["characteristic_length_m"],.1)
    assert np.isclose(frozen["sigma_safe"],.0723741717)
    assert np.isclose(frozen["delta_NUC"],.0297927413)
    assert np.isclose(frozen["q_interpolation_step_rad"],.05)
    assert DEFAULT_BUDGET["random_restarts"]==config["e06"]["ik_random_restarts"]
    assert DEFAULT_BUDGET["max_candidates"]==config["e06"]["ik_max_candidates"]
    assert DEFAULT_BUDGET["task_edge_samples"]==config["e06"]["task_edge_samples"]


def verify_default_replay(archived_rows,replay):
    expected={(r["placement_id"],r["skeleton_id"]):r for r in archived_rows if r["surface_id"]=="hemisphere" and r["placement_id"] in {"P_mid","P_hard"}}
    for row in replay:
        old=expected[(row["placement_id"],row["skeleton_id"])]
        if bool(old["lift_found"]) != bool(row["lift_found"]): raise RuntimeError("default replay disagrees with archived E06 lift_found")
    if len(replay)!=40: raise RuntimeError("default replay must contain all forty E06 failures")


def select_strong_subset(archived,replay):
    selected={}
    for placement in ("P_mid","P_hard"):
        values=sorted((r for r in replay if r["placement_id"]==placement),key=lambda r:(r["failure_progress"],r["skeleton_id"]))
        geometry=next(r["skeleton_id"] for r in archived if r["surface_id"]=="hemisphere" and r["placement_id"]==placement and r["geometry_baseline"])
        ids=[geometry,values[0]["skeleton_id"],values[(len(values)-1)//2]["skeleton_id"],values[-1]["skeleton_id"]]
        selected[placement]=list(dict.fromkeys(ids))
    return {"selection_rule":"geometry baseline, earliest, lower-median, latest failure progress; ties by skeleton ID","selected_ids":selected,"strong_budget":STRONG_BUDGET,"ara_frozen":False}


def make_default_plots(output,results,layers,surface):
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for placement in ("P_mid","P_hard"):
        values=np.sort([r["failure_progress"] for r in results if r["placement_id"]==placement]); axes[0].step(values,np.arange(1,len(values)+1)/len(values),where="post",label=placement)
        axes[1].hist(values,bins=10,alpha=.45,label=placement)
    axes[0].set(xlabel="First failure progress",ylabel="ECDF"); axes[1].set(xlabel="First failure progress",ylabel="Count"); [a.legend() for a in axes]; figure.tight_layout(); figure.savefig(output/"failure_progress_ecdf_hist.png",dpi=180); plt.close(figure)
    metrics=[("candidate_count_after_sigma","Safe pose candidates"),("valid_outgoing_edges","Valid outgoing edges"),("minimum_sigma_min_5","Minimum sigma_min_5"),("minimum_joint_limit_margin","Minimum joint-limit margin")]
    figure,axes=plt.subplots(2,2,figsize=(11,8))
    for axis,(key,title) in zip(axes.ravel(),metrics):
        for placement in ("P_mid","P_hard"):
            chosen=[r for r in layers if r["placement_id"]==placement]
            bins=np.linspace(0,1,31); centers=.5*(bins[:-1]+bins[1:]); means=[]
            for low,high in zip(bins[:-1],bins[1:]):
                vals=[r[key] for r in chosen if r[key] is not None and low<=r["normalized_progress"]<high]; means.append(np.nan if not vals else np.mean(vals))
            axis.plot(centers,means,label=placement)
        axis.set(xlabel="Path progress",ylabel=title); axis.grid(alpha=.2); axis.legend()
    figure.tight_layout(); figure.savefig(output/"continuation_quantities_vs_progress.png",dpi=180); plt.close(figure)
    figure=plt.figure(figsize=(10,5)); axis=figure.add_subplot(111,projection="3d"); axis.plot_trisurf(*surface.vertices.T,triangles=surface.faces,color="#dddddd",alpha=.2)
    for placement,color in (("P_mid","#1f77b4"),("P_hard","#d62728")):
        points=np.asarray([r["surface_location"] for r in layers if r["placement_id"]==placement and any(x["placement_id"]==placement and x["failure_pose_index"]==r["pose_index"] and x["skeleton_id"]==r["skeleton_id"] for x in results)])
        if len(points): axis.scatter(*points.T,s=18,label=placement,color=color)
    axis.legend(); axis.set_title("First failure locations (base-frame coordinates)"); figure.tight_layout(); figure.savefig(output/"hemisphere_failure_locations.png",dpi=180); plt.close(figure)
    categories={name:sum(r["provisional_failure_category"]==name for r in results) for name in sorted({r["provisional_failure_category"] for r in results})}
    figure,axis=plt.subplots(figsize=(7,4)); axis.bar(categories.keys(),categories.values()); axis.tick_params(axis="x",rotation=25); axis.set_ylabel("Failures"); figure.tight_layout(); figure.savefig(output/"default_failure_categories.png",dpi=180); plt.close(figure)


def summarize_strong(output,strong):
    default=[json.loads(x) for x in (output/"default_results.jsonl").read_text().splitlines() if x]
    lookup={(r["placement_id"],r["skeleton_id"]):r for r in default}; comparison=[]
    for row in strong:
        old=lookup[(row["placement_id"],row["skeleton_id"])]
        comparison.append({**row,"default_failure_progress":old["failure_progress"],"strong_failure_progress":row["failure_progress"],"recovered":row["lift_found"],"final_failure_category":"beam_or_search_exhaustion" if row["lift_found"] else row["provisional_failure_category"],"candidate_expansion_factor":row["catalog_safe_candidates_total"]/max(old["catalog_safe_candidates_total"],1),"runtime_expansion_factor":row["wall_time"]/max(old["wall_time"],1e-12)})
    write_rows(output/"strong_comparison.jsonl",comparison); write_csv(output/"strong_comparison.csv",comparison)
    summary={"strong_cases":len(comparison),"strong_recoveries":sum(r["recovered"] for r in comparison),"by_placement":{p:{"cases":sum(r["placement_id"]==p for r in comparison),"recoveries":sum(r["placement_id"]==p and r["recovered"] for r in comparison)} for p in ("P_mid","P_hard")}}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


def write_rows(path,rows): path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__=="__main__": main()
