#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import numpy as np

from diffusion_coverage.diagnostics.anisotropy import (
    absolute_joint_limit_margin,
    continue_probe_from_shared_q,
    require_frozen_experiment_inputs,
    select_uniform_indices,
)
from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract, make_e06_surface
from diffusion_coverage.geometry.robot_surface_metric import compute_robot_surface_metric, estimate_surface_contact_differential
from diffusion_coverage.geometry.surface_curve import trace_surface_curve
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


def parse_args():
    parser=argparse.ArgumentParser(description="E06-R R1 controlled directional probes")
    parser.add_argument("--stage",choices=("freeze-anchors","run"),required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/riemannian_anisotropy_v1.json")
    parser.add_argument("--scenes",type=Path,default=ROOT/"configs/riemannian_anisotropy_scenes_v1.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/riemannian_anisotropy_utility_v1")
    return parser.parse_args()


def main():
    args=parse_args(); config=json.loads(args.config.read_text()); scenes=json.loads(args.scenes.read_text()); require_frozen_experiment_inputs(scenes)
    archived,_,_=load_e06_contract(ROOT); args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=="freeze-anchors":
        frozen=freeze_anchors(config,scenes,archived)
        (args.output/"frozen_anchors.json").write_text(json.dumps(frozen,indent=2)+"\n")
        print(json.dumps({key:len(value) for key,value in frozen["anchors"].items()},indent=2)); return
    anchor_path=args.output/"frozen_anchors.json"
    if not anchor_path.exists(): raise RuntimeError("run freeze-anchors before R1")
    anchors=json.loads(anchor_path.read_text()); require_frozen_experiment_inputs(scenes,anchors)
    run_probes(config,scenes,anchors,archived,args.output)


def make_robot(archived):
    cfg=archived["config"]["robot"]
    return UR5eKinematics(cfg["model"],site_name=cfg["site_name"],tool_axis_index=cfg["tool_axis_index"],tool_axis_sign=cfg["tool_axis_sign"])


def freeze_anchors(config,scenes,archived):
    robot=make_robot(archived); r1=config["r1"]; contract=config["robot_contract"]; anchors={}; rejected={}
    for surface_id in ("saddle","hemisphere"):
        surface=make_e06_surface(archived["config"],surface_id,1)
        for level in ("P_low","P_mid","P_high"):
            scene=scenes["selected"][surface_id][level]; transform=np.asarray(scene["transform_base_from_surface"],dtype=float)
            witness=np.load(ROOT/scene["witness_file"]); q=witness["q"]; positions=witness["desired_positions"]
            cumulative=np.concatenate(([0.0],np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))))
            candidate_indices=np.unique(np.linspace(1,len(q)-2,min(450,len(q)-2),dtype=int)); eligible=[]; records={}; reject_counts={}
            for index in candidate_indices:
                reason=None; task=evaluate_task_kinematics_5d(robot,q[index],characteristic_length=contract["characteristic_length_m"])
                abs_margin=absolute_joint_limit_margin(robot,q[index:index+1])
                local_point=transform[:3,:3].T@(positions[index]-transform[:3,3])
                boundary_clearance=distance_to_mesh_boundary(surface,local_point)
                turn=baseline_turn_angle(positions,index)
                if boundary_clearance < r1["minimum_boundary_clearance_m"]: reason="surface_boundary_clearance"
                elif abs_margin <= r1["minimum_absolute_joint_margin_rad"]: reason="joint_margin"
                elif task.sigma_min_5 < r1["minimum_sigma_factor"]*contract["sigma_safe"]: reason="sigma_margin"
                elif turn > np.deg2rad(r1["maximum_baseline_turn_degrees"]): reason="baseline_turn"
                try:
                    contact=estimate_surface_contact_differential(surface,local_point,transform,task.axis_basis,characteristic_length=contract["characteristic_length_m"],finite_difference_step=config["metric"]["finite_difference_step_m"])
                    metric=compute_robot_surface_metric(task,contact.task_differential,minimum_singular_value=contract["sigma_safe"])
                    if metric.kappa_R > config["metric"]["maximum_kappa_R"]: reason=reason or "metric_conditioning"
                    tangent_surface=transform[:3,:3].T@contact.tangent_basis
                    for vector in metric.eigenvectors.T:
                        for sign in (-1.0,1.0):
                            trace_surface_curve(surface,contact.point_surface,sign*(tangent_surface@vector),max(r1["probe_lengths_m"]),maximum_step=r1["surface_curve_maximum_step_m"])
                except (ValueError,np.linalg.LinAlgError):
                    reason=reason or "local_surface_or_metric"
                if reason is None:
                    eligible.append(int(index)); records[int(index)]={
                        "witness_index":int(index),"point_surface":contact.point_surface.tolist(),"point_base":positions[index].tolist(),"q":q[index].tolist(),
                        "sigma_min_5":task.sigma_min_5,"minimum_absolute_joint_margin_rad":abs_margin,"surface_boundary_clearance_m":boundary_clearance,
                        "baseline_turn_rad":turn,"tangent_basis_surface":tangent_surface.tolist(),"eigenvalues":metric.eigenvalues.tolist(),"eigenvectors":metric.eigenvectors.tolist(),
                        "kappa_R":metric.kappa_R,"R_G0":metric.R_G,
                    }
                else: reject_counts[reason]=reject_counts.get(reason,0)+1
            selected=select_uniform_indices(cumulative,eligible,int(r1["maximum_anchors_per_scene"])); key=f"{surface_id}/{level}"
            anchors[key]=[{"anchor_id":f"A{number:02d}",**records[index]} for number,index in enumerate(selected)]
            rejected[key]={"candidate_samples":len(candidate_indices),"eligible":len(eligible),"selected":len(selected),"reasons":reject_counts}
    payload={"experiment":config["experiment"],"frozen_before_probe_results":True,"selection_rule":"preregistered deterministic eligibility then uniform baseline arclength","anchors":anchors,"rejection_summary":rejected}
    payload["content_hash"]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload


def run_probes(config,scenes,anchors,archived,output):
    robot=make_robot(archived); contract=config["robot_contract"]; r1=config["r1"]; rows=[]
    for scene_key,scene_anchors in anchors["anchors"].items():
        surface_id,level=scene_key.split("/"); surface=make_e06_surface(archived["config"],surface_id,int(archived["frozen_contract"]["coverage_samples_per_face"])); transform=np.asarray(scenes["selected"][surface_id][level]["transform_base_from_surface"],dtype=float)
        for anchor in scene_anchors:
            task=evaluate_task_kinematics_5d(robot,np.asarray(anchor["q"]),characteristic_length=contract["characteristic_length_m"])
            contact=estimate_surface_contact_differential(surface,np.asarray(anchor["point_surface"]),transform,task.axis_basis,characteristic_length=contract["characteristic_length_m"],finite_difference_step=config["metric"]["finite_difference_step_m"],tangent_basis_surface=np.asarray(anchor["tangent_basis_surface"]))
            metric=compute_robot_surface_metric(task,contact.task_differential,minimum_singular_value=contract["sigma_safe"])
            for requested_length in r1["probe_lengths_m"]:
                for eigen_index,eigen_name in ((0,"min"),(1,"max")):
                    for sign in (-1,1):
                        start=perf_counter(); direction=np.asarray(anchor["tangent_basis_surface"])@metric.eigenvectors[:,eigen_index]*sign
                        try:
                            curve=trace_surface_curve(surface,np.asarray(anchor["point_surface"]),direction,requested_length,maximum_step=r1["surface_curve_maximum_step_m"])
                            positions=curve.points@transform[:3,:3].T+transform[:3,3]; axes=-(curve.normals@transform[:3,:3].T); axes/=np.linalg.norm(axes,axis=1,keepdims=True)
                            continuation=continue_probe_from_shared_q(robot,np.asarray(anchor["q"]),positions,axes,maximum_joint_step=contract["maximum_continuation_joint_step_rad"],position_tolerance=contract["construction_position_tolerance_m"],axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]))
                            strict=None
                            if continuation.feasible and len(continuation.q_path)==len(positions):
                                strict=check_strict_coverage_execution(robot,(continuation.q_path,),(positions,),(axes,),surface,transform,footprint_radius=archived["config"]["coverage"]["footprint_radius_m"],position_tolerance=contract["position_tolerance_m"],axis_tolerance=np.deg2rad(contract["axis_tolerance_degrees"]),characteristic_length=contract["characteristic_length_m"],sigma_safe=contract["sigma_safe"],missed_tolerance=1.0,repeat_tolerance=10.0,interpolation_joint_step=contract["maximum_dense_q_step_rad"],coverage_path_sample_spacing=archived["frozen_contract"]["coverage_path_sample_spacing_m"])
                            passed=bool(strict is not None and strict.kinematics_pass)
                            lq=None if not passed else compute_joint_execution_cost((continuation.q_path,)).weighted_joint_length
                            lg=None if not passed else integrated_metric_length(robot,surface,transform,continuation.q_path,curve.points,contract,config)
                            row={"surface_id":surface_id,"anisotropy_level":level,"anchor_id":anchor["anchor_id"],"witness_index":anchor["witness_index"],"probe_length_requested":requested_length,"eigen_direction":eigen_name,"sign":sign,"R_G0":metric.R_G,"kappa_R":metric.kappa_R,"sqrt_lambda_initial":float(np.sqrt(metric.eigenvalues[eigen_index])),"surface_length_actual":curve.intrinsic_length,"relative_length_error":abs(curve.intrinsic_length-requested_length)/requested_length,"lift_found":continuation.feasible,"strict_kinematics_pass":passed,"L_q_actual":lq,"L_q_per_surface_length":None if lq is None else lq/curve.intrinsic_length,"L_G_integrated":lg,"min_sigma_min_5":None if strict is None else strict.min_sigma_min_5,"min_joint_limit_margin":None if strict is None else strict.min_joint_limit_margin,"minimum_absolute_joint_margin_rad":None if not passed else absolute_joint_limit_margin(robot,continuation.q_path),"max_position_error":continuation.max_position_error,"max_axis_error":continuation.max_axis_error,"continuation_runtime":perf_counter()-start,"failure_reason":continuation.failure_reason if strict is None else strict.failure_reason,"shared_q_hash":hashlib.sha256(np.asarray(anchor["q"],dtype=np.float64).tobytes()).hexdigest()}
                        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:
                            row={"surface_id":surface_id,"anisotropy_level":level,"anchor_id":anchor["anchor_id"],"witness_index":anchor["witness_index"],"probe_length_requested":requested_length,"eigen_direction":eigen_name,"sign":sign,"R_G0":metric.R_G,"kappa_R":metric.kappa_R,"sqrt_lambda_initial":float(np.sqrt(metric.eigenvalues[eigen_index])),"surface_length_actual":None,"relative_length_error":None,"lift_found":False,"strict_kinematics_pass":False,"L_q_actual":None,"L_q_per_surface_length":None,"L_G_integrated":None,"min_sigma_min_5":None,"min_joint_limit_margin":None,"minimum_absolute_joint_margin_rad":None,"max_position_error":None,"max_axis_error":None,"continuation_runtime":perf_counter()-start,"failure_reason":f"numerical_probe_failure:{type(error).__name__}","shared_q_hash":hashlib.sha256(np.asarray(anchor["q"],dtype=np.float64).tobytes()).hexdigest()}
                        rows.append(row); print(scene_key,anchor["anchor_id"],requested_length,eigen_name,sign,row["strict_kinematics_pass"],flush=True)
    r1dir=output/"r1_directional_probe"; r1dir.mkdir(exist_ok=True); write_rows(r1dir/"probe_results.jsonl",rows); write_csv(r1dir/"probe_results.csv",rows); write_rows(output/"probe_results.jsonl",rows); write_csv(output/"probe_results.csv",rows)


def integrated_metric_length(robot,surface,transform,q,points_surface,contract,config):
    total=0.0
    for index in range(len(q)-1):
        task=evaluate_task_kinematics_5d(robot,q[index],characteristic_length=contract["characteristic_length_m"])
        contact=estimate_surface_contact_differential(surface,points_surface[index],transform,task.axis_basis,characteristic_length=contract["characteristic_length_m"],finite_difference_step=config["metric"]["finite_difference_step_m"])
        metric=compute_robot_surface_metric(task,contact.task_differential,minimum_singular_value=contract["sigma_safe"])
        delta_base=transform[:3,:3]@(points_surface[index+1]-points_surface[index]); delta_xi=contact.tangent_basis.T@delta_base
        total+=float(np.sqrt(max(delta_xi@metric.matrix@delta_xi,0.0)))
    return total


def distance_to_mesh_boundary(surface,point):
    counts={}
    for face in surface.faces:
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b)))); counts[key]=counts.get(key,0)+1
    edges=[edge for edge,count in counts.items() if count==1]
    return min(point_segment_distance(point,surface.vertices[a],surface.vertices[b]) for a,b in edges)


def point_segment_distance(point,a,b):
    delta=b-a; t=np.clip(np.dot(point-a,delta)/max(np.dot(delta,delta),1e-15),0.0,1.0); return float(np.linalg.norm(point-(a+t*delta)))


def baseline_turn_angle(points,index,offset=4):
    first=points[index]-points[max(0,index-offset)]; second=points[min(len(points)-1,index+offset)]-points[index]
    if np.linalg.norm(first)<1e-10 or np.linalg.norm(second)<1e-10: return np.pi
    return float(np.arccos(np.clip(np.dot(first,second)/(np.linalg.norm(first)*np.linalg.norm(second)),-1.0,1.0)))


def write_rows(path,rows): path.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
def write_csv(path,rows):
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__=="__main__": main()
