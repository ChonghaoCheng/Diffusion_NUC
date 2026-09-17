from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

from diffusion_coverage.coverage.episode_summary import summarize_ordered_membership
from diffusion_coverage.planning.e12_programs import GeometryProgram, decode_program, encode_edge_sequence, load_geometry_library
from diffusion_coverage.robot.e09_execution import evaluate_synchronized_fk_trace, sphere_episode_counts_indexed, sphere_membership_stream
from diffusion_coverage.robot.e12_program_execution import densify_program_trace, lift_program
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from e09r1_runner_support import quadrature
from e09r1_runner_support import build_placement_graph, load_bank, load_robot_graph, save_plan, save_robot_graph, validate_unique_witness
from e10_runner_support import fixed_route_initialize
from e11_runner_support import _Q3Screen, _root_audit
from diffusion_coverage.solvers.structured_routing import greedy_prefix_completion, structured_anytime_search, replay_edge_sequence


def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,default=_json_default)+"\n")


def _json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline="") as f:return list(csv.DictReader(f))


def scene_map(root: Path) -> dict[str,dict[str,Any]]:
    out={}
    historical=json.loads((root/"configs/riemannian_anisotropy_scenes_v1.json").read_text())
    for x in historical["selected"]["hemisphere"].values():out[x["candidate_id"]]=x
    e11=json.loads((root/"configs/e11_transfer_scenes_v1.json").read_text())
    for x in e11["scenes"]:out[x["scene_id"]]=x
    e12=json.loads((root/"configs/e12_pose_splits_v1.json").read_text())
    for x in e12["scenes"]:out[x["scene_id"]]=x
    return out


def _source_from_base(positions: np.ndarray, transform: np.ndarray, radius: float) -> np.ndarray:
    raw=(positions-transform[:3,3])@transform[:3,:3]
    return radius*raw/np.linalg.norm(raw,axis=1,keepdims=True)


def _resample_sphere(points: np.ndarray, query: np.ndarray, radius: float) -> np.ndarray:
    p=np.asarray(points,dtype=np.float64);unit=p/np.linalg.norm(p,axis=1,keepdims=True)
    lengths=radius*np.arctan2(np.linalg.norm(np.cross(unit[:-1],unit[1:]),axis=1),np.sum(unit[:-1]*unit[1:],axis=1));cum=np.concatenate(([0.0],np.cumsum(lengths)));cum/=max(float(cum[-1]),1e-15)
    out=np.empty((len(query),3))
    for j,u in enumerate(query):
        i=max(0,min(int(np.searchsorted(cum,u,side="right"))-1,len(p)-2));span=cum[i+1]-cum[i];f=0 if span<=1e-15 else float((u-cum[i])/span);x=unit[i];y=unit[i+1];angle=float(np.arctan2(np.linalg.norm(np.cross(x,y)),np.dot(x,y)))
        v=(1-f)*x+f*y if angle<=1e-14 else (np.sin((1-f)*angle)*x+np.sin(f*angle)*y)/np.sin(angle);v/=np.linalg.norm(v);out[j]=radius*v
    return out


def _sequence_surface_at_u(sequence, u, lib):
    """Evaluate the original registered edge curve at concatenate_edges' exact parameter."""
    result=np.empty((len(u),3),dtype=np.float64)
    for edge_index,edge in enumerate(sequence):
        mask=(np.floor(np.minimum(u,len(sequence)-1e-12)).astype(int)==edge_index)
        if not np.any(mask):continue
        local=np.clip(u[mask]-edge_index,0.0,1.0);kind=str(edge.get("kind",""));geom=int(edge.get("geom_arc_id",-1))
        if geom>=0:points=lib.arc(geom)
        else:
            points=np.vstack((lib.ports[int(edge["start_port"])],lib.ports[int(edge["end_port"])]))
        result[mask]=_resample_sphere(points,local,lib.radius)
    return result


def prepare(root: Path, config: dict[str,Any], output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/config["inputs"]["geometry_bank_npz"],output/"geometry_only_library.npz")
    shutil.copyfile(root/config["inputs"]["geometry_bank_json"],output/"geometry_only_library.json")
    splits=root/config["inputs"]["pose_splits"]
    shutil.copyfile(splits,output/"pose_splits.json")
    allowed={"surface_geometry":["geometry_only_library.npz","geometry_only_library.json"],"task_inputs":["transform_base_from_surface","checked_q0","robot_model","physical_contract"],"forbidden":["robot graph","per-port q","edge feasibility","teacher future q","teacher graph edge IDs"]}
    manifest={"experiment":config["experiment"],"prepared_at":__import__('datetime').datetime.now().astimezone().isoformat(),"code_sha":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),"config_sha256":file_hash(root/"configs/e12_graph_free_global_generation_v1.json"),"pose_splits_sha256":file_hash(splits),"geometry_npz_sha256":file_hash(output/"geometry_only_library.npz"),"geometry_json_sha256":file_hash(output/"geometry_only_library.json"),"geometry_semantic_hash":config["geometry_hash"],"allowed_runtime_dependencies":allowed,"gpu_recorded_at_training":None,"collision_scope":config["collision_scope"]}
    write_json(output/"manifest.json",manifest);write_json(output/"allowed_runtime_dependencies.json",allowed);write_json(output/"prepare.checkpoint.json",{"complete":True})


def repair_e11_diagnostics(root: Path, config: dict[str,Any], output: Path) -> None:
    old=read_csv(root/"results/e11_mechanism_placement_transfer_v1/mechanism_diagnostics.csv")
    rows=[]
    for row in old:
        prefix=np.load(root/row["prefix_file"],allow_pickle=False)
        sigma=None
        if "sigma5" in prefix.files:sigma=float(np.min(prefix["sigma5"]))
        rows.append({**row,"correction_status":"direction_progress_recovered_from_executed_geometry_membership","legacy_forward_field_missing":row.get("fixed_route","")=="","prefix_sigma_measured":sigma,"prefix_sigma_status":"MEASURED" if sigma is not None else "NULL_NOT_ARCHIVED","historical_main_cells_rerun":False})
    write_csv(output/"e11_suffix_diagnostic_addendum.csv",rows);write_json(output/"repair-diagnostics.checkpoint.json",{"complete":True,"rows":len(rows),"main_cells_rerun":False})


def encode(root: Path, config: dict[str,Any], output: Path) -> None:
    lib=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));scenes=scene_map(root);manifest=read_csv(root/config["inputs"]["accepted_witness_manifest"]);rows=[];programs={}
    root_port=int(lib.arc_start[lib.routes["raster_u_phase_0.00/forward"][0]]);start=lib.ports[root_port]
    for item in manifest:
        z=np.load(root/item["file"],allow_pickle=False);refs=item["referenced_by"].split(";");scene_id=refs[0].split("/")[1];transform=np.asarray(scenes[scene_id]["transform_base_from_surface"],dtype=np.float64);active=np.asarray(z["activity"],bool);eligible=bool(active.all())
        row={"witness_hash":item["witness_hash"],"scene_id":scene_id,"referenced_by":item["referenced_by"],"actual_on_segments":int(active[0])+int(np.count_nonzero(active[1:]&~active[:-1])),"eligible_single_on":eligible}
        try:
            if not eligible:raise ValueError("OFF_or_multiple_ON_outside_scope")
            sequence=json.loads(str(z["sequence_json"]));program=encode_edge_sequence(sequence,lib,int(config["program"]["maximum_tokens"]));decoded=decode_program(program,lib,start,maximum_step=float(config["lifter"]["minimum_spacing_m"]),maximum_tokens=int(config["program"]["maximum_tokens"]));source=_source_from_base(np.asarray(z["target_position"]),transform,lib.radius);expected=_sequence_surface_at_u(sequence,np.asarray(z["u"],dtype=np.float64),lib);error=np.linalg.norm(source-expected,axis=1);max_pos=float(error.max());max_axis=float(np.arcsin(np.clip(max_pos/lib.radius,0,1)))
            q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);probe=np.asarray(q2["points"])[::max(1,len(q2["points"])//512)]
            membership_equal=bool(np.array_equal(sphere_membership_stream(probe,source,radius=lib.radius,footprint_radius=float(config["coverage"]["footprint_radius_m"])),sphere_membership_stream(probe,expected,radius=lib.radius,footprint_radius=float(config["coverage"]["footprint_radius_m"]))))
            roundtrip=max_pos<=float(config["program"]["roundtrip_position_m"])+1e-15 and max_axis<=float(config["program"]["roundtrip_axis_rad"])+1e-15 and membership_equal
            row.update({"encoding_status":"encoded" if roundtrip else "roundtrip_failed","program_hash":program.content_hash,"token_count":len(program.tokens),"scan_tokens":sum(x.kind=="SCAN" for x in program.tokens),"via_tokens":sum(x.kind=="VIA" for x in program.tokens),"max_position_reconstruction_m":max_pos,"max_axis_reconstruction_rad":max_axis,"membership_identity":membership_equal,"roundtrip_pass":roundtrip});programs[item["witness_hash"]]={"scene_id":scene_id,"program_json":program.to_json(),"program_hash":program.content_hash,"q0":np.asarray(z["q"])[0].tolist(),"start_surface_point":start.tolist()}
        except Exception as exc:row.update({"encoding_status":"failed","failure":f"{type(exc).__name__}: {exc}","roundtrip_pass":False})
        rows.append(row)
    write_csv(output/"witness_encoding.csv",rows);write_json(output/"encoded_programs.json",programs);write_json(output/"encode.checkpoint.json",{"complete":True,"witnesses":len(rows),"eligible":sum(x["eligible_single_on"] for x in rows),"roundtrip_pass":sum(bool(x.get("roundtrip_pass")) for x in rows)})


def _validate_trace(root,config,trace,surface,transform):
    radius=float(config["surface"]["radius_m"]);robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));metrics={};checks={}
    for temporal in ("T0","T1"):
        dense=densify_program_trace(trace,surface,transform,radius,joint_step=float(config["validation"]["temporal"][f"{temporal}_joint_step_rad"]),surface_step=float(config["validation"]["temporal"][f"{temporal}_surface_step_m"]));q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);check=evaluate_synchronized_fk_trace(robot,dense,transform,q2["points"][:1],q2["weights"][:1],sphere_radius=radius,footprint_radius=float(config["coverage"]["footprint_radius_m"]),characteristic_length=float(config["robot"]["characteristic_length_m"]));checks[temporal]=(dense,check)
    for temporal,name in (("T0","Q1"),("T0","Q2"),("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")):
        points,weights=quadrature(config,name,radius,root);dense,check=checks[temporal];counts=sphere_episode_counts_indexed(points,check.surface_points,dense.activity,radius=radius,footprint_radius=float(config["coverage"]["footprint_radius_m"]));total=float(weights.sum());metrics[f"E_miss_{temporal}_{name}"]=float(weights[counts==0].sum()/total);metrics[f"E_rep_{temporal}_{name}"]=float(np.dot(weights,np.maximum(counts-1,0))/total)
    t1=checks["T1"][1];required=[("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")];contract=all(metrics[f"E_miss_{t}_{q}"]<=float(config["coverage"]["missed_tolerance"])+1e-12 and metrics[f"E_rep_{t}_{q}"]<=float(config["coverage"]["repeat_tolerance"])+1e-12 for t,q in required);changes=max(max(abs(metrics[f"E_miss_T0_Q3"]-metrics[f"E_miss_T0_Q4"]),abs(metrics[f"E_rep_T0_Q3"]-metrics[f"E_rep_T0_Q4"])),max(abs(metrics[f"E_miss_T0_Q4"]-metrics[f"E_miss_T1_Q4"]),abs(metrics[f"E_rep_T0_Q4"]-metrics[f"E_rep_T1_Q4"])),max(abs(metrics[f"E_miss_T1_Q4"]-metrics[f"E_miss_T1_Q4a"]),abs(metrics[f"E_rep_T1_Q4"]-metrics[f"E_rep_T1_Q4a"])))
    motion=t1.max_position_error<=float(config["robot"]["position_tolerance_m"])+1e-12 and t1.max_axis_error<=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"]))+1e-12 and t1.min_sigma5>=float(config["robot"]["sigma_safe"])-1e-12 and t1.min_joint_margin>=-1e-12 and t1.collision_free
    status="accepted_under_E12_refined_sampled_checks" if motion and contract and changes<=float(config["coverage"]["resolution_tolerance"])+1e-12 else ("motion_contract_failed" if not motion else ("coverage_contract_failed" if contract is False and changes<=float(config["coverage"]["resolution_tolerance"])+1e-12 else "numerically_unresolved"))
    return {"validation_status":status,"max_resolution_change":changes,"min_sigma5":t1.min_sigma5,"max_position_error_m":t1.max_position_error,"max_axis_error_deg":float(np.rad2deg(t1.max_axis_error)),"min_joint_margin":t1.min_joint_margin,"collision_free":t1.collision_free,"J_q":float(np.linalg.norm(np.diff(trace.q,axis=0),axis=1).sum()),**metrics}


def oracle_relift(root: Path,config:dict[str,Any],output:Path)->None:
    lib=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));programs=json.loads((output/"encoded_programs.json").read_text());scenes=scene_map(root);rows=[];events=[];directory=output/"oracle_relift_witnesses";directory.mkdir(exist_ok=True)
    for order,(witness_hash,item) in enumerate(sorted(programs.items())):
        scene_id=item["scene_id"];scene=scenes[scene_id];transform=np.asarray(scene["transform_base_from_surface"],dtype=np.float64);robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));program=GeometryProgram.from_json(item["program_json"]);began=perf_counter();result=lift_program(program,lib,transform,np.asarray(item["q0"]),robot,config,start_surface_point=np.asarray(item["start_surface_point"]));events.append({"event_id":len(events),"kind":"oracle_lift","witness_hash":witness_hash,"scene_id":scene_id,"started_order":order,"duration_s":perf_counter()-began,"status":result.status,"ik_calls":result.ik_calls,"failure_reason":result.failure_reason,"failure_index":result.failure_index})
        row={"teacher_witness_hash":witness_hash,"scene_id":scene_id,"program_hash":program.content_hash,"lift_status":result.status,"ik_calls":result.ik_calls,"spacing_m":result.spacing_m,"halvings":result.halvings,"lift_seconds":result.elapsed_s,"failure_reason":result.failure_reason,"failure_index":result.failure_index}
        if result.trace is not None and result.decoded is not None:
            trace=result.trace;surface=result.decoded.surface_points;h=hashlib.sha256(trace.q.tobytes()+trace.u.tobytes()+trace.activity.tobytes()).hexdigest();path=directory/f"{h}.npz";np.savez_compressed(path,q=trace.q,u=trace.u,target_position=trace.target_position,target_axis=trace.target_axis,activity=trace.activity,surface_points=surface,program_json=np.asarray(program.to_json()),witness_hash=np.asarray(h));checked=_validate_trace(root,config,trace,surface,transform);row.update({"relift_witness_hash":h,"relift_file":str(path.relative_to(root)),**checked})
        rows.append(row);write_csv(output/"oracle_relift.partial.csv",rows)
    write_csv(output/"oracle_relift.csv",rows);write_csv(output/"execution_events.csv",events);passed=sum(x.get("validation_status")=="accepted_under_E12_refined_sampled_checks" for x in rows);write_json(output/"oracle-relift.checkpoint.json",{"complete":True,"attempted":len(rows),"accepted":passed,"interface_pass":passed>0})


def _teacher_config(root,config):
    c=json.loads(json.dumps(config));e11=json.loads((root/"configs/e11_mechanism_placement_transfer_v1.json").read_text());c["anchor_common_start_q"]=e11["anchor_common_start_q"];c["common_start_q"]={};return c


def _plan_program(plan,lib,config):
    active=np.asarray(plan["activity"],bool)
    if int(active[0])+int(np.count_nonzero(active[1:]&~active[:-1]))!=1:raise ValueError("not_single_ON")
    return encode_edge_sequence(json.loads(str(plan["sequence_json"])),lib,int(config["program"]["maximum_tokens"]))


def collect_train_val(root:Path,config:dict[str,Any],output:Path)->None:
    if not json.loads((output/"oracle-relift.checkpoint.json").read_text()).get("interface_pass"):raise RuntimeError("oracle interface gate did not pass")
    lib=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));bank=load_bank(root/"results/e11_mechanism_placement_transfer_v1");q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);scenes=[x for x in json.loads((root/config["inputs"]["pose_splits"]).read_text())["scenes"] if x["split"] in {"TRAIN","VALIDATION"}];c=_teacher_config(root,config);graphs=[];roots=[];candidates=[];labels=[];events=[];graph_dir=output/"teacher_graphs";plan_dir=output/"teacher_plans";graph_dir.mkdir(exist_ok=True);plan_dir.mkdir(exist_ok=True)
    root_port=int(bank["arc_start"][bank["routes"]["raster_u_phase_0.00/forward"][0]])
    for scene_order,scene0 in enumerate(scenes):
        sid=scene0["scene_id"];scene={**scene0,"source_anchor":scene0["anchor_id"],"rng_seed":2026091700+scene_order,"name":sid};began=perf_counter();q,root_rows=_root_audit(root,c,bank,scene);roots.extend(root_rows)
        if q is None:
            graphs.append({"scene_id":sid,"split":scene0["split"],"status":"start_search_failed","build_seconds":perf_counter()-began});continue
        c["common_start_q"][sid]=q.tolist();built=build_placement_graph(root,c,bank,{"candidate_id":sid,"placement_level":sid,"transform_base_from_surface":scene["transform_base_from_surface"],"rng_seed":scene["rng_seed"]},q2);path=graph_dir/f"hemisphere_{sid}.npz"
        if built["status"] in {"ready","recombination_limited"}:save_robot_graph(path,built)
        grow={k:v for k,v in built.items() if k not in {"nodes_q","node_ports","node_ranks","node_memberships","edges","edge_meta","witnesses","candidate_rows","attempt_rows"}};grow.update({"scene_id":sid,"split":scene0["split"],"status":built["status"],"graph_file":str(path),"graph_sha256":file_hash(path) if path.exists() else None,"build_seconds":perf_counter()-began,"root_q":q.tolist()});graphs.append(grow);write_json(output/"teacher_graph_manifest.partial.json",{"graphs":graphs});write_csv(output/"root_events.partial.csv",roots);print(json.dumps({"teacher_scene":sid,"status":built["status"],"nodes":grow.get("node_count"),"edges":grow.get("edge_count"),"ik_calls":grow.get("ik_calls"),"seconds":grow["build_seconds"]}),flush=True)
        if not path.exists():continue
        data=load_robot_graph(path);transform=np.asarray(scene["transform_base_from_surface"],dtype=np.float64);accepted_programs=set();validation_calls=0
        f_result,archive=fixed_route_initialize(data,int(data["start_node"]),1,c,wall_time=float(config["teacher"]["search_seconds"]),expanded_limit=int(config["teacher"]["expanded_limit"]));method_plans=[]
        if f_result.incumbent is not None:
            pf,_=save_plan(output,sid,1,"TEACHER_F",f_result.incumbent,f_result.incumbent.path,data);method_plans.append(("F",pf))
        screen=_Q3Screen(root,c,data,{"transform_base_from_surface":scene["transform_base_from_surface"]});p_result=greedy_prefix_completion(data=data,start_node=int(data["start_node"]),maximum_on_segments=1,initial_prefixes=(tuple(x["path"]) for x in archive),validated_fallback=None,config=c,wall_time_s=float(config["teacher"]["search_seconds"]),screen=screen)
        for i,candidate in enumerate(p_result.candidates):
            from diffusion_coverage.solvers.history_search import SearchLabel
            label=SearchLabel(candidate.node,candidate.covered,candidate.membership,candidate.repeat_error,candidate.joint_cost,candidate.used_on_segments,candidate.edge_ids);pf,_=save_plan(output,sid,1,f"TEACHER_P{i}",label,candidate.edge_ids,data);method_plans.append(("P",pf))
        def qualify(method_plans):
            nonlocal validation_calls
            for method,pf in method_plans:
                if validation_calls>=int(config["teacher"]["full_validation_calls"]):break
                plan=np.load(root/pf,allow_pickle=False);h=str(plan["witness_hash"]);validation_calls+=1;started=perf_counter();checked=validate_unique_witness(root,c,output,{"k":1},plan,data,{"transform_base_from_surface":scene["transform_base_from_surface"]},h);events.append({"scene_id":sid,"kind":"teacher_graph_validation","method":method,"witness_hash":h,"duration_s":perf_counter()-started,"status":checked["final"]["overall_status"],"cache_hit":False})
                if checked["final"]["overall_status"]!="accepted_under_E09_R1_refined_sampled_checks":continue
                try:program=_plan_program(plan,lib,config)
                except Exception as exc:candidates.append({"scene_id":sid,"method":method,"witness_hash":h,"graph_validation":checked["final"]["overall_status"],"program_status":f"failed:{exc}"});continue
                if program.content_hash in accepted_programs:continue
                robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));lift=lift_program(program,lib,transform,q,robot,config,start_surface_point=lib.ports[root_port]);candidate_status=lift.status;relift_status=None;relift_hash=None
                if lift.trace is not None and lift.decoded is not None:
                    checked2=_validate_trace(root,config,lift.trace,lift.decoded.surface_points,transform);relift_status=checked2["validation_status"];relift_hash=hashlib.sha256(lift.trace.q.tobytes()+lift.trace.u.tobytes()).hexdigest()
                candidates.append({"scene_id":sid,"split":scene0["split"],"method":method,"witness_hash":h,"graph_validation":checked["final"]["overall_status"],"program_hash":program.content_hash,"token_count":len(program.tokens),"candidate_lift":candidate_status,"candidate_validation":relift_status,"ik_calls":lift.ik_calls})
                if relift_status=="accepted_under_E12_refined_sampled_checks":accepted_programs.add(program.content_hash);labels.append({"scene_id":sid,"split":scene0["split"],"anchor_id":scene0["anchor_id"],"transform":scene0["transform_base_from_surface"],"q0":q.tolist(),"program_hash":program.content_hash,"program_json":program.to_json(),"source_method":method,"source_witness_hash":h,"relift_witness_hash":relift_hash})
                if len(accepted_programs)>=2:break
        qualify(method_plans)
        if not accepted_programs:
            screen=_Q3Screen(root,c,data,{"transform_base_from_surface":scene["transform_base_from_surface"]});a_result=structured_anytime_search(data=data,start_node=int(data["start_node"]),maximum_on_segments=1,initial_prefixes=(tuple(x["path"]) for x in archive),validated_fallback=None,config=c,wall_time_s=float(config["teacher"]["search_seconds"]),screen=screen,use_source_runs=False)
            extras=[]
            from diffusion_coverage.solvers.history_search import SearchLabel
            for i,candidate in enumerate(a_result.candidates):
                label=SearchLabel(candidate.node,candidate.covered,candidate.membership,candidate.repeat_error,candidate.joint_cost,candidate.used_on_segments,candidate.edge_ids);pf,_=save_plan(output,sid,1,f"TEACHER_A{i}",label,candidate.edge_ids,data);extras.append(("A",pf))
            qualify(extras)
        write_json(output/"collect-train-val.checkpoint.json",{"complete":False,"last_scene":sid,"graphs":len(graphs),"qualified_labels":len(labels)})
    write_json(output/"teacher_graph_manifest.json",{"graphs":graphs,"large_graphs_published":False});write_csv(output/"root_events.csv",roots);write_csv(output/"teacher_candidate_qualification.csv",candidates);write_json(output/"qualified_program_dataset.json",{"labels":labels});write_csv(output/"teacher_events.csv",events);write_json(output/"collect-train-val.checkpoint.json",{"complete":True,"attempted_tasks":len(scenes),"usable_graphs":sum(x.get("status") in {"ready","recombination_limited"} for x in graphs),"qualified_labels":len(labels),"qualified_tasks":len({x["scene_id"] for x in labels})})
