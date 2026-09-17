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

from diffusion_coverage.coverage.episode_summary import apply_edge_summary,initial_episode_state,summarize_ordered_membership
from diffusion_coverage.planning.e12_programs import GeometryProgram,ProgramToken,decode_program,encode_edge_sequence,load_geometry_library,sphere_to_stereographic
from diffusion_coverage.robot.e09_execution import evaluate_synchronized_fk_trace, sphere_episode_counts_indexed, sphere_membership_stream
from diffusion_coverage.robot.e12_program_execution import densify_program_trace, lift_program
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from e09r1_runner_support import quadrature
from e09r1_runner_support import build_placement_graph, load_bank, load_robot_graph, save_plan, save_robot_graph, validate_unique_witness
from e10_runner_support import fixed_route_initialize
from e11_runner_support import _Q3Screen, _root_audit
from diffusion_coverage.solvers.structured_routing import compose_episode_summaries,greedy_prefix_completion,structured_anytime_search,replay_edge_sequence


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
    # The inherited graph archive loader resolves its immutable geometry sibling by
    # historical filename.  These byte-identical aliases are not runtime robot data.
    shutil.copyfile(root/config["inputs"]["geometry_bank_npz"],output/"geometry_bank.npz")
    shutil.copyfile(root/config["inputs"]["geometry_bank_json"],output/"geometry_bank.json")
    splits=root/config["inputs"]["pose_splits"]
    shutil.copyfile(splits,output/"pose_splits.json")
    allowed={"surface_geometry":["geometry_only_library.npz","geometry_only_library.json"],"task_inputs":["transform_base_from_surface","checked_q0","robot_model","physical_contract"],"forbidden":["robot graph","per-port q","edge feasibility","teacher future q","teacher graph edge IDs"]}
    manifest={"experiment":config["experiment"],"prepared_at":__import__('datetime').datetime.now().astimezone().isoformat(),"code_sha":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),"config_sha256":file_hash(root/"configs/e12_graph_free_global_generation_v1.json"),"pose_splits_sha256":file_hash(splits),"geometry_npz_sha256":file_hash(output/"geometry_only_library.npz"),"geometry_json_sha256":file_hash(output/"geometry_only_library.json"),"geometry_semantic_hash":config["geometry_hash"],"allowed_runtime_dependencies":allowed,"gpu_recorded_at_training":None,"collision_scope":config["collision_scope"]}
    write_json(output/"manifest.json",manifest);write_json(output/"allowed_runtime_dependencies.json",allowed);write_json(output/"prepare.checkpoint.json",{"complete":True})


def repair_e11_diagnostics(root: Path, config: dict[str,Any], output: Path) -> None:
    from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
    from e11_runner_support import _fixed_suffix_diagnostic
    old=read_csv(root/"results/e11_mechanism_placement_transfer_v1/mechanism_diagnostics.csv")
    refs={x["scene_id"]:x for x in json.loads((root/"results/e11_mechanism_placement_transfer_v1/dev_graph_references.json").read_text())["graphs"]};lib=load_geometry_library(root/"results/e11_mechanism_placement_transfer_v1/geometry_bank.npz",root/"results/e11_mechanism_placement_transfer_v1/geometry_bank.json",radius=float(config["surface"]["radius_m"]));e11=json.loads((root/"configs/e11_mechanism_placement_transfer_v1.json").read_text());robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));rows=[]
    for row in old:
        sid=row["scene_id"];data=load_robot_graph(Path(refs[sid]["path"]));prefix_edges=tuple(int(x) for x in json.loads(row["prefix_edges_json"]));state=replay_edge_sequence(data["graph"],int(data["start_node"]),prefix_edges);last_edge=next((edge for edge in reversed(prefix_edges) if data["edge_meta"][edge]["kind"]=="source"),None);geom=None if last_edge is None else int(data["edge_meta"][last_edge]["geom_arc_id"]);family=None if geom is None else lib.families[int(lib.arc_family_index[geom])];direction=None if geom is None else ("forward" if bool(lib.arc_forward[geom]) else "reverse");route_name=None if family is None else f"{family}/{direction}";sequence=[] if route_name is None else list(data["routes"][route_name]);occurrences=[] if geom is None else [i for i,value in enumerate(sequence) if int(value)==geom];progress=None if not occurrences else occurrences[-1];diagnostic=_fixed_suffix_diagnostic(data,state,route_name,geom,e11)
        prefix=np.load(root/row["prefix_file"],allow_pickle=False);sigma=None
        if "q" in prefix.files and len(prefix["q"]):sigma=float(min(evaluate_task_kinematics_5d(robot,q,characteristic_length=float(config["robot"]["characteristic_length_m"])).sigma_min_5 for q in np.asarray(prefix["q"])))
        rows.append({**row,"legacy_fixed_route":row.get("fixed_route"),"corrected_prior_family":family,"corrected_direction":direction,"corrected_fixed_route":route_name,"corrected_progress_index":progress,**{f"corrected_{key}":value for key,value in diagnostic.items()},"correction_status":"recomputed_from_geom_arc_membership_and_executed_prefix","legacy_forward_field_missing":last_edge is not None and data["edge_meta"][last_edge].get("forward") is None,"prefix_sigma_measured":sigma,"prefix_sigma_status":"MEASURED_FROM_ARCHIVED_Q" if sigma is not None else "NULL_NOT_ARCHIVED","historical_main_cells_rerun":False})
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


def _validate_trace(root,config,trace,surface,transform,segment_ids=None):
    radius=float(config["surface"]["radius_m"]);robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));metrics={};checks={}
    q2_exact=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);membership=sphere_membership_stream(q2_exact["points"],surface,radius=radius,footprint_radius=float(config["coverage"]["footprint_radius_m"]));whole=summarize_ordered_membership(membership,q2_exact["weights"],active=trace.activity);composition_equal=True;composition_differing=0
    if segment_ids is not None:
        ids=np.asarray(segment_ids);boundaries=np.flatnonzero(ids[1:]!=ids[:-1])+1;starts=np.concatenate(([0],boundaries));ends=np.concatenate((boundaries+1,[len(ids)]));pieces=[summarize_ordered_membership(membership[:,lo:hi],q2_exact["weights"],active=trace.activity[lo:hi]) for lo,hi in zip(starts,ends,strict=True)];composed=compose_episode_summaries(pieces,q2_exact["weights"]);composition_differing=int(np.count_nonzero(composed.episode_counts!=whole.episode_counts));composition_equal=composition_differing==0 and np.array_equal(composed.footprint,whole.footprint)
    for temporal in ("T0","T1"):
        dense=densify_program_trace(trace,surface,transform,radius,joint_step=float(config["validation"]["temporal"][f"{temporal}_joint_step_rad"]),surface_step=float(config["validation"]["temporal"][f"{temporal}_surface_step_m"]));q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);check=evaluate_synchronized_fk_trace(robot,dense,transform,q2["points"][:1],q2["weights"][:1],sphere_radius=radius,footprint_radius=float(config["coverage"]["footprint_radius_m"]),characteristic_length=float(config["robot"]["characteristic_length_m"]));checks[temporal]=(dense,check)
    for temporal,name in (("T0","Q1"),("T0","Q2"),("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")):
        points,weights=quadrature(config,name,radius,root);dense,check=checks[temporal];counts=sphere_episode_counts_indexed(points,check.surface_points,dense.activity,radius=radius,footprint_radius=float(config["coverage"]["footprint_radius_m"]));total=float(weights.sum());metrics[f"E_miss_{temporal}_{name}"]=float(weights[counts==0].sum()/total);metrics[f"E_rep_{temporal}_{name}"]=float(np.dot(weights,np.maximum(counts-1,0))/total)
    t1=checks["T1"][1];required=[("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")];contract=all(metrics[f"E_miss_{t}_{q}"]<=float(config["coverage"]["missed_tolerance"])+1e-12 and metrics[f"E_rep_{t}_{q}"]<=float(config["coverage"]["repeat_tolerance"])+1e-12 for t,q in required);changes=max(max(abs(metrics[f"E_miss_T0_Q3"]-metrics[f"E_miss_T0_Q4"]),abs(metrics[f"E_rep_T0_Q3"]-metrics[f"E_rep_T0_Q4"])),max(abs(metrics[f"E_miss_T0_Q4"]-metrics[f"E_miss_T1_Q4"]),abs(metrics[f"E_rep_T0_Q4"]-metrics[f"E_rep_T1_Q4"])),max(abs(metrics[f"E_miss_T1_Q4"]-metrics[f"E_miss_T1_Q4a"]),abs(metrics[f"E_rep_T1_Q4"]-metrics[f"E_rep_T1_Q4a"])))
    motion=t1.max_position_error<=float(config["robot"]["position_tolerance_m"])+1e-12 and t1.max_axis_error<=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"]))+1e-12 and t1.min_sigma5>=float(config["robot"]["sigma_safe"])-1e-12 and t1.min_joint_margin>=-1e-12 and t1.collision_free
    status="accepted_under_E12_refined_sampled_checks" if motion and contract and changes<=float(config["coverage"]["resolution_tolerance"])+1e-12 else ("motion_contract_failed" if not motion else ("coverage_contract_failed" if contract is False and changes<=float(config["coverage"]["resolution_tolerance"])+1e-12 else "numerically_unresolved"))
    if not composition_equal:status="same_sample_composition_failed"
    return {"validation_status":status,"same_sample_composition_equal":composition_equal,"same_sample_differing_units":composition_differing,"max_resolution_change":changes,"min_sigma5":t1.min_sigma5,"max_position_error_m":t1.max_position_error,"max_axis_error_deg":float(np.rad2deg(t1.max_axis_error)),"min_joint_margin":t1.min_joint_margin,"collision_free":t1.collision_free,"J_q":float(np.linalg.norm(np.diff(trace.q,axis=0),axis=1).sum()),**metrics}


def oracle_relift(root: Path,config:dict[str,Any],output:Path)->None:
    lib=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));programs=json.loads((output/"encoded_programs.json").read_text());scenes=scene_map(root);rows=[];events=[];directory=output/"oracle_relift_witnesses";directory.mkdir(exist_ok=True)
    for order,(witness_hash,item) in enumerate(sorted(programs.items())):
        scene_id=item["scene_id"];scene=scenes[scene_id];transform=np.asarray(scene["transform_base_from_surface"],dtype=np.float64);robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));program=GeometryProgram.from_json(item["program_json"]);began=perf_counter();result=lift_program(program,lib,transform,np.asarray(item["q0"]),robot,config,start_surface_point=np.asarray(item["start_surface_point"]));events.append({"event_id":len(events),"kind":"oracle_lift","witness_hash":witness_hash,"scene_id":scene_id,"started_order":order,"duration_s":perf_counter()-began,"status":result.status,"ik_calls":result.ik_calls,"failure_reason":result.failure_reason,"failure_index":result.failure_index})
        row={"teacher_witness_hash":witness_hash,"scene_id":scene_id,"program_hash":program.content_hash,"lift_status":result.status,"ik_calls":result.ik_calls,"spacing_m":result.spacing_m,"halvings":result.halvings,"lift_seconds":result.elapsed_s,"failure_reason":result.failure_reason,"failure_index":result.failure_index}
        if result.trace is not None and result.decoded is not None:
            trace=result.trace;surface=result.decoded.surface_points;h=hashlib.sha256(trace.q.tobytes()+trace.u.tobytes()+trace.activity.tobytes()).hexdigest();path=directory/f"{h}.npz";np.savez_compressed(path,q=trace.q,u=trace.u,target_position=trace.target_position,target_axis=trace.target_axis,activity=trace.activity,surface_points=surface,program_json=np.asarray(program.to_json()),witness_hash=np.asarray(h));checked=_validate_trace(root,config,trace,surface,transform,result.decoded.segment_ids);row.update({"relift_witness_hash":h,"relift_file":str(path.relative_to(root)),**checked})
        rows.append(row);write_csv(output/"oracle_relift.partial.csv",rows)
    write_csv(output/"oracle_relift.csv",rows);write_csv(output/"execution_events.csv",events);passed=sum(x.get("validation_status")=="accepted_under_E12_refined_sampled_checks" for x in rows);write_json(output/"oracle-relift.checkpoint.json",{"complete":True,"attempted":len(rows),"accepted":passed,"interface_pass":passed>0})


def _teacher_config(root,config):
    c=json.loads(json.dumps(config));e11=json.loads((root/"configs/e11_mechanism_placement_transfer_v1.json").read_text());c["anchor_common_start_q"]=e11["anchor_common_start_q"];c["common_start_q"]={};return c


def _plan_program(plan,lib,config):
    active=np.asarray(plan["activity"],bool)
    if int(active[0])+int(np.count_nonzero(active[1:]&~active[:-1]))!=1:raise ValueError("not_single_ON")
    return encode_edge_sequence(json.loads(str(plan["sequence_json"])),lib,int(config["program"]["maximum_tokens"]))


def collect_train_val(root:Path,config:dict[str,Any],output:Path,scene_ids:set[str]|None=None)->None:
    if not json.loads((output/"oracle-relift.checkpoint.json").read_text()).get("interface_pass"):raise RuntimeError("oracle interface gate did not pass")
    lib=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));bank=load_bank(root/"results/e11_mechanism_placement_transfer_v1");q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);all_scenes=[x for x in json.loads((root/config["inputs"]["pose_splits"]).read_text())["scenes"] if x["split"] in {"TRAIN","VALIDATION"}];scenes=[(i,x) for i,x in enumerate(all_scenes) if scene_ids is None or x["scene_id"] in scene_ids];c=_teacher_config(root,config);existing=json.loads((output/"teacher_graph_manifest.partial.json").read_text()).get("graphs",[]) if (output/"teacher_graph_manifest.partial.json").exists() else [];graphs=[];roots=read_csv(output/"root_events.partial.csv") if (output/"root_events.partial.csv").exists() else [];candidates=read_csv(output/"teacher_candidate_qualification.partial.csv") if (output/"teacher_candidate_qualification.partial.csv").exists() else [];labels=json.loads((output/"qualified_program_dataset.partial.json").read_text()).get("labels",[]) if (output/"qualified_program_dataset.partial.json").exists() else [];events=read_csv(output/"teacher_events.partial.csv") if (output/"teacher_events.partial.csv").exists() else [];checkpoint_state=json.loads((output/"collect-train-val.checkpoint.json").read_text()) if (output/"collect-train-val.checkpoint.json").exists() else {};completed_scenes=set(checkpoint_state.get("completed_scenes",[]));graph_dir=output/"teacher_graphs";plan_dir=output/"teacher_plans";graph_dir.mkdir(exist_ok=True);plan_dir.mkdir(exist_ok=True)
    root_port=int(bank["arc_start"][bank["routes"]["raster_u_phase_0.00/forward"][0]])
    for scene_order,scene0 in scenes:
        sid=scene0["scene_id"];scene={**scene0,"source_anchor":scene0["anchor_id"],"rng_seed":2026091700+scene_order,"name":sid};began=perf_counter();path=graph_dir/f"hemisphere_{sid}.npz";prior=next((x for x in existing if x.get("scene_id")==sid and path.exists() and x.get("graph_sha256")==file_hash(path)),None)
        if prior is not None:
            q=np.asarray(prior["root_q"],dtype=np.float64);grow=prior;graphs.append(grow);print(json.dumps({"teacher_scene":sid,"status":"resumed_graph","nodes":grow.get("node_count"),"edges":grow.get("edge_count"),"ik_calls":grow.get("ik_calls"),"seconds":0.0}),flush=True)
        else:
            q,root_rows=_root_audit(root,c,bank,scene);roots.extend(root_rows)
            if q is None:
                graphs.append({"scene_id":sid,"split":scene0["split"],"status":"start_search_failed","build_seconds":perf_counter()-began});continue
            c["common_start_q"][sid]=q.tolist();built=build_placement_graph(root,c,bank,{"candidate_id":sid,"placement_level":sid,"transform_base_from_surface":scene["transform_base_from_surface"],"rng_seed":scene["rng_seed"]},q2)
            if built["status"] in {"ready","recombination_limited"}:save_robot_graph(path,built)
            grow={k:v for k,v in built.items() if k not in {"nodes_q","node_ports","node_ranks","node_memberships","edges","edge_meta","witnesses","candidate_rows","attempt_rows"}};grow.update({"scene_id":sid,"split":scene0["split"],"status":built["status"],"graph_file":str(path),"graph_sha256":file_hash(path) if path.exists() else None,"build_seconds":perf_counter()-began,"root_q":q.tolist()});graphs.append(grow);write_json(output/"teacher_graph_manifest.partial.json",{"graphs":graphs});write_csv(output/"root_events.partial.csv",roots);print(json.dumps({"teacher_scene":sid,"status":built["status"],"nodes":grow.get("node_count"),"edges":grow.get("edge_count"),"ik_calls":grow.get("ik_calls"),"seconds":grow["build_seconds"]}),flush=True)
        if not path.exists():continue
        if sid in completed_scenes:
            print(json.dumps({"teacher_scene":sid,"status":"resumed_complete","qualified_labels":sum(x["scene_id"]==sid for x in labels)}),flush=True);continue
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
                plan=np.load(output.parents[1]/pf,allow_pickle=False);h=str(plan["witness_hash"]);validation_calls+=1;started=perf_counter();checked=validate_unique_witness(root,c,output,{"k":1},plan,data,{"transform_base_from_surface":scene["transform_base_from_surface"]},h);events.append({"scene_id":sid,"kind":"teacher_graph_validation","method":method,"witness_hash":h,"duration_s":perf_counter()-started,"status":checked["final"]["overall_status"],"cache_hit":False})
                if checked["final"]["overall_status"]!="accepted_under_E09_R1_refined_sampled_checks":continue
                try:program=_plan_program(plan,lib,config)
                except Exception as exc:candidates.append({"scene_id":sid,"method":method,"witness_hash":h,"graph_validation":checked["final"]["overall_status"],"program_status":f"failed:{exc}"});continue
                if program.content_hash in accepted_programs:continue
                robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));lift=lift_program(program,lib,transform,q,robot,config,start_surface_point=lib.ports[root_port]);candidate_status=lift.status;relift_status=None;relift_hash=None
                if lift.trace is not None and lift.decoded is not None:
                    checked2=_validate_trace(root,config,lift.trace,lift.decoded.surface_points,transform,lift.decoded.segment_ids);relift_status=checked2["validation_status"];relift_hash=hashlib.sha256(lift.trace.q.tobytes()+lift.trace.u.tobytes()).hexdigest()
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
        completed_scenes.add(sid);write_csv(output/"teacher_candidate_qualification.partial.csv",candidates);write_json(output/"qualified_program_dataset.partial.json",{"labels":labels});write_csv(output/"teacher_events.partial.csv",events);write_json(output/"collect-train-val.checkpoint.json",{"complete":False,"last_scene":sid,"completed_scenes":sorted(completed_scenes),"graphs":len(graphs),"qualified_labels":len(labels)})
    write_json(output/"teacher_graph_manifest.json",{"graphs":graphs,"large_graphs_published":False});write_csv(output/"root_events.csv",roots);write_csv(output/"teacher_candidate_qualification.csv",candidates);write_json(output/"qualified_program_dataset.json",{"labels":labels});write_csv(output/"teacher_events.csv",events);write_json(output/"collect-train-val.checkpoint.json",{"complete":True,"attempted_tasks":len(scenes),"usable_graphs":sum(x.get("status") in {"ready","recombination_limited"} for x in graphs),"qualified_labels":len(labels),"qualified_tasks":len({x["scene_id"] for x in labels})})


def merge_teacher_shards(root:Path,config:dict[str,Any],output:Path)->None:
    shard_root=output/"teacher_shards";graphs=[];roots=[];candidates=[];labels=[];events=[];worker_rows=[]
    for shard in sorted(x for x in shard_root.iterdir() if x.is_dir()):
        checkpoint=json.loads((shard/"collect-train-val.checkpoint.json").read_text())
        if not checkpoint.get("complete"):raise RuntimeError(f"incomplete teacher shard {shard}")
        graphs.extend(json.loads((shard/"teacher_graph_manifest.json").read_text())["graphs"]);roots.extend(read_csv(shard/"root_events.csv"));candidates.extend(read_csv(shard/"teacher_candidate_qualification.csv"));labels.extend(json.loads((shard/"qualified_program_dataset.json").read_text())["labels"]);events.extend(read_csv(shard/"teacher_events.csv"));worker_rows.append({"shard":shard.name,"attempted_tasks":checkpoint["attempted_tasks"],"qualified_labels":checkpoint["qualified_labels"],"qualified_tasks":checkpoint["qualified_tasks"]})
    scene_ids=[x["scene_id"] for x in graphs]
    if len(scene_ids)!=24 or len(set(scene_ids))!=24:raise RuntimeError(f"teacher shards contain {len(scene_ids)} rows / {len(set(scene_ids))} unique scenes")
    graphs.sort(key=lambda x:x["scene_id"]);labels.sort(key=lambda x:(x["scene_id"],x["program_hash"]));write_json(output/"teacher_graph_manifest.json",{"graphs":graphs,"large_graphs_published":False,"parallel_workers":len(worker_rows),"measured_single_worker_peak_rss_bytes":5184572*1024});write_csv(output/"root_events.csv",roots);write_csv(output/"teacher_candidate_qualification.csv",candidates);write_json(output/"qualified_program_dataset.json",{"labels":labels});write_csv(output/"teacher_events.csv",events);write_csv(output/"teacher_worker_summary.csv",worker_rows);write_json(output/"collect-train-val.checkpoint.json",{"complete":True,"attempted_tasks":24,"usable_graphs":sum(x.get("status") in {"ready","recombination_limited"} for x in graphs),"qualified_labels":len(labels),"qualified_tasks":len({x["scene_id"] for x in labels}),"parallel_workers":len(worker_rows)})


def _rotation_log_angle(matrix:np.ndarray)->float:
    return float(np.arccos(np.clip((np.trace(matrix)-1.)/2.,-1.,1.)))


def graph_free_roots(root:Path,config:dict[str,Any],output:Path,split:str)->list[dict[str,Any]]:
    target=output/f"graph_free_roots_{split.lower()}.json"
    if target.exists():return json.loads(target.read_text())["roots"]
    scenes=[x for x in json.loads((root/config["inputs"]["pose_splits"]).read_text())["scenes"] if x["split"]==split];c=_teacher_config(root,config);bank=load_bank(root/"results/e11_mechanism_placement_transfer_v1");roots=[];events=[]
    teacher={x["scene_id"]:x for x in json.loads((output/"teacher_graph_manifest.json").read_text()).get("graphs",[])} if (output/"teacher_graph_manifest.json").exists() else {}
    for scene in scenes:
        sid=scene["scene_id"]
        if split=="VALIDATION" and sid in teacher and teacher[sid].get("root_q") is not None:
            q=np.asarray(teacher[sid]["root_q"],dtype=np.float64);rows=[{"scene_id":sid,"kind":"root_cache","status":"CACHE_HIT_teacher_graph_root"}];root_seconds=None
        else:
            prepared={**scene,"source_anchor":scene["anchor_id"],"rng_seed":2026091700+int(scene["scene_index"]),"name":sid};started=perf_counter();q,rows=_root_audit(root,c,bank,prepared);root_seconds=perf_counter()-started
        events.extend(rows);roots.append({"scene_id":sid,"split":split,"status":"admitted" if q is not None else "start_search_failed","q0":None if q is None else q.tolist(),"transform":scene["transform_base_from_surface"],"transform_sha256":scene["transform_sha256"],"anchor_id":scene["anchor_id"],"root_seconds":root_seconds})
        write_json(target,{"roots":roots});write_csv(output/f"root_events_{split.lower()}.csv",events)
    return roots


def _retrieval_programs(labels:list[dict[str,Any]],transform:np.ndarray,q0:np.ndarray)->list[dict[str,Any]]:
    ranked=[]
    for item in labels:
        source=np.asarray(item["transform"],dtype=np.float64);translation=float(np.sum(((transform[:3,3]-source[:3,3])/.05)**2));rotation=(_rotation_log_angle(source[:3,:3].T@transform[:3,:3])/np.deg2rad(20.))**2;joint=float(np.sum(((q0[:5]-np.asarray(item["q0"])[:5])/(np.pi/2))**2));ranked.append((translation+rotation+joint,item["scene_id"],item["program_hash"],item))
    ranked.sort(key=lambda x:(x[0],x[1],x[2]));selected=[];seen_programs=set();seen_tasks=set()
    for distinct_tasks in (True,False):
        for distance,task,program_hash,item in ranked:
            if program_hash in seen_programs or (distinct_tasks and task in seen_tasks):continue
            selected.append({"program_json":item["program_json"],"source_scene_id":task,"source_program_hash":program_hash,"retrieval_distance":distance});seen_programs.add(program_hash);seen_tasks.add(task)
            if len(selected)>=8:return selected
    return selected


def freeze_generated_candidates(root:Path,config:dict[str,Any],output:Path,split:str)->None:
    import torch
    from diffusion_coverage.learning.e12_inference import continuous_program,generate_symbol_slots,load_models,normalized_condition
    if not (output/"train.checkpoint.json").exists():raise RuntimeError("trained checkpoints are not frozen")
    roots=graph_free_roots(root,config,output,split);labels=json.loads((output/"qualified_program_dataset.json").read_text())["labels"];training=[x for x in labels if x["split"]=="TRAIN"]
    metadata=json.loads((output/"dataset_normalization_and_prototypes.json").read_text());library=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");models=load_models(output/"models",device);decoder=models["categorical"][0];rows=[]
    for root_row in roots:
        sid=root_row["scene_id"]
        if root_row["status"]!="admitted":
            for method in ("RETRIEVE","REG","FM"):rows.append({"scene_id":sid,"split":split,"method":method,"slot":None,"status":"NOT_RUN_root_failed"})
            continue
        transform=np.asarray(root_row["transform"],float);q0=np.asarray(root_row["q0"],float);condition=normalized_condition(transform,q0,metadata,device)
        for slot,item in enumerate(_retrieval_programs(training,transform,q0)):
            rows.append({"scene_id":sid,"split":split,"method":"RETRIEVE","slot":slot,"status":"frozen","program_json":item["program_json"],**{k:v for k,v in item.items() if k!="program_json"}})
        sequences=generate_symbol_slots(decoder,condition,seed=202609170000+int(next(x["scene_index"] for x in json.loads((root/config["inputs"]["pose_splits"]).read_text())["scenes"] if x["scene_id"]==sid))*100)
        for slot,symbols in enumerate(sequences):
            for method in ("REG","FM"):
                program,detail=continuous_program(method,models[method.lower()][0],symbols,condition,metadata,library,seed=202609170000+slot,maximum_tokens=int(config["program"]["maximum_tokens"]),fm_steps=int(config["learning"]["fm_ode_steps"]));rows.append({"scene_id":sid,"split":split,"method":method,"slot":slot,"status":"frozen" if program is not None else "syntactic_rejection","program_json":None if program is None else program.to_json(),"program_hash":None if program is None else program.content_hash,**detail})
    path=output/f"{split.lower()}_candidate_programs.json";write_json(path,{"split":split,"generated_before_evaluation":True,"device":str(device),"rows":rows});write_json(output/f"freeze-{split.lower()}-candidates.checkpoint.json",{"complete":True,"rows":len(rows),"sha256":file_hash(path)})


def _apply_cached_lazy_motion(state:dict[str,Any],motion:dict[str,Any],weights:np.ndarray)->dict[str,Any]:
    episode=apply_edge_summary(state["episode"],motion["summary"],weights)
    return {"q":motion["q"].copy(),"port":motion["port"],"episode":episode,"J_q":state["J_q"]+motion["cost"],"sequence":state["sequence"]+(motion["edge"],),"visited":state["visited"].copy()}


def run_p_lazy(root:Path,config:dict[str,Any],output:Path)->None:
    """On-demand geometry-arc continuation without loading any robot graph."""
    split="SEALED_TEST";frozen_path=output/"sealed_test_candidate_programs.json";frozen=json.loads(frozen_path.read_text());roots={x["scene_id"]:x for x in graph_free_roots(root,config,output,split)};library=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);sample_points=np.asarray(q2["points"]);weights=np.asarray(q2["weights"]);total=float(weights.sum());radius=float(config["surface"]["radius_m"]);footprint=float(config["coverage"]["footprint_radius_m"]);required=(1.-float(config["coverage"]["missed_tolerance"]))*total;rows=[];events=[]
    outgoing={port:[] for port in range(len(library.ports))}
    for arc_id,(start,kind) in enumerate(zip(library.arc_start,library.arc_kind,strict=True)):
        if str(kind) in {"source","cross_port"}:outgoing[int(start)].append(int(arc_id))
    for values in outgoing.values():values.sort()

    def arc_program(arc_id:int)->GeometryProgram:
        if str(library.arc_kind[arc_id])=="source":
            family,direction,a,b=library.arc_intervals[arc_id];return GeometryProgram((ProgramToken("SCAN",family,direction,min(a,b),max(a,b)),ProgramToken("END")))
        d1,d2=sphere_to_stereographic(library.ports[int(library.arc_end[arc_id])],library.radius);return GeometryProgram((ProgramToken("VIA",d1=d1,d2=d2),ProgramToken("END")))

    def edge_record(arc_id:int)->dict[str,Any]:
        return {"kind":str(library.arc_kind[arc_id]),"geom_arc_id":arc_id,"start_port":int(library.arc_start[arc_id]),"end_port":int(library.arc_end[arc_id])}

    for sid,root_row in sorted(roots.items()):
        began=perf_counter();deadline=began+float(config["deployment"]["p_lazy_query_deadline_s"]);call_limit=int(config["deployment"]["p_lazy_max_ik_calls"]);calls=0;cache={};complete=[];prefixes=[]
        if root_row["status"]!="admitted":rows.append({"scene_id":sid,"status":"NOT_RUN_root_failed"});continue
        transform=np.asarray(root_row["transform"],float);q0=np.asarray(root_row["q0"],float);root_port=int(library.arc_start[library.routes["raster_u_phase_0.00/forward"][0]]);initial_membership=sphere_membership_stream(sample_points,library.ports[root_port][None,:],radius=radius,footprint_radius=footprint)[:,0];initial=initial_episode_state(initial_membership);robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]))

        def attempt(state:dict[str,Any],arc_id:int):
            nonlocal calls
            key=hashlib.sha256(state["q"].tobytes()+np.asarray([arc_id],np.int64).tobytes()).hexdigest()
            if key in cache:
                events.append({"scene_id":sid,"kind":"P_lazy_arc_query","arc_id":arc_id,"start_port":state["port"],"end_port":int(library.arc_end[arc_id]),"ik_calls":0,"duration_s":0.,"status":"CACHE_HIT_valid" if cache[key] else "CACHE_HIT_failed","failure_reason":None,"cache_hit":True})
                motion=cache[key]
                if not motion:return False
                return _apply_cached_lazy_motion(state,motion,weights)
            if calls>=call_limit or perf_counter()>=deadline:return None
            local=json.loads(json.dumps(config));local["lifter"]["max_ik_calls"]=min(int(local["lifter"]["max_ik_calls"]),call_limit-calls);local["lifter"]["deadline_s"]=min(float(local["lifter"]["deadline_s"]),max(0.,deadline-perf_counter()));program=arc_program(arc_id);started=perf_counter();lift=lift_program(program,library,transform,state["q"],robot,local,start_surface_point=library.ports[state["port"]],allow_via_only=str(library.arc_kind[arc_id])=="cross_port");calls+=lift.ik_calls;event={"scene_id":sid,"kind":"P_lazy_arc_query","arc_id":arc_id,"start_port":state["port"],"end_port":int(library.arc_end[arc_id]),"ik_calls":lift.ik_calls,"duration_s":perf_counter()-started,"status":lift.status,"failure_reason":lift.failure_reason,"cache_hit":False};events.append(event)
            if lift.trace is None or lift.decoded is None:cache[key]=False;return False
            membership=sphere_membership_stream(sample_points,lift.decoded.surface_points,radius=radius,footprint_radius=footprint);summary=summarize_ordered_membership(membership,weights);cost=float(np.linalg.norm(np.diff(lift.trace.q,axis=0),axis=1).sum());edge=edge_record(arc_id);motion={"q":lift.trace.q[-1].copy(),"port":int(library.arc_end[arc_id]),"summary":summary,"cost":cost,"edge":edge};cache[key]=motion;return _apply_cached_lazy_motion(state,motion,weights)

        root_state={"q":q0,"port":root_port,"episode":initial,"J_q":0.,"sequence":tuple(),"visited":{}}
        for route_name in sorted(library.routes):
            state={**root_state,"visited":{}};route_prefix=[]
            for arc_id in library.routes[route_name]:
                child=attempt(state,int(arc_id))
                if not child:break
                state=child;route_prefix.append(state);covered=float(weights[state["episode"].covered].sum());progress=min(9,int(10*covered/total));prefixes.append({**state,"route":route_name,"progress":progress})
                if covered>=required-1e-15 and state["episode"].repeat_error<=float(config["coverage"]["repeat_tolerance"])+1e-12:
                    complete.append(state);break
        selected=[]
        for route_name in sorted(library.routes):
            for progress in range(10):
                choices=[x for x in prefixes if x["route"]==route_name and x["progress"]==progress]
                if choices:
                    low_j=min(choices,key=lambda x:(x["J_q"],x["episode"].repeat_error,len(x["sequence"])));low_r=min(choices,key=lambda x:(x["episode"].repeat_error,x["J_q"],len(x["sequence"])));selected.append(low_j)
                    if low_r["sequence"]!=low_j["sequence"]:selected.append(low_r)
        rollouts=[root_state]+selected;active=[True]*len(rollouts)
        while any(active) and calls<call_limit and perf_counter()<deadline and len(complete)<8:
            for index,state in enumerate(rollouts):
                if not active[index]:continue
                alternatives=[]
                for arc_id in outgoing[state["port"]]:
                    child=attempt(state,arc_id)
                    if not child:continue
                    if child["episode"].repeat_error>float(config["coverage"]["repeat_tolerance"])+1e-12:continue
                    key=(child["port"],child["episode"].covered.tobytes(),child["episode"].membership.tobytes())
                    prior=state["visited"].get(key)
                    if prior is not None and prior[0]<=child["episode"].repeat_error+1e-12 and prior[1]<=child["J_q"]+1e-12:continue
                    covered=float(weights[child["episode"].covered].sum());alternatives.append((max(0.,required-covered),child["episode"].repeat_error,child["J_q"],arc_id,key,child))
                if not alternatives:active[index]=False;continue
                _,_,_,_,key,chosen=min(alternatives,key=lambda x:x[:4]);chosen["visited"][key]=(chosen["episode"].repeat_error,chosen["J_q"]);rollouts[index]=chosen;covered=float(weights[chosen["episode"].covered].sum())
                if covered>=required-1e-15:
                    complete.append(chosen);active[index]=False
                if calls>=call_limit or perf_counter()>=deadline:break
        unique=set()
        for slot,state in enumerate(sorted(complete,key=lambda x:(x["episode"].repeat_error,x["J_q"],len(x["sequence"])))):
            try:program=encode_edge_sequence(list(state["sequence"]),library,int(config["program"]["maximum_tokens"]))
            except Exception as exc:rows.append({"scene_id":sid,"status":f"candidate_encoding_failed:{exc}"});continue
            if program.content_hash in unique:continue
            unique.add(program.content_hash);frozen["rows"].append({"scene_id":sid,"split":split,"method":"P_LAZY","slot":len(unique)-1,"status":"frozen","program_json":program.to_json(),"program_hash":program.content_hash,"Q2_miss":float(weights[~state["episode"].covered].sum()/total),"Q2_repeat":state["episode"].repeat_error,"query_J_q":state["J_q"]})
            if len(unique)>=8:break
        rows.append({"scene_id":sid,"status":"complete" if perf_counter()<deadline and calls<call_limit else "query_budget_limited","ik_calls":calls,"query_seconds":perf_counter()-began,"prefixes":len(prefixes),"rollouts":len(rollouts),"complete_candidates":len(unique),"cache_entries":len(cache)})
        write_json(frozen_path,frozen);write_csv(output/"p_lazy_query_results.partial.csv",rows);write_csv(output/"p_lazy_query_events.partial.csv",events)
    write_json(frozen_path,frozen);write_csv(output/"p_lazy_query_results.csv",rows);write_csv(output/"p_lazy_query_events.csv",events);write_json(output/"p-lazy.checkpoint.json",{"complete":True,"scenes":len(rows),"candidate_rows":sum(x.get('method')=='P_LAZY' for x in frozen['rows']),"frozen_candidates_sha256":file_hash(frozen_path)})


def evaluate_frozen_candidates(root:Path,config:dict[str,Any],output:Path,split:str)->None:
    frozen=json.loads((output/f"{split.lower()}_candidate_programs.json").read_text());roots={x["scene_id"]:x for x in graph_free_roots(root,config,output,split)};library=load_geometry_library(output/"geometry_only_library.npz",output/"geometry_only_library.json",radius=float(config["surface"]["radius_m"]));bank=load_bank(root/"results/e11_mechanism_placement_transfer_v1");root_port=int(bank["arc_start"][bank["routes"]["raster_u_phase_0.00/forward"][0]]);rows=[];events=[];accepted_dir=output/"graph_free_accepted_witnesses";accepted_dir.mkdir(exist_ok=True);contract_hash=file_hash(root/"configs/e12_graph_free_global_generation_v1.json")
    for sid in sorted(roots):
        root_row=roots[sid]
        methods=("RETRIEVE","REG","FM","P_LAZY") if split=="SEALED_TEST" else ("RETRIEVE","REG","FM")
        for method in methods:
            cell=[x for x in frozen["rows"] if x["scene_id"]==sid and x["method"]==method];began=perf_counter();root_charge=float(root_row.get("root_seconds") or 0.);deadline=began+max(0.,float(config["deployment"]["cell_deadline_s"])-root_charge);cache={};best=None;first=None;attempted=0
            if root_row["status"]!="admitted":
                rows.append({"scene_id":sid,"split":split,"method":method,"overall_status":"NOT_RUN_root_failed","candidate_slots":len(cell),"evaluated_slots":0});continue
            q0=np.asarray(root_row["q0"],float);transform=np.asarray(root_row["transform"],float)
            ordered_cell=sorted(cell,key=lambda x:(-1 if x.get("slot") is None else int(x["slot"])));halt_at=None
            for item_index,item in enumerate(ordered_cell):
                slot=item.get("slot");started=perf_counter();program_text=item.get("program_json");event={"event_id":len(events),"scene_id":sid,"split":split,"method":method,"slot":slot,"kind":"candidate_lift_and_full_validation","started_since_cell_s":started-began,"program_hash":item.get("program_hash")}
                if perf_counter()>=deadline:event.update({"status":"NOT_RUN_cell_deadline","duration_s":0.});events.append(event);continue
                if not program_text:event.update({"status":item.get("status","syntactic_rejection"),"duration_s":0.});events.append(event);attempted+=1;continue
                program=GeometryProgram.from_json(program_text);key=hashlib.sha256((sid+program.content_hash+contract_hash).encode()).hexdigest()
                if key in cache:
                    result=cache[key].copy();event.update({"status":result["status"],"cache_hit":True,"duration_s":perf_counter()-started});events.append(event);rowslot={**item,**result,"cache_hit":True};rows.append(rowslot);attempted+=1
                    if result.get("validation_status")=="accepted_under_E12_refined_sampled_checks":halt_at=item_index
                    if halt_at is not None:break
                    continue
                local=json.loads(json.dumps(config));local["lifter"]["deadline_s"]=max(0.,min(float(config["lifter"]["deadline_s"]),deadline-perf_counter()));robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));lift=lift_program(program,library,transform,q0,robot,local,start_surface_point=library.ports[root_port]);result={"status":lift.status,"lift_status":lift.status,"ik_calls":lift.ik_calls,"lift_seconds":lift.elapsed_s,"spacing_m":lift.spacing_m,"failure_reason":lift.failure_reason,"validation_status":None,"witness_hash":None}
                if lift.trace is not None and lift.decoded is not None and perf_counter()<deadline:
                    checked=_validate_trace(root,config,lift.trace,lift.decoded.surface_points,transform,lift.decoded.segment_ids);h=hashlib.sha256(lift.trace.q.tobytes()+lift.trace.u.tobytes()+lift.trace.activity.tobytes()).hexdigest();result.update({"status":checked["validation_status"],"validation_status":checked["validation_status"],"witness_hash":h,**checked})
                    if perf_counter()>deadline:result.update({"status":"validation_budget_limited","validation_status":"validation_budget_limited"})
                    if checked["validation_status"]=="accepted_under_E12_refined_sampled_checks":
                        path=accepted_dir/f"{h}.npz"
                        if not path.exists():np.savez_compressed(path,q=lift.trace.q,u=lift.trace.u,target_position=lift.trace.target_position,target_axis=lift.trace.target_axis,activity=lift.trace.activity,surface_points=lift.decoded.surface_points,program_json=np.asarray(program.to_json()),witness_hash=np.asarray(h))
                        objective=(int(lift.trace.activity[0])+int(np.count_nonzero(lift.trace.activity[1:]&~lift.trace.activity[:-1]))-1,float(checked["J_q"]));result["on_segments_minus_one"]=objective[0];best=result if best is None or objective<(best["on_segments_minus_one"],best["J_q"]) else best
                        if first is None:first=perf_counter()-began
                cache[key]=result.copy();event.update({"status":result["status"],"cache_hit":False,"duration_s":perf_counter()-started,"ik_calls":lift.ik_calls,"witness_hash":result.get("witness_hash")});events.append(event);rows.append({**item,**result,"cache_hit":False});attempted+=1
                if result.get("validation_status")=="accepted_under_E12_refined_sampled_checks":halt_at=item_index;break
            if halt_at is not None:
                for item in ordered_cell[halt_at+1:]:rows.append({**item,"status":"NOT_RUN_after_first_accepted","validation_status":"NOT_RUN"})
            summaries=[x for x in rows if x.get("scene_id")==sid and x.get("method")==method and x.get("split")==split];accepted=[x for x in summaries if x.get("validation_status")=="accepted_under_E12_refined_sampled_checks"]
            bestrow=min(accepted,key=lambda x:(int(x.get("on_segments_minus_one",0)),float(x["J_q"]))) if accepted else None
            first_slot=None if not accepted else min(int(x["slot"]) for x in accepted if x.get("slot") not in (None,""));fixed_k={}
            for kslot in (1,4,8):fixed_k[f"success_at_K{kslot}"]="True" if first_slot is not None and first_slot<kslot else ("False" if attempted>=min(kslot,len(cell)) else "CENSORED")
            rows.append({"row_type":"cell_summary","scene_id":sid,"split":split,"method":method,"overall_status":"accepted_under_E12_refined_sampled_checks" if bestrow else ("time_budget_limited" if perf_counter()>=deadline else "no_accepted_candidate"),"candidate_slots":len(cell),"evaluated_slots":attempted,"unique_candidates":len(cache),"first_accepted_s":first,"first_accepted_slot":first_slot,"best_witness_hash":None if bestrow is None else bestrow["witness_hash"],"best_J_q":None if bestrow is None else bestrow["J_q"],"root_seconds_charged":root_charge,"cell_seconds":perf_counter()-began,"cold_seconds":root_charge+perf_counter()-began,**fixed_k})
            write_csv(output/f"{split.lower()}_method_outcomes.partial.csv",rows);write_csv(output/f"{split.lower()}_execution_events.partial.csv",events)
    write_csv(output/f"{split.lower()}_method_outcomes.csv",rows);write_csv(output/f"{split.lower()}_execution_events.csv",events);write_json(output/f"evaluate-{split.lower()}.checkpoint.json",{"complete":True,"cells":len([x for x in rows if x.get('row_type')=='cell_summary']),"accepted_cells":sum(x.get('overall_status')=='accepted_under_E12_refined_sampled_checks' for x in rows if x.get('row_type')=='cell_summary')})


def test_graph_reference(root:Path,config:dict[str,Any],output:Path)->None:
    if not (output/"evaluate-sealed_test.checkpoint.json").exists():raise RuntimeError("graph-free sealed outputs must be frozen first")
    roots={x["scene_id"]:x for x in graph_free_roots(root,config,output,"SEALED_TEST")};scenes={x["scene_id"]:x for x in json.loads((root/config["inputs"]["pose_splits"]).read_text())["scenes"] if x["split"]=="SEALED_TEST"};bank=load_bank(root/"results/e11_mechanism_placement_transfer_v1");q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);c=_teacher_config(root,config);graph_dir=output/"test_graphs";graph_dir.mkdir(exist_ok=True);rows=[];events=[]
    for sid in sorted(scenes):
        cell_start=perf_counter();root_row=roots[sid];root_charge=float(root_row.get("root_seconds") or 0.);deadline=cell_start+max(0.,float(config["deployment"]["cell_deadline_s"])-root_charge);scene=scenes[sid];path=graph_dir/f"hemisphere_{sid}.npz"
        if root_row["status"]!="admitted":rows.append({"scene_id":sid,"method":"P_GRAPH","overall_status":"NOT_RUN_root_failed","cell_seconds":perf_counter()-cell_start});continue
        q=np.asarray(root_row["q0"],float);c["common_start_q"][sid]=q.tolist();began=perf_counter();built=build_placement_graph(root,c,bank,{"candidate_id":sid,"placement_level":sid,"transform_base_from_surface":scene["transform_base_from_surface"],"rng_seed":2026091700+int(scene["scene_index"])},q2);build_seconds=perf_counter()-began
        if built["status"] in {"ready","recombination_limited"}:save_robot_graph(path,built)
        graph_row={"scene_id":sid,"status":built["status"],"graph_file":str(path.relative_to(root)) if path.exists() else None,"graph_sha256":file_hash(path) if path.exists() else None,"build_seconds":build_seconds,"nodes":built.get("node_count"),"edges":built.get("edge_count"),"ik_calls":built.get("ik_calls")};manifest_path=output/"test_graph_manifest.partial.json";prior_graphs=json.loads(manifest_path.read_text()).get("graphs",[]) if manifest_path.exists() else [];write_json(manifest_path,{"graphs":[*prior_graphs,graph_row]})
        if not path.exists() or perf_counter()>=deadline:rows.append({"scene_id":sid,"method":"P_GRAPH","overall_status":"graph_build_failed" if not path.exists() else "time_budget_limited_after_graph_build",**graph_row,"cell_seconds":perf_counter()-cell_start});continue
        data=load_robot_graph(path);remaining=max(0.,deadline-perf_counter());f_result,archive=fixed_route_initialize(data,int(data["start_node"]),1,c,wall_time=min(float(config["teacher"]["search_seconds"]),remaining),expanded_limit=int(config["teacher"]["expanded_limit"]));plans=[]
        if f_result.incumbent is not None:
            pf,_=save_plan(output,sid,1,"P_GRAPH_F",f_result.incumbent,f_result.incumbent.path,data);plans.append(("F",pf,f_result.incumbent))
        fallback=None;accepted=[]
        for method,pf,label in plans:
            if perf_counter()>=deadline:break
            plan=np.load(root/pf,allow_pickle=False);started=perf_counter();local=json.loads(json.dumps(c));local["validation"]["deadline_s_per_unique_witness"]=max(0.,min(float(c["validation"]["deadline_s_per_unique_witness"]),deadline-perf_counter()));checked=validate_unique_witness(root,local,output,{"k":1},plan,data,{"transform_base_from_surface":scene["transform_base_from_surface"]},str(plan["witness_hash"]));status=checked["final"]["overall_status"] if perf_counter()<=deadline else "validation_budget_limited";events.append({"scene_id":sid,"method":"P_GRAPH_F","kind":"full_validation","status":status,"duration_s":perf_counter()-started,"witness_hash":str(plan["witness_hash"])});accepted.append((method,pf,label,status,checked))
            if status=="accepted_under_E09_R1_refined_sampled_checks":fallback=label
        remaining=max(0.,deadline-perf_counter());screen=_Q3Screen(root,c,data,{"transform_base_from_surface":scene["transform_base_from_surface"]});p_result=greedy_prefix_completion(data=data,start_node=int(data["start_node"]),maximum_on_segments=1,initial_prefixes=(tuple(x["path"]) for x in archive),validated_fallback=fallback,config=c,wall_time_s=remaining,screen=screen)
        for index,candidate in enumerate(p_result.candidates):
            if perf_counter()>=deadline:break
            from diffusion_coverage.solvers.history_search import SearchLabel
            label=SearchLabel(candidate.node,candidate.covered,candidate.membership,candidate.repeat_error,candidate.joint_cost,candidate.used_on_segments,candidate.edge_ids);pf,_=save_plan(output,sid,1,f"P_GRAPH_P{index}",label,candidate.edge_ids,data);plan=np.load(root/pf,allow_pickle=False);started=perf_counter();local=json.loads(json.dumps(c));local["validation"]["deadline_s_per_unique_witness"]=max(0.,min(float(c["validation"]["deadline_s_per_unique_witness"]),deadline-perf_counter()));checked=validate_unique_witness(root,local,output,{"k":1},plan,data,{"transform_base_from_surface":scene["transform_base_from_surface"]},str(plan["witness_hash"]));status=checked["final"]["overall_status"] if perf_counter()<=deadline else "validation_budget_limited";events.append({"scene_id":sid,"method":"P_GRAPH_P","kind":"full_validation","status":status,"duration_s":perf_counter()-started,"witness_hash":str(plan["witness_hash"])});accepted.append(("P",pf,label,status,checked))
        valid=[x for x in accepted if x[3]=="accepted_under_E09_R1_refined_sampled_checks"];best=min(valid,key=lambda x:(x[2].used_on_segments-1,x[2].joint_cost)) if valid else None;rows.append({"scene_id":sid,"method":"P_GRAPH","overall_status":"accepted_under_E12_refined_sampled_checks" if best else ("time_budget_limited" if perf_counter()>=deadline else "no_accepted_candidate"),**graph_row,"F_termination":f_result.termination,"P_termination":p_result.termination,"best_witness_hash":None if best is None else str(np.load(root/best[1],allow_pickle=False)["witness_hash"]),"best_J_q":None if best is None else best[2].joint_cost,"root_seconds_charged":root_charge,"cell_seconds":perf_counter()-cell_start,"cold_seconds":root_charge+perf_counter()-cell_start});write_csv(output/"test_graph_reference_results.partial.csv",rows);write_csv(output/"test_graph_reference_events.partial.csv",events)
    write_csv(output/"test_graph_reference_results.csv",rows);write_csv(output/"test_graph_reference_events.csv",events);write_json(output/"test-graph-reference.checkpoint.json",{"complete":True,"cells":len(rows),"accepted":sum(x["overall_status"]=='accepted_under_E12_refined_sampled_checks' for x in rows)})


def report_e12(root:Path,config:dict[str,Any],output:Path)->None:
    validation=read_csv(output/"validation_method_outcomes.csv");sealed=read_csv(output/"sealed_test_method_outcomes.csv");graph=read_csv(output/"test_graph_reference_results.csv");vsum=[x for x in validation if x.get("row_type")=="cell_summary"];ssum=[x for x in sealed if x.get("row_type")=="cell_summary"]
    main=[]
    for row in ssum:main.append({"scene_id":row["scene_id"],"method":row["method"],"overall_status":row["overall_status"],"best_witness_hash":row.get("best_witness_hash"),"best_J_q":row.get("best_J_q"),"time_to_first_accepted_s":row.get("first_accepted_s"),"cold_seconds":row.get("cold_seconds"),"evaluated_slots":row.get("evaluated_slots"),"candidate_slots":row.get("candidate_slots")})
    for row in graph:main.append({"scene_id":row["scene_id"],"method":"P_GRAPH","overall_status":row["overall_status"],"best_witness_hash":row.get("best_witness_hash"),"best_J_q":row.get("best_J_q"),"time_to_first_accepted_s":None,"cold_seconds":row.get("cold_seconds"),"evaluated_slots":None,"candidate_slots":None})
    write_csv(output/"sealed_test_main_results.csv",main);_make_e12_plots(root,config,output,main)
    accepted={method:sum(x["overall_status"]=="accepted_under_E12_refined_sampled_checks" for x in main if x["method"]==method) for method in ("RETRIEVE","REG","FM","P_LAZY","P_GRAPH")};val_accepted={method:sum(x["overall_status"]=="accepted_under_E12_refined_sampled_checks" for x in vsum if x["method"]==method) for method in ("RETRIEVE","REG","FM")};teacher=json.loads((output/"collect-train-val.checkpoint.json").read_text());training=json.loads((output/"training_manifest.json").read_text());oracle=read_csv(output/"oracle_relift.csv")
    lines=["# E12 graph-free global generation pilot","",f"Executed through report at {__import__('datetime').datetime.now().astimezone().isoformat()}.","","## Stages and fixed interface","",f"The 15 E11 accepted hashes were indexed; 12 single-ON witnesses were grammar-eligible, and {sum(x.get('validation_status')=='accepted_under_E12_refined_sampled_checks' for x in oracle)}/12 were re-lifted from q0 without reading a robot graph or future teacher q. The other three witnesses contain two ON segments and remain explicitly outside this k=1 pilot.",f"Teacher collection attempted {teacher.get('attempted_tasks')} TRAIN/VALIDATION poses and qualified {teacher.get('qualified_labels')} labels from {teacher.get('qualified_tasks')} tasks. Training used one seed on {training.get('device')} with final-update checkpoints.","","## Validation pilot (four registered poses)","","| task | RETRIEVE | REG | FM |","|---|---|---|---|"]
    for sid in sorted({x["scene_id"] for x in vsum}):lines.append("| "+sid+" | "+" | ".join(next(x["overall_status"] for x in vsum if x["scene_id"]==sid and x["method"]==method) for method in ("RETRIEVE","REG","FM"))+" |")
    lines += ["",f"Accepted cells: RETRIEVE {val_accepted['RETRIEVE']}/4, REG {val_accepted['REG']}/4, FM {val_accepted['FM']}/4. Validation outcomes were diagnostic only and did not alter the sealed policy.","","## Sealed test (all eight attempted poses)","","| task | RETRIEVE | REG | FM | P_lazy | P_graph |","|---|---|---|---|---|---|"]
    for sid in sorted({x["scene_id"] for x in main}):
        cells=[]
        for method in ("RETRIEVE","REG","FM","P_LAZY","P_GRAPH"):
            value=next((x["overall_status"] for x in main if x["scene_id"]==sid and x["method"]==method),"NOT_RUN");cells.append(value)
        lines.append("| "+sid+" | "+" | ".join(cells)+" |")
    lines += ["",f"Accepted cells: RETRIEVE {accepted['RETRIEVE']}/8, REG {accepted['REG']}/8, FM {accepted['FM']}/8, P_lazy {accepted['P_LAZY']}/8, P_graph {accepted['P_GRAPH']}/8.","","## Questions","",f"- **Q1 — graph-free representation and realization:** the oracle interface result is {sum(x.get('validation_status')=='accepted_under_E12_refined_sampled_checks' for x in oracle)}/12 eligible stored geometries accepted after independent q0-only realization. This is an interface diagnostic, not generated-plan success.",f"- **Q2 — FM generation:** FM produced independently accepted routes on {accepted['FM']}/8 sealed tasks under the fixed eight-slot policy.",f"- **Q3 — controls:** sealed accepted counts were RETRIEVE {accepted['RETRIEVE']}, REG {accepted['REG']}, FM {accepted['FM']}, P_lazy {accepted['P_LAZY']}, and P_graph {accepted['P_GRAPH']}. Incremental learning or stochastic-flow value is claimed only where these fixed controls differ.","- **Q4 — cold cost:** `sealed_test_main_results.csv` reports per-task charged root, candidate work, validation, and P_graph construction/search costs. Reusing no full robot graph is reflected only in RETRIEVE/REG/FM/P_lazy; teacher generation and training costs remain reported rather than treated as zero.","","## Boundaries","","This is one training seed and pose transfer around three anchors on one hemisphere. It does not test exogenous coverage history, RFM, non-spherical surfaces, force/contact control, hardware execution, or a faithful external published planner. Acceptance is under the finite refined sampled checker, not a continuous-time certificate. Collision claims cover only the pinned MuJoCo model."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    commands=[f"cd {root}","OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage prepare","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage repair-diagnostics","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage encode","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage oracle-relift","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage collect-train-val","CUDA_VISIBLE_DEVICES=1 /data/chocheng/.venvs/coverage-fm/bin/python scripts/train_e12_structured_generator.py","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage freeze-validation","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage validate-pilot","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage freeze-test","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage p-lazy","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage test-graph-free","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage test-graph-reference","/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e12_graph_free_global_generation_v1.py --stage report"];(output/"reproduction_commands.txt").write_text("\n".join(commands)+"\n");write_json(output/"report.checkpoint.json",{"complete":True,"main_cells":len(main),"validation_cells":len(vsum),"accepted":accepted})


def _make_e12_plots(root:Path,config:dict[str,Any],output:Path,main:list[dict[str,Any]])->None:
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
    scenes=scene_map(root);directory=output/"figures";directory.mkdir(exist_ok=True);q2=np.load(root/config["inputs"]["quadrature_q2"],allow_pickle=False);radius=float(config["surface"]["radius_m"]);index=[];seen=set()
    for row in main:
        witness=row.get("best_witness_hash")
        if not witness or witness in seen or row.get("overall_status")!="accepted_under_E12_refined_sampled_checks":continue
        seen.add(witness);path=output/"graph_free_accepted_witnesses"/f"{witness}.npz"
        if not path.exists():path=output/"selected_plan_witnesses"/f"{witness}.npz"
        if not path.exists():continue
        z=np.load(path,allow_pickle=False);q=np.asarray(z["q"]);activity=np.asarray(z["activity"],bool);transform=np.asarray(scenes[row["scene_id"]]["transform_base_from_surface"],float);robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));positions=[];sigma=[]
        for value in q:
            position,_=robot.forward(value);positions.append(position);sigma.append(evaluate_task_kinematics_5d(robot,value,characteristic_length=float(config["robot"]["characteristic_length_m"])).sigma_min_5)
        surface=_source_from_base(np.asarray(positions),transform,radius);counts=sphere_episode_counts_indexed(q2["points"],surface,activity,radius=radius,footprint_radius=float(config["coverage"]["footprint_radius_m"]));unit=np.asarray(q2["points"])/radius;az=np.arctan2(unit[:,1],unit[:,0]);polar=np.arccos(np.clip(unit[:,2],-1,1));stem=f"{row['scene_id']}_{witness[:12]}"
        fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection="3d");colors=np.where(counts==0,"#d62728",np.where(counts>1,"#ffbf00","#c7c7c7"));ax.scatter(q2["points"][:,0],q2["points"][:,1],q2["points"][:,2],c=colors,s=.25,alpha=.5);ax.plot(surface[activity,0],surface[activity,1],surface[activity,2],c="#1f77b4",lw=.45);ax.set_title(f"{row['scene_id']} executed route: red missed, amber repeated");fig.tight_layout();p3=directory/f"{stem}_whole_surface_3d.png";fig.savefig(p3,dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4));ax.scatter(az,polar,c=np.minimum(counts,2),s=.4,cmap="viridis",vmin=0,vmax=2,rasterized=True);ax.set(xlabel="azimuth [rad]",ylabel="polar angle [rad]",title=f"{row['scene_id']} Q2 episode map");fig.tight_layout();pu=directory/f"{stem}_coverage_unwrapped.png";fig.savefig(pu,dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,3));ax.plot(sigma,lw=.5);ax.axhline(float(config["robot"]["sigma_safe"]),c="r",ls="--");ax.set(xlabel="stored q sample",ylabel="sigma5",title=f"{row['scene_id']} full-route task margin");fig.tight_layout();ps=directory/f"{stem}_sigma5.png";fig.savefig(ps,dpi=160);plt.close(fig);index.append({"scene_id":row["scene_id"],"witness_hash":witness,"whole_surface_3d":str(p3.relative_to(root)),"coverage_unwrapped":str(pu.relative_to(root)),"sigma5":str(ps.relative_to(root))})
    write_json(output/"figure_index.json",{"figures":index})
