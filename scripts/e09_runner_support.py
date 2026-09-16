from __future__ import annotations

import csv
import ctypes
import gc
import hashlib
import heapq
import json
from pathlib import Path
import resource
from time import perf_counter
from typing import Any

import numpy as np

from diffusion_coverage.coverage.episode_summary import EpisodeEdgeSummary, EpisodeState, apply_edge_summary, initial_episode_state
from diffusion_coverage.coverage.ordered_trace_evaluator import resample_prescribed_path
from diffusion_coverage.robot.e09_execution import evaluate_e09_fk_trace, sphere_membership_stream
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics, transform_surface_pose_path
from diffusion_coverage.solvers.completion_bound import CompletionEdge
from diffusion_coverage.solvers.history_search import SearchGraph, SearchLabel, SearchMetrics, SearchResult, search_history_graph


FAMILIES = ("raster_u_phase_0.00", "raster_u_phase_0.25", "spiral_phase_0.00")


class IKCounter:
    def __init__(self, robot: UR5eKinematics, limit: int):
        self.robot = robot
        self.limit = limit
        self.calls = 0

    def solve(self, position, axis, seed, config):
        if self.calls >= self.limit:
            return None
        self.calls += 1
        return self.robot.solve_ik(
            position, axis, seed,
            position_tolerance=float(config["robot"]["ik_position_tolerance_m"]),
            axis_tolerance=np.deg2rad(float(config["robot"]["ik_axis_tolerance_degrees"])),
            max_iterations=int(config["robot"]["ik_max_iterations"]),
            damping=float(config["robot"]["ik_damping"]),
            max_update=float(config["robot"]["ik_max_update_rad"]),
            backend="task5",
        )


def build_all_graphs(root: Path, config: dict[str, Any], output: Path) -> None:
    bank = load_bank(output)
    q2 = np.load(root / config["inputs"]["quadrature_q2"], allow_pickle=False)
    scenes = selected_scenes(root, config)
    graph_dir = output / "graphs"; graph_dir.mkdir(exist_ok=True)
    rows = []
    checks = []
    for scene_index, scene in enumerate(scenes):
        started = perf_counter()
        built = build_placement_graph(root, config, bank, scene, q2)
        graph_file = graph_dir / f"hemisphere_{scene['candidate_id']}.npz"
        if built["status"] in {"ready", "recombination_limited"}:
            save_robot_graph(graph_file, built)
        row = {key: value for key, value in built.items() if key not in {"nodes_q", "node_ports", "node_memberships", "edges", "edge_meta", "witness_q", "witness_activity", "witness_target", "check_rows"}}
        row.update({"scene_id": scene["candidate_id"], "placement_level": scene["placement_level"], "graph_file": str(graph_file.relative_to(root)) if graph_file.exists() else None, "build_seconds": perf_counter() - started})
        rows.append(row)
        checks.extend(built.get("check_rows", []))
        print(json.dumps({"scene": scene["candidate_id"], "status": built["status"], "nodes": built.get("node_count", 0), "edges": built.get("edge_count", 0), "ik_calls": built.get("ik_calls", 0)}), flush=True)
    write_json(output / "graph_manifest.json", {"frozen_geometry_hash": bank["graph_hash"], "graphs_frozen_before_comparison": True, "placements": rows})
    write_csv(output / "node_and_edge_checks.csv", checks)
    write_csv(output / "failure_taxonomy.csv", failure_rows(rows, checks))
    checkpoint(output, "build", {"complete": True, "placements": len(rows), "usable_graphs": sum(r["status"] in {"ready", "recombination_limited"} for r in rows)})


def build_placement_graph(root, config, bank, scene, q2):
    robot = UR5eKinematics(config["inputs"]["robot_model"], site_name=config["robot"]["site_name"], tool_axis_index=int(config["robot"]["tool_axis_index"]), tool_axis_sign=float(config["robot"]["tool_axis_sign"]))
    counter = IKCounter(robot, int(config["construction"]["max_ik_calls"]))
    transform = np.asarray(scene["transform_base_from_surface"], dtype=np.float64)
    weights = np.asarray(q2["weights"], dtype=np.float64); samples = np.asarray(q2["points"], dtype=np.float64)
    rng = np.random.default_rng(int(config["seed"]) + int(scene["candidate_id"][1:]))
    symmetry = np.load(root / "results/symmetry_preserving_global_layout_v1/symmetry_orbits.npz", allow_pickle=False)
    seeds = [robot.home.copy(), np.asarray(symmetry["hemisphere_source_q_start"], dtype=np.float64)]
    seeds.extend(rng.uniform(robot.lower_limits, robot.upper_limits) for _ in range(6))
    nodes_q=[]; node_ports=[]; node_memberships=[]; edges=[]; edge_meta=[]; witness_q=[]; witness_activity=[]; witness_target=[]; checks=[]
    port_nodes: dict[int,list[int]] = {}
    failures: dict[str,int] = {}
    on_attempts=off_attempts=0

    def fail(reason): failures[reason]=failures.get(reason,0)+1
    def task_pose(points):
        normals=np.asarray(points)/float(config["surface"]["radius_m"])
        return transform_surface_pose_path(points,normals,transform,axis_opposes_normal=True)
    def admissible(q, position, axis):
        task=evaluate_task_kinematics_5d(robot,q,characteristic_length=float(config["robot"]["characteristic_length_m"])); checked=robot.evaluate_configuration(q)
        pe=float(np.linalg.norm(task.position-position)); ae=float(np.arccos(np.clip(np.dot(task.tool_axis,axis),-1,1)))
        return pe<=float(config["robot"]["position_tolerance_m"])+1e-12 and ae<=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"]))+1e-12 and task.sigma_min_5>=float(config["robot"]["sigma_safe"])-1e-12 and checked.collision_free
    def node_membership(q):
        position,_=robot.forward(q); inv=np.linalg.inv(transform); raw=(position-transform[:3,3])@transform[:3,:3]; point=float(config["surface"]["radius_m"])*raw/np.linalg.norm(raw)
        return sphere_membership_stream(samples,point[None,:],radius=float(config["surface"]["radius_m"]),footprint_radius=float(config["coverage"]["footprint_radius_m"]))[:,0]
    def add_node(port,q):
        for existing in port_nodes.get(port,[]):
            if np.max(np.abs(nodes_q[existing]-q))<=1e-9: return existing
        if len(port_nodes.get(port,[]))>=int(config["construction"]["max_q_candidates_per_port"]): return None
        if len(nodes_q)>=int(config["construction"]["max_nodes"]): return None
        idx=len(nodes_q); nodes_q.append(np.asarray(q,dtype=np.float64)); node_ports.append(int(port)); node_memberships.append(node_membership(q)); port_nodes.setdefault(int(port),[]).append(idx); return idx
    def dense_witness(q,target):
        qout=[q[0]]; tout=[target[0]]; radius=float(config["surface"]["radius_m"])
        for qa,qb,pa,pb in zip(q[:-1],q[1:],target[:-1],target[1:]):
            angle=np.arctan2(np.linalg.norm(np.cross(pa/radius,pb/radius)),np.dot(pa/radius,pb/radius)); dist=radius*angle
            count=max(1,int(np.ceil(max(dist/float(config["construction"]["edge_membership_spacing_m"]),np.max(np.abs(qb-qa))/float(config["construction"]["edge_joint_step_rad"])))))
            if angle>1e-14:
                frac=np.linspace(0,1,count+1)[1:]; pts=(np.sin((1-frac)*angle)[:,None]*(pa/radius)+np.sin(frac*angle)[:,None]*(pb/radius))/np.sin(angle)*radius
            else: pts=np.repeat(pb[None,:],count,axis=0)
            for j,f in enumerate(np.linspace(0,1,count+1)[1:]): qout.append((1-f)*qa+f*qb); tout.append(pts[j])
        return np.asarray(qout),np.asarray(tout)
    def add_edge(start,end,q,active,target,kind,geom_arc):
        nonlocal edges
        check=evaluate_e09_fk_trace(robot,q,active,target,transform,samples,weights,sphere_radius=float(config["surface"]["radius_m"]),footprint_radius=float(config["coverage"]["footprint_radius_m"]),characteristic_length=float(config["robot"]["characteristic_length_m"]))
        if not np.array_equal(check.summary.start_membership,node_memberships[start]) or not np.array_equal(check.summary.end_membership,node_memberships[end]): fail("endpoint_membership_mismatch"); return None
        axis_limit=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"])); pos_limit=float(config["robot"]["position_tolerance_m"]); sigma=float(config["robot"]["sigma_safe"])
        on=np.asarray(active,bool)
        valid=check.collision_free and check.min_joint_margin>=-1e-12 and (not np.any(on) or (check.max_position_error<=pos_limit+1e-12 and check.max_axis_error<=axis_limit+1e-12 and check.min_sigma5>=sigma-1e-12))
        checks.append({"scene_id":scene["candidate_id"],"entity":"edge","kind":kind,"geom_arc_id":geom_arc,"accepted":valid,"max_position_error_m":check.max_position_error,"max_axis_error_deg":np.rad2deg(check.max_axis_error),"min_sigma5":check.min_sigma5,"min_joint_margin":check.min_joint_margin,"collision_free":check.collision_free,"projection_residual_max_m":float(check.projection_residuals.max(initial=0.0))})
        if not valid: fail("edge_dense_contract"); return None
        eid=len(edges); cost=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()); edges.append(CompletionEdge(eid,start,end,check.summary,cost)); edge_meta.append({"kind":kind,"geom_arc_id":int(geom_arc),"start_port":int(node_ports[start]),"end_port":int(node_ports[end]),"family":bank["arc_family"][geom_arc] if geom_arc>=0 else kind}); witness_q.append(q); witness_activity.append(active); witness_target.append(target); return eid

    common_point=bank["ports"][int(bank["arc_start"][bank["routes"]["raster_u_phase_0.00/forward"][0]])]
    cp,ca=task_pose(common_point[None,:]); start_candidate=None
    for seed_index,seed in enumerate(seeds):
        candidate=counter.solve(cp[0],ca[0],seed,config)
        if candidate is not None and admissible(candidate.q,cp[0],ca[0]): start_candidate=candidate; break
        fail("start_seed_rejected")
    if start_candidate is None:
        return {"status":"start_search_failed","reason":"no_admissible_common_start","start_q":None,"node_count":0,"edge_count":0,"on_attempts":0,"off_attempts":0,"ik_calls":counter.calls,"verified_cross_port_on":0,"verified_off":0,"failure_counts":failures,"check_rows":checks,"collision_scope":config["collision_scope"]}
    start_port=int(bank["arc_start"][bank["routes"]["raster_u_phase_0.00/forward"][0]])
    start_node=add_node(start_port,start_candidate.q)

    # Build source fragments independently of whole-template success; preserve successful continuation between macros.
    for family in FAMILIES:
        current_q=start_candidate.q.copy() if family==FAMILIES[0] else None; current_node=start_node if family==FAMILIES[0] else None
        for geom_id in bank["routes"][f"{family}/forward"]:
            if on_attempts>=int(config["construction"]["max_on_attempts"]): break
            on_attempts+=1; points=resample_prescribed_path(bank["arc_points"][geom_id],surface_id="hemisphere",surface_metadata={"radius":float(config["surface"]["radius_m"])},maximum_step=float(config["construction"]["continuation_target_spacing_m"])); positions,axes=task_pose(points)
            if current_q is None or node_ports[current_node]!=int(bank["arc_start"][geom_id]):
                found=None
                for seed in seeds:
                    candidate=counter.solve(positions[0],axes[0],seed,config)
                    if candidate is not None and admissible(candidate.q,positions[0],axes[0]): found=candidate; break
                if found is None: fail("macro_start_not_found"); continue
                current_q=found.q; current_node=add_node(int(bank["arc_start"][geom_id]),current_q)
                if current_node is None: fail("node_capacity"); current_q=None; continue
            result=robot.continue_task_transition(current_q,positions,axes,maximum_joint_step=0.8,minimum_manipulability=0.0,position_tolerance=float(config["robot"]["position_tolerance_m"]),axis_tolerance=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"])),backend="task5")
            counter.calls += max(0,len(positions)-1)
            if not result.feasible: fail("on_continuation_"+str(result.failure_reason)); current_q=None; current_node=None; continue
            qdense,tdense=dense_witness(result.q_path,points)
            end_node=add_node(int(bank["arc_end"][geom_id]),qdense[-1])
            if end_node is None: fail("node_capacity"); current_q=None; current_node=None; continue
            eid=add_edge(current_node,end_node,qdense,np.ones(len(qdense),bool),tdense,"source",geom_id)
            if eid is None: current_q=None; current_node=None; continue
            reverse_geom=int(bank["reverse_arc"][geom_id]); add_edge(end_node,current_node,qdense[::-1].copy(),np.ones(len(qdense),bool),tdense[::-1].copy(),"source_reverse",reverse_geom)
            current_q=qdense[-1]; current_node=end_node

    # Task-preserving cross-port and common-entry connectors use exact stored endpoint q.
    connector_ids=list(bank["cross_arc_ids"])
    route_starts=[]
    for route,arc_ids in bank["routes"].items(): route_starts.append((route,int(bank["arc_start"][arc_ids[0]])))
    proposals=[("cross_port",gid) for gid in connector_ids]
    for route,port in route_starts:
        points=np.asarray([common_point,bank["ports"][port]])
        proposals.append(("entry:"+route,points))
    for kind,proposal in proposals:
        if on_attempts>=int(config["construction"]["max_on_attempts"]): break
        if isinstance(proposal,(int,np.integer)):
            gid=int(proposal); sp=int(bank["arc_start"][gid]); ep=int(bank["arc_end"][gid]); raw=bank["arc_points"][gid]
        else:
            gid=-1; sp=start_port; ep=min(range(len(bank["ports"])),key=lambda i:float(np.linalg.norm(bank["ports"][i]-proposal[-1]))); raw=proposal
        if not port_nodes.get(sp) or not port_nodes.get(ep): fail("connector_missing_endpoint_state"); continue
        for sn in list(port_nodes[sp]):
            for en in list(port_nodes[ep]):
                if on_attempts>=int(config["construction"]["max_on_attempts"]): break
                on_attempts+=1; points=resample_prescribed_path(raw,surface_id="hemisphere",surface_metadata={"radius":float(config["surface"]["radius_m"])},maximum_step=float(config["construction"]["continuation_target_spacing_m"]));
                if len(points) < 3:
                    middle = points[0] + points[-1]
                    if np.linalg.norm(middle) <= 1e-14:
                        fail("connector_antipodal"); continue
                    middle = float(config["surface"]["radius_m"]) * middle / np.linalg.norm(middle)
                    points = np.asarray([points[0], middle, points[-1]], dtype=np.float64)
                positions,axes=task_pose(points)
                result=robot.continue_task_transition_to_configuration(nodes_q[sn],nodes_q[en],positions,axes,maximum_joint_step=0.8,minimum_manipulability=0.0,position_tolerance=float(config["robot"]["position_tolerance_m"]),axis_tolerance=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"])),backend="task5")
                counter.calls+=max(0,len(positions)-2)
                if not result.feasible: fail("connector_"+str(result.failure_reason)); continue
                qdense,tdense=dense_witness(result.q_path,points)
                add_edge(sn,en,qdense,np.ones(len(qdense),bool),tdense,kind,gid)

    # Bounded nonlocal OFF reconfiguration proposals with explicit retreat/middle/return.
    all_pairs=[]
    for sn,q in enumerate(nodes_q):
        candidates=sorted(((float(np.linalg.norm(other-q)),en) for en,other in enumerate(nodes_q) if en!=sn and node_ports[en]!=node_ports[sn]))[:4]
        all_pairs.extend((sn,en) for _,en in candidates)
    for sn,en in all_pairs:
        if off_attempts>=int(config["construction"]["max_off_attempts"]): break
        off_attempts+=1; ps=bank["ports"][node_ports[sn]]; pe=bank["ports"][node_ports[en]]; pbase,abase=task_pose(np.asarray([ps,pe])); normals=np.asarray([ps,pe])/float(config["surface"]["radius_m"]); nb=normals@transform[:3,:3].T; retreat=float(config["construction"]["retreat_distance_m"])
        rs=counter.solve(pbase[0]+retreat*nb[0],abase[0],nodes_q[sn],config); re=counter.solve(pbase[1]+retreat*nb[1],abase[1],nodes_q[en],config)
        if rs is None or re is None: fail("off_retreat_ik"); continue
        n=max(2,int(np.ceil(np.max(np.abs(re.q-rs.q))/float(config["construction"]["edge_joint_step_rad"])))+1); middle=np.linspace(rs.q,re.q,n); q=np.vstack((nodes_q[sn],rs.q,middle[1:-1],re.q,nodes_q[en])); active=np.zeros(len(q),bool); active[[0,-1]]=True; target=np.vstack((ps,ps,np.linspace(ps,pe,max(0,len(q)-4)+2)[1:-1],pe,pe)) if len(q)>4 else np.asarray([ps,ps,pe,pe])[:len(q)]
        if target.shape!=(len(q),3): target=np.linspace(ps,pe,len(q)); target[0]=ps; target[-1]=pe
        if any(not robot.evaluate_configuration(value).collision_free for value in q): fail("off_collision"); continue
        add_edge(sn,en,q,active,target,"off_reconfiguration",-1)

    cross=sum(meta["kind"]=="cross_port" for meta in edge_meta); off=sum(meta["kind"]=="off_reconfiguration" for meta in edge_meta)
    status="ready" if cross else "recombination_limited"
    graph_hash=hash_robot_graph(nodes_q,node_ports,node_memberships,edges,edge_meta)
    return {"status":status,"reason":None,"graph_hash":graph_hash,"geometry_hash":bank["graph_hash"],"start_node":start_node,"start_q":nodes_q[start_node].tolist(),"node_count":len(nodes_q),"edge_count":len(edges),"on_attempts":on_attempts,"off_attempts":off_attempts,"ik_calls":counter.calls,"verified_cross_port_on":cross,"verified_off":off,"failure_counts":failures,"collision_scope":config["collision_scope"],"nodes_q":nodes_q,"node_ports":node_ports,"node_memberships":node_memberships,"edges":edges,"edge_meta":edge_meta,"witness_q":witness_q,"witness_activity":witness_activity,"witness_target":witness_target,"check_rows":checks}


def compare_all(root,config,output):
    manifest=json.loads((output/"graph_manifest.json").read_text()); results=[]; anytime=[]; pruning=[]; mechanism=None
    for task_index,row in enumerate(manifest["placements"]):
        for k in config["on_segment_budgets"]:
            if row["status"] not in {"ready","recombination_limited"}:
                for method in ("F","G0","G1"): results.append(empty_result(row,k,method,row["status"]));
                continue
            data=load_robot_graph(root/row["graph_file"]); graph=data["graph"]; start=int(data["start_node"])
            init=fixed_route_search(data,start,k,config,wall_time=min(30.0,float(config["search"]["initializer_seconds"])),expanded_limit=int(config["search"]["expanded_label_limit"]))
            common=init.incumbent
            release_memory()
            order=("G0","G1") if task_index%2==0 else ("G1","G0")
            method_results={"F":fixed_route_search(data,start,k,config,wall_time=float(config["search"]["wall_time_s"]),expanded_limit=int(config["search"]["expanded_label_limit"]),initial_incumbent=common)}
            release_memory()
            for method in order:
                method_results[method]=search_history_graph(graph,start_node=start,maximum_on_segments=k,missed_tolerance=float(config["coverage"]["missed_tolerance"]),repeat_tolerance=float(config["coverage"]["repeat_tolerance"]),use_completion_bound=method=="G1",wall_time_s=float(config["search"]["wall_time_s"]),expanded_limit=int(config["search"]["expanded_label_limit"]),checkpoint_times=tuple(float(x) for x in config["search"]["checkpoints_s"]),initial_incumbent=common,coverage_directed_order=True,memory_limit_bytes=int(float(config["search"]["private_memory_gib"])*(1024**3)),resident_label_limit=int(config["search"]["conservative_resident_label_limit"]))
                if mechanism is None and method_results[method].mechanism_sample is not None: mechanism={"scene_id":row["scene_id"],"k":k,**method_results[method].mechanism_sample}
                release_memory()
            for method in ("F","G0","G1"):
                result=method_results[method]; label=result.incumbent; plan_file=None
                if label is not None:
                    plan_file=save_plan(output,row["scene_id"],k,method,label,data)
                results.append(result_row(row,k,method,result,common,plan_file,graph.weights))
                for cp in result.checkpoints: anytime.append({"scene_id":row["scene_id"],"k":k,"method":method,**cp})
                pruning.append(pruning_row(row,k,method,result))
                write_csv(output/"global_results.partial.csv",results)
                write_csv(output/"anytime.partial.csv",anytime)
                write_csv(output/"pruning_stats.partial.csv",pruning)
                checkpoint(output,"compare-progress",{"complete":False,"cells":len(results),"last_scene":row["scene_id"],"last_k":k,"last_method":method})
                print(json.dumps({"scene_id":row["scene_id"],"k":k,"method":method,"termination":result.termination,"found":label is not None,"expanded":result.metrics.expanded,"seconds":result.elapsed_seconds}),flush=True)
    write_csv(output/"global_results.csv",results); write_csv(output/"fixed_route_results.csv",[r for r in results if r["method"]=="F"]); write_csv(output/"anytime.csv",anytime); write_csv(output/"pruning_stats.csv",pruning); write_json(output/"mechanism_example.json",mechanism or {"status":"not_observed"}); checkpoint(output,"compare",{"complete":True,"cells":len(results),"mechanism_example":mechanism is not None})


def fixed_route_search(data,start,k,config,*,wall_time,expanded_limit,initial_incumbent=None):
    graph=data["graph"]; meta=data["edge_meta"]; routes=data["routes"]; started=perf_counter(); metrics=SearchMetrics(); incumbent=initial_incumbent; first_time=None; serial=0; queue=[]
    initial=initial_episode_state(graph.node_membership[start]); root=SearchLabel(start,initial.covered,initial.membership,0.0,0.0,1,())
    outgoing={};
    for edge in graph.edges: outgoing.setdefault(edge.start,[]).append(edge)
    for route,sequence in sorted(routes.items()): heapq.heappush(queue,(0,-float(graph.weights[root.covered].sum()),0.0,serial,route,0,root)); serial+=1
    seen={}; termination="queue_exhausted"
    while queue:
        if perf_counter()-started>=wall_time: termination="wall_time"; break
        if metrics.expanded>=expanded_limit: termination="expanded_limit"; break
        if private_memory_bytes()>=int(float(config["search"]["private_memory_gib"])*(1024**3)): termination="memory_limit"; break
        if len(seen)+len(queue)>=int(config["search"]["conservative_resident_label_limit"]): termination="memory_limit_projected"; break
        _,_,_,_,route,index,label=heapq.heappop(queue); key=(route,index,label.node,np.packbits(label.covered).tobytes(),np.packbits(label.membership).tobytes(),label.used_on_segments)
        if seen.get(key,np.inf)<=label.joint_cost: metrics.dominance_pruned+=1; continue
        seen[key]=label.joint_cost; metrics.expanded+=1
        miss=float(graph.weights[~label.covered].sum()/graph.weights.sum())
        if miss<=float(config["coverage"]["missed_tolerance"])+1e-12 and label.repeat_error<=float(config["coverage"]["repeat_tolerance"])+1e-12:
            if first_time is None:first_time=perf_counter()-started
            if incumbent is None or (label.used_on_segments-1,label.joint_cost)<(incumbent.used_on_segments-1,incumbent.joint_cost): incumbent=label
            continue
        sequence=routes[route]
        candidates=[]
        if index==0 and label.node==start:
            route_start=int(data["arc_start"][sequence[0]])
            if int(data["node_ports"][label.node])==route_start: candidates.append((None,index))
            candidates.extend((edge,index) for edge in outgoing.get(label.node,()) if meta[edge.edge_id]["kind"]=="entry:"+route)
        elif index<len(sequence):
            candidates.extend((edge,index+1) for edge in outgoing.get(label.node,()) if meta[edge.edge_id]["geom_arc_id"]==int(sequence[index]))
            if k>label.used_on_segments: candidates.extend((edge,index) for edge in outgoing.get(label.node,()) if meta[edge.edge_id]["kind"]=="off_reconfiguration")
        for edge,next_index in candidates:
            if edge is None:
                heapq.heappush(queue,(label.used_on_segments-1,-float(graph.weights[label.covered].sum()),label.joint_cost,serial,route,next_index,label));serial+=1;continue
            metrics.generated+=1; used=label.used_on_segments+edge.summary.off_to_on_count
            if used>k: metrics.segment_pruned+=1;continue
            try: state=apply_edge_summary(EpisodeState(label.covered,label.membership,label.repeat_error),edge.summary,graph.weights)
            except ValueError: continue
            if state.repeat_error>float(config["coverage"]["repeat_tolerance"])+1e-12:metrics.repeat_pruned+=1;continue
            child=SearchLabel(edge.end,state.covered,state.membership,state.repeat_error,label.joint_cost+edge.joint_cost,used,label.path+(edge.edge_id,)); heapq.heappush(queue,(used-1,-float(graph.weights[state.covered].sum()),child.joint_cost,serial,route,next_index,child));serial+=1
    return SearchResult(incumbent,not queue and termination=="queue_exhausted",termination,perf_counter()-started,metrics,(),first_time,None)


def verify_all(root,config,output):
    rows=[]; comparison=read_csv(output/"global_results.csv")
    for row in comparison:
        if not row.get("plan_file"):
            rows.append({"scene_id":row["scene_id"],"k":row["k"],"method":row["method"],"status":"NO_PLAN","E_miss_Q1":None,"E_rep_Q1":None,"E_miss_Q2":None,"E_rep_Q2":None,"Q1_Q2_max_change":None,"min_sigma5":None,"max_position_error_m":None,"max_axis_error_deg":None,"collision_free":None,"whole_edge_summary_match":None,"plan_file":None}); continue
        graph_row=next(item for item in json.loads((output/"graph_manifest.json").read_text())["placements"] if item["scene_id"]==row["scene_id"]); data=load_robot_graph(root/graph_row["graph_file"]); scene=next(s for s in selected_scenes(root,config) if s["candidate_id"]==row["scene_id"]); plan=np.load(root/row["plan_file"],allow_pickle=False); q=plan["q"]; active=plan["activity"]; target=plan["target"]
        robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"])); metrics={}; checks={}
        for level in ("Q1","Q2"):
            quad=np.load(root/config["inputs"][f"quadrature_{level.lower()}"]); check=evaluate_e09_fk_trace(robot,q,active,target,np.asarray(scene["transform_base_from_surface"]),quad["points"],quad["weights"],sphere_radius=float(config["surface"]["radius_m"]),footprint_radius=float(config["coverage"]["footprint_radius_m"]),characteristic_length=float(config["robot"]["characteristic_length_m"])); total=float(quad["weights"].sum()); metrics[level]=(float(quad["weights"][~check.summary.footprint].sum()/total),float(np.dot(quad["weights"],np.maximum(check.summary.episode_counts-1,0))/total)); checks[level]=check
        change=max(abs(metrics["Q1"][0]-metrics["Q2"][0]),abs(metrics["Q1"][1]-metrics["Q2"][1])); c=checks["Q2"]; accepted=metrics["Q2"][0]<=float(config["coverage"]["missed_tolerance"])+1e-12 and metrics["Q2"][1]<=float(config["coverage"]["repeat_tolerance"])+1e-12 and change<=float(config["coverage"]["resolution_warning"])+1e-12 and c.max_position_error<=float(config["robot"]["position_tolerance_m"])+1e-12 and c.max_axis_error<=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"]))+1e-12 and c.min_sigma5>=float(config["robot"]["sigma_safe"])-1e-12 and c.collision_free
        status="accepted_execution" if accepted else ("numerically_unresolved" if change>float(config["coverage"]["resolution_warning"]) else "graph_feasible_but_validation_failed")
        rows.append({"scene_id":row["scene_id"],"k":row["k"],"method":row["method"],"status":status,"E_miss_Q1":metrics["Q1"][0],"E_rep_Q1":metrics["Q1"][1],"E_miss_Q2":metrics["Q2"][0],"E_rep_Q2":metrics["Q2"][1],"Q1_Q2_max_change":change,"min_sigma5":c.min_sigma5,"max_position_error_m":c.max_position_error,"max_axis_error_deg":np.rad2deg(c.max_axis_error),"collision_free":c.collision_free,"whole_edge_summary_match":abs(float(row["E_miss_graph"])-metrics["Q2"][0])<1e-12 and abs(float(row["E_rep_graph"])-metrics["Q2"][1])<1e-12,"plan_file":row["plan_file"]})
    write_csv(output/"final_validation.csv",rows); make_plots(root,config,output,rows); checkpoint(output,"verify",{"complete":True,"plans":sum(r["status"]!="NO_PLAN" for r in rows),"accepted":sum(r["status"]=="accepted_execution" for r in rows)})


def write_report(root,config,output):
    results=read_csv(output/"global_results.csv"); validation=read_csv(output/"final_validation.csv"); graphs=json.loads((output/"graph_manifest.json").read_text())["placements"]
    lines=["# E09 full-surface robot-aware coverage routing","","## Progress","","All three placements and both ON-segment budgets were retained. Graph construction, F/G0/G1 comparison, and independent Q1/Q2 validation completed where a common start and usable frozen graph existed. No FM, saddle, hardware, dynamics, or force experiment ran.","","## Six-task outcomes","","| placement | k | F | G0 | G1 |","|---|---:|---|---|---|"]
    for scene in config["placements"]:
        for k in config["on_segment_budgets"]:
            cells=[]
            for method in ("F","G0","G1"):
                vr=next((v for v in validation if v["scene_id"]==scene and int(v["k"])==k and v["method"]==method),None); rr=next(r for r in results if r["scene_id"]==scene and int(r["k"])==k and r["method"]==method)
                cells.append((vr["status"] if vr else "NO_PLAN")+f" ({rr['termination']})")
            lines.append(f"| {scene} | {k} | {cells[0]} | {cells[1]} | {cells[2]} |")
    accepted=[v for v in validation if v["status"]=="accepted_execution"]
    global_accepted=[v for v in accepted if v["method"] in {"G0","G1"}]
    lines += ["","## Answers","",f"**Q1.** {'Yes: at least one independently accepted complete global witness was found.' if global_accepted else 'No accepted complete global witness was found under the frozen numerical graph and budgets.'}"]
    improvements=[]
    for scene in config["placements"]:
        for k in config["on_segment_budgets"]:
            f=next((r for r in results if r["scene_id"]==scene and int(r["k"])==k and r["method"]=="F"),None); g=next((r for r in results if r["scene_id"]==scene and int(r["k"])==k and r["method"]=="G0"),None)
            if f and g and f["found"]=="True" and g["found"]=="True" and float(g["J_q"])<float(f["J_q"])-1e-12: improvements.append((scene,k))
    lines += ["",f"**Q2.** {'G0 improved Jq over F for '+str(improvements)+'.' if improvements else 'No validated improvement over the fixed-template baseline was measured.'}"]
    bound_prunes=sum(int(float(r["prospective_repeat_pruned"])) for r in results if r["method"]=="G1"); lines += ["",f"**Q3.** G1 recorded {bound_prunes} prospective-repeat prunes. Net timing is reported in `global_results.csv` and `pruning_stats.csv`; finite-graph feasible sets are unchanged.","","## Graph readiness"]
    for g in graphs: lines.append(f"- {g['scene_id']}: `{g['status']}`, {g.get('node_count',0)} states, {g.get('edge_count',0)} edges, {g.get('verified_cross_port_on',0)} verified cross-port ON edges, {g.get('verified_off',0)} OFF reconfigurations.")
    lines += ["","Every accepted edge ended at its stored q and used recomputed endpoint membership. Collision checks cover only the pinned MuJoCo model contacts; unmodeled workpiece, tool-body geometry beyond the XML, environment and cables remain outside the claim. Numerical failures do not prove physical infeasibility.","","## Interpretation boundary","","F is an internal fixed-library baseline. G0/G1 concern one frozen finite sampled graph. Path length Jq is neither time nor energy. The result is not continuous-time certification, hardware authorization, planner novelty, or an RSS claim. FM remained NOT_RUN."]
    (output/"report.md").write_text("\n".join(lines)+"\n"); checkpoint(output,"report",{"complete":True,"report":"results/e09_global_surface_routing_v1/report.md"})


def load_bank(output):
    doc=json.loads((output/"geometry_bank.json").read_text()); z=np.load(output/"geometry_bank.npz",allow_pickle=False); offsets=z["arc_offsets"]; points=tuple(np.asarray(z["arc_points"][offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)); routes={k:tuple(v) for k,v in doc["routes"].items()}; family=[]
    for i in range(len(points)):
        fi=int(z["arc_family_index"][i]); family.append(FAMILIES[fi] if fi>=0 else "cross_port")
    reverse={};
    for route,ids in routes.items():
        if route.endswith("/forward"):
            rev=routes[route.replace("/forward","/reverse")]
            for a,b in zip(ids,reversed(rev)):reverse[int(a)]=int(b)
    return {"graph_hash":doc["graph_hash"],"ports":z["ports"],"arc_points":points,"arc_start":z["arc_start"],"arc_end":z["arc_end"],"arc_kind":z["arc_kind"],"arc_family":tuple(family),"routes":routes,"reverse_arc":reverse,"cross_arc_ids":tuple(i for i,x in enumerate(z["arc_kind"]) if str(x)=="cross_port")}


def selected_scenes(root,config):
    doc=json.loads((root/config["inputs"]["placements"]).read_text()); chosen=doc["selected"]["hemisphere"]
    byid={}
    for level,value in chosen.items():
        record=dict(value); record["placement_level"]=level; byid[record["candidate_id"]]=record
    return [byid[s] for s in config["placements"]]


def save_robot_graph(path,b):
    q=[];active=[];target=[];offsets=[0]
    for a,c,t in zip(b["witness_q"],b["witness_activity"],b["witness_target"]):q.extend(a);active.extend(c);target.extend(t);offsets.append(len(q))
    e=b["edges"]; np.savez_compressed(path,nodes_q=np.asarray(b["nodes_q"]),node_ports=np.asarray(b["node_ports"]),node_memberships=np.asarray(b["node_memberships"]),weights=np.asarray(e[0].summary.footprint,dtype=float)*0+1 if False else np.asarray([]),edge_start=np.asarray([x.start for x in e]),edge_end=np.asarray([x.end for x in e]),edge_cost=np.asarray([x.joint_cost for x in e]),edge_footprint=np.asarray([x.summary.footprint for x in e]),edge_counts=np.asarray([x.summary.episode_counts for x in e],dtype=np.int16),edge_start_membership=np.asarray([x.summary.start_membership for x in e]),edge_end_membership=np.asarray([x.summary.end_membership for x in e]),edge_mass=np.asarray([x.summary.weighted_episode_mass for x in e]),edge_off_on=np.asarray([x.summary.off_to_on_count for x in e]),witness_q=np.asarray(q),witness_activity=np.asarray(active),witness_target=np.asarray(target),witness_offsets=np.asarray(offsets),metadata_json=np.asarray(json.dumps({"status":b["status"],"graph_hash":b["graph_hash"],"geometry_hash":b["geometry_hash"],"start_node":b["start_node"],"edge_meta":b["edge_meta"]},sort_keys=True)))


def load_robot_graph(path):
    z=np.load(path,allow_pickle=False); meta=json.loads(str(z["metadata_json"])); q2=np.load(Path(__file__).resolve().parents[1]/"results/e08_path_semantics_v1/quadrature_hemisphere_Q2.npz"); weights=np.asarray(q2["weights"]); edges=[]
    for i in range(len(z["edge_start"])):
        summary=EpisodeEdgeSummary(z["edge_footprint"][i],z["edge_counts"][i].astype(np.int64),z["edge_start_membership"][i],z["edge_end_membership"][i],float(z["edge_mass"][i]),int(z["edge_off_on"][i]));edges.append(CompletionEdge(i,int(z["edge_start"][i]),int(z["edge_end"][i]),summary,float(z["edge_cost"][i])))
    graph=SearchGraph(tuple(z["node_memberships"]),tuple(edges),weights,meta["graph_hash"]); bank=json.loads((path.parents[1]/"geometry_bank.json").read_text()); return {"graph":graph,"start_node":meta["start_node"],"edge_meta":meta["edge_meta"],"node_ports":z["node_ports"],"nodes_q":z["nodes_q"],"witness_q":z["witness_q"],"witness_activity":z["witness_activity"],"witness_target":z["witness_target"],"witness_offsets":z["witness_offsets"],"routes":{k:tuple(v) for k,v in bank["routes"].items()},"arc_start":np.load(path.parents[1]/"geometry_bank.npz")["arc_start"]}


def save_plan(output,scene,k,method,label,data):
    qs=[];acts=[];targets=[]; sequence=[]; on=off=entry=0.0
    for edge_id in label.path:
        lo,hi=data["witness_offsets"][edge_id:edge_id+2]; q=data["witness_q"][lo:hi]; a=data["witness_activity"][lo:hi]; t=data["witness_target"][lo:hi]
        if qs:q=q[1:];a=a[1:];t=t[1:]
        qs.extend(q);acts.extend(a);targets.extend(t); m=data["edge_meta"][edge_id]; cost=data["graph"].edges[edge_id].joint_cost
        if m["kind"]=="off_reconfiguration":off+=cost
        elif str(m["kind"]).startswith("entry:"):entry+=cost
        else:on+=cost
        sequence.append({"edge_id":edge_id,**m})
    directory=output/"selected_plan_witnesses";directory.mkdir(exist_ok=True);path=directory/f"{scene}_k{k}_{method}.npz";np.savez_compressed(path,q=np.asarray(qs),activity=np.asarray(acts),target=np.asarray(targets),edge_ids=np.asarray(label.path),sequence_json=np.asarray(json.dumps(sequence)),cost_decomposition=np.asarray([on,off,entry]));return str(path.relative_to(output.parents[1]))


def result_row(row,k,method,result,common,plan_file,weights):
    label=result.incumbent; miss=None if label is None else float(weights[~label.covered].sum()/weights.sum()); m=result.metrics
    return {"scene_id":row["scene_id"],"placement_level":row["placement_level"],"k":k,"method":method,"graph_hash":row.get("graph_hash"),"found":label is not None,"inherited_incumbent":common is not None,"first_solution_s":result.first_solution_seconds,"search_seconds":result.elapsed_seconds,"termination":result.termination,"graph_optimality_proved":result.optimality_proved,"on_segments":None if label is None else label.used_on_segments,"reconfigurations":None if label is None else label.used_on_segments-1,"J_q":None if label is None else label.joint_cost,"E_miss_graph":miss,"E_rep_graph":None if label is None else label.repeat_error,"expanded":m.expanded,"generated":m.generated,"dominance_pruned":m.dominance_pruned,"past_repeat_pruned":m.repeat_pruned,"segment_pruned":m.segment_pruned,"segment_reachability_pruned":m.segment_budget_reachability_pruned,"ordinary_reachability_pruned":m.reachability_pruned,"prospective_repeat_pruned":m.completion_bound_pruned,"bound_calls":m.completion_bound_calls,"bound_cache_hits":m.completion_bound_cache_hits,"bound_finite_positive":m.completion_bound_positive,"bound_seconds":m.completion_bound_seconds,"peak_memory_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"plan_file":plan_file}


def pruning_row(row,k,method,result):
    m=result.metrics;return {"scene_id":row["scene_id"],"k":k,"method":method,"segment_budget_reachability":m.segment_budget_reachability_pruned,"ordinary_reachability":m.reachability_pruned,"past_repeat":m.repeat_pruned,"dominance":m.dominance_pruned,"prospective_repeat":m.completion_bound_pruned,"bound_calls":m.completion_bound_calls,"bound_cache_hits":m.completion_bound_cache_hits,"bound_finite_positive":m.completion_bound_positive,"bound_seconds":m.completion_bound_seconds}


def empty_result(row,k,method,status): return {"scene_id":row["scene_id"],"placement_level":row["placement_level"],"k":k,"method":method,"graph_hash":row.get("graph_hash"),"found":False,"inherited_incumbent":False,"first_solution_s":None,"search_seconds":0.0,"termination":status,"graph_optimality_proved":False,"on_segments":None,"reconfigurations":None,"J_q":None,"E_miss_graph":None,"E_rep_graph":None,"expanded":0,"generated":0,"dominance_pruned":0,"past_repeat_pruned":0,"segment_pruned":0,"segment_reachability_pruned":0,"ordinary_reachability_pruned":0,"prospective_repeat_pruned":0,"bound_calls":0,"bound_cache_hits":0,"bound_finite_positive":0,"bound_seconds":0.0,"peak_memory_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"plan_file":None}


def failure_rows(rows,checks):
    output=[]
    for row in rows:
        for reason,count in row.get("failure_counts",{}).items():output.append({"scene_id":row["scene_id"],"stage":"construction","reason":reason,"count":count})
    return output


def make_plots(root,config,output,rows):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    directory=output/"figures";directory.mkdir(exist_ok=True)
    for row in rows:
        if row["status"]=="NO_PLAN":continue
        plan=np.load(root/row["plan_file"]);p=plan["target"];a=plan["activity"]
        fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(111,projection="3d");ax.plot(p[a,0],p[a,1],p[a,2],lw=.7,label="ON");
        if np.any(~a):ax.plot(p[~a,0],p[~a,1],p[~a,2],lw=.7,label="OFF")
        ax.set_title(f"{row['scene_id']} k={row['k']} {row['method']} whole hemisphere");ax.legend();fig.tight_layout();fig.savefig(directory/f"{row['scene_id']}_k{row['k']}_{row['method']}_route.png",dpi=150);plt.close(fig)


def hash_robot_graph(nodes,ports,memberships,edges,meta):
    h=hashlib.sha256();h.update(np.asarray(nodes).tobytes());h.update(np.asarray(ports).tobytes());h.update(np.asarray(memberships).tobytes())
    for edge,item in zip(edges,meta):h.update(np.asarray([edge.start,edge.end],dtype=np.int64).tobytes());h.update(edge.summary.episode_counts.tobytes());h.update(np.asarray([edge.joint_cost]).tobytes());h.update(json.dumps(item,sort_keys=True).encode())
    return h.hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def checkpoint(output,stage,payload):write_json(output/f"{stage}.checkpoint.json",{"stage":stage,**payload})
def write_csv(path,rows):
    if not rows:path.write_text("");return
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))


def private_memory_bytes():
    import os
    try:return int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    except (OSError,ValueError,IndexError):return 0


def release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError,AttributeError): pass
