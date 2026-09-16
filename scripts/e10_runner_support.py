from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from e09r1_runner_support import (
    concatenate_edges,
    file_hash,
    isolated_call,
    load_robot_graph,
    make_plots,
    quadrature,
    save_plan,
    selected_scenes,
    validate_unique_witness,
    write_csv,
    write_json,
)
from diffusion_coverage.robot.e09_execution import (
    evaluate_synchronized_fk_trace,
    sphere_episode_counts_indexed,
)
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.solvers.history_search import SearchLabel
from diffusion_coverage.solvers.structured_routing import (
    fixed_route_initialize,
    replay_edge_sequence,
    structured_anytime_search,
)


ACCEPTED = "accepted_under_E09_R1_refined_sampled_checks"


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _scene(root: Path, config: dict[str, Any], scene_id: str) -> dict[str, Any]:
    return next(item for item in selected_scenes(root, config) if item["candidate_id"] == scene_id)


def prepare(root: Path, config: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    parent = root / "results/e09_continuous_routing_repair_v1"
    published = json.loads((parent / "graph_manifest.json").read_text())
    references = []
    for scene_id in config["placements"]:
        source = Path(config["graph_sources"][scene_id]["path"])
        expected = config["graph_sources"][scene_id]["sha256"]
        actual = file_hash(source)
        if actual != expected:
            raise RuntimeError(f"graph hash mismatch for {scene_id}: {actual} != {expected}")
        row = next(x for x in published["multi_state_graphs"] if x["scene_id"] == scene_id)
        data = load_robot_graph(source)
        if data["graph"].graph_hash != row["graph_hash_multi"]:
            raise RuntimeError(f"semantic graph mismatch for {scene_id}")
        references.append({
            "scene_id": scene_id,
            "path": str(source),
            "sha256": actual,
            "graph_hash": data["graph"].graph_hash,
            "start_node": data["start_node"],
            "nodes": len(data["graph"].node_membership),
            "edges": len(data["graph"].edges),
            "inherited_build_seconds": row["build_seconds"],
            "inherited_graph_row": row,
        })
    if published["geometry_hash"] != config["expected_geometry_hash"]:
        raise RuntimeError("published geometry hash mismatch")
    manifest = {
        "experiment": config["experiment"],
        "prepared_at": datetime.now().astimezone().isoformat(),
        "timezone": str(datetime.now().astimezone().tzinfo),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "configuration_sha256": file_hash(root / "configs/e10_structured_anytime_routing_v1.json"),
        "geometry_hash": published["geometry_hash"],
        "graphs_reused_without_ik_or_rebuild": True,
        "graph_references": references,
        "parent_code_sha": config["parent_code_sha"],
        "parent_ara_sha": config["parent_ara_sha"],
        "tested_code_stage_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "model_path": config["inputs"]["robot_model"],
        "model_sha256": file_hash(Path(config["inputs"]["robot_model"])),
        "collision_scope": config["collision_scope"],
        "prospective_repeat_bound_calls_allowed": 0,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "config_snapshot.json", config)
    write_json(output / "graph_references.json", {"geometry_hash": published["geometry_hash"], "graphs": references})
    commands = [
        f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/e10-mpl /data/chocheng/.venvs/coverage-fm/bin/python scripts/run_e10_structured_anytime_routing_v1.py --stage {stage} --output results/e10_structured_anytime_routing_v1_reproduction"
        for stage in ("prepare", "tests", "initialize", "compare", "verify", "report")
    ]
    (output / "reproduction_commands.txt").write_text("\n".join(commands) + "\n")
    write_json(output / "prepare.checkpoint.json", {"complete": True, "graphs": len(references)})


def initialize(root: Path, config: dict[str, Any], output: Path) -> None:
    refs = json.loads((output / "graph_references.json").read_text())["graphs"]
    rows: list[dict[str, Any]] = []
    archive_doc: dict[str, Any] = {}
    routes: list[dict[str, Any]] = []
    validation_cache: dict[str, dict[str, Any]] = {}
    composition_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    for ref in refs:
        data = load_robot_graph(Path(ref["path"]))
        scene_id = ref["scene_id"]
        scene = _scene(root, config, scene_id)
        for k in config["on_segment_budgets"]:
            result, archive = fixed_route_initialize(
                data, int(data["start_node"]), int(k), config,
                wall_time=float(config["search"]["pipeline_core_seconds"]),
                expanded_limit=int(config["search"]["expanded_label_limit"]),
            )
            key = f"{scene_id}/k{k}"
            archive_doc[key] = [{**item, "path": list(item["path"])} for item in archive]
            plan_file = witness_hash = validation_status = None
            if result.incumbent is not None:
                plan_file, route = save_plan(output, scene_id, int(k), "F", result.incumbent, result.incumbent.path, data)
                routes.append(route)
                plan = np.load(root / plan_file, allow_pickle=False)
                witness_hash = str(plan["witness_hash"])
                if witness_hash not in validation_cache:
                    checked = validate_unique_witness(root, config, output, {"k": int(k)}, plan, data, scene, witness_hash)
                    validation_cache[witness_hash] = checked
                checked = validation_cache[witness_hash]
                validation_status = checked["final"]["overall_status"]
                composition_rows.append({"scene_id": scene_id, "k": k, "method": "F", "witness_hash": witness_hash, **checked["composition"]})
                resolution_rows.extend({"scene_id": scene_id, "k": k, "method": "F", "witness_hash": witness_hash, **item} for item in checked["resolution"])
            rows.append({
                "scene_id": scene_id, "k": k, "method": "F",
                "search_seconds": result.elapsed_seconds,
                "termination": result.termination,
                "queue_exhausted": result.optimality_proved,
                "expanded": result.metrics.expanded,
                "generated": result.metrics.generated,
                "archive_prefixes": len(archive),
                "graph_goal_found": result.incumbent is not None,
                "J_q": None if result.incumbent is None else result.incumbent.joint_cost,
                "E_miss_Q2_graph": None if result.incumbent is None else float(data["graph"].weights[~result.incumbent.covered].sum()/data["graph"].weights.sum()),
                "E_rep_Q2_graph": None if result.incumbent is None else result.incumbent.repeat_error,
                "plan_file": plan_file,
                "witness_hash": witness_hash,
                "refined_status": validation_status,
                "validated_fallback": validation_status == ACCEPTED,
            })
            write_csv(output / "initializer_results.partial.csv", rows)
            write_json(output / "shared_prefix_archive.partial.json", archive_doc)
    write_csv(output / "initializer_results.csv", rows)
    write_json(output / "shared_prefix_archive.json", archive_doc)
    write_csv(output / "route_classification.partial.csv", routes)
    write_csv(output / "same_sample_composition.partial.csv", composition_rows)
    write_csv(output / "validation_resolution.partial.csv", resolution_rows)
    # Compact validation records are enough to reconstruct the fallback verdict.
    fallback_rows = []
    for row in rows:
        if row["witness_hash"]:
            checked = validation_cache[row["witness_hash"]]["final"]
            fallback_rows.append({"scene_id": row["scene_id"], "k": row["k"], "method": "F", "plan_file": row["plan_file"], "witness_hash": row["witness_hash"], **checked})
    write_csv(output / "initializer_validation.csv", fallback_rows)
    write_json(output / "initialize.checkpoint.json", {"complete": True, "tasks": len(rows), "unique_validated": len(validation_cache)})


class _Q3Screen:
    def __init__(self, root: Path, config: dict[str, Any], data: dict[str, Any], scene: dict[str, Any]):
        self.root = root; self.config = config; self.data = data; self.scene = scene; self.robot = None

    def __call__(self, edge_ids: tuple[int, ...]) -> dict[str, Any]:
        if self.robot is None:
            self.robot = UR5eKinematics(self.config["inputs"]["robot_model"], site_name=self.config["robot"]["site_name"], tool_axis_index=int(self.config["robot"]["tool_axis_index"]), tool_axis_sign=float(self.config["robot"]["tool_axis_sign"]))
        transform = np.asarray(self.scene["transform_base_from_surface"], dtype=float)
        trace = concatenate_edges(self.data, edge_ids, float(self.config["validation"]["temporal"]["T0_joint_step_rad"]), float(self.config["validation"]["temporal"]["T0_surface_step_m"]), transform=transform, sphere_radius=float(self.config["surface"]["radius_m"]))
        radius=float(self.config["surface"]["radius_m"])
        checked = evaluate_synchronized_fk_trace(self.robot, trace, transform, np.asarray([[0.0,0.0,radius]]), np.ones(1), sphere_radius=radius, footprint_radius=float(self.config["coverage"]["footprint_radius_m"]), characteristic_length=float(self.config["robot"]["characteristic_length_m"]))
        points, weights = quadrature(self.config, "Q3", float(self.config["surface"]["radius_m"]), self.root)
        counts = sphere_episode_counts_indexed(points, checked.surface_points, trace.activity, radius=float(self.config["surface"]["radius_m"]), footprint_radius=float(self.config["coverage"]["footprint_radius_m"]))
        total = float(weights.sum()); miss = float(weights[counts == 0].sum()/total); repeat = float(np.dot(weights, np.maximum(counts-1, 0))/total)
        passed = miss <= float(self.config["coverage"]["missed_tolerance"])+1e-12 and repeat <= float(self.config["coverage"]["repeat_tolerance"])+1e-12
        return {"status": "Q3_PASS" if passed else "Q3_FAIL", "E_miss": miss, "E_rep": repeat, "trace_samples": len(trace.q)}


def _run_search(root, config, data, scene, k, archive, fallback, method, seconds):
    screen = _Q3Screen(root, config, data, scene)
    return structured_anytime_search(
        data,
        start_node=int(data["start_node"]),
        maximum_on_segments=int(k),
        initial_prefixes=(tuple(item["path"]) for item in archive),
        validated_fallback=fallback,
        use_source_runs=method == "B",
        config=config,
        wall_time_s=max(0.0, float(seconds)),
        screen=screen,
    )


def compare(root: Path, config: dict[str, Any], output: Path) -> None:
    refs = json.loads((output / "graph_references.json").read_text())["graphs"]
    initial = _csv(output / "initializer_results.csv")
    archive_doc = json.loads((output / "shared_prefix_archive.json").read_text())
    rows: list[dict[str, Any]] = []
    anytime: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    routes = _csv(output / "route_classification.partial.csv")
    candidates_doc: dict[str, Any] = {}
    task_index = 0
    for ref in refs:
        data = load_robot_graph(Path(ref["path"])); scene_id = ref["scene_id"]; scene = _scene(root, config, scene_id)
        for k in config["on_segment_budgets"]:
            init = next(x for x in initial if x["scene_id"] == scene_id and int(x["k"]) == int(k))
            f_seconds = float(init["search_seconds"]); remaining = max(0.0, float(config["search"]["pipeline_core_seconds"])-f_seconds)
            fallback = None
            if init["validated_fallback"] == "True":
                plan = np.load(root / init["plan_file"], allow_pickle=False)
                fallback = replay_edge_sequence(data["graph"], int(data["start_node"]), tuple(int(x) for x in plan["edge_ids"]))
            rows.append({
                "scene_id": scene_id, "k": k, "method": "F", "graph_hash": data["graph"].graph_hash,
                "initializer_seconds": f_seconds, "method_seconds": 0.0, "core_seconds": f_seconds,
                "termination": init["termination"], "graph_goal_found": init["graph_goal_found"],
                "validated_fallback_available": init["validated_fallback"], "fallback_retained": init["validated_fallback"],
                "novel_finalists": 0, "first_Q2_s": None, "first_Q3_pass_s": None,
                "expanded": init["expanded"], "generated_actions": init["generated"], "atomic_equivalent_work": init["generated"],
                "peak_open": None, "peak_ancestry": None, "peak_pareto_records": None, "heuristic_evictions": 0,
                "dominance_pruned": None, "repeat_pruned": None, "segment_pruned": None, "reachability_pruned": None, "objective_pruned": None,
                "prospective_bound_calls": 0, "screen_calls": 0, "screen_seconds": 0.0,
                "run_actions_evaluated": 0, "run_cache_hits": 0, "run_cache_misses": 0, "run_cache_peak_bytes": 0,
                "plan_files_json": json.dumps([init["plan_file"]] if init["plan_file"] else []),
            })
            order = config["search"]["method_order_by_task"][task_index % len(config["search"]["method_order_by_task"])]
            for method in order:
                result, peak = isolated_call(lambda method=method: _run_search(root, config, data, scene, k, archive_doc[f"{scene_id}/k{k}"], fallback, method, remaining))
                candidate_files = []
                candidate_meta = []
                for slot, candidate in enumerate(result.candidates):
                    label = SearchLabel(candidate.node, candidate.covered, candidate.membership, candidate.repeat_error, candidate.joint_cost, candidate.used_on_segments, candidate.edge_ids)
                    plan_file, route = save_plan(output, scene_id, int(k), f"{method}_novel{slot+1}", label, candidate.edge_ids, data)
                    routes.append(route); candidate_files.append(plan_file)
                    plan = np.load(root / plan_file, allow_pickle=False)
                    candidate_meta.append({"slot": slot+1, "plan_file": plan_file, "witness_hash": str(plan["witness_hash"]), "edge_ids": list(candidate.edge_ids), "screen_status": candidate.screen_status, "q2_miss": candidate.q2_miss, "q2_repeat": candidate.repeat_error, "q3_miss": candidate.q3_miss, "q3_repeat": candidate.q3_repeat})
                key = f"{scene_id}/k{k}/{method}"; candidates_doc[key] = candidate_meta
                for candidate in result.screened_candidates:
                    screens.append({"scene_id": scene_id, "k": k, "method": method, "edge_ids_sha256": hashlib.sha256(np.asarray(candidate.edge_ids,dtype=np.int64).tobytes()).hexdigest(), "screen_status": candidate.screen_status, "E_miss_Q2": candidate.q2_miss, "E_rep_Q2": candidate.repeat_error, "E_miss_Q3": candidate.q3_miss, "E_rep_Q3": candidate.q3_repeat, "seconds": candidate.screen_seconds})
                m = result.metrics
                rows.append({
                    "scene_id": scene_id, "k": k, "method": method, "graph_hash": data["graph"].graph_hash,
                    "initializer_seconds": f_seconds, "method_seconds": result.elapsed_seconds, "core_seconds": f_seconds+result.elapsed_seconds,
                    "remaining_budget_seconds": remaining, "termination": result.termination,
                    "graph_goal_found": bool(result.screened_candidates), "validated_fallback_available": fallback is not None,
                    "fallback_retained": result.fallback_retained, "novel_finalists": len(result.candidates),
                    "first_Q2_s": result.first_q2_seconds, "first_Q3_pass_s": result.first_q3_pass_seconds,
                    "expanded": m.expanded, "generated_actions": m.generated_actions, "atomic_equivalent_work": m.atomic_equivalent_work,
                    "peak_open": m.peak_open, "peak_ancestry": m.peak_ancestry, "peak_pareto_records": m.peak_pareto_records,
                    "heuristic_evictions": m.heuristic_evictions, "stale_labels": m.stale_labels, "rediscoveries": m.rediscoveries,
                    "dominance_pruned": m.dominance_pruned, "repeat_pruned": m.repeat_pruned, "segment_pruned": m.segment_pruned,
                    "reachability_pruned": m.reachability_pruned, "objective_pruned": m.objective_pruned,
                    "prospective_bound_calls": 0, "screen_calls": m.screen_calls, "screen_seconds": m.screen_seconds,
                    "run_actions_evaluated": m.run_actions_evaluated, "run_cache_hits": m.run_cache_hits, "run_cache_misses": m.run_cache_misses,
                    "run_cache_peak_bytes": m.run_cache_peak_bytes, "process_peak_bytes": peak,
                    "plan_files_json": json.dumps(candidate_files),
                })
                anytime.extend({"scene_id": scene_id, "k": k, "method": method, **cp} for cp in result.checkpoints)
                profiles.append({"scene_id": scene_id, "k": k, "method": method, **result.run_catalog})
                write_csv(output / "global_results.partial.csv", rows); write_json(output / "candidate_catalog.partial.json", candidates_doc)
                print(json.dumps({"scene": scene_id, "k": k, "method": method, "termination": result.termination, "expanded": m.expanded, "q2_candidates": len(result.screened_candidates), "finalists": len(result.candidates)}), flush=True)
            task_index += 1
            write_json(output / "compare.checkpoint.json", {"complete": False, "tasks_complete": task_index, "last_task": [scene_id, k]})
    write_csv(output / "global_results.csv", rows); write_csv(output / "anytime.csv", anytime); write_csv(output / "search_profile.csv", profiles); write_csv(output / "screening_results.csv", screens); write_csv(output / "route_classification.csv", routes); write_json(output / "candidate_catalog.json", candidates_doc)
    write_json(output / "action_catalog_summary.json", {"method_B": profiles, "run_lengths": config["search"]["run_lengths"], "branch_cap": config["search"]["run_branch_cap"], "cache_limit_mib": config["search"]["run_cache_mib"], "all_atomic_edges_retained": True})
    write_json(output / "compare.checkpoint.json", {"complete": True, "tasks": task_index, "cells": len(rows)})


def verify(root: Path, config: dict[str, Any], output: Path) -> None:
    refs = json.loads((output / "graph_references.json").read_text())["graphs"]
    results = _csv(output / "global_results.csv"); init_validation = _csv(output / "initializer_validation.csv")
    catalog = json.loads((output / "candidate_catalog.json").read_text())
    cache: dict[str, dict[str, Any]] = {}
    composition = _csv(output / "same_sample_composition.partial.csv"); resolution = _csv(output / "validation_resolution.partial.csv")
    final: list[dict[str, Any]] = []
    routes = _csv(output / "route_classification.csv")
    validation_seconds: dict[str, float] = {}
    load_rows=[]
    for ref in refs:
        began=perf_counter();load_robot_graph(Path(ref["path"]));load_rows.append({"scene_id":ref["scene_id"],"graph_load_seconds":perf_counter()-began,"bytes":Path(ref["path"]).stat().st_size})
    write_csv(output/"graph_load_benchmark.csv",load_rows)
    for row in results:
        scene_id=row["scene_id"]; k=int(row["k"]); method=row["method"]
        ref=next(x for x in refs if x["scene_id"]==scene_id);data=load_robot_graph(Path(ref["path"]));scene=_scene(root,config,scene_id)
        accepted=[]; checked_candidates=[]
        if method=="F":
            inherited=next((x for x in init_validation if x["scene_id"]==scene_id and int(x["k"])==k),None)
            if inherited:
                plan=np.load(root/inherited["plan_file"],allow_pickle=False);h=str(plan["witness_hash"])
                if h not in cache:
                    began=perf_counter();cache[h]=validate_unique_witness(root,config,output,{"k":k},plan,data,scene,h);validation_seconds[h]=perf_counter()-began
                checked=cache[h];inherited={"scene_id":scene_id,"k":k,"method":"F","plan_file":inherited["plan_file"],"witness_hash":h,**checked["final"]}
                if checked["final"]["overall_status"]==ACCEPTED:accepted.append((int(inherited["on_segments_checked"]),float(inherited["J_q_recorded"]),inherited["plan_file"],h,"retained_F"))
        else:
            fallback=next((x for x in init_validation if x["scene_id"]==scene_id and int(x["k"])==k and x["overall_status"]==ACCEPTED),None)
            if fallback:accepted.append((int(fallback["on_segments_checked"]),float(fallback["J_q_recorded"]),fallback["plan_file"],fallback["witness_hash"],"retained_F"))
            for item in catalog.get(f"{scene_id}/k{k}/{method}",[]):
                plan=np.load(root/item["plan_file"],allow_pickle=False);h=str(plan["witness_hash"]);began=perf_counter()
                if h not in cache:
                    cache[h]=validate_unique_witness(root,config,output,{"k":k},plan,data,scene,h);validation_seconds[h]=perf_counter()-began
                checked=cache[h]; composition.append({"scene_id":scene_id,"k":k,"method":method,"witness_hash":h,**checked["composition"]});resolution.extend({"scene_id":scene_id,"k":k,"method":method,"witness_hash":h,**x} for x in checked["resolution"])
                record={"plan_file":item["plan_file"],"witness_hash":h,**checked["final"]};checked_candidates.append(record)
                if checked["final"]["overall_status"]==ACCEPTED:accepted.append((int(checked["final"]["on_segments_checked"]),float(checked["final"]["J_q_recorded"]),item["plan_file"],h,"new_global_plan"))
        chosen=min(accepted,key=lambda x:(x[0]-1,x[1])) if accepted else None
        f_obj=None
        frow=next((x for x in init_validation if x["scene_id"]==scene_id and int(x["k"])==k and x["overall_status"]==ACCEPTED),None)
        if frow:f_obj=(int(frow["on_segments_checked"])-1,float(frow["J_q_recorded"]))
        outcome="no_accepted_plan"
        if chosen:
            outcome=chosen[4]
            if chosen[4]=="new_global_plan" and f_obj is not None and (chosen[0]-1,chosen[1])<f_obj:outcome="improved_F"
            elif chosen[4]=="new_global_plan" and f_obj is not None:outcome="retained_F"
        chosen_final=None
        if chosen:
            chosen_final=next((x for x in init_validation+checked_candidates if x.get("witness_hash")==chosen[3]),None)
        final.append({
            "scene_id":scene_id,"k":k,"method":method,"scoped_outcome":outcome,
            "overall_status":"NO_ACCEPTED_PLAN" if chosen is None else ACCEPTED,
            "selected_plan_file":None if chosen is None else chosen[2],"witness_hash":None if chosen is None else chosen[3],
            "novel_candidates_validated":len(checked_candidates),
            "novel_statuses_json":json.dumps([x["overall_status"] for x in checked_candidates]),
            "selected_validation_seconds_uncached_equivalent":None if chosen is None else validation_seconds.get(chosen[3]),
            "graph_load_seconds_equivalent":next(x["graph_load_seconds"] for x in load_rows if x["scene_id"]==scene_id),
            **({} if chosen_final is None else {key:value for key,value in chosen_final.items() if key not in {"scene_id","k","method","plan_file","witness_hash","overall_status"}}),
        })
    write_csv(output/"same_sample_composition.csv",composition);write_csv(output/"validation_resolution.csv",resolution);write_csv(output/"final_validation.csv",final)
    # Plots for unique newly validated witnesses and accepted fallback records.
    make_plots(root,config,output,final,cache)
    write_json(output/"mechanism_example.json",_mechanism(routes,final))
    write_json(output/"verify.checkpoint.json",{"complete":True,"unique_novel_witnesses":len(cache),"accepted_rows":sum(x["overall_status"]==ACCEPTED for x in final)})


def _mechanism(routes, final):
    accepted={x["witness_hash"] for x in final if x.get("scoped_outcome") in {"new_global_plan","improved_F"}}
    for route in routes:
        if route.get("witness_hash") in accepted and int(route.get("cross_port_on",0))>0:
            return {"status":"OBSERVED","type":"accepted_ON_recombination","witness_hash":route["witness_hash"],"classification":route["classification"],"sequence_json":route["sequence_json"]}
    return {"status":"NOT_OBSERVED","reason":"no independently accepted novel ON-recombination finalist"}


def report(root: Path, config: dict[str, Any], output: Path) -> None:
    results=_csv(output/"global_results.csv");final=_csv(output/"final_validation.csv");screens=_csv(output/"screening_results.csv")
    lines=["# E10 structured anytime global coverage routing", "", f"Executed: {datetime.now().astimezone().isoformat()}", "", "## Scope and implementation", "", "E10 reused the three frozen E09-R1 multi-state robot graphs without IK, connector, port, or geometry reconstruction. F is the corrected fixed-order reference. A uses bounded atomic-edge search; B changes only the action catalog by adding exact source-run compositions. Both retain a refined-validated F fallback and never call the prospective-repeat bound.", "", "## Six-task results", "", "| scene | k | F | A | B |", "|---|---:|---|---|---|"]
    for scene in config["placements"]:
        for k in config["on_segment_budgets"]:
            cells=[]
            for method in ("F","A","B"):
                rr=next(x for x in results if x["scene_id"]==scene and int(x["k"])==k and x["method"]==method);vv=next(x for x in final if x["scene_id"]==scene and int(x["k"])==k and x["method"]==method)
                cells.append(f"{vv['scoped_outcome']}; {rr['termination']}; {float(rr['core_seconds']):.2f}s; exp {rr['expanded']}")
            lines.append(f"| {scene} | {k} | {cells[0]} | {cells[1]} | {cells[2]} |")
    accepted_novel=[x for x in final if x["scoped_outcome"] in {"new_global_plan","improved_F"}]
    fallback=[x for x in final if x["scoped_outcome"]=="retained_F"]
    arows=[x for x in results if x["method"]=="A"];brows=[x for x in results if x["method"]=="B"]
    lines += ["", "## Findings", "", f"- Independently accepted novel global outputs: {len(accepted_novel)} method-task rows ({len(set(x['witness_hash'] for x in accepted_novel))} unique witnesses).", f"- Valid fixed-route fallback retained: {len(fallback)} method-task rows. Retention is an engineered no-regression property.", f"- Online screens executed: {len([x for x in screens if x['screen_status']!='SCREEN_NOT_RUN'])}; Q3 passes: {sum(x['screen_status']=='Q3_PASS' for x in screens)}.", f"- A expansions: {sum(int(x['expanded']) for x in arows)}; B expansions: {sum(int(x['expanded']) for x in brows)}. B source-run action evaluations: {sum(int(x['run_actions_evaluated']) for x in brows)}.", "- A/B are beam-limited anytime searches. Their exhaustion, timeout, or retained-record limit is not a finite-graph infeasibility or optimality certificate.", "- Refined acceptance is sampled under the inherited Q3/Q4/T1/Q4a schedule. It is not a continuous-time certificate or hardware authorization.", "", "## Reuse and limitations", "", "The graph collision claim is limited to the pinned MuJoCo model. Unmodeled workpiece, tool-body extent beyond the XML, environment, cables, dynamics, force, and control performance remain outside scope. These three placements are development cases. FM, hardware, G1, the single-state ablation, IK, and graph construction were NOT_RUN.", "", "## Files", "", "See `global_results.csv`, `final_validation.csv`, `screening_results.csv`, `validation_resolution.csv`, `route_classification.csv`, compact witnesses, and `reproduction_commands.txt` in this directory."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    write_json(output/"report.checkpoint.json",{"complete":True})
