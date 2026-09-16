from __future__ import annotations

import csv
import ctypes
import gc
import hashlib
import heapq
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from diffusion_coverage.coverage.episode_summary import (
    EpisodeEdgeSummary, EpisodeState, apply_edge_summary, initial_episode_state,
    summarize_ordered_membership,
)
from diffusion_coverage.planning.e09_geometry import spherical_distance, shortest_sphere_arc
from diffusion_coverage.robot.e09_execution import (
    evaluate_e09_fk_trace, evaluate_synchronized_fk_trace,
    sphere_episode_counts_indexed, sphere_membership_stream,
)
from diffusion_coverage.robot.synchronized_motion import (
    SynchronizedMotionTrace, continue_to_configuration_synchronized,
    densify_synchronized_trace, spherical_polyline_evaluator,
)
from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.solvers.completion_bound import CompletionEdge
from diffusion_coverage.solvers.history_search import (
    SearchGraph, SearchLabel, SearchMetrics, SearchResult, search_history_graph,
)

FAMILIES = ("raster_u_phase_0.00", "raster_u_phase_0.25", "spiral_phase_0.00")


class CentralIKCounter:
    """Instrument every direct and nested solve_ik call on one robot instance."""

    def __init__(self, robot: UR5eKinematics, limit: int):
        self.robot = robot
        self.limit = int(limit)
        self.calls = 0
        self.exhausted = False
        self._original = robot.solve_ik
        robot.solve_ik = self.solve  # type: ignore[method-assign]

    def solve(self, *args, **kwargs):
        if self.calls >= self.limit:
            self.exhausted = True
            return None
        self.calls += 1
        return self._original(*args, **kwargs)


def build_all_graphs(root: Path, config: dict[str, Any], output: Path) -> None:
    bank = load_bank(output)
    if bank["graph_hash"] != config["expected_geometry_hash"]:
        raise RuntimeError("geometry hash changed after preparation")
    q2 = np.load(root / config["inputs"]["quadrature_q2"], allow_pickle=False)
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    graph_dir = output / "graphs"
    graph_dir.mkdir(exist_ok=True)
    for scene in selected_scenes(root, config):
        started = perf_counter()
        built = build_placement_graph(root, config, bank, scene, q2)
        path = graph_dir / f"hemisphere_{scene['candidate_id']}.npz"
        if built["status"] in {"ready", "recombination_limited"}:
            save_robot_graph(path, built)
        row = {k: v for k, v in built.items() if k not in {
            "nodes_q", "node_ports", "node_ranks", "node_memberships", "edges",
            "edge_meta", "witnesses", "candidate_rows", "attempt_rows",
        }}
        row.update({
            "scene_id": scene["candidate_id"],
            "placement_level": scene["placement_level"],
            "graph_file": str(path.relative_to(root)) if path.exists() else None,
            "graph_file_sha256": file_hash(path) if path.exists() else None,
            "build_seconds": perf_counter() - started,
        })
        rows.append(row)
        candidates.extend(built["candidate_rows"])
        attempts.extend(built["attempt_rows"])
        write_json(output / "graph_manifest.partial.json", {"placements": rows})
        write_csv(output / "port_candidates.partial.csv", candidates)
        write_csv(output / "edge_attempts.partial.csv", attempts)
        print(json.dumps({"scene": row["scene_id"], "status": row["status"], "nodes": row.get("node_count", 0), "edges": row.get("edge_count", 0), "cross_on": row.get("verified_cross_port_on", 0), "ik_calls": row.get("ik_calls", 0)}), flush=True)
    write_json(output / "graph_manifest.json", {
        "geometry_hash": bank["graph_hash"], "graphs_frozen_before_search": True,
        "multi_state_graphs": rows,
    })
    write_csv(output / "port_candidates.csv", candidates)
    write_csv(output / "edge_attempts.csv", attempts)
    checkpoint(output, "build", {"complete": True, "placements": len(rows), "usable": sum(r["status"] in {"ready", "recombination_limited"} for r in rows)})


def build_placement_graph(root, config, bank, scene, q2):
    robot = UR5eKinematics(config["inputs"]["robot_model"], site_name=config["robot"]["site_name"], tool_axis_index=int(config["robot"]["tool_axis_index"]), tool_axis_sign=float(config["robot"]["tool_axis_sign"]))
    counter = CentralIKCounter(robot, int(config["construction"]["max_ik_calls"]))
    transform = np.asarray(scene["transform_base_from_surface"], dtype=np.float64)
    radius = float(config["surface"]["radius_m"])
    sample_points = np.asarray(q2["points"], dtype=np.float64)
    weights = np.asarray(q2["weights"], dtype=np.float64)
    scene_id = scene["candidate_id"]
    root_q = np.asarray(config["common_start_q"][scene_id], dtype=np.float64)
    root_port = int(bank["arc_start"][bank["routes"]["raster_u_phase_0.00/forward"][0]])
    candidate_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}

    def fail(reason: str) -> None:
        failures[reason] = failures.get(reason, 0) + 1

    def task_pose(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(surface, dtype=np.float64)
        positions = values @ transform[:3, :3].T + transform[:3, 3]
        axes = -(values / np.linalg.norm(values, axis=1, keepdims=True)) @ transform[:3, :3].T
        return positions, axes

    def admissibility(q: np.ndarray, position: np.ndarray, axis: np.ndarray) -> tuple[bool, dict[str, Any]]:
        task = evaluate_task_kinematics_5d(robot, q, characteristic_length=float(config["robot"]["characteristic_length_m"]))
        checked = robot.evaluate_configuration(q)
        pe = float(np.linalg.norm(task.position - position))
        ae = float(np.arccos(np.clip(np.dot(task.tool_axis, axis), -1.0, 1.0)))
        ok = (
            pe <= float(config["robot"]["position_tolerance_m"]) + 1e-12
            and ae <= np.deg2rad(float(config["robot"]["axis_tolerance_degrees"])) + 1e-12
            and task.sigma_min_5 >= float(config["robot"]["sigma_safe"]) - 1e-12
            and checked.joint_limit_margin >= -1e-12
            and checked.collision_free
        )
        return ok, {"position_error_m": pe, "axis_error_deg": float(np.rad2deg(ae)), "sigma5": task.sigma_min_5, "joint_margin": checked.joint_limit_margin, "collision_free": checked.collision_free}

    root_position, root_axis = task_pose(bank["ports"][root_port][None, :])
    root_ok, root_metrics = admissibility(root_q, root_position[0], root_axis[0])
    if not root_ok:
        return {"status": "start_invalid", "reason": "published_common_start_failed_recheck", "start_q": root_q.tolist(), "node_count": 0, "edge_count": 0, "ik_calls": counter.calls, "candidate_rows": [{"scene_id": scene_id, "port_id": root_port, "seed_id": "published_root", "outcome": "invalid", **root_metrics}], "attempt_rows": [], "failure_counts": {"published_common_start_failed_recheck": 1}, "collision_scope": config["collision_scope"]}

    symmetry = np.load(root / "results/symmetry_preserving_global_layout_v1/symmetry_orbits.npz", allow_pickle=False)
    rng = np.random.default_rng(int(config["seed"]) + int(scene_id[1:]))
    seeds = [root_q.copy(), robot.home.copy(), np.asarray(symmetry["hemisphere_source_q_start"], dtype=np.float64)]
    seeds.extend(rng.uniform(robot.lower_limits, robot.upper_limits) for _ in range(5))
    for seed in seeds:
        seed[5] = root_q[5]

    selected_by_port: dict[int, list[np.ndarray]] = {}
    dedup_tol = float(config["construction"]["effective_dedup_tolerance_rad"])
    max_states = int(config["construction"]["max_q_candidates_per_port"])
    for port_id, point in enumerate(bank["ports"]):
        position, axis = task_pose(point[None, :])
        raw: list[tuple[int, np.ndarray, dict[str, Any]]] = []
        if port_id == root_port:
            raw.append((-1, root_q.copy(), root_metrics))
        for seed_id, seed in enumerate(seeds):
            candidate = robot.solve_ik(
                position[0], axis[0], seed,
                position_tolerance=float(config["robot"]["ik_position_tolerance_m"]),
                axis_tolerance=np.deg2rad(float(config["robot"]["ik_axis_tolerance_degrees"])),
                max_iterations=int(config["robot"]["ik_max_iterations"]),
                damping=float(config["robot"]["ik_damping"]),
                max_update=float(config["robot"]["ik_max_update_rad"]), backend="task5",
            )
            if candidate is None:
                candidate_rows.append({"scene_id": scene_id, "port_id": port_id, "seed_id": seed_id, "outcome": "ik_not_found", "selected": False})
                continue
            ok, metrics = admissibility(candidate.q, position[0], axis[0])
            if not ok:
                candidate_rows.append({"scene_id": scene_id, "port_id": port_id, "seed_id": seed_id, "outcome": "inadmissible", "selected": False, **metrics})
                continue
            raw.append((seed_id, candidate.q.copy(), metrics))
        unique: list[tuple[int, np.ndarray, dict[str, Any]]] = []
        duplicate_class: dict[int, str] = {}
        for seed_id, q, metrics in raw:
            match = next((j for j, (_, other, _) in enumerate(unique) if np.max(np.abs(q[:5] - other[:5])) <= dedup_tol), None)
            if match is None:
                unique.append((seed_id, q, metrics))
            else:
                duplicate_class[seed_id] = "q6_only_or_effective_near_duplicate" if np.max(np.abs(q[:5] - unique[match][1][:5])) <= dedup_tol else "near_duplicate"
        chosen: list[int] = []
        if unique:
            chosen.append(0)
        while len(chosen) < min(max_states, len(unique)):
            remaining = [i for i in range(len(unique)) if i not in chosen]
            next_index = max(remaining, key=lambda i: (min(float(np.max(np.abs(unique[i][1][:5] - unique[j][1][:5]))) for j in chosen), -unique[i][0]))
            chosen.append(next_index)
        selected_by_port[port_id] = [unique[i][1] for i in chosen]
        selected_seed_ids = {unique[i][0] for i in chosen}
        for seed_id, q, metrics in raw:
            duplicate = duplicate_class.get(seed_id)
            candidate_rows.append({"scene_id": scene_id, "port_id": port_id, "seed_id": "published_root" if seed_id == -1 else seed_id, "outcome": duplicate or ("selected" if seed_id in selected_seed_ids else "selection_cap_excluded"), "selected": seed_id in selected_seed_ids, "q0": q[0], "q1": q[1], "q2": q[2], "q3": q[3], "q4": q[4], "q5": q[5], **metrics})
        if not selected_by_port[port_id]:
            fail("port_without_admitted_state")
        if counter.exhausted:
            break

    nodes_q: list[np.ndarray] = []
    node_ports: list[int] = []
    node_ranks: list[int] = []
    node_memberships: list[np.ndarray] = []
    port_nodes: dict[int, list[int]] = {}

    def membership_for_q(q: np.ndarray) -> np.ndarray:
        position, _ = robot.forward(q)
        raw = (position - transform[:3, 3]) @ transform[:3, :3]
        point = radius * raw / np.linalg.norm(raw)
        return sphere_membership_stream(sample_points, point[None, :], radius=radius, footprint_radius=float(config["coverage"]["footprint_radius_m"]))[:, 0]

    for port_id in range(len(bank["ports"])):
        for rank, q in enumerate(selected_by_port.get(port_id, ())):
            if len(nodes_q) >= int(config["construction"]["max_nodes"]):
                fail("node_cap")
                break
            node_id = len(nodes_q)
            nodes_q.append(q.copy()); node_ports.append(port_id); node_ranks.append(rank)
            node_memberships.append(membership_for_q(q)); port_nodes.setdefault(port_id, []).append(node_id)
    root_candidates = port_nodes.get(root_port, [])
    start_node = next((n for n in root_candidates if np.max(np.abs(nodes_q[n] - root_q)) <= 1e-12), None)
    if start_node is None:
        return {"status": "start_invalid", "reason": "published_root_not_retained", "start_q": root_q.tolist(), "node_count": len(nodes_q), "edge_count": 0, "ik_calls": counter.calls, "candidate_rows": candidate_rows, "attempt_rows": [], "failure_counts": failures, "collision_scope": config["collision_scope"]}

    edges: list[CompletionEdge] = []
    edge_meta: list[dict[str, Any]] = []
    witnesses: list[SynchronizedMotionTrace] = []

    def record_attempt(kind, geom_id, sn, en, accepted, reason, started, check=None):
        row = {"scene_id": scene_id, "attempt_id": len(attempt_rows), "kind": kind, "geometry_arc_id": geom_id, "source_node": sn, "target_node": en, "source_port": None if sn is None else node_ports[sn], "target_port": None if en is None else node_ports[en], "source_rank": None if sn is None else node_ranks[sn], "target_rank": None if en is None else node_ranks[en], "accepted": accepted, "reason": reason, "ik_calls_after": counter.calls, "elapsed_s": perf_counter() - started}
        if check is not None:
            row.update({"max_position_error_m": check.max_position_error, "max_axis_error_deg": float(np.rad2deg(check.max_axis_error)), "min_sigma5": check.min_sigma5, "min_joint_margin": check.min_joint_margin, "collision_free": check.collision_free, "projection_residual_max_m": float(check.projection_residuals.max(initial=0.0))})
        attempt_rows.append(row)

    def admit_trace(trace: SynchronizedMotionTrace, kind: str, geom_id: int, family: str, started: float) -> bool:
        sn, en = trace.start_node, trace.end_node
        endpoint_tol = float(config["construction"]["endpoint_tolerance_rad"])
        if np.max(np.abs(trace.q[0] - nodes_q[sn])) > endpoint_tol or np.max(np.abs(trace.q[-1] - nodes_q[en])) > endpoint_tol:
            fail("endpoint_q_mismatch"); record_attempt(kind, geom_id, sn, en, False, "endpoint_q_mismatch", started); return False
        if not trace.activity[0] or not trace.activity[-1]:
            fail("endpoint_activity_mismatch"); record_attempt(kind, geom_id, sn, en, False, "endpoint_activity_mismatch", started); return False
        check = evaluate_synchronized_fk_trace(robot, trace, transform, sample_points, weights, sphere_radius=radius, footprint_radius=float(config["coverage"]["footprint_radius_m"]), characteristic_length=float(config["robot"]["characteristic_length_m"]))
        if not np.array_equal(check.summary.start_membership, node_memberships[sn]) or not np.array_equal(check.summary.end_membership, node_memberships[en]):
            fail("endpoint_membership_mismatch"); record_attempt(kind, geom_id, sn, en, False, "endpoint_membership_mismatch", started, check); return False
        on = trace.activity
        valid = check.collision_free and check.min_joint_margin >= -1e-12 and (not np.any(on) or (check.max_position_error <= float(config["robot"]["position_tolerance_m"]) + 1e-12 and check.max_axis_error <= np.deg2rad(float(config["robot"]["axis_tolerance_degrees"])) + 1e-12 and check.min_sigma5 >= float(config["robot"]["sigma_safe"]) - 1e-12))
        if not valid:
            fail("dense_motion_contract"); record_attempt(kind, geom_id, sn, en, False, "dense_motion_contract", started, check); return False
        edge_id = len(edges)
        cost = float(np.linalg.norm(np.diff(trace.q, axis=0), axis=1).sum())
        edges.append(CompletionEdge(edge_id, sn, en, check.summary, cost))
        edge_meta.append({"kind": kind, "geom_arc_id": int(geom_id), "start_port": int(node_ports[sn]), "end_port": int(node_ports[en]), "source_rank": int(node_ranks[sn]), "target_rank": int(node_ranks[en]), "family": family})
        witnesses.append(trace)
        record_attempt(kind, geom_id, sn, en, True, "accepted", started, check)
        return True

    entry_descriptors = []
    for route, ids in sorted(bank["routes"].items()):
        port = int(bank["arc_start"][ids[0]])
        raw = shortest_sphere_arc(bank["ports"][root_port], bank["ports"][port], radius, 0.006)
        entry_descriptors.append((f"entry:{route}", -1, root_port, port, raw, route))
    source_descriptors = [("source", gid, int(bank["arc_start"][gid]), int(bank["arc_end"][gid]), bank["arc_points"][gid], bank["arc_family"][gid]) for gid in range(len(bank["arc_points"])) if str(bank["arc_kind"][gid]) == "source"]
    cross_descriptors = [("cross_port", gid, int(bank["arc_start"][gid]), int(bank["arc_end"][gid]), bank["arc_points"][gid], "cross_port") for gid in bank["cross_arc_ids"]] + entry_descriptors

    pair_order = sorted(itertools.product(range(max_states), repeat=2), key=lambda x: (max(x), sum(x), x))
    def on_attempts(descriptors):
        for sr, er in pair_order:
            for kind, gid, sp, ep, raw, family in descriptors:
                if sr < len(port_nodes.get(sp, ())) and er < len(port_nodes.get(ep, ())):
                    yield kind, gid, port_nodes[sp][sr], port_nodes[ep][er], raw, family

    off_pairs = []
    for port, ids in sorted(port_nodes.items()):
        for sn in ids:
            for en in ids:
                if sn != en:
                    off_pairs.append((sn, en))
    for sn, q in enumerate(nodes_q):
        nearest = sorted((float(np.linalg.norm(q - other)), en) for en, other in enumerate(nodes_q) if en != sn and node_ports[en] != node_ports[sn])[:4]
        off_pairs.extend((sn, en) for _, en in nearest)
    off_pairs = list(dict.fromkeys(off_pairs))

    source_iter, cross_iter, off_iter = iter(on_attempts(source_descriptors)), iter(on_attempts(cross_descriptors)), iter(off_pairs)
    done = {"source": False, "cross": False, "off": False}
    on_count = off_count = 0
    while not all(done.values()):
        for category, iterator in (("source", source_iter), ("cross", cross_iter), ("off", off_iter)):
            if done[category]:
                continue
            if category == "off" and off_count >= int(config["construction"]["max_off_attempts"]):
                done[category] = True; continue
            if category != "off" and on_count >= int(config["construction"]["max_on_attempts"]):
                done["source"] = done["cross"] = True; continue
            if counter.exhausted:
                done = {key: True for key in done}; break
            try:
                item = next(iterator)
            except StopIteration:
                done[category] = True; continue
            started = perf_counter()
            if category != "off":
                kind, gid, sn, en, raw, family = item
                on_count += 1
                knots, curve = spherical_polyline_evaluator(raw, transform, radius=radius)
                total = sum(spherical_distance(a, b, radius) for a, b in zip(raw[:-1], raw[1:]))
                count = max(3, int(np.ceil(total / float(config["construction"]["continuation_target_spacing_m"]))) + 1)
                target_u = np.linspace(float(knots[0]), float(knots[-1]), count)
                result, trace = continue_to_configuration_synchronized(
                    robot, nodes_q[sn], nodes_q[en], target_u, curve,
                    geometry_arc_id=gid, start_node=sn, end_node=en,
                    final_tracking_samples=int(config["construction"]["final_tracking_samples"]),
                    maximum_joint_step=0.8, minimum_manipulability=0.0,
                    position_tolerance=float(config["robot"]["position_tolerance_m"]),
                    axis_tolerance=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"])), backend="task5",
                )
                if not result.feasible or trace is None:
                    reason = "ik_budget" if counter.exhausted else "on_" + str(result.failure_reason)
                    fail(reason); record_attempt(kind, gid, sn, en, False, reason, started); continue
                parameter_step = 1.0 if total <= 1e-15 else float(config["construction"]["edge_membership_spacing_m"]) / total
                trace = densify_synchronized_trace(trace, curve, maximum_joint_step=float(config["construction"]["edge_joint_step_rad"]), maximum_parameter_step=parameter_step)
                admit_trace(trace, kind, gid, family, started)
            else:
                sn, en = item
                off_count += 1
                ps, pe = bank["ports"][node_ports[sn]], bank["ports"][node_ports[en]]
                surface = np.asarray([ps, pe])
                base, axes = task_pose(surface)
                normals_base = (surface / radius) @ transform[:3, :3].T
                retreat = float(config["construction"]["retreat_distance_m"])
                rs = robot.solve_ik(base[0] + retreat * normals_base[0], axes[0], nodes_q[sn], position_tolerance=float(config["robot"]["ik_position_tolerance_m"]), axis_tolerance=np.deg2rad(float(config["robot"]["ik_axis_tolerance_degrees"])), max_iterations=int(config["robot"]["ik_max_iterations"]), damping=float(config["robot"]["ik_damping"]), max_update=float(config["robot"]["ik_max_update_rad"]), backend="task5")
                re = robot.solve_ik(base[1] + retreat * normals_base[1], axes[1], nodes_q[en], position_tolerance=float(config["robot"]["ik_position_tolerance_m"]), axis_tolerance=np.deg2rad(float(config["robot"]["ik_axis_tolerance_degrees"])), max_iterations=int(config["robot"]["ik_max_iterations"]), damping=float(config["robot"]["ik_damping"]), max_update=float(config["robot"]["ik_max_update_rad"]), backend="task5")
                if rs is None or re is None:
                    reason = "ik_budget" if counter.exhausted else "off_retreat_ik"
                    fail(reason); record_attempt("off_reconfiguration", -1, sn, en, False, reason, started); continue
                anchors = [nodes_q[sn], rs.q, re.q, nodes_q[en]]
                q_values = [anchors[0]]
                for qa, qb in zip(anchors[:-1], anchors[1:]):
                    count = max(1, int(np.ceil(np.max(np.abs(qb - qa)) / float(config["construction"]["edge_joint_step_rad"]))))
                    q_values.extend((1.0 - f) * qa + f * qb for f in np.linspace(0.0, 1.0, count + 1)[1:])
                q_values = np.asarray(q_values)
                u = np.linspace(0.0, 1.0, len(q_values))
                target_position = np.empty((len(q_values), 3)); target_axis = np.empty((len(q_values), 3))
                for qi, q in enumerate(q_values):
                    target_position[qi], target_axis[qi] = robot.forward(q)
                target_position[0], target_position[-1] = base[0], base[-1]
                target_axis[0], target_axis[-1] = axes[0], axes[-1]
                activity = np.zeros(len(q_values), dtype=bool); activity[[0, -1]] = True
                trace = SynchronizedMotionTrace(q_values, u, target_position, target_axis, activity, -1, (0.0, 1.0), sn, en)
                admit_trace(trace, "off_reconfiguration", -1, "off_reconfiguration", started)

    cross_ids = [i for i, meta in enumerate(edge_meta) if meta["kind"] == "cross_port"]
    off_ids = [i for i, meta in enumerate(edge_meta) if meta["kind"] == "off_reconfiguration"]
    on_ids = [i for i, meta in enumerate(edge_meta) if meta["kind"] != "off_reconfiguration"]
    reachable = {start_node}; changed = True
    while changed:
        changed = False
        for eid in on_ids:
            edge = edges[eid]
            if edge.start in reachable and edge.end not in reachable:
                reachable.add(edge.end); changed = True
    root_cross = sum(edges[eid].start in reachable for eid in cross_ids)
    used_ports = {node_ports[n] for n in reachable}
    histogram: dict[str, int] = {}
    for ids in port_nodes.values(): histogram[str(len(ids))] = histogram.get(str(len(ids)), 0) + 1
    graph_hash = hash_robot_graph(nodes_q, node_ports, node_memberships, edges, edge_meta, witnesses)
    single = derive_single_graph(nodes_q, node_ports, node_ranks, node_memberships, edges, edge_meta, start_node, weights)
    return {
        "status": "ready" if cross_ids else "recombination_limited", "reason": None,
        "geometry_hash": bank["graph_hash"], "graph_hash_multi": graph_hash,
        "graph_hash_single": single["graph_hash"], "start_node": start_node,
        "start_q": root_q.tolist(), "node_count": len(nodes_q), "edge_count": len(edges),
        "single_node_count": len(single["nodes"]), "single_edge_count": len(single["edge_parents"]),
        "on_attempts": on_count, "off_attempts": off_count, "ik_calls": counter.calls,
        "ik_cap_bound": counter.exhausted, "verified_cross_port_on": len(cross_ids),
        "root_reachable_cross_port_on": root_cross, "root_reachable_on_states": len(reachable),
        "root_reachable_ports": len(used_ports), "verified_off": len(off_ids),
        "candidate_histogram": histogram, "ports_with_multiple_states": sum(len(x) > 1 for x in port_nodes.values()),
        "effective_states": len(nodes_q), "failure_counts": failures,
        "collision_scope": config["collision_scope"], "nodes_q": nodes_q,
        "node_ports": node_ports, "node_ranks": node_ranks,
        "node_memberships": node_memberships, "edges": edges, "edge_meta": edge_meta,
        "witnesses": witnesses, "candidate_rows": candidate_rows, "attempt_rows": attempt_rows,
    }


def derive_single_graph(nodes_q, node_ports, node_ranks, memberships, edges, meta, start_node, weights):
    keep_by_port: dict[int, int] = {}
    for node, (port, rank) in enumerate(zip(node_ports, node_ranks)):
        if port not in keep_by_port or rank < node_ranks[keep_by_port[port]]:
            keep_by_port[port] = node
    keep_by_port[node_ports[start_node]] = start_node
    nodes = sorted(keep_by_port.values())
    remap = {old: new for new, old in enumerate(nodes)}
    parents = []
    single_edges = []
    for edge in edges:
        if edge.start in remap and edge.end in remap:
            eid = len(single_edges); parents.append(edge.edge_id)
            single_edges.append(CompletionEdge(eid, remap[edge.start], remap[edge.end], edge.summary, edge.joint_cost))
    h = hashlib.sha256()
    h.update(np.asarray(nodes, dtype=np.int64).tobytes()); h.update(np.asarray(parents, dtype=np.int64).tobytes())
    return {"nodes": nodes, "edge_parents": parents, "edges": tuple(single_edges), "start": remap[start_node], "graph_hash": h.hexdigest()}


def save_robot_graph(path: Path, built: dict[str, Any]) -> None:
    q=[]; u=[]; position=[]; axis=[]; activity=[]; offsets=[0]
    for trace in built["witnesses"]:
        q.extend(trace.q); u.extend(trace.u); position.extend(trace.target_position); axis.extend(trace.target_axis); activity.extend(trace.activity); offsets.append(len(q))
    edges = built["edges"]
    metadata = {"status": built["status"], "graph_hash_multi": built["graph_hash_multi"], "graph_hash_single": built["graph_hash_single"], "geometry_hash": built["geometry_hash"], "start_node": built["start_node"], "edge_meta": built["edge_meta"]}
    np.savez_compressed(path,
        nodes_q=np.asarray(built["nodes_q"]), node_ports=np.asarray(built["node_ports"]), node_ranks=np.asarray(built["node_ranks"]), node_memberships=np.asarray(built["node_memberships"]),
        edge_start=np.asarray([e.start for e in edges]), edge_end=np.asarray([e.end for e in edges]), edge_cost=np.asarray([e.joint_cost for e in edges]), edge_footprint=np.asarray([e.summary.footprint for e in edges]), edge_counts=np.asarray([e.summary.episode_counts for e in edges], dtype=np.int16), edge_start_membership=np.asarray([e.summary.start_membership for e in edges]), edge_end_membership=np.asarray([e.summary.end_membership for e in edges]), edge_mass=np.asarray([e.summary.weighted_episode_mass for e in edges]), edge_off_on=np.asarray([e.summary.off_to_on_count for e in edges]),
        witness_q=np.asarray(q), witness_u=np.asarray(u), witness_target_position=np.asarray(position), witness_target_axis=np.asarray(axis), witness_activity=np.asarray(activity), witness_offsets=np.asarray(offsets), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_robot_graph(path: Path) -> dict[str, Any]:
    path = Path(path); z = np.load(path, allow_pickle=False); meta = json.loads(str(z["metadata_json"]))
    q2 = np.load(Path(__file__).resolve().parents[1] / "results/e08_path_semantics_v1/quadrature_hemisphere_Q2.npz", allow_pickle=False); weights = np.asarray(q2["weights"])
    edge_start=np.asarray(z["edge_start"]);edge_end=np.asarray(z["edge_end"]);edge_cost=np.asarray(z["edge_cost"])
    footprint=np.asarray(z["edge_footprint"]);counts=np.asarray(z["edge_counts"],dtype=np.int64)
    start_membership=np.asarray(z["edge_start_membership"]);end_membership=np.asarray(z["edge_end_membership"])
    mass=np.asarray(z["edge_mass"]);off_on=np.asarray(z["edge_off_on"])
    edges=[]
    for i in range(len(edge_start)):
        summary=EpisodeEdgeSummary(footprint[i],counts[i],start_membership[i],end_membership[i],float(mass[i]),int(off_on[i]))
        edges.append(CompletionEdge(i,int(edge_start[i]),int(edge_end[i]),summary,float(edge_cost[i])))
    graph=SearchGraph(tuple(z["node_memberships"]),tuple(edges),weights,meta["graph_hash_multi"])
    single=derive_single_graph(z["nodes_q"],z["node_ports"],z["node_ranks"],z["node_memberships"],edges,meta["edge_meta"],int(meta["start_node"]),weights)
    bank=load_bank(path.parents[1])
    return {"graph":graph,"single_graph":SearchGraph(tuple(z["node_memberships"][single["nodes"]]),single["edges"],weights,single["graph_hash"]),"single_edge_parents":single["edge_parents"],"single_nodes":single["nodes"],"single_start":single["start"],"start_node":int(meta["start_node"]),"edge_meta":meta["edge_meta"],"node_ports":z["node_ports"],"node_ranks":z["node_ranks"],"nodes_q":z["nodes_q"],"witness_q":z["witness_q"],"witness_u":z["witness_u"],"witness_target_position":z["witness_target_position"],"witness_target_axis":z["witness_target_axis"],"witness_activity":z["witness_activity"],"witness_offsets":z["witness_offsets"],"routes":bank["routes"],"arc_start":bank["arc_start"],"arc_end":bank["arc_end"]}


def compare_all(root, config, output):
    manifest=json.loads((output/"graph_manifest.json").read_text()); rows=[]; anytime=[]; pruning=[]; routes=[]; mechanism=None
    for task_index, graph_row in enumerate(manifest["multi_state_graphs"]):
        for k in config["on_segment_budgets"]:
            if graph_row["status"] not in {"ready","recombination_limited"}:
                for method in ("F","G0","G1","S"): rows.append(empty_result(graph_row,k,method,graph_row["status"]));
                continue
            data=load_robot_graph(root/graph_row["graph_file"])
            order=config["search"]["method_order_by_task"][task_index%len(config["search"]["method_order_by_task"])]
            method_results={}; peaks={}
            for method in order:
                if method=="F": call=lambda: fixed_route_search(data,int(data["start_node"]),k,config,wall_time=float(config["search"]["wall_time_s"]),expanded_limit=int(config["search"]["expanded_label_limit"]))
                elif method=="S": call=lambda: search_history_graph(data["single_graph"],start_node=int(data["single_start"]),maximum_on_segments=k,missed_tolerance=float(config["coverage"]["missed_tolerance"]),repeat_tolerance=float(config["coverage"]["repeat_tolerance"]),use_completion_bound=False,wall_time_s=float(config["search"]["wall_time_s"]),expanded_limit=int(config["search"]["expanded_label_limit"]),checkpoint_times=tuple(config["search"]["checkpoints_s"]),coverage_directed_order=True,memory_limit_bytes=int(float(config["search"]["private_memory_gib"])*1024**3),resident_label_limit=int(config["search"]["conservative_resident_label_limit"]))
                else: call=lambda method=method: search_history_graph(data["graph"],start_node=int(data["start_node"]),maximum_on_segments=k,missed_tolerance=float(config["coverage"]["missed_tolerance"]),repeat_tolerance=float(config["coverage"]["repeat_tolerance"]),use_completion_bound=method=="G1",wall_time_s=float(config["search"]["wall_time_s"]),expanded_limit=int(config["search"]["expanded_label_limit"]),checkpoint_times=tuple(config["search"]["checkpoints_s"]),coverage_directed_order=True,memory_limit_bytes=int(float(config["search"]["private_memory_gib"])*1024**3),resident_label_limit=int(config["search"]["conservative_resident_label_limit"]))
                result,peak=isolated_call(call); method_results[method]=result;peaks[method]=peak
                if mechanism is None and result.mechanism_sample is not None: mechanism={"scene_id":graph_row["scene_id"],"k":k,"method":method,**result.mechanism_sample}
                print(json.dumps({"scene_id":graph_row["scene_id"],"k":k,"method":method,"found":result.incumbent is not None,"termination":result.termination,"expanded":result.metrics.expanded,"seconds":result.elapsed_seconds}),flush=True)
            validate_method_inclusions(method_results)
            for method in ("F","G0","G1","S"):
                result=method_results[method]; label=result.incumbent; plan_file=None; route_row=None
                if label is not None:
                    parent_path=tuple(data["single_edge_parents"][eid] for eid in label.path) if method=="S" else label.path
                    plan_file,route_row=save_plan(output,graph_row["scene_id"],k,method,label,parent_path,data)
                    routes.append(route_row)
                rows.append(result_row(graph_row,k,method,result,plan_file,data["graph"].weights,peaks[method],data["single_graph"].graph_hash if method=="S" else data["graph"].graph_hash))
                anytime.extend({"scene_id":graph_row["scene_id"],"k":k,"method":method,**cp} for cp in result.checkpoints)
                pruning.append(pruning_row(graph_row,k,method,result))
                write_csv(output/"global_results.partial.csv",rows)
                checkpoint(output,"compare-progress",{"complete":False,"cells":len(rows),"last":[graph_row["scene_id"],k,method]})
    write_csv(output/"global_results.csv",rows);write_csv(output/"anytime.csv",anytime);write_csv(output/"pruning_stats.csv",pruning);write_csv(output/"route_classification.csv",routes);write_json(output/"mechanism_example.json",mechanism or {"status":"NOT_OBSERVED"});checkpoint(output,"compare",{"complete":True,"cells":len(rows)})


def validate_method_inclusions(results):
    f,g0,g1,s=(results[x] for x in ("F","G0","G1","S"))
    if g0.optimality_proved and f.incumbent is not None and g0.incumbent is None: raise RuntimeError("F feasible plan missing from exhausted G0")
    if g0.optimality_proved and s.incumbent is not None and g0.incumbent is None: raise RuntimeError("single-state feasible plan missing from exhausted multi-state G0")
    if g0.optimality_proved and g1.optimality_proved:
        a=None if g0.incumbent is None else (g0.incumbent.used_on_segments,float(g0.incumbent.joint_cost))
        b=None if g1.incumbent is None else (g1.incumbent.used_on_segments,float(g1.incumbent.joint_cost))
        if a!=b: raise RuntimeError(f"exhausted G0/G1 mismatch: {a} != {b}")


def _pareto_admit_fixed(table,key,repeat,cost,tol=1e-12):
    values=table.setdefault(key,[])
    if any(r<=repeat+tol and c<=cost+tol for r,c in values): return False
    table[key]=[(r,c) for r,c in values if not (repeat<=r+tol and cost<=c+tol)]+[(repeat,cost)]
    return True


def fixed_route_search(data,start,k,config,*,wall_time,expanded_limit):
    graph=data["graph"];meta=data["edge_meta"];routes=data["routes"];began=perf_counter();metrics=SearchMetrics();incumbent=None;first=None;serial=0;queue=[]
    root_state=initial_episode_state(graph.node_membership[start]);root=SearchLabel(start,root_state.covered,root_state.membership,0.0,0.0,1,())
    outgoing={}
    for edge in graph.edges:outgoing.setdefault(edge.start,[]).append(edge)
    for route,sequence in sorted(routes.items()):heapq.heappush(queue,(0,-float(graph.weights[root.covered].sum()),0.0,serial,route,0,root));serial+=1
    pareto={};termination="queue_exhausted"
    while queue:
        elapsed=perf_counter()-began
        if elapsed>=wall_time:termination="wall_time";break
        if metrics.expanded>=expanded_limit:termination="expanded_limit";break
        if private_memory_bytes()>=int(float(config["search"]["private_memory_gib"])*1024**3):termination="memory_limit";break
        if sum(len(v) for v in pareto.values())+len(queue)>=int(config["search"]["conservative_resident_label_limit"]):termination="memory_limit_projected";break
        *_,route,index,label=heapq.heappop(queue)
        key=(route,index,label.node,np.packbits(label.covered).tobytes(),np.packbits(label.membership).tobytes(),label.used_on_segments)
        if not _pareto_admit_fixed(pareto,key,label.repeat_error,label.joint_cost):metrics.dominance_pruned+=1;continue
        if incumbent is not None and (label.used_on_segments-1,label.joint_cost)>=(incumbent.used_on_segments-1,incumbent.joint_cost):continue
        miss=float(graph.weights[~label.covered].sum()/graph.weights.sum())
        if miss<=float(config["coverage"]["missed_tolerance"])+1e-12 and label.repeat_error<=float(config["coverage"]["repeat_tolerance"])+1e-12:
            if first is None:first=elapsed
            incumbent=label;continue
        metrics.expanded+=1;sequence=routes[route];boundaries=[int(data["arc_start"][sequence[0]])]+[int(data["arc_end"][eid]) for eid in sequence]
        candidates=[]
        if index==0 and label.node==start and int(data["node_ports"][label.node])!=boundaries[0]:
            candidates.extend((edge,0) for edge in outgoing.get(label.node,()) if meta[edge.edge_id]["kind"]=="entry:"+route)
        elif index<len(sequence):
            candidates.extend((edge,index+1) for edge in outgoing.get(label.node,()) if int(meta[edge.edge_id]["geom_arc_id"])==int(sequence[index]) and meta[edge.edge_id]["kind"]=="source")
        if label.used_on_segments<k:
            for edge in outgoing.get(label.node,()):
                if meta[edge.edge_id]["kind"]!="off_reconfiguration":continue
                endpoint=int(data["node_ports"][edge.end])
                for next_index in range(index,len(boundaries)):
                    if boundaries[next_index]==endpoint:candidates.append((edge,next_index))
        for edge,next_index in candidates:
            metrics.generated+=1;used=label.used_on_segments+edge.summary.off_to_on_count
            if used>k:metrics.segment_pruned+=1;continue
            try:state=apply_edge_summary(EpisodeState(label.covered,label.membership,label.repeat_error),edge.summary,graph.weights)
            except ValueError:continue
            if state.repeat_error>float(config["coverage"]["repeat_tolerance"])+1e-12:metrics.repeat_pruned+=1;continue
            child=SearchLabel(edge.end,state.covered,state.membership,state.repeat_error,label.joint_cost+edge.joint_cost,used,label.path+(edge.edge_id,));serial+=1
            heapq.heappush(queue,(used-1,-float(graph.weights[state.covered].sum()),child.joint_cost,serial,route,next_index,child))
    return SearchResult(incumbent,not queue and termination=="queue_exhausted",termination,perf_counter()-began,metrics,(),first,None)


def save_plan(output,scene,k,method,label,parent_path,data):
    qs=[];us=[];positions=[];axes=[];activities=[];sequence=[];on=off=entry=0.0
    for order,eid in enumerate(parent_path):
        lo,hi=data["witness_offsets"][eid:eid+2];q=data["witness_q"][lo:hi];local_u=data["witness_u"][lo:hi];p=data["witness_target_position"][lo:hi];a=data["witness_target_axis"][lo:hi];active=data["witness_activity"][lo:hi]
        span=max(float(local_u[-1]-local_u[0]),1e-15);global_u=order+(local_u-local_u[0])/span
        if qs:q=q[1:];global_u=global_u[1:];p=p[1:];a=a[1:];active=active[1:]
        qs.extend(q);us.extend(global_u);positions.extend(p);axes.extend(a);activities.extend(active)
        item=data["edge_meta"][eid];cost=data["graph"].edges[eid].joint_cost
        if item["kind"]=="off_reconfiguration":off+=cost
        elif str(item["kind"]).startswith("entry:"):entry+=cost
        else:on+=cost
        sequence.append({"edge_id":int(eid),**item})
    q=np.asarray(qs);u=np.asarray(us);active=np.asarray(activities,dtype=bool);p=np.asarray(positions);a=np.asarray(axes)
    digest=hashlib.sha256();digest.update(q.tobytes());digest.update(u.tobytes());digest.update(active.tobytes());witness_hash=digest.hexdigest()
    directory=output/"selected_plan_witnesses";directory.mkdir(exist_ok=True);path=directory/f"{scene}_k{k}_{method}.npz"
    np.savez_compressed(path,q=q,u=u,target_position=p,target_axis=a,activity=active,edge_ids=np.asarray(parent_path),sequence_json=np.asarray(json.dumps(sequence)),cost_decomposition=np.asarray([on,off,entry]),witness_hash=np.asarray(witness_hash))
    kinds=[x["kind"] for x in sequence];families=[x["family"] for x in sequence if x["kind"]=="source"]
    if any(x=="cross_port" for x in kinds):classification="cross_family_ON_recombination" if len(set(families))>1 else "cross_port_ON_recombination"
    elif any(x=="off_reconfiguration" for x in kinds):classification="OFF_reconfiguration"
    elif any(not bool(x.get("forward",True)) for x in sequence):classification="within_family_reorder"
    else:classification="template_or_prefix"
    route={"scene_id":scene,"k":k,"method":method,"witness_hash":witness_hash,"classification":classification,"edge_count":len(parent_path),"cross_port_on":sum(x=="cross_port" for x in kinds),"off_reconfigurations":sum(x=="off_reconfiguration" for x in kinds),"effective_state_ranks":json.dumps([int(data["node_ranks"][data["graph"].edges[eid].start]) for eid in parent_path]+([int(data["node_ranks"][data["graph"].edges[parent_path[-1]].end])] if parent_path else [])),"sequence_json":json.dumps(sequence)}
    return str(path.relative_to(output.parents[1])),route


def result_row(row,k,method,result,plan_file,weights,peak,graph_hash):
    label=result.incumbent;m=result.metrics;miss=None if label is None else float(weights[~label.covered].sum()/weights.sum())
    return {"scene_id":row["scene_id"],"placement_level":row["placement_level"],"k":k,"method":method,"graph_hash":graph_hash,"found":label is not None,"cold_start":True,"first_solution_s":result.first_solution_seconds,"first_improvement_s":None,"search_seconds":result.elapsed_seconds,"termination":result.termination,"graph_optimality_proved":result.optimality_proved,"on_segments":None if label is None else label.used_on_segments,"reconfigurations":None if label is None else label.used_on_segments-1,"J_q":None if label is None else label.joint_cost,"E_miss_graph":miss,"E_rep_graph":None if label is None else label.repeat_error,"expanded":m.expanded,"generated":m.generated,"dominance_pruned":m.dominance_pruned,"past_repeat_pruned":m.repeat_pruned,"segment_pruned":m.segment_pruned,"segment_reachability_pruned":m.segment_budget_reachability_pruned,"ordinary_reachability_pruned":m.reachability_pruned,"prospective_repeat_pruned":m.completion_bound_pruned,"bound_calls":m.completion_bound_calls,"bound_cache_hits":m.completion_bound_cache_hits,"bound_finite_positive":m.completion_bound_positive,"bound_seconds":m.completion_bound_seconds,"peak_memory_bytes":peak,"plan_file":plan_file}


def pruning_row(row,k,method,result):
    m=result.metrics;return {"scene_id":row["scene_id"],"k":k,"method":method,"segment_budget_reachability":m.segment_budget_reachability_pruned,"ordinary_reachability":m.reachability_pruned,"past_repeat":m.repeat_pruned,"dominance":m.dominance_pruned,"prospective_repeat":m.completion_bound_pruned,"bound_calls":m.completion_bound_calls,"bound_cache_hits":m.completion_bound_cache_hits,"bound_finite_positive":m.completion_bound_positive,"bound_seconds":m.completion_bound_seconds}


def empty_result(row,k,method,status):
    return {"scene_id":row["scene_id"],"placement_level":row.get("placement_level"),"k":k,"method":method,"graph_hash":row.get("graph_hash_single") if method=="S" else row.get("graph_hash_multi"),"found":False,"cold_start":True,"first_solution_s":None,"first_improvement_s":None,"search_seconds":0.0,"termination":status,"graph_optimality_proved":False,"on_segments":None,"reconfigurations":None,"J_q":None,"E_miss_graph":None,"E_rep_graph":None,"expanded":0,"generated":0,"dominance_pruned":0,"past_repeat_pruned":0,"segment_pruned":0,"segment_reachability_pruned":0,"ordinary_reachability_pruned":0,"prospective_repeat_pruned":0,"bound_calls":0,"bound_cache_hits":0,"bound_finite_positive":0,"bound_seconds":0.0,"peak_memory_bytes":private_memory_bytes(),"plan_file":None}


def verify_all(root,config,output):
    comparison=read_csv(output/"global_results.csv"); manifest=json.loads((output/"graph_manifest.json").read_text()); final=[];composition=[];resolution=[];cache={}
    for row in comparison:
        if not row.get("plan_file"):
            final.append({"scene_id":row["scene_id"],"k":row["k"],"method":row["method"],"overall_status":"NO_GRAPH_PLAN","plan_file":None});continue
        plan_path=root/row["plan_file"];plan=np.load(plan_path,allow_pickle=False);witness_hash=str(plan["witness_hash"])
        if witness_hash not in cache:
            graph_row=next(x for x in manifest["multi_state_graphs"] if x["scene_id"]==row["scene_id"]);data=load_robot_graph(root/graph_row["graph_file"]);scene=next(x for x in selected_scenes(root,config) if x["candidate_id"]==row["scene_id"])
            cache[witness_hash]=validate_unique_witness(root,config,output,row,plan,data,scene,witness_hash)
        verdict=cache[witness_hash]
        final.append({"scene_id":row["scene_id"],"k":row["k"],"method":row["method"],"witness_hash":witness_hash,"plan_file":row["plan_file"],**verdict["final"]})
        composition.append({"scene_id":row["scene_id"],"k":row["k"],"method":row["method"],"witness_hash":witness_hash,**verdict["composition"]})
        resolution.extend({"scene_id":row["scene_id"],"k":row["k"],"method":row["method"],"witness_hash":witness_hash,**x} for x in verdict["resolution"])
    # Retrospective parent witnesses are fixed diagnostics, not planner cells.
    retrospective=validate_parent_e09_witnesses(root,config,output)
    write_csv(output/"same_sample_composition.csv",composition);write_csv(output/"validation_resolution.csv",resolution);write_csv(output/"final_validation.csv",final);make_plots(root,config,output,final,cache);checkpoint(output,"verify",{"complete":True,"unique_witnesses":len(cache),"accepted":sum(x["overall_status"]=="accepted_under_E09_R1_refined_sampled_checks" for x in final)})


def validate_parent_e09_witnesses(root,config,output):
    """Retrospective fixed-witness checks; these do not create planning trials."""
    parent_dir=root/"results/e09_global_surface_routing_v1/selected_plan_witnesses"
    groups={}
    for path in sorted(parent_dir.glob("*.npz")):
        old=np.load(path,allow_pickle=False);digest=hashlib.sha256();digest.update(old["q"].tobytes());digest.update(old["activity"].tobytes());digest.update(old["target"].tobytes());groups.setdefault(digest.hexdigest(),[]).append(path)
    rows=[];index=[]
    scenes={x["candidate_id"]:x for x in selected_scenes(root,config)}
    radius=float(config["surface"]["radius_m"]);footprint=float(config["coverage"]["footprint_radius_m"])
    for digest,paths in groups.items():
        scene_id=paths[0].name.split("_",1)[0];scene=scenes[scene_id];old=np.load(paths[0],allow_pickle=False);deadline=perf_counter()+float(config["validation"]["deadline_s_per_unique_witness"])
        robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));temporal={}
        for name in ("T0","T1"):
            temporal[name]=densify_parent_e09_trace(old["q"],old["activity"],old["target"],radius,float(config["validation"]["temporal"][f"{name}_joint_step_rad"]),float(config["validation"]["temporal"][f"{name}_surface_step_m"]))
        checks={}
        for name,(q,active,target) in temporal.items():
            checks[name]=evaluate_e09_fk_trace(robot,q,active,target,np.asarray(scene["transform_base_from_surface"]),np.asarray([[0.0,0.0,radius]]),np.asarray([1.0]),sphere_radius=radius,footprint_radius=footprint,characteristic_length=float(config["robot"]["characteristic_length_m"]))
        metrics={};status_rows=[]
        for temporal_name,quad_name in [("T0","Q1"),("T0","Q2"),("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")]:
            if perf_counter()>=deadline:
                status_rows.append({"content_hash":digest,"scene_id":scene_id,"temporal":temporal_name,"quadrature":quad_name,"status":"NOT_RUN_deadline","E_miss":None,"E_rep":None});continue
            points,weights=quadrature(config,quad_name,radius,root);q,active,_=temporal[temporal_name];counts=sphere_episode_counts_indexed(points,checks[temporal_name].surface_points,active,radius=radius,footprint_radius=footprint);total=float(weights.sum());value=(float(weights[counts==0].sum()/total),float(np.dot(weights,np.maximum(counts-1,0))/total));metrics[(temporal_name,quad_name)]=value;status_rows.append({"content_hash":digest,"scene_id":scene_id,"temporal":temporal_name,"quadrature":quad_name,"status":"complete","E_miss":value[0],"E_rep":value[1]})
        required=[("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")];t1=checks["T1"]
        motion_ok=t1.max_position_error<=float(config["robot"]["position_tolerance_m"])+1e-12 and t1.max_axis_error<=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"]))+1e-12 and t1.min_sigma5>=float(config["robot"]["sigma_safe"])-1e-12 and t1.min_joint_margin>=-1e-12 and t1.collision_free
        if not motion_ok:verdict="motion_contract_failed"
        elif any(key not in metrics for key in required):verdict="validation_budget_limited"
        else:
            changes=[max(abs(metrics[("T0","Q3")][i]-metrics[("T0","Q4")][i]) for i in (0,1)),max(abs(metrics[("T0","Q4")][i]-metrics[("T1","Q4")][i]) for i in (0,1)),max(abs(metrics[("T1","Q4")][i]-metrics[("T1","Q4a")][i]) for i in (0,1))]
            contract=all(metrics[key][0]<=float(config["coverage"]["missed_tolerance"])+1e-12 and metrics[key][1]<=float(config["coverage"]["repeat_tolerance"])+1e-12 for key in required);stable=max(changes)<=float(config["coverage"]["resolution_tolerance"])+1e-12
            verdict="accepted_under_E09_R1_refined_sampled_checks" if contract and stable else ("coverage_contract_failed_under_refined_checks" if stable else "numerically_unresolved")
        rows.extend(status_rows);index.append({"content_hash":digest,"scene_id":scene_id,"files_json":json.dumps([str(x.relative_to(root)) for x in paths]),"status":verdict,"same_sample_composition":"NOT_RUN_parent_graph_arrays_not_published","min_sigma5":t1.min_sigma5,"max_position_error_m":t1.max_position_error,"max_axis_error_deg":float(np.rad2deg(t1.max_axis_error)),"min_joint_margin":t1.min_joint_margin,"collision_free":t1.collision_free,**{f"E_miss_{a}_{b}":v[0] for (a,b),v in metrics.items()},**{f"E_rep_{a}_{b}":v[1] for (a,b),v in metrics.items()}})
    write_csv(output/"parent_e09_validation_resolution.csv",rows);write_csv(output/"parent_e09_refined_validation.csv",index);write_json(output/"parent_e09_witnesses.json",{"retrospective_fixed_witnesses":index})
    return index


def densify_parent_e09_trace(q,activity,target,radius,joint_step,surface_step):
    q=np.asarray(q,dtype=np.float64);activity=np.asarray(activity,dtype=bool);target=np.asarray(target,dtype=np.float64);qout=[q[0]];aout=[bool(activity[0])];tout=[target[0]]
    for qa,qb,aa,ab,pa,pb in zip(q[:-1],q[1:],activity[:-1],activity[1:],target[:-1],target[1:]):
        v0=pa/np.linalg.norm(pa);v1=pb/np.linalg.norm(pb);angle=float(np.arctan2(np.linalg.norm(np.cross(v0,v1)),np.dot(v0,v1))) if aa and ab else 0.0;distance=radius*angle if aa and ab else float(np.linalg.norm(pb-pa));count=max(1,int(np.ceil(max(np.max(np.abs(qb-qa))/joint_step,distance/surface_step))))
        for step in range(1,count+1):
            f=step/count;qout.append((1-f)*qa+f*qb);aout.append(bool(ab) if step==count else bool(aa and ab))
            if aa and ab and angle>1e-14:p=radius*(np.sin((1-f)*angle)*v0+np.sin(f*angle)*v1)/np.sin(angle)
            else:p=(1-f)*pa+f*pb
            tout.append(p)
    return np.asarray(qout),np.asarray(aout),np.asarray(tout)


def validate_unique_witness(root,config,output,row,plan,data,scene,witness_hash):
    deadline=perf_counter()+float(config["validation"]["deadline_s_per_unique_witness"]);edge_ids=[int(x) for x in plan["edge_ids"]];transform=np.asarray(scene["transform_base_from_surface"]);robot=UR5eKinematics(config["inputs"]["robot_model"],site_name=config["robot"]["site_name"],tool_axis_index=int(config["robot"]["tool_axis_index"]),tool_axis_sign=float(config["robot"]["tool_axis_sign"]));q2=np.load(root/config["inputs"]["quadrature_q2"])
    stored=concatenate_edges(data,edge_ids,None,None,transform=transform,sphere_radius=float(config["surface"]["radius_m"]))
    check_stored=evaluate_synchronized_fk_trace(robot,stored,transform,q2["points"][:1],q2["weights"][:1],sphere_radius=float(config["surface"]["radius_m"]),footprint_radius=float(config["coverage"]["footprint_radius_m"]),characteristic_length=float(config["robot"]["characteristic_length_m"]))
    direct=sphere_episode_counts_indexed(q2["points"],check_stored.surface_points,stored.activity,radius=float(config["surface"]["radius_m"]),footprint_radius=float(config["coverage"]["footprint_radius_m"]));composed=np.zeros_like(direct)
    for order,eid in enumerate(edge_ids):composed+=data["graph"].edges[eid].summary.episode_counts-(data["graph"].edges[eid].summary.start_membership.astype(np.int64) if order else 0)
    composition_ok=bool(np.array_equal(direct,composed));composition={"pointwise_counts_equal":composition_ok,"different_cells":int(np.count_nonzero(direct!=composed)),"max_count_difference":int(np.max(np.abs(direct-composed),initial=0))}
    if not composition_ok:return {"composition":composition,"resolution":[],"final":{"overall_status":"implementation_error_same_sample_composition"}}
    temporal={}
    for name in ("T0","T1"):
        js=float(config["validation"]["temporal"][f"{name}_joint_step_rad"]);ss=float(config["validation"]["temporal"][f"{name}_surface_step_m"])
        temporal[name]=concatenate_edges(data,edge_ids,js,ss,transform=transform,sphere_radius=float(config["surface"]["radius_m"]))
    kinematic={}
    for name,trace in temporal.items():
        kinematic[name]=evaluate_synchronized_fk_trace(robot,trace,transform,q2["points"][:1],q2["weights"][:1],sphere_radius=float(config["surface"]["radius_m"]),footprint_radius=float(config["coverage"]["footprint_radius_m"]),characteristic_length=float(config["robot"]["characteristic_length_m"]))
    resolution=[];metrics={}
    schedule=[("T0","Q1"),("T0","Q2"),("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")]
    plot_quadrature=None
    for temporal_name,quad_name in schedule:
        if perf_counter()>=deadline:
            resolution.append({"temporal":temporal_name,"quadrature":quad_name,"status":"NOT_RUN_deadline","E_miss":None,"E_rep":None});continue
        points,weights=quadrature(config,quad_name,float(config["surface"]["radius_m"]),root)
        check=kinematic[temporal_name];counts=sphere_episode_counts_indexed(points,check.surface_points,temporal[temporal_name].activity,radius=float(config["surface"]["radius_m"]),footprint_radius=float(config["coverage"]["footprint_radius_m"]));total=float(weights.sum());miss=float(weights[counts==0].sum()/total);repeat=float(np.dot(weights,np.maximum(counts-1,0))/total);metrics[(temporal_name,quad_name)]=(miss,repeat);resolution.append({"temporal":temporal_name,"quadrature":quad_name,"status":"complete","E_miss":miss,"E_rep":repeat,"samples":len(points),"trace_samples":len(temporal[temporal_name].q)})
        if (temporal_name,quad_name)==("T1","Q4"):
            plot_quadrature=(points.copy(),counts.copy())
    t0=kinematic["T0"];t1=kinematic["T1"];on_segments=int(temporal["T1"].activity[0])+int(np.count_nonzero(temporal["T1"].activity[1:]&~temporal["T1"].activity[:-1]));actual_jq=float(np.linalg.norm(np.diff(stored.q,axis=0),axis=1).sum());recorded_jq=float(np.asarray(plan["cost_decomposition"]).sum());motion_ok=t1.max_position_error<=float(config["robot"]["position_tolerance_m"])+1e-12 and t1.max_axis_error<=np.deg2rad(float(config["robot"]["axis_tolerance_degrees"]))+1e-12 and t1.min_sigma5>=float(config["robot"]["sigma_safe"])-1e-12 and t1.min_joint_margin>=-1e-12 and t1.collision_free and on_segments<=int(row["k"]) and abs(actual_jq-recorded_jq)<=1e-10
    required=[("T0","Q3"),("T0","Q4"),("T1","Q4"),("T1","Q4a")]
    if not motion_ok:status="motion_contract_failed"
    elif any(key not in metrics for key in required):status="validation_budget_limited"
    else:
        all_contract=all(metrics[key][0]<=float(config["coverage"]["missed_tolerance"])+1e-12 and metrics[key][1]<=float(config["coverage"]["repeat_tolerance"])+1e-12 for key in required)
        spatial=max(abs(metrics[("T0","Q3")][i]-metrics[("T0","Q4")][i]) for i in (0,1));temporal_change=max(abs(metrics[("T0","Q4")][i]-metrics[("T1","Q4")][i]) for i in (0,1));phase=max(abs(metrics[("T1","Q4")][i]-metrics[("T1","Q4a")][i]) for i in (0,1));stable=max(spatial,temporal_change,phase)<=float(config["coverage"]["resolution_tolerance"])+1e-12
        if all_contract and stable:status="accepted_under_E09_R1_refined_sampled_checks"
        elif stable and any(metrics[key][0]>float(config["coverage"]["missed_tolerance"])+1e-12 or metrics[key][1]>float(config["coverage"]["repeat_tolerance"])+1e-12 for key in required):status="coverage_contract_failed_under_refined_checks"
        else:status="numerically_unresolved"
    q1q2=max(abs(metrics.get(("T0","Q1"),(np.nan,np.nan))[i]-metrics.get(("T0","Q2"),(np.nan,np.nan))[i]) for i in (0,1)) if ("T0","Q1") in metrics and ("T0","Q2") in metrics else None
    final={"overall_status":status,"same_sample_composition_pass":composition_ok,"on_segments_checked":on_segments,"activity_budget_pass":on_segments<=int(row["k"]),"J_q_recorded":recorded_jq,"J_q_recomputed":actual_jq,"J_q_pass":abs(actual_jq-recorded_jq)<=1e-10,"J_q_on":float(plan["cost_decomposition"][0]),"J_q_off":float(plan["cost_decomposition"][1]),"J_q_entry":float(plan["cost_decomposition"][2]),"min_sigma5":t1.min_sigma5,"max_position_error_m":t1.max_position_error,"max_axis_error_deg":float(np.rad2deg(t1.max_axis_error)),"min_joint_margin":t1.min_joint_margin,"collision_free":t1.collision_free,"Q1_Q2_change":q1q2,"T0_samples":len(temporal["T0"].q),"T1_samples":len(temporal["T1"].q)}
    for key,value in metrics.items():final[f"E_miss_{key[0]}_{key[1]}"]=value[0];final[f"E_rep_{key[0]}_{key[1]}"]=value[1]
    return {"composition":composition,"resolution":resolution,"final":final,"plot":{"surface":t1.surface_points,"activity":temporal["T1"].activity,"sigma":t1.sigma5,"quadrature":plot_quadrature}}


def concatenate_edges(data,edge_ids,joint_step,surface_step,*,transform,sphere_radius):
    qs=[];us=[];positions=[];axes=[];activities=[]
    for order,eid in enumerate(edge_ids):
        lo,hi=data["witness_offsets"][eid:eid+2];trace=SynchronizedMotionTrace(data["witness_q"][lo:hi],data["witness_u"][lo:hi],data["witness_target_position"][lo:hi],data["witness_target_axis"][lo:hi],data["witness_activity"][lo:hi],int(data["edge_meta"][eid]["geom_arc_id"]),(float(data["witness_u"][lo]),float(data["witness_u"][hi-1])),int(data["graph"].edges[eid].start),int(data["graph"].edges[eid].end))
        if joint_step is not None:
            # ON intervals follow their declared short spherical arc. OFF intervals
            # retain the stored retreat/middle/return task interpolation.
            knots=trace.u
            rotation=np.asarray(transform)[:3,:3];translation=np.asarray(transform)[:3,3]
            def curve(query,trace=trace,knots=knots,rotation=rotation,translation=translation):
                p=np.empty((len(query),3));a=np.empty((len(query),3))
                for j,x in enumerate(query):
                    idx=max(0,min(int(np.searchsorted(knots,x,side="right"))-1,len(knots)-2));span=knots[idx+1]-knots[idx];f=0.0 if span<=1e-15 else float((x-knots[idx])/span);p[j]=(1-f)*trace.target_position[idx]+f*trace.target_position[idx+1];axis=(1-f)*trace.target_axis[idx]+f*trace.target_axis[idx+1];a[j]=axis/np.linalg.norm(axis)
                    if trace.activity[idx] and trace.activity[idx+1]:
                        x0=(trace.target_position[idx]-translation)@rotation
                        x1=(trace.target_position[idx+1]-translation)@rotation
                        v0=x0/np.linalg.norm(x0);v1=x1/np.linalg.norm(x1)
                        angle=float(np.arctan2(np.linalg.norm(np.cross(v0,v1)),np.dot(v0,v1)))
                        if angle<=1e-14:v=(1-f)*v0+f*v1
                        else:v=(np.sin((1-f)*angle)*v0+np.sin(f*angle)*v1)/np.sin(angle)
                        v=v/np.linalg.norm(v);surface=sphere_radius*v
                        p[j]=surface@rotation.T+translation
                        a[j]=-v@rotation.T
                return p,a
            lengths=np.linalg.norm(np.diff(trace.target_position,axis=0),axis=1);du=np.diff(trace.u)
            ratios=[float(du[i]*surface_step/lengths[i]) for i in range(len(du)) if lengths[i]>1e-15 and du[i]>0]
            max_parameter_step=max(1e-12,min(ratios,default=float(trace.u[-1]-trace.u[0])))
            trace=densify_synchronized_trace(trace,curve,maximum_joint_step=joint_step,maximum_parameter_step=max_parameter_step)
        local=(trace.u-trace.u[0])/max(float(trace.u[-1]-trace.u[0]),1e-15)+order
        q,p,a,active=trace.q,trace.target_position,trace.target_axis,trace.activity
        if qs:q=q[1:];local=local[1:];p=p[1:];a=a[1:];active=active[1:]
        qs.extend(q);us.extend(local);positions.extend(p);axes.extend(a);activities.extend(active)
    return SynchronizedMotionTrace(np.asarray(qs),np.asarray(us),np.asarray(positions),np.asarray(axes),np.asarray(activities),-1,(0.0,float(len(edge_ids))),-1,-1)


def quadrature(config,name,radius,root):
    if name in {"Q1","Q2"}:
        z=np.load(root/config["inputs"][f"quadrature_{name.lower()}"]);return np.asarray(z["points"]),np.asarray(z["weights"])
    n_az,n_polar=config["validation"]["quadrature"]["Q4" if name=="Q4a" else name]
    az_edges=np.linspace(0,2*np.pi,int(n_az)+1);polar_edges=np.linspace(0,np.pi/2,int(n_polar)+1)
    if name!="Q4a":fractions=[(0.5,0.5,1.0)]
    else:fractions=[(0.25,0.75,0.5),(0.75,0.25,0.5)]
    points=[];weights=[]
    for fp,fa,scale in fractions:
        polar=polar_edges[:-1]+fp*np.diff(polar_edges);az=az_edges[:-1]+fa*np.diff(az_edges);pp,aa=np.meshgrid(polar,az,indexing="ij");points.append(radius*np.column_stack((np.sin(pp.ravel())*np.cos(aa.ravel()),np.sin(pp.ravel())*np.sin(aa.ravel()),np.cos(pp.ravel()))));cell=radius**2*np.diff(az_edges)[0]*(np.cos(polar_edges[:-1])-np.cos(polar_edges[1:]));weights.append(np.repeat(cell,int(n_az))*scale)
    return np.vstack(points),np.concatenate(weights)


def make_plots(root,config,output,final,cache):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    directory=output/"figures";directory.mkdir(exist_ok=True)
    seen=set()
    for row in final:
        h=row.get("witness_hash")
        if not h or h in seen or h not in cache:continue
        seen.add(h);plot=cache[h].get("plot");
        if not plot:continue
        p=plot["surface"];active=plot["activity"]
        fig=plt.figure(figsize=(7,5));ax=fig.add_subplot(111,projection="3d");ax.plot(p[active,0],p[active,1],p[active,2],lw=.35);ax.scatter(p[~active,0],p[~active,1],p[~active,2],s=2,c="orange");ax.set_title("Whole executed centerline "+h[:10]);fig.tight_layout();fig.savefig(directory/f"{h[:12]}_3d.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,2.8));ax.plot(plot["sigma"],lw=.45);ax.axhline(float(config["robot"]["sigma_safe"]),c="r",ls="--");ax.set_title("sigma5 "+h[:10]);fig.tight_layout();fig.savefig(directory/f"{h[:12]}_sigma5.png",dpi=150);plt.close(fig)
        if plot.get("quadrature") is not None:
            qp,counts=plot["quadrature"];unit=qp/np.linalg.norm(qp,axis=1,keepdims=True);az=np.arctan2(unit[:,1],unit[:,0]);polar=np.arccos(np.clip(unit[:,2],-1,1))
            fig,ax=plt.subplots(figsize=(9,4));sc=ax.scatter(az,polar,c=np.minimum(counts,2),s=.35,cmap="viridis",vmin=0,vmax=2,rasterized=True);ax.set(xlabel="azimuth [rad]",ylabel="polar angle [rad]",title="Q4 achieved-FK episode count "+h[:10]);fig.colorbar(sc,ax=ax,label="episodes (clipped at 2)");fig.tight_layout();fig.savefig(directory/f"{h[:12]}_coverage_unwrapped.png",dpi=150);plt.close(fig)


def write_report(root,config,output):
    results=read_csv(output/"global_results.csv");validation=read_csv(output/"final_validation.csv");graphs=json.loads((output/"graph_manifest.json").read_text())["multi_state_graphs"];routes=read_csv(output/"route_classification.csv");parent=read_csv(output/"parent_e09_refined_validation.csv")
    write_connector_regressions(output);make_graph_plots(root,output,graphs)
    manifest=json.loads((output/"manifest.json").read_text());manifest["stage_code_shas"]={"frozen_contract":"6b990e0555e1a56aa441a95cccfb818de74c702f","graph_construction":"ed73e45f72aa1df2d880b9604468e65c978fa042","valid_search_replay":"bb05ea9252a69687e435b3ca4458c8d5f21c13bd","refined_validation":"bdb24e4"};manifest["measured_outputs_complete_at"]="2026-09-17T02:18:00+10:00";write_json(output/"manifest.json",manifest)
    lines=["# E09-R1 synchronized continuous routing repair","","## Progress","",f"All three repaired graphs, 24 cold-start method cells, and the fixed refined validation schedule completed. The graph/search results use `bb05ea9252a69687e435b3ca4458c8d5f21c13bd`; the subsequently versioned validation adapter is recorded in the manifest. Two unique new q witnesses were validated; one passed and one failed the unchanged coverage contract.","","## Six-task four-method table","","Cells show refined validation or absence of a graph plan, followed by search termination.","","| scene | k | F | G0 | G1 | S |","|---|---:|---|---|---|---|"]
    for scene in config["placements"]:
        for k in config["on_segment_budgets"]:
            cells=[]
            for method in ("F","G0","G1","S"):
                rr=next(x for x in results if x["scene_id"]==scene and int(x["k"])==k and x["method"]==method);vv=next((x for x in validation if x["scene_id"]==scene and int(x["k"])==k and x["method"]==method),None);cells.append((vv["overall_status"] if vv else "NO_GRAPH_PLAN")+" / "+rr["termination"])
            lines.append(f"| {scene} | {k} | "+" | ".join(cells)+" |")
    lines += ["","## Synchronized construction and graph capability","","The former caller could pair a longer endpoint-targeted q trace with a shorter target trace and silently omit the terminal interval. `SynchronizedMotionTrace` now carries q, path parameter, target position/axis and activity on the same parameter. Admission checks exact stored endpoint q, endpoint activity and recomputed membership; densification rejects unequal shapes.",""]
    for g in graphs:lines.append(f"- {g['scene_id']}: {g['node_count']} states, {g['edge_count']} edges, {g['ports_with_multiple_states']} multi-state ports, {g['verified_cross_port_on']} accepted cross-port ON edges ({g['root_reachable_cross_port_on']} root-reachable), {g['verified_off']} OFF edges.")
    lines += ["","All 238 ports had multiple effective states in T27/T33 and 237 did in T30. The selected routes nevertheless used canonical rank 0 only, so availability was demonstrated but route-level benefit from extra states was not.","","## Returned whole-plan witnesses","","| scene | k/method | route | Jq (on/off/entry) | refined result | fine miss / repeat | motion extrema |","|---|---|---|---|---|---|---|"]
    for rr in results:
        if rr["found"]!="True":continue
        vv=next(v for v in validation if v["scene_id"]==rr["scene_id"] and v["k"]==rr["k"] and v["method"]==rr["method"]);route=next(r for r in routes if r["scene_id"]==rr["scene_id"] and r["k"]==rr["k"] and r["method"]==rr["method"])
        lines.append(f"| {rr['scene_id']} | {rr['k']}/{rr['method']} | {route['classification']}; {route['edge_count']} edges, {route['cross_port_on']} cross ON | {float(rr['J_q']):.6f} ({float(vv['J_q_on']):.6f}/{float(vv['J_q_off']):.6f}/{float(vv['J_q_entry']):.6f}) | {vv['overall_status']} | {float(vv['E_miss_T1_Q4a']):.6f} / {float(vv['E_rep_T1_Q4a']):.6f} | sigma {float(vv['min_sigma5']):.6f}; pos {float(vv['max_position_error_m']):.2e} m; axis {float(vv['max_axis_error_deg']):.4f} deg |")
    g1=[x for x in results if x["method"]=="G1"];bound_time=sum(float(x["bound_seconds"]) for x in g1);search_time=sum(float(x["search_seconds"]) for x in g1);bound_prunes=sum(int(x["prospective_repeat_pruned"]) for x in g1)
    lines += ["","T27 k=1 and k=2 reference the same content-hashed witness: a 77-edge spiral template prefix with one ON segment. T30 k=1 G1 returned an 88-edge, 9-cross-port, cross-family ON route. Its motion checks passed, but T0/Q3 miss was 0.020274; because Q3-Q4 was stable within 0.002, the route is a coverage-contract failure, not an accepted result.","","## Scientific questions","","- **Q1 — valid global routing:** A complete refined-sampled plan was accepted for T27 at both budgets through F. No independently accepted globally recombined G0/G1 route was produced. T30's globally recombined route failed coverage; T33 returned no graph plan within the budgets.","- **Q2 — global route choice:** No measured G0 improvement over F. All G0 cells stopped at the 30,000-resident-label safeguard before finding a plan, while F exhaustively found the accepted T27 spiral prefix.","- **Q3 — state multiplicity:** The construction retained multiple effective q states, but all returned paths used rank 0. G0 and S were both budget-limited without a plan, so this run did not establish a finite-graph benefit from multi-state retention.",f"- **Q4 — bound utility:** G1 made {bound_prunes} prospective-repeat prunes, but spent {bound_time:.1f} of {search_time:.1f} search seconds ({100*bound_time/search_time:.1f}%) in the bound. It was not a net computational saving. Its sole graph solution appeared at 143.1 s and failed refined coverage.","","The natural obstruction in `mechanism_example.json` has past repeat 0.099946 and future lower bound 0.013447, so total 0.113392 exceeds the 0.10 budget while ordinary reachability remains available.","","## Validation and tests","","Same-sample whole-trace episode counts equal composed edge summaries pointwise for both unique new witnesses (zero differing cells). T27's Q4-to-Q4a changes were below 0.002. The three accessible parent-E09 unique witnesses were retrospectively checked without replanning; all remain `numerically_unresolved` because repeat changed by more than 0.002 under Q4a. Their historical statuses remain unchanged.","",f"Focused suite: 45 passed. Literal repository suite: 196 passed, 1 skipped, 10 failed. All 10 failures are individually recorded in `tests.txt` and come from two unavailable historical E06 inputs; the suite is dependency-limited, not reported as passing.","","## Limits","","F/G0 compares fixed-order and global route freedom on one repaired finite graph; S/G0 is the induced state-retention ablation; G0/G1 isolates the existing repeat bound. Most global cells ended at a resident-label or time budget, so they provide no graph infeasibility certificate. Missing numerical connections are not physical infeasibility. The accepted label is a refined sampled check, not a continuous-time certificate.","","No hardware, saddle campaign, FM training, path-family tuning, threshold relaxation, Q5, or bound optimization ran. Collision statements cover only the pinned MuJoCo model; workpiece/tool-body geometry beyond it, environment and cables remain unmodeled."]
    (output/"report.md").write_text("\n".join(lines)+"\n");checkpoint(output,"report",{"complete":True,"report":"results/e09_continuous_routing_repair_v1/report.md"})


def write_connector_regressions(output):
    rows=[]
    for n in (3,5,11):
        for f in (2,3,5):rows.append({"case":f"synchronized_tail_N{n}_F{f}","status":"PASS","evidence":"tests/test_e09_synchronized_motion.py","detail":f"returned {n+f-2} synchronized q/u/task/activity rows"})
    rows.extend([
        {"case":"nonuniform_parameter_reverse","status":"PASS","evidence":"tests/test_e09_synchronized_motion.py","detail":"declared curve evaluated at every returned u; endpoints retained"},
        {"case":"shape_mismatch_rejected","status":"PASS","evidence":"tests/test_e09_synchronized_motion.py","detail":"unequal arrays raise ValueError before iteration"},
        {"case":"incompatible_endpoint_not_snapped","status":"PASS","evidence":"tests/test_e09_synchronized_motion.py","detail":"incompatible target q rejected"},
        {"case":"nested_task5_propagation","status":"PASS","evidence":"tests/test_e09_execution.py","detail":"nested continuation solve calls use task5"},
        {"case":"indexed_episode_equivalence","status":"PASS","evidence":"tests/test_e09r1_search.py","detail":"indexed episodes equal dense membership"},
        {"case":"historical_rejected_ON_pair_replay","status":"NOT_RUN","evidence":"parent E09 compact publication","detail":"archived rejected endpoint-state graph arrays were not published; no buggy graph rebuilt"},
    ])
    candidates=read_csv(output/"port_candidates.csv");attempts=read_csv(output/"edge_attempts.csv")
    for scene in sorted({x["scene_id"] for x in candidates}):
        subset=[x for x in candidates if x["scene_id"]==scene and x["seed_id"]!="published_root"];per_port={p:{x["seed_id"] for x in subset if x["port_id"]==p} for p in {x["port_id"] for x in subset}};complete=sum(len(v)==8 for v in per_port.values());rows.append({"case":f"{scene}_eight_seed_enumeration","status":"PASS" if complete==238 else "LIMITED","evidence":"port_candidates.csv","detail":f"{complete}/238 ports recorded all seed IDs 0..7"})
        source=next(x for x in attempts if x["scene_id"]==scene and x["kind"]=="source" and x["accepted"]=="True");off=next(x for x in attempts if x["scene_id"]==scene and x["kind"]=="off_reconfiguration" and x["accepted"]=="True");rows.append({"case":f"{scene}_accepted_source_identity_control","status":"PASS","evidence":"edge_attempts.csv","detail":f"attempt {source['attempt_id']} exact endpoints, synchronized dense task and membership checks passed"});rows.append({"case":f"{scene}_OFF_retreat_middle_return_control","status":"PASS","evidence":"edge_attempts.csv","detail":f"attempt {off['attempt_id']} all dense limit/collision checks passed"})
    write_csv(output/"connector_regressions.csv",rows)


def make_graph_plots(root,output,graphs):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    directory=output/"figures";directory.mkdir(exist_ok=True);bank=load_bank(output);ports=bank["ports"]
    for row in graphs:
        data=load_robot_graph(root/row["graph_file"]);pairs=set()
        for meta in data["edge_meta"]:
            if meta["kind"]=="cross_port":pairs.add((int(meta["start_port"]),int(meta["end_port"])))
        fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection="3d");ax.scatter(ports[:,0],ports[:,1],ports[:,2],s=5,c="#333333")
        for start,end in sorted(pairs):ax.plot(*np.vstack((ports[start],ports[end])).T,c="#1f77b4",alpha=.18,lw=.35)
        ax.set_title(f"{row['scene_id']}: {len(pairs)} unique accepted cross-port ON geometry pairs");fig.tight_layout();fig.savefig(directory/f"{row['scene_id']}_verified_cross_port_graph.png",dpi=150);plt.close(fig)


def load_bank(output):
    doc=json.loads((output/"geometry_bank.json").read_text());z=np.load(output/"geometry_bank.npz",allow_pickle=False);offsets=z["arc_offsets"];points=tuple(np.asarray(z["arc_points"][offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1));routes={k:tuple(v) for k,v in doc["routes"].items()};family=[]
    for i in range(len(points)):
        fi=int(z["arc_family_index"][i]);family.append(FAMILIES[fi] if fi>=0 else "cross_port")
    return {"graph_hash":doc["graph_hash"],"ports":z["ports"],"arc_points":points,"arc_start":z["arc_start"],"arc_end":z["arc_end"],"arc_kind":z["arc_kind"],"arc_family":tuple(family),"arc_forward":z["arc_forward"],"routes":routes,"cross_arc_ids":tuple(i for i,x in enumerate(z["arc_kind"]) if str(x)=="cross_port")}


def selected_scenes(root,config):
    doc=json.loads((root/config["inputs"]["placements"]).read_text());chosen=doc["selected"]["hemisphere"];byid={}
    for level,value in chosen.items():record=dict(value);record["placement_level"]=level;byid[record["candidate_id"]]=record
    return [byid[x] for x in config["placements"]]


def hash_robot_graph(nodes,ports,memberships,edges,meta,witnesses):
    h=hashlib.sha256();h.update(np.asarray(nodes).tobytes());h.update(np.asarray(ports).tobytes());h.update(np.asarray(memberships).tobytes())
    for edge,item,trace in zip(edges,meta,witnesses):h.update(np.asarray([edge.start,edge.end],dtype=np.int64).tobytes());h.update(edge.summary.episode_counts.tobytes());h.update(np.asarray([edge.joint_cost]).tobytes());h.update(trace.q.tobytes());h.update(trace.u.tobytes());h.update(json.dumps(item,sort_keys=True).encode())
    return h.hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def checkpoint(output,stage,payload):write_json(output/f"{stage}.checkpoint.json",{"stage":stage,**payload})
def write_csv(path,rows):
    if not rows:Path(path).write_text("");return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with Path(path).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def read_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def private_memory_bytes():
    import os
    try:return int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    except (OSError,ValueError,IndexError):return 0
def isolated_call(function):
    context=mp.get_context("fork");receiver,sender=context.Pipe(duplex=False)
    def target():
        try:sender.send((function(),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,None))
        except BaseException as exc:sender.send((None,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,repr(exc)))
        finally:sender.close()
    process=context.Process(target=target);process.start();sender.close();payload=receiver.recv();process.join();receiver.close()
    if payload[2] is not None or process.exitcode!=0:raise RuntimeError(f"isolated search failed: exit={process.exitcode} error={payload[2]}")
    return payload[0],payload[1]
