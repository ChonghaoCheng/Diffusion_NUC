#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion_coverage.coverage.evaluator import _length_and_sources
from diffusion_coverage.coverage.nuc_evaluator import evaluate_nuc_coverage
from diffusion_coverage.coverage.ordered_trace_evaluator import (
    evaluate_ordered_trace_mesh,
    evaluate_ordered_trace_reference,
    make_analytical_quadrature,
    reference_membership_states,
    resample_prescribed_path,
)
from diffusion_coverage.coverage.patterns import raster_pattern, spiral_pattern
from diffusion_coverage.diagnostics.symmetry_layout import analytical_points_and_normals
from diffusion_coverage.surface.primitives import make_hemisphere, make_saddle
from diffusion_coverage.surface.projection import project_points


STAGES = ("freeze", "tests", "geometry", "readiness", "report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen E08-R1 path-semantics audit")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e08_path_semantics_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/e08_path_semantics_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    functions = {
        "freeze": stage_freeze,
        "tests": stage_tests,
        "geometry": stage_geometry,
        "readiness": stage_readiness,
        "report": stage_report,
    }
    functions[args.stage](config, output)


def stage_freeze(config: dict[str, Any], output: Path) -> None:
    if (output / "geometry.checkpoint.json").exists():
        raise RuntimeError("geometry results already exist; frozen inputs cannot be replaced")
    e08 = json.loads((ROOT / config["inputs"]["e08_config"]).read_text())
    historical = json.loads((ROOT / config["inputs"]["historical_surface_config"]).read_text())
    arrays: dict[str, np.ndarray] = {}
    records = []
    index = 0
    for surface_id in ("saddle", "hemisphere"):
        surface = make_surface(surface_id, historical, config)
        for family in config["candidates"][surface_id]:
            proposal = make_proposal(surface, family, config)
            keys = []
            activity_keys = []
            digest = hashlib.sha256()
            for segment_index, path in enumerate(proposal.plan.active_paths()):
                path_key = f"candidate_{index}_path_{segment_index}"
                activity_key = f"candidate_{index}_activity_{segment_index}"
                activity = np.ones(len(path), dtype=bool)
                arrays[path_key] = np.asarray(path, dtype=np.float64)
                arrays[activity_key] = activity
                keys.append(path_key)
                activity_keys.append(activity_key)
                update_array_hash(digest, arrays[path_key])
                update_array_hash(digest, activity)
            records.append(
                {
                    "candidate_index": index,
                    "candidate_id": f"{surface_id}/{family['name']}",
                    "surface_id": surface_id,
                    "family": family["name"],
                    "max_segments": int(family["max_segments"]),
                    "overlap": float(family["overlap"]),
                    "path_keys": keys,
                    "activity_keys": activity_keys,
                    "segment_count": len(keys),
                    "candidate_sha256": digest.hexdigest(),
                    "generator": "pinned E08 raster_pattern/spiral_pattern",
                    "waypoint_spacing_m": 0.75 * config["coverage"]["footprint_radius_m"],
                }
            )
            index += 1
    candidate_file = output / "frozen_candidates.npz"
    np.savez_compressed(candidate_file, **arrays)

    quadrature_records = []
    for surface_id in ("saddle", "hemisphere"):
        metadata = historical["surfaces"][surface_id]
        for level in ("Q0", "Q1", "Q2"):
            quadrature = make_analytical_quadrature(surface_id, metadata, level)
            path = output / f"quadrature_{surface_id}_{level}.npz"
            np.savez_compressed(path, points=quadrature.points, weights=quadrature.weights, parameters=quadrature.parameters)
            quadrature_records.append(
                {
                    "surface_id": surface_id,
                    "level": level,
                    "definition": quadrature.definition,
                    "samples": len(quadrature.points),
                    "area": float(quadrature.weights.sum()),
                    "file": str(path.relative_to(ROOT)),
                    "sha256": file_hash(path),
                }
            )
    write_json(output / "quadrature_definitions.json", {"frozen_before_geometry": True, "quadratures": quadrature_records})
    write_json(
        output / "frozen_candidate_manifest.json",
        {
            "frozen_before_geometry": True,
            "candidate_file": str(candidate_file.relative_to(ROOT)),
            "candidate_file_sha256": file_hash(candidate_file),
            "candidates": records,
        },
    )
    inputs = {}
    for name, relative in config["inputs"].items():
        path = ROOT / relative
        inputs[name] = {"path": relative, "sha256": file_hash(path)}
    write_json(
        output / "freeze_manifest.json",
        {
            "registered_at": config["registered_at"],
            "executed_at": timestamp(),
            "parent_code_sha": config["parent_code_sha"],
            "parent_ara_sha": config["parent_ara_sha"],
            "config": str((ROOT / "configs/e08_path_semantics_v1.json").relative_to(ROOT)),
            "config_sha256": file_hash(ROOT / "configs/e08_path_semantics_v1.json"),
            "inputs": inputs,
            "candidate_manifest_sha256": file_hash(output / "frozen_candidate_manifest.json"),
            "quadrature_definitions_sha256": file_hash(output / "quadrature_definitions.json"),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    )
    checkpoint(output, "freeze", {"complete": True, "candidates": len(records), "quadratures": len(quadrature_records)})


def stage_tests(config: dict[str, Any], output: Path) -> None:
    require_checkpoint(output, "freeze")
    python = sys.executable
    commands = [
        [python, "-m", "pytest", "-q", "tests/test_ordered_trace_evaluator.py", "tests/test_history_search.py", "tests/test_completion_bound.py"],
        [python, "-m", "pytest", "-q"],
    ]
    sections = []
    passed = True
    for command in commands:
        started = perf_counter()
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        sections.append(
            "$ " + " ".join(command) + f"\nexit_code={result.returncode} runtime_s={perf_counter()-started:.6f}\n" + result.stdout + result.stderr
        )
        passed &= result.returncode == 0
    (output / "test_summary.txt").write_text("\n\n".join(sections))
    checkpoint(output, "tests", {"complete": passed, "commands": len(commands), "summary": "results/e08_path_semantics_v1/test_summary.txt"})
    if not passed:
        raise RuntimeError("correctness tests failed; geometry interpretation is blocked")


def stage_geometry(config: dict[str, Any], output: Path) -> None:
    require_checkpoint(output, "freeze")
    require_checkpoint(output, "tests")
    frozen = json.loads((output / "frozen_candidate_manifest.json").read_text())
    if file_hash(output / "frozen_candidates.npz") != frozen["candidate_file_sha256"]:
        raise RuntimeError("frozen candidate arrays changed")
    qdefs = json.loads((output / "quadrature_definitions.json").read_text())
    for item in qdefs["quadratures"]:
        if file_hash(ROOT / item["file"]) != item["sha256"]:
            raise RuntimeError("frozen quadrature changed")
    historical = json.loads((ROOT / config["inputs"]["historical_surface_config"]).read_text())
    archived_rows = read_csv(ROOT / config["inputs"]["canonical_qualification"])
    archived = {}
    for row in archived_rows:
        archived.setdefault((row["surface_id"], row["family"]), row)
    candidate_arrays = np.load(output / "frozen_candidates.npz")
    legacy_rows, comparison_rows, resolution_rows = [], [], []
    legacy_sources_payload: dict[str, np.ndarray] = {}
    final_cache = {}
    for record in frozen["candidates"]:
        surface_id = record["surface_id"]
        metadata = historical["surfaces"][surface_id]
        surface = make_surface(surface_id, historical, config)
        paths = tuple(np.asarray(candidate_arrays[key], float) for key in record["path_keys"])
        proposal = make_proposal(surface, next(x for x in config["candidates"][surface_id] if x["name"] == record["family"]), config)
        start = perf_counter()
        legacy = evaluate_nuc_coverage(
            surface,
            proposal.plan,
            footprint_radius=config["coverage"]["footprint_radius_m"],
            path_sample_spacing=config["coverage"]["legacy_path_step_m"],
        )
        source_counts = []
        for segment_index, path in enumerate(paths):
            projection = project_points(surface, path)
            _, sources = _length_and_sources(surface, projection, max_spacing=config["coverage"]["legacy_path_step_m"])
            key = f"candidate_{record['candidate_index']}_segment_{segment_index}"
            legacy_sources_payload[key] = sources
            source_counts.append(len(sources))
        historical_row = archived[(surface_id, record["family"])]
        miss_delta = legacy.missed_error - float(historical_row["intended_E_miss"])
        repeat_delta = legacy.repeat_error - float(historical_row["intended_E_rep"])
        legacy_rows.append(
            {
                "candidate_id": record["candidate_id"],
                "E_miss": legacy.missed_error,
                "E_rep": legacy.repeat_error,
                "E_NUC": legacy.nuc_error,
                "path_length_m": legacy.path_length,
                "historical_E_miss": historical_row["intended_E_miss"],
                "historical_E_rep": historical_row["intended_E_rep"],
                "miss_delta": miss_delta,
                "repeat_delta": repeat_delta,
                "replay_warning": max(abs(miss_delta), abs(repeat_delta)) > config["coverage"]["replay_warning_absolute"],
                "legacy_source_counts": ";".join(map(str, source_counts)),
                "runtime_s": perf_counter() - start,
            }
        )
        prescribed = tuple(
            resample_prescribed_path(
                path,
                surface_id=surface_id,
                surface_metadata=metadata,
                maximum_step=config["coverage"]["legacy_path_step_m"],
            )
            for path in paths
        )
        start = perf_counter()
        pmetrics = evaluate_ordered_trace_mesh(
            surface,
            prescribed,
            footprint_radius=config["coverage"]["footprint_radius_m"],
            chunk_size=config["coverage"]["distance_chunk_size"],
        )
        projection_residuals = np.concatenate([project_points(surface, trace).distances for trace in prescribed])
        comparison_rows.extend(
            [
                metric_row(record, "L", legacy.missed_error, legacy.missed_error, legacy.repeat_error, legacy.repeat_error, legacy.path_length, 0.0, legacy_rows[-1]["runtime_s"]),
                metric_row(record, "P", pmetrics.missed_lower, pmetrics.missed_upper, pmetrics.repeat_lower, pmetrics.repeat_upper, pmetrics.path_length, pmetrics.uncertain_area_fraction, perf_counter() - start, projection_max=float(projection_residuals.max()), projection_mean=float(projection_residuals.mean())),
            ]
        )
        for schedule_index, setting in enumerate(config["reference_schedule"]):
            path_step = float(setting["path_step_m"])
            level = setting["quadrature"]
            traces = tuple(
                resample_prescribed_path(path, surface_id=surface_id, surface_metadata=metadata, maximum_step=path_step)
                for path in paths
            )
            quadrature = load_quadrature(output, surface_id, level, metadata)
            started = perf_counter()
            metrics = evaluate_ordered_trace_reference(
                quadrature,
                traces,
                footprint_radius=config["coverage"]["footprint_radius_m"],
                surface_metadata=metadata,
                chunk_size=config["coverage"]["distance_chunk_size"],
            )
            row = metric_row(record, "R", metrics.missed_lower, metrics.missed_upper, metrics.repeat_lower, metrics.repeat_upper, metrics.path_length, metrics.uncertain_area_fraction, perf_counter() - started)
            row.update({"schedule_index": schedule_index, "path_step_m": path_step, "quadrature": level, "surface_samples": len(quadrature.points)})
            resolution_rows.append(row)
            if schedule_index == len(config["reference_schedule"]) - 1:
                comparison_rows.append(row.copy())
                final_cache[record["candidate_id"]] = (quadrature, traces, metrics, metadata)
        print(record["candidate_id"], "complete", flush=True)
    np.savez_compressed(output / "legacy_ordered_sources.npz", **legacy_sources_payload)
    write_csv(output / "legacy_replay.csv", legacy_rows)
    write_csv(output / "path_semantics_comparison.csv", comparison_rows)
    write_csv(output / "reference_resolution.csv", resolution_rows)
    decisions = decide_geometry(config, resolution_rows)
    write_csv(output / "geometry_decisions.csv", decisions)
    save_representative_traces(output, config, frozen["candidates"], final_cache)
    mesh_discrepancy = surface_discrepancy(config, historical, output)
    code_sha = git("rev-parse", "HEAD")
    manifest = {
        "experiment": config["experiment"],
        "executed_at": timestamp(),
        "tested_code_sha": code_sha,
        "branch": git("branch", "--show-current"),
        "config_sha256": file_hash(ROOT / "configs/e08_path_semantics_v1.json"),
        "frozen_candidate_manifest_sha256": file_hash(output / "frozen_candidate_manifest.json"),
        "quadrature_definitions_sha256": file_hash(output / "quadrature_definitions.json"),
        "legacy_ordered_sources_sha256": file_hash(output / "legacy_ordered_sources.npz"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "surface_representation_discrepancy": mesh_discrepancy,
        "scope": config["scope"],
    }
    write_json(output / "manifest.json", manifest)
    checkpoint(output, "geometry", {"complete": True, "candidates": len(decisions), "tested_code_sha": code_sha})


def stage_readiness(config: dict[str, Any], output: Path) -> None:
    require_checkpoint(output, "tests")
    sigma = json.loads((ROOT / config["inputs"]["e08_config"]).read_text())["robot"]["sigma_safe"]
    rows = []
    for row in read_csv(ROOT / config["inputs"]["ik_comparison"]):
        solved = row["complete_lift"].lower() == "true"
        numeric = (
            solved
            and float(row["max_position_error_m"]) <= 0.0001
            and float(row["max_axis_error_degrees"]) <= 0.1
            and float(row["min_sigma_min_5"]) >= sigma
            and float(row["min_joint_margin"]) >= 0.0
        )
        rows.append(
            {
                "surface_id": row["surface_id"],
                "scene_id": row["scene_id"],
                "backend": row["backend"],
                "target_sequence_solved": solved,
                "sampled_numeric_task_constraints_pass": numeric,
                "dense_transition_check_pass": "NOT_RUN",
                "coverage_contract_pass": "NOT_RUN",
                "overall_execution_pass": "NOT_RUN",
                "min_sigma_min_5": row["min_sigma_min_5"],
                "sigma_safe": sigma,
                "status_note": "pointwise target-sequence completion is not a dense continuous lift or coverage-qualified execution",
            }
        )
    write_csv(output / "corrected_ik_status.csv", rows)
    readiness = {
        "real_surface_graph_construction": "NOT_RUN",
        "real_surface_s0_s1": "NOT_RUN",
        "final_robot_plan_validation": "NOT_RUN",
        "runtime_benefit": "N/A",
        "pruning_efficacy": "N/A",
        "success_rate": "N/A",
        "task_preserving_on_connections": False,
        "explicit_node_activity": False,
        "recomputed_endpoint_membership_checked": True,
        "old_cross_chain_connection_semantics": "OFF reconfiguration",
        "collision_scope": "existing MuJoCo self-collision pairs only; workpiece, tool body beyond the model, and external environment were not modeled",
        "guard": "require_real_graph_readiness rejects missing ON connections, node activity, or endpoint membership checks",
        "focused_regressions": "see test_summary.txt",
    }
    write_json(output / "readiness_checks.json", readiness)
    checkpoint(output, "readiness", {"complete": True, "ik_rows": len(rows), "real_campaign": "NOT_RUN"})


def stage_report(config: dict[str, Any], output: Path) -> None:
    for stage in ("geometry", "readiness"):
        require_checkpoint(output, stage)
    legacy = read_csv(output / "legacy_replay.csv")
    comparisons = read_csv(output / "path_semantics_comparison.csv")
    decisions = read_csv(output / "geometry_decisions.csv")
    corrected = read_csv(output / "corrected_ik_status.csv")
    table = []
    for decision in decisions:
        candidate = decision["candidate_id"]
        values = {(row["condition"]): row for row in comparisons if row["candidate_id"] == candidate}
        table.append(
            f"| {candidate} | {float(values['L']['missed_lower']):.6f}/{float(values['L']['repeat_lower']):.6f} | "
            f"{float(values['P']['missed_lower']):.6f}/{float(values['P']['repeat_lower']):.6f} | "
            f"[{float(values['R']['missed_lower']):.6f},{float(values['R']['missed_upper']):.6f}] / "
            f"[{float(values['R']['repeat_lower']):.6f},{float(values['R']['repeat_upper']):.6f}] | {decision['decision']} |"
        )
    replay_warnings = sum(row["replay_warning"].lower() == "true" for row in legacy)
    numeric_fail = [f"{row['surface_id']}/{row['scene_id']}/{row['backend']}" for row in corrected if row["sampled_numeric_task_constraints_pass"].lower() == "false"]
    report = f"""# E08-R1 path-semantics repair

## Progress and frozen scope

All five bounded stages completed. Exactly eight archived E08 geometries were regenerated once,
hashed, and evaluated without placement repetition or path tuning. The footprint radius and
miss/repeat limits remained 0.008 m, 0.02, and 0.10. Historical files and evaluator defaults were
not changed. Legacy replay warnings above 1e-8: {replay_warnings}.

## L/P/R geometry results

Values are E_miss/E_rep. R shows lower/upper sampled-reference bounds.

| candidate | L | P | R Q2 | geometry decision |
|---|---:|---:|---:|---|
{os.linesep.join(table)}

L versus P changes only ordered trajectory reconstruction under the same mesh-distance backend.
L versus R additionally changes surface samples, area measure, and footprint distance, so it is
not a pure reconstruction attribution. Saddle uncertainty is retained with episode-count dynamic
programming. The final decisions also require Q1/Q2 changes no larger than 0.002. Geometry
acceptance, if any, does not imply robot qualification; robot requalification was NOT RUN.

## Corrected execution semantics

The historical `complete_lift` field is interpreted only as `target_sequence_solved`. Dense
transition, coverage-contract, and overall execution checks are NOT_RUN. Sampled numeric failures
under the frozen sigma threshold include: {', '.join(numeric_fail) if numeric_fail else 'none'}.
The full corrected view is in `corrected_ik_status.csv`.

## Search and graph readiness

Both search arms now use the same segment-budget-layered ordinary reachability check. Focused tests
separate segment-budget obstruction from a positive repeat-bound prune and compare both arms with
a bounded independent full-count oracle on a cyclic graph. No real graph or real S0/S1 campaign
was run. The old cross-chain edges are OFF reconfigurations, not task-preserving ON connections.
Future graph work still needs explicit node activity and independently checked ON connections.
Endpoint membership is recomputed and checked rather than overwritten. Collision claims remain
limited to modeled MuJoCo self-collision pairs; workpiece, full tool, and environment are absent.

## Interpretation boundary

These results concern sampled membership on the frozen eight paths. They do not certify continuous
coverage, robot safety, global infeasibility, planner superiority, FM benefit, or publication-level
novelty. Runtime and pruning benefit for E08 real-surface planning remain N/A because no case was
admitted and no such comparison ran.
"""
    (output / "report.md").write_text(report)
    commands = "\n".join(
        f"{sys.executable} scripts/audit_e08_path_semantics.py --stage {stage}"
        for stage in STAGES
    ) + "\n"
    (output / "reproduction_commands.txt").write_text(commands)
    checkpoint(output, "report", {"complete": True, "report": "results/e08_path_semantics_v1/report.md"})


def make_surface(surface_id: str, historical: dict[str, Any], config: dict[str, Any]):
    cfg = historical["surfaces"][surface_id]
    samples = int(config["coverage"]["legacy_samples_per_face"])
    if surface_id == "saddle":
        return make_saddle(width=cfg["width"], height=cfg["height"], curvature=cfg["curvature"], nx=12, ny=12, samples_per_face=samples)
    return make_hemisphere(radius=cfg["radius"], n_azimuth=24, n_polar=10, samples_per_face=samples)


def make_proposal(surface, family: dict[str, Any], config: dict[str, Any]):
    name = family["name"].removesuffix("_k2")
    common = {
        "footprint_radius": config["coverage"]["footprint_radius_m"],
        "max_segments": int(family["max_segments"]),
        "overlap": float(family["overlap"]),
        "waypoint_spacing": 0.75 * config["coverage"]["footprint_radius_m"],
    }
    if name.startswith("raster_"):
        mode, phase = name.split("_phase_")
        return raster_pattern(surface, sweep_axis=mode[-1], phase=float(phase), **common)
    return spiral_pattern(surface, phase=float(name.split("_phase_")[1]), **common)


def load_quadrature(output: Path, surface_id: str, level: str, metadata: dict[str, Any]):
    expected = make_analytical_quadrature(surface_id, metadata, level)
    data = np.load(output / f"quadrature_{surface_id}_{level}.npz")
    from diffusion_coverage.coverage.ordered_trace_evaluator import AnalyticalQuadrature
    return AnalyticalQuadrature(surface_id, level, data["points"], data["weights"], data["parameters"], expected.definition)


def metric_row(record, condition, miss_lo, miss_hi, rep_lo, rep_hi, length, uncertainty, runtime, projection_max=None, projection_mean=None):
    return {
        "candidate_id": record["candidate_id"],
        "surface_id": record["surface_id"],
        "family": record["family"],
        "condition": condition,
        "missed_lower": miss_lo,
        "missed_upper": miss_hi,
        "repeat_lower": rep_lo,
        "repeat_upper": rep_hi,
        "path_length_m": length,
        "uncertain_area_fraction": uncertainty,
        "mesh_projection_max_m": projection_max,
        "mesh_projection_mean_m": projection_mean,
        "runtime_s": runtime,
    }


def decide_geometry(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    miss_limit = config["coverage"]["missed_tolerance"]
    repeat_limit = config["coverage"]["repeat_tolerance"]
    delta_limit = config["coverage"]["resolution_change_warning_absolute"]
    candidates = sorted({row["candidate_id"] for row in rows})
    for candidate in candidates:
        values = [row for row in rows if row["candidate_id"] == candidate]
        q1 = next(row for row in values if row["schedule_index"] == 3)
        q2 = next(row for row in values if row["schedule_index"] == 4)
        deltas = {key: abs(float(q2[key]) - float(q1[key])) for key in ("missed_lower", "missed_upper", "repeat_lower", "repeat_upper")}
        def raw(row):
            if float(row["missed_upper"]) <= miss_limit and float(row["repeat_upper"]) <= repeat_limit:
                return "accepted_under_reference_checks"
            if float(row["missed_lower"]) > miss_limit or float(row["repeat_lower"]) > repeat_limit:
                return "rejected_under_reference_checks"
            return "unresolved"
        q1_status, q2_status = raw(q1), raw(q2)
        stable = max(deltas.values()) <= delta_limit and q1_status == q2_status
        decision = q2_status if stable else "unresolved"
        reasons = []
        if max(deltas.values()) > delta_limit:
            reasons.append("numerical_resolution_unresolved")
        if q1_status != q2_status:
            reasons.append("final_resolution_status_changed")
        if q2_status == "unresolved":
            reasons.append("reference_bounds_straddle_threshold")
        if not reasons:
            reasons.append("stable_reference_bounds")
        decisions.append({"candidate_id": candidate, "surface_id": q2["surface_id"], "decision": decision, "Q1_raw_status": q1_status, "Q2_raw_status": q2_status, "max_final_bound_change": max(deltas.values()), "reason": ";".join(reasons), "robot_requalification": "NOT_RUN"})
    return decisions


def save_representative_traces(output: Path, config: dict[str, Any], records, cache) -> None:
    data_rows = []
    plot_dir = output / "representative_plots"
    plot_dir.mkdir(exist_ok=True)
    for record in records:
        candidate = record["candidate_id"]
        quadrature, traces, metrics, metadata = cache[candidate]
        choices = [int(np.argmax(metrics.episode_max))]
        missed = np.flatnonzero(metrics.episode_max == 0)
        uncertain = np.flatnonzero(metrics.episode_min != metrics.episode_max)
        if len(missed): choices.append(int(missed[0]))
        if len(uncertain): choices.append(int(uncertain[0]))
        choices = list(dict.fromkeys(choices))[:3]
        fig, axes = plt.subplots(len(choices), 1, figsize=(8, 2.2 * len(choices)), squeeze=False)
        for row_index, sample_index in enumerate(choices):
            offset = 0
            timeline = []
            for segment_index, trace in enumerate(traces):
                states = reference_membership_states(record["surface_id"], quadrature.points[[sample_index]], trace, footprint_radius=config["coverage"]["footprint_radius_m"], surface_metadata=metadata)[0]
                for local_index, state in enumerate(states):
                    data_rows.append({"candidate_id": candidate, "sample_index": sample_index, "segment_index": segment_index, "time_index": local_index, "state": int(state), "x": quadrature.points[sample_index, 0], "y": quadrature.points[sample_index, 1], "z": quadrature.points[sample_index, 2], "weight": quadrature.weights[sample_index]})
                if timeline: timeline.append(np.asarray([-2], dtype=np.int8))
                timeline.append(states)
                offset += len(states)
            joined = np.concatenate(timeline)
            axes[row_index, 0].step(np.arange(len(joined)), joined, where="post")
            axes[row_index, 0].set_ylim(-2.3, 1.3)
            axes[row_index, 0].set_ylabel(f"sample {sample_index}")
        axes[-1, 0].set_xlabel("ordered trace sample (-2 denotes OFF segment boundary; -1 uncertain)")
        fig.suptitle(candidate)
        fig.tight_layout()
        fig.savefig(plot_dir / (candidate.replace("/", "_") + ".png"), dpi=140)
        plt.close(fig)
    write_csv(output / "representative_membership_source.csv", data_rows)


def surface_discrepancy(config, historical, output):
    records = []
    for surface_id in ("saddle", "hemisphere"):
        surface = make_surface(surface_id, historical, config)
        projected, _ = analytical_points_and_normals(surface_id, surface.sample_points, historical["surfaces"][surface_id])
        residual = np.linalg.norm(projected - surface.sample_points, axis=1)
        q2 = np.load(output / f"quadrature_{surface_id}_Q2.npz")
        records.append({"surface_id": surface_id, "legacy_mesh_samples": surface.num_samples, "legacy_mesh_area": float(surface.area_weights.sum()), "analytical_Q2_samples": len(q2["points"]), "analytical_Q2_area": float(q2["weights"].sum()), "area_change": float(q2["weights"].sum() - surface.area_weights.sum()), "legacy_sample_analytical_projection_max_m": float(residual.max()), "legacy_sample_analytical_projection_mean_m": float(residual.mean())})
    return records


def update_array_hash(digest, value):
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp() -> str:
    return subprocess.run(["date", "--iso-8601=seconds"], text=True, capture_output=True, check=True).stdout.strip()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint(output: Path, stage: str, payload: dict[str, Any]) -> None:
    write_json(output / f"{stage}.checkpoint.json", {"stage": stage, **payload})


def require_checkpoint(output: Path, stage: str) -> None:
    path = output / f"{stage}.checkpoint.json"
    if not path.exists() or not json.loads(path.read_text()).get("complete", False):
        raise RuntimeError(f"required stage {stage!r} is incomplete")


if __name__ == "__main__":
    main()
