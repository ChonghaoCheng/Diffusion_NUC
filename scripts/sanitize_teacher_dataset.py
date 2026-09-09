#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.coverage import (
    CoveragePlan,
    evaluate_coverage,
    load_teacher_instance,
    refine_coverage_by_insertion,
    surface_from_teacher_archive,
    constrained_coverage_key,
)
from diffusion_coverage.learning import load_manifest
from diffusion_coverage.learning import canonical_candidate_name
from diffusion_coverage.representation import canonicalize_surface_path, suggested_token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recheck and sanitize serialized teacher targets")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repair-steps", type=int, default=16)
    parser.add_argument(
        "--robust-margin",
        type=float,
        default=0.0,
        help="Require serialized targets to beat the missed-coverage tolerance by this margin",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--require-learning-roundtrip", action="store_true")
    parser.add_argument("--tokens-per-footprint-area", type=float, default=1.0)
    parser.add_argument("--minimum-path-tokens", type=int, default=32)
    parser.add_argument("--maximum-path-tokens", type=int, default=2048)
    parser.add_argument(
        "--allow-empty-instance",
        action="store_true",
        help="Skip instances with no admitted candidate instead of failing the dataset build",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances_dir = args.output / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(manifest_path)
    completed = set()
    if manifest_path.exists():
        completed = {
            json.loads(line)["instance_id"]
            for line in manifest_path.read_text().splitlines() if line
        }
    for row in load_manifest(args.dataset):
        instance_id = str(row["instance_id"])
        if instance_id in completed:
            continue
        archive = load_teacher_instance(args.dataset / str(row["path"]))
        metadata = archive["metadata"]
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        config = metadata["teacher_config"]
        radius = float(config["footprint_radius"])
        epsilon = float(config["missed_tolerance"])
        robust_epsilon = max(0.0, epsilon - args.robust_margin)
        proposal_names = np.asarray(archive["proposal_names"])
        original_metrics = np.asarray(archive["candidate_metrics"])
        candidates = []
        rejected = 0
        for index in range(len(original_metrics)):
            proposal_name = str(proposal_names[index])
            if (
                args.candidate_name is not None
                and canonical_candidate_name(proposal_name) != args.candidate_name
            ):
                continue
            plan = CoveragePlan(
                archive["candidate_waypoints"][index],
                archive["candidate_segment_mask"][index],
                archive["candidate_waypoint_mask"][index],
                metadata={"missed_tolerance": epsilon},
            )
            metrics = evaluate_coverage(surface, plan, footprint_radius=radius)
            accepted = None
            if metrics.missed_fraction <= robust_epsilon + 1e-12:
                accepted = (plan, metrics, float(original_metrics[index, 3]), proposal_name)
            else:
                result = refine_coverage_by_insertion(
                    surface, plan,
                    footprint_radius=radius,
                    missed_tolerance=robust_epsilon,
                    max_steps=args.repair_steps,
                )
                if result.metrics.missed_fraction <= robust_epsilon + 1e-12:
                    accepted = (
                        result.plan, result.metrics, float(original_metrics[index, 3]),
                        f"{proposal_name}_robustness_repair",
                    )
                else:
                    rejected += 1
            if accepted is not None and args.require_learning_roundtrip:
                roundtrip = learning_roundtrip_metrics(
                    surface,
                    accepted[0],
                    footprint_radius=radius,
                    tokens_per_footprint_area=args.tokens_per_footprint_area,
                    minimum_path_tokens=args.minimum_path_tokens,
                    maximum_path_tokens=args.maximum_path_tokens,
                )
                if roundtrip.missed_fraction > epsilon + 1e-12:
                    rejected += 1
                    accepted = None
            if accepted is not None:
                candidates.append(accepted)
        repaired = any(name.endswith("_robustness_repair") for *_, name in candidates)
        if not candidates:
            if args.allow_empty_instance:
                print(f"{instance_id:<24} skipped: no admitted candidate", flush=True)
                continue
            raise RuntimeError(f"no sanitized feasible candidate for {instance_id}")
        candidates.sort(
            key=lambda item: constrained_coverage_key(
                item[1].missed_fraction, item[1].path_length, robust_epsilon
            )
        )
        output_path = instances_dir / f"{instance_id}.npz"
        write_instance(
            output_path, archive, surface, candidates, rejected, repaired,
            robust_margin=args.robust_margin,
            candidate_name=args.candidate_name,
            learning_roundtrip=args.require_learning_roundtrip,
        )
        best = candidates[0][1]
        manifest_row = {
            "instance_id": instance_id,
            "path": str(output_path.relative_to(args.output)),
            "surface_id": str(row["surface_id"]),
            "num_vertices": surface.num_vertices,
            "num_faces": surface.num_faces,
            "num_candidates": len(candidates),
            "num_feasible_candidates": len(candidates),
            "best_missed_fraction": best.missed_fraction,
            "best_path_length": best.path_length,
            "rejected_serialized_candidates": rejected,
            "required_repair": repaired,
        }
        with manifest_path.open("a") as manifest_file:
            manifest_file.write(json.dumps(manifest_row, sort_keys=True) + "\n")
        print(
            f"{instance_id:<24} kept={len(candidates)} rejected={rejected} repaired={repaired}",
            flush=True,
        )


def learning_roundtrip_metrics(
    surface,
    plan,
    *,
    footprint_radius: float,
    tokens_per_footprint_area: float,
    minimum_path_tokens: int,
    maximum_path_tokens: int,
):
    num_tokens = suggested_token_count(
        surface.total_area,
        footprint_radius,
        tokens_per_footprint_area=tokens_per_footprint_area,
        minimum=minimum_path_tokens,
        maximum=maximum_path_tokens,
    )
    canonical = canonicalize_surface_path(
        surface,
        plan.active_paths()[0],
        num_tokens=num_tokens,
        preserve_source_waypoints=True,
    )
    points = np.asarray(surface.sample_points, dtype=np.float64)
    weights = np.asarray(surface.area_weights, dtype=np.float64)
    center = np.average(points, axis=0, weights=weights)
    scale = float(np.max(np.linalg.norm(points - center, axis=1)))
    normalized = ((canonical.points - center) / scale).astype(np.float32)
    restored = normalized.astype(np.float64) * scale + center
    return evaluate_coverage(
        surface, CoveragePlan(restored), footprint_radius=footprint_radius
    )


def write_instance(
    path,
    archive,
    surface,
    candidates,
    rejected: int,
    repaired: bool,
    *,
    robust_margin: float,
    candidate_name: str | None,
    learning_roundtrip: bool,
) -> None:
    max_segments = max(candidate[0].waypoints.shape[0] for candidate in candidates)
    max_waypoints = max(candidate[0].waypoints.shape[1] for candidate in candidates)
    waypoints = np.zeros((len(candidates), max_segments, max_waypoints, 3), dtype=np.float64)
    segment_mask = np.zeros((len(candidates), max_segments), dtype=bool)
    waypoint_mask = np.zeros((len(candidates), max_segments, max_waypoints), dtype=bool)
    metrics = np.zeros((len(candidates), 5), dtype=np.float64)
    names = []
    for index, (plan, plan_metrics, smoothness, name) in enumerate(candidates):
        segments, count, _ = plan.waypoints.shape
        waypoints[index, :segments, :count] = plan.waypoints
        segment_mask[index, :segments] = plan.segment_mask
        waypoint_mask[index, :segments, :count] = plan.waypoint_mask
        metrics[index] = (
            plan_metrics.missed_fraction,
            plan_metrics.path_length,
            plan_metrics.coverage_efficiency,
            smoothness,
            1.0,
        )
        names.append(name)
    metadata = dict(archive["metadata"])
    metadata["sanitization"] = {
        "hard_rechecked_after_serialization": True,
        "stored_waypoint_dtype": "float64",
        "rejected_candidates": rejected,
        "required_repair": repaired,
        "robust_margin": robust_margin,
        "candidate_name": candidate_name,
        "learning_roundtrip_required": learning_roundtrip,
    }
    np.savez_compressed(
        path,
        vertices=surface.vertices,
        faces=surface.faces,
        sample_points=surface.sample_points,
        sample_normals=surface.sample_normals,
        area_weights=surface.area_weights,
        sample_face_indices=surface.sample_face_indices,
        sample_barycentric=surface.sample_barycentric,
        candidate_waypoints=waypoints,
        candidate_segment_mask=segment_mask,
        candidate_waypoint_mask=waypoint_mask,
        candidate_metrics=metrics,
        proposal_names=np.asarray(names, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


if __name__ == "__main__":
    main()
