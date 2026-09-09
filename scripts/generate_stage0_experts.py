#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.coverage import ClassicalTeacherPlanner, TeacherDatasetWriter, TeacherPlannerConfig
from diffusion_coverage.surface import generate_random_surface


SURFACE_TYPES = ("cylinder", "hemisphere", "saddle", "freeform_patch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate varying-footprint Stage 0 expert paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-type", choices=SURFACE_TYPES, required=True)
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--radius-min", type=float, default=0.10)
    parser.add_argument("--radius-max", type=float, default=0.30)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--refinement-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances < 1 or not 0.0 < args.radius_min <= args.radius_max:
        raise ValueError("invalid instance count or radius range")
    writer = TeacherDatasetWriter(args.output, resume=args.resume)
    completed = writer.completed_instance_ids
    type_offset = SURFACE_TYPES.index(args.surface_type) * args.instances
    for local_index in range(args.instances):
        instance_id = f"{args.surface_type}_{local_index:04d}"
        if instance_id in completed:
            continue
        seed = args.seed + type_offset + local_index
        rng = np.random.default_rng(seed)
        radius = float(np.exp(rng.uniform(np.log(args.radius_min), np.log(args.radius_max))))
        surface = generate_random_surface(
            args.surface_type, seed=seed, resolution=args.resolution, samples_per_face=2
        )
        result = None
        for overlap in (0.70, 0.55, 0.45, 0.35, 0.28):
            config = TeacherPlannerConfig(
                footprint_radius=radius,
                missed_tolerance=args.epsilon,
                overlap=overlap,
                refinement_iterations=args.refinement_iterations,
                max_candidates=4,
                seed=seed,
            )
            result = ClassicalTeacherPlanner(config).solve(surface)
            if result.feasible_candidates:
                break
        assert result is not None
        if not result.feasible_candidates:
            raise RuntimeError(f"no feasible expert for {instance_id} at radius {radius:.6f}")
        writer.write(instance_id, surface, config, result)
        print(
            f"{instance_id} radius={radius:.4f} candidates={len(result.feasible_candidates)} "
            f"missed={result.best.metrics.missed_fraction:.4f} time={result.total_solve_time:.2f}s",
            flush=True,
        )
    print(f"manifest: {writer.manifest_path}")


if __name__ == "__main__":
    main()
