#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion_coverage.coverage import ClassicalTeacherPlanner, TeacherDatasetWriter, TeacherPlannerConfig
from diffusion_coverage.surface import SURFACE_TYPES, generate_random_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an M1 multi-solution teacher dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "teacher_m1_dataset")
    parser.add_argument("--instances-per-surface", type=int, default=1)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--refinement-iterations", type=int, default=3)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--fallback-overlap", type=float, default=0.55)
    parser.add_argument("--minimum-overlap", type=float, default=0.35)
    parser.add_argument("--surface-types", nargs="+", choices=SURFACE_TYPES, default=list(SURFACE_TYPES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances_per_surface < 1:
        raise ValueError("instances-per-surface must be positive")
    writer = TeacherDatasetWriter(args.output, resume=args.resume)
    completed_ids = writer.completed_instance_ids
    print("Instance                 Candidates   Feasible   Missed    Length    Time")
    print("-------------------------------------------------------------------------")
    for surface_type in args.surface_types:
        for local_index in range(args.instances_per_surface):
            seed = args.seed + SURFACE_TYPES.index(surface_type) * args.instances_per_surface + local_index
            instance_id = f"{surface_type}_{local_index:04d}"
            if instance_id in completed_ids:
                print(f"{instance_id:<24} {'already complete':>45}")
                continue
            surface = generate_random_surface(
                surface_type,
                seed=seed,
                resolution=args.resolution,
                samples_per_face=2,
            )
            overlaps = [args.overlap, args.fallback_overlap, 0.45, args.minimum_overlap]
            overlaps = list(dict.fromkeys(overlap for overlap in overlaps if overlap <= args.overlap))
            result = None
            for overlap in overlaps:
                config = TeacherPlannerConfig(
                    footprint_radius=args.radius,
                    missed_tolerance=args.epsilon,
                    refinement_iterations=args.refinement_iterations,
                    overlap=overlap,
                    seed=seed,
                )
                result = ClassicalTeacherPlanner(config).solve(surface)
                if result.feasible_candidates:
                    break
            assert result is not None
            if not result.feasible_candidates:
                raise RuntimeError(
                    f"no feasible teacher candidate for {instance_id}; "
                    "increase resolution or use a smaller overlap value"
                )
            writer.write(instance_id, surface, config, result)
            print(
                f"{instance_id:<24} {len(result.candidates):10d} "
                f"{len(result.feasible_candidates):10d} {result.best.metrics.missed_fraction:8.4f} "
                f"{result.best.metrics.path_length:9.3f} {result.total_solve_time:7.2f}s"
            )
    print(f"\nDataset manifest: {writer.manifest_path}")


if __name__ == "__main__":
    main()
