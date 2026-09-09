#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion_coverage.coverage import ClassicalTeacherPlanner, TeacherPlannerConfig
from diffusion_coverage.surface import SURFACE_TYPES, generate_random_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark M1 classical coverage teacher")
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--refinement-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "teacher_m1_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Surface           Raster miss/len     Spiral miss/len     Teacher miss/len   Feas.   Time")
    print("------------------------------------------------------------------------------------------")
    rows = []
    for surface_index, surface_type in enumerate(SURFACE_TYPES):
        surface = generate_random_surface(
            surface_type,
            seed=args.seed + surface_index,
            resolution=args.resolution,
            samples_per_face=2,
        )
        config = TeacherPlannerConfig(
            footprint_radius=args.radius,
            missed_tolerance=args.epsilon,
            refinement_iterations=args.refinement_iterations,
            seed=args.seed + surface_index,
        )
        result = ClassicalTeacherPlanner(config).solve(surface)
        raster = _best_family(result.initial_candidates, "raster", config)
        spiral = _best_family(result.initial_candidates, "spiral", config)
        best = result.best
        print(
            f"{surface_type:<17} "
            f"{raster.metrics.missed_fraction:6.3f}/{raster.metrics.path_length:7.2f}   "
            f"{spiral.metrics.missed_fraction:6.3f}/{spiral.metrics.path_length:7.2f}   "
            f"{best.metrics.missed_fraction:6.3f}/{best.metrics.path_length:7.2f}   "
            f"{len(result.feasible_candidates):5d} {result.total_solve_time:7.2f}s"
        )
        rows.append(
            {
                "surface": surface_type,
                "surface_metadata": surface.metadata,
                "num_vertices": surface.num_vertices,
                "num_faces": surface.num_faces,
                "footprint_radius": args.radius,
                "missed_tolerance": args.epsilon,
                "raster_missed_fraction": raster.metrics.missed_fraction,
                "raster_path_length": raster.metrics.path_length,
                "spiral_missed_fraction": spiral.metrics.missed_fraction,
                "spiral_path_length": spiral.metrics.path_length,
                "teacher_missed_fraction": best.metrics.missed_fraction,
                "teacher_path_length": best.metrics.path_length,
                "teacher_pattern": best.proposal_name,
                "num_feasible_candidates": len(result.feasible_candidates),
                "evaluated_plans": result.evaluated_plans,
                "total_solve_time": result.total_solve_time,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nRaw benchmark: {args.output}")


def _best_family(candidates, prefix: str, config: TeacherPlannerConfig):
    family = [candidate for candidate in candidates if candidate.proposal_name.startswith(prefix)]

    def objective(candidate):
        violation = max(0.0, candidate.metrics.missed_fraction - config.missed_tolerance)
        return (int(violation > 0.0), violation, candidate.metrics.path_length)

    return min(family, key=objective)


if __name__ == "__main__":
    main()
