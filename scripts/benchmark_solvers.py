#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_coverage.evaluation.benchmark import run_solvers
from diffusion_coverage.graph.synthetic_generator import SyntheticGeneratorConfig, generate_synthetic_instance
from diffusion_coverage.solvers import ExactSolver, GreedyGuidedExactSolver, GreedySolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--colours", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = SyntheticGeneratorConfig(num_nodes=args.nodes, num_colours=args.colours)
    solvers = {"Exact": ExactSolver(), "Greedy": GreedySolver(), "Greedy-Guided Exact": GreedyGuidedExactSolver()}
    totals = {name: {"optimal": 0, "search_nodes": 0, "time_to_best": 0.0} for name in solvers}

    for idx in range(args.instances):
        instance = generate_synthetic_instance(seed=args.seed + idx, config=cfg)
        rows = run_solvers(instance, solvers)
        for row in rows:
            totals[row.method]["optimal"] += int(row.optimal_hit)
            totals[row.method]["search_nodes"] += row.search_nodes
            totals[row.method]["time_to_best"] += row.time_to_best or row.total_solve_time

    print("Method                  Optimal hit   Search nodes   Time-to-best")
    print("-----------------------------------------------------------------")
    for method, vals in totals.items():
        optimal_rate = vals["optimal"] / args.instances
        avg_nodes = vals["search_nodes"] / args.instances
        avg_time = vals["time_to_best"] / args.instances
        print(f"{method:<22} {optimal_rate:>10.2%}   {avg_nodes:>12.1f}   {avg_time:>12.6f}s")


if __name__ == "__main__":
    main()
