#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_coverage.graph.synthetic_generator import SyntheticGeneratorConfig, generate_synthetic_instance
from diffusion_coverage.solvers import ExactSolver, GreedyGuidedExactSolver, GreedySolver


SOLVERS = {
    "Exact": ExactSolver,
    "Greedy": GreedySolver,
    "Greedy-Guided Exact": GreedyGuidedExactSolver,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--nodes-extra", type=int, default=4)
    parser.add_argument("--colours", type=int, default=4)
    parser.add_argument("--valid-prob", type=float, default=0.65)
    parser.add_argument("--edge-prob", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/scaling_phase1"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []

    for n_boundary in args.boundary_sizes:
        for local_idx in range(args.instances):
            seed = args.seed + n_boundary * 10000 + local_idx
            num_nodes = n_boundary + args.nodes_extra
            cfg = SyntheticGeneratorConfig(
                num_nodes=num_nodes,
                num_colours=args.colours,
                edge_probability=args.edge_prob,
                valid_colour_probability=args.valid_prob,
                boundary_fraction=n_boundary / num_nodes,
            )
            instance = generate_synthetic_instance(seed=seed, config=cfg)
            instance_id = f"b{n_boundary}_s{seed}"
            exact = ExactSolver().solve(instance)
            solver_results = {
                "Exact": exact,
                "Greedy": GreedySolver().solve(instance),
                "Greedy-Guided Exact": GreedyGuidedExactSolver().solve(instance),
            }
            for solver_name, solution in solver_results.items():
                raw_rows.append(_raw_row(instance, instance_id, seed, solver_name, solution, exact))

    csv_path = args.out_dir / "scaling_results.csv"
    jsonl_path = args.out_dir / "scaling_results.jsonl"
    _write_csv(csv_path, raw_rows)
    _write_jsonl(jsonl_path, raw_rows)

    _print_boundary_tables(raw_rows)
    _print_scaling_summary(raw_rows)
    _write_plots(args.out_dir, raw_rows)
    print(f"\nSaved raw CSV: {csv_path}")
    print(f"Saved raw JSONL: {jsonl_path}")
    print(f"Saved plots: {args.out_dir}")


def _raw_row(instance, instance_id: str, seed: int, solver: str, solution, optimum) -> dict:
    return {
        "instance_id": instance_id,
        "random_seed": seed,
        "num_nodes": instance.num_nodes,
        "num_boundary_nodes": int(instance.boundary_mask.sum()),
        "num_edges": instance.num_edges,
        "num_colours": instance.num_colours,
        "mean_valid_colours_per_node": float(instance.valid_colour_mask.sum(axis=1).mean()),
        "solver": solver,
        "feasible": bool(solution.feasible),
        "num_lift_offs": int(solution.num_lift_offs),
        "joint_motion_cost": float(solution.joint_motion_cost),
        "optimal_hit": bool(solution.feasible and solution.objective == optimum.objective),
        "search_nodes": int(solution.search_nodes),
        "evaluated_assignments": int(solution.evaluated_assignments),
        "proposal_time": solution.metadata.get("proposal_time"),
        "time_to_first_feasible": solution.metadata.get("time_to_first_feasible"),
        "time_to_best": solution.metadata.get("time_to_best"),
        "time_to_certified": solution.metadata.get("time_to_certified"),
        "total_solve_time": float(solution.metadata.get("total_solve_time", solution.solve_time)),
        "certified_optimal": bool(solution.certified_optimal),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _print_boundary_tables(rows: list[dict]) -> None:
    for boundary in sorted({row["num_boundary_nodes"] for row in rows}):
        print(f"\nBoundary nodes = {boundary}")
        print("Method                  Opt. hit   Search nodes   t_best      t_cert")
        print("---------------------------------------------------------------------")
        for solver in SOLVERS:
            vals = _select(rows, boundary, solver)
            opt_hit = mean(float(row["optimal_hit"]) for row in vals)
            nodes = median(row["search_nodes"] for row in vals)
            t_best = _median_optional(vals, "time_to_best")
            t_cert = _median_optional(vals, "time_to_certified")
            print(f"{solver:<22} {opt_hit:>8.0%}   {nodes:>12.1f}   {_fmt_time(t_best):>8}   {_fmt_time(t_cert):>8}")


def _print_scaling_summary(rows: list[dict]) -> None:
    print("\nBoundary   Exact nodes   Guided nodes   Reduction   Greedy opt-hit")
    print("------------------------------------------------------------------")
    for boundary in sorted({row["num_boundary_nodes"] for row in rows}):
        exact_nodes = median(row["search_nodes"] for row in _select(rows, boundary, "Exact"))
        guided_nodes = median(row["search_nodes"] for row in _select(rows, boundary, "Greedy-Guided Exact"))
        greedy_hit = mean(float(row["optimal_hit"]) for row in _select(rows, boundary, "Greedy"))
        reduction = 1.0 - guided_nodes / exact_nodes if exact_nodes else 0.0
        print(f"{boundary:<10} {exact_nodes:>11.1f}   {guided_nodes:>12.1f}   {reduction:>9.1%}   {greedy_hit:>13.0%}")


def _write_plots(out_dir: Path, rows: list[dict]) -> None:
    boundaries = sorted({row["num_boundary_nodes"] for row in rows})
    exact_nodes = [_median_metric(rows, b, "Exact", "search_nodes") for b in boundaries]
    guided_nodes = [_median_metric(rows, b, "Greedy-Guided Exact", "search_nodes") for b in boundaries]
    exact_t_best = [_median_metric(rows, b, "Exact", "time_to_best") for b in boundaries]
    guided_t_best = [_median_metric(rows, b, "Greedy-Guided Exact", "time_to_best") for b in boundaries]
    greedy_hit = [mean(float(row["optimal_hit"]) for row in _select(rows, b, "Greedy")) for b in boundaries]
    reduction = [1.0 - g / e if e else 0.0 for e, g in zip(exact_nodes, guided_nodes)]

    _line_svg(
        out_dir / "median_search_nodes.svg",
        "Median search nodes",
        boundaries,
        {"Exact": exact_nodes, "Greedy-Guided Exact": guided_nodes},
        log_y=True,
    )
    _line_svg(
        out_dir / "median_time_to_best.svg",
        "Median time-to-best",
        boundaries,
        {"Exact": exact_t_best, "Greedy-Guided Exact": guided_t_best},
        log_y=True,
    )
    _line_svg(out_dir / "greedy_opt_hit.svg", "Greedy optimal hit rate", boundaries, {"Greedy": greedy_hit})
    _line_svg(out_dir / "guided_node_reduction.svg", "Greedy-guided search-node reduction", boundaries, {"Reduction": reduction})


def _line_svg(path: Path, title: str, xs: list[int], series: dict[str, list[float]], log_y: bool = False) -> None:
    width, height = 760, 460
    left, right, top, bottom = 70, 30, 45, 60
    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed"]
    all_vals = [v for values in series.values() for v in values if v is not None]
    if log_y:
        all_vals = [max(v, 1e-9) for v in all_vals]
        transform = lambda y: __import__("math").log10(max(y, 1e-9))
    else:
        transform = lambda y: y
    y_vals = [transform(v) for v in all_vals] or [0.0, 1.0]
    y_min, y_max = min(y_vals), max(y_vals)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def px(x):
        return left + (x - x_min) / (x_max - x_min) * (width - left - right)

    def py(y):
        yy = transform(y)
        return height - bottom - (yy - y_min) / (y_max - y_min) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for idx, (name, ys) in enumerate(series.items()):
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        color = colors[idx % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>')
        ly = top + 24 * idx
        parts.append(f'<line x1="{width-230}" y1="{ly}" x2="{width-205}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{width-198}" y="{ly+5}" font-family="sans-serif" font-size="13">{name}</text>')
    for x in xs:
        parts.append(f'<text x="{px(x)-6:.1f}" y="{height-bottom+22}" font-family="sans-serif" font-size="12">{x}</text>')
    parts.append(f'<text x="{width/2-55:.1f}" y="{height-18}" font-family="sans-serif" font-size="13">boundary nodes</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def _select(rows: list[dict], boundary: int, solver: str) -> list[dict]:
    return [row for row in rows if row["num_boundary_nodes"] == boundary and row["solver"] == solver]


def _median_metric(rows: list[dict], boundary: int, solver: str, key: str) -> float:
    vals = [row[key] for row in _select(rows, boundary, solver) if row[key] is not None]
    return float(median(vals)) if vals else 0.0


def _median_optional(rows: list[dict], key: str) -> float | None:
    vals = [row[key] for row in rows if row[key] is not None]
    return float(median(vals)) if vals else None


def _fmt_time(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}s"


if __name__ == "__main__":
    main()
