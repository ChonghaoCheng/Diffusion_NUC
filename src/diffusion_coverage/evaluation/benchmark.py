from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from diffusion_coverage.problem import GraphSolution
from diffusion_coverage.solvers import BaseGraphSolver


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    feasible: bool
    optimal_hit: bool
    num_lift_offs: int
    joint_motion_cost: float
    search_nodes: int
    evaluated_assignments: int
    proposal_time: float | None
    time_to_first_feasible: float | None
    time_to_best: float | None
    time_to_certified: float | None
    total_solve_time: float
    certified_optimal: bool


def make_benchmark_row(method: str, solution: GraphSolution, optimum: GraphSolution) -> BenchmarkRow:
    return BenchmarkRow(
        method=method,
        feasible=solution.feasible,
        optimal_hit=solution.feasible and solution.objective == optimum.objective,
        num_lift_offs=solution.num_lift_offs,
        joint_motion_cost=solution.joint_motion_cost,
        search_nodes=solution.search_nodes,
        evaluated_assignments=solution.evaluated_assignments,
        proposal_time=solution.metadata.get("proposal_time"),
        time_to_first_feasible=solution.metadata.get("time_to_first_feasible"),
        time_to_best=solution.metadata.get("time_to_best"),
        time_to_certified=solution.metadata.get("time_to_certified"),
        total_solve_time=solution.metadata.get("total_solve_time", solution.solve_time),
        certified_optimal=solution.certified_optimal,
    )


def run_solvers(instance, solvers: dict[str, BaseGraphSolver]) -> list[BenchmarkRow]:
    optimum = solvers["Exact"].solve(instance)
    rows = [make_benchmark_row("Exact", optimum, optimum)]
    for name, solver in solvers.items():
        if name == "Exact":
            continue
        rows.append(make_benchmark_row(name, solver.solve(instance), optimum))
    return rows


def summarize_rows(rows: list[dict], solver: str) -> dict:
    selected = [row for row in rows if row["solver"] == solver]
    if not selected:
        return {}
    return {
        "instances": len(selected),
        "feasible_rate": mean(float(row["feasible"]) for row in selected),
        "optimal_hit_rate": mean(float(row["optimal_hit"]) for row in selected),
        "mean_search_nodes": mean(row["search_nodes"] for row in selected),
        "median_search_nodes": median(row["search_nodes"] for row in selected),
        "mean_evaluated_assignments": mean(row["evaluated_assignments"] for row in selected),
        "median_evaluated_assignments": median(row["evaluated_assignments"] for row in selected),
        "mean_time_to_first_feasible": _mean_optional(selected, "time_to_first_feasible"),
        "mean_time_to_best": _mean_optional(selected, "time_to_best"),
        "median_time_to_best": _median_optional(selected, "time_to_best"),
        "mean_time_to_certified": _mean_optional(selected, "time_to_certified"),
        "median_time_to_certified": _median_optional(selected, "time_to_certified"),
        "mean_total_solve_time": mean(row["total_solve_time"] for row in selected),
        "median_total_solve_time": median(row["total_solve_time"] for row in selected),
    }


def _mean_optional(rows: list[dict], key: str) -> float | None:
    vals = [row[key] for row in rows if row[key] is not None]
    return mean(vals) if vals else None


def _median_optional(rows: list[dict], key: str) -> float | None:
    vals = [row[key] for row in rows if row[key] is not None]
    return median(vals) if vals else None
