#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate pre-FM UR5e liftability audits")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [record for path in args.inputs for record in read_records(path)]
    if not records:
        raise ValueError("no liftability records found")
    write_records(args.output / "combined_raw_results.csv", records)
    summary = build_summary(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "report.md").write_text(render_report(summary))
    plot_results(summary, args.output / "liftability_summary.png")
    print(render_report(summary))


def read_records(path: Path) -> list[dict[str, object]]:
    csv_path = path / "raw_results.csv" if path.is_dir() else path
    with csv_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    records: list[dict[str, object]] = []
    for row in rows:
        converted: dict[str, object] = dict(row)
        for key in ("liftable", "surface_coverage_feasible", "solver_invoked"):
            converted[key] = str(row.get(key, "")).lower() == "true"
        for key in (
            "tau_degrees", "calibration_posewise_ik_fraction", "solve_time",
            "missed_fraction", "physical_path_length", "min_manipulability",
            "min_joint_limit_margin",
        ):
            value = row.get(key, "")
            converted[key] = float(value) if value not in ("", "None") else None
        records.append(converted)
    return records


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    taus = sorted({float(record["tau_degrees"]) for record in records})
    surfaces = sorted({str(record["surface_id"]) for record in records})
    by_tau: dict[str, object] = {}
    for tau in taus:
        selected = [record for record in records if float(record["tau_degrees"]) == tau]
        calibrated = [
            record for record in selected
            if float(record["calibration_posewise_ik_fraction"] or 0.0) >= 1.0 - 1e-12
        ]
        by_tau[str(tau)] = {
            "overall": summarize_group(selected),
            "fully_posewise_calibrated": summarize_group(calibrated),
            "by_surface": {
                surface: summarize_group(
                    [record for record in selected if str(record["surface_id"]) == surface]
                )
                for surface in surfaces
            },
            "by_surface_fully_posewise_calibrated": {
                surface: summarize_group(
                    [record for record in calibrated if str(record["surface_id"]) == surface]
                )
                for surface in surfaces
            },
        }
    return {
        "records": len(records),
        "unique_instances": len({str(record["instance_id"]) for record in records}),
        "taus_degrees": taus,
        "surfaces": surfaces,
        "by_tau": by_tau,
        "interpretation_limits": [
            "The checker is a finite-beam numerical IK continuation search, not an exact C-space topology certificate.",
            "Collision checking covers UR5e self-collision only; workpiece collision is not modeled.",
            "Base placements are selected by sparse posewise IK calibration and are not globally optimized.",
            "Inherited wider-tolerance successes enforce the mathematical monotonicity of feasible orientation cones.",
        ],
    }


def summarize_group(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"instances": 0}
    failures: dict[str, int] = {}
    for record in records:
        reason = "success" if bool(record["liftable"]) else str(record["failure_reason"])
        failures[reason] = failures.get(reason, 0) + 1
    solve_times = [float(record["solve_time"] or 0.0) for record in records]
    continuation_failures = failures.get("continuous_ik_search_failure", 0)
    return {
        "instances": len(records),
        "lift_success_rate": float(np.mean([bool(record["liftable"]) for record in records])),
        "surface_coverage_feasible_rate": float(
            np.mean([bool(record["surface_coverage_feasible"]) for record in records])
        ),
        "failure_counts": failures,
        "continuation_search_failure_rate": continuation_failures / len(records),
        "mean_calibration_posewise_ik_fraction": float(np.mean([
            float(record["calibration_posewise_ik_fraction"] or 0.0) for record in records
        ])),
        "mean_solve_time": float(np.mean(solve_times)),
        "median_solve_time": float(np.median(solve_times)),
    }


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# Pre-FM UR5e Liftability Audit",
        "",
        f"Instances: {summary['unique_instances']} ({summary['records']} instance-tolerance rows)",
        "",
        "| Tau | Group | N | Lift success | Continuation search failure | Posewise IK calibration |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for tau, tau_summary in summary["by_tau"].items():
        for group_name in ("overall", "fully_posewise_calibrated"):
            group = tau_summary[group_name]
            if group["instances"] == 0:
                continue
            lines.append(
                f"| {tau} | {group_name.replace('_', ' ')} | {group['instances']} "
                f"| {group['lift_success_rate']:.1%} "
                f"| {group['continuation_search_failure_rate']:.1%} "
                f"| {group['mean_calibration_posewise_ik_fraction']:.1%} |"
            )
    lines.extend(["", "## By Surface", ""])
    for tau, tau_summary in summary["by_tau"].items():
        lines.extend([
            f"### Tau = {tau} degrees",
            "",
            "| Surface | N | Lift success | Continuation search failure | Failure counts |",
            "|:---|---:|---:|---:|:---|",
        ])
        for surface, group in tau_summary["by_surface"].items():
            calibrated_group = tau_summary["by_surface_fully_posewise_calibrated"][surface]
            counts = ", ".join(f"{key}: {value}" for key, value in sorted(group["failure_counts"].items()))
            lines.append(
                f"| {surface} | {group['instances']} | {group['lift_success_rate']:.1%} "
                f"| {group['continuation_search_failure_rate']:.1%} | {counts} |"
            )
            if calibrated_group["instances"] != group["instances"]:
                calibrated_counts = ", ".join(
                    f"{key}: {value}"
                    for key, value in sorted(calibrated_group["failure_counts"].items())
                )
                lines.append(
                    f"| {surface} (posewise calibrated) | {calibrated_group['instances']} "
                    f"| {calibrated_group['lift_success_rate']:.1%} "
                    f"| {calibrated_group['continuation_search_failure_rate']:.1%} "
                    f"| {calibrated_counts} |"
                )
        lines.append("")
    lines.extend(["## Interpretation Limits", ""])
    lines.extend(f"- {item}" for item in summary["interpretation_limits"])
    return "\n".join(lines) + "\n"


def plot_results(summary: dict[str, object], path: Path) -> None:
    surfaces = summary["surfaces"]
    taus = summary["taus_degrees"]
    x = np.arange(len(surfaces), dtype=float)
    width = 0.8 / len(taus)
    figure, (axis_success, axis_failure) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for index, tau in enumerate(taus):
        groups = summary["by_tau"][str(tau)]["by_surface"]
        offset = (index - (len(taus) - 1) / 2) * width
        success = [groups[surface]["lift_success_rate"] for surface in surfaces]
        failure = [groups[surface]["continuation_search_failure_rate"] for surface in surfaces]
        axis_success.bar(x + offset, success, width, label=f"tau={tau:g} deg")
        axis_failure.bar(x + offset, failure, width, label=f"tau={tau:g} deg")
    for axis, title, ylabel in (
        (axis_success, "Continuous lift success", "Success rate"),
        (axis_failure, "Numerical continuation failures", "Failure rate"),
    ):
        axis.set_xticks(x, surfaces, rotation=20, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
