from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoverageMetrics, CoveragePlan
from diffusion_coverage.coverage.evaluator import evaluate_coverage
from diffusion_coverage.surface.geodesic import resample_projected_polyline, surface_sample_distances
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class CoverageRefinementResult:
    plan: CoveragePlan
    metrics: CoverageMetrics
    initial_metrics: CoverageMetrics
    steps: int


def refine_coverage_by_insertion(
    surface: SurfaceInstance,
    plan: CoveragePlan,
    *,
    footprint_radius: float,
    missed_tolerance: float,
    max_steps: int = 16,
) -> CoverageRefinementResult:
    """Repair missed regions by inserting farthest surface samples into one path."""

    if plan.num_segments != 1:
        raise ValueError("coverage insertion refinement currently supports one segment")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    path = project_points(surface, plan.active_paths()[0]).points
    initial_metrics = evaluate_coverage(surface, CoveragePlan(path), footprint_radius=footprint_radius)
    metrics = initial_metrics
    steps = 0
    while metrics.missed_fraction > missed_tolerance + 1e-12 and steps < max_steps:
        sources = resample_projected_polyline(
            surface, path, max_spacing=0.5 * footprint_radius
        )
        distances = surface_sample_distances(surface, sources)
        uncovered = distances > footprint_radius + 1e-12
        if not np.any(uncovered):
            break
        uncovered_indices = np.flatnonzero(uncovered)
        target_index = int(uncovered_indices[np.argmax(distances[uncovered_indices])])
        target = surface.sample_points[target_index]
        insertion_index = _minimum_increment_insertion(path, target)
        path = np.insert(path, insertion_index, target, axis=0)
        metrics = evaluate_coverage(surface, CoveragePlan(path), footprint_radius=footprint_radius)
        steps += 1
    return CoverageRefinementResult(
        plan=CoveragePlan(path, metadata={**plan.metadata, "coverage_repair_steps": steps}),
        metrics=metrics,
        initial_metrics=initial_metrics,
        steps=steps,
    )


def refine_coverage_by_shortcutting(
    surface: SurfaceInstance,
    plan: CoveragePlan,
    *,
    footprint_radius: float,
    missed_tolerance: float,
    max_passes: int = 8,
    strides: tuple[int, ...] = (2, 3, 4),
) -> CoverageRefinementResult:
    """Remove redundant waypoints while preserving hard finite-footprint feasibility."""

    if plan.num_segments != 1:
        raise ValueError("coverage shortcutting currently supports one segment")
    if max_passes < 0:
        raise ValueError("max_passes must be nonnegative")
    if not strides or any(stride < 2 for stride in strides):
        raise ValueError("shortcut strides must all be at least two")

    path = project_points(surface, plan.active_paths()[0]).points
    initial_metrics = evaluate_coverage(
        surface, CoveragePlan(path), footprint_radius=footprint_radius
    )
    metrics = initial_metrics
    if metrics.missed_fraction > missed_tolerance + 1e-12:
        return CoverageRefinementResult(
            plan=CoveragePlan(path, metadata={**plan.metadata, "shortcut_passes": 0}),
            metrics=metrics,
            initial_metrics=initial_metrics,
            steps=0,
        )

    accepted = 0
    for _ in range(max_passes):
        feasible_candidates: list[tuple[float, np.ndarray, CoverageMetrics]] = []
        for stride in strides:
            if len(path) <= stride + 1:
                continue
            for phase in range(stride):
                indices = np.unique(
                    np.concatenate(
                        ([0], np.arange(1 + phase, len(path) - 1, stride), [len(path) - 1])
                    )
                )
                if len(indices) < 2 or len(indices) >= len(path):
                    continue
                candidate_path = path[indices]
                candidate_metrics = evaluate_coverage(
                    surface,
                    CoveragePlan(candidate_path),
                    footprint_radius=footprint_radius,
                )
                if (
                    candidate_metrics.missed_fraction <= missed_tolerance + 1e-12
                    and candidate_metrics.path_length < metrics.path_length - 1e-10
                ):
                    feasible_candidates.append(
                        (candidate_metrics.path_length, candidate_path, candidate_metrics)
                    )
        if not feasible_candidates:
            break
        _, path, metrics = min(feasible_candidates, key=lambda item: item[0])
        accepted += 1

    return CoverageRefinementResult(
        plan=CoveragePlan(path, metadata={**plan.metadata, "shortcut_passes": accepted}),
        metrics=metrics,
        initial_metrics=initial_metrics,
        steps=accepted,
    )


def _minimum_increment_insertion(path: np.ndarray, target: np.ndarray) -> int:
    start = path[:-1]
    end = path[1:]
    added_length = (
        np.linalg.norm(start - target, axis=1)
        + np.linalg.norm(end - target, axis=1)
        - np.linalg.norm(end - start, axis=1)
    )
    return int(np.argmin(added_length)) + 1
