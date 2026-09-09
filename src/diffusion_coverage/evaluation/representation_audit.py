from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from diffusion_coverage.coverage import CoveragePlan, evaluate_coverage
from diffusion_coverage.representation import canonicalize_surface_path, fit_fixed_control_bspline
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class RepresentationAuditResult:
    num_control_points: int
    geometry_error: float
    delta_missed_fraction: float
    delta_path_length: float
    relative_path_length_error: float


def audit_fixed_control_bspline(
    surface: SurfaceInstance,
    path: np.ndarray,
    *,
    footprint_radius: float,
    num_reference_tokens: int,
    num_control_points: int,
) -> RepresentationAuditResult:
    return audit_fixed_control_bsplines(
        surface,
        path,
        footprint_radius=footprint_radius,
        num_reference_tokens=num_reference_tokens,
        control_point_counts=[num_control_points],
    )[0]


def audit_fixed_control_bsplines(
    surface: SurfaceInstance,
    path: np.ndarray,
    *,
    footprint_radius: float,
    num_reference_tokens: int,
    control_point_counts: list[int] | tuple[int, ...],
) -> list[RepresentationAuditResult]:
    reference = canonicalize_surface_path(
        surface, path, num_tokens=num_reference_tokens
    )
    reference_metrics = evaluate_coverage(
        surface, CoveragePlan(reference.points), footprint_radius=footprint_radius
    )
    results: list[RepresentationAuditResult] = []
    for num_control_points in control_point_counts:
        spline = fit_fixed_control_bspline(
            reference.points,
            reference.normalized_arclength,
            num_control_points=num_control_points,
        )
        reconstructed = project_points(
            surface, spline.evaluate(reference.normalized_arclength)
        ).points
        reconstructed_metrics = evaluate_coverage(
            surface, CoveragePlan(reconstructed), footprint_radius=footprint_radius
        )
        delta_length = reconstructed_metrics.path_length - reference_metrics.path_length
        results.append(
            RepresentationAuditResult(
                num_control_points=num_control_points,
                geometry_error=symmetric_modified_hausdorff(reference.points, reconstructed),
                delta_missed_fraction=reconstructed_metrics.missed_fraction - reference_metrics.missed_fraction,
                delta_path_length=delta_length,
                relative_path_length_error=delta_length / reference_metrics.path_length,
            )
        )
    return results


def symmetric_modified_hausdorff(path_a: np.ndarray, path_b: np.ndarray) -> float:
    first = np.asarray(path_a, dtype=np.float64)
    second = np.asarray(path_b, dtype=np.float64)
    distance_a = cKDTree(second).query(first, k=1)[0].mean()
    distance_b = cKDTree(first).query(second, k=1)[0].mean()
    return float(max(distance_a, distance_b))
