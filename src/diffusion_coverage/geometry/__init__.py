"""Geometry utilities shared by robot-aware surface experiments."""

from diffusion_coverage.geometry.robot_surface_metric import (
    RobotSurfaceMetric,
    compute_robot_surface_metric,
    estimate_surface_contact_differential,
    orthonormal_surface_tangent,
)
from diffusion_coverage.geometry.surface_curve import SurfaceCurve, trace_surface_curve

__all__ = [
    "RobotSurfaceMetric",
    "SurfaceCurve",
    "compute_robot_surface_metric",
    "estimate_surface_contact_differential",
    "orthonormal_surface_tangent",
    "trace_surface_curve",
]
