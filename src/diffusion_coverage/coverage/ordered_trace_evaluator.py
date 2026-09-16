from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from diffusion_coverage.coverage.nuc_evaluator import (
    _surface_sample_to_ordered_source_distances,
)
from diffusion_coverage.diagnostics.symmetry_layout import analytical_points_and_normals
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class AnalyticalQuadrature:
    surface_id: str
    level: str
    points: np.ndarray
    weights: np.ndarray
    parameters: np.ndarray
    definition: dict[str, Any]


@dataclass(frozen=True)
class OrderedTraceMetrics:
    missed_lower: float
    missed_upper: float
    repeat_lower: float
    repeat_upper: float
    path_length: float
    episode_min: np.ndarray
    episode_max: np.ndarray
    uncertain_area_fraction: float
    metadata: dict[str, Any]


def resample_prescribed_path(
    original_waypoints: np.ndarray,
    *,
    surface_id: str,
    surface_metadata: dict[str, Any],
    maximum_step: float,
) -> np.ndarray:
    """Sample the E08 chord-then-analytical-projection trajectory.

    Every call starts from the immutable original waypoint intervals. It never
    reconnects samples with a mesh shortest-path routine.
    """

    points = np.asarray(original_waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError("original_waypoints must have shape [T, 3]")
    if maximum_step <= 0.0:
        raise ValueError("maximum_step must be positive")
    projected, _ = analytical_points_and_normals(surface_id, points, surface_metadata)
    if not np.allclose(projected, points, rtol=0.0, atol=1e-10):
        raise ValueError("original waypoint is not on the analytical surface")
    output = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        intervals = max(1, int(np.ceil(np.linalg.norm(end - start) / maximum_step)))
        fractions = np.linspace(0.0, 1.0, intervals + 1)[1:]
        provisional = start[None, :] + fractions[:, None] * (end - start)[None, :]
        sampled, _ = analytical_points_and_normals(surface_id, provisional, surface_metadata)
        output.extend(sampled)
    return np.ascontiguousarray(output, dtype=np.float64)


def evaluate_ordered_trace_mesh(
    surface: SurfaceInstance,
    traces: tuple[np.ndarray, ...],
    *,
    footprint_radius: float,
    chunk_size: int = 64,
) -> OrderedTraceMetrics:
    """Evaluate prescribed ordered traces with the historical mesh-distance backend."""

    _validate_traces(traces, footprint_radius, chunk_size)
    counts = np.zeros(surface.num_samples, dtype=np.int64)
    for start in range(0, surface.num_samples, chunk_size):
        indices = np.arange(start, min(surface.num_samples, start + chunk_size))
        local = np.zeros(len(indices), dtype=np.int64)
        for trace in traces:
            distances = _surface_sample_to_ordered_source_distances(
                surface, trace, sample_indices=indices
            )
            membership = distances <= footprint_radius + 1e-12
            local += _episode_counts(membership)
        counts[indices] = local
    return _metrics_from_episode_bounds(
        surface.area_weights,
        counts,
        counts,
        path_length=sum(_polyline_length(trace) for trace in traces),
        uncertain=np.zeros(surface.num_samples, dtype=bool),
        metadata={"distance_backend": "legacy_mesh_edge_dijkstra", "trace_semantics": "prescribed"},
    )


def evaluate_ordered_trace_euclidean(
    sample_points: np.ndarray,
    weights: np.ndarray,
    traces: tuple[np.ndarray, ...],
    *,
    footprint_radius: float,
    chunk_size: int = 256,
) -> OrderedTraceMetrics:
    """Exact Euclidean footprint evaluation for planar analytical references."""

    _validate_traces(traces, footprint_radius, chunk_size)
    samples = np.asarray(sample_points, dtype=np.float64)
    area = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or area.shape != (len(samples),):
        raise ValueError("sample points and weights have incompatible shapes")
    counts = np.zeros(len(samples), dtype=np.int64)
    for start in range(0, len(samples), chunk_size):
        stop = min(len(samples), start + chunk_size)
        local = np.zeros(stop - start, dtype=np.int64)
        for trace in traces:
            distances = np.linalg.norm(samples[start:stop, None, :] - trace[None, :, :], axis=2)
            local += _episode_counts(distances <= footprint_radius + 1e-12)
        counts[start:stop] = local
    return _metrics_from_episode_bounds(
        area,
        counts,
        counts,
        path_length=sum(_polyline_length(trace) for trace in traces),
        uncertain=np.zeros(len(samples), dtype=bool),
        metadata={"distance_backend": "exact_euclidean", "trace_semantics": "prescribed"},
    )


def make_analytical_quadrature(
    surface_id: str,
    surface_metadata: dict[str, Any],
    level: str,
) -> AnalyticalQuadrature:
    """Return the preregistered nested-cell midpoint quadrature Q0/Q1/Q2."""

    if level not in {"Q0", "Q1", "Q2"}:
        raise ValueError("level must be Q0, Q1, or Q2")
    scale = 2 ** int(level[-1])
    if surface_id == "saddle":
        nx = ny = 48 * scale
        width = float(surface_metadata["width"])
        height = float(surface_metadata["height"])
        curvature = float(surface_metadata["curvature"])
        dx, dy = width / nx, height / ny
        x = -0.5 * width + (np.arange(nx) + 0.5) * dx
        y = -0.5 * height + (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")
        parameters = np.column_stack((xx.ravel(), yy.ravel()))
        z = curvature * (parameters[:, 0] ** 2 - parameters[:, 1] ** 2)
        points = np.column_stack((parameters, z))
        density = np.sqrt(
            1.0
            + (2.0 * curvature * parameters[:, 0]) ** 2
            + (2.0 * curvature * parameters[:, 1]) ** 2
        )
        weights = density * dx * dy
        definition = {
            "rule": "nested rectangular cells with midpoint surface-density quadrature",
            "nx": nx,
            "ny": ny,
            "dx_m": dx,
            "dy_m": dy,
        }
    elif surface_id == "hemisphere":
        n_azimuth = 48 * scale
        n_polar = 24 * scale
        radius = float(surface_metadata["radius"])
        dphi = 2.0 * np.pi / n_azimuth
        dtheta = 0.5 * np.pi / n_polar
        phi = (np.arange(n_azimuth) + 0.5) * dphi
        theta = (np.arange(n_polar) + 0.5) * dtheta
        pp, tt = np.meshgrid(phi, theta, indexing="xy")
        parameters = np.column_stack((pp.ravel(), tt.ravel()))
        points = radius * np.column_stack(
            (
                np.sin(parameters[:, 1]) * np.cos(parameters[:, 0]),
                np.sin(parameters[:, 1]) * np.sin(parameters[:, 0]),
                np.cos(parameters[:, 1]),
            )
        )
        polar_lo = np.arange(n_polar) * dtheta
        polar_hi = polar_lo + dtheta
        ring_cell_area = radius**2 * dphi * (np.cos(polar_lo) - np.cos(polar_hi))
        weights = np.repeat(ring_cell_area, n_azimuth)
        definition = {
            "rule": "nested azimuth-polar cells; midpoint locations and exact spherical cell area",
            "n_azimuth": n_azimuth,
            "n_polar": n_polar,
            "equator_azimuth_spacing_m": radius * dphi,
            "polar_spacing_m": radius * dtheta,
        }
    else:
        raise ValueError(f"unsupported analytical reference surface: {surface_id!r}")
    return AnalyticalQuadrature(
        surface_id,
        level,
        np.ascontiguousarray(points),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(parameters),
        definition,
    )


def evaluate_ordered_trace_reference(
    quadrature: AnalyticalQuadrature,
    traces: tuple[np.ndarray, ...],
    *,
    footprint_radius: float,
    surface_metadata: dict[str, Any],
    chunk_size: int = 64,
) -> OrderedTraceMetrics:
    """Evaluate exact sphere distances or certified saddle distance bounds."""

    _validate_traces(traces, footprint_radius, chunk_size)
    n = len(quadrature.points)
    episode_min = np.zeros(n, dtype=np.int64)
    episode_max = np.zeros(n, dtype=np.int64)
    uncertain = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        samples = quadrature.points[start:stop]
        lo_total = np.zeros(stop - start, dtype=np.int64)
        hi_total = np.zeros(stop - start, dtype=np.int64)
        local_uncertain = np.zeros(stop - start, dtype=bool)
        for trace in traces:
            if quadrature.surface_id == "hemisphere":
                radius = float(surface_metadata["radius"])
                fixed = sphere_pairwise_membership(
                    samples, trace, radius=radius, footprint_radius=footprint_radius
                )
                lo = hi = _episode_counts(fixed)
            else:
                lower, upper = saddle_pairwise_distance_bounds(
                    samples, trace, curvature=float(surface_metadata["curvature"])
                )
                definite_in = upper <= footprint_radius
                definite_out = lower > footprint_radius
                allowed_zero = ~definite_in
                allowed_one = ~definite_out
                lo, hi = uncertain_episode_bounds(allowed_zero, allowed_one)
                local_uncertain |= np.any(~(definite_in | definite_out), axis=1)
            lo_total += lo
            hi_total += hi
        episode_min[start:stop] = lo_total
        episode_max[start:stop] = hi_total
        uncertain[start:stop] = local_uncertain
    if quadrature.surface_id == "hemisphere":
        length = sum(sphere_polyline_length(trace, float(surface_metadata["radius"])) for trace in traces)
        backend = "exact_spherical_angular_distance"
    else:
        length = sum(saddle_polyline_length(trace, float(surface_metadata["curvature"])) for trace in traces)
        backend = "saddle_chord_lower_chart_segment_upper"
    return _metrics_from_episode_bounds(
        quadrature.weights,
        episode_min,
        episode_max,
        path_length=length,
        uncertain=uncertain,
        metadata={"distance_backend": backend, "trace_semantics": "prescribed", "quadrature": quadrature.level},
    )


def reference_membership_states(
    surface_id: str,
    sample_points: np.ndarray,
    trace: np.ndarray,
    *,
    footprint_radius: float,
    surface_metadata: dict[str, Any],
) -> np.ndarray:
    """Return 1=in, 0=out, and -1=uncertain for selected reference samples."""

    if surface_id == "hemisphere":
        return sphere_pairwise_membership(
            sample_points,
            trace,
            radius=float(surface_metadata["radius"]),
            footprint_radius=footprint_radius,
        ).astype(np.int8)
    if surface_id == "saddle":
        lower, upper = saddle_pairwise_distance_bounds(
            sample_points, trace, curvature=float(surface_metadata["curvature"])
        )
        states = np.full(lower.shape, -1, dtype=np.int8)
        states[upper <= footprint_radius] = 1
        states[lower > footprint_radius] = 0
        return states
    raise ValueError(f"unsupported surface: {surface_id!r}")


def sphere_pairwise_distances(samples: np.ndarray, sources: np.ndarray, *, radius: float) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64) / radius
    y = np.asarray(sources, dtype=np.float64) / radius
    cross = np.linalg.norm(np.cross(x[:, None, :], y[None, :, :]), axis=2)
    dot = np.einsum("ij,kj->ik", x, y)
    return radius * np.arctan2(cross, np.clip(dot, -1.0, 1.0))


def sphere_pairwise_membership(
    samples: np.ndarray,
    sources: np.ndarray,
    *,
    radius: float,
    footprint_radius: float,
) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64) / radius
    y = np.asarray(sources, dtype=np.float64) / radius
    dot = np.einsum("ij,kj->ik", x, y)
    return dot >= np.cos(footprint_radius / radius) - 1e-14


def saddle_pairwise_distance_bounds(
    samples: np.ndarray, sources: np.ndarray, *, curvature: float
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(samples, dtype=np.float64)[:, None, :]
    end = np.asarray(sources, dtype=np.float64)[None, :, :]
    lower = np.linalg.norm(end - start, axis=2)
    dx = end[..., 0] - start[..., 0]
    dy = end[..., 1] - start[..., 1]
    base = dx * dx + dy * dy
    alpha = 2.0 * curvature * (start[..., 0] * dx - start[..., 1] * dy)
    beta = 2.0 * curvature * (dx * dx - dy * dy)
    upper = _integral_sqrt_quadratic(base, alpha, beta)
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _integral_sqrt_quadratic(base: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    constant = np.abs(beta) <= 1e-14
    result = np.empty_like(base)
    result[constant] = np.sqrt(base[constant] + alpha[constant] ** 2)
    moving = ~constant
    if np.any(moving):
        a = base[moving]
        b = beta[moving]
        u0 = alpha[moving]
        u1 = u0 + b
        root_a = np.sqrt(a)

        def primitive(u: np.ndarray) -> np.ndarray:
            root = np.sqrt(a + u * u)
            asinh = np.where(a > 0.0, np.arcsinh(u / np.maximum(root_a, 1e-300)), 0.0)
            return 0.5 * (u * root + a * asinh)

        result[moving] = (primitive(u1) - primitive(u0)) / b
    return np.maximum(result, np.sqrt(base))


def uncertain_episode_bounds(
    allowed_zero: np.ndarray, allowed_one: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max 0->1 transition counts over all allowed membership sequences."""

    zero = np.asarray(allowed_zero, dtype=bool)
    one = np.asarray(allowed_one, dtype=bool)
    if zero.shape != one.shape or zero.ndim != 2 or zero.shape[1] < 1:
        raise ValueError("allowed masks must have matching shape [samples, time]")
    if np.any(~(zero | one)):
        raise ValueError("every state must allow at least one membership value")
    inf = np.iinfo(np.int32).max // 4
    neg = -inf
    min0 = np.where(zero[:, 0], 0, inf)
    min1 = np.where(one[:, 0], 1, inf)
    max0 = np.where(zero[:, 0], 0, neg)
    max1 = np.where(one[:, 0], 1, neg)
    for t in range(1, zero.shape[1]):
        next_min0 = np.where(zero[:, t], np.minimum(min0, min1), inf)
        next_min1 = np.where(one[:, t], np.minimum(min1, min0 + 1), inf)
        next_max0 = np.where(zero[:, t], np.maximum(max0, max1), neg)
        next_max1 = np.where(one[:, t], np.maximum(max1, max0 + 1), neg)
        min0, min1, max0, max1 = next_min0, next_min1, next_max0, next_max1
    return np.minimum(min0, min1).astype(np.int64), np.maximum(max0, max1).astype(np.int64)


def saddle_polyline_length(trace: np.ndarray, curvature: float) -> float:
    if len(trace) < 2:
        return 0.0
    start = np.asarray(trace[:-1], dtype=np.float64)
    end = np.asarray(trace[1:], dtype=np.float64)
    dx = end[:, 0] - start[:, 0]
    dy = end[:, 1] - start[:, 1]
    base = dx * dx + dy * dy
    alpha = 2.0 * curvature * (start[:, 0] * dx - start[:, 1] * dy)
    beta = 2.0 * curvature * (dx * dx - dy * dy)
    lengths = _integral_sqrt_quadratic(base, alpha, beta)
    return float(np.nextafter(lengths, np.inf).sum())


def sphere_polyline_length(trace: np.ndarray, radius: float) -> float:
    if len(trace) < 2:
        return 0.0
    start = np.asarray(trace[:-1], dtype=np.float64) / radius
    end = np.asarray(trace[1:], dtype=np.float64) / radius
    cross = np.linalg.norm(np.cross(start, end), axis=1)
    dot = np.einsum("ij,ij->i", start, end)
    return float((radius * np.arctan2(cross, np.clip(dot, -1.0, 1.0))).sum())


def _episode_counts(membership: np.ndarray) -> np.ndarray:
    values = np.asarray(membership, dtype=bool)
    starts = values.copy()
    starts[:, 1:] &= ~values[:, :-1]
    return starts.sum(axis=1, dtype=np.int64)


def _metrics_from_episode_bounds(
    weights: np.ndarray,
    episode_min: np.ndarray,
    episode_max: np.ndarray,
    *,
    path_length: float,
    uncertain: np.ndarray,
    metadata: dict[str, Any],
) -> OrderedTraceMetrics:
    area = np.asarray(weights, dtype=np.float64)
    total = float(area.sum())
    missed_lower = float(area[episode_max == 0].sum() / total)
    missed_upper = float(area[episode_min == 0].sum() / total)
    repeat_lower = float(np.dot(area, np.maximum(episode_min - 1, 0)) / total)
    repeat_upper = float(np.dot(area, np.maximum(episode_max - 1, 0)) / total)
    return OrderedTraceMetrics(
        missed_lower,
        missed_upper,
        repeat_lower,
        repeat_upper,
        float(path_length),
        episode_min,
        episode_max,
        float(area[np.asarray(uncertain, bool)].sum() / total),
        metadata,
    )


def _polyline_length(trace: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()) if len(trace) > 1 else 0.0


def _validate_traces(traces: tuple[np.ndarray, ...], radius: float, chunk_size: int) -> None:
    if not traces or radius <= 0.0 or chunk_size < 1:
        raise ValueError("nonempty traces, positive radius, and positive chunk size are required")
    for trace in traces:
        value = np.asarray(trace)
        if value.ndim != 2 or value.shape[1] != 3 or len(value) < 1 or not np.all(np.isfinite(value)):
            raise ValueError("each trace must have finite shape [T, 3]")
