from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoveragePlan
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class PatternProposal:
    name: str
    plan: CoveragePlan
    metadata: dict[str, float | int | str]


PATTERN_MODE_NAMES = (
    "raster_u_phase_0.00",
    "raster_u_phase_0.25",
    "raster_v_phase_0.00",
    "spiral_phase_0.00",
    "spiral_phase_0.25",
)


@dataclass(frozen=True)
class PatternMode:
    name: str
    family: str
    sweep_axis: str | None
    phase: float

    @property
    def mode_id(self) -> int:
        return PATTERN_MODE_NAMES.index(self.name)


def parse_pattern_mode(name: str) -> PatternMode:
    canonical = name.removesuffix("_robustness_repair")
    if canonical not in PATTERN_MODE_NAMES:
        raise ValueError(f"unsupported pattern mode: {name!r}")
    prefix, phase_text = canonical.split("_phase_")
    if prefix.startswith("raster_"):
        family = "raster"
        sweep_axis = prefix.removeprefix("raster_")
    else:
        family = "spiral"
        sweep_axis = None
    return PatternMode(canonical, family, sweep_axis, float(phase_text))


def generate_pattern_proposals(
    surface: SurfaceInstance,
    *,
    footprint_radius: float,
    max_segments: int = 1,
    overlap: float = 0.75,
    waypoint_spacing: float | None = None,
) -> list[PatternProposal]:
    if surface.surface_id not in _SUPPORTED_SURFACES:
        raise ValueError(f"no analytic pattern parameterization for surface_id={surface.surface_id!r}")
    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius must be positive")
    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    if not 0.0 < overlap <= 1.0:
        raise ValueError("overlap must lie in (0, 1]")
    spacing = 0.75 * footprint_radius if waypoint_spacing is None else waypoint_spacing
    if spacing <= 0.0:
        raise ValueError("waypoint_spacing must be positive")

    proposals = [
        raster_pattern(
            surface,
            footprint_radius=footprint_radius,
            max_segments=max_segments,
            overlap=overlap,
            waypoint_spacing=spacing,
            sweep_axis="u",
            phase=0.0,
        )
    ]
    periodic_u, periodic_v = _periodicity(surface)
    if periodic_u:
        proposals.append(
            raster_pattern(
                surface,
                footprint_radius=footprint_radius,
                max_segments=max_segments,
                overlap=overlap,
                waypoint_spacing=spacing,
                sweep_axis="u",
                phase=0.25,
            )
        )
    proposals.append(
        raster_pattern(
            surface,
            footprint_radius=footprint_radius,
            max_segments=max_segments,
            overlap=overlap,
            waypoint_spacing=spacing,
            sweep_axis="v",
            phase=0.0,
        )
    )
    if periodic_v:
        proposals.append(
            raster_pattern(
                surface,
                footprint_radius=footprint_radius,
                max_segments=max_segments,
                overlap=overlap,
                waypoint_spacing=spacing,
                sweep_axis="v",
                phase=0.25,
            )
        )
    proposals.append(
        spiral_pattern(
            surface,
            footprint_radius=footprint_radius,
            max_segments=max_segments,
            overlap=overlap,
            waypoint_spacing=spacing,
            phase=0.0,
        )
    )
    if periodic_u:
        proposals.append(
            spiral_pattern(
                surface,
                footprint_radius=footprint_radius,
                max_segments=max_segments,
                overlap=overlap,
                waypoint_spacing=spacing,
                phase=0.25,
            )
        )
    return proposals


def raster_pattern(
    surface: SurfaceInstance,
    *,
    footprint_radius: float,
    max_segments: int = 1,
    overlap: float = 0.75,
    waypoint_spacing: float | None = None,
    sweep_axis: str = "u",
    phase: float = 0.0,
) -> PatternProposal:
    if sweep_axis not in {"u", "v"}:
        raise ValueError("sweep_axis must be 'u' or 'v'")
    spacing = 0.75 * footprint_radius if waypoint_spacing is None else waypoint_spacing
    u_extent, v_extent = _intrinsic_extents(surface)
    periodic_u, periodic_v = _periodicity(surface)
    along_extent = u_extent if sweep_axis == "u" else v_extent
    cross_extent = v_extent if sweep_axis == "u" else u_extent
    along_periodic = periodic_u if sweep_axis == "u" else periodic_v
    cross_periodic = periodic_v if sweep_axis == "u" else periodic_u
    num_tracks = max(1, int(np.ceil(cross_extent / (2.0 * footprint_radius * overlap))))
    num_along = max(2, int(np.ceil(along_extent / spacing)) + 1)
    if cross_periodic:
        cross_coordinates = np.arange(num_tracks, dtype=np.float64) / num_tracks
    else:
        cross_coordinates = (np.arange(num_tracks, dtype=np.float64) + 0.5) / num_tracks
    if along_periodic:
        along_coordinates = np.linspace(phase, phase + 1.0, num_along)
    else:
        along_coordinates = np.linspace(0.0, 1.0, num_along)

    tracks: list[np.ndarray] = []
    for track_index, cross_coordinate in enumerate(cross_coordinates):
        along = along_coordinates if track_index % 2 == 0 else along_coordinates[::-1]
        if sweep_axis == "u":
            u = along
            v = np.full_like(along, cross_coordinate)
        else:
            u = np.full_like(along, cross_coordinate)
            v = along
        tracks.append(map_surface_parameters(surface, u, v))
    plan = paths_to_plan(_group_tracks(tracks, max_segments))
    name = f"raster_{sweep_axis}_phase_{phase:.2f}"
    return PatternProposal(
        name=name,
        plan=CoveragePlan(plan.waypoints, plan.segment_mask, plan.waypoint_mask, metadata={"pattern": name}),
        metadata={"num_tracks": num_tracks, "sweep_axis": sweep_axis, "phase": phase},
    )


def raster_control_token_count(
    surface: SurfaceInstance,
    *,
    footprint_radius: float,
    overlap: float,
    sweep_axis: str = "u",
) -> int:
    """Return the endpoint-control count for a known raster task geometry."""

    if sweep_axis not in {"u", "v"}:
        raise ValueError("sweep_axis must be 'u' or 'v'")
    if footprint_radius <= 0.0 or not 0.0 < overlap <= 1.0:
        raise ValueError("invalid footprint radius or overlap")
    u_extent, v_extent = _intrinsic_extents(surface)
    cross_extent = v_extent if sweep_axis == "u" else u_extent
    num_tracks = max(1, int(np.ceil(cross_extent / (2.0 * footprint_radius * overlap))))
    return 2 * num_tracks


def _raster_template_parameter_controls(
    surface: SurfaceInstance,
    *,
    footprint_radius: float,
    overlap: float,
    sweep_axis: str,
    phase: float,
) -> np.ndarray:
    """Construct raster endpoint controls directly in the analytic chart."""

    u_extent, v_extent = _intrinsic_extents(surface)
    periodic_u, periodic_v = _periodicity(surface)
    along_extent = u_extent if sweep_axis == "u" else v_extent
    cross_extent = v_extent if sweep_axis == "u" else u_extent
    along_periodic = periodic_u if sweep_axis == "u" else periodic_v
    cross_periodic = periodic_v if sweep_axis == "u" else periodic_u
    num_tracks = max(
        1, int(np.ceil(cross_extent / (2.0 * footprint_radius * overlap)))
    )
    if cross_periodic:
        cross_coordinates = np.arange(num_tracks, dtype=np.float64) / num_tracks
    else:
        cross_coordinates = (
            np.arange(num_tracks, dtype=np.float64) + 0.5
        ) / num_tracks
    along_endpoints = np.asarray(
        [phase, phase + 1.0] if along_periodic else [0.0, 1.0],
        dtype=np.float64,
    )
    controls = []
    for track_index, cross_coordinate in enumerate(cross_coordinates):
        along = along_endpoints if track_index % 2 == 0 else along_endpoints[::-1]
        if sweep_axis == "u":
            controls.extend((
                [along[0], cross_coordinate],
                [along[1], cross_coordinate],
            ))
        else:
            controls.extend((
                [cross_coordinate, along[0]],
                [cross_coordinate, along[1]],
            ))
    return np.asarray(controls, dtype=np.float64)


def spiral_pattern(
    surface: SurfaceInstance,
    *,
    footprint_radius: float,
    max_segments: int = 1,
    overlap: float = 0.75,
    waypoint_spacing: float | None = None,
    phase: float = 0.0,
) -> PatternProposal:
    spacing = 0.75 * footprint_radius if waypoint_spacing is None else waypoint_spacing
    u_extent, v_extent = _intrinsic_extents(surface)
    if surface.surface_id in {"plane", "saddle", "freeform_patch"}:
        uv = _rectangular_spiral_parameters(
            u_extent=u_extent,
            v_extent=v_extent,
            track_spacing=2.0 * footprint_radius * overlap,
            waypoint_spacing=spacing,
        )
        points = map_surface_parameters(surface, uv[:, 0], uv[:, 1])
    else:
        turns = max(1, int(np.ceil(v_extent / (2.0 * footprint_radius * overlap))))
        approximate_length = np.hypot(turns * u_extent, v_extent)
        num_points = max(3, int(np.ceil(approximate_length / spacing)) + 1)
        t = np.linspace(0.0, 1.0, num_points)
        if surface.surface_id == "torus":
            u = phase + turns * t
            v = t
        else:
            u = phase + turns * t
            v = t
        points = map_surface_parameters(surface, u, v)
    paths = [chunk for chunk in np.array_split(points, min(max_segments, max(1, len(points) // 2))) if len(chunk) >= 2]
    plan = paths_to_plan(paths)
    name = f"spiral_phase_{phase:.2f}"
    return PatternProposal(
        name=name,
        plan=CoveragePlan(plan.waypoints, plan.segment_mask, plan.waypoint_mask, metadata={"pattern": name}),
        metadata={"phase": phase},
    )


def map_surface_parameters(surface: SurfaceInstance, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u_array = np.asarray(u, dtype=np.float64)
    v_array = np.asarray(v, dtype=np.float64)
    if u_array.shape != v_array.shape:
        raise ValueError("u and v must have matching shapes")
    periodic_u, periodic_v = _periodicity(surface)
    u_eval = np.mod(u_array, 1.0) if periodic_u else np.clip(u_array, 0.0, 1.0)
    v_eval = np.mod(v_array, 1.0) if periodic_v else np.clip(v_array, 0.0, 1.0)
    metadata = surface.metadata

    if surface.surface_id == "plane":
        x = (u_eval - 0.5) * float(metadata["width"])
        y = (v_eval - 0.5) * float(metadata["height"])
        z = np.zeros_like(x)
    elif surface.surface_id == "saddle":
        x = (u_eval - 0.5) * float(metadata["width"])
        y = (v_eval - 0.5) * float(metadata["height"])
        z = float(metadata["curvature"]) * (x * x - y * y)
    elif surface.surface_id == "freeform_patch":
        width = float(metadata["width"])
        height = float(metadata["height"])
        amplitude = float(metadata["amplitude"])
        x = (u_eval - 0.5) * width
        y = (v_eval - 0.5) * height
        z = amplitude * (np.sin(np.pi * x / width) * np.cos(2.0 * np.pi * y / height) + 0.35 * x * y)
    elif surface.surface_id == "cylinder":
        radius = float(metadata["radius"])
        theta = 2.0 * np.pi * u_eval
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = (v_eval - 0.5) * float(metadata["height"])
    elif surface.surface_id == "hemisphere":
        radius = float(metadata["radius"])
        azimuth = 2.0 * np.pi * u_eval
        polar = 0.5 * np.pi * v_eval
        x = radius * np.sin(polar) * np.cos(azimuth)
        y = radius * np.sin(polar) * np.sin(azimuth)
        z = radius * np.cos(polar)
    elif surface.surface_id == "torus":
        major_radius = float(metadata["major_radius"])
        minor_radius = float(metadata["minor_radius"])
        theta = 2.0 * np.pi * u_eval
        phi = 2.0 * np.pi * v_eval
        radial = major_radius + minor_radius * np.cos(phi)
        x = radial * np.cos(theta)
        y = radial * np.sin(theta)
        z = minor_radius * np.sin(phi)
    else:
        raise ValueError(f"no analytic parameterization for surface_id={surface.surface_id!r}")
    return np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))


def inverse_surface_parameters(
    surface: SurfaceInstance,
    points: np.ndarray,
    *,
    unwrap_periodic: bool = True,
) -> np.ndarray:
    """Map ordered 3D surface points to the analytic chart used by pattern generation."""

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape [M, 3]")
    metadata = surface.metadata
    if surface.surface_id in {"plane", "saddle", "freeform_patch"}:
        u = xyz[:, 0] / float(metadata["width"]) + 0.5
        v = xyz[:, 1] / float(metadata["height"]) + 0.5
    elif surface.surface_id == "cylinder":
        u = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]) / (2.0 * np.pi), 1.0)
        v = xyz[:, 2] / float(metadata["height"]) + 0.5
    elif surface.surface_id == "hemisphere":
        radius = float(metadata["radius"])
        u = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]) / (2.0 * np.pi), 1.0)
        v = 2.0 * np.arccos(np.clip(xyz[:, 2] / radius, -1.0, 1.0)) / np.pi
    elif surface.surface_id == "torus":
        major_radius = float(metadata["major_radius"])
        radial = np.hypot(xyz[:, 0], xyz[:, 1])
        u = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]) / (2.0 * np.pi), 1.0)
        v = np.mod(
            np.arctan2(xyz[:, 2], radial - major_radius) / (2.0 * np.pi), 1.0
        )
    else:
        raise ValueError(f"no analytic parameterization for surface_id={surface.surface_id!r}")
    periodic_u, periodic_v = _periodicity(surface)
    if unwrap_periodic and periodic_u:
        u = np.unwrap(2.0 * np.pi * u) / (2.0 * np.pi)
    if unwrap_periodic and periodic_v:
        v = np.unwrap(2.0 * np.pi * v) / (2.0 * np.pi)
    return np.column_stack((u, v))


def simplify_parameter_polyline(
    parameters: np.ndarray,
    *,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Reduce an ordered UV polyline to its geometric control points."""

    points = np.asarray(parameters, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("parameters must have shape [M, 2] with M >= 2")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        denominator = float(np.dot(segment, segment))
        interior = points[start + 1 : end]
        if denominator <= 1e-20:
            distances = np.linalg.norm(interior - points[start], axis=1)
        else:
            projection = np.clip(
                ((interior - points[start]) @ segment) / denominator, 0.0, 1.0
            )
            closest = points[start] + projection[:, None] * segment
            distances = np.linalg.norm(interior - closest, axis=1)
        local = int(np.argmax(distances))
        if float(distances[local]) > tolerance:
            split = start + 1 + local
            keep[split] = True
            stack.extend(((start, split), (split, end)))
    return points[keep]


def densify_parameter_polyline(
    parameters: np.ndarray,
    *,
    maximum_step: float = 0.025,
) -> np.ndarray:
    """Densify unwrapped UV controls before periodic coordinates map to 3D."""

    points = np.asarray(parameters, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("parameters must have shape [M, 2] with M >= 2")
    if maximum_step <= 0.0:
        raise ValueError("maximum_step must be positive")
    output = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        intervals = max(1, int(np.ceil(np.linalg.norm(end - start) / maximum_step)))
        interpolation = np.linspace(0.0, 1.0, intervals + 1)[1:, None]
        output.extend(start[None, :] + interpolation * (end - start)[None, :])
    return np.asarray(output, dtype=np.float64)


def decode_raster_parameter_controls(
    surface: SurfaceInstance,
    parameters: np.ndarray,
    *,
    footprint_radius: float,
    sweep_axis: str = "u",
    waypoint_spacing: float | None = None,
) -> np.ndarray:
    """Decode alternating raster stroke endpoints without densifying connectors."""

    points = np.asarray(parameters, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("parameters must have shape [M, 2] with M >= 2")
    if sweep_axis not in {"u", "v"} or footprint_radius <= 0.0:
        raise ValueError("invalid sweep axis or footprint radius")
    spacing = 0.75 * footprint_radius if waypoint_spacing is None else waypoint_spacing
    if spacing <= 0.0:
        raise ValueError("waypoint_spacing must be positive")
    u_extent, v_extent = _intrinsic_extents(surface)
    output = [points[0]]
    for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        if segment_index % 2 == 0:
            delta = end - start
            physical_length = np.hypot(delta[0] * u_extent, delta[1] * v_extent)
            intervals = max(1, int(np.ceil(physical_length / spacing)))
        else:
            intervals = 1
        interpolation = np.linspace(0.0, 1.0, intervals + 1)[1:, None]
        output.extend(start[None, :] + interpolation * (end - start)[None, :])
    return np.asarray(output, dtype=np.float64)


def extract_structured_parameter_controls(
    surface: SurfaceInstance,
    path: np.ndarray,
    *,
    mode_name: str,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Extract topology-bearing UV controls for a supported pattern mode."""

    mode = parse_pattern_mode(mode_name)
    parameters = inverse_surface_parameters(surface, path, unwrap_periodic=True)
    if mode.family == "spiral" and surface.surface_id == "hemisphere":
        # Azimuth is undefined at the pole. Fix it to the requested phase so
        # numerical near-pole samples do not create spurious control points.
        start = np.asarray([mode.phase, 0.0], dtype=np.float64)
        return np.vstack((start, parameters[-1]))
    if (
        mode.family == "raster"
        and mode.sweep_axis == "v"
        and surface.surface_id == "hemisphere"
    ):
        parameters = _fix_hemisphere_pole_gauge(parameters)
    return simplify_parameter_polyline(parameters, tolerance=tolerance)


def _fix_hemisphere_pole_gauge(parameters: np.ndarray) -> np.ndarray:
    """Assign each duplicated pole endpoint to its adjacent raster track."""

    fixed = np.asarray(parameters, dtype=np.float64).copy()
    pole = fixed[:, 1] <= 1e-8
    indices = np.flatnonzero(pole)
    if not len(indices):
        return fixed
    run_starts = indices[np.r_[True, np.diff(indices) > 1]]
    run_ends = indices[np.r_[np.diff(indices) > 1, True]]
    for start, end in zip(run_starts, run_ends):
        previous_u = fixed[start - 1, 0] if start > 0 else None
        following_u = fixed[end + 1, 0] if end + 1 < len(fixed) else None
        if previous_u is None:
            fixed[start : end + 1, 0] = following_u
        elif following_u is None:
            fixed[start : end + 1, 0] = previous_u
        else:
            split = start + (end - start + 2) // 2
            fixed[start:split, 0] = previous_u
            fixed[split : end + 1, 0] = following_u
    return fixed


def structured_control_token_count(
    surface: SurfaceInstance,
    *,
    footprint_radius: float,
    overlap: float,
    mode_name: str,
) -> int:
    """Derive a structured control count from known task geometry and mode."""

    mode = parse_pattern_mode(mode_name)
    if mode.family == "raster":
        assert mode.sweep_axis is not None
        return raster_control_token_count(
            surface,
            footprint_radius=footprint_radius,
            overlap=overlap,
            sweep_axis=mode.sweep_axis,
        )
    proposal = spiral_pattern(
        surface,
        footprint_radius=footprint_radius,
        overlap=overlap,
        phase=mode.phase,
    )
    controls = extract_structured_parameter_controls(
        surface,
        proposal.plan.active_paths()[0],
        mode_name=mode.name,
    )
    return len(controls)


def structured_template_controls(
    surface: SurfaceInstance,
    *,
    footprint_radius: float,
    overlap: float,
    mode_name: str,
) -> np.ndarray:
    """Reconstruct the deterministic structured controls for one task and mode."""
    mode = parse_pattern_mode(mode_name)
    if mode.family == "raster":
        assert mode.sweep_axis is not None
        return _raster_template_parameter_controls(
            surface,
            footprint_radius=footprint_radius,
            overlap=overlap,
            sweep_axis=str(mode.sweep_axis),
            phase=mode.phase,
        )
    proposal = spiral_pattern(
        surface,
        footprint_radius=footprint_radius,
        overlap=overlap,
        phase=mode.phase,
    )
    return extract_structured_parameter_controls(
        surface, proposal.plan.active_paths()[0], mode_name=mode.name
    )


def structured_controls_to_residual(
    surface: SurfaceInstance,
    controls: np.ndarray,
    *,
    footprint_radius: float,
    overlap: float,
    mode_name: str,
) -> np.ndarray:
    """Express controls as physical template residuals in footprint-radius units."""
    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius must be positive")
    values = np.asarray(controls, dtype=np.float32)
    template = structured_template_controls(
        surface,
        footprint_radius=footprint_radius,
        overlap=overlap,
        mode_name=mode_name,
    ).astype(np.float32)
    if values.shape != template.shape:
        raise ValueError("controls do not match the structured template shape")
    delta = values - template
    for axis, periodic in enumerate(_periodicity(surface)):
        if periodic:
            delta[:, axis] -= np.round(np.median(delta[:, axis]))
    scale = np.asarray(_intrinsic_extents(surface), dtype=np.float32) / np.float32(
        footprint_radius
    )
    return (delta * scale).astype(np.float32)


def structured_residual_to_controls(
    surface: SurfaceInstance,
    residual: np.ndarray,
    *,
    footprint_radius: float,
    overlap: float,
    mode_name: str,
) -> np.ndarray:
    """Decode footprint-normalized residuals back to unwrapped UV controls."""
    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius must be positive")
    values = np.asarray(residual, dtype=np.float32)
    template = structured_template_controls(
        surface,
        footprint_radius=footprint_radius,
        overlap=overlap,
        mode_name=mode_name,
    ).astype(np.float32)
    if values.shape != template.shape:
        raise ValueError("residual does not match the structured template shape")
    scale = np.asarray(_intrinsic_extents(surface), dtype=np.float32) / np.float32(
        footprint_radius
    )
    return (template + values / scale).astype(np.float32).astype(np.float64)


def decode_structured_parameter_controls(
    surface: SurfaceInstance,
    parameters: np.ndarray,
    *,
    footprint_radius: float,
    mode_name: str,
) -> np.ndarray:
    """Decode structured mode controls to a dense unwrapped UV path."""

    mode = parse_pattern_mode(mode_name)
    if mode.family == "raster":
        assert mode.sweep_axis is not None
        return decode_raster_parameter_controls(
            surface,
            parameters,
            footprint_radius=footprint_radius,
            sweep_axis=mode.sweep_axis,
        )
    points = np.asarray(parameters, dtype=np.float64)
    u_extent, v_extent = _intrinsic_extents(surface)
    spacing = 0.75 * footprint_radius
    output = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        physical_length = np.hypot(delta[0] * u_extent, delta[1] * v_extent)
        intervals = max(1, int(np.ceil(physical_length / spacing)))
        interpolation = np.linspace(0.0, 1.0, intervals + 1)[1:, None]
        output.extend(start[None, :] + interpolation * (end - start)[None, :])
    return np.asarray(output, dtype=np.float64)


def paths_to_plan(paths: list[np.ndarray]) -> CoveragePlan:
    if not paths:
        raise ValueError("at least one path is required")
    arrays = [np.asarray(path, dtype=np.float64) for path in paths]
    if any(path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 for path in arrays):
        raise ValueError("each path must have shape [M, 3] with M >= 2")
    max_waypoints = max(len(path) for path in arrays)
    waypoints = np.zeros((len(arrays), max_waypoints, 3), dtype=np.float64)
    waypoint_mask = np.zeros((len(arrays), max_waypoints), dtype=bool)
    for index, path in enumerate(arrays):
        waypoints[index, : len(path)] = path
        waypoint_mask[index, : len(path)] = True
    return CoveragePlan(waypoints, np.ones(len(arrays), dtype=bool), waypoint_mask)


def _group_tracks(tracks: list[np.ndarray], max_segments: int) -> list[np.ndarray]:
    groups = np.array_split(np.arange(len(tracks)), min(max_segments, len(tracks)))
    return [np.concatenate([tracks[int(index)] for index in group], axis=0) for group in groups if len(group)]


def _rectangular_spiral_parameters(
    *,
    u_extent: float,
    v_extent: float,
    track_spacing: float,
    waypoint_spacing: float,
) -> np.ndarray:
    du = min(0.45, track_spacing / u_extent)
    dv = min(0.45, track_spacing / v_extent)
    corners: list[tuple[float, float]] = []
    left = bottom = 0.0
    right = top = 1.0
    while left <= right and bottom <= top:
        corners.extend(((left, bottom), (right, bottom), (right, top), (left, top)))
        left += du
        right -= du
        bottom += dv
        top -= dv
    if len(corners) < 2:
        corners = [(0.0, 0.5), (1.0, 0.5)]

    samples: list[np.ndarray] = [np.asarray(corners[0], dtype=np.float64)]
    for start, end in zip(corners[:-1], corners[1:]):
        delta = np.asarray(end) - np.asarray(start)
        physical_length = np.hypot(delta[0] * u_extent, delta[1] * v_extent)
        intervals = max(1, int(np.ceil(physical_length / waypoint_spacing)))
        t = np.linspace(0.0, 1.0, intervals + 1)[1:, None]
        samples.extend(np.asarray(start)[None, :] + t * delta[None, :])
    return np.asarray(samples, dtype=np.float64)


def _intrinsic_extents(surface: SurfaceInstance) -> tuple[float, float]:
    metadata = surface.metadata
    if surface.surface_id in {"plane", "saddle", "freeform_patch"}:
        return float(metadata["width"]), float(metadata["height"])
    if surface.surface_id == "cylinder":
        return 2.0 * np.pi * float(metadata["radius"]), float(metadata["height"])
    if surface.surface_id == "hemisphere":
        radius = float(metadata["radius"])
        return 2.0 * np.pi * radius, 0.5 * np.pi * radius
    if surface.surface_id == "torus":
        return (
            2.0 * np.pi * (float(metadata["major_radius"]) + float(metadata["minor_radius"])),
            2.0 * np.pi * float(metadata["minor_radius"]),
        )
    raise ValueError(f"no intrinsic extents for surface_id={surface.surface_id!r}")


def _periodicity(surface: SurfaceInstance) -> tuple[bool, bool]:
    if surface.surface_id in {"cylinder", "hemisphere"}:
        return True, False
    if surface.surface_id == "torus":
        return True, True
    return False, False


_SUPPORTED_SURFACES = {"plane", "cylinder", "hemisphere", "saddle", "torus", "freeform_patch"}
