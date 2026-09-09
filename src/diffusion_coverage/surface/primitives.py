from __future__ import annotations

from collections.abc import Callable

import numpy as np

from diffusion_coverage.surface.surface_instance import SurfaceInstance


def make_plane(
    *,
    width: float = 1.0,
    height: float = 1.0,
    nx: int = 16,
    ny: int = 16,
    samples_per_face: int = 4,
) -> SurfaceInstance:
    _require_positive(width=width, height=height)
    _require_resolution(nx=nx, ny=ny)
    x = np.linspace(-0.5 * width, 0.5 * width, nx + 1)
    y = np.linspace(-0.5 * height, 0.5 * height, ny + 1)
    vertices = np.array([[xi, yi, 0.0] for yi in y for xi in x], dtype=np.float64)
    faces = _grid_faces(nx, ny, wrap_x=False, wrap_y=False)
    return SurfaceInstance.from_mesh(
        vertices,
        faces,
        samples_per_face=samples_per_face,
        surface_id="plane",
        metadata={"width": width, "height": height},
    )


def make_cylinder(
    *,
    radius: float = 1.0,
    height: float = 2.0,
    n_azimuth: int = 48,
    n_height: int = 16,
    samples_per_face: int = 4,
) -> SurfaceInstance:
    _require_positive(radius=radius, height=height)
    if n_azimuth < 3 or n_height < 1:
        raise ValueError("cylinder requires n_azimuth >= 3 and n_height >= 1")
    theta = np.arange(n_azimuth, dtype=np.float64) * (2.0 * np.pi / n_azimuth)
    z = np.linspace(-0.5 * height, 0.5 * height, n_height + 1)
    vertices = np.array(
        [[radius * np.cos(t), radius * np.sin(t), zi] for zi in z for t in theta],
        dtype=np.float64,
    )
    faces = _grid_faces(n_azimuth, n_height, wrap_x=True, wrap_y=False)
    return SurfaceInstance.from_mesh(
        vertices,
        faces,
        samples_per_face=samples_per_face,
        surface_id="cylinder",
        metadata={"radius": radius, "height": height},
    )


def make_hemisphere(
    *,
    radius: float = 1.0,
    n_azimuth: int = 48,
    n_polar: int = 16,
    samples_per_face: int = 4,
) -> SurfaceInstance:
    _require_positive(radius=radius)
    if n_azimuth < 3 or n_polar < 1:
        raise ValueError("hemisphere requires n_azimuth >= 3 and n_polar >= 1")
    vertices = [[0.0, 0.0, radius]]
    for polar_index in range(1, n_polar + 1):
        polar = 0.5 * np.pi * polar_index / n_polar
        for azimuth_index in range(n_azimuth):
            azimuth = 2.0 * np.pi * azimuth_index / n_azimuth
            vertices.append(
                [
                    radius * np.sin(polar) * np.cos(azimuth),
                    radius * np.sin(polar) * np.sin(azimuth),
                    radius * np.cos(polar),
                ]
            )
    faces: list[list[int]] = []
    for azimuth_index in range(n_azimuth):
        current = 1 + azimuth_index
        following = 1 + (azimuth_index + 1) % n_azimuth
        faces.append([0, current, following])
    for polar_index in range(n_polar - 1):
        ring_start = 1 + polar_index * n_azimuth
        next_ring_start = ring_start + n_azimuth
        for azimuth_index in range(n_azimuth):
            following = (azimuth_index + 1) % n_azimuth
            a = ring_start + azimuth_index
            b = ring_start + following
            c = next_ring_start + azimuth_index
            d = next_ring_start + following
            faces.extend(([a, c, d], [a, d, b]))
    return SurfaceInstance.from_mesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        samples_per_face=samples_per_face,
        surface_id="hemisphere",
        metadata={"radius": radius},
    )


def make_saddle(
    *,
    width: float = 2.0,
    height: float = 2.0,
    curvature: float = 0.25,
    nx: int = 24,
    ny: int = 24,
    samples_per_face: int = 4,
) -> SurfaceInstance:
    return _make_height_field(
        lambda x, y: curvature * (x * x - y * y),
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        samples_per_face=samples_per_face,
        surface_id="saddle",
        metadata={"curvature": curvature},
    )


def make_freeform_patch(
    *,
    width: float = 2.0,
    height: float = 2.0,
    amplitude: float = 0.2,
    nx: int = 24,
    ny: int = 24,
    samples_per_face: int = 4,
) -> SurfaceInstance:
    return _make_height_field(
        lambda x, y: amplitude
        * (np.sin(np.pi * x / width) * np.cos(2.0 * np.pi * y / height) + 0.35 * x * y),
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        samples_per_face=samples_per_face,
        surface_id="freeform_patch",
        metadata={"amplitude": amplitude},
    )


def make_torus(
    *,
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    n_major: int = 48,
    n_minor: int = 20,
    samples_per_face: int = 4,
) -> SurfaceInstance:
    _require_positive(major_radius=major_radius, minor_radius=minor_radius)
    if minor_radius >= major_radius:
        raise ValueError("minor_radius must be smaller than major_radius")
    if n_major < 3 or n_minor < 3:
        raise ValueError("torus requires n_major >= 3 and n_minor >= 3")
    vertices = []
    for minor_index in range(n_minor):
        phi = 2.0 * np.pi * minor_index / n_minor
        radial = major_radius + minor_radius * np.cos(phi)
        for major_index in range(n_major):
            theta = 2.0 * np.pi * major_index / n_major
            vertices.append([radial * np.cos(theta), radial * np.sin(theta), minor_radius * np.sin(phi)])
    faces = _grid_faces(n_major, n_minor, wrap_x=True, wrap_y=True)
    return SurfaceInstance.from_mesh(
        np.asarray(vertices, dtype=np.float64),
        faces,
        samples_per_face=samples_per_face,
        surface_id="torus",
        metadata={"major_radius": major_radius, "minor_radius": minor_radius},
    )


def _make_height_field(
    height_function: Callable[[float, float], float],
    *,
    width: float,
    height: float,
    nx: int,
    ny: int,
    samples_per_face: int,
    surface_id: str,
    metadata: dict[str, float],
) -> SurfaceInstance:
    _require_positive(width=width, height=height)
    _require_resolution(nx=nx, ny=ny)
    x = np.linspace(-0.5 * width, 0.5 * width, nx + 1)
    y = np.linspace(-0.5 * height, 0.5 * height, ny + 1)
    vertices = np.array([[xi, yi, height_function(xi, yi)] for yi in y for xi in x], dtype=np.float64)
    faces = _grid_faces(nx, ny, wrap_x=False, wrap_y=False)
    return SurfaceInstance.from_mesh(
        vertices,
        faces,
        samples_per_face=samples_per_face,
        surface_id=surface_id,
        metadata={"width": width, "height": height, **metadata},
    )


def _grid_faces(nx: int, ny: int, *, wrap_x: bool, wrap_y: bool) -> np.ndarray:
    x_vertices = nx if wrap_x else nx + 1
    y_vertices = ny if wrap_y else ny + 1
    x_cells = nx
    y_cells = ny
    faces: list[list[int]] = []
    for y_index in range(y_cells):
        next_y = (y_index + 1) % y_vertices
        for x_index in range(x_cells):
            next_x = (x_index + 1) % x_vertices
            a = y_index * x_vertices + x_index
            b = y_index * x_vertices + next_x
            c = next_y * x_vertices + x_index
            d = next_y * x_vertices + next_x
            faces.extend(([a, b, d], [a, d, c]))
    return np.asarray(faces, dtype=np.int64)


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")


def _require_resolution(**values: int) -> None:
    for name, value in values.items():
        if value < 1:
            raise ValueError(f"{name} must be at least one")
