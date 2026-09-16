from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class GeometryArc:
    arc_id: int
    start_port: int
    end_port: int
    points: np.ndarray
    family: str
    family_index: int
    macro_index: int
    kind: str
    forward: bool


@dataclass(frozen=True)
class GeometryBank:
    ports: np.ndarray
    arcs: tuple[GeometryArc, ...]
    route_arc_ids: dict[str, tuple[int, ...]]
    graph_hash: str


def spherical_distance(first: np.ndarray, second: np.ndarray, radius: float) -> float:
    x = np.asarray(first, dtype=np.float64) / radius
    y = np.asarray(second, dtype=np.float64) / radius
    return float(radius * np.arctan2(np.linalg.norm(np.cross(x, y)), np.dot(x, y)))


def shortest_sphere_arc(first: np.ndarray, second: np.ndarray, radius: float, maximum_step: float) -> np.ndarray:
    x = np.asarray(first, dtype=np.float64) / radius
    y = np.asarray(second, dtype=np.float64) / radius
    angle = float(np.arctan2(np.linalg.norm(np.cross(x, y)), np.dot(x, y)))
    if angle >= np.pi - 1e-10:
        raise ValueError("antipodal sphere connector is ambiguous")
    count = max(1, int(np.ceil(radius * angle / maximum_step)))
    if angle <= 1e-14:
        return np.repeat((radius * x)[None, :], count + 1, axis=0)
    fractions = np.linspace(0.0, 1.0, count + 1)
    values = (
        np.sin((1.0 - fractions) * angle)[:, None] * x[None, :]
        + np.sin(fractions * angle)[:, None] * y[None, :]
    ) / np.sin(angle)
    return radius * values


def macro_boundaries(points: np.ndarray, target_length: float) -> tuple[int, ...]:
    values = np.asarray(points, dtype=np.float64)
    boundaries = [0]
    accumulated = 0.0
    for index, distance in enumerate(np.linalg.norm(np.diff(values, axis=0), axis=1), start=1):
        accumulated += float(distance)
        if accumulated >= target_length:
            boundaries.append(index)
            accumulated = 0.0
    if boundaries[-1] != len(values) - 1:
        boundaries.append(len(values) - 1)
    return tuple(boundaries)


def build_geometry_bank(
    paths: tuple[np.ndarray, ...],
    families: tuple[str, ...],
    *,
    radius: float,
    macro_length: float,
    connector_radius: float,
    nearest_count: int,
    port_tolerance: float,
) -> GeometryBank:
    if len(paths) != len(families):
        raise ValueError("paths and families must have equal length")
    ports: list[np.ndarray] = []
    arcs: list[GeometryArc] = []
    routes: dict[str, tuple[int, ...]] = {}
    adjacent: set[tuple[int, int]] = set()

    def port_id(point: np.ndarray) -> int:
        for index, existing in enumerate(ports):
            if np.linalg.norm(existing - point) <= port_tolerance:
                return index
        ports.append(np.asarray(point, dtype=np.float64).copy())
        return len(ports) - 1

    for family_index, (family, path) in enumerate(zip(families, paths)):
        values = np.asarray(path, dtype=np.float64)
        boundaries = macro_boundaries(values, macro_length)
        forward_ids = []
        for macro_index, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            start, end = port_id(values[lo]), port_id(values[hi])
            adjacent.add(tuple(sorted((start, end))))
            arc_id = len(arcs)
            arcs.append(GeometryArc(arc_id, start, end, values[lo : hi + 1].copy(), family, family_index, macro_index, "source", True))
            forward_ids.append(arc_id)
            reverse_id = len(arcs)
            arcs.append(GeometryArc(reverse_id, end, start, values[lo : hi + 1][::-1].copy(), family, family_index, macro_index, "source", False))
        routes[f"{family}/forward"] = tuple(forward_ids)
        routes[f"{family}/reverse"] = tuple(arcs[index].arc_id + 1 for index in reversed(forward_ids))

    port_array = np.asarray(ports, dtype=np.float64)
    for source in range(len(port_array)):
        candidates = []
        for target in range(len(port_array)):
            if source == target or tuple(sorted((source, target))) in adjacent:
                continue
            distance = spherical_distance(port_array[source], port_array[target], radius)
            if distance <= connector_radius + 1e-12:
                candidates.append((distance, target))
        for _, target in sorted(candidates)[:nearest_count]:
            points = shortest_sphere_arc(port_array[source], port_array[target], radius, min(0.006, connector_radius / 4.0))
            arc_id = len(arcs)
            arcs.append(GeometryArc(arc_id, source, target, points, "cross_port", -1, -1, "cross_port", True))

    digest = hashlib.sha256()
    digest.update(port_array.tobytes())
    for arc in arcs:
        digest.update(np.asarray([arc.start_port, arc.end_port, arc.family_index, arc.macro_index], dtype=np.int64).tobytes())
        digest.update(arc.points.tobytes())
        digest.update(f"{arc.kind}:{arc.family}:{arc.forward}".encode())
    return GeometryBank(port_array, tuple(arcs), routes, digest.hexdigest())
