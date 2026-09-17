from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from diffusion_coverage.planning.e09_geometry import shortest_sphere_arc


@dataclass(frozen=True)
class ProgramToken:
    kind: str
    family: str | None = None
    direction: int | None = None
    a: float | None = None
    b: float | None = None
    d1: float | None = None
    d2: float | None = None


@dataclass(frozen=True)
class GeometryProgram:
    tokens: tuple[ProgramToken, ...]

    def to_json(self) -> str:
        return json.dumps([asdict(x) for x in self.tokens], sort_keys=True, separators=(",", ":"))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "GeometryProgram":
        return cls(tuple(ProgramToken(**x) for x in json.loads(text)))


@dataclass(frozen=True)
class GeometryLibrary:
    radius: float
    ports: np.ndarray
    arc_points: np.ndarray
    arc_offsets: np.ndarray
    arc_start: np.ndarray
    arc_end: np.ndarray
    arc_family_index: np.ndarray
    arc_macro_index: np.ndarray
    arc_forward: np.ndarray
    arc_kind: np.ndarray
    families: tuple[str, ...]
    routes: dict[str, tuple[int, ...]]
    family_points: dict[str, np.ndarray]
    family_knots: dict[str, np.ndarray]
    arc_intervals: dict[int, tuple[str, int, float, float]]
    semantic_hash: str

    def arc(self, arc_id: int) -> np.ndarray:
        lo, hi = self.arc_offsets[arc_id : arc_id + 2]
        return self.arc_points[int(lo) : int(hi)]


@dataclass(frozen=True)
class DecodedProgram:
    surface_points: np.ndarray
    parameter: np.ndarray
    segment_ids: np.ndarray
    corrections: tuple[dict[str, Any], ...]


def _sphere_lengths(points: np.ndarray, radius: float) -> np.ndarray:
    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    return radius * np.arctan2(
        np.linalg.norm(np.cross(unit[:-1], unit[1:]), axis=1),
        np.sum(unit[:-1] * unit[1:], axis=1),
    )


def load_geometry_library(npz_path: str | Path, json_path: str | Path, *, radius: float) -> GeometryLibrary:
    arrays = np.load(npz_path, allow_pickle=False)
    meta = json.loads(Path(json_path).read_text())
    families = tuple(meta["families"])
    routes = {k: tuple(int(x) for x in v) for k, v in meta["routes"].items()}
    family_points: dict[str, np.ndarray] = {}
    family_knots: dict[str, np.ndarray] = {}
    intervals: dict[int, tuple[str, int, float, float]] = {}
    for family in families:
        ids = routes[f"{family}/forward"]
        pieces = []
        bounds = [0.0]
        for order, arc_id in enumerate(ids):
            lo, hi = arrays["arc_offsets"][arc_id : arc_id + 2]
            p = np.asarray(arrays["arc_points"][int(lo) : int(hi)], dtype=np.float64)
            if order:
                p = p[1:]
            pieces.append(p)
        points = np.concatenate(pieces, axis=0)
        lengths = _sphere_lengths(points, radius)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        knots = cumulative / cumulative[-1]
        family_points[family] = points
        family_knots[family] = knots
        cursor = 0
        for arc_id in ids:
            lo, hi = arrays["arc_offsets"][arc_id : arc_id + 2]
            count = int(hi - lo)
            a, b = float(knots[cursor]), float(knots[cursor + count - 1])
            intervals[int(arc_id)] = (family, 1, a, b)
            reverse_id = int(arc_id) + 1
            intervals[reverse_id] = (family, -1, a, b)
            cursor += count - 1
    return GeometryLibrary(
        float(radius), np.asarray(arrays["ports"], dtype=np.float64),
        np.asarray(arrays["arc_points"], dtype=np.float64), np.asarray(arrays["arc_offsets"], dtype=np.int64),
        np.asarray(arrays["arc_start"], dtype=np.int64), np.asarray(arrays["arc_end"], dtype=np.int64),
        np.asarray(arrays["arc_family_index"], dtype=np.int64), np.asarray(arrays["arc_macro_index"], dtype=np.int64),
        np.asarray(arrays["arc_forward"], dtype=bool), np.asarray(arrays["arc_kind"]), families, routes,
        family_points, family_knots, intervals, str(meta["graph_hash"]),
    )


def stereographic_to_sphere(d1: float, d2: float, radius: float) -> tuple[np.ndarray, dict[str, Any] | None]:
    d = np.asarray([d1, d2], dtype=np.float64)
    norm = float(np.linalg.norm(d))
    correction = None
    if norm > 1.0:
        original = d.copy()
        d /= norm
        correction = {"kind": "via_radial_projection", "magnitude": float(np.linalg.norm(original - d))}
    r2 = float(d @ d)
    n = np.asarray([2.0 * d[0], 2.0 * d[1], 1.0 - r2]) / (1.0 + r2)
    return radius * n, correction


def sphere_to_stereographic(point: np.ndarray, radius: float) -> tuple[float, float]:
    n = np.asarray(point, dtype=np.float64) / radius
    denom = 1.0 + float(n[2])
    if denom <= 1e-12:
        raise ValueError("south-pole stereographic coordinate is undefined")
    d = n[:2] / denom
    if np.linalg.norm(d) > 1.0 + 1e-10:
        raise ValueError("point is outside upper hemisphere")
    return float(d[0]), float(d[1])


def _sample_family(lib: GeometryLibrary, family: str, start: float, end: float, maximum_step: float) -> np.ndarray:
    points = lib.family_points[family]
    knots = lib.family_knots[family]
    reverse = end < start
    lo, hi = (end, start) if reverse else (start, end)
    if lo < -1e-12 or hi > 1.0 + 1e-12 or hi - lo <= 1e-14:
        raise ValueError("invalid or zero-span SCAN interval")
    total = float((hi - lo) * _sphere_lengths(points, lib.radius).sum())
    count = max(1, int(np.ceil(total / maximum_step)))
    query = np.linspace(start, end, count + 1)
    out = np.empty((len(query), 3), dtype=np.float64)
    for j, value in enumerate(query):
        idx = max(0, min(int(np.searchsorted(knots, value, side="right")) - 1, len(points) - 2))
        span = float(knots[idx + 1] - knots[idx])
        f = 0.0 if span <= 1e-15 else float((value - knots[idx]) / span)
        x = points[idx] / lib.radius; y = points[idx + 1] / lib.radius
        angle = float(np.arctan2(np.linalg.norm(np.cross(x, y)), np.dot(x, y)))
        if angle <= 1e-14:
            v = (1.0 - f) * x + f * y
            v /= np.linalg.norm(v)
        else:
            v = (np.sin((1.0 - f) * angle) * x + np.sin(f * angle) * y) / np.sin(angle)
        out[j] = lib.radius * v
    return out


def validate_program(program: GeometryProgram, families: Iterable[str], maximum_tokens: int = 64) -> None:
    allowed = set(families)
    if not program.tokens or len(program.tokens) > maximum_tokens or program.tokens[-1].kind != "END":
        raise ValueError("program must end within the registered token cap")
    if sum(x.kind == "SCAN" for x in program.tokens) == 0:
        raise ValueError("program must contain a SCAN")
    for i, token in enumerate(program.tokens):
        if token.kind == "SCAN":
            if token.family not in allowed or token.direction not in (-1, 1):
                raise ValueError("invalid SCAN symbol")
            if token.a is None or token.b is None or not (0.0 <= token.a < token.b <= 1.0):
                raise ValueError("invalid SCAN interval")
        elif token.kind == "VIA":
            if token.d1 is None or token.d2 is None:
                raise ValueError("invalid VIA")
        elif token.kind == "END":
            if i != len(program.tokens) - 1:
                raise ValueError("END must be terminal")
        else:
            raise ValueError("unknown program token")


def encode_edge_sequence(sequence: list[dict[str, Any]], lib: GeometryLibrary, maximum_tokens: int = 64) -> GeometryProgram:
    tokens: list[ProgramToken] = []
    pending: tuple[str, int, float, float, int] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            family, direction, a, b, _ = pending
            tokens.append(ProgramToken("SCAN", family, direction, min(a, b), max(a, b)))
            pending = None

    for edge in sequence:
        kind = str(edge.get("kind", ""))
        if kind == "source":
            arc_id = int(edge["geom_arc_id"])
            if arc_id not in lib.arc_intervals:
                raise ValueError(f"unknown source arc {arc_id}")
            family, direction, a, b = lib.arc_intervals[arc_id]
            macro = int(lib.arc_macro_index[arc_id])
            if pending is None:
                pending = (family, direction, a, b, macro)
            else:
                pf, pd, pa, pb, pm = pending
                consecutive = pf == family and pd == direction and ((direction == 1 and macro == pm + 1) or (direction == -1 and macro == pm - 1))
                if consecutive:
                    pending = (pf, pd, min(pa, a), max(pb, b), macro)
                else:
                    flush(); pending = (family, direction, a, b, macro)
        elif kind == "cross_port":
            flush()
            port = int(edge["end_port"])
            d1, d2 = sphere_to_stereographic(lib.ports[port], lib.radius)
            tokens.append(ProgramToken("VIA", d1=d1, d2=d2))
        elif kind.startswith("entry"):
            flush()
        elif kind.startswith("off"):
            raise ValueError("OFF witness is outside the E12 primary scope")
        else:
            raise ValueError(f"unsupported edge kind {kind}")
    flush(); tokens.append(ProgramToken("END"))
    program = GeometryProgram(tuple(tokens))
    validate_program(program, lib.families, maximum_tokens)
    return program


def decode_program(program: GeometryProgram, lib: GeometryLibrary, start_point: np.ndarray, *, maximum_step: float, maximum_tokens: int = 64) -> DecodedProgram:
    validate_program(program, lib.families, maximum_tokens)
    current = np.asarray(start_point, dtype=np.float64)
    points = [current.copy()]; segment_ids = [0]; corrections: list[dict[str, Any]] = []
    segment = 0
    for token_index, token in enumerate(program.tokens):
        if token.kind == "END":
            break
        segment += 1
        if token.kind == "VIA":
            target, correction = stereographic_to_sphere(float(token.d1), float(token.d2), lib.radius)
            if correction is not None:
                corrections.append({"token_index": token_index, **correction})
            arc = shortest_sphere_arc(current, target, lib.radius, maximum_step)
            if np.linalg.norm(current - target) <= 1e-14:
                arc = arc[:1]
            for p in arc[1:]: points.append(p); segment_ids.append(segment)
            current = target
            continue
        start = float(token.a if token.direction == 1 else token.b)
        end = float(token.b if token.direction == 1 else token.a)
        scan = _sample_family(lib, str(token.family), start, end, maximum_step)
        connector = shortest_sphere_arc(current, scan[0], lib.radius, maximum_step)
        for p in connector[1:]: points.append(p); segment_ids.append(segment)
        for p in scan[1:]: points.append(p); segment_ids.append(segment)
        current = scan[-1]
    values = np.asarray(points, dtype=np.float64)
    lengths = _sphere_lengths(values, lib.radius)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    parameter = cumulative / max(float(cumulative[-1]), 1e-15)
    return DecodedProgram(values, parameter, np.asarray(segment_ids, dtype=np.int32), tuple(corrections))
