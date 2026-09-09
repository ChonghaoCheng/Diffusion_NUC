from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from diffusion_coverage.coverage import CoveragePlan, inverse_surface_parameters
from diffusion_coverage.surface import SurfaceInstance


@dataclass(frozen=True)
class SyntheticColourFieldConfig:
    """Position-dependent synthetic IK-sheet availability in an analytic surface chart."""

    num_colours: int = 3
    frequency_u: int = 2
    frequency_v: int = 1
    overlap_fraction: float = 0.15
    warp_amplitude: float = 0.08
    phase: float = 0.0
    global_colours: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_colours < 2:
            raise ValueError("num_colours must be at least two")
        if self.frequency_u < 0 or self.frequency_v < 0:
            raise ValueError("colour frequencies must be nonnegative")
        if self.frequency_u == 0 and self.frequency_v == 0:
            raise ValueError("at least one colour frequency must be positive")
        if not 0.0 <= self.overlap_fraction < 0.5:
            raise ValueError("overlap_fraction must lie in [0, 0.5)")
        if self.warp_amplitude < 0.0:
            raise ValueError("warp_amplitude must be nonnegative")
        if any(colour < 0 or colour >= self.num_colours for colour in self.global_colours):
            raise ValueError("global_colours contains an out-of-range colour")
        if len(set(self.global_colours)) != len(self.global_colours):
            raise ValueError("global_colours must be unique")


@dataclass(frozen=True)
class ColourSegment:
    path_index: int
    start_sample: int
    end_sample: int
    colour: int


@dataclass(frozen=True)
class ColourLiftResult:
    feasible: bool
    min_segments: int | None
    segments: tuple[ColourSegment, ...]
    sampled_points: int
    valid_colour_fraction: float

    def within_budget(self, max_segments: int) -> bool:
        if max_segments < 1:
            raise ValueError("max_segments must be positive")
        return self.feasible and self.min_segments is not None and self.min_segments <= max_segments


class SyntheticColourField:
    """A controlled, spatially fragmented colour field over normalized surface UV."""

    def __init__(self, config: SyntheticColourFieldConfig) -> None:
        self.config = config

    def valid_colours(self, parameters: np.ndarray) -> np.ndarray:
        uv = np.asarray(parameters, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError("parameters must have shape [M, 2]")
        if not np.all(np.isfinite(uv)):
            raise ValueError("parameters must be finite")
        cfg = self.config
        u = uv[:, 0]
        v = uv[:, 1]
        warp = cfg.warp_amplitude * np.sin(2.0 * np.pi * (u + cfg.phase)) * np.sin(
            2.0 * np.pi * (v - cfg.phase)
        )
        latent = np.mod(
            cfg.frequency_u * u + cfg.frequency_v * v + warp + cfg.phase,
            1.0,
        )
        colour_coordinate = latent * cfg.num_colours
        base = np.floor(colour_coordinate).astype(np.int64) % cfg.num_colours
        fraction = colour_coordinate - np.floor(colour_coordinate)
        valid = np.zeros((len(uv), cfg.num_colours), dtype=bool)
        rows = np.arange(len(uv))
        valid[rows, base] = True
        if cfg.overlap_fraction > 0.0:
            lower = fraction < cfg.overlap_fraction
            upper = fraction > 1.0 - cfg.overlap_fraction
            valid[rows[lower], (base[lower] - 1) % cfg.num_colours] = True
            valid[rows[upper], (base[upper] + 1) % cfg.num_colours] = True
        if cfg.global_colours:
            valid[:, list(cfg.global_colours)] = True
        return valid


def minimum_colour_segments(valid_colour_mask: np.ndarray) -> tuple[int | None, tuple[tuple[int, int, int], ...]]:
    """Find the minimum fixed-colour interval partition of one sampled path."""

    mask = np.asarray(valid_colour_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] < 1 or mask.shape[1] < 1:
        raise ValueError("valid_colour_mask must have shape [M, C]")
    if np.any(mask.sum(axis=1) == 0):
        return None, ()

    num_samples, num_colours = mask.shape
    infinity = num_samples + 1
    costs = np.full((num_samples, num_colours), infinity, dtype=np.int64)
    previous = np.full((num_samples, num_colours), -1, dtype=np.int64)
    costs[0, mask[0]] = 1

    for sample in range(1, num_samples):
        valid_now = np.flatnonzero(mask[sample])
        best_previous = int(np.argmin(costs[sample - 1]))
        best_previous_cost = int(costs[sample - 1, best_previous])
        for colour in valid_now:
            continue_cost = int(costs[sample - 1, colour])
            switch_cost = best_previous_cost + 1
            if continue_cost <= switch_cost:
                costs[sample, colour] = continue_cost
                previous[sample, colour] = colour
            else:
                costs[sample, colour] = switch_cost
                previous[sample, colour] = best_previous

    final_colour = int(np.argmin(costs[-1]))
    minimum = int(costs[-1, final_colour])
    if minimum >= infinity:
        return None, ()
    colours = np.empty(num_samples, dtype=np.int64)
    colours[-1] = final_colour
    for sample in range(num_samples - 1, 0, -1):
        colours[sample - 1] = previous[sample, colours[sample]]

    intervals: list[tuple[int, int, int]] = []
    start = 0
    for sample in range(1, num_samples):
        if colours[sample] != colours[sample - 1]:
            intervals.append((start, sample - 1, int(colours[sample - 1])))
            start = sample
    intervals.append((start, num_samples - 1, int(colours[-1])))
    return minimum, tuple(intervals)


def evaluate_colour_lift(
    surface: SurfaceInstance,
    plan: CoveragePlan,
    field: SyntheticColourField,
    *,
    maximum_parameter_step: float = 0.01,
) -> ColourLiftResult:
    """Compute the exact discrete minimum segment count for a sampled workspace plan."""

    if maximum_parameter_step <= 0.0:
        raise ValueError("maximum_parameter_step must be positive")
    all_segments: list[ColourSegment] = []
    total_segments = 0
    total_samples = 0
    valid_samples = 0
    for path_index, path in enumerate(plan.active_paths()):
        parameters = inverse_surface_parameters(surface, path, unwrap_periodic=True)
        dense = _densify_parameters(parameters, maximum_step=maximum_parameter_step)
        mask = field.valid_colours(dense)
        total_samples += len(mask)
        valid_samples += int(np.count_nonzero(mask.any(axis=1)))
        minimum, intervals = minimum_colour_segments(mask)
        if minimum is None:
            return ColourLiftResult(
                feasible=False,
                min_segments=None,
                segments=(),
                sampled_points=total_samples,
                valid_colour_fraction=valid_samples / total_samples,
            )
        total_segments += minimum
        all_segments.extend(
            ColourSegment(path_index, start, end, colour)
            for start, end, colour in intervals
        )
    return ColourLiftResult(
        feasible=True,
        min_segments=total_segments,
        segments=tuple(all_segments),
        sampled_points=total_samples,
        valid_colour_fraction=valid_samples / total_samples,
    )


def _densify_parameters(parameters: np.ndarray, *, maximum_step: float) -> np.ndarray:
    uv = np.asarray(parameters, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2:
        raise ValueError("parameters must have shape [M, 2] with M >= 2")
    pieces = [uv[0]]
    for start, end in zip(uv[:-1], uv[1:]):
        distance = float(np.max(np.abs(end - start)))
        subdivisions = max(1, int(math.ceil(distance / maximum_step)))
        for index in range(1, subdivisions + 1):
            pieces.append(start + (index / subdivisions) * (end - start))
    return np.asarray(pieces, dtype=np.float64)
