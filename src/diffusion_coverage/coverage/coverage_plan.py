from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoveragePlan:
    """Fixed-capacity path representation with explicit segment and waypoint masks."""

    waypoints: np.ndarray
    segment_mask: np.ndarray | None = None
    waypoint_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        waypoints = np.asarray(self.waypoints, dtype=np.float64)
        if waypoints.ndim == 2:
            waypoints = waypoints[None, :, :]
        if waypoints.ndim != 3 or waypoints.shape[2] != 3:
            raise ValueError("waypoints must have shape [K, M, 3] or [M, 3]")
        if not np.all(np.isfinite(waypoints)):
            raise ValueError("waypoints must be finite")
        num_segments, max_waypoints, _ = waypoints.shape
        segment_mask = (
            np.ones(num_segments, dtype=bool)
            if self.segment_mask is None
            else np.asarray(self.segment_mask, dtype=bool)
        )
        waypoint_mask = (
            np.ones((num_segments, max_waypoints), dtype=bool)
            if self.waypoint_mask is None
            else np.asarray(self.waypoint_mask, dtype=bool)
        )
        if segment_mask.shape != (num_segments,):
            raise ValueError("segment_mask must have shape [K]")
        if waypoint_mask.shape != (num_segments, max_waypoints):
            raise ValueError("waypoint_mask must have shape [K, M]")
        for segment_index in range(num_segments):
            active = waypoint_mask[segment_index]
            if segment_mask[segment_index]:
                if active.sum() < 2:
                    raise ValueError("each active segment requires at least two active waypoints")
                active_indices = np.flatnonzero(active)
                if not np.array_equal(active_indices, np.arange(active_indices[-1] + 1)):
                    raise ValueError("active waypoints must form a contiguous prefix")
            elif np.any(active):
                raise ValueError("inactive segments cannot contain active waypoints")
        if not np.any(segment_mask):
            raise ValueError("a coverage plan requires at least one active segment")
        object.__setattr__(self, "waypoints", np.ascontiguousarray(waypoints))
        object.__setattr__(self, "segment_mask", np.ascontiguousarray(segment_mask))
        object.__setattr__(self, "waypoint_mask", np.ascontiguousarray(waypoint_mask))

    @property
    def num_segments(self) -> int:
        return int(self.segment_mask.sum())

    def active_paths(self) -> list[np.ndarray]:
        return [
            self.waypoints[index, self.waypoint_mask[index]]
            for index in np.flatnonzero(self.segment_mask)
        ]


@dataclass(frozen=True)
class CoverageMetrics:
    missed_fraction: float
    covered_area: float
    total_area: float
    path_length: float
    coverage_efficiency: float
    num_segments: int
    max_projection_distance: float
    mean_projection_distance: float
    metadata: dict[str, Any] = field(default_factory=dict)
