from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeEdgeSummary:
    """Exact summary of one fixed, ordered ON/OFF membership realization."""

    footprint: np.ndarray
    episode_counts: np.ndarray
    start_membership: np.ndarray
    end_membership: np.ndarray
    weighted_episode_mass: float
    off_to_on_count: int


@dataclass(frozen=True)
class EpisodeState:
    covered: np.ndarray
    membership: np.ndarray
    repeat_error: float


def summarize_ordered_membership(
    membership: np.ndarray,
    weights: np.ndarray,
    *,
    active: np.ndarray | None = None,
) -> EpisodeEdgeSummary:
    values = np.asarray(membership, dtype=bool)
    area = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(area) or values.shape[1] < 1:
        raise ValueError("membership must have shape [units, time] with time >= 1")
    if np.any(area <= 0.0) or not np.all(np.isfinite(area)):
        raise ValueError("weights must be positive and finite")
    activity = np.ones(values.shape[1], dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if activity.shape != (values.shape[1],):
        raise ValueError("active must have shape [time]")
    effective = values & activity[None, :]
    starts = effective.copy()
    starts[:, 1:] &= ~effective[:, :-1]
    counts = starts.sum(axis=1, dtype=np.int64)
    transitions = int(np.count_nonzero(activity[1:] & ~activity[:-1]))
    return EpisodeEdgeSummary(
        footprint=np.any(effective, axis=1),
        episode_counts=counts,
        start_membership=effective[:, 0].copy(),
        end_membership=effective[:, -1].copy(),
        weighted_episode_mass=float(np.dot(area, counts)),
        off_to_on_count=transitions,
    )


def initial_episode_state(initial_membership: np.ndarray) -> EpisodeState:
    value = np.asarray(initial_membership, dtype=bool)
    if value.ndim != 1:
        raise ValueError("initial_membership must be one dimensional")
    return EpisodeState(value.copy(), value.copy(), 0.0)


def apply_edge_summary(
    state: EpisodeState,
    edge: EpisodeEdgeSummary,
    weights: np.ndarray,
    *,
    atol: float = 1e-12,
) -> EpisodeState:
    area = np.asarray(weights, dtype=np.float64)
    if state.covered.shape != area.shape or state.membership.shape != area.shape:
        raise ValueError("state and weights have incompatible shapes")
    if edge.footprint.shape != area.shape:
        raise ValueError("edge and weights have incompatible shapes")
    if not np.array_equal(state.membership, edge.start_membership):
        raise ValueError("edge start membership does not match the current node state")
    total = float(area.sum())
    continued = float(area[state.membership & edge.start_membership].sum())
    newly_covered = float(area[edge.footprint & ~state.covered].sum())
    increment = (edge.weighted_episode_mass - continued - newly_covered) / total
    if increment < -atol:
        raise FloatingPointError("episode summary produced a negative repeat increment")
    increment = max(0.0, increment)
    return EpisodeState(
        covered=state.covered | edge.footprint,
        membership=edge.end_membership.copy(),
        repeat_error=float(state.repeat_error + increment),
    )


def concatenate_membership_edges(
    memberships: tuple[np.ndarray, ...],
    activities: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not memberships:
        raise ValueError("at least one edge is required")
    acts = activities or tuple(np.ones(np.asarray(value).shape[1], dtype=bool) for value in memberships)
    if len(acts) != len(memberships):
        raise ValueError("activities and memberships must have equal length")
    pieces: list[np.ndarray] = []
    active_pieces: list[np.ndarray] = []
    for index, (membership, active) in enumerate(zip(memberships, acts)):
        value = np.asarray(membership, dtype=bool)
        mask = np.asarray(active, dtype=bool)
        if index and not np.array_equal(value[:, 0] & mask[0], pieces[-1][:, -1] & active_pieces[-1][-1]):
            raise ValueError("membership edges do not share an exact endpoint")
        pieces.append(value if index == 0 else value[:, 1:])
        active_pieces.append(mask if index == 0 else mask[1:])
    return np.concatenate(pieces, axis=1), np.concatenate(active_pieces)
