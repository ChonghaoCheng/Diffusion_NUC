from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.nuc.adapter import NUCSkeleton


@dataclass(frozen=True)
class SkeletonPairMetrics:
    directed_transition_jaccard: float
    undirected_transition_jaccard: float
    tree_edge_jaccard: float
    normalized_sequence_distance: float
    geometric_path_distance: float
    tangent_disagreement: float


def directed_transitions(codes: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(codes, dtype=np.int64)
    return tuple((int(a), int(b)) for a, b in zip(values[:-1], values[1:]))


def undirected_transitions(codes: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in directed_transitions(codes)}


def transition_parent_types(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = directed_transitions(codes)
    within = np.asarray([a // 3 == b // 3 for a, b in edges], dtype=np.bool_)
    return within, ~within


def jaccard(first: set | tuple, second: set | tuple) -> float:
    left, right = set(first), set(second)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def normalized_sequence_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.int64).tolist()
    right = np.asarray(second, dtype=np.int64).tolist()
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def resample_normalized_arclength(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("points must have shape [N, 3] with N >= 2")
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-15:
        return np.repeat(values[:1], count, axis=0)
    target = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([np.interp(target, cumulative, values[:, axis]) for axis in range(3)])


def compare_skeletons(
    first: NUCSkeleton,
    second: NUCSkeleton,
    *,
    first_path: np.ndarray | None = None,
    second_path: np.ndarray | None = None,
    resample_count: int = 512,
) -> SkeletonPairMetrics:
    path_a = first.waypoints if first_path is None else first_path
    path_b = second.waypoints if second_path is None else second_path
    aligned_a = resample_normalized_arclength(path_a, resample_count)
    aligned_b = resample_normalized_arclength(path_b, resample_count)
    tangent_a = np.diff(aligned_a, axis=0)
    tangent_b = np.diff(aligned_b, axis=0)
    norm_a = np.linalg.norm(tangent_a, axis=1)
    norm_b = np.linalg.norm(tangent_b, axis=1)
    valid = (norm_a > 1e-12) & (norm_b > 1e-12)
    if np.any(valid):
        cosine = np.sum(tangent_a[valid] * tangent_b[valid], axis=1) / (norm_a[valid] * norm_b[valid])
        tangent_disagreement = float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0))))
    else:
        tangent_disagreement = 0.0
    tree_a = {tuple(map(int, edge[:2])) for edge in first.tree_edges}
    tree_b = {tuple(map(int, edge[:2])) for edge in second.tree_edges}
    return SkeletonPairMetrics(
        directed_transition_jaccard=jaccard(directed_transitions(first.topological_path), directed_transitions(second.topological_path)),
        undirected_transition_jaccard=jaccard(undirected_transitions(first.topological_path), undirected_transitions(second.topological_path)),
        tree_edge_jaccard=jaccard(tree_a, tree_b),
        normalized_sequence_distance=normalized_sequence_distance(first.topological_path, second.topological_path),
        geometric_path_distance=float(np.mean(np.linalg.norm(aligned_a - aligned_b, axis=1))),
        tangent_disagreement=tangent_disagreement,
    )
