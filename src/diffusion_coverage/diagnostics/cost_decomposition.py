from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.diagnostics.structure import directed_transitions


@dataclass(frozen=True)
class TransitionCost:
    transition_index: int
    source_code: int
    target_code: int
    q_start_index: int
    q_end_index: int
    joint_length: float


def decompose_transition_costs(
    codes: np.ndarray,
    q_path: np.ndarray,
    transition_sample_counts: np.ndarray,
    *,
    expected_total: float | None = None,
    tolerance: float = 1e-9,
) -> tuple[TransitionCost, ...]:
    q = np.asarray(q_path, dtype=np.float64)
    counts = np.asarray(transition_sample_counts, dtype=np.int64)
    edges = directed_transitions(codes)
    if len(counts) != len(edges) or np.any(counts < 2):
        raise ValueError("one sample count >= 2 is required for every transition")
    if 1 + int(np.sum(counts - 1)) != len(q):
        raise ValueError("transition sample counts do not partition the q witness")
    result = []
    cursor = 0
    for index, ((source, target), count) in enumerate(zip(edges, counts)):
        end = cursor + int(count) - 1
        length = float(np.linalg.norm(np.diff(q[cursor : end + 1], axis=0), axis=1).sum())
        result.append(TransitionCost(index, source, target, cursor, end, length))
        cursor = end
    total = sum(item.joint_length for item in result)
    direct = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    reference = direct if expected_total is None else float(expected_total)
    if not np.isclose(total, direct, atol=tolerance, rtol=tolerance):
        raise ValueError("transition attribution does not reproduce direct witness length")
    if not np.isclose(total, reference, atol=tolerance, rtol=tolerance):
        raise ValueError("transition attribution does not reproduce archived witness length")
    return tuple(result)


def classify_transition_commonality(
    code_sequences: list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[dict[tuple[int, int], float], set[tuple[int, int]], set[tuple[int, int]]]:
    if not code_sequences:
        raise ValueError("at least one skeleton is required")
    sets = [set(directed_transitions(codes)) for codes in code_sequences]
    union = set.union(*sets)
    common = set.intersection(*sets)
    frequency = {edge: sum(edge in values for values in sets) / len(sets) for edge in sorted(union)}
    return frequency, common, union - common
