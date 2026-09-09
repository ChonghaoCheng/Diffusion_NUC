from __future__ import annotations

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        end = start + 1
        while end < len(data) and data[order[end]] == data[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    return pearson(rankdata(np.asarray(first)), rankdata(np.asarray(second)))


def relative_spread(values: np.ndarray) -> float | None:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) < 2 or np.min(data) <= 0.0:
        return None
    return float((np.max(data) - np.min(data)) / np.min(data))
