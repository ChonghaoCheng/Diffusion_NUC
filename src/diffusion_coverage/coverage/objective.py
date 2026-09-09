from __future__ import annotations


def constrained_coverage_key(
    missed_fraction: float,
    path_length: float,
    missed_tolerance: float,
    *,
    secondary_cost: float = 0.0,
) -> tuple[int, float, float]:
    """Rank by feasibility, constraint violation, then path quality."""

    if not 0.0 <= missed_tolerance < 1.0:
        raise ValueError("missed_tolerance must lie in [0, 1)")
    if path_length < 0.0:
        raise ValueError("path_length must be nonnegative")
    violation = max(0.0, float(missed_fraction) - missed_tolerance)
    return (int(violation > 0.0), violation, float(path_length) + float(secondary_cost))
