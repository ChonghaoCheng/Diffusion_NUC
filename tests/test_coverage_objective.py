from __future__ import annotations

from diffusion_coverage.coverage import constrained_coverage_key


def test_feasible_coverage_is_ranked_by_length_not_extra_coverage():
    lower_miss_longer = constrained_coverage_key(0.0, 12.0, 0.05)
    higher_miss_shorter = constrained_coverage_key(0.04, 8.0, 0.05)
    assert higher_miss_shorter < lower_miss_longer


def test_infeasible_coverage_is_ranked_by_violation_before_length():
    smaller_violation = constrained_coverage_key(0.06, 100.0, 0.05)
    larger_violation = constrained_coverage_key(0.08, 1.0, 0.05)
    assert smaller_violation < larger_violation
