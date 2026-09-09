from __future__ import annotations

from itertools import product

import numpy as np

from diffusion_coverage.coverage import CoveragePlan, map_surface_parameters
from diffusion_coverage.liftability import (
    SyntheticColourField,
    SyntheticColourFieldConfig,
    evaluate_colour_lift,
    minimum_colour_segments,
)
from diffusion_coverage.surface import make_plane


def test_minimum_colour_segments_uses_overlap_without_free_switching() -> None:
    mask = np.asarray(
        [
            [True, False],
            [True, True],
            [False, True],
        ]
    )
    count, intervals = minimum_colour_segments(mask)
    assert count == 2
    assert intervals[0][0] == 0
    assert intervals[-1][1] == len(mask) - 1
    for start, end, colour in intervals:
        assert np.all(mask[start : end + 1, colour])


def test_minimum_colour_segments_reports_missing_ik_as_infeasible() -> None:
    count, intervals = minimum_colour_segments(
        np.asarray([[True, False], [False, False], [False, True]])
    )
    assert count is None
    assert intervals == ()


def test_synthetic_colour_field_always_has_a_valid_colour() -> None:
    field = SyntheticColourField(
        SyntheticColourFieldConfig(
            num_colours=4,
            frequency_u=3,
            frequency_v=2,
            overlap_fraction=0.1,
            warp_amplitude=0.12,
        )
    )
    uv = np.random.default_rng(4).uniform(-2.0, 3.0, size=(500, 2))
    assert np.all(field.valid_colours(uv).any(axis=1))


def test_global_colour_makes_any_path_single_segment_liftable() -> None:
    field = SyntheticColourField(
        SyntheticColourFieldConfig(
            num_colours=3,
            frequency_u=4,
            frequency_v=3,
            global_colours=(2,),
        )
    )
    uv = np.random.default_rng(9).uniform(-1.0, 2.0, size=(400, 2))
    count, intervals = minimum_colour_segments(field.valid_colours(uv))
    assert count == 1
    assert intervals == ((0, len(uv) - 1, 2),)


def test_colour_lift_budget_is_monotone_and_nontrivial_on_a_plane() -> None:
    surface = make_plane(nx=8, ny=8)
    u = np.linspace(0.0, 1.0, 30)
    v = np.full_like(u, 0.5)
    plan = CoveragePlan(map_surface_parameters(surface, u, v))
    field = SyntheticColourField(
        SyntheticColourFieldConfig(
            num_colours=3,
            frequency_u=2,
            frequency_v=0,
            overlap_fraction=0.1,
            warp_amplitude=0.0,
        )
    )
    result = evaluate_colour_lift(surface, plan, field)
    assert result.feasible
    assert result.min_segments is not None and result.min_segments > 1
    indicators = [result.within_budget(k) for k in range(1, result.min_segments + 2)]
    assert indicators == sorted(indicators)


def test_colour_permutation_does_not_change_minimum_segment_count() -> None:
    field = SyntheticColourField(SyntheticColourFieldConfig(num_colours=4))
    uv = np.column_stack((np.linspace(0.0, 1.0, 200), np.linspace(0.2, 0.8, 200)))
    mask = field.valid_colours(uv)
    original, _ = minimum_colour_segments(mask)
    permuted, _ = minimum_colour_segments(mask[:, [2, 0, 3, 1]])
    assert permuted == original


def test_minimum_colour_segments_matches_full_assignment_enumeration() -> None:
    rng = np.random.default_rng(13)
    for _ in range(100):
        mask = rng.random((6, 3)) < 0.55
        for sample in range(len(mask)):
            if not mask[sample].any():
                mask[sample, int(rng.integers(mask.shape[1]))] = True
        exact, _ = minimum_colour_segments(mask)
        brute_force = min(
            1 + sum(left != right for left, right in zip(colours[:-1], colours[1:]))
            for colours in product(range(mask.shape[1]), repeat=len(mask))
            if all(mask[sample, colour] for sample, colour in enumerate(colours))
        )
        assert exact == brute_force
