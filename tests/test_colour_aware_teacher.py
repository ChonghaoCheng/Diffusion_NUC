from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage.patterns import raster_pattern
from diffusion_coverage.liftability import (
    ColourAwareStructuredTeacher,
    ColourAwareTeacherConfig,
    SyntheticColourField,
    SyntheticColourFieldConfig,
    colour_aware_objective,
)
from diffusion_coverage.surface import make_plane


def test_colour_aware_objective_preserves_hard_coverage_priority() -> None:
    coverage_feasible = colour_aware_objective(
        missed_fraction=0.04,
        min_segments=20,
        path_length=100.0,
        missed_tolerance=0.05,
        max_segments=4,
    )
    coverage_infeasible = colour_aware_objective(
        missed_fraction=0.051,
        min_segments=1,
        path_length=1.0,
        missed_tolerance=0.05,
        max_segments=4,
    )
    assert coverage_feasible < coverage_infeasible


def test_colour_aware_objective_is_lexicographic_in_budget_then_length() -> None:
    over_budget = colour_aware_objective(
        missed_fraction=0.0,
        min_segments=5,
        path_length=1.0,
        missed_tolerance=0.05,
        max_segments=4,
    )
    within_budget = colour_aware_objective(
        missed_fraction=0.0,
        min_segments=4,
        path_length=100.0,
        missed_tolerance=0.05,
        max_segments=4,
    )
    shorter = colour_aware_objective(
        missed_fraction=0.0,
        min_segments=3,
        path_length=90.0,
        missed_tolerance=0.05,
        max_segments=4,
    )
    assert shorter < within_budget < over_budget


def test_colour_aware_teacher_is_deterministic_and_never_worse_than_template() -> None:
    surface = make_plane(width=1.0, height=1.0, nx=10, ny=10, samples_per_face=1)
    proposal = raster_pattern(
        surface,
        footprint_radius=0.16,
        overlap=0.75,
        sweep_axis="u",
    )
    field = SyntheticColourField(
        SyntheticColourFieldConfig(
            num_colours=3,
            frequency_u=0,
            frequency_v=1,
            overlap_fraction=0.15,
            warp_amplitude=0.04,
            phase=0.2,
        )
    )
    config = ColourAwareTeacherConfig(
        footprint_radius=0.16,
        missed_tolerance=0.08,
        max_segments=4,
        restarts=2,
        steps_per_restart=3,
        seed=7,
    )
    teacher = ColourAwareStructuredTeacher(config)
    first = teacher.solve(surface, proposal, field)
    second = teacher.solve(surface, proposal, field)
    key = lambda candidate: colour_aware_objective(
        missed_fraction=candidate.coverage.missed_fraction,
        min_segments=candidate.lift.min_segments,
        path_length=candidate.coverage.path_length,
        missed_tolerance=config.missed_tolerance,
        max_segments=config.max_segments,
    )
    assert key(first.best) <= key(first.template)
    assert key(first.best) == key(second.best)
    assert np.array_equal(first.best.controls, second.best.controls)
    assert first.evaluated_assignments == 1 + config.restarts * (1 + config.steps_per_restart)
