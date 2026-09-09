from __future__ import annotations

import numpy as np

from diffusion_coverage.coverage import (
    CoveragePlan,
    refine_coverage_by_insertion,
    refine_coverage_by_shortcutting,
)
from diffusion_coverage.surface import make_plane


def test_insertion_refinement_reduces_missed_coverage_without_leaving_surface():
    surface = make_plane(width=2.0, height=2.0, nx=8, ny=8, samples_per_face=2)
    path = np.column_stack((np.linspace(-1.0, 1.0, 12), np.zeros(12), np.zeros(12)))
    result = refine_coverage_by_insertion(
        surface,
        CoveragePlan(path),
        footprint_radius=0.25,
        missed_tolerance=0.1,
        max_steps=12,
    )
    assert result.steps > 0
    assert result.metrics.missed_fraction < result.initial_metrics.missed_fraction
    assert result.metrics.max_projection_distance < 1e-10


def test_shortcutting_reduces_redundant_travel_without_losing_hard_coverage():
    surface = make_plane(width=2.0, height=0.1, nx=24, ny=3, samples_per_face=2)
    x = np.linspace(-1.0, 1.0, 121)
    path = np.column_stack((x, 0.03 * np.sin(np.linspace(0.0, 24.0 * np.pi, len(x))), np.zeros_like(x)))
    result = refine_coverage_by_shortcutting(
        surface,
        CoveragePlan(path),
        footprint_radius=0.08,
        missed_tolerance=0.01,
        max_passes=6,
    )
    assert result.initial_metrics.missed_fraction <= 0.01
    assert result.metrics.missed_fraction <= 0.01
    assert result.metrics.path_length < result.initial_metrics.path_length
    assert len(result.plan.active_paths()[0]) < len(path)


def test_shortcutting_does_not_modify_an_infeasible_plan():
    surface = make_plane(width=2.0, height=2.0, nx=8, ny=8, samples_per_face=2)
    path = np.column_stack((np.linspace(-1.0, 1.0, 20), np.zeros(20), np.zeros(20)))
    result = refine_coverage_by_shortcutting(
        surface,
        CoveragePlan(path),
        footprint_radius=0.1,
        missed_tolerance=0.01,
    )
    assert result.initial_metrics.missed_fraction > 0.01
    assert result.steps == 0
    assert np.array_equal(result.plan.active_paths()[0], path)
