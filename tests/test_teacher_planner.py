from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from diffusion_coverage.coverage import (
    ClassicalTeacherPlanner,
    control_distance_radius,
    MultiStartStructuredTeacher,
    StructuredTeacherConfig,
    TeacherDatasetWriter,
    TeacherPlannerConfig,
    evaluate_coverage,
    extract_structured_parameter_controls,
    generate_pattern_proposals,
    load_teacher_instance,
    structured_controls_to_residual,
    structured_residual_to_controls,
    structured_template_controls,
)
from diffusion_coverage.surface import make_cylinder, make_hemisphere, make_plane, project_points


def test_pattern_proposals_respect_segment_budget_and_surface():
    surface = make_cylinder(n_azimuth=16, n_height=5, samples_per_face=1)
    proposals = generate_pattern_proposals(
        surface,
        footprint_radius=0.25,
        max_segments=2,
        overlap=0.65,
    )
    assert len(proposals) >= 3
    for proposal in proposals:
        assert proposal.plan.num_segments <= 2
        for path in proposal.plan.active_paths():
            projection = project_points(surface, path)
            assert projection.distances.max() < 0.025


def test_periodic_surface_proposals_include_hard_feasible_cross_axis_raster():
    for surface in (
        make_cylinder(n_azimuth=16, n_height=6, samples_per_face=1),
        make_hemisphere(n_azimuth=16, n_polar=6, samples_per_face=1),
    ):
        proposals = generate_pattern_proposals(
            surface,
            footprint_radius=0.25,
            overlap=0.7,
        )
        cross_axis = next(proposal for proposal in proposals if proposal.name == "raster_v_phase_0.00")
        metrics = evaluate_coverage(surface, cross_axis.plan, footprint_radius=0.25)
        assert metrics.missed_fraction <= 0.05


def test_teacher_returns_feasible_plan_on_plane():
    surface = make_plane(nx=8, ny=8, samples_per_face=2)
    config = TeacherPlannerConfig(
        footprint_radius=0.2,
        missed_tolerance=0.05,
        refinement_iterations=2,
        seed=4,
    )
    result = ClassicalTeacherPlanner(config).solve(surface)
    assert result.candidates
    assert result.feasible_candidates
    assert result.best.feasible
    assert result.best.metrics.missed_fraction <= config.missed_tolerance
    assert result.evaluated_plans >= len(result.initial_candidates)


def test_multistart_structured_teacher_preserves_hard_template_candidate():
    surface = make_plane(nx=5, ny=5, samples_per_face=1)
    proposal = generate_pattern_proposals(
        surface, footprint_radius=0.25, overlap=0.6
    )[0]
    config = StructuredTeacherConfig(
        footprint_radius=0.25,
        missed_tolerance=0.2,
        restarts=2,
        steps_per_restart=2,
        seed=11,
    )
    result = MultiStartStructuredTeacher(config).solve(surface, proposal)
    assert result.template.feasible
    assert result.candidates[0] is result.template
    assert result.evaluated_assignments == 1 + config.restarts * (
        1 + config.steps_per_restart
    )
    assert all(candidate.feasible for candidate in result.candidates)


def test_periodic_control_distance_ignores_integer_chart_shift():
    surface = make_cylinder(n_azimuth=8, n_height=4, samples_per_face=1)
    left = np.asarray([[0.0, 0.2], [1.0, 0.8]])
    right = left + np.asarray([1.0, 0.0])
    assert control_distance_radius(left, right, surface, 0.2) < 1e-12


def test_structured_template_residual_roundtrip_uses_footprint_units():
    surface = make_cylinder(n_azimuth=12, n_height=5, samples_per_face=1)
    radius = 0.2
    overlap = 0.65
    mode_name = "raster_u_phase_0.25"
    template = structured_template_controls(
        surface,
        footprint_radius=radius,
        overlap=overlap,
        mode_name=mode_name,
    )
    residual = structured_controls_to_residual(
        surface,
        template,
        footprint_radius=radius,
        overlap=overlap,
        mode_name=mode_name,
    )
    assert np.allclose(residual, 0.0)
    residual[:, 1] = np.linspace(-0.2, 0.2, len(residual))
    controls = structured_residual_to_controls(
        surface,
        residual,
        footprint_radius=radius,
        overlap=overlap,
        mode_name=mode_name,
    )
    recovered = structured_controls_to_residual(
        surface,
        controls,
        footprint_radius=radius,
        overlap=overlap,
        mode_name=mode_name,
    )
    assert np.allclose(recovered, residual, rtol=2e-5, atol=3e-7)


def test_hemisphere_cross_axis_template_has_deterministic_analytic_shape():
    surface = make_hemisphere(n_azimuth=24, n_polar=12, samples_per_face=2)
    radius = 0.09
    overlap = 0.7
    templates = [
        structured_template_controls(
            surface,
            footprint_radius=radius,
            overlap=overlap,
            mode_name="raster_v_phase_0.00",
        )
        for _ in range(20)
    ]
    assert all(template.shape == templates[0].shape for template in templates)
    assert all(np.array_equal(template, templates[0]) for template in templates)
    assert len(templates[0]) % 2 == 0
    assert np.allclose(np.sort(templates[0][:, 1].reshape(-1, 2), axis=1), [0.0, 1.0])
    proposal = next(
        proposal
        for proposal in generate_pattern_proposals(
            surface, footprint_radius=radius, overlap=overlap
        )
        if proposal.name == "raster_v_phase_0.00"
    )
    extracted = extract_structured_parameter_controls(
        surface,
        proposal.plan.active_paths()[0],
        mode_name=proposal.name,
    )
    assert extracted.shape == templates[0].shape


def test_teacher_refinement_never_makes_returned_best_objective_worse():
    surface = make_plane(nx=6, ny=6, samples_per_face=1)
    config = TeacherPlannerConfig(
        footprint_radius=0.22,
        missed_tolerance=0.08,
        refinement_iterations=3,
        seed=7,
    )
    result = ClassicalTeacherPlanner(config).solve(surface)

    def objective(candidate):
        violation = max(0.0, candidate.metrics.missed_fraction - config.missed_tolerance)
        quality = candidate.metrics.path_length + config.smoothness_weight * candidate.smoothness_cost
        return (int(violation > 0.0), violation, quality)

    assert objective(result.best) <= min(objective(candidate) for candidate in result.initial_candidates)


def test_teacher_dataset_round_trip_is_numeric_and_pickle_free():
    surface = make_plane(nx=5, ny=5, samples_per_face=1)
    config = TeacherPlannerConfig(
        footprint_radius=0.25,
        missed_tolerance=0.1,
        refinement_iterations=1,
        seed=2,
    )
    result = ClassicalTeacherPlanner(config).solve(surface)
    with TemporaryDirectory() as temporary_dir:
        writer = TeacherDatasetWriter(temporary_dir)
        instance_path = writer.write("plane_000", surface, config, result)
        loaded = load_teacher_instance(instance_path)
        assert loaded["candidate_waypoints"].ndim == 4
        assert loaded["candidate_waypoint_mask"].dtype == np.bool_
        assert loaded["metadata"]["instance_id"] == "plane_000"
        assert np.all(loaded["candidate_metrics"][:, 4] == 1.0)
        manifest = json.loads((Path(temporary_dir) / "manifest.jsonl").read_text().strip())
        assert manifest["num_candidates"] == len(result.feasible_candidates)
        with np.load(instance_path, allow_pickle=False) as archive:
            assert all(archive[key].dtype != object for key in archive.files)


def test_teacher_dataset_writer_can_resume_without_overwriting_instances():
    surface = make_plane(nx=4, ny=4, samples_per_face=1)
    config = TeacherPlannerConfig(footprint_radius=0.3, missed_tolerance=0.2, refinement_iterations=0)
    result = ClassicalTeacherPlanner(config).solve(surface)
    with TemporaryDirectory() as temporary_dir:
        writer = TeacherDatasetWriter(temporary_dir)
        writer.write("plane_000", surface, config, result)
        resumed = TeacherDatasetWriter(temporary_dir, resume=True)
        assert resumed.completed_instance_ids == {"plane_000"}
