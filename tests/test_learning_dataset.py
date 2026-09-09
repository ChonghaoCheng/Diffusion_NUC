from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np
import pytest

from diffusion_coverage.coverage import ClassicalTeacherPlanner, TeacherDatasetWriter, TeacherPlannerConfig
from diffusion_coverage.coverage import CoveragePlan, load_teacher_instance, surface_from_teacher_archive
from diffusion_coverage.coverage.evaluator import evaluate_coverage
from diffusion_coverage.coverage.resampling import resample_plan_fixed_waypoints
from diffusion_coverage.learning import (
    TeacherPathDataset,
    candidate_indices_by_name,
    canonical_candidate_name,
    collate_teacher_paths,
    create_instance_split,
    filter_manifest_by_candidate_name,
    load_manifest,
)
from diffusion_coverage.surface import make_cylinder, make_plane, project_points


def test_fixed_waypoint_resampling_stays_on_surface_and_preserves_coverage():
    surface = make_cylinder(n_azimuth=12, n_height=5, samples_per_face=1)
    config = TeacherPlannerConfig(
        footprint_radius=0.25,
        missed_tolerance=0.1,
        refinement_iterations=0,
        overlap=0.6,
        seed=2,
    )
    result = ClassicalTeacherPlanner(config).solve(surface)
    original = result.feasible_candidates[0].plan
    fixed = resample_plan_fixed_waypoints(surface, original, num_waypoints=128)
    projection = project_points(surface, fixed.active_paths()[0])
    metrics = evaluate_coverage(surface, fixed, footprint_radius=config.footprint_radius)
    assert fixed.active_paths()[0].shape == (128, 3)
    assert projection.distances.max() < 1e-10
    assert metrics.missed_fraction <= config.missed_tolerance + 0.03


def test_teacher_path_dataset_shapes_and_instance_split_do_not_overlap():
    with TemporaryDirectory() as temporary_dir:
        writer = TeacherDatasetWriter(temporary_dir)
        for index in range(3):
            surface = make_plane(nx=4 + index, ny=4, samples_per_face=1)
            config = TeacherPlannerConfig(
                footprint_radius=0.3,
                missed_tolerance=0.15,
                refinement_iterations=0,
                seed=index,
            )
            result = ClassicalTeacherPlanner(config).solve(surface)
            writer.write(f"plane_{index}", surface, config, result)
        rows = load_manifest(temporary_dir)
        train_ids, validation_ids = create_instance_split(rows, validation_fraction=1 / 3, seed=8)
        assert set(train_ids).isdisjoint(validation_ids)
        assert set(train_ids) | set(validation_ids) == {row["instance_id"] for row in rows}

        dataset = TeacherPathDataset(
            temporary_dir,
            instance_ids=train_ids,
            num_surface_points=32,
            num_path_waypoints=24,
        )
        sample = dataset[0]
        assert sample["surface"].shape == (32, 6)
        assert sample["path"].shape == (24, 3)
        assert sample["condition"].shape == (2,)
        assert np.isfinite(sample["path"].numpy()).all()
        assert sample["center"].dtype.is_floating_point
        assert sample["center"].numpy().dtype == np.float64
        assert sample["scale"].numpy().dtype == np.float64


def test_variable_token_dataset_and_collate_preserve_each_path_length():
    with TemporaryDirectory() as temporary_dir:
        writer = TeacherDatasetWriter(temporary_dir)
        settings = ((1.0, 1.0, 0.25), (2.0, 1.0, 0.2))
        for index, (width, height, radius) in enumerate(settings):
            surface = make_plane(width=width, height=height, nx=8, ny=4, samples_per_face=1)
            config = TeacherPlannerConfig(
                footprint_radius=radius,
                missed_tolerance=0.2,
                refinement_iterations=0,
                seed=index,
            )
            writer.write(f"variable_{index}", surface, config, ClassicalTeacherPlanner(config).solve(surface))
        dataset = TeacherPathDataset(
            temporary_dir,
            num_surface_points=32,
            num_path_waypoints=None,
            tokens_per_footprint_area=1.0,
            minimum_path_tokens=8,
            maximum_path_tokens=128,
        )
        first_by_instance = {}
        for sample_index, reference in enumerate(dataset.sample_index):
            first_by_instance.setdefault(reference.instance_index, sample_index)
        samples = [dataset[index] for index in first_by_instance.values()]
        counts = [int(sample["path"].shape[0]) for sample in samples]
        assert counts == [16, 50]
        best_dataset = TeacherPathDataset(
            temporary_dir,
            num_surface_points=32,
            num_path_waypoints=None,
            minimum_path_tokens=8,
            maximum_path_tokens=128,
            candidate_policy="best",
        )
        assert len(best_dataset) == 2
        batch = collate_teacher_paths(samples)
        assert batch["path"].shape == (2, 50, 3)
        assert batch["path_mask"].sum(dim=1).tolist() == counts
        for index, count in enumerate(counts):
            assert batch["path_arclength"][index, 0] == 0.0
            assert batch["path_arclength"][index, count - 1] == 1.0
            assert not batch["path_mask"][index, count:].any()


def test_teacher_archive_restores_exact_coverage_quadrature():
    with TemporaryDirectory() as temporary_dir:
        surface = make_plane(nx=6, ny=5, samples_per_face=4)
        config = TeacherPlannerConfig(
            footprint_radius=0.25,
            missed_tolerance=0.15,
            refinement_iterations=0,
            seed=4,
        )
        result = ClassicalTeacherPlanner(config).solve(surface)
        output = TeacherDatasetWriter(temporary_dir).write("roundtrip", surface, config, result)
        archive = load_teacher_instance(output)
        restored = surface_from_teacher_archive(archive, surface_id="plane")
        plan = CoveragePlan(
            archive["candidate_waypoints"][0],
            archive["candidate_segment_mask"][0],
            archive["candidate_waypoint_mask"][0],
        )
        metrics = evaluate_coverage(restored, plan, footprint_radius=config.footprint_radius)
        assert restored.num_samples == surface.num_samples
        assert np.isclose(metrics.missed_fraction, archive["candidate_metrics"][0, 0])
        assert np.isclose(metrics.path_length, archive["candidate_metrics"][0, 1])


def test_candidate_name_filter_uses_exact_family_and_phase_before_split():
    with TemporaryDirectory() as temporary_dir:
        surface = make_plane(nx=6, ny=5, samples_per_face=2)
        config = TeacherPlannerConfig(
            footprint_radius=0.25,
            missed_tolerance=0.15,
            refinement_iterations=0,
            seed=7,
        )
        result = ClassicalTeacherPlanner(config).solve(surface)
        output = TeacherDatasetWriter(temporary_dir).write("selector", surface, config, result)
        archive = load_teacher_instance(output)
        raw_name = str(archive["proposal_names"][-1])
        name = canonical_candidate_name(raw_name)
        rows = load_manifest(temporary_dir)
        assert len(filter_manifest_by_candidate_name(temporary_dir, rows, name)) == 1
        raw_only_rows = filter_manifest_by_candidate_name(
            temporary_dir, rows, name, allow_repaired=False
        )
        assert len(raw_only_rows) in {0, 1}
        raw_indices = candidate_indices_by_name(
            archive, None, allow_repaired=False
        )
        assert all(
            not str(archive["proposal_names"][index]).endswith("_robustness_repair")
            for index in raw_indices
        )
        assert not filter_manifest_by_candidate_name(temporary_dir, rows, "missing_phase")

        dataset = TeacherPathDataset(
            temporary_dir,
            num_surface_points=32,
            num_path_waypoints=None,
            minimum_path_tokens=8,
            maximum_path_tokens=128,
            candidate_policy="best",
            candidate_name=name,
        )
        sample = dataset[0]
        selected_name = canonical_candidate_name(
            str(archive["proposal_names"][int(sample["candidate_index"])])
        )
        assert selected_name == name
        uv_dataset = TeacherPathDataset(
            temporary_dir,
            num_surface_points=32,
            num_path_waypoints=None,
            minimum_path_tokens=8,
            maximum_path_tokens=128,
            candidate_policy="best",
            candidate_name=name,
            path_coordinate_system="analytic_uv",
        )
        assert uv_dataset[0]["path"].shape[-1] == 2
        assert np.isfinite(uv_dataset[0]["path"].numpy()).all()
        uv_batch = collate_teacher_paths([uv_dataset[0]])
        assert uv_batch["path"].shape[-1] == 2
        control_dataset = TeacherPathDataset(
            temporary_dir,
            num_surface_points=32,
            num_path_waypoints=None,
            minimum_path_tokens=8,
            maximum_path_tokens=128,
            candidate_policy="best",
            candidate_name=name,
            path_coordinate_system="analytic_uv_control",
        )
        assert control_dataset[0]["path"].shape[-1] == 2
        assert len(control_dataset[0]["path"]) < len(uv_dataset[0]["path"])
        structured_dataset = TeacherPathDataset(
            temporary_dir,
            num_surface_points=32,
            num_path_waypoints=None,
            minimum_path_tokens=2,
            maximum_path_tokens=128,
            candidate_policy="all",
            path_coordinate_system="analytic_uv_structured",
            include_mode_conditioning=True,
        )
        structured_sample = structured_dataset[0]
        assert structured_sample["path"].shape[-1] == 2
        assert structured_sample["condition"].shape == (7,)
        assert structured_sample["condition"][2:].sum() == 1.0
        assert 0 <= structured_sample["mode_id"] < 5
        residual_dataset = TeacherPathDataset(
            temporary_dir,
            num_surface_points=32,
            num_path_waypoints=None,
            minimum_path_tokens=2,
            maximum_path_tokens=128,
            candidate_policy="all",
            path_coordinate_system="analytic_uv_structured_residual",
            include_mode_conditioning=True,
        )
        residual_sample = residual_dataset[0]
        assert residual_sample["path"].shape == structured_sample["path"].shape
        assert np.isfinite(residual_sample["path"].numpy()).all()
        with pytest.raises(ValueError, match="no samples match"):
            TeacherPathDataset(temporary_dir, candidate_name="missing_phase")
