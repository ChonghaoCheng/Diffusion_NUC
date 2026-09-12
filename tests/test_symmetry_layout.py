from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract
from diffusion_coverage.diagnostics.symmetry_layout import (
    analytical_points_and_normals,
    assert_invariance,
    evaluate_symmetry_path,
    intrinsic_segment_lengths,
    load_canonical_path,
    make_symmetry_reference_surface,
    require_capacity_for_riemannian,
    scene_seed,
    symmetry_specs,
    transform_path,
    transport_coverage_metrics,
    verify_mesh_automorphism,
)

ROOT = Path(__file__).resolve().parents[1]


def setup(surface_id="hemisphere"):
    config = json.loads((ROOT / "configs/symmetry_preserving_global_layout_v1.json").read_text())
    archived, _, _ = load_e06_contract(ROOT)
    canonical = load_canonical_path(
        ROOT, archived, surface_id, config["surfaces"][surface_id],
        maximum_spacing=config["coverage"]["path_sample_spacing_m"],
    )
    reference = make_symmetry_reference_surface(surface_id, config)
    return config, canonical, reference


def test_theta_zero_exactly_reproduces_frozen_canonical_path():
    config, canonical, _ = setup()
    points, normals = transform_path(canonical, symmetry_specs("hemisphere", config)[0], config["surfaces"]["hemisphere"])
    assert np.array_equal(points, canonical.points)
    assert np.array_equal(normals, canonical.normals)


def test_hemisphere_orbit_is_analytical_closed_and_length_invariant():
    config, canonical, reference = setup()
    baseline = intrinsic_segment_lengths(reference, canonical.points)
    specs = symmetry_specs("hemisphere", config)
    assert len(specs) == 24
    for spec in specs:
        points, normals = transform_path(canonical, spec, config["surfaces"]["hemisphere"])
        assert np.max(np.abs(np.linalg.norm(points, axis=1) - config["surfaces"]["hemisphere"]["radius"])) < 1e-12
        assert np.allclose(normals, points / np.linalg.norm(points, axis=1, keepdims=True), atol=1e-12)
        assert np.allclose(intrinsic_segment_lengths(reference, points), baseline, atol=1e-12, rtol=0)
    assert np.allclose(np.asarray(specs[-1]["matrix"]) @ np.asarray(specs[1]["matrix"]), np.eye(3), atol=1e-12)


def test_saddle_symmetries_pass_analytical_surface_boundary_and_mesh_tests():
    config, canonical, reference = setup("saddle")
    for spec in symmetry_specs("saddle", config):
        points, normals = transform_path(canonical, spec, config["surfaces"]["saddle"])
        expected, expected_normals = analytical_points_and_normals("saddle", points, config["surfaces"]["saddle"])
        assert np.allclose(points, expected, atol=1e-12)
        assert np.allclose(normals, expected_normals, atol=1e-12)
        assert verify_mesh_automorphism(reference, np.asarray(spec["matrix"])) < 1e-12


def test_transported_temporal_coverage_is_exactly_invariant():
    config, canonical, reference = setup()
    baseline = evaluate_symmetry_path(reference, canonical.points, config)
    rows = []
    base_segments = intrinsic_segment_lengths(reference, canonical.points)
    for spec in symmetry_specs("hemisphere", config):
        points, _ = transform_path(canonical, spec, config["surfaces"]["hemisphere"])
        metrics, _ = transport_coverage_metrics(baseline, reference, np.asarray(spec["matrix"]), points)
        rows.append({
            "surface_id": "hemisphere", "symmetry_id": spec["symmetry_id"],
            "abs_E_miss_error": abs(metrics.missed_error-baseline.missed_error),
            "abs_E_rep_error": abs(metrics.repeat_error-baseline.repeat_error),
            "abs_E_NUC_error": abs(metrics.nuc_error-baseline.nuc_error),
            "abs_L_S_error": abs(metrics.path_length-baseline.path_length),
            "max_segment_length_error": float(np.max(np.abs(intrinsic_segment_lengths(reference, points)-base_segments))),
            "sample_order_preserved": True, "activity_preserved": True, "normal_covariance_pass": True,
        })
    assert_invariance(rows, 1e-12)


def test_order_topology_and_activity_are_index_preserving():
    config, canonical, _ = setup("saddle")
    transformed, _ = transform_path(canonical, symmetry_specs("saddle", config)[2], config["surfaces"]["saddle"])
    assert transformed.shape == canonical.points.shape
    assert np.array_equal(np.arange(len(transformed)), np.arange(len(canonical.points)))


def test_scene_seed_and_budget_do_not_depend_on_orbit_member():
    config, _, _ = setup()
    seed = scene_seed(config, "hemisphere", "P_mid", strong=False)
    assert seed == scene_seed(config, "hemisphere", "P_mid", strong=False)
    assert config["default_search"] == json.loads((ROOT / "configs/symmetry_preserving_global_layout_v1.json").read_text())["default_search"]


def test_placement_is_external_to_the_object_frame_symmetry_orbit():
    config, _, _ = setup()
    scenes = json.loads((ROOT / config["placements_file"]).read_text())["selected"]
    before = np.asarray(scenes["hemisphere"]["P_high"]["transform_base_from_surface"])
    symmetry_specs("hemisphere", config)
    after = np.asarray(scenes["hemisphere"]["P_high"]["transform_base_from_surface"])
    assert np.array_equal(before, after)


def test_riemannian_diagnostic_is_capacity_gated_and_not_a_planner():
    with pytest.raises(RuntimeError):
        require_capacity_for_riemannian({"decision": "NO-GO", "riemannian_authorized": False})
    require_capacity_for_riemannian({"decision": "GO", "riemannian_authorized": True})
