from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusion_coverage.diagnostics.local_deformation import (
    AdmissionLimits,
    accept_incumbent_update,
    control_indices,
    cumulative_length,
    deform_surface_path,
    euclidean_objective,
    hard_admission,
    proposal_schedule,
    terminal_q_mismatch,
)
from diffusion_coverage.geometry.surface_curve import trace_surface_curve
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.surface import make_saddle
from diffusion_coverage.surface.projection import project_points


ROOT = Path(__file__).resolve().parents[1]


def toy_path():
    surface = make_saddle(width=0.24, height=0.24, curvature=1.5, nx=20, ny=20, samples_per_face=1)
    curve = trace_surface_curve(surface, np.zeros(3), np.array([1.0, 0.15, 0.0]), 0.048, maximum_step=0.002)
    points = curve.points
    controls = control_indices(cumulative_length(points), 0, len(points) - 1, 7)
    return surface, points, controls


def test_fixed_controls_topology_and_mesh_membership():
    surface, points, controls = toy_path()
    parameters = np.array([[0.0005, 0.0], [0.0, -0.0005], [0.0004, 0.0002]])
    candidate = deform_surface_path(surface, points, controls, parameters, maximum_displacement=0.002, maximum_retraction_step=0.00025)
    assert np.array_equal(candidate.points[controls[[0, 1, 5, 6]]], points[controls[[0, 1, 5, 6]]])
    projection = project_points(surface, candidate.points)
    assert projection.distances.max() < 1e-10
    assert candidate.topology_preserved


def test_zero_deformation_preserves_order_and_endpoints():
    surface, points, controls = toy_path()
    candidate = deform_surface_path(surface, points, controls, np.zeros((3, 2)), maximum_displacement=0.002, maximum_retraction_step=0.00025)
    assert np.allclose(candidate.points, points, atol=1e-10)
    assert np.array_equal(candidate.points[[0, -1]], points[[0, -1]])


def test_M1_objective_cannot_access_robot_metric():
    def forbidden():
        raise AssertionError("M1 accessed G_exec")
    assert euclidean_objective(0.125, forbidden) == 0.125


def test_equal_budget_schedule_is_deterministic():
    first = proposal_schedule(60, [0.002, 0.001, 0.0005], 20, 17)
    second = proposal_schedule(60, [0.002, 0.001, 0.0005], 20, 17)
    assert first == second and len(first) == 60
    assert [sum(value[2] == radius for value in first) for radius in (0.002, 0.001, 0.0005)] == [20, 20, 20]


def test_rejected_candidate_cannot_update_incumbent():
    current = np.zeros((3, 2)); candidate = np.ones((3, 2))
    parameters, objective, accepted = accept_incumbent_update(current, 2.0, candidate, 1.0, admitted=False, accepted_count=0, maximum_accepted=12, improvement_tolerance=1e-10)
    assert not accepted and objective == 2.0 and np.array_equal(parameters, current)


def test_terminal_q_and_admission_contracts():
    assert np.isclose(terminal_q_mismatch(np.array([0.0, 0.04]), np.zeros(2)), 0.04)
    limits = AdmissionLimits(0.02, 0.03, 0.05, 0.07237417172157597)
    passed, reasons = hard_admission(surface_length=1.01, baseline_surface_length=1.0, nuc_error=0.1, missed_error=0.05, repeat_error=0.05, baseline_nuc_error=0.08, baseline_missed_error=0.04, baseline_repeat_error=0.04, terminal_mismatch=0.04, strict_pass=True, topology_preserved=True, limits=limits)
    assert passed and not reasons
    failed, reasons = hard_admission(surface_length=1.03, baseline_surface_length=1.0, nuc_error=0.2, missed_error=0.05, repeat_error=0.05, baseline_nuc_error=0.08, baseline_missed_error=0.04, baseline_repeat_error=0.04, terminal_mismatch=0.06, strict_pass=True, topology_preserved=True, limits=limits)
    assert not failed and {"surface_length", "coverage_nuc", "terminal_q"} <= set(reasons)


def test_frozen_sigma_and_window_count_and_hash():
    config = json.loads((ROOT / "configs/riemannian_local_deformation_v1.json").read_text())
    parent = json.loads((ROOT / "configs/riemannian_anisotropy_v1.json").read_text())
    frozen = json.loads((ROOT / "results/riemannian_local_deformation_v1/frozen_windows.json").read_text())
    assert config["robot_contract"]["sigma_safe"] == parent["robot_contract"]["sigma_safe"]
    assert frozen["frozen_before_deformation_results"]
    assert sum(map(len, frozen["windows_by_scene"].values())) == 60
    payload = dict(frozen); expected = payload.pop("content_hash")
    import hashlib
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == expected


def test_archived_local_baseline_witness_reconstruction_is_exact():
    frozen = json.loads((ROOT / "results/riemannian_local_deformation_v1/frozen_windows.json").read_text())
    window = frozen["windows_by_scene"]["saddle/P_low"][0]
    witness = np.load(ROOT / window["witness_file"])
    q = witness["q"][window["start_index"] : window["stop_index"] + 1]
    archived_slice_cost = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    reconstructed = compute_joint_execution_cost((q,)).weighted_joint_length
    assert np.isclose(reconstructed, archived_slice_cost, atol=1e-14)
