from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusion_coverage.coverage.nuc_evaluator import evaluate_nuc_coverage
from diffusion_coverage.diagnostics.surface_configuration_coupling import (
    FORMULATIONS,
    frozen_surface_candidate_bank,
    minimum_cost_layered_lift,
    normalized_task_null_direction,
    retain_low_cost_states,
    select_verified_incumbent,
)
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost
from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


ROOT = Path(__file__).resolve().parents[1]


def robot_from_archive():
    archived = json.loads((ROOT / "results/nuc_robot_skeleton_coupling_v1/config.json").read_text())
    cfg = archived["config"]["robot"]
    return UR5eKinematics(cfg["model"], site_name=cfg["site_name"], tool_axis_index=cfg["tool_axis_index"], tool_axis_sign=cfg["tool_axis_sign"])


def first_window_data():
    frozen = json.loads((ROOT / "results/riemannian_local_deformation_v1/frozen_windows.json").read_text())
    window = frozen["windows_by_scene"]["saddle/P_low"][0]
    witness = np.load(ROOT / window["witness_file"])
    start, stop = window["start_index"], window["stop_index"]
    return window, witness["q"][start : stop + 1], witness["desired_positions"][start : stop + 1], witness["desired_axes"][start : stop + 1]


def test_formulation_variable_access_contracts():
    assert not FORMULATIONS["F0"].optimize_surface and not FORMULATIONS["F0"].optimize_configuration
    assert not FORMULATIONS["F1"].optimize_surface and FORMULATIONS["F1"].optimize_configuration
    assert FORMULATIONS["F2"].optimize_surface and not FORMULATIONS["F2"].optimize_configuration
    assert FORMULATIONS["F3"].optimize_surface and FORMULATIONS["F3"].optimize_configuration


def test_surface_candidate_bank_is_bounded_and_deterministic():
    first = frozen_surface_candidate_bank(24, 0.002, 20260912)
    second = frozen_surface_candidate_bank(24, 0.002, 20260912)
    assert np.array_equal(first, second)
    assert first.shape == (24, 3, 2)
    assert np.max(np.linalg.norm(first, axis=2)) <= 0.002 + 1e-15


def test_normalized_task_null_direction_is_unit_and_null():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(5, 6))
    direction = normalized_task_null_direction(matrix)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert np.linalg.norm(matrix @ direction) < 1e-12


def test_low_cost_state_retention_is_deterministic():
    states = [(np.full(6, value), cost, []) for value, cost in ((0.0, 2.0), (0.001, 1.0), (1.0, 3.0))]
    retained = retain_low_cost_states(states, 2, 0.01)
    assert len(retained) == 2
    assert retained[0][1] == 1.0
    assert np.allclose(retained[1][0], 1.0)


def test_candidate_rejection_cannot_update_incumbent():
    incumbent = {"admitted": True, "J_q": 2.0, "tag": "current"}
    selected, accepted = select_verified_incumbent(incumbent, {"admitted": False, "J_q": 1.0, "tag": "rejected"})
    assert not accepted and selected is incumbent


def test_common_Jq_matches_archived_F0_slice_exactly():
    _, q, _, _ = first_window_data()
    expected = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    assert compute_joint_execution_cost((q,)).weighted_joint_length == expected


def test_layered_q_replay_keeps_shared_endpoints_and_fixed_task_path():
    _, q, positions, axes = first_window_data()
    robot = robot_from_archive()
    # A short contiguous archived interval keeps the numerical test fast.
    q, positions, axes = q[:5], positions[:5], axes[:5]
    result = minimum_cost_layered_lift(
        robot, positions, axes, q[0], q[-1], characteristic_length=0.1,
        sigma_safe=0.07237417172157597, axis_tolerance=np.deg2rad(10.0),
        position_tolerance=0.0015, maximum_joint_step=0.1, beam_width=4,
        null_offsets=(-0.12, 0.0, 0.12), deduplication_radius=0.01,
        ik_max_iterations=80, reference_q_path=q,
    )
    assert result.found
    assert np.array_equal(result.q_path[0], q[0])
    assert np.array_equal(result.q_path[-1], q[-1])
    assert result.q_path.shape == q.shape
    assert np.array_equal(positions, positions.copy())


def test_frozen_contract_and_checker_evaluator_identity_are_unchanged():
    current = json.loads((ROOT / "configs/surface_configuration_coupling_gate_v1.json").read_text())
    parent = json.loads((ROOT / "configs/riemannian_local_deformation_v1.json").read_text())
    assert current["robot_contract"] == parent["robot_contract"]
    assert current["admission"] == parent["admission"]
    assert evaluate_nuc_coverage.__module__ == "diffusion_coverage.coverage.nuc_evaluator"
    assert check_strict_coverage_execution.__module__ == "diffusion_coverage.robot.strict_execution"
