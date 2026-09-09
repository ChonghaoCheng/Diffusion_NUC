from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics, sample_axis_cone
from diffusion_coverage.robot.surface_ik_graph import (
    connected_component_labels,
    surface_edge_target_poses,
    surface_grid_edges,
)
from diffusion_coverage.surface import make_cylinder


MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_ur5e_ik_recovers_home_pose_from_perturbed_seed():
    robot = UR5eKinematics(MODEL)
    position, axis = robot.forward(robot.home)
    candidate = robot.solve_ik(position, axis, robot.home + 0.15)
    assert candidate is not None
    assert candidate.position_error <= 1e-3
    assert candidate.axis_error <= np.deg2rad(5.0)


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_fk_generated_short_path_is_continuously_liftable():
    robot = UR5eKinematics(MODEL)
    q_path = np.repeat(robot.home[None, :], 5, axis=0)
    q_path[:, 0] += np.linspace(0.0, 0.12, len(q_path))
    poses = [robot.forward(q) for q in q_path]
    result = robot.check_continuous_lift(
        np.asarray([pose[0] for pose in poses]),
        np.asarray([pose[1] for pose in poses]),
        random_restarts=2,
        rng=np.random.default_rng(4),
    )
    assert result.feasible
    assert result.q_path is not None


def test_surface_grid_edges_wrap_only_periodic_axis():
    open_edges = {tuple(edge) for edge in surface_grid_edges(3, 2, periodic_u=False).T}
    periodic_edges = {tuple(edge) for edge in surface_grid_edges(3, 2, periodic_u=True).T}
    assert (4, 0) not in open_edges
    assert (5, 1) not in open_edges
    assert (4, 0) in periodic_edges
    assert (5, 1) in periodic_edges


def test_surface_ik_components_join_only_compatible_candidates():
    edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    compatibility = (
        np.asarray([[True], [False]]),
        np.asarray([[True]]),
    )
    labels = connected_component_labels((2, 1, 1), edge_index, compatibility)
    assert labels[0][0] == labels[1][0] == labels[2][0]
    assert labels[0][1] != labels[0][0]


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_public_transition_checker_accepts_short_home_motion():
    robot = UR5eKinematics(MODEL)
    valid, adjusted, reason = robot.check_transition(
        robot.home, robot.home + np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert valid
    assert reason is None
    assert np.max(np.abs(adjusted - robot.home)) <= 0.01 + 1e-12


def test_axis_cone_sampling_is_normalized_and_respects_tolerance():
    axis = np.asarray([0.0, 0.0, 1.0])
    tolerance = np.deg2rad(10.0)
    directions = sample_axis_cone(axis, tolerance, 9)
    errors = np.arccos(np.clip(directions @ axis, -1.0, 1.0))
    assert directions.shape == (9, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.all(errors <= tolerance + 1e-12)
    assert np.array_equal(directions[0], axis)


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_nested_orientation_cone_preserves_inner_candidates():
    robot = UR5eKinematics(MODEL)
    position, axis = robot.forward(robot.home)
    inner = robot.enumerate_ik(
        position,
        axis,
        random_restarts=2,
        rng=np.random.default_rng(17),
        axis_tolerance=np.deg2rad(3.0),
        orientation_cone_samples=3,
        max_candidates=8,
    )
    nested = robot.enumerate_ik(
        position,
        axis,
        random_restarts=2,
        rng=np.random.default_rng(17),
        axis_tolerance=np.deg2rad(10.0),
        orientation_cone_samples=3,
        inner_cone_tolerance=np.deg2rad(3.0),
        inner_cone_samples=3,
        inner_max_candidates=8,
        max_candidates=16,
    )
    assert inner
    for candidate in inner:
        assert any(np.allclose(candidate.q, retained.q) for retained in nested)


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_task_transition_rejects_surface_tracking_error():
    robot = UR5eKinematics(MODEL)
    position, axis = robot.forward(robot.home)
    positions = np.repeat(position[None, :], 5, axis=0)
    axes = np.repeat(axis[None, :], 5, axis=0)
    accepted = robot.check_task_transition(robot.home, robot.home, positions, axes)
    assert accepted.feasible
    positions[2, 0] += 0.01
    rejected = robot.check_task_transition(robot.home, robot.home, positions, axes)
    assert not rejected.feasible
    assert rejected.failure_reason == "surface_tracking"


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_task_continuation_tracks_repeated_home_pose():
    robot = UR5eKinematics(MODEL)
    position, axis = robot.forward(robot.home)
    result = robot.continue_task_transition(
        robot.home,
        np.repeat(position[None, :], 4, axis=0),
        np.repeat(axis[None, :], 4, axis=0),
    )
    assert result.feasible
    assert result.q_path.shape == (4, 6)
    assert result.max_position_error <= 1e-3
    targeted = robot.continue_task_transition_to_configuration(
        robot.home,
        robot.home,
        np.repeat(position[None, :], 4, axis=0),
        np.repeat(axis[None, :], 4, axis=0),
    )
    assert targeted.feasible
    assert np.allclose(targeted.q_path[-1], robot.home)


def test_periodic_surface_edge_uses_short_chart_direction():
    surface = make_cylinder(
        radius=1.0, height=1.0, n_azimuth=64, n_height=8, samples_per_face=1
    )
    positions, axes = surface_edge_target_poses(
        surface,
        np.eye(4),
        np.asarray([0.9375, 0.5]),
        np.asarray([0.0625, 0.5]),
        samples=5,
        periodic_u=True,
    )
    assert positions.shape == axes.shape == (5, 3)
    assert positions[2, 0] > 0.95
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)
