from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from diffusion_coverage.robot.task_kinematics import (
    evaluate_task_kinematics_5d,
    normalize_task_jacobian,
    orthonormal_axis_basis,
    task_singularity_metrics,
)
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)


def test_tool_axis_basis_rotation_does_not_change_singular_values():
    rng = np.random.default_rng(2)
    jp = rng.normal(size=(3, 6))
    jw = rng.normal(size=(3, 6))
    axis = np.asarray([0.2, -0.4, 0.9])
    axis /= np.linalg.norm(axis)
    basis = orthonormal_axis_basis(axis)
    angle = 0.73
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    first = task_singularity_metrics(jp, jw, axis, characteristic_length=0.1, axis_basis=basis)[2]
    second = task_singularity_metrics(jp, jw, axis, characteristic_length=0.1, axis_basis=basis @ rotation)[2]
    assert np.allclose(first, second, atol=1e-11)


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_position_jacobian_rows_match_finite_differences():
    robot = UR5eKinematics(MODEL)
    q = robot.home + np.asarray([0.1, -0.08, 0.06, 0.04, -0.05, 0.03])
    metrics = evaluate_task_kinematics_5d(robot, q, characteristic_length=0.1)
    epsilon = 1e-6
    finite = np.column_stack([
        (robot.forward(q + epsilon * np.eye(6)[joint])[0] - robot.forward(q - epsilon * np.eye(6)[joint])[0]) / (2.0 * epsilon)
        for joint in range(6)
    ])
    assert np.allclose(metrics.jacobian_5[:3], finite, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_tool_axis_jacobian_rows_match_finite_differences():
    robot = UR5eKinematics(MODEL)
    q = robot.home + np.asarray([0.12, -0.07, 0.05, 0.08, -0.04, 0.02])
    metrics = evaluate_task_kinematics_5d(robot, q, characteristic_length=0.1)
    epsilon = 1e-6
    axis_derivative = np.column_stack([
        (robot.forward(q + epsilon * np.eye(6)[joint])[1] - robot.forward(q - epsilon * np.eye(6)[joint])[1]) / (2.0 * epsilon)
        for joint in range(6)
    ])
    u, v = metrics.axis_basis.T
    recovered_angular_components = np.vstack((-v @ axis_derivative, u @ axis_derivative))
    assert np.allclose(metrics.jacobian_5[3:], recovered_angular_components, atol=2e-6, rtol=2e-5)


def test_length_unit_scaling_preserves_normalized_singular_values():
    rng = np.random.default_rng(3)
    jp_m = rng.normal(size=(3, 6)) * 0.2
    jw = rng.normal(size=(3, 6))
    axis = np.asarray([0.0, 0.0, 1.0])
    metric_m = task_singularity_metrics(jp_m, jw, axis, characteristic_length=0.1)[2]
    metric_mm = task_singularity_metrics(1000.0 * jp_m, jw, axis, characteristic_length=100.0)[2]
    assert np.allclose(metric_m, metric_mm, atol=1e-12)


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_tool_axis_spin_does_not_change_axis_symmetric_task_definition():
    robot = UR5eKinematics(MODEL)
    q = robot.home + 0.05
    metrics = evaluate_task_kinematics_5d(robot, q, characteristic_length=0.1)
    spin = 1.1
    plane_rotation = np.asarray([[np.cos(spin), -np.sin(spin)], [np.sin(spin), np.cos(spin)]])
    rotated = task_singularity_metrics(
        metrics.jacobian_5[:3],
        np.linalg.lstsq(metrics.axis_basis.T, metrics.jacobian_5[3:], rcond=None)[0],
        metrics.tool_axis,
        characteristic_length=0.1,
        axis_basis=metrics.axis_basis @ plane_rotation,
    )[2]
    assert np.allclose(metrics.singular_values, rotated, atol=1e-10)


def test_normalized_jacobian_rejects_nonpositive_characteristic_length():
    with pytest.raises(ValueError):
        normalize_task_jacobian(np.zeros((5, 6)), 0.0)
