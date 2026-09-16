from __future__ import annotations

import numpy as np

from diffusion_coverage.robot.e09_execution import sphere_membership_stream
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics


def test_exact_sphere_membership_handles_seam_and_pole():
    radius = 0.14
    angles = np.asarray([1e-8, 2 * np.pi - 1e-8])
    samples = radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(2)))
    trace = np.asarray([[radius, 0.0, 0.0], [0.0, 0.0, radius]])
    membership = sphere_membership_stream(
        samples, trace, radius=radius, footprint_radius=0.008
    )
    assert np.all(membership[:, 0])
    assert not np.any(membership[:, 1])


def test_e09_continuation_forwards_task5(monkeypatch):
    robot = object.__new__(UR5eKinematics)
    robot.lower_limits = np.full(6, -10.0)
    robot.upper_limits = np.full(6, 10.0)
    calls = []

    class Candidate:
        q = np.zeros(6)

    def solve(_position, _axis, _seed, **kwargs):
        calls.append(kwargs.get("backend"))
        return Candidate()

    monkeypatch.setattr(robot, "solve_ik", solve)
    monkeypatch.setattr(robot, "check_transition", lambda *a, **k: (True, np.zeros(6), None))
    monkeypatch.setattr(robot, "forward", lambda q: (np.zeros(3), np.asarray([0.0, 0.0, 1.0])))
    positions = np.zeros((3, 3))
    axes = np.tile([0.0, 0.0, 1.0], (3, 1))
    result = robot.continue_task_transition(
        np.zeros(6), positions, axes, backend="task5"
    )
    assert result.feasible
    assert calls == ["task5", "task5"]


def test_off_samples_do_not_enter_on_task_residual_extrema(monkeypatch):
    from diffusion_coverage.robot.e09_execution import evaluate_e09_fk_trace
    from types import SimpleNamespace
    class FakeRobot:
        def evaluate_configuration(self, q):
            return SimpleNamespace(joint_limit_margin=1.0, collision_free=True)
    robot = FakeRobot()
    monkeypatch.setattr("diffusion_coverage.robot.e09_execution.evaluate_task_kinematics_5d", lambda r, q, characteristic_length: SimpleNamespace(position=np.asarray([q[0],0.0,0.14]), tool_axis=np.asarray([0.0,0.0,-1.0]), sigma_min_5=1.0))
    q = np.zeros((3, 6)); q[1, 0] = 0.02
    active = np.asarray([True, False, True])
    target = np.repeat(np.asarray([[0.0, 0.0, 0.14]]), 3, axis=0)
    check = evaluate_e09_fk_trace(
        robot, q, active, target, np.eye(4), target[:1], np.ones(1),
        sphere_radius=0.14, footprint_radius=0.008, characteristic_length=0.1,
    )
    assert check.max_position_error == 0.0
    assert check.max_axis_error == 0.0
