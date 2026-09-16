from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from diffusion_coverage.robot.synchronized_motion import (
    SynchronizedMotionTrace,
    continue_to_configuration_synchronized,
    densify_synchronized_trace,
    spherical_polyline_evaluator,
)
from diffusion_coverage.robot.ur5e_mujoco import TaskTransitionResult


class FakeRobot:
    def __init__(self):
        self.backends = []

    def continue_task_transition(self, start, positions, axes, **kwargs):
        self.backends.append(kwargs["backend"])
        q = np.repeat(np.asarray(start)[None, :], len(positions), axis=0)
        q[:, 0] = np.linspace(start[0], start[0] + 0.1, len(positions))
        return TaskTransitionResult(True, q, None, 0.0, 0.0)

    def check_task_transition(self, start, end, positions, axes, **kwargs):
        fractions = np.linspace(0.0, 1.0, len(positions))
        q = (1.0 - fractions[:, None]) * np.asarray(start) + fractions[:, None] * np.asarray(end)
        feasible = bool(np.max(np.abs(q[-1] - end)) <= 1e-12 and end[1] < 2.0)
        return TaskTransitionResult(feasible, q, None if feasible else "surface_tracking", 0.0 if feasible else 1.0, 0.0)


def _curve(count):
    radius = 0.14
    theta = np.linspace(0.1, 0.7, count)
    points = radius * np.column_stack((np.cos(theta), np.sin(theta), np.zeros(count)))
    return spherical_polyline_evaluator(points, np.eye(4), radius=radius)


@pytest.mark.parametrize("count", [3, 5, 11])
@pytest.mark.parametrize("tail", [2, 3, 5])
def test_targeted_tail_returns_synchronized_rows(count, tail):
    knots, curve = _curve(count)
    robot = FakeRobot()
    end = np.asarray([0.2, 0.3, 0, 0, 0, 0], dtype=float)
    result, trace = continue_to_configuration_synchronized(
        robot, np.zeros(6), end, knots, curve, geometry_arc_id=4,
        start_node=2, end_node=3, final_tracking_samples=tail,
    )
    assert result.feasible and trace is not None
    assert len(trace.q) == count + tail - 2
    assert len(trace.u) == len(trace.target_position) == len(trace.target_axis) == len(trace.activity) == len(trace.q)
    expected_position, expected_axis = curve(trace.u)
    np.testing.assert_allclose(trace.target_position, expected_position, atol=1e-14)
    np.testing.assert_allclose(trace.target_axis, expected_axis, atol=1e-14)
    np.testing.assert_allclose(trace.q[-1], end, atol=0.0)
    assert robot.backends == ["task5"]


def test_nonuniform_parameter_and_reverse_preserve_curve_endpoints():
    _, curve = _curve(5)
    u = np.asarray([0.0, 0.05, 0.4, 0.91, 1.0])
    robot = FakeRobot()
    result, trace = continue_to_configuration_synchronized(
        robot, np.zeros(6), np.ones(6) * 0.2, u, curve,
        geometry_arc_id=9, start_node=7, end_node=8, final_tracking_samples=5,
    )
    assert result.feasible
    reverse = trace.reversed()
    assert reverse.start_node == 8 and reverse.end_node == 7
    assert np.all(np.diff(reverse.u) >= 0.0)
    np.testing.assert_array_equal(reverse.q[0], trace.q[-1])
    np.testing.assert_array_equal(reverse.q[-1], trace.q[0])
    np.testing.assert_array_equal(reverse.target_position[0], trace.target_position[-1])


def test_densification_retains_terminal_interval_and_rejects_mismatch():
    knots, curve = _curve(3)
    _, trace = continue_to_configuration_synchronized(
        FakeRobot(), np.zeros(6), np.ones(6) * 0.2, knots, curve,
        geometry_arc_id=1, start_node=0, end_node=1, final_tracking_samples=5,
    )
    dense = densify_synchronized_trace(trace, curve, maximum_joint_step=0.025, maximum_parameter_step=0.05)
    np.testing.assert_array_equal(dense.q[0], trace.q[0])
    np.testing.assert_array_equal(dense.q[-1], trace.q[-1])
    assert dense.u[-1] == trace.u[-1]
    with pytest.raises(ValueError, match="unequal"):
        SynchronizedMotionTrace(trace.q, trace.u[:-1], trace.target_position, trace.target_axis, trace.activity, 1, (0, 1), 0, 1)


def test_incompatible_target_is_rejected_without_snapping():
    knots, curve = _curve(5)
    end = np.asarray([0.0, 2.5, 0, 0, 0, 0])
    result, trace = continue_to_configuration_synchronized(
        FakeRobot(), np.zeros(6), end, knots, curve,
        geometry_arc_id=3, start_node=0, end_node=1, final_tracking_samples=3,
    )
    assert not result.feasible
    assert trace is None


def test_stationary_parameter_tail_uses_declared_parameter():
    position = np.asarray([[0.0, 0.0, 0.14], [0.0, 0.0, 0.14]])
    knots, curve = spherical_polyline_evaluator(position, np.eye(4), radius=0.14)
    assert knots.tolist() == [0.0, 1.0]
    q = np.asarray([np.zeros(6), np.ones(6) * 0.01])
    p, a = curve(knots)
    trace = SynchronizedMotionTrace(q, knots, p, a, np.asarray([True, True]), 0, (0.0, 1.0), 0, 0)
    dense = densify_synchronized_trace(trace, curve, maximum_joint_step=0.005, maximum_parameter_step=0.25)
    assert len(dense.q) >= 5
    np.testing.assert_allclose(dense.target_position, np.repeat(position[:1], len(dense.q), axis=0))
