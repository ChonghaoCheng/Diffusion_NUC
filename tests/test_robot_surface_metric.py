from __future__ import annotations

from dataclasses import replace

import numpy as np

from diffusion_coverage.diagnostics.execution_metric import (
    local_task_increment,
    predicted_local_execution_length,
)
from diffusion_coverage.geometry.robot_surface_metric import (
    compute_robot_surface_metric,
    estimate_surface_contact_differential,
    orthonormal_surface_tangent,
    smooth_surface_normal,
)
from diffusion_coverage.geometry.surface_curve import trace_surface_curve
from diffusion_coverage.diagnostics.anisotropy import (
    continue_probe_from_shared_q,
    require_frozen_experiment_inputs,
)
from diffusion_coverage.robot.task_kinematics import TaskKinematics5D, orthonormal_axis_basis
from diffusion_coverage.surface import make_hemisphere, make_saddle


def fake_task(jacobian: np.ndarray) -> TaskKinematics5D:
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return TaskKinematics5D(
        position=np.zeros(3), tool_axis=np.array([0.0, 0.0, -1.0]),
        axis_basis=np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]),
        jacobian_5=jacobian.copy(), normalized_jacobian_5=jacobian.copy(),
        singular_values=singular, sigma_min_5=float(singular[-1]),
        mu_bar=float(np.prod(singular)), characteristic_length=1.0,
    )


def test_synthetic_diagonal_metric_predicts_ratio_two_and_sign_symmetry():
    jacobian = np.eye(5, 6)
    differential = np.zeros((5, 2)); differential[0, 0] = 1.0; differential[1, 1] = 2.0
    metric = compute_robot_surface_metric(fake_task(jacobian), differential)
    assert np.allclose(metric.matrix, np.diag([1.0, 4.0]))
    assert np.allclose(metric.eigenvalues, [1.0, 4.0])
    assert metric.R_G == 2.0
    for vector in metric.eigenvectors.T:
        assert np.isclose(vector @ metric.matrix @ vector, (-vector) @ metric.matrix @ (-vector))


def test_tangent_basis_rotation_preserves_eigenvalues_and_physical_directions():
    surface = make_hemisphere(radius=0.14, n_azimuth=24, n_polar=10, samples_per_face=1)
    point = np.array([0.06, 0.04, np.sqrt(0.14**2 - 0.06**2 - 0.04**2)])
    normal = point / np.linalg.norm(point); tangent = orthonormal_surface_tangent(normal)
    axis_basis = orthonormal_axis_basis(-normal)
    first = estimate_surface_contact_differential(surface, point, np.eye(4), axis_basis, characteristic_length=0.1, tangent_basis_surface=tangent)
    angle = 0.61; rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    second = estimate_surface_contact_differential(surface, point, np.eye(4), axis_basis, characteristic_length=0.1, tangent_basis_surface=tangent @ rotation)
    task = fake_task(np.column_stack((np.eye(5), np.zeros(5))))
    metric_first = compute_robot_surface_metric(task, first.task_differential)
    metric_second = compute_robot_surface_metric(task, second.task_differential)
    assert np.allclose(metric_first.eigenvalues, metric_second.eigenvalues, rtol=2e-3, atol=2e-3)
    directions_first = first.tangent_basis @ metric_first.eigenvectors
    directions_second = second.tangent_basis @ metric_second.eigenvectors
    assert np.allclose(np.abs(directions_first.T @ directions_second), np.eye(2), atol=2e-3)


def test_B_h_matches_finite_difference_and_contains_axis_change():
    surface = make_hemisphere(radius=0.14, n_azimuth=48, n_polar=24, samples_per_face=1)
    point = np.array([0.04, -0.03, np.sqrt(0.14**2 - 0.04**2 - 0.03**2)])
    _, normal = smooth_surface_normal(surface, point[None]); axis = -normal[0]
    axis_basis = orthonormal_axis_basis(axis)
    contact = estimate_surface_contact_differential(surface, point, np.eye(4), axis_basis, characteristic_length=0.1, finite_difference_step=2e-6)
    delta = np.array([2.0e-6, -1.5e-6])
    moved_point, moved_normal = smooth_surface_normal(surface, (contact.point_surface + contact.tangent_basis @ delta)[None])
    numerical = local_task_increment(
        contact.point_base, contact.tool_axis, moved_point[0], -moved_normal[0],
        axis_basis, characteristic_length=0.1,
    )
    predicted = contact.task_differential @ delta
    assert np.linalg.norm(contact.task_differential[3:]) > 0.0
    assert np.allclose(predicted, numerical, atol=2e-7, rtol=2e-2)


def test_surface_metric_local_cost_matches_existing_D3_formula():
    rng = np.random.default_rng(12)
    jacobian = rng.normal(size=(5, 6)); jacobian[:, :5] += 3.0 * np.eye(5)
    differential = rng.normal(size=(5, 2))
    task = fake_task(jacobian)
    metric = compute_robot_surface_metric(task, differential)
    delta_xi = np.array([2e-3, -1e-3])
    previous = predicted_local_execution_length(jacobian, differential @ delta_xi)
    current = np.sqrt(delta_xi @ metric.matrix @ delta_xi)
    assert np.isclose(previous, current, atol=1e-12, rtol=1e-12)
    assert metric.eigenvalues[0] > 0.0


def test_surface_curve_matches_requested_intrinsic_length():
    surface = make_saddle(width=0.24, height=0.24, curvature=1.5, nx=16, ny=16, samples_per_face=1)
    start = np.array([0.0, 0.0, 0.0]); direction = np.array([1.0, 0.2, 0.0])
    curve = trace_surface_curve(surface, start, direction, 0.004, maximum_step=2e-4)
    integrated = np.linalg.norm(np.diff(curve.points, axis=0), axis=1).sum()
    assert np.isclose(curve.intrinsic_length, integrated, atol=1e-12)
    assert abs(curve.intrinsic_length - 0.004) / 0.004 < 0.01
    assert np.allclose(np.linalg.norm(curve.normals, axis=1), 1.0)


def test_metric_rejects_nonpositive_matrix():
    differential = np.zeros((5, 2)); differential[0, 0] = 1.0
    with np.testing.assert_raises(ValueError):
        compute_robot_surface_metric(fake_task(np.eye(5, 6)), differential)


def test_probe_pair_preserves_identical_shared_q():
    class FakeRobot:
        def continue_task_transition(self, start, positions, axes, **kwargs):
            class Result:
                feasible=True; failure_reason=None; max_position_error=0.0; max_axis_error=0.0
                q_path=np.vstack((start, start + 0.01))
            return Result()
    q=np.arange(6,dtype=float)
    first=continue_probe_from_shared_q(FakeRobot(),q,np.zeros((2,3)),np.tile([0,0,1.],(2,1)),maximum_joint_step=.1,position_tolerance=.003,axis_tolerance=.1)
    second=continue_probe_from_shared_q(FakeRobot(),q,np.zeros((2,3)),np.tile([0,0,1.],(2,1)),maximum_joint_step=.1,position_tolerance=.003,axis_tolerance=.1)
    assert np.array_equal(first.q_path[0],q)
    assert np.array_equal(second.q_path[0],q)


def test_R1_refuses_unfrozen_scenes_or_anchors():
    with np.testing.assert_raises(RuntimeError): require_frozen_experiment_inputs({})
    with np.testing.assert_raises(RuntimeError): require_frozen_experiment_inputs({"frozen_before_R1":True},{})
