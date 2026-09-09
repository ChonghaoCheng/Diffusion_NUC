from __future__ import annotations

import numpy as np
import pytest

from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost


def test_joint_execution_length_is_invariant_to_linear_resampling():
    start = np.asarray([0.0, -0.2, 0.1])
    end = np.asarray([1.0, 0.4, -0.5])
    sparse = np.stack((start, end))
    dense = np.linspace(start, end, 101)
    sparse_cost = compute_joint_execution_cost((sparse,))
    dense_cost = compute_joint_execution_cost((dense,))
    assert np.isclose(sparse_cost.weighted_joint_length, dense_cost.weighted_joint_length)
    assert np.isclose(sparse_cost.unweighted_joint_travel, dense_cost.unweighted_joint_travel)
    assert np.allclose(sparse_cost.per_joint_travel, dense_cost.per_joint_travel)


def test_joint_execution_cost_uses_actual_unwrapped_witness():
    q = np.asarray([[3.0, 0.0], [-3.0, 0.0]])
    cost = compute_joint_execution_cost((q,))
    assert np.isclose(cost.weighted_joint_length, 6.0)
    assert np.isclose(cost.max_per_joint_travel, 6.0)


def test_joint_execution_cost_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        compute_joint_execution_cost((np.zeros((2, 2)),), weight_matrix=np.diag([1.0, 0.0]))
