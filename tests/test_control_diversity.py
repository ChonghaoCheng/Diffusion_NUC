from __future__ import annotations

import numpy as np

from diffusion_coverage.evaluation import (
    cluster_controls,
    control_energy_distance,
    expert_control_coverage,
    leave_one_out_control_coverage,
)


def test_control_clustering_and_expert_coverage_are_distance_based():
    controls = [
        np.zeros((3, 2)),
        np.full((3, 2), 0.01),
        np.full((3, 2), 0.30),
    ]
    clustered = cluster_controls(controls, threshold=0.05)
    assert clustered.num_clusters == 2
    assert clustered.assignments.tolist() == [0, 0, 1]
    assert 0.0 < clustered.normalized_entropy <= 1.0
    coverage, mean_distance = expert_control_coverage(
        controls[:2], [controls[0], controls[2]], threshold=0.05
    )
    assert coverage == 0.5
    assert mean_distance is not None and mean_distance > 0.1


def test_empty_generated_controls_cover_no_experts():
    coverage, mean_distance = expert_control_coverage(
        [], [np.zeros((2, 2))], threshold=0.05
    )
    assert coverage == 0.0
    assert mean_distance is None


def test_teacher_self_coverage_and_energy_distance_contextualize_sparse_samples():
    close = [np.zeros((2, 2)), np.full((2, 2), 0.01)]
    far = [np.full((2, 2), 0.5), np.full((2, 2), 0.6)]
    assert leave_one_out_control_coverage(close, threshold=0.05) == 1.0
    assert leave_one_out_control_coverage(far, threshold=0.05) == 0.0
    assert control_energy_distance(close, close) == 0.0
    assert control_energy_distance(close, far) > 0.5
