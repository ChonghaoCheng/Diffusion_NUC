from __future__ import annotations

import numpy as np

from diffusion_coverage.robot.qspace_coverage_teacher import (
    plan_candidate_routes,
    resample_qspace_segment,
    select_components_marginal,
    simplify_qspace_segment,
)
from diffusion_coverage.robot.surface_ik_graph import (
    SurfaceIKGraph,
    load_surface_ik_graph,
    save_surface_ik_graph,
)
from diffusion_coverage.robot.ur5e_mujoco import IKCandidate


def candidate(q0: float) -> IKCandidate:
    q = np.zeros(6)
    q[0] = q0
    return IKCandidate(q, 0.0, 0.0, 1.0, 0.5, True)


def test_component_teacher_routes_all_nodes_in_connected_component():
    graph = SurfaceIKGraph(
        uv=np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        positions=np.zeros((3, 3)),
        axes=np.tile([0.0, 0.0, 1.0], (3, 1)),
        candidates=((candidate(0.0),), (candidate(0.1),), (candidate(0.2),)),
        edge_index=np.asarray([[0, 1], [1, 2]]),
        edge_compatibility=(np.ones((1, 1), dtype=bool), np.ones((1, 1), dtype=bool)),
        component_labels=(np.asarray([0]), np.asarray([0]), np.asarray([0])),
        grid_shape=(3, 1),
    )
    selected, covered = select_components_marginal(graph, 1)
    routes = plan_candidate_routes(graph, selected)
    assert np.array_equal(selected, [0])
    assert covered.all()
    assert len(routes) == 1
    assert set(routes[0]) == {0, 1, 2}


def test_surface_ik_graph_roundtrip_preserves_routes(tmp_path):
    graph = SurfaceIKGraph(
        uv=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        positions=np.zeros((2, 3)),
        axes=np.tile([0.0, 0.0, 1.0], (2, 1)),
        candidates=((candidate(0.0),), (candidate(0.1),)),
        edge_index=np.asarray([[0], [1]]),
        edge_compatibility=(np.ones((1, 1), dtype=bool),),
        component_labels=(np.asarray([0]), np.asarray([0])),
        grid_shape=(2, 1),
        edge_witnesses=({(0, 0): np.vstack((candidate(0.0).q, candidate(0.1).q))},),
        metadata={"surface_id": "test"},
    )
    path = tmp_path / "graph.npz"
    save_surface_ik_graph(path, graph)
    restored = load_surface_ik_graph(path)
    assert np.load(path)["q_candidates"].dtype == np.float64
    assert np.array_equal(restored.edge_compatibility[0], graph.edge_compatibility[0])
    assert np.allclose(restored.candidates[1][0].q, graph.candidates[1][0].q)
    assert np.allclose(restored.edge_witnesses[0][(0, 0)], graph.edge_witnesses[0][(0, 0)])
    assert restored.metadata == graph.metadata


def test_qspace_resampling_preserves_endpoints_and_axis_norms():
    q = np.column_stack([np.linspace(0.0, value, 20) for value in range(1, 7)])
    positions = np.column_stack((np.linspace(0.0, 1.0, 20), np.zeros((20, 2))))
    axes = np.tile([0.0, 0.0, 1.0], (20, 1))

    q_new, positions_new, axes_new = resample_qspace_segment(
        q, positions, axes, num_tokens=7
    )

    assert q_new.shape == (7, 6)
    assert np.allclose(q_new[[0, -1]], q[[0, -1]])
    assert np.allclose(positions_new[[0, -1]], positions[[0, -1]])
    assert np.allclose(np.linalg.norm(axes_new, axis=1), 1.0)


def test_qspace_simplification_retains_joint_curve_bend():
    q = np.zeros((5, 6))
    q[:, 0] = [0.0, 0.5, 1.0, 1.0, 1.0]
    q[:, 1] = [0.0, 0.0, 0.0, 0.5, 1.0]
    positions = np.zeros((5, 3))
    axes = np.tile([0.0, 0.0, 1.0], (5, 1))

    q_new, _, _ = simplify_qspace_segment(
        q, positions, axes, maximum_joint_error=0.01
    )

    assert np.allclose(q_new[0], q[0])
    assert np.allclose(q_new[-1], q[-1])
    assert any(np.allclose(value, q[2]) for value in q_new)
