import numpy as np

from diffusion_coverage.nuc.robot_lift import NUCTransitionWitness
from diffusion_coverage.robot.execution_cost import compute_joint_execution_cost


def test_transition_witness_joint_length_uses_actual_float64_path():
    q = np.zeros((3, 6), dtype=np.float64)
    q[1, 0] = 0.4
    q[2, 0] = 0.1
    witness = NUCTransitionWitness(
        q=q,
        desired_positions=np.zeros((3, 3)),
        desired_axes=np.tile([0.0, 0.0, 1.0], (3, 1)),
        joint_length=compute_joint_execution_cost((q,)).weighted_joint_length,
    )
    assert witness.q.dtype == np.float64
    assert np.isclose(witness.joint_length, 0.7)
