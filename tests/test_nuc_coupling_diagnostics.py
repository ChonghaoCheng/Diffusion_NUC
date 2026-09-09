from __future__ import annotations

import numpy as np

from diffusion_coverage.diagnostics.continuation import FROZEN_ADMISSION_KEYS, classify_continuation_failure, require_unchanged_admission_contract
from diffusion_coverage.diagnostics.cost_decomposition import classify_transition_commonality, decompose_transition_costs
from diffusion_coverage.diagnostics.execution_metric import local_task_increment, predicted_local_execution_length
from diffusion_coverage.diagnostics.structure import compare_skeletons
from diffusion_coverage.nuc.adapter import NUCSkeleton
from diffusion_coverage.nuc.robot_lift import NUCContinuationLayerTrace


def skeleton(codes):
    values=np.asarray(codes,dtype=np.int64)
    return NUCSkeleton(values,np.column_stack((values,np.zeros((len(values),2)))),np.unique(values//3),np.empty((0,3),dtype=np.int64),"toy",None,(),int(values[0]//3))


def test_structure_metrics_ignore_metadata_and_match_identical_skeleton():
    first=skeleton([0,1,2,3]); second=skeleton([0,1,2,3])
    object.__setattr__(second,"metadata",{"irrelevant":"value"})
    metric=compare_skeletons(first,second,resample_count=20)
    assert metric.directed_transition_jaccard == 1.0
    assert metric.undirected_transition_jaccard == 1.0
    assert metric.normalized_sequence_distance == 0.0
    assert metric.geometric_path_distance == 0.0


def test_known_skeleton_transition_metrics():
    metric=compare_skeletons(skeleton([0,1,2]),skeleton([0,2,1]),resample_count=10)
    assert metric.directed_transition_jaccard == 0.0
    assert metric.undirected_transition_jaccard == 1.0/3.0
    assert np.isclose(metric.normalized_sequence_distance,2.0/3.0)


def test_transition_cost_partition_reproduces_witness_and_is_resampling_stable():
    codes=np.asarray([0,1,3])
    q=np.asarray([[0.,0.],[1.,0.],[1.,1.]])
    costs=decompose_transition_costs(codes,q,np.asarray([2,2]),expected_total=2.0)
    assert np.isclose(sum(item.joint_length for item in costs),2.0)
    dense=np.asarray([[0.,0.],[.5,0.],[1.,0.],[1.,.5],[1.,1.]])
    dense_costs=decompose_transition_costs(codes,dense,np.asarray([3,3]),expected_total=2.0)
    assert np.isclose(sum(item.joint_length for item in dense_costs),2.0)


def test_common_variable_transition_classification_is_deterministic():
    sequences=[np.asarray([0,1,2]),np.asarray([0,1,3])]
    first=classify_transition_commonality(sequences); second=classify_transition_commonality(sequences)
    assert first == second
    assert first[1] == {(0,1)}
    assert first[2] == {(1,2),(1,3)}


def test_identity_task_metric_has_expected_length_and_is_nonnegative():
    jacobian=np.column_stack((np.eye(5),np.zeros(5)))
    delta=np.asarray([3.,4.,0.,0.,0.])
    assert np.isclose(predicted_local_execution_length(jacobian,delta),5.0)
    assert predicted_local_execution_length(jacobian,-delta) >= 0.0


def test_task_metric_is_invariant_to_axis_plane_basis_rotation():
    rng=np.random.default_rng(4)
    jp=rng.normal(size=(3,6)); jr=rng.normal(size=(2,6)); j=np.vstack((jp,jr))
    angle=.61; rotation=np.asarray([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    delta=np.asarray([.1,-.2,.3,.04,-.05]); transformed=np.concatenate((delta[:3],rotation.T@delta[3:]))
    transformed_j=np.vstack((jp,rotation.T@jr))
    assert np.isclose(predicted_local_execution_length(j,delta),predicted_local_execution_length(transformed_j,transformed),atol=1e-11)


def test_pure_tool_axis_spin_has_zero_orientation_increment():
    basis=np.asarray([[1.,0.],[0.,1.],[0.,0.]])
    delta=local_task_increment(np.zeros(3),np.asarray([0.,0.,1.]),np.zeros(3),np.asarray([0.,0.,1.]),basis,characteristic_length=.1)
    assert np.allclose(delta,np.zeros(5))


def test_near_singular_metric_is_stable_above_threshold():
    jacobian=np.diag([1.,1.,1.,1.,.08])
    value=predicted_local_execution_length(jacobian,np.ones(5),minimum_singular_value=.0723741717)
    assert np.isfinite(value) and value > 0.0


def layer(raw, safe, incoming, outgoing):
    return NUCContinuationLayerTrace(1,0,1,.5,raw,raw,raw,raw,safe,incoming,outgoing,outgoing,outgoing,None,None,None,(0.,0.,0.),(0.,0.,1.))


def test_failure_classifier_distinguishes_empty_transition_and_recovery():
    assert classify_continuation_failure(layer(0,0,0,0)) == "pose_candidate_empty"
    assert classify_continuation_failure(layer(2,0,1,0)) == "safety_filter_exhaustion"
    transition=layer(2,2,2,0)
    assert classify_continuation_failure(transition) == "transition_graph_empty"
    assert classify_continuation_failure(transition,strong_search_recovered=True) == "beam_or_search_exhaustion"


def test_strong_search_cannot_change_frozen_admission_contract():
    contract={key:float(index) for index,key in enumerate(FROZEN_ADMISSION_KEYS)}
    require_unchanged_admission_contract(contract,dict(contract))
    changed=dict(contract); changed["sigma_safe"] += .01
    try:
        require_unchanged_admission_contract(contract,changed)
    except ValueError:
        pass
    else:
        raise AssertionError("a changed admission threshold was accepted")
