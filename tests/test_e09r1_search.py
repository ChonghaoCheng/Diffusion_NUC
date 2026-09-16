from __future__ import annotations

import numpy as np

from scripts.e09r1_runner_support import _pareto_admit_fixed, fixed_route_search
from diffusion_coverage.coverage.episode_summary import summarize_ordered_membership
from diffusion_coverage.robot.e09_execution import sphere_episode_counts_indexed, sphere_membership_stream
from diffusion_coverage.solvers.completion_bound import CompletionEdge
from diffusion_coverage.solvers.history_search import SearchGraph, search_history_graph


def _edge(edge_id, start, end, columns, weights, cost, off=False):
    values = np.asarray(columns, dtype=bool).T
    activity = np.ones(values.shape[1], dtype=bool)
    if off:
        activity = np.zeros(values.shape[1], dtype=bool); activity[[0, -1]] = True
    return CompletionEdge(edge_id, start, end, summarize_ordered_membership(values, weights, active=activity), cost)


def _config():
    return {"coverage": {"missed_tolerance": 0.0, "repeat_tolerance": 0.1}, "search": {"private_memory_gib": 6, "conservative_resident_label_limit": 30000}}


def test_fixed_pareto_retains_costlier_lower_repeat_prefix():
    weights = np.asarray([0.06, 0.44, 0.50])
    membership = tuple(np.eye(3, dtype=bool))
    cheap = _edge(0, 0, 1, [[1,0,0],[0,1,0],[1,1,0],[0,1,0]], weights, 1.0)
    low_repeat = _edge(1, 0, 1, [[1,0,0],[1,1,0],[0,1,0]], weights, 2.0)
    suffix = _edge(2, 1, 2, [[0,1,0],[1,1,1],[0,0,1]], weights, 1.0)
    graph = SearchGraph(membership, (cheap, low_repeat, suffix), weights, "pareto-fixed")
    data = {"graph": graph, "edge_meta": [
        {"kind":"source","geom_arc_id":10}, {"kind":"source","geom_arc_id":10}, {"kind":"source","geom_arc_id":11}],
        "routes":{"route/forward":(10,11)}, "arc_start":np.asarray([0]*12), "arc_end":np.asarray([0]*12),
        "node_ports":np.asarray([0,1,2]),
    }
    data["arc_start"][10]=0;data["arc_end"][10]=1;data["arc_start"][11]=1;data["arc_end"][11]=2
    result = fixed_route_search(data, 0, 1, _config(), wall_time=1.0, expanded_limit=100)
    assert result.incumbent is not None
    assert result.incumbent.path == (1, 2)
    assert result.incumbent.joint_cost == 3.0


def test_fixed_off_relocation_advances_progress_without_implicit_coverage():
    weights=np.ones(3)
    membership=tuple(np.eye(3,dtype=bool))
    off=_edge(0,0,1,[[1,0,0],[0,0,0],[0,1,0]],weights,0.5,off=True)
    suffix=_edge(1,1,2,[[0,1,0],[0,1,1],[0,0,1]],weights,1.0)
    graph=SearchGraph(membership,(off,suffix),weights,"off-progress")
    data={"graph":graph,"edge_meta":[{"kind":"off_reconfiguration","geom_arc_id":-1},{"kind":"source","geom_arc_id":11}],"routes":{"route/forward":(10,11)},"arc_start":np.asarray([0]*12),"arc_end":np.asarray([0]*12),"node_ports":np.asarray([0,1,2])}
    data["arc_start"][10]=0;data["arc_end"][10]=1;data["arc_start"][11]=1;data["arc_end"][11]=2
    result=fixed_route_search(data,0,2,_config(),wall_time=1.0,expanded_limit=100)
    assert result.incumbent is not None and result.incumbent.path==(0,1)
    assert result.incumbent.used_on_segments==2


def test_fixed_pareto_helper_is_two_resource():
    table={};key=("r",0)
    assert _pareto_admit_fixed(table,key,0.08,1.0)
    assert _pareto_admit_fixed(table,key,0.02,2.0)
    assert not _pareto_admit_fixed(table,key,0.09,3.0)
    assert len(table[key])==2


def test_indexed_episode_backend_matches_dense_membership():
    radius=0.14
    theta=np.linspace(0,2*np.pi,64,endpoint=False)
    samples=radius*np.column_stack((np.cos(theta),np.sin(theta),np.zeros_like(theta)))
    trace=samples[[0,1,2,20,1,2]]
    active=np.asarray([True,True,False,False,True,True])
    dense=sphere_membership_stream(samples,trace,radius=radius,footprint_radius=0.008)
    dense[:,~active]=False
    starts=dense.copy();starts[:,1:]&=~dense[:,:-1]
    indexed=sphere_episode_counts_indexed(samples,trace,active,radius=radius,footprint_radius=0.008)
    np.testing.assert_array_equal(indexed,starts.sum(axis=1))


def test_exhausted_single_solution_is_contained_in_full_graph():
    weights=np.ones(2);membership=tuple(np.eye(2,dtype=bool));edge=_edge(0,0,1,[[1,0],[1,1],[0,1]],weights,1.0)
    graph=SearchGraph(membership,(edge,),weights,"inclusion")
    outcomes=[]
    for use_bound in (False,True):
        outcomes.append(search_history_graph(graph,start_node=0,maximum_on_segments=1,missed_tolerance=0.0,repeat_tolerance=0.0,use_completion_bound=use_bound,wall_time_s=1.0,expanded_limit=100,checkpoint_times=()))
    assert all(x.optimality_proved and x.incumbent is not None for x in outcomes)
    assert outcomes[0].incumbent.joint_cost==outcomes[1].incumbent.joint_cost==1.0
