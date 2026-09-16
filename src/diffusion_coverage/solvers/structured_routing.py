from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np

from diffusion_coverage.coverage.episode_summary import (
    EpisodeEdgeSummary, EpisodeState, apply_edge_summary, initial_episode_state,
)
from diffusion_coverage.solvers.completion_bound import CompletionEdge
from diffusion_coverage.solvers.history_search import SearchGraph, SearchLabel, SearchMetrics


@dataclass(frozen=True)
class StructuredAction:
    start: int
    end: int
    summary: EpisodeEdgeSummary
    joint_cost: float
    edge_ids: tuple[int, ...]
    kind: str


@dataclass
class AnytimeMetrics:
    expanded: int = 0
    generated_actions: int = 0
    atomic_equivalent_work: int = 0
    dominance_pruned: int = 0
    repeat_pruned: int = 0
    segment_pruned: int = 0
    reachability_pruned: int = 0
    objective_pruned: int = 0
    heuristic_evictions: int = 0
    stale_labels: int = 0
    rediscoveries: int = 0
    peak_open: int = 0
    peak_ancestry: int = 0
    peak_pareto_records: int = 0
    run_actions_evaluated: int = 0
    run_cache_hits: int = 0
    run_cache_misses: int = 0
    run_cache_peak_bytes: int = 0
    screen_calls: int = 0
    screen_seconds: float = 0.0


@dataclass(frozen=True)
class Candidate:
    label_id: int
    edge_ids: tuple[int, ...]
    node: int
    covered: np.ndarray
    membership: np.ndarray
    repeat_error: float
    joint_cost: float
    used_on_segments: int
    q2_miss: float
    screen_status: str
    q3_miss: float | None = None
    q3_repeat: float | None = None
    screen_seconds: float = 0.0


@dataclass(frozen=True)
class AnytimeResult:
    candidates: tuple[Candidate, ...]
    screened_candidates: tuple[Candidate, ...]
    termination: str
    elapsed_seconds: float
    metrics: AnytimeMetrics
    checkpoints: tuple[dict[str, Any], ...]
    first_q2_seconds: float | None
    first_q3_pass_seconds: float | None
    fallback_retained: bool
    run_catalog: dict[str, Any]


@dataclass
class _Label:
    label_id: int
    node: int
    covered: np.ndarray
    membership: np.ndarray
    repeat_error: float
    joint_cost: float
    used_on_segments: int
    parent_id: int | None
    action_edges: tuple[int, ...]
    serial: int
    expanded: bool = False
    live: bool = False


def compose_episode_summaries(summaries: Iterable[EpisodeEdgeSummary], weights: np.ndarray) -> EpisodeEdgeSummary:
    values = tuple(summaries)
    if not values:
        raise ValueError("at least one summary is required")
    footprint = values[0].footprint.copy()
    counts = values[0].episode_counts.astype(np.int64, copy=True)
    start = values[0].start_membership.copy()
    end = values[0].end_membership.copy()
    off_to_on = int(values[0].off_to_on_count)
    for item in values[1:]:
        if not np.array_equal(end, item.start_membership):
            raise ValueError("summary endpoint membership mismatch")
        counts += item.episode_counts.astype(np.int64) - (end & item.start_membership).astype(np.int64)
        footprint |= item.footprint
        end = item.end_membership.copy()
        off_to_on += int(item.off_to_on_count)
    return EpisodeEdgeSummary(footprint, counts, start, end, float(np.dot(np.asarray(weights), counts)), off_to_on)


def replay_edge_sequence(graph: SearchGraph, start_node: int, edge_ids: Iterable[int]) -> SearchLabel:
    state = initial_episode_state(graph.node_membership[start_node])
    node = int(start_node); used = 1; cost = 0.0; path = []
    for edge_id in edge_ids:
        edge = graph.edges[int(edge_id)]
        if edge.start != node:
            raise ValueError("edge sequence has a node discontinuity")
        state = apply_edge_summary(state, edge.summary, graph.weights)
        used += edge.summary.off_to_on_count; cost += edge.joint_cost; node = edge.end; path.append(int(edge_id))
    return SearchLabel(node, state.covered, state.membership, state.repeat_error, cost, used, tuple(path))


def fixed_route_initialize(data: dict[str, Any], start: int, maximum_on_segments: int, config: dict[str, Any], *, wall_time: float, expanded_limit: int) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """Run the corrected F language while collecting deterministic exact prefixes."""
    import heapq
    from diffusion_coverage.solvers.history_search import SearchResult

    graph=data["graph"];meta=data["edge_meta"];routes=data["routes"];began=perf_counter();metrics=SearchMetrics();incumbent=None;first=None;serial=0;queue=[]
    root_state=initial_episode_state(graph.node_membership[start]);root=SearchLabel(start,root_state.covered,root_state.membership,0.0,0.0,1,())
    outgoing=defaultdict(list)
    for edge in graph.edges:outgoing[edge.start].append(edge)
    for values in outgoing.values():values.sort(key=lambda edge:(edge.joint_cost,edge.edge_id))
    for route in sorted(routes):heapq.heappush(queue,(0,-float(graph.weights[root.covered].sum()),0.0,serial,route,0,root));serial+=1
    pareto={};termination="queue_exhausted";archive={};total=float(graph.weights.sum())
    def pareto_admit(key,r,c,tol=1e-12):
        vals=pareto.setdefault(key,[])
        if any(rr<=r+tol and cc<=c+tol for rr,cc in vals):return False
        pareto[key]=[(rr,cc) for rr,cc in vals if not (r<=rr+tol and c<=cc+tol)]+[(r,c)];return True
    def archive_label(route,label):
        progress=min(9,int(10*float(graph.weights[label.covered].sum())/total));key=(route,progress);items=archive.setdefault(key,{})
        items[label.path]=label
        ranked_cost=sorted(items.values(),key=lambda x:(x.joint_cost,x.repeat_error,x.path))[:1]
        ranked_repeat=sorted(items.values(),key=lambda x:(x.repeat_error,x.joint_cost,x.path))[:1]
        keep={x.path:x for x in ranked_cost+ranked_repeat};archive[key]=keep
    while queue:
        elapsed=perf_counter()-began
        if elapsed>=wall_time:termination="wall_time";break
        if metrics.expanded>=expanded_limit:termination="expanded_limit";break
        *_,route,index,label=heapq.heappop(queue)
        key=(route,index,label.node,np.packbits(label.covered).tobytes(),np.packbits(label.membership).tobytes(),label.used_on_segments)
        if not pareto_admit(key,label.repeat_error,label.joint_cost):metrics.dominance_pruned+=1;continue
        archive_label(route,label)
        if incumbent is not None and (label.used_on_segments-1,label.joint_cost)>=(incumbent.used_on_segments-1,incumbent.joint_cost):continue
        miss=float(graph.weights[~label.covered].sum()/total)
        if miss<=float(config["coverage"]["missed_tolerance"])+1e-12 and label.repeat_error<=float(config["coverage"]["repeat_tolerance"])+1e-12:
            if first is None:first=elapsed
            incumbent=label;continue
        metrics.expanded+=1;sequence=routes[route];boundaries=[int(data["arc_start"][sequence[0]])]+[int(data["arc_end"][eid]) for eid in sequence];candidates=[]
        if index==0 and label.node==start and int(data["node_ports"][label.node])!=boundaries[0]:candidates.extend((edge,0) for edge in outgoing[label.node] if meta[edge.edge_id]["kind"]=="entry:"+route)
        elif index<len(sequence):candidates.extend((edge,index+1) for edge in outgoing[label.node] if int(meta[edge.edge_id]["geom_arc_id"])==int(sequence[index]) and meta[edge.edge_id]["kind"]=="source")
        if label.used_on_segments<maximum_on_segments:
            for edge in outgoing[label.node]:
                if meta[edge.edge_id]["kind"]!="off_reconfiguration":continue
                endpoint=int(data["node_ports"][edge.end])
                for next_index in range(index,len(boundaries)):
                    if boundaries[next_index]==endpoint:candidates.append((edge,next_index))
        for edge,next_index in candidates:
            metrics.generated+=1;used=label.used_on_segments+edge.summary.off_to_on_count
            if used>maximum_on_segments:metrics.segment_pruned+=1;continue
            try:state=apply_edge_summary(EpisodeState(label.covered,label.membership,label.repeat_error),edge.summary,graph.weights)
            except ValueError:continue
            if state.repeat_error>float(config["coverage"]["repeat_tolerance"])+1e-12:metrics.repeat_pruned+=1;continue
            child=SearchLabel(edge.end,state.covered,state.membership,state.repeat_error,label.joint_cost+edge.joint_cost,used,label.path+(edge.edge_id,));serial+=1
            heapq.heappush(queue,(used-1,-float(graph.weights[state.covered].sum()),child.joint_cost,serial,route,next_index,child))
    flat=[]
    for (route,progress),items in sorted(archive.items()):
        for label in sorted(items.values(),key=lambda x:(x.joint_cost,x.repeat_error,x.path)):flat.append({"route":route,"progress_bin":progress,"path":label.path})
    return SearchResult(incumbent,not queue and termination=="queue_exhausted",termination,perf_counter()-began,metrics,(),first,None),tuple(flat)


class SourceRunCache:
    def __init__(self,data:dict[str,Any],lengths:tuple[int,...],branch_cap:int,byte_limit:int):
        self.data=data;self.graph=data["graph"];self.lengths=lengths;self.branch_cap=branch_cap;self.byte_limit=byte_limit;self.cache=OrderedDict();self.bytes=0;self.omitted=0
        self.outgoing=defaultdict(list)
        for edge in self.graph.edges:self.outgoing[edge.start].append(edge)
        for values in self.outgoing.values():values.sort(key=lambda e:e.edge_id)
        self.successor={}
        for route,sequence in sorted(data["routes"].items()):
            for i,gid in enumerate(sequence[:-1]):self.successor[int(gid)]=int(sequence[i+1])
    def actions(self,node:int,metrics:AnytimeMetrics)->tuple[StructuredAction,...]:
        if node in self.cache:
            metrics.run_cache_hits+=1;value,size=self.cache.pop(node);self.cache[node]=(value,size);return value
        metrics.run_cache_misses+=1;out=[]
        for first in self.outgoing.get(node,()):
            meta=self.data["edge_meta"][first.edge_id]
            if meta["kind"]!="source":continue
            frontier=[(first,)]
            emitted=set()
            for depth in range(2,max(self.lengths)+1):
                next_frontier=[]
                for seq in frontier:
                    next_geom=self.successor.get(int(self.data["edge_meta"][seq[-1].edge_id]["geom_arc_id"]))
                    choices=[] if next_geom is None else [e for e in self.outgoing.get(seq[-1].end,()) if self.data["edge_meta"][e.edge_id]["kind"]=="source" and int(self.data["edge_meta"][e.edge_id]["geom_arc_id"])==next_geom]
                    for edge in choices:next_frontier.append(seq+(edge,))
                    if not choices and len(seq)>1 and len(seq) not in emitted:
                        out.extend(self._select([seq],len(seq)));emitted.add(len(seq))
                frontier=next_frontier
                if depth in self.lengths and frontier:
                    selected=self._select(frontier,depth);out.extend(selected);self.omitted+=max(0,len(frontier)-len(selected));emitted.add(depth)
                if not frontier:break
        unique={a.edge_ids:a for a in out};value=tuple(unique[k] for k in sorted(unique));size=sum(a.summary.footprint.nbytes+a.summary.episode_counts.nbytes+a.summary.start_membership.nbytes+a.summary.end_membership.nbytes+8*len(a.edge_ids) for a in value)
        while self.cache and self.bytes+size>self.byte_limit:
            _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
        if size<=self.byte_limit:self.cache[node]=(value,size);self.bytes+=size
        metrics.run_cache_peak_bytes=max(metrics.run_cache_peak_bytes,self.bytes);return value
    def _select(self,sequences,length):
        actions=[]
        for seq in sequences:
            summary=compose_episode_summaries((e.summary for e in seq),self.graph.weights);actions.append(StructuredAction(seq[0].start,seq[-1].end,summary,float(sum(e.joint_cost for e in seq)),tuple(e.edge_id for e in seq),"source_run"))
        actions.sort(key=lambda a:(a.end,a.summary.weighted_episode_mass,a.joint_cost,a.edge_ids));chosen=[];ends=set()
        for a in actions:
            if a.end not in ends:chosen.append(a);ends.add(a.end)
            if len(chosen)>=self.branch_cap:return chosen
        for a in actions:
            if a not in chosen:chosen.append(a)
            if len(chosen)>=self.branch_cap:break
        return chosen


def structured_anytime_search(
    data:dict[str,Any], *, start_node:int, maximum_on_segments:int,
    initial_prefixes:Iterable[Iterable[int]], validated_fallback:SearchLabel|None,
    use_source_runs:bool, config:dict[str,Any], wall_time_s:float,
    screen:Callable[[tuple[int,...]],dict[str,Any]]|None=None,
) -> AnytimeResult:
    graph=data["graph"];weights=np.asarray(graph.weights);total=float(weights.sum());began=perf_counter();deadline=began+max(0.0,wall_time_s);cfg=config["search"];metrics=AnytimeMetrics();labels={};buckets=defaultdict(list);pareto=defaultdict(list);child_refs=defaultdict(int);expanded_records=set();serial=0;next_id=0;live=set();candidates={};first_q2=None;first_q3=None;checkpoints=[];cp_index=0
    outgoing=defaultdict(list)
    for edge in graph.edges:outgoing[edge.start].append(edge)
    for values in outgoing.values():values.sort(key=lambda e:(e.joint_cost,e.edge_id))
    run_cache=SourceRunCache(data,tuple(cfg["run_lengths"]),int(cfg["run_branch_cap"]),int(cfg["run_cache_mib"])*1024**2)
    reachable_cache={}
    def history_key(label):return (label.node,np.packbits(label.covered).tobytes(),np.packbits(label.membership).tobytes(),label.used_on_segments)
    def rank(label):
        covered=float(weights[label.covered].sum());remaining=max(0.0,(1-float(config["coverage"]["missed_tolerance"]))*total-covered);return (remaining,label.repeat_error,label.joint_cost,label.serial)
    def bucket_key(label):return (label.used_on_segments,min(20,int(20*float(weights[label.covered].sum())/total)))
    def remove_pareto(label):
        key=history_key(label);pareto[key]=[i for i in pareto[key] if i!=label.label_id]
        if not pareto[key]:pareto.pop(key,None)
    def reclaim(label_id):
        """Reclaim evicted ancestry once neither OPEN nor Pareto nor a child uses it."""
        current=label_id
        while current in labels and current not in live and child_refs[current]==0:
            lab=labels[current]
            if any(current in values for values in pareto.values()):break
            parent=lab.parent_id
            labels.pop(current,None);expanded_records.discard(current);child_refs.pop(current,None)
            if parent is None:break
            child_refs[parent]=max(0,child_refs[parent]-1);current=parent
    def insert(label):
        key=history_key(label);records=[labels[i] for i in pareto[key] if i in labels]
        if any(x.repeat_error<=label.repeat_error+1e-12 and x.joint_cost<=label.joint_cost+1e-12 for x in records):metrics.dominance_pruned+=1;return False
        dominated=[x.label_id for x in records if label.repeat_error<=x.repeat_error+1e-12 and label.joint_cost<=x.joint_cost+1e-12]
        for i in dominated:
            if i in live:evict(i,heuristic=False)
            pareto[key]=[j for j in pareto[key] if j!=i]
            reclaim(i)
        labels[label.label_id]=label
        if label.parent_id is not None:child_refs[label.parent_id]+=1
        pareto[key].append(label.label_id);keyb=bucket_key(label);pool=[labels[i] for i in buckets[keyb] if i in live]+[label];pool.sort(key=rank);chosen=[];pernode=defaultdict(int)
        for item in pool:
            if pernode[item.node]<int(cfg["endpoint_diversity_limit"]):chosen.append(item);pernode[item.node]+=1
            if len(chosen)>=int(cfg["bucket_live_limit"]):break
        if len(chosen)<int(cfg["bucket_live_limit"]):
            for item in pool:
                if item not in chosen:chosen.append(item)
                if len(chosen)>=int(cfg["bucket_live_limit"]):break
        selected={x.label_id for x in chosen};old=set(buckets[keyb]);buckets[keyb]=[x.label_id for x in chosen]
        for i in old-selected:evict(i,heuristic=True)
        if label.label_id not in selected:
            remove_pareto(label);metrics.heuristic_evictions+=1;reclaim(label.label_id);return False
        label.live=True;live.add(label.label_id);metrics.peak_open=max(metrics.peak_open,len(live));metrics.peak_ancestry=max(metrics.peak_ancestry,len(labels));metrics.peak_pareto_records=max(metrics.peak_pareto_records,sum(len(v) for v in pareto.values()));return True
    def evict(label_id,heuristic):
        if label_id not in live:return
        live.discard(label_id);lab=labels[label_id];lab.live=False
        key=bucket_key(lab);buckets[key]=[i for i in buckets[key] if i!=label_id]
        if not lab.expanded:remove_pareto(lab)
        if heuristic:metrics.heuristic_evictions+=1
        reclaim(label_id)
    def add_replayed(path):
        nonlocal next_id,serial
        replay=replay_edge_sequence(graph,start_node,path);lab=_Label(next_id,replay.node,replay.covered.copy(),replay.membership.copy(),replay.repeat_error,replay.joint_cost,replay.used_on_segments,None,tuple(path),serial);next_id+=1;serial+=1;insert(lab)
    add_replayed(())
    for path in initial_prefixes:
        try:add_replayed(tuple(path))
        except (ValueError,FloatingPointError):continue
    used_cycle=list(range(1,maximum_on_segments+1));used_cursor=0
    def pop_next():
        nonlocal used_cursor
        for _ in range(len(used_cycle)):
            used=used_cycle[used_cursor%len(used_cycle)];used_cursor+=1
            for progress in range(20,-1,-1):
                key=(used,progress);ids=[i for i in buckets[key] if i in live]
                if ids:
                    best=min(ids,key=lambda i:rank(labels[i]))
                    labels[best].expanded=True
                    evict(best,heuristic=False)
                    expanded_records.add(best)
                    return labels[best]
        return None
    def path_of(label_id):
        pieces=[];cur=labels[label_id]
        while cur is not None:
            pieces.append(cur.action_edges);cur=labels.get(cur.parent_id) if cur.parent_id is not None else None
        return tuple(e for piece in reversed(pieces) for e in piece)
    def reachable(label):
        key=(label.node,maximum_on_segments-label.used_on_segments)
        if key not in reachable_cache:
            states={(label.node,label.used_on_segments)};stack=list(states);foot=np.zeros_like(label.covered)
            while stack:
                if perf_counter()>=deadline:return True
                node,used=stack.pop()
                for edge in outgoing.get(node,()):
                    nu=used+edge.summary.off_to_on_count
                    if nu>maximum_on_segments:continue
                    foot|=edge.summary.footprint;st=(edge.end,nu)
                    if st not in states:states.add(st);stack.append(st)
            reachable_cache[key]=foot
        return float(weights[label.covered|reachable_cache[key]].sum())>=(1-float(config["coverage"]["missed_tolerance"]))*total-1e-15
    def retain_candidate(label):
        nonlocal first_q2,first_q3
        path=path_of(label.label_id)
        if path in candidates:return
        elapsed=perf_counter()-began
        if first_q2 is None:first_q2=elapsed
        q2_miss=float(weights[~label.covered].sum()/total);status="SCREEN_NOT_RUN";q3m=q3r=None;screen_s=0.0
        if screen is not None and metrics.screen_calls<int(cfg["online_screen_limit"]) and perf_counter()<deadline:
            s=perf_counter();result=screen(path);screen_s=perf_counter()-s;metrics.screen_calls+=1;metrics.screen_seconds+=screen_s
            status=result.get("status","SCREEN_ERROR");q3m=result.get("E_miss");q3r=result.get("E_rep")
            if status=="Q3_PASS" and first_q3 is None:first_q3=perf_counter()-began
        candidates[path]=Candidate(label.label_id,path,label.node,label.covered.copy(),label.membership.copy(),label.repeat_error,label.joint_cost,label.used_on_segments,q2_miss,status,q3m,q3r,screen_s)
        ordered_obj=sorted(candidates.values(),key=lambda c:(c.used_on_segments-1,c.joint_cost,c.edge_ids))[:4]
        ordered_slack=sorted(candidates.values(),key=lambda c:(-(float(config["coverage"]["missed_tolerance"])-c.q2_miss+float(config["coverage"]["repeat_tolerance"])-c.repeat_error),c.used_on_segments,c.joint_cost,c.edge_ids))[:4]
        keep={c.edge_ids for c in ordered_obj+ordered_slack}
        for key in list(candidates):
            if key not in keep:candidates.pop(key)
    termination="beam_exhausted"
    while live:
        elapsed=perf_counter()-began
        while cp_index<len(cfg["checkpoints_s"]) and elapsed>=float(cfg["checkpoints_s"][cp_index]):
            checkpoints.append({"seconds":float(cfg["checkpoints_s"][cp_index]),"expanded":metrics.expanded,"candidates":len(candidates),"open":len(live)});cp_index+=1
        if perf_counter()>=deadline:termination="wall_time";break
        if metrics.expanded>=int(cfg["expanded_label_limit"]):termination="expanded_limit";break
        try:
            import resource
            peak_rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
            if peak_rss>=int(cfg["private_memory_gib"])*1024**3:termination="memory_limit";break
        except (ImportError,ValueError):
            pass
        if len(labels)>=int(cfg["retained_record_limit"]):termination="retained_record_budget";break
        label=pop_next()
        if label is None:break
        metrics.expanded+=1
        miss=float(weights[~label.covered].sum()/total)
        if miss<=float(config["coverage"]["missed_tolerance"])+1e-12 and label.repeat_error<=float(config["coverage"]["repeat_tolerance"])+1e-12:retain_candidate(label)
        if not reachable(label):metrics.reachability_pruned+=1;continue
        atomic=[StructuredAction(e.start,e.end,e.summary,e.joint_cost,(e.edge_id,),"atomic") for e in outgoing.get(label.node,())]
        actions=atomic+(list(run_cache.actions(label.node,metrics)) if use_source_runs else [])
        for action in actions:
            if perf_counter()>=deadline:termination="wall_time";break
            metrics.generated_actions+=1;metrics.atomic_equivalent_work+=len(action.edge_ids);metrics.run_actions_evaluated+=int(action.kind=="source_run")
            used=label.used_on_segments+action.summary.off_to_on_count
            if used>maximum_on_segments:metrics.segment_pruned+=1;continue
            try:state=apply_edge_summary(EpisodeState(label.covered,label.membership,label.repeat_error),action.summary,weights)
            except ValueError:continue
            if state.repeat_error>float(config["coverage"]["repeat_tolerance"])+1e-12:metrics.repeat_pruned+=1;continue
            cost=label.joint_cost+action.joint_cost
            if validated_fallback is not None and (used-1,cost)>=(validated_fallback.used_on_segments-1,validated_fallback.joint_cost):metrics.objective_pruned+=1;continue
            child=_Label(next_id,action.end,state.covered,state.membership,state.repeat_error,cost,used,label.label_id,action.edge_ids,serial);next_id+=1;serial+=1;insert(child)
        if termination=="wall_time":break
    elapsed=perf_counter()-began
    finalists=sorted((c for c in candidates.values() if c.screen_status=="Q3_PASS"),key=lambda c:(c.used_on_segments-1,c.joint_cost,c.edge_ids))[:1]
    remaining=[c for c in candidates.values() if c not in finalists and c.screen_status!="Q3_FAIL"]
    if remaining:finalists.append(sorted(remaining,key=lambda c:(-(float(config["coverage"]["missed_tolerance"])-c.q2_miss+float(config["coverage"]["repeat_tolerance"])-c.repeat_error),c.used_on_segments,c.joint_cost,c.edge_ids))[0])
    all_screened=tuple(sorted(candidates.values(),key=lambda c:(c.used_on_segments-1,c.joint_cost,c.edge_ids)))
    return AnytimeResult(tuple(finalists),all_screened,termination,elapsed,metrics,tuple(checkpoints),first_q2,first_q3,validated_fallback is not None,{"cache_entries":len(run_cache.cache),"cache_bytes":run_cache.bytes,"omitted_by_cap":run_cache.omitted})
