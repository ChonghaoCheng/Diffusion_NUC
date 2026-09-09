from __future__ import annotations

from dataclasses import asdict
from typing import Any

from diffusion_coverage.nuc.robot_lift import NUCContinuationLayerTrace, NUCLiftResult


FAILURE_CATEGORIES = (
    "pose_candidate_empty",
    "safety_filter_exhaustion",
    "transition_graph_empty",
    "beam_or_search_exhaustion",
    "unresolved_continuation_failure",
)


def classify_continuation_failure(
    trace: NUCContinuationLayerTrace,
    *,
    strong_search_recovered: bool = False,
) -> str:
    if strong_search_recovered:
        return "beam_or_search_exhaustion"
    if trace.candidate_count_before_safety == 0:
        return "pose_candidate_empty"
    if (
        trace.candidate_count_before_safety is not None
        and trace.candidate_count_before_safety > 0
        and trace.candidate_count_after_sigma == 0
    ):
        return "safety_filter_exhaustion"
    if trace.candidate_count_after_sigma > 0 and trace.propagated_incoming_edges > 0 and trace.valid_outgoing_edges == 0:
        return "transition_graph_empty"
    return "unresolved_continuation_failure"


def lift_trace_rows(lift: NUCLiftResult, identity: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**identity, **asdict(item)} for item in lift.metadata.get("layer_trace", ())]
