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

FROZEN_ADMISSION_KEYS = (
    "characteristic_length_m",
    "sigma_safe",
    "delta_NUC",
    "q_interpolation_step_rad",
    "axis_tolerance_degrees",
    "position_tolerance_m",
    "coverage_path_sample_spacing_m",
)


def require_unchanged_admission_contract(default: dict[str, float], strong: dict[str, float]) -> None:
    for key in FROZEN_ADMISSION_KEYS:
        if key not in default or key not in strong or default[key] != strong[key]:
            raise ValueError(f"strong-search replay changed frozen admission value: {key}")


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
