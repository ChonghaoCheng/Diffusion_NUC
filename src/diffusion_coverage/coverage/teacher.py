from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoverageMetrics, CoveragePlan
from diffusion_coverage.coverage.evaluator import evaluate_coverage
from diffusion_coverage.coverage.objective import constrained_coverage_key
from diffusion_coverage.coverage.patterns import PatternProposal, generate_pattern_proposals, paths_to_plan
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class TeacherPlannerConfig:
    footprint_radius: float
    max_segments: int = 1
    missed_tolerance: float = 0.02
    overlap: float = 0.7
    waypoint_spacing: float | None = None
    refinement_iterations: int = 12
    perturbation_scale: float = 0.35
    smoothness_weight: float = 0.01
    max_candidates: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.footprint_radius <= 0.0:
            raise ValueError("footprint_radius must be positive")
        if self.max_segments < 1:
            raise ValueError("max_segments must be positive")
        if not 0.0 <= self.missed_tolerance < 1.0:
            raise ValueError("missed_tolerance must lie in [0, 1)")
        if not 0.0 < self.overlap <= 1.0:
            raise ValueError("overlap must lie in (0, 1]")
        if self.refinement_iterations < 0 or self.max_candidates < 1:
            raise ValueError("refinement_iterations must be nonnegative and max_candidates positive")


@dataclass(frozen=True)
class TeacherCandidate:
    plan: CoveragePlan
    metrics: CoverageMetrics
    smoothness_cost: float
    proposal_name: str
    refinement_steps: int

    @property
    def feasible(self) -> bool:
        tolerance = float(self.plan.metadata.get("missed_tolerance", 0.0))
        return self.metrics.missed_fraction <= tolerance + 1e-12


@dataclass(frozen=True)
class TeacherResult:
    candidates: tuple[TeacherCandidate, ...]
    initial_candidates: tuple[TeacherCandidate, ...]
    proposal_time: float
    refinement_time: float
    total_solve_time: float
    evaluated_plans: int
    metadata: dict[str, int | float | str] = field(default_factory=dict)

    @property
    def best(self) -> TeacherCandidate:
        if not self.candidates:
            raise RuntimeError("teacher result has no candidates")
        return self.candidates[0]

    @property
    def feasible_candidates(self) -> tuple[TeacherCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.feasible)


class ClassicalTeacherPlanner:
    """Multi-start pattern teacher with constrained stochastic local refinement."""

    def __init__(self, config: TeacherPlannerConfig) -> None:
        self.config = config

    def solve(self, surface: SurfaceInstance) -> TeacherResult:
        solve_start = perf_counter()
        spacing = self.config.waypoint_spacing or 0.75 * self.config.footprint_radius
        proposal_start = perf_counter()
        proposals = generate_pattern_proposals(
            surface,
            footprint_radius=self.config.footprint_radius,
            max_segments=self.config.max_segments,
            overlap=self.config.overlap,
            waypoint_spacing=spacing,
        )
        proposal_time = perf_counter() - proposal_start

        rng = np.random.default_rng(self.config.seed)
        evaluated_plans = 0
        candidates: list[TeacherCandidate] = []
        initial_candidates: list[TeacherCandidate] = []
        refinement_start = perf_counter()
        for proposal in proposals:
            initial, count = self._evaluate(surface, proposal.plan, proposal.name, refinement_steps=0)
            evaluated_plans += count
            initial_candidates.append(initial)
            refined, count = self._refine(surface, initial, rng)
            evaluated_plans += count
            candidates.extend((initial, refined))
        refinement_time = perf_counter() - refinement_start

        candidates.sort(key=self._objective)
        unique: list[TeacherCandidate] = []
        fingerprints: set[tuple[int, int, int]] = set()
        for candidate in candidates:
            fingerprint = (
                int(round(candidate.metrics.missed_fraction * 1e8)),
                int(round(candidate.metrics.path_length * 1e8)),
                sum(len(path) for path in candidate.plan.active_paths()),
            )
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            unique.append(candidate)
            if len(unique) == self.config.max_candidates:
                break
        return TeacherResult(
            candidates=tuple(unique),
            initial_candidates=tuple(initial_candidates),
            proposal_time=proposal_time,
            refinement_time=refinement_time,
            total_solve_time=perf_counter() - solve_start,
            evaluated_plans=evaluated_plans,
            metadata={"num_initial_proposals": len(proposals), "geodesic_backend": "mesh_edge_dijkstra"},
        )

    def _evaluate(
        self,
        surface: SurfaceInstance,
        plan: CoveragePlan,
        proposal_name: str,
        *,
        refinement_steps: int,
    ) -> tuple[TeacherCandidate, int]:
        metadata = {**plan.metadata, "missed_tolerance": self.config.missed_tolerance}
        annotated_plan = CoveragePlan(plan.waypoints, plan.segment_mask, plan.waypoint_mask, metadata=metadata)
        metrics = evaluate_coverage(
            surface,
            annotated_plan,
            footprint_radius=self.config.footprint_radius,
            path_sample_spacing=0.5 * self.config.footprint_radius,
        )
        return (
            TeacherCandidate(
                plan=annotated_plan,
                metrics=metrics,
                smoothness_cost=plan_smoothness(annotated_plan),
                proposal_name=proposal_name,
                refinement_steps=refinement_steps,
            ),
            1,
        )

    def _refine(
        self,
        surface: SurfaceInstance,
        initial: TeacherCandidate,
        rng: np.random.Generator,
    ) -> tuple[TeacherCandidate, int]:
        current = initial
        evaluated = 0
        accepted_steps = 0
        for iteration in range(self.config.refinement_iterations):
            if rng.random() < 0.7:
                proposal_plan = _delete_random_waypoint(current.plan, rng)
            else:
                scale = self.config.perturbation_scale * self.config.footprint_radius
                scale *= 1.0 - 0.75 * iteration / max(1, self.config.refinement_iterations)
                proposal_plan = _perturb_random_waypoint(surface, current.plan, rng, scale)
            if proposal_plan is None:
                continue
            proposal, count = self._evaluate(
                surface,
                proposal_plan,
                current.proposal_name,
                refinement_steps=accepted_steps + 1,
            )
            evaluated += count
            if self._objective(proposal) < self._objective(current):
                current = proposal
                accepted_steps += 1
        return current, evaluated

    def _objective(self, candidate: TeacherCandidate) -> tuple[int, float, float]:
        return constrained_coverage_key(
            candidate.metrics.missed_fraction,
            candidate.metrics.path_length,
            self.config.missed_tolerance,
            secondary_cost=self.config.smoothness_weight * candidate.smoothness_cost,
        )


def plan_smoothness(plan: CoveragePlan) -> float:
    cost = 0.0
    for path in plan.active_paths():
        if len(path) >= 3:
            second_difference = path[2:] - 2.0 * path[1:-1] + path[:-2]
            cost += float(np.sum(second_difference * second_difference))
    return cost


def _delete_random_waypoint(plan: CoveragePlan, rng: np.random.Generator) -> CoveragePlan | None:
    paths = plan.active_paths()
    eligible = [index for index, path in enumerate(paths) if len(path) > 2]
    if not eligible:
        return None
    path_index = int(rng.choice(eligible))
    delete_index = int(rng.integers(1, len(paths[path_index]) - 1)) if len(paths[path_index]) > 3 else 1
    paths[path_index] = np.delete(paths[path_index], delete_index, axis=0)
    return paths_to_plan(paths)


def _perturb_random_waypoint(
    surface: SurfaceInstance,
    plan: CoveragePlan,
    rng: np.random.Generator,
    scale: float,
) -> CoveragePlan:
    paths = plan.active_paths()
    path_index = int(rng.integers(len(paths)))
    waypoint_index = int(rng.integers(len(paths[path_index])))
    point = paths[path_index][waypoint_index]
    projection = project_points(surface, point)
    normal = surface.face_normals[projection.face_indices[0]]
    direction = rng.normal(size=3)
    tangent = direction - normal * float(np.dot(direction, normal))
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm > 1e-12:
        perturbed = point + scale * tangent / tangent_norm
        paths[path_index][waypoint_index] = project_points(surface, perturbed).points[0]
    return paths_to_plan(paths)
