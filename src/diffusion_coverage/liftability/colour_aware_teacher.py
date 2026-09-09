from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.coverage import (
    CoverageMetrics,
    CoveragePlan,
    PatternProposal,
    decode_structured_parameter_controls,
    evaluate_coverage,
    extract_structured_parameter_controls,
    map_surface_parameters,
)
from diffusion_coverage.coverage.patterns import _intrinsic_extents, _periodicity
from diffusion_coverage.liftability.synthetic_colours import (
    ColourLiftResult,
    SyntheticColourField,
    evaluate_colour_lift,
)
from diffusion_coverage.surface import SurfaceInstance


@dataclass(frozen=True)
class ColourAwareTeacherConfig:
    footprint_radius: float
    missed_tolerance: float
    max_segments: int
    restarts: int = 4
    steps_per_restart: int = 12
    initial_sigma_radius: float = 0.30
    final_sigma_radius: float = 0.03
    maximum_parameter_step: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.footprint_radius <= 0.0:
            raise ValueError("footprint_radius must be positive")
        if not 0.0 <= self.missed_tolerance < 1.0:
            raise ValueError("missed_tolerance must lie in [0, 1)")
        if self.max_segments < 1 or self.restarts < 1 or self.steps_per_restart < 1:
            raise ValueError("segment, restart, and step counts must be positive")
        if not 0.0 < self.final_sigma_radius <= self.initial_sigma_radius:
            raise ValueError("invalid mutation scale schedule")
        if self.maximum_parameter_step <= 0.0:
            raise ValueError("maximum_parameter_step must be positive")


@dataclass(frozen=True)
class ColourAwareTeacherCandidate:
    controls: np.ndarray
    plan: CoveragePlan
    coverage: CoverageMetrics
    lift: ColourLiftResult
    restart: int
    evaluations: int

    @property
    def coverage_feasible(self) -> bool:
        return bool(self.plan.metadata["coverage_feasible"])

    @property
    def lift_feasible(self) -> bool:
        return bool(self.plan.metadata["lift_feasible"])

    @property
    def feasible(self) -> bool:
        return self.coverage_feasible and self.lift_feasible


@dataclass(frozen=True)
class ColourAwareTeacherResult:
    mode_name: str
    template: ColourAwareTeacherCandidate
    best: ColourAwareTeacherCandidate
    restart_finals: tuple[ColourAwareTeacherCandidate, ...]
    evaluated_assignments: int


def colour_aware_objective(
    *,
    missed_fraction: float,
    min_segments: int | None,
    path_length: float,
    missed_tolerance: float,
    max_segments: int,
) -> tuple[int, float, int, int, float]:
    """Lexicographic hard-coverage, segment-budget, then length objective."""

    coverage_excess = max(0.0, float(missed_fraction) - float(missed_tolerance))
    if coverage_excess > 1e-12:
        return (1, coverage_excess, 1, 10**9, float(path_length))
    if min_segments is None:
        return (0, 0.0, 1, 10**9, float(path_length))
    segment_excess = max(0, int(min_segments) - int(max_segments))
    return (
        0,
        0.0,
        int(segment_excess > 0),
        segment_excess,
        float(path_length),
    )


class ColourAwareStructuredTeacher:
    """Matched-budget derivative-free search with exact colour completion in the loop."""

    def __init__(self, config: ColourAwareTeacherConfig) -> None:
        self.config = config

    def solve(
        self,
        surface: SurfaceInstance,
        proposal: PatternProposal,
        field: SyntheticColourField,
    ) -> ColourAwareTeacherResult:
        mode_name = proposal.name
        template_controls = extract_structured_parameter_controls(
            surface, proposal.plan.active_paths()[0], mode_name=mode_name
        )
        template = self._evaluate(
            surface,
            template_controls,
            mode_name,
            field,
            restart=-1,
            evaluations=1,
        )
        best = template
        rng = np.random.default_rng(self.config.seed)
        evaluated = 1
        finals = []
        for restart in range(self.config.restarts):
            controls = self._mutate(
                surface,
                template_controls,
                rng,
                sigma_radius=self.config.initial_sigma_radius,
            )
            current = self._evaluate(
                surface,
                controls,
                mode_name,
                field,
                restart=restart,
                evaluations=1,
            )
            evaluated += 1
            local_evaluations = 1
            for step in range(self.config.steps_per_restart):
                fraction = step / max(1, self.config.steps_per_restart - 1)
                sigma = self.config.initial_sigma_radius * (
                    self.config.final_sigma_radius / self.config.initial_sigma_radius
                ) ** fraction
                proposed = self._evaluate(
                    surface,
                    self._mutate(surface, current.controls, rng, sigma_radius=sigma),
                    mode_name,
                    field,
                    restart=restart,
                    evaluations=local_evaluations + 1,
                )
                evaluated += 1
                local_evaluations += 1
                if self._key(proposed) < self._key(current):
                    current = proposed
            finals.append(current)
            if self._key(current) < self._key(best):
                best = current
        return ColourAwareTeacherResult(
            mode_name=mode_name,
            template=template,
            best=best,
            restart_finals=tuple(finals),
            evaluated_assignments=evaluated,
        )

    def _evaluate(
        self,
        surface: SurfaceInstance,
        controls: np.ndarray,
        mode_name: str,
        field: SyntheticColourField,
        *,
        restart: int,
        evaluations: int,
    ) -> ColourAwareTeacherCandidate:
        model_controls = np.asarray(controls, dtype=np.float32).astype(np.float64)
        parameters = decode_structured_parameter_controls(
            surface,
            model_controls,
            footprint_radius=self.config.footprint_radius,
            mode_name=mode_name,
        )
        world = map_surface_parameters(surface, parameters[:, 0], parameters[:, 1])
        raw_plan = CoveragePlan(world)
        coverage = evaluate_coverage(
            surface,
            raw_plan,
            footprint_radius=self.config.footprint_radius,
        )
        lift = evaluate_colour_lift(
            surface,
            raw_plan,
            field,
            maximum_parameter_step=self.config.maximum_parameter_step,
        )
        coverage_feasible = coverage.missed_fraction <= self.config.missed_tolerance + 1e-12
        lift_feasible = lift.within_budget(self.config.max_segments)
        plan = CoveragePlan(
            world,
            metadata={
                "mode_name": mode_name,
                "coverage_feasible": coverage_feasible,
                "lift_feasible": lift_feasible,
                "max_segments": self.config.max_segments,
            },
        )
        return ColourAwareTeacherCandidate(
            controls=model_controls,
            plan=plan,
            coverage=coverage,
            lift=lift,
            restart=restart,
            evaluations=evaluations,
        )

    def _mutate(
        self,
        surface: SurfaceInstance,
        controls: np.ndarray,
        rng: np.random.Generator,
        *,
        sigma_radius: float,
    ) -> np.ndarray:
        extents = np.asarray(_intrinsic_extents(surface), dtype=np.float64)
        scale = sigma_radius * self.config.footprint_radius / extents
        noise = rng.normal(size=np.asarray(controls).shape)
        if len(noise) >= 3:
            noise[1:-1] = 0.25 * noise[:-2] + 0.5 * noise[1:-1] + 0.25 * noise[2:]
        mutated = np.asarray(controls, dtype=np.float64) + noise * scale
        for axis, periodic in enumerate(_periodicity(surface)):
            if not periodic:
                mutated[:, axis] = np.clip(mutated[:, axis], 0.0, 1.0)
        return mutated

    def _key(self, candidate: ColourAwareTeacherCandidate) -> tuple[int, float, int, int, float]:
        return colour_aware_objective(
            missed_fraction=candidate.coverage.missed_fraction,
            min_segments=candidate.lift.min_segments,
            path_length=candidate.coverage.path_length,
            missed_tolerance=self.config.missed_tolerance,
            max_segments=self.config.max_segments,
        )
