from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.coverage.coverage_plan import CoverageMetrics, CoveragePlan
from diffusion_coverage.coverage.evaluator import evaluate_coverage
from diffusion_coverage.coverage.objective import constrained_coverage_key
from diffusion_coverage.coverage.patterns import (
    _intrinsic_extents,
    _periodicity,
    decode_structured_parameter_controls,
    extract_structured_parameter_controls,
    map_surface_parameters,
    PatternProposal,
)
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class StructuredTeacherConfig:
    footprint_radius: float
    missed_tolerance: float
    restarts: int = 8
    steps_per_restart: int = 32
    initial_sigma_radius: float = 0.30
    final_sigma_radius: float = 0.03
    maximum_length_ratio: float = 1.15
    minimum_diversity_radius: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.footprint_radius <= 0.0:
            raise ValueError("footprint_radius must be positive")
        if not 0.0 <= self.missed_tolerance < 1.0:
            raise ValueError("missed_tolerance must lie in [0, 1)")
        if self.restarts < 1 or self.steps_per_restart < 1:
            raise ValueError("restart and step counts must be positive")
        if not 0.0 < self.final_sigma_radius <= self.initial_sigma_radius:
            raise ValueError("invalid mutation scale schedule")
        if self.maximum_length_ratio < 1.0 or self.minimum_diversity_radius < 0.0:
            raise ValueError("invalid quality or diversity threshold")


@dataclass(frozen=True)
class StructuredTeacherCandidate:
    controls: np.ndarray
    plan: CoveragePlan
    metrics: CoverageMetrics
    restart: int
    evaluations: int
    distance_from_template_radius: float

    @property
    def feasible(self) -> bool:
        return bool(self.plan.metadata["feasible"])


@dataclass(frozen=True)
class StructuredTeacherResult:
    mode_name: str
    template: StructuredTeacherCandidate
    candidates: tuple[StructuredTeacherCandidate, ...]
    evaluated_assignments: int


class MultiStartStructuredTeacher:
    """Derivative-free search for diverse, hard-feasible structured controls."""

    def __init__(self, config: StructuredTeacherConfig) -> None:
        self.config = config

    def solve(
        self,
        surface: SurfaceInstance,
        proposal: PatternProposal,
    ) -> StructuredTeacherResult:
        mode_name = proposal.name
        template_controls = extract_structured_parameter_controls(
            surface, proposal.plan.active_paths()[0], mode_name=mode_name
        )
        template = self._evaluate(
            surface, template_controls, mode_name, restart=-1, evaluations=1,
            template_controls=template_controls,
        )
        rng = np.random.default_rng(self.config.seed)
        evaluated = 1
        finals: list[StructuredTeacherCandidate] = []
        for restart in range(self.config.restarts):
            controls = self._mutate(
                surface,
                template_controls,
                rng,
                sigma_radius=self.config.initial_sigma_radius,
            )
            current = self._evaluate(
                surface, controls, mode_name, restart=restart, evaluations=1,
                template_controls=template_controls,
            )
            evaluated += 1
            local_evaluations = 1
            for step in range(self.config.steps_per_restart):
                fraction = step / max(1, self.config.steps_per_restart - 1)
                sigma = self.config.initial_sigma_radius * (
                    self.config.final_sigma_radius / self.config.initial_sigma_radius
                ) ** fraction
                proposed_controls = self._mutate(
                    surface, current.controls, rng, sigma_radius=sigma
                )
                proposed = self._evaluate(
                    surface,
                    proposed_controls,
                    mode_name,
                    restart=restart,
                    evaluations=local_evaluations + 1,
                    template_controls=template_controls,
                )
                evaluated += 1
                local_evaluations += 1
                if self._key(proposed) < self._key(current):
                    current = proposed
            finals.append(current)

        # The analytic template is a proposal, not an admission exemption. In
        # threshold-sensitive charts its float32 model-space reconstruction can
        # be infeasible even when the original dense pattern was feasible.
        accepted = [template] if template.feasible else []
        for candidate in sorted(finals, key=self._key):
            if not candidate.feasible:
                continue
            if candidate.metrics.path_length > (
                self.config.maximum_length_ratio * template.metrics.path_length
            ):
                continue
            if all(
                control_distance_radius(
                    candidate.controls,
                    other.controls,
                    surface,
                    self.config.footprint_radius,
                ) >= self.config.minimum_diversity_radius
                for other in accepted
            ):
                accepted.append(candidate)
        return StructuredTeacherResult(
            mode_name=mode_name,
            template=template,
            candidates=tuple(accepted),
            evaluated_assignments=evaluated,
        )

    def _evaluate(
        self,
        surface: SurfaceInstance,
        controls: np.ndarray,
        mode_name: str,
        *,
        restart: int,
        evaluations: int,
        template_controls: np.ndarray,
    ) -> StructuredTeacherCandidate:
        # The learned model emits float32 controls. Optimize and hard-check that
        # exact numeric object so archive metrics cannot become threshold-fragile
        # after the training-input roundtrip.
        model_controls = np.asarray(controls, dtype=np.float32).astype(np.float64)
        parameters = decode_structured_parameter_controls(
            surface,
            model_controls,
            footprint_radius=self.config.footprint_radius,
            mode_name=mode_name,
        )
        world = map_surface_parameters(surface, parameters[:, 0], parameters[:, 1])
        metrics = evaluate_coverage(
            surface,
            CoveragePlan(world),
            footprint_radius=self.config.footprint_radius,
        )
        feasible = metrics.missed_fraction <= self.config.missed_tolerance + 1e-12
        plan = CoveragePlan(
            world,
            metadata={
                "mode_name": mode_name,
                "feasible": feasible,
                "missed_tolerance": self.config.missed_tolerance,
            },
        )
        return StructuredTeacherCandidate(
            controls=model_controls,
            plan=plan,
            metrics=metrics,
            restart=restart,
            evaluations=evaluations,
            distance_from_template_radius=control_distance_radius(
                model_controls,
                template_controls,
                surface,
                self.config.footprint_radius,
            ),
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
        periodic = _periodicity(surface)
        for axis, is_periodic in enumerate(periodic):
            if not is_periodic:
                mutated[:, axis] = np.clip(mutated[:, axis], 0.0, 1.0)
        return mutated

    def _key(self, candidate: StructuredTeacherCandidate) -> tuple[int, float, float]:
        return constrained_coverage_key(
            candidate.metrics.missed_fraction,
            candidate.metrics.path_length,
            self.config.missed_tolerance,
        )


def control_distance_radius(
    left: np.ndarray,
    right: np.ndarray,
    surface: SurfaceInstance,
    footprint_radius: float,
) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if delta.shape != np.asarray(left).shape or delta.shape != np.asarray(right).shape:
        raise ValueError("control arrays must have matching shapes")
    for axis, periodic in enumerate(_periodicity(surface)):
        if periodic:
            delta[:, axis] -= np.round(np.median(delta[:, axis]))
    physical = delta * np.asarray(_intrinsic_extents(surface), dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(physical**2, axis=1))) / footprint_radius)
