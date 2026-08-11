"""Lexicographic causal action selection using the common posterior."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .contracts import ActionProgram, CausalState, PredictedTrace
from .executor import CausalExecutor
from .posterior import CausalPosterior

SOURCE_TIERS = {
    "exact_route": 0,
    "protected_route": 0,
    "progressive_route": 1,
    "exact_frontier": 1,
    "causal_probe": 2,
    "frontier": 3,
    "multiform": 3,
    "generic": 3,
}


@dataclass(frozen=True)
class ProgramActionPrediction:
    program_hash: str
    probability: float
    trace: PredictedTrace


@dataclass(frozen=True)
class CausalActionAssessment:
    action_program: ActionProgram
    tier: int
    utility: float
    expected_goal: float
    expected_progress: float
    terminal_risk: float
    information_gain: float
    residual_mass: float
    veto: str = ""
    predictions: tuple[ProgramActionPrediction, ...] = ()


@dataclass(frozen=True)
class CausalDecision:
    chosen: CausalActionAssessment | None
    assessments: tuple[CausalActionAssessment, ...]
    reason: str


class CausalDecisionEngine:
    def __init__(
        self,
        *,
        executor: CausalExecutor,
        maximum_particles: int = 16,
        maximum_terminal_probe_risk: float = 0.05,
        information_gain_scale: float = 1.0,
    ) -> None:
        self.executor = executor
        self.maximum_particles = max(1, int(maximum_particles))
        self.maximum_terminal_probe_risk = min(
            1.0, max(0.0, float(maximum_terminal_probe_risk))
        )
        self.information_gain_scale = max(0.0, float(information_gain_scale))

    def decide(
        self,
        posterior: CausalPosterior,
        state: CausalState,
        candidates: Sequence[ActionProgram],
        *,
        danger_veto: Callable[[ActionProgram], bool] | None = None,
    ) -> CausalDecision:
        particles = posterior.top(self.maximum_particles)
        if not particles:
            return CausalDecision(None, (), "empty_posterior")
        unique: dict[str, ActionProgram] = {}
        for candidate in candidates:
            previous = unique.get(candidate.key)
            if previous is None or SOURCE_TIERS.get(
                candidate.source, 3
            ) < SOURCE_TIERS.get(previous.source, 3):
                unique[candidate.key] = candidate
        if not unique:
            return CausalDecision(None, (), "no_candidate")
        beta = 1.0 if posterior.normalized_entropy > 0.5 else 0.25
        assessments = tuple(
            self._assess(
                posterior,
                state,
                candidate,
                beta=beta,
                danger_veto=danger_veto,
            )
            for candidate in unique.values()
        )
        admissible = [assessment for assessment in assessments if not assessment.veto]
        if not admissible:
            return CausalDecision(None, assessments, "all_candidates_vetoed")
        best_tier = min(assessment.tier for assessment in admissible)
        chosen = max(
            (assessment for assessment in admissible if assessment.tier == best_tier),
            key=lambda assessment: (assessment.utility, -len(assessment.action_program.actions)),
        )
        return CausalDecision(chosen, assessments, "lexicographic_causal_choice")

    def _assess(
        self,
        posterior: CausalPosterior,
        state: CausalState,
        candidate: ActionProgram,
        *,
        beta: float,
        danger_veto: Callable[[ActionProgram], bool] | None,
    ) -> CausalActionAssessment:
        particles = posterior.top(self.maximum_particles)
        top_mass = sum(particle.probability for particle in particles)
        residual_mass = max(0.0, 1.0 - top_mass)
        predictions = tuple(
            ProgramActionPrediction(
                program_hash=particle.program.canonical_hash,
                probability=particle.probability,
                trace=self.executor.rollout(
                    particle.program,
                    state,
                    candidate,
                    horizon=len(candidate.actions),
                ),
            )
            for particle in particles
        )
        denominator = top_mass if top_mass > 0.0 else 1.0
        expected_goal = sum(
            item.probability * item.trace.final_prediction.goal_probability
            for item in predictions
        ) / denominator
        expected_progress = sum(
            item.probability
            * sum(prediction.progress_probability for prediction in item.trace.predictions)
            for item in predictions
        ) / denominator
        terminal_risk = sum(
            item.probability * _trace_terminal_probability(item.trace)
            for item in predictions
        ) / denominator
        information_gain = _signature_entropy(predictions, residual_mass)
        utility = (
            2.0 * expected_goal
            + expected_progress
            + beta * self.information_gain_scale * information_gain
            - 5.0 * terminal_risk
            - 0.02 * len(candidate.actions)
        )
        tier = SOURCE_TIERS.get(candidate.source, 3)
        veto = ""
        if danger_veto is not None and danger_veto(candidate):
            veto = "external_danger_veto"
        elif tier == 2 and terminal_risk > self.maximum_terminal_probe_risk:
            veto = "causal_probe_terminal_risk"
        return CausalActionAssessment(
            action_program=candidate,
            tier=tier,
            utility=utility,
            expected_goal=expected_goal,
            expected_progress=expected_progress,
            terminal_risk=terminal_risk,
            information_gain=information_gain,
            residual_mass=residual_mass,
            veto=veto,
            predictions=predictions,
        )


def _trace_terminal_probability(trace: PredictedTrace) -> float:
    survival = math.prod(
        1.0 - prediction.terminal_probability for prediction in trace.predictions
    )
    return 1.0 - survival


def _signature_entropy(
    predictions: Sequence[ProgramActionPrediction], residual_mass: float
) -> float:
    classes: dict[tuple[object, ...], float] = {}
    for prediction in predictions:
        signature = tuple(
            packet.structured_signature for packet in prediction.trace.predictions
        )
        classes[signature] = classes.get(signature, 0.0) + prediction.probability
    if residual_mass > 0.0:
        classes[("__residual__",)] = residual_mass
    return -sum(mass * math.log(mass) for mass in classes.values() if mass > 0.0)


__all__ = [
    "SOURCE_TIERS",
    "CausalActionAssessment",
    "CausalDecision",
    "CausalDecisionEngine",
    "ProgramActionPrediction",
]
