"""Counterfactual experiment design and Bayesian action selection for SAGE.T."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .contracts import (
    AbstractState,
    ActionCandidate,
    RolloutPrediction,
)
from .executor import ProgramExecutor
from .posterior import ProgramParticle, ProgramPosterior


@dataclass(frozen=True)
class CandidateSequence:
    actions: tuple[ActionCandidate, ...]
    source: str = "ordinary"

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("a candidate sequence cannot be empty")
        if len(self.actions) > 8:
            raise ValueError("a candidate sequence cannot exceed eight actions")
        if any(action.action_name == "RESET" for action in self.actions):
            raise ValueError("RESET is branch control, not a candidate action")

    @property
    def key(self) -> str:
        return "->".join(action.key for action in self.actions)


@dataclass(frozen=True)
class DisagreementBreakdown:
    observational: float = 0.0
    causal: float = 0.0
    teleological: float = 0.0
    planning: float = 0.0


@dataclass(frozen=True)
class ProgramSequencePrediction:
    program_hash: str
    probability: float
    rollout: RolloutPrediction


@dataclass(frozen=True)
class SequenceAssessment:
    candidate: CandidateSequence
    utility: float
    expected_goal: float
    expected_progress: float
    terminal_risk: float
    information_gain: float
    beta: float
    disagreement: DisagreementBreakdown
    residual_mass: float
    veto: str = ""
    predictions: tuple[ProgramSequencePrediction, ...] = ()

    @property
    def first_action(self) -> ActionCandidate:
        return self.candidate.actions[0]


@dataclass(frozen=True)
class BayesianDecision:
    chosen: SequenceAssessment | None
    assessments: tuple[SequenceAssessment, ...] = ()
    normalized_entropy: float = 0.0
    reason: str = ""

    @property
    def action(self) -> ActionCandidate | None:
        return None if self.chosen is None else self.chosen.first_action


@dataclass
class CounterfactualDecisionEngine:
    """Evaluate legal action sequences against the joint posterior.

    Programs are deterministic; uncertainty lives in the posterior mixture.
    Consequently, the entropy of their predicted observation classes is the
    exact mutual information for a noiseless observation model.
    """

    executor: ProgramExecutor = field(default_factory=ProgramExecutor)
    maximum_sequences: int = 64
    maximum_particles: int = 16
    ordinary_horizon: int = 3

    def decide(
        self,
        posterior: ProgramPosterior,
        state: AbstractState,
        legal_actions: Sequence[ActionCandidate],
        *,
        memory_macros: Sequence[Sequence[ActionCandidate]] = (),
        danger_veto: Callable[[ActionCandidate], bool] | None = None,
    ) -> BayesianDecision:
        unique_actions: dict[str, ActionCandidate] = {}
        for action in legal_actions:
            unique_actions.setdefault(action.key, action)
        actions = tuple(unique_actions.values())
        if not actions:
            return BayesianDecision(
                chosen=None,
                normalized_entropy=posterior.normalized_entropy,
                reason="no_legal_action",
            )
        particles = posterior.top(self.maximum_particles)
        if not particles:
            return BayesianDecision(
                chosen=None,
                normalized_entropy=posterior.normalized_entropy,
                reason="empty_posterior",
            )
        candidates = self.generate_sequences(
            actions,
            memory_macros=memory_macros,
        )
        beta = 1.0 if posterior.normalized_entropy > 0.5 else 0.25
        assessments = tuple(
            self.assess(
                candidate,
                particles=particles,
                state=state,
                beta=beta,
                danger_veto=danger_veto,
            )
            for candidate in candidates
        )
        admissible = tuple(item for item in assessments if not item.veto)
        chosen = (
            max(
                admissible,
                key=lambda item: (
                    item.utility,
                    -item.terminal_risk,
                    -len(item.candidate.actions),
                    item.candidate.key,
                ),
            )
            if admissible
            else None
        )
        return BayesianDecision(
            chosen=chosen,
            assessments=tuple(
                sorted(
                    assessments,
                    key=lambda item: (bool(item.veto), -item.utility),
                )
            ),
            normalized_entropy=posterior.normalized_entropy,
            reason="selected" if chosen is not None else "all_vetoed",
        )

    def generate_sequences(
        self,
        legal_actions: Sequence[ActionCandidate],
        *,
        memory_macros: Sequence[Sequence[ActionCandidate]] = (),
    ) -> tuple[CandidateSequence, ...]:
        legal_by_key = {action.key: action for action in legal_actions}
        legal_names = {action.action_name for action in legal_actions}
        generic_legal_names = {
            action.action_name for action in legal_actions if not action.action_data
        }
        candidates: list[CandidateSequence] = []
        seen: set[str] = set()

        def admit(sequence: CandidateSequence) -> None:
            if len(candidates) >= self.maximum_sequences:
                return
            if any(
                action.action_name not in legal_names
                or (
                    action.key not in legal_by_key
                    and action.action_name not in generic_legal_names
                )
                for action in sequence.actions
            ):
                return
            if sequence.key in seen:
                return
            seen.add(sequence.key)
            candidates.append(sequence)

        # Memory plans are proposals, never evidence.  They are considered
        # first so the 64-sequence budget cannot silently discard them.
        for raw_macro in memory_macros:
            macro = tuple(raw_macro[:8])
            if not macro:
                continue
            normalized = tuple(legal_by_key.get(action.key, action) for action in macro)
            admit(CandidateSequence(normalized, source="memory_macro"))

        horizon = max(1, min(3, int(self.ordinary_horizon)))
        for length in range(1, horizon + 1):
            for sequence in itertools.product(legal_actions, repeat=length):
                admit(CandidateSequence(tuple(sequence)))
                if len(candidates) >= self.maximum_sequences:
                    break
            if len(candidates) >= self.maximum_sequences:
                break
        return tuple(candidates)

    def assess(
        self,
        candidate: CandidateSequence,
        *,
        particles: Sequence[ProgramParticle],
        state: AbstractState,
        beta: float,
        danger_veto: Callable[[ActionCandidate], bool] | None = None,
    ) -> SequenceAssessment:
        if danger_veto is not None and danger_veto(candidate.actions[0]):
            return SequenceAssessment(
                candidate=candidate,
                utility=float("-inf"),
                expected_goal=0.0,
                expected_progress=0.0,
                terminal_risk=1.0,
                information_gain=0.0,
                beta=beta,
                disagreement=DisagreementBreakdown(),
                residual_mass=max(
                    0.0,
                    1.0 - sum(item.probability for item in particles),
                ),
                veto="observed_danger",
            )
        top_mass = sum(item.probability for item in particles)
        if top_mass <= 0.0:
            raise ValueError("posterior particle mass must be positive")
        residual_mass = max(0.0, 1.0 - top_mass)
        predictions = []
        for particle in particles:
            particle_state = (
                state
                if particle.state is None
                else particle.state.merge_observation(state)
            )
            rollout = self.executor.rollout(
                particle.program,
                particle_state,
                candidate.actions,
                maximum_actions=8,
            )
            predictions.append(
                ProgramSequencePrediction(
                    program_hash=particle.program.canonical_hash,
                    probability=particle.probability,
                    rollout=rollout,
                )
            )
        expected_goal = (
            sum(item.probability * _rollout_goal(item.rollout) for item in predictions)
            / top_mass
        )
        expected_progress = (
            sum(
                item.probability * _rollout_progress(item.rollout)
                for item in predictions
            )
            / top_mass
        )
        terminal_risk = (
            sum(
                item.probability * _rollout_terminal(item.rollout)
                for item in predictions
            )
            / top_mass
        )
        information_gain = _weighted_entropy_with_residual(
            (
                (
                    item.probability,
                    _rollout_signature(item.rollout),
                )
                for item in predictions
            ),
            residual_mass,
        )
        disagreement = DisagreementBreakdown(
            observational=_factor_entropy(predictions, "observational", residual_mass),
            causal=_factor_entropy(predictions, "causal", residual_mass),
            teleological=_factor_entropy(predictions, "teleological", residual_mass),
            planning=_factor_entropy(predictions, "planning", residual_mass),
        )
        utility = (
            2.0 * expected_goal
            + expected_progress
            + float(beta) * information_gain
            - 5.0 * terminal_risk
            - 0.02 * len(candidate.actions)
        )
        return SequenceAssessment(
            candidate=candidate,
            utility=utility,
            expected_goal=expected_goal,
            expected_progress=expected_progress,
            terminal_risk=terminal_risk,
            information_gain=information_gain,
            beta=float(beta),
            disagreement=disagreement,
            residual_mass=residual_mass,
            predictions=tuple(predictions),
        )


def _rollout_goal(rollout: RolloutPrediction) -> float:
    known = [
        float(packet.goal_probability)
        for packet in rollout.packets
        if packet.goal_probability is not None
    ]
    return known[-1] if known else 0.0


def _rollout_progress(rollout: RolloutPrediction) -> float:
    return sum(
        float(packet.progress_mean)
        for packet in rollout.packets
        if packet.progress_mean is not None
    )


def _rollout_terminal(rollout: RolloutPrediction) -> float:
    probabilities = [
        float(packet.terminal_probability)
        for packet in rollout.packets
        if packet.terminal_probability is not None
    ]
    if not probabilities:
        return 0.0
    survival = math.prod(1.0 - value for value in probabilities)
    return 1.0 - survival


def _rollout_signature(rollout: RolloutPrediction) -> tuple[object, ...]:
    return tuple(packet.full_signature for packet in rollout.packets)


def _factor_entropy(
    predictions: Sequence[ProgramSequencePrediction],
    factor: str,
    residual_mass: float,
) -> float:
    weighted_signatures = []
    for item in predictions:
        packets = item.rollout.packets
        if factor == "observational":
            signature = tuple(
                (
                    packet.channel_signature("objects"),
                    packet.channel_signature("relations"),
                    packet.channel_signature("topology"),
                )
                for packet in packets
            )
        elif factor == "causal":
            signature = tuple(
                (
                    packet.channel_signature("objects"),
                    packet.channel_signature("relations"),
                    packet.channel_signature("topology"),
                    packet.channel_signature("progress"),
                )
                for packet in packets
            )
        elif factor == "teleological":
            signature = tuple(
                (
                    packet.channel_signature("progress"),
                    packet.channel_signature("terminal"),
                    packet.channel_signature("goal"),
                )
                for packet in packets
            )
        elif factor == "planning":
            signature = (
                item.rollout.final_packet.full_signature,
                item.rollout.final_state.signature,
            )
        else:
            raise ValueError(f"unknown disagreement factor: {factor}")
        weighted_signatures.append((item.probability, signature))
    return _weighted_entropy_with_residual(
        weighted_signatures,
        residual_mass,
    )


def _weighted_entropy(
    weighted_signatures: Iterable[tuple[float, object]],
) -> float:
    classes: dict[object, float] = {}
    for probability, signature in weighted_signatures:
        classes[signature] = classes.get(signature, 0.0) + float(probability)
    return -sum(
        mass * math.log(max(mass, 1e-300)) for mass in classes.values() if mass > 0.0
    )


def _weighted_entropy_with_residual(
    weighted_signatures: Iterable[tuple[float, object]],
    residual_mass: float,
) -> float:
    classes: dict[object, float] = {}
    for probability, signature in weighted_signatures:
        classes[signature] = classes.get(signature, 0.0) + float(probability)
    if residual_mass > 0.0:
        # Keep the omitted tail as one aggregate particle.  Assigning it to the
        # largest evaluated outcome yields a conservative lower bound on
        # information gain without evaluating beyond the 16-particle budget.
        aggregate = max(classes, key=classes.get, default=("residual",))
        classes[aggregate] = classes.get(aggregate, 0.0) + residual_mass
    return _weighted_entropy((mass, signature) for signature, mass in classes.items())


__all__ = [
    "BayesianDecision",
    "CandidateSequence",
    "CounterfactualDecisionEngine",
    "DisagreementBreakdown",
    "ProgramSequencePrediction",
    "SequenceAssessment",
]
