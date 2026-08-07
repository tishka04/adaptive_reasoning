"""Bayesian particle posterior over complete executable SAGE.T programs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import (
    AbstractState,
    JointProgramHypothesis,
    ObservedTransition,
    PredictionPacket,
)
from .executor import PREDICTED_FALSE, ProgramExecutor
from .synthesis import AssembledProgram, ProgramMutator

DEFAULT_CHANNEL_WEIGHTS: Mapping[str, float] = {
    "objects": 1.0,
    "relations": 1.0,
    "topology": 1.0,
    "progress": 2.0,
    "terminal": 4.0,
    "goal": 2.0,
}


@dataclass(frozen=True)
class ProgramParticle:
    program: JointProgramHypothesis
    log_prior: float
    log_weight: float
    state: AbstractState | None = None
    latest_log_likelihood: float = 0.0
    observations: int = 0

    @property
    def probability(self) -> float:
        return math.exp(self.log_weight)


class ProgramPosterior:
    """Normalized log-space posterior with replay-validated local repair."""

    def __init__(
        self,
        *,
        executor: ProgramExecutor | None = None,
        mutator: ProgramMutator | None = None,
        maximum_particles: int = 64,
        channel_weights: Mapping[str, float] | None = None,
        unknown_coverage_penalty: float = 0.75,
        repair_ess_threshold: float = 2.0,
        repair_log_likelihood_threshold: float = -8.0,
    ) -> None:
        self.executor = executor or ProgramExecutor()
        self.mutator = mutator or ProgramMutator()
        self.maximum_particles = max(1, int(maximum_particles))
        self.channel_weights = dict(channel_weights or DEFAULT_CHANNEL_WEIGHTS)
        self.unknown_coverage_penalty = max(
            0.0,
            float(unknown_coverage_penalty),
        )
        self.repair_ess_threshold = max(1.0, float(repair_ess_threshold))
        self.repair_log_likelihood_threshold = float(repair_log_likelihood_threshold)
        self._particles: list[ProgramParticle] = []
        self._history: list[ObservedTransition] = []
        self._branch_index = 0
        self._repairs_attempted = 0
        self._repairs_admitted = 0

    @property
    def particles(self) -> tuple[ProgramParticle, ...]:
        return tuple(self._particles)

    @property
    def history(self) -> tuple[ObservedTransition, ...]:
        return tuple(self._history)

    def seed(
        self,
        programs: Sequence[AssembledProgram | JointProgramHypothesis],
        *,
        initial_state: AbstractState | None = None,
    ) -> None:
        seeded = []
        for item in programs:
            if isinstance(item, AssembledProgram):
                program = item.program
                prior = float(item.prior_logprob)
            else:
                program = item
                prior = _program_prior(program)
            seeded.append(
                ProgramParticle(
                    program=program,
                    log_prior=prior,
                    log_weight=prior,
                    state=initial_state,
                )
            )
        if self._history:
            seeded = [
                self._replay_particle(
                    replace(
                        particle,
                        state=None,
                    )
                )
                for particle in seeded
            ]
        self._particles = self._deduplicate(seeded)
        self._normalize()

    def add_programs(
        self,
        programs: Sequence[AssembledProgram | JointProgramHypothesis],
        *,
        initial_state: AbstractState | None = None,
    ) -> None:
        candidates = list(self._particles)
        existing = {particle.program.canonical_hash for particle in candidates}
        for item in programs:
            if isinstance(item, AssembledProgram):
                program = item.program
                prior = float(item.prior_logprob)
            else:
                program = item
                prior = _program_prior(program)
            if program.canonical_hash in existing:
                continue
            particle = ProgramParticle(
                program=program,
                log_prior=prior,
                log_weight=prior,
                state=initial_state,
            )
            candidates.append(particle)
            existing.add(program.canonical_hash)
        if self._history:
            # Existing particles are normalized posterior weights whereas a new
            # particle starts at its absolute prior.  Replay every candidate so
            # that pruning compares like with like.
            candidates = [
                self._replay_particle(
                    replace(
                        particle,
                        log_weight=particle.log_prior,
                        state=None,
                        latest_log_likelihood=0.0,
                        observations=0,
                    )
                )
                for particle in candidates
            ]
        self._particles = self._deduplicate(candidates)
        self._normalize()

    def observe(
        self,
        evidence: ObservedTransition,
        *,
        allow_repair: bool = True,
    ) -> None:
        if evidence.reset:
            return
        if not self._particles:
            self._history.append(evidence)
            return
        updated = []
        for particle in self._particles:
            public_before = evidence.state_before
            start = (
                public_before
                if particle.state is None
                else particle.state.merge_observation(public_before)
            )
            prediction = self.executor.step(
                particle.program,
                start,
                evidence.action,
            )
            likelihood = packet_log_likelihood(
                prediction,
                evidence.observation,
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            predicted_state = prediction.state_after or start
            next_state = predicted_state.merge_observation(evidence.state_after)
            updated.append(
                replace(
                    particle,
                    log_weight=particle.log_weight + likelihood,
                    state=next_state,
                    latest_log_likelihood=likelihood,
                    observations=particle.observations + 1,
                )
            )
        self._history.append(evidence)
        self._particles = self._deduplicate(updated)
        self._normalize()
        if allow_repair and self._needs_repair():
            self.repair(evidence)

    def repair(
        self,
        evidence: ObservedTransition | None = None,
    ) -> tuple[ProgramParticle, ...]:
        if not self._particles or not self._history:
            return ()
        target = evidence or self._history[-1]
        parents = sorted(
            self._particles,
            key=lambda particle: (
                particle.latest_log_likelihood,
                particle.log_weight,
            ),
            reverse=True,
        )[:4]
        self._repairs_attempted += len(parents)
        children = []
        existing = {particle.program.canonical_hash for particle in self._particles}
        for parent in parents:
            for program in self.mutator.mutate(parent.program, target):
                if program.canonical_hash in existing:
                    continue
                prior = parent.log_prior - max(
                    1.0,
                    float(program.edit_distance - parent.program.edit_distance),
                )
                child = self._replay_particle(
                    ProgramParticle(
                        program=program,
                        log_prior=prior,
                        log_weight=prior,
                    )
                )
                children.append(child)
                existing.add(program.canonical_hash)
        if not children:
            return ()
        self._repairs_admitted += len(children)
        # A repaired child receives a fresh edit-distance prior and must earn
        # all of its evidence again.  Replay the parents as well because their
        # current weights are normalized, not absolute joint scores.
        replayed_parents = [
            self._replay_particle(
                replace(
                    particle,
                    log_weight=particle.log_prior,
                    state=None,
                    latest_log_likelihood=0.0,
                    observations=0,
                )
            )
            for particle in self._particles
        ]
        self._particles = self._deduplicate([*replayed_parents, *children])
        self._normalize()
        return tuple(children)

    def marginalize(
        self,
        predictions: Mapping[str, PredictionPacket],
    ) -> PredictionPacket:
        weighted = [
            (particle.probability, predictions[particle.program.canonical_hash])
            for particle in self._particles
            if particle.program.canonical_hash in predictions
        ]
        if not weighted:
            return PredictionPacket()
        mass = sum(weight for weight, _ in weighted)
        if mass <= 0.0:
            return PredictionPacket()
        weighted = [(weight / mass, packet) for weight, packet in weighted]
        objects = _marginal_events(weighted, "object_deltas")
        relations = _marginal_events(weighted, "relation_deltas")
        topology = _marginal_events(weighted, "topology_deltas")
        known_channels = {
            channel
            for channel in (
                "objects",
                "relations",
                "topology",
                "progress",
                "terminal",
                "goal",
            )
            if sum(
                weight
                for weight, packet in weighted
                if channel in packet.known_channels
            )
            >= 0.5
        }
        return PredictionPacket(
            object_deltas=objects,
            relation_deltas=relations,
            topology_deltas=topology,
            progress_mean=_weighted_optional(
                weighted,
                "progress_mean",
            ),
            progress_distribution=_marginal_distribution(
                weighted,
                "progress_distribution",
            ),
            terminal_probability=_weighted_optional(
                weighted,
                "terminal_probability",
            ),
            goal_probability=_weighted_optional(
                weighted,
                "goal_probability",
            ),
            known_channels=frozenset(known_channels),
        )

    def start_branch(self, *, regime_index: int | None = None) -> None:
        self._branch_index += 1
        particles = []
        for particle in self._particles:
            state = particle.state
            if state is not None:
                state = AbstractState(
                    entities=state.entities,
                    true_facts=state.true_facts,
                    false_facts=state.false_facts,
                    counters=state.counters,
                    registers=(),
                    topology=state.topology,
                    regime_index=(
                        state.regime_index
                        if regime_index is None
                        else int(regime_index)
                    ),
                )
            particles.append(replace(particle, state=state))
        self._particles = particles

    @property
    def entropy(self) -> float:
        return -sum(
            probability * math.log(max(probability, 1e-300))
            for probability in (particle.probability for particle in self._particles)
            if probability > 0.0
        )

    @property
    def normalized_entropy(self) -> float:
        if len(self._particles) <= 1:
            return 0.0
        return self.entropy / math.log(len(self._particles))

    @property
    def effective_sample_size(self) -> float:
        denominator = sum(particle.probability**2 for particle in self._particles)
        return 0.0 if denominator <= 0.0 else 1.0 / denominator

    def top(self, maximum: int = 16) -> tuple[ProgramParticle, ...]:
        return tuple(
            sorted(
                self._particles,
                key=lambda particle: particle.log_weight,
                reverse=True,
            )[: max(0, int(maximum))]
        )

    def snapshot(
        self,
        *,
        maximum_programs: int | None = 8,
    ) -> Mapping[str, Any]:
        particles = (
            tuple(
                sorted(
                    self._particles,
                    key=lambda particle: particle.log_weight,
                    reverse=True,
                )
            )
            if maximum_programs is None
            else self.top(maximum_programs)
        )
        return {
            "particles": len(self._particles),
            "history": len(self._history),
            "branch_index": self._branch_index,
            "entropy": round(self.entropy, 6),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "effective_sample_size": round(
                self.effective_sample_size,
                6,
            ),
            "repairs_attempted": self._repairs_attempted,
            "repairs_admitted": self._repairs_admitted,
            "top": [
                {
                    "program_id": particle.program.program_id,
                    "program_hash": particle.program.canonical_hash,
                    "family": particle.program.semantic_family,
                    "probability": round(particle.probability, 8),
                    "latest_log_likelihood": round(
                        particle.latest_log_likelihood,
                        6,
                    ),
                    "node_count": particle.program.node_count,
                    "edit_distance": particle.program.edit_distance,
                }
                for particle in particles
            ],
        }

    def _needs_repair(self) -> bool:
        if not self._particles:
            return False
        best_latest = max(
            particle.latest_log_likelihood for particle in self._particles
        )
        return best_latest < self.repair_log_likelihood_threshold or (
            self.effective_sample_size < self.repair_ess_threshold
            and best_latest < -2.0
        )

    def _replay_particle(
        self,
        particle: ProgramParticle,
    ) -> ProgramParticle:
        log_weight = particle.log_prior
        state = None
        latest = 0.0
        observations = 0
        for evidence in self._history:
            if evidence.reset:
                state = None
                continue
            start = (
                evidence.state_before
                if state is None
                else state.merge_observation(evidence.state_before)
            )
            prediction = self.executor.step(
                particle.program,
                start,
                evidence.action,
            )
            latest = packet_log_likelihood(
                prediction,
                evidence.observation,
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            log_weight += latest
            predicted_state = prediction.state_after or start
            state = predicted_state.merge_observation(evidence.state_after)
            observations += 1
        return replace(
            particle,
            log_weight=log_weight,
            state=state,
            latest_log_likelihood=latest,
            observations=observations,
        )

    def _deduplicate(
        self,
        particles: Sequence[ProgramParticle],
    ) -> list[ProgramParticle]:
        unique: dict[str, ProgramParticle] = {}
        for particle in particles:
            key = particle.program.canonical_hash
            previous = unique.get(key)
            if previous is None or particle.log_weight > previous.log_weight:
                unique[key] = particle
        if self._history:
            semantic: dict[tuple[Any, ...], ProgramParticle] = {}
            for particle in unique.values():
                signature = self._observed_semantic_signature(particle.program)
                previous = semantic.get(signature)
                if previous is None:
                    semantic[signature] = particle
                    continue
                representative = (
                    particle if particle.log_weight > previous.log_weight else previous
                )
                semantic[signature] = replace(
                    representative,
                    log_prior=_logsumexp((previous.log_prior, particle.log_prior)),
                    log_weight=_logsumexp((previous.log_weight, particle.log_weight)),
                )
            unique = {
                particle.program.canonical_hash: particle
                for particle in semantic.values()
            }
        ranked = sorted(
            unique.values(),
            key=lambda particle: particle.log_weight,
            reverse=True,
        )
        selected = []
        family_counts: dict[tuple[str, str], int] = {}
        for particle in ranked:
            family = particle.program.semantic_family
            if family_counts.get(family, 0) >= 4:
                continue
            selected.append(particle)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) >= self.maximum_particles:
                return selected
        for particle in ranked:
            if particle in selected:
                continue
            selected.append(particle)
            if len(selected) >= self.maximum_particles:
                break
        return selected

    def _observed_semantic_signature(
        self,
        program: JointProgramHypothesis,
    ) -> tuple[Any, ...]:
        return tuple(
            (
                evidence.action.key,
                self.executor.step(
                    program,
                    evidence.state_before,
                    evidence.action,
                ).full_signature,
            )
            for evidence in self._history
            if not evidence.reset
        )

    def _normalize(self) -> None:
        if not self._particles:
            return
        normalizer = _logsumexp(particle.log_weight for particle in self._particles)
        self._particles = [
            replace(
                particle,
                log_weight=particle.log_weight - normalizer,
            )
            for particle in self._particles
        ]


def packet_log_likelihood(
    predicted: PredictionPacket,
    observed: PredictionPacket,
    *,
    channel_weights: Mapping[str, float] | None = None,
    unknown_coverage_penalty: float = 0.75,
) -> float:
    """Factorized likelihood that distinguishes false from unknown."""

    weights = dict(channel_weights or DEFAULT_CHANNEL_WEIGHTS)
    total = 0.0
    for channel, attribute in (
        ("objects", "object_deltas"),
        ("relations", "relation_deltas"),
        ("topology", "topology_deltas"),
    ):
        if channel not in observed.known_channels:
            continue
        if channel not in predicted.known_channels:
            total -= float(unknown_coverage_penalty) * weights[channel]
            continue
        predicted_events = dict(getattr(predicted, attribute))
        observed_events = {
            key
            for key, value in dict(getattr(observed, attribute)).items()
            if float(value) >= 0.5
        }
        universe = set(predicted_events) | observed_events
        if not universe:
            continue
        channel_total = 0.0
        for event in universe:
            probability = float(predicted_events.get(event, PREDICTED_FALSE))
            channel_total += _bernoulli_logprob(
                probability,
                event in observed_events,
            )
        total += weights[channel] * channel_total / len(universe)

    if "progress" in observed.known_channels:
        if (
            "progress" not in predicted.known_channels
            or predicted.progress_mean is None
            or observed.progress_mean is None
        ):
            total -= float(unknown_coverage_penalty) * weights["progress"]
        else:
            predicted_distribution = dict(predicted.progress_distribution)
            observed_distribution = dict(observed.progress_distribution)
            if predicted_distribution and observed_distribution:
                observed_bucket = max(
                    observed_distribution,
                    key=observed_distribution.get,
                )
                probability = predicted_distribution.get(
                    observed_bucket,
                    predicted_distribution.get("other", 1e-6),
                )
                total += weights["progress"] * math.log(max(1e-6, float(probability)))
            else:
                error = abs(
                    float(predicted.progress_mean) - float(observed.progress_mean)
                )
                total -= weights["progress"] * min(8.0, 4.0 * error)

    for channel, attribute in (
        ("terminal", "terminal_probability"),
        ("goal", "goal_probability"),
    ):
        if channel not in observed.known_channels:
            continue
        prediction = getattr(predicted, attribute)
        outcome = getattr(observed, attribute)
        if channel not in predicted.known_channels or prediction is None:
            total -= float(unknown_coverage_penalty) * weights[channel]
            continue
        if outcome is None:
            continue
        total += weights[channel] * _bernoulli_logprob(
            float(prediction),
            float(outcome) >= 0.5,
        )
    return float(total)


def _bernoulli_logprob(probability: float, outcome: bool) -> float:
    probability = max(1e-6, min(1.0 - 1e-6, float(probability)))
    return math.log(probability if outcome else 1.0 - probability)


def _logsumexp(values: Iterable[float]) -> float:
    items = tuple(float(value) for value in values)
    if not items:
        return float("-inf")
    maximum = max(items)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in items))


def _program_prior(program: JointProgramHypothesis) -> float:
    return (
        -0.05 * program.node_count
        - 0.25 * program.local_constant_count
        - float(program.edit_distance)
    )


def _marginal_events(
    weighted: Sequence[tuple[float, PredictionPacket]],
    attribute: str,
) -> dict[str, float]:
    keys = {key for _, packet in weighted for key in dict(getattr(packet, attribute))}
    return {
        key: sum(
            weight * float(dict(getattr(packet, attribute)).get(key, 0.05))
            for weight, packet in weighted
        )
        for key in sorted(keys)
    }


def _weighted_optional(
    weighted: Sequence[tuple[float, PredictionPacket]],
    attribute: str,
) -> float | None:
    available = [
        (weight, float(value))
        for weight, packet in weighted
        for value in (getattr(packet, attribute),)
        if value is not None
    ]
    mass = sum(weight for weight, _ in available)
    if mass <= 0.0:
        return None
    return sum(weight * value for weight, value in available) / mass


def _marginal_distribution(
    weighted: Sequence[tuple[float, PredictionPacket]],
    attribute: str,
) -> dict[str, float]:
    keys = {key for _, packet in weighted for key in dict(getattr(packet, attribute))}
    if not keys:
        return {}
    output = {
        key: sum(
            weight * float(dict(getattr(packet, attribute)).get(key, 0.0))
            for weight, packet in weighted
        )
        for key in keys
    }
    total = sum(output.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in output.items()}


__all__ = [
    "DEFAULT_CHANNEL_WEIGHTS",
    "ProgramParticle",
    "ProgramPosterior",
    "packet_log_likelihood",
]
