"""SAGE.T.A39: one robust posterior over complete causal programs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .comparison import ParticleComparison, compare_particle
from .contracts import CausalProgram, CausalState, GroundedAction, TransitionEvidence
from .executor import CausalExecutor
from .repair import CausalProgramRepairer


@dataclass(frozen=True)
class CausalParticle:
    program: CausalProgram
    log_prior: float
    log_weight: float
    lineage: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    latest_log_likelihood: float = 0.0

    @property
    def probability(self) -> float:
        return math.exp(self.log_weight)


@dataclass(frozen=True)
class PosteriorUpdate:
    evidence_id: str
    entropy_before: float
    entropy_after: float
    effective_sample_size: float
    comparisons: tuple[ParticleComparison, ...]
    pruned_programs: tuple[str, ...] = ()
    merged_programs: tuple[str, ...] = ()
    repair_children: tuple[str, ...] = ()

    @property
    def entropy_reduction(self) -> float:
        return self.entropy_before - self.entropy_after


@dataclass(frozen=True)
class InterventionSupport:
    trials: int = 0
    terminal_failures: int = 0

    @property
    def safe(self) -> bool:
        return self.trials > 0 and self.terminal_failures == 0


class CausalPosterior:
    """Normalized log-space posterior bound to exactly one executor."""

    def __init__(
        self,
        *,
        executor: CausalExecutor,
        maximum_particles: int = 64,
        maximum_family_particles: int = 4,
        minimum_particles: int = 2,
        mdl_beta: float = 1.0,
        prune_log_odds: float = 20.0,
        repairer: CausalProgramRepairer | None = None,
        repair_log_likelihood_threshold: float = -8.0,
        maximum_repair_parents: int = 4,
    ) -> None:
        self.executor = executor
        self.maximum_particles = max(1, int(maximum_particles))
        self.maximum_family_particles = max(1, int(maximum_family_particles))
        self.minimum_particles = max(
            1, min(self.maximum_particles, int(minimum_particles))
        )
        self.mdl_beta = max(0.0, float(mdl_beta))
        self.prune_log_odds = max(0.0, float(prune_log_odds))
        self.repairer = repairer or CausalProgramRepairer()
        self.repair_log_likelihood_threshold = float(repair_log_likelihood_threshold)
        self.maximum_repair_parents = max(0, min(4, int(maximum_repair_parents)))
        self._particles: list[CausalParticle] = []
        self._evidence: list[TransitionEvidence] = []

    @property
    def particles(self) -> tuple[CausalParticle, ...]:
        return tuple(self._particles)

    @property
    def evidence(self) -> tuple[TransitionEvidence, ...]:
        return tuple(self._evidence)

    @property
    def entropy(self) -> float:
        return -sum(
            particle.probability * particle.log_weight
            for particle in self._particles
            if particle.probability > 0.0
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

    def seed(self, programs: Sequence[CausalProgram]) -> None:
        seeded = [
            CausalParticle(
                program=program,
                log_prior=-self.mdl_beta * float(program.description_length),
                log_weight=-self.mdl_beta * float(program.description_length),
                lineage=(program.canonical_hash,),
            )
            for program in programs
        ]
        self._particles, _ = self._deduplicate(seeded)
        self._particles = self._select_diverse(self._particles)
        self._normalize()

    def add_programs(self, programs: Sequence[CausalProgram]) -> int:
        admitted = []
        existing = {
            particle.program.canonical_hash for particle in self._particles
        }
        for program in programs:
            if program.canonical_hash in existing:
                continue
            log_prior = -self.mdl_beta * float(program.description_length)
            log_weight = log_prior
            latest = 0.0
            evidence_ids = []
            for evidence in self._evidence:
                comparison = compare_particle(
                    program=program,
                    evidence=evidence,
                    executor=self.executor,
                )
                latest = comparison.log_likelihood
                log_weight += latest
                evidence_ids.append(evidence.evidence_id)
            admitted.append(
                CausalParticle(
                    program=program,
                    log_prior=log_prior,
                    log_weight=log_weight,
                    lineage=(program.canonical_hash,),
                    evidence_ids=tuple(evidence_ids),
                    latest_log_likelihood=latest,
                )
            )
        if not admitted:
            return 0
        combined, _ = self._deduplicate((*self._particles, *admitted))
        self._particles = self._select_diverse(combined)
        self._normalize()
        return len(admitted)

    def restore(
        self,
        particles: Sequence[CausalParticle],
        *,
        evidence: Sequence[TransitionEvidence] = (),
    ) -> None:
        self._particles, _ = self._deduplicate(tuple(particles))
        self._particles = self._select_diverse(self._particles)
        self._normalize()
        self._evidence = list(evidence)

    def update(self, evidence: TransitionEvidence) -> PosteriorUpdate:
        if not self._particles:
            raise RuntimeError("cannot update an empty causal posterior")
        entropy_before = self.entropy
        comparisons = tuple(
            compare_particle(
                program=particle.program,
                evidence=evidence,
                executor=self.executor,
            )
            for particle in self._particles
        )
        by_hash = {item.program_hash: item for item in comparisons}
        updated = [
            replace(
                particle,
                log_weight=particle.log_weight
                + by_hash[particle.program.canonical_hash].log_likelihood,
                evidence_ids=particle.evidence_ids + (evidence.evidence_id,),
                latest_log_likelihood=by_hash[
                    particle.program.canonical_hash
                ].log_likelihood,
            )
            for particle in self._particles
        ]
        best = max(particle.log_weight for particle in updated)
        retained = [
            particle
            for particle in updated
            if particle.log_weight >= best - self.prune_log_odds
        ]
        if not retained:
            retained = [max(updated, key=lambda particle: particle.log_weight)]
        if len(retained) < min(self.minimum_particles, len(updated)):
            retained_hashes = {
                particle.program.canonical_hash for particle in retained
            }
            for particle in sorted(
                updated, key=lambda item: item.log_weight, reverse=True
            ):
                if particle.program.canonical_hash in retained_hashes:
                    continue
                retained.append(particle)
                retained_hashes.add(particle.program.canonical_hash)
                if len(retained) >= min(self.minimum_particles, len(updated)):
                    break
        retained_hashes = {particle.program.canonical_hash for particle in retained}
        pruned = tuple(
            particle.program.canonical_hash
            for particle in updated
            if particle.program.canonical_hash not in retained_hashes
        )
        repair_children: list[CausalParticle] = []
        weak = sorted(retained, key=lambda item: item.latest_log_likelihood)[
            : self.maximum_repair_parents
        ]
        for parent in weak:
            if parent.latest_log_likelihood > self.repair_log_likelihood_threshold:
                continue
            comparison = by_hash[parent.program.canonical_hash]
            for child in self.repairer.propose(parent.program, evidence, comparison):
                child_prior = -self.mdl_beta * float(child.description_length)
                child_log_weight = child_prior
                child_evidence_ids = []
                child_comparison = None
                for historical in (*self._evidence, evidence):
                    child_comparison = compare_particle(
                        program=child,
                        evidence=historical,
                        executor=self.executor,
                    )
                    child_log_weight += child_comparison.log_likelihood
                    child_evidence_ids.append(historical.evidence_id)
                if child_comparison is None:
                    continue
                repair_children.append(
                    CausalParticle(
                        program=child,
                        log_prior=child_prior,
                        log_weight=child_log_weight,
                        lineage=parent.lineage + (child.canonical_hash,),
                        evidence_ids=tuple(child_evidence_ids),
                        latest_log_likelihood=child_comparison.log_likelihood,
                    )
                )
        deduplicated, merged = self._deduplicate(retained + repair_children)
        self._particles = self._select_diverse(deduplicated)
        self._normalize()
        self._evidence.append(evidence)
        return PosteriorUpdate(
            evidence_id=evidence.evidence_id,
            entropy_before=entropy_before,
            entropy_after=self.entropy,
            effective_sample_size=self.effective_sample_size,
            comparisons=comparisons,
            pruned_programs=pruned,
            merged_programs=merged,
            repair_children=tuple(item.program.canonical_hash for item in repair_children),
        )

    def top(self, maximum_particles: int = 16) -> tuple[CausalParticle, ...]:
        return tuple(
            sorted(self._particles, key=lambda item: item.log_weight, reverse=True)[
                : max(0, int(maximum_particles))
            ]
        )

    def snapshot(self, maximum_particles: int | None = 16) -> Mapping[str, object]:
        particles = self.top(len(self._particles) if maximum_particles is None else maximum_particles)
        return {
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "effective_sample_size": self.effective_sample_size,
            "evidence_count": len(self._evidence),
            "particles": [
                {
                    "program_id": item.program.program_id,
                    "program_hash": item.program.canonical_hash,
                    "probability": item.probability,
                    "log_prior": item.log_prior,
                    "latest_log_likelihood": item.latest_log_likelihood,
                    "lineage": list(item.lineage),
                    "evidence_ids": list(item.evidence_ids),
                }
                for item in particles
            ],
        }

    def intervention_support(
        self,
        state: CausalState,
        action: GroundedAction,
        *,
        tolerance: float = 1e-9,
    ) -> InterventionSupport:
        """Count exact-action evidence at the current declared-state projection."""

        declared_variables = tuple(
            sorted(
                {
                    variable.variable_id
                    for particle in self._particles
                    for variable in particle.program.variables
                }
            )
        )
        trials = 0
        terminal_failures = 0
        for evidence in self._evidence:
            if evidence.action.key != action.key:
                continue
            if not declared_variables or any(
                variable_id not in state.variables
                or variable_id not in evidence.state_before.variables
                or state.value(variable_id).total_variation(
                    evidence.state_before.value(variable_id)
                )
                > tolerance
                for variable_id in declared_variables
            ):
                continue
            trials += 1
            if evidence.terminal and evidence.success is not True:
                terminal_failures += 1
        return InterventionSupport(
            trials=trials,
            terminal_failures=terminal_failures,
        )

    def _deduplicate(
        self, particles: Sequence[CausalParticle]
    ) -> tuple[list[CausalParticle], tuple[str, ...]]:
        # Intentionally merge only canonical structural equality.  Equal
        # predictions on observed history are not interventional equivalence.
        unique: dict[str, CausalParticle] = {}
        merged: list[str] = []
        for particle in particles:
            key = particle.program.canonical_hash
            previous = unique.get(key)
            if previous is None:
                unique[key] = particle
                continue
            representative = (
                particle if particle.log_weight > previous.log_weight else previous
            )
            unique[key] = replace(
                representative,
                log_prior=_logsumexp((previous.log_prior, particle.log_prior)),
                log_weight=_logsumexp((previous.log_weight, particle.log_weight)),
                lineage=tuple(dict.fromkeys(previous.lineage + particle.lineage)),
                evidence_ids=tuple(
                    dict.fromkeys(previous.evidence_ids + particle.evidence_ids)
                ),
            )
            merged.append(key)
        return list(unique.values()), tuple(merged)

    def _select_diverse(self, particles: Sequence[CausalParticle]) -> list[CausalParticle]:
        ranked = sorted(particles, key=lambda item: item.log_weight, reverse=True)
        selected: list[CausalParticle] = []
        family_counts: dict[tuple[tuple[str, ...], str], int] = {}
        deferred: list[CausalParticle] = []
        for particle in ranked:
            family = particle.program.structural_family
            if family_counts.get(family, 0) >= self.maximum_family_particles:
                deferred.append(particle)
                continue
            selected.append(particle)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) >= self.maximum_particles:
                return selected
        selected.extend(deferred[: self.maximum_particles - len(selected)])
        return selected

    def _normalize(self) -> None:
        if not self._particles:
            return
        normalizer = _logsumexp(item.log_weight for item in self._particles)
        self._particles = [
            replace(item, log_weight=item.log_weight - normalizer)
            for item in self._particles
        ]


def _logsumexp(values: Sequence[float]) -> float:
    materialized = tuple(float(value) for value in values)
    maximum = max(materialized)
    return maximum + math.log(sum(math.exp(value - maximum) for value in materialized))


__all__ = [
    "CausalParticle",
    "CausalPosterior",
    "InterventionSupport",
    "PosteriorUpdate",
]
