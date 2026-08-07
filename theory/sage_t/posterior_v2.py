"""T8.6 calibrated posterior challengers over executable programs.

The T7 posterior is hash-frozen in :mod:`theory.sage_t.posterior`.  This
module subclasses it without modifying that scientific baseline.  The
``legacy`` update policy is intentionally behavior-equivalent; other policies
only change the evidence multiplier and, optionally, the repair engine.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from statistics import mean
from typing import Any

from .contracts import AbstractState, JointProgramHypothesis, ObservedTransition
from .posterior import (
    ProgramParticle,
    ProgramPosterior,
    _logsumexp,
    _program_prior,
    packet_log_likelihood,
)
from .synthesis import AssembledProgram


@dataclass(frozen=True)
class PosteriorUpdatePolicy:
    """Pre-registered evidence and repair policy for one T8.6 condition."""

    name: str = "legacy"
    likelihood_temperature: float = 1.0
    repeated_context_discount: bool = False
    semantic_collapse_entropy_maximum: float = 0.05
    semantic_collapse_surprise_minimum: float = 3.0
    repair_parent_limit: int = 4
    repair_child_limit: int = 0
    repair_survivor_limit: int = 0
    repair_minimum_improvement: float = 1.0
    incremental_repair: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("posterior update policy needs a name")
        if not 0.0 < float(self.likelihood_temperature) <= 1.0:
            raise ValueError("likelihood temperature must be in (0, 1]")
        if not 0.0 <= float(self.semantic_collapse_entropy_maximum) <= 1.0:
            raise ValueError("invalid semantic-collapse entropy threshold")
        if float(self.semantic_collapse_surprise_minimum) < 0.0:
            raise ValueError("semantic-collapse surprise must be nonnegative")
        if int(self.repair_parent_limit) < 1:
            raise ValueError("repair parent limit must be positive")
        if int(self.repair_child_limit) < 0:
            raise ValueError("repair child limit must be nonnegative")
        if int(self.repair_survivor_limit) < 0:
            raise ValueError("repair survivor limit must be nonnegative")
        if float(self.repair_minimum_improvement) < 0.0:
            raise ValueError("repair improvement must be nonnegative")

    @classmethod
    def legacy(cls) -> PosteriorUpdatePolicy:
        return cls()

    @classmethod
    def tempered(cls) -> PosteriorUpdatePolicy:
        return cls(name="tempered", likelihood_temperature=0.25)

    @classmethod
    def correlation_aware(cls) -> PosteriorUpdatePolicy:
        return cls(name="correlation_aware", repeated_context_discount=True)

    @classmethod
    def combined(cls) -> PosteriorUpdatePolicy:
        return cls(
            name="combined",
            likelihood_temperature=0.25,
            repeated_context_discount=True,
        )

    def with_repair_v2(self) -> PosteriorUpdatePolicy:
        return replace(
            self,
            name=f"{self.name}_repair_v2",
            repair_parent_limit=2,
            repair_child_limit=8,
            repair_survivor_limit=4,
            incremental_repair=True,
        )

    def evidence_multiplier(self, context_count: int) -> float:
        discount = (
            1.0 / math.sqrt(max(1, int(context_count)))
            if self.repeated_context_discount
            else 1.0
        )
        return float(self.likelihood_temperature) * discount


T8_6_POLICIES: Mapping[str, PosteriorUpdatePolicy] = {
    policy.name: policy
    for policy in (
        PosteriorUpdatePolicy.legacy(),
        PosteriorUpdatePolicy.tempered(),
        PosteriorUpdatePolicy.correlation_aware(),
        PosteriorUpdatePolicy.combined(),
    )
}


@dataclass(frozen=True)
class PosteriorUpdateDiagnostics:
    policy_name: str
    context_signature: str
    context_count: int
    evidence_multiplier: float
    raw_mixture_surprise: float
    raw_best_program_surprise: float
    raw_log_likelihood_minimum: float
    raw_log_likelihood_mean: float
    raw_log_likelihood_maximum: float
    effective_log_likelihood_maximum: float
    entropy_before: float
    entropy_after: float
    effective_sample_size_before: float
    effective_sample_size_after: float
    semantic_collapse: bool
    repair_cycle_delta: int
    repair_proposed_delta: int
    repair_evaluated_delta: int
    repair_survived_delta: int
    executor_cache_hits_delta: int
    executor_cache_misses_delta: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibratedProgramParticle(ProgramParticle):
    """Particle carrying both normalized mass and its absolute joint score."""

    log_joint: float = 0.0
    latest_raw_log_likelihood: float = 0.0


class CalibratedProgramPosterior(ProgramPosterior):
    """Program posterior with auditable generalized-Bayes updates."""

    def __init__(
        self,
        *,
        update_policy: PosteriorUpdatePolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.update_policy = update_policy or PosteriorUpdatePolicy.legacy()
        self._context_counts: dict[str, int] = {}
        self._repair_cycles = 0
        self._repairs_proposed = 0
        self._repairs_evaluated = 0
        self._repairs_survived = 0
        self.last_update_diagnostics: PosteriorUpdateDiagnostics | None = None

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
                CalibratedProgramParticle(
                    program=program,
                    log_prior=prior,
                    log_weight=prior,
                    log_joint=prior,
                    state=initial_state,
                )
            )
        if self._history:
            seeded = [
                self._replay_particle(replace(particle, state=None))
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
        candidates = [
            replace(particle, log_weight=particle.log_joint)
            for particle in self._particles
        ]
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
            candidates.append(
                CalibratedProgramParticle(
                    program=program,
                    log_prior=prior,
                    log_weight=prior,
                    log_joint=prior,
                    state=initial_state,
                )
            )
            existing.add(program.canonical_hash)
        if self._history:
            candidates = [
                self._replay_particle(
                    replace(
                        particle,
                        log_weight=particle.log_prior,
                        log_joint=particle.log_prior,
                        state=None,
                        latest_log_likelihood=0.0,
                        latest_raw_log_likelihood=0.0,
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
    ) -> PosteriorUpdateDiagnostics | None:
        if evidence.reset:
            return None
        if not self._particles:
            self._history.append(evidence)
            return None
        started = time.perf_counter()
        entropy_before = self.normalized_entropy
        ess_before = self.effective_sample_size
        cache_before = self.executor.summary()
        repair_before = self._repair_counter_snapshot()
        context_signature = self._context_signature(evidence)
        context_count = self._context_counts.get(context_signature, 0) + 1
        self._context_counts[context_signature] = context_count
        multiplier = self.update_policy.evidence_multiplier(context_count)
        raw_values = []
        updated = []
        for particle in self._particles:
            start = (
                evidence.state_before
                if particle.state is None
                else particle.state.merge_observation(evidence.state_before)
            )
            prediction = self.executor.step(
                particle.program,
                start,
                evidence.action,
            )
            raw = packet_log_likelihood(
                prediction,
                evidence.observation,
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            effective = multiplier * raw
            raw_values.append(raw)
            predicted_state = prediction.state_after or start
            next_state = predicted_state.merge_observation(evidence.state_after)
            joint = particle.log_joint + effective
            updated.append(
                replace(
                    particle,
                    log_weight=joint,
                    log_joint=joint,
                    state=next_state,
                    latest_log_likelihood=effective,
                    latest_raw_log_likelihood=raw,
                    observations=particle.observations + 1,
                )
            )
        mixture_log_probability = _logsumexp(
            particle.log_weight + raw
            for particle, raw in zip(self._particles, raw_values)
        )
        self._history.append(evidence)
        self._particles = self._deduplicate(updated)
        self._normalize()
        if allow_repair and self._needs_repair():
            self.repair(evidence)
        entropy_after = self.normalized_entropy
        repair_after = self._repair_counter_snapshot()
        cache_after = self.executor.summary()
        diagnostics = PosteriorUpdateDiagnostics(
            policy_name=self.update_policy.name,
            context_signature=context_signature,
            context_count=context_count,
            evidence_multiplier=multiplier,
            raw_mixture_surprise=-mixture_log_probability,
            raw_best_program_surprise=-max(raw_values),
            raw_log_likelihood_minimum=min(raw_values),
            raw_log_likelihood_mean=mean(raw_values),
            raw_log_likelihood_maximum=max(raw_values),
            effective_log_likelihood_maximum=multiplier * max(raw_values),
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            effective_sample_size_before=ess_before,
            effective_sample_size_after=self.effective_sample_size,
            semantic_collapse=(
                entropy_after
                < self.update_policy.semantic_collapse_entropy_maximum
                and -mixture_log_probability
                > self.update_policy.semantic_collapse_surprise_minimum
            ),
            repair_cycle_delta=(
                repair_after[0] - repair_before[0]
            ),
            repair_proposed_delta=(
                repair_after[1] - repair_before[1]
            ),
            repair_evaluated_delta=(
                repair_after[2] - repair_before[2]
            ),
            repair_survived_delta=(
                repair_after[3] - repair_before[3]
            ),
            executor_cache_hits_delta=(
                cache_after["cache_hits"] - cache_before["cache_hits"]
            ),
            executor_cache_misses_delta=(
                cache_after["cache_misses"] - cache_before["cache_misses"]
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        self.last_update_diagnostics = diagnostics
        return diagnostics

    def repair(
        self,
        evidence: ObservedTransition | None = None,
    ) -> tuple[CalibratedProgramParticle, ...]:
        if not self._particles or not self._history:
            return ()
        if self.update_policy.incremental_repair:
            return self._repair_v2(evidence)
        return self._repair_legacy(evidence)

    def _repair_legacy(
        self,
        evidence: ObservedTransition | None,
    ) -> tuple[CalibratedProgramParticle, ...]:
        target = evidence or self._history[-1]
        parents = sorted(
            self._particles,
            key=lambda particle: (
                particle.latest_log_likelihood,
                particle.log_weight,
            ),
            reverse=True,
        )[: self.update_policy.repair_parent_limit]
        self._repair_cycles += 1
        self._repairs_attempted += len(parents)
        children = []
        existing = {particle.program.canonical_hash for particle in self._particles}
        for parent in parents:
            for program in self.mutator.mutate(parent.program, target):
                self._repairs_proposed += 1
                if program.canonical_hash in existing:
                    continue
                self._repairs_evaluated += 1
                prior = parent.log_prior - max(
                    1.0,
                    float(program.edit_distance - parent.program.edit_distance),
                )
                child = self._replay_particle(
                    CalibratedProgramParticle(
                        program=program,
                        log_prior=prior,
                        log_weight=prior,
                        log_joint=prior,
                    )
                )
                children.append(child)
                existing.add(program.canonical_hash)
        if not children:
            return ()
        self._repairs_admitted += len(children)
        self._repairs_survived += len(children)
        replayed_parents = [
            self._replay_particle(
                replace(
                    particle,
                    log_weight=particle.log_prior,
                    log_joint=particle.log_prior,
                    state=None,
                    latest_log_likelihood=0.0,
                    latest_raw_log_likelihood=0.0,
                    observations=0,
                )
            )
            for particle in self._particles
        ]
        self._particles = self._deduplicate([*replayed_parents, *children])
        self._normalize()
        return tuple(children)

    def _repair_v2(
        self,
        evidence: ObservedTransition | None,
    ) -> tuple[CalibratedProgramParticle, ...]:
        target = evidence or self._history[-1]
        parents = sorted(
            self._particles,
            key=lambda particle: (
                particle.latest_raw_log_likelihood,
                particle.log_weight,
            ),
            reverse=True,
        )[: self.update_policy.repair_parent_limit]
        self._repair_cycles += 1
        self._repairs_attempted += 1
        evaluated = 0
        survivors = []
        existing_hashes = {
            particle.program.canonical_hash for particle in self._particles
        }
        existing_families = {
            particle.program.semantic_family for particle in self._particles
        }
        for parent in parents:
            for program in self.mutator.mutate(parent.program, target):
                self._repairs_proposed += 1
                if program.canonical_hash in existing_hashes:
                    continue
                if (
                    self.update_policy.repair_child_limit
                    and evaluated >= self.update_policy.repair_child_limit
                ):
                    break
                evaluated += 1
                self._repairs_evaluated += 1
                prior = parent.log_prior - max(
                    1.0,
                    float(program.edit_distance - parent.program.edit_distance),
                )
                child = self._replay_particle(
                    CalibratedProgramParticle(
                        program=program,
                        log_prior=prior,
                        log_weight=prior,
                        log_joint=prior,
                    )
                )
                improves = (
                    child.log_joint
                    >= parent.log_joint
                    + self.update_policy.repair_minimum_improvement
                )
                new_family = program.semantic_family not in existing_families
                existing_hashes.add(program.canonical_hash)
                if improves or new_family:
                    survivors.append(child)
                    existing_families.add(program.semantic_family)
                    if (
                        self.update_policy.repair_survivor_limit
                        and len(survivors)
                        >= self.update_policy.repair_survivor_limit
                    ):
                        break
            if (
                self.update_policy.repair_survivor_limit
                and len(survivors)
                >= self.update_policy.repair_survivor_limit
            ):
                break
            if (
                self.update_policy.repair_child_limit
                and evaluated >= self.update_policy.repair_child_limit
            ):
                break
        if not survivors:
            return ()
        self._repairs_admitted += len(survivors)
        self._repairs_survived += len(survivors)
        parents_absolute = [
            replace(particle, log_weight=particle.log_joint)
            for particle in self._particles
        ]
        self._particles = self._deduplicate([*parents_absolute, *survivors])
        self._normalize()
        return tuple(survivors)

    def _needs_repair(self) -> bool:
        if not self._particles:
            return False
        best_latest = max(
            particle.latest_raw_log_likelihood for particle in self._particles
        )
        return best_latest < self.repair_log_likelihood_threshold or (
            self.effective_sample_size < self.repair_ess_threshold
            and best_latest < -2.0
        )

    def _replay_particle(
        self,
        particle: CalibratedProgramParticle,
    ) -> CalibratedProgramParticle:
        joint = particle.log_prior
        state = None
        latest_raw = 0.0
        latest_effective = 0.0
        observations = 0
        local_context_counts: dict[str, int] = {}
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
            latest_raw = packet_log_likelihood(
                prediction,
                evidence.observation,
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            context = self._context_signature(evidence)
            count = local_context_counts.get(context, 0) + 1
            local_context_counts[context] = count
            latest_effective = (
                self.update_policy.evidence_multiplier(count) * latest_raw
            )
            joint += latest_effective
            predicted_state = prediction.state_after or start
            state = predicted_state.merge_observation(evidence.state_after)
            observations += 1
        return replace(
            particle,
            log_weight=joint,
            log_joint=joint,
            state=state,
            latest_log_likelihood=latest_effective,
            latest_raw_log_likelihood=latest_raw,
            observations=observations,
        )

    def _deduplicate(
        self,
        particles: Sequence[CalibratedProgramParticle],
    ) -> list[CalibratedProgramParticle]:
        unique: dict[str, CalibratedProgramParticle] = {}
        for particle in particles:
            key = particle.program.canonical_hash
            previous = unique.get(key)
            if previous is None or particle.log_joint > previous.log_joint:
                unique[key] = particle
        if self._history:
            semantic: dict[tuple[Any, ...], CalibratedProgramParticle] = {}
            for particle in unique.values():
                signature = self._observed_semantic_signature(particle.program)
                previous = semantic.get(signature)
                if previous is None:
                    semantic[signature] = particle
                    continue
                representative = (
                    particle
                    if particle.log_joint > previous.log_joint
                    else previous
                )
                merged_joint = _logsumexp(
                    (previous.log_joint, particle.log_joint)
                )
                semantic[signature] = replace(
                    representative,
                    log_prior=_logsumexp(
                        (previous.log_prior, particle.log_prior)
                    ),
                    log_weight=merged_joint,
                    log_joint=merged_joint,
                )
            unique = {
                particle.program.canonical_hash: particle
                for particle in semantic.values()
            }
        ranked = sorted(
            unique.values(),
            key=lambda particle: particle.log_joint,
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

    def _normalize(self) -> None:
        if not self._particles:
            return
        normalizer = _logsumexp(
            particle.log_joint for particle in self._particles
        )
        self._particles = [
            replace(
                particle,
                log_weight=particle.log_joint - normalizer,
            )
            for particle in self._particles
        ]

    def snapshot(
        self,
        *,
        maximum_programs: int | None = 8,
    ) -> Mapping[str, Any]:
        payload = dict(super().snapshot(maximum_programs=maximum_programs))
        particles = (
            tuple(
                sorted(
                    self._particles,
                    key=lambda item: item.log_weight,
                    reverse=True,
                )
            )
            if maximum_programs is None
            else self.top(maximum_programs)
        )
        by_hash = {
            particle.program.canonical_hash: particle for particle in particles
        }
        enriched = []
        for item in payload.get("top", ()):
            public = dict(item)
            particle = by_hash[public["program_hash"]]
            public["log_joint"] = round(particle.log_joint, 6)
            public["latest_raw_log_likelihood"] = round(
                particle.latest_raw_log_likelihood,
                6,
            )
            enriched.append(public)
        payload["top"] = enriched
        payload["update_policy"] = asdict(self.update_policy)
        payload["repair_cycles"] = self._repair_cycles
        payload["repairs_proposed"] = self._repairs_proposed
        payload["repairs_evaluated"] = self._repairs_evaluated
        payload["repairs_survived"] = self._repairs_survived
        payload["last_update"] = (
            None
            if self.last_update_diagnostics is None
            else self.last_update_diagnostics.to_dict()
        )
        return payload

    @staticmethod
    def _context_signature(evidence: ObservedTransition) -> str:
        return ":".join(
            (
                str(evidence.state_before.regime_index),
                evidence.state_before.signature,
                evidence.action.action_name,
            )
        )

    def _repair_counter_snapshot(self) -> tuple[int, int, int, int]:
        return (
            self._repair_cycles,
            self._repairs_proposed,
            self._repairs_evaluated,
            self._repairs_survived,
        )


__all__ = [
    "T8_6_POLICIES",
    "CalibratedProgramParticle",
    "CalibratedProgramPosterior",
    "PosteriorUpdateDiagnostics",
    "PosteriorUpdatePolicy",
]
