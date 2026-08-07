"""T8.6j exact incremental acceleration for the frozen minimum-KL posterior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import AbstractState, JointProgramHypothesis
from .posterior import _program_prior
from .posterior_v2 import CalibratedProgramParticle
from .posterior_v3 import packet_channel_log_likelihoods
from .posterior_v8 import MinimumKLFamilyFloorProgramPosterior
from .synthesis import AssembledProgram


@dataclass(frozen=True)
class _ReplayCacheEntry:
    history_length: int
    evidence_log_joint: float
    effective_log_likelihoods: tuple[float, ...]
    state: AbstractState | None
    latest_effective: float
    latest_raw: float
    observations: int
    context_counts: tuple[tuple[str, int], ...]


class IncrementalMinimumKLProgramPosterior(
    MinimumKLFamilyFloorProgramPosterior
):
    """Avoid full-history work when the canonical program set is unchanged.

    The absolute ``log_joint`` update, projections, repair policy and particle
    ordering remain inherited from T8.6g. New programs still take the frozen
    full-replay path; only provable no-op additions and semantic signatures are
    incremental.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._semantic_signature_cache: dict[str, tuple[Any, ...]] = {}
        self._semantic_signature_history_length: dict[str, int] = {}
        self._replay_cache: dict[str, _ReplayCacheEntry] = {}
        self._last_program_batch_signature: tuple[
            tuple[str, float], ...
        ] = ()
        self._noop_program_additions = 0
        self._full_program_additions = 0
        self._novel_programs_replayed = 0
        self._semantic_cache_hits = 0
        self._semantic_cache_extensions = 0

    def seed(
        self,
        programs: Sequence[AssembledProgram | JointProgramHypothesis],
        *,
        initial_state: AbstractState | None = None,
    ) -> None:
        self._semantic_signature_cache.clear()
        self._semantic_signature_history_length.clear()
        self._replay_cache.clear()
        self._last_program_batch_signature = self._program_batch_signature(
            programs
        )
        super().seed(programs, initial_state=initial_state)

    def add_programs(
        self,
        programs: Sequence[AssembledProgram | JointProgramHypothesis],
        *,
        initial_state: AbstractState | None = None,
    ) -> None:
        batch_signature = self._program_batch_signature(programs)
        previous_hashes = {
            program_hash
            for program_hash, _prior in self._last_program_batch_signature
        }
        novel = {
            program_hash
            for program_hash, _prior in batch_signature
            if program_hash not in previous_hashes
        }
        if batch_signature == self._last_program_batch_signature:
            self._noop_program_additions += 1
        else:
            self._full_program_additions += 1
            self._novel_programs_replayed += len(novel)
            self._last_program_batch_signature = batch_signature
        candidates = list(self._particles)
        existing = {
            particle.program.canonical_hash for particle in candidates
        }
        for item in programs:
            if isinstance(item, AssembledProgram):
                program = item.program
                prior = float(item.prior_logprob)
            else:
                program = item
                prior = float(_program_prior(program))
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
                self._incremental_replay_particle(particle)
                for particle in candidates
            ]
        self._particles = self._deduplicate(candidates)
        self._normalize()
        self._repeat_frozen_projection_after_add()

    @staticmethod
    def _program_batch_signature(
        programs: Sequence[AssembledProgram | JointProgramHypothesis],
    ) -> tuple[tuple[str, float], ...]:
        return tuple(
            sorted(
                (
                    item.program.canonical_hash,
                    float(item.prior_logprob),
                )
                if isinstance(item, AssembledProgram)
                else (item.canonical_hash, float(_program_prior(item)))
                for item in programs
            )
        )

    def _incremental_replay_particle(
        self,
        particle: CalibratedProgramParticle,
    ) -> CalibratedProgramParticle:
        program_hash = particle.program.canonical_hash
        entry = self._replay_cache.get(program_hash)
        if entry is None or entry.history_length > len(self._history):
            entry = _ReplayCacheEntry(
                history_length=0,
                evidence_log_joint=0.0,
                effective_log_likelihoods=(),
                state=None,
                latest_effective=0.0,
                latest_raw=0.0,
                observations=0,
                context_counts=(),
            )
        if entry.history_length < len(self._history):
            entry = self._extend_replay_cache(particle.program, entry)
        self._replay_cache[program_hash] = entry
        joint = particle.log_prior
        for likelihood in entry.effective_log_likelihoods:
            joint += likelihood
        return replace(
            particle,
            log_weight=joint,
            log_joint=joint,
            state=entry.state,
            latest_log_likelihood=entry.latest_effective,
            latest_raw_log_likelihood=entry.latest_raw,
            observations=entry.observations,
        )

    def _extend_replay_cache(
        self,
        program: JointProgramHypothesis,
        entry: _ReplayCacheEntry,
    ) -> _ReplayCacheEntry:
        joint = entry.evidence_log_joint
        state = entry.state
        latest_effective = entry.latest_effective
        latest_raw = entry.latest_raw
        observations = entry.observations
        counts = dict(entry.context_counts)
        likelihoods = list(entry.effective_log_likelihoods)
        for evidence in self._history[entry.history_length:]:
            if evidence.reset:
                state = None
                continue
            start = (
                evidence.state_before
                if state is None
                else state.merge_observation(evidence.state_before)
            )
            prediction = self.executor.step(
                program,
                start,
                evidence.action,
            )
            channel_scores = packet_channel_log_likelihoods(
                prediction,
                evidence.observation,
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            latest_raw = sum(channel_scores.values())
            context = self._context_signature(evidence)
            count = counts.get(context, 0) + 1
            counts[context] = count
            latest_effective = sum(
                score
                * self.update_policy.channel_multiplier(channel, count)
                for channel, score in channel_scores.items()
            )
            joint += latest_effective
            likelihoods.append(latest_effective)
            predicted_state = prediction.state_after or start
            state = predicted_state.merge_observation(evidence.state_after)
            observations += 1
        return _ReplayCacheEntry(
            history_length=len(self._history),
            evidence_log_joint=joint,
            effective_log_likelihoods=tuple(likelihoods),
            state=state,
            latest_effective=latest_effective,
            latest_raw=latest_raw,
            observations=observations,
            context_counts=tuple(sorted(counts.items())),
        )

    def _repeat_frozen_projection_after_add(self) -> None:
        diagnostics = self.last_update_diagnostics
        if diagnostics is None:
            return
        base = diagnostics.base
        projection = self._project_if_needed(
            raw_surprise=base.raw_mixture_surprise
        )
        self.last_update_diagnostics = self._diagnostics(base, projection)

    def _observed_semantic_signature(
        self,
        program: JointProgramHypothesis,
    ) -> tuple[Any, ...]:
        program_hash = program.canonical_hash
        history_length = len(self._history)
        processed = self._semantic_signature_history_length.get(
            program_hash, 0
        )
        signature = self._semantic_signature_cache.get(program_hash, ())
        if processed > history_length:
            processed = 0
            signature = ()
        if processed == history_length:
            self._semantic_cache_hits += 1
            return signature
        additions = []
        for evidence in self._history[processed:]:
            if evidence.reset:
                continue
            additions.append(
                (
                    evidence.action.key,
                    self.executor.step(
                        program,
                        evidence.state_before,
                        evidence.action,
                    ).full_signature,
                )
            )
        signature = (*signature, *additions)
        self._semantic_signature_cache[program_hash] = signature
        self._semantic_signature_history_length[program_hash] = history_length
        self._semantic_cache_extensions += len(additions)
        return signature

    def performance_snapshot(self) -> Mapping[str, int]:
        return {
            "noop_program_additions": self._noop_program_additions,
            "full_program_additions": self._full_program_additions,
            "novel_programs_replayed": self._novel_programs_replayed,
            "semantic_cache_hits": self._semantic_cache_hits,
            "semantic_cache_extensions": self._semantic_cache_extensions,
            "semantic_cache_programs": len(self._semantic_signature_cache),
        }

    def snapshot(
        self,
        *,
        maximum_programs: int | None = 8,
    ) -> dict[str, Any]:
        payload = dict(
            super().snapshot(maximum_programs=maximum_programs)
        )
        payload["incremental_performance"] = dict(
            self.performance_snapshot()
        )
        return payload


__all__ = ["IncrementalMinimumKLProgramPosterior"]
