"""T8.6b channel-selective generalized-Bayes posterior.

T8.6 showed that a global likelihood temperature calibrated terminal risk but
also weakened useful dynamics evidence.  This module keeps the frozen T8.6
implementation untouched and applies fixed, pre-registered temperatures per
likelihood channel.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from statistics import mean
from typing import Any

from .contracts import ObservedTransition, PredictionPacket
from .executor import PREDICTED_FALSE
from .posterior import (
    DEFAULT_CHANNEL_WEIGHTS,
    _bernoulli_logprob,
    _logsumexp,
)
from .posterior_v2 import (
    CalibratedProgramParticle,
    CalibratedProgramPosterior,
)

CHANNELS = (
    "objects",
    "relations",
    "topology",
    "progress",
    "terminal",
    "goal",
)


@dataclass(frozen=True)
class ChannelPosteriorUpdatePolicy:
    """Fixed channel temperatures and channel-scoped correlation discount."""

    name: str
    channel_temperatures: tuple[tuple[str, float], ...]
    repeated_context_channels: tuple[str, ...] = ()
    semantic_collapse_entropy_maximum: float = 0.05
    semantic_collapse_surprise_minimum: float = 3.0
    repair_parent_limit: int = 4
    repair_child_limit: int = 0
    repair_survivor_limit: int = 0
    repair_minimum_improvement: float = 1.0
    incremental_repair: bool = False

    def __post_init__(self) -> None:
        temperatures = dict(self.channel_temperatures)
        if not self.name:
            raise ValueError("channel posterior policy needs a name")
        if set(temperatures) != set(CHANNELS):
            raise ValueError("every likelihood channel needs one temperature")
        if any(not 0.0 < float(value) <= 1.0 for value in temperatures.values()):
            raise ValueError("channel temperatures must be in (0, 1]")
        if not set(self.repeated_context_channels) <= set(CHANNELS):
            raise ValueError("unknown repeated-context channel")
        if len(set(self.repeated_context_channels)) != len(
            self.repeated_context_channels
        ):
            raise ValueError("repeated-context channels must be unique")
        if int(self.repair_parent_limit) < 1:
            raise ValueError("repair parent limit must be positive")
        if int(self.repair_child_limit) < 0 or int(self.repair_survivor_limit) < 0:
            raise ValueError("repair budgets must be nonnegative")

    @classmethod
    def legacy(cls) -> ChannelPosteriorUpdatePolicy:
        return cls("legacy", tuple((channel, 1.0) for channel in CHANNELS))

    @classmethod
    def terminal_tempered(cls) -> ChannelPosteriorUpdatePolicy:
        return cls(
            "terminal_tempered",
            tuple(
                (channel, 0.25 if channel == "terminal" else 1.0)
                for channel in CHANNELS
            ),
        )

    @classmethod
    def teleology_tempered(cls) -> ChannelPosteriorUpdatePolicy:
        return cls(
            "teleology_tempered",
            tuple(
                (
                    channel,
                    0.25
                    if channel in {"progress", "terminal", "goal"}
                    else 1.0,
                )
                for channel in CHANNELS
            ),
        )

    @classmethod
    def teleology_correlation_aware(cls) -> ChannelPosteriorUpdatePolicy:
        return cls(
            "teleology_correlation_aware",
            tuple(
                (
                    channel,
                    0.25
                    if channel in {"progress", "terminal", "goal"}
                    else 1.0,
                )
                for channel in CHANNELS
            ),
            repeated_context_channels=("progress", "terminal", "goal"),
        )

    def channel_multiplier(self, channel: str, context_count: int) -> float:
        temperature = dict(self.channel_temperatures)[str(channel)]
        discount = (
            1.0 / math.sqrt(max(1, int(context_count)))
            if channel in self.repeated_context_channels
            else 1.0
        )
        return float(temperature) * discount

    def with_repair_v2(self) -> ChannelPosteriorUpdatePolicy:
        return replace(
            self,
            name=f"{self.name}_repair_v2",
            repair_parent_limit=2,
            repair_child_limit=8,
            repair_survivor_limit=4,
            incremental_repair=True,
        )


T8_6B_POLICIES: Mapping[str, ChannelPosteriorUpdatePolicy] = {
    policy.name: policy
    for policy in (
        ChannelPosteriorUpdatePolicy.legacy(),
        ChannelPosteriorUpdatePolicy.terminal_tempered(),
        ChannelPosteriorUpdatePolicy.teleology_tempered(),
        ChannelPosteriorUpdatePolicy.teleology_correlation_aware(),
    )
}


@dataclass(frozen=True)
class ChannelPosteriorUpdateDiagnostics:
    policy_name: str
    context_signature: str
    context_count: int
    channel_multipliers: Mapping[str, float]
    raw_channel_log_likelihood_mean: Mapping[str, float]
    effective_channel_log_likelihood_mean: Mapping[str, float]
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


def packet_channel_log_likelihoods(
    predicted: PredictionPacket,
    observed: PredictionPacket,
    *,
    channel_weights: Mapping[str, float] | None = None,
    unknown_coverage_penalty: float = 0.75,
) -> dict[str, float]:
    """Return weighted raw likelihood terms whose sum is the frozen score."""

    weights = dict(channel_weights or DEFAULT_CHANNEL_WEIGHTS)
    output = {channel: 0.0 for channel in CHANNELS}
    for channel, attribute in (
        ("objects", "object_deltas"),
        ("relations", "relation_deltas"),
        ("topology", "topology_deltas"),
    ):
        if channel not in observed.known_channels:
            continue
        if channel not in predicted.known_channels:
            output[channel] = -float(unknown_coverage_penalty) * weights[channel]
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
        score = sum(
            _bernoulli_logprob(
                float(predicted_events.get(event, PREDICTED_FALSE)),
                event in observed_events,
            )
            for event in universe
        )
        output[channel] = weights[channel] * score / len(universe)

    if "progress" in observed.known_channels:
        if (
            "progress" not in predicted.known_channels
            or predicted.progress_mean is None
            or observed.progress_mean is None
        ):
            output["progress"] = (
                -float(unknown_coverage_penalty) * weights["progress"]
            )
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
                output["progress"] = weights["progress"] * math.log(
                    max(1e-6, float(probability))
                )
            else:
                error = abs(
                    float(predicted.progress_mean) - float(observed.progress_mean)
                )
                output["progress"] = -weights["progress"] * min(
                    8.0,
                    4.0 * error,
                )

    for channel, attribute in (
        ("terminal", "terminal_probability"),
        ("goal", "goal_probability"),
    ):
        if channel not in observed.known_channels:
            continue
        prediction = getattr(predicted, attribute)
        outcome = getattr(observed, attribute)
        if channel not in predicted.known_channels or prediction is None:
            output[channel] = -float(unknown_coverage_penalty) * weights[channel]
            continue
        if outcome is not None:
            output[channel] = weights[channel] * _bernoulli_logprob(
                float(prediction),
                float(outcome) >= 0.5,
            )
    return output


class ChannelCalibratedProgramPosterior(CalibratedProgramPosterior):
    """Calibrated posterior applying evidence multipliers per channel."""

    def __init__(
        self,
        *,
        update_policy: ChannelPosteriorUpdatePolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            update_policy=update_policy or ChannelPosteriorUpdatePolicy.legacy(),
            **kwargs,
        )
        self.update_policy: ChannelPosteriorUpdatePolicy
        self.last_update_diagnostics: ChannelPosteriorUpdateDiagnostics | None = None

    def observe(
        self,
        evidence: ObservedTransition,
        *,
        allow_repair: bool = True,
    ) -> ChannelPosteriorUpdateDiagnostics | None:
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
        multipliers = {
            channel: self.update_policy.channel_multiplier(channel, context_count)
            for channel in CHANNELS
        }
        raw_values: list[float] = []
        effective_values: list[float] = []
        raw_by_channel: dict[str, list[float]] = {
            channel: [] for channel in CHANNELS
        }
        effective_by_channel: dict[str, list[float]] = {
            channel: [] for channel in CHANNELS
        }
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
            channel_scores = packet_channel_log_likelihoods(
                prediction,
                evidence.observation,
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            raw = sum(channel_scores.values())
            effective = sum(
                channel_scores[channel] * multipliers[channel]
                for channel in CHANNELS
            )
            raw_values.append(raw)
            effective_values.append(effective)
            for channel in CHANNELS:
                raw_by_channel[channel].append(channel_scores[channel])
                effective_by_channel[channel].append(
                    channel_scores[channel] * multipliers[channel]
                )
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
        diagnostics = ChannelPosteriorUpdateDiagnostics(
            policy_name=self.update_policy.name,
            context_signature=context_signature,
            context_count=context_count,
            channel_multipliers=multipliers,
            raw_channel_log_likelihood_mean={
                channel: mean(values)
                for channel, values in raw_by_channel.items()
            },
            effective_channel_log_likelihood_mean={
                channel: mean(values)
                for channel, values in effective_by_channel.items()
            },
            raw_mixture_surprise=-mixture_log_probability,
            raw_best_program_surprise=-max(raw_values),
            raw_log_likelihood_minimum=min(raw_values),
            raw_log_likelihood_mean=mean(raw_values),
            raw_log_likelihood_maximum=max(raw_values),
            effective_log_likelihood_maximum=max(effective_values),
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
            repair_cycle_delta=repair_after[0] - repair_before[0],
            repair_proposed_delta=repair_after[1] - repair_before[1],
            repair_evaluated_delta=repair_after[2] - repair_before[2],
            repair_survived_delta=repair_after[3] - repair_before[3],
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
            channel_scores = packet_channel_log_likelihoods(
                prediction,
                evidence.observation,
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            latest_raw = sum(channel_scores.values())
            context = self._context_signature(evidence)
            count = local_context_counts.get(context, 0) + 1
            local_context_counts[context] = count
            latest_effective = sum(
                score * self.update_policy.channel_multiplier(channel, count)
                for channel, score in channel_scores.items()
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


__all__ = [
    "CHANNELS",
    "T8_6B_POLICIES",
    "ChannelCalibratedProgramPosterior",
    "ChannelPosteriorUpdateDiagnostics",
    "ChannelPosteriorUpdatePolicy",
    "packet_channel_log_likelihoods",
]
