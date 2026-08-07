"""T8.6c defensive semantic-family posterior projection.

The projection is activated only when the raw posterior is both highly
concentrated and surprised.  It mixes a fixed amount of mass with a uniform
distribution over semantic families, preserving prior proportions within each
family.  No new hypothesis or empirical support is created.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from .contracts import AbstractState, JointProgramHypothesis, ObservedTransition
from .posterior import _logsumexp
from .posterior_v2 import CalibratedProgramParticle
from .posterior_v3 import (
    CHANNELS,
    ChannelCalibratedProgramPosterior,
    ChannelPosteriorUpdateDiagnostics,
    ChannelPosteriorUpdatePolicy,
)
from .synthesis import AssembledProgram


@dataclass(frozen=True)
class FamilyDiversityPolicy(ChannelPosteriorUpdatePolicy):
    """Channel policy plus a fixed collapse-triggered defensive mixture."""

    family_mixture: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= float(self.family_mixture) <= 0.25:
            raise ValueError("family mixture must be in [0, 0.25]")

    @classmethod
    def legacy(cls) -> FamilyDiversityPolicy:
        return cls(
            name="legacy",
            channel_temperatures=tuple((channel, 1.0) for channel in CHANNELS),
        )

    @classmethod
    def terminal_tempered(cls) -> FamilyDiversityPolicy:
        return cls(
            name="terminal_tempered",
            channel_temperatures=tuple(
                (channel, 0.25 if channel == "terminal" else 1.0)
                for channel in CHANNELS
            ),
        )

    @classmethod
    def terminal_tempered_family_mix(
        cls,
        mixture: float,
    ) -> FamilyDiversityPolicy:
        label = f"terminal_tempered_family_mix_{round(100 * mixture):02d}"
        return cls(
            name=label,
            channel_temperatures=tuple(
                (channel, 0.25 if channel == "terminal" else 1.0)
                for channel in CHANNELS
            ),
            family_mixture=float(mixture),
        )

    def with_repair_v2(self) -> FamilyDiversityPolicy:
        return replace(
            self,
            name=f"{self.name}_repair_v2",
            repair_parent_limit=2,
            repair_child_limit=8,
            repair_survivor_limit=4,
            incremental_repair=True,
        )


T8_6C_POLICIES = {
    policy.name: policy
    for policy in (
        FamilyDiversityPolicy.legacy(),
        FamilyDiversityPolicy.terminal_tempered(),
        FamilyDiversityPolicy.terminal_tempered_family_mix(0.02),
        FamilyDiversityPolicy.terminal_tempered_family_mix(0.05),
        FamilyDiversityPolicy.terminal_tempered_family_mix(0.10),
    )
}


@dataclass(frozen=True)
class FamilyDiversityDiagnostics:
    base: ChannelPosteriorUpdateDiagnostics
    pre_projection_entropy: float
    post_projection_entropy: float
    pre_projection_ess: float
    post_projection_ess: float
    family_count: int
    family_mixture: float
    regularization_applied: bool
    semantic_collapse: bool

    def to_dict(self) -> dict[str, Any]:
        payload = self.base.to_dict()
        payload.update(
            {
                "pre_projection_entropy": self.pre_projection_entropy,
                "post_projection_entropy": self.post_projection_entropy,
                "pre_projection_ess": self.pre_projection_ess,
                "post_projection_ess": self.post_projection_ess,
                "family_count": self.family_count,
                "family_mixture": self.family_mixture,
                "regularization_applied": self.regularization_applied,
                "entropy_after": self.post_projection_entropy,
                "effective_sample_size_after": self.post_projection_ess,
                "semantic_collapse": self.semantic_collapse,
            }
        )
        return payload


class FamilyDiversityProgramPosterior(ChannelCalibratedProgramPosterior):
    """Channel-calibrated posterior projected away from surprise collapse."""

    def __init__(
        self,
        *,
        update_policy: FamilyDiversityPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            update_policy=update_policy or FamilyDiversityPolicy.legacy(),
            **kwargs,
        )
        self.update_policy: FamilyDiversityPolicy
        self.last_update_diagnostics: FamilyDiversityDiagnostics | None = None
        self._family_projections = 0

    def seed(
        self,
        programs: tuple[
            AssembledProgram | JointProgramHypothesis,
            ...,
        ]
        | list[AssembledProgram | JointProgramHypothesis],
        *,
        initial_state: AbstractState | None = None,
    ) -> None:
        super().seed(programs, initial_state=initial_state)

    def add_programs(
        self,
        programs: tuple[
            AssembledProgram | JointProgramHypothesis,
            ...,
        ]
        | list[AssembledProgram | JointProgramHypothesis],
        *,
        initial_state: AbstractState | None = None,
    ) -> None:
        super().add_programs(programs, initial_state=initial_state)
        diagnostics = self.last_update_diagnostics
        if diagnostics is None:
            return
        base = diagnostics.base
        projection = self._project_if_needed(
            raw_surprise=base.raw_mixture_surprise,
        )
        self.last_update_diagnostics = self._diagnostics(
            base,
            projection,
        )

    def observe(
        self,
        evidence: ObservedTransition,
        *,
        allow_repair: bool = True,
    ) -> FamilyDiversityDiagnostics | None:
        base = super().observe(evidence, allow_repair=False)
        if base is None:
            return None
        projection = self._project_if_needed(
            raw_surprise=base.raw_mixture_surprise,
        )
        diagnostics = self._diagnostics(base, projection)
        self.last_update_diagnostics = diagnostics
        if allow_repair and self._needs_repair():
            self.repair(evidence)
        return diagnostics

    def snapshot(
        self,
        *,
        maximum_programs: int | None = 8,
    ) -> dict[str, Any]:
        payload = dict(super().snapshot(maximum_programs=maximum_programs))
        payload["family_projections"] = self._family_projections
        payload["family_mixture"] = self.update_policy.family_mixture
        payload["last_update"] = (
            None
            if self.last_update_diagnostics is None
            else self.last_update_diagnostics.to_dict()
        )
        return payload

    def _project_if_needed(
        self,
        *,
        raw_surprise: float,
    ) -> dict[str, Any]:
        pre_entropy = self.normalized_entropy
        pre_ess = self.effective_sample_size
        family_count = len(
            {particle.program.semantic_family for particle in self._particles}
        )
        triggered = (
            self.update_policy.family_mixture > 0.0
            and family_count > 1
            and pre_entropy
            < self.update_policy.semantic_collapse_entropy_maximum
            and float(raw_surprise)
            > self.update_policy.semantic_collapse_surprise_minimum
        )
        if triggered:
            self._project_family_mixture(self.update_policy.family_mixture)
            self._family_projections += 1
        post_entropy = self.normalized_entropy
        post_ess = self.effective_sample_size
        collapse = (
            post_entropy < self.update_policy.semantic_collapse_entropy_maximum
            and float(raw_surprise)
            > self.update_policy.semantic_collapse_surprise_minimum
        )
        return {
            "pre_entropy": pre_entropy,
            "post_entropy": post_entropy,
            "pre_ess": pre_ess,
            "post_ess": post_ess,
            "family_count": family_count,
            "triggered": triggered,
            "collapse": collapse,
        }

    def _project_family_mixture(self, mixture: float) -> None:
        if not self._particles:
            return
        by_family: dict[tuple[str, str], list[CalibratedProgramParticle]] = (
            defaultdict(list)
        )
        for particle in self._particles:
            by_family[particle.program.semantic_family].append(particle)
        if len(by_family) <= 1:
            return
        family_count = len(by_family)
        prior_reference: dict[str, float] = {}
        for particles in by_family.values():
            normalizer = _logsumexp(item.log_prior for item in particles)
            for particle in particles:
                prior_reference[particle.program.canonical_hash] = (
                    math.exp(particle.log_prior - normalizer) / family_count
                )
        absolute_normalizer = _logsumexp(
            particle.log_joint for particle in self._particles
        )
        projected = []
        for particle in self._particles:
            current = particle.probability
            reference = prior_reference[particle.program.canonical_hash]
            probability = (1.0 - mixture) * current + mixture * reference
            log_probability = math.log(max(1e-300, probability))
            projected.append(
                replace(
                    particle,
                    log_weight=log_probability,
                    log_joint=absolute_normalizer + log_probability,
                )
            )
        self._particles = projected

    def _diagnostics(
        self,
        base: ChannelPosteriorUpdateDiagnostics,
        projection: dict[str, Any],
    ) -> FamilyDiversityDiagnostics:
        return FamilyDiversityDiagnostics(
            base=base,
            pre_projection_entropy=float(projection["pre_entropy"]),
            post_projection_entropy=float(projection["post_entropy"]),
            pre_projection_ess=float(projection["pre_ess"]),
            post_projection_ess=float(projection["post_ess"]),
            family_count=int(projection["family_count"]),
            family_mixture=self.update_policy.family_mixture,
            regularization_applied=bool(projection["triggered"]),
            semantic_collapse=bool(projection["collapse"]),
        )


__all__ = [
    "T8_6C_POLICIES",
    "FamilyDiversityDiagnostics",
    "FamilyDiversityPolicy",
    "FamilyDiversityProgramPosterior",
]
