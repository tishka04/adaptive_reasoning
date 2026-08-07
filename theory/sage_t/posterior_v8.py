"""T8.6g minimum-KL family-mass projection under an entropy constraint."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .posterior import _logsumexp
from .posterior_v2 import CalibratedProgramParticle
from .posterior_v3 import CHANNELS, ChannelPosteriorUpdateDiagnostics
from .posterior_v6 import (
    AdaptiveFamilyFloorDiagnostics,
    AdaptiveFamilyFloorPolicy,
    AdaptiveFamilyFloorProgramPosterior,
)

FamilyKey = tuple[str, str]


@dataclass(frozen=True)
class MinimumKLFamilyFloorPolicy(AdaptiveFamilyFloorPolicy):
    """Entropy floor with a cap on family-mass total variation."""

    maximum_family_total_variation: float = 0.02

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < float(self.maximum_family_total_variation) <= 0.25:
            raise ValueError("maximum family total variation must be in (0, 0.25]")

    @classmethod
    def legacy(cls) -> MinimumKLFamilyFloorPolicy:
        return cls(
            name="legacy",
            channel_temperatures=tuple((channel, 1.0) for channel in CHANNELS),
        )

    @classmethod
    def terminal_tempered_20(cls) -> MinimumKLFamilyFloorPolicy:
        return cls(
            name="terminal_tempered_20",
            channel_temperatures=tuple(
                (channel, 0.20 if channel == "terminal" else 1.0)
                for channel in CHANNELS
            ),
        )

    @classmethod
    def minimum_kl_challenger(cls) -> MinimumKLFamilyFloorPolicy:
        return cls(
            name="terminal_tempered_20_family_floor_0501_minimum_kl",
            channel_temperatures=tuple(
                (channel, 0.20 if channel == "terminal" else 1.0)
                for channel in CHANNELS
            ),
            entropy_floor=0.0501,
            maximum_family_mixture=0.02,
            maximum_family_total_variation=0.02,
        )

    def with_repair_v2(self) -> MinimumKLFamilyFloorPolicy:
        return replace(
            self,
            name=f"{self.name}_repair_v2",
            repair_parent_limit=2,
            repair_child_limit=8,
            repair_survivor_limit=4,
            incremental_repair=True,
        )


T8_6G_POLICIES: Mapping[str, MinimumKLFamilyFloorPolicy] = {
    policy.name: policy
    for policy in (
        MinimumKLFamilyFloorPolicy.legacy(),
        MinimumKLFamilyFloorPolicy.terminal_tempered_20(),
        MinimumKLFamilyFloorPolicy.minimum_kl_challenger(),
    )
}


@dataclass(frozen=True)
class MinimumKLFamilyFloorDiagnostics(AdaptiveFamilyFloorDiagnostics):
    projection_alpha: float
    family_total_variation: float
    projection_kl: float

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "projection_alpha": self.projection_alpha,
                "family_total_variation": self.family_total_variation,
                "projection_kl": self.projection_kl,
            }
        )
        return payload


class MinimumKLFamilyFloorProgramPosterior(
    AdaptiveFamilyFloorProgramPosterior
):
    """I-project family masses while holding within-family posteriors fixed."""

    def __init__(
        self,
        *,
        update_policy: MinimumKLFamilyFloorPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            update_policy=(update_policy or MinimumKLFamilyFloorPolicy.legacy()),
            **kwargs,
        )
        self.update_policy: MinimumKLFamilyFloorPolicy
        self.last_update_diagnostics: MinimumKLFamilyFloorDiagnostics | None = None

    def _project_if_needed(
        self,
        *,
        raw_surprise: float,
    ) -> dict[str, Any]:
        pre_entropy = self.normalized_entropy
        pre_ess = self.effective_sample_size
        model = self._family_model()
        family_count = len(model["families"])
        triggered = (
            self.update_policy.entropy_floor > 0.0
            and family_count > 1
            and pre_entropy
            < self.update_policy.semantic_collapse_entropy_maximum
            and float(raw_surprise)
            > self.update_policy.semantic_collapse_surprise_minimum
        )
        alpha = 1.0
        total_variation = 0.0
        projection_kl = 0.0
        if triggered:
            alpha = self._minimum_kl_alpha(model)
            solution = self._family_solution(model, alpha)
            total_variation = float(solution["total_variation"])
            projection_kl = float(solution["kl"])
            self._apply_family_solution(model, solution)
            self._family_projections += 1
        post_entropy = self.normalized_entropy
        post_ess = self.effective_sample_size
        collapse = (
            post_entropy < self.update_policy.semantic_collapse_entropy_maximum
            and float(raw_surprise)
            > self.update_policy.semantic_collapse_surprise_minimum
        )
        floor_reached = (
            not triggered
            or post_entropy + self.update_policy.projection_tolerance
            >= self.update_policy.entropy_floor
        )
        return {
            "pre_entropy": pre_entropy,
            "post_entropy": post_entropy,
            "pre_ess": pre_ess,
            "post_ess": post_ess,
            "family_count": family_count,
            "triggered": triggered,
            "collapse": collapse,
            "applied_mixture": total_variation,
            "floor_reached": floor_reached,
            "projection_alpha": alpha,
            "family_total_variation": total_variation,
            "projection_kl": projection_kl,
        }

    def _family_model(self) -> dict[str, Any]:
        grouped: dict[FamilyKey, list[CalibratedProgramParticle]] = defaultdict(list)
        for particle in self._particles:
            grouped[particle.program.semantic_family].append(particle)
        families = tuple(sorted(grouped))
        log_mass: dict[FamilyKey, float] = {}
        conditional_entropy: dict[FamilyKey, float] = {}
        particle_family: dict[str, FamilyKey] = {}
        conditional_log_probability: dict[str, float] = {}
        for family in families:
            particles = grouped[family]
            normalizer = _logsumexp(item.log_weight for item in particles)
            log_mass[family] = normalizer
            entropy = 0.0
            for particle in particles:
                conditional_log = particle.log_weight - normalizer
                conditional = math.exp(conditional_log)
                entropy -= conditional * conditional_log
                particle_family[particle.program.canonical_hash] = family
                conditional_log_probability[
                    particle.program.canonical_hash
                ] = conditional_log
            conditional_entropy[family] = entropy
        return {
            "families": families,
            "log_mass": log_mass,
            "conditional_entropy": conditional_entropy,
            "particle_family": particle_family,
            "conditional_log_probability": conditional_log_probability,
        }

    def _family_solution(
        self,
        model: Mapping[str, Any],
        alpha: float,
    ) -> dict[str, Any]:
        families: tuple[FamilyKey, ...] = model["families"]
        log_mass: Mapping[FamilyKey, float] = model["log_mass"]
        conditional_entropy: Mapping[FamilyKey, float] = model[
            "conditional_entropy"
        ]
        logits = {
            family: (
                alpha * log_mass[family]
                + (1.0 - alpha) * conditional_entropy[family]
            )
            for family in families
        }
        normalizer = _logsumexp(logits.values())
        log_projected = {
            family: logits[family] - normalizer for family in families
        }
        projected = {
            family: math.exp(log_projected[family]) for family in families
        }
        current = {family: math.exp(log_mass[family]) for family in families}
        entropy = -sum(
            projected[family] * log_projected[family] for family in families
        ) + sum(
            projected[family] * conditional_entropy[family]
            for family in families
        )
        normalized_entropy = (
            0.0
            if len(self._particles) <= 1
            else entropy / math.log(len(self._particles))
        )
        total_variation = 0.5 * sum(
            abs(projected[family] - current[family]) for family in families
        )
        kl = sum(
            projected[family]
            * (log_projected[family] - log_mass[family])
            for family in families
            if projected[family] > 0.0
        )
        return {
            "log_projected": log_projected,
            "normalized_entropy": normalized_entropy,
            "total_variation": total_variation,
            "kl": max(0.0, kl),
        }

    def _minimum_kl_alpha(self, model: Mapping[str, Any]) -> float:
        target = self.update_policy.entropy_floor
        tolerance = self.update_policy.projection_tolerance
        maximum_tv = self.update_policy.maximum_family_total_variation

        maximum_entropy = self._family_solution(model, 0.0)
        if float(maximum_entropy["normalized_entropy"]) < target:
            required_alpha = 0.0
        else:
            lower = 0.0
            upper = 1.0
            while upper - lower > tolerance:
                candidate = (lower + upper) / 2.0
                entropy = float(
                    self._family_solution(model, candidate)["normalized_entropy"]
                )
                if entropy >= target:
                    lower = candidate
                else:
                    upper = candidate
            required_alpha = lower

        at_zero = self._family_solution(model, 0.0)
        if float(at_zero["total_variation"]) <= maximum_tv:
            cap_alpha = 0.0
        else:
            lower = 0.0
            upper = 1.0
            while upper - lower > tolerance:
                candidate = (lower + upper) / 2.0
                variation = float(
                    self._family_solution(model, candidate)["total_variation"]
                )
                if variation > maximum_tv:
                    lower = candidate
                else:
                    upper = candidate
            cap_alpha = upper
        return max(required_alpha, cap_alpha)

    def _apply_family_solution(
        self,
        model: Mapping[str, Any],
        solution: Mapping[str, Any],
    ) -> None:
        log_projected: Mapping[FamilyKey, float] = solution["log_projected"]
        particle_family: Mapping[str, FamilyKey] = model["particle_family"]
        conditional_log_probability: Mapping[str, float] = model[
            "conditional_log_probability"
        ]
        absolute_normalizer = _logsumexp(
            particle.log_joint for particle in self._particles
        )
        projected = []
        for particle in self._particles:
            program_hash = particle.program.canonical_hash
            family = particle_family[program_hash]
            log_probability = (
                log_projected[family]
                + conditional_log_probability[program_hash]
            )
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
    ) -> MinimumKLFamilyFloorDiagnostics:
        return MinimumKLFamilyFloorDiagnostics(
            base=base,
            pre_projection_entropy=float(projection["pre_entropy"]),
            post_projection_entropy=float(projection["post_entropy"]),
            pre_projection_ess=float(projection["pre_ess"]),
            post_projection_ess=float(projection["post_ess"]),
            family_count=int(projection["family_count"]),
            family_mixture=0.0,
            regularization_applied=bool(projection["triggered"]),
            semantic_collapse=bool(projection["collapse"]),
            entropy_floor=self.update_policy.entropy_floor,
            maximum_family_mixture=self.update_policy.maximum_family_mixture,
            applied_family_mixture=float(projection["applied_mixture"]),
            floor_reached=bool(projection["floor_reached"]),
            projection_alpha=float(projection["projection_alpha"]),
            family_total_variation=float(
                projection["family_total_variation"]
            ),
            projection_kl=float(projection["projection_kl"]),
        )


__all__ = [
    "T8_6G_POLICIES",
    "MinimumKLFamilyFloorDiagnostics",
    "MinimumKLFamilyFloorPolicy",
    "MinimumKLFamilyFloorProgramPosterior",
]
