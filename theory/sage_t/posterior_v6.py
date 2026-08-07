"""T8.6e minimal collapse-triggered semantic-family entropy floor."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .posterior import _logsumexp
from .posterior_v2 import CalibratedProgramParticle
from .posterior_v3 import CHANNELS
from .posterior_v4 import (
    FamilyDiversityDiagnostics,
    FamilyDiversityPolicy,
    FamilyDiversityProgramPosterior,
)


@dataclass(frozen=True)
class AdaptiveFamilyFloorPolicy(FamilyDiversityPolicy):
    """Channel calibration plus the minimum family mixture reaching a floor."""

    entropy_floor: float = 0.0
    maximum_family_mixture: float = 0.02
    projection_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family_mixture != 0.0:
            raise ValueError("adaptive family floors do not use a fixed mixture")
        if not 0.0 <= float(self.entropy_floor) <= 0.25:
            raise ValueError("entropy floor must be in [0, 0.25]")
        if not 0.0 < float(self.maximum_family_mixture) <= 0.25:
            raise ValueError("maximum family mixture must be in (0, 0.25]")
        if not 0.0 < float(self.projection_tolerance) <= 1e-4:
            raise ValueError("projection tolerance must be in (0, 1e-4]")
        if (
            self.entropy_floor > 0.0
            and self.entropy_floor
            <= self.semantic_collapse_entropy_maximum
        ):
            raise ValueError("positive entropy floor must exceed collapse threshold")

    @classmethod
    def legacy(cls) -> AdaptiveFamilyFloorPolicy:
        return cls(
            name="legacy",
            channel_temperatures=tuple((channel, 1.0) for channel in CHANNELS),
        )

    @classmethod
    def terminal_tempered_20(cls) -> AdaptiveFamilyFloorPolicy:
        return cls(
            name="terminal_tempered_20",
            channel_temperatures=tuple(
                (channel, 0.20 if channel == "terminal" else 1.0)
                for channel in CHANNELS
            ),
        )

    @classmethod
    def terminal_tempered_20_family_floor(
        cls,
        floor: float,
    ) -> AdaptiveFamilyFloorPolicy:
        basis_points = round(10_000 * float(floor))
        return cls(
            name=f"terminal_tempered_20_family_floor_{basis_points:04d}",
            channel_temperatures=tuple(
                (channel, 0.20 if channel == "terminal" else 1.0)
                for channel in CHANNELS
            ),
            entropy_floor=float(floor),
        )

    def with_repair_v2(self) -> AdaptiveFamilyFloorPolicy:
        return replace(
            self,
            name=f"{self.name}_repair_v2",
            repair_parent_limit=2,
            repair_child_limit=8,
            repair_survivor_limit=4,
            incremental_repair=True,
        )


T8_6E_POLICIES: Mapping[str, AdaptiveFamilyFloorPolicy] = {
    policy.name: policy
    for policy in (
        AdaptiveFamilyFloorPolicy.legacy(),
        AdaptiveFamilyFloorPolicy.terminal_tempered_20(),
        AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.0501),
        AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.0525),
        AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.0550),
    )
}


@dataclass(frozen=True)
class AdaptiveFamilyFloorDiagnostics(FamilyDiversityDiagnostics):
    entropy_floor: float
    maximum_family_mixture: float
    applied_family_mixture: float
    floor_reached: bool

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "entropy_floor": self.entropy_floor,
                "maximum_family_mixture": self.maximum_family_mixture,
                "applied_family_mixture": self.applied_family_mixture,
                "floor_reached": self.floor_reached,
            }
        )
        return payload


class AdaptiveFamilyFloorProgramPosterior(FamilyDiversityProgramPosterior):
    """Raise entropy just above the registered floor after a surprise collapse."""

    def __init__(
        self,
        *,
        update_policy: AdaptiveFamilyFloorPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            update_policy=(
                update_policy or AdaptiveFamilyFloorPolicy.legacy()
            ),
            **kwargs,
        )
        self.update_policy: AdaptiveFamilyFloorPolicy
        self.last_update_diagnostics: AdaptiveFamilyFloorDiagnostics | None = None

    def snapshot(
        self,
        *,
        maximum_programs: int | None = 8,
    ) -> dict[str, Any]:
        payload = dict(super().snapshot(maximum_programs=maximum_programs))
        payload["entropy_floor"] = self.update_policy.entropy_floor
        payload["maximum_family_mixture"] = (
            self.update_policy.maximum_family_mixture
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
            self.update_policy.entropy_floor > 0.0
            and family_count > 1
            and pre_entropy
            < self.update_policy.semantic_collapse_entropy_maximum
            and float(raw_surprise)
            > self.update_policy.semantic_collapse_surprise_minimum
        )
        applied_mixture = 0.0
        if triggered:
            reference = self._family_reference_probabilities()
            applied_mixture = self._minimum_mixture_for_floor(reference)
            if applied_mixture > 0.0:
                self._apply_reference_mixture(reference, applied_mixture)
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
            "applied_mixture": applied_mixture,
            "floor_reached": floor_reached,
        }

    def _family_reference_probabilities(self) -> dict[str, float]:
        by_family: dict[tuple[str, str], list[CalibratedProgramParticle]] = (
            defaultdict(list)
        )
        for particle in self._particles:
            by_family[particle.program.semantic_family].append(particle)
        if not by_family:
            return {}
        family_count = len(by_family)
        reference: dict[str, float] = {}
        for particles in by_family.values():
            normalizer = _logsumexp(item.log_prior for item in particles)
            for particle in particles:
                reference[particle.program.canonical_hash] = (
                    math.exp(particle.log_prior - normalizer) / family_count
                )
        return reference

    def _minimum_mixture_for_floor(
        self,
        reference: Mapping[str, float],
    ) -> float:
        target = self.update_policy.entropy_floor
        if target <= self.normalized_entropy or not self._particles:
            return 0.0
        upper = self.update_policy.maximum_family_mixture
        if self._projected_entropy(reference, upper) < target:
            return upper
        lower = 0.0
        tolerance = self.update_policy.projection_tolerance
        while upper - lower > tolerance:
            candidate = (lower + upper) / 2.0
            if self._projected_entropy(reference, candidate) >= target:
                upper = candidate
            else:
                lower = candidate
        return upper

    def _projected_entropy(
        self,
        reference: Mapping[str, float],
        mixture: float,
    ) -> float:
        if len(self._particles) <= 1:
            return 0.0
        probabilities = [
            (1.0 - mixture) * particle.probability
            + mixture * reference[particle.program.canonical_hash]
            for particle in self._particles
        ]
        entropy = -sum(
            probability * math.log(max(probability, 1e-300))
            for probability in probabilities
            if probability > 0.0
        )
        return entropy / math.log(len(probabilities))

    def _apply_reference_mixture(
        self,
        reference: Mapping[str, float],
        mixture: float,
    ) -> None:
        absolute_normalizer = _logsumexp(
            particle.log_joint for particle in self._particles
        )
        projected = []
        for particle in self._particles:
            probability = (
                (1.0 - mixture) * particle.probability
                + mixture * reference[particle.program.canonical_hash]
            )
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
        base: Any,
        projection: dict[str, Any],
    ) -> AdaptiveFamilyFloorDiagnostics:
        return AdaptiveFamilyFloorDiagnostics(
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
        )


__all__ = [
    "T8_6E_POLICIES",
    "AdaptiveFamilyFloorDiagnostics",
    "AdaptiveFamilyFloorPolicy",
    "AdaptiveFamilyFloorProgramPosterior",
]
