"""T8.6f family projection preserving posterior within-family ratios."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace

from .posterior import _logsumexp
from .posterior_v2 import CalibratedProgramParticle
from .posterior_v6 import (
    AdaptiveFamilyFloorPolicy,
    AdaptiveFamilyFloorProgramPosterior,
)


def posterior_conditional_floor_policy() -> AdaptiveFamilyFloorPolicy:
    """Return the single pre-registered T8.6f challenger."""

    return replace(
        AdaptiveFamilyFloorPolicy.terminal_tempered_20_family_floor(0.0501),
        name="terminal_tempered_20_family_floor_0501_posterior_conditional",
    )


T8_6F_POLICIES: Mapping[str, AdaptiveFamilyFloorPolicy] = {
    policy.name: policy
    for policy in (
        AdaptiveFamilyFloorPolicy.legacy(),
        AdaptiveFamilyFloorPolicy.terminal_tempered_20(),
        posterior_conditional_floor_policy(),
    )
}


class PosteriorConditionalFamilyFloorProgramPosterior(
    AdaptiveFamilyFloorProgramPosterior
):
    """Move mass across families without perturbing ratios inside a family."""

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
            normalizer = _logsumexp(item.log_weight for item in particles)
            for particle in particles:
                conditional = math.exp(particle.log_weight - normalizer)
                reference[particle.program.canonical_hash] = (
                    conditional / family_count
                )
        return reference


__all__ = [
    "T8_6F_POLICIES",
    "PosteriorConditionalFamilyFloorProgramPosterior",
    "posterior_conditional_floor_policy",
]
