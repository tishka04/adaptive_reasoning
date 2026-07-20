"""Online anti-unification of mediated carrier signatures.

Exact scene signatures are intentionally descriptive: shape, structural role,
and relation are all retained.  A mechanism can nevertheless persist while
some of those arguments change.  This module builds the common feature pattern
only after multiple progressive transitions have been observed, then tests the
pattern against non-progress and regressive controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple


FEATURE_ORDER = (
    "change",
    "color",
    "shape",
    "area",
    "boundary",
    "player_relation",
    "multiplicity",
    "role_adjacency",
    "role_alignment",
    "proximity",
    "vertical_relation",
    "horizontal_relation",
    "relation_alignment",
    "color_relation",
)

FEATURE_WEIGHTS = {
    "change": 4,
    "color": 7,
    "shape": 4,
    "area": 2,
    "boundary": 1,
    "player_relation": 1,
    "multiplicity": 1,
    "role_adjacency": 1,
    "role_alignment": 1,
    "proximity": 1,
    "vertical_relation": 1,
    "horizontal_relation": 1,
    "relation_alignment": 1,
    "color_relation": 1,
}


@dataclass(frozen=True)
class MediatedAbstractionHypothesis:
    signature: str
    features: Tuple[Tuple[str, str], ...]
    support_contexts: int
    control_contexts: int
    regression_contexts: int
    specificity: int
    weighted_specificity: int
    objective_color_aligned: bool
    ambiguous_at_best_rank: bool = False

    @property
    def supported(self) -> bool:
        return bool(
            self.support_contexts >= 2
            and self.support_contexts > self.control_contexts
            and self.regression_contexts == 0
            and not self.ambiguous_at_best_rank
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "features": dict(self.features),
            "support_contexts": self.support_contexts,
            "control_contexts": self.control_contexts,
            "regression_contexts": self.regression_contexts,
            "specificity": self.specificity,
            "weighted_specificity": self.weighted_specificity,
            "objective_color_aligned": self.objective_color_aligned,
            "ambiguous_at_best_rank": self.ambiguous_at_best_rank,
            "supported": self.supported,
        }


def parse_mediated_candidate(signature: str) -> Dict[str, str]:
    """Decode one SAGE.8v carrier into generic structural attributes."""
    parts = str(signature).split("::")
    if len(parts) != 4 or not parts[0].startswith("mediated:"):
        return {}
    if not parts[1].startswith("entity:"):
        return {}
    if not parts[2].startswith("role:"):
        return {}
    if not parts[3].startswith("relation:"):
        return {}
    entity = parts[1][len("entity:"):].split(":")
    role = parts[2][len("role:"):].split(":")
    relation = parts[3][len("relation:"):].split(":")
    if len(entity) < 3 or len(role) < 5 or len(relation) < 5:
        return {}
    return {
        "change": parts[0][len("mediated:"):],
        "color": entity[0],
        "shape": entity[1],
        "area": entity[2],
        "boundary": role[0],
        "player_relation": role[1],
        "multiplicity": role[2],
        "role_adjacency": role[3],
        "role_alignment": role[4],
        "proximity": relation[0],
        "vertical_relation": relation[1],
        "horizontal_relation": relation[2],
        "relation_alignment": relation[3],
        "color_relation": relation[4],
    }


def induce_mediated_abstraction(
    progress_candidate_sets: Sequence[Sequence[str]],
    *,
    control_candidate_sets: Sequence[Sequence[str]] = (),
    regression_candidate_sets: Sequence[Sequence[str]] = (),
    preferred_colors: Sequence[int] = (),
    max_frontier: int = 64,
) -> MediatedAbstractionHypothesis | None:
    """Return the most precise online pattern shared by progressive contexts."""
    parsed_progress = [
        _parsed_set(items) for items in progress_candidate_sets if items
    ]
    if len(parsed_progress) < 2 or any(not items for items in parsed_progress):
        return None
    preferred = {f"color{int(color)}" for color in preferred_colors}
    aligned_progress = [
        [item for item in items if item.get("color") in preferred]
        for items in parsed_progress
    ]
    use_aligned = bool(preferred and all(aligned_progress))
    contexts = aligned_progress if use_aligned else parsed_progress

    frontier = [dict(item) for item in contexts[0]]
    for context in contexts[1:]:
        next_frontier: Dict[Tuple[Tuple[str, str], ...], Dict[str, str]] = {}
        for hypothesis in frontier:
            for candidate in context:
                common = {
                    key: value for key, value in hypothesis.items()
                    if candidate.get(key) == value
                }
                if not _causally_identifying(common):
                    continue
                key = _ordered_features(common)
                next_frontier[key] = common
        frontier = sorted(
            next_frontier.values(),
            key=_feature_rank,
            reverse=True,
        )[:max(1, int(max_frontier))]
        if not frontier:
            return None

    controls = [_parsed_set(items) for items in control_candidate_sets if items]
    regressions = [
        _parsed_set(items) for items in regression_candidate_sets if items
    ]
    hypotheses = []
    for features in frontier:
        ordered = _ordered_features(features)
        control_count = sum(
            any(_matches(features, candidate) for candidate in context)
            for context in controls
        )
        regression_count = sum(
            any(_matches(features, candidate) for candidate in context)
            for context in regressions
        )
        objective_aligned = bool(
            preferred and features.get("color") in preferred
        )
        hypotheses.append(MediatedAbstractionHypothesis(
            signature=_abstraction_signature(ordered),
            features=ordered,
            support_contexts=len(parsed_progress),
            control_contexts=control_count,
            regression_contexts=regression_count,
            specificity=len(ordered),
            weighted_specificity=sum(
                FEATURE_WEIGHTS.get(key, 1) for key, _value in ordered
            ),
            objective_color_aligned=objective_aligned,
        ))
    eligible = [
        item for item in hypotheses
        if item.support_contexts > item.control_contexts
        and item.regression_contexts == 0
    ]
    if not eligible:
        return None
    ranked = sorted(eligible, key=_hypothesis_rank, reverse=True)
    best = ranked[0]
    best_rank = _hypothesis_rank(best)
    tied = {
        item.signature for item in ranked
        if _hypothesis_rank(item) == best_rank
    }
    if len(tied) > 1:
        return MediatedAbstractionHypothesis(
            signature=best.signature,
            features=best.features,
            support_contexts=best.support_contexts,
            control_contexts=best.control_contexts,
            regression_contexts=best.regression_contexts,
            specificity=best.specificity,
            weighted_specificity=best.weighted_specificity,
            objective_color_aligned=best.objective_color_aligned,
            ambiguous_at_best_rank=True,
        )
    return best


def abstraction_matches_candidate(
    abstraction: MediatedAbstractionHypothesis,
    candidate_signature: str,
) -> bool:
    return _matches(
        dict(abstraction.features),
        parse_mediated_candidate(candidate_signature),
    )


def _parsed_set(items: Sequence[str]) -> list[Dict[str, str]]:
    result = []
    for item in items:
        parsed = parse_mediated_candidate(str(item))
        if parsed:
            result.append(parsed)
    return result


def _causally_identifying(features: Mapping[str, str]) -> bool:
    return bool(
        features.get("change")
        and any(features.get(key) for key in ("color", "shape", "area"))
    )


def _ordered_features(
    features: Mapping[str, str],
) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        (key, str(features[key]))
        for key in FEATURE_ORDER
        if key in features
    )


def _feature_rank(features: Mapping[str, str]) -> Tuple[int, int, str]:
    ordered = _ordered_features(features)
    return (
        sum(FEATURE_WEIGHTS.get(key, 1) for key, _value in ordered),
        len(ordered),
        _abstraction_signature(ordered),
    )


def _hypothesis_rank(
    hypothesis: MediatedAbstractionHypothesis,
) -> Tuple[int, int, int, int]:
    return (
        int(hypothesis.objective_color_aligned),
        hypothesis.weighted_specificity,
        hypothesis.specificity,
        hypothesis.support_contexts - hypothesis.control_contexts,
    )


def _matches(pattern: Mapping[str, str], candidate: Mapping[str, str]) -> bool:
    return bool(pattern) and all(
        candidate.get(key) == value for key, value in pattern.items()
    )


def _abstraction_signature(features: Sequence[Tuple[str, str]]) -> str:
    return "mediated-abstract::" + "::".join(
        f"{key}:{value}" for key, value in features
    )


__all__ = [
    "MediatedAbstractionHypothesis",
    "abstraction_matches_candidate",
    "induce_mediated_abstraction",
    "parse_mediated_candidate",
]
