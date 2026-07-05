"""Contextual readiness for correspondence validation options.

A weak ready predicate can be true while the state is still not actually
validating. This layer records that distinction without refuting the global
correspondence rule.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .epistemic_metrics import HypothesisStatus
from .theory_option import TheoryOptionInvocation

WEAK_READY = "weak_ready_to_validate_correspondence"
STRONG_READY = "strong_ready_to_validate_correspondence"
NO_FAILED_SAME_CONTEXT = "no_recent_failed_validation_same_context"
PREDICTED_TARGET = "predicted_correspondence_target_identified"
RELATION_SATISFIED = "source_target_relation_satisfied"

_CONTEXT_PREDICATES = {
    "active_color_pair_10_11",
    "selected_pair_exists",
    "controller_on_source",
    "controller_points_to_source",
    "recent_control_switch",
    "source_target_aligned",
    "source_target_projected_aligned",
    "source_target_relation_satisfied",
    "selected_source_matches_target_shape",
    "correspondence_count_improved_recently",
}


@dataclass(frozen=True)
class ReadinessContext:
    """Stable context key for one attempted validation state."""

    target_rule: str
    predicates: tuple[str, ...]

    @property
    def digest(self) -> str:
        raw = "|".join((self.target_rule, *self.predicates))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @property
    def key(self) -> str:
        return f"context_{self.digest}"


@dataclass
class ContextualReadinessHypothesis:
    """Claim that one weak-ready context is sufficient for validation."""

    target_rule: str
    context_key: str
    predicates: tuple[str, ...]
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    @property
    def key(self) -> str:
        return (
            f"contextual_precondition::{self.target_rule}::"
            f"{self.context_key}::ready_to_validate_correspondence"
        )

    @property
    def support(self) -> int:
        return len(self.evidence_for)

    @property
    def contradictions(self) -> int:
        return len(self.evidence_against)

    def observe(self, *, succeeded: bool, label: str) -> None:
        if succeeded:
            self.evidence_for.append(label)
        else:
            self.evidence_against.append(label)
        self._recompute_status()

    def _recompute_status(self) -> None:
        if self.support > 0 and self.support >= self.contradictions:
            self.status = HypothesisStatus.CONFIRMED
        elif self.contradictions > self.support:
            self.status = HypothesisStatus.REFUTED
        else:
            self.status = HypothesisStatus.UNRESOLVED

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "status": self.status.value,
            "support": self.support,
            "contradictions": self.contradictions,
            "predicates": list(self.predicates),
        }


@dataclass(frozen=True)
class ReadinessAssessment:
    """Annotated predicates plus the contextual hypothesis being tested."""

    context: ReadinessContext
    predicates: tuple[str, ...]
    hypothesis: ContextualReadinessHypothesis

    @property
    def weak_ready(self) -> bool:
        return WEAK_READY in self.predicates

    @property
    def strong_ready(self) -> bool:
        return STRONG_READY in self.predicates

    @property
    def failed_same_context(self) -> bool:
        return NO_FAILED_SAME_CONTEXT not in self.predicates


class ContextualReadinessDiscriminator:
    """Track weak-ready contexts that failed validation."""

    def __init__(self) -> None:
        self._hypotheses: dict[str, ContextualReadinessHypothesis] = {}

    def assess(
        self,
        predicates_present: Iterable[str],
        *,
        target_rule: str,
    ) -> ReadinessAssessment:
        base = _normalize_predicates(predicates_present)
        weak_ready = _is_weak_ready(base)
        if weak_ready:
            base.add(WEAK_READY)
        if "selected_pair_exists" in base:
            base.add(PREDICTED_TARGET)
        if "source_target_aligned" in base or RELATION_SATISFIED in base:
            base.add(RELATION_SATISFIED)

        context = readiness_context(target_rule, base)
        hypothesis = self._hypothesis_for(target_rule, context)
        if hypothesis.status != HypothesisStatus.REFUTED:
            base.add(NO_FAILED_SAME_CONTEXT)
        if (
            weak_ready
            and PREDICTED_TARGET in base
            and RELATION_SATISFIED in base
            and NO_FAILED_SAME_CONTEXT in base
        ):
            base.add(STRONG_READY)
        return ReadinessAssessment(
            context=context,
            predicates=tuple(sorted(base)),
            hypothesis=hypothesis,
        )

    def can_attempt_validation(self, assessment: ReadinessAssessment) -> bool:
        """Allow strong ready, or one discriminating weak-ready probe."""
        if assessment.strong_ready:
            return True
        if not assessment.weak_ready:
            return False
        return assessment.hypothesis.status != HypothesisStatus.REFUTED

    def observe_validation_result(
        self,
        assessment: ReadinessAssessment,
        invocation: TheoryOptionInvocation | None,
    ) -> ContextualReadinessHypothesis:
        succeeded = bool(invocation and invocation.success)
        label = (
            f"observed:{invocation.actual_action}:step{invocation.step}"
            if invocation is not None
            else "observed:missing_invocation"
        )
        assessment.hypothesis.observe(succeeded=succeeded, label=label)
        return assessment.hypothesis

    def hypotheses(self) -> List[ContextualReadinessHypothesis]:
        return list(self._hypotheses.values())

    def refuted(self) -> List[ContextualReadinessHypothesis]:
        return [
            hypothesis for hypothesis in self.hypotheses()
            if hypothesis.status == HypothesisStatus.REFUTED
        ]

    def _hypothesis_for(
        self,
        target_rule: str,
        context: ReadinessContext,
    ) -> ContextualReadinessHypothesis:
        existing = self._hypotheses.get(context.key)
        if existing is not None:
            return existing
        hypothesis = ContextualReadinessHypothesis(
            target_rule=target_rule,
            context_key=context.key,
            predicates=context.predicates,
        )
        self._hypotheses[context.key] = hypothesis
        return hypothesis


def readiness_context(
    target_rule: str,
    predicates_present: Iterable[str],
) -> ReadinessContext:
    predicates = tuple(
        sorted(
            predicate for predicate in _normalize_predicates(predicates_present)
            if predicate in _CONTEXT_PREDICATES
        )
    )
    return ReadinessContext(target_rule=str(target_rule), predicates=predicates)


def _is_weak_ready(predicates: set[str]) -> bool:
    return (
        "ready_to_validate_correspondence" in predicates
        or (
            "selected_pair_exists" in predicates
            and "controller_on_source" in predicates
            and "recent_control_switch" in predicates
        )
    )


def _normalize_predicates(predicates: Iterable[str]) -> set[str]:
    return {str(predicate or "").strip().lower() for predicate in predicates}
