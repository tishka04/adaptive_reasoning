"""Promoted online relational rules backed only by independent live evidence.

Generic predictions start as experiment candidates.  They become reusable
rules only after the same observable outcome has been reproduced in several
distinct live contexts.  Priors and trace support never enter the promotion
counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple

from .epistemic_metrics import HypothesisRecord, HypothesisStatus
from .generic_discriminating_experiment_designer import DiscriminatingPrediction


def promoted_relational_rule_key(prediction_key: str) -> str:
    """Return the stable GameTheory key for a promoted prediction."""
    return f"promoted::{str(prediction_key)}"


@dataclass
class PromotedRelationalRule:
    """A live-confirmed, revisable action-to-relational-effect rule."""

    action: str
    family: str
    predicate: str
    source_color: int
    target_color: int | None
    expected_outcome: str
    source_prediction_key: str
    support: int = 0
    contradictions: int = 0
    experiments_spent: int = 0
    independent_contexts: set[str] = field(default_factory=set)
    functional_successes: int = 0
    level_successes: int = 0
    visual_only_outcomes: int = 0
    minimum_support: int = 2
    minimum_independent_contexts: int = 2
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    def __post_init__(self) -> None:
        self.action = str(self.action).upper()
        self.family = str(self.family or "relation").strip().lower()
        self.predicate = str(self.predicate or self.family).strip().lower()
        self.source_color = int(self.source_color)
        if self.target_color is not None:
            self.target_color = int(self.target_color)
        self.expected_outcome = str(self.expected_outcome)
        self.independent_contexts = {
            str(context) for context in self.independent_contexts if str(context)
        }
        self._recompute_status()

    @property
    def key(self) -> str:
        return promoted_relational_rule_key(self.source_prediction_key)

    @property
    def confidence(self) -> float:
        total = self.support + self.contradictions
        return 0.0 if total <= 0 else self.support / total

    @property
    def context_count(self) -> int:
        return len(self.independent_contexts)

    @property
    def goal_relevant(self) -> bool:
        """Whether the effect has a directed postcondition worth pursuing."""
        if self.family == "color_transform":
            return self.target_color is not None and self.expected_outcome == (
                f"{self.source_color}->{self.target_color}"
            )
        if self.family == "relation":
            return self.target_color is not None and self.expected_outcome in {
                "appears",
                "broken",
                "preserved",
            }
        return False

    @property
    def pair_colors(self) -> Tuple[int, int] | None:
        if self.target_color is None:
            return None
        return (self.source_color, self.target_color)

    def observe_application(
        self,
        *,
        expected_outcome_observed: bool,
        functional_progress: bool,
        level_progress: bool,
        visual_change: bool,
        context_signature: str = "",
    ) -> None:
        """Revise the rule after one option application in the live loop."""
        if context_signature:
            self.independent_contexts.add(str(context_signature))
        if expected_outcome_observed:
            self.support += 1
        else:
            self.contradictions += 1
        if functional_progress:
            self.functional_successes += 1
        if level_progress:
            self.level_successes += 1
        if visual_change and not functional_progress:
            self.visual_only_outcomes += 1
        self._recompute_status()

    def _recompute_status(self) -> None:
        minimum_support = max(1, int(self.minimum_support))
        minimum_contexts = max(1, int(self.minimum_independent_contexts))
        if (
            self.support >= minimum_support
            and self.context_count >= minimum_contexts
            and self.confidence >= 0.60
        ):
            self.status = HypothesisStatus.CONFIRMED
            return
        if self.contradictions >= 2 and self.confidence < 0.40:
            self.status = HypothesisStatus.REFUTED
            return
        self.status = HypothesisStatus.UNRESOLVED

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=(
                f"{self.action} produces {self.family}/{self.predicate}="
                f"{self.expected_outcome} for source color {self.source_color}"
            ),
            status=self.status,
            support=self.support,
            contradictions=self.contradictions,
            experiments_spent=self.experiments_spent,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source_prediction_key": self.source_prediction_key,
            "action": self.action,
            "family": self.family,
            "predicate": self.predicate,
            "source_color": self.source_color,
            "target_color": self.target_color,
            "expected_outcome": self.expected_outcome,
            "support": self.support,
            "contradictions": self.contradictions,
            "experiments_spent": self.experiments_spent,
            "independent_contexts": self.context_count,
            "confidence": round(self.confidence, 4),
            "functional_successes": self.functional_successes,
            "level_successes": self.level_successes,
            "visual_only_outcomes": self.visual_only_outcomes,
            "goal_relevant": self.goal_relevant,
            "status": self.status.value,
        }

    @classmethod
    def from_prediction(
        cls,
        prediction: DiscriminatingPrediction,
        *,
        support: int,
        contradictions: int,
        experiments_spent: int,
        independent_contexts: Iterable[str],
        minimum_support: int = 2,
        minimum_independent_contexts: int = 2,
    ) -> "PromotedRelationalRule":
        """Build a rule from live evidence; prediction priors are ignored."""
        return cls(
            action=prediction.action,
            family=prediction.normalized_family,
            predicate=prediction.predicate_name,
            source_color=prediction.source_color,
            target_color=prediction.target_color,
            expected_outcome=prediction.outcome,
            source_prediction_key=prediction.key,
            support=max(0, int(support)),
            contradictions=max(0, int(contradictions)),
            experiments_spent=max(0, int(experiments_spent)),
            independent_contexts=set(independent_contexts),
            minimum_support=max(1, int(minimum_support)),
            minimum_independent_contexts=max(
                1,
                int(minimum_independent_contexts),
            ),
        )


__all__ = [
    "PromotedRelationalRule",
    "promoted_relational_rule_key",
]
