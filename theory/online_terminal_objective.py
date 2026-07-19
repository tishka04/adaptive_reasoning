"""Online terminal-goal grounding for promoted relational options.

Promoted rules describe mechanics: an action reproducibly changes a colour or
establishes a relation.  This module deliberately keeps that evidence separate
from goal evidence.  A mechanical effect becomes terminally supported only
when a measured objective-distance reduction is followed by an observed level
completion or win in the same short causal window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from v3.schemas import GameObservation

from .online_relational_option import relation_holds
from .promoted_relational_rule import PromotedRelationalRule


WIN_STATES = {"WIN", "WON", "VICTORY"}


class TerminalObjectiveStatus(str, Enum):
    """Epistemic status of a candidate terminal objective."""

    CANDIDATE = "candidate"
    TERMINAL_SUPPORTED = "terminal_supported"
    REFUTED = "refuted"


@dataclass
class TerminalObjectiveHypothesis:
    """A measurable goal candidate kept distinct from its mechanic rules."""

    objective_id: str
    family: str
    source_color: int
    target_color: int | None = None
    predicate: str = ""
    supporting_rule_keys: set[str] = field(default_factory=set)
    terminal_support: int = 0
    terminal_contradictions: int = 0
    probe_actions: int = 0
    grounded_actions: int = 0
    distance_reductions: int = 0
    total_distance_reduction: float = 0.0
    nonterminal_completions: int = 0
    terminal_contexts: set[str] = field(default_factory=set)
    minimum_terminal_support: int = 1

    @property
    def status(self) -> TerminalObjectiveStatus:
        if self.terminal_support >= max(1, int(self.minimum_terminal_support)):
            return TerminalObjectiveStatus.TERMINAL_SUPPORTED
        if self.terminal_contradictions >= 2 and self.terminal_support == 0:
            return TerminalObjectiveStatus.REFUTED
        return TerminalObjectiveStatus.CANDIDATE

    @property
    def terminal_confidence(self) -> float:
        total = self.terminal_support + self.terminal_contradictions
        return 0.0 if total <= 0 else self.terminal_support / total

    def distance(self, observation: GameObservation) -> float | None:
        """Return a live, directional deficit; ``None`` means unmeasurable."""
        grid = np.asarray(observation.raw_grid, dtype=np.int32)
        source_count = int(np.sum(grid == int(self.source_color)))
        if self.family == "transform_color":
            return float(source_count)
        if self.family == "establish_relation":
            if source_count <= 0 or self.target_color is None:
                return None
            target_count = int(np.sum(grid == int(self.target_color)))
            if target_count <= 0:
                # Missing the partner and therefore the relation is a larger
                # deficit than merely having to establish the relation.
                return 2.0
            return 0.0 if relation_holds(
                observation,
                self.predicate,
                self.source_color,
                self.target_color,
            ) else 1.0
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "family": self.family,
            "source_color": self.source_color,
            "target_color": self.target_color,
            "predicate": self.predicate,
            "supporting_rule_keys": sorted(self.supporting_rule_keys),
            "status": self.status.value,
            "terminal_support": self.terminal_support,
            "terminal_contradictions": self.terminal_contradictions,
            "terminal_confidence": round(self.terminal_confidence, 4),
            "probe_actions": self.probe_actions,
            "grounded_actions": self.grounded_actions,
            "distance_reductions": self.distance_reductions,
            "total_distance_reduction": round(
                float(self.total_distance_reduction), 4
            ),
            "nonterminal_completions": self.nonterminal_completions,
            "independent_terminal_contexts": len(self.terminal_contexts),
        }


@dataclass
class ObjectiveTransitionEvidence:
    """Recent observed reduction eligible for later terminal credit."""

    objective_id: str
    rule_key: str
    transition_index: int
    context_signature: str
    distance_before: float
    distance_after: float
    credited: bool = False

    @property
    def reduction(self) -> float:
        return max(0.0, self.distance_before - self.distance_after)

    @property
    def completed(self) -> bool:
        return self.distance_before > 0.0 and self.distance_after <= 0.0


@dataclass(frozen=True)
class TerminalObjectiveAssessment:
    """Selection-time assessment of one rule-linked goal hypothesis."""

    objective_id: str
    status: TerminalObjectiveStatus
    distance: float | None
    selectable: bool
    is_probe: bool
    priority: float
    reason: str


class OnlineTerminalObjectiveStore:
    """Learn which measurable option postconditions are actual level goals."""

    def __init__(
        self,
        *,
        max_probe_actions_per_objective: int = 2,
        max_probe_actions_total: int = 16,
        terminal_credit_window: int = 6,
        minimum_terminal_support: int = 1,
    ) -> None:
        self.max_probe_actions_per_objective = max(
            0, int(max_probe_actions_per_objective)
        )
        self.max_probe_actions_total = max(0, int(max_probe_actions_total))
        self.terminal_credit_window = max(0, int(terminal_credit_window))
        self.minimum_terminal_support = max(1, int(minimum_terminal_support))
        self._objectives: Dict[str, TerminalObjectiveHypothesis] = {}
        self._recent_reductions: List[ObjectiveTransitionEvidence] = []
        self._probe_contexts: set[Tuple[str, int, int]] = set()
        self._transition_index = 0
        self._branch_index = 0
        self._probe_actions_total = 0
        self._terminal_events = 0
        self._credited_terminal_events = 0

    def start_branch(self) -> None:
        """Separate reset-local probe contexts while retaining learned goals."""
        # A reset closes the previous causal branch.  A completed local goal
        # that never reached a terminal signal must not silently disappear.
        self._reject_recent_completions()
        self._branch_index += 1
        self._recent_reductions.clear()

    def ensure_from_rule(
        self,
        rule: PromotedRelationalRule,
    ) -> TerminalObjectiveHypothesis | None:
        """Create a candidate goal from a mechanic without treating it as proof."""
        descriptor = _objective_descriptor(rule)
        if descriptor is None:
            return None
        objective_id, family = descriptor
        objective = self._objectives.get(objective_id)
        if objective is None:
            objective = TerminalObjectiveHypothesis(
                objective_id=objective_id,
                family=family,
                source_color=int(rule.source_color),
                target_color=(
                    None if rule.target_color is None else int(rule.target_color)
                ),
                predicate=str(rule.predicate),
                minimum_terminal_support=self.minimum_terminal_support,
            )
            self._objectives[objective_id] = objective
        objective.supporting_rule_keys.add(rule.key)
        return objective

    def seed_rules(
        self,
        rules: Iterable[PromotedRelationalRule],
    ) -> None:
        for rule in rules:
            self.ensure_from_rule(rule)

    def objective(self, objective_id: str) -> TerminalObjectiveHypothesis | None:
        return self._objectives.get(str(objective_id))

    def assess_rule(
        self,
        rule: PromotedRelationalRule,
        observation: GameObservation,
    ) -> TerminalObjectiveAssessment | None:
        objective = self.ensure_from_rule(rule)
        if objective is None:
            return None
        distance = objective.distance(observation)
        if distance is None:
            return TerminalObjectiveAssessment(
                objective.objective_id,
                objective.status,
                None,
                False,
                False,
                float("-inf"),
                "objective distance is not measurable in the current state",
            )
        if distance <= 0.0:
            return TerminalObjectiveAssessment(
                objective.objective_id,
                objective.status,
                distance,
                False,
                False,
                float("-inf"),
                "objective postcondition is already satisfied",
            )
        if objective.status == TerminalObjectiveStatus.REFUTED:
            return TerminalObjectiveAssessment(
                objective.objective_id,
                objective.status,
                distance,
                False,
                False,
                float("-inf"),
                "objective was not terminally useful in repeated completions",
            )
        if objective.status == TerminalObjectiveStatus.TERMINAL_SUPPORTED:
            return TerminalObjectiveAssessment(
                objective.objective_id,
                objective.status,
                distance,
                True,
                False,
                10.0
                + objective.terminal_confidence
                + 1.0 / (1.0 + distance),
                "objective has observed terminal support",
            )
        context = (objective.objective_id, self._branch_index, observation.grid_hash)
        can_probe = bool(
            objective.probe_actions < self.max_probe_actions_per_objective
            and self._probe_actions_total < self.max_probe_actions_total
            and context not in self._probe_contexts
        )
        return TerminalObjectiveAssessment(
            objective.objective_id,
            objective.status,
            distance,
            can_probe,
            can_probe,
            (
                1.0
                + 1.0 / (1.0 + distance)
                - 0.25 * objective.probe_actions
            ) if can_probe else float("-inf"),
            (
                "bounded probe of an unresolved terminal objective"
                if can_probe
                else "terminal-objective probe budget exhausted"
            ),
        )

    def record_selection(
        self,
        objective_id: str,
        observation: GameObservation,
        *,
        is_probe: bool,
    ) -> None:
        objective = self._objectives[str(objective_id)]
        if is_probe:
            objective.probe_actions += 1
            self._probe_actions_total += 1
            self._probe_contexts.add(
                (objective.objective_id, self._branch_index, observation.grid_hash)
            )
        else:
            objective.grounded_actions += 1

    def observe_transition(
        self,
        update: Any,
        *,
        objective_id: str = "",
        rule_key: str = "",
        context_signature: str = "",
    ) -> Dict[str, Any]:
        """Record distance change and assign only observed terminal credit."""
        self._transition_index += 1
        expired = self._expire_old_reductions()
        evidence: ObjectiveTransitionEvidence | None = None
        objective = self._objectives.get(str(objective_id)) if objective_id else None
        if objective is not None:
            before_distance = objective.distance(update.record.obs_before)
            after_distance = objective.distance(update.record.obs_after)
            if (
                before_distance is not None
                and after_distance is not None
                and after_distance < before_distance
            ):
                evidence = ObjectiveTransitionEvidence(
                    objective_id=objective.objective_id,
                    rule_key=str(rule_key),
                    transition_index=self._transition_index,
                    context_signature=str(context_signature),
                    distance_before=float(before_distance),
                    distance_after=float(after_distance),
                )
                objective.distance_reductions += 1
                objective.total_distance_reduction += evidence.reduction
                self._recent_reductions.append(evidence)

        level_progressed = bool(
            update.record.diff.level_complete
            or update.record.obs_after.levels_completed
            > update.record.obs_before.levels_completed
        )
        won = str(update.record.obs_after.game_state).upper() in WIN_STATES
        terminal_success = bool(level_progressed or won)
        credited: List[str] = []
        if terminal_success:
            self._terminal_events += 1
            credited = self._credit_recent_reductions(update)
            if credited:
                self._credited_terminal_events += 1
            self._recent_reductions.clear()
        elif update.record.diff.game_over:
            self._reject_recent_completions()
            self._recent_reductions.clear()

        return {
            "objective_id": str(objective_id),
            "distance_before": (
                None if evidence is None else evidence.distance_before
            ),
            "distance_after": None if evidence is None else evidence.distance_after,
            "distance_reduction": 0.0 if evidence is None else evidence.reduction,
            "objective_completed": bool(evidence is not None and evidence.completed),
            "terminal_success": terminal_success,
            "terminal_credited_objectives": credited,
            "expired_nonterminal_completions": expired,
        }

    def _expire_old_reductions(self) -> int:
        retained: List[ObjectiveTransitionEvidence] = []
        expired_completions = 0
        for evidence in self._recent_reductions:
            age = self._transition_index - evidence.transition_index
            if age <= self.terminal_credit_window:
                retained.append(evidence)
                continue
            if evidence.completed and not evidence.credited:
                objective = self._objectives[evidence.objective_id]
                objective.nonterminal_completions += 1
                objective.terminal_contradictions += 1
                expired_completions += 1
        self._recent_reductions = retained
        return expired_completions

    def _credit_recent_reductions(self, update: Any) -> List[str]:
        credited: List[str] = []
        terminal_context = (
            f"level:{update.record.obs_before.levels_completed}->"
            f"{update.record.obs_after.levels_completed}:"
            f"transition:{self._transition_index}"
        )
        for evidence in self._recent_reductions:
            evidence.credited = True
            objective = self._objectives[evidence.objective_id]
            if terminal_context in objective.terminal_contexts:
                continue
            objective.terminal_contexts.add(terminal_context)
            objective.terminal_support += 1
            credited.append(objective.objective_id)
        return sorted(set(credited))

    def _reject_recent_completions(self) -> None:
        rejected: set[str] = set()
        for evidence in self._recent_reductions:
            if not evidence.completed or evidence.credited:
                continue
            if evidence.objective_id in rejected:
                continue
            rejected.add(evidence.objective_id)
            objective = self._objectives[evidence.objective_id]
            objective.nonterminal_completions += 1
            objective.terminal_contradictions += 1

    def summary(self) -> Dict[str, Any]:
        objectives = sorted(
            self._objectives.values(), key=lambda item: item.objective_id
        )
        statuses = {
            status.value: sum(item.status == status for item in objectives)
            for status in TerminalObjectiveStatus
        }
        return {
            "objectives": len(objectives),
            "statuses": statuses,
            "probe_actions": self._probe_actions_total,
            "grounded_actions": sum(item.grounded_actions for item in objectives),
            "distance_reductions": sum(
                item.distance_reductions for item in objectives
            ),
            "nonterminal_completions": sum(
                item.nonterminal_completions for item in objectives
            ),
            "terminal_events": self._terminal_events,
            "credited_terminal_events": self._credited_terminal_events,
            "terminal_supported_objectives": sum(
                item.status == TerminalObjectiveStatus.TERMINAL_SUPPORTED
                for item in objectives
            ),
            "refuted_objectives": sum(
                item.status == TerminalObjectiveStatus.REFUTED
                for item in objectives
            ),
            "recent_reductions_awaiting_terminal_credit": len(
                self._recent_reductions
            ),
            "hypotheses": [item.to_dict() for item in objectives],
        }


def _objective_descriptor(
    rule: PromotedRelationalRule,
) -> Tuple[str, str] | None:
    if (
        rule.family == "color_transform"
        and rule.target_color is not None
        and rule.expected_outcome == f"{rule.source_color}->{rule.target_color}"
    ):
        return (
            f"terminal::transform_color::{rule.source_color}_to_{rule.target_color}",
            "transform_color",
        )
    if (
        rule.family == "relation"
        and rule.target_color is not None
        and rule.expected_outcome == "appears"
    ):
        return (
            "terminal::establish_relation::"
            f"{rule.predicate}::colors{rule.source_color}_{rule.target_color}",
            "establish_relation",
        )
    # A preserved relation has no directional deficit.  It remains valid
    # mechanic knowledge but cannot be pursued as a terminal objective yet.
    return None


__all__ = [
    "OnlineTerminalObjectiveStore",
    "ObjectiveTransitionEvidence",
    "TerminalObjectiveAssessment",
    "TerminalObjectiveHypothesis",
    "TerminalObjectiveStatus",
]
