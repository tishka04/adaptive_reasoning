"""Terminal-only online revision for actively generated goal hypotheses.

Mechanical evidence may make a goal testable, but it never makes the goal
true.  This store measures every live candidate, keeps delayed eligibility
traces, quarantines ambiguous terminal events, separates unsafe interventions
from false goals, and requires an independent terminal context or contrast
before a goal becomes exploitable.
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
    AMBIGUOUS_TERMINAL = "ambiguous_terminal"
    NEEDS_CONTRAST = "needs_contrast"
    TERMINAL_SUPPORTED = "terminal_supported"
    REFUTED = "refuted"


@dataclass
class TerminalObjectiveHypothesis:
    """A measurable goal candidate kept distinct from mechanic evidence."""

    objective_id: str
    family: str
    source_color: int | None = None
    target_color: int | None = None
    predicate: str = ""
    supporting_rule_keys: set[str] = field(default_factory=set)
    supporting_actions: set[str] = field(default_factory=set)
    generation_reasons: set[str] = field(default_factory=set)
    prior_priority: float = 0.0
    terminal_contradictions: int = 0
    probe_actions: int = 0
    grounded_actions: int = 0
    distance_reductions: int = 0
    total_distance_reduction: float = 0.0
    nonterminal_completions: int = 0
    ambiguous_terminal_events: int = 0
    terminal_contexts: set[str] = field(default_factory=set)
    terminal_interventions: set[str] = field(default_factory=set)
    unsafe_plan_failures: int = 0
    dangerous_interventions: Dict[str, int] = field(default_factory=dict)
    ablation_attempts: int = 0
    ablation_contrasts: int = 0
    ablation_contradictions: int = 0
    ablation_censored: int = 0
    minimum_independent_terminal_contexts: int = 2

    @property
    def terminal_support(self) -> int:
        return len(self.terminal_contexts)

    @property
    def status(self) -> TerminalObjectiveStatus:
        if self.terminal_support >= max(
            1, int(self.minimum_independent_terminal_contexts)
        ) or self.ablation_contrasts > 0:
            return TerminalObjectiveStatus.TERMINAL_SUPPORTED
        if self.terminal_support > 0:
            return TerminalObjectiveStatus.NEEDS_CONTRAST
        if self.terminal_contradictions >= 2:
            return TerminalObjectiveStatus.REFUTED
        if self.ambiguous_terminal_events > 0:
            return TerminalObjectiveStatus.AMBIGUOUS_TERMINAL
        return TerminalObjectiveStatus.CANDIDATE

    @property
    def terminal_confidence(self) -> float:
        positive = self.terminal_support + self.ablation_contrasts
        negative = self.terminal_contradictions + self.ablation_contradictions
        total = positive + negative
        return 0.0 if total <= 0 else positive / total

    def distance(self, observation: GameObservation) -> float | None:
        """Return a directional live deficit; ``None`` means unmeasurable."""
        source = self.source_color
        target = self.target_color
        if self.family in {"appear", "break"}:
            if source is None or target is None:
                return None
            source_present = any(obj.value == int(source) for obj in observation.objects)
            target_present = any(obj.value == int(target) for obj in observation.objects)
            if not source_present or not target_present:
                return 2.0 if self.family == "appear" else 0.0
            holds = relation_holds(
                observation,
                self.predicate,
                int(source),
                int(target),
            )
            if self.family == "appear":
                return 0.0 if holds else 1.0
            return 1.0 if holds else 0.0
        if self.family == "exhaust":
            if source is None:
                return None
            return float(sum(obj.value == int(source) for obj in observation.objects))
        if self.family == "reach":
            if target is None or observation.best_player is None:
                return None
            targets = [obj for obj in observation.objects if obj.value == int(target)]
            if not targets:
                return None
            player_row, player_col = observation.best_player.position
            cell_distance = min(
                abs(int(row) - int(player_row)) + abs(int(col) - int(player_col))
                for obj in targets
                for row, col in obj.cells
            )
            return float(max(0, cell_distance - 1))
        if self.family == "convert":
            if source is None or target is None:
                return None
            grid = np.asarray(observation.raw_grid, dtype=np.int32)
            return float(np.sum(grid == int(source)))
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "family": self.family,
            "source_color": self.source_color,
            "target_color": self.target_color,
            "predicate": self.predicate,
            "supporting_rule_keys": sorted(self.supporting_rule_keys),
            "supporting_actions": sorted(self.supporting_actions),
            "generation_reasons": sorted(self.generation_reasons),
            "prior_priority": round(float(self.prior_priority), 4),
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
            "ambiguous_terminal_events": self.ambiguous_terminal_events,
            "independent_terminal_contexts": self.terminal_support,
            "terminal_interventions": sorted(self.terminal_interventions),
            "unsafe_plan_failures": self.unsafe_plan_failures,
            "dangerous_interventions": dict(sorted(self.dangerous_interventions.items())),
            "ablation_attempts": self.ablation_attempts,
            "ablation_contrasts": self.ablation_contrasts,
            "ablation_contradictions": self.ablation_contradictions,
            "ablation_censored": self.ablation_censored,
        }


@dataclass
class ObjectiveTransitionEvidence:
    """One recent goal-distance reduction eligible for terminal credit."""

    objective_id: str
    rule_key: str
    intervention_id: str
    transition_index: int
    context_signature: str
    distance_before: float
    distance_after: float
    selected_objective: bool = False
    credited: bool = False

    @property
    def reduction(self) -> float:
        return max(0.0, self.distance_before - self.distance_after)

    @property
    def completed(self) -> bool:
        return self.distance_before > 0.0 and self.distance_after <= 0.0


@dataclass(frozen=True)
class TerminalObjectiveAssessment:
    """Selection-time assessment of one goal hypothesis."""

    objective_id: str
    status: TerminalObjectiveStatus
    distance: float | None
    selectable: bool
    is_probe: bool
    priority: float
    reason: str


@dataclass(frozen=True)
class AmbiguousTerminalEvent:
    """A terminal event preceded by several competing goal reductions."""

    transition_index: int
    objective_ids: Tuple[str, ...]
    context_signatures: Tuple[str, ...]


@dataclass
class ActiveAblation:
    """One terminal-only contrast that omits a provisionally supported goal."""

    objective_id: str
    intervention_id: str
    branch_index: int
    baseline_terminal_interventions: Tuple[str, ...] = ()
    terminal_observed: bool = False


class OnlineTerminalObjectiveStore:
    """Generate no proof: revise goal truth only from observed terminal outcomes."""

    def __init__(
        self,
        *,
        max_probe_actions_per_objective: int = 2,
        max_probe_actions_total: int = 16,
        terminal_credit_window: int = 6,
        minimum_terminal_support: int = 2,
        max_ablation_actions_per_objective: int = 1,
    ) -> None:
        self.max_probe_actions_per_objective = max(
            0, int(max_probe_actions_per_objective)
        )
        self.max_probe_actions_total = max(0, int(max_probe_actions_total))
        self.terminal_credit_window = max(0, int(terminal_credit_window))
        self.minimum_terminal_support = max(1, int(minimum_terminal_support))
        self.max_ablation_actions_per_objective = max(
            0, int(max_ablation_actions_per_objective)
        )
        self._objectives: Dict[str, TerminalObjectiveHypothesis] = {}
        self._recent_reductions: List[ObjectiveTransitionEvidence] = []
        self._ambiguous_events: List[AmbiguousTerminalEvent] = []
        self._probe_contexts: set[Tuple[str, int, int]] = set()
        self._transition_index = 0
        self._branch_index = 0
        self._probe_actions_total = 0
        self._terminal_events = 0
        self._credited_terminal_events = 0
        self._active_ablation: ActiveAblation | None = None

    @property
    def branch_index(self) -> int:
        return self._branch_index

    def start_branch(self) -> None:
        """Close one causal branch without treating censoring as terminal proof."""
        self._reject_recent_completions()
        if self._active_ablation is not None and not self._active_ablation.terminal_observed:
            objective = self._objectives.get(self._active_ablation.objective_id)
            if objective is not None:
                objective.ablation_censored += 1
        self._active_ablation = None
        self._branch_index += 1
        self._recent_reductions.clear()

    def register_generated(self, candidate: Any) -> TerminalObjectiveHypothesis:
        """Register a generated candidate while preserving all online evidence."""
        objective_id = str(candidate.objective_id)
        objective = self._objectives.get(objective_id)
        if objective is None:
            objective = TerminalObjectiveHypothesis(
                objective_id=objective_id,
                family=str(candidate.family),
                source_color=getattr(candidate, "source_color", None),
                target_color=getattr(candidate, "target_color", None),
                predicate=str(getattr(candidate, "predicate", "")),
                minimum_independent_terminal_contexts=self.minimum_terminal_support,
            )
            self._objectives[objective_id] = objective
        objective.supporting_rule_keys.update(
            str(key) for key in getattr(candidate, "supporting_rule_keys", ()) if key
        )
        objective.supporting_actions.update(
            str(action).upper()
            for action in getattr(candidate, "supporting_actions", ())
            if action
        )
        reason = str(getattr(candidate, "generation_reason", ""))
        if reason:
            objective.generation_reasons.add(reason)
        objective.prior_priority = max(
            objective.prior_priority,
            float(getattr(candidate, "prior_priority", 0.0) or 0.0),
        )
        return objective

    def register_generated_bounded(
        self,
        candidate: Any,
        *,
        max_objectives: int,
        max_per_family: int | None = None,
    ) -> TerminalObjectiveHypothesis | None:
        """Maintain a bounded bank, evicting only untouched weaker candidates."""
        objective_id = str(candidate.objective_id)
        if objective_id in self._objectives:
            return self.register_generated(candidate)
        limit = max(1, int(max_objectives))
        family = str(candidate.family)
        family_limit = (
            limit if max_per_family is None else max(1, int(max_per_family))
        )
        same_family = [
            objective
            for objective in self._objectives.values()
            if objective.family == family
        ]
        if len(self._objectives) < limit and len(same_family) < family_limit:
            return self.register_generated(candidate)
        evictable = [
            objective
            for objective in self._objectives.values()
            if objective.status == TerminalObjectiveStatus.CANDIDATE
            and objective.probe_actions == 0
            and objective.distance_reductions == 0
            and objective.terminal_support == 0
            and (
                len(same_family) < family_limit
                or objective.family == family
            )
        ]
        if not evictable:
            return None
        weakest = min(
            evictable,
            key=lambda item: (item.prior_priority, item.objective_id),
        )
        candidate_priority = float(getattr(candidate, "prior_priority", 0.0) or 0.0)
        if candidate_priority <= weakest.prior_priority:
            return None
        del self._objectives[weakest.objective_id]
        return self.register_generated(candidate)

    def ensure_from_rule(
        self,
        rule: PromotedRelationalRule,
    ) -> TerminalObjectiveHypothesis | None:
        descriptor = _objective_descriptor(rule)
        if descriptor is None:
            return None
        objective_id, family = descriptor
        candidate = _RuleGoalCandidate(
            objective_id=objective_id,
            family=family,
            source_color=int(rule.source_color),
            target_color=(
                None if rule.target_color is None else int(rule.target_color)
            ),
            predicate=str(rule.predicate),
            supporting_rule_keys=(rule.key,),
            supporting_actions=(rule.action,),
            generation_reason="directed_confirmed_mechanic",
            prior_priority=float(rule.confidence),
        )
        return self.register_generated(candidate)

    def seed_rules(self, rules: Iterable[PromotedRelationalRule]) -> None:
        for rule in rules:
            self.ensure_from_rule(rule)

    def objectives(self) -> List[TerminalObjectiveHypothesis]:
        return sorted(self._objectives.values(), key=lambda item: item.objective_id)

    def objective(self, objective_id: str) -> TerminalObjectiveHypothesis | None:
        return self._objectives.get(str(objective_id))

    def assess_rule(
        self,
        rule: PromotedRelationalRule,
        observation: GameObservation,
    ) -> TerminalObjectiveAssessment | None:
        objective = self.ensure_from_rule(rule)
        return None if objective is None else self.assess_objective(
            objective,
            observation,
        )

    def assess_objective(
        self,
        objective: TerminalObjectiveHypothesis,
        observation: GameObservation,
    ) -> TerminalObjectiveAssessment:
        distance = objective.distance(observation)
        if distance is None:
            return _blocked_assessment(objective, None, "objective is unmeasurable")
        if distance <= 0.0:
            return _blocked_assessment(
                objective, distance, "objective postcondition is already satisfied"
            )
        if objective.status == TerminalObjectiveStatus.REFUTED:
            return _blocked_assessment(objective, distance, "objective was refuted")
        if objective.status == TerminalObjectiveStatus.TERMINAL_SUPPORTED:
            return TerminalObjectiveAssessment(
                objective.objective_id,
                objective.status,
                distance,
                True,
                False,
                20.0 + objective.terminal_confidence + 1.0 / (1.0 + distance),
                "objective has independent observed terminal support",
            )
        context = (objective.objective_id, self._branch_index, observation.grid_hash)
        can_probe = bool(
            objective.probe_actions < self.max_probe_actions_per_objective
            and self._probe_actions_total < self.max_probe_actions_total
            and context not in self._probe_contexts
        )
        status_bonus = {
            TerminalObjectiveStatus.NEEDS_CONTRAST: 8.0,
            TerminalObjectiveStatus.AMBIGUOUS_TERMINAL: 5.0,
            TerminalObjectiveStatus.CANDIDATE: 1.0,
        }.get(objective.status, 0.0)
        return TerminalObjectiveAssessment(
            objective.objective_id,
            objective.status,
            distance,
            can_probe,
            can_probe,
            (
                status_bonus
                + objective.prior_priority
                + 1.0 / (1.0 + distance)
                - 0.25 * objective.probe_actions
            ) if can_probe else float("-inf"),
            (
                "terminal contrast required after provisional/ambiguous evidence"
                if objective.status in {
                    TerminalObjectiveStatus.NEEDS_CONTRAST,
                    TerminalObjectiveStatus.AMBIGUOUS_TERMINAL,
                }
                else "bounded online probe of a generated goal"
            ) if can_probe else "goal probe budget exhausted",
        )

    def record_selection(
        self,
        objective_id: str,
        observation: GameObservation,
        *,
        is_probe: bool,
        intervention_id: str = "",
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
        if intervention_id and self.intervention_is_unsafe(
            objective.objective_id, intervention_id
        ):
            raise ValueError("attempted a quarantined unsafe goal intervention")

    def begin_ablation(self, objective_id: str, intervention_id: str) -> None:
        objective = self._objectives[str(objective_id)]
        objective.ablation_attempts += 1
        self._active_ablation = ActiveAblation(
            objective_id=objective.objective_id,
            intervention_id=str(intervention_id),
            branch_index=self._branch_index,
            baseline_terminal_interventions=tuple(
                sorted(objective.terminal_interventions)
            ),
        )

    def ablation_targets(self) -> List[TerminalObjectiveHypothesis]:
        return [
            objective
            for objective in self.objectives()
            if objective.status == TerminalObjectiveStatus.NEEDS_CONTRAST
            and objective.ablation_attempts
            < self.max_ablation_actions_per_objective
        ]

    def intervention_is_unsafe(
        self,
        objective_id: str,
        intervention_id: str,
        *,
        minimum_failures: int = 1,
    ) -> bool:
        objective = self.objective(objective_id)
        if objective is None:
            return False
        return objective.dangerous_interventions.get(str(intervention_id), 0) >= max(
            1, int(minimum_failures)
        )

    def observe_transition(
        self,
        update: Any,
        *,
        objective_id: str = "",
        rule_key: str = "",
        intervention_id: str = "",
        ablation_of_objective_id: str = "",
        predicted_objective_ids: Iterable[str] = (),
        context_signature: str = "",
    ) -> Dict[str, Any]:
        """Measure all goals and revise truth only from terminal observations."""
        self._transition_index += 1
        expired = self._expire_old_reductions()
        if ablation_of_objective_id and self._active_ablation is None:
            self.begin_ablation(ablation_of_objective_id, intervention_id)

        level_progressed = bool(
            update.record.diff.level_complete
            or update.record.obs_after.levels_completed
            > update.record.obs_before.levels_completed
        )
        won = str(update.record.obs_after.game_state).upper() in WIN_STATES
        terminal_success = bool(level_progressed or won)
        predicted = {str(key) for key in predicted_objective_ids if str(key)}
        if terminal_success and not predicted and objective_id:
            predicted.add(str(objective_id))
        measurable_objectives = self.objectives()
        if terminal_success:
            # The next-level screen is not an action effect.  On the terminal
            # transition, only predeclared reductions remain causally eligible.
            measurable_objectives = [
                objective
                for objective in measurable_objectives
                if objective.objective_id in predicted
            ]
        current_evidence: List[ObjectiveTransitionEvidence] = []
        for objective in measurable_objectives:
            before_distance = objective.distance(update.record.obs_before)
            after_distance = objective.distance(update.record.obs_after)
            if (
                before_distance is None
                or after_distance is None
                or after_distance >= before_distance
            ):
                continue
            evidence = ObjectiveTransitionEvidence(
                objective_id=objective.objective_id,
                rule_key=str(rule_key) if objective.objective_id == objective_id else "",
                intervention_id=str(intervention_id),
                transition_index=self._transition_index,
                context_signature=str(context_signature),
                distance_before=float(before_distance),
                distance_after=float(after_distance),
                selected_objective=objective.objective_id == objective_id,
            )
            objective.distance_reductions += 1
            objective.total_distance_reduction += evidence.reduction
            current_evidence.append(evidence)
            self._recent_reductions.append(evidence)

        credited: List[str] = []
        ambiguous: List[str] = []
        if terminal_success:
            self._terminal_events += 1
            credited, ambiguous = self._resolve_terminal_event()
            self._resolve_active_ablation(
                credited_objective_ids=set(credited),
                ambiguous_objective_ids=set(ambiguous),
            )
            if credited:
                self._credited_terminal_events += 1
            self._recent_reductions.clear()
        elif update.record.diff.game_over:
            self._record_unsafe_intervention(objective_id, intervention_id)
            if self._active_ablation is not None:
                self._active_ablation.terminal_observed = True
                objective = self.objective(self._active_ablation.objective_id)
                if objective is not None:
                    objective.ablation_censored += 1
                self._active_ablation = None
            self._recent_reductions.clear()

        selected = next(
            (
                evidence
                for evidence in current_evidence
                if evidence.objective_id == str(objective_id)
            ),
            None,
        )
        return {
            "objective_id": str(objective_id),
            "distance_before": None if selected is None else selected.distance_before,
            "distance_after": None if selected is None else selected.distance_after,
            "distance_reduction": 0.0 if selected is None else selected.reduction,
            "objective_completed": bool(selected is not None and selected.completed),
            "all_reduced_objectives": sorted(
                evidence.objective_id for evidence in current_evidence
            ),
            "terminal_success": terminal_success,
            "terminal_credited_objectives": credited,
            "terminal_ambiguous_objectives": ambiguous,
            "expired_nonterminal_completions": expired,
            "unsafe_intervention_recorded": bool(
                update.record.diff.game_over and objective_id and intervention_id
            ),
        }

    def _expire_old_reductions(self) -> int:
        retained: List[ObjectiveTransitionEvidence] = []
        expired_completions = 0
        rejected: set[Tuple[str, str]] = set()
        for evidence in self._recent_reductions:
            age = self._transition_index - evidence.transition_index
            if age <= self.terminal_credit_window:
                retained.append(evidence)
                continue
            key = (evidence.objective_id, evidence.context_signature)
            if evidence.completed and not evidence.credited and key not in rejected:
                rejected.add(key)
                objective = self._objectives[evidence.objective_id]
                objective.nonterminal_completions += 1
                objective.terminal_contradictions += 1
                expired_completions += 1
        self._recent_reductions = retained
        return expired_completions

    def _resolve_terminal_event(self) -> Tuple[List[str], List[str]]:
        by_objective: Dict[str, List[ObjectiveTransitionEvidence]] = {}
        for evidence in self._recent_reductions:
            by_objective.setdefault(evidence.objective_id, []).append(evidence)
        objective_ids = sorted(by_objective)
        if not objective_ids:
            return [], []
        if len(objective_ids) > 1:
            contexts = sorted({
                evidence.context_signature
                for evidence in self._recent_reductions
                if evidence.context_signature
            })
            self._ambiguous_events.append(AmbiguousTerminalEvent(
                transition_index=self._transition_index,
                objective_ids=tuple(objective_ids),
                context_signatures=tuple(contexts),
            ))
            for objective_id in objective_ids:
                self._objectives[objective_id].ambiguous_terminal_events += 1
            return [], objective_ids

        objective_id = objective_ids[0]
        evidence_rows = by_objective[objective_id]
        context = "||".join(sorted({
            evidence.context_signature or f"transition:{evidence.transition_index}"
            for evidence in evidence_rows
        }))
        objective = self._objectives[objective_id]
        if context not in objective.terminal_contexts:
            objective.terminal_contexts.add(context)
        objective.terminal_interventions.update(
            evidence.intervention_id
            for evidence in evidence_rows
            if evidence.intervention_id
        )
        for evidence in evidence_rows:
            evidence.credited = True
        return [objective_id], []

    def _resolve_active_ablation(
        self,
        *,
        credited_objective_ids: set[str],
        ambiguous_objective_ids: set[str],
    ) -> None:
        ablation = self._active_ablation
        if ablation is None:
            return
        ablation.terminal_observed = True
        objective = self.objective(ablation.objective_id)
        if objective is not None:
            if objective.objective_id in credited_objective_ids:
                if (
                    ablation.intervention_id
                    not in ablation.baseline_terminal_interventions
                ):
                    objective.ablation_contrasts += 1
            elif objective.objective_id not in ambiguous_objective_ids:
                # A terminal outcome without reducing the ablated goal proves
                # that the goal was not necessary, not that its mechanic failed.
                objective.ablation_contradictions += 1
        self._active_ablation = None

    def _record_unsafe_intervention(
        self,
        objective_id: str,
        intervention_id: str,
    ) -> None:
        objective = self.objective(objective_id)
        if objective is None or not intervention_id:
            return
        objective.unsafe_plan_failures += 1
        objective.dangerous_interventions[intervention_id] = (
            objective.dangerous_interventions.get(intervention_id, 0) + 1
        )

    def _reject_recent_completions(self) -> None:
        rejected: set[Tuple[str, str]] = set()
        for evidence in self._recent_reductions:
            if not evidence.completed or evidence.credited:
                continue
            key = (evidence.objective_id, evidence.context_signature)
            if key in rejected:
                continue
            rejected.add(key)
            objective = self._objectives[evidence.objective_id]
            objective.nonterminal_completions += 1
            objective.terminal_contradictions += 1

    def summary(self) -> Dict[str, Any]:
        objectives = self.objectives()
        statuses = {
            status.value: sum(item.status == status for item in objectives)
            for status in TerminalObjectiveStatus
        }
        return {
            "objectives": len(objectives),
            "statuses": statuses,
            "probe_actions": self._probe_actions_total,
            "grounded_actions": sum(item.grounded_actions for item in objectives),
            "distance_reductions": sum(item.distance_reductions for item in objectives),
            "nonterminal_completions": sum(
                item.nonterminal_completions for item in objectives
            ),
            "terminal_events": self._terminal_events,
            "credited_terminal_events": self._credited_terminal_events,
            "ambiguous_terminal_events": len(self._ambiguous_events),
            "terminal_supported_objectives": sum(
                item.status == TerminalObjectiveStatus.TERMINAL_SUPPORTED
                for item in objectives
            ),
            "objectives_needing_contrast": sum(
                item.status == TerminalObjectiveStatus.NEEDS_CONTRAST
                for item in objectives
            ),
            "refuted_objectives": sum(
                item.status == TerminalObjectiveStatus.REFUTED
                for item in objectives
            ),
            "unsafe_plan_failures": sum(
                item.unsafe_plan_failures for item in objectives
            ),
            "ablation_attempts": sum(item.ablation_attempts for item in objectives),
            "ablation_contrasts": sum(item.ablation_contrasts for item in objectives),
            "ablation_contradictions": sum(
                item.ablation_contradictions for item in objectives
            ),
            "recent_reductions_awaiting_terminal_credit": len(
                self._recent_reductions
            ),
            "ambiguous_events": [
                {
                    "transition_index": event.transition_index,
                    "objective_ids": list(event.objective_ids),
                    "context_signatures": list(event.context_signatures),
                }
                for event in self._ambiguous_events[-10:]
            ],
            "hypotheses": [item.to_dict() for item in objectives],
        }


@dataclass(frozen=True)
class _RuleGoalCandidate:
    objective_id: str
    family: str
    source_color: int | None
    target_color: int | None
    predicate: str
    supporting_rule_keys: Tuple[str, ...]
    supporting_actions: Tuple[str, ...]
    generation_reason: str
    prior_priority: float


def _blocked_assessment(
    objective: TerminalObjectiveHypothesis,
    distance: float | None,
    reason: str,
) -> TerminalObjectiveAssessment:
    return TerminalObjectiveAssessment(
        objective.objective_id,
        objective.status,
        distance,
        False,
        False,
        float("-inf"),
        reason,
    )


def _objective_descriptor(
    rule: PromotedRelationalRule,
) -> Tuple[str, str] | None:
    if (
        rule.family == "color_transform"
        and rule.target_color is not None
        and rule.expected_outcome == f"{rule.source_color}->{rule.target_color}"
    ):
        return (
            f"terminal::convert::{rule.source_color}_to_{rule.target_color}",
            "convert",
        )
    if rule.family == "relation" and rule.target_color is not None:
        pair = _canonical_pair(rule.source_color, rule.target_color)
        if rule.expected_outcome == "appears":
            return (
                f"terminal::appear::{rule.predicate}::colors{pair[0]}_{pair[1]}",
                "appear",
            )
        if rule.expected_outcome == "broken":
            return (
                f"terminal::break::{rule.predicate}::colors{pair[0]}_{pair[1]}",
                "break",
            )
    return None


def _canonical_pair(first: int, second: int) -> Tuple[int, int]:
    return tuple(sorted((int(first), int(second))))  # type: ignore[return-value]


__all__ = [
    "ActiveAblation",
    "AmbiguousTerminalEvent",
    "OnlineTerminalObjectiveStore",
    "ObjectiveTransitionEvidence",
    "TerminalObjectiveAssessment",
    "TerminalObjectiveHypothesis",
    "TerminalObjectiveStatus",
]
