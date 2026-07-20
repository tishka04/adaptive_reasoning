"""Online composition and revision of state-conditioned goal sequences.

The composer never memorizes an action script.  It orders measurable goal
deficits, asks the existing intervention designer for exactly one primitive
action, and re-observes the state before deciding whether to continue.  Local
subgoal progress is useful control evidence, but only an observed level change
or WIN may support the terminal value of a complete sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple

from v3.schemas import GameObservation

from .online_causal_subgoal_graph import CausalSubgoalEdge
from .online_terminal_objective import (
    OnlineTerminalObjectiveStore,
    TerminalObjectiveHypothesis,
    TerminalObjectiveStatus,
)


WIN_STATES = {"WIN", "WON", "VICTORY"}


class TemporalPlanStatus(str, Enum):
    """Terminal epistemic status of an ordered subgoal hypothesis."""

    CANDIDATE = "candidate"
    NEEDS_CONTRAST = "needs_contrast"
    TERMINAL_SUPPORTED = "terminal_supported"
    REFUTED = "refuted"


@dataclass(frozen=True)
class TemporalSubgoalStep:
    """A state guard and a measurable completion threshold, not an action."""

    objective_id: str
    target_distance: float
    guard: str = "distance_measurable_and_above_target"
    role: str = "progress"

    def distance(self, observation: GameObservation, store: OnlineTerminalObjectiveStore) -> float | None:
        objective = store.objective(self.objective_id)
        return None if objective is None else objective.distance(observation)

    def is_satisfied(
        self,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
    ) -> bool:
        distance = self.distance(observation, store)
        return distance is not None and distance <= self.target_distance

    def is_enabled(
        self,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
    ) -> bool:
        objective = store.objective(self.objective_id)
        distance = self.distance(observation, store)
        return bool(
            objective is not None
            and objective.status != TerminalObjectiveStatus.REFUTED
            and distance is not None
            and distance > self.target_distance
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "target_distance": float(self.target_distance),
            "guard": self.guard,
            "role": self.role,
        }


@dataclass
class TemporalGoalPlan:
    """A bounded, revisable hypothesis about an ordered subgoal sequence."""

    plan_id: str
    target_objective_id: str
    steps: Tuple[TemporalSubgoalStep, ...]
    generation_reason: str
    prior_priority: float = 0.0
    causal_edge_key: str = ""
    causal_edge_utility: float = 0.0
    causal_confirmation_priority: bool = False
    minimum_independent_terminal_contexts: int = 2
    starts: int = 0
    actions: int = 0
    progress_events: int = 0
    step_completions: int = 0
    local_completions: int = 0
    nonterminal_completions: int = 0
    stalls: int = 0
    abandonments: int = 0
    unsafe_failures: int = 0
    terminal_bypasses: int = 0
    ambiguous_terminal_events: int = 0
    terminal_contradictions: int = 0
    terminal_contexts: set[str] = field(default_factory=set)
    terminal_interventions: set[str] = field(default_factory=set)
    dangerous_interventions: Dict[str, int] = field(default_factory=dict)
    abandonment_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def terminal_support(self) -> int:
        return len(self.terminal_contexts)

    @property
    def status(self) -> TemporalPlanStatus:
        if self.terminal_support >= max(
            1, int(self.minimum_independent_terminal_contexts)
        ):
            return TemporalPlanStatus.TERMINAL_SUPPORTED
        if self.terminal_support > 0:
            return TemporalPlanStatus.NEEDS_CONTRAST
        if self.terminal_contradictions >= 2:
            return TemporalPlanStatus.REFUTED
        return TemporalPlanStatus.CANDIDATE

    @property
    def expected_progress_probability(self) -> float:
        return (self.progress_events + 1.0) / (self.actions + 2.0)

    @property
    def expected_completion_probability(self) -> float:
        return (self.step_completions + 1.0) / (self.actions + 2.0)

    @property
    def expected_cost(self) -> float:
        if self.starts <= 0:
            return 1.0
        return max(1.0, float(self.actions) / float(self.starts))

    @property
    def risk_rate(self) -> float:
        return float(self.unsafe_failures) / max(1.0, float(self.actions))

    @property
    def selection_utility(self) -> float:
        abandonment_rate = float(self.abandonments) / max(1.0, float(self.starts))
        return (
            float(self.prior_priority)
            + float(self.causal_edge_utility)
            + 2.0 * self.expected_progress_probability
            + self.expected_completion_probability
            - 0.15 * self.expected_cost
            - 2.0 * self.risk_rate
            - 0.5 * abandonment_rate
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "target_objective_id": self.target_objective_id,
            "steps": [step.to_dict() for step in self.steps],
            "generation_reason": self.generation_reason,
            "prior_priority": round(float(self.prior_priority), 4),
            "causal_edge_key": self.causal_edge_key,
            "causal_edge_utility": round(float(self.causal_edge_utility), 4),
            "causal_confirmation_priority": self.causal_confirmation_priority,
            "status": self.status.value,
            "expected_progress_probability": round(
                self.expected_progress_probability, 4
            ),
            "expected_completion_probability": round(
                self.expected_completion_probability, 4
            ),
            "expected_cost": round(self.expected_cost, 4),
            "risk_rate": round(self.risk_rate, 4),
            "selection_utility": round(self.selection_utility, 4),
            "starts": self.starts,
            "actions": self.actions,
            "progress_events": self.progress_events,
            "step_completions": self.step_completions,
            "local_completions": self.local_completions,
            "nonterminal_completions": self.nonterminal_completions,
            "stalls": self.stalls,
            "abandonments": self.abandonments,
            "unsafe_failures": self.unsafe_failures,
            "terminal_bypasses": self.terminal_bypasses,
            "ambiguous_terminal_events": self.ambiguous_terminal_events,
            "terminal_support": self.terminal_support,
            "terminal_contradictions": self.terminal_contradictions,
            "terminal_contexts": sorted(self.terminal_contexts),
            "terminal_interventions": sorted(self.terminal_interventions),
            "dangerous_interventions": dict(sorted(self.dangerous_interventions.items())),
            "abandonment_reasons": dict(sorted(self.abandonment_reasons.items())),
        }


@dataclass
class ActiveTemporalPlan:
    plan_id: str
    branch_index: int
    step_index: int = 0
    actions: int = 0
    stalls_on_step: int = 0


@dataclass(frozen=True)
class TemporalStepSelection:
    plan_id: str
    target_objective_id: str
    plan_status: TemporalPlanStatus
    step_index: int
    step_count: int
    objective_id: str
    target_distance: float
    current_distance: float
    causal_edge_key: str
    causal_confirmation_priority: bool
    expected_progress_probability: float
    expected_cost: float
    selection_utility: float
    reason: str


@dataclass
class _RecentPlanCompletion:
    plan_id: str
    transition_index: int
    branch_index: int
    context_signature: str
    intervention_id: str


class OnlineTemporalGoalComposer:
    """Compose, execute, and terminally revise bounded subgoal chains."""

    def __init__(
        self,
        *,
        max_plans: int = 12,
        max_plan_starts_total: int = 12,
        max_starts_per_plan: int = 2,
        max_actions_per_plan: int = 8,
        max_stalls_per_step: int = 2,
        terminal_credit_window: int = 6,
        minimum_terminal_support: int = 2,
        max_confirmation_starts_per_branch: int = 2,
    ) -> None:
        self.max_plans = max(1, int(max_plans))
        self.max_plan_starts_total = max(0, int(max_plan_starts_total))
        self.max_starts_per_plan = max(1, int(max_starts_per_plan))
        self.max_actions_per_plan = max(1, int(max_actions_per_plan))
        self.max_stalls_per_step = max(1, int(max_stalls_per_step))
        self.terminal_credit_window = max(0, int(terminal_credit_window))
        self.minimum_terminal_support = max(1, int(minimum_terminal_support))
        self.max_confirmation_starts_per_branch = max(
            0, int(max_confirmation_starts_per_branch)
        )
        self._plans: Dict[str, TemporalGoalPlan] = {}
        self._active: ActiveTemporalPlan | None = None
        self._recent_completions: List[_RecentPlanCompletion] = []
        self._attempted_contexts: set[Tuple[str, int, int]] = set()
        self._transition_index = 0
        self._branch_index = 0
        self._plan_starts_total = 0
        self._confirmation_starts_total = 0
        self._confirmation_starts_this_branch = 0
        self._mediated_replication_starts_total = 0
        self._mediated_discrimination_starts_total = 0
        self._plans_generated_total = 0
        self._step_decisions = 0
        self._terminal_events = 0
        self._credited_terminal_events = 0
        self._ambiguous_terminal_events = 0

    @property
    def active_plan_id(self) -> str:
        return "" if self._active is None else self._active.plan_id

    def plans(self) -> List[TemporalGoalPlan]:
        return sorted(self._plans.values(), key=lambda item: item.plan_id)

    def plan(self, plan_id: str) -> TemporalGoalPlan | None:
        return self._plans.get(str(plan_id))

    def compose(
        self,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        causal_edges: Sequence[CausalSubgoalEdge] = (),
    ) -> List[TemporalGoalPlan]:
        """Build thresholds and learned-objective dependencies from live state."""
        candidates: List[TemporalGoalPlan] = []
        objectives = [
            objective for objective in store.objectives()
            if objective.status != TerminalObjectiveStatus.REFUTED
        ]
        for objective in objectives:
            distance = objective.distance(observation)
            if distance is None or distance <= 1.0:
                continue
            midpoint = max(1.0, float(int(distance // 2)))
            if midpoint >= distance:
                midpoint = max(0.0, distance - 1.0)
            steps = (
                TemporalSubgoalStep(
                    objective.objective_id,
                    midpoint,
                    role="measurable_milestone",
                ),
                TemporalSubgoalStep(
                    objective.objective_id,
                    0.0,
                    role="objective_completion",
                ),
            )
            candidates.append(self._candidate_plan(
                target=objective,
                steps=steps,
                reason="live_distance_threshold_decomposition",
            ))

        present_colors = {int(obj.value) for obj in observation.objects}
        conversions = [
            objective for objective in objectives
            if objective.family == "convert"
            and objective.target_color is not None
            and (objective.distance(observation) or 0.0) > 0.0
        ]
        for target in objectives:
            missing_colors: set[int] = set()
            if target.family == "appear":
                missing_colors = {
                    int(color)
                    for color in (target.source_color, target.target_color)
                    if color is not None and int(color) not in present_colors
                }
            elif (
                target.family == "reach"
                and target.target_color is not None
                and int(target.target_color) not in present_colors
            ):
                missing_colors = {int(target.target_color)}
            for dependency in conversions:
                if int(dependency.target_color) not in missing_colors:
                    continue
                dependency_distance = dependency.distance(observation)
                if dependency_distance is None or dependency_distance <= 0.0:
                    continue
                # One conversion is enough to make the missing target present;
                # the state is then re-observed before the terminal subgoal.
                dependency_target = max(0.0, float(dependency_distance) - 1.0)
                candidates.append(self._candidate_plan(
                    target=target,
                    steps=(
                        TemporalSubgoalStep(
                            dependency.objective_id,
                            dependency_target,
                            role="create_missing_precondition",
                        ),
                        TemporalSubgoalStep(
                            target.objective_id,
                            0.0,
                            role="dependent_objective_completion",
                        ),
                    ),
                    reason="live_objective_dependency_composition",
                    dependency=dependency,
                ))

        for edge in causal_edges:
            source = store.objective(edge.source_objective_id)
            target = store.objective(edge.target_objective_id)
            if source is None or target is None:
                continue
            source_distance = source.distance(observation)
            if source_distance is None or source_distance <= 0.0:
                continue
            source_threshold = max(0.0, float(source_distance) - 1.0)
            candidates.append(self._candidate_plan(
                target=target,
                steps=(
                    TemporalSubgoalStep(
                        source.objective_id,
                        source_threshold,
                        role="learned_causal_precondition",
                    ),
                    TemporalSubgoalStep(
                        target.objective_id,
                        0.0,
                        role="causally_enabled_objective",
                    ),
                ),
                reason="online_learned_causal_subgoal_dependency",
                dependency=source,
                causal_edge_key=edge.edge_key,
                priority_bonus=edge.utility,
                causal_confirmation_priority=(
                    edge.needs_independent_confirmation
                    and edge.confirmation_priority_bonus > 0.0
                ),
            ))

        ranked = sorted(
            candidates,
            key=lambda item: (
                item.prior_priority,
                -len(item.steps),
                item.plan_id,
            ),
            reverse=True,
        )
        for candidate in ranked:
            self._register(candidate)
        return self.plans()

    def select_step(
        self,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        *,
        preferred_causal_edge_key: str = "",
        preferred_discrimination_edge_key: str = "",
    ) -> TemporalStepSelection | None:
        """Return one enabled subgoal; callers still select only one action."""
        if self._active is not None:
            selection = self._selection_for_active(observation, store)
            if selection is not None:
                self._step_decisions += 1
                return selection
        normal_budget_available = (
            self._plan_starts_total < self.max_plan_starts_total
        )

        candidates: List[Tuple[Tuple[float, ...], TemporalGoalPlan]] = []
        for plan in self.plans():
            if plan.status == TemporalPlanStatus.REFUTED:
                continue
            mediated_replication = bool(
                str(preferred_causal_edge_key)
                and plan.causal_edge_key == str(preferred_causal_edge_key)
            )
            mediated_discrimination = bool(
                str(preferred_discrimination_edge_key)
                and plan.causal_edge_key
                == str(preferred_discrimination_edge_key)
            )
            mediated_reservation = bool(
                mediated_discrimination or mediated_replication
            )
            if (
                plan.starts >= self.max_starts_per_plan
                and not mediated_reservation
            ):
                continue
            reserved_confirmation = bool(
                not normal_budget_available
                and plan.causal_confirmation_priority
                and self._confirmation_starts_this_branch
                < self.max_confirmation_starts_per_branch
            )
            if (
                not normal_budget_available
                and not reserved_confirmation
                and not mediated_reservation
            ):
                continue
            if not plan.steps:
                continue
            context = (plan.plan_id, self._branch_index, observation.grid_hash)
            if context in self._attempted_contexts:
                continue
            first = plan.steps[0]
            distance = first.distance(observation, store)
            if distance is None or distance <= first.target_distance:
                continue
            target = store.objective(plan.target_objective_id)
            if target is None or target.status == TerminalObjectiveStatus.REFUTED:
                continue
            status_bonus = {
                TemporalPlanStatus.TERMINAL_SUPPORTED: 20.0,
                TemporalPlanStatus.NEEDS_CONTRAST: 8.0,
                TemporalPlanStatus.CANDIDATE: 1.0,
            }.get(plan.status, 0.0)
            candidates.append(((
                int(mediated_discrimination),
                int(mediated_replication),
                status_bonus,
                plan.selection_utility,
                -plan.starts,
                -len(plan.steps),
            ), plan))
        if not candidates:
            return None
        plan = max(candidates, key=lambda item: item[0])[1]
        self._active = ActiveTemporalPlan(
            plan_id=plan.plan_id,
            branch_index=self._branch_index,
        )
        plan.starts += 1
        self._plan_starts_total += 1
        if (
            str(preferred_discrimination_edge_key)
            and plan.causal_edge_key
            == str(preferred_discrimination_edge_key)
        ):
            self._mediated_discrimination_starts_total += 1
        elif (
            str(preferred_causal_edge_key)
            and plan.causal_edge_key == str(preferred_causal_edge_key)
        ):
            self._mediated_replication_starts_total += 1
        if not normal_budget_available and plan.causal_confirmation_priority:
            self._confirmation_starts_total += 1
            self._confirmation_starts_this_branch += 1
        self._attempted_contexts.add(
            (plan.plan_id, self._branch_index, observation.grid_hash)
        )
        selection = self._selection_for_active(observation, store)
        if selection is not None:
            self._step_decisions += 1
        return selection

    def reject_active_step(self, reason: str) -> None:
        """Abandon a chain when no safe intervention can satisfy its guard."""
        self._abandon_active(str(reason) or "no_intervention")

    def observe_transition(
        self,
        update: Any,
        *,
        store: OnlineTerminalObjectiveStore,
        plan_id: str = "",
        step_index: int | None = None,
        objective_id: str = "",
        target_distance: float | None = None,
        intervention_id: str = "",
        context_signature: str = "",
    ) -> Dict[str, Any]:
        """Revise a chain from actual distances and terminal outcomes only."""
        self._transition_index += 1
        expired = self._expire_completions()
        level_progressed = bool(
            update.record.diff.level_complete
            or update.record.obs_after.levels_completed
            > update.record.obs_before.levels_completed
        )
        won = str(update.record.obs_after.game_state).upper() in WIN_STATES
        terminal_success = bool(level_progressed or won)
        if terminal_success:
            self._terminal_events += 1

        outcome: Dict[str, Any] = {
            "plan_id": str(plan_id),
            "step_index": step_index,
            "objective_id": str(objective_id),
            "distance_before": None,
            "distance_after": None,
            "distance_reduction": 0.0,
            "step_completed": False,
            "plan_completed": False,
            "plan_abandoned": False,
            "abandonment_reason": "",
            "terminal_success": terminal_success,
            "terminal_credited_plans": [],
            "terminal_ambiguous_plans": [],
            "expired_nonterminal_completions": expired,
        }
        plan = self.plan(plan_id)
        active_matches = bool(
            plan is not None
            and self._active is not None
            and self._active.plan_id == str(plan_id)
            and step_index is not None
            and self._active.step_index == int(step_index)
        )
        if active_matches and plan is not None:
            active = self._active
            assert active is not None
            active.actions += 1
            plan.actions += 1
            objective = store.objective(objective_id)
            before_distance = (
                None if objective is None
                else objective.distance(update.record.obs_before)
            )
            after_distance = (
                None if objective is None
                else objective.distance(update.record.obs_after)
            )
            outcome["distance_before"] = before_distance
            outcome["distance_after"] = after_distance
            threshold = (
                plan.steps[active.step_index].target_distance
                if target_distance is None else float(target_distance)
            )
            reduced = bool(
                before_distance is not None
                and after_distance is not None
                and after_distance < before_distance
            )
            if reduced:
                reduction = float(before_distance) - float(after_distance)
                outcome["distance_reduction"] = reduction
                plan.progress_events += 1
                active.stalls_on_step = 0
            else:
                plan.stalls += 1
                active.stalls_on_step += 1

            if update.record.diff.game_over:
                plan.unsafe_failures += 1
                if intervention_id:
                    plan.dangerous_interventions[intervention_id] = (
                        plan.dangerous_interventions.get(intervention_id, 0) + 1
                    )
                self._abandon_active("game_over")
                outcome["plan_abandoned"] = True
                outcome["abandonment_reason"] = "game_over"
            elif (
                before_distance is not None
                and after_distance is not None
                and after_distance <= threshold
                and before_distance > threshold
            ):
                plan.step_completions += 1
                outcome["step_completed"] = True
                active.step_index += 1
                active.stalls_on_step = 0
                if active.step_index >= len(plan.steps):
                    plan.local_completions += 1
                    outcome["plan_completed"] = True
                    self._recent_completions.append(_RecentPlanCompletion(
                        plan_id=plan.plan_id,
                        transition_index=self._transition_index,
                        branch_index=self._branch_index,
                        context_signature=(
                            str(context_signature)
                            or f"transition:{self._transition_index}"
                        ),
                        intervention_id=str(intervention_id),
                    ))
                    self._active = None
            elif active.actions >= self.max_actions_per_plan:
                self._abandon_active("action_budget")
                outcome["plan_abandoned"] = True
                outcome["abandonment_reason"] = "action_budget"
            elif active.stalls_on_step >= self.max_stalls_per_step:
                self._abandon_active("state_condition_stalled")
                outcome["plan_abandoned"] = True
                outcome["abandonment_reason"] = "state_condition_stalled"

        if terminal_success:
            # A terminal transition before the final guard is a causal bypass,
            # not positive evidence for the incomplete sequence.
            if self._active is not None:
                active_plan = self.plan(self._active.plan_id)
                if active_plan is not None:
                    active_plan.terminal_bypasses += 1
                    active_plan.terminal_contradictions += 1
                self._abandon_active("terminal_before_sequence_completion")
                outcome["plan_abandoned"] = True
                outcome["abandonment_reason"] = (
                    "terminal_before_sequence_completion"
                )
            credited, ambiguous = self._resolve_terminal_event()
            outcome["terminal_credited_plans"] = credited
            outcome["terminal_ambiguous_plans"] = ambiguous
            if credited:
                self._credited_terminal_events += 1
        return outcome

    def start_branch(self) -> None:
        """Close branch-local execution while retaining learned plan evidence."""
        self._reject_recent_completions()
        if self._active is not None:
            self._abandon_active("branch_reset")
        self._branch_index += 1
        self._confirmation_starts_this_branch = 0
        self._recent_completions.clear()

    def summary(self) -> Dict[str, Any]:
        plans = self.plans()
        statuses = {
            status.value: sum(plan.status == status for plan in plans)
            for status in TemporalPlanStatus
        }
        return {
            "plans": len(plans),
            "plans_generated_total": self._plans_generated_total,
            "statuses": statuses,
            "active_plan_id": self.active_plan_id,
            "plan_starts": self._plan_starts_total,
            "reserved_confirmation_starts": self._confirmation_starts_total,
            "mediated_replication_preparation_starts": (
                self._mediated_replication_starts_total
            ),
            "mediated_discrimination_preparation_starts": (
                self._mediated_discrimination_starts_total
            ),
            "step_decisions": self._step_decisions,
            "actions": sum(plan.actions for plan in plans),
            "progress_events": sum(plan.progress_events for plan in plans),
            "step_completions": sum(plan.step_completions for plan in plans),
            "local_completions": sum(plan.local_completions for plan in plans),
            "nonterminal_completions": sum(
                plan.nonterminal_completions for plan in plans
            ),
            "stalls": sum(plan.stalls for plan in plans),
            "abandonments": sum(plan.abandonments for plan in plans),
            "unsafe_failures": sum(plan.unsafe_failures for plan in plans),
            "terminal_bypasses": sum(plan.terminal_bypasses for plan in plans),
            "terminal_events": self._terminal_events,
            "credited_terminal_events": self._credited_terminal_events,
            "ambiguous_terminal_events": self._ambiguous_terminal_events,
            "terminal_supported_plans": sum(
                plan.status == TemporalPlanStatus.TERMINAL_SUPPORTED
                for plan in plans
            ),
            "refuted_plans": sum(
                plan.status == TemporalPlanStatus.REFUTED for plan in plans
            ),
            "causal_dependency_plans": sum(
                bool(plan.causal_edge_key) for plan in plans
            ),
            "causal_dependency_plan_starts": sum(
                plan.starts for plan in plans if plan.causal_edge_key
            ),
            "causal_dependency_plan_actions": sum(
                plan.actions for plan in plans if plan.causal_edge_key
            ),
            "causal_dependency_progress_events": sum(
                plan.progress_events for plan in plans if plan.causal_edge_key
            ),
            "causal_dependency_step_completions": sum(
                plan.step_completions for plan in plans if plan.causal_edge_key
            ),
            "recent_completions_awaiting_terminal_credit": len(
                self._recent_completions
            ),
            "hypotheses": [plan.to_dict() for plan in plans],
        }

    def _candidate_plan(
        self,
        *,
        target: TerminalObjectiveHypothesis,
        steps: Sequence[TemporalSubgoalStep],
        reason: str,
        dependency: TerminalObjectiveHypothesis | None = None,
        causal_edge_key: str = "",
        priority_bonus: float = 0.0,
        causal_confirmation_priority: bool = False,
    ) -> TemporalGoalPlan:
        signature = "__then__".join(
            f"{step.objective_id}@{step.target_distance:g}" for step in steps
        )
        priority = float(target.prior_priority)
        if dependency is not None:
            priority += 0.5 * float(dependency.prior_priority)
        return TemporalGoalPlan(
            plan_id=f"temporal::{signature}",
            target_objective_id=target.objective_id,
            steps=tuple(steps),
            generation_reason=reason,
            prior_priority=priority,
            causal_edge_key=str(causal_edge_key),
            causal_edge_utility=float(priority_bonus),
            causal_confirmation_priority=bool(causal_confirmation_priority),
            minimum_independent_terminal_contexts=self.minimum_terminal_support,
        )

    def _register(self, candidate: TemporalGoalPlan) -> None:
        existing = self._plans.get(candidate.plan_id)
        if existing is not None:
            existing.prior_priority = max(
                existing.prior_priority, candidate.prior_priority
            )
            if candidate.causal_edge_key:
                existing.causal_edge_key = candidate.causal_edge_key
                existing.causal_edge_utility = candidate.causal_edge_utility
                existing.causal_confirmation_priority = (
                    candidate.causal_confirmation_priority
                )
            return
        if len(self._plans) >= self.max_plans:
            evictable = [
                plan for plan in self._plans.values()
                if plan.starts == 0 and plan.terminal_support == 0
            ]
            if not evictable:
                return
            weakest = min(
                evictable,
                key=lambda item: (item.prior_priority, item.plan_id),
            )
            if candidate.prior_priority <= weakest.prior_priority:
                return
            del self._plans[weakest.plan_id]
        self._plans[candidate.plan_id] = candidate
        self._plans_generated_total += 1

    def _selection_for_active(
        self,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
    ) -> TemporalStepSelection | None:
        active = self._active
        if active is None:
            return None
        plan = self.plan(active.plan_id)
        if plan is None or plan.status == TemporalPlanStatus.REFUTED:
            self._abandon_active("plan_unavailable_or_refuted")
            return None
        while active.step_index < len(plan.steps):
            step = plan.steps[active.step_index]
            distance = step.distance(observation, store)
            if distance is None:
                self._abandon_active("state_condition_unmeasurable")
                return None
            if distance > step.target_distance:
                return TemporalStepSelection(
                    plan_id=plan.plan_id,
                    target_objective_id=plan.target_objective_id,
                    plan_status=plan.status,
                    step_index=active.step_index,
                    step_count=len(plan.steps),
                    objective_id=step.objective_id,
                    target_distance=step.target_distance,
                    current_distance=float(distance),
                    causal_edge_key=plan.causal_edge_key,
                    causal_confirmation_priority=(
                        plan.causal_confirmation_priority
                    ),
                    expected_progress_probability=(
                        plan.expected_progress_probability
                    ),
                    expected_cost=plan.expected_cost,
                    selection_utility=plan.selection_utility,
                    reason=(
                        f"{plan.generation_reason}; guard {step.guard}: "
                        f"distance {distance:g} > {step.target_distance:g}"
                    ),
                )
            plan.step_completions += 1
            active.step_index += 1
            active.stalls_on_step = 0
        plan.local_completions += 1
        self._active = None
        return None

    def _abandon_active(self, reason: str) -> None:
        active = self._active
        if active is None:
            return
        plan = self.plan(active.plan_id)
        if plan is not None:
            plan.abandonments += 1
            plan.abandonment_reasons[reason] = (
                plan.abandonment_reasons.get(reason, 0) + 1
            )
        self._active = None

    def _expire_completions(self) -> int:
        retained: List[_RecentPlanCompletion] = []
        expired = 0
        for completion in self._recent_completions:
            age = self._transition_index - completion.transition_index
            if age <= self.terminal_credit_window:
                retained.append(completion)
                continue
            plan = self.plan(completion.plan_id)
            if plan is not None:
                plan.nonterminal_completions += 1
                plan.terminal_contradictions += 1
                expired += 1
        self._recent_completions = retained
        return expired

    def _reject_recent_completions(self) -> None:
        for completion in self._recent_completions:
            plan = self.plan(completion.plan_id)
            if plan is not None:
                plan.nonterminal_completions += 1
                plan.terminal_contradictions += 1

    def _resolve_terminal_event(self) -> Tuple[List[str], List[str]]:
        eligible = [
            completion for completion in self._recent_completions
            if completion.branch_index == self._branch_index
            and self._transition_index - completion.transition_index
            <= self.terminal_credit_window
        ]
        self._recent_completions.clear()
        plan_ids = sorted({completion.plan_id for completion in eligible})
        if not plan_ids:
            return [], []
        if len(plan_ids) > 1:
            self._ambiguous_terminal_events += 1
            for plan_id in plan_ids:
                plan = self.plan(plan_id)
                if plan is not None:
                    plan.ambiguous_terminal_events += 1
            return [], plan_ids
        plan_id = plan_ids[0]
        plan = self.plan(plan_id)
        if plan is None:
            return [], []
        contexts = sorted({
            completion.context_signature for completion in eligible
            if completion.plan_id == plan_id
        })
        interventions = {
            completion.intervention_id for completion in eligible
            if completion.plan_id == plan_id and completion.intervention_id
        }
        context = "||".join(contexts) or f"transition:{self._transition_index}"
        plan.terminal_contexts.add(context)
        plan.terminal_interventions.update(interventions)
        return [plan_id], []


__all__ = [
    "ActiveTemporalPlan",
    "OnlineTemporalGoalComposer",
    "TemporalGoalPlan",
    "TemporalPlanStatus",
    "TemporalStepSelection",
    "TemporalSubgoalStep",
]
