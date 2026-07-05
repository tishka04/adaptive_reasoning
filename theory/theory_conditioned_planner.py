"""A4 minimal planner conditioned on confirmed game theory.

This is intentionally a proof planner, not a general agent. It reads confirmed
rules from ``GameTheory`` and emits the next primitive environment action that
should test/use the dominant relation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .correspondence_hypothesis import (
    CorrespondenceObservation,
    normalize_pair_colors,
)
from .epistemic_metrics import EpistemicScore, HypothesisStatus, MechanicsOracle
from .live_transition_loop import LiveTransitionBeliefLoop
from .mechanic_hypothesis import GameTheory
from .precondition_hypothesis import (
    PreconditionHypothesis,
    PreconditionObservation,
    extract_precondition_predicates,
)


@dataclass(frozen=True)
class TheoryPlannedAction:
    """One primitive action selected because of a confirmed theory rule."""

    action: str
    purpose: str
    reason: str
    rule_key: str = ""
    precondition_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _normalize_action_name(self.action))


@dataclass
class TheoryPlan:
    """A short proof plan emitted by the theory-conditioned planner."""

    objective: str
    planned_actions: List[TheoryPlannedAction]
    source_rule_key: str = ""
    roles: Dict[str, List[str]] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    expected_signal: str = "level_complete_or_correspondence_improvement"

    @property
    def first_action(self) -> str | None:
        if not self.planned_actions:
            return None
        return self.planned_actions[0].action


@dataclass
class PlannerTraceEvent:
    """One planner decision observed against a live-like transition."""

    step: int
    planned_action: str
    actual_action: str
    executed: bool
    reason: str
    rule_key: str = ""
    level_complete: bool = False
    match_delta: float = 0.0
    global_delta: float = 0.0
    correspondence_improved: bool = False
    precondition_key: str = ""
    predicates_present: List[str] = field(default_factory=list)

    @property
    def correspondence_delta(self) -> float:
        return self.match_delta + self.global_delta

    @property
    def successful(self) -> bool:
        return self.executed and (self.level_complete or self.correspondence_improved)


@dataclass
class TheoryConditionedPlanningRun:
    """Result of replaying a trace while asking the proof planner to act."""

    loop: LiveTransitionBeliefLoop
    events: List[PlannerTraceEvent] = field(default_factory=list)

    @property
    def planned_events(self) -> List[PlannerTraceEvent]:
        return [event for event in self.events if event.planned_action]

    @property
    def correspondence_driven_events(self) -> List[PlannerTraceEvent]:
        return [
            event for event in self.events
            if event.rule_key.startswith("correspondence::")
        ]

    @property
    def executed_events(self) -> List[PlannerTraceEvent]:
        return [event for event in self.events if event.executed]

    @property
    def successful_events(self) -> List[PlannerTraceEvent]:
        return [event for event in self.events if event.successful]

    def score(self, oracle: MechanicsOracle) -> EpistemicScore:
        return self.loop.score(oracle)

    def summary(self) -> Dict[str, Any]:
        return {
            "transitions": self.loop.transition_count,
            "planner_events": len(self.planned_events),
            "correspondence_driven_events": len(self.correspondence_driven_events),
            "executed_planner_events": len(self.executed_events),
            "successful_planner_events": len(self.successful_events),
            "correspondence_rules": [
                rule.key for rule in self.loop.theory.correspondence_rules()
            ],
            "preconditions_confirmed": [
                hyp.key for hyp in self.loop.theory.precondition_hypotheses()
                if hyp.status == HypothesisStatus.CONFIRMED
            ],
        }


class TheoryConditionedPlanner:
    """Select the next action from confirmed theory, not from a score."""

    def __init__(self, *, require_preconditions: bool = False) -> None:
        self.require_preconditions = bool(require_preconditions)

    def plan(
        self,
        theory: GameTheory,
        available_actions: Sequence[Any] | None = None,
        precondition_features: Dict[str, Iterable[str]] | None = None,
    ) -> TheoryPlan | None:
        available = _normalize_actions(available_actions or theory.actions())
        roles = self.identify_roles(theory, available)
        features_by_rule = {
            key: set(value)
            for key, value in (precondition_features or {}).items()
        }
        for rule in _ranked_correspondence_rules(theory):
            if available and rule.action not in available:
                continue
            if rule.relation not in {"validates", "improves", "establishes"}:
                continue
            preconditions = _applicable_preconditions(
                theory,
                rule.key,
                features_by_rule.get(rule.key, set()),
            )
            if self.require_preconditions and not preconditions:
                continue
            roles.setdefault("validates_correspondence", [])
            if rule.action not in roles["validates_correspondence"]:
                roles["validates_correspondence"].append(rule.action)
            precondition_text = (
                f" gated by {preconditions[0].key}" if preconditions else ""
            )
            return TheoryPlan(
                objective=f"use confirmed correspondence rule {rule.key}",
                planned_actions=[
                    TheoryPlannedAction(
                        action=rule.action,
                        purpose="validate_correspondence",
                        reason=(
                            f"confirmed correspondence rule {rule.key}"
                            f"{precondition_text}"
                        ),
                        rule_key=rule.key,
                        precondition_key=preconditions[0].key if preconditions else "",
                    )
                ],
                source_rule_key=rule.key,
                roles=roles,
                preconditions=[hyp.key for hyp in preconditions],
            )
        return None

    def identify_roles(
        self,
        theory: GameTheory,
        available_actions: Sequence[Any] | None = None,
    ) -> Dict[str, List[str]]:
        available = set(_normalize_actions(available_actions or theory.actions()))
        roles: Dict[str, List[str]] = {
            "control_switch": [],
            "move": [],
            "validates_correspondence": [],
        }
        for hypothesis in theory.action_role_hypotheses():
            if hypothesis.status != HypothesisStatus.CONFIRMED:
                continue
            action = _normalize_action_name(hypothesis.action)
            if available and action not in available:
                continue
            if hypothesis.role in roles:
                roles[hypothesis.role].append(action)

        for action in theory.actions():
            action_name = _normalize_action_name(action)
            if available and action_name not in available:
                continue
            dominant = theory.dominant(action_name)
            if dominant is not None and dominant.kind == "move":
                roles["move"].append(action_name)

        for rule in theory.correspondence_rules():
            if available and rule.action not in available:
                continue
            roles["validates_correspondence"].append(rule.action)

        return {key: _dedupe(values) for key, values in roles.items()}


def run_planner_trace_file(
    trace_path: Path,
    *,
    game_id: str = "",
    task_program_path: Path | None = None,
    max_steps: int | None = None,
    background_value: int | None = None,
    infer_players: bool = True,
    verify_every: int = 1,
    correspondence_pair_colors: tuple[int, int] = (10, 11),
    planner: TheoryConditionedPlanner | None = None,
    learn_preconditions: bool = False,
) -> TheoryConditionedPlanningRun:
    """Replay a trace while asking the planner for the next env action."""
    pair_colors = normalize_pair_colors(correspondence_pair_colors)
    loop = LiveTransitionBeliefLoop(
        game_id=game_id,
        background_value=background_value,
        infer_players=infer_players,
        verify_every=verify_every,
        correspondence_pair_colors=pair_colors,
    )
    if task_program_path is not None:
        loop.seed_task_program(task_program_path)

    planner = planner or TheoryConditionedPlanner()
    learn_preconditions = bool(learn_preconditions or planner.require_preconditions)
    events: List[PlannerTraceEvent] = []
    steps_seen = 0
    previous_levels = 0
    previous_action = ""
    recent_actions: List[str] = []
    recent_correspondence_successes: List[bool] = []
    with open(Path(trace_path), "r", encoding="utf-8") as handle:
        for line in handle:
            if max_steps is not None and steps_seen >= max_steps:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            action = _normalize_action_name(item.get("action"))
            if action == "RESET":
                continue
            before = item.get("frame_before")
            after = item.get("frame_after")
            if before is None or after is None:
                continue

            available = _normalize_actions(item.get("available_actions") or [])
            loop.theory.seed_actions(available)
            precondition_features = _precondition_features_for_current_rules(
                loop.theory,
                before,
                previous_action=previous_action,
                recent_actions=recent_actions,
                recent_correspondence_successes=recent_correspondence_successes,
                enabled=learn_preconditions,
            )
            plan = planner.plan(
                loop.theory,
                available,
                precondition_features=precondition_features,
            )
            planned_action = plan.first_action if plan is not None else ""
            source_rule = plan.source_rule_key if plan is not None else ""
            source_precondition = (
                plan.planned_actions[0].precondition_key
                if plan is not None and plan.planned_actions
                else ""
            )
            reason = (
                plan.planned_actions[0].reason
                if plan is not None and plan.planned_actions
                else ""
            )

            update = loop.observe_grids(
                action=action,
                action_args=item.get("action_args"),
                grid_before=before,
                grid_after=after,
                available_actions=available,
                game_state_after=item.get("game_state_after", "NOT_FINISHED"),
                levels_completed_before=previous_levels,
                levels_completed_after=int(item.get("levels_completed_after", 0) or 0),
                timestamp=int(item.get("step", steps_seen) or 0),
                was_experiment=True,
            )

            if planned_action:
                executed = planned_action == action
                observation = CorrespondenceObservation.from_transition(
                    update.record,
                    pair_colors=pair_colors,
                )
                events.append(PlannerTraceEvent(
                    step=int(item.get("step", steps_seen) or 0),
                    planned_action=planned_action,
                    actual_action=action,
                    executed=executed,
                    reason=reason,
                    rule_key=source_rule,
                    level_complete=bool(update.effect.level_complete) if executed else False,
                    match_delta=observation.match_delta if executed else 0.0,
                    global_delta=observation.global_delta if executed else 0.0,
                    correspondence_improved=observation.improves if executed else False,
                    precondition_key=source_precondition,
                    predicates_present=sorted(precondition_features.get(source_rule, set())),
                ))

            if learn_preconditions:
                correspondence_successes = _observe_preconditions_from_transition(
                    loop.theory,
                    action=action,
                    precondition_features=precondition_features,
                    succeeded_by_rule=_correspondence_success_by_rule(
                        loop.theory,
                        update,
                        pair_colors=pair_colors,
                    ),
                    was_experiment=True,
                )
                recent_correspondence_successes.append(any(correspondence_successes))
                recent_correspondence_successes = recent_correspondence_successes[-4:]
            else:
                observation = CorrespondenceObservation.from_transition(
                    update.record,
                    pair_colors=pair_colors,
                )
                recent_correspondence_successes.append(
                    bool(observation.improves or update.effect.level_complete)
                )
                recent_correspondence_successes = recent_correspondence_successes[-4:]

            previous_levels = int(item.get("levels_completed_after", previous_levels) or 0)
            previous_action = action
            recent_actions.append(action)
            recent_actions = recent_actions[-4:]
            steps_seen += 1

    return TheoryConditionedPlanningRun(loop=loop, events=events)


def _ranked_correspondence_rules(theory: GameTheory):
    return sorted(
        theory.correspondence_rules(),
        key=lambda rule: (rule.confidence, rule.support, rule.key),
        reverse=True,
    )


def _applicable_preconditions(
    theory: GameTheory,
    target_rule: str,
    predicates_present: Iterable[str],
) -> List[PreconditionHypothesis]:
    return [
        hyp for hyp in theory.preconditions_for_rule(target_rule)
        if hyp.is_applicable(predicates_present)
    ]


def _precondition_features_for_current_rules(
    theory: GameTheory,
    grid: Any,
    *,
    previous_action: str,
    recent_actions: Iterable[str],
    recent_correspondence_successes: Iterable[bool],
    enabled: bool,
) -> Dict[str, set[str]]:
    if not enabled:
        return {}
    features: Dict[str, set[str]] = {}
    for rule in theory.correspondence_rules():
        _ensure_default_preconditions(theory, rule.key)
        features[rule.key] = extract_precondition_predicates(
            grid,
            target_rule=rule.key,
            pair_colors=rule.pair_colors,
            previous_action=previous_action,
            recent_actions=recent_actions,
            recent_correspondence_successes=recent_correspondence_successes,
        )
    return features


def _ensure_default_preconditions(theory: GameTheory, target_rule: str) -> None:
    if theory.preconditions_for_rule(target_rule):
        return
    theory.add_precondition(PreconditionHypothesis(
        target_rule=target_rule,
        predicate="ready_to_validate_correspondence",
    ))


def _correspondence_success_by_rule(
    theory: GameTheory,
    update: Any,
    *,
    pair_colors: tuple[int, int],
) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for rule in theory.correspondence_rules():
        observation = CorrespondenceObservation.from_transition(
            update.record,
            pair_colors=rule.pair_colors or pair_colors,
        )
        results[rule.key] = bool(observation.improves or update.effect.level_complete)
    return results


def _observe_preconditions_from_transition(
    theory: GameTheory,
    *,
    action: str,
    precondition_features: Dict[str, set[str]],
    succeeded_by_rule: Dict[str, bool],
    was_experiment: bool,
) -> List[bool]:
    successes: List[bool] = []
    for target_rule, predicates in precondition_features.items():
        succeeded = bool(succeeded_by_rule.get(target_rule, False))
        observation = PreconditionObservation(
            target_rule=target_rule,
            action=action,
            predicates_present=set(predicates),
            succeeded=succeeded,
        )
        theory.observe_precondition(observation, was_experiment=was_experiment)
        if observation.target_action_executed:
            successes.append(succeeded)
    return successes


def _normalize_actions(actions: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for action in actions:
        name = _normalize_action_name(action)
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_action_name(action: Any) -> str:
    if isinstance(action, (int, np.integer)):
        return f"ACTION{int(action)}"
    raw = str(action or "").strip().upper()
    if raw.isdigit():
        return f"ACTION{raw}"
    return raw


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
