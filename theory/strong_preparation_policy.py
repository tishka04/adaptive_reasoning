"""Strong preparation for correspondence validation options.

A10 separates "the option is globally true" from "this concrete context is
ready enough to invoke it". The policy below is deliberately symbolic: it
selects a preparation primitive because a named discriminating predicate is
missing, then checks whether that predicate appears in the next observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence

from .epistemic_metrics import HypothesisStatus
from .live_transition_loop import LiveTransitionBeliefLoop, LiveTransitionUpdate
from .precondition_hypothesis import extract_precondition_predicates
from .real_env_option_adapter import snapshot_frame
from .theory_option import TheoryOption


@dataclass
class StrongPreparationHypothesis:
    """A falsifiable claim that one action can establish a strong predicate."""

    action: str
    role: str
    target_predicate: str
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts_without_support: int = 10
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    def __post_init__(self) -> None:
        self.action = str(self.action or "").upper()
        self.role = str(self.role or "").strip().lower()
        self.target_predicate = str(self.target_predicate or "").strip().lower()
        self.max_attempts_without_support = max(
            1,
            int(self.max_attempts_without_support),
        )
        self._recompute_status()

    @property
    def key(self) -> str:
        return (
            f"strong_preparation::{self.action}::{self.role}"
            f"::{self.target_predicate}"
        )

    @property
    def support(self) -> int:
        return len(self.evidence_for)

    @property
    def contradictions(self) -> int:
        return len(self.evidence_against)

    @property
    def confidence(self) -> float:
        total = self.support + self.contradictions
        if total == 0:
            return 0.0
        return self.support / total

    def observe(self, *, succeeded: bool, label: str) -> None:
        self.attempts += 1
        if succeeded:
            self.evidence_for.append(label)
        elif self.attempts >= self.max_attempts_without_support:
            self.evidence_against.append(label)
        self._recompute_status()

    def _recompute_status(self) -> None:
        if self.support > 0 and self.support >= self.contradictions:
            self.status = HypothesisStatus.CONFIRMED
        elif self.contradictions > self.support:
            self.status = HypothesisStatus.REFUTED
        else:
            self.status = HypothesisStatus.UNRESOLVED


@dataclass(frozen=True)
class StrongPreparationPlan:
    """One justified action toward a discriminating readiness predicate."""

    action: str
    role: str
    target_predicate: str
    hypothesis_key: str
    reason: str


@dataclass(frozen=True)
class StrongPreparationStepResult:
    """Observed result of one A10 preparation action."""

    plan: StrongPreparationPlan
    predicates_before: List[str]
    predicates_after: List[str]
    strong_ready_before: bool
    strong_ready_after: bool
    missing_predicates_before: List[str]
    update: LiveTransitionUpdate
    next_frame: Any
    hypothesis_status: HypothesisStatus

    @property
    def action(self) -> str:
        return self.plan.action

    @property
    def informative(self) -> bool:
        return (
            self.strong_ready_after
            or self.plan.target_predicate in self.predicates_after
            or self.hypothesis_status == HypothesisStatus.REFUTED
        )


class StrongPrepareCorrespondencePolicy:
    """Reach strong correspondence readiness through named predicates."""

    def __init__(
        self,
        hypotheses: Sequence[StrongPreparationHypothesis] | None = None,
    ) -> None:
        self.hypotheses = list(
            hypotheses or default_ar25_strong_preparation_hypotheses()
        )

    def choose(
        self,
        *,
        option: TheoryOption,
        predicates_present: Iterable[str],
        available_actions: Sequence[str],
    ) -> StrongPreparationPlan | None:
        present = {str(predicate).strip().lower() for predicate in predicates_present}
        available = {str(action).upper() for action in available_actions}
        missing = discriminating_readiness_gaps(present)
        if not missing:
            return None
        if "active_color_pair_10_11" not in present:
            return None
        if "selected_source_matches_target_shape" in missing:
            return self._choose_for_predicate(
                "selected_source_matches_target_shape",
                role="move",
                available=available,
            )
        if "source_target_relation_satisfied" in missing:
            return self._choose_for_predicate(
                "source_target_relation_satisfied",
                role="move",
                available=available,
            )
        return None

    def prepare_once(
        self,
        *,
        env: Any,
        loop: LiveTransitionBeliefLoop,
        option: TheoryOption,
        current_frame: Any,
        previous_action: str = "",
        recent_actions: Iterable[str] = (),
        recent_correspondence_successes: Iterable[bool] = (),
        step: int = 0,
    ) -> StrongPreparationStepResult | None:
        """Execute one justified strong-preparation action."""
        before = snapshot_frame(current_frame)
        predicates_before = extract_precondition_predicates(
            before.grid,
            target_rule=option.target_rule,
            pair_colors=option.pair_colors,
            previous_action=previous_action,
            recent_actions=recent_actions,
            recent_correspondence_successes=recent_correspondence_successes,
        )
        plan = self.choose(
            option=option,
            predicates_present=predicates_before,
            available_actions=before.available_actions,
        )
        if plan is None:
            return None

        next_frame = _step_env(env, _to_env_action(plan.action))
        after = snapshot_frame(
            next_frame,
            fallback_available_actions=before.available_actions,
        )
        update = loop.observe_grids(
            action=plan.action,
            grid_before=before.grid,
            grid_after=after.grid,
            available_actions=before.available_actions,
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            levels_completed_before=before.levels_completed,
            levels_completed_after=after.levels_completed,
            timestamp=step,
            was_experiment=True,
        )
        after_recent = list(recent_actions) + [plan.action]
        predicates_after = extract_precondition_predicates(
            after.grid,
            target_rule=option.target_rule,
            pair_colors=option.pair_colors,
            previous_action=plan.action,
            recent_actions=after_recent[-4:],
            recent_correspondence_successes=recent_correspondence_successes,
        )
        strong_before = _strong_ready(predicates_before)
        strong_after = _strong_ready(predicates_after)
        succeeded = (
            strong_after
            or plan.target_predicate in predicates_after
        )
        hypothesis = self._hypothesis(plan.hypothesis_key)
        hypothesis.observe(
            succeeded=succeeded,
            label=f"observed:{plan.action}:step{int(step)}",
        )
        return StrongPreparationStepResult(
            plan=plan,
            predicates_before=sorted(predicates_before),
            predicates_after=sorted(predicates_after),
            strong_ready_before=strong_before,
            strong_ready_after=strong_after,
            missing_predicates_before=sorted(
                discriminating_readiness_gaps(predicates_before)
            ),
            update=update,
            next_frame=next_frame,
            hypothesis_status=hypothesis.status,
        )

    def _choose_for_predicate(
        self,
        target_predicate: str,
        *,
        role: str,
        available: set[str],
    ) -> StrongPreparationPlan | None:
        candidates = [
            hypothesis for hypothesis in self.hypotheses
            if hypothesis.role == role
            and hypothesis.target_predicate == target_predicate
            and hypothesis.action in available
            and hypothesis.status != HypothesisStatus.REFUTED
        ]
        if not candidates:
            return None
        return _plan_from_hypothesis(candidates[0])

    def _hypothesis(self, key: str) -> StrongPreparationHypothesis:
        for hypothesis in self.hypotheses:
            if hypothesis.key == key:
                return hypothesis
        raise KeyError(key)


def discriminating_readiness_gaps(
    predicates_present: Iterable[str],
) -> List[str]:
    present = {str(predicate).strip().lower() for predicate in predicates_present}
    if "source_target_relation_satisfied" in present:
        return []
    if "selected_source_matches_target_shape" not in present:
        return ["selected_source_matches_target_shape"]
    return ["source_target_relation_satisfied"]


def default_ar25_strong_preparation_hypotheses() -> List[StrongPreparationHypothesis]:
    """Minimal ar25 hypotheses for discriminating readiness."""
    return [
        StrongPreparationHypothesis(
            action="ACTION3",
            role="move",
            target_predicate="selected_source_matches_target_shape",
            max_attempts_without_support=6,
        ),
        StrongPreparationHypothesis(
            action="ACTION2",
            role="move",
            target_predicate="source_target_relation_satisfied",
            max_attempts_without_support=10,
        ),
    ]


def _plan_from_hypothesis(
    hypothesis: StrongPreparationHypothesis,
) -> StrongPreparationPlan:
    return StrongPreparationPlan(
        action=hypothesis.action,
        role=hypothesis.role,
        target_predicate=hypothesis.target_predicate,
        hypothesis_key=hypothesis.key,
        reason=(
            f"{hypothesis.key} tries to establish "
            f"{hypothesis.target_predicate}"
        ),
    )


def _strong_ready(predicates: Iterable[str]) -> bool:
    present = {str(predicate).strip().lower() for predicate in predicates}
    return "source_target_relation_satisfied" in present


def _step_env(env: Any, action: Any) -> Any:
    if hasattr(env, "take_action"):
        return env.take_action(action)
    if hasattr(env, "step"):
        frame = env.step(action)
        if frame is None:
            raise ValueError("env.step returned None")
        return frame
    raise TypeError("env must expose step(action) or take_action(action)")


def _to_env_action(action_name: str) -> Any:
    try:
        from arcengine import GameAction  # type: ignore

        if hasattr(GameAction, "from_name"):
            return GameAction.from_name(action_name)
        return getattr(GameAction, action_name)
    except Exception:
        return action_name
