"""One-step preparation policy for correspondence options.

The policy is intentionally symbolic: it selects a preparation action because a
named predicate is missing and an explicit action-role hypothesis says which
primitive can try to establish it. It does not rank actions by game score.
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
class PreparationActionHypothesis:
    """A falsifiable preparation claim for one primitive action."""

    action: str
    role: str
    target_predicate: str
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    attempts: int = 0
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    def __post_init__(self) -> None:
        self.action = str(self.action or "").upper()
        self.role = str(self.role or "").strip().lower()
        self.target_predicate = str(self.target_predicate or "").strip().lower()
        self._recompute_status()

    @property
    def key(self) -> str:
        return (
            f"preparation::{self.action}::{self.role}"
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
        else:
            self.evidence_against.append(label)
        self._recompute_status()

    def _recompute_status(self) -> None:
        if self.support > 0 and self.support >= self.contradictions:
            self.status = HypothesisStatus.CONFIRMED
        elif self.contradictions >= 3 and self.support == 0:
            self.status = HypothesisStatus.REFUTED
        else:
            self.status = HypothesisStatus.UNRESOLVED


@dataclass(frozen=True)
class PreparationPlan:
    """One justified preparation action."""

    action: str
    role: str
    target_predicate: str
    hypothesis_key: str
    reason: str


@dataclass(frozen=True)
class PreparationStepResult:
    """Observed result of executing one preparation action."""

    plan: PreparationPlan
    predicates_before: List[str]
    predicates_after: List[str]
    ready_before: bool
    ready_after: bool
    update: LiveTransitionUpdate
    next_frame: Any
    hypothesis_status: HypothesisStatus

    @property
    def action(self) -> str:
        return self.plan.action

    @property
    def informative(self) -> bool:
        return (
            self.ready_after
            or self.plan.target_predicate in self.predicates_after
            or self.hypothesis_status == HypothesisStatus.REFUTED
        )


class PrepareCorrespondencePolicy:
    """Prepare a correspondence option one primitive action at a time."""

    def __init__(
        self,
        hypotheses: Sequence[PreparationActionHypothesis] | None = None,
        *,
        max_repeat_move_attempts: int = 2,
    ) -> None:
        self.hypotheses = list(hypotheses or default_ar25_preparation_hypotheses())
        self.max_repeat_move_attempts = max(1, int(max_repeat_move_attempts))
        self._active_move_key = ""
        self._active_move_repeats = 0

    def choose(
        self,
        *,
        option: TheoryOption,
        predicates_present: Iterable[str],
        available_actions: Sequence[str],
    ) -> PreparationPlan | None:
        """Return a justified one-step preparation action, or None."""
        present = {str(predicate).strip().lower() for predicate in predicates_present}
        available = {str(action).upper() for action in available_actions}
        if "ready_to_validate_correspondence" in present:
            return None
        if "selected_pair_exists" not in present:
            return None
        if "controller_on_source" not in present:
            return self._choose_for_predicate(
                "controller_on_source",
                role="move",
                available=available,
                excluded={option.policy_action},
            )
        if "recent_control_switch" not in present:
            return self._choose_for_predicate(
                "recent_control_switch",
                role="control_switch",
                available=available,
                excluded={option.policy_action},
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
    ) -> PreparationStepResult | None:
        """Execute one justified preparation action and observe predicates."""
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
        ready_before = "ready_to_validate_correspondence" in predicates_before
        ready_after = "ready_to_validate_correspondence" in predicates_after
        succeeded = ready_after or plan.target_predicate in predicates_after
        hypothesis = self._hypothesis(plan.hypothesis_key)
        hypothesis.observe(
            succeeded=succeeded,
            label=f"observed:{plan.action}:step{int(step)}",
        )
        self._remember_attempt(plan, succeeded=succeeded)
        return PreparationStepResult(
            plan=plan,
            predicates_before=sorted(predicates_before),
            predicates_after=sorted(predicates_after),
            ready_before=ready_before,
            ready_after=ready_after,
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
        excluded: set[str],
    ) -> PreparationPlan | None:
        candidates = [
            hypothesis for hypothesis in self.hypotheses
            if hypothesis.role == role
            and hypothesis.target_predicate == target_predicate
            and hypothesis.action in available
            and hypothesis.action not in excluded
            and hypothesis.status != HypothesisStatus.REFUTED
        ]
        if not candidates:
            return None
        if role == "move" and self._active_move_key:
            active = self._hypothesis(self._active_move_key)
            if (
                active in candidates
                and self._active_move_repeats < self.max_repeat_move_attempts
            ):
                return _plan_from_hypothesis(active)
        return _plan_from_hypothesis(candidates[0])

    def _remember_attempt(
        self,
        plan: PreparationPlan,
        *,
        succeeded: bool,
    ) -> None:
        if plan.role != "move":
            self._active_move_key = ""
            self._active_move_repeats = 0
            return
        if succeeded:
            self._active_move_key = ""
            self._active_move_repeats = 0
            return
        if self._active_move_key == plan.hypothesis_key:
            self._active_move_repeats += 1
        else:
            self._active_move_key = plan.hypothesis_key
            self._active_move_repeats = 1

    def _hypothesis(self, key: str) -> PreparationActionHypothesis:
        for hypothesis in self.hypotheses:
            if hypothesis.key == key:
                return hypothesis
        raise KeyError(key)


def default_ar25_preparation_hypotheses() -> List[PreparationActionHypothesis]:
    """Minimal ar25 prep hypotheses: directional move, then control switch."""
    return [
        PreparationActionHypothesis(
            action="ACTION4",
            role="move",
            target_predicate="controller_on_source",
        ),
        PreparationActionHypothesis(
            action="ACTION1",
            role="move",
            target_predicate="controller_on_source",
        ),
        PreparationActionHypothesis(
            action="ACTION3",
            role="move",
            target_predicate="controller_on_source",
        ),
        PreparationActionHypothesis(
            action="ACTION5",
            role="control_switch",
            target_predicate="recent_control_switch",
        ),
    ]


def _plan_from_hypothesis(hypothesis: PreparationActionHypothesis) -> PreparationPlan:
    return PreparationPlan(
        action=hypothesis.action,
        role=hypothesis.role,
        target_predicate=hypothesis.target_predicate,
        hypothesis_key=hypothesis.key,
        reason=(
            f"{hypothesis.key} tries to establish "
            f"{hypothesis.target_predicate}"
        ),
    )


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
