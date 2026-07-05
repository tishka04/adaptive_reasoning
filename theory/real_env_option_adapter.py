"""Execute theory options against a live environment.

A7 keeps the boundary deliberately narrow: a confirmed option may execute its
fixed policy action only when its confirmed initiation predicate is already
true. Reaching that state is a later problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Sequence

import numpy as np

from .correspondence_hypothesis import CorrespondenceObservation
from .epistemic_metrics import EpistemicScore, MechanicsOracle
from .live_transition_loop import LiveTransitionBeliefLoop, LiveTransitionUpdate
from .precondition_hypothesis import (
    PreconditionObservation,
    extract_precondition_predicates,
)
from .theory_option import TheoryOption, TheoryOptionInvocation

ActionConverter = Callable[[str], Any]


@dataclass(frozen=True)
class EnvFrameSnapshot:
    """Small, duck-typed view of an ARC env frame."""

    grid: np.ndarray
    available_actions: List[str] = field(default_factory=list)
    game_state: str = "NOT_FINISHED"
    levels_completed: int = 0


@dataclass(frozen=True)
class TheoryOptionRunnerResult:
    """Outcome of one attempted live option invocation."""

    option_name: str
    target_rule: str
    policy_action: str
    predicates_present: List[str] = field(default_factory=list)
    initiated: bool = False
    transition_update: LiveTransitionUpdate | None = None
    invocation: TheoryOptionInvocation | None = None
    next_frame: Any | None = None
    reason: str = ""

    @property
    def action_executed(self) -> bool:
        return self.initiated and self.transition_update is not None

    @property
    def transition_produced(self) -> bool:
        return self.transition_update is not None

    @property
    def termination(self) -> str:
        if self.invocation is None:
            return "not_initiated" if not self.initiated else "unknown"
        return self.invocation.termination

    @property
    def success(self) -> bool:
        return bool(self.invocation and self.invocation.success)

    @property
    def contradiction(self) -> bool:
        return bool(self.invocation and self.invocation.contradiction)


@dataclass
class TheoryOptionRunnerRun:
    """Summary of option executions produced by a live runner."""

    loop: LiveTransitionBeliefLoop
    results: List[TheoryOptionRunnerResult] = field(default_factory=list)

    @property
    def option_invocations(self) -> int:
        return sum(1 for result in self.results if result.invocation is not None)

    @property
    def transition_count(self) -> int:
        return sum(1 for result in self.results if result.transition_produced)

    @property
    def successful_invocations(self) -> List[TheoryOptionInvocation]:
        return [
            result.invocation
            for result in self.results
            if result.invocation is not None and result.invocation.success
        ]

    @property
    def contradicted_invocations(self) -> List[TheoryOptionInvocation]:
        return [
            result.invocation
            for result in self.results
            if result.invocation is not None and result.invocation.contradiction
        ]

    @property
    def termination_success_rate(self) -> float:
        if self.option_invocations == 0:
            return 0.0
        return len(self.successful_invocations) / self.option_invocations

    def score(self, oracle: MechanicsOracle) -> EpistemicScore:
        return self.loop.score(oracle, experiment_actions=self.transition_count)

    def summary(self) -> dict[str, Any]:
        return {
            "transitions": self.loop.transition_count,
            "option_invocations": self.option_invocations,
            "option_successes": len(self.successful_invocations),
            "option_contradictions": len(self.contradicted_invocations),
            "termination_success_rate": round(self.termination_success_rate, 4),
            "trace_dependent": False,
        }


class TheoryOptionRunner:
    """Run one confirmed ``TheoryOption`` against a live env object."""

    def __init__(
        self,
        loop: LiveTransitionBeliefLoop,
        option: TheoryOption,
        *,
        action_converter: ActionConverter | None = None,
    ) -> None:
        self.loop = loop
        self.option = option
        self.action_converter = action_converter or _default_action_converter
        self.previous_action = ""
        self.recent_actions: List[str] = []
        self.recent_correspondence_successes: List[bool] = []
        self.results: List[TheoryOptionRunnerResult] = []

    def run_once(
        self,
        env: Any,
        current_frame: Any,
        *,
        previous_action: str | None = None,
        recent_actions: Iterable[Any] | None = None,
        recent_correspondence_successes: Iterable[bool] | None = None,
        step: int | None = None,
        was_experiment: bool = True,
    ) -> TheoryOptionRunnerResult:
        """Attempt one live invocation, producing at most one env transition."""
        before = snapshot_frame(current_frame)
        self.loop.theory.seed_actions(before.available_actions)
        prev_action = (
            _normalize_action_name(previous_action)
            if previous_action is not None
            else self.previous_action
        )
        recent = (
            [_normalize_action_name(action) for action in recent_actions]
            if recent_actions is not None
            else list(self.recent_actions)
        )
        recent_successes = (
            [bool(value) for value in recent_correspondence_successes]
            if recent_correspondence_successes is not None
            else list(self.recent_correspondence_successes)
        )
        predicates = extract_precondition_predicates(
            before.grid,
            target_rule=self.option.target_rule,
            pair_colors=self.option.pair_colors,
            previous_action=prev_action,
            recent_actions=recent,
            recent_correspondence_successes=recent_successes,
        )

        if not self.option.can_initiate(
            self.loop.theory,
            predicates,
            available_actions=before.available_actions,
        ):
            return self._record_result(
                TheoryOptionRunnerResult(
                    option_name=self.option.name,
                    target_rule=self.option.target_rule,
                    policy_action=self.option.policy_action,
                    predicates_present=sorted(predicates),
                    initiated=False,
                    reason="initiation_false",
                )
            )

        policy_action = self.option.policy()
        after_frame = _step_env(env, self.action_converter(policy_action))
        if after_frame is None:
            return self._record_result(
                TheoryOptionRunnerResult(
                    option_name=self.option.name,
                    target_rule=self.option.target_rule,
                    policy_action=policy_action,
                    predicates_present=sorted(predicates),
                    initiated=True,
                    reason="env_returned_no_frame",
                )
            )

        after = snapshot_frame(
            after_frame,
            fallback_available_actions=before.available_actions,
        )
        update = self.loop.observe_grids(
            action=policy_action,
            grid_before=before.grid,
            grid_after=after.grid,
            available_actions=before.available_actions or after.available_actions,
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            levels_completed_before=before.levels_completed,
            levels_completed_after=after.levels_completed,
            timestamp=self.loop.transition_count if step is None else int(step),
            was_experiment=was_experiment,
        )
        invocation = self.option.observe_termination(
            step=self.loop.transition_count - 1 if step is None else int(step),
            actual_action=policy_action,
            predicates_present=predicates,
            update=update,
        )
        succeeded = bool(invocation and invocation.success)
        self.loop.theory.observe_precondition(
            PreconditionObservation(
                target_rule=self.option.target_rule,
                action=policy_action,
                predicates_present=predicates,
                succeeded=succeeded,
            ),
            was_experiment=was_experiment,
        )
        self._remember(policy_action, update)
        return self._record_result(
            TheoryOptionRunnerResult(
                option_name=self.option.name,
                target_rule=self.option.target_rule,
                policy_action=policy_action,
                predicates_present=sorted(predicates),
                initiated=True,
                transition_update=update,
                invocation=invocation,
                next_frame=after_frame,
                reason="executed",
            )
        )

    def run(self) -> TheoryOptionRunnerRun:
        """Return an aggregate view of all attempts made by this runner."""
        return TheoryOptionRunnerRun(loop=self.loop, results=list(self.results))

    def _record_result(
        self,
        result: TheoryOptionRunnerResult,
    ) -> TheoryOptionRunnerResult:
        self.results.append(result)
        return result

    def _remember(self, action: str, update: LiveTransitionUpdate) -> None:
        observation = CorrespondenceObservation.from_transition(
            update.record,
            pair_colors=self.option.pair_colors,
        )
        self.previous_action = action
        self.recent_actions.append(action)
        self.recent_actions = self.recent_actions[-4:]
        self.recent_correspondence_successes.append(
            bool(observation.improves or update.effect.level_complete)
        )
        self.recent_correspondence_successes = (
            self.recent_correspondence_successes[-4:]
        )


def snapshot_frame(
    frame: Any,
    *,
    fallback_available_actions: Sequence[Any] | None = None,
) -> EnvFrameSnapshot:
    """Extract the observation fields needed by the belief loop."""
    grid = _frame_to_grid(frame)
    available = _normalize_actions(
        getattr(frame, "available_actions", None)
        or fallback_available_actions
        or []
    )
    return EnvFrameSnapshot(
        grid=grid,
        available_actions=available,
        game_state=_state_name(getattr(frame, "state", "NOT_FINISHED")),
        levels_completed=_safe_int(getattr(frame, "levels_completed", 0)),
    )


def _step_env(env: Any, action: Any) -> Any:
    if hasattr(env, "take_action"):
        return env.take_action(action)
    if hasattr(env, "step"):
        return env.step(action)
    raise TypeError("env must expose step(action) or take_action(action)")


def _frame_to_grid(frame: Any) -> np.ndarray:
    raw = getattr(frame, "frame", frame)
    if raw is None:
        raise ValueError("frame has no grid data")
    arr = np.asarray(raw, dtype=np.int32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[-1] if arr.shape[0] <= arr.shape[-1] else arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D frame grid, got shape {arr.shape}")
    return arr.astype(np.int32, copy=False)


def _default_action_converter(action_name: str) -> Any:
    try:
        from arcengine import GameAction  # type: ignore

        if hasattr(GameAction, "from_name"):
            return GameAction.from_name(action_name)
        return getattr(GameAction, action_name)
    except Exception:
        return action_name


def _normalize_actions(actions: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for action in actions:
        name = _normalize_action_name(action)
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_action_name(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, (int, np.integer)):
        if int(action) == 0:
            return "RESET"
        return f"ACTION{int(action)}"
    name = getattr(action, "name", None)
    if name:
        return str(name).strip().upper()
    raw = str(action).strip().upper()
    if "." in raw:
        raw = raw.split(".")[-1]
    if raw.isdigit():
        if int(raw) == 0:
            return "RESET"
        return f"ACTION{raw}"
    return raw


def _state_name(state: Any) -> str:
    name = getattr(state, "name", None)
    if name:
        return str(name)
    return str(state or "NOT_FINISHED")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
