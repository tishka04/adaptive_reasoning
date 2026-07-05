"""A11 prepare-correspondence option.

The validation option answers: "when ready, execute ACTION2". This module
adds the preceding option: "make the validation context ready" using only
named missing predicates and falsifiable preparation hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence

from .correspondence_hypothesis import CorrespondenceObservation
from .epistemic_metrics import HypothesisStatus
from .live_transition_loop import LiveTransitionBeliefLoop
from .precondition_hypothesis import extract_precondition_predicates
from .real_env_option_adapter import snapshot_frame
from .strong_preparation_policy import (
    StrongPreparationStepResult,
    StrongPrepareCorrespondencePolicy,
    discriminating_readiness_gaps,
)
from .theory_option import TheoryOption


@dataclass(frozen=True)
class PrepareCorrespondenceOptionInvocation:
    """One completed invocation of the preparation option."""

    option_name: str
    target_rule: str
    step_start: int
    step_end: int
    actions: List[str] = field(default_factory=list)
    predicates_initial: List[str] = field(default_factory=list)
    predicates_final: List[str] = field(default_factory=list)
    missing_predicates_initial: List[str] = field(default_factory=list)
    termination: str = "unknown"
    reason: str = ""

    @property
    def success(self) -> bool:
        return self.termination == "success"

    @property
    def contradiction(self) -> bool:
        return self.termination == "contradiction"

    @property
    def maxed(self) -> bool:
        return self.termination == "max_steps"


@dataclass(frozen=True)
class PrepareCorrespondenceOption:
    """An option that establishes strong readiness for a validation option."""

    name: str
    validation_option: TheoryOption
    max_steps: int = 20
    initiation_predicate: str = "strong_ready_to_validate_correspondence_false"
    termination_signals: tuple[str, ...] = (
        "strong_ready_to_validate_correspondence",
        "contradiction",
        "max_steps",
    )

    @property
    def target_rule(self) -> str:
        return self.validation_option.target_rule

    @property
    def pair_colors(self) -> tuple[int, int]:
        return self.validation_option.pair_colors

    @classmethod
    def from_validation_option(
        cls,
        validation_option: TheoryOption,
        *,
        max_steps: int = 20,
    ) -> "PrepareCorrespondenceOption":
        first, second = validation_option.pair_colors
        return cls(
            name=f"prepare_correspondence_colors{first}_{second}",
            validation_option=validation_option,
            max_steps=max(1, int(max_steps)),
        )

    def can_initiate(
        self,
        loop: LiveTransitionBeliefLoop,
        predicates_present: Iterable[str],
        available_actions: Sequence[str],
        *,
        policy: StrongPrepareCorrespondencePolicy | None = None,
    ) -> bool:
        present = _normalize_predicates(predicates_present)
        if _strong_ready(present):
            return False
        if not any(rule.key == self.target_rule for rule in loop.theory.correspondence_rules()):
            return False
        selected_policy = policy or StrongPrepareCorrespondencePolicy()
        return selected_policy.choose(
            option=self.validation_option,
            predicates_present=present,
            available_actions=available_actions,
        ) is not None


@dataclass(frozen=True)
class PrepareCorrespondenceOptionRunnerResult:
    """Result of running one preparation option invocation."""

    option: PrepareCorrespondenceOption
    invocation: PrepareCorrespondenceOptionInvocation | None
    steps: List[StrongPreparationStepResult] = field(default_factory=list)
    next_frame: Any | None = None
    previous_action: str = ""
    recent_actions: List[str] = field(default_factory=list)
    recent_correspondence_successes: List[bool] = field(default_factory=list)

    @property
    def initiated(self) -> bool:
        return self.invocation is not None

    @property
    def success(self) -> bool:
        return bool(self.invocation and self.invocation.success)

    @property
    def contradiction(self) -> bool:
        return bool(self.invocation and self.invocation.contradiction)

    @property
    def maxed(self) -> bool:
        return bool(self.invocation and self.invocation.maxed)

    @property
    def env_actions(self) -> int:
        return len(self.steps)

    @property
    def termination(self) -> str:
        if self.invocation is None:
            return "not_initiated"
        return self.invocation.termination


class PrepareCorrespondenceOptionRunner:
    """Execute a prepare-correspondence option against a live env."""

    def __init__(
        self,
        loop: LiveTransitionBeliefLoop,
        option: PrepareCorrespondenceOption,
        *,
        policy: StrongPrepareCorrespondencePolicy | None = None,
    ) -> None:
        self.loop = loop
        self.option = option
        self.policy = policy or StrongPrepareCorrespondencePolicy()

    def run_once(
        self,
        env: Any,
        current_frame: Any,
        *,
        previous_action: str = "",
        recent_actions: Iterable[str] = (),
        recent_correspondence_successes: Iterable[bool] = (),
        step: int = 0,
    ) -> PrepareCorrespondenceOptionRunnerResult:
        frame = current_frame
        prev = str(previous_action or "").upper()
        recent = [str(action or "").upper() for action in recent_actions]
        recent_successes = [
            bool(value) for value in recent_correspondence_successes
        ]
        initial_predicates = _extract_predicates(
            frame,
            option=self.option.validation_option,
            previous_action=prev,
            recent_actions=recent,
            recent_correspondence_successes=recent_successes,
        )
        snapshot = snapshot_frame(frame)
        if not self.option.can_initiate(
            self.loop,
            initial_predicates,
            snapshot.available_actions,
            policy=self.policy,
        ):
            return PrepareCorrespondenceOptionRunnerResult(
                option=self.option,
                invocation=None,
                next_frame=frame,
                previous_action=prev,
                recent_actions=recent[-4:],
                recent_correspondence_successes=recent_successes[-4:],
            )

        steps: List[StrongPreparationStepResult] = []
        final_predicates = set(initial_predicates)
        termination = "max_steps"
        reason = "max_steps"

        for offset in range(self.option.max_steps):
            if _strong_ready(final_predicates):
                termination = "success"
                reason = "strong_ready_before_step"
                break
            preparation = self.policy.prepare_once(
                env=env,
                loop=self.loop,
                option=self.option.validation_option,
                current_frame=frame,
                previous_action=prev,
                recent_actions=recent,
                recent_correspondence_successes=recent_successes,
                step=int(step) + offset,
            )
            if preparation is None:
                termination = "contradiction"
                reason = "no_applicable_predicate_hypothesis"
                break
            steps.append(preparation)
            frame = preparation.next_frame
            corr = CorrespondenceObservation.from_transition(
                preparation.update.record,
                pair_colors=self.option.pair_colors,
            )
            recent_successes.append(
                bool(corr.improves or preparation.update.effect.level_complete)
            )
            recent_successes = recent_successes[-4:]
            prev = preparation.action
            recent.append(preparation.action)
            recent = recent[-4:]
            final_predicates = set(preparation.predicates_after)
            if preparation.strong_ready_after:
                termination = "success"
                reason = "strong_ready_to_validate_correspondence"
                break
            if preparation.hypothesis_status == HypothesisStatus.REFUTED:
                termination = "contradiction"
                reason = "preparation_hypothesis_refuted"
                break

        if termination == "max_steps" and _strong_ready(final_predicates):
            termination = "success"
            reason = "strong_ready_to_validate_correspondence"

        invocation = PrepareCorrespondenceOptionInvocation(
            option_name=self.option.name,
            target_rule=self.option.target_rule,
            step_start=int(step),
            step_end=int(step) + len(steps),
            actions=[result.action for result in steps],
            predicates_initial=sorted(initial_predicates),
            predicates_final=sorted(final_predicates),
            missing_predicates_initial=sorted(
                discriminating_readiness_gaps(initial_predicates)
            ),
            termination=termination,
            reason=reason,
        )
        return PrepareCorrespondenceOptionRunnerResult(
            option=self.option,
            invocation=invocation,
            steps=steps,
            next_frame=frame,
            previous_action=prev,
            recent_actions=list(recent[-4:]),
            recent_correspondence_successes=list(recent_successes[-4:]),
        )


def _extract_predicates(
    frame: Any,
    *,
    option: TheoryOption,
    previous_action: str,
    recent_actions: Iterable[str],
    recent_correspondence_successes: Iterable[bool],
) -> set[str]:
    snapshot = snapshot_frame(frame)
    return extract_precondition_predicates(
        snapshot.grid,
        target_rule=option.target_rule,
        pair_colors=option.pair_colors,
        previous_action=previous_action,
        recent_actions=recent_actions,
        recent_correspondence_successes=recent_correspondence_successes,
    )


def _strong_ready(predicates: Iterable[str]) -> bool:
    present = _normalize_predicates(predicates)
    return (
        "strong_ready_to_validate_correspondence" in present
        or "source_target_relation_satisfied" in present
    )


def _normalize_predicates(predicates: Iterable[str]) -> set[str]:
    return {str(predicate or "").strip().lower() for predicate in predicates}
