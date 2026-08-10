"""Stable fresh-path continuation for the SAGE.T10.3.6 witness miss.

T10.3.6 correctly reconstructed the first nine canonical SU15 successors, but
recomputed a shorter chain before the tenth action and repeated waypoint nine.
T10.3.7 freezes only the path induced from the fresh reset in ephemeral memory;
each waypoint is still re-grounded against the current legal actions before it
is executed.  Neither the fresh coordinates nor action identities are written
to a transferable program.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import AbstractState, ActionCandidate
from .goal_directed_v10_3_2 import MAXIMUM_OPTION_HORIZON
from .goal_directed_v10_3_3 import DYNAMIC_SUCCESSOR, _same_action_data
from .goal_directed_v10_3_6 import FunctionalGoalDirectedSageTController
from .progress_witness_v10 import GroundedAction, SearchConfig, chain_successor_macro

FORMAT_VERSION = "sage-t10.3.7-stable-fresh-successor-plan-v1"


class StableFreshPathSageTController(FunctionalGoalDirectedSageTController):
    """Execute a fresh structural path without shortening it mid-option."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fresh_successor_plan: tuple[GroundedAction, ...] = ()
        self._fresh_plan_reacquisitions = 0
        self._fresh_plan_grounding_misses = 0

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self._fresh_successor_plan = ()

    def _continue_active_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
    ) -> ActionCandidate | None:
        option = self._active_option
        if option is None or self._active_cursor >= len(option.steps):
            return super()._continue_active_option(state, candidates)
        step = option.steps[self._active_cursor]
        if step.binding_method != DYNAMIC_SUCCESSOR:
            return super()._continue_active_option(state, candidates)
        if not self._fresh_successor_plan:
            grounded = tuple(
                GroundedAction(
                    candidate.action_name,
                    tuple(dict(candidate.action_data).items()),
                )
                for candidate in candidates
            )
            macro = chain_successor_macro(
                state,
                grounded,
                config=SearchConfig(maximum_horizon=MAXIMUM_OPTION_HORIZON),
            )
            if macro is None or not macro.actions:
                self._last_grounding_failure = "fresh_successor_plan_miss"
                self._fresh_plan_grounding_misses += 1
                return None
            self._fresh_successor_plan = tuple(macro.actions)
        if self._active_cursor >= len(self._fresh_successor_plan):
            self._last_grounding_failure = "fresh_successor_plan_exhausted"
            self._fresh_plan_grounding_misses += 1
            return None
        wanted = self._fresh_successor_plan[self._active_cursor]
        matches = [
            candidate
            for candidate in candidates
            if candidate.action_name == wanted.action_name
            and _same_action_data(candidate.action_data, wanted.data)
        ]
        if len(matches) != 1:
            self._last_grounding_failure = "fresh_successor_reacquisition_miss"
            self._fresh_plan_grounding_misses += 1
            return None
        selected = matches[0]
        self._pending_grounded_candidate = selected
        self._binding_method_uses[step.binding_method] += 1
        self._fresh_plan_reacquisitions += 1
        return selected

    def _finish_active_option(self, *, progressed: bool, reason: str) -> None:
        super()._finish_active_option(progressed=progressed, reason=reason)
        self._fresh_successor_plan = ()

    def note_level_change(self) -> None:
        self._fresh_successor_plan = ()
        super().note_level_change()

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "stable_fresh_successor_plan": True,
                "fresh_plan_reacquisitions": self._fresh_plan_reacquisitions,
                "fresh_plan_grounding_misses": self._fresh_plan_grounding_misses,
                "fresh_successor_plan_persisted": False,
            }
        )
        return base


__all__ = ["FORMAT_VERSION", "StableFreshPathSageTController"]
