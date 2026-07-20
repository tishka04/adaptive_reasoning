"""Bounded persistence after online downstream-objective progress.

SAGE.8r learns which semantic action is directionally compatible in a latent
mode.  That knowledge is only useful if the controller keeps pursuing the same
objective after the first measured reduction.  This policy grants additional
attempts, rollout budget, and credit horizon only after real pursuit progress;
trigger progress and priors never unlock persistence.
"""

from __future__ import annotations

from typing import Any, Dict


class OnlinePersistentPursuitPolicy:
    """Allocate bounded continuation budget from observed pursuit progress."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        base_actions_per_subgoal: int = 2,
        actions_per_progress: int = 2,
        max_actions_per_subgoal: int = 6,
        rollout_actions_per_progress: int = 2,
        max_rollout_actions: int = 10,
        credit_steps_per_progress: int = 4,
        max_credit_window: int = 16,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_actions_per_subgoal = max(
            1,
            int(base_actions_per_subgoal),
        )
        self.actions_per_progress = max(0, int(actions_per_progress))
        self.max_actions_per_subgoal = max(
            self.base_actions_per_subgoal,
            int(max_actions_per_subgoal),
        )
        self.rollout_actions_per_progress = max(
            0,
            int(rollout_actions_per_progress),
        )
        self.max_rollout_actions = max(1, int(max_rollout_actions))
        self.credit_steps_per_progress = max(
            0,
            int(credit_steps_per_progress),
        )
        self.max_credit_window = max(1, int(max_credit_window))
        self._commitment_selections = 0
        self._resumed_commitments = 0
        self._continuation_actions = 0
        self._directional_policy_actions = 0
        self._bridge_actions = 0
        self._entity_contrast_actions = 0
        self._mode_contrast_actions = 0
        self._continuation_progress_events = 0
        self._repeated_progress_events = 0
        self._completed_objectives = 0
        self._attempt_budget_extensions = 0
        self._rollout_budget_extensions = 0
        self._credit_window_extensions = 0
        self._closed_commitments = 0
        self._longest_continuation = 0
        self._continuation_lengths: Dict[str, int] = {}

    def action_limit(self, pursuit_progress_events: int) -> int:
        progress = max(0, int(pursuit_progress_events))
        if not self.enabled or progress <= 0:
            return self.base_actions_per_subgoal
        return min(
            self.max_actions_per_subgoal,
            self.base_actions_per_subgoal
            + self.actions_per_progress * progress,
        )

    def rollout_budget(
        self,
        base_budget: int,
        pursuit_progress_events: int,
    ) -> int:
        budget = max(1, int(base_budget))
        progress = max(0, int(pursuit_progress_events))
        if not self.enabled or progress <= 0:
            return budget
        return min(
            self.max_rollout_actions,
            budget + self.rollout_actions_per_progress * progress,
        )

    def credit_window(
        self,
        base_window: int,
        pursuit_progress_events: int,
    ) -> int:
        window = max(1, int(base_window))
        progress = max(0, int(pursuit_progress_events))
        if not self.enabled or progress <= 0:
            return window
        return min(
            self.max_credit_window,
            window + self.credit_steps_per_progress * progress,
        )

    def note_commitment_selection(
        self,
        *,
        subgoal_id: str,
        previous_subgoal_id: str,
        pursuit_progress_events: int,
        attempts: int,
    ) -> bool:
        """Record a selection that continues a progress-supported objective."""
        if not self.enabled or int(pursuit_progress_events) <= 0:
            return False
        continued = bool(
            str(subgoal_id)
            and str(subgoal_id) == str(previous_subgoal_id)
        )
        self._commitment_selections += 1
        if not continued:
            self._resumed_commitments += 1
        if int(attempts) >= self.base_actions_per_subgoal:
            self._attempt_budget_extensions += 1
        return continued

    def note_action_selection(
        self,
        *,
        subgoal_id: str,
        persistent: bool,
        directional_status: str,
    ) -> None:
        if not self.enabled or not persistent or not str(subgoal_id):
            return
        self._continuation_actions += 1
        status = str(directional_status)
        if status in {"progressive", "bridge"}:
            self._directional_policy_actions += 1
            if status == "bridge":
                self._bridge_actions += 1
        elif status == "needs_entity_contrast":
            self._entity_contrast_actions += 1
        elif status == "needs_mode_contrast":
            self._mode_contrast_actions += 1
        length = self._continuation_lengths.get(str(subgoal_id), 0) + 1
        self._continuation_lengths[str(subgoal_id)] = length
        self._longest_continuation = max(self._longest_continuation, length)

    def note_transition(
        self,
        *,
        persistent: bool,
        progress: bool,
        previous_progress_events: int,
        completed: bool,
    ) -> None:
        if not self.enabled or not persistent:
            return
        if progress:
            self._continuation_progress_events += 1
            if int(previous_progress_events) > 0:
                self._repeated_progress_events += 1
        if completed:
            self._completed_objectives += 1

    def note_rollout_budget_extension(self) -> None:
        if self.enabled:
            self._rollout_budget_extensions += 1

    def note_credit_window_extension(self) -> None:
        if self.enabled:
            self._credit_window_extensions += 1

    def close_commitment(self, subgoal_id: str) -> None:
        if not self.enabled or not str(subgoal_id):
            return
        if self._continuation_lengths.pop(str(subgoal_id), 0) > 0:
            self._closed_commitments += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "base_actions_per_subgoal": self.base_actions_per_subgoal,
            "actions_per_progress": self.actions_per_progress,
            "max_actions_per_subgoal": self.max_actions_per_subgoal,
            "rollout_actions_per_progress": self.rollout_actions_per_progress,
            "max_rollout_actions": self.max_rollout_actions,
            "credit_steps_per_progress": self.credit_steps_per_progress,
            "max_credit_window": self.max_credit_window,
            "commitment_selections": self._commitment_selections,
            "resumed_commitments": self._resumed_commitments,
            "continuation_actions": self._continuation_actions,
            "directional_policy_actions": self._directional_policy_actions,
            "bridge_actions": self._bridge_actions,
            "entity_contrast_actions": self._entity_contrast_actions,
            "mode_contrast_actions": self._mode_contrast_actions,
            "continuation_progress_events": (
                self._continuation_progress_events
            ),
            "repeated_progress_events": self._repeated_progress_events,
            "completed_objectives": self._completed_objectives,
            "attempt_budget_extensions": self._attempt_budget_extensions,
            "rollout_budget_extensions": self._rollout_budget_extensions,
            "credit_window_extensions": self._credit_window_extensions,
            "closed_commitments": self._closed_commitments,
            "active_commitments": len(self._continuation_lengths),
            "longest_continuation": self._longest_continuation,
        }


__all__ = ["OnlinePersistentPursuitPolicy"]
