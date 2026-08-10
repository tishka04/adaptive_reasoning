"""Scheduled real-time continuation for the T10.3.4 source controller.

T10.3.4 showed that an active SAGE.T option is cheap, while returning to the
full UnifiedCognitiveController decision and observation paths is not.  This
module keeps the unified controller as the integration boundary but replaces
those unbounded paths with a symmetric, preregistered real-time schedule:

* both arms receive the same cheap symbolic proposal;
* SAGE.T may override it in the active arm;
* every real transition updates the lightweight action/effect model and the
  same SAGE.T posterior;
* productive options may be extended online, but never beyond 32 actions;
* no coordinates or entity identities are added to persisted programs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..live_transition_loop import build_observation
from ..unified_cognitive_controller import (
    CognitiveDecision,
    _align_grids,
    _normalize_action,
    _normalize_actions,
)
from .contracts import ActionCandidate
from .goal_directed_v10_3_2 import (
    MAXIMUM_OPTION_HORIZON,
    GoalDirectedOption,
    OptionStep,
)
from .goal_directed_v10_3_4 import (
    DISCOVERY_WARMUP_ACTIONS,
    EXPLORATION_ACTIONS_BETWEEN_OPTIONS,
    TRANSITION_HISTORY_LIMIT,
    BoundedGoalDirectedSageTController,
    BoundedUnifiedCognitiveController,
    bounded_unified_config,
)

FORMAT_VERSION = "sage-t10.3.5-scheduled-real-time-v1"
MAXIMUM_CONTROLLER_CYCLE_P95_MS = 2500.0


def scheduled_unified_config(*, sage_t_authority_mode: str):
    """Reuse the symmetric frozen low-growth configuration from T10.3.4."""

    return bounded_unified_config(sage_t_authority_mode=sage_t_authority_mode)


def _extended_steps(option: GoalDirectedOption) -> tuple[OptionStep, ...]:
    """Double a productive option without exceeding the frozen 32-step cap."""

    current = option.steps
    target = min(MAXIMUM_OPTION_HORIZON, max(len(current) + 1, len(current) * 2))
    if option.schema == "mixed_automaton":
        return tuple(current[index % len(current)] for index in range(target))
    return tuple((*current, *(current[-1] for _ in range(target - len(current)))))


class ScheduledGoalDirectedSageTController(BoundedGoalDirectedSageTController):
    """Relational SAGE.T controller with evidence-gated option prolongation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("warmup_actions", DISCOVERY_WARMUP_ACTIONS)
        kwargs.setdefault(
            "exploration_interval", EXPLORATION_ACTIONS_BETWEEN_OPTIONS
        )
        super().__init__(*args, **kwargs)
        self._productive_option_extensions = 0
        self._terminal_option_contradictions = 0

    def observe_transition(self, record: Any) -> None:
        active = self._pending_option
        at_registered_end = bool(
            active is not None
            and self._active_cursor + 1 >= len(active.steps)
            and len(active.steps) < MAXIMUM_OPTION_HORIZON
        )
        productive = bool(
            not record.diff.is_noop
            and not record.diff.game_over
            and not (
                record.diff.level_complete
                or record.obs_after.levels_completed
                > record.obs_before.levels_completed
            )
        )
        if at_registered_end and productive and active is not None:
            extended = GoalDirectedOption(
                schema=active.schema,
                steps=_extended_steps(active),
                initiation=active.initiation,
                termination=active.termination,
                source="online_productive_extension",
            )
            self._active_option = extended
            self._pending_option = extended
            self._productive_option_extensions += 1
        if active is not None and record.diff.game_over:
            self._terminal_option_contradictions += 1
        super().observe_transition(record)

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "productive_option_extensions": self._productive_option_extensions,
                "terminal_option_contradictions": (
                    self._terminal_option_contradictions
                ),
                "maximum_extended_option_horizon": MAXIMUM_OPTION_HORIZON,
            }
        )
        return base


class ScheduledUnifiedCognitiveController(BoundedUnifiedCognitiveController):
    """Unified integration shell with a strictly lightweight online schedule."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._scheduled_sage_decisions = 0
        self._scheduled_legacy_decisions = 0
        self._lightweight_observations = 0
        self._full_unified_decisions = 0
        self._full_unified_observations = 0

    @staticmethod
    def _legal_candidates(
        available_action_candidates: Sequence[Any] | None,
        safe_actions: Sequence[str],
    ) -> tuple[Any, ...]:
        if available_action_candidates is None:
            return tuple(safe_actions)
        safe = set(safe_actions)
        filtered = tuple(
            candidate
            for candidate in available_action_candidates
            if _normalize_action(candidate) in safe
        )
        return filtered or tuple(available_action_candidates)

    def select_action(
        self,
        *,
        current_grid: Any,
        available_actions: Sequence[Any],
        legacy_action: Any,
        legacy_action_data: Mapping[str, Any] | None = None,
        available_action_candidates: Sequence[Any] | None = None,
        game_state: str = "NOT_FINISHED",
        levels_completed: int = 0,
    ) -> CognitiveDecision:
        """Arbitrate directly between the shared proposal and SAGE.T."""

        self._step += 1
        self._branch_step += 1
        actions = _normalize_actions(available_actions)
        legacy_name = _normalize_action(legacy_action)
        if legacy_name not in actions and actions:
            legacy_name = actions[0]
        self.theory.seed_actions(actions)
        observation = build_observation(
            current_grid,
            available_actions=actions,
            game_state=game_state,
            levels_completed=levels_completed,
            infer_players=True,
        )
        safe_actions = self._safe_actions(observation.grid_hash, actions) or list(
            actions
        )
        legal = self._legal_candidates(available_action_candidates, safe_actions)
        goal = self.sage_t_controller
        selected: ActionCandidate | None = None
        arbitration = None
        if isinstance(goal, ScheduledGoalDirectedSageTController):
            if goal.fast_path_ready:
                selected = goal.fast_active_decision(
                    symbolic_action_name=legacy_name,
                    symbolic_action_data=legacy_action_data,
                    observation=observation,
                    legal_actions=legal,
                    protected_route=False,
                    danger_veto=None,
                )
                if selected is not None:
                    self._bounded_fast_path_decisions += 1
                else:
                    self._bounded_fast_path_fallbacks += 1
            else:
                arbitration = goal.decide(
                    symbolic_action_name=legacy_name,
                    symbolic_action_data=legacy_action_data,
                    observation=observation,
                    legal_actions=legal,
                    mechanic_theory=self.theory,
                    goal_hypotheses=(),
                    route_memory=None,
                    danger_veto=None,
                    protected_route=False,
                )
                if arbitration.applied:
                    selected = ActionCandidate(
                        action_name=arbitration.action_name,
                        action_data=dict(arbitration.action_data),
                    )

        if selected is not None and selected.action_name in safe_actions:
            decision = CognitiveDecision(
                action_name=selected.action_name,
                action_data=dict(selected.action_data),
                source="sage_t_joint_program",
                reason=(
                    "scheduled_active_option_fast_path"
                    if arbitration is None
                    else "scheduled_sage_t_arbitration"
                ),
                confidence=max(
                    (
                        particle.probability
                        for particle in getattr(goal, "posterior", ()).particles
                    ),
                    default=0.0,
                ),
            )
            self._scheduled_sage_decisions += 1
        else:
            decision = CognitiveDecision(
                action_name=legacy_name,
                action_data=dict(legacy_action_data or {}),
                source="scheduled_legacy_proposal",
                reason="shared bounded symbolic proposal",
            )
            self._scheduled_legacy_decisions += 1
        self._pending_decision = decision
        self._pending_action_candidates = tuple(available_action_candidates or ())
        self._decision_sources[decision.source] += 1
        return decision

    def observe_transition(
        self,
        *,
        action: Any,
        grid_before: Any,
        grid_after: Any,
        available_actions: Sequence[Any] | None = None,
        game_state_before: str = "NOT_FINISHED",
        game_state_after: str = "NOT_FINISHED",
        levels_completed_before: int = 0,
        levels_completed_after: int = 0,
        action_data: Mapping[str, Any] | None = None,
    ):
        """Update only the live effect model and SAGE.T posterior."""

        action_name = _normalize_action(action)
        actions = _normalize_actions(available_actions or self.theory.actions())
        self.theory.seed_actions(actions)
        aligned_before, aligned_after = _align_grids(grid_before, grid_after)
        update = self.belief_loop.observe_grids(
            action=action_name,
            action_args=dict(action_data or {}),
            grid_before=aligned_before,
            grid_after=aligned_after,
            available_actions=actions,
            game_state_before=game_state_before,
            game_state_after=game_state_after,
            levels_completed_before=levels_completed_before,
            levels_completed_after=levels_completed_after,
            timestamp=self._observed_transitions,
            was_experiment=False,
        )
        self._observed_transitions += 1
        if update.record.diff.is_noop:
            self.anti_attractor.note_no_effect(
                update.record.obs_before.grid_hash,
                action_name,
            )
        self.anti_attractor.observe(
            grid_hash=update.record.obs_after.grid_hash,
            action_name=action_name,
            is_noop=bool(update.record.diff.is_noop),
        )
        if self.sage_t_controller is not None:
            self.sage_t_controller.observe_transition(update.record)
        transitions = self.belief_loop.profiler.transitions
        if len(transitions) > TRANSITION_HISTORY_LIMIT:
            del transitions[:-TRANSITION_HISTORY_LIMIT]
        self._maximum_retained_transitions = max(
            self._maximum_retained_transitions,
            len(transitions),
        )
        self._lightweight_observations += 1
        self._pending_decision = None
        self._pending_action_candidates = ()
        return update

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "scheduled_real_time_profile": True,
                "scheduled_sage_decisions": self._scheduled_sage_decisions,
                "scheduled_legacy_decisions": self._scheduled_legacy_decisions,
                "lightweight_observations": self._lightweight_observations,
                "full_unified_decisions": self._full_unified_decisions,
                "full_unified_observations": self._full_unified_observations,
                "posterior_updated_each_transition": True,
            }
        )
        return base


__all__ = [
    "FORMAT_VERSION",
    "MAXIMUM_CONTROLLER_CYCLE_P95_MS",
    "ScheduledGoalDirectedSageTController",
    "ScheduledUnifiedCognitiveController",
    "scheduled_unified_config",
]
