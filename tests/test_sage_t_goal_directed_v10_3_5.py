from __future__ import annotations

from types import MethodType

import numpy as np

from theory.sage_t.goal_directed_v10_3_2 import GoalDirectedOption, OptionStep
from theory.sage_t.goal_directed_v10_3_5 import (
    ScheduledGoalDirectedSageTController,
    ScheduledUnifiedCognitiveController,
    scheduled_unified_config,
)


class _Action:
    def __init__(self, name: str, action_args: dict | None = None) -> None:
        self.name = name
        self.action_args = action_args or {}


def _transition(controller, decision, before, after, *, game_over=False) -> None:
    controller.observe_transition(
        action=decision.action_name,
        action_data=decision.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=("ACTION1",),
        game_state_after="GAME_OVER" if game_over else "NOT_FINISHED",
    )


def test_scheduled_active_and_baseline_never_enter_full_unified_path() -> None:
    for active in (False, True):
        goal = (
            ScheduledGoalDirectedSageTController(
                phase="preflight", warmup_actions=0, exploration_interval=1
            )
            if active
            else None
        )
        controller = ScheduledUnifiedCognitiveController(
            "synthetic",
            config=scheduled_unified_config(
                sage_t_authority_mode="active" if active else "off"
            ),
            sage_t_controller=goal,
        )
        controller.on_reset()

        def forbidden(*_args, **_kwargs):
            raise AssertionError("full unified path was entered")

        controller._select_operator_plan = MethodType(forbidden, controller)
        before = np.zeros((5, 5), dtype=np.int16)
        decision = controller.select_action(
            current_grid=before,
            available_actions=("ACTION1",),
            legacy_action="ACTION1",
            available_action_candidates=(_Action("ACTION1"),),
        )
        after = before.copy()
        after[2, 2] = 1
        _transition(controller, decision, before, after)
        summary = controller.summary()
        assert summary["full_unified_decisions"] == 0
        assert summary["full_unified_observations"] == 0
        assert summary["lightweight_observations"] == 1
        assert summary["transitions_observed"] == 1


def test_productive_option_is_extended_but_never_past_32() -> None:
    goal = ScheduledGoalDirectedSageTController(
        phase="preflight", warmup_actions=0, exploration_interval=1
    )
    controller = ScheduledUnifiedCognitiveController(
        "synthetic",
        config=scheduled_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    goal._active_option = GoalDirectedOption(
        schema="repeat_target",
        steps=(OptionStep("ACTION1"), OptionStep("ACTION1")),
    )
    grid = np.zeros((5, 5), dtype=np.int16)
    for index in range(32):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=("ACTION1",),
            legacy_action="ACTION1",
            available_action_candidates=(_Action("ACTION1"),),
        )
        grid = before.copy()
        grid[2, 2] = 1 + ((index + 1) % 2)
        _transition(controller, decision, before, grid)
    summary = goal.summary()
    assert summary["productive_option_extensions"] == 4
    assert summary["maximum_extended_option_horizon"] == 32
    assert summary["fast_path_applied"] == 32


def test_terminal_option_is_recorded_as_contradiction() -> None:
    goal = ScheduledGoalDirectedSageTController(phase="preflight")
    controller = ScheduledUnifiedCognitiveController(
        "synthetic",
        config=scheduled_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    goal._active_option = GoalDirectedOption(
        schema="repeat_target",
        steps=(OptionStep("ACTION1"), OptionStep("ACTION1")),
    )
    before = np.zeros((5, 5), dtype=np.int16)
    decision = controller.select_action(
        current_grid=before,
        available_actions=("ACTION1",),
        legacy_action="ACTION1",
        available_action_candidates=(_Action("ACTION1"),),
    )
    _transition(controller, decision, before, before.copy(), game_over=True)
    summary = goal.summary()
    assert summary["terminal_option_contradictions"] == 1
    assert summary["option_contradictions"] == 1

