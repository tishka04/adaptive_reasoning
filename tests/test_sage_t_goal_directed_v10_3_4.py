from __future__ import annotations

from types import MethodType

import numpy as np

from theory.sage_t.goal_directed_v10_3_2 import GoalDirectedOption, OptionStep
from theory.sage_t.goal_directed_v10_3_4 import (
    OPERATOR_INDUCTION_INTERVAL,
    TRANSITION_HISTORY_LIMIT,
    BoundedGoalDirectedSageTController,
    BoundedUnifiedCognitiveController,
    bounded_unified_config,
)


class _Action:
    def __init__(self, name: str, action_args: dict | None = None) -> None:
        self.name = name
        self.action_args = action_args or {}


def test_bounded_profile_is_symmetric_and_disables_growth_modules() -> None:
    active = bounded_unified_config(sage_t_authority_mode="active")
    baseline = bounded_unified_config(sage_t_authority_mode="off")
    assert active.enable_operator_planning is False
    assert baseline.enable_operator_planning is False
    assert active.enable_horizon_stable_learning_epochs is False
    assert baseline.enable_horizon_stable_learning_epochs is False
    assert active.operator_induction_interval == OPERATOR_INDUCTION_INTERVAL
    assert baseline.operator_induction_interval == OPERATOR_INDUCTION_INTERVAL
    assert active.sage_t_authority_mode == "active"
    assert baseline.sage_t_authority_mode == "off"


def test_active_option_uses_fast_path_without_full_unified_search() -> None:
    goal = BoundedGoalDirectedSageTController(
        phase="preflight",
        warmup_actions=0,
        exploration_interval=1,
    )
    controller = BoundedUnifiedCognitiveController(
        "synthetic",
        config=bounded_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    goal._active_option = GoalDirectedOption(
        schema="repeat_target",
        steps=(OptionStep("ACTION1"), OptionStep("ACTION1")),
    )

    def forbidden_operator_search(*_args, **_kwargs):
        raise AssertionError("full unified operator search was entered")

    controller._select_operator_plan = MethodType(
        forbidden_operator_search,
        controller,
    )
    decision = controller.select_action(
        current_grid=np.zeros((5, 5), dtype=np.int16),
        available_actions=("ACTION1",),
        legacy_action="ACTION1",
        legacy_action_data={},
        available_action_candidates=(_Action("ACTION1"),),
    )
    assert decision.source == "sage_t_joint_program"
    assert controller.summary()["bounded_fast_path_decisions"] == 1
    assert goal.summary()["fast_path_applied"] == 1


def test_fast_path_grounding_miss_fails_closed_to_legacy_once() -> None:
    goal = BoundedGoalDirectedSageTController(phase="preflight")
    controller = BoundedUnifiedCognitiveController(
        "synthetic",
        config=bounded_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    goal._active_option = GoalDirectedOption(
        schema="mixed_automaton",
        steps=(OptionStep("ACTION2"),),
    )
    decision = controller.select_action(
        current_grid=np.zeros((5, 5), dtype=np.int16),
        available_actions=("ACTION1",),
        legacy_action="ACTION1",
        legacy_action_data={},
        available_action_candidates=(_Action("ACTION1"),),
    )
    assert decision.source == "bounded_legacy_fallback"
    assert goal.fast_path_ready is False
    assert controller.summary()["bounded_fast_path_fallbacks"] == 1


def test_transition_history_is_strictly_capped() -> None:
    controller = BoundedUnifiedCognitiveController(
        "synthetic",
        config=bounded_unified_config(sage_t_authority_mode="off"),
    )
    controller.on_reset()
    grid = np.zeros((5, 5), dtype=np.int16)
    legal = (_Action("ACTION1"),)
    for index in range(TRANSITION_HISTORY_LIMIT + 5):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=("ACTION1",),
            legacy_action="ACTION1",
            legacy_action_data={},
            available_action_candidates=legal,
        )
        grid = before.copy()
        grid[2, 2] = 1 + ((index + 1) % 2)
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=grid,
            available_actions=("ACTION1",),
        )
    assert len(controller.belief_loop.profiler.transitions) == TRANSITION_HISTORY_LIMIT
    assert (
        controller.summary()["maximum_retained_transitions"] == TRANSITION_HISTORY_LIMIT
    )
