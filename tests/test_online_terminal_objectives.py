"""Terminal grounding tests: mechanics are candidates, levels provide proof."""

from __future__ import annotations

import numpy as np

from theory.promoted_relational_rule import PromotedRelationalRule
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


def _transform_rule() -> PromotedRelationalRule:
    return PromotedRelationalRule(
        action="ACTION6",
        family="color_transform",
        predicate="source_target_color_transform",
        source_color=3,
        target_color=4,
        expected_outcome="3->4",
        source_prediction_key="terminal-grounding-transform",
        support=2,
        independent_contexts={"mechanic-a", "mechanic-b"},
    )


def _grid(*, source_position: tuple[int, int] | None = (2, 2)) -> np.ndarray:
    grid = np.zeros((9, 9), dtype=np.int32)
    if source_position is not None:
        grid[source_position] = 3
    return grid


def _controller(**config_overrides) -> tuple[
    UnifiedCognitiveController,
    PromotedRelationalRule,
]:
    controller = UnifiedCognitiveController(
        "synthetic-terminal-grounding",
        available_actions=["ACTION6"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=0,
            enable_active_goal_hypotheses=False,
            **config_overrides,
        ),
    )
    rule = _transform_rule()
    controller.theory.add_promoted_relational_rule(rule)
    return controller, rule


def _apply_transform(
    controller: UnifiedCognitiveController,
    before: np.ndarray,
    *,
    game_state_after: str = "NOT_FINISHED",
    levels_after: int = 0,
):
    after = before.copy()
    after[after == 3] = 4
    decision = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    controller.observe_transition(
        action="ACTION6",
        action_data=decision.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION6"],
        game_state_after=game_state_after,
        levels_completed_after=levels_after,
    )
    return decision, after


def test_mechanical_distance_reduction_is_not_terminal_proof():
    controller, _ = _controller()
    decision, after = _apply_transform(controller, _grid())

    assert decision.source == "terminal_objective_probe"
    summary = controller.summary()["terminal_objectives"]
    hypothesis = summary["hypotheses"][0]
    assert hypothesis["status"] == "candidate"
    assert hypothesis["distance_reductions"] == 1
    assert hypothesis["terminal_support"] == 0

    # The measured deficit is now zero, so SAGE does not repeat a locally
    # successful effect as if repetition itself implied goal progress.
    next_decision = controller.select_action(
        current_grid=after,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    assert next_decision.source == "legacy_fallback"


def test_delayed_level_completion_credits_recent_distance_reduction():
    controller, _ = _controller(terminal_objective_credit_window=3)
    _, transformed = _apply_transform(controller, _grid())

    terminal_decision = controller.select_action(
        current_grid=transformed,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    controller.observe_transition(
        action="ACTION6",
        action_data=terminal_decision.action_data,
        grid_before=transformed,
        grid_after=transformed,
        available_actions=["ACTION6"],
        levels_completed_before=0,
        levels_completed_after=1,
    )

    summary = controller.summary()["terminal_objectives"]
    assert summary["terminal_supported_objectives"] == 0
    assert summary["objectives_needing_contrast"] == 1
    assert summary["credited_terminal_events"] == 1
    assert summary["hypotheses"][0]["terminal_support"] == 1

    controller.on_reset()
    fresh = _grid(source_position=(4, 4))
    contrast = controller.select_action(
        current_grid=fresh,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    assert contrast.source == "terminal_objective_probe"
    assert contrast.objective_status == "needs_contrast"
    fresh_after = fresh.copy()
    fresh_after[fresh_after == 3] = 4
    controller.observe_transition(
        action="ACTION6",
        action_data=contrast.action_data,
        grid_before=fresh,
        grid_after=fresh_after,
        available_actions=["ACTION6"],
        levels_completed_after=1,
    )
    confirmed = controller.summary()["terminal_objectives"]
    assert confirmed["terminal_supported_objectives"] == 1
    assert confirmed["hypotheses"][0]["terminal_support"] == 2


def test_one_terminal_event_counts_once_after_multiple_reductions():
    controller, _ = _controller(
        max_terminal_objective_probes_per_objective=2,
        terminal_objective_credit_window=3,
    )
    before = _grid(source_position=(2, 2))
    before[5, 5] = 3
    middle = before.copy()
    middle[2, 2] = 4
    first = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    controller.observe_transition(
        action="ACTION6",
        action_data=first.action_data,
        grid_before=before,
        grid_after=middle,
        available_actions=["ACTION6"],
    )

    after = middle.copy()
    after[5, 5] = 4
    second = controller.select_action(
        current_grid=middle,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    controller.observe_transition(
        action="ACTION6",
        action_data=second.action_data,
        grid_before=middle,
        grid_after=after,
        available_actions=["ACTION6"],
        levels_completed_after=1,
    )

    hypothesis = controller.summary()["terminal_objectives"]["hypotheses"][0]
    assert hypothesis["distance_reductions"] == 2
    assert hypothesis["terminal_support"] == 1
    assert hypothesis["independent_terminal_contexts"] == 1


def test_terminal_event_outside_credit_window_does_not_ground_objective():
    controller, _ = _controller(terminal_objective_credit_window=1)
    _, transformed = _apply_transform(controller, _grid())

    for _ in range(2):
        neutral = controller.select_action(
            current_grid=transformed,
            available_actions=["ACTION6"],
            legacy_action="ACTION6",
        )
        controller.observe_transition(
            action="ACTION6",
            action_data=neutral.action_data,
            grid_before=transformed,
            grid_after=transformed,
            available_actions=["ACTION6"],
        )
    terminal = controller.select_action(
        current_grid=transformed,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    controller.observe_transition(
        action="ACTION6",
        action_data=terminal.action_data,
        grid_before=transformed,
        grid_after=transformed,
        available_actions=["ACTION6"],
        levels_completed_after=1,
    )

    hypothesis = controller.summary()["terminal_objectives"]["hypotheses"][0]
    assert hypothesis["terminal_support"] == 0
    assert hypothesis["nonterminal_completions"] == 1
    assert hypothesis["status"] == "candidate"


def test_reset_closes_uncredited_completed_objective_as_nonterminal():
    controller, _ = _controller()
    _apply_transform(controller, _grid())

    controller.on_reset()

    hypothesis = controller.summary()["terminal_objectives"]["hypotheses"][0]
    assert hypothesis["terminal_support"] == 0
    assert hypothesis["terminal_contradictions"] == 1
    assert hypothesis["nonterminal_completions"] == 1


def test_game_over_quarantines_the_plan_without_refuting_the_goal():
    controller, _ = _controller(
        max_terminal_objective_probes_per_objective=2,
        max_terminal_objective_probes_total=2,
    )
    first, _ = _apply_transform(
        controller,
        _grid(source_position=(2, 2)),
        game_state_after="GAME_OVER",
    )
    assert first.source == "terminal_objective_probe"

    controller.on_reset()
    second, _ = _apply_transform(
        controller,
        _grid(source_position=(5, 5)),
        game_state_after="GAME_OVER",
    )
    assert second.source == "terminal_objective_probe"

    hypothesis = controller.summary()["terminal_objectives"]["hypotheses"][0]
    assert hypothesis["terminal_support"] == 0
    assert hypothesis["terminal_contradictions"] == 0
    assert hypothesis["unsafe_plan_failures"] == 2
    assert hypothesis["status"] == "candidate"
