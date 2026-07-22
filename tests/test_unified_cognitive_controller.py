"""End-to-end tests for the consolidated live cognitive execution path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from theory.epistemic_metrics import HypothesisStatus
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.unified_cognitive_controller import (
    CognitiveDecision,
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


def _player_grid(column: int, *, width: int = 12) -> np.ndarray:
    grid = np.zeros((7, width), dtype=np.int32)
    grid[3, column] = 2
    return grid


def test_unified_controller_starts_with_a_discriminating_experiment():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION1", "ACTION2"],
    )

    decision = controller.select_action(
        current_grid=_player_grid(2),
        available_actions=["ACTION1", "ACTION2"],
        legacy_action="ACTION2",
    )

    assert decision.source == "discriminating_experiment"
    assert decision.action_name in {"ACTION1", "ACTION2"}
    assert len(decision.competing_hypotheses) == 2
    assert controller.summary()["decision_sources"] == {
        "discriminating_experiment": 1,
    }


def test_observed_transitions_update_theory_profiler_and_operator_library():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION1"],
        config=UnifiedCognitiveConfig(max_bootstrap_experiments=6),
    )

    for index in range(6):
        before = _player_grid(index + 1)
        after = _player_grid(index + 2)
        decision = controller.select_action(
            current_grid=before,
            available_actions=["ACTION1"],
            legacy_action="ACTION1",
        )
        assert decision.action_name == "ACTION1"
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION1"],
        )

    dominant = controller.theory.dominant("ACTION1")
    assert dominant is not None
    assert dominant.kind == "move"
    assert controller.belief_loop.profiler.get_stats("ACTION1").total_tries == 6
    assert "move_right_ACTION1" in controller.operator_inducer.operators

    planned = controller.select_action(
        current_grid=_player_grid(7),
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )
    assert planned.source == "operator_plan"
    assert planned.operator_id == "move_right_ACTION1"


def test_horizon_stable_epoch_yields_after_unproductive_operator_budget():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION1"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=6,
            horizon_learning_warmup_actions_per_branch=0,
            max_operator_plan_actions_without_objective_progress=1,
            enable_online_horizon_learning_arbiter=False,
        ),
    )

    for index in range(6):
        before = _player_grid(index + 1)
        after = _player_grid(index + 2)
        decision = controller.select_action(
            current_grid=before,
            available_actions=["ACTION1"],
            legacy_action="ACTION1",
        )
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION1"],
        )

    planned = controller.select_action(
        current_grid=_player_grid(7),
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )
    yielded = controller.select_action(
        current_grid=_player_grid(7),
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )

    assert planned.source == "operator_plan"
    assert yielded.source != "operator_plan"
    summary = controller.summary()
    assert summary["operator_plan_streak_peak"] == 1
    assert summary["operator_plan_budget_blocks"] == 1
    assert summary["operator_plan_actions_since_objective_progress"] == 1

    controller.on_reset()

    assert (
        controller.summary()[
            "operator_plan_actions_since_objective_progress"
        ]
        == 0
    )

    replanned = controller.select_action(
        current_grid=_player_grid(7),
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )
    assert replanned.source == "operator_plan"
    supported = controller.terminal_objectives.register_generated(
        GeneratedGoalHypothesis(
            objective_id="reach::3",
            family="reach",
            source_color=None,
            target_color=3,
            predicate="player_adjacent_to_target",
            supporting_rule_keys=(),
            supporting_actions=("ACTION1",),
            generation_reason="synthetic_terminal_supported_progress",
        )
    )
    supported.terminal_contexts.update({"terminal-a", "terminal-b"})
    before = _player_grid(7)
    before[3, 10] = 3
    after = _player_grid(8)
    after[3, 10] = 3

    controller.observe_transition(
        action=replanned.action_name,
        action_data=replanned.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1"],
    )

    summary = controller.summary()
    assert summary["operator_plan_progress_resets"] == 1
    assert summary["operator_plan_actions_since_objective_progress"] == 0


def test_online_horizon_arbiter_releases_irrelevant_operator_budget():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION1"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=6,
            horizon_learning_warmup_actions_per_branch=0,
            max_operator_plan_actions_without_objective_progress=1,
        ),
    )

    for index in range(6):
        before = _player_grid(index + 1)
        after = _player_grid(index + 2)
        decision = controller.select_action(
            current_grid=before,
            available_actions=["ACTION1"],
            legacy_action="ACTION1",
        )
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION1"],
        )

    first = controller.select_action(
        current_grid=_player_grid(7),
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )
    second = controller.select_action(
        current_grid=_player_grid(7),
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )

    assert first.source == "operator_plan"
    assert second.source == "operator_plan"
    summary = controller.summary()
    assert summary["operator_plan_budget_blocks"] == 0
    arbiter = summary["online_horizon_learning_arbiter"]
    assert arbiter["evaluations"] == 2
    assert arbiter["reservations"] == 0
    assert arbiter["releases"] == 2


def test_nonterminal_objective_completion_opens_terminal_frontier_suffix():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION1", "ACTION2"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=0,
            enable_active_goal_hypotheses=False,
            enable_operator_planning=False,
            enable_theory_planning=False,
            enable_promoted_options=False,
            enable_temporal_goal_composition=False,
            enable_causal_hierarchical_options=False,
            max_terminal_frontier_suffix_actions=2,
        ),
    )
    controller.terminal_objectives.register_generated(
        GeneratedGoalHypothesis(
            objective_id="reach::3",
            family="reach",
            source_color=None,
            target_color=3,
            predicate="player_adjacent_to_target",
            supporting_rule_keys=(),
            supporting_actions=("ACTION1",),
            generation_reason="synthetic_nonterminal_frontier",
        )
    )
    before = _player_grid(2)
    before[3, 4] = 3
    postcondition = _player_grid(3)
    postcondition[3, 4] = 3
    initial = CognitiveDecision(
        action_name="ACTION1",
        source="terminal_objective_probe",
        objective_id="reach::3",
    )
    controller._pending_decision = initial  # noqa: SLF001 - integration setup
    controller.observe_transition(
        action=initial.action_name,
        action_data=initial.action_data,
        grid_before=before,
        grid_after=postcondition,
        available_actions=["ACTION1", "ACTION2"],
        levels_completed_before=0,
        levels_completed_after=0,
    )

    suffix = controller.select_action(
        current_grid=postcondition,
        available_actions=["ACTION1", "ACTION2"],
        legacy_action="ACTION2",
    )

    assert suffix.source == "legacy_fallback"
    assert suffix.terminal_frontier_id
    assert suffix.terminal_frontier_objective_ids == ("reach::3",)
    assert suffix.terminal_frontier_suffix_step == 0
    next_level = postcondition.copy()
    next_level[0, 0] = 7
    controller.observe_transition(
        action=suffix.action_name,
        action_data=suffix.action_data,
        grid_before=postcondition,
        grid_after=next_level,
        available_actions=["ACTION1", "ACTION2"],
        levels_completed_before=0,
        levels_completed_after=1,
    )

    frontier = controller.summary()["terminal_negative_frontiers"]
    assert frontier["frontiers_captured"] == 1
    assert frontier["suffix_actions"] == 1
    assert frontier["terminal_credits"] == 1
    assert frontier["level_change_credits"] == 1
    assert frontier["successful_continuations"] == 1


def test_click_experiment_uses_objects_and_revises_relational_predictions():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION6"],
        config=UnifiedCognitiveConfig(enable_active_goal_hypotheses=False),
    )
    before = np.zeros((12, 12), dtype=np.int32)
    before[2, 2] = 2
    before[8, 8] = 3
    after = before.copy()
    after[2, 2] = 3

    selected_keys = ()
    for _ in range(2):
        decision = controller.select_action(
            current_grid=before,
            available_actions=["ACTION6"],
            legacy_action="ACTION6",
        )
        assert decision.source == "relational_experiment"
        assert decision.action_data == {"x": 2, "y": 2}
        assert len(decision.competing_hypotheses) == 2
        selected_keys = decision.competing_hypotheses
        controller.observe_transition(
            action="ACTION6",
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION6"],
        )

    statuses = {
        controller._predictions[key].status  # noqa: SLF001 - audit assertion
        for key in selected_keys
    }
    assert statuses == {HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED}
    summary = controller.summary()
    assert summary["relational_experiments_selected"] == 2
    assert summary["generic_hypothesis_revisions"] == 4


def test_observed_game_over_becomes_a_hard_safety_veto():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION1", "ACTION2"],
        config=UnifiedCognitiveConfig(max_bootstrap_experiments=0),
    )
    grid = _player_grid(2)
    first = controller.select_action(
        current_grid=grid,
        available_actions=["ACTION1", "ACTION2"],
        legacy_action="ACTION1",
    )
    assert first.action_name == "ACTION1"
    controller.observe_transition(
        action="ACTION1",
        grid_before=grid,
        grid_after=grid.copy(),
        available_actions=["ACTION1", "ACTION2"],
        game_state_after="GAME_OVER",
    )

    guarded = controller.select_action(
        current_grid=grid,
        available_actions=["ACTION1", "ACTION2"],
        legacy_action="ACTION1",
    )
    assert guarded.action_name == "ACTION2"
    assert guarded.source == "safety_veto"
    assert controller.summary()["danger_records"] == 1


def test_resized_frames_are_aligned_before_structured_revision():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION1"],
        config=UnifiedCognitiveConfig(max_bootstrap_experiments=0),
    )
    before = np.zeros((3, 3), dtype=np.int32)
    before[1, 1] = 2
    after = np.zeros((5, 5), dtype=np.int32)
    after[2, 2] = 2

    controller.select_action(
        current_grid=before,
        available_actions=["ACTION1"],
        legacy_action="ACTION1",
    )
    update = controller.observe_transition(
        action="ACTION1",
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1"],
    )

    assert update.record.obs_before.raw_grid.shape == (5, 5)
    assert update.record.obs_after.raw_grid.shape == (5, 5)
    assert controller.summary()["transitions_observed"] == 1


def test_registered_arc_agent_enables_the_unified_controller():
    project_root = Path(__file__).resolve().parents[1]
    registry = (
        project_root / "ARC-AGI-3-Agents" / "agents" / "__init__.py"
    ).read_text(encoding="utf-8")
    active_agent = (
        project_root
        / "ARC-AGI-3-Agents"
        / "agents"
        / "templates"
        / "adaptive_reasoning_agent.py"
    ).read_text(encoding="utf-8")

    assert "AVAILABLE_AGENTS[\"adaptivereasoning\"] = AdaptiveReasoning" in registry
    assert "ENABLE_UNIFIED_COGNITION: bool = True" in active_agent
    assert "self.cognitive_controller.select_action(" in active_agent
    assert "self.cognitive_controller.observe_transition(" in active_agent
    assert "_unified_fast_choice(" in active_agent
