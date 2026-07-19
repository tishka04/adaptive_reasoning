"""End-to-end tests for the consolidated live cognitive execution path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from theory.epistemic_metrics import HypothesisStatus
from theory.unified_cognitive_controller import (
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
