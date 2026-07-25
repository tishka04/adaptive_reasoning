"""Scientific protocol tests for SAGE.9v."""

from __future__ import annotations

from theory.frontier_exploration_benchmark import (
    summarize_frontier_exploration_protocol,
)


def _payload(*, enabled: bool, metrics: dict) -> dict:
    return {
        "held_out_games": ["synthetic-frontier"],
        "seeds": [0],
        "action_budget_per_reset": 40,
        "resets_per_game_seed_arm": 2,
        "paired_protocol": {
            "protocol_gate_passed": True,
            "frontier_oriented_exploration_enabled_in_unified": enabled,
        },
        "pairs": [{
            "game_id": "synthetic-frontier",
            "unified": {
                "levels_completed_delta": 0,
                "max_level_reached": 0,
                "wins": 0,
                "controller_errors": [],
                "frontier_stagnation_detections": 0,
                "frontier_entries": 0,
                "frontier_experiments": 0,
                "frontier_sequence_actions": 0,
                "frontier_multi_step_sequences": 0,
                "frontier_untested_state_actions": 0,
                "frontier_untested_actuator_actions": 0,
                "frontier_untested_object_actions": 0,
                "frontier_productive_experiments": 0,
                "frontier_novel_effects": 0,
                "frontier_novel_states": 0,
                "frontier_terminal_credits": 0,
                "frontier_information_gain": 0.0,
                **metrics,
            },
        }],
    }


def test_sage9v_gate_requires_new_state_and_causal_progress():
    active = _payload(
        enabled=True,
        metrics={
            "levels_completed_delta": 2,
            "max_level_reached": 2,
            "frontier_stagnation_detections": 3,
            "frontier_entries": 1,
            "frontier_experiments": 4,
            "frontier_untested_actuator_actions": 2,
            "frontier_productive_experiments": 2,
            "frontier_novel_effects": 2,
            "frontier_novel_states": 1,
            "frontier_information_gain": 3.0,
        },
    )
    ablated = _payload(
        enabled=False,
        metrics={
            "levels_completed_delta": 1,
            "max_level_reached": 1,
        },
    )

    result = summarize_frontier_exploration_protocol(active, ablated)

    assert result["protocol"]["protocol_gate_passed"] is True
    assert result["frontier_access_games"] == 1
    assert result["causal_frontier_progress_games"] == 1
    report = result["games"][0]
    assert report["information_gate_passed"] is True
    assert report["frontier_access_gate_passed"] is True
    assert report["causal_frontier_progress_gate_passed"] is True


def test_sage9v_does_not_call_effect_only_change_frontier_access():
    active = _payload(
        enabled=True,
        metrics={
            "frontier_stagnation_detections": 2,
            "frontier_experiments": 2,
            "frontier_untested_actuator_actions": 2,
            "frontier_productive_experiments": 1,
            "frontier_novel_effects": 1,
            "frontier_novel_states": 0,
            "frontier_information_gain": 1.0,
        },
    )
    ablated = _payload(enabled=False, metrics={})

    result = summarize_frontier_exploration_protocol(active, ablated)

    report = result["games"][0]
    assert report["information_gate_passed"] is True
    assert report["frontier_access_gate_passed"] is False
    assert report["causal_progress_advantage"] is False
    assert result["any_causal_frontier_progress_gate_passed"] is False
