"""Scientific protocol tests for SAGE.10a."""

from __future__ import annotations

from theory.delayed_frontier_credit_benchmark import (
    _run_procedural_delayed_credit_pair,
    summarize_delayed_frontier_credit_protocol,
)


def _payload(*, enabled: bool, metrics: dict) -> dict:
    defaults = {
        "levels_completed_delta": 0,
        "max_level_reached": 0,
        "wins": 0,
    }
    defaults.update(metrics)
    return {
        "held_out_games": ["synthetic-delayed-credit"],
        "seeds": [0],
        "action_budget_per_reset": 40,
        "resets_per_game_seed_arm": 2,
        "paired_protocol": {
            "protocol_gate_passed": True,
            (
                "delayed_frontier_terminal_credit_enabled_in_unified"
            ): enabled,
        },
        "pairs": [{
            "game_id": "synthetic-delayed-credit",
            "seed": 0,
            "unified": {
                **defaults,
                "reset_visual_digests": ["same-reset"],
                "configured_action_budget": 80,
                "resets_executed": 2,
                "actions_executed": 80,
                "failure_causes": {},
            },
        }],
    }


def test_procedural_delayed_credit_is_causally_necessary():
    result = _run_procedural_delayed_credit_pair()

    assert result["causal_gate_passed"] is True
    assert result["active"]["credit_delay_actions"] == 2
    assert result["active_transferred_policy"] is True
    assert result["ablated_transferred_policy"] is False
    assert result["active_terminal_success"] is True
    assert result["ablated_terminal_success"] is False


def test_sage10a_protocol_requires_isolated_ablation():
    active = _payload(
        enabled=True,
        metrics={
            "levels_completed_delta": 2,
            "max_level_reached": 2,
            "frontier_delayed_eligibilities_registered": 2,
            "frontier_delayed_terminal_credits": 1,
            "terminal_multiform_delayed_frontier_pattern_credits": 3,
            "terminal_multiform_selections": 1,
        },
    )
    ablated = _payload(
        enabled=False,
        metrics={
            "levels_completed_delta": 1,
            "max_level_reached": 1,
        },
    )

    result = summarize_delayed_frontier_credit_protocol(
        active,
        ablated,
    )

    assert result["protocol"]["protocol_gate_passed"] is True
    assert result["delayed_credit_games"] == 1
    assert result["relational_propagation_games"] == 1
    assert result["causal_arc_progress_games"] == 1


def test_sage10a_does_not_overclaim_credit_without_progress():
    active = _payload(
        enabled=True,
        metrics={
            "frontier_delayed_eligibilities_registered": 3,
            "frontier_delayed_terminal_credits": 1,
            "terminal_multiform_delayed_frontier_pattern_credits": 2,
        },
    )
    ablated = _payload(enabled=False, metrics={})

    result = summarize_delayed_frontier_credit_protocol(
        active,
        ablated,
    )

    assert result["any_delayed_credit_gate_passed"] is True
    assert result["any_relational_propagation_gate_passed"] is True
    assert result["any_causal_arc_progress_gate_passed"] is False
