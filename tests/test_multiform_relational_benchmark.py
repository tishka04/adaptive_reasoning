"""Scientific protocol tests for SAGE.9w."""

from __future__ import annotations

from theory.multiform_relational_benchmark import (
    summarize_multiform_relational_protocol,
)
from theory.unified_cognition_ab_benchmark import (
    MULTIFORM_RELATION_FAMILIES,
)


def _payload(*, enabled: bool, metrics: dict) -> dict:
    defaults = {
        "levels_completed_delta": 0,
        "max_level_reached": 0,
        "wins": 0,
        "controller_errors": [],
        "terminal_multiform_observations": 0,
        "terminal_multiform_terminal_examples": 0,
        "terminal_multiform_patterns_observed": 0,
        "terminal_multiform_pattern_hypotheses": 0,
        "terminal_multiform_confirmed_patterns": 0,
        "terminal_multiform_confirmed_families": 0,
        "terminal_multiform_actuator_models": 0,
        "terminal_multiform_terminal_pattern_credits": 0,
        "terminal_multiform_selections": 0,
        "terminal_multiform_transferred_selections": 0,
        "terminal_multiform_unsafe_model_blocks": 0,
    }
    for family in MULTIFORM_RELATION_FAMILIES:
        defaults[f"terminal_multiform_{family}_observations"] = 0
        defaults[f"terminal_multiform_{family}_selections"] = 0
    defaults.update(metrics)
    return {
        "held_out_games": ["synthetic-multiform"],
        "seeds": [0],
        "action_budget_per_reset": 40,
        "resets_per_game_seed_arm": 2,
        "paired_protocol": {
            "protocol_gate_passed": True,
            (
                "terminal_multiform_relational_induction_enabled_in_unified"
            ): enabled,
        },
        "pairs": [{
            "game_id": "synthetic-multiform",
            "unified": defaults,
        }],
    }


def test_sage9w_causal_gate_requires_acquisition_policy_and_progress():
    active = _payload(
        enabled=True,
        metrics={
            "levels_completed_delta": 2,
            "max_level_reached": 2,
            "terminal_multiform_terminal_examples": 2,
            "terminal_multiform_confirmed_patterns": 4,
            "terminal_multiform_confirmed_families": 3,
            "terminal_multiform_selections": 5,
            "terminal_multiform_count_selections": 3,
        },
    )
    ablated = _payload(
        enabled=False,
        metrics={
            "levels_completed_delta": 1,
            "max_level_reached": 1,
        },
    )

    result = summarize_multiform_relational_protocol(active, ablated)

    assert result["protocol"]["protocol_gate_passed"] is True
    assert result["multiform_acquisition_games"] == 1
    assert result["multiform_policy_games"] == 1
    assert result["causal_multiform_progress_games"] == 1
    assert result["games"][0][
        "causal_multiform_progress_gate_passed"
    ] is True


def test_sage9w_does_not_overclaim_passive_terminal_patterns():
    active = _payload(
        enabled=True,
        metrics={
            "terminal_multiform_terminal_examples": 3,
            "terminal_multiform_confirmed_patterns": 5,
            "terminal_multiform_confirmed_families": 4,
            "terminal_multiform_selections": 0,
        },
    )
    ablated = _payload(enabled=False, metrics={})

    result = summarize_multiform_relational_protocol(active, ablated)

    report = result["games"][0]
    assert report["multiform_acquisition_gate_passed"] is True
    assert report["multiform_policy_gate_passed"] is False
    assert report["causal_progress_advantage"] is False
    assert result["any_causal_multiform_progress_gate_passed"] is False
