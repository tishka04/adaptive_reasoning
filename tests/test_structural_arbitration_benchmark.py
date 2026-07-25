"""SAGE.9x procedural causal arbitration benchmark tests."""

from theory.structural_arbitration_benchmark import (
    run_structural_arbitration_benchmark,
    summarize_structural_arbitration_runs,
)


def test_procedural_benchmark_attributes_terminal_gain_to_arbitration():
    payload = run_structural_arbitration_benchmark(seeds=range(16))

    assert payload["protocol"]["protocol_gate_passed"] is True
    assert payload["ambiguous_episode_gate_passed"] is True
    assert payload["order_sensitivity_gate_passed"] is True
    assert payload["priority_permutation_gate_passed"] is True
    assert payload["active_terminal_successes"] == 16
    assert payload["sequential_terminal_successes"] == 0
    assert payload["active_single_theory_identifications"] == 16
    assert payload["sequential_single_theory_identifications"] == 0
    assert payload["causal_arbitration_gate_passed"] is True
    assert {
        run["true_hypothesis_priority_index"]
        for run in payload["runs"]
    } == {1, 2}


def test_summary_does_not_claim_causality_without_terminal_advantage():
    run = {
        "action_budget": 1,
        "plausible_hypotheses": 3,
        "active": {
            "action_selected": True,
            "terminal_success": False,
            "surviving_hypotheses": 2,
        },
        "sequential_ablation": {
            "action_selected": True,
            "terminal_success": False,
            "surviving_hypotheses": 2,
        },
        "active_and_sequential_actions_differ": True,
        "sequential_order_changes_terminal_outcome": True,
        "active_priority_permutation_invariant": True,
    }

    payload = summarize_structural_arbitration_runs([run])

    assert payload["protocol"]["protocol_gate_passed"] is True
    assert payload["causal_arbitration_gate_passed"] is False
