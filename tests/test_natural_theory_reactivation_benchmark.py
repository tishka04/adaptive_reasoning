"""SAGE.9y live R1 -> R2 -> R1 reactivation tests."""

from theory.natural_theory_reactivation_benchmark import (
    run_natural_theory_reactivation_benchmark,
)


def test_live_observation_return_reactivates_the_base_theory_program():
    payload = run_natural_theory_reactivation_benchmark()

    assert payload["protocol"]["protocol_gate_passed"] is True
    assert payload["protocol"]["hand_authored_assessments_used"] is False
    assert payload["observation_return_gate_passed"] is True
    assert payload["policy_sequence_gate_passed"] is True
    assert payload["natural_theory_reactivation_gate_passed"] is True

    active = payload["active"]
    ablated = payload["hierarchical_composition_ablated"]
    assert active["policy_sources"] == [
        "base",
        "exact_revision",
        "base",
    ]
    assert active["policy_theory_ids"][0] == active["policy_theory_ids"][2]
    assert active["policy_theory_ids"][0] != active["policy_theory_ids"][1]
    assert active["returned_theory_reactivated"] is True
    assert active["theory_switches"] == 2
    assert active["theory_reactivations"] == 1
    assert ablated["returned_theory_reactivated"] is False
    assert ablated["theory_switches"] == 0
    assert ablated["theory_reactivations"] == 0
