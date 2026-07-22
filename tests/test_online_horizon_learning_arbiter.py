"""Tests for SAGE.9e online long-horizon action allocation."""

from theory.online_horizon_learning_arbiter import (
    HorizonLearningSignals,
    OnlineHorizonLearningArbiter,
)


def test_arbiter_releases_operator_budget_without_online_learning_demand():
    arbiter = OnlineHorizonLearningArbiter(
        base_operator_action_budget=12,
    )

    allocation = arbiter.allocate(HorizonLearningSignals())

    assert allocation.reserve_learning is False
    assert allocation.operator_action_budget is None
    assert allocation.reasons == ("no_online_learning_demand",)
    assert arbiter.summary()["releases"] == 1


def test_arbiter_reserves_for_observed_causal_uncertainty():
    arbiter = OnlineHorizonLearningArbiter(
        base_operator_action_budget=12,
    )

    allocation = arbiter.allocate(HorizonLearningSignals(
        productive_causal_edges=1,
        unresolved_opened_options=1,
        supported_mediated_hyperedges=1,
        causal_target_distance=1.0,
    ))

    assert allocation.reserve_learning is True
    assert allocation.operator_action_budget == 12
    assert allocation.priority == 9
    assert allocation.causal_uncertainty_present is True
    assert "near_causal_target" in allocation.reasons
    assert "productive_causal_edge" in allocation.reasons
    summary = arbiter.summary()
    assert summary["causal_uncertainty_reservations"] == 1
    assert summary["budget_allocations"] == {"12": 1}


def test_arbiter_tightens_budget_for_near_terminal_contrast_only():
    arbiter = OnlineHorizonLearningArbiter(
        base_operator_action_budget=12,
        maximum_terminal_test_distance=2.0,
    )

    near = arbiter.allocate(HorizonLearningSignals(
        terminal_test_status="needs_contrast",
        terminal_test_distance=1.0,
    ))
    far = arbiter.allocate(HorizonLearningSignals(
        terminal_test_status="terminal_supported",
        terminal_test_distance=3.0,
    ))

    assert near.reserve_learning is True
    assert near.operator_action_budget == 6
    assert near.terminal_test_near is True
    assert far.reserve_learning is False
    assert arbiter.summary()["terminal_test_reservations"] == 1
