from dataclasses import dataclass

import numpy as np

from theory.p3.terminal_horizon_estimator import (
    TerminalHorizonObserver,
    detect_budget_bar_candidates,
    estimate_moves_remaining_fallback,
    estimate_terminal_horizon,
    moves_used_from_policy_state,
)


@dataclass
class _Step:
    env_actions: int = 1


def test_fallback_counts_all_env_actions():
    estimate = estimate_moves_remaining_fallback(
        {"env_actions_executed": 63},
        terminal_budget_estimate=64,
    )

    assert estimate.observed is False
    assert estimate.estimated_moves_remaining == 1
    assert estimate.estimated_total_budget == 64
    assert estimate.moves_used == 63
    assert estimate.source == "empirical_fallback"
    assert estimate.terminal_fraction_remaining == 1 / 64
    assert estimate.evidence["moves_used_variable"] == "env_actions_executed"


def test_moves_used_from_policy_state_accepts_step_sequences():
    assert moves_used_from_policy_state([_Step(), _Step(2), _Step()]) == 4


def test_terminal_horizon_prefers_metadata_then_hud_then_fallback():
    metadata_first = estimate_terminal_horizon(
        observation={
            "hud_bar": {
                "action_budget_bar_detected": True,
                "estimated_moves_remaining": 7,
                "filled_cells": 7,
                "empty_cells": 57,
                "confidence": 0.82,
            }
        },
        policy_state={"env_actions_executed": 10},
        environment_metadata={"moves_remaining": 20},
        terminal_budget_estimate=64,
    )
    hud = estimate_terminal_horizon(
        observation={
            "hud_bar": {
                "action_budget_bar_detected": True,
                "estimated_moves_remaining": 7,
                "filled_cells": 7,
                "empty_cells": 57,
                "confidence": 0.82,
            }
        },
        policy_state={"env_actions_executed": 10},
        terminal_budget_estimate=64,
    )
    metadata = estimate_terminal_horizon(
        policy_state={"env_actions_executed": 10},
        environment_metadata={"moves_remaining": 20, "total_budget": 64},
        terminal_budget_estimate=64,
    )
    fallback = estimate_terminal_horizon(
        policy_state={"env_actions_executed": 10},
        terminal_budget_estimate=64,
    )

    assert metadata_first.source == "environment_metadata"
    assert metadata_first.estimated_moves_remaining == 20
    assert hud.source == "hud_bar"
    assert hud.estimated_moves_remaining == 7
    assert hud.terminal_fraction_remaining == 7 / 64
    assert metadata.source == "environment_metadata"
    assert metadata.estimated_moves_remaining == 20
    assert fallback.source == "empirical_fallback"
    assert fallback.estimated_moves_remaining == 54


def _grid_with_top_bar(filled, total=8):
    grid = np.zeros((6, total), dtype=np.int32)
    grid[0, :filled] = 9
    grid[3, 3] = 4
    return grid


def test_detect_budget_bar_candidates_finds_edge_segment():
    candidates = detect_budget_bar_candidates(_grid_with_top_bar(6))

    assert candidates
    assert any(candidate.orientation == "horizontal_top" for candidate in candidates)


def test_terminal_horizon_observer_uses_monotone_hud_bar_history():
    history = [
        _grid_with_top_bar(8),
        _grid_with_top_bar(7),
        _grid_with_top_bar(6),
    ]
    estimate = TerminalHorizonObserver(terminal_budget_estimate=64).estimate(
        observation=_grid_with_top_bar(5),
        history=history,
        policy_state={"env_actions_executed": 3},
    )

    assert estimate.source == "hud_bar"
    assert estimate.observed is True
    assert estimate.estimated_moves_remaining == 5
    assert estimate.estimated_total_budget == 8
    assert estimate.confidence >= 0.7
    assert estimate.evidence["monotonic_delta_observed"] is True
    assert estimate.evidence["ticks_lost_per_action"] == 1.0


def test_terminal_horizon_observer_inverts_elapsed_bar_history():
    history = [
        _grid_with_top_bar(1),
        _grid_with_top_bar(2),
        _grid_with_top_bar(3),
    ]
    estimate = TerminalHorizonObserver(terminal_budget_estimate=8).estimate(
        observation=_grid_with_top_bar(4),
        history=history,
        policy_state={"env_actions_executed": 3},
    )

    assert estimate.source == "hud_bar"
    assert estimate.estimated_moves_remaining == 4
    assert estimate.evidence["bar_semantics"] == "elapsed_ticks_increasing"
    assert estimate.evidence["estimated_remaining_rule"] == "length - filled_length"


def test_terminal_horizon_observer_prefers_perfect_metadata_over_hud():
    estimate = TerminalHorizonObserver(terminal_budget_estimate=64).estimate(
        observation=_grid_with_top_bar(5),
        history=[_grid_with_top_bar(8), _grid_with_top_bar(7)],
        env_info={"moves_remaining": 22, "total_budget": 64},
        policy_state={"env_actions_executed": 3},
    )

    assert estimate.source == "environment_metadata"
    assert estimate.estimated_moves_remaining == 22
    assert estimate.confidence == 0.9
