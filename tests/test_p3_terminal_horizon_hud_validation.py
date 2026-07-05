import numpy as np

from theory.p3.terminal_horizon_estimator import TerminalHorizonObserver
from theory.p3.terminal_horizon_hud_validation import validate_hud_history


def _grid_with_top_bar(filled, total=8):
    grid = np.zeros((6, total), dtype=np.int32)
    grid[0, :filled] = 9
    grid[3, 3] = 4
    return grid


def test_validate_hud_history_accepts_stable_monotone_hud_sequence():
    grids = [_grid_with_top_bar(value) for value in [8, 7, 6, 5, 4]]
    payload = validate_hud_history(
        grid_history=grids,
        action_history=["ACTION6"] * 4,
        terminal_step=8,
        observer=TerminalHorizonObserver(terminal_budget_estimate=8),
    )

    summary = payload["summary"]
    assert summary["hud_bar_source_active_steps"] >= 3
    assert summary["stable_hud_bar_sequence_detected"] is True
    assert summary["ready_for_hud_p3_2b"] is True
    assert summary["support"] == 0
    assert payload["estimates"][-1]["source"] == "hud_bar"
    assert payload["estimates"][-1]["predicted_next_remaining"] == 3


def test_validate_hud_history_falls_back_without_stable_hud():
    grids = [np.zeros((6, 8), dtype=np.int32) for _ in range(5)]
    payload = validate_hud_history(
        grid_history=grids,
        action_history=["ACTION6"] * 4,
        terminal_step=8,
        observer=TerminalHorizonObserver(terminal_budget_estimate=8),
    )

    summary = payload["summary"]
    assert summary["hud_bar_source_active_steps"] == 0
    assert summary["stable_hud_bar_sequence_detected"] is False
    assert summary["source_remained_empirical_fallback"] is True
    assert summary["ready_for_hud_p3_2b"] is False
    assert payload["estimates"][-1]["source"] == "empirical_fallback"

