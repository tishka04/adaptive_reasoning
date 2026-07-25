"""SAGE.9z ARC multi-game replication gate tests."""

from theory.arc_multigame_replication_benchmark import (
    summarize_arc_multigame_replication_rows,
)


def _row(game, seed, budget, *, confirm=True, advantage=True):
    return {
        "game_id": game,
        "seed": seed,
        "action_budget": budget,
        "natural_break_candidate": True,
        "terminal_revision_observed": confirm,
        "causal_progress_advantage": advantage,
        "causal_natural_revision_gate_passed": bool(
            confirm and advantage
        ),
        "active_max_level_reached": 5 if advantage else 4,
        "ablated_max_level_reached": 4,
        "active_controller_errors": 0,
        "ablated_controller_errors": 0,
    }


def _protocols():
    return (
        {
            "action_budget": 80,
            "protocol_gate_passed": True,
            "relation_permutation_used": False,
        },
        {
            "action_budget": 160,
            "protocol_gate_passed": True,
            "relation_permutation_used": False,
        },
    )


def test_replication_requires_causal_revision_across_seeds_and_budgets():
    rows = [
        _row(game, seed, budget)
        for game in ("game-a", "game-b")
        for seed in (0, 1)
        for budget in (80, 160)
    ]

    payload = summarize_arc_multigame_replication_rows(
        rows,
        expected_games=("game-a", "game-b"),
        expected_seeds=(0, 1),
        expected_budgets=(80, 160),
        resets=4,
        condition_protocols=_protocols(),
    )

    assert payload["protocol"]["protocol_gate_passed"] is True
    assert payload["replicated_games"] == 2
    assert payload["any_replicated_natural_revision_gate_passed"] is True
    assert all(
        game["replicated_natural_revision_gate_passed"]
        for game in payload["games"]
    )


def test_replication_does_not_overclaim_without_terminal_revision():
    rows = [
        _row(
            game,
            seed,
            budget,
            confirm=False,
            advantage=True,
        )
        for game in ("game-a", "game-b")
        for seed in (0, 1)
        for budget in (80, 160)
    ]

    payload = summarize_arc_multigame_replication_rows(
        rows,
        expected_games=("game-a", "game-b"),
        expected_seeds=(0, 1),
        expected_budgets=(80, 160),
        resets=4,
        condition_protocols=_protocols(),
    )

    assert payload["protocol"]["protocol_gate_passed"] is True
    assert payload["natural_break_conditions"] == 8
    assert payload["terminal_revision_conditions"] == 0
    assert payload["causal_revision_conditions"] == 0
    assert payload["any_replicated_natural_revision_gate_passed"] is False
