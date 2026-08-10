from __future__ import annotations

from theory.sage_t import t10_3_7_protocol as protocol


def test_recovery_matrix_uses_fresh_seeds_and_same_bound() -> None:
    assert protocol.WITNESS_SEEDS == (3271, 3272)
    assert protocol.DISCOVERY_SEEDS == (3281, 3282, 3283, 3284)
    assert protocol.REPRODUCTION_SEEDS == (3291, 3292)
    assert protocol.SEQUENCE_SEEDS == (3301, 3302)
    assert protocol.CONFIRMATION_SEEDS == (3311, 3312)
    assert protocol.TOTAL_RESETS == 42
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 1856


def test_witness_has_two_resets_per_core_game() -> None:
    rows = protocol.work_specs("witness-core")
    assert len(rows) == 4
    assert all(work.action_budget == 16 for work in rows)
    assert {
        game: sum(work.game_id == game for work in rows)
        for game in protocol.CORE_GAMES
    } == {game: 2 for game in protocol.CORE_GAMES}


def test_parent_diagnosis_is_fail_closed() -> None:
    diagnosis = protocol.SUPERSEDED_T10_3_6
    assert diagnosis["lp85_level_delta"] == 1
    assert diagnosis["su15_level_delta"] == 0
    assert diagnosis["first_nine_su15_waypoints_exact"] is True
    assert diagnosis["tenth_waypoint_repeated_ninth"] is True
    assert diagnosis["used_for_training"] is False
