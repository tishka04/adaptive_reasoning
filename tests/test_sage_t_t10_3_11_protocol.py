from __future__ import annotations

from theory.sage_t import t10_3_11_protocol as protocol


def test_goal_matrix_is_paired_fresh_and_bounded() -> None:
    assert protocol.DISCOVERY_SEEDS == (3421, 3422, 3423, 3424)
    assert protocol.REPRODUCTION_SEEDS == (3431, 3432)
    assert protocol.CONFIRMATION_SEEDS == (3441, 3442)
    assert len(protocol.work_specs("discover-sequence")) == 24
    assert len(protocol.work_specs("reproduce-sequence")) == 6
    assert len(protocol.work_specs("confirm")) == 20
    assert protocol.TOTAL_RESETS == 50
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 4288

    rows = protocol.work_specs("discover-sequence")
    for game in protocol.SEQUENCE_GAMES:
        for seed in protocol.DISCOVERY_SEEDS:
            pair = [row for row in rows if row.game_id == game and row.seed == seed]
            assert {row.arm for row in pair} == set(protocol.DISCOVERY_ARMS)
            assert len({row.reset_index for row in pair}) == 1
            assert len({row.action_budget for row in pair}) == 1


def test_t10_3_10_negative_snapshot_is_exact_and_diagnostic_only() -> None:
    parent = protocol.SUPERSEDED_T10_3_10
    assert parent["status"] == "SUPERSEDED_COMPLETE_NEGATIVE"
    assert parent["intent_count"] == parent["event_count"] == 857
    assert parent["branch_count"] == 12
    assert parent["incomplete_work_count"] == 0
    assert parent["sequence_level_count"] == 0
    assert parent["controller_observe_error_count"] == 5
    assert parent["posterior_update_count"] == 852
    assert parent["maximum_sage_identical_action_run"] == 2
    assert parent["controller_cycle_p95_ms"] > 20_000
    assert parent["used_for_training"] is False
    assert parent["registry_used_as_prior"] is False
    assert parent["physical_actions_replayed"] == 0


def test_parent_replay_is_not_a_physical_phase() -> None:
    for phase in ("recover-t10-3-10", "replay-t10-3-10", "discover-core"):
        try:
            protocol.work_specs(phase)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{phase} must not be a T10.3.11 physical phase")

