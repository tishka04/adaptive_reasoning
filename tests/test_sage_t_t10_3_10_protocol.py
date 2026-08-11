from __future__ import annotations

from theory.sage_t import t10_3_10_protocol as protocol


def test_directional_matrix_is_fresh_bounded_and_counterbalanced() -> None:
    assert protocol.DISCOVERY_SEEDS == (3391, 3392, 3393, 3394)
    assert protocol.REPRODUCTION_SEEDS == (3401, 3402)
    assert protocol.CONFIRMATION_SEEDS == (3411, 3412)
    assert len(protocol.work_specs("discover-sequence")) == 12
    assert len(protocol.work_specs("reproduce-sequence")) == 6
    assert len(protocol.work_specs("confirm")) == 20
    assert protocol.TOTAL_RESETS == 38
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 3136

    rows = protocol.work_specs("confirm")
    for game in protocol.ALL_SOURCE_GAMES:
        game_rows = [row for row in rows if row.game_id == game]
        first_arms = {
            row.arm
            for seed in protocol.CONFIRMATION_SEEDS
            for row in game_rows
            if row.seed == seed and row.reset_index == 0
        }
        assert first_arms == set(protocol.CONFIRMATION_ARMS)


def test_parent_partial_snapshot_is_exact_and_never_replayed() -> None:
    parent = protocol.SUPERSEDED_T10_3_9
    assert parent["status"] == "SUPERSEDED_PARTIAL_EFFECT_CYCLE"
    assert parent["intent_count"] == parent["event_count"] == 153
    assert parent["branch_count"] == 1
    assert parent["incomplete_work_count"] == 1
    assert parent["completed_sequence_action_count"] == 96
    assert parent["interrupted_sequence_action_count"] == 57
    assert parent["sequence_level_count"] == 0
    assert parent["maximum_identical_action_run"] == 22
    assert parent["completed_reset_controller_cycle_p95_ms"] > 20_000
    assert parent["used_for_training"] is False
    assert parent["registry_used_as_prior"] is False
    assert parent["physical_actions_replayed"] == 0


def test_parent_replay_and_core_recollection_are_not_physical_phases() -> None:
    for phase in (
        "recover-t10-3-9",
        "replay-t10-3-9",
        "discover-core",
        "reproduce-core",
    ):
        try:
            protocol.work_specs(phase)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{phase} must not be a T10.3.10 physical phase")

