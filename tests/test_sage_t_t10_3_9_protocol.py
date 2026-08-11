from __future__ import annotations

from theory.sage_t import t10_3_9_protocol as protocol


def test_sequence_repair_matrix_is_bounded_and_fresh() -> None:
    assert protocol.DISCOVERY_SEEDS == (3361, 3362, 3363, 3364)
    assert protocol.REPRODUCTION_SEEDS == (3371, 3372)
    assert protocol.CONFIRMATION_SEEDS == (3381, 3382)
    assert len(protocol.work_specs("discover-sequence")) == 12
    assert len(protocol.work_specs("reproduce-sequence")) == 6
    assert len(protocol.work_specs("confirm")) == 20
    assert protocol.TOTAL_RESETS == 38
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 3136


def test_confirmation_is_counterbalanced() -> None:
    rows = protocol.work_specs("confirm")
    for game in protocol.ALL_SOURCE_GAMES:
        game_rows = [row for row in rows if row.game_id == game]
        first_arms = [
            row.arm
            for seed in protocol.CONFIRMATION_SEEDS
            for row in game_rows
            if row.seed == seed and row.reset_index == 0
        ]
        assert set(first_arms) == set(protocol.CONFIRMATION_ARMS)


def test_parent_negative_snapshot_is_exact_and_not_training_data() -> None:
    parent = protocol.SUPERSEDED_T10_3_8
    assert parent["verdict"] == "MIXED_SEQUENCE_MISS"
    assert parent["intent_count"] == parent["event_count"] == 407
    assert parent["sequence_action_count"] == 272
    assert parent["sequence_level_count"] == 0
    assert parent["re86_controller_error_count"] == 2
    assert parent["used_for_training"] is False
    assert parent["sequence_registry_used_as_prior"] is False
    assert parent["physical_actions_replayed"] == 0


def test_no_core_recollection_or_parent_sequence_replay_phase() -> None:
    for phase in ("discover-core", "reproduce-core", "replay-sequence"):
        try:
            protocol.work_specs(phase)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{phase} must not be a T10.3.9 physical phase")
