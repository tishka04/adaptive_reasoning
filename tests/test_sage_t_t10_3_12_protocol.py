from __future__ import annotations

from theory.sage_t import t10_3_12_protocol as protocol
from theory.sage_t.relational_program_v10_3_12 import ARMS


def test_active_matrix_is_latin_square_fresh_and_bounded() -> None:
    rows = protocol.work_specs("active-core")
    assert len(rows) == 32
    assert protocol.TOTAL_RESETS == 32
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 512
    for game in protocol.CORE_GAMES:
        for label in protocol.ACTIVE_LABELS:
            block = [row for row in rows if row.game_id == game and row.seed == label]
            assert len(block) == 4
            assert {row.arm for row in block} == set(ARMS)
            assert {row.action_budget for row in block} == {16}


def test_sequence_and_parent_replay_are_not_physical_phases() -> None:
    for phase in (
        "discover-sequence",
        "confirm",
        "recover-t10-3-11",
        "replay-t10-3-11",
    ):
        try:
            protocol.work_specs(phase)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{phase} must remain closed in T10.3.12")


def test_parent_snapshot_is_explicitly_incomplete_and_quarantined() -> None:
    parent = protocol.QUARANTINED_PARENT
    assert parent["status"] == "SUPERSEDED_INCOMPLETE_NEGATIVE"
    assert parent["authorized_actions"] == 1758
    assert parent["sealed_events"] == 1756
    assert parent["inflight_intents"] == 2
    assert parent["inflight_valid"] is False
    assert parent["runtime_permission_errors"] == 3
    assert parent["used_for_training"] is False
    assert parent["registry_loaded"] is False
    assert parent["physical_actions_replayed"] == 0


def test_artifact_contract_separates_collection_and_adjudication() -> None:
    active = protocol.ARTIFACT_CONTRACT["active-core"]
    adjudication = protocol.ARTIFACT_CONTRACT["adjudicate"]
    assert active["gate_field"] == "collection_complete"
    assert adjudication["gate_field"] == "passed"
    assert protocol.ACTIVE_LABELS == (3521, 3522, 3523, 3524)

