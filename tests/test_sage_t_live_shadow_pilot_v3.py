from __future__ import annotations

from theory.sage_t.live_shadow_pilot_v3 import (
    assessment_for_live_action,
    load_frozen_manifest,
)


def test_manifest_registers_only_action_metadata_normalization() -> None:
    manifest = load_frozen_manifest()

    assert manifest["measurement_changes"] == [
        "drop non-semantic game_id before action-key comparison"
    ]
    assert manifest["authority"]["mode"] == "shadow"
    assert manifest["challenger"]["can_satisfy_t8_0_gate"] is False


def test_live_action_matching_ignores_game_id_metadata() -> None:
    expected = {
        "sequence": ["ACTION1:{}"],
        "terminal_risk": 0.1,
    }
    decision = {"sequences": [expected]}

    matched = assessment_for_live_action(
        decision,
        'ACTION1:{"game_id":"lp85-305b61c3"}',
    )

    assert matched is expected


def test_click_matching_preserves_semantic_coordinates() -> None:
    decision = {
        "sequences": [
            {"sequence": ['ACTION6:{"x":1,"y":2}']},
            {"sequence": ['ACTION6:{"x":3,"y":4}']},
        ]
    }

    matched = assessment_for_live_action(
        decision,
        'ACTION6:{"game_id":"unit","x":3,"y":4}',
    )
    missing = assessment_for_live_action(
        decision,
        'ACTION6:{"game_id":"unit","x":8,"y":9}',
    )

    assert matched == decision["sequences"][1]
    assert missing is None
