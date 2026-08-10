from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_2_protocol as protocol


def test_matrix_has_exact_resets_actions_seeds_and_counterbalancing() -> None:
    core = protocol.work_specs("discover-core")
    sequence = protocol.work_specs("discover-sequence")
    confirmation = protocol.work_specs("confirm")
    assert len(core) == 4
    assert len(sequence) == 6
    assert len(confirmation) == 20
    assert len({row.work_id for row in (*core, *sequence, *confirmation)}) == 30
    assert protocol.maximum_actions_for_specs((*core, *sequence, *confirmation)) == 6144
    assert {row.seed for row in (*core, *sequence)} == {3141, 3142}
    assert {row.seed for row in confirmation} == {3151, 3152}

    first_arms = {}
    for row in confirmation:
        first_arms.setdefault((row.game_id, row.seed), row.arm)
    assert set(first_arms.values()) == set(protocol.CONFIRMATION_ARMS)


def test_manifest_binds_partial_parent_and_keeps_every_firewall_closed() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    parent = manifest["superseded_t10_3_1"]
    assert parent["status"] == "SUPERSEDED_PARTIAL"
    assert parent["intent_count"] == 95
    assert parent["event_count"] == 94
    assert parent["branch_count"] == 7
    assert parent["interrupted_intent_count"] == 1
    assert parent["used_for_training"] is False
    assert len(manifest["parent_artifacts"]) == 6
    assert not any(manifest["firewall"].values())
    assert manifest["matrix"]["total_resets"] == 30
    assert manifest["matrix"]["total_maximum_actions"] == 6144
    assert set(manifest["cli_phases"]) == {
        "freeze",
        "status",
        "audit",
        "preflight",
        "discover-core",
        "discover-sequence",
        "compile",
        "confirm",
        "report",
    }
