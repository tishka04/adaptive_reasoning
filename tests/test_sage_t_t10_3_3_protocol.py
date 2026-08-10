from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_3_protocol as protocol


def test_parent_snapshot_is_exact_and_diagnostic_only() -> None:
    diagnosis = protocol._parent_diagnosis(Path.cwd())
    for key, value in protocol.SUPERSEDED_T10_3_2.items():
        if key in diagnosis:
            assert diagnosis[key] == value
    assert diagnosis["intent_count"] == 99
    assert diagnosis["event_count"] == 99
    assert diagnosis["unchanged_frame_count"] == 91
    assert diagnosis["sage_t_decision_count"] == 0
    assert diagnosis["candidate_success_count"] == 0
    assert diagnosis["candidate_contradiction_count"] == 8


def test_fresh_matrix_is_exact_counterbalanced_and_nonoverlapping() -> None:
    core = protocol.work_specs("discover-core")
    sequence = protocol.work_specs("discover-sequence")
    confirmation = protocol.work_specs("confirm")
    assert len(core) == 4
    assert len(sequence) == 6
    assert len(confirmation) == 20
    assert len({row.work_id for row in (*core, *sequence, *confirmation)}) == 30
    assert protocol.maximum_actions_for_specs((*core, *sequence, *confirmation)) == 6144
    assert set(protocol.DISCOVERY_SEEDS).isdisjoint({3141, 3142})
    assert set(protocol.CONFIRMATION_SEEDS).isdisjoint({3151, 3152})
    first = {}
    for row in confirmation:
        first.setdefault((row.game_id, row.seed), row.arm)
    assert set(first.values()) == set(protocol.CONFIRMATION_ARMS)


def test_manifest_requires_ephemeral_binding_and_closed_firewalls() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    recovery = manifest["binding_recovery"]
    assert recovery["persistent_coordinates"] is False
    assert recovery["persistent_entity_identifiers"] is False
    assert recovery["branch_local_productive_anchor"] is True
    assert recovery["unique_structural_binding_requires_unique_candidate"] is True
    assert recovery["explicit_rejection_reasons"] is True
    assert manifest["superseded_t10_3_2"]["used_for_training"] is False
    assert not any(manifest["firewall"].values())
    assert manifest["matrix"]["total_resets"] == 30
    assert manifest["matrix"]["total_maximum_actions"] == 6144
