from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_4_protocol as protocol


def test_t10_3_3_snapshot_is_exact_positive_and_diagnostic_only() -> None:
    diagnosis = protocol._parent_diagnosis(Path.cwd())
    for key, value in protocol.SUPERSEDED_T10_3_3.items():
        if key in diagnosis:
            assert diagnosis[key] == value
    assert diagnosis["intent_count"] == 76
    assert diagnosis["event_count"] == 76
    assert diagnosis["level_delta"] == 1
    assert diagnosis["winning_step_index"] == 51
    assert diagnosis["winning_decision_source"] == "sage_t_joint_program"
    assert diagnosis["candidate_success_count"] == 1


def test_fresh_matrix_preserves_maxima_and_counterbalancing() -> None:
    core = protocol.work_specs("discover-core")
    sequence = protocol.work_specs("discover-sequence")
    confirmation = protocol.work_specs("confirm")
    assert len(core) == 4
    assert len(sequence) == 6
    assert len(confirmation) == 20
    assert len({row.work_id for row in (*core, *sequence, *confirmation)}) == 30
    assert protocol.maximum_actions_for_specs((*core, *sequence, *confirmation)) == 6144
    assert set(protocol.DISCOVERY_SEEDS).isdisjoint({3161, 3162})
    assert set(protocol.CONFIRMATION_SEEDS).isdisjoint({3171, 3172})
    first = {}
    for row in confirmation:
        first.setdefault((row.game_id, row.seed), row.arm)
    assert set(first.values()) == set(protocol.CONFIRMATION_ARMS)


def test_manifest_preregisters_bounded_compute_without_relaxing_latency() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    bounded = manifest["bounded_compute"]
    assert bounded["stop_after_first_sealed_level"] is True
    assert bounded["same_profile_for_active_and_baseline"] is True
    assert bounded["transition_history_limit"] == 32
    assert bounded["operator_induction_interval"] == 8
    assert bounded["operator_planning_enabled"] is False
    assert manifest["gates"]["maximum_decision_p95_ms"] == 2500.0
    assert manifest["superseded_t10_3_3"]["used_for_training"] is False
    assert manifest["superseded_t10_3_3"]["positive_witness_imported_as_prior"] is False
    assert not any(manifest["firewall"].values())
