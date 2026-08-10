from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_5_protocol as protocol


def test_t10_3_4_snapshot_is_exact_terminal_and_diagnostic_only() -> None:
    diagnosis = protocol._parent_diagnosis(Path.cwd())
    for key, value in diagnosis.items():
        assert value == protocol.SUPERSEDED_T10_3_4[key]
    assert diagnosis["intent_count"] == 126
    assert diagnosis["event_count"] == 126
    assert diagnosis["branch_count"] == 4
    assert diagnosis["level_delta"] == 1
    assert diagnosis["verdict"] == "BOUNDED_CORE_MISS"


def test_fresh_matrix_preserves_registered_limits_and_counterbalancing() -> None:
    core = protocol.work_specs("discover-core")
    sequence = protocol.work_specs("discover-sequence")
    confirmation = protocol.work_specs("confirm")
    assert len(core) == 4
    assert len(sequence) == 6
    assert len(confirmation) == 20
    assert len({row.work_id for row in (*core, *sequence, *confirmation)}) == 30
    assert protocol.maximum_actions_for_specs((*core, *sequence, *confirmation)) == 6144
    assert set(protocol.DISCOVERY_SEEDS).isdisjoint({3181, 3182})
    assert set(protocol.CONFIRMATION_SEEDS).isdisjoint({3191, 3192})
    first = {}
    for row in confirmation:
        first.setdefault((row.game_id, row.seed), row.arm)
    assert set(first.values()) == set(protocol.CONFIRMATION_ARMS)


def test_manifest_preregisters_real_time_schedule_without_relaxing_gates() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    schedule = manifest["scheduled_control"]
    assert schedule["full_unified_decision_path_enabled"] is False
    assert schedule["full_unified_observation_path_enabled"] is False
    assert schedule["sage_t_posterior_each_transition"] is True
    assert schedule["productive_option_extension"] is True
    assert schedule["maximum_option_horizon"] == 32
    assert manifest["gates"]["maximum_decision_p95_ms"] == 2500.0
    assert manifest["gates"]["maximum_controller_cycle_p95_ms"] == 2500.0
    assert manifest["superseded_t10_3_4"]["used_for_training"] is False
    assert not any(manifest["firewall"].values())

