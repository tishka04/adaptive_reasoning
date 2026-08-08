from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from theory.sage_t import t10_2_2_protocol as protocol


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _sign(payload: dict[str, Any], *, checksum_key: str) -> dict[str, Any]:
    return protocol.signed_payload(payload, checksum_key=checksum_key)


def _self_authenticating_checkpoint(
    *, manifest_checksum: str, revision: int
) -> dict[str, Any]:
    unsigned = {
        "format_version": "sage-t10.2.1-collection-checkpoint-v1",
        "manifest_checksum": manifest_checksum,
        "lane_registry_sha256": "a" * 64,
        "lane_reports": [],
        "cumulative_active_seconds": 1.0,
        "open_lane_id": None,
        "open_lane_elapsed_seconds": 0.0,
        "journal_reconstructed": False,
        "checkpoint_reconstructed": False,
        "physical_steps_replayed_on_resume": 0,
        "revision": revision,
    }
    return {**unsigned, "checkpoint_checksum": protocol.canonical_sha256(unsigned)}


def _collection_report(
    *, manifest_checksum: str, checkpoint_checksum: str
) -> dict[str, Any]:
    return _sign(
        {
            "format_version": "sage-t10.2.1-protocol-v1",
            "phase": "collect",
            "manifest_checksum": manifest_checksum,
            "durability": {"checkpoint_checksum": checkpoint_checksum},
        },
        checksum_key="report_checksum",
    )


# ---------------------------------------------------------------------------
# Frozen-kernel lineage.
# ---------------------------------------------------------------------------
def test_lane_registry_matches_frozen_matrix_and_lane_ids() -> None:
    lanes = protocol.source_lane_registry()
    assert len(lanes) == len(protocol.SOURCE_GAMES) * (
        len(protocol.DISCOVERY_SEEDS) + len(protocol.CONFIRMATION_SEEDS)
    )
    # lane_id must equal sha256 of the identity dict (matches t10_2_1 runtime).
    for lane in lanes:
        identity = {k: lane[k] for k in ("split", "game_id", "seed")}
        assert lane["lane_id"] == protocol.canonical_sha256(identity)
    splits = {lane["split"] for lane in lanes}
    assert splits == set(protocol.SOURCE_SPLITS)


def test_lane_id_matches_runtime_source_lane_key() -> None:
    runtime = pytest.importorskip("theory.sage_t.t10_2_1_runtime")
    for lane in protocol.source_lane_registry():
        key = runtime.SourceLaneKey(lane["split"], lane["game_id"], lane["seed"])
        assert key.lane_id == lane["lane_id"]


# ---------------------------------------------------------------------------
# Item 1: report/checkpoint revision + checksum synchronization.
# ---------------------------------------------------------------------------
def test_checkpoint_binding_pins_revision_and_checksum() -> None:
    manifest = "m" * 64
    checkpoint = _self_authenticating_checkpoint(manifest_checksum=manifest, revision=7)
    report = _collection_report(
        manifest_checksum=manifest,
        checkpoint_checksum=checkpoint["checkpoint_checksum"],
    )
    binding = protocol.build_checkpoint_binding(
        collection_report=report, checkpoint=checkpoint
    )
    assert binding["checkpoint_revision"] == 7
    assert binding["checkpoint_checksum"] == checkpoint["checkpoint_checksum"]
    assert binding["synchronized"] is True
    unsigned = {k: v for k, v in binding.items() if k != "binding_checksum"}
    assert binding["binding_checksum"] == protocol.canonical_sha256(unsigned)


def test_checkpoint_binding_rejects_checksum_divergence() -> None:
    manifest = "m" * 64
    checkpoint = _self_authenticating_checkpoint(manifest_checksum=manifest, revision=1)
    report = _collection_report(
        manifest_checksum=manifest, checkpoint_checksum="b" * 64
    )
    with pytest.raises(protocol.ManifestDriftError, match="checkpoint checksum diverged"):
        protocol.build_checkpoint_binding(
            collection_report=report, checkpoint=checkpoint
        )


def test_checkpoint_binding_rejects_tampered_checkpoint_body() -> None:
    manifest = "m" * 64
    checkpoint = _self_authenticating_checkpoint(manifest_checksum=manifest, revision=1)
    checkpoint["revision"] = 99  # body changed but checksum not recomputed
    report = _collection_report(
        manifest_checksum=manifest,
        checkpoint_checksum=checkpoint["checkpoint_checksum"],
    )
    with pytest.raises(protocol.ManifestDriftError, match="does not authenticate"):
        protocol.build_checkpoint_binding(
            collection_report=report, checkpoint=checkpoint
        )


def test_synchronize_report_with_checkpoint_reads_disk(tmp_path: Path) -> None:
    manifest = "m" * 64
    checkpoint = _self_authenticating_checkpoint(manifest_checksum=manifest, revision=3)
    report = _collection_report(
        manifest_checksum=manifest,
        checkpoint_checksum=checkpoint["checkpoint_checksum"],
    )
    (tmp_path / protocol.COLLECTION_REPORT_FILENAME).write_text(
        protocol.canonical_json(report) + "\n", encoding="utf-8"
    )
    (tmp_path / protocol.CHECKPOINT_FILENAME).write_text(
        protocol.canonical_json(checkpoint) + "\n", encoding="utf-8"
    )
    binding = protocol.synchronize_report_with_checkpoint(output_dir=tmp_path)
    assert binding["checkpoint_revision"] == 3


def test_synchronize_report_missing_checkpoint_fails(tmp_path: Path) -> None:
    manifest = "m" * 64
    report = _collection_report(manifest_checksum=manifest, checkpoint_checksum="c" * 64)
    (tmp_path / protocol.COLLECTION_REPORT_FILENAME).write_text(
        protocol.canonical_json(report) + "\n", encoding="utf-8"
    )
    with pytest.raises(protocol.ManifestDriftError, match="checkpoint is missing"):
        protocol.synchronize_report_with_checkpoint(output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Item 2: phase-level timing.
# ---------------------------------------------------------------------------
def test_startup_latency_is_lane_start_to_first_committed() -> None:
    lane = protocol.source_lane_registry()[0]
    timing = protocol.compute_lane_startup_timing(
        lane=lane,
        lane_started_seconds=100.0,
        first_committed_transition_seconds=103.5,
        lane_finished_seconds=160.0,
    )
    assert timing["committed_first_transition"] is True
    assert timing["startup_latency_seconds"] == pytest.approx(3.5)
    assert timing["interaction_seconds"] == pytest.approx(56.5)


def test_uncommitted_lane_has_no_startup_latency() -> None:
    lane = protocol.source_lane_registry()[0]
    timing = protocol.compute_lane_startup_timing(
        lane=lane,
        lane_started_seconds=10.0,
        first_committed_transition_seconds=None,
        lane_finished_seconds=25.0,
    )
    assert timing["committed_first_transition"] is False
    assert timing["startup_latency_seconds"] is None
    assert timing["interaction_seconds"] is None


def test_first_committed_outside_window_is_rejected() -> None:
    lane = protocol.source_lane_registry()[0]
    with pytest.raises(protocol.DataGateError):
        protocol.compute_lane_startup_timing(
            lane=lane,
            lane_started_seconds=10.0,
            first_committed_transition_seconds=5.0,
            lane_finished_seconds=25.0,
        )


def test_build_phase_timing_aggregates() -> None:
    lane = protocol.source_lane_registry()[0]
    committed = protocol.compute_lane_startup_timing(
        lane=lane,
        lane_started_seconds=0.0,
        first_committed_transition_seconds=2.0,
        lane_finished_seconds=10.0,
    )
    uncommitted = protocol.compute_lane_startup_timing(
        lane=lane,
        lane_started_seconds=0.0,
        first_committed_transition_seconds=None,
        lane_finished_seconds=10.0,
    )
    summary = protocol.build_phase_timing(lane_timings=[committed, uncommitted])
    assert summary["committed_lane_count"] == 1
    assert summary["uncommitted_lane_count"] == 1
    assert summary["max_startup_latency_seconds"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Item 3: fail fast on missing cross-fit artifacts and first-intent timeout.
# ---------------------------------------------------------------------------
def test_require_cross_fit_artifacts(tmp_path: Path) -> None:
    with pytest.raises(protocol.GateRefusalError, match="missing"):
        protocol.require_cross_fit_artifacts(tmp_path)
    path = tmp_path / protocol.CROSS_FIT_AUDIT_FILENAME
    path.write_text("", encoding="utf-8")
    with pytest.raises(protocol.GateRefusalError, match="empty"):
        protocol.require_cross_fit_artifacts(tmp_path)
    path.write_text("{}", encoding="utf-8")
    assert protocol.require_cross_fit_artifacts(tmp_path) == path


def test_first_intent_classification_relative_to_readiness() -> None:
    assert (
        protocol.classify_first_intent(
            controller_ready_at=10.0,
            environment_ready_at=12.0,
            first_intent_authorized_at=14.0,
            first_intent_budget_seconds=5.0,
        )
        == "ok"
    )
    # Late relative to readiness (max(10,12)=12, deadline 17).
    assert (
        protocol.classify_first_intent(
            controller_ready_at=10.0,
            environment_ready_at=12.0,
            first_intent_authorized_at=18.0,
            first_intent_budget_seconds=5.0,
        )
        == "first_intent_timeout"
    )
    assert (
        protocol.classify_first_intent(
            controller_ready_at=10.0,
            environment_ready_at=12.0,
            first_intent_authorized_at=None,
            first_intent_budget_seconds=5.0,
        )
        == "first_intent_timeout"
    )


def test_fail_fast_preflight_combined(tmp_path: Path) -> None:
    (tmp_path / protocol.CROSS_FIT_AUDIT_FILENAME).write_text("{}", encoding="utf-8")
    assert protocol.fail_fast_preflight(output_dir=tmp_path, first_intent_status="ok")
    with pytest.raises(protocol.GateRefusalError, match="first intent"):
        protocol.fail_fast_preflight(
            output_dir=tmp_path, first_intent_status="first_intent_timeout"
        )


# ---------------------------------------------------------------------------
# Item 4: readiness-gated interaction deadlines.
# ---------------------------------------------------------------------------
def test_readiness_gate_anchors_deadline_at_joint_readiness() -> None:
    lane = protocol.source_lane_registry()[0]
    gate = protocol.readiness_gate(
        lane=lane,
        controller_ready_at=8.0,
        environment_ready_at=11.0,
        interaction_budget_seconds=64.0,
    )
    assert gate["interaction_started_at"] == pytest.approx(11.0)
    assert gate["interaction_deadline"] == pytest.approx(75.0)
    assert gate["clock_anchored_at_readiness"] is True


def test_readiness_gate_refuses_before_readiness() -> None:
    lane = protocol.source_lane_registry()[0]
    with pytest.raises(protocol.GateRefusalError):
        protocol.readiness_gate(
            lane=lane,
            controller_ready_at=None,
            environment_ready_at=11.0,
            interaction_budget_seconds=64.0,
        )
    with pytest.raises(protocol.GateRefusalError):
        protocol.readiness_gate(
            lane=lane,
            controller_ready_at=8.0,
            environment_ready_at=None,
            interaction_budget_seconds=64.0,
        )


# ---------------------------------------------------------------------------
# Item 5: evidence funnel with rejection-reason accounting.
# ---------------------------------------------------------------------------
def test_evidence_funnel_fully_accounted() -> None:
    funnel = protocol.build_evidence_funnel(
        observed_intents=100,
        authorized_intents=100,
        sealed_events=90,
        rejections={"cooperative_reset_deadline": 6, "hard_reset_timeout": 4},
    )
    assert funnel["fully_accounted"] is True
    assert funnel["conservation_holds"] is True
    assert funnel["rejected_intents"] == 10


def test_evidence_funnel_detects_unaccounted() -> None:
    funnel = protocol.build_evidence_funnel(
        observed_intents=100,
        authorized_intents=100,
        sealed_events=90,
        rejections={"cooperative_reset_deadline": 5},
    )
    assert funnel["fully_accounted"] is False


def test_evidence_funnel_rejects_unregistered_reason() -> None:
    with pytest.raises(protocol.DataGateError, match="unregistered rejection reason"):
        protocol.build_evidence_funnel(
            observed_intents=10,
            authorized_intents=10,
            sealed_events=9,
            rejections={"made_up_reason": 1},
        )


def test_evidence_funnel_from_reset_reports() -> None:
    reports = [
        {"issued_intents": 64, "sealed_events": 64, "unresolved_intents": 0,
         "stop_reason": "registered_collection_deadline"},
        {"issued_intents": 64, "sealed_events": 60, "unresolved_intents": 4,
         "stop_reason": "cooperative_reset_deadline"},
    ]
    funnel = protocol.evidence_funnel_from_reset_reports(reports)
    assert funnel["sealed_events"] == 124
    assert funnel["rejections"] == {"cooperative_reset_deadline": 4}
    assert funnel["fully_accounted"] is True


# ---------------------------------------------------------------------------
# Item 6: schema families vs grounded instances.
# ---------------------------------------------------------------------------
def test_schema_families_separated_from_grounded_instances() -> None:
    evidence = protocol.partition_schema_evidence(
        learned_schema_counts={("move", 1): 5, ("rotate", 0): 2},
        independent_schema_counts={("move", 1): 3},
        grounding_counts={"move:[0]": 4, "move:[1]": 1, "rotate:[]": 2},
    )
    assert evidence["canonical_families"] == ["move:1", "rotate:0"]
    assert evidence["canonical_family_count"] == 2
    assert evidence["grounded_instance_count"] == 3
    assert evidence["families_are_coordinate_free"] is True
    # Two families but three groundings: grounding must never inflate capacity.
    assert evidence["learned_schema_counts"] == {"move:1": 5, "rotate:0": 2}


# ---------------------------------------------------------------------------
# Item 7: induction canary.
# ---------------------------------------------------------------------------
def test_induction_canary_green_with_default() -> None:
    canary = protocol.run_induction_canary()
    assert canary["passed"] is True
    assert canary["deterministic"] is True
    assert canary["induced_family_count"] >= 1


def test_induction_canary_red_when_induction_yields_nothing() -> None:
    canary = protocol.run_induction_canary(induct=lambda evidence: [])
    assert canary["passed"] is False


def test_induction_canary_red_when_nondeterministic() -> None:
    state = {"n": 0}

    def flaky(evidence: Any) -> list[str]:
        state["n"] += 1
        return ["move:1"] if state["n"] % 2 == 1 else ["rotate:0"]

    canary = protocol.run_induction_canary(induct=flaky)
    assert canary["deterministic"] is False
    assert canary["passed"] is False


# ---------------------------------------------------------------------------
# Item 8: interleaving + reserved confirmation capacity.
# ---------------------------------------------------------------------------
def test_schedule_interleaves_and_covers_all_lanes() -> None:
    schedule = protocol.interleaved_lane_schedule()
    order = schedule["order"]
    assert len(order) == len(protocol.source_lane_registry())
    # First two lanes should straddle both splits (discovery then confirmation).
    assert order[0]["split"] == "discovery"
    assert order[1]["split"] == "leave_one_game_out_confirmation"
    assert schedule["truncated"] is False


def test_reserved_confirmation_capacity_survives_truncation() -> None:
    reserve = protocol.reserved_confirmation_capacity()
    assert reserve >= 1
    schedule = protocol.interleaved_lane_schedule(lane_budget=reserve + 1)
    assert schedule["truncated"] is True
    confirmations = [
        lane
        for lane in schedule["order"]
        if lane["split"] == "leave_one_game_out_confirmation"
    ]
    assert len(confirmations) >= reserve


def test_schedule_refuses_budget_below_reserve() -> None:
    reserve = protocol.reserved_confirmation_capacity()
    with pytest.raises(protocol.ResourceGateError):
        protocol.interleaved_lane_schedule(lane_budget=reserve - 1)


# ---------------------------------------------------------------------------
# Item 9: smoke lanes.
# ---------------------------------------------------------------------------
def test_smoke_plan_one_lane_per_split_precedes_matrix() -> None:
    plan = protocol.smoke_lane_plan()
    assert plan["smoke_lane_count"] == len(protocol.SOURCE_SPLITS)
    assert plan["one_per_split"] is True
    assert plan["smoke_precedes_matrix"] is True
    smoke_splits = {lane["split"] for lane in plan["smoke_lanes"]}
    assert smoke_splits == set(protocol.SOURCE_SPLITS)
    assert (
        plan["smoke_lane_count"] + plan["remaining_matrix_lane_count"]
        == plan["total_lane_count"]
    )


# ---------------------------------------------------------------------------
# Invariant guards.
# ---------------------------------------------------------------------------
def test_schema_learning_failure_requires_delivered_independent_evidence() -> None:
    # Evidence not delivered -> re-attributed to acquisition.
    guarded = protocol.guard_schema_learning_verdict(
        verdict="OPTION_SYNTHESIS_MISS",
        independent_evidence_generated=True,
        independent_evidence_delivered=False,
    )
    assert guarded["verdict"] == protocol.EVIDENCE_UNMET_VERDICT
    assert guarded["adjusted"] is True

    # Evidence generated AND delivered -> the learner verdict stands.
    kept = protocol.guard_schema_learning_verdict(
        verdict="OPTION_SYNTHESIS_MISS",
        independent_evidence_generated=True,
        independent_evidence_delivered=True,
    )
    assert kept["verdict"] == "OPTION_SYNTHESIS_MISS"
    assert kept["adjusted"] is False


def test_non_schema_verdict_is_untouched() -> None:
    guarded = protocol.guard_schema_learning_verdict(
        verdict="FRAME_TRANSPORT_MISS",
        independent_evidence_generated=False,
        independent_evidence_delivered=False,
    )
    # FRAME_TRANSPORT_MISS is not in the schema-learning set.
    assert guarded["verdict"] == "FRAME_TRANSPORT_MISS"


def test_transfer_failure_never_declared_for_zero_intent_lanes() -> None:
    # Every lane issued zero intents -> transfer was never exercised.
    guarded = protocol.guard_transfer_verdict(
        verdict="SOURCE_VALIDATION_TRANSFER_MISS",
        lane_intent_counts={"lane-a": 0, "lane-b": 0},
    )
    assert guarded["verdict"] == protocol.EVIDENCE_UNMET_VERDICT
    assert guarded["adjusted"] is True
    assert guarded["zero_intent_lane_ids"] == ["lane-a", "lane-b"]


def test_transfer_failure_stands_with_qualifying_lane() -> None:
    guarded = protocol.guard_transfer_verdict(
        verdict="SOURCE_VALIDATION_TRANSFER_MISS",
        lane_intent_counts={"lane-a": 0, "lane-b": 64},
    )
    assert guarded["verdict"] == "SOURCE_VALIDATION_TRANSFER_MISS"
    assert guarded["qualifying_lane_ids"] == ["lane-b"]
    assert guarded["zero_intent_lane_ids"] == ["lane-a"]


def test_guard_rejects_unregistered_verdict() -> None:
    with pytest.raises(protocol.DataGateError):
        protocol.guard_transfer_verdict(
            verdict="NOT_A_VERDICT", lane_intent_counts={}
        )


# ---------------------------------------------------------------------------
# Top-level orchestration composition.
# ---------------------------------------------------------------------------
def _orchestration_inputs(*, proposed_verdict: str, intents: dict[str, int]):
    manifest = "m" * 64
    checkpoint = _self_authenticating_checkpoint(manifest_checksum=manifest, revision=2)
    report = _collection_report(
        manifest_checksum=manifest,
        checkpoint_checksum=checkpoint["checkpoint_checksum"],
    )
    binding = protocol.build_checkpoint_binding(
        collection_report=report, checkpoint=checkpoint
    )
    lane = protocol.source_lane_registry()[0]
    timing = protocol.build_phase_timing(
        lane_timings=[
            protocol.compute_lane_startup_timing(
                lane=lane,
                lane_started_seconds=0.0,
                first_committed_transition_seconds=1.0,
                lane_finished_seconds=10.0,
            )
        ]
    )
    funnel = protocol.build_evidence_funnel(
        observed_intents=10, authorized_intents=10, sealed_events=10, rejections={}
    )
    schema = protocol.partition_schema_evidence(
        learned_schema_counts={("move", 1): 1},
        independent_schema_counts={("move", 1): 1},
        grounding_counts={"move:[0]": 1},
    )
    canary = protocol.run_induction_canary()
    schedule = protocol.interleaved_lane_schedule()
    smoke = protocol.smoke_lane_plan()
    return dict(
        manifest_checksum=manifest,
        checkpoint_binding=binding,
        phase_timing=timing,
        evidence_funnel=funnel,
        schema_evidence=schema,
        induction_canary=canary,
        lane_schedule=schedule,
        smoke_plan=smoke,
        proposed_verdict=proposed_verdict,
        lane_intent_counts=intents,
    )


def test_orchestration_report_applies_both_guards() -> None:
    # Transfer miss but no lane issued intents -> re-attributed to acquisition.
    report = protocol.build_orchestration_report(
        **_orchestration_inputs(
            proposed_verdict="SOURCE_VALIDATION_TRANSFER_MISS",
            intents={"lane-a": 0},
        )
    )
    assert report["verdict"] == protocol.EVIDENCE_UNMET_VERDICT
    assert report["verdict_adjusted"] is True
    unsigned = {k: v for k, v in report.items() if k != "report_checksum"}
    assert report["report_checksum"] == protocol.canonical_sha256(unsigned)


def test_orchestration_report_keeps_supported_verdict() -> None:
    report = protocol.build_orchestration_report(
        **_orchestration_inputs(
            proposed_verdict="SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED",
            intents={"lane-a": 64},
        )
    )
    assert report["verdict"] == "SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED"
    assert report["verdict_adjusted"] is False


def test_orchestration_schema_verdict_needs_green_canary() -> None:
    inputs = _orchestration_inputs(
        proposed_verdict="COMMON_POSTERIOR_MISS", intents={"lane-a": 64}
    )
    # Force a red canary: independent evidence exists but induction failed.
    inputs["induction_canary"] = protocol.run_induction_canary(induct=lambda e: [])
    report = protocol.build_orchestration_report(**inputs)
    assert report["verdict"] == protocol.EVIDENCE_UNMET_VERDICT


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def test_cli_orchestrate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = "m" * 64
    checkpoint = _self_authenticating_checkpoint(manifest_checksum=manifest, revision=5)
    report = _collection_report(
        manifest_checksum=manifest,
        checkpoint_checksum=checkpoint["checkpoint_checksum"],
    )
    (tmp_path / protocol.COLLECTION_REPORT_FILENAME).write_text(
        protocol.canonical_json(report) + "\n", encoding="utf-8"
    )
    (tmp_path / protocol.CHECKPOINT_FILENAME).write_text(
        protocol.canonical_json(checkpoint) + "\n", encoding="utf-8"
    )
    code = protocol.main(["orchestrate", "--output-dir", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["checkpoint_revision"] == 5


def test_cli_delegates_kernel_phases() -> None:
    with pytest.raises(SystemExit):
        protocol.main(["collect"])
