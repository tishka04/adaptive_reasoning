from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from theory.sage_t import t10_2_2_protocol as protocol
from theory.sage_t import t10_2_2_runtime as runtime


# ---------------------------------------------------------------------------
# Lightweight fakes for the parent-side seam (no real multiprocessing).
# ---------------------------------------------------------------------------
class _FakeLane:
    def __init__(self, split: str, game_id: str, seed: int) -> None:
        self._d = {
            "split": split,
            "game_id": game_id,
            "seed": seed,
            "lane_id": protocol.canonical_sha256(
                {"split": split, "game_id": game_id, "seed": seed}
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(self._d)


class _FakeWork:
    def __init__(self, lane: _FakeLane, reset_index: int) -> None:
        self.lane = lane
        self.reset_index = reset_index
        self.work_id = protocol.canonical_sha256(
            {"lane_id": lane.to_dict()["lane_id"], "reset_index": reset_index}
        )


class _FakeOutcome:
    def __init__(self, status: str, stop_reason: str | None = None) -> None:
        self.status = status
        self.payload = {} if stop_reason is None else {"stop_reason": stop_reason}


class _ScriptedInnerWatchdog:
    """Feeds a fixed message script through the handler, then returns outcome."""

    def __init__(self, messages: list[dict[str, Any]], outcome: _FakeOutcome) -> None:
        self._messages = messages
        self._outcome = outcome
        self.acks: list[Any] = []

    def supervise(
        self, process: Any, *, message_handler, **kwargs: Any
    ) -> _FakeOutcome:
        for message in self._messages:
            ack = message_handler(message)
            self.acks.append(ack)
        return self._outcome


def _lane() -> _FakeLane:
    game = protocol.SOURCE_GAMES[0]
    return _FakeLane("discovery", game, protocol.DISCOVERY_SEEDS[0])


# ---------------------------------------------------------------------------
# Recorder.
# ---------------------------------------------------------------------------
def test_recorder_captures_readiness_and_first_events() -> None:
    recorder = runtime.CollectionTimingRecorder()
    work = _FakeWork(_lane(), 0)
    recorder.note_reset_started(work, 100.0)
    recorder.observe(
        work,
        {
            "kind": "reset_operation",
            "payload": {"operation": "open", "stage": "started"},
        },
        101.0,
    )
    recorder.observe(
        work,
        {
            "kind": "reset_operation",
            "payload": {"operation": "reset", "stage": "finished"},
        },
        102.0,
    )
    recorder.observe(work, {"kind": "action_intent"}, 103.0)
    recorder.observe(work, {"kind": "physical_event"}, 104.0)
    recorder.observe(work, {"kind": "physical_event"}, 105.0)  # only first counts
    recorder.note_reset_finished(work, 160.0, status="COMPLETE", stop_reason=None)

    timing = recorder.reset_timings()[0]
    assert timing.controller_ready_at == 101.0
    assert timing.environment_ready_at == 102.0
    assert timing.first_intent_at == 103.0
    assert timing.first_committed_at == 104.0
    assert timing.finished_at == 160.0
    assert timing.status == "COMPLETE"


def test_lane_windows_aggregate_over_resets() -> None:
    recorder = runtime.CollectionTimingRecorder()
    lane = _lane()
    w0, w1 = _FakeWork(lane, 0), _FakeWork(lane, 1)
    recorder.note_reset_started(w0, 10.0)
    recorder.observe(w0, {"kind": "physical_event"}, 12.0)
    recorder.note_reset_finished(w0, 20.0, status="COMPLETE", stop_reason=None)
    recorder.note_reset_started(w1, 21.0)
    recorder.observe(w1, {"kind": "physical_event"}, 22.0)
    recorder.note_reset_finished(w1, 30.0, status="COMPLETE", stop_reason=None)

    windows = recorder.lane_windows()
    assert len(windows) == 1
    window = windows[0]
    assert window["lane_started_seconds"] == 10.0
    assert window["first_committed_transition_seconds"] == 12.0
    assert window["lane_finished_seconds"] == 30.0


# ---------------------------------------------------------------------------
# Observing watchdog delegates and records without changing semantics.
# ---------------------------------------------------------------------------
def test_timing_watchdog_observes_and_delegates() -> None:
    recorder = runtime.CollectionTimingRecorder()
    work = _FakeWork(_lane(), 0)
    messages = [
        {
            "kind": "reset_operation",
            "payload": {"operation": "open", "stage": "started"},
        },
        {
            "kind": "reset_operation",
            "payload": {"operation": "reset", "stage": "finished"},
        },
        {"kind": "action_intent"},
        {"kind": "physical_event"},
    ]
    handled: list[str] = []

    def handler(message):
        handled.append(str(message.get("kind")))
        return {"ack": True}

    inner = _ScriptedInnerWatchdog(messages, _FakeOutcome("COMPLETE"))
    watchdog = runtime.LaneTimingWatchdog(recorder=recorder, inner=inner)
    outcome = watchdog.supervise(
        object(),
        work=work,
        cancel_event=object(),
        outbound_queue=object(),
        inbound_queue=object(),
        message_handler=handler,
        cooperative_seconds=55.0,
        hard_seconds=60.0,
        started_at=0.0,
    )
    assert outcome.status == "COMPLETE"
    # Every message was forwarded unchanged to the frozen handler.
    assert handled == [
        "reset_operation",
        "reset_operation",
        "action_intent",
        "physical_event",
    ]
    assert inner.acks == [{"ack": True}] * 4
    timing = recorder.reset_timings()[0]
    assert timing.controller_ready_at is not None
    assert timing.environment_ready_at is not None
    assert timing.first_intent_at is not None
    assert timing.first_committed_at is not None
    assert timing.status == "COMPLETE"


class _RecordingInner:
    """Captures the seconds and handler it is supervised with."""

    def __init__(self) -> None:
        self.cooperative_seconds: float | None = None
        self.hard_seconds: float | None = None
        self.handler: Any = None

    def supervise(
        self, process, *, message_handler, cooperative_seconds, hard_seconds, **kwargs
    ):
        self.cooperative_seconds = cooperative_seconds
        self.hard_seconds = hard_seconds
        self.handler = message_handler
        return _FakeOutcome("COMPLETE")


class _FakeHandler:
    """Stand-in for the frozen _ParentJournalMessageHandler."""

    def __init__(self, intent_deadline: float) -> None:
        self._intent_deadline = intent_deadline

    def __call__(self, message):
        return None


def test_action_budget_watchdog_widens_deadline_to_liveness() -> None:
    recorder = runtime.CollectionTimingRecorder()
    work = _FakeWork(_lane(), 0)
    inner_recorder = _RecordingInner()
    timing = runtime.LaneTimingWatchdog(recorder=recorder, inner=inner_recorder)
    watchdog = runtime.ActionBudgetWatchdog(recorder=recorder, inner=timing)
    handler = _FakeHandler(intent_deadline=42.0)  # the frozen 55 s-anchored value
    watchdog.supervise(
        object(),
        work=work,
        cancel_event=object(),
        outbound_queue=object(),
        inbound_queue=object(),
        message_handler=handler,
        cooperative_seconds=55.0,  # frozen value handed in
        hard_seconds=60.0,
        started_at=1000.0,
    )
    # The frozen 55/60 are ignored; supervision uses the wide liveness bound.
    assert inner_recorder.cooperative_seconds == runtime.RESET_LIVENESS_WALL_SECONDS
    assert inner_recorder.hard_seconds == (
        runtime.RESET_LIVENESS_WALL_SECONDS + runtime.RESET_LIVENESS_HARD_GRACE_SECONDS
    )
    # The handler's intent deadline is extended so actions run to the budget.
    assert handler._intent_deadline == 1000.0 + runtime.RESET_LIVENESS_WALL_SECONDS


def test_action_budget_factory_arms_independent_guard_with_wide_bounds(
    tmp_path: Path,
) -> None:
    class FakeProcess:
        pid = 12345

        def start(self) -> None:
            return None

    class FakeContext:
        def Queue(self):
            return object()

        def Event(self):
            return object()

        def Process(self, **kwargs):
            assert kwargs["name"].startswith("sage-t10-2-2-")
            return FakeProcess()

    class RecordingWatchdog:
        cooperative = None
        hard = None
        intent_deadline = None

        def supervise(
            self,
            process,
            *,
            message_handler,
            cooperative_seconds,
            hard_seconds,
            **kwargs,
        ):
            self.cooperative = cooperative_seconds
            self.hard = hard_seconds
            self.intent_deadline = message_handler._intent_deadline
            return runtime._t10_2_1_runtime.WorkerOutcome(
                status="COMPLETE",
                elapsed_seconds=1.0,
                payload={"completed": True, "stop_reason": "budget_exhausted"},
            )

    with runtime.execution_bindings(
        mode="smoke", artifact_root=tmp_path / "smoke"
    ) as lanes:
        journal = runtime.IncrementalDurableCollectionJournal(
            tmp_path / "smoke" / "source_collection_journal",
            manifest_checksum="c" * 64,
        )
        watchdog = RecordingWatchdog()
        factory = runtime.ActionBudgetSourceFactory(
            manifest={"manifest_checksum": "c" * 64}, watchdog=watchdog
        )
        work = runtime._t10_2_1_runtime.reset_work_specs(lanes[0])[0]
        factory.run_reset(
            work=work,
            journal=journal,
            discovery_events=(),
            continuation={},
            process_context=FakeContext(),
            lane_remaining_seconds=2700.0,
            cooperative_collection_remaining_seconds=42600.0,
            absolute_collection_remaining_seconds=43200.0,
            clock=lambda: 1000.0,
        )
        assert watchdog.cooperative == runtime.RESET_LIVENESS_WALL_SECONDS
        assert watchdog.hard == (
            runtime.RESET_LIVENESS_WALL_SECONDS
            + runtime.RESET_LIVENESS_HARD_GRACE_SECONDS
        )
        assert watchdog.intent_deadline == 1600.0


def test_budget_policy_is_action_based_and_honest_about_frozen_caps() -> None:
    policy = runtime.budget_policy()
    assert policy["budget_basis"] == "physical_action_count"
    assert policy["wall_clock_role"] == "liveness_watchdog_only"
    assert policy["reset_level_applied"] is True
    assert policy["lane_and_collection_level_applied"] is True
    assert policy["incremental_history_state_applied"] is True
    assert policy["compact_cursor_applied"] is True
    assert policy["reset_action_budget"] == 64
    assert policy["learned_preview_copy_applied"] is True
    assert policy["preview_copy_adapter_scope"] == [
        "GaugeProgramPosterior",
        "FactorizedGaugeProgramPosterior",
    ]


@pytest.mark.parametrize(
    "posterior_type",
    (
        runtime._t10_2_1_runtime.GaugeProgramPosterior,
        runtime._t10_2_1_runtime.FactorizedGaugeProgramPosterior,
    ),
)
def test_worker_scoped_preview_copy_isolates_mutable_posterior_state(
    posterior_type,
) -> None:
    posterior = posterior_type()
    immutable_particle = MappingProxyType({"contract": "frozen"})
    posterior._particles = [immutable_particle]
    posterior._event_ids = ["donor-event"]
    posterior._seen_event_ids = {"donor-event"}
    posterior.executor._step_cache[("p", "s", "a")] = object()

    base_type = runtime._t10_2_1_runtime.GaugeProgramPosterior
    with runtime.gauge_preview_copy_binding():
        preview = copy.deepcopy(posterior)

    assert preview is not posterior
    assert preview._particles is not posterior._particles
    assert preview._particles[0] is immutable_particle
    assert preview._event_ids is not posterior._event_ids
    assert preview._seen_event_ids is not posterior._seen_event_ids
    assert preview.executor is not posterior.executor
    assert preview.executor._step_cache is not posterior.executor._step_cache
    assert "__deepcopy__" not in base_type.__dict__


def test_provenance_includes_budget_policy() -> None:
    prov = runtime.build_implementation_provenance()
    assert prov["budget_policy"]["budget_basis"] == "physical_action_count"


def test_timing_watchdog_records_finish_on_handler_exception() -> None:
    recorder = runtime.CollectionTimingRecorder()
    work = _FakeWork(_lane(), 0)

    class _RaisingInner:
        def supervise(self, process, *, message_handler, **kwargs):
            message_handler({"kind": "action_intent"})
            raise RuntimeError("boom")

    watchdog = runtime.LaneTimingWatchdog(recorder=recorder, inner=_RaisingInner())
    with pytest.raises(RuntimeError, match="boom"):
        watchdog.supervise(
            object(),
            work=work,
            cancel_event=object(),
            outbound_queue=object(),
            inbound_queue=object(),
            message_handler=lambda m: None,
            cooperative_seconds=55.0,
            hard_seconds=60.0,
            started_at=0.0,
        )
    # The attempted intent was still observed, and finish recorded in `finally`.
    timing = recorder.reset_timings()[0]
    assert timing.first_intent_at is not None
    assert timing.finished_at is not None
    assert timing.status == "EXCEPTION"


# ---------------------------------------------------------------------------
# Sidecar report construction.
# ---------------------------------------------------------------------------
def _self_authenticating_checkpoint(*, manifest_checksum: str, revision: int, resets):
    lane = _lane().to_dict()
    lane_report = {"lane": lane, "resets": list(resets)}
    unsigned = {
        "format_version": "sage-t10.2.1-collection-checkpoint-v1",
        "manifest_checksum": manifest_checksum,
        "lane_registry_sha256": "a" * 64,
        "lane_reports": [lane_report],
        "cumulative_active_seconds": 1.0,
        "open_lane_id": None,
        "open_lane_elapsed_seconds": 0.0,
        "journal_reconstructed": False,
        "checkpoint_reconstructed": False,
        "physical_steps_replayed_on_resume": 0,
        "revision": revision,
    }
    return {**unsigned, "checkpoint_checksum": protocol.canonical_sha256(unsigned)}


def _smoke_inputs(*, confirmation_status: str):
    manifest = "s" * 64
    lane_reports = []
    recorder = runtime.CollectionTimingRecorder()
    plan = protocol.smoke_lane_plan()
    for lane_index, lane in enumerate(plan["smoke_lanes"]):
        resets = []
        fake_lane = _FakeLane(lane["split"], lane["game_id"], lane["seed"])
        for reset_index in range(plan["resets_per_lane"]):
            is_failed_learned = bool(
                lane["split"] == "leave_one_game_out_confirmation"
                and reset_index == 1
                and confirmation_status != "COMPLETE"
            )
            status = confirmation_status if is_failed_learned else "COMPLETE"
            stop_reason = None if status == "COMPLETE" else "worker_exception"
            controller = (
                "balanced_discovery"
                if lane["split"] == "discovery"
                else plan["confirmation_controller_sequence"][reset_index]
            )
            reset = {
                **_reset_report(
                    reset_index=reset_index,
                    issued=1,
                    sealed=1,
                    unresolved=0,
                    stop_reason=stop_reason or "game_over",
                ),
                "status": status,
                "work": {
                    "reset_index": reset_index,
                    "controller": controller,
                },
            }
            resets.append(reset)
            work = _FakeWork(fake_lane, reset_index)
            base = float((lane_index * plan["resets_per_lane"] + reset_index) * 10)
            recorder.note_reset_started(work, base)
            recorder.observe(
                work,
                {
                    "kind": "reset_operation",
                    "payload": {"operation": "open", "stage": "started"},
                },
                base + 1.0,
            )
            recorder.observe(
                work,
                {
                    "kind": "reset_operation",
                    "payload": {"operation": "reset", "stage": "finished"},
                },
                base + 2.0,
            )
            recorder.observe(work, {"kind": "action_intent"}, base + 3.0)
            recorder.observe(work, {"kind": "physical_event"}, base + 4.0)
            recorder.note_reset_finished(
                work,
                base + 5.0,
                status=status,
                stop_reason=stop_reason,
                error_kind=("RuntimeError" if stop_reason else None),
            )
        lane_status = (
            "COMPLETE"
            if all(reset["status"] == "COMPLETE" for reset in resets)
            else "ABORTED"
        )
        lane_reports.append(
            {"lane": dict(lane), "status": lane_status, "resets": resets}
        )

    unsigned_checkpoint = {
        "format_version": "sage-t10.2.1-collection-checkpoint-v1",
        "manifest_checksum": manifest,
        "lane_registry_sha256": "a" * 64,
        "lane_reports": lane_reports,
        "cumulative_active_seconds": 1.0,
        "open_lane_id": None,
        "open_lane_elapsed_seconds": 0.0,
        "journal_reconstructed": False,
        "checkpoint_reconstructed": False,
        "physical_steps_replayed_on_resume": 0,
        "revision": plan["reset_report_count"],
    }
    checkpoint = {
        **unsigned_checkpoint,
        "checkpoint_checksum": protocol.canonical_sha256(unsigned_checkpoint),
    }
    collection_report = protocol.signed_payload(
        {
            "format_version": "sage-t10.2.1-protocol-v1",
            "phase": "collect",
            "manifest_checksum": manifest,
            "durability": {"checkpoint_checksum": checkpoint["checkpoint_checksum"]},
            "action_accounting": {
                "authorized_intent_count": plan["reset_report_count"],
                "sealed_event_count": plan["reset_report_count"],
                "explicitly_unresolved_intent_count": 0,
                "unknown_intent_count": 0,
                "equation_holds": True,
            },
        },
        checksum_key="report_checksum",
    )
    cursor = protocol.signed_payload(
        {
            "format_version": runtime.CURSOR_FORMAT_VERSION,
            "full_history_scan_count": 1,
        },
        checksum_key="cursor_checksum",
    )
    return recorder, checkpoint, collection_report, cursor


def _reset_report(
    *, reset_index: int, issued: int, sealed: int, unresolved: int, stop_reason: str
):
    return {
        "reset_index": reset_index,
        "issued_intents": issued,
        "sealed_events": sealed,
        "unresolved_intents": unresolved,
        "stop_reason": stop_reason,
        "continuation": {
            "learned_schema_counts": {json.dumps(["move", 1]): 3},
            "independent_schema_counts": {json.dumps(["move", 1]): 2},
            "grounding_counts": {"move:[0]": 2, "move:[1]": 1},
        },
    }


def _build_report(recorder: runtime.CollectionTimingRecorder):
    manifest = "m" * 64
    resets = [
        _reset_report(
            reset_index=0,
            issued=64,
            sealed=64,
            unresolved=0,
            stop_reason="registered_collection_deadline",
        ),
        _reset_report(
            reset_index=1,
            issued=64,
            sealed=60,
            unresolved=4,
            stop_reason="cooperative_reset_deadline",
        ),
    ]
    checkpoint = _self_authenticating_checkpoint(
        manifest_checksum=manifest, revision=4, resets=resets
    )
    collection_report = protocol.signed_payload(
        {
            "format_version": "sage-t10.2.1-protocol-v1",
            "phase": "collect",
            "manifest_checksum": manifest,
            "durability": {"checkpoint_checksum": checkpoint["checkpoint_checksum"]},
        },
        checksum_key="report_checksum",
    )
    return runtime.build_t10_2_2_collection_report(
        recorder=recorder, checkpoint=checkpoint, collection_report=collection_report
    ), checkpoint


def _populate_recorder() -> runtime.CollectionTimingRecorder:
    recorder = runtime.CollectionTimingRecorder()
    lane = _lane()
    for reset_index, base in ((0, 0.0), (1, 100.0)):
        work = _FakeWork(lane, reset_index)
        recorder.note_reset_started(work, base)
        recorder.observe(
            work,
            {
                "kind": "reset_operation",
                "payload": {"operation": "open", "stage": "started"},
            },
            base + 1.0,
        )
        recorder.observe(
            work,
            {
                "kind": "reset_operation",
                "payload": {"operation": "reset", "stage": "finished"},
            },
            base + 2.0,
        )
        recorder.observe(work, {"kind": "action_intent"}, base + 3.0)
        recorder.observe(work, {"kind": "physical_event"}, base + 4.0)
        recorder.note_reset_finished(
            work, base + 60.0, status="COMPLETE", stop_reason=None
        )
    return recorder


def test_sidecar_includes_controller_activity_and_eligibility() -> None:
    report, _ = _build_report(_populate_recorder())
    # Point 5: controller-activity section present.
    assert (
        report["controller_activity"]["format_version"]
        == protocol.CONTROLLER_ACTIVITY_FORMAT_VERSION
    )
    # Point 6: event-eligibility ventilation present and internally consistent.
    eligibility = report["event_eligibility"]
    assert (
        eligibility["complete_reset_events"] + eligibility["incomplete_reset_events"]
        == eligibility["sealed_events_total"]
    )


def test_build_implementation_provenance_pins_bytes_and_schedule() -> None:
    prov = runtime.build_implementation_provenance()
    assert prov["format_version"] == runtime.PROVENANCE_FORMAT_VERSION
    # Every implementation file that exists is hashed.
    assert "theory/sage_t/t10_2_2_runtime.py" in prov["implementation_sha256"]
    assert "theory/sage_t/gauge_inference_v10_2.py" in prov["implementation_sha256"]
    # The interleaved/reserved schedule is made visible (point 7).
    assert prov["scheduler"]["interleaved"] is True
    assert prov["scheduler"]["reserved_confirmation_capacity"] >= 1
    # Discovery-then-confirmation is no longer opaque: order straddles splits.
    assert len(prov["scheduler"]["scheduled_order_lane_ids"]) == len(
        protocol.source_lane_registry()
    )


def test_sidecar_report_composes_all_items() -> None:
    report, checkpoint = _build_report(_populate_recorder())
    assert report["format_version"] == runtime.T10_2_2_COLLECTION_REPORT_FORMAT_VERSION
    # Item 1: bound to the exact checkpoint revision + checksum.
    assert report["checkpoint_binding"]["checkpoint_revision"] == 4
    assert (
        report["checkpoint_binding"]["checkpoint_checksum"]
        == checkpoint["checkpoint_checksum"]
    )
    # Item 2: startup latency measured, not folded into interaction time.
    assert report["phase_timing"]["committed_lane_count"] == 1
    assert report["phase_timing"]["max_startup_latency_seconds"] == pytest.approx(4.0)
    # Item 3/4: readiness gates present, first intent within budget.
    assert len(report["readiness_gates"]) == 2
    assert report["first_intent"]["any_timeout"] is False
    # Item 5: funnel fully accounted, rejection reasons registered.
    assert report["evidence_funnel"]["fully_accounted"] is True
    assert report["evidence_funnel"]["rejections"] == {"cooperative_reset_deadline": 4}
    # Item 6: families separated from grounded instances.
    assert report["schema_evidence"]["canonical_families"] == ["move:1"]
    assert report["schema_evidence"]["grounded_instance_count"] == 2
    # Item 7: green induction canary.
    assert report["induction_canary"]["passed"] is True
    # Signed.
    unsigned = {k: v for k, v in report.items() if k != "report_checksum"}
    assert report["report_checksum"] == protocol.canonical_sha256(unsigned)


def test_sidecar_flags_first_intent_timeout() -> None:
    recorder = runtime.CollectionTimingRecorder()
    work = _FakeWork(_lane(), 0)
    recorder.note_reset_started(work, 0.0)
    recorder.observe(
        work,
        {
            "kind": "reset_operation",
            "payload": {"operation": "open", "stage": "started"},
        },
        1.0,
    )
    recorder.observe(
        work,
        {
            "kind": "reset_operation",
            "payload": {"operation": "reset", "stage": "finished"},
        },
        2.0,
    )
    # First intent arrives well past the readiness-relative budget.
    recorder.observe(
        work, {"kind": "action_intent"}, 2.0 + runtime.FIRST_INTENT_BUDGET_SECONDS + 5.0
    )
    recorder.observe(
        work,
        {"kind": "physical_event"},
        2.0 + runtime.FIRST_INTENT_BUDGET_SECONDS + 6.0,
    )
    recorder.note_reset_finished(work, 120.0, status="COMPLETE", stop_reason=None)

    report, _ = _build_report(recorder)
    statuses = [row["status"] for row in report["first_intent"]["per_reset"]]
    assert "first_intent_timeout" in statuses
    assert report["first_intent"]["any_timeout"] is True


@pytest.mark.parametrize(
    ("confirmation_status", "expected_passed"),
    (("COMPLETE", True), ("ABORTED", False)),
)
def test_smoke_gate_requires_complete_confirmation(
    confirmation_status: str, expected_passed: bool
) -> None:
    recorder, checkpoint, collection_report, cursor = _smoke_inputs(
        confirmation_status=confirmation_status
    )
    report = runtime.build_t10_2_2_collection_report(
        recorder=recorder,
        checkpoint=checkpoint,
        collection_report=collection_report,
        mode="smoke",
        cursor=cursor,
    )
    gate = report["smoke_gate"]
    assert gate["passed"] is expected_passed
    assert gate["confirmation_complete_with_evidence"] is expected_passed
    assert gate["confirmation_controller_coverage"] is True
    assert gate["confirmation_controllers"] == [
        "capacity_matched_independent",
        "learned",
    ]
    if confirmation_status == "ABORTED":
        assert gate["worker_error_kinds"] == ["RuntimeError"]


def test_write_sidecar_to_disk(tmp_path: Path) -> None:
    recorder = _populate_recorder()
    manifest = "m" * 64
    resets = [
        _reset_report(
            reset_index=0,
            issued=64,
            sealed=64,
            unresolved=0,
            stop_reason="registered_collection_deadline",
        ),
    ]
    checkpoint = _self_authenticating_checkpoint(
        manifest_checksum=manifest, revision=2, resets=resets
    )
    collection_report = protocol.signed_payload(
        {
            "format_version": "sage-t10.2.1-protocol-v1",
            "phase": "collect",
            "manifest_checksum": manifest,
            "durability": {"checkpoint_checksum": checkpoint["checkpoint_checksum"]},
        },
        checksum_key="report_checksum",
    )
    (tmp_path / runtime.COLLECTION_REPORT_FILENAME).write_text(
        protocol.canonical_json(collection_report) + "\n", encoding="utf-8"
    )
    (tmp_path / runtime.CHECKPOINT_FILENAME).write_text(
        protocol.canonical_json(checkpoint) + "\n", encoding="utf-8"
    )
    report = runtime.write_t10_2_2_collection_report(
        recorder=recorder, output_dir=tmp_path
    )
    written = json.loads(
        (tmp_path / runtime.T10_2_2_COLLECTION_REPORT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert written == report
    assert written["checkpoint_binding"]["checkpoint_revision"] == 2

    original_bytes = (
        tmp_path / runtime.T10_2_2_COLLECTION_REPORT_FILENAME
    ).read_bytes()
    resumed = runtime.write_t10_2_2_collection_report(
        recorder=runtime.CollectionTimingRecorder(), output_dir=tmp_path
    )
    assert resumed == report
    assert (
        tmp_path / runtime.T10_2_2_COLLECTION_REPORT_FILENAME
    ).read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# IncrementalCollectionState: the O(N^2) -> O(N) fix.
# ---------------------------------------------------------------------------
def _reset_records():
    # Two discovery resets (complete) + one confirmation (complete, with an
    # unresolved intent) + one discovery reset aborted (its events must NOT
    # enter the discovery cache).
    return [
        {
            "work_id": "d0",
            "split": "discovery",
            "status": "COMPLETE",
            "issued": 24,
            "sealed": 24,
            "unresolved": 0,
            "posterior_updates": 0,
            "events": [{"event_id": "e0"}, {"event_id": "e1"}],
        },
        {
            "work_id": "d1",
            "split": "discovery",
            "status": "COMPLETE",
            "issued": 24,
            "sealed": 24,
            "unresolved": 0,
            "posterior_updates": 0,
            "events": [{"event_id": "e2"}],
        },
        {
            "work_id": "c0",
            "split": "leave_one_game_out_confirmation",
            "status": "COMPLETE",
            "issued": 24,
            "sealed": 20,
            "unresolved": 4,
            "posterior_updates": 20,
            "events": [{"event_id": "e3"}],
        },
        {
            "work_id": "d2",
            "split": "discovery",
            "status": "ABORTED",
            "issued": 23,
            "sealed": 23,
            "unresolved": 0,
            "posterior_updates": 0,
            "events": [{"event_id": "e9"}],
        },
    ]


def test_incremental_state_matches_full_recompute() -> None:
    state = runtime.IncrementalCollectionState()
    for record in _reset_records():
        state.record_reset(**record)

    acc = state.accounting()
    assert acc.authorized_intents == 24 + 24 + 24 + 23
    assert acc.sealed_events == 24 + 24 + 20 + 23
    assert acc.unresolved_intents == 4
    assert acc.equation_holds is True

    # Discovery cache: only COMPLETE discovery events (e0,e1,e2); NOT the
    # confirmation event (e3) nor the aborted discovery event (e9).
    ids = {e["event_id"] for e in state.discovery_events()}
    assert ids == {"e0", "e1", "e2"}
    assert state.completed_reset_count == 3


def test_incremental_state_record_is_idempotent() -> None:
    state = runtime.IncrementalCollectionState()
    record = _reset_records()[0]
    state.record_reset(**record)
    state.record_reset(**record)  # replay after a resume must not double-count
    assert state.accounting().sealed_events == 24
    assert len(state.discovery_events()) == 2
    assert state.recorded_reset_count == 1


def test_running_accounting_equation() -> None:
    acc = runtime.RunningAccounting()
    acc.add_reset(issued=64, sealed=60, unresolved=4, posterior_updates=60)
    assert acc.equation_holds is True
    acc.add_reset(issued=10, sealed=10, unresolved=1, posterior_updates=0)
    # 74 != 70 + 5 -> broken, must be detectable.
    assert acc.equation_holds is False


def test_smoke_execution_bindings_are_donor_safe_and_restore() -> None:
    parent_registry = runtime._t10_2_1_runtime.source_lane_registry
    parent_journal = runtime._t10_2_1_runtime.DurableCollectionJournal
    with runtime.execution_bindings(
        mode="smoke", artifact_root=protocol.DEFAULT_SMOKE_OUTPUT_DIR
    ) as lanes:
        assert len(lanes) == 3
        assert [lane.split for lane in lanes] == [
            "discovery",
            "discovery",
            "leave_one_game_out_confirmation",
        ]
        held_out = lanes[-1].game_id
        assert {lane.game_id for lane in lanes[:2]} == (
            set(protocol.SOURCE_GAMES) - {held_out}
        )
        assert all(
            len(runtime._t10_2_1_runtime.reset_work_specs(lane)) == 2 for lane in lanes
        )
        confirmation = runtime._t10_2_1_runtime.reset_work_specs(lanes[-1])
        assert [work.controller for work in confirmation] == [
            "capacity_matched_independent",
            "learned",
        ]
        assert runtime._t10_2_1_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS == 384
        assert (
            runtime._t10_2_1_runtime.DurableCollectionJournal
            is runtime.IncrementalDurableCollectionJournal
        )
    assert runtime._t10_2_1_runtime.source_lane_registry is parent_registry
    assert runtime._t10_2_1_runtime.DurableCollectionJournal is parent_journal


def test_incremental_journal_scans_once_and_resumes_from_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "smoke"
    with runtime.execution_bindings(mode="smoke", artifact_root=artifact_root) as lanes:
        journal = runtime.IncrementalDurableCollectionJournal(
            artifact_root / "source_collection_journal",
            manifest_checksum="a" * 64,
        )
        assert journal.full_history_scan_count == 1
        assert journal.accounting().equation_holds is True
        first = journal.reconstruct_checkpoint(
            cumulative_active_seconds=1.0,
            open_lane=lanes[0],
            open_lane_elapsed_seconds=0.5,
        )
        assert first.open_lane_id == lanes[0].lane_id
        assert journal.cursor_path.is_file()
        assert journal.checkpoint_path.is_file()

        def forbidden_scan(*args, **kwargs):
            raise AssertionError("hot loop attempted a whole-history scan")

        frozen_base = runtime.IncrementalDurableCollectionJournal.__mro__[1]
        monkeypatch.setattr(frozen_base, "accounting", forbidden_scan)
        monkeypatch.setattr(frozen_base, "lane_reports", forbidden_scan)
        assert journal.accounting().authorized_intent_count == 0
        assert journal.lane_reports() == ()
        assert journal.completed_discovery_events() == ()
        closed = journal.reconstruct_checkpoint(
            cumulative_active_seconds=2.0,
            close_open_lane=True,
        )
        assert closed.open_lane_id is None
        assert journal.full_history_scan_count == 1

    # Restore the base methods before constructing the resumed instance; its
    # one permitted resume scan must execute.
    monkeypatch.undo()
    with runtime.execution_bindings(mode="smoke", artifact_root=artifact_root):
        resumed = runtime.IncrementalDurableCollectionJournal(
            artifact_root / "source_collection_journal",
            manifest_checksum="a" * 64,
        )
        assert resumed.full_history_scan_count == 1
        assert resumed.accounting().equation_holds is True
        checkpoint = resumed.load_checkpoint()
        assert checkpoint is not None
        assert checkpoint.open_lane_id is None
        assert checkpoint.cumulative_active_seconds == pytest.approx(2.0)


def test_incremental_cursor_tampering_is_refused(tmp_path: Path) -> None:
    artifact_root = tmp_path / "smoke"
    with runtime.execution_bindings(mode="smoke", artifact_root=artifact_root) as lanes:
        journal = runtime.IncrementalDurableCollectionJournal(
            artifact_root / "source_collection_journal",
            manifest_checksum="b" * 64,
        )
        journal.reconstruct_checkpoint(
            cumulative_active_seconds=1.0,
            open_lane=lanes[0],
            open_lane_elapsed_seconds=0.5,
        )
        payload = json.loads(journal.cursor_path.read_text(encoding="utf-8"))
        payload["full_history_scan_count"] = 2
        journal.cursor_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(runtime._t10_2_1_runtime.JournalIntegrityError):
            runtime.IncrementalDurableCollectionJournal(
                artifact_root / "source_collection_journal",
                manifest_checksum="b" * 64,
            )
