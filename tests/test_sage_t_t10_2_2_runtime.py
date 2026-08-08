from __future__ import annotations

import json
from pathlib import Path
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

    def supervise(self, process: Any, *, message_handler, **kwargs: Any) -> _FakeOutcome:
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
        work, {"kind": "reset_operation", "payload": {"operation": "open", "stage": "started"}}, 101.0
    )
    recorder.observe(
        work, {"kind": "reset_operation", "payload": {"operation": "reset", "stage": "finished"}}, 102.0
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
        {"kind": "reset_operation", "payload": {"operation": "open", "stage": "started"}},
        {"kind": "reset_operation", "payload": {"operation": "reset", "stage": "finished"}},
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
    assert handled == ["reset_operation", "reset_operation", "action_intent", "physical_event"]
    assert inner.acks == [{"ack": True}] * 4
    timing = recorder.reset_timings()[0]
    assert timing.controller_ready_at is not None
    assert timing.environment_ready_at is not None
    assert timing.first_intent_at is not None
    assert timing.first_committed_at is not None
    assert timing.status == "COMPLETE"


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


def _reset_report(*, reset_index: int, issued: int, sealed: int, unresolved: int, stop_reason: str):
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
        _reset_report(reset_index=0, issued=64, sealed=64, unresolved=0,
                      stop_reason="registered_collection_deadline"),
        _reset_report(reset_index=1, issued=64, sealed=60, unresolved=4,
                      stop_reason="cooperative_reset_deadline"),
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
        recorder.observe(work, {"kind": "reset_operation", "payload": {"operation": "open", "stage": "started"}}, base + 1.0)
        recorder.observe(work, {"kind": "reset_operation", "payload": {"operation": "reset", "stage": "finished"}}, base + 2.0)
        recorder.observe(work, {"kind": "action_intent"}, base + 3.0)
        recorder.observe(work, {"kind": "physical_event"}, base + 4.0)
        recorder.note_reset_finished(work, base + 60.0, status="COMPLETE", stop_reason=None)
    return recorder


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
    recorder.observe(work, {"kind": "reset_operation", "payload": {"operation": "open", "stage": "started"}}, 1.0)
    recorder.observe(work, {"kind": "reset_operation", "payload": {"operation": "reset", "stage": "finished"}}, 2.0)
    # First intent arrives well past the readiness-relative budget.
    recorder.observe(work, {"kind": "action_intent"}, 2.0 + runtime.FIRST_INTENT_BUDGET_SECONDS + 5.0)
    recorder.observe(work, {"kind": "physical_event"}, 2.0 + runtime.FIRST_INTENT_BUDGET_SECONDS + 6.0)
    recorder.note_reset_finished(work, 120.0, status="COMPLETE", stop_reason=None)

    report, _ = _build_report(recorder)
    statuses = [row["status"] for row in report["first_intent"]["per_reset"]]
    assert "first_intent_timeout" in statuses
    assert report["first_intent"]["any_timeout"] is True


def test_write_sidecar_to_disk(tmp_path: Path) -> None:
    recorder = _populate_recorder()
    manifest = "m" * 64
    resets = [
        _reset_report(reset_index=0, issued=64, sealed=64, unresolved=0,
                      stop_reason="registered_collection_deadline"),
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
