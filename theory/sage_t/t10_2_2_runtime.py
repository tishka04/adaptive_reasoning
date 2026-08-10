"""Action-budget, incremental durable runtime for SAGE.T10.2.2.

The scientific kernel and append-only T10.2.1 journal records are reused without
editing their source bytes.  T10.2.2 owns a new artifact namespace, an
action-budget watchdog, donor-safe lane scheduling, an incremental journal view,
and a compact authenticated resume cursor.

The readiness observer remains parent-side and does not alter the scientific
child or the immutable intent/event schemas.  The T10.2.1 factory exposes an
injectable ``watchdog`` (see ``T10_2_1SourceFactory.__init__``) that stays on the
parent and is deliberately dropped when the factory is pickled to the child.
``LaneTimingWatchdog`` uses that seam to wrap the parent-side message handler and
record, per reset:

* controller readiness  (the moment the frozen worker begins ``open`` -- the
  posterior/controller is already fitted at that point);
* environment readiness (the moment ``reset`` finishes -- the environment is
  opened and reset);
* the first authorized intent (first ``action_intent`` message); and
* the first committed transition (first ``physical_event`` the parent seals).

From those, T10.2.2 derives, in a *sidecar* report that binds the frozen
checkpoint's exact revision and checksum:

* item 2 -- lane-start-to-first-committed startup latency, kept out of the
  interaction budget;
* item 3 -- a readiness-relative first-intent deadline;
* item 4 -- an interaction deadline anchored at ``max(controller, environment)``
  readiness;
* items 1/5/6/7 -- the checkpoint binding, evidence funnel, schema-family
  partition, and induction canary from :mod:`theory.sage_t.t10_2_2_protocol`.

The partial T10.2.1 artifact namespace is never opened for writing.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _t10_2_1
from . import t10_2_1_runtime as _t10_2_1_runtime
from . import t10_2_2_protocol as _protocol

FORMAT_VERSION = "sage-t10.2.2-runtime-v1"
T10_2_2_COLLECTION_REPORT_FORMAT_VERSION = "sage-t10.2.2-collection-report-v1"
T10_2_2_COLLECTION_REPORT_FILENAME = "t10_2_2_collection_report.json"
PROVENANCE_FORMAT_VERSION = "sage-t10.2.2-implementation-provenance-v1"
CURSOR_FORMAT_VERSION = "sage-t10.2.2-compact-cursor-v1"
CURSOR_FILENAME = "t10_2_2_collection_cursor.json"

# The implementations whose bytes define the T10.2.2 experimental identity.
# Two runs sharing a manifest but differing here are NOT the same experiment.
IMPLEMENTATION_FILES = (
    "theory/sage_t/t10_2_2_protocol.py",
    "theory/sage_t/t10_2_2_runtime.py",
    "theory/sage_t/t10_2_1_protocol.py",
    "theory/sage_t/t10_2_1_runtime.py",
    "theory/sage_t/gauge_inference_v10_2.py",
    "theory/sage_t/factorized_posterior_v10_2.py",
    "theory/sage_t/t10_2_runtime.py",
    "theory/sage_t/t10_2_protocol.py",
)

# The interaction budget is charged only from joint readiness (item 4); it
# mirrors the frozen cooperative reset budget so the science is unchanged.
INTERACTION_BUDGET_SECONDS = _t10_2_1_runtime.RESET_COOPERATIVE_SECONDS
# The first authorized intent must arrive within this readiness-relative window
# (item 3).  It is strictly smaller than the interaction budget.
FIRST_INTENT_BUDGET_SECONDS = 30.0

# --- Budget-by-actions policy (chosen T10.2.2 acquisition change) ------------
# The experimental budget is the number of physical actions per reset; the
# wall-clock is demoted to a WIDE, finite liveness watchdog whose only job is to
# kill a genuinely stuck process.  Applied through the injectable ``watchdog``
# seam, so no byte of the frozen T10.2.1 kernel is edited.  The canonical values
# live in the protocol layer (frozen into the T10.2.2 manifest).
RESET_LIVENESS_WALL_SECONDS = _protocol.RESET_LIVENESS_WALL_SECONDS
RESET_LIVENESS_HARD_GRACE_SECONDS = _protocol.RESET_LIVENESS_HARD_GRACE_SECONDS
LANE_LIVENESS_WALL_SECONDS = _protocol.LANE_LIVENESS_WALL_SECONDS
COLLECTION_LIVENESS_WALL_SECONDS = _protocol.COLLECTION_LIVENESS_WALL_SECONDS
COLLECTION_STOP_NEW_ACTIONS_SECONDS = _protocol.COLLECTION_STOP_NEW_ACTIONS_SECONDS

CHECKPOINT_FILENAME = _t10_2_1_runtime.CHECKPOINT_FILENAME
COLLECTION_REPORT_FILENAME = _t10_2_1_runtime.COLLECTION_REPORT_FILENAME

canonical_sha256 = _t10_2_1.canonical_sha256
signed_payload = _t10_2_1.signed_payload
write_compact_json = _t10_2_1.write_compact_json
_read_signed_json = _t10_2_1._read_signed_json
ManifestDriftError = _t10_2_1.ManifestDriftError
ProtocolError = _t10_2_1.ProtocolError


# ---------------------------------------------------------------------------
# Per-reset timing record captured by the parent observer.
# ---------------------------------------------------------------------------
@dataclass
class ResetTiming:
    work_id: str
    lane: dict[str, Any]
    reset_index: int
    reset_started_at: float | None = None
    controller_ready_at: float | None = None
    environment_ready_at: float | None = None
    first_intent_at: float | None = None
    first_committed_at: float | None = None
    finished_at: float | None = None
    status: str | None = None
    stop_reason: str | None = None
    error_kind: str | None = None
    outcome_payload_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "lane": dict(self.lane),
            "reset_index": self.reset_index,
            "reset_started_at": self.reset_started_at,
            "controller_ready_at": self.controller_ready_at,
            "environment_ready_at": self.environment_ready_at,
            "first_intent_at": self.first_intent_at,
            "first_committed_at": self.first_committed_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "error_kind": self.error_kind,
            "outcome_payload_sha256": self.outcome_payload_sha256,
        }


class CollectionTimingRecorder:
    """Accumulates parent-observed timings across every supervised reset."""

    def __init__(self) -> None:
        self._resets: dict[str, ResetTiming] = {}
        self._order: list[str] = []

    def _entry(self, work: Any) -> ResetTiming:
        work_id = str(work.work_id)
        if work_id not in self._resets:
            self._resets[work_id] = ResetTiming(
                work_id=work_id,
                lane=dict(work.lane.to_dict()),
                reset_index=int(work.reset_index),
            )
            self._order.append(work_id)
        return self._resets[work_id]

    def note_reset_started(self, work: Any, at: float) -> None:
        entry = self._entry(work)
        if entry.reset_started_at is None:
            entry.reset_started_at = float(at)

    def observe(self, work: Any, message: Mapping[str, Any], at: float) -> None:
        """Record a single parent-observed worker message timestamp."""

        entry = self._entry(work)
        kind = str(message.get("kind", ""))
        if kind == "reset_operation":
            payload = message.get("payload") or {}
            operation = str(payload.get("operation", ""))
            stage = str(payload.get("stage", ""))
            if (
                operation == "open"
                and stage == "started"
                and entry.controller_ready_at is None
            ):
                # The frozen worker fits the controller/posterior before ``open``.
                entry.controller_ready_at = float(at)
            elif operation == "reset" and stage == "finished":
                entry.environment_ready_at = float(at)
        elif kind == "action_intent" and entry.first_intent_at is None:
            entry.first_intent_at = float(at)
        elif kind == "physical_event" and entry.first_committed_at is None:
            entry.first_committed_at = float(at)

    def note_reset_finished(
        self,
        work: Any,
        at: float,
        *,
        status: str,
        stop_reason: str | None,
        error_kind: str | None = None,
        outcome_payload: Mapping[str, Any] | None = None,
    ) -> None:
        entry = self._entry(work)
        entry.finished_at = float(at)
        entry.status = status
        entry.stop_reason = stop_reason
        entry.error_kind = error_kind
        entry.outcome_payload_sha256 = (
            None if outcome_payload is None else canonical_sha256(outcome_payload)
        )

    def reset_timings(self) -> tuple[ResetTiming, ...]:
        return tuple(self._resets[work_id] for work_id in self._order)

    def lane_windows(self) -> list[dict[str, Any]]:
        """Aggregate reset timings into one start/first-committed/finish window."""

        lanes: dict[str, dict[str, Any]] = {}
        for timing in self.reset_timings():
            lane_id = str(timing.lane.get("lane_id"))
            window = lanes.setdefault(
                lane_id,
                {
                    "lane": dict(timing.lane),
                    "lane_started_seconds": None,
                    "first_committed_transition_seconds": None,
                    "lane_finished_seconds": None,
                },
            )
            window["lane_started_seconds"] = _min_opt(
                window["lane_started_seconds"], timing.reset_started_at
            )
            window["first_committed_transition_seconds"] = _min_opt(
                window["first_committed_transition_seconds"], timing.first_committed_at
            )
            window["lane_finished_seconds"] = _max_opt(
                window["lane_finished_seconds"], timing.finished_at
            )
        return list(lanes.values())


def _min_opt(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return float(candidate)
    return min(current, float(candidate))


def _max_opt(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return float(candidate)
    return max(current, float(candidate))


# ---------------------------------------------------------------------------
# Parent-side supervisor that observes and delegates.
# ---------------------------------------------------------------------------
class LaneTimingWatchdog:
    """Wraps a frozen ``ProcessResetWatchdog`` to observe parent messages.

    It never changes supervision semantics: every message is forwarded to the
    frozen handler unchanged and the frozen watchdog owns all timeout authority.
    """

    def __init__(
        self,
        *,
        recorder: CollectionTimingRecorder,
        clock: Callable[[], float] = time.perf_counter,
        inner: Any | None = None,
    ) -> None:
        self._recorder = recorder
        self._clock = clock
        self._inner = inner or _t10_2_1_runtime.ProcessResetWatchdog(clock=clock)

    def supervise(
        self,
        process: Any,
        *,
        work: Any,
        cancel_event: Any,
        outbound_queue: Any,
        inbound_queue: Any,
        message_handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        cooperative_seconds: float,
        hard_seconds: float,
        started_at: float,
    ) -> Any:
        self._recorder.note_reset_started(work, started_at)

        def observing_handler(
            message: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            # Observe first so a message that trips the intent deadline (and
            # raises inside the frozen handler) is still recorded as attempted.
            self._recorder.observe(work, message, float(self._clock()))
            return message_handler(message)

        status = "EXCEPTION"
        stop_reason: str | None = None
        error_kind: str | None = None
        outcome_payload: Mapping[str, Any] | None = None
        try:
            outcome = self._inner.supervise(
                process,
                work=work,
                cancel_event=cancel_event,
                outbound_queue=outbound_queue,
                inbound_queue=inbound_queue,
                message_handler=observing_handler,
                cooperative_seconds=cooperative_seconds,
                hard_seconds=hard_seconds,
                started_at=started_at,
            )
            status = str(getattr(outcome, "status", "UNKNOWN"))
            payload = getattr(outcome, "payload", None) or {}
            stop_reason = payload.get("stop_reason")
            error_kind = getattr(outcome, "error_kind", None)
            outcome_payload = payload
            return outcome
        finally:
            self._recorder.note_reset_finished(
                work,
                float(self._clock()),
                status=status,
                stop_reason=stop_reason,
                error_kind=error_kind,
                outcome_payload=outcome_payload,
            )


class ActionBudgetWatchdog:
    """Reset-level supervision under a budget-by-actions policy.

    The frozen ``run_reset`` computes a 55/60 s cooperative/hard deadline from the
    frozen work spec and builds a ``_ParentJournalMessageHandler`` whose intent
    deadline is anchored to that 55 s.  This watchdog, injected through the
    factory's ``watchdog`` seam, demotes the wall-clock to a WIDE finite liveness
    bound so the reset runs to its physical-action budget instead of being cut at
    55 s.  It does so without editing any frozen byte:

    * it ignores the 55/60 s it is handed and supervises with a wide liveness
      cooperative/hard pair; and
    * it widens the frozen handler's ``_intent_deadline`` to the same liveness
      bound so authorized intents are not refused before the action budget.

    Timing observation (items 2-4) is preserved by composing the timing
    watchdog.

    NOTE (honest scope): this changes the RESET budget only.  The LANE (250 s)
    and COLLECTION (5 400 s) wall caps and the manifest ``source_plan`` still live
    in the frozen lane/collection loop, so a *complete* action-budget acquisition
    additionally needs a T10.2.2 manifest + loop that registers the action-budget
    plan (next increment).
    """

    def __init__(
        self,
        *,
        recorder: CollectionTimingRecorder,
        clock: Callable[[], float] = time.perf_counter,
        reset_liveness_seconds: float = RESET_LIVENESS_WALL_SECONDS,
        hard_grace_seconds: float = RESET_LIVENESS_HARD_GRACE_SECONDS,
        inner: Any | None = None,
    ) -> None:
        if reset_liveness_seconds <= 0.0 or hard_grace_seconds < 0.0:
            raise ValueError("liveness bounds must be positive and finite")
        self._reset_liveness_seconds = float(reset_liveness_seconds)
        self._hard_grace_seconds = float(hard_grace_seconds)
        self._timing = inner or LaneTimingWatchdog(recorder=recorder, clock=clock)

    def supervise(
        self,
        process: Any,
        *,
        work: Any,
        cancel_event: Any,
        outbound_queue: Any,
        inbound_queue: Any,
        message_handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        cooperative_seconds: float,
        hard_seconds: float,
        started_at: float,
    ) -> Any:
        cooperative = self._reset_liveness_seconds
        hard = self._reset_liveness_seconds + self._hard_grace_seconds
        # Extend the frozen handler's intent deadline to the liveness bound so
        # actions are not refused before the physical-action budget is reached.
        if hasattr(message_handler, "_intent_deadline"):
            message_handler._intent_deadline = float(started_at) + cooperative
        return self._timing.supervise(
            process,
            work=work,
            cancel_event=cancel_event,
            outbound_queue=outbound_queue,
            inbound_queue=inbound_queue,
            message_handler=message_handler,
            cooperative_seconds=cooperative,
            hard_seconds=hard,
            started_at=started_at,
        )


@dataclass
class RunningAccounting:
    """O(1)-updatable running totals replacing per-reset ``journal.accounting()``."""

    authorized_intents: int = 0
    sealed_events: int = 0
    unresolved_intents: int = 0
    posterior_updates: int = 0

    def add_reset(
        self, *, issued: int, sealed: int, unresolved: int, posterior_updates: int
    ) -> None:
        self.authorized_intents += int(issued)
        self.sealed_events += int(sealed)
        self.unresolved_intents += int(unresolved)
        self.posterior_updates += int(posterior_updates)

    @property
    def equation_holds(self) -> bool:
        return self.authorized_intents == self.sealed_events + self.unresolved_intents

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized_intents": self.authorized_intents,
            "sealed_events": self.sealed_events,
            "unresolved_intents": self.unresolved_intents,
            "posterior_updates": self.posterior_updates,
            "equation_holds": self.equation_holds,
        }


class IncrementalCollectionState:
    """The O(N^2)->O(N) fix at the heart of the T10.2.2 acquisition loop.

    The frozen ``collect_source`` recomputes three whole-history aggregates
    *before every reset* -- ``accounting()`` (~2.2 s), ``lane_reports()``/
    ``reconstruct_checkpoint`` (~1.9 s) and ``_completed_discovery_events``
    (~2.7 s on a partial journal, growing).  That is what makes the collection
    O(N^2) and starves late lanes.

    This state object reconstructs those aggregates from the durable journal
    exactly ONCE (at start/resume, O(N)), then maintains them INCREMENTALLY as
    each reset commits (O(1) per reset).  It never re-scans the full history, so
    a T10.2.2 loop built on it is O(N) total.  The durable per-reset/per-event
    journal (the source of truth) is unchanged and still authoritative on crash.
    """

    def __init__(self) -> None:
        self._accounting = RunningAccounting()
        self._discovery_events: list[dict[str, Any]] = []
        self._discovery_event_ids: set[str] = set()
        self._recorded_reset_ids: set[str] = set()
        self._completed_reset_ids: set[str] = set()

    def record_reset(
        self,
        *,
        work_id: str,
        split: str,
        status: str,
        issued: int,
        sealed: int,
        unresolved: int,
        posterior_updates: int,
        events: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Fold one reset into the running aggregates in O(1) (idempotent)."""

        work_id = str(work_id)
        if work_id in self._recorded_reset_ids:
            return
        self._recorded_reset_ids.add(work_id)
        self._accounting.add_reset(
            issued=issued,
            sealed=sealed,
            unresolved=unresolved,
            posterior_updates=posterior_updates,
        )
        if str(status) == "COMPLETE":
            self._completed_reset_ids.add(work_id)
            if str(split) == "discovery":
                for event in events:
                    event_id = str(event.get("event_id", ""))
                    if event_id and event_id in self._discovery_event_ids:
                        continue
                    if event_id:
                        self._discovery_event_ids.add(event_id)
                    self._discovery_events.append(dict(event))

    def accounting(self) -> RunningAccounting:
        return self._accounting

    def discovery_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._discovery_events)

    @property
    def completed_reset_count(self) -> int:
        return len(self._completed_reset_ids)

    @property
    def recorded_reset_count(self) -> int:
        return len(self._recorded_reset_ids)


class IncrementalDurableCollectionJournal(_t10_2_1_runtime.DurableCollectionJournal):
    """T10.2.1 journal with one resume scan and O(1) running aggregates.

    Immutable record validation is delegated to the frozen journal.  The cache
    is reconstructed once at open, updated only after successful durable writes,
    and never used to relax a checksum, topology, or action-equation gate.
    """

    def __init__(self, root: str | Path, *, manifest_checksum: str) -> None:
        super().__init__(root, manifest_checksum=manifest_checksum)
        self._full_history_scan_count = 1
        initial = super().accounting()
        if initial.unknown_intent_count or not initial.equation_holds:
            raise _t10_2_1_runtime.JournalIntegrityError(
                "journal failed the one-time incremental-state reconstruction"
            )
        self._authorized = initial.authorized_intent_count
        self._sealed = initial.sealed_event_count
        self._unresolved = initial.explicitly_unresolved_intent_count
        self._updates = initial.posterior_update_count
        self._unknown = initial.unknown_intent_count

        lane_reports = super().lane_reports()
        self._lane_reports_by_id = {
            report.lane.lane_id: report for report in lane_reports
        }
        self._reset_reports_by_id: dict[str, Any] = {}
        self._reset_active_seconds = 0.0
        self._discovery_events: list[dict[str, Any]] = []
        self._discovery_event_ids: set[str] = set()
        for lane in _t10_2_1_runtime.source_lane_registry():
            for work in _t10_2_1_runtime.reset_work_specs(lane):
                report = super().read_reset_report(work)
                if report is None:
                    continue
                self._reset_reports_by_id[work.work_id] = report
                self._reset_active_seconds += float(report.elapsed_seconds)
                if report.status == "COMPLETE" and lane.split == "discovery":
                    for event in super().events_for_reset(work):
                        event_id = str(event.get("event_id", ""))
                        if not event_id or event_id in self._discovery_event_ids:
                            continue
                        self._discovery_event_ids.add(event_id)
                        self._discovery_events.append(dict(event))

        self._lane_reports_checksum = canonical_sha256(
            [report.to_dict() for report in self.lane_reports()]
        )
        self._persisted_checkpoint = super().load_checkpoint()
        self._cursor_state = self._load_cursor_state()
        self._current_checkpoint = self._checkpoint_from_state(
            self._cursor_state,
            fallback=self._persisted_checkpoint,
        )

    @property
    def cursor_path(self) -> Path:
        return self.root.parent / CURSOR_FILENAME

    @property
    def full_history_scan_count(self) -> int:
        return self._full_history_scan_count

    def accounting(self) -> _t10_2_1_runtime.JournalAccounting:
        return _t10_2_1_runtime.JournalAccounting(
            authorized_intent_count=self._authorized,
            sealed_event_count=self._sealed,
            explicitly_unresolved_intent_count=self._unresolved,
            unknown_intent_count=self._unknown,
            posterior_update_count=self._updates,
        )

    def lane_reports(self) -> tuple[Any, ...]:
        return tuple(
            self._lane_reports_by_id[lane.lane_id]
            for lane in _t10_2_1_runtime.source_lane_registry()
            if lane.lane_id in self._lane_reports_by_id
        )

    def completed_discovery_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._discovery_events)

    def record_intent(self, intent: Any) -> bool:
        written = super().record_intent(intent)
        if written:
            self._authorized += 1
        return written

    def record_physical_event(self, *, intent: Any, event: Mapping[str, Any]) -> Any:
        existed = self._event_path(intent).is_file()
        receipt = super().record_physical_event(intent=intent, event=event)
        if not existed:
            self._sealed += 1
        return receipt

    def record_posterior_update(self, *, intent: Any, receipt: Any) -> bool:
        written = super().record_posterior_update(intent=intent, receipt=receipt)
        if written:
            self._updates += 1
        return written

    def record_unresolved_intent(self, *, intent: Any, reason: Any) -> Any:
        existed = self._unresolved_path(intent).is_file()
        receipt = super().record_unresolved_intent(intent=intent, reason=reason)
        if not existed:
            self._unresolved += 1
        return receipt

    def write_reset_report(self, report: Any) -> bool:
        path = self._reset_root(report.work) / "reset_report.json"
        existed = path.is_file()
        written = super().write_reset_report(report)
        if written and not existed:
            self._reset_reports_by_id[report.work.work_id] = report
            self._reset_active_seconds += float(report.elapsed_seconds)
            if report.status == "COMPLETE" and report.work.lane.split == "discovery":
                for event in super().events_for_reset(report.work):
                    event_id = str(event.get("event_id", ""))
                    if not event_id or event_id in self._discovery_event_ids:
                        continue
                    self._discovery_event_ids.add(event_id)
                    self._discovery_events.append(dict(event))
        return written

    def write_lane_report(self, report: Any) -> bool:
        written = super().write_lane_report(report)
        if written:
            self._lane_reports_by_id[report.lane.lane_id] = report
            self._lane_reports_checksum = canonical_sha256(
                [item.to_dict() for item in self.lane_reports()]
            )
        return written

    def _cursor_unsigned(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "format_version": CURSOR_FORMAT_VERSION,
            "manifest_checksum": self.manifest_checksum,
            "lane_registry_sha256": canonical_sha256(
                [lane.to_dict() for lane in _t10_2_1_runtime.source_lane_registry()]
            ),
            "lane_reports_checksum": self._lane_reports_checksum,
            "accounting_checksum": canonical_sha256(self.accounting().to_dict()),
            "cumulative_active_seconds": float(state["cumulative_active_seconds"]),
            "open_lane_id": state.get("open_lane_id"),
            "open_lane_elapsed_seconds": float(
                state.get("open_lane_elapsed_seconds", 0.0)
            ),
            "reset_active_seconds": float(self._reset_active_seconds),
            "revision": int(state["revision"]),
            "full_checkpoint_revision": int(state.get("full_checkpoint_revision", -1)),
            "full_history_scan_count": self._full_history_scan_count,
        }

    def _write_cursor_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        payload = signed_payload(
            self._cursor_unsigned(state), checksum_key="cursor_checksum"
        )
        _t10_2_1_runtime._atomic_write_json(self.cursor_path, payload)
        return payload

    def _load_cursor_state(self) -> dict[str, Any]:
        checkpoint = self._persisted_checkpoint
        if not self.cursor_path.is_file():
            return {
                "cumulative_active_seconds": max(
                    self._reset_active_seconds,
                    0.0 if checkpoint is None else checkpoint.cumulative_active_seconds,
                ),
                "open_lane_id": None if checkpoint is None else checkpoint.open_lane_id,
                "open_lane_elapsed_seconds": (
                    0.0 if checkpoint is None else checkpoint.open_lane_elapsed_seconds
                ),
                "revision": -1 if checkpoint is None else checkpoint.revision,
                "full_checkpoint_revision": (
                    -1 if checkpoint is None else checkpoint.revision
                ),
            }
        payload = _t10_2_1_runtime._read_canonical_json(self.cursor_path)
        _t10_2_1_runtime._verify_signed(payload, checksum_key="cursor_checksum")
        expected_registry = canonical_sha256(
            [lane.to_dict() for lane in _t10_2_1_runtime.source_lane_registry()]
        )
        if (
            payload.get("format_version") != CURSOR_FORMAT_VERSION
            or payload.get("manifest_checksum") != self.manifest_checksum
            or payload.get("lane_registry_sha256") != expected_registry
            or payload.get("lane_reports_checksum") != self._lane_reports_checksum
            or payload.get("accounting_checksum")
            != canonical_sha256(self.accounting().to_dict())
            or int(payload.get("full_history_scan_count", -1)) != 1
        ):
            raise _t10_2_1_runtime.JournalIntegrityError(
                "compact cursor failed incremental reconstruction"
            )
        if checkpoint is not None and int(payload["revision"]) < checkpoint.revision:
            raise _t10_2_1_runtime.JournalIntegrityError(
                "compact cursor attempted to precede the full checkpoint"
            )
        return dict(payload)

    def _checkpoint_from_state(
        self,
        state: Mapping[str, Any],
        *,
        fallback: Any | None,
    ) -> Any:
        revision = int(state.get("revision", -1))
        if revision < 0 and fallback is None:
            return None
        return _t10_2_1_runtime.CollectionCheckpoint(
            manifest_checksum=self.manifest_checksum,
            lane_registry_sha256=canonical_sha256(
                [lane.to_dict() for lane in _t10_2_1_runtime.source_lane_registry()]
            ),
            lane_reports=self.lane_reports(),
            cumulative_active_seconds=float(state["cumulative_active_seconds"]),
            open_lane_id=state.get("open_lane_id"),
            open_lane_elapsed_seconds=float(
                state.get("open_lane_elapsed_seconds", 0.0)
            ),
            journal_reconstructed=True,
            checkpoint_reconstructed=True,
            physical_steps_replayed_on_resume=0,
            revision=max(0, revision),
        )

    def load_checkpoint(self) -> Any | None:
        return getattr(self, "_current_checkpoint", None)

    def _persist_full_checkpoint(self, checkpoint: Any) -> None:
        _t10_2_1_runtime._atomic_write_json(self.checkpoint_path, checkpoint.to_dict())
        reloaded = _t10_2_1_runtime.CollectionCheckpoint.from_dict(
            _t10_2_1_runtime._read_canonical_json(self.checkpoint_path)
        )
        if reloaded != checkpoint:
            raise _t10_2_1_runtime.JournalIntegrityError(
                "full checkpoint failed canonical reload"
            )
        self._persisted_checkpoint = checkpoint

    def write_checkpoint(self, checkpoint: Any) -> None:
        self._persist_full_checkpoint(checkpoint)

    def reconstruct_checkpoint(
        self,
        *,
        cumulative_active_seconds: float | None = None,
        open_lane: Any | None = None,
        open_lane_elapsed_seconds: float | None = None,
        close_open_lane: bool = False,
    ) -> Any:
        if close_open_lane and open_lane is not None:
            raise ValueError("checkpoint cannot open and close a lane together")
        state = dict(self._cursor_state)
        active = max(
            self._reset_active_seconds,
            float(state.get("cumulative_active_seconds", 0.0)),
            0.0
            if cumulative_active_seconds is None
            else float(cumulative_active_seconds),
        )
        if close_open_lane:
            open_lane_id = None
            open_seconds = 0.0
        elif open_lane is not None:
            if open_lane_elapsed_seconds is None:
                raise ValueError("open-lane checkpoint requires its active duration")
            open_lane_id = open_lane.lane_id
            open_seconds = max(
                float(open_lane_elapsed_seconds),
                float(state.get("open_lane_elapsed_seconds", 0.0))
                if state.get("open_lane_id") == open_lane_id
                else 0.0,
            )
        else:
            open_lane_id = state.get("open_lane_id")
            open_seconds = float(state.get("open_lane_elapsed_seconds", 0.0))
        changed = bool(
            active != float(state.get("cumulative_active_seconds", 0.0))
            or open_lane_id != state.get("open_lane_id")
            or open_seconds != float(state.get("open_lane_elapsed_seconds", 0.0))
            or state.get("lane_reports_checksum") != self._lane_reports_checksum
            or state.get("accounting_checksum")
            != canonical_sha256(self.accounting().to_dict())
        )
        if changed or int(state.get("revision", -1)) < 0:
            state["revision"] = int(state.get("revision", -1)) + 1
        state.update(
            {
                "cumulative_active_seconds": active,
                "open_lane_id": open_lane_id,
                "open_lane_elapsed_seconds": open_seconds,
                "lane_reports_checksum": self._lane_reports_checksum,
                "accounting_checksum": canonical_sha256(self.accounting().to_dict()),
            }
        )
        persist_full = bool(close_open_lane or not self.checkpoint_path.is_file())
        checkpoint = self._checkpoint_from_state(
            state, fallback=self._current_checkpoint
        )
        if checkpoint is None:
            raise _t10_2_1_runtime.JournalIntegrityError(
                "incremental checkpoint could not be materialized"
            )
        if persist_full:
            self._persist_full_checkpoint(checkpoint)
            state["full_checkpoint_revision"] = checkpoint.revision
        self._cursor_state = self._write_cursor_state(state)
        self._current_checkpoint = checkpoint
        return checkpoint


def _execution_lanes(mode: str) -> tuple[Any, ...]:
    payload = (
        _protocol.interleaved_lane_schedule()["order"]
        if mode == "full"
        else _protocol.smoke_lane_plan()["smoke_lanes"]
    )
    return tuple(_t10_2_1_runtime.SourceLaneKey.from_dict(lane) for lane in payload)


@contextmanager
def execution_bindings(*, mode: str, artifact_root: Path):
    """Bind the frozen acquisition loop to the registered T10.2.2 policy."""

    if mode not in {"full", "smoke"}:
        raise ValueError(f"invalid T10.2.2 execution mode: {mode}")
    lanes = _execution_lanes(mode)
    original_registry = _t10_2_1_runtime.source_lane_registry
    original_reset_specs = _t10_2_1_runtime.reset_work_specs
    original_completed = _t10_2_1_runtime._completed_discovery_events
    original_allowlist = _t10_2_1_runtime._COLLECTION_ROOT_ALLOWLIST
    replacements = {
        "DurableCollectionJournal": IncrementalDurableCollectionJournal,
        "DEFAULT_OUTPUT_DIR": Path(artifact_root),
        "LANE_HARD_SECONDS": LANE_LIVENESS_WALL_SECONDS,
        "COLLECTION_COOPERATIVE_SECONDS": COLLECTION_STOP_NEW_ACTIONS_SECONDS,
        "COLLECTION_ABSOLUTE_SECONDS": COLLECTION_LIVENESS_WALL_SECONDS,
    }
    if mode == "smoke":
        smoke = _protocol.smoke_lane_plan()
        replacements["SOURCE_RESETS_PER_LANE"] = int(smoke["resets_per_lane"])
        replacements["SOURCE_MAXIMUM_AUTHORIZED_INTENTS"] = (
            int(smoke["reset_report_count"]) * _protocol.RESET_ACTION_BUDGET
        )
    originals = {name: getattr(_t10_2_1_runtime, name) for name in replacements}

    def registered_lanes() -> tuple[Any, ...]:
        return lanes

    def registered_reset_specs(lane: Any) -> tuple[Any, ...]:
        if mode == "full":
            return original_reset_specs(lane)
        reset_count = int(_protocol.smoke_lane_plan()["resets_per_lane"])
        controllers = (
            ("balanced_discovery",) * reset_count
            if lane.split == "discovery"
            else _t10_2_1_runtime.confirmation_controller_order(lane.seed)[:reset_count]
        )
        return tuple(
            _t10_2_1_runtime.ResetWorkSpec(
                lane=lane,
                reset_index=reset_index,
                controller=controller,
            )
            for reset_index, controller in enumerate(controllers)
        )

    def completed_discovery_events(journal: Any) -> tuple[dict[str, Any], ...]:
        if isinstance(journal, IncrementalDurableCollectionJournal):
            return journal.completed_discovery_events()
        return original_completed(journal)

    try:
        for name, value in replacements.items():
            setattr(_t10_2_1_runtime, name, value)
        _t10_2_1_runtime.source_lane_registry = registered_lanes
        _t10_2_1_runtime.reset_work_specs = registered_reset_specs
        _t10_2_1_runtime._completed_discovery_events = completed_discovery_events
        _t10_2_1_runtime._COLLECTION_ROOT_ALLOWLIST = {
            **dict(original_allowlist),
            CURSOR_FILENAME: "file",
            T10_2_2_COLLECTION_REPORT_FILENAME: "file",
        }
        yield lanes
    finally:
        _t10_2_1_runtime.source_lane_registry = original_registry
        _t10_2_1_runtime.reset_work_specs = original_reset_specs
        _t10_2_1_runtime._completed_discovery_events = original_completed
        _t10_2_1_runtime._COLLECTION_ROOT_ALLOWLIST = original_allowlist
        for name, value in originals.items():
            setattr(_t10_2_1_runtime, name, value)


def budget_policy() -> dict[str, Any]:
    """Machine-readable descriptor of the T10.2.2 budget-by-actions policy."""

    return {
        "budget_basis": "physical_action_count",
        "reset_action_budget": _t10_2_1_runtime.SOURCE_ACTIONS_PER_RESET,
        "reset_liveness_wall_seconds": RESET_LIVENESS_WALL_SECONDS,
        "reset_liveness_hard_grace_seconds": RESET_LIVENESS_HARD_GRACE_SECONDS,
        "wall_clock_role": "liveness_watchdog_only",
        "frozen_reset_cooperative_seconds": _t10_2_1_runtime.RESET_COOPERATIVE_SECONDS,
        "frozen_lane_cap_seconds": _t10_2_1_runtime.LANE_HARD_SECONDS,
        "frozen_collection_cap_seconds": _t10_2_1_runtime.COLLECTION_ABSOLUTE_SECONDS,
        "reset_level_applied": True,
        "lane_and_collection_level_applied": True,
        "incremental_history_state_applied": True,
        "compact_cursor_applied": True,
        "preview_copy_adapter_scope": [
            "GaugeProgramPosterior",
            "FactorizedGaugeProgramPosterior",
        ],
        "learned_preview_copy_applied": True,
    }


class ActionBudgetSourceFactory(_t10_2_1_runtime.T10_2_1SourceFactory):
    """Frozen scientific worker with correctly armed wide parent kill guards."""

    def run_reset(
        self,
        *,
        work: Any,
        journal: Any,
        discovery_events: Sequence[Mapping[str, Any]],
        continuation: Mapping[str, Any],
        process_context: Any,
        lane_remaining_seconds: float,
        cooperative_collection_remaining_seconds: float,
        absolute_collection_remaining_seconds: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> Any:
        journal.assert_safe_resume_boundary(work)
        cooperative_seconds = min(
            RESET_LIVENESS_WALL_SECONDS,
            float(lane_remaining_seconds),
            float(cooperative_collection_remaining_seconds),
        )
        hard_seconds = min(
            RESET_LIVENESS_WALL_SECONDS + RESET_LIVENESS_HARD_GRACE_SECONDS,
            float(lane_remaining_seconds),
            float(absolute_collection_remaining_seconds),
        )
        if cooperative_seconds <= 0.0 or hard_seconds <= 0.0:
            global_budget_exhausted = bool(
                float(cooperative_collection_remaining_seconds) <= 0.0
                or float(absolute_collection_remaining_seconds) <= 0.0
            )
            return _t10_2_1_runtime.WorkerOutcome(
                status="COOPERATIVE_STOP",
                elapsed_seconds=0.0,
                payload={
                    "completed": False,
                    "stop_reason": (
                        "registered_collection_deadline"
                        if global_budget_exhausted
                        else "cooperative_reset_deadline"
                    ),
                },
                error_kind="WorkerTimeoutError",
            )
        reset_started = float(clock())
        if not math.isfinite(reset_started):
            raise _t10_2_1_runtime.WorkerProtocolError(
                "reset monotonic start is invalid"
            )
        hard_deadline_monotonic = time.monotonic() + hard_seconds
        context = process_context or multiprocessing.get_context("spawn")
        outbound_queue = context.Queue()
        inbound_queue = context.Queue()
        cancel_event = context.Event()
        process = context.Process(
            target=_action_budget_reset_worker_entry,
            args=(
                self.clone_for_worker(),
                work.to_dict(),
                tuple(dict(item) for item in discovery_events),
                dict(_t10_2_1_runtime._jsonable(continuation)),
                cancel_event,
                outbound_queue,
                inbound_queue,
            ),
            name=f"sage-t10-2-2-{work.work_id[:12]}",
        )
        hard_cancel_event = threading.Event()
        hard_guard = threading.Thread(
            target=_t10_2_1_runtime._reset_hard_watchdog_entry,
            args=(
                _t10_2_1_runtime.os.getpid(),
                hard_deadline_monotonic,
                hard_cancel_event,
            ),
            name=f"sage-t10-2-2-reset-hard-watchdog-{work.work_id[:8]}",
            daemon=True,
        )
        try:
            hard_guard.start()
        except Exception as exc:
            raise _t10_2_1_runtime.WorkerProtocolError(
                "external action-budget reset watchdog could not start"
            ) from exc
        try:
            process.start()
            worker_pid = getattr(process, "pid", None)
            if not isinstance(worker_pid, int) or worker_pid <= 0:
                _t10_2_1_runtime.ProcessResetWatchdog(clock=clock)._terminate(process)
                raise _t10_2_1_runtime.WorkerProtocolError(
                    "spawned reset worker lacks a process id"
                )
            handler = _t10_2_1_runtime._ParentJournalMessageHandler(
                journal=journal,
                work=work,
                manifest=self.manifest,
                clock=clock,
                intent_deadline=reset_started + min(cooperative_seconds, hard_seconds),
            )
            watchdog = self.watchdog or _t10_2_1_runtime.ProcessResetWatchdog(
                clock=clock
            )
            return watchdog.supervise(
                process,
                work=work,
                cancel_event=cancel_event,
                outbound_queue=outbound_queue,
                inbound_queue=inbound_queue,
                message_handler=handler,
                cooperative_seconds=cooperative_seconds,
                hard_seconds=hard_seconds,
                started_at=reset_started,
            )
        finally:
            hard_cancel_event.set()
            hard_guard.join(timeout=2.0)


def _gauge_preview_clone(self: Any, memo: dict[int, Any]) -> Any:
    """Clone mutable gauge-posterior state around frozen value contracts.

    The T10.2.1 preview uses ``copy.deepcopy``.  Both its learned joint posterior
    and factorized control hold immutable ``MappingProxyType`` contract views
    which cannot be pickled on Windows, even though observation only replaces
    frozen particles.  T10.2.2 therefore supplies this worker-scoped adapter:
    immutable particle/factor values are shared, while every mutable posterior
    container and executor cache is isolated.
    """

    existing = memo.get(id(self))
    if existing is not None:
        return existing
    executor = copy.copy(self.executor)
    if hasattr(executor, "_step_cache"):
        executor._step_cache = dict(getattr(self.executor, "_step_cache", {}))
    clone = type(self)(
        executor=executor,
        maximum_classes=self.maximum_classes,
        channel_weights=dict(self.channel_weights),
        unknown_coverage_penalty=self.unknown_coverage_penalty,
        commutativity_penalty=self.commutativity_penalty,
    )
    memo[id(self)] = clone
    clone._particles = list(self._particles)
    clone._event_ids = list(self._event_ids)
    clone._seen_event_ids = set(self._seen_event_ids)
    clone._branch_index = self._branch_index
    clone._top_class_key = self._top_class_key
    clone._top_class_streak = self._top_class_streak
    clone._collapsed = self._collapsed
    clone._last_update = self._last_update
    clone._residual_log_mass = self._residual_log_mass
    for attribute in (
        "_marginals",
        "_bank_metrics",
        "_factorized_updates",
        "_last_likelihood_decomposition_error",
        "_materialized_product_mass",
    ):
        if hasattr(self, attribute):
            setattr(clone, attribute, getattr(self, attribute))
    return clone


@contextmanager
def gauge_preview_copy_binding():
    """Install the T10.2.2 preview adapter only inside the spawned worker."""

    posterior_type = _t10_2_1_runtime.GaugeProgramPosterior
    had_local = "__deepcopy__" in posterior_type.__dict__
    original = posterior_type.__dict__.get("__deepcopy__")
    posterior_type.__deepcopy__ = _gauge_preview_clone
    try:
        yield
    finally:
        if had_local:
            posterior_type.__deepcopy__ = original
        else:
            delattr(posterior_type, "__deepcopy__")


def _action_budget_reset_worker_entry(*args: Any) -> None:
    """Run the frozen worker with only the registered preview-copy adapter."""

    with gauge_preview_copy_binding():
        _t10_2_1_runtime._reset_worker_entry(*args)


def timing_source_factory(
    *,
    manifest: Mapping[str, Any],
    recorder: CollectionTimingRecorder,
    clock: Callable[[], float] = time.perf_counter,
) -> _t10_2_1_runtime.T10_2_1SourceFactory:
    """A frozen T10.2.1 factory with a T10.2.2 timing watchdog attached."""

    return _t10_2_1_runtime.T10_2_1SourceFactory(
        manifest=manifest,
        watchdog=LaneTimingWatchdog(recorder=recorder, clock=clock),
    )


def action_budget_source_factory(
    *,
    manifest: Mapping[str, Any],
    recorder: CollectionTimingRecorder,
    clock: Callable[[], float] = time.perf_counter,
) -> ActionBudgetSourceFactory:
    """A frozen T10.2.1 factory supervised under the budget-by-actions policy."""

    return ActionBudgetSourceFactory(
        manifest=manifest,
        watchdog=ActionBudgetWatchdog(recorder=recorder, clock=clock),
    )


# ---------------------------------------------------------------------------
# Sidecar report construction (pure; unit-testable without multiprocessing).
# ---------------------------------------------------------------------------
def _iter_reset_reports(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    resets: list[dict[str, Any]] = []
    for lane in checkpoint.get("lane_reports", []) or []:
        if not isinstance(lane, Mapping):
            continue
        for reset in lane.get("resets", []) or []:
            if isinstance(reset, Mapping):
                resets.append(dict(reset))
    return resets


def _parse_schema_key(key: Any) -> tuple[str, int]:
    if isinstance(key, (list, tuple)) and len(key) == 2:
        return str(key[0]), int(key[1])
    try:
        parsed = json.loads(str(key))
    except json.JSONDecodeError:
        return str(key), 0
    if isinstance(parsed, list) and len(parsed) == 2:
        return str(parsed[0]), int(parsed[1])
    return str(key), 0


def _aggregate_schema_counts(
    reset_reports: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], int], dict[str, int]]:
    learned: Counter[tuple[str, int]] = Counter()
    independent: Counter[tuple[str, int]] = Counter()
    grounding: Counter[str] = Counter()
    for report in reset_reports:
        continuation = report.get("continuation")
        if not isinstance(continuation, Mapping):
            continue
        for key, value in (continuation.get("learned_schema_counts") or {}).items():
            learned[_parse_schema_key(key)] += int(value)
        for key, value in (continuation.get("independent_schema_counts") or {}).items():
            independent[_parse_schema_key(key)] += int(value)
        for key, value in (continuation.get("grounding_counts") or {}).items():
            grounding[str(key)] += int(value)
    return dict(learned), dict(independent), dict(grounding)


def _git_commit_sha(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def build_implementation_provenance(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Point 7: pin the exact implementation bytes and scheduling of this run.

    A manifest checksum alone cannot distinguish two runs whose kernel/engine
    bytes differ; T10.2.2 therefore records per-file portable digests, the git
    commit, and the interleaved/reserved schedule that is actually planned.
    """

    root = Path(repo_root or _t10_2_1._repo_root()).resolve()
    implementation_sha256: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = (root / relative).resolve()
        if path.is_file():
            implementation_sha256[relative] = _t10_2_1.canonical_file_sha256(path)
    schedule = _protocol.interleaved_lane_schedule()
    smoke = _protocol.smoke_lane_plan()
    return signed_payload(
        {
            "format_version": PROVENANCE_FORMAT_VERSION,
            "git_commit_sha": _git_commit_sha(root),
            "implementation_sha256": implementation_sha256,
            "runtime_format_version": FORMAT_VERSION,
            "protocol_format_version": _protocol.FORMAT_VERSION,
            "budget_policy": budget_policy(),
            "scheduler": {
                "interleaved": schedule["interleaved"],
                "reserved_confirmation_capacity": schedule[
                    "reserved_confirmation_capacity"
                ],
                "scheduled_order_lane_ids": [
                    lane["lane_id"] for lane in schedule["order"]
                ],
                "smoke_lane_ids": [lane["lane_id"] for lane in smoke["smoke_lanes"]],
            },
        },
        checksum_key="provenance_checksum",
    )


def build_t10_2_2_collection_report(
    *,
    recorder: CollectionTimingRecorder,
    checkpoint: Mapping[str, Any],
    collection_report: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    t10_2_2_manifest: Mapping[str, Any] | None = None,
    mode: str = "full",
    cursor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the readiness-anchored T10.2.2 sidecar over frozen artifacts."""

    kernel_manifest_checksum = collection_report.get("manifest_checksum")
    manifest_checksum = (
        t10_2_2_manifest.get("manifest_checksum")
        if t10_2_2_manifest is not None
        else kernel_manifest_checksum
    )
    checkpoint_binding = _protocol.build_checkpoint_binding(
        collection_report=collection_report, checkpoint=checkpoint
    )

    # Item 2: lane startup timing from the observed windows.
    lane_timings: list[dict[str, Any]] = []
    for window in recorder.lane_windows():
        started = window["lane_started_seconds"]
        finished = window["lane_finished_seconds"]
        if started is None or finished is None:
            continue
        lane_timings.append(
            _protocol.compute_lane_startup_timing(
                lane=window["lane"],
                lane_started_seconds=started,
                first_committed_transition_seconds=window[
                    "first_committed_transition_seconds"
                ],
                lane_finished_seconds=finished,
            )
        )
    phase_timing = _protocol.build_phase_timing(lane_timings=lane_timings)

    # Items 3 and 4: readiness gates and readiness-relative first-intent status.
    readiness_gates: list[dict[str, Any]] = []
    first_intent_statuses: list[dict[str, Any]] = []
    for timing in recorder.reset_timings():
        if timing.controller_ready_at is None or timing.environment_ready_at is None:
            first_intent_statuses.append(
                {
                    "work_id": timing.work_id,
                    "readiness_observed": False,
                    "status": None,
                }
            )
            continue
        readiness_gates.append(
            _protocol.readiness_gate(
                lane=timing.lane,
                controller_ready_at=timing.controller_ready_at,
                environment_ready_at=timing.environment_ready_at,
                interaction_budget_seconds=INTERACTION_BUDGET_SECONDS,
            )
        )
        status = _protocol.classify_first_intent(
            controller_ready_at=timing.controller_ready_at,
            environment_ready_at=timing.environment_ready_at,
            first_intent_authorized_at=timing.first_intent_at,
            first_intent_budget_seconds=FIRST_INTENT_BUDGET_SECONDS,
        )
        first_intent_statuses.append(
            {
                "work_id": timing.work_id,
                "readiness_observed": True,
                "status": status,
            }
        )

    # Item 5: evidence funnel from the frozen reset reports.
    reset_reports = _iter_reset_reports(checkpoint)
    evidence_funnel = _protocol.evidence_funnel_from_reset_reports(reset_reports)

    # Point 5: controller-activity semantics (posterior zeros -> NOT_APPLICABLE).
    controller_activity = _protocol.build_controller_activity(reset_reports)
    # Point 6: explicit event-eligibility ventilation.
    event_eligibility = _protocol.build_event_eligibility(reset_reports)

    # Item 6: canonical schema families vs grounded instances.
    learned, independent, grounding = _aggregate_schema_counts(reset_reports)
    schema_evidence = _protocol.partition_schema_evidence(
        learned_schema_counts=learned,
        independent_schema_counts=independent,
        grounding_counts=grounding,
    )

    # Item 7: controlled end-to-end induction canary.
    induction_canary = _protocol.run_induction_canary()

    first_intent_any_timeout = any(
        row.get("status") == "first_intent_timeout" for row in first_intent_statuses
    )
    incremental_runtime_gate = {
        "cursor_present": cursor is not None,
        "cursor_format_version": (
            None if cursor is None else cursor.get("format_version")
        ),
        "full_history_scan_count": (
            None if cursor is None else cursor.get("full_history_scan_count")
        ),
        "one_resume_scan_only": bool(
            cursor is not None
            and cursor.get("format_version") == CURSOR_FORMAT_VERSION
            and cursor.get("full_history_scan_count") == 1
        ),
        "compact_cursor_checksum": (
            None if cursor is None else cursor.get("cursor_checksum")
        ),
    }

    smoke_gate: dict[str, Any] | None = None
    if mode == "smoke":
        registered_plan = _protocol.smoke_lane_plan()
        lane_reports = [
            lane
            for lane in (checkpoint.get("lane_reports", []) or [])
            if isinstance(lane, Mapping)
        ]
        expected_lane_count = int(registered_plan["smoke_lane_count"])
        expected_reset_count = int(registered_plan["reset_report_count"])
        lane_statuses = [str(lane.get("status")) for lane in lane_reports]
        reset_statuses = [str(reset.get("status")) for reset in reset_reports]
        confirmation_resets = [
            reset
            for lane in lane_reports
            if isinstance(lane.get("lane"), Mapping)
            and lane["lane"].get("split") == "leave_one_game_out_confirmation"
            for reset in (lane.get("resets", []) or [])
            if isinstance(reset, Mapping)
        ]
        confirmation_controllers = [
            str((reset.get("work") or {}).get("controller"))
            for reset in confirmation_resets
            if isinstance(reset.get("work"), Mapping)
        ]
        accounting = collection_report.get("action_accounting")
        accounting = accounting if isinstance(accounting, Mapping) else {}
        accounting_closed = bool(
            accounting.get("equation_holds") is True
            and int(accounting.get("unknown_intent_count", -1)) == 0
            and int(accounting.get("explicitly_unresolved_intent_count", -1)) == 0
        )
        all_lanes_complete = bool(
            len(lane_reports) == expected_lane_count
            and all(status == "COMPLETE" for status in lane_statuses)
        )
        all_resets_complete = bool(
            len(reset_reports) == expected_reset_count
            and all(status == "COMPLETE" for status in reset_statuses)
        )
        confirmation_complete = bool(
            len(confirmation_resets)
            == len(registered_plan["confirmation_controller_sequence"])
            and all(reset.get("status") == "COMPLETE" for reset in confirmation_resets)
            and all(
                int(reset.get("sealed_events", 0)) > 0 for reset in confirmation_resets
            )
        )
        confirmation_controller_coverage = bool(
            confirmation_controllers
            == list(registered_plan["confirmation_controller_sequence"])
        )
        timing_complete = bool(
            len(recorder.reset_timings()) == expected_reset_count
            and all(timing.status == "COMPLETE" for timing in recorder.reset_timings())
        )
        smoke_gate = {
            "registered_plan": registered_plan,
            "donor_safe": registered_plan["confirmation_donor_safe"],
            "lane_report_count": len(lane_reports),
            "expected_lane_report_count": expected_lane_count,
            "lane_statuses": lane_statuses,
            "reset_report_count": len(reset_reports),
            "expected_reset_report_count": expected_reset_count,
            "reset_statuses": reset_statuses,
            "all_lanes_complete": all_lanes_complete,
            "all_resets_complete": all_resets_complete,
            "confirmation_complete_with_evidence": confirmation_complete,
            "confirmation_controllers": confirmation_controllers,
            "confirmation_controller_coverage": confirmation_controller_coverage,
            "action_accounting_closed": accounting_closed,
            "timing_complete": timing_complete,
            "no_first_intent_timeout": not first_intent_any_timeout,
            "worker_error_kinds": sorted(
                {
                    timing.error_kind
                    for timing in recorder.reset_timings()
                    if timing.error_kind is not None
                }
            ),
            "passed": bool(
                registered_plan["confirmation_donor_safe"]
                and all_lanes_complete
                and all_resets_complete
                and confirmation_complete
                and confirmation_controller_coverage
                and accounting_closed
                and timing_complete
                and not first_intent_any_timeout
                and incremental_runtime_gate["one_resume_scan_only"]
            ),
        }

    return signed_payload(
        {
            "format_version": T10_2_2_COLLECTION_REPORT_FORMAT_VERSION,
            "phase": "collect",
            "manifest_checksum": manifest_checksum,
            "kernel_manifest_checksum": kernel_manifest_checksum,
            "execution_mode": mode,
            "frozen_collection_report_checksum": collection_report.get(
                "report_checksum"
            ),
            "checkpoint_binding": checkpoint_binding,
            "phase_timing": phase_timing,
            "readiness_gates": readiness_gates,
            "first_intent": {
                "budget_seconds": FIRST_INTENT_BUDGET_SECONDS,
                "per_reset": first_intent_statuses,
                "any_timeout": first_intent_any_timeout,
            },
            "interaction_budget_seconds": INTERACTION_BUDGET_SECONDS,
            "evidence_funnel": evidence_funnel,
            "controller_activity": controller_activity,
            "event_eligibility": event_eligibility,
            "schema_evidence": schema_evidence,
            "induction_canary": induction_canary,
            "provenance": (dict(provenance) if provenance is not None else None),
            "incremental_runtime_gate": incremental_runtime_gate,
            "smoke_gate": smoke_gate,
            "reset_timings": [timing.to_dict() for timing in recorder.reset_timings()],
        },
        checksum_key="report_checksum",
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestDriftError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ManifestDriftError(f"{label} is not a JSON object: {path}")
    return dict(payload)


def write_t10_2_2_collection_report(
    *,
    recorder: CollectionTimingRecorder,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    """Read the frozen collection report + checkpoint and write the sidecar."""

    destination = Path(output_dir)
    collection_report = _read_signed_json(
        destination / COLLECTION_REPORT_FILENAME, checksum_key="report_checksum"
    )
    checkpoint = _load_json_object(
        destination / CHECKPOINT_FILENAME, label="collection checkpoint"
    )
    cursor_path = destination / CURSOR_FILENAME
    cursor = (
        _read_signed_json(cursor_path, checksum_key="cursor_checksum")
        if cursor_path.is_file()
        else None
    )
    provenance = build_implementation_provenance(repo_root)
    report_path = destination / T10_2_2_COLLECTION_REPORT_FILENAME
    if report_path.is_file():
        existing = _read_signed_json(report_path, checksum_key="report_checksum")
        expected_manifest_checksum = (
            manifest.get("manifest_checksum")
            if manifest is not None
            else collection_report.get("manifest_checksum")
        )
        expected_binding = _protocol.build_checkpoint_binding(
            collection_report=collection_report, checkpoint=checkpoint
        )
        bindings_match = bool(
            existing.get("format_version") == T10_2_2_COLLECTION_REPORT_FORMAT_VERSION
            and existing.get("execution_mode") == mode
            and existing.get("manifest_checksum") == expected_manifest_checksum
            and existing.get("kernel_manifest_checksum")
            == collection_report.get("manifest_checksum")
            and existing.get("frozen_collection_report_checksum")
            == collection_report.get("report_checksum")
            and existing.get("checkpoint_binding") == expected_binding
            and existing.get("provenance") == provenance
        )
        if not bindings_match:
            raise ManifestDriftError(
                "existing T10.2.2 sidecar does not match the immutable terminal "
                "collection artifacts"
            )
        return existing

    report = build_t10_2_2_collection_report(
        recorder=recorder,
        checkpoint=checkpoint,
        collection_report=collection_report,
        provenance=provenance,
        t10_2_2_manifest=manifest,
        mode=mode,
        cursor=cursor,
    )
    write_compact_json(report_path, report)
    return report


# ---------------------------------------------------------------------------
# Collect phase: run the frozen collection with a timing watchdog, then sidecar.
# ---------------------------------------------------------------------------
def collect_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    mode: str = "full",
    env_factory: _t10_2_1_runtime.T10_2_1SourceFactory | None = None,
    recorder: CollectionTimingRecorder | None = None,
    clock: Callable[[], float] = time.perf_counter,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the isolated T10.2.2 full or donor-safe smoke acquisition.

    The frozen ``collect_phase`` writes ``collection_report.json`` and
    ``source_collection_checkpoint.json`` into the manifest-registered artifact
    directory.  T10.2.2 then writes ``t10_2_2_collection_report.json`` beside
    them, binding the checkpoint's exact revision and checksum.
    """

    root = Path(repo_root or _t10_2_1._repo_root()).resolve()
    t10_2_2_manifest = _protocol.load_manifest(
        manifest_path, repo_root=root, verify_repository=True
    )
    kernel_manifest, kernel_path, artifact_root = _protocol.load_kernel_manifest(
        manifest=t10_2_2_manifest,
        mode=mode,
        repo_root=root,
    )
    registered_destination = (root / artifact_root).resolve()
    if output_dir is not None:
        candidate = Path(output_dir)
        candidate = (
            candidate if candidate.is_absolute() else root / candidate
        ).resolve()
        if candidate != registered_destination:
            raise ManifestDriftError(
                "T10.2.2 output escaped its registered isolated namespace"
            )
    recorder = recorder or CollectionTimingRecorder()
    relative_kernel = kernel_path.relative_to(root)
    with (
        _protocol.kernel_protocol_bindings(
            artifact_root=artifact_root,
            manifest_relative_path=relative_kernel,
            mode=mode,
        ),
        execution_bindings(mode=mode, artifact_root=artifact_root),
    ):
        verified_kernel = _t10_2_1.load_manifest(kernel_path, repo_root=root)
        if verified_kernel != kernel_manifest:
            raise ManifestDriftError("verified execution kernel escaped T10.2.2")
        if env_factory is None:
            env_factory = action_budget_source_factory(
                manifest=verified_kernel, recorder=recorder, clock=clock
            )
        elif env_factory.watchdog is None:
            env_factory.watchdog = ActionBudgetWatchdog(recorder=recorder, clock=clock)
        _t10_2_1.collect_phase(
            manifest_path=kernel_path,
            output_dir=artifact_root,
            repo_root=root,
            env_factory=env_factory,
            clock=clock,
            **kwargs,
        )

    return write_t10_2_2_collection_report(
        recorder=recorder,
        output_dir=registered_destination,
        repo_root=root,
        manifest=t10_2_2_manifest,
        mode=mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "smoke"))
    parser.add_argument("--manifest", default=str(_protocol.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=str(_t10_2_1._repo_root()))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recorder = CollectionTimingRecorder()
    try:
        payload = collect_phase(
            manifest_path=args.manifest,
            repo_root=args.repo_root,
            recorder=recorder,
            mode="smoke" if args.phase == "smoke" else "full",
        )
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(
            _t10_2_1.canonical_json(
                {"error": f"{type(exc).__name__}:{exc}", "phase": args.phase}
            )
        )
        return 2
    print(_t10_2_1.canonical_json(payload))
    return 0


__all__ = [
    "ActionBudgetWatchdog",
    "CURSOR_FILENAME",
    "CURSOR_FORMAT_VERSION",
    "CollectionTimingRecorder",
    "IncrementalCollectionState",
    "IncrementalDurableCollectionJournal",
    "RunningAccounting",
    "FIRST_INTENT_BUDGET_SECONDS",
    "FORMAT_VERSION",
    "IMPLEMENTATION_FILES",
    "INTERACTION_BUDGET_SECONDS",
    "LaneTimingWatchdog",
    "PROVENANCE_FORMAT_VERSION",
    "RESET_LIVENESS_WALL_SECONDS",
    "ResetTiming",
    "T10_2_2_COLLECTION_REPORT_FILENAME",
    "T10_2_2_COLLECTION_REPORT_FORMAT_VERSION",
    "action_budget_source_factory",
    "budget_policy",
    "build_implementation_provenance",
    "build_parser",
    "build_t10_2_2_collection_report",
    "collect_phase",
    "execution_bindings",
    "main",
    "timing_source_factory",
    "write_t10_2_2_collection_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
