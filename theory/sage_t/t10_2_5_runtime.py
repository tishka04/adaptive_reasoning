"""Crash-recovery runtime for SAGE.T10.2.5."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_2_runtime as _parent_runtime
from . import t10_2_4_protocol as _predecessor_protocol
from . import t10_2_4_runtime as _predecessor_runtime
from . import t10_2_5_protocol as _protocol

FORMAT_VERSION = "sage-t10.2.5-runtime-v1"
RECOVERY_REPORT_FORMAT_VERSION = "sage-t10.2.5-recovery-report-v1"
ACCEPTED_AUDIT_FORMAT_VERSION = "sage-t10.2.5-accepted-cross-fit-audit-v1"
COLLECTION_REPORT_FORMAT_VERSION = "sage-t10.2.5-collection-report-v1"
RECOVERY_REPORT_FILENAME = "recovery_report.json"
ACCEPTED_EVENT_FILENAME = "accepted_source_events.jsonl"
ACCEPTED_AUDIT_FILENAME = "accepted_cross_fit_audit.json"
COLLECTION_REPORT_FILENAME = "t10_2_5_collection_report.json"
CURSOR_REPAIR_RECEIPT_FILENAME = "parent_cursor_repair_receipt.json"
ORPHAN_CLOSURE_RECEIPT_FILENAME = "parent_orphan_closure_receipt.json"
RECOVERY_JOURNAL_DIRECTORY = "source_collection_journal"

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError
WorkerProtocolError = _kernel_runtime.WorkerProtocolError


def _terminate_worker_tree(worker_pid: int) -> None:
    """Terminate only a reset worker and its descendants, never its collector."""

    if int(worker_pid) <= 0 or int(worker_pid) == os.getpid():
        raise WorkerProtocolError("worker-only watchdog received the collector pid")
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill.exe", "/PID", str(int(worker_pid)), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=10.0,
        )
        if getattr(completed, "returncode", 1) == 0:
            return
    try:
        os.kill(int(worker_pid), signal.SIGTERM)
    except OSError:
        return


def _worker_hard_watchdog_entry(
    worker_pid: int,
    deadline_monotonic: float,
    cancel_event: threading.Event,
    fired_event: threading.Event,
) -> None:
    while True:
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0.0:
            break
        if cancel_event.wait(timeout=min(0.25, remaining)):
            return
    if cancel_event.is_set():
        return
    fired_event.set()
    _terminate_worker_tree(worker_pid)


def _worker_scoped_run_reset(
    self: Any,
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
    """T10.2.2 reset runner with the external guard scoped to the worker PID."""

    journal.assert_safe_resume_boundary(work)
    cooperative_seconds = min(
        _parent_runtime.RESET_LIVENESS_WALL_SECONDS,
        float(lane_remaining_seconds),
        float(cooperative_collection_remaining_seconds),
    )
    hard_seconds = min(
        _parent_runtime.RESET_LIVENESS_WALL_SECONDS
        + _parent_runtime.RESET_LIVENESS_HARD_GRACE_SECONDS,
        float(lane_remaining_seconds),
        float(absolute_collection_remaining_seconds),
    )
    if cooperative_seconds <= 0.0 or hard_seconds <= 0.0:
        global_budget_exhausted = bool(
            float(cooperative_collection_remaining_seconds) <= 0.0
            or float(absolute_collection_remaining_seconds) <= 0.0
        )
        return _kernel_runtime.WorkerOutcome(
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
        raise WorkerProtocolError("reset monotonic start is invalid")
    context = process_context or multiprocessing.get_context("spawn")
    outbound_queue = context.Queue()
    inbound_queue = context.Queue()
    cancel_event = context.Event()
    process = context.Process(
        target=_parent_runtime._action_budget_reset_worker_entry,
        args=(
            self.clone_for_worker(),
            work.to_dict(),
            tuple(dict(item) for item in discovery_events),
            dict(_kernel_runtime._jsonable(continuation)),
            cancel_event,
            outbound_queue,
            inbound_queue,
        ),
        name=f"sage-t10-2-5-{work.work_id[:12]}",
    )
    process.start()
    worker_pid = getattr(process, "pid", None)
    if not isinstance(worker_pid, int) or worker_pid <= 0:
        _kernel_runtime.ProcessResetWatchdog(clock=clock)._terminate(process)
        raise WorkerProtocolError("spawned reset worker lacks a process id")
    hard_cancel_event = threading.Event()
    hard_fired_event = threading.Event()
    hard_guard = threading.Thread(
        target=_worker_hard_watchdog_entry,
        args=(
            worker_pid,
            time.monotonic() + hard_seconds,
            hard_cancel_event,
            hard_fired_event,
        ),
        name=f"sage-t10-2-5-worker-watchdog-{work.work_id[:8]}",
        daemon=True,
    )
    try:
        hard_guard.start()
    except Exception as exc:
        _kernel_runtime.ProcessResetWatchdog(clock=clock)._terminate(process)
        raise WorkerProtocolError("worker-only hard watchdog could not start") from exc
    try:
        handler = _kernel_runtime._ParentJournalMessageHandler(
            journal=journal,
            work=work,
            manifest=self.manifest,
            clock=clock,
            intent_deadline=reset_started + min(cooperative_seconds, hard_seconds),
        )
        watchdog = self.watchdog or _kernel_runtime.ProcessResetWatchdog(clock=clock)
        outcome = watchdog.supervise(
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
        if hard_fired_event.is_set() and outcome.status != "HARD_TIMEOUT":
            return _kernel_runtime.WorkerOutcome(
                status="HARD_TIMEOUT",
                elapsed_seconds=max(0.0, float(clock()) - reset_started),
                payload={
                    "completed": False,
                    "work_id": work.work_id,
                    "stop_reason": "hard_reset_timeout",
                },
                error_kind="WorkerTimeoutError",
            )
        return outcome
    finally:
        hard_cancel_event.set()
        hard_guard.join(timeout=2.0)


@contextmanager
def worker_only_watchdog_binding():
    """Scope the corrected reset runner around parent and recovery collection."""

    original = _parent_runtime.ActionBudgetSourceFactory.run_reset
    _parent_runtime.ActionBudgetSourceFactory.run_reset = _worker_scoped_run_reset
    try:
        yield
    finally:
        _parent_runtime.ActionBudgetSourceFactory.run_reset = original


class RecoveryDualCachedSourceFactory(
    _predecessor_runtime.DualCachedDonorSourceFactory
):
    """Use T10.2.4 caches while signing new actions with the T10.2.5 manifest."""

    def __init__(self, *, cache_kernel_manifest_checksum: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cache_kernel_manifest_checksum = str(cache_kernel_manifest_checksum)

    def _coordinator(self) -> _predecessor_runtime.DualCacheCoordinator:
        return _predecessor_runtime.DualCacheCoordinator(
            root=self.cache_root,
            predecessor_root=self.predecessor_cache_root,
            continuation_manifest_checksum=self.continuation_manifest_checksum,
            predecessor_manifest_checksum=self.predecessor_manifest_checksum,
            parent_kernel_manifest_checksum=self.cache_kernel_manifest_checksum,
        )

    def clone_for_worker(self) -> "RecoveryDualCachedSourceFactory":
        return RecoveryDualCachedSourceFactory(
            manifest=self.manifest,
            cache_root=self.cache_root,
            predecessor_cache_root=self.predecessor_cache_root,
            continuation_manifest_checksum=self.continuation_manifest_checksum,
            predecessor_manifest_checksum=self.predecessor_manifest_checksum,
            cache_kernel_manifest_checksum=self.cache_kernel_manifest_checksum,
            runtime_loader=self._runtime_loader,
            bundle_builder=self._bundle_builder,
        )


def _load_execution_context(
    *, manifest_path: str | Path, repo_root: str | Path | None
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
]:
    root = Path(repo_root or _kernel_protocol._repo_root()).resolve()
    manifest = _protocol.load_manifest(
        manifest_path,
        repo_root=root,
        verify_repository=True,
        verify_live_migration=True,
    )
    predecessor = _protocol._load_predecessor(root)
    parent, kernel, kernel_path, artifact_root = _protocol._parent_execution(root)
    return root, manifest, predecessor, parent, kernel, kernel_path, artifact_root


def _write_once_payload(path: Path, payload: Mapping[str, Any], *, checksum_key: str) -> None:
    if path.is_file():
        existing = _kernel_protocol._read_signed_json(path, checksum_key=checksum_key)
        if dict(existing) != dict(payload):
            raise ManifestDriftError(f"existing {path.name} drifted")
        return
    _kernel_protocol.write_compact_json(path, payload)


def _repair_parent_cursor(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    kernel: Mapping[str, Any],
    kernel_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    destination = root / artifact_root
    repair = manifest["migration_receipt"]["cursor_repair"]
    expected = dict(repair["repaired_cursor"])
    receipt_path = (
        root / _protocol.DEFAULT_RECOVERY_ROOT / CURSOR_REPAIR_RECEIPT_FILENAME
    )
    lease = _kernel_runtime._CollectionLease.acquire(
        destination / ".active-collector.lock"
    )
    try:
        current = _kernel_protocol._read_signed_json(
            destination / _parent_runtime.CURSOR_FILENAME,
            checksum_key="cursor_checksum",
        )
        if current.get("cursor_checksum") == repair["old_cursor_checksum"]:
            snapshot = _protocol._journal_snapshot(
                root=root,
                kernel=kernel,
                kernel_path=kernel_path,
                artifact_root=artifact_root,
            )
            reconstructed = _protocol._expected_cursor_repair(
                cursor=current,
                snapshot=snapshot,
                kernel_manifest_checksum=str(kernel["manifest_checksum"]),
            )
            if reconstructed != expected:
                raise JournalIntegrityError("live cursor repair no longer reconstructs")
            _kernel_runtime._atomic_write_json(
                destination / _parent_runtime.CURSOR_FILENAME,
                expected,
            )
            current = _kernel_protocol._read_signed_json(
                destination / _parent_runtime.CURSOR_FILENAME,
                checksum_key="cursor_checksum",
            )
        elif current.get("cursor_checksum") != repair["repaired_cursor_checksum"]:
            if not receipt_path.is_file() or int(current.get("revision", -1)) < int(
                expected["revision"]
            ):
                raise JournalIntegrityError("parent cursor escaped T10.2.5 repair")
        if current.get("cursor_checksum") == repair["repaired_cursor_checksum"]:
            payload = signed_payload(
                {
                    "format_version": "sage-t10.2.5-parent-cursor-repair-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "migration_receipt_checksum": manifest["migration_receipt"][
                        "receipt_checksum"
                    ],
                    "old_cursor_checksum": repair["old_cursor_checksum"],
                    "repaired_cursor_checksum": repair[
                        "repaired_cursor_checksum"
                    ],
                    "physical_records_changed": False,
                    "replayed_physical_actions": 0,
                },
                checksum_key="repair_checksum",
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            _write_once_payload(
                receipt_path, payload, checksum_key="repair_checksum"
            )
        elif not receipt_path.is_file():
            raise JournalIntegrityError("advanced cursor lacks its T10.2.5 repair receipt")
        return _kernel_protocol._read_signed_json(
            receipt_path, checksum_key="repair_checksum"
        )
    finally:
        lease.release()


def _seal_parent_orphan_lane(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    kernel: Mapping[str, Any],
    kernel_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Append terminal reports for the superseded lane without new actions."""

    destination = root / artifact_root
    receipt_path = (
        root / _protocol.DEFAULT_RECOVERY_ROOT / ORPHAN_CLOSURE_RECEIPT_FILENAME
    )
    lease = _kernel_runtime._CollectionLease.acquire(
        destination / ".active-collector.lock"
    )
    try:
        relative_kernel = kernel_path.relative_to(root)
        with (
            _parent_protocol.kernel_protocol_bindings(
                artifact_root=artifact_root,
                manifest_relative_path=relative_kernel,
                mode="full",
            ),
            _parent_runtime.execution_bindings(
                mode="full", artifact_root=artifact_root
            ),
        ):
            journal = _parent_runtime.IncrementalDurableCollectionJournal(
                destination / _kernel_runtime.JOURNAL_DIRECTORY_NAME,
                manifest_checksum=str(kernel["manifest_checksum"]),
            )
            lane = _kernel_runtime.SourceLaneKey.from_dict(
                manifest["migration_receipt"]["orphan_lane"]
            )
            orphan_work_id = str(
                manifest["migration_receipt"]["orphan_reset"]["work"]["work_id"]
            )
            works = list(_kernel_runtime.reset_work_specs(lane))
            orphan_work = next(item for item in works if item.work_id == orphan_work_id)
            existing_lane = journal.read_lane_report(lane)
            if existing_lane is None:
                existing_orphan = journal.read_reset_report(orphan_work)
                if existing_orphan is None:
                    prior = _kernel_runtime._continuation_before(journal, orphan_work)
                    _kernel_runtime._attest_unresolved_after_worker(
                        journal,
                        orphan_work,
                        reason="parent_interrupted",
                    )
                    existing_orphan = _kernel_runtime._reset_report_from_outcome(
                        journal=journal,
                        work=orphan_work,
                        outcome=_kernel_runtime.WorkerOutcome(
                            status="FAILED",
                            elapsed_seconds=0.0,
                            payload={
                                "completed": False,
                                "stop_reason": "interrupted_before_reset_commit",
                                "continuation": prior,
                            },
                            error_kind="ParentInterrupted",
                        ),
                        prior_continuation=prior,
                    )
                if (
                    existing_orphan.status != "ABORTED"
                    or existing_orphan.stop_reason
                    != "interrupted_before_reset_commit"
                ):
                    raise JournalIntegrityError("orphan reset closure drifted")
                trailing = [
                    item for item in works if item.reset_index > orphan_work.reset_index
                ]
                if len(trailing) != 1:
                    raise ManifestDriftError("orphan lane trailing reset registry drifted")
                trailing_work = trailing[0]
                trailing_report = journal.read_reset_report(trailing_work)
                if trailing_report is None:
                    if journal.reset_accounting(trailing_work).authorized_intent_count:
                        raise JournalIntegrityError(
                            "unstarted orphan trailing reset acquired actions"
                        )
                    prior = _kernel_runtime._continuation_before(
                        journal, trailing_work
                    )
                    trailing_report = _kernel_runtime._reset_report_from_outcome(
                        journal=journal,
                        work=trailing_work,
                        outcome=_kernel_runtime.WorkerOutcome(
                            status="COOPERATIVE_STOP",
                            elapsed_seconds=0.0,
                            payload={
                                "completed": False,
                                "stop_reason": "parent_interrupted",
                                "continuation": prior,
                            },
                            error_kind="T10_2_5LaneSuperseded",
                        ),
                        prior_continuation=prior,
                    )
                if (
                    trailing_report.status != "ABORTED"
                    or trailing_report.issued_intents != 0
                    or trailing_report.stop_reason != "parent_interrupted"
                ):
                    raise JournalIntegrityError("trailing reset closure drifted")
                checkpoint = journal.load_checkpoint()
                elapsed = (
                    0.0
                    if checkpoint is None
                    else float(checkpoint.open_lane_elapsed_seconds)
                )
                existing_lane = _kernel_runtime._finalize_lane_report(
                    journal,
                    lane,
                    elapsed_seconds=elapsed,
                    timing_admissible=False,
                )
                journal.reconstruct_checkpoint(
                    cumulative_active_seconds=(
                        0.0
                        if checkpoint is None
                        else float(checkpoint.cumulative_active_seconds)
                    ),
                    close_open_lane=True,
                )
            if existing_lane.status != "ABORTED" or len(existing_lane.resets) != 4:
                raise JournalIntegrityError("orphan lane closure is not fail-closed")
            payload = signed_payload(
                {
                    "format_version": "sage-t10.2.5-parent-orphan-closure-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "migration_receipt_checksum": manifest["migration_receipt"][
                        "receipt_checksum"
                    ],
                    "lane_id": lane.lane_id,
                    "lane_report_checksum": existing_lane.report_checksum,
                    "orphan_reset_work_id": orphan_work_id,
                    "orphan_reset_report_checksum": next(
                        item.report_checksum
                        for item in existing_lane.resets
                        if item.work.work_id == orphan_work_id
                    ),
                    "new_physical_actions": 0,
                    "replayed_physical_actions": 0,
                    "lane_enters_model_fit": False,
                },
                checksum_key="closure_checksum",
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            _write_once_payload(
                receipt_path, payload, checksum_key="closure_checksum"
            )
            return payload
    finally:
        lease.release()


def _resume_parent_collection(
    *, root: Path, artifact_root: Path
) -> dict[str, Any]:
    report_path = root / artifact_root / _kernel_runtime.COLLECTION_REPORT_FILENAME
    if not report_path.is_file():
        with worker_only_watchdog_binding():
            _predecessor_runtime.collect_phase(
                manifest_path=root
                / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
                output_dir=artifact_root,
                repo_root=root,
            )
    return _kernel_protocol._read_signed_json(
        report_path, checksum_key="report_checksum"
    )


def _parent_state(
    *,
    root: Path,
    kernel: Mapping[str, Any],
    kernel_path: Path,
    artifact_root: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    relative_kernel = kernel_path.relative_to(root)
    with (
        _parent_protocol.kernel_protocol_bindings(
            artifact_root=artifact_root,
            manifest_relative_path=relative_kernel,
            mode="full",
        ),
        _parent_runtime.execution_bindings(mode="full", artifact_root=artifact_root),
    ):
        journal = _parent_runtime.IncrementalDurableCollectionJournal(
            root / artifact_root / _kernel_runtime.JOURNAL_DIRECTORY_NAME,
            manifest_checksum=str(kernel["manifest_checksum"]),
        )
        schedule = list(_parent_runtime._execution_lanes("full"))
        reports = list(journal.lane_reports())
        if len(reports) != len(schedule):
            raise ManifestDriftError("parent continuation did not reach all 18 lanes")
        by_id = {report.lane.lane_id: report for report in reports}
        orphan_lane_id = str(receipt["orphan_lane"]["lane_id"])
        orphan = by_id.get(orphan_lane_id)
        if orphan is None or orphan.status != "ABORTED":
            raise ManifestDriftError("parent orphan lane was not fail-closed")
        orphan_work_id = str(receipt["orphan_reset"]["work"]["work_id"])
        orphan_resets = [item for item in orphan.resets if item.work.work_id == orphan_work_id]
        if len(orphan_resets) != 1 or orphan_resets[0].status != "ABORTED":
            raise ManifestDriftError("parent orphan reset was not attested")
        if orphan_resets[0].stop_reason != "interrupted_before_reset_commit":
            raise ManifestDriftError("parent orphan stop reason drifted")
        complete_lanes = [report for report in reports if report.status == "COMPLETE"]
        complete_resets = [
            reset
            for report in reports
            for reset in report.resets
            if reset.status == "COMPLETE"
        ]
        all_resets = [reset for report in reports for reset in report.resets]
        if len(complete_lanes) != 17 or len(complete_resets) != 70 or len(all_resets) != 72:
            raise ManifestDriftError(
                "parent continuation contains failures beyond the registered orphan"
            )
        accounting = journal.accounting()
        if (
            accounting.unknown_intent_count
            or accounting.explicitly_unresolved_intent_count
            or not accounting.equation_holds
        ):
            raise JournalIntegrityError("parent terminal accounting is not closed")
        accepted_events_by_lane: dict[str, list[dict[str, Any]]] = {}
        for report in complete_lanes:
            accepted_events_by_lane[report.lane.lane_id] = [
                dict(event)
                for reset in report.resets
                for event in journal.events_for_reset(reset.work)
            ]
        orphan_events = [
            dict(event)
            for reset in orphan.resets
            for event in journal.events_for_reset(reset.work)
        ]
        discovery_events = list(journal.completed_discovery_events())
        return {
            "schedule": schedule,
            "reports": reports,
            "by_id": by_id,
            "complete_lanes": complete_lanes,
            "accepted_events_by_lane": accepted_events_by_lane,
            "orphan": orphan,
            "orphan_events": orphan_events,
            "discovery_events": discovery_events,
            "accounting": accounting.to_dict(),
            "physical_steps_replayed_on_resume": (
                journal.load_checkpoint().physical_steps_replayed_on_resume
            ),
        }


@contextmanager
def recovery_journal_bindings(receipt: Mapping[str, Any]):
    seeds = tuple(int(item) for item in receipt["recovery_seeds"])
    original_seeds = _kernel_runtime.CONFIRMATION_SEEDS
    original_registry = _kernel_runtime.source_lane_registry
    original_cap = _kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS
    _kernel_runtime.CONFIRMATION_SEEDS = (*original_seeds, *seeds)
    lanes = tuple(
        _kernel_runtime.SourceLaneKey.from_dict(item)
        for item in receipt["recovery_lanes"]
    )
    _kernel_runtime.source_lane_registry = lambda: lanes
    _kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS = (
        _protocol.RECOVERY_MAXIMUM_ACTIONS
    )
    try:
        yield lanes
    finally:
        _kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS = original_cap
        _kernel_runtime.source_lane_registry = original_registry
        _kernel_runtime.CONFIRMATION_SEEDS = original_seeds


def _prior_continuation(journal: Any, work: Any) -> dict[str, Any]:
    continuation: dict[str, Any] = {}
    for prior in _kernel_runtime.reset_work_specs(work.lane):
        if prior.reset_index >= work.reset_index:
            break
        report = journal.read_reset_report(prior)
        if report is None:
            raise JournalIntegrityError("recovery reset predecessor is missing")
        if report.status != "COMPLETE":
            raise JournalIntegrityError("recovery continuation crossed a failed reset")
        continuation = dict(report.continuation)
    return continuation


def _finalize_recovery_lane(
    journal: Any,
    lane: Any,
    *,
    discovery_events: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> Any:
    resets = tuple(
        report
        for work in _kernel_runtime.reset_work_specs(lane)
        if (report := journal.read_reset_report(work)) is not None
    )
    complete = bool(
        len(resets) == _protocol.RECOVERY_RESETS_PER_LANE
        and all(report.status == "COMPLETE" for report in resets)
        and not any(report.unresolved_intents for report in resets)
    )
    report = _kernel_runtime.LaneReport(
        lane=lane,
        status="COMPLETE" if complete else "ABORTED",
        resets=resets,
        issued_intents=sum(item.issued_intents for item in resets),
        sealed_events=sum(item.sealed_events for item in resets),
        unresolved_intents=sum(item.unresolved_intents for item in resets),
        elapsed_seconds=max(
            float(elapsed_seconds), sum(item.elapsed_seconds for item in resets)
        ),
        cross_fit_unit=(
            _kernel_runtime._cross_fit_unit_for_lane(
                lane, resets, discovery_events
            )
            if complete
            else None
        ),
    )
    journal.write_lane_report(report)
    return report


def _recovery_factory(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    kernel: Mapping[str, Any],
) -> RecoveryDualCachedSourceFactory:
    return RecoveryDualCachedSourceFactory(
        manifest=manifest,
        cache_root=root / _predecessor_protocol.DEFAULT_CACHE_ROOT,
        predecessor_cache_root=root
        / _predecessor_runtime._predecessor_protocol.DEFAULT_CACHE_ROOT,
        continuation_manifest_checksum=str(predecessor["manifest_checksum"]),
        predecessor_manifest_checksum=str(
            predecessor["predecessor_t10_2_3_manifest_checksum"]
        ),
        cache_kernel_manifest_checksum=str(kernel["manifest_checksum"]),
    )


def _collect_recovery(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    kernel: Mapping[str, Any],
    discovery_events: Sequence[Mapping[str, Any]],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    destination = root / _protocol.DEFAULT_RECOVERY_ROOT
    report_path = destination / RECOVERY_REPORT_FILENAME
    if report_path.is_file():
        return _kernel_protocol._read_signed_json(
            report_path, checksum_key="report_checksum"
        )
    destination.mkdir(parents=True, exist_ok=True)
    lease = _kernel_runtime._CollectionLease.acquire(
        destination / ".active-recovery.lock"
    )
    try:
        with recovery_journal_bindings(manifest["migration_receipt"]) as lanes:
            journal = _kernel_runtime.DurableCollectionJournal(
                destination / RECOVERY_JOURNAL_DIRECTORY,
                manifest_checksum=str(manifest["manifest_checksum"]),
            )
            factory = _recovery_factory(
                root=root,
                manifest=manifest,
                predecessor=predecessor,
                kernel=kernel,
            )
            accepted: Any | None = None
            recovery_started = float(clock())
            with (
                worker_only_watchdog_binding(),
                _predecessor_runtime.dual_cache_worker_binding(),
            ):
                for attempt_index, lane in enumerate(lanes):
                    existing_lane = journal.read_lane_report(lane)
                    if existing_lane is not None:
                        if existing_lane.status == "COMPLETE":
                            accepted = existing_lane
                            break
                        continue
                    lane_started = float(clock())
                    lane_failed = False
                    for work in _kernel_runtime.reset_work_specs(lane):
                        existing_reset = journal.read_reset_report(work)
                        if existing_reset is not None:
                            if existing_reset.status == "COMPLETE":
                                continue
                            lane_failed = True
                            break
                        partial = journal.reset_accounting(work)
                        prior = _prior_continuation(journal, work)
                        if partial.authorized_intent_count:
                            _kernel_runtime._attest_unresolved_after_worker(
                                journal,
                                work,
                                reason="parent_interrupted",
                            )
                            outcome = _kernel_runtime.WorkerOutcome(
                                status="FAILED",
                                elapsed_seconds=0.0,
                                payload={
                                    "completed": False,
                                    "stop_reason": "interrupted_before_reset_commit",
                                    "continuation": prior,
                                },
                                error_kind="ParentInterrupted",
                            )
                        else:
                            lane_elapsed = max(0.0, float(clock()) - lane_started)
                            total_elapsed = max(
                                0.0, float(clock()) - recovery_started
                            )
                            print(
                                canonical_json(
                                    {
                                        "phase": "t10_2_5_recovery_reset",
                                        "attempt_index": attempt_index,
                                        "lane_id": lane.lane_id,
                                        "game_id": lane.game_id,
                                        "seed": lane.seed,
                                        "reset_index": work.reset_index,
                                        "controller": work.controller,
                                        "collector_pid": os.getpid(),
                                        "watchdog_scope": "worker_process_tree_only",
                                    }
                                ),
                                flush=True,
                            )
                            outcome = factory.run_reset(
                                work=work,
                                journal=journal,
                                discovery_events=discovery_events,
                                continuation=prior,
                                process_context=multiprocessing.get_context("spawn"),
                                lane_remaining_seconds=(
                                    _parent_protocol.LANE_LIVENESS_WALL_SECONDS
                                    - lane_elapsed
                                ),
                                cooperative_collection_remaining_seconds=(
                                    _protocol.MAXIMUM_RECOVERY_LANES
                                    * _parent_protocol.LANE_LIVENESS_WALL_SECONDS
                                    - total_elapsed
                                ),
                                absolute_collection_remaining_seconds=(
                                    _protocol.MAXIMUM_RECOVERY_LANES
                                    * _parent_protocol.LANE_LIVENESS_WALL_SECONDS
                                    - total_elapsed
                                ),
                                clock=clock,
                            )
                        reset_report = _kernel_runtime._reset_report_from_outcome(
                            journal=journal,
                            work=work,
                            outcome=outcome,
                            prior_continuation=prior,
                        )
                        journal.reconstruct_checkpoint(
                            cumulative_active_seconds=max(
                                0.0, float(clock()) - recovery_started
                            ),
                            open_lane=lane,
                            open_lane_elapsed_seconds=max(
                                0.0, float(clock()) - lane_started
                            ),
                        )
                        if reset_report.status != "COMPLETE":
                            lane_failed = True
                            break
                    lane_report = _finalize_recovery_lane(
                        journal,
                        lane,
                        discovery_events=discovery_events,
                        elapsed_seconds=max(0.0, float(clock()) - lane_started),
                    )
                    journal.reconstruct_checkpoint(
                        cumulative_active_seconds=max(
                            0.0, float(clock()) - recovery_started
                        ),
                        close_open_lane=True,
                    )
                    if lane_report.status == "COMPLETE" and not lane_failed:
                        accepted = lane_report
                        break
            lane_reports = list(journal.lane_reports())
            accounting = journal.accounting()
            if accounting.unknown_intent_count or not accounting.equation_holds:
                raise JournalIntegrityError("recovery journal accounting is open")
            payload = signed_payload(
                {
                    "format_version": RECOVERY_REPORT_FORMAT_VERSION,
                    "phase": "recovery",
                    "status": (
                        "PASS_T10_2_5_RECOVERY"
                        if accepted is not None
                        else "FAIL_T10_2_5_RECOVERY"
                    ),
                    "manifest_checksum": manifest["manifest_checksum"],
                    "migration_receipt_checksum": manifest["migration_receipt"][
                        "receipt_checksum"
                    ],
                    "attempted_lane_count": len(lane_reports),
                    "maximum_recovery_lanes": _protocol.MAXIMUM_RECOVERY_LANES,
                    "accepted_lane": (
                        None
                        if accepted is None
                        else _protocol._lane_fingerprint(accepted)
                    ),
                    "attempted_lanes": [
                        _protocol._lane_fingerprint(item) for item in lane_reports
                    ],
                    "accounting": accounting.to_dict(),
                    "physical_steps_replayed": 0,
                    "orphan_events_replayed": 0,
                    "watchdog_scope": "worker_process_tree_only",
                    "firewall": {
                        "source_validation_opened": False,
                        "ar25_opened": False,
                        "holdout_opened": False,
                        "production_authority": False,
                    },
                },
                checksum_key="report_checksum",
            )
            _write_once_payload(report_path, payload, checksum_key="report_checksum")
            return payload
    finally:
        lease.release()


def _recovery_state(
    *, root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    destination = root / _protocol.DEFAULT_RECOVERY_ROOT
    with recovery_journal_bindings(manifest["migration_receipt"]):
        journal = _kernel_runtime.DurableCollectionJournal(
            destination / RECOVERY_JOURNAL_DIRECTORY,
            manifest_checksum=str(manifest["manifest_checksum"]),
        )
        reports = list(journal.lane_reports())
        accepted = next((item for item in reports if item.status == "COMPLETE"), None)
        if accepted is None:
            raise ManifestDriftError("T10.2.5 recovery has no accepted lane")
        events = [
            dict(event)
            for reset in accepted.resets
            for event in journal.events_for_reset(reset.work)
        ]
        return {
            "journal": journal,
            "reports": reports,
            "accepted": accepted,
            "events": events,
            "accounting": journal.accounting().to_dict(),
            "physical_steps_replayed_on_resume": (
                journal.load_checkpoint().physical_steps_replayed_on_resume
            ),
        }


def _accepted_cross_fit_audit(
    *,
    manifest: Mapping[str, Any],
    parent_state: Mapping[str, Any],
    recovery_state: Mapping[str, Any],
    accepted_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    orphan_lane = manifest["migration_receipt"]["orphan_lane"]
    recovery_lane = recovery_state["accepted"].lane
    units: list[dict[str, Any]] = []
    for report in parent_state["complete_lanes"]:
        if report.cross_fit_unit is not None:
            units.append(dict(report.cross_fit_unit))
    replacement_unit = dict(recovery_state["accepted"].cross_fit_unit)
    replacement_unit["physical_recovery_seed"] = replacement_unit["seed"]
    replacement_unit["seed"] = orphan_lane["seed"]
    replacement_unit["logical_replacement_for_lane_id"] = orphan_lane["lane_id"]
    units.append(replacement_unit)
    source_games = set(_kernel_protocol.SOURCE_GAMES)
    donor_isolation = all(
        set(unit.get("training_games", ()))
        == source_games - {str(unit.get("held_out_game", ""))}
        and int(unit.get("held_out_prefit_events_used", -1)) == 0
        for unit in units
    )
    recovery_controllers = [
        reset.work.controller for reset in recovery_state["accepted"].resets
    ]
    expected_controllers = list(
        _kernel_runtime.confirmation_controller_order(int(orphan_lane["seed"]))
    )
    event_ids = [str(event.get("event_id", "")) for event in accepted_events]
    checks = {
        "logical_confirmation_unit_count": len(units) == 9,
        "recovery_controller_order_preserved": recovery_controllers
        == expected_controllers,
        "recovery_held_out_game_preserved": recovery_lane.game_id
        == orphan_lane["game_id"],
        "donor_isolation": donor_isolation,
        "held_out_prefit_events_excluded": all(
            int(unit.get("held_out_prefit_events_used", -1)) == 0 for unit in units
        ),
        "accepted_event_ids_unique": bool(event_ids)
        and len(event_ids) == len(set(event_ids)),
    }
    return signed_payload(
        {
            "format_version": ACCEPTED_AUDIT_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "orphan_lane": dict(orphan_lane),
            "recovery_lane": recovery_lane.to_dict(),
            "units": units,
            "checks": checks,
            "passed": all(checks.values()),
        },
        checksum_key="audit_checksum",
    )


def _build_accepted_collection(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    parent_report: Mapping[str, Any],
    parent_state: Mapping[str, Any],
    recovery_report: Mapping[str, Any],
    recovery_state: Mapping[str, Any],
) -> dict[str, Any]:
    if recovery_report.get("status") != "PASS_T10_2_5_RECOVERY":
        raise ManifestDriftError("T10.2.5 recovery did not produce an accepted lane")
    destination = root / _protocol.DEFAULT_RECOVERY_ROOT
    orphan_lane_id = str(manifest["migration_receipt"]["orphan_lane"]["lane_id"])
    accepted_events: list[dict[str, Any]] = []
    accepted_lane_reports: list[dict[str, Any]] = []
    for lane in parent_state["schedule"]:
        if lane.lane_id == orphan_lane_id:
            accepted_events.extend(dict(item) for item in recovery_state["events"])
            fingerprint = _protocol._lane_fingerprint(recovery_state["accepted"])
            fingerprint["logical_lane"] = dict(
                manifest["migration_receipt"]["orphan_lane"]
            )
            fingerprint["physical_recovery_lane"] = recovery_state[
                "accepted"
            ].lane.to_dict()
            accepted_lane_reports.append(fingerprint)
            continue
        report = parent_state["by_id"].get(lane.lane_id)
        if report is None or report.status != "COMPLETE":
            raise ManifestDriftError("a non-orphan parent lane is incomplete")
        accepted_events.extend(parent_state["accepted_events_by_lane"][lane.lane_id])
        accepted_lane_reports.append(_protocol._lane_fingerprint(report))
    event_path = destination / ACCEPTED_EVENT_FILENAME
    _kernel_runtime._write_event_ledger(event_path, accepted_events)
    audit = _accepted_cross_fit_audit(
        manifest=manifest,
        parent_state=parent_state,
        recovery_state=recovery_state,
        accepted_events=accepted_events,
    )
    audit_path = destination / ACCEPTED_AUDIT_FILENAME
    _write_once_payload(audit_path, audit, checksum_key="audit_checksum")
    accepted_reset_count = sum(len(item["resets"]) for item in accepted_lane_reports)
    accepted_sealed = sum(int(item["sealed_events"]) for item in accepted_lane_reports)
    parent_accounting = parent_state["accounting"]
    recovery_accounting = recovery_state["accounting"]
    attempted_authorized = int(parent_accounting["authorized_intent_count"]) + int(
        recovery_accounting["authorized_intent_count"]
    )
    attempted_sealed = int(parent_accounting["sealed_event_count"]) + int(
        recovery_accounting["sealed_event_count"]
    )
    attempted_unresolved = int(
        parent_accounting["explicitly_unresolved_intent_count"]
    ) + int(recovery_accounting["explicitly_unresolved_intent_count"])
    excluded_sealed = attempted_sealed - accepted_sealed
    event_ids = [str(event.get("event_id", "")) for event in accepted_events]
    orphan = manifest["migration_receipt"]["orphan_lane"]
    orphan_events_excluded = all(
        not (
            str(event.get("game_id", "")) == orphan["game_id"]
            and int(event.get("seed", -1)) == int(orphan["seed"])
            and str(event.get("split", "")) == orphan["split"]
        )
        for event in accepted_events
    )
    checks = {
        "parent_only_registered_orphan_failed": len(parent_state["complete_lanes"])
        == 17,
        "accepted_logical_lane_count": len(accepted_lane_reports) == 18,
        "accepted_complete_reset_count": accepted_reset_count == 72,
        "accepted_event_count_matches_reports": len(accepted_events)
        == accepted_sealed,
        "accepted_event_ids_unique": bool(event_ids)
        and len(event_ids) == len(set(event_ids)),
        "attempted_action_equation_holds": attempted_authorized
        == attempted_sealed + attempted_unresolved,
        "excluded_actions_accounted": excluded_sealed >= len(
            parent_state["orphan_events"]
        ),
        "orphan_lane_events_excluded": orphan_events_excluded,
        "cross_fit_audit_passed": audit["passed"] is True,
        "parent_physical_actions_not_replayed": parent_state[
            "physical_steps_replayed_on_resume"
        ]
        == 0,
        "recovery_physical_actions_not_replayed": recovery_state[
            "physical_steps_replayed_on_resume"
        ]
        == 0,
        "authority_closed": True,
    }
    payload = signed_payload(
        {
            "format_version": COLLECTION_REPORT_FORMAT_VERSION,
            "phase": "collect",
            "status": (
                "T10_2_5_SOURCE_COLLECTION_COMPLETE"
                if all(checks.values())
                else "DATA_OR_PROVENANCE_INVALID"
            ),
            "manifest_checksum": manifest["manifest_checksum"],
            "migration_receipt_checksum": manifest["migration_receipt"][
                "receipt_checksum"
            ],
            "parent_collection_report_checksum": parent_report["report_checksum"],
            "recovery_report_checksum": recovery_report["report_checksum"],
            "accepted_event_count": len(accepted_events),
            "accepted_lane_count": len(accepted_lane_reports),
            "accepted_reset_count": accepted_reset_count,
            "accepted_lanes": accepted_lane_reports,
            "accepted_events": _kernel_protocol._t10_2.artifact_descriptor(event_path),
            "accepted_cross_fit_audit": _kernel_protocol._t10_2.artifact_descriptor(
                audit_path
            ),
            "action_accounting": {
                "attempted_authorized_intents": attempted_authorized,
                "attempted_sealed_events": attempted_sealed,
                "attempted_explicitly_unresolved_intents": attempted_unresolved,
                "accepted_sealed_events": accepted_sealed,
                "excluded_sealed_events": excluded_sealed,
                "equation_holds": attempted_authorized
                == attempted_sealed + attempted_unresolved,
            },
            "checks": checks,
            "passed": all(checks.values()),
            "replayed_physical_actions": 0,
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="report_checksum",
    )
    report_path = destination / COLLECTION_REPORT_FILENAME
    _write_once_payload(report_path, payload, checksum_key="report_checksum")
    return payload


def collect_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    (
        root,
        manifest,
        predecessor,
        _parent,
        kernel,
        kernel_path,
        artifact_root,
    ) = _load_execution_context(manifest_path=manifest_path, repo_root=repo_root)
    _repair_parent_cursor(
        root=root,
        manifest=manifest,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
    )
    _seal_parent_orphan_lane(
        root=root,
        manifest=manifest,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
    )
    parent_report = _resume_parent_collection(root=root, artifact_root=artifact_root)
    parent_state = _parent_state(
        root=root,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
        receipt=manifest["migration_receipt"],
    )
    recovery_report = _collect_recovery(
        root=root,
        manifest=manifest,
        predecessor=predecessor,
        kernel=kernel,
        discovery_events=parent_state["discovery_events"],
    )
    if recovery_report.get("status") != "PASS_T10_2_5_RECOVERY":
        return recovery_report
    recovery_state = _recovery_state(root=root, manifest=manifest)
    return _build_accepted_collection(
        root=root,
        manifest=manifest,
        parent_report=parent_report,
        parent_state=parent_state,
        recovery_report=recovery_report,
        recovery_state=recovery_state,
    )


def status_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root, manifest, *_ = _load_execution_context(
        manifest_path=manifest_path, repo_root=repo_root
    )
    destination = root / _protocol.DEFAULT_RECOVERY_ROOT
    collection_path = destination / COLLECTION_REPORT_FILENAME
    recovery_path = destination / RECOVERY_REPORT_FILENAME
    return {
        "status": (
            "COMPLETE_T10_2_5_COLLECTION"
            if collection_path.is_file()
            else "RECOVERY_T10_2_5_IN_PROGRESS"
            if recovery_path.is_file()
            else "READY_T10_2_5_RECOVERY"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "migration": _protocol.verify_migration_receipt_live(
            manifest["migration_receipt"], repo_root=root
        ),
        "recovery_report_present": recovery_path.is_file(),
        "collection_report_present": collection_path.is_file(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("status", "collect"))
    parser.add_argument("--manifest", default=str(_protocol.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = (
            status_phase(manifest_path=args.manifest, repo_root=args.repo_root)
            if args.phase == "status"
            else collect_phase(manifest_path=args.manifest, repo_root=args.repo_root)
        )
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
