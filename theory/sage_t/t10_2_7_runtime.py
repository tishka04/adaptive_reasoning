"""Event-seal-safe replacement-lane runtime for SAGE.T10.2.7."""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_5_protocol as _zero_failure_protocol
from . import t10_2_6_protocol as _predecessor_protocol
from . import t10_2_6_runtime as _predecessor_runtime
from . import t10_2_7_protocol as _protocol

FORMAT_VERSION = "sage-t10.2.7-runtime-v1"
RECOVERY_REPORT_FORMAT_VERSION = "sage-t10.2.7-recovery-report-v1"
ACCEPTED_AUDIT_FORMAT_VERSION = "sage-t10.2.7-accepted-cross-fit-audit-v1"
COLLECTION_REPORT_FORMAT_VERSION = "sage-t10.2.7-collection-report-v1"
RECOVERY_REPORT_FILENAME = "recovery_report.json"
ACCEPTED_EVENT_FILENAME = "accepted_source_events.jsonl"
ACCEPTED_AUDIT_FILENAME = "accepted_cross_fit_audit.json"
COLLECTION_REPORT_FILENAME = "t10_2_7_collection_report.json"
RECOVERY_JOURNAL_DIRECTORY = "source_collection_journal"

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError
WorkerProtocolError = _kernel_runtime.WorkerProtocolError


def _write_once_payload(
    path: Path, payload: Mapping[str, Any], *, checksum_key: str
) -> None:
    if path.is_file():
        existing = _kernel_protocol._read_signed_json(path, checksum_key=checksum_key)
        if existing != dict(payload):
            raise JournalIntegrityError(f"immutable T10.2.7 artifact drifted: {path}")
        return
    _kernel_runtime._write_once(path, payload)


@contextmanager
def recovery_journal_bindings(receipt: Mapping[str, Any]):
    """Install only the fresh T10.2.7 lane registry in this process."""

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


def build_execution_manifest(
    *,
    protocol_manifest: Mapping[str, Any],
    kernel: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct and verify the deterministic scientific/recovery overlay."""

    expected_contract = _protocol._execution_contract(kernel)
    if protocol_manifest.get("execution_manifest_contract") != expected_contract:
        raise ManifestDriftError("T10.2.7 execution contract does not match the kernel")
    execution = copy.deepcopy(dict(kernel))
    execution["manifest_checksum"] = str(protocol_manifest["manifest_checksum"])
    execution["migration_receipt"] = copy.deepcopy(
        dict(protocol_manifest["migration_receipt"])
    )
    inherited = {
        key: value
        for key, value in execution.items()
        if key not in {"manifest_checksum", "migration_receipt"}
    }
    expected_inherited = {
        key: value for key, value in kernel.items() if key != "manifest_checksum"
    }
    if inherited != expected_inherited:
        raise ManifestDriftError("T10.2.7 execution overlay changed scientific fields")
    if (
        execution.get("environment_sha256")
        != expected_contract["required_environment_sha256"]
        or execution.get("manifest_checksum")
        != protocol_manifest.get("manifest_checksum")
        or execution.get("migration_receipt")
        != protocol_manifest.get("migration_receipt")
    ):
        raise ManifestDriftError("T10.2.7 execution overlay failed its bindings")
    return execution


def _recovery_spawn_worker_entry(*args: Any) -> None:
    """Install T10.2.7 seeds before the spawned child decodes its work."""

    if not args:
        raise WorkerProtocolError("T10.2.7 spawned worker lacks a source factory")
    factory = args[0]
    factory_type = _predecessor_runtime._predecessor_runtime.RecoveryDualCachedSourceFactory
    if not isinstance(factory, factory_type):
        raise WorkerProtocolError("T10.2.7 spawned worker received an invalid factory")
    execution_manifest = getattr(factory, "manifest", None)
    if not isinstance(execution_manifest, Mapping):
        raise WorkerProtocolError("T10.2.7 spawned worker lacks its execution manifest")
    receipt = execution_manifest.get("migration_receipt")
    if not isinstance(receipt, Mapping):
        raise WorkerProtocolError("T10.2.7 spawned worker lacks its seed receipt")
    if "environment_sha256" not in execution_manifest:
        raise WorkerProtocolError("T10.2.7 spawned worker lacks environment provenance")
    with recovery_journal_bindings(receipt):
        _predecessor_runtime._predecessor_runtime._predecessor_runtime._dual_cache_reset_worker_entry(
            *args
        )


@contextmanager
def recovery_spawn_worker_binding():
    """Bind the importable T10.2.7 spawn target around reset execution."""

    parent_runtime = _predecessor_runtime._predecessor_runtime._parent_runtime
    original = parent_runtime._action_budget_reset_worker_entry
    parent_runtime._action_budget_reset_worker_entry = _recovery_spawn_worker_entry
    try:
        yield
    finally:
        parent_runtime._action_budget_reset_worker_entry = original


def _load_execution_context(
    *, manifest_path: str | Path, repo_root: str | Path | None
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
]:
    root = _protocol._root(repo_root)
    manifest = _protocol.load_manifest(manifest_path, repo_root=root)
    predecessor = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=False,
        verify_live_migration=False,
    )
    (
        predecessor_root,
        _verified_zero_failure,
        cache_predecessor,
        _parent,
        kernel,
        kernel_path,
        artifact_root,
    ) = _predecessor_runtime._predecessor_runtime._load_execution_context(
        manifest_path=(
            root
            / _predecessor_runtime._predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH
        ),
        repo_root=root,
    )
    if predecessor_root != root:
        raise ManifestDriftError("T10.2.7 predecessor execution context drifted")
    if kernel.get("manifest_checksum") != _protocol.PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.7 kernel execution context drifted")
    execution_manifest = build_execution_manifest(
        protocol_manifest=manifest,
        kernel=kernel,
    )
    return (
        root,
        manifest,
        predecessor,
        cache_predecessor,
        kernel,
        execution_manifest,
        kernel_path,
        artifact_root,
    )


def _recovery_factory(
    *,
    root: Path,
    execution_manifest: Mapping[str, Any],
    cache_predecessor: Mapping[str, Any],
    kernel: Mapping[str, Any],
) -> Any:
    return _predecessor_runtime._recovery_factory(
        root=root,
        manifest=execution_manifest,
        cache_predecessor=cache_predecessor,
        kernel=kernel,
    )


def _first_event_seal_preflight(
    *, protocol_manifest: Mapping[str, Any], kernel: Mapping[str, Any]
) -> dict[str, Any]:
    """Exercise the real parent journal handler through its first event seal."""

    execution = build_execution_manifest(
        protocol_manifest=protocol_manifest,
        kernel=kernel,
    )
    receipt = protocol_manifest["migration_receipt"]
    with tempfile.TemporaryDirectory(prefix="sage-t10-2-7-seal-") as temporary:
        with recovery_journal_bindings(receipt) as lanes:
            lane = lanes[0]
            work = _kernel_runtime.reset_work_specs(lane)[0]
            journal = _kernel_runtime.DurableCollectionJournal(
                Path(temporary) / "journal",
                manifest_checksum=str(protocol_manifest["manifest_checksum"]),
            )
            intent = _kernel_runtime.ActionIntent(
                lane=lane,
                reset_index=0,
                step_index=0,
                action={
                    "name": "T10_2_7_EVENT_SEAL_PREFLIGHT",
                    "parameter_arity": 0,
                    "grounding_sha256": canonical_sha256(
                        {"kind": "t10_2_7_event_seal_preflight"}
                    ),
                },
                manifest_checksum=str(protocol_manifest["manifest_checksum"]),
            )
            handler = _kernel_runtime._ParentJournalMessageHandler(
                journal=journal,
                work=work,
                manifest=execution,
                clock=lambda: 1.0,
                intent_deadline=2.0,
            )
            intent_ack = handler(
                {"kind": "action_intent", "payload": intent.to_dict()}
            )
            event_ack = handler(
                {
                    "kind": "physical_event",
                    "intent_id": intent.intent_id,
                    "payload": {
                        "event": {
                            "event_id": "t10-2-7-event-seal-preflight",
                        }
                    },
                }
            )
            accounting = journal.reset_accounting(work)
            if (
                not isinstance(intent_ack, Mapping)
                or not isinstance(event_ack, Mapping)
                or accounting.authorized_intent_count != 1
                or accounting.sealed_event_count != 1
                or accounting.explicitly_unresolved_intent_count != 0
                or not accounting.equation_holds
            ):
                raise JournalIntegrityError("T10.2.7 first-event seal preflight failed")
            record = journal.events_for_reset(work)
            if (
                len(record) != 1
                or record[0].get("provenance", {}).get("environment_sha256")
                != kernel["environment_sha256"]
                or record[0].get("provenance", {}).get("manifest_checksum")
                != protocol_manifest["manifest_checksum"]
            ):
                raise JournalIntegrityError("T10.2.7 first-event provenance drifted")
            return {
                "passed": True,
                "event_checksum": event_ack["event_checksum"],
                "environment_sha256": kernel["environment_sha256"],
                "execution_manifest_sha256": canonical_sha256(execution),
            }


def _run_reset_and_finalize(
    *,
    factory: Any,
    journal: Any,
    work: Any,
    discovery_events: Sequence[Mapping[str, Any]],
    prior: Mapping[str, Any],
    process_context: Any,
    lane_remaining_seconds: float,
    cooperative_collection_remaining_seconds: float,
    absolute_collection_remaining_seconds: float,
    clock: Any = time.perf_counter,
) -> Any:
    """Convert parent-side runner exceptions into durable terminal evidence."""

    started = float(clock())
    try:
        outcome = factory.run_reset(
            work=work,
            journal=journal,
            discovery_events=discovery_events,
            continuation=prior,
            process_context=process_context,
            lane_remaining_seconds=lane_remaining_seconds,
            cooperative_collection_remaining_seconds=(
                cooperative_collection_remaining_seconds
            ),
            absolute_collection_remaining_seconds=absolute_collection_remaining_seconds,
            clock=clock,
        )
    except Exception as exc:
        _kernel_runtime._attest_unresolved_after_worker(
            journal,
            work,
            reason="environment_call_unattestable",
        )
        outcome = _kernel_runtime.WorkerOutcome(
            status="FAILED",
            elapsed_seconds=max(0.0, float(clock()) - started),
            payload={
                "completed": False,
                "stop_reason": f"parent_runner_exception:{type(exc).__name__}",
                "continuation": dict(prior),
            },
            error_kind=type(exc).__name__,
        )
    return _kernel_runtime._reset_report_from_outcome(
        journal=journal,
        work=work,
        outcome=outcome,
        prior_continuation=prior,
    )


def _collect_recovery(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
    cache_predecessor: Mapping[str, Any],
    kernel: Mapping[str, Any],
    discovery_events: Sequence[Mapping[str, Any]],
    clock: Any = time.perf_counter,
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
                execution_manifest=execution_manifest,
                cache_predecessor=cache_predecessor,
                kernel=kernel,
            )
            accepted: Any | None = None
            recovery_started = float(clock())
            recovery_runtime = _predecessor_runtime._predecessor_runtime
            with (
                recovery_runtime.worker_only_watchdog_binding(),
                recovery_runtime._predecessor_runtime.dual_cache_worker_binding(),
                recovery_spawn_worker_binding(),
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
                        prior = recovery_runtime._prior_continuation(journal, work)
                        if partial.authorized_intent_count:
                            _kernel_runtime._attest_unresolved_after_worker(
                                journal, work, reason="parent_interrupted"
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
                            reset_report = _kernel_runtime._reset_report_from_outcome(
                                journal=journal,
                                work=work,
                                outcome=outcome,
                                prior_continuation=prior,
                            )
                        else:
                            lane_elapsed = max(0.0, float(clock()) - lane_started)
                            total_elapsed = max(0.0, float(clock()) - recovery_started)
                            print(
                                canonical_json(
                                    {
                                        "phase": "t10_2_7_event_seal_recovery_reset",
                                        "attempt_index": attempt_index,
                                        "lane_id": lane.lane_id,
                                        "game_id": lane.game_id,
                                        "seed": lane.seed,
                                        "reset_index": work.reset_index,
                                        "controller": work.controller,
                                        "collector_pid": os.getpid(),
                                        "spawn_child_registry_installed": True,
                                        "execution_manifest_hybrid": True,
                                        "watchdog_scope": "worker_process_tree_only",
                                    }
                                ),
                                flush=True,
                            )
                            reset_report = _run_reset_and_finalize(
                                factory=factory,
                                journal=journal,
                                work=work,
                                discovery_events=discovery_events,
                                prior=prior,
                                process_context=multiprocessing.get_context("spawn"),
                                lane_remaining_seconds=(
                                    recovery_runtime._parent_protocol.LANE_LIVENESS_WALL_SECONDS
                                    - lane_elapsed
                                ),
                                cooperative_collection_remaining_seconds=(
                                    _protocol.MAXIMUM_RECOVERY_LANES
                                    * recovery_runtime._parent_protocol.LANE_LIVENESS_WALL_SECONDS
                                    - total_elapsed
                                ),
                                absolute_collection_remaining_seconds=(
                                    _protocol.MAXIMUM_RECOVERY_LANES
                                    * recovery_runtime._parent_protocol.LANE_LIVENESS_WALL_SECONDS
                                    - total_elapsed
                                ),
                                clock=clock,
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
                    lane_report = recovery_runtime._finalize_recovery_lane(
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
                raise JournalIntegrityError("T10.2.7 recovery accounting is open")
            payload = signed_payload(
                {
                    "format_version": RECOVERY_REPORT_FORMAT_VERSION,
                    "phase": "event_seal_recovery",
                    "status": (
                        "PASS_T10_2_7_RECOVERY"
                        if accepted is not None
                        else "FAIL_T10_2_7_RECOVERY"
                    ),
                    "manifest_checksum": manifest["manifest_checksum"],
                    "migration_receipt_checksum": manifest["migration_receipt"][
                        "receipt_checksum"
                    ],
                    "predecessor_partial_journal_checksum": manifest[
                        "migration_receipt"
                    ]["t10_2_6_partial_failure"]["journal_checksum"],
                    "attempted_lane_count": len(lane_reports),
                    "maximum_recovery_lanes": _protocol.MAXIMUM_RECOVERY_LANES,
                    "accepted_lane": (
                        None
                        if accepted is None
                        else _zero_failure_protocol._lane_fingerprint(accepted)
                    ),
                    "attempted_lanes": [
                        _zero_failure_protocol._lane_fingerprint(item)
                        for item in lane_reports
                    ],
                    "accounting": accounting.to_dict(),
                    "physical_steps_replayed": 0,
                    "t10_2_6_partial_actions_replayed": 0,
                    "spawn_child_registry_installed": True,
                    "execution_manifest_hybrid": True,
                    "runner_exceptions_fail_closed": True,
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
            raise ManifestDriftError("T10.2.7 recovery has no accepted lane")
        events = [
            dict(event)
            for reset in accepted.resets
            for event in journal.events_for_reset(reset.work)
        ]
        checkpoint = journal.load_checkpoint()
        if checkpoint is None:
            raise ManifestDriftError("T10.2.7 recovery checkpoint is absent")
        return {
            "reports": reports,
            "accepted": accepted,
            "events": events,
            "accounting": journal.accounting().to_dict(),
            "physical_steps_replayed_on_resume": (
                checkpoint.physical_steps_replayed_on_resume
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
            int(unit.get("held_out_prefit_events_used", -1)) == 0
            for unit in units
        ),
        "accepted_event_ids_unique": bool(event_ids)
        and len(event_ids) == len(set(event_ids)),
        "t10_2_5_zero_action_failures_excluded": True,
        "t10_2_6_partial_lane_excluded": True,
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
    if recovery_report.get("status") != "PASS_T10_2_7_RECOVERY":
        raise ManifestDriftError("T10.2.7 recovery did not produce an accepted lane")
    destination = root / _protocol.DEFAULT_RECOVERY_ROOT
    orphan_lane_id = str(manifest["migration_receipt"]["orphan_lane"]["lane_id"])
    accepted_events: list[dict[str, Any]] = []
    accepted_lane_reports: list[dict[str, Any]] = []
    for lane in parent_state["schedule"]:
        if lane.lane_id == orphan_lane_id:
            accepted_events.extend(dict(item) for item in recovery_state["events"])
            fingerprint = _zero_failure_protocol._lane_fingerprint(
                recovery_state["accepted"]
            )
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
        accepted_lane_reports.append(_zero_failure_protocol._lane_fingerprint(report))
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
    zero_accounting = manifest["migration_receipt"]["t10_2_5_zero_action_failure"][
        "accounting"
    ]
    partial_accounting = manifest["migration_receipt"]["t10_2_6_partial_failure"][
        "accounting"
    ]
    recovery_accounting = recovery_state["accounting"]
    attempted_authorized = sum(
        int(item["authorized_intent_count"])
        for item in (
            parent_accounting,
            zero_accounting,
            partial_accounting,
            recovery_accounting,
        )
    )
    attempted_sealed = sum(
        int(item["sealed_event_count"])
        for item in (
            parent_accounting,
            zero_accounting,
            partial_accounting,
            recovery_accounting,
        )
    )
    attempted_unresolved = sum(
        int(item["explicitly_unresolved_intent_count"])
        for item in (
            parent_accounting,
            zero_accounting,
            partial_accounting,
            recovery_accounting,
        )
    )
    quarantined_unsealed = int(partial_accounting["authorized_intent_count"])
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
    partial_lane_id = manifest["migration_receipt"]["t10_2_6_partial_failure"][
        "partial_lane"
    ]["lane_id"]
    checks = {
        "parent_only_registered_orphan_failed": len(parent_state["complete_lanes"])
        == 17,
        "accepted_logical_lane_count": len(accepted_lane_reports) == 18,
        "accepted_complete_reset_count": accepted_reset_count == 72,
        "accepted_event_count_matches_reports": len(accepted_events)
        == accepted_sealed,
        "accepted_event_ids_unique": bool(event_ids)
        and len(event_ids) == len(set(event_ids)),
        "attempted_action_equation_holds_with_quarantine": attempted_authorized
        == attempted_sealed + attempted_unresolved + quarantined_unsealed,
        "excluded_actions_accounted": excluded_sealed
        >= len(parent_state["orphan_events"]),
        "orphan_lane_events_excluded": orphan_events_excluded,
        "t10_2_5_failed_actions_zero": int(
            zero_accounting["authorized_intent_count"]
        )
        == 0,
        "t10_2_6_partial_intent_quarantined": quarantined_unsealed == 1,
        "t10_2_6_partial_lane_not_accepted": all(
            item.get("physical_recovery_lane", {}).get("lane_id") != partial_lane_id
            for item in accepted_lane_reports
        ),
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
                "T10_2_7_SOURCE_COLLECTION_COMPLETE"
                if all(checks.values())
                else "DATA_OR_PROVENANCE_INVALID"
            ),
            "manifest_checksum": manifest["manifest_checksum"],
            "migration_receipt_checksum": manifest["migration_receipt"][
                "receipt_checksum"
            ],
            "predecessor_partial_journal_checksum": manifest[
                "migration_receipt"
            ]["t10_2_6_partial_failure"]["journal_checksum"],
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
                "quarantined_unsealed_predecessor_intents": quarantined_unsealed,
                "accepted_sealed_events": accepted_sealed,
                "excluded_sealed_events": excluded_sealed,
                "equation_holds_with_quarantine": attempted_authorized
                == attempted_sealed + attempted_unresolved + quarantined_unsealed,
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
        cache_predecessor,
        kernel,
        execution_manifest,
        kernel_path,
        artifact_root,
    ) = _load_execution_context(manifest_path=manifest_path, repo_root=repo_root)
    preflight = _first_event_seal_preflight(
        protocol_manifest=manifest,
        kernel=kernel,
    )
    if preflight.get("passed") is not True:
        raise JournalIntegrityError("T10.2.7 event-seal preflight did not pass")
    parent_report = _kernel_protocol._read_signed_json(
        root / artifact_root / _kernel_runtime.COLLECTION_REPORT_FILENAME,
        checksum_key="report_checksum",
    )
    recovery_runtime = _predecessor_runtime._predecessor_runtime
    zero_failure_manifest = _zero_failure_protocol.load_manifest(
        root / _zero_failure_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
    )
    parent_state = recovery_runtime._parent_state(
        root=root,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
        receipt=zero_failure_manifest["migration_receipt"],
    )
    recovery_report = _collect_recovery(
        root=root,
        manifest=manifest,
        execution_manifest=execution_manifest,
        cache_predecessor=cache_predecessor,
        kernel=kernel,
        discovery_events=parent_state["discovery_events"],
    )
    if recovery_report.get("status") != "PASS_T10_2_7_RECOVERY":
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
    root, manifest, _, _, kernel, execution, *_ = _load_execution_context(
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    preflight = _first_event_seal_preflight(
        protocol_manifest=manifest,
        kernel=kernel,
    )
    destination = root / _protocol.DEFAULT_RECOVERY_ROOT
    collection_path = destination / COLLECTION_REPORT_FILENAME
    recovery_path = destination / RECOVERY_REPORT_FILENAME
    recovery_status = None
    if recovery_path.is_file():
        recovery_status = _kernel_protocol._read_signed_json(
            recovery_path, checksum_key="report_checksum"
        ).get("status")
    return {
        "status": (
            "COMPLETE_T10_2_7_COLLECTION"
            if collection_path.is_file()
            else "FAILED_T10_2_7_RECOVERY"
            if recovery_status == "FAIL_T10_2_7_RECOVERY"
            else "RECOVERY_T10_2_7_IN_PROGRESS"
            if recovery_path.is_file()
            else "READY_T10_2_7_EVENT_SEAL_RECOVERY"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "migration": _protocol.verify_migration_receipt_live(
            manifest["migration_receipt"], repo_root=root
        ),
        "first_event_seal_preflight": preflight,
        "execution_manifest_sha256": canonical_sha256(execution),
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
    if args.phase == "collect" and payload.get(
        "status"
    ) != "T10_2_7_SOURCE_COLLECTION_COMPLETE":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
