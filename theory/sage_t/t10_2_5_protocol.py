"""Preregistered crash-recovery continuation for SAGE.T10.2.5.

T10.2.5 supersedes only the orchestration of the active T10.2.4 collection.
The frozen T10.2.2 scientific kernel, the T10.2.4 donor caches, and every
durable parent-journal record remain immutable.  One nonterminal confirmation
reset was left without a reset report when the Windows hard watchdog killed
the collector process tree.  This protocol records that exact orphan, excludes
its whole lane from scientific fitting, and preregisters bounded deterministic
replacement lanes in a separate append-only journal.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_2_runtime as _parent_runtime
from . import t10_2_3_protocol as _lineage_protocol
from . import t10_2_4_protocol as _predecessor_protocol

FORMAT_VERSION = "sage-t10.2.5-protocol-v1"
MIGRATION_FORMAT_VERSION = "sage-t10.2.5-migration-receipt-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_2_5_RECOVERY"
PREDECESSOR_MANIFEST_CHECKSUM = (
    "310f0e986bd14bf572b00bacdcde9ac5f07d7f50ab993744ad1d89862fbcc660"
)
PARENT_KERNEL_MANIFEST_CHECKSUM = (
    "3058989d51f8bc7ab0c65fd201941b20bc4d1cfa7754f1cb207598697594a428"
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_5_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_MIGRATION_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_5_migration_receipt.json"
)
DEFAULT_MIGRATION_PATH = Path(__file__).with_name(DEFAULT_MIGRATION_RELATIVE_PATH.name)
DEFAULT_RECOVERY_ROOT = Path("training") / "sage_t" / "t10_2_5_recovery"
MAXIMUM_RECOVERY_LANES = 3
RECOVERY_RESETS_PER_LANE = 4
RECOVERY_ACTIONS_PER_RESET = 64
RECOVERY_MAXIMUM_ACTIONS = (
    MAXIMUM_RECOVERY_LANES * RECOVERY_RESETS_PER_LANE * RECOVERY_ACTIONS_PER_RESET
)

DEFAULT_CODE_FILES = (
    "theory/sage_t/t10_2_5_protocol.py",
    "theory/sage_t/t10_2_5_runtime.py",
    "tests/test_sage_t_t10_2_5_protocol.py",
    "tests/test_sage_t_t10_2_5_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_2_5_RECOVERY_PROTOCOL.md",
    "reports/SAGE_T10_2_5_RECOVERY_RUNBOOK.md",
)

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
write_compact_json = _kernel_protocol.write_compact_json
_read_signed_json = _kernel_protocol._read_signed_json
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError
_BASE_DURABLE_COLLECTION_JOURNAL = _kernel_runtime.DurableCollectionJournal


def recovery_policy() -> dict[str, Any]:
    return {
        "change_scope": "crash_recovery_orchestration_only",
        "parent_scientific_kernel_unchanged": True,
        "parent_collection_journal_append_only": True,
        "parent_completed_records_mutated": False,
        "stale_incremental_cursor_repaired_from_full_journal_scan": True,
        "cursor_repair_changes_physical_records": False,
        "orphaned_physical_actions_replayed": False,
        "orphaned_lane_enters_model_fit": False,
        "orphan_reset_closed_as_interrupted": True,
        "unstarted_orphan_lane_reset_closed_without_actions": True,
        "replacement_scope": "whole_confirmation_lane",
        "replacement_controller_order_preserved": True,
        "replacement_seed_selection": "deterministic_receipt_derived_odd_seeds",
        "maximum_recovery_lanes": MAXIMUM_RECOVERY_LANES,
        "recovery_resets_per_lane": RECOVERY_RESETS_PER_LANE,
        "recovery_actions_per_reset": RECOVERY_ACTIONS_PER_RESET,
        "recovery_maximum_actions": RECOVERY_MAXIMUM_ACTIONS,
        "failed_recovery_lane_enters_model_fit": False,
        "watchdog_kill_scope": "reset_worker_process_tree_only",
        "collector_pid_may_be_killed_by_reset_watchdog": False,
        "parent_t10_2_4_caches_read_only": True,
        "accepted_logical_lane_count": 18,
        "accepted_complete_reset_count": 72,
        "validation_and_ar25_authority_opened": False,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "parent_collection_root": _parent_protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        "predecessor_manifest": (
            _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH.as_posix()
        ),
        "predecessor_cache_root": _predecessor_protocol.DEFAULT_CACHE_ROOT.as_posix(),
        "migration_receipt": DEFAULT_MIGRATION_RELATIVE_PATH.as_posix(),
        "recovery_root": DEFAULT_RECOVERY_ROOT.as_posix(),
        "recovery_journal": "source_collection_journal",
        "recovery_report": "recovery_report.json",
        "parent_cursor_repair_receipt": "parent_cursor_repair_receipt.json",
        "parent_orphan_closure_receipt": "parent_orphan_closure_receipt.json",
        "accepted_event_ledger": "accepted_source_events.jsonl",
        "accepted_cross_fit_audit": "accepted_cross_fit_audit.json",
        "collection_report": "t10_2_5_collection_report.json",
    }


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


def _load_predecessor(root: Path) -> dict[str, Any]:
    manifest = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        # The crash left the compact cursor one reset behind the durable
        # journal.  T10.2.5 authenticates and repairs that exact derivative,
        # so predecessor live verification must not run before the repair.
        verify_repository=False,
        verify_live_migration=False,
    )
    if manifest.get("manifest_checksum") != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.4 predecessor manifest checksum drifted")
    if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(
        root, _predecessor_protocol.DEFAULT_CODE_FILES, portable=True
    ):
        raise ManifestDriftError("T10.2.4 predecessor code bytes drifted")
    if manifest.get("document_sha256") != _kernel_protocol._hash_paths(
        root, _predecessor_protocol.DEFAULT_DOCUMENT_FILES, portable=True
    ):
        raise ManifestDriftError("T10.2.4 predecessor documentation drifted")
    return manifest


def _parent_execution(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    parent, kernel, kernel_path, artifact_root = _predecessor_protocol._parent_execution(
        root
    )
    if kernel.get("manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.2 execution kernel checksum drifted")
    return parent, kernel, kernel_path, artifact_root


def _record_binding(journal: Any, work: Any) -> dict[str, Any]:
    intents = [item.to_dict() for item in journal.intents_for_reset(work)]
    events = [dict(item) for item in journal.events_for_reset(work)]
    updates = [item.to_dict() for item in journal.updates_for_reset(work)]
    accounting = journal.reset_accounting(work).to_dict()
    report = journal.read_reset_report(work)
    return {
        "work": work.to_dict(),
        "accounting": accounting,
        "intent_count": len(intents),
        "event_count": len(events),
        "posterior_update_count": len(updates),
        "ordered_intents_sha256": canonical_sha256(intents),
        "ordered_events_sha256": canonical_sha256(events),
        "ordered_event_ids_sha256": canonical_sha256(
            [str(event.get("event_id", "")) for event in events]
        ),
        "ordered_updates_sha256": canonical_sha256(updates),
        "report": (
            None
            if report is None
            else {
                "status": report.status,
                "report_checksum": report.report_checksum,
                "stop_reason": report.stop_reason,
            }
        ),
    }


def _lane_fingerprint(report: Any) -> dict[str, Any]:
    return _lineage_protocol._lane_fingerprint(report)


def _cache_fingerprints(root: Path) -> list[dict[str, Any]]:
    cache_root = root / _predecessor_protocol.DEFAULT_CACHE_ROOT
    rows: list[dict[str, Any]] = []
    if not cache_root.is_dir():
        return rows
    for path in sorted(cache_root.glob("*/*/metadata.json")):
        metadata = _read_signed_json(path, checksum_key="cache_checksum")
        if metadata.get("finalized") is not True:
            raise ManifestDriftError("T10.2.4 cache is not finalized")
        state_name = str(metadata.get("state_file", ""))
        state_path = path.parent / state_name
        if state_name not in {"state-a.pkl", "state-b.pkl"} or not state_path.is_file():
            raise ManifestDriftError("T10.2.4 cache state is missing")
        digest = hashlib.sha256()
        size = 0
        with state_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                size += len(chunk)
                digest.update(chunk)
        if size != int(metadata.get("state_bytes", -1)):
            raise ManifestDriftError("T10.2.4 cache size drifted")
        if digest.hexdigest() != metadata.get("state_sha256"):
            raise ManifestDriftError("T10.2.4 cache state drifted")
        rows.append(
            {
                "kind": metadata["binding"]["kind"],
                "cache_key": metadata["binding"]["cache_key"],
                "cache_checksum": metadata["cache_checksum"],
                "state_sha256": metadata["state_sha256"],
                "state_bytes": size,
            }
        )
    return rows


def _journal_snapshot(
    *, root: Path, kernel: Mapping[str, Any], kernel_path: Path, artifact_root: Path
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
        # Use the full journal reader deliberately: the compact cursor is the
        # crashed derivative being migrated and cannot be authoritative here.
        journal = _BASE_DURABLE_COLLECTION_JOURNAL(
            root / artifact_root / _kernel_runtime.JOURNAL_DIRECTORY_NAME,
            manifest_checksum=str(kernel["manifest_checksum"]),
        )
        schedule = list(_parent_runtime._execution_lanes("full"))
        lane_reports = list(journal.lane_reports())
        if len(lane_reports) >= len(schedule):
            open_lane = None
            open_resets: list[dict[str, Any]] = []
        else:
            open_lane = schedule[len(lane_reports)]
            open_resets = [
                _record_binding(journal, work)
                for work in _kernel_runtime.reset_work_specs(open_lane)
            ]
        accounting_object = journal.accounting()
        cursor_accounting = accounting_object.to_dict()
        discovery = list(_kernel_runtime._completed_discovery_events(journal))
        reset_reports = [
            report
            for lane in schedule
            for work in _kernel_runtime.reset_work_specs(lane)
            if (report := journal.read_reset_report(work)) is not None
        ]
        return {
            "completed_lanes": [_lane_fingerprint(item) for item in lane_reports],
            "accounting": {
                **cursor_accounting,
                "posterior_update_count": accounting_object.posterior_update_count,
            },
            "cursor_accounting": cursor_accounting,
            "lane_reports_checksum": canonical_sha256(
                [report.to_dict() for report in lane_reports]
            ),
            "reset_active_seconds": sum(
                float(report.elapsed_seconds) for report in reset_reports
            ),
            "discovery": {
                "count": len(discovery),
                "ordered_event_ids_sha256": canonical_sha256(
                    [str(event.get("event_id", "")) for event in discovery]
                ),
                "ordered_events_sha256": canonical_sha256(discovery),
            },
            "open_lane": None if open_lane is None else open_lane.to_dict(),
            "open_resets": open_resets,
        }


def _specific_reset_binding(
    *,
    root: Path,
    kernel: Mapping[str, Any],
    kernel_path: Path,
    artifact_root: Path,
    work_payload: Mapping[str, Any],
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
        journal = _BASE_DURABLE_COLLECTION_JOURNAL(
            root / artifact_root / _kernel_runtime.JOURNAL_DIRECTORY_NAME,
            manifest_checksum=str(kernel["manifest_checksum"]),
        )
        work = _kernel_runtime.ResetWorkSpec.from_dict(work_payload)
        return _record_binding(journal, work)


def _derive_recovery_seeds(anchor: Mapping[str, Any]) -> list[int]:
    seeds: list[int] = []
    used = set(int(item) for item in (*_kernel_runtime.DISCOVERY_SEEDS, *_kernel_runtime.CONFIRMATION_SEEDS))
    index = 0
    while len(seeds) < MAXIMUM_RECOVERY_LANES:
        digest = canonical_sha256({"anchor": dict(anchor), "candidate_index": index})
        candidate = 1_000_001 + (int(digest[:12], 16) % 1_000_000)
        if candidate % 2 == 0:
            candidate += 1
        if candidate not in used:
            seeds.append(candidate)
            used.add(candidate)
        index += 1
    return seeds


def _expected_cursor_repair(
    *,
    cursor: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    kernel_manifest_checksum: str,
) -> dict[str, Any]:
    payload = {
        "format_version": _parent_runtime.CURSOR_FORMAT_VERSION,
        "manifest_checksum": str(kernel_manifest_checksum),
        "lane_registry_sha256": cursor["lane_registry_sha256"],
        "lane_reports_checksum": snapshot["lane_reports_checksum"],
        "accounting_checksum": canonical_sha256(snapshot["cursor_accounting"]),
        "cumulative_active_seconds": float(cursor["cumulative_active_seconds"]),
        "open_lane_id": cursor.get("open_lane_id"),
        "open_lane_elapsed_seconds": float(cursor["open_lane_elapsed_seconds"]),
        "reset_active_seconds": float(snapshot["reset_active_seconds"]),
        "revision": int(cursor["revision"]) + 1,
        "full_checkpoint_revision": int(cursor["full_checkpoint_revision"]),
        "full_history_scan_count": 1,
    }
    return signed_payload(payload, checksum_key="cursor_checksum")


def _recovery_lane(game_id: str, seed: int) -> dict[str, Any]:
    identity = {
        "split": "leave_one_game_out_confirmation",
        "game_id": str(game_id),
        "seed": int(seed),
    }
    return {**identity, "lane_id": canonical_sha256(identity)}


def _require_orphan_snapshot(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    completed = snapshot.get("completed_lanes")
    open_lane = snapshot.get("open_lane")
    resets = snapshot.get("open_resets")
    if not isinstance(completed, list) or len(completed) != 15:
        raise ManifestDriftError("T10.2.5 requires the exact fifteen-lane prefix")
    if any(item.get("status") != "COMPLETE" for item in completed):
        raise ManifestDriftError("T10.2.5 frozen prefix contains an incomplete lane")
    if not isinstance(open_lane, Mapping) or open_lane.get("split") != (
        "leave_one_game_out_confirmation"
    ):
        raise ManifestDriftError("T10.2.5 orphan is not a confirmation lane")
    if not isinstance(resets, list) or len(resets) != RECOVERY_RESETS_PER_LANE:
        raise ManifestDriftError("T10.2.5 orphan reset registry drifted")
    reported = [item for item in resets if item.get("report") is not None]
    partial = [
        item
        for item in resets
        if item.get("report") is None
        and int((item.get("accounting") or {}).get("authorized_intent_count", 0)) > 0
    ]
    untouched = [
        item
        for item in resets
        if item.get("report") is None
        and int((item.get("accounting") or {}).get("authorized_intent_count", 0)) == 0
    ]
    if len(reported) != 2 or any(item["report"]["status"] != "COMPLETE" for item in reported):
        raise ManifestDriftError("T10.2.5 requires two complete predecessor resets")
    if len(partial) != 1 or len(untouched) != 1:
        raise ManifestDriftError("T10.2.5 requires one orphan and one untouched reset")
    orphan = partial[0]
    accounting = orphan["accounting"]
    if not (
        accounting.get("equation_holds") is True
        and int(accounting.get("authorized_intent_count", -1)) > 0
        and accounting.get("authorized_intent_count") == accounting.get("sealed_event_count")
        and accounting.get("sealed_event_count") == orphan.get("posterior_update_count")
        and int(accounting.get("explicitly_unresolved_intent_count", -1)) == 0
        and int(accounting.get("unknown_intent_count", -1)) == 0
    ):
        raise JournalIntegrityError("T10.2.5 orphan accounting is not fully sealed")
    if int(orphan["work"]["reset_index"]) != 2:
        raise ManifestDriftError("T10.2.5 orphan reset index drifted")
    if orphan["work"]["controller"] != "capacity_matched_independent":
        raise ManifestDriftError("T10.2.5 orphan controller drifted")
    return dict(open_lane), dict(orphan)


def build_migration_receipt(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    _, kernel, kernel_path, artifact_root = _parent_execution(root)
    destination = root / artifact_root
    lease = _kernel_runtime._CollectionLease.acquire(
        destination / ".active-collector.lock"
    )
    try:
        snapshot = _journal_snapshot(
            root=root,
            kernel=kernel,
            kernel_path=kernel_path,
            artifact_root=artifact_root,
        )
        open_lane, orphan = _require_orphan_snapshot(snapshot)
        accounting = snapshot["accounting"]
        if (
            accounting.get("equation_holds") is not True
            or accounting.get("unknown_intent_count")
            or accounting.get("explicitly_unresolved_intent_count")
        ):
            raise JournalIntegrityError("T10.2.5 parent accounting is open")
        checkpoint = _read_signed_json(
            destination / _parent_protocol.CHECKPOINT_FILENAME,
            checksum_key="checkpoint_checksum",
        )
        cursor = _read_signed_json(
            destination / _parent_runtime.CURSOR_FILENAME,
            checksum_key="cursor_checksum",
        )
        invocation = _kernel_runtime._read_invocation_state(
            destination / _kernel_runtime.INVOCATION_STATE_FILENAME
        )
        terminal = _kernel_runtime._read_invocation_terminal(
            destination / _kernel_runtime.INVOCATION_TERMINAL_FILENAME,
            opened=invocation,
        )
        if invocation is None or invocation.get("status") != "OPEN" or terminal is not None:
            raise ManifestDriftError("T10.2.5 requires an unterminated OPEN invocation")
        if (destination / _kernel_runtime.COLLECTION_REPORT_FILENAME).exists():
            raise ManifestDriftError("T10.2.5 parent collection is already terminal")
        if cursor.get("open_lane_id") != open_lane["lane_id"]:
            raise ManifestDriftError("T10.2.5 cursor escaped the orphan lane")
        anchor = {
            "checkpoint_checksum": checkpoint["checkpoint_checksum"],
            "cursor_checksum": cursor["cursor_checksum"],
            "open_lane_id": open_lane["lane_id"],
            "orphan_work_id": orphan["work"]["work_id"],
            "orphan_events_sha256": orphan["ordered_events_sha256"],
        }
        seeds = _derive_recovery_seeds(anchor)
        lanes = [_recovery_lane(str(open_lane["game_id"]), seed) for seed in seeds]
        caches = _cache_fingerprints(root)
        repaired_cursor = _expected_cursor_repair(
            cursor=cursor,
            snapshot=snapshot,
            kernel_manifest_checksum=str(kernel["manifest_checksum"]),
        )
        return signed_payload(
            {
                "format_version": MIGRATION_FORMAT_VERSION,
                "predecessor_t10_2_4_manifest_checksum": predecessor[
                    "manifest_checksum"
                ],
                "parent_kernel_manifest_checksum": kernel["manifest_checksum"],
                "initial_checkpoint": {
                    "revision": checkpoint["revision"],
                    "checksum": checkpoint["checkpoint_checksum"],
                },
                "initial_cursor": {
                    "revision": cursor["revision"],
                    "checksum": cursor["cursor_checksum"],
                    "open_lane_id": cursor["open_lane_id"],
                },
                "cursor_repair": {
                    "reason": "durable_records_committed_after_last_compact_cursor",
                    "old_cursor_checksum": cursor["cursor_checksum"],
                    "repaired_cursor": repaired_cursor,
                    "repaired_cursor_checksum": repaired_cursor["cursor_checksum"],
                    "physical_records_changed": False,
                },
                "open_invocation": {
                    "invocation_id": invocation["invocation_id"],
                    "state_checksum": invocation["state_checksum"],
                    "terminal_absent": True,
                },
                "completed_lane_count": len(snapshot["completed_lanes"]),
                "completed_reset_count": sum(
                    len(item["resets"]) for item in snapshot["completed_lanes"]
                ),
                "completed_lanes": snapshot["completed_lanes"],
                "completed_lanes_sha256": canonical_sha256(
                    snapshot["completed_lanes"]
                ),
                "orphan_lane": open_lane,
                "open_reset_bindings": snapshot["open_resets"],
                "orphan_reset": orphan,
                "initial_accounting": accounting,
                "initial_discovery_evidence": snapshot["discovery"],
                "recovery_seed_anchor": anchor,
                "recovery_seeds": seeds,
                "recovery_lanes": lanes,
                "adopted_t10_2_4_caches": caches,
                "adopted_t10_2_4_caches_sha256": canonical_sha256(caches),
                "replay_authorized": False,
                "orphan_lane_fit_authorized": False,
            },
            checksum_key="receipt_checksum",
        )
    finally:
        lease.release()


def verify_migration_receipt_live(
    receipt: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    root = _root(repo_root)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.2.5 migration receipt checksum drifted")
    if receipt.get("format_version") != MIGRATION_FORMAT_VERSION:
        raise ManifestDriftError("T10.2.5 migration receipt format drifted")
    predecessor = _load_predecessor(root)
    if receipt.get("predecessor_t10_2_4_manifest_checksum") != predecessor.get(
        "manifest_checksum"
    ):
        raise ManifestDriftError("T10.2.5 predecessor binding drifted")
    _, kernel, kernel_path, artifact_root = _parent_execution(root)
    if receipt.get("parent_kernel_manifest_checksum") != kernel.get("manifest_checksum"):
        raise ManifestDriftError("T10.2.5 kernel binding drifted")
    snapshot = _journal_snapshot(
        root=root,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
    )
    frozen = receipt.get("completed_lanes")
    live_lanes = snapshot.get("completed_lanes")
    if not isinstance(frozen, list) or not isinstance(live_lanes, list):
        raise ManifestDriftError("T10.2.5 frozen lane registry is malformed")
    if live_lanes[: len(frozen)] != frozen:
        raise JournalIntegrityError("a frozen T10.2.5 parent lane changed")
    initial = receipt.get("initial_accounting")
    live_accounting = snapshot.get("accounting")
    if not isinstance(initial, Mapping) or not isinstance(live_accounting, Mapping):
        raise ManifestDriftError("T10.2.5 accounting receipt is malformed")
    for key in ("authorized_intent_count", "sealed_event_count", "posterior_update_count"):
        if int(live_accounting.get(key, -1)) < int(initial.get(key, -1)):
            raise JournalIntegrityError("T10.2.5 parent accounting regressed")
    if live_accounting.get("unknown_intent_count") or not live_accounting.get(
        "equation_holds"
    ):
        raise JournalIntegrityError("T10.2.5 live parent accounting is open")
    orphan = receipt.get("orphan_reset")
    if not isinstance(orphan, Mapping):
        raise ManifestDriftError("T10.2.5 orphan receipt is malformed")
    live_orphan = _specific_reset_binding(
        root=root,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
        work_payload=orphan["work"],
    )
    for key in (
        "accounting",
        "intent_count",
        "event_count",
        "posterior_update_count",
        "ordered_intents_sha256",
        "ordered_events_sha256",
        "ordered_event_ids_sha256",
        "ordered_updates_sha256",
    ):
        if live_orphan.get(key) != orphan.get(key):
            raise JournalIntegrityError("T10.2.5 orphan physical records changed")
    cursor_repair = receipt.get("cursor_repair")
    if not isinstance(cursor_repair, Mapping) or not isinstance(
        cursor_repair.get("repaired_cursor"), Mapping
    ):
        raise ManifestDriftError("T10.2.5 cursor repair receipt is malformed")
    cursor = _read_signed_json(
        root / artifact_root / _parent_runtime.CURSOR_FILENAME,
        checksum_key="cursor_checksum",
    )
    accepted_cursor_checksums = {
        receipt["initial_cursor"]["checksum"],
        cursor_repair["repaired_cursor_checksum"],
    }
    if cursor.get("cursor_checksum") not in accepted_cursor_checksums:
        if int(cursor.get("revision", -1)) < int(
            cursor_repair["repaired_cursor"]["revision"]
        ):
            raise JournalIntegrityError("T10.2.5 live cursor regressed")
        if cursor.get("accounting_checksum") != canonical_sha256(
            snapshot["cursor_accounting"]
        ):
            raise JournalIntegrityError("T10.2.5 live cursor accounting drifted")
    seeds = receipt.get("recovery_seeds")
    anchor = receipt.get("recovery_seed_anchor")
    if not isinstance(seeds, list) or not isinstance(anchor, Mapping):
        raise ManifestDriftError("T10.2.5 recovery seed receipt is malformed")
    if seeds != _derive_recovery_seeds(anchor):
        raise ManifestDriftError("T10.2.5 recovery seeds drifted")
    frozen_caches = receipt.get("adopted_t10_2_4_caches")
    current_caches = _cache_fingerprints(root)
    if not isinstance(frozen_caches, list) or current_caches[: len(frozen_caches)] != frozen_caches:
        raise JournalIntegrityError("an adopted T10.2.4 cache changed")
    return {
        "migration_verified": True,
        "frozen_completed_lanes": len(frozen),
        "current_parent_lane_reports": len(live_lanes),
        "orphan_lane_id": receipt["orphan_lane"]["lane_id"],
        "orphan_reset_work_id": orphan["work"]["work_id"],
        "orphan_sealed_events": orphan["event_count"],
        "recovery_seeds": list(seeds),
        "maximum_recovery_lanes": MAXIMUM_RECOVERY_LANES,
        "replay_authorized": False,
    }


def build_manifest(
    *, repo_root: str | Path | None = None, migration_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    _, kernel, _, _ = _parent_execution(root)
    verify_migration_receipt_live(migration_receipt, repo_root=root)
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "status": MANIFEST_STATUS,
            "hash_algorithm": _kernel_protocol.HASH_ALGORITHM,
            "predecessor_t10_2_4_manifest_checksum": predecessor[
                "manifest_checksum"
            ],
            "parent_kernel_manifest_checksum": kernel["manifest_checksum"],
            "registered_phases": ["freeze", "status", "collect"],
            "portable_code_sha256": _kernel_protocol._hash_paths(
                root, DEFAULT_CODE_FILES, portable=True
            ),
            "document_sha256": _kernel_protocol._hash_paths(
                root, DEFAULT_DOCUMENT_FILES, portable=True
            ),
            "recovery_policy": recovery_policy(),
            "artifact_contract": artifact_contract(),
            "migration_receipt": dict(migration_receipt),
        },
        checksum_key="manifest_checksum",
    )


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    migration_path: str | Path = DEFAULT_MIGRATION_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(repo_root)
    receipt = build_migration_receipt(repo_root=root)
    manifest = build_manifest(repo_root=root, migration_receipt=receipt)
    receipt_path = Path(migration_path)
    manifest_path = Path(output_path)
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    write_compact_json(receipt_path, receipt)
    write_compact_json(manifest_path, manifest)
    return manifest


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
    verify_repository: bool = True,
    verify_live_migration: bool = True,
) -> dict[str, Any]:
    root = _root(repo_root)
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    manifest = _read_signed_json(source, checksum_key="manifest_checksum")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ManifestDriftError("T10.2.5 manifest format drifted")
    if manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.2.5 manifest status drifted")
    if manifest.get("recovery_policy") != recovery_policy():
        raise ManifestDriftError("T10.2.5 recovery policy drifted")
    if manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.2.5 artifact contract drifted")
    if manifest.get(
        "predecessor_t10_2_4_manifest_checksum"
    ) != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.5 predecessor manifest drifted")
    if manifest.get("parent_kernel_manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.5 kernel manifest drifted")
    receipt = manifest.get("migration_receipt")
    if not isinstance(receipt, Mapping):
        raise ManifestDriftError("T10.2.5 migration receipt is missing")
    materialized = _read_signed_json(
        root / DEFAULT_MIGRATION_RELATIVE_PATH,
        checksum_key="receipt_checksum",
    )
    if materialized != receipt:
        raise ManifestDriftError("materialized T10.2.5 receipt drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_CODE_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.5 code bytes drifted")
        if manifest.get("document_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_DOCUMENT_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.5 documentation bytes drifted")
        _load_predecessor(root)
    if verify_live_migration:
        verify_migration_receipt_live(receipt, repo_root=root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "status"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "freeze":
            payload = freeze_manifest(
                output_path=args.manifest,
                repo_root=args.repo_root,
            )
        else:
            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            payload = {
                "status": "READY_T10_2_5_RECOVERY",
                "manifest_checksum": manifest["manifest_checksum"],
                "migration": verify_migration_receipt_live(
                    manifest["migration_receipt"], repo_root=args.repo_root
                ),
            }
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
