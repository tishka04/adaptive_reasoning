"""Preregistered, replay-free continuation protocol for SAGE.T10.2.3.

T10.2.3 does not change the T10.2.2 scientific kernel, lane schedule, action
budget, journal schema, or authority boundary.  It corrects one orchestration
failure: confirmation workers repeatedly rebuilt the same exact donor
posterior inside the reset liveness watchdog.  The continuation precomputes
that posterior in an authenticated, resumable side cache before a reset starts.

The migration receipt binds the immutable T10.2.2 prefix that existed at
freeze time.  A continuation is refused if any completed lane, reset report,
or sealed-event digest from that prefix changes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_2_runtime as _parent_runtime

FORMAT_VERSION = "sage-t10.2.3-protocol-v1"
MIGRATION_FORMAT_VERSION = "sage-t10.2.3-migration-receipt-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_2_3_CONTINUATION"
PARENT_MANIFEST_CHECKSUM = (
    "5f134e334df65b23c6df189fa672ac6069d15ab9b8a5e95ada3577f34dc1401c"
)
PARENT_KERNEL_MANIFEST_CHECKSUM = (
    "3058989d51f8bc7ab0c65fd201941b20bc4d1cfa7754f1cb207598697594a428"
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_3_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_MIGRATION_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_3_migration_receipt.json"
)
DEFAULT_MIGRATION_PATH = Path(__file__).with_name(DEFAULT_MIGRATION_RELATIVE_PATH.name)
DEFAULT_CACHE_ROOT = Path("training") / "sage_t" / "t10_2_3_donor_cache"

DEFAULT_CODE_FILES = (
    "theory/sage_t/t10_2_3_protocol.py",
    "theory/sage_t/t10_2_3_runtime.py",
    "tests/test_sage_t_t10_2_3_protocol.py",
    "tests/test_sage_t_t10_2_3_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_2_3_CONTINUATION_PROTOCOL.md",
    "reports/SAGE_T10_2_3_CONTINUATION_RUNBOOK.md",
)

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
canonical_file_sha256 = _kernel_protocol.canonical_file_sha256
signed_payload = _kernel_protocol.signed_payload
write_compact_json = _kernel_protocol.write_compact_json
_read_signed_json = _kernel_protocol._read_signed_json
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError
JournalConflictError = _kernel_runtime.JournalConflictError


def continuation_policy() -> dict[str, Any]:
    """Return the frozen optimization and authority boundary."""

    return {
        "change_scope": "orchestration_only",
        "parent_scientific_kernel_unchanged": True,
        "parent_journal_namespace_reused": True,
        "parent_journal_records_mutated": False,
        "completed_physical_actions_replayed": False,
        "cache_role": "exact_donor_posterior_memoization",
        "cache_authority": "none",
        "cache_candidate_limit": 256,
        "cache_checkpoint_interval_events": 8,
        "cache_maximum_bytes": 536_870_912,
        "cache_pickle_protocol": 5,
        "cache_root": DEFAULT_CACHE_ROOT.as_posix(),
        "cache_built_before_reset_watchdog": True,
        "cache_build_time_charged_to_lane_and_collection": True,
        "cache_key_binds_ordered_donor_events": True,
        "cache_resume_requires_exact_prefix": True,
        "holdout_evidence_enters_cache": False,
        "validation_and_ar25_authority_opened": False,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "parent_collection_root": _parent_protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        "parent_manifest": _parent_protocol.DEFAULT_MANIFEST_RELATIVE_PATH.as_posix(),
        "parent_kernel_manifest": (
            _parent_protocol.DEFAULT_KERNEL_MANIFEST_RELATIVE_PATH.as_posix()
        ),
        "migration_receipt": DEFAULT_MIGRATION_RELATIVE_PATH.as_posix(),
        "cache_root": DEFAULT_CACHE_ROOT.as_posix(),
        "continuation_report": "t10_2_3_continuation_report.json",
        "parent_collection_root_allowlist_unchanged": True,
    }


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


def _load_parent(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    parent = _parent_protocol.load_manifest(
        root / _parent_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=True,
    )
    if parent.get("manifest_checksum") != PARENT_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.2 parent manifest checksum drifted")
    kernel, kernel_path, artifact_root = _parent_protocol.load_kernel_manifest(
        manifest=parent, mode="full", repo_root=root
    )
    if kernel.get("manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.2 execution kernel checksum drifted")
    return parent, kernel, kernel_path, artifact_root


def _lane_fingerprint(report: Any) -> dict[str, Any]:
    return {
        "lane_id": report.lane.lane_id,
        "lane": report.lane.to_dict(),
        "status": report.status,
        "report_checksum": report.report_checksum,
        "issued_intents": report.issued_intents,
        "sealed_events": report.sealed_events,
        "unresolved_intents": report.unresolved_intents,
        "resets": [
            {
                "work_id": reset.work.work_id,
                "report_checksum": reset.report_checksum,
                "event_ids_sha256": reset.event_ids_sha256,
                "issued_intents": reset.issued_intents,
                "sealed_events": reset.sealed_events,
                "unresolved_intents": reset.unresolved_intents,
                "status": reset.status,
            }
            for reset in report.resets
        ],
    }


def _journal_snapshot(
    *, root: Path, parent: Mapping[str, Any], kernel: Mapping[str, Any], kernel_path: Path,
    artifact_root: Path,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    relative_kernel = kernel_path.relative_to(root)
    with (
        _parent_protocol.kernel_protocol_bindings(
            artifact_root=artifact_root,
            manifest_relative_path=relative_kernel,
            mode="full",
        ),
        _parent_runtime.execution_bindings(mode="full", artifact_root=artifact_root),
    ):
        verified = _kernel_protocol.load_manifest(kernel_path, repo_root=root)
        if verified != kernel:
            raise ManifestDriftError("verified T10.2.2 kernel drifted")
        journal = _parent_runtime.IncrementalDurableCollectionJournal(
            root / artifact_root / _kernel_runtime.JOURNAL_DIRECTORY_NAME,
            manifest_checksum=str(kernel["manifest_checksum"]),
        )
        lane_fingerprints = [
            _lane_fingerprint(report) for report in journal.lane_reports()
        ]
        accounting_object = journal.accounting()
        accounting = {
            **accounting_object.to_dict(),
            "posterior_update_count": accounting_object.posterior_update_count,
        }
        discovery_events = list(journal.completed_discovery_events())
        discovery_binding = {
            "count": len(discovery_events),
            "ordered_event_ids_sha256": canonical_sha256(
                [str(event.get("event_id", "")) for event in discovery_events]
            ),
            "ordered_events_sha256": canonical_sha256(discovery_events),
        }
        return journal, lane_fingerprints, accounting, discovery_binding


def build_migration_receipt(
    *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Attest the exact complete prefix of the stopped T10.2.2 collection."""

    root = _root(repo_root)
    parent, kernel, kernel_path, artifact_root = _load_parent(root)
    destination = root / artifact_root
    lease = _kernel_runtime._CollectionLease.acquire(
        destination / ".active-collector.lock"
    )
    try:
        journal, lanes, accounting, discovery = _journal_snapshot(
            root=root,
            parent=parent,
            kernel=kernel,
            kernel_path=kernel_path,
            artifact_root=artifact_root,
        )
        if not lanes or len(lanes) >= len(_parent_runtime._execution_lanes("full")):
            raise ManifestDriftError(
                "migration requires a non-empty, non-terminal completed prefix"
            )
        if any(item["status"] != "COMPLETE" for item in lanes):
            raise ManifestDriftError("migration prefix contains an incomplete lane")
        if accounting.get("unknown_intent_count") or not accounting.get(
            "equation_holds"
        ):
            raise JournalIntegrityError("migration accounting is not fail-closed")
        if accounting.get("explicitly_unresolved_intent_count"):
            raise JournalIntegrityError("migration prefix has unresolved intents")
        checkpoint = _read_signed_json(
            destination / _parent_protocol.CHECKPOINT_FILENAME,
            checksum_key="checkpoint_checksum",
        )
        cursor = _read_signed_json(
            destination / _parent_runtime.CURSOR_FILENAME,
            checksum_key="cursor_checksum",
        )
        completed_ids = [item["lane_id"] for item in lanes]
        schedule = list(_parent_runtime._execution_lanes("full"))
        expected_prefix = [lane.lane_id for lane in schedule[: len(lanes)]]
        if completed_ids != expected_prefix:
            raise ManifestDriftError("completed lanes are not the registered prefix")
        next_lane = schedule[len(lanes)]
        if cursor.get("open_lane_id") not in (None, next_lane.lane_id):
            raise ManifestDriftError("resume cursor escaped the next registered lane")
        return signed_payload(
            {
                "format_version": MIGRATION_FORMAT_VERSION,
                "parent_t10_2_2_manifest_checksum": parent["manifest_checksum"],
                "parent_kernel_manifest_checksum": kernel["manifest_checksum"],
                "initial_checkpoint": {
                    "revision": checkpoint["revision"],
                    "checksum": checkpoint["checkpoint_checksum"],
                },
                "initial_cursor": {
                    "revision": cursor["revision"],
                    "checksum": cursor["cursor_checksum"],
                    "open_lane_id": cursor.get("open_lane_id"),
                },
                "completed_lane_count": len(lanes),
                "completed_reset_count": sum(len(item["resets"]) for item in lanes),
                "completed_lanes": lanes,
                "completed_lanes_sha256": canonical_sha256(lanes),
                "initial_accounting": accounting,
                "initial_discovery_evidence": discovery,
                "next_lane": next_lane.to_dict(),
                "replay_authorized": False,
            },
            checksum_key="receipt_checksum",
        )
    finally:
        lease.release()


def verify_migration_receipt_live(
    receipt: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Verify that the frozen prefix remains an immutable subset of live state."""

    root = _root(repo_root)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("migration receipt checksum drifted")
    if receipt.get("format_version") != MIGRATION_FORMAT_VERSION:
        raise ManifestDriftError("migration receipt format drifted")
    parent, kernel, kernel_path, artifact_root = _load_parent(root)
    if receipt.get("parent_t10_2_2_manifest_checksum") != parent["manifest_checksum"]:
        raise ManifestDriftError("migration parent manifest binding drifted")
    if receipt.get("parent_kernel_manifest_checksum") != kernel["manifest_checksum"]:
        raise ManifestDriftError("migration kernel binding drifted")
    _, lanes, accounting, discovery = _journal_snapshot(
        root=root,
        parent=parent,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
    )
    frozen = receipt.get("completed_lanes")
    if not isinstance(frozen, list) or lanes[: len(frozen)] != frozen:
        raise JournalIntegrityError("a frozen T10.2.2 lane or reset report changed")
    initial = receipt.get("initial_accounting")
    if not isinstance(initial, Mapping):
        raise ManifestDriftError("migration accounting receipt is malformed")
    for key in (
        "authorized_intent_count",
        "sealed_event_count",
        "posterior_update_count",
    ):
        if int(accounting.get(key, -1)) < int(initial.get(key, -1)):
            raise JournalIntegrityError("live accounting fell behind the frozen prefix")
    if accounting.get("unknown_intent_count") or not accounting.get("equation_holds"):
        raise JournalIntegrityError("live journal accounting is not fail-closed")
    if accounting.get("explicitly_unresolved_intent_count"):
        raise JournalIntegrityError("live journal contains unresolved intents")
    initial_discovery = receipt.get("initial_discovery_evidence")
    if not isinstance(initial_discovery, Mapping):
        raise ManifestDriftError("migration discovery binding is malformed")
    if int(discovery["count"]) < int(initial_discovery.get("count", -1)):
        raise JournalIntegrityError("live discovery evidence lost frozen events")
    return {
        "migration_verified": True,
        "frozen_completed_lanes": len(frozen),
        "current_completed_lanes": len(lanes),
        "frozen_checkpoint_checksum": receipt["initial_checkpoint"]["checksum"],
        "next_lane": receipt["next_lane"],
        "replay_authorized": False,
    }


def build_manifest(
    *, repo_root: str | Path | None = None, migration_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    parent, kernel, _, _ = _load_parent(root)
    verify_migration_receipt_live(migration_receipt, repo_root=root)
    payload = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "hash_algorithm": _kernel_protocol.HASH_ALGORITHM,
        "parent_t10_2_2_manifest_checksum": parent["manifest_checksum"],
        "parent_kernel_manifest_checksum": kernel["manifest_checksum"],
        "registered_phases": ["freeze", "status", "prepare", "continue"],
        "portable_code_sha256": _kernel_protocol._hash_paths(
            root, DEFAULT_CODE_FILES, portable=True
        ),
        "document_sha256": _kernel_protocol._hash_paths(
            root, DEFAULT_DOCUMENT_FILES, portable=True
        ),
        "continuation_policy": continuation_policy(),
        "artifact_contract": artifact_contract(),
        "migration_receipt": dict(migration_receipt),
    }
    return signed_payload(payload, checksum_key="manifest_checksum")


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    migration_path: str | Path = DEFAULT_MIGRATION_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(repo_root)
    receipt = build_migration_receipt(repo_root=root)
    manifest = build_manifest(repo_root=root, migration_receipt=receipt)
    migration_destination = Path(migration_path)
    if not migration_destination.is_absolute():
        migration_destination = root / migration_destination
    manifest_destination = Path(output_path)
    if not manifest_destination.is_absolute():
        manifest_destination = root / manifest_destination
    write_compact_json(migration_destination, receipt)
    write_compact_json(manifest_destination, manifest)
    return manifest


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
    verify_repository: bool = True,
    verify_live_migration: bool = True,
) -> dict[str, Any]:
    root = _root(repo_root)
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = _read_signed_json(manifest_path, checksum_key="manifest_checksum")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ManifestDriftError("T10.2.3 manifest format drifted")
    if manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.2.3 manifest status drifted")
    if manifest.get("continuation_policy") != continuation_policy():
        raise ManifestDriftError("T10.2.3 continuation policy drifted")
    if manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.2.3 artifact contract drifted")
    if manifest.get("parent_t10_2_2_manifest_checksum") != PARENT_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.3 parent binding drifted")
    if manifest.get("parent_kernel_manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.3 kernel binding drifted")
    receipt = manifest.get("migration_receipt")
    if not isinstance(receipt, Mapping):
        raise ManifestDriftError("T10.2.3 migration receipt is missing")
    materialized = _read_signed_json(
        root / DEFAULT_MIGRATION_RELATIVE_PATH, checksum_key="receipt_checksum"
    )
    if materialized != receipt:
        raise ManifestDriftError("materialized migration receipt drifted")
    if verify_repository:
        current_code = _kernel_protocol._hash_paths(
            root, DEFAULT_CODE_FILES, portable=True
        )
        current_docs = _kernel_protocol._hash_paths(
            root, DEFAULT_DOCUMENT_FILES, portable=True
        )
        if manifest.get("portable_code_sha256") != current_code:
            raise ManifestDriftError("T10.2.3 code bytes drifted")
        if manifest.get("document_sha256") != current_docs:
            raise ManifestDriftError("T10.2.3 documentation bytes drifted")
        _load_parent(root)
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
                output_path=args.manifest, repo_root=args.repo_root
            )
        else:
            manifest = load_manifest(
                args.manifest, repo_root=args.repo_root, verify_live_migration=True
            )
            payload = {
                "status": "READY_T10_2_3_CONTINUATION",
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
