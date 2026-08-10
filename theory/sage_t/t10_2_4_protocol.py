"""Preregistered dual-cache continuation protocol for SAGE.T10.2.4.

T10.2.4 migrates the immutable thirteen-lane T10.2.3 prefix.  It preserves the
T10.2.2 scientific kernel and adds exact memoization for the two donor-only
fits still performed inside confirmation workers:

* the five-factor ``FactorizedGaugeProgramPosterior`` control; and
* later learned-reset reconstructions of the already cached gauge posterior.

Only calls whose ordered events, candidate bank, posterior type and frozen
limits match an authenticated donor cache are intercepted.  All other fits are
delegated to the frozen implementation.
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
from . import t10_2_3_protocol as _predecessor_protocol

FORMAT_VERSION = "sage-t10.2.4-protocol-v1"
MIGRATION_FORMAT_VERSION = "sage-t10.2.4-migration-receipt-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_2_4_CONTINUATION"
PREDECESSOR_MANIFEST_CHECKSUM = (
    "723275b224cfeccde6e4d7a52eb03b14d9b704e24844594d545bb4b13dec8c38"
)
PARENT_KERNEL_MANIFEST_CHECKSUM = (
    "3058989d51f8bc7ab0c65fd201941b20bc4d1cfa7754f1cb207598697594a428"
)
SUPERSEDED_PREFLIGHT_MANIFEST_CHECKSUM = (
    "aadf02d5cc15028d31adfffda4b7464aed90130a0a80284f8a8de67120a011b2"
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_4_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_MIGRATION_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_4_migration_receipt.json"
)
DEFAULT_MIGRATION_PATH = Path(__file__).with_name(DEFAULT_MIGRATION_RELATIVE_PATH.name)
DEFAULT_CACHE_ROOT = Path("training") / "sage_t" / "t10_2_4_dual_cache"

DEFAULT_CODE_FILES = (
    "theory/sage_t/t10_2_4_protocol.py",
    "theory/sage_t/t10_2_4_runtime.py",
    "tests/test_sage_t_t10_2_4_protocol.py",
    "tests/test_sage_t_t10_2_4_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_2_4_DUAL_CACHE_PROTOCOL.md",
    "reports/SAGE_T10_2_4_DUAL_CACHE_RUNBOOK.md",
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


def continuation_policy() -> dict[str, Any]:
    return {
        "change_scope": "orchestration_only",
        "parent_scientific_kernel_unchanged": True,
        "parent_journal_namespace_reused": True,
        "parent_journal_records_mutated": False,
        "completed_physical_actions_replayed": False,
        "predecessor_cache_root_read_only": True,
        "predecessor_gauge_cache_adopted_when_exact": True,
        "new_cache_kinds": ["gauge", "factorized"],
        "factorized_candidate_bank_unchanged": True,
        "fit_interception_requires_exact_events": True,
        "fit_interception_requires_exact_candidates": True,
        "fit_interception_requires_exact_posterior_type": True,
        "nonmatching_fits_delegated": True,
        "cache_candidate_limit": 256,
        "cache_checkpoint_interval_events": 8,
        "cache_maximum_bytes": 536_870_912,
        "cache_pickle_protocol": 5,
        "cache_root": DEFAULT_CACHE_ROOT.as_posix(),
        "cache_built_before_reset_watchdog": True,
        "cache_build_time_charged_to_lane_and_collection": True,
        "holdout_evidence_enters_cache": False,
        "validation_and_ar25_authority_opened": False,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "parent_collection_root": _parent_protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        "predecessor_manifest": (
            _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH.as_posix()
        ),
        "predecessor_cache_root": (
            _predecessor_protocol.DEFAULT_CACHE_ROOT.as_posix()
        ),
        "migration_receipt": DEFAULT_MIGRATION_RELATIVE_PATH.as_posix(),
        "cache_root": DEFAULT_CACHE_ROOT.as_posix(),
        "continuation_report": "t10_2_4_continuation_report.json",
        "parent_collection_root_allowlist_unchanged": True,
    }


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


def _load_predecessor(root: Path) -> dict[str, Any]:
    manifest = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=True,
        verify_live_migration=True,
    )
    if manifest.get("manifest_checksum") != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.3 predecessor manifest checksum drifted")
    return manifest


def _parent_execution(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    parent, kernel, kernel_path, artifact_root = _predecessor_protocol._load_parent(
        root
    )
    if kernel.get("manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.2 execution kernel checksum drifted")
    return parent, kernel, kernel_path, artifact_root


def _predecessor_cache_fingerprints(root: Path) -> list[dict[str, Any]]:
    cache_root = root / _predecessor_protocol.DEFAULT_CACHE_ROOT
    fingerprints: list[dict[str, Any]] = []
    if not cache_root.is_dir():
        return fingerprints
    for metadata_path in sorted(cache_root.glob("*/metadata.json")):
        metadata = _read_signed_json(metadata_path, checksum_key="cache_checksum")
        if metadata.get("finalized") is not True:
            raise ManifestDriftError("predecessor donor cache is incomplete")
        state_name = str(metadata.get("state_file", ""))
        if state_name not in {"state-a.pkl", "state-b.pkl"}:
            raise ManifestDriftError("predecessor donor cache slot drifted")
        state_path = metadata_path.parent / state_name
        if not state_path.is_file() or state_path.is_symlink():
            raise ManifestDriftError("predecessor donor cache state is missing")
        raw_hash = hashlib.sha256()
        size = 0
        with state_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                size += len(chunk)
                raw_hash.update(chunk)
        if size != int(metadata.get("state_bytes", -1)):
            raise ManifestDriftError("predecessor donor cache size drifted")
        if raw_hash.hexdigest() != metadata.get("state_sha256"):
            raise ManifestDriftError("predecessor donor cache state drifted")
        fingerprints.append(
            {
                "cache_key": metadata["binding"]["cache_key"],
                "cache_checksum": metadata["cache_checksum"],
                "state_sha256": metadata["state_sha256"],
                "state_bytes": size,
                "donors": metadata["binding"]["donors"],
            }
        )
    return fingerprints


def _snapshot(
    *, root: Path, kernel: Mapping[str, Any], kernel_path: Path, artifact_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _, lanes, accounting, discovery = _predecessor_protocol._journal_snapshot(
        root=root,
        parent={},
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
    )
    return lanes, accounting, discovery


def build_migration_receipt(
    *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    _, kernel, kernel_path, artifact_root = _parent_execution(root)
    destination = root / artifact_root
    lease = _kernel_runtime._CollectionLease.acquire(
        destination / ".active-collector.lock"
    )
    try:
        lanes, accounting, discovery = _snapshot(
            root=root,
            kernel=kernel,
            kernel_path=kernel_path,
            artifact_root=artifact_root,
        )
        schedule = list(_parent_runtime._execution_lanes("full"))
        if not lanes or len(lanes) >= len(schedule):
            raise ManifestDriftError(
                "T10.2.4 migration requires a nonterminal complete prefix"
            )
        if any(item["status"] != "COMPLETE" for item in lanes):
            raise ManifestDriftError("T10.2.4 migration prefix is incomplete")
        if [item["lane_id"] for item in lanes] != [
            lane.lane_id for lane in schedule[: len(lanes)]
        ]:
            raise ManifestDriftError("T10.2.4 lanes are not the registered prefix")
        if accounting.get("unknown_intent_count") or not accounting.get(
            "equation_holds"
        ):
            raise JournalIntegrityError("T10.2.4 migration accounting is open")
        if accounting.get("explicitly_unresolved_intent_count"):
            raise JournalIntegrityError("T10.2.4 migration has unresolved intents")
        checkpoint = _read_signed_json(
            destination / _parent_protocol.CHECKPOINT_FILENAME,
            checksum_key="checkpoint_checksum",
        )
        cursor = _read_signed_json(
            destination / _parent_runtime.CURSOR_FILENAME,
            checksum_key="cursor_checksum",
        )
        next_lane = schedule[len(lanes)]
        if cursor.get("open_lane_id") not in (None, next_lane.lane_id):
            raise ManifestDriftError("T10.2.4 cursor escaped the next lane")
        caches = _predecessor_cache_fingerprints(root)
        return signed_payload(
            {
                "format_version": MIGRATION_FORMAT_VERSION,
                "predecessor_t10_2_3_manifest_checksum": predecessor[
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
                    "open_lane_id": cursor.get("open_lane_id"),
                },
                "completed_lane_count": len(lanes),
                "completed_reset_count": sum(len(item["resets"]) for item in lanes),
                "completed_lanes": lanes,
                "completed_lanes_sha256": canonical_sha256(lanes),
                "initial_accounting": accounting,
                "initial_discovery_evidence": discovery,
                "adopted_predecessor_caches": caches,
                "adopted_predecessor_caches_sha256": canonical_sha256(caches),
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
    root = _root(repo_root)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.2.4 migration receipt checksum drifted")
    if receipt.get("format_version") != MIGRATION_FORMAT_VERSION:
        raise ManifestDriftError("T10.2.4 migration receipt format drifted")
    predecessor = _load_predecessor(root)
    if receipt.get("predecessor_t10_2_3_manifest_checksum") != predecessor.get(
        "manifest_checksum"
    ):
        raise ManifestDriftError("T10.2.4 predecessor binding drifted")
    _, kernel, kernel_path, artifact_root = _parent_execution(root)
    if receipt.get("parent_kernel_manifest_checksum") != kernel.get(
        "manifest_checksum"
    ):
        raise ManifestDriftError("T10.2.4 kernel binding drifted")
    lanes, accounting, discovery = _snapshot(
        root=root,
        kernel=kernel,
        kernel_path=kernel_path,
        artifact_root=artifact_root,
    )
    frozen = receipt.get("completed_lanes")
    if not isinstance(frozen, list) or lanes[: len(frozen)] != frozen:
        raise JournalIntegrityError("a frozen T10.2.4 lane or reset changed")
    initial = receipt.get("initial_accounting")
    if not isinstance(initial, Mapping):
        raise ManifestDriftError("T10.2.4 accounting receipt is malformed")
    for key in (
        "authorized_intent_count",
        "sealed_event_count",
        "posterior_update_count",
    ):
        if int(accounting.get(key, -1)) < int(initial.get(key, -1)):
            raise JournalIntegrityError("T10.2.4 live accounting regressed")
    if accounting.get("unknown_intent_count") or not accounting.get("equation_holds"):
        raise JournalIntegrityError("T10.2.4 live accounting is open")
    if accounting.get("explicitly_unresolved_intent_count"):
        raise JournalIntegrityError("T10.2.4 live journal has unresolved intents")
    initial_discovery = receipt.get("initial_discovery_evidence")
    if not isinstance(initial_discovery, Mapping) or int(discovery["count"]) < int(
        initial_discovery.get("count", -1)
    ):
        raise JournalIntegrityError("T10.2.4 discovery evidence regressed")
    current_caches = _predecessor_cache_fingerprints(root)
    if current_caches != receipt.get("adopted_predecessor_caches"):
        raise JournalIntegrityError("an adopted T10.2.3 cache changed")
    return {
        "migration_verified": True,
        "frozen_completed_lanes": len(frozen),
        "current_completed_lanes": len(lanes),
        "frozen_checkpoint_checksum": receipt["initial_checkpoint"]["checksum"],
        "next_lane": receipt["next_lane"],
        "adopted_predecessor_cache_count": len(current_caches),
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
            "predecessor_t10_2_3_manifest_checksum": predecessor[
                "manifest_checksum"
            ],
            "parent_kernel_manifest_checksum": kernel["manifest_checksum"],
            "supersedes_preflight_manifest_checksum": (
                SUPERSEDED_PREFLIGHT_MANIFEST_CHECKSUM
            ),
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
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    manifest = _read_signed_json(source, checksum_key="manifest_checksum")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ManifestDriftError("T10.2.4 manifest format drifted")
    if manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.2.4 manifest status drifted")
    if manifest.get("continuation_policy") != continuation_policy():
        raise ManifestDriftError("T10.2.4 continuation policy drifted")
    if manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.2.4 artifact contract drifted")
    if manifest.get(
        "predecessor_t10_2_3_manifest_checksum"
    ) != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.4 predecessor manifest drifted")
    if manifest.get(
        "parent_kernel_manifest_checksum"
    ) != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.4 kernel manifest drifted")
    if manifest.get(
        "supersedes_preflight_manifest_checksum"
    ) != SUPERSEDED_PREFLIGHT_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.4 preflight supersession drifted")
    receipt = manifest.get("migration_receipt")
    if not isinstance(receipt, Mapping):
        raise ManifestDriftError("T10.2.4 migration receipt is missing")
    materialized = _read_signed_json(
        root / DEFAULT_MIGRATION_RELATIVE_PATH, checksum_key="receipt_checksum"
    )
    if materialized != receipt:
        raise ManifestDriftError("materialized T10.2.4 receipt drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_CODE_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.4 code bytes drifted")
        if manifest.get("document_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_DOCUMENT_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.4 documentation bytes drifted")
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
                output_path=args.manifest, repo_root=args.repo_root
            )
        else:
            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            payload = {
                "status": "READY_T10_2_4_CONTINUATION",
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
