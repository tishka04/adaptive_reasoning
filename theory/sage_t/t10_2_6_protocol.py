"""Preregistered Windows-spawn recovery migration for SAGE.T10.2.6.

T10.2.6 supersedes only the failed replacement-lane orchestration of T10.2.5.
The parent scientific journal, the T10.2.4 donor caches, and the immutable
T10.2.5 zero-action failure remain read-only.  Fresh deterministic replacement
lanes are registered in a new append-only namespace, and their seeds must be
installed inside every spawned child before a work specification is decoded.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_5_protocol as _predecessor_protocol
from . import t10_2_5_runtime as _predecessor_runtime

FORMAT_VERSION = "sage-t10.2.6-protocol-v1"
MIGRATION_FORMAT_VERSION = "sage-t10.2.6-migration-receipt-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_2_6_SPAWN_RECOVERY"
PREDECESSOR_MANIFEST_CHECKSUM = (
    "9cc7588b03a1717aca355c75c7922601d7930bfacfe1d672bc07681c0402cd8a"
)
PREDECESSOR_FAILURE_REPORT_CHECKSUM = (
    "d8c2e96e3ae0e2e8c59c09bdb3254bcb78a0775f94e280a9a2d6d11d6c2ee720"
)
PARENT_KERNEL_MANIFEST_CHECKSUM = (
    "3058989d51f8bc7ab0c65fd201941b20bc4d1cfa7754f1cb207598697594a428"
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_6_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_MIGRATION_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_6_migration_receipt.json"
)
DEFAULT_MIGRATION_PATH = Path(__file__).with_name(DEFAULT_MIGRATION_RELATIVE_PATH.name)
DEFAULT_RECOVERY_ROOT = Path("training") / "sage_t" / "t10_2_6_spawn_recovery"
MAXIMUM_RECOVERY_LANES = 3
RECOVERY_RESETS_PER_LANE = 4
RECOVERY_ACTIONS_PER_RESET = 64
RECOVERY_MAXIMUM_ACTIONS = (
    MAXIMUM_RECOVERY_LANES * RECOVERY_RESETS_PER_LANE * RECOVERY_ACTIONS_PER_RESET
)

DEFAULT_CODE_FILES = (
    "theory/sage_t/t10_2_6_protocol.py",
    "theory/sage_t/t10_2_6_runtime.py",
    "tests/test_sage_t_t10_2_6_protocol.py",
    "tests/test_sage_t_t10_2_6_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_2_6_SPAWN_RECOVERY_PROTOCOL.md",
    "reports/SAGE_T10_2_6_SPAWN_RECOVERY_RUNBOOK.md",
)

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
write_compact_json = _kernel_protocol.write_compact_json
_read_signed_json = _kernel_protocol._read_signed_json
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError


def recovery_policy() -> dict[str, Any]:
    return {
        "change_scope": "windows_spawn_child_registry_only",
        "parent_scientific_kernel_unchanged": True,
        "parent_collection_journal_read_only": True,
        "parent_completed_records_mutated": False,
        "predecessor_t10_2_5_failure_immutable": True,
        "predecessor_failed_attempts_had_zero_actions": True,
        "predecessor_failed_attempts_enter_model_fit": False,
        "predecessor_failed_attempts_replayed": False,
        "spawn_child_registers_recovery_seeds_before_work_decode": True,
        "replacement_scope": "whole_confirmation_lane",
        "replacement_controller_order_preserved": True,
        "replacement_seed_selection": "deterministic_failure_receipt_derived_odd_seeds",
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
        "collect_cli_nonzero_on_failed_gate": True,
        "validation_and_ar25_authority_opened": False,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "parent_collection_root": _parent_protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        "predecessor_manifest": (
            _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH.as_posix()
        ),
        "predecessor_failure_root": (
            _predecessor_protocol.DEFAULT_RECOVERY_ROOT.as_posix()
        ),
        "migration_receipt": DEFAULT_MIGRATION_RELATIVE_PATH.as_posix(),
        "recovery_root": DEFAULT_RECOVERY_ROOT.as_posix(),
        "recovery_journal": "source_collection_journal",
        "recovery_report": "recovery_report.json",
        "accepted_event_ledger": "accepted_source_events.jsonl",
        "accepted_cross_fit_audit": "accepted_cross_fit_audit.json",
        "collection_report": "t10_2_6_collection_report.json",
    }


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


def _load_predecessor(root: Path) -> dict[str, Any]:
    manifest = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
    )
    if manifest.get("manifest_checksum") != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.6 predecessor manifest drifted")
    if manifest.get("parent_kernel_manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.6 parent kernel drifted")
    return manifest


def _recovery_lane(game_id: str, seed: int) -> dict[str, Any]:
    identity = {
        "split": "leave_one_game_out_confirmation",
        "game_id": str(game_id),
        "seed": int(seed),
    }
    return {**identity, "lane_id": canonical_sha256(identity)}


def _derive_recovery_seeds(anchor: Mapping[str, Any]) -> list[int]:
    blocked = set(
        int(item)
        for item in (
            *_kernel_runtime.DISCOVERY_SEEDS,
            *_kernel_runtime.CONFIRMATION_SEEDS,
            *anchor.get("predecessor_recovery_seeds", ()),
        )
    )
    seeds: list[int] = []
    index = 0
    while len(seeds) < MAXIMUM_RECOVERY_LANES:
        digest = canonical_sha256(
            {"anchor": dict(anchor), "t10_2_6_candidate_index": index}
        )
        candidate = 2_000_001 + (int(digest[:12], 16) % 1_000_000)
        if candidate % 2 == 0:
            candidate += 1
        if candidate not in blocked:
            seeds.append(candidate)
            blocked.add(candidate)
        index += 1
    return seeds


def _validate_zero_action_failure(
    *,
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("report_checksum") != PREDECESSOR_FAILURE_REPORT_CHECKSUM:
        raise ManifestDriftError("T10.2.6 predecessor failure checksum drifted")
    if report.get("status") != "FAIL_T10_2_5_RECOVERY":
        raise ManifestDriftError("T10.2.6 requires the exact T10.2.5 failure")
    if report.get("manifest_checksum") != predecessor.get("manifest_checksum"):
        raise ManifestDriftError("T10.2.5 failure escaped its manifest")
    if report.get("accepted_lane") is not None:
        raise ManifestDriftError("T10.2.5 unexpectedly accepted a lane")
    if int(report.get("attempted_lane_count", -1)) != 3:
        raise ManifestDriftError("T10.2.5 failed-attempt count drifted")
    accounting = report.get("accounting")
    if not isinstance(accounting, Mapping):
        raise ManifestDriftError("T10.2.5 failure accounting is absent")
    for key in (
        "authorized_intent_count",
        "sealed_event_count",
        "explicitly_unresolved_intent_count",
        "unknown_intent_count",
    ):
        if int(accounting.get(key, -1)) != 0:
            raise JournalIntegrityError("T10.2.5 failure was not pre-action")
    if accounting.get("equation_holds") is not True:
        raise JournalIntegrityError("T10.2.5 failure accounting is open")
    if report.get("physical_steps_replayed") != 0 or report.get(
        "orphan_events_replayed"
    ) != 0:
        raise JournalIntegrityError("T10.2.5 failure replayed physical evidence")

    lane_reports = checkpoint.get("lane_reports")
    expected_lanes = predecessor["migration_receipt"]["recovery_lanes"]
    if not isinstance(lane_reports, list) or len(lane_reports) != len(expected_lanes):
        raise ManifestDriftError("T10.2.5 failed journal lane count drifted")
    by_id = {str(item.get("lane", {}).get("lane_id", "")): item for item in lane_reports}
    attempts: list[dict[str, Any]] = []
    for expected in expected_lanes:
        lane = by_id.get(str(expected["lane_id"]))
        if not isinstance(lane, Mapping) or lane.get("status") != "ABORTED":
            raise ManifestDriftError("a T10.2.5 failed lane is missing")
        resets = lane.get("resets")
        if not isinstance(resets, list) or len(resets) != 1:
            raise ManifestDriftError("a T10.2.5 failed lane crossed reset zero")
        reset = resets[0]
        if (
            reset.get("status") != "ABORTED"
            or reset.get("stop_reason") != "worker_exited"
            or int(reset.get("issued_intents", -1)) != 0
            or int(reset.get("sealed_events", -1)) != 0
            or int(reset.get("unresolved_intents", -1)) != 0
            or int(reset.get("posterior_updates", -1)) != 0
        ):
            raise JournalIntegrityError("a T10.2.5 failure was not a zero-action spawn exit")
        attempts.append(
            {
                "lane": dict(expected),
                "lane_report_checksum": lane["report_checksum"],
                "reset_index": int(reset["work"]["reset_index"]),
                "reset_work_id": reset["work"]["work_id"],
                "reset_report_checksum": reset["report_checksum"],
                "status": reset["status"],
                "stop_reason": reset["stop_reason"],
                "issued_intents": 0,
                "sealed_events": 0,
                "unresolved_intents": 0,
            }
        )
    return attempts


def _read_failure_snapshot(
    root: Path, predecessor: Mapping[str, Any]
) -> dict[str, Any]:
    destination = root / _predecessor_protocol.DEFAULT_RECOVERY_ROOT
    report = _read_signed_json(
        destination / _predecessor_runtime.RECOVERY_REPORT_FILENAME,
        checksum_key="report_checksum",
    )
    checkpoint = _read_signed_json(
        destination / _kernel_runtime.CHECKPOINT_FILENAME,
        checksum_key="checkpoint_checksum",
    )
    attempts = _validate_zero_action_failure(
        report=report, checkpoint=checkpoint, predecessor=predecessor
    )
    for forbidden in (
        _predecessor_runtime.ACCEPTED_EVENT_FILENAME,
        _predecessor_runtime.ACCEPTED_AUDIT_FILENAME,
        _predecessor_runtime.COLLECTION_REPORT_FILENAME,
    ):
        if (destination / forbidden).exists():
            raise ManifestDriftError("T10.2.5 failure unexpectedly produced accepted output")
    return {
        "report": report,
        "checkpoint": checkpoint,
        "attempts": attempts,
    }


def _read_parent_snapshot(
    root: Path, predecessor: Mapping[str, Any]
) -> dict[str, Any]:
    destination = root / _parent_protocol.DEFAULT_OUTPUT_DIR
    checkpoint = _read_signed_json(
        destination / _parent_protocol.CHECKPOINT_FILENAME,
        checksum_key="checkpoint_checksum",
    )
    report = _read_signed_json(
        destination / _kernel_runtime.COLLECTION_REPORT_FILENAME,
        checksum_key="report_checksum",
    )
    lanes = checkpoint.get("lane_reports")
    if not isinstance(lanes, list) or len(lanes) != 18:
        raise ManifestDriftError("T10.2.6 requires all eighteen parent lane reports")
    orphan_id = predecessor["migration_receipt"]["orphan_lane"]["lane_id"]
    complete_lanes = [item for item in lanes if item.get("status") == "COMPLETE"]
    aborted_lanes = [item for item in lanes if item.get("status") == "ABORTED"]
    resets = [reset for lane in lanes for reset in lane.get("resets", ())]
    if (
        len(complete_lanes) != 17
        or len(aborted_lanes) != 1
        or aborted_lanes[0].get("lane", {}).get("lane_id") != orphan_id
        or sum(item.get("status") == "COMPLETE" for item in resets) != 70
        or sum(item.get("status") == "ABORTED" for item in resets) != 2
        or checkpoint.get("open_lane_id") is not None
        or int(checkpoint.get("physical_steps_replayed_on_resume", -1)) != 0
    ):
        raise JournalIntegrityError("T10.2.6 parent terminal topology drifted")
    if checkpoint.get("manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.6 parent checkpoint escaped its kernel")
    if report.get("status") != "DATA_OR_PROVENANCE_INVALID":
        raise ManifestDriftError("T10.2.6 expected the fail-closed parent sidecar")
    return {
        "checkpoint": checkpoint,
        "report": report,
        "complete_lane_count": len(complete_lanes),
        "complete_reset_count": 70,
        "aborted_lane_id": orphan_id,
    }


def build_migration_receipt(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    failure = _read_failure_snapshot(root, predecessor)
    parent = _read_parent_snapshot(root, predecessor)
    anchor = {
        "predecessor_manifest_checksum": predecessor["manifest_checksum"],
        "predecessor_failure_report_checksum": failure["report"]["report_checksum"],
        "predecessor_failure_checkpoint_checksum": failure["checkpoint"][
            "checkpoint_checksum"
        ],
        "parent_checkpoint_checksum": parent["checkpoint"]["checkpoint_checksum"],
        "failed_attempts_sha256": canonical_sha256(failure["attempts"]),
        "predecessor_recovery_seeds": list(
            predecessor["migration_receipt"]["recovery_seeds"]
        ),
    }
    seeds = _derive_recovery_seeds(anchor)
    orphan = dict(predecessor["migration_receipt"]["orphan_lane"])
    return signed_payload(
        {
            "format_version": MIGRATION_FORMAT_VERSION,
            "predecessor_t10_2_5_manifest_checksum": predecessor[
                "manifest_checksum"
            ],
            "parent_kernel_manifest_checksum": PARENT_KERNEL_MANIFEST_CHECKSUM,
            "parent_terminal": {
                "checkpoint_revision": parent["checkpoint"]["revision"],
                "checkpoint_checksum": parent["checkpoint"]["checkpoint_checksum"],
                "collection_report_checksum": parent["report"]["report_checksum"],
                "collection_report_status": parent["report"]["status"],
                "complete_lane_count": parent["complete_lane_count"],
                "complete_reset_count": parent["complete_reset_count"],
                "aborted_lane_id": parent["aborted_lane_id"],
                "physical_steps_replayed_on_resume": 0,
            },
            "predecessor_failure": {
                "report_checksum": failure["report"]["report_checksum"],
                "status": failure["report"]["status"],
                "checkpoint_revision": failure["checkpoint"]["revision"],
                "checkpoint_checksum": failure["checkpoint"]["checkpoint_checksum"],
                "accounting": dict(failure["report"]["accounting"]),
                "failed_attempts": failure["attempts"],
                "failed_attempts_sha256": canonical_sha256(failure["attempts"]),
                "physical_actions_issued": 0,
                "physical_actions_replayed": 0,
                "fit_authorized": False,
            },
            "orphan_lane": orphan,
            "spawn_failure": {
                "kind": "unregistered_recovery_seed_in_spawned_child",
                "worker_stop_reason": "worker_exited",
                "failure_boundary": "before_reset_work_decode",
                "parent_only_registry_patch_not_inherited_by_spawn": True,
                "fix": "register_recovery_seeds_in_child_before_work_decode",
            },
            "recovery_seed_anchor": anchor,
            "recovery_seeds": seeds,
            "recovery_lanes": [
                _recovery_lane(str(orphan["game_id"]), seed) for seed in seeds
            ],
            "replay_authorized": False,
            "parent_mutation_authorized": False,
            "predecessor_failure_mutation_authorized": False,
        },
        checksum_key="receipt_checksum",
    )


def verify_migration_receipt_live(
    receipt: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    root = _root(repo_root)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.2.6 migration receipt checksum drifted")
    if receipt.get("format_version") != MIGRATION_FORMAT_VERSION:
        raise ManifestDriftError("T10.2.6 migration receipt format drifted")
    predecessor = _load_predecessor(root)
    if receipt.get("predecessor_t10_2_5_manifest_checksum") != predecessor.get(
        "manifest_checksum"
    ):
        raise ManifestDriftError("T10.2.6 predecessor binding drifted")
    failure = _read_failure_snapshot(root, predecessor)
    parent = _read_parent_snapshot(root, predecessor)
    frozen_failure = receipt.get("predecessor_failure")
    frozen_parent = receipt.get("parent_terminal")
    if not isinstance(frozen_failure, Mapping) or not isinstance(frozen_parent, Mapping):
        raise ManifestDriftError("T10.2.6 terminal receipts are malformed")
    if (
        frozen_failure.get("report_checksum")
        != failure["report"]["report_checksum"]
        or frozen_failure.get("checkpoint_checksum")
        != failure["checkpoint"]["checkpoint_checksum"]
        or frozen_failure.get("failed_attempts") != failure["attempts"]
    ):
        raise JournalIntegrityError("T10.2.5 failure evidence changed")
    if (
        frozen_parent.get("checkpoint_checksum")
        != parent["checkpoint"]["checkpoint_checksum"]
        or frozen_parent.get("collection_report_checksum")
        != parent["report"]["report_checksum"]
    ):
        raise JournalIntegrityError("T10.2.6 parent terminal evidence changed")
    anchor = receipt.get("recovery_seed_anchor")
    seeds = receipt.get("recovery_seeds")
    if not isinstance(anchor, Mapping) or not isinstance(seeds, list):
        raise ManifestDriftError("T10.2.6 recovery seed receipt is malformed")
    if seeds != _derive_recovery_seeds(anchor):
        raise ManifestDriftError("T10.2.6 recovery seeds drifted")
    expected_lanes = [
        _recovery_lane(str(receipt["orphan_lane"]["game_id"]), int(seed))
        for seed in seeds
    ]
    if receipt.get("recovery_lanes") != expected_lanes:
        raise ManifestDriftError("T10.2.6 recovery lane registry drifted")
    return {
        "migration_verified": True,
        "parent_complete_lanes": 17,
        "parent_complete_resets": 70,
        "predecessor_failed_attempts": 3,
        "predecessor_failed_actions": 0,
        "recovery_seeds": list(seeds),
        "maximum_recovery_lanes": MAXIMUM_RECOVERY_LANES,
        "spawn_child_registry_required": True,
        "replay_authorized": False,
    }


def build_manifest(
    *, repo_root: str | Path | None = None, migration_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    verify_migration_receipt_live(migration_receipt, repo_root=root)
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "status": MANIFEST_STATUS,
            "hash_algorithm": _kernel_protocol.HASH_ALGORITHM,
            "predecessor_t10_2_5_manifest_checksum": predecessor[
                "manifest_checksum"
            ],
            "parent_kernel_manifest_checksum": PARENT_KERNEL_MANIFEST_CHECKSUM,
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
        raise ManifestDriftError("T10.2.6 manifest format drifted")
    if manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.2.6 manifest status drifted")
    if manifest.get("recovery_policy") != recovery_policy():
        raise ManifestDriftError("T10.2.6 recovery policy drifted")
    if manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.2.6 artifact contract drifted")
    if manifest.get(
        "predecessor_t10_2_5_manifest_checksum"
    ) != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.6 predecessor checksum drifted")
    if manifest.get("parent_kernel_manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.6 kernel checksum drifted")
    receipt = manifest.get("migration_receipt")
    if not isinstance(receipt, Mapping):
        raise ManifestDriftError("T10.2.6 migration receipt is missing")
    materialized = _read_signed_json(
        root / DEFAULT_MIGRATION_RELATIVE_PATH,
        checksum_key="receipt_checksum",
    )
    if materialized != receipt:
        raise ManifestDriftError("materialized T10.2.6 receipt drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_CODE_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.6 code bytes drifted")
        if manifest.get("document_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_DOCUMENT_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.6 documentation bytes drifted")
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
            payload = freeze_manifest(output_path=args.manifest, repo_root=args.repo_root)
        else:
            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            payload = {
                "status": "READY_T10_2_6_SPAWN_RECOVERY",
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
