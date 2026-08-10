"""Event-seal recovery migration for SAGE.T10.2.7.

T10.2.7 never resumes or edits the partial T10.2.6 lane.  It freezes that
lane as quarantined predecessor evidence, derives fresh replacement lanes,
and specifies a deterministic execution-manifest overlay that retains every
field of the frozen T10.2.2 scientific kernel while replacing only the
manifest identity and migration receipt used by the recovery journal.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_5_protocol as _zero_failure_protocol
from . import t10_2_6_protocol as _predecessor_protocol
from . import t10_2_6_runtime as _predecessor_runtime

FORMAT_VERSION = "sage-t10.2.7-protocol-v1"
MIGRATION_FORMAT_VERSION = "sage-t10.2.7-migration-receipt-v1"
EXECUTION_CONTRACT_FORMAT_VERSION = "sage-t10.2.7-execution-contract-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_2_7_EVENT_SEAL_RECOVERY"
PREDECESSOR_MANIFEST_CHECKSUM = (
    "4790d8d29b4c33b4453c7dc742024f727c9b7adad7665340bd3d11011dbe0e82"
)
PARENT_KERNEL_MANIFEST_CHECKSUM = (
    "3058989d51f8bc7ab0c65fd201941b20bc4d1cfa7754f1cb207598697594a428"
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_7_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_MIGRATION_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_7_migration_receipt.json"
)
DEFAULT_MIGRATION_PATH = Path(__file__).with_name(DEFAULT_MIGRATION_RELATIVE_PATH.name)
DEFAULT_RECOVERY_ROOT = Path("training") / "sage_t" / "t10_2_7_event_seal_recovery"
MAXIMUM_RECOVERY_LANES = 3
RECOVERY_RESETS_PER_LANE = 4
RECOVERY_ACTIONS_PER_RESET = 64
RECOVERY_MAXIMUM_ACTIONS = (
    MAXIMUM_RECOVERY_LANES * RECOVERY_RESETS_PER_LANE * RECOVERY_ACTIONS_PER_RESET
)

DEFAULT_CODE_FILES = (
    "theory/sage_t/t10_2_7_protocol.py",
    "theory/sage_t/t10_2_7_runtime.py",
    "tests/test_sage_t_t10_2_7_protocol.py",
    "tests/test_sage_t_t10_2_7_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_2_7_EVENT_SEAL_RECOVERY_PROTOCOL.md",
    "reports/SAGE_T10_2_7_EVENT_SEAL_RECOVERY_RUNBOOK.md",
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
        "change_scope": "execution_manifest_event_seal_and_durable_failure_only",
        "parent_scientific_kernel_unchanged": True,
        "parent_collection_journal_read_only": True,
        "t10_2_5_zero_action_failure_read_only": True,
        "t10_2_6_partial_journal_read_only": True,
        "t10_2_6_partial_lane_enters_model_fit": False,
        "t10_2_6_partial_lane_replayed": False,
        "t10_2_6_unsealed_intent_classification": (
            "potentially_executed_environment_call_unattestable"
        ),
        "replacement_scope": "whole_confirmation_lane",
        "fresh_recovery_seeds_required": True,
        "spawn_child_registers_recovery_seeds_before_work_decode": True,
        "execution_manifest_inherits_frozen_kernel_payload": True,
        "execution_manifest_overlays_only": [
            "manifest_checksum",
            "migration_receipt",
        ],
        "first_event_must_seal_before_collection_authorization": True,
        "runner_exception_creates_unresolved_receipt": True,
        "runner_exception_creates_terminal_reset_and_lane_reports": True,
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
        "predecessor_partial_root": (
            _predecessor_protocol.DEFAULT_RECOVERY_ROOT.as_posix()
        ),
        "migration_receipt": DEFAULT_MIGRATION_RELATIVE_PATH.as_posix(),
        "recovery_root": DEFAULT_RECOVERY_ROOT.as_posix(),
        "recovery_journal": "source_collection_journal",
        "recovery_report": "recovery_report.json",
        "accepted_event_ledger": "accepted_source_events.jsonl",
        "accepted_cross_fit_audit": "accepted_cross_fit_audit.json",
        "collection_report": "t10_2_7_collection_report.json",
    }


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


def _io_path(path: Path) -> Path:
    """Use the Windows extended-path namespace for deeply nested journals."""

    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def _load_predecessor(
    root: Path, *, verify_live_migration: bool = False
) -> dict[str, Any]:
    manifest = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=True,
        verify_live_migration=verify_live_migration,
    )
    if manifest.get("manifest_checksum") != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.7 predecessor manifest drifted")
    if manifest.get("parent_kernel_manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.7 parent kernel drifted")
    return manifest


def _kernel_manifest(root: Path) -> dict[str, Any]:
    kernel = _read_signed_json(
        root / _parent_protocol.DEFAULT_KERNEL_MANIFEST_RELATIVE_PATH,
        checksum_key="manifest_checksum",
    )
    if kernel.get("manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.7 materialized scientific kernel drifted")
    if not isinstance(kernel.get("environment_sha256"), str):
        raise ManifestDriftError("T10.2.7 scientific kernel lacks environment_sha256")
    return kernel


def _recovery_lane(game_id: str, seed: int) -> dict[str, Any]:
    identity = {
        "split": "leave_one_game_out_confirmation",
        "game_id": str(game_id),
        "seed": int(seed),
    }
    return {**identity, "lane_id": canonical_sha256(identity)}


def _partial_journal_snapshot(
    root: Path, predecessor: Mapping[str, Any]
) -> dict[str, Any]:
    destination = root / _predecessor_protocol.DEFAULT_RECOVERY_ROOT
    journal_root = _io_path(
        destination / _predecessor_runtime.RECOVERY_JOURNAL_DIRECTORY
    )
    metadata = _read_signed_json(
        journal_root / "journal.json", checksum_key="journal_checksum"
    )
    if metadata.get("manifest_checksum") != predecessor.get("manifest_checksum"):
        raise ManifestDriftError("T10.2.6 partial journal escaped its manifest")

    intent_paths = sorted(journal_root.glob("lanes/*/resets/*/intents/*.json"))
    if len(intent_paths) != 1:
        raise JournalIntegrityError("T10.2.7 requires exactly one T10.2.6 intent")
    intent_payload = _read_signed_json(intent_paths[0], checksum_key="intent_checksum")

    receipt = predecessor["migration_receipt"]
    expected_lane = dict(receipt["recovery_lanes"][0])
    if (
        intent_payload.get("lane") != expected_lane
        or int(intent_payload.get("reset_index", -1)) != 0
        or int(intent_payload.get("step_index", -1)) != 0
        or intent_payload.get("manifest_checksum") != predecessor["manifest_checksum"]
    ):
        raise JournalIntegrityError("T10.2.6 partial intent identity drifted")

    with _predecessor_runtime.recovery_journal_bindings(receipt) as lanes:
        if lanes[0].to_dict() != expected_lane:
            raise ManifestDriftError("T10.2.6 first recovery lane drifted")
        work = _kernel_runtime.reset_work_specs(lanes[0])[0]
        intent = _kernel_runtime.ActionIntent.from_dict(intent_payload)
        journal = _kernel_runtime.DurableCollectionJournal(
            journal_root,
            manifest_checksum=str(predecessor["manifest_checksum"]),
        )
        runtime_accounting = journal.accounting()
        reset_accounting = journal.reset_accounting(work)
        intents = journal.intents_for_reset(work)
        if intents != (intent,):
            raise JournalIntegrityError("T10.2.6 partial intent did not reconstruct")
        topology_drifted = bool(
            runtime_accounting.authorized_intent_count != 1
            or runtime_accounting.sealed_event_count != 0
            or runtime_accounting.explicitly_unresolved_intent_count != 0
            or runtime_accounting.posterior_update_count != 0
            or runtime_accounting.unknown_intent_count not in {0, 1}
            or reset_accounting != runtime_accounting
            or runtime_accounting.equation_holds
            or journal.lane_reports()
            or journal.load_checkpoint() is not None
        )
        if topology_drifted:
            raise JournalIntegrityError(
                "T10.2.6 partial journal topology drifted: "
                + canonical_json(
                    {
                        "accounting": runtime_accounting.to_dict(),
                        "reset_accounting": reset_accounting.to_dict(),
                        "same_accounting": reset_accounting == accounting,
                        "lane_report_count": len(journal.lane_reports()),
                        "checkpoint_present": journal.load_checkpoint() is not None,
                    }
                )
            )

    actual_files = sorted(
        path.relative_to(journal_root).as_posix()
        for path in journal_root.rglob("*")
        if path.is_file()
    )
    expected_files = sorted(
        ["journal.json", intent_paths[0].relative_to(journal_root).as_posix()]
    )
    if actual_files != expected_files:
        raise JournalIntegrityError("T10.2.6 partial journal gained records")
    # Pathlib's Windows extended-path namespace makes the frozen journal's
    # legacy topology checker count the one long intent path as unknown.  The
    # exact two-file allowlist above resolves that ambiguity without mutating
    # the predecessor, so the attested scientific accounting records zero
    # unknown files and preserves the deliberately open 1 = 0 + 0 boundary.
    accounting_payload = {
        "authorized_intent_count": 1,
        "sealed_event_count": 0,
        "explicitly_unresolved_intent_count": 0,
        "unknown_intent_count": 0,
        "maximum_authorized_intents": (
            _predecessor_protocol.RECOVERY_MAXIMUM_ACTIONS
        ),
        "equation_holds": False,
    }
    forbidden_outputs = (
        _predecessor_runtime.RECOVERY_REPORT_FILENAME,
        _predecessor_runtime.ACCEPTED_EVENT_FILENAME,
        _predecessor_runtime.ACCEPTED_AUDIT_FILENAME,
        _predecessor_runtime.COLLECTION_REPORT_FILENAME,
        _kernel_runtime.CHECKPOINT_FILENAME,
    )
    if any((destination / name).exists() for name in forbidden_outputs[:-1]):
        raise JournalIntegrityError("T10.2.6 partial run gained a terminal output")
    if (journal_root / forbidden_outputs[-1]).exists():
        raise JournalIntegrityError("T10.2.6 partial run gained a checkpoint")

    return {
        "journal_checksum": metadata["journal_checksum"],
        "journal_files": actual_files,
        "journal_files_sha256": canonical_sha256(actual_files),
        "intent": intent.to_dict(),
        "intent_relative_path": (
            Path(_predecessor_runtime.RECOVERY_JOURNAL_DIRECTORY)
            / intent_paths[0].relative_to(journal_root)
        ).as_posix(),
        "accounting": accounting_payload,
        "legacy_long_path_unknown_count": runtime_accounting.unknown_intent_count,
        "lane": expected_lane,
        "reset_work_id": work.work_id,
        "terminal_outputs_present": False,
        "checkpoint_present": False,
    }


def _derive_recovery_seeds(anchor: Mapping[str, Any]) -> list[int]:
    blocked = set(
        int(item)
        for item in (
            *_kernel_runtime.DISCOVERY_SEEDS,
            *_kernel_runtime.CONFIRMATION_SEEDS,
            *anchor.get("t10_2_5_recovery_seeds", ()),
            *anchor.get("t10_2_6_recovery_seeds", ()),
        )
    )
    seeds: list[int] = []
    index = 0
    while len(seeds) < MAXIMUM_RECOVERY_LANES:
        digest = canonical_sha256(
            {"anchor": dict(anchor), "t10_2_7_candidate_index": index}
        )
        candidate = 3_000_001 + (int(digest[:12], 16) % 1_000_000)
        if candidate % 2 == 0:
            candidate += 1
        if candidate not in blocked:
            seeds.append(candidate)
            blocked.add(candidate)
        index += 1
    return seeds


def _execution_contract(kernel: Mapping[str, Any]) -> dict[str, Any]:
    inherited = {key: value for key, value in kernel.items() if key != "manifest_checksum"}
    return signed_payload(
        {
            "format_version": EXECUTION_CONTRACT_FORMAT_VERSION,
            "source_kernel_manifest_checksum": kernel["manifest_checksum"],
            "source_kernel_payload_sha256": canonical_sha256(dict(kernel)),
            "inherited_kernel_payload_sha256": canonical_sha256(inherited),
            "required_environment_sha256": kernel["environment_sha256"],
            "overlay": {
                "manifest_checksum": "protocol_manifest.manifest_checksum",
                "migration_receipt": "protocol_manifest.migration_receipt",
            },
            "overlay_keys": ["manifest_checksum", "migration_receipt"],
        },
        checksum_key="contract_checksum",
    )


def build_migration_receipt(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(repo_root)
    # The expensive predecessor terminal topology is revalidated once while
    # freezing. Subsequent T10.2.7 verification binds its signed manifest and
    # independently revalidates the exact partial journal.
    predecessor = _load_predecessor(root, verify_live_migration=True)
    partial = _partial_journal_snapshot(root, predecessor)
    predecessor_receipt = predecessor["migration_receipt"]
    anchor = {
        "predecessor_manifest_checksum": predecessor["manifest_checksum"],
        "predecessor_migration_receipt_checksum": predecessor_receipt[
            "receipt_checksum"
        ],
        "parent_checkpoint_checksum": predecessor_receipt["parent_terminal"][
            "checkpoint_checksum"
        ],
        "t10_2_5_failure_report_checksum": predecessor_receipt[
            "predecessor_failure"
        ]["report_checksum"],
        "partial_journal_checksum": partial["journal_checksum"],
        "partial_intent_checksum": partial["intent"]["intent_checksum"],
        "t10_2_5_recovery_seeds": list(
            predecessor_receipt["recovery_seed_anchor"]["predecessor_recovery_seeds"]
        ),
        "t10_2_6_recovery_seeds": list(predecessor_receipt["recovery_seeds"]),
    }
    seeds = _derive_recovery_seeds(anchor)
    orphan = dict(predecessor_receipt["orphan_lane"])
    return signed_payload(
        {
            "format_version": MIGRATION_FORMAT_VERSION,
            "predecessor_t10_2_6_manifest_checksum": predecessor[
                "manifest_checksum"
            ],
            "parent_kernel_manifest_checksum": PARENT_KERNEL_MANIFEST_CHECKSUM,
            "parent_terminal": dict(predecessor_receipt["parent_terminal"]),
            "t10_2_5_zero_action_failure": dict(
                predecessor_receipt["predecessor_failure"]
            ),
            "t10_2_6_partial_failure": {
                "failure_kind": "missing_scientific_environment_provenance_field",
                "observed_exception": "KeyError:environment_sha256",
                "failure_boundary": "parent_first_physical_event_seal",
                "journal_checksum": partial["journal_checksum"],
                "journal_files": partial["journal_files"],
                "journal_files_sha256": partial["journal_files_sha256"],
                "intent": partial["intent"],
                "intent_relative_path": partial["intent_relative_path"],
                "accounting": partial["accounting"],
                "partial_lane": partial["lane"],
                "reset_work_id": partial["reset_work_id"],
                "potentially_executed_physical_actions": 1,
                "sealed_events": 0,
                "unresolved_receipts": 0,
                "terminal_outputs_present": False,
                "checkpoint_present": False,
                "whole_lane_quarantined": True,
                "fit_authorized": False,
                "replay_authorized": False,
            },
            "orphan_lane": orphan,
            "recovery_seed_anchor": anchor,
            "recovery_seeds": seeds,
            "recovery_lanes": [
                _recovery_lane(str(orphan["game_id"]), seed) for seed in seeds
            ],
            "execution_fix": {
                "kind": "scientific_kernel_payload_plus_recovery_identity_overlay",
                "required_kernel_field": "environment_sha256",
                "first_event_seal_preflight_required": True,
                "runner_exception_fail_closed": True,
            },
            "replay_authorized": False,
            "parent_mutation_authorized": False,
            "predecessor_partial_mutation_authorized": False,
        },
        checksum_key="receipt_checksum",
    )


def verify_migration_receipt_live(
    receipt: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    root = _root(repo_root)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.2.7 migration receipt checksum drifted")
    if receipt.get("format_version") != MIGRATION_FORMAT_VERSION:
        raise ManifestDriftError("T10.2.7 migration receipt format drifted")
    predecessor = _load_predecessor(root)
    if receipt.get("predecessor_t10_2_6_manifest_checksum") != predecessor.get(
        "manifest_checksum"
    ):
        raise ManifestDriftError("T10.2.7 predecessor binding drifted")
    partial = _partial_journal_snapshot(root, predecessor)
    frozen = receipt.get("t10_2_6_partial_failure")
    if not isinstance(frozen, Mapping):
        raise ManifestDriftError("T10.2.7 partial-failure receipt is absent")
    if (
        frozen.get("journal_checksum") != partial["journal_checksum"]
        or frozen.get("journal_files") != partial["journal_files"]
        or frozen.get("intent") != partial["intent"]
        or frozen.get("accounting") != partial["accounting"]
        or frozen.get("whole_lane_quarantined") is not True
        or frozen.get("fit_authorized") is not False
        or frozen.get("replay_authorized") is not False
    ):
        raise JournalIntegrityError("T10.2.6 quarantined evidence changed")
    anchor = receipt.get("recovery_seed_anchor")
    seeds = receipt.get("recovery_seeds")
    if not isinstance(anchor, Mapping) or not isinstance(seeds, list):
        raise ManifestDriftError("T10.2.7 recovery seed receipt is malformed")
    if seeds != _derive_recovery_seeds(anchor):
        raise ManifestDriftError("T10.2.7 recovery seeds drifted")
    expected_lanes = [
        _recovery_lane(str(receipt["orphan_lane"]["game_id"]), int(seed))
        for seed in seeds
    ]
    if receipt.get("recovery_lanes") != expected_lanes:
        raise ManifestDriftError("T10.2.7 recovery lane registry drifted")
    return {
        "migration_verified": True,
        "parent_complete_lanes": 17,
        "parent_complete_resets": 70,
        "t10_2_5_failed_actions": 0,
        "t10_2_6_quarantined_intents": 1,
        "t10_2_6_sealed_events": 0,
        "t10_2_6_potentially_executed_actions": 1,
        "recovery_seeds": list(seeds),
        "maximum_recovery_lanes": MAXIMUM_RECOVERY_LANES,
        "first_event_seal_preflight_required": True,
        "replay_authorized": False,
    }


def build_manifest(
    *, repo_root: str | Path | None = None, migration_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    verify_migration_receipt_live(migration_receipt, repo_root=root)
    kernel = _kernel_manifest(root)
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "status": MANIFEST_STATUS,
            "hash_algorithm": _kernel_protocol.HASH_ALGORITHM,
            "predecessor_t10_2_6_manifest_checksum": predecessor[
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
            "execution_manifest_contract": _execution_contract(kernel),
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
        raise ManifestDriftError("T10.2.7 manifest format drifted")
    if manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.2.7 manifest status drifted")
    if manifest.get("recovery_policy") != recovery_policy():
        raise ManifestDriftError("T10.2.7 recovery policy drifted")
    if manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.2.7 artifact contract drifted")
    if manifest.get(
        "predecessor_t10_2_6_manifest_checksum"
    ) != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.7 predecessor checksum drifted")
    if manifest.get("parent_kernel_manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.7 kernel checksum drifted")
    kernel = _kernel_manifest(root)
    if manifest.get("execution_manifest_contract") != _execution_contract(kernel):
        raise ManifestDriftError("T10.2.7 execution manifest contract drifted")
    receipt = manifest.get("migration_receipt")
    if not isinstance(receipt, Mapping):
        raise ManifestDriftError("T10.2.7 migration receipt is missing")
    materialized = _read_signed_json(
        root / DEFAULT_MIGRATION_RELATIVE_PATH,
        checksum_key="receipt_checksum",
    )
    if materialized != receipt:
        raise ManifestDriftError("materialized T10.2.7 receipt drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_CODE_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.7 code bytes drifted")
        if manifest.get("document_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_DOCUMENT_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.7 documentation bytes drifted")
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
                "status": "READY_T10_2_7_EVENT_SEAL_RECOVERY",
                "manifest_checksum": manifest["manifest_checksum"],
                "execution_manifest_contract_checksum": manifest[
                    "execution_manifest_contract"
                ]["contract_checksum"],
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
