"""Preregistered lineage-aware offline QA gate for SAGE.T10.2.8.

T10.2.8 consumes only the immutable accepted outputs of T10.2.7.  It performs
no environment calls, no replay, no fitting, and no validation/AR25 access.
The protocol authenticates the mixed parent/recovery provenance lineages before
computing the frozen T10.2 scientific QA metrics and enforcing the original
pre-fit stop gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_2_protocol as _parent_protocol
from . import t10_2_7_protocol as _predecessor_protocol
from . import t10_2_7_runtime as _predecessor_runtime

FORMAT_VERSION = "sage-t10.2.8-protocol-v1"
HANDOFF_FORMAT_VERSION = "sage-t10.2.8-handoff-receipt-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_2_8_OFFLINE_QA"
PREDECESSOR_MANIFEST_CHECKSUM = (
    "17696b86ec916071369d022aa9e39b7b713babf109aef13bccf8ef1da336805c"
)
PREDECESSOR_COLLECTION_REPORT_CHECKSUM = (
    "f5140e12969dd47e04c29e828020d703249ca4f46ff7d8c2f216d26b1b37b688"
)
PARENT_KERNEL_MANIFEST_CHECKSUM = (
    "3058989d51f8bc7ab0c65fd201941b20bc4d1cfa7754f1cb207598697594a428"
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_8_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_HANDOFF_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_8_handoff_receipt.json"
)
DEFAULT_HANDOFF_PATH = Path(__file__).with_name(DEFAULT_HANDOFF_RELATIVE_PATH.name)
DEFAULT_OUTPUT_ROOT = Path("training") / "sage_t" / "t10_2_8_offline_qa"

DEFAULT_CODE_FILES = (
    "theory/sage_t/t10_2_8_protocol.py",
    "theory/sage_t/t10_2_8_runtime.py",
    "tests/test_sage_t_t10_2_8_protocol.py",
    "tests/test_sage_t_t10_2_8_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_2_8_OFFLINE_QA_PROTOCOL.md",
    "reports/SAGE_T10_2_8_OFFLINE_QA_RUNBOOK.md",
)

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
write_compact_json = _kernel_protocol.write_compact_json
_read_signed_json = _kernel_protocol._read_signed_json
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError


def qa_policy() -> dict[str, Any]:
    return {
        "change_scope": "lineage_aware_offline_compile_and_qa_only",
        "predecessor_manifest_read_only": True,
        "predecessor_collection_read_only": True,
        "accepted_event_ledger_read_only": True,
        "parent_and_recovery_journals_read_only": True,
        "environment_calls_authorized": 0,
        "physical_actions_authorized": 0,
        "physical_replay_authorized": False,
        "model_fit_authorized": False,
        "source_train_authorized": False,
        "source_validation_authorized": False,
        "ar25_authorized": False,
        "holdout_authorized": False,
        "lineage_validation_required_before_qa": True,
        "parent_event_manifest": PARENT_KERNEL_MANIFEST_CHECKSUM,
        "recovery_event_manifest": PREDECESSOR_MANIFEST_CHECKSUM,
        "scientific_qa_thresholds_unchanged": True,
        "qa_failure_stops_before_fit": True,
        "qa_pass_only_authorizes_separate_future_protocol": True,
        "terminal_report_write_once": True,
        "compile_cli_nonzero_on_failed_gate": True,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "predecessor_root": _predecessor_protocol.DEFAULT_RECOVERY_ROOT.as_posix(),
        "predecessor_collection_report": (
            _predecessor_runtime.COLLECTION_REPORT_FILENAME
        ),
        "predecessor_recovery_report": _predecessor_runtime.RECOVERY_REPORT_FILENAME,
        "accepted_event_ledger": _predecessor_runtime.ACCEPTED_EVENT_FILENAME,
        "accepted_cross_fit_audit": _predecessor_runtime.ACCEPTED_AUDIT_FILENAME,
        "predecessor_checkpoint": _kernel_runtime.CHECKPOINT_FILENAME,
        "handoff_receipt": DEFAULT_HANDOFF_RELATIVE_PATH.as_posix(),
        "output_root": DEFAULT_OUTPUT_ROOT.as_posix(),
        "lineage_audit": "lineage_audit.json",
        "qa_report": "qa_report.json",
        "terminal_report": "t10_2_8_report.json",
    }


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


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
        raise ManifestDriftError("T10.2.8 predecessor manifest drifted")
    if manifest.get("parent_kernel_manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.8 parent kernel drifted")
    return manifest


def _kernel_manifest(root: Path) -> dict[str, Any]:
    kernel = _read_signed_json(
        root / _parent_protocol.DEFAULT_KERNEL_MANIFEST_RELATIVE_PATH,
        checksum_key="manifest_checksum",
    )
    if kernel.get("manifest_checksum") != PARENT_KERNEL_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.8 scientific kernel drifted")
    return kernel


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return _kernel_protocol._t10_2.artifact_descriptor(path)


def _lineage_registry(
    collection: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw_lanes = collection.get("accepted_lanes")
    if not isinstance(raw_lanes, list) or len(raw_lanes) != 18:
        raise JournalIntegrityError("T10.2.8 requires eighteen accepted lanes")
    registry: list[dict[str, Any]] = []
    recovery_count = 0
    for fingerprint in raw_lanes:
        if not isinstance(fingerprint, Mapping):
            raise JournalIntegrityError("T10.2.8 accepted lane is malformed")
        physical_recovery = fingerprint.get("physical_recovery_lane")
        logical_replacement = fingerprint.get("logical_lane")
        if physical_recovery is not None or logical_replacement is not None:
            if not isinstance(physical_recovery, Mapping) or not isinstance(
                logical_replacement, Mapping
            ):
                raise JournalIntegrityError("T10.2.8 replacement lineage is incomplete")
            physical = dict(physical_recovery)
            logical = dict(logical_replacement)
            lineage = "t10_2_7_recovery"
            provenance_manifest = predecessor["manifest_checksum"]
            recovery_count += 1
        else:
            lane = fingerprint.get("lane")
            if not isinstance(lane, Mapping):
                raise JournalIntegrityError("T10.2.8 parent lane identity is absent")
            physical = dict(lane)
            logical = dict(lane)
            lineage = "t10_2_2_parent"
            provenance_manifest = PARENT_KERNEL_MANIFEST_CHECKSUM
        registry.append(
            {
                "lineage": lineage,
                "physical_lane": physical,
                "logical_lane": logical,
                "provenance_manifest_checksum": provenance_manifest,
                "expected_event_count": int(fingerprint.get("sealed_events", -1)),
                "lane_report_checksum": str(fingerprint.get("report_checksum", "")),
            }
        )
    if recovery_count != 1:
        raise JournalIntegrityError("T10.2.8 requires exactly one recovery lineage")
    physical_keys = [
        (
            item["physical_lane"]["split"],
            item["physical_lane"]["game_id"],
            int(item["physical_lane"]["seed"]),
        )
        for item in registry
    ]
    logical_ids = [str(item["logical_lane"]["lane_id"]) for item in registry]
    if len(set(physical_keys)) != 18 or len(set(logical_ids)) != 18:
        raise JournalIntegrityError("T10.2.8 lineage registry contains duplicates")
    if sum(item["expected_event_count"] for item in registry) != int(
        collection.get("accepted_event_count", -1)
    ):
        raise JournalIntegrityError("T10.2.8 lineage event accounting drifted")
    return registry


def _read_predecessor_outputs(
    root: Path, predecessor: Mapping[str, Any]
) -> dict[str, Any]:
    destination = root / _predecessor_protocol.DEFAULT_RECOVERY_ROOT
    collection = _read_signed_json(
        destination / _predecessor_runtime.COLLECTION_REPORT_FILENAME,
        checksum_key="report_checksum",
    )
    recovery = _read_signed_json(
        destination / _predecessor_runtime.RECOVERY_REPORT_FILENAME,
        checksum_key="report_checksum",
    )
    audit = _read_signed_json(
        destination / _predecessor_runtime.ACCEPTED_AUDIT_FILENAME,
        checksum_key="audit_checksum",
    )
    checkpoint = _read_signed_json(
        destination / _kernel_runtime.CHECKPOINT_FILENAME,
        checksum_key="checkpoint_checksum",
    )
    ledger_path = destination / _predecessor_runtime.ACCEPTED_EVENT_FILENAME
    audit_path = destination / _predecessor_runtime.ACCEPTED_AUDIT_FILENAME
    ledger_descriptor = _artifact_descriptor(ledger_path)
    audit_descriptor = _artifact_descriptor(audit_path)
    if (
        collection.get("report_checksum")
        != PREDECESSOR_COLLECTION_REPORT_CHECKSUM
        or collection.get("manifest_checksum") != predecessor["manifest_checksum"]
        or collection.get("status") != "T10_2_7_SOURCE_COLLECTION_COMPLETE"
        or collection.get("passed") is not True
        or any(value is not True for value in collection.get("checks", {}).values())
        or int(collection.get("accepted_lane_count", -1)) != 18
        or int(collection.get("accepted_reset_count", -1)) != 72
        or int(collection.get("accepted_event_count", -1)) != 1370
    ):
        raise JournalIntegrityError("T10.2.7 accepted collection gate drifted")
    if (
        recovery.get("status") != "PASS_T10_2_7_RECOVERY"
        or recovery.get("manifest_checksum") != predecessor["manifest_checksum"]
        or int(recovery.get("attempted_lane_count", -1)) != 1
        or recovery.get("physical_steps_replayed") != 0
        or recovery.get("t10_2_6_partial_actions_replayed") != 0
    ):
        raise JournalIntegrityError("T10.2.7 recovery receipt drifted")
    if (
        audit.get("passed") is not True
        or audit.get("manifest_checksum") != predecessor["manifest_checksum"]
        or any(value is not True for value in audit.get("checks", {}).values())
        or len(audit.get("units", ())) != 9
        or collection.get("accepted_cross_fit_audit") != audit_descriptor
        or collection.get("accepted_events") != ledger_descriptor
    ):
        raise JournalIntegrityError("T10.2.7 accepted artifact binding drifted")
    if (
        checkpoint.get("manifest_checksum") != predecessor["manifest_checksum"]
        or checkpoint.get("physical_steps_replayed_on_resume") != 0
        or len(checkpoint.get("lane_reports", ())) != 1
    ):
        raise JournalIntegrityError("T10.2.7 recovery checkpoint drifted")
    registry = _lineage_registry(collection, predecessor)
    return {
        "destination": destination,
        "collection": collection,
        "recovery": recovery,
        "audit": audit,
        "checkpoint": checkpoint,
        "ledger_descriptor": ledger_descriptor,
        "audit_descriptor": audit_descriptor,
        "lineage_registry": registry,
    }


def build_handoff_receipt(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root, verify_live_migration=True)
    outputs = _read_predecessor_outputs(root, predecessor)
    kernel = _kernel_manifest(root)
    collection = outputs["collection"]
    return signed_payload(
        {
            "format_version": HANDOFF_FORMAT_VERSION,
            "predecessor_t10_2_7_manifest_checksum": predecessor[
                "manifest_checksum"
            ],
            "parent_kernel_manifest_checksum": kernel["manifest_checksum"],
            "scientific_environment_sha256": kernel["environment_sha256"],
            "predecessor_collection": {
                "report_checksum": collection["report_checksum"],
                "status": collection["status"],
                "accepted_event_count": collection["accepted_event_count"],
                "accepted_lane_count": collection["accepted_lane_count"],
                "accepted_reset_count": collection["accepted_reset_count"],
                "action_accounting": dict(collection["action_accounting"]),
                "replayed_physical_actions": collection["replayed_physical_actions"],
            },
            "recovery_report_checksum": outputs["recovery"]["report_checksum"],
            "accepted_cross_fit_audit_checksum": outputs["audit"][
                "audit_checksum"
            ],
            "accepted_event_ledger": outputs["ledger_descriptor"],
            "accepted_cross_fit_audit": outputs["audit_descriptor"],
            "checkpoint_checksum": outputs["checkpoint"]["checkpoint_checksum"],
            "lineage_registry": outputs["lineage_registry"],
            "lineage_registry_sha256": canonical_sha256(
                outputs["lineage_registry"]
            ),
            "qa_gate": dict(kernel["qa_gate"]),
            "environment_calls_authorized": 0,
            "physical_actions_authorized": 0,
            "physical_replay_authorized": False,
            "model_fit_authorized": False,
            "source_validation_authorized": False,
            "ar25_authorized": False,
            "holdout_authorized": False,
        },
        checksum_key="receipt_checksum",
    )


def verify_handoff_receipt_live(
    receipt: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    root = _root(repo_root)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.2.8 handoff receipt checksum drifted")
    if receipt.get("format_version") != HANDOFF_FORMAT_VERSION:
        raise ManifestDriftError("T10.2.8 handoff receipt format drifted")
    predecessor = _load_predecessor(root, verify_live_migration=False)
    outputs = _read_predecessor_outputs(root, predecessor)
    kernel = _kernel_manifest(root)
    if (
        receipt.get("predecessor_t10_2_7_manifest_checksum")
        != predecessor["manifest_checksum"]
        or receipt.get("parent_kernel_manifest_checksum")
        != kernel["manifest_checksum"]
        or receipt.get("scientific_environment_sha256")
        != kernel["environment_sha256"]
        or receipt.get("accepted_event_ledger") != outputs["ledger_descriptor"]
        or receipt.get("accepted_cross_fit_audit") != outputs["audit_descriptor"]
        or receipt.get("lineage_registry") != outputs["lineage_registry"]
        or receipt.get("lineage_registry_sha256")
        != canonical_sha256(outputs["lineage_registry"])
        or receipt.get("qa_gate") != kernel["qa_gate"]
    ):
        raise JournalIntegrityError("T10.2.8 live handoff evidence changed")
    collection = receipt.get("predecessor_collection")
    if not isinstance(collection, Mapping) or (
        collection.get("report_checksum")
        != outputs["collection"]["report_checksum"]
        or receipt.get("recovery_report_checksum")
        != outputs["recovery"]["report_checksum"]
        or receipt.get("accepted_cross_fit_audit_checksum")
        != outputs["audit"]["audit_checksum"]
        or receipt.get("checkpoint_checksum")
        != outputs["checkpoint"]["checkpoint_checksum"]
    ):
        raise JournalIntegrityError("T10.2.8 terminal receipt binding changed")
    return {
        "handoff_verified": True,
        "accepted_event_count": 1370,
        "accepted_lane_count": 18,
        "accepted_reset_count": 72,
        "lineage_count": 2,
        "parent_lineage_lane_count": 17,
        "recovery_lineage_lane_count": 1,
        "physical_actions_authorized": 0,
        "model_fit_authorized": False,
        "source_validation_authorized": False,
    }


def build_manifest(
    *, repo_root: str | Path | None = None, handoff_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root, verify_live_migration=False)
    verify_handoff_receipt_live(handoff_receipt, repo_root=root)
    kernel = _kernel_manifest(root)
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "status": MANIFEST_STATUS,
            "hash_algorithm": _kernel_protocol.HASH_ALGORITHM,
            "predecessor_t10_2_7_manifest_checksum": predecessor[
                "manifest_checksum"
            ],
            "parent_kernel_manifest_checksum": kernel["manifest_checksum"],
            "registered_phases": ["freeze", "status", "compile"],
            "portable_code_sha256": _kernel_protocol._hash_paths(
                root, DEFAULT_CODE_FILES, portable=True
            ),
            "document_sha256": _kernel_protocol._hash_paths(
                root, DEFAULT_DOCUMENT_FILES, portable=True
            ),
            "qa_policy": qa_policy(),
            "artifact_contract": artifact_contract(),
            "qa_gate": dict(kernel["qa_gate"]),
            "handoff_receipt": dict(handoff_receipt),
        },
        checksum_key="manifest_checksum",
    )


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    handoff_path: str | Path = DEFAULT_HANDOFF_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(repo_root)
    receipt = build_handoff_receipt(repo_root=root)
    manifest = build_manifest(repo_root=root, handoff_receipt=receipt)
    receipt_path = Path(handoff_path)
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
    verify_live_handoff: bool = True,
) -> dict[str, Any]:
    root = _root(repo_root)
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    manifest = _read_signed_json(source, checksum_key="manifest_checksum")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ManifestDriftError("T10.2.8 manifest format drifted")
    if manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.2.8 manifest status drifted")
    if manifest.get("qa_policy") != qa_policy():
        raise ManifestDriftError("T10.2.8 QA policy drifted")
    if manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.2.8 artifact contract drifted")
    if manifest.get(
        "predecessor_t10_2_7_manifest_checksum"
    ) != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.8 predecessor checksum drifted")
    kernel = _kernel_manifest(root)
    if (
        manifest.get("parent_kernel_manifest_checksum")
        != PARENT_KERNEL_MANIFEST_CHECKSUM
        or manifest.get("qa_gate") != kernel["qa_gate"]
    ):
        raise ManifestDriftError("T10.2.8 scientific QA contract drifted")
    receipt = manifest.get("handoff_receipt")
    if not isinstance(receipt, Mapping):
        raise ManifestDriftError("T10.2.8 handoff receipt is absent")
    materialized = _read_signed_json(
        root / DEFAULT_HANDOFF_RELATIVE_PATH,
        checksum_key="receipt_checksum",
    )
    if materialized != receipt:
        raise ManifestDriftError("materialized T10.2.8 handoff drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_CODE_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.8 code bytes drifted")
        if manifest.get("document_sha256") != _kernel_protocol._hash_paths(
            root, DEFAULT_DOCUMENT_FILES, portable=True
        ):
            raise ManifestDriftError("T10.2.8 documentation bytes drifted")
        _load_predecessor(root, verify_live_migration=False)
    if verify_live_handoff:
        verify_handoff_receipt_live(receipt, repo_root=root)
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
                "status": "READY_T10_2_8_OFFLINE_QA",
                "manifest_checksum": manifest["manifest_checksum"],
                "handoff": verify_handoff_receipt_live(
                    manifest["handoff_receipt"], repo_root=args.repo_root
                ),
            }
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
