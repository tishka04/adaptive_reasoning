"""Append-only source-seed registry correction for SAGE.T10.2.8.

T10.2.8 stopped before scientific QA because its lineage adapter reused the
legacy T10.2 seed registry.  T10.2.9 authenticates that fail-closed terminal,
proves that no fit or active validation occurred, and freezes the one permitted
correction: recognize the durable T10.2.1 discovery/confirmation seeds while
validating the immutable T10.2.7 accepted ledger.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_7_protocol as _source_protocol
from . import t10_2_8_protocol as _predecessor_protocol
from . import t10_2_8_runtime as _predecessor_runtime

FORMAT_VERSION = "sage-t10.2.9-protocol-v1"
HANDOFF_FORMAT_VERSION = "sage-t10.2.9-handoff-receipt-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_2_9_OFFLINE_QA"
PREDECESSOR_MANIFEST_CHECKSUM = (
    "38c018b92c48125b060788e9f2694263ae2d856b9268d4364af6ebbf81105eff"
)
PREDECESSOR_TERMINAL_CHECKSUM = (
    "3433eb60b54a41ef7dd8b6599b39e910c18c3f3754f871d446948dcd0558f6b2"
)
PREDECESSOR_LINEAGE_AUDIT_CHECKSUM = (
    "98f6871fe8e19436ad8f7b361d20689e1eabbb9631f828f94c8dc2e1769e7708"
)
PREDECESSOR_QA_REPORT_CHECKSUM = (
    "643a0832f4cf349a6ecc096b770999e88085d18cdcb49b9aa5682316fe894c38"
)
EXPECTED_ADAPTER_ERROR = (
    "DataGateError:source seed/split mismatch: "
    "4cdd1f9b16f7c178b65d4a32fc181ebd6541a4709676e3c2740a3ead65a8ddb1"
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_9_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_HANDOFF_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_9_handoff_receipt.json"
)
DEFAULT_HANDOFF_PATH = Path(__file__).with_name(DEFAULT_HANDOFF_RELATIVE_PATH.name)
DEFAULT_OUTPUT_ROOT = Path("training") / "sage_t" / "t10_2_9_offline_qa"

DEFAULT_CODE_FILES = (
    "theory/sage_t/t10_2_9_protocol.py",
    "theory/sage_t/t10_2_9_runtime.py",
    "tests/test_sage_t_t10_2_9_protocol.py",
    "tests/test_sage_t_t10_2_9_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_2_9_SEED_ADAPTER_PROTOCOL.md",
    "reports/SAGE_T10_2_9_SEED_ADAPTER_RUNBOOK.md",
)

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
write_compact_json = _kernel_protocol.write_compact_json
_read_signed_json = _kernel_protocol._read_signed_json
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError


def correction_policy() -> dict[str, Any]:
    return {
        "change_scope": "durable_source_seed_registry_adapter_only",
        "predecessor_t10_2_8_terminal_read_only": True,
        "source_t10_2_7_accepted_ledger_read_only": True,
        "durable_discovery_seeds": list(_kernel_protocol.DISCOVERY_SEEDS),
        "durable_confirmation_seeds": list(_kernel_protocol.CONFIRMATION_SEEDS),
        "legacy_seed_registry_rejected": True,
        "scientific_qa_computation_unchanged": True,
        "scientific_qa_thresholds_unchanged": True,
        "environment_calls_authorized": 0,
        "physical_actions_authorized": 0,
        "physical_replay_authorized": False,
        "model_fit_authorized": False,
        "source_train_authorized": False,
        "source_validation_authorized": False,
        "ar25_authorized": False,
        "holdout_authorized": False,
        "qa_failure_stops_before_fit": True,
        "qa_pass_only_authorizes_separate_future_protocol": True,
        "terminal_report_write_once": True,
        "compile_cli_nonzero_on_failed_gate": True,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "predecessor_manifest": _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH.as_posix(),
        "predecessor_root": _predecessor_protocol.DEFAULT_OUTPUT_ROOT.as_posix(),
        "predecessor_lineage_audit": _predecessor_runtime.LINEAGE_AUDIT_FILENAME,
        "predecessor_qa_report": _predecessor_runtime.QA_REPORT_FILENAME,
        "predecessor_terminal_report": _predecessor_runtime.TERMINAL_REPORT_FILENAME,
        "source_root": _source_protocol.DEFAULT_RECOVERY_ROOT.as_posix(),
        "handoff_receipt": DEFAULT_HANDOFF_RELATIVE_PATH.as_posix(),
        "output_root": DEFAULT_OUTPUT_ROOT.as_posix(),
        "lineage_audit": "lineage_audit.json",
        "qa_report": "qa_report.json",
        "terminal_report": "t10_2_9_report.json",
    }


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return _kernel_protocol._t10_2.artifact_descriptor(path)


def _load_predecessor(root: Path) -> dict[str, Any]:
    manifest = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=True,
        verify_live_handoff=True,
    )
    if manifest.get("manifest_checksum") != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.9 predecessor manifest drifted")
    return manifest


def _read_predecessor_evidence(
    root: Path, predecessor: Mapping[str, Any]
) -> dict[str, Any]:
    destination = root / _predecessor_protocol.DEFAULT_OUTPUT_ROOT
    lineage_path = destination / _predecessor_runtime.LINEAGE_AUDIT_FILENAME
    qa_path = destination / _predecessor_runtime.QA_REPORT_FILENAME
    terminal_path = destination / _predecessor_runtime.TERMINAL_REPORT_FILENAME
    lineage = _read_signed_json(lineage_path, checksum_key="audit_checksum")
    qa = _read_signed_json(qa_path, checksum_key="report_checksum")
    terminal = _read_signed_json(terminal_path, checksum_key="terminal_checksum")
    lineage_descriptor = _artifact_descriptor(lineage_path)
    qa_descriptor = _artifact_descriptor(qa_path)
    terminal_descriptor = _artifact_descriptor(terminal_path)
    if (
        terminal.get("terminal_checksum") != PREDECESSOR_TERMINAL_CHECKSUM
        or terminal.get("manifest_checksum") != predecessor["manifest_checksum"]
        or terminal.get("status") != "DATA_OR_PROVENANCE_INVALID"
        or terminal.get("lineage_passed") is not False
        or terminal.get("qa_passed") is not False
        or terminal.get("fit_authorized") is not False
        or terminal.get("source_train_authorized") is not False
        or terminal.get("physical_actions_executed") != 0
        or terminal.get("physical_actions_replayed") != 0
        or terminal.get("lineage_audit") != lineage_descriptor
        or terminal.get("qa_report") != qa_descriptor
    ):
        raise ManifestDriftError("T10.2.8 fail-closed terminal drifted")
    if (
        lineage.get("audit_checksum") != PREDECESSOR_LINEAGE_AUDIT_CHECKSUM
        or lineage.get("status") != "DATA_OR_PROVENANCE_INVALID"
        or lineage.get("passed") is not False
        or lineage.get("schema_error") != EXPECTED_ADAPTER_ERROR
        or lineage.get("validation_errors") != []
        or lineage.get("event_count") != 1370
        or lineage.get("checks", {}).get("event_schema_and_provenance") is not False
        or any(
            value is not True
            for key, value in lineage.get("checks", {}).items()
            if key != "event_schema_and_provenance"
        )
    ):
        raise ManifestDriftError("T10.2.8 adapter failure diagnosis drifted")
    if (
        qa.get("report_checksum") != PREDECESSOR_QA_REPORT_CHECKSUM
        or qa.get("status") != "NOT_EVALUATED_LINEAGE_FAILURE"
        or qa.get("event_count") != 0
        or qa.get("metrics") != {}
        or qa.get("behavior_diagnostics") != {}
        or qa.get("fit_authorized") is not False
    ):
        raise ManifestDriftError("T10.2.8 scientific QA abstention drifted")
    firewall = terminal.get("firewall", {})
    if any(firewall.get(key) is not False for key in (
        "source_validation_opened", "ar25_opened", "holdout_opened", "production_authority"
    )):
        raise ManifestDriftError("T10.2.8 firewall evidence drifted")
    registry = predecessor["handoff_receipt"]["lineage_registry"]
    recovery_seeds = sorted(
        int(item["physical_lane"]["seed"])
        for item in registry
        if item["lineage"] == "t10_2_7_recovery"
    )
    if recovery_seeds != [3_119_945]:
        raise ManifestDriftError("T10.2.9 recovery seed registry drifted")
    return {
        "lineage": lineage,
        "qa": qa,
        "terminal": terminal,
        "lineage_descriptor": lineage_descriptor,
        "qa_descriptor": qa_descriptor,
        "terminal_descriptor": terminal_descriptor,
        "recovery_seeds": recovery_seeds,
    }


def build_handoff_receipt(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    evidence = _read_predecessor_evidence(root, predecessor)
    return signed_payload(
        {
            "format_version": HANDOFF_FORMAT_VERSION,
            "predecessor_t10_2_8_manifest_checksum": predecessor["manifest_checksum"],
            "predecessor_terminal": evidence["terminal_descriptor"],
            "predecessor_terminal_checksum": evidence["terminal"]["terminal_checksum"],
            "predecessor_lineage_audit": evidence["lineage_descriptor"],
            "predecessor_lineage_audit_checksum": evidence["lineage"]["audit_checksum"],
            "predecessor_qa_report": evidence["qa_descriptor"],
            "predecessor_qa_report_checksum": evidence["qa"]["report_checksum"],
            "source_handoff_receipt_checksum": predecessor["handoff_receipt"]["receipt_checksum"],
            "source_t10_2_7_manifest_checksum": predecessor["predecessor_t10_2_7_manifest_checksum"],
            "parent_kernel_manifest_checksum": predecessor["parent_kernel_manifest_checksum"],
            "adapter_failure": {
                "classification": "IMPLEMENTATION_ADAPTER_DEFECT",
                "schema_error": evidence["lineage"]["schema_error"],
                "scientific_qa_evaluated": False,
                "model_fit_opened": False,
                "active_validation_opened": False,
            },
            "source_seed_registry": {
                "discovery": list(_kernel_protocol.DISCOVERY_SEEDS),
                "leave_one_game_out_confirmation": list(_kernel_protocol.CONFIRMATION_SEEDS),
                "recovery_confirmation": evidence["recovery_seeds"],
            },
            "qa_gate": dict(predecessor["qa_gate"]),
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
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.2.9 handoff receipt checksum drifted")
    if receipt.get("format_version") != HANDOFF_FORMAT_VERSION:
        raise ManifestDriftError("T10.2.9 handoff receipt format drifted")
    fresh = build_handoff_receipt(repo_root=repo_root)
    if dict(receipt) != fresh:
        raise ManifestDriftError("T10.2.9 live handoff evidence changed")
    return {
        "handoff_verified": True,
        "predecessor_fail_closed": True,
        "scientific_qa_previously_evaluated": False,
        "durable_discovery_seeds": list(_kernel_protocol.DISCOVERY_SEEDS),
        "durable_confirmation_seeds": list(_kernel_protocol.CONFIRMATION_SEEDS),
        "recovery_confirmation_seeds": list(receipt["source_seed_registry"]["recovery_confirmation"]),
        "physical_actions_authorized": 0,
        "model_fit_authorized": False,
    }


def build_manifest(
    *, repo_root: str | Path | None = None, handoff_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    predecessor = _load_predecessor(root)
    verify_handoff_receipt_live(handoff_receipt, repo_root=root)
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "status": MANIFEST_STATUS,
            "hash_algorithm": _kernel_protocol.HASH_ALGORITHM,
            "predecessor_t10_2_8_manifest_checksum": predecessor["manifest_checksum"],
            "registered_phases": ["freeze", "status", "compile"],
            "portable_code_sha256": _kernel_protocol._hash_paths(root, DEFAULT_CODE_FILES, portable=True),
            "document_sha256": _kernel_protocol._hash_paths(root, DEFAULT_DOCUMENT_FILES, portable=True),
            "correction_policy": correction_policy(),
            "artifact_contract": artifact_contract(),
            "qa_gate": dict(predecessor["qa_gate"]),
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
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.2.9 manifest identity drifted")
    if manifest.get("predecessor_t10_2_8_manifest_checksum") != PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.9 predecessor checksum drifted")
    if manifest.get("correction_policy") != correction_policy() or manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.2.9 frozen policy drifted")
    predecessor = _load_predecessor(root)
    if manifest.get("qa_gate") != predecessor["qa_gate"]:
        raise ManifestDriftError("T10.2.9 scientific QA gate drifted")
    receipt = manifest.get("handoff_receipt")
    if not isinstance(receipt, Mapping):
        raise ManifestDriftError("T10.2.9 handoff receipt is absent")
    materialized = _read_signed_json(root / DEFAULT_HANDOFF_RELATIVE_PATH, checksum_key="receipt_checksum")
    if materialized != receipt:
        raise ManifestDriftError("materialized T10.2.9 handoff drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(root, DEFAULT_CODE_FILES, portable=True):
            raise ManifestDriftError("T10.2.9 code bytes drifted")
        if manifest.get("document_sha256") != _kernel_protocol._hash_paths(root, DEFAULT_DOCUMENT_FILES, portable=True):
            raise ManifestDriftError("T10.2.9 documentation bytes drifted")
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
                "status": "READY_T10_2_9_OFFLINE_QA",
                "manifest_checksum": manifest["manifest_checksum"],
                "handoff": verify_handoff_receipt_live(manifest["handoff_receipt"], repo_root=args.repo_root),
            }
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
