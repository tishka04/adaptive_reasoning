"""Corrected, strictly offline QA runtime for SAGE.T10.2.9."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_7_protocol as _source_protocol
from . import t10_2_7_runtime as _source_runtime
from . import t10_2_8_protocol as _predecessor_protocol
from . import t10_2_8_runtime as _predecessor_runtime
from . import t10_2_9_protocol as _protocol

FORMAT_VERSION = "sage-t10.2.9-runtime-v1"
LINEAGE_AUDIT_FORMAT_VERSION = "sage-t10.2.9-lineage-audit-v1"
QA_REPORT_FORMAT_VERSION = "sage-t10.2.9-qa-report-v1"
TERMINAL_REPORT_FORMAT_VERSION = "sage-t10.2.9-terminal-report-v1"
LINEAGE_AUDIT_FILENAME = "lineage_audit.json"
QA_REPORT_FILENAME = "qa_report.json"
TERMINAL_REPORT_FILENAME = "t10_2_9_report.json"

canonical_json = _kernel_protocol.canonical_json
signed_payload = _kernel_protocol.signed_payload
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _predecessor_runtime.JournalIntegrityError
_science = _kernel_protocol._t10_2


def _write_once_payload(
    path: Path, payload: Mapping[str, Any], *, checksum_key: str
) -> None:
    if path.is_file():
        existing = _kernel_protocol._read_signed_json(path, checksum_key=checksum_key)
        if existing != dict(payload):
            raise JournalIntegrityError(f"immutable T10.2.9 artifact drifted: {path}")
        return
    _predecessor_runtime._kernel_runtime._write_once(path, payload)


@contextmanager
def _durable_seed_registry_binding(recovery_seeds: Sequence[int]):
    """Temporarily bind the validator to the durable collection seed registry."""

    original_discovery = _science.DISCOVERY_SEEDS
    original_confirmation = _science.CONFIRMATION_SEEDS
    discovery = tuple(int(seed) for seed in _kernel_protocol.DISCOVERY_SEEDS)
    confirmation = tuple(int(seed) for seed in _kernel_protocol.CONFIRMATION_SEEDS)
    additions = tuple(
        int(seed) for seed in recovery_seeds if int(seed) not in confirmation
    )
    _science.DISCOVERY_SEEDS = discovery
    _science.CONFIRMATION_SEEDS = (*confirmation, *additions)
    try:
        yield
    finally:
        _science.DISCOVERY_SEEDS = original_discovery
        _science.CONFIRMATION_SEEDS = original_confirmation


def _load_execution_context(
    *, manifest_path: str | Path, repo_root: str | Path | None
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    root = _protocol._root(repo_root)
    manifest = _protocol.load_manifest(manifest_path, repo_root=root)
    predecessor = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=True,
        verify_live_handoff=True,
    )
    source = _source_protocol.load_manifest(
        root / _source_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=False,
        verify_live_migration=False,
    )
    if source.get("manifest_checksum") != predecessor.get(
        "predecessor_t10_2_7_manifest_checksum"
    ):
        raise ManifestDriftError("T10.2.9 source manifest drifted")
    kernel = _predecessor_protocol._kernel_manifest(root)
    source_root = root / _source_protocol.DEFAULT_RECOVERY_ROOT
    return root, manifest, predecessor, source, kernel, source_root


def build_lineage_audit(
    *,
    manifest: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    source: Mapping[str, Any],
    kernel: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registry = predecessor["handoff_receipt"]["lineage_registry"]
    recovery_seeds = [
        int(item["physical_lane"]["seed"])
        for item in registry
        if item["lineage"] == "t10_2_7_recovery"
    ]
    with _durable_seed_registry_binding(recovery_seeds):
        predecessor_audit = _predecessor_runtime.build_lineage_audit(
            manifest=predecessor,
            predecessor=source,
            kernel=kernel,
            events=events,
        )
    seed_registry = manifest["handoff_receipt"]["source_seed_registry"]
    checks = dict(predecessor_audit["checks"])
    checks.update(
        {
            "t10_2_8_failure_authenticated": manifest["handoff_receipt"][
                "predecessor_terminal_checksum"
            ]
            == _protocol.PREDECESSOR_TERMINAL_CHECKSUM,
            "durable_discovery_seed_registry": seed_registry["discovery"]
            == list(_kernel_protocol.DISCOVERY_SEEDS),
            "durable_confirmation_seed_registry": seed_registry[
                "leave_one_game_out_confirmation"
            ]
            == list(_kernel_protocol.CONFIRMATION_SEEDS),
            "recovery_seed_registry": sorted(seed_registry["recovery_confirmation"])
            == sorted(recovery_seeds),
            "scientific_qa_previously_abstained": manifest["handoff_receipt"][
                "adapter_failure"
            ]["scientific_qa_evaluated"]
            is False,
        }
    )
    passed = all(checks.values())
    return signed_payload(
        {
            "format_version": LINEAGE_AUDIT_FORMAT_VERSION,
            "phase": "corrected_lineage_audit",
            "status": "PASS_T10_2_9_LINEAGE" if passed else "DATA_OR_PROVENANCE_INVALID",
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": manifest["handoff_receipt"]["receipt_checksum"],
            "source_t10_2_8_manifest_checksum": predecessor["manifest_checksum"],
            "source_t10_2_7_manifest_checksum": source["manifest_checksum"],
            "predecessor_corrected_audit_checksum": predecessor_audit["audit_checksum"],
            "event_count": predecessor_audit["event_count"],
            "event_ids_sha256": predecessor_audit["event_ids_sha256"],
            "lineage_event_counts": dict(predecessor_audit["lineage_event_counts"]),
            "registered_lane_count": predecessor_audit["registered_lane_count"],
            "source_seed_registry": dict(seed_registry),
            "validation_errors": list(predecessor_audit["validation_errors"]),
            "schema_error": predecessor_audit["schema_error"],
            "checks": checks,
            "passed": passed,
            "firewall": dict(predecessor_audit["firewall"]),
        },
        checksum_key="audit_checksum",
    )


def build_qa_report(
    *,
    manifest: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Wrap the unchanged T10.2.8 QA computation under the corrected protocol."""

    predecessor_qa = _predecessor_runtime.build_qa_report(
        manifest=predecessor,
        events=events,
    )
    passed = predecessor_qa["passed"] is True
    return signed_payload(
        {
            "format_version": QA_REPORT_FORMAT_VERSION,
            "phase": "offline_qa",
            "status": "PASS_T10_2_9_QA" if passed else "FAIL_T10_2_9_QA",
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": manifest["handoff_receipt"]["receipt_checksum"],
            "source_t10_2_8_manifest_checksum": predecessor["manifest_checksum"],
            "unchanged_qa_computation_checksum": predecessor_qa["report_checksum"],
            "event_count": predecessor_qa["event_count"],
            "event_ids_sha256": predecessor_qa["event_ids_sha256"],
            "metrics": dict(predecessor_qa["metrics"]),
            "behavior_diagnostics": dict(predecessor_qa["behavior_diagnostics"]),
            "checks": dict(predecessor_qa["checks"]),
            "failed_checks": list(predecessor_qa["failed_checks"]),
            "passed": passed,
            "fit_authorized": False,
            "firewall": dict(predecessor_qa["firewall"]),
        },
        checksum_key="report_checksum",
    )


def _not_evaluated_qa(
    *, manifest: Mapping[str, Any], lineage_audit: Mapping[str, Any]
) -> dict[str, Any]:
    return signed_payload(
        {
            "format_version": QA_REPORT_FORMAT_VERSION,
            "phase": "offline_qa",
            "status": "NOT_EVALUATED_LINEAGE_FAILURE",
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": manifest["handoff_receipt"]["receipt_checksum"],
            "lineage_audit_checksum": lineage_audit["audit_checksum"],
            "event_count": 0,
            "metrics": {},
            "behavior_diagnostics": {},
            "checks": {"lineage_validated_before_qa": False},
            "failed_checks": ["lineage_validated_before_qa"],
            "passed": False,
            "fit_authorized": False,
            "firewall": {
                "environment_calls": 0,
                "physical_actions": 0,
                "physical_replay": 0,
                "model_fit_opened": False,
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
            },
        },
        checksum_key="report_checksum",
    )


def compile_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root, manifest, predecessor, source, kernel, source_root = _load_execution_context(
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    destination = root / _protocol.DEFAULT_OUTPUT_ROOT
    terminal_path = destination / TERMINAL_REPORT_FILENAME
    if terminal_path.is_file():
        return _kernel_protocol._read_signed_json(terminal_path, checksum_key="terminal_checksum")
    events = _predecessor_runtime._read_accepted_events(
        destination=source_root,
        receipt=predecessor["handoff_receipt"],
    )
    lineage = build_lineage_audit(
        manifest=manifest,
        predecessor=predecessor,
        source=source,
        kernel=kernel,
        events=events,
    )
    qa = (
        build_qa_report(manifest=manifest, predecessor=predecessor, events=events)
        if lineage["passed"] is True
        else _not_evaluated_qa(manifest=manifest, lineage_audit=lineage)
    )
    destination.mkdir(parents=True, exist_ok=True)
    lineage_path = destination / LINEAGE_AUDIT_FILENAME
    qa_path = destination / QA_REPORT_FILENAME
    _write_once_payload(lineage_path, lineage, checksum_key="audit_checksum")
    _write_once_payload(qa_path, qa, checksum_key="report_checksum")
    lineage_passed = lineage["passed"] is True
    qa_passed = qa["passed"] is True
    terminal = signed_payload(
        {
            "format_version": TERMINAL_REPORT_FORMAT_VERSION,
            "phase": "compile",
            "status": (
                "PASS_T10_2_9_QA_READY_FOR_SEPARATE_SOURCE_TRAIN_PROTOCOL"
                if lineage_passed and qa_passed
                else "FAIL_T10_2_9_QA_STOP_BEFORE_FIT"
                if lineage_passed
                else "DATA_OR_PROVENANCE_INVALID"
            ),
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": manifest["handoff_receipt"]["receipt_checksum"],
            "predecessor_t10_2_8_terminal_checksum": _protocol.PREDECESSOR_TERMINAL_CHECKSUM,
            "lineage_audit": _protocol._artifact_descriptor(lineage_path),
            "lineage_audit_checksum": lineage["audit_checksum"],
            "qa_report": _protocol._artifact_descriptor(qa_path),
            "qa_report_checksum": qa["report_checksum"],
            "lineage_passed": lineage_passed,
            "qa_passed": qa_passed,
            "passed": lineage_passed and qa_passed,
            "failed_qa_checks": list(qa.get("failed_checks", ())),
            "fit_authorized": False,
            "source_train_authorized": False,
            "next_protocol_authorized": lineage_passed and qa_passed,
            "stop_before_fit": not (lineage_passed and qa_passed),
            "physical_actions_executed": 0,
            "physical_actions_replayed": 0,
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="terminal_checksum",
    )
    _write_once_payload(terminal_path, terminal, checksum_key="terminal_checksum")
    return terminal


def status_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root, manifest, *_ = _load_execution_context(manifest_path=manifest_path, repo_root=repo_root)
    terminal_path = root / _protocol.DEFAULT_OUTPUT_ROOT / TERMINAL_REPORT_FILENAME
    if terminal_path.is_file():
        terminal = _kernel_protocol._read_signed_json(terminal_path, checksum_key="terminal_checksum")
        return {
            "status": "COMPLETE_T10_2_9_OFFLINE_QA",
            "manifest_checksum": manifest["manifest_checksum"],
            "terminal_status": terminal["status"],
            "terminal_checksum": terminal["terminal_checksum"],
            "lineage_passed": terminal["lineage_passed"],
            "qa_passed": terminal["qa_passed"],
            "fit_authorized": False,
            "source_validation_opened": False,
            "ar25_opened": False,
        }
    return {
        "status": "READY_T10_2_9_OFFLINE_QA",
        "manifest_checksum": manifest["manifest_checksum"],
        "handoff": _protocol.verify_handoff_receipt_live(manifest["handoff_receipt"], repo_root=root),
        "physical_actions_authorized": 0,
        "fit_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("status", "compile"))
    parser.add_argument("--manifest", default=str(_protocol.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = (
            status_phase(manifest_path=args.manifest, repo_root=args.repo_root)
            if args.phase == "status"
            else compile_phase(manifest_path=args.manifest, repo_root=args.repo_root)
        )
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    if args.phase == "compile" and payload.get("status") != (
        "PASS_T10_2_9_QA_READY_FOR_SEPARATE_SOURCE_TRAIN_PROTOCOL"
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
