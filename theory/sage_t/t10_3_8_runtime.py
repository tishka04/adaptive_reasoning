"""Durable witness-gate adjudication and continuation for SAGE.T10.3.8."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import t10_3_2_runtime as durable
from . import t10_3_5_runtime as shell
from . import t10_3_6_runtime as base
from . import t10_3_8_protocol as protocol
from .goal_directed_v10_3_2 import ProgressProgramRegistry
from .goal_directed_v10_3_5 import ScheduledUnifiedCognitiveController, scheduled_unified_config
from .goal_directed_v10_3_7 import StableFreshPathSageTController

AUDIT_FILENAME = "offline_audit.json"
ADJUDICATION_FILENAME = base.WITNESS_REPORT_FILENAME
CORE_REPORT_FILENAME = base.CORE_REPORT_FILENAME
REPRODUCTION_REPORT_FILENAME = base.REPRODUCTION_REPORT_FILENAME
SEQUENCE_REPORT_FILENAME = base.SEQUENCE_REPORT_FILENAME
COMPILE_REPORT_FILENAME = base.COMPILE_REPORT_FILENAME
CONFIRMATION_REPORT_FILENAME = base.CONFIRMATION_REPORT_FILENAME
TERMINAL_REPORT_FILENAME = base.TERMINAL_REPORT_FILENAME


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = protocol.sha256_payload(result)
    return result


def _destination(root: Path) -> Path:
    return root.resolve() / protocol.DEFAULT_OUTPUT_DIR


def _artifact_path(root: Path, filename: str) -> Path:
    return _destination(root) / filename


def _parent_report(root: Path) -> dict[str, Any]:
    return protocol._read_signed(
        root
        / "training"
        / "sage_t"
        / "t10_3_7_stable_successor_recovery"
        / "canonical_witness_report.json",
        "report_checksum",
    )


def _controller_pair(
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None,
) -> tuple[ScheduledUnifiedCognitiveController, StableFreshPathSageTController | None]:
    if work.arm == "unified_sage_t_off":
        return (
            ScheduledUnifiedCognitiveController(
                work.game_id,
                config=scheduled_unified_config(sage_t_authority_mode="off"),
            ),
            None,
        )
    phase = "confirmation" if work.phase == "confirm" else "discovery"
    goal = StableFreshPathSageTController(
        phase=phase,
        registry=registry,
        registry_checksum=registry_checksum,
        attestation_scope=work.work_id,
        exploration_offset=work.reset_index,
        prefer_mixed=work.phase == "discover-sequence",
    )
    return (
        ScheduledUnifiedCognitiveController(
            work.game_id,
            config=scheduled_unified_config(sage_t_authority_mode="active"),
            sage_t_controller=goal,
        ),
        goal,
    )


@contextmanager
def _contracts() -> Iterator[None]:
    old_base_protocol = base.protocol
    old_base_pair = base._controller_pair
    old_shell_protocol = shell.protocol
    old_shell_pair = shell._controller_pair
    old_durable_protocol = durable.protocol
    base.protocol = protocol
    base._controller_pair = _controller_pair
    shell.protocol = protocol
    shell._controller_pair = _controller_pair
    durable.protocol = protocol
    try:
        yield
    finally:
        durable.protocol = old_durable_protocol
        shell._controller_pair = old_shell_pair
        shell.protocol = old_shell_protocol
        base._controller_pair = old_base_pair
        base.protocol = old_base_protocol


def _replace_version(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("t10.3.6", "t10.3.8").replace("T10_3_6", "T10_3_8")
    if isinstance(value, list):
        return [_replace_version(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_version(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _replace_version(item) for key, item in value.items()}
    return value


@contextmanager
def _versioned_writes() -> Iterator[None]:
    original = protocol.write_json_once

    def write(path: Path, payload: Mapping[str, Any]) -> None:
        transformed = _replace_version(dict(payload))
        checksum_field = next(
            (
                key
                for key in ("report_checksum", "audit_checksum", "preflight_checksum")
                if key in transformed
            ),
            None,
        )
        if checksum_field is not None:
            transformed.pop(checksum_field, None)
            transformed[checksum_field] = protocol.sha256_payload(transformed)
        original(path, transformed)

    protocol.write_json_once = write
    try:
        yield
    finally:
        protocol.write_json_once = original


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    return durable._read_signed(_artifact_path(root, filename), checksum_field)


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent = _parent_report(root)
    parent_report_path = (
        root
        / "training"
        / "sage_t"
        / "t10_3_7_stable_successor_recovery"
        / "canonical_witness_report.json"
    )
    false_checks = tuple(
        sorted(key for key, value in parent.get("checks", {}).items() if value is False)
    )
    other_checks = {
        key: value
        for key, value in parent.get("checks", {}).items()
        if key != "historical_grounded_actions_loaded"
    }
    levels = parent.get("metrics", {}).get("levels", {})
    contract = manifest["adjudication_contract"]
    checks = {
        "parent_report_file_bound": protocol.file_sha256(parent_report_path)
        == protocol.PARENT_ARTIFACTS["t10_3_7_witness_report"]["sha256"],
        "unique_false_check": false_checks == ("historical_grounded_actions_loaded",),
        "expected_negative_value": parent["checks"]["historical_grounded_actions_loaded"] is False,
        "all_other_checks_true": bool(other_checks) and all(other_checks.values()),
        "level_each_core_game": all(int(levels.get(game, 0)) >= 1 for game in protocol.CORE_GAMES),
        "parent_passed_false": parent.get("passed") is False,
        "parent_verdict_is_aggregation_miss": parent.get("verdict") == "CANONICAL_WITNESS_MISS",
        "normalization_preregistered": contract["normalized_positive_gate"]
        == "historical_grounded_actions_absent",
        "no_recollection": contract["witness_recollection_authorized"] is False,
        "no_parent_fit": contract["parent_events_fit_authorized"] is False,
        "source_firewall_closed": not any(manifest["firewall"].values()),
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.8-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_report_checksum": parent["report_checksum"],
            "checks": checks,
            "physical_actions": 0,
            "status": "PASS_T10_3_8_OFFLINE_AUDIT" if all(checks.values()) else "INVALID_PROVENANCE",
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.8 adjudication audit failed")
    return payload


def adjudicate(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    audit_payload = _read_signed(root, AUDIT_FILENAME, "audit_checksum")
    if audit_payload.get("status") != "PASS_T10_3_8_OFFLINE_AUDIT":
        raise protocol.ScientificGateMiss("offline audit forbids witness adjudication")
    parent = _parent_report(root)
    normalized_checks = {
        key: value
        for key, value in parent["checks"].items()
        if key != "historical_grounded_actions_loaded"
    }
    normalized_checks["historical_grounded_actions_absent"] = (
        parent["checks"]["historical_grounded_actions_loaded"] is False
    )
    passed = all(normalized_checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.8-canonical-witness-adjudication-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "adjudicate",
            "parent_manifest_checksum": parent["manifest_checksum"],
            "parent_report_checksum": parent["report_checksum"],
            "canonical_descriptors": parent["canonical_descriptors"],
            "metrics": {
                **dict(parent["metrics"]),
                "parent_physical_actions": int(parent["metrics"]["actions"]),
                "new_physical_actions": 0,
                "witness_recollection_actions": 0,
            },
            "checks": normalized_checks,
            "parent_false_checks": ["historical_grounded_actions_loaded"],
            "normalization": {
                "from": {"historical_grounded_actions_loaded": False},
                "to": {"historical_grounded_actions_absent": True},
                "semantic_change": False,
                "boolean_polarity_correction": True,
            },
            "receipt_checksums": list(parent["receipt_checksums"]),
            "adjudication_only": True,
            "parent_events_used_for_training": 0,
            "physical_actions_replayed": 0,
            "passed": passed,
            "verdict": "PASS_T10_3_8_CANONICAL_WITNESS_ADJUDICATION"
            if passed
            else "CANONICAL_WITNESS_ADJUDICATION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, ADJUDICATION_FILENAME), report)
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def _delegate(root: Path, manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    adjudication = _read_signed(root, ADJUDICATION_FILENAME, "report_checksum")
    if adjudication.get("passed") is not True:
        raise protocol.ScientificGateMiss("witness adjudication forbids physical continuation")
    with _contracts(), _versioned_writes():
        if phase in {"discover-core", "reproduce-core"}:
            try:
                base.run_core_phase(root, manifest, phase)
            except protocol.ScientificGateMiss:
                pass
            result = _read_signed(
                root,
                CORE_REPORT_FILENAME if phase == "discover-core" else REPRODUCTION_REPORT_FILENAME,
                "report_checksum",
            )
        elif phase == "discover-sequence":
            try:
                base.run_sequence(root, manifest)
            except protocol.ScientificGateMiss:
                pass
            result = _read_signed(root, SEQUENCE_REPORT_FILENAME, "report_checksum")
        elif phase == "compile":
            try:
                base.compile_registry(root, manifest)
            except protocol.ScientificGateMiss:
                pass
            result = _read_signed(root, COMPILE_REPORT_FILENAME, "report_checksum")
        elif phase == "confirm":
            try:
                base.run_confirmation(root, manifest)
            except protocol.ScientificGateMiss:
                pass
            result = _read_signed(root, CONFIRMATION_REPORT_FILENAME, "report_checksum")
        else:
            raise ValueError(f"unsupported delegated phase: {phase}")
    if result.get("passed") is not True:
        raise protocol.ScientificGateMiss(str(result.get("verdict", phase)))
    return result


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    definitions = (
        ("audit", AUDIT_FILENAME, "audit_checksum"),
        ("adjudication", ADJUDICATION_FILENAME, "report_checksum"),
        ("core", CORE_REPORT_FILENAME, "report_checksum"),
        ("reproduction", REPRODUCTION_REPORT_FILENAME, "report_checksum"),
        ("sequence", SEQUENCE_REPORT_FILENAME, "report_checksum"),
        ("compile", COMPILE_REPORT_FILENAME, "report_checksum"),
        ("confirmation", CONFIRMATION_REPORT_FILENAME, "report_checksum"),
    )
    artifacts = {}
    for name, filename, checksum in definitions:
        path = _artifact_path(root, filename)
        artifacts[name] = durable._read_signed(path, checksum) if path.is_file() else None
    if artifacts["audit"] is None or artifacts["audit"].get("status") != "PASS_T10_3_8_OFFLINE_AUDIT":
        verdict = "INVALID_PROVENANCE"
    elif artifacts["adjudication"] is None or artifacts["adjudication"].get("passed") is not True:
        verdict = "CANONICAL_WITNESS_ADJUDICATION_MISS"
    elif artifacts["core"] is None or artifacts["core"].get("passed") is not True:
        verdict = "CORE_DISCOVERY_MISS"
    elif artifacts["reproduction"] is None or artifacts["reproduction"].get("passed") is not True:
        verdict = "CORE_REPRODUCTION_MISS"
    elif artifacts["sequence"] is None or artifacts["sequence"].get("passed") is not True:
        verdict = "MIXED_SEQUENCE_MISS"
    elif artifacts["compile"] is None or artifacts["compile"].get("passed") is not True:
        verdict = "REGISTRY_REPRODUCTION_MISS"
    elif artifacts["confirmation"] is None or artifacts["confirmation"].get("passed") is not True:
        verdict = "SOURCE_CONFIRMATION_MISS"
    else:
        verdict = "PASS_T10_3_8_FUNCTIONAL_END_TO_END_SOURCE"
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.8-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "artifacts": {
                name: None if value is None else next(
                    (value[key] for key in ("audit_checksum", "report_checksum") if key in value),
                    None,
                )
                for name, value in artifacts.items()
            },
            "accounting": accounting,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
            "witness_recollection_actions": 0,
            "latency_is_telemetry_only": True,
            "firewall": manifest["firewall"],
            "physical_actions_replayed": 0,
            "production_authority": False,
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, TERMINAL_REPORT_FILENAME), report)
    return report


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _contracts():
        payload = dict(durable.status(root, manifest))
    payload["protocol"] = "SAGE.T10.3.8"
    payload["adjudication_contract"] = manifest["adjudication_contract"]
    payload["maximum_actions"] = protocol.TOTAL_MAXIMUM_ACTIONS
    payload["maximum_resets"] = protocol.TOTAL_RESETS
    return payload


def _emit(payload: Mapping[str, Any]) -> None:
    print(_canonical(payload), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze", "status", "audit", "adjudicate", "discover-core",
            "reproduce-core", "discover-sequence", "compile", "confirm", "report",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.phase == "freeze":
            manifest, migration = protocol.freeze_manifest(root)
            _emit(
                {
                    "phase": "freeze",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "migration_receipt_checksum": migration["receipt_checksum"],
                    "status": manifest["status"],
                }
            )
            return 0
        manifest = protocol.load_manifest(root)
        if args.phase == "status":
            _emit(status(root, manifest))
            return 0
        if args.phase == "audit":
            _emit(audit(root, manifest))
            return 0
        if args.phase == "adjudicate":
            _emit(adjudicate(root, manifest))
            return 0
        if args.phase in {"discover-core", "reproduce-core", "discover-sequence", "compile", "confirm"}:
            _emit(_delegate(root, manifest, args.phase))
            return 0
        report = terminal_report(root, manifest)
        _emit(report)
        return 0 if report["verdict"] == "PASS_T10_3_8_FUNCTIONAL_END_TO_END_SOURCE" else 3
    except protocol.ScientificGateMiss as exc:
        _emit({"phase": args.phase, "error": str(exc), "exit_code": 3})
        return 3
    except (protocol.IntegrityError, OSError, ValueError, KeyError) as exc:
        _emit({"phase": args.phase, "error": f"{type(exc).__name__}:{exc}", "exit_code": 2})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "adjudicate", "audit", "main", "status", "terminal_report",
]
