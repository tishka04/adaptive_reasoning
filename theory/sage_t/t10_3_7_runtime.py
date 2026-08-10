"""Durable stable-successor recovery runtime for SAGE.T10.3.7."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import goal_directed_v10_3_7 as stable_module
from . import t10_3_2_runtime as durable
from . import t10_3_5_runtime as shell
from . import t10_3_6_runtime as base
from . import t10_3_7_protocol as protocol
from .contracts import AbstractState, ActionCandidate
from .goal_directed_v10_3_2 import GoalDirectedOption, OptionStep, ProgressProgramRegistry
from .goal_directed_v10_3_3 import DYNAMIC_SUCCESSOR
from .goal_directed_v10_3_5 import ScheduledUnifiedCognitiveController, scheduled_unified_config
from .goal_directed_v10_3_7 import StableFreshPathSageTController
from .progress_witness_v10 import CandidateMacro, GroundedAction

AUDIT_FILENAME = base.AUDIT_FILENAME
PREFLIGHT_FILENAME = base.PREFLIGHT_FILENAME
WITNESS_REPORT_FILENAME = base.WITNESS_REPORT_FILENAME
CORE_REPORT_FILENAME = base.CORE_REPORT_FILENAME
REPRODUCTION_REPORT_FILENAME = base.REPRODUCTION_REPORT_FILENAME
SEQUENCE_REPORT_FILENAME = base.SEQUENCE_REPORT_FILENAME
COMPILE_REPORT_FILENAME = base.COMPILE_REPORT_FILENAME
CONFIRMATION_REPORT_FILENAME = base.CONFIRMATION_REPORT_FILENAME
TERMINAL_REPORT_FILENAME = base.TERMINAL_REPORT_FILENAME

EXPECTED_SU15_WAYPOINT_CHECKSUMS = (
    "7b36e22d3032bc45ef4f44863673c5b2908053832d243abe81c778e6de75ac63",
    "6462f19598d368efc3d68bd6f6144790412541a36f02285fdc6c991c366dc716",
    "a8ceed62a801b6d9b1821ea2a926057b7db16d605f8748f32975c4e1651c0b60",
    "bfe881031f87e02d9c6b936929cc45eb40ea62c265a1d6e796e4ca77a2bb4f1e",
    "7e7363b949d19d5c6333e2e7042fcb687053a958ff614aff7afb268eb409b16c",
    "1c3b15fbdd39d0bc38695645bd5b3109cb407d9b8e2e74812e3192d0dc3c43cc",
    "d13d47c7440ff21b5a12b91dc569ccdbd2fbc9a1095eb5ec7529ee2c583cac3b",
    "e9451ef7bf32343cdf7b3e9b76eb888ffbf1937bded111c5ab88dedfdd7ab078",
    "ae576fa1a405d100efd4e2a43b3157eb298eccb269fd78a1005f0af2d8cc7b5b",
    "6a357d3a3ee7bba19bb05561ae363ec49e15af77238f9c4256aa90cc51e4ddff",
)


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
    witness = protocol.WITNESS_PROGRAMS.get(work.game_id) if work.phase == "witness-core" else None
    phase = "confirmation" if work.phase == "confirm" else "preflight" if witness else "discovery"
    goal = StableFreshPathSageTController(
        phase=phase,
        registry=registry,
        registry_checksum=registry_checksum,
        attestation_scope=work.work_id,
        exploration_offset=work.reset_index,
        witness_schema=None if witness is None else str(witness["macro_schema"]),
        witness_horizon=None if witness is None else int(witness["horizon"]),
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
        return value.replace("t10.3.6", "t10.3.7").replace("T10_3_6", "T10_3_7")
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
                for key in (
                    "report_checksum", "audit_checksum", "preflight_checksum"
                )
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


def _parent_su15_sequences(root: Path) -> list[list[str]]:
    parent_root = root / "training" / "sage_t" / "t10_3_6_functional_end_to_end" / "journal"
    receipts = [
        protocol.parent._read_signed(path, "receipt_checksum")
        for path in sorted((parent_root / "branches").rglob("*.json"))
    ]
    output = []
    for receipt in receipts:
        if receipt.get("game_id") != "su15-4c352900":
            continue
        rows = [
            protocol.parent._read_signed(path, "intent_checksum")
            for path in sorted((parent_root / "intents" / str(receipt["work_id"])).glob("*.json"))
        ]
        output.append([str(row["action"]["argument_checksum"]) for row in rows])
    return output


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    sequences = _parent_su15_sequences(root)
    parent_snapshot = manifest["superseded_t10_3_6"]
    contract = manifest["functional_contract"]
    checks = {
        "parent_snapshot_exact": all(
            parent_snapshot.get(key) == value for key, value in protocol.SUPERSEDED_T10_3_6.items()
        ),
        "two_su15_sequences_present": len(sequences) == 2,
        "first_nine_waypoints_exact": len(sequences) == 2 and all(
            tuple(row[:9]) == EXPECTED_SU15_WAYPOINT_CHECKSUMS[:9] for row in sequences
        ),
        "tenth_repeated_ninth": len(sequences) == 2 and all(
            len(row) >= 10 and row[9] == EXPECTED_SU15_WAYPOINT_CHECKSUMS[8] for row in sequences
        ),
        "missing_tenth_identified": len(sequences) == 2 and all(
            row[9] != EXPECTED_SU15_WAYPOINT_CHECKSUMS[9] for row in sequences
        ),
        "fresh_plan_ephemeral": contract["fresh_path_held_ephemerally_during_option"],
        "waypoint_reacquisition": contract["each_waypoint_reacquired_from_current_legal_actions"],
        "parent_training_forbidden": parent_snapshot["used_for_training"] is False,
        "latency_not_gate": contract["no_latency_scientific_gate"],
        "source_firewall_closed": not any(manifest["firewall"].values()),
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.7-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "parent_events_used_for_training": 0,
            "physical_actions": 0,
            "status": "PASS_T10_3_7_OFFLINE_AUDIT" if all(checks.values()) else "INVALID_PROVENANCE",
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.7 provenance audit failed")
    return payload


def _synthetic_stable_path() -> dict[str, Any]:
    candidates = tuple(
        ActionCandidate("ACTION6", {"x": index, "y": 20 - index})
        for index in range(1, 11)
    )
    macro = CandidateMacro(
        schema="path_successor",
        relation="successor_toward_enclosure",
        actions=tuple(
            GroundedAction(
                candidate.action_name,
                tuple(dict(candidate.action_data).items()),
            )
            for candidate in candidates
        ),
    )
    old = stable_module.chain_successor_macro
    stable_module.chain_successor_macro = lambda *args, **kwargs: macro
    try:
        controller = StableFreshPathSageTController(phase="preflight")
        controller._active_option = GoalDirectedOption(
            schema="path_successor",
            steps=tuple(
                OptionStep("ACTION6", binding_method=DYNAMIC_SUCCESSOR)
                for _ in range(10)
            ),
            source="synthetic_fresh_path",
        )
        state = AbstractState(entities=())
        selected = []
        for cursor in range(10):
            controller._active_cursor = cursor
            action = controller._continue_active_option(state, tuple(reversed(candidates)))
            selected.append(dict(action.action_data) if action is not None else None)
        summary = controller.summary()
    finally:
        stable_module.chain_successor_macro = old
    return {
        "selected": selected,
        "expected": [dict(candidate.action_data) for candidate in candidates],
        "reacquisitions": int(summary["fresh_plan_reacquisitions"]),
        "grounding_misses": int(summary["fresh_plan_grounding_misses"]),
        "plan_persisted": bool(summary["fresh_successor_plan_persisted"]),
    }


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = _synthetic_stable_path()
    balanced = base._synthetic_binding_cycle(offset=0)
    cyclic = base._synthetic_binding_cycle(offset=0, cycle=True)
    checks = {
        "ten_waypoints_in_order": path["selected"] == path["expected"],
        "ten_waypoints_reacquired": path["reacquisitions"] == 10,
        "zero_path_grounding_miss": path["grounding_misses"] == 0,
        "fresh_path_not_persisted": path["plan_persisted"] is False,
        "lp85_binding_preserved": balanced["option_successes"] == 1,
        "visual_cycle_not_credited": cyclic["option_successes"] == 0,
        "latency_telemetry_only": manifest["functional_contract"]["latency_is_telemetry_only"],
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.7-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": {"stable_path": path, "balanced_binding": balanced, "cycle": cyclic},
            "checks": checks,
            "physical_actions": 0,
            "status": "PASS_T10_3_7_PREFLIGHT" if all(checks.values()) else "FUNCTIONAL_WIRING_MISS",
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.7 stable-path preflight failed")
    return payload


def _delegate(root: Path, manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    with _contracts(), _versioned_writes():
        if phase == "witness-core":
            try:
                base.run_witness_core(root, manifest)
            except protocol.ScientificGateMiss:
                pass
            result = _read_signed(root, WITNESS_REPORT_FILENAME, "report_checksum")
        elif phase in {"discover-core", "reproduce-core"}:
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
        ("preflight", PREFLIGHT_FILENAME, "preflight_checksum"),
        ("witness", WITNESS_REPORT_FILENAME, "report_checksum"),
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
    if artifacts["audit"] is None or artifacts["audit"].get("status") != "PASS_T10_3_7_OFFLINE_AUDIT":
        verdict = "INVALID_PROVENANCE"
    elif artifacts["preflight"] is None or artifacts["preflight"].get("status") != "PASS_T10_3_7_PREFLIGHT":
        verdict = "FUNCTIONAL_WIRING_MISS"
    elif artifacts["witness"] is None or artifacts["witness"].get("passed") is not True:
        verdict = "CANONICAL_WITNESS_MISS"
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
        verdict = "PASS_T10_3_7_FUNCTIONAL_END_TO_END_SOURCE"
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.7-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "artifacts": {
                name: None if value is None else next(
                    (value[key] for key in ("audit_checksum", "preflight_checksum", "report_checksum") if key in value),
                    None,
                )
                for name, value in artifacts.items()
            },
            "accounting": accounting,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
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
    payload["protocol"] = "SAGE.T10.3.7"
    payload["functional_contract"] = manifest["functional_contract"]
    payload["latency_is_telemetry_only"] = True
    return payload


def _emit(payload: Mapping[str, Any]) -> None:
    print(_canonical(payload), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze", "status", "audit", "preflight", "witness-core",
            "discover-core", "reproduce-core", "discover-sequence", "compile",
            "confirm", "report",
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
        if args.phase == "preflight":
            _emit(preflight(root, manifest))
            return 0
        if args.phase in {"witness-core", "discover-core", "reproduce-core", "discover-sequence", "compile", "confirm"}:
            _emit(_delegate(root, manifest, args.phase))
            return 0
        report = terminal_report(root, manifest)
        _emit(report)
        return 0 if report["verdict"] == "PASS_T10_3_7_FUNCTIONAL_END_TO_END_SOURCE" else 3
    except protocol.ScientificGateMiss as exc:
        _emit({"phase": args.phase, "error": str(exc), "exit_code": 3})
        return 3
    except (protocol.IntegrityError, OSError, ValueError, KeyError) as exc:
        _emit({"phase": args.phase, "error": f"{type(exc).__name__}:{exc}", "exit_code": 2})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_SU15_WAYPOINT_CHECKSUMS", "audit", "main", "preflight",
    "status", "terminal_report",
]
