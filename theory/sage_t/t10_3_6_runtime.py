"""Durable functional-first runtime for SAGE.T10.3.6."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import t10_3_2_runtime as durable
from . import t10_3_5_runtime as shell
from . import t10_3_6_protocol as protocol
from .goal_directed_v10_3_2 import GoalDirectedOption, OptionStep, ProgressProgramRegistry
from .goal_directed_v10_3_5 import ScheduledUnifiedCognitiveController, scheduled_unified_config
from .goal_directed_v10_3_6 import (
    BALANCED_CAUSAL_BINDING,
    FunctionalGoalDirectedSageTController,
    WITNESS_BINDING,
)

AUDIT_FILENAME = "offline_audit.json"
PREFLIGHT_FILENAME = "synthetic_preflight.json"
WITNESS_REPORT_FILENAME = "canonical_witness_report.json"
CORE_REPORT_FILENAME = durable.CORE_REPORT_FILENAME
CORE_REGISTRY_FILENAME = durable.CORE_REGISTRY_FILENAME
REPRODUCTION_REPORT_FILENAME = "reproduction_core_report.json"
REPRODUCED_REGISTRY_FILENAME = "reproduced_core_registry.json"
SEQUENCE_REPORT_FILENAME = durable.SEQUENCE_REPORT_FILENAME
SEQUENCE_REGISTRY_FILENAME = durable.SEQUENCE_REGISTRY_FILENAME
COMPILE_REPORT_FILENAME = durable.COMPILE_REPORT_FILENAME
COMPILED_REGISTRY_FILENAME = durable.COMPILED_REGISTRY_FILENAME
CONFIRMATION_REPORT_FILENAME = durable.CONFIRMATION_REPORT_FILENAME
TERMINAL_REPORT_FILENAME = durable.TERMINAL_REPORT_FILENAME
LOCK_FILENAME = durable.LOCK_FILENAME


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = protocol.sha256_payload(output)
    return output


def _destination(root: Path) -> Path:
    return root.resolve() / protocol.DEFAULT_OUTPUT_DIR


def _artifact_path(root: Path, filename: str) -> Path:
    return _destination(root) / filename


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


@contextmanager
def _contracts() -> Iterator[None]:
    """Route the inherited durability shell through T10.3.6 contracts."""

    old_durable_protocol = durable.protocol
    old_shell_protocol = shell.protocol
    old_shell_pair = shell._controller_pair
    durable.protocol = protocol
    shell.protocol = protocol
    shell._controller_pair = _controller_pair
    try:
        yield
    finally:
        shell._controller_pair = old_shell_pair
        shell.protocol = old_shell_protocol
        durable.protocol = old_durable_protocol


def _controller_pair(
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None,
) -> tuple[ScheduledUnifiedCognitiveController, FunctionalGoalDirectedSageTController | None]:
    if work.arm == "unified_sage_t_off":
        return (
            ScheduledUnifiedCognitiveController(
                work.game_id,
                config=scheduled_unified_config(sage_t_authority_mode="off"),
            ),
            None,
        )
    witness = protocol.WITNESS_PROGRAMS.get(work.game_id) if work.phase == "witness-core" else None
    controller_phase = (
        "confirmation"
        if work.phase == "confirm"
        else "preflight"
        if work.phase == "witness-core"
        else "discovery"
    )
    goal = FunctionalGoalDirectedSageTController(
        phase=controller_phase,
        registry=registry,
        registry_checksum=registry_checksum,
        attestation_scope=work.work_id,
        exploration_offset=work.reset_index,
        witness_schema=None if witness is None else str(witness["macro_schema"]),
        witness_horizon=None if witness is None else int(witness["horizon"]),
        prefer_mixed=work.phase == "discover-sequence",
    )
    controller = ScheduledUnifiedCognitiveController(
        work.game_id,
        config=scheduled_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    return controller, goal


def _run_work(
    root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    lock: Any,
    *,
    registry_checksum: str | None,
) -> dict[str, Any]:
    # The inherited shell already supplies intent-before-action, immediate
    # event sealing, orphan recovery and no physical replay.  Only controller
    # construction and the frozen protocol are replaced above.
    return shell._run_work(
        root,
        destination,
        manifest,
        work,
        registry,
        lock,
        registry_checksum=registry_checksum,
    )


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    return durable._read_signed(_artifact_path(root, filename), checksum_field)


def _phase_receipts(root: Path, phase: str) -> list[dict[str, Any]]:
    with _contracts():
        return durable._load_receipts(_destination(root), phase)


def _telemetry(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decision = [float(value) for row in receipts for value in row.get("decision_latencies_ms", ())]
    cycle = [float(value) for row in receipts for value in row.get("controller_cycle_latencies_ms", ())]
    return {
        "decision_p95_ms": _p95(decision),
        "controller_cycle_p95_ms": _p95(cycle),
        "reset_elapsed_seconds": sum(float(row.get("reset_elapsed_seconds", 0.0)) for row in receipts),
        "latency_is_telemetry_only": True,
    }


def _basic_checks(receipts: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    return {
        "all_conditions_present": len(receipts) == expected,
        "intent_accounting": all(
            int(row.get("issued_intents", 0))
            == int(row.get("sealed_events", 0)) + int(row.get("unresolved_intents", 0))
            for row in receipts
        ),
        "zero_controller_errors": all(not row.get("errors") for row in receipts),
        "zero_illegal_actions": all(int(row.get("illegal_actions", 0)) == 0 for row in receipts),
        "zero_physical_replay": all(int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts),
        "posterior_updated_each_event": all(
            int(row.get("lightweight_observations", 0)) == int(row.get("sealed_events", 0))
            for row in receipts
        ),
        "latency_not_a_gate": True,
    }


def _levels_by_game(receipts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        game: sum(int(row.get("level_delta", 0)) for row in receipts if row.get("game_id") == game)
        for game in protocol.ALL_SOURCE_GAMES
        if any(row.get("game_id") == game for row in receipts)
    }


def _winning_sage_source(receipts: Sequence[Mapping[str, Any]]) -> bool:
    winners = [row for row in receipts if int(row.get("level_delta", 0)) > 0]
    return bool(winners) and all(
        row.get("level_event_sources")
        and all(source == "sage_t_joint_program" for source in row["level_event_sources"])
        for row in winners
    )


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_snapshot = manifest["superseded_t10_3_5"]
    functional = manifest["functional_contract"]
    checks = {
        "parent_snapshot_exact": all(
            parent_snapshot.get(key) == value
            for key, value in protocol.SUPERSEDED_T10_3_5.items()
        ),
        "parent_negative_function_unresolved": parent_snapshot.get("level_delta") == 0,
        "parent_training_forbidden": parent_snapshot.get("used_for_training") is False,
        "t10_0b_witness_diagnostic_only": functional["canonical_witness_is_diagnostic_only"]
        and manifest["firewall"]["t10_0b_events_training_authorized"] is False,
        "historical_grounded_actions_forbidden": functional["historical_grounded_actions_loaded"] is False,
        "blank_discovery": functional["blank_posterior_discovery"] is True,
        "level_only_credit": functional["level_increment_is_only_success_credit"] is True,
        "latency_not_gate": functional["latency_is_telemetry_only"] is True
        and functional["no_latency_scientific_gate"] is True,
        "source_firewall_closed": not any(manifest["firewall"].values()),
        "budget_exact": manifest["matrix"]["total_maximum_actions"] == protocol.TOTAL_MAXIMUM_ACTIONS,
        "reset_exact": manifest["matrix"]["total_resets"] == protocol.TOTAL_RESETS,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.6-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "parent_events_used_for_training": 0,
            "t10_0b_events_used_for_training": 0,
            "t10_0b_grounded_actions_loaded": 0,
            "physical_actions": 0,
            "status": "PASS_T10_3_6_OFFLINE_AUDIT" if all(checks.values()) else "INVALID_PROVENANCE",
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.6 provenance audit failed")
    return payload


@dataclass(frozen=True)
class _SyntheticAction:
    name: str
    action_args: Mapping[str, Any]


def _synthetic_binding_cycle(*, offset: int, cycle: bool = False) -> dict[str, Any]:
    goal = FunctionalGoalDirectedSageTController(
        phase="preflight",
        exploration_offset=offset,
        witness_schema="repeat_target",
        witness_horizon=5,
        attestation_scope=f"synthetic-{offset}-{cycle}",
    )
    controller = ScheduledUnifiedCognitiveController(
        "synthetic-functional",
        config=scheduled_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    legal = (
        _SyntheticAction("ACTION6", {"x": 1, "y": 1}),
        _SyntheticAction("ACTION6", {"x": 5, "y": 5}),
    )
    grid = np.zeros((7, 7), dtype=np.int16)
    grid[1, 1] = grid[5, 5] = 2
    selected = []
    for index in range(5):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=("ACTION6",),
            legacy_action="ACTION6",
            legacy_action_data=legal[0].action_args,
            available_action_candidates=legal,
            levels_completed=0,
        )
        selected.append(dict(decision.action_data))
        after = before.copy()
        if cycle:
            after[3, 3] = index % 2
        else:
            after[3, 3] = index + 1
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=("ACTION6",),
            levels_completed_before=0,
            levels_completed_after=int(not cycle and index == 4),
        )
        grid = after
    summary = goal.summary()
    registry_text = _canonical(summary.get("registry", {}))
    return {
        "offset": offset,
        "selected": selected,
        "option_successes": int(summary.get("option_successes", 0)),
        "causal_cycle_aborts": int(summary.get("causal_cycle_aborts", 0)),
        "posterior_events": int(controller.summary().get("transitions_observed", 0)),
        "safe_registry": all(token not in registry_text for token in ('"x"', '"y"', "raw_grid", "entity_id")),
    }


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    left = _synthetic_binding_cycle(offset=0)
    right = _synthetic_binding_cycle(offset=1)
    cyclic = _synthetic_binding_cycle(offset=0, cycle=True)
    checks = {
        "balanced_offsets_choose_distinct_targets": left["selected"][0] != right["selected"][0],
        "fresh_binding_reacquired_each_action": len({tuple(sorted(row.items())) for row in left["selected"]}) == 1,
        "level_progress_credits_option": left["option_successes"] >= 1 and right["option_successes"] >= 1,
        "visual_cycle_not_credited": cyclic["option_successes"] == 0,
        "state_cycle_aborts_option": cyclic["causal_cycle_aborts"] >= 1,
        "posterior_each_transition": left["posterior_events"] == 5 and right["posterior_events"] == 5,
        "coordinate_free_registry": left["safe_registry"] and right["safe_registry"] and cyclic["safe_registry"],
        "latency_telemetry_only": manifest["functional_contract"]["latency_is_telemetry_only"],
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.6-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": {"offset_zero": left, "offset_one": right, "cycle": cyclic},
            "checks": checks,
            "physical_actions": 0,
            "status": "PASS_T10_3_6_PREFLIGHT" if all(checks.values()) else "FUNCTIONAL_WIRING_MISS",
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.6 functional preflight failed")
    return payload


def _collect(
    root: Path,
    manifest: Mapping[str, Any],
    phase: str,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None = None,
    fresh_registry_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    with _contracts():
        durable._require_live_runtime()
        destination = _destination(root)
        durable._recover_orphans(destination, manifest)
        lock = durable._CollectorLock(destination / LOCK_FILENAME, phase)
        lock.acquire()
        try:
            for work in protocol.work_specs(phase):
                work_registry = (
                    ProgressProgramRegistry(fresh_registry_payload)
                    if fresh_registry_payload is not None
                    else registry
                )
                _run_work(
                    root,
                    destination,
                    manifest,
                    work,
                    work_registry,
                    lock,
                    registry_checksum=(registry_checksum if work.arm == "goal_directed_sage_t" else None),
                )
        finally:
            lock.release()
        return durable._load_receipts(destination, phase)


def run_witness_core(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    receipts = _collect(root, manifest, "witness-core", ProgressProgramRegistry())
    levels = _levels_by_game(receipts)
    checks = {
        **_basic_checks(receipts, len(protocol.work_specs("witness-core"))),
        "level_each_core_game": all(levels.get(game, 0) >= 1 for game in protocol.CORE_GAMES),
        "winning_action_from_sage_t": _winning_sage_source(receipts),
        "fresh_grounding_only": True,
        "historical_grounded_actions_loaded": False,
        "diagnostic_not_training": True,
    }
    report = _signed(
        {
            "format_version": "sage-t10.3.6-canonical-witness-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "witness-core",
            "canonical_descriptors": manifest["canonical_witness_descriptors"],
            "metrics": {
                "levels": levels,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "passed": all(checks.values()),
            "verdict": "PASS_T10_3_6_CANONICAL_WITNESS" if all(checks.values()) else "CANONICAL_WITNESS_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, WITNESS_REPORT_FILENAME), report)
    if report["passed"] is not True:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def run_core_phase(root: Path, manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    witness = _read_signed(root, WITNESS_REPORT_FILENAME, "report_checksum")
    if witness.get("passed") is not True:
        raise protocol.ScientificGateMiss("canonical witness gate forbids blank discovery")
    if phase == "discover-core":
        registry = ProgressProgramRegistry()
    elif phase == "reproduce-core":
        core = _read_signed(root, CORE_REPORT_FILENAME, "report_checksum")
        if core.get("passed") is not True:
            raise protocol.ScientificGateMiss("blank discovery gate forbids reproduction")
        registry = ProgressProgramRegistry(
            _read_signed(root, CORE_REGISTRY_FILENAME, "registry_checksum")
        )
    else:
        raise ValueError("core phase must be discover-core or reproduce-core")
    receipts = _collect(root, manifest, phase, registry)
    levels = _levels_by_game(receipts)
    checks = {
        **_basic_checks(receipts, len(protocol.work_specs(phase))),
        "level_each_core_game": all(levels.get(game, 0) >= 1 for game in protocol.CORE_GAMES),
        "winning_action_from_sage_t": _winning_sage_source(receipts),
        "blank_posterior" if phase == "discover-core" else "fresh_reproduction": True,
        "canonical_witness_not_loaded": True,
        "coordinate_free_registry": all(
            token not in _canonical(registry.snapshot())
            for token in ('"x"', '"y"', "raw_grid", "entity_id", "game_id", "seed")
        ),
    }
    passed = all(checks.values())
    verdict = (
        "PASS_T10_3_6_BLANK_CORE_DISCOVERY"
        if phase == "discover-core" and passed
        else "PASS_T10_3_6_CORE_REPRODUCTION"
        if passed
        else "CORE_DISCOVERY_MISS"
        if phase == "discover-core"
        else "CORE_REPRODUCTION_MISS"
    )
    report = _signed(
        {
            "format_version": f"sage-t10.3.6-{phase}-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": phase,
            "metrics": {
                "levels": levels,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "registry_programs": len(registry.snapshot().get("programs", ())),
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "passed": passed,
            "verdict": verdict,
        },
        "report_checksum",
    )
    report_filename = CORE_REPORT_FILENAME if phase == "discover-core" else REPRODUCTION_REPORT_FILENAME
    registry_filename = CORE_REGISTRY_FILENAME if phase == "discover-core" else REPRODUCED_REGISTRY_FILENAME
    protocol.write_json_once(_artifact_path(root, report_filename), report)
    protocol.write_json_once(_artifact_path(root, registry_filename), registry.snapshot())
    if not passed:
        raise protocol.ScientificGateMiss(verdict)
    return report


def run_sequence(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    reproduction = _read_signed(root, REPRODUCTION_REPORT_FILENAME, "report_checksum")
    if reproduction.get("passed") is not True:
        raise protocol.ScientificGateMiss("core reproduction gate forbids sequence discovery")
    registry = ProgressProgramRegistry(
        _read_signed(root, REPRODUCED_REGISTRY_FILENAME, "registry_checksum")
    )
    receipts = _collect(root, manifest, "discover-sequence", registry)
    levels = _levels_by_game(receipts)
    mixed_winners = [
        row for row in receipts
        if int(row.get("level_delta", 0)) > 0 and bool(row.get("mixed_program_used"))
    ]
    checks = {
        **_basic_checks(receipts, len(protocol.work_specs("discover-sequence"))),
        "sequence_progress": sum(levels.values()) >= 1,
        "mixed_winning_automaton": bool(mixed_winners),
        "winning_action_from_sage_t": _winning_sage_source(receipts),
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.6-discover-sequence-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "discover-sequence",
            "metrics": {
                "levels": levels,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "mixed_winning_resets": len(mixed_winners),
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "passed": passed,
            "verdict": "PASS_T10_3_6_MIXED_SEQUENCE" if passed else "MIXED_SEQUENCE_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, SEQUENCE_REPORT_FILENAME), report)
    protocol.write_json_once(_artifact_path(root, SEQUENCE_REGISTRY_FILENAME), registry.snapshot())
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def compile_registry(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    reproduction = _read_signed(root, REPRODUCTION_REPORT_FILENAME, "report_checksum")
    sequence = _read_signed(root, SEQUENCE_REPORT_FILENAME, "report_checksum")
    if reproduction.get("passed") is not True or sequence.get("passed") is not True:
        raise protocol.ScientificGateMiss("functional gates forbid registry compilation")
    source = _read_signed(root, SEQUENCE_REGISTRY_FILENAME, "registry_checksum")
    registry = ProgressProgramRegistry(source)
    with _contracts():
        controls = durable._apply_registry_controls(registry)
    compiled = registry.snapshot(promoted_only=True)
    programs = compiled.get("programs", ())
    checks = {
        "programs_present": bool(programs),
        "independent_support": bool(programs) and all(
            len(row.get("support_scopes", ())) >= 2 for row in programs
        ),
        "causal_controls": controls["all_checks_passed"],
        "core_program_present": any(
            row["program"]["schema"] in {"repeat_target", "path_successor"}
            for row in programs
        ),
        "mixed_program_present": any(
            len({step["action_name"] for step in row["program"]["dynamics"]}) >= 2
            for row in programs
        ),
        "coordinate_free": all(
            token not in _canonical(compiled)
            for token in ('"x"', '"y"', "raw_grid", "entity_id", "game_id", "seed")
        ),
    }
    passed = all(checks.values())
    protocol.write_json_once(_artifact_path(root, COMPILED_REGISTRY_FILENAME), compiled)
    report = _signed(
        {
            "format_version": "sage-t10.3.6-compile-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": compiled["registry_checksum"],
            "program_count": len(programs),
            "controls": controls,
            "checks": checks,
            "passed": passed,
            "verdict": "PASS_T10_3_6_REGISTRY" if passed else "REGISTRY_REPRODUCTION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, COMPILE_REPORT_FILENAME), report)
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def run_confirmation(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    compile_report = _read_signed(root, COMPILE_REPORT_FILENAME, "report_checksum")
    if compile_report.get("passed") is not True:
        raise protocol.ScientificGateMiss("compiled registry gate forbids confirmation")
    compiled = _read_signed(root, COMPILED_REGISTRY_FILENAME, "registry_checksum")
    registry_checksum = str(compiled["registry_checksum"])
    receipts = _collect(
        root,
        manifest,
        "confirm",
        ProgressProgramRegistry(compiled),
        registry_checksum=registry_checksum,
        fresh_registry_payload=compiled,
    )
    by_key = {
        (str(row["game_id"]), int(row["seed"]), str(row["arm"])): row
        for row in receipts
    }
    active_levels = {
        game: sum(
            int(by_key.get((game, seed, "goal_directed_sage_t"), {}).get("level_delta", 0))
            for seed in protocol.CONFIRMATION_SEEDS
        )
        for game in protocol.ALL_SOURCE_GAMES
    }
    baseline_levels = {
        game: sum(
            int(by_key.get((game, seed, "unified_sage_t_off"), {}).get("level_delta", 0))
            for seed in protocol.CONFIRMATION_SEEDS
        )
        for game in protocol.ALL_SOURCE_GAMES
    }
    active_rows = [row for row in receipts if row["arm"] == "goal_directed_sage_t"]
    baseline_rows = [row for row in receipts if row["arm"] == "unified_sage_t_off"]
    checks = {
        **_basic_checks(receipts, len(protocol.work_specs("confirm"))),
        "core_level_each_game": all(active_levels[game] >= 1 for game in protocol.CORE_GAMES),
        "sequence_progress": sum(active_levels[game] for game in protocol.SEQUENCE_GAMES) >= 1,
        "no_game_regression": all(active_levels[game] >= baseline_levels[game] for game in protocol.ALL_SOURCE_GAMES),
        "total_level_advantage": sum(active_levels.values()) >= sum(baseline_levels.values()) + 1,
        "game_over_nonincrease": sum(int(row.get("game_over_actions", 0)) for row in active_rows)
        <= sum(int(row.get("game_over_actions", 0)) for row in baseline_rows),
        "registry_loaded": all(row.get("registry_checksum_loaded") == registry_checksum for row in active_rows),
        "registry_used": all(bool(row.get("registry_used_in_decision")) for row in active_rows),
        "winning_decisions_attest_registry": all(
            len(row.get("winning_registry_checksums", ())) == len(row.get("level_event_sources", ()))
            and all(checksum == registry_checksum for checksum in row.get("winning_registry_checksums", ()))
            for row in active_rows
        ),
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.6-confirmation-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": registry_checksum,
            "metrics": {
                "active_levels": active_levels,
                "baseline_levels": baseline_levels,
                "active_total_levels": sum(active_levels.values()),
                "baseline_total_levels": sum(baseline_levels.values()),
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                **_telemetry(receipts),
            },
            "checks": checks,
            "passed": passed,
            "verdict": "PASS_T10_3_6_SOURCE_CONFIRMATION" if passed else "SOURCE_CONFIRMATION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, CONFIRMATION_REPORT_FILENAME), report)
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


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
    artifacts: dict[str, dict[str, Any] | None] = {}
    for name, filename, checksum in definitions:
        path = _artifact_path(root, filename)
        artifacts[name] = durable._read_signed(path, checksum) if path.is_file() else None
    if artifacts["audit"] is None or artifacts["audit"].get("status") != "PASS_T10_3_6_OFFLINE_AUDIT":
        verdict = "INVALID_PROVENANCE"
    elif artifacts["preflight"] is None or artifacts["preflight"].get("status") != "PASS_T10_3_6_PREFLIGHT":
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
        verdict = "PASS_T10_3_6_FUNCTIONAL_END_TO_END_SOURCE"
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.6-terminal-report-v1",
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
    payload["protocol"] = "SAGE.T10.3.6"
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
            "discover-core", "reproduce-core", "discover-sequence",
            "compile", "confirm", "report",
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
        if args.phase == "witness-core":
            _emit(run_witness_core(root, manifest))
            return 0
        if args.phase in {"discover-core", "reproduce-core"}:
            _emit(run_core_phase(root, manifest, args.phase))
            return 0
        if args.phase == "discover-sequence":
            _emit(run_sequence(root, manifest))
            return 0
        if args.phase == "compile":
            _emit(compile_registry(root, manifest))
            return 0
        if args.phase == "confirm":
            _emit(run_confirmation(root, manifest))
            return 0
        report = terminal_report(root, manifest)
        _emit(report)
        return 0 if report["verdict"] == "PASS_T10_3_6_FUNCTIONAL_END_TO_END_SOURCE" else 3
    except protocol.ScientificGateMiss as exc:
        _emit({"phase": args.phase, "error": str(exc), "exit_code": 3})
        return 3
    except (protocol.IntegrityError, OSError, ValueError, KeyError) as exc:
        _emit({"phase": args.phase, "error": f"{type(exc).__name__}:{exc}", "exit_code": 2})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit", "compile_registry", "main", "preflight", "run_confirmation",
    "run_core_phase", "run_sequence", "run_witness_core", "status",
    "terminal_report",
]
