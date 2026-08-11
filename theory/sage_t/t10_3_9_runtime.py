"""Durable causal-subgoal transfer runtime for SAGE.T10.3.9."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import t10_3_2_runtime as durable
from . import t10_3_5_runtime as shell
from . import t10_3_9_protocol as protocol
from .goal_directed_v10_3_2 import (
    OptionStep,
    ProgressProgramRegistry,
)
from .goal_directed_v10_3_5 import (
    ScheduledUnifiedCognitiveController,
    scheduled_unified_config,
)
from .goal_directed_v10_3_9 import (
    CausalSubgoalAutomatonInducer,
    CausalSubgoalSageTController,
    robust_effect_descriptor,
)

AUDIT_FILENAME = "offline_audit.json"
PREFLIGHT_FILENAME = "synthetic_preflight.json"
SEQUENCE_REPORT_FILENAME = "discovery_sequence_report.json"
SEQUENCE_REGISTRY_FILENAME = "sequence_registry_candidates.json"
REPRODUCTION_REPORT_FILENAME = "reproduction_sequence_report.json"
REPRODUCED_REGISTRY_FILENAME = "reproduced_sequence_registry.json"
COMPILE_REPORT_FILENAME = "compile_report.json"
COMPILED_REGISTRY_FILENAME = "compiled_registry.json"
CONFIRMATION_REPORT_FILENAME = "confirmation_report.json"
TERMINAL_REPORT_FILENAME = "terminal_report.json"
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


def _parent_root(root: Path) -> Path:
    return root.resolve() / "training" / "sage_t" / "t10_3_8_witness_gate_adjudication"


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    return durable._read_signed(_artifact_path(root, filename), checksum_field)


def _read_parent(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    return durable._read_signed(_parent_root(root) / filename, checksum_field)


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


@contextmanager
def _contracts() -> Iterator[None]:
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
) -> tuple[ScheduledUnifiedCognitiveController, CausalSubgoalSageTController | None]:
    if work.arm == "unified_sage_t_off":
        return (
            ScheduledUnifiedCognitiveController(
                work.game_id,
                config=scheduled_unified_config(sage_t_authority_mode="off"),
            ),
            None,
        )
    controller_phase = "confirmation" if work.phase == "confirm" else "discovery"
    goal = CausalSubgoalSageTController(
        phase=controller_phase,
        registry=registry,
        registry_checksum=registry_checksum,
        attestation_scope=work.work_id,
        exploration_seed=work.seed,
        reproduce_mixed_registry=work.phase == "reproduce-sequence",
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
    return shell._run_work(
        root,
        destination,
        manifest,
        work,
        registry,
        lock,
        registry_checksum=registry_checksum,
    )


def _phase_receipts(root: Path, phase: str) -> list[dict[str, Any]]:
    with _contracts():
        return durable._load_receipts(_destination(root), phase)


def _telemetry(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = [
        float(value)
        for row in receipts
        for value in row.get("decision_latencies_ms", ())
    ]
    cycles = [
        float(value)
        for row in receipts
        for value in row.get("controller_cycle_latencies_ms", ())
    ]
    return {
        "decision_p95_ms": _p95(decisions),
        "controller_cycle_p95_ms": _p95(cycles),
        "reset_elapsed_seconds": sum(
            float(row.get("reset_elapsed_seconds", 0.0)) for row in receipts
        ),
        "latency_is_telemetry_only": True,
    }


def _basic_checks(
    receipts: Sequence[Mapping[str, Any]], expected: int
) -> dict[str, Any]:
    return {
        "all_conditions_present": len(receipts) == expected,
        "intent_accounting": all(
            int(row.get("issued_intents", 0))
            == int(row.get("sealed_events", 0)) + int(row.get("unresolved_intents", 0))
            for row in receipts
        ),
        "zero_controller_errors": all(not row.get("errors") for row in receipts),
        "zero_illegal_actions": all(
            int(row.get("illegal_actions", 0)) == 0 for row in receipts
        ),
        "zero_physical_replay": all(
            int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts
        ),
        "posterior_updated_each_event": all(
            int(row.get("lightweight_observations", 0))
            == int(row.get("sealed_events", 0))
            for row in receipts
        ),
        "latency_not_a_gate": True,
    }


def _levels_by_game(receipts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        game: sum(
            int(row.get("level_delta", 0))
            for row in receipts
            if row.get("game_id") == game
        )
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


def _mixed_winners(receipts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in receipts
        if int(row.get("level_delta", 0)) > 0 and bool(row.get("mixed_program_used"))
    ]


def _trajectory_fingerprints(
    root: Path, receipts: Sequence[Mapping[str, Any]]
) -> dict[str, list[str]]:
    destination = _destination(root)
    by_game: dict[str, list[str]] = {}
    for row in receipts:
        work_id = str(row["work_id"])
        directory = destination / "journal" / "intents" / work_id
        actions = []
        for path in sorted(directory.glob("*.json")) if directory.exists() else ():
            intent = durable._read_signed(path, "intent_checksum")
            actions.append(str(intent.get("action", {}).get("name", "")))
        fingerprint = protocol.sha256_payload(actions)
        by_game.setdefault(str(row["game_id"]), []).append(fingerprint)
    return {game: sorted(values) for game, values in sorted(by_game.items())}


def _diversified_trajectories(fingerprints: Mapping[str, Sequence[str]]) -> bool:
    return bool(fingerprints) and all(
        len(values) >= 2 and len(set(values)) >= 2 for values in fingerprints.values()
    )


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_sequence = _read_parent(root, SEQUENCE_REPORT_FILENAME, "report_checksum")
    parent_terminal = _read_parent(root, TERMINAL_REPORT_FILENAME, "report_checksum")
    diagnosis = manifest["superseded_t10_3_8"]["diagnosis"]
    contract = manifest["functional_contract"]
    checks = {
        "parent_snapshot_exact": all(
            manifest["superseded_t10_3_8"].get(key) == value
            for key, value in protocol.SUPERSEDED_T10_3_8.items()
        ),
        "parent_terminal_negative": parent_terminal.get("verdict") == "MIXED_SEQUENCE_MISS",
        "parent_sequence_zero_levels": sum(
            int(value) for value in parent_sequence.get("metrics", {}).get("levels", {}).values()
        )
        == 0,
        "parent_sequence_actions_exact": diagnosis["sequence_action_count"] == 272,
        "re86_observation_errors_identified": diagnosis["re86_controller_error_count"] == 2,
        "ls20_repeated_budget_exhaustion_identified": diagnosis[
            "ls20_budget_exhaustion_count"
        ]
        == 2,
        "sc25_repeated_terminal_identified": diagnosis["sc25_terminal_count"] == 2,
        "parent_events_diagnostic_only": contract["parent_sequence_events_diagnostic_only"]
        and manifest["firewall"]["t10_3_8_sequence_events_training_authorized"] is False,
        "parent_sequence_registry_forbidden": contract["parent_sequence_registry_loaded"] is False
        and manifest["firewall"]["t10_3_8_sequence_registry_prior_authorized"] is False,
        "core_prior_support_zero": contract["core_registry_prior_support_zero"]
        and manifest["firewall"]["t10_3_8_core_registry_structural_prior_authorized"],
        "level_only_credit": contract["level_increment_is_only_success_credit"],
        "effect_novelty_exploration_only": contract["effect_novelty_is_exploration_only"],
        "source_firewalls_closed": not any(
            value
            for key, value in manifest["firewall"].items()
            if key not in {"t10_3_8_core_registry_structural_prior_authorized"}
        ),
        "budget_exact": manifest["matrix"]["total_maximum_actions"]
        == protocol.TOTAL_MAXIMUM_ACTIONS,
        "reset_exact": manifest["matrix"]["total_resets"] == protocol.TOTAL_RESETS,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.9-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_terminal_report_checksum": parent_terminal["report_checksum"],
            "parent_sequence_report_checksum": parent_sequence["report_checksum"],
            "diagnosis": diagnosis,
            "checks": checks,
            "parent_sequence_events_used_for_training": 0,
            "parent_sequence_registry_loaded": False,
            "parent_core_registry_local_support": 0,
            "physical_actions": 0,
            "status": "PASS_T10_3_9_OFFLINE_AUDIT" if all(checks.values()) else "INVALID_PROVENANCE",
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.9 provenance audit failed")
    return payload


@dataclass(frozen=True)
class _SyntheticAction:
    name: str
    action_args: Mapping[str, Any]


def _synthetic_record(
    action: str,
    index: int,
    *,
    progressed: bool = False,
    terminal: bool = False,
    displacement: Any = (0, 1),
) -> Any:
    return SimpleNamespace(
        action=SimpleNamespace(name=action),
        diff=SimpleNamespace(
            level_complete=progressed,
            game_over=terminal,
            is_noop=False,
            moved_objects=(index,),
            created_objects=(),
            removed_objects=(),
            num_changed=4,
            player_displacement=displacement,
        ),
        obs_before=SimpleNamespace(levels_completed=0),
        obs_after=SimpleNamespace(levels_completed=int(progressed)),
    )


def _synthetic_controller(seed: int) -> dict[str, Any]:
    goal = CausalSubgoalSageTController(
        phase="preflight",
        exploration_seed=seed,
        attestation_scope=f"synthetic-{seed}",
    )
    controller = ScheduledUnifiedCognitiveController(
        "synthetic-causal-subgoal",
        config=scheduled_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    legal = tuple(_SyntheticAction(f"ACTION{index}", {}) for index in range(1, 4))
    names = tuple(item.name for item in legal)
    grid = np.zeros((9, 9), dtype=np.int16)
    decisions = []
    for index in range(20):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=names,
            legacy_action=names[index % len(names)],
            legacy_action_data={},
            available_action_candidates=legal,
            levels_completed=0,
        )
        decisions.append(str(decision.action_name))
        grid = before.copy()
        grid[2 + (index % 3), 2 + ((index // 3) % 3)] = index + 1
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=grid,
            available_actions=names,
            levels_completed_before=0,
            levels_completed_after=int(index == 19),
        )
    summary = goal.summary()
    registry = summary.get("registry", {})
    serialized = _canonical(registry)
    return {
        "decision_fingerprint": protocol.sha256_payload(decisions),
        "distinct_actions": len(set(decisions)),
        "option_successes": int(summary.get("option_successes", 0)),
        "posterior_events": int(controller.summary().get("transitions_observed", 0)),
        "causal_frontier_trials": int(summary.get("causal_frontier_trials", 0)),
        "effect_graph_edges": int(summary.get("causal_inducer", {}).get("effect_graph_edges", 0)),
        "observation_rejections": int(
            summary.get("causal_inducer", {}).get("observation_rejections", 0)
        ),
        "coordinate_free": all(
            token not in serialized
            for token in ('"x"', '"y"', "raw_grid", "entity_id", "game_id", '"seed"', '"color"')
        ),
    }


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    extended = robust_effect_descriptor(
        _synthetic_record("ACTION1", 0, displacement=("actor", 1, -2))
    )
    short = robust_effect_descriptor(
        _synthetic_record("ACTION1", 0, displacement=(1,))
    )
    left = _synthetic_controller(3361)
    right = _synthetic_controller(3362)
    inducer = CausalSubgoalAutomatonInducer()
    inducer.start_branch()
    for index, action in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION1")):
        inducer.observe(
            _synthetic_record(action, index),
            selected_step=OptionStep(action),
            active_option=None,
        )
    mixed = inducer.compose_frontier(("ACTION1", "ACTION2", "ACTION3"), rotation=1)
    mixed_payload = {} if mixed is None else mixed.safe_payload
    serialized = _canonical(mixed_payload)
    checks = {
        "extended_displacement_total": extended["actor_axis"] in {"horizontal", "vertical", "none", "unknown"},
        "short_displacement_total": short["actor_axis"] == "unknown",
        "seed_diversifies_decisions": left["decision_fingerprint"] != right["decision_fingerprint"],
        "all_schemas_explored": left["distinct_actions"] >= 2 and right["distinct_actions"] >= 2,
        "posterior_each_transition": left["posterior_events"] == 20
        and right["posterior_events"] == 20,
        "level_only_success_credit": left["option_successes"] >= 1
        and right["option_successes"] >= 1,
        "causal_graph_built": left["effect_graph_edges"] >= 2
        and right["effect_graph_edges"] >= 2,
        "mixed_frontier_built": mixed is not None and mixed.mixed
        and len(mixed.steps) >= 16,
        "no_observation_rejection": left["observation_rejections"] == 0
        and right["observation_rejections"] == 0,
        "coordinate_free": left["coordinate_free"]
        and right["coordinate_free"]
        and all(
            token not in serialized
            for token in ('"x"', '"y"', "raw_grid", "entity_id", "game_id", '"seed"', '"color"')
        ),
        "latency_telemetry_only": manifest["functional_contract"]["latency_is_telemetry_only"],
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.9-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": {
                "seed_3361": left,
                "seed_3362": right,
                "extended_displacement_axis": extended["actor_axis"],
                "short_displacement_axis": short["actor_axis"],
                "mixed_action_schema_count": 0 if mixed is None else len(set(mixed.action_schemas)),
            },
            "checks": checks,
            "physical_actions": 0,
            "status": "PASS_T10_3_9_PREFLIGHT" if all(checks.values()) else "CAUSAL_SUBGOAL_WIRING_MISS",
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.9 causal-subgoal preflight failed")
    return payload


def _require_offline_gates(root: Path) -> None:
    audit_payload = _read_signed(root, AUDIT_FILENAME, "audit_checksum")
    preflight_payload = _read_signed(root, PREFLIGHT_FILENAME, "preflight_checksum")
    if audit_payload.get("status") != "PASS_T10_3_9_OFFLINE_AUDIT":
        raise protocol.ScientificGateMiss("offline audit forbids physical collection")
    if preflight_payload.get("status") != "PASS_T10_3_9_PREFLIGHT":
        raise protocol.ScientificGateMiss("synthetic preflight forbids physical collection")


def _collect(
    root: Path,
    manifest: Mapping[str, Any],
    phase: str,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None = None,
    fresh_registry_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    _require_offline_gates(root)
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
                    registry_checksum=(
                        registry_checksum if work.arm == "goal_directed_sage_t" else None
                    ),
                )
        finally:
            lock.release()
        return durable._load_receipts(destination, phase)


def discover_sequence(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_core = _read_parent(root, "reproduced_core_registry.json", "registry_checksum")
    registry = ProgressProgramRegistry(parent_core)
    if any(registry.local_support(option.option_id) for option in registry.candidates()):
        raise protocol.IntegrityError("parent core registry imported nonzero local support")
    receipts = _collect(root, manifest, "discover-sequence", registry)
    levels = _levels_by_game(receipts)
    mixed_winners = _mixed_winners(receipts)
    fingerprints = _trajectory_fingerprints(root, receipts)
    checks = {
        **_basic_checks(receipts, len(protocol.work_specs("discover-sequence"))),
        "sequence_progress": sum(levels.values()) >= 1,
        "mixed_winning_automaton": bool(mixed_winners),
        "winning_action_from_sage_t": _winning_sage_source(receipts),
        "seed_diversified_trajectories": _diversified_trajectories(fingerprints),
        "parent_core_registry_local_support_zero": True,
        "parent_sequence_registry_not_loaded": True,
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.9-discover-sequence-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "discover-sequence",
            "metrics": {
                "levels": levels,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "mixed_winning_resets": len(mixed_winners),
                "registry_programs": len(registry.snapshot().get("programs", ())),
                "trajectory_fingerprints": fingerprints,
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "passed": passed,
            "verdict": "PASS_T10_3_9_CAUSAL_SEQUENCE_DISCOVERY" if passed else "CAUSAL_SEQUENCE_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, SEQUENCE_REPORT_FILENAME), report)
    protocol.write_json_once(_artifact_path(root, SEQUENCE_REGISTRY_FILENAME), registry.snapshot())
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def reproduce_sequence(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    discovery = _read_signed(root, SEQUENCE_REPORT_FILENAME, "report_checksum")
    if discovery.get("passed") is not True:
        raise protocol.ScientificGateMiss("sequence discovery gate forbids reproduction")
    source = _read_signed(root, SEQUENCE_REGISTRY_FILENAME, "registry_checksum")
    registry = ProgressProgramRegistry(source)
    receipts = _collect(root, manifest, "reproduce-sequence", registry)
    levels = _levels_by_game(receipts)
    discovery_positive = {
        game
        for game, value in discovery.get("metrics", {}).get("levels", {}).items()
        if int(value) > 0
    }
    reproduced_positive = {game for game, value in levels.items() if int(value) > 0}
    mixed_winners = _mixed_winners(receipts)
    fingerprints = _trajectory_fingerprints(root, receipts)
    checks = {
        **_basic_checks(receipts, len(protocol.work_specs("reproduce-sequence"))),
        "fresh_sequence_progress": sum(levels.values()) >= 1,
        "same_game_reproduced": bool(discovery_positive & reproduced_positive),
        "mixed_winning_automaton": bool(mixed_winners),
        "winning_action_from_sage_t": _winning_sage_source(receipts),
        "seed_diversified_trajectories": _diversified_trajectories(fingerprints),
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.9-reproduce-sequence-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "reproduce-sequence",
            "metrics": {
                "levels": levels,
                "discovery_positive_games": sorted(discovery_positive),
                "reproduced_positive_games": sorted(reproduced_positive),
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "mixed_winning_resets": len(mixed_winners),
                "registry_programs": len(registry.snapshot().get("programs", ())),
                "trajectory_fingerprints": fingerprints,
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "passed": passed,
            "verdict": "PASS_T10_3_9_CAUSAL_SEQUENCE_REPRODUCTION" if passed else "CAUSAL_SEQUENCE_REPRODUCTION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, REPRODUCTION_REPORT_FILENAME), report)
    protocol.write_json_once(_artifact_path(root, REPRODUCED_REGISTRY_FILENAME), registry.snapshot())
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def compile_registry(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    reproduction = _read_signed(root, REPRODUCTION_REPORT_FILENAME, "report_checksum")
    if reproduction.get("passed") is not True:
        raise protocol.ScientificGateMiss("sequence reproduction gate forbids compilation")
    source = _read_signed(root, REPRODUCED_REGISTRY_FILENAME, "registry_checksum")
    registry = ProgressProgramRegistry(source)
    with _contracts():
        controls = durable._apply_registry_controls(registry)
    compiled = registry.snapshot(promoted_only=True)
    programs = list(compiled.get("programs", ()))
    mixed = [
        row
        for row in programs
        if len({step["action_name"] for step in row["program"]["dynamics"]}) >= 2
    ]
    serialized = _canonical(compiled)
    checks = {
        "programs_present": bool(programs),
        "mixed_program_present": bool(mixed),
        "independent_support": bool(mixed)
        and all(len(row.get("support_scopes", ())) >= 2 for row in mixed),
        "causal_controls": controls["all_checks_passed"],
        "coordinate_free": all(
            token not in serialized
            for token in ('"x"', '"y"', "raw_grid", "entity_id", "game_id", '"seed"', '"color"')
        ),
    }
    passed = all(checks.values())
    protocol.write_json_once(_artifact_path(root, COMPILED_REGISTRY_FILENAME), compiled)
    report = _signed(
        {
            "format_version": "sage-t10.3.9-compile-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": compiled["registry_checksum"],
            "program_count": len(programs),
            "mixed_program_count": len(mixed),
            "controls": controls,
            "checks": checks,
            "passed": passed,
            "verdict": "PASS_T10_3_9_CAUSAL_REGISTRY" if passed else "CAUSAL_REGISTRY_REPRODUCTION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, COMPILE_REPORT_FILENAME), report)
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def confirm(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
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
            int(
                by_key.get((game, seed, "goal_directed_sage_t"), {}).get(
                    "level_delta", 0
                )
            )
            for seed in protocol.CONFIRMATION_SEEDS
        )
        for game in protocol.ALL_SOURCE_GAMES
    }
    baseline_levels = {
        game: sum(
            int(
                by_key.get((game, seed, "unified_sage_t_off"), {}).get(
                    "level_delta", 0
                )
            )
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
        "no_game_regression": all(
            active_levels[game] >= baseline_levels[game] for game in protocol.ALL_SOURCE_GAMES
        ),
        "total_level_advantage": sum(active_levels.values())
        >= sum(baseline_levels.values()) + 1,
        "game_over_nonincrease": sum(
            int(row.get("game_over_actions", 0)) for row in active_rows
        )
        <= sum(int(row.get("game_over_actions", 0)) for row in baseline_rows),
        "registry_loaded": all(
            row.get("registry_checksum_loaded") == registry_checksum for row in active_rows
        ),
        "registry_used": all(bool(row.get("registry_used_in_decision")) for row in active_rows),
        "winning_decisions_attest_registry": all(
            len(row.get("winning_registry_checksums", ()))
            == len(row.get("level_event_sources", ()))
            and all(
                checksum == registry_checksum
                for checksum in row.get("winning_registry_checksums", ())
            )
            for row in active_rows
        ),
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.9-confirmation-report-v1",
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
            "verdict": "PASS_T10_3_9_SOURCE_CONFIRMATION" if passed else "SOURCE_CONFIRMATION_MISS",
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
        ("discovery", SEQUENCE_REPORT_FILENAME, "report_checksum"),
        ("reproduction", REPRODUCTION_REPORT_FILENAME, "report_checksum"),
        ("compile", COMPILE_REPORT_FILENAME, "report_checksum"),
        ("confirmation", CONFIRMATION_REPORT_FILENAME, "report_checksum"),
    )
    artifacts: dict[str, dict[str, Any] | None] = {}
    for name, filename, checksum in definitions:
        path = _artifact_path(root, filename)
        artifacts[name] = durable._read_signed(path, checksum) if path.is_file() else None
    if artifacts["audit"] is None or artifacts["audit"].get("status") != "PASS_T10_3_9_OFFLINE_AUDIT":
        verdict = "INVALID_PROVENANCE"
    elif artifacts["preflight"] is None or artifacts["preflight"].get("status") != "PASS_T10_3_9_PREFLIGHT":
        verdict = "CAUSAL_SUBGOAL_WIRING_MISS"
    elif artifacts["discovery"] is None or artifacts["discovery"].get("passed") is not True:
        verdict = "CAUSAL_SEQUENCE_MISS"
    elif artifacts["reproduction"] is None or artifacts["reproduction"].get("passed") is not True:
        verdict = "CAUSAL_SEQUENCE_REPRODUCTION_MISS"
    elif artifacts["compile"] is None or artifacts["compile"].get("passed") is not True:
        verdict = "CAUSAL_REGISTRY_REPRODUCTION_MISS"
    elif artifacts["confirmation"] is None or artifacts["confirmation"].get("passed") is not True:
        verdict = "SOURCE_CONFIRMATION_MISS"
    else:
        verdict = "PASS_T10_3_9_CAUSAL_SUBGOAL_SOURCE"
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.9-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "artifacts": {
                name: None
                if value is None
                else next(
                    (
                        value[key]
                        for key in ("audit_checksum", "preflight_checksum", "report_checksum")
                        if key in value
                    ),
                    None,
                )
                for name, value in artifacts.items()
            },
            "accounting": accounting,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
            "parent_sequence_events_used_for_training": 0,
            "parent_sequence_registry_loaded": False,
            "physical_actions_replayed": 0,
            "latency_is_telemetry_only": True,
            "firewall": manifest["firewall"],
            "production_authority": False,
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, TERMINAL_REPORT_FILENAME), report)
    return report


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    artifact_specs = (
        (AUDIT_FILENAME, "audit_checksum"),
        (PREFLIGHT_FILENAME, "preflight_checksum"),
        (SEQUENCE_REPORT_FILENAME, "report_checksum"),
        (REPRODUCTION_REPORT_FILENAME, "report_checksum"),
        (COMPILE_REPORT_FILENAME, "report_checksum"),
        (CONFIRMATION_REPORT_FILENAME, "report_checksum"),
        (TERMINAL_REPORT_FILENAME, "report_checksum"),
    )
    artifacts = {}
    for filename, checksum in artifact_specs:
        path = _artifact_path(root, filename)
        artifacts[filename] = (
            None if not path.is_file() else durable._read_signed(path, checksum)[checksum]
        )
    live = bool(accounting.get("live_collector_lock"))
    state = (
        "RUNNING"
        if live
        else "INTERRUPTED"
        if accounting.get("incomplete_work_ids")
        else "READY"
    )
    return {
        "phase": "status",
        "protocol": "SAGE.T10.3.9",
        "status": state,
        "manifest_checksum": manifest["manifest_checksum"],
        "accounting": accounting,
        "completed_resets": len(
            tuple((_destination(root) / "journal" / "branches").rglob("receipt.json"))
        )
        if (_destination(root) / "journal" / "branches").exists()
        else 0,
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        "maximum_resets": protocol.TOTAL_RESETS,
        "artifacts": artifacts,
        "firewall": manifest["firewall"],
    }


def _emit(payload: Mapping[str, Any]) -> None:
    print(_canonical(payload), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze", "status", "audit", "preflight", "discover-sequence",
            "reproduce-sequence", "compile", "confirm", "report",
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
        if args.phase == "discover-sequence":
            _emit(discover_sequence(root, manifest))
            return 0
        if args.phase == "reproduce-sequence":
            _emit(reproduce_sequence(root, manifest))
            return 0
        if args.phase == "compile":
            _emit(compile_registry(root, manifest))
            return 0
        if args.phase == "confirm":
            _emit(confirm(root, manifest))
            return 0
        report = terminal_report(root, manifest)
        _emit(report)
        return 0 if report["verdict"] == "PASS_T10_3_9_CAUSAL_SUBGOAL_SOURCE" else 3
    except protocol.ScientificGateMiss as exc:
        _emit({"phase": args.phase, "error": str(exc), "exit_code": 3})
        return 3
    except (protocol.IntegrityError, OSError, ValueError, KeyError) as exc:
        _emit(
            {
                "phase": args.phase,
                "error": f"{type(exc).__name__}:{exc}",
                "exit_code": 2,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit", "compile_registry", "confirm", "discover_sequence", "main",
    "preflight", "reproduce_sequence", "status", "terminal_report",
]
