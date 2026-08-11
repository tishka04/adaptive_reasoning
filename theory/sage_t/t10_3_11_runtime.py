"""Durable bounded-goal runtime for SAGE.T10.3.11."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import t10_3_2_runtime as durable
from . import t10_3_5_runtime as shell
from . import t10_3_11_protocol as protocol
from .goal_directed_v10_3_2 import ProgressProgramRegistry
from .goal_directed_v10_3_11 import (
    CONTROLLER_TRANSITION_LIMIT,
    POSTERIOR_HISTORY_LIMIT,
    GoalConditionedSageTController,
    GoalConditionedUnifiedCognitiveController,
    goal_conditioned_unified_config,
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

_ACTIVE_PAIRS: dict[
    str,
    tuple[
        GoalConditionedUnifiedCognitiveController,
        GoalConditionedSageTController | None,
    ],
] = {}


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


def _parent_core_path(root: Path) -> Path:
    return (
        root.resolve()
        / "training"
        / "sage_t"
        / "t10_3_8_witness_gate_adjudication"
        / "reproduced_core_registry.json"
    )


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    return durable._read_signed(_artifact_path(root, filename), checksum_field)


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
) -> tuple[
    GoalConditionedUnifiedCognitiveController,
    GoalConditionedSageTController | None,
]:
    if work.arm == "unified_sage_t_off":
        pair = (
            GoalConditionedUnifiedCognitiveController(
                work.game_id,
                config=goal_conditioned_unified_config(sage_t_authority_mode="off"),
                goal_conditioning_enabled=False,
            ),
            None,
        )
    else:
        conditioned = work.arm == "goal_conditioned_sage_t"
        controller_phase = "confirmation" if work.phase == "confirm" else "discovery"
        goal = GoalConditionedSageTController(
            phase=controller_phase,
            registry=registry,
            registry_checksum=registry_checksum,
            attestation_scope=work.work_id,
            exploration_seed=work.seed,
            reproduce_mixed_registry=work.phase == "reproduce-sequence",
            goal_conditioning_enabled=conditioned,
        )
        pair = (
            GoalConditionedUnifiedCognitiveController(
                work.game_id,
                config=goal_conditioned_unified_config(sage_t_authority_mode="active"),
                sage_t_controller=goal,
                goal_conditioning_enabled=conditioned,
            ),
            goal,
        )
    _ACTIVE_PAIRS[work.work_id] = pair
    return pair


def _diagnostic_path(destination: Path, work: protocol.WorkSpec) -> Path:
    return destination / "branch_diagnostics" / f"{work.work_id}.json"


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
    receipt = shell._run_work(
        root,
        destination,
        manifest,
        work,
        registry,
        lock,
        registry_checksum=registry_checksum,
    )
    pair = _ACTIVE_PAIRS.pop(work.work_id, None)
    if pair is None:
        return receipt
    controller, goal = pair
    controller_summary = dict(controller.summary())
    goal_summary = {} if goal is None else dict(goal.summary())
    posterior = dict(goal_summary.get("bounded_program_posterior", {}))
    diagnostic = _signed(
        {
            "format_version": "sage-t10.3.11-branch-diagnostic-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "phase": work.phase,
            "arm": work.arm,
            "receipt_checksum": receipt["receipt_checksum"],
            "sealed_events": int(receipt.get("sealed_events", 0)),
            "posterior_observations": int(posterior.get("observations", 0)),
            "posterior_maximum_history": int(posterior.get("maximum_history", 0)),
            "posterior_history_limit": int(posterior.get("history_limit", 0)),
            "posterior_repairs_attempted": int(posterior.get("repairs_attempted", 0)),
            "posterior_repairs_admitted": int(posterior.get("repairs_admitted", 0)),
            "live_repairs_suppressed": int(posterior.get("live_repairs_suppressed", 0)),
            "program_reassemblies": int(goal_summary.get("program_reassemblies", 0)),
            "maximum_controller_transitions": int(
                goal_summary.get("maximum_controller_transitions", 0)
            ),
            "posterior_observation_rejections": int(
                goal_summary.get("posterior_observation_rejections", 0)
            ),
            "observation_errors": dict(goal_summary.get("observation_errors", {})),
            "last_observation_error_digest": str(
                goal_summary.get("last_observation_error_digest", "")
            ),
            "goal_generation_calls": int(controller_summary.get("goal_generation_calls", 0)),
            "maximum_live_objectives": int(
                controller_summary.get("maximum_live_objectives", 0)
            ),
            "objective_observations": int(
                controller_summary.get("objective_observations", 0)
            ),
            "objective_distance_reductions": int(
                controller_summary.get("objective_distance_reductions", 0)
            ),
            "goal_hypotheses_received": int(
                goal_summary.get("goal_hypotheses_received", 0)
            ),
            "goal_conditioned_options": int(
                goal_summary.get("goal_conditioned_options", 0)
            ),
            "goal_conditioned_actions": int(
                goal_summary.get("goal_conditioned_actions", 0)
            ),
            "goal_conditioning_enabled": bool(
                goal_summary.get("goal_conditioning_enabled", False)
            ),
            "raw_frames_retained": False,
            "goal_payloads_persisted": False,
            "objective_ids_persisted": False,
            "ephemeral_action_data_persisted": False,
        },
        "diagnostic_checksum",
    )
    protocol.write_json_once(_diagnostic_path(destination, work), diagnostic)
    return receipt


def _load_diagnostics(root: Path, phase: str) -> list[dict[str, Any]]:
    base = _destination(root) / "branch_diagnostics"
    rows = []
    for path in sorted(base.glob("*.json")) if base.exists() else ():
        row = durable._read_signed(path, "diagnostic_checksum")
        if row.get("phase") == phase:
            rows.append(row)
    return rows


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
    receipts: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    expected: int,
) -> dict[str, Any]:
    diagnostic_by_id = {str(row["work_id"]): row for row in diagnostics}
    sage_rows = [row for row in receipts if row.get("arm") != "unified_sage_t_off"]
    sage_diagnostics = [
        diagnostic_by_id[str(row["work_id"])]
        for row in sage_rows
        if str(row["work_id"]) in diagnostic_by_id
    ]
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
        "sage_diagnostics_present": len(sage_diagnostics) == len(sage_rows),
        "posterior_updated_each_event": all(
            int(row.get("posterior_observations", 0))
            == int(receipt.get("sealed_events", 0))
            for receipt, row in (
                (receipt, diagnostic_by_id.get(str(receipt["work_id"]), {}))
                for receipt in sage_rows
            )
        ),
        "posterior_observations_not_rejected": all(
            int(row.get("posterior_observation_rejections", 0)) == 0
            for row in sage_diagnostics
        ),
        "program_posterior_history_bounded": all(
            int(row.get("posterior_maximum_history", 0)) <= POSTERIOR_HISTORY_LIMIT
            for row in sage_diagnostics
        ),
        "controller_transition_history_bounded": all(
            int(row.get("maximum_controller_transitions", 0))
            <= CONTROLLER_TRANSITION_LIMIT
            for row in sage_diagnostics
        ),
        "no_live_posterior_repair": all(
            int(row.get("posterior_repairs_attempted", 0)) == 0
            and int(row.get("posterior_repairs_admitted", 0)) == 0
            for row in sage_diagnostics
        ),
        "latency_not_a_gate": True,
    }


def _levels_by_arm(
    receipts: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, int]:
    return {
        game: sum(
            int(row.get("level_delta", 0))
            for row in receipts
            if row.get("game_id") == game and row.get("arm") == arm
        )
        for game in protocol.ALL_SOURCE_GAMES
        if any(row.get("game_id") == game for row in receipts)
    }


def _winning_sage_source(receipts: Sequence[Mapping[str, Any]], arm: str) -> bool:
    winners = [
        row
        for row in receipts
        if row.get("arm") == arm and int(row.get("level_delta", 0)) > 0
    ]
    return bool(winners) and all(
        row.get("level_event_sources")
        and all(source == "sage_t_joint_program" for source in row["level_event_sources"])
        for row in winners
    )


def _mixed_winners(
    receipts: Sequence[Mapping[str, Any]], arm: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in receipts
        if row.get("arm") == arm
        and int(row.get("level_delta", 0)) > 0
        and bool(row.get("mixed_program_used"))
    ]


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    diagnosis = manifest["superseded_t10_3_10"]["diagnosis"]
    contract = manifest["functional_contract"]
    checks = {
        "parent_snapshot_exact": all(
            manifest["superseded_t10_3_10"].get(key) == value
            for key, value in protocol.SUPERSEDED_T10_3_10.items()
        ),
        "parent_complete_negative_exact": diagnosis["intent_count"] == 857
        and diagnosis["event_count"] == 857
        and diagnosis["branch_count"] == 12
        and diagnosis["incomplete_work_count"] == 0,
        "parent_zero_levels": diagnosis["sequence_level_count"] == 0,
        "parent_observation_failure_identified": diagnosis[
            "controller_observe_error_count"
        ]
        == 5,
        "parent_growth_identified": diagnosis["controller_cycle_p95_ms"] > 20_000.0,
        "parent_events_diagnostic_only": contract["parent_events_diagnostic_only"]
        and manifest["firewall"]["t10_3_10_events_training_authorized"] is False,
        "parent_registry_forbidden": contract["parent_registry_loaded"] is False
        and manifest["firewall"]["t10_3_10_registry_prior_authorized"] is False,
        "bounded_program_posterior": contract["program_posterior_history_limit"]
        == POSTERIOR_HISTORY_LIMIT
        and contract["live_posterior_repair"] is False,
        "goal_conditioning_active": contract["goal_hypotheses_forwarded_to_sage_t"]
        and contract["paired_goal_ablation"],
        "level_only_credit": contract["level_increment_is_only_success_credit"]
        and contract["goal_distance_is_planning_evidence_only"],
        "source_firewalls_closed": bool(
            manifest["firewall"]["t10_3_8_core_registry_structural_prior_authorized"]
            and not any(
                value
                for key, value in manifest["firewall"].items()
                if key != "t10_3_8_core_registry_structural_prior_authorized"
            )
        ),
        "budget_exact": manifest["matrix"]["total_maximum_actions"]
        == protocol.TOTAL_MAXIMUM_ACTIONS,
        "reset_exact": manifest["matrix"]["total_resets"] == protocol.TOTAL_RESETS,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.11-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "diagnosis": diagnosis,
            "checks": checks,
            "parent_events_used_for_training": 0,
            "parent_registry_loaded": False,
            "parent_physical_actions_replayed": 0,
            "physical_actions": 0,
            "status": "PASS_T10_3_11_OFFLINE_AUDIT" if all(checks.values()) else "INVALID_PROVENANCE",
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.11 provenance audit failed")
    return payload


@dataclass(frozen=True)
class _SyntheticAction:
    name: str
    action_args: Mapping[str, Any]


def _delayed_goal_loop(seed: int) -> dict[str, Any]:
    registry = ProgressProgramRegistry()
    goal = GoalConditionedSageTController(
        phase="preflight",
        registry=registry,
        exploration_seed=seed,
        attestation_scope=f"delayed-goal-{seed}",
    )
    controller = GoalConditionedUnifiedCognitiveController(
        "synthetic-delayed-goal",
        config=goal_conditioned_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    legal = (_SyntheticAction("ACTION2", {}), _SyntheticAction("ACTION3", {}))
    names = tuple(action.name for action in legal)
    grid = np.zeros((9, 9), dtype=np.int16)
    grid[4, 4] = 2
    grid[1, 7] = 3
    actions: list[str] = []
    levels = 0
    for index in range(24):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=names,
            legacy_action=names[(index + seed) % len(names)],
            legacy_action_data={},
            available_action_candidates=legal,
            levels_completed=levels,
        )
        action = str(decision.action_name)
        actions.append(action)
        grid = before.copy()
        grid[0, index % grid.shape[1]] = 4 if action == "ACTION2" else 5
        alternating = len(actions) >= 18 and all(
            actions[offset] != actions[offset - 1]
            for offset in range(1, 18)
        )
        after_levels = int(alternating)
        controller.observe_transition(
            action=action,
            action_data=dict(decision.action_data),
            grid_before=before,
            grid_after=grid,
            available_actions=names,
            levels_completed_before=levels,
            levels_completed_after=after_levels,
        )
        levels = after_levels
        if levels:
            break
    controller_summary = dict(controller.summary())
    goal_summary = dict(goal.summary())
    posterior = dict(goal_summary["bounded_program_posterior"])
    serialized = _canonical(goal_summary.get("registry", {}))
    return {
        "won": levels == 1,
        "actions": len(actions),
        "distinct_actions": len(set(actions)),
        "alternating_prefix": all(
            actions[index] != actions[index - 1] for index in range(1, len(actions))
        ),
        "goal_generation_calls": int(controller_summary["goal_generation_calls"]),
        "maximum_live_objectives": int(controller_summary["maximum_live_objectives"]),
        "goal_conditioned_options": int(goal_summary["goal_conditioned_options"]),
        "goal_conditioned_actions": int(goal_summary["goal_conditioned_actions"]),
        "posterior_observations": int(posterior["observations"]),
        "posterior_maximum_history": int(posterior["maximum_history"]),
        "posterior_repairs_attempted": int(posterior["repairs_attempted"]),
        "program_reassemblies": int(goal_summary["program_reassemblies"]),
        "coordinate_free": all(
            token not in serialized
            for token in ('"x"', '"y"', "raw_grid", "entity_id", "game_id", '"seed"', '"color"')
        ),
    }


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    left = _delayed_goal_loop(3421)
    right = _delayed_goal_loop(3422)
    checks = {
        "delayed_mixed_win_from_blank_posterior": left["won"] and right["won"],
        "success_after_sixteen_actions": left["actions"] >= 18 and right["actions"] >= 18,
        "mixed_automaton_required": left["distinct_actions"] == 2
        and right["distinct_actions"] == 2
        and left["alternating_prefix"]
        and right["alternating_prefix"],
        "goal_generation_live": left["goal_generation_calls"] >= left["actions"]
        and right["goal_generation_calls"] >= right["actions"],
        "goal_hypotheses_reach_sage_t": left["maximum_live_objectives"] > 0
        and right["maximum_live_objectives"] > 0
        and left["goal_conditioned_options"] > 0
        and right["goal_conditioned_options"] > 0,
        "goal_option_controls_physical_actions": left["goal_conditioned_actions"] >= 12
        and right["goal_conditioned_actions"] >= 12,
        "posterior_each_transition": left["posterior_observations"] == left["actions"]
        and right["posterior_observations"] == right["actions"],
        "posterior_history_strictly_bounded": left["posterior_maximum_history"]
        <= POSTERIOR_HISTORY_LIMIT
        and right["posterior_maximum_history"] <= POSTERIOR_HISTORY_LIMIT,
        "no_live_posterior_repair": left["posterior_repairs_attempted"] == 0
        and right["posterior_repairs_attempted"] == 0,
        "reassembly_not_per_event": left["program_reassemblies"] <= 2
        and right["program_reassemblies"] <= 2,
        "coordinate_free": left["coordinate_free"] and right["coordinate_free"],
        "level_only_success_credit": manifest["functional_contract"][
            "level_increment_is_only_success_credit"
        ],
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.11-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": {"seed_3421": left, "seed_3422": right},
            "checks": checks,
            "physical_actions": 0,
            "status": "PASS_T10_3_11_PREFLIGHT" if all(checks.values()) else "GOAL_WIRING_MISS",
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.11 bounded-goal preflight failed")
    return payload


def _require_offline_gates(root: Path) -> None:
    audit_payload = _read_signed(root, AUDIT_FILENAME, "audit_checksum")
    preflight_payload = _read_signed(root, PREFLIGHT_FILENAME, "preflight_checksum")
    if audit_payload.get("status") != "PASS_T10_3_11_OFFLINE_AUDIT":
        raise protocol.ScientificGateMiss("offline audit forbids physical collection")
    if preflight_payload.get("status") != "PASS_T10_3_11_PREFLIGHT":
        raise protocol.ScientificGateMiss("synthetic preflight forbids physical collection")


def _collect(
    root: Path,
    manifest: Mapping[str, Any],
    phase: str,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None = None,
    fresh_registry_payload: Mapping[str, Any] | None = None,
    ablation_registry_payload: Mapping[str, Any] | None = None,
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
                if fresh_registry_payload is not None:
                    work_registry = ProgressProgramRegistry(fresh_registry_payload)
                elif work.arm == "goal_ablation_sage_t" and ablation_registry_payload is not None:
                    work_registry = ProgressProgramRegistry(ablation_registry_payload)
                else:
                    work_registry = registry
                _run_work(
                    root,
                    destination,
                    manifest,
                    work,
                    work_registry,
                    lock,
                    registry_checksum=(
                        registry_checksum
                        if work.arm == "goal_conditioned_sage_t"
                        else None
                    ),
                )
        finally:
            lock.release()
        return durable._load_receipts(destination, phase)


def discover_sequence(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_core = durable._read_signed(_parent_core_path(root), "registry_checksum")
    registry = ProgressProgramRegistry(parent_core)
    if any(registry.local_support(option.option_id) for option in registry.candidates()):
        raise protocol.IntegrityError("parent core registry imported nonzero local support")
    receipts = _collect(
        root,
        manifest,
        "discover-sequence",
        registry,
        ablation_registry_payload=parent_core,
    )
    diagnostics = _load_diagnostics(root, "discover-sequence")
    active_levels = _levels_by_arm(receipts, "goal_conditioned_sage_t")
    ablation_levels = _levels_by_arm(receipts, "goal_ablation_sage_t")
    active_diagnostics = [
        row for row in diagnostics if row.get("arm") == "goal_conditioned_sage_t"
    ]
    ablation_diagnostics = [
        row for row in diagnostics if row.get("arm") == "goal_ablation_sage_t"
    ]
    mixed_winners = _mixed_winners(receipts, "goal_conditioned_sage_t")
    checks = {
        **_basic_checks(receipts, diagnostics, len(protocol.work_specs("discover-sequence"))),
        "sequence_progress": sum(active_levels.values()) >= 1,
        "mixed_winning_automaton": bool(mixed_winners),
        "winning_action_from_sage_t": _winning_sage_source(
            receipts, "goal_conditioned_sage_t"
        ),
        "goals_generated_and_consumed": bool(active_diagnostics)
        and all(int(row.get("maximum_live_objectives", 0)) > 0 for row in active_diagnostics)
        and sum(int(row.get("goal_conditioned_options", 0)) for row in active_diagnostics) > 0
        and sum(int(row.get("goal_conditioned_actions", 0)) for row in active_diagnostics) > 0,
        "goal_ablation_clean": bool(ablation_diagnostics)
        and all(int(row.get("goal_conditioned_actions", 0)) == 0 for row in ablation_diagnostics),
        "no_game_below_ablation": all(
            active_levels.get(game, 0) >= ablation_levels.get(game, 0)
            for game in protocol.SEQUENCE_GAMES
        ),
        "goal_conditioning_total_advantage": sum(active_levels.values())
        >= sum(ablation_levels.values()) + 1,
        "parent_core_registry_local_support_zero": True,
        "t10_3_10_registry_not_loaded": True,
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.11-discover-sequence-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "discover-sequence",
            "metrics": {
                "goal_conditioned_levels": active_levels,
                "goal_ablation_levels": ablation_levels,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "mixed_winning_resets": len(mixed_winners),
                "registry_programs": len(registry.snapshot().get("programs", ())),
                "goal_conditioned_actions": sum(
                    int(row.get("goal_conditioned_actions", 0))
                    for row in active_diagnostics
                ),
                "objective_distance_reductions": sum(
                    int(row.get("objective_distance_reductions", 0))
                    for row in active_diagnostics
                ),
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "diagnostic_checksums": [row["diagnostic_checksum"] for row in diagnostics],
            "passed": passed,
            "verdict": "PASS_T10_3_11_GOAL_SEQUENCE_DISCOVERY" if passed else "GOAL_SEQUENCE_MISS",
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
        raise protocol.ScientificGateMiss("sequence discovery forbids reproduction")
    source = _read_signed(root, SEQUENCE_REGISTRY_FILENAME, "registry_checksum")
    registry = ProgressProgramRegistry(source)
    receipts = _collect(root, manifest, "reproduce-sequence", registry)
    diagnostics = _load_diagnostics(root, "reproduce-sequence")
    levels = _levels_by_arm(receipts, "goal_conditioned_sage_t")
    discovery_positive = {
        game
        for game, value in discovery["metrics"]["goal_conditioned_levels"].items()
        if int(value) > 0
    }
    reproduced_positive = {game for game, value in levels.items() if int(value) > 0}
    checks = {
        **_basic_checks(receipts, diagnostics, len(protocol.work_specs("reproduce-sequence"))),
        "fresh_sequence_progress": sum(levels.values()) >= 1,
        "same_game_reproduced": bool(discovery_positive & reproduced_positive),
        "mixed_winning_automaton": bool(
            _mixed_winners(receipts, "goal_conditioned_sage_t")
        ),
        "winning_action_from_sage_t": _winning_sage_source(
            receipts, "goal_conditioned_sage_t"
        ),
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.11-reproduce-sequence-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "reproduce-sequence",
            "metrics": {
                "levels": levels,
                "discovery_positive_games": sorted(discovery_positive),
                "reproduced_positive_games": sorted(reproduced_positive),
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "diagnostic_checksums": [row["diagnostic_checksum"] for row in diagnostics],
            "passed": passed,
            "verdict": "PASS_T10_3_11_GOAL_SEQUENCE_REPRODUCTION" if passed else "GOAL_SEQUENCE_REPRODUCTION_MISS",
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
        raise protocol.ScientificGateMiss("sequence reproduction forbids compilation")
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
            "format_version": "sage-t10.3.11-compile-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": compiled["registry_checksum"],
            "program_count": len(programs),
            "mixed_program_count": len(mixed),
            "controls": controls,
            "checks": checks,
            "passed": passed,
            "verdict": "PASS_T10_3_11_GOAL_REGISTRY" if passed else "GOAL_REGISTRY_REPRODUCTION_MISS",
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
        raise protocol.ScientificGateMiss("compiled registry forbids confirmation")
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
    diagnostics = _load_diagnostics(root, "confirm")
    active_levels = _levels_by_arm(receipts, "goal_conditioned_sage_t")
    baseline_levels = _levels_by_arm(receipts, "unified_sage_t_off")
    active_rows = [row for row in receipts if row.get("arm") == "goal_conditioned_sage_t"]
    baseline_rows = [row for row in receipts if row.get("arm") == "unified_sage_t_off"]
    checks = {
        **_basic_checks(receipts, diagnostics, len(protocol.work_specs("confirm"))),
        "core_level_each_game": all(active_levels.get(game, 0) >= 1 for game in protocol.CORE_GAMES),
        "sequence_progress": sum(active_levels.get(game, 0) for game in protocol.SEQUENCE_GAMES) >= 1,
        "no_game_regression": all(
            active_levels.get(game, 0) >= baseline_levels.get(game, 0)
            for game in protocol.ALL_SOURCE_GAMES
        ),
        "total_level_advantage": sum(active_levels.values())
        >= sum(baseline_levels.values()) + 1,
        "game_over_nonincrease": sum(int(row.get("game_over_actions", 0)) for row in active_rows)
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
            "format_version": "sage-t10.3.11-confirmation-report-v1",
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
            "verdict": "PASS_T10_3_11_SOURCE_CONFIRMATION" if passed else "SOURCE_CONFIRMATION_MISS",
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
    if artifacts["audit"] is None or artifacts["audit"].get("status") != "PASS_T10_3_11_OFFLINE_AUDIT":
        verdict = "INVALID_PROVENANCE"
    elif artifacts["preflight"] is None or artifacts["preflight"].get("status") != "PASS_T10_3_11_PREFLIGHT":
        verdict = "GOAL_WIRING_MISS"
    elif artifacts["discovery"] is None or artifacts["discovery"].get("passed") is not True:
        verdict = "GOAL_SEQUENCE_MISS"
    elif artifacts["reproduction"] is None or artifacts["reproduction"].get("passed") is not True:
        verdict = "GOAL_SEQUENCE_REPRODUCTION_MISS"
    elif artifacts["compile"] is None or artifacts["compile"].get("passed") is not True:
        verdict = "GOAL_REGISTRY_REPRODUCTION_MISS"
    elif artifacts["confirmation"] is None or artifacts["confirmation"].get("passed") is not True:
        verdict = "SOURCE_CONFIRMATION_MISS"
    else:
        verdict = "PASS_T10_3_11_BOUNDED_GOAL_SOURCE"
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.11-terminal-report-v1",
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
            "parent_events_used_for_training": 0,
            "parent_registry_loaded": False,
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
    state = (
        "RUNNING"
        if accounting.get("live_collector_lock")
        else "INTERRUPTED"
        if accounting.get("incomplete_work_ids")
        else "READY"
    )
    branch_root = _destination(root) / "journal" / "branches"
    return {
        "phase": "status",
        "protocol": "SAGE.T10.3.11",
        "status": state,
        "manifest_checksum": manifest["manifest_checksum"],
        "accounting": accounting,
        "completed_resets": len(tuple(branch_root.rglob("receipt.json")))
        if branch_root.exists()
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
            "freeze",
            "status",
            "audit",
            "preflight",
            "discover-sequence",
            "reproduce-sequence",
            "compile",
            "confirm",
            "report",
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
        return 0 if report["verdict"] == "PASS_T10_3_11_BOUNDED_GOAL_SOURCE" else 3
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
    "audit",
    "compile_registry",
    "confirm",
    "discover_sequence",
    "main",
    "preflight",
    "reproduce_sequence",
    "status",
    "terminal_report",
]
