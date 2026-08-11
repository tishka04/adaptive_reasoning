"""Durable directional-progress runtime for SAGE.T10.3.10."""

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
from . import t10_3_10_protocol as protocol
from .goal_directed_v10_3_2 import ProgressProgramRegistry
from .goal_directed_v10_3_5 import scheduled_unified_config
from .goal_directed_v10_3_10 import (
    BOUNDED_TRANSITION_HISTORY,
    MAXIMUM_DIRECTIONAL_OPTION_HORIZON,
    MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN,
    DirectionalProgressAutomatonInducer,
    DirectionalProgressSageTController,
    DirectionalProgressUnifiedCognitiveController,
    directional_milestone_descriptor,
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
        DirectionalProgressUnifiedCognitiveController,
        DirectionalProgressSageTController | None,
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
    DirectionalProgressUnifiedCognitiveController,
    DirectionalProgressSageTController | None,
]:
    if work.arm == "unified_sage_t_off":
        pair = (
            DirectionalProgressUnifiedCognitiveController(
                work.game_id,
                config=scheduled_unified_config(sage_t_authority_mode="off"),
            ),
            None,
        )
    else:
        controller_phase = "confirmation" if work.phase == "confirm" else "discovery"
        goal = DirectionalProgressSageTController(
            phase=controller_phase,
            registry=registry,
            registry_checksum=registry_checksum,
            attestation_scope=work.work_id,
            exploration_seed=work.seed,
            reproduce_mixed_registry=work.phase == "reproduce-sequence",
        )
        pair = (
            DirectionalProgressUnifiedCognitiveController(
                work.game_id,
                config=scheduled_unified_config(sage_t_authority_mode="active"),
                sage_t_controller=goal,
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
    inducer = dict(goal_summary.get("directional_inducer", {}))
    diagnostic = _signed(
        {
            "format_version": "sage-t10.3.10-branch-diagnostic-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "phase": work.phase,
            "arm": work.arm,
            "receipt_checksum": receipt["receipt_checksum"],
            "sealed_events": int(receipt.get("sealed_events", 0)),
            "directional_gain_events": int(
                inducer.get("directional_gain_events", 0)
            ),
            "repeated_effect_events": int(
                inducer.get("repeated_effect_events", 0)
            ),
            "effect_stall_aborts": int(goal_summary.get("effect_stall_aborts", 0)),
            "exact_state_cycle_aborts": int(
                goal_summary.get("causal_cycle_aborts", 0)
            ),
            "maximum_identical_action_effect_streak": int(
                inducer.get("maximum_identical_action_effect_streak", 0)
            ),
            "maximum_planned_identical_action_run": int(
                inducer.get("maximum_planned_identical_action_run", 0)
            ),
            "frontiers_composed": int(inducer.get("frontiers_composed", 0)),
            "maximum_retained_transitions": int(
                controller_summary.get("maximum_retained_transitions", 0)
            ),
            "posterior_updates": int(
                controller_summary.get("directional_bounded_observations", 0)
            ),
            "online_relational_rule_verification": bool(
                controller_summary.get("online_relational_rule_verification", True)
            ),
            "raw_frames_retained": False,
            "ephemeral_action_data_persisted": False,
        },
        "diagnostic_checksum",
    )
    protocol.write_json_once(_diagnostic_path(destination, work), diagnostic)
    return receipt


def _load_diagnostics(root: Path, phase: str) -> list[dict[str, Any]]:
    base = _destination(root) / "branch_diagnostics"
    output = []
    for path in sorted(base.glob("*.json")) if base.exists() else ():
        row = durable._read_signed(path, "diagnostic_checksum")
        if row.get("phase") == phase:
            output.append(row)
    return output


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
    active = [row for row in receipts if row.get("arm") == "goal_directed_sage_t"]
    diagnostic_ids = {str(row["work_id"]) for row in diagnostics}
    return {
        "all_conditions_present": len(receipts) == expected,
        "intent_accounting": all(
            int(row.get("issued_intents", 0))
            == int(row.get("sealed_events", 0))
            + int(row.get("unresolved_intents", 0))
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
        "active_diagnostics_present": all(
            str(row["work_id"]) in diagnostic_ids for row in active
        ),
        "bounded_transition_history": all(
            int(row.get("maximum_retained_transitions", 0))
            <= BOUNDED_TRANSITION_HISTORY
            for row in diagnostics
        ),
        "online_rule_verification_deferred": all(
            row.get("online_relational_rule_verification") is False
            for row in diagnostics
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


def _trajectory_metrics(
    root: Path, receipts: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, list[str]], int]:
    destination = _destination(root)
    by_game: dict[str, list[str]] = {}
    maximum_sage_run = 0
    for row in receipts:
        directory = destination / "journal" / "intents" / str(row["work_id"])
        actions: list[str] = []
        last_sage = None
        sage_run = 0
        for path in sorted(directory.glob("*.json")) if directory.exists() else ():
            intent = durable._read_signed(path, "intent_checksum")
            action = str(intent.get("action", {}).get("name", ""))
            actions.append(action)
            if intent.get("decision_source") == "sage_t_joint_program":
                if action == last_sage:
                    sage_run += 1
                else:
                    last_sage = action
                    sage_run = 1
                maximum_sage_run = max(maximum_sage_run, sage_run)
            else:
                last_sage = None
                sage_run = 0
        by_game.setdefault(str(row["game_id"]), []).append(
            protocol.sha256_payload(actions)
        )
    return (
        {game: sorted(values) for game, values in sorted(by_game.items())},
        maximum_sage_run,
    )


def _diversified_trajectories(fingerprints: Mapping[str, Sequence[str]]) -> bool:
    return bool(fingerprints) and all(
        len(values) >= 2 and len(set(values)) >= 2 for values in fingerprints.values()
    )


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    diagnosis = manifest["superseded_t10_3_9"]["diagnosis"]
    contract = manifest["functional_contract"]
    checks = {
        "parent_snapshot_exact": all(
            manifest["superseded_t10_3_9"].get(key) == value
            for key, value in protocol.SUPERSEDED_T10_3_9.items()
        ),
        "parent_partial_exact": diagnosis["intent_count"] == 153
        and diagnosis["event_count"] == 153
        and diagnosis["branch_count"] == 1
        and diagnosis["incomplete_work_count"] == 1,
        "parent_zero_levels": diagnosis["sequence_level_count"] == 0,
        "parent_cycle_identified": diagnosis["maximum_identical_action_run"] == 22,
        "parent_growth_identified": diagnosis[
            "completed_reset_controller_cycle_p95_ms"
        ]
        > 20_000.0,
        "parent_events_diagnostic_only": contract["parent_events_diagnostic_only"]
        and manifest["firewall"]["t10_3_9_events_training_authorized"] is False,
        "parent_registry_forbidden": contract["parent_registry_loaded"] is False
        and manifest["firewall"]["t10_3_9_registry_prior_authorized"] is False,
        "level_only_credit": contract["level_increment_is_only_success_credit"],
        "directional_gain_exploration_only": contract[
            "directional_structural_gain_is_exploration_only"
        ],
        "raw_state_novelty_forbidden": contract["raw_state_novelty_rewarded"]
        is False,
        "bounded_posterior_contract": contract["transition_history_limit"]
        == BOUNDED_TRANSITION_HISTORY
        and contract["relational_rule_verification_deferred"],
        "source_firewalls_closed": bool(
            manifest["firewall"][
                "t10_3_8_core_registry_structural_prior_authorized"
            ]
            and not any(
                value
                for key, value in manifest["firewall"].items()
                if key != "t10_3_8_core_registry_structural_prior_authorized"
            )
        ),
        "budget_exact": manifest["matrix"]["total_maximum_actions"]
        == protocol.TOTAL_MAXIMUM_ACTIONS,
        "reset_exact": manifest["matrix"]["total_resets"]
        == protocol.TOTAL_RESETS,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.10-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "diagnosis": diagnosis,
            "checks": checks,
            "parent_events_used_for_training": 0,
            "parent_registry_loaded": False,
            "parent_physical_actions_replayed": 0,
            "physical_actions": 0,
            "status": (
                "PASS_T10_3_10_OFFLINE_AUDIT"
                if all(checks.values())
                else "INVALID_PROVENANCE"
            ),
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.10 provenance audit failed")
    return payload


@dataclass(frozen=True)
class _SyntheticAction:
    name: str
    action_args: Mapping[str, Any]


def _synthetic_record(action: str, index: int, *, progressed: bool = False) -> Any:
    before = np.zeros((7, 7), dtype=np.int16)
    after = before.copy()
    after[1, 1] = index + 1
    return SimpleNamespace(
        action=SimpleNamespace(name=action),
        diff=SimpleNamespace(
            level_complete=progressed,
            game_over=False,
            is_noop=False,
            moved_objects=((index, (0, 0), (0, 1)),),
            created_objects=(),
            removed_objects=(),
            num_changed=1,
            player_displacement=(0, 1),
        ),
        obs_before=SimpleNamespace(
            levels_completed=0,
            grid_hash=index,
            raw_grid=before,
            objects=(object(),),
            available_actions=("ACTION1", "ACTION2", "ACTION3"),
        ),
        obs_after=SimpleNamespace(
            levels_completed=int(progressed),
            grid_hash=index + 1,
            raw_grid=after,
            objects=(object(),),
            available_actions=("ACTION1", "ACTION2", "ACTION3"),
        ),
    )


def _longest_run(actions: Sequence[str]) -> int:
    maximum = 0
    previous = None
    run = 0
    for action in actions:
        if action == previous:
            run += 1
        else:
            previous = action
            run = 1
        maximum = max(maximum, run)
    return maximum


def _synthetic_closed_loop(seed: int) -> dict[str, Any]:
    goal = DirectionalProgressSageTController(
        phase="preflight",
        exploration_seed=seed,
        attestation_scope=f"directional-synthetic-{seed}",
    )
    controller = DirectionalProgressUnifiedCognitiveController(
        "synthetic-directional-progress",
        config=scheduled_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
    )
    controller.on_reset()
    legal = tuple(_SyntheticAction(f"ACTION{index}", {}) for index in range(1, 4))
    names = tuple(item.name for item in legal)
    grid = np.zeros((9, 9), dtype=np.int16)
    last_action = None
    switches = 0
    levels = 0
    decisions: list[str] = []
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
        decisions.append(action)
        if last_action is not None and action != last_action:
            switches += 1
        last_action = action
        grid = before.copy()
        # Repeating an action changes a timer cell but not the causal mode.
        # Switching actions advances a separate structural marker.
        grid[0, index % grid.shape[1]] = (index % 4) + 1
        grid[2 + min(3, switches), 2] = 7
        after_levels = int(switches >= 3)
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
    summary = dict(controller.summary())
    goal_summary = dict(goal.summary())
    serialized = _canonical(goal_summary.get("registry", {}))
    return {
        "won": levels == 1,
        "actions": len(decisions),
        "decision_fingerprint": protocol.sha256_payload(decisions),
        "distinct_actions": len(set(decisions)),
        "longest_action_run": _longest_run(decisions),
        "posterior_updates": int(summary.get("directional_bounded_observations", 0)),
        "maximum_retained_transitions": int(
            summary.get("maximum_retained_transitions", 0)
        ),
        "online_rule_verification": bool(
            summary.get("online_relational_rule_verification", True)
        ),
        "option_successes": int(goal_summary.get("option_successes", 0)),
        "coordinate_free": all(
            token not in serialized
            for token in (
                '"x"',
                '"y"',
                "raw_grid",
                "entity_id",
                "game_id",
                '"seed"',
                '"color"',
            )
        ),
    }


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    repeated = DirectionalProgressAutomatonInducer()
    repeated.start_branch()
    for index in range(3):
        repeated.observe(
            _synthetic_record("ACTION1", index),
            selected_step=None,
            active_option=None,
        )
    repeated_summary = repeated.summary()
    frontier = repeated.compose_frontier(
        ("ACTION1", "ACTION2", "ACTION3"), rotation=1, horizon=6
    )
    frontier_actions = () if frontier is None else frontier.action_schemas
    descriptor = directional_milestone_descriptor(_synthetic_record("ACTION1", 9))
    left = _synthetic_closed_loop(3391)
    right = _synthetic_closed_loop(3392)
    checks = {
        "changing_frame_not_new_milestone": repeated_summary[
            "directional_gain_events"
        ]
        == 1,
        "repeated_effect_stall_detected": repeated.last_transition_stalled,
        "mixed_frontier_built": frontier is not None
        and frontier.mixed
        and len(frontier.steps) <= MAXIMUM_DIRECTIONAL_OPTION_HORIZON,
        "frontier_identical_run_bounded": _longest_run(frontier_actions)
        <= MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN,
        "closed_loop_wins_from_blank_posterior": left["won"] and right["won"],
        "closed_loop_uses_multiple_actions": left["distinct_actions"] >= 2
        and right["distinct_actions"] >= 2,
        "posterior_each_transition": left["posterior_updates"] == left["actions"]
        and right["posterior_updates"] == right["actions"],
        "history_bounded": left["maximum_retained_transitions"]
        <= BOUNDED_TRANSITION_HISTORY
        and right["maximum_retained_transitions"] <= BOUNDED_TRANSITION_HISTORY,
        "online_rule_verification_deferred": left["online_rule_verification"]
        is False
        and right["online_rule_verification"] is False,
        "level_only_success_credit": left["option_successes"] >= 1
        and right["option_successes"] >= 1,
        "descriptor_identity_free": set(descriptor)
        == {
            "mode",
            "level_progress",
            "terminal",
            "noop",
            "actor_axis",
            "component_delta_sign",
            "action_space_delta_sign",
            "shape_changed",
            "object_set_changed",
            "actor_or_object_moved",
        },
        "coordinate_free": left["coordinate_free"] and right["coordinate_free"],
        "latency_telemetry_only": manifest["functional_contract"][
            "latency_is_telemetry_only"
        ],
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.10-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": {
                "seed_3391": left,
                "seed_3392": right,
                "repeated_effect": repeated_summary,
                "frontier_actions": list(frontier_actions),
            },
            "checks": checks,
            "physical_actions": 0,
            "status": (
                "PASS_T10_3_10_PREFLIGHT"
                if all(checks.values())
                else "DIRECTIONAL_PROGRESS_WIRING_MISS"
            ),
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.10 directional preflight failed")
    return payload


def _require_offline_gates(root: Path) -> None:
    audit_payload = _read_signed(root, AUDIT_FILENAME, "audit_checksum")
    preflight_payload = _read_signed(root, PREFLIGHT_FILENAME, "preflight_checksum")
    if audit_payload.get("status") != "PASS_T10_3_10_OFFLINE_AUDIT":
        raise protocol.ScientificGateMiss("offline audit forbids physical collection")
    if preflight_payload.get("status") != "PASS_T10_3_10_PREFLIGHT":
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
                        registry_checksum
                        if work.arm == "goal_directed_sage_t"
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
    receipts = _collect(root, manifest, "discover-sequence", registry)
    diagnostics = _load_diagnostics(root, "discover-sequence")
    levels = _levels_by_game(receipts)
    mixed_winners = _mixed_winners(receipts)
    fingerprints, maximum_sage_run = _trajectory_metrics(root, receipts)
    checks = {
        **_basic_checks(
            receipts,
            diagnostics,
            len(protocol.work_specs("discover-sequence")),
        ),
        "sequence_progress": sum(levels.values()) >= 1,
        "mixed_winning_automaton": bool(mixed_winners),
        "winning_action_from_sage_t": _winning_sage_source(receipts),
        "seed_diversified_trajectories": _diversified_trajectories(fingerprints),
        "sage_t_identical_action_run_bounded": maximum_sage_run
        <= MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN,
        "parent_core_registry_local_support_zero": True,
        "t10_3_9_registry_not_loaded": True,
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.10-discover-sequence-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "discover-sequence",
            "metrics": {
                "levels": levels,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "mixed_winning_resets": len(mixed_winners),
                "registry_programs": len(registry.snapshot().get("programs", ())),
                "trajectory_fingerprints": fingerprints,
                "maximum_sage_identical_action_run": maximum_sage_run,
                "directional_gain_events": sum(
                    int(row.get("directional_gain_events", 0))
                    for row in diagnostics
                ),
                "effect_stall_aborts": sum(
                    int(row.get("effect_stall_aborts", 0)) for row in diagnostics
                ),
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "diagnostic_checksums": [
                row["diagnostic_checksum"] for row in diagnostics
            ],
            "passed": passed,
            "verdict": (
                "PASS_T10_3_10_DIRECTIONAL_SEQUENCE_DISCOVERY"
                if passed
                else "DIRECTIONAL_SEQUENCE_MISS"
            ),
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, SEQUENCE_REPORT_FILENAME), report)
    protocol.write_json_once(
        _artifact_path(root, SEQUENCE_REGISTRY_FILENAME), registry.snapshot()
    )
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
    levels = _levels_by_game(receipts)
    discovery_positive = {
        game
        for game, value in discovery.get("metrics", {}).get("levels", {}).items()
        if int(value) > 0
    }
    reproduced_positive = {game for game, value in levels.items() if int(value) > 0}
    mixed_winners = _mixed_winners(receipts)
    fingerprints, maximum_sage_run = _trajectory_metrics(root, receipts)
    checks = {
        **_basic_checks(
            receipts,
            diagnostics,
            len(protocol.work_specs("reproduce-sequence")),
        ),
        "fresh_sequence_progress": sum(levels.values()) >= 1,
        "same_game_reproduced": bool(discovery_positive & reproduced_positive),
        "mixed_winning_automaton": bool(mixed_winners),
        "winning_action_from_sage_t": _winning_sage_source(receipts),
        "seed_diversified_trajectories": _diversified_trajectories(fingerprints),
    }
    passed = all(checks.values())
    report = _signed(
        {
            "format_version": "sage-t10.3.10-reproduce-sequence-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": "reproduce-sequence",
            "metrics": {
                "levels": levels,
                "discovery_positive_games": sorted(discovery_positive),
                "reproduced_positive_games": sorted(reproduced_positive),
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "mixed_winning_resets": len(mixed_winners),
                "maximum_sage_identical_action_run": maximum_sage_run,
                **_telemetry(receipts),
            },
            "checks": checks,
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "diagnostic_checksums": [
                row["diagnostic_checksum"] for row in diagnostics
            ],
            "passed": passed,
            "verdict": (
                "PASS_T10_3_10_DIRECTIONAL_SEQUENCE_REPRODUCTION"
                if passed
                else "DIRECTIONAL_SEQUENCE_REPRODUCTION_MISS"
            ),
        },
        "report_checksum",
    )
    protocol.write_json_once(
        _artifact_path(root, REPRODUCTION_REPORT_FILENAME), report
    )
    protocol.write_json_once(
        _artifact_path(root, REPRODUCED_REGISTRY_FILENAME), registry.snapshot()
    )
    if not passed:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def compile_registry(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    reproduction = _read_signed(
        root, REPRODUCTION_REPORT_FILENAME, "report_checksum"
    )
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
            for token in (
                '"x"',
                '"y"',
                "raw_grid",
                "entity_id",
                "game_id",
                '"seed"',
                '"color"',
            )
        ),
    }
    passed = all(checks.values())
    protocol.write_json_once(_artifact_path(root, COMPILED_REGISTRY_FILENAME), compiled)
    report = _signed(
        {
            "format_version": "sage-t10.3.10-compile-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": compiled["registry_checksum"],
            "program_count": len(programs),
            "mixed_program_count": len(mixed),
            "controls": controls,
            "checks": checks,
            "passed": passed,
            "verdict": (
                "PASS_T10_3_10_DIRECTIONAL_REGISTRY"
                if passed
                else "DIRECTIONAL_REGISTRY_REPRODUCTION_MISS"
            ),
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
        **_basic_checks(receipts, diagnostics, len(protocol.work_specs("confirm"))),
        "core_level_each_game": all(
            active_levels[game] >= 1 for game in protocol.CORE_GAMES
        ),
        "sequence_progress": sum(
            active_levels[game] for game in protocol.SEQUENCE_GAMES
        )
        >= 1,
        "no_game_regression": all(
            active_levels[game] >= baseline_levels[game]
            for game in protocol.ALL_SOURCE_GAMES
        ),
        "total_level_advantage": sum(active_levels.values())
        >= sum(baseline_levels.values()) + 1,
        "game_over_nonincrease": sum(
            int(row.get("game_over_actions", 0)) for row in active_rows
        )
        <= sum(int(row.get("game_over_actions", 0)) for row in baseline_rows),
        "registry_loaded": all(
            row.get("registry_checksum_loaded") == registry_checksum
            for row in active_rows
        ),
        "registry_used": all(
            bool(row.get("registry_used_in_decision")) for row in active_rows
        ),
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
            "format_version": "sage-t10.3.10-confirmation-report-v1",
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
            "verdict": (
                "PASS_T10_3_10_SOURCE_CONFIRMATION"
                if passed
                else "SOURCE_CONFIRMATION_MISS"
            ),
        },
        "report_checksum",
    )
    protocol.write_json_once(
        _artifact_path(root, CONFIRMATION_REPORT_FILENAME), report
    )
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
    if artifacts["audit"] is None or artifacts["audit"].get(
        "status"
    ) != "PASS_T10_3_10_OFFLINE_AUDIT":
        verdict = "INVALID_PROVENANCE"
    elif artifacts["preflight"] is None or artifacts["preflight"].get(
        "status"
    ) != "PASS_T10_3_10_PREFLIGHT":
        verdict = "DIRECTIONAL_PROGRESS_WIRING_MISS"
    elif artifacts["discovery"] is None or artifacts["discovery"].get("passed") is not True:
        verdict = "DIRECTIONAL_SEQUENCE_MISS"
    elif artifacts["reproduction"] is None or artifacts["reproduction"].get(
        "passed"
    ) is not True:
        verdict = "DIRECTIONAL_SEQUENCE_REPRODUCTION_MISS"
    elif artifacts["compile"] is None or artifacts["compile"].get("passed") is not True:
        verdict = "DIRECTIONAL_REGISTRY_REPRODUCTION_MISS"
    elif artifacts["confirmation"] is None or artifacts["confirmation"].get(
        "passed"
    ) is not True:
        verdict = "SOURCE_CONFIRMATION_MISS"
    else:
        verdict = "PASS_T10_3_10_DIRECTIONAL_SOURCE"
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.10-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "artifacts": {
                name: None
                if value is None
                else next(
                    (
                        value[key]
                        for key in (
                            "audit_checksum",
                            "preflight_checksum",
                            "report_checksum",
                        )
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
    live = bool(accounting.get("live_collector_lock"))
    state = (
        "RUNNING"
        if live
        else "INTERRUPTED"
        if accounting.get("incomplete_work_ids")
        else "READY"
    )
    branch_root = _destination(root) / "journal" / "branches"
    return {
        "phase": "status",
        "protocol": "SAGE.T10.3.10",
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
        return 0 if report["verdict"] == "PASS_T10_3_10_DIRECTIONAL_SOURCE" else 3
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
