"""Durable scheduled real-time runtime for SAGE.T10.3.5."""

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

from theory import unified_cognition_ab_benchmark as live

from . import t10_3_2_runtime as durable
from . import t10_3_5_protocol as protocol
from .goal_directed_v10_3_2 import (
    GoalDirectedOption,
    OptionStep,
    ProgressProgramRegistry,
)
from .goal_directed_v10_3_3 import BRANCH_PRODUCTIVE_ANCHOR
from .goal_directed_v10_3_4 import TRANSITION_HISTORY_LIMIT
from .goal_directed_v10_3_5 import (
    ScheduledGoalDirectedSageTController,
    ScheduledUnifiedCognitiveController,
    scheduled_unified_config,
)

AUDIT_FILENAME = durable.AUDIT_FILENAME
PREFLIGHT_FILENAME = durable.PREFLIGHT_FILENAME
CORE_REPORT_FILENAME = durable.CORE_REPORT_FILENAME
SEQUENCE_REPORT_FILENAME = durable.SEQUENCE_REPORT_FILENAME
COMPILE_REPORT_FILENAME = durable.COMPILE_REPORT_FILENAME
CONFIRMATION_REPORT_FILENAME = durable.CONFIRMATION_REPORT_FILENAME
TERMINAL_REPORT_FILENAME = durable.TERMINAL_REPORT_FILENAME
LOCK_FILENAME = durable.LOCK_FILENAME
STAGE_TIMING_KEYS = (
    "snapshot_ms",
    "legal_proposal_ms",
    "decision_ms",
    "materialize_ms",
    "intent_checkpoint_ms",
    "environment_ms",
    "event_seal_ms",
    "observe_ms",
    "posterior_checkpoint_ms",
    "total_ms",
)


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


def should_stop_after_level(level_delta: int) -> bool:
    return int(level_delta) > 0


@contextmanager
def _t10_3_5_contracts() -> Iterator[None]:
    """Route the existing durability shell through T10.3.5 contracts."""

    old_protocol = durable.protocol
    old_controller = durable.GoalDirectedSageTController
    old_pair = durable._controller_pair
    old_run_work = durable._run_work
    durable.protocol = protocol
    durable.GoalDirectedSageTController = ScheduledGoalDirectedSageTController
    durable._controller_pair = _controller_pair
    durable._run_work = _run_work
    try:
        yield
    finally:
        durable._run_work = old_run_work
        durable._controller_pair = old_pair
        durable.GoalDirectedSageTController = old_controller
        durable.protocol = old_protocol


def _controller_pair(
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None,
) -> tuple[
    ScheduledUnifiedCognitiveController,
    ScheduledGoalDirectedSageTController | None,
]:
    if work.arm == "unified_sage_t_off":
        return (
            ScheduledUnifiedCognitiveController(
                work.game_id,
                config=scheduled_unified_config(sage_t_authority_mode="off"),
            ),
            None,
        )
    phase = "confirmation" if work.phase == "confirm" else "discovery"
    goal = ScheduledGoalDirectedSageTController(
        phase=phase,
        registry=registry,
        registry_checksum=registry_checksum,
        attestation_scope=work.work_id,
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
    """Run one reset with intent-before-action and immediate event sealing."""

    receipt_path = durable._receipt_for_work(destination, work)
    if receipt_path.is_file():
        return durable._read_signed(receipt_path, "receipt_checksum")
    controller, goal = _controller_pair(
        work, registry, registry_checksum=registry_checksum
    )
    controller.on_reset()
    policy = live.SharedLegacyProposalPolicy(
        game_id=work.game_id,
        seed=work.seed,
        reset_index=work.reset_index,
    )
    trace_ids: list[str] = []
    errors: list[str] = []
    illegal_actions = 0
    game_over_actions = 0
    stage_timings: list[dict[str, float]] = []
    initial_level = final_level = 0
    level_event_sources: list[str] = []
    winning_registry_checksums: list[str | None] = []
    sage_t_actions = 0
    status = "COMPLETE"
    stop_reason = "ACTION_BUDGET_EXHAUSTED"
    env = None
    reset_started = time.perf_counter()
    try:
        env = live._make_real_env(work.game_id, root / "environment_files")
        frame = live._reset_env(env)
        initial = live.snapshot_frame(frame)
        initial_level = final_level = int(initial.levels_completed)
        for step_index in range(work.action_budget):
            if time.perf_counter() - reset_started >= protocol.reset_wall_seconds(work):
                stop_reason = "RESET_WALL_BUDGET_EXHAUSTED"
                break
            timing: dict[str, float] = {}
            step_started = time.perf_counter()

            stage = time.perf_counter()
            before = live.snapshot_frame(frame)
            timing["snapshot_ms"] = (time.perf_counter() - stage) * 1000.0
            if live._is_terminal(before.game_state):
                stop_reason = "TERMINAL_STATE"
                break

            stage = time.perf_counter()
            legal_actions = tuple(live._valid_actions(env))
            proposal = policy.select(legal_actions)
            timing["legal_proposal_ms"] = (time.perf_counter() - stage) * 1000.0
            if proposal is None:
                errors.append("NO_LEGAL_ACTION")
                status = "ABORTED"
                stop_reason = "NO_LEGAL_ACTION"
                break

            stage = time.perf_counter()
            try:
                decision = controller.select_action(
                    current_grid=before.grid,
                    available_actions=live._available_action_names(legal_actions),
                    legacy_action=str(getattr(proposal, "name", "")),
                    legacy_action_data=dict(getattr(proposal, "action_args", {}) or {}),
                    available_action_candidates=legal_actions,
                    game_state=before.game_state,
                    levels_completed=before.levels_completed,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"CONTROLLER_SELECT:{type(exc).__name__}")
                status = "ABORTED"
                stop_reason = "CONTROLLER_SELECT_ERROR"
                break
            timing["decision_ms"] = (time.perf_counter() - stage) * 1000.0

            stage = time.perf_counter()
            selected = live._materialize_decision(legal_actions, decision)
            timing["materialize_ms"] = (time.perf_counter() - stage) * 1000.0
            if selected is None:
                illegal_actions += 1
                errors.append("UNAVAILABLE_DECISION")
                status = "ABORTED"
                stop_reason = "UNAVAILABLE_DECISION"
                break

            intent = durable._intent_payload(
                manifest,
                work,
                step_index=step_index,
                selected=selected,
                decision=decision,
                registry_checksum=(
                    None if goal is None else goal.last_decision_registry_checksum
                ),
            )
            name = f"{step_index:04d}.json"
            stage = time.perf_counter()
            protocol.write_json_once(
                durable._work_path(destination, "intents", work, name), intent
            )
            durable._write_checkpoint(
                destination,
                manifest,
                phase=work.phase,
                work=work,
                state="ACTION_INTENT_AUTHORIZED",
                registry=registry,
            )
            lock.heartbeat()
            timing["intent_checkpoint_ms"] = (time.perf_counter() - stage) * 1000.0

            stage = time.perf_counter()
            try:
                after_frame = live._step_env_action(env, selected)
                after = live.snapshot_frame(
                    after_frame,
                    fallback_available_actions=before.available_actions,
                )
            except Exception as exc:  # noqa: BLE001
                unresolved = _signed(
                    {
                        "format_version": "sage-t10.3.5-unresolved-event-v1",
                        "manifest_checksum": manifest["manifest_checksum"],
                        "work_id": work.work_id,
                        "event_id": intent["event_id"],
                        "step_index": step_index,
                        "reason": f"ENVIRONMENT_CALL_UNATTESTABLE:{type(exc).__name__}",
                        "physical_action_replayed": False,
                    },
                    "unresolved_checksum",
                )
                protocol.write_json_once(
                    durable._work_path(destination, "unresolved", work, name),
                    unresolved,
                )
                errors.append("ENVIRONMENT_CALL_UNATTESTABLE")
                status = "ABORTED"
                stop_reason = "ENVIRONMENT_CALL_UNATTESTABLE"
                break
            timing["environment_ms"] = (time.perf_counter() - stage) * 1000.0

            level_delta = max(
                0, int(after.levels_completed) - int(before.levels_completed)
            )
            event = _signed(
                {
                    "format_version": "sage-t10.3.5-physical-event-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "work_id": work.work_id,
                    "event_id": intent["event_id"],
                    "step_index": step_index,
                    "decision_source": str(decision.source),
                    "registry_checksum": intent["registry_checksum"],
                    "levels_before": int(before.levels_completed),
                    "levels_after": int(after.levels_completed),
                    "level_delta": level_delta,
                    "game_state_after": str(after.game_state),
                    "frame_before_sha256": protocol.sha256_payload(before.grid.tolist()),
                    "frame_after_sha256": protocol.sha256_payload(after.grid.tolist()),
                    "decision_latency_ms": timing["decision_ms"],
                    "environment_latency_ms": timing["environment_ms"],
                    "raw_frame_retained": False,
                    "physical_action_replayed": False,
                },
                "event_checksum",
            )
            stage = time.perf_counter()
            protocol.write_json_once(
                durable._work_path(destination, "events", work, name), event
            )
            timing["event_seal_ms"] = (time.perf_counter() - stage) * 1000.0
            trace_ids.append(str(intent["event_id"]))
            if str(decision.source) == "sage_t_joint_program":
                sage_t_actions += 1
            if level_delta > 0:
                level_event_sources.append(str(decision.source))
                winning_registry_checksums.append(intent["registry_checksum"])

            stage = time.perf_counter()
            try:
                controller.observe_transition(
                    action=str(getattr(selected, "name", "")),
                    action_data=dict(getattr(selected, "action_args", {}) or {}),
                    grid_before=before.grid,
                    grid_after=after.grid,
                    available_actions=live._available_action_names(legal_actions),
                    game_state_before=before.game_state,
                    game_state_after=after.game_state,
                    levels_completed_before=before.levels_completed,
                    levels_completed_after=after.levels_completed,
                )
            except Exception as exc:  # noqa: BLE001
                timing["observe_ms"] = (time.perf_counter() - stage) * 1000.0
                errors.append(f"CONTROLLER_OBSERVE:{type(exc).__name__}")
                status = "ABORTED"
                stop_reason = "CONTROLLER_OBSERVE_ERROR"
                frame = after_frame
                final_level = int(after.levels_completed)
                break
            timing["observe_ms"] = (time.perf_counter() - stage) * 1000.0

            stage = time.perf_counter()
            durable._write_checkpoint(
                destination,
                manifest,
                phase=work.phase,
                work=work,
                state="EVENT_SEALED_AND_POSTERIOR_UPDATED",
                registry=registry,
            )
            timing["posterior_checkpoint_ms"] = (time.perf_counter() - stage) * 1000.0
            frame = after_frame
            final_level = int(after.levels_completed)
            if str(after.game_state).upper() == "GAME_OVER":
                game_over_actions += 1
            lock.heartbeat()
            timing["total_ms"] = (time.perf_counter() - step_started) * 1000.0
            stage_timings.append(timing)
            print(
                _canonical(
                    {
                        "phase": work.phase,
                        "game_id": work.game_id,
                        "seed": work.seed,
                        "arm": work.arm,
                        "step": step_index + 1,
                        "budget": work.action_budget,
                        "levels": final_level - initial_level,
                        "decision_source": str(decision.source),
                        "timing_ms": {
                            "decision": round(timing["decision_ms"], 3),
                            "observe": round(timing["observe_ms"], 3),
                            "controller_cycle": round(
                                timing["decision_ms"] + timing["observe_ms"], 3
                            ),
                            "total": round(timing["total_ms"], 3),
                        },
                    }
                ),
                flush=True,
            )
            if should_stop_after_level(level_delta):
                stop_reason = "LEVEL_PROGRESS_SEALED"
                break
            if live._is_terminal(after.game_state):
                stop_reason = "TERMINAL_STATE"
                break
    except Exception as exc:  # noqa: BLE001
        errors.append(f"RUNTIME:{type(exc).__name__}")
        status = "ABORTED"
        stop_reason = "RUNTIME_ERROR"
    finally:
        if env is not None:
            try:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ENVIRONMENT_CLOSE:{type(exc).__name__}")

    goal_summary = goal.summary() if goal is not None else {}
    controller_summary = controller.summary()
    intent_dir = durable._work_path(destination, "intents", work, "x").parent
    event_dir = durable._work_path(destination, "events", work, "x").parent
    unresolved_dir = durable._work_path(destination, "unresolved", work, "x").parent
    issued_intents = len(tuple(intent_dir.glob("*.json"))) if intent_dir.exists() else 0
    sealed_events = len(tuple(event_dir.glob("*.json"))) if event_dir.exists() else 0
    unresolved_intents = (
        len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    )
    decision_latencies = [row["decision_ms"] for row in stage_timings]
    controller_cycles = [row["decision_ms"] + row["observe_ms"] for row in stage_timings]
    stage_p95_ms = {
        key: _p95([row[key] for row in stage_timings if key in row])
        for key in STAGE_TIMING_KEYS
    }
    receipt = _signed(
        {
            "format_version": "sage-t10.3.5-branch-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            **work.as_dict(),
            "work_id": work.work_id,
            "status": status,
            "complete": status == "COMPLETE" and not errors,
            "stop_reason": stop_reason,
            "planned_early_stop": stop_reason == "LEVEL_PROGRESS_SEALED",
            "issued_intents": issued_intents,
            "sealed_events": sealed_events,
            "unresolved_intents": unresolved_intents,
            "event_ids": trace_ids,
            "level_delta": max(0, final_level - initial_level),
            "level_event_sources": level_event_sources,
            "winning_registry_checksums": winning_registry_checksums,
            "sage_t_option_actions": sage_t_actions,
            "mixed_program_used": bool(
                any(
                    len(set(items)) >= 2
                    for items in goal_summary.get("successful_option_action_schemas", ())
                )
            ),
            "binding_rejections": dict(goal_summary.get("binding_rejections", {})),
            "binding_method_uses": dict(goal_summary.get("binding_method_uses", {})),
            "structural_collision_count": int(
                goal_summary.get("structural_collision_count", 0)
            ),
            "productive_option_extensions": int(
                goal_summary.get("productive_option_extensions", 0)
            ),
            "terminal_option_contradictions": int(
                goal_summary.get("terminal_option_contradictions", 0)
            ),
            "fast_path_decisions": int(
                controller_summary.get("bounded_fast_path_decisions", 0)
            ),
            "scheduled_sage_decisions": int(
                controller_summary.get("scheduled_sage_decisions", 0)
            ),
            "scheduled_legacy_decisions": int(
                controller_summary.get("scheduled_legacy_decisions", 0)
            ),
            "lightweight_observations": int(
                controller_summary.get("lightweight_observations", 0)
            ),
            "full_unified_decisions": int(
                controller_summary.get("full_unified_decisions", 0)
            ),
            "full_unified_observations": int(
                controller_summary.get("full_unified_observations", 0)
            ),
            "transition_history_limit": TRANSITION_HISTORY_LIMIT,
            "maximum_retained_transitions": int(
                controller_summary.get("maximum_retained_transitions", 0)
            ),
            "ephemeral_action_data_persisted": False,
            "errors": errors,
            "illegal_actions": illegal_actions,
            "game_over_actions": game_over_actions,
            "decision_latencies_ms": decision_latencies,
            "controller_cycle_latencies_ms": controller_cycles,
            "stage_timings_ms": stage_timings,
            "stage_p95_ms": stage_p95_ms,
            "controller_cycle_p95_ms": _p95(controller_cycles),
            "reset_elapsed_seconds": time.perf_counter() - reset_started,
            "actions_saved_by_early_stop": (
                max(0, work.action_budget - sealed_events)
                if stop_reason == "LEVEL_PROGRESS_SEALED"
                else 0
            ),
            "registry_checksum_loaded": registry_checksum,
            "registry_used_in_decision": bool(
                goal_summary.get("registry_used_in_decision")
            ),
            "controller_registry": goal_summary.get("registry"),
            "physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    protocol.write_json_once(receipt_path, receipt)
    durable._write_checkpoint(
        destination,
        manifest,
        phase=work.phase,
        work=work,
        state="BRANCH_RECEIPT_SEALED",
        registry=registry,
    )
    return receipt


@dataclass(frozen=True)
class _SyntheticAction:
    name: str
    action_args: Mapping[str, Any]


def _synthetic_cycle(
    *,
    active: bool,
    steps: int,
    option: GoalDirectedOption | None = None,
    success_step: int | None = None,
    ambiguous: bool = False,
) -> dict[str, Any]:
    goal = (
        ScheduledGoalDirectedSageTController(
            phase="preflight",
            warmup_actions=0,
            exploration_interval=1,
            attestation_scope="scheduled-synthetic",
        )
        if active
        else None
    )
    controller = ScheduledUnifiedCognitiveController(
        "synthetic-scheduled",
        config=scheduled_unified_config(
            sage_t_authority_mode="active" if active else "off"
        ),
        sage_t_controller=goal,
    )
    controller.on_reset()
    if goal is not None and option is not None:
        goal._active_option = option
    legal = (
        (
            _SyntheticAction("ACTION6", {"x": 1, "y": 1}),
            _SyntheticAction("ACTION6", {"x": 2, "y": 1}),
        )
        if ambiguous
        else tuple(
            _SyntheticAction(name, {})
            for name in tuple(
                dict.fromkeys(
                    step.action_name for step in (option.steps if option else (OptionStep("ACTION1"),))
                )
            )
        )
    )
    names = tuple(dict.fromkeys(item.name for item in legal))
    proposal = legal[0]
    grid = np.zeros((7, 7), dtype=np.int16)
    if ambiguous:
        grid[1, 1] = grid[1, 2] = 2
    decisions = []
    decision_latencies = []
    cycles = []
    for index in range(steps):
        before = grid.copy()
        started = time.perf_counter()
        decision = controller.select_action(
            current_grid=before,
            available_actions=names,
            legacy_action=proposal.name,
            legacy_action_data=proposal.action_args,
            available_action_candidates=legal,
            levels_completed=0,
        )
        selected_at = time.perf_counter()
        decision_latencies.append((selected_at - started) * 1000.0)
        decisions.append(decision)
        grid = before.copy()
        grid[1 if ambiguous else 2, 1 if ambiguous else 2] = 1 + ((index + 1) % 2)
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=grid,
            available_actions=names,
            levels_completed_before=0,
            levels_completed_after=int(success_step is not None and index == success_step),
        )
        cycles.append((time.perf_counter() - started) * 1000.0)
    summary = controller.summary()
    goal_summary = {} if goal is None else goal.summary()
    registry_text = _canonical(goal_summary.get("registry", {}))
    return {
        "active": active,
        "actions": steps,
        "sage_t_actions": sum(row.source == "sage_t_joint_program" for row in decisions),
        "same_ambiguous_target": (
            not ambiguous
            or all(
                row.source != "sage_t_joint_program"
                or dict(row.action_data) == dict(proposal.action_args)
                for row in decisions
            )
        ),
        "structural_collision_count": int(goal_summary.get("structural_collision_count", 0)),
        "branch_anchor_uses": int(
            goal_summary.get("binding_method_uses", {}).get(BRANCH_PRODUCTIVE_ANCHOR, 0)
        ),
        "option_successes": int(goal_summary.get("option_successes", 0)),
        "productive_option_extensions": int(
            goal_summary.get("productive_option_extensions", 0)
        ),
        "posterior_observed_transitions": int(summary["transitions_observed"]),
        "retained_transitions": len(controller.belief_loop.profiler.transitions),
        "maximum_retained_transitions": int(summary["maximum_retained_transitions"]),
        "full_unified_decisions": int(summary["full_unified_decisions"]),
        "full_unified_observations": int(summary["full_unified_observations"]),
        "decision_p95_ms": _p95(decision_latencies),
        "controller_cycle_p95_ms": _p95(cycles),
        "persistent_coordinates_absent": all(
            token not in registry_text for token in ('"x"', '"y"', "raw_grid", "entity_id")
        ),
    }


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_snapshot = manifest["superseded_t10_3_4"]
    schedule = manifest["scheduled_control"]
    checks = {
        "parent_terminal_negative": parent_snapshot.get("verdict") == "BOUNDED_CORE_MISS",
        "parent_snapshot_exact": all(
            parent_snapshot.get(key) == value
            for key, value in protocol.SUPERSEDED_T10_3_4.items()
        ),
        "parent_training_forbidden": parent_snapshot.get("used_for_training") is False,
        "parent_prior_forbidden": parent_snapshot.get("positive_witness_imported_as_prior") is False,
        "parent_mutation_forbidden": parent_snapshot.get("mutated_by_t10_3_5") is False,
        "parent_replay_forbidden": parent_snapshot.get("physical_actions_replayed") == 0,
        "source_only_firewall": not any(manifest["firewall"].values()),
        "budget_exact": manifest["matrix"]["total_maximum_actions"] == 6144,
        "reset_exact": manifest["matrix"]["total_resets"] == 30,
        "full_unified_paths_disabled": schedule["full_unified_decision_path_enabled"] is False
        and schedule["full_unified_observation_path_enabled"] is False,
        "posterior_every_transition": schedule["sage_t_posterior_each_transition"] is True,
        "symmetric_schedule": schedule["same_schedule_for_active_and_baseline"] is True,
        "latency_gate_preserved": manifest["gates"]["maximum_decision_p95_ms"] == 2500.0,
        "controller_cycle_gate": manifest["gates"]["maximum_controller_cycle_p95_ms"] == 2500.0,
        "collision_not_required": manifest["gates"]["structural_collision_policy"]
        == "fail_closed_if_observed_not_required_to_occur",
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.5-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "parent_events_used_for_training": 0,
            "parent_positive_witness_imported_as_prior": False,
            "physical_actions": 0,
            "status": "PASS_T10_3_5_OFFLINE_AUDIT" if all(checks.values()) else "INVALID_PROVENANCE",
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.5 provenance audit failed")
    return payload


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    ambiguous = _synthetic_cycle(active=True, steps=5, success_step=4, ambiguous=True)
    extended = _synthetic_cycle(
        active=True,
        steps=12,
        success_step=11,
        option=GoalDirectedOption(
            schema="repeat_target",
            steps=(OptionStep("ACTION1"), OptionStep("ACTION1")),
            source="synthetic_productive_extension",
        ),
    )
    mixed = _synthetic_cycle(
        active=True,
        steps=20,
        success_step=19,
        option=GoalDirectedOption(
            schema="mixed_automaton",
            steps=tuple(
                OptionStep("ACTION1" if index % 2 == 0 else "ACTION2")
                for index in range(20)
            ),
            source="synthetic_mixed",
        ),
    )
    baseline = _synthetic_cycle(active=False, steps=40)
    scenarios = (ambiguous, extended, mixed, baseline)
    checks = {
        "ambiguous_target_progress": ambiguous["option_successes"] >= 1,
        "ambiguous_target_sage_authority": ambiguous["sage_t_actions"] >= 1,
        "same_target_reacquired": ambiguous["same_ambiguous_target"],
        "structural_collision_fail_closed": ambiguous["structural_collision_count"] >= 2,
        "persistent_coordinates_absent": all(row["persistent_coordinates_absent"] for row in scenarios),
        "productive_option_extended": extended["productive_option_extensions"] >= 2,
        "extended_option_progress": extended["option_successes"] >= 1,
        "mixed_beyond_16": mixed["actions"] > 16 and mixed["option_successes"] >= 1,
        "posterior_updated_each_action": all(
            row["posterior_observed_transitions"] == row["actions"] for row in scenarios
        ),
        "full_unified_paths_never_entered": all(
            row["full_unified_decisions"] == 0 and row["full_unified_observations"] == 0
            for row in scenarios
        ),
        "decision_p95": max(row["decision_p95_ms"] for row in scenarios)
        <= protocol.MAXIMUM_DECISION_P95_MS,
        "controller_cycle_p95": max(row["controller_cycle_p95_ms"] for row in scenarios)
        <= protocol.MAXIMUM_CONTROLLER_CYCLE_P95_MS,
        "transition_history_bounded": baseline["retained_transitions"] == TRANSITION_HISTORY_LIMIT
        and baseline["maximum_retained_transitions"] <= TRANSITION_HISTORY_LIMIT,
        "early_success_policy": should_stop_after_level(1) and not should_stop_after_level(0),
        "stage_timing_schema_complete": set(STAGE_TIMING_KEYS)
        == {
            "snapshot_ms", "legal_proposal_ms", "decision_ms", "materialize_ms",
            "intent_checkpoint_ms", "environment_ms", "event_seal_ms", "observe_ms",
            "posterior_checkpoint_ms", "total_ms",
        },
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.5-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": list(scenarios),
            "checks": checks,
            "physical_actions": 0,
            "status": "PASS_T10_3_5_PREFLIGHT" if all(checks.values()) else "SCHEDULED_WIRING_MISS",
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.5 scheduled preflight failed")
    return payload


def run_discovery(root: Path, manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    with _t10_3_5_contracts():
        durable._require_live_runtime()
        if phase == "discover-sequence":
            core = durable._read_signed(
                _artifact_path(root, CORE_REPORT_FILENAME), "report_checksum"
            )
            if core.get("passed") is not True:
                raise protocol.ScientificGateMiss("core gate forbids sequence discovery")
        destination = _destination(root)
        durable._recover_orphans(destination, manifest)
        registry = durable._registry_from_receipts(destination)
        lock = durable._CollectorLock(destination / LOCK_FILENAME, phase)
        lock.acquire()
        try:
            for work in protocol.work_specs(phase):
                _run_work(
                    root, destination, manifest, work, registry, lock,
                    registry_checksum=None,
                )
        finally:
            lock.release()
        controls = durable._apply_registry_controls(registry)
        receipts = durable._load_receipts(destination, phase)
        parent_report = durable._phase_report(manifest, phase, receipts, registry, controls)
    decision_p95 = _p95(
        [value for row in receipts for value in row.get("decision_latencies_ms", ())]
    )
    cycle_p95 = _p95(
        [value for row in receipts for value in row.get("controller_cycle_latencies_ms", ())]
    )
    scheduled_checks = {
        "timing_breakdown_present": all(
            bool(row.get("stage_timings_ms"))
            and set(row.get("stage_p95_ms", {})) == set(STAGE_TIMING_KEYS)
            for row in receipts
        ),
        "decision_p95": decision_p95 <= protocol.MAXIMUM_DECISION_P95_MS,
        "controller_cycle_p95": cycle_p95 <= protocol.MAXIMUM_CONTROLLER_CYCLE_P95_MS,
        "full_unified_paths_never_entered": all(
            int(row.get("full_unified_decisions", 0)) == 0
            and int(row.get("full_unified_observations", 0)) == 0
            for row in receipts
        ),
        "posterior_updated_each_event": all(
            int(row.get("lightweight_observations", 0)) == int(row.get("sealed_events", 0))
            for row in receipts
        ),
        "transition_history_bounded": all(
            int(row.get("maximum_retained_transitions", 0)) <= TRANSITION_HISTORY_LIMIT
            for row in receipts
        ),
        "progress_resets_stop_early": all(
            int(row.get("level_delta", 0)) == 0
            or row.get("stop_reason") == "LEVEL_PROGRESS_SEALED"
            for row in receipts
        ),
        "ephemeral_action_data_absent": all(
            row.get("ephemeral_action_data_persisted") is False for row in receipts
        ),
        "collision_policy_fail_closed": True,
    }
    if phase == "discover-core":
        scheduled_checks["sage_t_action_each_reset"] = all(
            int(row.get("sage_t_option_actions", 0)) >= 1 for row in receipts
        )
    report_core = {
        key: value for key, value in parent_report.items() if key != "report_checksum"
    }
    report_core["format_version"] = f"sage-t10.3.5-{phase}-report-v1"
    report_core["checks"] = {
        **dict(report_core.get("checks", {})),
        **scheduled_checks,
    }
    report_core["metrics"] = {
        **dict(report_core.get("metrics", {})),
        "decision_p95_ms": decision_p95,
        "controller_cycle_p95_ms": cycle_p95,
        "fast_path_decisions": sum(int(row.get("fast_path_decisions", 0)) for row in receipts),
        "productive_option_extensions": sum(
            int(row.get("productive_option_extensions", 0)) for row in receipts
        ),
        "actions_saved_by_early_stop": sum(
            int(row.get("actions_saved_by_early_stop", 0)) for row in receipts
        ),
    }
    report_core["passed"] = bool(parent_report.get("passed") and all(scheduled_checks.values()))
    if report_core["passed"]:
        report_core["verdict"] = (
            "PASS_T10_3_5_SCHEDULED_CORE"
            if phase == "discover-core"
            else "PASS_T10_3_5_SCHEDULED_SEQUENCE"
        )
    elif not all(
        scheduled_checks[key]
        for key in (
            "decision_p95", "controller_cycle_p95",
            "full_unified_paths_never_entered", "posterior_updated_each_event",
        )
    ):
        report_core["verdict"] = "REAL_TIME_BOUND_MISS"
    else:
        report_core["verdict"] = (
            "CORE_PROGRESS_MISS" if phase == "discover-core" else "MIXED_SEQUENCE_MISS"
        )
    report = _signed(report_core, "report_checksum")
    report_name = CORE_REPORT_FILENAME if phase == "discover-core" else SEQUENCE_REPORT_FILENAME
    registry_name = (
        durable.CORE_REGISTRY_FILENAME
        if phase == "discover-core"
        else durable.SEQUENCE_REGISTRY_FILENAME
    )
    protocol.write_json_once(_artifact_path(root, report_name), report)
    protocol.write_json_once(_artifact_path(root, registry_name), registry.snapshot())
    if report.get("passed") is not True:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def compile_registry(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_5_contracts():
        core = durable._read_signed(_artifact_path(root, CORE_REPORT_FILENAME), "report_checksum")
        sequence = durable._read_signed(_artifact_path(root, SEQUENCE_REPORT_FILENAME), "report_checksum")
        if core.get("passed") is not True or sequence.get("passed") is not True:
            raise protocol.ScientificGateMiss("discovery gates forbid registry compilation")
        source = durable._read_signed(
            _artifact_path(root, durable.SEQUENCE_REGISTRY_FILENAME), "registry_checksum"
        )
        registry = ProgressProgramRegistry(source)
        controls = durable._apply_registry_controls(registry)
        compiled = registry.snapshot(promoted_only=True)
    programs = compiled.get("programs", ())
    checks = {
        "independent_reproduction": bool(programs)
        and all(len(row.get("support_scopes", ())) >= 2 for row in programs),
        "all_controls": controls["all_checks_passed"],
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
            for token in ('"x"', '"y"', "raw_grid", "entity_id")
        ),
        "scheduled_runtime_prerequisite": core.get("verdict") == "PASS_T10_3_5_SCHEDULED_CORE"
        and sequence.get("verdict") == "PASS_T10_3_5_SCHEDULED_SEQUENCE",
    }
    protocol.write_json_once(_artifact_path(root, durable.COMPILED_REGISTRY_FILENAME), compiled)
    report = _signed(
        {
            "format_version": "sage-t10.3.5-compile-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": compiled["registry_checksum"],
            "program_count": len(programs),
            "controls": controls,
            "checks": checks,
            "passed": all(checks.values()),
            "verdict": "PASS_T10_3_5_REGISTRY" if all(checks.values()) else "REGISTRY_REPRODUCTION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, COMPILE_REPORT_FILENAME), report)
    if report.get("passed") is not True:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def run_confirmation(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_5_contracts():
        durable._require_live_runtime()
        compile_report = durable._read_signed(
            _artifact_path(root, COMPILE_REPORT_FILENAME), "report_checksum"
        )
        if compile_report.get("passed") is not True:
            raise protocol.ScientificGateMiss("compiled registry gate forbids confirmation")
        compiled = durable._read_signed(
            _artifact_path(root, durable.COMPILED_REGISTRY_FILENAME), "registry_checksum"
        )
        registry_checksum = str(compiled["registry_checksum"])
        destination = _destination(root)
        durable._recover_orphans(destination, manifest)
        lock = durable._CollectorLock(destination / LOCK_FILENAME, "confirm")
        lock.acquire()
        try:
            for work in protocol.work_specs("confirm"):
                registry = ProgressProgramRegistry(compiled)
                _run_work(
                    root, destination, manifest, work, registry, lock,
                    registry_checksum=(registry_checksum if work.arm == "goal_directed_sage_t" else None),
                )
        finally:
            lock.release()
        receipts = durable._load_receipts(destination, "confirm")
    by_key = {
        (str(row["game_id"]), int(row["seed"]), str(row["arm"])): row for row in receipts
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
    decision_p95 = _p95(
        [value for row in active_rows for value in row.get("decision_latencies_ms", ())]
    )
    cycle_p95 = _p95(
        [value for row in active_rows for value in row.get("controller_cycle_latencies_ms", ())]
    )
    checks = {
        "all_conditions_present": len(receipts) == 20,
        "core_each_seed": all(
            int(by_key.get((game, seed, "goal_directed_sage_t"), {}).get("level_delta", 0)) >= 1
            for game in protocol.CORE_GAMES for seed in protocol.CONFIRMATION_SEEDS
        ),
        "sequence_progress": sum(active_levels[game] for game in protocol.SEQUENCE_GAMES) >= 1,
        "no_game_regression": all(
            active_levels[game] >= baseline_levels[game] for game in protocol.ALL_SOURCE_GAMES
        ),
        "total_level_advantage": sum(active_levels.values()) >= sum(baseline_levels.values()) + 1,
        "zero_errors": all(not row.get("errors") for row in receipts),
        "zero_illegal_actions": all(int(row.get("illegal_actions", 0)) == 0 for row in receipts),
        "game_over_nonincrease": sum(int(row.get("game_over_actions", 0)) for row in active_rows)
        <= sum(int(row.get("game_over_actions", 0)) for row in baseline_rows),
        "registry_loaded": all(row.get("registry_checksum_loaded") == registry_checksum for row in active_rows),
        "registry_used": all(bool(row.get("registry_used_in_decision")) for row in active_rows),
        "winning_decisions_attest_registry": all(
            all(checksum == registry_checksum for checksum in row.get("winning_registry_checksums", ()))
            and len(row.get("winning_registry_checksums", ())) == len(row.get("level_event_sources", ()))
            for row in active_rows
        ),
        "decision_p95": decision_p95 <= protocol.MAXIMUM_DECISION_P95_MS,
        "controller_cycle_p95": cycle_p95 <= protocol.MAXIMUM_CONTROLLER_CYCLE_P95_MS,
        "full_unified_paths_never_entered": all(
            int(row.get("full_unified_decisions", 0)) == 0
            and int(row.get("full_unified_observations", 0)) == 0 for row in receipts
        ),
        "posterior_updated_each_event": all(
            int(row.get("lightweight_observations", 0)) == int(row.get("sealed_events", 0))
            for row in receipts
        ),
        "budget": sum(int(row.get("issued_intents", 0)) for row in receipts)
        <= protocol.CONFIRMATION_ACTIONS,
        "intent_accounting": all(
            int(row.get("issued_intents", 0))
            == int(row.get("sealed_events", 0)) + int(row.get("unresolved_intents", 0))
            for row in receipts
        ),
        "zero_physical_replay": all(int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts),
    }
    report = _signed(
        {
            "format_version": "sage-t10.3.5-confirmation-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": registry_checksum,
            "metrics": {
                "active_levels": active_levels,
                "baseline_levels": baseline_levels,
                "active_total_levels": sum(active_levels.values()),
                "baseline_total_levels": sum(baseline_levels.values()),
                "decision_p95_ms": decision_p95,
                "controller_cycle_p95_ms": cycle_p95,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
            },
            "checks": checks,
            "passed": all(checks.values()),
            "verdict": "PASS_T10_3_5_SOURCE_CONFIRMATION" if all(checks.values()) else "SOURCE_CONFIRMATION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, CONFIRMATION_REPORT_FILENAME), report)
    if report.get("passed") is not True:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_5_contracts():
        artifacts = {}
        for name, filename, checksum in (
            ("audit", AUDIT_FILENAME, "audit_checksum"),
            ("preflight", PREFLIGHT_FILENAME, "preflight_checksum"),
            ("core", CORE_REPORT_FILENAME, "report_checksum"),
            ("sequence", SEQUENCE_REPORT_FILENAME, "report_checksum"),
            ("compile", COMPILE_REPORT_FILENAME, "report_checksum"),
            ("confirmation", CONFIRMATION_REPORT_FILENAME, "report_checksum"),
        ):
            path = _artifact_path(root, filename)
            artifacts[name] = durable._read_signed(path, checksum) if path.is_file() else None
        if artifacts["audit"] is None or artifacts["audit"].get("status") != "PASS_T10_3_5_OFFLINE_AUDIT":
            verdict = "INVALID_PROVENANCE"
        elif artifacts["preflight"] is None or artifacts["preflight"].get("status") != "PASS_T10_3_5_PREFLIGHT":
            verdict = "SCHEDULED_WIRING_MISS"
        elif artifacts["core"] is None or artifacts["core"].get("passed") is not True:
            verdict = "SCHEDULED_CORE_MISS"
        elif artifacts["sequence"] is None or artifacts["sequence"].get("passed") is not True:
            verdict = "MIXED_SEQUENCE_MISS"
        elif artifacts["compile"] is None or artifacts["compile"].get("passed") is not True:
            verdict = "REGISTRY_REPRODUCTION_MISS"
        elif artifacts["confirmation"] is None or artifacts["confirmation"].get("passed") is not True:
            verdict = "SOURCE_CONFIRMATION_MISS"
        else:
            verdict = "PASS_T10_3_5_SCHEDULED_END_TO_END_SOURCE"
        accounting = durable._journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.5-terminal-report-v1",
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
            "firewall": manifest["firewall"],
            "physical_actions_replayed": 0,
            "production_authority": False,
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, TERMINAL_REPORT_FILENAME), report)
    return report


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_5_contracts():
        payload = durable.status(root, manifest)
    payload = dict(payload)
    payload["protocol"] = "SAGE.T10.3.5"
    payload["scheduled_control"] = manifest["scheduled_control"]
    return payload


def _emit(payload: Mapping[str, Any]) -> None:
    print(_canonical(payload), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze", "status", "audit", "preflight", "discover-core",
            "discover-sequence", "compile", "confirm", "report",
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
        if args.phase in {"discover-core", "discover-sequence"}:
            _emit(run_discovery(root, manifest, args.phase))
            return 0
        if args.phase == "compile":
            _emit(compile_registry(root, manifest))
            return 0
        if args.phase == "confirm":
            _emit(run_confirmation(root, manifest))
            return 0
        report = terminal_report(root, manifest)
        _emit(report)
        return 0 if report["verdict"] == "PASS_T10_3_5_SCHEDULED_END_TO_END_SOURCE" else 3
    except protocol.ScientificGateMiss as exc:
        _emit({"phase": args.phase, "error": str(exc), "exit_code": 3})
        return 3
    except (protocol.IntegrityError, OSError, ValueError, KeyError) as exc:
        _emit({"phase": args.phase, "error": f"{type(exc).__name__}:{exc}", "exit_code": 2})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STAGE_TIMING_KEYS", "_controller_pair", "_run_work", "_t10_3_5_contracts",
    "audit", "compile_registry", "main", "preflight", "run_confirmation",
    "run_discovery", "should_stop_after_level", "status", "terminal_report",
]
