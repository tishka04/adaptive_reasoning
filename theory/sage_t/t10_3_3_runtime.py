"""Durable runtime for the T10.3.3 relational-binding continuation."""

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
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import t10_3_2_runtime as durable
from . import t10_3_3_protocol as protocol
from .goal_directed_v10_3_2 import (
    GoalDirectedOption,
    OptionStep,
    ProgressProgramRegistry,
)
from .goal_directed_v10_3_3 import (
    BRANCH_PRODUCTIVE_ANCHOR,
    RelationalGoalDirectedSageTController,
)

AUDIT_FILENAME = durable.AUDIT_FILENAME
PREFLIGHT_FILENAME = durable.PREFLIGHT_FILENAME
CORE_REPORT_FILENAME = durable.CORE_REPORT_FILENAME
SEQUENCE_REPORT_FILENAME = durable.SEQUENCE_REPORT_FILENAME
COMPILE_REPORT_FILENAME = durable.COMPILE_REPORT_FILENAME
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


@contextmanager
def _t10_3_3_contracts() -> Iterator[None]:
    """Route the proven T10.3.2 durability shell through new frozen contracts."""

    old_protocol = durable.protocol
    old_controller = durable.GoalDirectedSageTController
    old_run_work = durable._run_work
    durable.protocol = protocol
    durable.GoalDirectedSageTController = RelationalGoalDirectedSageTController
    durable._run_work = _run_work
    try:
        yield
    finally:
        durable._run_work = old_run_work
        durable.GoalDirectedSageTController = old_controller
        durable.protocol = old_protocol


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
    receipt_path = durable._receipt_for_work(destination, work)
    if receipt_path.is_file():
        return durable._read_signed(receipt_path, "receipt_checksum")
    controller, goal = durable._controller_pair(
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
    decision_latencies: list[float] = []
    initial_level = final_level = 0
    level_event_sources: list[str] = []
    winning_registry_checksums: list[str | None] = []
    sage_t_actions = 0
    status = "COMPLETE"
    env = None
    try:
        env = live._make_real_env(work.game_id, root / "environment_files")
        frame = live._reset_env(env)
        initial = live.snapshot_frame(frame)
        initial_level = final_level = int(initial.levels_completed)
        for step_index in range(work.action_budget):
            before = live.snapshot_frame(frame)
            if live._is_terminal(before.game_state):
                break
            legal_actions = tuple(live._valid_actions(env))
            proposal = policy.select(legal_actions)
            if proposal is None:
                errors.append("NO_LEGAL_ACTION")
                status = "ABORTED"
                break
            started = time.perf_counter()
            try:
                decision = controller.select_action(
                    current_grid=before.grid,
                    available_actions=live._available_action_names(legal_actions),
                    legacy_action=str(getattr(proposal, "name", "")),
                    legacy_action_data=dict(
                        getattr(proposal, "action_args", {}) or {}
                    ),
                    available_action_candidates=legal_actions,
                    game_state=before.game_state,
                    levels_completed=before.levels_completed,
                )
            except Exception as exc:  # noqa: BLE001 - controller boundary
                errors.append(f"CONTROLLER_SELECT:{type(exc).__name__}")
                status = "ABORTED"
                break
            decision_latencies.append((time.perf_counter() - started) * 1000.0)
            selected = live._materialize_decision(legal_actions, decision)
            if selected is None:
                illegal_actions += 1
                errors.append("UNAVAILABLE_DECISION")
                status = "ABORTED"
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
            try:
                after_frame = live._step_env_action(env, selected)
                after = live.snapshot_frame(
                    after_frame,
                    fallback_available_actions=before.available_actions,
                )
            except Exception as exc:  # noqa: BLE001 - SDK boundary
                unresolved = _signed(
                    {
                        "format_version": "sage-t10.3.3-unresolved-event-v1",
                        "manifest_checksum": manifest["manifest_checksum"],
                        "work_id": work.work_id,
                        "event_id": intent["event_id"],
                        "step_index": step_index,
                        "reason": (
                            "ENVIRONMENT_CALL_UNATTESTABLE:"
                            f"{type(exc).__name__}"
                        ),
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
                break
            level_delta = max(
                0, int(after.levels_completed) - int(before.levels_completed)
            )
            event = _signed(
                {
                    "format_version": "sage-t10.3.3-physical-event-v1",
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
                    "frame_before_sha256": protocol.sha256_payload(
                        before.grid.tolist()
                    ),
                    "frame_after_sha256": protocol.sha256_payload(
                        after.grid.tolist()
                    ),
                    "raw_frame_retained": False,
                    "physical_action_replayed": False,
                },
                "event_checksum",
            )
            protocol.write_json_once(
                durable._work_path(destination, "events", work, name), event
            )
            trace_ids.append(str(intent["event_id"]))
            if str(decision.source) == "sage_t_joint_program":
                sage_t_actions += 1
            if level_delta > 0:
                level_event_sources.append(str(decision.source))
                winning_registry_checksums.append(intent["registry_checksum"])
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
                if level_delta > 0 and not live._is_terminal(after.game_state):
                    controller.on_level_change()
            except Exception as exc:  # noqa: BLE001 - posterior boundary
                errors.append(f"CONTROLLER_OBSERVE:{type(exc).__name__}")
                status = "ABORTED"
                frame = after_frame
                final_level = int(after.levels_completed)
                break
            durable._write_checkpoint(
                destination,
                manifest,
                phase=work.phase,
                work=work,
                state="EVENT_SEALED_AND_POSTERIOR_UPDATED",
                registry=registry,
            )
            frame = after_frame
            final_level = int(after.levels_completed)
            if str(after.game_state).upper() == "GAME_OVER":
                game_over_actions += 1
            lock.heartbeat()
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
                    }
                ),
                flush=True,
            )
            if live._is_terminal(after.game_state):
                break
    except Exception as exc:  # noqa: BLE001 - durable receipt must be sealed
        errors.append(f"RUNTIME:{type(exc).__name__}")
        status = "ABORTED"
    finally:
        if env is not None:
            try:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            except Exception as exc:  # noqa: BLE001 - external cleanup
                errors.append(f"ENVIRONMENT_CLOSE:{type(exc).__name__}")
    summary = goal.summary() if goal is not None else {}
    intent_dir = durable._work_path(destination, "intents", work, "x").parent
    event_dir = durable._work_path(destination, "events", work, "x").parent
    unresolved_dir = durable._work_path(destination, "unresolved", work, "x").parent
    issued_intents = len(tuple(intent_dir.glob("*.json"))) if intent_dir.exists() else 0
    sealed_events = len(tuple(event_dir.glob("*.json"))) if event_dir.exists() else 0
    unresolved_intents = (
        len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    )
    receipt = _signed(
        {
            "format_version": "sage-t10.3.3-branch-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            **work.as_dict(),
            "work_id": work.work_id,
            "status": status,
            "complete": status == "COMPLETE" and not errors,
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
                    for items in summary.get(
                        "successful_option_action_schemas", ()
                    )
                )
            ),
            "binding_rejections": dict(summary.get("binding_rejections", {})),
            "binding_method_uses": dict(summary.get("binding_method_uses", {})),
            "structural_collision_count": int(
                summary.get("structural_collision_count", 0)
            ),
            "proposal_reacquisitions": int(
                summary.get("proposal_reacquisitions", 0)
            ),
            "ephemeral_action_data_persisted": False,
            "errors": errors,
            "illegal_actions": illegal_actions,
            "game_over_actions": game_over_actions,
            "decision_latencies_ms": decision_latencies,
            "registry_checksum_loaded": registry_checksum,
            "registry_used_in_decision": bool(
                summary.get("registry_used_in_decision")
            ),
            "controller_registry": (
                goal.registry.snapshot() if goal is not None else None
            ),
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


def _unified_controller(
    sage_t: RelationalGoalDirectedSageTController,
) -> UnifiedCognitiveController:
    controller = UnifiedCognitiveController(
        "synthetic-source",
        config=UnifiedCognitiveConfig(
            sage_t_authority_mode="active",
            sage_t_counterfactual_gate_passed=True,
            sage_t_active_gate_passed=True,
        ),
        sage_t_controller=sage_t,
    )
    controller.on_reset()
    return controller


def _run_fixed_option(option: GoalDirectedOption, success_step: int) -> dict[str, Any]:
    sage_t = RelationalGoalDirectedSageTController(
        phase="preflight",
        warmup_actions=0,
        attestation_scope=f"synthetic-{option.schema}",
    )
    controller = _unified_controller(sage_t)
    sage_t._active_option = option
    names = tuple(sorted(set(option.action_schemas)))
    legal = tuple(_SyntheticAction(name, {}) for name in names)
    grid = np.zeros((5, 5), dtype=np.int16)
    sources = []
    for index, expected in enumerate(option.steps[: success_step + 1]):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=names,
            legacy_action=names[0],
            legacy_action_data={},
            available_action_candidates=legal,
            game_state="NOT_FINISHED",
            levels_completed=0,
        )
        sources.append(
            decision.source == "sage_t_joint_program"
            and decision.action_name == expected.action_name
        )
        grid = before.copy()
        grid[2, 2] = 1 + ((index + 1) % 2)
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=grid,
            available_actions=names,
            levels_completed_before=0,
            levels_completed_after=int(index == success_step),
        )
    return {
        "schema": option.schema,
        "actions": success_step + 1,
        "level_progress": sage_t.summary()["option_successes"] >= 1,
        "same_controller_closed_loop": all(sources),
        "posterior_observed_transitions": controller.summary()[
            "transitions_observed"
        ],
    }


def _run_ambiguous_target_preflight() -> dict[str, Any]:
    sage_t = RelationalGoalDirectedSageTController(
        phase="preflight",
        warmup_actions=0,
        exploration_interval=1,
        attestation_scope="synthetic-ambiguous-repeat",
    )
    controller = _unified_controller(sage_t)
    legal = (
        _SyntheticAction("ACTION6", {"x": 1, "y": 1}),
        _SyntheticAction("ACTION6", {"x": 2, "y": 1}),
    )
    grid = np.zeros((7, 7), dtype=np.int16)
    grid[1, 1] = 2
    grid[1, 2] = 2
    target = dict(legal[0].action_args)
    sage_sources = 0
    selected_target = []
    for index in range(5):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=("ACTION6",),
            legacy_action="ACTION6",
            legacy_action_data=target,
            available_action_candidates=legal,
            game_state="NOT_FINISHED",
            levels_completed=0,
        )
        if decision.source == "sage_t_joint_program":
            sage_sources += 1
            selected_target.append(dict(decision.action_data) == target)
        grid = before.copy()
        grid[1, 1] = 2 + ((index + 1) % 2)
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=grid,
            available_actions=("ACTION6",),
            levels_completed_before=0,
            levels_completed_after=int(index == 4),
        )
    summary = sage_t.summary()
    encoded_registry = _canonical(summary["registry"])
    return {
        "schema": "ambiguous_repeat_target",
        "actions": 5,
        "level_progress": summary["option_successes"] >= 1,
        "sage_t_actions": sage_sources,
        "same_target_reacquired": bool(selected_target) and all(selected_target),
        "branch_anchor_used": int(
            summary["binding_method_uses"].get(BRANCH_PRODUCTIVE_ANCHOR, 0)
        )
        >= 1,
        "structural_collision_observed": int(
            summary["structural_collision_count"]
        )
        >= 2,
        "persistent_coordinates_absent": all(
            token not in encoded_registry
            for token in ('"x"', '"y"', "raw_grid", "entity_id")
        ),
        "posterior_observed_transitions": controller.summary()[
            "transitions_observed"
        ],
    }


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_snapshot = manifest["superseded_t10_3_2"]
    checks = {
        "parent_snapshot_bound": all(
            parent_snapshot.get(key) == value
            for key, value in protocol.SUPERSEDED_T10_3_2.items()
        ),
        "parent_training_forbidden": parent_snapshot.get("used_for_training")
        is False,
        "parent_mutation_forbidden": parent_snapshot.get("mutated_by_t10_3_3")
        is False,
        "parent_replay_forbidden": parent_snapshot.get("physical_actions_replayed")
        == 0,
        "binding_ambiguity_attested": (
            parent_snapshot.get("sage_t_decision_count") == 0
            and parent_snapshot.get("candidate_success_count") == 0
            and parent_snapshot.get("candidate_contradiction_count") == 8
        ),
        "source_only_firewall": not any(manifest["firewall"].values()),
        "budget_exact": manifest["matrix"]["total_maximum_actions"] == 6144,
        "reset_exact": manifest["matrix"]["total_resets"] == 30,
        "ephemeral_binding_only": manifest["binding_recovery"][
            "persistent_coordinates"
        ]
        is False,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.3-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "parent_events_used_for_training": 0,
            "physical_actions": 0,
            "status": (
                "PASS_T10_3_3_OFFLINE_AUDIT"
                if all(checks.values())
                else "INVALID_PROVENANCE"
            ),
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.3 provenance audit failed")
    return payload


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    ambiguous = _run_ambiguous_target_preflight()
    path = _run_fixed_option(
        GoalDirectedOption(
            schema="path_successor",
            steps=tuple(OptionStep("ACTION1") for _ in range(10)),
            source="synthetic_path",
        ),
        9,
    )
    mixed = _run_fixed_option(
        GoalDirectedOption(
            schema="mixed_automaton",
            steps=tuple(
                OptionStep("ACTION1" if index % 2 == 0 else "ACTION2")
                for index in range(20)
            ),
            source="synthetic_mixed",
        ),
        19,
    )
    scenarios = [ambiguous, path, mixed]
    checks = {
        "ambiguous_target_progress": ambiguous["level_progress"],
        "ambiguous_target_sage_authority": ambiguous["sage_t_actions"] >= 1,
        "same_target_reacquired": ambiguous["same_target_reacquired"],
        "branch_anchor_used": ambiguous["branch_anchor_used"],
        "structural_collision_observed": ambiguous[
            "structural_collision_observed"
        ],
        "persistent_coordinates_absent": ambiguous[
            "persistent_coordinates_absent"
        ],
        "path_length_10": path["actions"] == 10 and path["level_progress"],
        "mixed_beyond_16": mixed["actions"] > 16 and mixed["level_progress"],
        "same_controller_closed_loop": path["same_controller_closed_loop"]
        and mixed["same_controller_closed_loop"],
        "posterior_updated_each_action": all(
            row["posterior_observed_transitions"] == row["actions"]
            for row in scenarios
        ),
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.3-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": scenarios,
            "checks": checks,
            "physical_actions": 0,
            "status": (
                "PASS_T10_3_3_PREFLIGHT" if all(checks.values()) else "WIRING_MISS"
            ),
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.3 synthetic binding recovery failed")
    return payload


def run_discovery(
    root: Path, manifest: Mapping[str, Any], phase: str
) -> dict[str, Any]:
    with _t10_3_3_contracts():
        durable._require_live_runtime()
        if phase == "discover-sequence":
            core = durable._read_signed(
                _artifact_path(root, CORE_REPORT_FILENAME), "report_checksum"
            )
            if core.get("passed") is not True:
                raise protocol.ScientificGateMiss(
                    "core gate forbids sequence discovery"
                )
        destination = _destination(root)
        durable._recover_orphans(destination, manifest)
        registry = durable._registry_from_receipts(destination)
        lock = durable._CollectorLock(destination / LOCK_FILENAME, phase)
        lock.acquire()
        try:
            for work in protocol.work_specs(phase):
                _run_work(
                    root,
                    destination,
                    manifest,
                    work,
                    registry,
                    lock,
                    registry_checksum=None,
                )
        finally:
            lock.release()
        controls = durable._apply_registry_controls(registry)
        receipts = durable._load_receipts(destination, phase)
        parent_report = durable._phase_report(
            manifest, phase, receipts, registry, controls
        )
        report_core = {
            key: value
            for key, value in parent_report.items()
            if key != "report_checksum"
        }
        report_core["format_version"] = f"sage-t10.3.3-{phase}-report-v1"
        if phase == "discover-core":
            lp85_rows = [
                row for row in receipts if str(row.get("game_id", "")).startswith("lp85")
            ]
            binding_checks = {
                "sage_t_action_each_reset": all(
                    int(row.get("sage_t_option_actions", 0)) >= 1
                    for row in receipts
                ),
                "explicit_binding_telemetry": all(
                    isinstance(row.get("binding_rejections"), Mapping)
                    and isinstance(row.get("binding_method_uses"), Mapping)
                    for row in receipts
                ),
                "lp85_collision_observed": bool(lp85_rows)
                and all(
                    int(row.get("structural_collision_count", 0)) >= 2
                    for row in lp85_rows
                ),
                "lp85_ephemeral_anchor_used": bool(lp85_rows)
                and all(
                    int(
                        row.get("binding_method_uses", {}).get(
                            BRANCH_PRODUCTIVE_ANCHOR, 0
                        )
                    )
                    >= 1
                    for row in lp85_rows
                ),
                "ephemeral_action_data_absent": all(
                    row.get("ephemeral_action_data_persisted") is False
                    for row in receipts
                ),
            }
            report_core["checks"] = {
                **dict(report_core.get("checks", {})),
                **binding_checks,
            }
            report_core["passed"] = bool(
                parent_report.get("passed") and all(binding_checks.values())
            )
            report_core["verdict"] = (
                "PASS_T10_3_3_CORE_BINDING_RECOVERY"
                if report_core["passed"]
                else (
                    "CORE_PROGRESS_MISS"
                    if not parent_report.get("passed")
                    else "BINDING_RECOVERY_MISS"
                )
            )
        else:
            report_core["verdict"] = (
                "PASS_T10_3_3_SEQUENCE_DISCOVERY"
                if parent_report.get("passed")
                else "MIXED_SEQUENCE_MISS"
            )
        report = _signed(report_core, "report_checksum")
        report_name = (
            CORE_REPORT_FILENAME
            if phase == "discover-core"
            else SEQUENCE_REPORT_FILENAME
        )
        registry_name = (
            durable.CORE_REGISTRY_FILENAME
            if phase == "discover-core"
            else durable.SEQUENCE_REGISTRY_FILENAME
        )
        protocol.write_json_once(_artifact_path(root, report_name), report)
        protocol.write_json_once(
            _artifact_path(root, registry_name), registry.snapshot()
        )
        if report.get("passed") is not True:
            raise protocol.ScientificGateMiss(str(report["verdict"]))
        return report


def compile_registry(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_3_contracts():
        core = durable._read_signed(
            _artifact_path(root, CORE_REPORT_FILENAME), "report_checksum"
        )
        sequence = durable._read_signed(
            _artifact_path(root, SEQUENCE_REPORT_FILENAME), "report_checksum"
        )
        if core.get("passed") is not True or sequence.get("passed") is not True:
            raise protocol.ScientificGateMiss(
                "discovery gates forbid registry compilation"
            )
        source = durable._read_signed(
            _artifact_path(root, durable.SEQUENCE_REGISTRY_FILENAME),
            "registry_checksum",
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
                len(
                    {
                        step["action_name"]
                        for step in row["program"]["dynamics"]
                    }
                )
                >= 2
                for row in programs
            ),
            "coordinate_free": all(
                token not in _canonical(compiled)
                for token in ('"x"', '"y"', "raw_grid", "entity_id")
            ),
        }
        protocol.write_json_once(
            _artifact_path(root, durable.COMPILED_REGISTRY_FILENAME), compiled
        )
        report = _signed(
            {
                "format_version": "sage-t10.3.3-compile-report-v1",
                "manifest_checksum": manifest["manifest_checksum"],
                "registry_checksum": compiled["registry_checksum"],
                "program_count": len(programs),
                "controls": controls,
                "checks": checks,
                "passed": all(checks.values()),
                "verdict": (
                    "PASS_T10_3_3_REGISTRY"
                    if all(checks.values())
                    else "REGISTRY_REPRODUCTION_MISS"
                ),
            },
            "report_checksum",
        )
        protocol.write_json_once(
            _artifact_path(root, COMPILE_REPORT_FILENAME), report
        )
        if report.get("passed") is not True:
            raise protocol.ScientificGateMiss(str(report["verdict"]))
        return report


def run_confirmation(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_3_contracts():
        durable._require_live_runtime()
        compile_report = durable._read_signed(
            _artifact_path(root, COMPILE_REPORT_FILENAME), "report_checksum"
        )
        if compile_report.get("passed") is not True:
            raise protocol.ScientificGateMiss(
                "compiled registry gate forbids confirmation"
            )
        compiled = durable._read_signed(
            _artifact_path(root, durable.COMPILED_REGISTRY_FILENAME),
            "registry_checksum",
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
                    root,
                    destination,
                    manifest,
                    work,
                    registry,
                    lock,
                    registry_checksum=(
                        registry_checksum
                        if work.arm == "goal_directed_sage_t"
                        else None
                    ),
                )
        finally:
            lock.release()
        receipts = durable._load_receipts(destination, "confirm")
        by_key = {
            (str(row["game_id"]), int(row["seed"]), str(row["arm"])): row
            for row in receipts
        }
        active_levels = {}
        baseline_levels = {}
        for game in protocol.ALL_SOURCE_GAMES:
            active_levels[game] = sum(
                int(
                    by_key.get(
                        (game, seed, "goal_directed_sage_t"), {}
                    ).get("level_delta", 0)
                )
                for seed in protocol.CONFIRMATION_SEEDS
            )
            baseline_levels[game] = sum(
                int(
                    by_key.get(
                        (game, seed, "unified_sage_t_off"), {}
                    ).get("level_delta", 0)
                )
                for seed in protocol.CONFIRMATION_SEEDS
            )
        active_rows = [
            row for row in receipts if row["arm"] == "goal_directed_sage_t"
        ]
        baseline_rows = [
            row for row in receipts if row["arm"] == "unified_sage_t_off"
        ]
        latencies = sorted(
            float(value)
            for row in active_rows
            for value in row.get("decision_latencies_ms", ())
        )
        p95 = (
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            if latencies
            else 0.0
        )
        checks = {
            "all_conditions_present": len(receipts) == 20,
            "core_each_seed": all(
                int(
                    by_key.get(
                        (game, seed, "goal_directed_sage_t"), {}
                    ).get("level_delta", 0)
                )
                >= 1
                for game in protocol.CORE_GAMES
                for seed in protocol.CONFIRMATION_SEEDS
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
            "zero_errors": all(not row.get("errors") for row in receipts),
            "zero_illegal_actions": all(
                int(row.get("illegal_actions", 0)) == 0 for row in receipts
            ),
            "game_over_nonincrease": sum(
                int(row.get("game_over_actions", 0)) for row in active_rows
            )
            <= sum(
                int(row.get("game_over_actions", 0)) for row in baseline_rows
            ),
            "registry_loaded": all(
                row.get("registry_checksum_loaded") == registry_checksum
                for row in active_rows
            ),
            "registry_used": all(
                bool(row.get("registry_used_in_decision")) for row in active_rows
            ),
            "winning_decisions_attest_registry": all(
                all(
                    checksum == registry_checksum
                    for checksum in row.get("winning_registry_checksums", ())
                )
                and len(row.get("winning_registry_checksums", ()))
                == len(row.get("level_event_sources", ()))
                for row in active_rows
            ),
            "sage_t_action_each_active_reset": all(
                int(row.get("sage_t_option_actions", 0)) >= 1
                for row in active_rows
            ),
            "ephemeral_action_data_absent": all(
                row.get("ephemeral_action_data_persisted") is False
                for row in active_rows
            ),
            "decision_p95": p95 <= 2500.0,
            "budget": sum(
                int(row.get("issued_intents", 0)) for row in receipts
            )
            <= protocol.CONFIRMATION_ACTIONS,
            "intent_accounting": all(
                int(row.get("issued_intents", 0))
                == int(row.get("sealed_events", 0))
                + int(row.get("unresolved_intents", 0))
                for row in receipts
            ),
            "zero_physical_replay": all(
                int(row.get("physical_actions_replayed", 0)) == 0
                for row in receipts
            ),
        }
        report = _signed(
            {
                "format_version": "sage-t10.3.3-confirmation-report-v1",
                "manifest_checksum": manifest["manifest_checksum"],
                "registry_checksum": registry_checksum,
                "metrics": {
                    "active_levels": active_levels,
                    "baseline_levels": baseline_levels,
                    "active_total_levels": sum(active_levels.values()),
                    "baseline_total_levels": sum(baseline_levels.values()),
                    "decision_p95_ms": p95,
                    "actions": sum(
                        int(row.get("issued_intents", 0)) for row in receipts
                    ),
                },
                "checks": checks,
                "passed": all(checks.values()),
                "verdict": (
                    "PASS_T10_3_3_SOURCE_CONFIRMATION"
                    if all(checks.values())
                    else "SOURCE_CONFIRMATION_MISS"
                ),
            },
            "report_checksum",
        )
        protocol.write_json_once(
            _artifact_path(root, CONFIRMATION_REPORT_FILENAME), report
        )
        if report.get("passed") is not True:
            raise protocol.ScientificGateMiss(str(report["verdict"]))
        return report


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_3_contracts():
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
            artifacts[name] = (
                durable._read_signed(path, checksum) if path.is_file() else None
            )
        if (
            artifacts["audit"] is None
            or artifacts["audit"].get("status")
            != "PASS_T10_3_3_OFFLINE_AUDIT"
        ):
            verdict = "INVALID_PROVENANCE"
        elif (
            artifacts["preflight"] is None
            or artifacts["preflight"].get("status")
            != "PASS_T10_3_3_PREFLIGHT"
        ):
            verdict = "WIRING_MISS"
        elif artifacts["core"] is None or artifacts["core"].get("passed") is not True:
            verdict = "BINDING_RECOVERY_MISS"
        elif (
            artifacts["sequence"] is None
            or artifacts["sequence"].get("passed") is not True
        ):
            verdict = "MIXED_SEQUENCE_MISS"
        elif (
            artifacts["compile"] is None
            or artifacts["compile"].get("passed") is not True
        ):
            verdict = "REGISTRY_REPRODUCTION_MISS"
        elif (
            artifacts["confirmation"] is None
            or artifacts["confirmation"].get("passed") is not True
        ):
            verdict = "SOURCE_CONFIRMATION_MISS"
        else:
            verdict = "PASS_T10_3_3_RELATIONAL_BINDING_SOURCE"
        accounting = durable._journal_accounting(_destination(root))
        report = _signed(
            {
                "format_version": "sage-t10.3.3-terminal-report-v1",
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
                "firewall": manifest["firewall"],
                "physical_actions_replayed": 0,
                "production_authority": False,
            },
            "report_checksum",
        )
        protocol.write_json_once(
            _artifact_path(root, TERMINAL_REPORT_FILENAME), report
        )
        return report


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _t10_3_3_contracts():
        payload = durable.status(root, manifest)
    payload = dict(payload)
    payload["protocol"] = "SAGE.T10.3.3"
    return payload


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
            "discover-core",
            "discover-sequence",
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
        return (
            0
            if report["verdict"]
            == "PASS_T10_3_3_RELATIONAL_BINDING_SOURCE"
            else 3
        )
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
    "main",
    "preflight",
    "run_confirmation",
    "run_discovery",
    "status",
    "terminal_report",
]
