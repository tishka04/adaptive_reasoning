"""Durable closed-loop runtime for SAGE.T10.3.2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from theory import unified_cognition_ab_benchmark as live
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import t10_3_2_protocol as protocol
from .goal_directed_v10_3_2 import (
    GoalDirectedOption,
    GoalDirectedSageTController,
    OptionStep,
    ProgressProgramRegistry,
)

AUDIT_FILENAME = "offline_audit.json"
PREFLIGHT_FILENAME = "synthetic_preflight.json"
CORE_REPORT_FILENAME = "discovery_core_report.json"
CORE_REGISTRY_FILENAME = "core_registry_candidates.json"
SEQUENCE_REPORT_FILENAME = "discovery_sequence_report.json"
SEQUENCE_REGISTRY_FILENAME = "sequence_registry_candidates.json"
COMPILED_REGISTRY_FILENAME = "compiled_progress_registry.json"
COMPILE_REPORT_FILENAME = "compile_report.json"
CONFIRMATION_REPORT_FILENAME = "confirmation_report.json"
TERMINAL_REPORT_FILENAME = "terminal_report.json"
CHECKPOINT_FILENAME = "checkpoint.json"
LOCK_FILENAME = "collector.lock.json"
LOCK_HEARTBEAT_TIMEOUT_SECONDS = 900.0


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = protocol.sha256_payload(result)
    return result


def _read_signed(path: Path, checksum_field: str) -> dict[str, Any]:
    if not path.is_file():
        raise protocol.IntegrityError(f"required artifact is absent: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol.verify_signed(payload, checksum_field)
    return payload


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _destination(root: Path) -> Path:
    return root.resolve() / protocol.DEFAULT_OUTPUT_DIR


def _artifact_path(root: Path, name: str) -> Path:
    return _destination(root) / name


def _work_path(destination: Path, category: str, work: protocol.WorkSpec, name: str) -> Path:
    return destination / "journal" / category / work.work_id / name


def _all_files(destination: Path, category: str) -> tuple[Path, ...]:
    directory = destination / "journal" / category
    return tuple(sorted(directory.rglob("*.json"))) if directory.exists() else ()


def _lock_path(destination: Path) -> Path:
    return destination / LOCK_FILENAME


def _lock_live(payload: Mapping[str, Any]) -> bool:
    try:
        age = time.time() - float(payload["heartbeat"])
        return age <= LOCK_HEARTBEAT_TIMEOUT_SECONDS
    except (KeyError, TypeError, ValueError):
        return False


@dataclass
class _CollectorLock:
    path: Path
    phase: str
    nonce: str = ""

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if _lock_live(existing):
                raise protocol.IntegrityError("another T10.3.2 collector is active")
        self.nonce = uuid.uuid4().hex
        payload = {
            "format_version": "sage-t10.3.2-collector-lock-v1",
            "pid": os.getpid(),
            "process_start": time.time(),
            "nonce": self.nonce,
            "phase": self.phase,
            "heartbeat": time.time(),
        }
        encoded = _canonical(payload) + "\n"
        try:
            with self.path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
        except FileExistsError:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if _lock_live(existing):
                raise protocol.IntegrityError(
                    "another T10.3.2 collector is active"
                ) from None
            _write_atomic(self.path, payload)

    def heartbeat(self) -> None:
        if not self.path.is_file():
            raise protocol.IntegrityError("collector lock disappeared")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("nonce") != self.nonce:
            raise protocol.IntegrityError("collector lock ownership changed")
        payload["heartbeat"] = time.time()
        _write_atomic(self.path, payload)

    def release(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("nonce") == self.nonce:
            self.path.unlink()


def _journal_accounting(destination: Path) -> dict[str, Any]:
    intents = _all_files(destination, "intents")
    events = _all_files(destination, "events")
    unresolved = _all_files(destination, "unresolved")
    intent_ids = {path.relative_to(destination / "journal" / "intents").as_posix() for path in intents}
    event_ids = {path.relative_to(destination / "journal" / "events").as_posix() for path in events}
    unresolved_ids = {
        path.relative_to(destination / "journal" / "unresolved").as_posix()
        for path in unresolved
    }
    inflight = sorted(intent_ids - event_ids - unresolved_ids)
    lock_payload = None
    lock = _lock_path(destination)
    if lock.is_file():
        try:
            lock_payload = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            lock_payload = None
    live_lock = bool(lock_payload and _lock_live(lock_payload))
    equation = len(intents) == len(events) + len(unresolved) + len(inflight)
    valid_inflight = len(inflight) <= 1 and (not inflight or live_lock)
    started_work_ids = {path.parent.name for path in intents}
    completed_work_ids = {path.parent.name for path in _all_files(destination, "branches")}
    incomplete_work_ids = sorted(started_work_ids - completed_work_ids)
    return {
        "authorized_actions": len(intents),
        "sealed_events": len(events),
        "unresolved_intents": len(unresolved),
        "inflight_intents": len(inflight),
        "inflight_paths": inflight,
        "live_collector_lock": live_lock,
        "equation_holds": equation,
        "inflight_valid": valid_inflight,
        "incomplete_work_ids": incomplete_work_ids,
    }


def _recover_orphans(destination: Path, manifest: Mapping[str, Any]) -> int:
    accounting = _journal_accounting(destination)
    if accounting["live_collector_lock"]:
        raise protocol.IntegrityError("cannot recover while a collector lock is live")
    recovered = 0
    for relative in accounting["inflight_paths"]:
        intent_path = destination / "journal" / "intents" / relative
        unresolved_path = destination / "journal" / "unresolved" / relative
        intent = _read_signed(intent_path, "intent_checksum")
        payload = _signed(
            {
                "format_version": "sage-t10.3.2-unresolved-event-v1",
                "manifest_checksum": manifest["manifest_checksum"],
                "work_id": intent["work_id"],
                "event_id": intent["event_id"],
                "step_index": intent["step_index"],
                "reason": "PROCESS_INTERRUPTION",
                "physical_action_replayed": False,
            },
            "unresolved_checksum",
        )
        protocol.write_json_once(unresolved_path, payload)
        recovered += 1
    checkpoint = None
    checkpoint_path = destination / CHECKPOINT_FILENAME
    if checkpoint_path.is_file():
        checkpoint = _read_signed(checkpoint_path, "checkpoint_checksum")
    intents_root = destination / "journal" / "intents"
    if intents_root.exists():
        for work_dir in sorted(path for path in intents_root.iterdir() if path.is_dir()):
            intent_paths = tuple(sorted(work_dir.glob("*.json")))
            if not intent_paths:
                continue
            first_intent = _read_signed(intent_paths[0], "intent_checksum")
            work_payload = first_intent.get("work")
            if not isinstance(work_payload, Mapping):
                raise protocol.IntegrityError(
                    "interrupted intent is missing its preregistered work receipt"
                )
            work = protocol.WorkSpec(
                phase=str(work_payload["phase"]),
                game_id=str(work_payload["game_id"]),
                seed=int(work_payload["seed"]),
                arm=str(work_payload["arm"]),
                reset_index=int(work_payload["reset_index"]),
                action_budget=int(work_payload["action_budget"]),
            )
            if work.work_id != work_dir.name or first_intent.get("work_id") != work.work_id:
                raise protocol.IntegrityError("interrupted work identity drifted")
            receipt_path = _receipt_for_work(destination, work)
            if receipt_path.is_file():
                continue
            events_dir = destination / "journal" / "events" / work.work_id
            unresolved_dir = destination / "journal" / "unresolved" / work.work_id
            event_paths = (
                tuple(sorted(events_dir.glob("*.json"))) if events_dir.exists() else ()
            )
            unresolved_paths = (
                tuple(sorted(unresolved_dir.glob("*.json")))
                if unresolved_dir.exists()
                else ()
            )
            events = [_read_signed(path, "event_checksum") for path in event_paths]
            intents = [_read_signed(path, "intent_checksum") for path in intent_paths]
            controller_registry = None
            if checkpoint is not None and checkpoint.get("current_work_id") == work.work_id:
                snapshot = checkpoint.get("controller_registry")
                if isinstance(snapshot, Mapping):
                    controller_registry = dict(snapshot)
            winning_checksums = []
            intents_by_event = {str(row["event_id"]): row for row in intents}
            for event in events:
                if int(event.get("level_delta", 0)) > 0:
                    checksum = intents_by_event.get(str(event["event_id"]), {}).get(
                        "registry_checksum"
                    )
                    winning_checksums.append(checksum)
            receipt = _signed(
                {
                    "format_version": "sage-t10.3.2-branch-receipt-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    **work.as_dict(),
                    "work_id": work.work_id,
                    "status": "ABORTED_PROCESS_INTERRUPTION",
                    "complete": False,
                    "issued_intents": len(intent_paths),
                    "sealed_events": len(event_paths),
                    "unresolved_intents": len(unresolved_paths),
                    "event_ids": [str(row["event_id"]) for row in events],
                    "level_delta": sum(int(row.get("level_delta", 0)) for row in events),
                    "level_event_sources": [
                        str(row.get("decision_source", ""))
                        for row in events
                        if int(row.get("level_delta", 0)) > 0
                    ],
                    "winning_registry_checksums": winning_checksums,
                    "sage_t_option_actions": sum(
                        str(row.get("decision_source", "")) == "sage_t_joint_program"
                        for row in events
                    ),
                    "mixed_program_used": False,
                    "errors": ["PROCESS_INTERRUPTION"],
                    "illegal_actions": 0,
                    "game_over_actions": sum(
                        str(row.get("game_state_after", "")).upper() == "GAME_OVER"
                        for row in events
                    ),
                    "decision_latencies_ms": [],
                    "registry_checksum_loaded": next(
                        (
                            row.get("registry_checksum")
                            for row in intents
                            if row.get("registry_checksum") is not None
                        ),
                        None,
                    ),
                    "registry_used_in_decision": any(
                        row.get("registry_checksum") is not None for row in intents
                    ),
                    "controller_registry": controller_registry,
                    "physical_actions_replayed": 0,
                },
                "receipt_checksum",
            )
            protocol.write_json_once(receipt_path, receipt)
    return recovered


def _receipt_paths(destination: Path) -> tuple[Path, ...]:
    return _all_files(destination, "branches")


def _receipt_for_work(destination: Path, work: protocol.WorkSpec) -> Path:
    return _work_path(destination, "branches", work, "receipt.json")


def _load_receipts(destination: Path, phase: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for path in _receipt_paths(destination):
        row = _read_signed(path, "receipt_checksum")
        if phase is None or row.get("phase") == phase:
            rows.append(row)
    return rows


def _registry_from_receipts(destination: Path) -> ProgressProgramRegistry:
    registry = ProgressProgramRegistry()
    for receipt in _load_receipts(destination):
        snapshot = receipt.get("controller_registry")
        if isinstance(snapshot, Mapping):
            registry.merge(snapshot)
    return registry


def _write_checkpoint(
    destination: Path,
    manifest: Mapping[str, Any],
    *,
    phase: str,
    work: protocol.WorkSpec | None,
    state: str,
    registry: ProgressProgramRegistry | None,
) -> dict[str, Any]:
    payload = _signed(
        {
            "format_version": "sage-t10.3.2-checkpoint-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": phase,
            "state": state,
            "current_work_id": None if work is None else work.work_id,
            "accounting": _journal_accounting(destination),
            "completed_resets": len(_receipt_paths(destination)),
            "maximum_resets": protocol.TOTAL_RESETS,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "controller_registry": (
                None if registry is None else registry.snapshot()
            ),
            "physical_actions_replayed": 0,
        },
        "checkpoint_checksum",
    )
    _write_atomic(destination / CHECKPOINT_FILENAME, payload)
    return payload


def _variant_binding_swap(option: GoalDirectedOption) -> GoalDirectedOption:
    steps = list(option.steps)
    first = steps[0]
    replacement = "ACTION1" if first.action_name != "ACTION1" else "ACTION2"
    steps[0] = OptionStep(
        action_name=replacement,
        binding_method="binding_swap_control",
        structural_signature=None,
        expected_effect="control",
    )
    return GoalDirectedOption(
        schema=("mixed_automaton" if len({item.action_name for item in steps}) >= 2 else option.schema),
        steps=tuple(steps),
        source="binding_swap_control",
    )


def _variant_order(option: GoalDirectedOption) -> GoalDirectedOption:
    steps = list(reversed(option.steps))
    if steps == list(option.steps):
        first = steps.pop(0)
        alternate = "ACTION1" if first.action_name != "ACTION1" else "ACTION2"
        steps.insert(
            max(1, len(steps) // 2),
            OptionStep(action_name=alternate, binding_method="order_control"),
        )
        steps = steps[: len(option.steps)]
    return GoalDirectedOption(
        schema="mixed_automaton" if len({item.action_name for item in steps}) >= 2 else option.schema,
        steps=tuple(steps),
        source="order_permutation_control",
    )


def _variant_ablation(option: GoalDirectedOption) -> GoalDirectedOption:
    if len(option.steps) > 1:
        steps = option.steps[:-1]
    else:
        alternate = "ACTION1" if option.steps[0].action_name != "ACTION1" else "ACTION2"
        steps = (OptionStep(action_name=alternate),)
    return GoalDirectedOption(
        schema=("mixed_automaton" if len({item.action_name for item in steps}) >= 2 else option.schema),
        steps=steps,
        source="automaton_ablation_control",
    )


def _apply_registry_controls(registry: ProgressProgramRegistry) -> dict[str, Any]:
    checked = []
    for option in registry.candidates():
        evidence = registry.evidence(option.option_id)
        if evidence is None or not evidence.reproducible:
            continue
        original_probability = (
            len(evidence.success_attestations) + 1.0
        ) / (
            len(evidence.success_attestations)
            + len(evidence.contradiction_attestations)
            + 2.0
        )
        variants = {
            "binding_swap": _variant_binding_swap(option),
            "order_permutation": _variant_order(option),
            "automaton_ablation": _variant_ablation(option),
        }
        outcomes = {}
        for name, variant in variants.items():
            variant_evidence = registry.evidence(variant.option_id)
            successes = 0 if variant_evidence is None else len(variant_evidence.success_attestations)
            contradictions = 0 if variant_evidence is None else len(variant_evidence.contradiction_attestations)
            probability = (successes + 1.0) / (successes + contradictions + 2.0)
            outcomes[name] = bool(
                variant.option_id != option.option_id
                and probability < original_probability
            )
        registry.note_controls(
            option.option_id,
            binding_swap=outcomes["binding_swap"],
            order_permutation=outcomes["order_permutation"],
            automaton_ablation=outcomes["automaton_ablation"],
        )
        checked.append(
            {
                "option_id": option.option_id,
                "original_probability": original_probability,
                "checks": outcomes,
            }
        )
    return {
        "checked_programs": checked,
        "all_checks_passed": bool(checked)
        and all(all(row["checks"].values()) for row in checked),
    }


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent = manifest["superseded_t10_3_1"]
    checks = {
        "parent_artifacts_bound": len(manifest["parent_artifacts"]) == 6,
        "t10_3_1_superseded_partial": parent.get("status") == "SUPERSEDED_PARTIAL",
        "t10_3_1_counts_frozen": all(
            parent.get(key) == value
            for key, value in {
                "intent_count": 95,
                "event_count": 94,
                "branch_count": 7,
                "interrupted_intent_count": 1,
            }.items()
        ),
        "parent_training_forbidden": parent.get("used_for_training") is False,
        "parent_mutation_forbidden": parent.get("mutated_by_t10_3_2") is False,
        "parent_replay_forbidden": parent.get("physical_actions_replayed") == 0,
        "source_only_firewall": not any(manifest["firewall"].values()),
        "budget_exact": manifest["matrix"]["total_maximum_actions"] == 6144,
        "reset_exact": manifest["matrix"]["total_resets"] == 30,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.2-offline-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "physical_actions": 0,
            "parent_events_used_for_training": 0,
            "status": (
                "PASS_T10_3_2_OFFLINE_AUDIT"
                if all(checks.values())
                else "INVALID_PROVENANCE"
            ),
        },
        "audit_checksum",
    )
    protocol.write_json_once(_artifact_path(root, AUDIT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.2 provenance audit failed")
    return payload


@dataclass(frozen=True)
class _SyntheticAction:
    name: str
    action_args: Mapping[str, Any]


def _run_synthetic_option(option: GoalDirectedOption, success_step: int) -> dict[str, Any]:
    sage_t = GoalDirectedSageTController(
        phase="preflight",
        warmup_actions=0,
        attestation_scope=f"synthetic-{option.schema}",
    )
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
    empty_posterior = len(sage_t.posterior.particles) == 0
    sage_t._active_option = option
    grid = np.zeros((5, 5), dtype=np.int16)
    legal_actions = tuple(
        _SyntheticAction(name, {}) for name in ("ACTION1", "ACTION2", "ACTION6")
    )
    source_matches = []
    for index, step in enumerate(option.steps[: success_step + 1]):
        before = grid.copy()
        decision = controller.select_action(
            current_grid=before,
            available_actions=tuple(item.name for item in legal_actions),
            legacy_action="ACTION1",
            legacy_action_data={},
            available_action_candidates=legal_actions,
            game_state="NOT_FINISHED",
            levels_completed=0,
        )
        source_matches.append(
            decision.source == "sage_t_joint_program"
            and decision.action_name == step.action_name
        )
        grid = before.copy()
        grid[2, 2] = 1 + ((index + 1) % 2)
        controller.observe_transition(
            action=decision.action_name,
            action_data=decision.action_data,
            grid_before=before,
            grid_after=grid,
            available_actions=tuple(item.name for item in legal_actions),
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            levels_completed_before=0,
            levels_completed_after=int(index == success_step),
        )
    summary = sage_t.summary()
    return {
        "schema": option.schema,
        "empty_posterior_at_start": empty_posterior,
        "level_progress": summary["option_successes"] >= 1,
        "same_controller_closed_loop": all(source_matches),
        "posterior_observed_transitions": controller.summary()["transitions_observed"],
        "actions": success_step + 1,
        "maximum_horizon": summary["maximum_option_horizon"],
    }


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    repeat = GoalDirectedOption(
        schema="repeat_target",
        steps=tuple(OptionStep("ACTION6") for _ in range(8)),
        source="synthetic_generation",
    )
    path = GoalDirectedOption(
        schema="path_successor",
        steps=tuple(OptionStep("ACTION6") for _ in range(10)),
        source="synthetic_generation",
    )
    mixed_steps = tuple(
        OptionStep("ACTION1" if index % 2 == 0 else "ACTION2")
        for index in range(20)
    )
    mixed = GoalDirectedOption(
        schema="mixed_automaton",
        steps=mixed_steps,
        source="synthetic_generation",
    )
    scenarios = [
        _run_synthetic_option(repeat, 4),
        _run_synthetic_option(path, 9),
        _run_synthetic_option(mixed, 19),
    ]
    checks = {
        "all_start_empty": all(row["empty_posterior_at_start"] for row in scenarios),
        "all_reach_level": all(row["level_progress"] for row in scenarios),
        "same_controller_closed_loop": all(
            row["same_controller_closed_loop"] for row in scenarios
        ),
        "posterior_updated_each_action": all(
            row["posterior_observed_transitions"] == row["actions"]
            for row in scenarios
        ),
        "path_length_10": scenarios[1]["actions"] == 10,
        "mixed_beyond_16": scenarios[2]["actions"] > 16,
        "mixed_has_two_schemas": len(set(mixed.action_schemas)) >= 2,
        "option_horizon_32": all(row["maximum_horizon"] == 32 for row in scenarios),
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.2-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "scenarios": scenarios,
            "checks": checks,
            "physical_actions": 0,
            "status": "PASS_T10_3_2_PREFLIGHT" if all(checks.values()) else "WIRING_MISS",
        },
        "preflight_checksum",
    )
    protocol.write_json_once(_artifact_path(root, PREFLIGHT_FILENAME), payload)
    if not all(checks.values()):
        raise protocol.ScientificGateMiss("T10.3.2 synthetic wiring failed")
    return payload


def _controller_pair(
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None,
) -> tuple[UnifiedCognitiveController, GoalDirectedSageTController | None]:
    if work.arm == "unified_sage_t_off":
        return (
            UnifiedCognitiveController(
                work.game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
            ),
            None,
        )
    phase = "confirmation" if work.phase == "confirm" else "discovery"
    goal = GoalDirectedSageTController(
        phase=phase,
        registry=registry,
        registry_checksum=registry_checksum,
        attestation_scope=work.work_id,
    )
    controller = UnifiedCognitiveController(
        work.game_id,
        config=UnifiedCognitiveConfig(
            sage_t_authority_mode="active",
            sage_t_counterfactual_gate_passed=True,
            sage_t_active_gate_passed=True,
        ),
        sage_t_controller=goal,
    )
    return controller, goal


def _intent_payload(
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    *,
    step_index: int,
    selected: Any,
    decision: Any,
    registry_checksum: str | None,
) -> dict[str, Any]:
    event_id = protocol.sha256_payload(
        {"manifest": manifest["manifest_checksum"], "work": work.work_id, "step": step_index}
    )
    return _signed(
        {
            "format_version": "sage-t10.3.2-action-intent-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "work": work.as_dict(),
            "event_id": event_id,
            "step_index": step_index,
            "action": {
                "name": str(getattr(selected, "name", "")),
                "argument_checksum": protocol.sha256_payload(
                    dict(getattr(selected, "action_args", {}) or {})
                ),
                "parameter_arity": len(
                    dict(getattr(selected, "action_args", {}) or {})
                ),
            },
            "decision_source": str(decision.source),
            "decision_reason": str(decision.reason),
            "registry_checksum": registry_checksum,
            "physical_action_replayed": False,
        },
        "intent_checksum",
    )


def _run_work(
    root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    lock: _CollectorLock,
    *,
    registry_checksum: str | None,
) -> dict[str, Any]:
    receipt_path = _receipt_for_work(destination, work)
    if receipt_path.is_file():
        return _read_signed(receipt_path, "receipt_checksum")
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
                    legacy_action_data=dict(getattr(proposal, "action_args", {}) or {}),
                    available_action_candidates=legal_actions,
                    game_state=before.game_state,
                    levels_completed=before.levels_completed,
                )
            except Exception as exc:  # noqa: BLE001 - controller boundary is fail-closed
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
            intent = _intent_payload(
                manifest,
                work,
                step_index=step_index,
                selected=selected,
                decision=decision,
                registry_checksum=(
                    None
                    if goal is None
                    else goal.last_decision_registry_checksum
                ),
            )
            name = f"{step_index:04d}.json"
            protocol.write_json_once(
                _work_path(destination, "intents", work, name), intent
            )
            _write_checkpoint(
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
            except Exception as exc:  # noqa: BLE001 - SDK boundary is unattestable
                unresolved = _signed(
                    {
                        "format_version": "sage-t10.3.2-unresolved-event-v1",
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
                    _work_path(destination, "unresolved", work, name), unresolved
                )
                errors.append("ENVIRONMENT_CALL_UNATTESTABLE")
                status = "ABORTED"
                break
            level_delta = max(
                0, int(after.levels_completed) - int(before.levels_completed)
            )
            event = _signed(
                {
                    "format_version": "sage-t10.3.2-physical-event-v1",
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
                    "raw_frame_retained": False,
                    "physical_action_replayed": False,
                },
                "event_checksum",
            )
            protocol.write_json_once(
                _work_path(destination, "events", work, name), event
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
            except Exception as exc:  # noqa: BLE001 - posterior boundary is fail-closed
                errors.append(f"CONTROLLER_OBSERVE:{type(exc).__name__}")
                status = "ABORTED"
                frame = after_frame
                final_level = int(after.levels_completed)
                break
            _write_checkpoint(
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
    except Exception as exc:  # noqa: BLE001 - durable branch receipt must be sealed
        errors.append(f"RUNTIME:{type(exc).__name__}")
        status = "ABORTED"
    finally:
        if env is not None:
            try:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            except Exception as exc:  # noqa: BLE001 - external environment cleanup
                errors.append(f"ENVIRONMENT_CLOSE:{type(exc).__name__}")
    summary = goal.summary() if goal is not None else {}
    intent_dir = _work_path(destination, "intents", work, "x").parent
    event_dir = _work_path(destination, "events", work, "x").parent
    unresolved_dir = _work_path(destination, "unresolved", work, "x").parent
    issued_intents = len(tuple(intent_dir.glob("*.json"))) if intent_dir.exists() else 0
    sealed_events = len(tuple(event_dir.glob("*.json"))) if event_dir.exists() else 0
    unresolved_intents = (
        len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    )
    receipt = _signed(
        {
            "format_version": "sage-t10.3.2-branch-receipt-v1",
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
            "errors": errors,
            "illegal_actions": illegal_actions,
            "game_over_actions": game_over_actions,
            "decision_latencies_ms": decision_latencies,
            "registry_checksum_loaded": registry_checksum,
            "registry_used_in_decision": bool(summary.get("registry_used_in_decision")),
            "controller_registry": (
                goal.registry.snapshot() if goal is not None else None
            ),
            "physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    protocol.write_json_once(receipt_path, receipt)
    _write_checkpoint(
        destination,
        manifest,
        phase=work.phase,
        work=work,
        state="BRANCH_RECEIPT_SEALED",
        registry=registry,
    )
    return receipt


def _phase_report(
    manifest: Mapping[str, Any],
    phase: str,
    receipts: Sequence[Mapping[str, Any]],
    registry: ProgressProgramRegistry,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    latencies = sorted(
        float(value)
        for receipt in receipts
        for value in receipt.get("decision_latencies_ms", ())
    )
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    common_checks = {
        "all_conditions_present": len(receipts) == len(protocol.work_specs(phase)),
        "zero_errors": all(not receipt.get("errors") for receipt in receipts),
        "zero_illegal_actions": all(int(receipt.get("illegal_actions", 0)) == 0 for receipt in receipts),
        "zero_game_over": all(int(receipt.get("game_over_actions", 0)) == 0 for receipt in receipts),
        "zero_physical_replay": all(int(receipt.get("physical_actions_replayed", 0)) == 0 for receipt in receipts),
        "latency_p95": p95 <= 2500.0,
        "intent_accounting": all(
            int(receipt.get("issued_intents", 0))
            == int(receipt.get("sealed_events", 0))
            + int(receipt.get("unresolved_intents", 0))
            for receipt in receipts
        ),
        "phase_budget": sum(
            int(receipt.get("issued_intents", 0)) for receipt in receipts
        )
        <= protocol.maximum_actions_for_phase(phase),
    }
    if phase == "discover-core":
        progress = {
            (str(row["game_id"]), int(row["seed"])): int(row.get("level_delta", 0))
            for row in receipts
        }
        scientific = {
            "level_each_game_and_seed": all(
                progress.get((game, seed), 0) >= 1
                for game in protocol.CORE_GAMES
                for seed in protocol.DISCOVERY_SEEDS
            ),
            "winning_sage_t_source": all(
                "sage_t_joint_program" in row.get("level_event_sources", ())
                for row in receipts
            ),
        }
        verdict = "PASS_T10_3_2_CORE_DISCOVERY" if all((*common_checks.values(), *scientific.values())) else "CORE_PROGRESS_MISS"
    else:
        progressed = [row for row in receipts if int(row.get("level_delta", 0)) >= 1]
        scientific = {
            "sequence_progress": len({str(row["game_id"]) for row in progressed}) >= 1,
            "mixed_program_progress": any(
                bool(row.get("mixed_program_used"))
                and "sage_t_joint_program" in row.get("level_event_sources", ())
                for row in progressed
            ),
        }
        verdict = "PASS_T10_3_2_SEQUENCE_DISCOVERY" if all((*common_checks.values(), *scientific.values())) else "MIXED_SEQUENCE_MISS"
    return _signed(
        {
            "format_version": f"sage-t10.3.2-{phase}-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "phase": phase,
            "receipts": [row["receipt_checksum"] for row in receipts],
            "metrics": {
                "resets": len(receipts),
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
                "levels": sum(int(row.get("level_delta", 0)) for row in receipts),
                "decision_p95_ms": p95,
                "promoted_registry_programs": len(registry.transferred_options()),
            },
            "checks": {**common_checks, **scientific, "registry_controls": controls},
            "verdict": verdict,
            "passed": verdict.startswith("PASS_"),
        },
        "report_checksum",
    )


def _require_live_runtime() -> Mapping[str, Any]:
    try:
        import arc_agi
    except ImportError as exc:
        capabilities: dict[str, Any] = {
            "ready": False,
            "reason": f"arc_agi_import_error:{exc}",
        }
    else:
        required = ("Arcade", "OperationMode", "EnvironmentWrapper")
        missing = [name for name in required if not hasattr(arc_agi, name)]
        versions = {}
        for package in ("arc-agi", "arcengine"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "missing"
        capabilities = {
            "ready": not missing,
            "reason": "" if not missing else f"missing_sdk_symbols:{','.join(missing)}",
            "versions": versions,
            "arc_agi_path": str(getattr(arc_agi, "__file__", "")),
        }
    if capabilities.get("ready") is not True:
        raise protocol.IntegrityError(
            f"ARC-AGI runtime is unavailable: {capabilities.get('reason', 'unknown')}"
        )
    return capabilities


def run_discovery(root: Path, manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    _require_live_runtime()
    if phase == "discover-sequence":
        core = _read_signed(_artifact_path(root, CORE_REPORT_FILENAME), "report_checksum")
        if core.get("passed") is not True:
            raise protocol.ScientificGateMiss("core gate forbids sequence discovery")
    destination = _destination(root)
    _recover_orphans(destination, manifest)
    registry = _registry_from_receipts(destination)
    lock = _CollectorLock(_lock_path(destination), phase)
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
    controls = _apply_registry_controls(registry)
    receipts = _load_receipts(destination, phase)
    report = _phase_report(manifest, phase, receipts, registry, controls)
    report_name = CORE_REPORT_FILENAME if phase == "discover-core" else SEQUENCE_REPORT_FILENAME
    registry_name = CORE_REGISTRY_FILENAME if phase == "discover-core" else SEQUENCE_REGISTRY_FILENAME
    protocol.write_json_once(_artifact_path(root, report_name), report)
    protocol.write_json_once(_artifact_path(root, registry_name), registry.snapshot())
    if report.get("passed") is not True:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def compile_registry(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    core = _read_signed(_artifact_path(root, CORE_REPORT_FILENAME), "report_checksum")
    sequence = _read_signed(_artifact_path(root, SEQUENCE_REPORT_FILENAME), "report_checksum")
    if core.get("passed") is not True or sequence.get("passed") is not True:
        raise protocol.ScientificGateMiss("discovery gates forbid registry compilation")
    source = _read_signed(_artifact_path(root, SEQUENCE_REGISTRY_FILENAME), "registry_checksum")
    registry = ProgressProgramRegistry(source)
    controls = _apply_registry_controls(registry)
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
    }
    protocol.write_json_once(_artifact_path(root, COMPILED_REGISTRY_FILENAME), compiled)
    report = _signed(
        {
            "format_version": "sage-t10.3.2-compile-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": compiled["registry_checksum"],
            "program_count": len(programs),
            "controls": controls,
            "checks": checks,
            "passed": all(checks.values()),
            "verdict": "PASS_T10_3_2_REGISTRY" if all(checks.values()) else "REGISTRY_REPRODUCTION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, COMPILE_REPORT_FILENAME), report)
    if report.get("passed") is not True:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def run_confirmation(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_live_runtime()
    compile_report = _read_signed(_artifact_path(root, COMPILE_REPORT_FILENAME), "report_checksum")
    if compile_report.get("passed") is not True:
        raise protocol.ScientificGateMiss("compiled registry gate forbids confirmation")
    compiled = _read_signed(_artifact_path(root, COMPILED_REGISTRY_FILENAME), "registry_checksum")
    registry_checksum = str(compiled["registry_checksum"])
    destination = _destination(root)
    _recover_orphans(destination, manifest)
    lock = _CollectorLock(_lock_path(destination), "confirm")
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
    receipts = _load_receipts(destination, "confirm")
    by_key = {
        (str(row["game_id"]), int(row["seed"]), str(row["arm"])): row
        for row in receipts
    }
    active_levels = {}
    baseline_levels = {}
    for game in protocol.ALL_SOURCE_GAMES:
        active_levels[game] = sum(
            int(by_key.get((game, seed, "goal_directed_sage_t"), {}).get("level_delta", 0))
            for seed in protocol.CONFIRMATION_SEEDS
        )
        baseline_levels[game] = sum(
            int(by_key.get((game, seed, "unified_sage_t_off"), {}).get("level_delta", 0))
            for seed in protocol.CONFIRMATION_SEEDS
        )
    active_rows = [row for row in receipts if row["arm"] == "goal_directed_sage_t"]
    baseline_rows = [row for row in receipts if row["arm"] == "unified_sage_t_off"]
    latencies = sorted(
        float(value)
        for row in active_rows
        for value in row.get("decision_latencies_ms", ())
    )
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    checks = {
        "all_conditions_present": len(receipts) == 20,
        "core_each_seed": all(
            int(by_key.get((game, seed, "goal_directed_sage_t"), {}).get("level_delta", 0)) >= 1
            for game in protocol.CORE_GAMES
            for seed in protocol.CONFIRMATION_SEEDS
        ),
        "sequence_progress": sum(active_levels[game] for game in protocol.SEQUENCE_GAMES) >= 1,
        "no_game_regression": all(active_levels[game] >= baseline_levels[game] for game in protocol.ALL_SOURCE_GAMES),
        "total_level_advantage": sum(active_levels.values()) >= sum(baseline_levels.values()) + 1,
        "zero_errors": all(not row.get("errors") for row in receipts),
        "zero_illegal_actions": all(int(row.get("illegal_actions", 0)) == 0 for row in receipts),
        "game_over_nonincrease": sum(int(row.get("game_over_actions", 0)) for row in active_rows) <= sum(int(row.get("game_over_actions", 0)) for row in baseline_rows),
        "registry_loaded": all(row.get("registry_checksum_loaded") == registry_checksum for row in active_rows),
        "registry_used": all(bool(row.get("registry_used_in_decision")) for row in active_rows),
        "winning_decisions_attest_registry": all(
            all(checksum == registry_checksum for checksum in row.get("winning_registry_checksums", ()))
            and len(row.get("winning_registry_checksums", ()))
            == len(row.get("level_event_sources", ()))
            for row in active_rows
        ),
        "decision_p95": p95 <= 2500.0,
        "budget": sum(int(row.get("issued_intents", 0)) for row in receipts) <= protocol.CONFIRMATION_ACTIONS,
        "intent_accounting": all(
            int(row.get("issued_intents", 0))
            == int(row.get("sealed_events", 0))
            + int(row.get("unresolved_intents", 0))
            for row in receipts
        ),
        "zero_physical_replay": all(int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts),
    }
    report = _signed(
        {
            "format_version": "sage-t10.3.2-confirmation-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": registry_checksum,
            "metrics": {
                "active_levels": active_levels,
                "baseline_levels": baseline_levels,
                "active_total_levels": sum(active_levels.values()),
                "baseline_total_levels": sum(baseline_levels.values()),
                "decision_p95_ms": p95,
                "actions": sum(int(row.get("issued_intents", 0)) for row in receipts),
            },
            "checks": checks,
            "passed": all(checks.values()),
            "verdict": "PASS_T10_3_2_SOURCE_CONFIRMATION" if all(checks.values()) else "SOURCE_CONFIRMATION_MISS",
        },
        "report_checksum",
    )
    protocol.write_json_once(_artifact_path(root, CONFIRMATION_REPORT_FILENAME), report)
    if report.get("passed") is not True:
        raise protocol.ScientificGateMiss(str(report["verdict"]))
    return report


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
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
        artifacts[name] = _read_signed(path, checksum) if path.is_file() else None
    if artifacts["audit"] is None or artifacts["audit"].get("status") != "PASS_T10_3_2_OFFLINE_AUDIT":
        verdict = "INVALID_PROVENANCE"
    elif artifacts["preflight"] is None or artifacts["preflight"].get("status") != "PASS_T10_3_2_PREFLIGHT":
        verdict = "WIRING_MISS"
    elif artifacts["core"] is None or artifacts["core"].get("passed") is not True:
        verdict = "CORE_PROGRESS_MISS"
    elif artifacts["sequence"] is None or artifacts["sequence"].get("passed") is not True:
        verdict = "MIXED_SEQUENCE_MISS"
    elif artifacts["compile"] is None or artifacts["compile"].get("passed") is not True:
        verdict = "REGISTRY_REPRODUCTION_MISS"
    elif artifacts["confirmation"] is None or artifacts["confirmation"].get("passed") is not True:
        verdict = "SOURCE_CONFIRMATION_MISS"
    else:
        verdict = "PASS_T10_3_2_END_TO_END_SOURCE"
    accounting = _journal_accounting(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.2-terminal-report-v1",
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
    protocol.write_json_once(_artifact_path(root, TERMINAL_REPORT_FILENAME), report)
    return report


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = _destination(root)
    accounting = _journal_accounting(destination)
    if not accounting["equation_holds"]:
        raise protocol.IntegrityError("intent accounting equation does not hold")
    if accounting["inflight_intents"] and not accounting["inflight_valid"]:
        state = "INTERRUPTED"
    elif accounting["live_collector_lock"]:
        state = "RUNNING"
    elif accounting["inflight_intents"] or accounting["incomplete_work_ids"]:
        state = "INTERRUPTED"
    elif _artifact_path(root, TERMINAL_REPORT_FILENAME).is_file():
        state = "TERMINAL"
    else:
        state = "READY"
    artifacts = {}
    for filename in (
        AUDIT_FILENAME,
        PREFLIGHT_FILENAME,
        CORE_REPORT_FILENAME,
        SEQUENCE_REPORT_FILENAME,
        COMPILE_REPORT_FILENAME,
        CONFIRMATION_REPORT_FILENAME,
        TERMINAL_REPORT_FILENAME,
    ):
        path = _artifact_path(root, filename)
        artifacts[filename] = protocol.file_sha256(path) if path.is_file() else None
    return {
        "phase": "status",
        "status": state,
        "manifest_checksum": manifest["manifest_checksum"],
        "accounting": accounting,
        "completed_resets": len(_receipt_paths(destination)),
        "maximum_resets": protocol.TOTAL_RESETS,
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
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
        return 0 if report["verdict"] == "PASS_T10_3_2_END_TO_END_SOURCE" else 3
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
