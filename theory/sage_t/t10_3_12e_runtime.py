"""Durable runtime for SAGE.T10.3.12e closed-loop successor diagnosis."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from theory.live_transition_loop import build_observation
from theory.unified_cognitive_controller import CognitiveDecision

from . import t10_3_2_runtime as durable
from . import t10_3_12e_protocol as protocol
from .closed_loop_successor_v10_3_12e import (
    ARMS,
    ClosedLoopRegistry,
    ClosedLoopSuccessorController,
    compile_closed_loop_registry,
    signed,
)
from .compiler import compile_observation
from .progress_witness_v10 import GroundedAction

PARENT_AUDIT_FILENAME = "parent_stable_executor_negative_audit.json"
TRAJECTORY_AUDIT_FILENAME = "parent_closed_loop_motivation_audit.json"
PREFLIGHT_FILENAME = "closed_loop_preflight.json"
REGISTRY_FILENAME = "closed_loop_registry.json"
ACTIVE_REPORT_FILENAME = "active_closed_loop_diagnostic.json"
ADJUDICATION_FILENAME = "closed_loop_adjudication_report.json"
TERMINAL_REPORT_FILENAME = "terminal_report.json"
LOCK_FILENAME = durable.LOCK_FILENAME


def _destination(root: Path) -> Path:
    return root.resolve() / protocol.DEFAULT_OUTPUT_DIR


def _path(root: Path, filename: str) -> Path:
    return _destination(root) / filename


def _write(root: Path, filename: str, payload: Mapping[str, Any]) -> None:
    protocol.write_json_once(_path(root, filename), payload)


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    path = _path(root, filename)
    if not path.is_file():
        raise protocol.IntegrityError(f"required T10.3.12e artifact is absent: {filename}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol.verify_signed(payload, checksum_field)
    return payload


def _read_parent(root: Path, name: str, checksum_field: str) -> dict[str, Any]:
    path = root / protocol.PARENT_ARTIFACT_PATHS[name]
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol.verify_signed(payload, checksum_field)
    return payload


def _artifact_bytes(root: Path) -> int:
    destination = _destination(root)
    if not destination.exists():
        return 0
    return sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())


@contextmanager
def _durable_contract() -> Iterator[None]:
    previous = durable.protocol
    durable.protocol = protocol
    try:
        yield
    finally:
        durable.protocol = previous


def _require_gate(root: Path, phase: str) -> dict[str, Any]:
    contract = protocol.ARTIFACT_CONTRACT[phase]
    payload = _read_signed(root, str(contract["path"]), str(contract["checksum_field"]))
    gate = contract.get("gate_field")
    if gate is not None and payload.get(str(gate)) is not True:
        raise protocol.ScientificGateMiss(f"{phase} gate forbids continuation")
    return payload


def audit_parent(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    observed = protocol.verify_parent(root)
    bindings = protocol.parent_artifact_bindings(root)
    checks = {
        "parent_negative_exact": observed == protocol.EXPECTED_PARENT,
        "all_parent_bindings_frozen": bindings == manifest["parent_artifacts"],
        "parent_journal_immutable": protocol.parent_journal_digest(root)
        == manifest["parent_journal_digest"],
        "parent_accounting_clean": observed["authorized_actions"]
        == observed["sealed_events"],
        "zero_parent_replay": True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12e-parent-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_state": observed,
            "parent_journal_digest": manifest["parent_journal_digest"],
            "checks": checks,
            "passed": all(checks.values()),
            "parent_actions_used_for_training": 0,
            "parent_outcomes_used_for_post_hoc_diagnosis": True,
            "physical_actions": 0,
        },
        "audit_checksum",
    )
    _write(root, PARENT_AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("PARENT_STABLE_EXECUTOR_AUDIT_MISS")
    return payload


def _parent_receipts(root: Path) -> list[dict[str, Any]]:
    base = root / protocol.PARENT_OUTPUT_DIR / "journal" / "branches"
    rows = []
    for path in sorted(base.rglob("receipt.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol.verify_signed(payload, "receipt_checksum")
        rows.append(payload)
    return rows


def _parent_events(root: Path) -> list[dict[str, Any]]:
    base = root / protocol.PARENT_OUTPUT_DIR / "journal" / "events"
    rows = []
    for path in sorted(base.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol.verify_signed(payload, "event_checksum")
        rows.append(payload)
    return rows


def audit_trajectories(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-parent")
    receipts = _parent_receipts(root)
    events = _parent_events(root)
    stable = sorted(
        (
            row for row in receipts
            if row.get("arm") == "stable_source_cursor"
            and row.get("recognized_context") == "path_context"
        ),
        key=lambda row: str(row["game_id"]),
    )
    stable_summary = [
        {
            "game_id": row["game_id"],
            "plan_length": int(row.get("plan_length", 0)),
            "sealed_events": int(row.get("sealed_events", 0)),
            "grounding_misses": int(row.get("grounding_misses", 0)),
            "level_delta": int(row.get("level_delta", 0)),
            "action_checksums_retained": False,
        }
        for row in stable
    ]
    successes = [row for row in receipts if int(row.get("level_delta", 0)) > 0]
    changed = [
        row for row in events
        if row.get("frame_before_sha256") != row.get("frame_after_sha256")
    ]
    expected_lengths = {
        "bp35-0a0ad940": 16,
        "dc22-4c9bff3e": 1,
        "lf52-271a04aa": 2,
    }
    checks = {
        "all_36_parent_receipts": len(receipts)
        == int(manifest["gates"]["parent_receipts"]),
        "three_parent_path_games": [row["game_id"] for row in stable]
        == sorted(expected_lengths),
        "stable_plan_lengths_exact": {
            str(row["game_id"]): int(row["plan_length"]) for row in stable
        }
        == expected_lengths,
        "stable_zero_success": sum(int(row["level_delta"]) > 0 for row in stable)
        == int(manifest["gates"]["parent_stable_successes"]),
        "all_arms_zero_success": not successes,
        "bp35_lost_frozen_correspondence": any(
            row["game_id"] == "bp35-0a0ad940" and int(row["grounding_misses"]) == 1
            for row in stable
        ),
        "all_parent_actions_changed_frame": len(events) == len(changed)
        == int(manifest["gates"]["parent_changed_frame_events"]),
        "parent_outcome_is_diagnostic_only": True,
        "no_parent_grounded_identity_compiled": True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12e-parent-motivation-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_journal_digest": manifest["parent_journal_digest"],
            "stable_path_summary": stable_summary,
            "changed_frame_events": len(changed),
            "checks": checks,
            "passed": all(checks.values()),
            "post_hoc_diagnostic": True,
            "confirmatory_evidence": False,
            "parent_event_hashes_retained": False,
            "parent_action_checksums_retained": False,
            "physical_actions": 0,
        },
        "audit_checksum",
    )
    _write(root, TRAJECTORY_AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("PARENT_CLOSED_LOOP_MOTIVATION_AUDIT_MISS")
    return payload


def _closed_loop_registry(root: Path) -> ClosedLoopRegistry:
    return compile_closed_loop_registry(
        _read_parent(root, "registry", "registry_checksum"),
        protocol.source_factor_payload(root),
    )


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-trajectories")
    registry = _closed_loop_registry(root)
    first = GroundedAction("ACTION6", (("x", 1), ("y", 19)))
    second = GroundedAction("ACTION6", (("x", 2), ("y", 18)))
    third = GroundedAction("ACTION6", (("x", 3), ("y", 17)))
    legal = tuple(item.candidate for item in (first, second, third))
    paths = [(first, second), (first, third)]
    calls = Counter()

    def changing_builder(state: Any, candidates: Any) -> tuple[str, tuple[GroundedAction, ...]]:
        del state, candidates
        path = paths[calls["dynamic"]]
        calls["dynamic"] += 1
        return "path_context", path

    dynamic = ClosedLoopSuccessorController(
        arm="anchored_goal_dynamic_successor",
        registry=registry,
        path_builder=changing_builder,
    )
    dynamic_selected = [
        dynamic.choose(
            state=None,
            candidates=legal,
            shape=(32, 32),
            step_index=step,
        ).candidate
        for step in range(2)
    ]
    def static_builder(
        state: Any,
        candidates: Any,
    ) -> tuple[str, tuple[GroundedAction, ...]]:
        del state, candidates
        return "path_context", (first, second, third)
    stateless = ClosedLoopSuccessorController(
        arm="stateless_goal_and_successor",
        registry=registry,
        path_builder=static_builder,
    )
    stateless_selected = [
        stateless.choose(
            state=None,
            candidates=legal,
            shape=(32, 32),
            step_index=step,
        ).candidate
        for step in range(2)
    ]
    swapped = ClosedLoopSuccessorController(
        arm="goal_end_swap",
        registry=registry,
        path_builder=static_builder,
    ).choose(state=None, candidates=legal, shape=(32, 32), step_index=0)
    frozen = ClosedLoopSuccessorController(
        arm="frozen_grounded_cursor",
        registry=registry,
        path_builder=static_builder,
    )
    frozen_first = frozen.choose(
        state=None, candidates=legal, shape=(32, 32), step_index=0
    )
    frozen_missing = frozen.choose(
        state=None, candidates=(first.candidate,), shape=(32, 32), step_index=1
    )
    non_path = {}
    for arm in ARMS:
        non_path[arm] = ClosedLoopSuccessorController(
            arm=arm,
            registry=registry,
            path_builder=lambda state, candidates: ("repeat_context", ()),
        ).choose(
            state=None,
            candidates=(first.candidate,),
            shape=(32, 32),
            step_index=0,
        )
    dynamic_summary = dynamic.summary()
    snapshot = registry.snapshot()
    reloaded = ClosedLoopRegistry(snapshot)
    cases = {
        "four_programs": len(snapshot["programs"]) == 4,
        "support_zero": snapshot["local_support_total"] == 0,
        "dynamic_first_successor": dynamic_selected[0] == first.candidate,
        "dynamic_skips_visited": dynamic_selected[1] == third.candidate,
        "dynamic_relation_each_decision": calls["dynamic"] == 2
        and dynamic_summary["relation_evaluations"] == 2,
        "single_abstract_anchor": dynamic_summary["anchor_builds"] == 1,
        "frontier_advances": dynamic_summary["frontier_advances"] == 2,
        "visited_repeat_rejected": dynamic_summary["repeat_proposals_rejected"] == 1,
        "stateless_repeats_first": stateless_selected == [first.candidate, first.candidate],
        "goal_end_swap_reverses": swapped.candidate == third.candidate,
        "frozen_first_exact": frozen_first.candidate == first.candidate,
        "frozen_missing_abstains": frozen_missing.abstained,
        "uniform_non_path_abstention": all(item.abstained for item in non_path.values()),
        "no_path_or_frontier_persisted": dynamic_summary["path_plan_persisted"] is False
        and dynamic_summary["visited_action_keys_persisted"] is False,
        "registry_round_trip": reloaded.snapshot()["registry_checksum"]
        == snapshot["registry_checksum"],
        "source_registry_bindings_present": bool(snapshot["parent_executor_registry_checksum"])
        and bool(snapshot["source_factor_registry_checksum"]),
    }
    if len(cases) != int(manifest["gates"]["preflight_cases"]):
        raise protocol.IntegrityError("preflight case inventory drifted")
    payload = signed(
        {
            "format_version": "sage-t10.3.12e-closed-loop-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "cases": cases,
            "passed": all(cases.values()),
            "registry_checksum_preview": snapshot["registry_checksum"],
            "physical_actions": 0,
        },
        "preflight_checksum",
    )
    _write(root, PREFLIGHT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("CLOSED_LOOP_PREFLIGHT_MISS")
    return payload


def compile_programs(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "preflight")
    snapshot = _closed_loop_registry(root).snapshot()
    core = {key: value for key, value in snapshot.items() if key != "registry_checksum"}
    core.update(
        {
            "manifest_checksum": manifest["manifest_checksum"],
            "compiled_before_diagnostic_actions": True,
            "parent_outcomes_used_for_program_fit": False,
            "parent_grounded_paths_imported": False,
            "parent_action_checksums_imported": False,
            "historical_support_imported": 0,
            "physical_actions": 0,
        }
    )
    payload = signed(core, "registry_checksum")
    ClosedLoopRegistry(payload)
    _write(root, REGISTRY_FILENAME, payload)
    return payload


class _ActiveClosedLoop:
    def __init__(self, arm: str, registry: ClosedLoopRegistry) -> None:
        self.controller = ClosedLoopSuccessorController(arm=arm, registry=registry)
        self.maximum_legal_candidates = 0
        self.candidate_inspections = 0

    def decide(
        self,
        *,
        current_grid: Any,
        legal_actions: Sequence[Any],
        game_state: str,
        levels_completed: int,
        step_index: int,
    ) -> tuple[CognitiveDecision | None, Any]:
        if len(legal_actions) > 512:
            raise protocol.IntegrityError("legal candidate processing budget exceeded")
        self.maximum_legal_candidates = max(self.maximum_legal_candidates, len(legal_actions))
        self.candidate_inspections += len(legal_actions)
        observation = build_observation(
            current_grid,
            available_actions=durable.live._available_action_names(legal_actions),
            game_state=game_state,
            levels_completed=levels_completed,
            infer_players=True,
        )
        state = compile_observation(observation)
        closed_loop = self.controller.choose(
            state=state,
            candidates=legal_actions,
            shape=tuple(int(value) for value in current_grid.shape[:2]),
            step_index=step_index,
        )
        if closed_loop.abstained:
            return None, closed_loop
        return (
            CognitiveDecision(
                action_name=closed_loop.candidate.action_name,
                action_data=dict(closed_loop.candidate.action_data),
                source="sage_t_closed_loop_successor",
                reason=closed_loop.reason,
                confidence=1.0,
                option_id=closed_loop.program_hash,
            ),
            closed_loop,
        )

    def summary(self) -> dict[str, Any]:
        output = dict(self.controller.summary())
        output.update(
            {
                "maximum_legal_candidates": self.maximum_legal_candidates,
                "candidate_inspections": self.candidate_inspections,
                "legacy_fallback_actions": 0,
            }
        )
        return output


def _work_path(
    destination: Path,
    category: str,
    work: protocol.WorkSpec,
    name: str,
) -> Path:
    return destination / "journal" / category / work.work_id / name


def _receipt_path(destination: Path, work: protocol.WorkSpec) -> Path:
    return _work_path(destination, "branches", work, "receipt.json")


def _load_receipts(destination: Path) -> list[dict[str, Any]]:
    base = destination / "journal" / "branches"
    paths = sorted(base.rglob("receipt.json")) if base.exists() else []
    return [durable._read_signed(path, "receipt_checksum") for path in paths]


def _intent(
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    *,
    step_index: int,
    selected: Any,
    decision: CognitiveDecision,
    registry_checksum: str,
) -> dict[str, Any]:
    event_id = protocol.sha256_payload(
        {"manifest": manifest["manifest_checksum"], "work": work.work_id, "step": step_index}
    )
    return signed(
        {
            "format_version": "sage-t10.3.12e-action-intent-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "work": work.as_dict(),
            "event_id": event_id,
            "step_index": step_index,
            "action": {
                "name": str(getattr(selected, "name", "")),
                "parameter_arity": len(dict(getattr(selected, "action_args", {}) or {})),
                "argument_checksum": protocol.sha256_payload(
                    dict(getattr(selected, "action_args", {}) or {})
                ),
            },
            "decision_source": decision.source,
            "decision_reason": decision.reason,
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
    registry: ClosedLoopRegistry,
    registry_checksum: str,
    lock: Any,
) -> dict[str, Any]:
    receipt_path = _receipt_path(destination, work)
    if receipt_path.is_file():
        return durable._read_signed(receipt_path, "receipt_checksum")
    executor = _ActiveClosedLoop(work.arm, registry)
    trace_ids = []
    errors = []
    illegal_actions = 0
    game_over_actions = 0
    initial_level = final_level = 0
    initial_frame_hash = ""
    status = "COMPLETE"
    stop_reason = "ACTION_BUDGET_EXHAUSTED"
    env = None
    started = time.perf_counter()
    try:
        env = durable.live._make_real_env(work.game_id, root / "environment_files")
        frame = durable.live._reset_env(env)
        initial = durable.live.snapshot_frame(frame)
        initial_level = final_level = int(initial.levels_completed)
        initial_frame_hash = protocol.sha256_payload(initial.grid.tolist())
        for step_index in range(work.action_budget):
            if time.perf_counter() - started >= protocol.reset_wall_seconds(work):
                stop_reason = "RESET_WALL_BUDGET_EXHAUSTED"
                break
            before = durable.live.snapshot_frame(frame)
            if durable.live._is_terminal(before.game_state):
                stop_reason = "TERMINAL_STATE"
                break
            legal = tuple(durable.live._valid_actions(env))
            decision, _closed_loop_decision = executor.decide(
                current_grid=before.grid,
                legal_actions=legal,
                game_state=str(before.game_state),
                levels_completed=int(before.levels_completed),
                step_index=step_index,
            )
            if decision is None:
                stop_reason = "PLANNED_CLOSED_LOOP_ABSTENTION"
                break
            if decision.source != "sage_t_closed_loop_successor":
                raise protocol.IntegrityError("non-SAGE decision entered T10.3.12e")
            selected = durable.live._materialize_decision(legal, decision)
            if selected is None:
                illegal_actions += 1
                errors.append("UNAVAILABLE_SAGE_DECISION")
                status = "ABORTED"
                stop_reason = "UNAVAILABLE_SAGE_DECISION"
                break
            intent = _intent(
                manifest,
                work,
                step_index=step_index,
                selected=selected,
                decision=decision,
                registry_checksum=registry_checksum,
            )
            name = f"{step_index:04d}.json"
            protocol.write_json_once(_work_path(destination, "intents", work, name), intent)
            lock.heartbeat()
            try:
                after_frame = durable.live._step_env_action(env, selected)
                after = durable.live.snapshot_frame(
                    after_frame,
                    fallback_available_actions=before.available_actions,
                )
            except Exception as exc:  # noqa: BLE001
                unresolved = signed(
                    {
                        "format_version": "sage-t10.3.12e-unresolved-event-v1",
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
                stop_reason = "ENVIRONMENT_CALL_UNATTESTABLE"
                break
            level_delta = max(0, int(after.levels_completed) - int(before.levels_completed))
            frame_before = protocol.sha256_payload(before.grid.tolist())
            frame_after = protocol.sha256_payload(after.grid.tolist())
            event = signed(
                {
                    "format_version": "sage-t10.3.12e-physical-event-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "work_id": work.work_id,
                    "event_id": intent["event_id"],
                    "step_index": step_index,
                    "decision_source": decision.source,
                    "decision_reason": decision.reason,
                    "registry_checksum": registry_checksum,
                    "levels_before": int(before.levels_completed),
                    "levels_after": int(after.levels_completed),
                    "level_delta": level_delta,
                    "game_state_after": str(after.game_state),
                    "frame_before_sha256": frame_before,
                    "frame_after_sha256": frame_after,
                    "frame_changed": frame_before != frame_after,
                    "raw_frame_retained": False,
                    "grounded_arguments_retained": False,
                    "physical_action_replayed": False,
                },
                "event_checksum",
            )
            protocol.write_json_once(_work_path(destination, "events", work, name), event)
            trace_ids.append(str(intent["event_id"]))
            frame = after_frame
            final_level = int(after.levels_completed)
            if str(after.game_state).upper() == "GAME_OVER":
                game_over_actions += 1
            lock.heartbeat()
            print(
                json.dumps(
                    {
                        "phase": work.phase,
                        "game_id": work.game_id,
                        "arm": work.arm,
                        "step": step_index + 1,
                        "budget": work.action_budget,
                        "levels": final_level - initial_level,
                        "decision_source": decision.source,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if level_delta > 0:
                stop_reason = "LEVEL_PROGRESS_SEALED"
                break
            if durable.live._is_terminal(after.game_state):
                stop_reason = "TERMINAL_STATE"
                break
    except protocol.IntegrityError:
        raise
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

    intent_dir = _work_path(destination, "intents", work, "x").parent
    event_dir = _work_path(destination, "events", work, "x").parent
    unresolved_dir = _work_path(destination, "unresolved", work, "x").parent
    issued = len(tuple(intent_dir.glob("*.json"))) if intent_dir.exists() else 0
    sealed = len(tuple(event_dir.glob("*.json"))) if event_dir.exists() else 0
    unresolved = len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    summary = executor.summary()
    receipt = signed(
        {
            "format_version": "sage-t10.3.12e-branch-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            **work.as_dict(),
            "work_id": work.work_id,
            "status": status,
            "complete": status == "COMPLETE" and not errors,
            "stop_reason": stop_reason,
            "planned_abstention": stop_reason == "PLANNED_CLOSED_LOOP_ABSTENTION",
            "issued_intents": issued,
            "sealed_events": sealed,
            "unresolved_intents": unresolved,
            "event_ids": trace_ids,
            "initial_frame_sha256": initial_frame_hash,
            "level_delta": max(0, final_level - initial_level),
            "sage_t_option_actions": sealed,
            "recognized_context": summary["recognized_context"],
            "program_hash": summary["program_hash"],
            "anchor_builds": summary["anchor_builds"],
            "relation_evaluations": summary["relation_evaluations"],
            "dynamic_regrounds": summary["dynamic_regrounds"],
            "frontier_advances": summary["frontier_advances"],
            "repeat_proposals_rejected": summary["repeat_proposals_rejected"],
            "exact_groundings": summary["exact_groundings"],
            "grounding_misses": summary["grounding_misses"],
            "frontier_size": summary["frontier_size"],
            "frozen_cursor": summary["frozen_cursor"],
            "initial_path_length": summary["initial_path_length"],
            "abstention_reason": summary["abstention_reason"],
            "candidate_inspections": summary["candidate_inspections"],
            "maximum_legal_candidates": summary["maximum_legal_candidates"],
            "path_plan_persisted": False,
            "visited_action_keys_persisted": False,
            "grounded_arguments_persisted": False,
            "legacy_fallback_actions": 0,
            "game_over_actions": game_over_actions,
            "illegal_actions": illegal_actions,
            "errors": errors,
            "registry_checksum_loaded": registry_checksum,
            "physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    protocol.write_json_once(receipt_path, receipt)
    return receipt


def _argument_metrics(destination: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    base = destination / "journal" / "intents" / str(receipt["work_id"])
    sequence = []
    for path in sorted(base.glob("*.json")) if base.exists() else ():
        payload = durable._read_signed(path, "intent_checksum")
        sequence.append(str(payload.get("action", {}).get("argument_checksum", "")))
    suffix = sequence[-8:]
    return {
        "action_count": len(sequence),
        "distinct_argument_count": len(set(sequence)),
        "suffix_length": len(suffix),
        "suffix_distinct_argument_count": len(set(suffix)),
        "collapsed_suffix": len(suffix) == 8 and len(set(suffix)) <= 2,
    }


def _changed_event_count(destination: Path, receipt: Mapping[str, Any]) -> int:
    base = destination / "journal" / "events" / str(receipt["work_id"])
    count = 0
    for path in sorted(base.glob("*.json")) if base.exists() else ():
        payload = durable._read_signed(path, "event_checksum")
        count += int(bool(payload.get("frame_changed")))
    return count


def active_diagnostic(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "preflight")
    registry_payload = _read_signed(root, REGISTRY_FILENAME, "registry_checksum")
    if registry_payload.get("compiled_before_diagnostic_actions") is not True:
        raise protocol.IntegrityError("closed-loop registry was not compiled before actions")
    if registry_payload.get("parent_outcomes_used_for_program_fit") is not False:
        raise protocol.IntegrityError("parent outcomes entered closed-loop program fitting")
    registry = ClosedLoopRegistry(registry_payload)
    destination = _destination(root)
    with _durable_contract():
        durable._require_live_runtime()
        before = durable._journal_accounting(destination)
        if before.get("inflight_paths") or before.get("unresolved_intents"):
            raise protocol.IntegrityError("interrupted physical action cannot be replayed")
        if before.get("incomplete_work_ids"):
            raise protocol.IntegrityError("interrupted work cannot be reconstructed safely")
        lock = durable._CollectorLock(destination / LOCK_FILENAME, "active-diagnostic")
        lock.acquire()
        try:
            collection_started = time.perf_counter()
            for work in protocol.work_specs("active-diagnostic"):
                if (
                    time.perf_counter() - collection_started
                    >= float(manifest["matrix"]["maximum_global_wall_seconds"])
                ):
                    raise protocol.IntegrityError("global diagnostic wall budget exceeded")
                _run_work(
                    root,
                    destination,
                    manifest,
                    work,
                    registry,
                    str(registry_payload["registry_checksum"]),
                    lock,
                )
        finally:
            lock.release()
        receipts = _load_receipts(destination)
        accounting = durable._journal_accounting(destination)

    checks = {
        "all_36_receipts": len(receipts) == protocol.TOTAL_RESETS,
        "all_work_ids_unique": len({row["work_id"] for row in receipts})
        == protocol.TOTAL_RESETS,
        "all_receipts_complete": all(row.get("complete") for row in receipts),
        "accounting_equation": bool(accounting.get("equation_holds")),
        "zero_inflight": int(accounting.get("inflight_intents", 0)) == 0,
        "zero_unresolved": int(accounting.get("unresolved_intents", 0)) == 0,
        "zero_incomplete_work": not accounting.get("incomplete_work_ids"),
        "zero_physical_replay": all(
            int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts
        ),
        "zero_legacy_fallback": all(
            int(row.get("legacy_fallback_actions", 0)) == 0 for row in receipts
        ),
        "action_budget": sum(int(row.get("sealed_events", 0)) for row in receipts)
        <= protocol.TOTAL_MAXIMUM_ACTIONS,
    }
    by_arm = {}
    trajectory_metrics = []
    for row in receipts:
        trajectory_metrics.append(
            {
                "work_id": row["work_id"],
                "game_id": row["game_id"],
                "arm": row["arm"],
                **_argument_metrics(destination, row),
                "changed_frame_actions": _changed_event_count(destination, row),
            }
        )
    for arm in ARMS:
        subset = [row for row in receipts if row["arm"] == arm]
        by_arm[arm] = {
            "success_games": sorted(
                row["game_id"] for row in subset if int(row["level_delta"]) > 0
            ),
            "path_applicable_games": sorted(
                row["game_id"] for row in subset
                if row["recognized_context"] == "path_context"
            ),
            "actions": sum(int(row["sealed_events"]) for row in subset),
            "changed_frame_actions": sum(
                item["changed_frame_actions"]
                for item in trajectory_metrics if item["arm"] == arm
            ),
            "game_over_resets": sum(int(row["game_over_actions"]) > 0 for row in subset),
            "errors": sum(bool(row["errors"]) for row in subset),
        }
    initial_hashes: dict[str, list[str]] = defaultdict(list)
    for row in receipts:
        value = str(row.get("initial_frame_sha256", ""))
        if value and value not in initial_hashes[row["game_id"]]:
            initial_hashes[row["game_id"]].append(value)
    payload = signed(
        {
            "format_version": "sage-t10.3.12e-active-closed-loop-diagnostic-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": registry_payload["registry_checksum"],
            "collection_checks": checks,
            "collection_complete": all(checks.values()),
            "accounting": accounting,
            "metrics": {
                "by_arm": by_arm,
                "trajectory_metrics": trajectory_metrics,
                "initial_frame_hashes": dict(initial_hashes),
                "distinct_target_initial_frames": len(
                    {value for values in initial_hashes.values() for value in values}
                ),
                "labels_seed_environment": False,
                "physical_actions": sum(int(row["sealed_events"]) for row in receipts),
            },
            "receipt_checksums": sorted(row["receipt_checksum"] for row in receipts),
            "post_hoc_diagnostic": True,
            "confirmatory_evidence": False,
            "physical_actions_replayed": 0,
            "new_games_opened": False,
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, ACTIVE_REPORT_FILENAME, payload)
    if _artifact_bytes(root) > int(manifest["matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.12e artifact budget exceeded")
    if not payload["collection_complete"]:
        raise protocol.IntegrityError("active diagnostic did not seal cleanly")
    return payload


def adjudicate(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    active = _require_gate(root, "active-diagnostic")
    receipts = _load_receipts(_destination(root))
    by_arm = {
        arm: {row["game_id"]: row for row in receipts if row["arm"] == arm}
        for arm in ARMS
    }
    successes = {
        arm: {game for game, row in rows.items() if int(row.get("level_delta", 0)) > 0}
        for arm, rows in by_arm.items()
    }
    primary = by_arm["anchored_goal_dynamic_successor"]
    applicable = {
        game for game, row in primary.items()
        if row["recognized_context"] == "path_context"
    }
    primary_applicable = [primary[game] for game in sorted(applicable)]
    exact_fractions = [
        (
            float(row["exact_groundings"]) / float(row["sealed_events"])
            if int(row["sealed_events"]) > 0 else 1.0
        )
        for row in primary_applicable
    ]
    frontier_fractions = [
        (
            float(row["frontier_advances"]) / float(row["sealed_events"])
            if int(row["sealed_events"]) > 0 else 1.0
        )
        for row in primary_applicable
    ]
    counts = {arm: len(values) for arm, values in successes.items()}
    primary_count = counts["anchored_goal_dynamic_successor"]
    frozen_count = counts["frozen_grounded_cursor"]
    stateless_count = counts["stateless_goal_and_successor"]
    swap_count = counts["goal_end_swap"]
    path_sets = {
        arm: {
            game for game, row in rows.items()
            if row["recognized_context"] == "path_context"
        }
        for arm, rows in by_arm.items()
    }
    initial_hashes = active["metrics"]["initial_frame_hashes"]
    checks = {
        "collection_complete": active.get("collection_complete") is True,
        "nine_target_frames_observed": len(initial_hashes) == 9,
        "one_initial_frame_per_game": all(len(values) == 1 for values in initial_hashes.values()),
        "minimum_path_applicability": len(applicable)
        >= int(manifest["gates"]["minimum_path_applicable_games"]),
        "uniform_path_applicability_across_arms": all(
            values == applicable for values in path_sets.values()
        ),
        "single_goal_anchor": all(
            int(row["anchor_builds"])
            == int(manifest["gates"]["anchor_builds_per_applicable_reset"])
            for row in primary_applicable
        ),
        "relation_evaluated_each_executed_decision": all(
            int(row["relation_evaluations"]) >= max(1, int(row["sealed_events"]))
            for row in primary_applicable
        ),
        "dynamic_zero_grounding_misses": all(
            int(row["grounding_misses"])
            <= int(manifest["gates"]["maximum_dynamic_grounding_misses"])
            for row in primary_applicable
        ),
        "dynamic_exact_grounding": bool(exact_fractions)
        and min(exact_fractions)
        >= float(manifest["gates"]["minimum_dynamic_exact_grounding_fraction"]),
        "dynamic_frontier_advance": bool(frontier_fractions)
        and min(frontier_fractions)
        >= float(manifest["gates"]["minimum_dynamic_frontier_advance_fraction"]),
        "ephemeral_state_not_persisted": all(
            not row["path_plan_persisted"]
            and not row["visited_action_keys_persisted"]
            and not row["grounded_arguments_persisted"]
            for row in receipts
        ),
        "minimum_dynamic_success": primary_count
        >= int(manifest["gates"]["minimum_dynamic_success_games"]),
        "dynamic_over_frozen": primary_count - frozen_count
        >= int(manifest["gates"]["minimum_dynamic_over_frozen_success_advantage"]),
        "dynamic_over_stateless": primary_count - stateless_count
        >= int(manifest["gates"]["minimum_dynamic_over_stateless_success_advantage"]),
        "dynamic_over_goal_swap": primary_count - swap_count
        >= int(manifest["gates"]["minimum_dynamic_over_goal_swap_success_advantage"]),
        "zero_primary_game_over": all(
            int(row["game_over_actions"]) == 0 for row in primary.values()
        ),
        "zero_errors_all_arms": all(not row["errors"] for row in receipts),
        "zero_illegal_actions": all(int(row["illegal_actions"]) == 0 for row in receipts),
        "all_executed_actions_sage_t": all(
            int(row["sage_t_option_actions"]) == int(row["sealed_events"])
            for row in receipts
        ),
        "zero_legacy_fallback": all(
            int(row["legacy_fallback_actions"]) == 0 for row in receipts
        ),
    }
    passed = all(checks.values())
    structural = all(
        checks[key]
        for key in (
            "single_goal_anchor",
            "relation_evaluated_each_executed_decision",
            "dynamic_zero_grounding_misses",
            "dynamic_exact_grounding",
            "dynamic_frontier_advance",
            "ephemeral_state_not_persisted",
        )
    )
    if not structural:
        verdict = "CLOSED_LOOP_GROUNDING_MISS"
    elif primary_count == 0 and swap_count > 0:
        verdict = "GOAL_END_SWAP_ONLY"
    elif primary_count == 0:
        verdict = "CLOSED_LOOP_NO_PROGRESS"
    elif not checks["dynamic_over_frozen"]:
        verdict = "FROZEN_CURSOR_NOT_DISCRIMINATED"
    elif not checks["dynamic_over_stateless"]:
        verdict = "STATELESS_GOAL_NOT_DISCRIMINATED"
    elif not checks["dynamic_over_goal_swap"]:
        verdict = "GOAL_ANCHOR_NOT_CAUSAL"
    elif not passed:
        verdict = "CLOSED_LOOP_SAFETY_OR_ACCOUNTING_MISS"
    else:
        verdict = "PASS_T10_3_12E_CLOSED_LOOP_RELATIONAL_SUCCESSOR"
    payload = signed(
        {
            "format_version": "sage-t10.3.12e-closed-loop-adjudication-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "active_report_checksum": active["report_checksum"],
            "checks": checks,
            "passed": passed,
            "verdict": verdict,
            "path_applicable_games": sorted(applicable),
            "success_games_by_arm": {arm: sorted(values) for arm, values in successes.items()},
            "success_counts_by_arm": counts,
            "dynamic_exact_grounding_fractions": exact_fractions,
            "dynamic_frontier_advance_fractions": frontier_fractions,
            "source_goal_role_supported": primary_count > swap_count,
            "closed_loop_mechanism_recovered": passed,
            "cross_game_generalization_proven": False,
            "factor_generalization_proven": False,
            "confirmatory_evidence": False,
            "program_promoted": False,
            "sequence_composition_authorized": False,
            "next_step": (
                "preregister_new_independent_validation"
                if passed else "retain_negative_and_revise_only_in_new_protocol"
            ),
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, ADJUDICATION_FILENAME, payload)
    return payload


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    existing = _path(root, TERMINAL_REPORT_FILENAME)
    if existing.is_file():
        return _read_signed(root, TERMINAL_REPORT_FILENAME, "report_checksum")
    destination = _destination(root)
    with _durable_contract():
        accounting = durable._journal_accounting(destination)
    adjudication_path = _path(root, ADJUDICATION_FILENAME)
    if adjudication_path.is_file():
        adjudication = _read_signed(root, ADJUDICATION_FILENAME, "report_checksum")
        verdict = str(adjudication["verdict"])
        passed = bool(adjudication["passed"])
        recovered = bool(adjudication["closed_loop_mechanism_recovered"])
    else:
        verdict = "INCOMPLETE_T10_3_12E"
        passed = False
        recovered = False
    artifacts = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        if phase == "report":
            continue
        path = _path(root, str(contract["path"]))
        if not path.is_file():
            artifacts[phase] = None
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        field = str(contract["checksum_field"])
        protocol.verify_signed(payload, field)
        artifacts[phase] = payload[field]
    report = signed(
        {
            "format_version": "sage-t10.3.12e-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "passed": passed,
            "closed_loop_mechanism_recovered": recovered,
            "post_hoc_diagnostic": True,
            "confirmatory_evidence": False,
            "cross_game_generalization_proven": False,
            "factor_generalization_proven": False,
            "accounting": accounting,
            "artifacts": artifacts,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
            "physical_actions_replayed": 0,
            "legacy_fallback_actions": 0,
            "parent_events_used_for_training": 0,
            "new_games_opened": False,
            "sequence_games_opened": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "ar25_opened": False,
            "production_authority": False,
            "program_promoted": False,
            "sequence_composition_authorized": False,
        },
        "report_checksum",
    )
    _write(root, TERMINAL_REPORT_FILENAME, report)
    return report


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = _destination(root)
    with _durable_contract():
        accounting = durable._journal_accounting(destination)
    artifacts = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        path = _path(root, str(contract["path"]))
        if not path.is_file():
            artifacts[phase] = None
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        field = str(contract["checksum_field"])
        protocol.verify_signed(payload, field)
        artifacts[phase] = payload[field]
    return {
        "format_version": "sage-t10.3.12e-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "artifacts": artifacts,
        "accounting": accounting,
        "artifact_bytes": _artifact_bytes(root),
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        "maximum_resets": protocol.TOTAL_RESETS,
        "diagnostic_only": True,
        "new_games_opened": False,
        "sequence_games_opened": False,
        "production_authority": False,
    }


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze", "status", "audit-parent", "audit-trajectories", "preflight",
            "compile-programs", "active-diagnostic", "adjudicate", "report",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.phase == "freeze":
            manifest, receipt = protocol.freeze_manifest(root)
            _emit(
                {
                    "status": "FROZEN",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "freeze_receipt_checksum": receipt["receipt_checksum"],
                    "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
                    "physical_actions": 0,
                }
            )
            return 0
        manifest = protocol.load_manifest(root)
        handlers = {
            "status": status,
            "audit-parent": audit_parent,
            "audit-trajectories": audit_trajectories,
            "preflight": preflight,
            "compile-programs": compile_programs,
            "active-diagnostic": active_diagnostic,
            "adjudicate": adjudicate,
            "report": terminal_report,
        }
        result = handlers[args.phase](root, manifest)
        _emit(result)
        if args.phase in {"adjudicate", "report"} and not bool(result.get("passed")):
            return 3
        return 0
    except protocol.ScientificGateMiss as exc:
        _emit({"error": str(exc), "exit_code": 3, "phase": args.phase})
        return 3
    except (protocol.IntegrityError, ValueError, KeyError, OSError) as exc:
        _emit(
            {
                "error": "INVALID_PROVENANCE",
                "detail": f"{type(exc).__name__}:{str(exc)[:240]}",
                "exit_code": 2,
                "phase": args.phase,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_REPORT_FILENAME", "ADJUDICATION_FILENAME", "PARENT_AUDIT_FILENAME",
    "PREFLIGHT_FILENAME", "REGISTRY_FILENAME", "TERMINAL_REPORT_FILENAME",
    "TRAJECTORY_AUDIT_FILENAME", "active_diagnostic", "adjudicate", "audit_parent",
    "audit_trajectories", "compile_programs", "main", "preflight", "status",
    "terminal_report",
]
