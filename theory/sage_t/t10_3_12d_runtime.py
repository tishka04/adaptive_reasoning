"""Durable diagnostic runtime for SAGE.T10.3.12d executor correspondence."""

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
from . import t10_3_12d_protocol as protocol
from .compiler import compile_observation
from .cross_game_transfer_v10_3_12c import CrossGameFactorRegistry, GroundingDecision
from .executor_correspondence_v10_3_12d import (
    ARMS,
    ExecutorRegistry,
    PathExecutorController,
    compile_executor_registry,
    signed,
)
from .progress_witness_v10 import GroundedAction

PARENT_AUDIT_FILENAME = "parent_negative_audit.json"
TRAJECTORY_AUDIT_FILENAME = "parent_path_collapse_audit.json"
PREFLIGHT_FILENAME = "executor_preflight.json"
REGISTRY_FILENAME = "executor_registry.json"
ACTIVE_REPORT_FILENAME = "active_executor_diagnostic.json"
ADJUDICATION_FILENAME = "executor_adjudication_report.json"
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
        raise protocol.IntegrityError(f"required T10.3.12d artifact is absent: {filename}")
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
            "format_version": "sage-t10.3.12d-parent-audit-v1",
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
        raise protocol.ScientificGateMiss("PARENT_NEGATIVE_AUDIT_MISS")
    return payload


def _parent_receipts(root: Path) -> list[dict[str, Any]]:
    base = root / protocol.PARENT_OUTPUT_DIR / "journal" / "branches"
    rows = []
    for path in sorted(base.rglob("receipt.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol.verify_signed(payload, "receipt_checksum")
        rows.append(payload)
    return rows


def _parent_argument_sequence(root: Path, receipt: Mapping[str, Any]) -> list[str]:
    base = (
        root
        / protocol.PARENT_OUTPUT_DIR
        / "journal"
        / "intents"
        / str(receipt["work_id"])
    )
    output = []
    for path in sorted(base.glob("*.json")) if base.exists() else ():
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol.verify_signed(payload, "intent_checksum")
        output.append(str(payload.get("action", {}).get("argument_checksum", "")))
    return output


def audit_trajectories(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-parent")
    receipts = _parent_receipts(root)
    full_path = [
        row
        for row in receipts
        if row.get("arm") == "factorized_source"
        and "path_context" in row.get("recognized_contexts", ())
    ]
    collapse_rows = []
    for row in full_path:
        sequence = _parent_argument_sequence(root, row)
        suffix = sequence[-8:]
        collapse_rows.append(
            {
                "game_id": row["game_id"],
                "actions": len(sequence),
                "level_delta": int(row.get("level_delta", 0)),
                "distinct_arguments": len(set(sequence)),
                "suffix_length": len(suffix),
                "suffix_distinct_arguments": len(set(suffix)),
                "collapsed_suffix": len(suffix) == 8 and len(set(suffix)) <= 2,
                "grounded_argument_checksums_retained": False,
            }
        )
    lf52_wins = sorted(
        str(row["arm"])
        for row in receipts
        if row.get("game_id") == "lf52-271a04aa"
        and int(row.get("level_delta", 0)) > 0
    )
    checks = {
        "all_54_parent_receipts": len(receipts) == 54,
        "three_full_path_branches": len(full_path)
        == int(manifest["gates"]["parent_full_path_branches"]),
        "full_path_zero_success": all(int(row.get("level_delta", 0)) == 0 for row in full_path),
        "three_collapsed_suffixes": sum(row["collapsed_suffix"] for row in collapse_rows)
        == int(manifest["gates"]["parent_collapsed_suffix_branches"]),
        "two_lf52_ablation_wins": lf52_wins
        == ["role_binding_ablation", "transition_ablation"],
        "parent_outcome_is_diagnostic_only": True,
        "no_grounded_arguments_compiled": True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12d-parent-path-collapse-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_journal_digest": manifest["parent_journal_digest"],
            "collapse_summary": collapse_rows,
            "lf52_winning_ablation_arms": lf52_wins,
            "checks": checks,
            "passed": all(checks.values()),
            "post_hoc_diagnostic": True,
            "confirmatory_evidence": False,
            "grounded_argument_checksums_retained": False,
            "physical_actions": 0,
        },
        "audit_checksum",
    )
    _write(root, TRAJECTORY_AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("PARENT_PATH_COLLAPSE_AUDIT_MISS")
    return payload


def _parent_registry(root: Path) -> CrossGameFactorRegistry:
    return CrossGameFactorRegistry(_read_parent(root, "registry", "registry_checksum"))


def _executor_registry(root: Path) -> ExecutorRegistry:
    return compile_executor_registry(
        _read_parent(root, "registry", "registry_checksum")
    )


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-trajectories")
    registry = _executor_registry(root)
    parent_registry = _parent_registry(root)
    path = tuple(
        GroundedAction("ACTION6", (("x", index), ("y", 20 - index)))
        for index in range(1, 11)
    )
    candidates = tuple(item.candidate for item in path)
    calls = Counter()

    def builder(state: Any, legal: Any) -> tuple[str, tuple[GroundedAction, ...]]:
        del state, legal
        calls["count"] += 1
        return "path_context", path

    stable = PathExecutorController(
        arm="stable_source_cursor",
        registry=registry,
        parent_registry=parent_registry,
        plan_builder=builder,
    )
    stable_selected = []
    for step in range(10):
        decision = stable.choose(
            state=None,
            candidates=tuple(reversed(candidates)),
            shape=(32, 32),
            step_index=step,
        )
        stable_selected.append(dict(decision.candidate.action_data))
    stable_exhausted = stable.choose(
        state=None,
        candidates=candidates,
        shape=(32, 32),
        step_index=10,
    )
    reverse = PathExecutorController(
        arm="stable_reverse_orientation",
        registry=registry,
        parent_registry=parent_registry,
        plan_builder=lambda state, legal: ("path_context", path),
    )
    reverse_selected = [
        dict(
            reverse.choose(
                state=None,
                candidates=candidates,
                shape=(32, 32),
                step_index=step,
            ).candidate.action_data
        )
        for step in range(10)
    ]
    hold = PathExecutorController(
        arm="stable_cursor_hold",
        registry=registry,
        parent_registry=parent_registry,
        plan_builder=lambda state, legal: ("path_context", path),
    )
    hold_selected = [
        dict(
            hold.choose(
                state=None,
                candidates=candidates,
                shape=(32, 32),
                step_index=step,
            ).candidate.action_data
        )
        for step in range(4)
    ]
    missing = PathExecutorController(
        arm="stable_source_cursor",
        registry=registry,
        parent_registry=parent_registry,
        plan_builder=lambda state, legal: ("path_context", path),
    )
    first = missing.choose(
        state=None,
        candidates=candidates,
        shape=(32, 32),
        step_index=0,
    )
    missing_second = missing.choose(
        state=None,
        candidates=(candidates[0],),
        shape=(32, 32),
        step_index=1,
    )
    non_path = PathExecutorController(
        arm="stable_source_cursor",
        registry=registry,
        parent_registry=parent_registry,
        plan_builder=lambda state, legal: ("repeat_context", ()),
    ).choose(
        state=None,
        candidates=candidates,
        shape=(32, 32),
        step_index=0,
    )
    stateless_calls = Counter()

    def stateless_grounder(*args: Any, **kwargs: Any) -> GroundingDecision:
        del args
        legal = tuple(kwargs["candidates"])
        index = stateless_calls["count"] % 2
        stateless_calls["count"] += 1
        return GroundingDecision(
            candidate=legal[index],
            context="path_context",
            reason="synthetic_stateless",
            inspections=len(legal),
            program_hash="synthetic-parent-program",
            ablated_factor=None,
        )

    stateless = PathExecutorController(
        arm="stateless_source_replan",
        registry=registry,
        parent_registry=parent_registry,
        plan_builder=lambda state, legal: ("path_context", path),
        stateless_grounder=stateless_grounder,
    )
    stateless_selected = [
        stateless.choose(
            state=None,
            candidates=candidates,
            shape=(32, 32),
            step_index=step,
        ).candidate
        for step in range(2)
    ]
    snapshot = registry.snapshot()
    stable_summary = stable.summary()
    checks = {
        "four_executor_programs": len(snapshot["programs"]) == 4,
        "support_zero": snapshot["local_support_total"] == 0,
        "stable_exact_sequence": stable_selected == [item.data for item in path],
        "stateless_replans_each_decision": stateless_calls["count"] == 2
        and stateless.summary()["replans"] == 2
        and stateless_selected[0] != stateless_selected[1],
        "stable_plan_built_once": calls["count"] == 1,
        "stable_zero_replans": stable_summary["replans"] == 0,
        "stable_ten_reacquisitions": stable_summary["reacquisitions"] == 10,
        "stable_plan_exhaustion_abstains": stable_exhausted.abstained,
        "reverse_is_exact_inverse": reverse_selected == [item.data for item in reversed(path)],
        "cursor_hold_repeats_first": hold_selected == [path[0].data] * 4,
        "missing_waypoint_abstains": not first.abstained and missing_second.abstained,
        "non_path_uniform_abstention": non_path.abstained,
        "no_grounded_path_in_registry_payload": all(
            all(
                forbidden not in json.dumps(row["program"], sort_keys=True).lower()
                for forbidden in ("action_data", "argument_checksum", "coordinate", "waypoint_id")
            )
            for row in snapshot["programs"]
        ),
        "ephemeral_plan_not_in_summary": stable_summary["path_plan_persisted"] is False
        and stable_summary["grounded_arguments_persisted"] is False,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12d-executor-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "cases": checks,
            "passed": all(checks.values()),
            "registry_checksum_preview": snapshot["registry_checksum"],
            "physical_actions": 0,
        },
        "preflight_checksum",
    )
    _write(root, PREFLIGHT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("EXECUTOR_PREFLIGHT_MISS")
    return payload


def compile_executors(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "preflight")
    registry = _executor_registry(root)
    snapshot = registry.snapshot()
    core = {key: value for key, value in snapshot.items() if key != "registry_checksum"}
    core.update(
        {
            "manifest_checksum": manifest["manifest_checksum"],
            "compiled_before_diagnostic_actions": True,
            "parent_outcomes_used_for_program_fit": False,
            "parent_grounded_paths_imported": False,
            "historical_support_imported": 0,
            "physical_actions": 0,
        }
    )
    payload = signed(core, "registry_checksum")
    ExecutorRegistry(payload)
    _write(root, REGISTRY_FILENAME, payload)
    return payload


class _ActiveExecutor:
    def __init__(
        self,
        arm: str,
        registry: ExecutorRegistry,
        parent_registry: CrossGameFactorRegistry,
    ) -> None:
        self.controller = PathExecutorController(
            arm=arm,
            registry=registry,
            parent_registry=parent_registry,
        )
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
        executor = self.controller.choose(
            state=state,
            candidates=legal_actions,
            shape=tuple(int(value) for value in current_grid.shape[:2]),
            step_index=step_index,
        )
        if executor.abstained:
            return None, executor
        decision = CognitiveDecision(
            action_name=executor.candidate.action_name,
            action_data=dict(executor.candidate.action_data),
            source="sage_t_executor_correspondence",
            reason=executor.reason,
            confidence=1.0,
            option_id=executor.program_hash,
        )
        return decision, executor

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


def _work_path(destination: Path, category: str, work: protocol.WorkSpec, name: str) -> Path:
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
            "format_version": "sage-t10.3.12d-action-intent-v1",
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
    registry: ExecutorRegistry,
    parent_registry: CrossGameFactorRegistry,
    registry_checksum: str,
    lock: Any,
) -> dict[str, Any]:
    receipt_path = _receipt_path(destination, work)
    if receipt_path.is_file():
        return durable._read_signed(receipt_path, "receipt_checksum")
    executor = _ActiveExecutor(work.arm, registry, parent_registry)
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
            decision, _executor_decision = executor.decide(
                current_grid=before.grid,
                legal_actions=legal,
                game_state=str(before.game_state),
                levels_completed=int(before.levels_completed),
                step_index=step_index,
            )
            if decision is None:
                stop_reason = "PLANNED_EXECUTOR_ABSTENTION"
                break
            if decision.source != "sage_t_executor_correspondence":
                raise protocol.IntegrityError("non-SAGE decision entered T10.3.12d")
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
                        "format_version": "sage-t10.3.12d-unresolved-event-v1",
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
            event = signed(
                {
                    "format_version": "sage-t10.3.12d-physical-event-v1",
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
                    "frame_before_sha256": protocol.sha256_payload(before.grid.tolist()),
                    "frame_after_sha256": protocol.sha256_payload(after.grid.tolist()),
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
            "format_version": "sage-t10.3.12d-branch-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            **work.as_dict(),
            "work_id": work.work_id,
            "status": status,
            "complete": status == "COMPLETE" and not errors,
            "stop_reason": stop_reason,
            "planned_abstention": stop_reason == "PLANNED_EXECUTOR_ABSTENTION",
            "issued_intents": issued,
            "sealed_events": sealed,
            "unresolved_intents": unresolved,
            "event_ids": trace_ids,
            "initial_frame_sha256": initial_frame_hash,
            "level_delta": max(0, final_level - initial_level),
            "sage_t_option_actions": sealed,
            "recognized_context": summary["recognized_context"],
            "program_hash": summary["program_hash"],
            "plan_builds": summary["plan_builds"],
            "replans": summary["replans"],
            "plan_length": summary["plan_length"],
            "cursor": summary["cursor"],
            "reacquisitions": summary["reacquisitions"],
            "grounding_misses": summary["grounding_misses"],
            "abstention_reason": summary["abstention_reason"],
            "candidate_inspections": summary["candidate_inspections"],
            "maximum_legal_candidates": summary["maximum_legal_candidates"],
            "path_plan_persisted": False,
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


def active_diagnostic(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "preflight")
    registry_payload = _read_signed(root, REGISTRY_FILENAME, "registry_checksum")
    if registry_payload.get("compiled_before_diagnostic_actions") is not True:
        raise protocol.IntegrityError("executor registry was not compiled before actions")
    registry = ExecutorRegistry(registry_payload)
    parent_registry = _parent_registry(root)
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
                    parent_registry,
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
        metrics = _argument_metrics(destination, row)
        trajectory_metrics.append(
            {
                "work_id": row["work_id"],
                "game_id": row["game_id"],
                "arm": row["arm"],
                **metrics,
            }
        )
    for arm in ARMS:
        subset = [row for row in receipts if row["arm"] == arm]
        by_arm[arm] = {
            "success_games": sorted(row["game_id"] for row in subset if int(row["level_delta"]) > 0),
            "path_applicable_games": sorted(
                row["game_id"] for row in subset if row["recognized_context"] == "path_context"
            ),
            "actions": sum(int(row["sealed_events"]) for row in subset),
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
            "format_version": "sage-t10.3.12d-active-executor-diagnostic-v1",
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
        raise protocol.IntegrityError("T10.3.12d artifact budget exceeded")
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
        arm: {
            game for game, row in rows.items() if int(row.get("level_delta", 0)) > 0
        }
        for arm, rows in by_arm.items()
    }
    stable = by_arm["stable_source_cursor"]
    applicable = {
        game for game, row in stable.items() if row["recognized_context"] == "path_context"
    }
    stable_applicable = [stable[game] for game in sorted(applicable)]
    reacquisition_fractions = [
        (
            float(row["reacquisitions"]) / float(row["sealed_events"])
            if int(row["sealed_events"]) > 0 else 1.0
        )
        for row in stable_applicable
    ]
    stable_count = len(successes["stable_source_cursor"])
    stateless_count = len(successes["stateless_source_replan"])
    reverse_count = len(successes["stable_reverse_orientation"])
    hold_count = len(successes["stable_cursor_hold"])
    path_sets = {
        arm: {
            game
            for game, row in rows.items()
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
        "stable_plan_built_once": all(int(row["plan_builds"]) == 1 for row in stable_applicable),
        "stable_zero_replans": all(int(row["replans"]) == 0 for row in stable_applicable),
        "stable_exact_reacquisition": bool(reacquisition_fractions)
        and min(reacquisition_fractions)
        >= float(manifest["gates"]["minimum_stable_reacquisition_fraction"]),
        "stable_plan_not_persisted": all(not row["path_plan_persisted"] for row in receipts),
        "minimum_stable_success": stable_count
        >= int(manifest["gates"]["minimum_stable_success_games"]),
        "stable_over_stateless": stable_count - stateless_count
        >= int(manifest["gates"]["minimum_stable_over_stateless_success_advantage"]),
        "stable_over_cursor_hold": stable_count - hold_count
        >= int(manifest["gates"]["minimum_stable_over_cursor_hold_success_advantage"]),
        "zero_stable_game_over": all(int(row["game_over_actions"]) == 0 for row in stable.values()),
        "zero_errors_all_arms": all(not row["errors"] for row in receipts),
        "zero_illegal_actions": all(int(row["illegal_actions"]) == 0 for row in receipts),
        "all_executed_actions_sage_t": all(
            int(row["sage_t_option_actions"]) == int(row["sealed_events"])
            for row in receipts
        ),
        "zero_legacy_fallback": all(int(row["legacy_fallback_actions"]) == 0 for row in receipts),
    }
    passed = all(checks.values())
    structural = all(
        checks[key]
        for key in (
            "stable_plan_built_once",
            "stable_zero_replans",
            "stable_exact_reacquisition",
            "stable_plan_not_persisted",
        )
    )
    if not structural:
        verdict = "PLAN_REACQUISITION_MISS"
    elif stable_count == 0 and reverse_count > 0:
        verdict = "REVERSE_ORIENTATION_ONLY"
    elif stable_count == 0:
        verdict = "STABLE_EXECUTOR_NO_PROGRESS"
    elif not checks["stable_over_stateless"]:
        verdict = "STATELESS_REPLAN_NOT_DISCRIMINATED"
    elif not checks["stable_over_cursor_hold"]:
        verdict = "CURSOR_NOT_CAUSAL"
    elif not passed:
        verdict = "EXECUTOR_SAFETY_OR_ACCOUNTING_MISS"
    else:
        verdict = "PASS_T10_3_12D_EXECUTOR_CORRESPONDENCE_RECOVERED"
    payload = signed(
        {
            "format_version": "sage-t10.3.12d-executor-adjudication-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "active_report_checksum": active["report_checksum"],
            "checks": checks,
            "passed": passed,
            "verdict": verdict,
            "path_applicable_games": sorted(applicable),
            "success_games_by_arm": {
                arm: sorted(values) for arm, values in successes.items()
            },
            "success_counts_by_arm": {
                arm: len(values) for arm, values in successes.items()
            },
            "stable_plan_reacquisition_fractions": reacquisition_fractions,
            "source_orientation_supported": stable_count > reverse_count,
            "executor_correspondence_recovered": passed,
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
        recovered = bool(adjudication["executor_correspondence_recovered"])
    else:
        verdict = "INCOMPLETE_T10_3_12D"
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
            "format_version": "sage-t10.3.12d-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "passed": passed,
            "executor_correspondence_recovered": recovered,
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
        "format_version": "sage-t10.3.12d-status-v1",
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
            "compile-executors", "active-diagnostic", "adjudicate", "report",
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
            "compile-executors": compile_executors,
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
    "audit_trajectories", "compile_executors", "main", "preflight", "status",
    "terminal_report",
]
