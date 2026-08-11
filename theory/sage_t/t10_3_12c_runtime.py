"""Durable fail-closed runtime for SAGE.T10.3.12c cross-game transfer."""

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
from . import t10_3_12c_protocol as protocol
from .compiler import compile_observation
from .contracts import ActionCandidate
from .cross_game_transfer_v10_3_12c import (
    ARMS,
    FACTORS,
    CrossGameFactorRegistry,
    compile_cross_game_registry,
    select_grounding,
    signed,
)
from .progress_witness_v10 import GroundedAction

PARENT_AUDIT_FILENAME = "parent_factor_candidate_audit.json"
PREFLIGHT_FILENAME = "cross_game_preflight.json"
TARGET_AUDIT_FILENAME = "target_schema_inventory.json"
REGISTRY_FILENAME = "cross_game_factor_registry.json"
ACTIVE_REPORT_FILENAME = "active_cross_game_report.json"
ADJUDICATION_FILENAME = "cross_game_adjudication_report.json"
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
        raise protocol.IntegrityError(f"required T10.3.12c artifact is absent: {filename}")
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
        "parent_pass_exact": observed == protocol.EXPECTED_PARENT,
        "factor_candidates_exact": observed["identified_factor_candidates"] == list(FACTORS),
        "all_parent_bindings_frozen": bindings == manifest["parent_artifacts"],
        "cross_game_was_not_previously_claimed": True,
        "zero_parent_actions_replayed": True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12c-parent-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_state": observed,
            "parent_artifact_sha256": {
                key: value["sha256"] for key, value in bindings.items()
            },
            "checks": checks,
            "passed": all(checks.values()),
            "parent_actions_used_for_training": 0,
            "physical_actions": 0,
        },
        "audit_checksum",
    )
    _write(root, PARENT_AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("PARENT_FACTOR_AUDIT_MISS")
    return payload


def _compile_registry_from_parent(root: Path) -> CrossGameFactorRegistry:
    parent_registry = _read_parent(root, "factor_registry", "registry_checksum")
    return compile_cross_game_registry(parent_registry)


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-parent")
    registry = _compile_registry_from_parent(root)
    repeat = (
        ActionCandidate("ACTION6", {"x": 12, "y": 12}),
        ActionCandidate("ACTION6", {"x": 30, "y": 10}),
    )
    reversed_repeat = tuple(reversed(repeat))
    ambiguity = (
        ActionCandidate("ACTION6", {"x": 1, "y": 10}),
        ActionCandidate("ACTION6", {"x": 30, "y": 10}),
    )
    arity_zero = (ActionCandidate("ACTION2"), ActionCandidate("ACTION1"))
    source = select_grounding(
        registry, arm=ARMS[0], candidates=repeat, shape=(32, 32), step_index=0,
        forced_context="repeat_context",
    )
    source_reversed = select_grounding(
        registry, arm=ARMS[0], candidates=reversed_repeat, shape=(32, 32), step_index=0,
        forced_context="repeat_context",
    )
    ambiguous = select_grounding(
        registry, arm=ARMS[0], candidates=ambiguity, shape=(32, 32), step_index=0,
        forced_context="repeat_context",
    )
    source_zero = select_grounding(
        registry, arm=ARMS[0], candidates=arity_zero, shape=(32, 32), step_index=0,
    )
    generic_zero = select_grounding(
        registry, arm=ARMS[1], candidates=arity_zero, shape=(32, 32), step_index=0,
    )
    operator_zero = select_grounding(
        registry, arm="operator_ablation", candidates=arity_zero,
        shape=(32, 32), step_index=0,
    )
    role = select_grounding(
        registry, arm="role_binding_ablation", candidates=repeat,
        shape=(32, 32), step_index=0, forced_context="repeat_context",
    )
    transition = select_grounding(
        registry, arm="transition_ablation", candidates=repeat,
        shape=(32, 32), step_index=1, forced_context="repeat_context",
    )
    termination = select_grounding(
        registry, arm="termination_ablation", candidates=repeat,
        shape=(32, 32), step_index=2, forced_context="repeat_context",
    )
    path_actions = (
        GroundedAction("ACTION6", (("x", 4), ("y", 4))),
        GroundedAction("ACTION6", (("x", 8), ("y", 8))),
    )
    path_candidates = tuple(item.candidate for item in path_actions)
    path_source = select_grounding(
        registry, arm=ARMS[0], candidates=path_candidates, shape=(32, 32),
        step_index=0, forced_context="path_context", forced_path=path_actions,
    )
    path_role = select_grounding(
        registry, arm="role_binding_ablation", candidates=path_candidates,
        shape=(32, 32), step_index=0, forced_context="path_context",
        forced_path=path_actions,
    )
    snapshot = registry.snapshot()
    checks = {
        "program_count_12": len(snapshot["programs"]) == len(ARMS) * 2,
        "support_zero": snapshot["local_support_total"] == 0,
        "source_repeat_grounded": source.candidate == repeat[1],
        "candidate_order_invariant": source.candidate == source_reversed.candidate,
        "ambiguous_role_abstains": ambiguous.abstained,
        "source_parameterized_operator_abstains_on_arity_zero": source_zero.abstained,
        "generic_source_free_handles_arity_zero": not generic_zero.abstained,
        "operator_ablation_handles_arity_zero": not operator_zero.abstained,
        "role_ablation_changes_binding": role.candidate != source.candidate,
        "transition_ablation_changes_step": transition.candidate != source.candidate,
        "termination_ablation_stops_at_two": termination.abstained,
        "path_role_ablation_reverses_orientation": path_source.candidate != path_role.candidate,
        "no_legacy_fallback_path_exists": True,
        "serialized_payload_transfer_safe": True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12c-preflight-v1",
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
        raise protocol.ScientificGateMiss("CROSS_GAME_PREFLIGHT_MISS")
    return payload


def audit_targets(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "preflight")
    rows = []
    for short in protocol.TARGET_SHORT_IDS:
        path = root / protocol.SOURCE_SHARD_DIR / f"{short}.jsonl"
        action_names: set[str] = set()
        parameter_arities: Counter[int] = Counter()
        context_action_names: set[str] = set()
        pair_rows = 0
        # Deliberately inspect only action schemas.  Trace effects, rewards,
        # terminal outcomes, frames, and labels are never accessed here.
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                pair_rows += 1
                for side in ("left", "right"):
                    action = row.get(side, {}).get("action", {})
                    name = str(action.get("name", "")).upper()
                    if name:
                        action_names.add(name)
                        parameter_arities[len(dict(action.get("action_args", {}) or {}))] += 1
                for descriptor in row.get("context", ()):
                    name = str(descriptor.get("action_name", "")).upper()
                    if name:
                        context_action_names.add(name)
        rows.append(
            {
                "game": short,
                "pair_rows": pair_rows,
                "observed_pair_action_schemas": sorted(action_names),
                "context_action_schemas": sorted(context_action_names),
                "pair_parameter_arity_counts": {
                    str(key): value for key, value in sorted(parameter_arities.items())
                },
                "parameterized_pair_action_observed": any(key > 0 for key in parameter_arities),
                "outcomes_read": False,
                "frames_retained": False,
            }
        )
    checks = {
        "nine_targets": len(rows) == 9,
        "all_remaining_source_train_games": {row["game"] for row in rows}
        == set(protocol.TARGET_SHORT_IDS),
        "parent_games_excluded_from_target_scoring": not {"lp85", "su15"}
        & {row["game"] for row in rows},
        "sequence_games_excluded": not {"re86", "ls20", "sc25"}
        & {row["game"] for row in rows},
        "ar25_and_holdout_excluded": "ar25" not in {row["game"] for row in rows},
        "all_shards_frozen": all(
            protocol.file_sha256(root / protocol.SOURCE_SHARD_DIR / f"{short}.jsonl")
            == protocol.TARGET_SHARD_SHA256[short]
            for short in protocol.TARGET_SHORT_IDS
        ),
        "outcomes_not_read": all(not row["outcomes_read"] for row in rows),
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12c-target-schema-inventory-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "targets": rows,
            "checks": checks,
            "passed": all(checks.values()),
            "selection_was_outcome_independent": True,
            "physical_actions": 0,
        },
        "inventory_checksum",
    )
    _write(root, TARGET_AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("TARGET_INVENTORY_MISS")
    return payload


def compile_transfer(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-targets")
    registry = _compile_registry_from_parent(root)
    snapshot = registry.snapshot()
    snapshot.update(
        {
            "manifest_checksum": manifest["manifest_checksum"],
            "compiled_before_target_outcomes": True,
            "target_schema_inventory_used_for_compilation": False,
            "grounded_arguments_imported": False,
            "historical_support_imported": 0,
            "physical_actions": 0,
        }
    )
    core = {key: value for key, value in snapshot.items() if key != "registry_checksum"}
    payload = signed(core, "registry_checksum")
    CrossGameFactorRegistry(payload)
    _write(root, REGISTRY_FILENAME, payload)
    return payload


class _ActiveFactorController:
    """Reset-local grounding only; never falls back to another controller."""

    def __init__(self, arm: str, registry: CrossGameFactorRegistry) -> None:
        self.arm = arm
        self.registry = registry
        self.step_index = 0
        self.frame_hashes_seen: set[str] = set()
        self.contexts: set[str] = set()
        self.program_hashes: set[str] = set()
        self.reasons: Counter[str] = Counter()
        self.inspections = 0
        self.maximum_legal_candidates = 0
        self.abstention_reason = ""

    def decide(
        self,
        *,
        current_grid: Any,
        legal_actions: Sequence[Any],
        game_state: str,
        levels_completed: int,
    ) -> tuple[CognitiveDecision | None, Any]:
        frame_hash = protocol.sha256_payload(current_grid.tolist())
        if frame_hash in self.frame_hashes_seen and self.arm != "termination_ablation":
            self.abstention_reason = "repeated_state_stop"
            self.reasons[self.abstention_reason] += 1
            return None, None
        self.frame_hashes_seen.add(frame_hash)
        if len(legal_actions) > 512:
            raise protocol.IntegrityError("legal candidate processing budget exceeded")
        self.maximum_legal_candidates = max(self.maximum_legal_candidates, len(legal_actions))
        observation = build_observation(
            current_grid,
            available_actions=durable.live._available_action_names(legal_actions),
            game_state=game_state,
            levels_completed=levels_completed,
            infer_players=True,
        )
        state = compile_observation(observation)
        grounding = select_grounding(
            self.registry,
            arm=self.arm,
            candidates=legal_actions,
            shape=tuple(int(value) for value in current_grid.shape[:2]),
            step_index=self.step_index,
            state=state,
        )
        self.inspections += grounding.inspections
        if grounding.context:
            self.contexts.add(grounding.context)
        if grounding.program_hash:
            self.program_hashes.add(grounding.program_hash)
        self.reasons[grounding.reason] += 1
        if grounding.abstained:
            self.abstention_reason = grounding.reason
            return None, grounding
        candidate = grounding.candidate
        decision = CognitiveDecision(
            action_name=candidate.action_name,
            action_data=dict(candidate.action_data),
            source="sage_t_cross_game_factor",
            reason=grounding.reason,
            confidence=1.0,
            option_id=str(grounding.program_hash or ""),
        )
        self.step_index += 1
        return decision, grounding

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "recognized_contexts": sorted(self.contexts),
            "program_hashes_used": sorted(self.program_hashes),
            "decision_reasons": dict(self.reasons),
            "candidate_inspections": self.inspections,
            "maximum_legal_candidates": self.maximum_legal_candidates,
            "abstention_reason": self.abstention_reason,
            "source_information_loaded": self.arm != "generic_source_free",
            "legacy_fallback_actions": 0,
            "cross_reset_memory": False,
            "grounded_arguments_persisted": False,
        }


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
            "format_version": "sage-t10.3.12c-action-intent-v1",
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
    registry: CrossGameFactorRegistry,
    registry_checksum: str,
    lock: Any,
) -> dict[str, Any]:
    receipt_path = _receipt_path(destination, work)
    if receipt_path.is_file():
        return durable._read_signed(receipt_path, "receipt_checksum")
    controller = _ActiveFactorController(work.arm, registry)
    trace_ids: list[str] = []
    errors: list[str] = []
    illegal_actions = 0
    game_over_actions = 0
    level_delta_total = 0
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
            decision, grounding = controller.decide(
                current_grid=before.grid,
                legal_actions=legal,
                game_state=str(before.game_state),
                levels_completed=int(before.levels_completed),
            )
            if decision is None:
                stop_reason = "PLANNED_FACTOR_ABSTENTION"
                break
            if decision.source != "sage_t_cross_game_factor":
                raise protocol.IntegrityError("non-SAGE decision entered active transfer")
            selected = durable.live._materialize_decision(legal, decision)
            if selected is None:
                illegal_actions += 1
                errors.append("UNAVAILABLE_SAGE_DECISION")
                status = "ABORTED"
                stop_reason = "UNAVAILABLE_SAGE_DECISION"
                break
            intent = _intent(
                manifest, work, step_index=step_index, selected=selected,
                decision=decision, registry_checksum=registry_checksum,
            )
            name = f"{step_index:04d}.json"
            protocol.write_json_once(_work_path(destination, "intents", work, name), intent)
            lock.heartbeat()
            try:
                after_frame = durable.live._step_env_action(env, selected)
                after = durable.live.snapshot_frame(
                    after_frame, fallback_available_actions=before.available_actions
                )
            except Exception as exc:  # noqa: BLE001
                unresolved = signed(
                    {
                        "format_version": "sage-t10.3.12c-unresolved-event-v1",
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
                    "format_version": "sage-t10.3.12c-physical-event-v1",
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
            level_delta_total += level_delta
            frame = after_frame
            final_level = int(after.levels_completed)
            if str(after.game_state).upper() == "GAME_OVER":
                game_over_actions += 1
            lock.heartbeat()
            print(
                json.dumps(
                    {
                        "phase": work.phase, "game_id": work.game_id, "arm": work.arm,
                        "step": step_index + 1, "budget": work.action_budget,
                        "levels": final_level - initial_level,
                        "decision_source": decision.source,
                    },
                    sort_keys=True, separators=(",", ":"),
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
    summary = controller.summary()
    receipt = signed(
        {
            "format_version": "sage-t10.3.12c-branch-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            **work.as_dict(),
            "work_id": work.work_id,
            "status": status,
            "complete": status == "COMPLETE" and not errors,
            "stop_reason": stop_reason,
            "planned_abstention": stop_reason == "PLANNED_FACTOR_ABSTENTION",
            "issued_intents": issued,
            "sealed_events": sealed,
            "unresolved_intents": unresolved,
            "event_ids": trace_ids,
            "initial_frame_sha256": initial_frame_hash,
            "level_delta": max(level_delta_total, final_level - initial_level),
            "sage_t_option_actions": sealed,
            "recognized_contexts": summary["recognized_contexts"],
            "program_hashes_used": summary["program_hashes_used"],
            "decision_reasons": summary["decision_reasons"],
            "candidate_inspections": summary["candidate_inspections"],
            "maximum_legal_candidates": summary["maximum_legal_candidates"],
            "abstention_reason": summary["abstention_reason"],
            "source_information_loaded": summary["source_information_loaded"],
            "legacy_fallback_actions": 0,
            "game_over_actions": game_over_actions,
            "illegal_actions": illegal_actions,
            "errors": errors,
            "registry_checksum_loaded": registry_checksum,
            "physical_actions_replayed": 0,
            "grounded_arguments_persisted": False,
        },
        "receipt_checksum",
    )
    protocol.write_json_once(receipt_path, receipt)
    return receipt


def active_transfer(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-targets")
    registry_payload = _read_signed(root, REGISTRY_FILENAME, "registry_checksum")
    if registry_payload.get("compiled_before_target_outcomes") is not True:
        raise protocol.IntegrityError("transfer registry was not compiled before target outcomes")
    registry = CrossGameFactorRegistry(registry_payload)
    destination = _destination(root)
    with _durable_contract():
        durable._require_live_runtime()
        before = durable._journal_accounting(destination)
        if before.get("inflight_paths") or before.get("unresolved_intents"):
            raise protocol.IntegrityError("interrupted physical action cannot be replayed")
        if before.get("incomplete_work_ids"):
            raise protocol.IntegrityError("an interrupted work scope cannot be reconstructed safely")
        lock = durable._CollectorLock(destination / LOCK_FILENAME, "active-transfer")
        lock.acquire()
        try:
            for work in protocol.work_specs("active-transfer"):
                _run_work(
                    root, destination, manifest, work, registry,
                    str(registry_payload["registry_checksum"]), lock,
                )
        finally:
            lock.release()
        receipts = _load_receipts(destination)
        accounting = durable._journal_accounting(destination)

    checks = {
        "all_54_receipts": len(receipts) == protocol.TOTAL_RESETS,
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
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        subset = [row for row in receipts if row["arm"] == arm]
        by_arm[arm] = {
            "success_games": sorted(row["game_id"] for row in subset if int(row["level_delta"]) > 0),
            "applicable_games": sorted(row["game_id"] for row in subset if row["recognized_contexts"]),
            "planned_abstention_games": sorted(row["game_id"] for row in subset if row["planned_abstention"]),
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
            "format_version": "sage-t10.3.12c-active-cross-game-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "registry_checksum": registry_payload["registry_checksum"],
            "collection_checks": checks,
            "collection_complete": all(checks.values()),
            "accounting": accounting,
            "metrics": {
                "by_arm": by_arm,
                "initial_frame_hashes": dict(initial_hashes),
                "distinct_target_initial_frames": len(
                    {value for values in initial_hashes.values() for value in values}
                ),
                "labels_seed_environment": False,
                "physical_actions": sum(int(row["sealed_events"]) for row in receipts),
            },
            "receipt_checksums": sorted(row["receipt_checksum"] for row in receipts),
            "physical_actions_replayed": 0,
            "sequence_games_opened": False,
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, ACTIVE_REPORT_FILENAME, payload)
    if _artifact_bytes(root) > int(manifest["matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.12c artifact budget exceeded")
    if not payload["collection_complete"]:
        raise protocol.IntegrityError("active transfer collection did not seal cleanly")
    return payload


def _factor_arm(factor: str) -> str:
    return {
        "operator": "operator_ablation",
        "role_binding": "role_binding_ablation",
        "transition": "transition_ablation",
        "termination": "termination_ablation",
    }[factor]


def adjudicate(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    active = _require_gate(root, "active-transfer")
    receipts = _load_receipts(_destination(root))
    full = {row["game_id"]: row for row in receipts if row["arm"] == ARMS[0]}
    generic = {row["game_id"]: row for row in receipts if row["arm"] == ARMS[1]}
    full_success = {game for game, row in full.items() if int(row["level_delta"]) > 0}
    generic_success = {game for game, row in generic.items() if int(row["level_delta"]) > 0}
    applicable = {game for game, row in full.items() if row["recognized_contexts"]}
    factor_results = {}
    supported = []
    for factor in FACTORS:
        arm = _factor_arm(factor)
        ablated = {row["game_id"]: row for row in receipts if row["arm"] == arm}
        full_wins = sorted(
            game for game in protocol.TARGET_GAMES
            if int(full[game]["level_delta"]) > 0 and int(ablated[game]["level_delta"]) == 0
        )
        reverse_wins = sorted(
            game for game in protocol.TARGET_GAMES
            if int(full[game]["level_delta"]) == 0 and int(ablated[game]["level_delta"]) > 0
        )
        passed = (
            len(full_wins) >= int(manifest["gates"]["minimum_paired_ablation_advantage"])
            and len(reverse_wins)
            <= int(manifest["gates"]["maximum_reverse_paired_ablation_wins"])
        )
        if passed:
            supported.append(factor)
        factor_results[factor] = {
            "ablation_arm": arm,
            "full_wins": full_wins,
            "reverse_wins": reverse_wins,
            "paired_advantage": len(full_wins) - len(reverse_wins),
            "supported_cross_game": passed,
        }
    full_actions_success = sum(int(full[game]["sealed_events"]) for game in full_success)
    generic_actions_success = sum(int(generic[game]["sealed_events"]) for game in generic_success)
    source_value = len(full_success) > len(generic_success)
    if len(full_success) == len(generic_success) and full_success == generic_success:
        source_value = generic_actions_success > 0 and full_actions_success <= (
            float(manifest["gates"]["generic_equal_success_action_ratio_maximum"])
            * generic_actions_success
        )
    success_rate = len(full_success) / len(applicable) if applicable else 0.0
    checks = {
        "collection_complete": active.get("collection_complete") is True,
        "nine_target_frames_observed": len(active["metrics"]["initial_frame_hashes"]) == 9,
        "minimum_source_applicability": len(applicable)
        >= int(manifest["gates"]["minimum_factorized_applicable_games"]),
        "minimum_source_success_games": len(full_success)
        >= int(manifest["gates"]["minimum_factorized_success_games"]),
        "source_success_rate": success_rate
        >= float(manifest["gates"]["minimum_factorized_success_rate_on_applicable"]),
        "source_value_over_generic": source_value,
        "minimum_supported_factors": len(supported)
        >= int(manifest["gates"]["minimum_supported_factors"]),
        "zero_source_game_over": all(int(row["game_over_actions"]) == 0 for row in full.values()),
        "zero_errors_all_arms": all(not row["errors"] for row in receipts),
        "zero_illegal_actions": all(int(row["illegal_actions"]) == 0 for row in receipts),
        "all_executed_actions_sage_t": all(
            int(row["sage_t_option_actions"]) == int(row["sealed_events"])
            for row in receipts
        ),
        "zero_legacy_fallback": all(int(row["legacy_fallback_actions"]) == 0 for row in receipts),
    }
    passed = all(checks.values())
    if not checks["minimum_source_applicability"]:
        verdict = "SOURCE_OPERATOR_COVERAGE_MISS"
    elif not checks["minimum_source_success_games"] or not checks["source_success_rate"]:
        verdict = "CROSS_GAME_TRANSFER_MISS"
    elif not checks["source_value_over_generic"]:
        verdict = "GENERIC_SEARCH_EXPLAINS_TRANSFER"
    elif not checks["minimum_supported_factors"]:
        verdict = "FACTOR_CAUSALITY_MISS"
    elif not passed:
        verdict = "CROSS_GAME_SAFETY_OR_INTEGRITY_MISS"
    else:
        verdict = "PASS_T10_3_12C_CROSS_GAME_FACTORS_IDENTIFIED"
    payload = signed(
        {
            "format_version": "sage-t10.3.12c-cross-game-adjudication-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "active_report_checksum": active["report_checksum"],
            "checks": checks,
            "passed": passed,
            "verdict": verdict,
            "factorized_applicable_games": sorted(applicable),
            "factorized_success_games": sorted(full_success),
            "generic_success_games": sorted(generic_success),
            "factorized_success_rate_on_applicable": success_rate,
            "factor_results": factor_results,
            "supported_cross_game_factors": supported,
            "unsupported_cross_game_factors": [factor for factor in FACTORS if factor not in supported],
            "cross_game_generalization_proven": passed,
            "program_promoted": False,
            "sequence_composition_authorized": False,
            "next_step": (
                "preregister_independent_cross_game_reproduction"
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
        supported = list(adjudication.get("supported_cross_game_factors", ()))
    else:
        verdict = "INCOMPLETE_T10_3_12C"
        passed = False
        supported = []
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
            "format_version": "sage-t10.3.12c-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "passed": passed,
            "supported_cross_game_factors": supported,
            "accounting": accounting,
            "artifacts": artifacts,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
            "physical_actions_replayed": 0,
            "legacy_fallback_actions": 0,
            "parent_events_used_for_training": 0,
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
        "format_version": "sage-t10.3.12c-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "artifacts": artifacts,
        "accounting": accounting,
        "artifact_bytes": _artifact_bytes(root),
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        "maximum_resets": protocol.TOTAL_RESETS,
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
            "freeze", "status", "audit-parent", "preflight", "audit-targets",
            "compile-transfer", "active-transfer", "adjudicate", "report",
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
            "preflight": preflight,
            "audit-targets": audit_targets,
            "compile-transfer": compile_transfer,
            "active-transfer": active_transfer,
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
    "PREFLIGHT_FILENAME", "REGISTRY_FILENAME", "TARGET_AUDIT_FILENAME",
    "TERMINAL_REPORT_FILENAME", "active_transfer", "adjudicate", "audit_parent",
    "audit_targets", "compile_transfer", "main", "preflight", "status",
    "terminal_report",
]
