"""Durable corrected source runtime for SAGE.T10.3.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_1_protocol as protocol
from . import t10_3_runtime as _parent
from .contracts import ActionCandidate
from .frame_adapters_v10_3 import resolve_pre_action_root, structural_signature
from .frame_adapters_v10_3_1 import (
    project_goal_transition,
    shared_quotient_transport,
)
from .progress_witness_v10 import GroundedAction, SearchConfig, chain_successor_macro

FORMAT_VERSION = "sage-t10.3.1-goal-progress-runtime-v1"
EVENT_FORMAT_VERSION = "sage-t10.3.1-physical-event-v1"
INTENT_FORMAT_VERSION = "sage-t10.3.1-action-intent-v1"
BRANCH_FORMAT_VERSION = "sage-t10.3.1-branch-receipt-v1"
CHECKPOINT_FORMAT_VERSION = "sage-t10.3.1-checkpoint-v1"
COMPILED_FORMAT_VERSION = "sage-t10.3.1-compiled-event-v1"
AUDIT_FILENAME = "offline_audit.json"
CHECKPOINT_FILENAME = "checkpoint.json"
COMPILED_LEDGER_FILENAME = "compiled_source_events.jsonl"
COMPACT_LEDGER_FILENAME = "compact_ledger.json"
QA_FILENAME = "qa_report.json"
MODEL_FILENAME = "model_recipe.json"
CONFIRMATION_FILENAME = "confirmation_report.json"
TERMINAL_FILENAME = "t10_3_1_report.json"

ScientificGateMiss = _parent.ScientificGateMiss
IntegrityError = _parent.IntegrityError
_science = _parent._science
_sha = _parent._sha
_signed = _parent._signed
_write_signed_once = _parent._write_signed_once
_read_signed = _parent._read_signed
_write_jsonl_once = _parent._write_jsonl_once
assert_no_forbidden_persistence = _parent.assert_no_forbidden_persistence


def _output(root: Path) -> Path:
    return root / protocol.DEFAULT_OUTPUT_ROOT


def _firewall() -> dict[str, bool]:
    return {
        "source_validation_opened": False,
        "ar25_opened": False,
        "holdout_opened": False,
        "production_authority": False,
    }


@dataclass(frozen=True)
class WorkSpec:
    phase: str
    game_id: str
    seed: int
    controller: str
    reset_index: int

    @property
    def work_id(self) -> str:
        return _sha(
            {
                "protocol": "t10_3_1",
                "phase": self.phase,
                "game_id": self.game_id,
                "seed": self.seed,
                "controller": self.controller,
                "reset_index": self.reset_index,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "game_id": self.game_id,
            "seed": self.seed,
            "controller": self.controller,
            "reset_index": self.reset_index,
            "work_id": self.work_id,
        }


def build_work_specs(phase: str) -> tuple[WorkSpec, ...]:
    rows: list[WorkSpec] = []
    if phase == "panel":
        reset_index = 0
        for game in protocol.SOURCE_GAMES:
            for seed in protocol.PANEL_SEEDS:
                for controller in protocol.PANEL_ARMS:
                    rows.append(WorkSpec(phase, game, seed, controller, reset_index))
                    reset_index += 1
        return tuple(rows)
    if phase == "confirmation":
        reset_index = protocol.PANEL_RESETS
        for game_index, game in enumerate(protocol.SOURCE_GAMES):
            for seed_index, seed in enumerate(protocol.CONFIRMATION_SEEDS):
                controllers = list(protocol.CONFIRMATION_CONTROLLERS)
                if (game_index + seed_index) % 2:
                    controllers.reverse()
                for controller in controllers:
                    rows.append(WorkSpec(phase, game, seed, controller, reset_index))
                    reset_index += 1
        return tuple(rows)
    raise ValueError("work phase must be panel or confirmation")


def _journal_root(destination: Path) -> Path:
    return destination / "journal"


def _work_path(destination: Path, category: str, work: WorkSpec, name: str) -> Path:
    return _journal_root(destination) / category / work.work_id / name


def _event_id(work: WorkSpec, step_index: int) -> str:
    return _sha({"work_id": work.work_id, "step_index": step_index})


def _checkpoint(destination: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    completed: list[str] = []
    intent_count = event_count = unresolved_count = action_count = branch_count = 0
    for phase in ("panel", "confirmation"):
        for work in build_work_specs(phase):
            receipt_path = _work_path(destination, "branches", work, "receipt.json")
            if receipt_path.exists():
                receipt = _read_signed(receipt_path, "receipt_checksum")
                completed.append(work.work_id)
                branch_count += 1
                action_count += int(receipt.get("issued_intents", 0))
            for category, accumulator in (
                ("intents", "intent"),
                ("events", "event"),
                ("unresolved", "unresolved"),
            ):
                directory = _work_path(destination, category, work, "placeholder").parent
                count = len(tuple(directory.glob("*.json"))) if directory.exists() else 0
                if accumulator == "intent":
                    intent_count += count
                elif accumulator == "event":
                    event_count += count
                else:
                    unresolved_count += count
    if intent_count != event_count + unresolved_count:
        raise IntegrityError("intent accounting equation does not hold")
    if action_count != intent_count:
        raise IntegrityError("branch receipts do not account for every intent")
    if intent_count > protocol.TOTAL_MAXIMUM_ACTIONS:
        raise IntegrityError("T10.3.1 physical action budget exceeded")
    payload = _signed(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "completed_work_ids": sorted(completed),
            "completed_branch_count": branch_count,
            "authorized_intent_count": intent_count,
            "sealed_event_count": event_count,
            "explicitly_unresolved_intent_count": unresolved_count,
            "physical_actions_replayed": 0,
            "maximum_authorized_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "equation_holds": intent_count == event_count + unresolved_count,
            "firewall": _firewall(),
        },
        "checkpoint_checksum",
    )
    destination.mkdir(parents=True, exist_ok=True)
    from .t10_2_1_runtime import _atomic_write_json

    _atomic_write_json(destination / CHECKPOINT_FILENAME, payload)
    return payload


def build_offline_audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_root = root / protocol.PARENT_OUTPUT_ROOT
    exact_common = complete_pairs = event_count = 0
    event_root = parent_root / "journal" / "events"
    for path in event_root.rglob("*.json"):
        event = _read_signed(path, "event_checksum")
        frames = event.get("model_view", {}).get("frames", {})
        transport, _certificates = shared_quotient_transport(frames)
        event_count += 1
        complete_pairs += int(
            bool(
                frames.get("allocentric_object_relative", {}).get("complete")
                and frames.get("action_aligned_relational", {}).get("complete")
            )
        )
        exact_common += int(transport["multiframe_exact_nonidentity"])
    migration = manifest["migration_receipt"]
    passed = bool(
        event_count == 540
        and exact_common == complete_pairs
        and exact_common > 0
        and migration.get("parent_events_used_for_fit") == 0
        and migration.get("parent_events_relabelled") == 0
    )
    return _signed(
        {
            "format_version": "sage-t10.3.1-offline-audit-v1",
            "phase": "audit",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "status": "PASS_T10_3_1_OFFLINE_AUDIT" if passed else "PROVENANCE_INVALID",
            "parent_event_count": event_count,
            "parent_complete_relational_pair_count": complete_pairs,
            "parent_exact_common_quotient_count": exact_common,
            "correction_feasible_on_complete_pairs": exact_common == complete_pairs,
            "parent_events_used_for_fit": 0,
            "parent_events_relabelled": 0,
            "fresh_panel_required": True,
            "physical_actions_executed": 0,
            "firewall": _firewall(),
        },
        "audit_checksum",
    )


def audit_phase(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = build_offline_audit(root, manifest)
    _write_signed_once(_output(root) / AUDIT_FILENAME, payload)
    return payload


def _require_audit(destination: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = destination / AUDIT_FILENAME
    if not path.exists():
        raise ScientificGateMiss("offline audit must pass before collection")
    audit = _read_signed(path, "audit_checksum")
    if audit.get("manifest_checksum") != manifest["manifest_checksum"] or audit.get("passed") is not True:
        raise ScientificGateMiss("offline migration audit failed")
    return audit


def _action_name(action: Any) -> str:
    return _parent._action_name(action)


def _action_data(action: Any) -> dict[str, Any]:
    return _parent._action_data(action)


def _abstract_state(snapshot: Any, legal_actions: Sequence[Any]) -> Any:
    return _parent._abstract_state(snapshot, legal_actions)


@dataclass(frozen=True)
class _Candidate:
    concrete: Any
    grounded: GroundedAction
    signature: str
    method: str
    unique: bool


def _groundable_candidates(state: Any, legal_actions: Sequence[Any]) -> tuple[_Candidate, ...]:
    rows: list[_Candidate] = []
    for concrete in legal_actions:
        grounded = GroundedAction.from_view(concrete)
        candidate = ActionCandidate(grounded.action_name, grounded.data)
        resolved = resolve_pre_action_root(state, candidate)
        if resolved.entity_id is None or not resolved.unique:
            continue
        rows.append(
            _Candidate(
                concrete=concrete,
                grounded=grounded,
                signature=structural_signature(state, resolved.entity_id),
                method=resolved.method,
                unique=True,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.signature,
                row.grounded.action_name,
                len(row.grounded.action_data),
            ),
        )
    )


def choose_regrounded_action(
    work: WorkSpec,
    state: Any,
    legal_actions: Sequence[Any],
    *,
    step_index: int,
    preferred_grounding_key: str | None = None,
) -> Any | None:
    """Choose from the current legal inventory; no grounded action survives a step."""

    rows = _groundable_candidates(state, legal_actions)
    if not rows:
        return None
    parameterized = tuple(row for row in rows if row.grounded.action_name == "ACTION6")
    movements = tuple(
        row
        for row in rows
        if row.grounded.action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
    )
    if preferred_grounding_key is not None:
        return next(
            (
                row.concrete
                for row in rows
                if row.grounded.key == preferred_grounding_key
            ),
            None,
        )

    learned = work.phase == "confirmation" and work.controller == "learned"
    independent = work.controller == "capacity_matched_independent"
    chain = chain_successor_macro(
        state,
        tuple(row.grounded for row in rows),
        config=SearchConfig(maximum_horizon=protocol.MAXIMUM_ACTIONS_PER_RESET),
    )
    chain_actions = tuple(chain.actions) if chain is not None else ()

    def concrete_for(grounded: GroundedAction) -> Any | None:
        return next(
            (row.concrete for row in rows if row.grounded.key == grounded.key),
            None,
        )

    if learned:
        if chain_actions:
            return concrete_for(chain_actions[0])
        if parameterized:
            return parameterized[0].concrete
        if movements:
            return movements[step_index % len(movements)].concrete
        return rows[step_index % len(rows)].concrete

    if work.game_id.startswith("su15"):
        if chain_actions:
            if work.controller == "canonical_option":
                return concrete_for(chain_actions[0])
            if work.controller == "binding_swap":
                return concrete_for(chain_actions[1]) if len(chain_actions) > 1 else None
            if work.controller == "option_intervention":
                return concrete_for(chain_actions[-1])
            if independent:
                excluded = {item.key for item in chain_actions[:2]}
                controls = tuple(row for row in rows if row.grounded.key not in excluded)
                return controls[step_index % len(controls)].concrete if controls else None
        if independent:
            return rows[(step_index * 3 + 1) % len(rows)].concrete
        return None

    if work.game_id.startswith("lp85"):
        if not parameterized:
            return None
        if work.controller == "canonical_option":
            return parameterized[0].concrete
        if work.controller == "binding_swap":
            return parameterized[1].concrete if len(parameterized) > 1 else None
        if work.controller == "option_intervention":
            return parameterized[step_index % min(2, len(parameterized))].concrete if len(parameterized) > 1 else None
        if independent:
            controls = tuple(row for row in rows if row.grounded.key != parameterized[0].grounded.key)
            return controls[step_index % len(controls)].concrete if controls else None

    if work.game_id.startswith("bp35"):
        if work.controller == "canonical_option":
            desired = f"ACTION{step_index % 4 + 1}"
            return next((row.concrete for row in movements if row.grounded.action_name == desired), None)
        if work.controller == "binding_swap":
            choices = tuple(reversed(parameterized or movements or rows))
            return choices[step_index % len(choices)].concrete
        if work.controller == "option_intervention":
            choices = parameterized if step_index % 2 == 0 and parameterized else movements or rows
            return choices[step_index % len(choices)].concrete
        if independent:
            return rows[(step_index * 3 + 1) % len(rows)].concrete

    if independent:
        canonical = chain_actions[0].key if chain_actions else parameterized[0].grounded.key if parameterized else ""
        controls = tuple(row for row in rows if row.grounded.key != canonical)
        return controls[step_index % len(controls)].concrete if controls else None
    return rows[step_index % len(rows)].concrete


def classify_grounding_stop(controller: str, step_index: int) -> tuple[str, bool]:
    """Return status and whether an unattested initial grounding miss occurred."""

    control = controller == "capacity_matched_independent"
    if step_index == 0:
        return (
            "CONTROL_GROUNDING_MISS" if control else "OPTION_GROUNDING_MISS",
            True,
        )
    return (
        "CONTROL_TERMINATED_NO_GROUNDING"
        if control
        else "OPTION_TERMINATED_NO_GROUNDING",
        False,
    )


def _build_goal_projection(
    *,
    before: Any,
    after: Any,
    action: Any,
    legal_actions: Sequence[Any],
    event_id: str,
    step_index: int,
    game_id: str,
) -> Any:
    from theory.live_transition_loop import build_transition_record

    from .compiler import compile_transition_record

    before_grid = getattr(before, "grid", None)
    after_grid = getattr(after, "grid", None)
    if before_grid is None and isinstance(before, Mapping):
        before_grid = before.get("grid")
    if after_grid is None and isinstance(after, Mapping):
        after_grid = after.get("grid")
    if before_grid is None or after_grid is None:
        raise IntegrityError("runtime snapshots must expose grids")
    record = build_transition_record(
        action=_action_name(action),
        action_args=_action_data(action),
        grid_before=before_grid,
        grid_after=after_grid,
        available_actions=tuple(_action_name(item) for item in legal_actions),
        game_state_before=_science._snapshot_state(before) or "NOT_FINISHED",
        game_state_after=_science._snapshot_state(after) or "NOT_FINISHED",
        levels_completed_before=_science._snapshot_levels(before),
        levels_completed_after=_science._snapshot_levels(after),
        timestamp=step_index,
    )
    evidence = compile_transition_record(record, source_game_id=game_id)
    return project_goal_transition(evidence, event_id=event_id)


def _physical_event(projection: Any, *, work: WorkSpec, step_index: int) -> dict[str, Any]:
    compact = _science._compact_event(
        projection.bundle,
        controller=work.controller,
        reset_index=work.reset_index,
        step_index=step_index,
        progressing_sequence_rank=None,
        donor_game_count=2,
        capacity_slots=4,
    )
    frames = compact.get("model_view", {}).get("frames", {})
    transport, certificates = shared_quotient_transport(frames)
    root = frames.get("root_only", {}) if isinstance(frames, Mapping) else {}
    all_unchanged = bool(
        frames
        and all(
            frame.get("before_hash") == frame.get("after_hash")
            for frame in frames.values()
            if isinstance(frame, Mapping)
        )
    )
    old_labels = compact.get("labels", {})
    old_correspondence = compact.get("correspondence", {})
    confident = int(projection.binding.pre_action_complete and projection.binding.unique)
    compact.update(
        {
            "format_version": EVENT_FORMAT_VERSION,
            "work_id": work.work_id,
            "phase": work.phase,
            "branch_outcome_pending": True,
            "binding": projection.binding.as_dict(),
            "observations": {
                "physical_no_effect": all_unchanged,
                "root_effect": bool(root.get("before_hash") != root.get("after_hash")),
                "level_complete": bool(old_labels.get("level_complete")),
                "game_over": bool(old_labels.get("game_over")),
            },
            "labels": {},
            "learned_predicates": ["goal_reachable_within_option"],
            "correspondence": {
                "method": "unique_pre_action_root_binding",
                "confident_matches": confident,
                "ambiguous_matches": 1 - confident,
                "fully_ambiguous_matches": 1 - confident,
                "entities_considered": 1,
                "fraction_denominator": 1,
                "confident_fraction": float(confident),
                "fully_ambiguous_fraction": float(1 - confident),
                "after_root_available": projection.binding.after_root_available,
                "entity_correspondence_diagnostic": old_correspondence,
            },
            "transport": transport,
            "transport_certificates": certificates,
            "transport_orbits": [],
        }
    )
    compact["provenance"].update(
        {
            "collector": FORMAT_VERSION,
            "root_adapter": "sage-t10.3.1-frame-adapter-v1",
            "transport_adapter": "orientation_erased_structural_relational_v1",
            "parent_t10_3_event": False,
            "raw_runtime_state_retained": False,
        }
    )
    assert_no_forbidden_persistence(compact)
    unsigned = {key: value for key, value in compact.items() if key != "event_checksum"}
    return _signed(unsigned, "event_checksum")


def _intent_payload(
    manifest: Mapping[str, Any],
    work: WorkSpec,
    *,
    step_index: int,
    action: Any,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    assert_no_forbidden_persistence({"binding": dict(binding)})
    return _signed(
        {
            "format_version": INTENT_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "event_id": _event_id(work, step_index),
            "step_index": step_index,
            "action_schema": _action_name(action),
            "parameter_arity": len(_action_data(action)),
            "binding": dict(binding),
            "regrounded_from_current_legal_inventory": True,
            "physical_action_authorized": True,
            "physical_replay_authorized": False,
        },
        "intent_checksum",
    )


def _read_events_for_work(destination: Path, work: WorkSpec) -> list[dict[str, Any]]:
    directory = _work_path(destination, "events", work, "placeholder").parent
    if not directory.exists():
        return []
    return [_read_signed(path, "event_checksum") for path in sorted(directory.glob("*.json"))]


def _recover_interrupted_work(
    destination: Path, manifest: Mapping[str, Any], work: WorkSpec
) -> dict[str, Any] | None:
    receipt_path = _work_path(destination, "branches", work, "receipt.json")
    if receipt_path.exists():
        return _read_signed(receipt_path, "receipt_checksum")
    intent_dir = _work_path(destination, "intents", work, "placeholder").parent
    intents = sorted(intent_dir.glob("*.json")) if intent_dir.exists() else []
    if not intents:
        return None
    event_ids: list[str] = []
    unresolved_ids: list[str] = []
    for intent_path in intents:
        intent = _read_signed(intent_path, "intent_checksum")
        event_path = _work_path(destination, "events", work, intent_path.name)
        if event_path.exists():
            event_ids.append(str(_read_signed(event_path, "event_checksum")["event_id"]))
            continue
        unresolved = _signed(
            {
                "format_version": "sage-t10.3.1-unresolved-event-v1",
                "manifest_checksum": manifest["manifest_checksum"],
                "work_id": work.work_id,
                "event_id": intent["event_id"],
                "step_index": intent["step_index"],
                "complete": False,
                "reason": "CRASH_BETWEEN_INTENT_AND_SEAL",
                "physical_action_replayed": False,
            },
            "unresolved_checksum",
        )
        _write_signed_once(
            _work_path(destination, "unresolved", work, intent_path.name), unresolved
        )
        unresolved_ids.append(str(intent["event_id"]))
    events = _read_events_for_work(destination, work)
    progressed = any(event.get("observations", {}).get("level_complete") for event in events)
    receipt = _signed(
        {
            "format_version": BRANCH_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            **work.as_dict(),
            "status": "INTERRUPTED_UNKNOWN" if unresolved_ids else "RECOVERED_AFTER_EVENT_SEAL",
            "complete": not unresolved_ids,
            "goal_reachable_within_option": progressed if not unresolved_ids else None,
            "issued_intents": len(intents),
            "sealed_events": len(event_ids),
            "unresolved_intents": len(unresolved_ids),
            "event_ids": event_ids,
            "unresolved_event_ids": unresolved_ids,
            "level_delta": int(progressed),
            "errors": ["CRASH_BETWEEN_INTENT_AND_SEAL"] if unresolved_ids else [],
            "illegal_actions": 0,
            "game_over": any(event.get("observations", {}).get("game_over") for event in events),
            "physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    _write_signed_once(receipt_path, receipt)
    return receipt


def _run_work(
    destination: Path,
    manifest: Mapping[str, Any],
    work: WorkSpec,
    runtime: Any,
) -> dict[str, Any]:
    recovered = _recover_interrupted_work(destination, manifest, work)
    if recovered is not None:
        return recovered
    environment = _science._open_runtime(runtime, work.game_id, work.seed)
    event_ids: list[str] = []
    errors: list[str] = []
    issued = illegal_actions = 0
    game_over = False
    initial_levels = final_levels = 0
    status = "COMPLETE"
    persistent_grounding_key: str | None = None
    try:
        frame = _science._reset_runtime(runtime, environment)
        legal = _science._legal_runtime(runtime, environment)
        snapshot = _science._snapshot_runtime(runtime, frame, fallback_available_actions=legal)
        initial_levels = final_levels = _science._snapshot_levels(snapshot)
        for step_index in range(protocol.MAXIMUM_ACTIONS_PER_RESET):
            legal = _science._legal_runtime(runtime, environment)
            state = _abstract_state(snapshot, legal)
            concrete = choose_regrounded_action(
                work,
                state,
                legal,
                step_index=step_index,
                preferred_grounding_key=persistent_grounding_key,
            )
            if concrete is None:
                status, is_error = classify_grounding_stop(
                    work.controller, step_index
                )
                if is_error:
                    errors.append(status)
                break
            if (
                persistent_grounding_key is None
                and work.game_id.startswith("lp85")
                and work.controller in {"canonical_option", "binding_swap"}
            ):
                persistent_grounding_key = GroundedAction.from_view(concrete).key
            resolved = resolve_pre_action_root(
                state, ActionCandidate(_action_name(concrete), _action_data(concrete))
            )
            signature = (
                structural_signature(state, resolved.entity_id)
                if resolved.entity_id is not None
                else None
            )
            binding = {
                "format_version": "sage-t10.3.1-intent-binding-v1",
                "method": resolved.method,
                "structural_signature": signature,
                "unique": resolved.unique,
                "complete": resolved.entity_id is not None and resolved.unique,
                "raw_identifier_retained": False,
                "spatial_anchor_retained": False,
            }
            intent = _intent_payload(
                manifest,
                work,
                step_index=step_index,
                action=concrete,
                binding=binding,
            )
            name = f"{step_index:02d}.json"
            _write_signed_once(_work_path(destination, "intents", work, name), intent)
            issued += 1
            before = snapshot
            try:
                next_frame = _science._step_runtime(runtime, environment, concrete)
                next_legal = _science._legal_runtime(runtime, environment)
                after = _science._snapshot_runtime(
                    runtime, next_frame, fallback_available_actions=next_legal
                )
            except Exception as exc:
                unresolved = _signed(
                    {
                        "format_version": "sage-t10.3.1-unresolved-event-v1",
                        "manifest_checksum": manifest["manifest_checksum"],
                        "work_id": work.work_id,
                        "event_id": intent["event_id"],
                        "step_index": step_index,
                        "complete": False,
                        "reason": f"ENVIRONMENT_CALL_UNATTESTABLE:{type(exc).__name__}",
                        "physical_action_replayed": False,
                    },
                    "unresolved_checksum",
                )
                _write_signed_once(
                    _work_path(destination, "unresolved", work, name), unresolved
                )
                errors.append("ENVIRONMENT_CALL_UNATTESTABLE")
                status = "ABORTED"
                break
            projection = _build_goal_projection(
                before=before,
                after=after,
                action=concrete,
                legal_actions=legal,
                event_id=intent["event_id"],
                step_index=step_index,
                game_id=work.game_id,
            )
            event = _physical_event(projection, work=work, step_index=step_index)
            _write_signed_once(_work_path(destination, "events", work, name), event)
            event_ids.append(str(event["event_id"]))
            snapshot = after
            final_levels = _science._snapshot_levels(snapshot)
            game_over = _science._is_game_over(snapshot)
            if final_levels > initial_levels or _science._is_terminal(snapshot):
                break
    finally:
        _science._close_runtime(runtime, environment)
    unresolved_dir = _work_path(destination, "unresolved", work, "placeholder").parent
    unresolved_count = len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    complete = unresolved_count == 0 and not errors
    progressed = final_levels > initial_levels
    receipt = _signed(
        {
            "format_version": BRANCH_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            **work.as_dict(),
            "status": status,
            "complete": complete,
            "goal_reachable_within_option": progressed if complete else None,
            "issued_intents": issued,
            "sealed_events": len(event_ids),
            "unresolved_intents": unresolved_count,
            "event_ids": event_ids,
            "unresolved_event_ids": [],
            "level_delta": max(0, final_levels - initial_levels),
            "errors": errors,
            "illegal_actions": illegal_actions,
            "game_over": game_over,
            "dynamic_regrounding_each_step": True,
            "physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    _write_signed_once(_work_path(destination, "branches", work, "receipt.json"), receipt)
    return receipt


def collect_phase(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    destination = _output(root)
    _require_audit(destination, manifest)
    runtime = runtime or _science._default_runtime_loader()
    receipts = []
    for work in build_work_specs("panel"):
        receipts.append(_run_work(destination, manifest, work, runtime))
        checkpoint = _checkpoint(destination, manifest)
        print(
            protocol.canonical_json(
                {
                    "phase": "t10_3_1_collect",
                    "completed_resets": checkpoint["completed_branch_count"],
                    "panel_resets": protocol.PANEL_RESETS,
                    "authorized_actions": checkpoint["authorized_intent_count"],
                    "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
                }
            ),
            flush=True,
        )
    checkpoint = _checkpoint(destination, manifest)
    return {
        "phase": "collect",
        "status": "PANEL_COLLECTION_COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "completed_panel_resets": len(receipts),
        "checkpoint_checksum": checkpoint["checkpoint_checksum"],
        "authorized_actions": checkpoint["authorized_intent_count"],
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        "physical_actions_replayed": 0,
        "firewall": _firewall(),
    }


def backfill_branch_labels(
    events: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_work = str(receipt.get("work_id", ""))
    label = receipt.get("goal_reachable_within_option")
    if receipt.get("complete") is not True or not isinstance(label, bool):
        label = None
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("work_id") != expected_work:
            raise IntegrityError("branch label attempted to cross a reset boundary")
        row = {
            "format_version": COMPILED_FORMAT_VERSION,
            "event_id": event["event_id"],
            "physical_event_checksum": event["event_checksum"],
            "work_id": expected_work,
            "phase": receipt["phase"],
            "source_game": receipt["game_id"],
            "seed": receipt["seed"],
            "controller": receipt["controller"],
            "reset_index": receipt["reset_index"],
            "step_index": event["step_index"],
            "complete": bool(label is not None and event.get("binding", {}).get("pre_action_complete")),
            "model_view": event.get("model_view", {}),
            "binding": event.get("binding", {}),
            "correspondence": event.get("correspondence", {}),
            "transport": event.get("transport", {}),
            "transport_certificates": event.get("transport_certificates", []),
            "observations": event.get("observations", {}),
            "labels": {"goal_reachable_within_option": label},
            "learned_predicates": ["goal_reachable_within_option"],
            "prefix": event.get("prefix", {}),
            "selection": {
                "controller": receipt["controller"],
                "step_index": event["step_index"],
                "action_name": event.get("action", {}).get("name"),
                "parameter_arity": event.get("action", {}).get("data", {}).get("parameter_arity", 0),
            },
            "provenance": {
                "physical_event_checksum": event["event_checksum"],
                "branch_receipt_checksum": receipt["receipt_checksum"],
                "parent_t10_3_event": False,
                "raw_runtime_state_retained": False,
                "game_seed_or_entity_identity_used_as_feature": False,
            },
        }
        rows.append(_signed(row, "compiled_checksum"))
    return rows


def _all_receipts(destination: Path, phase: str) -> list[dict[str, Any]]:
    rows = []
    for work in build_work_specs(phase):
        path = _work_path(destination, "branches", work, "receipt.json")
        if not path.exists():
            raise ScientificGateMiss(f"missing {phase} branch receipt: {work.work_id}")
        rows.append(_read_signed(path, "receipt_checksum"))
    return rows


def _compiled_rows(destination: Path) -> list[dict[str, Any]]:
    path = destination / COMPILED_LEDGER_FILENAME
    if not path.exists():
        raise ScientificGateMiss("compiled source ledger is absent")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid compiled line {line_number}") from exc
            unsigned = {key: value for key, value in row.items() if key != "compiled_checksum"}
            if _sha(unsigned) != row.get("compiled_checksum"):
                raise IntegrityError(f"compiled checksum drift at line {line_number}")
            rows.append(row)
    return rows


def build_qa_report(
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    *,
    manifest_checksum: str,
) -> dict[str, Any]:
    trials = sum(int(row.get("correspondence", {}).get("fraction_denominator", 0)) for row in rows)
    confident = sum(int(row.get("correspondence", {}).get("confident_matches", 0)) for row in rows)
    ambiguous = sum(int(row.get("correspondence", {}).get("fully_ambiguous_matches", 0)) for row in rows)
    confident_fraction = confident / max(1, trials)
    ambiguous_fraction = ambiguous / max(1, trials)
    nonterminal = [
        row
        for row in rows
        if not row.get("observations", {}).get("level_complete")
        and not row.get("observations", {}).get("game_over")
    ]
    coherent = 0
    comparable_certificates = []
    for row in nonterminal:
        frames = row.get("model_view", {}).get("frames", {})
        complete_frames = sum(
            bool(frame.get("complete"))
            for frame in frames.values()
            if isinstance(frame, Mapping)
        )
        if complete_frames >= 2 and row.get("transport", {}).get("multiframe_exact_nonidentity"):
            coherent += 1
        comparable_certificates.extend(
            certificate
            for certificate in row.get("transport_certificates", ())
            if certificate.get("comparable")
        )
    coherent_fraction = coherent / max(1, len(nonterminal))
    targets = [row.get("labels", {}).get("goal_reachable_within_option") for row in rows]
    known = [value for value in targets if isinstance(value, bool)]
    positives = sum(known)
    prevalence = positives / max(1, len(known))
    positive_games = {
        row.get("source_game")
        for row in rows
        if row.get("labels", {}).get("goal_reachable_within_option") is True
    }
    canonical = [
        receipt
        for receipt in receipts
        if receipt.get("game_id") in protocol.POSITIVE_WITNESS_GAMES
        and receipt.get("controller") == "canonical_option"
    ]
    reproduced = bool(
        len(canonical) == 8
        and all(
            receipt.get("goal_reachable_within_option") is True
            and receipt.get("errors") == []
            and receipt.get("illegal_actions") == 0
            and receipt.get("game_over") is False
            for receipt in canonical
        )
    )
    unknown_targets = sum(value is None for value in targets)
    unresolved = int(checkpoint.get("explicitly_unresolved_intent_count", 0))
    after_available = sum(bool(row.get("binding", {}).get("after_root_available")) for row in rows)
    checks = {
        "events_present": bool(rows),
        "intent_accounting_equation": bool(checkpoint.get("equation_holds")) and unresolved == 0,
        "branch_label_completeness": unknown_targets == 0,
        "root_correspondence_confident": confident_fraction >= 0.90,
        "root_ambiguity_below_limit": ambiguous_fraction < 0.10,
        "multiframe_coherent_prefixes": coherent_fraction >= 0.50,
        "exact_nonidentity_transport_present": bool(comparable_certificates),
        "comparable_transports_commutative": bool(
            comparable_certificates
            and all(item.get("commutativity", {}).get("exact") for item in comparable_certificates)
        ),
        "comparable_transports_round_trip_exact": bool(
            comparable_certificates
            and all(item.get("round_trip_exact") for item in comparable_certificates)
        ),
        "incomplete_projection_not_exact": all(
            not item.get("exact") or item.get("projection_complete")
            for row in rows
            for item in row.get("transport_certificates", ())
        ),
        "goal_reachable_prevalence": 0.005 <= prevalence <= 0.95,
        "goal_reachable_support": positives >= 32 and len(positive_games) >= 2,
        "positive_witness_reproduced": reproduced,
        "source_validation_closed": True,
        "ar25_closed": True,
        "holdout_closed": True,
    }
    passed = all(checks.values())
    return _signed(
        {
            "format_version": "sage-t10.3.1-qa-report-v1",
            "phase": "compile",
            "manifest_checksum": manifest_checksum,
            "status": "PASS_T10_3_1_QA" if passed else "QA_MISS",
            "passed": passed,
            "checks": checks,
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "metrics": {
                "event_count": len(rows),
                "nonterminal_prefix_count": len(nonterminal),
                "confident_root_correspondence_fraction": confident_fraction,
                "fully_ambiguous_root_fraction": ambiguous_fraction,
                "after_root_available_fraction": after_available / max(1, len(rows)),
                "multiframe_coherent_prefix_fraction": coherent_fraction,
                "exact_nonidentity_event_count": sum(
                    bool(row.get("transport", {}).get("multiframe_exact_nonidentity")) for row in rows
                ),
                "goal_reachable_positive_count": positives,
                "goal_reachable_total_count": len(known),
                "goal_reachable_prevalence": prevalence,
                "goal_reachable_positive_games": sorted(positive_games),
                "unknown_target_count": unknown_targets,
                "unresolved_intent_count": unresolved,
            },
            "fit_authorized": passed,
            "parent_t10_3_events_used_for_fit": 0,
            "firewall": _firewall(),
        },
        "report_checksum",
    )


def compile_phase(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = _output(root)
    _require_audit(destination, manifest)
    receipts = _all_receipts(destination, "panel")
    rows: list[dict[str, Any]] = []
    for work, receipt in zip(build_work_specs("panel"), receipts, strict=True):
        rows.extend(backfill_branch_labels(_read_events_for_work(destination, work), receipt))
        unresolved_dir = _work_path(destination, "unresolved", work, "placeholder").parent
        for path in sorted(unresolved_dir.glob("*.json")) if unresolved_dir.exists() else ():
            unresolved = _read_signed(path, "unresolved_checksum")
            rows.append(
                _signed(
                    {
                        "format_version": COMPILED_FORMAT_VERSION,
                        "event_id": unresolved["event_id"],
                        "work_id": work.work_id,
                        "phase": work.phase,
                        "source_game": work.game_id,
                        "seed": work.seed,
                        "controller": work.controller,
                        "reset_index": work.reset_index,
                        "step_index": unresolved["step_index"],
                        "complete": False,
                        "model_view": {},
                        "binding": {"pre_action_complete": False},
                        "correspondence": {},
                        "transport": {},
                        "transport_certificates": [],
                        "observations": {},
                        "labels": {"goal_reachable_within_option": None},
                        "learned_predicates": ["goal_reachable_within_option"],
                        "prefix": {"nonterminal": True, "evaluable": False, "coherent_frames": 0},
                        "selection": {"controller": work.controller, "step_index": unresolved["step_index"]},
                        "provenance": {
                            "unresolved_checksum": unresolved["unresolved_checksum"],
                            "physical_action_replayed": False,
                            "parent_t10_3_event": False,
                        },
                    },
                    "compiled_checksum",
                )
            )
    rows.sort(key=lambda row: (int(row["reset_index"]), int(row["step_index"])))
    _write_jsonl_once(destination / COMPILED_LEDGER_FILENAME, rows)
    ledger_bytes = (destination / COMPILED_LEDGER_FILENAME).read_bytes()
    compact = _signed(
        {
            "format_version": "sage-t10.3.1-compact-ledger-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "event_count": len(rows),
            "event_ids_sha256": _sha([row["event_id"] for row in rows]),
            "compiled_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "compiled_ledger_bytes": len(ledger_bytes),
            "parent_t10_3_event_count": 0,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        },
        "ledger_checksum",
    )
    _write_signed_once(destination / COMPACT_LEDGER_FILENAME, compact)
    checkpoint = _checkpoint(destination, manifest)
    qa = build_qa_report(
        rows,
        receipts,
        checkpoint,
        manifest_checksum=manifest["manifest_checksum"],
    )
    _write_signed_once(destination / QA_FILENAME, qa)
    if not qa["passed"]:
        raise ScientificGateMiss(protocol.canonical_json(qa))
    return qa


def _features(row: Mapping[str, Any], *, transport_enabled: bool = True) -> list[float]:
    base = _parent._features(row, transport_enabled=transport_enabled)
    transport = row.get("transport", {})
    return [
        *base,
        float(bool(transport.get("common_quotient_changed"))) if transport_enabled else 0.0,
        min(1.0, float(transport.get("common_quotient_delta_count", 0)) / 8.0)
        if transport_enabled
        else 0.0,
    ]


def _fit_logistic(
    rows: Sequence[Mapping[str, Any]], *, transport_enabled: bool = True
) -> list[float]:
    data = [
        (row, bool(row.get("labels", {}).get("goal_reachable_within_option")))
        for row in rows
        if isinstance(row.get("labels", {}).get("goal_reachable_within_option"), bool)
        and row.get("complete")
    ]
    if not data or len({label for _row, label in data}) < 2:
        raise ScientificGateMiss("cross-fit training fold lacks both target classes")
    weights = [0.0] * (len(_features(data[0][0])) + 1)
    for iteration in range(800):
        gradient = [0.0] * len(weights)
        for row, label in data:
            vector = [1.0, *_features(row, transport_enabled=transport_enabled)]
            probability = _parent._sigmoid(
                sum(weight * value for weight, value in zip(weights, vector, strict=True))
            )
            error = probability - float(label)
            for index, value in enumerate(vector):
                gradient[index] += error * value
        rate = 0.08 / (iteration + 1) ** 0.5
        for index in range(len(weights)):
            penalty = 0.002 * weights[index] if index else 0.0
            weights[index] -= rate * (gradient[index] / len(data) + penalty)
    return weights


def _predict(
    weights: Sequence[float], row: Mapping[str, Any], *, transport_enabled: bool = True
) -> float:
    vector = [1.0, *_features(row, transport_enabled=transport_enabled)]
    return _parent._sigmoid(
        sum(weight * value for weight, value in zip(weights, vector, strict=True))
    )


def _identity_probe(rows: Sequence[Mapping[str, Any]]) -> float:
    games = list(protocol.SOURCE_GAMES)
    train = [row for row in rows if int(row.get("seed", 0)) % 2 == 1 and row.get("complete")]
    test = [row for row in rows if int(row.get("seed", 0)) % 2 == 0 and row.get("complete")]
    centroids: dict[str, list[float]] = {}
    for game in games:
        vectors = [_features(row) for row in train if row.get("source_game") == game]
        if not vectors:
            return 1.0
        centroids[game] = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(vectors[0]))
        ]
    recalls = []
    for game in games:
        items = [row for row in test if row.get("source_game") == game]
        if not items:
            return 1.0
        correct = 0
        for row in items:
            vector = _features(row)
            predicted = min(
                games,
                key=lambda candidate: sum(
                    (value - center) ** 2
                    for value, center in zip(vector, centroids[candidate], strict=True)
                ),
            )
            correct += predicted == game
        recalls.append(correct / len(items))
    return sum(recalls) / len(recalls)


def build_model_recipe(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    complete_rows = [
        row
        for row in rows
        if row.get("complete")
        and isinstance(row.get("labels", {}).get("goal_reachable_within_option"), bool)
    ]
    folds: dict[str, Any] = {}
    predictions: dict[str, float] = {}
    for held_out in protocol.SOURCE_GAMES:
        train = [row for row in complete_rows if row.get("source_game") != held_out]
        test = [row for row in complete_rows if row.get("source_game") == held_out]
        weights = _fit_logistic(train)
        ablated_weights = _fit_logistic(train, transport_enabled=False)
        scores = [_predict(weights, row) for row in test]
        ablated_scores = [
            _predict(ablated_weights, row, transport_enabled=False) for row in test
        ]
        labels = [bool(row["labels"]["goal_reachable_within_option"]) for row in test]
        prevalence = sum(
            bool(row["labels"]["goal_reachable_within_option"]) for row in train
        ) / max(1, len(train))
        brier = sum((score - float(label)) ** 2 for score, label in zip(scores, labels, strict=True)) / max(1, len(labels))
        baseline = sum((prevalence - float(label)) ** 2 for label in labels) / max(1, len(labels))
        for row, score in zip(test, scores, strict=True):
            predictions[str(row["event_id"])] = score
        folds[held_out] = {
            "event_count": len(test),
            "auroc": _parent._auroc(labels, scores),
            "no_transport_auroc": _parent._auroc(labels, ablated_scores),
            "brier": brier,
            "baseline_brier": baseline,
            "brier_improvement": baseline - brier,
            "weights": [round(value, 12) for value in weights],
            "feature_names": [
                "intercept",
                "binding_complete",
                "complete_frame_fraction",
                "exact_nonidentity_transport",
                "exact_transport_fraction",
                "root_effect",
                "physical_no_effect",
                "step_fraction",
                "parameterized",
                "common_quotient_changed",
                "common_quotient_delta_fraction",
            ],
        }
    branch_scores: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in complete_rows:
        branch_scores[(str(row["source_game"]), int(row["seed"]), str(row["controller"]))].append(
            predictions[str(row["event_id"])]
        )
    ranks: dict[str, list[int]] = defaultdict(list)
    margins: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for game in protocol.SOURCE_GAMES:
        for seed in protocol.PANEL_SEEDS:
            scores = {
                controller: sum(branch_scores.get((game, seed, controller), ()))
                / max(1, len(branch_scores.get((game, seed, controller), ())))
                for controller in protocol.PANEL_ARMS
            }
            canonical = scores["canonical_option"]
            ranks[game].append(
                1
                + sum(
                    value > canonical
                    for controller, value in scores.items()
                    if controller != "canonical_option"
                )
            )
            margins[game]["binding_swap"].append(canonical - scores["binding_swap"])
            margins[game]["option_intervention"].append(
                canonical - scores["option_intervention"]
            )
    positive_checks = {}
    for game in protocol.POSITIVE_WITNESS_GAMES:
        positive_checks[game] = {
            "auroc": folds[game]["auroc"] >= 0.75,
            "brier_improvement": folds[game]["brier_improvement"] > 0.0,
            "top_8": max(ranks[game], default=99) <= 8,
            "median_rank": _parent._median(ranks[game]) <= 4,
            "binding_swap_margin": min(margins[game]["binding_swap"], default=-1.0) > 0.0,
            "option_intervention_margin": min(
                margins[game]["option_intervention"], default=-1.0
            )
            > 0.0,
        }
    full_mean = sum(fold["auroc"] for fold in folds.values()) / len(folds)
    ablated_mean = sum(fold["no_transport_auroc"] for fold in folds.values()) / len(folds)
    identity_accuracy = _identity_probe(complete_rows)
    checks = {
        "positive_game_auroc": all(item["auroc"] for item in positive_checks.values()),
        "positive_game_brier_improvement": all(
            item["brier_improvement"] for item in positive_checks.values()
        ),
        "positive_option_top_8": all(item["top_8"] for item in positive_checks.values()),
        "positive_option_median_rank": all(
            item["median_rank"] for item in positive_checks.values()
        ),
        "binding_swap_margins": all(
            item["binding_swap_margin"] for item in positive_checks.values()
        ),
        "option_intervention_margins": all(
            item["option_intervention_margin"] for item in positive_checks.values()
        ),
        "identity_only_no_transport_degradation": full_mean > ablated_mean,
        "identity_probe_limit": identity_accuracy - (1.0 / 3.0) <= 0.10,
    }
    manifest_proxy = {
        "manifest_checksum": manifest["manifest_checksum"],
        "handoff_receipt": {
            "canonical_witnesses": manifest["migration_receipt"]["canonical_witnesses"]
        },
    }
    seed_alias = dict(zip(protocol.PANEL_SEEDS, _parent.protocol.PANEL_SEEDS, strict=True))
    contract_rows = [
        {**json.loads(protocol.canonical_json(row)), "seed": seed_alias.get(int(row.get("seed", 0)), 0)}
        for row in complete_rows
    ]
    contract_recipe = _parent.build_model_recipe(contract_rows, manifest_proxy)
    passed = all(checks.values())
    return _signed(
        {
            "format_version": "sage-t10.3.1-model-recipe-v1",
            "phase": "fit",
            "manifest_checksum": manifest["manifest_checksum"],
            "status": "PASS_T10_3_1_OPTION_FIT" if passed else "OPTION_INDUCTION_MISS",
            "passed": passed,
            "target": "goal_reachable_within_option",
            "cross_fit": folds,
            "positive_game_checks": positive_checks,
            "option_ranks": {game: values for game, values in sorted(ranks.items())},
            "paired_margins": {
                game: {name: values for name, values in sorted(items.items())}
                for game, items in sorted(margins.items())
            },
            "identity_probe_balanced_accuracy": identity_accuracy,
            "identity_probe_excess": identity_accuracy - (1.0 / 3.0),
            "full_mean_auroc": full_mean,
            "no_transport_mean_auroc": ablated_mean,
            "checks": checks,
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "joint_program_hypotheses": contract_recipe["joint_program_hypotheses"],
            "mixed_option_automata": contract_recipe["mixed_option_automata"],
            "factorized_posterior": contract_recipe["factorized_posterior"],
            "fresh_panel_seeds": list(protocol.PANEL_SEEDS),
            "seed_used_as_feature": False,
            "feature_firewall": contract_recipe["feature_firewall"],
            "parent_t10_3_events_used": 0,
            "automatic_retuning_performed": False,
            "firewall": _firewall(),
        },
        "recipe_checksum",
    )


def fit_phase(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = _output(root)
    qa = _read_signed(destination / QA_FILENAME, "report_checksum")
    if qa.get("passed") is not True:
        raise ScientificGateMiss("QA gate forbids model fit")
    recipe = build_model_recipe(_compiled_rows(destination), manifest)
    _write_signed_once(destination / MODEL_FILENAME, recipe)
    return recipe


def build_confirmation_report(
    receipts: Sequence[Mapping[str, Any]], *, manifest_checksum: str, recipe_checksum: str
) -> dict[str, Any]:
    parent = _parent.build_confirmation_report(
        receipts,
        manifest_checksum=manifest_checksum,
        recipe_checksum=recipe_checksum,
    )
    unsigned = {key: value for key, value in parent.items() if key != "report_checksum"}
    unsigned.update(
        {
            "format_version": "sage-t10.3.1-confirmation-report-v1",
            "status": "PASS_T10_3_1_SOURCE_CONFIRMATION" if parent["passed"] else "SOURCE_CONFIRMATION_MISS",
            "fresh_confirmation_seeds": list(protocol.CONFIRMATION_SEEDS),
            "parent_t10_3_events_used": 0,
        }
    )
    return _signed(unsigned, "report_checksum")


def confirm_phase(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    destination = _output(root)
    recipe = _read_signed(destination / MODEL_FILENAME, "recipe_checksum")
    if recipe.get("passed") is not True:
        raise ScientificGateMiss("model gate forbids source confirmation")
    runtime = runtime or _science._default_runtime_loader()
    receipts = []
    for work in build_work_specs("confirmation"):
        receipts.append(_run_work(destination, manifest, work, runtime))
        checkpoint = _checkpoint(destination, manifest)
        print(
            protocol.canonical_json(
                {
                    "phase": "t10_3_1_confirm",
                    "completed_resets": checkpoint["completed_branch_count"],
                    "total_resets": protocol.TOTAL_RESETS,
                    "authorized_actions": checkpoint["authorized_intent_count"],
                    "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
                }
            ),
            flush=True,
        )
    report = build_confirmation_report(
        receipts,
        manifest_checksum=manifest["manifest_checksum"],
        recipe_checksum=recipe["recipe_checksum"],
    )
    _write_signed_once(destination / CONFIRMATION_FILENAME, report)
    return report


def _verdict(
    audit: Mapping[str, Any],
    qa: Mapping[str, Any],
    model: Mapping[str, Any] | None,
    confirmation: Mapping[str, Any] | None,
) -> str:
    if audit.get("passed") is not True:
        return "PROVENANCE_INVALID"
    checks = qa.get("checks", {})
    if not checks.get("root_correspondence_confident") or not checks.get("root_ambiguity_below_limit"):
        return "ROOTING_MISS"
    if not checks.get("positive_witness_reproduced"):
        return "WITNESS_REPRODUCTION_MISS"
    if qa.get("passed") is not True:
        return "QA_MISS"
    if model is None:
        raise ScientificGateMiss("fit phase has not produced a model recipe")
    model_checks = model.get("checks", {})
    if (
        not model_checks.get("positive_game_auroc")
        or not model_checks.get("positive_game_brier_improvement")
        or not model_checks.get("binding_swap_margins")
        or not model_checks.get("option_intervention_margins")
    ):
        return "CAUSAL_SEMANTICS_MISS"
    if model.get("passed") is not True:
        return "OPTION_INDUCTION_MISS"
    if confirmation is None:
        raise ScientificGateMiss("confirmation phase has not completed")
    if confirmation.get("passed") is not True:
        return "SOURCE_CONFIRMATION_MISS"
    return "PASS_T10_3_SOURCE_PILOT"


def report_phase(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = _output(root)
    audit = _read_signed(destination / AUDIT_FILENAME, "audit_checksum")
    qa = _read_signed(destination / QA_FILENAME, "report_checksum")
    model_path = destination / MODEL_FILENAME
    confirmation_path = destination / CONFIRMATION_FILENAME
    model = _read_signed(model_path, "recipe_checksum") if model_path.exists() else None
    confirmation = (
        _read_signed(confirmation_path, "report_checksum")
        if confirmation_path.exists()
        else None
    )
    verdict = _verdict(audit, qa, model, confirmation)
    checkpoint = _checkpoint(destination, manifest)
    passed = verdict == "PASS_T10_3_SOURCE_PILOT"
    report = _signed(
        {
            "format_version": "sage-t10.3.1-terminal-report-v1",
            "phase": "report",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_terminal_checksum": protocol.PARENT_TERMINAL_CHECKSUM,
            "verdict": verdict,
            "status": verdict,
            "passed": passed,
            "audit_checksum": audit["audit_checksum"],
            "qa_report_checksum": qa["report_checksum"],
            "model_recipe_checksum": model.get("recipe_checksum") if model else None,
            "confirmation_report_checksum": confirmation.get("report_checksum") if confirmation else None,
            "accounting": {
                "authorized_intent_count": checkpoint["authorized_intent_count"],
                "sealed_event_count": checkpoint["sealed_event_count"],
                "explicitly_unresolved_intent_count": checkpoint["explicitly_unresolved_intent_count"],
                "equation_holds": checkpoint["equation_holds"],
                "physical_actions_replayed": 0,
                "maximum_authorized_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            },
            "authorization": {
                "separate_validation_protocol_preparation": passed,
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
            "parent_t10_3_events_used_for_fit": 0,
            "automatic_retuning_performed": False,
            "firewall": _firewall(),
        },
        "terminal_checksum",
    )
    _write_signed_once(destination / TERMINAL_FILENAME, report)
    return report


def status_phase(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = _output(root)
    artifacts = {
        "audit": (AUDIT_FILENAME, "audit_checksum"),
        "qa": (QA_FILENAME, "report_checksum"),
        "model": (MODEL_FILENAME, "recipe_checksum"),
        "confirmation": (CONFIRMATION_FILENAME, "report_checksum"),
        "terminal": (TERMINAL_FILENAME, "terminal_checksum"),
    }
    status = {}
    for name, (filename, checksum_key) in artifacts.items():
        path = destination / filename
        status[name] = (
            {"present": True, "checksum": _read_signed(path, checksum_key)[checksum_key]}
            if path.exists()
            else {"present": False}
        )
    checkpoint = (
        _checkpoint(destination, manifest)
        if destination.exists()
        else {
            "completed_branch_count": 0,
            "authorized_intent_count": 0,
            "sealed_event_count": 0,
            "explicitly_unresolved_intent_count": 0,
        }
    )
    return {
        "phase": "status",
        "status": "READY_T10_3_1_SOURCE_PILOT",
        "manifest_checksum": manifest["manifest_checksum"],
        "artifacts": status,
        "progress": {
            "completed_resets": checkpoint["completed_branch_count"],
            "total_resets": protocol.TOTAL_RESETS,
            "authorized_actions": checkpoint["authorized_intent_count"],
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "sealed_events": checkpoint["sealed_event_count"],
            "unresolved_intents": checkpoint["explicitly_unresolved_intent_count"],
            "physical_actions_replayed": 0,
        },
        "firewall": _firewall(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=protocol.PHASES)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--manifest", default=str(protocol.DEFAULT_MANIFEST_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = protocol._root(args.repo_root)
    try:
        if args.phase == "freeze":
            payload = protocol.freeze_manifest(output_path=args.manifest, repo_root=root)
        else:
            manifest = protocol.load_manifest(args.manifest, repo_root=root)
            dispatch = {
                "status": status_phase,
                "audit": audit_phase,
                "collect": collect_phase,
                "compile": compile_phase,
                "fit": fit_phase,
                "confirm": confirm_phase,
                "report": report_phase,
            }
            payload = dispatch[args.phase](root, manifest)
    except ScientificGateMiss as exc:
        message = str(exc)
        print(
            message
            if message.startswith("{")
            else protocol.canonical_json(
                {"phase": args.phase, "status": "SCIENTIFIC_GATE_MISS", "error": message}
            )
        )
        return 3
    except (
        protocol.ManifestDriftError,
        protocol.ProtocolError,
        IntegrityError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(protocol.canonical_json({"phase": args.phase, "error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(protocol.canonical_json(payload))
    if args.phase in {"audit", "fit", "confirm", "report"} and payload.get("passed") is False:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
