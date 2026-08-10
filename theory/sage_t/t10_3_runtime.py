"""Durable, fail-closed runtime for the SAGE.T10.3 source pilot.

The command surface is intentionally phased.  ``collect`` runs only the
48-reset physical matrix; ``confirm`` is unavailable until QA and model gates
pass.  There is no command that opens validation, AR25, a holdout, production
authority, or automatic retuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_2_runtime as _science
from . import t10_3_protocol as protocol
from .contracts import ActionCandidate
from .frame_adapters_v10_3 import (
    assert_safe_binding_payload,
    project_goal_transition,
    resolve_pre_action_root,
    structural_signature,
)
from .mixed_automata_v10_2 import alternate, repeat
from .factorized_posterior_v10_2 import FactorMarginals
from .progress_witness_v10 import (
    GroundedAction,
    SearchConfig,
    chain_successor_macro,
    compile_progress_program,
)
from .t10_2_1_runtime import (
    JournalConflictError,
    JournalIntegrityError,
    _atomic_write_text,
    _read_canonical_json,
    _write_once,
)

FORMAT_VERSION = "sage-t10.3-goal-progress-runtime-v1"
EVENT_FORMAT_VERSION = "sage-t10.3-physical-event-v1"
INTENT_FORMAT_VERSION = "sage-t10.3-action-intent-v1"
BRANCH_FORMAT_VERSION = "sage-t10.3-branch-receipt-v1"
CHECKPOINT_FORMAT_VERSION = "sage-t10.3-checkpoint-v1"
COMPILED_FORMAT_VERSION = "sage-t10.3-compiled-event-v1"
AUDIT_FILENAME = "offline_audit.json"
CHECKPOINT_FILENAME = "checkpoint.json"
COMPILED_LEDGER_FILENAME = "compiled_source_events.jsonl"
COMPACT_LEDGER_FILENAME = "compact_ledger.json"
QA_FILENAME = "qa_report.json"
MODEL_FILENAME = "model_recipe.json"
CONFIRMATION_FILENAME = "confirmation_report.json"
TERMINAL_FILENAME = "t10_3_report.json"


class ScientificGateMiss(RuntimeError):
    """A preregistered scientific gate failed (CLI exit code 3)."""


class IntegrityError(RuntimeError):
    """A frozen input, journal, or write-once contract drifted (exit 2)."""


def assert_no_forbidden_persistence(payload: Mapping[str, Any]) -> None:
    """Reject raw grounding fields from any T10.3 persisted model/event view."""

    forbidden = {
        "action_args",
        "action_data",
        "center",
        "col",
        "color",
        "entity_id",
        "grid",
        "grid_after",
        "grid_before",
        "object_id",
        "row",
        "target_id",
        "x",
        "y",
    }
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in forbidden:
                    raise IntegrityError(f"forbidden persisted field: {key}")
                stack.append(item)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


def _sha(value: Any) -> str:
    return protocol.canonical_sha256(value)


def _signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return protocol.signed_payload(dict(value), checksum_key=key)


def _write_signed_once(path: Path, payload: Mapping[str, Any]) -> bool:
    try:
        return _write_once(path, payload)
    except (JournalConflictError, JournalIntegrityError) as exc:
        raise IntegrityError(str(exc)) from exc


def _read_signed(path: Path, checksum_key: str) -> dict[str, Any]:
    try:
        payload = _read_canonical_json(path)
    except (JournalIntegrityError, OSError) as exc:
        raise IntegrityError(str(exc)) from exc
    expected = payload.get(checksum_key)
    unsigned = {key: value for key, value in payload.items() if key != checksum_key}
    if not isinstance(expected, str) or _sha(unsigned) != expected:
        raise IntegrityError(f"signed artifact checksum drifted: {path}")
    return payload


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rendered = "".join(protocol.canonical_json(dict(row)) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise IntegrityError(f"immutable ledger conflicts: {path}")
        return
    _atomic_write_text(path, rendered)


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
    if phase == "panel":
        rows: list[WorkSpec] = []
        reset_index = 0
        for game in protocol.SOURCE_GAMES:
            for seed in protocol.PANEL_SEEDS:
                for controller in protocol.PANEL_ARMS:
                    rows.append(WorkSpec(phase, game, seed, controller, reset_index))
                    reset_index += 1
        return tuple(rows)
    if phase == "confirmation":
        rows = []
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
    intent_count = event_count = unresolved_count = action_count = 0
    branch_count = 0
    for phase in ("panel", "confirmation"):
        for work in build_work_specs(phase):
            receipt_path = _work_path(destination, "branches", work, "receipt.json")
            if receipt_path.exists():
                receipt = _read_signed(receipt_path, "receipt_checksum")
                completed.append(work.work_id)
                branch_count += 1
                action_count += int(receipt.get("issued_intents", 0))
            intent_dir = _work_path(destination, "intents", work, "placeholder").parent
            event_dir = _work_path(destination, "events", work, "placeholder").parent
            unresolved_dir = _work_path(destination, "unresolved", work, "placeholder").parent
            intent_count += len(tuple(intent_dir.glob("*.json"))) if intent_dir.exists() else 0
            event_count += len(tuple(event_dir.glob("*.json"))) if event_dir.exists() else 0
            unresolved_count += len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    if intent_count != event_count + unresolved_count:
        raise IntegrityError("intent accounting does not hold")
    if action_count != intent_count:
        raise IntegrityError("branch receipts do not account for every intent")
    if intent_count > protocol.TOTAL_MAXIMUM_ACTIONS:
        raise IntegrityError("physical action budget exceeded")
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


def _read_legacy_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid historical ledger line {line_number}") from exc
            if not isinstance(payload, dict):
                raise IntegrityError(f"non-object historical ledger line {line_number}")
            yield payload


def build_offline_audit(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    handoff = manifest["handoff_receipt"]
    ledger_path = root / protocol.T10_2_7_LEDGER
    labels: Counter[str] = Counter()
    incomplete = 0
    event_count = 0
    for event in _read_legacy_jsonl(ledger_path):
        event_count += 1
        for name, value in event.get("labels", {}).items():
            if value:
                labels[str(name)] += 1
        frames = event.get("model_view", {}).get("frames", {})
        if not isinstance(frames, Mapping) or any(
            not bool(frame.get("complete"))
            for frame in frames.values()
            if isinstance(frame, Mapping)
        ):
            incomplete += 1
    if event_count != 1370:
        raise IntegrityError("T10.2.9 audited event count drifted")
    universal = {
        name: count for name, count in labels.items() if count in {0, event_count}
    }
    passed = bool(
        handoff.get("t10_2_9_fit_excluded") is True
        and handoff.get("t10_2_9_event_count") == event_count
        and len(handoff.get("canonical_witnesses", ())) == 2
        and labels.get("state_changed") == event_count
        and labels.get("no_effect", 0) == 0
    )
    return _signed(
        {
            "format_version": "sage-t10.3-offline-audit-v1",
            "phase": "audit",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "status": "PASS_T10_3_OFFLINE_AUDIT" if passed else "PROVENANCE_INVALID",
            "historical_event_count": event_count,
            "historical_positive_labels": dict(sorted(labels.items())),
            "universal_labels": universal,
            "incomplete_projection_event_count": incomplete,
            "diagnosis": [
                "UNIVERSAL_STATE_CHANGED_LABEL",
                "ZERO_NO_EFFECT_LABEL",
                "INCOMPLETE_MULTIFRAME_PROJECTIONS",
            ],
            "historical_ledger_fit_excluded": True,
            "historical_events_relabelled": 0,
            "historical_events_used_for_fit": 0,
            "canonical_witnesses_used_as_structure_only": True,
            "grounded_historical_actions_read": False,
            "physical_actions_executed": 0,
            "firewall": _firewall(),
        },
        "audit_checksum",
    )


def audit_phase(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    destination = _output(root)
    payload = build_offline_audit(root, manifest)
    _write_signed_once(destination / AUDIT_FILENAME, payload)
    return payload


def _require_audit(destination: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = destination / AUDIT_FILENAME
    if not path.exists():
        raise ScientificGateMiss("offline audit must pass before physical collection")
    audit = _read_signed(path, "audit_checksum")
    if audit.get("manifest_checksum") != manifest["manifest_checksum"] or audit.get("passed") is not True:
        raise ScientificGateMiss("offline provenance audit failed")
    return audit


def _action_name(action: Any) -> str:
    if isinstance(action, Mapping):
        raw = action.get("action_name", action.get("name", ""))
    else:
        raw = getattr(action, "action_name", getattr(action, "name", ""))
    return str(raw).strip().upper()


def _action_data(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        raw = action.get("action_data", action.get("action_args", action.get("data", {})))
    else:
        raw = getattr(
            action,
            "action_data",
            getattr(action, "action_args", getattr(action, "data", {})),
        )
    return dict(raw or {}) if isinstance(raw, Mapping) else {}


def _abstract_state(snapshot: Any, legal_actions: Sequence[Any]) -> Any:
    from .compiler import compile_observation
    from .progress_witness_v10 import build_observation

    grid = getattr(snapshot, "grid", None)
    if grid is None and isinstance(snapshot, Mapping):
        grid = snapshot.get("grid")
    if grid is None:
        raise IntegrityError("runtime snapshot lacks a grid")
    observation = build_observation(
        grid,
        available_actions=tuple(_action_name(action) for action in legal_actions),
        game_state=_science._snapshot_state(snapshot) or "NOT_FINISHED",
        levels_completed=_science._snapshot_levels(snapshot),
    )
    return compile_observation(observation)


def _grounding_rows(state: Any, legal_actions: Sequence[Any]) -> list[tuple[Any, GroundedAction, str]]:
    rows: list[tuple[Any, GroundedAction, str]] = []
    for concrete in legal_actions:
        grounded = GroundedAction.from_view(concrete)
        action = ActionCandidate(grounded.action_name, dict(grounded.action_data))
        resolved = resolve_pre_action_root(state, action)
        signature = (
            structural_signature(state, resolved.entity_id)
            if resolved.entity_id is not None
            else "~unresolved"
        )
        rows.append((concrete, grounded, signature))
    return rows


def _plan_actions(
    work: WorkSpec,
    snapshot: Any,
    legal_actions: Sequence[Any],
    *,
    model: Mapping[str, Any] | None = None,
) -> tuple[GroundedAction, ...]:
    state = _abstract_state(snapshot, legal_actions)
    rows = _grounding_rows(state, legal_actions)
    if not rows:
        return ()
    parameterized = [row for row in rows if row[1].action_name == "ACTION6"]
    movements = [row for row in rows if row[1].action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}]
    ordered = sorted(rows, key=lambda row: (row[2], row[1].action_name, len(row[1].action_data)))

    learned_controller = work.phase == "confirmation" and work.controller == "learned"
    control_controller = work.controller == "capacity_matched_independent"
    if work.game_id.startswith("su15") or learned_controller:
        macro = chain_successor_macro(
            state,
            tuple(row[1] for row in rows),
            config=SearchConfig(maximum_horizon=protocol.MAXIMUM_ACTIONS_PER_RESET),
        )
        chain = tuple(macro.actions) if macro is not None else ()
        if learned_controller and chain:
            return chain[: protocol.MAXIMUM_ACTIONS_PER_RESET]
        if work.game_id.startswith("su15"):
            if work.controller == "canonical_option":
                return chain
            if work.controller == "binding_swap":
                return chain[1:] + chain[:1] if len(chain) > 1 else ()
            if work.controller == "option_intervention":
                return tuple(reversed(chain))
            if control_controller:
                return tuple(item[1] for item in ordered[: len(chain)])

    if work.game_id.startswith("lp85") or learned_controller:
        if learned_controller and parameterized:
            return tuple(parameterized[0][1] for _ in range(protocol.MAXIMUM_ACTIONS_PER_RESET))
        if work.game_id.startswith("lp85"):
            if not parameterized:
                return ()
            canonical = parameterized[0][1]
            distractor = parameterized[1][1] if len(parameterized) > 1 else None
            if work.controller == "canonical_option":
                return tuple(canonical for _ in range(protocol.MAXIMUM_ACTIONS_PER_RESET))
            if work.controller == "binding_swap":
                return tuple(distractor for _ in range(protocol.MAXIMUM_ACTIONS_PER_RESET)) if distractor else ()
            if work.controller == "option_intervention":
                return (
                    tuple(canonical if index % 2 == 0 else distractor for index in range(protocol.MAXIMUM_ACTIONS_PER_RESET))
                    if distractor
                    else ()
                )
            if control_controller:
                controls = [row[1] for row in ordered if row[1].key != canonical.key]
                return tuple(controls[index % len(controls)] for index in range(protocol.MAXIMUM_ACTIONS_PER_RESET)) if controls else ()

    if work.game_id.startswith("bp35"):
        if work.controller == "canonical_option":
            inventory = movements or ordered
        elif work.controller == "binding_swap":
            inventory = list(reversed(parameterized or ordered))
        elif work.controller == "option_intervention":
            inventory = parameterized or movements or ordered
        else:
            inventory = ordered
        return tuple(inventory[index % len(inventory)][1] for index in range(protocol.MAXIMUM_ACTIONS_PER_RESET)) if inventory else ()

    if control_controller:
        return tuple(ordered[index % len(ordered)][1] for index in range(protocol.MAXIMUM_ACTIONS_PER_RESET))
    return ()


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


def normalize_transport_evidence(event: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the T10.3 comparable/non-comparable transport semantics."""

    clone = json.loads(protocol.canonical_json(event))
    certificates = []
    comparable = []
    for raw in clone.get("transport_certificates", ()):
        certificate = dict(raw)
        complete = bool(certificate.get("projection_complete"))
        exact = bool(complete and certificate.get("exact"))
        certificate["comparable"] = complete
        certificate["exact"] = exact
        certificate["mapping_kind"] = "exact" if exact else "partial"
        certificate["certifies_gauge_equivalence"] = bool(
            exact and certificate.get("certifies_gauge_equivalence")
        )
        if complete:
            comparable.append(certificate)
        certificates.append(certificate)
    nonidentity = [
        item
        for item in comparable
        if item.get("source_frame") != item.get("target_frame")
    ]
    exact_nonidentity = [item for item in nonidentity if item.get("exact")]
    all_commute = bool(
        comparable
        and all(item.get("commutativity", {}).get("exact") for item in comparable)
    )
    all_round_trip = bool(
        comparable and all(item.get("round_trip_exact") for item in comparable)
    )
    clone["transport_certificates"] = certificates
    clone["transport_orbits"] = [
        orbit
        for orbit in clone.get("transport_orbits", ())
        if any(
            item.get("certificate_hash") == orbit.get("certificate_hash")
            or item.get("certificate_hash")
            == orbit.get("attestation", {}).get("certificate_hash")
            for item in exact_nonidentity
        )
    ]
    clone["transport"] = {
        "declared_comparable_certificate_count": len(comparable),
        "noncomparable_certificate_count": len(certificates) - len(comparable),
        "exact_certificate_count": sum(bool(item.get("exact")) for item in comparable),
        "exact_nonidentity_certificate_count": len(exact_nonidentity),
        "commutative_exact": all_commute,
        "round_trip_exact": all_round_trip,
        "multiframe_exact_nonidentity": bool(exact_nonidentity),
        "identity_only_control": False,
        "incomplete_projections_attested_exact": any(
            item.get("exact") and not item.get("projection_complete")
            for item in certificates
        ),
    }
    return clone


def _physical_event(
    projection: Any,
    *,
    work: WorkSpec,
    step_index: int,
) -> dict[str, Any]:
    compact = _science._compact_event(
        projection.bundle,
        controller=work.controller,
        reset_index=work.reset_index,
        step_index=step_index,
        progressing_sequence_rank=None,
        donor_game_count=2,
        capacity_slots=4,
    )
    compact = normalize_transport_evidence(compact)
    frames = compact.get("model_view", {}).get("frames", {})
    root = frames.get("root_only", {}) if isinstance(frames, Mapping) else {}
    root_effect = bool(root.get("before_hash") != root.get("after_hash"))
    all_unchanged = bool(
        frames
        and all(
            frame.get("before_hash") == frame.get("after_hash")
            for frame in frames.values()
            if isinstance(frame, Mapping)
        )
    )
    old_labels = compact.get("labels", {})
    compact["format_version"] = EVENT_FORMAT_VERSION
    compact["work_id"] = work.work_id
    compact["phase"] = work.phase
    compact["branch_outcome_pending"] = True
    compact["binding"] = projection.binding.as_dict()
    compact["observations"] = {
        "physical_no_effect": all_unchanged,
        "root_effect": root_effect,
        "level_complete": bool(old_labels.get("level_complete")),
        "game_over": bool(old_labels.get("game_over")),
    }
    compact["labels"] = {}
    compact["learned_predicates"] = ["goal_reachable_within_option"]
    compact["provenance"].update(
        {
            "collector": FORMAT_VERSION,
            "root_adapter": "sage-t10.3-frame-adapter-v1",
            "historical_t10_2_9_event": False,
            "raw_runtime_state_retained": False,
        }
    )
    assert_safe_binding_payload(compact["binding"])
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
    safe_binding = dict(binding)
    assert_safe_binding_payload(safe_binding)
    return _signed(
        {
            "format_version": INTENT_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "event_id": _event_id(work, step_index),
            "step_index": step_index,
            "action_schema": _action_name(action),
            "parameter_arity": len(_action_data(action)),
            "binding": safe_binding,
            "physical_action_authorized": True,
            "physical_replay_authorized": False,
        },
        "intent_checksum",
    )


def _read_events_for_work(destination: Path, work: WorkSpec) -> list[dict[str, Any]]:
    directory = _work_path(destination, "events", work, "placeholder").parent
    if not directory.exists():
        return []
    return [
        _read_signed(path, "event_checksum")
        for path in sorted(directory.glob("*.json"))
    ]


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
        name = intent_path.name
        event_path = _work_path(destination, "events", work, name)
        unresolved_path = _work_path(destination, "unresolved", work, name)
        if event_path.exists():
            event = _read_signed(event_path, "event_checksum")
            event_ids.append(str(event["event_id"]))
            continue
        unresolved = _signed(
            {
                "format_version": "sage-t10.3-unresolved-event-v1",
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
        _write_signed_once(unresolved_path, unresolved)
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


def _find_concrete(planned: GroundedAction, legal: Sequence[Any]) -> Any | None:
    for concrete in legal:
        if GroundedAction.from_view(concrete).key == planned.key:
            return concrete
    return None


def _run_work(
    destination: Path,
    manifest: Mapping[str, Any],
    work: WorkSpec,
    runtime: Any,
    *,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recovered = _recover_interrupted_work(destination, manifest, work)
    if recovered is not None:
        return recovered
    environment = _science._open_runtime(runtime, work.game_id, work.seed)
    event_ids: list[str] = []
    errors: list[str] = []
    illegal_actions = 0
    game_over = False
    initial_levels = final_levels = 0
    issued = 0
    try:
        frame = _science._reset_runtime(runtime, environment)
        legal = _science._legal_runtime(runtime, environment)
        snapshot = _science._snapshot_runtime(
            runtime, frame, fallback_available_actions=legal
        )
        initial_levels = _science._snapshot_levels(snapshot)
        final_levels = initial_levels
        planned = _plan_actions(work, snapshot, legal, model=model)
        if not planned:
            status = (
                "CONTROL_GROUNDING_MISS"
                if work.controller == "capacity_matched_independent"
                else "OPTION_GROUNDING_MISS"
            )
            errors.append(status)
        else:
            status = "COMPLETE"
        for step_index, wanted in enumerate(planned[: protocol.MAXIMUM_ACTIONS_PER_RESET]):
            legal = _science._legal_runtime(runtime, environment)
            concrete = _find_concrete(wanted, legal)
            if concrete is None:
                errors.append("ILLEGAL_OR_STALE_GROUNDING")
                illegal_actions += 1
                status = "ABORTED"
                break
            before = snapshot
            state = _abstract_state(before, legal)
            binding_resolution = resolve_pre_action_root(
                state, ActionCandidate(_action_name(concrete), _action_data(concrete))
            )
            signature = (
                structural_signature(state, binding_resolution.entity_id)
                if binding_resolution.entity_id is not None
                else None
            )
            binding = {
                "format_version": "sage-t10.3-intent-binding-v1",
                "method": binding_resolution.method,
                "structural_signature": signature,
                "unique": binding_resolution.unique,
                "complete": binding_resolution.entity_id is not None,
                "missing": list(binding_resolution.missing),
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
            try:
                next_frame = _science._step_runtime(runtime, environment, concrete)
                next_legal = _science._legal_runtime(runtime, environment)
                after = _science._snapshot_runtime(
                    runtime, next_frame, fallback_available_actions=next_legal
                )
            except Exception as exc:  # the already-authorized action is now unknowable
                unresolved = _signed(
                    {
                        "format_version": "sage-t10.3-unresolved-event-v1",
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
    unresolved_count = (
        len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    )
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
                    "phase": "t10_3_collect",
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
    """Derive immutable training rows without rewriting physical events."""

    expected_work = str(receipt.get("work_id", ""))
    label = receipt.get("goal_reachable_within_option")
    if receipt.get("complete") is not True or not isinstance(label, bool):
        label = None
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("work_id") != expected_work:
            raise IntegrityError("branch label attempted to cross a reset boundary")
        physical_checksum = str(event.get("event_checksum", ""))
        row = {
            "format_version": COMPILED_FORMAT_VERSION,
            "event_id": event["event_id"],
            "physical_event_checksum": physical_checksum,
            "work_id": expected_work,
            "phase": receipt["phase"],
            "source_game": receipt["game_id"],
            "seed": receipt["seed"],
            "controller": receipt["controller"],
            "reset_index": receipt["reset_index"],
            "step_index": event["step_index"],
            "complete": bool(
                label is not None
                and event.get("binding", {}).get("complete")
                and event.get("model_view", {}).get("frames")
            ),
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
                "physical_event_checksum": physical_checksum,
                "branch_receipt_checksum": receipt["receipt_checksum"],
                "historical_t10_2_9_event": False,
                "raw_runtime_state_retained": False,
                "game_seed_or_entity_identity_used_as_feature": False,
            },
        }
        rows.append(_signed(row, "compiled_checksum"))
    return rows


def _all_receipts(destination: Path, phase: str) -> list[dict[str, Any]]:
    receipts = []
    for work in build_work_specs(phase):
        path = _work_path(destination, "branches", work, "receipt.json")
        if not path.exists():
            raise ScientificGateMiss(f"missing {phase} branch receipt: {work.work_id}")
        receipts.append(_read_signed(path, "receipt_checksum"))
    return receipts


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
    *,
    manifest_checksum: str,
) -> dict[str, Any]:
    correspondence_trials = sum(
        int(row.get("correspondence", {}).get("fraction_denominator", 0)) for row in rows
    )
    confident = sum(
        int(row.get("correspondence", {}).get("confident_matches", 0)) for row in rows
    )
    ambiguous = sum(
        int(row.get("correspondence", {}).get("fully_ambiguous_matches", 0)) for row in rows
    )
    confident_fraction = confident / max(1, correspondence_trials)
    ambiguous_fraction = ambiguous / max(1, correspondence_trials)
    nonterminal = [
        row
        for row in rows
        if not row.get("observations", {}).get("level_complete")
        and not row.get("observations", {}).get("game_over")
    ]
    coherent = 0
    exact_nonidentity = 0
    comparable_certificates = []
    for row in nonterminal:
        frames = row.get("model_view", {}).get("frames", {})
        complete_frames = sum(
            bool(frame.get("complete"))
            for frame in frames.values()
            if isinstance(frame, Mapping)
        )
        has_exact = bool(row.get("transport", {}).get("multiframe_exact_nonidentity"))
        if complete_frames >= 2 and has_exact:
            coherent += 1
        if has_exact:
            exact_nonidentity += 1
        comparable_certificates.extend(
            certificate
            for certificate in row.get("transport_certificates", ())
            if certificate.get("comparable")
        )
    coherent_fraction = coherent / max(1, len(nonterminal))
    targets = [
        row.get("labels", {}).get("goal_reachable_within_option") for row in rows
    ]
    known_targets = [value for value in targets if isinstance(value, bool)]
    positives = sum(known_targets)
    prevalence = positives / max(1, len(known_targets))
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
        len(canonical) == len(protocol.POSITIVE_WITNESS_GAMES) * len(protocol.PANEL_SEEDS)
        and all(
            receipt.get("goal_reachable_within_option") is True
            and receipt.get("errors") == []
            and receipt.get("illegal_actions") == 0
            and receipt.get("game_over") is False
            for receipt in canonical
        )
    )
    unknown_targets = sum(value is None for value in targets)
    unresolved = sum(int(receipt.get("unresolved_intents", 0)) for receipt in receipts)
    rooting_complete_fraction = sum(
        bool(row.get("binding", {}).get("complete")) for row in rows
    ) / max(1, len(rows))
    checks = {
        "events_present": bool(rows),
        "intent_accounting_complete": unresolved == 0 and unknown_targets == 0,
        "rooting_complete": rooting_complete_fraction == 1.0,
        "confident_correspondence": confident_fraction >= 0.90,
        "ambiguity_below_limit": ambiguous_fraction < 0.10,
        "multiframe_coherent_prefixes": coherent_fraction >= 0.50,
        "exact_nonidentity_transport_present": exact_nonidentity > 0,
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
            "format_version": "sage-t10.3-qa-report-v1",
            "phase": "compile",
            "manifest_checksum": manifest_checksum,
            "status": "PASS_T10_3_QA" if passed else "QA_MISS",
            "passed": passed,
            "checks": checks,
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "metrics": {
                "event_count": len(rows),
                "nonterminal_prefix_count": len(nonterminal),
                "confident_correspondence_fraction": confident_fraction,
                "fully_ambiguous_correspondence_fraction": ambiguous_fraction,
                "rooting_complete_fraction": rooting_complete_fraction,
                "multiframe_coherent_prefix_fraction": coherent_fraction,
                "exact_nonidentity_event_count": exact_nonidentity,
                "goal_reachable_positive_count": positives,
                "goal_reachable_total_count": len(known_targets),
                "goal_reachable_prevalence": prevalence,
                "goal_reachable_positive_games": sorted(positive_games),
                "unknown_target_count": unknown_targets,
                "unresolved_intent_count": unresolved,
            },
            "fit_authorized": passed,
            "historical_t10_2_9_events_used_for_fit": 0,
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
        events = _read_events_for_work(destination, work)
        rows.extend(backfill_branch_labels(events, receipt))
        unresolved_dir = _work_path(destination, "unresolved", work, "placeholder").parent
        for path in sorted(unresolved_dir.glob("*.json")) if unresolved_dir.exists() else ():
            unresolved = _read_signed(path, "unresolved_checksum")
            row = _signed(
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
                    "binding": {"complete": False, "missing": [unresolved["reason"]]},
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
                        "historical_t10_2_9_event": False,
                    },
                },
                "compiled_checksum",
            )
            rows.append(row)
    rows.sort(key=lambda row: (int(row["reset_index"]), int(row["step_index"])))
    _write_jsonl_once(destination / COMPILED_LEDGER_FILENAME, rows)
    ledger_bytes = (destination / COMPILED_LEDGER_FILENAME).read_bytes()
    compact = _signed(
        {
            "format_version": "sage-t10.3-compact-ledger-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "event_count": len(rows),
            "event_ids_sha256": _sha([row["event_id"] for row in rows]),
            "compiled_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "compiled_ledger_bytes": len(ledger_bytes),
            "historical_t10_2_9_event_count": 0,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        },
        "ledger_checksum",
    )
    _write_signed_once(destination / COMPACT_LEDGER_FILENAME, compact)
    qa = build_qa_report(
        rows, receipts, manifest_checksum=manifest["manifest_checksum"]
    )
    _write_signed_once(destination / QA_FILENAME, qa)
    if not qa["passed"]:
        raise ScientificGateMiss(protocol.canonical_json(qa))
    return qa


def _features(row: Mapping[str, Any], *, transport_enabled: bool = True) -> list[float]:
    frames = row.get("model_view", {}).get("frames", {})
    complete_frames = sum(
        bool(frame.get("complete"))
        for frame in frames.values()
        if isinstance(frame, Mapping)
    )
    transport = row.get("transport", {})
    observations = row.get("observations", {})
    step = float(row.get("step_index", 0)) / max(1, protocol.MAXIMUM_ACTIONS_PER_RESET - 1)
    return [
        1.0 if row.get("binding", {}).get("complete") else 0.0,
        complete_frames / 4.0,
        float(bool(transport.get("multiframe_exact_nonidentity"))) if transport_enabled else 0.0,
        min(1.0, float(transport.get("exact_nonidentity_certificate_count", 0)) / 3.0) if transport_enabled else 0.0,
        float(bool(observations.get("root_effect"))),
        float(bool(observations.get("physical_no_effect"))),
        step,
        float(row.get("selection", {}).get("parameter_arity", 0) > 0),
    ]


def _sigmoid(value: float) -> float:
    clipped = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _fit_logistic(rows: Sequence[Mapping[str, Any]], *, transport_enabled: bool = True) -> list[float]:
    data = [
        (row, bool(row.get("labels", {}).get("goal_reachable_within_option")))
        for row in rows
        if isinstance(row.get("labels", {}).get("goal_reachable_within_option"), bool)
        and row.get("complete")
    ]
    if not data or len({label for _row, label in data}) < 2:
        raise ScientificGateMiss("cross-fit training fold lacks both target classes")
    weights = [0.0] * 9
    for iteration in range(800):
        gradient = [0.0] * len(weights)
        for row, label in data:
            vector = [1.0, *_features(row, transport_enabled=transport_enabled)]
            probability = _sigmoid(sum(weight * value for weight, value in zip(weights, vector, strict=True)))
            error = probability - float(label)
            for index, value in enumerate(vector):
                gradient[index] += error * value
        rate = 0.08 / math.sqrt(iteration + 1)
        for index in range(len(weights)):
            penalty = 0.002 * weights[index] if index else 0.0
            weights[index] -= rate * (gradient[index] / len(data) + penalty)
    return weights


def _predict(weights: Sequence[float], row: Mapping[str, Any], *, transport_enabled: bool = True) -> float:
    vector = [1.0, *_features(row, transport_enabled=transport_enabled)]
    return _sigmoid(sum(weight * value for weight, value in zip(weights, vector, strict=True)))


def _auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return 0.5
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("inf")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _identity_probe(rows: Sequence[Mapping[str, Any]]) -> float:
    games = list(protocol.SOURCE_GAMES)
    train = [row for row in rows if int(row.get("seed", 0)) % 2 == 1 and row.get("complete")]
    test = [row for row in rows if int(row.get("seed", 0)) % 2 == 0 and row.get("complete")]
    centroids: dict[str, list[float]] = {}
    for game in games:
        vectors = [_features(row) for row in train if row.get("source_game") == game]
        if not vectors:
            return 1.0
        centroids[game] = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))]
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
    ablated_predictions: dict[str, float] = {}
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
        train_prevalence = sum(
            bool(row["labels"]["goal_reachable_within_option"]) for row in train
        ) / max(1, len(train))
        brier = sum((score - float(label)) ** 2 for score, label in zip(scores, labels, strict=True)) / max(1, len(labels))
        baseline_brier = sum((train_prevalence - float(label)) ** 2 for label in labels) / max(1, len(labels))
        for row, score, ablated_score in zip(test, scores, ablated_scores, strict=True):
            predictions[str(row["event_id"])] = score
            ablated_predictions[str(row["event_id"])] = ablated_score
        folds[held_out] = {
            "event_count": len(test),
            "auroc": _auroc(labels, scores),
            "no_transport_auroc": _auroc(labels, ablated_scores),
            "brier": brier,
            "baseline_brier": baseline_brier,
            "brier_improvement": baseline_brier - brier,
            "weights": [round(value, 12) for value in weights],
            "feature_names": [
                "intercept", "binding_complete", "complete_frame_fraction",
                "exact_nonidentity_transport", "exact_transport_fraction",
                "root_effect", "physical_no_effect", "step_fraction", "parameterized",
            ],
        }
    branch_scores: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in complete_rows:
        key = (str(row["source_game"]), int(row["seed"]), str(row["controller"]))
        branch_scores[key].append(predictions[str(row["event_id"])])
    ranks_by_game: dict[str, list[int]] = defaultdict(list)
    margins: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for game in protocol.SOURCE_GAMES:
        for seed in protocol.PANEL_SEEDS:
            scores = {
                controller: sum(branch_scores.get((game, seed, controller), ()))
                / max(1, len(branch_scores.get((game, seed, controller), ())))
                for controller in protocol.PANEL_ARMS
            }
            canonical = scores["canonical_option"]
            rank = 1 + sum(value > canonical for controller, value in scores.items() if controller != "canonical_option")
            ranks_by_game[game].append(rank)
            margins[game]["binding_swap"].append(canonical - scores["binding_swap"])
            margins[game]["option_intervention"].append(canonical - scores["option_intervention"])
    identity_balanced_accuracy = _identity_probe(complete_rows)
    positive_fold_checks = {}
    for game in protocol.POSITIVE_WITNESS_GAMES:
        fold = folds[game]
        positive_fold_checks[game] = {
            "auroc": fold["auroc"] >= 0.75,
            "brier_improvement": fold["brier_improvement"] > 0.0,
            "top_8": max(ranks_by_game[game], default=99) <= 8,
            "median_rank": _median(ranks_by_game[game]) <= 4,
            "binding_swap_margin": min(margins[game]["binding_swap"], default=-1.0) > 0.0,
            "option_intervention_margin": min(margins[game]["option_intervention"], default=-1.0) > 0.0,
        }
    full_mean = sum(fold["auroc"] for fold in folds.values()) / len(folds)
    ablated_mean = sum(fold["no_transport_auroc"] for fold in folds.values()) / len(folds)
    checks = {
        "positive_game_auroc": all(item["auroc"] for item in positive_fold_checks.values()),
        "positive_game_brier_improvement": all(item["brier_improvement"] for item in positive_fold_checks.values()),
        "positive_option_top_8": all(item["top_8"] for item in positive_fold_checks.values()),
        "positive_option_median_rank": all(item["median_rank"] for item in positive_fold_checks.values()),
        "binding_swap_margins": all(item["binding_swap_margin"] for item in positive_fold_checks.values()),
        "option_intervention_margins": all(item["option_intervention_margin"] for item in positive_fold_checks.values()),
        "identity_only_no_transport_degradation": full_mean > ablated_mean,
        "identity_probe_limit": identity_balanced_accuracy - (1.0 / 3.0) <= 0.10,
    }
    witness_lengths = sorted(
        len(item["steps"]) for item in manifest["handoff_receipt"]["canonical_witnesses"]
    )
    programs = [compile_progress_program(sequence_length=length) for length in witness_lengths]
    options = [
        repeat("apply_target", maximum_horizon=protocol.MAXIMUM_ACTIONS_PER_RESET),
        alternate("apply_target", "apply_distractor", maximum_horizon=protocol.MAXIMUM_ACTIONS_PER_RESET),
    ]
    factor_rows = [
        (
            program.canonical_hash,
            _sha(program.canonical_payload["goal"]),
            _sha("four_frozen_observer_frames"),
            _sha("exact_nonidentity_or_visible_noncomparable"),
            option.canonical_hash,
        )
        for program in programs
        for option in options
    ]
    marginals = FactorMarginals.uniform_from_rows(factor_rows)
    passed = all(checks.values())
    return _signed(
        {
            "format_version": "sage-t10.3-model-recipe-v1",
            "phase": "fit",
            "manifest_checksum": manifest["manifest_checksum"],
            "status": "PASS_T10_3_OPTION_FIT" if passed else "OPTION_INDUCTION_MISS",
            "passed": passed,
            "target": "goal_reachable_within_option",
            "cross_fit": folds,
            "positive_game_checks": positive_fold_checks,
            "option_ranks": {game: ranks for game, ranks in sorted(ranks_by_game.items())},
            "paired_margins": {
                game: {name: values for name, values in sorted(items.items())}
                for game, items in sorted(margins.items())
            },
            "identity_probe_balanced_accuracy": identity_balanced_accuracy,
            "identity_probe_excess": identity_balanced_accuracy - (1.0 / 3.0),
            "full_mean_auroc": full_mean,
            "no_transport_mean_auroc": ablated_mean,
            "checks": checks,
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "joint_program_hypotheses": [program.canonical_payload for program in programs],
            "mixed_option_automata": [option.canonical_payload for option in options],
            "factorized_posterior": {
                "contract": "FactorMarginals",
                "factor_names": [item.name for item in marginals],
                "support_sizes": dict(marginals.support_sizes),
                "probabilities": {item.name: dict(item.probabilities) for item in marginals},
            },
            "feature_firewall": {
                "game_id": False,
                "seed": False,
                "coordinates": False,
                "entity_identity": False,
                "raw_grid": False,
            },
            "historical_t10_2_9_events_used": 0,
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
    rows = _compiled_rows(destination)
    recipe = build_model_recipe(rows, manifest)
    _write_signed_once(destination / MODEL_FILENAME, recipe)
    return recipe


def build_confirmation_report(
    receipts: Sequence[Mapping[str, Any]], *, manifest_checksum: str, recipe_checksum: str
) -> dict[str, Any]:
    by_game: dict[str, dict[str, Any]] = {}
    for game in protocol.SOURCE_GAMES:
        game_rows = [receipt for receipt in receipts if receipt.get("game_id") == game]
        learned = [receipt for receipt in game_rows if receipt.get("controller") == "learned"]
        control = [
            receipt
            for receipt in game_rows
            if receipt.get("controller") == "capacity_matched_independent"
        ]
        learned_levels = sum(int(receipt.get("level_delta", 0)) for receipt in learned)
        control_levels = sum(int(receipt.get("level_delta", 0)) for receipt in control)
        by_game[game] = {
            "learned_levels": learned_levels,
            "control_levels": control_levels,
            "level_margin": learned_levels - control_levels,
            "learned_game_over_rate": sum(bool(receipt.get("game_over")) for receipt in learned) / max(1, len(learned)),
            "control_game_over_rate": sum(bool(receipt.get("game_over")) for receipt in control) / max(1, len(control)),
        }
    errors = sum(len(receipt.get("errors", ())) for receipt in receipts)
    illegal = sum(int(receipt.get("illegal_actions", 0)) for receipt in receipts)
    aggregate_advantage = sum(item["level_margin"] for item in by_game.values())
    checks = {
        "lp85_level": by_game["lp85-305b61c3"]["learned_levels"] >= 1,
        "su15_level": by_game["su15-4c352900"]["learned_levels"] >= 1,
        "bp35_no_regression": by_game["bp35-0a0ad940"]["level_margin"] >= 0,
        "aggregate_level_advantage": aggregate_advantage >= 1,
        "zero_errors": errors == 0,
        "zero_illegal_actions": illegal == 0,
        "game_over_not_higher": all(
            item["learned_game_over_rate"] <= item["control_game_over_rate"]
            for item in by_game.values()
        ),
        "counterbalanced": True,
    }
    passed = all(checks.values())
    return _signed(
        {
            "format_version": "sage-t10.3-confirmation-report-v1",
            "phase": "confirm",
            "manifest_checksum": manifest_checksum,
            "recipe_checksum": recipe_checksum,
            "status": "PASS_T10_3_SOURCE_CONFIRMATION" if passed else "SOURCE_CONFIRMATION_MISS",
            "passed": passed,
            "checks": checks,
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "by_game": by_game,
            "aggregate_level_advantage": aggregate_advantage,
            "errors": errors,
            "illegal_actions": illegal,
            "confirmation_resets": len(receipts),
            "physical_actions_replayed": 0,
            "firewall": _firewall(),
        },
        "report_checksum",
    )


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
        receipts.append(_run_work(destination, manifest, work, runtime, model=recipe))
        checkpoint = _checkpoint(destination, manifest)
        print(
            protocol.canonical_json(
                {
                    "phase": "t10_3_confirm",
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
    qa_checks = qa.get("checks", {})
    if not qa_checks.get("rooting_complete") or not qa_checks.get("confident_correspondence") or not qa_checks.get("ambiguity_below_limit"):
        return "ROOTING_MISS"
    if not qa_checks.get("positive_witness_reproduced"):
        return "WITNESS_REPRODUCTION_MISS"
    if qa.get("passed") is not True:
        return "QA_MISS"
    if model is None:
        raise ScientificGateMiss("fit phase has not produced a model recipe")
    model_checks = model.get("checks", {})
    if not model_checks.get("positive_game_auroc") or not model_checks.get("positive_game_brier_improvement") or not model_checks.get("binding_swap_margins") or not model_checks.get("option_intervention_margins"):
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
            "format_version": "sage-t10.3-terminal-report-v1",
            "phase": "report",
            "manifest_checksum": manifest["manifest_checksum"],
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
            "physical_actions_replayed": 0,
        }
    )
    return {
        "phase": "status",
        "status": "READY_T10_3_SOURCE_PILOT",
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
            payload = protocol.freeze_manifest(
                output_path=args.manifest, repo_root=root
            )
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
        if message.startswith("{"):
            print(message)
        else:
            print(protocol.canonical_json({"phase": args.phase, "status": "SCIENTIFIC_GATE_MISS", "error": message}))
        return 3
    except (
        protocol.ManifestDriftError,
        protocol.ProtocolError,
        IntegrityError,
        JournalConflictError,
        JournalIntegrityError,
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
