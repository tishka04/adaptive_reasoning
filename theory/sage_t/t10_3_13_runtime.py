"""Dormant, durable runtime for SAGE T10.3.13 prospective confirmation.

Import and ``status`` are protected-data cold: neither imports an environment
module from ``environment_files`` nor constructs a protected environment.  The
first permitted reset is inside ``active-confirmation`` after an explicit
authorization receipt, a frozen manifest, and both offline gates.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from theory.live_transition_loop import build_observation, build_transition_record

from . import t10_3_2_runtime as durable
from . import t10_3_13_protocol as protocol
from .compiler import compile_observation, compile_transition_record

PARENT_AUDIT_FILENAME = "parent_candidate_audit.json"
PREFLIGHT_FILENAME = "prospective_preflight.json"
ACTIVE_REPORT_FILENAME = "active_confirmation_report.json"
ADJUDICATION_FILENAME = "prospective_adjudication.json"
TERMINAL_REPORT_FILENAME = "terminal_report.json"
LOCK_FILENAME = durable.LOCK_FILENAME


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = protocol.sha256_payload(output)
    return output


def _destination(root: Path) -> Path:
    return root.resolve() / protocol.DEFAULT_OUTPUT_DIR


def _path(root: Path, filename: str) -> Path:
    return _destination(root) / filename


def _write(root: Path, filename: str, payload: Mapping[str, Any]) -> None:
    protocol.write_json_once(_path(root, filename), payload)


def _read_path_signed(path: Path, checksum_field: str) -> dict[str, Any]:
    if not path.is_file():
        raise protocol.IntegrityError(f"required artifact is absent: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol.verify_signed(payload, checksum_field)
    return payload


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    return _read_path_signed(_path(root, filename), checksum_field)


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


def _accounting(root: Path) -> dict[str, Any]:
    with _durable_contract():
        return durable._journal_accounting(_destination(root))


def _require_gate(root: Path, phase: str) -> dict[str, Any]:
    contract = protocol.ARTIFACT_CONTRACT[phase]
    payload = _read_signed(
        root,
        str(contract["path"]),
        str(contract["checksum_field"]),
    )
    gate = contract.get("gate_field")
    if gate is not None and payload.get(str(gate)) is not True:
        raise protocol.ScientificGateMiss(f"{phase} gate forbids continuation")
    return payload


def _journal_paths(destination: Path, category: str, pattern: str = "*.json") -> list[Path]:
    base = destination / "journal" / category
    return sorted(base.rglob(pattern)) if base.exists() else []


def _opening_paths(destination: Path) -> list[Path]:
    return _journal_paths(destination, "openings", "opening.json")


def _receipt_paths(destination: Path) -> list[Path]:
    return _journal_paths(destination, "branches", "receipt.json")


def _load_openings(destination: Path) -> list[dict[str, Any]]:
    return [_read_path_signed(path, "opening_checksum") for path in _opening_paths(destination)]


def _load_receipts(destination: Path) -> list[dict[str, Any]]:
    return [_read_path_signed(path, "receipt_checksum") for path in _receipt_paths(destination)]


def _safe_optional_receipt(path: Path, checksum_field: str) -> dict[str, Any] | None:
    return _read_path_signed(path, checksum_field) if path.is_file() else None


def status(root: Path) -> dict[str, Any]:
    """Report dormant or active state without loading a manifest or a game."""

    root = root.resolve()
    destination = _destination(root)
    accounting = _accounting(root)
    manifest = _safe_optional_receipt(
        root / protocol.DEFAULT_MANIFEST_PATH, "manifest_checksum"
    )
    authorization = _safe_optional_receipt(
        root / protocol.DEFAULT_AUTHORIZATION_PATH, "authorization_checksum"
    )
    freeze_receipt = _safe_optional_receipt(
        root / protocol.DEFAULT_FREEZE_RECEIPT_PATH, "receipt_checksum"
    )
    openings = _load_openings(destination)
    receipts = _load_receipts(destination)
    receipt_work_ids = {str(row.get("work_id", "")) for row in receipts}
    artifacts: dict[str, str | None] = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        path = _path(root, str(contract["path"]))
        if not path.is_file():
            artifacts[phase] = None
            continue
        payload = _read_path_signed(path, str(contract["checksum_field"]))
        artifacts[phase] = str(payload[contract["checksum_field"]])
    protected_trace_absent = bool(
        not openings
        and not receipts
        and int(accounting.get("authorized_actions", 0)) == 0
        and int(accounting.get("sealed_events", 0)) == 0
        and int(accounting.get("unresolved_intents", 0)) == 0
        and int(accounting.get("inflight_intents", 0)) == 0
        and not accounting.get("live_collector_lock")
    )
    return {
        "format_version": "sage-t10.3.13-status-v1",
        "manifest_frozen": manifest is not None,
        "manifest_checksum": None if manifest is None else manifest["manifest_checksum"],
        "authorization_present": authorization is not None,
        "authorization_checksum": (
            None if authorization is None else authorization["authorization_checksum"]
        ),
        "freeze_receipt_present": freeze_receipt is not None,
        "artifacts": artifacts,
        "accounting": accounting,
        "protected_opening_receipts": len(openings),
        "completed_work_receipts": len(receipts),
        "holdout_opened": bool(
            openings
            or receipts
            or int(accounting.get("authorized_actions", 0))
            or int(accounting.get("sealed_events", 0))
        ),
        "holdout_not_opened_proven": protected_trace_absent,
        "protected_frames_read": sum(
            int(row.get("protected_frame_snapshots", 0)) for row in receipts
        )
        + sum(
            int(row.get("protected_frame_snapshots", 1))
            for row in openings
            if str(row.get("work_id", "")) not in receipt_work_ids
        ),
        "physical_actions": int(accounting.get("sealed_events", 0)),
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        "maximum_resets": protocol.TOTAL_RESETS,
        "ar25_opened": False,
        "source_validation_opened": False,
        "sequence_games_opened": False,
        "production_authority": False,
    }


def audit_parent(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    state = protocol.verify_parent_candidate(root)
    authorization = protocol.load_authorization(root)
    expected_pair = protocol.candidate_pair(str(state["verdict"]))
    accounting = _accounting(root)
    cold = bool(
        not _opening_paths(_destination(root))
        and not _receipt_paths(_destination(root))
        and int(accounting.get("authorized_actions", 0)) == 0
        and int(accounting.get("sealed_events", 0)) == 0
        and int(accounting.get("unresolved_intents", 0)) == 0
        and int(accounting.get("inflight_intents", 0)) == 0
        and not accounting.get("live_collector_lock")
    )
    checks = {
        "parent_candidate_exact": state == manifest.get("parent_state"),
        "parent_pass_verdict": state.get("verdict")
        in {protocol.SOURCE_PASS, protocol.GENERIC_PASS},
        "candidate_pair_frozen": expected_pair
        == (state.get("candidate_arm"), state.get("control_arm")),
        "authorization_bound": authorization.get("authorization_checksum")
        == manifest.get("authorization_checksum"),
        "authorization_zero_frames": authorization.get(
            "protected_frames_read_at_authorization"
        )
        == 0,
        "authorization_zero_actions": authorization.get(
            "physical_actions_at_authorization"
        )
        == 0,
        "zero_protected_openings": not _opening_paths(_destination(root)),
        "zero_protected_receipts": not _receipt_paths(_destination(root)),
        "zero_physical_actions": int(accounting.get("authorized_actions", 0)) == 0,
        "holdout_not_opened": cold,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.13-parent-candidate-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_state": state,
            "authorization_checksum": authorization["authorization_checksum"],
            "checks": checks,
            "passed": all(checks.values()),
            "protected_games_instantiated": 0,
            "protected_frames_read": 0,
            "physical_actions": 0,
            "holdout_opened": False,
            "production_authority": False,
        },
        "audit_checksum",
    )
    _write(root, PARENT_AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("T10_3_13_PARENT_CANDIDATE_AUDIT_MISS")
    return payload


def _causal_api() -> tuple[type[Any], type[Any], Any]:
    """Import the frozen 12f causal code only after an offline/active phase starts."""

    from .causal_procedure_v10_3_12f import (  # local by design
        CausalOutcome,
        CausalProcedureController,
        CausalProcedurePrior,
        preflight_prior,
    )

    return CausalProcedurePrior, CausalProcedureController, (CausalOutcome, preflight_prior)


def _read_parent_prior(root: Path, manifest: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    path = root.resolve() / protocol.PARENT_PRIOR_PATH
    payload = _read_path_signed(path, "prior_checksum")
    if payload.get("prior_checksum") != manifest["parent_state"]["prior_checksum"]:
        raise protocol.IntegrityError("prospective prior is detached from T10.3.12f")
    prior_type, _, _ = _causal_api()
    return prior_type(payload), payload


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-parent")
    prior, prior_payload = _read_parent_prior(root, manifest)
    _, controller_type, helpers = _causal_api()
    _, preflight_prior = helpers
    prior_checks = preflight_prior(prior)
    candidate = controller_type(
        str(manifest["matrix"]["candidate_arm"]), scope=0, prior=prior
    )
    control = controller_type(
        str(manifest["matrix"]["control_arm"]), scope=0, prior=prior
    )
    specs = protocol.work_specs(
        "active-confirmation",
        candidate=str(manifest["matrix"]["candidate_arm"]),
        control=str(manifest["matrix"]["control_arm"]),
    )
    candidate_summary = candidate.summary()
    control_summary = control.summary()
    accounting = _accounting(root)
    cold = bool(
        not _opening_paths(_destination(root))
        and not _receipt_paths(_destination(root))
        and int(accounting.get("authorized_actions", 0)) == 0
        and int(accounting.get("sealed_events", 0)) == 0
        and int(accounting.get("unresolved_intents", 0)) == 0
        and int(accounting.get("inflight_intents", 0)) == 0
        and not accounting.get("live_collector_lock")
    )
    checks = {
        "parent_audit_passed": True,
        "prior_checksum_exact": prior.prior_checksum == prior_payload["prior_checksum"],
        "prior_preflight_passed": prior_checks.get("passed") is True,
        "candidate_arm_exact": candidate_summary.get("arm")
        == manifest["matrix"]["candidate_arm"],
        "control_arm_exact": control_summary.get("arm")
        == manifest["matrix"]["control_arm"],
        "fresh_zero_action_controllers": candidate_summary.get("actions") == 0
        and control_summary.get("actions") == 0,
        "ten_unique_work_specs": len(specs) == protocol.TOTAL_RESETS
        and len({work.work_id for work in specs}) == protocol.TOTAL_RESETS,
        "action_budget_exact": protocol.maximum_actions_for_specs(specs)
        == protocol.TOTAL_MAXIMUM_ACTIONS,
        "zero_protected_openings": not _opening_paths(_destination(root)),
        "zero_protected_receipts": not _receipt_paths(_destination(root)),
        "zero_physical_actions": int(accounting.get("authorized_actions", 0)) == 0,
        "holdout_not_opened": cold,
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.13-prospective-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "prior_checksum": prior.prior_checksum,
            "candidate_program_hash": candidate_summary.get("program_hash"),
            "control_program_hash": control_summary.get("program_hash"),
            "checks": checks,
            "passed": all(checks.values()),
            "protected_games_instantiated": 0,
            "protected_frames_read": 0,
            "physical_actions": 0,
            "holdout_opened": False,
            "production_authority": False,
        },
        "preflight_checksum",
    )
    _write(root, PREFLIGHT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("T10_3_13_PREFLIGHT_MISS")
    return payload


def _work_path(
    destination: Path,
    category: str,
    work: protocol.WorkSpec,
    name: str,
) -> Path:
    return destination / "journal" / category / work.work_id / name


def _receipt_path(destination: Path, work: protocol.WorkSpec) -> Path:
    return _work_path(destination, "branches", work, "receipt.json")


def _opening_path(destination: Path, work: protocol.WorkSpec) -> Path:
    return _work_path(destination, "openings", work, "opening.json")


def _make_environment(root: Path, game_id: str) -> Any:
    return durable.live._make_real_env(game_id, root / "environment_files")


def _reset_environment(environment: Any) -> Any:
    return durable.live._reset_env(environment)


def _snapshot(frame: Any, *, fallback_available_actions: Sequence[Any] = ()) -> Any:
    return durable.live.snapshot_frame(
        frame,
        fallback_available_actions=fallback_available_actions,
    )


def _legal_actions(environment: Any) -> tuple[Any, ...]:
    return tuple(durable.live._valid_actions(environment))


def _materialize(legal: Sequence[Any], decision: Any) -> Any | None:
    return durable.live._materialize_decision(legal, decision)


def _step_environment(environment: Any, selected: Any) -> Any:
    return durable.live._step_env_action(environment, selected)


def _close_environment(environment: Any) -> None:
    close = getattr(environment, "close", None)
    if callable(close):
        close()


def _frame_hash(snapshot: Any) -> str:
    grid = snapshot.grid
    payload = grid.tolist() if hasattr(grid, "tolist") else grid
    return protocol.sha256_payload(payload)


def _compile_state(snapshot: Any, legal: Sequence[Any]) -> Any:
    observation = build_observation(
        snapshot.grid,
        available_actions=durable.live._available_action_names(legal),
        game_state=str(snapshot.game_state),
        levels_completed=int(snapshot.levels_completed),
        infer_players=True,
    )
    return compile_observation(observation)


def _compile_transition(
    *,
    before: Any,
    after: Any,
    selected: Any,
    legal_before: Sequence[Any],
    legal_after: Sequence[Any],
) -> tuple[Any, Any]:
    record = build_transition_record(
        action=str(getattr(selected, "name", "")),
        grid_before=before.grid,
        grid_after=after.grid,
        available_actions=durable.live._available_action_names(legal_before),
        game_state_before=str(before.game_state),
        game_state_after=str(after.game_state),
        levels_completed_before=int(before.levels_completed),
        levels_completed_after=int(after.levels_completed),
        action_args=dict(getattr(selected, "action_args", {}) or {}),
        infer_players=True,
    )
    observed = compile_transition_record(record)
    before_names = set(durable.live._available_action_names(legal_before))
    after_names = set(durable.live._available_action_names(legal_after))
    action_space_changed = before_names != after_names

    # Correspondence is recomputed from this target-local transition.  Births,
    # deaths, ambiguous matches, and non one-to-one matches cannot support a
    # persistent-object or relational causal delta.
    from theory.sage12.mt.transition import compile_mt_transition  # local by design
    from theory.sage12.topological_invariants_v4_19 import (  # local by design
        correspondence_quality,
    )

    action_name = str(getattr(selected, "name", ""))
    action_data = dict(getattr(selected, "action_args", {}) or {})
    mt = compile_mt_transition(
        before.grid,
        action_name,
        after.grid,
        action_data=action_data,
        productive=not record.diff.is_noop,
        risk=bool(record.diff.game_over),
    )
    quality = correspondence_quality(mt)
    persistent = [
        row
        for row in mt.correspondences
        if row.kind == "persist"
        and len(row.before_ids) == 1
        and len(row.after_ids) == 1
        and float(row.confidence) >= 0.60
    ]
    before_ids = [row.before_ids[0] for row in persistent]
    after_ids = [row.after_ids[0] for row in persistent]
    one_to_one = bool(persistent) and len(before_ids) == len(set(before_ids)) and len(
        after_ids
    ) == len(set(after_ids))
    confidence = float(quality.get("mean_confidence", 0.0))
    trusted = bool(
        one_to_one
        and not quality.get("fully_ambiguous")
        and confidence >= 0.60
    )
    level_delta = max(
        0,
        int(after.levels_completed) - int(before.levels_completed),
    )
    from .causal_procedure_v10_3_12f import (  # local by design
        causal_outcome_from_mt_transition,
    )

    if trusted:
        try:
            outcome = causal_outcome_from_mt_transition(
                mt,
                action_space_changed=action_space_changed,
                level_delta=level_delta,
                game_over=bool(record.diff.game_over),
            )
        except ValueError as exc:
            if "simultaneously added and removed" not in str(exc):
                raise
            trusted = False
    if not trusted:
        causal_outcome, _ = _causal_api()[2]
        outcome = causal_outcome(
            action_space_changed=action_space_changed,
            noop=bool(record.diff.is_noop),
            game_over=bool(record.diff.game_over),
            level_delta=level_delta,
            quality=min(confidence, 0.599999),
        )
    return observed, outcome


def _intent(
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    *,
    step_index: int,
    decision: Any,
    prior_checksum: str,
) -> dict[str, Any]:
    event_id = protocol.sha256_payload(
        {"manifest": manifest["manifest_checksum"], "work": work.work_id, "step": step_index}
    )
    return _signed(
        {
            "format_version": "sage-t10.3.13-action-intent-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "authorization_checksum": manifest["authorization_checksum"],
            "work_id": work.work_id,
            "work": work.as_dict(),
            "event_id": event_id,
            "step_index": step_index,
            "procedure_decision": decision.safe_payload,
            "prior_checksum": prior_checksum,
            "raw_action_retained": False,
            "grounded_arguments_retained": False,
            "physical_action_replayed": False,
        },
        "intent_checksum",
    )


def _validate_receipt_binding(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    prior_checksum: str,
) -> None:
    if receipt.get("complete") is not True:
        raise protocol.IntegrityError("protected receipt is not complete")
    if receipt.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise protocol.IntegrityError("protected receipt is detached from manifest")
    if receipt.get("authorization_checksum") != manifest["authorization_checksum"]:
        raise protocol.IntegrityError("protected receipt is detached from authorization")
    if receipt.get("prior_checksum_loaded") != prior_checksum:
        raise protocol.IntegrityError("protected receipt is detached from prior")
    if receipt.get("work_id") != work.work_id or any(
        receipt.get(key) != value for key, value in work.as_dict().items()
    ):
        raise protocol.IntegrityError("protected receipt work specification drifted")


def _run_work(
    root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    prior: Any,
    prior_checksum: str,
    lock: Any,
) -> dict[str, Any]:
    receipt_path = _receipt_path(destination, work)
    if receipt_path.is_file():
        receipt = _read_path_signed(receipt_path, "receipt_checksum")
        _validate_receipt_binding(
            receipt,
            manifest=manifest,
            work=work,
            prior_checksum=prior_checksum,
        )
        return receipt
    if _opening_path(destination, work).exists():
        raise protocol.IntegrityError("opened protected reset cannot be replayed")

    _, controller_type, _ = _causal_api()
    controller = controller_type(work.arm, scope=0, prior=prior)
    event_ids: list[str] = []
    prequential_losses: list[float] = []
    errors: list[str] = []
    illegal_actions = 0
    game_over_actions = 0
    noop_actions = 0
    protected_frame_snapshots = 0
    initial_level = 0
    final_level = 0
    initial_frame_hash = ""
    first_level_action: int | None = None
    status_value = "COMPLETE"
    stop_reason = "ACTION_BUDGET_EXHAUSTED"
    environment: Any | None = None
    frame: Any | None = None
    current_intent: dict[str, Any] | None = None
    current_name = ""
    current_event_sealed = False
    started = time.perf_counter()
    try:
        environment = _make_environment(root, work.game_id)
        frame = _reset_environment(environment)
        initial = _snapshot(frame)
        protected_frame_snapshots += 1
        initial_level = final_level = int(initial.levels_completed)
        initial_frame_hash = _frame_hash(initial)
        opening = _signed(
            {
                "format_version": "sage-t10.3.13-protected-opening-v1",
                "manifest_checksum": manifest["manifest_checksum"],
                "authorization_checksum": manifest["authorization_checksum"],
                **work.as_dict(),
                "work_id": work.work_id,
                "initial_frame_sha256": initial_frame_hash,
                "protected_frame_snapshots": 1,
                "raw_frame_retained": False,
                "physical_actions": 0,
                "physical_action_replayed": False,
            },
            "opening_checksum",
        )
        protocol.write_json_once(_opening_path(destination, work), opening)
        lock.heartbeat()

        for step_index in range(work.action_budget):
            if time.perf_counter() - started >= protocol.RESET_WALL_SECONDS:
                stop_reason = "RESET_WALL_BUDGET_EXHAUSTED"
                break
            before = _snapshot(frame)
            protected_frame_snapshots += 1
            if durable.live._is_terminal(before.game_state):
                stop_reason = "TERMINAL_STATE"
                break
            legal_before = _legal_actions(environment)
            state_before = _compile_state(before, legal_before)
            decision = controller.propose(
                state_before,
                legal_before,
                shape=tuple(int(value) for value in before.grid.shape[:2]),
                step_index=step_index,
            )
            if decision.abstained:
                stop_reason = "PLANNED_CAUSAL_ABSTENTION"
                break
            selected = _materialize(legal_before, decision.candidate)
            if selected is None:
                illegal_actions += 1
                errors.append("UNAVAILABLE_CAUSAL_DECISION")
                status_value = "ABORTED"
                stop_reason = "UNAVAILABLE_CAUSAL_DECISION"
                break
            intent = _intent(
                manifest,
                work,
                step_index=step_index,
                decision=decision,
                prior_checksum=prior_checksum,
            )
            name = f"{step_index:04d}.json"
            current_intent = intent
            current_name = name
            current_event_sealed = False
            protocol.write_json_once(_work_path(destination, "intents", work, name), intent)
            lock.heartbeat()
            try:
                after_frame = _step_environment(environment, selected)
                after = _snapshot(
                    after_frame,
                    fallback_available_actions=before.available_actions,
                )
                protected_frame_snapshots += 1
            except Exception as exc:  # noqa: BLE001
                unresolved = _signed(
                    {
                        "format_version": "sage-t10.3.13-unresolved-event-v1",
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
                status_value = "ABORTED"
                stop_reason = "ENVIRONMENT_CALL_UNATTESTABLE"
                break

            legal_after = _legal_actions(environment)
            observed, outcome = _compile_transition(
                before=before,
                after=after,
                selected=selected,
                legal_before=legal_before,
                legal_after=legal_after,
            )
            outcome_signature = getattr(outcome, "signature", None)
            if outcome_signature is None:
                outcome_signature = protocol.sha256_payload(outcome.safe_payload)[:20]
            observed_before = getattr(observed, "state_before", None)
            observed_after = getattr(observed, "state_after", None)
            event = _signed(
                {
                    "format_version": "sage-t10.3.13-physical-event-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "authorization_checksum": manifest["authorization_checksum"],
                    "work_id": work.work_id,
                    "event_id": intent["event_id"],
                    "step_index": step_index,
                    "procedure_prediction": decision.safe_payload,
                    "causal_outcome": outcome.safe_payload,
                    "outcome_signature": outcome_signature,
                    "abstract_state_before": getattr(
                        observed_before, "signature", None
                    ),
                    "abstract_state_after": getattr(
                        observed_after, "signature", None
                    ),
                    "levels_before": int(before.levels_completed),
                    "levels_after": int(after.levels_completed),
                    "level_delta": int(outcome.level_delta),
                    "game_state_after": str(after.game_state),
                    "frame_before_sha256": _frame_hash(before),
                    "frame_after_sha256": _frame_hash(after),
                    "raw_frame_retained": False,
                    "grounded_arguments_retained": False,
                    "physical_action_replayed": False,
                },
                "event_checksum",
            )
            # The physical event is sealed before the posterior may observe it.
            protocol.write_json_once(_work_path(destination, "events", work, name), event)
            current_event_sealed = True
            event_ids.append(str(intent["event_id"]))
            lock.heartbeat()
            update = controller.observe(observed, outcome=outcome)
            probability = max(1e-9, float(update.predicted_probability))
            prequential_losses.append(-math.log(probability))
            update_safe = dict(update.safe_payload)
            update_payload = _signed(
                {
                    "format_version": "sage-t10.3.13-procedure-update-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "work_id": work.work_id,
                    "event_id": intent["event_id"],
                    "step_index": step_index,
                    "phase_before": update_safe.get("phase_before"),
                    "phase_after": update_safe.get("phase_after"),
                    "predicted_family": update_safe.get("predicted_family"),
                    "predicted_probability": update_safe.get(
                        "predicted_probability"
                    ),
                    "outcome_signature": protocol.sha256_payload(
                        update_safe.get("outcome", {})
                    )[:20],
                    "posterior_digest": protocol.sha256_payload(
                        update_safe.get("posterior", {})
                    ),
                    "mismatch": update_safe.get("mismatch"),
                    "revised": update_safe.get("revised"),
                    "abstained": update_safe.get("abstained"),
                    "reason": update_safe.get("reason"),
                    "prequential_log_loss": prequential_losses[-1],
                    "grounded_payload_retained": False,
                },
                "update_checksum",
            )
            protocol.write_json_once(_work_path(destination, "updates", work, name), update_payload)
            frame = after_frame
            final_level = int(after.levels_completed)
            game_over_actions += int(outcome.game_over)
            noop_actions += int(outcome.noop)
            current_intent = None
            if time.perf_counter() - started > protocol.RESET_WALL_SECONDS:
                errors.append("RESET_WALL_BUDGET_EXCEEDED")
                status_value = "ABORTED"
                stop_reason = "RESET_WALL_BUDGET_EXCEEDED"
                break
            if int(outcome.level_delta) > 0:
                first_level_action = step_index + 1
                stop_reason = "LEVEL_PROGRESS_SEALED"
                break
            if durable.live._is_terminal(after.game_state):
                stop_reason = "TERMINAL_STATE"
                break
    except protocol.IntegrityError:
        raise
    except Exception as exc:  # noqa: BLE001
        errors.append(f"RUNTIME:{type(exc).__name__}")
        status_value = "ABORTED"
        stop_reason = "RUNTIME_ERROR"
        if current_intent is not None and current_name and not current_event_sealed:
            event_path = _work_path(destination, "events", work, current_name)
            unresolved_path = _work_path(
                destination, "unresolved", work, current_name
            )
            if not event_path.exists() and not unresolved_path.exists():
                unresolved_payload = _signed(
                    {
                        "format_version": "sage-t10.3.13-unresolved-event-v1",
                        "manifest_checksum": manifest["manifest_checksum"],
                        "work_id": work.work_id,
                        "event_id": current_intent["event_id"],
                        "step_index": current_intent["step_index"],
                        "reason": f"POST_ACTION_PIPELINE:{type(exc).__name__}",
                        "physical_action_replayed": False,
                    },
                    "unresolved_checksum",
                )
                protocol.write_json_once(unresolved_path, unresolved_payload)
    finally:
        if environment is not None:
            try:
                _close_environment(environment)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ENVIRONMENT_CLOSE:{type(exc).__name__}")

    intent_dir = _work_path(destination, "intents", work, "x").parent
    event_dir = _work_path(destination, "events", work, "x").parent
    update_dir = _work_path(destination, "updates", work, "x").parent
    unresolved_dir = _work_path(destination, "unresolved", work, "x").parent
    issued = len(list(intent_dir.glob("*.json"))) if intent_dir.exists() else 0
    sealed = len(list(event_dir.glob("*.json"))) if event_dir.exists() else 0
    observed_count = len(list(update_dir.glob("*.json"))) if update_dir.exists() else 0
    unresolved = len(list(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    summary = dict(controller.summary())
    belief = dict(summary.pop("belief", {}) or {})
    if belief:
        family_rows = dict(belief.get("families", {}) or {})
        summary.update(
            {
                "posterior_digest": protocol.sha256_payload(belief),
                "posterior_observations": int(belief.get("observations", 0)),
                "verified_context_diversity": max(
                    (
                        int(dict(row).get("distinct_contexts", 0))
                        for row in family_rows.values()
                        if isinstance(row, Mapping)
                    ),
                    default=0,
                ),
                "distinct_interventions": int(
                    belief.get("distinct_interventions", 0)
                ),
            }
        )
    complete = bool(
        status_value == "COMPLETE"
        and not errors
        and issued == sealed == observed_count
        and unresolved == 0
    )
    utility = (
        (work.action_budget + 1 - first_level_action) / work.action_budget
        if first_level_action is not None
        else 0.0
    )
    receipt = _signed(
        {
            "format_version": "sage-t10.3.13-branch-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "authorization_checksum": manifest["authorization_checksum"],
            **work.as_dict(),
            "work_id": work.work_id,
            "status": status_value,
            "complete": complete,
            "stop_reason": stop_reason,
            "planned_abstention": stop_reason == "PLANNED_CAUSAL_ABSTENTION",
            "issued_intents": issued,
            "sealed_events": sealed,
            "observed_updates": observed_count,
            "unresolved_intents": unresolved,
            "event_ids": event_ids,
            "initial_frame_sha256": initial_frame_hash,
            "level_delta": max(0, final_level - initial_level),
            "first_level_action": first_level_action,
            "utility": utility,
            "prequential_log_loss": (
                statistics.fmean(prequential_losses) if prequential_losses else None
            ),
            "procedure_summary": summary,
            "causal_proposals": issued,
            "causal_observations": observed_count,
            "illegal_actions": illegal_actions,
            "legacy_fallback_actions": 0,
            "game_over_actions": game_over_actions,
            "noop_actions": noop_actions,
            "errors": errors,
            "prior_checksum_loaded": prior_checksum,
            "protected_frame_snapshots": protected_frame_snapshots,
            "raw_frames_persisted": False,
            "grounded_arguments_persisted": False,
            "physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    protocol.write_json_once(receipt_path, receipt)
    return receipt


def active_confirmation(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-parent")
    _require_gate(root, "preflight")
    prior, prior_payload = _read_parent_prior(root, manifest)
    candidate = str(manifest["matrix"]["candidate_arm"])
    control = str(manifest["matrix"]["control_arm"])
    specs = protocol.work_specs(
        "active-confirmation", candidate=candidate, control=control
    )
    destination = _destination(root)
    expected_ids = {work.work_id for work in specs}
    with _durable_contract():
        durable._require_live_runtime()
        before = durable._journal_accounting(destination)
        if not before.get("equation_holds") or not before.get("inflight_valid"):
            raise protocol.IntegrityError(
                "protected journal accounting is invalid before confirmation"
            )
        if before.get("inflight_paths") or before.get("unresolved_intents"):
            raise protocol.IntegrityError("interrupted protected action cannot be replayed")
        if before.get("incomplete_work_ids"):
            raise protocol.IntegrityError("incomplete protected work cannot be replayed")
        existing_receipts = _load_receipts(destination)
        existing_openings = _load_openings(destination)
        receipt_ids = {str(row.get("work_id")) for row in existing_receipts}
        opening_ids = {path.parent.name for path in _opening_paths(destination)}
        if receipt_ids - expected_ids or opening_ids - expected_ids:
            raise protocol.IntegrityError("protected journal contains an unknown work id")
        specs_by_id = {work.work_id: work for work in specs}
        for row in (*existing_openings, *existing_receipts):
            if row.get("manifest_checksum") != manifest["manifest_checksum"]:
                raise protocol.IntegrityError("protected journal is detached from manifest")
            if row.get("authorization_checksum") != manifest["authorization_checksum"]:
                raise protocol.IntegrityError("protected journal is detached from authorization")
        for row in existing_receipts:
            work_id = str(row.get("work_id", ""))
            _validate_receipt_binding(
                row,
                manifest=manifest,
                work=specs_by_id[work_id],
                prior_checksum=str(prior_payload["prior_checksum"]),
            )
        if opening_ids - receipt_ids:
            raise protocol.IntegrityError("opened protected reset lacks a sealed receipt")
        lock = durable._CollectorLock(destination / LOCK_FILENAME, "active-confirmation")
        lock.acquire()
        try:
            started = time.perf_counter()
            for work in specs:
                if time.perf_counter() - started >= protocol.GLOBAL_WALL_SECONDS:
                    raise protocol.IntegrityError("global confirmation wall budget exceeded")
                _run_work(
                    root,
                    destination,
                    manifest,
                    work,
                    prior,
                    str(prior_payload["prior_checksum"]),
                    lock,
                )
                if _artifact_bytes(root) > int(
                    manifest["matrix"]["maximum_artifact_bytes"]
                ):
                    raise protocol.IntegrityError(
                        "T10.3.13 artifact budget exceeded during confirmation"
                    )
                if time.perf_counter() - started > protocol.GLOBAL_WALL_SECONDS:
                    raise protocol.IntegrityError(
                        "global confirmation wall budget exceeded"
                    )
        finally:
            lock.release()
        receipts = _load_receipts(destination)
        openings = _load_openings(destination)
        accounting = durable._journal_accounting(destination)

    by_game: dict[str, dict[str, dict[str, Any]]] = {}
    for row in receipts:
        by_game.setdefault(str(row["game_id"]), {})[str(row["role"])] = row
    pair_hashes_match = bool(
        len(by_game) == len(protocol.PROTECTED_GAMES)
        and all(
            set(rows) == {"candidate", "control"}
            and rows["candidate"].get("initial_frame_sha256")
            == rows["control"].get("initial_frame_sha256")
            for rows in by_game.values()
        )
    )
    checks = {
        "all_ten_receipts": len(receipts) == protocol.TOTAL_RESETS,
        "all_ten_openings": len(openings) == protocol.TOTAL_RESETS,
        "all_work_ids_unique": len({row["work_id"] for row in receipts})
        == protocol.TOTAL_RESETS,
        "work_matrix_exact": {str(row.get("work_id")) for row in receipts}
        == expected_ids,
        "all_receipts_complete": all(row.get("complete") for row in receipts),
        "all_receipts_bound_to_manifest": all(
            row.get("manifest_checksum") == manifest["manifest_checksum"]
            for row in receipts
        ),
        "all_receipts_bound_to_authorization": all(
            row.get("authorization_checksum") == manifest["authorization_checksum"]
            for row in receipts
        ),
        "all_receipts_loaded_frozen_prior": all(
            row.get("prior_checksum_loaded") == prior_payload["prior_checksum"]
            for row in receipts
        ),
        "all_openings_bound": all(
            row.get("manifest_checksum") == manifest["manifest_checksum"]
            and row.get("authorization_checksum") == manifest["authorization_checksum"]
            for row in openings
        ),
        "initial_hashes_match_within_pairs": pair_hashes_match,
        "accounting_equation": bool(accounting.get("equation_holds")),
        "zero_inflight": int(accounting.get("inflight_intents", 0)) == 0,
        "zero_unresolved": int(accounting.get("unresolved_intents", 0)) == 0,
        "zero_incomplete_work": not accounting.get("incomplete_work_ids"),
        "propose_observe_balanced": all(
            int(row.get("causal_proposals", -1))
            == int(row.get("sealed_events", -2))
            == int(row.get("causal_observations", -3))
            for row in receipts
        ),
        "action_budget": int(accounting.get("authorized_actions", 0))
        <= protocol.TOTAL_MAXIMUM_ACTIONS,
        "zero_errors": all(not row.get("errors") for row in receipts),
        "zero_illegal_actions": all(int(row.get("illegal_actions", 0)) == 0 for row in receipts),
        "zero_legacy_fallback": all(
            int(row.get("legacy_fallback_actions", 0)) == 0 for row in receipts
        ),
        "zero_physical_replay": all(
            int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts
        ),
        "no_raw_frames_persisted": all(
            row.get("raw_frames_persisted") is False for row in receipts
        ),
        "no_grounded_arguments_persisted": all(
            row.get("grounded_arguments_persisted") is False for row in receipts
        ),
    }
    collection_complete = all(checks.values())
    metrics = {
        "actions": int(accounting.get("sealed_events", 0)),
        "protected_openings": len(openings),
        "protected_frame_snapshots": sum(
            int(row.get("protected_frame_snapshots", 0)) for row in receipts
        ),
        "initial_hashes_match_within_pairs": pair_hashes_match,
        "candidate_arm": candidate,
        "control_arm": control,
        "receipt_count": len(receipts),
    }
    payload = _signed(
        {
            "format_version": "sage-t10.3.13-active-confirmation-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "authorization_checksum": manifest["authorization_checksum"],
            "prior_checksum": prior_payload["prior_checksum"],
            "collection_checks": checks,
            "collection_complete": collection_complete,
            "accounting": accounting,
            "metrics": metrics,
            "receipt_checksums": sorted(row["receipt_checksum"] for row in receipts),
            "holdout_opened": True,
            "one_final_confirmation_only": True,
            "physical_actions_replayed": 0,
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, ACTIVE_REPORT_FILENAME, payload)
    if _artifact_bytes(root) > int(manifest["matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.13 artifact budget exceeded")
    if not collection_complete:
        raise protocol.IntegrityError("protected confirmation did not seal cleanly")
    return payload


def _finite_loss(row: Mapping[str, Any]) -> float | None:
    value = row.get("prequential_log_loss")
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def adjudicate(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    active = _require_gate(root, "active-confirmation")
    receipts = _load_receipts(_destination(root))
    by_game: dict[str, dict[str, dict[str, Any]]] = {}
    for row in receipts:
        by_game.setdefault(str(row["game_id"]), {})[str(row["role"])] = row
    if set(by_game) != set(protocol.PROTECTED_GAMES) or any(
        set(rows) != {"candidate", "control"} for rows in by_game.values()
    ):
        raise protocol.IntegrityError("prospective receipt matrix is incomplete")
    for rows in by_game.values():
        if rows["candidate"].get("arm") != manifest["matrix"]["candidate_arm"]:
            raise protocol.IntegrityError("candidate receipt arm drifted")
        if rows["control"].get("arm") != manifest["matrix"]["control_arm"]:
            raise protocol.IntegrityError("control receipt arm drifted")
        for row in rows.values():
            if row.get("manifest_checksum") != manifest["manifest_checksum"]:
                raise protocol.IntegrityError("prospective receipt manifest drifted")
            if row.get("authorization_checksum") != manifest["authorization_checksum"]:
                raise protocol.IntegrityError("prospective receipt authorization drifted")

    candidate_successes = {
        game
        for game, rows in by_game.items()
        if int(rows["candidate"].get("level_delta", 0)) > 0
    }
    control_successes = {
        game
        for game, rows in by_game.items()
        if int(rows["control"].get("level_delta", 0)) > 0
    }
    higher_utility: list[str] = []
    lower_utility: list[str] = []
    better_log_loss: list[str] = []
    utility_by_game: dict[str, dict[str, float]] = {}
    log_loss_by_game: dict[str, dict[str, float | None]] = {}
    for game, rows in sorted(by_game.items()):
        candidate_utility = float(rows["candidate"].get("utility", 0.0))
        control_utility = float(rows["control"].get("utility", 0.0))
        utility_by_game[game] = {
            "candidate": candidate_utility,
            "control": control_utility,
        }
        if candidate_utility > control_utility:
            higher_utility.append(game)
        elif candidate_utility < control_utility:
            lower_utility.append(game)
        candidate_loss = _finite_loss(rows["candidate"])
        control_loss = _finite_loss(rows["control"])
        log_loss_by_game[game] = {
            "candidate": candidate_loss,
            "control": control_loss,
        }
        if (
            candidate_loss is not None
            and control_loss is not None
            and candidate_loss < control_loss
        ):
            better_log_loss.append(game)

    net_advantage = len(candidate_successes) - len(control_successes)
    checks = {
        "collection_complete": active.get("collection_complete") is True,
        "candidate_successes_at_least_three": len(candidate_successes)
        >= int(manifest["gates"]["minimum_candidate_success_games"]),
        "net_success_advantage_at_least_two": net_advantage
        >= int(manifest["gates"]["minimum_net_success_advantage"]),
        "higher_utility_on_at_least_four_games": len(higher_utility)
        >= int(manifest["gates"]["minimum_games_with_higher_utility"]),
        "never_lower_utility": len(lower_utility)
        <= int(manifest["gates"]["maximum_games_with_lower_utility"]),
        "better_log_loss_on_at_least_four_games": len(better_log_loss)
        >= int(manifest["gates"]["minimum_games_with_better_log_loss"]),
        "initial_hashes_match_within_pairs": active.get("metrics", {}).get(
            "initial_hashes_match_within_pairs"
        )
        is True,
        "zero_errors": all(not row.get("errors") for row in receipts),
        "zero_illegal_actions": all(int(row.get("illegal_actions", 0)) == 0 for row in receipts),
        "zero_legacy_fallback": all(
            int(row.get("legacy_fallback_actions", 0)) == 0 for row in receipts
        ),
        "zero_physical_replay": all(
            int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts
        ),
    }
    passed = all(checks.values())
    candidate_arm = str(manifest["matrix"]["candidate_arm"])
    if passed and candidate_arm == "source_closed_loop":
        verdict = "PASS_PROSPECTIVE_SOURCE_INFORMED_CAUSAL_PROCEDURE"
    elif passed and candidate_arm == "uniform_closed_loop":
        verdict = "PASS_PROSPECTIVE_GENERIC_CAUSAL_PROCEDURE"
    elif not checks["candidate_successes_at_least_three"]:
        verdict = "PROSPECTIVE_SUCCESS_GATE_MISS"
    elif not checks["net_success_advantage_at_least_two"]:
        verdict = "PROSPECTIVE_NET_ADVANTAGE_MISS"
    elif not (
        checks["higher_utility_on_at_least_four_games"]
        and checks["never_lower_utility"]
    ):
        verdict = "PROSPECTIVE_UTILITY_GATE_MISS"
    elif not checks["better_log_loss_on_at_least_four_games"]:
        verdict = "PROSPECTIVE_CAUSAL_PREDICTION_GATE_MISS"
    else:
        verdict = "PROSPECTIVE_SAFETY_OR_PROVENANCE_MISS"
    payload = _signed(
        {
            "format_version": "sage-t10.3.13-prospective-adjudication-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "active_report_checksum": active["report_checksum"],
            "checks": checks,
            "passed": passed,
            "verdict": verdict,
            "candidate_arm": candidate_arm,
            "control_arm": manifest["matrix"]["control_arm"],
            "candidate_success_games": sorted(candidate_successes),
            "control_success_games": sorted(control_successes),
            "net_success_advantage": net_advantage,
            "higher_utility_games": higher_utility,
            "lower_utility_games": lower_utility,
            "better_log_loss_games": better_log_loss,
            "utility_by_game": utility_by_game,
            "prequential_log_loss_by_game": log_loss_by_game,
            "prospective_bounded_generalization_evidence": passed,
            "universal_arc_generalization_proven": False,
            "one_final_confirmation_consumed": True,
            "program_promoted": False,
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
    accounting = _accounting(root)
    receipts = _load_receipts(_destination(root))
    physical_replays = sum(
        int(row.get("physical_actions_replayed", 0)) for row in receipts
    )
    legacy_fallbacks = sum(
        int(row.get("legacy_fallback_actions", 0)) for row in receipts
    )
    if not (
        accounting.get("equation_holds")
        and accounting.get("inflight_valid")
        and int(accounting.get("inflight_intents", 0)) == 0
        and int(accounting.get("unresolved_intents", 0)) == 0
        and not accounting.get("incomplete_work_ids")
        and not accounting.get("live_collector_lock")
    ) or physical_replays or legacy_fallbacks:
        raise protocol.IntegrityError(
            "T10.3.13 terminal accounting or execution integrity is not clean"
        )
    adjudication_path = _path(root, ADJUDICATION_FILENAME)
    if adjudication_path.is_file():
        adjudication = _read_signed(root, ADJUDICATION_FILENAME, "report_checksum")
        verdict = str(adjudication["verdict"])
        passed = bool(adjudication["passed"])
        prospective = bool(adjudication["prospective_bounded_generalization_evidence"])
    else:
        audit_path = _path(root, PARENT_AUDIT_FILENAME)
        preflight_path = _path(root, PREFLIGHT_FILENAME)
        audit = (
            _read_path_signed(audit_path, "audit_checksum")
            if audit_path.is_file()
            else None
        )
        preflight_report = (
            _read_path_signed(preflight_path, "preflight_checksum")
            if preflight_path.is_file()
            else None
        )
        if audit is not None and audit.get("passed") is False:
            verdict = "T10_3_13_PARENT_CANDIDATE_AUDIT_MISS"
        elif preflight_report is not None and preflight_report.get("passed") is False:
            verdict = "T10_3_13_PREFLIGHT_MISS"
        else:
            raise protocol.ScientificGateMiss(
                "T10.3.13 is incomplete; report cannot be sealed before adjudication"
            )
        passed = False
        prospective = False
    artifacts: dict[str, str | None] = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        if phase == "report":
            continue
        path = _path(root, str(contract["path"]))
        if not path.is_file():
            artifacts[phase] = None
            continue
        payload = _read_path_signed(path, str(contract["checksum_field"]))
        artifacts[phase] = str(payload[contract["checksum_field"]])
    openings = _load_openings(_destination(root))
    report = _signed(
        {
            "format_version": "sage-t10.3.13-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "authorization_checksum": manifest["authorization_checksum"],
            "parent_state": manifest["parent_state"],
            "candidate_arm": manifest["matrix"]["candidate_arm"],
            "control_arm": manifest["matrix"]["control_arm"],
            "verdict": verdict,
            "passed": passed,
            "prospective_bounded_generalization_evidence": prospective,
            "universal_arc_generalization_proven": False,
            "accounting": accounting,
            "artifacts": artifacts,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
            "holdout_opened": bool(openings),
            "holdout_opening_receipts": len(openings),
            "one_final_confirmation_consumed": bool(openings),
            "physical_actions_replayed": physical_replays,
            "legacy_fallback_actions": legacy_fallbacks,
            "ar25_opened": False,
            "source_validation_opened": False,
            "sequence_games_opened": False,
            "production_authority": False,
            "program_promoted": False,
            "automatic_retuning": False,
        },
        "report_checksum",
    )
    _write(root, TERMINAL_REPORT_FILENAME, report)
    return report


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "status",
            "authorize-holdout",
            "freeze",
            "audit-parent",
            "preflight",
            "active-confirmation",
            "adjudicate",
            "report",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--acknowledgement", default="")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.phase == "status":
            _emit(status(root))
            return 0
        if args.phase == "authorize-holdout":
            authorization = protocol.authorize_holdout(
                root,
                acknowledgement=str(args.acknowledgement),
            )
            _emit(
                {
                    "status": "HOLDOUT_AUTHORIZED_NOT_OPENED",
                    "authorization_checksum": authorization["authorization_checksum"],
                    "protected_frames_read": 0,
                    "physical_actions": 0,
                    "holdout_opened": False,
                }
            )
            return 0
        if args.phase == "freeze":
            manifest, receipt = protocol.freeze_manifest(root)
            _emit(
                {
                    "status": "FROZEN_NOT_OPENED",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "freeze_receipt_checksum": receipt["receipt_checksum"],
                    "protected_frames_read": 0,
                    "physical_actions": 0,
                    "holdout_opened": False,
                }
            )
            return 0
        manifest = protocol.load_manifest(root)
        handlers = {
            "audit-parent": audit_parent,
            "preflight": preflight,
            "active-confirmation": active_confirmation,
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
    except (protocol.IntegrityError, ValueError, KeyError, OSError, ImportError) as exc:
        _emit(
            {
                "error": "INVALID_PROVENANCE",
                "detail": f"{type(exc).__name__}:{str(exc)[:240]}",
                "exit_code": 2,
                "phase": args.phase,
            }
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        _emit(
            {
                "error": "RUNTIME_FAILURE",
                "detail": f"{type(exc).__name__}:{str(exc)[:240]}",
                "exit_code": 2,
                "phase": args.phase,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_REPORT_FILENAME",
    "ADJUDICATION_FILENAME",
    "LOCK_FILENAME",
    "PARENT_AUDIT_FILENAME",
    "PREFLIGHT_FILENAME",
    "TERMINAL_REPORT_FILENAME",
    "active_confirmation",
    "adjudicate",
    "audit_parent",
    "main",
    "preflight",
    "status",
    "terminal_report",
]
