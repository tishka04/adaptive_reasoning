"""Durable runtime for SAGE T10.3.12f causal-procedure transfer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from theory.live_transition_loop import build_observation, build_transition_record
from theory.sage12.mt.transition import compile_mt_transition
from theory.sage12.topological_invariants_v4_19 import (
    causal_factors,
    correspondence_quality,
)
from theory.unified_cognitive_controller import CognitiveDecision

from . import t10_3_2_runtime as durable
from . import t10_3_12f_protocol as protocol
from .causal_procedure_v10_3_12f import (
    ARMS,
    CausalLabelQAError,
    CausalOutcome,
    CausalProcedureController,
    CausalProcedurePrior,
    InterventionSignature,
    ProcedureNotSourceIdentifiableError,
    SourceProcedureProjection,
    abstract_context_signature,
    assert_causal_transfer_safe,
    causal_outcome_from_mt_transition,
    compile_source_prior,
    evaluate_source_prior,
    permuted_prior,
    preflight_prior,
    qa_source_projections,
    signed,
    uniform_prior,
)
from .compiler import compile_observation, compile_transition_record
from .contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    ObservedTransition,
    PredictionPacket,
)

AUDIT_FILENAME = "parent_closed_loop_negative_audit.json"
SOURCE_QA_FILENAME = "source_causal_qa_report.json"
PRIOR_FILENAME = "causal_procedure_prior.json"
PRIOR_FAILURE_FILENAME = "causal_procedure_prior_failure.json"
SOURCE_EVALUATION_FILENAME = "source_procedure_evaluation.json"
PREFLIGHT_FILENAME = "causal_procedure_preflight.json"
ACTIVE_REPORT_FILENAME = "active_historical_report.json"
ADJUDICATION_FILENAME = "causal_procedure_adjudication.json"
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
        raise protocol.IntegrityError(f"required T10.3.12f artifact is absent: {filename}")
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


def _causal_payload_is_safe(payload: Mapping[str, Any]) -> bool:
    try:
        assert_causal_transfer_safe(payload)
    except ValueError:
        return False
    return True


def audit(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent = protocol.verify_parent(root)
    source = protocol.verify_source_evidence(root)
    checks = {
        "parent_exact": parent == manifest["parent_state"],
        "source_exact": source == manifest["source_state"],
        "parent_bindings_exact": protocol.parent_artifact_bindings(root)
        == manifest["parent_artifacts"],
        "source_bindings_exact": protocol.source_artifact_bindings(root)
        == manifest["source_artifacts"],
        "parent_journal_immutable": protocol.parent_journal_digest(root)
        == manifest["parent_journal_digest"],
        "parent_events_not_training": not manifest["firewall"][
            "t10_3_12e_events_training_authorized"
        ],
        "holdout_closed": not manifest["firewall"]["holdout_opened"],
        "t10_3_13_closed": not manifest["firewall"]["t10_3_13_authorized"],
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12f-parent-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "passed": all(checks.values()),
            "parent_state": parent,
            "source_state": source,
            "physical_actions": 0,
            "holdout_opened": False,
            "production_authority": False,
        },
        "audit_checksum",
    )
    _write(root, AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.IntegrityError("T10.3.12f parent/source audit failed")
    return payload


def _trace_projection(
    trace: Mapping[str, Any],
    *,
    source_slot: str,
    group_index: int,
    terminal_family: str | None = None,
) -> SourceProcedureProjection:
    before_grid = trace.get("frame_before")
    after_grid = trace.get("frame_after")
    if before_grid is None or after_grid is None:
        raise protocol.IntegrityError("source transition is missing a frame pair")
    action_name = str(trace.get("selected_action_name", ""))
    action_data = dict(trace.get("selected_action_data", {}) or {})
    before_level = int(trace.get("levels_completed_before", 0))
    after_level = int(trace.get("levels_completed_after", before_level))
    game_after = str(trace.get("game_state_after", "NOT_FINISHED"))
    record = build_transition_record(
        action=action_name,
        action_args=action_data,
        grid_before=before_grid,
        grid_after=after_grid,
        available_actions=trace.get("available_action_names", ()),
        game_state_before=str(trace.get("game_state_before", "NOT_FINISHED")),
        game_state_after=game_after,
        levels_completed_before=before_level,
        levels_completed_after=after_level,
        infer_players=True,
    )
    mt = compile_mt_transition(
        before_grid,
        action_name,
        after_grid,
        action_data=action_data,
        productive=not record.diff.is_noop,
        risk=bool(record.diff.game_over),
    )
    quality = correspondence_quality(mt)
    correspondences = [row for row in mt.correspondences if row.kind not in {"birth", "death"}]
    persistent = [
        row
        for row in correspondences
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
    factors = causal_factors(
        mt,
        terminal_progress=after_level > before_level,
        risk=game_after.upper() == "GAME_OVER",
    )
    # The source shard records only the pre-action legal set.  Absence of a
    # post-action set is treated as unobserved, never as evidence of stability.
    action_space_changed = False
    try:
        outcome = causal_outcome_from_mt_transition(
            mt,
            action_space_changed=action_space_changed,
            level_delta=max(0, after_level - before_level),
            game_over=game_after.upper() == "GAME_OVER",
        )
        relation_conflict = False
    except ValueError as exc:
        if "conflicting relation delta" not in str(exc):
            raise
        outcome = CausalOutcome(noop=True, quality=0.0)
        relation_conflict = True
    if terminal_family is not None:
        family = terminal_family
    elif outcome.noop or outcome.game_over:
        family = "null_or_unsafe"
    elif action_space_changed:
        family = "state_conditioned_switch"
    elif factors["contact_added"] or factors["contact_removed"]:
        family = "relational_successor"
    elif factors["relative_motion"] or factors["morphology_changed"]:
        family = "stable_repeat"
    else:
        family = "state_conditioned_switch"
    if outcome.relations_added or outcome.relations_removed:
        if outcome.relations_added and outcome.relations_removed:
            relation_direction = "mixed"
        elif outcome.relations_added:
            relation_direction = "add"
        else:
            relation_direction = "remove"
        move_bucket = (
            "zero"
            if outcome.persistent_moves == 0
            else "one"
            if outcome.persistent_moves == 1
            else "many"
        )
        transform_bucket = (
            "zero"
            if outcome.persistent_transformations == 0
            else "one"
            if outcome.persistent_transformations == 1
            else "many"
        )
        outcome_mode = (
            f"relation_{relation_direction}_move_{move_bucket}_"
            f"transform_{transform_bucket}"
        )
    else:
        outcome_mode = outcome.mode
    return SourceProcedureProjection(
        source_slot=source_slot,
        group_index=group_index,
        inferred_family=family,
        outcome_mode=outcome_mode,
        correspondence_confidence=float(outcome.quality),
        persistent_one_to_one=one_to_one,
        ambiguous=bool(quality["fully_ambiguous"]),
        relation_conflict=relation_conflict,
        birth_or_death_relation=False,
        level_delta=outcome.level_delta,
        terminal_chain_link=terminal_family is not None,
    )


def _source_terminal_chain_families(root: Path) -> dict[str, dict[str, str]]:
    base = root / protocol.SOURCE_SUCCESS_JOURNAL_DIR
    game_slots = {
        "lp85-305b61c3": "source_a",
        "su15-4c352900": "source_b",
    }
    sequences: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for receipt_path in sorted((base / "branches").rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        protocol.verify_signed(receipt, "receipt_checksum")
        game_id = str(receipt.get("game_id", ""))
        if game_id not in game_slots or int(receipt.get("level_delta", 0)) <= 0:
            continue
        work_id = str(receipt.get("work_id", ""))
        intents: dict[str, dict[str, Any]] = {}
        for path in sorted((base / "intents" / work_id).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol.verify_signed(payload, "intent_checksum")
            intents[str(payload["event_id"])] = payload
        events = []
        for path in sorted((base / "events" / work_id).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol.verify_signed(payload, "event_checksum")
            events.append(payload)
        events.sort(key=lambda row: int(row.get("step_index", -1)))
        sequence = tuple(
            str(intents[str(event["event_id"])]["action"]["argument_checksum"])
            for event in events
            if "sage_t" in str(event.get("decision_source", "")).lower()
            and str(event.get("event_id", "")) in intents
        )
        if sequence:
            sequences[game_slots[game_id]].append(sequence)
    links: dict[str, dict[str, str]] = {}
    for slot in ("source_a", "source_b"):
        if not sequences[slot]:
            raise protocol.IntegrityError(
                f"no successful T10.3.8 terminal chain for {slot}"
            )
        sequence = min(sequences[slot], key=lambda row: (len(row), row))
        distinct = len(set(sequence))
        if len(sequence) >= 2 and distinct == 1:
            family = "stable_repeat"
        elif len(sequence) >= 3 and distinct == len(sequence):
            family = "relational_successor"
        else:
            raise protocol.IntegrityError(
                f"successful source chain has no unambiguous procedure family: {slot}"
            )
        links[slot] = {checksum: family for checksum in sequence}
    return links


def _source_projections(root: Path) -> tuple[SourceProcedureProjection, ...]:
    tagged_rows: list[tuple[SourceProcedureProjection, str]] = []
    terminal_links = _source_terminal_chain_families(root)
    for short, slot in (("lp85", "source_a"), ("su15", "source_b")):
        path = root / protocol.SOURCE_SHARD_DIR / f"{short}.jsonl"
        expected = 328 if short == "lp85" else 432
        arms = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                pair = json.loads(line)
                reset_index = int(pair.get("reset_index", 0))
                root_index = int(pair.get("root_index", line_index))
                group_index = reset_index * 100_000 + root_index
                for side in ("left", "right"):
                    arm = pair.get(side, {})
                    trace = arm.get("trace", {}) if isinstance(arm, Mapping) else {}
                    argument_checksum = protocol.sha256_payload(
                        dict(trace.get("selected_action_data", {}) or {})
                    )
                    tagged_rows.append(
                        (
                            _trace_projection(
                                trace,
                                source_slot=slot,
                                group_index=group_index,
                                terminal_family=terminal_links[slot].get(
                                    argument_checksum
                                ),
                            ),
                            argument_checksum,
                        )
                    )
                    arms += 1
        if arms != expected:
            raise protocol.IntegrityError(
                f"source shard arm count drifted for {short}: {arms} != {expected}"
            )
    modes_by_intervention: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    groups_by_intervention: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row, checksum in tagged_rows:
        if row.admissible and row.inferred_family != "null_or_unsafe":
            key = (row.source_slot, checksum)
            modes_by_intervention[key][row.outcome_mode] += 1
            groups_by_intervention[key].add(row.group_index)
    rows: list[SourceProcedureProjection] = []
    for row, checksum in tagged_rows:
        if row.terminal_chain_link or row.inferred_family == "null_or_unsafe":
            rows.append(row)
            continue
        key = (row.source_slot, checksum)
        modes = modes_by_intervention[key]
        support = sum(modes.values())
        dominant = max(modes.values(), default=0) / max(1, support)
        repeatable = bool(
            len(groups_by_intervention[key]) >= 2 and dominant >= 0.80
        )
        rows.append(
            replace(
                row,
                inferred_family=(
                    "stable_repeat" if repeatable else "state_conditioned_switch"
                ),
            )
        )
    return tuple(rows)


def qa_source(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit")
    projections = _source_projections(root)
    qa = qa_source_projections(projections)
    per_slot = qa["per_source"]
    checks = {
        "all_760_interventions_reconstructed": len(projections) == 760,
        "two_modes_each_source": all(
            int(per_slot[slot]["outcome_modes"])
            >= int(manifest["gates"]["source_minimum_effect_modes_per_game"])
            for slot in ("source_a", "source_b")
        ),
        "no_universal_label": all(
            float(per_slot[slot]["dominant_mode_fraction"])
            < float(manifest["gates"]["source_maximum_single_label_fraction_exclusive"])
            for slot in ("source_a", "source_b")
        ),
        "core_qa_passed": bool(qa["passed"]),
        "balanced_source_contract": qa["source_contribution_fractions"]
        == {"source_a": 0.5, "source_b": 0.5},
        "terminal_chain_link_each_source": all(
            int(per_slot[slot]["terminal_chain_link_projections"]) > 0
            for slot in ("source_a", "source_b")
        ),
        "terminal_chain_families_identified": {
            row.inferred_family
            for row in projections
            if row.source_slot == "source_a" and row.terminal_chain_link
        }
        == {"stable_repeat"}
        and {
            row.inferred_family
            for row in projections
            if row.source_slot == "source_b" and row.terminal_chain_link
        }
        == {"relational_successor"},
        "zero_source_physical_actions": True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12f-source-qa-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "passed": all(checks.values()),
            "qa": qa,
            "projection_counts": {
                "source_a": sum(row.source_slot == "source_a" for row in projections),
                "source_b": sum(row.source_slot == "source_b" for row in projections),
            },
            "raw_frames_persisted": False,
            "grounded_arguments_persisted": False,
            "physical_actions": 0,
            "verdict": "PASS_CAUSAL_LABEL_QA" if all(checks.values()) else "CAUSAL_LABEL_QA_MISS",
        },
        "report_checksum",
    )
    _write(root, SOURCE_QA_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("CAUSAL_LABEL_QA_MISS")
    return payload


def compile_prior(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "qa-source")
    destination = _destination(root)
    with _durable_contract():
        accounting = durable._journal_accounting(destination)
    if int(accounting.get("authorized_actions", 0)) != 0:
        raise protocol.IntegrityError("source prior was not compiled before target actions")
    try:
        compilation = compile_source_prior(_source_projections(root))
    except CausalLabelQAError as exc:
        _write(
            root,
            PRIOR_FAILURE_FILENAME,
            signed(
                {
                    "format_version": "sage-t10.3.12f-prior-failure-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "verdict": "CAUSAL_LABEL_QA_MISS",
                    "physical_actions": 0,
                },
                "report_checksum",
            ),
        )
        raise protocol.ScientificGateMiss("CAUSAL_LABEL_QA_MISS") from exc
    except ProcedureNotSourceIdentifiableError as exc:
        _write(
            root,
            PRIOR_FAILURE_FILENAME,
            signed(
                {
                    "format_version": "sage-t10.3.12f-prior-failure-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "verdict": "PROCEDURE_NOT_SOURCE_IDENTIFIABLE",
                    "physical_actions": 0,
                },
                "report_checksum",
            ),
        )
        raise protocol.ScientificGateMiss("PROCEDURE_NOT_SOURCE_IDENTIFIABLE") from exc
    payload = compilation.prior.snapshot()
    assert_causal_transfer_safe(payload)
    _write(root, PRIOR_FILENAME, payload)
    return payload


def evaluate_source(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "qa-source")
    prior_payload = _read_signed(root, PRIOR_FILENAME, "prior_checksum")
    prior = CausalProcedurePrior(prior_payload)
    with _durable_contract():
        accounting = durable._journal_accounting(_destination(root))
    if int(accounting.get("authorized_actions", 0)) != 0:
        raise protocol.IntegrityError(
            "source evaluation occurred after target action authorization"
        )
    result = evaluate_source_prior(prior, _source_projections(root))
    checks = {
        "grouped_leave_one_root_out": result["grouped_leave_one_root_out"] is True,
        "five_percent_log_loss_each_source": all(
            float(result["per_source"][slot]["log_loss_improvement_fraction"])
            >= float(manifest["gates"]["minimum_log_loss_improvement_over_permuted_each_source"])
            for slot in ("source_a", "source_b")
        ),
        "twenty_percent_cost_reduction_one_source": max(
            float(result["per_source"][slot]["identification_cost_reduction_fraction"])
            for slot in ("source_a", "source_b")
        )
        >= float(manifest["gates"]["minimum_identification_intervention_reduction_on_one_source"]),
        "no_cost_regression_other_source": min(
            float(result["per_source"][slot]["identification_cost_reduction_fraction"])
            for slot in ("source_a", "source_b")
        )
        >= -float(manifest["gates"]["maximum_identification_regression_on_other_source"]),
        "core_evaluation_passed": result["passed"] is True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12f-source-evaluation-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "prior_checksum": prior_payload["prior_checksum"],
            "checks": checks,
            "passed": all(checks.values()),
            "evaluation": result,
            "physical_actions": 0,
            "verdict": (
                "PASS_SOURCE_PROCEDURE_IDENTIFICATION"
                if all(checks.values())
                else "PROCEDURE_NOT_SOURCE_IDENTIFIABLE"
            ),
        },
        "report_checksum",
    )
    _write(root, SOURCE_EVALUATION_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("PROCEDURE_NOT_SOURCE_IDENTIFIABLE")
    return payload


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "evaluate-source")
    prior_payload = _read_signed(root, PRIOR_FILENAME, "prior_checksum")
    prior = CausalProcedurePrior(prior_payload)
    with _durable_contract():
        accounting = durable._journal_accounting(_destination(root))
    if int(accounting.get("authorized_actions", 0)) != 0:
        raise protocol.IntegrityError("preflight occurred after target actions")
    core_check = preflight_prior(prior)
    uniform = uniform_prior()
    synthetic_rows = [
        SourceProcedureProjection("source_a", index, "stable_repeat", mode)
        for index, mode in enumerate(("persistent_motion", "relation_change"))
    ] + [
        SourceProcedureProjection("source_b", index, "relational_successor", mode)
        for index, mode in enumerate(("relation_change", "persistent_motion"))
    ]
    universal_rows = [
        SourceProcedureProjection(slot, index, "stable_repeat", "noop")
        for slot in ("source_a", "source_b")
        for index in range(4)
    ]
    candidates = tuple(
        ActionCandidate("ACTION6", {"x": index, "y": 0}) for index in range(20)
    )
    forward = CausalProcedureController(
        "uniform_closed_loop", scope=2, prior=uniform
    )
    reverse = CausalProcedureController(
        "uniform_closed_loop", scope=2, prior=uniform
    )

    def synthetic_state(
        index: int,
        *,
        prefix: str,
        appearance: str,
        mirrored: bool,
    ) -> AbstractState:
        centers = ((0.0, 0.0), (0.0, 1.0))
        if mirrored:
            centers = tuple((2.0 - row, 2.0 - column) for row, column in centers)
        return AbstractState(
            entities=tuple(
                AbstractEntity(
                    f"{prefix}{entity_index}",
                    ("node",),
                    (("appearance", appearance),),
                    center,
                )
                for entity_index, center in enumerate(centers)
            ),
            counters=(("levels_completed", 0.0),),
            topology=(("components", index + 1),),
            regime_index=index,
        )

    original_state = synthetic_state(
        0, prefix="e", appearance="red", mirrored=False
    )
    transformed_state = synthetic_state(
        0, prefix="z", appearance="blue", mirrored=True
    )
    order_left = forward.propose(original_state, candidates)
    order_right = reverse.propose(original_state, tuple(reversed(candidates)))
    biphasic_guard = False
    try:
        forward.propose(original_state, candidates)
    except RuntimeError:
        biphasic_guard = True

    high_prior = CausalProcedurePrior(
        weights={
            "stable_repeat": 0.85,
            "relational_successor": 0.05,
            "state_conditioned_switch": 0.05,
            "null_or_unsafe": 0.05,
        }
    )
    loop = CausalProcedureController("source_closed_loop", prior=high_prior)
    candidate = ActionCandidate("ACTION6", {"x": 0, "y": 0})
    productive = CausalOutcome(persistent_moves=1)
    first = loop.propose(synthetic_state(0, prefix="e", appearance="red", mirrored=False), [candidate])
    first_update = loop.observe(
        state_before=synthetic_state(0, prefix="e", appearance="red", mirrored=False),
        state_after=synthetic_state(1, prefix="e", appearance="red", mirrored=False),
        selected=first.candidate,
        outcome=productive,
    )
    second = loop.propose(
        synthetic_state(1, prefix="e", appearance="red", mirrored=False),
        [candidate],
    )
    second_update = loop.observe(
        state_before=synthetic_state(1, prefix="e", appearance="red", mirrored=False),
        state_after=synthetic_state(2, prefix="e", appearance="red", mirrored=False),
        selected=second.candidate,
        outcome=productive,
    )
    third = loop.propose(
        synthetic_state(2, prefix="e", appearance="red", mirrored=False),
        [candidate],
    )
    mismatch = loop.observe(
        state_before=synthetic_state(2, prefix="e", appearance="red", mirrored=False),
        state_after=synthetic_state(2, prefix="e", appearance="red", mirrored=False),
        selected=third.candidate,
        outcome=CausalOutcome(noop=True),
    )
    birth_death_transition = ObservedTransition(
        state_before=original_state,
        action=candidate,
        state_after=transformed_state,
        observation=PredictionPacket(
            object_deltas={"created": 1.0, "removed": 1.0},
            known_channels=frozenset({"objects"}),
            state_after=transformed_state,
        ),
        events=("created", "removed"),
    )
    birth_death = CausalOutcome.from_observed_transition(birth_death_transition)
    empty = CausalProcedureController("uniform_closed_loop").propose(
        original_state, ()
    )
    with tempfile.TemporaryDirectory(prefix="sage-t10-3-12f-preflight-") as temporary:
        temporary_destination = Path(temporary)
        intent_path = temporary_destination / "journal" / "intents" / "work" / "0000.json"
        event_path = temporary_destination / "journal" / "events" / "work" / "0000.json"
        protocol.write_json_once(intent_path, {"event": "synthetic"})
        protocol.write_json_once(event_path, {"event": "synthetic"})
        with _durable_contract():
            synthetic_accounting = durable._journal_accounting(temporary_destination)
    permuted = permuted_prior(prior)
    checks = {
        "prior_round_trip": core_check["passed"] is True,
        "permutation_preserves_entropy": core_check["permutation_preserves_entropy"] is True,
        "uniform_is_exchangeable": len(set(uniform.weights.values())) == 1,
        "balanced_synthetic_qa": qa_source_projections(synthetic_rows)["passed"] is True,
        "universal_label_stops": qa_source_projections(universal_rows)["passed"] is False,
        "transfer_payload_safe": _causal_payload_is_safe(prior_payload),
        "posterior_update_is_post_action": biphasic_guard
        and first_update.phase_after == "VERIFY",
        "identify_verify_control_sequence": second_update.phase_after == "CONTROL",
        "mismatch_revision_is_bounded": mismatch.revised is True
        and loop.summary()["revisions"] <= 2,
        "d4_palette_and_id_invariance": abstract_context_signature(original_state)
        == abstract_context_signature(transformed_state),
        "candidate_order_invariance": order_left.candidate is not None
        and order_right.candidate is not None
        and order_left.candidate.key == order_right.candidate.key,
        "argument_and_action_identity_invariance": InterventionSignature.from_candidate(
            ActionCandidate("ACTION6", {"x": 1, "y": 2})
        )
        == InterventionSignature.from_candidate(
            ActionCandidate("ACTION19", {"x": 9, "y": 8})
        ),
        "birth_death_relation_exclusion": birth_death.persistent_moves == 0
        and birth_death.persistent_transformations == 0
        and not birth_death.relations_added
        and not birth_death.relations_removed,
        "no_grounding_abstains": empty.abstained,
        "permuted_prior_ranking_inverts": max(
            prior.weights, key=prior.weights.__getitem__
        )
        != max(permuted.weights, key=permuted.weights.__getitem__),
        "durable_equation_synthetic": synthetic_accounting["equation_holds"]
        and synthetic_accounting["inflight_intents"] == 0,
        "holdout_closed": manifest["firewall"]["holdout_opened"] is False,
    }
    assert_causal_transfer_safe(prior_payload)
    payload = signed(
        {
            "format_version": "sage-t10.3.12f-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "prior_checksum": prior_payload["prior_checksum"],
            "checks": checks,
            "passed": all(checks.values()),
            "physical_actions": 0,
        },
        "preflight_checksum",
    )
    _write(root, PREFLIGHT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss("CAUSAL_PROCEDURE_PREFLIGHT_MISS")
    return payload


class _ActiveProcedure:
    def __init__(self, work: protocol.WorkSpec, prior: CausalProcedurePrior) -> None:
        self.controller = CausalProcedureController(
            work.arm,
            scope=work.scope_index,
            prior=prior,
        )
        self.maximum_legal_candidates = 0
        self.candidate_inspections = 0
        self.interventions_before_verification: int | None = None
        self.entered_verify = False
        self.entered_control = False

    def decide(
        self,
        *,
        current_grid: Any,
        legal_actions: Sequence[Any],
        game_state: str,
        levels_completed: int,
        step_index: int,
    ) -> tuple[CognitiveDecision | None, Any, Any]:
        observation = build_observation(
            current_grid,
            available_actions=durable.live._available_action_names(legal_actions),
            game_state=game_state,
            levels_completed=levels_completed,
            infer_players=True,
        )
        state = compile_observation(observation)
        decision = self.controller.propose(state, legal_actions, step_index=step_index)
        self.maximum_legal_candidates = max(self.maximum_legal_candidates, len(legal_actions))
        self.candidate_inspections += decision.candidates_inspected
        if decision.abstained:
            return None, decision, state
        return (
            CognitiveDecision(
                action_name=decision.candidate.action_name,
                action_data=dict(decision.candidate.action_data),
                source="sage_t_causal_procedure",
                reason=decision.reason,
                confidence=decision.predicted_probability,
                option_id=decision.program_hash,
            ),
            decision,
            state,
        )

    def summary(self) -> dict[str, Any]:
        output = dict(self.controller.summary())
        belief = dict(output.pop("belief", {}) or {})
        family_rows = dict(belief.get("families", {}) or {})
        output.update(
            {
                "maximum_legal_candidates": self.maximum_legal_candidates,
                "candidate_inspections": self.candidate_inspections,
                "interventions_before_verification": self.interventions_before_verification,
                "entered_verify": self.entered_verify,
                "entered_control": self.entered_control,
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
        return output


def _work_path(destination: Path, category: str, work: protocol.WorkSpec, name: str) -> Path:
    return destination / "journal" / category / work.work_id / name


def _receipt_path(destination: Path, work: protocol.WorkSpec) -> Path:
    return _work_path(destination, "branches", work, "receipt.json")


def _load_receipts(destination: Path) -> list[dict[str, Any]]:
    base = destination / "journal" / "branches"
    paths = sorted(base.rglob("receipt.json")) if base.exists() else []
    return [durable._read_signed(path, "receipt_checksum") for path in paths]


def _validate_receipt_binding(
    receipt: Mapping[str, Any],
    *,
    work: protocol.WorkSpec,
    manifest_checksum: str,
    prior_checksum: str,
) -> None:
    if receipt.get("complete") is not True:
        raise protocol.IntegrityError("branch receipt is not complete")
    if receipt.get("manifest_checksum") != manifest_checksum:
        raise protocol.IntegrityError("branch receipt is detached from the manifest")
    if receipt.get("prior_checksum_loaded") != prior_checksum:
        raise protocol.IntegrityError("branch receipt is detached from the causal prior")
    if receipt.get("work_id") != work.work_id:
        raise protocol.IntegrityError("branch receipt work id drifted")
    for key, value in work.as_dict().items():
        if receipt.get(key) != value:
            raise protocol.IntegrityError(f"branch receipt work field drifted: {key}")
    issued = int(receipt.get("issued_intents", -1))
    sealed = int(receipt.get("sealed_events", -1))
    observed = int(receipt.get("observed_updates", -1))
    if not 0 <= issued <= work.action_budget or not 0 <= sealed <= issued:
        raise protocol.IntegrityError("branch receipt action bounds drifted")
    if not 0 <= observed <= sealed:
        raise protocol.IntegrityError("branch receipt observation bounds drifted")


def _intent(
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    *,
    step_index: int,
    selected: Any,
    decision: Any,
    prior_checksum: str,
) -> dict[str, Any]:
    args = dict(getattr(selected, "action_args", {}) or {})
    event_id = protocol.sha256_payload(
        {"manifest": manifest["manifest_checksum"], "work": work.work_id, "step": step_index}
    )
    return signed(
        {
            "format_version": "sage-t10.3.12f-action-intent-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "event_id": event_id,
            "step_index": step_index,
            "decision": decision.safe_payload,
            "physical_action": {
                "operator_checksum": protocol.sha256_payload(
                    str(getattr(selected, "name", ""))
                ),
                "parameter_arity": len(args),
                "argument_checksum": protocol.sha256_payload(args),
            },
            "prior_checksum": prior_checksum,
            "raw_arguments_retained": False,
            "physical_action_replayed": False,
        },
        "intent_checksum",
    )


def _frame_hash(snapshot: Any) -> str:
    grid = snapshot.grid
    value = grid.tolist() if hasattr(grid, "tolist") else grid
    return protocol.sha256_payload(value)


def _compile_causal_transition(
    *,
    before: Any,
    after: Any,
    selected: Any,
    legal_before: Sequence[Any],
    legal_after: Sequence[Any],
) -> tuple[Any, CausalOutcome, dict[str, Any]]:
    action_name = str(getattr(selected, "name", ""))
    action_data = dict(getattr(selected, "action_args", {}) or {})
    record = build_transition_record(
        action=action_name,
        action_args=action_data,
        grid_before=before.grid,
        grid_after=after.grid,
        available_actions=durable.live._available_action_names(legal_before),
        game_state_before=str(before.game_state),
        game_state_after=str(after.game_state),
        levels_completed_before=int(before.levels_completed),
        levels_completed_after=int(after.levels_completed),
        infer_players=True,
    )
    observed = compile_transition_record(record)
    mt = compile_mt_transition(
        before.grid,
        action_name,
        after.grid,
        action_data=action_data,
        productive=not record.diff.is_noop,
        risk=str(after.game_state).upper() == "GAME_OVER",
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
    admissible = bool(one_to_one and not quality["fully_ambiguous"])
    correspondence_score = (
        float(quality["mean_confidence"]) if admissible else 0.0
    )
    names_before = set(durable.live._available_action_names(legal_before))
    names_after = set(durable.live._available_action_names(legal_after))
    action_space_changed = names_before != names_after
    relation_conflict = False
    try:
        outcome = causal_outcome_from_mt_transition(
            mt,
            action_space_changed=action_space_changed,
            level_delta=max(
                0,
                int(after.levels_completed) - int(before.levels_completed),
            ),
            game_over=str(after.game_state).upper() == "GAME_OVER",
        )
    except ValueError as exc:
        if "conflicting relation delta" not in str(exc):
            raise
        relation_conflict = True
        outcome = CausalOutcome(
            noop=bool(record.diff.is_noop),
            game_over=str(after.game_state).upper() == "GAME_OVER",
            level_delta=max(
                0,
                int(after.levels_completed) - int(before.levels_completed),
            ),
            quality=0.0,
        )
    diagnostics = {
        "correspondence_quality": correspondence_score,
        "persistent_one_to_one": one_to_one,
        "ambiguous_correspondence": bool(quality["fully_ambiguous"]),
        "relation_conflict_rejected": relation_conflict,
        "birth_death_relation_evidence_used": False,
    }
    return observed, outcome, diagnostics


def _run_work(
    root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    prior: CausalProcedurePrior,
    prior_checksum: str,
    lock: Any,
) -> dict[str, Any]:
    receipt_path = _receipt_path(destination, work)
    if receipt_path.is_file():
        receipt = durable._read_signed(receipt_path, "receipt_checksum")
        _validate_receipt_binding(
            receipt,
            work=work,
            manifest_checksum=str(manifest["manifest_checksum"]),
            prior_checksum=prior_checksum,
        )
        return receipt

    executor = _ActiveProcedure(work, prior)
    event_ids: list[str] = []
    prequential_losses: list[float] = []
    errors: list[str] = []
    illegal_actions = 0
    game_over_actions = 0
    noop_actions = 0
    initial_level = final_level = 0
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
        environment = durable.live._make_real_env(
            work.game_id,
            root / "environment_files",
        )
        frame = durable.live._reset_env(environment)
        initial = durable.live.snapshot_frame(frame)
        initial_level = final_level = int(initial.levels_completed)
        initial_frame_hash = _frame_hash(initial)

        for step_index in range(work.action_budget):
            if time.perf_counter() - started >= protocol.reset_wall_seconds(work):
                stop_reason = "RESET_WALL_BUDGET_EXHAUSTED"
                break
            before = durable.live.snapshot_frame(frame)
            if durable.live._is_terminal(before.game_state):
                stop_reason = "TERMINAL_STATE"
                break
            legal_before = tuple(durable.live._valid_actions(environment))
            cognitive, decision, state_before = executor.decide(
                current_grid=before.grid,
                legal_actions=legal_before,
                game_state=str(before.game_state),
                levels_completed=int(before.levels_completed),
                step_index=step_index,
            )
            if cognitive is None:
                stop_reason = "PLANNED_CAUSAL_ABSTENTION"
                break
            if cognitive.source != "sage_t_causal_procedure":
                raise protocol.IntegrityError(
                    "non-causal-procedure decision entered T10.3.12f"
                )
            selected = durable.live._materialize_decision(legal_before, cognitive)
            if selected is None:
                illegal_actions += 1
                errors.append("UNAVAILABLE_CAUSAL_DECISION")
                status_value = "ABORTED"
                stop_reason = "UNAVAILABLE_CAUSAL_DECISION"
                break

            current_intent = _intent(
                manifest,
                work,
                step_index=step_index,
                selected=selected,
                decision=decision,
                prior_checksum=prior_checksum,
            )
            current_name = f"{step_index:04d}.json"
            current_event_sealed = False
            protocol.write_json_once(
                _work_path(destination, "intents", work, current_name),
                current_intent,
            )
            lock.heartbeat()
            try:
                after_frame = durable.live._step_env_action(environment, selected)
                after = durable.live.snapshot_frame(
                    after_frame,
                    fallback_available_actions=before.available_actions,
                )
            except Exception as exc:  # noqa: BLE001
                unresolved = signed(
                    {
                        "format_version": "sage-t10.3.12f-unresolved-event-v1",
                        "manifest_checksum": manifest["manifest_checksum"],
                        "work_id": work.work_id,
                        "event_id": current_intent["event_id"],
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
                    _work_path(destination, "unresolved", work, current_name),
                    unresolved,
                )
                errors.append("ENVIRONMENT_CALL_UNATTESTABLE")
                status_value = "ABORTED"
                stop_reason = "ENVIRONMENT_CALL_UNATTESTABLE"
                break

            legal_after = tuple(durable.live._valid_actions(environment))
            compilation_error = ""
            try:
                observed, outcome, evidence = _compile_causal_transition(
                    before=before,
                    after=after,
                    selected=selected,
                    legal_before=legal_before,
                    legal_after=legal_after,
                )
            except Exception as exc:  # noqa: BLE001
                compilation_error = type(exc).__name__
                observed = None
                outcome = CausalOutcome(
                    noop=_frame_hash(before) == _frame_hash(after),
                    game_over=str(after.game_state).upper() == "GAME_OVER",
                    level_delta=max(
                        0,
                        int(after.levels_completed) - int(before.levels_completed),
                    ),
                    quality=0.0,
                )
                evidence = {
                    "correspondence_quality": 0.0,
                    "persistent_one_to_one": False,
                    "ambiguous_correspondence": True,
                    "relation_conflict_rejected": False,
                    "birth_death_relation_evidence_used": False,
                }

            state_after = compile_observation(
                build_observation(
                    after.grid,
                    available_actions=durable.live._available_action_names(legal_after),
                    game_state=str(after.game_state),
                    levels_completed=int(after.levels_completed),
                    infer_players=True,
                )
            )
            event = signed(
                {
                    "format_version": "sage-t10.3.12f-physical-event-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "work_id": work.work_id,
                    "event_id": current_intent["event_id"],
                    "step_index": step_index,
                    "procedure_prediction": decision.safe_payload,
                    "causal_outcome": outcome.safe_payload,
                    "outcome_signature": outcome.signature,
                    "abstract_state_before": state_before.signature,
                    "abstract_state_after": state_after.signature,
                    "abstract_context_before": abstract_context_signature(state_before),
                    "correspondence": evidence,
                    "causal_compilation_error": compilation_error or None,
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
            # The action result is durable before any target-local posterior update.
            protocol.write_json_once(
                _work_path(destination, "events", work, current_name),
                event,
            )
            current_event_sealed = True
            event_ids.append(str(current_intent["event_id"]))
            lock.heartbeat()
            if observed is None:
                update = executor.controller.observe(
                    state_before=state_before,
                    state_after=state_after,
                    selected=decision.candidate,
                    outcome=outcome,
                )
            else:
                update = executor.controller.observe(observed, outcome=outcome)
            if (
                executor.interventions_before_verification is None
                and update.phase_after == "CONTROL"
            ):
                executor.interventions_before_verification = step_index + 1
            executor.entered_verify = bool(
                getattr(executor, "entered_verify", False)
                or update.phase_before == "VERIFY"
                or update.phase_after == "VERIFY"
            )
            executor.entered_control = bool(
                getattr(executor, "entered_control", False)
                or update.phase_before == "CONTROL"
                or update.phase_after == "CONTROL"
            )
            probability = max(1e-12, float(update.predicted_probability))
            prequential_losses.append(-math.log(probability))
            update_safe = update.safe_payload
            update_payload = signed(
                {
                    "format_version": "sage-t10.3.12f-procedure-update-v1",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "work_id": work.work_id,
                    "event_id": current_intent["event_id"],
                    "step_index": step_index,
                    "phase_before": update.phase_before,
                    "phase_after": update.phase_after,
                    "predicted_family": update.predicted_family,
                    "predicted_probability": update.predicted_probability,
                    "outcome_signature": outcome.signature,
                    "posterior_digest": protocol.sha256_payload(
                        update_safe["posterior"]
                    ),
                    "mismatch": update.mismatch,
                    "revised": update.revised,
                    "abstained": update.abstained,
                    "reason": update.reason,
                    "prequential_log_loss": prequential_losses[-1],
                    "grounded_payload_retained": False,
                },
                "update_checksum",
            )
            protocol.write_json_once(
                _work_path(destination, "updates", work, current_name),
                update_payload,
            )
            frame = after_frame
            final_level = int(after.levels_completed)
            game_over_actions += int(outcome.game_over)
            noop_actions += int(outcome.noop)
            current_intent = None
            if time.perf_counter() - started > protocol.reset_wall_seconds(work):
                errors.append("RESET_WALL_BUDGET_EXCEEDED")
                status_value = "ABORTED"
                stop_reason = "RESET_WALL_BUDGET_EXCEEDED"
                break
            if compilation_error:
                errors.append(f"CAUSAL_COMPILATION:{compilation_error}")
                status_value = "ABORTED"
                stop_reason = "CAUSAL_COMPILATION_ERROR"
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
            unresolved_path = _work_path(destination, "unresolved", work, current_name)
            if not event_path.exists() and not unresolved_path.exists():
                unresolved = signed(
                    {
                        "format_version": "sage-t10.3.12f-unresolved-event-v1",
                        "manifest_checksum": manifest["manifest_checksum"],
                        "work_id": work.work_id,
                        "event_id": current_intent["event_id"],
                        "step_index": current_intent["step_index"],
                        "reason": f"POST_ACTION_PIPELINE:{type(exc).__name__}",
                        "physical_action_replayed": False,
                    },
                    "unresolved_checksum",
                )
                protocol.write_json_once(unresolved_path, unresolved)
    finally:
        if environment is not None:
            try:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ENVIRONMENT_CLOSE:{type(exc).__name__}")

    intent_dir = _work_path(destination, "intents", work, "x").parent
    event_dir = _work_path(destination, "events", work, "x").parent
    update_dir = _work_path(destination, "updates", work, "x").parent
    unresolved_dir = _work_path(destination, "unresolved", work, "x").parent
    issued = len(tuple(intent_dir.glob("*.json"))) if intent_dir.exists() else 0
    sealed = len(tuple(event_dir.glob("*.json"))) if event_dir.exists() else 0
    observed_count = len(tuple(update_dir.glob("*.json"))) if update_dir.exists() else 0
    unresolved = (
        len(tuple(unresolved_dir.glob("*.json"))) if unresolved_dir.exists() else 0
    )
    summary = executor.summary()
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
    receipt = signed(
        {
            "format_version": "sage-t10.3.12f-branch-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
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
            "raw_frames_persisted": False,
            "grounded_arguments_persisted": False,
            "physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    protocol.write_json_once(receipt_path, receipt)
    return receipt


def _receipt_metrics(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in receipts if row.get("arm") == arm]
        losses = [
            float(row["prequential_log_loss"])
            for row in rows
            if row.get("prequential_log_loss") is not None
        ]
        by_arm[arm] = {
            "resets": len(rows),
            "success_resets": sum(int(row.get("level_delta", 0)) > 0 for row in rows),
            "success_games": sorted(
                {
                    str(row["game_id"])
                    for row in rows
                    if int(row.get("level_delta", 0)) > 0
                }
            ),
            "actions": sum(int(row.get("sealed_events", 0)) for row in rows),
            "mean_utility": statistics.fmean(
                [float(row.get("utility", 0.0)) for row in rows]
            )
            if rows
            else 0.0,
            "mean_prequential_log_loss": (
                statistics.fmean(losses) if losses else None
            ),
            "mismatches": sum(
                int(dict(row.get("procedure_summary", {})).get("mismatches", 0))
                for row in rows
            ),
            "revisions": sum(
                int(dict(row.get("procedure_summary", {})).get("revisions", 0))
                for row in rows
            ),
            "planned_abstentions": sum(bool(row.get("planned_abstention")) for row in rows),
            "game_over_resets": sum(int(row.get("game_over_actions", 0)) > 0 for row in rows),
            "noop_actions": sum(int(row.get("noop_actions", 0)) for row in rows),
        }
    return by_arm


def active_historical(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "preflight")
    prior_payload = _read_signed(root, PRIOR_FILENAME, "prior_checksum")
    prior = CausalProcedurePrior(prior_payload)
    if prior.kind != "source_informed":
        raise protocol.IntegrityError("active prior is not source-informed")
    specs = protocol.work_specs("active-historical")
    expected_ids = {work.work_id for work in specs}
    destination = _destination(root)
    with _durable_contract():
        durable._require_live_runtime()
        before = durable._journal_accounting(destination)
        if not before.get("equation_holds") or not before.get("inflight_valid"):
            raise protocol.IntegrityError(
                "historical journal accounting is invalid before collection"
            )
        if before.get("inflight_paths") or before.get("unresolved_intents"):
            raise protocol.IntegrityError("interrupted physical action cannot be replayed")
        if before.get("incomplete_work_ids"):
            raise protocol.IntegrityError("interrupted work cannot be reconstructed safely")
        existing = _load_receipts(destination)
        specs_by_id = {work.work_id: work for work in specs}
        for row in existing:
            work_id = str(row.get("work_id", ""))
            if work_id not in expected_ids:
                raise protocol.IntegrityError(
                    "historical journal contains an unknown work id"
                )
            _validate_receipt_binding(
                row,
                work=specs_by_id[work_id],
                manifest_checksum=str(manifest["manifest_checksum"]),
                prior_checksum=str(prior_payload["prior_checksum"]),
            )
        lock = durable._CollectorLock(destination / LOCK_FILENAME, "active-historical")
        lock.acquire()
        try:
            collection_started = time.perf_counter()
            for work in specs:
                if (
                    time.perf_counter() - collection_started
                    >= float(manifest["matrix"]["maximum_global_wall_seconds"])
                ):
                    raise protocol.IntegrityError(
                        "global historical wall budget exceeded"
                    )
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
                        "T10.3.12f artifact budget exceeded during collection"
                    )
                if (
                    time.perf_counter() - collection_started
                    > float(manifest["matrix"]["maximum_global_wall_seconds"])
                ):
                    raise protocol.IntegrityError(
                        "global historical wall budget exceeded"
                    )
        finally:
            lock.release()
        receipts = _load_receipts(destination)
        accounting = durable._journal_accounting(destination)

    receipt_ids = {str(row.get("work_id")) for row in receipts}
    specs_by_id = {work.work_id: work for work in specs}
    for row in receipts:
        work_id = str(row.get("work_id", ""))
        if work_id not in specs_by_id:
            raise protocol.IntegrityError("historical receipt has an unknown work id")
        _validate_receipt_binding(
            row,
            work=specs_by_id[work_id],
            manifest_checksum=str(manifest["manifest_checksum"]),
            prior_checksum=str(prior_payload["prior_checksum"]),
        )
    update_paths = sorted((destination / "journal" / "updates").rglob("*.json")) if (
        destination / "journal" / "updates"
    ).exists() else []
    checks = {
        "all_144_receipts": len(receipts) == protocol.TOTAL_RESETS,
        "all_expected_work_ids": receipt_ids == expected_ids,
        "all_receipts_complete": all(bool(row.get("complete")) for row in receipts),
        "accounting_equation": bool(accounting.get("equation_holds")),
        "inflight_valid": bool(accounting.get("inflight_valid")),
        "zero_inflight": int(accounting.get("inflight_intents", 0)) == 0,
        "zero_unresolved": int(accounting.get("unresolved_intents", 0)) == 0,
        "zero_incomplete_work": not accounting.get("incomplete_work_ids"),
        "every_event_observed": len(update_paths)
        == int(accounting.get("sealed_events", 0)),
        "zero_physical_replay": all(
            int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts
        ),
        "zero_legacy_fallback": all(
            int(row.get("legacy_fallback_actions", 0)) == 0 for row in receipts
        ),
        "zero_illegal_actions": all(
            int(row.get("illegal_actions", 0)) == 0 for row in receipts
        ),
        "action_budget": int(accounting.get("sealed_events", 0))
        <= protocol.TOTAL_MAXIMUM_ACTIONS,
    }
    initial_hashes: dict[str, list[str]] = defaultdict(list)
    for row in receipts:
        game = str(row.get("game_id", ""))
        value = str(row.get("initial_frame_sha256", ""))
        if value and value not in initial_hashes[game]:
            initial_hashes[game].append(value)
    payload = signed(
        {
            "format_version": "sage-t10.3.12f-active-historical-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "prior_checksum": prior_payload["prior_checksum"],
            "collection_checks": checks,
            "collection_complete": all(checks.values()),
            "accounting": accounting,
            "metrics": {
                "by_arm": _receipt_metrics(receipts),
                "initial_frame_hashes": dict(initial_hashes),
                "distinct_initial_frame_hashes": len(
                    {value for values in initial_hashes.values() for value in values}
                ),
                "work_scopes_are_environment_seeds": False,
                "physical_actions": int(accounting.get("sealed_events", 0)),
            },
            "receipt_checksums": sorted(
                str(row["receipt_checksum"]) for row in receipts
            ),
            "historical_diagnostic_only": True,
            "confirmatory_evidence": False,
            "holdout_opened": False,
            "production_authority": False,
            "physical_actions_replayed": 0,
        },
        "report_checksum",
    )
    _write(root, ACTIVE_REPORT_FILENAME, payload)
    if _artifact_bytes(root) > int(manifest["matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.12f artifact budget exceeded")
    if not payload["collection_complete"]:
        raise protocol.IntegrityError("historical collection did not seal cleanly")
    return payload


def _game_arm_values(
    receipts: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, dict[str, float | None]]:
    values: dict[str, dict[str, float | None]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for game in protocol.TARGET_GAMES:
            rows = [
                row
                for row in receipts
                if row.get("arm") == arm and row.get("game_id") == game
            ]
            observed = [
                float(row[field]) for row in rows if row.get(field) is not None
            ]
            values[arm][game] = statistics.fmean(observed) if observed else None
    return values


def _identification_advantage(
    receipts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require reset-robust identification plus an advantage over controls.

    This is only a diagnostic negative-result classifier.  It cannot create a
    PASS and it never substitutes prediction quality for a level delta.
    """

    minimum_verified_games = int(
        manifest["gates"]["minimum_identification_verified_games"]
    )
    minimum_better_games = int(
        manifest["gates"]["minimum_identification_better_games_each_control"]
    )
    loss = _game_arm_values(receipts, "prequential_log_loss")
    verification_cost: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    verified_games: dict[str, list[str]] = {arm: [] for arm in ARMS}

    for arm in ARMS:
        for game in protocol.TARGET_GAMES:
            rows = [
                row
                for row in receipts
                if row.get("arm") == arm and row.get("game_id") == game
            ]
            costs: list[float] = []
            reset_verified: list[bool] = []
            for row in rows:
                summary = dict(row.get("procedure_summary", {}))
                verified = bool(
                    summary.get("entered_control")
                    and int(summary.get("verified_context_diversity", 0)) >= 2
                )
                reset_verified.append(verified)
                value = summary.get("interventions_before_verification")
                costs.append(
                    float(value)
                    if verified and value is not None
                    else float(protocol.ACTION_BUDGET + 1)
                )
            verification_cost[arm][game] = (
                statistics.fmean(costs)
                if len(rows) == len(protocol.WORK_SCOPES) and len(costs) == len(rows)
                else float(protocol.ACTION_BUDGET + 1)
            )
            if (
                len(rows) == len(protocol.WORK_SCOPES)
                and len(reset_verified) == len(rows)
                and all(reset_verified)
            ):
                verified_games[arm].append(game)

    candidates = {
        "source_closed_loop": (
            "uniform_closed_loop",
            "permuted_source_closed_loop",
            "source_open_loop",
        ),
        "uniform_closed_loop": ("source_open_loop",),
    }
    diagnostics: dict[str, Any] = {}
    selected: str | None = None
    for candidate, controls in candidates.items():
        comparisons: dict[str, Any] = {}
        for control in controls:
            paired = [
                game
                for game in protocol.TARGET_GAMES
                if loss[candidate][game] is not None and loss[control][game] is not None
            ]
            loss_better = [
                game
                for game in paired
                if float(loss[candidate][game]) < float(loss[control][game])
            ]
            cost_better = [
                game
                for game in protocol.TARGET_GAMES
                if verification_cost[candidate][game] < verification_cost[control][game]
            ]
            comparisons[control] = {
                "paired_log_loss_games": len(paired),
                "better_log_loss_games": len(loss_better),
                "better_verification_cost_games": len(cost_better),
                "passed": bool(
                    len(loss_better) >= minimum_better_games
                    and len(cost_better) >= minimum_better_games
                ),
            }
        candidate_passed = bool(
            len(verified_games[candidate]) >= minimum_verified_games
            and all(row["passed"] for row in comparisons.values())
        )
        diagnostics[candidate] = {
            "verified_games": verified_games[candidate],
            "comparisons": comparisons,
            "passed": candidate_passed,
        }
        if selected is None and candidate_passed:
            selected = candidate

    return {
        "passed": selected is not None,
        "candidate_arm": selected,
        "minimum_verified_games": minimum_verified_games,
        "minimum_better_games_each_control": minimum_better_games,
        "candidates": diagnostics,
    }


def _exact_positive_sign_permutation(differences: Sequence[float]) -> float:
    nonzero = [float(value) for value in differences if abs(float(value)) > 1e-15]
    if not nonzero:
        return 1.0
    observed = sum(nonzero)
    extreme = 0
    total = 1 << len(nonzero)
    for mask in range(total):
        value = sum(
            item if mask & (1 << index) else -item
            for index, item in enumerate(nonzero)
        )
        extreme += int(value >= observed - 1e-15)
    return extreme / total


def _contrast(
    values: Mapping[str, Mapping[str, float | None]],
    left: str,
    right: str,
) -> dict[str, Any]:
    pairs = []
    for game in protocol.TARGET_GAMES:
        left_value = values[left][game]
        right_value = values[right][game]
        if left_value is not None and right_value is not None:
            pairs.append((game, float(left_value), float(right_value)))
    differences = [left_value - right_value for _, left_value, right_value in pairs]
    return {
        "left": left,
        "right": right,
        "game_differences": {game: left_value - right_value for game, left_value, right_value in pairs},
        "mean_difference": statistics.fmean(differences) if differences else 0.0,
        "raw_p_value": _exact_positive_sign_permutation(differences),
        "paired_games": len(pairs),
    }


def _holm(
    contrasts: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    ordered = sorted(
        contrasts,
        key=lambda name: (float(contrasts[name]["raw_p_value"]), name),
    )
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    still_rejecting = True
    rejected: dict[str, bool] = {}
    for index, name in enumerate(ordered):
        raw = float(contrasts[name]["raw_p_value"])
        running = max(running, min(1.0, (count - index) * raw))
        adjusted[name] = running
        threshold = alpha / (count - index)
        rejected[name] = bool(still_rejecting and raw <= threshold)
        if raw > threshold:
            still_rejecting = False
    return {
        name: {
            **dict(contrasts[name]),
            "holm_adjusted_p_value": adjusted[name],
            "holm_reject": rejected[name],
        }
        for name in contrasts
    }


SOURCE_PASS = (
    "PASS_T10_3_12F_HISTORICAL_SOURCE_INFORMED_"
    "CAUSAL_PROCEDURE_CANDIDATE"
)
GENERIC_PASS = (
    "PASS_T10_3_12F_HISTORICAL_GENERIC_CAUSAL_PROCEDURE_CANDIDATE"
)


def adjudicate(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    active = _require_gate(root, "active-historical")
    receipts = _load_receipts(_destination(root))
    utility = _game_arm_values(receipts, "utility")
    log_loss = _game_arm_values(receipts, "prequential_log_loss")
    identification = _identification_advantage(receipts, manifest)
    source_raw = {
        "source_vs_uniform": _contrast(
            utility, "source_closed_loop", "uniform_closed_loop"
        ),
        "source_vs_permuted": _contrast(
            utility, "source_closed_loop", "permuted_source_closed_loop"
        ),
        "source_vs_open_loop": _contrast(
            utility, "source_closed_loop", "source_open_loop"
        ),
    }
    source_tests = _holm(
        source_raw,
        alpha=float(manifest["gates"]["holm_familywise_alpha"]),
    )
    generic_test = _contrast(
        utility,
        "uniform_closed_loop",
        "source_open_loop",
    )
    generic_test["significant"] = bool(
        float(generic_test["mean_difference"]) > 0
        and float(generic_test["raw_p_value"])
        <= float(manifest["gates"]["holm_familywise_alpha"])
    )
    success_games = {
        arm: sorted(
            {
                str(row["game_id"])
                for row in receipts
                if row.get("arm") == arm and int(row.get("level_delta", 0)) > 0
            }
        )
        for arm in ARMS
    }
    source_specific = all(
        bool(row["holm_reject"]) and float(row["mean_difference"]) > 0
        for row in source_tests.values()
    )
    source_advantage_over_uniform = bool(
        source_tests["source_vs_uniform"]["holm_reject"]
        and float(source_tests["source_vs_uniform"]["mean_difference"]) > 0
    )
    source_candidate = bool(
        len(success_games["source_closed_loop"])
        >= int(manifest["gates"]["minimum_candidate_success_games"])
        and source_specific
    )
    generic_candidate = bool(
        len(success_games["uniform_closed_loop"])
        >= int(manifest["gates"]["minimum_candidate_success_games"])
        and generic_test["significant"]
        and not source_advantage_over_uniform
    )
    safety = {
        "active_collection_complete": active.get("collection_complete") is True,
        "all_receipts_present": len(receipts) == protocol.TOTAL_RESETS,
        "zero_illegal_actions": all(
            int(row.get("illegal_actions", 0)) == 0 for row in receipts
        ),
        "zero_legacy_fallback": all(
            int(row.get("legacy_fallback_actions", 0)) == 0 for row in receipts
        ),
        "zero_physical_replay": all(
            int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts
        ),
        "all_actions_observed": all(
            int(row.get("sealed_events", 0))
            == int(row.get("observed_updates", 0))
            for row in receipts
        ),
    }
    if not all(safety.values()):
        raise protocol.IntegrityError("T10.3.12f adjudication safety gate failed")
    passed = source_candidate or generic_candidate
    candidate_arm: str | None = None
    control_arm: str | None = None
    if source_candidate:
        verdict = SOURCE_PASS
        candidate_arm = "source_closed_loop"
        control_arm = "uniform_closed_loop"
    elif generic_candidate:
        verdict = GENERIC_PASS
        candidate_arm = "uniform_closed_loop"
        control_arm = "source_open_loop"
    else:
        maximum_success_games = max(map(len, success_games.values()), default=0)
        verified = bool(identification["passed"])
        if maximum_success_games == 0:
            verdict = (
                "CAUSAL_IDENTIFICATION_WITHOUT_CONTROL"
                if verified
                else "CAUSAL_PROCEDURE_NO_TARGET_PROGRESS"
            )
        elif maximum_success_games == 1:
            verdict = "SINGLE_GAME_EFFECT_ONLY"
        elif not (
            source_tests["source_vs_permuted"]["holm_reject"]
            and float(source_tests["source_vs_permuted"]["mean_difference"]) > 0
        ):
            verdict = "SOURCE_PRIOR_NOT_SPECIFIC"
        elif verified:
            verdict = "CAUSAL_IDENTIFICATION_WITHOUT_CONTROL"
        else:
            verdict = "CAUSAL_PROCEDURE_NOT_CLOSED_LOOP_SPECIFIC"
    payload = signed(
        {
            "format_version": "sage-t10.3.12f-adjudication-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "verdict": verdict,
            "candidate_arm": candidate_arm,
            "control_arm": control_arm,
            "success_games": success_games,
            "game_mean_utility": utility,
            "game_mean_prequential_log_loss": log_loss,
            "identification_advantage": identification,
            "source_contrasts_holm": source_tests,
            "generic_vs_open_loop": generic_test,
            "safety_checks": safety,
            "statistical_unit": "game_mean_over_four_work_scopes",
            "historical_candidate_only": True,
            "confirmatory_evidence": False,
            "prospective_generalization_proven": False,
            "program_promoted": False,
            "t10_3_13_authorized": False,
            "holdout_opened": False,
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, ADJUDICATION_FILENAME, payload)
    return payload


def _artifact_checksums(root: Path) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        if phase == "report":
            output[phase] = None
            continue
        path = _path(root, str(contract["path"]))
        if not path.is_file():
            output[phase] = None
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        field = str(contract["checksum_field"])
        protocol.verify_signed(payload, field)
        output[phase] = str(payload[field])
    failure_path = _path(root, PRIOR_FAILURE_FILENAME)
    if failure_path.is_file():
        failure = _read_signed(root, PRIOR_FAILURE_FILENAME, "report_checksum")
        output["compile-prior-failure"] = str(failure["report_checksum"])
    return output


def _negative_verdict(root: Path) -> str:
    adjudication_path = _path(root, ADJUDICATION_FILENAME)
    if adjudication_path.is_file():
        return str(
            _read_signed(root, ADJUDICATION_FILENAME, "report_checksum").get(
                "verdict", "INCOMPLETE_T10_3_12F"
            )
        )
    active_path = _path(root, ACTIVE_REPORT_FILENAME)
    if active_path.is_file():
        return "INCOMPLETE_T10_3_12F"
    preflight_path = _path(root, PREFLIGHT_FILENAME)
    if preflight_path.is_file():
        preflight_payload = _read_signed(
            root, PREFLIGHT_FILENAME, "preflight_checksum"
        )
        return (
            "INCOMPLETE_T10_3_12F"
            if preflight_payload.get("passed") is True
            else "CAUSAL_PROCEDURE_PREFLIGHT_MISS"
        )
    evaluation_path = _path(root, SOURCE_EVALUATION_FILENAME)
    if evaluation_path.is_file():
        evaluation = _read_signed(
            root, SOURCE_EVALUATION_FILENAME, "report_checksum"
        )
        return (
            "INCOMPLETE_T10_3_12F"
            if evaluation.get("passed") is True
            else str(evaluation.get("verdict", "PROCEDURE_NOT_SOURCE_IDENTIFIABLE"))
        )
    failure_path = _path(root, PRIOR_FAILURE_FILENAME)
    if failure_path.is_file():
        failure = _read_signed(root, PRIOR_FAILURE_FILENAME, "report_checksum")
        return str(failure.get("verdict", "PROCEDURE_NOT_SOURCE_IDENTIFIABLE"))
    qa_path = _path(root, SOURCE_QA_FILENAME)
    if qa_path.is_file():
        qa = _read_signed(root, SOURCE_QA_FILENAME, "report_checksum")
        return (
            "INCOMPLETE_T10_3_12F"
            if qa.get("passed") is True
            else str(qa.get("verdict", "CAUSAL_LABEL_QA_MISS"))
        )
    return "INCOMPLETE_T10_3_12F"


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    adjudication_path = _path(root, ADJUDICATION_FILENAME)
    adjudication = (
        _read_signed(root, ADJUDICATION_FILENAME, "report_checksum")
        if adjudication_path.is_file()
        else None
    )
    passed = bool(adjudication and adjudication.get("passed") is True)
    verdict = (
        str(adjudication["verdict"])
        if adjudication is not None
        else _negative_verdict(root)
    )
    with _durable_contract():
        accounting = durable._journal_accounting(_destination(root))
    receipts = _load_receipts(_destination(root))
    physical_replays = sum(
        int(row.get("physical_actions_replayed", 0)) for row in receipts
    )
    legacy_fallbacks = sum(
        int(row.get("legacy_fallback_actions", 0)) for row in receipts
    )
    accounting_clean = bool(
        accounting.get("equation_holds")
        and accounting.get("inflight_valid")
        and int(accounting.get("inflight_intents", 0)) == 0
        and int(accounting.get("unresolved_intents", 0)) == 0
        and not accounting.get("incomplete_work_ids")
        and not accounting.get("live_collector_lock")
    )
    if not accounting_clean or physical_replays or legacy_fallbacks:
        raise protocol.IntegrityError(
            "T10.3.12f terminal accounting or execution integrity is not clean"
        )
    for field in (
        "holdout_opened",
        "ar25_opened",
        "source_validation_opened",
        "sequence_games_opened",
        "production_authority",
        "t10_3_13_authorized",
    ):
        if manifest["firewall"].get(field) is not False:
            raise protocol.IntegrityError(f"T10.3.12f firewall drifted: {field}")
    artifacts = _artifact_checksums(root)
    payload = signed(
        {
            "format_version": "sage-t10.3.12f-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "passed": passed,
            "candidate_arm": (
                None if adjudication is None else adjudication.get("candidate_arm")
            ),
            "control_arm": (
                None if adjudication is None else adjudication.get("control_arm")
            ),
            "artifacts": artifacts,
            "accounting": accounting,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
            "historical_candidate_only": True,
            "confirmatory_evidence": False,
            "prospective_generalization_proven": False,
            "parent_events_used_for_training": 0,
            "target_history_events_used_for_initialization": 0,
            "physical_actions_replayed": physical_replays,
            "legacy_fallback_actions": legacy_fallbacks,
            "program_promoted": False,
            "t10_3_13_authorized": False,
            "holdout_opened": False,
            "ar25_opened": False,
            "source_validation_opened": False,
            "sequence_games_opened": False,
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, TERMINAL_REPORT_FILENAME, payload)
    return payload


def status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / protocol.DEFAULT_MANIFEST_PATH
    manifest_checksum: str | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol.verify_signed(manifest, "manifest_checksum")
        manifest_checksum = str(manifest["manifest_checksum"])
    with _durable_contract():
        accounting = durable._journal_accounting(_destination(root))
    artifacts = _artifact_checksums(root)
    terminal_path = _path(root, TERMINAL_REPORT_FILENAME)
    terminal = (
        _read_signed(root, TERMINAL_REPORT_FILENAME, "report_checksum")
        if terminal_path.is_file()
        else None
    )
    return {
        "format_version": "sage-t10.3.12f-status-v1",
        "manifest_frozen": manifest_checksum is not None,
        "manifest_checksum": manifest_checksum,
        "artifacts": artifacts,
        "accounting": accounting,
        "verdict": None if terminal is None else terminal.get("verdict"),
        "passed": False if terminal is None else bool(terminal.get("passed")),
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        "maximum_resets": protocol.TOTAL_RESETS,
        "historical_diagnostic_only": True,
        "holdout_opened": False,
        "t10_3_13_authorized": False,
        "ar25_opened": False,
        "source_validation_opened": False,
        "sequence_games_opened": False,
        "production_authority": False,
    }


def _emit(payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze",
            "status",
            "audit",
            "qa-source",
            "compile-prior",
            "evaluate-source",
            "preflight",
            "active-historical",
            "adjudicate",
            "report",
        ),
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    phase = str(args.phase)
    code = 0
    try:
        if phase == "status":
            result = status(root)
        elif phase == "freeze":
            manifest, receipt = protocol.freeze_manifest(root)
            result = {
                "format_version": "sage-t10.3.12f-freeze-result-v1",
                "manifest_checksum": manifest["manifest_checksum"],
                "receipt_checksum": receipt["receipt_checksum"],
                "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
                "maximum_resets": protocol.TOTAL_RESETS,
                "holdout_opened": False,
                "t10_3_13_authorized": False,
                "production_authority": False,
            }
        else:
            manifest = protocol.load_manifest(root)
            handlers = {
                "audit": audit,
                "qa-source": qa_source,
                "compile-prior": compile_prior,
                "evaluate-source": evaluate_source,
                "preflight": preflight,
                "active-historical": active_historical,
                "adjudicate": adjudicate,
                "report": terminal_report,
            }
            result = handlers[phase](root, manifest)
            if phase in {"adjudicate", "report"} and result.get("passed") is not True:
                code = 3
    except protocol.ScientificGateMiss as exc:
        result = {
            "error": str(exc),
            "exit_code": 3,
            "phase": phase,
        }
        code = 3
    except (protocol.IntegrityError, OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "error": f"{type(exc).__name__}:{exc}",
            "exit_code": 2,
            "phase": phase,
        }
        code = 2
    _emit(result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
