"""P3.G3 out-of-sample contextual post-stop policy validation.

P3.G2 learned a candidate-only selector from the M3.G4 safe-stop scope map.
This module freezes that adapter, samples fresh safe-stop endpoints outside the
M3.G4/P3.G2 substrate, and executes the selected option plus static baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from theory.m3.objective_conversion_diverse_safe_stop_validation import (
    captured_safe_stop_from_g3_record,
    hold_baseline_band,
    per_safe_stop_validation_record,
    terminal_horizon_band,
    terminal_horizon_remaining,
)
from theory.m3.objective_conversion_experiment_executor import (
    CapturedSafeStop,
    ObjectiveConversionCell,
    execute_objective_conversion_cell,
    execution_cell_signature,
)
from theory.m3.objective_conversion_safe_stop_diversity_sampler import (
    SafeStopCandidatePlan,
    execute_safe_stop_candidate_plan,
    prefix_hash_for_steps,
    relation_state_signature,
)
from theory.p3.abstract_mechanic_policy_probe import TRUTH_STATUS
from theory.p3.contextual_post_stop_conversion_policy_probe import (
    ACTION6_ACTION3,
    ACTION6_ACTION4,
    ACTION6_ONLY,
    CONTEXTUAL_POLICY,
    DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH,
    DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH,
    DEFAULT_M3G4_DIVERSE_VALIDATION_PATH,
    HOLD_OR_STOP_STATE,
    aggregate_option_records,
    build_baseline_policy_records,
    option_from_candidate,
    scope_key,
    sequence_key,
    validate_adapter_source,
    validate_m3g4_source,
)


DEFAULT_OUT_OF_SAMPLE_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH = (
    Path("diagnostics")
    / "p3"
    / "out_of_sample_contextual_post_stop_policy_validation.json"
)
P3G3_SCHEMA_VERSION = (
    "p3.out_of_sample_contextual_post_stop_policy_validation.v1"
)
DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_MIN_OUT_OF_SAMPLE_SAFE_STOPS = 4
DEFAULT_MAX_CANDIDATE_PLANS = 28

OUT_OF_SAMPLE_SAFE_STOP_ACCEPTED = "OUT_OF_SAMPLE_SAFE_STOP_ACCEPTED"
IN_SAMPLE_SAFE_STOP_REJECTED = "IN_SAMPLE_SAFE_STOP_REJECTED"
DUPLICATE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED = (
    "DUPLICATE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED"
)
UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED = "UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED"
REPLAY_INEXACT_OUT_OF_SAMPLE_SAFE_STOP_REJECTED = (
    "REPLAY_INEXACT_OUT_OF_SAMPLE_SAFE_STOP_REJECTED"
)
BLOCKED_OUT_OF_SAMPLE_SAFE_STOP_REJECTED = "BLOCKED_OUT_OF_SAMPLE_SAFE_STOP_REJECTED"

POST_STOP_CONTEXTUAL_POLICY_GENERALIZES_CANDIDATE_ONLY = (
    "POST_STOP_CONTEXTUAL_POLICY_GENERALIZES_CANDIDATE_ONLY"
)
OUT_OF_SAMPLE_POLICY_UTILITY_CANDIDATE_ONLY = (
    "OUT_OF_SAMPLE_POLICY_UTILITY_CANDIDATE_ONLY"
)
IN_SAMPLE_CONTEXTUAL_POLICY_ONLY_CANDIDATE_ONLY = (
    "IN_SAMPLE_CONTEXTUAL_POLICY_ONLY_CANDIDATE_ONLY"
)
OUT_OF_SAMPLE_POLICY_RISKY_CANDIDATE_ONLY = (
    "OUT_OF_SAMPLE_POLICY_RISKY_CANDIDATE_ONLY"
)
OUT_OF_SAMPLE_POLICY_HARMFUL_CANDIDATE_ONLY = (
    "OUT_OF_SAMPLE_POLICY_HARMFUL_CANDIDATE_ONLY"
)
OUT_OF_SAMPLE_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY = (
    "OUT_OF_SAMPLE_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY"
)


def run_out_of_sample_contextual_post_stop_policy_validation(
    *,
    adapter_path: str | Path = DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH,
    source_m3g4_path: str | Path = DEFAULT_M3G4_DIVERSE_VALIDATION_PATH,
    source_p3g2_probe_path: str | Path = DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    min_out_of_sample_safe_stops: int = DEFAULT_MIN_OUT_OF_SAMPLE_SAFE_STOPS,
    max_candidate_plans: int | None = DEFAULT_MAX_CANDIDATE_PLANS,
    candidate_plans: Sequence[SafeStopCandidatePlan | Mapping[str, Any]]
    | None = None,
    candidate_executor: Callable[[SafeStopCandidatePlan], Mapping[str, Any]]
    | None = None,
    captured_builder: Callable[[Mapping[str, Any]], CapturedSafeStop] | None = None,
    cell_executor: Callable[
        [ObjectiveConversionCell, CapturedSafeStop], Mapping[str, Any]
    ]
    | None = None,
) -> Dict[str, Any]:
    adapter_payload = _load_json(adapter_path)
    validate_adapter_source(adapter_payload)
    adapter = dict(adapter_payload.get("contextual_post_stop_policy_adapter", {}) or {})

    source_m3g4_payload = _load_json(source_m3g4_path)
    validate_m3g4_source(source_m3g4_payload)
    source_p3g2_payload = _load_json(source_p3g2_probe_path)
    validate_p3g2_probe_source(source_p3g2_payload)
    in_sample_identity = in_sample_safe_stop_identity(source_m3g4_payload)

    plans = normalize_candidate_plans(
        candidate_plans
        if candidate_plans is not None
        else generate_out_of_sample_candidate_plans()
    )
    if max_candidate_plans is not None:
        plans = plans[: max(0, int(max_candidate_plans))]

    if candidate_executor is None:

        def default_candidate_executor(
            plan: SafeStopCandidatePlan,
        ) -> Mapping[str, Any]:
            return execute_safe_stop_candidate_plan(
                plan,
                environments_dir=environments_dir,
                game_id=game_id,
            )

        candidate_executor = default_candidate_executor
    if captured_builder is None:

        def default_captured_builder(record: Mapping[str, Any]) -> CapturedSafeStop:
            return captured_safe_stop_from_g3_record(
                record,
                environments_dir=environments_dir,
            )

        captured_builder = default_captured_builder
    if cell_executor is None:

        def default_cell_executor(
            cell: ObjectiveConversionCell,
            captured: CapturedSafeStop,
        ) -> Mapping[str, Any]:
            return execute_objective_conversion_cell(
                cell,
                captured=captured,
                environments_dir=environments_dir,
            )

        cell_executor = default_cell_executor

    raw_candidate_records = [dict(candidate_executor(plan)) for plan in plans]
    candidate_records, accepted_safe_stops = classify_out_of_sample_safe_stops(
        raw_candidate_records,
        in_sample_identity=in_sample_identity,
        min_out_of_sample_safe_stops=int(min_out_of_sample_safe_stops),
    )
    cells = static_policy_cells(game_id=game_id)

    execution_cells: List[Dict[str, Any]] = []
    for safe_stop in accepted_safe_stops:
        captured = captured_builder(safe_stop)
        for cell in cells:
            row = dict(cell_executor(cell, captured))
            row.update(safe_stop_cell_context(safe_stop))
            row.update(
                {
                    "cell_result_counted_as_confirmation": False,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": TRUTH_STATUS,
                    "wrong_confirmations": 0,
                }
            )
            execution_cells.append(row)

    per_safe_stop_records = [
        per_safe_stop_validation_record(
            safe_stop_record=safe_stop,
            cells=[
                cell
                for cell in execution_cells
                if str(cell.get("safe_stop_id", "")) == str(safe_stop.get("safe_stop_id", ""))
            ],
        )
        for safe_stop in accepted_safe_stops
    ]
    policy_records = [
        select_frozen_contextual_policy_record(record, adapter=adapter)
        for record in per_safe_stop_records
    ]
    baseline_records = build_baseline_policy_records(per_safe_stop_records)
    baseline_aggregates = {
        condition: aggregate_option_records(condition, records)
        for condition, records in sorted(baseline_records.items())
    }
    contextual_aggregate = aggregate_option_records(CONTEXTUAL_POLICY, policy_records)
    validation_status = out_of_sample_policy_status(
        accepted_safe_stops=accepted_safe_stops,
        contextual_aggregate=contextual_aggregate,
        baseline_aggregates=baseline_aggregates,
        unsafe_extension_options_avoided=sum(
            int(record.get("unsafe_extension_options_avoided", 0) or 0)
            for record in policy_records
        ),
        min_out_of_sample_safe_stops=int(min_out_of_sample_safe_stops),
    )

    summary = summarize_out_of_sample_validation(
        plans=plans,
        candidate_records=candidate_records,
        accepted_safe_stops=accepted_safe_stops,
        execution_cells=execution_cells,
        policy_records=policy_records,
        baseline_aggregates=baseline_aggregates,
        contextual_aggregate=contextual_aggregate,
        validation_status=validation_status,
        source_p3g2_payload=source_p3g2_payload,
        min_out_of_sample_safe_stops=int(min_out_of_sample_safe_stops),
    )
    return {
        "config": {
            "schema_version": P3G3_SCHEMA_VERSION,
            "stage": "P3.G3",
            "adapter_path": str(adapter_path),
            "source_m3g4_path": str(source_m3g4_path),
            "source_p3g2_probe_path": str(source_p3g2_probe_path),
            "environments_dir": (
                None if environments_dir is None else str(environments_dir)
            ),
            "game_id": game_id,
            "adapter_relearned": False,
            "source_cells_rerun": True,
            "execution_performed": True,
            "policy_options": [
                HOLD_OR_STOP_STATE,
                ACTION6_ONLY,
                ACTION6_ACTION3,
                ACTION6_ACTION4,
            ],
            "selection_uses_out_of_sample_candidate_outcomes": False,
        },
        "source_p3g2_summary": source_p3g2_summary(source_p3g2_payload),
        "out_of_sample_candidate_plans": [plan.to_dict() for plan in plans],
        "out_of_sample_candidate_records": candidate_records,
        "accepted_out_of_sample_safe_stops": accepted_safe_stops,
        "execution_cells": execution_cells,
        "per_safe_stop_validation_records": per_safe_stop_records,
        "policy_decision_records": policy_records,
        "baseline_aggregates": baseline_aggregates,
        "contextual_policy_aggregate": contextual_aggregate,
        "summary": summary,
        "policy_utility_status": validation_status,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "adapter_counted_as_mechanic_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def generate_out_of_sample_candidate_plans() -> Tuple[SafeStopCandidatePlan, ...]:
    plans: List[SafeStopCandidatePlan] = []
    fixed_patterns = (
        (
            "late_action6_after_relation",
            ("ACTION3", "ACTION4", "ACTION3", "ACTION4", "ACTION3", "ACTION6"),
        ),
        (
            "late_action6_after_phase_shift",
            ("ACTION4", "ACTION3", "ACTION4", "ACTION3", "ACTION4", "ACTION6"),
        ),
        ("mixed_relation_action6", ("ACTION3", "ACTION6", "ACTION4", "ACTION3")),
        ("mixed_phase_action6", ("ACTION4", "ACTION6", "ACTION3", "ACTION4")),
        ("double_action6", ("ACTION6", "ACTION6")),
        ("paired_relation_blocks_a3", ("ACTION3", "ACTION3", "ACTION4", "ACTION4")),
        ("paired_relation_blocks_a4", ("ACTION4", "ACTION4", "ACTION3", "ACTION3")),
    )
    for index, (name, actions) in enumerate(fixed_patterns, start=1):
        plans.append(
            safe_stop_plan(
                plan_id=f"p3_g3::{name}",
                actions=actions,
                sampling_family=name,
                tie_break_seed=index,
            )
        )

    for length in (1, 3, 5, 7, 9, 11, 13, 15):
        actions = tuple(["ACTION3", "ACTION4"] * (length // 2) + ["ACTION6"])
        plans.append(
            safe_stop_plan(
                plan_id=f"p3_g3::relation_prefix_action6_tail::{length:02d}",
                actions=actions,
                sampling_family="relation_prefix_action6_tail",
                tie_break_seed=100 + length,
            )
        )

    for length in (2, 4, 6, 8):
        actions = tuple(["ACTION6", "ACTION3"] * length)
        plans.append(
            safe_stop_plan(
                plan_id=f"p3_g3::action6_relation_weave::{length:02d}",
                actions=actions,
                sampling_family="action6_relation_weave",
                tie_break_seed=200 + length,
            )
        )

    return tuple(deduplicate_plans(plans))


def safe_stop_plan(
    *,
    plan_id: str,
    actions: Sequence[str],
    sampling_family: str,
    tie_break_seed: int,
) -> SafeStopCandidatePlan:
    return SafeStopCandidatePlan(
        plan_id=plan_id,
        planned_prefix=tuple({"action": action, "action_args": {}} for action in actions),
        sampling_family=sampling_family,
        anti_attractor_rationale=(
            "Out-of-sample endpoint not used by M3.G4/P3.G2."
        ),
        tie_break_seed=int(tie_break_seed),
    )


def normalize_candidate_plans(
    plans: Sequence[SafeStopCandidatePlan | Mapping[str, Any]],
) -> Tuple[SafeStopCandidatePlan, ...]:
    normalized: List[SafeStopCandidatePlan] = []
    for index, plan in enumerate(plans, start=1):
        if isinstance(plan, SafeStopCandidatePlan):
            normalized.append(plan)
            continue
        normalized.append(
            SafeStopCandidatePlan(
                plan_id=str(plan.get("plan_id", f"p3_g3::plan::{index:03d}")),
                planned_prefix=tuple(
                    dict(step) for step in plan.get("planned_prefix", []) or []
                ),
                sampling_family=str(plan.get("sampling_family", "external_oos")),
                anti_attractor_rationale=str(
                    plan.get("anti_attractor_rationale", "")
                ),
                tie_break_seed=int(plan.get("tie_break_seed", index) or index),
            )
        )
    return tuple(deduplicate_plans(normalized))


def deduplicate_plans(
    plans: Sequence[SafeStopCandidatePlan],
) -> Tuple[SafeStopCandidatePlan, ...]:
    seen: set[str] = set()
    unique: List[SafeStopCandidatePlan] = []
    for plan in plans:
        if not plan.planned_prefix:
            continue
        key = json.dumps(list(plan.planned_prefix), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(plan)
    return tuple(unique)


def classify_out_of_sample_safe_stops(
    records: Sequence[Mapping[str, Any]],
    *,
    in_sample_identity: Mapping[str, set[str]],
    min_out_of_sample_safe_stops: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen_state_hashes: set[str] = set()
    seen_prefix_hashes: set[str] = set()
    seen_relation_signatures: set[str] = set()
    classified: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    for record in records:
        row = {
            **dict(record),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "wrong_confirmations": 0,
            "safe_stop_candidate_counted_as_confirmation": False,
        }
        state_hash = str(row.get("safe_stop_state_hash", ""))
        prefix_hash = str(row.get("captured_prefix_hash", ""))
        relation_signature = str(row.get("relation_state_signature", ""))
        status = OUT_OF_SAMPLE_SAFE_STOP_ACCEPTED
        reason = ""
        if not bool(row.get("execution_performed", False)):
            status = BLOCKED_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = str(row.get("blocked_reason", "blocked_candidate"))
        elif not bool(row.get("replay_exact", False)):
            status = REPLAY_INEXACT_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = "candidate_prefix_not_replay_exact"
        elif not bool(row.get("non_terminal", False)) or not bool(
            row.get("terminal_safe", False)
        ):
            status = UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = "terminal_or_terminal_unsafe_endpoint"
        elif not bool(row.get("hold_baseline_measurable", False)):
            status = UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = "hold_baseline_unmeasurable"
        elif state_hash in in_sample_identity["state_hashes"] or prefix_hash in in_sample_identity[
            "prefix_hashes"
        ]:
            status = IN_SAMPLE_SAFE_STOP_REJECTED
            reason = "state_or_prefix_seen_in_m3_g4_p3_g2"
        elif (
            state_hash in seen_state_hashes
            or prefix_hash in seen_prefix_hashes
            or relation_signature in seen_relation_signatures
        ):
            status = DUPLICATE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = "state_prefix_or_relation_signature_not_novel_oos"

        row["out_of_sample_acceptance_status"] = status
        row["rejection_reason"] = reason
        row["candidate_needed_for_p3_g3"] = (
            len(accepted) < int(min_out_of_sample_safe_stops)
            and status == OUT_OF_SAMPLE_SAFE_STOP_ACCEPTED
        )
        classified.append(row)
        if status == OUT_OF_SAMPLE_SAFE_STOP_ACCEPTED:
            seen_state_hashes.add(state_hash)
            seen_prefix_hashes.add(prefix_hash)
            seen_relation_signatures.add(relation_signature)
            accepted.append(out_of_sample_safe_stop_public_record(row, len(accepted) + 1))
    return classified, accepted


def out_of_sample_safe_stop_public_record(
    row: Mapping[str, Any],
    index: int,
) -> Dict[str, Any]:
    return {
        "safe_stop_id": f"p3_g3::oos_safe_stop::{index:03d}",
        "source_plan_id": str(row.get("plan_id", "")),
        "sampling_family": str(row.get("sampling_family", "")),
        "captured_prefix": [dict(step) for step in row.get("captured_prefix", []) or []],
        "captured_prefix_len": int(row.get("captured_prefix_len", 0) or 0),
        "captured_prefix_hash": str(row.get("captured_prefix_hash", "")),
        "safe_stop_state_hash": str(row.get("safe_stop_state_hash", "")),
        "relation_state_signature": str(row.get("relation_state_signature", "")),
        "terminal_horizon_estimate": dict(row.get("terminal_horizon_estimate", {}) or {}),
        "hold_baseline_terminal_adjusted_progress": float(
            row.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
        ),
        "hold_baseline_levels_completed": int(
            row.get("hold_baseline_levels_completed", 0) or 0
        ),
        "hold_baseline_terminal": bool(row.get("hold_baseline_terminal", False)),
        "replay_exact": bool(row.get("replay_exact", False)),
        "non_terminal": bool(row.get("non_terminal", False)),
        "terminal_safe": bool(row.get("terminal_safe", False)),
        "hold_baseline_measurable": bool(row.get("hold_baseline_measurable", False)),
        "out_of_sample_from_m3_g4_p3_g2": True,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "wrong_confirmations": 0,
        "safe_stop_record_counted_as_confirmation": False,
    }


def static_policy_cells(*, game_id: str) -> Tuple[ObjectiveConversionCell, ...]:
    return (
        ObjectiveConversionCell(
            cell_signature=execution_cell_signature(
                game_id=game_id,
                condition_kind="hold",
                action_or_sequence=None,
                post_stop_horizon=0,
            ),
            game_id=game_id,
            condition_kind="hold",
            condition_id=HOLD_OR_STOP_STATE,
            action_or_sequence=None,
            post_stop_horizon=0,
        ),
        candidate_cell(game_id=game_id, sequence=("ACTION6",)),
        candidate_cell(game_id=game_id, sequence=("ACTION6", "ACTION3")),
        candidate_cell(game_id=game_id, sequence=("ACTION6", "ACTION4")),
    )


def candidate_cell(*, game_id: str, sequence: Sequence[str]) -> ObjectiveConversionCell:
    return ObjectiveConversionCell(
        cell_signature=execution_cell_signature(
            game_id=game_id,
            condition_kind="candidate",
            action_or_sequence=list(sequence),
            post_stop_horizon=len(sequence),
        ),
        game_id=game_id,
        condition_kind="candidate",
        condition_id="candidate_" + "_".join(sequence),
        action_or_sequence=tuple(sequence),
        post_stop_horizon=len(sequence),
    )


def select_frozen_contextual_policy_record(
    safe_stop: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_sequence, reason, blocked_sequences = frozen_contextual_sequence(
        safe_stop,
        adapter=adapter,
    )
    candidate_by_sequence = {
        sequence_key(record.get("action_or_sequence", [])): dict(record)
        for record in safe_stop.get("candidate_records", []) or []
        if isinstance(record, Mapping)
    }
    selected_candidate = candidate_by_sequence.get(selected_sequence)
    if selected_candidate is None and selected_sequence != ACTION6_ONLY:
        selected_sequence = ACTION6_ONLY
        reason = "selected_extension_unavailable_fallback_action6"
        selected_candidate = candidate_by_sequence.get(selected_sequence)
    if selected_candidate is None:
        selected = {
            "selected_option": HOLD_OR_STOP_STATE,
            "selected_action_or_sequence": [],
            "selected_reason": "selected_action6_unavailable_fallback_hold",
            "terminal_adjusted_progress": float(
                safe_stop.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
            ),
            "terminal_reentry": False,
            "objective_completion_signal": False,
            "levels_completed_after_rollout": 0,
            "delta_vs_hold": 0.0,
        }
    else:
        selected = option_from_candidate(selected_candidate, selected_reason=reason)

    unsafe_extensions = [
        record
        for sequence, record in candidate_by_sequence.items()
        if sequence in {ACTION6_ACTION3, ACTION6_ACTION4}
        and sequence != str(selected.get("selected_option", ""))
        and bool(record.get("candidate_terminal_reentry", False))
    ]
    action6 = candidate_by_sequence.get(ACTION6_ONLY, {})
    action6_taps = float(
        action6.get(
            "candidate_terminal_adjusted_progress_after_stop",
            safe_stop.get("hold_baseline_terminal_adjusted_progress", 0.0),
        )
        or 0.0
    )
    selected_taps = float(selected.get("terminal_adjusted_progress", 0.0) or 0.0)
    return {
        "safe_stop_id": str(safe_stop.get("safe_stop_id", "")),
        "sampling_family": str(safe_stop.get("sampling_family", "")),
        "terminal_horizon_remaining": safe_stop.get("terminal_horizon_remaining"),
        "terminal_horizon_band": str(safe_stop.get("terminal_horizon_band", "")),
        "hold_baseline_terminal_adjusted_progress": float(
            safe_stop.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
        ),
        "hold_baseline_band": str(safe_stop.get("hold_baseline_band", "")),
        "selection_uses_out_of_sample_candidate_outcomes": False,
        "frozen_adapter_scope_blockers": frozen_scope_blockers(
            safe_stop,
            adapter=adapter,
        ),
        "blocked_extension_sequences_by_rule": blocked_sequences,
        **selected,
        "delta_vs_action6_only": round(selected_taps - action6_taps, 6),
        "unsafe_extension_options_avoided": len(unsafe_extensions),
        "policy_decision_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def frozen_contextual_sequence(
    safe_stop: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    blockers = frozen_scope_blockers(safe_stop, adapter=adapter)
    blocked_sequences = [
        {
            "sequence_key": sequence,
            "blockers": list(blockers),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        }
        for sequence in adapter.get("extension_sequence_preference", []) or []
        if blockers
    ]
    if blockers:
        return ACTION6_ONLY, "frozen_scope_gate_fallback_action6", blocked_sequences
    preference = [str(value) for value in adapter.get("extension_sequence_preference", []) or []]
    selected = preference[0] if preference else ACTION6_ONLY
    return selected, "frozen_scope_gate_allows_extension", blocked_sequences


def frozen_scope_blockers(
    safe_stop: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
) -> List[str]:
    risk_model = dict(adapter.get("risk_model", {}) or {})
    pair_key = scope_key(
        str(safe_stop.get("sampling_family", "")),
        str(safe_stop.get("terminal_horizon_band", "")),
    )
    hold_band = str(safe_stop.get("hold_baseline_band", ""))
    blockers: List[str] = []
    if pair_key in set(risk_model.get("blocked_sampling_family_horizon_pairs", []) or []):
        blockers.append("blocked_sampling_family_horizon_pair")
    if hold_band in set(risk_model.get("blocked_hold_baseline_bands", []) or []):
        blockers.append("blocked_hold_baseline_band")
    return blockers


def safe_stop_cell_context(record: Mapping[str, Any]) -> Dict[str, Any]:
    hold_value = float(record.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0)
    return {
        "safe_stop_id": str(record.get("safe_stop_id", "")),
        "source_plan_id": str(record.get("source_plan_id", "")),
        "sampling_family": str(record.get("sampling_family", "")),
        "terminal_horizon_remaining": terminal_horizon_remaining(record),
        "terminal_horizon_band": terminal_horizon_band(record),
        "hold_baseline_band": hold_baseline_band(hold_value),
    }


def out_of_sample_policy_status(
    *,
    accepted_safe_stops: Sequence[Mapping[str, Any]],
    contextual_aggregate: Mapping[str, Any],
    baseline_aggregates: Mapping[str, Mapping[str, Any]],
    unsafe_extension_options_avoided: int,
    min_out_of_sample_safe_stops: int,
) -> str:
    if int(contextual_aggregate.get("objective_completion_runs", 0) or 0) > 0:
        return OUT_OF_SAMPLE_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY
    if len(accepted_safe_stops) < int(min_out_of_sample_safe_stops):
        return IN_SAMPLE_CONTEXTUAL_POLICY_ONLY_CANDIDATE_ONLY
    contextual_terminal = float(contextual_aggregate.get("terminal_rate", 0.0) or 0.0)
    if contextual_terminal > 0.0:
        return OUT_OF_SAMPLE_POLICY_RISKY_CANDIDATE_ONLY
    contextual_taps = float(
        contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    action6_aggregate = baseline_aggregates.get(ACTION6_ONLY, {})
    action6_taps = float(
        action6_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    action6_terminal = float(action6_aggregate.get("terminal_rate", 0.0) or 0.0)
    static_extension_taps = [
        float(
            baseline_aggregates.get(sequence, {}).get(
                "mean_terminal_adjusted_progress", 0.0
            )
            or 0.0
        )
        for sequence in (ACTION6_ACTION3, ACTION6_ACTION4)
    ]
    best_static_extension_taps = max(static_extension_taps or [0.0])
    if (
        contextual_taps > action6_taps
        and contextual_terminal <= action6_terminal
        and (
            int(unsafe_extension_options_avoided) > 0
            or contextual_taps >= best_static_extension_taps
        )
    ):
        return POST_STOP_CONTEXTUAL_POLICY_GENERALIZES_CANDIDATE_ONLY
    if contextual_taps >= action6_taps and contextual_terminal <= action6_terminal:
        return OUT_OF_SAMPLE_POLICY_UTILITY_CANDIDATE_ONLY
    return OUT_OF_SAMPLE_POLICY_HARMFUL_CANDIDATE_ONLY


def summarize_out_of_sample_validation(
    *,
    plans: Sequence[SafeStopCandidatePlan],
    candidate_records: Sequence[Mapping[str, Any]],
    accepted_safe_stops: Sequence[Mapping[str, Any]],
    execution_cells: Sequence[Mapping[str, Any]],
    policy_records: Sequence[Mapping[str, Any]],
    baseline_aggregates: Mapping[str, Mapping[str, Any]],
    contextual_aggregate: Mapping[str, Any],
    validation_status: str,
    source_p3g2_payload: Mapping[str, Any],
    min_out_of_sample_safe_stops: int,
) -> Dict[str, Any]:
    action6_mean = float(
        baseline_aggregates.get(ACTION6_ONLY, {}).get(
            "mean_terminal_adjusted_progress", 0.0
        )
        or 0.0
    )
    contextual_mean = float(
        contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    unsafe_avoided = sum(
        int(record.get("unsafe_extension_options_avoided", 0) or 0)
        for record in policy_records
    )
    return {
        "policy_utility_status": validation_status,
        "adapter_relearned": False,
        "source_cells_rerun": True,
        "execution_performed": True,
        "candidate_plans": len(plans),
        "candidate_plans_executed": len(
            [record for record in candidate_records if bool(record.get("execution_performed", False))]
        ),
        "accepted_out_of_sample_safe_stops": len(accepted_safe_stops),
        "min_out_of_sample_safe_stops_required": int(min_out_of_sample_safe_stops),
        "in_sample_safe_stops_rejected": len(
            [
                record
                for record in candidate_records
                if str(record.get("out_of_sample_acceptance_status", ""))
                == IN_SAMPLE_SAFE_STOP_REJECTED
            ]
        ),
        "duplicate_out_of_sample_safe_stops_rejected": len(
            [
                record
                for record in candidate_records
                if str(record.get("out_of_sample_acceptance_status", ""))
                == DUPLICATE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            ]
        ),
        "unsafe_or_terminal_safe_stops_rejected": len(
            [
                record
                for record in candidate_records
                if str(record.get("out_of_sample_acceptance_status", ""))
                == UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            ]
        ),
        "cells_executed": len(
            [cell for cell in execution_cells if bool(cell.get("execution_performed", False))]
        ),
        "selected_hold_count": sum(
            1 for record in policy_records if record.get("selected_option") == HOLD_OR_STOP_STATE
        ),
        "selected_action6_only_count": sum(
            1 for record in policy_records if record.get("selected_option") == ACTION6_ONLY
        ),
        "selected_extension_count": sum(
            1
            for record in policy_records
            if str(record.get("selected_option", "")).startswith("ACTION6,ACTION")
        ),
        "terminal_rate": contextual_aggregate.get("terminal_rate", 0.0),
        "mean_terminal_adjusted_progress": contextual_aggregate.get(
            "mean_terminal_adjusted_progress", 0.0
        ),
        "mean_delta_vs_hold": contextual_aggregate.get("mean_delta_vs_hold", 0.0),
        "mean_delta_vs_action6_only": round(contextual_mean - action6_mean, 6),
        "improvement_over_action6_only": contextual_mean > action6_mean,
        "unsafe_extension_options_avoided": unsafe_avoided,
        "objective_completion_signal": bool(
            int(contextual_aggregate.get("objective_completion_runs", 0) or 0) > 0
        ),
        "objective_completion_runs": int(
            contextual_aggregate.get("objective_completion_runs", 0) or 0
        ),
        "baseline_aggregates": dict(baseline_aggregates),
        "contextual_policy_aggregate": dict(contextual_aggregate),
        "source_p3g2_policy_utility_status": str(
            source_p3g2_payload.get("policy_utility_status", "")
            or (source_p3g2_payload.get("summary", {}) or {}).get(
                "policy_utility_status", ""
            )
        ),
        "source_p3g2_mean_terminal_adjusted_progress": (
            source_p3g2_payload.get("summary", {}) or {}
        ).get("mean_terminal_adjusted_progress"),
        "source_p3g2_terminal_rate": (
            source_p3g2_payload.get("summary", {}) or {}
        ).get("terminal_rate"),
        "selection_uses_out_of_sample_candidate_outcomes": False,
        "policy_result_counted_as_scientific_verdict": False,
        "adapter_counted_as_mechanic_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def in_sample_safe_stop_identity(payload: Mapping[str, Any]) -> Dict[str, set[str]]:
    records = [
        dict(record)
        for record in payload.get("per_safe_stop_validation_records", []) or []
        if isinstance(record, Mapping)
    ]
    return {
        "state_hashes": {
            str(record.get("safe_stop_state_hash", ""))
            for record in records
            if record.get("safe_stop_state_hash")
        },
        "prefix_hashes": {
            str(record.get("captured_prefix_hash", ""))
            for record in records
            if record.get("captured_prefix_hash")
        },
    }


def source_p3g2_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    return {
        "source_policy_utility_status": str(
            payload.get("policy_utility_status", "")
            or summary.get("policy_utility_status", "")
        ),
        "source_safe_stops_evaluated": int(summary.get("safe_stops_evaluated", 0) or 0),
        "source_terminal_rate": summary.get("terminal_rate"),
        "source_mean_terminal_adjusted_progress": summary.get(
            "mean_terminal_adjusted_progress"
        ),
        "source_cells_rerun": bool(
            (payload.get("config", {}) or {}).get("source_cells_rerun", False)
        ),
        "source_execution_performed": bool(
            (payload.get("config", {}) or {}).get("execution_performed", False)
        ),
        "source_counted_as_scientific_verdict": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def validate_p3g2_probe_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G2 source support must remain 0")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("P3.G2 source cannot be a scientific verdict")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("P3.G2 source must not write A32/A33")
    if not payload.get("policy_decision_records"):
        raise ValueError("P3.G3 requires P3.G2 policy decision records")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_out_of_sample_contextual_post_stop_policy_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OUT_OF_SAMPLE_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P3.G3 out-of-sample contextual post-stop policy validation.",
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH)
    parser.add_argument("--source-m3g4", type=Path, default=DEFAULT_M3G4_DIVERSE_VALIDATION_PATH)
    parser.add_argument("--source-p3g2-probe", type=Path, default=DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--min-out-of-sample-safe-stops", type=int, default=DEFAULT_MIN_OUT_OF_SAMPLE_SAFE_STOPS)
    parser.add_argument("--max-candidate-plans", type=int, default=DEFAULT_MAX_CANDIDATE_PLANS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_OF_SAMPLE_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_out_of_sample_contextual_post_stop_policy_validation(
        adapter_path=args.adapter,
        source_m3g4_path=args.source_m3g4,
        source_p3g2_probe_path=args.source_p3g2_probe,
        environments_dir=args.environments_dir,
        game_id=args.game_id,
        min_out_of_sample_safe_stops=args.min_out_of_sample_safe_stops,
        max_candidate_plans=args.max_candidate_plans,
    )
    write_out_of_sample_contextual_post_stop_policy_validation(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "policy_utility_status": payload["summary"]["policy_utility_status"],
                "accepted_out_of_sample_safe_stops": payload["summary"][
                    "accepted_out_of_sample_safe_stops"
                ],
                "mean_terminal_adjusted_progress": payload["summary"][
                    "mean_terminal_adjusted_progress"
                ],
                "terminal_rate": payload["summary"]["terminal_rate"],
                "mean_delta_vs_action6_only": payload["summary"][
                    "mean_delta_vs_action6_only"
                ],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
