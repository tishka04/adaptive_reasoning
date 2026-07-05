"""P3.G4 risk-targeted out-of-sample post-stop policy validation.

P3.G3 showed out-of-sample utility but not risk-aware dominance, because the
static two-action extensions stayed terminal-safe in that slice. P3.G4 searches
fresh OOS safe-stops near the known risky feature region before considering
gate relaxation.
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
)
from theory.m3.objective_conversion_safe_stop_diversity_sampler import (
    SafeStopCandidatePlan,
    execute_safe_stop_candidate_plan,
)
from theory.p3.abstract_mechanic_policy_probe import TRUTH_STATUS
from theory.p3.contextual_post_stop_conversion_policy_probe import (
    ACTION6_ACTION3,
    ACTION6_ACTION4,
    ACTION6_ONLY,
    CONTEXTUAL_POLICY,
    DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH,
    DEFAULT_M3G4_DIVERSE_VALIDATION_PATH,
    HOLD_OR_STOP_STATE,
    aggregate_option_records,
    build_baseline_policy_records,
    validate_adapter_source,
    validate_m3g4_source,
)
from theory.p3.out_of_sample_contextual_post_stop_policy_validation import (
    DEFAULT_OUT_OF_SAMPLE_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
    DUPLICATE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED,
    IN_SAMPLE_SAFE_STOP_REJECTED,
    OUT_OF_SAMPLE_SAFE_STOP_ACCEPTED,
    UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED,
    generate_out_of_sample_candidate_plans,
    in_sample_safe_stop_identity,
    normalize_candidate_plans,
    out_of_sample_safe_stop_public_record,
    safe_stop_cell_context,
    select_frozen_contextual_policy_record,
    source_p3g2_summary,
    static_policy_cells,
    validate_p3g2_probe_source,
)


DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH = (
    Path("diagnostics")
    / "p3"
    / "risk_targeted_contextual_post_stop_policy_validation.json"
)
P3G4_SCHEMA_VERSION = "p3.risk_targeted_contextual_post_stop_policy_validation.v1"
DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "contextual_post_stop_conversion_policy_probe.json"
)
DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_MIN_RISK_TARGETED_SAFE_STOPS = 4
DEFAULT_MAX_CANDIDATE_PLANS = 48

NON_TARGET_RISK_CONTEXT_REJECTED = "NON_TARGET_RISK_CONTEXT_REJECTED"
RISK_TARGETED_SAFE_STOP_ACCEPTED = "RISK_TARGETED_SAFE_STOP_ACCEPTED"

RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY = (
    "RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY"
)
TARGETED_RISK_NOT_REPRODUCED_CANDIDATE_ONLY = (
    "TARGETED_RISK_NOT_REPRODUCED_CANDIDATE_ONLY"
)
RISK_TARGETED_POLICY_RISKY_CANDIDATE_ONLY = (
    "RISK_TARGETED_POLICY_RISKY_CANDIDATE_ONLY"
)
INSUFFICIENT_RISK_TARGETED_OOS_CONTEXTS_CANDIDATE_ONLY = (
    "INSUFFICIENT_RISK_TARGETED_OOS_CONTEXTS_CANDIDATE_ONLY"
)
RISK_TARGETED_POLICY_HARMFUL_CANDIDATE_ONLY = (
    "RISK_TARGETED_POLICY_HARMFUL_CANDIDATE_ONLY"
)
RISK_TARGETED_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY = (
    "RISK_TARGETED_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY"
)


def run_risk_targeted_contextual_post_stop_policy_validation(
    *,
    adapter_path: str | Path = DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH,
    source_m3g4_path: str | Path = DEFAULT_M3G4_DIVERSE_VALIDATION_PATH,
    source_p3g2_probe_path: str | Path = DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH,
    source_p3g3_path: str
    | Path = DEFAULT_OUT_OF_SAMPLE_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    min_risk_targeted_safe_stops: int = DEFAULT_MIN_RISK_TARGETED_SAFE_STOPS,
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
    source_p3g3_payload = _load_json(source_p3g3_path)
    validate_p3g3_source(source_p3g3_payload)
    excluded_identity = combined_seen_identity(source_m3g4_payload, source_p3g3_payload)

    plans = normalize_candidate_plans(
        candidate_plans
        if candidate_plans is not None
        else generate_risk_targeted_candidate_plans()
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
    candidate_records, accepted_safe_stops = classify_risk_targeted_safe_stops(
        raw_candidate_records,
        excluded_identity=excluded_identity,
        min_risk_targeted_safe_stops=int(min_risk_targeted_safe_stops),
    )

    execution_cells: List[Dict[str, Any]] = []
    for safe_stop in accepted_safe_stops:
        captured = captured_builder(safe_stop)
        for cell in static_policy_cells(game_id=game_id):
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
    risk_stats = risk_targeted_extension_risk_stats(
        per_safe_stop_records=per_safe_stop_records,
        policy_records=policy_records,
        baseline_aggregates=baseline_aggregates,
    )
    validation_status = risk_targeted_policy_status(
        accepted_safe_stops=accepted_safe_stops,
        contextual_aggregate=contextual_aggregate,
        baseline_aggregates=baseline_aggregates,
        risk_stats=risk_stats,
        min_risk_targeted_safe_stops=int(min_risk_targeted_safe_stops),
    )

    summary = summarize_risk_targeted_validation(
        plans=plans,
        candidate_records=candidate_records,
        accepted_safe_stops=accepted_safe_stops,
        execution_cells=execution_cells,
        policy_records=policy_records,
        baseline_aggregates=baseline_aggregates,
        contextual_aggregate=contextual_aggregate,
        risk_stats=risk_stats,
        validation_status=validation_status,
        source_p3g2_payload=source_p3g2_payload,
        source_p3g3_payload=source_p3g3_payload,
        min_risk_targeted_safe_stops=int(min_risk_targeted_safe_stops),
    )
    return {
        "config": {
            "schema_version": P3G4_SCHEMA_VERSION,
            "stage": "P3.G4",
            "adapter_path": str(adapter_path),
            "source_m3g4_path": str(source_m3g4_path),
            "source_p3g2_probe_path": str(source_p3g2_probe_path),
            "source_p3g3_path": str(source_p3g3_path),
            "environments_dir": (
                None if environments_dir is None else str(environments_dir)
            ),
            "game_id": game_id,
            "adapter_relearned": False,
            "source_cells_rerun": True,
            "execution_performed": True,
            "selection_uses_risk_targeted_candidate_outcomes": False,
            "risk_targeting_features": [
                "hold_baseline_terminal_adjusted_progress>=100",
                "terminal_horizon_remaining<=54",
                "hold_high_ge_120",
                "horizon_mid_45_54",
                "horizon_near_lt_45",
            ],
        },
        "source_p3g2_summary": source_p3g2_summary(source_p3g2_payload),
        "source_p3g3_summary": source_p3g3_summary(source_p3g3_payload),
        "risk_targeted_candidate_plans": [plan.to_dict() for plan in plans],
        "risk_targeted_candidate_records": candidate_records,
        "accepted_risk_targeted_safe_stops": accepted_safe_stops,
        "execution_cells": execution_cells,
        "per_safe_stop_validation_records": per_safe_stop_records,
        "policy_decision_records": policy_records,
        "baseline_aggregates": baseline_aggregates,
        "contextual_policy_aggregate": contextual_aggregate,
        "risk_targeted_extension_risk_stats": risk_stats,
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


def generate_risk_targeted_candidate_plans() -> Tuple[SafeStopCandidatePlan, ...]:
    plans: List[SafeStopCandidatePlan] = []
    for length in range(9, 17):
        base = alternating_prefix("ACTION3", "ACTION4", length)
        phase = alternating_prefix("ACTION4", "ACTION3", length)
        variants = (
            ("base_tail_action6_action3", base + ("ACTION6", "ACTION3")),
            ("base_tail_action6_action4", base + ("ACTION6", "ACTION4")),
            ("base_tail_action3_action6", base + ("ACTION3", "ACTION6")),
            ("base_tail_action4_action6", base + ("ACTION4", "ACTION6")),
            ("phase_tail_action6_action3", phase + ("ACTION6", "ACTION3")),
            ("phase_tail_action6_action4", phase + ("ACTION6", "ACTION4")),
        )
        for name, actions in variants:
            plans.append(
                risk_plan(
                    plan_id=f"p3_g4::{name}::{length:02d}",
                    actions=actions,
                    sampling_family=name,
                    tie_break_seed=length,
                )
            )

    # Add a few shorter OOS controls so the artifact can distinguish risk
    # targeting failure from a general execution failure.
    for plan in generate_out_of_sample_candidate_plans():
        if "relation_prefix_action6_tail" in plan.plan_id and plan.plan_id.endswith(
            ("11", "13", "15")
        ):
            plans.append(
                risk_plan(
                    plan_id=plan.plan_id.replace("p3_g3", "p3_g4"),
                    actions=[
                        str(step.get("action", ""))
                        for step in plan.planned_prefix
                    ],
                    sampling_family="risk_targeted_" + plan.sampling_family,
                    tie_break_seed=int(plan.tie_break_seed) + 400,
                )
            )
    return tuple(deduplicate_plans(plans))


def risk_plan(
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
            "Risk-targeted OOS endpoint near high-hold / mid-horizon extension risk."
        ),
        tie_break_seed=int(tie_break_seed),
    )


def alternating_prefix(action_a: str, action_b: str, length: int) -> Tuple[str, ...]:
    return tuple(action_a if index % 2 == 0 else action_b for index in range(length))


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


def classify_risk_targeted_safe_stops(
    records: Sequence[Mapping[str, Any]],
    *,
    excluded_identity: Mapping[str, set[str]],
    min_risk_targeted_safe_stops: int,
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
        status = RISK_TARGETED_SAFE_STOP_ACCEPTED
        reason = ""
        if not bool(row.get("execution_performed", False)):
            status = "BLOCKED_RISK_TARGETED_SAFE_STOP_REJECTED"
            reason = str(row.get("blocked_reason", "blocked_candidate"))
        elif not bool(row.get("replay_exact", False)):
            status = "REPLAY_INEXACT_RISK_TARGETED_SAFE_STOP_REJECTED"
            reason = "candidate_prefix_not_replay_exact"
        elif not bool(row.get("non_terminal", False)) or not bool(
            row.get("terminal_safe", False)
        ):
            status = UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = "terminal_or_terminal_unsafe_endpoint"
        elif not bool(row.get("hold_baseline_measurable", False)):
            status = UNSAFE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = "hold_baseline_unmeasurable"
        elif state_hash in excluded_identity["state_hashes"] or prefix_hash in excluded_identity[
            "prefix_hashes"
        ]:
            status = IN_SAMPLE_SAFE_STOP_REJECTED
            reason = "state_or_prefix_seen_in_m3_g4_p3_g2_or_p3_g3"
        elif (
            state_hash in seen_state_hashes
            or prefix_hash in seen_prefix_hashes
            or relation_signature in seen_relation_signatures
        ):
            status = DUPLICATE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            reason = "state_prefix_or_relation_signature_not_novel_risk_targeted_oos"
        elif not is_risk_targeted_context(row):
            status = NON_TARGET_RISK_CONTEXT_REJECTED
            reason = "safe_stop_not_in_high_hold_or_mid_horizon_target_region"

        row["risk_targeted_acceptance_status"] = status
        row["out_of_sample_acceptance_status"] = (
            OUT_OF_SAMPLE_SAFE_STOP_ACCEPTED
            if status == RISK_TARGETED_SAFE_STOP_ACCEPTED
            else status
        )
        row["rejection_reason"] = reason
        row["candidate_needed_for_p3_g4"] = (
            len(accepted) < int(min_risk_targeted_safe_stops)
            and status == RISK_TARGETED_SAFE_STOP_ACCEPTED
        )
        row["risk_targeting_features"] = risk_targeting_features(row)
        classified.append(row)
        if status == RISK_TARGETED_SAFE_STOP_ACCEPTED:
            seen_state_hashes.add(state_hash)
            seen_prefix_hashes.add(prefix_hash)
            seen_relation_signatures.add(relation_signature)
            public = out_of_sample_safe_stop_public_record(row, len(accepted) + 1)
            public["safe_stop_id"] = f"p3_g4::risk_oos_safe_stop::{len(accepted) + 1:03d}"
            public["risk_targeting_features"] = risk_targeting_features(row)
            public["risk_targeted_from_p3_g4"] = True
            accepted.append(public)
    return classified, accepted


def is_risk_targeted_context(record: Mapping[str, Any]) -> bool:
    hold = float(record.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0)
    estimate = dict(record.get("terminal_horizon_estimate", {}) or {})
    remaining = estimate.get("estimated_moves_remaining")
    horizon = None if remaining is None else int(remaining)
    return hold >= 100.0 or (horizon is not None and horizon <= 54)


def risk_targeting_features(record: Mapping[str, Any]) -> List[str]:
    features: List[str] = []
    hold = float(record.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0)
    estimate = dict(record.get("terminal_horizon_estimate", {}) or {})
    remaining = estimate.get("estimated_moves_remaining")
    horizon = None if remaining is None else int(remaining)
    if hold >= 120.0:
        features.append("hold_high_ge_120")
    elif hold >= 100.0:
        features.append("hold_elevated_ge_100")
    if horizon is not None:
        if horizon < 45:
            features.append("horizon_near_lt_45")
        elif horizon <= 54:
            features.append("horizon_mid_45_54")
    return features


def risk_targeted_extension_risk_stats(
    *,
    per_safe_stop_records: Sequence[Mapping[str, Any]],
    policy_records: Sequence[Mapping[str, Any]],
    baseline_aggregates: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    terminal_extension_records: List[Dict[str, Any]] = []
    for safe_stop in per_safe_stop_records:
        for candidate in safe_stop.get("candidate_records", []) or []:
            if not isinstance(candidate, Mapping):
                continue
            sequence = ",".join(str(action) for action in candidate.get("action_or_sequence", []) or [])
            if sequence not in {ACTION6_ACTION3, ACTION6_ACTION4}:
                continue
            if not bool(candidate.get("candidate_terminal_reentry", False)):
                continue
            terminal_extension_records.append(
                {
                    "safe_stop_id": str(safe_stop.get("safe_stop_id", "")),
                    "sampling_family": str(safe_stop.get("sampling_family", "")),
                    "terminal_horizon_band": str(safe_stop.get("terminal_horizon_band", "")),
                    "hold_baseline_band": str(safe_stop.get("hold_baseline_band", "")),
                    "action_or_sequence": list(candidate.get("action_or_sequence", []) or []),
                    "candidate_terminal_adjusted_progress_after_stop": float(
                        candidate.get("candidate_terminal_adjusted_progress_after_stop", 0.0) or 0.0
                    ),
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": TRUTH_STATUS,
                }
            )
    unsafe_avoided = sum(
        int(record.get("unsafe_extension_options_avoided", 0) or 0)
        for record in policy_records
    )
    return {
        "static_extension_terminal_options": len(terminal_extension_records),
        "static_extension_terminal_safe_stops": len(
            {record["safe_stop_id"] for record in terminal_extension_records}
        ),
        "terminal_extension_records": terminal_extension_records,
        "unsafe_extension_options_avoided_by_selector": unsafe_avoided,
        "action6_action3_terminal_rate": float(
            baseline_aggregates.get(ACTION6_ACTION3, {}).get("terminal_rate", 0.0) or 0.0
        ),
        "action6_action4_terminal_rate": float(
            baseline_aggregates.get(ACTION6_ACTION4, {}).get("terminal_rate", 0.0) or 0.0
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def risk_targeted_policy_status(
    *,
    accepted_safe_stops: Sequence[Mapping[str, Any]],
    contextual_aggregate: Mapping[str, Any],
    baseline_aggregates: Mapping[str, Mapping[str, Any]],
    risk_stats: Mapping[str, Any],
    min_risk_targeted_safe_stops: int,
) -> str:
    if int(contextual_aggregate.get("objective_completion_runs", 0) or 0) > 0:
        return RISK_TARGETED_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY
    if len(accepted_safe_stops) < int(min_risk_targeted_safe_stops):
        return INSUFFICIENT_RISK_TARGETED_OOS_CONTEXTS_CANDIDATE_ONLY
    contextual_terminal = float(contextual_aggregate.get("terminal_rate", 0.0) or 0.0)
    if contextual_terminal > 0.0:
        return RISK_TARGETED_POLICY_RISKY_CANDIDATE_ONLY
    contextual_taps = float(contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0)
    action6_taps = float(
        baseline_aggregates.get(ACTION6_ONLY, {}).get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    terminal_options = int(risk_stats.get("static_extension_terminal_options", 0) or 0)
    unsafe_avoided = int(risk_stats.get("unsafe_extension_options_avoided_by_selector", 0) or 0)
    if terminal_options > 0 and unsafe_avoided > 0 and contextual_taps >= action6_taps:
        return RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY
    if terminal_options == 0 and contextual_taps >= action6_taps:
        return TARGETED_RISK_NOT_REPRODUCED_CANDIDATE_ONLY
    return RISK_TARGETED_POLICY_HARMFUL_CANDIDATE_ONLY


def summarize_risk_targeted_validation(
    *,
    plans: Sequence[SafeStopCandidatePlan],
    candidate_records: Sequence[Mapping[str, Any]],
    accepted_safe_stops: Sequence[Mapping[str, Any]],
    execution_cells: Sequence[Mapping[str, Any]],
    policy_records: Sequence[Mapping[str, Any]],
    baseline_aggregates: Mapping[str, Mapping[str, Any]],
    contextual_aggregate: Mapping[str, Any],
    risk_stats: Mapping[str, Any],
    validation_status: str,
    source_p3g2_payload: Mapping[str, Any],
    source_p3g3_payload: Mapping[str, Any],
    min_risk_targeted_safe_stops: int,
) -> Dict[str, Any]:
    action6_mean = float(
        baseline_aggregates.get(ACTION6_ONLY, {}).get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    contextual_mean = float(contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0)
    return {
        "policy_utility_status": validation_status,
        "adapter_relearned": False,
        "source_cells_rerun": True,
        "execution_performed": True,
        "candidate_plans": len(plans),
        "candidate_plans_executed": len(
            [record for record in candidate_records if bool(record.get("execution_performed", False))]
        ),
        "accepted_risk_targeted_safe_stops": len(accepted_safe_stops),
        "min_risk_targeted_safe_stops_required": int(min_risk_targeted_safe_stops),
        "in_sample_or_previous_oos_safe_stops_rejected": len(
            [
                record
                for record in candidate_records
                if str(record.get("risk_targeted_acceptance_status", ""))
                == IN_SAMPLE_SAFE_STOP_REJECTED
            ]
        ),
        "duplicate_risk_targeted_safe_stops_rejected": len(
            [
                record
                for record in candidate_records
                if str(record.get("risk_targeted_acceptance_status", ""))
                == DUPLICATE_OUT_OF_SAMPLE_SAFE_STOP_REJECTED
            ]
        ),
        "non_target_risk_contexts_rejected": len(
            [
                record
                for record in candidate_records
                if str(record.get("risk_targeted_acceptance_status", ""))
                == NON_TARGET_RISK_CONTEXT_REJECTED
            ]
        ),
        "unsafe_or_terminal_safe_stops_rejected": len(
            [
                record
                for record in candidate_records
                if str(record.get("risk_targeted_acceptance_status", ""))
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
        "mean_terminal_adjusted_progress": contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0),
        "mean_delta_vs_hold": contextual_aggregate.get("mean_delta_vs_hold", 0.0),
        "mean_delta_vs_action6_only": round(contextual_mean - action6_mean, 6),
        "improvement_over_action6_only": contextual_mean >= action6_mean,
        "static_extension_terminal_options": risk_stats.get("static_extension_terminal_options", 0),
        "static_extension_terminal_safe_stops": risk_stats.get("static_extension_terminal_safe_stops", 0),
        "unsafe_extension_options_avoided": risk_stats.get("unsafe_extension_options_avoided_by_selector", 0),
        "objective_completion_signal": bool(
            int(contextual_aggregate.get("objective_completion_runs", 0) or 0) > 0
        ),
        "objective_completion_runs": int(contextual_aggregate.get("objective_completion_runs", 0) or 0),
        "baseline_aggregates": dict(baseline_aggregates),
        "contextual_policy_aggregate": dict(contextual_aggregate),
        "risk_targeted_extension_risk_stats": dict(risk_stats),
        "source_p3g2_policy_utility_status": str(
            source_p3g2_payload.get("policy_utility_status", "")
            or (source_p3g2_payload.get("summary", {}) or {}).get("policy_utility_status", "")
        ),
        "source_p3g3_policy_utility_status": str(
            source_p3g3_payload.get("policy_utility_status", "")
            or (source_p3g3_payload.get("summary", {}) or {}).get("policy_utility_status", "")
        ),
        "selection_uses_risk_targeted_candidate_outcomes": False,
        "policy_result_counted_as_scientific_verdict": False,
        "adapter_counted_as_mechanic_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def combined_seen_identity(
    source_m3g4_payload: Mapping[str, Any],
    source_p3g3_payload: Mapping[str, Any],
) -> Dict[str, set[str]]:
    identity = in_sample_safe_stop_identity(source_m3g4_payload)
    for record in source_p3g3_payload.get("accepted_out_of_sample_safe_stops", []) or []:
        if not isinstance(record, Mapping):
            continue
        if record.get("safe_stop_state_hash"):
            identity["state_hashes"].add(str(record.get("safe_stop_state_hash", "")))
        if record.get("captured_prefix_hash"):
            identity["prefix_hashes"].add(str(record.get("captured_prefix_hash", "")))
    return identity


def source_p3g3_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    return {
        "source_policy_utility_status": str(
            payload.get("policy_utility_status", "")
            or summary.get("policy_utility_status", "")
        ),
        "source_accepted_out_of_sample_safe_stops": int(
            summary.get("accepted_out_of_sample_safe_stops", 0) or 0
        ),
        "source_terminal_rate": summary.get("terminal_rate"),
        "source_mean_terminal_adjusted_progress": summary.get("mean_terminal_adjusted_progress"),
        "source_counted_as_scientific_verdict": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def validate_p3g3_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G3 source support must remain 0")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("P3.G3 source cannot be a scientific verdict")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("P3.G3 source must not write A32/A33")
    if not payload.get("accepted_out_of_sample_safe_stops"):
        raise ValueError("P3.G4 requires P3.G3 accepted OOS safe-stops")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_risk_targeted_contextual_post_stop_policy_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P3.G4 risk-targeted OOS contextual post-stop validation.",
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH)
    parser.add_argument("--source-m3g4", type=Path, default=DEFAULT_M3G4_DIVERSE_VALIDATION_PATH)
    parser.add_argument("--source-p3g2-probe", type=Path, default=DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH)
    parser.add_argument("--source-p3g3", type=Path, default=DEFAULT_OUT_OF_SAMPLE_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--min-risk-targeted-safe-stops", type=int, default=DEFAULT_MIN_RISK_TARGETED_SAFE_STOPS)
    parser.add_argument("--max-candidate-plans", type=int, default=DEFAULT_MAX_CANDIDATE_PLANS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_risk_targeted_contextual_post_stop_policy_validation(
        adapter_path=args.adapter,
        source_m3g4_path=args.source_m3g4,
        source_p3g2_probe_path=args.source_p3g2_probe,
        source_p3g3_path=args.source_p3g3,
        environments_dir=args.environments_dir,
        game_id=args.game_id,
        min_risk_targeted_safe_stops=args.min_risk_targeted_safe_stops,
        max_candidate_plans=args.max_candidate_plans,
    )
    write_risk_targeted_contextual_post_stop_policy_validation(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "policy_utility_status": payload["summary"]["policy_utility_status"],
                "accepted_risk_targeted_safe_stops": payload["summary"][
                    "accepted_risk_targeted_safe_stops"
                ],
                "static_extension_terminal_options": payload["summary"][
                    "static_extension_terminal_options"
                ],
                "unsafe_extension_options_avoided": payload["summary"][
                    "unsafe_extension_options_avoided"
                ],
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
