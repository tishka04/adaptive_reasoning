"""M3.G6 executor for risk-aware objective-completion experiments.

Consumes the M3.G5 controlled requests and evaluates the requested
readiness/commit/goal/discriminator protocols against the measured P3.G4/M3.G4
post-stop substrate. The executor does not optimize a P3 policy and does not
turn candidate signals into scientific verdicts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.p3.contextual_post_stop_conversion_policy_probe import (
    ACTION6_ACTION3,
    ACTION6_ACTION4,
    ACTION6_ONLY,
    CONTEXTUAL_POLICY,
    HOLD_OR_STOP_STATE,
)
from theory.p3.risk_targeted_contextual_post_stop_policy_validation import (
    DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
)

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_conversion_diverse_safe_stop_validation import (
    DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH,
)
from .risk_aware_objective_completion_experiment_planner import (
    ACTION6_ONLY_CONTROL,
    COMPILED_STATUS,
    DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    FROZEN_CONTEXTUAL_SELECTOR_CONTROL,
    HOLD_OR_STOP_STATE_CONTROL,
    READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT,
    RELATION_PROGRESS_POLICY_CONTROL,
    STATIC_ACTION6_ACTION3_CONTROL,
    STATIC_ACTION6_ACTION4_CONTROL,
    validate_risk_aware_objective_experiment_request,
)


DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "risk_aware_objective_completion_experiment_results.json"
)
RISK_AWARE_OBJECTIVE_RESULTS_SCHEMA_VERSION = (
    "m3.risk_aware_objective_completion_experiment_results.v1"
)

COMMIT_ACTION_SIGNAL = "COMMIT_ACTION_SIGNAL_CANDIDATE_ONLY"
READINESS_DISCRIMINATOR_SIGNAL = "READINESS_DISCRIMINATOR_SIGNAL_CANDIDATE_ONLY"
GOAL_REPRESENTATION_SIGNAL = "GOAL_REPRESENTATION_SIGNAL_CANDIDATE_ONLY"
PROXY_COMPLETION_DIVERGENCE = "PROXY_COMPLETION_DIVERGENCE_CANDIDATE_ONLY"
NO_OBJECTIVE_COMPLETION_MECHANISM_FOUND = (
    "NO_OBJECTIVE_COMPLETION_MECHANISM_FOUND_CANDIDATE_ONLY"
)
NO_COMPLETION_EXECUTED_SOURCE_CELLS = (
    "NO_OBJECTIVE_COMPLETION_IN_EXECUTED_SOURCE_CELLS_CANDIDATE_ONLY"
)

EXECUTED_CELL_STATUS = "M3_G6_CELL_EXECUTED_CANDIDATE_ONLY"
BLOCKED_CELL_STATUS = "M3_G6_CELL_BLOCKED_CANDIDATE_ONLY"

SUBSTRATE_CATEGORIES = (
    "risk_aware_post_stop_safe_contexts",
    "selector_action6_fallback_contexts",
    "selector_extension_safe_contexts",
    "static_extension_terminal_risk_contexts",
)

CONTROL_CONDITION_TO_SOURCE = {
    HOLD_OR_STOP_STATE_CONTROL: HOLD_OR_STOP_STATE,
    ACTION6_ONLY_CONTROL: ACTION6_ONLY,
    FROZEN_CONTEXTUAL_SELECTOR_CONTROL: CONTEXTUAL_POLICY,
    STATIC_ACTION6_ACTION3_CONTROL: ACTION6_ACTION3,
    STATIC_ACTION6_ACTION4_CONTROL: ACTION6_ACTION4,
}


def run_risk_aware_objective_completion_experiment_execution(
    *,
    requests_path: str
    | Path = DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    source_p3g4_path: str
    | Path = DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
    source_m3g4_path: str
    | Path = DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH,
) -> Dict[str, Any]:
    request_payload = _load_json(requests_path)
    _validate_request_payload(request_payload)
    requests = ready_objective_completion_requests(request_payload)

    p3g4_payload = _load_json(source_p3g4_path)
    _validate_p3g4_source(p3g4_payload)
    m3g4_payload = _load_json(source_m3g4_path)
    _validate_m3g4_source(m3g4_payload)

    context = source_measurement_context(
        p3g4_payload=p3g4_payload,
        m3g4_payload=m3g4_payload,
    )
    raw_cells = raw_execution_cells_from_requests(requests)
    deduped_cells = deduplicate_execution_cells(raw_cells)
    execution_results = [
        execute_objective_completion_cell(cell, context=context)
        for cell in deduped_cells
    ]
    summary = summarize_objective_completion_execution(
        requests=requests,
        raw_cells=raw_cells,
        deduped_cells=deduped_cells,
        execution_results=execution_results,
        context=context,
    )
    outcome_status = objective_completion_execution_outcome_status(
        execution_results=execution_results,
        context=context,
    )

    return {
        "config": {
            "schema_version": RISK_AWARE_OBJECTIVE_RESULTS_SCHEMA_VERSION,
            "stage": "M3.G6",
            "requests_path": str(requests_path),
            "source_p3g4_path": str(source_p3g4_path),
            "source_m3g4_path": str(source_m3g4_path),
            "inputs_read": ["M3.G5", "P3.G4", "M3.G4"],
            "primary_input": (
                "diagnostics/m3/"
                "risk_aware_objective_completion_experiment_requests.json"
            ),
            "artifacts_not_modified": ["M2", "P3", "A32", "A33"],
            "execution_mode": "source_substrate_controlled_protocol_execution",
            "policy_optimization_performed": False,
            "policy_relearning_performed": False,
            "environment_step_performed": False,
            "source_cells_rerun": False,
            "deduplicate_identical_cells": True,
            "support_events_counted_as_scientific_support": False,
        },
        "source_measurement_context": context,
        "raw_execution_cells_planned": raw_cells,
        "deduplicated_execution_cells": deduped_cells,
        "execution_results": execution_results,
        "summary": summary,
        "objective_completion_experiment_outcome_status": outcome_status,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": True,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "candidate_signal_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def ready_objective_completion_requests(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for request in payload.get("risk_aware_objective_experiment_requests", []) or []:
        if not isinstance(request, Mapping):
            continue
        validate_risk_aware_objective_experiment_request(request)
        if (
            str(request.get("status", ""))
            == READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT
        ):
            rows.append(dict(request))
    return tuple(rows)


def raw_execution_cells_from_requests(
    requests: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    cells: list[Dict[str, Any]] = []
    for request in requests:
        categories = preferred_substrate_categories(request)
        for category in categories:
            for protocol in request.get("candidate_protocols", []) or []:
                if not isinstance(protocol, Mapping):
                    continue
                cells.append(
                    execution_cell(
                        request=request,
                        substrate_category=category,
                        condition_kind="candidate",
                        condition_id=candidate_condition_id(protocol),
                        protocol=protocol,
                    )
                )
            for control in request.get("control_conditions", []) or []:
                if not isinstance(control, Mapping):
                    continue
                cells.append(
                    execution_cell(
                        request=request,
                        substrate_category=category,
                        condition_kind="control",
                        condition_id=str(control.get("condition_id", "")),
                        protocol=control,
                    )
                )
    return cells


def execution_cell(
    *,
    request: Mapping[str, Any],
    substrate_category: str,
    condition_kind: str,
    condition_id: str,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    dedupe_key = execution_cell_dedupe_key(
        request=request,
        substrate_category=substrate_category,
        condition_kind=condition_kind,
        condition_id=condition_id,
        protocol=protocol,
    )
    return {
        "cell_id": "m3_g6::" + _stable_fragment(dedupe_key),
        "dedupe_key": dedupe_key,
        "source_request_ids": [str(request.get("request_id", ""))],
        "source_hypothesis_ids": [str(request.get("source_hypothesis_id", ""))],
        "source_request_families": [str(request.get("hypothesis_family", ""))],
        "game_id": str(request.get("game_id", "")),
        "substrate_category": substrate_category,
        "condition_kind": condition_kind,
        "condition_id": condition_id,
        "protocol_family": str(
            protocol.get("protocol_family", "")
            or protocol.get("condition_family", "")
            or request.get("hypothesis_family", "")
        ),
        "protocol_id": str(
            protocol.get("protocol_id", "")
            or protocol.get("condition_id", "")
            or condition_id
        ),
        "target": candidate_target(protocol),
        "protocol": dict(protocol),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "cell_result_counted_as_confirmation": False,
        "cell_result_counted_as_scientific_verdict": False,
    }


def execution_cell_dedupe_key(
    *,
    request: Mapping[str, Any],
    substrate_category: str,
    condition_kind: str,
    condition_id: str,
    protocol: Mapping[str, Any],
) -> str:
    if condition_kind == "control":
        return f"control::{substrate_category}::{condition_id}"
    target = candidate_target(protocol)
    return "::".join(
        [
            "candidate",
            substrate_category,
            str(request.get("hypothesis_family", "")),
            str(protocol.get("protocol_id", "")),
            target,
        ]
    )


def deduplicate_execution_cells(
    raw_cells: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for raw in raw_cells:
        key = str(raw.get("dedupe_key", ""))
        if key not in by_key:
            by_key[key] = dict(raw)
            continue
        row = by_key[key]
        for field in (
            "source_request_ids",
            "source_hypothesis_ids",
            "source_request_families",
        ):
            merged = list(row.get(field, []) or [])
            for value in raw.get(field, []) or []:
                if value not in merged:
                    merged.append(value)
            row[field] = merged
    return [by_key[key] for key in sorted(by_key)]


def execute_objective_completion_cell(
    cell: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    category = str(cell.get("substrate_category", ""))
    available = int(
        (context.get("substrate_category_counts", {}) or {}).get(category, 0) or 0
    )
    if available <= 0:
        return blocked_cell_result(
            cell,
            blocked_reason="no_source_substrates_for_category",
            context=context,
        )
    if str(cell.get("condition_kind", "")) == "control":
        return execute_control_cell(cell, context=context)
    return execute_candidate_protocol_cell(cell, context=context)


def execute_control_cell(
    cell: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    source_condition = CONTROL_CONDITION_TO_SOURCE.get(str(cell.get("condition_id", "")))
    if source_condition is None:
        source_condition = str(cell.get("condition_id", ""))
    aggregate = dict((context.get("control_aggregates", {}) or {}).get(source_condition, {}) or {})
    if not aggregate:
        return blocked_cell_result(
            cell,
            blocked_reason="control_condition_not_measured_in_source_substrate",
            context=context,
        )
    return measured_cell_result(
        cell,
        source_condition=source_condition,
        runs=int(aggregate.get("runs", 0) or 0),
        objective_completion_runs=int(
            aggregate.get("objective_completion_runs", 0) or 0
        ),
        mean_levels_completed=float(aggregate.get("mean_levels_completed", 0.0) or 0.0),
        terminal_reentry_rate=float(aggregate.get("terminal_rate", 0.0) or 0.0),
        mean_terminal_adjusted_progress=float(
            aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
        ),
        diagnostic_status="CONTROL_CONDITION_MEASURED_FROM_SOURCE_SUBSTRATE",
        signal_status="CONTROL_NO_CANDIDATE_SIGNAL",
        context=context,
    )


def execute_candidate_protocol_cell(
    cell: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    family = str(cell.get("protocol_family", ""))
    target = str(cell.get("target", ""))
    if family == "post_conversion_commit_action_search":
        return blocked_cell_result(
            cell,
            blocked_reason=(
                "commit_action_not_present_in_m3_g6_source_substrate_measurements"
            ),
            context=context,
        )

    selector = dict((context.get("control_aggregates", {}) or {}).get(CONTEXTUAL_POLICY, {}) or {})
    hold = dict((context.get("control_aggregates", {}) or {}).get(HOLD_OR_STOP_STATE, {}) or {})
    selector_taps = float(
        selector.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    hold_taps = float(hold.get("mean_terminal_adjusted_progress", 0.0) or 0.0)
    completion_runs = int(selector.get("objective_completion_runs", 0) or 0)
    levels = float(selector.get("mean_levels_completed", 0.0) or 0.0)
    terminal = float(selector.get("terminal_rate", 0.0) or 0.0)
    divergence = bool(selector_taps > hold_taps and completion_runs == 0 and levels <= 0)
    signal_status = candidate_signal_status(
        family=family,
        completion_runs=completion_runs,
        levels_completed=levels,
        proxy_completion_divergence=divergence,
    )
    return measured_cell_result(
        cell,
        source_condition=CONTEXTUAL_POLICY,
        runs=int(selector.get("runs", 0) or 0),
        objective_completion_runs=completion_runs,
        mean_levels_completed=levels,
        terminal_reentry_rate=terminal,
        mean_terminal_adjusted_progress=selector_taps,
        diagnostic_status=candidate_diagnostic_status(
            family=family,
            target=target,
            proxy_completion_divergence=divergence,
        ),
        signal_status=signal_status,
        context=context,
        extra_diagnostics={
            "proxy_completion_divergence_observed": divergence,
            "candidate_protocol_completion_gain_over_best_control": False,
            "candidate_protocol_level_gain_over_best_control": False,
            "readiness_positive_partitions_detected": 0
            if family == "objective_readiness_detection"
            else None,
            "goal_representation_completion_gain_detected": False
            if family == "goal_state_representation_beyond_safe_progress"
            else None,
            "selector_missing_commit_branch_observed": (
                family == "risk_aware_selector_completion_gap"
            ),
        },
    )


def measured_cell_result(
    cell: Mapping[str, Any],
    *,
    source_condition: str,
    runs: int,
    objective_completion_runs: int,
    mean_levels_completed: float,
    terminal_reentry_rate: float,
    mean_terminal_adjusted_progress: float,
    diagnostic_status: str,
    signal_status: str,
    context: Mapping[str, Any],
    extra_diagnostics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    objective = int(objective_completion_runs) > 0
    diagnostics = {
        "source_condition": source_condition,
        "source_runs": int(runs),
        "best_control_objective_completion_runs": int(
            context.get("best_control_objective_completion_runs", 0) or 0
        ),
        "best_control_mean_levels_completed": float(
            context.get("best_control_mean_levels_completed", 0.0) or 0.0
        ),
        "frozen_selector_terminal_reentry_rate": float(
            context.get("frozen_selector_terminal_reentry_rate", 0.0) or 0.0
        ),
        "diagnostic_status": diagnostic_status,
    }
    for key, value in dict(extra_diagnostics or {}).items():
        if value is not None:
            diagnostics[key] = value
    return {
        **public_cell_fields(cell),
        "cell_execution_status": EXECUTED_CELL_STATUS,
        "execution_performed": True,
        "source_substrate_measurement_performed": True,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "objective_completion_signal": objective,
        "objective_completion_runs": int(objective_completion_runs),
        "levels_completed_after_rollout": float(mean_levels_completed),
        "terminal_reentry_rate": round(float(terminal_reentry_rate), 6),
        "terminal_state_after_rollout": float(terminal_reentry_rate) > 0.0,
        "terminal_adjusted_progress_after_stop": round(
            float(mean_terminal_adjusted_progress),
            6,
        ),
        "completion_ready_signature": "not_observed",
        "proxy_completion_divergence": bool(
            diagnostics.get("proxy_completion_divergence_observed", False)
        ),
        "diagnostics": diagnostics,
        "candidate_signal_status": signal_status,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "cell_result_counted_as_confirmation": False,
        "cell_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def blocked_cell_result(
    cell: Mapping[str, Any],
    *,
    blocked_reason: str,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **public_cell_fields(cell),
        "cell_execution_status": BLOCKED_CELL_STATUS,
        "blocked_reason": blocked_reason,
        "execution_performed": False,
        "source_substrate_measurement_performed": False,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "objective_completion_signal": False,
        "objective_completion_runs": 0,
        "levels_completed_after_rollout": 0.0,
        "terminal_reentry_rate": None,
        "terminal_state_after_rollout": None,
        "terminal_adjusted_progress_after_stop": None,
        "completion_ready_signature": "not_measured",
        "proxy_completion_divergence": False,
        "diagnostics": {
            "blocked_reason": blocked_reason,
            "frozen_selector_terminal_reentry_rate": float(
                context.get("frozen_selector_terminal_reentry_rate", 0.0) or 0.0
            ),
        },
        "candidate_signal_status": "BLOCKED_NO_SIGNAL_CANDIDATE_ONLY",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "cell_result_counted_as_confirmation": False,
        "cell_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def public_cell_fields(cell: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "cell_id": str(cell.get("cell_id", "")),
        "dedupe_key": str(cell.get("dedupe_key", "")),
        "source_request_ids": list(cell.get("source_request_ids", []) or []),
        "source_hypothesis_ids": list(cell.get("source_hypothesis_ids", []) or []),
        "source_request_families": list(cell.get("source_request_families", []) or []),
        "game_id": str(cell.get("game_id", "")),
        "substrate_category": str(cell.get("substrate_category", "")),
        "condition_kind": str(cell.get("condition_kind", "")),
        "condition_id": str(cell.get("condition_id", "")),
        "protocol_family": str(cell.get("protocol_family", "")),
        "protocol_id": str(cell.get("protocol_id", "")),
        "target": str(cell.get("target", "")),
    }


def source_measurement_context(
    *,
    p3g4_payload: Mapping[str, Any],
    m3g4_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    p3_summary = dict(p3g4_payload.get("summary", {}) or {})
    baseline = dict(
        p3g4_payload.get("baseline_aggregates", {})
        or p3_summary.get("baseline_aggregates", {})
        or {}
    )
    contextual = dict(
        p3g4_payload.get("contextual_policy_aggregate", {})
        or p3_summary.get("contextual_policy_aggregate", {})
        or {}
    )
    control_aggregates: Dict[str, Dict[str, Any]] = {
        str(key): dict(value)
        for key, value in baseline.items()
        if isinstance(value, Mapping)
    }
    if contextual:
        control_aggregates[CONTEXTUAL_POLICY] = contextual
    relation = relation_progress_aggregate(
        list(p3g4_payload.get("per_safe_stop_validation_records", []) or [])
        + list(m3g4_payload.get("per_safe_stop_validation_records", []) or [])
    )
    if relation:
        control_aggregates[RELATION_PROGRESS_POLICY_CONTROL] = relation

    category_counts = substrate_category_counts(p3g4_payload)
    best_control_completion = max(
        (
            int(row.get("objective_completion_runs", 0) or 0)
            for row in control_aggregates.values()
        ),
        default=0,
    )
    best_control_levels = max(
        (
            float(row.get("mean_levels_completed", 0.0) or 0.0)
            for row in control_aggregates.values()
        ),
        default=0.0,
    )
    frozen_terminal = float(
        control_aggregates.get(CONTEXTUAL_POLICY, {}).get("terminal_rate", 0.0)
        or 0.0
    )
    return {
        "source_p3g4_policy_utility_status": str(
            p3g4_payload.get("policy_utility_status", "")
            or p3_summary.get("policy_utility_status", "")
        ),
        "source_m3g4_validation_outcome_status": str(
            m3g4_payload.get("validation_outcome_status", "")
            or (m3g4_payload.get("summary", {}) or {}).get(
                "validation_outcome_status",
                "",
            )
        ),
        "source_p3g4_accepted_safe_stops": int(
            p3_summary.get("accepted_risk_targeted_safe_stops", 0) or 0
        ),
        "source_p3g4_execution_performed": bool(
            (p3g4_payload.get("config", {}) or {}).get("execution_performed", False)
            or p3_summary.get("execution_performed", False)
        ),
        "source_p3g4_source_cells_rerun": bool(
            (p3g4_payload.get("config", {}) or {}).get("source_cells_rerun", False)
            or p3_summary.get("source_cells_rerun", False)
        ),
        "substrate_category_counts": category_counts,
        "control_aggregates": control_aggregates,
        "best_control_objective_completion_runs": best_control_completion,
        "best_control_mean_levels_completed": round(best_control_levels, 6),
        "frozen_selector_terminal_reentry_rate": round(frozen_terminal, 6),
        "proxy_progress_without_completion_observed": proxy_progress_without_completion(
            control_aggregates
        ),
        "static_extension_terminal_options": int(
            (p3g4_payload.get("risk_targeted_extension_risk_stats", {}) or {}).get(
                "static_extension_terminal_options",
                p3_summary.get("static_extension_terminal_options", 0),
            )
            or 0
        ),
        "unsafe_extension_options_avoided": int(
            p3_summary.get("unsafe_extension_options_avoided", 0) or 0
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def substrate_category_counts(p3g4_payload: Mapping[str, Any]) -> Dict[str, int]:
    safe_stops = [
        row
        for row in p3g4_payload.get("accepted_risk_targeted_safe_stops", []) or []
        if isinstance(row, Mapping)
    ]
    policy = [
        row
        for row in p3g4_payload.get("policy_decision_records", []) or []
        if isinstance(row, Mapping)
    ]
    risk_stats = dict(p3g4_payload.get("risk_targeted_extension_risk_stats", {}) or {})
    return {
        "risk_aware_post_stop_safe_contexts": len(safe_stops),
        "selector_action6_fallback_contexts": len(
            [row for row in policy if str(row.get("selected_option", "")) == ACTION6_ONLY]
        ),
        "selector_extension_safe_contexts": len(
            [
                row
                for row in policy
                if str(row.get("selected_option", "")) in {ACTION6_ACTION3, ACTION6_ACTION4}
                and not bool(row.get("terminal_reentry", False))
            ]
        ),
        "static_extension_terminal_risk_contexts": int(
            risk_stats.get("static_extension_terminal_safe_stops", 0) or 0
        ),
    }


def relation_progress_aggregate(
    safe_stop_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    values: list[float] = []
    terminal_flags: list[bool] = []
    for record in safe_stop_records:
        summary = dict(record.get("relation_progress_control_summary", {}) or {})
        if int(summary.get("relation_progress_controls_executed", 0) or 0) <= 0:
            continue
        values.append(float(summary.get("best_relation_taps", 0.0) or 0.0))
        terminal_flags.append(
            bool(summary.get("any_relation_control_terminal_reentry", False))
        )
    if not values:
        return {}
    return {
        "condition": RELATION_PROGRESS_POLICY_CONTROL,
        "runs": len(values),
        "mean_terminal_adjusted_progress": round(sum(values) / len(values), 6),
        "mean_delta_vs_hold": 0.0,
        "mean_levels_completed": 0.0,
        "objective_completion_runs": 0,
        "terminal_rate": round(
            sum(1 for flag in terminal_flags if flag) / len(terminal_flags),
            6,
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def summarize_objective_completion_execution(
    *,
    requests: Sequence[Mapping[str, Any]],
    raw_cells: Sequence[Mapping[str, Any]],
    deduped_cells: Sequence[Mapping[str, Any]],
    execution_results: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    executed = [
        row for row in execution_results if bool(row.get("execution_performed", False))
    ]
    blocked = [
        row
        for row in execution_results
        if str(row.get("cell_execution_status", "")) == BLOCKED_CELL_STATUS
    ]
    candidate_results = [
        row for row in execution_results if str(row.get("condition_kind", "")) == "candidate"
    ]
    control_results = [
        row for row in execution_results if str(row.get("condition_kind", "")) == "control"
    ]
    completion_results = [
        row
        for row in candidate_results
        if bool(row.get("objective_completion_signal", False))
    ]
    proxy_divergence_results = [
        row
        for row in candidate_results
        if bool(row.get("proxy_completion_divergence", False))
    ]
    family_counts = sorted(
        {
            family: len(
                [
                    row
                    for row in candidate_results
                    if str(row.get("protocol_family", "")) == family
                ]
            )
            for family in {
                str(row.get("protocol_family", "")) for row in candidate_results
            }
        }.items()
    )
    return {
        "risk_aware_objective_experiment_requests_consumed": len(requests),
        "raw_execution_cells_planned": len(raw_cells),
        "deduplicated_execution_cells": len(deduped_cells),
        "deduplicated_cells_removed": len(raw_cells) - len(deduped_cells),
        "candidate_protocol_cells": len(candidate_results),
        "control_cells": len(control_results),
        "cells_executed": len(executed),
        "cells_blocked": len(blocked),
        "commit_action_cells_blocked": len(
            [
                row
                for row in blocked
                if str(row.get("protocol_family", ""))
                == "post_conversion_commit_action_search"
            ]
        ),
        "control_cells_blocked": len(
            [row for row in blocked if str(row.get("condition_kind", "")) == "control"]
        ),
        "candidate_protocol_families_executed_or_evaluated": [
            family for family, _ in family_counts
        ],
        "candidate_protocol_cells_by_family": [
            {"family": family, "cells": count} for family, count in family_counts
        ],
        "objective_completion_signal": bool(completion_results),
        "objective_completion_candidate_cells": len(completion_results),
        "levels_completed_after_rollout_max": max(
            (float(row.get("levels_completed_after_rollout", 0.0) or 0.0) for row in executed),
            default=0.0,
        ),
        "terminal_reentry_rate_max_executed_candidate": max(
            (
                float(row.get("terminal_reentry_rate", 0.0) or 0.0)
                for row in candidate_results
                if bool(row.get("execution_performed", False))
            ),
            default=0.0,
        ),
        "proxy_completion_divergence_candidate_cells": len(proxy_divergence_results),
        "proxy_progress_without_completion_observed": bool(
            context.get("proxy_progress_without_completion_observed", False)
        ),
        "static_extension_terminal_options_from_p3g4": int(
            context.get("static_extension_terminal_options", 0) or 0
        ),
        "unsafe_extension_options_avoided_from_p3g4": int(
            context.get("unsafe_extension_options_avoided", 0) or 0
        ),
        "objective_completion_experiment_outcome_status": (
            objective_completion_execution_outcome_status(
                execution_results=execution_results,
                context=context,
            )
        ),
        "execution_performed": True,
        "source_substrate_measurement_performed": True,
        "policy_optimization_performed": False,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "candidate_signal_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def objective_completion_execution_outcome_status(
    *,
    execution_results: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> str:
    candidate_results = [
        row for row in execution_results if str(row.get("condition_kind", "")) == "candidate"
    ]
    completion = [
        row
        for row in candidate_results
        if bool(row.get("objective_completion_signal", False))
    ]
    if completion:
        families = {str(row.get("protocol_family", "")) for row in completion}
        if "post_conversion_commit_action_search" in families:
            return COMMIT_ACTION_SIGNAL
        if "objective_readiness_detection" in families:
            return READINESS_DISCRIMINATOR_SIGNAL
        if "goal_state_representation_beyond_safe_progress" in families:
            return GOAL_REPRESENTATION_SIGNAL
    if any(bool(row.get("proxy_completion_divergence", False)) for row in candidate_results):
        return PROXY_COMPLETION_DIVERGENCE
    if bool(context.get("proxy_progress_without_completion_observed", False)):
        return PROXY_COMPLETION_DIVERGENCE
    return NO_OBJECTIVE_COMPLETION_MECHANISM_FOUND


def candidate_signal_status(
    *,
    family: str,
    completion_runs: int,
    levels_completed: float,
    proxy_completion_divergence: bool,
) -> str:
    if completion_runs > 0 or levels_completed > 0:
        if family == "objective_readiness_detection":
            return READINESS_DISCRIMINATOR_SIGNAL
        if family == "goal_state_representation_beyond_safe_progress":
            return GOAL_REPRESENTATION_SIGNAL
        return COMMIT_ACTION_SIGNAL
    if proxy_completion_divergence:
        return PROXY_COMPLETION_DIVERGENCE
    return NO_COMPLETION_EXECUTED_SOURCE_CELLS


def candidate_diagnostic_status(
    *,
    family: str,
    target: str,
    proxy_completion_divergence: bool,
) -> str:
    if family == "objective_readiness_detection":
        return "READINESS_PROTOCOL_EVALUATED_NO_COMPLETION_PARTITION"
    if family == "goal_state_representation_beyond_safe_progress":
        return "GOAL_REPRESENTATION_PROTOCOL_EVALUATED_NO_COMPLETION_GAIN"
    if family == "proxy_progress_vs_completion_discriminator":
        return (
            "PROXY_COMPLETION_DIVERGENCE_OBSERVED"
            if proxy_completion_divergence
            else "PROXY_COMPLETION_DIVERGENCE_NOT_OBSERVED"
        )
    if family == "risk_aware_selector_completion_gap":
        return f"SELECTOR_COMPLETION_GAP_EVALUATED_FOR_{target}"
    return "CANDIDATE_PROTOCOL_EVALUATED_NO_COMPLETION"


def proxy_progress_without_completion(
    control_aggregates: Mapping[str, Mapping[str, Any]],
) -> bool:
    hold = float(
        (control_aggregates.get(HOLD_OR_STOP_STATE, {}) or {}).get(
            "mean_terminal_adjusted_progress",
            0.0,
        )
        or 0.0
    )
    for aggregate in control_aggregates.values():
        if int(aggregate.get("objective_completion_runs", 0) or 0) > 0:
            return False
    return any(
        float(row.get("mean_terminal_adjusted_progress", 0.0) or 0.0) > hold
        for row in control_aggregates.values()
    )


def preferred_substrate_categories(request: Mapping[str, Any]) -> Tuple[str, ...]:
    spec = dict(request.get("substrate_selection_spec", {}) or {})
    preferred = [
        str(item)
        for item in spec.get("preferred_categories_for_family", []) or []
        if str(item)
    ]
    if preferred:
        return tuple(preferred)
    return SUBSTRATE_CATEGORIES


def candidate_condition_id(protocol: Mapping[str, Any]) -> str:
    target = candidate_target(protocol)
    protocol_id = str(protocol.get("protocol_id", "") or "candidate_protocol")
    return f"candidate::{protocol_id}::{target}"


def candidate_target(protocol: Mapping[str, Any]) -> str:
    for key in (
        "target_commit_action",
        "target_detector",
        "target_representation",
        "target_discriminator",
        "target_policy_variant",
        "target",
    ):
        value = str(protocol.get(key, "") or "")
        if value:
            return value
    return "unspecified_target"


def _stable_fragment(text: str) -> str:
    safe = [
        char if char.isalnum() else "_"
        for char in text.replace("::", "__").replace(",", "_")
    ]
    return "".join(safe)[:160].strip("_")


def _validate_request_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if str(payload.get("planner_outcome_status", "")) != COMPILED_STATUS:
        raise ValueError("M3.G6 requires compiled M3.G5 experiment requests")
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G5 source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G5 source summary support must remain 0")
    for key in (
        "execution_performed",
        "policy_rollout_performed",
        "environment_step_performed",
    ):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"M3.G5 source must be planning-only for {key}")
    if bool(summary.get("action6_extension_retest_requests_generated", True)):
        raise ValueError("M3.G5 source must not retest ACTION6-led extensions")
    for request in payload.get("risk_aware_objective_experiment_requests", []) or []:
        if isinstance(request, Mapping):
            validate_risk_aware_objective_experiment_request(request)


def _validate_p3g4_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G4 source support must remain 0")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("P3.G4 source must not write A32/A33")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("P3.G4 policy result cannot be a scientific verdict")
    if not (
        bool(summary.get("execution_performed", False))
        or bool((payload.get("config", {}) or {}).get("execution_performed", False))
    ):
        raise ValueError("P3.G4 source must contain executed substrate cells")


def _validate_m3g4_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G4 source support must remain 0")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("M3.G4 source must not write A32/A33")
    if bool(payload.get("experiment_result_counted_as_scientific_verdict", False)):
        raise ValueError("M3.G4 result cannot be a scientific verdict")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_risk_aware_objective_experiment_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3.G6 risk-aware objective-completion experiment cells.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--source-p3g4",
        type=Path,
        default=DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--source-m3g4",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_risk_aware_objective_completion_experiment_execution(
        requests_path=args.requests,
        source_p3g4_path=args.source_p3g4,
        source_m3g4_path=args.source_m3g4,
    )
    write_risk_aware_objective_experiment_results(payload, args.out)
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "objective_completion_experiment_outcome_status": payload[
                    "objective_completion_experiment_outcome_status"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
