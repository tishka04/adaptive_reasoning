"""SAGE.7b exact execution of the third-game parameterized frontier."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from theory.m1.controlled_followup_experiment import controlled_delta, metric_signal
from theory.m2.validators import validate_m3_request
from theory.non_ar25_active_micro_run import _env_dir

from .live_mini_frontier_m3_executor import EnvFactory, _execute_request_arm
from .parameterized_control_acquisition import discrimination_status_from_delta
from .third_unknown_game_parameterized_frontier import (
    DEFAULT_CONTROLS_PER_REQUEST,
    DEFAULT_REQUESTS_PER_BUDGET,
    DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH,
    PARAMETERIZED_CONTROL_POLICY,
    SAGE7A_FRONTIER_GENERATED,
    SAGE7A_PARAMETERIZED_EXECUTION_REQUIRED,
    SAGE7A_SCHEMA_VERSION,
    SAGE7A_TRUTH_STATUS,
)


DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage7b_third_unknown_game_parameterized_execution.json"
)

SAGE7B_SCHEMA_VERSION = "sage.third_unknown_game_parameterized_execution.v1"
SAGE7B_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_7B"
SAGE7B_EXECUTION_COMPLETED = (
    "SAGE_THIRD_UNKNOWN_GAME_PARAMETERIZED_EXECUTION_COMPLETED_CANDIDATE_ONLY"
)
SAGE7B_EXECUTION_INCOMPLETE = (
    "SAGE_THIRD_UNKNOWN_GAME_PARAMETERIZED_EXECUTION_INCOMPLETE_CANDIDATE_ONLY"
)
SAGE7B_CONSOLIDATION_REQUIRED = (
    "SAGE7C_PARAMETERIZED_EVENT_CONSOLIDATION_REQUIRED_CANDIDATE_ONLY"
)
SAGE7B_NO_CONSOLIDATION_REQUESTED = (
    "NO_PARAMETERIZED_EVENT_CONSOLIDATION_REQUESTED_CANDIDATE_ONLY"
)

EXECUTION_MODE = "EXACT_LIVE_PREFIX_ONE_TARGET_TWO_PRE_REGISTERED_CONTROLS"
EXPECTED_CONTEXT_STATE_ORIGIN = "sage7_third_game_live_prefix_frame_before"
EXPECTED_GENERATOR = "SAGE.7a_third_unknown_game_parameterized_frontier"


def run_sage7b_parameterized_execution(
    *,
    source_sage7a_path: str | Path = DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute every SAGE.7a target once and both controls without substitution."""
    source = _load_json(source_sage7a_path)
    validate_sage7b_source(source)
    game_id = str(source.get("summary", {}).get("game_id", ""))
    budgets = tuple(
        int(value)
        for value in source.get("summary", {}).get("budgets_evaluated", []) or []
    )
    requests = sorted(
        (
            dict(row)
            for row in source.get("mini_frontier_m3_requests", []) or []
            if isinstance(row, Mapping)
        ),
        key=request_sort_key,
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    experiments: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for request in requests:
        result = execute_parameterized_frontier_request(
            request,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        if str(result.get("execution_status", "")) == "EXECUTED":
            experiments.append(result)
        else:
            blocked.append(result)

    per_budget = build_sage7b_budget_records(
        budgets=budgets,
        requests=requests,
        experiments=experiments,
        blocked=blocked,
    )
    gate = build_sage7b_gate(
        source=source,
        game_id=game_id,
        budgets=budgets,
        requests=requests,
        experiments=experiments,
        blocked=blocked,
    )
    outcome = (
        SAGE7B_EXECUTION_COMPLETED
        if gate and all(gate.values())
        else SAGE7B_EXECUTION_INCOMPLETE
    )
    summary = summarize_sage7b(
        source=source,
        game_id=game_id,
        budgets=budgets,
        requests=requests,
        experiments=experiments,
        blocked=blocked,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE7B_SCHEMA_VERSION,
            "source_sage7a_path": str(source_sage7a_path),
            "game_id": game_id,
            "budgets": list(budgets),
            "environments_dir": str(env_dir),
            "execution_mode": EXECUTION_MODE,
            "execute_all_source_requests": True,
            "target_executions_per_request": 1,
            "control_executions_per_request": DEFAULT_CONTROLS_PER_REQUEST,
            "target_and_controls_replay_same_prefix": True,
            "context_snapshot_hash_verification_required": True,
            "exact_target_action_args_required": True,
            "exact_control_action_args_required": True,
            "variant_substitution_allowed": False,
            "request_deduplication_allowed": False,
            "cross_budget_replications_preserved": True,
            "parameterized_variants_are_distinct_action_families": False,
            "benchmark_run": False,
            "inputs_read": ["SAGE.7a"],
            "artifacts_not_modified": [
                "SAGE.7a",
                "SAGE.7",
                "SAGE.6",
                "A32",
                "A33.1",
                "A33.2",
                "A33.3",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
            "scientific_policy": {
                "raw_comparisons_are_candidate_only": True,
                "positive_deltas_are_not_scientific_support": True,
                "negative_deltas_are_not_refutations": True,
                "zero_deltas_are_not_non_identifiability_decisions": True,
                "source_scoped_mechanics_are_quarantined": True,
                "a32_a33_write_performed": False,
            },
        },
        "source_sage7a_context": build_source_sage7a_context(source),
        "executed_parameterized_experiments": experiments,
        "blocked_parameterized_experiments": blocked,
        "per_budget_results": per_budget,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE7B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": True,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "raw_events_counted_as_scientific_support": False,
        "positive_deltas_counted_as_support": False,
        "negative_deltas_counted_as_refutation": False,
        "zero_deltas_counted_as_non_identifiability": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage7b_parameterized_execution(payload, output_path)
    return payload


def execute_parameterized_frontier_request(
    request: Mapping[str, Any],
    *,
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    """Execute one target and its exact pre-registered parameter controls."""
    target_action = str(request.get("target_action", ""))
    target_args = dict(request.get("target_action_args", {}) or {})
    controls = [
        dict(row)
        for row in request.get("pre_registered_parameterized_control_variants", [])
        or []
        if isinstance(row, Mapping)
    ]
    target = _execute_request_arm(
        request,
        action_name=target_action,
        action_args=target_args,
        arm="pre_registered_parameterized_target",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if str(target.get("status", "")) != "EXECUTED":
        return _blocked_result(
            request,
            reason=str(target.get("reason", "target_execution_blocked")),
            target_arm=target,
        )

    control_arms: List[Dict[str, Any]] = []
    for index, control in enumerate(controls):
        arm = _execute_request_arm(
            request,
            action_name=str(control.get("action", "")),
            action_args=dict(control.get("action_args", {}) or {}),
            arm=f"pre_registered_parameterized_control_{index + 1}",
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        control_arms.append(arm)
        if str(arm.get("status", "")) != "EXECUTED":
            return _blocked_result(
                request,
                reason=str(arm.get("reason", f"control_{index + 1}_blocked")),
                target_arm=target,
                control_arms=control_arms,
            )

    substitutions = exact_protocol_substitution_reasons(
        request=request,
        target_arm=target,
        control_arms=control_arms,
    )
    if substitutions:
        return _blocked_result(
            request,
            reason="protocol_substitution_detected:" + ",".join(substitutions),
            target_arm=target,
            control_arms=control_arms,
        )

    metric = str(request.get("metric", ""))
    comparisons: List[Dict[str, Any]] = []
    for index, (control, control_arm) in enumerate(
        zip(controls, control_arms, strict=True), start=1
    ):
        delta = controlled_delta(
            control_arm["measurement_for_delta"],
            target["measurement_for_delta"],
            predicted_metric=metric,
        )
        effect = float(delta.get("effect_size", 0.0) or 0.0)
        comparisons.append(
            {
                "comparison_id": f"{request.get('request_id')}::control_{index}",
                "control_index": index,
                "target_action": target_action,
                "target_action_args": target_args,
                "control_action": str(control.get("action", "")),
                "control_action_args": dict(control.get("action_args", {}) or {}),
                "metric": metric,
                "target_signal": metric_signal(target["measurement_for_delta"], metric),
                "control_signal": metric_signal(
                    control_arm["measurement_for_delta"], metric
                ),
                "controlled_delta": delta,
                "discrimination_status": discrimination_status_from_delta(delta),
                "positive_delta_events": 1 if effect > 0 else 0,
                "negative_delta_events": 1 if effect < 0 else 0,
                "zero_delta_events": 1 if effect == 0 else 0,
                "target_measurement_signature": arm_measurement_signature(
                    target, metric=metric
                ),
                "control_measurement_signature": arm_measurement_signature(
                    control_arm, metric=metric
                ),
                "context_snapshot_hash_verified": bool(
                    target.get("context_snapshot_hash_verified", False)
                    and control_arm.get("context_snapshot_hash_verified", False)
                ),
                "protocol_exact_match": True,
                "support": 0,
                "truth_status": SAGE7B_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "comparison_counted_as_scientific_support": False,
            }
        )

    return {
        "execution_status": "EXECUTED",
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "source_transition_id": str(request.get("source_transition_id", "")),
        "game_id": str(request.get("game_id", "")),
        "budget": budget_from_request(request),
        "source_step": int(request.get("source_step", 0) or 0),
        "hypothesis_family": str(request.get("hypothesis_family", "")),
        "metric": metric,
        "context_snapshot_hash": str(request.get("context_snapshot_hash", "")),
        "execution_mode": EXECUTION_MODE,
        "target_action": target_action,
        "target_action_args": target_args,
        "parameterized_action_family": str(
            request.get("parameterized_action_family", "")
        ),
        "parameterized_control_policy": str(
            request.get("parameterized_control_policy", "")
        ),
        "pre_registered_parameterized_control_variants": controls,
        "target_arm": target,
        "control_arms": control_arms,
        "parameterized_comparisons": comparisons,
        "target_executions": 1,
        "control_executions": len(control_arms),
        "total_arm_executions": 1 + len(control_arms),
        "comparison_events": len(comparisons),
        "positive_delta_events": sum(
            int(row["positive_delta_events"]) for row in comparisons
        ),
        "negative_delta_events": sum(
            int(row["negative_delta_events"]) for row in comparisons
        ),
        "zero_delta_events": sum(int(row["zero_delta_events"]) for row in comparisons),
        "context_snapshot_hash_verified": all(
            bool(arm.get("context_snapshot_hash_verified", False))
            for arm in [target, *control_arms]
        ),
        "live_prefix_replay_exact": True,
        "protocol_exact_match": True,
        "protocol_substitution_detected": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "support": 0,
        "truth_status": SAGE7B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "raw_events_counted_as_scientific_support": False,
        "observation_counted_as_confirmation": False,
    }


def exact_protocol_substitution_reasons(
    *,
    request: Mapping[str, Any],
    target_arm: Mapping[str, Any],
    control_arms: Sequence[Mapping[str, Any]],
) -> List[str]:
    reasons: List[str] = []
    expected_controls = [
        dict(row)
        for row in request.get("pre_registered_parameterized_control_variants", [])
        or []
        if isinstance(row, Mapping)
    ]
    if str(target_arm.get("action", "")) != str(request.get("target_action", "")):
        reasons.append("target_action")
    if _canonical_json(target_arm.get("action_args", {}) or {}) != _canonical_json(
        request.get("target_action_args", {}) or {}
    ):
        reasons.append("target_action_args")
    if len(control_arms) != len(expected_controls):
        reasons.append("control_count")
    for index, expected in enumerate(expected_controls):
        if index >= len(control_arms):
            break
        observed = control_arms[index]
        if str(observed.get("action", "")) != str(expected.get("action", "")):
            reasons.append(f"control_{index + 1}_action")
        if _canonical_json(observed.get("action_args", {}) or {}) != _canonical_json(
            expected.get("action_args", {}) or {}
        ):
            reasons.append(f"control_{index + 1}_action_args")
    return reasons


def arm_measurement_signature(arm: Mapping[str, Any], *, metric: str) -> Dict[str, Any]:
    measurement = dict(arm.get("measurement_for_delta", {}) or {})
    return {
        "metric": metric,
        "metric_signal": metric_signal(measurement, metric),
        "changed": bool(measurement.get("changed", False)),
        "changed_pixels": int(measurement.get("changed_pixels", 0) or 0),
        "local_patch_available": bool(measurement.get("local_patch_available", False)),
        "local_changed_pixels": int(measurement.get("local_changed_pixels", 0) or 0),
        "terminal_state_after_rollout": bool(
            measurement.get("terminal_state_after_rollout", False)
        ),
        "observed_signal_source": str(measurement.get("observed_signal_source", "")),
    }


def build_sage7b_budget_records(
    *,
    budgets: Sequence[int],
    requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for budget in budgets:
        request_rows = [
            row for row in requests if budget_from_request(row) == int(budget)
        ]
        request_ids = {str(row.get("request_id", "")) for row in request_rows}
        executed = [
            row for row in experiments if str(row.get("request_id", "")) in request_ids
        ]
        blocked_rows = [
            row for row in blocked if str(row.get("request_id", "")) in request_ids
        ]
        comparisons = [
            comparison
            for row in executed
            for comparison in row.get("parameterized_comparisons", []) or []
        ]
        records.append(
            {
                "budget": int(budget),
                "requests_available": len(request_rows),
                "requests_executed": len(executed),
                "requests_blocked": len(blocked_rows),
                "target_arm_executions": sum(
                    int(row.get("target_executions", 0) or 0) for row in executed
                ),
                "control_arm_executions": sum(
                    int(row.get("control_executions", 0) or 0) for row in executed
                ),
                "comparison_events": len(comparisons),
                "positive_delta_events": sum(
                    int(row.get("positive_delta_events", 0) or 0) for row in comparisons
                ),
                "negative_delta_events": sum(
                    int(row.get("negative_delta_events", 0) or 0) for row in comparisons
                ),
                "zero_delta_events": sum(
                    int(row.get("zero_delta_events", 0) or 0) for row in comparisons
                ),
                "exact_protocol_events": sum(
                    bool(row.get("protocol_exact_match", False)) for row in executed
                ),
                "exact_live_prefix_replays": sum(
                    bool(row.get("live_prefix_replay_exact", False)) for row in executed
                ),
                "support": 0,
                "truth_status": SAGE7B_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
            }
        )
    return records


def build_sage7b_gate(
    *,
    source: Mapping[str, Any],
    game_id: str,
    budgets: Sequence[int],
    requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    expected_requests = len(budgets) * DEFAULT_REQUESTS_PER_BUDGET
    expected_controls = expected_requests * DEFAULT_CONTROLS_PER_REQUEST
    expected_comparisons = expected_controls
    source_ids = {str(row.get("request_id", "")) for row in requests}
    executed_ids = {str(row.get("request_id", "")) for row in experiments}
    comparisons = [
        comparison
        for row in experiments
        for comparison in row.get("parameterized_comparisons", []) or []
    ]
    comparison_ids = [str(row.get("comparison_id", "")) for row in comparisons]
    return {
        "source_sage7a_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        ),
        "source_requests_complete": len(requests) == expected_requests,
        "all_source_requests_executed": bool(requests)
        and not blocked
        and source_ids == executed_ids,
        "all_experiments_scoped_to_selected_game": all(
            str(row.get("game_id", "")) == game_id for row in experiments
        ),
        "target_arm_count_exact": sum(
            int(row.get("target_executions", 0) or 0) for row in experiments
        )
        == expected_requests,
        "control_arm_count_exact": sum(
            int(row.get("control_executions", 0) or 0) for row in experiments
        )
        == expected_controls,
        "comparison_count_exact": len(comparisons) == expected_comparisons,
        "comparison_ids_unique": "" not in comparison_ids
        and len(comparison_ids) == len(set(comparison_ids)),
        "all_live_prefix_replays_exact": bool(experiments)
        and all(
            bool(row.get("live_prefix_replay_exact", False))
            and bool(row.get("context_snapshot_hash_verified", False))
            for row in experiments
        ),
        "all_protocols_exact_without_substitution": bool(experiments)
        and all(
            bool(row.get("protocol_exact_match", False))
            and not bool(row.get("protocol_substitution_detected", True))
            for row in experiments
        ),
        "single_action6_family_preserved": bool(experiments)
        and all(
            str(row.get("target_action", "")) == "ACTION6"
            and str(row.get("parameterized_action_family", "")) == "ACTION6"
            and all(
                str(control.get("action", "")) == "ACTION6"
                for control in row.get(
                    "pre_registered_parameterized_control_variants", []
                )
                or []
            )
            for row in experiments
        ),
        "parameterized_variants_not_relabelled_as_actions": all(
            not bool(
                row.get("parameterized_controls_counted_as_distinct_actions", True)
            )
            for row in experiments
        ),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0 for row in experiments
        )
        and all(int(row.get("support", 0) or 0) == 0 for row in comparisons),
        "source_registry_quarantine_preserved": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        )
        == 0
        and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
        and not bool(source.get("scope_generalization_performed", False)),
    }


def summarize_sage7b(
    *,
    source: Mapping[str, Any],
    game_id: str,
    budgets: Sequence[int],
    requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    outcome: str,
) -> Dict[str, Any]:
    comparisons = [
        comparison
        for row in experiments
        for comparison in row.get("parameterized_comparisons", []) or []
    ]
    statuses = Counter(str(row.get("discrimination_status", "")) for row in comparisons)
    metrics = Counter(str(row.get("metric", "")) for row in comparisons)
    effects = [
        float(row.get("controlled_delta", {}).get("effect_size", 0.0) or 0.0)
        for row in comparisons
    ]
    ready = outcome == SAGE7B_EXECUTION_COMPLETED and bool(comparisons)
    return {
        "game_id": game_id,
        "source_sage7a_outcome_status": str(source.get("outcome_status", "")),
        "budgets_evaluated": list(budgets),
        "requests_available": len(requests),
        "requests_executed": len(experiments),
        "requests_blocked": len(blocked),
        "target_arm_executions": sum(
            int(row.get("target_executions", 0) or 0) for row in experiments
        ),
        "control_arm_executions": sum(
            int(row.get("control_executions", 0) or 0) for row in experiments
        ),
        "total_arm_executions": sum(
            int(row.get("total_arm_executions", 0) or 0) for row in experiments
        ),
        "comparison_events": len(comparisons),
        "requests_executed_by_budget": {
            str(key): value
            for key, value in sorted(
                Counter(int(row.get("budget", 0) or 0) for row in experiments).items()
            )
        },
        "metrics_executed": dict(sorted(metrics.items())),
        "discrimination_statuses": dict(sorted(statuses.items())),
        "positive_delta_events": sum(value > 0 for value in effects),
        "negative_delta_events": sum(value < 0 for value in effects),
        "zero_delta_events": sum(value == 0 for value in effects),
        "max_effect_size": max(effects, default=0.0),
        "min_effect_size": min(effects, default=0.0),
        "distinct_effect_sizes": sorted(set(effects)),
        "live_prefix_replay_exact_events": sum(
            bool(row.get("live_prefix_replay_exact", False)) for row in experiments
        ),
        "protocol_exact_match_events": sum(
            bool(row.get("protocol_exact_match", False)) for row in experiments
        ),
        "protocol_substitution_events": sum(
            bool(row.get("protocol_substitution_detected", False))
            for row in experiments
        ),
        "action_families": sorted(
            {str(row.get("parameterized_action_family", "")) for row in experiments}
        ),
        "distinct_action_families": len(
            {str(row.get("parameterized_action_family", "")) for row in experiments}
        ),
        "parameterized_variants_counted_as_distinct_actions": False,
        "ready_for_event_consolidation": ready,
        "required_next_step": (
            SAGE7B_CONSOLIDATION_REQUIRED
            if ready
            else SAGE7B_NO_CONSOLIDATION_REQUESTED
        ),
        "gate_passed": outcome == SAGE7B_EXECUTION_COMPLETED,
        "outcome_status": outcome,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "support": 0,
        "truth_status": SAGE7B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def build_source_sage7a_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(source.get("summary", {}) or {})
    return {
        "source_outcome_status": str(source.get("outcome_status", "")),
        "game_id": str(summary.get("game_id", "")),
        "budgets": list(summary.get("budgets_evaluated", []) or []),
        "effective_requests_generated": int(
            summary.get("effective_requests_generated", 0) or 0
        ),
        "parameterized_control_variants_pre_registered": int(
            summary.get("parameterized_control_variants_pre_registered", 0) or 0
        ),
        "ready_for_parameterized_m3_execution": bool(
            summary.get("ready_for_parameterized_m3_execution", False)
        ),
        "required_next_step": str(summary.get("required_next_step", "")),
        "parameterized_variants_counted_as_distinct_actions": False,
        "source_scoped_mechanics_reused": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            source.get("cross_game_mechanics_imported", 0) or 0
        ),
        "source_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE7B_TRUTH_STATUS,
    }


def validate_sage7b_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE7A_SCHEMA_VERSION:
        raise ValueError("SAGE.7a schema version is not supported by SAGE.7b")
    if str(source.get("outcome_status", "")) != SAGE7A_FRONTIER_GENERATED:
        raise ValueError("SAGE.7b requires the completed SAGE.7a frontier")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.7a source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE7A_TRUTH_STATUS:
        raise ValueError("SAGE.7a truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.7a source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.7a support must remain 0")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.7a cannot perform a scientific verdict")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.7a cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.7a wrong_confirmations must remain 0")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
        or bool(source.get("scope_generalization_performed", False))
    ):
        raise ValueError("SAGE.7a cannot import or generalize quarantined mechanics")
    if bool(source.get("parameterized_controls_counted_as_distinct_actions", True)):
        raise ValueError("SAGE.7a controls cannot be relabelled as distinct actions")
    if (
        int(config.get("requests_per_budget", 0) or 0) != DEFAULT_REQUESTS_PER_BUDGET
        or int(config.get("controls_per_request", 0) or 0)
        != DEFAULT_CONTROLS_PER_REQUEST
        or not bool(config.get("control_selection_occurs_before_m3_execution", False))
        or bool(config.get("outcome_metrics_used_for_control_selection", True))
        or bool(config.get("parameterized_variants_are_distinct_action_families", True))
    ):
        raise ValueError("SAGE.7a pre-registered control design must remain fixed")
    if not bool(summary.get("gate_passed", False)) or not bool(
        summary.get("ready_for_parameterized_m3_execution", False)
    ):
        raise ValueError("SAGE.7a execution gate must pass before SAGE.7b")
    if (
        str(summary.get("required_next_step", ""))
        != SAGE7A_PARAMETERIZED_EXECUTION_REQUIRED
    ):
        raise ValueError("SAGE.7a must explicitly request parameterized execution")
    budgets = [int(value) for value in summary.get("budgets_evaluated", []) or []]
    if budgets != [50, 150, 300]:
        raise ValueError("SAGE.7b requires the fixed SAGE.7a budgets")
    requests = [
        row
        for row in source.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping)
    ]
    expected = len(budgets) * DEFAULT_REQUESTS_PER_BUDGET
    if len(requests) != expected or len(requests) != int(
        summary.get("effective_requests_generated", 0) or 0
    ):
        raise ValueError("SAGE.7a request count must be exact")
    if len({str(row.get("request_id", "")) for row in requests}) != len(requests):
        raise ValueError("SAGE.7a request ids must be unique")
    for request in requests:
        validation = validate_m3_request(request)
        if not validation.valid:
            raise ValueError(f"invalid SAGE.7a request: {validation.errors}")
        controls = (
            request.get("pre_registered_parameterized_control_variants", []) or []
        )
        if str(request.get("status", "")) != "READY_FOR_M3":
            raise ValueError("every SAGE.7a request must be ready for M3")
        if (
            str(request.get("replayability", "")) != "LIVE_PREFIX_REPLAY_CONTEXT"
            or str(request.get("context_state_origin", ""))
            != EXPECTED_CONTEXT_STATE_ORIGIN
        ):
            raise ValueError("every SAGE.7a request must preserve exact live replay")
        if str(request.get("generated_by", "")) != EXPECTED_GENERATOR:
            raise ValueError("SAGE.7a generator identity must be preserved")
        if (
            str(request.get("parameterized_control_policy", ""))
            != PARAMETERIZED_CONTROL_POLICY
        ):
            raise ValueError("SAGE.7a parameterized control policy must be exact")
        if (
            str(request.get("target_action", "")) != "ACTION6"
            or str(request.get("parameterized_action_family", "")) != "ACTION6"
        ):
            raise ValueError("SAGE.7b requires one ACTION6 family")
        if not request.get("target_action_args"):
            raise ValueError("SAGE.7a target args must be explicit")
        if len(controls) != DEFAULT_CONTROLS_PER_REQUEST:
            raise ValueError("SAGE.7a must pre-register exactly two controls")
        target_key = _canonical_json(request.get("target_action_args", {}) or {})
        control_keys = [
            _canonical_json(control.get("action_args", {}) or {})
            for control in controls
        ]
        if len(control_keys) != len(set(control_keys)):
            raise ValueError("SAGE.7a parameterized controls must be unique")
        protocol = dict(request.get("parameterized_control_protocol", {}) or {})
        if (
            str(protocol.get("selection_timing", ""))
            != "BEFORE_M3_EXECUTION_AND_OUTCOME_OBSERVATION"
            or not bool(protocol.get("same_action_family_required", False))
            or bool(protocol.get("variants_counted_as_distinct_actions", True))
            or bool(protocol.get("variant_substitution_allowed", True))
        ):
            raise ValueError("SAGE.7a request protocol must remain pre-registered")
        for control in controls:
            if (
                str(control.get("action", "")) != "ACTION6"
                or not control.get("action_args")
                or _canonical_json(control.get("action_args", {}) or {}) == target_key
                or not bool(control.get("live_legal_at_context", False))
                or bool(control.get("counted_as_distinct_action", True))
            ):
                raise ValueError("SAGE.7a parameterized controls must remain exact")
    gate = dict(source.get("gate", {}) or {})
    if not gate or not all(bool(value) for value in gate.values()):
        raise ValueError("every SAGE.7a source gate must pass")


def write_sage7b_parameterized_execution(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def budget_from_request(row: Mapping[str, Any]) -> int:
    transition_id = str(row.get("source_transition_id", ""))
    marker = "::budget_"
    try:
        return int(transition_id.split(marker, 1)[1].split("::", 1)[0])
    except (IndexError, ValueError):
        return 0


def request_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        budget_from_request(row),
        int(row.get("source_step", 0) or 0),
        str(row.get("request_id", "")),
    )


def _blocked_result(
    request: Mapping[str, Any],
    *,
    reason: str,
    target_arm: Mapping[str, Any] | None = None,
    control_arms: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    return {
        "execution_status": "BLOCKED",
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "source_transition_id": str(request.get("source_transition_id", "")),
        "game_id": str(request.get("game_id", "")),
        "budget": budget_from_request(request),
        "source_step": int(request.get("source_step", 0) or 0),
        "execution_mode": EXECUTION_MODE,
        "blocked_reason": reason,
        "target_arm": dict(target_arm or {}),
        "control_arms": [dict(row) for row in control_arms],
        "parameterized_comparisons": [],
        "support": 0,
        "truth_status": SAGE7B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "raw_events_counted_as_scientific_support": False,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-sage7a", default=str(DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH)
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH)
    )
    args = parser.parse_args(argv)
    payload = run_sage7b_parameterized_execution(
        source_sage7a_path=args.source_sage7a,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
