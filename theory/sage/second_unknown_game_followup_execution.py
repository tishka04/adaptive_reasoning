"""SAGE.6e exact execution of the second-game pre-registered followups."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from theory.non_ar25_active_micro_run import _env_dir

from .live_mini_frontier_m3_executor import (
    EnvFactory,
    execute_live_prefix_mini_frontier_request,
)
from .second_unknown_game_handoff_compiler import (
    CONTROL_DIVERSITY_REQUEST,
    DEFAULT_SAGE6D_HANDOFF_PATH,
    NEUTRAL_REPLICATION_REQUEST,
    SAGE6D_HANDOFF_COMPILED,
    SAGE6D_SCHEMA_VERSION,
    SAGE6D_TRUTH_STATUS,
)


DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH = (
    Path("diagnostics") / "sage" / "sage6e_second_unknown_game_followup_execution.json"
)

SAGE6E_SCHEMA_VERSION = "sage.second_unknown_game_followup_execution.v1"
SAGE6E_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_6E"
SAGE6E_EXECUTION_COMPLETED = (
    "SAGE_SECOND_UNKNOWN_GAME_FOLLOWUPS_EXECUTED_CANDIDATE_ONLY"
)
SAGE6E_EXECUTION_INCOMPLETE = (
    "SAGE_SECOND_UNKNOWN_GAME_FOLLOWUP_EXECUTION_INCOMPLETE_CANDIDATE_ONLY"
)

CONTROL_EFFECT_POSITIVE = "CONTROL_EFFECT_POSITIVE_CANDIDATE_ONLY"
CONTROL_EFFECT_NONPOSITIVE = "CONTROL_EFFECT_NONPOSITIVE_CANDIDATE_ONLY"
NEUTRAL_REPLICATION_MATCHED = "NEUTRAL_CONTEXT_REPLICATION_MATCHED_CANDIDATE_ONLY"
NEUTRAL_REPLICATION_DEVIATED = "NEUTRAL_CONTEXT_REPLICATION_DEVIATED_CANDIDATE_ONLY"
CONTROL_SENSITIVE_PATTERN = (
    "DISTINCT_CONTROL_NULLIFIES_PRIOR_POSITIVE_EFFECT_ACROSS_ALL_BUDGETS_"
    "WITH_NEUTRAL_REPLICATION_CANDIDATE_ONLY"
)
MIXED_FOLLOWUP_PATTERN = "MIXED_PRE_REGISTERED_FOLLOWUP_RESULTS_CANDIDATE_ONLY"


def run_sage6e_second_unknown_game_followup_execution(
    *,
    source_sage6d_path: str | Path = DEFAULT_SAGE6D_HANDOFF_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute every SAGE.6d protocol in its pre-registered order."""
    source = _load_json(source_sage6d_path)
    validate_sage6e_source(source)
    protocols = [
        dict(row)
        for row in source.get("pre_registered_followup_protocols", []) or []
        if isinstance(row, Mapping)
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    executed: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for protocol in protocols:
        result = execute_sage6e_followup_protocol(
            protocol,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        if str(result.get("execution_status", "")) == "EXECUTED":
            executed.append(result)
        else:
            blocked.append(result)

    assessment = assess_sage6e_results(protocols=protocols, executed=executed)
    per_budget = build_sage6e_budget_records(
        protocols=protocols,
        executed=executed,
        blocked=blocked,
    )
    audit = build_sage6e_execution_audit(protocols)
    gate = build_sage6e_gate(
        source=source,
        protocols=protocols,
        executed=executed,
        blocked=blocked,
        assessment=assessment,
        audit=audit,
    )
    outcome = (
        SAGE6E_EXECUTION_COMPLETED
        if gate and all(gate.values())
        else SAGE6E_EXECUTION_INCOMPLETE
    )
    summary = summarize_sage6e(
        source=source,
        protocols=protocols,
        executed=executed,
        blocked=blocked,
        assessment=assessment,
        gate=gate,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE6E_SCHEMA_VERSION,
            "source_sage6d_path": str(source_sage6d_path),
            "game_id": str(summary.get("game_id", "")),
            "budgets": list(summary.get("budgets", []) or []),
            "environments_dir": str(env_dir),
            "execution_mode": "live_prefix_replay_context",
            "protocol_selection_policy": (
                "EXECUTE_ALL_SOURCE_PROTOCOLS_IN_PRE_REGISTERED_ORDER"
            ),
            "protocol_substitution_allowed": False,
            "context_substitution_allowed": False,
            "budget_substitution_allowed": False,
            "target_action_substitution_allowed": False,
            "control_action_substitution_allowed": False,
            "context_snapshot_hash_verification_required": True,
            "target_and_control_replay_same_prefix": True,
            "benchmark_run": False,
            "inputs_read": ["SAGE.6d"],
            "scientific_policy": {
                "raw_events_are_candidate_only": True,
                "pre_registered_consistency_is_not_confirmation": True,
                "pre_registered_deviation_is_not_refutation": True,
                "support_events_are_not_scientific_support": True,
                "control_sensitive_pattern_is_not_scientific_verdict": True,
                "a32_a33_write_performed": False,
            },
            "artifacts_not_modified": [
                "SAGE.6d",
                "SAGE.6c",
                "SAGE.6b",
                "SAGE.6a",
                "SAGE.6",
                "SAGE.5",
                "A32",
                "A33",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
        },
        "source_sage6d_context": build_source_sage6d_context(source),
        "execution_audit": audit,
        "pre_registered_protocols_executed": protocols,
        "controlled_followup_results": executed,
        "blocked_followup_results": blocked,
        "per_budget_results": per_budget,
        "pre_registered_outcome_assessment": assessment,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE6E_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "support_events_counted_as_support": False,
        "contradiction_events_counted_as_refutation": False,
        "pre_registered_consistency_counted_as_confirmation": False,
        "pre_registered_deviation_counted_as_refutation": False,
        "control_sensitive_pattern_counted_as_scientific_verdict": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage6e_second_unknown_game_followup_execution(payload, output_path)
    return payload


def execute_sage6e_followup_protocol(
    protocol: Mapping[str, Any],
    *,
    environments_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute one protocol with its exact pre-registered target and control."""
    request = protocol_as_explicit_m3_request(protocol)
    raw = execute_live_prefix_mini_frontier_request(
        request,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    result = dict(raw)
    result.update(
        {
            "protocol_id": str(protocol.get("protocol_id", "")),
            "handoff_id": str(protocol.get("handoff_id", "")),
            "request_type": str(protocol.get("request_type", "")),
            "source_context_cluster_id": str(
                protocol.get("source_context_cluster_id", "")
            ),
            "source_event_id": str(protocol.get("source_event_id", "")),
            "source_request_id": str(protocol.get("source_request_id", "")),
            "budget": int(protocol.get("budget", 0) or 0),
            "source_step": int(protocol.get("source_step", 0) or 0),
            "pre_registered_target_action": str(protocol.get("target_action", "")),
            "pre_registered_control_action": str(protocol.get("control_action", "")),
            "prior_control_action": str(protocol.get("prior_control_action", "")),
            "expected_prior_effect_size": float(
                protocol.get("expected_prior_effect_size", 0.0) or 0.0
            ),
            "pre_registered_interpretation": dict(
                protocol.get("pre_registered_interpretation", {}) or {}
            ),
            "executed_by": "SAGE.6e_exact_pre_registered_followup_execution",
            "protocol_substitution_performed": False,
            "context_substitution_performed": False,
            "budget_substitution_performed": False,
            "target_action_substitution_performed": False,
            "control_action_substitution_performed": False,
            "support": 0,
            "truth_status": SAGE6E_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "support_events_counted_as_support": False,
            "contradiction_event_counted_as_refutation": False,
            "result_counted_as_confirmation": False,
            "result_counted_as_scientific_verdict": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        }
    )
    if str(result.get("execution_status", "")) != "EXECUTED":
        result.update(
            {
                "protocol_execution_exact": False,
                "pre_registered_condition_met": False,
                "pre_registered_deviation_detected": False,
                "pre_registered_result_status": "NOT_EVALUATED_EXECUTION_BLOCKED",
            }
        )
        return result

    effect_size = float(
        result.get("controlled_delta", {}).get("effect_size", 0.0) or 0.0
    )
    condition_met = pre_registered_condition_met(protocol, effect_size)
    result_status = classify_pre_registered_result(protocol, effect_size)
    exact = bool(
        result.get("live_prefix_replay_exact", False)
        and str(result.get("target_action", ""))
        == str(protocol.get("target_action", ""))
        and str(result.get("control_action", ""))
        == str(protocol.get("control_action", ""))
        and str(result.get("context_snapshot_hash", ""))
        == str(protocol.get("context_snapshot_hash", ""))
        and list(result.get("context_replay", []) or [])
        == list(protocol.get("context_replay", []) or [])
        and list(result.get("context_replay_args", []) or [])
        == list(protocol.get("context_replay_args", []) or [])
    )
    result.update(
        {
            "effect_size": effect_size,
            "protocol_execution_exact": exact,
            "pre_registered_condition_met": condition_met,
            "pre_registered_deviation_detected": not condition_met,
            "pre_registered_result_status": result_status,
        }
    )
    return result


def protocol_as_explicit_m3_request(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Adapt one immutable protocol to the public live-prefix executor contract."""
    return {
        "request_id": str(protocol.get("source_request_id", "")),
        "source_hypothesis_id": str(protocol.get("source_hypothesis_id", "")),
        "source_transition_id": str(protocol.get("source_transition_id", "")),
        "game_id": str(protocol.get("game_id", "")),
        "metric": str(protocol.get("metric", "")),
        "hypothesis_family": str(protocol.get("hypothesis_family", "")),
        "context_replay": list(protocol.get("context_replay", []) or []),
        "context_replay_args": [
            dict(row or {}) for row in protocol.get("context_replay_args", []) or []
        ],
        "context_snapshot_hash": str(protocol.get("context_snapshot_hash", "")),
        "target_action": str(protocol.get("target_action", "")),
        "target_action_args": (
            dict(protocol.get("target_action_args", {}) or {})
            if protocol.get("target_action_args") is not None
            else None
        ),
        "suggested_control_actions": [str(protocol.get("control_action", ""))],
    }


def pre_registered_condition_met(
    protocol: Mapping[str, Any], effect_size: float
) -> bool:
    request_type = str(protocol.get("request_type", ""))
    if request_type == CONTROL_DIVERSITY_REQUEST:
        return float(effect_size) > 0.0
    if request_type == NEUTRAL_REPLICATION_REQUEST:
        return float(effect_size) == 0.0
    raise ValueError(f"unsupported SAGE.6e request type: {request_type}")


def classify_pre_registered_result(
    protocol: Mapping[str, Any], effect_size: float
) -> str:
    request_type = str(protocol.get("request_type", ""))
    if request_type == CONTROL_DIVERSITY_REQUEST:
        return (
            CONTROL_EFFECT_POSITIVE if effect_size > 0.0 else CONTROL_EFFECT_NONPOSITIVE
        )
    if request_type == NEUTRAL_REPLICATION_REQUEST:
        return (
            NEUTRAL_REPLICATION_MATCHED
            if effect_size == 0.0
            else NEUTRAL_REPLICATION_DEVIATED
        )
    raise ValueError(f"unsupported SAGE.6e request type: {request_type}")


def assess_sage6e_results(
    *,
    protocols: Sequence[Mapping[str, Any]],
    executed: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    protocol_by_id = {str(row.get("protocol_id", "")): row for row in protocols}
    diversity = [
        row
        for row in executed
        if str(row.get("request_type", "")) == CONTROL_DIVERSITY_REQUEST
    ]
    neutral = [
        row
        for row in executed
        if str(row.get("request_type", "")) == NEUTRAL_REPLICATION_REQUEST
    ]
    diversity_effects = [float(row.get("effect_size", 0.0) or 0.0) for row in diversity]
    prior_effects = [
        float(
            protocol_by_id.get(str(row.get("protocol_id", "")), {}).get(
                "expected_prior_effect_size", 0.0
            )
            or 0.0
        )
        for row in diversity
    ]
    all_diversity_zero = bool(diversity) and all(
        value == 0.0 for value in diversity_effects
    )
    all_prior_positive = bool(prior_effects) and all(
        value > 0.0 for value in prior_effects
    )
    neutral_matched = (
        len(neutral) == 1 and float(neutral[0].get("effect_size", 1.0) or 0.0) == 0.0
    )
    control_sensitive = bool(
        len(diversity) == 3
        and all_diversity_zero
        and all_prior_positive
        and neutral_matched
    )
    return {
        "assessment_id": "sage6e::pre_registered_outcome_assessment::001",
        "game_id": str(executed[0].get("game_id", "")) if executed else "",
        "control_diversity_protocols_executed": len(diversity),
        "control_diversity_budgets": sorted(
            {int(row.get("budget", 0) or 0) for row in diversity}
        ),
        "prior_control_action": "ACTION1",
        "distinct_control_action": "ACTION3",
        "prior_control_effect_sizes": prior_effects,
        "distinct_control_effect_sizes": diversity_effects,
        "distinct_control_matches_target_across_all_budgets": all_diversity_zero,
        "prior_positive_effect_was_control_dependent_candidate_only": (
            control_sensitive
        ),
        "neutral_replication_protocols_executed": len(neutral),
        "neutral_replication_effect_size": (
            float(neutral[0].get("effect_size", 0.0) or 0.0)
            if len(neutral) == 1
            else None
        ),
        "neutral_context_replication_matched": neutral_matched,
        "pre_registered_conditions_met": sum(
            1
            for row in executed
            if bool(row.get("pre_registered_condition_met", False))
        ),
        "pre_registered_deviations": sum(
            1
            for row in executed
            if bool(row.get("pre_registered_deviation_detected", False))
        ),
        "candidate_status": (
            CONTROL_SENSITIVE_PATTERN if control_sensitive else MIXED_FOLLOWUP_PATTERN
        ),
        "ready_for_post_execution_consolidation": len(executed) == len(protocols)
        and bool(protocols),
        "ready_for_A32_review": False,
        "ready_for_A32_review_blocked_reason": (
            "POST_EXECUTION_CONTEXT_PRESERVING_CONSOLIDATION_REQUIRED"
        ),
        "required_followups": [
            "CONSOLIDATE_CONTROL_DEPENDENT_EFFECTS",
            "PRESERVE_ACTION1_ACTION3_CONTROL_IDENTITIES",
            "REASSESS_A32_ELIGIBILITY_AFTER_CONSOLIDATION",
        ],
        "control_sensitive_pattern_counted_as_scientific_verdict": False,
        "pre_registered_deviations_counted_as_refutations": False,
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": SAGE6E_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def build_sage6e_budget_records(
    *,
    protocols: Sequence[Mapping[str, Any]],
    executed: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    budgets = sorted({int(row.get("budget", 0) or 0) for row in protocols})
    records: List[Dict[str, Any]] = []
    for budget in budgets:
        planned = [row for row in protocols if int(row.get("budget", 0) or 0) == budget]
        completed = [
            row for row in executed if int(row.get("budget", 0) or 0) == budget
        ]
        failures = [row for row in blocked if int(row.get("budget", 0) or 0) == budget]
        records.append(
            {
                "budget": budget,
                "protocol_ids": [str(row.get("protocol_id", "")) for row in planned],
                "protocols_planned": len(planned),
                "protocols_executed": len(completed),
                "protocols_blocked": len(failures),
                "target_actions_executed": _counts(
                    row.get("target_action", "") for row in completed
                ),
                "control_actions_executed": _counts(
                    row.get("control_action", "") for row in completed
                ),
                "effect_sizes": [
                    float(row.get("effect_size", 0.0) or 0.0) for row in completed
                ],
                "all_protocols_exact": len(completed) == len(planned)
                and all(
                    bool(row.get("protocol_execution_exact", False))
                    for row in completed
                ),
                "support": 0,
                "truth_status": SAGE6E_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return records


def build_sage6e_execution_audit(
    protocols: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "execution_rank": index,
            "protocol_id": str(row.get("protocol_id", "")),
            "source_request_id": str(row.get("source_request_id", "")),
            "budget": int(row.get("budget", 0) or 0),
            "source_step": int(row.get("source_step", 0) or 0),
            "target_action": str(row.get("target_action", "")),
            "control_action": str(row.get("control_action", "")),
            "selected": True,
            "selection_reason": "PRE_REGISTERED_SOURCE_ORDER",
            "outcome_metrics_read_for_selection": False,
            "substitution_allowed": False,
            "support": 0,
            "truth_status": SAGE6E_TRUTH_STATUS,
        }
        for index, row in enumerate(protocols, start=1)
    ]


def build_sage6e_gate(
    *,
    source: Mapping[str, Any],
    protocols: Sequence[Mapping[str, Any]],
    executed: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
    audit: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    protocol_ids = [str(row.get("protocol_id", "")) for row in protocols]
    executed_ids = [str(row.get("protocol_id", "")) for row in executed]
    source_budgets = {
        int(value) for value in source.get("summary", {}).get("budgets", []) or []
    }
    diversity = [
        row
        for row in executed
        if str(row.get("request_type", "")) == CONTROL_DIVERSITY_REQUEST
    ]
    neutral = [
        row
        for row in executed
        if str(row.get("request_type", "")) == NEUTRAL_REPLICATION_REQUEST
    ]
    raw_total = sum(
        int(row.get("support_events", 0) or 0)
        + int(row.get("contradiction_events", 0) or 0)
        + int(row.get("neutral_events", 0) or 0)
        for row in executed
    )
    return {
        "source_sage6d_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source.get("gate", {}).values()),
        "all_pre_registered_protocols_selected_in_source_order": len(audit)
        == len(protocols)
        and [str(row.get("protocol_id", "")) for row in audit] == protocol_ids
        and all(
            bool(row.get("selected", False))
            and not bool(row.get("outcome_metrics_read_for_selection", True))
            for row in audit
        ),
        "all_protocols_executed_without_block": bool(protocols)
        and len(executed) == len(protocols)
        and not blocked,
        "executed_protocol_ids_match_source_exactly": protocol_ids == executed_ids
        and "" not in protocol_ids
        and len(protocol_ids) == len(set(protocol_ids)),
        "all_replays_exact_and_hash_verified": bool(executed)
        and all(
            bool(row.get("live_prefix_replay_exact", False))
            and bool(row.get("target_context_signature_verified", False))
            and bool(row.get("control_context_signature_verified", False))
            for row in executed
        ),
        "no_protocol_context_budget_or_action_substitution": bool(executed)
        and all(
            bool(row.get("protocol_execution_exact", False))
            and not bool(row.get("protocol_substitution_performed", True))
            and not bool(row.get("context_substitution_performed", True))
            and not bool(row.get("budget_substitution_performed", True))
            and not bool(row.get("target_action_substitution_performed", True))
            and not bool(row.get("control_action_substitution_performed", True))
            for row in executed
        ),
        "distinct_control_executed_once_per_budget": len(diversity)
        == len(source_budgets)
        and {int(row.get("budget", 0) or 0) for row in diversity} == source_budgets
        and {str(row.get("control_action", "")) for row in diversity} == {"ACTION3"},
        "neutral_context_replication_executed_once": len(neutral) == 1
        and str(neutral[0].get("control_action", "")) == "ACTION1",
        "pre_registered_interpretations_applied_without_verdict": all(
            str(row.get("pre_registered_result_status", ""))
            in {
                CONTROL_EFFECT_POSITIVE,
                CONTROL_EFFECT_NONPOSITIVE,
                NEUTRAL_REPLICATION_MATCHED,
                NEUTRAL_REPLICATION_DEVIATED,
            }
            and not bool(row.get("result_counted_as_confirmation", True))
            and not bool(row.get("result_counted_as_scientific_verdict", True))
            for row in executed
        ),
        "raw_event_accounting_exact": raw_total == len(executed),
        "ready_for_consolidation_but_not_a32_review": bool(
            assessment.get("ready_for_post_execution_consolidation", False)
        )
        and not bool(assessment.get("ready_for_A32_review", True)),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0
            and str(row.get("truth_status", "")) == SAGE6E_TRUTH_STATUS
            for row in [*executed, assessment, *audit]
        ),
        "source_registry_quarantine_preserved": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        )
        == 0
        and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
        and not bool(source.get("scope_generalization_performed", False)),
    }


def summarize_sage6e(
    *,
    source: Mapping[str, Any],
    protocols: Sequence[Mapping[str, Any]],
    executed: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
    gate: Mapping[str, bool],
    outcome: str,
) -> Dict[str, Any]:
    effects = [float(row.get("effect_size", 0.0) or 0.0) for row in executed]
    return {
        "source_sage6d_outcome_status": str(source.get("outcome_status", "")),
        "game_id": str(source.get("summary", {}).get("game_id", "")),
        "budgets": list(source.get("summary", {}).get("budgets", []) or []),
        "protocols_available": len(protocols),
        "protocols_selected": len(protocols),
        "protocols_executed": len(executed),
        "protocols_blocked": len(blocked),
        "protocols_executed_by_budget": _counts(
            int(row.get("budget", 0) or 0) for row in executed
        ),
        "control_diversity_protocols_executed": sum(
            1
            for row in executed
            if str(row.get("request_type", "")) == CONTROL_DIVERSITY_REQUEST
        ),
        "neutral_replication_protocols_executed": sum(
            1
            for row in executed
            if str(row.get("request_type", "")) == NEUTRAL_REPLICATION_REQUEST
        ),
        "live_prefix_replay_exact_events": sum(
            1 for row in executed if bool(row.get("live_prefix_replay_exact", False))
        ),
        "context_snapshot_hash_verified_events": sum(
            1
            for row in executed
            if bool(row.get("target_context_signature_verified", False))
            and bool(row.get("control_context_signature_verified", False))
        ),
        "protocol_execution_exact_events": sum(
            1 for row in executed if bool(row.get("protocol_execution_exact", False))
        ),
        "target_actions_executed": _counts(
            row.get("target_action", "") for row in executed
        ),
        "control_actions_executed": _counts(
            row.get("control_action", "") for row in executed
        ),
        "target_signal_total": sum(
            float(row.get("target_signal", 0.0) or 0.0) for row in executed
        ),
        "control_signal_total": sum(
            float(row.get("control_signal", 0.0) or 0.0) for row in executed
        ),
        "controlled_effect_sizes": effects,
        "positive_effect_events": sum(value > 0.0 for value in effects),
        "negative_effect_events": sum(value < 0.0 for value in effects),
        "zero_effect_events": sum(value == 0.0 for value in effects),
        "raw_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in executed
        ),
        "raw_contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in executed
        ),
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in executed
        ),
        "pre_registered_conditions_met": int(
            assessment.get("pre_registered_conditions_met", 0) or 0
        ),
        "pre_registered_deviations": int(
            assessment.get("pre_registered_deviations", 0) or 0
        ),
        "distinct_control_matches_target_across_all_budgets": bool(
            assessment.get("distinct_control_matches_target_across_all_budgets", False)
        ),
        "neutral_context_replication_matched": bool(
            assessment.get("neutral_context_replication_matched", False)
        ),
        "candidate_assessment_status": str(assessment.get("candidate_status", "")),
        "ready_for_post_execution_consolidation": bool(
            assessment.get("ready_for_post_execution_consolidation", False)
        ),
        "ready_for_A32_review": bool(assessment.get("ready_for_A32_review", False)),
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "outcome_status": outcome,
        "support": 0,
        "truth_status": SAGE6E_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def validate_sage6e_source(source: Mapping[str, Any]) -> None:
    """Validate that SAGE.6d is immutable, complete, and non-verdict."""
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6D_SCHEMA_VERSION:
        raise ValueError("SAGE.6d schema version is not supported by SAGE.6e")
    if str(source.get("outcome_status", "")) != SAGE6D_HANDOFF_COMPILED:
        raise ValueError("SAGE.6e requires a completed SAGE.6d handoff")
    if str(source.get("truth_status", "")) != SAGE6D_TRUTH_STATUS:
        raise ValueError("SAGE.6d truth must remain unevaluated")
    if (
        str(source.get("status", "")) != "UNRESOLVED"
        or str(source.get("revision_status", "")) != "CANDIDATE_ONLY"
        or int(source.get("support", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6d must remain unresolved candidate-only with support 0")
    if any(
        bool(source.get(flag, False))
        for flag in (
            "execution_performed",
            "revision_performed",
            "confirmation_performed",
            "refutation_performed",
            "a32_intake_requested",
            "a32_write_performed",
            "a33_write_performed",
            "scope_generalization_performed",
        )
    ):
        raise ValueError(
            "SAGE.6d cannot contain execution, verdict, or registry writes"
        )
    if (
        not source.get("gate")
        or not all(bool(value) for value in source.get("gate", {}).values())
        or not bool(summary.get("gate_passed", False))
    ):
        raise ValueError("every SAGE.6d source gate must pass")

    handoffs = [
        row for row in source.get("handoff_items", []) or [] if isinstance(row, Mapping)
    ]
    protocols = [
        row
        for row in source.get("pre_registered_followup_protocols", []) or []
        if isinstance(row, Mapping)
    ]
    if len(handoffs) != 1:
        raise ValueError("SAGE.6e requires exactly one SAGE.6d handoff")
    handoff = handoffs[0]
    if (
        not bool(handoff.get("ready_for_followup_execution", False))
        or bool(handoff.get("ready_for_A32_review", True))
        or bool(handoff.get("a32_intake_requested", True))
    ):
        raise ValueError("SAGE.6d handoff must be ready only for followup execution")
    if len(protocols) != int(summary.get("pre_registered_followup_protocols", 0) or 0):
        raise ValueError("SAGE.6d protocol count must match its summary")
    protocol_ids = [str(row.get("protocol_id", "")) for row in protocols]
    if (
        "" in protocol_ids
        or len(protocol_ids) != len(set(protocol_ids))
        or protocol_ids
        != [
            str(value)
            for value in handoff.get("pre_registered_followup_protocol_ids", []) or []
        ]
    ):
        raise ValueError("SAGE.6d protocol identities and order must remain exact")

    game_id = str(summary.get("game_id", ""))
    budgets = {int(value) for value in summary.get("budgets", []) or []}
    diversity = [
        row
        for row in protocols
        if str(row.get("request_type", "")) == CONTROL_DIVERSITY_REQUEST
    ]
    neutral = [
        row
        for row in protocols
        if str(row.get("request_type", "")) == NEUTRAL_REPLICATION_REQUEST
    ]
    if (
        len(diversity) != len(budgets)
        or {int(row.get("budget", 0) or 0) for row in diversity} != budgets
        or {str(row.get("control_action", "")) for row in diversity} != {"ACTION3"}
        or any(
            float(row.get("expected_prior_effect_size", 0.0) or 0.0) <= 0.0
            for row in diversity
        )
    ):
        raise ValueError("SAGE.6d control-diversity protocols must remain exact")
    if (
        len(neutral) != 1
        or str(neutral[0].get("control_action", "")) != "ACTION1"
        or float(neutral[0].get("expected_prior_effect_size", 1.0) or 0.0) != 0.0
    ):
        raise ValueError("SAGE.6d neutral replication protocol must remain exact")

    context_hashes = [str(row.get("context_snapshot_hash", "")) for row in protocols]
    if "" in context_hashes or len(context_hashes) != len(set(context_hashes)):
        raise ValueError("SAGE.6d protocol contexts must remain unique")
    for protocol in protocols:
        if (
            str(protocol.get("game_id", "")) != game_id
            or str(protocol.get("target_action", "")) != "ACTION2"
            or str(protocol.get("execution_status", ""))
            != "PRE_REGISTERED_NOT_EXECUTED"
            or bool(protocol.get("execution_performed", True))
            or not bool(protocol.get("exact_replay_required", False))
            or not bool(protocol.get("context_hash_verification_required", False))
            or not bool(
                protocol.get("target_and_control_must_start_from_same_context", False)
            )
            or bool(protocol.get("cross_context_substitution_allowed", True))
            or bool(protocol.get("cross_context_merge_allowed", True))
            or len(protocol.get("context_replay", []) or [])
            != len(protocol.get("context_replay_args", []) or [])
            or int(protocol.get("support", 0) or 0) != 0
            or str(protocol.get("truth_status", "")) != SAGE6D_TRUTH_STATUS
        ):
            raise ValueError(
                "SAGE.6d protocol replay and candidate state must remain exact"
            )

    manifest = list(handoff.get("context_cluster_manifest", []) or [])
    manifest_by_id = {
        str(row.get("context_cluster_id", "")): row
        for row in manifest
        if isinstance(row, Mapping)
    }
    for protocol in protocols:
        cluster = manifest_by_id.get(str(protocol.get("source_context_cluster_id", "")))
        if (
            cluster is None
            or str(cluster.get("context_snapshot_hash", ""))
            != str(protocol.get("context_snapshot_hash", ""))
            or str(cluster.get("selected_followup_protocol_id", ""))
            != str(protocol.get("protocol_id", ""))
        ):
            raise ValueError("SAGE.6d protocol and context manifest must align exactly")


def build_source_sage6d_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": str(source.get("config", {}).get("schema_version", "")),
        "game_id": str(source.get("summary", {}).get("game_id", "")),
        "budgets": list(source.get("summary", {}).get("budgets", []) or []),
        "outcome_status": str(source.get("outcome_status", "")),
        "protocols_pre_registered": int(
            source.get("summary", {}).get("pre_registered_followup_protocols", 0) or 0
        ),
        "gate_passed": bool(source.get("summary", {}).get("gate_passed", False)),
        "support": int(source.get("support", 0) or 0),
        "truth_status": str(source.get("truth_status", "")),
        "revision_status": str(source.get("revision_status", "")),
        "source_counted_as_scientific_support": False,
    }


def write_sage6e_second_unknown_game_followup_execution(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    counts = Counter(str(value) for value in values if str(value))
    return {key: int(counts[key]) for key in sorted(counts)}


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute every SAGE.6d followup exactly as pre-registered."
    )
    parser.add_argument("--source-sage6d", default=str(DEFAULT_SAGE6D_HANDOFF_PATH))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH))
    args = parser.parse_args(argv)
    run_sage6e_second_unknown_game_followup_execution(
        source_sage6d_path=args.source_sage6d,
        environments_dir=args.environments_dir,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
