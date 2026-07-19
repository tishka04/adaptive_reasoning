"""SAGE.7c context-aware consolidation of parameterized ACTION6 events."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .parameterized_control_acquisition import (
    CONTROL_EXCEEDS_TARGET_EFFECT,
    DISCRIMINATING_TARGET_EFFECT,
    NON_DISCRIMINATING_EQUAL_EFFECT,
)
from .third_unknown_game_parameterized_execution import (
    DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH,
    SAGE7B_CONSOLIDATION_REQUIRED,
    SAGE7B_EXECUTION_COMPLETED,
    SAGE7B_SCHEMA_VERSION,
    SAGE7B_TRUTH_STATUS,
)


DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage7c_third_unknown_game_parameterized_consolidation.json"
)

SAGE7C_SCHEMA_VERSION = "sage.third_unknown_game_parameterized_consolidation.v1"
SAGE7C_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_7C"
SAGE7C_CONTROL_DEPENDENT_DOSSIER = (
    "SAGE_THIRD_UNKNOWN_GAME_CONTROL_DEPENDENT_PARAMETERIZED_DOSSIER_CANDIDATE_ONLY"
)
SAGE7C_NO_ELIGIBLE_DOSSIER = (
    "SAGE_THIRD_UNKNOWN_GAME_NO_ELIGIBLE_PARAMETERIZED_DOSSIER_CANDIDATE_ONLY"
)
SAGE7C_HANDOFF_REQUIRED = "SAGE7D_A32_CONTROL_DEPENDENT_HANDOFF_REQUIRED_CANDIDATE_ONLY"
SAGE7C_NO_HANDOFF_REQUESTED = "NO_A32_HANDOFF_REQUESTED_CANDIDATE_ONLY"

CONTROL_DEPENDENT_CONTEXT = "CONTROL_DEPENDENT_PARAMETERIZED_CONTEXT_CANDIDATE_ONLY"
NON_DISCRIMINATING_CONTEXT = (
    "ALL_PARAMETERIZED_CONTROLS_NON_DISCRIMINATING_CONTEXT_CANDIDATE_ONLY"
)
CONTRADICTORY_CONTEXT = "CONTRADICTORY_PARAMETERIZED_CONTEXT_CANDIDATE_ONLY"
INCOMPLETE_CONTEXT = "INCOMPLETE_PARAMETERIZED_CONTEXT_CANDIDATE_ONLY"

A32_ELIGIBLE_CONTROL_DEPENDENT = (
    "A32_REVIEW_ELIGIBLE_CONTROL_DEPENDENT_PARAMETERIZED_CONTRAST_CANDIDATE_ONLY"
)
A32_INELIGIBLE_NON_DISCRIMINATING = (
    "A32_REVIEW_INELIGIBLE_NON_DISCRIMINATING_PARAMETERIZED_CANDIDATE_ONLY"
)
A32_INELIGIBLE_INCOMPLETE = (
    "A32_REVIEW_INELIGIBLE_INCOMPLETE_PARAMETERIZED_CANDIDATE_ONLY"
)
A32_INELIGIBLE_CONTRADICTORY = (
    "A32_REVIEW_INELIGIBLE_CONTRADICTORY_PARAMETERIZED_CANDIDATE_ONLY"
)

AUTONOMOUS_EFFECT_UNRESOLVED = "UNRESOLVED_CONTROL_DEPENDENT_TARGET_EFFECT"
MIN_INDEPENDENT_CONTEXTS = 3
MIN_CONTROLS_PER_CONTEXT = 2
MIN_CROSS_BUDGET_REPLICATED_CONTEXTS = 1


def run_sage7c_parameterized_consolidation(
    *,
    source_sage7b_path: str | Path = DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Consolidate exact comparisons without treating budget repeats as contexts."""
    source = _load_json(source_sage7b_path)
    validate_sage7c_source(source)
    comparison_groups = consolidate_exact_comparisons(source)
    context_assessments = build_parameterized_context_assessments(comparison_groups)
    candidates = build_parameterized_candidate_dossiers(context_assessments)
    gate = build_sage7c_gate(
        source=source,
        comparison_groups=comparison_groups,
        context_assessments=context_assessments,
        candidates=candidates,
    )
    eligible = [
        row for row in candidates if bool(row.get("a32_handoff_eligible", False))
    ]
    outcome = (
        SAGE7C_CONTROL_DEPENDENT_DOSSIER
        if all(gate.values()) and eligible
        else SAGE7C_NO_ELIGIBLE_DOSSIER
    )
    summary = summarize_sage7c(
        source=source,
        comparison_groups=comparison_groups,
        context_assessments=context_assessments,
        candidates=candidates,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE7C_SCHEMA_VERSION,
            "source_sage7b_path": str(source_sage7b_path),
            "comparison_grouping_key": [
                "game_id",
                "context_snapshot_hash",
                "target_action",
                "target_action_args",
                "control_action",
                "control_action_args",
                "metric",
            ],
            "context_grouping_key": [
                "game_id",
                "context_snapshot_hash",
                "target_action",
                "target_action_args",
                "metric",
            ],
            "candidate_grouping_key": [
                "game_id",
                "target_action",
                "target_action_args",
                "metric",
            ],
            "cross_budget_replays_counted_as_independent_contexts": False,
            "minimum_independent_contexts": MIN_INDEPENDENT_CONTEXTS,
            "minimum_controls_per_context": MIN_CONTROLS_PER_CONTEXT,
            "minimum_cross_budget_replicated_contexts": (
                MIN_CROSS_BUDGET_REPLICATED_CONTEXTS
            ),
            "eligibility_criteria_fixed_before_a32_verdict": True,
            "autonomous_target_effect_requires_discrimination_from_all_controls": True,
            "relational_control_dependence_is_a_separate_candidate_type": True,
            "parameterized_variants_are_distinct_action_families": False,
            "inputs_read": ["SAGE.7b"],
            "artifacts_not_modified": [
                "SAGE.7b",
                "SAGE.7a",
                "SAGE.7",
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
                "raw_events_are_not_scientific_support": True,
                "technical_repetitions_are_not_independent_contexts": True,
                "candidate_eligibility_is_not_an_a32_decision": True,
                "control_dependence_does_not_confirm_an_autonomous_effect": True,
                "a32_a33_write_performed": False,
            },
        },
        "source_sage7b_context": build_source_sage7b_context(source),
        "consolidated_comparison_groups": comparison_groups,
        "parameterized_context_assessments": context_assessments,
        "parameterized_candidate_dossiers": candidates,
        "a32_handoff_candidates": eligible,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE7C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "raw_events_counted_as_scientific_support": False,
        "technical_repetitions_counted_as_independent_contexts": False,
        "candidate_eligibility_counted_as_a32_decision": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage7c_parameterized_consolidation(payload, output_path)
    return payload


def consolidate_exact_comparisons(source: Mapping[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for experiment in source.get("executed_parameterized_experiments", []) or []:
        if not isinstance(experiment, Mapping):
            continue
        for comparison in experiment.get("parameterized_comparisons", []) or []:
            if not isinstance(comparison, Mapping):
                continue
            row = {
                **dict(comparison),
                "request_id": str(experiment.get("request_id", "")),
                "budget": int(experiment.get("budget", 0) or 0),
                "source_step": int(experiment.get("source_step", 0) or 0),
                "game_id": str(experiment.get("game_id", "")),
                "context_snapshot_hash": str(
                    experiment.get("context_snapshot_hash", "")
                ),
            }
            grouped[_comparison_key(row)].append(row)

    consolidated: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        first = rows[0]
        effects = [
            float(row.get("controlled_delta", {}).get("effect_size", 0.0) or 0.0)
            for row in rows
        ]
        target_signals = [float(row.get("target_signal", 0.0) or 0.0) for row in rows]
        control_signals = [float(row.get("control_signal", 0.0) or 0.0) for row in rows]
        statuses = {str(row.get("discrimination_status", "")) for row in rows}
        consistent = (
            len(set(effects)) == 1
            and len(set(target_signals)) == 1
            and len(set(control_signals)) == 1
            and len(statuses) == 1
        )
        consolidated.append(
            {
                "comparison_group_id": f"sage7c::comparison::{_short_digest(key)}",
                "comparison_signature": json.loads(key),
                "game_id": str(first.get("game_id", "")),
                "context_snapshot_hash": str(first.get("context_snapshot_hash", "")),
                "target_action": str(first.get("target_action", "")),
                "target_action_args": dict(first.get("target_action_args", {}) or {}),
                "control_action": str(first.get("control_action", "")),
                "control_action_args": dict(first.get("control_action_args", {}) or {}),
                "metric": str(first.get("metric", "")),
                "target_signal": target_signals[0],
                "control_signal": control_signals[0],
                "effect_size": effects[0],
                "discrimination_status": str(first.get("discrimination_status", "")),
                "raw_event_count": len(rows),
                "technical_replication_events": max(0, len(rows) - 1),
                "budgets_observed": sorted(
                    {int(row.get("budget", 0) or 0) for row in rows}
                ),
                "source_steps": sorted(
                    {int(row.get("source_step", 0) or 0) for row in rows}
                ),
                "source_request_ids": sorted(
                    {str(row.get("request_id", "")) for row in rows}
                ),
                "source_comparison_ids": sorted(
                    {str(row.get("comparison_id", "")) for row in rows}
                ),
                "cross_budget_replicated": len(
                    {int(row.get("budget", 0) or 0) for row in rows}
                )
                > 1,
                "replication_consistent": consistent,
                "independent_context_count": 1,
                "support": 0,
                "truth_status": SAGE7C_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "raw_events_counted_as_scientific_support": False,
            }
        )
    return consolidated


def build_parameterized_context_assessments(
    comparison_groups: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in comparison_groups:
        grouped[_context_key(row)].append(dict(row))

    assessments: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda row: _canonical_json(row.get("control_action_args", {})),
        )
        first = rows[0]
        statuses = [str(row.get("discrimination_status", "")) for row in rows]
        effects = [float(row.get("effect_size", 0.0) or 0.0) for row in rows]
        if any(status == CONTROL_EXCEEDS_TARGET_EFFECT for status in statuses):
            context_status = CONTRADICTORY_CONTEXT
        elif (
            DISCRIMINATING_TARGET_EFFECT in statuses
            and NON_DISCRIMINATING_EQUAL_EFFECT in statuses
            and len(rows) >= MIN_CONTROLS_PER_CONTEXT
        ):
            context_status = CONTROL_DEPENDENT_CONTEXT
        elif statuses and all(
            status == NON_DISCRIMINATING_EQUAL_EFFECT for status in statuses
        ):
            context_status = NON_DISCRIMINATING_CONTEXT
        else:
            context_status = INCOMPLETE_CONTEXT
        assessments.append(
            {
                "context_assessment_id": f"sage7c::context::{_short_digest(key)}",
                "context_signature": json.loads(key),
                "game_id": str(first.get("game_id", "")),
                "context_snapshot_hash": str(first.get("context_snapshot_hash", "")),
                "target_action": str(first.get("target_action", "")),
                "target_action_args": dict(first.get("target_action_args", {}) or {}),
                "metric": str(first.get("metric", "")),
                "parameterized_controls": [
                    {
                        "action": str(row.get("control_action", "")),
                        "action_args": dict(row.get("control_action_args", {}) or {}),
                        "target_signal": float(row.get("target_signal", 0.0) or 0.0),
                        "control_signal": float(row.get("control_signal", 0.0) or 0.0),
                        "effect_size": float(row.get("effect_size", 0.0) or 0.0),
                        "discrimination_status": str(
                            row.get("discrimination_status", "")
                        ),
                        "comparison_group_id": str(row.get("comparison_group_id", "")),
                    }
                    for row in rows
                ],
                "parameterized_controls_count": len(rows),
                "positive_control_contrasts": sum(effect > 0 for effect in effects),
                "negative_control_contrasts": sum(effect < 0 for effect in effects),
                "neutral_control_contrasts": sum(effect == 0 for effect in effects),
                "context_status": context_status,
                "raw_event_count": sum(
                    int(row.get("raw_event_count", 0) or 0) for row in rows
                ),
                "technical_replication_events": sum(
                    int(row.get("technical_replication_events", 0) or 0) for row in rows
                ),
                "budgets_observed": sorted(
                    {
                        int(budget)
                        for row in rows
                        for budget in row.get("budgets_observed", []) or []
                    }
                ),
                "cross_budget_replicated": any(
                    bool(row.get("cross_budget_replicated", False)) for row in rows
                ),
                "all_repetitions_consistent": all(
                    bool(row.get("replication_consistent", False)) for row in rows
                ),
                "independent_context_count": 1,
                "support": 0,
                "truth_status": SAGE7C_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "context_assessment_counted_as_scientific_support": False,
            }
        )
    return assessments


def build_parameterized_candidate_dossiers(
    assessments: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in assessments:
        grouped[_candidate_key(row)].append(dict(row))

    dossiers: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key], key=lambda row: str(row.get("context_snapshot_hash", ""))
        )
        first = rows[0]
        control_statuses: Dict[str, set[str]] = defaultdict(set)
        control_effects: Dict[str, set[float]] = defaultdict(set)
        control_contexts: Dict[str, set[str]] = defaultdict(set)
        control_payloads: Dict[str, Dict[str, Any]] = {}
        for context in rows:
            for control in context.get("parameterized_controls", []) or []:
                control_key = _canonical_json(
                    {
                        "action": control.get("action"),
                        "action_args": control.get("action_args", {}),
                    }
                )
                control_payloads[control_key] = {
                    "action": str(control.get("action", "")),
                    "action_args": dict(control.get("action_args", {}) or {}),
                }
                control_statuses[control_key].add(
                    str(control.get("discrimination_status", ""))
                )
                control_effects[control_key].add(
                    float(control.get("effect_size", 0.0) or 0.0)
                )
                control_contexts[control_key].add(
                    str(context.get("context_snapshot_hash", ""))
                )
        positive_controls = [
            control_payloads[control_key]
            for control_key in sorted(control_payloads)
            if control_statuses[control_key] == {DISCRIMINATING_TARGET_EFFECT}
            and all(effect > 0 for effect in control_effects[control_key])
            and len(control_statuses[control_key]) == 1
            and len(control_contexts[control_key]) == len(rows)
        ]
        neutral_controls = [
            control_payloads[control_key]
            for control_key in sorted(control_payloads)
            if control_statuses[control_key] == {NON_DISCRIMINATING_EQUAL_EFFECT}
            and control_effects[control_key] == {0.0}
            and len(control_contexts[control_key]) == len(rows)
        ]
        negative_events = sum(
            int(row.get("negative_control_contrasts", 0) or 0) for row in rows
        )
        control_dependent_contexts = sum(
            str(row.get("context_status", "")) == CONTROL_DEPENDENT_CONTEXT
            for row in rows
        )
        neutral_contexts = sum(
            str(row.get("context_status", "")) == NON_DISCRIMINATING_CONTEXT
            for row in rows
        )
        contradictory_contexts = sum(
            str(row.get("context_status", "")) == CONTRADICTORY_CONTEXT for row in rows
        )
        cross_budget_contexts = sum(
            bool(row.get("cross_budget_replicated", False)) for row in rows
        )
        minimum_controls_met = all(
            int(row.get("parameterized_controls_count", 0) or 0)
            >= MIN_CONTROLS_PER_CONTEXT
            for row in rows
        )
        eligible = (
            len(rows) >= MIN_INDEPENDENT_CONTEXTS
            and control_dependent_contexts == len(rows)
            and cross_budget_contexts >= MIN_CROSS_BUDGET_REPLICATED_CONTEXTS
            and minimum_controls_met
            and bool(positive_controls)
            and bool(neutral_controls)
            and negative_events == 0
            and all(bool(row.get("all_repetitions_consistent", False)) for row in rows)
        )
        if eligible:
            eligibility = A32_ELIGIBLE_CONTROL_DEPENDENT
        elif contradictory_contexts or negative_events:
            eligibility = A32_INELIGIBLE_CONTRADICTORY
        elif neutral_contexts == len(rows):
            eligibility = A32_INELIGIBLE_NON_DISCRIMINATING
        else:
            eligibility = A32_INELIGIBLE_INCOMPLETE
        dossiers.append(
            {
                "candidate_id": f"sage7c::candidate::{_short_digest(key)}",
                "candidate_signature": json.loads(key),
                "game_id": str(first.get("game_id", "")),
                "candidate_type": (
                    "CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST"
                    if eligible
                    else "PARAMETERIZED_ACTION6_TARGET_CANDIDATE"
                ),
                "target_action": str(first.get("target_action", "")),
                "target_action_args": dict(first.get("target_action_args", {}) or {}),
                "metric": str(first.get("metric", "")),
                "independent_contexts": len(rows),
                "context_assessment_ids": [
                    str(row.get("context_assessment_id", "")) for row in rows
                ],
                "context_snapshot_hashes": [
                    str(row.get("context_snapshot_hash", "")) for row in rows
                ],
                "control_dependent_contexts": control_dependent_contexts,
                "non_discriminating_contexts": neutral_contexts,
                "contradictory_contexts": contradictory_contexts,
                "cross_budget_replicated_contexts": cross_budget_contexts,
                "raw_comparison_events": sum(
                    int(row.get("raw_event_count", 0) or 0) for row in rows
                ),
                "technical_replication_events": sum(
                    int(row.get("technical_replication_events", 0) or 0) for row in rows
                ),
                "differentiating_control_variants": positive_controls,
                "equivalent_control_variants": neutral_controls,
                "negative_control_events": negative_events,
                "minimum_independent_contexts_required": MIN_INDEPENDENT_CONTEXTS,
                "minimum_controls_per_context_required": MIN_CONTROLS_PER_CONTEXT,
                "minimum_cross_budget_replicated_contexts_required": (
                    MIN_CROSS_BUDGET_REPLICATED_CONTEXTS
                ),
                "minimum_controls_met": minimum_controls_met,
                "all_repetitions_consistent": all(
                    bool(row.get("all_repetitions_consistent", False)) for row in rows
                ),
                "autonomous_target_effect_status": AUTONOMOUS_EFFECT_UNRESOLVED,
                "a32_eligibility_status": eligibility,
                "a32_handoff_eligible": eligible,
                "candidate_support_events": sum(
                    int(row.get("positive_control_contrasts", 0) or 0) for row in rows
                ),
                "candidate_support_events_counted_as_scientific_support": False,
                "technical_repetitions_counted_as_independent_contexts": False,
                "parameterized_controls_counted_as_distinct_actions": False,
                "support": 0,
                "truth_status": SAGE7C_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "a32_decision_performed": False,
            }
        )
    return dossiers


def build_sage7c_gate(
    *,
    source: Mapping[str, Any],
    comparison_groups: Sequence[Mapping[str, Any]],
    context_assessments: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    raw_events = int(source.get("summary", {}).get("comparison_events", 0) or 0)
    group_ids = [str(row.get("comparison_group_id", "")) for row in comparison_groups]
    context_ids = [
        str(row.get("context_assessment_id", "")) for row in context_assessments
    ]
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    return {
        "source_sage7b_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        ),
        "all_raw_events_consolidated": sum(
            int(row.get("raw_event_count", 0) or 0) for row in comparison_groups
        )
        == raw_events,
        "comparison_groups_unique": "" not in group_ids
        and len(group_ids) == len(set(group_ids)),
        "context_assessments_unique": "" not in context_ids
        and len(context_ids) == len(set(context_ids)),
        "candidate_dossiers_unique": "" not in candidate_ids
        and len(candidate_ids) == len(set(candidate_ids)),
        "all_exact_repetitions_consistent": bool(comparison_groups)
        and all(
            bool(row.get("replication_consistent", False)) for row in comparison_groups
        ),
        "every_context_has_two_parameterized_controls": bool(context_assessments)
        and all(
            int(row.get("parameterized_controls_count", 0) or 0)
            == MIN_CONTROLS_PER_CONTEXT
            for row in context_assessments
        ),
        "raw_events_not_relabelled_as_independent_contexts": sum(
            int(row.get("independent_context_count", 0) or 0)
            for row in comparison_groups
        )
        == len(comparison_groups)
        and sum(
            int(row.get("technical_replication_events", 0) or 0)
            for row in comparison_groups
        )
        == raw_events - len(comparison_groups),
        "control_dependent_candidate_eligible": any(
            bool(row.get("a32_handoff_eligible", False)) for row in candidates
        ),
        "autonomous_effect_kept_unresolved": all(
            str(row.get("autonomous_target_effect_status", ""))
            == AUTONOMOUS_EFFECT_UNRESOLVED
            for row in candidates
        ),
        "parameterized_variants_not_relabelled_as_actions": all(
            not bool(
                row.get("parameterized_controls_counted_as_distinct_actions", True)
            )
            for row in candidates
        ),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0 for row in comparison_groups
        )
        and all(int(row.get("support", 0) or 0) == 0 for row in context_assessments)
        and all(int(row.get("support", 0) or 0) == 0 for row in candidates),
        "source_registry_quarantine_preserved": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        )
        == 0
        and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
        and not bool(source.get("scope_generalization_performed", False)),
    }


def summarize_sage7c(
    *,
    source: Mapping[str, Any],
    comparison_groups: Sequence[Mapping[str, Any]],
    context_assessments: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    outcome: str,
) -> Dict[str, Any]:
    eligible = [
        row for row in candidates if bool(row.get("a32_handoff_eligible", False))
    ]
    raw_events = int(source.get("summary", {}).get("comparison_events", 0) or 0)
    ready = outcome == SAGE7C_CONTROL_DEPENDENT_DOSSIER and bool(eligible)
    return {
        "game_id": str(source.get("summary", {}).get("game_id", "")),
        "source_sage7b_outcome_status": str(source.get("outcome_status", "")),
        "raw_comparison_events": raw_events,
        "consolidated_comparison_groups": len(comparison_groups),
        "technical_replication_events": sum(
            int(row.get("technical_replication_events", 0) or 0)
            for row in comparison_groups
        ),
        "independent_parameterized_contexts": len(context_assessments),
        "cross_budget_replicated_contexts": sum(
            bool(row.get("cross_budget_replicated", False))
            for row in context_assessments
        ),
        "control_dependent_contexts": sum(
            str(row.get("context_status", "")) == CONTROL_DEPENDENT_CONTEXT
            for row in context_assessments
        ),
        "non_discriminating_contexts": sum(
            str(row.get("context_status", "")) == NON_DISCRIMINATING_CONTEXT
            for row in context_assessments
        ),
        "contradictory_contexts": sum(
            str(row.get("context_status", "")) == CONTRADICTORY_CONTEXT
            for row in context_assessments
        ),
        "parameterized_candidate_dossiers": len(candidates),
        "a32_handoff_eligible_candidates": len(eligible),
        "eligible_candidate_ids": [
            str(row.get("candidate_id", "")) for row in eligible
        ],
        "eligible_candidate_target_args": [
            dict(row.get("target_action_args", {}) or {}) for row in eligible
        ],
        "eligible_candidate_independent_contexts": sum(
            int(row.get("independent_contexts", 0) or 0) for row in eligible
        ),
        "eligible_candidate_cross_budget_replicated_contexts": sum(
            int(row.get("cross_budget_replicated_contexts", 0) or 0) for row in eligible
        ),
        "eligible_candidate_raw_comparison_events": sum(
            int(row.get("raw_comparison_events", 0) or 0) for row in eligible
        ),
        "eligible_candidate_technical_replication_events": sum(
            int(row.get("technical_replication_events", 0) or 0) for row in eligible
        ),
        "autonomous_target_effects_confirmed": 0,
        "autonomous_target_effects_unresolved": len(candidates),
        "parameterized_variants_counted_as_distinct_actions": False,
        "ready_for_a32_handoff_compilation": ready,
        "required_next_step": (
            SAGE7C_HANDOFF_REQUIRED if ready else SAGE7C_NO_HANDOFF_REQUESTED
        ),
        "gate_passed": outcome == SAGE7C_CONTROL_DEPENDENT_DOSSIER,
        "outcome_status": outcome,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "support": 0,
        "truth_status": SAGE7C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def build_source_sage7b_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(source.get("summary", {}) or {})
    return {
        "source_outcome_status": str(source.get("outcome_status", "")),
        "game_id": str(summary.get("game_id", "")),
        "budgets": list(summary.get("budgets_evaluated", []) or []),
        "requests_executed": int(summary.get("requests_executed", 0) or 0),
        "comparison_events": int(summary.get("comparison_events", 0) or 0),
        "positive_delta_events": int(summary.get("positive_delta_events", 0) or 0),
        "negative_delta_events": int(summary.get("negative_delta_events", 0) or 0),
        "zero_delta_events": int(summary.get("zero_delta_events", 0) or 0),
        "ready_for_event_consolidation": bool(
            summary.get("ready_for_event_consolidation", False)
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
        "truth_status": SAGE7C_TRUTH_STATUS,
    }


def validate_sage7c_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE7B_SCHEMA_VERSION:
        raise ValueError("SAGE.7b schema version is not supported by SAGE.7c")
    if str(source.get("outcome_status", "")) != SAGE7B_EXECUTION_COMPLETED:
        raise ValueError("SAGE.7c requires the completed SAGE.7b execution")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.7b source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE7B_TRUTH_STATUS:
        raise ValueError("SAGE.7b truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.7b source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.7b support must remain 0")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.7b cannot perform a scientific verdict")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.7b cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.7b wrong_confirmations must remain 0")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
        or bool(source.get("scope_generalization_performed", False))
    ):
        raise ValueError("SAGE.7b cannot import or generalize quarantined mechanics")
    if bool(source.get("parameterized_controls_counted_as_distinct_actions", True)):
        raise ValueError("SAGE.7b controls cannot be relabelled as distinct actions")
    if (
        bool(source.get("raw_events_counted_as_scientific_support", True))
        or bool(source.get("positive_deltas_counted_as_support", True))
        or bool(source.get("negative_deltas_counted_as_refutation", True))
        or bool(source.get("zero_deltas_counted_as_non_identifiability", True))
    ):
        raise ValueError("SAGE.7b raw events cannot carry a scientific verdict")
    if not bool(summary.get("gate_passed", False)) or not bool(
        summary.get("ready_for_event_consolidation", False)
    ):
        raise ValueError("SAGE.7b consolidation gate must pass before SAGE.7c")
    if str(summary.get("required_next_step", "")) != SAGE7B_CONSOLIDATION_REQUIRED:
        raise ValueError("SAGE.7b must explicitly request event consolidation")
    experiments = [
        row
        for row in source.get("executed_parameterized_experiments", []) or []
        if isinstance(row, Mapping)
    ]
    blocked = source.get("blocked_parameterized_experiments", []) or []
    if blocked or len(experiments) != int(summary.get("requests_executed", 0) or 0):
        raise ValueError("SAGE.7b execution must be complete and unblocked")
    comparison_ids: List[str] = []
    for experiment in experiments:
        comparisons = [
            row
            for row in experiment.get("parameterized_comparisons", []) or []
            if isinstance(row, Mapping)
        ]
        if (
            not bool(experiment.get("live_prefix_replay_exact", False))
            or not bool(experiment.get("protocol_exact_match", False))
            or bool(experiment.get("protocol_substitution_detected", True))
            or len(comparisons) != 2
            or int(experiment.get("support", 0) or 0) != 0
        ):
            raise ValueError(
                "every SAGE.7b experiment must be exact and candidate-only"
            )
        for comparison in comparisons:
            comparison_ids.append(str(comparison.get("comparison_id", "")))
            if (
                not bool(comparison.get("context_snapshot_hash_verified", False))
                or not bool(comparison.get("protocol_exact_match", False))
                or int(comparison.get("support", 0) or 0) != 0
                or bool(
                    comparison.get("comparison_counted_as_scientific_support", True)
                )
            ):
                raise ValueError("every SAGE.7b comparison must remain raw and exact")
    if len(comparison_ids) != int(summary.get("comparison_events", 0) or 0):
        raise ValueError("SAGE.7b comparison count must be exact")
    if "" in comparison_ids or len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("SAGE.7b comparison ids must be unique")
    gate = dict(source.get("gate", {}) or {})
    if not gate or not all(bool(value) for value in gate.values()):
        raise ValueError("every SAGE.7b source gate must pass")


def write_sage7c_parameterized_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _comparison_key(row: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "game_id": str(row.get("game_id", "")),
            "context_snapshot_hash": str(row.get("context_snapshot_hash", "")),
            "target_action": str(row.get("target_action", "")),
            "target_action_args": dict(row.get("target_action_args", {}) or {}),
            "control_action": str(row.get("control_action", "")),
            "control_action_args": dict(row.get("control_action_args", {}) or {}),
            "metric": str(row.get("metric", "")),
        }
    )


def _context_key(row: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "game_id": str(row.get("game_id", "")),
            "context_snapshot_hash": str(row.get("context_snapshot_hash", "")),
            "target_action": str(row.get("target_action", "")),
            "target_action_args": dict(row.get("target_action_args", {}) or {}),
            "metric": str(row.get("metric", "")),
        }
    )


def _candidate_key(row: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "game_id": str(row.get("game_id", "")),
            "target_action": str(row.get("target_action", "")),
            "target_action_args": dict(row.get("target_action_args", {}) or {}),
            "metric": str(row.get("metric", "")),
        }
    )


def _short_digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-sage7b", default=str(DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH)
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH)
    )
    args = parser.parse_args(argv)
    payload = run_sage7c_parameterized_consolidation(
        source_sage7b_path=args.source_sage7b,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
