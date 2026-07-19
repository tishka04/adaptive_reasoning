"""SAGE.7d candidate-only A32.7 handoff for the third unknown game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .parameterized_control_acquisition import (
    DISCRIMINATING_TARGET_EFFECT,
    NON_DISCRIMINATING_EQUAL_EFFECT,
)
from .third_unknown_game_parameterized_consolidation import (
    A32_ELIGIBLE_CONTROL_DEPENDENT,
    AUTONOMOUS_EFFECT_UNRESOLVED,
    CONTROL_DEPENDENT_CONTEXT,
    DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH,
    MIN_CONTROLS_PER_CONTEXT,
    MIN_CROSS_BUDGET_REPLICATED_CONTEXTS,
    MIN_INDEPENDENT_CONTEXTS,
    SAGE7C_CONTROL_DEPENDENT_DOSSIER,
    SAGE7C_HANDOFF_REQUIRED,
    SAGE7C_SCHEMA_VERSION,
    SAGE7C_TRUTH_STATUS,
)


DEFAULT_SAGE7D_A32_HANDOFF_PATH = (
    Path("diagnostics") / "sage" / "sage7d_third_unknown_game_a32_handoff.json"
)

SAGE7D_SCHEMA_VERSION = "sage.third_unknown_game_a32_handoff.v1"
SAGE7D_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_7D"
SAGE7D_HANDOFF_COMPILED = "SAGE_THIRD_UNKNOWN_GAME_A32_HANDOFF_COMPILED_CANDIDATE_ONLY"
SAGE7D_HANDOFF_INCOMPLETE = (
    "SAGE_THIRD_UNKNOWN_GAME_A32_HANDOFF_INCOMPLETE_CANDIDATE_ONLY"
)
SAGE7D_A32_REVIEW_REQUIRED = (
    "A32_7_CONTROL_DEPENDENT_PARAMETERIZED_SCIENTIFIC_REVIEW_REQUIRED"
)
SAGE7D_NO_A32_REVIEW_REQUESTED = "NO_A32_7_REVIEW_REQUESTED_CANDIDATE_ONLY"

HANDOFF_READY = "READY_FOR_A32_7_SCIENTIFIC_REVIEW_CANDIDATE_ONLY"
RELATIONAL_CANDIDATE_TYPE = "CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST"

ALLOWED_A32_7_DECISIONS = (
    "CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION",
    "KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT",
    "REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION",
    "REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION",
)


def run_sage7d_a32_handoff(
    *,
    source_sage7c_path: str | Path = DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Compile only the eligible SAGE.7c relational candidate for A32.7."""
    source = _load_json(source_sage7c_path)
    validate_sage7d_source(source)
    handoff_items, exclusions = compile_sage7d_handoff(source)
    gate = build_sage7d_gate(
        source=source,
        handoff_items=handoff_items,
        exclusions=exclusions,
    )
    outcome = (
        SAGE7D_HANDOFF_COMPILED
        if gate and all(gate.values())
        else SAGE7D_HANDOFF_INCOMPLETE
    )
    summary = summarize_sage7d(
        source=source,
        handoff_items=handoff_items,
        exclusions=exclusions,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE7D_SCHEMA_VERSION,
            "source_sage7c_path": str(source_sage7c_path),
            "inputs_read": ["SAGE.7c"],
            "compilation_policy": {
                "include_only_a32_handoff_eligible_candidates": True,
                "exclude_non_discriminating_candidates": True,
                "preserve_exact_context_hashes": True,
                "preserve_target_and_control_args": True,
                "preserve_metric_and_source_links": True,
                "technical_repetitions_are_not_independent_contexts": True,
                "autonomous_target_effect_remains_unresolved": True,
                "allowed_a32_decisions_are_not_preselected": True,
                "handoff_is_not_scientific_support": True,
                "a32_write_performed": False,
                "a33_write_performed": False,
            },
            "review_thresholds_preserved": {
                "minimum_independent_contexts": MIN_INDEPENDENT_CONTEXTS,
                "minimum_controls_per_context": MIN_CONTROLS_PER_CONTEXT,
                "minimum_cross_budget_replicated_contexts": (
                    MIN_CROSS_BUDGET_REPLICATED_CONTEXTS
                ),
            },
            "allowed_a32_7_decisions": list(ALLOWED_A32_7_DECISIONS),
            "artifacts_not_modified": [
                "SAGE.7c",
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
        },
        "source_sage7c_context": build_source_sage7c_context(source),
        "a32_review_handoff_items": handoff_items,
        "excluded_candidate_audit": exclusions,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE7D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "handoff_counted_as_scientific_support": False,
        "candidate_eligibility_counted_as_a32_decision": False,
        "a32_decision_preselected": False,
        "technical_repetitions_counted_as_independent_contexts": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage7d_a32_handoff(payload, output_path)
    return payload


def compile_sage7d_handoff(
    source: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = [
        dict(row)
        for row in source.get("parameterized_candidate_dossiers", []) or []
        if isinstance(row, Mapping)
    ]
    contexts_by_id = {
        str(row.get("context_assessment_id", "")): dict(row)
        for row in source.get("parameterized_context_assessments", []) or []
        if isinstance(row, Mapping) and str(row.get("context_assessment_id", ""))
    }
    handoff_items: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not bool(candidate.get("a32_handoff_eligible", False)):
            exclusions.append(
                {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "game_id": str(candidate.get("game_id", "")),
                    "target_action": str(candidate.get("target_action", "")),
                    "target_action_args": dict(
                        candidate.get("target_action_args", {}) or {}
                    ),
                    "metric": str(candidate.get("metric", "")),
                    "a32_eligibility_status": str(
                        candidate.get("a32_eligibility_status", "")
                    ),
                    "exclusion_reason": "SOURCE_CANDIDATE_NOT_A32_HANDOFF_ELIGIBLE",
                    "included_in_handoff": False,
                    "support": 0,
                    "truth_status": SAGE7D_TRUTH_STATUS,
                }
            )
            continue

        contexts = [
            contexts_by_id[str(context_id)]
            for context_id in candidate.get("context_assessment_ids", []) or []
        ]
        manifest = [build_context_manifest_record(context) for context in contexts]
        differentiating = [
            dict(row)
            for row in candidate.get("differentiating_control_variants", []) or []
        ]
        equivalent = [
            dict(row) for row in candidate.get("equivalent_control_variants", []) or []
        ]
        handoff_items.append(
            {
                "handoff_id": "sage7d::a32_7_handoff::001",
                "source_candidate_id": str(candidate.get("candidate_id", "")),
                "game_id": str(candidate.get("game_id", "")),
                "candidate_type": RELATIONAL_CANDIDATE_TYPE,
                "target_action": str(candidate.get("target_action", "")),
                "target_action_args": dict(
                    candidate.get("target_action_args", {}) or {}
                ),
                "metric": str(candidate.get("metric", "")),
                "relational_contrast": {
                    "target_variant": {
                        "action": str(candidate.get("target_action", "")),
                        "action_args": dict(
                            candidate.get("target_action_args", {}) or {}
                        ),
                    },
                    "differentiating_control_variants": differentiating,
                    "equivalent_control_variants": equivalent,
                    "relation_scope": (
                        "EXACT_TN36_LIVE_PREFIX_CONTEXTS_AND_PARAMETER_VARIANTS"
                    ),
                    "autonomous_target_effect_status": str(
                        candidate.get("autonomous_target_effect_status", "")
                    ),
                },
                "context_manifest": manifest,
                "independent_contexts": int(
                    candidate.get("independent_contexts", 0) or 0
                ),
                "cross_budget_replicated_contexts": int(
                    candidate.get("cross_budget_replicated_contexts", 0) or 0
                ),
                "raw_comparison_events": int(
                    candidate.get("raw_comparison_events", 0) or 0
                ),
                "technical_replication_events": int(
                    candidate.get("technical_replication_events", 0) or 0
                ),
                "candidate_support_events": int(
                    candidate.get("candidate_support_events", 0) or 0
                ),
                "minimum_independent_contexts_required": int(
                    candidate.get("minimum_independent_contexts_required", 0) or 0
                ),
                "minimum_controls_per_context_required": int(
                    candidate.get("minimum_controls_per_context_required", 0) or 0
                ),
                "minimum_cross_budget_replicated_contexts_required": int(
                    candidate.get(
                        "minimum_cross_budget_replicated_contexts_required", 0
                    )
                    or 0
                ),
                "source_a32_eligibility_status": str(
                    candidate.get("a32_eligibility_status", "")
                ),
                "scientific_questions_for_a32_7": [
                    (
                        "Does the exact target/differentiating/equivalent control "
                        "relation warrant a scope-limited confirmation?"
                    ),
                    (
                        "Does equivalence with one pre-registered control require "
                        "the autonomous target effect to remain unresolved?"
                    ),
                    (
                        "Are eight independent contexts and four cross-budget "
                        "replications sufficient without counting technical repeats?"
                    ),
                    (
                        "Should A32.7 confirm the relation, keep it unresolved, "
                        "request more tests, or refute it?"
                    ),
                ],
                "allowed_a32_7_decisions": list(ALLOWED_A32_7_DECISIONS),
                "handoff_status": HANDOFF_READY,
                "ready_for_A32_7_review": True,
                "a32_intake_requested": False,
                "a32_decision_preselected": False,
                "autonomous_target_effect_confirmed": False,
                "technical_repetitions_counted_as_independent_contexts": False,
                "parameterized_controls_counted_as_distinct_actions": False,
                "candidate_support_events_counted_as_scientific_support": False,
                "handoff_counted_as_scientific_support": False,
                "support": 0,
                "truth_status": SAGE7D_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return handoff_items, exclusions


def build_context_manifest_record(context: Mapping[str, Any]) -> Dict[str, Any]:
    controls = [
        dict(row)
        for row in context.get("parameterized_controls", []) or []
        if isinstance(row, Mapping)
    ]
    return {
        "context_assessment_id": str(context.get("context_assessment_id", "")),
        "context_snapshot_hash": str(context.get("context_snapshot_hash", "")),
        "game_id": str(context.get("game_id", "")),
        "target_action": str(context.get("target_action", "")),
        "target_action_args": dict(context.get("target_action_args", {}) or {}),
        "metric": str(context.get("metric", "")),
        "context_status": str(context.get("context_status", "")),
        "budgets_observed": list(context.get("budgets_observed", []) or []),
        "cross_budget_replicated": bool(context.get("cross_budget_replicated", False)),
        "raw_event_count": int(context.get("raw_event_count", 0) or 0),
        "technical_replication_events": int(
            context.get("technical_replication_events", 0) or 0
        ),
        "parameterized_controls": controls,
        "source_comparison_group_ids": [
            str(row.get("comparison_group_id", "")) for row in controls
        ],
        "all_repetitions_consistent": bool(
            context.get("all_repetitions_consistent", False)
        ),
        "independent_context_count": 1,
        "support": 0,
        "truth_status": SAGE7D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
    }


def build_sage7d_gate(
    *,
    source: Mapping[str, Any],
    handoff_items: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    handoff = dict(handoff_items[0]) if len(handoff_items) == 1 else {}
    manifest = list(handoff.get("context_manifest", []) or [])
    relation = dict(handoff.get("relational_contrast", {}) or {})
    source_candidates = list(source.get("parameterized_candidate_dossiers", []) or [])
    return {
        "source_sage7c_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        ),
        "one_eligible_candidate_produced_one_handoff": len(handoff_items) == 1
        and int(
            source.get("summary", {}).get("a32_handoff_eligible_candidates", 0) or 0
        )
        == 1,
        "all_noneligible_candidates_excluded": len(exclusions)
        == len(source_candidates) - 1
        and all(not bool(row.get("included_in_handoff", True)) for row in exclusions),
        "exact_context_manifest_preserved": len(manifest)
        == int(handoff.get("independent_contexts", 0) or 0)
        == 8
        and len({str(row.get("context_snapshot_hash", "")) for row in manifest}) == 8,
        "every_context_preserves_two_control_roles": all(
            int(len(row.get("parameterized_controls", []) or []))
            == MIN_CONTROLS_PER_CONTEXT
            and str(row.get("context_status", "")) == CONTROL_DEPENDENT_CONTEXT
            and _context_preserves_relational_controls(
                row,
                differentiating=relation.get("differentiating_control_variants", [])
                or [],
                equivalent=relation.get("equivalent_control_variants", []) or [],
            )
            for row in manifest
        ),
        "relational_target_and_controls_preserved": bool(relation.get("target_variant"))
        and len(relation.get("differentiating_control_variants", []) or []) == 1
        and len(relation.get("equivalent_control_variants", []) or []) == 1,
        "source_counts_preserved": int(handoff.get("raw_comparison_events", 0) or 0)
        == int(
            source.get("summary", {}).get("eligible_candidate_raw_comparison_events", 0)
            or 0
        )
        and int(handoff.get("technical_replication_events", 0) or 0)
        == int(
            source.get("summary", {}).get(
                "eligible_candidate_technical_replication_events", 0
            )
            or 0
        ),
        "technical_repetitions_not_recounted": all(
            int(row.get("independent_context_count", 0) or 0) == 1 for row in manifest
        )
        and not bool(
            handoff.get("technical_repetitions_counted_as_independent_contexts", True)
        ),
        "autonomous_target_effect_kept_unresolved": str(
            relation.get("autonomous_target_effect_status", "")
        )
        == AUTONOMOUS_EFFECT_UNRESOLVED
        and not bool(handoff.get("autonomous_target_effect_confirmed", True)),
        "handoff_ready_without_preselected_decision": str(
            handoff.get("handoff_status", "")
        )
        == HANDOFF_READY
        and bool(handoff.get("ready_for_A32_7_review", False))
        and not bool(handoff.get("a32_decision_preselected", True)),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0
            and not bool(row.get("revision_performed", False))
            for row in [*handoff_items, *manifest, *exclusions]
        ),
        "source_registry_quarantine_preserved": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        )
        == 0
        and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
        and not bool(source.get("scope_generalization_performed", False)),
    }


def summarize_sage7d(
    *,
    source: Mapping[str, Any],
    handoff_items: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    outcome: str,
) -> Dict[str, Any]:
    handoff = dict(handoff_items[0]) if len(handoff_items) == 1 else {}
    ready = outcome == SAGE7D_HANDOFF_COMPILED and bool(handoff_items)
    return {
        "game_id": str(source.get("summary", {}).get("game_id", "")),
        "source_sage7c_outcome_status": str(source.get("outcome_status", "")),
        "source_candidate_dossiers": int(
            source.get("summary", {}).get("parameterized_candidate_dossiers", 0) or 0
        ),
        "source_a32_handoff_eligible_candidates": int(
            source.get("summary", {}).get("a32_handoff_eligible_candidates", 0) or 0
        ),
        "handoff_items": len(handoff_items),
        "excluded_candidate_items": len(exclusions),
        "handoff_contexts": len(handoff.get("context_manifest", []) or []),
        "handoff_cross_budget_replicated_contexts": int(
            handoff.get("cross_budget_replicated_contexts", 0) or 0
        ),
        "handoff_raw_comparison_events": int(
            handoff.get("raw_comparison_events", 0) or 0
        ),
        "handoff_technical_replication_events": int(
            handoff.get("technical_replication_events", 0) or 0
        ),
        "handoff_candidate_support_events": int(
            handoff.get("candidate_support_events", 0) or 0
        ),
        "allowed_a32_7_decisions": list(ALLOWED_A32_7_DECISIONS),
        "a32_decision_preselected": False,
        "autonomous_target_effects_confirmed": 0,
        "autonomous_target_effects_unresolved": 1 if handoff else 0,
        "parameterized_variants_counted_as_distinct_actions": False,
        "ready_for_a32_7_scientific_review": ready,
        "required_next_step": (
            SAGE7D_A32_REVIEW_REQUIRED if ready else SAGE7D_NO_A32_REVIEW_REQUESTED
        ),
        "gate_passed": outcome == SAGE7D_HANDOFF_COMPILED,
        "outcome_status": outcome,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "support": 0,
        "truth_status": SAGE7D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def build_source_sage7c_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(source.get("summary", {}) or {})
    return {
        "source_outcome_status": str(source.get("outcome_status", "")),
        "game_id": str(summary.get("game_id", "")),
        "raw_comparison_events": int(summary.get("raw_comparison_events", 0) or 0),
        "independent_parameterized_contexts": int(
            summary.get("independent_parameterized_contexts", 0) or 0
        ),
        "a32_handoff_eligible_candidates": int(
            summary.get("a32_handoff_eligible_candidates", 0) or 0
        ),
        "ready_for_a32_handoff_compilation": bool(
            summary.get("ready_for_a32_handoff_compilation", False)
        ),
        "required_next_step": str(summary.get("required_next_step", "")),
        "source_scoped_mechanics_reused": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            source.get("cross_game_mechanics_imported", 0) or 0
        ),
        "source_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE7D_TRUTH_STATUS,
    }


def validate_sage7d_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE7C_SCHEMA_VERSION:
        raise ValueError("SAGE.7c schema version is not supported by SAGE.7d")
    if str(source.get("outcome_status", "")) != SAGE7C_CONTROL_DEPENDENT_DOSSIER:
        raise ValueError("SAGE.7d requires the eligible SAGE.7c dossier")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.7c source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE7C_TRUTH_STATUS:
        raise ValueError("SAGE.7c truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.7c source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.7c support must remain 0")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.7c cannot perform a scientific verdict")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.7c cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.7c wrong_confirmations must remain 0")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
        or bool(source.get("scope_generalization_performed", False))
    ):
        raise ValueError("SAGE.7c cannot import or generalize quarantined mechanics")
    if (
        bool(source.get("raw_events_counted_as_scientific_support", True))
        or bool(
            source.get("technical_repetitions_counted_as_independent_contexts", True)
        )
        or bool(source.get("candidate_eligibility_counted_as_a32_decision", True))
        or bool(source.get("parameterized_controls_counted_as_distinct_actions", True))
    ):
        raise ValueError("SAGE.7c cannot leak support, verdict, or action identity")
    if not bool(summary.get("gate_passed", False)) or not bool(
        summary.get("ready_for_a32_handoff_compilation", False)
    ):
        raise ValueError("SAGE.7c handoff gate must pass before SAGE.7d")
    if str(summary.get("required_next_step", "")) != SAGE7C_HANDOFF_REQUIRED:
        raise ValueError("SAGE.7c must explicitly request handoff compilation")
    candidates = [
        row
        for row in source.get("parameterized_candidate_dossiers", []) or []
        if isinstance(row, Mapping)
    ]
    eligible = [
        row for row in candidates if bool(row.get("a32_handoff_eligible", False))
    ]
    if len(candidates) != int(summary.get("parameterized_candidate_dossiers", 0) or 0):
        raise ValueError("SAGE.7c candidate count must be exact")
    if len(eligible) != 1 or len(eligible) != int(
        summary.get("a32_handoff_eligible_candidates", 0) or 0
    ):
        raise ValueError("SAGE.7c must expose exactly one eligible candidate")
    candidate = eligible[0]
    if (
        str(candidate.get("candidate_type", "")) != RELATIONAL_CANDIDATE_TYPE
        or str(candidate.get("a32_eligibility_status", ""))
        != A32_ELIGIBLE_CONTROL_DEPENDENT
        or str(candidate.get("autonomous_target_effect_status", ""))
        != AUTONOMOUS_EFFECT_UNRESOLVED
        or int(candidate.get("support", 0) or 0) != 0
        or bool(candidate.get("a32_decision_performed", True))
    ):
        raise ValueError("SAGE.7c eligible candidate must remain relational-only")
    if (
        int(candidate.get("independent_contexts", 0) or 0) < MIN_INDEPENDENT_CONTEXTS
        or int(candidate.get("cross_budget_replicated_contexts", 0) or 0)
        < MIN_CROSS_BUDGET_REPLICATED_CONTEXTS
        or len(candidate.get("differentiating_control_variants", []) or []) != 1
        or len(candidate.get("equivalent_control_variants", []) or []) != 1
    ):
        raise ValueError("SAGE.7c candidate does not preserve eligibility evidence")
    context_ids = set(candidate.get("context_assessment_ids", []) or [])
    contexts = [
        row
        for row in source.get("parameterized_context_assessments", []) or []
        if isinstance(row, Mapping)
        and str(row.get("context_assessment_id", "")) in context_ids
    ]
    if len(contexts) != int(candidate.get("independent_contexts", 0) or 0):
        raise ValueError("SAGE.7c candidate contexts must be complete")
    if any(
        str(row.get("context_status", "")) != CONTROL_DEPENDENT_CONTEXT
        or int(row.get("parameterized_controls_count", 0) or 0)
        != MIN_CONTROLS_PER_CONTEXT
        or not bool(row.get("all_repetitions_consistent", False))
        or int(row.get("support", 0) or 0) != 0
        or not _context_preserves_relational_controls(
            row,
            differentiating=candidate.get("differentiating_control_variants", []) or [],
            equivalent=candidate.get("equivalent_control_variants", []) or [],
        )
        for row in contexts
    ):
        raise ValueError("SAGE.7c candidate contexts must remain exact")
    gate = dict(source.get("gate", {}) or {})
    if not gate or not all(bool(value) for value in gate.values()):
        raise ValueError("every SAGE.7c source gate must pass")


def write_sage7d_a32_handoff(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE7D_A32_HANDOFF_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _context_preserves_relational_controls(
    context: Mapping[str, Any],
    *,
    differentiating: Sequence[Mapping[str, Any]],
    equivalent: Sequence[Mapping[str, Any]],
) -> bool:
    controls = [
        row
        for row in context.get("parameterized_controls", []) or []
        if isinstance(row, Mapping)
    ]
    expected_diff = {
        _canonical_control(row) for row in differentiating if isinstance(row, Mapping)
    }
    expected_equivalent = {
        _canonical_control(row) for row in equivalent if isinstance(row, Mapping)
    }
    observed = {_canonical_control(row): row for row in controls}
    if set(observed) != expected_diff | expected_equivalent:
        return False
    return all(
        str(observed[key].get("discrimination_status", ""))
        == DISCRIMINATING_TARGET_EFFECT
        and float(observed[key].get("effect_size", 0.0) or 0.0) > 0
        for key in expected_diff
    ) and all(
        str(observed[key].get("discrimination_status", ""))
        == NON_DISCRIMINATING_EQUAL_EFFECT
        and float(observed[key].get("effect_size", 0.0) or 0.0) == 0.0
        for key in expected_equivalent
    )


def _canonical_control(row: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "action": str(row.get("action", "")),
            "action_args": dict(row.get("action_args", {}) or {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-sage7c", default=str(DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH)
    )
    parser.add_argument("--out", default=str(DEFAULT_SAGE7D_A32_HANDOFF_PATH))
    args = parser.parse_args(argv)
    payload = run_sage7d_a32_handoff(
        source_sage7c_path=args.source_sage7c,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
