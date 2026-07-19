"""A32.7 scientific revision of the SAGE.7d parameterized relation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus
from theory.sage.third_unknown_game_a32_handoff import (
    ALLOWED_A32_7_DECISIONS,
    DEFAULT_SAGE7D_A32_HANDOFF_PATH,
    HANDOFF_READY,
    RELATIONAL_CANDIDATE_TYPE,
    SAGE7D_A32_REVIEW_REQUIRED,
    SAGE7D_HANDOFF_COMPILED,
    SAGE7D_SCHEMA_VERSION,
    SAGE7D_TRUTH_STATUS,
)
from theory.sage.third_unknown_game_parameterized_consolidation import (
    AUTONOMOUS_EFFECT_UNRESOLVED,
    CONTROL_DEPENDENT_CONTEXT,
    MIN_CONTROLS_PER_CONTEXT,
    MIN_CROSS_BUDGET_REPLICATED_CONTEXTS,
    MIN_INDEPENDENT_CONTEXTS,
)


DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH = (
    Path("diagnostics")
    / "a32"
    / "third_unknown_game_parameterized_relation_revision_decisions.json"
)

A32_7_SCHEMA_VERSION = (
    "a32.third_unknown_game_parameterized_relation_revision_decisions.v1"
)
A32_7_DECISION_SCOPE = "A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION"
A32_7_TRUTH_STATUS = (
    "SCOPED_A32_PARAMETERIZED_RELATION_DECISION_WITHOUT_EXTERNAL_ORACLE"
)

CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION = (
    "CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION"
)
KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT = (
    "KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT"
)
REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION = (
    "REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION"
)
REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION = (
    "REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION"
)

A32_7_RELATION_CONFIRMED = (
    "A32_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION_CONFIRMED"
)
A32_7_RELATION_REFUTED = "A32_CONTROL_DEPENDENT_PARAMETERIZED_RELATION_REFUTED"
A32_7_RELATION_UNRESOLVED = "A32_CONTROL_DEPENDENT_PARAMETERIZED_RELATION_UNRESOLVED"


@dataclass(frozen=True)
class A32ThirdUnknownGameParameterizedRelationDecision:
    """One scientific decision over one exact SAGE.7d handoff."""

    handoff_id: str
    candidate_key: str
    game_id: str
    candidate_type: str
    metric: str
    target_action: str
    target_action_args: Dict[str, Any]
    differentiating_control_variants: Tuple[Dict[str, Any], ...]
    equivalent_control_variants: Tuple[Dict[str, Any], ...]
    decision: str
    autonomous_target_effect_decision: str
    recommended_next_step: str
    reasons: Tuple[str, ...]
    evidence_summary: Dict[str, Any]
    scope_limits: Dict[str, Any]
    claim_dispositions: Dict[str, Any]
    source_context_assessment_ids: Tuple[str, ...]
    source_context_snapshot_hashes: Tuple[str, ...]
    source_comparison_group_ids: Tuple[str, ...]
    input_record: HypothesisRecord
    decision_record: HypothesisRecord
    scientific_review_performed: bool = True
    revision_performed: bool = True
    confirmation_performed: bool = False
    refutation_performed: bool = False
    a33_ready: bool = False
    a33_write_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "candidate_key": self.candidate_key,
            "game_id": self.game_id,
            "candidate_type": self.candidate_type,
            "metric": self.metric,
            "target_action": self.target_action,
            "target_action_args": dict(self.target_action_args),
            "differentiating_control_variants": [
                dict(row) for row in self.differentiating_control_variants
            ],
            "equivalent_control_variants": [
                dict(row) for row in self.equivalent_control_variants
            ],
            "decision": self.decision,
            "autonomous_target_effect_decision": (
                self.autonomous_target_effect_decision
            ),
            "recommended_next_step": self.recommended_next_step,
            "reasons": list(self.reasons),
            "evidence_summary": dict(self.evidence_summary),
            "scope_limits": dict(self.scope_limits),
            "claim_dispositions": dict(self.claim_dispositions),
            "source_context_assessment_ids": list(self.source_context_assessment_ids),
            "source_context_snapshot_hashes": list(self.source_context_snapshot_hashes),
            "source_comparison_group_ids": list(self.source_comparison_group_ids),
            "input_record": record_to_dict(self.input_record),
            "decision_record": record_to_dict(self.decision_record),
            "scientific_review_performed": self.scientific_review_performed,
            "revision_performed": self.revision_performed,
            "confirmation_performed": self.confirmation_performed,
            "refutation_performed": self.refutation_performed,
            "a33_ready": self.a33_ready,
            "a33_write_performed": self.a33_write_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "autonomous_target_effect_confirmed": False,
            "autonomous_target_effect_refuted": False,
            "equivalent_control_counted_as_refutation": False,
            "technical_repetitions_counted_as_support": False,
            "sage_candidate_events_counted_as_support_before_a32_review": False,
            "scope_limited_confirmation_generalized_beyond_game": False,
        }


def run_a32_third_unknown_game_parameterized_relation_revision(
    *,
    source_sage7d_path: str | Path = DEFAULT_SAGE7D_A32_HANDOFF_PATH,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Consume SAGE.7d and emit the explicit A32.7 scientific decision."""
    source = _load_json(source_sage7d_path)
    decisions = build_a32_third_unknown_game_parameterized_relation_decisions(source)
    handoffs = build_a33_parameterized_relation_handoff_candidates(decisions, source)
    gate = build_a32_7_gate(source, decisions, handoffs)
    if not gate or not all(gate.values()):
        raise ValueError("A32.7 scientific decision gate did not pass")
    summary = summarize_a32_7(decisions, handoffs, gate)
    payload = {
        "config": {
            "schema_version": A32_7_SCHEMA_VERSION,
            "source_sage7d_path": str(source_sage7d_path),
            "inputs_read": ["SAGE.7d"],
            "decision_scope": A32_7_DECISION_SCOPE,
            "artifacts_not_modified": [
                "SAGE.7d",
                "SAGE.7c",
                "SAGE.7b",
                "M2",
                "M3",
                "A33.1",
                "A33.2",
                "A33.3",
            ],
            "decision_policy": {
                "confirmation_requires_complete_sage7d_handoff": True,
                "confirmation_requires_independent_exact_contexts": True,
                "confirmation_requires_two_common_parameterized_controls": True,
                "confirmation_requires_stable_positive_differentiating_effect": True,
                "confirmation_requires_stable_zero_equivalent_control_effect": True,
                "confirmation_requires_cross_budget_replication": True,
                "scientific_support_is_one_per_independent_context": True,
                "technical_repetitions_are_not_scientific_support": True,
                "raw_sage_events_are_not_direct_scientific_support": True,
                "equivalent_control_makes_autonomous_effect_non_identifiable": True,
                "equivalent_control_is_not_refutation": True,
                "confirmation_is_game_context_metric_and_parameter_scoped": True,
                "a33_write_performed": False,
            },
        },
        "source_sage7d_summary": dict(source.get("summary", {}) or {}),
        "revision_decisions": [row.to_dict() for row in decisions],
        "input_records": [record_to_dict(row.input_record) for row in decisions],
        "decision_records": [record_to_dict(row.decision_record) for row in decisions],
        "autonomous_claim_dispositions": [
            dict(row.claim_dispositions["autonomous_parameterized_target_effect"])
            for row in decisions
        ],
        "a33_handoff_candidates": handoffs,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": summary["status"],
        "truth_status": A32_7_TRUTH_STATUS,
        "scientific_review_performed": bool(decisions),
        "revision_performed": bool(decisions),
        "confirmation_performed": any(row.confirmation_performed for row in decisions),
        "refutation_performed": any(row.refutation_performed for row in decisions),
        "a33_ready": bool(handoffs),
        "a33_write_performed": False,
        "wrong_confirmations": 0,
        "support": sum(
            row.decision_record.support
            for row in decisions
            if row.decision_record.status == HypothesisStatus.CONFIRMED
        ),
        "autonomous_target_effect_confirmed": False,
        "autonomous_target_effect_refuted": False,
        "autonomous_target_effect_status": HypothesisStatus.UNRESOLVED.value,
        "equivalent_control_counted_as_refutation": False,
        "technical_repetitions_counted_as_support": False,
        "sage_candidate_events_counted_as_support_before_a32_review": False,
        "scope_limited_confirmation_generalized_beyond_game": False,
    }
    if output_path is not None:
        write_a32_third_unknown_game_parameterized_relation_revision(
            payload, output_path
        )
    return payload


def build_a32_third_unknown_game_parameterized_relation_decisions(
    source: Mapping[str, Any],
) -> Tuple[A32ThirdUnknownGameParameterizedRelationDecision, ...]:
    validate_sage7d_parameterized_relation_revision_source(source)
    return tuple(
        decision_from_parameterized_relation_handoff(dict(handoff))
        for handoff in source.get("a32_review_handoff_items", []) or []
        if isinstance(handoff, Mapping)
    )


def decision_from_parameterized_relation_handoff(
    handoff: Mapping[str, Any],
) -> A32ThirdUnknownGameParameterizedRelationDecision:
    contexts = [
        dict(row)
        for row in handoff.get("context_manifest", []) or []
        if isinstance(row, Mapping)
    ]
    relation = dict(handoff.get("relational_contrast", {}) or {})
    reasons = parameterized_relation_revision_decision_reasons(handoff, contexts)
    decision = parameterized_relation_revision_decision_label(reasons)
    confirmed = (
        decision == CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
    )
    refuted = decision == REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
    unique_hashes = {str(row.get("context_snapshot_hash", "")) for row in contexts}
    scientific_support = len(unique_hashes) if confirmed else 0
    source_comparison_ids = tuple(
        str(comparison_id)
        for context in contexts
        for comparison_id in context.get("source_comparison_group_ids", []) or []
    )
    target_action = str(handoff.get("target_action", ""))
    target_args = dict(handoff.get("target_action_args", {}) or {})
    game_id = str(handoff.get("game_id", ""))
    metric = str(handoff.get("metric", ""))
    candidate_key = (
        f"a32.7::{game_id}::{target_action}::{_canonical_json(target_args)}::"
        f"control_dependent_parameterized_relation::{metric}"
    )
    differentiating = tuple(
        dict(row) for row in relation.get("differentiating_control_variants", []) or []
    )
    equivalent = tuple(
        dict(row) for row in relation.get("equivalent_control_variants", []) or []
    )
    diff_effects = _effects_for_controls(contexts, differentiating)
    equivalent_effects = _effects_for_controls(contexts, equivalent)
    target_signals = [
        float(control.get("target_signal", 0.0) or 0.0)
        for context in contexts
        for control in context.get("parameterized_controls", []) or []
    ]
    diff_control_signals = _control_signals_for_controls(contexts, differentiating)
    equivalent_control_signals = _control_signals_for_controls(contexts, equivalent)
    evidence_summary = {
        "source_contexts_reviewed": len(contexts),
        "independent_contexts": len(unique_hashes),
        "cross_budget_replicated_contexts": sum(
            bool(row.get("cross_budget_replicated", False)) for row in contexts
        ),
        "raw_comparison_events": int(handoff.get("raw_comparison_events", 0) or 0),
        "technical_replication_events": int(
            handoff.get("technical_replication_events", 0) or 0
        ),
        "candidate_support_events": int(
            handoff.get("candidate_support_events", 0) or 0
        ),
        "differentiating_control_effect_sizes": diff_effects,
        "equivalent_control_effect_sizes": equivalent_effects,
        "target_signals": target_signals,
        "differentiating_control_signals": diff_control_signals,
        "equivalent_control_signals": equivalent_control_signals,
        "negative_effect_events": sum(
            effect < 0 for effect in [*diff_effects, *equivalent_effects]
        ),
        "source_support_before_a32_review": int(handoff.get("support", 0) or 0),
        "scientific_support_after_a32_review": scientific_support,
        "scientific_support_basis": "ONE_PER_INDEPENDENT_EXACT_CONTEXT",
        "raw_comparison_events_promoted_directly": 0,
        "technical_replication_events_counted_as_support": 0,
    }
    scope_limits = {
        "game_id": game_id,
        "candidate_key": candidate_key,
        "candidate_type": str(handoff.get("candidate_type", "")),
        "metric": metric,
        "target_action": target_action,
        "target_action_args": target_args,
        "differentiating_control_variants": [dict(row) for row in differentiating],
        "equivalent_control_variants": [dict(row) for row in equivalent],
        "context_assessment_ids": [
            str(row.get("context_assessment_id", "")) for row in contexts
        ],
        "context_snapshot_hashes": sorted(unique_hashes),
        "budgets_observed": sorted(
            {
                int(budget)
                for row in contexts
                for budget in row.get("budgets_observed", []) or []
            }
        ),
        "not_generalized_beyond_game": True,
        "not_generalized_beyond_exact_contexts": True,
        "not_generalized_beyond_metric": True,
        "not_generalized_beyond_parameter_variants": True,
        "autonomous_target_effect_excluded_from_confirmation": True,
        "technical_repetitions_excluded_from_support": True,
    }
    claim_dispositions = {
        "control_dependent_parameterized_relation": {
            "status": (
                HypothesisStatus.CONFIRMED.value
                if confirmed
                else HypothesisStatus.REFUTED.value
                if refuted
                else HypothesisStatus.UNRESOLVED.value
            ),
            "decision": decision,
            "support": scientific_support,
            "confirmation_performed": confirmed,
            "refutation_performed": refuted,
        },
        "autonomous_parameterized_target_effect": {
            "status": HypothesisStatus.UNRESOLVED.value,
            "decision": KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT,
            "support": 0,
            "confirmation_performed": False,
            "refutation_performed": False,
            "reason": (
                "target matches an exact pre-registered equivalent control in "
                "every reviewed context"
            ),
        },
    }
    input_record = HypothesisRecord(
        key=candidate_key,
        description=(
            "SAGE.7d control-dependent parameterized relation before A32.7 review."
        ),
        status=HypothesisStatus.UNRESOLVED,
        support=0,
        contradictions=0,
        experiments_spent=0,
    )
    if confirmed:
        decision_record = HypothesisRecord(
            key=candidate_key,
            description=(
                "A32.7 confirmed only the exact tn36 target/differentiating/"
                "equivalent control relation in the eight reviewed contexts."
            ),
            status=HypothesisStatus.CONFIRMED,
            support=scientific_support,
            contradictions=0,
            experiments_spent=len(unique_hashes),
        )
        recommended_next = "SUBMIT_SCOPE_LOCKED_PARAMETERIZED_RELATION_TO_A33_4"
    elif refuted:
        decision_record = HypothesisRecord(
            key=candidate_key,
            description="A32.7 refuted the proposed parameterized relation.",
            status=HypothesisStatus.REFUTED,
            support=0,
            contradictions=max(1, evidence_summary["negative_effect_events"]),
            experiments_spent=len(unique_hashes),
        )
        recommended_next = "DO_NOT_SUBMIT_PARAMETERIZED_RELATION_TO_A33"
    else:
        decision_record = HypothesisRecord(
            key=candidate_key,
            description="A32.7 kept the proposed parameterized relation unresolved.",
            status=HypothesisStatus.UNRESOLVED,
            support=0,
            contradictions=0,
            experiments_spent=len(unique_hashes),
        )
        recommended_next = "REQUEST_MORE_EXACT_PARAMETERIZED_CONTEXTS"
    return A32ThirdUnknownGameParameterizedRelationDecision(
        handoff_id=str(handoff.get("handoff_id", "")),
        candidate_key=candidate_key,
        game_id=game_id,
        candidate_type=str(handoff.get("candidate_type", "")),
        metric=metric,
        target_action=target_action,
        target_action_args=target_args,
        differentiating_control_variants=differentiating,
        equivalent_control_variants=equivalent,
        decision=decision,
        autonomous_target_effect_decision=(
            KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT
        ),
        recommended_next_step=recommended_next,
        reasons=reasons,
        evidence_summary=evidence_summary,
        scope_limits=scope_limits,
        claim_dispositions=claim_dispositions,
        source_context_assessment_ids=tuple(
            str(row.get("context_assessment_id", "")) for row in contexts
        ),
        source_context_snapshot_hashes=tuple(sorted(unique_hashes)),
        source_comparison_group_ids=source_comparison_ids,
        input_record=input_record,
        decision_record=decision_record,
        confirmation_performed=confirmed,
        refutation_performed=refuted,
        a33_ready=confirmed,
    )


def parameterized_relation_revision_decision_reasons(
    handoff: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    reasons: List[str] = []
    relation = dict(handoff.get("relational_contrast", {}) or {})
    differentiating = [
        dict(row) for row in relation.get("differentiating_control_variants", []) or []
    ]
    equivalent = [
        dict(row) for row in relation.get("equivalent_control_variants", []) or []
    ]
    unique_hashes = {str(row.get("context_snapshot_hash", "")) for row in contexts}
    complete = (
        str(handoff.get("handoff_status", "")) == HANDOFF_READY
        and bool(handoff.get("ready_for_A32_7_review", False))
        and not bool(handoff.get("a32_decision_preselected", True))
        and len(contexts) >= MIN_INDEPENDENT_CONTEXTS
        and len(unique_hashes) == len(contexts)
        and len(differentiating) == 1
        and len(equivalent) == 1
    )
    reasons.append(
        "sage7d_parameterized_relation_review_dossier_complete"
        if complete
        else "sage7d_parameterized_relation_review_dossier_incomplete"
    )
    if len(unique_hashes) == len(contexts):
        reasons.append("reviewed_contexts_are_independent")
    cross_budget = sum(
        bool(row.get("cross_budget_replicated", False)) for row in contexts
    )
    if cross_budget >= MIN_CROSS_BUDGET_REPLICATED_CONTEXTS:
        reasons.append("cross_budget_replication_threshold_met")
    else:
        reasons.append("cross_budget_replication_threshold_not_met")
    two_controls = bool(contexts) and all(
        len(row.get("parameterized_controls", []) or []) == MIN_CONTROLS_PER_CONTEXT
        for row in contexts
    )
    if two_controls:
        reasons.append("two_pre_registered_controls_preserved_per_context")
    diff_effects = _effects_for_controls(contexts, differentiating)
    equivalent_effects = _effects_for_controls(contexts, equivalent)
    stable_positive = (
        len(diff_effects) == len(contexts)
        and bool(diff_effects)
        and min(diff_effects) > 0.0
        and max(diff_effects) - min(diff_effects) == 0.0
    )
    stable_equivalent = (
        len(equivalent_effects) == len(contexts)
        and bool(equivalent_effects)
        and set(equivalent_effects) == {0.0}
    )
    if stable_positive:
        reasons.append("differentiating_control_effect_is_stable_positive")
    elif any(effect < 0 for effect in diff_effects):
        reasons.append("differentiating_control_contradiction_observed")
    else:
        reasons.append("differentiating_control_effect_not_stable_positive")
    if stable_equivalent:
        reasons.extend(
            [
                "equivalent_control_matches_target_in_every_context",
                "autonomous_target_effect_is_not_identifiable",
                "equivalent_control_is_not_refutation",
            ]
        )
    elif any(effect < 0 for effect in equivalent_effects):
        reasons.append("equivalent_control_contradiction_observed")
    else:
        reasons.append("equivalent_control_effect_not_stable_zero")
    all_contexts_exact = bool(contexts) and all(
        str(row.get("context_status", "")) == CONTROL_DEPENDENT_CONTEXT
        and bool(row.get("all_repetitions_consistent", False))
        and int(row.get("support", 0) or 0) == 0
        for row in contexts
    )
    if all_contexts_exact:
        reasons.append("all_contexts_exact_consistent_and_support_free_before_review")
    negative = any(effect < 0 for effect in [*diff_effects, *equivalent_effects])
    if negative:
        reasons.append("parameterized_relation_contradiction_observed")
    if (
        complete
        and cross_budget >= MIN_CROSS_BUDGET_REPLICATED_CONTEXTS
        and two_controls
        and stable_positive
        and stable_equivalent
        and all_contexts_exact
        and not negative
    ):
        reasons.append(
            "scope_limited_parameterized_relation_confirmation_criteria_satisfied"
        )
    reasons.extend(
        [
            "technical_repetitions_not_counted_as_support",
            "raw_sage_events_not_counted_as_support_before_a32_review",
            "scientific_support_limited_to_independent_exact_contexts",
            "confirmation_not_generalized_beyond_exact_parameter_variants",
        ]
    )
    return tuple(reasons)


def parameterized_relation_revision_decision_label(
    reasons: Sequence[str],
) -> str:
    reason_set = set(reasons)
    if (
        "parameterized_relation_contradiction_observed" in reason_set
        or "differentiating_control_contradiction_observed" in reason_set
        or "equivalent_control_contradiction_observed" in reason_set
    ):
        return REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
    if "sage7d_parameterized_relation_review_dossier_incomplete" in reason_set:
        return REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION
    if (
        "scope_limited_parameterized_relation_confirmation_criteria_satisfied"
        in reason_set
    ):
        return CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
    return REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION


def build_a33_parameterized_relation_handoff_candidates(
    decisions: Sequence[A32ThirdUnknownGameParameterizedRelationDecision],
    source: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    source_handoffs = {
        str(row.get("handoff_id", "")): row
        for row in source.get("a32_review_handoff_items", []) or []
        if isinstance(row, Mapping)
    }
    handoffs: List[Dict[str, Any]] = []
    for decision in decisions:
        if not decision.a33_ready:
            continue
        source_handoff = source_handoffs[decision.handoff_id]
        handoffs.append(
            {
                "handoff_id": decision.handoff_id,
                "candidate_key": decision.candidate_key,
                "game_id": decision.game_id,
                "registry_entry_type": (
                    "CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST"
                ),
                "candidate_type": decision.candidate_type,
                "metric": decision.metric,
                "target_action": decision.target_action,
                "target_action_args": dict(decision.target_action_args),
                "differentiating_control_variants": [
                    dict(row) for row in decision.differentiating_control_variants
                ],
                "equivalent_control_variants": [
                    dict(row) for row in decision.equivalent_control_variants
                ],
                "status": HypothesisStatus.CONFIRMED.value,
                "support": decision.decision_record.support,
                "contradictions": decision.decision_record.contradictions,
                "experiments_spent": decision.decision_record.experiments_spent,
                "scoped_claim": decision.decision_record.description,
                "context_manifest": [
                    dict(row)
                    for row in source_handoff.get("context_manifest", []) or []
                ],
                "source_context_assessment_ids": list(
                    decision.source_context_assessment_ids
                ),
                "source_context_snapshot_hashes": list(
                    decision.source_context_snapshot_hashes
                ),
                "source_comparison_group_ids": list(
                    decision.source_comparison_group_ids
                ),
                "autonomous_target_effect_status": (HypothesisStatus.UNRESOLVED.value),
                "autonomous_target_effect_confirmed": False,
                "equivalent_control_counted_as_refutation": False,
                "ready_for_A33_4_registry_review": True,
                "a33_write_performed": False,
                "not_generalized_beyond_game": True,
                "not_generalized_beyond_exact_contexts": True,
                "not_generalized_beyond_metric": True,
                "not_generalized_beyond_parameter_variants": True,
                "technical_repetitions_counted_as_support": False,
                "parameterized_controls_counted_as_distinct_actions": False,
            }
        )
    return handoffs


def build_a32_7_gate(
    source: Mapping[str, Any],
    decisions: Sequence[A32ThirdUnknownGameParameterizedRelationDecision],
    handoffs: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    expected_contexts = int(source.get("summary", {}).get("handoff_contexts", 0) or 0)
    scientific_support = sum(
        row.decision_record.support
        for row in decisions
        if row.decision_record.status == HypothesisStatus.CONFIRMED
    )
    return {
        "source_sage7d_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source.get("gate", {}).values()),
        "every_handoff_decided_once": len(decisions)
        == int(source.get("summary", {}).get("handoff_items", 0) or 0)
        == 1,
        "input_records_preserve_sage_support_zero": all(
            row.input_record.status == HypothesisStatus.UNRESOLVED
            and row.input_record.support == 0
            for row in decisions
        ),
        "scientific_support_equals_independent_contexts": scientific_support
        == expected_contexts
        == 8,
        "confirmation_is_relational_only": all(
            row.confirmation_performed
            and not bool(
                row.claim_dispositions["autonomous_parameterized_target_effect"][
                    "confirmation_performed"
                ]
            )
            for row in decisions
        ),
        "equivalent_control_keeps_autonomous_effect_unresolved": all(
            row.claim_dispositions["autonomous_parameterized_target_effect"]["status"]
            == HypothesisStatus.UNRESOLVED.value
            and not bool(
                row.claim_dispositions["autonomous_parameterized_target_effect"][
                    "refutation_performed"
                ]
            )
            for row in decisions
        ),
        "technical_repetitions_excluded_from_support": all(
            int(row.evidence_summary["technical_replication_events_counted_as_support"])
            == 0
            and bool(
                row.scope_limits.get(
                    "technical_repetitions_excluded_from_support", False
                )
            )
            for row in decisions
        ),
        "a33_handoff_matches_confirmed_relational_decision": len(handoffs)
        == sum(row.a33_ready for row in decisions)
        == 1
        and all(
            row.get("registry_entry_type")
            == "CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST"
            and bool(row.get("ready_for_A33_4_registry_review", False))
            and int(row.get("support", 0) or 0) == expected_contexts
            for row in handoffs
        ),
        "no_a33_write_performed": all(
            not bool(row.get("a33_write_performed", True)) for row in handoffs
        )
        and all(not row.a33_write_performed for row in decisions),
        "scope_locks_preserved": all(
            bool(row.scope_limits.get("not_generalized_beyond_game", False))
            and bool(
                row.scope_limits.get("not_generalized_beyond_exact_contexts", False)
            )
            and bool(
                row.scope_limits.get("not_generalized_beyond_parameter_variants", False)
            )
            and bool(
                row.scope_limits.get(
                    "autonomous_target_effect_excluded_from_confirmation", False
                )
            )
            for row in decisions
        ),
        "wrong_confirmations_zero": all(
            row.wrong_confirmations == 0 for row in decisions
        ),
    }


def summarize_a32_7(
    decisions: Sequence[A32ThirdUnknownGameParameterizedRelationDecision],
    handoffs: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    confirmed = [row for row in decisions if row.confirmation_performed]
    refuted = [row for row in decisions if row.refutation_performed]
    unresolved = [
        row
        for row in decisions
        if row.decision_record.status == HypothesisStatus.UNRESOLVED
    ]
    if confirmed and len(confirmed) == len(decisions):
        status = "CONFIRMED"
        outcome = A32_7_RELATION_CONFIRMED
    elif refuted and len(refuted) == len(decisions):
        status = "REFUTED"
        outcome = A32_7_RELATION_REFUTED
    else:
        status = "UNRESOLVED"
        outcome = A32_7_RELATION_UNRESOLVED
    return {
        "source_handoffs_consumed": len(decisions),
        "scientific_revision_decisions": len(decisions),
        "scope_limited_parameterized_relations_confirmed": len(confirmed),
        "parameterized_relations_refuted": len(refuted),
        "parameterized_relations_unresolved": len(unresolved),
        "autonomous_target_effects_confirmed": 0,
        "autonomous_target_effects_kept_unresolved": len(decisions),
        "decision_records_confirmed": len(confirmed),
        "decision_records_unresolved": len(unresolved),
        "decision_records_refuted": len(refuted),
        "scientific_support_counted_by_a32": sum(
            row.decision_record.support for row in confirmed
        ),
        "raw_comparison_events_promoted_directly": 0,
        "technical_replication_events_counted_as_support": 0,
        "independent_contexts_counted_as_support": sum(
            int(row.evidence_summary["independent_contexts"]) for row in confirmed
        ),
        "a33_ready_candidates": len(handoffs),
        "a33_write_performed": False,
        "equivalent_control_counted_as_refutation": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "status": status,
        "outcome_status": outcome,
    }


def validate_sage7d_parameterized_relation_revision_source(
    source: Mapping[str, Any],
) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE7D_SCHEMA_VERSION:
        raise ValueError("SAGE.7d schema version is not supported by A32.7")
    if str(source.get("outcome_status", "")) != SAGE7D_HANDOFF_COMPILED:
        raise ValueError("A32.7 requires the completed SAGE.7d handoff")
    if str(source.get("truth_status", "")) != SAGE7D_TRUTH_STATUS:
        raise ValueError("SAGE.7d truth must remain unevaluated before A32.7")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.7d must remain candidate-only before A32.7")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.7d source must remain unresolved")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.7d support must remain 0 before A32.7")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.7d cannot perform the A32.7 decision")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.7d cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.7d wrong_confirmations must remain 0")
    forbidden = (
        "handoff_counted_as_scientific_support",
        "candidate_eligibility_counted_as_a32_decision",
        "a32_decision_preselected",
        "technical_repetitions_counted_as_independent_contexts",
        "parameterized_controls_counted_as_distinct_actions",
    )
    if any(bool(source.get(field, False)) for field in forbidden):
        raise ValueError(
            "SAGE.7d cannot pre-count support, verdict, or action identity"
        )
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
        or bool(source.get("scope_generalization_performed", False))
    ):
        raise ValueError("SAGE.7d cannot import or generalize quarantined mechanics")
    if not bool(summary.get("gate_passed", False)) or not bool(
        summary.get("ready_for_a32_7_scientific_review", False)
    ):
        raise ValueError("SAGE.7d review gate must pass before A32.7")
    if str(summary.get("required_next_step", "")) != SAGE7D_A32_REVIEW_REQUIRED:
        raise ValueError("SAGE.7d must explicitly request A32.7 review")
    handoffs = [
        row
        for row in source.get("a32_review_handoff_items", []) or []
        if isinstance(row, Mapping)
    ]
    if len(handoffs) != int(summary.get("handoff_items", 0) or 0) or len(handoffs) != 1:
        raise ValueError("A32.7 requires exactly one SAGE.7d handoff")
    handoff = handoffs[0]
    contexts = [
        row
        for row in handoff.get("context_manifest", []) or []
        if isinstance(row, Mapping)
    ]
    relation = dict(handoff.get("relational_contrast", {}) or {})
    if (
        str(handoff.get("candidate_type", "")) != RELATIONAL_CANDIDATE_TYPE
        or str(handoff.get("handoff_status", "")) != HANDOFF_READY
        or not bool(handoff.get("ready_for_A32_7_review", False))
        or bool(handoff.get("a32_decision_preselected", True))
        or int(handoff.get("support", 0) or 0) != 0
        or bool(handoff.get("revision_performed", True))
    ):
        raise ValueError("SAGE.7d handoff must remain review-ready and undecided")
    if tuple(handoff.get("allowed_a32_7_decisions", []) or []) != tuple(
        ALLOWED_A32_7_DECISIONS
    ):
        raise ValueError("SAGE.7d allowed decision set must remain exact")
    if (
        str(relation.get("autonomous_target_effect_status", ""))
        != AUTONOMOUS_EFFECT_UNRESOLVED
        or len(relation.get("differentiating_control_variants", []) or []) != 1
        or len(relation.get("equivalent_control_variants", []) or []) != 1
    ):
        raise ValueError("SAGE.7d relation roles must remain exact")
    if len(contexts) != int(handoff.get("independent_contexts", 0) or 0):
        raise ValueError("SAGE.7d context count must be exact")
    hashes = [str(row.get("context_snapshot_hash", "")) for row in contexts]
    if "" in hashes or len(hashes) != len(set(hashes)):
        raise ValueError("SAGE.7d contexts must be independent and identified")
    if any(
        str(row.get("context_status", "")) != CONTROL_DEPENDENT_CONTEXT
        or len(row.get("parameterized_controls", []) or []) != MIN_CONTROLS_PER_CONTEXT
        or not bool(row.get("all_repetitions_consistent", False))
        or int(row.get("support", 0) or 0) != 0
        for row in contexts
    ):
        raise ValueError("SAGE.7d contexts must remain exact and support-free")
    raw_events = sum(int(row.get("raw_event_count", 0) or 0) for row in contexts)
    repetitions = sum(
        int(row.get("technical_replication_events", 0) or 0) for row in contexts
    )
    if raw_events != int(
        handoff.get("raw_comparison_events", 0) or 0
    ) or repetitions != int(handoff.get("technical_replication_events", 0) or 0):
        raise ValueError("SAGE.7d raw and repetition counts must be exact")
    gate = dict(source.get("gate", {}) or {})
    if not gate or not all(bool(value) for value in gate.values()):
        raise ValueError("every SAGE.7d source gate must pass")


def _effects_for_controls(
    contexts: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> List[float]:
    wanted = {_canonical_control(row) for row in variants}
    return [
        float(control.get("effect_size", 0.0) or 0.0)
        for context in contexts
        for control in context.get("parameterized_controls", []) or []
        if _canonical_control(control) in wanted
    ]


def _control_signals_for_controls(
    contexts: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> List[float]:
    wanted = {_canonical_control(row) for row in variants}
    return [
        float(control.get("control_signal", 0.0) or 0.0)
        for context in contexts
        for control in context.get("parameterized_controls", []) or []
        if _canonical_control(control) in wanted
    ]


def _canonical_control(row: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "action": str(row.get("action", "")),
            "action_args": dict(row.get("action_args", {}) or {}),
        }
    )


def record_to_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_third_unknown_game_parameterized_relation_revision(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sage7d", default=str(DEFAULT_SAGE7D_A32_HANDOFF_PATH))
    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH
        ),
    )
    args = parser.parse_args(argv)
    payload = run_a32_third_unknown_game_parameterized_relation_revision(
        source_sage7d_path=args.source_sage7d,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
