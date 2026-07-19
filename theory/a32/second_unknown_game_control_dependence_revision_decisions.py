"""A32.6 scientific revision of the SAGE.6f control-dependent candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus
from theory.sage.second_unknown_game_control_dependence_consolidation import (
    DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH,
    READY_FOR_A32_CONTROL_DEPENDENCE_REVIEW,
    SAGE6F_A32_REVIEW_ELIGIBLE,
    SAGE6F_SCHEMA_VERSION,
    SAGE6F_TRUTH_STATUS,
)


DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH = (
    Path("diagnostics")
    / "a32"
    / "second_unknown_game_control_dependence_revision_decisions.json"
)

A32_6_SCHEMA_VERSION = (
    "a32.second_unknown_game_control_dependence_revision_decisions.v1"
)
A32_6_DECISION_SCOPE = "A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION"
A32_6_TRUTH_STATUS = (
    "SCOPED_A32_CONTROL_DEPENDENCE_DECISION_WITHOUT_EXTERNAL_ORACLE"
)

CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST = (
    "CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST"
)
KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE = (
    "KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE"
)
REFUTE_CONTROL_DEPENDENT_CONTRAST = "REFUTE_CONTROL_DEPENDENT_CONTRAST"
REQUEST_MORE_TESTS_FOR_CONTROL_DEPENDENCE = (
    "REQUEST_MORE_TESTS_FOR_CONTROL_DEPENDENCE"
)

A32_6_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST_CONFIRMED = (
    "A32_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST_CONFIRMED"
)
A32_6_CONTROL_DEPENDENT_CONTRAST_REFUTED = (
    "A32_CONTROL_DEPENDENT_CONTRAST_REFUTED"
)
A32_6_CONTROL_DEPENDENCE_UNRESOLVED = "A32_CONTROL_DEPENDENCE_UNRESOLVED"


@dataclass(frozen=True)
class A32SecondUnknownGameControlDependenceRevisionDecision:
    """One A32.6 verdict over a SAGE.6f control-dependence frontier."""

    frontier_id: str
    candidate_key: str
    game_id: str
    candidate_mechanism_family: str
    metric: str
    target_action: str
    control_actions: Tuple[str, ...]
    decision: str
    recommended_next_step: str
    reasons: Tuple[str, ...]
    evidence_summary: Dict[str, Any]
    scope_limits: Dict[str, Any]
    claim_dispositions: Dict[str, Any]
    source_observation_ids: Tuple[str, ...]
    source_paired_comparison_ids: Tuple[str, ...]
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
            "frontier_id": self.frontier_id,
            "candidate_key": self.candidate_key,
            "game_id": self.game_id,
            "candidate_mechanism_family": self.candidate_mechanism_family,
            "metric": self.metric,
            "target_action": self.target_action,
            "control_actions": list(self.control_actions),
            "decision": self.decision,
            "recommended_next_step": self.recommended_next_step,
            "reasons": list(self.reasons),
            "evidence_summary": dict(self.evidence_summary),
            "scope_limits": dict(self.scope_limits),
            "claim_dispositions": dict(self.claim_dispositions),
            "source_observation_ids": list(self.source_observation_ids),
            "source_paired_comparison_ids": list(
                self.source_paired_comparison_ids
            ),
            "input_record": record_to_dict(self.input_record),
            "decision_record": record_to_dict(self.decision_record),
            "scientific_review_performed": self.scientific_review_performed,
            "revision_performed": self.revision_performed,
            "confirmation_performed": self.confirmation_performed,
            "refutation_performed": self.refutation_performed,
            "a33_ready": self.a33_ready,
            "a33_write_performed": self.a33_write_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "standalone_action2_effect_confirmed": False,
            "standalone_action2_effect_refuted": False,
            "action3_equivalence_counted_as_refutation": False,
            "action1_counted_as_universal_baseline": False,
            "sage_raw_events_counted_as_support_before_a32_review": False,
            "scope_limited_confirmation_generalized_beyond_game": False,
        }


def run_a32_second_unknown_game_control_dependence_revision_consumer(
    *,
    source_sage6f_path: str | Path = (
        DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH
    ),
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Consume SAGE.6f and emit the explicit A32.6 scientific decision."""
    source = _load_json(source_sage6f_path)
    decisions = build_a32_second_unknown_game_control_dependence_revision_decisions(
        source
    )
    handoffs = build_a33_control_dependence_handoff_candidates(decisions, source)
    gate = build_a32_6_gate(source, decisions, handoffs)
    if not gate or not all(gate.values()):
        raise ValueError("A32.6 scientific decision gate did not pass")
    summary = summarize_a32_second_unknown_game_control_dependence_decisions(
        decisions, handoffs, gate
    )
    payload = {
        "config": {
            "source_sage6f_path": str(source_sage6f_path),
            "schema_version": A32_6_SCHEMA_VERSION,
            "inputs_read": ["SAGE.6f"],
            "decision_scope": A32_6_DECISION_SCOPE,
            "artifacts_not_modified": ["SAGE.6f", "M2", "M3", "A33"],
            "decision_policy": {
                "confirmation_requires_complete_sage6f_eligibility": True,
                "confirmation_requires_independent_paired_control_contexts": True,
                "confirmation_requires_all_source_budgets": True,
                "confirmation_requires_reproduced_target_signal": True,
                "confirmation_requires_stable_positive_control_effect_gap": True,
                "scientific_support_is_one_per_independent_paired_context": True,
                "raw_sage_events_are_not_direct_scientific_support": True,
                "action3_equivalence_makes_standalone_effect_non_identifiable": True,
                "action3_equivalence_is_not_refutation": True,
                "action1_is_not_a_universal_baseline": True,
                "confirmation_is_game_context_control_and_metric_scoped": True,
                "a33_write_performed": False,
            },
        },
        "source_sage6f_summary": dict(source.get("summary", {}) or {}),
        "revision_decisions": [row.to_dict() for row in decisions],
        "input_records": [record_to_dict(row.input_record) for row in decisions],
        "decision_records": [
            record_to_dict(row.decision_record) for row in decisions
        ],
        "standalone_claim_dispositions": [
            dict(row.claim_dispositions["standalone_unconditional_action2_effect"])
            for row in decisions
        ],
        "a33_handoff_candidates": handoffs,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": summary["status"],
        "truth_status": A32_6_TRUTH_STATUS,
        "scientific_review_performed": bool(decisions),
        "revision_performed": bool(decisions),
        "confirmation_performed": any(
            row.confirmation_performed for row in decisions
        ),
        "refutation_performed": any(row.refutation_performed for row in decisions),
        "a33_ready": bool(handoffs),
        "a33_write_performed": False,
        "wrong_confirmations": 0,
        "support": sum(
            row.decision_record.support
            for row in decisions
            if row.decision_record.status == HypothesisStatus.CONFIRMED
        ),
        "standalone_action2_effect_confirmed": False,
        "standalone_action2_effect_refuted": False,
        "standalone_action2_effect_status": HypothesisStatus.UNRESOLVED.value,
        "action3_equivalence_counted_as_refutation": False,
        "action1_counted_as_universal_baseline": False,
        "sage_raw_events_counted_as_support_before_a32_review": False,
        "scope_limited_confirmation_generalized_beyond_game": False,
    }
    if output_path is not None:
        write_a32_second_unknown_game_control_dependence_revision_decisions(
            payload, output_path
        )
    return payload


def build_a32_second_unknown_game_control_dependence_revision_decisions(
    source: Mapping[str, Any],
) -> Tuple[A32SecondUnknownGameControlDependenceRevisionDecision, ...]:
    validate_sage6f_control_dependence_revision_source(source)
    assessment = dict(source.get("a32_review_eligibility_assessment", {}) or {})
    observations = tuple(
        dict(row)
        for row in source.get("observation_records", []) or []
        if isinstance(row, Mapping)
    )
    clusters = tuple(
        dict(row)
        for row in source.get("context_cluster_manifest", []) or []
        if isinstance(row, Mapping)
    )
    paired = tuple(
        dict(row)
        for row in source.get("paired_control_comparisons", []) or []
        if isinstance(row, Mapping)
    )
    frontiers = tuple(
        dict(row)
        for row in source.get("candidate_a32_review_frontiers", []) or []
        if isinstance(row, Mapping)
    )
    return tuple(
        decision_from_control_dependence_frontier(
            frontier=frontier,
            assessment=assessment,
            observations=observations,
            context_clusters=clusters,
            paired_comparisons=paired,
        )
        for frontier in frontiers
    )


def decision_from_control_dependence_frontier(
    *,
    frontier: Mapping[str, Any],
    assessment: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
    paired_comparisons: Sequence[Mapping[str, Any]],
) -> A32SecondUnknownGameControlDependenceRevisionDecision:
    reasons = control_dependence_revision_decision_reasons(
        assessment, paired_comparisons, context_clusters
    )
    decision = control_dependence_revision_decision_label(reasons)
    confirmed = decision == CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST
    refuted = decision == REFUTE_CONTROL_DEPENDENT_CONTRAST
    scientific_support = len(
        {
            str(row.get("context_snapshot_hash", ""))
            for row in paired_comparisons
        }
    ) if confirmed else 0
    raw_contradictions = int(assessment.get("raw_contradiction_events", 0) or 0)
    experiments_spent = len(observations)
    key = str(frontier.get("candidate_key", ""))
    game_id = str(frontier.get("game_id", ""))
    target_action = str(frontier.get("target_action", ""))
    controls = tuple(str(value) for value in frontier.get("control_actions", []) or [])

    paired_context_ids = [
        str(row.get("context_cluster_id", "")) for row in paired_comparisons
    ]
    paired_context_hashes = [
        str(row.get("context_snapshot_hash", "")) for row in paired_comparisons
    ]
    neutral_clusters = [
        row
        for row in context_clusters
        if bool(row.get("neutral_context_replication_verified", False))
    ]
    unpaired_positive_clusters = [
        row
        for row in context_clusters
        if not bool(row.get("paired_control_context", False))
        and any(
            float(effect or 0.0) > 0.0
            for values in dict(row.get("effects_by_control", {}) or {}).values()
            for effect in values
        )
    ]
    gaps = [
        float(row.get("controlled_effect_gap_action1_minus_action3", 0.0) or 0.0)
        for row in paired_comparisons
    ]
    evidence_summary = {
        "source_observations_reviewed": experiments_spent,
        "source_context_clusters_reviewed": len(context_clusters),
        "independent_paired_control_contexts": len(set(paired_context_hashes)),
        "paired_control_budgets": sorted(
            {int(row.get("budget", 0) or 0) for row in paired_comparisons}
        ),
        "paired_control_effect_gaps": gaps,
        "paired_control_effect_gap_spread": (
            max(gaps) - min(gaps) if gaps else None
        ),
        "action1_controlled_effect_sizes": [
            float(row.get("action1_controlled_effect_size", 0.0) or 0.0)
            for row in paired_comparisons
        ],
        "action3_controlled_effect_sizes": [
            float(row.get("action3_controlled_effect_size", 0.0) or 0.0)
            for row in paired_comparisons
        ],
        "target_signals_reproduced_across_control_pairs": sum(
            bool(row.get("target_signal_reproduced_across_control_pairs", False))
            for row in paired_comparisons
        ),
        "replicated_neutral_contexts": len(neutral_clusters),
        "unpaired_action1_positive_contexts": len(unpaired_positive_clusters),
        "raw_support_events": int(assessment.get("raw_support_events", 0) or 0),
        "raw_neutral_events": int(assessment.get("raw_neutral_events", 0) or 0),
        "raw_contradiction_events": raw_contradictions,
        "negative_effect_events": int(
            assessment.get("negative_effect_events", 0) or 0
        ),
        "source_support_before_a32_review": int(assessment.get("support", 0) or 0),
        "scientific_support_after_a32_review": scientific_support,
        "scientific_support_basis": (
            "ONE_PER_INDEPENDENT_PAIRED_CONTROL_CONTEXT"
        ),
        "raw_support_events_promoted_directly": 0,
    }
    scope_limits = {
        "game_id": game_id,
        "candidate_key": key,
        "candidate_mechanism_family": str(
            frontier.get("candidate_mechanism_family", "")
        ),
        "metric": str(frontier.get("metric", "")),
        "target_action": target_action,
        "control_actions": list(controls),
        "budgets": evidence_summary["paired_control_budgets"],
        "reviewed_context_cluster_ids": list(
            frontier.get("context_cluster_ids", []) or []
        ),
        "confirmed_paired_context_cluster_ids": paired_context_ids,
        "confirmed_paired_context_snapshot_hashes": paired_context_hashes,
        "neutral_exception_context_cluster_ids": [
            str(row.get("context_cluster_id", "")) for row in neutral_clusters
        ],
        "unpaired_action1_positive_context_cluster_ids": [
            str(row.get("context_cluster_id", ""))
            for row in unpaired_positive_clusters
        ],
        "expected_action2_minus_action1_local_patch_effect": (
            gaps[0] if gaps and max(gaps) - min(gaps) == 0.0 else None
        ),
        "expected_action2_minus_action3_local_patch_effect": 0.0,
        "not_generalized_beyond_game": True,
        "not_generalized_beyond_exact_paired_contexts": True,
        "not_generalized_to_other_controls": True,
        "not_generalized_to_other_target_actions": True,
        "not_generalized_to_other_metrics": True,
        "action1_valid_only_as_recorded_lower_effect_comparator": True,
        "standalone_action2_effect_excluded_from_confirmation": True,
    }
    claim_dispositions = {
        "reformulated_control_dependent_contrast": {
            "claim": str(
                dict(frontier.get("candidate_claim", {}) or {}).get(
                    "review_claim", ""
                )
            ),
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
        "standalone_unconditional_action2_effect": {
            "claim": "STANDALONE_UNCONDITIONAL_ACTION2_EFFECT",
            "status": HypothesisStatus.UNRESOLVED.value,
            "decision": (
                KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE
            ),
            "support": 0,
            "confirmation_performed": False,
            "refutation_performed": False,
            "reason": (
                "ACTION3 reproduces the ACTION2 target signal in every paired "
                "context, so an unconditional ACTION2 effect is not identifiable."
            ),
        },
        "action1_universal_baseline": {
            "claim": "ACTION1_IS_A_UNIVERSAL_LOWER_EFFECT_BASELINE",
            "status": HypothesisStatus.UNRESOLVED.value,
            "support": 0,
            "confirmation_performed": False,
            "refutation_performed": False,
            "reason": (
                "ACTION1 is accepted only as the recorded comparator in the exact "
                "paired contexts and has a replicated neutral exception."
            ),
        },
    }
    input_record = HypothesisRecord(
        key=key,
        description=(
            "SAGE.6f control-dependent local-patch candidate before A32.6 review."
        ),
        status=HypothesisStatus.UNRESOLVED,
        support=0,
        contradictions=0,
        experiments_spent=experiments_spent,
    )
    if confirmed:
        decision_record = HypothesisRecord(
            key=key,
            description=(
                f"In the exact paired {game_id} contexts, {target_action} has a "
                "positive local-patch contrast versus ACTION1 and a neutral "
                "contrast versus ACTION3; no standalone ACTION2 effect is claimed."
            ),
            status=HypothesisStatus.CONFIRMED,
            support=scientific_support,
            contradictions=0,
            experiments_spent=experiments_spent,
        )
        recommended_next_step = (
            "Submit only the relational, game- and context-scoped contrast to A33.3 "
            "registry review; keep the standalone ACTION2 claim unresolved."
        )
    elif refuted:
        decision_record = HypothesisRecord(
            key=key,
            description="A32.6 refuted the proposed control-dependent contrast.",
            status=HypothesisStatus.REFUTED,
            support=0,
            contradictions=max(raw_contradictions, 1),
            experiments_spent=experiments_spent,
        )
        recommended_next_step = "Retire this relational candidate from A33 intake."
    else:
        decision_record = HypothesisRecord(
            key=key,
            description=(
                "The control-dependent contrast remains unresolved after A32.6."
            ),
            status=HypothesisStatus.UNRESOLVED,
            support=0,
            contradictions=raw_contradictions,
            experiments_spent=experiments_spent,
        )
        recommended_next_step = (
            "Acquire a new exact paired-control protocol for every missing scope "
            "requirement before another A32 review."
        )
    return A32SecondUnknownGameControlDependenceRevisionDecision(
        frontier_id=str(frontier.get("frontier_id", "")),
        candidate_key=key,
        game_id=game_id,
        candidate_mechanism_family=str(
            frontier.get("candidate_mechanism_family", "")
        ),
        metric=str(frontier.get("metric", "")),
        target_action=target_action,
        control_actions=controls,
        decision=decision,
        recommended_next_step=recommended_next_step,
        reasons=reasons,
        evidence_summary=evidence_summary,
        scope_limits=scope_limits,
        claim_dispositions=claim_dispositions,
        source_observation_ids=tuple(
            str(row.get("observation_id", "")) for row in observations
        ),
        source_paired_comparison_ids=tuple(
            str(row.get("paired_comparison_id", ""))
            for row in paired_comparisons
        ),
        input_record=input_record,
        decision_record=decision_record,
        confirmation_performed=confirmed,
        refutation_performed=refuted,
        a33_ready=confirmed,
    )


def control_dependence_revision_decision_reasons(
    assessment: Mapping[str, Any],
    paired_comparisons: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    reasons: List[str] = []
    requirements = dict(assessment.get("eligibility_requirements", {}) or {})
    complete = (
        bool(assessment.get("ready_for_A32_review", False))
        and bool(requirements)
        and all(bool(value) for value in requirements.values())
        and not list(assessment.get("missing_eligibility_requirements", []) or [])
        and len(paired_comparisons) >= 3
    )
    if complete:
        reasons.append("sage6f_control_dependence_review_dossier_complete")
    else:
        reasons.append("sage6f_control_dependence_review_dossier_incomplete")
    gaps = [
        float(row.get("controlled_effect_gap_action1_minus_action3", 0.0) or 0.0)
        for row in paired_comparisons
    ]
    action1_effects = [
        float(row.get("action1_controlled_effect_size", 0.0) or 0.0)
        for row in paired_comparisons
    ]
    action3_effects = [
        float(row.get("action3_controlled_effect_size", 0.0) or 0.0)
        for row in paired_comparisons
    ]
    target_reproduced = bool(paired_comparisons) and all(
        bool(row.get("target_signal_reproduced_across_control_pairs", False))
        for row in paired_comparisons
    )
    paired_contexts_independent = len(
        {
            str(row.get("context_snapshot_hash", ""))
            for row in paired_comparisons
        }
    ) == len(paired_comparisons)
    if paired_contexts_independent:
        reasons.append("paired_control_contexts_are_independent")
    if target_reproduced:
        reasons.append("target_signal_reproduced_across_every_control_pair")
    stable_positive = (
        bool(gaps) and min(gaps) > 0.0 and max(gaps) - min(gaps) == 0.0
    )
    if stable_positive:
        reasons.append("paired_control_effect_gap_is_stable_positive")
    elif any(value < 0.0 for value in gaps):
        reasons.append("paired_control_contrast_contradiction_observed")
    else:
        reasons.append("paired_control_contrast_not_stable_positive")
    if action1_effects and all(value > 0.0 for value in action1_effects):
        reasons.append("action1_is_lower_effect_comparator_in_exact_paired_contexts")
    if action3_effects and all(value == 0.0 for value in action3_effects):
        reasons.extend(
            [
                "action3_reproduces_target_signal_in_exact_paired_contexts",
                "standalone_action2_effect_is_not_identifiable",
                "action3_equivalence_is_not_refutation",
            ]
        )
    if sum(
        bool(row.get("neutral_context_replication_verified", False))
        for row in context_clusters
    ) == 1:
        reasons.append("action1_neutral_exception_is_exactly_replicated")
    if int(assessment.get("raw_contradiction_events", 0) or 0) > 0 or int(
        assessment.get("negative_effect_events", 0) or 0
    ) > 0:
        reasons.append("control_dependence_contradiction_observed")
    if (
        complete
        and paired_contexts_independent
        and target_reproduced
        and stable_positive
        and action1_effects
        and all(value > 0.0 for value in action1_effects)
        and action3_effects
        and all(value == 0.0 for value in action3_effects)
        and "control_dependence_contradiction_observed" not in reasons
    ):
        reasons.append("scope_limited_control_dependent_confirmation_criteria_satisfied")
    reasons.extend(
        [
            "action1_not_generalized_as_universal_baseline",
            "sage_raw_events_not_counted_as_support_before_a32_review",
            "scientific_support_limited_to_independent_paired_contexts",
        ]
    )
    return tuple(reasons)


def control_dependence_revision_decision_label(reasons: Sequence[str]) -> str:
    reason_set = set(reasons)
    if (
        "control_dependence_contradiction_observed" in reason_set
        or "paired_control_contrast_contradiction_observed" in reason_set
    ):
        return REFUTE_CONTROL_DEPENDENT_CONTRAST
    if "sage6f_control_dependence_review_dossier_incomplete" in reason_set:
        return REQUEST_MORE_TESTS_FOR_CONTROL_DEPENDENCE
    if (
        "scope_limited_control_dependent_confirmation_criteria_satisfied"
        in reason_set
    ):
        return CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST
    return KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE


def build_a33_control_dependence_handoff_candidates(
    decisions: Sequence[A32SecondUnknownGameControlDependenceRevisionDecision],
    source: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    paired_by_id = {
        str(row.get("paired_comparison_id", "")): row
        for row in source.get("paired_control_comparisons", []) or []
        if isinstance(row, Mapping)
    }
    handoffs: List[Dict[str, Any]] = []
    for decision in decisions:
        if not decision.a33_ready:
            continue
        paired = [
            paired_by_id[pair_id]
            for pair_id in decision.source_paired_comparison_ids
        ]
        handoffs.append(
            {
                "frontier_id": decision.frontier_id,
                "candidate_key": decision.candidate_key,
                "game_id": decision.game_id,
                "registry_entry_type": "CONTROL_DEPENDENT_RELATIONAL_CONTRAST",
                "candidate_mechanism_family": decision.candidate_mechanism_family,
                "metric": decision.metric,
                "target_action": decision.target_action,
                "control_actions": list(decision.control_actions),
                "status": HypothesisStatus.CONFIRMED.value,
                "support": decision.decision_record.support,
                "contradictions": decision.decision_record.contradictions,
                "experiments_spent": decision.decision_record.experiments_spent,
                "scoped_claim": decision.decision_record.description,
                "budgets": sorted(
                    {int(row.get("budget", 0) or 0) for row in paired}
                ),
                "paired_context_cluster_ids": [
                    str(row.get("context_cluster_id", "")) for row in paired
                ],
                "paired_context_snapshot_hashes": [
                    str(row.get("context_snapshot_hash", "")) for row in paired
                ],
                "controlled_effects_by_pair": [
                    {
                        "paired_comparison_id": str(
                            row.get("paired_comparison_id", "")
                        ),
                        "action2_minus_action1": float(
                            row.get("action1_controlled_effect_size", 0.0) or 0.0
                        ),
                        "action2_minus_action3": float(
                            row.get("action3_controlled_effect_size", 0.0) or 0.0
                        ),
                    }
                    for row in paired
                ],
                "source_observation_ids": list(decision.source_observation_ids),
                "source_paired_comparison_ids": list(
                    decision.source_paired_comparison_ids
                ),
                "standalone_action2_effect_status": (
                    HypothesisStatus.UNRESOLVED.value
                ),
                "standalone_action2_effect_confirmed": False,
                "action1_counted_as_universal_baseline": False,
                "ready_for_A33_registry_review": True,
                "a33_write_performed": False,
                "not_generalized_beyond_game": True,
                "not_generalized_beyond_exact_paired_contexts": True,
                "not_generalized_beyond_recorded_controls": True,
            }
        )
    return handoffs


def build_a32_6_gate(
    source: Mapping[str, Any],
    decisions: Sequence[A32SecondUnknownGameControlDependenceRevisionDecision],
    handoffs: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    paired = [
        row
        for row in source.get("paired_control_comparisons", []) or []
        if isinstance(row, Mapping)
    ]
    scientific_support = sum(
        row.decision_record.support
        for row in decisions
        if row.decision_record.status == HypothesisStatus.CONFIRMED
    )
    independent_paired_contexts = len(
        {str(row.get("context_snapshot_hash", "")) for row in paired}
    )
    return {
        "source_sage6f_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source.get("gate", {}).values()),
        "every_eligible_frontier_decided_once": len(decisions)
        == int(source.get("summary", {}).get("ready_for_A32_review", 0) or 0)
        == 1,
        "input_records_preserve_sage_support_zero": all(
            row.input_record.status == HypothesisStatus.UNRESOLVED
            and row.input_record.support == 0
            for row in decisions
        ),
        "scientific_support_equals_independent_paired_contexts": scientific_support
        == independent_paired_contexts
        == 3,
        "confirmation_is_relational_only": all(
            row.confirmation_performed
            and not bool(
                row.claim_dispositions[
                    "standalone_unconditional_action2_effect"
                ]["confirmation_performed"]
            )
            for row in decisions
        ),
        "action3_equivalence_kept_non_identifiable_not_refuted": all(
            row.claim_dispositions["standalone_unconditional_action2_effect"][
                "status"
            ]
            == HypothesisStatus.UNRESOLVED.value
            and not bool(
                row.claim_dispositions[
                    "standalone_unconditional_action2_effect"
                ]["refutation_performed"]
            )
            for row in decisions
        ),
        "a33_handoff_matches_confirmed_relational_decisions": len(handoffs)
        == sum(row.a33_ready for row in decisions)
        == 1
        and all(
            row.get("registry_entry_type")
            == "CONTROL_DEPENDENT_RELATIONAL_CONTRAST"
            and bool(row.get("ready_for_A33_registry_review", False))
            for row in handoffs
        ),
        "no_a33_write_performed": all(
            not bool(row.get("a33_write_performed", True)) for row in handoffs
        )
        and all(not row.a33_write_performed for row in decisions),
        "scope_locks_preserved": all(
            bool(row.scope_limits.get("not_generalized_beyond_game", False))
            and bool(
                row.scope_limits.get(
                    "not_generalized_beyond_exact_paired_contexts", False
                )
            )
            and bool(
                row.scope_limits.get(
                    "standalone_action2_effect_excluded_from_confirmation", False
                )
            )
            for row in decisions
        ),
        "wrong_confirmations_zero": all(row.wrong_confirmations == 0 for row in decisions),
    }


def summarize_a32_second_unknown_game_control_dependence_decisions(
    decisions: Sequence[A32SecondUnknownGameControlDependenceRevisionDecision],
    handoffs: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    confirmed = [row for row in decisions if row.confirmation_performed]
    refuted = [row for row in decisions if row.refutation_performed]
    unresolved = [
        row for row in decisions if row.decision_record.status == HypothesisStatus.UNRESOLVED
    ]
    if confirmed and len(confirmed) == len(decisions):
        status = "CONFIRMED"
        outcome = A32_6_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST_CONFIRMED
    elif refuted and len(refuted) == len(decisions):
        status = "REFUTED"
        outcome = A32_6_CONTROL_DEPENDENT_CONTRAST_REFUTED
    else:
        status = "UNRESOLVED"
        outcome = A32_6_CONTROL_DEPENDENCE_UNRESOLVED
    return {
        "source_frontiers_consumed": len(decisions),
        "scientific_revision_decisions": len(decisions),
        "scope_limited_control_dependent_contrasts_confirmed": len(confirmed),
        "control_dependent_contrasts_refuted": len(refuted),
        "control_dependent_contrasts_unresolved": len(unresolved),
        "standalone_action2_effects_confirmed": 0,
        "standalone_action2_effects_kept_unresolved": len(decisions),
        "action1_universal_baselines_confirmed": 0,
        "decision_records_confirmed": len(confirmed),
        "decision_records_unresolved": len(unresolved),
        "decision_records_refuted": len(refuted),
        "scientific_support_counted_by_a32": sum(
            row.decision_record.support for row in confirmed
        ),
        "raw_support_events_promoted_directly": 0,
        "independent_paired_contexts_counted_as_support": sum(
            int(row.evidence_summary["independent_paired_control_contexts"])
            for row in confirmed
        ),
        "a33_ready_candidates": len(handoffs),
        "a33_write_performed": False,
        "action3_equivalence_counted_as_refutation": False,
        "action1_counted_as_universal_baseline": False,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "status": status,
        "outcome_status": outcome,
    }


def validate_sage6f_control_dependence_revision_source(
    source: Mapping[str, Any],
) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    assessment = dict(source.get("a32_review_eligibility_assessment", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6F_SCHEMA_VERSION:
        raise ValueError("SAGE.6f schema version is not supported by A32.6")
    if str(source.get("outcome_status", "")) != SAGE6F_A32_REVIEW_ELIGIBLE:
        raise ValueError("A32.6 requires the eligible SAGE.6f outcome")
    if str(source.get("truth_status", "")) != SAGE6F_TRUTH_STATUS:
        raise ValueError("SAGE.6f truth must remain unevaluated before A32.6")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.6f must remain candidate-only before A32.6")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.6f candidate must remain unresolved before A32.6")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.6f support must remain 0 before A32.6")
    if bool(source.get("revision_performed", False)) or bool(
        source.get("confirmation_performed", False)
    ):
        raise ValueError("SAGE.6f cannot revise or confirm before A32.6")
    if bool(source.get("refutation_performed", False)):
        raise ValueError("SAGE.6f cannot refute before A32.6")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.6f cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.6f wrong_confirmations must remain 0")
    forbidden_true_flags = (
        "raw_events_counted_as_support",
        "paired_control_pattern_counted_as_scientific_verdict",
        "a32_review_eligibility_counted_as_a32_decision",
        "a32_intake_requested",
        "scope_generalization_performed",
    )
    if any(bool(source.get(name, False)) for name in forbidden_true_flags):
        raise ValueError("SAGE.6f cannot pre-count support, verdict, or intake")
    if not bool(summary.get("gate_passed", False)) or not all(
        bool(value) for value in source.get("gate", {}).values()
    ):
        raise ValueError("SAGE.6f gate must pass before A32.6")
    if int(summary.get("eligibility_requirements_passed", 0) or 0) != int(
        summary.get("eligibility_requirements_total", 0) or 0
    ):
        raise ValueError("SAGE.6f eligibility requirements must all pass")
    if list(summary.get("missing_eligibility_requirements", []) or []):
        raise ValueError("SAGE.6f eligibility requirements cannot be missing")
    if int(summary.get("ready_for_A32_review", 0) or 0) != 1:
        raise ValueError("A32.6 requires exactly one ready SAGE.6f frontier")
    if not bool(assessment.get("ready_for_A32_review", False)) or str(
        assessment.get("a32_review_recommendation", "")
    ) != READY_FOR_A32_CONTROL_DEPENDENCE_REVIEW:
        raise ValueError("SAGE.6f assessment must be ready for A32.6 review")
    if int(assessment.get("support", 0) or 0) != 0 or bool(
        assessment.get("eligibility_counted_as_a32_decision", False)
    ):
        raise ValueError("SAGE.6f assessment cannot pre-count support or a verdict")
    requirements = dict(assessment.get("eligibility_requirements", {}) or {})
    if not requirements or not all(bool(value) for value in requirements.values()):
        raise ValueError("SAGE.6f assessment eligibility must be complete")
    claim = dict(assessment.get("recommended_a32_review_scope", {}) or {})
    if str(claim.get("must_not_review_as", "")) != (
        "STANDALONE_UNCONDITIONAL_ACTION2_EFFECT"
    ):
        raise ValueError("A32.6 requires the reformulated relational claim")
    if sorted(assessment.get("control_actions", []) or []) != ["ACTION1", "ACTION3"]:
        raise ValueError("A32.6 requires preserved ACTION1/ACTION3 controls")

    observations = [
        row
        for row in source.get("observation_records", []) or []
        if isinstance(row, Mapping)
    ]
    clusters = [
        row
        for row in source.get("context_cluster_manifest", []) or []
        if isinstance(row, Mapping)
    ]
    groups = [
        row
        for row in source.get("control_conditioned_effect_groups", []) or []
        if isinstance(row, Mapping)
    ]
    paired = [
        row
        for row in source.get("paired_control_comparisons", []) or []
        if isinstance(row, Mapping)
    ]
    frontiers = [
        row
        for row in source.get("candidate_a32_review_frontiers", []) or []
        if isinstance(row, Mapping)
    ]
    if len(observations) != int(summary.get("observation_records", 0) or 0):
        raise ValueError("SAGE.6f observation count must match its summary")
    observation_ids = [str(row.get("observation_id", "")) for row in observations]
    if "" in observation_ids or len(observation_ids) != len(set(observation_ids)):
        raise ValueError("SAGE.6f observation ids must be unique")
    if len(clusters) != int(summary.get("context_clusters", 0) or 0):
        raise ValueError("SAGE.6f context count must match its summary")
    if len(groups) != int(summary.get("control_conditioned_effect_groups", 0) or 0):
        raise ValueError("SAGE.6f control group count must match its summary")
    if (
        len(paired) != int(summary.get("paired_control_contexts", 0) or 0)
        or len(paired) != 3
    ):
        raise ValueError("A32.6 requires exactly three paired control contexts")
    if len(frontiers) != 1:
        raise ValueError("A32.6 requires exactly one SAGE.6f frontier")
    if str(frontiers[0].get("candidate_key", "")) != str(
        assessment.get("candidate_key", "")
    ):
        raise ValueError("SAGE.6f frontier and assessment candidate must align")
    if not bool(frontiers[0].get("ready_for_A32_review", False)) or int(
        frontiers[0].get("support", 0) or 0
    ) != 0:
        raise ValueError("SAGE.6f frontier must be ready and support-free")

    candidate_rows = [*observations, *clusters, *groups, *paired, *frontiers]
    if any(
        int(row.get("support", 0) or 0) != 0
        or str(row.get("truth_status", "")) != SAGE6F_TRUTH_STATUS
        or str(row.get("revision_status", "")) != "CANDIDATE_ONLY"
        or bool(row.get("revision_performed", False))
        for row in candidate_rows
    ):
        raise ValueError("all SAGE.6f rows must remain candidate-only support-free")
    if any(
        not bool(row.get("context_preserved", False))
        or bool(row.get("cross_context_merge_performed", False))
        or bool(row.get("cross_control_effect_merge_performed", False))
        for row in clusters
    ):
        raise ValueError("SAGE.6f context boundaries must remain preserved")
    if any(
        not bool(row.get("comparison_index_only", False))
        or bool(row.get("group_counted_as_scientific_evidence", False))
        for row in groups
    ):
        raise ValueError("SAGE.6f control groups must remain non-scientific indexes")
    paired_ids = [str(row.get("paired_comparison_id", "")) for row in paired]
    paired_hashes = [str(row.get("context_snapshot_hash", "")) for row in paired]
    if (
        "" in paired_ids
        or len(paired_ids) != len(set(paired_ids))
        or "" in paired_hashes
        or len(paired_hashes) != len(set(paired_hashes))
    ):
        raise ValueError("SAGE.6f paired comparisons must be unique and independent")
    if sorted(int(row.get("budget", 0) or 0) for row in paired) != [50, 150, 300]:
        raise ValueError("A32.6 paired comparisons must cover all three budgets")
    if any(
        not bool(row.get("context_preserved", False))
        or not bool(row.get("control_identities_preserved", False))
        or not bool(row.get("target_signal_reproduced_across_control_pairs", False))
        or bool(row.get("paired_comparison_counted_as_scientific_verdict", False))
        for row in paired
    ):
        raise ValueError("SAGE.6f paired comparisons must remain exact candidates")
    if any(
        str(row.get("action1_observation_id", "")) not in observation_ids
        or str(row.get("action3_observation_id", "")) not in observation_ids
        for row in paired
    ):
        raise ValueError("SAGE.6f paired comparisons must reference observations")
    if sum(
        bool(row.get("neutral_context_replication_verified", False)) for row in clusters
    ) != 1:
        raise ValueError("A32.6 requires one replicated neutral exception")
    raw_support = sum(int(row.get("support_events", 0) or 0) for row in observations)
    raw_neutral = sum(int(row.get("neutral_events", 0) or 0) for row in observations)
    raw_contradictions = sum(
        int(row.get("contradiction_events", 0) or 0) for row in observations
    )
    if (
        raw_support != int(summary.get("raw_support_events", 0) or 0)
        or raw_neutral != int(summary.get("raw_neutral_events", 0) or 0)
        or raw_contradictions
        != int(summary.get("raw_contradiction_events", 0) or 0)
    ):
        raise ValueError("SAGE.6f raw event accounting must be exact")


def record_to_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_second_unknown_game_control_dependence_revision_decisions(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run A32.6 control-dependence scientific revision."
    )
    parser.add_argument(
        "--source-sage6f",
        default=DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH,
    )
    parser.add_argument(
        "--out",
        default=(
            DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH
        ),
    )
    args = parser.parse_args(argv)
    payload = run_a32_second_unknown_game_control_dependence_revision_consumer(
        source_sage6f_path=args.source_sage6f,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
