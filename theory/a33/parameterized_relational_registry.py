"""A33.4 registry for the A32.7 parameterized relational contrast."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.a32.third_unknown_game_parameterized_relation_revision_decisions import (
    A32_7_RELATION_CONFIRMED,
    A32_7_SCHEMA_VERSION,
    A32_7_TRUTH_STATUS,
    CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION,
    DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH,
    KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT,
)


DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH = (
    Path("diagnostics") / "a33" / "parameterized_relational_registry.json"
)

A33_4_SCHEMA_VERSION = "a33.parameterized_relational_registry.v1"
A33_4_TRUTH_STATUS = "NOT_REEVALUATED_BY_A33_4"
A33_4_ENTRY_ADDED = "A33_PARAMETERIZED_RELATIONAL_REGISTRY_ENTRY_ADDED"
A33_4_NO_ELIGIBLE_ENTRY = "A33_NO_ELIGIBLE_PARAMETERIZED_RELATIONAL_ENTRY"
CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST = (
    "CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST"
)


@dataclass(frozen=True)
class ParameterizedRelationalRegistryEntry:
    """One A32.7-confirmed relation locked to exact parameterized contexts."""

    key: str
    handoff_id: str
    game_id: str
    registry_entry_type: str
    candidate_type: str
    predicted_metric: str
    target_action: str
    target_action_args: Dict[str, Any]
    differentiating_control_variants: Tuple[Dict[str, Any], ...]
    equivalent_control_variants: Tuple[Dict[str, Any], ...]
    confirmed_support: int
    contradictions: int
    experiments_spent: int
    budgets: Tuple[int, ...]
    context_assessment_ids: Tuple[str, ...]
    context_snapshot_hashes: Tuple[str, ...]
    source_comparison_group_ids: Tuple[str, ...]
    context_manifest: Tuple[Dict[str, Any], ...]
    scoped_claim: str
    autonomous_target_effect_status: str
    source_artifact: str
    raw_comparison_events_provenance: int
    technical_replication_events_provenance: int
    known_scope: str = "game_metric_target_parameterized_controls_exact_contexts"
    source_decision: str = (
        CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
    )
    status: str = "confirmed"
    evidence_notes: Tuple[str, ...] = field(default_factory=tuple)
    scope_game_locked: bool = True
    scope_metric_locked: bool = True
    scope_target_parameter_locked: bool = True
    scope_control_parameters_locked: bool = True
    scope_contexts_locked: bool = True
    scope_budgets_locked: bool = True
    not_generalized_beyond_game: bool = True
    not_generalized_beyond_metric: bool = True
    not_generalized_beyond_parameter_variants: bool = True
    not_generalized_beyond_exact_contexts: bool = True
    autonomous_target_effect_excluded: bool = True
    raw_events_excluded_from_direct_support: bool = True
    technical_repetitions_excluded_from_support: bool = True
    a32_confirmation_reused_without_reevaluation: bool = True
    a33_confirmation_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "handoff_id": self.handoff_id,
            "game_id": self.game_id,
            "registry_entry_type": self.registry_entry_type,
            "candidate_type": self.candidate_type,
            "predicted_metric": self.predicted_metric,
            "target_action": self.target_action,
            "target_action_args": dict(self.target_action_args),
            "differentiating_control_variants": [
                dict(row) for row in self.differentiating_control_variants
            ],
            "equivalent_control_variants": [
                dict(row) for row in self.equivalent_control_variants
            ],
            "confirmed_support": int(self.confirmed_support),
            "contradictions": int(self.contradictions),
            "experiments_spent": int(self.experiments_spent),
            "budgets": list(self.budgets),
            "context_assessment_ids": list(self.context_assessment_ids),
            "context_snapshot_hashes": list(self.context_snapshot_hashes),
            "source_comparison_group_ids": list(self.source_comparison_group_ids),
            "context_manifest": [dict(row) for row in self.context_manifest],
            "scoped_claim": self.scoped_claim,
            "autonomous_target_effect_status": (self.autonomous_target_effect_status),
            "source_artifact": self.source_artifact,
            "raw_comparison_events_provenance": int(
                self.raw_comparison_events_provenance
            ),
            "technical_replication_events_provenance": int(
                self.technical_replication_events_provenance
            ),
            "known_scope": self.known_scope,
            "source_decision": self.source_decision,
            "status": self.status,
            "evidence_notes": list(self.evidence_notes),
            "scope_game_locked": self.scope_game_locked,
            "scope_metric_locked": self.scope_metric_locked,
            "scope_target_parameter_locked": self.scope_target_parameter_locked,
            "scope_control_parameters_locked": self.scope_control_parameters_locked,
            "scope_contexts_locked": self.scope_contexts_locked,
            "scope_budgets_locked": self.scope_budgets_locked,
            "not_generalized_beyond_game": self.not_generalized_beyond_game,
            "not_generalized_beyond_metric": self.not_generalized_beyond_metric,
            "not_generalized_beyond_parameter_variants": (
                self.not_generalized_beyond_parameter_variants
            ),
            "not_generalized_beyond_exact_contexts": (
                self.not_generalized_beyond_exact_contexts
            ),
            "autonomous_target_effect_excluded": (
                self.autonomous_target_effect_excluded
            ),
            "raw_events_excluded_from_direct_support": (
                self.raw_events_excluded_from_direct_support
            ),
            "technical_repetitions_excluded_from_support": (
                self.technical_repetitions_excluded_from_support
            ),
            "a32_confirmation_reused_without_reevaluation": (
                self.a32_confirmation_reused_without_reevaluation
            ),
            "a33_confirmation_performed": self.a33_confirmation_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_parameterized_relational_registry_generation(
    *,
    decisions_path: str | Path = (
        DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH
    ),
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Validate A32.7 and register its confirmed relation without a new verdict."""
    source = _load_json(decisions_path)
    entries = build_parameterized_relational_registry(
        source,
        source_artifact=str(decisions_path),
    )
    excluded = build_excluded_parameterized_claim_audit(source)
    gate = build_a33_4_gate(source, entries, excluded)
    if not gate or not all(gate.values()):
        raise ValueError("A33.4 parameterized relational registry gate did not pass")
    summary = summarize_parameterized_relational_registry(entries, excluded, gate)
    payload = {
        "config": {
            "decisions_path": str(decisions_path),
            "schema_version": A33_4_SCHEMA_VERSION,
            "inputs_read": ["A32.7"],
            "registry_scope": "A33_PARAMETERIZED_RELATIONAL_CONFIRMED_MEMORY",
            "artifacts_not_modified": [
                "A32.7",
                "A33.1",
                "A33.2",
                "A33.3",
                "M2",
                "M3",
                "A34",
            ],
            "registration_policy": {
                "a32_parameterized_relation_confirmation_required": True,
                "a32_confirmed_record_required": True,
                "zero_contradictions_required": True,
                "one_support_per_independent_exact_context_required": True,
                "exact_game_metric_target_controls_context_scope_required": True,
                "autonomous_target_effect_must_remain_unresolved": True,
                "raw_events_are_provenance_not_direct_support": True,
                "technical_repetitions_are_provenance_not_support": True,
                "a33_does_not_reevaluate_truth": True,
                "existing_a33_registries_not_mutated": True,
            },
        },
        "source_a32_7_summary": dict(source.get("summary", {}) or {}),
        "parameterized_relational_contrasts": [entry.to_dict() for entry in entries],
        "excluded_claim_audit": excluded,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "REGISTERED" if entries else "NO_ELIGIBLE_ENTRY",
        "truth_status": A33_4_TRUTH_STATUS,
        "source_stage": "A32.7",
        "registry_stage": "A33.4",
        "registry_validation_performed": True,
        "registration_performed": bool(entries),
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "a33_write_performed": True,
        "source_a32_7_mutated": False,
        "legacy_a33_1_registry_mutated": False,
        "scoped_a33_2_registry_mutated": False,
        "relational_a33_3_registry_mutated": False,
        "support": sum(entry.confirmed_support for entry in entries),
        "support_origin": "A32.7_DECISION_RECORDS_ONLY",
        "raw_comparison_events_imported_as_support": 0,
        "technical_replication_events_imported_as_support": 0,
        "autonomous_target_effect_registered": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_parameterized_relational_registry(payload, output_path)
    return payload


def build_parameterized_relational_registry(
    decisions_payload: Mapping[str, Any],
    *,
    source_artifact: str = "",
) -> Tuple[ParameterizedRelationalRegistryEntry, ...]:
    """Extract the complete A32.7 handoff without reevaluating its truth."""
    validate_a32_7_parameterized_relational_registry_source(decisions_payload)
    decisions_by_handoff = {
        str(row.get("handoff_id", "")): row
        for row in decisions_payload.get("revision_decisions", []) or []
        if isinstance(row, Mapping)
    }
    entries: List[ParameterizedRelationalRegistryEntry] = []
    for handoff in decisions_payload.get("a33_handoff_candidates", []) or []:
        handoff_id = str(handoff.get("handoff_id", ""))
        decision = decisions_by_handoff[handoff_id]
        evidence = dict(decision.get("evidence_summary", {}) or {})
        entry = ParameterizedRelationalRegistryEntry(
            key=str(handoff.get("candidate_key", "")),
            handoff_id=handoff_id,
            game_id=str(handoff.get("game_id", "")),
            registry_entry_type=str(handoff.get("registry_entry_type", "")),
            candidate_type=str(handoff.get("candidate_type", "")),
            predicted_metric=str(handoff.get("metric", "")),
            target_action=str(handoff.get("target_action", "")),
            target_action_args=dict(handoff.get("target_action_args", {}) or {}),
            differentiating_control_variants=tuple(
                dict(row)
                for row in handoff.get("differentiating_control_variants", []) or []
            ),
            equivalent_control_variants=tuple(
                dict(row)
                for row in handoff.get("equivalent_control_variants", []) or []
            ),
            confirmed_support=int(handoff.get("support", 0) or 0),
            contradictions=int(handoff.get("contradictions", 0) or 0),
            experiments_spent=int(handoff.get("experiments_spent", 0) or 0),
            budgets=tuple(
                sorted(
                    {
                        int(budget)
                        for context in handoff.get("context_manifest", []) or []
                        for budget in context.get("budgets_observed", []) or []
                    }
                )
            ),
            context_assessment_ids=tuple(
                str(value)
                for value in handoff.get("source_context_assessment_ids", []) or []
            ),
            context_snapshot_hashes=tuple(
                str(value)
                for value in handoff.get("source_context_snapshot_hashes", []) or []
            ),
            source_comparison_group_ids=tuple(
                str(value)
                for value in handoff.get("source_comparison_group_ids", []) or []
            ),
            context_manifest=tuple(
                dict(row)
                for row in handoff.get("context_manifest", []) or []
                if isinstance(row, Mapping)
            ),
            scoped_claim=str(handoff.get("scoped_claim", "")),
            autonomous_target_effect_status=str(
                handoff.get("autonomous_target_effect_status", "")
            ),
            source_artifact=source_artifact,
            raw_comparison_events_provenance=int(
                evidence.get("raw_comparison_events", 0) or 0
            ),
            technical_replication_events_provenance=int(
                evidence.get("technical_replication_events", 0) or 0
            ),
            evidence_notes=(
                "a32_7_relational_confirmation_reused_without_reevaluation",
                "eight_independent_exact_parameterized_contexts",
                "target_minus_differentiating_control_positive",
                "target_minus_equivalent_control_neutral",
                "autonomous_target_effect_remains_unresolved",
                "raw_events_and_technical_repetitions_are_provenance_only",
                "scope_locked_to_game_metric_parameters_and_contexts",
            ),
        )
        decision_record = dict(decision.get("decision_record", {}) or {})
        if str(decision_record.get("key", "")) != entry.key:
            raise ValueError("A32.7 decision record and A33.4 handoff key must align")
        entries.append(entry)
    return tuple(entries)


def build_excluded_parameterized_claim_audit(
    decisions_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Expose the autonomous target effect deliberately omitted from memory."""
    excluded: List[Dict[str, Any]] = []
    for decision in decisions_payload.get("revision_decisions", []) or []:
        dispositions = dict(decision.get("claim_dispositions", {}) or {})
        disposition = dict(
            dispositions.get("autonomous_parameterized_target_effect", {}) or {}
        )
        excluded.append(
            {
                "candidate_key": str(decision.get("candidate_key", "")),
                "game_id": str(decision.get("game_id", "")),
                "target_action": str(decision.get("target_action", "")),
                "target_action_args": dict(
                    decision.get("target_action_args", {}) or {}
                ),
                "claim": "AUTONOMOUS_PARAMETERIZED_TARGET_EFFECT",
                "claim_status": str(disposition.get("status", "")),
                "claim_decision": str(disposition.get("decision", "")),
                "claim_support": int(disposition.get("support", 0) or 0),
                "exclusion_reason": (
                    "AUTONOMOUS_TARGET_EFFECT_NON_IDENTIFIABLE_NOT_REGISTRY_ELIGIBLE"
                ),
                "registered": False,
                "counted_as_refutation": False,
            }
        )
    return excluded


def build_a33_4_gate(
    source: Mapping[str, Any],
    entries: Sequence[ParameterizedRelationalRegistryEntry],
    excluded: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    return {
        "source_a32_7_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source.get("gate", {}).values()),
        "every_confirmed_parameterized_handoff_registered_once": len(entries)
        == int(source.get("summary", {}).get("a33_ready_candidates", 0) or 0)
        == 1,
        "support_imported_from_a32_decision_only": sum(
            entry.confirmed_support for entry in entries
        )
        == int(source.get("support", 0) or 0)
        == 8,
        "independent_context_support_preserved": all(
            entry.confirmed_support == len(entry.context_snapshot_hashes) == 8
            and len(set(entry.context_snapshot_hashes)) == 8
            for entry in entries
        ),
        "parameterized_relation_scope_fully_locked": all(
            entry.scope_game_locked
            and entry.scope_metric_locked
            and entry.scope_target_parameter_locked
            and entry.scope_control_parameters_locked
            and entry.scope_contexts_locked
            and entry.scope_budgets_locked
            for entry in entries
        ),
        "autonomous_target_effect_excluded": len(excluded) == 1
        and all(
            str(row.get("claim_status", "")) == "unresolved"
            and int(row.get("claim_support", 0) or 0) == 0
            and not bool(row.get("registered", True))
            and not bool(row.get("counted_as_refutation", True))
            for row in excluded
        ),
        "raw_and_technical_events_not_recounted": all(
            entry.raw_events_excluded_from_direct_support
            and entry.technical_repetitions_excluded_from_support
            for entry in entries
        ),
        "no_a33_scientific_verdict_created": all(
            entry.a32_confirmation_reused_without_reevaluation
            and not entry.a33_confirmation_performed
            for entry in entries
        ),
        "existing_a33_registries_quarantined": True,
        "wrong_confirmations_zero": all(
            entry.wrong_confirmations == 0 for entry in entries
        ),
    }


def summarize_parameterized_relational_registry(
    entries: Sequence[ParameterizedRelationalRegistryEntry],
    excluded: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    return {
        "source_candidates_consumed": len(entries),
        "a32_7_handoff_candidates_consumed": len(entries),
        "parameterized_relational_contrasts_registered": len(entries),
        "autonomous_target_effects_excluded": len(excluded),
        "confirmed_support_imported_from_a32_7": sum(
            entry.confirmed_support for entry in entries
        ),
        "raw_comparison_events_imported_as_support": 0,
        "technical_replication_events_imported_as_support": 0,
        "raw_comparison_events_preserved_as_provenance": sum(
            entry.raw_comparison_events_provenance for entry in entries
        ),
        "technical_replications_preserved_as_provenance": sum(
            entry.technical_replication_events_provenance for entry in entries
        ),
        "experiments_spent_total": sum(entry.experiments_spent for entry in entries),
        "registered_exact_contexts": sum(
            len(entry.context_snapshot_hashes) for entry in entries
        ),
        "registered_parameterized_control_variants": sum(
            len(entry.differentiating_control_variants)
            + len(entry.equivalent_control_variants)
            for entry in entries
        ),
        "registered_budgets": sum(len(entry.budgets) for entry in entries),
        "scope_locked_entries": sum(
            entry.scope_game_locked
            and entry.scope_metric_locked
            and entry.scope_target_parameter_locked
            and entry.scope_control_parameters_locked
            and entry.scope_contexts_locked
            and entry.scope_budgets_locked
            for entry in entries
        ),
        "a33_truth_reevaluations": 0,
        "a33_confirmations_performed": 0,
        "legacy_a33_1_registry_mutated": False,
        "scoped_a33_2_registry_mutated": False,
        "relational_a33_3_registry_mutated": False,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "outcome_status": A33_4_ENTRY_ADDED if entries else A33_4_NO_ELIGIBLE_ENTRY,
    }


def validate_a32_7_parameterized_relational_registry_source(
    source: Mapping[str, Any],
) -> None:
    """Reject an A32.7 payload that would broaden or invent registry truth."""
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != A32_7_SCHEMA_VERSION:
        raise ValueError("A32.7 schema version is not supported by A33.4")
    if str(source.get("outcome_status", "")) != A32_7_RELATION_CONFIRMED:
        raise ValueError("A33.4 requires the confirmed A32.7 relation outcome")
    if str(source.get("truth_status", "")) != A32_7_TRUTH_STATUS:
        raise ValueError("A33.4 requires the exact A32.7 truth status")
    if (
        str(source.get("status", "")) != "CONFIRMED"
        or not bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", True))
        or not bool(source.get("a33_ready", False))
    ):
        raise ValueError("A33.4 requires one confirmed A32.7 handoff")
    if bool(source.get("a33_write_performed", True)):
        raise ValueError("A32.7 cannot pre-write the A33.4 registry")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("A32.7 wrong_confirmations must remain 0")
    if int(source.get("support", 0) or 0) != 8:
        raise ValueError("A32.7 support must equal eight independent contexts")
    if (
        bool(source.get("autonomous_target_effect_confirmed", True))
        or bool(source.get("autonomous_target_effect_refuted", True))
        or str(source.get("autonomous_target_effect_status", "")) != "unresolved"
    ):
        raise ValueError("the autonomous target effect must remain unresolved")
    if (
        bool(source.get("equivalent_control_counted_as_refutation", True))
        or bool(source.get("technical_repetitions_counted_as_support", True))
        or bool(
            source.get(
                "sage_candidate_events_counted_as_support_before_a32_review", True
            )
        )
        or bool(source.get("scope_limited_confirmation_generalized_beyond_game", True))
    ):
        raise ValueError("A32.7 exclusions and scope limits must remain intact")
    if not bool(summary.get("gate_passed", False)) or not all(
        bool(value) for value in source.get("gate", {}).values()
    ):
        raise ValueError("every A32.7 source gate must pass")
    decisions = [
        row
        for row in source.get("revision_decisions", []) or []
        if isinstance(row, Mapping)
    ]
    handoffs = [
        row
        for row in source.get("a33_handoff_candidates", []) or []
        if isinstance(row, Mapping)
    ]
    if len(decisions) != len(handoffs) or len(decisions) != 1:
        raise ValueError("A33.4 requires one aligned A32.7 decision and handoff")
    decision = decisions[0]
    handoff = handoffs[0]
    decision_record = dict(decision.get("decision_record", {}) or {})
    claims = dict(decision.get("claim_dispositions", {}) or {})
    relation_claim = dict(
        claims.get("control_dependent_parameterized_relation", {}) or {}
    )
    autonomous_claim = dict(
        claims.get("autonomous_parameterized_target_effect", {}) or {}
    )
    if (
        str(decision.get("decision", ""))
        != CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
        or not bool(decision.get("confirmation_performed", False))
        or not bool(decision.get("a33_ready", False))
        or str(decision_record.get("status", "")) != "confirmed"
        or int(decision_record.get("support", 0) or 0) != 8
        or int(decision_record.get("contradictions", 0) or 0) != 0
        or int(decision_record.get("experiments_spent", 0) or 0) != 8
        or str(relation_claim.get("status", "")) != "confirmed"
        or int(relation_claim.get("support", 0) or 0) != 8
    ):
        raise ValueError("A32.7 relational decision must remain confirmed and exact")
    if (
        str(autonomous_claim.get("status", "")) != "unresolved"
        or str(autonomous_claim.get("decision", ""))
        != KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT
        or int(autonomous_claim.get("support", 0) or 0) != 0
        or bool(autonomous_claim.get("confirmation_performed", True))
        or bool(autonomous_claim.get("refutation_performed", True))
    ):
        raise ValueError("A32.7 autonomous claim must remain excluded and unresolved")
    if (
        str(handoff.get("handoff_id", "")) != str(decision.get("handoff_id", ""))
        or str(handoff.get("candidate_key", "")) != str(decision_record.get("key", ""))
        or str(handoff.get("registry_entry_type", ""))
        != CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST
        or str(handoff.get("status", "")) != "confirmed"
        or int(handoff.get("support", 0) or 0) != 8
        or int(handoff.get("contradictions", 0) or 0) != 0
        or not bool(handoff.get("ready_for_A33_4_registry_review", False))
        or bool(handoff.get("a33_write_performed", True))
    ):
        raise ValueError("A32.7 handoff must remain exact and A33.4-ready")
    expected_identity = {
        "game_id": "tn36-ab4f63cc",
        "metric": "local_patch_before_after",
        "target_action": "ACTION6",
        "target_action_args": {"x": 25, "y": 42},
        "differentiating_control_variants": [
            {"action": "ACTION6", "action_args": {"x": 34, "y": 51}}
        ],
        "equivalent_control_variants": [
            {"action": "ACTION6", "action_args": {"x": 41, "y": 44}}
        ],
    }
    for field_name, expected in expected_identity.items():
        if handoff.get(field_name) != expected:
            raise ValueError("A32.7 exact game, metric, and parameter scope changed")
    contexts = [
        row
        for row in handoff.get("context_manifest", []) or []
        if isinstance(row, Mapping)
    ]
    hashes = [str(row.get("context_snapshot_hash", "")) for row in contexts]
    assessment_ids = [str(row.get("context_assessment_id", "")) for row in contexts]
    if len(contexts) != 8 or len(set(hashes)) != 8 or "" in hashes:
        raise ValueError("A32.7 must preserve eight exact unique contexts")
    if sorted(hashes) != sorted(
        str(value) for value in handoff.get("source_context_snapshot_hashes", []) or []
    ) or assessment_ids != list(handoff.get("source_context_assessment_ids", []) or []):
        raise ValueError("A32.7 context identities must remain aligned")
    if any(
        int(row.get("support", 0) or 0) != 0
        or len(row.get("parameterized_controls", []) or []) != 2
        or not bool(row.get("all_repetitions_consistent", False))
        for row in contexts
    ):
        raise ValueError("A32.7 context evidence must remain exact and support-free")
    evidence = dict(decision.get("evidence_summary", {}) or {})
    if (
        evidence.get("differentiating_control_effect_sizes") != [2.0] * 8
        or evidence.get("equivalent_control_effect_sizes") != [0.0] * 8
        or int(evidence.get("raw_comparison_events", 0) or 0) != 26
        or int(evidence.get("technical_replication_events", 0) or 0) != 10
        or int(evidence.get("scientific_support_after_a32_review", 0) or 0) != 8
        or int(evidence.get("raw_comparison_events_promoted_directly", 0) or 0) != 0
        or int(evidence.get("technical_replication_events_counted_as_support", 0) or 0)
        != 0
    ):
        raise ValueError("A32.7 evidence and provenance counts must remain exact")
    scope_fields = (
        "not_generalized_beyond_game",
        "not_generalized_beyond_exact_contexts",
        "not_generalized_beyond_metric",
        "not_generalized_beyond_parameter_variants",
    )
    if any(not bool(handoff.get(field_name, False)) for field_name in scope_fields):
        raise ValueError("A32.7 parameterized relational scope must remain locked")
    if (
        bool(handoff.get("technical_repetitions_counted_as_support", True))
        or bool(handoff.get("parameterized_controls_counted_as_distinct_actions", True))
        or bool(handoff.get("autonomous_target_effect_confirmed", True))
        or str(handoff.get("autonomous_target_effect_status", "")) != "unresolved"
    ):
        raise ValueError("A32.7 excluded claims cannot enter the A33.4 registry")


def write_parameterized_relational_registry(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        default=str(
            DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH
        ),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH),
    )
    args = parser.parse_args(argv)
    payload = run_parameterized_relational_registry_generation(
        decisions_path=args.decisions,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
