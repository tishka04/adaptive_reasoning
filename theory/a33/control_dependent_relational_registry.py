"""A33.3 registry for A32.6 control-dependent relational contrasts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.a32.second_unknown_game_control_dependence_revision_decisions import (
    A32_6_SCHEMA_VERSION,
    A32_6_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST_CONFIRMED,
    A32_6_TRUTH_STATUS,
    CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST,
    DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH,
    KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE,
)


DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH = (
    Path("diagnostics")
    / "a33"
    / "control_dependent_relational_registry.json"
)

A33_3_SCHEMA_VERSION = "a33.control_dependent_relational_registry.v1"
A33_3_TRUTH_STATUS = "NOT_REEVALUATED_BY_A33_3"
A33_3_ENTRY_ADDED = "A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_ENTRY_ADDED"
A33_3_NO_ELIGIBLE_ENTRY = "A33_NO_ELIGIBLE_CONTROL_DEPENDENT_RELATIONAL_ENTRY"
CONTROL_DEPENDENT_RELATIONAL_CONTRAST = "CONTROL_DEPENDENT_RELATIONAL_CONTRAST"


@dataclass(frozen=True)
class ControlDependentRelationalRegistryEntry:
    """One A32-confirmed relation kept inside its exact paired-control scope."""

    key: str
    frontier_id: str
    game_id: str
    registry_entry_type: str
    mechanic_family: str
    predicted_metric: str
    target_action: str
    control_actions: Tuple[str, ...]
    confirmed_support: int
    contradictions: int
    experiments_spent: int
    budgets: Tuple[int, ...]
    paired_context_cluster_ids: Tuple[str, ...]
    paired_context_snapshot_hashes: Tuple[str, ...]
    controlled_effects_by_pair: Tuple[Dict[str, Any], ...]
    source_observation_ids: Tuple[str, ...]
    source_paired_comparison_ids: Tuple[str, ...]
    scoped_claim: str
    standalone_action2_effect_status: str
    source_artifact: str
    known_scope: str = "game_exact_paired_contexts_target_controls_metric"
    source_decision: str = CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST
    status: str = "confirmed"
    evidence_notes: Tuple[str, ...] = field(default_factory=tuple)
    scope_game_locked: bool = True
    scope_contexts_locked: bool = True
    scope_target_action_locked: bool = True
    scope_control_actions_locked: bool = True
    scope_metric_locked: bool = True
    scope_budgets_locked: bool = True
    not_generalized_beyond_game: bool = True
    not_generalized_beyond_exact_paired_contexts: bool = True
    not_generalized_beyond_recorded_controls: bool = True
    standalone_action2_effect_excluded: bool = True
    action1_universal_baseline_excluded: bool = True
    a32_confirmation_reused_without_reevaluation: bool = True
    a33_confirmation_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "frontier_id": self.frontier_id,
            "game_id": self.game_id,
            "registry_entry_type": self.registry_entry_type,
            "mechanic_family": self.mechanic_family,
            "predicted_metric": self.predicted_metric,
            "target_action": self.target_action,
            "control_actions": list(self.control_actions),
            "confirmed_support": int(self.confirmed_support),
            "contradictions": int(self.contradictions),
            "experiments_spent": int(self.experiments_spent),
            "budgets": list(self.budgets),
            "paired_context_cluster_ids": list(self.paired_context_cluster_ids),
            "paired_context_snapshot_hashes": list(
                self.paired_context_snapshot_hashes
            ),
            "controlled_effects_by_pair": [
                dict(row) for row in self.controlled_effects_by_pair
            ],
            "source_observation_ids": list(self.source_observation_ids),
            "source_paired_comparison_ids": list(
                self.source_paired_comparison_ids
            ),
            "scoped_claim": self.scoped_claim,
            "standalone_action2_effect_status": (
                self.standalone_action2_effect_status
            ),
            "source_artifact": self.source_artifact,
            "known_scope": self.known_scope,
            "source_decision": self.source_decision,
            "status": self.status,
            "evidence_notes": list(self.evidence_notes),
            "scope_game_locked": self.scope_game_locked,
            "scope_contexts_locked": self.scope_contexts_locked,
            "scope_target_action_locked": self.scope_target_action_locked,
            "scope_control_actions_locked": self.scope_control_actions_locked,
            "scope_metric_locked": self.scope_metric_locked,
            "scope_budgets_locked": self.scope_budgets_locked,
            "not_generalized_beyond_game": self.not_generalized_beyond_game,
            "not_generalized_beyond_exact_paired_contexts": (
                self.not_generalized_beyond_exact_paired_contexts
            ),
            "not_generalized_beyond_recorded_controls": (
                self.not_generalized_beyond_recorded_controls
            ),
            "standalone_action2_effect_excluded": (
                self.standalone_action2_effect_excluded
            ),
            "action1_universal_baseline_excluded": (
                self.action1_universal_baseline_excluded
            ),
            "a32_confirmation_reused_without_reevaluation": (
                self.a32_confirmation_reused_without_reevaluation
            ),
            "a33_confirmation_performed": self.a33_confirmation_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_control_dependent_relational_registry_generation(
    *,
    decisions_path: str | Path = (
        DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH
    ),
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Validate A32.6 and register only its relational confirmed handoff."""
    source = _load_json(decisions_path)
    entries = build_control_dependent_relational_registry(
        source,
        source_artifact=str(decisions_path),
    )
    excluded = build_excluded_relational_claim_audit(source)
    gate = build_a33_3_gate(source, entries, excluded)
    if not gate or not all(gate.values()):
        raise ValueError("A33.3 relational registry gate did not pass")
    summary = summarize_control_dependent_relational_registry(
        entries, excluded, gate
    )
    payload = {
        "config": {
            "decisions_path": str(decisions_path),
            "schema_version": A33_3_SCHEMA_VERSION,
            "inputs_read": ["A32.6"],
            "registry_scope": "A33_CONTROL_DEPENDENT_RELATIONAL_CONFIRMED_MEMORY",
            "artifacts_not_modified": [
                "A32.6",
                "A33.1",
                "A33.2",
                "M2",
                "M3",
                "A34",
                "A35",
                "A36",
                "A37",
                "A38",
                "A39",
            ],
            "registration_policy": {
                "a32_relational_confirmation_required": True,
                "a32_confirmed_record_required": True,
                "zero_contradictions_required": True,
                "one_support_per_independent_paired_context_required": True,
                "exact_game_context_target_control_metric_scope_required": True,
                "standalone_action2_effect_must_remain_unresolved": True,
                "action1_universal_baseline_must_remain_unresolved": True,
                "a33_does_not_reevaluate_truth": True,
                "legacy_a33_1_registry_not_mutated": True,
                "scoped_a33_2_registry_not_mutated": True,
            },
        },
        "source_a32_6_summary": dict(source.get("summary", {}) or {}),
        "control_dependent_relational_contrasts": [
            entry.to_dict() for entry in entries
        ],
        "excluded_claim_audit": excluded,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "REGISTERED" if entries else "NO_ELIGIBLE_ENTRY",
        "truth_status": A33_3_TRUTH_STATUS,
        "source_stage": "A32.6",
        "registry_stage": "A33.3",
        "registry_validation_performed": True,
        "registration_performed": bool(entries),
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "a33_write_performed": True,
        "source_a32_6_mutated": False,
        "legacy_a33_1_registry_mutated": False,
        "scoped_a33_2_registry_mutated": False,
        "support": sum(entry.confirmed_support for entry in entries),
        "support_origin": "A32.6_DECISION_RECORDS_ONLY",
        "raw_sage_support_events_imported_directly": 0,
        "standalone_action2_effect_registered": False,
        "action1_universal_baseline_registered": False,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_control_dependent_relational_registry(payload, output_path)
    return payload


def build_control_dependent_relational_registry(
    decisions_payload: Mapping[str, Any],
    *,
    source_artifact: str = "",
) -> Tuple[ControlDependentRelationalRegistryEntry, ...]:
    """Extract only complete A32.6 relational handoffs without reevaluation."""
    validate_a32_6_relational_registry_source(decisions_payload)
    decisions_by_frontier = {
        str(row.get("frontier_id", "")): row
        for row in decisions_payload.get("revision_decisions", []) or []
        if isinstance(row, Mapping)
    }
    entries: List[ControlDependentRelationalRegistryEntry] = []
    for handoff in decisions_payload.get("a33_handoff_candidates", []) or []:
        frontier_id = str(handoff.get("frontier_id", ""))
        decision = decisions_by_frontier[frontier_id]
        decision_record = dict(decision.get("decision_record", {}) or {})
        entries.append(
            ControlDependentRelationalRegistryEntry(
                key=str(handoff.get("candidate_key", "")),
                frontier_id=frontier_id,
                game_id=str(handoff.get("game_id", "")),
                registry_entry_type=str(handoff.get("registry_entry_type", "")),
                mechanic_family=str(
                    handoff.get("candidate_mechanism_family", "")
                ),
                predicted_metric=str(handoff.get("metric", "")),
                target_action=str(handoff.get("target_action", "")),
                control_actions=tuple(
                    str(value) for value in handoff.get("control_actions", []) or []
                ),
                confirmed_support=int(handoff.get("support", 0) or 0),
                contradictions=int(handoff.get("contradictions", 0) or 0),
                experiments_spent=int(handoff.get("experiments_spent", 0) or 0),
                budgets=tuple(
                    int(value) for value in handoff.get("budgets", []) or []
                ),
                paired_context_cluster_ids=tuple(
                    str(value)
                    for value in handoff.get("paired_context_cluster_ids", []) or []
                ),
                paired_context_snapshot_hashes=tuple(
                    str(value)
                    for value in handoff.get("paired_context_snapshot_hashes", [])
                    or []
                ),
                controlled_effects_by_pair=tuple(
                    dict(row)
                    for row in handoff.get("controlled_effects_by_pair", []) or []
                    if isinstance(row, Mapping)
                ),
                source_observation_ids=tuple(
                    str(value)
                    for value in handoff.get("source_observation_ids", []) or []
                ),
                source_paired_comparison_ids=tuple(
                    str(value)
                    for value in handoff.get("source_paired_comparison_ids", []) or []
                ),
                scoped_claim=str(handoff.get("scoped_claim", "")),
                standalone_action2_effect_status=str(
                    handoff.get("standalone_action2_effect_status", "")
                ),
                source_artifact=source_artifact,
                evidence_notes=(
                    "a32_6_relational_confirmation_reused_without_reevaluation",
                    "three_independent_exact_paired_control_contexts",
                    "action2_minus_action1_positive_action2_minus_action3_neutral",
                    "standalone_action2_effect_remains_unresolved",
                    "action1_not_registered_as_universal_baseline",
                    "scope_locked_to_game_contexts_target_controls_metric_and_budgets",
                ),
            )
        )
        if str(decision_record.get("key", "")) != entries[-1].key:
            raise ValueError("A32.6 decision record and A33.3 handoff key must align")
    return tuple(entries)


def build_excluded_relational_claim_audit(
    decisions_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Expose the two A32.6 claims deliberately omitted from confirmed memory."""
    decisions = [
        row
        for row in decisions_payload.get("revision_decisions", []) or []
        if isinstance(row, Mapping)
    ]
    excluded: List[Dict[str, Any]] = []
    for decision in decisions:
        dispositions = dict(decision.get("claim_dispositions", {}) or {})
        for disposition_key, exclusion_reason in (
            (
                "standalone_unconditional_action2_effect",
                "STANDALONE_ACTION2_EFFECT_NON_IDENTIFIABLE_NOT_REGISTRY_ELIGIBLE",
            ),
            (
                "action1_universal_baseline",
                "ACTION1_UNIVERSAL_BASELINE_UNRESOLVED_NOT_REGISTRY_ELIGIBLE",
            ),
        ):
            disposition = dict(dispositions.get(disposition_key, {}) or {})
            excluded.append(
                {
                    "candidate_key": str(decision.get("candidate_key", "")),
                    "game_id": str(decision.get("game_id", "")),
                    "target_action": str(decision.get("target_action", "")),
                    "claim": str(disposition.get("claim", "")),
                    "claim_status": str(disposition.get("status", "")),
                    "claim_support": int(disposition.get("support", 0) or 0),
                    "exclusion_reason": exclusion_reason,
                    "registered": False,
                    "counted_as_refutation": False,
                }
            )
    return excluded


def summarize_control_dependent_relational_registry(
    entries: Sequence[ControlDependentRelationalRegistryEntry],
    excluded: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    scope_locked = sum(
        entry.scope_game_locked
        and entry.scope_contexts_locked
        and entry.scope_target_action_locked
        and entry.scope_control_actions_locked
        and entry.scope_metric_locked
        and entry.scope_budgets_locked
        for entry in entries
    )
    return {
        "source_candidates_consumed": len(entries),
        "a32_6_handoff_candidates_consumed": len(entries),
        "control_dependent_relational_contrasts_registered": len(entries),
        "unresolved_claims_excluded": len(excluded),
        "standalone_action2_effects_excluded": sum(
            "STANDALONE_ACTION2_EFFECT" in str(row.get("exclusion_reason", ""))
            for row in excluded
        ),
        "action1_universal_baselines_excluded": sum(
            "ACTION1_UNIVERSAL_BASELINE" in str(row.get("exclusion_reason", ""))
            for row in excluded
        ),
        "confirmed_support_imported_from_a32_6": sum(
            entry.confirmed_support for entry in entries
        ),
        "raw_sage_support_events_imported_directly": 0,
        "experiments_spent_total": sum(entry.experiments_spent for entry in entries),
        "registered_paired_contexts": sum(
            len(entry.paired_context_snapshot_hashes) for entry in entries
        ),
        "registered_control_actions": sum(
            len(entry.control_actions) for entry in entries
        ),
        "registered_budgets": sum(len(entry.budgets) for entry in entries),
        "scope_locked_entries": scope_locked,
        "a33_truth_reevaluations": 0,
        "a33_confirmations_performed": 0,
        "legacy_a33_1_registry_mutated": False,
        "scoped_a33_2_registry_mutated": False,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "outcome_status": A33_3_ENTRY_ADDED if entries else A33_3_NO_ELIGIBLE_ENTRY,
    }


def build_a33_3_gate(
    source: Mapping[str, Any],
    entries: Sequence[ControlDependentRelationalRegistryEntry],
    excluded: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    return {
        "source_a32_6_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source.get("gate", {}).values()),
        "every_confirmed_relational_handoff_registered_once": len(entries)
        == int(source.get("summary", {}).get("a33_ready_candidates", 0) or 0)
        == 1,
        "support_imported_from_a32_decision_only": sum(
            entry.confirmed_support for entry in entries
        )
        == int(source.get("support", 0) or 0)
        == 3,
        "paired_context_support_preserved": all(
            entry.confirmed_support == len(entry.paired_context_snapshot_hashes) == 3
            for entry in entries
        ),
        "relational_scope_fully_locked": all(
            entry.scope_game_locked
            and entry.scope_contexts_locked
            and entry.scope_target_action_locked
            and entry.scope_control_actions_locked
            and entry.scope_metric_locked
            and entry.scope_budgets_locked
            for entry in entries
        ),
        "standalone_and_baseline_claims_excluded": len(excluded) == 2
        and all(
            str(row.get("claim_status", "")) == "unresolved"
            and int(row.get("claim_support", 0) or 0) == 0
            and not bool(row.get("registered", True))
            and not bool(row.get("counted_as_refutation", True))
            for row in excluded
        ),
        "no_a33_scientific_verdict_created": all(
            entry.a32_confirmation_reused_without_reevaluation
            and not entry.a33_confirmation_performed
            for entry in entries
        ),
        "legacy_registries_quarantined": True,
        "wrong_confirmations_zero": all(entry.wrong_confirmations == 0 for entry in entries),
    }


def validate_a32_6_relational_registry_source(source: Mapping[str, Any]) -> None:
    """Reject any A32.6 payload that would broaden or invent registry truth."""
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != A32_6_SCHEMA_VERSION:
        raise ValueError("A32.6 schema version is not supported by A33.3")
    if str(source.get("outcome_status", "")) != (
        A32_6_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST_CONFIRMED
    ):
        raise ValueError("A33.3 expects the confirmed A32.6 relational outcome")
    if str(source.get("status", "")) != "CONFIRMED":
        raise ValueError("A33.3 expects a confirmed A32.6 relational record")
    if str(source.get("truth_status", "")) != A32_6_TRUTH_STATUS:
        raise ValueError("A32.6 truth status is not eligible for A33.3")
    if not bool(source.get("scientific_review_performed", False)) or not bool(
        source.get("revision_performed", False)
    ):
        raise ValueError("A32.6 scientific revision must be complete")
    if not bool(source.get("confirmation_performed", False)):
        raise ValueError("A32.6 must provide a confirmation before A33.3")
    if bool(source.get("refutation_performed", False)):
        raise ValueError("A33.3 source cannot contain a refutation")
    if not bool(source.get("a33_ready", False)):
        raise ValueError("A32.6 must explicitly mark an A33-ready handoff")
    if bool(source.get("a33_write_performed", False)):
        raise ValueError("A32.6 cannot pre-write the A33.3 registry")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("A32.6 wrong_confirmations must remain 0")
    if bool(source.get("standalone_action2_effect_confirmed", False)) or bool(
        source.get("standalone_action2_effect_refuted", False)
    ):
        raise ValueError("A32.6 standalone ACTION2 effect must remain unresolved")
    if str(source.get("standalone_action2_effect_status", "")) != "unresolved":
        raise ValueError("A32.6 standalone ACTION2 status must be unresolved")
    if bool(source.get("action3_equivalence_counted_as_refutation", False)):
        raise ValueError("A32.6 ACTION3 equivalence cannot count as refutation")
    if bool(source.get("action1_counted_as_universal_baseline", False)):
        raise ValueError("A32.6 cannot register ACTION1 as a universal baseline")
    if bool(source.get("sage_raw_events_counted_as_support_before_a32_review", False)):
        raise ValueError("A32.6 cannot inherit pre-counted SAGE support")
    if bool(source.get("scope_limited_confirmation_generalized_beyond_game", False)):
        raise ValueError("A32.6 confirmation cannot be generalized beyond its game")
    if not bool(summary.get("gate_passed", False)) or not all(
        bool(value) for value in source.get("gate", {}).values()
    ):
        raise ValueError("A32.6 gate must pass before A33.3")
    if int(summary.get("scientific_support_counted_by_a32", 0) or 0) != 3 or int(
        summary.get("raw_support_events_promoted_directly", 0) or 0
    ) != 0:
        raise ValueError("A33.3 requires three paired-context A32 supports only")

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
    if len(decisions) != 1 or len(handoffs) != 1:
        raise ValueError("A33.3 requires exactly one A32.6 decision and handoff")
    if len(decisions) != int(summary.get("scientific_revision_decisions", 0) or 0):
        raise ValueError("A32.6 revision decision count must be exact")
    if len(handoffs) != int(summary.get("a33_ready_candidates", 0) or 0):
        raise ValueError("A32.6 A33-ready candidate count must be exact")

    decision = decisions[0]
    handoff = handoffs[0]
    record = dict(decision.get("decision_record", {}) or {})
    scope = dict(decision.get("scope_limits", {}) or {})
    dispositions = dict(decision.get("claim_dispositions", {}) or {})
    relational = dict(dispositions.get("reformulated_control_dependent_contrast", {}) or {})
    standalone = dict(dispositions.get("standalone_unconditional_action2_effect", {}) or {})
    baseline = dict(dispositions.get("action1_universal_baseline", {}) or {})
    if str(decision.get("decision", "")) != (
        CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST
    ):
        raise ValueError("A33.3 accepts only A32.6 relational confirmations")
    if str(record.get("status", "")) != "confirmed" or int(
        record.get("support", 0) or 0
    ) != 3:
        raise ValueError("A33.3 accepts only the support-3 confirmed A32.6 record")
    if int(record.get("contradictions", 0) or 0) != 0 or int(
        record.get("experiments_spent", 0) or 0
    ) != 10:
        raise ValueError("A33.3 requires the exact contradiction-free A32.6 record")
    if not bool(decision.get("confirmation_performed", False)) or not bool(
        decision.get("a33_ready", False)
    ):
        raise ValueError("A32.6 relational decision must be confirmed and A33-ready")
    if bool(decision.get("refutation_performed", False)) or bool(
        decision.get("a33_write_performed", False)
    ):
        raise ValueError("A32.6 relational decision cannot refute or pre-write A33")
    if (
        str(relational.get("status", "")) != "confirmed"
        or int(relational.get("support", 0) or 0) != 3
        or not bool(relational.get("confirmation_performed", False))
    ):
        raise ValueError("A32.6 relational claim disposition must be confirmed")
    if (
        str(standalone.get("status", "")) != "unresolved"
        or int(standalone.get("support", 0) or 0) != 0
        or str(standalone.get("decision", ""))
        != KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE
        or bool(standalone.get("confirmation_performed", False))
        or bool(standalone.get("refutation_performed", False))
    ):
        raise ValueError("A32.6 standalone ACTION2 disposition must remain unresolved")
    if (
        str(baseline.get("status", "")) != "unresolved"
        or int(baseline.get("support", 0) or 0) != 0
        or bool(baseline.get("confirmation_performed", False))
        or bool(baseline.get("refutation_performed", False))
    ):
        raise ValueError("A32.6 ACTION1 baseline disposition must remain unresolved")

    key = str(handoff.get("candidate_key", ""))
    if not key or key != str(decision.get("candidate_key", "")) or key != str(
        record.get("key", "")
    ):
        raise ValueError("A32.6 handoff and confirmed record keys must align")
    if str(handoff.get("frontier_id", "")) != str(
        decision.get("frontier_id", "")
    ):
        raise ValueError("A32.6 handoff frontier must match its decision")
    if str(handoff.get("registry_entry_type", "")) != (
        CONTROL_DEPENDENT_RELATIONAL_CONTRAST
    ):
        raise ValueError("A33.3 requires the relational registry entry type")
    if str(handoff.get("status", "")) != "confirmed" or int(
        handoff.get("support", 0) or 0
    ) != int(record.get("support", 0) or 0):
        raise ValueError("A32.6 handoff support must match its confirmed record")
    if int(handoff.get("contradictions", 0) or 0) != 0 or int(
        handoff.get("experiments_spent", 0) or 0
    ) != int(record.get("experiments_spent", 0) or 0):
        raise ValueError("A32.6 handoff evidence counts must match its record")
    if not bool(handoff.get("ready_for_A33_registry_review", False)) or bool(
        handoff.get("a33_write_performed", False)
    ):
        raise ValueError("A32.6 handoff must be ready without pre-writing A33")
    if (
        not bool(handoff.get("not_generalized_beyond_game", False))
        or not bool(handoff.get("not_generalized_beyond_exact_paired_contexts", False))
        or not bool(handoff.get("not_generalized_beyond_recorded_controls", False))
    ):
        raise ValueError("A32.6 handoff relational scope must remain locked")
    if bool(handoff.get("standalone_action2_effect_confirmed", False)) or str(
        handoff.get("standalone_action2_effect_status", "")
    ) != "unresolved":
        raise ValueError("A32.6 handoff standalone ACTION2 effect must stay unresolved")
    if bool(handoff.get("action1_counted_as_universal_baseline", False)):
        raise ValueError("A32.6 handoff cannot make ACTION1 a universal baseline")

    game_id = str(handoff.get("game_id", ""))
    target_action = str(handoff.get("target_action", ""))
    controls = [str(value) for value in handoff.get("control_actions", []) or []]
    metric = str(handoff.get("metric", ""))
    budgets = [int(value) for value in handoff.get("budgets", []) or []]
    context_ids = [
        str(value) for value in handoff.get("paired_context_cluster_ids", []) or []
    ]
    context_hashes = [
        str(value)
        for value in handoff.get("paired_context_snapshot_hashes", []) or []
    ]
    pair_ids = [
        str(value)
        for value in handoff.get("source_paired_comparison_ids", []) or []
    ]
    effects = [
        row
        for row in handoff.get("controlled_effects_by_pair", []) or []
        if isinstance(row, Mapping)
    ]
    observation_ids = [
        str(value) for value in handoff.get("source_observation_ids", []) or []
    ]
    if game_id != "wa30-ee6fef47" or target_action != "ACTION2":
        raise ValueError("A33.3 is locked to the reviewed wa30 ACTION2 relation")
    if controls != ["ACTION1", "ACTION3"] or metric != "local_patch_before_after":
        raise ValueError("A33.3 controls and metric must preserve A32.6")
    if budgets != [50, 150, 300]:
        raise ValueError("A33.3 budgets must preserve the three A32.6 budgets")
    if (
        len(context_ids) != len(set(context_ids))
        or len(context_hashes) != len(set(context_hashes))
        or len(pair_ids) != len(set(pair_ids))
        or len(context_ids) != 3
        or len(context_hashes) != 3
        or len(pair_ids) != 3
        or "" in [*context_ids, *context_hashes, *pair_ids]
    ):
        raise ValueError("A32.6 handoff paired contexts must be three exact uniques")
    if len(observation_ids) != 10 or len(observation_ids) != len(set(observation_ids)):
        raise ValueError("A32.6 handoff must preserve ten unique source observations")
    if len(effects) != 3 or [
        str(row.get("paired_comparison_id", "")) for row in effects
    ] != pair_ids:
        raise ValueError("A32.6 handoff controlled effects must align with its pairs")
    if any(
        float(row.get("action2_minus_action1", 0.0) or 0.0) != 32.0
        or float(row.get("action2_minus_action3", 0.0) or 0.0) != 0.0
        for row in effects
    ):
        raise ValueError("A33.3 requires the reviewed 32/0 paired contrasts")
    if (
        str(scope.get("game_id", "")) != game_id
        or str(scope.get("target_action", "")) != target_action
        or list(scope.get("control_actions", []) or []) != controls
        or str(scope.get("metric", "")) != metric
        or list(scope.get("budgets", []) or []) != budgets
        or list(scope.get("confirmed_paired_context_cluster_ids", []) or [])
        != context_ids
        or list(scope.get("confirmed_paired_context_snapshot_hashes", []) or [])
        != context_hashes
    ):
        raise ValueError("A32.6 handoff must preserve its decision scope exactly")
    if list(decision.get("source_paired_comparison_ids", []) or []) != pair_ids or list(
        decision.get("source_observation_ids", []) or []
    ) != observation_ids:
        raise ValueError("A32.6 handoff source ids must match its decision")
    if str(handoff.get("scoped_claim", "")) != str(record.get("description", "")):
        raise ValueError("A32.6 handoff claim must match its confirmed record")
    if int(source.get("support", 0) or 0) != int(record.get("support", 0) or 0):
        raise ValueError("A32.6 top-level support must match its decision record")


def write_control_dependent_relational_registry(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the A33.3 control-dependent relational registry."
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=(
            DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_control_dependent_relational_registry_generation(
        decisions_path=args.decisions,
        output_path=args.out,
    )
    print(
        json.dumps(
            {"output_path": str(args.out), "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
