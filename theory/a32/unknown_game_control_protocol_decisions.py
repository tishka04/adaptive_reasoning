"""A32.4 decision over the SAGE.5i unknown-game control protocol proposal."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus
from theory.sage.control_surface_expansion import (
    DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH,
    SAGE5I_ACTION_DISTINCT_EXHAUSTED,
    SAGE5I_TRUTH_STATUS,
)


DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH = (
    Path("diagnostics") / "a32" / "unknown_game_control_protocol_decisions.json"
)

A32_4_SCHEMA_VERSION = "a32.unknown_game_control_protocol_decisions.v1"
A32_4_DECISION_SCOPE = "A32_UNKNOWN_GAME_CONTROL_PROTOCOL_REVIEW"

AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL = (
    "AUTHORIZE_PRE_REGISTERED_PARAMETERIZED_CONTROL_PROTOCOL"
)
RETAIN_STRICT_ACTION_DISTINCT_REQUIREMENT = "RETAIN_STRICT_ACTION_DISTINCT_REQUIREMENT"
REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE = (
    "REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE"
)

A32_4_PROTOCOL_AUTHORIZED = (
    "A32_PARAMETERIZED_CONTROL_PROTOCOL_AUTHORIZED_NO_CONFIRMATION"
)
A32_4_STRICT_REQUIREMENT_RETAINED = (
    "A32_STRICT_ACTION_DISTINCT_REQUIREMENT_RETAINED_NO_CONFIRMATION"
)
A32_4_UNIDENTIFIABLE_REJECTED = (
    "A32_CURRENT_ACTION_SURFACE_UNIDENTIFIABLE_NO_REFUTATION"
)

MIN_PARAMETER_VARIANTS = 2
MIN_REPLAY_EXACT_CONTEXTS_PER_VARIANT = 2


@dataclass(frozen=True)
class A32UnknownGameControlProtocolDecision:
    """One non-verdict A32.4 protocol decision for an unknown-game candidate."""

    candidate_id: str
    candidate_key: str
    game_id: str
    action: str
    action_args: Dict[str, Any] | None
    decision: str
    recommended_next_step: str
    reasons: Tuple[str, ...]
    evidence_summary: Dict[str, Any]
    authorized_control_variants: Tuple[Dict[str, Any], ...]
    preregistered_experiments: Tuple[Dict[str, Any], ...]
    input_record: HypothesisRecord
    decision_record: HypothesisRecord
    scientific_review_performed: bool = True
    experimental_protocol_authorized: bool = False
    revision_performed: bool = False
    confirmation_performed: bool = False
    refutation_performed: bool = False
    a33_ready: bool = False
    a33_write_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_key": self.candidate_key,
            "game_id": self.game_id,
            "action": self.action,
            "action_args": (
                dict(self.action_args) if self.action_args is not None else None
            ),
            "decision": self.decision,
            "recommended_next_step": self.recommended_next_step,
            "reasons": list(self.reasons),
            "evidence_summary": dict(self.evidence_summary),
            "authorized_control_variants": [
                dict(row) for row in self.authorized_control_variants
            ],
            "preregistered_experiments": [
                dict(row) for row in self.preregistered_experiments
            ],
            "input_record": record_to_dict(self.input_record),
            "decision_record": record_to_dict(self.decision_record),
            "scientific_review_performed": self.scientific_review_performed,
            "experimental_protocol_authorized": (self.experimental_protocol_authorized),
            "revision_performed": self.revision_performed,
            "confirmation_performed": self.confirmation_performed,
            "refutation_performed": self.refutation_performed,
            "a33_ready": self.a33_ready,
            "a33_write_performed": self.a33_write_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "action_distinct_requirement_retained_as_default": True,
            "candidate_specific_protocol_exception": (
                self.experimental_protocol_authorized
            ),
            "parameterized_controls_counted_as_distinct_actions": False,
            "parameterized_controls_counted_as_evidence_before_execution": False,
            "bounded_exhaustion_counted_as_refutation": False,
            "source_events_counted_as_scientific_support": False,
        }


def run_a32_unknown_game_control_protocol_decision_consumer(
    *,
    source_sage5i_path: str | Path = (DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH),
) -> Dict[str, Any]:
    """Review SAGE.5i and produce explicit A32.4 protocol decisions."""
    source = _load_json(source_sage5i_path)
    decisions = build_a32_unknown_game_control_protocol_decisions(source)
    requested_experiments = [
        dict(experiment)
        for decision in decisions
        for experiment in decision.preregistered_experiments
    ]
    summary = summarize_a32_unknown_game_control_protocol_decisions(decisions)
    return {
        "config": {
            "source_sage5i_path": str(source_sage5i_path),
            "schema_version": A32_4_SCHEMA_VERSION,
            "inputs_read": ["SAGE.5i"],
            "decision_scope": A32_4_DECISION_SCOPE,
            "artifacts_not_modified": ["SAGE.5i", "M2", "M3", "A33"],
            "decision_policy": {
                "historical_action_distinct_requirement_retained_as_default": True,
                "candidate_specific_parameterized_exception_requires_bounded_exhaustion": True,
                "minimum_parameter_variants": MIN_PARAMETER_VARIANTS,
                "minimum_replay_exact_contexts_per_variant": (
                    MIN_REPLAY_EXACT_CONTEXTS_PER_VARIANT
                ),
                "zero_contradiction_events_required": True,
                "source_support_remains_zero": True,
                "execution_performed": False,
                "confirmation_performed": False,
                "refutation_performed": False,
                "a33_write_performed": False,
            },
        },
        "source_sage5i_summary": dict(source.get("summary", {}) or {}),
        "protocol_decisions": [decision.to_dict() for decision in decisions],
        "requested_followup_experiments": requested_experiments,
        "input_records": [record_to_dict(row.input_record) for row in decisions],
        "decision_records": [record_to_dict(row.decision_record) for row in decisions],
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "scientific_review_performed": bool(decisions),
        "protocol_decision_performed": bool(decisions),
        "experimental_protocol_authorized": any(
            row.experimental_protocol_authorized for row in decisions
        ),
        "execution_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "a33_ready": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
        "support": 0,
        "truth_status": "UNRESOLVED_AFTER_A32_4_PROTOCOL_DECISION",
        "action_distinct_requirement_retained_as_default": True,
        "parameterized_controls_counted_as_distinct_actions": False,
        "parameterized_controls_counted_as_evidence_before_execution": False,
        "bounded_exhaustion_counted_as_refutation": False,
        "source_events_counted_as_scientific_support": False,
    }


def build_a32_unknown_game_control_protocol_decisions(
    source: Mapping[str, Any],
) -> Tuple[A32UnknownGameControlProtocolDecision, ...]:
    validate_sage5i_protocol_source(source)
    assessments = {
        str(row.get("candidate_id", "")): dict(row)
        for row in source.get("updated_candidate_assessments", []) or []
        if isinstance(row, Mapping) and str(row.get("candidate_id", ""))
    }
    audits_by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for row in source.get("context_surface_audits", []) or []:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_id", ""))
        audits_by_candidate.setdefault(candidate_id, []).append(dict(row))

    decisions: List[A32UnknownGameControlProtocolDecision] = []
    for index, surface_row in enumerate(
        source.get("candidate_control_surface_results", []) or [],
        start=1,
    ):
        if not isinstance(surface_row, Mapping):
            continue
        surface = dict(surface_row)
        candidate_id = str(surface.get("candidate_id", ""))
        assessment = assessments.get(candidate_id, {})
        decisions.append(
            decision_from_candidate(
                surface=surface,
                assessment=assessment,
                audits=audits_by_candidate.get(candidate_id, []),
                decision_index=index,
            )
        )
    return tuple(decisions)


def decision_from_candidate(
    *,
    surface: Mapping[str, Any],
    assessment: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
    decision_index: int,
) -> A32UnknownGameControlProtocolDecision:
    reasons = protocol_decision_reasons(surface=surface, assessment=assessment)
    decision = protocol_decision_label(reasons)
    available_options = [
        dict(row)
        for row in surface.get("parameterized_control_options", []) or []
        if isinstance(row, Mapping)
    ]
    selected_variants = (
        select_preregistered_variants(available_options)
        if decision == AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL
        else ()
    )
    experiments = (
        build_preregistered_experiments(
            candidate_id=str(surface.get("candidate_id", "")),
            candidate_key=str(surface.get("candidate_key", "")),
            action=str(surface.get("action", "")),
            action_args=surface.get("action_args"),
            selected_variants=selected_variants,
            audits=audits,
            decision_index=decision_index,
        )
        if selected_variants
        else ()
    )
    evidence = protocol_evidence_summary(surface=surface, assessment=assessment)
    input_record = unresolved_record(surface)
    return A32UnknownGameControlProtocolDecision(
        candidate_id=str(surface.get("candidate_id", "")),
        candidate_key=str(surface.get("candidate_key", "")),
        game_id=str(surface.get("game_id", "")),
        action=str(surface.get("action", "")),
        action_args=(
            dict(surface.get("action_args", {}) or {})
            if surface.get("action_args") is not None
            else None
        ),
        decision=decision,
        recommended_next_step=recommended_next_step_for(decision),
        reasons=reasons,
        evidence_summary=evidence,
        authorized_control_variants=selected_variants,
        preregistered_experiments=experiments,
        input_record=input_record,
        decision_record=input_record,
        experimental_protocol_authorized=(
            decision == AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL
        ),
    )


def protocol_decision_reasons(
    *,
    surface: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> Tuple[str, ...]:
    reasons: List[str] = []
    missing = set(assessment.get("missing_revision_requirements", []) or [])
    contradictions = int(assessment.get("contradiction_events_after", 0) or 0)
    raw_support = int(assessment.get("raw_support_events_after", 0) or 0)
    contexts = int(assessment.get("independent_context_events_after", 0) or 0)
    exhausted = bool(surface.get("control_surface_exhausted_action_distinct", False))
    parameter_options = int(surface.get("parameterized_control_option_count", 0) or 0)
    replay_exact_contexts = int(surface.get("replay_exact_contexts", 0) or 0)

    if raw_support < 3:
        reasons.append("insufficient_raw_support_events")
    if contexts < 2:
        reasons.append("insufficient_independent_context_events")
    if contradictions > 0:
        reasons.append("contradiction_events_present")
    if missing - {"minimum_distinct_control_actions"}:
        reasons.append("non_control_revision_requirements_missing")
    if not exhausted:
        reasons.append("action_distinct_surface_not_exhausted")
    if exhausted and parameter_options < MIN_PARAMETER_VARIANTS:
        reasons.append("insufficient_parameterized_control_variants")
    if replay_exact_contexts < (
        MIN_PARAMETER_VARIANTS * MIN_REPLAY_EXACT_CONTEXTS_PER_VARIANT
    ):
        reasons.append("insufficient_replay_exact_contexts_for_protocol")

    if not reasons and missing == {"minimum_distinct_control_actions"}:
        reasons.extend(
            [
                "only_action_distinct_control_requirement_missing",
                "bounded_action_distinct_surface_exhausted",
                "minimum_parameterized_control_variants_available",
                "candidate_specific_protocol_exception_pre_registered",
            ]
        )
    elif not reasons:
        reasons.append("retain_historical_action_distinct_requirement")
    return tuple(reasons)


def protocol_decision_label(reasons: Sequence[str]) -> str:
    reason_set = set(reasons)
    if "insufficient_parameterized_control_variants" in reason_set:
        return REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE
    authorization_reasons = {
        "only_action_distinct_control_requirement_missing",
        "bounded_action_distinct_surface_exhausted",
        "minimum_parameterized_control_variants_available",
        "candidate_specific_protocol_exception_pre_registered",
    }
    if authorization_reasons.issubset(reason_set):
        return AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL
    return RETAIN_STRICT_ACTION_DISTINCT_REQUIREMENT


def recommended_next_step_for(decision: str) -> str:
    if decision == AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL:
        return "SAGE_PRE_REGISTERED_PARAMETERIZED_CONTROL_ACQUISITION"
    if decision == REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE:
        return "STOP_CURRENT_CANDIDATE_WITHOUT_SCIENTIFIC_REFUTATION"
    return "KEEP_UNRESOLVED_OR_SEARCH_NEW_ACTION_DISTINCT_SURFACE"


def select_preregistered_variants(
    options: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    unique: Dict[str, Dict[str, Any]] = {}
    for row in options:
        if bool(row.get("counted_as_action_distinct_control", False)):
            continue
        variant = {
            "action": str(row.get("action", "")),
            "action_args": dict(row.get("action_args", {}) or {}),
            "parameterized_control_role": str(
                row.get("parameterized_control_role", "")
            ),
            "counted_as_action_distinct_control": False,
        }
        unique[_canonical_json(variant)] = variant
    ordered = sorted(
        unique.values(),
        key=_variant_sort_key,
    )
    if len(ordered) < MIN_PARAMETER_VARIANTS:
        return ()
    pairs = [
        (left, right)
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
    ]
    max_distance = max(_variant_distance(left, right) for left, right in pairs)
    most_diverse = [
        (left, right)
        for left, right in pairs
        if _variant_distance(left, right) == max_distance
    ]
    most_diverse.sort(
        key=lambda pair: (
            _canonical_json(pair[0]),
            _canonical_json(pair[1]),
        )
    )
    return most_diverse[0]


def build_preregistered_experiments(
    *,
    candidate_id: str,
    candidate_key: str,
    action: str,
    action_args: Any,
    selected_variants: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    decision_index: int,
) -> Tuple[Dict[str, Any], ...]:
    assignments = select_preregistered_contexts(
        audits,
        variant_count=len(selected_variants),
        contexts_per_variant=MIN_REPLAY_EXACT_CONTEXTS_PER_VARIANT,
    )
    if len(assignments) != len(selected_variants):
        return ()

    measurement = measurement_from_candidate_key(candidate_key)
    experiments: List[Dict[str, Any]] = []
    experiment_index = 0
    for variant_index, (variant, contexts) in enumerate(
        zip(selected_variants, assignments),
        start=1,
    ):
        for context in contexts:
            experiment_index += 1
            experiments.append(
                {
                    "experiment_id": (
                        f"a32.4::parameterized_control_experiment::"
                        f"{decision_index:03d}::{experiment_index:03d}"
                    ),
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_key,
                    "variant_index": variant_index,
                    "target_action": action,
                    "target_action_args": (
                        dict(action_args or {}) if action_args is not None else None
                    ),
                    "control_action": str(variant.get("action", "")),
                    "control_action_args": dict(variant.get("action_args", {}) or {}),
                    "measurement": measurement,
                    "source_audit_id": str(context.get("audit_id", "")),
                    "source_request_id": str(context.get("source_request_id", "")),
                    "context_snapshot_hash": str(
                        context.get("context_snapshot_hash", "")
                    ),
                    "budget": int(context.get("budget", 0) or 0),
                    "paired_target_control_required": True,
                    "exact_context_replay_required": True,
                    "same_measurement_for_target_and_control_required": True,
                    "evaluation_rule": (
                        "compare_paired_target_and_control_effect_signatures_"
                        "without_post_hoc_metric_change"
                    ),
                    "parameterized_control_counted_as_distinct_action": False,
                    "status": "PRE_REGISTERED_NOT_EXECUTED",
                    "support": 0,
                }
            )
    return tuple(experiments)


def select_preregistered_contexts(
    audits: Sequence[Mapping[str, Any]],
    *,
    variant_count: int,
    contexts_per_variant: int,
) -> Tuple[Tuple[Dict[str, Any], ...], ...]:
    exact = [
        dict(row)
        for row in audits
        if bool(row.get("live_prefix_replay_exact", False))
        and str(row.get("source_request_id", ""))
        and str(row.get("context_snapshot_hash", ""))
    ]
    exact.sort(
        key=lambda row: (
            int(row.get("budget", 0) or 0),
            str(row.get("source_request_id", "")),
        )
    )
    needed = variant_count * contexts_per_variant
    if variant_count <= 0 or contexts_per_variant <= 0 or len(exact) < needed:
        return ()

    by_budget: Dict[int, List[Dict[str, Any]]] = {}
    for row in exact:
        by_budget.setdefault(int(row.get("budget", 0) or 0), []).append(row)
    budgets = sorted(by_budget)
    if (
        contexts_per_variant == 2
        and len(budgets) >= 2
        and len(by_budget[budgets[0]]) >= variant_count
        and len(by_budget[budgets[-1]]) >= variant_count
    ):
        return tuple(
            (
                by_budget[budgets[0]][index],
                by_budget[budgets[-1]][index],
            )
            for index in range(variant_count)
        )

    selected = exact[:needed]
    return tuple(
        tuple(
            selected[index * contexts_per_variant : (index + 1) * contexts_per_variant]
        )
        for index in range(variant_count)
    )


def protocol_evidence_summary(
    *,
    surface: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "raw_support_events": int(assessment.get("raw_support_events_after", 0) or 0),
        "independent_context_events": int(
            assessment.get("independent_context_events_after", 0) or 0
        ),
        "distinct_control_action_names": int(
            assessment.get("distinct_control_actions_after", 0) or 0
        ),
        "contradiction_events": int(
            assessment.get("contradiction_events_after", 0) or 0
        ),
        "missing_revision_requirements": list(
            assessment.get("missing_revision_requirements", []) or []
        ),
        "matching_contexts_scanned": int(surface.get("contexts_scanned", 0) or 0),
        "replay_exact_contexts_scanned": int(
            surface.get("replay_exact_contexts", 0) or 0
        ),
        "bounded_action_distinct_surface_exhausted": bool(
            surface.get("control_surface_exhausted_action_distinct", False)
        ),
        "parameterized_control_options_available": int(
            surface.get("parameterized_control_option_count", 0) or 0
        ),
        "source_support_counted_as_scientific_support": False,
        "bounded_exhaustion_counted_as_refutation": False,
    }


def unresolved_record(surface: Mapping[str, Any]) -> HypothesisRecord:
    return HypothesisRecord(
        key=str(surface.get("candidate_key", "")),
        description=(
            f"Unknown-game effect candidate for {surface.get('action', '')} "
            "under the bounded SAGE.5i replay surface."
        ),
        status=HypothesisStatus.UNRESOLVED,
        support=0,
        contradictions=0,
        experiments_spent=0,
    )


def measurement_from_candidate_key(candidate_key: str) -> str:
    measurements = [
        part
        for part in str(candidate_key).split("::")
        if part.endswith("_before_after")
    ]
    if len(measurements) != 1:
        raise ValueError("candidate key must identify exactly one before/after metric")
    return measurements[0]


def summarize_a32_unknown_game_control_protocol_decisions(
    decisions: Sequence[A32UnknownGameControlProtocolDecision],
) -> Dict[str, Any]:
    authorized = [
        row
        for row in decisions
        if row.decision == AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL
    ]
    retained = [
        row
        for row in decisions
        if row.decision == RETAIN_STRICT_ACTION_DISTINCT_REQUIREMENT
    ]
    rejected = [
        row
        for row in decisions
        if row.decision == REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE
    ]
    if authorized and len(authorized) == len(decisions):
        outcome_status = A32_4_PROTOCOL_AUTHORIZED
    elif rejected and len(rejected) == len(decisions):
        outcome_status = A32_4_UNIDENTIFIABLE_REJECTED
    else:
        outcome_status = A32_4_STRICT_REQUIREMENT_RETAINED
    return {
        "source_candidates_consumed": len(decisions),
        "protocol_decisions": len(decisions),
        "parameterized_protocols_authorized": len(authorized),
        "strict_action_distinct_requirements_retained_without_exception": len(retained),
        "candidates_rejected_as_unidentifiable": len(rejected),
        "authorized_parameter_variants": sum(
            len(row.authorized_control_variants) for row in authorized
        ),
        "preregistered_followup_experiments": sum(
            len(row.preregistered_experiments) for row in authorized
        ),
        "decision_records_unresolved": sum(
            row.decision_record.status == HypothesisStatus.UNRESOLVED
            for row in decisions
        ),
        "decision_records_confirmed": 0,
        "decision_records_refuted": 0,
        "scientific_review_performed": bool(decisions),
        "protocol_decision_performed": bool(decisions),
        "experimental_protocol_authorized": bool(authorized),
        "execution_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "a33_ready_candidates": 0,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
        "support": 0,
        "outcome_status": outcome_status,
    }


def validate_sage5i_protocol_source(source: Mapping[str, Any]) -> None:
    summary = dict(source.get("summary", {}) or {})
    config = dict(source.get("config", {}) or {})
    proposal = dict(source.get("a32_parameterized_control_protocol_proposal", {}) or {})
    if str(config.get("schema_version", "")) != "sage.control_surface_expansion.v1":
        raise ValueError("SAGE.5i schema version is not supported by A32.4")
    if str(source.get("truth_status", "")) != SAGE5I_TRUTH_STATUS:
        raise ValueError("SAGE.5i truth status must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.5i revision status must remain candidate-only")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.5i candidates must remain unresolved")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.5i support must remain 0")
    if bool(source.get("revision_performed", False)):
        raise ValueError("SAGE.5i must not perform revision")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.5i wrong_confirmations must remain 0")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.5i cannot write A32/A33")
    if bool(source.get("parameterized_controls_counted_as_distinct_actions", False)):
        raise ValueError("SAGE.5i cannot relabel parameterized controls")
    if bool(
        source.get("bounded_control_surface_exhaustion_counted_as_refutation", False)
    ):
        raise ValueError("SAGE.5i bounded exhaustion cannot count as refutation")
    if str(source.get("outcome_status", "")) != SAGE5I_ACTION_DISTINCT_EXHAUSTED:
        raise ValueError("A32.4 requires SAGE.5i action-distinct exhaustion")
    if not bool(summary.get("gate_passed", False)):
        raise ValueError("SAGE.5i gate must pass before A32.4 review")
    if not bool(summary.get("bounded_action_distinct_exhaustion_proven", False)):
        raise ValueError("SAGE.5i bounded exhaustion must be proven")
    if not bool(summary.get("all_candidate_contexts_audited_exact", False)):
        raise ValueError("all SAGE.5i candidate contexts must replay exactly")
    if not bool(summary.get("bounded_scope_only", False)):
        raise ValueError("SAGE.5i exhaustion must remain bounded in scope")
    if not bool(proposal.get("proposal_required", False)):
        raise ValueError("SAGE.5i must request an A32 protocol decision")
    if str(proposal.get("proposal_status", "")) != (
        "A32_REVIEW_REQUIRED_DO_NOT_AUTO_RELAX_CRITERION"
    ):
        raise ValueError("SAGE.5i must prohibit automatic criterion relaxation")
    if bool(proposal.get("protocol_proposal_counted_as_revision", False)):
        raise ValueError("SAGE.5i protocol proposal cannot count as revision")
    if not _required_allowed_decisions().issubset(
        set(proposal.get("allowed_decisions", []) or [])
    ):
        raise ValueError("SAGE.5i must expose every allowed A32 decision")

    surfaces = [
        row
        for row in source.get("candidate_control_surface_results", []) or []
        if isinstance(row, Mapping)
    ]
    assessments = [
        row
        for row in source.get("updated_candidate_assessments", []) or []
        if isinstance(row, Mapping)
    ]
    if not surfaces or len(surfaces) != len(assessments):
        raise ValueError(
            "SAGE.5i surfaces and assessments must be non-empty and aligned"
        )
    surface_ids = {str(row.get("candidate_id", "")) for row in surfaces}
    assessment_ids = {str(row.get("candidate_id", "")) for row in assessments}
    if "" in surface_ids or surface_ids != assessment_ids:
        raise ValueError("SAGE.5i surface and assessment candidate ids must align")
    if set(proposal.get("candidates_affected", []) or []) != surface_ids:
        raise ValueError("SAGE.5i proposal must cover every candidate")

    audits = list(source.get("context_surface_audits", []) or [])
    expected_audits = sum(int(row.get("contexts_scanned", 0) or 0) for row in surfaces)
    if len(audits) != expected_audits:
        raise ValueError("SAGE.5i context audit count must match candidate surfaces")
    audit_ids: set[str] = set()
    request_ids: set[str] = set()
    for row in audits:
        if not isinstance(row, Mapping) or not bool(
            row.get("live_prefix_replay_exact", False)
        ):
            raise ValueError("every SAGE.5i context audit must be replay-exact")
        audit_id = str(row.get("audit_id", ""))
        request_id = str(row.get("source_request_id", ""))
        if not audit_id or audit_id in audit_ids:
            raise ValueError("SAGE.5i context audit ids must be unique")
        if not request_id or request_id in request_ids:
            raise ValueError("SAGE.5i source request ids must be unique")
        if str(row.get("candidate_id", "")) not in surface_ids:
            raise ValueError("SAGE.5i audits must belong to reviewed candidates")
        audit_ids.add(audit_id)
        request_ids.add(request_id)
    for row in surfaces:
        if int(row.get("contexts_scanned", 0) or 0) != int(
            row.get("replay_exact_contexts", 0) or 0
        ):
            raise ValueError("every SAGE.5i candidate surface must replay exactly")
        options = [
            option
            for option in row.get("parameterized_control_options", []) or []
            if isinstance(option, Mapping)
        ]
        if len(options) != int(row.get("parameterized_control_option_count", 0) or 0):
            raise ValueError("SAGE.5i parameterized option count must be exact")
        candidate_audits = [
            audit
            for audit in audits
            if str(audit.get("candidate_id", "")) == str(row.get("candidate_id", ""))
        ]
        if len(candidate_audits) != int(row.get("contexts_scanned", 0) or 0):
            raise ValueError("SAGE.5i audits must align with each candidate surface")
        unique_contexts = {
            str(audit.get("context_snapshot_hash", ""))
            for audit in candidate_audits
            if str(audit.get("context_snapshot_hash", ""))
        }
        if len(unique_contexts) != int(row.get("unique_contexts", 0) or 0):
            raise ValueError("SAGE.5i candidate context hashes must be unique")
        if any(
            bool(option.get("counted_as_action_distinct_control", False))
            for option in options
        ):
            raise ValueError("parameterized options cannot be action-distinct controls")


def _required_allowed_decisions() -> set[str]:
    return {
        "retain_strict_action_distinct_requirement_and_keep_unresolved",
        "authorize_pre_registered_parameterized_control_protocol",
        "reject_candidate_as_unidentifiable_in_current_action_surface",
    }


def record_to_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_unknown_game_control_protocol_decisions(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _variant_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    left_args = dict(left.get("action_args", {}) or {})
    right_args = dict(right.get("action_args", {}) or {})
    distance = 0.0
    for key in set(left_args) | set(right_args):
        left_value = left_args.get(key)
        right_value = right_args.get(key)
        if (
            isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
        ):
            distance += abs(float(left_value) - float(right_value))
        elif left_value != right_value:
            distance += 1.0
    if str(left.get("action", "")) != str(right.get("action", "")):
        distance += 1.0
    return distance


def _variant_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    args = dict(row.get("action_args", {}) or {})
    sortable_args = []
    for key, value in sorted(args.items(), key=lambda item: str(item[0])):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            sortable_value: Tuple[int, Any] = (0, float(value))
        else:
            sortable_value = (1, _canonical_json(value))
        sortable_args.append((str(key), sortable_value))
    return (str(row.get("action", "")), tuple(sortable_args))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A32.4 unknown-game control protocol decisions.",
    )
    parser.add_argument(
        "--source-sage5i",
        type=Path,
        default=DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_unknown_game_control_protocol_decision_consumer(
        source_sage5i_path=args.source_sage5i,
    )
    write_a32_unknown_game_control_protocol_decisions(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "decision_scope": A32_4_DECISION_SCOPE,
                "a33_write_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
