"""A32.5 scientific revision over the SAGE.5j parameterized controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus
from theory.sage.parameterized_control_acquisition import (
    CANDIDATE_CONTROL_EXCEEDS,
    CANDIDATE_DISCRIMINATING,
    CANDIDATE_NON_DISCRIMINATING,
    DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH,
    SAGE5J_MIXED_DISCRIMINATION,
    SAGE5J_SCHEMA_VERSION,
    SAGE5J_TRUTH_STATUS,
)


DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH = (
    Path("diagnostics")
    / "a32"
    / "unknown_game_parameterized_control_revision_decisions.json"
)

A32_5_SCHEMA_VERSION = (
    "a32.unknown_game_parameterized_control_revision_decisions.v1"
)
A32_5_DECISION_SCOPE = "A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION"

CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION = (
    "CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION"
)
KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL = (
    "KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL"
)
REFUTE_AFTER_PARAMETERIZED_CONTROL_CONTRADICTION = (
    "REFUTE_AFTER_PARAMETERIZED_CONTROL_CONTRADICTION"
)
REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL = (
    "REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL"
)

A32_5_MIXED_SCOPE_CONFIRMATION_AND_NON_IDENTIFIABILITY = (
    "A32_SCOPE_LIMITED_CONFIRMATION_AND_NON_IDENTIFIABILITY"
)
A32_5_ALL_SCOPE_CONFIRMED = "A32_ALL_CANDIDATES_SCOPE_LIMITED_CONFIRMED"
A32_5_NO_CONFIRMATION = "A32_NO_CANDIDATE_CONFIRMED_AFTER_PARAMETERIZED_REVISION"


@dataclass(frozen=True)
class A32UnknownGameParameterizedControlRevisionDecision:
    """One A32.5 verdict over a completed SAGE.5j candidate protocol."""

    candidate_id: str
    candidate_key: str
    game_id: str
    action: str
    action_args: Dict[str, Any] | None
    decision: str
    recommended_next_step: str
    reasons: Tuple[str, ...]
    evidence_summary: Dict[str, Any]
    scope_limits: Dict[str, Any]
    source_experiment_ids: Tuple[str, ...]
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
            "scope_limits": dict(self.scope_limits),
            "source_experiment_ids": list(self.source_experiment_ids),
            "input_record": record_to_dict(self.input_record),
            "decision_record": record_to_dict(self.decision_record),
            "scientific_review_performed": self.scientific_review_performed,
            "revision_performed": self.revision_performed,
            "confirmation_performed": self.confirmation_performed,
            "refutation_performed": self.refutation_performed,
            "a33_ready": self.a33_ready,
            "a33_write_performed": self.a33_write_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "parameterized_controls_counted_as_distinct_actions": False,
            "sage_candidate_events_counted_as_support_before_a32_review": False,
            "neutral_events_counted_as_refutation": False,
            "non_discrimination_counted_as_refutation": False,
            "scope_limited_confirmation_generalized_beyond_game": False,
        }


def run_a32_unknown_game_parameterized_control_revision_consumer(
    *,
    source_sage5j_path: str | Path = (
        DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH
    ),
) -> Dict[str, Any]:
    """Consume SAGE.5j and emit the explicit A32.5 scientific decisions."""
    source = _load_json(source_sage5j_path)
    decisions = build_a32_unknown_game_parameterized_control_revision_decisions(
        source
    )
    summary = summarize_a32_unknown_game_parameterized_control_revision_decisions(
        decisions
    )
    handoffs = build_a33_handoff_candidates(decisions, source)
    return {
        "config": {
            "source_sage5j_path": str(source_sage5j_path),
            "schema_version": A32_5_SCHEMA_VERSION,
            "inputs_read": ["SAGE.5j"],
            "decision_scope": A32_5_DECISION_SCOPE,
            "artifacts_not_modified": ["SAGE.5j", "M2", "M3", "A33"],
            "decision_policy": {
                "confirmation_requires_complete_exact_preregistered_protocol": True,
                "confirmation_requires_all_pairs_discriminating": True,
                "confirmation_requires_zero_contradiction_events": True,
                "confirmation_is_game_and_candidate_scoped": True,
                "equal_parameterized_control_effect_is_non_identifiability": True,
                "non_discrimination_is_not_refutation": True,
                "parameterized_controls_are_not_distinct_actions": True,
                "a33_write_performed": False,
            },
        },
        "source_sage5j_summary": dict(source.get("summary", {}) or {}),
        "revision_decisions": [row.to_dict() for row in decisions],
        "input_records": [record_to_dict(row.input_record) for row in decisions],
        "decision_records": [
            record_to_dict(row.decision_record) for row in decisions
        ],
        "a33_handoff_candidates": handoffs,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": summary["status"],
        "truth_status": "SCOPED_A32_DECISION_WITHOUT_EXTERNAL_ORACLE",
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
        "parameterized_controls_counted_as_distinct_actions": False,
        "sage_candidate_events_counted_as_support_before_a32_review": False,
        "neutral_events_counted_as_refutation": False,
        "non_discrimination_counted_as_refutation": False,
        "scope_limited_confirmation_generalized_beyond_game": False,
    }


def build_a32_unknown_game_parameterized_control_revision_decisions(
    source: Mapping[str, Any],
) -> Tuple[A32UnknownGameParameterizedControlRevisionDecision, ...]:
    validate_sage5j_parameterized_control_revision_source(source)
    experiments_by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for experiment in source.get("executed_parameterized_control_experiments", []) or []:
        candidate_id = str(experiment.get("candidate_id", ""))
        experiments_by_candidate.setdefault(candidate_id, []).append(dict(experiment))

    decisions: List[A32UnknownGameParameterizedControlRevisionDecision] = []
    for assessment in source.get("candidate_protocol_assessments", []) or []:
        candidate_id = str(assessment.get("candidate_id", ""))
        decisions.append(
            decision_from_protocol_assessment(
                assessment,
                experiments_by_candidate.get(candidate_id, []),
            )
        )
    return tuple(decisions)


def decision_from_protocol_assessment(
    assessment: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
) -> A32UnknownGameParameterizedControlRevisionDecision:
    reasons = parameterized_control_revision_decision_reasons(
        assessment,
        experiments,
    )
    decision = parameterized_control_revision_decision_label(reasons)
    experiment_rows = tuple(dict(row) for row in experiments)
    executed = int(assessment.get("executed_experiments", 0) or 0)
    raw_support = int(assessment.get("raw_support_events", 0) or 0)
    raw_neutral = int(assessment.get("raw_neutral_events", 0) or 0)
    raw_contradictions = int(assessment.get("raw_contradiction_events", 0) or 0)
    confirmed = decision == CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION
    refuted = decision == REFUTE_AFTER_PARAMETERIZED_CONTROL_CONTRADICTION
    action_args = assessment.get("action_args")
    args = dict(action_args) if isinstance(action_args, Mapping) else None
    key = str(assessment.get("candidate_key", ""))
    action = str(assessment.get("action", ""))
    game_id = str(assessment.get("game_id", ""))

    evidence_summary = {
        "requested_experiments": int(
            assessment.get("requested_experiments", 0) or 0
        ),
        "executed_experiments": executed,
        "blocked_experiments": int(
            assessment.get("blocked_experiments", 0) or 0
        ),
        "replay_exact_protocol_matches": int(
            assessment.get("replay_exact_protocol_matches", 0) or 0
        ),
        "distinct_parameterized_interventions": int(
            assessment.get("distinct_parameterized_interventions_executed", 0) or 0
        ),
        "raw_support_events": raw_support,
        "raw_neutral_events": raw_neutral,
        "raw_contradiction_events": raw_contradictions,
        "parameterized_protocol_result": str(
            assessment.get("parameterized_protocol_result", "")
        ),
        "target_signals": [float(row.get("target_signal", 0.0)) for row in experiments],
        "control_signals": [
            float(row.get("control_signal", 0.0)) for row in experiments
        ],
        "source_support_before_a32_review": int(assessment.get("support", 0) or 0),
        "scientific_support_after_a32_review": raw_support if confirmed else 0,
    }
    scope_limits = _scope_limits(assessment, experiment_rows)
    input_record = HypothesisRecord(
        key=key,
        description=(
            f"Unknown-game effect candidate for {action} before A32.5 review."
        ),
        status=HypothesisStatus.UNRESOLVED,
        support=0,
        contradictions=0,
        experiments_spent=executed,
    )
    if confirmed:
        decision_record = HypothesisRecord(
            key=key,
            description=(
                f"Scope-limited {game_id} effect for {action}, discriminated from "
                "the pre-registered parameterized controls."
            ),
            status=HypothesisStatus.CONFIRMED,
            support=raw_support,
            contradictions=raw_contradictions,
            experiments_spent=executed,
        )
        recommended_next_step = (
            "Submit only this game- and candidate-scoped decision to A33 registry "
            "review; do not generalize the mechanic."
        )
    elif refuted:
        decision_record = HypothesisRecord(
            key=key,
            description=f"A32.5 refuted unknown-game effect candidate for {action}.",
            status=HypothesisStatus.REFUTED,
            support=0,
            contradictions=raw_contradictions,
            experiments_spent=executed,
        )
        recommended_next_step = "Retire this candidate from confirmed-memory intake."
    else:
        decision_record = HypothesisRecord(
            key=key,
            description=(
                f"Unknown-game effect candidate for {action} remains unresolved "
                "after A32.5 review."
            ),
            status=HypothesisStatus.UNRESOLVED,
            support=0,
            contradictions=raw_contradictions,
            experiments_spent=executed,
        )
        if decision == KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL:
            recommended_next_step = (
                "Stop treating the target arguments as an identifiable mechanic; "
                "reformulate only if a position-invariant ACTION6 claim is needed."
            )
        else:
            recommended_next_step = (
                "Complete a newly pre-registered exact protocol before another verdict."
            )

    return A32UnknownGameParameterizedControlRevisionDecision(
        candidate_id=str(assessment.get("candidate_id", "")),
        candidate_key=key,
        game_id=game_id,
        action=action,
        action_args=args,
        decision=decision,
        recommended_next_step=recommended_next_step,
        reasons=reasons,
        evidence_summary=evidence_summary,
        scope_limits=scope_limits,
        source_experiment_ids=tuple(
            str(row.get("protocol_experiment_id", "")) for row in experiment_rows
        ),
        input_record=input_record,
        decision_record=decision_record,
        confirmation_performed=confirmed,
        refutation_performed=refuted,
        a33_ready=confirmed,
    )


def parameterized_control_revision_decision_reasons(
    assessment: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    reasons: List[str] = []
    requested = int(assessment.get("requested_experiments", 0) or 0)
    executed = int(assessment.get("executed_experiments", 0) or 0)
    exact = int(assessment.get("replay_exact_protocol_matches", 0) or 0)
    blocked = int(assessment.get("blocked_experiments", 0) or 0)
    support = int(assessment.get("raw_support_events", 0) or 0)
    neutral = int(assessment.get("raw_neutral_events", 0) or 0)
    contradictions = int(assessment.get("raw_contradiction_events", 0) or 0)
    result = str(assessment.get("parameterized_protocol_result", ""))
    complete = (
        requested > 0
        and executed == requested
        and exact == requested
        and len(experiments) == requested
        and blocked == 0
        and bool(assessment.get("protocol_execution_complete", False))
        and bool(assessment.get("variant_replication_complete", False))
        and bool(assessment.get("ready_for_A32_protocol_result_review", False))
    )
    if complete:
        reasons.extend(
            [
                "pre_registered_protocol_execution_complete",
                "all_paired_experiments_replay_and_protocol_exact",
                "parameterized_variant_replication_complete",
            ]
        )
    else:
        reasons.append("pre_registered_protocol_execution_incomplete")
    if contradictions > 0 or result == CANDIDATE_CONTROL_EXCEEDS:
        reasons.append("parameterized_control_contradiction_observed")
    if complete and result == CANDIDATE_DISCRIMINATING:
        if support == executed and neutral == 0 and contradictions == 0:
            reasons.extend(
                [
                    "all_exact_pairs_discriminate_target_from_parameterized_controls",
                    "zero_neutral_or_contradiction_events",
                    "scope_limited_confirmation_criteria_satisfied",
                ]
            )
        else:
            reasons.append("discriminating_result_counts_are_inconsistent")
    if complete and result == CANDIDATE_NON_DISCRIMINATING:
        if neutral == executed and support == 0 and contradictions == 0:
            reasons.extend(
                [
                    "all_exact_pairs_reproduce_target_effect_in_parameterized_controls",
                    "target_argument_specific_effect_is_not_identifiable",
                    "non_discrimination_is_not_refutation_of_generic_action_effect",
                ]
            )
        else:
            reasons.append("non_discriminating_result_counts_are_inconsistent")
    reasons.extend(
        [
            "parameterized_controls_not_relabelled_as_distinct_actions",
            "sage_candidate_events_not_counted_as_support_before_a32_review",
        ]
    )
    return tuple(reasons)


def parameterized_control_revision_decision_label(reasons: Sequence[str]) -> str:
    reason_set = set(reasons)
    if "parameterized_control_contradiction_observed" in reason_set:
        return REFUTE_AFTER_PARAMETERIZED_CONTROL_CONTRADICTION
    if "pre_registered_protocol_execution_incomplete" in reason_set:
        return REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL
    if "scope_limited_confirmation_criteria_satisfied" in reason_set:
        return CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION
    if "target_argument_specific_effect_is_not_identifiable" in reason_set:
        return KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL
    return REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL


def build_a33_handoff_candidates(
    decisions: Sequence[A32UnknownGameParameterizedControlRevisionDecision],
    source: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    experiments_by_candidate: Dict[str, List[Mapping[str, Any]]] = {}
    for row in source.get("executed_parameterized_control_experiments", []) or []:
        experiments_by_candidate.setdefault(str(row.get("candidate_id", "")), []).append(
            row
        )
    handoffs: List[Dict[str, Any]] = []
    for decision in decisions:
        if not decision.a33_ready:
            continue
        experiments = experiments_by_candidate.get(decision.candidate_id, [])
        variants = {
            _canonical_json(
                {
                    "action": row.get("control_action"),
                    "action_args": row.get("control_action_args"),
                }
            )
            for row in experiments
        }
        handoffs.append(
            {
                "candidate_id": decision.candidate_id,
                "candidate_key": decision.candidate_key,
                "game_id": decision.game_id,
                "action": decision.action,
                "action_args": decision.action_args,
                "status": HypothesisStatus.CONFIRMED.value,
                "support": decision.decision_record.support,
                "contradictions": decision.decision_record.contradictions,
                "experiments_spent": decision.decision_record.experiments_spent,
                "scoped_claim": decision.decision_record.description,
                "measurement": _single_value(experiments, "measurement"),
                "budgets": sorted({int(row.get("budget", 0) or 0) for row in experiments}),
                "context_snapshot_hashes": sorted(
                    {str(row.get("context_snapshot_hash", "")) for row in experiments}
                ),
                "parameterized_control_variants": [
                    json.loads(row) for row in sorted(variants)
                ],
                "source_experiment_ids": list(decision.source_experiment_ids),
                "ready_for_A33_registry_review": True,
                "a33_write_performed": False,
                "not_generalized_beyond_game": True,
                "not_generalized_beyond_candidate_scope": True,
            }
        )
    return handoffs


def summarize_a32_unknown_game_parameterized_control_revision_decisions(
    decisions: Sequence[A32UnknownGameParameterizedControlRevisionDecision],
) -> Dict[str, Any]:
    confirmed = [row for row in decisions if row.confirmation_performed]
    refuted = [row for row in decisions if row.refutation_performed]
    non_identifiable = [
        row
        for row in decisions
        if row.decision == KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL
    ]
    incomplete = [
        row
        for row in decisions
        if row.decision == REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL
    ]
    if confirmed and len(confirmed) == len(decisions):
        outcome = A32_5_ALL_SCOPE_CONFIRMED
    elif confirmed and non_identifiable:
        outcome = A32_5_MIXED_SCOPE_CONFIRMATION_AND_NON_IDENTIFIABILITY
    else:
        outcome = A32_5_NO_CONFIRMATION
    if confirmed and (non_identifiable or incomplete) and not refuted:
        status = "MIXED_CONFIRMED_AND_UNRESOLVED"
    elif confirmed and refuted:
        status = "MIXED_CONFIRMED_AND_REFUTED"
    elif confirmed:
        status = "CONFIRMED"
    elif refuted and len(refuted) == len(decisions):
        status = "REFUTED"
    else:
        status = "UNRESOLVED"
    return {
        "source_candidates_consumed": len(decisions),
        "scientific_revision_decisions": len(decisions),
        "scope_limited_confirmations": len(confirmed),
        "non_identifiable_candidates_kept_unresolved": len(non_identifiable),
        "candidates_refuted": len(refuted),
        "candidates_requesting_more_tests": len(incomplete),
        "decision_records_confirmed": len(confirmed),
        "decision_records_unresolved": sum(
            row.decision_record.status == HypothesisStatus.UNRESOLVED
            for row in decisions
        ),
        "decision_records_refuted": len(refuted),
        "a33_ready_candidates": len(confirmed),
        "scientific_support_counted_by_a32": sum(
            row.decision_record.support for row in confirmed
        ),
        "neutral_events_counted_as_refutation": False,
        "non_discrimination_counted_as_refutation": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
        "status": status,
        "outcome_status": outcome,
    }


def validate_sage5j_parameterized_control_revision_source(
    source: Mapping[str, Any],
) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE5J_SCHEMA_VERSION:
        raise ValueError("SAGE.5j schema version is not supported by A32.5")
    if str(source.get("truth_status", "")) != SAGE5J_TRUTH_STATUS:
        raise ValueError("SAGE.5j truth status must remain unevaluated before A32.5")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.5j must remain candidate-only before A32.5")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.5j candidates must remain unresolved before A32.5")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.5j support must remain 0 before A32.5")
    if not bool(source.get("execution_performed", False)):
        raise ValueError("SAGE.5j must execute the protocol before A32.5")
    if bool(source.get("revision_performed", False)) or bool(
        source.get("confirmation_performed", False)
    ):
        raise ValueError("SAGE.5j cannot revise or confirm before A32.5")
    if bool(source.get("refutation_performed", False)):
        raise ValueError("SAGE.5j cannot refute before A32.5")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.5j cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.5j wrong_confirmations must remain 0")
    if bool(source.get("parameterized_controls_counted_as_distinct_actions", False)):
        raise ValueError("SAGE.5j cannot relabel parameterized controls")
    if bool(
        source.get("parameterized_control_events_counted_as_scientific_support", False)
    ):
        raise ValueError("SAGE.5j cannot count candidate events as scientific support")
    if bool(source.get("neutral_events_counted_as_refutation", False)):
        raise ValueError("SAGE.5j neutral events cannot count as refutation")
    if bool(source.get("protocol_result_counted_as_a32_decision", False)):
        raise ValueError("SAGE.5j cannot make the A32.5 decision")
    if str(source.get("outcome_status", "")) != SAGE5J_MIXED_DISCRIMINATION:
        raise ValueError("A32.5 expects the mixed SAGE.5j acquisition outcome")
    required_summary_flags = (
        "gate_passed",
        "all_pre_registered_experiments_executed_exactly",
        "all_pre_registered_experiments_resolved",
    )
    if not all(bool(summary.get(name, False)) for name in required_summary_flags):
        raise ValueError("SAGE.5j exact execution gate must pass before A32.5")
    if int(summary.get("protocol_substitutions_detected", 0) or 0) != 0:
        raise ValueError("A32.5 rejects substituted SAGE.5j experiments")
    if int(summary.get("experiments_blocked", 0) or 0) != 0:
        raise ValueError("A32.5 requires zero blocked SAGE.5j experiments")

    assessments = [
        row
        for row in source.get("candidate_protocol_assessments", []) or []
        if isinstance(row, Mapping)
    ]
    experiments = [
        row
        for row in source.get("executed_parameterized_control_experiments", []) or []
        if isinstance(row, Mapping)
    ]
    if not assessments or not experiments:
        raise ValueError("A32.5 requires assessments and executed experiments")
    candidate_ids = [str(row.get("candidate_id", "")) for row in assessments]
    if "" in candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("SAGE.5j assessment candidate ids must be unique")
    experiment_ids = [str(row.get("protocol_experiment_id", "")) for row in experiments]
    if "" in experiment_ids or len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("SAGE.5j experiment ids must be unique")
    if set(str(row.get("candidate_id", "")) for row in experiments) != set(
        candidate_ids
    ):
        raise ValueError("SAGE.5j experiments and assessments must align")
    if len(experiments) != int(summary.get("experiments_executed", 0) or 0):
        raise ValueError("SAGE.5j executed experiment count must be exact")

    total_support = total_neutral = total_contradictions = 0
    for assessment in assessments:
        candidate_id = str(assessment.get("candidate_id", ""))
        rows = [
            row
            for row in experiments
            if str(row.get("candidate_id", "")) == candidate_id
        ]
        requested = int(assessment.get("requested_experiments", 0) or 0)
        if requested <= 0 or len(rows) != requested:
            raise ValueError("SAGE.5j candidate experiment count must be exact")
        if int(assessment.get("executed_experiments", 0) or 0) != requested:
            raise ValueError("SAGE.5j candidate execution must be complete")
        if int(assessment.get("replay_exact_protocol_matches", 0) or 0) != requested:
            raise ValueError("SAGE.5j candidate protocol matches must be exact")
        if not bool(assessment.get("protocol_execution_complete", False)) or not bool(
            assessment.get("variant_replication_complete", False)
        ):
            raise ValueError("SAGE.5j candidate protocol must be complete")
        if not bool(assessment.get("ready_for_A32_protocol_result_review", False)):
            raise ValueError("SAGE.5j candidate must be ready for A32.5 review")
        if int(assessment.get("support", 0) or 0) != 0:
            raise ValueError("SAGE.5j candidate support must remain 0")
        if bool(assessment.get("protocol_events_counted_as_scientific_support", False)):
            raise ValueError("SAGE.5j assessment cannot pre-count scientific support")
        if bool(assessment.get("neutral_events_counted_as_refutation", False)):
            raise ValueError("SAGE.5j assessment cannot pre-count refutation")
        if bool(assessment.get("parameterized_interventions_counted_as_distinct_actions", False)):
            raise ValueError("SAGE.5j assessment cannot relabel parameterized controls")
        support = sum(int(row.get("support_events", 0) or 0) for row in rows)
        neutral = sum(int(row.get("neutral_events", 0) or 0) for row in rows)
        contradictions = sum(
            int(row.get("contradiction_events", 0) or 0) for row in rows
        )
        if support != int(assessment.get("raw_support_events", 0) or 0):
            raise ValueError("SAGE.5j raw support count must match experiments")
        if neutral != int(assessment.get("raw_neutral_events", 0) or 0):
            raise ValueError("SAGE.5j raw neutral count must match experiments")
        if contradictions != int(
            assessment.get("raw_contradiction_events", 0) or 0
        ):
            raise ValueError("SAGE.5j raw contradiction count must match experiments")
        total_support += support
        total_neutral += neutral
        total_contradictions += contradictions
        for row in rows:
            if str(row.get("execution_status", "")) != "EXECUTED":
                raise ValueError("every SAGE.5j experiment must be executed")
            if not bool(row.get("live_prefix_replay_exact", False)) or not bool(
                row.get("protocol_exact_match", False)
            ):
                raise ValueError("every SAGE.5j experiment must replay exactly")
            if bool(row.get("protocol_substitution_detected", False)):
                raise ValueError("SAGE.5j experiment substitutions are forbidden")
            if int(row.get("support", 0) or 0) != 0:
                raise ValueError("SAGE.5j experiment support must remain 0")
            if bool(
                row.get("parameterized_control_event_counted_as_scientific_support", False)
            ):
                raise ValueError("SAGE.5j experiment cannot pre-count support")
            if bool(row.get("neutral_event_counted_as_refutation", False)):
                raise ValueError("SAGE.5j experiment cannot pre-count refutation")
            if bool(row.get("parameterized_control_counted_as_distinct_action", False)):
                raise ValueError("SAGE.5j experiment cannot relabel its control")
            if str(row.get("truth_status", "")) != SAGE5J_TRUTH_STATUS:
                raise ValueError("SAGE.5j experiment truth must remain unevaluated")
    if total_support != int(summary.get("raw_support_events", 0) or 0):
        raise ValueError("SAGE.5j summary support count must be exact")
    if total_neutral != int(summary.get("raw_neutral_events", 0) or 0):
        raise ValueError("SAGE.5j summary neutral count must be exact")
    if total_contradictions != int(
        summary.get("raw_contradiction_events", 0) or 0
    ):
        raise ValueError("SAGE.5j summary contradiction count must be exact")


def _scope_limits(
    assessment: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "game_id": str(assessment.get("game_id", "")),
        "candidate_id": str(assessment.get("candidate_id", "")),
        "action": str(assessment.get("action", "")),
        "action_args": assessment.get("action_args"),
        "measurement": _single_value(experiments, "measurement"),
        "budgets": sorted({int(row.get("budget", 0) or 0) for row in experiments}),
        "context_snapshot_hashes": sorted(
            {str(row.get("context_snapshot_hash", "")) for row in experiments}
        ),
        "not_generalized_beyond_game": True,
        "not_generalized_beyond_candidate_scope": True,
        "not_generalized_to_other_actions": True,
    }


def _single_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in rows}
    return next(iter(values)) if len(values) == 1 else None


def record_to_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_unknown_game_parameterized_control_revision_decisions(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH
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


def _load_json(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run A32.5 parameterized-control scientific revision."
    )
    parser.add_argument(
        "--source-sage5j",
        default=DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH,
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH,
    )
    args = parser.parse_args(argv)
    payload = run_a32_unknown_game_parameterized_control_revision_consumer(
        source_sage5j_path=args.source_sage5j
    )
    write_a32_unknown_game_parameterized_control_revision_decisions(payload, args.out)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
