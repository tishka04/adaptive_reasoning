"""A32.3 scientific decision for patch-similarity revision intake."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus

from .patch_similarity_revision_intake import (
    DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH,
)


DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_OUTPUT_PATH = (
    Path("diagnostics") / "a32" / "patch_similarity_revision_decisions.json"
)

CONFIRM_AFTER_SCOPE_LIMITED_REVISION = "CONFIRM_AFTER_SCOPE_LIMITED_REVISION"
REFUTE_AFTER_REVISION = "REFUTE_AFTER_REVISION"
REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS = "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS"
SCOPE_LIMITED_CANDIDATE_ONLY = "SCOPE_LIMITED_CANDIDATE_ONLY"


@dataclass(frozen=True)
class A32PatchSimilarityRevisionDecision:
    """One explicit A32.3 decision over an accepted patch-similarity candidate."""

    queue_item_id: str
    game_id: str
    key: str
    description: str
    decision: str
    recommended_next_step: str
    reasons: Tuple[str, ...]
    scope_limits: Dict[str, Any]
    requested_followup_tests: Tuple[Dict[str, Any], ...]
    evidence_summary: Dict[str, Any]
    input_record: HypothesisRecord
    decision_record: HypothesisRecord
    scientific_review_performed: bool = True
    revision_performed: bool = False
    confirmation_performed: bool = False
    refutation_performed: bool = False
    a33_ready: bool = False
    a33_write_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_item_id": self.queue_item_id,
            "game_id": self.game_id,
            "key": self.key,
            "description": self.description,
            "decision": self.decision,
            "recommended_next_step": self.recommended_next_step,
            "reasons": list(self.reasons),
            "scope_limits": dict(self.scope_limits),
            "requested_followup_tests": [
                dict(row) for row in self.requested_followup_tests
            ],
            "evidence_summary": dict(self.evidence_summary),
            "input_record": record_to_dict(self.input_record),
            "decision_record": record_to_dict(self.decision_record),
            "scientific_review_performed": self.scientific_review_performed,
            "revision_performed": self.revision_performed,
            "confirmation_performed": self.confirmation_performed,
            "refutation_performed": self.refutation_performed,
            "a33_ready": self.a33_ready,
            "a33_write_performed": self.a33_write_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": False,
            "prior_counted_as_proof": False,
            "observation_counted_as_confirmation": False,
            "rule_counted_as_confirmation": False,
            "diagnostic_contradictions_counted_as_refutation": False,
        }


def run_a32_patch_similarity_revision_decision_consumer(
    *,
    intake_path: str | Path = DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(intake_path)
    decisions = build_a32_patch_similarity_revision_decisions(payload)
    return {
        "config": {
            "intake_path": str(intake_path),
            "schema_version": "a32.patch_similarity_revision_decisions.v1",
            "inputs_read": ["A32.2"],
            "decision_scope": "A32_PATCH_SIMILARITY_SCOPE_REVIEW",
            "artifacts_not_modified": ["M3", "A33"],
            "decision_policy": {
                "confirm_requires_multi_context_or_multi_game_scope": True,
                "diagnostic_contradictions_do_not_refute": True,
                "a33_write_performed": False,
            },
        },
        "summary": summarize_patch_similarity_revision_decisions(decisions),
        "revision_decisions": [decision.to_dict() for decision in decisions],
        "input_records": [record_to_dict(decision.input_record) for decision in decisions],
        "decision_records": [
            record_to_dict(decision.decision_record) for decision in decisions
        ],
        "scientific_review_performed": bool(decisions),
        "revision_performed": any(decision.revision_performed for decision in decisions),
        "confirmation_performed": any(
            decision.confirmation_performed for decision in decisions
        ),
        "refutation_performed": any(
            decision.refutation_performed for decision in decisions
        ),
        "a33_ready": any(decision.a33_ready for decision in decisions),
        "a33_write_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
        "rule_counted_as_confirmation": False,
        "diagnostic_contradictions_counted_as_refutation": False,
    }


def build_a32_patch_similarity_revision_decisions(
    intake_payload: Mapping[str, Any],
) -> Tuple[A32PatchSimilarityRevisionDecision, ...]:
    decisions: list[A32PatchSimilarityRevisionDecision] = []
    for item in intake_payload.get("accepted_candidates", []) or []:
        if isinstance(item, Mapping):
            decisions.append(decision_from_accepted_candidate(item))
    return tuple(decisions)


def decision_from_accepted_candidate(
    item: Mapping[str, Any],
) -> A32PatchSimilarityRevisionDecision:
    evidence = dict(item.get("evidence_summary", {}) or {})
    reasons = decision_reasons(item, evidence=evidence)
    decision = decision_label(reasons)
    input_record = unresolved_record_from_candidate(item, evidence)
    decision_record = decision_record_for(
        item,
        evidence=evidence,
        decision=decision,
    )
    return A32PatchSimilarityRevisionDecision(
        queue_item_id=str(item.get("queue_item_id", "")),
        game_id=str(item.get("game_id", "")),
        key=str(item.get("key", "")),
        description=str(item.get("description", "")),
        decision=decision,
        recommended_next_step=recommended_next_step_for(decision),
        reasons=tuple(reasons),
        scope_limits=scope_limits_for(item),
        requested_followup_tests=tuple(requested_followup_tests_for(item, decision)),
        evidence_summary=evidence,
        input_record=input_record,
        decision_record=decision_record,
        revision_performed=decision == CONFIRM_AFTER_SCOPE_LIMITED_REVISION,
        confirmation_performed=decision == CONFIRM_AFTER_SCOPE_LIMITED_REVISION,
        refutation_performed=decision == REFUTE_AFTER_REVISION,
        a33_ready=decision == CONFIRM_AFTER_SCOPE_LIMITED_REVISION,
        a33_write_performed=False,
        wrong_confirmations=0,
    )


def decision_reasons(
    item: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
) -> Tuple[str, ...]:
    reasons: list[str] = []
    if str(item.get("intake_status", "")) != "ACCEPTED_FOR_SCIENTIFIC_REVISION":
        reasons.append("candidate_not_accepted_by_intake")
    if str(item.get("status", "")).upper() != "UNRESOLVED":
        reasons.append("candidate_status_not_unresolved")
    if str(item.get("revision_status", "")).upper() != "CANDIDATE_ONLY":
        reasons.append("candidate_revision_status_not_candidate_only")
    if int(item.get("support", 0) or 0) != 0:
        reasons.append("candidate_support_not_zero")
    if bool(item.get("revision_performed", False)):
        reasons.append("candidate_already_revised")
    if int(item.get("wrong_confirmations", 0) or 0) != 0:
        reasons.append("wrong_confirmations_present")
    if bool(item.get("diagnostic_contradictions_counted_as_refutation", False)):
        reasons.append("diagnostic_contradictions_counted_as_refutation")
    if str(item.get("changed_pixels_role", "")) != "effect_radar_not_success_metric":
        reasons.append("changed_pixels_not_diagnostic")

    success_count = int(evidence.get("successful_args_total_count", 0) or 0)
    failure_count = int(evidence.get("failed_args_count", 0) or 0)
    success_metric_contradictions = int(
        evidence.get("source_success_metric_contradiction_events", 0) or 0
    )
    if success_count < 2:
        reasons.append("insufficient_distinct_successful_args")
    if failure_count < 1:
        reasons.append("missing_negative_case")
    if success_metric_contradictions > 0:
        reasons.append("success_metric_contradictions_present")

    if not fatal_or_refuting_reasons(reasons):
        if is_scope_limited(item):
            reasons.append("mono_game_scope")
            reasons.append("mono_context_scope")
            reasons.append("scope_not_a33_ready")
        elif has_confirmation_scope(evidence):
            reasons.append("scope_limited_confirmation_criteria_satisfied")

    if not reasons:
        reasons.append("request_more_tests_with_scope_limits")
    return tuple(reasons)


def decision_label(reasons: Sequence[str]) -> str:
    reason_set = set(reasons)
    if "success_metric_contradictions_present" in reason_set:
        return REFUTE_AFTER_REVISION
    if any(
        reason
        in reason_set
        for reason in (
            "candidate_not_accepted_by_intake",
            "candidate_status_not_unresolved",
            "candidate_revision_status_not_candidate_only",
            "candidate_support_not_zero",
            "candidate_already_revised",
            "wrong_confirmations_present",
            "diagnostic_contradictions_counted_as_refutation",
            "changed_pixels_not_diagnostic",
        )
    ):
        return REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS
    if "insufficient_distinct_successful_args" in reason_set or "missing_negative_case" in reason_set:
        return REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS
    if "scope_limited_confirmation_criteria_satisfied" in reason_set:
        return CONFIRM_AFTER_SCOPE_LIMITED_REVISION
    if "scope_not_a33_ready" in reason_set:
        return SCOPE_LIMITED_CANDIDATE_ONLY
    return REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS


def recommended_next_step_for(decision: str) -> str:
    if decision == SCOPE_LIMITED_CANDIDATE_ONLY:
        return REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS
    if decision == REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS:
        return REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS
    if decision == REFUTE_AFTER_REVISION:
        return "STOP_OR_REFORMULATE_PATCH_SIMILARITY_HYPOTHESIS"
    return "A33_SCOPE_LIMITED_REGISTRY_REVIEW"


def fatal_or_refuting_reasons(reasons: Sequence[str]) -> bool:
    return any(
        reason
        in {
            "candidate_not_accepted_by_intake",
            "candidate_status_not_unresolved",
            "candidate_revision_status_not_candidate_only",
            "candidate_support_not_zero",
            "candidate_already_revised",
            "wrong_confirmations_present",
            "diagnostic_contradictions_counted_as_refutation",
            "changed_pixels_not_diagnostic",
            "insufficient_distinct_successful_args",
            "missing_negative_case",
            "success_metric_contradictions_present",
        }
        for reason in reasons
    )


def is_scope_limited(item: Mapping[str, Any]) -> bool:
    game_id = str(item.get("game_id", ""))
    context = tuple(str(action) for action in item.get("context_replay", []) or [])
    return bool(game_id) and context == ("ACTION6", "ACTION3", "ACTION4")


def has_confirmation_scope(evidence: Mapping[str, Any]) -> bool:
    return (
        int(evidence.get("distinct_games_validated", 0) or 0) >= 2
        or int(evidence.get("distinct_contexts_validated", 0) or 0) >= 2
    )


def scope_limits_for(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "game_id": str(item.get("game_id", "")),
        "context_replay": list(item.get("context_replay", []) or []),
        "context_replay_args": [
            dict(args)
            for args in item.get("context_replay_args", []) or []
            if isinstance(args, Mapping)
        ],
        "target_action": str(item.get("target_action", "")),
        "candidate_rule_family": str(item.get("candidate_rule_family", "")),
        "successful_args_total": [
            dict(args)
            for args in item.get("successful_args_total", []) or []
            if isinstance(args, Mapping)
        ],
        "failed_args": [
            dict(args)
            for args in item.get("failed_args", []) or []
            if isinstance(args, Mapping)
        ],
        "not_generalized_beyond_context": True,
        "not_a33_ready": True,
    }


def requested_followup_tests_for(
    item: Mapping[str, Any],
    decision: str,
) -> Tuple[Dict[str, Any], ...]:
    if decision not in {
        SCOPE_LIMITED_CANDIDATE_ONLY,
        REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS,
    }:
        return ()
    base_context = list(item.get("context_replay", []) or [])
    base_args = [
        dict(args)
        for args in item.get("context_replay_args", []) or []
        if isinstance(args, Mapping)
    ]
    return (
        {
            "followup_family": "outside_known_y12_region_probe",
            "purpose": "test whether patch-similarity holds outside the current success line/region",
            "context_replay": base_context,
            "context_replay_args": base_args,
            "target_action": str(item.get("target_action", "")),
            "exclude_known_args": [
                dict(args)
                for args in item.get("successful_args_total", []) or []
                if isinstance(args, Mapping)
            ],
            "success_metrics": list(item.get("success_metrics", []) or []),
            "diagnostic_metrics": list(item.get("diagnostic_metrics", []) or []),
            "status": "REQUESTED",
            "support": 0,
        },
        {
            "followup_family": "alternate_repositioning_context_probe",
            "purpose": "test whether ACTION4 creates patch-similar ACTION6 affordances in another replay context",
            "context_replay": base_context,
            "target_action": str(item.get("target_action", "")),
            "success_metrics": list(item.get("success_metrics", []) or []),
            "diagnostic_metrics": list(item.get("diagnostic_metrics", []) or []),
            "status": "REQUESTED",
            "support": 0,
        },
    )


def unresolved_record_from_candidate(
    item: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> HypothesisRecord:
    return HypothesisRecord(
        key=str(item.get("key", "")),
        description=str(item.get("description", "")),
        status=HypothesisStatus.UNRESOLVED,
        support=0,
        contradictions=0,
        experiments_spent=int(evidence.get("controlled_experiments_run", 0) or 0),
    )


def decision_record_for(
    item: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    decision: str,
) -> HypothesisRecord:
    if decision == CONFIRM_AFTER_SCOPE_LIMITED_REVISION:
        return HypothesisRecord(
            key=str(item.get("key", "")),
            description=str(item.get("description", "")),
            status=HypothesisStatus.CONFIRMED,
            support=int(evidence.get("source_success_metric_support_events", 0) or 0),
            contradictions=0,
            experiments_spent=int(evidence.get("controlled_experiments_run", 0) or 0),
        )
    if decision == REFUTE_AFTER_REVISION:
        return HypothesisRecord(
            key=str(item.get("key", "")),
            description=str(item.get("description", "")),
            status=HypothesisStatus.REFUTED,
            support=0,
            contradictions=int(
                evidence.get("source_success_metric_contradiction_events", 0) or 0
            ),
            experiments_spent=int(evidence.get("controlled_experiments_run", 0) or 0),
        )
    return unresolved_record_from_candidate(item, evidence)


def summarize_patch_similarity_revision_decisions(
    decisions: Sequence[A32PatchSimilarityRevisionDecision],
) -> Dict[str, Any]:
    return {
        "intake_candidates_consumed": len(decisions),
        "confirm_after_scope_limited_revision": count_decisions(
            decisions,
            CONFIRM_AFTER_SCOPE_LIMITED_REVISION,
        ),
        "refute_after_revision": count_decisions(decisions, REFUTE_AFTER_REVISION),
        "request_more_tests_with_scope_limits": count_decisions(
            decisions,
            REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS,
        ),
        "scope_limited_candidate_only": count_decisions(
            decisions,
            SCOPE_LIMITED_CANDIDATE_ONLY,
        ),
        "recommended_more_tests": len(
            [
                decision
                for decision in decisions
                if decision.recommended_next_step
                == REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS
            ]
        ),
        "a33_ready_candidates": len(
            [decision for decision in decisions if decision.a33_ready]
        ),
        "a33_write_performed": False,
        "scientific_review_performed": bool(decisions),
        "revision_performed": any(decision.revision_performed for decision in decisions),
        "confirmation_performed": any(
            decision.confirmation_performed for decision in decisions
        ),
        "refutation_performed": any(
            decision.refutation_performed for decision in decisions
        ),
        "input_records_unresolved": len(
            [
                decision
                for decision in decisions
                if decision.input_record.status == HypothesisStatus.UNRESOLVED
            ]
        ),
        "decision_records_confirmed": len(
            [
                decision
                for decision in decisions
                if decision.decision_record.status == HypothesisStatus.CONFIRMED
            ]
        ),
        "decision_records_refuted": len(
            [
                decision
                for decision in decisions
                if decision.decision_record.status == HypothesisStatus.REFUTED
            ]
        ),
        "decision_records_unresolved": len(
            [
                decision
                for decision in decisions
                if decision.decision_record.status == HypothesisStatus.UNRESOLVED
            ]
        ),
        "source_success_metric_support_events": sum(
            int(
                decision.evidence_summary.get(
                    "source_success_metric_support_events",
                    0,
                )
                or 0
            )
            for decision in decisions
        ),
        "source_success_metric_contradiction_events": sum(
            int(
                decision.evidence_summary.get(
                    "source_success_metric_contradiction_events",
                    0,
                )
                or 0
            )
            for decision in decisions
        ),
        "diagnostic_contradictions_counted_as_refutation": False,
        "m3_artifacts_mutated": False,
        "wrong_confirmations": 0,
    }


def count_decisions(
    decisions: Sequence[A32PatchSimilarityRevisionDecision],
    label: str,
) -> int:
    return len([decision for decision in decisions if decision.decision == label])


def record_to_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_patch_similarity_revision_decisions(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A32.3 patch-similarity scientific revision decision.",
    )
    parser.add_argument(
        "--intake",
        type=Path,
        default=DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_patch_similarity_revision_decision_consumer(
        intake_path=args.intake,
    )
    write_a32_patch_similarity_revision_decisions(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "decision_scope": "A32_PATCH_SIMILARITY_SCOPE_REVIEW",
                "a33_write_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
