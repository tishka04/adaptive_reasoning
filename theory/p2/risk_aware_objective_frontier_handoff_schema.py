"""P2.G5 risk-aware objective-completion frontier validator/request schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.policy_frontier_records import TRUTH_STATUS
from theory.p2.risk_aware_post_stop_frontier_records import (
    DEFAULT_P2_RISK_AWARE_POST_STOP_FRONTIER_RECORDS_OUTPUT_PATH,
    OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION,
    RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER,
    RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION,
)


DEFAULT_P2_RISK_AWARE_OBJECTIVE_HANDOFF_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "risk_aware_objective_frontier_handoff_requests.json"
)
HANDOFF_TYPE = "RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST"
HANDOFF_TARGET = "M2_OR_M3"
TARGET_MODULES = ["M2.G2"]
SUGGESTED_FOLLOWUP_MODULES = ["M3.G5"]


def run_risk_aware_objective_frontier_handoff_schema(
    *,
    risk_aware_frontier_records_path: str
    | Path = DEFAULT_P2_RISK_AWARE_POST_STOP_FRONTIER_RECORDS_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(risk_aware_frontier_records_path)
    _validate_source_payload(payload)
    records = [
        dict(record)
        for record in payload.get("risk_aware_post_stop_frontier_records", []) or []
        if isinstance(record, Mapping)
    ]
    evaluations = [
        evaluate_risk_aware_objective_frontier_for_review(record)
        for record in records
    ]
    accepted = [
        row for row in evaluations if row["risk_aware_objective_review_accepted"]
    ]
    rejected = [
        row for row in evaluations if not row["risk_aware_objective_review_accepted"]
    ]
    requests = [
        build_risk_aware_objective_request(
            review,
            request_index=index,
            source_records_path=risk_aware_frontier_records_path,
        )
        for index, review in enumerate(accepted, start=1)
    ]
    for request in requests:
        validate_risk_aware_objective_request(request)

    return {
        "config": {
            "schema_version": "p2.risk_aware_objective_frontier_handoff_requests.v1",
            "risk_aware_frontier_records_path": str(risk_aware_frontier_records_path),
            "inputs_read": ["P2.G4"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "validation_mode": "risk_aware_objective_frontier_review_no_write",
            "handoff_target": HANDOFF_TARGET,
            "handoff_type": HANDOFF_TYPE,
            "target_modules": TARGET_MODULES,
            "suggested_followup_modules": SUGGESTED_FOLLOWUP_MODULES,
            "direct_downstream_write_performed": False,
            "ready_for_direct_downstream_write": False,
        },
        "frontier_evaluations": evaluations,
        "accepted_risk_aware_objective_reviews": accepted,
        "rejected_risk_aware_objective_reviews": rejected,
        "risk_aware_objective_handoff_requests": requests,
        "summary": {
            "risk_aware_frontiers_seen": len(evaluations),
            "risk_aware_objective_reviews_accepted": len(accepted),
            "risk_aware_objective_reviews_rejected": len(rejected),
            "risk_aware_objective_handoff_requests": len(requests),
            "handoff_type": HANDOFF_TYPE if requests else "",
            "target": HANDOFF_TARGET if requests else "",
            "target_modules": TARGET_MODULES if requests else [],
            "suggested_followup_modules": (
                SUGGESTED_FOLLOWUP_MODULES if requests else []
            ),
            "ready_for_m2_or_m3_risk_aware_objective_branch": bool(requests),
            "ready_for_direct_downstream_write": False,
            "a33_ready": False,
            "rejection_reasons": sorted(
                {
                    reason
                    for row in rejected
                    for reason in row.get("rejection_reasons", []) or []
                }
            ),
            "a40_write_performed": False,
            "m2_write_performed": False,
            "m3_write_performed": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "handoff_request_counted_as_confirmation": False,
        "frontier_validation_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "ready_for_direct_downstream_write": False,
        "a33_ready": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def evaluate_risk_aware_objective_frontier_for_review(
    frontier: Mapping[str, Any],
) -> Dict[str, Any]:
    reasons = []
    evidence = dict(frontier.get("evidence", {}) or {})
    if str(frontier.get("frontier_type", "")) != (
        RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER
    ):
        reasons.append("NOT_RISK_AWARE_POST_STOP_FRONTIER")
    if str(frontier.get("frontier_reason", "")) != (
        RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION
    ):
        reasons.append("UNEXPECTED_RISK_AWARE_FRONTIER_REASON")
    if str(frontier.get("blocked_capability", "")) != (
        OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
    ):
        reasons.append("UNEXPECTED_BLOCKED_CAPABILITY")
    if float(evidence.get("terminal_rate", 1.0) or 0.0) != 0.0:
        reasons.append("TERMINAL_SAFETY_NOT_OBSERVED")
    if float(evidence.get("mean_delta_vs_hold", 0.0) or 0.0) <= 0.0:
        reasons.append("OOS_UTILITY_VS_HOLD_NOT_OBSERVED")
    if not bool(evidence.get("improvement_over_action6_only", False)):
        reasons.append("UTILITY_VS_ACTION6_NOT_OBSERVED")
    if int(evidence.get("static_extension_terminal_options", 0) or 0) <= 0:
        reasons.append("STATIC_EXTENSION_RISK_NOT_REPRODUCED")
    if int(evidence.get("unsafe_extension_options_avoided", 0) or 0) <= 0:
        reasons.append("RISK_AWARE_SELECTION_NOT_OBSERVED")
    if bool(evidence.get("objective_completion_signal", False)):
        reasons.append("OBJECTIVE_COMPLETION_ALREADY_OBSERVED")
    if int(evidence.get("objective_completion_runs", 0) or 0) != 0:
        reasons.append("OBJECTIVE_COMPLETION_RUNS_NOT_ZERO")
    if bool(evidence.get("adapter_relearned", True)):
        reasons.append("ADAPTER_NOT_FROZEN")
    if not bool(evidence.get("source_cells_rerun", False)):
        reasons.append("SOURCE_CELLS_NOT_RERUN")
    if bool(evidence.get("selection_uses_risk_targeted_candidate_outcomes", True)):
        reasons.append("SELECTION_USES_RISK_TARGETED_OUTCOMES")
    if not bool(frontier.get("ready_for_risk_aware_objective_frontier_review", False)):
        reasons.append("NOT_READY_FOR_RISK_AWARE_OBJECTIVE_REVIEW")
    if bool(frontier.get("ready_for_m2_or_m3", False)):
        reasons.append("INPUT_ALREADY_READY_FOR_M2_OR_M3")
    if bool(frontier.get("ready_for_direct_downstream_write", False)):
        reasons.append("DIRECT_DOWNSTREAM_WRITE_NOT_ALLOWED")
    if bool(frontier.get("ready_for_saturation_handoff", False)):
        reasons.append("SATURATION_HANDOFF_NOT_ALLOWED")
    if bool(frontier.get("a33_ready", False)):
        reasons.append("A33_READY_NOT_ALLOWED")
    if str(frontier.get("status", "")) in {"CONFIRMED", "REFUTED"}:
        reasons.append("CONFIRMED_OR_REFUTED_NOT_ALLOWED")
    if str(frontier.get("status", "")) != "UNRESOLVED":
        reasons.append("STATUS_NOT_UNRESOLVED")
    if int(frontier.get("support", 0) or 0) != 0:
        reasons.append("SUPPORT_MUST_REMAIN_ZERO")
    if str(frontier.get("revision_status", "")) != "CANDIDATE_ONLY":
        reasons.append("REVISION_STATUS_NOT_CANDIDATE_ONLY")
    if str(frontier.get("truth_status", "")) != TRUTH_STATUS:
        reasons.append("TRUTH_STATUS_NOT_EVALUATED_BY_P2_REQUIRED")
    if bool(frontier.get("revision_performed", False)):
        reasons.append("REVISION_PERFORMED_NOT_ALLOWED")
    if int(frontier.get("wrong_confirmations", 0) or 0) != 0:
        reasons.append("WRONG_CONFIRMATIONS_NOT_ZERO")
    if bool(frontier.get("risk_aware_frontier_counted_as_confirmation", False)):
        reasons.append("FRONTIER_COUNTED_AS_CONFIRMATION")
    if bool(frontier.get("policy_result_counted_as_scientific_verdict", False)):
        reasons.append("POLICY_RESULT_COUNTED_AS_VERDICT")
    if bool(frontier.get("risk_aware_policy_counted_as_objective_solution", False)):
        reasons.append("POLICY_COUNTED_AS_OBJECTIVE_SOLUTION")
    if not list(frontier.get("blocked_capability_hypotheses", []) or []):
        reasons.append("MISSING_BLOCKED_CAPABILITY_HYPOTHESES")
    if not list(frontier.get("desired_hypothesis_families", []) or []):
        reasons.append("MISSING_DESIRED_HYPOTHESIS_FAMILIES")
    if not list(frontier.get("requested_experiment_styles", []) or []):
        reasons.append("MISSING_REQUESTED_EXPERIMENT_STYLES")
    if not list(frontier.get("scientific_questions", []) or []):
        reasons.append("MISSING_SCIENTIFIC_QUESTIONS")

    reasons = sorted(set(reasons))
    accepted = not reasons
    return {
        "frontier_id": str(frontier.get("frontier_id", "")),
        "game_id": str(frontier.get("game_id", "")),
        "source": str(frontier.get("source", "")),
        "frontier_type": str(frontier.get("frontier_type", "")),
        "frontier_reason": str(frontier.get("frontier_reason", "")),
        "blocked_capability": str(frontier.get("blocked_capability", "")),
        "blocked_capability_hypotheses": list(
            frontier.get("blocked_capability_hypotheses", []) or []
        ),
        "desired_hypothesis_families": list(
            frontier.get("desired_hypothesis_families", []) or []
        ),
        "requested_experiment_styles": list(
            frontier.get("requested_experiment_styles", []) or []
        ),
        "scientific_questions": list(frontier.get("scientific_questions", []) or []),
        "evidence": dict(evidence),
        "risk_region_snapshot": dict(frontier.get("risk_region_snapshot", {}) or {}),
        "policy_aggregate_snapshot": dict(
            frontier.get("policy_aggregate_snapshot", {}) or {}
        ),
        "ready_for_risk_aware_objective_frontier_review": bool(
            frontier.get("ready_for_risk_aware_objective_frontier_review", False)
        ),
        "ready_for_m2_or_m3": False,
        "risk_aware_objective_review_accepted": accepted,
        "risk_aware_objective_review_target": (
            "risk_aware_objective_completion_frontier_review" if accepted else ""
        ),
        "rejection_reasons": reasons,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_risk_aware_objective_request(
    review: Mapping[str, Any],
    *,
    request_index: int,
    source_records_path: str | Path,
) -> Dict[str, Any]:
    game_id = str(review.get("game_id", ""))
    request_id = (
        f"p2g5::{game_id or 'unknown_game'}::risk_aware_objective_completion::"
        f"{int(request_index):03d}"
    )
    return {
        "request_id": request_id,
        "source_records_path": str(source_records_path),
        "source_frontier_id": str(review.get("frontier_id", "")),
        "handoff_type": HANDOFF_TYPE,
        "target": HANDOFF_TARGET,
        "target_modules": TARGET_MODULES,
        "suggested_followup_modules": SUGGESTED_FOLLOWUP_MODULES,
        "game_id": game_id,
        "frontier_type": str(review.get("frontier_type", "")),
        "frontier_reason": str(review.get("frontier_reason", "")),
        "blocked_capability": str(review.get("blocked_capability", "")),
        "risk_aware_objective_review_accepted": bool(
            review.get("risk_aware_objective_review_accepted", False)
        ),
        "candidate_problem_statement": (
            "risk_aware_safe_post_stop_progress_does_not_complete_objective"
        ),
        "blocked_capability_hypotheses": list(
            review.get("blocked_capability_hypotheses", []) or []
        ),
        "requested_hypothesis_families": list(
            review.get("desired_hypothesis_families", []) or []
        ),
        "requested_experiment_styles": list(
            review.get("requested_experiment_styles", []) or []
        ),
        "scientific_questions": list(review.get("scientific_questions", []) or []),
        "evidence_summary": _request_evidence_summary(review),
        "suggested_initial_experiment_matrix": {
            "base_state_family": "risk_aware_terminal_safe_post_stop_conversion_state",
            "source_policy_options": [
                "hold_or_stop_state",
                "ACTION6",
                "ACTION6,ACTION3",
                "ACTION6,ACTION4",
                "contextual_post_stop_conversion_policy",
            ],
            "readiness_feature_candidates": [
                "sampling_family",
                "terminal_horizon_remaining",
                "terminal_horizon_band",
                "hold_baseline_terminal_adjusted_progress",
                "hold_baseline_band",
                "relation_delta_after_stop",
                "new_relation_states",
                "changed_pixels",
                "global_configuration_signature",
            ],
            "post_conversion_commit_action_candidates": [
                "ACTION1",
                "ACTION2",
                "ACTION3",
                "ACTION4",
                "ACTION5",
                "ACTION6",
            ],
            "controls": [
                "hold_or_stop_state",
                "ACTION6_only",
                "frozen_contextual_selector",
                "always_extension_static_policy",
            ],
            "success_metrics": [
                "objective_completion_signal",
                "levels_completed_after_rollout",
            ],
            "safety_metrics": [
                "terminal_reentry_rate",
                "terminal_adjusted_progress_after_stop",
            ],
            "discriminator_metrics": [
                "proxy_progress_without_completion",
                "objective_readiness_precision",
                "commit_action_delta_vs_selector",
            ],
        },
        "ready_for_risk_aware_objective_hypothesis_generation": True,
        "ready_for_m2_or_m3_risk_aware_objective_branch": True,
        "ready_for_direct_downstream_write": False,
        "a33_ready": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "handoff_request_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }


def validate_risk_aware_objective_request(request: Mapping[str, Any]) -> None:
    if not request.get("request_id"):
        raise ValueError("request_id is required")
    if str(request.get("handoff_type", "")) != HANDOFF_TYPE:
        raise ValueError("risk-aware objective request has invalid handoff_type")
    if str(request.get("target", "")) != HANDOFF_TARGET:
        raise ValueError("risk-aware objective request target must be M2_OR_M3")
    if list(request.get("target_modules", []) or []) != TARGET_MODULES:
        raise ValueError("risk-aware objective request target_modules are invalid")
    if str(request.get("frontier_type", "")) != (
        RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER
    ):
        raise ValueError("request frontier_type must be risk-aware no-completion")
    if str(request.get("frontier_reason", "")) != (
        RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION
    ):
        raise ValueError("request frontier_reason must be risk-aware utility gap")
    if str(request.get("blocked_capability", "")) != (
        OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
    ):
        raise ValueError("request blocked_capability is not risk-aware completion")
    if not bool(request.get("risk_aware_objective_review_accepted", False)):
        raise ValueError("risk-aware objective review must be accepted first")
    if not list(request.get("blocked_capability_hypotheses", []) or []):
        raise ValueError("risk-aware objective request needs blocked hypotheses")
    if not list(request.get("requested_hypothesis_families", []) or []):
        raise ValueError("risk-aware objective request needs hypothesis families")
    if not list(request.get("requested_experiment_styles", []) or []):
        raise ValueError("risk-aware objective request needs experiment styles")
    if not list(request.get("scientific_questions", []) or []):
        raise ValueError("risk-aware objective request needs scientific questions")
    if not dict(request.get("suggested_initial_experiment_matrix", {}) or {}):
        raise ValueError("risk-aware objective request needs experiment matrix")
    if bool(request.get("ready_for_direct_downstream_write", False)):
        raise ValueError("P2.G5 cannot be ready for direct downstream write")
    if bool(request.get("a33_ready", False)):
        raise ValueError("risk-aware objective request cannot be A33-ready")
    if str(request.get("status", "")) != "UNRESOLVED":
        raise ValueError("risk-aware objective request must remain UNRESOLVED")
    if int(request.get("support", 0) or 0) != 0:
        raise ValueError("risk-aware objective request support must remain 0")
    if str(request.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("risk-aware objective request must remain candidate-only")
    if str(request.get("truth_status", "")) != TRUTH_STATUS:
        raise ValueError("risk-aware objective request truth_status must remain P2-local")
    if bool(request.get("revision_performed", False)):
        raise ValueError("risk-aware objective request revision_performed must be false")
    if int(request.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("risk-aware objective request wrong_confirmations must remain 0")
    if bool(request.get("handoff_request_counted_as_confirmation", False)):
        raise ValueError("risk-aware objective request cannot count as confirmation")
    if bool(request.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot count as scientific verdict")


def _request_evidence_summary(review: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = dict(review.get("evidence", {}) or {})
    risk = dict(review.get("risk_region_snapshot", {}) or {})
    aggregates = dict(review.get("policy_aggregate_snapshot", {}) or {})
    return {
        "accepted_risk_targeted_safe_stops": int(
            evidence.get("accepted_risk_targeted_safe_stops", 0) or 0
        ),
        "contextual_terminal_rate": float(evidence.get("terminal_rate", 0.0) or 0.0),
        "contextual_mean_terminal_adjusted_progress": float(
            evidence.get("mean_terminal_adjusted_progress", 0.0) or 0.0
        ),
        "mean_delta_vs_hold": float(evidence.get("mean_delta_vs_hold", 0.0) or 0.0),
        "mean_delta_vs_action6_only": float(
            evidence.get("mean_delta_vs_action6_only", 0.0) or 0.0
        ),
        "static_extension_terminal_options": int(
            evidence.get("static_extension_terminal_options", 0) or 0
        ),
        "static_extension_terminal_safe_stops": int(
            evidence.get("static_extension_terminal_safe_stops", 0) or 0
        ),
        "unsafe_extension_options_avoided": int(
            evidence.get("unsafe_extension_options_avoided", 0) or 0
        ),
        "objective_completion_signal": bool(
            evidence.get("objective_completion_signal", False)
        ),
        "objective_completion_runs": int(
            evidence.get("objective_completion_runs", 0) or 0
        ),
        "action6_action3_terminal_rate": float(
            risk.get("action6_action3_terminal_rate", 0.0) or 0.0
        ),
        "action6_action4_terminal_rate": float(
            risk.get("action6_action4_terminal_rate", 0.0) or 0.0
        ),
        "policy_aggregate_snapshot": aggregates,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(payload.get("revision_performed", False)) or bool(
        summary.get("revision_performed", False)
    ):
        raise ValueError("source revision_performed must be false")
    if int(payload.get("wrong_confirmations", summary.get("wrong_confirmations", 0)) or 0) != 0:
        raise ValueError("source wrong_confirmations must remain 0")
    if bool(summary.get("ready_for_m2_or_m3", False)):
        raise ValueError("P2.G5 source cannot already be ready for M2/M3")
    if not bool(summary.get("ready_for_risk_aware_objective_frontier_review", False)):
        raise ValueError("P2.G5 requires a risk-aware objective review-ready source")
    if bool(payload.get("ready_for_direct_downstream_write", False)) or bool(
        summary.get("ready_for_direct_downstream_write", False)
    ):
        raise ValueError("P2.G5 source cannot request direct downstream write")
    if bool(payload.get("a33_ready", False)) or bool(summary.get("a33_ready", False)):
        raise ValueError("P2.G5 source cannot be A33-ready")
    for key in (
        "a40_write_performed",
        "m2_write_performed",
        "m3_write_performed",
        "a32_write_performed",
        "a33_write_performed",
    ):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"source must not have {key}")
    if bool(payload.get("risk_aware_frontier_counted_as_confirmation", False)):
        raise ValueError("risk-aware frontier cannot be confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")


def write_risk_aware_objective_handoff_requests(
    payload: Mapping[str, Any],
    output_path: str
    | Path = DEFAULT_P2_RISK_AWARE_OBJECTIVE_HANDOFF_REQUESTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P2.G5 risk-aware objective frontier handoff schema.",
    )
    parser.add_argument(
        "--risk-aware-frontier-records",
        type=Path,
        default=DEFAULT_P2_RISK_AWARE_POST_STOP_FRONTIER_RECORDS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_RISK_AWARE_OBJECTIVE_HANDOFF_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_risk_aware_objective_frontier_handoff_schema(
        risk_aware_frontier_records_path=args.risk_aware_frontier_records,
    )
    write_risk_aware_objective_handoff_requests(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
