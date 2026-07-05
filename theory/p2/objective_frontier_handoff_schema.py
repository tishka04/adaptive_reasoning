"""P2.6 objective-frontier request schema for downstream hypothesis work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.objective_frontier_handoff_validator import (
    DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH,
)
from theory.p2.policy_frontier_records import TRUTH_STATUS
from theory.p2.terminal_outcome_frontier import (
    LOCAL_PRODUCTIVE_TERMINAL_FAILED,
    OBJECTIVE_ALIGNMENT_FRONTIER,
)


DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "bp35_objective_frontier_handoff_requests.json"
)
HANDOFF_TYPE = "OBJECTIVE_ALIGNMENT_FRONTIER_REQUEST"
HANDOFF_TARGET = "M2_OR_M3"


def run_objective_frontier_handoff_schema(
    *,
    objective_frontier_handoff_validation_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(objective_frontier_handoff_validation_path)
    _validate_source_payload(payload)
    accepted = [
        dict(row)
        for row in payload.get("accepted_objective_reviews", []) or []
        if isinstance(row, Mapping)
    ]
    requests = [
        build_objective_frontier_request(
            review,
            request_index=index,
            source_validation_path=objective_frontier_handoff_validation_path,
        )
        for index, review in enumerate(accepted, start=1)
    ]
    for request in requests:
        validate_objective_frontier_request(request)

    return {
        "config": {
            "schema_version": "p2.objective_frontier_handoff_requests.v1",
            "objective_frontier_handoff_validation_path": str(
                objective_frontier_handoff_validation_path
            ),
            "inputs_read": ["P2.5"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "handoff_target": HANDOFF_TARGET,
            "handoff_type": HANDOFF_TYPE,
            "direct_downstream_write_performed": False,
        },
        "objective_frontier_requests": requests,
        "summary": {
            "source_objective_reviews_accepted": len(accepted),
            "objective_frontier_requests": len(requests),
            "target": HANDOFF_TARGET if requests else "",
            "handoff_type": HANDOFF_TYPE if requests else "",
            "ready_for_m2_or_m3_objective_branch": bool(requests),
            "ready_for_saturation_handoff": False,
            "saturation_handoff_requests": 0,
            "a33_ready": False,
            "a40_write_performed": False,
            "m2_write_performed": False,
            "m3_write_performed": False,
            "a32_write_performed": False,
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
        "objective_frontier_request_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a33_ready": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
    }


def build_objective_frontier_request(
    review: Mapping[str, Any],
    *,
    request_index: int,
    source_validation_path: str | Path,
) -> Dict[str, Any]:
    game_id = str(review.get("game_id", ""))
    request_id = (
        f"p2_o1::{game_id or 'unknown_game'}::objective_alignment::"
        f"{int(request_index):03d}"
    )
    return {
        "request_id": request_id,
        "source_validation_path": str(source_validation_path),
        "source_frontier_id": str(review.get("frontier_id", "")),
        "handoff_type": HANDOFF_TYPE,
        "target": HANDOFF_TARGET,
        "target_modules": ["M2.O1", "M3.O1"],
        "game_id": game_id,
        "frontier_type": str(review.get("frontier_type", "")),
        "frontier_reason": str(review.get("frontier_reason", "")),
        "objective_review_accepted": bool(
            review.get("objective_review_accepted", False)
        ),
        "terminal_runs": int(review.get("terminal_runs", 0) or 0),
        "terminal_budgets": list(review.get("terminal_budgets", []) or []),
        "observed_pattern": {
            "local_affordance_productive": True,
            "terminal_objective_failed": True,
            "movement_refresh_triggers": 0,
            "saturation_handoff_ready": False,
            "known_target_action": "ACTION6",
            "known_failure_state": "GAME_OVER",
            "known_levels_completed": 0,
        },
        "scientific_questions": [
            "When should ACTION6 be stopped despite useful local effects?",
            "Which signals predict GAME_OVER during repeated ACTION6 exploitation?",
            "Which subgoal should replace repeated patch-similar ACTION6 exploitation?",
            "Which stop-switch condition can be tested by M3 without confirming it?",
        ],
        "desired_hypothesis_families": [
            "stop_switch_criterion",
            "terminal_risk_predictor",
            "subgoal_switch_after_local_affordance",
            "global_objective_alignment_metric",
        ],
        "requested_experiment_styles": [
            "compare_continue_action6_vs_stop_or_switch",
            "prefix_length_terminal_risk_probe",
            "alternative_subgoal_after_productive_action6_probe",
            "global_outcome_metric_vs_local_effect_metric_probe",
        ],
        "ready_for_objective_hypothesis_generation": True,
        "ready_for_m2_or_m3_objective_branch": True,
        "ready_for_saturation_handoff": False,
        "a33_ready": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "objective_frontier_request_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }


def validate_objective_frontier_request(request: Mapping[str, Any]) -> None:
    if not request.get("request_id"):
        raise ValueError("request_id is required")
    if str(request.get("handoff_type", "")) != HANDOFF_TYPE:
        raise ValueError("objective handoff request has invalid handoff_type")
    if str(request.get("target", "")) != HANDOFF_TARGET:
        raise ValueError("objective handoff request target must be M2_OR_M3")
    if str(request.get("frontier_type", "")) != OBJECTIVE_ALIGNMENT_FRONTIER:
        raise ValueError("request frontier_type must be OBJECTIVE_ALIGNMENT_FRONTIER")
    if str(request.get("frontier_reason", "")) != LOCAL_PRODUCTIVE_TERMINAL_FAILED:
        raise ValueError("request frontier_reason is not the objective terminal reason")
    if not bool(request.get("objective_review_accepted", False)):
        raise ValueError("objective review must be accepted before request creation")
    if bool(request.get("ready_for_saturation_handoff", False)):
        raise ValueError("objective request cannot be ready for saturation handoff")
    if bool(request.get("a33_ready", False)):
        raise ValueError("objective request cannot be A33-ready")
    if str(request.get("status", "")) != "UNRESOLVED":
        raise ValueError("objective request must remain UNRESOLVED")
    if int(request.get("support", 0) or 0) != 0:
        raise ValueError("objective request support must remain 0")
    if str(request.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("objective request must remain candidate-only")
    if str(request.get("truth_status", "")) != TRUTH_STATUS:
        raise ValueError("objective request truth_status must remain P2-local")
    if bool(request.get("revision_performed", False)):
        raise ValueError("objective request revision_performed must be false")
    if int(request.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("objective request wrong_confirmations must remain 0")
    if bool(request.get("objective_frontier_request_counted_as_confirmation", False)):
        raise ValueError("objective request cannot count as confirmation")
    if bool(request.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot count as scientific verdict")


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(summary.get("revision_performed", False)):
        raise ValueError("source summary revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source summary wrong_confirmations must remain 0")
    if int(summary.get("saturation_handoffs_accepted", 0) or 0) != 0:
        raise ValueError("P2.6 cannot consume saturation handoffs")
    if bool(summary.get("ready_for_p2_4_saturation_handoff", False)):
        raise ValueError("P2.6 source cannot be ready for saturation handoff")
    for key in (
        "a40_write_performed",
        "m2_write_performed",
        "m3_write_performed",
        "a32_write_performed",
    ):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"source must not have {key}")
    if bool(payload.get("objective_handoff_validation_counted_as_confirmation", False)):
        raise ValueError("objective handoff validation cannot be confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")


def write_objective_frontier_handoff_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_REQUESTS_OUTPUT_PATH
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
        description="Run P2.6 objective-frontier handoff schema.",
    )
    parser.add_argument(
        "--objective-frontier-validation",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_frontier_handoff_schema(
        objective_frontier_handoff_validation_path=args.objective_frontier_validation,
    )
    write_objective_frontier_handoff_requests(payload, args.out)
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
