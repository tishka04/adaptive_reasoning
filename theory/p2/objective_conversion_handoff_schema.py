"""P2.G3 objective-conversion request schema for downstream hypothesis work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.objective_conversion_frontier_records import (
    OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
    OBJECTIVE_CONVERSION_FRONTIER,
    TERMINAL_SAFE_BUT_PASSIVE,
)
from theory.p2.objective_conversion_frontier_validator import (
    DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_VALIDATION_OUTPUT_PATH,
)
from theory.p2.policy_frontier_records import TRUTH_STATUS


DEFAULT_P2_OBJECTIVE_CONVERSION_HANDOFF_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "objective_conversion_handoff_requests.json"
)
HANDOFF_TYPE = "OBJECTIVE_CONVERSION_FRONTIER_REQUEST"
HANDOFF_TARGET = "M2_OR_M3"
TARGET_MODULES = ["M2.G1", "M3.G1"]


def run_objective_conversion_handoff_schema(
    *,
    objective_conversion_frontier_validation_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_VALIDATION_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(objective_conversion_frontier_validation_path)
    _validate_source_payload(payload)
    accepted = [
        dict(row)
        for row in payload.get("accepted_objective_conversion_reviews", []) or []
        if isinstance(row, Mapping)
    ]
    requests = [
        build_objective_conversion_request(
            review,
            request_index=index,
            source_validation_path=objective_conversion_frontier_validation_path,
        )
        for index, review in enumerate(accepted, start=1)
    ]
    for request in requests:
        validate_objective_conversion_request(request)

    return {
        "config": {
            "schema_version": "p2.objective_conversion_handoff_requests.v1",
            "objective_conversion_frontier_validation_path": str(
                objective_conversion_frontier_validation_path
            ),
            "inputs_read": ["P2.G2"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "handoff_target": HANDOFF_TARGET,
            "handoff_type": HANDOFF_TYPE,
            "target_modules": TARGET_MODULES,
            "direct_downstream_write_performed": False,
            "ready_for_direct_downstream_write": False,
        },
        "objective_conversion_handoff_requests": requests,
        "summary": {
            "source_objective_conversion_reviews_accepted": len(accepted),
            "objective_conversion_handoff_requests": len(requests),
            "handoff_type": HANDOFF_TYPE if requests else "",
            "target": HANDOFF_TARGET if requests else "",
            "target_modules": TARGET_MODULES if requests else [],
            "ready_for_m2_or_m3_objective_conversion_branch": bool(requests),
            "ready_for_direct_downstream_write": False,
            "a33_ready": False,
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
        "policy_result_counted_as_scientific_verdict": False,
        "ready_for_direct_downstream_write": False,
        "a33_ready": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def build_objective_conversion_request(
    review: Mapping[str, Any],
    *,
    request_index: int,
    source_validation_path: str | Path,
) -> Dict[str, Any]:
    game_id = str(review.get("game_id", ""))
    request_id = (
        f"p2g3::{game_id or 'unknown_game'}::objective_conversion::"
        f"{int(request_index):03d}"
    )
    hypothesis_families = list(review.get("desired_hypothesis_families", []) or [])
    experiment_styles = list(review.get("requested_experiment_styles", []) or [])
    questions = list(review.get("scientific_questions", []) or [])
    return {
        "request_id": request_id,
        "source_validation_path": str(source_validation_path),
        "source_frontier_id": str(review.get("frontier_id", "")),
        "handoff_type": HANDOFF_TYPE,
        "target": HANDOFF_TARGET,
        "target_modules": TARGET_MODULES,
        "game_id": game_id,
        "frontier_type": str(review.get("frontier_type", "")),
        "frontier_reason": str(review.get("frontier_reason", "")),
        "blocked_capability": str(review.get("blocked_capability", "")),
        "objective_conversion_review_accepted": bool(
            review.get("objective_conversion_review_accepted", False)
        ),
        "candidate_problem_statement": (
            "terminal_safe_policy_state_does_not_convert_to_objective_completion"
        ),
        "requested_hypothesis_families": hypothesis_families,
        "requested_experiment_styles": experiment_styles,
        "scientific_questions": questions,
        "suggested_initial_experiment_matrix": {
            "base_state_family": "terminal_safe_stop_or_avoidance_state",
            "single_step_actions": ["ACTION3", "ACTION4", "ACTION6"],
            "short_sequences": [
                ["ACTION3", "ACTION4"],
                ["ACTION4", "ACTION3"],
                ["ACTION6", "ACTION3"],
                ["ACTION6", "ACTION4"],
            ],
            "controls": ["hold_or_stop_state", "relation_progress_policy"],
            "success_metrics": [
                "objective_completion_signal",
                "levels_completed_after_rollout",
                "terminal_adjusted_progress_after_stop",
            ],
            "diagnostic_metrics": [
                "relation_delta_after_stop",
                "terminal_reentry_rate",
                "changed_pixels",
            ],
        },
        "ready_for_objective_conversion_hypothesis_generation": True,
        "ready_for_m2_or_m3_objective_conversion_branch": True,
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


def validate_objective_conversion_request(request: Mapping[str, Any]) -> None:
    if not request.get("request_id"):
        raise ValueError("request_id is required")
    if str(request.get("handoff_type", "")) != HANDOFF_TYPE:
        raise ValueError("objective conversion request has invalid handoff_type")
    if str(request.get("target", "")) != HANDOFF_TARGET:
        raise ValueError("objective conversion request target must be M2_OR_M3")
    if list(request.get("target_modules", []) or []) != TARGET_MODULES:
        raise ValueError("objective conversion request target_modules are invalid")
    if str(request.get("frontier_type", "")) != OBJECTIVE_CONVERSION_FRONTIER:
        raise ValueError("request frontier_type must be OBJECTIVE_CONVERSION_FRONTIER")
    if str(request.get("frontier_reason", "")) != TERMINAL_SAFE_BUT_PASSIVE:
        raise ValueError("request frontier_reason must be TERMINAL_SAFE_BUT_PASSIVE")
    if str(request.get("blocked_capability", "")) != OBJECTIVE_CONVERSION_AFTER_SAFE_STOP:
        raise ValueError("request blocked_capability is not objective conversion")
    if not bool(request.get("objective_conversion_review_accepted", False)):
        raise ValueError("objective conversion review must be accepted first")
    if not list(request.get("requested_hypothesis_families", []) or []):
        raise ValueError("objective conversion request needs hypothesis families")
    if not list(request.get("requested_experiment_styles", []) or []):
        raise ValueError("objective conversion request needs experiment styles")
    if not list(request.get("scientific_questions", []) or []):
        raise ValueError("objective conversion request needs scientific questions")
    if bool(request.get("ready_for_direct_downstream_write", False)):
        raise ValueError("P2.G3 cannot be ready for direct downstream write")
    if bool(request.get("a33_ready", False)):
        raise ValueError("objective conversion request cannot be A33-ready")
    if str(request.get("status", "")) != "UNRESOLVED":
        raise ValueError("objective conversion request must remain UNRESOLVED")
    if int(request.get("support", 0) or 0) != 0:
        raise ValueError("objective conversion request support must remain 0")
    if str(request.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("objective conversion request must remain candidate-only")
    if str(request.get("truth_status", "")) != TRUTH_STATUS:
        raise ValueError("objective conversion request truth_status must remain P2-local")
    if bool(request.get("revision_performed", False)):
        raise ValueError("objective conversion request revision_performed must be false")
    if int(request.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("objective conversion request wrong_confirmations must remain 0")
    if bool(request.get("handoff_request_counted_as_confirmation", False)):
        raise ValueError("objective conversion request cannot count as confirmation")
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
    if bool(summary.get("ready_for_m2_or_m3", False)):
        raise ValueError("P2.G3 source cannot already be ready for M2/M3")
    if not bool(summary.get("objective_conversion_review_no_write", False)):
        raise ValueError("P2.G3 requires objective conversion review no-write")
    for key in (
        "a40_write_performed",
        "m2_write_performed",
        "m3_write_performed",
        "a32_write_performed",
        "a33_write_performed",
    ):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"source must not have {key}")
    if bool(payload.get("objective_conversion_validation_counted_as_confirmation", False)):
        raise ValueError("objective conversion validation cannot be confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")


def write_objective_conversion_handoff_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_CONVERSION_HANDOFF_REQUESTS_OUTPUT_PATH
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
        description="Run P2.G3 objective-conversion handoff schema.",
    )
    parser.add_argument(
        "--objective-conversion-frontier-validation",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_VALIDATION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_CONVERSION_HANDOFF_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_handoff_schema(
        objective_conversion_frontier_validation_path=(
            args.objective_conversion_frontier_validation
        ),
    )
    write_objective_conversion_handoff_requests(payload, args.out)
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
