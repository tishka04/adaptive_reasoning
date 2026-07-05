"""P2.G2 no-write validator for objective-conversion frontier records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.objective_conversion_frontier_records import (
    DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_RECORDS_OUTPUT_PATH,
    OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
    OBJECTIVE_CONVERSION_FRONTIER,
    TERMINAL_SAFE_BUT_PASSIVE,
)
from theory.p2.policy_frontier_records import TRUTH_STATUS


DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_VALIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "objective_conversion_frontier_validation.json"
)


def run_objective_conversion_frontier_validation(
    *,
    objective_conversion_frontier_records_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_RECORDS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(objective_conversion_frontier_records_path)
    _validate_source_payload(payload)
    records = [
        dict(record)
        for record in payload.get("objective_conversion_frontier_records", []) or []
        if isinstance(record, Mapping)
    ]
    evaluations = [
        evaluate_objective_conversion_frontier_for_review(record)
        for record in records
    ]
    accepted = [
        row for row in evaluations if row["objective_conversion_review_accepted"]
    ]
    rejected = [
        row for row in evaluations if not row["objective_conversion_review_accepted"]
    ]
    return {
        "config": {
            "schema_version": "p2.objective_conversion_frontier_validation.v1",
            "objective_conversion_frontier_records_path": str(
                objective_conversion_frontier_records_path
            ),
            "inputs_read": ["P2.G1"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "validation_mode": "objective_conversion_review_no_write",
            "direct_m2_m3_handoff_allowed": False,
        },
        "frontier_evaluations": evaluations,
        "accepted_objective_conversion_reviews": accepted,
        "rejected_objective_conversion_reviews": rejected,
        "summary": {
            "objective_conversion_frontiers_seen": len(evaluations),
            "objective_conversion_reviews_accepted": len(accepted),
            "objective_conversion_reviews_rejected": len(rejected),
            "rejection_reasons": sorted(
                {
                    reason
                    for row in rejected
                    for reason in row.get("rejection_reasons", []) or []
                }
            ),
            "ready_for_objective_conversion_review": bool(accepted),
            "ready_for_m2_or_m3": False,
            "objective_conversion_review_no_write": True,
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
        "objective_conversion_validation_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def evaluate_objective_conversion_frontier_for_review(
    frontier: Mapping[str, Any],
) -> Dict[str, Any]:
    reasons = []
    if str(frontier.get("frontier_type", "")) != OBJECTIVE_CONVERSION_FRONTIER:
        reasons.append("NOT_OBJECTIVE_CONVERSION_FRONTIER")
    if str(frontier.get("frontier_reason", "")) != TERMINAL_SAFE_BUT_PASSIVE:
        reasons.append("UNEXPECTED_OBJECTIVE_CONVERSION_FRONTIER_REASON")
    if str(frontier.get("blocked_capability", "")) != OBJECTIVE_CONVERSION_AFTER_SAFE_STOP:
        reasons.append("UNEXPECTED_BLOCKED_CAPABILITY")
    if not bool(frontier.get("ready_for_objective_conversion_review", False)):
        reasons.append("NOT_READY_FOR_OBJECTIVE_CONVERSION_REVIEW")
    if bool(frontier.get("ready_for_m2_or_m3", False)):
        reasons.append("DIRECT_M2_M3_HANDOFF_NOT_ALLOWED_IN_P2_G2")
    if bool(frontier.get("ready_for_saturation_handoff", False)):
        reasons.append("SATURATION_HANDOFF_NOT_ALLOWED_FOR_OBJECTIVE_CONVERSION")
    if bool(frontier.get("a33_ready", False)):
        reasons.append("A33_READY_NOT_ALLOWED")
    if str(frontier.get("status", "")) != "UNRESOLVED":
        reasons.append("STATUS_NOT_UNRESOLVED")
    if str(frontier.get("status", "")) in {"CONFIRMED", "REFUTED"}:
        reasons.append("STATUS_CONFIRMED_OR_REFUTED_NOT_ALLOWED")
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
    if bool(frontier.get("policy_result_counted_as_scientific_verdict", False)):
        reasons.append("POLICY_RESULT_COUNTED_AS_VERDICT")
    if bool(frontier.get("objective_conversion_frontier_counted_as_confirmation", False)):
        reasons.append("OBJECTIVE_CONVERSION_FRONTIER_COUNTED_AS_CONFIRMATION")
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
        "frontier_type": str(frontier.get("frontier_type", "")),
        "frontier_reason": str(frontier.get("frontier_reason", "")),
        "blocked_capability": str(frontier.get("blocked_capability", "")),
        "source": str(frontier.get("source", "")),
        "desired_hypothesis_families": list(
            frontier.get("desired_hypothesis_families", []) or []
        ),
        "requested_experiment_styles": list(
            frontier.get("requested_experiment_styles", []) or []
        ),
        "scientific_questions": list(frontier.get("scientific_questions", []) or []),
        "ready_for_objective_conversion_review": bool(
            frontier.get("ready_for_objective_conversion_review", False)
        ),
        "ready_for_m2_or_m3": False,
        "objective_conversion_review_accepted": accepted,
        "objective_conversion_review_target": (
            "objective_conversion_frontier_review" if accepted else ""
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
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")
    if bool(payload.get("objective_conversion_frontier_counted_as_confirmation", False)):
        raise ValueError("objective conversion frontier cannot count as confirmation")
    if bool(payload.get("a40_write_performed", False)) or bool(
        summary.get("a40_write_performed", False)
    ):
        raise ValueError("source must not have written A40")
    if bool(payload.get("m2_write_performed", False)) or bool(
        summary.get("m2_write_performed", False)
    ):
        raise ValueError("source must not have written M2")
    if bool(payload.get("m3_write_performed", False)) or bool(
        summary.get("m3_write_performed", False)
    ):
        raise ValueError("source must not have written M3")
    if bool(payload.get("a32_write_performed", False)) or bool(
        summary.get("a32_write_performed", False)
    ):
        raise ValueError("source must not have written A32")
    if bool(payload.get("a33_write_performed", False)) or bool(
        summary.get("a33_write_performed", False)
    ):
        raise ValueError("source must not have written A33")


def write_objective_conversion_frontier_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_VALIDATION_OUTPUT_PATH
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
        description="Run P2.G2 objective-conversion frontier validation.",
    )
    parser.add_argument(
        "--objective-conversion-frontier-records",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_RECORDS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_CONVERSION_FRONTIER_VALIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_frontier_validation(
        objective_conversion_frontier_records_path=(
            args.objective_conversion_frontier_records
        ),
    )
    write_objective_conversion_frontier_validation(payload, args.out)
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
