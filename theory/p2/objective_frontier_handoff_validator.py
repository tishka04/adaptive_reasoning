"""P2.5 no-write validator for objective-alignment frontier handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.policy_frontier_records import TRUTH_STATUS
from theory.p2.terminal_outcome_frontier import (
    DEFAULT_P2_TERMINAL_OUTCOME_FRONTIER_OUTPUT_PATH,
    LOCAL_PRODUCTIVE_TERMINAL_FAILED,
    OBJECTIVE_ALIGNMENT_FRONTIER,
)


DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "bp35_objective_frontier_handoff_validation.json"
)


def run_objective_frontier_handoff_validation(
    *,
    terminal_outcome_frontier_path: str | Path = (
        DEFAULT_P2_TERMINAL_OUTCOME_FRONTIER_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(terminal_outcome_frontier_path)
    _validate_source_payload(payload)
    frontier = payload.get("terminal_outcome_frontier")
    evaluations = []
    if isinstance(frontier, Mapping):
        evaluations.append(evaluate_objective_frontier_for_handoff(frontier))
    accepted = [row for row in evaluations if row["objective_review_accepted"]]
    rejected = [row for row in evaluations if not row["objective_review_accepted"]]
    return {
        "config": {
            "schema_version": "p2.objective_frontier_handoff_validation.v1",
            "terminal_outcome_frontier_path": str(terminal_outcome_frontier_path),
            "inputs_read": ["P2.4-terminal"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "validation_mode": "objective_frontier_review_no_write",
            "saturation_handoff_allowed": False,
        },
        "frontier_evaluations": evaluations,
        "accepted_objective_reviews": accepted,
        "rejected_objective_reviews": rejected,
        "summary": {
            "objective_frontiers_seen": len(evaluations),
            "objective_reviews_accepted": len(accepted),
            "objective_reviews_rejected": len(rejected),
            "rejection_reasons": sorted(
                {
                    reason
                    for row in rejected
                    for reason in row.get("rejection_reasons", []) or []
                }
            ),
            "saturation_handoffs_accepted": 0,
            "ready_for_p2_4_saturation_handoff": False,
            "ready_for_objective_frontier_review": bool(accepted),
            "objective_review_no_write": True,
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
        "objective_handoff_validation_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
    }


def evaluate_objective_frontier_for_handoff(
    frontier: Mapping[str, Any],
) -> Dict[str, Any]:
    reasons = []
    if str(frontier.get("frontier_type", "")) != OBJECTIVE_ALIGNMENT_FRONTIER:
        reasons.append("NOT_OBJECTIVE_ALIGNMENT_FRONTIER")
    if str(frontier.get("frontier_reason", "")) != LOCAL_PRODUCTIVE_TERMINAL_FAILED:
        reasons.append("UNEXPECTED_OBJECTIVE_FRONTIER_REASON")
    if not bool(frontier.get("ready_for_objective_frontier_review", False)):
        reasons.append("NOT_READY_FOR_OBJECTIVE_FRONTIER_REVIEW")
    if bool(frontier.get("ready_for_p2_4_saturation_handoff", False)):
        reasons.append("SATURATION_HANDOFF_NOT_ALLOWED_FOR_OBJECTIVE_FRONTIER")
    if bool(frontier.get("source_saturation_handoff_ready", False)):
        reasons.append("SOURCE_SATURATION_HANDOFF_READY_NOT_OBJECTIVE_ONLY")
    if bool(frontier.get("ready_for_m2_or_m3", False)):
        reasons.append("DIRECT_M2_M3_HANDOFF_NOT_ALLOWED_IN_P2_5")
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
    if bool(frontier.get("policy_result_counted_as_scientific_verdict", False)):
        reasons.append("POLICY_RESULT_COUNTED_AS_VERDICT")
    if bool(frontier.get("objective_frontier_counted_as_confirmation", False)):
        reasons.append("OBJECTIVE_FRONTIER_COUNTED_AS_CONFIRMATION")

    reasons = sorted(set(reasons))
    accepted = not reasons
    return {
        "frontier_id": str(frontier.get("frontier_id", "")),
        "game_id": str(frontier.get("game_id", "")),
        "frontier_type": str(frontier.get("frontier_type", "")),
        "frontier_reason": str(frontier.get("frontier_reason", "")),
        "source": str(frontier.get("source", "")),
        "terminal_runs": int(frontier.get("terminal_runs", 0) or 0),
        "terminal_budgets": list(frontier.get("terminal_budgets", []) or []),
        "ready_for_objective_frontier_review": bool(
            frontier.get("ready_for_objective_frontier_review", False)
        ),
        "ready_for_p2_4_saturation_handoff": False,
        "objective_review_accepted": accepted,
        "objective_review_target": (
            "objective_frontier_review" if accepted else ""
        ),
        "rejection_reasons": reasons,
        "saturation_handoff_accepted": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(summary.get("revision_performed", False)):
        raise ValueError("source summary revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source summary wrong_confirmations must remain 0")
    if bool(payload.get("terminal_outcome_frontier_counted_as_confirmation", False)):
        raise ValueError("terminal frontier cannot be counted as confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be counted as scientific verdict")
    if bool(payload.get("a40_write_performed", False)):
        raise ValueError("source must not have written A40")
    if bool(payload.get("m2_write_performed", False)):
        raise ValueError("source must not have written M2")
    if bool(payload.get("m3_write_performed", False)):
        raise ValueError("source must not have written M3")


def write_objective_frontier_handoff_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH
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
        description="Run P2.5 objective-frontier handoff validation.",
    )
    parser.add_argument(
        "--terminal-outcome-frontier",
        type=Path,
        default=DEFAULT_P2_TERMINAL_OUTCOME_FRONTIER_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_frontier_handoff_validation(
        terminal_outcome_frontier_path=args.terminal_outcome_frontier,
    )
    write_objective_frontier_handoff_validation(payload, args.out)
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
