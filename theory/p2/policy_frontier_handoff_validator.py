"""P2.2 validator for policy FrontierRecord handoff eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.policy_frontier_records import (
    DEFAULT_P2_POLICY_FRONTIER_RECORDS_OUTPUT_PATH,
    TRUTH_STATUS,
)


DEFAULT_P2_POLICY_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "bp35_policy_frontier_handoff_validation.json"
)


def run_policy_frontier_handoff_validation(
    *,
    policy_frontier_records_path: str | Path = (
        DEFAULT_P2_POLICY_FRONTIER_RECORDS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(policy_frontier_records_path)
    _validate_source_payload(payload)
    records = _all_records(payload)
    evaluations = [evaluate_record_for_handoff(record) for record in records]
    accepted = [row for row in evaluations if row["handoff_accepted"]]
    rejected = [row for row in evaluations if not row["handoff_accepted"]]
    return {
        "config": {
            "schema_version": "p2.policy_frontier_handoff_validation.v1",
            "policy_frontier_records_path": str(policy_frontier_records_path),
            "inputs_read": ["P2.1"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
        },
        "record_evaluations": evaluations,
        "accepted_handoffs": accepted,
        "rejected_handoffs": rejected,
        "summary": {
            "records_seen": len(records),
            "real_records_seen": len(
                [record for record in records if not bool(record.get("synthetic_fixture"))]
            ),
            "synthetic_records_seen": len(
                [record for record in records if bool(record.get("synthetic_fixture"))]
            ),
            "handoffs_accepted": len(accepted),
            "handoffs_rejected": len(rejected),
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
        "handoff_validation_counted_as_confirmation": False,
    }


def evaluate_record_for_handoff(record: Mapping[str, Any]) -> Dict[str, Any]:
    reasons = []
    if bool(record.get("synthetic_fixture", False)):
        reasons.append("SYNTHETIC_FIXTURE_NOT_FOR_SCIENTIFIC_HANDOFF")
    if bool(record.get("synthetic_fixture_not_for_scientific_handoff", False)):
        reasons.append("SYNTHETIC_FIXTURE_NOT_FOR_SCIENTIFIC_HANDOFF")
    if not bool(record.get("ready_for_m2_or_m3", False)):
        reasons.append("NOT_READY_FOR_M2_OR_M3")
    if int(record.get("support", 0) or 0) != 0:
        reasons.append("SUPPORT_MUST_REMAIN_ZERO")
    if str(record.get("revision_status", "")) != "CANDIDATE_ONLY":
        reasons.append("REVISION_STATUS_NOT_CANDIDATE_ONLY")
    if str(record.get("truth_status", "")) != TRUTH_STATUS:
        reasons.append("TRUTH_STATUS_NOT_EVALUATED_BY_P2_REQUIRED")
    if bool(record.get("policy_result_counted_as_scientific_verdict", False)):
        reasons.append("POLICY_RESULT_COUNTED_AS_VERDICT")
    if bool(record.get("frontier_trigger_counted_as_confirmation", False)):
        reasons.append("FRONTIER_TRIGGER_COUNTED_AS_CONFIRMATION")
    if bool(record.get("revision_performed", False)):
        reasons.append("REVISION_PERFORMED_NOT_ALLOWED")
    if int(record.get("wrong_confirmations", 0) or 0) != 0:
        reasons.append("WRONG_CONFIRMATIONS_NOT_ZERO")
    reasons = sorted(set(reasons))
    return {
        "frontier_id": str(record.get("frontier_id", "")),
        "game_id": str(record.get("game_id", "")),
        "source": str(record.get("source", "")),
        "frontier_reason": str(record.get("frontier_reason", "")),
        "exhausted_policy": str(record.get("exhausted_policy", "")),
        "refresh_candidates": list(record.get("refresh_candidates", []) or []),
        "synthetic_fixture": bool(record.get("synthetic_fixture", False)),
        "ready_for_m2_or_m3": bool(record.get("ready_for_m2_or_m3", False)),
        "handoff_accepted": not reasons,
        "rejection_reasons": reasons,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _all_records(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for key in ("real_frontier_records", "synthetic_frontier_records"):
        for record in payload.get(key, []) or []:
            if isinstance(record, Mapping):
                rows.append(dict(record))
    return rows


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary", {}) or {}
    if isinstance(summary, Mapping):
        if int(summary.get("support", 0) or 0) != 0:
            raise ValueError("source summary support must remain 0")
        if bool(summary.get("revision_performed", False)):
            raise ValueError("source summary revision_performed must be false")
        if int(summary.get("wrong_confirmations", 0) or 0) != 0:
            raise ValueError("source summary wrong_confirmations must remain 0")
        if bool(summary.get("synthetic_records_counted_as_handoff", False)):
            raise ValueError("synthetic records cannot be counted as handoff")
    if bool(payload.get("frontier_records_counted_as_confirmation", False)):
        raise ValueError("frontier records cannot be counted as confirmation")


def write_policy_frontier_handoff_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_P2_POLICY_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH
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
        description="Run P2.2 policy FrontierRecord handoff validation.",
    )
    parser.add_argument(
        "--policy-frontier-records",
        type=Path,
        default=DEFAULT_P2_POLICY_FRONTIER_RECORDS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_POLICY_FRONTIER_HANDOFF_VALIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_policy_frontier_handoff_validation(
        policy_frontier_records_path=args.policy_frontier_records,
    )
    write_policy_frontier_handoff_validation(payload, args.out)
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
