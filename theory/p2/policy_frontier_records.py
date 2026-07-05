"""P2.1 Policy FrontierRecord schema from P1 runtime frontier triggers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


DEFAULT_P1_MOVEMENT_FRONTIER_TRIGGER_PATH = (
    Path("diagnostics") / "p1" / "bp35_movement_policy_frontier_trigger.json"
)
DEFAULT_P2_POLICY_FRONTIER_RECORDS_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "bp35_policy_frontier_records.json"
)
TRUTH_STATUS = "NOT_EVALUATED_BY_P2"


@dataclass(frozen=True)
class PolicyFrontierRecord:
    frontier_id: str
    source: str
    game_id: str
    frontier_context_id: str
    frontier_reason: str
    exhausted_policy: str
    refresh_candidates: tuple[str, ...]
    ready_for_m2_or_m3: bool
    synthetic_fixture: bool
    synthetic_fixture_not_for_scientific_handoff: bool
    policy_result_counted_as_scientific_verdict: bool = False
    frontier_trigger_counted_as_confirmation: bool = False
    status: str = "UNRESOLVED"
    support: int = 0
    revision_status: str = "CANDIDATE_ONLY"
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "source": self.source,
            "game_id": self.game_id,
            "frontier_context_id": self.frontier_context_id,
            "frontier_reason": self.frontier_reason,
            "exhausted_policy": self.exhausted_policy,
            "refresh_candidates": list(self.refresh_candidates),
            "ready_for_m2_or_m3": self.ready_for_m2_or_m3,
            "synthetic_fixture": self.synthetic_fixture,
            "synthetic_fixture_not_for_scientific_handoff": (
                self.synthetic_fixture_not_for_scientific_handoff
            ),
            "policy_result_counted_as_scientific_verdict": (
                self.policy_result_counted_as_scientific_verdict
            ),
            "frontier_trigger_counted_as_confirmation": (
                self.frontier_trigger_counted_as_confirmation
            ),
            "status": self.status,
            "support": int(self.support),
            "revision_status": self.revision_status,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_policy_frontier_records(
    *,
    p1_frontier_trigger_path: str | Path = DEFAULT_P1_MOVEMENT_FRONTIER_TRIGGER_PATH,
) -> Dict[str, Any]:
    trigger_payload = _load_json(p1_frontier_trigger_path)
    _validate_p1_trigger_payload(trigger_payload)
    real_records = [
        record
        for record in [
            _record_from_p1_frontier(
                trigger_payload.get("real_rollout_evaluation", {}) or {},
                source="P1.11",
            )
        ]
        if record is not None
    ]
    synthetic_records = [
        record
        for record in [
            _record_from_p1_frontier(
                trigger_payload.get("synthetic_saturation_fixture", {}) or {},
                source="P1.11.synthetic_fixture",
            )
        ]
        if record is not None
    ]
    for record in [*real_records, *synthetic_records]:
        validate_policy_frontier_record(record)

    real_handoffs = [record for record in real_records if record.ready_for_m2_or_m3]
    synthetic_validation_records = [
        record
        for record in synthetic_records
        if record.synthetic_fixture_not_for_scientific_handoff
    ]
    return {
        "config": {
            "schema_version": "p2.policy_frontier_record.v1",
            "source": "P1.11",
            "p1_frontier_trigger_path": str(p1_frontier_trigger_path),
            "inputs_read": ["P1.11"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["P1", "A40", "M2", "M3", "A32", "A33"],
        },
        "real_frontier_records": [record.to_dict() for record in real_records],
        "synthetic_frontier_records": [
            record.to_dict() for record in synthetic_records
        ],
        "summary": {
            "real_frontier_records": len(real_records),
            "synthetic_frontier_records": len(synthetic_records),
            "real_handoffs_produced": len(real_handoffs),
            "synthetic_records_for_schema_validation_only": len(
                synthetic_validation_records
            ),
            "ready_for_m2_or_m3_records": len(real_handoffs),
            "synthetic_records_counted_as_handoff": False,
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
        "frontier_records_counted_as_confirmation": False,
    }


def validate_policy_frontier_record(record: PolicyFrontierRecord) -> None:
    if not record.frontier_id:
        raise ValueError("frontier_id is required")
    if not record.game_id:
        raise ValueError("game_id is required")
    if not record.frontier_reason:
        raise ValueError("frontier_reason is required")
    if not record.exhausted_policy:
        raise ValueError("exhausted_policy is required")
    if record.status != "UNRESOLVED":
        raise ValueError("PolicyFrontierRecord status must remain UNRESOLVED")
    if record.support != 0:
        raise ValueError("PolicyFrontierRecord support must remain 0")
    if record.revision_status != "CANDIDATE_ONLY":
        raise ValueError("PolicyFrontierRecord must remain candidate-only")
    if record.truth_status != TRUTH_STATUS:
        raise ValueError("PolicyFrontierRecord truth_status must remain P2-local")
    if record.revision_performed:
        raise ValueError("PolicyFrontierRecord revision_performed must be false")
    if record.wrong_confirmations != 0:
        raise ValueError("PolicyFrontierRecord wrong_confirmations must remain 0")
    if record.policy_result_counted_as_scientific_verdict:
        raise ValueError("policy result cannot be counted as scientific verdict")
    if record.frontier_trigger_counted_as_confirmation:
        raise ValueError("frontier trigger cannot be counted as confirmation")
    if record.synthetic_fixture and record.ready_for_m2_or_m3:
        raise ValueError("synthetic fixture cannot be ready for M2/M3 handoff")
    if (
        record.synthetic_fixture
        and not record.synthetic_fixture_not_for_scientific_handoff
    ):
        raise ValueError("synthetic fixture must be marked non-scientific")


def _record_from_p1_frontier(
    evaluation: Mapping[str, Any],
    *,
    source: str,
) -> PolicyFrontierRecord | None:
    if not bool(evaluation.get("frontier_triggered", False)):
        return None
    frontier = evaluation.get("frontier_record", {}) or {}
    if not isinstance(frontier, Mapping):
        return None
    synthetic = bool(frontier.get("synthetic_fixture", False))
    synthetic_not_handoff = bool(
        frontier.get("synthetic_fixture_not_for_scientific_handoff", False)
    )
    return PolicyFrontierRecord(
        frontier_id=str(frontier.get("frontier_id", "")),
        source=source,
        game_id=str(frontier.get("game_id", "")),
        frontier_context_id=str(frontier.get("frontier_context_id", "")),
        frontier_reason=str(frontier.get("frontier_reason", "")),
        exhausted_policy=str(frontier.get("exhausted_policy", "")),
        refresh_candidates=tuple(
            str(value) for value in frontier.get("refresh_candidates", []) or []
        ),
        ready_for_m2_or_m3=bool(
            frontier.get("ready_for_p2_frontier_extraction", False)
            and not synthetic
            and not synthetic_not_handoff
        ),
        synthetic_fixture=synthetic,
        synthetic_fixture_not_for_scientific_handoff=synthetic_not_handoff,
        policy_result_counted_as_scientific_verdict=bool(
            frontier.get("policy_result_counted_as_scientific_verdict", False)
        ),
        frontier_trigger_counted_as_confirmation=bool(
            frontier.get("frontier_trigger_counted_as_confirmation", False)
        ),
        status=str(frontier.get("status", "UNRESOLVED")),
        support=int(frontier.get("support", 0) or 0),
        revision_status=str(frontier.get("revision_status", "CANDIDATE_ONLY")),
        truth_status=TRUTH_STATUS,
        revision_performed=bool(frontier.get("revision_performed", False)),
        wrong_confirmations=int(frontier.get("wrong_confirmations", 0) or 0),
    )


def _validate_p1_trigger_payload(payload: Mapping[str, Any]) -> None:
    for section_name in ("summary",):
        section = payload.get(section_name, {}) or {}
        if not isinstance(section, Mapping):
            continue
        if int(section.get("support", 0) or 0) != 0:
            raise ValueError(f"{section_name}.support must remain 0")
        if bool(section.get("policy_result_counted_as_scientific_verdict", False)):
            raise ValueError(
                f"{section_name}.policy_result_counted_as_scientific_verdict must be false"
            )
        if bool(section.get("revision_performed", False)):
            raise ValueError(f"{section_name}.revision_performed must be false")
        if int(section.get("wrong_confirmations", 0) or 0) != 0:
            raise ValueError(f"{section_name}.wrong_confirmations must remain 0")


def write_policy_frontier_records(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P2_POLICY_FRONTIER_RECORDS_OUTPUT_PATH,
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
        description="Run P2.1 policy FrontierRecord extraction.",
    )
    parser.add_argument(
        "--p1-frontier-trigger",
        type=Path,
        default=DEFAULT_P1_MOVEMENT_FRONTIER_TRIGGER_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_POLICY_FRONTIER_RECORDS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_policy_frontier_records(
        p1_frontier_trigger_path=args.p1_frontier_trigger,
    )
    write_policy_frontier_records(payload, args.out)
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
