"""A32 - candidate-only intake from M3 into the A15-A31 revision contract.

A32 verifies that the M3.6 queue can be consumed as A15-A31 candidate input.
It does not perform belief revision and does not emit confirmations/refutations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.epistemic_metrics import (
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    score_beliefs,
)
from theory.m3.a15_revision_queue import DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH


DEFAULT_A32_OUTPUT_PATH = Path("diagnostics") / "a32" / "m3_revision_intake.json"
ACCEPTED_CANDIDATE = "ACCEPTED_CANDIDATE"
REJECTED_CANDIDATE = "REJECTED_CANDIDATE"


@dataclass(frozen=True)
class A32RevisionIntakeCandidate:
    """One queue item accepted as candidate-only input for A15-A31."""

    queue_item_id: str
    game_id: str
    key: str
    description: str
    evidence_summary: Dict[str, Any]
    control_evidence: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    intake_status: str = ACCEPTED_CANDIDATE
    consumer_contract: str = "HypothesisRecord"
    requested_next_step: str = "A15_A31_REVISION_DECISION_REQUIRED"
    allowed_revision_outcomes: Tuple[str, ...] = (
        "confirm_after_revision",
        "refute_after_revision",
        "request_followup_test",
    )
    status: str = "UNRESOLVED"
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    contradictions: int = 0
    controlled_test_required: bool = True
    revision_performed: bool = False
    wrong_confirmations: int = 0
    observation_counted_as_confirmation: bool = False
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=self.description,
            status=HypothesisStatus.UNRESOLVED,
            support=0,
            contradictions=0,
            experiments_spent=int(
                self.evidence_summary.get("controlled_experiments_run", 0) or 0
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_item_id": self.queue_item_id,
            "game_id": self.game_id,
            "key": self.key,
            "description": self.description,
            "intake_status": self.intake_status,
            "consumer_contract": self.consumer_contract,
            "requested_next_step": self.requested_next_step,
            "allowed_revision_outcomes": list(self.allowed_revision_outcomes),
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "contradictions": int(self.contradictions),
            "controlled_test_required": self.controlled_test_required,
            "evidence_summary": dict(self.evidence_summary),
            "control_evidence": [dict(row) for row in self.control_evidence],
            "candidate_record": candidate_record_dict(self.to_record()),
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def run_a32_m3_revision_intake(
    *,
    queue_path: str | Path = DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Validate and adapt the M3.6 queue as A15-A31 candidate input."""
    payload = _load_json(queue_path)
    accepted, rejected = build_a32_revision_intake_candidates(payload)
    scores = score_intake_candidates_by_game(accepted)
    return {
        "config": {
            "queue_path": str(queue_path),
        },
        "summary": summarize_a32_revision_intake(accepted, rejected, scores),
        "accepted_candidates": [item.to_dict() for item in accepted],
        "rejected_candidates": [dict(item) for item in rejected],
        "candidate_records": [
            candidate_record_dict(item.to_record()) for item in accepted
        ],
        "scores_by_game": {
            game_id: score.to_dict() for game_id, score in sorted(scores.items())
        },
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "contradictions": 0,
        "controlled_test_required": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "observation_counted_as_confirmation": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def build_a32_revision_intake_candidates(
    queue_payload: Mapping[str, Any],
) -> tuple[Tuple[A32RevisionIntakeCandidate, ...], Tuple[Dict[str, Any], ...]]:
    accepted: list[A32RevisionIntakeCandidate] = []
    rejected: list[Dict[str, Any]] = []
    for item in queue_payload.get("queue_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        reason = rejection_reason(item)
        if reason:
            rejected.append(
                {
                    "queue_item_id": str(item.get("queue_item_id", "")),
                    "hypothesis_key": str(item.get("hypothesis_key", "")),
                    "intake_status": REJECTED_CANDIDATE,
                    "reason": reason,
                }
            )
            continue
        accepted.append(candidate_from_queue_item(item))
    return tuple(accepted), tuple(rejected)


def candidate_from_queue_item(
    item: Mapping[str, Any],
) -> A32RevisionIntakeCandidate:
    return A32RevisionIntakeCandidate(
        queue_item_id=str(item.get("queue_item_id", "")),
        game_id=str(item.get("game_id", "")),
        key=str(item.get("hypothesis_key", "")),
        description=str(item.get("description", "")),
        evidence_summary=dict(item.get("evidence_summary", {}) or {}),
        control_evidence=tuple(
            dict(row)
            for row in item.get("control_evidence", []) or []
            if isinstance(row, Mapping)
        ),
    )


def rejection_reason(item: Mapping[str, Any]) -> str:
    evidence = dict(item.get("evidence_summary", {}) or {})
    if str(item.get("status", "")).upper() != "UNRESOLVED":
        return "status_not_unresolved"
    if str(item.get("revision_status", "")).upper() != "CANDIDATE_ONLY":
        return "revision_status_not_candidate_only"
    if int(item.get("support", 0) or 0) != 0:
        return "support_must_remain_zero"
    if int(item.get("contradictions", 0) or 0) != 0:
        return "contradictions_must_remain_zero"
    if not bool(item.get("controlled_test_required", True)):
        return "controlled_test_required_missing"
    if int(evidence.get("support_events", 0) or 0) < 3:
        return "insufficient_support_events"
    if int(evidence.get("independent_support_events", 0) or 0) < 2:
        return "insufficient_independent_support_events"
    if int(evidence.get("contradiction_events", 0) or 0) != 0:
        return "contradiction_events_present"
    return ""


def score_intake_candidates_by_game(
    candidates: Sequence[A32RevisionIntakeCandidate],
) -> Dict[str, Any]:
    records_by_game: Dict[str, list[HypothesisRecord]] = {}
    for candidate in candidates:
        records_by_game.setdefault(candidate.game_id, []).append(candidate.to_record())
    return {
        game_id: score_beliefs(
            records,
            MechanicsOracle(game_id),
            experiment_actions=sum(record.experiments_spent for record in records),
        )
        for game_id, records in records_by_game.items()
    }


def summarize_a32_revision_intake(
    accepted: Sequence[A32RevisionIntakeCandidate],
    rejected: Sequence[Mapping[str, Any]],
    scores_by_game: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "accepted_candidates": len(accepted),
        "rejected_candidates": len(rejected),
        "candidate_records": len(accepted),
        "support_events": sum(
            int(item.evidence_summary.get("support_events", 0) or 0)
            for item in accepted
        ),
        "independent_support_events": sum(
            int(item.evidence_summary.get("independent_support_events", 0) or 0)
            for item in accepted
        ),
        "reused_control_support_events": sum(
            int(item.evidence_summary.get("reused_control_support_events", 0) or 0)
            for item in accepted
        ),
        "contradiction_events": sum(
            int(item.evidence_summary.get("contradiction_events", 0) or 0)
            for item in accepted
        ),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "contradictions": 0,
        "controlled_test_required": True,
        "revision_performed": False,
        "observation_counted_as_confirmation": False,
        "wrong_confirmations": sum(
            int(score.wrong_confirmations) for score in scores_by_game.values()
        ),
        "hypotheses_confirmed": sum(
            int(score.hypotheses_confirmed) for score in scores_by_game.values()
        ),
        "hypotheses_refuted": sum(
            int(score.hypotheses_refuted) for score in scores_by_game.values()
        ),
    }


def candidate_record_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_revision_intake(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A32_OUTPUT_PATH,
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
        description="Run A32 candidate-only intake from the M3 revision queue.",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_A32_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_m3_revision_intake(queue_path=args.queue)
    write_a32_revision_intake(payload, args.out)
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
