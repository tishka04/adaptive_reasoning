"""A32 minimal A15 queue consumer.

A32 is the first explicit scientific decision step after M3.6. It consumes the
candidate-only M3 queue and writes decisions under ``diagnostics/a32`` only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus
from theory.m3.a15_revision_queue import DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH


DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH = (
    Path("diagnostics") / "a32" / "a15_revision_decisions.json"
)

REVISION_ACCEPTED_AS_CONFIRMED = "REVISION_ACCEPTED_AS_CONFIRMED"
REVISION_REJECTED_AS_INSUFFICIENT = "REVISION_REJECTED_AS_INSUFFICIENT"
FOLLOWUP_REQUIRED = "FOLLOWUP_REQUIRED"


@dataclass(frozen=True)
class A32RevisionDecision:
    """One explicit A32 scientific decision."""

    queue_item_id: str
    game_id: str
    key: str
    description: str
    decision: str
    reasons: Tuple[str, ...]
    evidence_summary: Dict[str, Any]
    control_evidence: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    input_record: HypothesisRecord | None = None
    decision_record: HypothesisRecord | None = None
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        input_record = self.input_record or unresolved_record_from_decision(self)
        decision_record = self.decision_record or decision_record_from_decision(self)
        return {
            "queue_item_id": self.queue_item_id,
            "game_id": self.game_id,
            "key": self.key,
            "description": self.description,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "evidence_summary": dict(self.evidence_summary),
            "control_evidence": [dict(row) for row in self.control_evidence],
            "input_record": record_to_dict(input_record),
            "decision_record": record_to_dict(decision_record),
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": False,
            "prior_counted_as_proof": False,
            "observation_counted_as_confirmation": False,
        }


def run_a32_revision_decision_consumer(
    *,
    queue_path: str | Path = DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Read the M3.6 queue and produce explicit A32 decisions."""
    payload = _load_json(queue_path)
    decisions = build_a32_revision_decisions(payload)
    return {
        "config": {
            "queue_path": str(queue_path),
        },
        "summary": summarize_a32_revision_decisions(decisions),
        "revision_decisions": [decision.to_dict() for decision in decisions],
        "input_records": [
            record_to_dict(decision.input_record or unresolved_record_from_decision(decision))
            for decision in decisions
        ],
        "decision_records": [
            record_to_dict(
                decision.decision_record or decision_record_from_decision(decision)
            )
            for decision in decisions
        ],
        "decision_scope": "A32_ONLY",
        "m3_artifacts_mutated": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def build_a32_revision_decisions(
    queue_payload: Mapping[str, Any],
) -> Tuple[A32RevisionDecision, ...]:
    decisions: list[A32RevisionDecision] = []
    for item in queue_payload.get("queue_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        decisions.append(decision_from_queue_item(item))
    return tuple(decisions)


def decision_from_queue_item(item: Mapping[str, Any]) -> A32RevisionDecision:
    key = str(item.get("hypothesis_key", ""))
    evidence = dict(item.get("evidence_summary", {}) or {})
    controls = tuple(
        dict(row)
        for row in item.get("control_evidence", []) or []
        if isinstance(row, Mapping)
    )
    reasons = decision_reasons(item, evidence=evidence, controls=controls)
    decision = decision_label(reasons)
    base = A32RevisionDecision(
        queue_item_id=str(item.get("queue_item_id", "")),
        game_id=str(item.get("game_id", "")),
        key=key,
        description=str(item.get("description", "")),
        decision=decision,
        reasons=tuple(reasons),
        evidence_summary=evidence,
        control_evidence=controls,
        revision_performed=decision == REVISION_ACCEPTED_AS_CONFIRMED,
    )
    input_record = unresolved_record_from_decision(base)
    decision_record = decision_record_from_decision(base)
    return A32RevisionDecision(
        queue_item_id=base.queue_item_id,
        game_id=base.game_id,
        key=base.key,
        description=base.description,
        decision=base.decision,
        reasons=base.reasons,
        evidence_summary=base.evidence_summary,
        control_evidence=base.control_evidence,
        input_record=input_record,
        decision_record=decision_record,
        revision_performed=base.revision_performed,
        wrong_confirmations=0,
    )


def decision_reasons(
    item: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    reasons: list[str] = []
    if not str(item.get("hypothesis_key", "")).startswith("mechanic_prediction::"):
        reasons.append("unsupported_hypothesis_family")
    if str(item.get("status", "")).upper() != "UNRESOLVED":
        reasons.append("queue_status_not_unresolved")
    if str(item.get("revision_status", "")).upper() != "CANDIDATE_ONLY":
        reasons.append("queue_revision_status_not_candidate_only")
    if int(item.get("support", 0) or 0) != 0:
        reasons.append("queue_support_not_zero_before_revision")
    if int(item.get("contradictions", 0) or 0) != 0:
        reasons.append("queue_contradictions_not_zero_before_revision")
    if not bool(item.get("controlled_test_required", True)):
        reasons.append("controlled_test_required_missing")

    support_events = int(evidence.get("support_events", 0) or 0)
    independent_support_events = int(
        evidence.get("independent_support_events", 0) or 0
    )
    contradiction_events = int(evidence.get("contradiction_events", 0) or 0)
    distinct_independent_controls = {
        str(row.get("control_action", ""))
        for row in controls
        if int(row.get("independent_support_events", 0) or 0) > 0
        and str(row.get("control_action", ""))
    }

    if support_events < 3:
        reasons.append("insufficient_support_events")
    if independent_support_events < 2:
        reasons.append("insufficient_independent_support_events")
    if contradiction_events > 0:
        reasons.append("contradiction_events_present")
    if len(distinct_independent_controls) < 2:
        reasons.append("insufficient_distinct_independent_controls")

    if not reasons:
        reasons.append("a32_revision_criteria_satisfied")
    return tuple(reasons)


def decision_label(reasons: Sequence[str]) -> str:
    reason_set = set(reasons)
    if reason_set == {"a32_revision_criteria_satisfied"}:
        return REVISION_ACCEPTED_AS_CONFIRMED
    if "contradiction_events_present" in reason_set:
        return FOLLOWUP_REQUIRED
    if any(reason.startswith("insufficient_") for reason in reason_set):
        return FOLLOWUP_REQUIRED
    return REVISION_REJECTED_AS_INSUFFICIENT


def unresolved_record_from_decision(decision: A32RevisionDecision) -> HypothesisRecord:
    return HypothesisRecord(
        key=decision.key,
        description=decision.description,
        status=HypothesisStatus.UNRESOLVED,
        support=0,
        contradictions=0,
        experiments_spent=int(
            decision.evidence_summary.get("controlled_experiments_run", 0) or 0
        ),
    )


def decision_record_from_decision(decision: A32RevisionDecision) -> HypothesisRecord:
    if decision.decision == REVISION_ACCEPTED_AS_CONFIRMED:
        return HypothesisRecord(
            key=decision.key,
            description=decision.description,
            status=HypothesisStatus.CONFIRMED,
            support=int(
                decision.evidence_summary.get("independent_support_events", 0) or 0
            ),
            contradictions=0,
            experiments_spent=int(
                decision.evidence_summary.get("controlled_experiments_run", 0) or 0
            ),
        )
    return unresolved_record_from_decision(decision)


def summarize_a32_revision_decisions(
    decisions: Sequence[A32RevisionDecision],
) -> Dict[str, Any]:
    return {
        "queue_items_consumed": len(decisions),
        "revision_accepted_as_confirmed": len(
            [
                decision
                for decision in decisions
                if decision.decision == REVISION_ACCEPTED_AS_CONFIRMED
            ]
        ),
        "revision_rejected_as_insufficient": len(
            [
                decision
                for decision in decisions
                if decision.decision == REVISION_REJECTED_AS_INSUFFICIENT
            ]
        ),
        "followup_required": len(
            [decision for decision in decisions if decision.decision == FOLLOWUP_REQUIRED]
        ),
        "support_events": sum(
            int(decision.evidence_summary.get("support_events", 0) or 0)
            for decision in decisions
        ),
        "independent_support_events": sum(
            int(decision.evidence_summary.get("independent_support_events", 0) or 0)
            for decision in decisions
        ),
        "reused_control_support_events": sum(
            int(decision.evidence_summary.get("reused_control_support_events", 0) or 0)
            for decision in decisions
        ),
        "contradiction_events": sum(
            int(decision.evidence_summary.get("contradiction_events", 0) or 0)
            for decision in decisions
        ),
        "m3_artifacts_mutated": False,
        "wrong_confirmations": 0,
    }


def record_to_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_revision_decisions(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH,
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
        description="Run A32 scientific decisions from the M3.6 revision queue.",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_revision_decision_consumer(queue_path=args.queue)
    write_a32_revision_decisions(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "decision_scope": "A32_ONLY",
                "m3_artifacts_mutated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
