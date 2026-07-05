"""A32.2 intake for patch-similarity generative candidates.

This module only decides whether a rich M3.21 queue item is admissible for a
later scientific revision. It does not confirm, refute, or write A33.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus
from theory.m3.patch_similarity_a32_revision_queue import (
    DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH,
)


DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH = (
    Path("diagnostics") / "a32" / "patch_similarity_revision_intake.json"
)

ACCEPTED_FOR_SCIENTIFIC_REVISION = "ACCEPTED_FOR_SCIENTIFIC_REVISION"
REJECTED_FROM_INTAKE = "REJECTED_FROM_INTAKE"


@dataclass(frozen=True)
class A32PatchSimilarityRevisionIntakeCandidate:
    """One M3.21 queue item accepted as an unresolved scientific candidate."""

    queue_item_id: str
    game_id: str
    key: str
    description: str
    source_generativity_consolidation_id: str
    candidate_rule_family: str
    candidate_mechanic: str
    candidate_generativity: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    successful_args_total: Tuple[Dict[str, Any], ...]
    failed_args: Tuple[Dict[str, Any], ...]
    success_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    changed_pixels_role: str
    evidence_summary: Dict[str, Any]
    intake_status: str = ACCEPTED_FOR_SCIENTIFIC_REVISION
    consumer_contract: str = "HypothesisRecord"
    requested_next_step: str = "A15_A31_PATCH_SIMILARITY_REVIEW_REQUIRED"
    allowed_revision_outcomes: Tuple[str, ...] = (
        "confirm_after_scope_limited_revision",
        "refute_after_revision",
        "request_more_tests_with_scope_limits",
        "scope_limited_candidate_only",
    )
    status: str = "UNRESOLVED"
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = "NOT_EVALUATED_BY_A32_INTAKE"
    revision_performed: bool = False
    wrong_confirmations: int = 0
    observation_counted_as_confirmation: bool = False
    rule_counted_as_confirmation: bool = False
    generative_sequence_counted_as_confirmation: bool = False
    diagnostic_contradictions_counted_as_refutation: bool = False
    ready_for_revision_is_not_verdict: bool = True

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
            "source_generativity_consolidation_id": (
                self.source_generativity_consolidation_id
            ),
            "candidate_rule_family": self.candidate_rule_family,
            "candidate_mechanic": self.candidate_mechanic,
            "candidate_generativity": self.candidate_generativity,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "successful_args_total": [
                dict(item) for item in self.successful_args_total
            ],
            "failed_args": [dict(item) for item in self.failed_args],
            "success_metrics": list(self.success_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "changed_pixels_role": self.changed_pixels_role,
            "evidence_summary": dict(self.evidence_summary),
            "intake_status": self.intake_status,
            "consumer_contract": self.consumer_contract,
            "requested_next_step": self.requested_next_step,
            "allowed_revision_outcomes": list(self.allowed_revision_outcomes),
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
            "rule_counted_as_confirmation": self.rule_counted_as_confirmation,
            "generative_sequence_counted_as_confirmation": (
                self.generative_sequence_counted_as_confirmation
            ),
            "diagnostic_contradictions_counted_as_refutation": (
                self.diagnostic_contradictions_counted_as_refutation
            ),
            "ready_for_revision_is_not_verdict": self.ready_for_revision_is_not_verdict,
            "candidate_record": candidate_record_dict(self.to_record()),
        }


def run_a32_patch_similarity_revision_intake(
    *,
    queue_path: str | Path = DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(queue_path)
    accepted, rejected = build_a32_patch_similarity_revision_intake_candidates(payload)
    return {
        "config": {
            "queue_path": str(queue_path),
            "schema_version": "a32.patch_similarity_revision_intake.v1",
            "inputs_read": ["M3.21"],
            "output_scope": "A32_INTAKE_ONLY",
            "artifacts_not_modified": ["M3", "A33"],
            "intake_policy": {
                "produce_hypothesis_records_only": True,
                "revision_performed": False,
                "confirmation_performed": False,
                "refutation_performed": False,
                "a33_write_performed": False,
                "diagnostic_contradictions_do_not_refute": True,
            },
        },
        "summary": summarize_patch_similarity_revision_intake(accepted, rejected),
        "accepted_candidates": [item.to_dict() for item in accepted],
        "rejected_candidates": [dict(item) for item in rejected],
        "candidate_records": [
            candidate_record_dict(item.to_record()) for item in accepted
        ],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": "NOT_EVALUATED_BY_A32_INTAKE",
        "revision_performed": False,
        "hypotheses_confirmed": 0,
        "hypotheses_refuted": 0,
        "wrong_confirmations": 0,
        "observation_counted_as_confirmation": False,
        "rule_counted_as_confirmation": False,
        "generative_sequence_counted_as_confirmation": False,
        "diagnostic_contradictions_counted_as_refutation": False,
        "a33_write_performed": False,
    }


def build_a32_patch_similarity_revision_intake_candidates(
    queue_payload: Mapping[str, Any],
) -> tuple[
    Tuple[A32PatchSimilarityRevisionIntakeCandidate, ...],
    Tuple[Dict[str, Any], ...],
]:
    accepted: list[A32PatchSimilarityRevisionIntakeCandidate] = []
    rejected: list[Dict[str, Any]] = []
    for item in queue_payload.get("queue_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        reason = rejection_reason(item, queue_payload=queue_payload)
        if reason:
            rejected.append(
                {
                    "queue_item_id": str(item.get("queue_item_id", "")),
                    "hypothesis_key": str(item.get("hypothesis_key", "")),
                    "intake_status": REJECTED_FROM_INTAKE,
                    "reason": reason,
                    "status": "UNRESOLVED",
                    "revision_status": "CANDIDATE_ONLY",
                    "support": 0,
                    "truth_status": "NOT_EVALUATED_BY_A32_INTAKE",
                    "revision_performed": False,
                    "wrong_confirmations": 0,
                    "a33_write_performed": False,
                }
            )
            continue
        accepted.append(candidate_from_queue_item(item))
    return tuple(accepted), tuple(rejected)


def candidate_from_queue_item(
    item: Mapping[str, Any],
) -> A32PatchSimilarityRevisionIntakeCandidate:
    evidence = dict(item.get("evidence_summary", {}) or {})
    return A32PatchSimilarityRevisionIntakeCandidate(
        queue_item_id=str(item.get("queue_item_id", "")),
        game_id=str(item.get("game_id", "")),
        key=str(item.get("hypothesis_key", "")),
        description=str(item.get("description", "")),
        source_generativity_consolidation_id=str(
            item.get("source_generativity_consolidation_id", "")
        ),
        candidate_rule_family=str(item.get("candidate_rule_family", "")),
        candidate_mechanic=str(item.get("candidate_mechanic", "")),
        candidate_generativity=str(item.get("candidate_generativity", "")),
        context_replay=tuple(str(action) for action in item.get("context_replay", []) or []),
        context_replay_args=_context_args_tuple(item.get("context_replay_args")),
        target_action=str(item.get("target_action", "")),
        successful_args_total=args_tuple(item.get("successful_args_total")),
        failed_args=args_tuple(item.get("failed_args")),
        success_metrics=tuple(str(metric) for metric in item.get("success_metrics", []) or []),
        diagnostic_metrics=tuple(
            str(metric) for metric in item.get("diagnostic_metrics", []) or []
        ),
        changed_pixels_role=str(item.get("changed_pixels_role", "")),
        evidence_summary=evidence,
    )


def rejection_reason(
    item: Mapping[str, Any],
    *,
    queue_payload: Mapping[str, Any],
) -> str:
    evidence = dict(item.get("evidence_summary", {}) or {})
    summary = dict(queue_payload.get("summary", {}) or {})
    if bool(summary.get("a33_write_performed", False)):
        return "a33_write_attempted"
    if bool(item.get("a33_write_performed", False)):
        return "a33_write_attempted"
    if str(item.get("status", "")).upper() != "UNRESOLVED":
        return "status_not_unresolved"
    if str(item.get("revision_status", "")).upper() != "CANDIDATE_ONLY":
        return "revision_status_not_candidate_only"
    if int(item.get("support", 0) or 0) != 0:
        return "support_must_remain_zero"
    if not bool(item.get("ready_for_a32_revision", False)):
        return "not_ready_for_a32_revision"
    if not bool(item.get("ready_for_a32_revision_is_not_verdict", False)):
        return "ready_for_a32_revision_not_marked_non_verdict"
    if bool(item.get("revision_performed", False)):
        return "revision_already_performed"
    if int(item.get("wrong_confirmations", 0) or 0) != 0:
        return "wrong_confirmations_must_remain_zero"
    if source_success_metric_contradictions(evidence) > 0:
        return "success_metric_contradiction_events_present"
    if int(evidence.get("successful_args_total_count", 0) or 0) < 2:
        return "insufficient_successful_args"
    if int(evidence.get("failed_args_count", 0) or 0) < 1:
        return "missing_negative_case"
    if str(item.get("changed_pixels_role", "")) != "effect_radar_not_success_metric":
        return "changed_pixels_role_not_diagnostic"
    if bool(item.get("diagnostic_contradictions_counted_as_refutation", False)):
        return "diagnostic_contradiction_interpreted_as_refutation"
    if bool(evidence.get("diagnostic_contradictions_counted_as_refutation", False)):
        return "diagnostic_contradiction_interpreted_as_refutation"
    return ""


def summarize_patch_similarity_revision_intake(
    accepted: Sequence[A32PatchSimilarityRevisionIntakeCandidate],
    rejected: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "queue_items_consumed": len(accepted) + len(rejected),
        "accepted_for_scientific_revision": len(accepted),
        "rejected_from_intake": len(rejected),
        "candidate_records": len(accepted),
        "successful_args_total_count": sum(
            int(item.evidence_summary.get("successful_args_total_count", 0) or 0)
            for item in accepted
        ),
        "failed_args_count": sum(
            int(item.evidence_summary.get("failed_args_count", 0) or 0)
            for item in accepted
        ),
        "source_success_metric_support_events": sum(
            int(item.evidence_summary.get("source_success_metric_support_events", 0) or 0)
            for item in accepted
        ),
        "source_success_metric_contradiction_events": sum(
            source_success_metric_contradictions(item.evidence_summary)
            for item in accepted
        ),
        "source_diagnostic_contradiction_events": sum(
            int(
                item.evidence_summary.get("source_diagnostic_contradiction_events", 0)
                or 0
            )
            for item in accepted
        ),
        "diagnostic_contradictions_counted_as_refutation": False,
        "changed_pixels_kept_diagnostic": all(
            item.changed_pixels_role == "effect_radar_not_success_metric"
            and "changed_pixels" in item.diagnostic_metrics
            for item in accepted
        )
        if accepted
        else True,
        "hypothesis_records_unresolved": len(accepted),
        "requested_next_step": (
            "A15_A31_PATCH_SIMILARITY_REVIEW_REQUIRED" if accepted else ""
        ),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": "NOT_EVALUATED_BY_A32_INTAKE",
        "revision_performed": False,
        "hypotheses_confirmed": 0,
        "hypotheses_refuted": 0,
        "wrong_confirmations": 0,
        "a33_write_performed": False,
        "m3_artifacts_mutated": False,
    }


def source_success_metric_contradictions(evidence: Mapping[str, Any]) -> int:
    return int(
        evidence.get(
            "source_success_metric_contradiction_events",
            evidence.get("success_metric_contradiction_events", 0),
        )
        or 0
    )


def args_tuple(raw: Any) -> Tuple[Dict[str, Any], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def candidate_record_dict(record: HypothesisRecord) -> Dict[str, Any]:
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a32_patch_similarity_revision_intake(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _context_args_tuple(raw: Any) -> Tuple[Dict[str, Any], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A32.2 patch-similarity revision intake.",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_patch_similarity_revision_intake(queue_path=args.queue)
    write_a32_patch_similarity_revision_intake(payload, args.out)
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
