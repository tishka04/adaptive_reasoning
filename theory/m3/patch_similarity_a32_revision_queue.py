"""M3.21 candidate-only bridge from patch generativity to A32 review."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.epistemic_metrics import HypothesisRecord, HypothesisStatus

from .dynamic_retarget_patch_similarity_generativity_consolidation import (
    DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH,
    PATCH_GENERATIVITY_STATUS,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "patch_similarity_a32_revision_queue.json"
)

PATCH_SIMILARITY_CANDIDATE_MECHANIC = (
    "repositioning_opens_patch_similar_action6_affordances"
)


@dataclass(frozen=True)
class PatchSimilarityA32RevisionQueueItem:
    """One generative patch-similarity candidate queued for A32 review."""

    queue_item_id: str
    source_generativity_consolidation_id: str
    source_selection_rule_consolidation_id: str
    source_selection_rule_candidate_id: str
    source_mechanism_candidate_id: str
    game_id: str
    hypothesis_key: str
    description: str
    candidate_rule_family: str
    candidate_mechanic: str
    candidate_generativity: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    successful_args_total: Tuple[Dict[str, Any], ...]
    failed_args: Tuple[Dict[str, Any], ...]
    new_expansion_successes: Tuple[Dict[str, Any], ...]
    success_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    changed_pixels_role: str
    pattern_hypothesis: str
    evidence_summary: Dict[str, Any]
    ready_for_a32_revision: bool
    ready_for_a32_revision_is_not_verdict: bool = True
    status: str = "UNRESOLVED"
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0
    observation_counted_as_confirmation: bool = False
    rule_counted_as_confirmation: bool = False
    generative_sequence_counted_as_confirmation: bool = False
    diagnostic_contradictions_counted_as_refutation: bool = False
    a32_queue_item_is_not_verdict: bool = True

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.hypothesis_key,
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
            "source_generativity_consolidation_id": (
                self.source_generativity_consolidation_id
            ),
            "source_selection_rule_consolidation_id": (
                self.source_selection_rule_consolidation_id
            ),
            "source_selection_rule_candidate_id": (
                self.source_selection_rule_candidate_id
            ),
            "source_mechanism_candidate_id": self.source_mechanism_candidate_id,
            "game_id": self.game_id,
            "hypothesis_key": self.hypothesis_key,
            "description": self.description,
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
            "new_expansion_successes": [
                dict(item) for item in self.new_expansion_successes
            ],
            "success_metrics": list(self.success_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "changed_pixels_role": self.changed_pixels_role,
            "pattern_hypothesis": self.pattern_hypothesis,
            "evidence_summary": dict(self.evidence_summary),
            "ready_for_a32_revision": self.ready_for_a32_revision,
            "ready_for_a32_revision_is_not_verdict": (
                self.ready_for_a32_revision_is_not_verdict
            ),
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
            "a32_queue_item_is_not_verdict": self.a32_queue_item_is_not_verdict,
            "candidate_record": candidate_record_dict(self.to_record()),
        }


def run_patch_similarity_a32_revision_queue_generation(
    *,
    generativity_consolidation_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(generativity_consolidation_path)
    items, rejected = build_patch_similarity_a32_revision_queue_items(
        payload,
        source_artifact=str(generativity_consolidation_path),
    )
    return {
        "config": {
            "generativity_consolidation_path": str(generativity_consolidation_path),
            "schema_version": "m3.patch_similarity_a32_revision_queue.v1",
            "inputs_read": ["M3.20"],
            "artifacts_not_modified": ["A32", "A33"],
            "bridge_policy": {
                "execution_performed": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
                "support_forced_to_zero": True,
                "success_metric_contradictions_reject_queue_item": True,
                "diagnostic_contradictions_do_not_refute_queue_item": True,
                "ready_for_a32_revision_is_not_verdict": True,
            },
        },
        "summary": summarize_patch_similarity_a32_revision_queue(items, rejected),
        "queue_items": [item.to_dict() for item in items],
        "rejected_queue_items": [dict(item) for item in rejected],
        "candidate_records": [
            candidate_record_dict(item.to_record()) for item in items
        ],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "observation_counted_as_confirmation": False,
        "rule_counted_as_confirmation": False,
        "generative_sequence_counted_as_confirmation": False,
        "diagnostic_contradictions_counted_as_refutation": False,
    }


def build_patch_similarity_a32_revision_queue_items(
    generativity_payload: Mapping[str, Any],
    *,
    source_artifact: str = "",
) -> tuple[
    Tuple[PatchSimilarityA32RevisionQueueItem, ...],
    Tuple[Dict[str, Any], ...],
]:
    items: list[PatchSimilarityA32RevisionQueueItem] = []
    rejected: list[Dict[str, Any]] = []
    for row in generativity_payload.get("generativity_consolidations", []) or []:
        if not isinstance(row, Mapping):
            continue
        reason = rejection_reason(row)
        if reason:
            rejected.append(
                {
                    "source_generativity_consolidation_id": str(
                        row.get("generativity_consolidation_id", "")
                    ),
                    "intake_status": "REJECTED_CANDIDATE",
                    "reason": reason,
                    "status": "UNRESOLVED",
                    "revision_status": "CANDIDATE_ONLY",
                    "support": 0,
                    "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                    "wrong_confirmations": 0,
                    "a32_write_performed": False,
                    "a33_write_performed": False,
                }
            )
            continue
        items.append(queue_item_from_consolidation(row, source_artifact=source_artifact))
    return tuple(items), tuple(rejected)


def queue_item_from_consolidation(
    row: Mapping[str, Any],
    *,
    source_artifact: str = "",
) -> PatchSimilarityA32RevisionQueueItem:
    game_id = str(row.get("game_id", ""))
    target_action = str(row.get("target_action", ""))
    candidate_rule_family = str(row.get("candidate_rule_family", ""))
    hypothesis_key = "::".join(
        [
            "patch_similarity_rule",
            game_id,
            "ACTION4_ACTION6",
            candidate_rule_family or "unknown_rule",
        ]
    )
    return PatchSimilarityA32RevisionQueueItem(
        queue_item_id=queue_item_id(row),
        source_generativity_consolidation_id=str(
            row.get("generativity_consolidation_id", "")
        ),
        source_selection_rule_consolidation_id=str(
            row.get("source_selection_rule_consolidation_id", "")
        ),
        source_selection_rule_candidate_id=str(
            row.get("source_selection_rule_candidate_id", "")
        ),
        source_mechanism_candidate_id=str(
            row.get("source_mechanism_candidate_id", "")
        ),
        game_id=game_id,
        hypothesis_key=hypothesis_key,
        description=(
            "ACTION4 after ACTION6/ACTION3 may open patch-similar ACTION6 "
            "affordances selected by local_patch_transformability."
        ),
        candidate_rule_family=candidate_rule_family,
        candidate_mechanic=PATCH_SIMILARITY_CANDIDATE_MECHANIC,
        candidate_generativity=str(
            row.get("candidate_generativity", PATCH_GENERATIVITY_STATUS)
        ),
        context_replay=tuple(str(item) for item in row.get("context_replay", []) or []),
        context_replay_args=_context_args_tuple(row.get("context_replay_args")),
        target_action=target_action,
        successful_args_total=tuple(args_tuple(row.get("successful_args_total"))),
        failed_args=tuple(args_tuple(row.get("failed_args"))),
        new_expansion_successes=tuple(args_tuple(row.get("new_expansion_successes"))),
        success_metrics=tuple(str(item) for item in row.get("success_metrics", []) or []),
        diagnostic_metrics=tuple(
            str(item) for item in row.get("diagnostic_metrics", []) or []
        ),
        changed_pixels_role=str(row.get("changed_pixels_role", "")),
        pattern_hypothesis=str(row.get("pattern_hypothesis", "")),
        evidence_summary=evidence_summary_from_consolidation(
            row,
            source_artifact=source_artifact,
        ),
        ready_for_a32_revision=True,
    )


def evidence_summary_from_consolidation(
    row: Mapping[str, Any],
    *,
    source_artifact: str,
) -> Dict[str, Any]:
    return {
        "source_artifact": source_artifact,
        "successful_args_total_count": len(args_tuple(row.get("successful_args_total"))),
        "failed_args_count": len(args_tuple(row.get("failed_args"))),
        "new_expansion_successes_count": len(
            args_tuple(row.get("new_expansion_successes"))
        ),
        "source_success_metric_support_events": int(
            row.get("source_success_metric_support_events", 0) or 0
        ),
        "source_success_metric_contradiction_events": int(
            row.get("source_success_metric_contradiction_events", 0) or 0
        ),
        "source_diagnostic_contradiction_events": int(
            row.get("source_diagnostic_contradiction_events", 0) or 0
        ),
        "controlled_experiments_run": int(
            row.get("source_success_metric_support_events", 0) or 0
        )
        + int(row.get("source_success_metric_contradiction_events", 0) or 0)
        + int(row.get("source_diagnostic_contradiction_events", 0) or 0)
        + int(row.get("source_neutral_events", 0) or 0),
        "success_metrics": list(row.get("success_metrics", []) or []),
        "diagnostic_metrics": list(row.get("diagnostic_metrics", []) or []),
        "changed_pixels_role": str(row.get("changed_pixels_role", "")),
        "candidate_generativity": str(row.get("candidate_generativity", "")),
        "ready_rule": (
            "ready_for_a32_revision_queue and "
            "source_success_metric_contradiction_events==0"
        ),
        "diagnostic_contradictions_counted_as_refutation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def summarize_patch_similarity_a32_revision_queue(
    items: Sequence[PatchSimilarityA32RevisionQueueItem],
    rejected: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "generativity_consolidations_consumed": len(items) + len(rejected),
        "queue_items": len(items),
        "candidate_records": len(items),
        "rejected_queue_items": len(rejected),
        "ready_for_a32_revision_candidates": len(items),
        "ready_for_a32_revision_is_not_verdict": True,
        "success_metric_support_events": sum(
            int(item.evidence_summary.get("source_success_metric_support_events", 0) or 0)
            for item in items
        ),
        "success_metric_contradiction_events": sum(
            int(
                item.evidence_summary.get(
                    "source_success_metric_contradiction_events", 0
                )
                or 0
            )
            for item in items
        ),
        "diagnostic_contradiction_events": sum(
            int(
                item.evidence_summary.get(
                    "source_diagnostic_contradiction_events", 0
                )
                or 0
            )
            for item in items
        ),
        "diagnostic_contradictions_counted_as_refutation": False,
        "changed_pixels_kept_diagnostic": all(
            "changed_pixels" in item.diagnostic_metrics
            and item.changed_pixels_role == "effect_radar_not_success_metric"
            for item in items
        )
        if items
        else True,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_remains_only_verdict_location": True,
        "generative_sequence_counted_as_confirmation": False,
    }


def rejection_reason(row: Mapping[str, Any]) -> str:
    if str(row.get("status", "")).upper() != "UNRESOLVED":
        return "status_not_unresolved"
    if str(row.get("revision_status", "")).upper() != "CANDIDATE_ONLY":
        return "revision_status_not_candidate_only"
    if int(row.get("support", 0) or 0) != 0:
        return "support_must_remain_zero"
    if str(row.get("truth_status", "")) != M3_REFINEMENT_TRUTH_STATUS:
        return "truth_status_not_m3_candidate"
    if bool(row.get("revision_performed", False)):
        return "revision_already_performed"
    if int(row.get("wrong_confirmations", 0) or 0) != 0:
        return "wrong_confirmations_must_remain_zero"
    if not bool(row.get("ready_for_a32_revision_queue", False)):
        return "not_ready_for_a32_revision_queue"
    if not bool(row.get("a32_queue_ready_is_not_verdict", False)):
        return "a32_queue_readiness_not_marked_non_verdict"
    if int(row.get("source_success_metric_contradiction_events", 0) or 0) > 0:
        return "success_metric_contradiction_events_present"
    if str(row.get("candidate_generativity", "")) != PATCH_GENERATIVITY_STATUS:
        return "candidate_generativity_not_supported_patch_expansion"
    if str(row.get("changed_pixels_role", "")) != "effect_radar_not_success_metric":
        return "changed_pixels_role_not_diagnostic"
    return ""


def queue_item_id(row: Mapping[str, Any]) -> str:
    game_token = str(row.get("game_id", "")).split("-", 1)[0] or "unknown_game"
    rule_family = str(row.get("candidate_rule_family", "")) or "unknown_rule"
    return "::".join(["m3_21", game_token, "ACTION4_ACTION6", rule_family])


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


def write_patch_similarity_a32_revision_queue(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH,
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
        description="Build M3.21 patch-similarity A32 revision queue.",
    )
    parser.add_argument(
        "--generativity-consolidation",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_patch_similarity_a32_revision_queue_generation(
        generativity_consolidation_path=args.generativity_consolidation,
    )
    write_patch_similarity_a32_revision_queue(payload, args.out)
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
