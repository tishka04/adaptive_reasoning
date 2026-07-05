"""M3.6 candidate-only bridge toward A15-A31 revision.

This module turns M3 planning output into a revision queue. It deliberately
does not revise hypotheses and does not emit confirmations/refutations.
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

from .scientific_planning_loop import DEFAULT_M3_OUTPUT_PATH


DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "a15_revision_queue.json"
)


@dataclass(frozen=True)
class A15RevisionQueueItem:
    """One M3-supported hypothesis queued as candidate input for A15-A31."""

    queue_item_id: str
    game_id: str
    hypothesis_key: str
    description: str
    evidence_summary: Dict[str, Any]
    control_evidence: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    source_planning_artifact: str = ""
    status: str = "UNRESOLVED"
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    contradictions: int = 0
    controlled_test_required: bool = True
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False

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
            "game_id": self.game_id,
            "hypothesis_key": self.hypothesis_key,
            "description": self.description,
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "contradictions": int(self.contradictions),
            "controlled_test_required": self.controlled_test_required,
            "evidence_summary": dict(self.evidence_summary),
            "control_evidence": [dict(row) for row in self.control_evidence],
            "source_planning_artifact": self.source_planning_artifact,
            "a15_a31_candidate": True,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
        }


def run_a15_revision_queue_generation(
    *,
    planning_result_path: str | Path = DEFAULT_M3_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Build a candidate-only A15 revision queue from an M3 planning artifact."""
    payload = _load_json(planning_result_path)
    items = build_a15_revision_queue_items(
        payload,
        source_planning_artifact=str(planning_result_path),
    )
    scores = score_queue_items_by_game(items)
    return {
        "config": {
            "planning_result_path": str(planning_result_path),
        },
        "summary": summarize_a15_revision_queue(items, scores),
        "queue_items": [item.to_dict() for item in items],
        "candidate_records": [candidate_record_dict(item) for item in items],
        "scores_by_game": {
            game_id: score.to_dict() for game_id, score in sorted(scores.items())
        },
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def build_a15_revision_queue_items(
    planning_payload: Mapping[str, Any],
    *,
    source_planning_artifact: str = "",
) -> Tuple[A15RevisionQueueItem, ...]:
    """Select M3 hypotheses ready for A15 candidate review."""
    if not bool(planning_payload.get("propose_ready_for_A15_revision", False)):
        return ()

    entries = [
        dict(item)
        for item in planning_payload.get("updated_ledger_entries", []) or []
        if bool(_entry_ready_for_queue(item))
    ]
    experiments = [
        dict(item) for item in planning_payload.get("controlled_experiments", []) or []
    ]
    ledger_by_key = _ledger_descriptions(planning_payload)
    queue_items: list[A15RevisionQueueItem] = []
    for index, entry in enumerate(entries, start=1):
        key = str(entry.get("key", ""))
        control_rows = [row for row in experiments if row.get("hypothesis_key") == key]
        queue_items.append(
            A15RevisionQueueItem(
                queue_item_id=f"m3_6::{index:04d}::{key}",
                game_id=str(entry.get("game_id", "")),
                hypothesis_key=key,
                description=ledger_by_key.get(key, key),
                evidence_summary=evidence_summary_from_entry(entry),
                control_evidence=tuple(control_evidence_rows(control_rows)),
                source_planning_artifact=source_planning_artifact,
            )
        )
    return tuple(queue_items)


def evidence_summary_from_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "controlled_experiments_run": int(
            entry.get("controlled_experiments_run", 0) or 0
        ),
        "support_events": int(entry.get("support_events", 0) or 0),
        "independent_support_events": int(
            entry.get("independent_support_events", 0) or 0
        ),
        "reused_control_support_events": int(
            entry.get("reused_control_support_events", 0) or 0
        ),
        "contradiction_events": int(entry.get("contradiction_events", 0) or 0),
        "ready_rule": (
            "independent_support_events>=2 and support_events>=3 "
            "and contradiction_events==0"
        ),
    }


def control_evidence_rows(
    experiments: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for experiment in experiments:
        delta = dict(experiment.get("delta", {}) or {})
        rows.append(
            {
                "target_action": str(experiment.get("target_action", "")),
                "control_action": str(experiment.get("control_action", "")),
                "predicted_metric": str(experiment.get("predicted_metric", "")),
                "baseline_signal": float(delta.get("baseline_signal", 0.0) or 0.0),
                "perturbation_signal": float(
                    delta.get("perturbation_signal", 0.0) or 0.0
                ),
                "effect_size": float(delta.get("effect_size", 0.0) or 0.0),
                "direction": str(delta.get("direction", "")),
                "support_events": int(experiment.get("support_events", 0) or 0),
                "independent_support_events": int(
                    experiment.get("independent_support_events", 0) or 0
                ),
                "reused_control_support_events": int(
                    experiment.get("reused_control_support_events", 0) or 0
                ),
                "contradiction_events": int(
                    experiment.get("contradiction_events", 0) or 0
                ),
                "control_reuse_reason": str(
                    experiment.get("control_reuse_reason", "") or ""
                ),
                "status": "UNRESOLVED",
            }
        )
    return tuple(rows)


def score_queue_items_by_game(
    items: Sequence[A15RevisionQueueItem],
) -> Dict[str, Any]:
    records_by_game: Dict[str, list[HypothesisRecord]] = {}
    for item in items:
        records_by_game.setdefault(item.game_id, []).append(item.to_record())
    return {
        game_id: score_beliefs(
            records,
            MechanicsOracle(game_id),
            experiment_actions=sum(record.experiments_spent for record in records),
        )
        for game_id, records in records_by_game.items()
    }


def summarize_a15_revision_queue(
    items: Sequence[A15RevisionQueueItem],
    scores_by_game: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "queue_items": len(items),
        "candidate_records": len(items),
        "ready_for_a15_revision_candidates": len(items),
        "support_events": sum(
            int(item.evidence_summary.get("support_events", 0) or 0)
            for item in items
        ),
        "independent_support_events": sum(
            int(item.evidence_summary.get("independent_support_events", 0) or 0)
            for item in items
        ),
        "reused_control_support_events": sum(
            int(item.evidence_summary.get("reused_control_support_events", 0) or 0)
            for item in items
        ),
        "contradiction_events": sum(
            int(item.evidence_summary.get("contradiction_events", 0) or 0)
            for item in items
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
    }


def candidate_record_dict(item: A15RevisionQueueItem) -> Dict[str, Any]:
    record = item.to_record()
    return {
        "key": record.key,
        "description": record.description,
        "status": record.status.value,
        "support": int(record.support),
        "contradictions": int(record.contradictions),
        "experiments_spent": int(record.experiments_spent),
    }


def write_a15_revision_queue(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _entry_ready_for_queue(entry: Mapping[str, Any]) -> bool:
    support_events = int(entry.get("support_events", 0) or 0)
    independent_support_events = int(
        entry.get("independent_support_events", 0) or 0
    )
    contradiction_events = int(entry.get("contradiction_events", 0) or 0)
    return (
        str(entry.get("status", "")).upper() == "UNRESOLVED"
        and str(entry.get("revision_status", "")).upper() == "CANDIDATE_ONLY"
        and int(entry.get("support", 0) or 0) == 0
        and bool(entry.get("controlled_test_required", True))
        and independent_support_events >= 2
        and support_events >= 3
        and contradiction_events == 0
    )


def _ledger_descriptions(payload: Mapping[str, Any]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    planning_state = dict(payload.get("planning_state", {}) or {})
    for entry in planning_state.get("ledger_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        key = str(entry.get("key", ""))
        if key:
            descriptions[key] = str(entry.get("description", key))
    return descriptions


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the M3.6 candidate-only A15 revision queue.",
    )
    parser.add_argument(
        "--planning-result",
        type=Path,
        default=DEFAULT_M3_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a15_revision_queue_generation(
        planning_result_path=args.planning_result,
    )
    write_a15_revision_queue(payload, args.out)
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
