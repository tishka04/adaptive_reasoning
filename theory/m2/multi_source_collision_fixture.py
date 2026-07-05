"""Explicit multi-source collision fixture for M2.12d."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .hypothesis_merger import merge_hypotheses
from .normalizer import normalize_raw_proposals
from .schema import M2_TRUTH_STATUS, RawHypothesisProposal


DEFAULT_M2_MULTI_SOURCE_COLLISION_PATH = (
    Path("diagnostics") / "m2" / "multi_source_collision_fixture.json"
)


def build_collision_frontier() -> Dict[str, Any]:
    return {
        "request_id": "frontier::after_ACTION6::step_1::context_not_covered_by_scope",
        "source_step": 1,
        "game_id": "bp35-0a0ad940",
        "frontier_context_id": "after_ACTION6",
        "context_signature": ["ACTION6"],
        "reason": "context_not_covered_by_scope",
        "live_state_signature": "fixture:after_ACTION6",
        "fallback_action": "ACTION3",
        "available_actions": ["ACTION3", "ACTION4", "ACTION6"],
        "ready_for_m1_or_m3": True,
        "status": "OPEN",
    }


def build_collision_proposals(
    frontier: Mapping[str, Any],
) -> tuple[RawHypothesisProposal, ...]:
    base = {
        "source_request_id": str(frontier["request_id"]),
        "game_id": str(frontier["game_id"]),
        "frontier_context_id": str(frontier["frontier_context_id"]),
        "frontier_reason": str(frontier["reason"]),
        "frontier_step": int(frontier["source_step"]),
        "hypothesis_family": "post_consumption_transition",
        "candidate_action": "ACTION4",
        "predicted_metric": "local_patch_before_after",
        "predicted_effect": "ACTION4 may prepare a new target after ACTION6.",
        "suggested_control_actions": ("ACTION3",),
        "required_context_replay": ("ACTION6",),
        "expected_signal_type": "target_action_changes_local_patch_more_than_control",
    }
    return (
        RawHypothesisProposal(
            **{
                **base,
                "proposal_id": "raw::after_ACTION6::heuristic::collision",
                "source": "heuristic",
                "rationale": "Heuristic collision fixture.",
            }
        ),
        RawHypothesisProposal(
            **{
                **base,
                "proposal_id": "raw::after_ACTION6::llm::collision",
                "source": "llm",
                "rationale": "Mock LLM collision fixture.",
            }
        ),
    )


def run_multi_source_collision_fixture() -> Dict[str, Any]:
    frontier = build_collision_frontier()
    proposals = build_collision_proposals(frontier)
    normalized, rejected = normalize_raw_proposals(
        proposals,
        frontiers_by_request_id={str(frontier["request_id"]): frontier},
    )
    merged = merge_hypotheses(
        normalized,
        frontiers_by_request_id={str(frontier["request_id"]): frontier},
    )
    merged_sources_recorded = any(
        set(hypothesis.source_generation.sources) == {"heuristic", "llm"}
        for hypothesis in merged
    )
    return {
        "config": {"schema_version": "m2.multi_source_collision.v1"},
        "frontier": dict(frontier),
        "raw_proposals": [proposal.to_dict() for proposal in proposals],
        "merged_hypotheses": [hypothesis.to_dict() for hypothesis in merged],
        "rejected_proposals": [item.to_dict() for item in rejected],
        "summary": {
            "raw_proposals_seen": len(proposals),
            "normalized_hypotheses": len(normalized),
            "deduplicated_hypotheses": len(merged),
            "merged_sources_recorded": merged_sources_recorded,
            "priority_score_counted_as_support": any(
                hypothesis.source_generation.priority_score_counted_as_support
                for hypothesis in merged
            ),
            "support_unchanged_at_zero": all(
                hypothesis.support == 0 for hypothesis in merged
            ),
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
    }


def write_multi_source_collision_fixture(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_MULTI_SOURCE_COLLISION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
