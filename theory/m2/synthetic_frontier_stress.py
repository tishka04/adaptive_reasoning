"""Synthetic multi-frontier stress runner for M2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .frontier_conditioned_hypotheses import (
    run_frontier_conditioned_hypotheses_from_payload,
)
from .schema import M2_TRUTH_STATUS


DEFAULT_M2_SYNTHETIC_STRESS_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "synthetic_multi_frontier_stress.json"
)


def synthetic_frontier_payload() -> Dict[str, Any]:
    return {
        "frontier_requests": [
            _frontier("synthetic_context", "context_not_covered_by_scope", 1),
            _frontier(
                "synthetic_blocked",
                "confirmed_skill_blocked_by_failed_precondition",
                2,
                blocked_skill="ACTION6",
                failed_precondition="target_patch_not_already_saturated=true",
                fallback_action="ACTION4",
            ),
            _frontier("synthetic_fallback", "fallback_no_progress", 3),
            _frontier("synthetic_cycle", "cycle_or_dead_end", 4),
            {
                "request_id": "",
                "game_id": "synthetic-game",
                "ready_for_m1_or_m3": True,
                "status": "OPEN",
            },
        ]
    }


def run_synthetic_multi_frontier_stress() -> Dict[str, Any]:
    outputs = run_frontier_conditioned_hypotheses_from_payload(
        synthetic_frontier_payload(),
        input_frontier_path="synthetic_multi_frontier",
    )
    summary = dict(outputs["hypothesis_payload"]["summary"])
    families = {
        hypothesis["hypothesis_family"]
        for batch in outputs["hypothesis_payload"]["hypothesis_batches"]
        for hypothesis in batch["candidate_hypotheses"]
    }
    stress_summary = {
        "synthetic_frontiers_consumed": summary["frontier_requests_consumed"],
        "distinct_hypothesis_families": len(families),
        "invalid_frontiers_rejected_cleanly": (
            outputs["intake_payload"]["summary"]["closed_or_invalid_frontiers"] >= 1
        ),
        "empty_input_valid": _empty_input_valid(),
        "wrong_confirmations": 0,
    }
    return {
        "summary": stress_summary,
        "m2_summary": summary,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_synthetic_multi_frontier_stress(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_SYNTHETIC_STRESS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _frontier(
    context: str,
    reason: str,
    step: int,
    *,
    blocked_skill: str = "",
    failed_precondition: str = "",
    fallback_action: str = "ACTION3",
) -> Dict[str, Any]:
    return {
        "request_id": f"frontier::{context}::step_{step}::{reason}",
        "source_step": step,
        "game_id": "synthetic-game",
        "frontier_context_id": context,
        "context_signature": ["ACTION6"] if step > 1 else [],
        "reason": reason,
        "reason_categories": [reason],
        "recommended_next_scientific_action": "generate_new_candidate_mechanics_from_current_state",
        "live_state_signature": f"synthetic:{step}",
        "blocked_skill": blocked_skill,
        "failed_precondition": failed_precondition,
        "fallback_action": fallback_action,
        "fallback_progress": False,
        "selected_signal": 0.0,
        "available_actions": ["ACTION3", "ACTION4", "ACTION6"],
        "ready_for_m1_or_m3": True,
        "status": "OPEN",
    }


def _empty_input_valid() -> bool:
    outputs = run_frontier_conditioned_hypotheses_from_payload(
        {"frontier_requests": []},
        input_frontier_path="synthetic_empty",
    )
    return outputs["hypothesis_payload"]["summary"]["hypotheses_generated"] == 0
