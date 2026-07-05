"""A40 frontier handoff requests.

A40 turns policy-frontier observations into explicit scientific learning
requests. It does not test, confirm, refute, or revise anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.a39.precondition_aware_policy_rollout import (
    DEFAULT_A39_POLICY_ROLLOUT_OUTPUT_PATH,
)


DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH = (
    Path("diagnostics") / "a40" / "frontier_handoff_requests.json"
)
TRUTH_STATUS = "NOT_REEVALUATED_BY_A40"


@dataclass(frozen=True)
class FrontierHandoffRequest:
    """One request to restart scientific learning from a live frontier."""

    request_id: str
    source_step: int
    game_id: str
    key: str
    frontier_context_id: str
    context_signature: Tuple[str, ...]
    reason: str
    reason_categories: Tuple[str, ...]
    recommended_next_scientific_action: str
    live_state_signature: str
    post_action_state_signature: str = ""
    blocked_skill: str = ""
    failed_precondition: str = ""
    fallback_action: str = ""
    fallback_progress: bool = False
    selected_signal: float = 0.0
    ready_for_m1_or_m3: bool = True
    status: str = "OPEN"
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_step": int(self.source_step),
            "game_id": self.game_id,
            "key": self.key,
            "frontier_context_id": self.frontier_context_id,
            "context_signature": list(self.context_signature),
            "reason": self.reason,
            "reason_categories": list(self.reason_categories),
            "recommended_next_scientific_action": self.recommended_next_scientific_action,
            "live_state_signature": self.live_state_signature,
            "post_action_state_signature": self.post_action_state_signature,
            "blocked_skill": self.blocked_skill,
            "failed_precondition": self.failed_precondition,
            "fallback_action": self.fallback_action,
            "fallback_progress": self.fallback_progress,
            "selected_signal": self.selected_signal,
            "ready_for_m1_or_m3": self.ready_for_m1_or_m3,
            "status": self.status,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_frontier_handoff_requests(
    *,
    policy_rollout_path: str | Path = DEFAULT_A39_POLICY_ROLLOUT_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Build frontier requests from an A39 precondition-aware rollout."""
    rollout_payload = _load_json(policy_rollout_path)
    requests = build_frontier_handoff_requests(
        rollout_payload.get("rollout_steps", []) or []
    )
    return {
        "config": {
            "policy_rollout_path": str(policy_rollout_path),
            "inputs_read": ["A39"],
            "artifacts_not_modified": ["A33", "A35", "A38", "A39"],
            "no_environment_execution": True,
            "no_scientific_revision": True,
        },
        "summary": summarize_frontier_requests(requests),
        "frontier_requests": [request.to_dict() for request in requests],
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_frontier_handoff_requests(
    rollout_steps: Sequence[Mapping[str, Any]],
) -> Tuple[FrontierHandoffRequest, ...]:
    """Create one deduplicated request for each frontier rollout step."""
    requests = []
    for step in rollout_steps:
        if not isinstance(step, Mapping):
            continue
        categories = frontier_categories(step)
        if not categories:
            continue
        reason = primary_reason(categories)
        requests.append(
            FrontierHandoffRequest(
                request_id=request_id_for_step(step, reason),
                source_step=int(step.get("step", 0) or 0),
                game_id=str(step.get("game_id", "")),
                key=str(step.get("key", "")),
                frontier_context_id=frontier_context_id(step),
                context_signature=tuple(
                    str(action) for action in step.get("context_signature", []) or []
                ),
                reason=reason,
                reason_categories=tuple(categories),
                recommended_next_scientific_action=recommended_action(reason),
                live_state_signature=str(step.get("state_signature_before", "")),
                post_action_state_signature=str(step.get("state_signature_after", "")),
                blocked_skill=str(step.get("blocked_action", "")),
                failed_precondition=str(step.get("failed_precondition", "")),
                fallback_action=str(step.get("policy_selected_action", "")),
                fallback_progress=bool(step.get("functional_progress", False)),
                selected_signal=float(step.get("selected_signal", 0.0) or 0.0),
            )
        )
    return tuple(requests)


def frontier_categories(step: Mapping[str, Any]) -> Tuple[str, ...]:
    categories = []
    if bool(step.get("blocked_confirmed_mechanic")) or bool(
        step.get("fallback_due_to_failed_precondition")
    ):
        categories.append("blocked_skill")
    if not bool(step.get("context_match")) and not bool(
        step.get("selected_from_confirmed_mechanic")
    ):
        categories.append("uncovered_context")
    if (
        not bool(step.get("selected_from_confirmed_mechanic"))
        and not bool(step.get("functional_progress"))
        and not bool(step.get("useful_new_state"))
    ):
        categories.append("fallback_no_progress")
    if bool(step.get("dead_end_or_cycle")):
        categories.append("cycle_or_dead_end")
    if str(step.get("error", "")):
        categories.append("rollout_error")
    return tuple(dict.fromkeys(categories))


def primary_reason(categories: Sequence[str]) -> str:
    ordered = (
        ("blocked_skill", "confirmed_skill_blocked_by_failed_precondition"),
        ("uncovered_context", "context_not_covered_by_scope"),
        ("fallback_no_progress", "fallback_no_progress"),
        ("cycle_or_dead_end", "cycle_or_dead_end_detected"),
        ("rollout_error", "rollout_error"),
    )
    category_set = set(categories)
    for category, reason in ordered:
        if category in category_set:
            return reason
    return "frontier_observed"


def recommended_action(reason: str) -> str:
    if reason == "confirmed_skill_blocked_by_failed_precondition":
        return "generate_new_candidate_mechanics_from_current_state"
    if reason == "context_not_covered_by_scope":
        return "generate_new_candidate_mechanics_from_current_state"
    if reason == "fallback_no_progress":
        return "design_controlled_experiment_for_fallback_no_progress"
    if reason == "cycle_or_dead_end_detected":
        return "generate_escape_or_recovery_candidate_mechanics"
    return "create_scientific_followup_request"


def frontier_context_id(step: Mapping[str, Any]) -> str:
    blocked_context = str(step.get("blocked_context_id", ""))
    if blocked_context:
        return blocked_context
    context_id = str(step.get("context_id", ""))
    if context_id:
        return context_id
    signature = "_then_".join(str(action) for action in step.get("context_signature", []) or [])
    return f"after_{signature}" if signature else "reset_exact"


def request_id_for_step(step: Mapping[str, Any], reason: str) -> str:
    context = frontier_context_id(step)
    step_index = int(step.get("step", 0) or 0)
    return f"frontier::{context}::step_{step_index}::{reason}"


def summarize_frontier_requests(
    requests: Sequence[FrontierHandoffRequest],
) -> Dict[str, Any]:
    return {
        "frontier_requests_created": len(requests),
        "blocked_skill_frontiers": count_category(requests, "blocked_skill"),
        "uncovered_context_frontiers": count_category(requests, "uncovered_context"),
        "fallback_no_progress_frontiers": count_category(
            requests,
            "fallback_no_progress",
        ),
        "cycle_or_dead_end_frontiers": count_category(requests, "cycle_or_dead_end"),
        "rollout_error_frontiers": count_category(requests, "rollout_error"),
        "ready_for_m1_or_m3": bool(requests),
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def count_category(
    requests: Sequence[FrontierHandoffRequest],
    category: str,
) -> int:
    return len(
        [
            request
            for request in requests
            if str(category) in set(request.reason_categories)
        ]
    )


def write_frontier_handoff_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH,
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
        description="Build A40 frontier handoff requests from A39 rollout.",
    )
    parser.add_argument(
        "--policy-rollout",
        type=Path,
        default=DEFAULT_A39_POLICY_ROLLOUT_OUTPUT_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_frontier_handoff_requests(
        policy_rollout_path=args.policy_rollout,
    )
    write_frontier_handoff_requests(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "truth_status": TRUTH_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
