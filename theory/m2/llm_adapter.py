"""LLM adapter contract for M2.

The real LLM is intentionally not wired here. This file defines the stable
interface and prompt payload shape consumed by mock or future local generators.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .metric_registry import available_metrics
from .normalizer import allowed_actions_for_frontier
from .schema import LocalContextSummary, RawHypothesisProposal


class LLMHypothesisGenerator(Protocol):
    def generate(
        self,
        frontier_request: Mapping[str, Any],
        local_context_summary: LocalContextSummary | None = None,
    ) -> list[RawHypothesisProposal]:
        ...


def build_local_context_summary(
    frontier_request: Mapping[str, Any],
) -> LocalContextSummary:
    return LocalContextSummary(
        selected_signal=float(frontier_request.get("selected_signal", 0.0) or 0.0),
        fallback_action=str(frontier_request.get("fallback_action", "")),
        fallback_progress=bool(frontier_request.get("fallback_progress", False)),
        known_consumed_skill=str(frontier_request.get("blocked_skill", "")),
        allowed_actions=allowed_actions_for_frontier(frontier_request),
        available_metrics=available_metrics(),
    )


def build_llm_prompt_payload(
    frontier_request: Mapping[str, Any],
    local_context_summary: LocalContextSummary | None = None,
) -> dict[str, Any]:
    context = local_context_summary or build_local_context_summary(frontier_request)
    return {
        "frontier_context_id": str(frontier_request.get("frontier_context_id", "")),
        "reason": str(frontier_request.get("reason", "")),
        "blocked_skill": str(frontier_request.get("blocked_skill", "")),
        "failed_precondition": str(frontier_request.get("failed_precondition", "")),
        "context_signature": list(frontier_request.get("context_signature", []) or []),
        "live_state_signature": str(frontier_request.get("live_state_signature", "")),
        "selected_signal": context.selected_signal,
        "fallback_action": context.fallback_action,
        "fallback_progress": context.fallback_progress,
        "known_consumed_skill": context.known_consumed_skill,
        "allowed_actions": list(context.allowed_actions),
        "available_metrics": list(context.available_metrics),
        "constraints": [
            "do_not_confirm",
            "candidate_hypotheses_only",
            "mark_each_hypothesis_testable_or_non_testable",
            "make_each_hypothesis_falsifiable",
            "maximum_5_hypotheses",
        ],
    }


def truncate_llm_proposals(
    proposals: Sequence[RawHypothesisProposal],
    *,
    max_hypotheses: int = 5,
) -> list[RawHypothesisProposal]:
    return list(proposals[:max_hypotheses])
