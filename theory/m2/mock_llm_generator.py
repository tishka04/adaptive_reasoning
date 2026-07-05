"""Mock LLM generator for M2 contract and adversarial tests."""

from __future__ import annotations

from typing import Any, Mapping

from .llm_adapter import (
    LLMHypothesisGenerator,
    build_local_context_summary,
    truncate_llm_proposals,
)
from .normalizer import allowed_actions_for_frontier, default_controls_for_action
from .schema import LocalContextSummary, RawHypothesisProposal


class MockLLMGenerator(LLMHypothesisGenerator):
    """Deterministic stand-in for a future local/offline LLM."""

    def __init__(self, mode: str = "normal", max_hypotheses: int = 5):
        self.mode = mode
        self.max_hypotheses = max_hypotheses

    def generate(
        self,
        frontier_request: Mapping[str, Any],
        local_context_summary: LocalContextSummary | None = None,
    ) -> list[RawHypothesisProposal]:
        context = local_context_summary or build_local_context_summary(frontier_request)
        if self.mode == "many":
            proposals = [
                self._proposal(frontier_request, context, index=index + 1)
                for index in range(50)
            ]
            return truncate_llm_proposals(
                proposals,
                max_hypotheses=self.max_hypotheses,
            )
        proposal = self._proposal(frontier_request, context, index=1)
        if self.mode == "confirmed":
            proposal = _replace_raw_status(proposal, raw_status="CONFIRMED")
        if self.mode == "action9":
            proposal = _replace_candidate(proposal, candidate_action="ACTION9")
        if self.mode == "unknown_metric":
            proposal = _replace_metric(proposal, predicted_metric="soft_unmeasured_metric")
        if self.mode == "support_one":
            proposal = _replace_support(proposal, raw_support=1)
        if self.mode == "truth_confirmed":
            proposal = _replace_truth(proposal, raw_truth_status="CONFIRMED")
        if self.mode == "revision_true":
            proposal = _replace_revision(proposal, raw_revision_performed=True)
        return [proposal]

    def _proposal(
        self,
        frontier_request: Mapping[str, Any],
        context: LocalContextSummary,
        *,
        index: int,
    ) -> RawHypothesisProposal:
        actions = allowed_actions_for_frontier(frontier_request)
        action = "ACTION4" if "ACTION4" in actions else actions[0]
        replay = (
            tuple(str(item) for item in frontier_request.get("context_signature", []) or [])
            or (str(frontier_request.get("blocked_skill", "")),)
        )
        replay = tuple(item for item in replay if item)
        frontier_context = str(frontier_request.get("frontier_context_id", "unknown"))
        return RawHypothesisProposal(
            proposal_id=f"raw::{frontier_context}::llm::{index:03d}",
            source="llm",
            source_request_id=str(frontier_request.get("request_id", "")),
            game_id=str(frontier_request.get("game_id", "")),
            frontier_context_id=frontier_context,
            frontier_reason=str(frontier_request.get("reason", "")),
            frontier_step=_optional_int(frontier_request.get("source_step")),
            hypothesis_family="post_consumption_transition",
            candidate_action=action,
            predicted_metric="local_patch_before_after",
            predicted_effect=(
                f"{action} may move the system away from the saturated or "
                "uncovered local context"
            ),
            rationale=(
                "The current frontier lacks a confirmed reusable action; this is "
                "a candidate only."
            ),
            suggested_control_actions=default_controls_for_action(action, actions),
            required_context_replay=replay,
            expected_signal_type="target_action_changes_local_patch_more_than_control",
            test_hint=f"Compare {action} against dynamic M3 controls.",
        )


def _replace_raw_status(
    proposal: RawHypothesisProposal,
    *,
    raw_status: str,
) -> RawHypothesisProposal:
    return RawHypothesisProposal(**{**proposal.to_dict(), "raw_status": raw_status})


def _replace_candidate(
    proposal: RawHypothesisProposal,
    *,
    candidate_action: str,
) -> RawHypothesisProposal:
    return RawHypothesisProposal(**{**proposal.to_dict(), "candidate_action": candidate_action})


def _replace_metric(
    proposal: RawHypothesisProposal,
    *,
    predicted_metric: str,
) -> RawHypothesisProposal:
    return RawHypothesisProposal(**{**proposal.to_dict(), "predicted_metric": predicted_metric})


def _replace_support(
    proposal: RawHypothesisProposal,
    *,
    raw_support: int,
) -> RawHypothesisProposal:
    return RawHypothesisProposal(**{**proposal.to_dict(), "raw_support": raw_support})


def _replace_truth(
    proposal: RawHypothesisProposal,
    *,
    raw_truth_status: str,
) -> RawHypothesisProposal:
    return RawHypothesisProposal(**{**proposal.to_dict(), "raw_truth_status": raw_truth_status})


def _replace_revision(
    proposal: RawHypothesisProposal,
    *,
    raw_revision_performed: bool,
) -> RawHypothesisProposal:
    return RawHypothesisProposal(
        **{**proposal.to_dict(), "raw_revision_performed": raw_revision_performed}
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
