"""Operational guards for M2 hypotheses whose target skill is blocked live."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PRECONDITION_RESET_FAMILIES = {
    "precondition_reset_action",
    "state_reset_or_reorientation",
}


def blocked_skill_precondition_still_active(
    *,
    frontier: Mapping[str, Any],
    candidate_action: str,
    hypothesis_family: str,
    context_replay: Sequence[str],
) -> bool:
    """Return true when M2 should prevent READY_FOR_M3 for a blocked skill.

    If A40 says a confirmed skill was blocked by a failed precondition, M2 should
    not spend M3 budget on retesting that same skill unless the hypothesis
    explicitly includes a precondition-reset mechanism.
    """
    blocked_skill = str(frontier.get("blocked_skill", ""))
    failed_precondition = str(frontier.get("failed_precondition", ""))
    if not blocked_skill or not failed_precondition:
        return False
    if str(candidate_action) != blocked_skill:
        return False
    if str(hypothesis_family) in PRECONDITION_RESET_FAMILIES:
        return False
    if "RESET" in {str(item).upper() for item in context_replay}:
        return False
    return True
