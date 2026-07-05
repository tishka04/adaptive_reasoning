"""Deterministic M2 heuristic generator."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .normalizer import allowed_actions_for_frontier, default_controls_for_action
from .schema import RawHypothesisProposal


def generate_heuristic_proposals(
    frontier_requests: Sequence[Mapping[str, Any]],
) -> tuple[RawHypothesisProposal, ...]:
    proposals: list[RawHypothesisProposal] = []
    for frontier in frontier_requests:
        proposals.extend(generate_for_frontier(frontier))
    return tuple(proposals)


def generate_for_frontier(
    frontier: Mapping[str, Any],
) -> tuple[RawHypothesisProposal, ...]:
    reason = str(frontier.get("reason", ""))
    if reason == "context_not_covered_by_scope":
        specs = _context_not_covered_specs(frontier)
    elif reason == "confirmed_skill_blocked_by_failed_precondition":
        specs = _blocked_precondition_specs(frontier)
    elif reason == "fallback_no_progress":
        specs = _fallback_no_progress_specs(frontier)
    elif reason in {"cycle_or_dead_end", "cycle_or_dead_end_detected"}:
        specs = _cycle_or_dead_end_specs(frontier)
    else:
        specs = _exploratory_specs(frontier)
    return tuple(
        _proposal_from_spec(frontier, index=index + 1, spec=spec)
        for index, spec in enumerate(specs)
    )


def _context_not_covered_specs(frontier: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actions = allowed_actions_for_frontier(frontier)
    fallback = str(frontier.get("fallback_action", ""))
    first = fallback if fallback in actions else _first_available(actions, "ACTION3")
    second = _first_available(actions, "ACTION4", exclude={first})
    replay = tuple(str(item) for item in frontier.get("context_signature", []) or [])
    return tuple(
        item
        for item in (
            {
                "family": "post_consumption_transition",
                "action": first,
                "metric": "local_patch_before_after",
                "effect": (
                    f"{first} may expose or prepare a new local target after the "
                    "frontier context consumed the previous effect"
                ),
                "rationale": "Uncovered context after a consumed or scoped skill.",
                "replay": replay,
            },
            {
                "family": "new_target_patch_exposure",
                "action": second,
                "metric": "local_patch_before_after",
                "effect": (
                    f"{second} may reveal a new target patch from the uncovered "
                    "frontier state"
                ),
                "rationale": "A different available action should be tested as a target revealer.",
                "replay": replay,
            },
        )
        if item["action"]
    )


def _blocked_precondition_specs(frontier: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actions = allowed_actions_for_frontier(frontier)
    blocked = str(frontier.get("blocked_skill", ""))
    fallback = str(frontier.get("fallback_action", ""))
    exit_action = fallback if fallback in actions and fallback != blocked else _first_available(actions, "ACTION4", exclude={blocked})
    sequence_context = tuple(action for action in (blocked, "ACTION3") if action)
    replay = (blocked,) if blocked else tuple(str(item) for item in frontier.get("context_signature", []) or [])
    return tuple(
        item
        for item in (
            {
                "family": "saturation_escape_action",
                "action": exit_action,
                "metric": "local_patch_before_after",
                "effect": (
                    f"{exit_action} may move the system away from the saturated "
                    "local patch and expose a new target"
                ),
                "rationale": "The confirmed skill is blocked by an active precondition.",
                "replay": replay,
            },
            {
                "family": "skill_sequence_continuation",
                "action": exit_action,
                "metric": "local_patch_before_after",
                "effect": (
                    f"Alternating the consumed context with {exit_action} may "
                    "expose a new target for the blocked skill"
                ),
                "rationale": "A post-consumption sequence may be needed after saturation.",
                "replay": sequence_context or replay,
            },
        )
        if item["action"]
    )


def _fallback_no_progress_specs(frontier: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actions = allowed_actions_for_frontier(frontier)
    fallback = str(frontier.get("fallback_action", ""))
    alternative = _first_available(actions, "ACTION4", "ACTION3", exclude={fallback})
    replay = tuple(str(item) for item in frontier.get("context_signature", []) or [])
    target = alternative or fallback or _first_available(actions, "ACTION3")
    return tuple(
        item
        for item in (
            {
                "family": "fallback_alternative_action",
                "action": target,
                "metric": "changed_pixels",
                "effect": f"{target} may produce progress where the current fallback did not",
                "rationale": "Fallback produced no progress, so another action family should be tested.",
                "replay": replay,
            },
            {
                "family": "metric_shift_needed",
                "action": fallback or target,
                "metric": "contact_graph_before_after",
                "effect": "The useful signal may be nonlocal or topological rather than local patch change",
                "rationale": "No local progress was observed under the fallback action.",
                "replay": replay,
            },
        )
        if item["action"]
    )


def _cycle_or_dead_end_specs(frontier: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actions = allowed_actions_for_frontier(frontier)
    replay = tuple(str(item) for item in frontier.get("context_signature", []) or [])
    first = _first_available(actions, "ACTION4", "ACTION3")
    second = _first_available(actions, "ACTION3", "ACTION4", exclude={first})
    return tuple(
        item
        for item in (
            {
                "family": "cycle_escape_action",
                "action": first,
                "metric": "changed_pixels",
                "effect": f"{first} may exit the repeated or dead-end context",
                "rationale": "A cycle frontier needs an escape action candidate.",
                "replay": replay,
            },
            {
                "family": "branch_preservation_action",
                "action": second,
                "metric": "topology_before_after",
                "effect": f"{second} may preserve or open an alternate branch",
                "rationale": "Avoiding repeated contexts may require a different branch.",
                "replay": replay,
            },
        )
        if item["action"]
    )


def _exploratory_specs(frontier: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actions = allowed_actions_for_frontier(frontier)
    action = _first_available(actions, "ACTION3", "ACTION4")
    replay = tuple(str(item) for item in frontier.get("context_signature", []) or [])
    if not action:
        return ()
    return (
        {
            "family": "frontier_exploratory_probe",
            "action": action,
            "metric": "changed_pixels",
            "effect": f"{action} may produce an observable frontier transition",
            "rationale": "Unknown frontier reason; create a minimal falsifiable probe.",
            "replay": replay,
        },
    )


def _proposal_from_spec(
    frontier: Mapping[str, Any],
    *,
    index: int,
    spec: Mapping[str, Any],
) -> RawHypothesisProposal:
    action = str(spec["action"])
    allowed = allowed_actions_for_frontier(frontier)
    context = str(frontier.get("frontier_context_id", "unknown"))
    request_id = str(frontier.get("request_id", ""))
    proposal_id = f"raw::{context}::heuristic::{index:03d}"
    return RawHypothesisProposal(
        proposal_id=proposal_id,
        source="heuristic",
        source_request_id=request_id,
        game_id=str(frontier.get("game_id", "")),
        frontier_context_id=context,
        frontier_reason=str(frontier.get("reason", "")),
        frontier_step=_optional_int(frontier.get("source_step")),
        hypothesis_family=str(spec["family"]),
        candidate_action=action,
        predicted_metric=str(spec["metric"]),
        predicted_effect=str(spec["effect"]),
        rationale=str(spec["rationale"]),
        suggested_control_actions=default_controls_for_action(action, allowed),
        required_context_replay=tuple(str(item) for item in spec.get("replay", ()) or ()),
        expected_signal_type="target_action_changes_local_patch_more_than_control",
        test_hint=f"Compare {action} against dynamic M3 controls from the same frontier state.",
    )


def _first_available(
    actions: Sequence[str],
    *preferred: str,
    exclude: set[str] | None = None,
) -> str:
    excluded = exclude or set()
    available = [str(action) for action in actions if str(action) not in excluded]
    for item in preferred:
        if item in available:
            return item
    return available[0] if available else ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
