"""Normalize raw M2 proposals into guarded frontier-conditioned hypotheses."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from .context_replay import canonical_context_replay, context_replay_warnings
from .metric_registry import (
    UNLOCK_HYPOTHESIS_FAMILY,
    default_falsification_for_metric,
    is_metric_executor_supported,
    is_metric_measurable,
    is_unlock_only_metric,
    metric_executor_block_reason,
)
from .precondition_guard import blocked_skill_precondition_still_active
from .schema import (
    ContextSnapshot,
    FalsificationCriterion,
    FrontierConditionedHypothesis,
    HypothesisTestability,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_HYPOTHESIS_STATUS,
    M2_TRUTH_STATUS,
    RawHypothesisProposal,
    RejectedProposal,
    SourceGenerationAudit,
)
from .validators import validate_hypothesis


DEFAULT_ALLOWED_ACTIONS_BY_GAME = {
    "bp35-0a0ad940": ("ACTION3", "ACTION4", "ACTION6"),
}
FALLBACK_ALLOWED_ACTIONS = (
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
    "ACTION8",
)


def normalize_raw_proposal(
    proposal: RawHypothesisProposal | Mapping[str, Any],
    *,
    frontier_request: Mapping[str, Any] | None = None,
    allowed_actions: Sequence[str] | None = None,
) -> FrontierConditionedHypothesis | RejectedProposal:
    data = _proposal_data(proposal)
    proposal_id = str(data.get("proposal_id", data.get("id", "")) or "")
    source = str(data.get("source", ""))
    source_request_id = str(
        data.get("source_request_id")
        or (frontier_request or {}).get("request_id", "")
    )

    if not proposal_id:
        proposal_id = f"raw::{source or 'unknown'}::{source_request_id or 'missing'}"
    if source not in {"heuristic", "llm", "world_model"}:
        return RejectedProposal(
            proposal_id=proposal_id,
            source=source,
            source_request_id=source_request_id,
            reason="invalid_source",
        )
    if bool(data.get("raw_revision_performed", data.get("revision_performed", False))):
        return RejectedProposal(
            proposal_id=proposal_id,
            source=source,
            source_request_id=source_request_id,
            reason="raw_revision_performed_true",
        )

    game_id = str(data.get("game_id") or (frontier_request or {}).get("game_id", ""))
    frontier_context_id = str(
        data.get("frontier_context_id")
        or (frontier_request or {}).get("frontier_context_id", "")
    )
    frontier_reason = str(
        data.get("frontier_reason") or (frontier_request or {}).get("reason", "")
    )
    frontier_step = _optional_int(
        data.get("frontier_step", (frontier_request or {}).get("source_step"))
    )
    action = str(data.get("candidate_action", ""))
    metric = str(data.get("predicted_metric", "local_patch_before_after"))

    allowed = tuple(str(item) for item in (allowed_actions or ()))
    if not allowed:
        allowed = allowed_actions_for_frontier(frontier_request or {}, game_id=game_id)
    if allowed and action not in set(allowed):
        return RejectedProposal(
            proposal_id=proposal_id,
            source=source,
            source_request_id=source_request_id,
            reason="unknown_candidate_action",
            details=action,
        )
    if not is_metric_measurable(metric):
        return RejectedProposal(
            proposal_id=proposal_id,
            source=source,
            source_request_id=source_request_id,
            reason="unknown_metric",
            details=metric,
        )
    hypothesis_family = str(data.get("hypothesis_family", ""))
    if is_unlock_only_metric(metric) and hypothesis_family != UNLOCK_HYPOTHESIS_FAMILY:
        return RejectedProposal(
            proposal_id=proposal_id,
            source=source,
            source_request_id=source_request_id,
            reason="metric_requires_affordance_or_unlock_family",
            details=metric,
        )
    unlock_target_actions = tuple(
        str(item)
        for item in data.get("unlock_target_actions", []) or ()
        if str(item)
    )

    warnings = normalization_warnings(data)
    proposed_replay = tuple(
        str(item)
        for item in (
            data.get("required_context_replay")
            or _default_context_replay(frontier_request or {})
        )
        or ()
    )
    replay_actions = canonical_context_replay(
        frontier_request or {},
        proposed_replay=proposed_replay,
    )
    warnings.extend(
        context_replay_warnings(
            frontier_request or {},
            proposed_replay=proposed_replay,
            canonical_replay=replay_actions,
        )
    )
    controls = tuple(
        str(item)
        for item in data.get("suggested_control_actions", []) or []
        if str(item) and str(item) != action
    )
    if not controls:
        controls = default_controls_for_action(action, allowed)
        warnings.append("suggested_controls_filled_from_available_actions")

    falsification = _falsification_from_data(data.get("falsification"), metric)
    expected_signal = str(
        data.get("expected_signal_type")
        or data.get("expected_signal")
        or "target_action_changes_local_patch_more_than_control"
    )
    testable = bool(action and metric and replay_actions is not None)
    blocking_reason = None if testable else "missing_testable_fields"
    executor_block_reason = metric_executor_block_reason(metric)
    if testable and executor_block_reason is not None:
        testable = False
        blocking_reason = executor_block_reason
        warnings.append(executor_block_reason)
    if blocked_skill_precondition_still_active(
        frontier=frontier_request or {},
        candidate_action=action,
        hypothesis_family=hypothesis_family,
        context_replay=replay_actions,
    ):
        testable = False
        blocking_reason = "blocked_skill_precondition_still_active"
        warnings.append("blocked_skill_precondition_guard_applied")
    testability = HypothesisTestability(
        testable=testable,
        recommended_test_type="controlled_action_vs_alternative",
        target_action=action,
        suggested_control_actions=controls,
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=metric,
        required_context_replay=replay_actions,
        required_action_args=_tuple_dicts_or_none(data.get("required_action_args")),
        expected_signal_type=expected_signal,
        measurable_by_existing_extractor=is_metric_executor_supported(metric),
        blocking_reason=blocking_reason,
    )
    context_snapshot = ContextSnapshot(
        replay_actions=replay_actions,
        replay_action_args=testability.required_action_args,
        frame_before_hash=_optional_str(
            data.get("frame_before_hash")
            or (frontier_request or {}).get("frame_before_hash")
        ),
        live_state_signature=_optional_str(
            data.get("live_state_signature")
            or (frontier_request or {}).get("live_state_signature")
        ),
        available_actions=allowed,
        local_patch=_local_patch(data.get("local_patch")),
        terminal_state=bool(data.get("terminal_state", False)),
    )
    rationale = str(data.get("rationale", ""))
    audit = SourceGenerationAudit(
        sources=(source,),
        raw_proposal_ids=(proposal_id,),
        rationales=(rationale,) if rationale else (),
        normalization_warnings=tuple(warnings),
        priority_score=float(data.get("priority_hint", 0.0) or 0.0),
        priority_score_counted_as_support=False,
    )
    hypothesis = FrontierConditionedHypothesis(
        hypothesis_id=str(
            data.get("hypothesis_id")
            or f"m2::{frontier_context_id or 'unknown'}::{proposal_id}"
        ),
        source_request_id=source_request_id,
        game_id=game_id,
        frontier_context_id=frontier_context_id,
        frontier_reason=frontier_reason,
        frontier_step=frontier_step,
        hypothesis_family=hypothesis_family,
        candidate_action=action,
        predicted_metric=metric,
        predicted_effect=str(data.get("predicted_effect", "")),
        rationale=rationale,
        testability=testability,
        falsification=falsification,
        context_snapshot=context_snapshot,
        source_generation=audit,
        unlock_target_actions=unlock_target_actions,
        status=M2_HYPOTHESIS_STATUS,
        support=0,
        controlled_test_required=True,
        truth_status=M2_TRUTH_STATUS,
        revision_performed=False,
        wrong_confirmations=0,
        trace_support_counted_as_proof=False,
        prior_counted_as_proof=False,
    )
    result = validate_hypothesis(hypothesis)
    if not result.valid:
        return RejectedProposal(
            proposal_id=proposal_id,
            source=source,
            source_request_id=source_request_id,
            reason="normalized_hypothesis_invalid",
            details=";".join(result.errors),
        )
    return hypothesis


def normalize_raw_proposals(
    proposals: Sequence[RawHypothesisProposal | Mapping[str, Any]],
    *,
    frontiers_by_request_id: Mapping[str, Mapping[str, Any]] | None = None,
    max_proposals: int | None = None,
) -> tuple[tuple[FrontierConditionedHypothesis, ...], tuple[RejectedProposal, ...]]:
    normalized: list[FrontierConditionedHypothesis] = []
    rejected: list[RejectedProposal] = []
    seen = 0
    for proposal in proposals:
        if max_proposals is not None and seen >= max_proposals:
            break
        seen += 1
        data = _proposal_data(proposal)
        request_id = str(data.get("source_request_id", ""))
        frontier = (frontiers_by_request_id or {}).get(request_id, {})
        result = normalize_raw_proposal(proposal, frontier_request=frontier)
        if isinstance(result, RejectedProposal):
            rejected.append(result)
        else:
            normalized.append(result)
    return tuple(normalized), tuple(rejected)


def raw_proposals_from_json_text(text: str) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ()
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, Mapping))
    if isinstance(payload, Mapping):
        rows = payload.get("hypotheses") or payload.get("proposals") or []
        if isinstance(rows, list):
            return tuple(item for item in rows if isinstance(item, Mapping))
    return ()


def allowed_actions_for_frontier(
    frontier: Mapping[str, Any],
    *,
    game_id: str | None = None,
) -> tuple[str, ...]:
    explicit = frontier.get("available_actions", []) or []
    if explicit:
        return tuple(str(item) for item in explicit if str(item))
    gid = str(game_id or frontier.get("game_id", ""))
    if gid in DEFAULT_ALLOWED_ACTIONS_BY_GAME:
        return DEFAULT_ALLOWED_ACTIONS_BY_GAME[gid]
    actions = []
    for field in ("fallback_action", "blocked_skill"):
        action = str(frontier.get(field, ""))
        if action:
            actions.append(action)
    actions.extend(str(item) for item in frontier.get("context_signature", []) or [])
    if actions:
        return tuple(dict.fromkeys(actions))
    return FALLBACK_ALLOWED_ACTIONS


def default_controls_for_action(
    target_action: str,
    allowed_actions: Sequence[str],
) -> tuple[str, ...]:
    target = str(target_action)
    preferred = ("ACTION3", "ACTION4", "ACTION1", "ACTION2", "ACTION5", "ACTION6")
    allowed = [str(action) for action in allowed_actions if str(action) != target]
    ordered = [action for action in preferred if action in set(allowed)]
    ordered.extend(action for action in sorted(allowed) if action not in set(ordered))
    return tuple(ordered)


def normalization_warnings(data: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if str(data.get("raw_status", data.get("status", ""))).upper() not in {
        "",
        M2_HYPOTHESIS_STATUS,
    }:
        warnings.append("status_forced_unresolved")
    if int(data.get("raw_support", data.get("support", 0)) or 0) != 0:
        warnings.append("support_forced_zero")
    if str(data.get("raw_truth_status", data.get("truth_status", ""))) not in {
        "",
        M2_TRUTH_STATUS,
    }:
        warnings.append("truth_status_forced_not_evaluated_by_m2")
    return warnings


def _falsification_from_data(
    value: Any,
    metric: str,
) -> FalsificationCriterion:
    if isinstance(value, Mapping):
        return FalsificationCriterion(
            metric=str(value.get("metric", metric)),
            support_condition=str(
                value.get(
                    "support_condition",
                    "target_action_signal > best_control_signal",
                )
            ),
            failure_condition=str(
                value.get(
                    "failure_condition",
                    "target_action_signal <= best_control_signal",
                )
            ),
            minimum_effect_size=value.get("minimum_effect_size", 1),
        )
    return default_falsification_for_metric(metric)


def _default_context_replay(frontier: Mapping[str, Any]) -> tuple[str, ...]:
    reason = str(frontier.get("reason", ""))
    blocked = str(frontier.get("blocked_skill", ""))
    if reason == "confirmed_skill_blocked_by_failed_precondition" and blocked:
        return (blocked,)
    return tuple(str(item) for item in frontier.get("context_signature", []) or [])


def _proposal_data(proposal: RawHypothesisProposal | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(proposal, RawHypothesisProposal):
        return proposal.to_dict()
    if isinstance(proposal, Mapping):
        return dict(proposal)
    return {}


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _tuple_dicts_or_none(value: Any) -> tuple[Dict[str, Any], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _local_patch(value: Any) -> tuple[tuple[int, ...], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    rows = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return None
        rows.append(tuple(int(cell) for cell in row))
    return tuple(rows)
