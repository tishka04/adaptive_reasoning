"""Loop guard for repeated SAGE policy fallback actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


DEFAULT_SAGE1B_POLICY_LOOP_GUARD_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage1b_policy_loop_guard_results.json"
)
SAGE1B_SCHEMA_VERSION = "sage.policy_loop_guard_results.v1"
SAGE1B_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_1B"
LOOP_GUARD_SWITCH_REASON = "policy_loop_guard_switch_after_exhaustion"
LOOP_GUARD_PASS_REASON = "policy_loop_guard_not_triggered"
LOOP_GUARD_BLOCKED_REASON = "policy_loop_guard_no_legal_switch_available"


@dataclass(frozen=True)
class PolicyLoopGuardDecision:
    """Decision metadata after checking a proposed policy action."""

    selected_action_raw: Any | None
    loop_guard_enabled: bool = True
    loop_guard_triggered: bool = False
    repeated_same_action_args_detected: bool = False
    fallback_loop_interrupted: bool = False
    switch_action_selected_after_exhaustion: bool = False
    selected_switch_action: str = ""
    selected_switch_action_args: Dict[str, Any] = field(default_factory=dict)
    blocked_action: str = ""
    blocked_action_args: Dict[str, Any] = field(default_factory=dict)
    consecutive_repeats_before: int = 0
    max_same_action_arg_repeats: int = 0
    loop_guard_reason: str = LOOP_GUARD_PASS_REASON

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_guard_enabled": self.loop_guard_enabled,
            "loop_guard_triggered": self.loop_guard_triggered,
            "repeated_same_action_args_detected": (
                self.repeated_same_action_args_detected
            ),
            "fallback_loop_interrupted": self.fallback_loop_interrupted,
            "switch_action_selected_after_exhaustion": (
                self.switch_action_selected_after_exhaustion
            ),
            "selected_switch_action": self.selected_switch_action,
            "selected_switch_action_args": dict(self.selected_switch_action_args),
            "blocked_action": self.blocked_action,
            "blocked_action_args": dict(self.blocked_action_args),
            "consecutive_repeats_before": int(self.consecutive_repeats_before),
            "max_same_action_arg_repeats": int(self.max_same_action_arg_repeats),
            "loop_guard_reason": self.loop_guard_reason,
        }


def apply_policy_loop_guard(
    *,
    proposed_action: Any | None,
    valid_actions: Sequence[Any],
    prefix: Sequence[Any],
    decision_reason: str,
    target_action: str,
    max_same_action_arg_repeats: int = 2,
    switch_preference: Sequence[str] = (),
    fallback_decision_reason: str = "candidate_policy_live_target_fallback",
) -> PolicyLoopGuardDecision:
    """Interrupt a repeated target fallback by selecting a legal switch action."""
    threshold = max(1, int(max_same_action_arg_repeats))
    if proposed_action is None:
        return PolicyLoopGuardDecision(
            selected_action_raw=None,
            max_same_action_arg_repeats=threshold,
        )

    proposed_name = action_name(proposed_action)
    proposed_args = action_args(proposed_action)
    repeats_before = consecutive_repeat_count(prefix, proposed_name, proposed_args)
    repeated = repeats_before >= threshold
    target_fallback = (
        str(decision_reason) == fallback_decision_reason
        and proposed_name == str(target_action)
    )
    if not repeated or not target_fallback:
        return PolicyLoopGuardDecision(
            selected_action_raw=proposed_action,
            repeated_same_action_args_detected=repeated,
            blocked_action=proposed_name if repeated else "",
            blocked_action_args=proposed_args if repeated else {},
            consecutive_repeats_before=repeats_before,
            max_same_action_arg_repeats=threshold,
        )

    switch = select_legal_switch_action(
        valid_actions,
        blocked_action=proposed_name,
        blocked_action_args=proposed_args,
        switch_preference=switch_preference,
    )
    if switch is None:
        return PolicyLoopGuardDecision(
            selected_action_raw=proposed_action,
            loop_guard_triggered=True,
            repeated_same_action_args_detected=True,
            blocked_action=proposed_name,
            blocked_action_args=proposed_args,
            consecutive_repeats_before=repeats_before,
            max_same_action_arg_repeats=threshold,
            loop_guard_reason=LOOP_GUARD_BLOCKED_REASON,
        )

    return PolicyLoopGuardDecision(
        selected_action_raw=switch,
        loop_guard_triggered=True,
        repeated_same_action_args_detected=True,
        fallback_loop_interrupted=True,
        switch_action_selected_after_exhaustion=True,
        selected_switch_action=action_name(switch),
        selected_switch_action_args=action_args(switch),
        blocked_action=proposed_name,
        blocked_action_args=proposed_args,
        consecutive_repeats_before=repeats_before,
        max_same_action_arg_repeats=threshold,
        loop_guard_reason=LOOP_GUARD_SWITCH_REASON,
    )


def select_legal_switch_action(
    valid_actions: Sequence[Any],
    *,
    blocked_action: str,
    blocked_action_args: Mapping[str, Any],
    switch_preference: Sequence[str] = (),
) -> Any | None:
    blocked_key = action_key(blocked_action, blocked_action_args)
    candidates = [
        action
        for action in valid_actions
        if action_name(action) and action_name(action) != "RESET"
        and action_key(action_name(action), action_args(action)) != blocked_key
    ]
    if not candidates:
        return None

    preferences = [str(name) for name in switch_preference if str(name)]
    for preferred in preferences:
        for action in candidates:
            if action_name(action) == preferred:
                return action
    return candidates[0]


def consecutive_repeat_count(
    history: Sequence[Any],
    action: str,
    action_args_: Mapping[str, Any] | None = None,
) -> int:
    wanted = action_key(action, action_args_ or {})
    count = 0
    for item in reversed(list(history)):
        if action_key(action_name(item), action_args(item)) != wanted:
            break
        count += 1
    return count


def max_consecutive_same_action_arg_repeats(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    best = 0
    current = 0
    previous = ""
    for row in rows:
        key = action_key(
            str(row.get("selected_action", "")),
            dict(row.get("selected_action_args", {}) or {}),
        )
        if key and key == previous:
            current += 1
        else:
            current = 1 if key else 0
            previous = key
        best = max(best, current)
    return best


def action_key(action: str, action_args_: Mapping[str, Any] | None = None) -> str:
    if not str(action):
        return ""
    return json.dumps(
        {
            "action": str(action),
            "action_args": normalize_action_args(action_args_ or {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def action_name(action: Any) -> str:
    if isinstance(action, Mapping):
        return str(action.get("name", action.get("action", "")))
    return str(getattr(action, "name", ""))


def action_args(action: Any) -> Dict[str, Any]:
    if isinstance(action, Mapping):
        return normalize_action_args(action.get("action_args", {}) or {})
    return normalize_action_args(getattr(action, "action_args", {}) or {})


def normalize_action_args(action_args_: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): value for key, value in dict(action_args_ or {}).items()}
