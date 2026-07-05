"""Canonical context replay helpers for M2 frontiers."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


ACTION_RE = re.compile(r"ACTION\d+")


def canonical_context_replay(
    frontier: Mapping[str, Any],
    *,
    proposed_replay: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return the causal replay sequence for an A40 frontier.

    A40 may expose compact signatures like ``context_signature=["ACTION3"]`` for
    a live state whose context id encodes an earlier consumed action. For M3
    reproducibility, M2 must preserve the causal prefix.
    """
    context_id = str(frontier.get("frontier_context_id", ""))
    from_context_id = _actions_from_context_id(context_id)
    if from_context_id:
        return from_context_id
    proposed = tuple(str(item) for item in proposed_replay if str(item))
    if proposed:
        return proposed
    return tuple(str(item) for item in frontier.get("context_signature", []) or [])


def context_replay_warnings(
    frontier: Mapping[str, Any],
    *,
    proposed_replay: Sequence[str],
    canonical_replay: Sequence[str],
) -> tuple[str, ...]:
    proposed = tuple(str(item) for item in proposed_replay if str(item))
    canonical = tuple(str(item) for item in canonical_replay if str(item))
    warnings = []
    if proposed and proposed != canonical:
        warnings.append("context_replay_canonicalized")
    context_id = str(frontier.get("frontier_context_id", ""))
    live_after_actions = _live_after_actions(context_id)
    for action in live_after_actions:
        if action not in set(canonical):
            warnings.append(f"context_replay_missing_{action}")
    return tuple(warnings)


def _actions_from_context_id(context_id: str) -> tuple[str, ...]:
    if not context_id:
        return ()
    live_after = _live_after_actions(context_id)
    if live_after:
        prefix = context_id.split("_live_after_", 1)[0]
        after_prefix = _after_actions(prefix)
        return tuple(dict.fromkeys((*live_after, *after_prefix)))
    return _after_actions(context_id)


def _live_after_actions(context_id: str) -> tuple[str, ...]:
    actions = []
    for suffix in context_id.split("_live_after_")[1:]:
        actions.extend(ACTION_RE.findall(suffix))
    return tuple(actions)


def _after_actions(text: str) -> tuple[str, ...]:
    if not text.startswith("after_"):
        return ()
    return tuple(ACTION_RE.findall(text))
