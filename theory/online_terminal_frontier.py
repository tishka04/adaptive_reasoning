"""Terminal-only exploration beyond locally completed goal hypotheses.

SAGE can often drive a measurable hypothesis to its postcondition without
finishing the level.  Such a state is useful evidence about *where the current
goal stops being sufficient*, but it is not positive terminal evidence.  This
module keeps those states as negative terminal frontiers and runs a bounded
continuation from them.  A continuation is credited only when the environment
reports a level change or a win.

The explorer is deliberately agnostic to game identity and objective family.
It receives stable state signatures, objective identifiers, and concrete legal
actions observed by the live controller.  Failed local progress never promotes
a continuation.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TerminalFrontierAction:
    """One concrete action eligible for a bounded frontier suffix."""

    action_name: str
    action_data: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def from_parts(
        cls,
        action_name: str,
        action_data: Mapping[str, Any] | None = None,
    ) -> "TerminalFrontierAction":
        return cls(
            action_name=str(action_name).upper(),
            action_data=tuple(sorted(dict(action_data or {}).items())),
        )

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self.action_data)

    @property
    def signature(self) -> str:
        return f"{self.action_name}|{self.action_data}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_name,
            "action_data": self.data,
        }


@dataclass(frozen=True)
class TerminalFrontierSelection:
    """One action selected as part of a terminally evaluated suffix."""

    frontier_id: str
    objective_ids: Tuple[str, ...]
    action: TerminalFrontierAction
    step_index: int
    action_limit: int
    replaying_successful_continuation: bool
    reason: str


@dataclass
class SuccessfulContinuation:
    """A suffix whose final observed transition changed level or won."""

    actions: Tuple[TerminalFrontierAction, ...]
    state_signatures: Tuple[str, ...]
    confirmations: int = 1

    @property
    def signature(self) -> Tuple[str, ...]:
        return tuple(action.signature for action in self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "state_signatures": list(self.state_signatures),
            "confirmations": self.confirmations,
        }


@dataclass
class TerminalNegativeFrontier:
    """A postcondition state observed without terminal success."""

    frontier_id: str
    state_signature: str
    objective_ids: Tuple[str, ...]
    context_signatures: set[str] = field(default_factory=set)
    captures: int = 0
    trials: int = 0
    terminal_credits: int = 0
    nonterminal_suffixes: int = 0
    unsafe_suffixes: int = 0
    censored_suffixes: int = 0
    successful_continuations: Dict[
        Tuple[str, ...], SuccessfulContinuation
    ] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "state_signature": self.state_signature,
            "objective_ids": list(self.objective_ids),
            "context_signatures": sorted(self.context_signatures),
            "captures": self.captures,
            "trials": self.trials,
            "terminal_credits": self.terminal_credits,
            "nonterminal_suffixes": self.nonterminal_suffixes,
            "unsafe_suffixes": self.unsafe_suffixes,
            "censored_suffixes": self.censored_suffixes,
            "successful_continuations": [
                continuation.to_dict()
                for _, continuation in sorted(
                    self.successful_continuations.items()
                )
            ],
        }


@dataclass
class _ActiveSuffix:
    frontier_id: str
    actions: list[TerminalFrontierAction]
    state_signatures: list[str]
    replay: SuccessfulContinuation | None = None
    pending: TerminalFrontierSelection | None = None


class OnlineTerminalFrontierExplorer:
    """Explore and terminally credit bounded suffixes from negative frontiers."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_frontiers: int = 24,
        max_suffix_actions: int = 6,
        max_trials_per_frontier: int = 4,
        max_candidates_per_state: int = 12,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_frontiers = max(1, int(max_frontiers))
        self.max_suffix_actions = max(1, int(max_suffix_actions))
        self.max_trials_per_frontier = max(
            1,
            int(max_trials_per_frontier),
        )
        self.max_candidates_per_state = max(
            1,
            int(max_candidates_per_state),
        )
        self._frontiers: Dict[str, TerminalNegativeFrontier] = {}
        self._actions_by_state: Dict[
            str,
            Dict[str, TerminalFrontierAction],
        ] = {}
        self._choice_counts: Counter[Tuple[str, Tuple[str, ...], str]] = (
            Counter()
        )
        self._active: _ActiveSuffix | None = None
        self._frontiers_captured = 0
        self._duplicate_captures = 0
        self._trials_started = 0
        self._suffix_actions = 0
        self._terminal_credits = 0
        self._level_change_credits = 0
        self._win_credits = 0
        self._successful_replays = 0
        self._replay_divergences = 0
        self._capacity_blocks = 0
        self._branch_trial_blocks = 0
        self._trial_started_this_branch = False

    @property
    def active_frontier_id(self) -> str:
        return "" if self._active is None else self._active.frontier_id

    @property
    def active_suffix_started(self) -> bool:
        """Whether at least one action of the active suffix was selected."""
        return bool(
            self._active is not None
            and (self._active.actions or self._active.pending is not None)
        )

    @property
    def active_replay_available(self) -> bool:
        """Whether the active trial can replay a terminal-credited suffix."""
        return bool(
            self._active is not None
            and self._active.replay is not None
        )

    def remember_action(
        self,
        state_signature: str,
        action_name: str,
        action_data: Mapping[str, Any] | None = None,
    ) -> None:
        """Retain a concrete action only after it was legal and executed."""
        state = str(state_signature)
        action = TerminalFrontierAction.from_parts(action_name, action_data)
        if not state or not action.action_name:
            return
        candidates = self._actions_by_state.setdefault(state, {})
        if action.signature in candidates:
            return
        if len(candidates) >= self.max_candidates_per_state:
            return
        candidates[action.signature] = action

    def capture(
        self,
        *,
        state_signature: str,
        objective_ids: Iterable[str],
        context_signature: str = "",
    ) -> str:
        """Capture a nonterminal postcondition and start one bounded trial."""
        if not self.enabled:
            return ""
        objectives = tuple(sorted({str(item) for item in objective_ids if item}))
        state = str(state_signature)
        if not state or not objectives:
            return ""
        frontier_id = _frontier_id(state, objectives)
        frontier = self._frontiers.get(frontier_id)
        if self._active is not None or self._trial_started_this_branch:
            self._branch_trial_blocks += 1
            if frontier is None:
                return ""
            self._duplicate_captures += 1
            frontier.captures += 1
            if context_signature:
                frontier.context_signatures.add(str(context_signature))
            return frontier_id
        if frontier is None:
            if len(self._frontiers) >= self.max_frontiers:
                self._capacity_blocks += 1
                return ""
            frontier = TerminalNegativeFrontier(
                frontier_id=frontier_id,
                state_signature=state,
                objective_ids=objectives,
            )
            self._frontiers[frontier_id] = frontier
            self._frontiers_captured += 1
        else:
            self._duplicate_captures += 1
        frontier.captures += 1
        if context_signature:
            frontier.context_signatures.add(str(context_signature))
        replay = self._best_successful_continuation(frontier)
        if replay is None and frontier.trials >= self.max_trials_per_frontier:
            return frontier_id
        if replay is not None and replay.confirmations >= 2:
            return frontier_id
        frontier.trials += 1
        self._trials_started += 1
        self._trial_started_this_branch = True
        self._active = _ActiveSuffix(
            frontier_id=frontier_id,
            actions=[],
            state_signatures=[state],
            replay=replay,
        )
        return frontier_id

    def select(
        self,
        *,
        state_signature: str,
        available_actions: Sequence[str],
        proposed_actions: Sequence[TerminalFrontierAction] = (),
        restrict_to_proposed: bool = False,
    ) -> TerminalFrontierSelection | None:
        """Select a replay action or the least-tested concrete continuation."""
        active = self._active
        if not self.enabled or active is None or active.pending is not None:
            return None
        frontier = self._frontiers[active.frontier_id]
        allowed = {str(action).upper() for action in available_actions}
        state = str(state_signature)

        if active.replay is not None:
            step = len(active.actions)
            expected_states = active.replay.state_signatures
            if step < len(expected_states) and state != expected_states[step]:
                self._replay_divergences += 1
                active.replay = None
            elif step < len(active.replay.actions):
                action = active.replay.actions[step]
                if action.action_name in allowed:
                    return self._record_selection(
                        active,
                        frontier,
                        action,
                        replaying=True,
                    )
                self._replay_divergences += 1
                active.replay = None

        candidates: Dict[str, TerminalFrontierAction] = {}
        for action in proposed_actions:
            if action.action_name in allowed:
                candidates.setdefault(action.signature, action)
        if not restrict_to_proposed:
            for action in self._actions_by_state.get(state, {}).values():
                if action.action_name in allowed:
                    candidates.setdefault(action.signature, action)
            for action_name in sorted(allowed):
                if action_name == "ACTION6":
                    continue
                action = TerminalFrontierAction.from_parts(action_name)
                candidates.setdefault(action.signature, action)
        if not candidates:
            return None
        prefix = tuple(action.signature for action in active.actions)
        action = min(
            candidates.values(),
            key=lambda candidate: (
                self._choice_counts[
                    (frontier.frontier_id, prefix, candidate.signature)
                ],
                candidate.signature,
            ),
        )
        return self._record_selection(
            active,
            frontier,
            action,
            replaying=False,
        )

    def observe_transition(
        self,
        *,
        state_signature_before: str,
        state_signature_after: str,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        level_progressed: bool,
        won: bool,
        game_over: bool,
    ) -> Dict[str, Any]:
        """Revise a suffix using terminal outcomes and nothing weaker."""
        self.remember_action(
            state_signature_before,
            action_name,
            action_data,
        )
        active = self._active
        if active is None or active.pending is None:
            return _empty_outcome()
        selection = active.pending
        observed = TerminalFrontierAction.from_parts(action_name, action_data)
        replaying = bool(
            selection.replaying_successful_continuation
            and observed.signature == selection.action.signature
        )
        if selection.replaying_successful_continuation and not replaying:
            self._replay_divergences += 1
            active.replay = None
        active.pending = None
        active.actions.append(observed)
        active.state_signatures.append(str(state_signature_after))
        self._suffix_actions += 1
        terminal_success = bool(level_progressed or won)
        frontier = self._frontiers[active.frontier_id]
        outcome = {
            "frontier_id": frontier.frontier_id,
            "objective_ids": list(frontier.objective_ids),
            "suffix_step": len(active.actions) - 1,
            "terminal_success": terminal_success,
            "level_progressed": bool(level_progressed),
            "won": bool(won),
            "game_over": bool(game_over),
            "credited": False,
            "replaying_successful_continuation": replaying,
        }
        if terminal_success:
            actions = tuple(active.actions)
            states = tuple(active.state_signatures)
            signature = tuple(action.signature for action in actions)
            continuation = frontier.successful_continuations.get(signature)
            if continuation is None:
                continuation = SuccessfulContinuation(actions, states)
                frontier.successful_continuations[signature] = continuation
            else:
                continuation.confirmations += 1
            frontier.terminal_credits += 1
            self._terminal_credits += 1
            self._level_change_credits += int(bool(level_progressed))
            self._win_credits += int(bool(won))
            if replaying:
                self._successful_replays += 1
            outcome["credited"] = True
            self._active = None
        elif game_over:
            frontier.unsafe_suffixes += 1
            self._active = None
        elif len(active.actions) >= self.max_suffix_actions:
            frontier.nonterminal_suffixes += 1
            self._active = None
        return outcome

    def start_branch(self) -> None:
        """Censor an unfinished suffix without inventing negative credit."""
        if self._active is not None:
            self._frontiers[self._active.frontier_id].censored_suffixes += 1
        self._active = None
        self._trial_started_this_branch = False

    def frontiers(self) -> Tuple[TerminalNegativeFrontier, ...]:
        return tuple(
            self._frontiers[key]
            for key in sorted(self._frontiers)
        )

    def summary(self) -> Dict[str, Any]:
        """Return auditable attribution counters for SAGE.9f."""
        successful = sum(
            len(frontier.successful_continuations)
            for frontier in self._frontiers.values()
        )
        return {
            "enabled": self.enabled,
            "max_frontiers": self.max_frontiers,
            "max_suffix_actions": self.max_suffix_actions,
            "max_trials_per_frontier": self.max_trials_per_frontier,
            "frontiers": len(self._frontiers),
            "frontiers_captured": self._frontiers_captured,
            "duplicate_captures": self._duplicate_captures,
            "capacity_blocks": self._capacity_blocks,
            "branch_trial_blocks": self._branch_trial_blocks,
            "trials_started": self._trials_started,
            "suffix_actions": self._suffix_actions,
            "terminal_credits": self._terminal_credits,
            "level_change_credits": self._level_change_credits,
            "win_credits": self._win_credits,
            "successful_continuations": successful,
            "successful_replays": self._successful_replays,
            "replay_divergences": self._replay_divergences,
            "nonterminal_suffixes": sum(
                frontier.nonterminal_suffixes
                for frontier in self._frontiers.values()
            ),
            "unsafe_suffixes": sum(
                frontier.unsafe_suffixes
                for frontier in self._frontiers.values()
            ),
            "censored_suffixes": sum(
                frontier.censored_suffixes
                for frontier in self._frontiers.values()
            ),
            "active_frontier_id": self.active_frontier_id,
            "records": [frontier.to_dict() for frontier in self.frontiers()],
        }

    def _record_selection(
        self,
        active: _ActiveSuffix,
        frontier: TerminalNegativeFrontier,
        action: TerminalFrontierAction,
        *,
        replaying: bool,
    ) -> TerminalFrontierSelection:
        prefix = tuple(item.signature for item in active.actions)
        self._choice_counts[(frontier.frontier_id, prefix, action.signature)] += 1
        selection = TerminalFrontierSelection(
            frontier_id=frontier.frontier_id,
            objective_ids=frontier.objective_ids,
            action=action,
            step_index=len(active.actions),
            action_limit=self.max_suffix_actions,
            replaying_successful_continuation=replaying,
            reason=(
                "replay terminal-credited continuation from identical frontier"
                if replaying
                else "bounded contrast after nonterminal objective postcondition"
            ),
        )
        active.pending = selection
        return selection

    @staticmethod
    def _best_successful_continuation(
        frontier: TerminalNegativeFrontier,
    ) -> SuccessfulContinuation | None:
        if not frontier.successful_continuations:
            return None
        return max(
            frontier.successful_continuations.values(),
            key=lambda item: (
                item.confirmations,
                -len(item.actions),
                item.signature,
            ),
        )


def _frontier_id(state: str, objectives: Sequence[str]) -> str:
    payload = f"{state}|{'|'.join(objectives)}".encode("utf-8")
    return f"terminal-frontier::{hashlib.sha1(payload).hexdigest()[:16]}"


def _empty_outcome() -> Dict[str, Any]:
    return {
        "frontier_id": "",
        "objective_ids": [],
        "suffix_step": None,
        "terminal_success": False,
        "level_progressed": False,
        "won": False,
        "game_over": False,
        "credited": False,
        "replaying_successful_continuation": False,
    }


__all__ = [
    "OnlineTerminalFrontierExplorer",
    "SuccessfulContinuation",
    "TerminalFrontierAction",
    "TerminalFrontierSelection",
    "TerminalNegativeFrontier",
]
