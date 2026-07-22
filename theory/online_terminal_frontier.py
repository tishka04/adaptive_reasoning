"""Terminal-only exploration beyond locally completed goal hypotheses.

SAGE can often drive a measurable hypothesis to its postcondition without
finishing the level.  Such a state is useful evidence about *where the current
goal stops being sufficient*, but it is not positive terminal evidence.  This
module keeps those states as negative terminal frontiers and runs a bounded
continuation from them.  Repeated frontiers whose current bound is exhausted
receive a larger continuation horizon, without using intermediate progress as
evidence.  A continuation is credited only when the environment reports a
level change or a win.  When the bounded horizon expires, a dormant lineage
can keep observing the unchanged live policy.  A later terminal event only
nominates that lineage; exact terminal replay is required before credit.

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
    replaying_dormant_terminal_candidate: bool
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
class DormantTerminalContinuation:
    """A delayed-terminal lineage awaiting exact online replay."""

    actions: Tuple[TerminalFrontierAction, ...]
    state_signatures: Tuple[str, ...]
    level_progressed: bool
    won: bool
    terminal_observations: int = 1
    replay_attempts: int = 0
    confirmations: int = 0
    refutations: int = 0
    divergences: int = 0

    @property
    def signature(self) -> Tuple[str, ...]:
        return tuple(action.signature for action in self.actions)

    @property
    def status(self) -> str:
        if self.confirmations:
            return "terminal_confirmed"
        if self.refutations:
            return "refuted"
        if self.divergences:
            return "inconclusive_divergence"
        return "awaiting_replay"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "state_signatures": list(self.state_signatures),
            "level_progressed": self.level_progressed,
            "won": self.won,
            "terminal_observations": self.terminal_observations,
            "replay_attempts": self.replay_attempts,
            "confirmations": self.confirmations,
            "refutations": self.refutations,
            "divergences": self.divergences,
            "status": self.status,
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
    allocated_action_limit: int = 0
    horizon_extensions: int = 0
    horizon_history: list[int] = field(default_factory=list)
    longest_suffix_actions: int = 0
    successful_continuations: Dict[
        Tuple[str, ...], SuccessfulContinuation
    ] = field(default_factory=dict)
    dormant_terminal_candidates: Dict[
        Tuple[str, ...], DormantTerminalContinuation
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
            "allocated_action_limit": self.allocated_action_limit,
            "horizon_extensions": self.horizon_extensions,
            "horizon_history": list(self.horizon_history),
            "longest_suffix_actions": self.longest_suffix_actions,
            "successful_continuations": [
                continuation.to_dict()
                for _, continuation in sorted(
                    self.successful_continuations.items()
                )
            ],
            "dormant_terminal_candidates": [
                continuation.to_dict()
                for _, continuation in sorted(
                    self.dormant_terminal_candidates.items()
                )
            ],
        }


@dataclass
class _ActiveSuffix:
    frontier_id: str
    actions: list[TerminalFrontierAction]
    state_signatures: list[str]
    action_limit: int
    replay: SuccessfulContinuation | None = None
    dormant_candidate_replay: DormantTerminalContinuation | None = None
    pending: TerminalFrontierSelection | None = None


@dataclass
class _DormantLineage:
    frontier_id: str
    actions: list[TerminalFrontierAction]
    state_signatures: list[str]


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
        enable_adaptive_horizon: bool = True,
        max_adaptive_suffix_actions: int = 24,
        adaptive_horizon_increment: int = 6,
        enable_dormant_terminal_lineage: bool = True,
        max_dormant_lineage_actions: int = 80,
        max_dormant_candidates_per_frontier: int = 4,
        max_dormant_candidate_replays: int = 1,
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
        self.enable_adaptive_horizon = bool(enable_adaptive_horizon)
        self.max_adaptive_suffix_actions = max(
            self.max_suffix_actions,
            int(max_adaptive_suffix_actions),
        )
        self.adaptive_horizon_increment = max(
            1,
            int(adaptive_horizon_increment),
        )
        self.enable_dormant_terminal_lineage = bool(
            enable_dormant_terminal_lineage
        )
        self.max_dormant_lineage_actions = max(
            self.max_adaptive_suffix_actions,
            int(max_dormant_lineage_actions),
        )
        self.max_dormant_candidates_per_frontier = max(
            1,
            int(max_dormant_candidates_per_frontier),
        )
        self.max_dormant_candidate_replays = max(
            1,
            int(max_dormant_candidate_replays),
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
        self._dormant: _DormantLineage | None = None
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
        self._adaptive_horizon_extensions = 0
        self._adaptive_horizon_actions_granted = 0
        self._extended_suffix_actions = 0
        self._dormant_lineages_started = 0
        self._dormant_lineage_actions = 0
        self._dormant_lineage_terminal_candidates = 0
        self._dormant_lineage_level_candidates = 0
        self._dormant_lineage_win_candidates = 0
        self._dormant_lineage_censored = 0
        self._dormant_lineage_expired = 0
        self._dormant_lineage_unsafe = 0
        self._dormant_candidate_capacity_blocks = 0
        self._dormant_candidate_replay_attempts = 0
        self._dormant_candidate_replay_actions = 0
        self._dormant_candidate_confirmations = 0
        self._dormant_candidate_refutations = 0
        self._dormant_candidate_divergences = 0
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
        """Whether the active trial has a confirmed or candidate replay."""
        return bool(
            self._active is not None
            and (
                self._active.replay is not None
                or self._active.dormant_candidate_replay is not None
            )
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
                allocated_action_limit=self.max_suffix_actions,
                horizon_history=[self.max_suffix_actions],
            )
            self._frontiers[frontier_id] = frontier
            self._frontiers_captured += 1
        else:
            self._duplicate_captures += 1
        frontier.captures += 1
        if context_signature:
            frontier.context_signatures.add(str(context_signature))
        replay = self._best_successful_continuation(frontier)
        dormant_candidate = (
            None
            if replay is not None
            else self._best_dormant_terminal_candidate(frontier)
        )
        if (
            replay is None
            and dormant_candidate is None
            and frontier.trials >= self.max_trials_per_frontier
        ):
            return frontier_id
        if replay is not None and replay.confirmations >= 2:
            return frontier_id
        if replay is None and dormant_candidate is None:
            self._extend_exhausted_frontier_horizon(frontier)
        if dormant_candidate is not None:
            dormant_candidate.replay_attempts += 1
            self._dormant_candidate_replay_attempts += 1
        frontier.trials += 1
        self._trials_started += 1
        self._trial_started_this_branch = True
        self._active = _ActiveSuffix(
            frontier_id=frontier_id,
            actions=[],
            state_signatures=[state],
            action_limit=(
                len(dormant_candidate.actions)
                if dormant_candidate is not None
                else (
                    len(replay.actions)
                    if replay is not None
                    else frontier.allocated_action_limit
                )
            ),
            replay=replay,
            dormant_candidate_replay=dormant_candidate,
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

        replay_sequence = (
            active.replay
            if active.replay is not None
            else active.dormant_candidate_replay
        )
        if replay_sequence is not None:
            step = len(active.actions)
            expected_states = replay_sequence.state_signatures
            if step < len(expected_states) and state != expected_states[step]:
                self._replay_divergences += 1
                if active.dormant_candidate_replay is not None:
                    active.dormant_candidate_replay.divergences += 1
                    self._dormant_candidate_divergences += 1
                self._active = None
                return None
            elif step < len(replay_sequence.actions):
                action = replay_sequence.actions[step]
                if action.action_name in allowed:
                    return self._record_selection(
                        active,
                        frontier,
                        action,
                        replaying=active.replay is not None,
                        replaying_dormant_candidate=(
                            active.dormant_candidate_replay is not None
                        ),
                    )
                self._replay_divergences += 1
                if active.dormant_candidate_replay is not None:
                    active.dormant_candidate_replay.divergences += 1
                    self._dormant_candidate_divergences += 1
                self._active = None
                return None

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
            replaying_dormant_candidate=False,
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
        observed = TerminalFrontierAction.from_parts(action_name, action_data)
        active = self._active
        if active is None or active.pending is None:
            return self._observe_dormant_lineage(
                state_signature_after=str(state_signature_after),
                observed=observed,
                level_progressed=bool(level_progressed),
                won=bool(won),
                game_over=bool(game_over),
            )
        selection = active.pending
        dormant_candidate = active.dormant_candidate_replay
        replaying = bool(
            selection.replaying_successful_continuation
            and observed.signature == selection.action.signature
        )
        replaying_dormant_candidate = bool(
            selection.replaying_dormant_terminal_candidate
            and observed.signature == selection.action.signature
        )
        if selection.replaying_successful_continuation and not replaying:
            self._replay_divergences += 1
            active.replay = None
        if (
            selection.replaying_dormant_terminal_candidate
            and not replaying_dormant_candidate
        ):
            self._replay_divergences += 1
            self._dormant_candidate_divergences += 1
            if dormant_candidate is not None:
                dormant_candidate.divergences += 1
            active.dormant_candidate_replay = None
        active.pending = None
        active.actions.append(observed)
        active.state_signatures.append(str(state_signature_after))
        self._suffix_actions += 1
        if selection.replaying_dormant_terminal_candidate:
            self._dormant_candidate_replay_actions += 1
        if (
            len(active.actions) > self.max_suffix_actions
            and active.replay is None
            and dormant_candidate is None
        ):
            self._extended_suffix_actions += 1
        terminal_success = bool(level_progressed or won)
        frontier = self._frontiers[active.frontier_id]
        frontier.longest_suffix_actions = max(
            frontier.longest_suffix_actions,
            len(active.actions),
        )
        outcome = {
            "frontier_id": frontier.frontier_id,
            "objective_ids": list(frontier.objective_ids),
            "suffix_step": len(active.actions) - 1,
            "action_limit": active.action_limit,
            "adaptive_horizon": bool(
                active.action_limit > self.max_suffix_actions
                and not selection.replaying_successful_continuation
                and not selection.replaying_dormant_terminal_candidate
            ),
            "terminal_success": terminal_success,
            "level_progressed": bool(level_progressed),
            "won": bool(won),
            "game_over": bool(game_over),
            "credited": False,
            "replaying_successful_continuation": replaying,
            "replaying_dormant_terminal_candidate": (
                replaying_dormant_candidate
            ),
            "dormant_terminal_candidate_nominated": False,
            "dormant_terminal_candidate_confirmed": False,
        }
        if terminal_success:
            actions = tuple(active.actions)
            states = tuple(active.state_signatures)
            if selection.replaying_dormant_terminal_candidate:
                if replaying_dormant_candidate:
                    if dormant_candidate is not None:
                        dormant_candidate.confirmations += 1
                    self._dormant_candidate_confirmations += 1
                    self._credit_continuation(
                        frontier,
                        actions,
                        states,
                        level_progressed=bool(level_progressed),
                        won=bool(won),
                        replaying=True,
                    )
                    outcome["credited"] = True
                    outcome["dormant_terminal_candidate_confirmed"] = True
                else:
                    candidate = self._record_dormant_terminal_candidate(
                        frontier,
                        actions,
                        states,
                        level_progressed=bool(level_progressed),
                        won=bool(won),
                    )
                    outcome["dormant_terminal_candidate_nominated"] = bool(
                        candidate is not None
                    )
            else:
                self._credit_continuation(
                    frontier,
                    actions,
                    states,
                    level_progressed=bool(level_progressed),
                    won=bool(won),
                    replaying=replaying,
                )
                outcome["credited"] = True
            self._active = None
        elif selection.replaying_dormant_terminal_candidate:
            if replaying_dormant_candidate and (
                game_over or len(active.actions) >= active.action_limit
            ):
                if dormant_candidate is not None:
                    dormant_candidate.refutations += 1
                self._dormant_candidate_refutations += 1
            if game_over:
                frontier.unsafe_suffixes += 1
            if (
                not replaying_dormant_candidate
                or game_over
                or len(active.actions) >= active.action_limit
            ):
                self._active = None
        elif game_over:
            frontier.unsafe_suffixes += 1
            self._active = None
        elif len(active.actions) >= active.action_limit:
            frontier.nonterminal_suffixes += 1
            if active.replay is None:
                self._start_dormant_lineage(active)
            self._active = None
        return outcome

    def start_branch(self) -> None:
        """Censor an unfinished suffix without inventing negative credit."""
        if self._active is not None:
            self._frontiers[self._active.frontier_id].censored_suffixes += 1
        if self._dormant is not None:
            self._dormant_lineage_censored += 1
        self._active = None
        self._dormant = None
        self._trial_started_this_branch = False

    def frontiers(self) -> Tuple[TerminalNegativeFrontier, ...]:
        return tuple(
            self._frontiers[key]
            for key in sorted(self._frontiers)
        )

    def summary(self) -> Dict[str, Any]:
        """Return auditable attribution counters for SAGE.9f-SAGE.9h."""
        successful = sum(
            len(frontier.successful_continuations)
            for frontier in self._frontiers.values()
        )
        dormant_candidates = [
            candidate
            for frontier in self._frontiers.values()
            for candidate in frontier.dormant_terminal_candidates.values()
        ]
        return {
            "enabled": self.enabled,
            "max_frontiers": self.max_frontiers,
            "max_suffix_actions": self.max_suffix_actions,
            "max_trials_per_frontier": self.max_trials_per_frontier,
            "adaptive_horizon_enabled": self.enable_adaptive_horizon,
            "max_adaptive_suffix_actions": self.max_adaptive_suffix_actions,
            "adaptive_horizon_increment": self.adaptive_horizon_increment,
            "dormant_terminal_lineage_enabled": (
                self.enable_dormant_terminal_lineage
            ),
            "max_dormant_lineage_actions": self.max_dormant_lineage_actions,
            "max_dormant_candidates_per_frontier": (
                self.max_dormant_candidates_per_frontier
            ),
            "max_dormant_candidate_replays": (
                self.max_dormant_candidate_replays
            ),
            "frontiers": len(self._frontiers),
            "frontiers_captured": self._frontiers_captured,
            "duplicate_captures": self._duplicate_captures,
            "capacity_blocks": self._capacity_blocks,
            "branch_trial_blocks": self._branch_trial_blocks,
            "adaptive_horizon_extensions": self._adaptive_horizon_extensions,
            "adaptive_horizon_actions_granted": (
                self._adaptive_horizon_actions_granted
            ),
            "extended_suffix_actions": self._extended_suffix_actions,
            "frontiers_with_extended_horizon": sum(
                int(frontier.horizon_extensions > 0)
                for frontier in self._frontiers.values()
            ),
            "maximum_allocated_horizon": max(
                (
                    frontier.allocated_action_limit
                    for frontier in self._frontiers.values()
                ),
                default=self.max_suffix_actions,
            ),
            "dormant_lineages_started": self._dormant_lineages_started,
            "dormant_lineage_actions": self._dormant_lineage_actions,
            "dormant_lineage_terminal_candidates": (
                self._dormant_lineage_terminal_candidates
            ),
            "dormant_lineage_level_candidates": (
                self._dormant_lineage_level_candidates
            ),
            "dormant_lineage_win_candidates": (
                self._dormant_lineage_win_candidates
            ),
            "dormant_lineage_censored": self._dormant_lineage_censored,
            "dormant_lineage_expired": self._dormant_lineage_expired,
            "dormant_lineage_unsafe": self._dormant_lineage_unsafe,
            "dormant_terminal_candidates": len(dormant_candidates),
            "dormant_candidate_capacity_blocks": (
                self._dormant_candidate_capacity_blocks
            ),
            "dormant_candidate_replay_attempts": (
                self._dormant_candidate_replay_attempts
            ),
            "dormant_candidate_replay_actions": (
                self._dormant_candidate_replay_actions
            ),
            "dormant_candidate_confirmations": (
                self._dormant_candidate_confirmations
            ),
            "dormant_candidate_refutations": (
                self._dormant_candidate_refutations
            ),
            "dormant_candidate_divergences": (
                self._dormant_candidate_divergences
            ),
            "maximum_dormant_candidate_length": max(
                (len(candidate.actions) for candidate in dormant_candidates),
                default=0,
            ),
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
            "active_dormant_frontier_id": (
                "" if self._dormant is None else self._dormant.frontier_id
            ),
            "records": [frontier.to_dict() for frontier in self.frontiers()],
        }

    def _record_selection(
        self,
        active: _ActiveSuffix,
        frontier: TerminalNegativeFrontier,
        action: TerminalFrontierAction,
        *,
        replaying: bool,
        replaying_dormant_candidate: bool,
    ) -> TerminalFrontierSelection:
        prefix = tuple(item.signature for item in active.actions)
        self._choice_counts[(frontier.frontier_id, prefix, action.signature)] += 1
        selection = TerminalFrontierSelection(
            frontier_id=frontier.frontier_id,
            objective_ids=frontier.objective_ids,
            action=action,
            step_index=len(active.actions),
            action_limit=active.action_limit,
            replaying_successful_continuation=replaying,
            replaying_dormant_terminal_candidate=(
                replaying_dormant_candidate
            ),
            reason=(
                "replay terminal-credited continuation from identical frontier"
                if replaying
                else "replay delayed-terminal candidate from identical frontier"
                if replaying_dormant_candidate
                else (
                    "adaptive terminal-only continuation after exhausted "
                    "negative frontier"
                    if active.action_limit > self.max_suffix_actions
                    else (
                        "bounded contrast after nonterminal objective "
                        "postcondition"
                    )
                )
            ),
        )
        active.pending = selection
        return selection

    def _start_dormant_lineage(self, active: _ActiveSuffix) -> None:
        """Keep observing the live policy after its bounded suffix expires."""
        if (
            not self.enable_dormant_terminal_lineage
            or len(active.actions) >= self.max_dormant_lineage_actions
        ):
            return
        self._dormant = _DormantLineage(
            frontier_id=active.frontier_id,
            actions=list(active.actions),
            state_signatures=list(active.state_signatures),
        )
        self._dormant_lineages_started += 1

    def _observe_dormant_lineage(
        self,
        *,
        state_signature_after: str,
        observed: TerminalFrontierAction,
        level_progressed: bool,
        won: bool,
        game_over: bool,
    ) -> Dict[str, Any]:
        lineage = self._dormant
        if not self.enable_dormant_terminal_lineage or lineage is None:
            return _empty_outcome()
        lineage.actions.append(observed)
        lineage.state_signatures.append(str(state_signature_after))
        self._dormant_lineage_actions += 1
        frontier = self._frontiers[lineage.frontier_id]
        terminal_success = bool(level_progressed or won)
        outcome = _empty_outcome()
        outcome.update(
            {
                "frontier_id": frontier.frontier_id,
                "objective_ids": list(frontier.objective_ids),
                "suffix_step": len(lineage.actions) - 1,
                "action_limit": self.max_dormant_lineage_actions,
                "terminal_success": terminal_success,
                "level_progressed": bool(level_progressed),
                "won": bool(won),
                "game_over": bool(game_over),
                "dormant_lineage_observation": True,
            }
        )
        if terminal_success:
            candidate = self._record_dormant_terminal_candidate(
                frontier,
                tuple(lineage.actions),
                tuple(lineage.state_signatures),
                level_progressed=bool(level_progressed),
                won=bool(won),
            )
            outcome["dormant_terminal_candidate_nominated"] = bool(
                candidate is not None
            )
            self._dormant = None
        elif game_over:
            self._dormant_lineage_unsafe += 1
            self._dormant = None
        elif len(lineage.actions) >= self.max_dormant_lineage_actions:
            self._dormant_lineage_expired += 1
            self._dormant = None
        return outcome

    def _record_dormant_terminal_candidate(
        self,
        frontier: TerminalNegativeFrontier,
        actions: Tuple[TerminalFrontierAction, ...],
        state_signatures: Tuple[str, ...],
        *,
        level_progressed: bool,
        won: bool,
    ) -> DormantTerminalContinuation | None:
        """Nominate delayed terminal evidence without granting credit."""
        signature = tuple(action.signature for action in actions)
        candidate = frontier.dormant_terminal_candidates.get(signature)
        if candidate is not None:
            candidate.terminal_observations += 1
            candidate.level_progressed = bool(
                candidate.level_progressed or level_progressed
            )
            candidate.won = bool(candidate.won or won)
            return candidate
        if (
            len(frontier.dormant_terminal_candidates)
            >= self.max_dormant_candidates_per_frontier
        ):
            self._dormant_candidate_capacity_blocks += 1
            return None
        candidate = DormantTerminalContinuation(
            actions=actions,
            state_signatures=state_signatures,
            level_progressed=bool(level_progressed),
            won=bool(won),
        )
        frontier.dormant_terminal_candidates[signature] = candidate
        self._dormant_lineage_terminal_candidates += 1
        self._dormant_lineage_level_candidates += int(bool(level_progressed))
        self._dormant_lineage_win_candidates += int(bool(won))
        return candidate

    def _credit_continuation(
        self,
        frontier: TerminalNegativeFrontier,
        actions: Tuple[TerminalFrontierAction, ...],
        state_signatures: Tuple[str, ...],
        *,
        level_progressed: bool,
        won: bool,
        replaying: bool,
    ) -> None:
        """Credit only an actually observed terminal continuation."""
        signature = tuple(action.signature for action in actions)
        continuation = frontier.successful_continuations.get(signature)
        if continuation is None:
            continuation = SuccessfulContinuation(actions, state_signatures)
            frontier.successful_continuations[signature] = continuation
        else:
            continuation.confirmations += 1
        frontier.terminal_credits += 1
        self._terminal_credits += 1
        self._level_change_credits += int(bool(level_progressed))
        self._win_credits += int(bool(won))
        if replaying:
            self._successful_replays += 1

    def _extend_exhausted_frontier_horizon(
        self,
        frontier: TerminalNegativeFrontier,
    ) -> None:
        """Grant one larger bound only after the previous bound was exhausted."""
        if (
            not self.enable_adaptive_horizon
            or frontier.allocated_action_limit >= self.max_adaptive_suffix_actions
            or frontier.nonterminal_suffixes <= frontier.horizon_extensions
        ):
            return
        previous = frontier.allocated_action_limit
        allocated = min(
            self.max_adaptive_suffix_actions,
            previous + self.adaptive_horizon_increment,
        )
        if allocated <= previous:
            return
        frontier.allocated_action_limit = allocated
        frontier.horizon_extensions += 1
        frontier.horizon_history.append(allocated)
        self._adaptive_horizon_extensions += 1
        self._adaptive_horizon_actions_granted += allocated - previous

    def _best_dormant_terminal_candidate(
        self,
        frontier: TerminalNegativeFrontier,
    ) -> DormantTerminalContinuation | None:
        candidates = [
            candidate
            for candidate in frontier.dormant_terminal_candidates.values()
            if candidate.confirmations == 0
            and candidate.refutations == 0
            and candidate.replay_attempts < self.max_dormant_candidate_replays
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item.terminal_observations,
                -len(item.actions),
                item.signature,
            ),
        )

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
        "action_limit": 0,
        "adaptive_horizon": False,
        "terminal_success": False,
        "level_progressed": False,
        "won": False,
        "game_over": False,
        "credited": False,
        "replaying_successful_continuation": False,
        "replaying_dormant_terminal_candidate": False,
        "dormant_lineage_observation": False,
        "dormant_terminal_candidate_nominated": False,
        "dormant_terminal_candidate_confirmed": False,
    }


__all__ = [
    "DormantTerminalContinuation",
    "OnlineTerminalFrontierExplorer",
    "SuccessfulContinuation",
    "TerminalFrontierAction",
    "TerminalFrontierSelection",
    "TerminalNegativeFrontier",
]
