"""Verified per-level route memory and candidate-only route shortening.

The memory is deliberately game agnostic.  A route is keyed by the exact
observed start-state signature and contains only actions that were executed in
the current branch before a real level transition or win.  Shorter routes are
scientific candidates until their own replay reaches a terminal; failed or
divergent candidates are never promoted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .online_terminal_frontier import TerminalFrontierAction


@dataclass
class LevelRoute:
    """One exact or shortening-candidate route for a completed level."""

    route_id: str
    start_state_signature: str
    actions: Tuple[TerminalFrontierAction, ...]
    state_signatures: Tuple[str, ...] = ()
    parent_route_id: str = ""
    original_action_count: int = 0
    attempts: int = 0
    confirmations: int = 0
    refutations: int = 0
    divergences: int = 0
    censored: int = 0

    @property
    def shortening_candidate(self) -> bool:
        return bool(self.parent_route_id)

    @property
    def candidate_only(self) -> bool:
        return bool(self.shortening_candidate and self.confirmations == 0)

    @property
    def status(self) -> str:
        if self.confirmations:
            return "terminal_confirmed"
        if self.refutations:
            return "refuted"
        if self.divergences:
            return "inconclusive_divergence"
        if self.shortening_candidate:
            return "candidate_only"
        return "observed_terminal_route"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "start_state_signature": self.start_state_signature,
            "actions": [action.to_dict() for action in self.actions],
            "state_signatures": list(self.state_signatures),
            "parent_route_id": self.parent_route_id,
            "original_action_count": self.original_action_count,
            "attempts": self.attempts,
            "confirmations": self.confirmations,
            "refutations": self.refutations,
            "divergences": self.divergences,
            "censored": self.censored,
            "shortening_candidate": self.shortening_candidate,
            "candidate_only": self.candidate_only,
            "status": self.status,
        }


@dataclass(frozen=True)
class LevelRouteSelection:
    """One exact action selected from a remembered per-level route."""

    route_id: str
    action: TerminalFrontierAction
    step_index: int
    action_limit: int
    confirmed_route: bool
    shortening_candidate: bool
    reason: str


@dataclass
class _ActiveRoute:
    route: LevelRoute
    actions: list[TerminalFrontierAction]
    pending: TerminalFrontierAction | None = None


class OnlineLevelRouteMemory:
    """Compile, verify, replay, and shorten exact per-level routes online."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        enable_shortening: bool = True,
        max_routes: int = 48,
        max_actions_per_route: int = 4000,
        max_replay_attempts: int = 8,
    ) -> None:
        self.enabled = bool(enabled)
        self.enable_shortening = bool(enable_shortening)
        self.max_routes = max(1, int(max_routes))
        self.max_actions_per_route = max(1, int(max_actions_per_route))
        self.max_replay_attempts = max(1, int(max_replay_attempts))
        self._routes: Dict[str, LevelRoute] = {}
        self._branch_actions: list[TerminalFrontierAction] = []
        self._branch_states: list[str] = []
        self._active: _ActiveRoute | None = None
        self._route_start_checked = False
        self._routes_observed = 0
        self._route_replay_attempts = 0
        self._route_replay_actions = 0
        self._route_confirmations = 0
        self._route_refutations = 0
        self._route_divergences = 0
        self._route_censored = 0
        self._shortening_candidates = 0
        self._shortening_confirmations = 0
        self._shortening_refutations = 0
        self._shortening_actions_saved = 0
        self._completed_level_action_counts: list[int] = []

    @property
    def active_route_available(self) -> bool:
        if not self.enabled:
            return False
        if self._active is not None:
            return True
        if self._route_start_checked or self._branch_actions:
            return False
        return any(
            route.actions
            and route.refutations == 0
            and route.attempts < self.max_replay_attempts
            for route in self._routes.values()
        )

    def routes(self) -> Tuple[LevelRoute, ...]:
        return tuple(sorted(
            self._routes.values(),
            key=lambda item: item.route_id,
        ))

    def select(
        self,
        *,
        state_signature: str,
        available_actions: Sequence[str],
    ) -> LevelRouteSelection | None:
        if not self.enabled:
            return None
        state = str(state_signature)
        allowed = {str(action).upper() for action in available_actions}
        active = self._active
        if active is None:
            if self._route_start_checked or self._branch_actions:
                return None
            self._route_start_checked = True
            candidates = [
                route
                for route in self._routes.values()
                if route.start_state_signature == state
                and route.actions
                and route.refutations == 0
                and route.attempts < self.max_replay_attempts
            ]
            if not candidates:
                return None
            route = min(
                candidates,
                key=lambda item: (
                    0 if item.candidate_only else 1,
                    len(item.actions),
                    -item.confirmations,
                    item.attempts,
                    item.route_id,
                ),
            )
            route.attempts += 1
            self._route_replay_attempts += 1
            active = _ActiveRoute(route=route, actions=[])
            self._active = active
        if active.pending is not None:
            return None
        route = active.route
        step = len(active.actions)
        if step >= len(route.actions):
            self._active = None
            return None
        if (
            route.state_signatures
            and step < len(route.state_signatures)
            and route.state_signatures[step] != state
        ):
            route.divergences += 1
            self._route_divergences += 1
            self._active = None
            return None
        action = route.actions[step]
        if action.action_name not in allowed:
            route.divergences += 1
            self._route_divergences += 1
            self._active = None
            return None
        active.pending = action
        return LevelRouteSelection(
            route_id=route.route_id,
            action=action,
            step_index=step,
            action_limit=len(route.actions),
            confirmed_route=bool(route.confirmations),
            shortening_candidate=route.shortening_candidate,
            reason=(
                "verify shorter per-level route against a real terminal"
                if route.candidate_only
                else "replay shortest terminal-verified per-level route"
            ),
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
        if not self.enabled:
            return {
                "route_id": "",
                "route_observation": False,
                "route_confirmed": False,
                "route_refuted": False,
                "route_diverged": False,
                "shortening_candidate": False,
            }
        observed = TerminalFrontierAction.from_parts(
            action_name,
            action_data,
        )
        self._record_branch_transition(
            state_signature_before=str(state_signature_before),
            state_signature_after=str(state_signature_after),
            observed=observed,
        )
        active = self._active
        outcome = {
            "route_id": "",
            "route_observation": False,
            "route_confirmed": False,
            "route_refuted": False,
            "route_diverged": False,
            "shortening_candidate": False,
        }
        if active is not None and active.pending is not None:
            route = active.route
            expected = active.pending
            active.pending = None
            step = len(active.actions)
            expected_after = (
                route.state_signatures[step + 1]
                if route.state_signatures
                and step + 1 < len(route.state_signatures)
                else ""
            )
            exact = bool(
                observed.signature == expected.signature
                and (
                    not expected_after
                    or expected_after == str(state_signature_after)
                )
            )
            outcome.update({
                "route_id": route.route_id,
                "route_observation": True,
                "shortening_candidate": route.shortening_candidate,
            })
            if not exact:
                route.divergences += 1
                self._route_divergences += 1
                outcome["route_diverged"] = True
                self._active = None
            else:
                active.actions.append(observed)
                self._route_replay_actions += 1
                terminal_success = bool(level_progressed or won)
                complete = len(active.actions) >= len(route.actions)
                if terminal_success:
                    if not complete:
                        route.divergences += 1
                        self._route_divergences += 1
                        outcome["route_diverged"] = True
                    else:
                        route.confirmations += 1
                        self._route_confirmations += 1
                        self._completed_level_action_counts.append(
                            len(active.actions)
                        )
                        outcome["route_confirmed"] = True
                        if route.shortening_candidate:
                            self._shortening_confirmations += 1
                            self._shortening_actions_saved += max(
                                0,
                                route.original_action_count
                                - len(route.actions),
                            )
                        self._generate_shortening_candidate(route)
                    self._active = None
                elif game_over or complete:
                    route.refutations += 1
                    self._route_refutations += 1
                    outcome["route_refuted"] = True
                    if route.shortening_candidate:
                        self._shortening_refutations += 1
                        parent = self._routes.get(route.parent_route_id)
                        if parent is not None:
                            self._generate_shortening_candidate(parent)
                    self._active = None
        if (
            (level_progressed or won)
            and not outcome["route_confirmed"]
            and self._branch_actions
        ):
            self._record_observed_terminal_route()
        return outcome

    def start_branch(self) -> None:
        if self._active is not None:
            self._active.route.censored += 1
            self._route_censored += 1
        self._active = None
        self._branch_actions = []
        self._branch_states = []
        self._route_start_checked = False

    def summary(self) -> Dict[str, Any]:
        confirmed = [
            route for route in self._routes.values()
            if route.confirmations > 0
        ]
        average_actions = (
            sum(self._completed_level_action_counts)
            / len(self._completed_level_action_counts)
            if self._completed_level_action_counts else 0.0
        )
        return {
            "enabled": self.enabled,
            "shortening_enabled": self.enable_shortening,
            "routes": len(self._routes),
            "observed_routes": self._routes_observed,
            "confirmed_routes": len(confirmed),
            "route_replay_attempts": self._route_replay_attempts,
            "route_replay_actions": self._route_replay_actions,
            "route_confirmations": self._route_confirmations,
            "route_refutations": self._route_refutations,
            "route_divergences": self._route_divergences,
            "route_censored": self._route_censored,
            "shortening_candidates": self._shortening_candidates,
            "shortening_confirmations": self._shortening_confirmations,
            "shortening_refutations": self._shortening_refutations,
            "shortening_actions_saved": self._shortening_actions_saved,
            "completed_levels_measured": len(
                self._completed_level_action_counts
            ),
            "average_actions_per_completed_level": round(
                average_actions,
                4,
            ),
            "minimum_confirmed_route_length": min(
                (len(route.actions) for route in confirmed),
                default=0,
            ),
            "active_route_id": (
                "" if self._active is None else self._active.route.route_id
            ),
            "records": [route.to_dict() for route in self.routes()],
        }

    def _record_branch_transition(
        self,
        *,
        state_signature_before: str,
        state_signature_after: str,
        observed: TerminalFrontierAction,
    ) -> None:
        before = str(state_signature_before)
        after = str(state_signature_after)
        if not self._branch_states:
            self._branch_states = [before]
        elif self._branch_states[-1] != before:
            self._branch_actions = []
            self._branch_states = [before]
        if len(self._branch_actions) >= self.max_actions_per_route:
            return
        self._branch_actions.append(observed)
        self._branch_states.append(after)

    def _record_observed_terminal_route(self) -> None:
        actions = tuple(self._branch_actions)
        states = tuple(self._branch_states)
        if (
            not actions
            or len(states) != len(actions) + 1
            or len(self._routes) >= self.max_routes
        ):
            return
        route_id = _route_id(states[0], actions, parent_route_id="")
        route = self._routes.get(route_id)
        if route is None:
            route = LevelRoute(
                route_id=route_id,
                start_state_signature=states[0],
                actions=actions,
                state_signatures=states,
                original_action_count=len(actions),
                confirmations=1,
            )
            self._routes[route_id] = route
            self._routes_observed += 1
            self._route_confirmations += 1
        else:
            route.confirmations += 1
            self._route_confirmations += 1
        self._completed_level_action_counts.append(len(actions))
        self._generate_shortening_candidate(route)

    def _generate_shortening_candidate(self, route: LevelRoute) -> None:
        if (
            not self.enable_shortening
            or route.confirmations <= 0
            or len(route.actions) <= 1
            or len(self._routes) >= self.max_routes
        ):
            return
        length = len(route.actions)
        removals = []
        for span in (3, 2, 1):
            if span >= length:
                continue
            removals.extend(
                tuple(range(start, start + span))
                for start in range(0, length - span + 1)
            )
        for removed in removals:
            removed_set = set(removed)
            candidate_actions = tuple(
                action
                for index, action in enumerate(route.actions)
                if index not in removed_set
            )
            if not candidate_actions:
                continue
            candidate_id = _route_id(
                route.start_state_signature,
                candidate_actions,
                parent_route_id=route.route_id,
            )
            if candidate_id in self._routes:
                continue
            self._routes[candidate_id] = LevelRoute(
                route_id=candidate_id,
                start_state_signature=route.start_state_signature,
                actions=candidate_actions,
                parent_route_id=route.route_id,
                original_action_count=(
                    route.original_action_count or len(route.actions)
                ),
            )
            self._shortening_candidates += 1
            return


def _route_id(
    start_state_signature: str,
    actions: Sequence[TerminalFrontierAction],
    *,
    parent_route_id: str,
) -> str:
    payload = (
        str(start_state_signature),
        tuple(action.signature for action in actions),
        str(parent_route_id),
    )
    return (
        "level-route::"
        f"{hashlib.sha1(repr(payload).encode('utf-8')).hexdigest()[:16]}"
    )


__all__ = [
    "LevelRoute",
    "LevelRouteSelection",
    "OnlineLevelRouteMemory",
]
