"""Anti-attractor control for V5 (ported from learned_guided_search).

Two channels that together break the local optima the greedy controller and
the cold structured agents both fall into on hard games (e.g. ar25, where V5
burned 66k actions stuck at level 0 with ontology=[global_transform, noop]):

  Channel 1 — no-op ban
      An action repeatedly observed to produce **zero** changed cells in a
      given abstract state is banned there. Callers veto/penalise it.

  Channel 2 — soft preventive escape
      Fire *early* (before a long strict no-op stall) when any of:
        (a) strict no-op stall          (steps_since_change >= STAGNATION_ESCAPE_AFTER)
        (b) action-repetition loop       (last REPEAT_WINDOW actions <= 2 distinct)
        (c) low state novelty            (last NOVELTY_WINDOW states <= NOVELTY_MIN_UNIQUE)
      On escape, prefer a simple (non-click) action not used recently, not
      known-lethal, and with the fewest observed no-ops in this state. A
      cooldown avoids thrashing.

This module is observation-only; it never steps the env. The agent feeds it
``observe(...)`` after every transition and queries ``should_escape`` /
``pick_escape_action`` / ``is_banned_noop`` during decision making.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

NoEffectKey = Tuple[int, str]


class AntiAttractor:
    def __init__(
        self,
        *,
        no_effect_ban_after: int = 3,
        stagnation_escape_after: int = 8,
        repeat_window: int = 8,
        repeat_max_distinct: int = 2,
        novelty_window: int = 12,
        novelty_min_unique: int = 3,
        escape_ban_last_k: int = 3,
        escape_cooldown: int = 5,
    ) -> None:
        self.NO_EFFECT_BAN_AFTER = int(no_effect_ban_after)
        self.STAGNATION_ESCAPE_AFTER = int(stagnation_escape_after)
        self.REPEAT_WINDOW = int(repeat_window)
        self.REPEAT_MAX_DISTINCT = int(repeat_max_distinct)
        self.NOVELTY_WINDOW = int(novelty_window)
        self.NOVELTY_MIN_UNIQUE = int(novelty_min_unique)
        self.ESCAPE_BAN_LAST_K = int(escape_ban_last_k)
        self.ESCAPE_COOLDOWN = int(escape_cooldown)

        self._no_effect_count: Dict[NoEffectKey, int] = {}
        self._action_usage: Dict[str, int] = {}
        self._action_history: List[str] = []
        self._recent_state_sigs: List[int] = []
        self._steps_since_change: int = 0
        self._last_escape_step: int = -10 ** 9

    # -----------------------------------------------------------------
    def observe(self, *, grid_hash: int, action_name: str, is_noop: bool) -> None:
        """Record one transition's outcome (call once per step)."""
        self._action_usage[action_name] = self._action_usage.get(action_name, 0) + 1
        self._action_history.append(action_name)
        self._recent_state_sigs.append(int(grid_hash))
        if len(self._recent_state_sigs) > self.NOVELTY_WINDOW:
            self._recent_state_sigs.pop(0)
        self._steps_since_change = self._steps_since_change + 1 if is_noop else 0

    def note_no_effect(self, grid_hash: int, action_name: str) -> None:
        """Record that ``action_name`` was a no-op in ``grid_hash``.

        ``grid_hash`` here is the state *before* the action (the no-op origin),
        matching the veto query in ``is_banned_noop``.
        """
        key = (int(grid_hash), str(action_name))
        self._no_effect_count[key] = self._no_effect_count.get(key, 0) + 1

    # -----------------------------------------------------------------
    def is_banned_noop(self, grid_hash: int, action_name: str) -> bool:
        return (
            self._no_effect_count.get((int(grid_hash), str(action_name)), 0)
            >= self.NO_EFFECT_BAN_AFTER
        )

    def should_escape(self, current_step: int) -> bool:
        if (current_step - self._last_escape_step) <= self.ESCAPE_COOLDOWN:
            return False
        repeat_loop = (
            len(self._action_history) >= self.REPEAT_WINDOW
            and len(set(self._action_history[-self.REPEAT_WINDOW:])) <= self.REPEAT_MAX_DISTINCT
        )
        novelty_stall = (
            len(self._recent_state_sigs) >= self.NOVELTY_WINDOW
            and len(set(self._recent_state_sigs[-self.NOVELTY_WINDOW:])) <= self.NOVELTY_MIN_UNIQUE
        )
        return (
            self._steps_since_change >= self.STAGNATION_ESCAPE_AFTER
            or repeat_loop
            or novelty_stall
        )

    def pick_escape_action(
        self,
        *,
        available: List[str],
        grid_hash: int,
        is_lethal: Optional[Callable[[int, str], bool]] = None,
    ) -> Optional[str]:
        """Choose a fresh, safe action to break the attractor.

        Prefer non-click actions; fall back to clicks only if nothing else is
        viable. Among candidates, pick the one with the fewest observed no-ops
        in this state, then least overall usage.
        """
        gh = int(grid_hash)
        banned = set(self._action_history[-self.ESCAPE_BAN_LAST_K:])

        def _viable(pool: List[str]) -> List[str]:
            out = []
            for a in pool:
                if a in banned or a == "RESET":
                    continue
                if is_lethal is not None and is_lethal(gh, a):
                    continue
                out.append(a)
            return out

        candidates = _viable([a for a in available if a != "ACTION6"])
        if not candidates:
            candidates = _viable(list(available))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda a: (
                self._no_effect_count.get((gh, a), 0),
                self._action_usage.get(a, 0),
            ),
        )

    def note_escape(self, current_step: int) -> None:
        self._last_escape_step = int(current_step)

    # -----------------------------------------------------------------
    @property
    def steps_since_change(self) -> int:
        return self._steps_since_change
