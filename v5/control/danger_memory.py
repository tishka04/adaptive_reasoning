"""Persistent danger memory for V5 (ported from learned_guided_search).

A small, self-contained observation store: which ``(state, action)`` pairs
have been observed to cause a GAME_OVER. Unlike a learned danger *model*, this
is a hard, observation-dominated veto — once a wall is hit, the agent refuses
to walk into it again from the same abstract state.

In the original greedy controller this was the single most effective organ:
``danger_memory[(state_sig, action)] = 1.0`` applied a ``-1e6`` penalty that
dominated any model prediction. Here it is exposed as a reusable module so the
structured V5 agent can veto lethal actions at selection time (V3 only recorded
a weak ``death_<hash>`` failure motif that never gated the chosen action).

The state signature is the grid hash of the state *before* the action; the
action key is the primitive name plus, for clicks, the ``(x, y)`` target.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from ..schemas import PrimitiveAction

# (grid_hash, action_key) -> 1.0 when observed lethal.
DangerKey = Tuple[int, str]


def action_key(primitive: PrimitiveAction) -> str:
    """Stable key for a primitive: name, plus coarse (x, y) for clicks."""
    if primitive.x is not None or primitive.y is not None:
        return f"{primitive.name}@{primitive.x},{primitive.y}"
    return primitive.name


class DangerMemoryV5:
    """Observation-dominated lethal-action veto, shareable across a game."""

    def __init__(self) -> None:
        self._deaths: Dict[DangerKey, float] = {}
        self.records: int = 0  # total record_death calls (incl. repeats)

    # -- writes --------------------------------------------------------
    def record_death(self, grid_hash: int, key: str) -> None:
        """Remember that ``key`` led to a GAME_OVER from ``grid_hash``."""
        self._deaths[(int(grid_hash), str(key))] = 1.0
        self.records += 1

    def record_primitive_death(self, grid_hash: int, primitive: PrimitiveAction) -> None:
        self.record_death(grid_hash, action_key(primitive))

    # -- reads ---------------------------------------------------------
    def is_lethal(self, grid_hash: int, key: str) -> bool:
        return self._deaths.get((int(grid_hash), str(key)), 0.0) >= 1.0

    def is_primitive_lethal(self, grid_hash: int, primitive: PrimitiveAction) -> bool:
        return self.is_lethal(grid_hash, action_key(primitive))

    def lethal_names(self, grid_hash: int, available: Optional[list] = None) -> Set[str]:
        """Return the set of *name-only* actions known lethal in this state.

        Click targets collapse to the bare action name only when every recorded
        variant is lethal-by-name; otherwise name-level moves stay allowed.
        """
        gh = int(grid_hash)
        names: Set[str] = set()
        for (h, key), val in self._deaths.items():
            if h != gh or val < 1.0:
                continue
            names.add(key.split("@", 1)[0])
        if available is not None:
            names &= set(available)
        return names

    def __len__(self) -> int:
        return len(self._deaths)

    # -- persistence (for future cross-game / cross-attempt seeding) ---
    def to_dict(self) -> Dict[str, float]:
        return {f"{h}|{key}": v for (h, key), v in self._deaths.items()}

    @classmethod
    def from_dict(cls, payload: Dict[str, float]) -> "DangerMemoryV5":
        mem = cls()
        for flat, val in (payload or {}).items():
            h_str, _, key = flat.partition("|")
            try:
                mem._deaths[(int(h_str), key)] = float(val)
            except ValueError:
                continue
        return mem

    def merge(self, other: "DangerMemoryV5") -> None:
        for k, v in other._deaths.items():
            if v >= 1.0:
                self._deaths[k] = 1.0
