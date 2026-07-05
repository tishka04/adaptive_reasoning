"""Context-conditioned action profiling.

Tracks action effects not just globally but *by context* — what the action
does depends on local state (adjacent objects, player position, etc.).
This is the foundation for operator induction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..schemas import (
    FrameDiff,
    GameObservation,
    TransitionRecord,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Context hashing
# =====================================================================

def compute_context_key(obs: GameObservation) -> Tuple[str, ...]:
    """Produce a discrete context key from the observation.

    Designed to be coarse enough to accumulate stats quickly,
    fine enough to distinguish meaningfully different situations.
    """
    parts: List[str] = []
    player = obs.best_player
    grid = obs.raw_grid

    # Player presence
    if player is not None and player.confidence > 0.4:
        parts.append("player_yes")
        r, c = player.position
        H, W = grid.shape

        # Adjacent walls / objects
        for name, dr, dc in [("up", -1, 0), ("down", 1, 0),
                              ("left", 0, -1), ("right", 0, 1)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                parts.append(f"wall_{name}")
            elif int(grid[nr, nc]) != 0:
                parts.append(f"obj_{name}_v{int(grid[nr, nc])}")
            else:
                parts.append(f"free_{name}")
    else:
        parts.append("player_no")

    # Danger zone
    if obs.danger_map is not None and player is not None:
        r, c = player.position
        if 0 <= r < obs.danger_map.shape[0] and 0 <= c < obs.danger_map.shape[1]:
            if float(obs.danger_map[r, c]) > 0.3:
                parts.append("danger")

    return tuple(parts)


# =====================================================================
# Per-context stats
# =====================================================================

@dataclass
class ContextActionStats:
    """Stats for one action in one context bucket."""
    tries: int = 0
    changes: int = 0                      # transitions with any grid change
    deaths: int = 0
    wins: int = 0
    mean_diff_size: float = 0.0
    mean_disp: Tuple[float, float] = (0.0, 0.0)  # avg (dy, dx)

    # Running sums for incremental mean
    _sum_diff_size: float = 0.0
    _sum_dy: float = 0.0
    _sum_dx: float = 0.0

    @property
    def change_rate(self) -> float:
        return self.changes / max(self.tries, 1)

    @property
    def death_rate(self) -> float:
        return self.deaths / max(self.tries, 1)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.tries, 1)

    def record(self, diff: FrameDiff) -> None:
        self.tries += 1
        if diff.num_changed > 0:
            self.changes += 1
        if diff.game_over:
            self.deaths += 1
        if diff.level_complete:
            self.wins += 1

        self._sum_diff_size += diff.num_changed
        self.mean_diff_size = self._sum_diff_size / self.tries

        if diff.player_displacement:
            dy, dx = diff.player_displacement
            self._sum_dy += dy
            self._sum_dx += dx
        self.mean_disp = (self._sum_dy / self.tries, self._sum_dx / self.tries)


# =====================================================================
# Global action stats (aggregated across contexts)
# =====================================================================

@dataclass
class ActionStats:
    """Aggregated stats for one action across all contexts."""
    total_tries: int = 0
    total_changes: int = 0
    deaths: int = 0
    wins: int = 0
    avg_player_disp: Tuple[float, float] = (0.0, 0.0)
    context_buckets: Dict[Tuple[str, ...], ContextActionStats] = field(
        default_factory=dict
    )

    # Running sums
    _sum_dy: float = 0.0
    _sum_dx: float = 0.0

    @property
    def change_rate(self) -> float:
        return self.total_changes / max(self.total_tries, 1)

    @property
    def death_rate(self) -> float:
        return self.deaths / max(self.total_tries, 1)

    def record(self, ctx_key: Tuple[str, ...], diff: FrameDiff) -> None:
        self.total_tries += 1
        if diff.num_changed > 0:
            self.total_changes += 1
        if diff.game_over:
            self.deaths += 1
        if diff.level_complete:
            self.wins += 1

        if diff.player_displacement:
            dy, dx = diff.player_displacement
            self._sum_dy += dy
            self._sum_dx += dx
        self.avg_player_disp = (
            self._sum_dy / self.total_tries,
            self._sum_dx / self.total_tries,
        )

        # Context bucket
        if ctx_key not in self.context_buckets:
            self.context_buckets[ctx_key] = ContextActionStats()
        self.context_buckets[ctx_key].record(diff)


# =====================================================================
# ActionProfiler
# =====================================================================

class ActionProfiler:
    """Context-conditioned action profiler.

    Tracks per-action, per-context statistics from observed transitions.
    Feeds directly into operator induction and experiment design.
    """

    def __init__(self) -> None:
        self.stats: Dict[str, ActionStats] = {}       # action_name → stats
        self.transitions: List[TransitionRecord] = []  # bounded recent history
        self._max_transitions: int = 500

    def record_transition(self, record: TransitionRecord) -> None:
        """Record one observed transition."""
        action_name = record.action.name
        ctx_key = compute_context_key(record.obs_before)

        if action_name not in self.stats:
            self.stats[action_name] = ActionStats()
        self.stats[action_name].record(ctx_key, record.diff)

        self.transitions.append(record)
        if len(self.transitions) > self._max_transitions:
            self.transitions = self.transitions[-self._max_transitions:]

        logger.debug(
            f"Profiler: {action_name} in ctx={ctx_key[:3]}.. → "
            f"changed={record.diff.num_changed}, "
            f"disp={record.diff.player_displacement}"
        )

    @property
    def total_transitions(self) -> int:
        """How many recent transitions are currently retained."""
        return len(self.transitions)

    def num_actions_with_tries(self, min_tries: int = 1) -> int:
        """How many primitive actions have at least ``min_tries`` observations."""
        return sum(
            1 for stats in self.stats.values()
            if stats.total_tries >= min_tries
        )

    def get_stats(self, action_name: str) -> Optional[ActionStats]:
        return self.stats.get(action_name)

    def get_context_stats(
        self, action_name: str, ctx_key: Tuple[str, ...]
    ) -> Optional[ContextActionStats]:
        s = self.stats.get(action_name)
        if s is None:
            return None
        return s.context_buckets.get(ctx_key)

    def action_coverage(self, available: List[str], min_tries: int = 2) -> float:
        """Fraction of available actions tried at least min_tries times."""
        if not available:
            return 0.0
        covered = sum(
            1 for a in available
            if a in self.stats and self.stats[a].total_tries >= min_tries
        )
        return covered / len(available)

    def most_uncertain_action(self, available: List[str]) -> Optional[str]:
        """Return the available action with fewest tries (most uncertain)."""
        candidates = [(a, self.stats.get(a)) for a in available]
        # Untried actions first
        untried = [a for a, s in candidates if s is None or s.total_tries == 0]
        if untried:
            return untried[0]
        # Then least-tried
        tried = [(a, s.total_tries) for a, s in candidates if s is not None]
        if tried:
            return min(tried, key=lambda x: x[1])[0]
        return available[0] if available else None

    def is_consistent(self, action_name: str, min_tries: int = 3) -> bool:
        """Check if action has a consistent effect (for mechanism locking)."""
        s = self.stats.get(action_name)
        if s is None or s.total_tries < min_tries:
            return False
        cr = s.change_rate
        return cr >= 0.8 or cr <= 0.15

    def dominant_displacement(self, action_name: str) -> Optional[Tuple[int, int]]:
        """Return the dominant player displacement for this action, if any."""
        s = self.stats.get(action_name)
        if s is None or s.total_tries < 3:
            return None
        dy, dx = s.avg_player_disp
        # Only count as movement if consistently displaces
        if abs(dy) > 0.4 or abs(dx) > 0.4:
            return (1 if dy > 0.4 else (-1 if dy < -0.4 else 0),
                    1 if dx > 0.4 else (-1 if dx < -0.4 else 0))
        return None

    def least_tried_actions(
        self, available: List[str], top_k: int = 3,
    ) -> List[str]:
        """Return the top_k least-tried available actions (excluding RESET)."""
        scored = []
        for a in available:
            if a == "RESET":
                continue
            s = self.stats.get(a)
            tries = s.total_tries if s is not None else 0
            scored.append((tries, a))
        scored.sort()
        return [a for _, a in scored[:top_k]]

    def summary(self) -> str:
        """Human-readable summary of all profiled actions."""
        lines = []
        for name, s in sorted(self.stats.items()):
            lines.append(
                f"  {name}: tries={s.total_tries}, "
                f"Δ={s.change_rate:.2f}, "
                f"☠={s.death_rate:.2f}, "
                f"disp=({s.avg_player_disp[0]:.1f},{s.avg_player_disp[1]:.1f}), "
                f"contexts={len(s.context_buckets)}"
            )
        return "\n".join(lines) if lines else "  (no actions profiled)"
