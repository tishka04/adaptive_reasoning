"""Ontology-conditioned action profiling for V4."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..schemas import ObservationV4, TransitionRecord


def compute_context_key(obs: ObservationV4) -> tuple[str, ...]:
    parts: list[str] = []
    player = obs.best_player
    grid = obs.raw_grid

    if player is None:
        parts.append("player_no")
        return tuple(parts)

    parts.append("player_yes")
    r, c = player.position
    h, w = grid.shape
    for name, dr, dc in [("u", -1, 0), ("d", 1, 0), ("l", 0, -1), ("r", 0, 1)]:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            parts.append(f"wall_{name}")
        else:
            value = int(grid[nr, nc])
            if value == 0:
                parts.append(f"free_{name}")
            else:
                parts.append(f"obj_{name}_{value}")
    parts.append(f"objs_{min(len(obs.objects), 8)}")
    return tuple(parts)


@dataclass
class ContextActionStats:
    tries: int = 0
    changes: int = 0
    deaths: int = 0
    wins: int = 0
    mean_diff_size: float = 0.0
    mean_disp: tuple[float, float] = (0.0, 0.0)
    coord_uses: int = 0

    _sum_diff_size: float = 0.0
    _sum_dy: float = 0.0
    _sum_dx: float = 0.0

    def record(self, transition: TransitionRecord) -> None:
        diff = transition.diff
        self.tries += 1
        if diff.num_changed > 0:
            self.changes += 1
        if diff.game_over:
            self.deaths += 1
        if diff.level_complete:
            self.wins += 1
        if transition.action.x is not None:
            self.coord_uses += 1

        self._sum_diff_size += diff.num_changed
        self.mean_diff_size = self._sum_diff_size / self.tries
        if diff.player_displacement is not None:
            dy, dx = diff.player_displacement
            self._sum_dy += dy
            self._sum_dx += dx
        self.mean_disp = (self._sum_dy / self.tries, self._sum_dx / self.tries)

    @property
    def change_rate(self) -> float:
        return self.changes / max(self.tries, 1)

    @property
    def death_rate(self) -> float:
        return self.deaths / max(self.tries, 1)


@dataclass
class ActionStats:
    total_tries: int = 0
    total_changes: int = 0
    deaths: int = 0
    wins: int = 0
    avg_player_disp: tuple[float, float] = (0.0, 0.0)
    context_buckets: dict[tuple[str, ...], ContextActionStats] = field(default_factory=dict)
    ontology_tries: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    ontology_changes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    coord_uses: int = 0

    _sum_dy: float = 0.0
    _sum_dx: float = 0.0

    def record(
        self,
        ctx_key: tuple[str, ...],
        transition: TransitionRecord,
        ontology_id: str,
    ) -> None:
        diff = transition.diff
        self.total_tries += 1
        self.ontology_tries[ontology_id] += 1
        if diff.num_changed > 0:
            self.total_changes += 1
            self.ontology_changes[ontology_id] += 1
        if diff.game_over:
            self.deaths += 1
        if diff.level_complete:
            self.wins += 1
        if transition.action.x is not None:
            self.coord_uses += 1

        if diff.player_displacement is not None:
            dy, dx = diff.player_displacement
            self._sum_dy += dy
            self._sum_dx += dx
        self.avg_player_disp = (
            self._sum_dy / self.total_tries,
            self._sum_dx / self.total_tries,
        )

        if ctx_key not in self.context_buckets:
            self.context_buckets[ctx_key] = ContextActionStats()
        self.context_buckets[ctx_key].record(transition)

    @property
    def change_rate(self) -> float:
        return self.total_changes / max(self.total_tries, 1)

    @property
    def death_rate(self) -> float:
        return self.deaths / max(self.total_tries, 1)

    def ontology_change_rate(self, ontology_id: str) -> float:
        return self.ontology_changes.get(ontology_id, 0) / max(
            self.ontology_tries.get(ontology_id, 0), 1
        )


class ActionProfiler:
    """Track action effects globally, by context, and by ontology."""

    def __init__(self) -> None:
        self.stats: dict[str, ActionStats] = {}
        self.transitions: list[TransitionRecord] = []
        self._max_transitions = 800

    def update(
        self,
        obs: ObservationV4,
        action,
        transition: TransitionRecord,
        ontology_id: str,
    ) -> None:
        action_name = action.name
        ctx_key = compute_context_key(obs)
        if action_name not in self.stats:
            self.stats[action_name] = ActionStats()
        self.stats[action_name].record(ctx_key, transition, ontology_id)
        self.transitions.append(transition)
        if len(self.transitions) > self._max_transitions:
            self.transitions = self.transitions[-self._max_transitions :]

    @property
    def total_transitions(self) -> int:
        return len(self.transitions)

    def get_stats(self, action_name: str) -> Optional[ActionStats]:
        return self.stats.get(action_name)

    def action_coverage(self, available: list[str], min_tries: int = 2) -> float:
        if not available:
            return 0.0
        covered = sum(
            1
            for action in available
            if action in self.stats and self.stats[action].total_tries >= min_tries
        )
        return covered / max(len(available), 1)

    def dominant_displacement(self, action_name: str) -> Optional[tuple[int, int]]:
        stats = self.stats.get(action_name)
        if stats is None or stats.total_tries < 2:
            return None
        dy, dx = stats.avg_player_disp
        if abs(dy) > 0.35 or abs(dx) > 0.35:
            return (
                1 if dy > 0.35 else (-1 if dy < -0.35 else 0),
                1 if dx > 0.35 else (-1 if dx < -0.35 else 0),
            )
        return None

    def least_tried_actions(self, available: list[str], k: int = 3) -> list[str]:
        ranked = sorted(
            available,
            key=lambda action: self.stats.get(action).total_tries if action in self.stats else 0,
        )
        return ranked[:k]

    def num_actions_with_tries(self, min_tries: int = 1) -> int:
        return sum(1 for stat in self.stats.values() if stat.total_tries >= min_tries)
