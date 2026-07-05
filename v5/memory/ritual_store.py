"""Ritual store for V5.

Compact, bounded store of compiled `(prefix, terminal_signature)`
templates. Survives pruning by `survival_score`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..schemas_ext import Ritual


class RitualStore:
    """Bounded, ontology-tagged ritual library."""

    def __init__(self, max_active: int = 6) -> None:
        self._rituals: Dict[str, Ritual] = {}
        self.max_active = max_active

    # -----------------------------------------------------------------
    def add(self, ritual: Ritual) -> None:
        existing = self._rituals.get(ritual.ritual_id)
        if existing is None:
            ritual.survival_score = self._score(ritual)
            self._rituals[ritual.ritual_id] = ritual
        else:
            existing.success_rate = max(existing.success_rate, ritual.success_rate)
            if len(ritual.prefix) < len(existing.prefix):
                existing.prefix = ritual.prefix
            existing.terminal_signature = ritual.terminal_signature
            existing.survival_score = self._score(existing)
        self._prune()

    def all(self) -> List[Ritual]:
        return sorted(
            self._rituals.values(),
            key=lambda r: r.survival_score,
            reverse=True,
        )

    def best_for(self, ontology_kind: str) -> Optional[Ritual]:
        candidates = [
            r for r in self._rituals.values() if r.ontology_kind == ontology_kind
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.success_rate, -len(r.prefix)))

    def __len__(self) -> int:
        return len(self._rituals)

    # -----------------------------------------------------------------
    def _score(self, ritual: Ritual) -> float:
        brevity = 1.0 / max(len(ritual.prefix), 1)
        levels = float(ritual.terminal_signature.get("levels_completed", 0))
        return (
            0.65 * ritual.success_rate
            + 0.20 * brevity
            + 0.15 * min(1.0, levels / 3.0)
        )

    def _prune(self) -> None:
        if len(self._rituals) <= self.max_active:
            return
        ranked = sorted(
            self._rituals.values(),
            key=lambda r: r.survival_score,
            reverse=True,
        )
        self._rituals = {r.ritual_id: r for r in ranked[: self.max_active]}
