"""Online structural-change signals for terminal frontier discovery.

SAGE.9i removes the dependency between terminal exploration and an already
completed goal hypothesis.  This detector observes only the live transition
record and emits a passive frontier signal when the frame exhibits a real
structural effect.  The signal is deliberately not a reward: it cannot promote
an action, objective, or continuation.

The vocabulary is game-agnostic and fixed before evaluation:

* ``mode_change`` -- canvas, background, or value vocabulary changed;
* ``entity_appearance`` / ``entity_disappearance`` -- connected components
  appeared or vanished;
* ``entity_motion`` -- an extracted object or the player moved;
* ``entity_transform`` -- the multiset of object types changed;
* ``relation_change`` -- coarse alignment/contact relations changed;
* ``structural_effect`` -- conservative fallback for any non-noop cell effect.

The terminal-frontier explorer still keys recurrence by the exact post-state.
These labels are diagnostic trigger evidence only, so they do not smuggle an
answer or a held-out representation into replay.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Tuple

import numpy as np

from v3.schemas import FrameDiff, ObjectInfo


@dataclass(frozen=True)
class StructuralFrontierSignal:
    """One terminal-neutral structural boundary observed online."""

    families: Tuple[str, ...]
    trigger_signature: str
    equivalence_signature: str
    changed_cells: int
    objects_before: int
    objects_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "trigger_signature": self.trigger_signature,
            "equivalence_signature": self.equivalence_signature,
            "changed_cells": self.changed_cells,
            "objects_before": self.objects_before,
            "objects_after": self.objects_after,
        }


class OnlineStructuralFrontierDetector:
    """Generate passive structural boundaries from live observations."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_changed_cells: int = 1,
        max_relation_objects: int = 48,
    ) -> None:
        self.enabled = bool(enabled)
        self.minimum_changed_cells = max(1, int(minimum_changed_cells))
        self.max_relation_objects = max(2, int(max_relation_objects))
        self._transitions = 0
        self._signals = 0
        self._captures = 0
        self._capture_blocks = 0
        self._noop_suppressions = 0
        self._terminal_suppressions = 0
        self._family_counts: Counter[str] = Counter()
        self._recent_signals: list[StructuralFrontierSignal] = []

    def observe_transition(
        self,
        *,
        grid_before: Any,
        grid_after: Any,
        objects_before: Sequence[ObjectInfo],
        objects_after: Sequence[ObjectInfo],
        diff: FrameDiff,
        terminal_success: bool = False,
        game_over: bool = False,
    ) -> StructuralFrontierSignal | None:
        """Return a boundary signal without assigning terminal value."""
        self._transitions += 1
        if not self.enabled:
            return None
        if terminal_success or game_over:
            self._terminal_suppressions += 1
            return None
        if int(diff.num_changed) < self.minimum_changed_cells:
            self._noop_suppressions += 1
            return None

        before = np.asarray(grid_before, dtype=np.int32)
        after = np.asarray(grid_after, dtype=np.int32)
        families: set[str] = set()
        if (
            before.shape != after.shape
            or _background_value(before) != _background_value(after)
            or set(int(value) for value in np.unique(before))
            != set(int(value) for value in np.unique(after))
        ):
            families.add("mode_change")
        if diff.created_objects:
            families.add("entity_appearance")
        if diff.removed_objects:
            families.add("entity_disappearance")
        if diff.moved_objects or diff.player_displacement is not None:
            families.add("entity_motion")
        if _object_type_multiset(objects_before) != _object_type_multiset(
            objects_after
        ):
            families.add("entity_transform")
        if _relation_signature(
            objects_before,
            limit=self.max_relation_objects,
        ) != _relation_signature(
            objects_after,
            limit=self.max_relation_objects,
        ):
            families.add("relation_change")
        if not families:
            # A cell-level causal effect is still a valid candidate boundary.
            # It remains neutral until terminal replay supplies evidence.
            families.add("structural_effect")

        ordered = tuple(sorted(families))
        payload = "|".join(
            (
                ",".join(ordered),
                str(int(diff.num_changed)),
                str(len(objects_before)),
                str(len(objects_after)),
                str(before.shape),
                str(after.shape),
            )
        )
        signal = StructuralFrontierSignal(
            families=ordered,
            trigger_signature=(
                f"structural-trigger::{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"
            ),
            equivalence_signature=_structural_equivalence_signature(
                after,
                objects_after,
                ordered,
            ),
            changed_cells=int(diff.num_changed),
            objects_before=len(objects_before),
            objects_after=len(objects_after),
        )
        self._signals += 1
        self._family_counts.update(ordered)
        self._recent_signals.append(signal)
        if len(self._recent_signals) > 16:
            self._recent_signals.pop(0)
        return signal

    def record_capture(self, frontier_id: str) -> None:
        """Record whether a signal actually opened a frontier trial."""
        if frontier_id:
            self._captures += 1
        else:
            self._capture_blocks += 1

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "minimum_changed_cells": self.minimum_changed_cells,
            "transitions_observed": self._transitions,
            "signals_generated": self._signals,
            "frontiers_captured": self._captures,
            "capture_blocks": self._capture_blocks,
            "noop_suppressions": self._noop_suppressions,
            "terminal_suppressions": self._terminal_suppressions,
            "family_counts": dict(sorted(self._family_counts.items())),
            "recent_signals": [
                signal.to_dict() for signal in self._recent_signals
            ],
        }


def _background_value(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _object_type(obj: ObjectInfo) -> tuple[Any, ...]:
    r_min, c_min, r_max, c_max = obj.bbox
    shape = tuple(
        sorted(
            (
                int(row - r_min),
                int(column - c_min),
            )
            for row, column in obj.cells
        )
    )
    return (
        int(obj.value),
        int(obj.area),
        int(r_max - r_min + 1),
        int(c_max - c_min + 1),
        shape,
    )


def _structural_equivalence_signature(
    grid: np.ndarray,
    objects: Sequence[ObjectInfo],
    families: Sequence[str],
) -> str:
    """Position-free signature used only to nominate online transfer tests."""
    values, counts = np.unique(grid, return_counts=True)
    value_counts = tuple(
        sorted((int(value), int(count)) for value, count in zip(values, counts))
    )
    descriptor = (
        tuple(int(item) for item in grid.shape),
        _background_value(grid),
        value_counts,
        _object_type_multiset(objects),
        tuple(sorted(str(item) for item in families)),
    )
    digest = hashlib.sha1(repr(descriptor).encode("utf-8")).hexdigest()[:16]
    return f"structural-equivalence::{digest}"


def _object_type_multiset(
    objects: Iterable[ObjectInfo],
) -> tuple[tuple[tuple[Any, ...], int], ...]:
    counts = Counter(_object_type(obj) for obj in objects)
    return tuple(sorted(counts.items(), key=lambda item: repr(item[0])))


def _relation_signature(
    objects: Sequence[ObjectInfo],
    *,
    limit: int,
) -> tuple[tuple[Any, ...], ...]:
    """Coarse relation multiset independent of transient object identifiers."""
    selected = sorted(
        ((obj, _object_type(obj)) for obj in objects),
        key=lambda item: (item[1], item[0].bbox, item[0].object_id),
    )[:limit]
    relations: Counter[tuple[Any, ...]] = Counter()
    for index, (left, left_type) in enumerate(selected):
        lr, lc = left.center
        l_min_r, l_min_c, l_max_r, l_max_c = left.bbox
        for right, right_type in selected[index + 1 :]:
            rr, rc = right.center
            r_min_r, r_min_c, r_max_r, r_max_c = right.bbox
            kinds: list[str] = []
            if int(round(lr)) == int(round(rr)):
                kinds.append("row_aligned")
            if int(round(lc)) == int(round(rc)):
                kinds.append("column_aligned")
            row_gap = max(0, r_min_r - l_max_r - 1, l_min_r - r_max_r - 1)
            col_gap = max(0, r_min_c - l_max_c - 1, l_min_c - r_max_c - 1)
            if row_gap == 0 and col_gap == 0:
                kinds.append("touching_or_overlapping")
            if not kinds:
                continue
            pair = tuple(sorted((left_type, right_type), key=repr))
            for kind in kinds:
                relations[(kind, pair)] += 1
    return tuple(
        sorted(
            ((kind, pair, count) for (kind, pair), count in relations.items()),
            key=repr,
        )
    )


__all__ = [
    "OnlineStructuralFrontierDetector",
    "StructuralFrontierSignal",
]
