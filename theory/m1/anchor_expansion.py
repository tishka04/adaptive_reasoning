"""M1.3b source-target anchor expansion.

This module proposes extra source-target color anchors from raw object changes.
It never confirms a hypothesis; every emitted anchor is still only trace-backed
material for A12/A25 to test later.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np


AnchorExpander = Callable[..., Iterable["ExpandedAnchor"]]


@dataclass(frozen=True)
class ExpandedAnchor:
    """One trace-derived source-target anchor candidate."""

    source_color: int
    target_color: int
    support: int = 1
    predicates: Tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class _Component:
    color: int
    points: Tuple[Tuple[int, int], ...]
    bbox: Tuple[int, int, int, int]

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            sum(y for y, _ in self.points) / max(1, self.size),
            sum(x for _, x in self.points) / max(1, self.size),
        )

    @property
    def shape_signature(self) -> Tuple[Tuple[int, int], ...]:
        min_y = min((point[0] for point in self.points), default=0)
        min_x = min((point[1] for point in self.points), default=0)
        return tuple(sorted((y - min_y, x - min_x) for y, x in self.points))


def build_m1_anchor_expander(
    *,
    max_created_removed_distance: float = 6.0,
    max_motion_distance: float = 16.0,
) -> AnchorExpander:
    """Build the deterministic M1.3b anchor expander."""

    def _expander(
        before: np.ndarray,
        after: np.ndarray,
        *,
        background: int | None = None,
    ) -> Tuple[ExpandedAnchor, ...]:
        return expand_anchors(
            before,
            after,
            background=background,
            max_created_removed_distance=max_created_removed_distance,
            max_motion_distance=max_motion_distance,
        )

    return _expander


def expand_anchors(
    before: np.ndarray,
    after: np.ndarray,
    *,
    background: int | None = None,
    max_created_removed_distance: float = 6.0,
    max_motion_distance: float = 16.0,
) -> Tuple[ExpandedAnchor, ...]:
    """Return object-centric source-target anchors for one transition."""
    before_grid = np.asarray(before, dtype=np.int32)
    after_grid = np.asarray(after, dtype=np.int32)
    if before_grid.shape != after_grid.shape or before_grid.ndim != 2:
        return ()
    bg = _infer_background(before_grid, after_grid) if background is None else int(background)
    before_components = _components(before_grid, background=bg)
    after_components = _components(after_grid, background=bg)
    anchors: List[ExpandedAnchor] = []
    anchors.extend(
        _created_removed_anchors(
            before_grid,
            after_grid,
            before_components,
            after_components,
            max_distance=max_created_removed_distance,
        )
    )
    anchors.extend(
        _motion_anchors(
            before_grid,
            after_grid,
            before_components,
            after_components,
            background=bg,
            max_distance=max_motion_distance,
        )
    )
    anchors.extend(_contact_anchors(before_grid, after_grid, background=bg))
    return _merge_anchors(anchors)


def _created_removed_anchors(
    before: np.ndarray,
    after: np.ndarray,
    before_components: Sequence[_Component],
    after_components: Sequence[_Component],
    *,
    max_distance: float,
) -> List[ExpandedAnchor]:
    removed, created = _unmatched_components(before_components, after_components)
    anchors: List[ExpandedAnchor] = []
    for source in removed:
        for target in created:
            if source.color == target.color:
                continue
            score = _component_similarity(source, target)
            if score <= 0:
                continue
            distance = _centroid_distance(source, target)
            if distance > float(max_distance) and source.shape_signature != target.shape_signature:
                continue
            predicates = _relation_predicates_for_pair(
                before,
                after,
                source.color,
                target.color,
                extra=("m1_anchor_created_removed",),
            )
            support = max(1, min(source.size, target.size))
            anchors.append(
                ExpandedAnchor(
                    source_color=source.color,
                    target_color=target.color,
                    support=support,
                    predicates=predicates,
                    reason="created_removed",
                )
            )
    return anchors


def _motion_anchors(
    before: np.ndarray,
    after: np.ndarray,
    before_components: Sequence[_Component],
    after_components: Sequence[_Component],
    *,
    background: int,
    max_distance: float,
) -> List[ExpandedAnchor]:
    moved = _moved_components(
        before_components,
        after_components,
        max_distance=max_distance,
    )
    anchors: List[ExpandedAnchor] = []
    for source, target in moved:
        touched_colors = _motion_contact_colors(
            before,
            after,
            source,
            target,
            background=background,
        )
        if not touched_colors:
            touched_colors = _nearby_colors(after, target, radius=1, background=background)
        for color in sorted(touched_colors):
            if color == source.color:
                continue
            predicates = _relation_predicates_for_pair(
                before,
                after,
                source.color,
                color,
                extra=("m1_anchor_motion",),
            )
            anchors.append(
                ExpandedAnchor(
                    source_color=source.color,
                    target_color=int(color),
                    support=max(1, source.size),
                    predicates=predicates,
                    reason="motion",
                )
            )
    return anchors


def _contact_anchors(
    before: np.ndarray,
    after: np.ndarray,
    *,
    background: int,
) -> List[ExpandedAnchor]:
    before_contacts = _color_contacts(before, background=background)
    after_contacts = _color_contacts(after, background=background)
    modified = before_contacts.symmetric_difference(after_contacts)
    anchors: List[ExpandedAnchor] = []
    for source, target in sorted(modified):
        predicates = _relation_predicates_for_pair(
            before,
            after,
            source,
            target,
            extra=("m1_anchor_contact",),
        )
        anchors.append(
            ExpandedAnchor(
                source_color=int(source),
                target_color=int(target),
                support=1,
                predicates=predicates,
                reason="contact",
            )
        )
    return anchors


def _unmatched_components(
    before: Sequence[_Component],
    after: Sequence[_Component],
) -> Tuple[List[_Component], List[_Component]]:
    used_after: set[int] = set()
    matched_before: set[int] = set()
    for before_index, source in enumerate(before):
        candidates = [
            (index, target)
            for index, target in enumerate(after)
            if index not in used_after
            and source.color == target.color
            and source.size == target.size
            and source.shape_signature == target.shape_signature
        ]
        if not candidates:
            continue
        target_index, _ = min(
            candidates,
            key=lambda item: _centroid_distance(source, item[1]),
        )
        used_after.add(target_index)
        matched_before.add(before_index)
    removed = [
        component for index, component in enumerate(before) if index not in matched_before
    ]
    created = [
        component for index, component in enumerate(after) if index not in used_after
    ]
    return removed, created


def _moved_components(
    before: Sequence[_Component],
    after: Sequence[_Component],
    *,
    max_distance: float,
) -> List[Tuple[_Component, _Component]]:
    used_after: set[int] = set()
    moved: List[Tuple[_Component, _Component]] = []
    for source in before:
        candidates = [
            (index, target)
            for index, target in enumerate(after)
            if index not in used_after
            and source.color == target.color
            and source.size == target.size
            and source.shape_signature == target.shape_signature
        ]
        if not candidates:
            continue
        target_index, target = min(
            candidates,
            key=lambda item: _centroid_distance(source, item[1]),
        )
        used_after.add(target_index)
        distance = _centroid_distance(source, target)
        if 0 < distance <= float(max_distance):
            moved.append((source, target))
    return moved


def _motion_contact_colors(
    before: np.ndarray,
    after: np.ndarray,
    source: _Component,
    target: _Component,
    *,
    background: int,
) -> set[int]:
    before_neighbors = _nearby_colors(
        before,
        source,
        radius=1,
        background=background,
    )
    after_neighbors = _nearby_colors(
        after,
        target,
        radius=1,
        background=background,
    )
    return before_neighbors.symmetric_difference(after_neighbors)


def _nearby_colors(
    grid: np.ndarray,
    component: _Component,
    *,
    radius: int,
    background: int,
) -> set[int]:
    colors: set[int] = set()
    height, width = grid.shape
    owned = set(component.points)
    for y, x in component.points:
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in owned:
                    continue
                if 0 <= ny < height and 0 <= nx < width:
                    color = int(grid[ny, nx])
                    if color != int(background) and color != component.color:
                        colors.add(color)
    return colors


def _relation_predicates_for_pair(
    before: np.ndarray,
    after: np.ndarray,
    source_color: int,
    target_color: int,
    *,
    extra: Sequence[str],
) -> Tuple[str, ...]:
    predicates: List[str] = list(extra)
    if _both_present(before, source_color, target_color) or _both_present(
        after,
        source_color,
        target_color,
    ):
        predicates.append("paired_with")
    if _same_shape_exists(before, source_color, target_color) or _same_shape_exists(
        after,
        source_color,
        target_color,
    ):
        predicates.append("same_shape")
    if _aligned_exists(before, source_color, target_color) or _aligned_exists(
        after,
        source_color,
        target_color,
    ):
        predicates.append("aligned_with")
    if _adjacent_exists(before, source_color, target_color) or _adjacent_exists(
        after,
        source_color,
        target_color,
    ):
        predicates.append("adjacent_to")
    return tuple(_dedupe(predicates))


def _merge_anchors(anchors: Iterable[ExpandedAnchor]) -> Tuple[ExpandedAnchor, ...]:
    support: Counter[Tuple[int, int]] = Counter()
    predicates: dict[Tuple[int, int], Counter[str]] = {}
    reasons: dict[Tuple[int, int], List[str]] = {}
    for anchor in anchors:
        source = int(anchor.source_color)
        target = int(anchor.target_color)
        if source == target:
            continue
        key = (source, target)
        support[key] += max(1, int(anchor.support))
        predicates.setdefault(key, Counter()).update(str(item) for item in anchor.predicates)
        reasons.setdefault(key, [])
        if anchor.reason and anchor.reason not in reasons[key]:
            reasons[key].append(anchor.reason)
    result: List[ExpandedAnchor] = []
    for (source, target), value in support.items():
        result.append(
            ExpandedAnchor(
                source_color=source,
                target_color=target,
                support=int(value),
                predicates=tuple(predicates[(source, target)].keys()),
                reason="+".join(reasons.get((source, target), ())),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda anchor: (
                -anchor.support,
                anchor.source_color,
                anchor.target_color,
                anchor.reason,
            ),
        )
    )


def _component_similarity(source: _Component, target: _Component) -> float:
    size_ratio = min(source.size, target.size) / max(source.size, target.size)
    shape_bonus = 1.0 if source.shape_signature == target.shape_signature else 0.0
    extent_bonus = 1.0 if (source.height, source.width) == (target.height, target.width) else 0.0
    return (size_ratio + shape_bonus + extent_bonus) / 3.0


def _centroid_distance(source: _Component, target: _Component) -> float:
    sy, sx = source.centroid
    ty, tx = target.centroid
    return math.hypot(ty - sy, tx - sx)


def _both_present(grid: np.ndarray, source_color: int, target_color: int) -> bool:
    return bool(np.any(grid == int(source_color)) and np.any(grid == int(target_color)))


def _same_shape_exists(
    grid: np.ndarray,
    source_color: int,
    target_color: int,
) -> bool:
    sources = _components(grid, color=source_color)
    targets = _components(grid, color=target_color)
    for source in sources:
        for target in targets:
            if (
                source.size == target.size
                and source.height == target.height
                and source.width == target.width
            ):
                return True
    return False


def _aligned_exists(grid: np.ndarray, source_color: int, target_color: int) -> bool:
    sources = _components(grid, color=source_color)
    targets = _components(grid, color=target_color)
    for source in sources:
        sy, sx = source.centroid
        for target in targets:
            ty, tx = target.centroid
            if abs(sy - ty) <= 1.0 or abs(sx - tx) <= 1.0:
                return True
    return False


def _adjacent_exists(grid: np.ndarray, source_color: int, target_color: int) -> bool:
    sources = _components(grid, color=source_color)
    targets = _components(grid, color=target_color)
    for source in sources:
        for target in targets:
            if _bbox_distance(source.bbox, target.bbox) <= 1:
                return True
    return False


def _bbox_distance(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> int:
    f_y1, f_x1, f_y2, f_x2 = first
    s_y1, s_x1, s_y2, s_x2 = second
    dy = max(0, max(s_y1 - f_y2 - 1, f_y1 - s_y2 - 1))
    dx = max(0, max(s_x1 - f_x2 - 1, f_x1 - s_x2 - 1))
    return max(dy, dx)


def _components(
    grid: np.ndarray,
    *,
    background: int | None = None,
    color: int | None = None,
) -> List[_Component]:
    if color is None:
        if background is None:
            background = _infer_background(grid)
        mask = grid != int(background)
    else:
        mask = grid == int(color)
    height, width = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[_Component] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            component_color = int(grid[y, x])
            stack = [(y, x)]
            seen[y, x] = True
            points: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and mask[ny, nx]
                        and int(grid[ny, nx]) == component_color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            components.append(
                _Component(
                    color=component_color,
                    points=tuple(sorted(points)),
                    bbox=(min(ys), min(xs), max(ys), max(xs)),
                )
            )
    return components


def _color_contacts(grid: np.ndarray, *, background: int) -> set[Tuple[int, int]]:
    contacts: set[Tuple[int, int]] = set()
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            current = int(grid[y, x])
            if current == int(background):
                continue
            for dy, dx in ((1, 0), (0, 1)):
                ny, nx = y + dy, x + dx
                if ny >= height or nx >= width:
                    continue
                other = int(grid[ny, nx])
                if other == int(background) or other == current:
                    continue
                contacts.add(tuple(sorted((current, other))))
    return contacts


def _infer_background(*grids: np.ndarray) -> int:
    values, counts = np.unique(
        np.concatenate([np.asarray(grid).ravel() for grid in grids]),
        return_counts=True,
    )
    return int(values[int(np.argmax(counts))])


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
