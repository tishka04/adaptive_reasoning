"""Online induction of terminally validated local stencil relations.

The learner deliberately knows nothing about a game id, a level, or a fixed
coordinate.  It receives the parameterized click actions exposed by the live
environment and looks for missing cells in their regular lattice.  A missing
cell surrounded by clickable cells is treated as a candidate visual stencil.

Ordinary transitions may teach only the local effect of a click (for example,
that it toggles equality with a neighbouring stencil centre).  A visual
marker-to-goal relation is learned only from a terminal transition that the
outer terminal-attribution mechanism has independently confirmed.  The
learned relation can then be applied to a new arrangement during the same
game.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class RelationalStencilSelection:
    """One exact click selected from a learned terminal relation."""

    action_name: str
    action_data: Dict[str, Any]
    violations_before: int
    expected_violations_after: int
    supporting_constraints: int
    stencil_count: int
    reason: str


@dataclass(frozen=True)
class StencilTheoryAssessment:
    """Auditable fit of one relational rule family to one live layout."""

    structural_signature: str
    applicable: bool
    total_constraints: int
    total_violations: int
    improving_actions: int
    stencil_count: int
    click_count: int


@dataclass(frozen=True)
class _ClickCandidate:
    action_name: str
    action_data: Dict[str, Any]
    coordinate: Coordinate


@dataclass(frozen=True)
class _Stencil:
    coordinate: Coordinate
    center_color: int


@dataclass(frozen=True)
class _Layout:
    grid: np.ndarray
    background: int
    spacing: int
    tile_size: int
    clicks: Dict[Coordinate, _ClickCandidate]
    colors: Dict[Coordinate, int]
    stencils: Tuple[_Stencil, ...]


class OnlineTerminalRelationalStencilLearner:
    """Learn a local visual goal exclusively from confirmed terminal evidence."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_terminal_support: int = 2,
        permute_confirmed_relation: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.minimum_terminal_support = max(
            1,
            int(minimum_terminal_support),
        )
        self.permute_confirmed_relation = bool(
            permute_confirmed_relation
        )
        self._click_relation_effects: Counter[Tuple[bool, bool]] = Counter()
        self._marker_goal_votes: Counter[Tuple[str, bool]] = Counter()
        self._selection_counts: Counter[Coordinate] = Counter()
        self._terminal_examples = 0
        self._terminal_constraints = 0
        self._decisions = 0
        self._effect_observations = 0
        self._layouts_detected = 0

    def observe_transition(
        self,
        *,
        grid_before: Any,
        grid_after: Any,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        available_action_candidates: Sequence[Any] | None,
        terminal_success_confirmed: bool,
    ) -> None:
        """Observe mechanics, and learn goals only from confirmed success."""
        if not self.enabled or str(action_name) != "ACTION6":
            return
        candidates = tuple(available_action_candidates or ())
        before = _discover_layout(grid_before, candidates)
        if before is None:
            return
        self._layouts_detected += 1
        clicked = _coordinate(action_data)
        if clicked is None or clicked not in before.clicks:
            return

        after = _discover_layout(grid_after, candidates)
        if after is not None and _same_stencil_geometry(before, after):
            observed = self._observe_click_relation_effect(
                before,
                after,
                clicked,
            )
            self._effect_observations += observed

        if terminal_success_confirmed:
            self._learn_confirmed_terminal_example(before, clicked)

    def select(
        self,
        *,
        current_grid: Any,
        available_action_candidates: Sequence[Any] | None,
    ) -> RelationalStencilSelection | None:
        """Select a click that reduces violations of the learned relation."""
        if not self.enabled or self._terminal_examples <= 0:
            return None
        layout = _discover_layout(
            current_grid,
            tuple(available_action_candidates or ()),
        )
        if layout is None:
            return None
        rules = self.selection_rules()
        if not rules:
            return None
        return self._select_from_layout(
            layout,
            rules=rules,
            reason_prefix="confirmed terminal stencil relation",
        )

    def select_with_rules(
        self,
        *,
        current_grid: Any,
        available_action_candidates: Sequence[Any] | None,
        rules: Mapping[str, bool],
        reason_prefix: str,
    ) -> RelationalStencilSelection | None:
        """Select under an explicit, still-testable revision hypothesis."""
        if not self.enabled or not rules:
            return None
        layout = _discover_layout(
            current_grid,
            tuple(available_action_candidates or ()),
        )
        if layout is None:
            return None
        return self._select_from_layout(
            layout,
            rules=rules,
            reason_prefix=str(reason_prefix),
        )

    def assess(
        self,
        *,
        current_grid: Any,
        available_action_candidates: Sequence[Any] | None,
        rules: Mapping[str, bool] | None = None,
    ) -> StencilTheoryAssessment:
        """Measure current rule fit without changing any learned state."""
        layout = _discover_layout(
            current_grid,
            tuple(available_action_candidates or ()),
        )
        if layout is None:
            return StencilTheoryAssessment(
                structural_signature="",
                applicable=False,
                total_constraints=0,
                total_violations=0,
                improving_actions=0,
                stencil_count=0,
                click_count=0,
            )
        active_rules = dict(
            self.selection_rules() if rules is None else rules
        )
        if not active_rules:
            return StencilTheoryAssessment(
                structural_signature=_layout_structural_signature(layout),
                applicable=False,
                total_constraints=0,
                total_violations=0,
                improving_actions=0,
                stencil_count=len(layout.stencils),
                click_count=len(layout.clicks),
            )
        total_constraints = 0
        total_violations = 0
        improving_actions = 0
        for coordinate in layout.clicks:
            constraints = self._constraints_for_click(
                layout,
                coordinate,
                active_rules,
            )
            total_constraints += len(constraints)
            violations = sum(
                actual != desired
                for actual, desired in constraints
            )
            total_violations += violations
            expected_after = sum(
                self._predicted_relation_after_click(actual) != desired
                for actual, desired in constraints
            )
            if violations > expected_after:
                improving_actions += 1
        return StencilTheoryAssessment(
            structural_signature=_layout_structural_signature(layout),
            applicable=bool(total_constraints),
            total_constraints=total_constraints,
            total_violations=total_violations,
            improving_actions=improving_actions,
            stencil_count=len(layout.stencils),
            click_count=len(layout.clicks),
        )

    def confirmed_rules(self) -> Dict[str, bool]:
        """Return terminally grounded semantics without policy ablations."""
        return dict(self._confirmed_rules())

    def selection_rules(self) -> Dict[str, bool]:
        """Return the live policy semantics, including permutation control."""
        rules = self.confirmed_rules()
        if self.permute_confirmed_relation:
            return {
                marker: not relation
                for marker, relation in rules.items()
            }
        return rules

    def _select_from_layout(
        self,
        layout: _Layout,
        *,
        rules: Mapping[str, bool],
        reason_prefix: str,
    ) -> RelationalStencilSelection | None:
        """Select the best one-step constraint reduction in a layout."""

        best: tuple[
            tuple[int, int, int, int, int],
            _ClickCandidate,
            int,
            int,
            int,
        ] | None = None
        for coordinate, candidate in layout.clicks.items():
            constraints = self._constraints_for_click(
                layout,
                coordinate,
                rules,
            )
            if not constraints:
                continue
            violations_before = sum(
                actual != desired
                for actual, desired in constraints
            )
            if violations_before <= 0:
                continue
            expected_after = sum(
                self._predicted_relation_after_click(actual) != desired
                for actual, desired in constraints
            )
            reduction = violations_before - expected_after
            if reduction <= 0:
                continue
            key = (
                reduction,
                violations_before,
                len(constraints),
                -self._selection_counts[coordinate],
                -coordinate[1] * 10_000 - coordinate[0],
            )
            if best is None or key > best[0]:
                best = (
                    key,
                    candidate,
                    violations_before,
                    expected_after,
                    len(constraints),
                )
        if best is None:
            return None

        _, candidate, before_count, after_count, support = best
        self._selection_counts[candidate.coordinate] += 1
        self._decisions += 1
        return RelationalStencilSelection(
            action_name=candidate.action_name,
            action_data=dict(candidate.action_data),
            violations_before=before_count,
            expected_violations_after=after_count,
            supporting_constraints=support,
            stencil_count=len(layout.stencils),
            reason=(
                f"{reason_prefix} predicts a reduction "
                f"from {before_count} to {after_count} local violations"
            ),
        )

    def summary(self) -> Dict[str, Any]:
        rules = self._confirmed_rules()
        return {
            "enabled": self.enabled,
            "permute_confirmed_relation": (
                self.permute_confirmed_relation
            ),
            "terminal_examples": self._terminal_examples,
            "terminal_constraints": self._terminal_constraints,
            "effect_observations": self._effect_observations,
            "layouts_detected": self._layouts_detected,
            "decisions": self._decisions,
            "confirmed_marker_rules": {
                marker: (
                    "equal_to_center"
                    if desired_equal
                    else "different_from_center"
                )
                for marker, desired_equal in sorted(rules.items())
            },
            "selection_marker_rules": {
                marker: (
                    "equal_to_center"
                    if desired_equal
                    else "different_from_center"
                )
                for marker, desired_equal in sorted(
                    self.selection_rules().items()
                )
            },
            "marker_goal_votes": {
                f"{marker}:{'equal' if relation else 'different'}": count
                for (marker, relation), count in sorted(
                    self._marker_goal_votes.items()
                )
            },
            "click_relation_effects": {
                f"{'equal' if before else 'different'}->"
                f"{'equal' if after else 'different'}": count
                for (before, after), count in sorted(
                    self._click_relation_effects.items()
                )
            },
        }

    def _observe_click_relation_effect(
        self,
        before: _Layout,
        after: _Layout,
        clicked: Coordinate,
    ) -> int:
        after_stencils = {
            stencil.coordinate: stencil
            for stencil in after.stencils
        }
        observations = 0
        for stencil in before.stencils:
            if not _is_neighbor(
                clicked,
                stencil.coordinate,
                before.spacing,
            ):
                continue
            after_stencil = after_stencils.get(stencil.coordinate)
            if after_stencil is None or clicked not in after.colors:
                continue
            relation_before = (
                before.colors[clicked] == stencil.center_color
            )
            relation_after = (
                after.colors[clicked] == after_stencil.center_color
            )
            self._click_relation_effects[
                (relation_before, relation_after)
            ] += 1
            observations += 1
        return observations

    def _learn_confirmed_terminal_example(
        self,
        layout: _Layout,
        clicked: Coordinate,
    ) -> None:
        learned = 0
        for stencil in layout.stencils:
            sx, sy = stencil.coordinate
            for coordinate, color in layout.colors.items():
                if not _is_neighbor(
                    coordinate,
                    stencil.coordinate,
                    layout.spacing,
                ):
                    continue
                dx = (coordinate[0] - sx) // layout.spacing
                dy = (coordinate[1] - sy) // layout.spacing
                marker = _marker_role(layout, stencil, dx, dy)
                relation = color == stencil.center_color
                if coordinate == clicked:
                    relation = self._predicted_relation_after_click(relation)
                self._marker_goal_votes[(marker, relation)] += 1
                learned += 1
        if learned:
            self._terminal_examples += 1
            self._terminal_constraints += learned

    def _confirmed_rules(self) -> Dict[str, bool]:
        rules: Dict[str, bool] = {}
        for marker in ("void", "filled"):
            equal = self._marker_goal_votes[(marker, True)]
            different = self._marker_goal_votes[(marker, False)]
            support = max(equal, different)
            if support < self.minimum_terminal_support:
                continue
            if equal == different:
                continue
            rules[marker] = equal > different
        return rules

    def _constraints_for_click(
        self,
        layout: _Layout,
        coordinate: Coordinate,
        rules: Mapping[str, bool],
    ) -> Tuple[Tuple[bool, bool], ...]:
        constraints = []
        for stencil in layout.stencils:
            if not _is_neighbor(
                coordinate,
                stencil.coordinate,
                layout.spacing,
            ):
                continue
            dx = (
                coordinate[0] - stencil.coordinate[0]
            ) // layout.spacing
            dy = (
                coordinate[1] - stencil.coordinate[1]
            ) // layout.spacing
            marker = _marker_role(layout, stencil, dx, dy)
            if marker not in rules:
                continue
            actual = layout.colors[coordinate] == stencil.center_color
            constraints.append((actual, bool(rules[marker])))
        return tuple(constraints)

    def _predicted_relation_after_click(self, relation: bool) -> bool:
        outcomes = {
            after: self._click_relation_effects[(relation, after)]
            for after in (False, True)
        }
        if max(outcomes.values(), default=0) <= 0:
            return not relation
        return max(
            outcomes,
            key=lambda after: (outcomes[after], after != relation),
        )


def _discover_layout(
    grid: Any,
    candidates: Sequence[Any],
) -> _Layout | None:
    array = np.asarray(grid)
    if array.ndim != 2 or array.size <= 0:
        return None
    clicks = _click_candidates(candidates)
    if len(clicks) < 3:
        return None
    spacing = _lattice_spacing(tuple(clicks))
    if spacing < 3:
        return None
    tile_size = max(3, spacing - 2)
    values, counts = np.unique(array, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    colors = {
        coordinate: _sample_center_color(
            array,
            coordinate,
            tile_size,
        )
        for coordinate in clicks
    }
    stencils = []
    xs = range(
        min(x for x, _ in clicks),
        max(x for x, _ in clicks) + 1,
        spacing,
    )
    ys = range(
        min(y for _, y in clicks),
        max(y for _, y in clicks) + 1,
        spacing,
    )
    for y in ys:
        for x in xs:
            coordinate = (x, y)
            if coordinate in clicks:
                continue
            neighbors = sum(
                (x + dx * spacing, y + dy * spacing) in clicks
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if dx or dy
            )
            if neighbors < 3:
                continue
            center = _sample_center_color(array, coordinate, tile_size)
            if center == background:
                continue
            stencils.append(_Stencil(coordinate, center))
    if not stencils:
        return None
    return _Layout(
        grid=array,
        background=background,
        spacing=spacing,
        tile_size=tile_size,
        clicks=clicks,
        colors=colors,
        stencils=tuple(stencils),
    )


def _click_candidates(
    candidates: Sequence[Any],
) -> Dict[Coordinate, _ClickCandidate]:
    result: Dict[Coordinate, _ClickCandidate] = {}
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            name = str(
                candidate.get("action_name", candidate.get("name", ""))
            )
            data = dict(
                candidate.get(
                    "action_data",
                    candidate.get("action_args", {}),
                )
                or {}
            )
        else:
            name = str(
                getattr(
                    candidate,
                    "action_name",
                    getattr(candidate, "name", ""),
                )
            )
            data = dict(
                getattr(
                    candidate,
                    "action_data",
                    getattr(candidate, "action_args", {}),
                )
                or {}
            )
        if name != "ACTION6":
            continue
        coordinate = _coordinate(data)
        if coordinate is None:
            continue
        result[coordinate] = _ClickCandidate(name, data, coordinate)
    return result


def _coordinate(data: Mapping[str, Any] | None) -> Coordinate | None:
    if not data or "x" not in data or "y" not in data:
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (TypeError, ValueError):
        return None


def _lattice_spacing(coordinates: Sequence[Coordinate]) -> int:
    differences: Counter[int] = Counter()
    for index, (x1, y1) in enumerate(coordinates):
        for x2, y2 in coordinates[index + 1 :]:
            if y1 == y2 and x1 != x2:
                differences[abs(x2 - x1)] += 1
            if x1 == x2 and y1 != y2:
                differences[abs(y2 - y1)] += 1
    if not differences:
        return 0
    smallest = min(differences)
    return int(smallest)


def _sample_center_color(
    grid: np.ndarray,
    coordinate: Coordinate,
    tile_size: int,
) -> int:
    x, y = coordinate
    offset = max(0, tile_size // 2)
    sample_y = min(max(0, y + offset), grid.shape[0] - 1)
    sample_x = min(max(0, x + offset), grid.shape[1] - 1)
    return int(grid[sample_y, sample_x])


def _marker_role(
    layout: _Layout,
    stencil: _Stencil,
    dx: int,
    dy: int,
) -> str:
    column = dx + 1
    row = dy + 1
    sample_x = (
        stencil.coordinate[0]
        + ((2 * column + 1) * layout.tile_size) // 6
    )
    sample_y = (
        stencil.coordinate[1]
        + ((2 * row + 1) * layout.tile_size) // 6
    )
    sample_x = min(max(0, sample_x), layout.grid.shape[1] - 1)
    sample_y = min(max(0, sample_y), layout.grid.shape[0] - 1)
    color = int(layout.grid[sample_y, sample_x])
    # ARC object sprites use zero-valued cells as transparent/void even when
    # the rendered board background has another palette index.  The rule is
    # therefore expressed as a structural occupancy role, not as the board's
    # modal colour or a game-specific marker colour.
    return "void" if color == 0 else "filled"


def _is_neighbor(
    coordinate: Coordinate,
    stencil: Coordinate,
    spacing: int,
) -> bool:
    dx = coordinate[0] - stencil[0]
    dy = coordinate[1] - stencil[1]
    return (
        dx in {-spacing, 0, spacing}
        and dy in {-spacing, 0, spacing}
        and (dx != 0 or dy != 0)
    )


def _same_stencil_geometry(before: _Layout, after: _Layout) -> bool:
    return (
        before.spacing == after.spacing
        and set(before.clicks) == set(after.clicks)
        and {
            stencil.coordinate for stencil in before.stencils
        }
        == {
            stencil.coordinate for stencil in after.stencils
        }
    )


def _layout_structural_signature(layout: _Layout) -> str:
    """Return a palette- and translation-independent layout fingerprint."""
    min_x = min(x for x, _ in layout.clicks)
    min_y = min(y for _, y in layout.clicks)
    spacing = max(1, int(layout.spacing))
    normalized_clicks = tuple(sorted(
        (
            (x - min_x) // spacing,
            (y - min_y) // spacing,
        )
        for x, y in layout.clicks
    ))
    normalized_stencils = []
    for stencil in layout.stencils:
        sx, sy = stencil.coordinate
        roles = tuple(
            _marker_role(layout, stencil, dx, dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dx or dy
        )
        normalized_stencils.append((
            (sx - min_x) // spacing,
            (sy - min_y) // spacing,
            roles,
        ))
    payload = (
        normalized_clicks,
        tuple(sorted(normalized_stencils)),
    )
    return hashlib.sha1(
        repr(payload).encode("utf-8")
    ).hexdigest()[:16]


__all__ = [
    "OnlineTerminalRelationalStencilLearner",
    "RelationalStencilSelection",
    "StencilTheoryAssessment",
]
