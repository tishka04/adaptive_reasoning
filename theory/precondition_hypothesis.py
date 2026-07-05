"""Falsifiable preconditions for confirmed theory rules.

A correspondence rule says what an action can do. A precondition hypothesis
asks when that rule is applicable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .correspondence_hypothesis import normalize_pair_colors
from .epistemic_metrics import HypothesisRecord, HypothesisStatus

MIN_PRECONDITION_SUPPORT = 3
PRECONDITION_CONFIRM_CONFIDENCE = 0.55


def precondition_key(target_rule: str, predicate: str) -> str:
    return f"precondition::{target_rule}::{normalize_precondition_predicate(predicate)}"


def normalize_precondition_predicate(predicate: str) -> str:
    return str(predicate or "").strip().lower().replace("-", "_").replace(" ", "_")


def target_action_from_rule_key(target_rule: str) -> str:
    parts = str(target_rule or "").split("::")
    if len(parts) >= 2:
        return str(parts[1]).upper()
    return ""


@dataclass
class PreconditionObservation:
    """One opportunity to test a rule precondition."""

    target_rule: str
    action: str
    predicates_present: Set[str] = field(default_factory=set)
    succeeded: bool = False
    target_action: str = ""

    def __post_init__(self) -> None:
        self.action = str(self.action or "").upper()
        self.predicates_present = {
            normalize_precondition_predicate(predicate)
            for predicate in self.predicates_present
        }
        if not self.target_action:
            self.target_action = target_action_from_rule_key(self.target_rule)
        self.target_action = str(self.target_action or "").upper()

    @property
    def target_action_executed(self) -> bool:
        return self.action == self.target_action


@dataclass
class PreconditionHypothesis:
    """A claim that a predicate marks when a target rule is ready to apply."""

    target_rule: str
    predicate: str = "ready_to_validate_correspondence"
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    experiments_spent: int = 0
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    def __post_init__(self) -> None:
        self.predicate = normalize_precondition_predicate(self.predicate)
        self._recompute_status()

    @property
    def key(self) -> str:
        return precondition_key(self.target_rule, self.predicate)

    @property
    def support(self) -> int:
        return len(self.evidence_for)

    @property
    def contradictions(self) -> int:
        return len(self.evidence_against)

    @property
    def confidence(self) -> float:
        total = self.support + self.contradictions
        if total == 0:
            return 0.0
        return self.support / total

    def is_applicable(self, predicates_present: Iterable[str]) -> bool:
        if self.status != HypothesisStatus.CONFIRMED:
            return False
        present = {
            normalize_precondition_predicate(predicate)
            for predicate in predicates_present
        }
        return self.predicate in present

    def observe(
        self,
        observation: PreconditionObservation,
        *,
        was_experiment: bool = False,
    ) -> None:
        if observation.target_rule != self.target_rule:
            return
        if self.predicate not in observation.predicates_present:
            return
        if not observation.target_action_executed:
            return
        if was_experiment:
            self.experiments_spent += 1
        label = f"observed:{observation.action}"
        if observation.succeeded:
            self.evidence_for.append(label)
        else:
            self.evidence_against.append(label)
        self._recompute_status()

    def _recompute_status(self) -> None:
        if self.status == HypothesisStatus.CONFIRMED:
            if (
                self.contradictions >= self.support + MIN_PRECONDITION_SUPPORT
                and self.confidence < 0.45
            ):
                self.status = HypothesisStatus.REFUTED
            return
        if (
            self.support >= MIN_PRECONDITION_SUPPORT
            and self.confidence >= PRECONDITION_CONFIRM_CONFIDENCE
        ):
            self.status = HypothesisStatus.CONFIRMED
        elif (
            self.contradictions >= MIN_PRECONDITION_SUPPORT
            and self.contradictions >= 2 * max(1, self.support)
        ):
            self.status = HypothesisStatus.REFUTED
        else:
            self.status = HypothesisStatus.UNRESOLVED

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=(
                f"{self.predicate} gates applicability of {self.target_rule}"
            ),
            status=self.status,
            support=self.support,
            contradictions=self.contradictions,
            experiments_spent=self.experiments_spent,
        )


def extract_precondition_predicates(
    grid: Any,
    *,
    target_rule: str,
    pair_colors: Tuple[int, int],
    previous_action: str = "",
    recent_actions: Iterable[str] = (),
    recent_correspondence_successes: Iterable[bool] = (),
) -> Set[str]:
    """Extract named candidate predicates for a target correspondence rule."""
    from task_program_guided_level7 import match_score

    pair = normalize_pair_colors(pair_colors)
    score = match_score(grid, pair_colors=pair)
    predicates: Set[str] = set()
    first_count = int(score.component_counts.get(str(pair[0]), 0))
    second_count = int(score.component_counts.get(str(pair[1]), 0))
    previous = str(previous_action or "").upper()
    recent = [str(action or "").upper() for action in recent_actions]
    target_action = target_action_from_rule_key(target_rule)

    if first_count > 0 and second_count > 0:
        predicates.add("selected_pair_exists")
        predicates.add(f"active_color_pair_{pair[0]}_{pair[1]}")
        if pair == (10, 11):
            predicates.add("active_pair_is_colors10_11")
    if score.matched_pairs > 0 or _best_pair_score(score) >= 4.15:
        predicates.add("source_target_aligned")
    predicates.update(
        _projected_source_target_predicates(
            grid,
            pair_colors=pair,
            target_action=target_action,
        )
    )
    if int(score.cursor_near_target) > 0:
        predicates.add("controller_on_source")
        predicates.add("controller_points_to_source")
    if previous == "ACTION5":
        predicates.add("last_action_was_control_switch")
    if "ACTION5" in recent:
        predicates.add("recent_control_switch")
    if previous == target_action:
        predicates.add("last_action_was_validator")
    if any(bool(value) for value in recent_correspondence_successes):
        predicates.add("correspondence_count_improved_recently")

    if _ready_to_validate(predicates, target_action=target_action):
        predicates.add("weak_ready_to_validate_correspondence")
        predicates.add("ready_to_validate_correspondence")
    return {normalize_precondition_predicate(predicate) for predicate in predicates}


def _ready_to_validate(predicates: Set[str], *, target_action: str) -> bool:
    if "source_target_relation_satisfied" in predicates and bool(target_action):
        return True
    return (
        "selected_pair_exists" in predicates
        and "controller_on_source" in predicates
        and "recent_control_switch" in predicates
        and bool(target_action)
    )


def _best_pair_score(score: Any) -> float:
    values = [
        float(item.get("score", 0.0))
        for item in getattr(score, "best_pairs", []) or []
    ]
    if not values:
        return 0.0
    return float(np.max(values))


@dataclass(frozen=True)
class _GridComponent:
    color: int
    points: Tuple[Tuple[int, int], ...]
    bbox: Tuple[int, int, int, int]

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def centroid_y(self) -> float:
        return sum(y for y, _ in self.points) / max(1, self.size)

    @property
    def centroid_x(self) -> float:
        return sum(x for _, x in self.points) / max(1, self.size)


def _projected_source_target_predicates(
    grid: Any,
    *,
    pair_colors: Tuple[int, int],
    target_action: str,
) -> Set[str]:
    """ar25 object-level readiness predicates from the rendered observation.

    The relevant relation is geometric: after the candidate validation action,
    the mirror-projected source shape should cover the target shape. This is a
    predicate over objects in the frame, not an action score.
    """
    if pair_colors != (10, 11):
        return set()
    arr = np.asarray(grid, dtype=np.int32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return set()

    mirror = _largest_component(_components(arr, pair_colors[0]))
    source = _selected_source_component(arr)
    target_points = _target_points(arr, pair_colors[1])
    if mirror is None or source is None or not target_points:
        return set()

    target_xs = {x for _, x in target_points}
    axis_x = mirror.centroid_x
    dy, dx = _action_delta_pixels(target_action, mirror)
    projected_now = _project_points(source.points, axis_x=axis_x, dy=0, dx=0, shape=arr.shape)
    projected_after = _project_points(
        source.points,
        axis_x=axis_x,
        dy=dy,
        dx=dx,
        shape=arr.shape,
    )
    projected_xs = {x for _, x in projected_now}
    x_coverage = len(projected_xs & target_xs) / max(1, len(target_xs))
    target_coverage = len(projected_after & target_points) / max(1, len(target_points))

    predicates: Set[str] = {
        "selected_source_matches_target_color",
    }
    if x_coverage >= 0.85:
        predicates.add("selected_source_matches_target_shape")
        predicates.add("source_target_projected_aligned")
    if target_coverage >= 0.75:
        predicates.add("source_target_relation_satisfied")
        predicates.add("source_target_aligned")
    return predicates


def _components(grid: np.ndarray, color: int) -> List[_GridComponent]:
    mask = grid == int(color)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[_GridComponent] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
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
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            components.append(
                _GridComponent(
                    color=int(color),
                    points=tuple(points),
                    bbox=(min(ys), min(xs), max(ys), max(xs)),
                )
            )
    return components


def _largest_component(
    components: Sequence[_GridComponent],
) -> _GridComponent | None:
    if not components:
        return None
    return max(components, key=lambda component: component.size)


def _selected_source_component(grid: np.ndarray) -> _GridComponent | None:
    height, width = grid.shape
    candidates = [
        component for component in _components(grid, 5)
        if component.size >= 6
        and component.bbox[2] < height - 1
        and component.bbox[3] < width - 1
    ]
    return _largest_component(candidates)


def _target_points(grid: np.ndarray, target_color: int) -> Set[Tuple[int, int]]:
    _, width = grid.shape
    points: Set[Tuple[int, int]] = set()
    for component in _components(grid, target_color):
        if component.bbox[3] >= width - 1:
            continue
        points.update(component.points)
    return points


def _action_delta_pixels(
    action: str,
    mirror: _GridComponent,
) -> Tuple[int, int]:
    cell = max(1, mirror.bbox[3] - mirror.bbox[1] + 1)
    action_name = str(action or "").upper()
    if action_name == "ACTION1":
        return -cell, 0
    if action_name == "ACTION2":
        return cell, 0
    if action_name == "ACTION3":
        return 0, -cell
    if action_name == "ACTION4":
        return 0, cell
    return 0, 0


def _project_points(
    points: Iterable[Tuple[int, int]],
    *,
    axis_x: float,
    dy: int,
    dx: int,
    shape: Tuple[int, int],
) -> Set[Tuple[int, int]]:
    height, width = shape
    projected: Set[Tuple[int, int]] = set()
    for y, x in points:
        source_y = int(y + dy)
        source_x = int(x + dx)
        reflected_x = int(round((2 * axis_x) - source_x))
        if 0 <= source_y < height and 0 <= reflected_x < width:
            projected.add((source_y, reflected_x))
    return projected
