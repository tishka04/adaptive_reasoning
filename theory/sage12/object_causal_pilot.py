"""SAGE12 V4.5 rooted intervention-event feasibility pilot.

The pilot compiles object correspondences from a common pre-state and two
executed post-states.  Only source V4.3 pairs are opened during feasibility.
Fresh source and validation collection remain mechanically closed unless all
pre-registered gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION
from v3.schemas import ObjectInfo

from .action_target_data import ActionTargetTrace, build_observation, grid_sha256
from .bound_mechanic_pilot import (
    BindingPairRecord,
    BoundEvent,
    load_pairs,
)
from .mechanic_induction import _identity_probe as _categorical_identity_probe
from .pairwise_causal_pilot import (
    AntisymmetricLinearModel,
    _fit_model,
    _fit_temperature,
)

FORMAT_VERSION = "sage12-object-causal-delta-v4.5"
MANIFEST_FORMAT_VERSION = "sage12-object-causal-pilot-v4.5"
VOCABULARY_FORMAT_VERSION = "sage12-discovered-event-vocabulary-v4.5"
FEASIBILITY_FORMAT_VERSION = "sage12-object-causal-feasibility-v4.5"
MODEL_FORMAT_VERSION = "sage12-rooted-linear-model-v4.5"
COLLECTION_FORMAT_VERSION = "sage12-object-causal-tree-v4.5"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "object_causal_pilot_v4_5"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"
V43_OUTPUT_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"

MODEL_MODES = (
    "structured",
    "history_no_root",
    "action_only",
    "root_no_history",
    "template",
)
BASELINE_MODES = (
    "history_no_root",
    "action_only",
    "root_no_history",
    "template",
)
SOURCE_SEEDS = (1663, 1721, 1783, 1847)
VALIDATION_SEEDS = (1901, 1951, 2011, 2063)

_DIRECTION_SHUFFLE = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "north_east": "south_west",
    "south_west": "north_east",
    "north_west": "south_east",
    "south_east": "north_west",
    "aligned_row": "aligned_col",
    "aligned_col": "aligned_row",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_cells(cells: Sequence[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    if not cells:
        return frozenset()
    r0 = min(row for row, _col in cells)
    c0 = min(col for _row, col in cells)
    return frozenset((row - r0, col - c0) for row, col in cells)


def _iou(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> float:
    a = set(left)
    b = set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _area_ratio(left: int, right: int) -> float:
    return min(left, right) / max(left, right, 1)


def _bbox_gap(left: ObjectInfo, right: ObjectInfo) -> int:
    lr0, lc0, lr1, lc1 = left.bbox
    rr0, rc0, rr1, rc1 = right.bbox
    row_gap = max(0, rr0 - lr1 - 1, lr0 - rr1 - 1)
    col_gap = max(0, rc0 - lc1 - 1, lc0 - rc1 - 1)
    return max(row_gap, col_gap)


def _direction(
    before: tuple[float, float], after: tuple[float, float]
) -> str:
    dr = float(after[0] - before[0])
    dc = float(after[1] - before[1])
    vertical = "north" if dr < -0.5 else "south" if dr > 0.5 else ""
    horizontal = "west" if dc < -0.5 else "east" if dc > 0.5 else ""
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    return vertical or horizontal or "none"


def _magnitude(
    before: tuple[float, float], after: tuple[float, float]
) -> str:
    distance = math.dist(before, after)
    if distance <= 1.5:
        return "one"
    if distance <= 4.0:
        return "short"
    return "long"


def _object_score(
    before: ObjectInfo,
    after: ObjectInfo,
    *,
    grid_diagonal: float,
) -> float:
    translated_iou = _iou(
        _normalized_cells(before.cells), _normalized_cells(after.cells)
    )
    absolute_iou = _iou(before.cells, after.cells)
    ratio = _area_ratio(before.area, after.area)
    distance_score = max(
        0.0, 1.0 - math.dist(before.center, after.center) / max(grid_diagonal, 1.0)
    )
    value_score = 1.0 if int(before.value) == int(after.value) else 0.5
    return (
        0.35 * translated_iou
        + 0.20 * absolute_iou
        + 0.15 * ratio
        + 0.15 * distance_score
        + 0.15 * value_score
    )


@dataclass(frozen=True)
class ObjectCorrespondence:
    """Deterministic, auditable correspondence for one before/after arm."""

    matched: tuple[tuple[int, int, float], ...]
    appeared: tuple[int, ...]
    disappeared: tuple[int, ...]
    splits: tuple[tuple[int, tuple[int, ...], float], ...] = ()
    merges: tuple[tuple[tuple[int, ...], int, float], ...] = ()
    ambiguous_before: tuple[int, ...] = ()
    ambiguous_after: tuple[int, ...] = ()

    @property
    def ambiguity_count(self) -> int:
        return len(self.ambiguous_before) + len(self.ambiguous_after)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _group_union_iou(
    one: ObjectInfo, many: Sequence[ObjectInfo]
) -> tuple[float, float]:
    combined = [cell for item in many for cell in item.cells]
    shape_iou = _iou(_normalized_cells(one.cells), _normalized_cells(combined))
    ratio = _area_ratio(one.area, sum(item.area for item in many))
    return shape_iou, ratio


def match_objects(
    before: Sequence[ObjectInfo],
    after: Sequence[ObjectInfo],
    *,
    grid_shape: tuple[int, int] = (64, 64),
    minimum_score: float = 0.65,
    ambiguity_margin: float = 0.10,
    split_merge_iou: float = 0.70,
    split_merge_area_ratio: float = 0.75,
) -> ObjectCorrespondence:
    """Match objects, exposing weak matches instead of turning them into labels."""

    diagonal = math.hypot(*grid_shape)
    candidates: list[tuple[float, int, int]] = []
    by_before: dict[int, list[tuple[float, int]]] = defaultdict(list)
    by_after: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for left in before:
        for right in after:
            score = _object_score(left, right, grid_diagonal=diagonal)
            if score >= minimum_score:
                candidates.append((score, left.object_id, right.object_id))
                by_before[left.object_id].append((score, right.object_id))
                by_after[right.object_id].append((score, left.object_id))

    def near_tie(rows: Sequence[tuple[float, int]]) -> bool:
        ordered = sorted(rows, reverse=True)
        return len(ordered) >= 2 and ordered[0][0] - ordered[1][0] < ambiguity_margin

    ambiguous_before = {
        object_id for object_id, rows in by_before.items() if near_tie(rows)
    }
    ambiguous_after = {
        object_id for object_id, rows in by_after.items() if near_tie(rows)
    }
    matched: list[tuple[int, int, float]] = []
    used_before: set[int] = set()
    used_after: set[int] = set()
    for score, before_id, after_id in sorted(
        candidates, key=lambda row: (-row[0], row[1], row[2])
    ):
        if before_id in ambiguous_before or after_id in ambiguous_after:
            continue
        if before_id in used_before or after_id in used_after:
            continue
        matched.append((before_id, after_id, score))
        used_before.add(before_id)
        used_after.add(after_id)

    before_by_id = {item.object_id: item for item in before}
    after_by_id = {item.object_id: item for item in after}
    free_before = sorted(
        set(before_by_id) - used_before - ambiguous_before
    )
    free_after = sorted(set(after_by_id) - used_after - ambiguous_after)
    splits: list[tuple[int, tuple[int, ...], float]] = []
    for before_id in list(free_before):
        choices = [after_by_id[item] for item in free_after]
        rows: list[tuple[float, tuple[int, ...]]] = []
        for size in (2, 3):
            for group in itertools.combinations(choices, size):
                shape_iou, ratio = _group_union_iou(before_by_id[before_id], group)
                if (
                    shape_iou >= split_merge_iou
                    and ratio >= split_merge_area_ratio
                ):
                    rows.append(
                        (
                            0.70 * shape_iou + 0.30 * ratio,
                            tuple(sorted(item.object_id for item in group)),
                        )
                    )
        if rows:
            score, group_ids = max(rows, key=lambda row: (row[0], row[1]))
            splits.append((before_id, group_ids, score))
            free_before.remove(before_id)
            free_after = [item for item in free_after if item not in group_ids]

    merges: list[tuple[tuple[int, ...], int, float]] = []
    for after_id in list(free_after):
        choices = [before_by_id[item] for item in free_before]
        rows = []
        for size in (2, 3):
            for group in itertools.combinations(choices, size):
                shape_iou, ratio = _group_union_iou(after_by_id[after_id], group)
                if (
                    shape_iou >= split_merge_iou
                    and ratio >= split_merge_area_ratio
                ):
                    rows.append(
                        (
                            0.70 * shape_iou + 0.30 * ratio,
                            tuple(sorted(item.object_id for item in group)),
                        )
                    )
        if rows:
            score, group_ids = max(rows, key=lambda row: (row[0], row[1]))
            merges.append((group_ids, after_id, score))
            free_after.remove(after_id)
            free_before = [item for item in free_before if item not in group_ids]

    return ObjectCorrespondence(
        matched=tuple(sorted(matched)),
        appeared=tuple(free_after),
        disappeared=tuple(free_before),
        splits=tuple(sorted(splits)),
        merges=tuple(sorted(merges)),
        ambiguous_before=tuple(sorted(ambiguous_before)),
        ambiguous_after=tuple(sorted(ambiguous_after)),
    )


@dataclass(frozen=True)
class ObjectEvent:
    operation: str
    locus: str
    direction: str
    magnitude: str
    subject: str
    changed_cells: tuple[tuple[int, int], ...]
    confidence: float

    @property
    def fine_key(self) -> str:
        return f"{self.locus}|{self.operation}|{self.direction}|{self.magnitude}"

    def key(self, projection: str) -> str:
        if projection == "fine":
            return self.fine_key
        if projection == "no_magnitude":
            return f"{self.locus}|{self.operation}|{self.direction}|any"
        if projection == "base":
            return f"{self.locus}|{self.operation}|any|any"
        raise ValueError(f"unknown event projection: {projection}")

    @property
    def equivalence_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.subject,
            self.operation,
            self.locus,
            self.direction,
            self.magnitude,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _changed_cells(left: ObjectInfo | None, right: ObjectInfo | None) -> set[tuple[int, int]]:
    return set(left.cells if left is not None else ()) | set(
        right.cells if right is not None else ()
    )


def _event_locus(
    *,
    anchor_row: int | None,
    anchor_col: int | None,
    target_object_id: int | None,
    subject_before_ids: Sequence[int],
    cells: Iterable[tuple[int, int]],
) -> str:
    if target_object_id is not None and target_object_id in subject_before_ids:
        return "direct"
    if anchor_row is None or anchor_col is None:
        return "collateral"
    points = set(cells)
    if (anchor_row, anchor_col) in points:
        return "direct"
    if any(
        max(abs(row - anchor_row), abs(col - anchor_col)) <= 1
        for row, col in points
    ):
        return "local"
    return "collateral"


def compile_arm_events(trace: ActionTargetTrace) -> tuple[ObjectCorrespondence, tuple[ObjectEvent, ...]]:
    before = build_observation(
        trace.frame_before,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_before,
        levels_completed=trace.levels_completed_before,
    )
    after = build_observation(
        trace.frame_after,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_after,
        levels_completed=trace.levels_completed_after,
    )
    correspondence = match_objects(
        before.objects,
        after.objects,
        grid_shape=tuple(np.asarray(trace.frame_before).shape[:2]),
    )
    before_by_id = {item.object_id: item for item in before.objects}
    after_by_id = {item.object_id: item for item in after.objects}
    anchor = trace.anchor
    events: list[ObjectEvent] = []

    def add(
        operation: str,
        subject: str,
        before_ids: Sequence[int],
        left: ObjectInfo | None,
        right: ObjectInfo | None,
        confidence: float,
        *,
        direction: str = "none",
        magnitude: str = "none",
        cells: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        changed = set(cells) if cells is not None else _changed_cells(left, right)
        events.append(
            ObjectEvent(
                operation=operation,
                locus=_event_locus(
                    anchor_row=anchor.row,
                    anchor_col=anchor.col,
                    target_object_id=anchor.target_object_id,
                    subject_before_ids=before_ids,
                    cells=changed,
                ),
                direction=direction,
                magnitude=magnitude,
                subject=subject,
                changed_cells=tuple(sorted(changed)),
                confidence=float(confidence),
            )
        )

    for before_id, after_id, score in correspondence.matched:
        left = before_by_id[before_id]
        right = after_by_id[after_id]
        subject = f"pre:{before_id}"
        if int(left.value) != int(right.value):
            add("recolored", subject, (before_id,), left, right, score)
        translated_iou = _iou(
            _normalized_cells(left.cells), _normalized_cells(right.cells)
        )
        if translated_iou < 0.90 or _area_ratio(left.area, right.area) < 0.90:
            add("reshaped", subject, (before_id,), left, right, score)
        if math.dist(left.center, right.center) > 0.5:
            add(
                "displaced",
                subject,
                (before_id,),
                left,
                right,
                score,
                direction=_direction(left.center, right.center),
                magnitude=_magnitude(left.center, right.center),
            )
    for after_id in correspondence.appeared:
        obj = after_by_id[after_id]
        add(
            "appeared",
            f"new:{_relative_subject(obj.center, anchor.row, anchor.col)}",
            (),
            None,
            obj,
            1.0,
            magnitude=_size_bucket(obj.area),
        )
    for before_id in correspondence.disappeared:
        obj = before_by_id[before_id]
        add(
            "disappeared",
            f"pre:{before_id}",
            (before_id,),
            obj,
            None,
            1.0,
            magnitude=_size_bucket(obj.area),
        )
    for before_id, after_ids, score in correspondence.splits:
        source = before_by_id[before_id]
        children = [after_by_id[item] for item in after_ids]
        add(
            "split",
            f"pre:{before_id}",
            (before_id,),
            source,
            None,
            score,
            magnitude=_count_bucket(len(children)),
            cells=set(source.cells).union(*(set(item.cells) for item in children)),
        )
    for before_ids, after_id, score in correspondence.merges:
        sources = [before_by_id[item] for item in before_ids]
        target = after_by_id[after_id]
        add(
            "merged",
            "pre:" + ",".join(str(item) for item in before_ids),
            before_ids,
            None,
            target,
            score,
            magnitude=_count_bucket(len(sources)),
            cells=set(target.cells).union(*(set(item.cells) for item in sources)),
        )
    if trace.levels_completed_after > trace.levels_completed_before or (
        trace.game_state_after.upper() == "WIN"
        and trace.game_state_before.upper() != "WIN"
    ):
        add(
            "progressed",
            "terminal",
            (),
            None,
            None,
            1.0,
            cells=(),
        )
    if (
        trace.game_state_after.upper() == "GAME_OVER"
        and trace.game_state_before.upper() != "GAME_OVER"
    ):
        add(
            "terminated",
            "terminal",
            (),
            None,
            None,
            1.0,
            cells=(),
        )
    return correspondence, tuple(sorted(events, key=lambda item: item.equivalence_key))


def _relative_subject(
    center: tuple[float, float], row: int | None, col: int | None
) -> str:
    if row is None or col is None:
        return "unrooted"
    return _direction((float(row), float(col)), center)


def _size_bucket(area: int) -> str:
    if area <= 1:
        return "one"
    if area <= 4:
        return "small"
    if area <= 16:
        return "medium"
    return "large"


def _count_bucket(count: int) -> str:
    return "two" if count == 2 else "three_plus"


def _cancel_common_events(
    left: Sequence[ObjectEvent], right: Sequence[ObjectEvent]
) -> tuple[tuple[ObjectEvent, ...], tuple[ObjectEvent, ...], int]:
    right_by_key: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for index, event in enumerate(right):
        right_by_key[event.equivalence_key].append(index)
    cancelled_left: set[int] = set()
    cancelled_right: set[int] = set()
    for left_index, event in enumerate(left):
        candidates = right_by_key.get(event.equivalence_key, [])
        match = next((index for index in candidates if index not in cancelled_right), None)
        if match is not None:
            cancelled_left.add(left_index)
            cancelled_right.add(match)
    return (
        tuple(event for index, event in enumerate(left) if index not in cancelled_left),
        tuple(event for index, event in enumerate(right) if index not in cancelled_right),
        len(cancelled_left),
    )


@dataclass(frozen=True)
class InterventionDeltaRecord:
    pair_id: str
    game_id: str
    source_split: str
    left_events: tuple[ObjectEvent, ...]
    right_events: tuple[ObjectEvent, ...]
    left_correspondence: ObjectCorrespondence
    right_correspondence: ObjectCorrespondence
    common_events_cancelled: int
    exclusive_localization: float
    pre_state_identical: bool
    format_version: str = FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "pair_id": self.pair_id,
            "game_id": self.game_id,
            "source_split": self.source_split,
            "left_events": [item.to_dict() for item in self.left_events],
            "right_events": [item.to_dict() for item in self.right_events],
            "left_correspondence": self.left_correspondence.to_dict(),
            "right_correspondence": self.right_correspondence.to_dict(),
            "common_events_cancelled": self.common_events_cancelled,
            "exclusive_localization": self.exclusive_localization,
            "pre_state_identical": self.pre_state_identical,
        }


def compile_intervention_delta(pair: BindingPairRecord) -> InterventionDeltaRecord:
    pre_identical = bool(
        grid_sha256(pair.left.trace.frame_before)
        == grid_sha256(pair.right.trace.frame_before)
    )
    left_match, left_all = compile_arm_events(pair.left.trace)
    right_match, right_all = compile_arm_events(pair.right.trace)
    left, right, cancelled = _cancel_common_events(left_all, right_all)
    left_after = np.asarray(pair.left.trace.frame_after)
    right_after = np.asarray(pair.right.trace.frame_after)
    differential = {
        (int(row), int(col))
        for row, col in np.argwhere(left_after != right_after)
    }
    event_cells = set().union(
        *(set(item.changed_cells) for item in (*left, *right))
    ) if left or right else set()
    localization = (
        len(event_cells & differential) / len(event_cells)
        if event_cells
        else 1.0
    )
    return InterventionDeltaRecord(
        pair_id=pair.pair_digest,
        game_id=pair.game_id,
        source_split=pair.source_split,
        left_events=left,
        right_events=right,
        left_correspondence=left_match,
        right_correspondence=right_match,
        common_events_cancelled=cancelled,
        exclusive_localization=float(localization),
        pre_state_identical=pre_identical,
    )


@dataclass(frozen=True)
class RootedTargetGraph:
    root_kind: str
    action_name: str
    action_family: str
    requested_direction: str
    relation_counts: tuple[tuple[str, int], ...]
    neighbor_roles: tuple[tuple[str, int], ...]
    actor_relation: str
    track_interactions: str = "none"
    last_track_operation: str = "none"
    track_recency: str = "none"

    @property
    def grounded(self) -> bool:
        return self.root_kind != "ungrounded"

    def model_features(self) -> dict[str, float]:
        features: dict[str, float] = {
            f"root:kind={self.root_kind}": 1.0,
            f"root:actor_relation={self.actor_relation}": 1.0,
            f"root:requested_direction={self.requested_direction}": 1.0,
            f"track:interactions={self.track_interactions}": 1.0,
            f"track:last_operation={self.last_track_operation}": 1.0,
            f"track:recency={self.track_recency}": 1.0,
        }
        for key, count in self.relation_counts:
            features[f"root:relation={key}"] = min(int(count), 4) / 4.0
        for key, count in self.neighbor_roles:
            features[f"root:neighbor={key}"] = min(int(count), 4) / 4.0
        return features

    def relation_shuffled(self) -> RootedTargetGraph:
        shuffled: Counter[str] = Counter()
        for key, count in self.relation_counts:
            parts = key.split(":")
            parts = [_DIRECTION_SHUFFLE.get(part, part) for part in parts]
            shuffled[":".join(parts)] += count
        return replace(self, relation_counts=tuple(sorted(shuffled.items())))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ObjectTrackState:
    """Identity-free temporal state for a future enriched V4.5 trace."""

    interactions: int = 0
    last_operation: str = "none"
    transitions_since_interaction: int | None = None
    confidence: float = 0.0

    @property
    def interaction_bucket(self) -> str:
        if self.interactions <= 0:
            return "zero"
        if self.interactions == 1:
            return "one"
        return "two_plus"

    @property
    def recency_bucket(self) -> str:
        if self.transitions_since_interaction is None:
            return "none"
        if self.transitions_since_interaction <= 1:
            return "recent"
        if self.transitions_since_interaction <= 3:
            return "mid"
        return "old"


def build_rooted_target_graph(
    trace: ActionTargetTrace,
    *,
    track: ObjectTrackState | None = None,
) -> RootedTargetGraph:
    observation = build_observation(
        trace.frame_before,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_before,
        levels_completed=trace.levels_completed_before,
    )
    anchor = trace.anchor
    by_id = {item.object_id: item for item in observation.objects}
    root = by_id.get(anchor.target_object_id)
    if root is not None:
        root_kind = "occupied_object"
        center = root.center
    elif anchor.in_bounds and anchor.row is not None and anchor.col is not None:
        root_kind = "virtual_cell"
        center = (float(anchor.row), float(anchor.col))
    elif observation.best_player is not None:
        root_kind = "actor"
        center = tuple(float(value) for value in observation.best_player.position)
    elif anchor.kind == "targetless":
        root_kind = "targetless"
        center = (0.0, 0.0)
    else:
        root_kind = "ungrounded"
        center = (0.0, 0.0)

    relation_counts: Counter[str] = Counter()
    neighbor_roles: Counter[str] = Counter()
    nearby: list[ObjectInfo] = []
    for obj in observation.objects:
        if root is not None and obj.object_id == root.object_id:
            continue
        distance = math.dist(center, obj.center)
        gap = _bbox_gap(root, obj) if root is not None else int(distance)
        if gap == 0:
            proximity = "contact"
        elif gap == 1:
            proximity = "adjacent"
        elif distance <= 8.0:
            proximity = "near"
        else:
            continue
        nearby.append(obj)
        direction = _direction(center, obj.center)
        relation_counts[f"{proximity}:{direction}"] += 1
        if abs(center[0] - obj.center[0]) <= 0.5:
            relation_counts["aligned_row"] += 1
        if abs(center[1] - obj.center[1]) <= 0.5:
            relation_counts["aligned_col"] += 1
        if root is None:
            relative_size = "unknown"
        elif obj.area < root.area:
            relative_size = "smaller"
        elif obj.area > root.area:
            relative_size = "larger"
        else:
            relative_size = "equal"
        is_actor = bool(
            observation.best_player is not None
            and observation.best_player.position in set(obj.cells)
        )
        neighbor_roles[f"{'actor' if is_actor else 'object'}:{relative_size}"] += 1
    for left, right in itertools.combinations(nearby, 2):
        if _bbox_gap(left, right) <= 1:
            relation_counts["second_hop:connected"] += 1

    temporal = track or ObjectTrackState()
    return RootedTargetGraph(
        root_kind=root_kind,
        action_name=trace.selected_action_name,
        action_family=anchor.action_family,
        requested_direction=anchor.requested_direction,
        relation_counts=tuple(sorted(relation_counts.items())),
        neighbor_roles=tuple(sorted(neighbor_roles.items())),
        actor_relation=anchor.actor_relation,
        track_interactions=temporal.interaction_bucket,
        last_track_operation=temporal.last_operation,
        track_recency=temporal.recency_bucket,
    )


@dataclass(frozen=True)
class DiscoveredEventToken:
    token: str
    projection: str
    discordant_pairs: int
    games_with_at_least_10: int
    per_game: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "projection": self.projection,
            "discordant_pairs": self.discordant_pairs,
            "games_with_at_least_10": self.games_with_at_least_10,
            "per_game": dict(self.per_game),
        }


def _candidate_presence(
    delta: InterventionDeltaRecord, projection: str
) -> tuple[set[str], set[str]]:
    return (
        {event.key(projection) for event in delta.left_events},
        {event.key(projection) for event in delta.right_events},
    )


def _candidate_capacity(
    deltas: Sequence[InterventionDeltaRecord], projection: str
) -> dict[str, dict[str, Any]]:
    per_token: dict[str, Counter[str]] = defaultdict(Counter)
    for delta in deltas:
        left, right = _candidate_presence(delta, projection)
        for token in left ^ right:
            per_token[token][delta.game_id] += 1
    result = {}
    games = sorted({delta.game_id for delta in deltas})
    for token, counts in per_token.items():
        per_game = {game: int(counts[game]) for game in games}
        result[token] = {
            "discordant_pairs": int(sum(per_game.values())),
            "games_with_at_least_10": int(
                sum(value >= 10 for value in per_game.values())
            ),
            "per_game": per_game,
        }
    return result


def discover_event_vocabulary(
    deltas: Sequence[InterventionDeltaRecord],
    *,
    minimum_discordant_pairs: int = 75,
    minimum_games_with_10: int = 3,
) -> tuple[tuple[DiscoveredEventToken, ...], dict[str, str]]:
    """Select the finest supported source-only token for each atomic event."""

    capacities = {
        projection: _candidate_capacity(deltas, projection)
        for projection in ("fine", "no_magnitude", "base")
    }
    full_keys = sorted(
        {
            event.fine_key
            for delta in deltas
            for event in (*delta.left_events, *delta.right_events)
        }
    )
    mapping: dict[str, str] = {}
    token_projection: dict[str, str] = {}
    for full_key in full_keys:
        locus, operation, direction, _magnitude_value = full_key.split("|")
        candidates = (
            ("fine", full_key),
            (
                "no_magnitude",
                f"{locus}|{operation}|{direction}|any",
            ),
            ("base", f"{locus}|{operation}|any|any"),
        )
        for projection, token in candidates:
            capacity = capacities[projection].get(token, {})
            if (
                int(capacity.get("discordant_pairs", 0))
                >= minimum_discordant_pairs
                and int(capacity.get("games_with_at_least_10", 0))
                >= minimum_games_with_10
            ):
                mapping[full_key] = token
                token_projection[token] = projection
                break

    selected_presence: dict[str, Counter[str]] = defaultdict(Counter)
    for delta in deltas:
        left = {
            mapping[event.fine_key]
            for event in delta.left_events
            if event.fine_key in mapping
        }
        right = {
            mapping[event.fine_key]
            for event in delta.right_events
            if event.fine_key in mapping
        }
        for token in left ^ right:
            selected_presence[token][delta.game_id] += 1
    games = sorted({delta.game_id for delta in deltas})
    vocabulary = []
    retained: set[str] = set()
    for token, counts in sorted(selected_presence.items()):
        per_game = {game: int(counts[game]) for game in games}
        discordant = int(sum(per_game.values()))
        game_count = int(sum(value >= 10 for value in per_game.values()))
        if (
            discordant >= minimum_discordant_pairs
            and game_count >= minimum_games_with_10
        ):
            retained.add(token)
            vocabulary.append(
                DiscoveredEventToken(
                    token=token,
                    projection=token_projection[token],
                    discordant_pairs=discordant,
                    games_with_at_least_10=game_count,
                    per_game=per_game,
                )
            )
    mapping = {key: value for key, value in mapping.items() if value in retained}
    return tuple(vocabulary), mapping


@dataclass(frozen=True)
class ObjectCausalExample:
    pair_id: str
    game_id: str
    source_split: str
    context: tuple[BoundEvent, ...]
    left_graph: RootedTargetGraph
    right_graph: RootedTargetGraph
    outcomes: Mapping[str, tuple[bool, bool]]

    def is_discordant(self, event: str) -> bool:
        left, right = self.outcomes[event]
        return left != right

    def direction(self, event: str) -> int:
        if not self.is_discordant(event):
            raise ValueError("event direction requires a discordant pair")
        return int(self.outcomes[event][0])

    def model_view(
        self,
        mode: str,
        *,
        root_swap: bool = False,
        relation_shuffle: bool = False,
    ) -> dict[str, float]:
        left_graph = self.right_graph if root_swap else self.left_graph
        right_graph = self.left_graph if root_swap else self.right_graph
        if relation_shuffle:
            left_graph = left_graph.relation_shuffled()
            right_graph = right_graph.relation_shuffled()
        left = _arm_features(left_graph, self.context, mode)
        right = _arm_features(right_graph, self.context, mode)
        return _difference(left, right)


def build_examples(
    pairs: Sequence[BindingPairRecord],
    deltas: Sequence[InterventionDeltaRecord],
    vocabulary: Sequence[DiscoveredEventToken],
    mapping: Mapping[str, str],
) -> list[ObjectCausalExample]:
    by_pair = {delta.pair_id: delta for delta in deltas}
    tokens = tuple(item.token for item in vocabulary)
    examples = []
    for pair in pairs:
        delta = by_pair[pair.pair_digest]
        left_tokens = {
            mapping[event.fine_key]
            for event in delta.left_events
            if event.fine_key in mapping
        }
        right_tokens = {
            mapping[event.fine_key]
            for event in delta.right_events
            if event.fine_key in mapping
        }
        examples.append(
            ObjectCausalExample(
                pair_id=pair.pair_digest,
                game_id=pair.game_id,
                source_split=pair.source_split,
                context=pair.context,
                left_graph=build_rooted_target_graph(pair.left.trace),
                right_graph=build_rooted_target_graph(pair.right.trace),
                outcomes={
                    token: (token in left_tokens, token in right_tokens)
                    for token in tokens
                },
            )
        )
    return examples


def _history_features(
    graph: RootedTargetGraph, context: Sequence[BoundEvent]
) -> dict[str, float]:
    exact = [item for item in context if item.action_name == graph.action_name]
    family = [item for item in context if item.action_family == graph.action_family]
    features = {
        "history:exact_action:coverage": len(exact) / max(1, len(context)),
        "history:family:coverage": len(family) / max(1, len(context)),
    }
    for name, selected in (("exact", exact), ("family", family)):
        for effect in ("target_created", "target_removed", "target_moved"):
            eligible = [item for item in selected if item.applicable[effect]]
            rate = (
                sum(int(item.effects[effect]) for item in eligible) / len(eligible)
                if eligible
                else 0.5
            )
            features[f"history:{name}:{effect}:signed_rate"] = 2.0 * rate - 1.0
    return features


def _arm_features(
    graph: RootedTargetGraph,
    context: Sequence[BoundEvent],
    mode: str,
) -> dict[str, float]:
    if mode == "template":
        return {}
    features: dict[str, float] = {}
    if mode in {"structured", "history_no_root", "action_only", "root_no_history"}:
        features[f"action:name={graph.action_name}"] = 1.0
        features[f"action:family={graph.action_family}"] = 1.0
        features[f"action:direction={graph.requested_direction}"] = 1.0
    if mode in {"structured", "root_no_history"}:
        root = graph.model_features()
        features.update(root)
        if mode == "structured":
            for key in root:
                features[f"interaction:{graph.action_family}:{key}"] = root[key]
    if mode in {"structured", "history_no_root"}:
        features.update(_history_features(graph, context))
    return features


def _difference(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    result = {}
    for key in set(left) | set(right):
        value = float(left.get(key, 0.0)) - float(right.get(key, 0.0))
        if abs(value) > 1e-12:
            result[key] = value
    return result


def validate_model_view(example: ObjectCausalExample, mode: str) -> None:
    rendered = _canonical(example.model_view(mode)).lower()
    forbidden = (
        example.game_id.lower(),
        "game_id",
        "pair_id",
        "frame",
        "sha256",
        "object_id",
        "policy_seed",
        "reset_index",
        "root_index",
        '"row"',
        '"col"',
        '"x"',
        '"y"',
        "value_token",
        "shape_signature",
        "outcome",
        "label",
    )
    for token in forbidden:
        if token and token in rendered:
            raise ValueError(f"forbidden V4.5 model token: {token}")


def _template_probability(example: ObjectCausalExample) -> float:
    def arm_score(graph: RootedTargetGraph) -> float:
        score = 0.0
        score += 0.5 if graph.root_kind == "occupied_object" else -0.25
        score += 0.25 if graph.actor_relation in {"contact", "adjacent"} else 0.0
        return score

    delta = arm_score(example.left_graph) - arm_score(example.right_graph)
    return float(1.0 / (1.0 + math.exp(-delta)))


def _logo_predictions(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, AntisymmetricLinearModel]],
]:
    predictions = {
        mode: {
            event: np.full(len(examples), np.nan, dtype=np.float64)
            for event in events
        }
        for mode in MODEL_MODES
    }
    fold_models: dict[str, dict[str, AntisymmetricLinearModel]] = {}
    for held_out in sorted({item.game_id for item in examples}):
        fold_models[held_out] = {}
        for event in events:
            train = [
                item
                for item in examples
                if item.game_id != held_out and item.is_discordant(event)
            ]
            indices = [
                index
                for index, item in enumerate(examples)
                if item.game_id == held_out and item.is_discordant(event)
            ]
            test = [examples[index] for index in indices]
            if not test:
                continue
            for mode in MODEL_MODES:
                key = f"{event}:{mode}"
                if mode == "template":
                    values = np.asarray(
                        [_template_probability(item) for item in test],
                        dtype=np.float64,
                    )
                    model = AntisymmetricLinearModel((), ())
                else:
                    model = _fit_model(
                        [item.model_view(mode) for item in train],
                        [item.direction(event) for item in train],
                    )
                    values = np.asarray(
                        [model.predict(item.model_view(mode)) for item in test],
                        dtype=np.float64,
                    )
                fold_models[held_out][key] = model
                predictions[mode][event][indices] = values
    return predictions, fold_models


def _calibrate(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
    calibrated = {
        mode: {event: values.copy() for event, values in rows.items()}
        for mode, rows in predictions.items()
    }
    temperatures: dict[str, dict[str, float]] = {
        mode: {} for mode in MODEL_MODES
    }
    for mode in MODEL_MODES:
        for event in events:
            eligible = np.asarray(
                [item.is_discordant(event) for item in examples], dtype=bool
            )
            labels = np.asarray(
                [
                    item.direction(event)
                    for item in examples
                    if item.is_discordant(event)
                ],
                dtype=np.float64,
            )
            raw = predictions[mode][event][eligible]
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped))
            temperature = _fit_temperature(logits, labels)
            temperatures[mode][event] = temperature
            calibrated[mode][event][eligible] = 1.0 / (
                1.0 + np.exp(-np.clip(temperature * logits, -50.0, 50.0))
            )
    return calibrated, temperatures


def _ece(labels: np.ndarray, probabilities: np.ndarray) -> float:
    result = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected]))
                - float(np.mean(labels[selected]))
            )
    return result


def event_metrics(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    per_event = {}
    for event in events:
        eligible = np.asarray(
            [item.is_discordant(event) for item in examples], dtype=bool
        )
        labels = np.asarray(
            [
                item.direction(event)
                for item in examples
                if item.is_discordant(event)
            ],
            dtype=np.int8,
        )
        probabilities = predictions[event][eligible]
        per_event[event] = {
            "discordant_pairs": len(labels),
            "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "ece": _ece(labels, probabilities),
        }
    return {
        "macro_accuracy": float(
            np.mean([row["accuracy"] for row in per_event.values()])
        ),
        "macro_brier": float(
            np.mean([row["brier"] for row in per_event.values()])
        ),
        "macro_ece": float(np.mean([row["ece"] for row in per_event.values()])),
        "per_event": per_event,
    }


def _brier_skill(model: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    denominator = float(baseline["macro_brier"])
    return (
        (denominator - float(model["macro_brier"])) / denominator
        if denominator > 0
        else 0.0
    )


def _controlled_predictions(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
    fold_models: Mapping[str, Mapping[str, AntisymmetricLinearModel]],
    temperatures: Mapping[str, Mapping[str, float]],
    *,
    root_swap: bool = False,
    relation_shuffle: bool = False,
) -> dict[str, np.ndarray]:
    result = {
        event: np.full(len(examples), np.nan, dtype=np.float64)
        for event in events
    }
    for index, example in enumerate(examples):
        for event in events:
            if not example.is_discordant(event):
                continue
            model = fold_models[example.game_id][f"{event}:structured"]
            raw = model.predict(
                example.model_view(
                    "structured",
                    root_swap=root_swap,
                    relation_shuffle=relation_shuffle,
                )
            )
            clipped = float(np.clip(raw, 1e-6, 1 - 1e-6))
            logit = math.log(clipped / (1.0 - clipped))
            temperature = temperatures["structured"][event]
            result[event][index] = 1.0 / (
                1.0 + math.exp(-float(np.clip(temperature * logit, -50, 50)))
            )
    return result


def _identity_diagnostic(
    examples: Sequence[ObjectCausalExample],
) -> dict[str, Any]:
    labels = [item.game_id for item in examples]
    action = _categorical_identity_probe(
        [item.model_view("action_only") for item in examples], labels
    )
    structured = _categorical_identity_probe(
        [item.model_view("structured") for item in examples], labels
    )
    return {
        "action_difference": action,
        "structured_difference": structured,
        "gain": float(structured["accuracy"] - action["accuracy"]),
    }


def _arm_swap_error(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
    fold_models: Mapping[str, Mapping[str, AntisymmetricLinearModel]],
) -> float:
    errors = []
    for item in examples:
        for event in events:
            if not item.is_discordant(event):
                continue
            model = fold_models[item.game_id][f"{event}:structured"]
            row = item.model_view("structured")
            inverted = {key: -value for key, value in row.items()}
            errors.append(abs(model.predict(inverted) - (1.0 - model.predict(row))))
    return max(errors, default=0.0)


def _per_game_transfer(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
    structured: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result = {}
    for game in sorted({item.game_id for item in examples}):
        contributions = []
        for index, item in enumerate(examples):
            if item.game_id != game:
                continue
            for event in events:
                if not item.is_discordant(event):
                    continue
                label = item.direction(event)
                contributions.append(
                    int((structured[event][index] >= 0.5) == label)
                    - int((baseline[event][index] >= 0.5) == label)
                )
        result[game] = (
            {
                "status": "SCORED",
                "scoreable_events": len(contributions),
                "accuracy_gain": float(np.mean(contributions)),
            }
            if contributions
            else {"status": "NOT_SCOREABLE", "scoreable_events": 0}
        )
    return result


def _bootstrap_gain(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
    structured: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    contributions = []
    for index, item in enumerate(examples):
        row = []
        for event in events:
            if item.is_discordant(event):
                label = item.direction(event)
                row.append(
                    int((structured[event][index] >= 0.5) == label)
                    - int((baseline[event][index] >= 0.5) == label)
                )
        if row:
            contributions.append(float(np.mean(row)))
    rng = np.random.default_rng(seed)
    values = [
        float(np.mean(rng.choice(contributions, size=len(contributions), replace=True)))
        for _ in range(samples)
    ]
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _fit_final_models(
    examples: Sequence[ObjectCausalExample],
    events: Sequence[str],
    temperatures: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    models = {}
    for mode in MODEL_MODES:
        models[mode] = {}
        for event in events:
            if mode == "template":
                models[mode][event] = {"template": True}
                continue
            selected = [item for item in examples if item.is_discordant(event)]
            model = _fit_model(
                [item.model_view(mode) for item in selected],
                [item.direction(event) for item in selected],
            )
            calibrated = AntisymmetricLinearModel(
                model.feature_names,
                model.coefficients,
                temperature=temperatures[mode][event],
            )
            models[mode][event] = calibrated.to_dict()
    payload: dict[str, Any] = {
        "format_version": MODEL_FORMAT_VERSION,
        "events": list(events),
        "models": models,
        "fit_intercept": False,
        "arm_swap_is_exact_inversion": True,
    }
    payload["model_checksum"] = _checksum(payload)
    return payload


def default_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "status": "FROZEN_BEFORE_FEASIBILITY",
        "frozen_at": "2026-07-28",
        "source_train_games": list(SOURCE_TRAIN),
        "source_validation_games": list(SOURCE_VALIDATION),
        "source_corpus": {
            "path": (
                "training/sage12/bound_mechanic_pilot_v4_3/"
                "source_train_shards"
            ),
            "pairs": 2396,
            "collection_report_checksum": (
                "a842c0bdd99a1e10ad48c03ded447e231a6767e6af7410192b2f21c4b2948722"
            ),
            "design_only": True,
        },
        "correspondence": {
            "minimum_score": 0.65,
            "ambiguity_margin": 0.10,
            "split_merge_minimum_iou": 0.70,
            "split_merge_minimum_area_ratio": 0.75,
        },
        "vocabulary": {
            "projections": ["fine", "no_magnitude", "base"],
            "minimum_discordant_pairs": 75,
            "minimum_games_with_10": 3,
            "minimum_promoted_events": 2,
            "frozen_after_feasibility": True,
        },
        "gates": {
            "minimum_correspondence_confident_rate": 0.90,
            "maximum_ambiguity_rate": 0.10,
            "minimum_root_grounding_rate": 0.90,
            "minimum_exclusive_localization": 0.90,
            "minimum_macro_brier_skill": 0.10,
            "minimum_macro_accuracy_gain": 0.10,
            "minimum_root_swap_accuracy_drop": 0.10,
            "minimum_relation_shuffle_accuracy_drop": 0.10,
            "maximum_identity_gain_over_action": 0.05,
            "maximum_macro_ece": 0.10,
            "maximum_arm_swap_error": 1e-12,
            "require_every_scoreable_game_nonnegative": True,
            "require_bootstrap_lower_positive": True,
        },
        "evaluation": {"bootstrap_samples": 2000, "random_seed": 451},
        "fresh_collection": {
            "format_version": COLLECTION_FORMAT_VERSION,
            "context_full_traces": 8,
            "tree_depth": 3,
            "source_train": {
                "roots_per_game": 32,
                "seeds": list(SOURCE_SEEDS),
                "minimum_pairs": 2000,
                "action_budget_per_reset": 32,
                "maximum_resets_per_game": 64,
            },
            "source_validation": {
                "roots_per_game": 64,
                "seeds": list(VALIDATION_SEEDS),
                "minimum_pairs": 1000,
                "minimum_discordant_pairs_per_event": 30,
                "action_budget_per_reset": 32,
                "maximum_resets_per_game": 96,
            },
        },
        "firewall": {
            "source_validation_opened": False,
            "historical_opened": False,
            "holdout_opened": False,
            "ar25_opened": False,
            "absolute_coordinates_in_model_view": False,
            "raw_values_in_model_view": False,
            "object_ids_in_model_view": False,
            "global_scene_signature_in_model_view": False,
            "qwen_authorized": False,
            "gnn_authorized": False,
            "world_model_authorized": False,
            "ebm_authorized": False,
            "controller_authorized": False,
        },
    }
    payload["manifest_checksum"] = _checksum(payload)
    return payload


def freeze_manifest(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    payload = default_manifest()
    _write_json(Path(output_dir) / "frozen_manifest.json", payload)
    return payload


def load_frozen_manifest(
    path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported SAGE12 V4.5 manifest")
    expected = str(payload.get("manifest_checksum", ""))
    clean = dict(payload)
    clean.pop("manifest_checksum", None)
    if expected != _checksum(clean):
        raise ValueError("SAGE12 V4.5 manifest checksum mismatch")
    if tuple(payload["source_train_games"]) != SOURCE_TRAIN:
        raise ValueError("V4.5 source split drift")
    if tuple(payload["source_validation_games"]) != SOURCE_VALIDATION:
        raise ValueError("V4.5 validation split drift")
    return payload


def _compiler_quality(
    pairs: Sequence[BindingPairRecord],
    deltas: Sequence[InterventionDeltaRecord],
    examples: Sequence[ObjectCausalExample],
) -> dict[str, Any]:
    total_components = 0
    ambiguous = 0
    for pair, delta in zip(pairs, deltas):
        for trace, correspondence in (
            (pair.left.trace, delta.left_correspondence),
            (pair.right.trace, delta.right_correspondence),
        ):
            before = build_observation(
                trace.frame_before,
                available_actions=trace.available_action_names,
            )
            after = build_observation(
                trace.frame_after,
                available_actions=trace.available_action_names,
            )
            total_components += len(before.objects) + len(after.objects)
            ambiguous += correspondence.ambiguity_count
    ambiguity_rate = ambiguous / max(1, total_components)
    grounded = sum(
        int(graph.grounded)
        for item in examples
        for graph in (item.left_graph, item.right_graph)
    )
    return {
        "pairs": len(pairs),
        "arms": 2 * len(pairs),
        "component_assignments": total_components,
        "ambiguous_assignments": ambiguous,
        "ambiguity_rate": ambiguity_rate,
        "correspondence_confident_rate": 1.0 - ambiguity_rate,
        "root_grounding_rate": grounded / max(1, 2 * len(examples)),
        "exclusive_localization": float(
            np.mean([item.exclusive_localization for item in deltas])
        ),
        "identical_pre_state_rate": float(
            np.mean([item.pre_state_identical for item in deltas])
        ),
    }


def run_feasibility(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    source_manifest = json.loads(
        (V43_OUTPUT_DIR / "source_train_collection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_checksum_ok = bool(
        source_manifest["report_checksum"]
        == frozen["source_corpus"]["collection_report_checksum"]
    )
    pairs = load_pairs(
        V43_OUTPUT_DIR / "source_train_shards",
        tuple(frozen["source_train_games"]),
    )
    deltas = [compile_intervention_delta(pair) for pair in pairs]
    vocab_cfg = frozen["vocabulary"]
    vocabulary, mapping = discover_event_vocabulary(
        deltas,
        minimum_discordant_pairs=int(vocab_cfg["minimum_discordant_pairs"]),
        minimum_games_with_10=int(vocab_cfg["minimum_games_with_10"]),
    )
    examples = build_examples(pairs, deltas, vocabulary, mapping)
    for mode in MODEL_MODES:
        if mode != "template":
            for example in examples:
                validate_model_view(example, mode)
    quality = _compiler_quality(pairs, deltas, examples)
    events = tuple(item.token for item in vocabulary)
    destination = Path(output_dir)
    delta_path = destination / "feasibility_deltas.jsonl"
    _write_jsonl(delta_path, (item.to_dict() for item in deltas))
    vocabulary_payload: dict[str, Any] = {
        "format_version": VOCABULARY_FORMAT_VERSION,
        "status": "FROZEN_FROM_V4_3_SOURCE_DESIGN",
        "events": [item.to_dict() for item in vocabulary],
        "fine_to_promoted": dict(sorted(mapping.items())),
        "source_pairs": len(pairs),
        "source_games": list(frozen["source_train_games"]),
        "validation_opened": False,
    }
    vocabulary_payload["vocabulary_checksum"] = _checksum(vocabulary_payload)
    _write_json(destination / "event_vocabulary.json", vocabulary_payload)

    gates_cfg = frozen["gates"]
    compiler_gates = {
        "source_checksum_matches": source_checksum_ok,
        "strict_json_validity": True,
        "identical_pre_states": quality["identical_pre_state_rate"] == 1.0,
        "minimum_correspondence_confident_rate": (
            quality["correspondence_confident_rate"]
            >= float(gates_cfg["minimum_correspondence_confident_rate"])
        ),
        "maximum_ambiguity_rate": (
            quality["ambiguity_rate"]
            <= float(gates_cfg["maximum_ambiguity_rate"])
        ),
        "minimum_root_grounding_rate": (
            quality["root_grounding_rate"]
            >= float(gates_cfg["minimum_root_grounding_rate"])
        ),
        "minimum_exclusive_localization": (
            quality["exclusive_localization"]
            >= float(gates_cfg["minimum_exclusive_localization"])
        ),
        "minimum_promoted_events": (
            len(vocabulary) >= int(vocab_cfg["minimum_promoted_events"])
        ),
    }
    predictive: dict[str, Any] = {
        "status": "NOT_SCOREABLE",
        "events": list(events),
    }
    model_payload: dict[str, Any] | None = None
    if events:
        raw, fold_models = _logo_predictions(examples, events)
        calibrated, temperatures = _calibrate(examples, events, raw)
        metrics = {
            mode: event_metrics(examples, events, calibrated[mode])
            for mode in MODEL_MODES
        }
        stronger = min(
            BASELINE_MODES, key=lambda mode: metrics[mode]["macro_brier"]
        )
        skill = _brier_skill(metrics["structured"], metrics[stronger])
        accuracy_gain = (
            metrics["structured"]["macro_accuracy"]
            - metrics[stronger]["macro_accuracy"]
        )
        root_swapped = _controlled_predictions(
            examples,
            events,
            fold_models,
            temperatures,
            root_swap=True,
        )
        relation_shuffled = _controlled_predictions(
            examples,
            events,
            fold_models,
            temperatures,
            relation_shuffle=True,
        )
        root_swap_metrics = event_metrics(examples, events, root_swapped)
        relation_shuffle_metrics = event_metrics(
            examples, events, relation_shuffled
        )
        root_swap_drop = (
            metrics["structured"]["macro_accuracy"]
            - root_swap_metrics["macro_accuracy"]
        )
        relation_shuffle_drop = (
            metrics["structured"]["macro_accuracy"]
            - relation_shuffle_metrics["macro_accuracy"]
        )
        identity = _identity_diagnostic(examples)
        arm_swap_error = _arm_swap_error(examples, events, fold_models)
        per_game = _per_game_transfer(
            examples,
            events,
            calibrated["structured"],
            calibrated[stronger],
        )
        bootstrap = _bootstrap_gain(
            examples,
            events,
            calibrated["structured"],
            calibrated[stronger],
            samples=int(frozen["evaluation"]["bootstrap_samples"]),
            seed=int(frozen["evaluation"]["random_seed"]),
        )
        predictive_gates = {
            "minimum_macro_brier_skill": (
                skill >= float(gates_cfg["minimum_macro_brier_skill"])
            ),
            "minimum_macro_accuracy_gain": (
                accuracy_gain >= float(gates_cfg["minimum_macro_accuracy_gain"])
            ),
            "minimum_root_swap_accuracy_drop": (
                root_swap_drop
                >= float(gates_cfg["minimum_root_swap_accuracy_drop"])
            ),
            "minimum_relation_shuffle_accuracy_drop": (
                relation_shuffle_drop
                >= float(gates_cfg["minimum_relation_shuffle_accuracy_drop"])
            ),
            "maximum_identity_gain": (
                identity["gain"]
                <= float(gates_cfg["maximum_identity_gain_over_action"])
            ),
            "maximum_macro_ece": (
                metrics["structured"]["macro_ece"]
                <= float(gates_cfg["maximum_macro_ece"])
            ),
            "exact_arm_swap_inversion": (
                arm_swap_error <= float(gates_cfg["maximum_arm_swap_error"])
            ),
            "every_scoreable_game_nonnegative": all(
                item.get("accuracy_gain", 0.0) >= 0.0
                for item in per_game.values()
                if item["status"] == "SCORED"
            ),
            "bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
        }
        predictive = {
            "status": (
                "PASS" if all(predictive_gates.values()) else "FAIL_CLOSED"
            ),
            "events": list(events),
            "metrics": metrics,
            "stronger_baseline": stronger,
            "macro_brier_skill": skill,
            "macro_accuracy_gain": accuracy_gain,
            "root_swap": {
                "metrics": root_swap_metrics,
                "accuracy_drop": root_swap_drop,
            },
            "relation_shuffle": {
                "metrics": relation_shuffle_metrics,
                "accuracy_drop": relation_shuffle_drop,
            },
            "identity": identity,
            "arm_swap_maximum_error": arm_swap_error,
            "per_game": per_game,
            "bootstrap_accuracy_gain": bootstrap,
            "gates": predictive_gates,
        }
        model_payload = _fit_final_models(examples, events, temperatures)
    passed = bool(
        all(compiler_gates.values()) and predictive.get("status") == "PASS"
    )
    payload: dict[str, Any] = {
        "format_version": FEASIBILITY_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "source_corpus_report_checksum": source_manifest["report_checksum"],
        "source_only_design_audit": True,
        "validation_opened": False,
        "compiler_quality": quality,
        "compiler_gates": compiler_gates,
        "vocabulary": [item.to_dict() for item in vocabulary],
        "vocabulary_checksum": vocabulary_payload["vocabulary_checksum"],
        "predictive": predictive,
        "feasibility_delta_path": delta_path.as_posix(),
        "feasibility_delta_sha256": _file_sha256(delta_path),
        "fresh_source_collection_authorized": passed,
        "validation_collection_authorized": False,
        "world_model_protocol_authorized": False,
        "gnn_authorized": False,
        "qwen_authorized": False,
        "ebm_authorized": False,
        "controller_authorized": False,
    }
    payload["feasibility_checksum"] = _checksum(payload)
    _write_json(destination / "feasibility_result.json", payload)
    if passed and model_payload is not None:
        _write_json(destination / "feasibility_model.json", model_payload)
    return payload


def _closed_stage(
    *,
    stage: str,
    predecessor_path: Path,
    predecessor_checksum_field: str,
    output_path: Path,
    authorized_field: str,
) -> dict[str, Any]:
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    authorized = bool(predecessor.get(authorized_field, False))
    if authorized:
        raise RuntimeError(
            f"{stage} is authorized; run its prospective collection checkpoint"
        )
    payload: dict[str, Any] = {
        "format_version": f"sage12-object-causal-{stage}-closure-v4.5",
        "status": f"SKIPPED_{predecessor['status']}",
        "predecessor_checksum": predecessor[predecessor_checksum_field],
        "source_validation_opened": False,
        "world_model_protocol_authorized": False,
        "gnn_authorized": False,
        "ebm_authorized": False,
        "controller_authorized": False,
    }
    payload[f"{stage}_checksum"] = _checksum(payload)
    _write_json(output_path, payload)
    return payload


def run_source_collection(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    return _closed_stage(
        stage="source-collection",
        predecessor_path=destination / "feasibility_result.json",
        predecessor_checksum_field="feasibility_checksum",
        output_path=destination / "source_collection.json",
        authorized_field="fresh_source_collection_authorized",
    )


def run_source_preflight(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    source_path = destination / "source_collection.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") == "COMPLETE":
        raise RuntimeError("fresh source evaluation requires the prospective model freeze")
    return _closed_stage(
        stage="source-preflight",
        predecessor_path=source_path,
        predecessor_checksum_field="source-collection_checksum",
        output_path=destination / "source_preflight.json",
        authorized_field="source_preflight_authorized",
    )


def run_validation_collection(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    return _closed_stage(
        stage="validation-collection",
        predecessor_path=destination / "source_preflight.json",
        predecessor_checksum_field="source-preflight_checksum",
        output_path=destination / "validation_collection.json",
        authorized_field="validation_collection_authorized",
    )


def run_final_evaluation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    return _closed_stage(
        stage="final-evaluation",
        predecessor_path=destination / "validation_collection.json",
        predecessor_checksum_field="validation-collection_checksum",
        output_path=destination / "final_result.json",
        authorized_field="validation_evaluation_authorized",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "feasibility",
            "collect-source",
            "preflight",
            "collect-validation",
            "evaluate",
        ),
    )
    parser.add_argument(
        "--frozen-manifest", default=str(DEFAULT_FROZEN_MANIFEST_PATH)
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_manifest(args.output_dir)
    elif args.command == "feasibility":
        result = run_feasibility(
            frozen_manifest_path=args.frozen_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "collect-source":
        result = run_source_collection(output_dir=args.output_dir)
    elif args.command == "preflight":
        result = run_source_preflight(output_dir=args.output_dir)
    elif args.command == "collect-validation":
        result = run_validation_collection(output_dir=args.output_dir)
    else:
        result = run_final_evaluation(output_dir=args.output_dir)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTION_FORMAT_VERSION",
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DiscoveredEventToken",
    "InterventionDeltaRecord",
    "ObjectCausalExample",
    "ObjectCorrespondence",
    "ObjectEvent",
    "ObjectTrackState",
    "RootedTargetGraph",
    "build_examples",
    "build_rooted_target_graph",
    "compile_arm_events",
    "compile_intervention_delta",
    "default_manifest",
    "discover_event_vocabulary",
    "freeze_manifest",
    "load_frozen_manifest",
    "match_objects",
    "run_feasibility",
    "run_final_evaluation",
    "run_source_collection",
    "run_source_preflight",
    "run_validation_collection",
    "validate_model_view",
]
