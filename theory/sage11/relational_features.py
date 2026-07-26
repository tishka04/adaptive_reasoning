"""Versioned object-relational features shared by collection and live use."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


RELATIONAL_FEATURE_FORMAT_VERSION = "sage11-object-relations-v1"
COUNT_BUCKETS: Tuple[str, ...] = (
    "zero",
    "one",
    "few",
    "some",
    "many",
)
DISTANCE_BUCKETS: Tuple[str, ...] = (
    "unavailable",
    "zero",
    "one",
    "two_to_four",
    "five_plus",
)
PAIR_DISTANCE_BUCKETS: Tuple[str, ...] = (
    "unavailable",
    "contact",
    "near",
    "medium",
    "far",
)
DIRECTIONS: Tuple[str, ...] = ("negative", "zero", "positive")
ASPECTS: Tuple[str, ...] = ("square", "vertical", "horizontal")
OBJECT_SIZE_BUCKETS: Tuple[str, ...] = ("one", "few", "some", "many")

STATE_RELATIONAL_FEATURE_NAMES: Tuple[str, ...] = (
    *(f"state_object_count:{bucket}" for bucket in COUNT_BUCKETS),
    "state_player:present",
    "state_object_pair:any_contact",
    "state_object_pair:any_row_aligned",
    "state_object_pair:any_column_aligned",
    *(
        f"state_object_pair:min_distance_{bucket}"
        for bucket in PAIR_DISTANCE_BUCKETS
    ),
    "state_player_object:any_contact",
    "state_player_object:any_row_aligned",
    "state_player_object:any_column_aligned",
    *(
        f"state_player_object:min_distance_{bucket}"
        for bucket in DISTANCE_BUCKETS
    ),
)
ACTION_RELATIONAL_FEATURE_NAMES: Tuple[str, ...] = (
    "action_target:has_xy",
    "action_target:inside_object",
    "action_target:contacts_object",
    "action_target:row_aligned_with_object",
    "action_target:column_aligned_with_object",
    *(
        f"action_target:object_distance_{bucket}"
        for bucket in DISTANCE_BUCKETS
    ),
    *(
        f"action_target:nearest_object_row_{direction}"
        for direction in DIRECTIONS
    ),
    *(
        f"action_target:nearest_object_column_{direction}"
        for direction in DIRECTIONS
    ),
    *(
        f"action_target:nearest_object_size_{bucket}"
        for bucket in OBJECT_SIZE_BUCKETS
    ),
    *(
        f"action_target:nearest_object_aspect_{aspect}"
        for aspect in ASPECTS
    ),
    *(
        f"action_target:player_distance_{bucket}"
        for bucket in DISTANCE_BUCKETS
    ),
    "action_target:row_aligned_with_player",
    "action_target:column_aligned_with_player",
)
RELATIONAL_FEATURE_NAMES: Tuple[str, ...] = (
    STATE_RELATIONAL_FEATURE_NAMES + ACTION_RELATIONAL_FEATURE_NAMES
)


@dataclass(frozen=True)
class RelationalFeatureSchema:
    """Fixed ordering for archived and live object-relational features."""

    format_version: str = RELATIONAL_FEATURE_FORMAT_VERSION
    feature_names: Tuple[str, ...] = RELATIONAL_FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.format_version != RELATIONAL_FEATURE_FORMAT_VERSION:
            raise ValueError("unsupported SAGE.11 relational feature version")
        if self.feature_names != RELATIONAL_FEATURE_NAMES:
            raise ValueError("SAGE.11 relational feature ordering is fixed")

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def feature_to_index(self) -> Mapping[str, int]:
        return {
            name: index
            for index, name in enumerate(self.feature_names)
        }

    @property
    def state_feature_indices(self) -> Tuple[int, ...]:
        return tuple(range(len(STATE_RELATIONAL_FEATURE_NAMES)))

    @property
    def action_dependent_feature_indices(self) -> Tuple[int, ...]:
        return tuple(
            range(
                len(STATE_RELATIONAL_FEATURE_NAMES),
                self.feature_count,
            )
        )

    @property
    def checksum(self) -> str:
        payload = {
            "format_version": self.format_version,
            "feature_names": list(self.feature_names),
        }
        return hashlib.sha256(json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "feature_names": list(self.feature_names),
            "feature_count": self.feature_count,
            "state_feature_count": len(self.state_feature_indices),
            "action_dependent_feature_count": len(
                self.action_dependent_feature_indices
            ),
            "checksum": self.checksum,
        }


RELATIONAL_FEATURE_SCHEMA = RelationalFeatureSchema()


def encode_relational_features(
    observation: Any,
    *,
    action_name: str,
    action_data: Mapping[str, Any] | None,
    schema: RelationalFeatureSchema = RELATIONAL_FEATURE_SCHEMA,
) -> np.ndarray:
    """Encode generic geometry without raw coordinates or game identity."""
    vector = np.zeros(schema.feature_count, dtype=np.float32)
    index = schema.feature_to_index
    objects = tuple(
        _ObjectGeometry.from_object(item)
        for item in tuple(getattr(observation, "objects", ()) or ())
    )
    player = _player_position(observation)
    non_player_objects = tuple(
        item
        for item in objects
        if player is None or not item.contains(*player)
    )

    vector[index[
        f"state_object_count:{_count_bucket(len(non_player_objects))}"
    ]] = 1.0
    vector[index["state_player:present"]] = float(player is not None)
    _encode_object_pair_state(vector, index, non_player_objects)
    _encode_player_object_state(
        vector,
        index,
        player,
        non_player_objects,
    )
    _encode_action_target(
        vector,
        index,
        action_name=str(action_name),
        action_data=action_data,
        player=player,
        objects=non_player_objects,
    )
    return vector


def _encode_object_pair_state(
    vector: np.ndarray,
    index: Mapping[str, int],
    objects: Sequence["_ObjectGeometry"],
) -> None:
    distances = []
    any_row_aligned = False
    any_column_aligned = False
    for left_index, left in enumerate(objects):
        for right in objects[left_index + 1:]:
            distances.append(_bbox_distance(left.bbox, right.bbox))
            any_row_aligned = bool(
                any_row_aligned
                or _ranges_overlap(
                    left.bbox[0],
                    left.bbox[2],
                    right.bbox[0],
                    right.bbox[2],
                )
            )
            any_column_aligned = bool(
                any_column_aligned
                or _ranges_overlap(
                    left.bbox[1],
                    left.bbox[3],
                    right.bbox[1],
                    right.bbox[3],
                )
            )
    minimum = min(distances) if distances else None
    vector[index["state_object_pair:any_contact"]] = float(
        minimum is not None and minimum <= 1
    )
    vector[index["state_object_pair:any_row_aligned"]] = float(
        any_row_aligned
    )
    vector[index["state_object_pair:any_column_aligned"]] = float(
        any_column_aligned
    )
    vector[index[
        "state_object_pair:min_distance_"
        f"{_pair_distance_bucket(minimum)}"
    ]] = 1.0


def _encode_player_object_state(
    vector: np.ndarray,
    index: Mapping[str, int],
    player: tuple[int, int] | None,
    objects: Sequence["_ObjectGeometry"],
) -> None:
    distances = (
        [
            _point_bbox_distance(player, item.bbox)
            for item in objects
        ]
        if player is not None
        else []
    )
    minimum = min(distances) if distances else None
    vector[index["state_player_object:any_contact"]] = float(
        minimum is not None and minimum <= 1
    )
    vector[index["state_player_object:any_row_aligned"]] = float(
        player is not None
        and any(item.bbox[0] <= player[0] <= item.bbox[2] for item in objects)
    )
    vector[index["state_player_object:any_column_aligned"]] = float(
        player is not None
        and any(item.bbox[1] <= player[1] <= item.bbox[3] for item in objects)
    )
    vector[index[
        "state_player_object:min_distance_"
        f"{_distance_bucket(minimum)}"
    ]] = 1.0


def _encode_action_target(
    vector: np.ndarray,
    index: Mapping[str, int],
    *,
    action_name: str,
    action_data: Mapping[str, Any] | None,
    player: tuple[int, int] | None,
    objects: Sequence["_ObjectGeometry"],
) -> None:
    target = _action_target(action_name, action_data)
    if target is None:
        vector[index[
            "action_target:object_distance_unavailable"
        ]] = 1.0
        vector[index[
            "action_target:player_distance_unavailable"
        ]] = 1.0
        return
    vector[index["action_target:has_xy"]] = 1.0

    ranked = sorted(
        objects,
        key=lambda item: (
            _point_bbox_distance(target, item.bbox),
            item.area,
            item.bbox,
        ),
    )
    nearest_distance: int | None = None
    nearest: _ObjectGeometry | None = None
    if ranked:
        nearest = ranked[0]
        nearest_distance = _point_bbox_distance(target, nearest.bbox)
    vector[index["action_target:inside_object"]] = float(
        nearest_distance == 0
    )
    vector[index["action_target:contacts_object"]] = float(
        nearest_distance is not None and nearest_distance <= 1
    )
    vector[index["action_target:row_aligned_with_object"]] = float(
        any(item.bbox[0] <= target[0] <= item.bbox[2] for item in objects)
    )
    vector[index["action_target:column_aligned_with_object"]] = float(
        any(item.bbox[1] <= target[1] <= item.bbox[3] for item in objects)
    )
    vector[index[
        f"action_target:object_distance_{_distance_bucket(nearest_distance)}"
    ]] = 1.0
    if nearest is not None:
        row_direction = _direction(target[0] - nearest.center[0])
        column_direction = _direction(target[1] - nearest.center[1])
        vector[index[
            f"action_target:nearest_object_row_{row_direction}"
        ]] = 1.0
        vector[index[
            f"action_target:nearest_object_column_{column_direction}"
        ]] = 1.0
        vector[index[
            "action_target:nearest_object_size_"
            f"{_object_size_bucket(nearest.area)}"
        ]] = 1.0
        vector[index[
            "action_target:nearest_object_aspect_"
            f"{nearest.aspect}"
        ]] = 1.0

    player_distance = (
        abs(target[0] - player[0]) + abs(target[1] - player[1])
        if player is not None
        else None
    )
    vector[index[
        f"action_target:player_distance_{_distance_bucket(player_distance)}"
    ]] = 1.0
    vector[index["action_target:row_aligned_with_player"]] = float(
        player is not None and target[0] == player[0]
    )
    vector[index["action_target:column_aligned_with_player"]] = float(
        player is not None and target[1] == player[1]
    )


@dataclass(frozen=True)
class _ObjectGeometry:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int

    @classmethod
    def from_object(cls, item: Any) -> "_ObjectGeometry":
        raw_bbox = tuple(
            int(value)
            for value in tuple(
                getattr(item, "bbox", (0, 0, 0, 0))
                or (0, 0, 0, 0)
            )
        )
        if len(raw_bbox) != 4:
            raise ValueError("object bbox must contain four coordinates")
        raw_center = tuple(
            float(value)
            for value in tuple(
                getattr(item, "center", ()) or ()
            )
        )
        center = (
            (raw_bbox[0] + raw_bbox[2]) / 2.0,
            (raw_bbox[1] + raw_bbox[3]) / 2.0,
        )
        if len(raw_center) == 2:
            center = (raw_center[0], raw_center[1])
        area = max(1, int(getattr(item, "area", 1) or 1))
        return cls(bbox=raw_bbox, center=center, area=area)

    @property
    def aspect(self) -> str:
        height = self.bbox[2] - self.bbox[0] + 1
        width = self.bbox[3] - self.bbox[1] + 1
        if height == width:
            return "square"
        return "vertical" if height > width else "horizontal"

    def contains(self, row: int, column: int) -> bool:
        return (
            self.bbox[0] <= row <= self.bbox[2]
            and self.bbox[1] <= column <= self.bbox[3]
        )


def _player_position(observation: Any) -> tuple[int, int] | None:
    player = getattr(observation, "best_player", None)
    if player is None:
        return None
    position = tuple(getattr(player, "position", ()) or ())
    if len(position) != 2:
        return None
    return int(position[0]), int(position[1])


def _action_target(
    action_name: str,
    action_data: Mapping[str, Any] | None,
) -> tuple[int, int] | None:
    del action_name
    data = dict(action_data or {})
    if "x" not in data or "y" not in data:
        return None
    try:
        return int(data["y"]), int(data["x"])
    except (TypeError, ValueError):
        return None


def _bbox_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    row_gap = max(left[0] - right[2], right[0] - left[2], 0)
    column_gap = max(left[1] - right[3], right[1] - left[3], 0)
    return int(row_gap + column_gap)


def _point_bbox_distance(
    point: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> int:
    row, column = point
    row_gap = max(bbox[0] - row, row - bbox[2], 0)
    column_gap = max(bbox[1] - column, column - bbox[3], 0)
    return int(row_gap + column_gap)


def _ranges_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return max(left_start, right_start) <= min(left_end, right_end)


def _count_bucket(value: int) -> str:
    count = max(0, int(value))
    if count == 0:
        return "zero"
    if count == 1:
        return "one"
    if count <= 4:
        return "few"
    if count <= 15:
        return "some"
    return "many"


def _object_size_bucket(value: int) -> str:
    bucket = _count_bucket(value)
    return "one" if bucket == "zero" else bucket


def _distance_bucket(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 4:
        return "two_to_four"
    return "five_plus"


def _pair_distance_bucket(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if value <= 1:
        return "contact"
    if value <= 4:
        return "near"
    if value <= 12:
        return "medium"
    return "far"


def _direction(value: float) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


__all__ = [
    "ACTION_RELATIONAL_FEATURE_NAMES",
    "RELATIONAL_FEATURE_FORMAT_VERSION",
    "RELATIONAL_FEATURE_NAMES",
    "RELATIONAL_FEATURE_SCHEMA",
    "RelationalFeatureSchema",
    "STATE_RELATIONAL_FEATURE_NAMES",
    "encode_relational_features",
]
