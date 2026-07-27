"""Action-target grounded data contract for the SAGE12 V3 pilot.

The raw trace keeps enough information to audit an observed label.  Model
features are produced only by :func:`model_features`; provenance, absolute
coordinates, colours, grids, and future state never enter that projection.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from theory.live_transition_loop import build_observation, build_transition_record
from v3.schemas import GameObservation, ObjectInfo, TransitionRecord


TRACE_FORMAT_VERSION = "sage12-action-target-trace-v3"
EFFECT_LABELS = (
    "actor_displaced",
    "target_created",
    "target_removed",
    "target_moved",
)
PROJECTION_LADDER = ("full", "no_shape", "coarse")

_MOVE_DIRECTIONS: dict[str, tuple[str, int, int]] = {
    "ACTION1": ("up", -1, 0),
    "ACTION2": ("down", 1, 0),
    "ACTION3": ("left", 0, -1),
    "ACTION4": ("right", 0, 1),
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


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def grid_sha256(grid: Any) -> str:
    array = np.asarray(grid, dtype=np.int16)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def infer_background(grid: Any) -> int:
    array = np.asarray(grid, dtype=np.int32)
    values, counts = np.unique(array, return_counts=True)
    return int(values[int(np.argmax(counts))])


@dataclass(frozen=True)
class ActionTargetAnchor:
    """A pre-action anchor, with absolute fields retained as provenance only."""

    kind: str
    action_family: str
    requested_direction: str = "none"
    row: int | None = None
    col: int | None = None
    in_bounds: bool = False
    occupied: bool = False
    target_object_id: int | None = None
    target_area_bucket: str = "none"
    target_aspect_bucket: str = "none"
    target_affordance: str = "none"
    actor_relation: str = "unknown"
    actor_relative_direction: str = "unknown"
    path_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionTargetAnchor":
        return cls(
            kind=str(payload["kind"]),
            action_family=str(payload["action_family"]),
            requested_direction=str(payload.get("requested_direction", "none")),
            row=_optional_int(payload.get("row")),
            col=_optional_int(payload.get("col")),
            in_bounds=bool(payload.get("in_bounds", False)),
            occupied=bool(payload.get("occupied", False)),
            target_object_id=_optional_int(payload.get("target_object_id")),
            target_area_bucket=str(payload.get("target_area_bucket", "none")),
            target_aspect_bucket=str(payload.get("target_aspect_bucket", "none")),
            target_affordance=str(payload.get("target_affordance", "none")),
            actor_relation=str(payload.get("actor_relation", "unknown")),
            actor_relative_direction=str(
                payload.get("actor_relative_direction", "unknown")
            ),
            path_status=str(payload.get("path_status", "unknown")),
        )

    def model_view(self, projection: str = "full") -> dict[str, Any]:
        if projection not in PROJECTION_LADDER:
            raise ValueError(f"unknown action-target projection: {projection}")
        features: dict[str, Any] = {
            "action_family": self.action_family,
            "requested_direction": self.requested_direction,
            "anchor_kind": self.kind,
            "anchor_occupied": int(self.occupied),
            "actor_relation": self.actor_relation,
            "actor_relative_direction": self.actor_relative_direction,
            "path_status": self.path_status,
            "target_affordance": self.target_affordance,
            "target_area_bucket": self.target_area_bucket,
            "target_aspect_bucket": self.target_aspect_bucket,
        }
        if projection in {"no_shape", "coarse"}:
            features.pop("target_area_bucket")
            features.pop("target_aspect_bucket")
            features.pop("target_affordance")
        if projection == "coarse":
            features["actor_relation"] = _coarse_relation(self.actor_relation)
            features.pop("actor_relative_direction")
        return features


@dataclass(frozen=True)
class ObservedActionTargetEffects:
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    ambiguity_reasons: tuple[str, ...] = ()
    noop: bool = False
    level_complete: bool = False
    game_over: bool = False

    def __post_init__(self) -> None:
        if set(self.labels) != set(EFFECT_LABELS):
            raise ValueError("V3 effects require the complete frozen label set")
        if set(self.applicable) != set(EFFECT_LABELS):
            raise ValueError("V3 effects require applicability for every label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": {key: bool(self.labels[key]) for key in EFFECT_LABELS},
            "applicable": {
                key: bool(self.applicable[key]) for key in EFFECT_LABELS
            },
            "ambiguity_reasons": list(self.ambiguity_reasons),
            "noop": bool(self.noop),
            "level_complete": bool(self.level_complete),
            "game_over": bool(self.game_over),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ObservedActionTargetEffects":
        labels = dict(payload.get("labels", {}))
        applicable = dict(payload.get("applicable", {}))
        return cls(
            labels={key: bool(labels.get(key, False)) for key in EFFECT_LABELS},
            applicable={
                key: bool(applicable.get(key, False)) for key in EFFECT_LABELS
            },
            ambiguity_reasons=tuple(
                str(item) for item in payload.get("ambiguity_reasons", ())
            ),
            noop=bool(payload.get("noop", False)),
            level_complete=bool(payload.get("level_complete", False)),
            game_over=bool(payload.get("game_over", False)),
        )


@dataclass(frozen=True)
class ActionTargetTrace:
    game_id: str
    source_split: str
    policy_seed: int
    reset_index: int
    step_index: int
    collection_phase: str
    available_action_names: tuple[str, ...]
    selected_action_name: str
    selected_action_data: Mapping[str, Any]
    anchor: ActionTargetAnchor
    effects: ObservedActionTargetEffects
    frame_before: Any
    frame_after: Any
    game_state_before: str
    game_state_after: str
    levels_completed_before: int
    levels_completed_after: int
    trace_digest: str = ""
    format_version: str = TRACE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != TRACE_FORMAT_VERSION:
            raise ValueError("unsupported SAGE12 action-target trace version")
        if self.source_split not in {"source_train", "source_validation"}:
            raise ValueError("SAGE12 V3 traces are source-only")
        if self.selected_action_name not in self.available_action_names:
            raise ValueError("executed action must have been legal")
        if not self.trace_digest:
            digest_payload = {
                "game_id": self.game_id,
                "source_split": self.source_split,
                "policy_seed": self.policy_seed,
                "reset_index": self.reset_index,
                "step_index": self.step_index,
                "frame_before_sha256": grid_sha256(self.frame_before),
                "action_name": self.selected_action_name,
                "action_data": dict(self.selected_action_data),
            }
            object.__setattr__(
                self,
                "trace_digest",
                hashlib.sha256(_canonical(digest_payload).encode("utf-8")).hexdigest(),
            )

    def model_features(self, projection: str = "full") -> dict[str, Any]:
        """Return the only fields authorized as model input."""
        return {
            "selected_action_name": self.selected_action_name,
            **self.anchor.model_view(projection),
        }

    def exact_repeat_key(self) -> str:
        payload = {
            "frame_before_sha256": grid_sha256(self.frame_before),
            "action_name": self.selected_action_name,
            "action_data": dict(self.selected_action_data),
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "game_id": self.game_id,
            "source_split": self.source_split,
            "policy_seed": self.policy_seed,
            "reset_index": self.reset_index,
            "step_index": self.step_index,
            "collection_phase": self.collection_phase,
            "available_action_names": list(self.available_action_names),
            "selected_action_name": self.selected_action_name,
            "selected_action_data": _json_safe(self.selected_action_data),
            "anchor": self.anchor.to_dict(),
            "effects": self.effects.to_dict(),
            "frame_before": _json_safe(self.frame_before),
            "frame_after": _json_safe(self.frame_after),
            "frame_before_sha256": grid_sha256(self.frame_before),
            "frame_after_sha256": grid_sha256(self.frame_after),
            "game_state_before": self.game_state_before,
            "game_state_after": self.game_state_after,
            "levels_completed_before": self.levels_completed_before,
            "levels_completed_after": self.levels_completed_after,
            "trace_digest": self.trace_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionTargetTrace":
        return cls(
            game_id=str(payload["game_id"]),
            source_split=str(payload["source_split"]),
            policy_seed=int(payload["policy_seed"]),
            reset_index=int(payload["reset_index"]),
            step_index=int(payload["step_index"]),
            collection_phase=str(payload.get("collection_phase", "base")),
            available_action_names=tuple(
                str(item) for item in payload["available_action_names"]
            ),
            selected_action_name=str(payload["selected_action_name"]),
            selected_action_data=dict(payload.get("selected_action_data", {})),
            anchor=ActionTargetAnchor.from_dict(payload["anchor"]),
            effects=ObservedActionTargetEffects.from_dict(payload["effects"]),
            frame_before=payload["frame_before"],
            frame_after=payload["frame_after"],
            game_state_before=str(payload.get("game_state_before", "NOT_FINISHED")),
            game_state_after=str(payload.get("game_state_after", "NOT_FINISHED")),
            levels_completed_before=int(payload.get("levels_completed_before", 0)),
            levels_completed_after=int(payload.get("levels_completed_after", 0)),
            trace_digest=str(payload.get("trace_digest", "")),
            format_version=str(payload["format_version"]),
        )


@dataclass(frozen=True)
class ObjectMatchResult:
    matched: Mapping[int, int]
    created: tuple[int, ...]
    removed: tuple[int, ...]
    ambiguous_before: tuple[int, ...]
    ambiguous_after: tuple[int, ...]


def resolve_action_target(
    observation: GameObservation,
    action_name: str,
    action_data: Mapping[str, Any] | None = None,
) -> ActionTargetAnchor:
    """Resolve a legal primitive action to one pre-action semantic anchor."""
    name = str(action_name).strip().upper()
    data = dict(action_data or {})
    grid = np.asarray(observation.raw_grid, dtype=np.int32)
    background = infer_background(grid)
    player = observation.best_player.position if observation.best_player else None
    direction = "none"
    row: int | None = None
    col: int | None = None
    family = "other"
    kind = "targetless"
    if name in _MOVE_DIRECTIONS:
        family = "move"
        direction, dy, dx = _MOVE_DIRECTIONS[name]
        if player is not None:
            row, col = int(player[0] + dy), int(player[1] + dx)
        kind = "move_destination"
    elif name == "ACTION6" and "x" in data and "y" in data:
        family = "click"
        row = _optional_int(data.get("y"))
        col = _optional_int(data.get("x"))
        kind = "clicked_cell"

    in_bounds = bool(
        row is not None
        and col is not None
        and 0 <= row < grid.shape[0]
        and 0 <= col < grid.shape[1]
    )
    occupied = bool(in_bounds and int(grid[row, col]) != background)
    target = _object_at(observation.objects, row, col) if in_bounds else None
    if family == "click":
        kind = "clicked_object" if target is not None else "clicked_empty"
    affordance = _affordance_for(observation, target)
    relation, relative = _actor_anchor_relation(player, row, col)
    path_status = (
        "open" if in_bounds and not occupied else "blocked" if in_bounds else "unknown"
    )
    return ActionTargetAnchor(
        kind=kind,
        action_family=family,
        requested_direction=direction,
        row=row,
        col=col,
        in_bounds=in_bounds,
        occupied=occupied,
        target_object_id=target.object_id if target is not None else None,
        target_area_bucket=_area_bucket(target.area) if target is not None else "none",
        target_aspect_bucket=(
            _aspect_bucket(target) if target is not None else "none"
        ),
        target_affordance=affordance,
        actor_relation=relation,
        actor_relative_direction=relative,
        path_status=path_status,
    )


def conservative_match_objects(
    before: Sequence[ObjectInfo],
    after: Sequence[ObjectInfo],
    *,
    maximum_distance: float = 8.0,
    minimum_score: float = 0.35,
    ambiguity_margin: float = 0.08,
) -> ObjectMatchResult:
    """Conservatively match components and expose near-tie ambiguity.

    Candidate matches require equal values and a bounded area ratio.  Matching
    is deterministic and one-to-one.  Ambiguous contenders are deliberately
    left unmatched so that they cannot become confident supervision.
    """
    candidates: list[tuple[float, int, int]] = []
    scores_by_before: dict[int, list[tuple[float, int]]] = {}
    scores_by_after: dict[int, list[tuple[float, int]]] = {}
    for left in before:
        for right in after:
            if int(left.value) != int(right.value):
                continue
            area_ratio = min(left.area, right.area) / max(left.area, right.area, 1)
            if area_ratio < 0.5:
                continue
            distance = math.dist(left.center, right.center)
            if distance > maximum_distance:
                continue
            overlap = _cell_iou(left.cells, right.cells)
            distance_score = max(0.0, 1.0 - distance / maximum_distance)
            score = 0.55 * overlap + 0.25 * distance_score + 0.20 * area_ratio
            if score < minimum_score:
                continue
            candidates.append((score, left.object_id, right.object_id))
            scores_by_before.setdefault(left.object_id, []).append(
                (score, right.object_id)
            )
            scores_by_after.setdefault(right.object_id, []).append(
                (score, left.object_id)
            )

    ambiguous_before = {
        object_id
        for object_id, values in scores_by_before.items()
        if _near_tie(values, ambiguity_margin)
    }
    ambiguous_after = {
        object_id
        for object_id, values in scores_by_after.items()
        if _near_tie(values, ambiguity_margin)
    }
    matched: dict[int, int] = {}
    used_after: set[int] = set()
    for _score, before_id, after_id in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if before_id in ambiguous_before or after_id in ambiguous_after:
            continue
        if before_id in matched or after_id in used_after:
            continue
        matched[before_id] = after_id
        used_after.add(after_id)
    before_ids = {item.object_id for item in before}
    after_ids = {item.object_id for item in after}
    created = tuple(sorted(after_ids - set(matched.values()) - ambiguous_after))
    removed = tuple(sorted(before_ids - set(matched) - ambiguous_before))
    return ObjectMatchResult(
        matched=matched,
        created=created,
        removed=removed,
        ambiguous_before=tuple(sorted(ambiguous_before)),
        ambiguous_after=tuple(sorted(ambiguous_after)),
    )


def observed_action_target_effects(
    transition: TransitionRecord,
    anchor: ActionTargetAnchor,
) -> ObservedActionTargetEffects:
    matches = conservative_match_objects(
        transition.obs_before.objects,
        transition.obs_after.objects,
    )
    before_by_id = {
        item.object_id: item for item in transition.obs_before.objects
    }
    after_by_id = {item.object_id: item for item in transition.obs_after.objects}
    reasons: list[str] = []

    actor_applicable = bool(
        transition.obs_before.best_player is not None
        and transition.obs_after.best_player is not None
    )
    if not actor_applicable:
        reasons.append("actor_not_stably_identified")
    target_applicable = bool(anchor.in_bounds and anchor.kind != "targetless")
    target_id = anchor.target_object_id
    if target_id is not None and target_id in matches.ambiguous_before:
        target_applicable = False
        reasons.append("target_match_ambiguous")

    target_removed = bool(target_id is not None and target_id in matches.removed)
    target_moved = False
    if target_id is not None and target_id in matches.matched:
        after_id = matches.matched[target_id]
        before_obj = before_by_id[target_id]
        after_obj = after_by_id[after_id]
        target_moved = not np.allclose(before_obj.center, after_obj.center)

    created_at_anchor = False
    if target_applicable and anchor.row is not None and anchor.col is not None:
        for after_id in matches.created:
            if (int(anchor.row), int(anchor.col)) in set(after_by_id[after_id].cells):
                created_at_anchor = True
                break

    return ObservedActionTargetEffects(
        labels={
            "actor_displaced": transition.diff.player_displacement is not None,
            "target_created": created_at_anchor,
            "target_removed": target_removed,
            "target_moved": target_moved,
        },
        applicable={
            "actor_displaced": actor_applicable,
            "target_created": target_applicable,
            "target_removed": target_applicable and target_id is not None,
            "target_moved": target_applicable and target_id is not None,
        },
        ambiguity_reasons=tuple(sorted(set(reasons))),
        noop=transition.diff.is_noop,
        level_complete=transition.diff.level_complete,
        game_over=transition.diff.game_over,
    )


def build_action_target_trace(
    *,
    game_id: str,
    source_split: str,
    policy_seed: int,
    reset_index: int,
    step_index: int,
    collection_phase: str,
    available_action_names: Sequence[str],
    selected_action_name: str,
    selected_action_data: Mapping[str, Any] | None,
    frame_before: Any,
    frame_after: Any,
    game_state_before: str,
    game_state_after: str,
    levels_completed_before: int,
    levels_completed_after: int,
) -> ActionTargetTrace:
    before = build_observation(
        frame_before,
        available_actions=available_action_names,
        game_state=game_state_before,
        levels_completed=levels_completed_before,
        infer_players=True,
    )
    anchor = resolve_action_target(
        before, selected_action_name, selected_action_data
    )
    transition = build_transition_record(
        action=selected_action_name,
        action_args=dict(selected_action_data or {}),
        grid_before=frame_before,
        grid_after=frame_after,
        available_actions=available_action_names,
        game_state_before=game_state_before,
        game_state_after=game_state_after,
        levels_completed_before=levels_completed_before,
        levels_completed_after=levels_completed_after,
        infer_players=True,
    )
    effects = observed_action_target_effects(transition, anchor)
    return ActionTargetTrace(
        game_id=game_id,
        source_split=source_split,
        policy_seed=policy_seed,
        reset_index=reset_index,
        step_index=step_index,
        collection_phase=collection_phase,
        available_action_names=tuple(available_action_names),
        selected_action_name=str(selected_action_name),
        selected_action_data=dict(selected_action_data or {}),
        anchor=anchor,
        effects=effects,
        frame_before=frame_before,
        frame_after=frame_after,
        game_state_before=str(game_state_before),
        game_state_after=str(game_state_after),
        levels_completed_before=int(levels_completed_before),
        levels_completed_after=int(levels_completed_after),
    )


def feature_row(
    trace: ActionTargetTrace,
    projection: str,
    *,
    include_action: bool = True,
) -> dict[str, Any]:
    raw = trace.model_features(projection)
    if not include_action:
        raw.pop("selected_action_name", None)
        raw.pop("action_family", None)
        raw.pop("requested_direction", None)
    encoded: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            encoded[key] = int(value)
        elif isinstance(value, (int, float)):
            encoded[key] = value
        else:
            encoded[f"{key}:{value}"] = 1
    return encoded


def validate_model_projection(trace: ActionTargetTrace, projection: str) -> None:
    rendered = _canonical(trace.model_features(projection)).lower()
    forbidden = (
        trace.game_id.lower(),
        grid_sha256(trace.frame_before).lower(),
        '"row"',
        '"col"',
        "frame_before",
        "frame_after",
        "policy_seed",
        "reset_index",
        "step_index",
    )
    for token in forbidden:
        if token and token in rendered:
            raise ValueError(f"forbidden model-input token: {token}")


def iter_effect_rows(
    traces: Sequence[ActionTargetTrace],
) -> Iterable[tuple[ActionTargetTrace, str, int]]:
    for trace in traces:
        for label in EFFECT_LABELS:
            if trace.effects.applicable[label]:
                yield trace, label, int(trace.effects.labels[label])


def _object_at(
    objects: Sequence[ObjectInfo], row: int | None, col: int | None
) -> ObjectInfo | None:
    if row is None or col is None:
        return None
    wanted = (int(row), int(col))
    candidates = [item for item in objects if wanted in set(item.cells)]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.area, item.object_id))


def _affordance_for(
    observation: GameObservation, target: ObjectInfo | None
) -> str:
    if target is None:
        return "none"
    values = [
        str(item.kind.value)
        for item in observation.affordances
        if item.target == target.object_id
    ]
    return sorted(values)[0] if values else "unknown"


def _area_bucket(area: int) -> str:
    if area <= 1:
        return "one"
    if area <= 4:
        return "small"
    if area <= 16:
        return "medium"
    return "large"


def _aspect_bucket(obj: ObjectInfo) -> str:
    r0, c0, r1, c1 = obj.bbox
    height = max(1, r1 - r0 + 1)
    width = max(1, c1 - c0 + 1)
    ratio = width / height
    if ratio > 1.5:
        return "wide"
    if ratio < 2.0 / 3.0:
        return "tall"
    return "square"


def _actor_anchor_relation(
    player: tuple[int, int] | None,
    row: int | None,
    col: int | None,
) -> tuple[str, str]:
    if player is None or row is None or col is None:
        return "unknown", "unknown"
    dy = int(row) - int(player[0])
    dx = int(col) - int(player[1])
    distance = abs(dy) + abs(dx)
    if distance == 0:
        relation = "contact"
    elif distance == 1:
        relation = "adjacent"
    elif distance <= 4:
        relation = "near"
    else:
        relation = "far"
    if abs(dy) >= abs(dx) and dy < 0:
        relative = "north"
    elif abs(dy) >= abs(dx) and dy > 0:
        relative = "south"
    elif dx < 0:
        relative = "west"
    elif dx > 0:
        relative = "east"
    else:
        relative = "same"
    return relation, relative


def _coarse_relation(value: str) -> str:
    if value in {"contact", "adjacent"}:
        return "touching"
    if value in {"near", "far"}:
        return value
    return "unknown"


def _cell_iou(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> float:
    a = set(left)
    b = set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _near_tie(values: Sequence[tuple[float, int]], margin: float) -> bool:
    if len(values) < 2:
        return False
    ordered = sorted(values, reverse=True)
    return ordered[0][0] - ordered[1][0] < margin


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ActionTargetAnchor",
    "ActionTargetTrace",
    "EFFECT_LABELS",
    "ObjectMatchResult",
    "ObservedActionTargetEffects",
    "PROJECTION_LADDER",
    "TRACE_FORMAT_VERSION",
    "build_action_target_trace",
    "conservative_match_objects",
    "feature_row",
    "grid_sha256",
    "infer_background",
    "observed_action_target_effects",
    "resolve_action_target",
    "validate_model_projection",
]
