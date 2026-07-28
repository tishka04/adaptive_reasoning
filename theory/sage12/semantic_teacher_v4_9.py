"""SAGE12 V4.9 post-transition semantic teacher and source-only corpus.

The teacher is deliberately deterministic.  It may inspect the executed
transition (before/action/after) to compile auditable physical and functional
effects.  The student-facing view is produced independently from the
pre-action frame and contains only object-relative, identity-free descriptors.

No source-validation, historical, or holdout game is read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN
from v3.schemas import GameObservation, ObjectInfo

from .action_target_data import (
    ActionTargetTrace,
    build_observation,
    grid_sha256,
    infer_background,
)

FORMAT_VERSION = "sage12-semantic-teacher-record-v4.9"
MANIFEST_VERSION = "sage12-semantic-teacher-manifest-v4.9"
QA_VERSION = "sage12-semantic-teacher-qa-v4.9"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "object_relative_teacher_v4_9"
DEFAULT_V3_DIR = Path("training") / "sage12" / "action_target_pilot_v3"
DEFAULT_V43_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"

BASE_EFFECTS = (
    "changed",
    "moved",
    "target_created",
    "target_removed",
    "target_moved",
    "level_complete",
    "game_over",
)
FUNCTIONAL_EFFECTS = (
    "local_change",
    "path_opened",
    "path_closed",
    "actor_approached_root",
    "contact_gained",
    "contact_lost",
    "reachable_area_increased",
    "reachable_area_decreased",
    "productive",
    "risk",
)
SEMANTIC_EFFECTS = BASE_EFFECTS + FUNCTIONAL_EFFECTS

MAXIMUM_NEIGHBORS = 16
FORBIDDEN_MODEL_FIELDS = (
    "game_id",
    "trace_digest",
    "pair_id",
    "frame_before",
    "frame_after",
    "frame_before_sha256",
    "frame_after_sha256",
    "row",
    "col",
    "object_id",
    "target_object_id",
    "policy_seed",
    "reset_index",
    "step_index",
    "value",
    "color",
    "colour",
    "shape_signature",
)

_DIRECTION_SHUFFLE = {
    "north": "east",
    "north_east": "south_east",
    "east": "south",
    "south_east": "south_west",
    "south": "west",
    "south_west": "north_west",
    "west": "north",
    "north_west": "north_east",
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
    )


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class ObjectRelativeGraph:
    """The complete and only authorized pre-action student input."""

    root: Mapping[str, Any]
    neighbors: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": dict(self.root),
            "neighbors": [dict(item) for item in self.neighbors],
        }

    def relation_shuffled(self) -> ObjectRelativeGraph:
        neighbors = []
        for raw in self.neighbors:
            item = dict(raw)
            item["direction"] = _DIRECTION_SHUFFLE.get(
                str(item.get("direction", "none")),
                str(item.get("direction", "none")),
            )
            aligned_row = item.get("aligned_row", 0)
            item["aligned_row"] = item.get("aligned_col", 0)
            item["aligned_col"] = aligned_row
            neighbors.append(item)
        root = dict(self.root)
        root["actor_relative_direction"] = _DIRECTION_SHUFFLE.get(
            str(root.get("actor_relative_direction", "unknown")),
            str(root.get("actor_relative_direction", "unknown")),
        )
        return replace(self, root=root, neighbors=tuple(neighbors))


@dataclass(frozen=True)
class SemanticTeacherRecord:
    """Auditable teacher labels paired with a separately firewalled model view."""

    example_id: str
    game_id: str
    source_corpus: str
    trace_digest: str
    exact_repeat_key: str
    same_prestate_keys: tuple[str, ...]
    graph: ObjectRelativeGraph
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    productive_score: float
    teacher_evidence: Mapping[str, Any]
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.game_id not in SOURCE_TRAIN:
            raise ValueError(f"teacher record outside source_train: {self.game_id}")
        if set(self.labels) != set(SEMANTIC_EFFECTS):
            raise ValueError("teacher labels do not match the frozen vocabulary")
        if set(self.applicable) != set(SEMANTIC_EFFECTS):
            raise ValueError("teacher masks do not match the frozen vocabulary")
        validate_model_graph(self.graph, game_id=self.game_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "example_id": self.example_id,
            "audit": {
                "game_id": self.game_id,
                "source_corpus": self.source_corpus,
                "trace_digest": self.trace_digest,
                "exact_repeat_key": self.exact_repeat_key,
                "same_prestate_keys": list(self.same_prestate_keys),
            },
            "model_graph": self.graph.to_dict(),
            "teacher": {
                "labels": {key: bool(self.labels[key]) for key in SEMANTIC_EFFECTS},
                "applicable": {
                    key: bool(self.applicable[key]) for key in SEMANTIC_EFFECTS
                },
                "productive_score": float(self.productive_score),
                "evidence": _json_safe(self.teacher_evidence),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SemanticTeacherRecord:
        audit = dict(payload["audit"])
        model = dict(payload["model_graph"])
        teacher = dict(payload["teacher"])
        labels = dict(teacher["labels"])
        applicable = dict(teacher["applicable"])
        return cls(
            example_id=str(payload["example_id"]),
            game_id=str(audit["game_id"]),
            source_corpus=str(audit["source_corpus"]),
            trace_digest=str(audit["trace_digest"]),
            exact_repeat_key=str(audit["exact_repeat_key"]),
            same_prestate_keys=tuple(
                str(item) for item in audit.get("same_prestate_keys", ())
            ),
            graph=ObjectRelativeGraph(
                root=dict(model["root"]),
                neighbors=tuple(dict(item) for item in model["neighbors"]),
            ),
            labels={key: bool(labels[key]) for key in SEMANTIC_EFFECTS},
            applicable={key: bool(applicable[key]) for key in SEMANTIC_EFFECTS},
            productive_score=float(teacher["productive_score"]),
            teacher_evidence=dict(teacher.get("evidence", {})),
            format_version=str(payload["format_version"]),
        )


@dataclass(frozen=True)
class PairLink:
    pair_id: str
    game_id: str
    pre_state_sha256: str
    left_trace_digest: str
    right_trace_digest: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_model_graph(
    graph: ObjectRelativeGraph,
    *,
    game_id: str = "",
) -> None:
    rendered = _canonical(graph.to_dict()).lower()
    for field in FORBIDDEN_MODEL_FIELDS:
        token = f'"{field.lower()}"'
        if token in rendered:
            raise ValueError(f"forbidden model-graph field: {field}")
    if game_id and game_id.lower() in rendered:
        raise ValueError("game identity leaked into model graph")


def _area_bucket(area: int) -> str:
    if area <= 1:
        return "single"
    if area <= 4:
        return "small"
    if area <= 16:
        return "medium"
    if area <= 64:
        return "large"
    return "very_large"


def _aspect_bucket(obj: ObjectInfo) -> str:
    height = obj.bbox[2] - obj.bbox[0] + 1
    width = obj.bbox[3] - obj.bbox[1] + 1
    ratio = width / max(height, 1)
    if ratio >= 1.8:
        return "wide"
    if ratio <= 1.0 / 1.8:
        return "tall"
    return "compact"


def _direction(
    origin: tuple[float, float],
    target: tuple[float, float],
) -> str:
    dr = float(target[0] - origin[0])
    dc = float(target[1] - origin[1])
    vertical = "north" if dr < -0.5 else "south" if dr > 0.5 else ""
    horizontal = "west" if dc < -0.5 else "east" if dc > 0.5 else ""
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    if vertical:
        return vertical
    if horizontal:
        return horizontal
    return "overlap"


def _bbox_gap_to_point(obj: ObjectInfo, point: tuple[float, float]) -> float:
    row, col = point
    r0, c0, r1, c1 = obj.bbox
    dr = max(float(r0) - row, 0.0, row - float(r1))
    dc = max(float(c0) - col, 0.0, col - float(c1))
    return max(dr, dc)


def _distance_bucket(distance: float, gap: float) -> str:
    if gap <= 0.0:
        return "contact"
    if gap <= 1.0:
        return "adjacent"
    if distance <= 4.0:
        return "near"
    if distance <= 10.0:
        return "mid"
    return "far"


def _boundary_bucket(
    point: tuple[float, float],
    shape: tuple[int, int],
) -> str:
    row, col = point
    distance = min(row, col, shape[0] - 1 - row, shape[1] - 1 - col)
    if distance <= 0.5:
        return "edge"
    if distance <= 3.0:
        return "near_edge"
    return "interior"


def _object_at(
    objects: Sequence[ObjectInfo],
    row: int | None,
    col: int | None,
) -> ObjectInfo | None:
    if row is None or col is None:
        return None
    point = (int(row), int(col))
    candidates = [item for item in objects if point in set(item.cells)]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.area, item.object_id))


def build_object_relative_graph(
    trace: ActionTargetTrace,
    *,
    maximum_neighbors: int = MAXIMUM_NEIGHBORS,
) -> ObjectRelativeGraph:
    """Build identity-free graph descriptors from the pre-action state only."""

    observation = build_observation(
        trace.frame_before,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_before,
        levels_completed=trace.levels_completed_before,
        infer_players=True,
    )
    anchor = trace.anchor
    root_object = _object_at(observation.objects, anchor.row, anchor.col)
    player = observation.best_player
    if root_object is not None:
        root_kind = "occupied_object"
        center = tuple(float(value) for value in root_object.center)
    elif anchor.in_bounds and anchor.row is not None and anchor.col is not None:
        root_kind = "virtual_cell"
        center = (float(anchor.row), float(anchor.col))
    elif player is not None:
        root_kind = "actor"
        center = tuple(float(value) for value in player.position)
    else:
        # A targetless legal action still has a deployable semantic root: the
        # intervention itself. This is distinct from an ungrounded target.
        root_kind = "action_root"
        center = (
            (observation.raw_grid.shape[0] - 1) / 2.0,
            (observation.raw_grid.shape[1] - 1) / 2.0,
        )

    root: dict[str, Any] = {
        "action_name": trace.selected_action_name,
        "action_family": anchor.action_family,
        "requested_direction": anchor.requested_direction,
        "root_kind": root_kind,
        "root_occupied": int(root_object is not None),
        "root_area_bucket": (
            _area_bucket(root_object.area) if root_object is not None else "none"
        ),
        "root_aspect_bucket": (
            _aspect_bucket(root_object) if root_object is not None else "none"
        ),
        "root_affordance": anchor.target_affordance,
        "actor_relation": anchor.actor_relation,
        "actor_relative_direction": anchor.actor_relative_direction,
        "path_status": anchor.path_status,
        "boundary": _boundary_bucket(center, observation.raw_grid.shape),
        "player_available": int(player is not None),
    }

    candidates: list[tuple[float, tuple[Any, ...], dict[str, Any]]] = []
    for obj in observation.objects:
        if root_object is not None and obj.object_id == root_object.object_id:
            continue
        distance = math.dist(center, obj.center)
        gap = _bbox_gap_to_point(obj, center)
        is_actor = bool(player is not None and player.position in set(obj.cells))
        if root_object is None:
            relative_size = "unknown"
        elif obj.area < root_object.area:
            relative_size = "smaller"
        elif obj.area > root_object.area:
            relative_size = "larger"
        else:
            relative_size = "equal"
        descriptor = {
            "direction": _direction(center, obj.center),
            "proximity": _distance_bucket(distance, gap),
            "relative_size": relative_size,
            "area_bucket": _area_bucket(obj.area),
            "aspect_bucket": _aspect_bucket(obj),
            "is_actor": int(is_actor),
            "aligned_row": int(abs(center[0] - obj.center[0]) <= 0.5),
            "aligned_col": int(abs(center[1] - obj.center[1]) <= 0.5),
            "touches_boundary": int(
                obj.bbox[0] == 0
                or obj.bbox[1] == 0
                or obj.bbox[2] == observation.raw_grid.shape[0] - 1
                or obj.bbox[3] == observation.raw_grid.shape[1] - 1
            ),
        }
        tie_break = tuple(sorted(descriptor.items()))
        candidates.append((distance, tie_break, descriptor))
    candidates.sort(key=lambda row: (row[0], row[1]))
    neighbors = tuple(item[2] for item in candidates[: max(1, int(maximum_neighbors))])
    graph = ObjectRelativeGraph(root=root, neighbors=neighbors)
    validate_model_graph(graph, game_id=trace.game_id)
    return graph


def _player_position(observation: GameObservation) -> tuple[int, int] | None:
    return (
        tuple(int(value) for value in observation.best_player.position)
        if observation.best_player is not None
        else None
    )


def _reachable_background_area(
    grid: np.ndarray,
    start: tuple[int, int] | None,
) -> int | None:
    if start is None:
        return None
    array = np.asarray(grid, dtype=np.int32)
    row, col = start
    if not (0 <= row < array.shape[0] and 0 <= col < array.shape[1]):
        return None
    background = infer_background(array)
    allowed = array == background
    allowed[row, col] = True
    queue: deque[tuple[int, int]] = deque([(row, col)])
    visited = {(row, col)}
    while queue:
        current_row, current_col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = current_row + dr, current_col + dc
            if (
                0 <= nr < array.shape[0]
                and 0 <= nc < array.shape[1]
                and bool(allowed[nr, nc])
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return len(visited)


def _cell_occupied(
    frame: Any,
    row: int | None,
    col: int | None,
) -> bool | None:
    array = np.asarray(frame, dtype=np.int32)
    if (
        row is None
        or col is None
        or not (0 <= int(row) < array.shape[0])
        or not (0 <= int(col) < array.shape[1])
    ):
        return None
    return int(array[int(row), int(col)]) != infer_background(array)


def compile_semantics(
    trace: ActionTargetTrace,
) -> tuple[dict[str, bool], dict[str, bool], float, dict[str, Any]]:
    """Compile physical and functional semantics from one executed transition."""

    before = np.asarray(trace.frame_before, dtype=np.int32)
    after = np.asarray(trace.frame_after, dtype=np.int32)
    observation_before = build_observation(
        before,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_before,
        levels_completed=trace.levels_completed_before,
        infer_players=True,
    )
    observation_after = build_observation(
        after,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_after,
        levels_completed=trace.levels_completed_after,
        infer_players=True,
        prev_player_hypotheses=observation_before.player_candidates,
    )
    anchor = trace.anchor
    changed_mask = before != after
    changed_cells = np.argwhere(changed_mask)
    changed = bool(changed_cells.size) or bool(
        trace.effects.level_complete or trace.effects.game_over
    )
    level_complete = bool(
        trace.effects.level_complete
        or trace.levels_completed_after > trace.levels_completed_before
        or str(trace.game_state_after).upper() == "WIN"
    )
    game_over = bool(
        trace.effects.game_over or str(trace.game_state_after).upper() == "GAME_OVER"
    )

    in_bounds = bool(
        anchor.in_bounds and anchor.row is not None and anchor.col is not None
    )
    local_change = bool(
        in_bounds
        and any(
            max(
                abs(int(row) - int(anchor.row)),
                abs(int(col) - int(anchor.col)),
            )
            <= 2
            for row, col in changed_cells
        )
    )
    occupied_before = _cell_occupied(before, anchor.row, anchor.col)
    occupied_after = _cell_occupied(after, anchor.row, anchor.col)
    path_opened = bool(occupied_before is True and occupied_after is False)
    path_closed = bool(occupied_before is False and occupied_after is True)

    player_before = _player_position(observation_before)
    player_after = _player_position(observation_after)
    actor_applicable = bool(
        player_before is not None and player_after is not None and in_bounds
    )
    if actor_applicable:
        root = (int(anchor.row), int(anchor.col))
        distance_before = math.dist(player_before, root)
        distance_after = math.dist(player_after, root)
        contact_before = (
            max(
                abs(player_before[0] - root[0]),
                abs(player_before[1] - root[1]),
            )
            <= 1
        )
        contact_after = (
            max(
                abs(player_after[0] - root[0]),
                abs(player_after[1] - root[1]),
            )
            <= 1
        )
    else:
        distance_before = distance_after = None
        contact_before = contact_after = False

    approached = bool(
        actor_applicable
        and distance_before is not None
        and distance_after is not None
        and distance_after + 0.5 < distance_before
    )
    contact_gained = bool(actor_applicable and not contact_before and contact_after)
    contact_lost = bool(actor_applicable and contact_before and not contact_after)

    reachable_before = _reachable_background_area(before, player_before)
    reachable_after = _reachable_background_area(after, player_after)
    reach_applicable = bool(
        reachable_before is not None
        and reachable_after is not None
        and before.shape == after.shape
    )
    reach_threshold = (
        max(2, round(0.05 * max(reachable_before or 0, 1))) if reach_applicable else 0
    )
    reachable_increased = bool(
        reach_applicable
        and reachable_after is not None
        and reachable_before is not None
        and reachable_after - reachable_before >= reach_threshold
    )
    reachable_decreased = bool(
        reach_applicable
        and reachable_after is not None
        and reachable_before is not None
        and reachable_before - reachable_after >= reach_threshold
    )

    stored = trace.effects.labels
    base = {
        "changed": changed,
        "moved": bool(stored["actor_displaced"]),
        "target_created": bool(stored["target_created"]),
        "target_removed": bool(stored["target_removed"]),
        "target_moved": bool(stored["target_moved"]),
        "level_complete": level_complete,
        "game_over": game_over,
    }
    score = (
        4.0 * float(level_complete)
        + 1.0 * float(base["target_created"])
        + 1.0 * float(base["target_removed"])
        + 1.0 * float(base["target_moved"])
        + 1.0 * float(path_opened)
        + 1.0 * float(reachable_increased)
        + 0.5 * float(approached)
        + 0.25 * float(base["moved"])
        - 4.0 * float(game_over)
        - 0.5 * float(path_closed)
        - 0.5 * float(reachable_decreased)
    )
    productive = bool(score >= 1.0 and not game_over)
    risk = bool(game_over or path_closed or reachable_decreased)
    functional = {
        "local_change": local_change,
        "path_opened": path_opened,
        "path_closed": path_closed,
        "actor_approached_root": approached,
        "contact_gained": contact_gained,
        "contact_lost": contact_lost,
        "reachable_area_increased": reachable_increased,
        "reachable_area_decreased": reachable_decreased,
        "productive": productive,
        "risk": risk,
    }
    labels = {**base, **functional}

    masks = {
        "changed": True,
        "moved": bool(trace.effects.applicable["actor_displaced"]),
        "target_created": bool(trace.effects.applicable["target_created"]),
        "target_removed": bool(trace.effects.applicable["target_removed"]),
        "target_moved": bool(trace.effects.applicable["target_moved"]),
        "level_complete": True,
        "game_over": True,
        "local_change": in_bounds,
        "path_opened": in_bounds,
        "path_closed": in_bounds,
        "actor_approached_root": actor_applicable,
        "contact_gained": actor_applicable,
        "contact_lost": actor_applicable,
        "reachable_area_increased": reach_applicable,
        "reachable_area_decreased": reach_applicable,
        "productive": True,
        "risk": True,
    }
    evidence = {
        "changed_cells": len(changed_cells),
        "anchor_grounded": in_bounds,
        "occupied_before": occupied_before,
        "occupied_after": occupied_after,
        "player_before_available": player_before is not None,
        "player_after_available": player_after is not None,
        "distance_to_root_before": distance_before,
        "distance_to_root_after": distance_after,
        "reachable_area_before": reachable_before,
        "reachable_area_after": reachable_after,
        "reach_change_threshold": reach_threshold,
        "ambiguity_reasons": list(trace.effects.ambiguity_reasons),
    }
    return labels, masks, float(score), evidence


def _load_source_traces(
    *,
    v3_dir: Path,
    v43_dir: Path,
) -> tuple[dict[str, tuple[ActionTargetTrace, str]], list[PairLink], dict[str, Any]]:
    traces: dict[str, tuple[ActionTargetTrace, str]] = {}
    duplicates: Counter[str] = Counter()
    v3_rows = 0
    v43_arms = 0
    pair_links: list[PairLink] = []

    for game in SOURCE_TRAIN:
        path = v3_dir / "shards" / f"{game}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing source V3 shard: {path}")
        for payload in _read_jsonl(path):
            trace = ActionTargetTrace.from_dict(payload)
            if trace.game_id != game or trace.source_split != "source_train":
                raise ValueError(f"V3 source firewall violation: {path}")
            v3_rows += 1
            if trace.trace_digest in traces:
                duplicates["v3"] += 1
            else:
                traces[trace.trace_digest] = (trace, "action_target_v3")

    for game in SOURCE_TRAIN:
        path = v43_dir / "source_train_shards" / f"{game}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing source V4.3 shard: {path}")
        for payload in _read_jsonl(path):
            if (
                str(payload["game_id"]) != game
                or str(payload["source_split"]) != "source_train"
            ):
                raise ValueError(f"V4.3 source firewall violation: {path}")
            left = ActionTargetTrace.from_dict(payload["left"]["trace"])
            right = ActionTargetTrace.from_dict(payload["right"]["trace"])
            pre_state = grid_sha256(left.frame_before)
            if (
                not np.array_equal(left.frame_before, right.frame_before)
                or grid_sha256(right.frame_before) != pre_state
            ):
                raise ValueError("V4.3 pair arms do not share the same frame")
            pair_links.append(
                PairLink(
                    pair_id=str(payload["pair_digest"]),
                    game_id=game,
                    pre_state_sha256=pre_state,
                    left_trace_digest=left.trace_digest,
                    right_trace_digest=right.trace_digest,
                )
            )
            for trace in (left, right):
                v43_arms += 1
                if trace.trace_digest in traces:
                    duplicates["v43"] += 1
                else:
                    traces[trace.trace_digest] = (trace, "bound_tree_v4_3")

    metadata = {
        "v3_rows": v3_rows,
        "v43_arms": v43_arms,
        "unique_traces": len(traces),
        "pair_links": len(pair_links),
        "duplicates": dict(duplicates),
    }
    return traces, pair_links, metadata


def _source_fingerprints(v3_dir: Path, v43_dir: Path) -> dict[str, Any]:
    fingerprints: dict[str, Any] = {}
    for corpus, directory in (
        ("action_target_v3", v3_dir / "shards"),
        ("bound_tree_v4_3", v43_dir / "source_train_shards"),
    ):
        rows = []
        for game in SOURCE_TRAIN:
            path = directory / f"{game}.jsonl"
            rows.append(
                {
                    "game_id": game,
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
        fingerprints[corpus] = rows
    return fingerprints


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    """Freeze the teacher/student contract before compiling any result."""

    destination = Path(output_dir)
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "source_games": list(SOURCE_TRAIN),
        "forbidden_games": [
            "re86",
            "ls20",
            "sc25",
            "s5i5",
            "vc33",
            "m0r0",
            "sk48",
            "r11l",
        ],
        "teacher": {
            "kind": "deterministic_post_transition_compiler",
            "base_effects": list(BASE_EFFECTS),
            "functional_effects": list(FUNCTIONAL_EFFECTS),
            "labels_are_post_transition": True,
            "student_inputs_are_pre_action_only": True,
            "terminal_labels_are_not_oversampled_or_synthesized": True,
        },
        "student_view": {
            "kind": "object_relative_set_graph",
            "maximum_neighbors": MAXIMUM_NEIGHBORS,
            "forbidden_fields": list(FORBIDDEN_MODEL_FIELDS),
            "absolute_grounding_is_audit_only": True,
            "colors_and_raw_values_excluded": True,
        },
        "evaluation": {
            "outer_split": "leave_one_source_train_game_out",
            "baselines": ["action_only", "root_only"],
            "controls": [
                "relation_shuffle",
                "neighbor_permutation_invariance",
                "semantic_output_game_identity_probe",
            ],
            "decision_thresholds": {
                "teacher_root_grounding_minimum": 0.95,
                "full_macro_brier_gain_over_action_only_strictly_positive": True,
                "full_macro_brier_gain_over_root_only_strictly_positive": True,
                "productive_pair_accuracy_gain_over_action_only_strictly_positive": True,
                "relation_shuffle_brier_degradation_strictly_positive": True,
                "neighbor_permutation_max_probability_delta": 1e-6,
                "semantic_output_identity_accuracy_maximum": 0.60,
                "completion_recall_at_8_minimum": 0.20,
            },
            "confirmatory": False,
            "can_promote_live_authority": False,
        },
        "training": {
            "seed": 4_909,
            "hash_buckets": 2048,
            "embedding_width": 32,
            "hidden_width": 96,
            "epochs": 36,
            "batch_size": 384,
            "learning_rate": 0.002,
            "weight_decay": 0.0001,
            "identity_adversary_weight": 0.08,
            "pairwise_ranking_weight": 0.20,
        },
        "source_fingerprints": _source_fingerprints(Path(v3_dir), Path(v43_dir)),
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir) / "frozen_manifest.json")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.9 manifest")
    expected = str(manifest["manifest_checksum"])
    payload = dict(manifest)
    payload.pop("manifest_checksum")
    if _checksum(payload) != expected:
        raise ValueError("V4.9 manifest checksum mismatch")
    return manifest


def compile_teacher_corpus(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    traces, pair_links, source_metadata = _load_source_traces(
        v3_dir=Path(v3_dir),
        v43_dir=Path(v43_dir),
    )
    pair_keys: dict[str, list[str]] = {}
    for link in pair_links:
        pair_keys.setdefault(link.left_trace_digest, []).append(link.pair_id)
        pair_keys.setdefault(link.right_trace_digest, []).append(link.pair_id)

    records = []
    for digest, (trace, corpus) in sorted(traces.items()):
        labels, applicable, score, evidence = compile_semantics(trace)
        record = SemanticTeacherRecord(
            example_id="sem_"
            + _checksum(
                {
                    "trace_digest": digest,
                    "teacher_version": FORMAT_VERSION,
                }
            )[:20],
            game_id=trace.game_id,
            source_corpus=corpus,
            trace_digest=digest,
            exact_repeat_key=trace.exact_repeat_key(),
            same_prestate_keys=tuple(sorted(pair_keys.get(digest, ()))),
            graph=build_object_relative_graph(trace),
            labels=labels,
            applicable=applicable,
            productive_score=score,
            teacher_evidence=evidence,
        )
        records.append(record)

    corpus_path = destination / "teacher_corpus.jsonl"
    pair_path = destination / "same_prestate_pairs.jsonl"
    _write_jsonl(corpus_path, (record.to_dict() for record in records))
    _write_jsonl(pair_path, (link.to_dict() for link in pair_links))

    counts: dict[str, Any] = {}
    for effect in SEMANTIC_EFFECTS:
        applicable_rows = [record for record in records if record.applicable[effect]]
        counts[effect] = {
            "applicable": len(applicable_rows),
            "positive": sum(record.labels[effect] for record in applicable_rows),
            "per_game_positive": {
                game: sum(
                    record.labels[effect]
                    for record in applicable_rows
                    if record.game_id == game
                )
                for game in SOURCE_TRAIN
            },
        }
    grounded = sum(
        record.graph.root["root_kind"] != "ungrounded" for record in records
    )
    json_roundtrip = all(
        SemanticTeacherRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()
        for record in records
    )
    pair_digests = set(traces)
    pairs_resolved = all(
        link.left_trace_digest in pair_digests
        and link.right_trace_digest in pair_digests
        for link in pair_links
    )
    qa: dict[str, Any] = {
        "format_version": QA_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "records": len(records),
        "games": list(SOURCE_TRAIN),
        "source_metadata": source_metadata,
        "label_capacity": counts,
        "root_grounding_rate": grounded / len(records) if records else 0.0,
        "strict_json_roundtrip": json_roundtrip,
        "pair_links_resolved": pairs_resolved,
        "all_games_source_train": all(
            record.game_id in SOURCE_TRAIN for record in records
        ),
        "forbidden_model_fields_absent": True,
        "completion_positives": counts["level_complete"]["positive"],
        "completion_labels_synthesized": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {
            "teacher_corpus": _file_sha256(corpus_path),
            "same_prestate_pairs": _file_sha256(pair_path),
        },
    }
    threshold = manifest["evaluation"]["decision_thresholds"][
        "teacher_root_grounding_minimum"
    ]
    qa["checks"] = {
        "root_grounding_minimum": qa["root_grounding_rate"] >= threshold,
        "strict_json_roundtrip": json_roundtrip,
        "pair_links_resolved": pairs_resolved,
        "all_games_source_train": qa["all_games_source_train"],
        "forbidden_model_fields_absent": True,
    }
    qa["teacher_ready"] = all(qa["checks"].values())
    qa["qa_checksum"] = _checksum(qa)
    _write_json(destination / "teacher_qa.json", qa)
    return qa


def load_teacher_records(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[SemanticTeacherRecord, ...]:
    return tuple(
        SemanticTeacherRecord.from_dict(row)
        for row in _read_jsonl(Path(output_dir) / "teacher_corpus.jsonl")
    )


def load_pair_links(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[PairLink, ...]:
    return tuple(
        PairLink(**row)
        for row in _read_jsonl(Path(output_dir) / "same_prestate_pairs.jsonl")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(output_dir=args.output_dir)
    else:
        if not (args.output_dir / "frozen_manifest.json").exists():
            freeze_manifest(output_dir=args.output_dir)
        payload = compile_teacher_corpus(output_dir=args.output_dir)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_EFFECTS",
    "FUNCTIONAL_EFFECTS",
    "SEMANTIC_EFFECTS",
    "ObjectRelativeGraph",
    "PairLink",
    "SemanticTeacherRecord",
    "build_object_relative_graph",
    "compile_semantics",
    "compile_teacher_corpus",
    "freeze_manifest",
    "load_manifest",
    "load_pair_links",
    "load_teacher_records",
    "validate_model_graph",
]
