"""Causal frontier-recovery diagnostic for AR25 trace distillation.

This script starts from the best human trace, replays it to the last known
safe state at a target level, then runs short local search branches from that
frontier. The key question is intentionally narrow:

    Can the agent find at least one safe state that the human trace did not
    visit after reaching the same frontier?

Example:
    python level7_frontier_recovery.py --game ar25 --target-level 7
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from trace_replay_verifier import (  # noqa: WPS450 - reuse verified env bootstrap
    ENV_DIR,
    PROJECT_ROOT,
    Arcade,
    OperationMode,
    SelectedEpisode,
    _action_enum,
    _available_action_names,
    _grid_summary,
    _grids_equal,
    _load_selected_episode,
    _primary_grid,
    _resolve_full_game_id,
    _state_name,
    _step_action_data,
)
from human_trace.schema import StepRecord


logging.disable(logging.CRITICAL)

DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "level_7_frontier_recovery"
ACTION_NAMES = [f"ACTION{i}" for i in range(1, 8)]


@dataclass
class LevelFrontier:
    """Concrete trace indices that define the local repair problem."""

    target_level: int
    level_start_index: int
    frontier_index: int
    terminal_index: int
    level_start_frame: List[List[int]]
    frontier_frame: List[List[int]]
    terminal_frame: List[List[int]]
    immediate_danger_action: str
    danger_actions: List[str]
    visited_safe_hashes: Set[str] = field(default_factory=set)
    danger_state_hashes: Set[str] = field(default_factory=set)


@dataclass
class BranchNode:
    """One evaluated local future branch."""

    actions: List[str]
    action_data: List[Optional[Dict[str, int]]]
    state: str
    level: int
    grid_hash: str
    score: float
    depth: int
    progress_delta: int = 0
    novel_safe_states: int = 0
    changed_cells_from_frontier: int = 0
    repeat_penalty: float = 0.0
    death_risk: float = 0.0
    danger_prefix_len: int = 0
    reached_next_level: bool = False
    died: bool = False
    available_actions: List[str] = field(default_factory=list)
    novel_hashes: Set[str] = field(default_factory=set)
    path_hashes: List[str] = field(default_factory=list)
    env: Any = field(default=None, repr=False, compare=False)

    def to_report(self) -> Dict[str, Any]:
        return {
            "actions": list(self.actions),
            "action_data": list(self.action_data),
            "state": self.state,
            "level": self.level,
            "grid_hash": self.grid_hash,
            "score": round(float(self.score), 4),
            "depth": self.depth,
            "progress_delta": self.progress_delta,
            "novel_safe_states": self.novel_safe_states,
            "changed_cells_from_frontier": self.changed_cells_from_frontier,
            "repeat_penalty": round(float(self.repeat_penalty), 4),
            "death_risk": round(float(self.death_risk), 4),
            "danger_prefix_len": self.danger_prefix_len,
            "reached_next_level": self.reached_next_level,
            "died": self.died,
            "available_actions": list(self.available_actions),
        }


def _hash_grid(grid: Sequence[Sequence[int]]) -> str:
    arr = np.array(grid, dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _changed_cells(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> int:
    left = np.array(a, dtype=np.int32)
    right = np.array(b, dtype=np.int32)
    if left.shape != right.shape:
        return int(left.size + right.size)
    return int(np.count_nonzero(left != right))


def _histogram_delta(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> int:
    left = np.array(a, dtype=np.int32).ravel()
    right = np.array(b, dtype=np.int32).ravel()
    values = set(int(v) for v in left) | set(int(v) for v in right)
    return int(
        sum(
            abs(int(np.count_nonzero(left == value)) - int(np.count_nonzero(right == value)))
            for value in values
        )
    )


def _available_names_from_raw(raw: Any) -> List[str]:
    return _available_action_names(getattr(raw, "available_actions", []) or [])


def _non_reset_steps(steps: Sequence[StepRecord]) -> List[StepRecord]:
    return [step for step in steps if step.action != "RESET"]


def _danger_prefix_len(actions: Sequence[str], danger_actions: Sequence[str]) -> int:
    count = 0
    for got, danger in zip(actions, danger_actions):
        if got != danger:
            break
        count += 1
    return count


def _connected_components(grid: Sequence[Sequence[int]], limit: int = 12) -> List[Dict[str, int]]:
    arr = np.array(grid, dtype=np.int32)
    seen = np.zeros(arr.shape, dtype=bool)
    components: List[Dict[str, int]] = []
    height, width = arr.shape
    for y in range(height):
        for x in range(width):
            color = int(arr[y, x])
            if color == 0 or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if seen[ny, nx] or int(arr[ny, nx]) != color:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "min_y": min(ys),
                    "min_x": min(xs),
                    "max_y": max(ys),
                    "max_x": max(xs),
                }
            )
    components.sort(key=lambda item: (item["size"], item["color"]), reverse=True)
    return components[:limit]


def _component_signature(component: Dict[str, int]) -> Tuple[int, int, int, int, int, int]:
    return (
        int(component["color"]),
        int(component["size"]),
        int(component["min_y"]),
        int(component["min_x"]),
        int(component["max_y"]),
        int(component["max_x"]),
    )


def _component_delta(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> int:
    before_set = {_component_signature(item) for item in _connected_components(before, limit=256)}
    after_set = {_component_signature(item) for item in _connected_components(after, limit=256)}
    return len(before_set ^ after_set)


def _diff_bbox(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> Optional[Dict[str, Any]]:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return {
            "shape_before": list(left.shape),
            "shape_after": list(right.shape),
        }
    coords = np.argwhere(left != right)
    if coords.size == 0:
        return None
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    return {
        "min_y": int(min_y),
        "min_x": int(min_x),
        "max_y": int(max_y),
        "max_x": int(max_x),
        "height": int(max_y - min_y + 1),
        "width": int(max_x - min_x + 1),
        "changed_cells": int(coords.shape[0]),
    }


def _changed_value_report(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    *,
    limit: int = 24,
) -> Dict[str, Any]:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return {"shape_mismatch": True}
    mask = left != right
    before_values = sorted(int(v) for v in np.unique(left[mask]))
    after_values = sorted(int(v) for v in np.unique(right[mask]))
    transitions: Dict[str, int] = {}
    for old, new in zip(left[mask].tolist(), right[mask].tolist()):
        key = f"{int(old)}->{int(new)}"
        transitions[key] = transitions.get(key, 0) + 1
    top_transitions = dict(
        sorted(transitions.items(), key=lambda item: item[1], reverse=True)[:limit]
    )
    return {
        "before_values": before_values,
        "after_values": after_values,
        "transitions": top_transitions,
    }


def _components_touching_bbox(
    grid: Sequence[Sequence[int]],
    bbox: Optional[Dict[str, Any]],
    *,
    limit: int = 20,
) -> List[Dict[str, int]]:
    if not bbox or "min_y" not in bbox:
        return []
    out: List[Dict[str, int]] = []
    for component in _connected_components(grid, limit=512):
        if component["max_y"] < bbox["min_y"] or component["min_y"] > bbox["max_y"]:
            continue
        if component["max_x"] < bbox["min_x"] or component["min_x"] > bbox["max_x"]:
            continue
        out.append(component)
    out.sort(key=lambda item: (item["size"], item["color"]), reverse=True)
    return out[:limit]


def _distance_to_center(
    grid: Sequence[Sequence[int]],
    bbox: Optional[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    if not bbox or "min_y" not in bbox:
        return None
    arr = np.array(grid, dtype=np.int32)
    height, width = arr.shape if arr.ndim == 2 else (64, 64)
    center_y = (height - 1) * 0.5
    center_x = (width - 1) * 0.5
    bbox_center_y = (float(bbox["min_y"]) + float(bbox["max_y"])) * 0.5
    bbox_center_x = (float(bbox["min_x"]) + float(bbox["max_x"])) * 0.5
    distance = float(np.hypot(bbox_center_y - center_y, bbox_center_x - center_x))
    diagonal = float(np.hypot(max(1, height - 1), max(1, width - 1)))
    return {
        "bbox_center_y": round(bbox_center_y, 3),
        "bbox_center_x": round(bbox_center_x, 3),
        "grid_center_y": round(center_y, 3),
        "grid_center_x": round(center_x, 3),
        "euclidean": round(distance, 3),
        "normalized": round(distance / max(diagonal, 1.0), 4),
    }


def _symmetry_score(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> Dict[str, float]:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return {"vertical": 0.0, "horizontal": 0.0, "rotational": 0.0}
    coords = {
        (int(y), int(x))
        for y, x in np.argwhere(left != right)
    }
    if not coords:
        return {"vertical": 1.0, "horizontal": 1.0, "rotational": 1.0}
    height, width = left.shape

    def overlap(reflected: Set[Tuple[int, int]]) -> float:
        return round(len(coords & reflected) / max(1, len(coords)), 4)

    vertical = {(y, width - 1 - x) for y, x in coords}
    horizontal = {(height - 1 - y, x) for y, x in coords}
    rotational = {(height - 1 - y, width - 1 - x) for y, x in coords}
    return {
        "vertical": overlap(vertical),
        "horizontal": overlap(horizontal),
        "rotational": overlap(rotational),
    }


def _change_mechanism_report(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    bbox = _diff_bbox(before, after)
    before_components = _components_touching_bbox(before, bbox)
    after_components = _components_touching_bbox(after, bbox)
    before_set = {_component_signature(item) for item in before_components}
    after_set = {_component_signature(item) for item in after_components}
    return {
        "changed_values": _changed_value_report(before, after),
        "changed_components_before_after": {
            "before": before_components,
            "after": after_components,
            "removed_count": len(before_set - after_set),
            "added_count": len(after_set - before_set),
        },
        "bounding_box_of_diff": bbox,
        "distance_to_center": _distance_to_center(before, bbox),
        "symmetry_score": _symmetry_score(before, after),
        "changed_cells": _changed_cells(before, after),
        "component_delta": _component_delta(before, after),
        "histogram_delta": _histogram_delta(before, after),
    }


def _compact_change_report(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    report = _change_mechanism_report(before, after)
    return {
        "bounding_box_of_diff": report["bounding_box_of_diff"],
        "changed_values": report["changed_values"],
        "changed_cells": report["changed_cells"],
        "component_delta": report["component_delta"],
        "histogram_delta": report["histogram_delta"],
        "distance_to_center": report["distance_to_center"],
        "symmetry_score": report["symmetry_score"],
    }


def _grid_from_raw(raw: Any) -> List[List[int]]:
    return _primary_grid(raw) if raw is not None else [[0]]


def _grid_from_env(env: Any) -> List[List[int]]:
    return _grid_from_raw(getattr(env, "observation_space", None))


def _clamp_coordinate(value: float) -> int:
    return max(0, min(63, int(round(value))))


def _coordinate_candidates(
    grid: Sequence[Sequence[int]],
    *,
    grid_size: int,
    max_candidates: int,
    seeds: Sequence[Dict[str, int]] = (),
) -> List[Dict[str, int]]:
    """Generate plausible ACTION6 click coordinates from objects + coarse grid."""
    arr = np.array(grid, dtype=np.int32)
    height, width = arr.shape if arr.ndim == 2 else (64, 64)
    scale_x = 63.0 / max(1, width - 1)
    scale_y = 63.0 / max(1, height - 1)

    out: List[Dict[str, int]] = []
    seen: Set[Tuple[int, int]] = set()

    def add(x: float, y: float) -> None:
        key = (_clamp_coordinate(x), _clamp_coordinate(y))
        if key in seen or len(out) >= max_candidates:
            return
        seen.add(key)
        out.append({"x": key[0], "y": key[1]})

    for seed in seeds:
        if "x" in seed and "y" in seed:
            add(float(seed["x"]), float(seed["y"]))

    for component in _connected_components(grid, limit=32):
        center_x = (component["min_x"] + component["max_x"]) * 0.5 * scale_x
        center_y = (component["min_y"] + component["max_y"]) * 0.5 * scale_y
        add(center_x, center_y)

    add(32, 32)
    side = max(2, int(grid_size))
    for row in range(side):
        for col in range(side):
            add((col + 0.5) * 64.0 / side, (row + 0.5) * 64.0 / side)
            if len(out) >= max_candidates:
                return out
    return out


def _same_bbox(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    if not a or not b:
        return False
    keys = ("min_y", "min_x", "max_y", "max_x")
    return all(int(a.get(key, -1)) == int(b.get(key, -2)) for key in keys)


def _add_coord(
    out: List[Dict[str, int]],
    seen: Set[Tuple[int, int]],
    *,
    x: float,
    y: float,
    max_candidates: int,
) -> None:
    key = (_clamp_coordinate(x), _clamp_coordinate(y))
    if key in seen or len(out) >= max_candidates:
        return
    seen.add(key)
    out.append({"x": key[0], "y": key[1]})


def _bbox_targeted_action6_candidates(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    *,
    max_candidates: int,
) -> List[Dict[str, int]]:
    """Generate click targets from the bbox and symmetries of an observed diff."""
    bbox = _diff_bbox(before, after)
    if not bbox or "min_x" not in bbox:
        return []
    arr = np.array(before, dtype=np.int32)
    height, width = arr.shape if arr.ndim == 2 else (64, 64)
    min_x = float(bbox["min_x"])
    max_x = float(bbox["max_x"])
    min_y = float(bbox["min_y"])
    max_y = float(bbox["max_y"])
    mid_x = (min_x + max_x) * 0.5
    mid_y = (min_y + max_y) * 0.5

    out: List[Dict[str, int]] = []
    seen: Set[Tuple[int, int]] = set()

    anchor_points = [
        (min_x, min_y),
        (max_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (mid_x, min_y),
        (mid_x, max_y),
        (min_x, mid_y),
        (max_x, mid_y),
        (mid_x, mid_y),
    ]
    for x, y in anchor_points:
        _add_coord(out, seen, x=x, y=y, max_candidates=max_candidates)

    changed = np.argwhere(np.array(before, dtype=np.int32) != np.array(after, dtype=np.int32))
    if changed.size:
        sample_count = max(1, min(24, max_candidates - len(out)))
        if len(changed) <= sample_count:
            sample_indices = range(len(changed))
        else:
            sample_indices = np.linspace(0, len(changed) - 1, sample_count, dtype=int)
        for idx in sample_indices:
            y, x = changed[int(idx)]
            _add_coord(out, seen, x=float(x), y=float(y), max_candidates=max_candidates)

    base_points = list(out)
    for point in base_points:
        x = float(point["x"])
        y = float(point["y"])
        mirrors = [
            (float(width - 1) - x, y),
            (x, float(height - 1) - y),
            (float(width - 1) - x, float(height - 1) - y),
        ]
        for mx, my in mirrors:
            _add_coord(out, seen, x=mx, y=my, max_candidates=max_candidates)
            if len(out) >= max_candidates:
                return out
    return out


def _compare_states(
    label: str,
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    return {
        "label": label,
        "before": _grid_summary(before),
        "after": _grid_summary(after),
        "changed_cells": _changed_cells(before, after),
        "before_components": _connected_components(before),
        "after_components": _connected_components(after),
    }


def find_level_frontier(
    selection: SelectedEpisode,
    *,
    target_level: int,
    danger_window: int,
) -> LevelFrontier:
    """Find the last safe trace state before GAME_OVER at target_level."""
    steps = list(selection.steps)
    if not steps:
        raise ValueError("Selected episode has no steps")

    previous_level = 0
    level_start_index: Optional[int] = None
    level_start_frame: Optional[List[List[int]]] = None
    for idx, step in enumerate(steps):
        level_after = int(step.levels_completed_after)
        if previous_level < target_level <= level_after:
            next_step = steps[idx + 1] if idx + 1 < len(steps) else None
            level_start_index = idx + 1 if next_step is not None else idx
            level_start_frame = (
                next_step.frame_before if next_step is not None else step.frame_after
            )
            break
        previous_level = level_after

    if level_start_index is None or level_start_frame is None:
        raise ValueError(f"Trace never reaches levels_completed={target_level}")

    terminal_index: Optional[int] = None
    for idx in range(len(steps) - 1, -1, -1):
        step = steps[idx]
        if (
            step.action != "RESET"
            and step.game_state_after == "GAME_OVER"
            and int(step.levels_completed_after) == int(target_level)
        ):
            terminal_index = idx
            break

    if terminal_index is None:
        raise ValueError(
            f"No GAME_OVER transition found after target level {target_level}"
        )

    terminal_step = steps[terminal_index]
    suffix_start = max(level_start_index, terminal_index - max(1, danger_window) + 1)
    danger_steps = [step for step in steps[suffix_start : terminal_index + 1] if step.action != "RESET"]
    danger_actions = [step.action for step in danger_steps]

    visited_safe_hashes: Set[str] = set()
    for step in steps[level_start_index : terminal_index + 1]:
        if step.action == "RESET":
            continue
        visited_safe_hashes.add(_hash_grid(step.frame_before))
        if step.game_state_after != "GAME_OVER":
            visited_safe_hashes.add(_hash_grid(step.frame_after))

    return LevelFrontier(
        target_level=int(target_level),
        level_start_index=level_start_index,
        frontier_index=terminal_index,
        terminal_index=terminal_index,
        level_start_frame=level_start_frame,
        frontier_frame=terminal_step.frame_before,
        terminal_frame=terminal_step.frame_after,
        immediate_danger_action=terminal_step.action,
        danger_actions=danger_actions,
        visited_safe_hashes=visited_safe_hashes,
        danger_state_hashes={_hash_grid(terminal_step.frame_after)},
    )


def _replay_prefix(
    arc: Arcade,
    full_game_id: str,
    steps: Sequence[StepRecord],
    *,
    stop_before_index: int,
    expected_frame: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[Any, Any]:
    env = arc.make(full_game_id)
    if env is None:
        raise ValueError(f"Could not make environment for {full_game_id}")

    raw = getattr(env, "observation_space", None)
    for idx, step in enumerate(steps[:stop_before_index]):
        try:
            raw = env.step(
                _action_enum(step.action),
                data=_step_action_data(full_game_id, step),
            )
        except TypeError:
            raw = env.step(_action_enum(step.action))
        if raw is None:
            raise RuntimeError(f"Replay returned None at trace index {idx}")
        if _state_name(getattr(raw, "state", "UNKNOWN")) == "GAME_OVER":
            raise RuntimeError(f"Replay hit GAME_OVER before frontier at trace index {idx}")

    if expected_frame is not None and raw is not None:
        actual = _primary_grid(raw)
        if not _grids_equal(actual, expected_frame):
            raise RuntimeError(
                "Replay prefix did not align with expected frontier frame: "
                f"expected={_grid_summary(expected_frame)} actual={_grid_summary(actual)}"
            )
    return env, raw


def _click_targets(
    steps: Sequence[StepRecord],
    *,
    around_index: int,
    window: int,
    limit: int,
) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    seen: Set[Tuple[int, int]] = set()
    start = max(0, around_index - window)
    end = min(len(steps), around_index + window + 1)
    for step in steps[start:end]:
        data = step.action_args or {}
        if "x" not in data or "y" not in data:
            continue
        key = (int(data["x"]), int(data["y"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"x": key[0], "y": key[1]})
        if len(out) >= limit:
            break
    return out


def _action_variants(
    action: str,
    *,
    full_game_id: str,
    click_targets: Sequence[Dict[str, int]],
) -> List[Optional[Dict[str, int]]]:
    if action != "ACTION6":
        return [None]
    if click_targets:
        return [dict(target) for target in click_targets]
    return [{"x": 32, "y": 32}]


def _step_branch(
    env: Any,
    *,
    full_game_id: str,
    action: str,
    action_data: Optional[Dict[str, int]],
) -> Any:
    data: Dict[str, Any] = {"game_id": full_game_id}
    if action_data:
        data.update(action_data)
    return env.step(_action_enum(action), data=data)


def _make_child(
    parent: BranchNode,
    *,
    raw: Any,
    env: Any,
    action: str,
    action_data: Optional[Dict[str, int]],
    frontier: LevelFrontier,
    frontier_grid: Sequence[Sequence[int]],
) -> BranchNode:
    if raw is None:
        state = "ERROR"
        level = parent.level
        grid = frontier_grid
        available: List[str] = []
    else:
        state = _state_name(getattr(raw, "state", "UNKNOWN"))
        level = int(getattr(raw, "levels_completed", 0) or 0)
        grid = _primary_grid(raw)
        available = _available_names_from_raw(raw)

    grid_hash = _hash_grid(grid)
    died = raw is None or state == "GAME_OVER"
    danger_state_hit = grid_hash in frontier.danger_state_hashes
    danger_prefix = _danger_prefix_len(parent.actions + [action], frontier.danger_actions)
    repeated = grid_hash in set(parent.path_hashes)
    novel_hashes = set(parent.novel_hashes)
    if not died and grid_hash not in frontier.visited_safe_hashes:
        novel_hashes.add(grid_hash)

    progress_delta = max(0, level - frontier.target_level)
    reached_next_level = level > frontier.target_level or state == "WIN"
    changed = _changed_cells(frontier_grid, grid)
    repeat_penalty = parent.repeat_penalty + (1.0 if repeated else 0.0)
    death_risk = 0.0
    if died:
        death_risk += 1.0
    if danger_state_hit:
        death_risk += 0.7
    if danger_prefix:
        death_risk += 0.15 * danger_prefix
    if (parent.actions + [action])[:1] == [frontier.immediate_danger_action]:
        death_risk += 0.5

    score = (
        12.0 * progress_delta
        + 2.4 * len(novel_hashes)
        + 0.015 * changed
        - 9.0 * death_risk
        - 1.25 * repeat_penalty
        - 0.05 * (len(parent.actions) + 1)
    )

    return BranchNode(
        actions=parent.actions + [action],
        action_data=parent.action_data + [dict(action_data) if action_data else None],
        state=state,
        level=level,
        grid_hash=grid_hash,
        score=score,
        depth=len(parent.actions) + 1,
        progress_delta=progress_delta,
        novel_safe_states=len(novel_hashes),
        changed_cells_from_frontier=changed,
        repeat_penalty=repeat_penalty,
        death_risk=death_risk,
        danger_prefix_len=danger_prefix,
        reached_next_level=reached_next_level,
        died=died,
        available_actions=available,
        novel_hashes=novel_hashes,
        path_hashes=parent.path_hashes + [grid_hash],
        env=env,
    )


def _summarize_beam(
    *,
    evaluated: Sequence[BranchNode],
    root: BranchNode,
    frontier: LevelFrontier,
) -> Dict[str, Any]:
    safe_nodes = [node for node in evaluated if not node.died]
    best = max(safe_nodes or list(evaluated), key=lambda item: item.score, default=root)
    novel_hashes: Set[str] = set()
    for node in safe_nodes:
        novel_hashes.update(node.novel_hashes)
    reaches_next = any(node.reached_next_level for node in evaluated)
    best_progress = max((node.progress_delta for node in evaluated), default=0)
    avoids_known = (
        not best.died
        and (not best.actions or best.actions[0] != frontier.immediate_danger_action)
        and best.grid_hash not in frontier.danger_state_hashes
    )

    return {
        "evaluated_branches": len(evaluated),
        "reaches_level_8": bool(reaches_next),
        "avoids_known_death_suffix": bool(avoids_known),
        "best_progress_delta": int(best_progress),
        "novel_safe_states": len(novel_hashes),
        "best": best.to_report(),
        "top": [
            node.to_report()
            for node in sorted(
                safe_nodes or list(evaluated),
                key=lambda item: item.score,
                reverse=True,
            )[:8]
        ],
    }


def _frontier_beam_search(
    *,
    base_env: Any,
    base_raw: Any,
    full_game_id: str,
    selection: SelectedEpisode,
    frontier: LevelFrontier,
    horizon: int,
    beam_width: int,
    max_click_targets: int,
) -> Tuple[Dict[str, Any], List[BranchNode]]:
    """Run a short beam from one already-aligned frontier state."""
    frontier_grid = _primary_grid(base_raw)
    root_hash = _hash_grid(frontier_grid)
    root = BranchNode(
        actions=[],
        action_data=[],
        state=_state_name(getattr(base_raw, "state", "UNKNOWN")),
        level=int(getattr(base_raw, "levels_completed", 0) or 0),
        grid_hash=root_hash,
        score=0.0,
        depth=0,
        available_actions=_available_names_from_raw(base_raw),
        path_hashes=[root_hash],
        env=copy.deepcopy(base_env),
    )
    click_targets = _click_targets(
        selection.steps,
        around_index=frontier.frontier_index,
        window=30,
        limit=max_click_targets,
    )

    beam = [root]
    evaluated: List[BranchNode] = []
    for _depth in range(1, max(1, horizon) + 1):
        expanded: List[BranchNode] = []
        for node in beam:
            if node.died or node.reached_next_level:
                expanded.append(node)
                continue
            available = node.available_actions or ACTION_NAMES
            for action in available:
                if action == "RESET":
                    continue
                for data in _action_variants(
                    action,
                    full_game_id=full_game_id,
                    click_targets=click_targets,
                ):
                    branch_env = copy.deepcopy(node.env)
                    raw = _step_branch(
                        branch_env,
                        full_game_id=full_game_id,
                        action=action,
                        action_data=data,
                    )
                    child = _make_child(
                        node,
                        raw=raw,
                        env=branch_env,
                        action=action,
                        action_data=data,
                        frontier=frontier,
                        frontier_grid=frontier_grid,
                    )
                    expanded.append(child)
        expanded.sort(key=lambda item: item.score, reverse=True)
        evaluated.extend(expanded)
        beam = expanded[: max(1, beam_width)]

    report = {
        "horizon": int(horizon),
        "beam_width": int(beam_width),
    }
    report.update(_summarize_beam(evaluated=evaluated, root=root, frontier=frontier))
    return report, evaluated


def run_frontier_beam(
    *,
    base_env: Any,
    base_raw: Any,
    full_game_id: str,
    selection: SelectedEpisode,
    frontier: LevelFrontier,
    horizon: int,
    beam_width: int,
    max_click_targets: int,
) -> Dict[str, Any]:
    report, _nodes = _frontier_beam_search(
        base_env=base_env,
        base_raw=base_raw,
        full_game_id=full_game_id,
        selection=selection,
        frontier=frontier,
        horizon=horizon,
        beam_width=beam_width,
        max_click_targets=max_click_targets,
    )
    return report


def run_suffix_repair(
    *,
    arc: Arcade,
    full_game_id: str,
    selection: SelectedEpisode,
    frontier: LevelFrontier,
    repair_window: int,
    repair_points: int,
    horizon: int,
    beam_width: int,
    max_click_targets: int,
) -> Dict[str, Any]:
    """Try local mutations from several points before the fatal suffix."""
    start = max(frontier.level_start_index, frontier.terminal_index - max(1, repair_window))
    candidates = list(range(start, frontier.terminal_index + 1))
    if repair_points > 0 and len(candidates) > repair_points:
        positions = np.linspace(0, len(candidates) - 1, num=repair_points, dtype=int)
        candidates = [candidates[int(pos)] for pos in positions]

    repairs: List[Dict[str, Any]] = []
    for trace_index in candidates:
        step = selection.steps[trace_index]
        if step.action == "RESET":
            continue
        try:
            env, raw = _replay_prefix(
                arc,
                full_game_id,
                selection.steps,
                stop_before_index=trace_index,
                expected_frame=step.frame_before,
            )
        except RuntimeError as exc:
            repairs.append(
                {
                    "trace_index": trace_index,
                    "trace_action": step.action,
                    "skipped": str(exc),
                }
            )
            continue
        local_frontier = LevelFrontier(
            target_level=frontier.target_level,
            level_start_index=frontier.level_start_index,
            frontier_index=trace_index,
            terminal_index=frontier.terminal_index,
            level_start_frame=frontier.level_start_frame,
            frontier_frame=step.frame_before,
            terminal_frame=frontier.terminal_frame,
            immediate_danger_action=step.action,
            danger_actions=[
                item.action
                for item in selection.steps[trace_index : frontier.terminal_index + 1]
                if item.action != "RESET"
            ],
            visited_safe_hashes=frontier.visited_safe_hashes,
            danger_state_hashes=frontier.danger_state_hashes,
        )
        result = run_frontier_beam(
            base_env=env,
            base_raw=raw,
            full_game_id=full_game_id,
            selection=selection,
            frontier=local_frontier,
            horizon=horizon,
            beam_width=beam_width,
            max_click_targets=max_click_targets,
        )
        result.update(
            {
                "trace_index": trace_index,
                "trace_action": step.action,
                "steps_before_death": frontier.terminal_index - trace_index,
            }
        )
        repairs.append(result)

    ranked = [
        repair
        for repair in repairs
        if isinstance(repair.get("best"), dict)
    ]
    ranked.sort(
        key=lambda item: (
            int(item.get("best_progress_delta", 0)),
            int(item.get("novel_safe_states", 0)),
            float(item["best"].get("score", -999.0)),
        ),
        reverse=True,
    )
    return {
        "repair_window": int(repair_window),
        "repair_points": len(candidates),
        "evaluated_points": len(repairs),
        "best": ranked[0] if ranked else None,
        "points": repairs,
    }


def _select_curriculum_nodes(
    nodes: Sequence[BranchNode],
    frontier: LevelFrontier,
    *,
    limit: int,
) -> List[BranchNode]:
    out: List[BranchNode] = []
    seen: Set[str] = set()
    for node in sorted(
        nodes,
        key=lambda item: (
            int(item.progress_delta),
            int(item.novel_safe_states),
            float(item.score),
        ),
        reverse=True,
    ):
        if node.died or node.grid_hash in frontier.visited_safe_hashes:
            continue
        if node.grid_hash in seen:
            continue
        seen.add(node.grid_hash)
        out.append(node)
        if len(out) >= max(1, limit):
            break
    return out


def run_frontier_curriculum(
    *,
    base_env: Any,
    base_raw: Any,
    full_game_id: str,
    selection: SelectedEpisode,
    frontier: LevelFrontier,
    seed_horizon: int,
    seed_beam_width: int,
    curriculum_states: int,
    curriculum_horizon: int,
    curriculum_beam_width: int,
    max_click_targets: int,
) -> Dict[str, Any]:
    """Re-open the most promising novel safe states as local curricula."""
    seed_report, seed_nodes = _frontier_beam_search(
        base_env=base_env,
        base_raw=base_raw,
        full_game_id=full_game_id,
        selection=selection,
        frontier=frontier,
        horizon=seed_horizon,
        beam_width=seed_beam_width,
        max_click_targets=max_click_targets,
    )
    seed_safe_hashes = {
        node.grid_hash
        for node in seed_nodes
        if not node.died and node.grid_hash not in frontier.danger_state_hashes
    }
    curriculum_roots = _select_curriculum_nodes(
        seed_nodes,
        frontier,
        limit=curriculum_states,
    )

    rows: List[Dict[str, Any]] = []
    curriculum_novel_hashes: Set[str] = set()
    reaches_next = bool(seed_report.get("reaches_level_8"))
    best_progress = int(seed_report.get("best_progress_delta", 0))
    for index, root in enumerate(curriculum_roots):
        root_raw = getattr(root.env, "observation_space", None)
        if root_raw is None:
            continue
        root_grid = _grid_from_raw(root_raw)
        local_frontier = LevelFrontier(
            target_level=frontier.target_level,
            level_start_index=frontier.level_start_index,
            frontier_index=frontier.frontier_index,
            terminal_index=frontier.terminal_index,
            level_start_frame=frontier.level_start_frame,
            frontier_frame=root_grid,
            terminal_frame=frontier.terminal_frame,
            immediate_danger_action=frontier.immediate_danger_action,
            danger_actions=frontier.danger_actions,
            visited_safe_hashes=frontier.visited_safe_hashes | seed_safe_hashes | {root.grid_hash},
            danger_state_hashes=frontier.danger_state_hashes,
        )
        report, child_nodes = _frontier_beam_search(
            base_env=root.env,
            base_raw=root_raw,
            full_game_id=full_game_id,
            selection=selection,
            frontier=local_frontier,
            horizon=curriculum_horizon,
            beam_width=curriculum_beam_width,
            max_click_targets=max_click_targets,
        )
        reaches_next = reaches_next or bool(report.get("reaches_level_8"))
        best_progress = max(best_progress, int(report.get("best_progress_delta", 0)))
        for child in child_nodes:
            if not child.died:
                curriculum_novel_hashes.update(child.novel_hashes)
        rows.append(
            {
                "rank": index + 1,
                "seed": root.to_report(),
                "extension": report,
                "combined_best_actions": (
                    list(root.actions)
                    + list((report.get("best") or {}).get("actions") or [])
                ),
            }
        )

    return {
        "seed_horizon": int(seed_horizon),
        "seed_beam_width": int(seed_beam_width),
        "curriculum_states": len(curriculum_roots),
        "curriculum_horizon": int(curriculum_horizon),
        "curriculum_beam_width": int(curriculum_beam_width),
        "seed_novel_safe_states": int(seed_report.get("novel_safe_states", 0)),
        "curriculum_novel_safe_states": len(curriculum_novel_hashes),
        "total_unique_novel_safe_states": len(
            {
                node.grid_hash
                for node in seed_nodes
                if not node.died and node.grid_hash not in frontier.visited_safe_hashes
            }
            | curriculum_novel_hashes
        ),
        "reaches_level_8": reaches_next,
        "best_progress_delta": best_progress,
        "roots": rows,
    }


def _apply_action_sequence(
    base_env: Any,
    *,
    full_game_id: str,
    actions: Sequence[str],
    action_data: Sequence[Optional[Dict[str, int]]],
) -> Tuple[Any, Any]:
    env = copy.deepcopy(base_env)
    raw = getattr(env, "observation_space", None)
    for action, data in zip(actions, action_data):
        raw = _step_branch(
            env,
            full_game_id=full_game_id,
            action=action,
            action_data=data,
        )
        if raw is None or _state_name(getattr(raw, "state", "UNKNOWN")) == "GAME_OVER":
            break
    return env, raw


def _best_action6_node(nodes: Sequence[BranchNode]) -> Optional[BranchNode]:
    candidates = [
        node
        for node in nodes
        if not node.died and "ACTION6" in node.actions
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            int(item.progress_delta),
            int(item.novel_safe_states),
            float(item.score),
        ),
    )


def _score_action6_result(
    *,
    pre_click_grid: Sequence[Sequence[int]],
    final_grid: Sequence[Sequence[int]],
    final_state: str,
    final_level: int,
    frontier: LevelFrontier,
) -> Dict[str, Any]:
    died = final_state in {"GAME_OVER", "ERROR"}
    grid_hash = _hash_grid(final_grid)
    progress_delta = max(0, int(final_level) - int(frontier.target_level))
    components_changed = _component_delta(pre_click_grid, final_grid)
    histogram_delta = _histogram_delta(pre_click_grid, final_grid)
    safe_novelty = (
        not died
        and grid_hash not in frontier.visited_safe_hashes
        and grid_hash not in frontier.danger_state_hashes
    )
    danger_hit = grid_hash in frontier.danger_state_hashes
    score = (
        100.0 * progress_delta
        + 6.0 * components_changed
        + 0.08 * histogram_delta
        + (8.0 if safe_novelty else 0.0)
        + (3.0 if not died else -80.0)
        - (25.0 if danger_hit else 0.0)
    )
    return {
        "score": round(float(score), 4),
        "progress_delta": progress_delta,
        "new_components_changed": components_changed,
        "histogram_delta": histogram_delta,
        "safe_novelty": bool(safe_novelty),
        "avoid_death": not died,
        "died": died,
        "danger_state_hit": danger_hit,
        "grid_hash": grid_hash,
    }


def run_action6_coordinate_sweep(
    *,
    base_env: Any,
    base_raw: Any,
    full_game_id: str,
    selection: SelectedEpisode,
    frontier: LevelFrontier,
    seed_horizon: int,
    seed_beam_width: int,
    sweep_grid_size: int,
    sweep_max_candidates: int,
    max_click_targets: int,
) -> Dict[str, Any]:
    """Sweep ACTION6 coordinates from the best non-fatal ACTION6 beam path."""
    seed_report, seed_nodes = _frontier_beam_search(
        base_env=base_env,
        base_raw=base_raw,
        full_game_id=full_game_id,
        selection=selection,
        frontier=frontier,
        horizon=seed_horizon,
        beam_width=seed_beam_width,
        max_click_targets=max_click_targets,
    )
    action6_node = _best_action6_node(seed_nodes)
    if action6_node is None:
        return {
            "skipped": "no safe ACTION6 path found in seed beam",
            "seed_horizon": int(seed_horizon),
            "seed_beam_width": int(seed_beam_width),
            "seed_best": seed_report.get("best"),
        }

    action6_index = action6_node.actions.index("ACTION6")
    prefix_actions = action6_node.actions[:action6_index]
    prefix_data = action6_node.action_data[:action6_index]
    suffix_actions = action6_node.actions[action6_index + 1 :]
    suffix_data = action6_node.action_data[action6_index + 1 :]
    pre_env, pre_raw = _apply_action_sequence(
        base_env,
        full_game_id=full_game_id,
        actions=prefix_actions,
        action_data=prefix_data,
    )
    pre_click_grid = _grid_from_raw(pre_raw)
    seeds = _click_targets(
        selection.steps,
        around_index=frontier.frontier_index,
        window=80,
        limit=max_click_targets,
    )
    candidates = _coordinate_candidates(
        pre_click_grid,
        grid_size=sweep_grid_size,
        max_candidates=sweep_max_candidates,
        seeds=seeds,
    )

    rows: List[Dict[str, Any]] = []
    novel_safe_hashes: Set[str] = set()
    for coord in candidates:
        env = copy.deepcopy(pre_env)
        raw = _step_branch(
            env,
            full_game_id=full_game_id,
            action="ACTION6",
            action_data=coord,
        )
        click_state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
        click_level = int(getattr(raw, "levels_completed", frontier.target_level) or 0) if raw is not None else frontier.target_level
        if raw is not None and click_state != "GAME_OVER":
            for action, data in zip(suffix_actions, suffix_data):
                raw = _step_branch(
                    env,
                    full_game_id=full_game_id,
                    action=action,
                    action_data=data,
                )
                if raw is None or _state_name(getattr(raw, "state", "ERROR")) == "GAME_OVER":
                    break
        final_state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
        final_level = int(getattr(raw, "levels_completed", click_level) or 0) if raw is not None else click_level
        final_grid = _grid_from_raw(raw)
        scored = _score_action6_result(
            pre_click_grid=pre_click_grid,
            final_grid=final_grid,
            final_state=final_state,
            final_level=final_level,
            frontier=frontier,
        )
        if scored["safe_novelty"]:
            novel_safe_hashes.add(str(scored["grid_hash"]))
        rows.append(
            {
                "coord": dict(coord),
                "path": list(prefix_actions) + ["ACTION6"] + list(suffix_actions),
                "prefix_actions": list(prefix_actions),
                "continuation_actions": list(suffix_actions),
                "click_state": click_state,
                "click_level": click_level,
                "final_state": final_state,
                "final_level": final_level,
                "changed_cells": _changed_cells(pre_click_grid, final_grid),
                **scored,
            }
        )

    rows.sort(
        key=lambda item: (
            int(item.get("progress_delta", 0)),
            int(item.get("new_components_changed", 0)),
            int(item.get("histogram_delta", 0)),
            1 if item.get("safe_novelty") else 0,
            1 if item.get("avoid_death") else 0,
            float(item.get("score", -999.0)),
        ),
        reverse=True,
    )
    return {
        "seed_horizon": int(seed_horizon),
        "seed_beam_width": int(seed_beam_width),
        "source_action6_path": action6_node.to_report(),
        "prefix_before_action6": list(prefix_actions),
        "continuation_after_action6": list(suffix_actions),
        "candidate_count": len(candidates),
        "reaches_level_8": any(int(item.get("progress_delta", 0)) > 0 for item in rows),
        "best_progress_delta": max((int(item.get("progress_delta", 0)) for item in rows), default=0),
        "novel_safe_states": len(novel_safe_hashes),
        "best": rows[0] if rows else None,
        "top": rows[:12],
    }


def _setup_action6_center_state(
    base_env: Any,
    *,
    full_game_id: str,
    prefix_actions: Sequence[str],
    center: Dict[str, int],
) -> Dict[str, Any]:
    prefix_data: List[Optional[Dict[str, int]]] = [None for _ in prefix_actions]
    pre_click_env, pre_click_raw = _apply_action_sequence(
        base_env,
        full_game_id=full_game_id,
        actions=prefix_actions,
        action_data=prefix_data,
    )
    pre_click_grid = _grid_from_raw(pre_click_raw)
    post_click_env = copy.deepcopy(pre_click_env)
    post_click_raw = _step_branch(
        post_click_env,
        full_game_id=full_game_id,
        action="ACTION6",
        action_data=center,
    )
    post_click_grid = _grid_from_raw(post_click_raw)
    return {
        "prefix_actions": list(prefix_actions),
        "center": dict(center),
        "pre_click_env": pre_click_env,
        "pre_click_raw": pre_click_raw,
        "pre_click_grid": pre_click_grid,
        "post_click_env": post_click_env,
        "post_click_raw": post_click_raw,
        "post_click_grid": post_click_grid,
        "post_click_state": _state_name(getattr(post_click_raw, "state", "ERROR")) if post_click_raw is not None else "ERROR",
        "post_click_level": int(getattr(post_click_raw, "levels_completed", 0) or 0) if post_click_raw is not None else 0,
    }


def _mechanism_score(
    *,
    base_grid: Sequence[Sequence[int]],
    final_grid: Sequence[Sequence[int]],
    final_state: str,
    final_level: int,
    target_level: int,
    new_safe_state: bool,
    no_op_loop: bool,
    repeatable_effect: bool,
    level_up_weight: float = 10000.0,
    component_weight: float = 500.0,
    repeatable_weight: float = 200.0,
    new_safe_weight: float = 100.0,
    histogram_weight: float = 0.05,
    death_penalty: float = 5000.0,
    no_op_penalty: float = 200.0,
) -> Dict[str, Any]:
    level_up = int(final_level) > int(target_level) or final_state == "WIN"
    death = final_state in {"GAME_OVER", "ERROR"}
    persistent_component_change = _component_delta(base_grid, final_grid)
    histogram_delta = _histogram_delta(base_grid, final_grid)
    changed_cells = _changed_cells(base_grid, final_grid)
    score = (
        float(level_up_weight) * (1 if level_up else 0)
        + float(component_weight) * persistent_component_change
        + float(repeatable_weight) * (1 if repeatable_effect else 0)
        + float(new_safe_weight) * (1 if new_safe_state else 0)
        + float(histogram_weight) * histogram_delta
        - float(death_penalty) * (1 if death else 0)
        - float(no_op_penalty) * (1 if no_op_loop else 0)
    )
    return {
        "score": round(float(score), 4),
        "level_up": bool(level_up),
        "persistent_component_change": int(persistent_component_change),
        "reversible_or_repeatable_effect": bool(repeatable_effect),
        "new_safe_state": bool(new_safe_state),
        "death": bool(death),
        "no_op_loop": bool(no_op_loop),
        "histogram_delta": int(histogram_delta),
        "changed_cells": int(changed_cells),
    }


def _max_consecutive(actions: Sequence[str], action_name: str) -> int:
    best = 0
    current = 0
    for action in actions:
        if action == action_name:
            current += 1
        else:
            current = 0
        best = max(best, current)
    return best


def _has_structural_action(actions: Sequence[str]) -> bool:
    return any(action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"} for action in actions)


def _strip_runtime_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"env", "raw", "path_hashes"}
    }


def _mechanism_branch(
    *,
    env: Any,
    full_game_id: str,
    action: str,
    action_data: Optional[Dict[str, int]],
    base_grid: Sequence[Sequence[int]],
    target_level: int,
    known_safe_hashes: Set[str],
    danger_hashes: Set[str],
    path_hashes: Set[str],
    path_actions: Sequence[str],
) -> Dict[str, Any]:
    branch_env = copy.deepcopy(env)
    raw = _step_branch(
        branch_env,
        full_game_id=full_game_id,
        action=action,
        action_data=action_data,
    )
    state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
    level = int(getattr(raw, "levels_completed", target_level) or 0) if raw is not None else target_level
    grid = _grid_from_raw(raw)
    grid_hash = _hash_grid(grid)
    death = state in {"GAME_OVER", "ERROR"}
    no_op_loop = grid_hash in path_hashes or _changed_cells(base_grid, grid) == 0
    new_safe = (
        not death
        and grid_hash not in known_safe_hashes
        and grid_hash not in danger_hashes
    )

    repeatable_effect = False
    repeat_state = "SKIPPED"
    repeat_hash = None
    if not death and action != "RESET":
        repeat_env = copy.deepcopy(branch_env)
        repeat_raw = _step_branch(
            repeat_env,
            full_game_id=full_game_id,
            action=action,
            action_data=action_data,
        )
        repeat_state = _state_name(getattr(repeat_raw, "state", "ERROR")) if repeat_raw is not None else "ERROR"
        repeat_grid = _grid_from_raw(repeat_raw)
        repeat_hash = _hash_grid(repeat_grid)
        repeatable_effect = (
            repeat_state != "GAME_OVER"
            and repeat_hash != grid_hash
            and _changed_cells(grid, repeat_grid) > 0
        )

    scored = _mechanism_score(
        base_grid=base_grid,
        final_grid=grid,
        final_state=state,
        final_level=level,
        target_level=target_level,
        new_safe_state=new_safe,
        no_op_loop=no_op_loop,
        repeatable_effect=repeatable_effect,
    )
    return {
        "actions": list(path_actions) + [action],
        "action_data": [None for _ in path_actions] + [dict(action_data) if action_data else None],
        "action": action,
        "action_data_last": dict(action_data) if action_data else None,
        "state": state,
        "level": level,
        "grid_hash": grid_hash,
        "repeat_state": repeat_state,
        "repeat_hash": repeat_hash,
        "change": _change_mechanism_report(base_grid, grid),
        "env": branch_env,
        "raw": raw,
        **scored,
    }


def _anti_cycle_mechanism_beam(
    *,
    start_env: Any,
    start_raw: Any,
    full_game_id: str,
    base_grid: Sequence[Sequence[int]],
    frontier: LevelFrontier,
    horizon: int,
    beam_width: int,
    click_targets: Sequence[Dict[str, int]],
    known_safe_hashes: Set[str],
    require_structural_by_step: Optional[int] = None,
    component_weight: float = 500.0,
    histogram_weight: float = 0.05,
) -> Dict[str, Any]:
    """Beam search that suppresses safe ACTION7 loops after a mechanism click."""
    root_hash = _hash_grid(_grid_from_raw(start_raw))
    root = {
        "env": copy.deepcopy(start_env),
        "raw": start_raw,
        "actions": [],
        "action_data": [],
        "path_hashes": {root_hash},
        "score": 0.0,
    }
    beam: List[Dict[str, Any]] = [root]
    evaluated: List[Dict[str, Any]] = []
    pruned_action7_cycles = 0
    pruned_action6_no_bbox_change = 0
    pruned_missing_structural = 0

    for _depth in range(1, max(1, horizon) + 1):
        expanded: List[Dict[str, Any]] = []
        for node in beam:
            raw = node.get("raw")
            state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
            if state in {"GAME_OVER", "ERROR", "WIN"}:
                expanded.append(node)
                continue
            current_grid = _grid_from_raw(raw)
            available = _available_names_from_raw(raw) or ACTION_NAMES
            for action in available:
                if action == "RESET":
                    continue
                next_actions = list(node.get("actions") or []) + [action]
                if _max_consecutive(next_actions, "ACTION7") > 2:
                    pruned_action7_cycles += 1
                    continue
                if (
                    require_structural_by_step is not None
                    and len(next_actions) >= int(require_structural_by_step)
                    and not _has_structural_action(next_actions)
                ):
                    pruned_missing_structural += 1
                    continue
                for data in _action_variants(
                    action,
                    full_game_id=full_game_id,
                    click_targets=click_targets,
                ):
                    branch_env = copy.deepcopy(node["env"])
                    branch_raw = _step_branch(
                        branch_env,
                        full_game_id=full_game_id,
                        action=action,
                        action_data=data,
                    )
                    branch_grid = _grid_from_raw(branch_raw)
                    if action == "ACTION6" and _diff_bbox(current_grid, branch_grid) is None:
                        pruned_action6_no_bbox_change += 1
                        continue
                    branch_state = _state_name(getattr(branch_raw, "state", "ERROR")) if branch_raw is not None else "ERROR"
                    branch_level = int(getattr(branch_raw, "levels_completed", frontier.target_level) or 0) if branch_raw is not None else frontier.target_level
                    branch_hash = _hash_grid(branch_grid)
                    death = branch_state in {"GAME_OVER", "ERROR"}
                    no_op_loop = branch_hash in set(node.get("path_hashes") or set()) or _changed_cells(base_grid, branch_grid) == 0
                    new_safe = (
                        not death
                        and branch_hash not in known_safe_hashes
                        and branch_hash not in frontier.danger_state_hashes
                    )

                    repeatable_effect = False
                    repeat_state = "SKIPPED"
                    repeat_hash = None
                    if not death:
                        repeat_env = copy.deepcopy(branch_env)
                        repeat_raw = _step_branch(
                            repeat_env,
                            full_game_id=full_game_id,
                            action=action,
                            action_data=data,
                        )
                        repeat_state = _state_name(getattr(repeat_raw, "state", "ERROR")) if repeat_raw is not None else "ERROR"
                        repeat_grid = _grid_from_raw(repeat_raw)
                        repeat_hash = _hash_grid(repeat_grid)
                        repeatable_effect = (
                            repeat_state != "GAME_OVER"
                            and repeat_hash != branch_hash
                            and _changed_cells(branch_grid, repeat_grid) > 0
                        )

                    scored = _mechanism_score(
                        base_grid=base_grid,
                        final_grid=branch_grid,
                        final_state=branch_state,
                        final_level=branch_level,
                        target_level=frontier.target_level,
                        new_safe_state=new_safe,
                        no_op_loop=no_op_loop,
                        repeatable_effect=repeatable_effect,
                        component_weight=component_weight,
                        histogram_weight=histogram_weight,
                    )
                    structural_action_seen = _has_structural_action(next_actions)
                    action7_penalty = 150.0 * next_actions.count("ACTION7")
                    missing_structural_penalty = 2000.0 if not structural_action_seen else 0.0
                    adjusted_score = (
                        float(scored["score"])
                        - action7_penalty
                        - missing_structural_penalty
                    )
                    branch = {
                        "env": branch_env,
                        "raw": branch_raw,
                        "actions": next_actions,
                        "action_data": list(node.get("action_data") or []) + [dict(data) if data else None],
                        "action": action,
                        "action_data_last": dict(data) if data else None,
                        "state": branch_state,
                        "level": branch_level,
                        "grid_hash": branch_hash,
                        "repeat_state": repeat_state,
                        "repeat_hash": repeat_hash,
                        "structural_action_seen": structural_action_seen,
                        "action7_count": next_actions.count("ACTION7"),
                        "max_consecutive_action7": _max_consecutive(next_actions, "ACTION7"),
                        "anti_cycle_penalty": round(action7_penalty + missing_structural_penalty, 4),
                        "score": round(adjusted_score, 4),
                        "raw_mechanism_score": scored["score"],
                        "change": _change_mechanism_report(base_grid, branch_grid),
                        "path_hashes": set(node.get("path_hashes") or set()) | {branch_hash},
                        **{key: value for key, value in scored.items() if key != "score"},
                    }
                    expanded.append(branch)
        expanded.sort(key=lambda item: float(item.get("score", -999999.0)), reverse=True)
        for item in expanded:
            if "change" in item:
                evaluated.append(item)
                if item.get("new_safe_state"):
                    known_safe_hashes.add(str(item["grid_hash"]))
        beam = expanded[: max(1, beam_width)]

    ranked = sorted(evaluated, key=lambda item: float(item.get("score", -999999.0)), reverse=True)
    safe_structural = [
        item
        for item in ranked
        if not item.get("death") and item.get("structural_action_seen")
    ]
    safe_any = [item for item in ranked if not item.get("death")]
    selected = safe_structural or safe_any or ranked
    novel_safe_hashes = {
        str(item["grid_hash"])
        for item in evaluated
        if item.get("new_safe_state")
    }
    return {
        "horizon": int(horizon),
        "beam_width": int(beam_width),
        "evaluated_branches": len(evaluated),
        "pruned_action7_cycles": pruned_action7_cycles,
        "pruned_action6_no_bbox_change": pruned_action6_no_bbox_change,
        "pruned_missing_structural_by_step": pruned_missing_structural,
        "require_structural_by_step": require_structural_by_step,
        "component_weight": component_weight,
        "histogram_weight": histogram_weight,
        "reaches_level_8": any(item.get("level_up") for item in evaluated),
        "best_progress_delta": max(
            (max(0, int(item.get("level", frontier.target_level)) - frontier.target_level) for item in evaluated),
            default=0,
        ),
        "novel_safe_states": len(novel_safe_hashes),
        "best": _strip_runtime_fields(selected[0]) if selected else None,
        "top": [_strip_runtime_fields(item) for item in selected[:12]],
        "top_dead": [_strip_runtime_fields(item) for item in ranked if item.get("death")][:5],
    }


def run_post_action6_center_policy_search(
    *,
    base_env: Any,
    full_game_id: str,
    frontier: LevelFrontier,
    prefix_actions: Sequence[str],
    center: Dict[str, int],
    horizon: int,
    beam_width: int,
    max_click_targets: int,
) -> Dict[str, Any]:
    """Search for mechanism-relevant actions after ACTION6(center)."""
    setup = _setup_action6_center_state(
        base_env,
        full_game_id=full_game_id,
        prefix_actions=prefix_actions,
        center=center,
    )
    if setup["post_click_state"] in {"GAME_OVER", "ERROR"}:
        return {
            "skipped": f"ACTION6 center led to {setup['post_click_state']}",
            "prefix_actions": list(prefix_actions),
            "center": dict(center),
            "what_changed_after_ACTION6_center": _change_mechanism_report(
                setup["pre_click_grid"],
                setup["post_click_grid"],
            ),
        }

    post_grid = setup["post_click_grid"]
    root_hash = _hash_grid(post_grid)
    known_safe_hashes = set(frontier.visited_safe_hashes) | {root_hash}
    click_targets = [dict(center)]
    click_targets.extend(
        _coordinate_candidates(
            post_grid,
            grid_size=4,
            max_candidates=max(1, max_click_targets),
        )
    )

    root = {
        "env": setup["post_click_env"],
        "raw": setup["post_click_raw"],
        "actions": [],
        "path_hashes": {root_hash},
        "score": 0.0,
    }
    beam: List[Dict[str, Any]] = [root]
    evaluated: List[Dict[str, Any]] = []

    for _depth in range(1, max(1, horizon) + 1):
        expanded: List[Dict[str, Any]] = []
        for node in beam:
            raw = node.get("raw")
            state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
            if state in {"GAME_OVER", "ERROR", "WIN"}:
                expanded.append(node)
                continue
            available = _available_names_from_raw(raw) or ACTION_NAMES
            for action in available:
                if action == "RESET":
                    continue
                for data in _action_variants(
                    action,
                    full_game_id=full_game_id,
                    click_targets=click_targets,
                ):
                    branch = _mechanism_branch(
                        env=node["env"],
                        full_game_id=full_game_id,
                        action=action,
                        action_data=data,
                        base_grid=post_grid,
                        target_level=frontier.target_level,
                        known_safe_hashes=known_safe_hashes,
                        danger_hashes=frontier.danger_state_hashes,
                        path_hashes=set(node.get("path_hashes") or set()),
                        path_actions=list(node.get("actions") or []),
                    )
                    branch["path_hashes"] = set(node.get("path_hashes") or set()) | {branch["grid_hash"]}
                    expanded.append(branch)
        expanded.sort(key=lambda item: float(item.get("score", -999999.0)), reverse=True)
        for item in expanded:
            if "change" in item:
                evaluated.append(item)
                if item.get("new_safe_state"):
                    known_safe_hashes.add(str(item["grid_hash"]))
        beam = expanded[: max(1, beam_width)]

    ranked = sorted(evaluated, key=lambda item: float(item.get("score", -999999.0)), reverse=True)
    safe_ranked = [
        item
        for item in ranked
        if not item.get("death")
    ]
    novel_safe_hashes = {
        str(item["grid_hash"])
        for item in evaluated
        if item.get("new_safe_state")
    }
    level_up = any(item.get("level_up") for item in evaluated)
    best_progress_delta = max(
        (max(0, int(item.get("level", frontier.target_level)) - frontier.target_level) for item in evaluated),
        default=0,
    )
    compact_top: List[Dict[str, Any]] = []
    for item in (safe_ranked or ranked)[:12]:
        compact = {
            key: value
            for key, value in item.items()
            if key not in {"env", "raw", "path_hashes"}
        }
        compact_top.append(compact)
    compact_dead_top: List[Dict[str, Any]] = []
    for item in [row for row in ranked if row.get("death")][:5]:
        compact_dead_top.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"env", "raw", "path_hashes"}
            }
        )

    return {
        "prefix_actions": list(prefix_actions),
        "center": dict(center),
        "horizon": int(horizon),
        "beam_width": int(beam_width),
        "evaluated_branches": len(evaluated),
        "what_changed_after_ACTION6_center": _change_mechanism_report(
            setup["pre_click_grid"],
            setup["post_click_grid"],
        ),
        "post_click": {
            "state": setup["post_click_state"],
            "level": setup["post_click_level"],
            "grid_hash": root_hash,
        },
        "reaches_level_8": bool(level_up),
        "best_progress_delta": int(best_progress_delta),
        "novel_safe_states": len(novel_safe_hashes),
        "best": compact_top[0] if compact_top else None,
        "top": compact_top,
        "top_dead": compact_dead_top,
    }


def run_bbox_targeted_action6_sweep(
    *,
    base_env: Any,
    full_game_id: str,
    frontier: LevelFrontier,
    prefix_actions: Sequence[str],
    center: Dict[str, int],
    candidate_limit: int,
    followup_roots: int,
    followup_horizon: int,
    followup_beam_width: int,
    require_structural_by_step: Optional[int] = None,
    component_weight: float = 500.0,
    histogram_weight: float = 0.05,
) -> Dict[str, Any]:
    """Target a second ACTION6 at geometry induced by the center-click diff."""
    setup = _setup_action6_center_state(
        base_env,
        full_game_id=full_game_id,
        prefix_actions=prefix_actions,
        center=center,
    )
    pre_click_grid = setup["pre_click_grid"]
    post_click_grid = setup["post_click_grid"]
    center_change = _change_mechanism_report(pre_click_grid, post_click_grid)
    center_bbox = center_change.get("bounding_box_of_diff")
    candidates = _bbox_targeted_action6_candidates(
        pre_click_grid,
        post_click_grid,
        max_candidates=candidate_limit,
    )
    if setup["post_click_state"] in {"GAME_OVER", "ERROR"}:
        return {
            "skipped": f"ACTION6 center led to {setup['post_click_state']}",
            "prefix_actions": list(prefix_actions),
            "center": dict(center),
            "what_changed_after_ACTION6_center": center_change,
            "candidate_count": len(candidates),
        }

    candidate_rows: List[Dict[str, Any]] = []
    known_safe_hashes = set(frontier.visited_safe_hashes) | {
        _hash_grid(post_click_grid),
    }
    for coord in candidates:
        env = copy.deepcopy(setup["post_click_env"])
        raw = _step_branch(
            env,
            full_game_id=full_game_id,
            action="ACTION6",
            action_data=coord,
        )
        state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
        level = int(getattr(raw, "levels_completed", frontier.target_level) or 0) if raw is not None else frontier.target_level
        grid = _grid_from_raw(raw)
        grid_hash = _hash_grid(grid)
        second_bbox = _diff_bbox(post_click_grid, grid)
        bbox_changed = second_bbox is not None
        death = state in {"GAME_OVER", "ERROR"}
        new_safe = (
            not death
            and grid_hash not in known_safe_hashes
            and grid_hash not in frontier.danger_state_hashes
        )
        scored = _mechanism_score(
            base_grid=post_click_grid,
            final_grid=grid,
            final_state=state,
            final_level=level,
            target_level=frontier.target_level,
            new_safe_state=new_safe,
            no_op_loop=not bbox_changed,
            repeatable_effect=False,
        )
        row = {
            "coord": dict(coord),
            "state": state,
            "level": level,
            "grid_hash": grid_hash,
            "bbox_changed": bool(bbox_changed),
            "bbox_same_as_center": _same_bbox(second_bbox, center_bbox),
            "bbox": second_bbox,
            "change": _change_mechanism_report(post_click_grid, grid),
            "env": env,
            "raw": raw,
            **scored,
        }
        candidate_rows.append(row)
        if new_safe:
            known_safe_hashes.add(grid_hash)

    viable = [
        row
        for row in candidate_rows
        if row.get("bbox_changed") and not row.get("death")
    ]
    viable.sort(
        key=lambda item: (
            int(item.get("level_up", False)),
            int(item.get("persistent_component_change", 0)),
            int(item.get("histogram_delta", 0)),
            1 if item.get("new_safe_state") else 0,
            float(item.get("score", -999999.0)),
        ),
        reverse=True,
    )

    followups: List[Dict[str, Any]] = []
    bbox_click_targets = [dict(row["coord"]) for row in viable[: max(1, followup_roots)]]
    for rank, row in enumerate(viable[: max(1, followup_roots)], start=1):
        beam_report = _anti_cycle_mechanism_beam(
            start_env=row["env"],
            start_raw=row["raw"],
            full_game_id=full_game_id,
            base_grid=post_click_grid,
            frontier=frontier,
            horizon=followup_horizon,
            beam_width=followup_beam_width,
            click_targets=bbox_click_targets,
            known_safe_hashes=known_safe_hashes | {str(row["grid_hash"])},
            require_structural_by_step=require_structural_by_step,
            component_weight=component_weight,
            histogram_weight=histogram_weight,
        )
        best = beam_report.get("best") or {}
        combined_score = float(row.get("score", 0.0)) + float(best.get("score", 0.0))
        followups.append(
            {
                "rank": rank,
                "second_click": _strip_runtime_fields(row),
                "anti_cycle_beam": beam_report,
                "combined_best_actions": ["ACTION6"] + list(best.get("actions") or []),
                "combined_score": round(combined_score, 4),
            }
        )

    followups.sort(
        key=lambda item: (
            bool((item.get("anti_cycle_beam") or {}).get("reaches_level_8")),
            int((item.get("anti_cycle_beam") or {}).get("novel_safe_states", 0)),
            float(item.get("combined_score", -999999.0)),
        ),
        reverse=True,
    )
    compact_candidates = [
        _strip_runtime_fields(row)
        for row in viable[:12]
    ]
    novel_safe_hashes = {
        str(row["grid_hash"])
        for row in viable
        if row.get("new_safe_state")
    }
    for followup in followups:
        for item in (followup.get("anti_cycle_beam") or {}).get("top", []):
            if item.get("new_safe_state"):
                novel_safe_hashes.add(str(item["grid_hash"]))

    best_followup = followups[0] if followups else None
    return {
        "prefix_actions": list(prefix_actions),
        "center": dict(center),
        "what_changed_after_ACTION6_center": center_change,
        "candidate_count": len(candidates),
        "viable_candidate_count": len(viable),
        "followup_roots": min(len(viable), max(1, followup_roots)),
        "followup_horizon": int(followup_horizon),
        "followup_beam_width": int(followup_beam_width),
        "require_structural_by_step": require_structural_by_step,
        "component_weight": component_weight,
        "histogram_weight": histogram_weight,
        "anti_cycle_rules": {
            "max_consecutive_ACTION7": 2,
            "requires_ACTION1_to_ACTION5_for_best": True,
            "requires_ACTION1_to_ACTION5_by_step": require_structural_by_step,
            "ACTION6_requires_bbox_change": True,
        },
        "reaches_level_8": any(
            bool((item.get("anti_cycle_beam") or {}).get("reaches_level_8"))
            for item in followups
        ),
        "best_progress_delta": max(
            [
                int(row.get("progress_delta", 0))
                for row in viable
            ]
            + [
                int((item.get("anti_cycle_beam") or {}).get("best_progress_delta", 0))
                for item in followups
            ]
            + [0]
        ),
        "novel_safe_states": len(novel_safe_hashes),
        "best": best_followup,
        "top_candidates": compact_candidates,
        "followups": followups,
    }


def run_action3_phase_motif_grid(
    *,
    base_env: Any,
    full_game_id: str,
    frontier: LevelFrontier,
    prefix_actions: Sequence[str],
    center: Dict[str, int],
    candidate_limit: int,
    target_roots: int,
    max_k: int,
    max_m: int,
) -> Dict[str, Any]:
    """Evaluate ACTION7^k -> ACTION3 -> ACTION7^m motifs after bbox ACTION6."""
    setup = _setup_action6_center_state(
        base_env,
        full_game_id=full_game_id,
        prefix_actions=prefix_actions,
        center=center,
    )
    pre_click_grid = setup["pre_click_grid"]
    post_click_grid = setup["post_click_grid"]
    center_change = _compact_change_report(pre_click_grid, post_click_grid)
    candidates = _bbox_targeted_action6_candidates(
        pre_click_grid,
        post_click_grid,
        max_candidates=candidate_limit,
    )
    if setup["post_click_state"] in {"GAME_OVER", "ERROR"}:
        return {
            "skipped": f"ACTION6 center led to {setup['post_click_state']}",
            "prefix_actions": list(prefix_actions),
            "center": dict(center),
            "what_changed_after_ACTION6_center": center_change,
            "candidate_count": len(candidates),
        }

    viable: List[Dict[str, Any]] = []
    known_safe_hashes = set(frontier.visited_safe_hashes) | {_hash_grid(post_click_grid)}
    for coord in candidates:
        env = copy.deepcopy(setup["post_click_env"])
        raw = _step_branch(
            env,
            full_game_id=full_game_id,
            action="ACTION6",
            action_data=coord,
        )
        state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
        level = int(getattr(raw, "levels_completed", frontier.target_level) or 0) if raw is not None else frontier.target_level
        grid = _grid_from_raw(raw)
        grid_hash = _hash_grid(grid)
        bbox = _diff_bbox(post_click_grid, grid)
        if bbox is None or state in {"GAME_OVER", "ERROR"}:
            continue
        new_safe = grid_hash not in known_safe_hashes and grid_hash not in frontier.danger_state_hashes
        scored = _mechanism_score(
            base_grid=post_click_grid,
            final_grid=grid,
            final_state=state,
            final_level=level,
            target_level=frontier.target_level,
            new_safe_state=new_safe,
            no_op_loop=False,
            repeatable_effect=False,
            component_weight=120.0,
            histogram_weight=0.01,
        )
        viable.append(
            {
                "coord": dict(coord),
                "env": env,
                "raw": raw,
                "grid": grid,
                "grid_hash": grid_hash,
                "bbox": bbox,
                "state": state,
                "level": level,
                **scored,
            }
        )
        if new_safe:
            known_safe_hashes.add(grid_hash)

    viable.sort(
        key=lambda item: (
            int(item.get("level_up", False)),
            int(item.get("persistent_component_change", 0)),
            int(item.get("histogram_delta", 0)),
            1 if item.get("new_safe_state") else 0,
            float(item.get("score", -999999.0)),
        ),
        reverse=True,
    )
    targets = viable[: max(1, target_roots)]

    rows: List[Dict[str, Any]] = []
    action3_distinct_count = 0
    new_safe_hashes: Set[str] = set()
    for target_rank, target in enumerate(targets, start=1):
        after_action6_grid = target["grid"]
        after_action6_bbox = target["bbox"]
        for k in range(max(0, max_k) + 1):
            prepared_env = copy.deepcopy(target["env"])
            prepared_raw = target["raw"]
            prep_actions: List[str] = []
            died_during_prep = False
            for _ in range(k):
                prepared_raw = _step_branch(
                    prepared_env,
                    full_game_id=full_game_id,
                    action="ACTION7",
                    action_data=None,
                )
                prep_actions.append("ACTION7")
                if prepared_raw is None or _state_name(getattr(prepared_raw, "state", "ERROR")) == "GAME_OVER":
                    died_during_prep = True
                    break
            if died_during_prep:
                continue

            before_action3_grid = _grid_from_raw(prepared_raw)
            action3_env = copy.deepcopy(prepared_env)
            action3_raw = _step_branch(
                action3_env,
                full_game_id=full_game_id,
                action="ACTION3",
                action_data=None,
            )
            action3_state = _state_name(getattr(action3_raw, "state", "ERROR")) if action3_raw is not None else "ERROR"
            action3_level = int(getattr(action3_raw, "levels_completed", frontier.target_level) or 0) if action3_raw is not None else frontier.target_level
            after_action3_grid = _grid_from_raw(action3_raw)
            bbox_after_action3 = _diff_bbox(before_action3_grid, after_action3_grid)
            bbox_vs_after_action6 = _diff_bbox(after_action6_grid, after_action3_grid)
            action3_component_delta = _component_delta(before_action3_grid, after_action3_grid)
            action3_distinct = (
                bbox_after_action3 is not None
                and not _same_bbox(bbox_after_action3, after_action6_bbox)
            )
            if action3_distinct:
                action3_distinct_count += 1

            for m in range(max(0, max_m) + 1):
                post_env = copy.deepcopy(action3_env)
                post_raw = action3_raw
                suffix_actions: List[str] = []
                died_during_suffix = action3_state in {"GAME_OVER", "ERROR"}
                for _ in range(m):
                    if died_during_suffix:
                        break
                    post_raw = _step_branch(
                        post_env,
                        full_game_id=full_game_id,
                        action="ACTION7",
                        action_data=None,
                    )
                    suffix_actions.append("ACTION7")
                    if post_raw is None or _state_name(getattr(post_raw, "state", "ERROR")) == "GAME_OVER":
                        died_during_suffix = True
                        break
                if died_during_suffix:
                    final_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
                else:
                    final_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]

                for final_action in final_actions:
                    final_env = copy.deepcopy(post_env)
                    final_data = dict(target["coord"]) if final_action == "ACTION6" else None
                    final_raw = _step_branch(
                        final_env,
                        full_game_id=full_game_id,
                        action=final_action,
                        action_data=final_data,
                    )
                    final_state = _state_name(getattr(final_raw, "state", "ERROR")) if final_raw is not None else "ERROR"
                    final_level = int(getattr(final_raw, "levels_completed", frontier.target_level) or 0) if final_raw is not None else frontier.target_level
                    final_grid = _grid_from_raw(final_raw)
                    final_hash = _hash_grid(final_grid)
                    death = final_state in {"GAME_OVER", "ERROR"}
                    new_safe = (
                        not death
                        and final_hash not in known_safe_hashes
                        and final_hash not in frontier.danger_state_hashes
                    )
                    if new_safe:
                        new_safe_hashes.add(final_hash)
                        known_safe_hashes.add(final_hash)
                    final_component_delta = _component_delta(after_action6_grid, final_grid)
                    level_up = final_level > frontier.target_level or final_state == "WIN"
                    no_op = _changed_cells(after_action3_grid, final_grid) == 0
                    score = (
                        10000.0 * (1 if level_up else 0)
                        + 1500.0 * (1 if action3_distinct else 0)
                        + 250.0 * action3_component_delta
                        + 80.0 * final_component_delta
                        + 100.0 * (1 if new_safe else 0)
                        - 5000.0 * (1 if death else 0)
                        - 300.0 * (1 if no_op else 0)
                        - 60.0 * (k + m)
                    )
                    rows.append(
                        {
                            "target_rank": target_rank,
                            "target_coord": dict(target["coord"]),
                            "k_action7_before_action3": k,
                            "m_action7_after_action3": m,
                            "final_action": final_action,
                            "sequence": (
                                ["ACTION6"]
                                + prep_actions
                                + ["ACTION3"]
                                + suffix_actions
                                + [final_action]
                            ),
                            "state_after_ACTION3": action3_state,
                            "level_after_ACTION3": action3_level,
                            "bbox_after_ACTION3": bbox_after_action3,
                            "bbox_after_ACTION3_distinct_from_ACTION6": action3_distinct,
                            "bbox_vs_after_ACTION6": bbox_vs_after_action6,
                            "delta_vs_after_ACTION6": _compact_change_report(
                                after_action6_grid,
                                after_action3_grid,
                            ),
                            "new_components_changed_after_ACTION3": action3_component_delta,
                            "final_state": final_state,
                            "final_level": final_level,
                            "level_up": bool(level_up),
                            "death": bool(death),
                            "new_safe_state": bool(new_safe),
                            "final_component_delta_vs_after_ACTION6": final_component_delta,
                            "score": round(float(score), 4),
                        }
                    )

    rows.sort(
        key=lambda item: (
            bool(item.get("level_up")),
            bool(item.get("bbox_after_ACTION3_distinct_from_ACTION6")),
            int(item.get("new_components_changed_after_ACTION3", 0)),
            bool(item.get("new_safe_state")),
            float(item.get("score", -999999.0)),
        ),
        reverse=True,
    )
    safe_rows = [row for row in rows if not row.get("death")]
    selected = safe_rows or rows
    distinct_action3_rows = [
        row for row in rows if row.get("bbox_after_ACTION3_distinct_from_ACTION6")
    ]
    return {
        "prefix_actions": list(prefix_actions),
        "center": dict(center),
        "candidate_count": len(candidates),
        "viable_candidate_count": len(viable),
        "target_roots": len(targets),
        "k_range": [0, int(max_k)],
        "m_range": [0, int(max_m)],
        "evaluated_motifs": len(rows),
        "what_changed_after_ACTION6_center": center_change,
        "reaches_level_8": any(row.get("level_up") for row in rows),
        "best_progress_delta": max(
            (max(0, int(row.get("final_level", frontier.target_level)) - frontier.target_level) for row in rows),
            default=0,
        ),
        "new_safe_states": len(new_safe_hashes),
        "action3_distinct_bbox_count": action3_distinct_count,
        "best": selected[0] if selected else None,
        "top": selected[:24],
        "top_action3_distinct": distinct_action3_rows[:12],
        "targets": [
            {
                key: value
                for key, value in target.items()
                if key not in {"env", "raw", "grid"}
            }
            for target in targets
        ],
    }


def run_action3_precondition_matrix(
    *,
    base_env: Any,
    full_game_id: str,
    frontier: LevelFrontier,
    prefix_actions: Sequence[str],
    center: Dict[str, int],
    candidate_limit: int,
    target_roots: int,
    max_k: int,
) -> Dict[str, Any]:
    """Stop immediately after ACTION3 to classify its safety preconditions."""
    setup = _setup_action6_center_state(
        base_env,
        full_game_id=full_game_id,
        prefix_actions=prefix_actions,
        center=center,
    )
    pre_click_grid = setup["pre_click_grid"]
    post_click_grid = setup["post_click_grid"]
    center_change = _compact_change_report(pre_click_grid, post_click_grid)
    candidates = _bbox_targeted_action6_candidates(
        pre_click_grid,
        post_click_grid,
        max_candidates=candidate_limit,
    )
    if setup["post_click_state"] in {"GAME_OVER", "ERROR"}:
        return {
            "skipped": f"ACTION6 center led to {setup['post_click_state']}",
            "prefix_actions": list(prefix_actions),
            "center": dict(center),
            "what_changed_after_ACTION6_center": center_change,
            "candidate_count": len(candidates),
        }

    viable: List[Dict[str, Any]] = []
    known_safe_hashes = set(frontier.visited_safe_hashes) | {_hash_grid(post_click_grid)}
    for coord in candidates:
        env = copy.deepcopy(setup["post_click_env"])
        raw = _step_branch(
            env,
            full_game_id=full_game_id,
            action="ACTION6",
            action_data=coord,
        )
        state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
        level = int(getattr(raw, "levels_completed", frontier.target_level) or 0) if raw is not None else frontier.target_level
        grid = _grid_from_raw(raw)
        bbox = _diff_bbox(post_click_grid, grid)
        if bbox is None or state in {"GAME_OVER", "ERROR"}:
            continue
        scored = _mechanism_score(
            base_grid=post_click_grid,
            final_grid=grid,
            final_state=state,
            final_level=level,
            target_level=frontier.target_level,
            new_safe_state=_hash_grid(grid) not in known_safe_hashes,
            no_op_loop=False,
            repeatable_effect=False,
            component_weight=120.0,
            histogram_weight=0.01,
        )
        viable.append(
            {
                "coord": dict(coord),
                "env": env,
                "raw": raw,
                "grid": grid,
                "bbox": bbox,
                "state": state,
                "level": level,
                **scored,
            }
        )

    viable.sort(
        key=lambda item: (
            int(item.get("level_up", False)),
            int(item.get("persistent_component_change", 0)),
            int(item.get("histogram_delta", 0)),
            1 if item.get("new_safe_state") else 0,
            float(item.get("score", -999999.0)),
        ),
        reverse=True,
    )
    targets = viable[: max(1, target_roots)]

    rows: List[Dict[str, Any]] = []
    buckets = {
        "safe_distinct_bbox": [],
        "fatal_distinct_bbox": [],
        "safe_no_distinct_bbox": [],
        "fatal_no_distinct_bbox": [],
    }
    for target_rank, target in enumerate(targets, start=1):
        after_action6_grid = target["grid"]
        after_action6_bbox = target["bbox"]
        for k in range(max(0, max_k) + 1):
            env = copy.deepcopy(target["env"])
            raw = target["raw"]
            died_during_prep = False
            for _ in range(k):
                raw = _step_branch(
                    env,
                    full_game_id=full_game_id,
                    action="ACTION7",
                    action_data=None,
                )
                if raw is None or _state_name(getattr(raw, "state", "ERROR")) == "GAME_OVER":
                    died_during_prep = True
                    break
            if died_during_prep:
                continue

            before_action3_grid = _grid_from_raw(raw)
            raw = _step_branch(
                env,
                full_game_id=full_game_id,
                action="ACTION3",
                action_data=None,
            )
            state = _state_name(getattr(raw, "state", "ERROR")) if raw is not None else "ERROR"
            level = int(getattr(raw, "levels_completed", frontier.target_level) or 0) if raw is not None else frontier.target_level
            after_action3_grid = _grid_from_raw(raw)
            bbox_after_action3 = _diff_bbox(before_action3_grid, after_action3_grid)
            distinct_bbox = (
                bbox_after_action3 is not None
                and not _same_bbox(bbox_after_action3, after_action6_bbox)
            )
            death = state in {"GAME_OVER", "ERROR"}
            bucket = (
                "fatal_distinct_bbox"
                if death and distinct_bbox
                else "safe_distinct_bbox"
                if distinct_bbox
                else "fatal_no_distinct_bbox"
                if death
                else "safe_no_distinct_bbox"
            )
            row = {
                "target_rank": target_rank,
                "target_coord": dict(target["coord"]),
                "k_action7_before_ACTION3": k,
                "sequence": ["ACTION6"] + ["ACTION7"] * k + ["ACTION3"],
                "state_after_ACTION3": state,
                "level_after_ACTION3": level,
                "death": bool(death),
                "bbox_after_ACTION3": bbox_after_action3,
                "bbox_after_ACTION3_distinct_from_ACTION6": bool(distinct_bbox),
                "delta_vs_after_ACTION6": _compact_change_report(
                    after_action6_grid,
                    after_action3_grid,
                ),
                "new_components_changed_after_ACTION3": _component_delta(
                    before_action3_grid,
                    after_action3_grid,
                ),
                "bucket": bucket,
            }
            rows.append(row)
            buckets[bucket].append(row)

    rows.sort(
        key=lambda item: (
            item["bucket"] == "safe_distinct_bbox",
            item["bucket"] == "fatal_distinct_bbox",
            int(item.get("new_components_changed_after_ACTION3", 0)),
            -int(item.get("k_action7_before_ACTION3", 0)),
        ),
        reverse=True,
    )
    return {
        "prefix_actions": list(prefix_actions),
        "center": dict(center),
        "candidate_count": len(candidates),
        "viable_candidate_count": len(viable),
        "target_roots": len(targets),
        "k_range": [0, int(max_k)],
        "evaluated_trials": len(rows),
        "what_changed_after_ACTION6_center": center_change,
        "ACTION3_safe_distinct_bbox_count": len(buckets["safe_distinct_bbox"]),
        "fatal_distinct_bbox_count": len(buckets["fatal_distinct_bbox"]),
        "safe_no_distinct_bbox_count": len(buckets["safe_no_distinct_bbox"]),
        "fatal_no_distinct_bbox_count": len(buckets["fatal_no_distinct_bbox"]),
        "best": rows[0] if rows else None,
        "safe_distinct_bbox": buckets["safe_distinct_bbox"][:12],
        "fatal_distinct_bbox": buckets["fatal_distinct_bbox"][:12],
        "safe_no_distinct_bbox": buckets["safe_no_distinct_bbox"][:12],
        "top": rows[:24],
        "targets": [
            {
                key: value
                for key, value in target.items()
                if key not in {"env", "raw", "grid"}
            }
            for target in targets
        ],
    }


def build_hypothesis_report(
    selection: SelectedEpisode,
    frontier: LevelFrontier,
) -> Dict[str, Any]:
    """Summarize what changed from level start to fatal frontier/death."""
    return {
        "level_start_to_frontier": _compare_states(
            "level_start_to_frontier",
            frontier.level_start_frame,
            frontier.frontier_frame,
        ),
        "frontier_to_death": _compare_states(
            "frontier_to_death",
            frontier.frontier_frame,
            frontier.terminal_frame,
        ),
        "danger": {
            "trace_step": selection.steps[frontier.terminal_index].step,
            "trace_index": frontier.terminal_index,
            "immediate_action": frontier.immediate_danger_action,
            "danger_actions": list(frontier.danger_actions),
            "danger_state_hashes": sorted(frontier.danger_state_hashes),
        },
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_report(payload: Dict[str, Any], output: Path) -> None:
    frontier = payload["frontier"]
    beam = payload.get("frontier_beam") or {}
    best = beam.get("best") or {}
    repair = payload.get("counterfactual_suffix_repair") or {}
    curriculum = payload.get("frontier_curriculum") or {}
    action6 = payload.get("level7_action6_coordinate_sweep") or {}
    post_center = payload.get("post_action6_center_policy_search") or {}
    bbox_sweep = payload.get("bbox_targeted_action6_sweep") or {}
    bbox_focused = payload.get("bbox_focused_deepening") or {}
    motif_grid = payload.get("action3_phase_motif_grid") or {}
    action3_matrix = payload.get("action3_precondition_matrix") or {}
    print("=" * 88)
    print("Level-7 frontier recovery")
    print("=" * 88)
    print(f"game:        {payload['game']}")
    print(f"resolved:    {payload['resolved_game_id']}")
    print(f"episode:     {payload['episode_id']} ({payload['selected_reason']})")
    print(
        "frontier:    "
        f"level={frontier['target_level']} "
        f"trace_index={frontier['frontier_index']} "
        f"danger={frontier['immediate_danger_action']} "
        f"suffix={frontier['danger_actions']}"
    )
    if beam:
        print(
            "beam:        "
            f"reaches_level_8={beam.get('reaches_level_8')} "
            f"avoids_known_death_suffix={beam.get('avoids_known_death_suffix')} "
            f"best_progress_delta={beam.get('best_progress_delta')} "
            f"novel_safe_states={beam.get('novel_safe_states')}"
        )
    if beam and best:
        print(
            "best:        "
            f"actions={best.get('actions')} "
            f"state={best.get('state')} "
            f"level={best.get('level')} "
            f"score={best.get('score')}"
        )
    if repair:
        repaired = repair.get("best") or {}
        repaired_best = repaired.get("best") or {}
        print(
            "repair:      "
            f"points={repair.get('evaluated_points')} "
            f"best_trace_index={repaired.get('trace_index')} "
            f"actions={repaired_best.get('actions')}"
        )
    if curriculum:
        print(
            "curriculum:  "
            f"states={curriculum.get('curriculum_states')} "
            f"new_safe={curriculum.get('curriculum_novel_safe_states')} "
            f"total_safe={curriculum.get('total_unique_novel_safe_states')} "
            f"reaches_level_8={curriculum.get('reaches_level_8')}"
        )
    if action6:
        if action6.get("skipped"):
            print(f"action6:     skipped={action6['skipped']}")
        else:
            action6_best = action6.get("best") or {}
            print(
                "action6:     "
                f"candidates={action6.get('candidate_count')} "
                f"new_safe={action6.get('novel_safe_states')} "
                f"best_coord={action6_best.get('coord')} "
                f"score={action6_best.get('score')}"
            )
    if post_center:
        if post_center.get("skipped"):
            print(f"post-a6:     skipped={post_center['skipped']}")
        else:
            post_best = post_center.get("best") or {}
            print(
                "post-a6:     "
                f"branches={post_center.get('evaluated_branches')} "
                f"new_safe={post_center.get('novel_safe_states')} "
                f"reaches_level_8={post_center.get('reaches_level_8')} "
                f"best_actions={post_best.get('actions')} "
                f"score={post_best.get('score')}"
            )
    if bbox_sweep:
        if bbox_sweep.get("skipped"):
            print(f"bbox-a6:     skipped={bbox_sweep['skipped']}")
        else:
            bbox_best = bbox_sweep.get("best") or {}
            print(
                "bbox-a6:     "
                f"candidates={bbox_sweep.get('candidate_count')} "
                f"viable={bbox_sweep.get('viable_candidate_count')} "
                f"new_safe={bbox_sweep.get('novel_safe_states')} "
                f"reaches_level_8={bbox_sweep.get('reaches_level_8')} "
                f"best={bbox_best.get('combined_best_actions')}"
            )
    if bbox_focused:
        if bbox_focused.get("skipped"):
            print(f"focused:     skipped={bbox_focused['skipped']}")
        else:
            focused_best = bbox_focused.get("best") or {}
            print(
                "focused:     "
                f"viable={bbox_focused.get('viable_candidate_count')} "
                f"new_safe={bbox_focused.get('novel_safe_states')} "
                f"reaches_level_8={bbox_focused.get('reaches_level_8')} "
                f"best={focused_best.get('combined_best_actions')}"
            )
    if motif_grid:
        if motif_grid.get("skipped"):
            print(f"motif:       skipped={motif_grid['skipped']}")
        else:
            motif_best = motif_grid.get("best") or {}
            print(
                "motif:       "
                f"targets={motif_grid.get('target_roots')} "
                f"motifs={motif_grid.get('evaluated_motifs')} "
                f"new_safe={motif_grid.get('new_safe_states')} "
                f"action3_distinct={motif_grid.get('action3_distinct_bbox_count')} "
                f"best={motif_best.get('sequence')}"
            )
    if action3_matrix:
        if action3_matrix.get("skipped"):
            print(f"a3-matrix:   skipped={action3_matrix['skipped']}")
        else:
            print(
                "a3-matrix:   "
                f"trials={action3_matrix.get('evaluated_trials')} "
                f"safe_distinct={action3_matrix.get('ACTION3_safe_distinct_bbox_count')} "
                f"fatal_distinct={action3_matrix.get('fatal_distinct_bbox_count')} "
                f"safe_no_distinct={action3_matrix.get('safe_no_distinct_bbox_count')}"
            )
    print(f"json:        {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--traces", type=Path, default=PROJECT_ROOT / "human_traces")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--target-level", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--beam", type=int, default=24)
    parser.add_argument("--danger-window", type=int, default=8)
    parser.add_argument("--repair-window", type=int, default=16)
    parser.add_argument("--repair-points", type=int, default=8)
    parser.add_argument("--repair-horizon", type=int, default=4)
    parser.add_argument("--repair-beam", type=int, default=16)
    parser.add_argument("--curriculum-states", type=int, default=10)
    parser.add_argument("--curriculum-horizon", type=int, default=4)
    parser.add_argument("--curriculum-beam", type=int, default=16)
    parser.add_argument("--sweep-grid-size", type=int, default=8)
    parser.add_argument("--sweep-max-candidates", type=int, default=96)
    parser.add_argument("--post-center-horizon", type=int, default=5)
    parser.add_argument("--post-center-beam", type=int, default=24)
    parser.add_argument("--action6-center-x", type=int, default=31)
    parser.add_argument("--action6-center-y", type=int, default=31)
    parser.add_argument("--bbox-candidate-limit", type=int, default=48)
    parser.add_argument("--bbox-followup-roots", type=int, default=8)
    parser.add_argument("--bbox-followup-horizon", type=int, default=3)
    parser.add_argument("--bbox-followup-beam", type=int, default=12)
    parser.add_argument("--focused-followup-horizon", type=int, default=6)
    parser.add_argument("--focused-followup-beam", type=int, default=24)
    parser.add_argument("--focused-structural-by-step", type=int, default=3)
    parser.add_argument("--focused-component-weight", type=float, default=120.0)
    parser.add_argument("--focused-histogram-weight", type=float, default=0.01)
    parser.add_argument("--motif-target-roots", type=int, default=7)
    parser.add_argument("--motif-max-k", type=int, default=4)
    parser.add_argument("--motif-max-m", type=int, default=4)
    parser.add_argument("--action3-matrix-max-k", type=int, default=12)
    parser.add_argument("--max-click-targets", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=[
            "frontier_beam",
            "suffix_repair",
            "frontier_curriculum",
            "action6_sweep",
            "post_action6_center",
            "bbox_targeted_action6_sweep",
            "bbox_focused_deepening",
            "action3_motif_grid",
            "action3_precondition_matrix",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    full_game_id = _resolve_full_game_id(arc, args.game)
    selection = _load_selected_episode(
        args.traces,
        requested_game=args.game,
        resolved_game_id=full_game_id,
        episode_id=args.episode_id,
        require_win=False,
    )
    frontier = find_level_frontier(
        selection,
        target_level=args.target_level,
        danger_window=args.danger_window,
    )
    base_env, base_raw = _replay_prefix(
        arc,
        full_game_id,
        selection.steps,
        stop_before_index=frontier.frontier_index,
        expected_frame=frontier.frontier_frame,
    )

    payload: Dict[str, Any] = {
        "game": selection.game_id,
        "resolved_game_id": full_game_id,
        "episode_id": selection.episode_id,
        "selected_reason": selection.selection_reason,
        "trace_final_state": str(selection.episode.final_state if selection.episode else ""),
        "trace_levels_completed": int(selection.episode.levels_completed if selection.episode else 0),
        "frontier": {
            "target_level": frontier.target_level,
            "level_start_index": frontier.level_start_index,
            "frontier_index": frontier.frontier_index,
            "terminal_index": frontier.terminal_index,
            "immediate_danger_action": frontier.immediate_danger_action,
            "danger_actions": list(frontier.danger_actions),
            "visited_safe_states": len(frontier.visited_safe_hashes),
            "danger_state_hashes": sorted(frontier.danger_state_hashes),
            "frontier_hash": _hash_grid(frontier.frontier_frame),
        },
        "level_7_hypothesis_extraction": build_hypothesis_report(selection, frontier),
    }

    if args.mode in {"frontier_beam", "all"}:
        payload["frontier_beam"] = run_frontier_beam(
            base_env=base_env,
            base_raw=base_raw,
            full_game_id=full_game_id,
            selection=selection,
            frontier=frontier,
            horizon=args.horizon,
            beam_width=args.beam,
            max_click_targets=args.max_click_targets,
        )

    if args.mode in {"suffix_repair", "all"}:
        payload["counterfactual_suffix_repair"] = run_suffix_repair(
            arc=arc,
            full_game_id=full_game_id,
            selection=selection,
            frontier=frontier,
            repair_window=args.repair_window,
            repair_points=args.repair_points,
            horizon=args.repair_horizon,
            beam_width=args.repair_beam,
            max_click_targets=args.max_click_targets,
        )

    if args.mode in {"frontier_curriculum", "all"}:
        payload["frontier_curriculum"] = run_frontier_curriculum(
            base_env=base_env,
            base_raw=base_raw,
            full_game_id=full_game_id,
            selection=selection,
            frontier=frontier,
            seed_horizon=args.horizon,
            seed_beam_width=args.beam,
            curriculum_states=args.curriculum_states,
            curriculum_horizon=args.curriculum_horizon,
            curriculum_beam_width=args.curriculum_beam,
            max_click_targets=args.max_click_targets,
        )

    if args.mode in {"action6_sweep", "all"}:
        payload["level7_action6_coordinate_sweep"] = run_action6_coordinate_sweep(
            base_env=base_env,
            base_raw=base_raw,
            full_game_id=full_game_id,
            selection=selection,
            frontier=frontier,
            seed_horizon=args.horizon,
            seed_beam_width=args.beam,
            sweep_grid_size=args.sweep_grid_size,
            sweep_max_candidates=args.sweep_max_candidates,
            max_click_targets=args.max_click_targets,
        )

    if args.mode in {"post_action6_center", "all"}:
        payload["post_action6_center_policy_search"] = run_post_action6_center_policy_search(
            base_env=base_env,
            full_game_id=full_game_id,
            frontier=frontier,
            prefix_actions=["ACTION7", "ACTION7", "ACTION7"],
            center={"x": int(args.action6_center_x), "y": int(args.action6_center_y)},
            horizon=args.post_center_horizon,
            beam_width=args.post_center_beam,
            max_click_targets=args.max_click_targets,
        )

    if args.mode in {"bbox_targeted_action6_sweep", "all"}:
        payload["bbox_targeted_action6_sweep"] = run_bbox_targeted_action6_sweep(
            base_env=base_env,
            full_game_id=full_game_id,
            frontier=frontier,
            prefix_actions=["ACTION7", "ACTION7", "ACTION7"],
            center={"x": int(args.action6_center_x), "y": int(args.action6_center_y)},
            candidate_limit=args.bbox_candidate_limit,
            followup_roots=args.bbox_followup_roots,
            followup_horizon=args.bbox_followup_horizon,
            followup_beam_width=args.bbox_followup_beam,
        )

    if args.mode in {"bbox_focused_deepening"}:
        payload["bbox_focused_deepening"] = run_bbox_targeted_action6_sweep(
            base_env=base_env,
            full_game_id=full_game_id,
            frontier=frontier,
            prefix_actions=["ACTION7", "ACTION7", "ACTION7"],
            center={"x": int(args.action6_center_x), "y": int(args.action6_center_y)},
            candidate_limit=args.bbox_candidate_limit,
            followup_roots=7,
            followup_horizon=args.focused_followup_horizon,
            followup_beam_width=args.focused_followup_beam,
            require_structural_by_step=args.focused_structural_by_step,
            component_weight=args.focused_component_weight,
            histogram_weight=args.focused_histogram_weight,
        )

    if args.mode in {"action3_motif_grid"}:
        payload["action3_phase_motif_grid"] = run_action3_phase_motif_grid(
            base_env=base_env,
            full_game_id=full_game_id,
            frontier=frontier,
            prefix_actions=["ACTION7", "ACTION7", "ACTION7"],
            center={"x": int(args.action6_center_x), "y": int(args.action6_center_y)},
            candidate_limit=args.bbox_candidate_limit,
            target_roots=args.motif_target_roots,
            max_k=args.motif_max_k,
            max_m=args.motif_max_m,
        )

    if args.mode in {"action3_precondition_matrix"}:
        payload["action3_precondition_matrix"] = run_action3_precondition_matrix(
            base_env=base_env,
            full_game_id=full_game_id,
            frontier=frontier,
            prefix_actions=["ACTION7", "ACTION7", "ACTION7"],
            center={"x": int(args.action6_center_x), "y": int(args.action6_center_y)},
            candidate_limit=args.bbox_candidate_limit,
            target_roots=7,
            max_k=args.action3_matrix_max_k,
        )

    output = args.json_out
    if output is None:
        suffix = f"{full_game_id}.{selection.episode_id}.level{args.target_level}.json"
        output = DEFAULT_REPORT_DIR / suffix
    _write_report(output, payload)
    _print_report(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
