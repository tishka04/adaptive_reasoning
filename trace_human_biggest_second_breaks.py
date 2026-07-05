"""Find where the human trace breaks the largest color-11 component.

This diagnostic scans a human episode at a target level and ranks transitions
that either shrink the largest second-color component or increase the number
of second-color components. It is meant to identify the missing spatial
precondition for level-7 fragmentation, not to solve the level directly.

Example:
    python trace_human_biggest_second_breaks.py --game ar25 --pair-colors 10 11
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from compare_human_agent_global_states import _component_brief
from level7_frontier_recovery import (
    PROJECT_ROOT,
    _changed_value_report,
    _components_touching_bbox,
    _diff_bbox,
    _distance_to_center,
    _hash_grid,
    _symmetry_score,
    find_level_frontier,
)
from task_program_guided_level7 import (
    _auto_levelup_state_features,
    _cached_global_correspondence_score,
    _connected_components_for_colors,
    _grid_changed_cells,
    _offending_component_diagnostics,
    match_score,
)
from trace_replay_verifier import _load_selected_episode


DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "human_biggest_second_breaks"


def _parse_pair_colors(parts: Sequence[str]) -> Tuple[int, int]:
    values: List[str] = []
    for item in parts:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if len(values) != 2:
        raise ValueError("--pair-colors must look like '10 11' or '10,11'")
    return (int(values[0]), int(values[1]))


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _second_components_report(
    grid: Sequence[Sequence[int]],
    *,
    second_color: int,
    limit: int = 8,
) -> Dict[str, Any]:
    components = sorted(
        _connected_components_for_colors(grid, [second_color]),
        key=lambda item: item.size,
        reverse=True,
    )
    largest = components[0] if components else None
    return {
        "count": len(components),
        "size_sum": int(sum(component.size for component in components)),
        "largest": _component_brief(largest) if largest is not None else None,
        "top": [_component_brief(component) for component in components[:limit]],
    }


def _boundary_contact(component: Optional[Dict[str, Any]], shape: Tuple[int, int]) -> Dict[str, bool]:
    if not component:
        return {"top": False, "bottom": False, "left": False, "right": False}
    bbox = component.get("bbox") or {}
    height, width = shape
    return {
        "top": int(bbox.get("min_y", 9999)) <= 0,
        "bottom": int(bbox.get("max_y", -1)) >= height - 1,
        "left": int(bbox.get("min_x", 9999)) <= 0,
        "right": int(bbox.get("max_x", -1)) >= width - 1,
    }


def _bbox_intersects(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
    if not left or not right:
        return False
    return not (
        int(left.get("max_y", -1)) < int(right.get("min_y", 9999))
        or int(left.get("min_y", 9999)) > int(right.get("max_y", -1))
        or int(left.get("max_x", -1)) < int(right.get("min_x", 9999))
        or int(left.get("min_x", 9999)) > int(right.get("max_x", -1))
    )


def _cursor_context(grid: Sequence[Sequence[int]], *, colors: Sequence[int] = (4, 5)) -> Dict[str, Any]:
    components = sorted(
        _connected_components_for_colors(grid, colors),
        key=lambda item: item.size,
        reverse=True,
    )
    return {
        "colors": [int(color) for color in colors],
        "count": len(components),
        "top": [_component_brief(component) for component in components[:8]],
    }


def _action_context(
    before: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    arr = np.array(before, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    second = _second_components_report(before, second_color=pair_colors[1])
    largest = second.get("largest")
    offenders = _offending_component_diagnostics(before, pair_colors=pair_colors, limit=10)
    return {
        "grid_hash": _hash_grid(before),
        "shape": [int(shape[0]), int(shape[1])],
        "second_components": second,
        "largest_second_boundary_contact": _boundary_contact(largest, shape),
        "cursor_context": _cursor_context(before),
        "offending_components": {
            "offending_count": offenders.get("offending_count"),
            "largest_offender": offenders.get("largest_offender"),
            "target_components": list(offenders.get("target_components") or [])[:6],
        },
    }


def _transition_report(
    step: Any,
    *,
    trace_index: int,
    level_before: int,
    pair_colors: Tuple[int, int],
    previous_actions: Sequence[str],
    next_actions: Sequence[str],
) -> Dict[str, Any]:
    before = step.frame_before
    after = step.frame_after
    before_match = match_score(before, pair_colors=pair_colors)
    after_match = match_score(after, pair_colors=pair_colors)
    before_global = _cached_global_correspondence_score(before, pair_colors=pair_colors)
    after_global = _cached_global_correspondence_score(after, pair_colors=pair_colors)
    before_features = _auto_levelup_state_features(
        before,
        pair_colors=pair_colors,
        match=before_match,
        global_score=before_global,
    )
    after_features = _auto_levelup_state_features(
        after,
        pair_colors=pair_colors,
        match=after_match,
        global_score=after_global,
    )
    bbox = _diff_bbox(before, after)
    second_before = _second_components_report(before, second_color=pair_colors[1])
    second_after = _second_components_report(after, second_color=pair_colors[1])
    largest_before_bbox = (second_before.get("largest") or {}).get("bbox")
    return {
        "trace_index": int(trace_index),
        "trace_step": int(step.step),
        "action": step.action,
        "action_args": step.action_args,
        "previous_actions": list(previous_actions),
        "next_actions": list(next_actions),
        "intent": step.intent,
        "hypothesis": step.hypothesis,
        "level_before": int(level_before),
        "level_after": int(step.levels_completed_after),
        "state_after": step.game_state_after,
        "level_up": int(step.levels_completed_after) > int(level_before),
        "before_hash": _hash_grid(before),
        "after_hash": _hash_grid(after),
        "changed_cells": int(_grid_changed_cells(before, after)),
        "diff_bbox": bbox,
        "diff_intersects_largest_second_before": _bbox_intersects(bbox, largest_before_bbox),
        "diff_distance_to_center": _distance_to_center(before, bbox),
        "diff_symmetry": _symmetry_score(before, after),
        "changed_values": _changed_value_report(before, after),
        "components_touching_diff_before": _components_touching_bbox(before, bbox, limit=12),
        "components_touching_diff_after": _components_touching_bbox(after, bbox, limit=12),
        "before_context": _action_context(before, pair_colors=pair_colors),
        "after_second_components": second_after,
        "features_before": {
            "second_count": _round(before_features.get("second_count", 0.0)),
            "second_size_sum": _round(before_features.get("second_size_sum", 0.0)),
            "second_largest_size": _round(before_features.get("second_largest_size", 0.0)),
            "largest_offender_size": _round(before_features.get("largest_offender_size", 0.0)),
            "unmatched_total": _round(before_features.get("unmatched_total", 0.0)),
            "dotted_violations": _round(before_features.get("dotted_violations", 0.0)),
            "nearest_distance_mean": _round(before_features.get("nearest_distance_mean", 0.0)),
            "global_score": _round(before_global.score),
            "match_score": _round(before_match.score),
        },
        "features_after": {
            "second_count": _round(after_features.get("second_count", 0.0)),
            "second_size_sum": _round(after_features.get("second_size_sum", 0.0)),
            "second_largest_size": _round(after_features.get("second_largest_size", 0.0)),
            "largest_offender_size": _round(after_features.get("largest_offender_size", 0.0)),
            "unmatched_total": _round(after_features.get("unmatched_total", 0.0)),
            "dotted_violations": _round(after_features.get("dotted_violations", 0.0)),
            "nearest_distance_mean": _round(after_features.get("nearest_distance_mean", 0.0)),
            "global_score": _round(after_global.score),
            "match_score": _round(after_match.score),
        },
        "deltas": {
            "second_largest_drop": _round(
                before_features.get("second_largest_size", 0.0)
                - after_features.get("second_largest_size", 0.0)
            ),
            "largest_offender_drop": _round(
                before_features.get("largest_offender_size", 0.0)
                - after_features.get("largest_offender_size", 0.0)
            ),
            "second_count_gain": _round(
                after_features.get("second_count", 0.0)
                - before_features.get("second_count", 0.0)
            ),
            "second_size_sum_delta": _round(
                after_features.get("second_size_sum", 0.0)
                - before_features.get("second_size_sum", 0.0)
            ),
            "unmatched_total_delta": _round(
                after_features.get("unmatched_total", 0.0)
                - before_features.get("unmatched_total", 0.0)
            ),
            "global_delta": _round(after_global.score - before_global.score),
            "match_delta": _round(after_match.score - before_match.score),
        },
        "second_components_before": second_before,
    }


def _interesting_score(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
    deltas = item.get("deltas") or {}
    return (
        float(deltas.get("second_largest_drop", 0.0)),
        float(deltas.get("largest_offender_drop", 0.0)),
        float(deltas.get("second_count_gain", 0.0)),
        float(item.get("changed_cells", 0)),
    )


def analyze_human_biggest_second_breaks(
    *,
    game: str,
    episode_id: Optional[str],
    target_level: int,
    pair_colors: Tuple[int, int],
    danger_window: int,
    min_largest_drop: float,
    min_count_gain: float,
) -> Dict[str, Any]:
    selection = _load_selected_episode(
        PROJECT_ROOT / "human_traces",
        requested_game=game,
        resolved_game_id=game,
        episode_id=episode_id,
        require_win=False,
    )
    frontier = find_level_frontier(
        selection,
        target_level=target_level,
        danger_window=danger_window,
    )
    steps = list(selection.steps)
    reports: List[Dict[str, Any]] = []
    previous_level = 0
    for index, step in enumerate(steps):
        level_before = previous_level
        previous_level = int(step.levels_completed_after)
        if step.action == "RESET":
            continue
        if index < frontier.level_start_index or index > frontier.terminal_index:
            continue
        if int(level_before) != int(target_level):
            continue
        reports.append(
            _transition_report(
                step,
                trace_index=index,
                level_before=level_before,
                pair_colors=pair_colors,
                previous_actions=[
                    item.action
                    for item in steps[max(frontier.level_start_index, index - 6) : index]
                    if item.action != "RESET"
                ],
                next_actions=[
                    item.action
                    for item in steps[index + 1 : min(frontier.terminal_index + 1, index + 7)]
                    if item.action != "RESET"
                ],
            )
        )

    ranked_by_largest_drop = sorted(
        reports,
        key=lambda item: (
            float((item.get("deltas") or {}).get("second_largest_drop", 0.0)),
            float((item.get("deltas") or {}).get("largest_offender_drop", 0.0)),
            float((item.get("deltas") or {}).get("second_count_gain", 0.0)),
            int(item.get("changed_cells", 0)),
        ),
        reverse=True,
    )
    ranked_by_count_gain = sorted(
        reports,
        key=lambda item: (
            float((item.get("deltas") or {}).get("second_count_gain", 0.0)),
            float((item.get("deltas") or {}).get("second_largest_drop", 0.0)),
            int(item.get("changed_cells", 0)),
        ),
        reverse=True,
    )
    breaks = [
        item
        for item in reports
        if float((item.get("deltas") or {}).get("second_largest_drop", 0.0)) >= float(min_largest_drop)
        or float((item.get("deltas") or {}).get("second_count_gain", 0.0)) >= float(min_count_gain)
    ]
    action_counts = Counter(item["action"] for item in breaks)
    return {
        "game": selection.game_id,
        "episode_id": selection.episode_id,
        "target_level": int(target_level),
        "pair_colors": list(pair_colors),
        "frontier": {
            "level_start_index": int(frontier.level_start_index),
            "terminal_index": int(frontier.terminal_index),
            "danger_actions": list(frontier.danger_actions),
        },
        "thresholds": {
            "min_largest_drop": _round(min_largest_drop),
            "min_count_gain": _round(min_count_gain),
        },
        "transition_count": len(reports),
        "break_count": len(breaks),
        "break_action_counts": dict(action_counts),
        "best_largest_drop": ranked_by_largest_drop[0] if ranked_by_largest_drop else None,
        "best_count_gain": ranked_by_count_gain[0] if ranked_by_count_gain else None,
        "top_largest_drops": ranked_by_largest_drop[:12],
        "top_count_gains": ranked_by_count_gain[:12],
        "break_transitions": sorted(breaks, key=_interesting_score, reverse=True),
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_report(payload: Dict[str, Any], output: Path) -> None:
    print("=" * 88)
    print("Human biggest-second break trace diagnostic")
    print("=" * 88)
    print(f"game:        {payload['game']}")
    print(f"episode:     {payload['episode_id']}")
    print(f"level:       {payload['target_level']}")
    print(f"pair colors: {payload['pair_colors']}")
    print(f"transitions: {payload['transition_count']}")
    print(f"breaks:      {payload['break_count']} actions={payload['break_action_counts']}")

    for label, key in (("best drop", "best_largest_drop"), ("best count", "best_count_gain")):
        item = payload.get(key)
        if not item:
            continue
        deltas = item.get("deltas") or {}
        before = item.get("features_before") or {}
        after = item.get("features_after") or {}
        print(
            f"{label}:   "
            f"idx={item.get('trace_index')} "
            f"action={item.get('action')} "
            f"drop={deltas.get('second_largest_drop')} "
            f"count_gain={deltas.get('second_count_gain')} "
            f"largest={before.get('second_largest_size')}->{after.get('second_largest_size')} "
            f"count={before.get('second_count')}->{after.get('second_count')} "
            f"level_up={item.get('level_up')}"
        )
        bbox = item.get("diff_bbox")
        if bbox:
            print(f"             bbox={bbox}")
    print(f"json:        {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--target-level", type=int, default=7)
    parser.add_argument("--pair-colors", nargs="+", default=["10", "11"])
    parser.add_argument("--danger-window", type=int, default=8)
    parser.add_argument("--min-largest-drop", type=float, default=8.0)
    parser.add_argument("--min-count-gain", type=float, default=4.0)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    pair_colors = _parse_pair_colors(args.pair_colors)
    payload = analyze_human_biggest_second_breaks(
        game=args.game,
        episode_id=args.episode_id,
        target_level=args.target_level,
        pair_colors=pair_colors,
        danger_window=args.danger_window,
        min_largest_drop=args.min_largest_drop,
        min_count_gain=args.min_count_gain,
    )
    colors_suffix = f"colors{pair_colors[0]}_{pair_colors[1]}"
    output = args.report_dir / (
        f"{payload['game']}.{payload['episode_id']}.{colors_suffix}.level{args.target_level}.biggest_second_breaks.json"
    )
    _write_report(output, payload)
    _print_report(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
