"""Inspect the current match_score on human trace states.

The goal is to separate a representation bug from a search bug:

* If human pre-level-up states also score as only one matched pair, then the
  current scorer cannot recognize successful/quasi-successful configurations.
* If human states score with multiple pairs, then the search/action space is
  failing to reach those states.

Example:
    python inspect_match_score_on_human_trace.py --game ar25 --pair-colors 10 11 --around-level 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from level7_frontier_recovery import (
    PROJECT_ROOT,
    Arcade,
    OperationMode,
    _hash_grid,
    _resolve_full_game_id,
    find_level_frontier,
)
from task_program_guided_level7 import (
    _grid_changed_cells,
    global_correspondence_score,
    match_score,
)
from trace_replay_verifier import _load_selected_episode
from trace_rule_inference import EpisodeBundle, build_level_segments


DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "rule_inference"


def _parse_pair_colors(parts: Sequence[str]) -> Tuple[int, int]:
    values: List[str] = []
    for item in parts:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if len(values) != 2:
        raise ValueError("--pair-colors must look like '10 11' or '10,11'")
    return (int(values[0]), int(values[1]))


def _score_report(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    label: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    score = match_score(grid, pair_colors=pair_colors)
    global_score = global_correspondence_score(grid, pair_colors=pair_colors)
    report = {
        "label": label,
        "hash": _hash_grid(grid),
        "match_score": score.to_report(),
        "global_correspondence_score": global_score.to_report(),
    }
    if extra:
        report.update(extra)
    return report


def _pre_levelup_reports(
    steps: Sequence[Any],
    *,
    pair_colors: Tuple[int, int],
    max_level: int,
) -> List[Dict[str, Any]]:
    bundle = EpisodeBundle(episode=None, steps=list(steps))
    reports: List[Dict[str, Any]] = []
    for segment in build_level_segments(bundle, max_level=max_level):
        reports.append(
            _score_report(
                segment.level_up_before_grid,
                pair_colors=pair_colors,
                label=f"pre_levelup_{segment.level_number}",
                extra={
                    "level_number": int(segment.level_number),
                    "trace_start_step": int(segment.trace_start_step),
                    "trace_end_step": int(segment.trace_end_step),
                    "level_up_action": segment.level_up_action,
                    "n_actions": len(segment.actions),
                    "action_tail": segment.actions[-12:],
                },
            )
        )
    return reports


def _around_level_reports(
    steps: Sequence[Any],
    *,
    pair_colors: Tuple[int, int],
    level_start_index: int,
    terminal_index: int,
    stride: int,
    window: int,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    action3_count = 0
    previous_action3_grid: Optional[Sequence[Sequence[int]]] = None
    selected_indices = set()
    for idx in range(level_start_index, terminal_index + 1):
        if idx < level_start_index + window:
            selected_indices.add(idx)
        if idx > terminal_index - window:
            selected_indices.add(idx)
        if stride > 0 and (idx - level_start_index) % stride == 0:
            selected_indices.add(idx)

    for idx in range(level_start_index, terminal_index + 1):
        step = steps[idx]
        action = getattr(step, "action", "")
        action3_changed = None
        if action == "ACTION3":
            action3_count += 1
            action3_changed = _grid_changed_cells(step.frame_before, step.frame_after)
            previous_action3_grid = step.frame_after

        if idx not in selected_indices and action3_changed != 0:
            continue

        reports.append(
            _score_report(
                step.frame_before,
                pair_colors=pair_colors,
                label=f"level_window_before_{idx}",
                extra={
                    "trace_index": int(idx),
                    "trace_step": int(step.step),
                    "action": action,
                    "state_after": step.game_state_after,
                    "levels_completed_after": int(step.levels_completed_after),
                    "action3_count_before": int(action3_count - (1 if action == "ACTION3" else 0)),
                    "action3_changed_cells": action3_changed,
                    "previous_action3_hash": _hash_grid(previous_action3_grid) if previous_action3_grid is not None else None,
                },
            )
        )
        if action == "ACTION3":
            reports.append(
                _score_report(
                    step.frame_after,
                    pair_colors=pair_colors,
                    label=f"level_window_after_action3_{idx}",
                    extra={
                        "trace_index": int(idx),
                        "trace_step": int(step.step),
                        "action": action,
                        "state_after": step.game_state_after,
                        "levels_completed_after": int(step.levels_completed_after),
                        "action3_count_after": int(action3_count),
                        "action3_changed_cells": action3_changed,
                    },
                )
            )
    return reports


def _summarize_reports(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    matched = [
        int((item.get("match_score") or {}).get("matched_pairs", 0))
        for item in reports
    ]
    unmatched_second = [
        int((item.get("match_score") or {}).get("unmatched_second", 0))
        for item in reports
    ]
    violations = [
        int((item.get("match_score") or {}).get("dotted_constraint_violations", 0))
        for item in reports
    ]
    global_scores = [
        float((item.get("global_correspondence_score") or {}).get("score", 0.0))
        for item in reports
    ]
    return {
        "count": len(reports),
        "max_matched_pairs": max(matched, default=0),
        "min_unmatched_second": min(unmatched_second, default=0),
        "min_dotted_constraint_violations": min(violations, default=0),
        "max_global_correspondence_score": round(max(global_scores, default=0.0), 4),
        "avg_global_correspondence_score": round(
            sum(global_scores) / max(1, len(global_scores)),
            4,
        ),
        "matched_pairs_hist": {
            str(value): matched.count(value)
            for value in sorted(set(matched))
        },
    }


def _print_report(payload: Dict[str, Any], output: Path) -> None:
    print("=" * 88)
    print("Inspect match_score on human trace")
    print("=" * 88)
    print(f"game:        {payload['game_id']}")
    print(f"episode:     {payload['episode_id']}")
    print(f"pair colors: {payload['pair_colors']}")
    print(f"pre-levelup: {payload['pre_levelup_summary']}")
    print(f"level window:{payload['level_window_summary']}")
    print(f"json:        {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--pair-colors", nargs="+", required=True)
    parser.add_argument("--around-level", type=int, default=7)
    parser.add_argument("--max-level", type=int, default=7)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    pair_colors = _parse_pair_colors(args.pair_colors)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(PROJECT_ROOT / "environment_files"),
    )
    full_game_id = _resolve_full_game_id(arc, args.game)
    selection = _load_selected_episode(
        PROJECT_ROOT / "human_traces",
        requested_game=args.game,
        resolved_game_id=full_game_id,
        episode_id=args.episode_id,
        require_win=False,
    )
    pre_levelup = _pre_levelup_reports(
        selection.steps,
        pair_colors=pair_colors,
        max_level=max(1, int(args.max_level)),
    )
    level_window: List[Dict[str, Any]] = []
    frontier_info: Optional[Dict[str, Any]] = None
    try:
        frontier = find_level_frontier(
            selection,
            target_level=int(args.around_level),
            danger_window=8,
        )
        frontier_info = {
            "level_start_index": int(frontier.level_start_index),
            "terminal_index": int(frontier.terminal_index),
            "immediate_danger_action": frontier.immediate_danger_action,
            "danger_actions": list(frontier.danger_actions),
        }
        level_window = _around_level_reports(
            selection.steps,
            pair_colors=pair_colors,
            level_start_index=frontier.level_start_index,
            terminal_index=frontier.terminal_index,
            stride=max(1, int(args.stride)),
            window=max(0, int(args.window)),
        )
    except Exception as exc:
        frontier_info = {"error": str(exc)}

    payload = {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "pair_colors": list(pair_colors),
        "around_level": int(args.around_level),
        "frontier": frontier_info,
        "pre_levelup_summary": _summarize_reports(pre_levelup),
        "level_window_summary": _summarize_reports(level_window),
        "pre_levelup": pre_levelup,
        "level_window": level_window,
    }
    output = args.report_dir / f"{full_game_id}.match_score_human_trace.colors{pair_colors[0]}_{pair_colors[1]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_report(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
