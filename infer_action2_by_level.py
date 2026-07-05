"""Infer ACTION2 success condition per level.

Compare failed vs successful ACTION2 attempts at the SAME level
to find what actually discriminates success.

Example:
    python infer_action2_by_level.py --game ar25 --pair-colors 10 11
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trace_replay_verifier import (
    PROJECT_ROOT,
    Arcade,
    OperationMode,
    _resolve_full_game_id,
    _load_selected_episode,
    _action_enum,
    _primary_grid,
)
from task_program_guided_level7 import (
    match_score,
    _connected_components_for_colors,
    _pair_alignment,
    Component,
)


@dataclass
class PerLevelAnalysis:
    """Analysis of ACTION2 attempts at a specific level."""
    level: int
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    successful_attempt: Optional[Dict[str, Any]] = None
    
    # Delta from last failure to success
    delta_to_success: Dict[str, Any] = field(default_factory=dict)
    
    def to_report(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "failed_count": len(self.failed_attempts),
            "failed_examples": self.failed_attempts[-3:] if self.failed_attempts else [],  # Last 3 failures
            "successful": self.successful_attempt,
            "delta_last_failure_to_success": self.delta_to_success,
        }


def _extract_detailed_state(
    grid: Sequence[Sequence[int]],
    pair_colors: Tuple[int, int],
    step_index: int,
    level: int,
    action_history: List[str] = None,
) -> Dict[str, Any]:
    """Extract detailed state features including cursor/component info."""
    
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    
    # Get match score
    match = match_score(grid, pair_colors=pair_colors)
    
    # Get components by color
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    cursor_components = _connected_components_for_colors(grid, [4, 5])  # Cursor colors
    
    # Find cursor position (centroid of cursor component)
    cursor_pos = None
    if cursor_components:
        cursor = cursor_components[0]
        cursor_pos = (cursor.centroid_y, cursor.centroid_x)
    
    # Find best pair and cursor-to-pair distance
    best_pair_info = None
    cursor_to_best_pair_dist = float('inf')
    
    unused_second: set = set(range(len(second_components)))
    for first in first_components:
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if not candidates:
            continue
        idx, alignment = max(candidates, key=lambda item: float(item[1]["score"]))
        unused_second.remove(idx)
        
        second = second_components[idx]
        
        if cursor_pos:
            # Distance from cursor to pair center
            pair_cy = (first.centroid_y + second.centroid_y) / 2
            pair_cx = (first.centroid_x + second.centroid_x) / 2
            dist = np.hypot(cursor_pos[0] - pair_cy, cursor_pos[1] - pair_cx)
            
            if dist < cursor_to_best_pair_dist:
                cursor_to_best_pair_dist = dist
                best_pair_info = {
                    "first": {"y": first.centroid_y, "x": first.centroid_x, "size": first.size},
                    "second": {"y": second.centroid_y, "x": second.centroid_x, "size": second.size},
                    "pair_center": {"y": pair_cy, "x": pair_cx},
                    "alignment_score": float(alignment["score"]),
                }
    
    # Find unmatched components
    if best_pair_info:
        unmatched_first = [c for c in first_components if not (
            c.centroid_y == best_pair_info["first"]["y"] and c.centroid_x == best_pair_info["first"]["x"]
        )]
    else:
        unmatched_first = list(first_components)
    unmatched_second = list(second_components)
    
    # Check if cursor is near any unmatched component
    cursor_near_unmatched = False
    cursor_to_unmatched_dist = float('inf')
    if cursor_pos:
        for comp in first_components + second_components:
            dist = np.hypot(cursor_pos[0] - comp.centroid_y, cursor_pos[1] - comp.centroid_x)
            if dist < cursor_to_unmatched_dist:
                cursor_to_unmatched_dist = dist
            # Consider "near" if within reasonable distance
            if dist < 8.0:
                cursor_near_unmatched = True
    
    return {
        "step_index": step_index,
        "level": level,
        "match_score": round(match.score, 4),
        "matched_pairs": match.matched_pairs,
        "unmatched_total": match.unmatched_first + match.unmatched_second,
        "cursor_near_target": int(match.cursor_near_target),
        "cursor_pos": {"y": round(cursor_pos[0], 2), "x": round(cursor_pos[1], 2)} if cursor_pos else None,
        "cursor_to_best_pair_dist": round(cursor_to_best_pair_dist, 2),
        "cursor_to_unmatched_dist": round(cursor_to_unmatched_dist, 2),
        "cursor_near_unmatched": cursor_near_unmatched,
        "best_pair": best_pair_info,
        "component_counts": {
            "first_color": len(first_components),
            "second_color": len(second_components),
            "cursor": len(cursor_components),
        },
        "action_history_last3": action_history[-3:] if action_history else [],
    }


def analyze_by_level(
    *,
    game_id: str,
    episode_id: Optional[str] = None,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    """Analyze ACTION2 success condition per level."""
    
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(PROJECT_ROOT / "environment_files"),
    )
    full_game_id = _resolve_full_game_id(arc, game_id)
    selection = _load_selected_episode(
        PROJECT_ROOT / "human_traces",
        requested_game=game_id,
        resolved_game_id=full_game_id,
        episode_id=episode_id,
        require_win=False,
    )
    
    print(f"Analyzing {len(selection.steps)} steps from {full_game_id}")
    
    env = arc.make(full_game_id)
    if env is None:
        raise ValueError(f"Could not make environment for {full_game_id}")
    env.reset()
    
    # Replay and collect states
    level_analyses: Dict[int, PerLevelAnalysis] = {}
    current_level = 0
    action_history: List[str] = []
    
    for i, step in enumerate(selection.steps):
        action = getattr(step, "action", None)
        
        # Get state BEFORE action
        raw = getattr(env, "observation_space", None)
        grid = _primary_grid(raw)
        level = int(getattr(raw, "levels_completed", 0) or 0)
        
        if level not in level_analyses:
            level_analyses[level] = PerLevelAnalysis(level=level)
        
        # Check if next action is ACTION2
        if i + 1 < len(selection.steps):
            next_step = selection.steps[i + 1]
            next_action = getattr(next_step, "action", "")
            
            if next_action == "ACTION2" and grid:
                state = _extract_detailed_state(
                    grid, pair_colors, i, level,
                    action_history=action_history[:],
                )
                
                # Determine if this ACTION2 will succeed
                # Look ahead to check level after ACTION2
                temp_env = arc.make(full_game_id)
                if temp_env:
                    temp_env.reset()
                    for j in range(i + 1):
                        s = selection.steps[j]
                        if s.action and s.action != "RESET":
                            temp_env.step(_action_enum(s.action))
                    temp_env.step(_action_enum("ACTION2"))
                    next_raw = getattr(temp_env, "observation_space", None)
                    next_level = int(getattr(next_raw, "levels_completed", 0) or 0)
                    action2_succeeded = next_level > level
                    
                    state["action2_will_succeed"] = action2_succeeded
                    
                    if action2_succeeded:
                        level_analyses[level].successful_attempt = state
                    else:
                        level_analyses[level].failed_attempts.append(state)
        
        # Take action
        if action and action != "RESET":
            env.step(_action_enum(action), data=getattr(step, "action_data", None) or {})
            action_history.append(action)
        
        # Track level changes
        new_raw = getattr(env, "observation_space", None)
        new_level = int(getattr(new_raw, "levels_completed", 0) or 0)
        if new_level > current_level:
            current_level = new_level
    
    # Compute deltas and per-level insights
    per_level_reports = []
    global_insights = {
        "cursor_near_target_pattern": [],
        "cursor_to_unmatched_dist_pattern": [],
        "last_action_pattern": [],
    }
    
    for level, analysis in sorted(level_analyses.items()):
        if analysis.successful_attempt and analysis.failed_attempts:
            last_failure = analysis.failed_attempts[-1]
            success = analysis.successful_attempt
            
            # Compute delta
            delta = {
                "match_score_delta": round(success["match_score"] - last_failure["match_score"], 4),
                "unmatched_delta": success["unmatched_total"] - last_failure["unmatched_total"],
                "cursor_near_target_change": success["cursor_near_target"] - last_failure["cursor_near_target"],
                "cursor_to_unmatched_dist_delta": round(
                    success["cursor_to_unmatched_dist"] - last_failure["cursor_to_unmatched_dist"], 2
                ),
                "last_failure_step": last_failure["step_index"],
                "success_step": success["step_index"],
                "steps_between": success["step_index"] - last_failure["step_index"],
            }
            analysis.delta_to_success = delta
            
            # Track patterns
            global_insights["cursor_near_target_pattern"].append({
                "level": level,
                "failure": last_failure["cursor_near_target"],
                "success": success["cursor_near_target"],
            })
            global_insights["cursor_to_unmatched_dist_pattern"].append({
                "level": level,
                "failure": last_failure["cursor_to_unmatched_dist"],
                "success": success["cursor_to_unmatched_dist"],
                "delta": round(success["cursor_to_unmatched_dist"] - last_failure["cursor_to_unmatched_dist"], 2),
            })
            global_insights["last_action_pattern"].append({
                "level": level,
                "last_action_before_failure": last_failure["action_history_last3"][-1] if last_failure["action_history_last3"] else None,
                "last_action_before_success": success["action_history_last3"][-1] if success["action_history_last3"] else None,
            })
        
        per_level_reports.append(analysis.to_report())
    
    # Extract common patterns
    common_patterns = _extract_common_patterns(global_insights)
    
    return {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "pair_colors": pair_colors,
        "per_level_analysis": per_level_reports,
        "global_patterns": global_insights,
        "common_patterns": common_patterns,
        "suggested_condition": _suggest_condition(common_patterns),
    }


def _extract_common_patterns(insights: Dict) -> Dict[str, Any]:
    """Extract common patterns across levels."""
    
    patterns = {}
    
    # Cursor near target pattern
    if insights["cursor_near_target_pattern"]:
        all_success_have_target = all(p["success"] > 0 for p in insights["cursor_near_target_pattern"])
        all_failure_no_target = all(p["failure"] == 0 for p in insights["cursor_near_target_pattern"])
        patterns["cursor_near_target_discriminates"] = all_success_have_target and all_failure_no_target
    
    # Cursor distance pattern
    if insights["cursor_to_unmatched_dist_pattern"]:
        dist_deltas = [p["delta"] for p in insights["cursor_to_unmatched_dist_pattern"]]
        patterns["cursor_distance_improves"] = all(d < 0 for d in dist_deltas)  # Negative = closer
        patterns["mean_distance_delta"] = round(np.mean(dist_deltas), 2) if dist_deltas else 0
    
    return patterns


def _suggest_condition(patterns: Dict) -> str:
    """Suggest a condition based on patterns."""
    
    if patterns.get("cursor_near_target_discriminates"):
        return "cursor_near_target > 0 AND distance_to_unmatched_components < threshold"
    
    if patterns.get("cursor_distance_improves"):
        return "cursor_distance_to_target_components <= learned_threshold"
    
    return "unknown - analyze per-level deltas"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--pair-colors", nargs=2, type=int, default=[10, 11])
    args = parser.parse_args()
    
    pair_colors = tuple(args.pair_colors)
    
    result = analyze_by_level(
        game_id=args.game,
        episode_id=args.episode_id,
        pair_colors=pair_colors,
    )
    
    print("\n" + "=" * 80)
    print("PER-LEVEL ACTION2 SUCCESS ANALYSIS")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    # Save
    output_dir = PROJECT_ROOT / "diagnostics" / "rule_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['game_id']}.action2_by_level.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
