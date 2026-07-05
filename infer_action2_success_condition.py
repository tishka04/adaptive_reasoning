"""Infer ACTION2 success condition from human trace analysis.

This script analyzes successful vs failed ACTION2 attempts to learn
the discriminating factors for submit readiness.

Example:
    python infer_action2_success_condition.py --game ar25 --pair-colors 10 11
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
class StateFeatures:
    """Feature vector extracted from a game state."""
    step_index: int
    level: int
    action_next: str = ""  # What action follows this state
    action2_succeeded: bool = False
    
    # Match features
    matched_pairs: int = 0
    unmatched_first: int = 0
    unmatched_second: int = 0
    unmatched_total: int = 0
    match_score: float = 0.0
    
    # Pair quality features
    best_pair_score: float = 0.0
    mean_pair_score: float = 0.0
    min_pair_score: float = 0.0
    
    # Component features
    first_color_count: int = 0
    second_color_count: int = 0
    total_components: int = 0
    
    # Shape features
    mean_size_ratio: float = 0.0  # size(first) / size(second) for matched pairs
    mean_bbox_iou: float = 0.0
    mean_centroid_distance: float = 0.0
    
    # Alignment features  
    mean_row_alignment: float = 0.0
    mean_col_alignment: float = 0.0
    
    # Cursor feature
    cursor_near_target: float = 0.0
    
    # Shape score distribution
    pair_scores: List[float] = field(default_factory=list)
    
    def to_report(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "level": self.level,
            "action_next": self.action_next,
            "action2_succeeded": self.action2_succeeded,
            "matched_pairs": self.matched_pairs,
            "unmatched_total": self.unmatched_total,
            "match_score": round(self.match_score, 4),
            "best_pair_score": round(self.best_pair_score, 4),
            "mean_pair_score": round(self.mean_pair_score, 4),
            "min_pair_score": round(self.min_pair_score, 4),
            "component_counts": {
                "first_color": self.first_color_count,
                "second_color": self.second_color_count,
                "total": self.total_components,
            },
            "shape_features": {
                "mean_size_ratio": round(self.mean_size_ratio, 4),
                "mean_bbox_iou": round(self.mean_bbox_iou, 4),
                "mean_centroid_distance": round(self.mean_centroid_distance, 4),
                "mean_row_alignment": round(self.mean_row_alignment, 4),
                "mean_col_alignment": round(self.mean_col_alignment, 4),
            },
            "cursor_near_target": round(self.cursor_near_target, 4),
        }


def _extract_state_features(
    grid: Sequence[Sequence[int]],
    pair_colors: Tuple[int, int],
    step_index: int,
    level: int,
    action_next: str = "",
    action2_succeeded: bool = False,
) -> StateFeatures:
    """Extract comprehensive features from a game state."""
    
    # Get match score
    match = match_score(grid, pair_colors=pair_colors)
    
    # Get components
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    
    # Compute pair alignments for all matched pairs
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    
    unused_second: set = set(range(len(second_components)))
    pair_scores: List[float] = []
    size_ratios: List[float] = []
    bbox_ious: List[float] = []
    centroid_dists: List[float] = []
    row_alignments: List[float] = []
    col_alignments: List[float] = []
    
    for first in first_components:
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if not candidates:
            continue
        idx, best = max(candidates, key=lambda item: float(item[1]["score"]))
        unused_second.remove(idx)
        
        pair_scores.append(float(best["score"]))
        
        # Size ratio
        first_size = first.size
        second_size = second_components[idx].size
        if second_size > 0:
            size_ratios.append(first_size / second_size)
        
        # BBox IoU approximation
        first_area = (first.max_y - first.min_y + 1) * (first.max_x - first.min_x + 1)
        second = second_components[idx]
        second_area = (second.max_y - second.min_y + 1) * (second.max_x - second.min_x + 1)
        
        # Intersection
        inter_min_y = max(first.min_y, second.min_y)
        inter_max_y = min(first.max_y, second.max_y)
        inter_min_x = max(first.min_x, second.min_x)
        inter_max_x = min(first.max_x, second.max_x)
        
        if inter_max_y >= inter_min_y and inter_max_x >= inter_min_x:
            inter_area = (inter_max_y - inter_min_y + 1) * (inter_max_x - inter_min_x + 1)
            union_area = first_area + second_area - inter_area
            if union_area > 0:
                bbox_ious.append(inter_area / union_area)
        
        # Centroid distance
        cy1, cx1 = first.centroid_y, first.centroid_x
        cy2, cx2 = second.centroid_y, second.centroid_x
        centroid_dists.append(np.hypot(cy1 - cy2, cx1 - cx2))
        
        # Row/column alignment (normalized)
        row_align = 1.0 - abs(cy1 - cy2) / max(shape[0], 1)
        col_align = 1.0 - abs(cx1 - cx2) / max(shape[1], 1)
        row_alignments.append(row_align)
        col_alignments.append(col_align)
    
    return StateFeatures(
        step_index=step_index,
        level=level,
        action_next=action_next,
        action2_succeeded=action2_succeeded,
        matched_pairs=match.matched_pairs,
        unmatched_first=match.unmatched_first,
        unmatched_second=match.unmatched_second,
        unmatched_total=match.unmatched_first + match.unmatched_second,
        match_score=match.score,
        best_pair_score=max(pair_scores) if pair_scores else 0.0,
        mean_pair_score=np.mean(pair_scores) if pair_scores else 0.0,
        min_pair_score=min(pair_scores) if pair_scores else 0.0,
        first_color_count=len(first_components),
        second_color_count=len(second_components),
        total_components=len(first_components) + len(second_components),
        mean_size_ratio=np.mean(size_ratios) if size_ratios else 0.0,
        mean_bbox_iou=np.mean(bbox_ious) if bbox_ious else 0.0,
        mean_centroid_distance=np.mean(centroid_dists) if centroid_dists else 0.0,
        mean_row_alignment=np.mean(row_alignments) if row_alignments else 0.0,
        mean_col_alignment=np.mean(col_alignments) if col_alignments else 0.0,
        cursor_near_target=match.cursor_near_target,
        pair_scores=pair_scores,
    )


def analyze_trace(
    *,
    game_id: str,
    episode_id: Optional[str] = None,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    """Analyze human trace to infer ACTION2 success condition."""
    
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
    
    # Collect features for all states
    all_states: List[StateFeatures] = []
    current_level = 0
    level_transitions: List[Dict] = []
    
    for i, step in enumerate(selection.steps):
        action = getattr(step, "action", None)
        
        # Get state BEFORE action
        raw = getattr(env, "observation_space", None)
        grid = _primary_grid(raw)
        level = int(getattr(raw, "levels_completed", 0) or 0)
        
        # Determine if next action is ACTION2 and if it succeeded
        action_next = ""
        action2_succeeded = False
        
        if i + 1 < len(selection.steps):
            next_step = selection.steps[i + 1]
            action_next = getattr(next_step, "action", "")
            
            # ACTION2 succeeded if level increased after it
            if action_next == "ACTION2":
                # Look ahead to find level after ACTION2
                temp_env = arc.make(full_game_id)
                if temp_env:
                    # Replay up to this point
                    temp_env.reset()
                    for j in range(i + 1):
                        s = selection.steps[j]
                        if s.action and s.action != "RESET":
                            temp_env.step(_action_enum(s.action))
                    # Now step ACTION2
                    temp_env.step(_action_enum("ACTION2"))
                    next_raw = getattr(temp_env, "observation_space", None)
                    next_level = int(getattr(next_raw, "levels_completed", 0) or 0)
                    action2_succeeded = next_level > level
        
        if grid:
            features = _extract_state_features(
                grid, pair_colors, i, level,
                action_next=action_next,
                action2_succeeded=action2_succeeded,
            )
            all_states.append(features)
        
        # Take the actual action
        if action and action != "RESET":
            env.step(_action_enum(action), data=getattr(step, "action_data", None) or {})
        
        # Track level transitions
        new_raw = getattr(env, "observation_space", None)
        new_level = int(getattr(new_raw, "levels_completed", 0) or 0)
        if new_level > current_level:
            level_transitions.append({
                "step_index": i,
                "from_level": current_level,
                "to_level": new_level,
                "action_that_caused": action,
            })
            current_level = new_level
    
    # Analyze pre-ACTION2 states
    pre_action2_states = [s for s in all_states if s.action_next == "ACTION2"]
    successful_pre_states = [s for s in pre_action2_states if s.action2_succeeded]
    failed_pre_states = [s for s in pre_action2_states if not s.action2_succeeded]
    
    print(f"\nFound {len(pre_action2_states)} pre-ACTION2 states:")
    print(f"  - {len(successful_pre_states)} led to level-up (SUCCESS)")
    print(f"  - {len(failed_pre_states)} did not (FAILED)")
    
    # Learn discriminating thresholds
    thresholds = _learn_discriminating_thresholds(successful_pre_states, failed_pre_states)
    
    # Recalibrate expected values
    calibration = _recalibrate_from_successful_states(successful_pre_states)
    
    return {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "pair_colors": pair_colors,
        "total_states_analyzed": len(all_states),
        "level_transitions": level_transitions,
        "pre_action2_analysis": {
            "total_pre_action2": len(pre_action2_states),
            "successful_pre_states": len(successful_pre_states),
            "failed_pre_states": len(failed_pre_states),
            "successful_examples": [s.to_report() for s in successful_pre_states[:3]],
            "failed_examples": [s.to_report() for s in failed_pre_states[:3]] if failed_pre_states else [],
        },
        "learned_thresholds": thresholds,
        "recalibrated_model": calibration,
        "success_manifold_summary": _compute_success_manifold(successful_pre_states),
    }


def _learn_discriminating_thresholds(
    successful: List[StateFeatures],
    failed: List[StateFeatures],
) -> Dict[str, Any]:
    """Learn thresholds that separate successful from failed states."""
    
    if not successful or not failed:
        return {"error": "Need both successful and failed examples to learn"}
    
    thresholds = {}
    
    # For each feature, find threshold that best separates success from failure
    features_to_check = [
        ("matched_pairs", lambda s: s.matched_pairs, "max"),
        ("unmatched_total", lambda s: s.unmatched_total, "min"),
        ("match_score", lambda s: s.match_score, "max"),
        ("best_pair_score", lambda s: s.best_pair_score, "max"),
        ("mean_pair_score", lambda s: s.mean_pair_score, "max"),
        ("min_pair_score", lambda s: s.min_pair_score, "max"),
        ("mean_bbox_iou", lambda s: s.mean_bbox_iou, "max"),
        ("mean_centroid_distance", lambda s: s.mean_centroid_distance, "min"),
    ]
    
    for name, extractor, mode in features_to_check:
        success_values = [extractor(s) for s in successful]
        failed_values = [extractor(s) for s in failed]
        
        if mode == "max":
            # Threshold is min of successful (must be >= this)
            threshold = min(success_values)
            discriminating = threshold > max(failed_values)
        else:
            # Threshold is max of successful (must be <= this)
            threshold = max(success_values)
            discriminating = threshold < min(failed_values)
        
        thresholds[name] = {
            "threshold": round(threshold, 4),
            "success_range": [round(min(success_values), 4), round(max(success_values), 4)],
            "failed_range": [round(min(failed_values), 4), round(max(failed_values), 4)],
            "perfectly_discriminating": discriminating,
        }
    
    # Find simple rule
    perfect = [k for k, v in thresholds.items() if v.get("perfectly_discriminating")]
    
    return {
        "feature_thresholds": thresholds,
        "perfect_discriminators": perfect,
        "suggested_rule": _suggest_rule(thresholds, perfect),
    }


def _suggest_rule(thresholds: Dict, perfect: List[str]) -> str:
    """Suggest a simple rule based on learned thresholds."""
    
    if "matched_pairs" in perfect:
        t = thresholds["matched_pairs"]["threshold"]
        return f"matched_pairs >= {t}"
    
    if "best_pair_score" in perfect:
        t = thresholds["best_pair_score"]["threshold"]
        return f"best_pair_score >= {t}"
    
    # Fallback: use match_score
    if "match_score" in thresholds:
        t = thresholds["match_score"]["threshold"]
        return f"match_score >= {t}"
    
    return "unknown - insufficient data"


def _recalibrate_from_successful_states(states: List[StateFeatures]) -> Dict[str, Any]:
    """Recalibrate model parameters from successful pre-submit states."""
    
    if not states:
        return {"error": "No successful states to calibrate from"}
    
    matched_pairs = [s.matched_pairs for s in states]
    match_scores = [s.match_score for s in states]
    
    return {
        "old_method": {
            "expected_matched_pairs": "median of all ready states",
            "ready_threshold": "p25 of match scores",
        },
        "new_method": {
            "expected_matched_pairs_min": int(min(matched_pairs)),
            "expected_matched_pairs_mean": round(np.mean(matched_pairs), 2),
            "match_score_min": round(min(match_scores), 4),
            "match_score_p25": round(np.percentile(match_scores, 25), 4),
            "match_score_mean": round(np.mean(match_scores), 4),
            "margin": 0.5,
            "suggested_ready_threshold": round(min(match_scores) - 0.5, 4),
        },
        "calibration_notes": [
            "Use MIN of successful states as threshold, not median",
            "This ensures gate passes ALL states that led to success",
            "Margin allows for slight variations",
        ],
    }


def _compute_success_manifold(states: List[StateFeatures]) -> Dict[str, Any]:
    """Compute statistics about the success manifold."""
    
    if not states:
        return {}
    
    return {
        "count": len(states),
        "matched_pairs_distribution": {
            "values": sorted(set(s.matched_pairs for s in states)),
            "mode": max(set(s.matched_pairs for s in states), 
                       key=lambda x: sum(1 for s in states if s.matched_pairs == x)),
        },
        "key_invariants": [
            "All successful states have specific matched_pairs count",
            "Unmatched_total tends to be low",
            "Pair quality (score) is above threshold",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--pair-colors", nargs=2, type=int, default=[10, 11])
    args = parser.parse_args()
    
    pair_colors = tuple(args.pair_colors)
    
    result = analyze_trace(
        game_id=args.game,
        episode_id=args.episode_id,
        pair_colors=pair_colors,
    )
    
    print("\n" + "=" * 80)
    print("ACTION2 SUCCESS CONDITION ANALYSIS")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    # Save
    output_dir = PROJECT_ROOT / "diagnostics" / "rule_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['game_id']}.action2_success_condition.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
