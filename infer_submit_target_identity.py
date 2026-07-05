"""Infer which exact target identity enables ACTION2 success.

Analyzes pre-ACTION2 states to identify the specific target component
the cursor must be near for submit to succeed.

Example:
    python infer_submit_target_identity.py --game ar25 --pair-colors 10 11
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
class TargetIdentity:
    """Identity of the target component the cursor is near."""
    step_index: int
    level: int
    action2_succeeded: bool = False
    
    # Cursor position
    cursor_y: float = 0.0
    cursor_x: float = 0.0
    
    # Nearest target info
    nearest_target_color: int = 0
    nearest_target_size: int = 0
    nearest_target_min_y: int = 0
    nearest_target_min_x: int = 0
    nearest_target_max_y: int = 0
    nearest_target_max_x: int = 0
    nearest_target_centroid_y: float = 0.0
    nearest_target_centroid_x: float = 0.0
    distance_to_nearest: float = float('inf')
    
    # Pair context
    is_in_best_pair: bool = False  # Is this target part of the best-aligned pair?
    best_pair_score: float = 0.0
    pair_rank: int = 0  # 1=best, 2=second best, etc.
    
    # Matching context
    is_unmatched: bool = True  # Not yet paired
    is_matched: bool = False   # Already has a partner
    matched_pair_quality: float = 0.0
    
    # Component properties
    component_area: int = 0
    component_aspect_ratio: float = 1.0
    component_touching_boundary: bool = False
    
    # Global state context
    total_unmatched_first: int = 0
    total_unmatched_second: int = 0
    total_matched_pairs: int = 0
    global_match_score: float = 0.0
    
    def to_report(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "level": self.level,
            "action2_succeeded": self.action2_succeeded,
            "cursor": {"y": round(self.cursor_y, 2), "x": round(self.cursor_x, 2)},
            "nearest_target": {
                "color": self.nearest_target_color,
                "size": self.nearest_target_size,
                "area": self.component_area,
                "centroid": {"y": round(self.nearest_target_centroid_y, 2), 
                           "x": round(self.nearest_target_centroid_x, 2)},
                "bbox": [self.nearest_target_min_y, self.nearest_target_min_x,
                        self.nearest_target_max_y, self.nearest_target_max_x],
                "distance_from_cursor": round(self.distance_to_nearest, 2),
            },
            "pair_context": {
                "is_in_best_pair": self.is_in_best_pair,
                "best_pair_score": round(self.best_pair_score, 4),
                "pair_rank": self.pair_rank,
                "is_unmatched": self.is_unmatched,
                "is_matched": self.is_matched,
            },
            "global_context": {
                "unmatched_first": self.total_unmatched_first,
                "unmatched_second": self.total_unmatched_second,
                "matched_pairs": self.total_matched_pairs,
                "match_score": round(self.global_match_score, 4),
            },
        }


def _analyze_target_identity(
    grid: Sequence[Sequence[int]],
    pair_colors: Tuple[int, int],
    step_index: int,
    level: int,
    action2_succeeded: bool,
) -> TargetIdentity:
    """Analyze which target the cursor is nearest to."""
    
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    
    # Get match score and components
    match = match_score(grid, pair_colors=pair_colors)
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    cursor_components = _connected_components_for_colors(grid, [4, 5])
    
    # Cursor position
    cursor_y, cursor_x = 0.0, 0.0
    if cursor_components:
        cursor = cursor_components[0]
        cursor_y, cursor_x = cursor.centroid_y, cursor.centroid_x
    
    # Find all pairs and their scores
    unused_second: set = set(range(len(second_components)))
    pairs: List[Tuple[Component, Component, Dict[str, Any], float]] = []
    
    for first in first_components:
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if not candidates:
            continue
        idx, alignment = max(candidates, key=lambda item: float(item[1]["score"]))
        unused_second.remove(idx)
        score = float(alignment["score"])
        pairs.append((first, second_components[idx], alignment, score))
    
    # Sort pairs by score to get ranking
    pairs.sort(key=lambda p: p[3], reverse=True)
    
    # Find which component (from any pair or unmatched) is nearest to cursor
    all_first = set(range(len(first_components)))
    all_second = set(range(len(second_components)))
    matched_first = set()
    matched_second = set()
    
    for rank, (first, second, align, score) in enumerate(pairs, 1):
        first_idx = first_components.index(first)
        second_idx = second_components.index(second)
        matched_first.add(first_idx)
        matched_second.add(second_idx)
    
    # Find nearest component to cursor
    nearest_comp = None
    nearest_dist = float('inf')
    nearest_is_first = True
    nearest_idx = -1
    
    for i, comp in enumerate(first_components + second_components):
        is_first = i < len(first_components)
        actual_idx = i if is_first else i - len(first_components)
        actual_comp = first_components[actual_idx] if is_first else second_components[actual_idx]
        
        dist = np.hypot(cursor_y - actual_comp.centroid_y, cursor_x - actual_comp.centroid_x)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_comp = actual_comp
            nearest_is_first = is_first
            nearest_idx = actual_idx
    
    # Determine pair context for nearest component
    is_in_best_pair = False
    pair_rank = 0
    best_pair_score = 0.0
    is_unmatched = True
    is_matched = False
    matched_quality = 0.0
    
    if nearest_comp:
        # Check if this component is in any pair
        for rank, (first, second, align, score) in enumerate(pairs, 1):
            if rank == 1:
                best_pair_score = score
            if (nearest_is_first and first is nearest_comp) or \
               (not nearest_is_first and second is nearest_comp):
                is_in_best_pair = (rank == 1)
                pair_rank = rank
                is_unmatched = False
                is_matched = True
                matched_quality = score
                break
    
    # Build result
    result = TargetIdentity(
        step_index=step_index,
        level=level,
        action2_succeeded=action2_succeeded,
        cursor_y=cursor_y,
        cursor_x=cursor_x,
        nearest_target_color=nearest_comp.color if nearest_comp else 0,
        nearest_target_size=nearest_comp.size if nearest_comp else 0,
        nearest_target_min_y=nearest_comp.min_y if nearest_comp else 0,
        nearest_target_min_x=nearest_comp.min_x if nearest_comp else 0,
        nearest_target_max_y=nearest_comp.max_y if nearest_comp else 0,
        nearest_target_max_x=nearest_comp.max_x if nearest_comp else 0,
        nearest_target_centroid_y=nearest_comp.centroid_y if nearest_comp else 0.0,
        nearest_target_centroid_x=nearest_comp.centroid_x if nearest_comp else 0.0,
        distance_to_nearest=nearest_dist,
        is_in_best_pair=is_in_best_pair,
        best_pair_score=best_pair_score,
        pair_rank=pair_rank,
        is_unmatched=is_unmatched,
        is_matched=is_matched,
        matched_pair_quality=matched_quality,
        component_area=(nearest_comp.max_y - nearest_comp.min_y + 1) * 
                      (nearest_comp.max_x - nearest_comp.min_x + 1) if nearest_comp else 0,
        component_aspect_ratio=((nearest_comp.max_x - nearest_comp.min_x + 1) / 
                                max(1, nearest_comp.max_y - nearest_comp.min_y + 1)) 
                                if nearest_comp else 1.0,
        component_touching_boundary=(nearest_comp.min_y <= 0 or nearest_comp.min_x <= 0 or
                                    nearest_comp.max_y >= shape[0] - 1 or 
                                    nearest_comp.max_x >= shape[1] - 1) if nearest_comp else False,
        total_unmatched_first=len(first_components) - len(matched_first),
        total_unmatched_second=len(second_components) - len(matched_second),
        total_matched_pairs=len(pairs),
        global_match_score=match.score,
    )
    
    return result


def analyze_target_identity(
    *,
    game_id: str,
    episode_id: Optional[str] = None,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    """Analyze which target identity enables ACTION2 success."""
    
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
    
    # Collect target identities for all pre-ACTION2 states
    identities: List[TargetIdentity] = []
    
    for i, step in enumerate(selection.steps):
        action = getattr(step, "action", None)
        
        # Get state BEFORE action
        raw = getattr(env, "observation_space", None)
        grid = _primary_grid(raw)
        level = int(getattr(raw, "levels_completed", 0) or 0)
        
        # Check if next action is ACTION2
        if i + 1 < len(selection.steps) and grid:
            next_step = selection.steps[i + 1]
            next_action = getattr(next_step, "action", "")
            
            if next_action == "ACTION2":
                # Determine if this ACTION2 will succeed
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
                    
                    identity = _analyze_target_identity(
                        grid, pair_colors, i, level, action2_succeeded
                    )
                    identities.append(identity)
        
        # Take action
        if action and action != "RESET":
            env.step(_action_enum(action), data=getattr(step, "action_data", None) or {})
    
    # Analyze patterns
    success_identities = [id for id in identities if id.action2_succeeded]
    failed_identities = [id for id in identities if not id.action2_succeeded]
    
    print(f"\nFound {len(identities)} pre-ACTION2 states:")
    print(f"  - {len(success_identities)} succeeded")
    print(f"  - {len(failed_identities)} failed")
    
    # Compare success vs failed patterns
    comparison = _compare_patterns(success_identities, failed_identities)
    
    return {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "pair_colors": pair_colors,
        "total_analyzed": len(identities),
        "success_count": len(success_identities),
        "failed_count": len(failed_identities),
        "success_examples": [id.to_report() for id in success_identities[:3]],
        "failed_examples": [id.to_report() for id in failed_identities[:3]] if failed_identities else [],
        "pattern_comparison": comparison,
        "suggested_rule": _suggest_target_rule(comparison),
    }


def _compare_patterns(
    success: List[TargetIdentity],
    failed: List[TargetIdentity],
) -> Dict[str, Any]:
    """Compare patterns between successful and failed target identities."""
    
    if not success or not failed:
        return {"error": "Need both success and failed examples"}
    
    comparison = {}
    
    # Compare each relevant feature
    features = [
        ("distance_to_nearest", lambda x: x.distance_to_nearest, "min", "lower_is_better"),
        ("is_in_best_pair", lambda x: 1 if x.is_in_best_pair else 0, "mean", "higher_is_better"),
        ("pair_rank", lambda x: x.pair_rank, "mean", "lower_is_better"),
        ("is_unmatched", lambda x: 1 if x.is_unmatched else 0, "mean", "higher_is_better"),
        ("nearest_target_color", lambda x: x.nearest_target_color, "mode", "exact_match"),
        ("best_pair_score", lambda x: x.best_pair_score, "min", "higher_is_better"),
    ]
    
    for name, extractor, agg_type, better in features:
        success_vals = [extractor(s) for s in success]
        failed_vals = [extractor(f) for f in failed]
        
        if agg_type == "min":
            s_agg, f_agg = min(success_vals), min(failed_vals)
        elif agg_type == "mean":
            s_agg, f_agg = np.mean(success_vals), np.mean(failed_vals)
        elif agg_type == "mode":
            s_agg = max(set(success_vals), key=success_vals.count)
            f_agg = max(set(failed_vals), key=failed_vals.count)
        
        comparison[name] = {
            "success": round(s_agg, 4) if isinstance(s_agg, float) else s_agg,
            "failed": round(f_agg, 4) if isinstance(f_agg, float) else f_agg,
            "discriminates": s_agg != f_agg,
        }
    
    return comparison


def _suggest_target_rule(comparison: Dict) -> str:
    """Suggest a rule based on pattern comparison."""
    
    perfect_discriminators = [
        k for k, v in comparison.items() 
        if isinstance(v, dict) and v.get("discriminates")
    ]
    
    if "is_in_best_pair" in perfect_discriminators:
        return "cursor must be near component in BEST pair (highest alignment)"
    
    if "is_unmatched" in perfect_discriminators:
        return "cursor must be near UNMATCHED component"
    
    if "distance_to_nearest" in perfect_discriminators:
        return "cursor must be very close to target (< threshold)"
    
    if "nearest_target_color" in perfect_discriminators:
        return f"cursor must be near specific color: {comparison['nearest_target_color']['success']}"
    
    return "analyze per-level examples for pattern"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--pair-colors", nargs=2, type=int, default=[10, 11])
    args = parser.parse_args()
    
    pair_colors = tuple(args.pair_colors)
    
    result = analyze_target_identity(
        game_id=args.game,
        episode_id=args.episode_id,
        pair_colors=pair_colors,
    )
    
    print("\n" + "=" * 80)
    print("SUBMIT TARGET IDENTITY ANALYSIS")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    # Save
    output_dir = PROJECT_ROOT / "diagnostics" / "rule_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['game_id']}.submit_target_identity.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
