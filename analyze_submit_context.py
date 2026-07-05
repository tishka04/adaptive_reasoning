"""Analyze submit attempts with detailed spatial context.

Compares failed level 7 submits vs successful 1-6 submits
to find the missing contextual variable.

Example:
    python analyze_submit_context.py --game ar25 --pair-colors 10 11
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
class SubmitContext:
    """Detailed context of a submit attempt."""
    step_index: int
    level: int
    action2_succeeded: bool
    
    # Target info
    target_color: int = 0
    target_size: int = 0
    target_distance: float = 0.0
    target_centroid_y: float = 0.0
    target_centroid_x: float = 0.0
    target_is_unmatched: bool = False
    
    # Spatial context
    nearest_10_distance: float = float('inf')
    nearest_10_centroid_y: float = 0.0
    nearest_10_centroid_x: float = 0.0
    best_10_pair_score: float = 0.0
    
    # Ranking context
    target_rank_among_unmatched_11: int = 0
    is_last_unmatched_singleton: bool = False
    unmatched_11_count: int = 0
    
    # Alignment to existing matched pairs
    alignment_to_nearest_matched_pair: float = 0.0
    spatial_coherence_score: float = 0.0
    
    # Global state
    matched_pairs: int = 0
    unmatched_total: int = 0
    match_score: float = 0.0
    
    # Derived features
    has_plausible_color10_context: bool = False
    is_residual_of_correspondence: bool = False
    
    def to_report(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "level": self.level,
            "action2_succeeded": self.action2_succeeded,
            "target": {
                "color": self.target_color,
                "size": self.target_size,
                "distance": round(self.target_distance, 2),
                "centroid": {"y": round(self.target_centroid_y, 2), "x": round(self.target_centroid_x, 2)},
                "is_unmatched": self.target_is_unmatched,
            },
            "spatial_context": {
                "nearest_10_distance": round(self.nearest_10_distance, 2),
                "nearest_10_centroid": {"y": round(self.nearest_10_centroid_y, 2), "x": round(self.nearest_10_centroid_x, 2)},
                "best_10_pair_score": round(self.best_10_pair_score, 4),
            },
            "ranking_context": {
                "target_rank_among_unmatched_11": self.target_rank_among_unmatched_11,
                "is_last_unmatched_singleton": self.is_last_unmatched_singleton,
                "unmatched_11_count": self.unmatched_11_count,
            },
            "alignment_context": {
                "alignment_to_nearest_matched_pair": round(self.alignment_to_nearest_matched_pair, 4),
                "spatial_coherence_score": round(self.spatial_coherence_score, 4),
            },
            "global_state": {
                "matched_pairs": self.matched_pairs,
                "unmatched_total": self.unmatched_total,
                "match_score": round(self.match_score, 4),
            },
            "derived_features": {
                "has_plausible_color10_context": self.has_plausible_color10_context,
                "is_residual_of_correspondence": self.is_residual_of_correspondence,
            },
        }


def _analyze_submit_context(
    grid: Sequence[Sequence[int]],
    pair_colors: Tuple[int, int],
    step_index: int,
    level: int,
    action2_succeeded: bool,
) -> SubmitContext:
    """Analyze detailed context of a submit attempt."""
    
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    
    # Get match score and components
    match = match_score(grid, pair_colors=pair_colors)
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])  # Color 10
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])  # Color 11
    cursor_components = _connected_components_for_colors(grid, [4, 5])
    
    # Find all pairs
    unused_second: set = set(range(len(second_components)))
    pairs: List[Tuple[Component, Component, float]] = []  # (first, second, score)
    
    for first in first_components:
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if candidates:
            idx, alignment = max(candidates, key=lambda item: float(item[1]["score"]))
            unused_second.remove(idx)
            score = float(alignment["score"])
            pairs.append((first, second_components[idx], score))
    
    # Get cursor position
    cursor_y, cursor_x = 0.0, 0.0
    if cursor_components:
        cursor = cursor_components[0]
        cursor_y, cursor_x = cursor.centroid_y, cursor.centroid_x
    
    # Find all unmatched components
    matched_first = {first_components.index(p[0]) for p in pairs}
    matched_second = {second_components.index(p[1]) for p in pairs}
    
    unmatched_first = [first_components[i] for i in range(len(first_components)) if i not in matched_first]
    unmatched_second = [second_components[i] for i in range(len(second_components)) if i not in matched_second]
    
    # Find target: nearest small unmatched 11
    target = None
    target_dist = float('inf')
    for comp in unmatched_second:
        if comp.size <= 2:  # Small component
            dist = np.hypot(cursor_y - comp.centroid_y, cursor_x - comp.centroid_x)
            if dist < target_dist:
                target_dist = dist
                target = comp
    
    if target is None:
        # Fallback: any unmatched 11
        for comp in unmatched_second:
            dist = np.hypot(cursor_y - comp.centroid_y, cursor_x - comp.centroid_x)
            if dist < target_dist:
                target_dist = dist
                target = comp
    
    # Find nearest 10
    nearest_10 = None
    nearest_10_dist = float('inf')
    for comp in first_components:
        dist = np.hypot(target.centroid_y - comp.centroid_y, target.centroid_x - comp.centroid_x) if target else float('inf')
        if dist < nearest_10_dist:
            nearest_10_dist = dist
            nearest_10 = comp
    
    # Compute best pair score with nearest 10
    best_10_pair_score = 0.0
    if target and nearest_10:
        alignment = _pair_alignment(nearest_10, target, shape)
        best_10_pair_score = float(alignment["score"])
    
    # Rank target among unmatched 11
    unmatched_11_sorted = sorted(unmatched_second, key=lambda c: c.size)
    target_rank = 0
    if target:
        for i, comp in enumerate(unmatched_11_sorted, 1):
            if comp is target:
                target_rank = i
                break
    
    is_last_singleton = len(unmatched_second) == 1 and target and target.size == 1
    
    # Compute alignment to nearest matched pair
    alignment_to_matched = 0.0
    if pairs and target:
        # Find nearest matched pair
        nearest_pair = min(pairs, key=lambda p: np.hypot(
            target.centroid_y - (p[0].centroid_y + p[1].centroid_y)/2,
            target.centroid_x - (p[0].centroid_x + p[1].centroid_x)/2
        ))
        # Compute spatial coherence: how well does target extend the pattern?
        pair_center_y = (nearest_pair[0].centroid_y + nearest_pair[1].centroid_y) / 2
        pair_center_x = (nearest_pair[0].centroid_x + nearest_pair[1].centroid_x) / 2
        target_to_pair_dist = np.hypot(target.centroid_y - pair_center_y, target.centroid_x - pair_center_x)
        alignment_to_matched = 1.0 / (1.0 + target_to_pair_dist / 10.0)  # Normalize
    
    # Spatial coherence: is target positioned like it completes a pattern?
    spatial_coherence = 0.0
    if len(pairs) >= 1 and target:
        # Check if target is positioned relative to existing pairs in a structured way
        avg_pair_distance = np.mean([
            np.hypot(p[0].centroid_y - p[1].centroid_y, p[0].centroid_x - p[1].centroid_x)
            for p in pairs
        ])
        if nearest_10:
            target_10_dist = np.hypot(target.centroid_y - nearest_10.centroid_y, 
                                      target.centroid_x - nearest_10.centroid_x)
            spatial_coherence = 1.0 - abs(target_10_dist - avg_pair_distance) / max(avg_pair_distance, 1.0)
    
    # Derived features
    has_plausible_context = best_10_pair_score >= 2.0 and nearest_10_dist <= 15.0
    is_residual = len(pairs) >= 1 and len(unmatched_second) <= 3
    
    return SubmitContext(
        step_index=step_index,
        level=level,
        action2_succeeded=action2_succeeded,
        target_color=target.color if target else 0,
        target_size=target.size if target else 0,
        target_distance=target_dist,
        target_centroid_y=target.centroid_y if target else 0.0,
        target_centroid_x=target.centroid_x if target else 0.0,
        target_is_unmatched=target is not None and target in unmatched_second if target else False,
        nearest_10_distance=nearest_10_dist,
        nearest_10_centroid_y=nearest_10.centroid_y if nearest_10 else 0.0,
        nearest_10_centroid_x=nearest_10.centroid_x if nearest_10 else 0.0,
        best_10_pair_score=best_10_pair_score,
        target_rank_among_unmatched_11=target_rank,
        is_last_unmatched_singleton=is_last_singleton,
        unmatched_11_count=len(unmatched_second),
        alignment_to_nearest_matched_pair=alignment_to_matched,
        spatial_coherence_score=max(0.0, spatial_coherence),
        matched_pairs=len(pairs),
        unmatched_total=len(unmatched_first) + len(unmatched_second),
        match_score=match.score,
        has_plausible_color10_context=has_plausible_context,
        is_residual_of_correspondence=is_residual,
    )


def analyze_submit_contexts(
    *,
    game_id: str,
    episode_id: Optional[str] = None,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    """Analyze submit contexts for all ACTION2 attempts."""
    
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
    
    # Collect contexts for all pre-ACTION2 states
    contexts: List[SubmitContext] = []
    
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
                    
                    context = _analyze_submit_context(
                        grid, pair_colors, i, level, action2_succeeded
                    )
                    contexts.append(context)
        
        # Take action
        if action and action != "RESET":
            env.step(_action_enum(action), data=getattr(step, "action_data", None) or {})
    
    # Separate by level and success
    level_0_6_success = [c for c in contexts if c.level <= 6 and c.action2_succeeded]
    level_7_attempts = [c for c in contexts if c.level == 7]
    
    # Compare patterns
    comparison = _compare_context_patterns(level_0_6_success, level_7_attempts)
    
    return {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "pair_colors": pair_colors,
        "total_contexts": len(contexts),
        "level_0_6_success": len(level_0_6_success),
        "level_7_attempts": len(level_7_attempts),
        "success_examples": [c.to_report() for c in level_0_6_success[:3]],
        "level_7_examples": [c.to_report() for c in level_7_attempts[:3]] if level_7_attempts else [],
        "pattern_comparison": comparison,
        "suggested_rule": _suggest_context_rule(comparison),
    }


def _compare_context_patterns(
    success: List[SubmitContext],
    level_7: List[SubmitContext],
) -> Dict[str, Any]:
    """Compare patterns between successful 0-6 and level 7 attempts."""
    
    if not success or not level_7:
        return {"error": "Need both success and level 7 examples"}
    
    features = [
        ("matched_pairs", lambda x: x.matched_pairs, "mean"),
        ("unmatched_total", lambda x: x.unmatched_total, "mean"),
        ("best_10_pair_score", lambda x: x.best_10_pair_score, "mean"),
        ("nearest_10_distance", lambda x: x.nearest_10_distance, "mean"),
        ("spatial_coherence_score", lambda x: x.spatial_coherence_score, "mean"),
        ("is_residual_of_correspondence", lambda x: 1 if x.is_residual_of_correspondence else 0, "mean"),
        ("has_plausible_color10_context", lambda x: 1 if x.has_plausible_color10_context else 0, "mean"),
    ]
    
    comparison = {}
    for name, extractor, agg_type in features:
        success_vals = [extractor(s) for s in success]
        level_7_vals = [extractor(l) for l in level_7]
        
        s_agg = np.mean(success_vals) if agg_type == "mean" else max(set(success_vals), key=success_vals.count)
        l_agg = np.mean(level_7_vals) if agg_type == "mean" else max(set(level_7_vals), key=level_7_vals.count)
        
        comparison[name] = {
            "success_0_6": round(s_agg, 4) if isinstance(s_agg, float) else s_agg,
            "level_7": round(l_agg, 4) if isinstance(l_agg, float) else l_agg,
            "discriminates": abs(s_agg - l_agg) > 0.1 if isinstance(s_agg, (int, float)) else s_agg != l_agg,
        }
    
    return comparison


def _suggest_context_rule(comparison: Dict) -> str:
    """Suggest a rule based on context comparison."""
    
    discriminators = [
        k for k, v in comparison.items()
        if isinstance(v, dict) and v.get("discriminates")
    ]
    
    if "spatial_coherence_score" in discriminators:
        return "target must have high spatial_coherence_score (aligned with existing pairs)"
    
    if "is_residual_of_correspondence" in discriminators:
        return "target must be residual of correspondence (few unmatched remaining)"
    
    if "matched_pairs" in discriminators:
        return f"matched_pairs >= {comparison['matched_pairs']['success_0_6']}"
    
    return "analyze individual contexts for pattern"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--pair-colors", nargs=2, type=int, default=[10, 11])
    args = parser.parse_args()
    
    pair_colors = tuple(args.pair_colors)
    
    result = analyze_submit_contexts(
        game_id=args.game,
        episode_id=args.episode_id,
        pair_colors=pair_colors,
    )
    
    print("\n" + "=" * 80)
    print("SUBMIT CONTEXT ANALYSIS")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    # Save
    output_dir = PROJECT_ROOT / "diagnostics" / "rule_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['game_id']}.submit_context.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
