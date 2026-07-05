"""Analyze ACTION3 safety and productivity at level 7.

Classifies ACTION3 into 3 classes:
- fatal: immediate GAME_OVER
- safe_noop_or_cycle: safe but useless (no-op or cycle)
- safe_productive: safe AND produces meaningful grid change

Example:
    python analyze_action3_safety.py --game ar25 --target-level 7 --pair-colors 10 11
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

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
    MatchScore,
)


Action3Class = Literal["fatal", "safe_noop_or_cycle", "safe_productive"]


@dataclass
class Action3Context:
    """Context and outcome of a single ACTION3 execution."""
    source: str  # "human_trace" or "generated_motif"
    step_index: int
    level: int
    
    # Pre-ACTION3 state
    cursor_y: float = 0.0
    cursor_x: float = 0.0
    grid_before: List[List[int]] = field(default_factory=list)
    match_score_before: float = 0.0
    matched_pairs_before: int = 0
    unmatched_total_before: int = 0
    small_11_count: int = 0
    bbox_before: Tuple[int, int, int, int] = (0, 0, 0, 0)  # y_min, x_min, y_max, x_max
    
    # History
    previous_action: str = ""
    previous_action_sequence_3: List[str] = field(default_factory=list)
    had_action6_before: bool = False
    steps_since_action6: int = -1
    action3_count_in_level: int = 0
    
    # Human manifold features
    grid_hash_seen_in_human_trace: bool = False
    distance_to_human_level7_manifold: float = float('inf')
    
    # Post-ACTION3 outcome
    action3_class: Action3Class = "safe_noop_or_cycle"
    died: bool = False
    level_after: int = 0
    grid_after: List[List[int]] = field(default_factory=list)
    match_score_after: float = 0.0
    
    # Effect analysis
    action3_effect_type: str = "no_op"  # no_op, small_shift, distinct_bbox, component_rewrite, game_over
    changed_cells: int = 0
    bbox_delta: Tuple[int, int, int, int] = (0, 0, 0, 0)
    
    def to_report(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "step_index": self.step_index,
            "level": self.level,
            "pre_state": {
                "cursor": {"y": self.cursor_y, "x": self.cursor_x},
                "match_score": round(self.match_score_before, 4),
                "matched_pairs": self.matched_pairs_before,
                "unmatched_total": self.unmatched_total_before,
                "small_11_count": self.small_11_count,
                "bbox": self.bbox_before,
            },
            "history": {
                "previous_action": self.previous_action,
                "previous_action_sequence_3": self.previous_action_sequence_3,
                "had_action6_before": self.had_action6_before,
                "steps_since_action6": self.steps_since_action6,
                "action3_count_in_level": self.action3_count_in_level,
            },
            "manifold": {
                "grid_hash_seen_in_human_trace": self.grid_hash_seen_in_human_trace,
                "distance_to_human_level7_manifold": round(self.distance_to_human_level7_manifold, 4),
            },
            "outcome": {
                "class": self.action3_class,
                "died": self.died,
                "level_after": self.level_after,
                "match_score_after": round(self.match_score_after, 4),
            },
            "effect": {
                "type": self.action3_effect_type,
                "changed_cells": self.changed_cells,
                "bbox_delta": self.bbox_delta,
            },
        }


def _compute_bbox(grid: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
    """Compute bounding box of non-zero cells."""
    arr = np.array(grid)
    if arr.size == 0:
        return (0, 0, 0, 0)
    rows, cols = np.where(arr != 0)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def _bbox_delta(b1: Tuple[int, ...], b2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute delta between two bboxes."""
    return tuple(b2[i] - b1[i] for i in range(len(b1)))


def _analyze_action3_effect(
    grid_before: Sequence[Sequence[int]],
    grid_after: Sequence[Sequence[int]],
    died: bool,
    level_before: int,
    level_after: int,
) -> Tuple[Action3Class, str, int, Tuple[int, int, int, int]]:
    """Analyze the effect of ACTION3 and classify it.
    
    Returns: (action3_class, effect_type, changed_cells, bbox_delta)
    """
    if died:
        return ("fatal", "game_over", 0, (0, 0, 0, 0))
    
    arr_before = np.array(grid_before)
    arr_after = np.array(grid_after)
    
    if arr_before.shape != arr_after.shape:
        # Grid shape changed - major rewrite
        return ("safe_productive", "component_rewrite", -1, (0, 0, 0, 0))
    
    diff = arr_before != arr_after
    changed_cells = int(diff.sum())
    
    if changed_cells == 0:
        return ("safe_noop_or_cycle", "no_op", 0, (0, 0, 0, 0))
    
    # Compute bbox changes
    bbox_before = _compute_bbox(grid_before)
    bbox_after = _compute_bbox(grid_after)
    bbox_delta = _bbox_delta(bbox_before, bbox_after)
    
    # Check if level progressed
    if level_after > level_before:
        return ("safe_productive", "level_up", changed_cells, bbox_delta)
    
    # Check for distinct bbox change
    bbox_distinct = any(abs(d) > 5 for d in bbox_delta)
    if bbox_distinct:
        return ("safe_productive", "distinct_bbox", changed_cells, bbox_delta)
    
    # Small shift or minor change
    if changed_cells <= 3:
        return ("safe_noop_or_cycle", "small_shift", changed_cells, bbox_delta)
    
    # Default: productive if meaningful change
    return ("safe_productive", "component_rewrite", changed_cells, bbox_delta)


def _extract_action3_contexts_from_human_trace(
    *,
    game_id: str,
    episode_id: Optional[str] = None,
    pair_colors: Tuple[int, int],
) -> Tuple[List[Action3Context], set]:
    """Extract ACTION3 contexts from human trace."""
    
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
    
    print(f"Analyzing human trace: {len(selection.steps)} steps from {full_game_id}")
    
    env = arc.make(full_game_id)
    if env is None:
        raise ValueError(f"Could not make environment for {full_game_id}")
    env.reset()
    
    # Phase 1: Build human level 7 manifold index (grid hashes from human trace)
    print("Building human level 7 manifold index...")
    human_level7_hashes: set = set()
    temp_env = arc.make(full_game_id)
    if temp_env:
        temp_env.reset()
        for step in selection.steps:
            action = getattr(step, "action", None)
            raw = getattr(temp_env, "observation_space", None)
            level = int(getattr(raw, "levels_completed", 0) or 0)
            if level == 7:
                grid = _primary_grid(raw)
                if grid is not None:
                    grid_hash = hash(grid.tobytes() if hasattr(grid, 'tobytes') else str(grid))
                    human_level7_hashes.add(grid_hash)
            if action and action != "RESET":
                temp_env.step(_action_enum(action))
    print(f"  Indexed {len(human_level7_hashes)} unique human level 7 states")
    
    # Phase 2: Analyze ACTION3 contexts
    contexts: List[Action3Context] = []
    had_action6 = False
    steps_since_action6 = -1
    action3_count = 0
    
    for i, step in enumerate(selection.steps):
        action = getattr(step, "action", None)
        
        # Get state BEFORE action
        raw = getattr(env, "observation_space", None)
        grid_before = _primary_grid(raw)
        level_before = int(getattr(raw, "levels_completed", 0) or 0)
        
        # Check if this is ACTION3 at level 7
        if action == "ACTION3" and level_before == 7 and grid_before:
            match_before = match_score(grid_before, pair_colors=pair_colors)
            
            # Get cursor position
            cursor_components = _connected_components_for_colors(grid_before, [4, 5])
            cursor_y, cursor_x = 0.0, 0.0
            if cursor_components:
                cursor = cursor_components[0]
                cursor_y, cursor_x = cursor.centroid_y, cursor.centroid_x
            
            # Track previous action sequence (last 3 actions)
            prev_actions_3 = [
                selection.steps[j].action if j >= 0 and hasattr(selection.steps[j], 'action') else ""
                for j in range(i-3, i)
            ]
            
            # Get small 11 count
            color_11_components = _connected_components_for_colors(grid_before, [pair_colors[1]])
            small_11_count = sum(1 for c in color_11_components if c.size <= 2)
            
            # Execute ACTION3
            temp_env = arc.make(full_game_id)
            if temp_env:
                temp_env.reset()
                for j in range(i):
                    s = selection.steps[j]
                    if s.action and s.action != "RESET":
                        temp_env.step(_action_enum(s.action))
                
                temp_env.step(_action_enum("ACTION3"))
                raw_after = getattr(temp_env, "observation_space", None)
                grid_after = _primary_grid(raw_after)
                level_after = int(getattr(raw_after, "levels_completed", 0) or 0)
                died = level_after < level_before or getattr(raw_after, "state", "") == "GAME_OVER"
                
                match_after = match_score(grid_after, pair_colors=pair_colors) if grid_after else MatchScore()
                
                # Analyze effect
                action3_class, effect_type, changed_cells, bbox_delta = _analyze_action3_effect(
                    grid_before, grid_after, died, level_before, level_after
                )
                
                # Compute grid hash and manifold distance
                grid_hash = hash(grid_before.tobytes() if hasattr(grid_before, 'tobytes') else str(grid_before))
                in_human_trace = grid_hash in human_level7_hashes
                
                # For generated motifs, compute distance (for now, binary: 0 if in, inf if not)
                # TODO: implement proper distance metric
                manifold_distance = 0.0 if in_human_trace else float('inf')
                
                context = Action3Context(
                    source="human_trace",
                    step_index=i,
                    level=level_before,
                    cursor_y=cursor_y,
                    cursor_x=cursor_x,
                    grid_before=grid_before.tolist() if hasattr(grid_before, 'tolist') else list(grid_before),
                    match_score_before=match_before.score,
                    matched_pairs_before=match_before.matched_pairs,
                    unmatched_total_before=match_before.unmatched_first + match_before.unmatched_second,
                    small_11_count=small_11_count,
                    bbox_before=_compute_bbox(grid_before),
                    previous_action=selection.steps[i-1].action if i > 0 else "",
                    previous_action_sequence_3=prev_actions_3,
                    had_action6_before=had_action6,
                    steps_since_action6=steps_since_action6,
                    action3_count_in_level=action3_count,
                    grid_hash_seen_in_human_trace=in_human_trace,
                    distance_to_human_level7_manifold=manifold_distance,
                    action3_class=action3_class,
                    died=died,
                    level_after=level_after,
                    grid_after=grid_after.tolist() if hasattr(grid_after, 'tolist') else list(grid_after),
                    match_score_after=match_after.score,
                    action3_effect_type=effect_type,
                    changed_cells=changed_cells,
                    bbox_delta=bbox_delta,
                )
                contexts.append(context)
                action3_count += 1
        
        # Update tracking
        if action == "ACTION6":
            had_action6 = True
            steps_since_action6 = 0
        elif had_action6 and steps_since_action6 >= 0:
            steps_since_action6 += 1
        
        # Take action
        if action and action != "RESET":
            env.step(_action_enum(action))
    
    return contexts, human_level7_hashes


def _extract_action3_from_generated_motifs(
    *,
    game_id: str,
    pair_colors: Tuple[int, int],
    human_manifold_hashes: set,
) -> List[Action3Context]:
    """Extract ACTION3 contexts from generated motif rollouts."""
    import glob
    
    contexts: List[Action3Context] = []
    
    # Find guided search result files
    pattern = str(PROJECT_ROOT / "diagnostics" / "task_program_guided_level7" / f"{game_id}*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"  No guided search results found for {game_id}")
        return contexts
    
    print(f"  Found {len(result_files)} guided search result files")
    
    for result_file in result_files[:5]:  # Limit to first 5 files
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            
            # Get guided_search section
            guided = data.get("guided_search", {})
            if not guided:
                continue
            
            # Collect nodes from various sections (mixed dict/list structure)
            candidate_sections = [
                "best",
                "best_match", 
                "top",
                "top_match",
                "top_action6",
                "top_contradictions",
            ]
            
            nodes = []
            for section in candidate_sections:
                value = guided.get(section)
                if isinstance(value, dict):
                    # Single node (like "best")
                    nodes.append(value)
                elif isinstance(value, list):
                    # List of nodes
                    for item in value:
                        if isinstance(item, dict):
                            nodes.append(item)
            
            # Also check frontier
            frontier = data.get("frontier", [])
            if isinstance(frontier, list):
                for item in frontier:
                    if isinstance(item, dict):
                        nodes.append(item)
            
            # Process nodes with ACTION3
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                    
                actions = node.get("actions", [])
                if not isinstance(actions, list) or "ACTION3" not in actions:
                    continue
                
                level = node.get("level", 0)
                if level != 7:
                    continue
                
                # Get node state
                grid_preview = node.get("grid_preview", [])
                if not grid_preview:
                    continue
                
                # Compute grid hash and manifold distance
                grid_hash = hash(str(grid_preview))
                in_human_trace = grid_hash in human_manifold_hashes
                manifold_distance = 0.0 if in_human_trace else float('inf')
                
                # Classify outcome
                died = node.get("died", False) or node.get("state") == "GAME_OVER"
                action3_class: Action3Class = "fatal" if died else "safe_productive"
                
                # Extract previous actions sequence (last 3 before final ACTION3)
                action3_positions = [i for i, a in enumerate(actions) if a == "ACTION3"]
                if action3_positions:
                    last_a3_pos = action3_positions[-1]
                    prev_actions_3 = actions[max(0, last_a3_pos-3):last_a3_pos]
                else:
                    prev_actions_3 = actions[-4:-1] if len(actions) >= 4 else actions[:-1]
                
                context = Action3Context(
                    source="generated_motif",
                    step_index=node.get("depth", 0),
                    level=level,
                    grid_before=grid_preview,
                    match_score_before=node.get("match_score", 0.0),
                    previous_action=actions[-2] if len(actions) >= 2 else "",
                    previous_action_sequence_3=prev_actions_3,
                    had_action6_before="ACTION6" in actions[:-1],
                    action3_count_in_level=sum(1 for a in actions if a == "ACTION3"),
                    grid_hash_seen_in_human_trace=in_human_trace,
                    distance_to_human_level7_manifold=manifold_distance,
                    action3_class=action3_class,
                    died=died,
                    action3_effect_type="unknown" if not died else "game_over",
                )
                contexts.append(context)
                
        except Exception as e:
            print(f"  Error loading {result_file}: {e}")
            continue
    
    return contexts


def _compare_action3_classes(contexts: List[Action3Context]) -> Dict[str, Any]:
    """Compare features across ACTION3 classes."""
    
    fatal = [c for c in contexts if c.action3_class == "fatal"]
    noop = [c for c in contexts if c.action3_class == "safe_noop_or_cycle"]
    productive = [c for c in contexts if c.action3_class == "safe_productive"]
    
    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    comparison = {
        "counts": {
            "fatal": len(fatal),
            "safe_noop_or_cycle": len(noop),
            "safe_productive": len(productive),
        },
        "features": {
            "had_action6_before": {
                "fatal": avg([1.0 if c.had_action6_before else 0.0 for c in fatal]),
                "safe_noop": avg([1.0 if c.had_action6_before else 0.0 for c in noop]),
                "productive": avg([1.0 if c.had_action6_before else 0.0 for c in productive]),
            },
            "small_11_count": {
                "fatal": avg([c.small_11_count for c in fatal]),
                "safe_noop": avg([c.small_11_count for c in noop]),
                "productive": avg([c.small_11_count for c in productive]),
            },
            "match_score_before": {
                "fatal": avg([c.match_score_before for c in fatal]),
                "safe_noop": avg([c.match_score_before for c in noop]),
                "productive": avg([c.match_score_before for c in productive]),
            },
        },
        "effect_types": {
            "productive": list(set(c.action3_effect_type for c in productive)),
        },
    }
    
    return comparison


def analyze_action3_safety(
    *,
    game_id: str,
    episode_id: Optional[str] = None,
    pair_colors: Tuple[int, int],
    include_generated_motifs: bool = False,
) -> Dict[str, Any]:
    """Analyze ACTION3 safety at level 7."""
    
    # Extract from human trace
    contexts, human_level7_hashes = _extract_action3_contexts_from_human_trace(
        game_id=game_id,
        episode_id=episode_id,
        pair_colors=pair_colors,
    )
    
    # Add generated motifs if requested
    if include_generated_motifs:
        print("Loading generated motifs...")
        generated_contexts = _extract_action3_from_generated_motifs(
            game_id=game_id,
            pair_colors=pair_colors,
            human_manifold_hashes=human_level7_hashes,
        )
        contexts.extend(generated_contexts)
        print(f"  Added {len(generated_contexts)} generated ACTION3 contexts")
    
    # Compare classes
    comparison = _compare_action3_classes(contexts)
    
    # Generate predictive rule hypothesis
    productive = [c for c in contexts if c.action3_class == "safe_productive"]
    noop = [c for c in contexts if c.action3_class == "safe_noop_or_cycle"]
    
    if productive:
        # Key insight: productive ACTION3 happens at low action3_count, before saturation
        avg_count_productive = sum(c.action3_count_in_level for c in productive) / len(productive)
        avg_count_noop = sum(c.action3_count_in_level for c in noop) / len(noop) if noop else 0
        
        rule = f"ACTION3 productive iff (state ∈ human_manifold AND action3_count < {avg_count_noop:.0f})"
        rule += f" [productive avg count: {avg_count_productive:.1f}, noop avg count: {avg_count_noop:.1f}]"
    else:
        rule = "No productive ACTION3 found in trace"
    
    return {
        "experiment": "action3_productivity_level7",
        "game_id": game_id,
        "pair_colors": pair_colors,
        "sources": ["human_trace"],
        "total_action3_analyzed": len(contexts),
        "classification": comparison["counts"],
        "feature_comparison": comparison["features"],
        "productive_effect_types": comparison["effect_types"]["productive"],
        "predictive_rule_hypothesis": rule,
        "examples": {
            "fatal": [c.to_report() for c in contexts if c.action3_class == "fatal"][:2],
            "safe_noop": [c.to_report() for c in contexts if c.action3_class == "safe_noop_or_cycle"][:2],
            "productive": [c.to_report() for c in contexts if c.action3_class == "safe_productive"][:2],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--target-level", type=int, default=7)
    parser.add_argument("--pair-colors", nargs=2, type=int, default=[10, 11])
    parser.add_argument("--include-generated-motifs", action="store_true")
    args = parser.parse_args()
    
    pair_colors = tuple(args.pair_colors)
    
    result = analyze_action3_safety(
        game_id=args.game,
        episode_id=args.episode_id,
        pair_colors=pair_colors,
        include_generated_motifs=args.include_generated_motifs,
    )
    
    print("\n" + "=" * 80)
    print("ACTION3 SAFETY ANALYSIS")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    # Save
    output_dir = PROJECT_ROOT / "diagnostics" / "rule_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['game_id']}.action3_safety.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
