"""Violation detector audit - understand what dotted_constraint_violations actually measures.

This script analyzes successful level completions (1-6) to understand:
1. What dotted_constraint_violations==0 actually captures
2. Whether the detector predicts successful submits
3. What the real topology of the "dotted line" constraint is

Example:
    python violation_detector_audit.py --game ar25 --episode f0fe23029ffa
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from level7_frontier_recovery import (
    PROJECT_ROOT,
    Arcade,
    OperationMode,
    _resolve_full_game_id,
    _primary_grid,
)
from trace_replay_verifier import _load_selected_episode, _step_action_data, _action_enum
from task_program_guided_level7 import (
    Component,
    _connected_components_for_colors,
    _boundary_violations,
    _get_boundary_violation_components,
    match_score,
)


@dataclass
class ViolationAudit:
    """Detailed audit of what the violation detector sees at a specific state."""

    step_index: int
    level: int
    action_taken: str
    dotted_constraint_violations: int
    boundary_violation_components: List[Dict[str, Any]] = field(default_factory=list)
    all_components: List[Dict[str, Any]] = field(default_factory=list)
    grid_shape: Tuple[int, int] = (64, 64)
    matched_pairs: int = 0
    unmatched_first: int = 0
    unmatched_second: int = 0
    match_score: float = 0.0
    state_after: str = ""

    def to_report(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "level": self.level,
            "action_taken": self.action_taken,
            "dotted_constraint_violations": self.dotted_constraint_violations,
            "boundary_violation_components": self.boundary_violation_components,
            "all_components_count": len(self.all_components),
            "all_components": self.all_components[:5],  # Limit output
            "grid_shape": self.grid_shape,
            "matched_pairs": self.matched_pairs,
            "unmatched_first": self.unmatched_first,
            "unmatched_second": self.unmatched_second,
            "match_score": round(self.match_score, 4),
            "state_after": self.state_after,
        }


def _find_dotted_line_cells(grid: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """
    Attempt to find cells that form the 'dotted line'.
    In ARC-AGI-3, this is typically a line of cells with a specific color
    (often color 5 or similar) that forms a boundary.
    """
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2:
        return []

    height, width = arr.shape
    dotted_cells: List[Tuple[int, int]] = []

    # Common dotted line colors in ARC-AGI-3
    dotted_colors = {5, 6, 7}  # Adjust based on observation

    for y in range(height):
        for x in range(width):
            if int(arr[y, x]) in dotted_colors:
                dotted_cells.append((y, x))

    return dotted_cells


def _audit_single_step(
    step_index: int,
    step: Any,
    grid: Sequence[Sequence[int]],
    pair_colors: Tuple[int, int],
    level: int,
) -> ViolationAudit:
    """Audit a single step's violation detection."""

    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)

    # Get components
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    all_components = first_components + second_components

    # Count violations
    violations = _boundary_violations(all_components, shape)
    violation_comps = _get_boundary_violation_components(all_components, shape)

    # Get match score details
    match = match_score(grid, pair_colors=pair_colors)

    # Try to find dotted line cells
    dotted_line_cells = _find_dotted_line_cells(grid)

    # Convert components to serializable format
    violation_comps_data = [
        {
            "color": c.color,
            "size": c.size,
            "min_y": c.min_y,
            "min_x": c.min_x,
            "max_y": c.max_y,
            "max_x": c.max_x,
            "centroid_y": c.centroid_y,
            "centroid_x": c.centroid_x,
        }
        for c in violation_comps
    ]

    all_comps_data = [
        {
            "color": c.color,
            "size": c.size,
            "min_y": c.min_y,
            "min_x": c.min_x,
            "max_y": c.max_y,
            "max_x": c.max_x,
            "centroid_y": c.centroid_y,
            "centroid_x": c.centroid_x,
        }
        for c in all_components[:10]  # Limit for readability
    ]

    return ViolationAudit(
        step_index=step_index,
        level=level,
        action_taken=getattr(step, "action", "UNKNOWN"),
        dotted_constraint_violations=violations,
        boundary_violation_components=violation_comps_data,
        all_components=all_comps_data,
        grid_shape=shape,
        matched_pairs=match.matched_pairs,
        unmatched_first=match.unmatched_first,
        unmatched_second=match.unmatched_second,
        match_score=match.score,
        state_after=getattr(step, "state", "UNKNOWN"),
    )


def run_detector_audit(
    *,
    game_id: str,
    episode_id: Optional[str] = None,
    max_level: int = 6,
    pair_colors: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Run violation detector audit on successful level completions."""

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
        require_win=False,  # Accept traces that reach level 7 (even if not WIN)
    )

    # Infer pair colors from successful submits if not provided
    if pair_colors is None:
        pair_colors = _infer_pair_colors_from_trace(selection.steps)

    print(f"Auditing {full_game_id} episode {selection.episode_id}")
    print(f"Using pair colors: {pair_colors}")

    # Replay and audit each step
    audits: List[ViolationAudit] = []
    level_transitions: List[Dict[str, Any]] = []

    env = arc.make(full_game_id)
    if env is None:
        raise ValueError(f"Could not make environment for {full_game_id}")
    env.reset()

    current_level = 0

    for i, step in enumerate(selection.steps):
        # Take action
        action = getattr(step, "action", None)

        if action and action != "RESET":
            env.step(
                _action_enum(action),
                data=_step_action_data(full_game_id, step),
            )

        # Get current state
        raw = getattr(env, "observation_space", None)
        level = int(getattr(raw, "levels_completed", 0) or 0)
        grid = _primary_grid(raw)

        # Detect level transitions
        if level > current_level:
            level_transitions.append({
                "step_index": i,
                "from_level": current_level,
                "to_level": level,
                "action_that_caused": action,
                "state_after": getattr(raw, "state", "UNKNOWN"),
            })
            current_level = level

        # Audit every 10 steps and around level transitions
        if i % 10 == 0 or level > 0:
            audit = _audit_single_step(i, step, grid, pair_colors, level)
            audits.append(audit)

        if level >= max_level:
            break

    # Analyze patterns
    submits = [a for a in audits if a.action_taken == "ACTION2"]
    zero_violation_states = [a for a in audits if a.dotted_constraint_violations == 0]
    pre_submit_states = []

    # Find states just before successful submits
    for i, audit in enumerate(audits):
        if audit.action_taken == "ACTION2":
            # Look back for the preceding state
            preceding = [a for a in audits[:i] if a.level == audit.level]
            if preceding:
                pre_submit_states.append({
                    "submit_step": i,
                    "preceding": preceding[-1].to_report(),
                    "submit_action": audit.to_report(),
                })

    return {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "pair_colors": pair_colors,
        "audited_steps": len(audits),
        "level_transitions": level_transitions,
        "submit_actions": len(submits),
        "zero_violation_states": len(zero_violation_states),
        "zero_violation_details": [a.to_report() for a in zero_violation_states[:5]],
        "pre_submit_analysis": pre_submit_states[:5],
        "key_finding": _analyze_detector_accuracy(audits, level_transitions),
    }


def _infer_pair_colors_from_trace(steps: Sequence[Any]) -> Tuple[int, int]:
    """Infer pair colors by looking at grid state changes."""
    # Default fallback
    return (10, 11)


def _analyze_detector_accuracy(
    audits: Sequence[ViolationAudit],
    transitions: Sequence[Dict[str, Any]],
) -> str:
    """Analyze whether the detector predicts level transitions."""

    # Check if zero violations correlate with level transitions
    violations_before_transition = []

    for trans in transitions:
        step_idx = trans["step_index"]
        preceding = [a for a in audits if a.step_index < step_idx and a.level == trans["from_level"]]
        if preceding:
            violations_before_transition.append(preceding[-1].dotted_constraint_violations)

    if not violations_before_transition:
        return "No level transitions found in audited range"

    avg_violations = sum(violations_before_transition) / len(violations_before_transition)
    zero_count = sum(1 for v in violations_before_transition if v == 0)

    if zero_count == len(violations_before_transition):
        return f"ALL {len(violations_before_transition)} transitions happened with dotted_constraint_violations=0 (detector is correct)"
    elif zero_count == 0:
        return f"ALL {len(violations_before_transition)} transitions happened WITH violations (avg={avg_violations:.1f}) - detector is WRONG"
    else:
        return f"Mixed: {zero_count}/{len(violations_before_transition)} transitions at zero violations - detector is PARTIALLY correct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--max-level", type=int, default=6)
    parser.add_argument("--pair-colors", nargs=2, type=int, default=None)
    args = parser.parse_args()

    pair_colors = tuple(args.pair_colors) if args.pair_colors else None

    result = run_detector_audit(
        game_id=args.game,
        episode_id=args.episode_id,
        max_level=args.max_level,
        pair_colors=pair_colors,
    )

    print("\n" + "=" * 80)
    print("VIOLATION DETECTOR AUDIT RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))

    # Save to file
    output_dir = PROJECT_ROOT / "diagnostics" / "rule_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['game_id']}.violation_detector_audit.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
