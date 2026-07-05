"""Compute structured diffs between consecutive game frames.

This is the raw material for all mechanic inference — every transition is
decomposed into changed cells, moved/created/removed objects, and player
displacement.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schemas import FrameDiff, ObjectInfo
from .object_extractor import match_objects


def compute_frame_diff(
    grid_before: np.ndarray,
    grid_after: np.ndarray,
    objects_before: List[ObjectInfo],
    objects_after: List[ObjectInfo],
    player_pos_before: Optional[Tuple[int, int]],
    player_pos_after: Optional[Tuple[int, int]],
    game_state: str,
    prev_levels: int,
    curr_levels: int,
) -> FrameDiff:
    """Build a FrameDiff from two frames and their extracted object lists."""

    # ── Cell-level diff ──
    mask = grid_before != grid_after
    changed_rc = list(zip(*np.where(mask)))  # list of (r, c)
    vals_before = [int(grid_before[r, c]) for r, c in changed_rc]
    vals_after = [int(grid_after[r, c]) for r, c in changed_rc]

    # ── Object-level diff ──
    matched, created, removed = match_objects(objects_before, objects_after)

    # Moved objects: matched objects whose centers shifted
    center_before = {o.object_id: o.center for o in objects_before}
    center_after = {o.object_id: o.center for o in objects_after}
    moved: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
    for before_id, after_id in matched.items():
        cb = center_before[before_id]
        ca = center_after[after_id]
        if cb != ca:
            moved.append((
                before_id,
                (int(round(cb[0])), int(round(cb[1]))),
                (int(round(ca[0])), int(round(ca[1]))),
            ))

    # ── Player displacement ──
    player_disp: Optional[Tuple[int, int]] = None
    if player_pos_before is not None and player_pos_after is not None:
        dy = player_pos_after[0] - player_pos_before[0]
        dx = player_pos_after[1] - player_pos_before[1]
        if dy != 0 or dx != 0:
            player_disp = (dy, dx)

    return FrameDiff(
        changed_cells=changed_rc,
        changed_values_before=vals_before,
        changed_values_after=vals_after,
        created_objects=created,
        removed_objects=removed,
        moved_objects=moved,
        player_displacement=player_disp,
        game_over=(game_state == "GAME_OVER"),
        level_complete=(curr_levels > prev_levels),
        num_changed=len(changed_rc),
    )


def diff_region_mask(
    grid_before: np.ndarray,
    grid_after: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask of cells that changed."""
    return grid_before != grid_after


def diff_summary(diff: FrameDiff) -> str:
    """One-line human-readable summary of a FrameDiff."""
    parts = [f"{diff.num_changed} cells changed"]
    if diff.player_displacement:
        parts.append(f"player moved {diff.player_displacement}")
    if diff.created_objects:
        parts.append(f"created obj {diff.created_objects}")
    if diff.removed_objects:
        parts.append(f"removed obj {diff.removed_objects}")
    if diff.moved_objects:
        parts.append(f"{len(diff.moved_objects)} obj moved")
    if diff.game_over:
        parts.append("GAME OVER")
    if diff.level_complete:
        parts.append("LEVEL COMPLETE")
    if diff.is_noop:
        parts.append("NOOP")
    return " | ".join(parts)
