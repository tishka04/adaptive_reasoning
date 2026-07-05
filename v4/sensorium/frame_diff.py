"""Frame differ for the V4 sensorium."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..schemas import FrameDiff, ObjectInfo
from .object_tracker import ObjectTracker


class FrameDiffer:
    """Compute structured diffs between two grid observations."""

    def __init__(self) -> None:
        self._tracker = ObjectTracker()

    def diff(
        self,
        prev_grid: np.ndarray,
        next_grid: np.ndarray,
        prev_objects: list[ObjectInfo],
        next_objects: list[ObjectInfo],
        prev_player: Optional[tuple[int, int]],
        next_player: Optional[tuple[int, int]],
        game_state: str,
        prev_levels: int,
        next_levels: int,
    ) -> FrameDiff:
        mask = prev_grid != next_grid
        changed_cells = list(zip(*np.where(mask)))
        before_values = [int(prev_grid[r, c]) for r, c in changed_cells]
        after_values = [int(next_grid[r, c]) for r, c in changed_cells]

        matched = self._tracker.match(prev_objects, next_objects)
        created_ids = sorted(
            obj.object_id for obj in next_objects if obj.object_id not in matched.values()
        )
        removed_ids = sorted(
            obj.object_id for obj in prev_objects if obj.object_id not in matched
        )

        before_centers = {obj.object_id: obj.center for obj in prev_objects}
        after_centers = {obj.object_id: obj.center for obj in next_objects}
        moved_objects: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for before_id, after_id in matched.items():
            cb = before_centers[before_id]
            ca = after_centers[after_id]
            if cb != ca:
                moved_objects.append(
                    (
                        before_id,
                        (int(round(cb[0])), int(round(cb[1]))),
                        (int(round(ca[0])), int(round(ca[1]))),
                    )
                )

        player_displacement: Optional[tuple[int, int]] = None
        if prev_player is not None and next_player is not None:
            dy = next_player[0] - prev_player[0]
            dx = next_player[1] - prev_player[1]
            if dy != 0 or dx != 0:
                player_displacement = (dy, dx)

        level_complete = next_levels > prev_levels
        game_over = game_state == "GAME_OVER"
        return FrameDiff(
            changed_cells=changed_cells,
            before_values=before_values,
            after_values=after_values,
            created_object_ids=created_ids,
            removed_object_ids=removed_ids,
            moved_objects=moved_objects,
            player_displacement=player_displacement,
            is_noop=(not changed_cells and not game_over and not level_complete),
            game_over=game_over,
            level_complete=level_complete,
            num_changed=len(changed_cells),
        )
