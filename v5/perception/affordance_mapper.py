"""Infer affordances (interaction possibilities) from objects and context.

Maps objects + local context → what can the agent probably DO with them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..schemas import (
    Affordance,
    AffordanceKind,
    GameObservation,
    LocalContext,
    ObjectInfo,
)


def map_affordances(
    objects: List[ObjectInfo],
    grid: np.ndarray,
    player_pos: Optional[Tuple[int, int]] = None,
    danger_map: Optional[np.ndarray] = None,
    known_lethal_values: Optional[set] = None,
    known_collectible_values: Optional[set] = None,
) -> List[Affordance]:
    """Generate affordance annotations for all objects."""
    affordances: List[Affordance] = []
    H, W = grid.shape
    lethal = known_lethal_values or set()
    collectible = known_collectible_values or set()

    for obj in objects:
        # Known lethal
        if obj.value in lethal:
            affordances.append(Affordance(
                kind=AffordanceKind.HAZARDOUS,
                target=obj.object_id,
                confidence=0.9,
            ))
            continue

        # Known collectible
        if obj.value in collectible:
            affordances.append(Affordance(
                kind=AffordanceKind.COLLECTIBLE,
                target=obj.object_id,
                confidence=0.85,
            ))
            continue

        # Danger-map based hazard detection
        if danger_map is not None:
            r, c = int(round(obj.center[0])), int(round(obj.center[1]))
            if 0 <= r < H and 0 <= c < W:
                if float(danger_map[r, c]) > 0.5:
                    affordances.append(Affordance(
                        kind=AffordanceKind.HAZARDOUS,
                        target=obj.object_id,
                        confidence=float(danger_map[r, c]),
                    ))
                    continue

        # Small isolated objects near player → clickable or collectible
        if obj.area <= 3 and player_pos is not None:
            dist = abs(obj.center[0] - player_pos[0]) + abs(obj.center[1] - player_pos[1])
            if dist < 8:
                affordances.append(Affordance(
                    kind=AffordanceKind.CLICKABLE,
                    target=obj.object_id,
                    confidence=0.4,
                ))

        # Large rectangles → potentially movable
        if 4 <= obj.area <= 16:
            br, bc, er, ec = obj.bbox
            width = ec - bc + 1
            height = er - br + 1
            if 0.5 <= width / max(height, 1) <= 2.0:
                affordances.append(Affordance(
                    kind=AffordanceKind.MOVABLE,
                    target=obj.object_id,
                    confidence=0.3,
                ))

        # Default: traversable if not already tagged
        tagged_ids = {a.target for a in affordances}
        if obj.object_id not in tagged_ids:
            affordances.append(Affordance(
                kind=AffordanceKind.UNKNOWN,
                target=obj.object_id,
                confidence=0.2,
            ))

    return affordances


def build_local_contexts(
    grid: np.ndarray,
    positions: List[Tuple[int, int]],
    radius: int = 2,
    danger_map: Optional[np.ndarray] = None,
) -> List[LocalContext]:
    """Build LocalContext descriptors around given positions."""
    H, W = grid.shape
    contexts: List[LocalContext] = []

    for r, c in positions:
        # Extract patch
        r0 = max(0, r - radius)
        r1 = min(H, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(W, c + radius + 1)
        patch = grid[r0:r1, c0:c1].copy()

        # Nearby object values (unique non-zero)
        nearby_vals = sorted(set(int(v) for v in patch.flat if v != 0))

        # Free directions
        free = []
        for name, dr, dc in [("up", -1, 0), ("down", 1, 0),
                              ("left", 0, -1), ("right", 0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == 0:
                free.append(name)

        # Danger
        dscore = 0.0
        if danger_map is not None and 0 <= r < H and 0 <= c < W:
            dscore = float(danger_map[r, c])

        contexts.append(LocalContext(
            center=(r, c),
            patch=patch,
            nearby_object_values=nearby_vals,
            free_directions=free,
            danger_score=dscore,
        ))

    return contexts
