"""Extract connected-component objects from ARC grids.

Lightweight flood-fill extraction that identifies distinct coloured objects,
player candidates, and basic spatial relationships.  Runs in <1ms on 64×64.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..schemas import ObjectInfo, PlayerHypothesis


# =====================================================================
# Object extraction
# =====================================================================

def extract_objects(
    grid: np.ndarray,
    background_value: int = 0,
    min_area: int = 1,
    max_area: int = 500,
) -> List[ObjectInfo]:
    """Extract connected components of non-background cells.

    Uses 4-connected flood fill.  Each component becomes an ObjectInfo.
    """
    H, W = grid.shape
    visited: np.ndarray = np.zeros((H, W), dtype=bool)
    objects: List[ObjectInfo] = []
    obj_id = 0

    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val == background_value or visited[r, c]:
                continue
            # Flood-fill from (r, c)
            cells: List[Tuple[int, int]] = []
            queue: deque[Tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                        if int(grid[nr, nc]) == val:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            area = len(cells)
            if area < min_area or area > max_area:
                continue

            rows = [c[0] for c in cells]
            cols = [c[1] for c in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            center = (sum(rows) / area, sum(cols) / area)

            # Shape signature: sorted relative offsets from center
            cr_int, cc_int = int(round(center[0])), int(round(center[1]))
            offsets = tuple(sorted((r - cr_int, c - cc_int) for r, c in cells))

            objects.append(ObjectInfo(
                object_id=obj_id,
                value=val,
                cells=cells,
                bbox=bbox,
                center=center,
                area=area,
                shape_signature=hash(offsets),  # compact hash
            ))
            obj_id += 1

    return objects


# =====================================================================
# Object tracking across frames
# =====================================================================

def match_objects(
    objs_before: List[ObjectInfo],
    objs_after: List[ObjectInfo],
    max_dist: float = 5.0,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Match objects between two frames by value + proximity.

    Returns:
        matched: {before_id: after_id}
        created: [after_ids with no match]
        removed: [before_ids with no match]
    """
    matched: Dict[int, int] = {}
    used_after: Set[int] = set()

    # Group by value for efficient matching
    by_val_after: Dict[int, List[ObjectInfo]] = {}
    for o in objs_after:
        by_val_after.setdefault(o.value, []).append(o)

    for ob in objs_before:
        candidates = by_val_after.get(ob.value, [])
        best_id = None
        best_dist = max_dist + 1
        for oa in candidates:
            if oa.object_id in used_after:
                continue
            dist = ((ob.center[0] - oa.center[0])**2 +
                    (ob.center[1] - oa.center[1])**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = oa.object_id
        if best_id is not None and best_dist <= max_dist:
            matched[ob.object_id] = best_id
            used_after.add(best_id)

    after_ids = {o.object_id for o in objs_after}
    before_ids = {o.object_id for o in objs_before}
    created = sorted(after_ids - set(matched.values()))
    removed = sorted(before_ids - set(matched.keys()))

    return matched, created, removed


# =====================================================================
# Player hypothesis generation
# =====================================================================

def generate_player_hypotheses(
    grid: np.ndarray,
    objects: List[ObjectInfo],
    prev_hypotheses: Optional[List[PlayerHypothesis]] = None,
    displacement: Optional[Tuple[int, int]] = None,
) -> List[PlayerHypothesis]:
    """Generate ranked player hypotheses from the current frame.

    Heuristics:
    - Small objects (area 1-4) are more likely players
    - Objects that moved (if displacement known) get a boost
    - Continuity with previous hypotheses gets a boost
    - Unique-valued objects get a boost
    """
    hypotheses: List[PlayerHypothesis] = []

    # Count objects per value for uniqueness scoring
    val_counts: Dict[int, int] = {}
    for o in objects:
        val_counts[o.value] = val_counts.get(o.value, 0) + 1

    for o in objects:
        if o.area > 9:  # too large to be a player
            continue

        conf = 0.0
        evidence: Dict = {}

        # Small objects are better candidates
        if o.area <= 2:
            conf += 0.3
            evidence["small"] = True
        elif o.area <= 4:
            conf += 0.15

        # Unique value → more likely player
        if val_counts[o.value] == 1:
            conf += 0.25
            evidence["unique_value"] = True

        # Continuity: if previous hypothesis had same value
        if prev_hypotheses:
            for ph in prev_hypotheses:
                if ph.value == o.value:
                    conf += 0.2 * ph.confidence
                    evidence["continuity"] = True
                    break

        # If we know displacement and this object is near expected position
        if displacement and prev_hypotheses:
            dy, dx = displacement
            for ph in prev_hypotheses:
                if ph.value == o.value:
                    expected = (ph.position[0] + dy, ph.position[1] + dx)
                    pos = (int(round(o.center[0])), int(round(o.center[1])))
                    if pos == expected:
                        conf += 0.3
                        evidence["moved_as_expected"] = True

        conf = min(1.0, conf)
        if conf > 0.1:
            hypotheses.append(PlayerHypothesis(
                value=o.value,
                position=(int(round(o.center[0])), int(round(o.center[1]))),
                confidence=conf,
                evidence=evidence,
            ))

    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    return hypotheses[:5]  # top 5
