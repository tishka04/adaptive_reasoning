"""Topology analysis for V4."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from ..schemas import ObservationV4, TopologyState


class TopologyMonitor:
    """Track reachability and region unlocking over time."""

    def __init__(self) -> None:
        self._prev_region_count: int = 0
        self._prev_player_region: Optional[int] = None

    def analyze(self, obs: ObservationV4) -> TopologyState:
        grid = obs.raw_grid
        player = obs.best_player
        traversable_values = {0}
        if player is not None:
            traversable_values.add(player.value)

        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        regions: list[set[tuple[int, int]]] = []
        player_region_id: Optional[int] = None

        for r in range(h):
            for c in range(w):
                if visited[r, c] or int(grid[r, c]) not in traversable_values:
                    continue
                region: set[tuple[int, int]] = set()
                queue: deque[tuple[int, int]] = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    region.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            if int(grid[nr, nc]) in traversable_values:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                region_id = len(regions)
                regions.append(region)
                if player is not None and player.position in region:
                    player_region_id = region_id

        edges: list[tuple[int, int]] = []
        for i, a in enumerate(regions):
            bbox_a = {
                cell
                for r, c in a
                for cell in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            }
            for j in range(i + 1, len(regions)):
                if regions[j] & bbox_a:
                    edges.append((i, j))

        unlocked: list[int] = []
        if len(regions) > self._prev_region_count:
            unlocked.extend(range(self._prev_region_count, len(regions)))
        if player_region_id is not None and self._prev_player_region is not None:
            if player_region_id != self._prev_player_region:
                unlocked.append(player_region_id)

        self._prev_region_count = len(regions)
        self._prev_player_region = player_region_id

        return TopologyState(
            reachable_regions=regions,
            region_graph_edges=edges,
            player_region_id=player_region_id,
            unlocked_regions=sorted(set(unlocked)),
        )
