"""
Grid analysis utilities for ARC-AGI-3 environments.

Provides:
  - Frame parsing and differencing
  - Object detection via connected components
  - Moving-entity tracking (player detection)
  - Feature extraction for the state encoder
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class GridObject:
    """A connected component of same-valued cells."""
    value: int
    cells: Set[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)

    @property
    def center(self) -> Tuple[float, float]:
        my, mx, My, Mx = self.bbox
        return ((my + My) / 2.0, (mx + Mx) / 2.0)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1


@dataclass
class FrameDiff:
    """Summary of what changed between two grids."""
    changed_cells: Set[Tuple[int, int]]
    appeared: Dict[int, Set[Tuple[int, int]]]   # value → new cells
    disappeared: Dict[int, Set[Tuple[int, int]]]  # value → removed cells
    moved_objects: List[Dict[str, Any]]
    num_changes: int
    anything_changed: bool


class GridAnalyzer:
    """Analyzes ARC-AGI-3 game grids for the adaptive reasoning agent."""

    # ------------------------------------------------------------------
    # Frame parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_frame(frame: List[List[List[int]]]) -> np.ndarray:
        """Extract the primary grid from a frame (last grid in the list)."""
        if not frame:
            return np.zeros((1, 1), dtype=np.int32)
        grid = frame[-1] if isinstance(frame, list) else frame
        return np.array(grid, dtype=np.int32)

    @staticmethod
    def parse_all_grids(frame: List[List[List[int]]]) -> List[np.ndarray]:
        """Parse all grids in a frame (some games use multiple layers)."""
        if not frame:
            return [np.zeros((1, 1), dtype=np.int32)]
        return [np.array(g, dtype=np.int32) for g in frame]

    # ------------------------------------------------------------------
    # Grid differencing
    # ------------------------------------------------------------------
    @staticmethod
    def compute_diff(grid1: np.ndarray, grid2: np.ndarray) -> FrameDiff:
        """Compute what changed between two grids."""
        if grid1.shape != grid2.shape:
            # Grid resized — treat everything as changed
            all_cells = set()
            for y in range(grid2.shape[0]):
                for x in range(grid2.shape[1]):
                    all_cells.add((y, x))
            return FrameDiff(
                changed_cells=all_cells,
                appeared={}, disappeared={},
                moved_objects=[], num_changes=len(all_cells),
                anything_changed=True,
            )

        mask = grid1 != grid2
        changed_cells = set(zip(*np.where(mask)))
        num_changes = int(mask.sum())

        appeared: Dict[int, Set[Tuple[int, int]]] = {}
        disappeared: Dict[int, Set[Tuple[int, int]]] = {}

        for y, x in changed_cells:
            old_val = int(grid1[y, x])
            new_val = int(grid2[y, x])
            disappeared.setdefault(old_val, set()).add((y, x))
            appeared.setdefault(new_val, set()).add((y, x))

        # Movement detection: find values that lost cells in one
        # region and gained them in another.  Allow approximate matches
        # (player may leave a trail, push objects, etc).
        moved_objects = []
        for val in set(appeared.keys()) & set(disappeared.keys()):
            app = appeared[val]
            dis = disappeared[val]
            if not app or not dis:
                continue
            # Accept exact match OR close ratio (within 2x)
            ratio = len(app) / len(dis)
            if ratio < 0.5 or ratio > 2.0:
                continue
            # Compute displacement from center of disappeared to center of appeared
            app_center = np.mean(list(app), axis=0)
            dis_center = np.mean(list(dis), axis=0)
            dy = app_center[0] - dis_center[0]
            dx = app_center[1] - dis_center[1]
            if abs(dy) + abs(dx) > 0:
                moved_objects.append({
                    "value": val,
                    "displacement": (float(dy), float(dx)),
                    "from_center": (float(dis_center[0]), float(dis_center[1])),
                    "to_center": (float(app_center[0]), float(app_center[1])),
                    "num_cells": min(len(app), len(dis)),
                })

        return FrameDiff(
            changed_cells=changed_cells,
            appeared=appeared,
            disappeared=disappeared,
            moved_objects=moved_objects,
            num_changes=num_changes,
            anything_changed=num_changes > 0,
        )

    # ------------------------------------------------------------------
    # Object detection (connected components)
    # ------------------------------------------------------------------
    @staticmethod
    def find_objects(
        grid: np.ndarray,
        ignore_values: Optional[Set[int]] = None,
        min_size: int = 1,
        connectivity: int = 4,
    ) -> List[GridObject]:
        """Find connected components in the grid."""
        if ignore_values is None:
            ignore_values = set()

        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        objects: List[GridObject] = []

        if connectivity == 8:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for sy in range(h):
            for sx in range(w):
                if visited[sy, sx]:
                    continue
                val = int(grid[sy, sx])
                if val in ignore_values:
                    visited[sy, sx] = True
                    continue

                # BFS flood fill
                cells: Set[Tuple[int, int]] = set()
                queue = deque([(sy, sx)])
                visited[sy, sx] = True
                min_y, min_x = sy, sx
                max_y, max_x = sy, sx

                while queue:
                    y, x = queue.popleft()
                    cells.add((y, x))
                    min_y = min(min_y, y)
                    min_x = min(min_x, x)
                    max_y = max(max_y, y)
                    max_x = max(max_x, x)

                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            if int(grid[ny, nx]) == val:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                if len(cells) >= min_size:
                    objects.append(GridObject(
                        value=val,
                        cells=cells,
                        bbox=(min_y, min_x, max_y, max_x),
                    ))

        return objects

    # ------------------------------------------------------------------
    # Player / entity detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_player(
        diff: FrameDiff,
        prev_objects: List[GridObject],
        curr_objects: List[GridObject],
    ) -> Optional[Dict[str, Any]]:
        """Identify the player as the object that moved in response to an action."""
        if not diff.moved_objects:
            return None

        # The player is typically a small object that moved
        best = None
        for mo in diff.moved_objects:
            if best is None or mo["num_cells"] < best["num_cells"]:
                best = mo

        if best:
            return {
                "value": best["value"],
                "position": best["to_center"],
                "displacement": best["displacement"],
                "size": best["num_cells"],
            }
        return None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def grid_to_features(grid: np.ndarray, feature_dim: int = 64) -> np.ndarray:
        """Extract a fixed-size feature vector from a grid.

        Features include:
          - Grid dimensions
          - Value histogram (16 bins for values 0-15)
          - Number of objects
          - Largest object size
          - Grid entropy
          - Symmetry scores
        """
        features = np.zeros(feature_dim, dtype=np.float32)
        h, w = grid.shape

        # Dimensions (normalized)
        features[0] = h / 64.0
        features[1] = w / 64.0
        features[2] = (h * w) / (64.0 * 64.0)

        # Value histogram (bins 0-15)
        for v in range(16):
            count = int(np.sum(grid == v))
            features[3 + v] = count / max(h * w, 1)

        # Number of unique values
        features[19] = len(np.unique(grid)) / 16.0

        # Symmetry scores
        # Horizontal symmetry
        flipped_h = np.fliplr(grid)
        features[20] = np.mean(grid == flipped_h)
        # Vertical symmetry
        flipped_v = np.flipud(grid)
        features[21] = np.mean(grid == flipped_v)

        # Entropy
        hist = np.bincount(grid.flatten(), minlength=16).astype(np.float32)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        features[22] = -np.sum(hist * np.log2(hist))

        # Edge density (cells different from their right/down neighbor)
        if h > 1:
            features[23] = np.mean(grid[:-1, :] != grid[1:, :])
        if w > 1:
            features[24] = np.mean(grid[:, :-1] != grid[:, 1:])

        # Object statistics (lightweight — limit to avoid slow BFS on large grids)
        if h * w <= 4096:  # Only for grids up to 64x64
            objects = GridAnalyzer.find_objects(grid, min_size=2)
            features[25] = min(len(objects), 100) / 100.0
            if objects:
                sizes = [o.size for o in objects]
                features[26] = max(sizes) / max(h * w, 1)
                features[27] = np.mean(sizes) / max(h * w, 1)
                features[28] = len(set(o.value for o in objects)) / 16.0

        return features

    # ------------------------------------------------------------------
    # Utility: grid hashing for deduplication
    # ------------------------------------------------------------------
    @staticmethod
    def grid_hash(grid: np.ndarray) -> int:
        """Compute a hash of the grid for state deduplication."""
        return hash(grid.tobytes())

    @staticmethod
    def grids_equal(g1: np.ndarray, g2: np.ndarray) -> bool:
        """Check if two grids are identical."""
        if g1.shape != g2.shape:
            return False
        return bool(np.array_equal(g1, g2))
