"""Object extraction and tracking for V4."""

from __future__ import annotations

from typing import Optional

import numpy as np

from v3.perception.object_extractor import match_objects

from ..schemas import ObjectInfo, ObservationV4, PlayerHypothesis


class ObjectTracker:
    """Extract and track discrete entities across frames."""

    def extract(self, grid: np.ndarray) -> list[ObjectInfo]:
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        objects: list[ObjectInfo] = []
        obj_id = 0

        for r in range(h):
            for c in range(w):
                value = int(grid[r, c])
                if value == 0 or visited[r, c]:
                    continue

                stack = [(r, c)]
                cells: list[tuple[int, int]] = []
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            if int(grid[nr, nc]) == value:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                area = len(cells)
                rows = [cell[0] for cell in cells]
                cols = [cell[1] for cell in cells]
                center = (sum(rows) / area, sum(cols) / area)
                cr_int, cc_int = int(round(center[0])), int(round(center[1]))
                offsets = sorted((rr - cr_int, cc - cc_int) for rr, cc in cells)
                signature = tuple(v for pair in offsets[:8] for v in pair)

                objects.append(
                    ObjectInfo(
                        object_id=obj_id,
                        value=value,
                        cells=cells,
                        bbox=(min(rows), min(cols), max(rows), max(cols)),
                        center=center,
                        area=area,
                        shape_signature=signature,
                    )
                )
                obj_id += 1

        return objects

    def match(
        self,
        prev_objects: list[ObjectInfo],
        next_objects: list[ObjectInfo],
    ) -> dict[int, int]:
        matched, _, _ = match_objects(prev_objects, next_objects)
        return matched

    def player_hypotheses(
        self,
        prev_obs: Optional[ObservationV4],
        next_objects: list[ObjectInfo],
    ) -> list[PlayerHypothesis]:
        """Maintain multiple candidate ontologies for player identity."""
        val_counts: dict[int, int] = {}
        for obj in next_objects:
            val_counts[obj.value] = val_counts.get(obj.value, 0) + 1

        prev_candidates = prev_obs.player_hypotheses if prev_obs is not None else []
        hypotheses: list[PlayerHypothesis] = []
        for obj in next_objects:
            if obj.area > 9:
                continue

            score = 0.0
            evidence: dict[str, float] = {}
            if obj.area <= 2:
                score += 0.35
                evidence["small"] = 1.0
            elif obj.area <= 4:
                score += 0.20
                evidence["medium"] = 0.5

            if val_counts.get(obj.value, 0) == 1:
                score += 0.25
                evidence["unique_value"] = 1.0

            for prev in prev_candidates:
                if prev.value == obj.value:
                    score += 0.25 * prev.confidence
                    evidence["continuity"] = prev.confidence
                    dist = abs(prev.position[0] - round(obj.center[0])) + abs(
                        prev.position[1] - round(obj.center[1])
                    )
                    if dist <= 2:
                        score += 0.15
                        evidence["near_previous"] = 1.0
                    break

            if score >= 0.1:
                hypotheses.append(
                    PlayerHypothesis(
                        value=obj.value,
                        position=(int(round(obj.center[0])), int(round(obj.center[1]))),
                        confidence=min(1.0, score),
                        evidence=evidence,
                    )
                )

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:5]
