"""Five-channel surprise signal for V5.

Ported idea from V4 but rewritten to consume V3 schemas directly.
Each channel is a float in [0, 1]; `.total` is the mean clamped to 1.
The surprise field is produced per frame and consumed by the progress
tracker (to bias SP) and the dissent controller (to decide interrupts).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..schemas import FrameDiff, GameObservation
from ..schemas_ext import SurpriseField


def compute_surprise(
    obs: GameObservation,
    diff: Optional[FrameDiff],
    prev_obs: Optional[GameObservation] = None,
    visited_hashes: Optional[Iterable[int]] = None,
) -> SurpriseField:
    """Compute a five-channel surprise field.

    Channels:
      pixel      - proportion of cells that changed
      object     - created + removed objects
      causal     - diff happened but no player displacement (unexplained)
      topology   - object count delta suggests regional change
      semantic   - novel state hash / value bag
    """
    if diff is None:
        return SurpriseField()

    grid = obs.raw_grid
    total_cells = max(1, grid.shape[0] * grid.shape[1])
    pixel = min(1.0, diff.num_changed / float(total_cells))

    object_surprise = min(
        1.0,
        (len(diff.created_objects) + len(diff.removed_objects)) / 6.0,
    )

    # causal: something changed but player didn't move (implies world-change)
    if diff.num_changed > 0 and diff.player_displacement is None:
        causal = min(1.0, diff.num_changed / max(1.0, total_cells / 10.0))
    elif diff.num_changed > 0 and diff.player_displacement == (0, 0):
        causal = 0.2
    else:
        causal = 0.0

    # topology: change in object count
    if prev_obs is not None:
        delta_objs = abs(len(obs.objects) - len(prev_obs.objects))
        topology = min(1.0, delta_objs / 4.0)
    else:
        topology = 0.0

    # semantic: novel state hash
    if visited_hashes is not None:
        semantic = 0.0 if obs.grid_hash in visited_hashes else 0.6
    else:
        semantic = 0.0

    # salient cells: pixel positions that flipped (capped)
    salient = list(diff.changed_cells[:12])

    return SurpriseField(
        pixel_surprise=float(pixel),
        object_surprise=float(object_surprise),
        causal_surprise=float(causal),
        topology_surprise=float(topology),
        semantic_surprise=float(semantic),
        salient_cells=salient,
    )
