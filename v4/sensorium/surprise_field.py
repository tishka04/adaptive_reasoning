"""Surprise estimation for V4."""

from __future__ import annotations

from typing import Iterable, Optional

from ..schemas import Effect, FrameDiff, SurpriseField


class SurpriseFieldBuilder:
    """Aggregate multiple kinds of surprise into a single field."""

    def build(
        self,
        predicted_effects: Optional[Iterable[Effect]],
        observed_diff: Optional[FrameDiff],
        topology_delta: float,
        semantic_novelty: float = 0.0,
    ) -> SurpriseField:
        if observed_diff is None:
            return SurpriseField()

        predicted = list(predicted_effects or [])
        pixel_surprise = min(1.0, observed_diff.num_changed / 25.0)
        object_events = (
            len(observed_diff.created_object_ids)
            + len(observed_diff.removed_object_ids)
            + len(observed_diff.moved_objects)
        )
        object_surprise = min(1.0, object_events / 8.0)

        mismatch = 0.0
        if predicted:
            mismatches = 0
            for effect in predicted:
                if effect.kind == "player_displacement":
                    expected = effect.args.get("delta")
                    if expected != observed_diff.player_displacement:
                        mismatches += 1
                elif effect.kind == "noop":
                    if not observed_diff.is_noop:
                        mismatches += 1
                elif effect.kind == "grid_change":
                    threshold = int(effect.args.get("min_cells", 1))
                    if observed_diff.num_changed < threshold:
                        mismatches += 1
            mismatch = mismatches / max(len(predicted), 1)

        salient_cells = list(observed_diff.changed_cells[:20])
        return SurpriseField(
            pixel_surprise=pixel_surprise,
            object_surprise=object_surprise,
            causal_surprise=min(1.0, mismatch),
            topology_surprise=min(1.0, topology_delta),
            semantic_surprise=min(1.0, semantic_novelty),
            salient_cells=salient_cells,
        )
