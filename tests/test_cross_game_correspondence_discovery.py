"""A12 cross-game correspondence discovery tests."""

from __future__ import annotations

from pathlib import Path

from theory.cross_game_correspondence_discovery import (
    discover_cross_game_correspondences,
)
from theory.epistemic_metrics import HypothesisStatus


def test_ft09_discovers_non_ar25_source_target_correspondence_candidates():
    result = discover_cross_game_correspondences(
        Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        top_k=5,
    )

    assert result.game_id == "ft09-0d8bbf25"
    assert result.candidates
    assert result.candidates[0].key == (
        "correspondence::ACTION6::modifies::colors14_15"
    )
    assert result.candidates[0].source_color == 14
    assert result.candidates[0].target_color == 15
    assert result.source_colors
    assert result.target_colors

    predicates = set(result.source_target_predicates)
    assert "source_target_color_transform" in predicates
    assert "same_shape" in predicates
    assert "aligned_with" in predicates

    top = result.candidates[0]
    assert "selected_pair_exists" in top.weak_ready_candidates
    assert "source_target_relation_satisfied" in top.strong_ready_candidates
    assert "selected_source_matches_target_shape" in top.strong_ready_candidates

    hypotheses = result.hypotheses
    assert hypotheses
    assert hypotheses[0].key == top.key
    assert hypotheses[0].status == HypothesisStatus.UNRESOLVED
    assert result.wrong_confirmations == 0
    assert result.score is not None
    assert result.score.wrong_confirmations == 0
