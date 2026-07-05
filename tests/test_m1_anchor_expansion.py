import json
from pathlib import Path

import numpy as np

from theory.cross_game_correspondence_discovery import (
    discover_cross_game_correspondences,
)
from theory.m1.anchor_expansion import (
    build_m1_anchor_expander,
    expand_anchors,
)
from theory.m1.predicate_generation import run_predicate_coverage_pretest


def _write_trace(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(before, after, *, action="ACTION1"):
    return {
        "game_id": "zz13-anchor",
        "episode_id": "ep1",
        "step": 1,
        "frame_before": before,
        "available_actions": [1],
        "action": action,
        "action_args": None,
        "frame_after": after,
        "game_state_after": "NOT_FINISHED",
        "levels_completed_after": 0,
        "intent": "",
        "hypothesis": "",
        "t_ms": 0,
    }


def test_created_removed_anchor_pairs_components_without_direct_color_transform():
    before = np.asarray(
        [
            [1, 0, 2, 0],
            [1, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    after = np.asarray(
        [
            [0, 0, 2, 0],
            [0, 0, 2, 0],
            [0, 2, 0, 0],
            [0, 2, 0, 0],
        ],
        dtype=np.int32,
    )

    anchors = expand_anchors(before, after, background=0)
    by_pair = {(anchor.source_color, anchor.target_color): anchor for anchor in anchors}

    assert (1, 2) in by_pair
    assert "m1_anchor_created_removed" in by_pair[(1, 2)].predicates
    assert "same_shape" in by_pair[(1, 2)].predicates


def test_motion_anchor_uses_changed_contacts_as_targets():
    before = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    after = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )

    anchors = expand_anchors(before, after, background=0)
    by_pair = {(anchor.source_color, anchor.target_color): anchor for anchor in anchors}

    assert (1, 2) in by_pair
    assert "m1_anchor_motion" in by_pair[(1, 2)].predicates
    assert "adjacent_to" in by_pair[(1, 2)].predicates


def test_contact_anchor_emits_relation_predicate_for_contact_change():
    before = np.asarray(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
        ],
        dtype=np.int32,
    )
    after = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
        ],
        dtype=np.int32,
    )

    anchors = expand_anchors(before, after, background=0)
    by_pair = {(anchor.source_color, anchor.target_color): anchor for anchor in anchors}

    assert (1, 2) in by_pair
    assert "m1_anchor_contact" in by_pair[(1, 2)].predicates
    assert "adjacent_to" in by_pair[(1, 2)].predicates


def test_anchor_expansion_is_opt_in_for_discovery(tmp_path: Path):
    trace = tmp_path / "zz13-anchor.steps.jsonl"
    before = [[1, 0, 2, 0], [1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    after = [[0, 0, 2, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 2, 0, 0]]
    _write_trace(trace, [_row(before, after)])

    off = discover_cross_game_correspondences(
        trace,
        min_pixel_support=1,
        top_k=10,
    )
    on = discover_cross_game_correspondences(
        trace,
        min_pixel_support=1,
        top_k=10,
        anchor_expander=build_m1_anchor_expander(),
    )

    assert off.candidates == []
    assert len(on.candidates) == 1
    assert on.candidates[0].pair_colors == (1, 2)
    assert {item.name for item in on.candidates[0].predicates} >= {
        "m1_anchor_created_removed",
        "same_shape",
    }
    assert on.wrong_confirmations == 0


def test_anchor_expansion_pretest_increases_candidate_pairs(tmp_path: Path):
    trace = tmp_path / "zz13-anchor.steps.jsonl"
    accepted = tmp_path / "accepted_invariants.json"
    accepted.write_text("[]\n", encoding="utf-8")
    before = [[1, 0, 2, 0], [1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    after = [[0, 0, 2, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 2, 0, 0]]
    _write_trace(trace, [_row(before, after)])

    result = run_predicate_coverage_pretest(
        [trace],
        accepted_invariants_path=accepted,
        min_pixel_support=1,
        top_k=10,
        anchor_expander=build_m1_anchor_expander(),
    )
    comparison = result.comparisons[0]

    assert result.anchor_expansion_enabled is True
    assert comparison.on.candidate_pairs_per_trace > comparison.off.candidate_pairs_per_trace
    assert comparison.on.relation_candidates_generated > comparison.off.relation_candidates_generated
    assert result.trace_support_counted_as_proof is False
    assert result.prior_counted_as_proof is False
