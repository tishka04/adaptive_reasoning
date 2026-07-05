import json
from pathlib import Path

import numpy as np

from theory.cross_game_correspondence_discovery import (
    discover_cross_game_correspondences,
)
from theory.m1.invariant_miner import LatentInvariant
from theory.m1.predicate_generation import (
    actionable_invariant_rules,
    generate_predicates,
    historical_predicates,
    run_predicate_coverage_pretest,
)


def _invariant(
    attribute: str,
    outcome: str,
    *,
    rejected_reason=None,
    games=("g1", "g2"),
):
    return LatentInvariant(
        name=f"{attribute}_{outcome}",
        attribute=attribute,
        outcome=outcome,
        support=10,
        games_supporting=frozenset(games),
        cross_game_score=1.0,
        novelty_score=0.8,
        rejected_reason=rejected_reason,
    )


def _write_trace(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_m1_predicate_generator_preserves_historical_output_when_used_as_fallback(tmp_path: Path):
    trace = tmp_path / "zz00-test.steps.jsonl"
    _write_trace(
        trace,
        [
            {
                "game_id": "zz00-test",
                "episode_id": "ep1",
                "step": 1,
                "frame_before": [[1, 2], [0, 0]],
                "available_actions": [1],
                "action": "ACTION1",
                "action_args": None,
                "frame_after": [[2, 2], [0, 0]],
                "game_state_after": "NOT_FINISHED",
                "levels_completed_after": 0,
                "intent": "",
                "hypothesis": "",
                "t_ms": 0,
            }
        ],
    )

    default = discover_cross_game_correspondences(
        trace,
        min_pixel_support=1,
        top_k=10,
    )
    fallback = discover_cross_game_correspondences(
        trace,
        min_pixel_support=1,
        top_k=10,
        predicate_generator=historical_predicates,
    )

    assert fallback.to_dict() == default.to_dict()


def test_m1_predicate_generation_injects_only_actionable_cross_game_invariants():
    rules = actionable_invariant_rules(
        [
            _invariant("object_creation", "appears"),
            _invariant("object_removal", "appears"),
            _invariant("object_count", "preserved"),
            _invariant("color_pair_change", "appears"),
            _invariant("grid_extent", "preserved"),
            _invariant("terminal_state", "absent"),
            _invariant("object_motion", "appears", rejected_reason="single_game"),
            _invariant("adjacency", "modified", games=("g1",)),
        ]
    )
    before = np.asarray([[1, 0], [0, 0]], dtype=np.int32)
    after = np.asarray([[2, 2], [0, 0]], dtype=np.int32)

    predicates = set(
        generate_predicates(
            before,
            after,
            source_color=1,
            target_color=2,
            rules=rules,
        )
    )

    assert "source_target_color_transform" in predicates
    assert "m1_object_creation_appears" in predicates
    assert "m1_object_removal_appears" in predicates
    assert "m1_object_count_preserved" in predicates
    assert "m1_color_pair_change_appears" in predicates
    assert "m1_grid_extent_preserved" not in predicates
    assert "m1_terminal_state_absent" not in predicates
    assert "m1_object_motion_appears" not in predicates
    assert "m1_adjacency_modified" not in predicates


def test_m1_predicate_generation_can_emit_adjacency_preserved():
    rules = actionable_invariant_rules([_invariant("adjacency", "preserved")])
    before = np.asarray([[1, 2], [0, 0]], dtype=np.int32)
    after = np.asarray([[1, 2], [0, 0]], dtype=np.int32)

    predicates = set(
        generate_predicates(
            before,
            after,
            source_color=1,
            target_color=2,
            rules=rules,
        )
    )

    assert "m1_adjacency_preserved" in predicates


def test_m1_predicate_coverage_pretest_reports_off_on_expansion(tmp_path: Path):
    trace = tmp_path / "zz01-test.steps.jsonl"
    accepted = tmp_path / "accepted_invariants.json"
    accepted.write_text(
        json.dumps(
            [
                _invariant("object_creation", "appears").to_dict(),
                _invariant("object_removal", "appears").to_dict(),
                _invariant("color_pair_change", "appears").to_dict(),
            ]
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace,
        [
            {
                "game_id": "zz01-test",
                "episode_id": "ep1",
                "step": 1,
                "frame_before": [[1, 0], [0, 0]],
                "available_actions": [1],
                "action": "ACTION1",
                "action_args": None,
                "frame_after": [[2, 2], [0, 0]],
                "game_state_after": "NOT_FINISHED",
                "levels_completed_after": 0,
                "intent": "",
                "hypothesis": "",
                "t_ms": 0,
            }
        ],
    )

    result = run_predicate_coverage_pretest(
        [trace],
        accepted_invariants_path=accepted,
        min_pixel_support=1,
        top_k=10,
    )
    comparison = result.comparisons[0]

    assert result.rules
    assert comparison.on.unique_predicates_per_trace > comparison.off.unique_predicates_per_trace
    assert comparison.on.relation_candidates_generated > comparison.off.relation_candidates_generated
    assert comparison.on.candidate_pairs_per_trace == comparison.off.candidate_pairs_per_trace
    assert result.trace_support_counted_as_proof is False
    assert result.prior_counted_as_proof is False
