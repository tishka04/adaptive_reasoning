import json
from pathlib import Path

from theory.m1.invariant_miner import (
    REJECTED_REASONS,
    cross_game_score,
    mine_invariants,
    raw_attribute_outcomes,
    write_invariant_outputs,
)


FORBIDDEN_PREDICATES = {
    "source_target_color_transform",
    "paired_with",
    "same_shape",
    "aligned_with",
    "adjacent_to",
}


def _row(
    game_id: str,
    *,
    action: str = "ACTION1",
    shape_changed: bool = False,
    object_sizes_changed: bool = False,
    object_motion: bool = False,
    forbidden_same_shape: bool = False,
):
    before_sizes = [1, 3]
    after_sizes = [1, 4] if object_sizes_changed else [1, 3]
    row = {
        "game_id": game_id,
        "action": action,
        "shape_changed": shape_changed,
        "object_sizes_before": before_sizes,
        "object_sizes_after": after_sizes,
        "object_motion_vectors": (
            [{"color": 1, "size": 1, "distance": 1.0}] if object_motion else []
        ),
    }
    if forbidden_same_shape:
        row["same_shape"] = True
        row["aligned_with"] = True
    return row


def test_miner_consumes_only_m1_raw_fields_not_theory_predicates():
    row = _row("g1", forbidden_same_shape=True)

    outcomes = raw_attribute_outcomes(row)
    result = mine_invariants(
        [row, _row("g2", forbidden_same_shape=True)],
        min_games=2,
        min_intra_game_support=0.6,
        min_novelty=0.3,
    )

    names = {
        value
        for invariant in result.accepted_invariants + result.rejected_invariants
        for value in (invariant.name, invariant.attribute)
    }
    assert ("same_shape", "modified") not in outcomes
    assert FORBIDDEN_PREDICATES.isdisjoint(names)


def test_single_game_high_support_is_rejected():
    rows = [_row("solo", object_motion=True) for _ in range(10)]

    result = mine_invariants(rows, min_games=2, min_intra_game_support=0.9)

    rejected = [
        invariant
        for invariant in result.rejected_invariants
        if invariant.attribute == "object_motion" and invariant.outcome == "appears"
    ]
    assert rejected
    assert rejected[0].rejected_reason == "single_game"
    assert rejected[0].support == 10


def test_multi_game_moderate_support_is_accepted():
    rows = []
    for game_id in ("g1", "g2"):
        rows.extend([_row(game_id, object_motion=True) for _ in range(2)])
        rows.append(_row(game_id, object_motion=False))

    result = mine_invariants(rows, min_games=2, min_intra_game_support=0.6)

    accepted = [
        invariant
        for invariant in result.accepted_invariants
        if invariant.attribute == "object_motion" and invariant.outcome == "appears"
    ]
    assert accepted
    assert accepted[0].games_supporting == frozenset({"g1", "g2"})
    assert accepted[0].novelty_score >= 0.3
    assert accepted[0].rejected_reason is None


def test_cross_game_score_is_monotone_in_number_of_games():
    one_game = cross_game_score({"g1": 0.8})
    two_games = cross_game_score({"g1": 0.8, "g2": 0.8})
    three_games = cross_game_score({"g1": 0.8, "g2": 0.8, "g3": 0.8})

    assert one_game < two_games < three_games


def test_quasi_duplicate_raw_signatures_are_rejected_by_low_novelty():
    rows = [_row("g1"), _row("g1"), _row("g2"), _row("g2")]

    result = mine_invariants(rows, min_games=2, min_intra_game_support=0.9)

    shape_candidates = [
        invariant
        for invariant in result.accepted_invariants + result.rejected_invariants
        if invariant.attribute in {"shape_profile", "shape_inventory"}
        and invariant.outcome == "preserved"
    ]
    accepted = [invariant for invariant in shape_candidates if invariant.rejected_reason is None]
    low_novelty = [
        invariant
        for invariant in shape_candidates
        if invariant.rejected_reason == "low_novelty"
    ]
    assert len(accepted) == 1
    assert low_novelty


def test_rejected_invariants_are_persisted_with_reasons(tmp_path: Path):
    rows = [_row("solo", object_motion=True) for _ in range(3)]
    result = mine_invariants(rows, min_games=2, min_intra_game_support=0.9)
    accepted_path = tmp_path / "accepted_invariants.json"
    rejected_path = tmp_path / "rejected_invariants.json"

    write_invariant_outputs(
        result,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
    )

    rejected_rows = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert accepted_path.exists()
    assert rejected_rows
    assert {row["rejected_reason"] for row in rejected_rows} <= REJECTED_REASONS
