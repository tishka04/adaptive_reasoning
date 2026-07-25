"""SAGE.9w multi-form relation induction tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation
from theory.online_multiform_relational_learner import (
    OnlineMultiformRelationalLearner,
    extract_multiform_relation_patterns,
)


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=("ACTION6",),
        game_state="NOT_FINISHED",
        levels_completed=0,
        infer_players=False,
    )


def _square_grid(
    *,
    color: int = 2,
    row: int = 2,
    column: int = 2,
) -> np.ndarray:
    grid = np.zeros((9, 11), dtype=np.int32)
    grid[row:row + 2, column:column + 2] = color
    return grid


def test_extracts_count_appearance_and_disappearance_patterns():
    before = _square_grid()
    before[6:8, 7:9] = 2
    after = _square_grid()

    patterns = extract_multiform_relation_patterns(
        _observation(before),
        _observation(after),
    )
    families = {pattern.family for pattern in patterns}

    assert "count" in families
    assert "disappearance" in families
    assert any(
        pattern.predicate == "object_count_decrease"
        for pattern in patterns
    )

    appeared = extract_multiform_relation_patterns(
        _observation(after),
        _observation(before),
    )
    assert "appearance" in {pattern.family for pattern in appeared}


def test_extracts_transformation_trajectory_and_correspondence():
    before = _square_grid(color=2, row=2, column=2)
    after = _square_grid(color=7, row=3, column=5)

    patterns = extract_multiform_relation_patterns(
        _observation(before),
        _observation(after),
    )
    families = {pattern.family for pattern in patterns}

    assert {
        "transformation",
        "trajectory",
        "correspondence",
    }.issubset(families)
    assert any(
        pattern.predicate == "attribute_converted"
        for pattern in patterns
    )
    assert any(
        pattern.direction == "down_right_few"
        for pattern in patterns
    )


def test_extracts_spatial_relation_breaks():
    before = np.zeros((9, 11), dtype=np.int32)
    before[2, 2] = 2
    before[2, 5] = 3
    after = np.zeros_like(before)
    after[2, 2] = 2
    after[6, 5] = 3

    patterns = extract_multiform_relation_patterns(
        _observation(before),
        _observation(after),
    )
    spatial = [
        pattern for pattern in patterns
        if pattern.family == "spatial"
    ]

    assert spatial
    assert any(
        pattern.predicate == "aligned_row"
        and pattern.direction == "broken"
        for pattern in spatial
    )


def test_pattern_signatures_are_palette_and_translation_invariant():
    first_before = _square_grid(color=2, row=1, column=1)
    first_after = _square_grid(color=3, row=1, column=3)
    second_before = _square_grid(color=7, row=4, column=4)
    second_after = _square_grid(color=8, row=4, column=6)

    first = extract_multiform_relation_patterns(
        _observation(first_before),
        _observation(first_after),
    )
    second = extract_multiform_relation_patterns(
        _observation(second_before),
        _observation(second_after),
    )

    assert {pattern.signature for pattern in first} == {
        pattern.signature for pattern in second
    }


def test_terminal_support_is_required_then_transfers_to_new_palette_position():
    learner = OnlineMultiformRelationalLearner(
        minimum_terminal_support=2,
    )
    first_before = _square_grid(color=2, row=1, column=1)
    empty = np.zeros_like(first_before)
    first_action = {"x": 1, "y": 1}
    learner.start_branch()
    learner.observe_transition(
        observation_before=_observation(first_before),
        observation_after=_observation(empty),
        action_name="ACTION6",
        action_data=first_action,
        terminal_success=True,
        game_over=False,
    )

    assert learner.confirmed_patterns() == ()
    assert learner.select(
        observation=_observation(first_before),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(name="ACTION6", action_args=first_action),
        ),
    ) is None

    second_before = _square_grid(color=7, row=4, column=5)
    learner.start_branch()
    learner.observe_transition(
        observation_before=_observation(second_before),
        observation_after=_observation(empty),
        action_name="ACTION6",
        action_data={"x": 5, "y": 4},
        terminal_success=True,
        game_over=True,
    )

    third = _square_grid(color=9, row=6, column=7)
    selection = learner.select(
        observation=_observation(third),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(
                name="ACTION6",
                action_args={"x": 7, "y": 6},
            ),
        ),
    )

    assert selection is not None
    assert selection.action_data == {"x": 7, "y": 6}
    assert {
        "count",
        "disappearance",
    }.issubset(selection.predicted_families)
    summary = learner.summary()
    assert summary["terminal_examples"] == 2
    assert summary["confirmed_patterns"] >= 2
    assert summary["selections"] == 1
    assert summary["transferred_selections"] == 1


def test_delayed_frontier_credit_confirms_only_same_branch_patterns():
    learner = OnlineMultiformRelationalLearner(
        minimum_terminal_support=2,
    )
    empty = np.zeros((9, 11), dtype=np.int32)
    first_before = _square_grid(color=2, row=1, column=1)
    learner.start_branch()
    learner.observe_transition(
        observation_before=_observation(first_before),
        observation_after=_observation(empty),
        action_name="ACTION6",
        action_data={"x": 1, "y": 1},
        terminal_success=False,
        game_over=False,
        delayed_frontier_eligibility_id="eligibility-1",
    )
    resolved = learner.resolve_delayed_frontier_credit(
        credited_eligibility_ids=("eligibility-1",),
    )

    assert resolved["credited_eligibilities"] == 1
    assert resolved["credited_patterns"] >= 2
    assert learner.confirmed_patterns() == ()

    second_before = _square_grid(color=7, row=4, column=5)
    learner.start_branch()
    learner.observe_transition(
        observation_before=_observation(second_before),
        observation_after=_observation(empty),
        action_name="ACTION6",
        action_data={"x": 5, "y": 4},
        terminal_success=True,
        game_over=False,
    )

    third = _square_grid(color=9, row=6, column=7)
    selection = learner.select(
        observation=_observation(third),
        available_actions=("ACTION6",),
        available_action_candidates=(
            SimpleNamespace(
                name="ACTION6",
                action_args={"x": 7, "y": 6},
            ),
        ),
    )

    assert selection is not None
    summary = learner.summary()
    assert summary["delayed_frontier_eligibilities_registered"] == 1
    assert summary["delayed_frontier_eligibilities_credited"] == 1
    assert summary["delayed_frontier_pattern_credits"] >= 2
    assert summary["delayed_frontier_credit_branches"] == 1


def test_expired_delayed_frontier_pattern_never_gains_authority():
    learner = OnlineMultiformRelationalLearner(
        minimum_terminal_support=1,
    )
    before = _square_grid()
    learner.observe_transition(
        observation_before=_observation(before),
        observation_after=_observation(np.zeros_like(before)),
        action_name="ACTION6",
        action_data={"x": 2, "y": 2},
        terminal_success=False,
        game_over=False,
        delayed_frontier_eligibility_id="expired",
    )

    resolved = learner.resolve_delayed_frontier_credit(
        expired_eligibility_ids=("expired",),
    )

    assert resolved["expired_eligibilities"] == 1
    assert learner.confirmed_patterns() == ()
    summary = learner.summary()
    assert summary["delayed_frontier_eligibilities_pending"] == 0
    assert summary["delayed_frontier_eligibilities_expired"] == 1


def test_multiform_learner_ablation_is_inert():
    learner = OnlineMultiformRelationalLearner(enabled=False)
    before = _square_grid()

    assert learner.observe_transition(
        observation_before=_observation(before),
        observation_after=_observation(np.zeros_like(before)),
        action_name="ACTION6",
        action_data={"x": 2, "y": 2},
        terminal_success=True,
        game_over=False,
    ) == ()
    assert learner.summary()["observations"] == 0
