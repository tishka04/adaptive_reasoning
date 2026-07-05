from types import SimpleNamespace

import numpy as np

from theory.m1.polymorphic_a25_pretest import (
    ACTION_NOT_AVAILABLE,
    BLOCKED,
    MISSING_POSITION_ARGUMENT,
    NO_MEASURABLE_BEFORE_AFTER_METRIC,
    NO_LIVE_OBJECT_ANCHOR,
    TESTABLE,
    PolymorphicA25PretestRow,
    evaluate_candidate_testability,
    summarize_polymorphic_pretest,
)


def _action(name, **action_args):
    return SimpleNamespace(name=name, action_args=dict(action_args))


def _candidate(candidate_type, *, action="ACTION1", game_id="aa-game"):
    return {
        "game_id": game_id,
        "candidate_type": candidate_type,
        "action": action,
        "test_goal": f"test {candidate_type}",
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def test_object_motion_candidate_is_testable_with_live_object_and_action():
    grid = np.asarray(
        [
            [0, 0, 0],
            [0, 4, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    row = evaluate_candidate_testability(
        _candidate("object_motion_candidate"),
        live_grid=grid,
        valid_actions=[_action("ACTION1")],
        candidate_id="cand-1",
    )

    assert row.testability_status == TESTABLE
    assert row.blocking_reason is None
    assert row.required_observation == "object_positions_before_after"
    assert row.available_live_affordance["live_non_background_object_count"] == 1
    assert row.status == "UNRESOLVED"
    assert row.trace_support_counted_as_proof is False
    assert row.prior_counted_as_proof is False


def test_position_candidate_requires_live_position_argument():
    grid = np.asarray(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=np.int32,
    )

    row = evaluate_candidate_testability(
        _candidate("position_effect_candidate"),
        live_grid=grid,
        valid_actions=[_action("ACTION1")],
        candidate_id="cand-2",
    )

    assert row.testability_status == BLOCKED
    assert row.blocking_reason == MISSING_POSITION_ARGUMENT
    assert row.required_observation == "local_patch_before_after"


def test_missing_action_blocks_before_mechanic_checks():
    grid = np.asarray([[0, 1]], dtype=np.int32)

    row = evaluate_candidate_testability(
        _candidate("object_lifecycle_candidate", action="ACTION2"),
        live_grid=grid,
        valid_actions=[_action("ACTION1")],
        candidate_id="cand-3",
    )

    assert row.testability_status == BLOCKED
    assert row.blocking_reason == ACTION_NOT_AVAILABLE
    assert row.available_live_affordance["matching_action_count"] == 0


def test_object_candidate_without_live_object_reports_anchor_blocker():
    grid = np.asarray(
        [
            [0, 0],
            [0, 0],
        ],
        dtype=np.int32,
    )

    row = evaluate_candidate_testability(
        _candidate("object_motion_candidate"),
        live_grid=grid,
        valid_actions=[_action("ACTION1")],
        candidate_id="cand-4",
    )

    assert row.testability_status == BLOCKED
    assert row.blocking_reason == NO_LIVE_OBJECT_ANCHOR


def test_unknown_candidate_type_reports_no_measurable_metric():
    grid = np.asarray([[0, 1]], dtype=np.int32)

    row = evaluate_candidate_testability(
        _candidate("unmapped_mechanic_candidate"),
        live_grid=grid,
        valid_actions=[_action("ACTION1")],
        candidate_id="cand-5",
    )

    assert row.testability_status == BLOCKED
    assert row.blocking_reason == NO_MEASURABLE_BEFORE_AFTER_METRIC
    assert row.required_observation == "unknown"


def test_summarize_polymorphic_pretest_reports_kpis():
    rows = [
        PolymorphicA25PretestRow(
            candidate_id="a",
            game_id="aa-game",
            candidate_type="object_motion_candidate",
            action="ACTION1",
            testability_status=TESTABLE,
            required_observation="object_positions_before_after",
        ),
        PolymorphicA25PretestRow(
            candidate_id="b",
            game_id="bb-game",
            candidate_type="position_effect_candidate",
            action="ACTION1",
            testability_status=BLOCKED,
            required_observation="local_patch_before_after",
            blocking_reason=MISSING_POSITION_ARGUMENT,
        ),
    ]

    summary = summarize_polymorphic_pretest(rows)

    assert summary["mechanic_candidates_total"] == 2
    assert summary["mechanic_candidates_testable"] == 1
    assert summary["testable_by_type"] == {"object_motion_candidate": 1}
    assert summary["testable_by_game"] == {"aa-game": 1}
    assert summary["blocking_reasons"] == {MISSING_POSITION_ARGUMENT: 1}
    assert summary["wrong_confirmations"] == 0
