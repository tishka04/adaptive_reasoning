import pytest

from theory.p1 import bp35_policy_frontier_trigger as trigger
from theory.p1.bp35_sage_candidate_policy_probe import (
    CONDITIONAL_ACTION4_REFRESH_POLICY,
    CONDITIONAL_MOVEMENT_REFRESH_POLICY,
    TRUTH_STATUS,
)


def _matrix_with_summary(summary, *, counted_as_confirmation_field=None):
    counted_field = counted_as_confirmation_field or (
        "movement_refresh_counted_as_confirmation"
        if summary.get("condition") == CONDITIONAL_MOVEMENT_REFRESH_POLICY
        else "conditional_refresh_counted_as_confirmation"
    )
    return {
        "summary": {
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "aggregate": {
            "refresh_runs": 1,
            "refresh_total_triggers": int(
                summary.get("conditional_refresh_triggers", 0)
                or summary.get("conditional_movement_refresh_triggers", 0)
                or 0
            ),
            counted_field: False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "budget_runs": [
            {
                "budget": 8,
                "tie_break_seed": 0,
                "conditions_run": [summary.get("condition")],
                "condition_summaries": [summary],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                counted_field: False,
            }
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_real_matrix_without_refresh_saturation_does_not_trigger_frontier(tmp_path):
    summary = {
        "condition": CONDITIONAL_ACTION4_REFRESH_POLICY,
        "conditional_refresh_triggers": 0,
        "useful_action6_after_conditional_refresh_steps": 0,
        "new_action6_affordances_after_refresh": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    payload = trigger.run_bp35_policy_frontier_trigger(
        conditional_refresh_matrix_path=_write_matrix(tmp_path, _matrix_with_summary(summary)),
        refresh_mode="action4",
        include_synthetic_fixture=True,
    )

    assert payload["real_rollout_evaluation"]["frontier_triggered"] is False
    assert payload["real_rollout_evaluation"]["frontier_reason"] == "NO_REFRESH_SATURATION_OBSERVED"
    assert payload["summary"]["real_ready_for_p2_frontier_extraction"] is False
    assert payload["summary"]["synthetic_fixture_frontier_triggered"] is True
    assert payload["summary"]["synthetic_fixture_validates_trigger_logic"] is True
    assert payload["summary"]["support"] == 0


def test_exhausted_conditional_refresh_triggers_frontier_candidate_only():
    summary = {
        "condition": CONDITIONAL_ACTION4_REFRESH_POLICY,
        "conditional_refresh_triggers": 1,
        "action6_after_conditional_refresh_steps": 1,
        "useful_action6_after_conditional_refresh_steps": 0,
        "new_action6_affordances_after_refresh": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    result = trigger.evaluate_policy_frontier_trigger(
        _matrix_with_summary(summary),
        refresh_mode="action4",
        synthetic_fixture=False,
        source_label="unit_test",
    )

    assert result["frontier_triggered"] is True
    assert result["frontier_reason"] == trigger.ACTION4_FRONTIER_REASON
    assert result["ready_for_p2_frontier_extraction"] is True
    assert result["frontier_record"]["status"] == "UNRESOLVED"
    assert result["frontier_record"]["support"] == 0
    assert result["frontier_trigger_counted_as_confirmation"] is False


def test_refresh_with_new_action6_affordance_does_not_trigger_frontier():
    summary = {
        "condition": CONDITIONAL_ACTION4_REFRESH_POLICY,
        "conditional_refresh_triggers": 1,
        "action6_after_conditional_refresh_steps": 1,
        "useful_action6_after_conditional_refresh_steps": 1,
        "new_action6_affordances_after_refresh": 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    result = trigger.evaluate_policy_frontier_trigger(
        _matrix_with_summary(summary),
        refresh_mode="action4",
        synthetic_fixture=False,
        source_label="unit_test",
    )

    assert result["frontier_triggered"] is False
    assert result["frontier_reason"] == "REFRESH_DID_NOT_MEET_EXHAUSTED_FRONTIER_CRITERION"
    assert result["ready_for_p2_frontier_extraction"] is False
    assert result["support"] == 0


def test_frontier_trigger_rejects_matrix_that_counts_confirmation():
    payload = _matrix_with_summary(
        {
            "condition": CONDITIONAL_ACTION4_REFRESH_POLICY,
            "conditional_refresh_triggers": 0,
            "support": 0,
        }
    )
    payload["aggregate"]["conditional_refresh_counted_as_confirmation"] = True

    with pytest.raises(ValueError, match="counted_as_confirmation"):
        trigger.evaluate_policy_frontier_trigger(payload)


def test_movement_refresh_default_without_saturation_does_not_trigger_frontier(tmp_path):
    summary = {
        "condition": CONDITIONAL_MOVEMENT_REFRESH_POLICY,
        "conditional_movement_refresh_triggers": 0,
        "useful_action6_after_conditional_movement_refresh_steps": 0,
        "new_action6_affordances_after_movement_refresh": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    payload = trigger.run_bp35_policy_frontier_trigger(
        movement_refresh_matrix_path=_write_matrix(tmp_path, _matrix_with_summary(summary)),
        include_synthetic_fixture=True,
    )

    assert payload["config"]["refresh_mode"] == "movement"
    assert payload["real_rollout_evaluation"]["frontier_triggered"] is False
    assert payload["real_rollout_evaluation"]["frontier_reason"] == "NO_MOVEMENT_REFRESH_SATURATION_OBSERVED"
    assert payload["real_rollout_evaluation"]["exhausted_policy"] == CONDITIONAL_MOVEMENT_REFRESH_POLICY
    assert payload["real_rollout_evaluation"]["refresh_candidates"] == [
        "ACTION3",
        "ACTION4",
        "ACTION1",
        "ACTION2",
    ]
    assert payload["summary"]["synthetic_fixture_frontier_triggered"] is True
    assert payload["summary"]["synthetic_fixture_validates_trigger_logic"] is True
    assert payload["summary"]["support"] == 0


def test_exhausted_movement_refresh_triggers_frontier_candidate_only():
    summary = {
        "condition": CONDITIONAL_MOVEMENT_REFRESH_POLICY,
        "conditional_movement_refresh_triggers": 1,
        "action6_after_conditional_movement_refresh_steps": 1,
        "useful_action6_after_conditional_movement_refresh_steps": 0,
        "new_action6_affordances_after_movement_refresh": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    result = trigger.evaluate_policy_frontier_trigger(
        _matrix_with_summary(summary),
        synthetic_fixture=False,
        source_label="unit_test",
    )

    assert result["frontier_triggered"] is True
    assert result["frontier_reason"] == trigger.MOVEMENT_FRONTIER_REASON
    assert result["ready_for_p2_frontier_extraction"] is True
    assert result["frontier_record"]["exhausted_policy"] == CONDITIONAL_MOVEMENT_REFRESH_POLICY
    assert result["frontier_record"]["refresh_candidates"] == [
        "ACTION3",
        "ACTION4",
        "ACTION1",
        "ACTION2",
    ]
    assert result["frontier_record"]["support"] == 0


def _write_matrix(tmp_path, payload):
    path = tmp_path / "matrix.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
