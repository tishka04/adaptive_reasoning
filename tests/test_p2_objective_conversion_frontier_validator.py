import json

import pytest

from theory.p2 import objective_conversion_frontier_validator as validator
from theory.p2.objective_conversion_frontier_records import (
    OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
    OBJECTIVE_CONVERSION_FRONTIER,
    TERMINAL_SAFE_BUT_PASSIVE,
)
from theory.p2.policy_frontier_records import TRUTH_STATUS


def _frontier(**overrides):
    row = {
        "frontier_id": "p2g1::bp35::terminal_safe_but_passive::objective_conversion",
        "source": "P3.G1",
        "frontier_type": OBJECTIVE_CONVERSION_FRONTIER,
        "frontier_reason": TERMINAL_SAFE_BUT_PASSIVE,
        "game_id": "bp35-0a0ad940",
        "blocked_capability": OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
        "ready_for_objective_conversion_review": True,
        "ready_for_m2_or_m3": False,
        "ready_for_saturation_handoff": False,
        "a33_ready": False,
        "desired_hypothesis_families": [
            "post_safe_stop_objective_conversion",
            "subgoal_target_reselection",
        ],
        "requested_experiment_styles": [
            "stop_state_action_matrix",
            "post_safe_stop_short_sequence_probe",
        ],
        "scientific_questions": [
            "Which action converts a safe stop state into objective completion?",
        ],
        "policy_result_counted_as_scientific_verdict": False,
        "objective_conversion_frontier_counted_as_confirmation": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    row.update(overrides)
    return row


def _payload(records=None, *, support=0, verdict=False, m2_write=False):
    return {
        "config": {"schema_version": "p2.objective_conversion_frontier_records.v1"},
        "objective_conversion_frontier_records": list(records or []),
        "summary": {
            "frontier_records": len(records or []),
            "ready_for_objective_conversion_review": bool(records),
            "ready_for_m2_or_m3": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "a40_write_performed": False,
            "m2_write_performed": m2_write,
            "m3_write_performed": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "policy_result_counted_as_scientific_verdict": verdict,
        "objective_conversion_frontier_counted_as_confirmation": False,
        "a40_write_performed": False,
        "m2_write_performed": m2_write,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_objective_conversion_validator_accepts_no_write_review_candidate(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(_payload([_frontier()])), encoding="utf-8")

    payload = validator.run_objective_conversion_frontier_validation(
        objective_conversion_frontier_records_path=path,
    )

    assert payload["summary"]["objective_conversion_frontiers_seen"] == 1
    assert payload["summary"]["objective_conversion_reviews_accepted"] == 1
    assert payload["summary"]["objective_conversion_reviews_rejected"] == 0
    assert payload["summary"]["ready_for_objective_conversion_review"] is True
    assert payload["summary"]["ready_for_m2_or_m3"] is False
    assert payload["summary"]["objective_conversion_review_no_write"] is True
    assert payload["summary"]["a40_write_performed"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["a32_write_performed"] is False
    assert payload["summary"]["a33_write_performed"] is False
    assert payload["summary"]["support"] == 0
    accepted = payload["accepted_objective_conversion_reviews"][0]
    assert accepted["objective_conversion_review_target"] == (
        "objective_conversion_frontier_review"
    )
    assert accepted["support"] == 0


def test_objective_conversion_validator_rejects_direct_handoff_and_a33_ready():
    result = validator.evaluate_objective_conversion_frontier_for_review(
        _frontier(ready_for_m2_or_m3=True, a33_ready=True)
    )

    assert result["objective_conversion_review_accepted"] is False
    assert "DIRECT_M2_M3_HANDOFF_NOT_ALLOWED_IN_P2_G2" in result[
        "rejection_reasons"
    ]
    assert "A33_READY_NOT_ALLOWED" in result["rejection_reasons"]
    assert result["ready_for_m2_or_m3"] is False


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"support": 1}, "SUPPORT_MUST_REMAIN_ZERO"),
        ({"status": "CONFIRMED"}, "STATUS_CONFIRMED_OR_REFUTED_NOT_ALLOWED"),
        ({"status": "REFUTED"}, "STATUS_CONFIRMED_OR_REFUTED_NOT_ALLOWED"),
        ({"revision_status": "CONFIRMED"}, "REVISION_STATUS_NOT_CANDIDATE_ONLY"),
        (
            {"policy_result_counted_as_scientific_verdict": True},
            "POLICY_RESULT_COUNTED_AS_VERDICT",
        ),
        (
            {"objective_conversion_frontier_counted_as_confirmation": True},
            "OBJECTIVE_CONVERSION_FRONTIER_COUNTED_AS_CONFIRMATION",
        ),
        ({"desired_hypothesis_families": []}, "MISSING_DESIRED_HYPOTHESIS_FAMILIES"),
        ({"requested_experiment_styles": []}, "MISSING_REQUESTED_EXPERIMENT_STYLES"),
        ({"scientific_questions": []}, "MISSING_SCIENTIFIC_QUESTIONS"),
    ],
)
def test_objective_conversion_validator_rejects_bad_records(override, reason):
    result = validator.evaluate_objective_conversion_frontier_for_review(
        _frontier(**override)
    )

    assert result["objective_conversion_review_accepted"] is False
    assert reason in result["rejection_reasons"]
    assert result["support"] == 0


def test_objective_conversion_validator_handles_no_frontiers_as_noop(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = validator.run_objective_conversion_frontier_validation(
        objective_conversion_frontier_records_path=path,
    )

    assert payload["summary"]["objective_conversion_frontiers_seen"] == 0
    assert payload["summary"]["objective_conversion_reviews_accepted"] == 0
    assert payload["summary"]["ready_for_objective_conversion_review"] is False
    assert payload["summary"]["support"] == 0


@pytest.mark.parametrize(
    ("support", "verdict", "m2_write", "message"),
    [
        (1, False, False, "support must remain 0"),
        (0, True, False, "scientific verdict"),
        (0, False, True, "must not have written M2"),
    ],
)
def test_objective_conversion_validator_rejects_bad_source_payload(
    tmp_path,
    support,
    verdict,
    m2_write,
    message,
):
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            _payload(
                [_frontier()],
                support=support,
                verdict=verdict,
                m2_write=m2_write,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        validator.run_objective_conversion_frontier_validation(
            objective_conversion_frontier_records_path=path,
        )
