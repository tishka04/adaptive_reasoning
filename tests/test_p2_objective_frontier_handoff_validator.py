import json

import pytest

from theory.p2 import objective_frontier_handoff_validator as validator
from theory.p2.policy_frontier_records import TRUTH_STATUS
from theory.p2.terminal_outcome_frontier import (
    LOCAL_PRODUCTIVE_TERMINAL_FAILED,
    OBJECTIVE_ALIGNMENT_FRONTIER,
)


def _frontier(**overrides):
    row = {
        "frontier_id": (
            "p2_terminal::bp35::conditional_movement_refresh::"
            "local_affordance_productive_but_terminal"
        ),
        "frontier_type": OBJECTIVE_ALIGNMENT_FRONTIER,
        "frontier_reason": LOCAL_PRODUCTIVE_TERMINAL_FAILED,
        "game_id": "bp35-0a0ad940",
        "source": "P2.3",
        "source_saturation_handoff_ready": False,
        "ready_for_p2_4_saturation_handoff": False,
        "ready_for_objective_frontier_review": True,
        "ready_for_m2_or_m3": False,
        "terminal_runs": 15,
        "terminal_budgets": [64, 96, 128, 192, 256],
        "policy_result_counted_as_scientific_verdict": False,
        "objective_frontier_counted_as_confirmation": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    row.update(overrides)
    return row


def _payload(frontier=None, *, support=0, verdict=False):
    return {
        "config": {"schema_version": "p2.terminal_outcome_frontier.v1"},
        "terminal_outcome_frontier": frontier,
        "summary": {
            "ready_for_objective_frontier_review": frontier is not None,
            "ready_for_p2_4_saturation_handoff": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "terminal_outcome_frontier_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": verdict,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_objective_frontier_validator_accepts_no_write_review_candidate(tmp_path):
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(_payload(_frontier())), encoding="utf-8")

    payload = validator.run_objective_frontier_handoff_validation(
        terminal_outcome_frontier_path=path,
    )

    assert payload["summary"]["objective_frontiers_seen"] == 1
    assert payload["summary"]["objective_reviews_accepted"] == 1
    assert payload["summary"]["saturation_handoffs_accepted"] == 0
    assert payload["summary"]["ready_for_p2_4_saturation_handoff"] is False
    assert payload["summary"]["a40_write_performed"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["a32_write_performed"] is False
    assert payload["summary"]["support"] == 0
    assert payload["accepted_objective_reviews"][0]["objective_review_target"] == (
        "objective_frontier_review"
    )


def test_objective_frontier_validator_rejects_saturation_handoff_flags():
    result = validator.evaluate_objective_frontier_for_handoff(
        _frontier(
            ready_for_p2_4_saturation_handoff=True,
            source_saturation_handoff_ready=True,
        )
    )

    assert result["objective_review_accepted"] is False
    assert "SATURATION_HANDOFF_NOT_ALLOWED_FOR_OBJECTIVE_FRONTIER" in result[
        "rejection_reasons"
    ]
    assert "SOURCE_SATURATION_HANDOFF_READY_NOT_OBJECTIVE_ONLY" in result[
        "rejection_reasons"
    ]
    assert result["saturation_handoff_accepted"] is False


def test_objective_frontier_validator_rejects_direct_m2_m3_handoff_or_support():
    direct = validator.evaluate_objective_frontier_for_handoff(
        _frontier(ready_for_m2_or_m3=True)
    )
    support = validator.evaluate_objective_frontier_for_handoff(
        _frontier(support=1)
    )

    assert direct["objective_review_accepted"] is False
    assert "DIRECT_M2_M3_HANDOFF_NOT_ALLOWED_IN_P2_5" in direct[
        "rejection_reasons"
    ]
    assert support["objective_review_accepted"] is False
    assert "SUPPORT_MUST_REMAIN_ZERO" in support["rejection_reasons"]


def test_objective_frontier_validator_handles_no_frontier_as_noop(tmp_path):
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(_payload(None)), encoding="utf-8")

    payload = validator.run_objective_frontier_handoff_validation(
        terminal_outcome_frontier_path=path,
    )

    assert payload["summary"]["objective_frontiers_seen"] == 0
    assert payload["summary"]["objective_reviews_accepted"] == 0
    assert payload["summary"]["ready_for_objective_frontier_review"] is False
    assert payload["summary"]["support"] == 0


def test_objective_frontier_validator_rejects_source_verdict_flags(tmp_path):
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(_payload(_frontier(), verdict=True)), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific verdict"):
        validator.run_objective_frontier_handoff_validation(
            terminal_outcome_frontier_path=path,
        )
