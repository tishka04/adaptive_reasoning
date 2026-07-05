import json

import pytest

from theory.p2 import objective_frontier_handoff_schema as schema
from theory.p2.policy_frontier_records import TRUTH_STATUS
from theory.p2.terminal_outcome_frontier import (
    LOCAL_PRODUCTIVE_TERMINAL_FAILED,
    OBJECTIVE_ALIGNMENT_FRONTIER,
)


def _accepted_review(**overrides):
    row = {
        "frontier_id": (
            "p2_terminal::bp35::conditional_movement_refresh::"
            "local_affordance_productive_but_terminal"
        ),
        "game_id": "bp35-0a0ad940",
        "frontier_type": OBJECTIVE_ALIGNMENT_FRONTIER,
        "frontier_reason": LOCAL_PRODUCTIVE_TERMINAL_FAILED,
        "source": "P2.3",
        "terminal_runs": 15,
        "terminal_budgets": [64, 96, 128, 192, 256],
        "ready_for_objective_frontier_review": True,
        "ready_for_p2_4_saturation_handoff": False,
        "objective_review_accepted": True,
        "saturation_handoff_accepted": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    row.update(overrides)
    return row


def _validation_payload(accepted_reviews=None, *, support=0, saturation=False):
    accepted_reviews = list(accepted_reviews or [])
    return {
        "config": {
            "schema_version": "p2.objective_frontier_handoff_validation.v1",
        },
        "accepted_objective_reviews": accepted_reviews,
        "rejected_objective_reviews": [],
        "summary": {
            "objective_frontiers_seen": len(accepted_reviews),
            "objective_reviews_accepted": len(accepted_reviews),
            "objective_reviews_rejected": 0,
            "saturation_handoffs_accepted": 1 if saturation else 0,
            "ready_for_p2_4_saturation_handoff": bool(saturation),
            "ready_for_objective_frontier_review": bool(accepted_reviews),
            "a40_write_performed": False,
            "m2_write_performed": False,
            "m3_write_performed": False,
            "a32_write_performed": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "objective_handoff_validation_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_objective_handoff_schema_builds_candidate_only_request(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(
        json.dumps(_validation_payload([_accepted_review()])),
        encoding="utf-8",
    )

    payload = schema.run_objective_frontier_handoff_schema(
        objective_frontier_handoff_validation_path=path,
    )

    assert payload["summary"]["objective_frontier_requests"] == 1
    assert payload["summary"]["target"] == "M2_OR_M3"
    assert payload["summary"]["ready_for_m2_or_m3_objective_branch"] is True
    assert payload["summary"]["ready_for_saturation_handoff"] is False
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["a40_write_performed"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    request = payload["objective_frontier_requests"][0]
    assert request["handoff_type"] == "OBJECTIVE_ALIGNMENT_FRONTIER_REQUEST"
    assert request["target"] == "M2_OR_M3"
    assert request["a33_ready"] is False
    assert request["support"] == 0
    assert "When should ACTION6 be stopped" in request["scientific_questions"][0]
    assert "stop_switch_criterion" in request["desired_hypothesis_families"]


def test_objective_handoff_schema_noops_without_accepted_review(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(_validation_payload([])), encoding="utf-8")

    payload = schema.run_objective_frontier_handoff_schema(
        objective_frontier_handoff_validation_path=path,
    )

    assert payload["objective_frontier_requests"] == []
    assert payload["summary"]["ready_for_m2_or_m3_objective_branch"] is False
    assert payload["summary"]["support"] == 0


def test_objective_handoff_schema_rejects_source_saturation_or_support(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(_validation_payload([_accepted_review()], support=1)),
        encoding="utf-8",
    )
    saturation_path = tmp_path / "saturation.json"
    saturation_path.write_text(
        json.dumps(_validation_payload([_accepted_review()], saturation=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        schema.run_objective_frontier_handoff_schema(
            objective_frontier_handoff_validation_path=support_path,
        )
    with pytest.raises(ValueError, match="saturation handoffs"):
        schema.run_objective_frontier_handoff_schema(
            objective_frontier_handoff_validation_path=saturation_path,
        )


def test_validate_objective_frontier_request_rejects_confirmation_flags():
    request = schema.build_objective_frontier_request(
        _accepted_review(),
        request_index=1,
        source_validation_path="validation.json",
    )
    request["support"] = 1

    with pytest.raises(ValueError, match="support must remain 0"):
        schema.validate_objective_frontier_request(request)

    request["support"] = 0
    request["a33_ready"] = True
    with pytest.raises(ValueError, match="A33-ready"):
        schema.validate_objective_frontier_request(request)


def test_validate_objective_frontier_request_rejects_wrong_target_or_reason():
    request = schema.build_objective_frontier_request(
        _accepted_review(),
        request_index=1,
        source_validation_path="validation.json",
    )
    request["target"] = "A33"
    with pytest.raises(ValueError, match="target must be M2_OR_M3"):
        schema.validate_objective_frontier_request(request)

    request["target"] = "M2_OR_M3"
    request["frontier_reason"] = "OTHER"
    with pytest.raises(ValueError, match="objective terminal reason"):
        schema.validate_objective_frontier_request(request)
