import json

import pytest

from theory.p2 import objective_conversion_handoff_schema as schema
from theory.p2.objective_conversion_frontier_records import (
    OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
    OBJECTIVE_CONVERSION_FRONTIER,
    TERMINAL_SAFE_BUT_PASSIVE,
)
from theory.p2.policy_frontier_records import TRUTH_STATUS


def _accepted_review(**overrides):
    row = {
        "frontier_id": "p2g1::bp35::terminal_safe_but_passive::objective_conversion",
        "game_id": "bp35-0a0ad940",
        "frontier_type": OBJECTIVE_CONVERSION_FRONTIER,
        "frontier_reason": TERMINAL_SAFE_BUT_PASSIVE,
        "blocked_capability": OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
        "source": "P3.G1",
        "objective_conversion_review_accepted": True,
        "objective_conversion_review_target": "objective_conversion_frontier_review",
        "ready_for_objective_conversion_review": True,
        "ready_for_m2_or_m3": False,
        "desired_hypothesis_families": [
            "post_safe_stop_objective_conversion",
            "subgoal_target_reselection",
            "objective_readiness_condition",
            "terminal_safe_sequence_search",
        ],
        "requested_experiment_styles": [
            "stop_state_action_matrix",
            "post_safe_stop_short_sequence_probe",
            "relation_target_ablation_after_safe_stop",
            "objective_completion_vs_relation_progress_discriminator",
        ],
        "scientific_questions": [
            "After terminal-safe stop, which action converts relation progress?",
            "Which objective-readiness signal is missing?",
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    row.update(overrides)
    return row


def _validation_payload(
    accepted_reviews=None,
    *,
    support=0,
    no_write=True,
    ready_for_m2_or_m3=False,
    m3_write=False,
    verdict=False,
):
    accepted_reviews = list(accepted_reviews or [])
    return {
        "config": {
            "schema_version": "p2.objective_conversion_frontier_validation.v1",
        },
        "accepted_objective_conversion_reviews": accepted_reviews,
        "rejected_objective_conversion_reviews": [],
        "summary": {
            "objective_conversion_frontiers_seen": len(accepted_reviews),
            "objective_conversion_reviews_accepted": len(accepted_reviews),
            "objective_conversion_reviews_rejected": 0,
            "ready_for_objective_conversion_review": bool(accepted_reviews),
            "ready_for_m2_or_m3": ready_for_m2_or_m3,
            "objective_conversion_review_no_write": no_write,
            "a40_write_performed": False,
            "m2_write_performed": False,
            "m3_write_performed": m3_write,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "objective_conversion_validation_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": verdict,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": m3_write,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_objective_conversion_handoff_schema_builds_candidate_only_request(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(
        json.dumps(_validation_payload([_accepted_review()])),
        encoding="utf-8",
    )

    payload = schema.run_objective_conversion_handoff_schema(
        objective_conversion_frontier_validation_path=path,
    )

    assert payload["summary"]["objective_conversion_handoff_requests"] == 1
    assert payload["summary"]["handoff_type"] == (
        "OBJECTIVE_CONVERSION_FRONTIER_REQUEST"
    )
    assert payload["summary"]["target"] == "M2_OR_M3"
    assert payload["summary"]["target_modules"] == ["M2.G1", "M3.G1"]
    assert payload["summary"]["ready_for_m2_or_m3_objective_conversion_branch"] is True
    assert payload["summary"]["ready_for_direct_downstream_write"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["support"] == 0
    request = payload["objective_conversion_handoff_requests"][0]
    assert request["source_frontier_id"] == (
        "p2g1::bp35::terminal_safe_but_passive::objective_conversion"
    )
    assert request["frontier_type"] == OBJECTIVE_CONVERSION_FRONTIER
    assert request["frontier_reason"] == TERMINAL_SAFE_BUT_PASSIVE
    assert request["blocked_capability"] == OBJECTIVE_CONVERSION_AFTER_SAFE_STOP
    assert request["target"] == "M2_OR_M3"
    assert request["target_modules"] == ["M2.G1", "M3.G1"]
    assert request["ready_for_direct_downstream_write"] is False
    assert request["support"] == 0
    assert "post_safe_stop_objective_conversion" in request[
        "requested_hypothesis_families"
    ]
    assert "stop_state_action_matrix" in request["requested_experiment_styles"]


def test_objective_conversion_handoff_schema_noops_without_accepted_review(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(_validation_payload([])), encoding="utf-8")

    payload = schema.run_objective_conversion_handoff_schema(
        objective_conversion_frontier_validation_path=path,
    )

    assert payload["objective_conversion_handoff_requests"] == []
    assert payload["summary"]["objective_conversion_handoff_requests"] == 0
    assert payload["summary"]["ready_for_m2_or_m3_objective_conversion_branch"] is False
    assert payload["summary"]["ready_for_direct_downstream_write"] is False
    assert payload["summary"]["support"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"support": 1}, "support must remain 0"),
        ({"no_write": False}, "requires objective conversion review no-write"),
        ({"ready_for_m2_or_m3": True}, "cannot already be ready for M2/M3"),
        ({"m3_write": True}, "must not have m3_write_performed"),
        ({"verdict": True}, "scientific verdict"),
    ],
)
def test_objective_conversion_handoff_schema_rejects_bad_source(tmp_path, kwargs, message):
    path = tmp_path / "validation.json"
    path.write_text(
        json.dumps(_validation_payload([_accepted_review()], **kwargs)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        schema.run_objective_conversion_handoff_schema(
            objective_conversion_frontier_validation_path=path,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"support": 1}, "support must remain 0"),
        ({"a33_ready": True}, "A33-ready"),
        ({"ready_for_direct_downstream_write": True}, "direct downstream write"),
        ({"handoff_request_counted_as_confirmation": True}, "count as confirmation"),
        ({"policy_result_counted_as_scientific_verdict": True}, "scientific verdict"),
        ({"requested_hypothesis_families": []}, "needs hypothesis families"),
        ({"requested_experiment_styles": []}, "needs experiment styles"),
        ({"scientific_questions": []}, "needs scientific questions"),
    ],
)
def test_validate_objective_conversion_request_rejects_bad_request(override, message):
    request = schema.build_objective_conversion_request(
        _accepted_review(),
        request_index=1,
        source_validation_path="validation.json",
    )
    request.update(override)

    with pytest.raises(ValueError, match=message):
        schema.validate_objective_conversion_request(request)


def test_validate_objective_conversion_request_rejects_wrong_target_or_reason():
    request = schema.build_objective_conversion_request(
        _accepted_review(),
        request_index=1,
        source_validation_path="validation.json",
    )
    request["target"] = "A33"
    with pytest.raises(ValueError, match="target must be M2_OR_M3"):
        schema.validate_objective_conversion_request(request)

    request["target"] = "M2_OR_M3"
    request["frontier_reason"] = "OTHER"
    with pytest.raises(ValueError, match="TERMINAL_SAFE_BUT_PASSIVE"):
        schema.validate_objective_conversion_request(request)
