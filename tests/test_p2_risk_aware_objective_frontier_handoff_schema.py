import json
from pathlib import Path

import pytest

from theory.p2 import risk_aware_objective_frontier_handoff_schema as schema
from theory.p2.policy_frontier_records import TRUTH_STATUS
from theory.p2.risk_aware_post_stop_frontier_records import (
    OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION,
    RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER,
    RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_record(**overrides) -> dict:
    record = {
        "frontier_id": "p2g4::bp35::risk_aware_post_stop_no_objective_completion",
        "source": "P3.G4",
        "game_id": "bp35-0a0ad940",
        "frontier_type": RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER,
        "frontier_reason": RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION,
        "blocked_capability": OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION,
        "evidence": {
            "accepted_risk_targeted_safe_stops": 7,
            "terminal_rate": 0.0,
            "mean_terminal_adjusted_progress": 142.857143,
            "mean_delta_vs_hold": 7.857143,
            "mean_delta_vs_action6_only": 2.857143,
            "improvement_over_action6_only": True,
            "static_extension_terminal_options": 4,
            "static_extension_terminal_safe_stops": 2,
            "unsafe_extension_options_avoided": 4,
            "objective_completion_signal": False,
            "objective_completion_runs": 0,
            "adapter_relearned": False,
            "source_cells_rerun": True,
            "selection_uses_risk_targeted_candidate_outcomes": False,
        },
        "risk_region_snapshot": {
            "action6_action3_terminal_rate": 0.285714,
            "action6_action4_terminal_rate": 0.285714,
            "static_extension_terminal_options": 4,
            "static_extension_terminal_safe_stops": 2,
            "unsafe_extension_options_avoided_by_selector": 4,
            "terminal_extension_records": [],
        },
        "policy_aggregate_snapshot": {
            "contextual_policy": {
                "condition": "contextual_post_stop_conversion_policy",
                "mean_terminal_adjusted_progress": 142.857143,
                "terminal_rate": 0.0,
            }
        },
        "blocked_capability_hypotheses": [
            "proxy_progress_not_completion_condition",
            "objective_readiness_detector_missing",
            "terminal_commit_or_submit_action_missing",
            "goal_representation_missing_beyond_safe_progress",
            "conversion_state_useful_but_not_completion_trigger",
        ],
        "desired_hypothesis_families": [
            "objective_readiness_detection",
            "post_conversion_commit_action_search",
            "goal_state_representation_beyond_safe_progress",
            "proxy_progress_vs_completion_discriminator",
            "risk_aware_selector_completion_gap",
        ],
        "requested_experiment_styles": [
            "post_selector_objective_readiness_probe",
            "post_conversion_commit_action_matrix",
            "terminal_safe_progress_vs_completion_discriminator",
        ],
        "scientific_questions": [
            "Which signal marks a post-stop state as ready for objective completion?",
            "Does a terminal-safe conversion require a commit action?",
        ],
        "ready_for_risk_aware_objective_frontier_review": True,
        "ready_for_m2_or_m3": False,
        "ready_for_direct_downstream_write": False,
        "ready_for_saturation_handoff": False,
        "a33_ready": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "risk_aware_frontier_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "risk_aware_policy_counted_as_objective_solution": False,
    }
    record.update(overrides)
    return record


def _source_payload(
    records=None,
    *,
    support=0,
    ready=True,
    m2_ready=False,
    m3_write=False,
    verdict=False,
) -> dict:
    records = list(records if records is not None else [_valid_record()])
    return {
        "risk_aware_post_stop_frontier_records": records,
        "summary": {
            "frontier_records": len(records),
            "ready_for_risk_aware_objective_frontier_review": ready,
            "ready_for_m2_or_m3": m2_ready,
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
        "risk_aware_frontier_counted_as_confirmation": False,
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


def test_risk_aware_objective_handoff_schema_builds_candidate_only_request(
    tmp_path: Path,
) -> None:
    path = _write_json(tmp_path / "p2g4.json", _source_payload())

    payload = schema.run_risk_aware_objective_frontier_handoff_schema(
        risk_aware_frontier_records_path=path,
    )

    assert payload["summary"]["risk_aware_objective_reviews_accepted"] == 1
    assert payload["summary"]["risk_aware_objective_handoff_requests"] == 1
    assert payload["summary"]["handoff_type"] == (
        "RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST"
    )
    assert payload["summary"]["target"] == "M2_OR_M3"
    assert payload["summary"]["target_modules"] == ["M2.G2"]
    assert payload["summary"]["ready_for_direct_downstream_write"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    request = payload["risk_aware_objective_handoff_requests"][0]
    assert request["source_frontier_id"] == (
        "p2g4::bp35::risk_aware_post_stop_no_objective_completion"
    )
    assert request["frontier_type"] == (
        RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER
    )
    assert request["blocked_capability"] == (
        OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
    )
    assert request["ready_for_m2_or_m3_risk_aware_objective_branch"] is True
    assert request["ready_for_direct_downstream_write"] is False
    assert request["support"] == 0
    assert "objective_readiness_detector_missing" in request[
        "blocked_capability_hypotheses"
    ]
    assert "objective_readiness_detection" in request[
        "requested_hypothesis_families"
    ]
    matrix = request["suggested_initial_experiment_matrix"]
    assert "ACTION6,ACTION3" in matrix["source_policy_options"]
    assert "ACTION1" in matrix["post_conversion_commit_action_candidates"]
    assert "objective_completion_signal" in matrix["success_metrics"]


def test_risk_aware_objective_handoff_schema_noops_without_records(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "p2g4.json", _source_payload(records=[]))

    payload = schema.run_risk_aware_objective_frontier_handoff_schema(
        risk_aware_frontier_records_path=path,
    )

    assert payload["frontier_evaluations"] == []
    assert payload["risk_aware_objective_handoff_requests"] == []
    assert payload["summary"]["risk_aware_objective_handoff_requests"] == 0
    assert payload["summary"]["ready_for_m2_or_m3_risk_aware_objective_branch"] is False
    assert payload["summary"]["support"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"support": 1}, "support must remain 0"),
        ({"ready": False}, "review-ready source"),
        ({"m2_ready": True}, "cannot already be ready for M2/M3"),
        ({"m3_write": True}, "must not have m3_write_performed"),
        ({"verdict": True}, "scientific verdict"),
    ],
)
def test_risk_aware_objective_handoff_schema_rejects_bad_source(
    tmp_path: Path,
    kwargs,
    message,
) -> None:
    path = _write_json(tmp_path / "p2g4.json", _source_payload(**kwargs))

    with pytest.raises(ValueError, match=message):
        schema.run_risk_aware_objective_frontier_handoff_schema(
            risk_aware_frontier_records_path=path,
        )


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (_valid_record(evidence={**_valid_record()["evidence"], "terminal_rate": 0.5}), "TERMINAL_SAFETY_NOT_OBSERVED"),
        (
            _valid_record(
                evidence={
                    **_valid_record()["evidence"],
                    "objective_completion_signal": True,
                }
            ),
            "OBJECTIVE_COMPLETION_ALREADY_OBSERVED",
        ),
        (_valid_record(ready_for_m2_or_m3=True), "INPUT_ALREADY_READY_FOR_M2_OR_M3"),
        (_valid_record(support=1), "SUPPORT_MUST_REMAIN_ZERO"),
    ],
)
def test_evaluate_risk_aware_objective_frontier_rejects_invalid_record(
    record,
    reason,
) -> None:
    evaluation = schema.evaluate_risk_aware_objective_frontier_for_review(record)

    assert evaluation["risk_aware_objective_review_accepted"] is False
    assert reason in evaluation["rejection_reasons"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"support": 1}, "support must remain 0"),
        ({"a33_ready": True}, "A33-ready"),
        ({"ready_for_direct_downstream_write": True}, "direct downstream write"),
        ({"handoff_request_counted_as_confirmation": True}, "count as confirmation"),
        ({"policy_result_counted_as_scientific_verdict": True}, "scientific verdict"),
        ({"blocked_capability_hypotheses": []}, "needs blocked hypotheses"),
        ({"requested_hypothesis_families": []}, "needs hypothesis families"),
        ({"requested_experiment_styles": []}, "needs experiment styles"),
        ({"scientific_questions": []}, "needs scientific questions"),
    ],
)
def test_validate_risk_aware_objective_request_rejects_bad_request(
    override,
    message,
) -> None:
    review = schema.evaluate_risk_aware_objective_frontier_for_review(_valid_record())
    request = schema.build_risk_aware_objective_request(
        review,
        request_index=1,
        source_records_path="p2g4.json",
    )
    request.update(override)

    with pytest.raises(ValueError, match=message):
        schema.validate_risk_aware_objective_request(request)


def test_validate_risk_aware_objective_request_rejects_wrong_target_or_reason() -> None:
    review = schema.evaluate_risk_aware_objective_frontier_for_review(_valid_record())
    request = schema.build_risk_aware_objective_request(
        review,
        request_index=1,
        source_records_path="p2g4.json",
    )
    request["target"] = "A33"
    with pytest.raises(ValueError, match="target must be M2_OR_M3"):
        schema.validate_risk_aware_objective_request(request)

    request["target"] = "M2_OR_M3"
    request["frontier_reason"] = "OTHER"
    with pytest.raises(ValueError, match="frontier_reason"):
        schema.validate_risk_aware_objective_request(request)
