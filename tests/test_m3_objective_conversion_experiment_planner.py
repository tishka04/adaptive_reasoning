"""Tests for M3.G1.1 objective-conversion experiment planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.m2.objective_conversion_hypothesis_generator import (
    DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH,
)
from theory.m3.objective_conversion_experiment_planner import (
    HOLD_OR_STOP_STATE_CONTROL,
    READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT,
    RELATION_PROGRESS_POLICY_CONTROL,
    build_objective_conversion_experiment_request,
    objective_conversion_hypotheses_from_payload,
    run_objective_conversion_experiment_planning,
    validate_objective_conversion_experiment_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES_PATH = REPO_ROOT / DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH


def _load_payload() -> dict:
    return run_objective_conversion_experiment_planning(
        objective_conversion_hypotheses_path=HYPOTHESES_PATH,
    )


def test_planner_emits_one_request_per_ready_hypothesis() -> None:
    payload = _load_payload()
    summary = payload["summary"]
    assert summary["objective_conversion_hypotheses_consumed"] == 12
    assert summary["objective_conversion_experiment_requests_generated"] == 12
    assert len(payload["objective_conversion_experiment_requests"]) == 12


def test_planner_covers_all_four_families() -> None:
    payload = _load_payload()
    assert payload["summary"]["all_four_families_covered"] is True
    families = set(payload["summary"]["covered_hypothesis_families"])
    assert families == {
        "post_safe_stop_objective_conversion",
        "subgoal_target_reselection",
        "objective_readiness_condition",
        "terminal_safe_sequence_search",
    }


def test_every_request_has_both_controls_and_delta_decision_rule() -> None:
    payload = _load_payload()
    for request in payload["objective_conversion_experiment_requests"]:
        control_families = {
            control["condition_family"] for control in request["control_conditions"]
        }
        assert HOLD_OR_STOP_STATE_CONTROL in control_families
        assert RELATION_PROGRESS_POLICY_CONTROL in control_families
        assert (
            "delta_terminal_adjusted_progress_vs_hold"
            in request["decision_rule"]["central"]
        )
        assert "terminal_reentry == false" in request["decision_rule"]["central"]
        assert request["primary_metrics"][0] == (
            "delta_terminal_adjusted_progress_vs_hold"
        )


def test_relation_progress_control_is_horizon_matched_to_candidate_len() -> None:
    payload = _load_payload()
    for request in payload["objective_conversion_experiment_requests"]:
        candidate_horizons = {
            int(condition["post_stop_horizon"])
            for condition in request["candidate_conditions"]
        }
        relation_horizons = {
            int(control["post_stop_horizon"])
            for control in request["control_conditions"]
            if control["condition_family"] == RELATION_PROGRESS_POLICY_CONTROL
        }
        # Every candidate horizon must have a matching relation-progress control.
        assert candidate_horizons <= relation_horizons
        for condition in request["candidate_conditions"]:
            assert int(condition["post_stop_horizon"]) == int(
                condition["candidate_len"]
            )


def test_short_sequence_family_carries_length_two_candidates() -> None:
    payload = _load_payload()
    seq_requests = [
        request
        for request in payload["objective_conversion_experiment_requests"]
        if request["hypothesis_family"] == "terminal_safe_sequence_search"
    ]
    assert seq_requests
    found_length_two = any(
        int(condition["candidate_len"]) == 2
        for request in seq_requests
        for condition in request["candidate_conditions"]
    )
    assert found_length_two


def test_safe_stop_spec_points_to_p3g1_lambda0() -> None:
    payload = _load_payload()
    request = payload["objective_conversion_experiment_requests"][0]
    spec = request["safe_stop_spec"]
    assert spec["source"] == "P3.G1"
    assert spec["policy_condition"] == "objective_aware_abstract_policy_lambda_0"
    assert spec["stop_trigger_reason"] == "objective_aware_terminal_risk_score_stop"
    assert spec["base_state_family"] == "terminal_safe_stop_or_avoidance_state"


def test_guardrails_locked_everywhere() -> None:
    payload = _load_payload()
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["execution_performed"] is False
    assert payload["m2_hypothesis_counted_as_confirmation"] is False
    assert payload["experiment_result_counted_as_scientific_verdict"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    for request in payload["objective_conversion_experiment_requests"]:
        assert request["support"] == 0
        assert request["status"] == READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT
        assert request["truth_status"] == "NOT_EVALUATED_BY_M3"
        assert request["execution_performed"] is False
        validate_objective_conversion_experiment_request(request)


def test_planner_rejects_source_with_support() -> None:
    payload = json.loads(HYPOTHESES_PATH.read_text(encoding="utf-8"))
    payload["summary"]["support"] = 1
    hypotheses = objective_conversion_hypotheses_from_payload(payload)
    # Hypotheses still parse, but the source guard should reject the payload.
    assert hypotheses
    with pytest.raises(ValueError):
        from theory.m3.objective_conversion_experiment_planner import (
            _validate_source_payload,
        )

        _validate_source_payload(payload)


def test_build_request_is_candidate_only_and_validates() -> None:
    payload = json.loads(HYPOTHESES_PATH.read_text(encoding="utf-8"))
    hypotheses = objective_conversion_hypotheses_from_payload(payload)
    request = build_objective_conversion_experiment_request(hypotheses[0])
    validate_objective_conversion_experiment_request(request)
    data = request.to_dict()
    assert data["support"] == 0
    assert data["candidate_conditions"]
