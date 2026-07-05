"""Tests for M3.G5 risk-aware objective-completion experiment planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.m2.risk_aware_objective_completion_hypothesis_generator import (
    DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH,
    SUBSTRATE_ACTIONS_NOT_TARGETS,
)
from theory.m3.risk_aware_objective_completion_experiment_planner import (
    ACTION6_ONLY_CONTROL,
    COMPILED_STATUS,
    FROZEN_CONTEXTUAL_SELECTOR_CONTROL,
    HOLD_OR_STOP_STATE_CONTROL,
    READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT,
    RELATION_PROGRESS_POLICY_CONTROL,
    STATIC_ACTION6_ACTION3_CONTROL,
    STATIC_ACTION6_ACTION4_CONTROL,
    build_risk_aware_objective_experiment_request,
    risk_aware_objective_hypotheses_from_payload,
    run_risk_aware_objective_completion_experiment_planning,
    validate_risk_aware_objective_experiment_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES_PATH = REPO_ROOT / DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH


def _load_payload() -> dict:
    return run_risk_aware_objective_completion_experiment_planning(
        risk_aware_objective_hypotheses_path=HYPOTHESES_PATH,
    )


def test_planner_emits_one_request_per_m2g2_hypothesis() -> None:
    payload = _load_payload()
    summary = payload["summary"]
    assert summary["risk_aware_objective_hypotheses_consumed"] == 15
    assert summary["risk_aware_objective_experiment_requests_generated"] == 15
    assert len(payload["risk_aware_objective_experiment_requests"]) == 15
    assert payload["planner_outcome_status"] == COMPILED_STATUS
    assert summary["planner_outcome_status"] == COMPILED_STATUS


def test_planner_covers_all_five_families_and_styles() -> None:
    payload = _load_payload()
    assert payload["summary"]["all_five_families_covered"] is True
    assert set(payload["summary"]["covered_hypothesis_families"]) == {
        "objective_readiness_detection",
        "post_conversion_commit_action_search",
        "goal_state_representation_beyond_safe_progress",
        "proxy_progress_vs_completion_discriminator",
        "risk_aware_selector_completion_gap",
    }
    assert set(payload["summary"]["covered_experiment_styles"]) == {
        "post_selector_objective_readiness_probe",
        "post_conversion_commit_action_matrix",
        "terminal_safe_progress_vs_completion_discriminator",
        "horizon_conditioned_completion_trigger_search",
        "risk_aware_policy_ablation_with_completion_metrics",
    }


def test_every_request_has_required_substrates_controls_and_completion_metrics() -> None:
    payload = _load_payload()
    expected_controls = {
        HOLD_OR_STOP_STATE_CONTROL,
        ACTION6_ONLY_CONTROL,
        FROZEN_CONTEXTUAL_SELECTOR_CONTROL,
        STATIC_ACTION6_ACTION3_CONTROL,
        STATIC_ACTION6_ACTION4_CONTROL,
        RELATION_PROGRESS_POLICY_CONTROL,
    }
    expected_substrates = {
        "risk_aware_post_stop_safe_contexts",
        "selector_action6_fallback_contexts",
        "selector_extension_safe_contexts",
        "static_extension_terminal_risk_contexts",
    }
    for request in payload["risk_aware_objective_experiment_requests"]:
        controls = {
            control["condition_id"] for control in request["control_conditions"]
        }
        assert expected_controls <= controls
        substrates = {
            row["category"]
            for row in request["substrate_selection_spec"]["substrate_categories"]
        }
        assert expected_substrates <= substrates
        assert "objective_completion_signal" in request["primary_metrics"]
        assert "levels_completed_after_rollout" in request["primary_metrics"]
        assert "terminal_reentry_rate" in request["primary_metrics"]
        assert "terminal_adjusted_progress_after_stop" in request["safety_metrics"]
        assert "proxy_completion_divergence" in request["diagnostic_metrics"]
        assert "completion_ready_signature" in request["diagnostic_metrics"]
        assert request["required_observables"]
        assert request["candidate_protocols"]
        validate_risk_aware_objective_experiment_request(request)


def test_family_protocols_are_specialized() -> None:
    payload = _load_payload()
    by_family = {}
    for request in payload["risk_aware_objective_experiment_requests"]:
        by_family.setdefault(request["hypothesis_family"], []).append(request)
    assert {
        protocol["protocol_family"]
        for request in by_family["objective_readiness_detection"]
        for protocol in request["candidate_protocols"]
    } == {"objective_readiness_detection"}
    assert any(
        "ACTION7" in protocol.get("optional_commit_actions_if_available", [])
        for request in by_family["post_conversion_commit_action_search"]
        for protocol in request["candidate_protocols"]
    )
    assert {
        protocol["protocol_id"]
        for request in by_family["goal_state_representation_beyond_safe_progress"]
        for protocol in request["candidate_protocols"]
    } == {"goal_representation_discriminator"}
    assert {
        protocol["protocol_id"]
        for request in by_family["proxy_progress_vs_completion_discriminator"]
        for protocol in request["candidate_protocols"]
    } == {"proxy_completion_discriminator"}
    assert {
        protocol["protocol_id"]
        for request in by_family["risk_aware_selector_completion_gap"]
        for protocol in request["candidate_protocols"]
    } == {"selector_gap_policy_ablation"}


def test_action6_led_extensions_are_controls_not_target_protocols() -> None:
    payload = _load_payload()
    forbidden = set(SUBSTRATE_ACTIONS_NOT_TARGETS)
    assert payload["summary"]["action6_extension_retest_requests_generated"] is False
    for request in payload["risk_aware_objective_experiment_requests"]:
        for protocol in request["candidate_protocols"]:
            target = (
                protocol.get("target_commit_action")
                or protocol.get("target_detector")
                or protocol.get("target_representation")
                or protocol.get("target_discriminator")
                or protocol.get("target_policy_variant")
                or protocol.get("target")
            )
            assert target not in forbidden
    static_controls = {
        STATIC_ACTION6_ACTION3_CONTROL,
        STATIC_ACTION6_ACTION4_CONTROL,
    }
    for request in payload["risk_aware_objective_experiment_requests"]:
        controls = {
            control["condition_id"] for control in request["control_conditions"]
        }
        assert static_controls <= controls


def test_guardrails_locked_everywhere() -> None:
    payload = _load_payload()
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["execution_performed"] is False
    assert payload["policy_rollout_performed"] is False
    assert payload["environment_step_performed"] is False
    assert payload["m2_hypothesis_counted_as_confirmation"] is False
    assert payload["experiment_result_counted_as_scientific_verdict"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    for request in payload["risk_aware_objective_experiment_requests"]:
        assert request["support"] == 0
        assert request["status"] == READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT
        assert request["truth_status"] == "NOT_EVALUATED_BY_M3"
        assert request["execution_performed"] is False
        assert request["policy_rollout_performed"] is False
        assert request["environment_step_performed"] is False
        assert request["m2_hypothesis_counted_as_confirmation"] is False
        assert request["experiment_request_counted_as_support"] is False
        assert request["experiment_result_counted_as_scientific_verdict"] is False
        assert request["a32_write_performed"] is False
        assert request["a33_write_performed"] is False


def test_planner_rejects_source_with_support_or_execution() -> None:
    payload = json.loads(HYPOTHESES_PATH.read_text(encoding="utf-8"))
    payload["summary"]["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        from theory.m3.risk_aware_objective_completion_experiment_planner import (
            _validate_source_payload,
        )

        _validate_source_payload(payload)

    payload = json.loads(HYPOTHESES_PATH.read_text(encoding="utf-8"))
    payload["summary"]["environment_step_performed"] = True
    with pytest.raises(ValueError, match="environment_step_performed"):
        from theory.m3.risk_aware_objective_completion_experiment_planner import (
            _validate_source_payload,
        )

        _validate_source_payload(payload)


def test_build_request_is_candidate_only_and_validates() -> None:
    payload = json.loads(HYPOTHESES_PATH.read_text(encoding="utf-8"))
    hypotheses = risk_aware_objective_hypotheses_from_payload(payload)
    request = build_risk_aware_objective_experiment_request(hypotheses[0])
    validate_risk_aware_objective_experiment_request(request)
    data = request.to_dict()
    assert data["support"] == 0
    assert data["candidate_protocols"]
    assert data["substrate_selection_spec"]["selection_counted_as_support"] is False
