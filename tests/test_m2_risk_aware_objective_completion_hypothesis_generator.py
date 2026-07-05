import json

import pytest

from theory.m2 import risk_aware_objective_completion_hypothesis_generator as generator
from theory.m2.metric_registry import is_metric_measurable


def _risk_aware_objective_request(**overrides):
    row = {
        "request_id": "p2g5::bp35-0a0ad940::risk_aware_objective_completion::001",
        "source_frontier_id": "p2g4::bp35::risk_aware_post_stop_no_objective_completion",
        "handoff_type": "RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST",
        "target": "M2_OR_M3",
        "target_modules": ["M2.G2"],
        "game_id": "bp35-0a0ad940",
        "frontier_type": "RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER",
        "frontier_reason": "RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION",
        "blocked_capability": "objective_completion_after_risk_aware_safe_conversion",
        "risk_aware_objective_review_accepted": True,
        "candidate_problem_statement": (
            "risk_aware_safe_post_stop_progress_does_not_complete_objective"
        ),
        "blocked_capability_hypotheses": [
            "proxy_progress_not_completion_condition",
            "objective_readiness_detector_missing",
            "terminal_commit_or_submit_action_missing",
            "goal_representation_missing_beyond_safe_progress",
            "conversion_state_useful_but_not_completion_trigger",
        ],
        "requested_hypothesis_families": [
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
            "horizon_conditioned_completion_trigger_search",
            "risk_aware_policy_ablation_with_completion_metrics",
        ],
        "scientific_questions": [
            "Which signal marks a post-stop state as ready for objective completion?",
            "Does a terminal-safe conversion require a commit action?",
        ],
        "evidence_summary": {
            "contextual_terminal_rate": 0.0,
            "mean_delta_vs_hold": 7.857143,
            "mean_delta_vs_action6_only": 2.857143,
            "objective_completion_signal": False,
            "objective_completion_runs": 0,
            "support": 0,
        },
        "suggested_initial_experiment_matrix": {
            "base_state_family": "risk_aware_terminal_safe_post_stop_conversion_state",
            "source_policy_options": [
                "hold_or_stop_state",
                "ACTION6",
                "ACTION6,ACTION3",
                "ACTION6,ACTION4",
                "contextual_post_stop_conversion_policy",
            ],
            "readiness_feature_candidates": [
                "sampling_family",
                "terminal_horizon_remaining",
                "terminal_horizon_band",
                "hold_baseline_terminal_adjusted_progress",
                "hold_baseline_band",
                "relation_delta_after_stop",
                "new_relation_states",
                "changed_pixels",
                "global_configuration_signature",
            ],
            "post_conversion_commit_action_candidates": [
                "ACTION1",
                "ACTION2",
                "ACTION3",
                "ACTION4",
                "ACTION5",
                "ACTION6",
            ],
            "controls": [
                "hold_or_stop_state",
                "ACTION6_only",
                "frozen_contextual_selector",
                "always_extension_static_policy",
            ],
            "success_metrics": [
                "objective_completion_signal",
                "levels_completed_after_rollout",
            ],
            "safety_metrics": [
                "terminal_reentry_rate",
                "terminal_adjusted_progress_after_stop",
            ],
            "discriminator_metrics": [
                "proxy_progress_without_completion",
                "objective_readiness_precision",
                "commit_action_delta_vs_selector",
            ],
        },
        "ready_for_risk_aware_objective_hypothesis_generation": True,
        "ready_for_m2_or_m3_risk_aware_objective_branch": True,
        "ready_for_direct_downstream_write": False,
        "a33_ready": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P2",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "handoff_request_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }
    row.update(overrides)
    return row


def _payload(
    requests=None,
    *,
    support=0,
    direct_write=False,
    a33_ready=False,
    m3_write=False,
    verdict=False,
):
    requests = list(requests or [])
    return {
        "config": {
            "schema_version": "p2.risk_aware_objective_frontier_handoff_requests.v1",
        },
        "risk_aware_objective_handoff_requests": requests,
        "summary": {
            "risk_aware_objective_handoff_requests": len(requests),
            "ready_for_m2_or_m3_risk_aware_objective_branch": bool(requests),
            "ready_for_direct_downstream_write": direct_write,
            "a33_ready": a33_ready,
            "a40_write_performed": False,
            "m2_write_performed": False,
            "m3_write_performed": m3_write,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P2",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "handoff_request_counted_as_confirmation": False,
        "frontier_validation_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": verdict,
        "ready_for_direct_downstream_write": direct_write,
        "a33_ready": a33_ready,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": m3_write,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P2",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_risk_aware_objective_metrics_are_registered_for_testability():
    assert is_metric_measurable("levels_completed_after_rollout")
    assert is_metric_measurable("terminal_state_after_rollout")
    assert is_metric_measurable("topology_before_after")
    assert is_metric_measurable("contact_graph_before_after")
    assert is_metric_measurable("object_shape_zone_before_after")


def test_risk_aware_objective_hypotheses_generate_completion_gap_candidates(tmp_path):
    path = tmp_path / "risk_aware_objective_requests.json"
    path.write_text(
        json.dumps(_payload([_risk_aware_objective_request()])),
        encoding="utf-8",
    )

    payload = generator.run_risk_aware_objective_completion_hypothesis_generator(
        risk_aware_objective_requests_path=path,
    )

    assert payload["summary"]["risk_aware_objective_requests_consumed"] == 1
    assert payload["summary"]["hypotheses_generated"] == 15
    assert payload["summary"]["testable_hypotheses"] == 15
    assert payload["summary"]["ready_for_m3_g5_candidate_experiment_request"] == 15
    assert payload["summary"]["all_requested_hypothesis_families_covered"] is True
    assert payload["summary"]["all_hypotheses_have_falsification_signal"] is True
    assert payload["summary"]["all_hypotheses_map_to_requested_experiment_style"] is True
    assert payload["summary"]["action6_extension_retest_hypotheses_generated"] is False
    assert payload["summary"]["execution_performed"] is False
    assert payload["summary"]["policy_rollout_performed"] is False
    assert payload["summary"]["environment_step_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["a32_write_performed"] is False
    assert payload["summary"]["a33_write_performed"] is False
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["truth_status"] == "NOT_EVALUATED_BY_M2"

    batch = payload["risk_aware_objective_hypothesis_batches"][0]
    families = {
        row["hypothesis_family"] for row in batch["candidate_hypotheses"]
    }
    assert families == {
        "objective_readiness_detection",
        "post_conversion_commit_action_search",
        "goal_state_representation_beyond_safe_progress",
        "proxy_progress_vs_completion_discriminator",
        "risk_aware_selector_completion_gap",
    }
    styles = {
        row["requested_experiment_style"] for row in batch["candidate_hypotheses"]
    }
    assert styles == set(
        _risk_aware_objective_request()["requested_experiment_styles"]
    )

    forbidden_targets = {"ACTION6", "ACTION6,ACTION3", "ACTION6,ACTION4"}
    for hypothesis in batch["candidate_hypotheses"]:
        assert hypothesis["candidate_action"] not in forbidden_targets
        assert hypothesis["status"] == "UNRESOLVED"
        assert hypothesis["support"] == 0
        assert hypothesis["revision_status"] == "CANDIDATE_ONLY"
        assert hypothesis["truth_status"] == "NOT_EVALUATED_BY_M2"
        assert hypothesis["revision_performed"] is False
        assert hypothesis["wrong_confirmations"] == 0
        assert hypothesis["controlled_test_required"] is True
        assert hypothesis["ready_for_m3_g5"] is True
        assert hypothesis["ready_for_a32"] is False
        assert hypothesis["ready_for_a33"] is False
        assert hypothesis["risk_aware_objective_hypothesis_counted_as_confirmation"] is False
        assert hypothesis["request_counted_as_scientific_verdict"] is False
        assert hypothesis["policy_result_counted_as_scientific_verdict"] is False
        assert hypothesis["required_observables"]
        assert hypothesis["forbidden_interpretations"]
        assert hypothesis["falsification"]["support_condition"]
        assert hypothesis["falsification"]["failure_condition"]
        assert hypothesis["falsification_signal"]
        assert hypothesis["source_generation"]["priority_score_counted_as_support"] is False
        assert hypothesis["trace_support_counted_as_proof"] is False
        assert hypothesis["prior_counted_as_proof"] is False


def test_risk_aware_objective_hypotheses_reject_invalid_request_contract(tmp_path):
    path = tmp_path / "risk_aware_objective_requests.json"
    invalid = _risk_aware_objective_request(ready_for_direct_downstream_write=True)
    path.write_text(json.dumps(_payload([invalid])), encoding="utf-8")

    payload = generator.run_risk_aware_objective_completion_hypothesis_generator(
        risk_aware_objective_requests_path=path,
    )

    assert payload["summary"]["risk_aware_objective_requests_consumed"] == 0
    assert payload["summary"]["risk_aware_objective_requests_rejected"] == 1
    assert payload["summary"]["hypotheses_generated"] == 0
    assert payload["summary"]["support"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"support": 1}, "support must remain 0"),
        ({"direct_write": True}, "direct write"),
        ({"a33_ready": True}, "A33-ready"),
        ({"m3_write": True}, "must not have m3_write_performed"),
        ({"verdict": True}, "scientific verdict"),
    ],
)
def test_risk_aware_objective_hypotheses_reject_bad_source_payload(
    tmp_path,
    kwargs,
    message,
):
    path = tmp_path / "risk_aware_objective_requests.json"
    path.write_text(
        json.dumps(_payload([_risk_aware_objective_request()], **kwargs)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        generator.run_risk_aware_objective_completion_hypothesis_generator(
            risk_aware_objective_requests_path=path,
        )


def test_risk_aware_objective_hypotheses_noop_on_empty_request_set(tmp_path):
    path = tmp_path / "risk_aware_objective_requests.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = generator.run_risk_aware_objective_completion_hypothesis_generator(
        risk_aware_objective_requests_path=path,
    )

    assert payload["risk_aware_objective_hypothesis_batches"] == []
    assert payload["summary"]["risk_aware_objective_requests_consumed"] == 0
    assert payload["summary"]["hypotheses_generated"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["execution_performed"] is False
    assert payload["policy_rollout_performed"] is False
    assert payload["environment_step_performed"] is False
