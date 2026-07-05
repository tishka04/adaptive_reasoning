import json

import pytest

from theory.m2 import objective_conversion_hypothesis_generator as generator
from theory.m2.metric_registry import is_metric_measurable


def _objective_conversion_request(**overrides):
    row = {
        "request_id": "p2g3::bp35-0a0ad940::objective_conversion::001",
        "source_frontier_id": "p2g1::bp35::terminal_safe_but_passive::objective_conversion",
        "handoff_type": "OBJECTIVE_CONVERSION_FRONTIER_REQUEST",
        "target": "M2_OR_M3",
        "target_modules": ["M2.G1", "M3.G1"],
        "game_id": "bp35-0a0ad940",
        "frontier_type": "OBJECTIVE_CONVERSION_FRONTIER",
        "frontier_reason": "TERMINAL_SAFE_BUT_PASSIVE",
        "blocked_capability": "objective_conversion_after_safe_stop",
        "objective_conversion_review_accepted": True,
        "candidate_problem_statement": (
            "terminal_safe_policy_state_does_not_convert_to_objective_completion"
        ),
        "requested_hypothesis_families": [
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
        "suggested_initial_experiment_matrix": {
            "base_state_family": "terminal_safe_stop_or_avoidance_state",
            "single_step_actions": ["ACTION3", "ACTION4", "ACTION6"],
            "short_sequences": [
                ["ACTION3", "ACTION4"],
                ["ACTION4", "ACTION3"],
                ["ACTION6", "ACTION3"],
                ["ACTION6", "ACTION4"],
            ],
            "controls": ["hold_or_stop_state", "relation_progress_policy"],
            "success_metrics": [
                "objective_completion_signal",
                "levels_completed_after_rollout",
                "terminal_adjusted_progress_after_stop",
            ],
            "diagnostic_metrics": [
                "relation_delta_after_stop",
                "terminal_reentry_rate",
                "changed_pixels",
            ],
        },
        "ready_for_objective_conversion_hypothesis_generation": True,
        "ready_for_m2_or_m3_objective_conversion_branch": True,
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


def _payload(requests=None, *, support=0, direct_write=False, m3_write=False):
    requests = list(requests or [])
    return {
        "config": {
            "schema_version": "p2.objective_conversion_handoff_requests.v1",
        },
        "objective_conversion_handoff_requests": requests,
        "summary": {
            "source_objective_conversion_reviews_accepted": len(requests),
            "objective_conversion_handoff_requests": len(requests),
            "ready_for_m2_or_m3_objective_conversion_branch": bool(requests),
            "ready_for_direct_downstream_write": direct_write,
            "a33_ready": False,
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
        "policy_result_counted_as_scientific_verdict": False,
        "ready_for_direct_downstream_write": direct_write,
        "a33_ready": False,
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


def test_objective_conversion_metrics_are_registered_for_testability():
    assert is_metric_measurable("final_game_state")
    assert is_metric_measurable("levels_completed_after_rollout")
    assert is_metric_measurable("terminal_state_after_rollout")
    assert is_metric_measurable("objective_progress_proxy")


def test_objective_conversion_hypotheses_generate_twelve_falsifiable_candidates(tmp_path):
    path = tmp_path / "objective_conversion_requests.json"
    path.write_text(
        json.dumps(_payload([_objective_conversion_request()])),
        encoding="utf-8",
    )

    payload = generator.run_objective_conversion_hypothesis_generator(
        objective_conversion_handoff_requests_path=path,
    )

    assert payload["summary"]["objective_conversion_requests_consumed"] == 1
    assert payload["summary"]["hypotheses_generated"] == 12
    assert payload["summary"]["testable_hypotheses"] == 12
    assert payload["summary"]["ready_for_m3_candidate_experiment_request"] == 12
    assert payload["summary"]["all_requested_hypothesis_families_covered"] is True
    assert payload["summary"]["all_hypotheses_have_falsification_signal"] is True
    assert payload["summary"]["all_hypotheses_map_to_requested_experiment_style"] is True
    assert payload["summary"]["execution_performed"] is False
    assert payload["summary"]["policy_rollout_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["truth_status"] == "NOT_EVALUATED_BY_M2"

    batch = payload["objective_conversion_hypothesis_batches"][0]
    families = {
        row["hypothesis_family"] for row in batch["candidate_hypotheses"]
    }
    assert families == {
        "post_safe_stop_objective_conversion",
        "subgoal_target_reselection",
        "objective_readiness_condition",
        "terminal_safe_sequence_search",
    }
    styles = {
        row["requested_experiment_style"] for row in batch["candidate_hypotheses"]
    }
    assert styles == set(_objective_conversion_request()["requested_experiment_styles"])

    for hypothesis in batch["candidate_hypotheses"]:
        assert hypothesis["status"] == "UNRESOLVED"
        assert hypothesis["support"] == 0
        assert hypothesis["revision_status"] == "CANDIDATE_ONLY"
        assert hypothesis["truth_status"] == "NOT_EVALUATED_BY_M2"
        assert hypothesis["revision_performed"] is False
        assert hypothesis["wrong_confirmations"] == 0
        assert hypothesis["controlled_test_required"] is True
        assert hypothesis["ready_for_m3_candidate_experiment_request"] is True
        assert hypothesis["objective_conversion_hypothesis_counted_as_confirmation"] is False
        assert hypothesis["policy_result_counted_as_scientific_verdict"] is False
        assert hypothesis["falsification"]["support_condition"]
        assert hypothesis["falsification"]["failure_condition"]
        assert hypothesis["falsification_signal"]
        assert hypothesis["requested_experiment_style"]
        assert hypothesis["source_generation"]["priority_score_counted_as_support"] is False
        assert hypothesis["trace_support_counted_as_proof"] is False
        assert hypothesis["prior_counted_as_proof"] is False


def test_objective_conversion_hypotheses_reject_invalid_request_contract(tmp_path):
    path = tmp_path / "objective_conversion_requests.json"
    invalid = _objective_conversion_request(ready_for_direct_downstream_write=True)
    path.write_text(json.dumps(_payload([invalid])), encoding="utf-8")

    payload = generator.run_objective_conversion_hypothesis_generator(
        objective_conversion_handoff_requests_path=path,
    )

    assert payload["summary"]["objective_conversion_requests_consumed"] == 0
    assert payload["summary"]["objective_conversion_requests_rejected"] == 1
    assert payload["summary"]["hypotheses_generated"] == 0
    assert payload["summary"]["support"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"support": 1}, "support must remain 0"),
        ({"direct_write": True}, "direct write"),
        ({"m3_write": True}, "must not have m3_write_performed"),
    ],
)
def test_objective_conversion_hypotheses_reject_bad_source_payload(
    tmp_path,
    kwargs,
    message,
):
    path = tmp_path / "objective_conversion_requests.json"
    path.write_text(
        json.dumps(_payload([_objective_conversion_request()], **kwargs)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        generator.run_objective_conversion_hypothesis_generator(
            objective_conversion_handoff_requests_path=path,
        )


def test_objective_conversion_hypotheses_noop_on_empty_request_set(tmp_path):
    path = tmp_path / "objective_conversion_requests.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = generator.run_objective_conversion_hypothesis_generator(
        objective_conversion_handoff_requests_path=path,
    )

    assert payload["objective_conversion_hypothesis_batches"] == []
    assert payload["summary"]["objective_conversion_requests_consumed"] == 0
    assert payload["summary"]["hypotheses_generated"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["execution_performed"] is False
    assert payload["policy_rollout_performed"] is False
