import json

import pytest

from theory.m2 import objective_conditioned_hypotheses as objective
from theory.m2.metric_registry import is_metric_measurable


def _objective_request(**overrides):
    row = {
        "request_id": "p2_o1::bp35-0a0ad940::objective_alignment::001",
        "source_frontier_id": (
            "p2_terminal::bp35::conditional_movement_refresh::"
            "local_affordance_productive_but_terminal"
        ),
        "handoff_type": "OBJECTIVE_ALIGNMENT_FRONTIER_REQUEST",
        "target": "M2_OR_M3",
        "target_modules": ["M2.O1", "M3.O1"],
        "game_id": "bp35-0a0ad940",
        "frontier_type": "OBJECTIVE_ALIGNMENT_FRONTIER",
        "frontier_reason": (
            "LOCAL_AFFORDANCE_PRODUCTIVE_BUT_TERMINAL_OBJECTIVE_FAILED"
        ),
        "objective_review_accepted": True,
        "terminal_runs": 15,
        "terminal_budgets": [64, 96, 128, 192, 256],
        "observed_pattern": {
            "local_affordance_productive": True,
            "terminal_objective_failed": True,
            "movement_refresh_triggers": 0,
            "saturation_handoff_ready": False,
            "known_target_action": "ACTION6",
            "known_failure_state": "GAME_OVER",
            "known_levels_completed": 0,
        },
        "ready_for_m2_or_m3_objective_branch": True,
        "ready_for_objective_hypothesis_generation": True,
        "ready_for_saturation_handoff": False,
        "a33_ready": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P2",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "objective_frontier_request_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }
    row.update(overrides)
    return row


def _payload(requests=None, *, support=0, saturation=False):
    requests = list(requests or [])
    return {
        "config": {
            "schema_version": "p2.objective_frontier_handoff_requests.v1",
        },
        "objective_frontier_requests": requests,
        "summary": {
            "source_objective_reviews_accepted": len(requests),
            "objective_frontier_requests": len(requests),
            "ready_for_m2_or_m3_objective_branch": bool(requests),
            "ready_for_saturation_handoff": bool(saturation),
            "a33_ready": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P2",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": "UNRESOLVED",
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P2",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "objective_frontier_request_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a33_ready": False,
    }


def test_objective_metrics_are_registered_for_testability():
    assert is_metric_measurable("final_game_state")
    assert is_metric_measurable("levels_completed_after_rollout")
    assert is_metric_measurable("terminal_state_after_rollout")
    assert is_metric_measurable("objective_progress_proxy")


def test_objective_conditioned_hypotheses_generate_four_falsifiable_candidates(tmp_path):
    path = tmp_path / "objective_requests.json"
    path.write_text(json.dumps(_payload([_objective_request()])), encoding="utf-8")

    payload = objective.run_objective_conditioned_hypotheses(
        objective_frontier_requests_path=path,
    )

    assert payload["summary"]["objective_requests_consumed"] == 1
    assert payload["summary"]["hypotheses_generated"] == 4
    assert payload["summary"]["testable_hypotheses"] == 4
    assert payload["summary"]["ready_for_m3_candidate_experiment_request"] == 4
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["truth_status"] == "NOT_EVALUATED_BY_M2"
    batch = payload["objective_hypothesis_batches"][0]
    families = {
        row["hypothesis_family"] for row in batch["candidate_hypotheses"]
    }
    assert families == {
        "stop_switch_criterion",
        "terminal_risk_predictor",
        "subgoal_switch_after_local_affordance",
        "global_objective_alignment_metric",
    }
    for hypothesis in batch["candidate_hypotheses"]:
        assert hypothesis["status"] == "UNRESOLVED"
        assert hypothesis["support"] == 0
        assert hypothesis["revision_status"] == "CANDIDATE_ONLY"
        assert hypothesis["truth_status"] == "NOT_EVALUATED_BY_M2"
        assert hypothesis["revision_performed"] is False
        assert hypothesis["wrong_confirmations"] == 0
        assert hypothesis["ready_for_m3_candidate_experiment_request"] is True
        assert hypothesis["source_generation"]["priority_score_counted_as_support"] is False
        assert hypothesis["falsification"]["support_condition"]
        assert hypothesis["falsification"]["failure_condition"]


def test_objective_conditioned_hypotheses_reject_invalid_request_contract(tmp_path):
    path = tmp_path / "objective_requests.json"
    invalid = _objective_request(ready_for_saturation_handoff=True)
    path.write_text(json.dumps(_payload([invalid])), encoding="utf-8")

    payload = objective.run_objective_conditioned_hypotheses(
        objective_frontier_requests_path=path,
    )

    assert payload["summary"]["objective_requests_consumed"] == 0
    assert payload["summary"]["objective_requests_rejected"] == 1
    assert payload["summary"]["hypotheses_generated"] == 0
    assert payload["summary"]["support"] == 0


def test_objective_conditioned_hypotheses_reject_source_support_or_saturation(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(_payload([_objective_request()], support=1)),
        encoding="utf-8",
    )
    saturation_path = tmp_path / "saturation.json"
    saturation_path.write_text(
        json.dumps(_payload([_objective_request()], saturation=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        objective.run_objective_conditioned_hypotheses(
            objective_frontier_requests_path=support_path,
        )
    with pytest.raises(ValueError, match="saturation handoff"):
        objective.run_objective_conditioned_hypotheses(
            objective_frontier_requests_path=saturation_path,
        )


def test_objective_conditioned_hypotheses_noop_on_empty_request_set(tmp_path):
    path = tmp_path / "objective_requests.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = objective.run_objective_conditioned_hypotheses(
        objective_frontier_requests_path=path,
    )

    assert payload["objective_hypothesis_batches"] == []
    assert payload["summary"]["objective_requests_consumed"] == 0
    assert payload["summary"]["hypotheses_generated"] == 0
    assert payload["support"] == 0
