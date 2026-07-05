import json

import pytest

from theory.m3 import objective_stop_switch_experiment_planner as planner


def _hypothesis(*, family="stop_switch_criterion", ready=True, support=0):
    return {
        "hypothesis_id": f"m2_o1::fixture::{family}",
        "source_request_id": "p2_o1::bp35-0a0ad940::objective_alignment::001",
        "game_id": "bp35-0a0ad940",
        "frontier_context_id": "p2_terminal::bp35::objective",
        "frontier_reason": "LOCAL_AFFORDANCE_PRODUCTIVE_BUT_TERMINAL_OBJECTIVE_FAILED",
        "hypothesis_family": family,
        "candidate_action": "ACTION6" if family != "subgoal_switch_after_local_affordance" else "ACTION3",
        "predicted_metric": "terminal_state_after_rollout",
        "predicted_effect": f"{family} predicted effect",
        "rationale": "objective fixture",
        "testability": {
            "testable": ready,
            "target_action": "ACTION6",
            "metric": "terminal_state_after_rollout",
            "expected_signal_type": "objective_condition_signal",
        },
        "falsification": {
            "metric": "terminal_state_after_rollout",
            "support_condition": "target > control",
            "failure_condition": "target <= control",
            "minimum_effect_size": 1,
        },
        "status": "UNRESOLVED",
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M2",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "ready_for_m3_candidate_experiment_request": ready,
    }


def _payload(hypotheses=None, *, support=0, verdict=False):
    hypotheses = list(hypotheses or [])
    return {
        "objective_hypothesis_batches": [
            {
                "source_request_id": "p2_o1::bp35-0a0ad940::objective_alignment::001",
                "candidate_hypotheses": hypotheses,
            }
        ],
        "summary": {
            "hypotheses_generated": len(hypotheses),
            "testable_hypotheses": len([row for row in hypotheses if row.get("testability", {}).get("testable")]),
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M2",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M2",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "objective_hypotheses_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": verdict,
    }


def test_objective_stop_switch_planner_builds_protocol_requests(tmp_path):
    path = tmp_path / "objective_hypotheses.json"
    families = [
        "stop_switch_criterion",
        "terminal_risk_predictor",
        "subgoal_switch_after_local_affordance",
        "global_objective_alignment_metric",
    ]
    path.write_text(
        json.dumps(_payload([_hypothesis(family=family) for family in families])),
        encoding="utf-8",
    )

    payload = planner.run_objective_stop_switch_experiment_planning(
        objective_hypotheses_path=path,
    )

    summary = payload["summary"]
    assert summary["objective_hypotheses_consumed"] == 4
    assert summary["objective_experiment_requests_generated"] == 4
    assert summary["prefix_lengths"] == [6, 12, 24, 48, 64]
    assert summary["conditions_per_request"] == 6
    assert summary["planned_condition_cells"] == 120
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert summary["truth_status"] == "NOT_EVALUATED_BY_M3"

    request = payload["objective_stop_switch_experiment_requests"][0]
    assert request["status"] == "READY_FOR_M3_OBJECTIVE_EXPERIMENT"
    assert request["revision_status"] == "CANDIDATE_ONLY"
    assert request["support"] == 0
    assert request["execution_performed"] is False
    assert request["wrong_confirmations"] == 0
    assert request["objective_request_counted_as_support"] is False
    assert request["policy_result_counted_as_scientific_verdict"] is False
    assert request["experimental_conditions"][0]["condition_id"] == "continue_action6"
    assert request["experimental_conditions"][1]["condition_id"] == "stop_policy"
    assert {row["condition_id"] for row in request["experimental_conditions"][2:]} == {
        "switch_ACTION3",
        "switch_ACTION4",
        "switch_ACTION1",
        "switch_ACTION2",
    }
    assert request["primary_objective_metrics"] == [
        "final_game_state",
        "terminal_state_after_rollout",
        "levels_completed_after_rollout",
        "objective_progress_proxy",
    ]
    assert request["local_diagnostic_metrics"] == [
        "local_effect_metric",
        "repeated_action6_count",
        "useful_action6_steps",
    ]


def test_objective_stop_switch_planner_skips_not_ready_hypotheses(tmp_path):
    path = tmp_path / "objective_hypotheses.json"
    path.write_text(
        json.dumps(
            _payload(
                [
                    _hypothesis(family="stop_switch_criterion", ready=True),
                    _hypothesis(family="terminal_risk_predictor", ready=False),
                ]
            )
        ),
        encoding="utf-8",
    )

    payload = planner.run_objective_stop_switch_experiment_planning(
        objective_hypotheses_path=path,
    )

    assert payload["summary"]["objective_hypotheses_consumed"] == 2
    assert payload["summary"]["objective_experiment_requests_generated"] == 1
    assert payload["summary"]["skipped_objective_hypotheses"] == 1
    assert payload["skipped_objective_hypotheses"][0]["status"] == (
        "BLOCKED_OBJECTIVE_EXPERIMENT"
    )
    assert payload["summary"]["support"] == 0


def test_objective_stop_switch_planner_rejects_source_support_or_verdict(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(_payload([_hypothesis()], support=1)),
        encoding="utf-8",
    )
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text(
        json.dumps(_payload([_hypothesis()], verdict=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        planner.run_objective_stop_switch_experiment_planning(
            objective_hypotheses_path=support_path,
        )
    with pytest.raises(ValueError, match="scientific verdict"):
        planner.run_objective_stop_switch_experiment_planning(
            objective_hypotheses_path=verdict_path,
        )


def test_validate_objective_request_rejects_execution_or_support():
    request = planner.build_objective_stop_switch_request(
        _hypothesis(),
        prefix_lengths=(6,),
    ).to_dict()
    request["execution_performed"] = True
    with pytest.raises(ValueError, match="cannot execute"):
        planner.validate_objective_stop_switch_request(request)

    request["execution_performed"] = False
    request["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        planner.validate_objective_stop_switch_request(request)
