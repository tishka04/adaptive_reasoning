import json

import theory.m3.dynamic_retarget_selection_followup_planner as planner


def _rule_set():
    return {
        "selection_rule_candidate_id": "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule",
        "source_mechanism_candidate_id": "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region",
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "game_id": "bp35-0a0ad940",
        "candidate_mechanic": "repositioning_opens_new_action6_target_region",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "repositioning_action": "ACTION4",
        "target_action": "ACTION6",
        "successful_retargets": [
            {"x": 12, "y": 0},
            {"x": 24, "y": 0},
            {"x": 30, "y": 12},
            {"x": 36, "y": 12},
        ],
        "failed_retargets": [{"x": 30, "y": 0}],
        "positive_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "non_decisive_or_negative_metrics": [
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "candidate_rules": [
            {
                "rule_id": "rule::band",
                "rule_family": "row_or_band_dependent_retarget",
                "support": 0,
            },
            {
                "rule_id": "rule::patch",
                "rule_family": "local_patch_transformability",
                "support": 0,
            },
            {
                "rule_id": "rule::pixels",
                "rule_family": "specific_effect_over_global_pixels",
                "support": 0,
            },
        ],
        "observed_contrasts": {
            "same_x_mixed_outcomes": [
                {
                    "x": 30,
                    "successful_y_values": [12],
                    "failed_y_values": [0],
                }
            ],
            "same_y_mixed_outcomes": [
                {
                    "y": 0,
                    "successful_x_values": [12, 24],
                    "failed_x_values": [30],
                }
            ],
            "pure_x_rule_blocked": True,
            "pure_y_rule_blocked": True,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def _payload(rule_sets):
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "selection_rule_sets": rule_sets,
    }


def test_selection_followup_planner_generates_bounded_discriminating_requests(
    tmp_path,
):
    path = tmp_path / "dynamic_retarget_selection_rules.json"
    path.write_text(json.dumps(_payload([_rule_set()])), encoding="utf-8")

    payload = planner.run_dynamic_retarget_selection_followup_planning(
        selection_rules_path=path,
        max_followup_requests=8,
    )

    summary = payload["summary"]
    assert summary["selection_rule_sets_consumed"] == 1
    assert summary["candidate_followup_requests"] == 8
    assert summary["followup_requests_generated"] == 8
    assert summary["request_budget_respected"] is True
    assert summary["explicit_arg_requests"] == 6
    assert summary["dynamic_arg_resolution_requests"] == 2
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    requests = payload["followup_experiment_requests"]
    explicit_args = [
        request["target_action_args"]
        for request in requests
        if request["target_action_args"] is not None
    ]
    assert explicit_args == [
        {"x": 30, "y": 6},
        {"x": 30, "y": 18},
        {"x": 30, "y": 24},
        {"x": 6, "y": 0},
        {"x": 18, "y": 0},
        {"x": 36, "y": 0},
    ]
    already_tested = [
        {"x": 12, "y": 0},
        {"x": 24, "y": 0},
        {"x": 30, "y": 0},
        {"x": 30, "y": 12},
        {"x": 36, "y": 12},
    ]
    for args in already_tested:
        assert args not in explicit_args

    patch_requests = [
        request
        for request in requests
        if request["target_action_arg_policy"]
        == "local_patch_similarity_after_repositioning"
    ]
    assert len(patch_requests) == 2
    assert patch_requests[0]["seed_successful_args"]
    assert patch_requests[0]["seed_failed_args"] == []
    assert patch_requests[1]["seed_successful_args"] == []
    assert patch_requests[1]["seed_failed_args"] == [{"x": 30, "y": 0}]
    for request in requests:
        assert request["status"] == "READY_FOR_M3_SELECTION_FOLLOWUP"
        assert request["metrics"] == [
            "local_patch_before_after",
            "object_positions_before_after",
            "changed_pixels",
            "contact_graph_before_after",
        ]
        assert request["support"] == 0
        assert request["truth_status"] == "NOT_EVALUATED_BY_M3"
        assert request["revision_performed"] is False
        assert request["wrong_confirmations"] == 0
        assert request["rule_counted_as_confirmation"] is False
        assert request["followup_request_counted_as_support"] is False
        assert request["execution_performed"] is False


def test_selection_followup_planner_truncates_by_budget(tmp_path):
    path = tmp_path / "dynamic_retarget_selection_rules.json"
    path.write_text(json.dumps(_payload([_rule_set()])), encoding="utf-8")

    payload = planner.run_dynamic_retarget_selection_followup_planning(
        selection_rules_path=path,
        max_followup_requests=4,
    )

    assert payload["summary"]["candidate_followup_requests"] == 8
    assert payload["summary"]["followup_requests_generated"] == 4
    assert payload["summary"]["truncated_followup_requests"] == 4
    assert len(payload["followup_experiment_requests"]) == 4
    assert len(payload["truncated_followup_experiment_requests"]) == 4
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0


def test_selection_followup_planner_handles_empty_input(tmp_path):
    path = tmp_path / "dynamic_retarget_selection_rules.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = planner.run_dynamic_retarget_selection_followup_planning(
        selection_rules_path=path,
    )

    assert payload["summary"]["selection_rule_sets_consumed"] == 0
    assert payload["summary"]["followup_requests_generated"] == 0
    assert payload["followup_experiment_requests"] == []
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
