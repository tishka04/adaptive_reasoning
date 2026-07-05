import json

import theory.m3.dynamic_retarget_selection_rule_induction as induction


def _mechanism_candidate():
    return {
        "mechanism_candidate_id": "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region",
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "game_id": "bp35-0a0ad940",
        "candidate_mechanic": "repositioning_opens_new_action6_target_region",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "initial_consumed_args": {"x": 18, "y": 0},
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
        "selection_problem_open": True,
        "mechanism_support_events": 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload(candidate):
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "mechanism_candidates": [candidate],
        "support": 0,
        "wrong_confirmations": 0,
    }


def test_dynamic_retarget_selection_rule_induction_builds_rules(tmp_path):
    path = tmp_path / "dynamic_retarget_mechanism_candidates.json"
    path.write_text(json.dumps(_payload(_mechanism_candidate())), encoding="utf-8")

    payload = induction.run_dynamic_retarget_selection_rule_induction(
        mechanism_candidates_path=path
    )

    summary = payload["summary"]
    assert summary["mechanism_candidates_consumed"] == 1
    assert summary["selection_rule_sets"] == 1
    assert summary["candidate_rules"] == 3
    assert summary["rules_with_falsification"] == 3
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    rule_set = payload["selection_rule_sets"][0]
    assert (
        rule_set["selection_rule_candidate_id"]
        == "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule"
    )
    assert rule_set["successful_retargets"] == [
        {"x": 12, "y": 0},
        {"x": 24, "y": 0},
        {"x": 30, "y": 12},
        {"x": 36, "y": 12},
    ]
    assert rule_set["failed_retargets"] == [{"x": 30, "y": 0}]
    assert rule_set["positive_metrics"] == [
        "local_patch_before_after",
        "object_positions_before_after",
    ]
    assert rule_set["non_decisive_or_negative_metrics"] == [
        "changed_pixels",
        "contact_graph_before_after",
    ]
    assert rule_set["observed_contrasts"]["same_x_mixed_outcomes"] == [
        {
            "x": 30,
            "successful_y_values": [12],
            "failed_y_values": [0],
        }
    ]
    assert rule_set["observed_contrasts"]["same_y_mixed_outcomes"] == [
        {
            "y": 0,
            "successful_x_values": [12, 24],
            "failed_x_values": [30],
        }
    ]
    families = {rule["rule_family"] for rule in rule_set["candidate_rules"]}
    assert families == {
        "row_or_band_dependent_retarget",
        "local_patch_transformability",
        "specific_effect_over_global_pixels",
    }
    for rule in rule_set["candidate_rules"]:
        assert rule["falsification_criterion"]
        assert rule["support"] == 0
        assert rule["truth_status"] == "NOT_EVALUATED_BY_M3"
        assert rule["revision_performed"] is False
        assert rule["wrong_confirmations"] == 0
        assert rule["rule_counted_as_confirmation"] is False


def test_dynamic_retarget_selection_rule_induction_skips_without_contrast(tmp_path):
    candidate = _mechanism_candidate()
    candidate["failed_retargets"] = []
    path = tmp_path / "dynamic_retarget_mechanism_candidates.json"
    path.write_text(json.dumps(_payload(candidate)), encoding="utf-8")

    payload = induction.run_dynamic_retarget_selection_rule_induction(
        mechanism_candidates_path=path
    )

    assert payload["summary"]["mechanism_candidates_consumed"] == 1
    assert payload["summary"]["selection_rule_sets"] == 0
    assert payload["summary"]["candidate_rules"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["selection_rule_sets"] == []
