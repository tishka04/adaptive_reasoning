import json

from theory.m3.refined_followup_planner import run_refined_followup_planning


def _global_motion_hypothesis():
    return {
        "refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "source_hypothesis_ids": [
            "m2::after_ACTION3_live_after_ACTION6::h001",
            "m2::after_ACTION3_live_after_ACTION6::h002",
        ],
        "game_id": "bp35-0a0ad940",
        "context_replay": ["ACTION6", "ACTION3"],
        "context_replay_args": [{"x": 18, "y": 0}, {}],
        "target_action": "ACTION4",
        "candidate_mechanic": "global_object_repositioning_after_consumption",
        "observed_effect_family": "global_motion",
        "positive_observations": [
            "changed_pixels_support",
            "object_positions_before_after_support",
        ],
        "neutral_observations": [
            "local_patch_before_after_neutral",
            "contact_graph_before_after_neutral",
        ],
        "diagnostic_only_observations": ["topology_before_after"],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload(hypotheses):
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "refined_candidate_hypotheses": hypotheses,
    }


def test_followup_planner_builds_reactivation_request_from_global_motion(tmp_path):
    path = tmp_path / "refined_candidate_hypotheses_from_m2.json"
    path.write_text(json.dumps(_payload([_global_motion_hypothesis()])), encoding="utf-8")

    payload = run_refined_followup_planning(refined_hypotheses_path=path)

    summary = payload["summary"]
    assert summary["refined_hypotheses_consumed"] == 1
    assert summary["followup_requests_generated"] == 1
    assert summary["reactivation_tests_generated"] == 1
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    request = payload["followup_experiment_requests"][0]
    assert request["source_refined_hypothesis_id"] == (
        "m3_8::bp35::A6_A3::ACTION4::global_motion"
    )
    assert request["source_hypothesis_ids"] == [
        "m2::after_ACTION3_live_after_ACTION6::h001",
        "m2::after_ACTION3_live_after_ACTION6::h002",
    ]
    assert request["context_replay"] == ["ACTION6", "ACTION3", "ACTION4"]
    assert request["context_replay_args"] == [{"x": 18, "y": 0}, {}, {}]
    assert request["target_action"] == "ACTION6"
    assert request["suggested_control_actions"] == ["ACTION3", "ACTION4"]
    assert request["control_policy"] == "m3_dynamic_available_controls"
    assert request["metrics"] == [
        "local_patch_before_after",
        "changed_pixels",
        "object_positions_before_after",
        "contact_graph_before_after",
    ]
    assert "re-enables ACTION6" in request["hypothesis_tested"]
    assert request["status"] == "READY_FOR_M3_FOLLOWUP"
    assert request["revision_status"] == "CANDIDATE_ONLY"
    assert request["support"] == 0
    assert request["controlled_test_required"] is True
    assert request["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert request["revision_performed"] is False
    assert request["wrong_confirmations"] == 0
    assert request["followup_request_counted_as_support"] is False


def test_followup_planner_skips_unsupported_refined_mechanic(tmp_path):
    hypothesis = _global_motion_hypothesis()
    hypothesis["refined_hypothesis_id"] = "m3_8::bp35::A6::ACTION3::local_patch"
    hypothesis["candidate_mechanic"] = "local_patch_effect_after_consumption"
    path = tmp_path / "refined_candidate_hypotheses_from_m2.json"
    path.write_text(json.dumps(_payload([hypothesis])), encoding="utf-8")

    payload = run_refined_followup_planning(refined_hypotheses_path=path)

    assert payload["summary"]["followup_requests_generated"] == 0
    assert payload["summary"]["skipped_refined_hypotheses"] == 1
    assert payload["summary"]["execution_performed"] is False
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    skipped = payload["skipped_refined_hypotheses"][0]
    assert skipped["status"] == "BLOCKED_NO_FOLLOWUP"
    assert skipped["support"] == 0
    assert skipped["revision_performed"] is False
    assert skipped["wrong_confirmations"] == 0
