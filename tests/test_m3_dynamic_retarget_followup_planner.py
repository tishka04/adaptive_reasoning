import json

import theory.m3.dynamic_retarget_followup_planner as planner


def _row(metric, *, control_action="ACTION3"):
    observed_baseline = {
        "metric": metric,
        "measurable": True,
        "changed_pixels": 47,
    }
    if metric == "object_positions_before_after":
        observed_baseline["motion_vectors"] = [
            {"color": 9, "dx": 6.0, "dy": 0.0, "size": 6},
            {"color": 11, "dx": -8.0, "dy": 0.0, "size": 2},
        ]
        observed_baseline["moved_component_count"] = 5
    return {
        "request_id": "m3_9::m3_8_bp35_A6_A3_ACTION4_global_motion::retest_ACTION6",
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "source_hypothesis_ids": [
            "m2::after_ACTION3_live_after_ACTION6::h001",
            "m2::after_ACTION3_live_after_ACTION6::h002",
        ],
        "game_id": "bp35-0a0ad940",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "target_action_args": {"x": 18, "y": 0},
        "control_action": control_action,
        "metric": metric,
        "baseline_signal": 47.0 if metric == "changed_pixels" else 5.0,
        "perturbation_signal": 1.0 if metric == "changed_pixels" else 2.0,
        "observed_baseline": observed_baseline,
        "observed_perturbation": {
            "metric": metric,
            "measurable": True,
            "changed_pixels": 1,
        },
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "support_events": 0,
        "contradiction_events": 1,
        "neutral_events": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload(rows):
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "controlled_experiments": rows,
    }


def test_dynamic_retarget_planner_generates_bounded_new_action6_args(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "refined_followup_experiment_results.json"
    path.write_text(
        json.dumps(
            _payload(
                [
                    _row("changed_pixels", control_action="ACTION3"),
                    _row("object_positions_before_after", control_action="ACTION3"),
                    _row("changed_pixels", control_action="ACTION4"),
                    _row("object_positions_before_after", control_action="ACTION4"),
                ]
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(planner, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        planner,
        "available_target_args_after_replay",
        lambda **kwargs: (
            {"x": 18, "y": 0},
            {"x": 24, "y": 0},
            {"x": 12, "y": 0},
            {"x": 30, "y": 0},
            {"x": 30, "y": 12},
            {"x": 36, "y": 12},
        ),
    )

    payload = planner.run_dynamic_retarget_followup_planning(
        followup_results_path=path,
        max_candidate_args=3,
    )

    summary = payload["summary"]
    assert summary["followup_results_consumed"] == 1
    assert summary["retarget_groups"] == 1
    assert summary["candidate_args_generated"] == 3
    assert summary["followup_requests_generated"] == 3
    assert summary["excluded_args_count"] == 1
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    target_args = [
        request["target_action_args"]
        for request in payload["followup_experiment_requests"]
    ]
    assert {"x": 18, "y": 0} not in target_args
    assert {"x": 24, "y": 0} in target_args
    assert {"x": 12, "y": 0} in target_args
    first = payload["followup_experiment_requests"][0]
    assert first["context_replay"] == ["ACTION6", "ACTION3", "ACTION4"]
    assert first["context_replay_args"] == [{"x": 18, "y": 0}, {}, {}]
    assert first["target_action"] == "ACTION6"
    assert first["target_action_arg_policy"] == "dynamic_retarget_after_repositioning"
    assert first["excluded_args"] == [{"x": 18, "y": 0}]
    assert first["suggested_control_actions"] == ["ACTION3", "ACTION4"]
    assert first["metrics"] == [
        "local_patch_before_after",
        "changed_pixels",
        "object_positions_before_after",
        "contact_graph_before_after",
    ]
    assert first["status"] == "READY_FOR_M3_FOLLOWUP"
    assert first["revision_status"] == "CANDIDATE_ONLY"
    assert first["support"] == 0
    assert first["wrong_confirmations"] == 0
    assert first["followup_request_counted_as_support"] is False
    assert "object_motion_offset_hint" in first["candidate_arg_generation_sources"]


def test_dynamic_retarget_planner_skips_when_only_excluded_arg_is_available(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "refined_followup_experiment_results.json"
    path.write_text(
        json.dumps(_payload([_row("object_positions_before_after")])),
        encoding="utf-8",
    )
    monkeypatch.setattr(planner, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        planner,
        "available_target_args_after_replay",
        lambda **kwargs: ({"x": 18, "y": 0},),
    )

    payload = planner.run_dynamic_retarget_followup_planning(
        followup_results_path=path,
        max_candidate_args=5,
    )

    assert payload["summary"]["candidate_args_generated"] == 0
    assert payload["summary"]["followup_requests_generated"] == 0
    assert payload["summary"]["skipped_retarget_groups"] == 1
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["skipped_retarget_groups"][0]["status"] == (
        "BLOCKED_NO_DYNAMIC_RETARGET"
    )
