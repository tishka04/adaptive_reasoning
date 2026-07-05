import json

import theory.m3.dynamic_retarget_followup_executor as executor


def _request(rank, args):
    return {
        "request_id": f"m3_11::retarget_ACTION6_{rank:02d}",
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "source_hypothesis_ids": [
            "m2::after_ACTION3_live_after_ACTION6::h001",
            "m2::after_ACTION3_live_after_ACTION6::h002",
        ],
        "game_id": "bp35-0a0ad940",
        "hypothesis_tested": "ACTION4 may create a new ACTION6 target after repositioning",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "target_action_args": dict(args),
        "target_action_arg_policy": "dynamic_retarget_after_repositioning",
        "suggested_control_actions": ["ACTION3", "ACTION4"],
        "metrics": ["local_patch_before_after", "changed_pixels"],
        "candidate_arg_rank": rank,
        "candidate_arg_score": 0.0,
        "candidate_arg_generation_sources": [
            "live_available_target_after_replay",
            "object_motion_offset_hint",
        ],
        "status": "READY_FOR_M3_FOLLOWUP",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload(requests):
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "followup_experiment_requests": requests,
    }


def test_dynamic_retarget_executor_aggregates_arg_and_mechanism_levels(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "dynamic_retarget_followup_requests.json"
    path.write_text(
        json.dumps(_payload([_request(1, {"x": 12, "y": 0}), _request(2, {"x": 24, "y": 0})])),
        encoding="utf-8",
    )
    monkeypatch.setattr(executor, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        executor,
        "available_followup_controls",
        lambda request, *, environments_dir: ("ACTION3", "ACTION4"),
    )

    def fake_execute_metric_followup_experiment(
        request,
        *,
        metric,
        control_action,
        target_action_args,
        target_action_arg_policy,
        environments_dir,
    ):
        supported = target_action_args == {"x": 12, "y": 0} and metric == "changed_pixels"
        return {
            "request_id": request["request_id"],
            "source_refined_hypothesis_id": request["source_refined_hypothesis_id"],
            "game_id": request["game_id"],
            "target_action": request["target_action"],
            "target_action_args": dict(target_action_args),
            "target_action_arg_policy": target_action_arg_policy,
            "control_action": control_action,
            "metric": metric,
            "delta": {"effect_size": 3.0 if supported else 0.0},
            "support_events": 1 if supported else 0,
            "contradiction_events": 0,
            "neutral_events": 0 if supported else 1,
            "controlled_experiments_run": 1,
            "status": "UNRESOLVED",
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
        }

    monkeypatch.setattr(
        executor,
        "execute_metric_followup_experiment",
        fake_execute_metric_followup_experiment,
    )

    payload = executor.run_dynamic_retarget_followup_execution(
        retarget_requests_path=path
    )

    summary = payload["summary"]
    assert summary["retarget_requests_consumed"] == 2
    assert summary["retarget_requests_executed"] == 2
    assert summary["controlled_experiments_run"] == 8
    assert summary["tested_candidate_args"] == 2
    assert summary["args_with_grounded_support"] == 1
    assert summary["arg_level_support_events"] == 2
    assert summary["mechanism_support_events"] == 1
    assert summary["arg_level_support_events_counted_as_mechanism_support"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    per_arg = payload["per_arg_results"]
    assert per_arg[0]["target_action_args"] == {"x": 12, "y": 0}
    assert per_arg[0]["arg_has_grounded_support"] is True
    assert per_arg[0]["support"] == 0
    assert per_arg[1]["target_action_args"] == {"x": 24, "y": 0}
    assert per_arg[1]["arg_has_grounded_support"] is False
    mechanism = payload["mechanism_summary"]
    assert mechanism["best_arg"] == {"x": 12, "y": 0}
    assert mechanism["mechanism_support_events"] == 1
    assert mechanism["support"] == 0
    assert payload["updated_candidate_records"][0]["support"] == 0


def test_dynamic_retarget_executor_skips_non_ready_requests(monkeypatch, tmp_path):
    request = _request(1, {"x": 12, "y": 0})
    request["status"] = "BLOCKED_NO_DYNAMIC_RETARGET"
    path = tmp_path / "dynamic_retarget_followup_requests.json"
    path.write_text(json.dumps(_payload([request])), encoding="utf-8")
    monkeypatch.setattr(executor, "_configure_offline_env", lambda env_dir: None)

    payload = executor.run_dynamic_retarget_followup_execution(
        retarget_requests_path=path
    )

    assert payload["summary"]["retarget_requests_consumed"] == 0
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["mechanism_support_events"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
