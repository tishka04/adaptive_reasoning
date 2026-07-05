import json

import theory.m3.refined_followup_executor as executor


def _request(metrics=None, target_action_args=None):
    return {
        "request_id": "m3_9::m3_8_bp35_A6_A3_ACTION4_global_motion::retest_ACTION6",
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "source_hypothesis_ids": [
            "m2::after_ACTION3_live_after_ACTION6::h001",
            "m2::after_ACTION3_live_after_ACTION6::h002",
        ],
        "game_id": "bp35-0a0ad940",
        "hypothesis_tested": (
            "ACTION4 is a global repositioning/reset operator that re-enables "
            "ACTION6 after consumption"
        ),
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "target_action_args": target_action_args,
        "suggested_control_actions": ["ACTION3", "ACTION4"],
        "metrics": metrics
        or ["local_patch_before_after", "changed_pixels"],
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


def test_followup_executor_resolves_target_args_from_previous_occurrence(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "refined_followup_experiment_requests.json"
    path.write_text(json.dumps(_payload([_request()])), encoding="utf-8")
    calls = []

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
        calls.append((metric, control_action, dict(target_action_args or {})))
        support = 1 if metric == "changed_pixels" else 0
        return {
            "request_id": request["request_id"],
            "source_refined_hypothesis_id": request["source_refined_hypothesis_id"],
            "game_id": request["game_id"],
            "metric": metric,
            "control_action": control_action,
            "target_action": request["target_action"],
            "target_action_args": dict(target_action_args or {}),
            "target_action_arg_policy": target_action_arg_policy,
            "target_action_args_resolved": target_action_args is not None,
            "support_events": support,
            "contradiction_events": 0,
            "neutral_events": 0 if support else 1,
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

    payload = executor.run_refined_followup_execution(followup_requests_path=path)

    assert all(args == {"x": 18, "y": 0} for _, _, args in calls)
    assert payload["summary"]["followup_requests_consumed"] == 1
    assert payload["summary"]["followup_requests_executed"] == 1
    assert payload["summary"]["controlled_experiments_run"] == 4
    assert payload["summary"]["metrics_executed"] == 2
    assert payload["summary"]["controls_executed"] == 2
    assert payload["summary"]["target_action_args_resolved"] is True
    assert payload["summary"]["support_events"] == 2
    assert payload["summary"]["neutral_events"] == 2
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["updated_candidate_records"][0]["support"] == 0


def test_resolve_target_action_args_prefers_explicit_args():
    args, policy = executor.resolve_target_action_args(
        _request(target_action_args={"x": 24, "y": 0})
    )

    assert args == {"x": 24, "y": 0}
    assert policy == "explicit_target_action_args"


def test_followup_executor_skips_non_ready_requests(monkeypatch, tmp_path):
    request = _request()
    request["status"] = "BLOCKED_NO_FOLLOWUP"
    path = tmp_path / "refined_followup_experiment_requests.json"
    path.write_text(json.dumps(_payload([request])), encoding="utf-8")
    monkeypatch.setattr(executor, "_configure_offline_env", lambda env_dir: None)

    payload = executor.run_refined_followup_execution(followup_requests_path=path)

    assert payload["summary"]["followup_requests_consumed"] == 0
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
