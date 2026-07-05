import json

import numpy as np

import theory.m3.dynamic_retarget_selection_followup_executor as executor


def _request(
    request_id,
    *,
    args=None,
    policy="explicit_selection_rule_probe_args",
    probe_family="same_x_different_band_probe",
    seed_successful_args=None,
    seed_failed_args=None,
):
    return {
        "request_id": request_id,
        "source_selection_rule_candidate_id": "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule",
        "source_mechanism_candidate_id": "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region",
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "source_rule_id": "rule::band",
        "game_id": "bp35-0a0ad940",
        "rule_family": (
            "local_patch_transformability"
            if "patch" in probe_family
            else "row_or_band_dependent_retarget"
        ),
        "probe_family": probe_family,
        "hypothesis_tested": "selection rule followup",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "target_action_args": args,
        "target_action_arg_policy": policy,
        "known_successful_retargets": [{"x": 1, "y": 1}],
        "known_failed_retargets": [{"x": 9, "y": 9}],
        "excluded_args": [{"x": 1, "y": 1}, {"x": 9, "y": 9}],
        "seed_successful_args": seed_successful_args or [],
        "seed_failed_args": seed_failed_args or [],
        "suggested_control_actions": ["ACTION3", "ACTION4"],
        "metrics": ["local_patch_before_after", "changed_pixels"],
        "success_metrics": ["local_patch_before_after"],
        "diagnostic_metrics": ["changed_pixels"],
        "expected_signal": "test",
        "falsification_criterion": "test",
        "status": "READY_FOR_M3_SELECTION_FOLLOWUP",
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


def test_local_patch_similarity_resolution_excludes_known_args(monkeypatch):
    request = _request(
        "dynamic",
        args=None,
        policy="local_patch_similarity_after_repositioning",
        probe_family="local_patch_success_similarity_probe",
        seed_successful_args=[{"x": 1, "y": 1}],
    )
    grid = np.array(
        [
            [1, 1, 1, 8],
            [1, 2, 1, 8],
            [1, 1, 1, 8],
            [7, 7, 7, 8],
        ],
        dtype=np.int32,
    )
    monkeypatch.setattr(executor, "grid_after_replay", lambda request, *, environments_dir: grid)

    selected, candidates = executor.resolve_local_patch_similarity_args(
        request,
        available_args=[{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 3}],
        environments_dir="env",
        max_dynamic_args_per_request=1,
    )

    assert selected == ({"x": 2, "y": 1},)
    assert candidates[0]["action_args"] == {"x": 2, "y": 1}
    assert {"x": 1, "y": 1} not in [row["action_args"] for row in candidates]


def test_selection_followup_executor_runs_resolved_and_blocks_unavailable(
    monkeypatch,
    tmp_path,
):
    requests = [
        _request("explicit_available", args={"x": 4, "y": 4}),
        _request("explicit_unavailable", args={"x": 8, "y": 8}),
        _request(
            "dynamic_patch",
            args=None,
            policy="local_patch_similarity_after_repositioning",
            probe_family="local_patch_success_similarity_probe",
            seed_successful_args=[{"x": 1, "y": 1}],
        ),
    ]
    path = tmp_path / "selection_followup_requests.json"
    path.write_text(json.dumps(_payload(requests)), encoding="utf-8")

    grid = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    monkeypatch.setattr(executor, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        executor,
        "available_args_for_request",
        lambda request, *, environments_dir: ({"x": 4, "y": 4}, {"x": 2, "y": 1}),
    )
    monkeypatch.setattr(executor, "grid_after_replay", lambda request, *, environments_dir: grid)
    monkeypatch.setattr(
        executor,
        "available_followup_controls",
        lambda request, *, environments_dir: ("ACTION3", "ACTION4"),
    )

    def fake_execute(
        request,
        *,
        metric,
        control_action,
        target_action_args,
        target_action_arg_policy,
        environments_dir,
    ):
        support = 1 if metric == "local_patch_before_after" else 0
        contradiction = 1 if metric == "changed_pixels" else 0
        return {
            "request_id": request["request_id"],
            "source_refined_hypothesis_id": request["source_refined_hypothesis_id"],
            "game_id": request["game_id"],
            "hypothesis_tested": request["hypothesis_tested"],
            "context_replay": request["context_replay"],
            "context_replay_args": request["context_replay_args"],
            "target_action": request["target_action"],
            "target_action_args": dict(target_action_args),
            "target_action_arg_policy": target_action_arg_policy,
            "target_action_args_resolved": True,
            "control_action": control_action,
            "metric": metric,
            "delta": {"effect_size": 1 if support else -1},
            "support_events": support,
            "contradiction_events": contradiction,
            "neutral_events": 0,
            "controlled_experiments_run": 1,
            "diagnostic_only": False,
            "status": "UNRESOLVED",
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
        }

    monkeypatch.setattr(executor, "execute_metric_followup_experiment", fake_execute)

    payload = executor.run_dynamic_retarget_selection_followup_execution(
        selection_followup_requests_path=path,
    )

    summary = payload["summary"]
    assert summary["selection_followup_requests_consumed"] == 3
    assert summary["selection_followup_requests_executed"] == 2
    assert summary["controlled_experiments_run"] == 8
    assert summary["explicit_requests"] == 2
    assert summary["explicit_requests_available"] == 1
    assert summary["explicit_requests_blocked_unavailable"] == 1
    assert summary["explicit_requests_blocked_excluded"] == 0
    assert summary["dynamic_arg_resolution_requests"] == 1
    assert summary["dynamic_arg_resolution_requests_resolved"] == 1
    assert summary["resolved_request_arg_pairs"] == 2
    assert summary["unique_resolved_target_arg_sets"] == 2
    assert summary["duplicate_resolved_target_arg_sets"] == 0
    assert summary["duplicate_resolved_target_arg_sets_counted_as_independent"] is False
    assert summary["success_metric_support_events"] == 4
    assert summary["diagnostic_contradiction_events"] == 4
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    resolutions = payload["target_arg_resolutions"]
    assert resolutions[1]["blocked_reason"] == "explicit_args_not_available_after_replay"
    assert resolutions[2]["resolved_target_action_args"] == [{"x": 2, "y": 1}]
    assert resolutions[2]["resolution_basis"] == "nearest_patch_signature_to_success_seeds"
    assert resolutions[2]["excluded_known_args_respected"] is True

    per_request = payload["per_request_results"]
    assert len(per_request) == 3
    assert any(row["status"] == "BLOCKED_NOT_EXECUTED" for row in per_request)
    assert all(row["support"] == 0 for row in per_request)
    assert payload["rule_family_summary"][0]["support"] == 0


def test_selection_followup_executor_handles_empty_input(tmp_path):
    path = tmp_path / "selection_followup_requests.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = executor.run_dynamic_retarget_selection_followup_execution(
        selection_followup_requests_path=path,
    )

    assert payload["summary"]["selection_followup_requests_consumed"] == 0
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
