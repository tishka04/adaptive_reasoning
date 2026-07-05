import json

import theory.m3.a32_requested_patch_similarity_followup_executor as executor


def _request(followup_family, args_list):
    return {
        "request_id": f"m3_22::{followup_family}",
        "source_a32_queue_item_id": (
            "m3_21::bp35::ACTION4_ACTION6::local_patch_transformability"
        ),
        "source_a32_decision": "SCOPE_LIMITED_CANDIDATE_ONLY",
        "source_a32_recommended_next_step": "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS",
        "source_a32_decision_reasons": [
            "mono_game_scope",
            "mono_context_scope",
            "scope_not_a33_ready",
        ],
        "a32_decision_counted_as_confirmation": False,
        "game_id": "bp35-0a0ad940",
        "candidate_rule_family": "local_patch_transformability",
        "followup_family": followup_family,
        "purpose": f"purpose {followup_family}",
        "context_replay": (
            ["ACTION6", "ACTION4"]
            if followup_family == "alternate_repositioning_context_probe"
            else ["ACTION6", "ACTION3", "ACTION4"]
        ),
        "context_replay_args": (
            [{"x": 18, "y": 0}, {}]
            if followup_family == "alternate_repositioning_context_probe"
            else [{"x": 18, "y": 0}, {}, {}]
        ),
        "target_action": "ACTION6",
        "target_action_arg_policy": "patch_similarity_excluding_known_args",
        "resolved_target_action_args": [dict(args) for args in args_list],
        "candidate_resolution_args": [
            {
                "action_args": dict(args),
                "success_patch_distance": 0.0,
                "failure_patch_distance": 3.0,
                "similarity_interpretation": "success_like",
            }
            for args in args_list
        ],
        "excluded_args": [{"x": 30, "y": 0}],
        "seed_successful_args": [{"x": 12, "y": 0}, {"x": 24, "y": 0}],
        "seed_failed_args": [{"x": 30, "y": 0}],
        "suggested_control_actions": ["ACTION3", "ACTION4"],
        "control_policy": "m3_dynamic_available_controls",
        "metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "success_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "diagnostic_metrics": [
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "expected_signal": "scope follow-up signal",
        "falsification_criterion": "success metrics fail",
        "planning_rationale": f"rationale {followup_family}",
        "resolution_basis": f"basis {followup_family}",
        "status": "READY_FOR_M3_A32_PATCH_FOLLOWUP",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "execution_performed": False,
    }


def _payload(requests):
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "a32_requested_followup_requests": requests,
    }


def test_a32_requested_patch_similarity_followup_executor_separates_scope_roles(
    monkeypatch,
    tmp_path,
):
    requests = [
        _request(
            "outside_known_y12_region_probe",
            [{"x": 18, "y": 0}, {"x": 30, "y": 0}],
        ),
        _request(
            "alternate_repositioning_context_probe",
            [{"x": 12, "y": 0}, {"x": 24, "y": 0}],
        ),
    ]
    path = tmp_path / "a32_followup_requests.json"
    path.write_text(json.dumps(_payload(requests)), encoding="utf-8")

    monkeypatch.setattr(executor, "_configure_offline_env", lambda env_dir: None)
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
        is_alternate = request["context_replay"] == ["ACTION6", "ACTION4"]
        success_metric = metric in {
            "local_patch_before_after",
            "object_positions_before_after",
        }
        diagnostic_metric = metric == "changed_pixels"
        support = 1 if is_alternate and success_metric else 0
        contradiction = 1 if (not is_alternate and success_metric) else 0
        if diagnostic_metric:
            contradiction = 1
        return {
            "request_id": request["request_id"],
            "game_id": request["game_id"],
            "hypothesis_tested": request["hypothesis_tested"],
            "context_replay": request["context_replay"],
            "context_replay_args": request["context_replay_args"],
            "target_action": request["target_action"],
            "target_action_args": dict(target_action_args),
            "target_action_arg_policy": target_action_arg_policy,
            "target_action_args_resolved": True,
            "control_action": control_action,
            "control_action_args": {},
            "metric": metric,
            "delta": {"effect_size": 1 if support else -1 if contradiction else 0},
            "support_events": support,
            "contradiction_events": contradiction,
            "neutral_events": 0 if support or contradiction else 1,
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

    payload = executor.run_a32_requested_patch_similarity_followup_execution(
        a32_followup_requests_path=path,
    )

    summary = payload["summary"]
    assert summary["a32_followup_requests_consumed"] == 2
    assert summary["unique_execution_signatures"] == 4
    assert summary["unique_target_arg_sets_executed"] == 4
    assert summary["outside_known_y12_region_signatures"] == 2
    assert summary["alternate_repositioning_context_signatures"] == 2
    assert summary["controlled_experiments_run"] == 32
    assert summary["success_metric_support_events"] == 8
    assert summary["success_metric_contradiction_events"] == 8
    assert summary["diagnostic_contradiction_events"] == 8
    assert summary["alternate_context_args_with_success_metric_support"] == 2
    assert summary["outside_boundary_failures_counted_as_rule_refutation"] is False
    assert summary["diagnostic_contradictions_counted_as_refutation"] is False
    assert summary["a32_decision_counted_as_confirmation"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    outside_rows = [
        row
        for row in payload["per_signature_execution"]
        if row["followup_families"] == ["outside_known_y12_region_probe"]
    ]
    assert len(outside_rows) == 2
    assert all(
        row["scope_interpretation"]
        == "outside_region_boundary_reinforced_candidate_only"
        for row in outside_rows
    )
    assert all(
        row["outside_boundary_failure_counted_as_rule_refutation"] is False
        for row in outside_rows
    )

    alternate_rows = [
        row
        for row in payload["per_signature_execution"]
        if row["followup_families"] == ["alternate_repositioning_context_probe"]
    ]
    assert len(alternate_rows) == 2
    assert all(
        row["scope_interpretation"]
        == "alternate_context_scope_expanded_candidate_only"
        for row in alternate_rows
    )
    assert all(row["support"] == 0 for row in alternate_rows)

    family_summary = {
        row["followup_family"]: row for row in payload["family_summary"]
    }
    assert (
        family_summary["outside_known_y12_region_probe"]["family_role"]
        == "boundary_negative_control_not_global_refutation"
    )
    assert (
        family_summary["outside_known_y12_region_probe"][
            "outside_boundary_failures_counted_as_rule_refutation"
        ]
        is False
    )
    assert (
        family_summary["alternate_repositioning_context_probe"][
            "family_interpretation"
        ]
        == "alternate_context_scope_expanded_candidate_only"
    )


def test_a32_requested_patch_similarity_followup_executor_blocks_without_controls(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "a32_followup_requests.json"
    path.write_text(
        json.dumps(
            _payload(
                [
                    _request(
                        "outside_known_y12_region_probe",
                        [{"x": 18, "y": 0}],
                    )
                ]
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(executor, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        executor,
        "available_followup_controls",
        lambda request, *, environments_dir: (),
    )

    payload = executor.run_a32_requested_patch_similarity_followup_execution(
        a32_followup_requests_path=path,
    )

    assert payload["summary"]["a32_followup_requests_consumed"] == 1
    assert payload["summary"]["unique_execution_signatures"] == 1
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["blocked_experiments"] == 1
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["per_signature_execution"][0]["status"] == "BLOCKED_NOT_EXECUTED"
    assert (
        payload["per_signature_execution"][0][
            "outside_boundary_failure_counted_as_rule_refutation"
        ]
        is False
    )


def test_a32_requested_patch_similarity_followup_executor_handles_empty_input(tmp_path):
    path = tmp_path / "a32_followup_requests.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = executor.run_a32_requested_patch_similarity_followup_execution(
        a32_followup_requests_path=path,
    )

    assert payload["summary"]["a32_followup_requests_consumed"] == 0
    assert payload["summary"]["unique_execution_signatures"] == 0
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
