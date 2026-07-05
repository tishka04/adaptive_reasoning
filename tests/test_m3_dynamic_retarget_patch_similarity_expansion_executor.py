import json

import theory.m3.dynamic_retarget_patch_similarity_expansion_executor as executor


def _request(probe_family, request_id=None):
    return {
        "request_id": request_id or f"m3_18::{probe_family}::x48_y12",
        "source_selection_rule_consolidation_id": (
            "m3_17::bp35::ACTION4_ACTION6::selection_rule_consolidation"
        ),
        "source_selection_rule_candidate_id": (
            "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule"
        ),
        "source_mechanism_candidate_id": (
            "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region"
        ),
        "game_id": "bp35-0a0ad940",
        "rule_family": "local_patch_transformability",
        "probe_family": probe_family,
        "hypothesis_tested": f"hypothesis {probe_family}",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "target_action_args": {"x": 48, "y": 12},
        "target_action_arg_policy": "patch_similarity_excluding_known_args",
        "resolved_target_action_args": [{"x": 48, "y": 12}],
        "candidate_resolution_args": [
            {
                "action_args": {"x": 48, "y": 12},
                "success_patch_distance": 0.0,
                "failure_patch_distance": 7.0,
                "similarity_interpretation": "success_like",
            }
        ],
        "excluded_args": [{"x": 42, "y": 12}],
        "seed_successful_args": [{"x": 42, "y": 12}],
        "seed_failed_args": [{"x": 30, "y": 0}],
        "suggested_control_actions": ["ACTION3", "ACTION4"],
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
        "planning_rationale": f"rationale {probe_family}",
        "resolution_basis": f"basis {probe_family}",
        "falsification_criterion": f"falsify {probe_family}",
        "status": "READY_FOR_M3_PATCH_EXPANSION",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload(requests):
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "expansion_experiment_requests": requests,
    }


def test_patch_similarity_expansion_executor_deduplicates_rationales(
    monkeypatch,
    tmp_path,
):
    requests = [
        _request("success_patch_similarity_expansion"),
        _request("failure_patch_negative_control_expansion"),
        _request("mixed_patch_boundary_probe"),
    ]
    path = tmp_path / "expansion_requests.json"
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
        support = 1 if metric in {
            "local_patch_before_after",
            "object_positions_before_after",
        } else 0
        contradiction = 1 if metric == "changed_pixels" else 0
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

    payload = executor.run_dynamic_retarget_patch_similarity_expansion_execution(
        expansion_requests_path=path,
    )

    summary = payload["summary"]
    assert summary["expansion_requests_consumed"] == 3
    assert summary["unique_execution_signatures"] == 1
    assert summary["unique_target_arg_sets_executed"] == 1
    assert summary["duplicate_request_rationales_preserved"] == 3
    assert summary["duplicate_request_rationales_counted_as_independent"] is False
    assert summary["controlled_experiments_run"] == 8
    assert summary["success_metric_support_events"] == 4
    assert summary["diagnostic_contradiction_events"] == 2
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    per_signature = payload["per_signature_execution"]
    assert len(per_signature) == 1
    row = per_signature[0]
    assert row["target_action_args"] == {"x": 48, "y": 12}
    assert row["duplicate_request_rationales_preserved"] == 3
    assert row["probe_families"] == [
        "failure_patch_negative_control_expansion",
        "mixed_patch_boundary_probe",
        "success_patch_similarity_expansion",
    ]
    assert row["controlled_experiments_run"] == 8
    assert row["support"] == 0
    assert row["expansion_request_counted_as_support"] is False

    provenance = payload["provenance_rationales"][0]
    assert provenance["duplicate_request_rationales_preserved"] == 3
    assert provenance["duplicate_request_rationales_counted_as_independent"] is False
    assert len(provenance["request_rationales"]) == 3

    expansion_summary = payload["expansion_summary"]
    assert expansion_summary["tested_unique_arg_sets"] == 1
    assert expansion_summary["args_with_grounded_support"] == 1
    assert expansion_summary["mechanism_support_events"] == 1
    assert expansion_summary["signature_level_support_events_counted_as_mechanism_support"] is False


def test_patch_similarity_expansion_executor_blocks_without_controls(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "expansion_requests.json"
    path.write_text(json.dumps(_payload([_request("success_patch_similarity_expansion")])), encoding="utf-8")
    monkeypatch.setattr(executor, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        executor,
        "available_followup_controls",
        lambda request, *, environments_dir: (),
    )

    payload = executor.run_dynamic_retarget_patch_similarity_expansion_execution(
        expansion_requests_path=path,
    )

    assert payload["summary"]["expansion_requests_consumed"] == 1
    assert payload["summary"]["unique_execution_signatures"] == 1
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["blocked_experiments"] == 1
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["per_signature_execution"][0]["status"] == "BLOCKED_NOT_EXECUTED"


def test_patch_similarity_expansion_executor_handles_empty_input(tmp_path):
    path = tmp_path / "expansion_requests.json"
    path.write_text(json.dumps(_payload([])), encoding="utf-8")

    payload = executor.run_dynamic_retarget_patch_similarity_expansion_execution(
        expansion_requests_path=path,
    )

    assert payload["summary"]["expansion_requests_consumed"] == 0
    assert payload["summary"]["unique_execution_signatures"] == 0
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
