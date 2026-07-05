import json

import numpy as np

import theory.m3.dynamic_retarget_patch_similarity_expansion_planner as planner


def _consolidation():
    return {
        "selection_rule_consolidation_id": (
            "m3_17::bp35::ACTION4_ACTION6::selection_rule_consolidation"
        ),
        "source_selection_rule_candidate_id": (
            "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule"
        ),
        "source_mechanism_candidate_id": (
            "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region"
        ),
        "game_id": "bp35-0a0ad940",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "best_current_rule_family": "local_patch_transformability",
        "unique_new_successful_args": [{"x": 4, "y": 4}],
        "known_successful_retargets": [{"x": 1, "y": 1}],
        "known_failed_retargets": [{"x": 8, "y": 1}],
        "excluded_args_for_next_expansion": [
            {"x": 1, "y": 1},
            {"x": 4, "y": 4},
            {"x": 8, "y": 1},
            {"x": 18, "y": 0},
        ],
        "success_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "diagnostic_metrics": [
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def _payload():
    return {
        "summary": {"support": 0, "wrong_confirmations": 0},
        "selection_rule_consolidations": [_consolidation()],
    }


def test_patch_similarity_expansion_planner_excludes_known_args(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "selection_rule_consolidation.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    grid = np.array(
        [
            [1, 1, 1, 7, 7, 7, 9, 9, 9, 4],
            [1, 2, 1, 7, 2, 7, 9, 3, 9, 4],
            [1, 1, 1, 7, 7, 7, 9, 9, 9, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 2, 5, 5, 5, 5, 5],
        ],
        dtype=np.int32,
    )
    monkeypatch.setattr(planner, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        planner,
        "available_args_for_request",
        lambda request, *, environments_dir: (
            {"x": 1, "y": 1},
            {"x": 2, "y": 1},
            {"x": 8, "y": 1},
            {"x": 4, "y": 4},
        ),
    )
    monkeypatch.setattr(
        planner,
        "grid_after_replay",
        lambda request, *, environments_dir: grid,
    )

    payload = planner.run_dynamic_retarget_patch_similarity_expansion_planning(
        selection_rule_consolidation_path=path,
        max_dynamic_args=3,
    )

    summary = payload["summary"]
    assert summary["selection_rule_consolidations_consumed"] == 1
    assert summary["live_available_args_seen"] == 4
    assert summary["excluded_args_count"] == 4
    assert summary["available_args_after_exclusion"] == 1
    assert summary["expansion_requests_generated"] == 3
    assert summary["resolved_request_arg_pairs"] == 3
    assert summary["unique_resolved_target_arg_sets"] == 1
    assert summary["duplicate_resolved_target_arg_sets"] == 2
    assert summary["duplicate_resolved_target_arg_sets_counted_as_independent"] is False
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    group = payload["candidate_arg_groups"][0]
    assert group["available_args_after_exclusion"] == [{"x": 2, "y": 1}]
    assert group["seed_successful_args"] == [{"x": 1, "y": 1}, {"x": 4, "y": 4}]
    assert group["seed_failed_args"] == [{"x": 8, "y": 1}]

    requests = payload["expansion_experiment_requests"]
    assert {request["probe_family"] for request in requests} == {
        "success_patch_similarity_expansion",
        "failure_patch_negative_control_expansion",
        "mixed_patch_boundary_probe",
    }
    assert all(
        request["target_action_arg_policy"] == "patch_similarity_excluding_known_args"
        for request in requests
    )
    assert all(request["resolved_target_action_args"] == [{"x": 2, "y": 1}] for request in requests)
    assert all(request["support"] == 0 for request in requests)
    assert all(request["truth_status"] == "NOT_EVALUATED_BY_M3" for request in requests)
    assert all(request["revision_performed"] is False for request in requests)
    assert all(request["wrong_confirmations"] == 0 for request in requests)
    assert all(request["execution_performed"] is False for request in requests)


def test_patch_similarity_expansion_planner_blocks_without_new_args(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "selection_rule_consolidation.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(planner, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        planner,
        "available_args_for_request",
        lambda request, *, environments_dir: (
            {"x": 1, "y": 1},
            {"x": 4, "y": 4},
        ),
    )
    monkeypatch.setattr(
        planner,
        "grid_after_replay",
        lambda request, *, environments_dir: np.zeros((3, 3), dtype=np.int32),
    )

    payload = planner.run_dynamic_retarget_patch_similarity_expansion_planning(
        selection_rule_consolidation_path=path,
    )

    assert payload["summary"]["expansion_requests_generated"] == 0
    assert payload["summary"]["blocked_expansion_requests"] == 3
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["expansion_experiment_requests"] == []


def test_patch_similarity_expansion_planner_handles_empty_input(tmp_path):
    path = tmp_path / "selection_rule_consolidation.json"
    path.write_text(json.dumps({}), encoding="utf-8")

    payload = planner.run_dynamic_retarget_patch_similarity_expansion_planning(
        selection_rule_consolidation_path=path,
    )

    assert payload["summary"]["selection_rule_consolidations_consumed"] == 0
    assert payload["summary"]["expansion_requests_generated"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
