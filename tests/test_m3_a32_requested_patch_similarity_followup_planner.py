import json

import numpy as np

import theory.m3.a32_requested_patch_similarity_followup_planner as planner


def _decision():
    return {
        "queue_item_id": "m3_21::bp35::ACTION4_ACTION6::local_patch_transformability",
        "game_id": "bp35-0a0ad940",
        "decision": "SCOPE_LIMITED_CANDIDATE_ONLY",
        "recommended_next_step": "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS",
        "reasons": ["mono_game_scope", "mono_context_scope", "scope_not_a33_ready"],
        "scope_limits": {
            "candidate_rule_family": "local_patch_transformability",
            "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
            "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
            "target_action": "ACTION6",
            "successful_args_total": [
                {"x": 1, "y": 1},
                {"x": 4, "y": 4},
            ],
            "failed_args": [{"x": 8, "y": 1}],
        },
        "requested_followup_tests": [
            {
                "followup_family": "outside_known_y12_region_probe",
                "purpose": "test outside known region",
                "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
                "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
                "target_action": "ACTION6",
                "exclude_known_args": [{"x": 1, "y": 1}, {"x": 4, "y": 4}],
                "success_metrics": [
                    "local_patch_before_after",
                    "object_positions_before_after",
                ],
                "diagnostic_metrics": [
                    "changed_pixels",
                    "contact_graph_before_after",
                ],
            },
            {
                "followup_family": "alternate_repositioning_context_probe",
                "purpose": "test alternate context",
                "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
                "target_action": "ACTION6",
                "success_metrics": [
                    "local_patch_before_after",
                    "object_positions_before_after",
                ],
                "diagnostic_metrics": [
                    "changed_pixels",
                    "contact_graph_before_after",
                ],
            },
        ],
    }


def _payload():
    return {
        "summary": {
            "scope_limited_candidate_only": 1,
            "recommended_more_tests": 1,
            "support": 0,
            "wrong_confirmations": 0,
        },
        "revision_decisions": [_decision()],
    }


def _grid():
    return np.array(
        [
            [1, 1, 1, 7, 7, 7, 9, 9, 9, 4],
            [1, 2, 1, 7, 2, 7, 9, 3, 9, 4],
            [1, 1, 1, 7, 7, 7, 9, 9, 9, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 2, 5, 5, 5, 5, 5],
        ],
        dtype=np.int32,
    )


def test_a32_requested_patch_similarity_followup_planner(monkeypatch, tmp_path):
    path = tmp_path / "patch_similarity_revision_decisions.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    monkeypatch.setattr(planner, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        planner,
        "available_args_for_request",
        lambda request, *, environments_dir: (
            {"x": 1, "y": 1},
            {"x": 2, "y": 1},
            {"x": 4, "y": 4},
            {"x": 8, "y": 1},
        ),
    )
    monkeypatch.setattr(
        planner,
        "grid_after_replay",
        lambda request, *, environments_dir: _grid(),
    )

    payload = planner.run_a32_requested_patch_similarity_followup_planning(
        a32_decisions_path=path,
        max_dynamic_args=2,
    )

    summary = payload["summary"]
    assert summary["a32_revision_decisions_consumed"] == 1
    assert summary["a32_requested_followup_tests_seen"] == 2
    assert summary["planned_followup_requests"] == 2
    assert summary["blocked_followup_requests"] == 0
    assert summary["outside_known_y12_region_requests"] == 1
    assert summary["alternate_repositioning_context_requests"] == 1
    assert summary["resolved_request_arg_pairs"] == 4
    assert summary["a32_decision_counted_as_confirmation"] is False
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    requests = payload["a32_requested_followup_requests"]
    assert {request["followup_family"] for request in requests} == {
        "outside_known_y12_region_probe",
        "alternate_repositioning_context_probe",
    }
    assert all(
        request["source_a32_decision"] == "SCOPE_LIMITED_CANDIDATE_ONLY"
        for request in requests
    )
    assert all(
        request["source_a32_recommended_next_step"]
        == "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS"
        for request in requests
    )
    assert all(
        request["a32_decision_counted_as_confirmation"] is False
        for request in requests
    )
    assert all(request["support"] == 0 for request in requests)
    assert all(request["truth_status"] == "NOT_EVALUATED_BY_M3" for request in requests)
    assert all(request["revision_performed"] is False for request in requests)
    assert all(request["wrong_confirmations"] == 0 for request in requests)
    assert all(request["execution_performed"] is False for request in requests)

    outside = [
        request
        for request in requests
        if request["followup_family"] == "outside_known_y12_region_probe"
    ][0]
    assert outside["context_replay"] == ["ACTION6", "ACTION3", "ACTION4"]
    assert outside["excluded_args"] == [{"x": 1, "y": 1}, {"x": 4, "y": 4}]
    assert outside["resolved_target_action_args"] == [
        {"x": 2, "y": 1},
        {"x": 8, "y": 1},
    ]
    assert all(
        row["outside_known_y12_region"]
        for row in outside["candidate_resolution_args"]
        if row["action_args"] in outside["resolved_target_action_args"]
    )

    alternate = [
        request
        for request in requests
        if request["followup_family"] == "alternate_repositioning_context_probe"
    ][0]
    assert alternate["context_replay"] == ["ACTION6", "ACTION4"]
    assert alternate["resolved_target_action_args"] == [
        {"x": 1, "y": 1},
        {"x": 4, "y": 4},
    ]


def test_a32_requested_patch_similarity_followup_blocks_without_live_args(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "patch_similarity_revision_decisions.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    monkeypatch.setattr(planner, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(
        planner,
        "available_args_for_request",
        lambda request, *, environments_dir: (),
    )
    monkeypatch.setattr(
        planner,
        "grid_after_replay",
        lambda request, *, environments_dir: _grid(),
    )

    payload = planner.run_a32_requested_patch_similarity_followup_planning(
        a32_decisions_path=path,
    )

    assert payload["summary"]["planned_followup_requests"] == 0
    assert payload["summary"]["blocked_followup_requests"] == 2
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert all(
        row["a32_decision_counted_as_confirmation"] is False
        for row in payload["blocked_followup_requests"]
    )


def test_a32_requested_patch_similarity_followup_handles_empty_input(tmp_path):
    path = tmp_path / "patch_similarity_revision_decisions.json"
    path.write_text(json.dumps({}), encoding="utf-8")

    payload = planner.run_a32_requested_patch_similarity_followup_planning(
        a32_decisions_path=path,
    )

    assert payload["summary"]["a32_revision_decisions_consumed"] == 0
    assert payload["summary"]["planned_followup_requests"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
