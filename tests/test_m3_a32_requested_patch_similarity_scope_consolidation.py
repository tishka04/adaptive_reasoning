import json

import theory.m3.a32_requested_patch_similarity_scope_consolidation as consolidation


def _experiment(
    *,
    family,
    target_args,
    support,
    contradiction,
    metric="local_patch_before_after",
    metric_role="success_metric",
):
    return {
        "source_a32_queue_item_id": (
            "m3_21::bp35::ACTION4_ACTION6::local_patch_transformability"
        ),
        "source_a32_decision": "SCOPE_LIMITED_CANDIDATE_ONLY",
        "source_a32_recommended_next_step": "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS",
        "game_id": "bp35-0a0ad940",
        "candidate_rule_family": "local_patch_transformability",
        "context_replay": (
            ["ACTION6", "ACTION4"]
            if family == "alternate_repositioning_context_probe"
            else ["ACTION6", "ACTION3", "ACTION4"]
        ),
        "context_replay_args": (
            [{"x": 18, "y": 0}, {}]
            if family == "alternate_repositioning_context_probe"
            else [{"x": 18, "y": 0}, {}, {}]
        ),
        "target_action": "ACTION6",
        "target_action_args": dict(target_args),
        "followup_families": [family],
        "metric": metric,
        "metric_role": metric_role,
        "success_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "diagnostic_metrics": ["changed_pixels", "contact_graph_before_after"],
        "seed_successful_args": [
            {"x": 12, "y": 0},
            {"x": 24, "y": 0},
            {"x": 30, "y": 12},
            {"x": 36, "y": 12},
            {"x": 42, "y": 12},
            {"x": 48, "y": 12},
        ],
        "seed_failed_args": [{"x": 30, "y": 0}],
        "support_events": support,
        "contradiction_events": contradiction,
        "neutral_events": 0,
        "controlled_experiments_run": 1,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def _signature(family, target_args, success, contradiction):
    return {
        "source_a32_queue_item_id": (
            "m3_21::bp35::ACTION4_ACTION6::local_patch_transformability"
        ),
        "source_a32_decision": "SCOPE_LIMITED_CANDIDATE_ONLY",
        "source_a32_recommended_next_step": "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS",
        "game_id": "bp35-0a0ad940",
        "candidate_rule_family": "local_patch_transformability",
        "followup_families": [family],
        "context_replay": (
            ["ACTION6", "ACTION4"]
            if family == "alternate_repositioning_context_probe"
            else ["ACTION6", "ACTION3", "ACTION4"]
        ),
        "context_replay_args": (
            [{"x": 18, "y": 0}, {}]
            if family == "alternate_repositioning_context_probe"
            else [{"x": 18, "y": 0}, {}, {}]
        ),
        "target_action": "ACTION6",
        "target_action_args": dict(target_args),
        "scope_interpretation": (
            "alternate_context_scope_expanded_candidate_only"
            if family == "alternate_repositioning_context_probe"
            else "outside_region_boundary_reinforced_candidate_only"
        ),
        "success_metric_support_events": success,
        "success_metric_contradiction_events": contradiction,
        "diagnostic_contradiction_events": 2,
        "signature_has_success_metric_support": success > 0,
        "outside_boundary_failure_counted_as_rule_refutation": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def _payload():
    return {
        "summary": {
            "a32_followup_requests_consumed": 2,
            "unique_execution_signatures": 4,
            "controlled_experiments_run": 32,
            "success_metric_support_events": 8,
            "success_metric_contradiction_events": 4,
            "diagnostic_contradiction_events": 8,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "wrong_confirmations": 0,
        },
        "family_summary": [
            {
                "followup_family": "alternate_repositioning_context_probe",
                "family_interpretation": "alternate_context_scope_expanded_candidate_only",
                "args_with_success_metric_support": 2,
                "args_with_success_metric_contradiction": 0,
                "success_metric_support_events": 8,
                "success_metric_contradiction_events": 0,
                "diagnostic_contradiction_events": 4,
                "outside_boundary_failures_counted_as_rule_refutation": False,
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            },
            {
                "followup_family": "outside_known_y12_region_probe",
                "family_interpretation": (
                    "outside_known_region_boundary_reinforced_candidate_only"
                ),
                "args_with_success_metric_support": 0,
                "args_with_success_metric_contradiction": 2,
                "success_metric_support_events": 0,
                "success_metric_contradiction_events": 4,
                "diagnostic_contradiction_events": 4,
                "outside_boundary_failures_counted_as_rule_refutation": False,
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            },
        ],
        "per_signature_execution": [
            _signature("outside_known_y12_region_probe", {"x": 18, "y": 0}, 0, 2),
            _signature("outside_known_y12_region_probe", {"x": 30, "y": 0}, 0, 2),
            _signature(
                "alternate_repositioning_context_probe",
                {"x": 12, "y": 0},
                4,
                0,
            ),
            _signature(
                "alternate_repositioning_context_probe",
                {"x": 24, "y": 0},
                4,
                0,
            ),
        ],
        "controlled_experiments": [
            _experiment(
                family="outside_known_y12_region_probe",
                target_args={"x": 18, "y": 0},
                support=0,
                contradiction=1,
            ),
            _experiment(
                family="alternate_repositioning_context_probe",
                target_args={"x": 12, "y": 0},
                support=1,
                contradiction=0,
            ),
            _experiment(
                family="alternate_repositioning_context_probe",
                target_args={"x": 24, "y": 0},
                support=1,
                contradiction=0,
                metric="changed_pixels",
                metric_role="diagnostic_metric",
            ),
        ],
    }


def test_a32_requested_patch_similarity_scope_consolidation(tmp_path):
    path = tmp_path / "a32_followup_results.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = consolidation.run_a32_requested_patch_similarity_scope_consolidation(
        a32_followup_results_path=path,
    )

    summary = payload["summary"]
    assert summary["scope_consolidations"] == 1
    assert summary["scope_assessments"] == ["SCOPE_EXPANDED_CANDIDATE_ONLY"]
    assert summary["scope_expanded_candidate_only"] == 1
    assert summary["ready_for_agent_policy_probe"] is True
    assert summary["a33_ready"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0
    assert summary["scope_expansion_counted_as_confirmation"] is False

    item = payload["scope_consolidations"][0]
    assert (
        item["scope_consolidation_id"]
        == "m3_24::bp35-0a0ad940::patch_similarity_scope::ACTION6"
    )
    assert item["scope_assessment"] == "SCOPE_EXPANDED_CANDIDATE_ONLY"
    assert item["initial_context_supported"] is True
    assert item["alternate_context_supported"] is True
    assert item["outside_region_boundary_reinforced"] is True
    assert item["alternate_context_success_args"] == [
        {"x": 12, "y": 0},
        {"x": 24, "y": 0},
    ]
    assert item["outside_boundary_args"] == [
        {"x": 18, "y": 0},
        {"x": 30, "y": 0},
    ]
    assert item["known_failed_args"] == [{"x": 30, "y": 0}]
    assert item["success_metrics"] == [
        "local_patch_before_after",
        "object_positions_before_after",
    ]
    assert item["changed_pixels_role"] == "effect_radar_not_success_metric"
    assert item["a33_ready"] is False
    assert item["ready_for_agent_policy_probe"] is True
    assert item["agent_policy_probe_status"] == "EXPERIMENTAL_POLICY_CANDIDATE_ONLY"
    assert item["support"] == 0
    assert item["revision_status"] == "CANDIDATE_ONLY"
    assert item["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert item["wrong_confirmations"] == 0
    assert item["scope_expansion_counted_as_confirmation"] is False


def test_a32_requested_patch_similarity_scope_consolidation_handles_empty_input(
    tmp_path,
):
    path = tmp_path / "a32_followup_results.json"
    path.write_text(json.dumps({}), encoding="utf-8")

    payload = consolidation.run_a32_requested_patch_similarity_scope_consolidation(
        a32_followup_results_path=path,
    )

    assert payload["summary"]["scope_consolidations"] == 0
    assert payload["summary"]["ready_for_agent_policy_probe"] is False
    assert payload["summary"]["a33_ready"] is False
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["scope_consolidations"] == []
