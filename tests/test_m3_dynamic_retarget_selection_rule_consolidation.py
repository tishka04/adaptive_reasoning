import json

import theory.m3.dynamic_retarget_selection_rule_consolidation as consolidation


def _payload():
    return {
        "summary": {
            "selection_followup_requests_consumed": 8,
            "selection_followup_requests_executed": 2,
            "success_metric_support_events": 8,
            "diagnostic_contradiction_events": 4,
            "duplicate_resolved_target_arg_sets_counted_as_independent": False,
            "support": 0,
            "wrong_confirmations": 0,
        },
        "target_arg_resolutions": [
            {
                "request_id": "row-x30-y6",
                "rule_family": "row_or_band_dependent_retarget",
                "probe_family": "same_x_different_band_probe",
                "resolved_target_action_args": [],
                "blocked_reason": "explicit_args_not_available_after_replay",
            },
            {
                "request_id": "row-x18-y0",
                "rule_family": "row_or_band_dependent_retarget",
                "probe_family": "same_y_neighbor_x_probe",
                "resolved_target_action_args": [],
                "blocked_reason": "explicit_args_excluded_known_arg",
                "excluded_known_args_guard_triggered": True,
            },
            {
                "request_id": "patch-success",
                "rule_family": "local_patch_transformability",
                "probe_family": "local_patch_success_similarity_probe",
                "resolved_target_action_args": [{"x": 42, "y": 12}],
                "blocked_reason": None,
            },
            {
                "request_id": "patch-failure",
                "rule_family": "local_patch_transformability",
                "probe_family": "local_patch_failure_similarity_probe",
                "resolved_target_action_args": [{"x": 42, "y": 12}],
                "blocked_reason": None,
            },
        ],
        "per_request_results": [
            {
                "request_id": "patch-success",
                "source_selection_rule_candidate_id": (
                    "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule"
                ),
                "source_mechanism_candidate_id": (
                    "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region"
                ),
                "rule_family": "local_patch_transformability",
                "probe_family": "local_patch_success_similarity_probe",
                "target_action": "ACTION6",
                "resolved_target_action_args": [{"x": 42, "y": 12}],
                "request_has_success_metric_support": True,
                "grounded_success_metrics": [
                    "local_patch_before_after",
                    "object_positions_before_after",
                ],
                "controlled_experiments_run": 8,
                "success_metric_support_events": 4,
                "diagnostic_contradiction_events": 2,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            },
            {
                "request_id": "patch-failure",
                "source_selection_rule_candidate_id": (
                    "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule"
                ),
                "source_mechanism_candidate_id": (
                    "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region"
                ),
                "rule_family": "local_patch_transformability",
                "probe_family": "local_patch_failure_similarity_probe",
                "target_action": "ACTION6",
                "resolved_target_action_args": [{"x": 42, "y": 12}],
                "request_has_success_metric_support": True,
                "grounded_success_metrics": [
                    "local_patch_before_after",
                    "object_positions_before_after",
                ],
                "controlled_experiments_run": 8,
                "success_metric_support_events": 4,
                "diagnostic_contradiction_events": 2,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            },
            {
                "request_id": "row-x30-y6",
                "source_selection_rule_candidate_id": (
                    "m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule"
                ),
                "source_mechanism_candidate_id": (
                    "m3_13::bp35::A6_A3_A4::ACTION6::retarget_region"
                ),
                "rule_family": "row_or_band_dependent_retarget",
                "probe_family": "same_x_different_band_probe",
                "resolved_target_action_args": [],
                "request_has_success_metric_support": False,
                "grounded_success_metrics": [],
                "controlled_experiments_run": 0,
                "blocked_reason": "no_target_args_resolved",
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            },
        ],
        "rule_family_summary": [
            {
                "rule_family": "local_patch_transformability",
                "requests": 2,
                "requests_with_success_metric_support": 2,
                "blocked_requests": 0,
                "controlled_experiments_run": 16,
                "success_metric_support_events": 8,
                "success_metric_contradiction_events": 0,
                "diagnostic_support_events": 0,
                "diagnostic_contradiction_events": 4,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            },
            {
                "rule_family": "row_or_band_dependent_retarget",
                "requests": 6,
                "requests_with_success_metric_support": 0,
                "blocked_requests": 6,
                "controlled_experiments_run": 0,
                "success_metric_support_events": 0,
                "success_metric_contradiction_events": 0,
                "diagnostic_support_events": 0,
                "diagnostic_contradiction_events": 0,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            },
        ],
        "controlled_experiments": [
            {
                "request_id": "patch-success",
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
                "target_action_args": {"x": 42, "y": 12},
                "known_successful_retargets": [
                    {"x": 12, "y": 0},
                    {"x": 24, "y": 0},
                    {"x": 30, "y": 12},
                    {"x": 36, "y": 12},
                ],
                "known_failed_retargets": [{"x": 30, "y": 0}],
                "rule_family": "local_patch_transformability",
                "metric": "local_patch_before_after",
                "metric_role": "success_metric",
                "diagnostic_metrics": [
                    "changed_pixels",
                    "contact_graph_before_after",
                ],
                "support_events": 1,
                "contradiction_events": 0,
                "controlled_experiments_run": 1,
            },
            {
                "request_id": "patch-success",
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
                "target_action_args": {"x": 42, "y": 12},
                "known_successful_retargets": [
                    {"x": 12, "y": 0},
                    {"x": 24, "y": 0},
                    {"x": 30, "y": 12},
                    {"x": 36, "y": 12},
                ],
                "known_failed_retargets": [{"x": 30, "y": 0}],
                "rule_family": "local_patch_transformability",
                "metric": "changed_pixels",
                "metric_role": "diagnostic_metric",
                "diagnostic_metrics": [
                    "changed_pixels",
                    "contact_graph_before_after",
                ],
                "support_events": 0,
                "contradiction_events": 1,
                "controlled_experiments_run": 1,
            },
        ],
    }


def test_selection_rule_consolidation_preserves_caveats(tmp_path):
    path = tmp_path / "selection_followup_results.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = consolidation.run_dynamic_retarget_selection_rule_consolidation(
        selection_followup_results_path=path
    )

    summary = payload["summary"]
    assert summary["selection_rule_consolidations"] == 1
    assert summary["best_current_rule_families"] == ["local_patch_transformability"]
    assert summary["row_or_band_directly_tested"] is False
    assert summary["row_or_band_not_directly_contradicted"] is True
    assert summary["blocked_probe_counted_as_contradiction"] is False
    assert summary["duplicate_resolution_counted_as_independent"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    item = payload["selection_rule_consolidations"][0]
    assert (
        item["selection_rule_consolidation_id"]
        == "m3_17::bp35::ACTION4_ACTION6::selection_rule_consolidation"
    )
    assert item["best_current_rule_family"] == "local_patch_transformability"
    assert item["confidence_basis"] == "only_executed_followups_support_patch_similarity"
    assert "not executable" in item["important_caveat"]
    assert item["unique_new_successful_args"] == [{"x": 42, "y": 12}]
    assert item["known_successful_retargets"] == [
        {"x": 12, "y": 0},
        {"x": 24, "y": 0},
        {"x": 30, "y": 12},
        {"x": 36, "y": 12},
    ]
    assert item["known_failed_retargets"] == [{"x": 30, "y": 0}]
    assert item["excluded_args_for_next_expansion"] == [
        {"x": 12, "y": 0},
        {"x": 18, "y": 0},
        {"x": 24, "y": 0},
        {"x": 30, "y": 0},
        {"x": 30, "y": 12},
        {"x": 36, "y": 12},
        {"x": 42, "y": 12},
    ]
    assert item["success_metrics"] == [
        "local_patch_before_after",
        "object_positions_before_after",
    ]
    assert item["diagnostic_metrics"] == [
        "changed_pixels",
        "contact_graph_before_after",
    ]
    assert (
        item["diagnostic_metric_interpretation"]["changed_pixels_role"]
        == "effect_radar_not_retarget_success_metric"
    )
    assert item["row_or_band_directly_tested"] is False
    assert item["row_or_band_not_directly_contradicted"] is True
    assert item["support"] == 0
    assert item["rule_counted_as_confirmation"] is False

    assessments = {
        row["rule_family"]: row for row in item["rule_family_assessments"]
    }
    assert (
        assessments["local_patch_transformability"]["assessment"]
        == "supported_candidate_only_by_executed_followups"
    )
    assert (
        assessments["row_or_band_dependent_retarget"]["assessment"]
        == "not_directly_tested_blocked_or_excluded"
    )
    assert (
        assessments["row_or_band_dependent_retarget"]["directly_contradicted"]
        is False
    )
    assert assessments["row_or_band_dependent_retarget"]["blocked_reason_counts"] == {
        "explicit_args_excluded_known_arg": 1,
        "explicit_args_not_available_after_replay": 1,
    }
    assert (
        assessments["specific_effect_over_global_pixels"]["changed_pixels_role"]
        == "effect_radar_not_retarget_success_metric"
    )


def test_selection_rule_consolidation_handles_empty_input(tmp_path):
    path = tmp_path / "selection_followup_results.json"
    path.write_text(json.dumps({}), encoding="utf-8")

    payload = consolidation.run_dynamic_retarget_selection_rule_consolidation(
        selection_followup_results_path=path
    )

    assert payload["summary"]["selection_rule_consolidations"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["selection_rule_consolidations"] == []
