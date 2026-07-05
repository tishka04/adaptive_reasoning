import json

import theory.m3.dynamic_retarget_patch_similarity_generativity_consolidation as consolidation


def _experiment(metric, metric_role, support, contradiction, control_action="ACTION3"):
    return {
        "request_id": "m3_18::success_patch_similarity_expansion::x48_y12",
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
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "target_action_args": {"x": 48, "y": 12},
        "rule_family": "local_patch_transformability",
        "metric": metric,
        "metric_role": metric_role,
        "success_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "diagnostic_metrics": [
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "seed_successful_args": [
            {"x": 12, "y": 0},
            {"x": 24, "y": 0},
            {"x": 30, "y": 12},
            {"x": 36, "y": 12},
            {"x": 42, "y": 12},
        ],
        "seed_failed_args": [{"x": 30, "y": 0}],
        "support_events": support,
        "contradiction_events": contradiction,
        "neutral_events": 0,
        "control_action": control_action,
        "controlled_experiments_run": 1,
        "diagnostic_only": metric_role == "diagnostic_metric",
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def _payload():
    return {
        "summary": {
            "expansion_requests_consumed": 3,
            "unique_execution_signatures": 1,
            "unique_target_arg_sets_executed": 1,
            "controlled_experiments_run": 8,
            "success_metric_support_events": 4,
            "success_metric_contradiction_events": 0,
            "diagnostic_support_events": 0,
            "diagnostic_contradiction_events": 2,
            "neutral_events": 2,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "wrong_confirmations": 0,
        },
        "per_signature_execution": [
            {
                "execution_signature": "sig-x48-y12",
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
                "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
                "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
                "target_action": "ACTION6",
                "target_action_args": {"x": 48, "y": 12},
                "rule_family": "local_patch_transformability",
                "grounded_success_metrics": [
                    "local_patch_before_after",
                    "object_positions_before_after",
                ],
                "signature_has_success_metric_support": True,
                "controlled_experiments_run": 8,
                "success_metric_support_events": 4,
                "success_metric_contradiction_events": 0,
                "diagnostic_contradiction_events": 2,
                "neutral_events": 2,
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            }
        ],
        "expansion_summary": {
            "tested_unique_arg_sets": 1,
            "args_with_grounded_support": 1,
            "best_arg": {"x": 48, "y": 12},
            "mechanism_support_events": 1,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "wrong_confirmations": 0,
        },
        "controlled_experiments": [
            _experiment("local_patch_before_after", "success_metric", 1, 0),
            _experiment("local_patch_before_after", "success_metric", 1, 0, "ACTION4"),
            _experiment("object_positions_before_after", "success_metric", 1, 0),
            _experiment(
                "object_positions_before_after",
                "success_metric",
                1,
                0,
                "ACTION4",
            ),
            _experiment("changed_pixels", "diagnostic_metric", 0, 1),
            _experiment("changed_pixels", "diagnostic_metric", 0, 1, "ACTION4"),
            _experiment("contact_graph_before_after", "diagnostic_metric", 0, 0),
            _experiment(
                "contact_graph_before_after",
                "diagnostic_metric",
                0,
                0,
                "ACTION4",
            ),
        ],
    }


def test_patch_similarity_generativity_consolidation(tmp_path):
    path = tmp_path / "patch_expansion_results.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = (
        consolidation.run_dynamic_retarget_patch_similarity_generativity_consolidation(
            expansion_results_path=path,
        )
    )

    summary = payload["summary"]
    assert summary["patch_similarity_generativity_consolidations"] == 1
    assert summary["candidate_rule_families"] == ["local_patch_transformability"]
    assert summary["candidate_generativity"] == [
        "SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"
    ]
    assert summary["successful_args_total_count"] == 6
    assert summary["failed_args_count"] == 1
    assert summary["new_expansion_successes_count"] == 2
    assert summary["ready_for_a32_revision_queue"] is True
    assert summary["ready_for_a32_revision_queue_is_not_verdict"] is True
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0
    assert summary["generative_sequence_counted_as_confirmation"] is False

    item = payload["generativity_consolidations"][0]
    assert (
        item["generativity_consolidation_id"]
        == "m3_20::bp35::ACTION4_ACTION6::patch_similarity_generativity"
    )
    assert item["candidate_rule_family"] == "local_patch_transformability"
    assert (
        item["candidate_generativity"]
        == "SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"
    )
    assert item["initial_success_args"] == [
        {"x": 12, "y": 0},
        {"x": 24, "y": 0},
        {"x": 30, "y": 12},
        {"x": 36, "y": 12},
    ]
    assert item["prior_expansion_successes"] == [{"x": 42, "y": 12}]
    assert item["new_executed_expansion_successes"] == [{"x": 48, "y": 12}]
    assert item["successful_args_total"] == [
        {"x": 12, "y": 0},
        {"x": 24, "y": 0},
        {"x": 30, "y": 12},
        {"x": 36, "y": 12},
        {"x": 42, "y": 12},
        {"x": 48, "y": 12},
    ]
    assert item["failed_args"] == [{"x": 30, "y": 0}]
    assert item["new_expansion_successes"] == [
        {"x": 42, "y": 12},
        {"x": 48, "y": 12},
    ]
    assert (
        item["pattern_hypothesis"]
        == "success_like_patch_line_or_region_after_repositioning"
    )
    assert item["success_metrics"] == [
        "local_patch_before_after",
        "object_positions_before_after",
    ]
    assert item["diagnostic_metrics"] == [
        "changed_pixels",
        "contact_graph_before_after",
    ]
    assert item["changed_pixels_role"] == "effect_radar_not_success_metric"
    assert item["source_success_metric_support_events"] == 4
    assert item["source_success_metric_contradiction_events"] == 0
    assert item["source_diagnostic_contradiction_events"] == 2
    assert item["ready_for_a32_revision_queue"] is True
    assert item["support"] == 0
    assert item["revision_status"] == "CANDIDATE_ONLY"
    assert item["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert item["wrong_confirmations"] == 0
    assert item["generative_sequence_counted_as_confirmation"] is False
    assert item["a32_queue_ready_is_not_verdict"] is True


def test_patch_similarity_generativity_handles_empty_input(tmp_path):
    path = tmp_path / "patch_expansion_results.json"
    path.write_text(json.dumps({}), encoding="utf-8")

    payload = (
        consolidation.run_dynamic_retarget_patch_similarity_generativity_consolidation(
            expansion_results_path=path,
        )
    )

    assert payload["summary"]["patch_similarity_generativity_consolidations"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["generativity_consolidations"] == []
