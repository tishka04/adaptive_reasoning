import json

import theory.m3.dynamic_retarget_mechanism_consolidation as consolidation


def _per_arg(args, *, supported, rank):
    return {
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "target_action": "ACTION6",
        "target_action_args": dict(args),
        "target_action_arg_policy": "dynamic_retarget_after_repositioning",
        "candidate_arg_rank": rank,
        "candidate_arg_score": 0.0,
        "metrics_tested": [
            "local_patch_before_after",
            "changed_pixels",
            "object_positions_before_after",
            "contact_graph_before_after",
        ],
        "controls_tested": ["ACTION3", "ACTION4"],
        "support_events": 4 if supported else 0,
        "contradiction_events": 2 if supported else 4,
        "neutral_events": 2 if supported else 4,
        "grounded_support_metrics": (
            ["local_patch_before_after", "object_positions_before_after"]
            if supported
            else []
        ),
        "arg_has_grounded_support": supported,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _experiment(args, metric, *, supported):
    return {
        "request_id": f"request::{args['x']}::{args['y']}::{metric}",
        "source_refined_hypothesis_id": "m3_8::bp35::A6_A3::ACTION4::global_motion",
        "source_hypothesis_ids": [
            "m2::after_ACTION3_live_after_ACTION6::h001",
            "m2::after_ACTION3_live_after_ACTION6::h002",
        ],
        "game_id": "bp35-0a0ad940",
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "target_action_args": dict(args),
        "target_action_arg_policy": "dynamic_retarget_after_repositioning",
        "control_action": "ACTION3",
        "metric": metric,
        "support_events": (
            1
            if supported
            and metric in {"local_patch_before_after", "object_positions_before_after"}
            else 0
        ),
        "contradiction_events": 1
        if metric == "changed_pixels" or not supported
        else 0,
        "neutral_events": 1
        if supported and metric == "contact_graph_before_after"
        else 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload():
    supported_args = [{"x": 12, "y": 0}, {"x": 24, "y": 0}]
    failed_args = [{"x": 30, "y": 0}]
    experiments = []
    for args in supported_args + failed_args:
        for metric in [
            "local_patch_before_after",
            "changed_pixels",
            "object_positions_before_after",
            "contact_graph_before_after",
        ]:
            experiments.append(_experiment(args, metric, supported=args in supported_args))
    return {
        "summary": {
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "wrong_confirmations": 0,
        },
        "controlled_experiments": experiments,
        "per_arg_results": [
            _per_arg({"x": 12, "y": 0}, supported=True, rank=1),
            _per_arg({"x": 24, "y": 0}, supported=True, rank=2),
            _per_arg({"x": 30, "y": 0}, supported=False, rank=3),
        ],
        "mechanism_summary": {
            "tested_candidate_args": 3,
            "args_with_grounded_support": 2,
            "best_arg": {"x": 12, "y": 0},
            "mechanism_support_events": 1,
            "arg_level_support_events": 8,
            "arg_level_support_events_counted_as_mechanism_support": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "wrong_confirmations": 0,
        },
    }


def test_dynamic_retarget_mechanism_consolidation_builds_candidate(tmp_path):
    path = tmp_path / "dynamic_retarget_followup_results.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = consolidation.run_dynamic_retarget_mechanism_consolidation(
        retarget_results_path=path
    )

    summary = payload["summary"]
    assert summary["candidate_mechanisms"] == 1
    assert summary["successful_retargets"] == 2
    assert summary["failed_retargets"] == 1
    assert summary["mechanism_support_events"] == 1
    assert summary["mechanism_support_events_counted_as_support"] is False
    assert summary["arg_level_support_events_counted_as_mechanism_support"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    candidate = payload["mechanism_candidates"][0]
    assert (
        candidate["candidate_mechanic"]
        == "repositioning_opens_new_action6_target_region"
    )
    assert candidate["context_replay"] == ["ACTION6", "ACTION3", "ACTION4"]
    assert candidate["initial_consumed_args"] == {"x": 18, "y": 0}
    assert candidate["successful_retargets"] == [
        {"x": 12, "y": 0},
        {"x": 24, "y": 0},
    ]
    assert candidate["failed_retargets"] == [{"x": 30, "y": 0}]
    assert candidate["positive_metrics"] == [
        "local_patch_before_after",
        "object_positions_before_after",
    ]
    assert candidate["non_decisive_or_negative_metrics"] == [
        "changed_pixels",
        "contact_graph_before_after",
    ]
    assert (
        candidate["metric_interpretation"]["changed_pixels_role"]
        == "effect_radar_not_retarget_success_metric"
    )
    assert candidate["selection_problem_open"] is True
    assert candidate["support"] == 0
    assert candidate["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert candidate["revision_performed"] is False
    assert candidate["wrong_confirmations"] == 0


def test_dynamic_retarget_mechanism_consolidation_requires_success(tmp_path):
    payload = _payload()
    for row in payload["per_arg_results"]:
        row["arg_has_grounded_support"] = False
        row["support_events"] = 0
        row["grounded_support_metrics"] = []
    path = tmp_path / "dynamic_retarget_followup_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = consolidation.run_dynamic_retarget_mechanism_consolidation(
        retarget_results_path=path
    )

    assert result["summary"]["candidate_mechanisms"] == 0
    assert result["summary"]["support"] == 0
    assert result["summary"]["wrong_confirmations"] == 0
    assert result["mechanism_candidates"] == []
