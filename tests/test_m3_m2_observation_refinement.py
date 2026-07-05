import json

from theory.m3.m2_observation_refinement import run_m2_observation_refinement


def _row(
    *,
    source_hypothesis_id,
    metric,
    signal_source="metric_extractor",
    support_events=0,
    neutral_events=0,
    diagnostic_only=False,
    raw_support_events=0,
    baseline_signal=0.0,
    perturbation_signal=0.0,
):
    return {
        "request_id": source_hypothesis_id.replace("m2::", "m2m3::"),
        "source_hypothesis_id": source_hypothesis_id,
        "hypothesis_key": source_hypothesis_id,
        "game_id": "bp35-0a0ad940",
        "context_replay": ["ACTION6", "ACTION3"],
        "context_replay_args": [{"x": 18, "y": 0}, {}],
        "target_action": "ACTION4",
        "control_action": "ACTION3",
        "metric": metric,
        "predicted_metric": metric,
        "signal_source": signal_source,
        "metric_grounding_status": (
            "diagnostic_only" if diagnostic_only else "grounded_metric_extractor"
        ),
        "diagnostic_only": diagnostic_only,
        "support_events": support_events,
        "contradiction_events": 0,
        "neutral_events": neutral_events,
        "raw_support_events": raw_support_events,
        "controlled_experiments_run": 1,
        "baseline_signal": baseline_signal,
        "perturbation_signal": perturbation_signal,
        "delta": {
            "effect_size": perturbation_signal - baseline_signal,
            "direction": "support" if support_events else "neutral",
            "signal_source": signal_source,
        },
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload(rows):
    return {
        "summary": {
            "support": 0,
            "wrong_confirmations": 0,
        },
        "controlled_experiments": rows,
    }


def test_refinement_merges_duplicate_m2_hypotheses_by_mechanistic_signature(
    tmp_path,
):
    rows = []
    for source_id in (
        "m2::after_ACTION3_live_after_ACTION6::h001",
        "m2::after_ACTION3_live_after_ACTION6::h002",
    ):
        rows.extend(
            [
                _row(
                    source_hypothesis_id=source_id,
                    metric="local_patch_before_after",
                    neutral_events=1,
                ),
                _row(
                    source_hypothesis_id=source_id,
                    metric="changed_pixels",
                    signal_source="raw_changed_pixels",
                    support_events=1,
                    baseline_signal=1.0,
                    perturbation_signal=47.0,
                ),
                _row(
                    source_hypothesis_id=source_id,
                    metric="object_positions_before_after",
                    support_events=1,
                    baseline_signal=2.0,
                    perturbation_signal=5.0,
                ),
                _row(
                    source_hypothesis_id=source_id,
                    metric="contact_graph_before_after",
                    neutral_events=1,
                ),
                _row(
                    source_hypothesis_id=source_id,
                    metric="topology_before_after",
                    signal_source="ungrounded_metric",
                    diagnostic_only=True,
                    neutral_events=1,
                    raw_support_events=1,
                    baseline_signal=1.0,
                    perturbation_signal=47.0,
                ),
            ]
        )
    path = tmp_path / "m2_candidate_experiment_results.json"
    path.write_text(json.dumps(_payload(rows)), encoding="utf-8")

    payload = run_m2_observation_refinement(experiment_results_path=path)

    summary = payload["summary"]
    assert summary["source_experiments"] == 10
    assert summary["source_hypotheses_consumed"] == 2
    assert summary["experimental_signature_groups"] == 5
    assert summary["unique_grounded_positive_signatures"] == 2
    assert summary["input_support_events"] == 4
    assert summary["input_support_events_after_signature_dedup"] == 2
    assert summary["refined_candidate_hypotheses"] == 1
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0
    assert summary["input_support_events_counted_as_support"] is False

    refined = payload["refined_candidate_hypotheses"][0]
    assert refined["refined_hypothesis_id"] == (
        "m3_8::bp35::A6_A3::ACTION4::global_motion"
    )
    assert refined["candidate_mechanic"] == (
        "global_object_repositioning_after_consumption"
    )
    assert refined["source_hypothesis_ids"] == [
        "m2::after_ACTION3_live_after_ACTION6::h001",
        "m2::after_ACTION3_live_after_ACTION6::h002",
    ]
    assert refined["positive_observations"] == [
        "changed_pixels_support",
        "object_positions_before_after_support",
    ]
    assert refined["neutral_observations"] == [
        "local_patch_before_after_neutral",
        "contact_graph_before_after_neutral",
    ]
    assert refined["diagnostic_only_observations"] == [
        "topology_before_after",
    ]
    assert refined["status"] == "UNRESOLVED"
    assert refined["revision_status"] == "CANDIDATE_ONLY"
    assert refined["support"] == 0
    assert refined["controlled_test_required"] is True
    assert refined["wrong_confirmations"] == 0
    assert refined["input_support_events_counted_as_support"] is False
    assert "derived_from_observations" in refined
    assert "confirmed_by" not in refined


def test_refinement_does_not_create_hypothesis_from_diagnostic_only_support(
    tmp_path,
):
    rows = [
        _row(
            source_hypothesis_id="m2::after_ACTION3_live_after_ACTION6::h001",
            metric="topology_before_after",
            signal_source="ungrounded_metric",
            diagnostic_only=True,
            neutral_events=1,
            raw_support_events=1,
            baseline_signal=1.0,
            perturbation_signal=47.0,
        )
    ]
    path = tmp_path / "m2_candidate_experiment_results.json"
    path.write_text(json.dumps(_payload(rows)), encoding="utf-8")

    payload = run_m2_observation_refinement(experiment_results_path=path)

    assert payload["summary"]["unique_grounded_positive_signatures"] == 0
    assert payload["summary"]["unique_diagnostic_only_signatures"] == 1
    assert payload["summary"]["refined_candidate_hypotheses"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
