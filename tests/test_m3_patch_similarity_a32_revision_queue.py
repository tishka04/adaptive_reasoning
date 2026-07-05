import copy
import json

import theory.m3.patch_similarity_a32_revision_queue as bridge


def _consolidation(**overrides):
    item = {
        "generativity_consolidation_id": (
            "m3_20::bp35::ACTION4_ACTION6::patch_similarity_generativity"
        ),
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
        "candidate_rule_family": "local_patch_transformability",
        "candidate_generativity": (
            "SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"
        ),
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "successful_args_total": [
            {"x": 12, "y": 0},
            {"x": 24, "y": 0},
            {"x": 30, "y": 12},
            {"x": 36, "y": 12},
            {"x": 42, "y": 12},
            {"x": 48, "y": 12},
        ],
        "failed_args": [{"x": 30, "y": 0}],
        "new_expansion_successes": [
            {"x": 42, "y": 12},
            {"x": 48, "y": 12},
        ],
        "success_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "diagnostic_metrics": [
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "changed_pixels_role": "effect_radar_not_success_metric",
        "pattern_hypothesis": "success_like_patch_line_or_region_after_repositioning",
        "source_success_metric_support_events": 4,
        "source_success_metric_contradiction_events": 0,
        "source_diagnostic_contradiction_events": 2,
        "source_neutral_events": 2,
        "ready_for_a32_revision_queue": True,
        "a32_queue_ready_is_not_verdict": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "generative_sequence_counted_as_confirmation": False,
    }
    item.update(overrides)
    return item


def _payload(item=None):
    return {
        "summary": {
            "patch_similarity_generativity_consolidations": 1,
            "ready_for_a32_revision_queue": True,
            "support": 0,
            "wrong_confirmations": 0,
        },
        "generativity_consolidations": [item or _consolidation()],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def test_patch_similarity_a32_revision_queue_accepts_m3_20_candidate(tmp_path):
    path = tmp_path / "generativity.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = bridge.run_patch_similarity_a32_revision_queue_generation(
        generativity_consolidation_path=path,
    )

    summary = payload["summary"]
    assert summary["generativity_consolidations_consumed"] == 1
    assert summary["queue_items"] == 1
    assert summary["rejected_queue_items"] == 0
    assert summary["ready_for_a32_revision_candidates"] == 1
    assert summary["ready_for_a32_revision_is_not_verdict"] is True
    assert summary["success_metric_support_events"] == 4
    assert summary["success_metric_contradiction_events"] == 0
    assert summary["diagnostic_contradiction_events"] == 2
    assert summary["diagnostic_contradictions_counted_as_refutation"] is False
    assert summary["changed_pixels_kept_diagnostic"] is True
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0

    item = payload["queue_items"][0]
    assert (
        item["queue_item_id"]
        == "m3_21::bp35::ACTION4_ACTION6::local_patch_transformability"
    )
    assert (
        item["source_generativity_consolidation_id"]
        == "m3_20::bp35::ACTION4_ACTION6::patch_similarity_generativity"
    )
    assert item["candidate_rule_family"] == "local_patch_transformability"
    assert (
        item["candidate_mechanic"]
        == "repositioning_opens_patch_similar_action6_affordances"
    )
    assert (
        item["candidate_generativity"]
        == "SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"
    )
    assert item["context_replay"] == ["ACTION6", "ACTION3", "ACTION4"]
    assert item["context_replay_args"] == [{"x": 18, "y": 0}, {}, {}]
    assert item["target_action"] == "ACTION6"
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
    assert item["success_metrics"] == [
        "local_patch_before_after",
        "object_positions_before_after",
    ]
    assert item["diagnostic_metrics"] == [
        "changed_pixels",
        "contact_graph_before_after",
    ]
    assert item["changed_pixels_role"] == "effect_radar_not_success_metric"
    assert item["evidence_summary"]["source_diagnostic_contradiction_events"] == 2
    assert (
        item["evidence_summary"]["diagnostic_contradictions_counted_as_refutation"]
        is False
    )
    assert item["ready_for_a32_revision"] is True
    assert item["ready_for_a32_revision_is_not_verdict"] is True
    assert item["support"] == 0
    assert item["revision_status"] == "CANDIDATE_ONLY"
    assert item["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert item["wrong_confirmations"] == 0
    assert item["generative_sequence_counted_as_confirmation"] is False
    assert item["diagnostic_contradictions_counted_as_refutation"] is False
    assert item["candidate_record"]["status"] == "unresolved"
    assert item["candidate_record"]["support"] == 0


def test_patch_similarity_a32_revision_queue_rejects_not_ready_candidate(tmp_path):
    item = _consolidation(ready_for_a32_revision_queue=False)
    path = tmp_path / "generativity.json"
    path.write_text(json.dumps(_payload(item)), encoding="utf-8")

    payload = bridge.run_patch_similarity_a32_revision_queue_generation(
        generativity_consolidation_path=path,
    )

    assert payload["summary"]["queue_items"] == 0
    assert payload["summary"]["rejected_queue_items"] == 1
    assert payload["rejected_queue_items"][0]["reason"] == (
        "not_ready_for_a32_revision_queue"
    )
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0


def test_patch_similarity_a32_revision_queue_rejects_success_metric_contradiction(
    tmp_path,
):
    item = _consolidation(source_success_metric_contradiction_events=1)
    path = tmp_path / "generativity.json"
    path.write_text(json.dumps(_payload(item)), encoding="utf-8")

    payload = bridge.run_patch_similarity_a32_revision_queue_generation(
        generativity_consolidation_path=path,
    )

    assert payload["summary"]["queue_items"] == 0
    assert payload["summary"]["rejected_queue_items"] == 1
    assert payload["rejected_queue_items"][0]["reason"] == (
        "success_metric_contradiction_events_present"
    )
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0


def test_patch_similarity_a32_revision_queue_diagnostic_contradiction_is_not_rejection(
    tmp_path,
):
    item = _consolidation(source_diagnostic_contradiction_events=12)
    path = tmp_path / "generativity.json"
    path.write_text(json.dumps(_payload(item)), encoding="utf-8")

    payload = bridge.run_patch_similarity_a32_revision_queue_generation(
        generativity_consolidation_path=path,
    )

    assert payload["summary"]["queue_items"] == 1
    assert payload["summary"]["diagnostic_contradiction_events"] == 12
    assert payload["summary"]["diagnostic_contradictions_counted_as_refutation"] is False
    assert payload["queue_items"][0]["changed_pixels_role"] == (
        "effect_radar_not_success_metric"
    )
    assert payload["queue_items"][0]["support"] == 0
    assert payload["queue_items"][0]["wrong_confirmations"] == 0


def test_patch_similarity_a32_revision_queue_does_not_mutate_input(tmp_path):
    source = _payload()
    before = copy.deepcopy(source)
    path = tmp_path / "generativity.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    bridge.run_patch_similarity_a32_revision_queue_generation(
        generativity_consolidation_path=path,
    )

    assert json.loads(path.read_text(encoding="utf-8")) == before
