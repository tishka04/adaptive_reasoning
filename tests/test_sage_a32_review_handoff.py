import copy
import json

import pytest

from theory.sage import a32_review_handoff as handoff


def test_sage5g_compiles_candidate_only_handoffs(tmp_path):
    source_path = tmp_path / "sage5f.json"
    output_path = tmp_path / "sage5g.json"
    source_path.write_text(json.dumps(_source_payload()), encoding="utf-8")

    payload = handoff.run_sage5g_a32_review_handoff(
        source_sage5f_path=source_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert payload["truth_status"] == handoff.SAGE5G_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["execution_performed"] is False
    assert payload["revision_performed"] is False
    assert payload["candidate_review_item_counted_as_revision"] is False
    assert payload["independent_context_events_counted_as_scientific_support"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    summary = payload["summary"]
    assert summary["gate_passed"] is True
    assert summary["handoff_items"] == 2
    assert summary["items_ready_for_A32_review"] == 2
    assert summary["items_without_followup_requirements"] == 0
    assert summary["followup_requests"] == 4
    assert summary["raw_support_events_in_handoff"] == 5
    assert summary["raw_contradiction_events_in_handoff"] == 0
    assert summary["related_nonmerged_cluster_links"] == 1
    assert summary["outcome_status"] == handoff.SAGE5G_HANDOFF_COMPILED


def test_sage5g_action6_requires_control_diversity(tmp_path):
    payload = _run_fixture(tmp_path)
    item = payload["a32_review_candidate_items"][0]

    assert item["action"] == "ACTION6"
    assert item["action_args"] == {"x": 26, "y": 57}
    assert item["predicted_metric"] == "local_patch_before_after"
    assert item["predicted_effect_signature"]["changed_cells"] == 20
    assert item["predicted_effect_signature"]["color_transitions"] == {"4->0": 20}
    assert item["budgets"] == [50, 150, 300]
    assert item["raw_support_events"] == 3
    assert item["independent_context_events"] == 3
    assert item["distinct_control_actions"] == 1
    assert item["missing_revision_requirements"] == [
        "minimum_distinct_control_actions"
    ]
    assert (
        item["a32_intake_recommendation"]
        == handoff.FOLLOWUP_REQUIRED_CONTROL_DIVERSITY
    )
    assert len(item["requested_followups"]) == 1
    request = item["requested_followups"][0]
    assert request["request_type"] == "ACQUIRE_DISTINCT_CONTROL_ACTION"
    assert request["excluded_control_actions"] == ["ACTION5"]
    assert request["minimum_replay_exact_contexts"] == 2


def test_sage5g_action5_links_nonmerged_cluster_and_requests_three_followups(
    tmp_path,
):
    payload = _run_fixture(tmp_path)
    item = payload["a32_review_candidate_items"][1]

    assert item["action"] == "ACTION5"
    assert item["action_args"] is None
    assert item["source_cluster_ids"] == [
        "sage5f::candidate_mechanism_cluster::003"
    ]
    assert item["related_nonmerged_cluster_ids"] == [
        "sage5f::candidate_mechanism_cluster::002"
    ]
    assert item["raw_support_events"] == 2
    assert item["independent_context_events"] == 2
    assert item["distinct_control_actions"] == 1
    assert item["missing_revision_requirements"] == [
        "minimum_support_events",
        "minimum_distinct_control_actions",
    ]
    assert (
        item["a32_intake_recommendation"]
        == handoff.FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY
    )
    assert [row["request_type"] for row in item["requested_followups"]] == [
        "ACQUIRE_ADDITIONAL_COMPARABLE_SUPPORT",
        "ACQUIRE_DISTINCT_CONTROL_ACTION",
        "CROSS_MEASURE_RELATED_NONMERGED_CLUSTER",
    ]
    cross_measurement = item["requested_followups"][2]
    assert cross_measurement["required_measurements"] == [
        "local_patch",
        "object_delta",
    ]
    assert cross_measurement["clusters_must_remain_unmerged"] is True


def test_sage5g_real_sage5f_artifact_matches_expected_result():
    payload = handoff.run_sage5g_a32_review_handoff()

    assert payload["summary"]["handoff_items"] == 2
    assert payload["summary"]["followup_requests"] == 4
    assert payload["summary"]["raw_support_events_in_handoff"] == 5
    assert payload["summary"]["related_nonmerged_cluster_links"] == 1
    assert [
        item["a32_intake_recommendation"]
        for item in payload["a32_review_candidate_items"]
    ] == [
        handoff.FOLLOWUP_REQUIRED_CONTROL_DIVERSITY,
        handoff.FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY,
    ]


def test_sage5g_rejects_source_that_counts_support(tmp_path):
    source = _source_payload()
    source["support"] = 1
    path = tmp_path / "invalid_support.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        handoff.run_sage5g_a32_review_handoff(source_sage5f_path=path)


def test_sage5g_rejects_source_that_writes_a32_or_a33(tmp_path):
    source = _source_payload()
    source["a32_write_performed"] = True
    path = tmp_path / "invalid_write.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot write A32/A33"):
        handoff.run_sage5g_a32_review_handoff(source_sage5f_path=path)


def test_sage5g_rejects_frontier_referencing_missing_cluster(tmp_path):
    source = _source_payload()
    source["candidate_a32_review_frontiers"][0]["source_cluster_id"] = "missing"
    path = tmp_path / "missing_cluster.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="references unknown cluster"):
        handoff.run_sage5g_a32_review_handoff(source_sage5f_path=path)


def test_sage5g_no_ready_frontiers_emits_no_handoff_items(tmp_path):
    source = _source_payload()
    source["candidate_a32_review_frontiers"] = []
    source["summary"]["ready_for_A32_review_candidates"] = 0
    path = tmp_path / "no_frontier.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    payload = handoff.run_sage5g_a32_review_handoff(source_sage5f_path=path)

    assert payload["a32_review_candidate_items"] == []
    assert payload["requested_followups"] == []
    assert payload["summary"]["gate_passed"] is False
    assert payload["outcome_status"] == handoff.SAGE5G_NO_HANDOFF_ITEMS
    assert payload["support"] == 0


def test_sage5g_does_not_mutate_source(tmp_path):
    source = _source_payload()
    before = copy.deepcopy(source)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    handoff.run_sage5g_a32_review_handoff(source_sage5f_path=path)

    assert json.loads(path.read_text(encoding="utf-8")) == before


def test_sage5g_recommendation_classification_is_explicit():
    assert (
        handoff.a32_intake_recommendation(
            ["minimum_support_events", "minimum_distinct_control_actions"]
        )
        == handoff.FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY
    )
    assert (
        handoff.a32_intake_recommendation(["minimum_distinct_control_actions"])
        == handoff.FOLLOWUP_REQUIRED_CONTROL_DIVERSITY
    )
    assert (
        handoff.a32_intake_recommendation([])
        == handoff.READY_FOR_A32_INTAKE_CANDIDATE_ONLY
    )


def _run_fixture(tmp_path):
    path = tmp_path / "sage5f.json"
    path.write_text(json.dumps(_source_payload()), encoding="utf-8")
    return handoff.run_sage5g_a32_review_handoff(source_sage5f_path=path)


def _source_payload():
    events = [
        _event(1, 50, 6, "ACTION6", {"x": 26, "y": 57}, "ACTION5", "c1"),
        _event(2, 150, 12, "ACTION6", {"x": 26, "y": 57}, "ACTION5", "c2"),
        _event(3, 300, 20, "ACTION6", {"x": 26, "y": 57}, "ACTION5", "c3"),
        _event(
            4,
            50,
            1,
            "ACTION5",
            None,
            "ACTION6",
            "c4",
            family="object_delta_candidate",
            component_delta={"3": 1},
        ),
        _event(5, 150, 11, "ACTION5", None, "ACTION6", "c5"),
        _event(6, 300, 19, "ACTION5", None, "ACTION6", "c6"),
    ]
    clusters = [
        _cluster(
            1,
            events[:3],
            action="ACTION6",
            action_args={"x": 26, "y": 57},
            family="local_patch_change_candidate",
            changed_cells=20,
            color_transitions={"4->0": 20},
            component_delta={},
            effect_size=4.0,
            ready=True,
        ),
        _cluster(
            2,
            events[3:4],
            action="ACTION5",
            action_args=None,
            family="object_delta_candidate",
            changed_cells=21,
            color_transitions={"0->4": 20, "2->3": 1},
            component_delta={"3": 1},
            effect_size=17.0,
            ready=False,
        ),
        _cluster(
            3,
            events[4:],
            action="ACTION5",
            action_args=None,
            family="local_patch_change_candidate",
            changed_cells=21,
            color_transitions={"0->4": 20, "2->3": 1},
            component_delta={},
            effect_size=17.0,
            ready=True,
        ),
    ]
    return {
        "outcome_status": "SAGE_MINI_FRONTIER_EVENTS_CLUSTERED_CANDIDATE_ONLY",
        "summary": {
            "candidate_mechanism_clusters": 3,
            "ready_for_A32_review_candidates": 2,
            "support": 0,
        },
        "event_records": events,
        "candidate_mechanism_clusters": clusters,
        "candidate_a32_review_frontiers": [
            _frontier(1, clusters[0]["cluster_id"]),
            _frontier(2, clusters[2]["cluster_id"]),
        ],
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_SAGE_5F",
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "candidate_a32_frontier_counted_as_revision": False,
        "ready_for_A32_review_is_not_verdict": True,
        "cluster_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _event(
    index,
    budget,
    step,
    action,
    args,
    control,
    context,
    *,
    family="local_patch_change_candidate",
    component_delta=None,
):
    action6 = action == "ACTION6"
    transitions = {"4->0": 20} if action6 else {"0->4": 20, "2->3": 1}
    changed_cells = 20 if action6 else 21
    target_signal = 5.0 if action6 else 21.0
    control_signal = 1.0 if action6 else 4.0
    return {
        "event_id": f"sage5f::mini_frontier_event::{index:03d}",
        "request_id": f"request-{index}",
        "source_hypothesis_id": f"hypothesis-{index}",
        "source_transition_id": f"sage5e::fake::budget_{budget:03d}::step_{step:04d}",
        "game_id": "sb26-7fbdac44",
        "budget": budget,
        "source_step": step,
        "hypothesis_family": family,
        "target_action": action,
        "target_action_args": copy.deepcopy(args),
        "control_action": control,
        "metric": "local_patch_before_after",
        "context_snapshot_hash": context,
        "dedup_key": f"dedup-{index}",
        "changed_cells": changed_cells,
        "changed_bbox": {
            "x_min": 25,
            "y_min": 56 if action6 else 53,
            "x_max": 30 if action6 else 54 + index,
            "y_max": 61,
        },
        "bbox_shape": "6x6" if action6 else f"{30 + index}x9",
        "color_transitions": transitions,
        "component_delta_by_color": dict(component_delta or {}),
        "terminal_after": False,
        "levels_delta": 0,
        "target_signal": target_signal,
        "control_signal": control_signal,
        "effect_size": target_signal - control_signal,
        "support_events": 1,
        "contradiction_events": 0,
        "neutral_events": 0,
        "live_prefix_replay_exact": True,
        "target_context_signature_verified": True,
        "control_context_signature_verified": True,
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_SAGE_5F",
        "revision_status": "CANDIDATE_ONLY",
    }


def _cluster(
    index,
    events,
    *,
    action,
    action_args,
    family,
    changed_cells,
    color_transitions,
    component_delta,
    effect_size,
    ready,
):
    cluster_id = f"sage5f::candidate_mechanism_cluster::{index:03d}"
    budgets = sorted({row["budget"] for row in events})
    contexts = sorted({row["context_snapshot_hash"] for row in events})
    return {
        "cluster_id": cluster_id,
        "cluster_key": f"cluster-{index}",
        "game_id": "sb26-7fbdac44",
        "hypothesis_family": family,
        "actions": [action],
        "target_action_args_signature": json.dumps(
            action_args,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "budgets": budgets,
        "contexts": contexts,
        "context_count": len(contexts),
        "request_ids": [row["request_id"] for row in events],
        "source_hypothesis_ids": [row["source_hypothesis_id"] for row in events],
        "source_transition_ids": [row["source_transition_id"] for row in events],
        "observed_effect_pattern": {
            "metrics": ["local_patch_before_after"],
            "color_transition_signatures": {
                json.dumps(color_transitions, sort_keys=True, separators=(",", ":")): len(events)
            },
            "component_delta_signatures": {
                json.dumps(component_delta, sort_keys=True, separators=(",", ":")): len(events)
            },
            "changed_cells_values": [changed_cells],
            "bbox_shape_counts": {row["bbox_shape"]: 1 for row in events},
            "effect_size_values": [effect_size],
            "mean_effect_size": effect_size,
            "terminal_after_values": [False],
            "levels_delta_values": [0],
            "same_bbox_shape": len({row["bbox_shape"] for row in events}) == 1,
        },
        "raw_support_events": len(events),
        "raw_contradiction_events": 0,
        "raw_neutral_events": 0,
        "executed_events": len(events),
        "live_prefix_replay_exact_events": len(events),
        "all_replays_exact": True,
        "unique_dedup_keys": len(events),
        "candidate_status": (
            "ROBUST_MULTI_BUDGET_CANDIDATE_ONLY"
            if ready
            else "LOCAL_SUPPORT_CANDIDATE_ONLY"
        ),
        "ready_for_A32_review": ready,
        "ready_for_A32_review_is_not_verdict": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_SAGE_5F",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "support_events_counted_as_support": False,
        "cluster_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _frontier(index, cluster_id):
    return {
        "frontier_id": f"sage5f::candidate_a32_frontier::{index:03d}",
        "source_cluster_id": cluster_id,
        "ready_for_A32_review": True,
        "ready_for_A32_review_is_not_verdict": True,
        "candidate_a32_frontier_counted_as_revision": False,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
