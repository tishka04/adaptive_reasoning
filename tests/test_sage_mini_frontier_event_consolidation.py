import copy
import json

import pytest

from theory.sage import mini_frontier_event_consolidation as consolidation


def test_sage5f_builds_candidate_only_multi_budget_clusters(tmp_path):
    path = tmp_path / "sage5e.json"
    out = tmp_path / "sage5f.json"
    path.write_text(json.dumps(_source_payload()), encoding="utf-8")

    payload = consolidation.run_sage5f_mini_frontier_event_consolidation(
        source_sage5e_path=path,
        output_path=out,
    )

    assert out.exists()
    assert payload["truth_status"] == consolidation.SAGE5F_TRUTH_STATUS
    assert payload["support"] == 0
    assert payload["support_events_counted_as_support"] is False
    assert payload["support_events_counted_as_scientific_support"] is False
    assert payload["cluster_status_counted_as_scientific_verdict"] is False
    assert payload["candidate_a32_frontier_counted_as_revision"] is False
    assert payload["ready_for_A32_review_is_not_verdict"] is True
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert "verdict" not in payload

    summary = payload["summary"]
    assert summary["gate_passed"] is True
    assert summary["event_records"] == 4
    assert summary["multi_budget_clusters"] >= 1
    assert summary["robust_multi_budget_clusters"] >= 1
    assert summary["ready_for_A32_review_candidates"] >= 1
    assert summary["raw_support_events"] == 4
    assert summary["raw_contradiction_events"] == 0
    assert summary["clustered_support_events_counted_as_support"] is False

    robust = [
        row
        for row in payload["candidate_mechanism_clusters"]
        if row["candidate_status"]
        == consolidation.ROBUST_MULTI_BUDGET_CANDIDATE_ONLY
    ]
    assert robust
    first = robust[0]
    assert first["support"] == 0
    assert first["budgets"] == [50, 150, 300]
    assert first["raw_support_events"] == 3
    assert first["raw_contradiction_events"] == 0
    assert first["ready_for_A32_review"] is True
    assert first["ready_for_A32_review_is_not_verdict"] is True
    assert first["observed_effect_pattern"]["same_color_transition_pattern"] is True


def test_sage5f_marks_mixed_cluster_not_ready_for_a32(tmp_path):
    source = _source_payload()
    source["controlled_experiments"][1]["contradiction_events"] = 1
    source["controlled_experiments"][1]["support_events"] = 0
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    payload = consolidation.run_sage5f_mini_frontier_event_consolidation(
        source_sage5e_path=path,
    )

    mixed = [
        row
        for row in payload["candidate_mechanism_clusters"]
        if row["candidate_status"] == consolidation.MIXED_CANDIDATE_ONLY
    ]
    assert mixed
    assert mixed[0]["ready_for_A32_review"] is False
    assert payload["summary"]["ready_for_A32_review_candidates"] == 0
    assert payload["candidate_a32_review_frontiers"] == []
    assert payload["summary"]["support"] == 0


def test_sage5f_rejects_source_that_counts_support_or_evidence(tmp_path):
    support_source = _source_payload()
    support_source["support"] = 1
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support_source), encoding="utf-8")

    evidence_source = _source_payload()
    evidence_source["mini_frontier_execution_counted_as_evidence"] = True
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence_source), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        consolidation.run_sage5f_mini_frontier_event_consolidation(
            source_sage5e_path=support_path,
        )
    with pytest.raises(ValueError, match="execution cannot count as evidence"):
        consolidation.run_sage5f_mini_frontier_event_consolidation(
            source_sage5e_path=evidence_path,
        )


def _source_payload():
    requests = [
        _request("050", 1, "ACTION6", {"x": 1, "y": 1}, "local_patch_change_candidate"),
        _request("150", 7, "ACTION6", {"x": 1, "y": 1}, "local_patch_change_candidate"),
        _request("300", 9, "ACTION6", {"x": 1, "y": 1}, "local_patch_change_candidate"),
        _request("050", 2, "ACTION5", None, "object_delta_candidate"),
    ]
    experiments = [
        _experiment(requests[0], target_signal=5, control_signal=1),
        _experiment(requests[1], target_signal=5, control_signal=1),
        _experiment(requests[2], target_signal=5, control_signal=1),
        _experiment(requests[3], target_signal=21, control_signal=4),
    ]
    return {
        "outcome_status": "SAGE_DISTRIBUTED_LIVE_MINI_FRONTIER_GENERATED_AND_EXECUTED_CANDIDATE_ONLY",
        "summary": {
            "effective_requests_generated": len(requests),
            "requests_executed": len(experiments),
            "support_events": len(experiments),
            "contradiction_events": 0,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
        },
        "mini_frontier_m3_requests": requests,
        "controlled_experiments": experiments,
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_SAGE_5E",
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "support_events_counted_as_support": False,
        "mini_frontier_execution_counted_as_evidence": False,
        "policy_result_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _request(budget, step, action, args, family):
    if action == "ACTION6":
        diff = {
            "changed_cells": 20,
            "changed_bbox": {
                "x_min": 25,
                "y_min": 56,
                "x_max": 30,
                "y_max": 61,
            },
            "color_transitions": {"4->0": 20},
            "component_delta_by_color": {},
            "terminal_after": False,
            "levels_delta": 0,
        }
    else:
        diff = {
            "changed_cells": 21,
            "changed_bbox": {
                "x_min": 25,
                "y_min": 53,
                "x_max": 63,
                "y_max": 61,
            },
            "color_transitions": {"0->4": 20, "2->3": 1},
            "component_delta_by_color": {"3": 1},
            "terminal_after": False,
            "levels_delta": 0,
        }
    return {
        "request_id": f"m2m3::sage5e::distributed_live_mini_frontier::{budget}::{step:04d}",
        "source_hypothesis_id": f"sage5e::distributed_live_mini_frontier::{budget}::{step:04d}",
        "source_transition_id": f"sage5e::fake::budget_{budget}::step_{step:04d}",
        "source_step": step,
        "game_id": "fake-unknown",
        "hypothesis_family": family,
        "target_action": action,
        "target_action_args": copy.deepcopy(args),
        "context_snapshot_hash": f"context-{budget}-{step}",
        "dedup_key": f"dedup-{budget}-{step}",
        "diff_signature": diff,
        "metric": "local_patch_before_after",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M2",
    }


def _experiment(request, *, target_signal, control_signal):
    effect = float(target_signal) - float(control_signal)
    return {
        "execution_status": "EXECUTED",
        "request_id": request["request_id"],
        "source_hypothesis_id": request["source_hypothesis_id"],
        "source_transition_id": request["source_transition_id"],
        "game_id": request["game_id"],
        "hypothesis_family": request["hypothesis_family"],
        "target_action": request["target_action"],
        "target_action_args": copy.deepcopy(request["target_action_args"]),
        "control_action": "ACTION5"
        if request["target_action"] == "ACTION6"
        else "ACTION6",
        "metric": "local_patch_before_after",
        "context_snapshot_hash": request["context_snapshot_hash"],
        "target_signal": float(target_signal),
        "control_signal": float(control_signal),
        "controlled_delta": {"effect_size": effect},
        "support_events": 1 if effect > 0 else 0,
        "contradiction_events": 1 if effect < 0 else 0,
        "neutral_events": 1 if effect == 0 else 0,
        "live_prefix_replay_exact": True,
        "target_context_signature_verified": True,
        "control_context_signature_verified": True,
        "support": 0,
        "support_events_counted_as_support": False,
        "truth_status": "NOT_EVALUATED_BY_SAGE_5E",
        "revision_status": "CANDIDATE_ONLY",
    }
