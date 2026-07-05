import json

import pytest

from theory.m3 import generic_mechanic_evidence_consolidation as consolidation


def _results_payload(*, support=0, semantic_confirmation=False):
    request_results = [
        _request_result(
            "req_actor",
            "entity_role",
            support_events=1,
            neutral_events=0,
            interpretation="actor_candidate_action_differentiated",
        ),
        _request_result(
            "req_hud",
            "entity_role",
            support_events=0,
            neutral_events=1,
            interpretation="hud_timer_candidate_not_decisive",
        ),
        _request_result(
            "req_effect",
            "action_effect",
            support_events=1,
            neutral_events=0,
            interpretation="target_action_effect_exceeds_at_least_one_control",
        ),
        _request_result(
            "req_relation",
            "relation_change",
            support_events=0,
            contradiction_events=1,
            neutral_events=0,
            interpretation="target_action_lacks_relation_delta_seen_in_control",
        ),
    ]
    return {
        "summary": {
            "generic_requests_consumed": len(request_results),
            "support_events": 2,
            "contradiction_events": 1,
            "neutral_events": 1,
            "planned_execution_cells": 12,
            "unique_execution_cells": 3,
            "duplicate_execution_cells": 9,
            "duplicate_execution_cells_counted_as_independent": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "semantic_interpretation_counted_as_confirmation": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "execution_cells": [
            {"cell_signature": "cell_a3", "replay_policy": "initial_reset_same_state"},
            {"cell_signature": "cell_a4", "replay_policy": "initial_reset_same_state"},
            {"cell_signature": "cell_a6", "replay_policy": "initial_reset_same_state"},
        ],
        "request_results": request_results,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "semantic_interpretation_counted_as_confirmation": semantic_confirmation,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _request_result(
    request_id,
    family,
    *,
    support_events=0,
    contradiction_events=0,
    neutral_events=0,
    interpretation="fixture",
):
    return {
        "request_id": request_id,
        "source_hypothesis_id": f"m1g0::{request_id}",
        "source_mechanic_family": family,
        "test_type": f"{family}_probe",
        "observation_interpretation": interpretation,
        "cells_linked": 3,
        "blocked_cells": 0,
        "support_events": support_events,
        "contradiction_events": contradiction_events,
        "neutral_events": neutral_events,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
        "request_result_counted_as_confirmation": False,
        "source_confidence_counted_as_support": False,
    }


def test_generic_evidence_consolidation_marks_reset_only_support_as_duplicate_evidence(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_results_payload()), encoding="utf-8")

    payload = consolidation.run_generic_mechanic_evidence_consolidation(
        generic_results_path=path,
    )

    summary = payload["summary"]
    assert summary["source_support_events"] == 2
    assert summary["unique_execution_cells"] == 3
    assert summary["independent_contexts"] == 1
    assert summary["context_diversity_assessment"] == "LOW_RESET_ONLY"
    assert summary["support_events_counted_as_scientific_support"] is False
    assert summary["duplicate_links_counted_as_independent"] is False
    assert summary["support"] == 0
    assert summary["truth_status"] == "NOT_EVALUATED_BY_M3"

    statuses = {row["request_id"]: row["candidate_status"] for row in payload["hypothesis_consolidations"]}
    assert statuses["req_actor"] == "DUPLICATE_EVIDENCE_ONLY"
    assert statuses["req_hud"] == "NEUTRAL_CANDIDATE_ONLY"
    assert statuses["req_relation"] == "CONTRADICTED_CANDIDATE_ONLY"
    assert all(row["support"] == 0 for row in payload["hypothesis_consolidations"])
    assert all(
        row["support_events_counted_as_scientific_support"] is False
        for row in payload["hypothesis_consolidations"]
    )

    entity_family = next(
        row for row in payload["family_consolidations"] if row["family"] == "entity_role"
    )
    assert entity_family["candidate_status"] == "DUPLICATE_EVIDENCE_ONLY"
    assert entity_family["family_consolidation_counted_as_confirmation"] is False


def test_candidate_status_allows_supported_only_when_context_diversity_present():
    assert (
        consolidation.candidate_status(
            support_events=1,
            contradiction_events=0,
            neutral_events=0,
            independent_contexts=2,
            context_diversity_label="CONTEXT_DIVERSITY_PRESENT",
        )
        == "SUPPORTED_CANDIDATE_ONLY"
    )
    assert (
        consolidation.candidate_status(
            support_events=1,
            contradiction_events=0,
            neutral_events=0,
            independent_contexts=1,
            context_diversity_label="LOW_RESET_ONLY",
        )
        == "DUPLICATE_EVIDENCE_ONLY"
    )


def test_generic_evidence_consolidation_rejects_support_or_semantic_confirmation(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_results_payload(support=1)), encoding="utf-8")
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(
        json.dumps(_results_payload(semantic_confirmation=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        consolidation.run_generic_mechanic_evidence_consolidation(
            generic_results_path=support_path,
        )
    with pytest.raises(ValueError, match="semantic interpretation"):
        consolidation.run_generic_mechanic_evidence_consolidation(
            generic_results_path=semantic_path,
        )
