import json

import pytest

from theory.m3 import generic_mechanic_contextual_evidence_consolidation as consolidation


def _contextual_results_payload(
    *,
    support=0,
    semantic_confirmation=False,
    contextual_support=False,
):
    request_results = [
        _request_result(
            "req_actor",
            "entity_role",
            "actor_persistence_outside_reset",
            candidate_status="CONTEXT_REPRODUCED_CANDIDATE_ONLY",
            support_events=1,
            independent_contexts=1,
            interpretation="actor_candidate_trackable_and_action_affected_outside_reset",
        ),
        _request_result(
            "req_effect",
            "action_effect",
            "action_effect_stability",
            candidate_status="CONTEXT_REPRODUCED_CANDIDATE_ONLY",
            support_events=1,
            independent_contexts=1,
            interpretation="action_effect_reproduced_after_non_reset_prefix",
        ),
        _request_result(
            "req_counter",
            "dynamic_invariant",
            "dynamic_invariant_temporal",
            candidate_status="CONTEXT_FAILED_CANDIDATE_ONLY",
            contradiction_events=1,
            independent_contexts=3,
            interpretation="dynamic_invariant_not_observed_across_contextual_temporal_sequences",
            remaining_semantics_unknown=True,
        ),
        _request_result(
            "req_drift",
            "dynamic_invariant",
            "exogenous_motion_recurrence",
            candidate_status="TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY",
            support_events=1,
            independent_contexts=3,
            interpretation="dynamic_invariant_regular_across_contextual_temporal_sequences",
        ),
    ]
    return {
        "config": {
            "contextual_requests_path": "fixture_requests.json",
            "schema_version": "m3.generic_mechanic_contextual_followup_results.v1",
        },
        "summary": {
            "contextual_followup_requests_consumed": len(request_results),
            "planned_contextual_cells": 12,
            "unique_contextual_execution_cells": 6,
            "duplicate_contextual_cells": 6,
            "duplicate_contextual_cells_counted_as_independent": False,
            "independent_contexts": 6,
            "context_diversity_assessment": "MULTI_PREFIX_CONTEXTS",
            "controlled_experiments_run": 6,
            "support_events": 3,
            "contradiction_events": 1,
            "neutral_events": 0,
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
            {"cell_signature": "cell_a", "replay_policy": "contextual_prefix_replay"},
            {"cell_signature": "cell_b", "replay_policy": "contextual_prefix_replay"},
            {"cell_signature": "cell_c", "replay_policy": "contextual_temporal_sequence"},
        ],
        "request_results": request_results,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "contextual_signal_counted_as_scientific_support": contextual_support,
        "semantic_interpretation_counted_as_confirmation": semantic_confirmation,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _request_result(
    request_id,
    family,
    followup_family,
    *,
    candidate_status,
    support_events=0,
    contradiction_events=0,
    neutral_events=0,
    independent_contexts=1,
    interpretation="fixture",
    remaining_semantics_unknown=None,
):
    return {
        "request_id": request_id,
        "source_hypothesis_id": f"m1g0::{request_id}",
        "source_mechanic_family": family,
        "followup_family": followup_family,
        "candidate_status": candidate_status,
        "observation_interpretation": interpretation,
        "cells_linked": 2,
        "executed_cells": 2,
        "blocked_cells": 0,
        "independent_contexts": independent_contexts,
        "support_events": support_events,
        "contradiction_events": contradiction_events,
        "neutral_events": neutral_events,
        "remaining_semantics_unknown": remaining_semantics_unknown,
        "request_result_counted_as_confirmation": False,
        "contextual_signal_counted_as_scientific_support": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def test_contextual_evidence_consolidation_localizes_contradiction_and_keeps_support_zero(tmp_path):
    path = tmp_path / "contextual_results.json"
    path.write_text(json.dumps(_contextual_results_payload()), encoding="utf-8")

    payload = consolidation.run_generic_mechanic_contextual_evidence_consolidation(
        contextual_results_path=path,
    )

    summary = payload["summary"]
    assert summary["source_support_events"] == 3
    assert summary["source_contradiction_events"] == 1
    assert summary["independent_contexts"] == 6
    assert summary["context_diversity_assessment"] == "MULTI_PREFIX_CONTEXTS"
    assert summary["contextual_events_counted_as_scientific_support"] is False
    assert summary["contradiction_counted_as_refutation"] is False
    assert summary["ready_for_symbolic_model_candidate_only"] is True
    assert summary["support"] == 0

    statuses = {
        row["request_id"]: row["candidate_status"]
        for row in payload["hypothesis_consolidations"]
    }
    assert statuses["req_actor"] == "CONTEXT_STABLE_CANDIDATE_ONLY"
    assert statuses["req_effect"] == "CONTEXT_STABLE_CANDIDATE_ONLY"
    assert statuses["req_counter"] == "TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY"
    assert statuses["req_drift"] == "TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY"
    assert len(payload["contradiction_followups"]) == 1
    contradiction = payload["contradiction_followups"][0]
    assert contradiction["source_request_id"] == "req_counter"
    assert contradiction["contradiction_counted_as_refutation"] is False
    assert contradiction["followup_required_for_contradiction"] is True
    assert all(row["support"] == 0 for row in payload["hypothesis_consolidations"])


def test_contextual_family_readiness_is_candidate_only(tmp_path):
    path = tmp_path / "contextual_results.json"
    path.write_text(json.dumps(_contextual_results_payload()), encoding="utf-8")

    payload = consolidation.run_generic_mechanic_contextual_evidence_consolidation(
        contextual_results_path=path,
    )

    family_statuses = {
        row["family"]: row["candidate_status"]
        for row in payload["family_consolidations"]
    }
    assert family_statuses["entity_role"] == "READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY"
    assert family_statuses["action_effect"] == "READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY"
    assert family_statuses["dynamic_invariant"] == "READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY"
    readiness = payload["symbolic_model_readiness"]
    assert readiness["ready_for_symbolic_model_candidate_only"] is True
    assert readiness["symbolic_model_induction_performed"] is False
    assert readiness["support"] == 0


def test_contextual_evidence_consolidation_rejects_support_or_confirmation(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(_contextual_results_payload(support=1)),
        encoding="utf-8",
    )
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(
        json.dumps(_contextual_results_payload(semantic_confirmation=True)),
        encoding="utf-8",
    )
    contextual_support_path = tmp_path / "contextual_support.json"
    contextual_support_path.write_text(
        json.dumps(_contextual_results_payload(contextual_support=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        consolidation.run_generic_mechanic_contextual_evidence_consolidation(
            contextual_results_path=support_path,
        )
    with pytest.raises(ValueError, match="semantic interpretation"):
        consolidation.run_generic_mechanic_contextual_evidence_consolidation(
            contextual_results_path=semantic_path,
        )
    with pytest.raises(ValueError, match="contextual signals"):
        consolidation.run_generic_mechanic_contextual_evidence_consolidation(
            contextual_results_path=contextual_support_path,
        )


def test_contextual_candidate_status_mapping():
    assert (
        consolidation.contextual_candidate_status(
            raw_status="CONTEXT_REPRODUCED_CANDIDATE_ONLY",
            support_events=1,
            contradiction_events=0,
            neutral_events=0,
        )
        == "CONTEXT_STABLE_CANDIDATE_ONLY"
    )
    assert (
        consolidation.contextual_candidate_status(
            raw_status="CONTEXT_FAILED_CANDIDATE_ONLY",
            support_events=0,
            contradiction_events=1,
            neutral_events=0,
        )
        == "TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY"
    )
