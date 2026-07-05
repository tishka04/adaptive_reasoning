import json

import pytest

from theory.m3 import generic_mechanic_contextual_followup_planner as planner


def _consolidation_payload(*, support=0, diversity="LOW_RESET_ONLY"):
    rows = [
        _consolidation_row(
            "req_actor",
            "entity_role",
            "DUPLICATE_EVIDENCE_ONLY",
            "actor_controllability_probe",
        ),
        _consolidation_row(
            "req_effect",
            "action_effect",
            "DUPLICATE_EVIDENCE_ONLY",
            "action_effect_causality_probe",
        ),
        _consolidation_row(
            "req_relation",
            "relation_change",
            "DUPLICATE_EVIDENCE_ONLY",
            "relation_change_probe",
        ),
        _consolidation_row(
            "req_counter",
            "dynamic_invariant",
            "NEUTRAL_CANDIDATE_ONLY",
            "dynamic_invariant_probe",
        ),
        _consolidation_row(
            "req_drift",
            "dynamic_invariant",
            "DUPLICATE_EVIDENCE_ONLY",
            "dynamic_invariant_probe",
        ),
    ]
    return {
        "config": {
            "generic_results_path": "",
            "execution_performed": False,
        },
        "summary": {
            "hypothesis_consolidations": len(rows),
            "candidate_status_counts": {
                "DUPLICATE_EVIDENCE_ONLY": 4,
                "NEUTRAL_CANDIDATE_ONLY": 1,
            },
            "context_diversity_assessment": diversity,
            "support_events_counted_as_scientific_support": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "hypothesis_consolidations": rows,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "support_events_counted_as_scientific_support": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _consolidation_row(request_id, family, candidate_status, test_type):
    return {
        "consolidation_id": f"m3g0_3::{request_id}",
        "request_id": request_id,
        "source_hypothesis_id": f"m1g0::{request_id}",
        "source_mechanic_family": family,
        "test_type": test_type,
        "candidate_status": candidate_status,
        "requires_contextual_followup": True,
        "support_events_counted_as_scientific_support": False,
        "duplicate_links_counted_as_independent": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def _requests_payload():
    requests = [
        _request(
            "req_actor",
            "entity_role",
            "actor_controllability_probe",
            target_entity="E_actor",
            candidate_role="controllable_actor",
            conditions=["ACTION3", "ACTION4", "ACTION6"],
            metrics=["entity_centroid_delta"],
        ),
        _request(
            "req_effect",
            "action_effect",
            "action_effect_causality_probe",
            target_action="ACTION3",
            control_actions=["ACTION4", "ACTION6"],
            predicted_effect_family="move_entity",
            metrics=["centroid_delta"],
        ),
        _request(
            "req_relation",
            "relation_change",
            "relation_change_probe",
            target_action="ACTION3",
            control_actions=["ACTION4", "ACTION6"],
            source_entity="E_actor",
            relation_target_entity="E_target",
            relation_delta_type="distance_decreases",
            metrics=["distance_before_after"],
        ),
        _request(
            "req_counter",
            "dynamic_invariant",
            "dynamic_invariant_probe",
            target_entity="E_hud",
            invariant_family="monotone_counter",
            invariant_id="m1g0::invariant::E_hud::monotone_counter",
            remaining_semantics_unknown=True,
            conditions=["ACTION3", "ACTION4", "ACTION6"],
            metrics=["monotonicity"],
        ),
        _request(
            "req_drift",
            "dynamic_invariant",
            "dynamic_invariant_probe",
            target_entity="E_drift",
            invariant_family="exogenous_motion",
            invariant_id="m1g0::invariant::E_drift::exogenous_motion",
            conditions=["ACTION3", "ACTION4", "ACTION6"],
            metrics=["value_delta_per_action"],
        ),
    ]
    return {
        "config": {
            "observed_actions": ["ACTION3", "ACTION4", "ACTION6"],
            "execution_performed": False,
        },
        "generic_mechanic_experiment_requests": requests,
        "summary": {
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def _request(
    request_id,
    family,
    test_type,
    *,
    target_action=None,
    control_actions=None,
    conditions=None,
    target_entity=None,
    candidate_role=None,
    source_entity=None,
    relation_target_entity=None,
    predicted_effect_family=None,
    relation_delta_type=None,
    invariant_family=None,
    invariant_id=None,
    remaining_semantics_unknown=None,
    metrics=None,
):
    return {
        "request_id": request_id,
        "source_hypothesis_id": f"m1g0::{request_id}",
        "source_mechanic_family": family,
        "test_type": test_type,
        "game_id": "bp35-0a0ad940",
        "target_action": target_action,
        "control_actions": list(control_actions or []),
        "conditions": list(conditions or []),
        "target_entity": target_entity,
        "candidate_role": candidate_role,
        "source_entity": source_entity,
        "relation_target_entity": relation_target_entity,
        "predicted_effect_family": predicted_effect_family,
        "relation_delta_type": relation_delta_type,
        "invariant_family": invariant_family,
        "invariant_id": invariant_id,
        "remaining_semantics_unknown": remaining_semantics_unknown,
        "metrics": list(metrics or []),
        "status": "READY_FOR_M3_GENERIC_EXPERIMENT",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "execution_performed": False,
        "wrong_confirmations": 0,
    }


def test_contextual_followup_planner_generates_multi_family_non_reset_requests(tmp_path):
    consolidation_path = tmp_path / "consolidation.json"
    requests_path = tmp_path / "requests.json"
    consolidation_path.write_text(json.dumps(_consolidation_payload()), encoding="utf-8")
    requests_path.write_text(json.dumps(_requests_payload()), encoding="utf-8")

    payload = planner.run_generic_mechanic_contextual_followup_planning(
        evidence_consolidation_path=consolidation_path,
        generic_requests_path=requests_path,
        max_followup_requests=16,
    )

    summary = payload["summary"]
    requests = payload["generic_contextual_followup_requests"]
    assert summary["source_context_diversity_assessment"] == "LOW_RESET_ONLY"
    assert summary["contextual_followup_requests_generated"] == len(requests)
    assert summary["target_context_diversity"] == "MULTI_PREFIX_CONTEXTS"
    assert summary["goal"] == "increase_context_diversity"
    assert summary["duplicate_evidence_promoted"] is False
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert {row["followup_family"] for row in requests} >= {
        "actor_persistence_outside_reset",
        "action_effect_stability",
        "relation_change_recurrence",
        "dynamic_invariant_temporal",
        "exogenous_motion_recurrence",
    }
    assert any(row["context_replay"] for row in requests if row["target_action"])
    assert all(row["execution_performed"] is False for row in requests)
    assert all(row["support"] == 0 for row in requests)
    assert all(row["truth_status"] == "NOT_EVALUATED_BY_M3" for row in requests)
    assert all(row["duplicate_evidence_promoted"] is False for row in requests)


def test_contextual_followup_preserves_monotone_counter_unknown_semantics(tmp_path):
    consolidation_path = tmp_path / "consolidation.json"
    requests_path = tmp_path / "requests.json"
    consolidation_path.write_text(json.dumps(_consolidation_payload()), encoding="utf-8")
    requests_path.write_text(json.dumps(_requests_payload()), encoding="utf-8")

    payload = planner.run_generic_mechanic_contextual_followup_planning(
        evidence_consolidation_path=consolidation_path,
        generic_requests_path=requests_path,
    )

    counter = next(
        row
        for row in payload["generic_contextual_followup_requests"]
        if row["invariant_family"] == "monotone_counter"
    )
    assert counter["remaining_semantics_unknown"] is True
    assert counter["temporal_action_sequences"]
    assert counter["semantic_interpretation_counted_as_confirmation"] is False
    assert counter["target_action"] is None


def test_contextual_followup_rejects_non_low_reset_or_support(tmp_path):
    requests_path = tmp_path / "requests.json"
    requests_path.write_text(json.dumps(_requests_payload()), encoding="utf-8")
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(_consolidation_payload(support=1)),
        encoding="utf-8",
    )
    diverse_path = tmp_path / "diverse.json"
    diverse_path.write_text(
        json.dumps(_consolidation_payload(diversity="MULTI_PREFIX_CONTEXTS")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        planner.run_generic_mechanic_contextual_followup_planning(
            evidence_consolidation_path=support_path,
            generic_requests_path=requests_path,
        )
    with pytest.raises(ValueError, match="LOW_RESET_ONLY"):
        planner.run_generic_mechanic_contextual_followup_planning(
            evidence_consolidation_path=diverse_path,
            generic_requests_path=requests_path,
        )


def test_contextual_followup_validation_rejects_execution_or_unobserved_action():
    request = planner.GenericContextualFollowupRequest(
        request_id="fixture",
        source_consolidation_id="m3g0_3::fixture",
        source_request_id="req",
        source_hypothesis_id="m1g0::fixture",
        source_candidate_status="DUPLICATE_EVIDENCE_ONLY",
        source_mechanic_family="action_effect",
        followup_family="action_effect_stability",
        game_id="bp35",
        context_replay=("ACTION3",),
        target_action="ACTION4",
        control_actions=("ACTION6",),
        temporal_action_sequences=(),
    ).to_dict()
    request["execution_performed"] = True
    with pytest.raises(ValueError, match="cannot execute"):
        planner.validate_contextual_followup_request(
            request,
            observed_actions=("ACTION3", "ACTION4", "ACTION6"),
        )
    request["execution_performed"] = False
    request["target_action"] = "ACTION9"
    with pytest.raises(ValueError, match="unobserved action"):
        planner.validate_contextual_followup_request(
            request,
            observed_actions=("ACTION3", "ACTION4", "ACTION6"),
        )
