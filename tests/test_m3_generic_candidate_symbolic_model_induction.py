import json

import pytest

from theory.m3 import generic_candidate_symbolic_model_induction as induction


def _consolidation_payload(*, support=0, ready=True, semantic_confirmation=False):
    records = [
        _record(
            "req_actor",
            "entity_role",
            "m1g0::entity_role::E_actor::controllable_actor",
            "CONTEXT_STABLE_CANDIDATE_ONLY",
            ready=True,
        ),
        _record(
            "req_action3_move",
            "action_effect",
            "m1g0::action_effect::ACTION3::move_entity",
            "CONTEXT_STABLE_CANDIDATE_ONLY",
            ready=True,
        ),
        _record(
            "req_relation",
            "relation_change",
            "m1g0::relation_change::ACTION3::E_actor::E_target",
            "CONTEXT_STABLE_CANDIDATE_ONLY",
            ready=True,
        ),
        _record(
            "req_counter",
            "dynamic_invariant",
            "m1g0::dynamic_invariant::E_hud::monotone_counter",
            "TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY",
            ready=False,
            contradictions=1,
        ),
        _record(
            "req_drift",
            "dynamic_invariant",
            "m1g0::dynamic_invariant::E_drift::exogenous_motion",
            "TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY",
            ready=True,
        ),
    ]
    return {
        "config": {
            "contextual_results_path": "",
            "execution_performed": False,
        },
        "summary": {
            "hypothesis_consolidations": len(records),
            "context_diversity_assessment": "MULTI_PREFIX_CONTEXTS",
            "independent_contexts": 6,
            "ready_for_symbolic_model_candidate_only": ready,
            "contextual_events_counted_as_scientific_support": False,
            "contradiction_counted_as_refutation": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "hypothesis_consolidations": records,
        "symbolic_model_readiness": {
            "ready_for_symbolic_model_candidate_only": ready,
            "symbolic_model_induction_performed": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
        },
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "contextual_events_counted_as_scientific_support": False,
        "semantic_interpretation_counted_as_confirmation": semantic_confirmation,
        "contradiction_counted_as_refutation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _record(
    request_id,
    family,
    hypothesis_id,
    status,
    *,
    ready,
    contradictions=0,
):
    return {
        "consolidation_id": f"m3g0_6::{request_id}",
        "request_id": request_id,
        "source_hypothesis_id": hypothesis_id,
        "source_mechanic_family": family,
        "followup_family": f"{family}_fixture",
        "candidate_status": status,
        "raw_support_events": 1 if ready else 0,
        "raw_contradiction_events": contradictions,
        "ready_for_symbolic_model_candidate_only": ready,
        "followup_required_for_contradiction": contradictions > 0,
        "contextual_events_counted_as_scientific_support": False,
        "contradiction_counted_as_refutation": False,
        "semantic_interpretation": "unknown",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def _requests_payload():
    return {
        "generic_contextual_followup_requests": [
            _request(
                "req_actor",
                "entity_role",
                target_entity="E_actor",
                candidate_role="controllable_actor",
            ),
            _request(
                "req_action3_move",
                "action_effect",
                target_action="ACTION3",
                predicted_effect_family="move_entity",
            ),
            _request(
                "req_relation",
                "relation_change",
                target_action="ACTION3",
                source_entity="E_actor",
                relation_target_entity="E_target",
                relation_delta_type="distance_decreases",
            ),
            _request(
                "req_counter",
                "dynamic_invariant",
                target_entity="E_hud",
                invariant_family="monotone_counter",
                invariant_id="m1g0::invariant::E_hud::monotone_counter",
                remaining_semantics_unknown=True,
            ),
            _request(
                "req_drift",
                "dynamic_invariant",
                target_entity="E_drift",
                invariant_family="exogenous_motion",
                invariant_id="m1g0::invariant::E_drift::exogenous_motion",
            ),
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def _request(
    request_id,
    family,
    *,
    target_entity=None,
    candidate_role=None,
    target_action=None,
    predicted_effect_family=None,
    source_entity=None,
    relation_target_entity=None,
    relation_delta_type=None,
    invariant_family=None,
    invariant_id=None,
    remaining_semantics_unknown=None,
):
    return {
        "request_id": request_id,
        "source_mechanic_family": family,
        "target_entity": target_entity,
        "candidate_role": candidate_role,
        "target_action": target_action,
        "predicted_effect_family": predicted_effect_family,
        "source_entity": source_entity,
        "relation_target_entity": relation_target_entity,
        "relation_delta_type": relation_delta_type,
        "invariant_family": invariant_family,
        "invariant_id": invariant_id,
        "remaining_semantics_unknown": remaining_semantics_unknown,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def test_candidate_symbolic_model_induction_builds_model_without_confirmation(tmp_path):
    consolidation_path = tmp_path / "consolidation.json"
    requests_path = tmp_path / "requests.json"
    consolidation_path.write_text(json.dumps(_consolidation_payload()), encoding="utf-8")
    requests_path.write_text(json.dumps(_requests_payload()), encoding="utf-8")

    payload = induction.run_generic_candidate_symbolic_model_induction(
        contextual_consolidation_path=consolidation_path,
        contextual_requests_path=requests_path,
    )

    summary = payload["summary"]
    model = payload["candidate_symbolic_model"]
    assert summary["actor_candidates"] == 1
    assert summary["action_models"] == 1
    assert summary["relation_effects"] == 1
    assert summary["dynamic_invariants"] == 1
    assert summary["caveats"] == 1
    assert summary["ready_for_policy_probe_candidate_only"] is True
    assert summary["model_counted_as_confirmation"] is False
    assert summary["support"] == 0
    assert model["model_status"] == "CANDIDATE_ONLY"
    assert model["symbolic_model_induction_counted_as_verdict"] is False
    assert model["a32_write_performed"] is False
    assert model["a33_write_performed"] is False
    assert model["actor_candidates"][0]["entity_id"] == "E_actor"
    assert model["action_models"]["ACTION3"]["candidate_effects"] == ["move_entity"]
    assert model["relation_model"]["actor_relation_effects"][0]["relation_delta_type"] == "distance_decreases"
    assert model["dynamic_invariants"]["E_drift"]["family"] == "exogenous_motion"
    assert model["dynamic_invariants"]["E_drift"]["semantic_interpretation"] == "unknown"
    assert model["caveats"][0]["entity_id"] == "E_hud"
    assert model["caveats"][0]["contradiction_counted_as_refutation"] is False


def test_candidate_symbolic_model_rejects_unready_or_support_or_semantic_confirmation(tmp_path):
    requests_path = tmp_path / "requests.json"
    requests_path.write_text(json.dumps(_requests_payload()), encoding="utf-8")
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_consolidation_payload(support=1)), encoding="utf-8")
    unready_path = tmp_path / "unready.json"
    unready_path.write_text(json.dumps(_consolidation_payload(ready=False)), encoding="utf-8")
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(
        json.dumps(_consolidation_payload(semantic_confirmation=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        induction.run_generic_candidate_symbolic_model_induction(
            contextual_consolidation_path=support_path,
            contextual_requests_path=requests_path,
        )
    with pytest.raises(ValueError, match="not ready"):
        induction.run_generic_candidate_symbolic_model_induction(
            contextual_consolidation_path=unready_path,
            contextual_requests_path=requests_path,
        )
    with pytest.raises(ValueError, match="semantic interpretation"):
        induction.run_generic_candidate_symbolic_model_induction(
            contextual_consolidation_path=semantic_path,
            contextual_requests_path=requests_path,
        )


def test_symbolic_model_keeps_duplicate_lists_deduped(tmp_path):
    payload = _consolidation_payload()
    payload["hypothesis_consolidations"].append(
        _record(
            "req_actor",
            "entity_role",
            "m1g0::entity_role::E_actor::controllable_actor",
            "CONTEXT_STABLE_CANDIDATE_ONLY",
            ready=True,
        )
    )
    consolidation_path = tmp_path / "consolidation.json"
    requests_path = tmp_path / "requests.json"
    consolidation_path.write_text(json.dumps(payload), encoding="utf-8")
    requests_path.write_text(json.dumps(_requests_payload()), encoding="utf-8")

    result = induction.run_generic_candidate_symbolic_model_induction(
        contextual_consolidation_path=consolidation_path,
        contextual_requests_path=requests_path,
    )

    actor = result["candidate_symbolic_model"]["actor_candidates"][0]
    assert actor["source_hypothesis_ids"] == [
        "m1g0::entity_role::E_actor::controllable_actor"
    ]
    assert actor["source_request_ids"] == ["req_actor"]
