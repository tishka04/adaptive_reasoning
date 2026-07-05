import json
from collections import Counter

import pytest

from theory.m3 import generic_mechanic_experiment_planner as planner


def _m1_payload(*, support=0, ready=True, verdict=False):
    roles = [
        _role("E_actor", "controllable_actor", 0.91),
        _role("E_hud", "timer_or_hud", 1.0),
        _role("E_target", "target_candidate", 0.7),
    ]
    hypotheses = [
        _hypothesis("entity_role", "m1g0::entity_role::E_actor::controllable_actor", ["E_actor"]),
        _hypothesis("entity_role", "m1g0::entity_role::E_hud::timer_or_hud", ["E_hud"]),
        _hypothesis("entity_role", "m1g0::entity_role::E_target::target_candidate", ["E_target"]),
        _hypothesis("action_effect", "m1g0::action_effect::ACTION3::move_entity", ["E_actor"], actions=["ACTION3"]),
        _hypothesis("action_effect", "m1g0::action_effect::ACTION3::tick_latent", ["E_hud"], actions=["ACTION3"], latents=["terminal_horizon_candidate"]),
        _hypothesis("action_effect", "m1g0::action_effect::ACTION4::transform_entity", ["E_actor"], actions=["ACTION4"]),
        _hypothesis("action_effect", "m1g0::action_effect::ACTION6::change_relation", ["E_actor", "E_target"], actions=["ACTION6"]),
        _hypothesis("action_effect", "m1g0::action_effect::ACTION9::move_entity", ["E_actor"], actions=["ACTION9"]),
        _relation("ACTION3", "E_actor", "E_target", "touches", "contact_created"),
        _relation("ACTION3", "E_actor", "E_target", "adjacent_to", "contact_created"),
        _relation("ACTION3", "E_actor", "E_target", "distance", "distance_decreases"),
        _relation("ACTION4", "E_actor", "E_other", "distance", "distance_decreases"),
        _relation("ACTION6", "E_actor", "E_target", "near", "near_relation_created"),
        _hypothesis("dynamic_invariant", "m1g0::dynamic_invariant::E_hud::monotone_counter", ["E_hud"]),
        _hypothesis("dynamic_invariant", "m1g0::dynamic_invariant::E_actor::exogenous_motion", ["E_actor"]),
    ]
    return {
        "summary": {
            "game_id": "bp35-0a0ad940",
            "ready_for_m3_g0": ready,
            "mechanic_hypotheses_generated": len(hypotheses),
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M1",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "role_hypothesis_ledger": roles,
        "action_effect_abstractions": [
            {"action": "ACTION3", "support": 0},
            {"action": "ACTION4", "support": 0},
            {"action": "ACTION6", "support": 0},
        ],
        "dynamic_invariant_candidates": [
            {
                "invariant_id": "m1g0::invariant::E_hud::monotone_counter",
                "invariant_family": "monotone_counter",
                "entity_id": "E_hud",
                "affected_entities": ["E_hud"],
                "monotonicity_score": 1.0,
                "action_correlation_score": 1.0,
                "policy_relevance": "terminal_or_resource_horizon_candidate",
                "evidence": {"remaining_semantics_unknown": True},
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M1",
            },
            {
                "invariant_id": "m1g0::invariant::E_actor::exogenous_motion",
                "invariant_family": "exogenous_motion",
                "entity_id": "E_actor",
                "affected_entities": ["E_actor"],
                "monotonicity_score": 0.0,
                "action_correlation_score": 0.2,
                "policy_relevance": "forced_motion_candidate",
                "evidence": {},
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M1",
            },
        ],
        "mechanic_hypotheses": hypotheses,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M1",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "policy_result_counted_as_scientific_verdict": verdict,
    }


def _role(entity_id, role, score):
    return {
        "entity_id": entity_id,
        "role_hypotheses": [
            {
                "entity_id": entity_id,
                "role": role,
                "score": score,
                "evidence": ["fixture"],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M1",
                "role_counted_as_confirmation": False,
            }
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M1",
    }


def _hypothesis(family, hypothesis_id, entities, *, actions=None, latents=None):
    return {
        "mechanic_hypothesis_id": hypothesis_id,
        "mechanic_family": family,
        "entities": entities,
        "actions": list(actions or []),
        "relations": [],
        "latent_variables": list(latents or []),
        "preconditions": [],
        "predicted_effects": [f"{hypothesis_id} predicted effect"],
        "test_suggestions": ["fixture test"],
        "confidence": 1.0,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M1",
        "controlled_test_required": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "confidence_counted_as_support": False,
    }


def _relation(action, source, target, relation, delta):
    row = _hypothesis(
        "relation_change",
        f"m1g0::relation_change::{action}::{source}::{target}::{relation}::{delta}",
        [source, target],
        actions=[action],
    )
    row["relations"] = [relation]
    row["preconditions"] = ["controllable_actor", "target_candidate"]
    return row


def test_generic_mechanic_planner_builds_balanced_candidate_only_requests(tmp_path):
    path = tmp_path / "m1.json"
    path.write_text(json.dumps(_m1_payload()), encoding="utf-8")

    payload = planner.run_generic_mechanic_experiment_planning(
        general_mechanic_candidates_path=path,
        max_requests=16,
    )

    summary = payload["summary"]
    requests = payload["generic_mechanic_experiment_requests"]
    by_family = Counter(row["source_mechanic_family"] for row in requests)
    assert summary["generic_experiment_requests_generated"] == len(requests)
    assert len(requests) <= 16
    assert by_family["entity_role"] == 2
    assert by_family["action_effect"] == 4
    assert by_family["relation_change"] >= 1
    assert by_family["dynamic_invariant"] == 2
    assert summary["execution_performed"] is False
    assert summary["support"] == 0
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False

    assert {row["test_type"] for row in requests} >= {
        "actor_controllability_probe",
        "action_effect_causality_probe",
        "relation_change_probe",
        "dynamic_invariant_probe",
    }
    assert all(row["support"] == 0 for row in requests)
    assert all(row["revision_status"] == "CANDIDATE_ONLY" for row in requests)
    assert all(row["truth_status"] == "NOT_EVALUATED_BY_M3" for row in requests)
    assert all(row["execution_performed"] is False for row in requests)
    assert all(row["priority_score_counted_as_support"] is False for row in requests)
    assert all(row["generic_request_counted_as_confirmation"] is False for row in requests)


def test_generic_mechanic_planner_uses_only_observed_actions_and_relation_pair_cap(tmp_path):
    path = tmp_path / "m1.json"
    path.write_text(json.dumps(_m1_payload()), encoding="utf-8")

    payload = planner.run_generic_mechanic_experiment_planning(
        general_mechanic_candidates_path=path,
        max_requests=16,
        max_same_entity_pair=1,
    )

    requests = payload["generic_mechanic_experiment_requests"]
    observed = set(payload["summary"]["observed_actions"])
    action_values = {
        value
        for row in requests
        for value in [row.get("target_action"), *row.get("control_actions", []), *row.get("conditions", [])]
        if value is not None
    }
    assert "ACTION9" not in action_values
    assert action_values <= observed

    relation_pairs = Counter(
        (row["source_entity"], row["relation_target_entity"])
        for row in requests
        if row["source_mechanic_family"] == "relation_change"
    )
    assert relation_pairs[("E_actor", "E_target")] <= 1


def test_generic_mechanic_planner_preserves_unknown_hud_semantics(tmp_path):
    path = tmp_path / "m1.json"
    path.write_text(json.dumps(_m1_payload()), encoding="utf-8")

    payload = planner.run_generic_mechanic_experiment_planning(
        general_mechanic_candidates_path=path,
    )

    monotone = next(
        row
        for row in payload["generic_mechanic_experiment_requests"]
        if row["invariant_family"] == "monotone_counter"
    )
    assert monotone["remaining_semantics_unknown"] is True
    assert monotone["m1_support_counted_as_m3_support"] is False
    assert monotone["source_confidence_counted_as_support"] is False


def test_generic_mechanic_planner_rejects_source_support_or_not_ready(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_m1_payload(support=1)), encoding="utf-8")
    not_ready_path = tmp_path / "not_ready.json"
    not_ready_path.write_text(json.dumps(_m1_payload(ready=False)), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        planner.run_generic_mechanic_experiment_planning(
            general_mechanic_candidates_path=support_path,
        )
    with pytest.raises(ValueError, match="not ready"):
        planner.run_generic_mechanic_experiment_planning(
            general_mechanic_candidates_path=not_ready_path,
        )


def test_validate_generic_request_rejects_execution_support_or_unobserved_action():
    request = planner.GenericMechanicExperimentRequest(
        request_id="fixture",
        source_hypothesis_id="m1g0::fixture",
        source_mechanic_family="action_effect",
        test_type="action_effect_causality_probe",
        game_id="bp35",
        hypothesis_tested="fixture",
        priority_score=1.0,
        target_action="ACTION3",
        control_actions=("ACTION4",),
        metrics=("centroid_delta",),
    ).to_dict()
    request["execution_performed"] = True
    with pytest.raises(ValueError, match="cannot execute"):
        planner.validate_generic_mechanic_request(
            request,
            observed_actions=("ACTION3", "ACTION4"),
        )

    request["execution_performed"] = False
    request["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        planner.validate_generic_mechanic_request(
            request,
            observed_actions=("ACTION3", "ACTION4"),
        )

    request["support"] = 0
    request["target_action"] = "ACTION9"
    with pytest.raises(ValueError, match="was not observed"):
        planner.validate_generic_mechanic_request(
            request,
            observed_actions=("ACTION3", "ACTION4"),
        )
