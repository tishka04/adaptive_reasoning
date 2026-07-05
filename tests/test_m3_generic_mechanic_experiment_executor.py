import json

import pytest

from theory.m3 import generic_mechanic_experiment_executor as executor


def _request_payload(*, support=0, executed=False):
    requests = [
        _request(
            request_id="req_actor",
            family="entity_role",
            test_type="actor_controllability_probe",
            target_entity="E_actor",
            candidate_role="controllable_actor",
            conditions=["ACTION3", "ACTION4", "ACTION6"],
            executed=executed,
        ),
        _request(
            request_id="req_effect_move",
            family="action_effect",
            test_type="action_effect_causality_probe",
            target_action="ACTION3",
            control_actions=["ACTION4", "ACTION6"],
            predicted_effect_family="move_entity",
            affected_entities=["E_actor"],
        ),
        _request(
            request_id="req_relation",
            family="relation_change",
            test_type="relation_change_probe",
            target_action="ACTION3",
            control_actions=["ACTION4", "ACTION6"],
            source_entity="E_actor",
            relation_target_entity="E_target",
            relation_delta_type="distance_decreases",
        ),
        _request(
            request_id="req_invariant",
            family="dynamic_invariant",
            test_type="dynamic_invariant_probe",
            target_entity="E_hud",
            invariant_family="monotone_counter",
            conditions=["ACTION3", "ACTION4", "ACTION6"],
            remaining_semantics_unknown=True,
        ),
    ]
    return {
        "config": {
            "general_mechanic_candidates_path": "unused_fixture_m1.json",
            "observed_actions": ["ACTION3", "ACTION4", "ACTION6"],
            "execution_performed": False,
        },
        "generic_mechanic_experiment_requests": requests,
        "summary": {
            "generic_experiment_requests_generated": len(requests),
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "execution_performed": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "generic_request_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _request(
    *,
    request_id,
    family,
    test_type,
    target_action=None,
    control_actions=None,
    target_entity=None,
    source_entity=None,
    relation_target_entity=None,
    candidate_role=None,
    predicted_effect_family=None,
    relation_delta_type=None,
    invariant_family=None,
    conditions=None,
    affected_entities=None,
    remaining_semantics_unknown=None,
    executed=False,
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
        "source_entity": source_entity,
        "relation_target_entity": relation_target_entity,
        "candidate_role": candidate_role,
        "predicted_effect_family": predicted_effect_family,
        "relation_delta_type": relation_delta_type,
        "invariant_family": invariant_family,
        "affected_entities": list(affected_entities or []),
        "remaining_semantics_unknown": remaining_semantics_unknown,
        "metrics": ["fixture_metric"],
        "status": "READY_FOR_M3_GENERIC_EXPERIMENT",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "execution_performed": executed,
        "wrong_confirmations": 0,
        "priority_score_counted_as_support": False,
        "m1_support_counted_as_m3_support": False,
        "generic_request_counted_as_confirmation": False,
    }


def _m1_payload():
    return {
        "entity_tracks": [
            {
                "entity_id": "E_actor",
                "color": 7,
                "bbox_sequence": [[1, 1, 1, 1]],
                "centroid_sequence": [[1.0, 1.0]],
                "shape_signature_sequence": ["actor_shape"],
            },
            {
                "entity_id": "E_hud",
                "color": 9,
                "bbox_sequence": [[5, 0, 5, 5]],
                "centroid_sequence": [[5.0, 2.5]],
                "shape_signature_sequence": ["hud_shape"],
            },
            {
                "entity_id": "E_target",
                "color": 4,
                "bbox_sequence": [[1, 5, 1, 5]],
                "centroid_sequence": [[1.0, 5.0]],
                "shape_signature_sequence": ["target_shape"],
            },
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M1",
    }


def _fake_cell_executor(cell):
    actor_delta = {
        "changed": cell.action == "ACTION3",
        "effect_families": ["move_entity"] if cell.action == "ACTION3" else [],
        "centroid_delta": [0, 1] if cell.action == "ACTION3" else [0, 0],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }
    hud_delta = {
        "changed": True,
        "effect_families": ["transform_entity"],
        "size_delta": -1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }
    relation_delta = {
        "delta_types": ["distance_decreases"] if cell.action == "ACTION3" else [],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }
    return {
        **cell.to_dict(),
        "status": "EXECUTED",
        "entity_delta": {
            "E_actor": actor_delta,
            "E_hud": hud_delta,
            "E_target": {"changed": False, "effect_families": []},
        },
        "relation_delta": {"E_actor::E_target": relation_delta},
        "invariant_delta": {
            "E_hud": {
                "value_changed": True,
                "direction": "decreasing",
                "semantic_interpretation": "unknown",
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
        },
        "global_delta": {
            "changed_pixels": 5 if cell.action == "ACTION3" else 1,
            "terminal_state": False,
            "level_completed": False,
        },
        "controlled_experiments_run": 1,
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "execution_performed": True,
        "semantic_interpretation": "unknown",
    }


def test_generic_executor_deduplicates_action_cells_and_keeps_candidate_only(tmp_path):
    requests_path = tmp_path / "requests.json"
    m1_path = tmp_path / "m1.json"
    requests_path.write_text(json.dumps(_request_payload()), encoding="utf-8")
    m1_path.write_text(json.dumps(_m1_payload()), encoding="utf-8")

    payload = executor.run_generic_mechanic_experiment_execution(
        generic_requests_path=requests_path,
        m1_candidates_path=m1_path,
        cell_executor=_fake_cell_executor,
    )

    summary = payload["summary"]
    assert summary["generic_requests_consumed"] == 4
    assert summary["unique_execution_cells"] == 3
    assert summary["planned_execution_cells"] > summary["unique_execution_cells"]
    assert summary["duplicate_execution_cells_counted_as_independent"] is False
    assert summary["controlled_experiments_run"] == 3
    assert summary["support"] == 0
    assert summary["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False
    assert payload["support"] == 0
    assert payload["request_result_counted_as_confirmation"] is False
    assert payload["semantic_interpretation_counted_as_confirmation"] is False

    assert len(payload["hypothesis_observation_links"]) == summary["planned_execution_cells"]
    assert all(
        row["duplicate_execution_cell_counted_as_independent"] is False
        for row in payload["hypothesis_observation_links"]
    )
    assert any(row["support_events"] > 0 for row in payload["request_results"])
    assert all(row["support"] == 0 for row in payload["request_results"])


def test_generic_executor_rejects_source_support_or_preexecuted_request(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_request_payload(support=1)), encoding="utf-8")
    executed_path = tmp_path / "executed.json"
    executed_path.write_text(json.dumps(_request_payload(executed=True)), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        executor.run_generic_mechanic_experiment_execution(
            generic_requests_path=support_path,
            cell_executor=_fake_cell_executor,
        )
    with pytest.raises(ValueError, match="must not already be executed"):
        executor.run_generic_mechanic_experiment_execution(
            generic_requests_path=executed_path,
            cell_executor=_fake_cell_executor,
        )


def test_generic_executor_cell_building_merges_requested_entities_and_pairs():
    requests = _request_payload()["generic_mechanic_experiment_requests"]

    planned, links = executor.build_generic_execution_cells(requests)
    unique = executor.unique_generic_execution_cells(planned)

    assert links
    assert {cell.action for cell in unique} == {"ACTION3", "ACTION4", "ACTION6"}
    action3 = next(cell for cell in unique if cell.action == "ACTION3")
    assert "E_actor" in action3.requested_entities
    assert "E_hud" in action3.requested_entities
    assert ("E_actor", "E_target") in action3.requested_relation_pairs
