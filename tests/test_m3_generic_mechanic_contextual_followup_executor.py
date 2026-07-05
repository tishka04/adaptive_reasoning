import json

import pytest

from theory.m3 import generic_mechanic_contextual_followup_executor as executor


def _contextual_payload(*, support=0, executed=False):
    requests = [
        _request(
            request_id="req_actor",
            family="entity_role",
            followup_family="actor_persistence_outside_reset",
            context_replay=["ACTION3"],
            target_action="ACTION4",
            control_actions=["ACTION6"],
            target_entity="E_actor",
            candidate_role="controllable_actor",
        ),
        _request(
            request_id="req_effect",
            family="action_effect",
            followup_family="action_effect_stability",
            context_replay=["ACTION4"],
            target_action="ACTION3",
            control_actions=["ACTION6"],
            target_entity="E_actor",
            predicted_effect_family="move_entity",
        ),
        _request(
            request_id="req_relation",
            family="relation_change",
            followup_family="relation_change_recurrence",
            context_replay=["ACTION4"],
            target_action="ACTION3",
            control_actions=["ACTION6"],
            source_entity="E_actor",
            relation_target_entity="E_target",
            relation_delta_type="distance_decreases",
        ),
        _request(
            request_id="req_counter",
            family="dynamic_invariant",
            followup_family="dynamic_invariant_temporal",
            target_entity="E_hud",
            invariant_family="monotone_counter",
            temporal_action_sequences=[["ACTION3", "ACTION4", "ACTION6"]],
            remaining_semantics_unknown=True,
        ),
        _request(
            request_id="req_drift",
            family="dynamic_invariant",
            followup_family="exogenous_motion_recurrence",
            target_entity="E_drift",
            invariant_family="exogenous_motion",
            temporal_action_sequences=[["ACTION3", "ACTION4", "ACTION6"]],
        ),
    ]
    return {
        "config": {
            "execution_performed": False,
            "target_context_diversity": "MULTI_PREFIX_CONTEXTS",
            "observed_actions": ["ACTION3", "ACTION4", "ACTION6"],
        },
        "summary": {
            "contextual_followup_requests_generated": len(requests),
            "duplicate_evidence_promoted": False,
            "execution_performed": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "generic_contextual_followup_requests": requests,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "execution_performed": executed,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "duplicate_evidence_promoted": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _request(
    *,
    request_id,
    family,
    followup_family,
    context_replay=None,
    target_action=None,
    control_actions=None,
    temporal_action_sequences=None,
    target_entity=None,
    candidate_role=None,
    source_entity=None,
    relation_target_entity=None,
    predicted_effect_family=None,
    relation_delta_type=None,
    invariant_family=None,
    remaining_semantics_unknown=None,
):
    return {
        "request_id": request_id,
        "source_hypothesis_id": f"m1g0::{request_id}",
        "source_mechanic_family": family,
        "source_candidate_status": "DUPLICATE_EVIDENCE_ONLY",
        "followup_family": followup_family,
        "game_id": "bp35-0a0ad940",
        "context_replay": list(context_replay or []),
        "target_action": target_action,
        "control_actions": list(control_actions or []),
        "temporal_action_sequences": [list(row) for row in temporal_action_sequences or []],
        "target_entity": target_entity,
        "candidate_role": candidate_role,
        "source_entity": source_entity,
        "relation_target_entity": relation_target_entity,
        "predicted_effect_family": predicted_effect_family,
        "relation_delta_type": relation_delta_type,
        "invariant_family": invariant_family,
        "remaining_semantics_unknown": remaining_semantics_unknown,
        "metrics": ["fixture_metric"],
        "status": "READY_FOR_M3_GENERIC_CONTEXTUAL_FOLLOWUP",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "duplicate_evidence_promoted": False,
        "followup_request_counted_as_support": False,
        "followup_request_counted_as_confirmation": False,
    }


def _fake_contextual_executor(cell):
    actor_changed = cell.action in {"ACTION3", "ACTION4"} or bool(
        cell.temporal_action_sequence
    )
    relation_delta = ["distance_decreases"] if cell.action == "ACTION3" else []
    temporal_regularities = {}
    transition_rows = []
    if cell.temporal_action_sequence:
        transition_rows = [
            {
                "action": action,
                "invariant_delta": {
                    entity: {
                        "value_changed": True,
                        "direction": "decreasing",
                        "semantic_interpretation": "unknown",
                    }
                    for entity in cell.requested_entities
                },
                "semantic_interpretation": "unknown",
            }
            for action in cell.temporal_action_sequence
        ]
        temporal_regularities = {
            entity: {
                "value_delta_per_action": [True for _ in cell.temporal_action_sequence],
                "changed_count": len(cell.temporal_action_sequence),
                "steps_observed": len(cell.temporal_action_sequence),
                "monotonicity": True,
                "direction": "decreasing",
                "action_correlation": True,
                "terminal_correlation": "unknown",
                "semantic_interpretation": "unknown",
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
            for entity in cell.requested_entities
        }
    return {
        **cell.to_dict(),
        "status": "EXECUTED",
        "replay_trace": [{"action": action, "action_args": {}} for action in cell.context_replay],
        "entity_delta": {
            entity: {
                "before_present": True,
                "after_present": True,
                "changed": actor_changed,
                "effect_families": ["move_entity"] if actor_changed else [],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
            for entity in cell.requested_entities
        },
        "relation_delta": {
            f"{left}::{right}": {
                "delta_types": relation_delta,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
            for left, right in cell.requested_relation_pairs
        },
        "invariant_delta": {
            entity: {
                "value_changed": True,
                "direction": "decreasing",
                "semantic_interpretation": "unknown",
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
            for entity in cell.requested_entities
        },
        "temporal_transition_rows": transition_rows,
        "temporal_regularities": temporal_regularities,
        "global_delta": {
            "changed_pixels": 5,
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
        "wrong_confirmations": 0,
        "semantic_interpretation": "unknown",
    }


def test_contextual_executor_deduplicates_cells_and_keeps_candidate_only(tmp_path):
    requests_path = tmp_path / "contextual_requests.json"
    requests_path.write_text(json.dumps(_contextual_payload()), encoding="utf-8")

    payload = executor.run_generic_mechanic_contextual_followup_execution(
        contextual_requests_path=requests_path,
        cell_executor=_fake_contextual_executor,
    )

    summary = payload["summary"]
    assert summary["contextual_followup_requests_consumed"] == 5
    assert summary["planned_contextual_cells"] > summary["unique_contextual_execution_cells"]
    assert summary["duplicate_contextual_cells_counted_as_independent"] is False
    assert summary["independent_contexts"] >= 3
    assert summary["context_diversity_assessment"] == "MULTI_PREFIX_CONTEXTS"
    assert summary["controlled_experiments_run"] == summary["unique_contextual_execution_cells"]
    assert summary["support"] == 0
    assert summary["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False
    assert payload["contextual_signal_counted_as_scientific_support"] is False
    assert payload["semantic_interpretation_counted_as_confirmation"] is False
    assert all(row["support"] == 0 for row in payload["request_results"])
    assert any(
        row["candidate_status"] == "TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY"
        for row in payload["request_results"]
    )


def test_contextual_executor_cell_building_merges_duplicate_contexts():
    requests = _contextual_payload()["generic_contextual_followup_requests"]

    planned, links = executor.build_contextual_execution_cells(requests)
    unique = executor.unique_contextual_execution_cells(planned)

    assert links
    assert len(planned) > len(unique)
    assert any(cell.context_replay == ("ACTION4",) and cell.action == "ACTION3" for cell in unique)
    assert any(cell.temporal_action_sequence == ("ACTION3", "ACTION4", "ACTION6") for cell in unique)
    assert all(link["duplicate_contextual_cell_counted_as_independent"] is False for link in links)


def test_contextual_executor_rejects_source_support_or_preexecuted_payload(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_contextual_payload(support=1)), encoding="utf-8")
    executed_path = tmp_path / "executed.json"
    executed_path.write_text(json.dumps(_contextual_payload(executed=True)), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        executor.run_generic_mechanic_contextual_followup_execution(
            contextual_requests_path=support_path,
            cell_executor=_fake_contextual_executor,
        )
    with pytest.raises(ValueError, match="must not already be executed"):
        executor.run_generic_mechanic_contextual_followup_execution(
            contextual_requests_path=executed_path,
            cell_executor=_fake_contextual_executor,
        )


def test_contextual_executor_preserves_unknown_remaining_semantics(tmp_path):
    requests_path = tmp_path / "contextual_requests.json"
    requests_path.write_text(json.dumps(_contextual_payload()), encoding="utf-8")

    payload = executor.run_generic_mechanic_contextual_followup_execution(
        contextual_requests_path=requests_path,
        cell_executor=_fake_contextual_executor,
    )

    counter = next(row for row in payload["request_results"] if row["request_id"] == "req_counter")
    assert counter["remaining_semantics_unknown"] is True
    assert counter["request_result_counted_as_confirmation"] is False
    assert counter["contextual_signal_counted_as_scientific_support"] is False
