from theory.m2.object_world_model_generator import (
    UNLOCK_ONLY_ACTIONS,
    build_object_world_model_invariant_packet,
    validate_object_world_model_packet,
)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_object_world_model_packet_generated_candidate_only():
    payload = build_object_world_model_invariant_packet()

    validate_object_world_model_packet(payload)
    assert "mechanistic_context_candidates" in payload
    assert "world_model_candidates" not in payload
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["truth_status"] == "NOT_EVALUATED_BY_M2"
    assert payload["summary"]["revision_status"] == "CANDIDATE_ONLY"
    assert payload["summary"]["a32_write_performed"] is False
    assert payload["summary"]["a33_write_performed"] is False
    assert payload["summary"]["world_model_prediction_counted_as_evidence"] is False
    assert payload["summary"]["world_model_score_counted_as_support"] is False

    guarded = [row for row in _walk(payload) if "support" in row]
    assert guarded
    for row in guarded:
        assert row["support"] == 0
        assert row["truth_status"] == "NOT_EVALUATED_BY_M2"
        assert row["revision_status"] == "CANDIDATE_ONLY"
        assert row["a32_write_performed"] is False
        assert row["a33_write_performed"] is False


def test_relation_progress_is_not_completion_signal():
    payload = build_object_world_model_invariant_packet()
    relation_rows = [
        row
        for row in payload["mechanistic_context_candidates"]
        if row["candidate_type"] == "relation_progress_context"
    ]

    assert relation_rows
    assert relation_rows[0]["relation_interpretation"] == "non_completion_signal"
    assert relation_rows[0]["completion_signal"] is False
    assert relation_rows[0]["progress_counted_as_completion"] is False


def test_hud_is_terminal_avoidance_only():
    payload = build_object_world_model_invariant_packet()
    hud_rows = [
        row
        for row in payload["mechanistic_context_candidates"]
        if row["candidate_type"] == "hud_horizon_context"
    ]

    assert hud_rows
    assert hud_rows[0]["hud_policy"] == "terminal_avoidance_only"
    assert hud_rows[0]["hud_counted_as_objective_signal"] is False
    assert hud_rows[0]["hud_counted_as_completion_signal"] is False


def test_unavailable_actions_are_unlock_only():
    payload = build_object_world_model_invariant_packet()
    precondition_rows = [
        row
        for row in payload["mechanistic_context_candidates"]
        if row["candidate_type"] == "unavailable_action_precondition_context"
    ]

    assert precondition_rows
    assert tuple(precondition_rows[0]["unlock_target_actions"]) == UNLOCK_ONLY_ACTIONS
    assert precondition_rows[0]["unavailable_action_policy"] == "unlock_only"
    assert precondition_rows[0]["unavailable_actions_counted_as_support"] is False


def test_entities_are_role_candidates_not_hard_coded_ids():
    payload = build_object_world_model_invariant_packet()
    role_rows = [
        row
        for row in payload["mechanistic_context_candidates"]
        if row["candidate_type"] == "object_role_context"
    ]

    assert role_rows
    for entity in role_rows[0]["entities"]:
        assert entity["role_candidate"]
        assert entity["entity_id"] is None
        assert entity["hard_coded_entity_id_used"] is False
