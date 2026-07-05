import pytest

from theory.m2.object_world_model_generator import (
    build_object_world_model_invariant_packet,
    validate_object_world_model_packet,
)


def test_object_world_model_sanitizes_adversarial_support_and_writes():
    payload = build_object_world_model_invariant_packet(
        raw_context={
            "mechanistic_context_candidates": [
                {
                    "candidate_id": "bad::relation",
                    "candidate_type": "relation_progress_context",
                    "support": 99,
                    "truth_status": "CONFIRMED",
                    "revision_status": "A32_READY",
                    "a32_write_performed": True,
                    "a33_write_performed": True,
                    "completion_signal": True,
                    "progress_counted_as_completion": True,
                    "world_model_score_counted_as_support": True,
                },
                {
                    "candidate_id": "bad::entity",
                    "candidate_type": "object_role_context",
                    "entities": [
                        {
                            "role": "actor",
                            "entity_id": "hard-coded-7",
                            "hard_coded_entity_id_used": True,
                        }
                    ],
                },
            ]
        }
    )

    validate_object_world_model_packet(payload)
    bad_relation = [
        row
        for row in payload["mechanistic_context_candidates"]
        if row["candidate_id"] == "bad::relation"
    ][0]
    assert bad_relation["support"] == 0
    assert bad_relation["truth_status"] == "NOT_EVALUATED_BY_M2"
    assert bad_relation["revision_status"] == "CANDIDATE_ONLY"
    assert bad_relation["a32_write_performed"] is False
    assert bad_relation["a33_write_performed"] is False
    assert bad_relation["relation_interpretation"] == "non_completion_signal"
    assert bad_relation["completion_signal"] is False
    assert bad_relation["progress_counted_as_completion"] is False
    assert bad_relation["world_model_score_counted_as_support"] is False

    bad_entity = [
        row
        for row in payload["mechanistic_context_candidates"]
        if row["candidate_id"] == "bad::entity"
    ][0]
    assert bad_entity["entities"] == [
        {
            "role_candidate": "actor",
            "entity_id": None,
            "hard_coded_entity_id_used": False,
        }
    ]


def test_object_world_model_validator_rejects_world_model_candidates_section():
    payload = build_object_world_model_invariant_packet()
    payload["world_model_candidates"] = []

    with pytest.raises(ValueError, match="mechanistic_context_candidates"):
        validate_object_world_model_packet(payload)
