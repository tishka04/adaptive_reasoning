from theory.m2.hypothesis_merger import merge_hypotheses
from theory.m2.local_llm_generator import (
    UNLOCK_HYPOTHESIS_FAMILY,
    LocalLLMConfig,
    StateConditionedMockLLM,
    apply_boundary_guards,
    build_situation_packet,
    guard_llm_item,
    llm_item_to_raw_proposal,
    normalizer_frontier_request,
    parse_llm_json,
    run_generation_from_packet,
)
from theory.m2.normalizer import normalize_raw_proposal
from theory.m2.schema import FrontierConditionedHypothesis, RawHypothesisProposal


def _frontier_request():
    return {
        "request_id": "p2g5::bp35-0a0ad940::risk_aware_objective_completion::001",
        "game_id": "bp35-0a0ad940",
        "frontier_type": "RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER",
        "blocked_capability": "objective_completion_after_risk_aware_safe_conversion",
        "frontier_reason": "RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION",
        "source_frontier_id": "p2g4::bp35::risk_aware_post_stop_no_objective_completion",
        "requested_hypothesis_families": [
            "objective_readiness_detection",
            "post_conversion_commit_action_search",
        ],
    }


def _m1_payload():
    # Deliberately use non-E001 ids so a hardcoded id would fail this fixture.
    return {
        "role_hypothesis_ledger": [
            {
                "entity_id": "E777",
                "role_hypotheses": [
                    {"role": "timer_or_hud", "score": 1.0},
                    {"role": "transformed_object", "score": 0.5},
                ],
            },
            {
                "entity_id": "E182",
                "role_hypotheses": [
                    {"role": "controllable_actor", "score": 0.9},
                ],
            },
        ],
        "action_effect_abstractions": [
            {"action": "ACTION3", "effect_families": ["move_entity", "change_relation"]},
            {"action": "ACTION4", "effect_families": ["transform_entity"]},
        ],
        "dynamic_invariant_candidates": [
            {"family": "monotone_counter_candidate", "entity_id": "E777"},
        ],
        "relation_delta_rows": [{"action": "ACTION3"}],
    }


def _m3g6_payload():
    return {
        "objective_completion_experiment_outcome_status": (
            "PROXY_COMPLETION_DIVERGENCE_CANDIDATE_ONLY"
        ),
        "summary": {
            "objective_completion_signal": False,
            "proxy_progress_without_completion_observed": True,
            "commit_action_cells_blocked": 6,
            "levels_completed_after_rollout_max": 0.0,
            "cells_executed": 66,
        },
    }


def _packet():
    return build_situation_packet(
        _frontier_request(),
        m1_payload=_m1_payload(),
        m3g6_payload=_m3g6_payload(),
        m2g2_payload={"summary": {"hypotheses_generated": 15}},
    )


def _run(config=None):
    return run_generation_from_packet(
        _packet(),
        frontier_request=_frontier_request(),
        config=config or LocalLLMConfig(),
    )


def test_llm_disabled_by_default_uses_mock_backend():
    summary = _run()["hypothesis_payload"]["summary"]

    assert summary["llm_enabled"] is False
    assert summary["local_llm_backend"] == "mock"
    assert summary["fallback_used"] is True
    assert summary["local_llm_error"] is None
    assert summary["hypotheses_generated"] >= 1


def test_mock_yields_valid_hypotheses_across_six_families():
    summary = _run()["hypothesis_payload"]["summary"]

    assert summary["valid_hypotheses"] >= 1
    families = set(summary["hypothesis_families_covered"])
    assert families == {
        "objective_readiness_detection",
        "post_conversion_commit_action_search",
        "goal_state_representation_beyond_safe_progress",
        "proxy_progress_vs_completion_discriminator",
        "risk_aware_selector_completion_gap",
        UNLOCK_HYPOTHESIS_FAMILY,
    }
    # objective_completion_signal + terminal_reentry_rate metrics are runnable now;
    # available_actions_before_after stays blocked-but-candidate.
    assert summary["ready_for_m3_candidate_experiment_request"] == 5
    assert summary["blocked_not_testable_hypotheses"] == 1
    assert summary["direct_unavailable_action_hypotheses"] == 0
    assert summary["action6_extension_retest_hypotheses_generated"] is False


def test_all_hypotheses_stay_candidate_only():
    outputs = _run()
    hypotheses = outputs["hypothesis_payload"]["candidate_hypotheses"]

    assert hypotheses
    for hypothesis in hypotheses:
        assert hypothesis["status"] == "UNRESOLVED"
        assert hypothesis["support"] == 0
        assert hypothesis["truth_status"] == "NOT_EVALUATED_BY_M2"
        assert hypothesis["wrong_confirmations"] == 0
    summary = outputs["hypothesis_payload"]["summary"]
    assert summary["support"] == 0
    assert summary["revision_status"] == "CANDIDATE_ONLY"
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False


def test_non_json_output_is_rejected():
    assert parse_llm_json("Here is my answer: ...") == ()
    assert parse_llm_json("") == ()


def test_support_positive_is_rejected_at_boundary():
    reason = guard_llm_item(
        {"candidate_action": "ACTION3", "support": 1},
        available_actions=["ACTION3", "ACTION4", "ACTION6"],
    )
    assert reason == "asserts_support"


def test_confirmed_and_refuted_are_rejected_at_boundary():
    for status in ("CONFIRMED", "REFUTED"):
        reason = guard_llm_item(
            {"candidate_action": "ACTION3", "status": status},
            available_actions=["ACTION3", "ACTION4", "ACTION6"],
        )
        assert reason == "asserts_status"


def test_direct_unavailable_action_is_rejected():
    reason = guard_llm_item(
        {"candidate_action": "ACTION2"},
        available_actions=["ACTION3", "ACTION4", "ACTION6"],
    )
    assert reason == "direct_unavailable_action"


def test_unlock_target_action_allowed_as_unlock_not_candidate():
    item = {
        "source": "llm",
        "hypothesis_family": UNLOCK_HYPOTHESIS_FAMILY,
        "candidate_action": "ACTION3",
        "primary_metric": "available_actions_before_after",
        "unlock_target_actions": ["ACTION2"],
        "hypothesis_text": "ACTION2 may unlock after a geometric transition.",
        "context_replay": ["ACTION6"],
    }
    assert guard_llm_item(item, available_actions=["ACTION3", "ACTION4", "ACTION6"]) is None

    raw = llm_item_to_raw_proposal(item, frontier_request=_frontier_request(), index=1)
    packet = _packet()
    hypothesis = normalize_raw_proposal(
        raw, frontier_request=normalizer_frontier_request(_frontier_request(), packet)
    )
    assert isinstance(hypothesis, FrontierConditionedHypothesis)
    assert hypothesis.candidate_action == "ACTION3"
    assert hypothesis.unlock_target_actions == ("ACTION2",)
    # available_actions_before_after is known but not yet executor-supported.
    assert hypothesis.testability.testable is False
    assert (
        hypothesis.testability.blocking_reason
        == "metric_known_but_unmeasurable_in_current_executor"
    )


def test_entities_are_read_by_role_candidate_not_hardcoded_id():
    packet = _packet()
    roles = packet.current_state_summary["entity_role_candidates"]
    assert "timer_or_hud" in roles
    assert roles["timer_or_hud"][0]["entity_id"] == "E777"
    assert packet.current_state_summary["hud_or_horizon"]["detected"] is True
    assert "controllable_actor" in roles


def test_mechanistic_context_section_is_not_called_world_model():
    packet = _packet().to_dict()
    assert "mechanistic_context_candidates" in packet
    assert "world_model_candidates" not in packet
    assert (
        packet["mechanistic_context_candidates"]["prediction_confidence_is_not_evidence"]
        is True
    )


def test_merger_dedups_heuristic_and_llm_without_raising_support():
    packet = _packet()
    normalizer_frontier = normalizer_frontier_request(_frontier_request(), packet)

    llm_item = {
        "source": "llm",
        "hypothesis_family": "objective_readiness_detection",
        "candidate_action": "ACTION3",
        "primary_metric": "objective_completion_signal",
        "hypothesis_text": "Readiness feature predicts completion.",
        "context_replay": ["ACTION6"],
        "expected_signal_type": "readiness_feature_predicts_completion_more_than_controls",
    }
    llm_raw = llm_item_to_raw_proposal(
        llm_item, frontier_request=_frontier_request(), index=1
    )
    heuristic_raw = RawHypothesisProposal(
        proposal_id="raw::heuristic::dup",
        source="heuristic",
        source_request_id=normalizer_frontier["request_id"],
        game_id=normalizer_frontier["game_id"],
        frontier_context_id=normalizer_frontier["frontier_context_id"],
        frontier_reason=normalizer_frontier["reason"],
        frontier_step=None,
        hypothesis_family="objective_readiness_detection",
        candidate_action="ACTION3",
        predicted_metric="objective_completion_signal",
        predicted_effect="Readiness feature predicts completion.",
        rationale="Heuristic candidate only.",
        required_context_replay=("ACTION6",),
        expected_signal_type="readiness_feature_predicts_completion_more_than_controls",
    )

    normalized = []
    for raw in (llm_raw, heuristic_raw):
        result = normalize_raw_proposal(raw, frontier_request=normalizer_frontier)
        assert isinstance(result, FrontierConditionedHypothesis)
        normalized.append(result)

    merged = merge_hypotheses(
        normalized,
        frontiers_by_request_id={normalizer_frontier["request_id"]: normalizer_frontier},
    )
    assert len(merged) == 1
    assert set(merged[0].source_generation.sources) == {"heuristic", "llm"}
    assert merged[0].support == 0


def test_boundary_guards_split_accepted_and_rejected():
    items = StateConditionedMockLLM().generate_items(_packet())
    accepted, rejected = apply_boundary_guards(
        items, available_actions=["ACTION3", "ACTION4", "ACTION6"]
    )
    assert len(accepted) == 6
    assert rejected == ()
