import json

from theory.a32.patch_similarity_revision_decisions import (
    CONFIRM_AFTER_SCOPE_LIMITED_REVISION,
    REFUTE_AFTER_REVISION,
    REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS,
    SCOPE_LIMITED_CANDIDATE_ONLY,
    build_a32_patch_similarity_revision_decisions,
    decision_label,
    decision_reasons,
    run_a32_patch_similarity_revision_decision_consumer,
)


def _accepted_candidate(**overrides):
    item = {
        "queue_item_id": "m3_21::bp35::ACTION4_ACTION6::local_patch_transformability",
        "game_id": "bp35-0a0ad940",
        "key": (
            "patch_similarity_rule::bp35-0a0ad940::ACTION4_ACTION6::"
            "local_patch_transformability"
        ),
        "description": (
            "ACTION4 after ACTION6/ACTION3 may open patch-similar ACTION6 "
            "affordances selected by local_patch_transformability."
        ),
        "intake_status": "ACCEPTED_FOR_SCIENTIFIC_REVISION",
        "candidate_rule_family": "local_patch_transformability",
        "candidate_mechanic": "repositioning_opens_patch_similar_action6_affordances",
        "candidate_generativity": (
            "SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"
        ),
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "successful_args_total": [
            {"x": 12, "y": 0},
            {"x": 24, "y": 0},
            {"x": 30, "y": 12},
            {"x": 36, "y": 12},
            {"x": 42, "y": 12},
            {"x": 48, "y": 12},
        ],
        "failed_args": [{"x": 30, "y": 0}],
        "success_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "diagnostic_metrics": [
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "changed_pixels_role": "effect_radar_not_success_metric",
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_A32_INTAKE",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "diagnostic_contradictions_counted_as_refutation": False,
        "evidence_summary": {
            "controlled_experiments_run": 8,
            "successful_args_total_count": 6,
            "failed_args_count": 1,
            "new_expansion_successes_count": 2,
            "source_success_metric_support_events": 4,
            "source_success_metric_contradiction_events": 0,
            "source_diagnostic_contradiction_events": 2,
            "diagnostic_contradictions_counted_as_refutation": False,
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M3",
            "wrong_confirmations": 0,
        },
    }
    item.update(overrides)
    return item


def _payload(item=None):
    return {
        "summary": {
            "accepted_for_scientific_revision": 1,
            "support": 0,
            "wrong_confirmations": 0,
        },
        "accepted_candidates": [item or _accepted_candidate()],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_A32_INTAKE",
        "wrong_confirmations": 0,
    }


def test_patch_similarity_revision_decision_is_scope_limited_candidate_only():
    decisions = build_a32_patch_similarity_revision_decisions(_payload())
    decision = decisions[0].to_dict()

    assert decision["decision"] == SCOPE_LIMITED_CANDIDATE_ONLY
    assert decision["recommended_next_step"] == REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS
    assert "mono_game_scope" in decision["reasons"]
    assert "mono_context_scope" in decision["reasons"]
    assert "scope_not_a33_ready" in decision["reasons"]
    assert decision["scope_limits"]["not_generalized_beyond_context"] is True
    assert decision["scope_limits"]["not_a33_ready"] is True
    assert len(decision["requested_followup_tests"]) == 2
    assert decision["requested_followup_tests"][0]["followup_family"] == (
        "outside_known_y12_region_probe"
    )
    assert decision["input_record"]["status"] == "unresolved"
    assert decision["input_record"]["support"] == 0
    assert decision["decision_record"]["status"] == "unresolved"
    assert decision["decision_record"]["support"] == 0
    assert decision["scientific_review_performed"] is True
    assert decision["revision_performed"] is False
    assert decision["confirmation_performed"] is False
    assert decision["refutation_performed"] is False
    assert decision["a33_ready"] is False
    assert decision["a33_write_performed"] is False
    assert decision["wrong_confirmations"] == 0
    assert decision["diagnostic_contradictions_counted_as_refutation"] is False


def test_patch_similarity_revision_decision_run_writes_no_confirmation(tmp_path):
    path = tmp_path / "patch_similarity_revision_intake.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = run_a32_patch_similarity_revision_decision_consumer(intake_path=path)

    summary = payload["summary"]
    assert summary["intake_candidates_consumed"] == 1
    assert summary["scope_limited_candidate_only"] == 1
    assert summary["recommended_more_tests"] == 1
    assert summary["confirm_after_scope_limited_revision"] == 0
    assert summary["refute_after_revision"] == 0
    assert summary["a33_ready_candidates"] == 0
    assert summary["a33_write_performed"] is False
    assert summary["scientific_review_performed"] is True
    assert summary["revision_performed"] is False
    assert summary["confirmation_performed"] is False
    assert summary["refutation_performed"] is False
    assert summary["decision_records_unresolved"] == 1
    assert summary["decision_records_confirmed"] == 0
    assert summary["decision_records_refuted"] == 0
    assert summary["wrong_confirmations"] == 0
    assert payload["decision_records"][0]["status"] == "unresolved"
    assert payload["decision_records"][0]["support"] == 0
    assert payload["a33_write_performed"] is False


def test_patch_similarity_revision_decision_requests_more_tests_when_weak():
    weak = _accepted_candidate(
        evidence_summary={
            **_accepted_candidate()["evidence_summary"],
            "successful_args_total_count": 1,
        }
    )
    reasons = decision_reasons(weak, evidence=weak["evidence_summary"])

    assert "insufficient_distinct_successful_args" in reasons
    assert decision_label(reasons) == REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS


def test_patch_similarity_revision_decision_refutes_success_metric_contradiction():
    contradicted = _accepted_candidate(
        evidence_summary={
            **_accepted_candidate()["evidence_summary"],
            "source_success_metric_contradiction_events": 1,
        }
    )
    decisions = build_a32_patch_similarity_revision_decisions(_payload(contradicted))
    decision = decisions[0].to_dict()

    assert decision["decision"] == REFUTE_AFTER_REVISION
    assert "success_metric_contradictions_present" in decision["reasons"]
    assert decision["decision_record"]["status"] == "refuted"
    assert decision["decision_record"]["contradictions"] == 1
    assert decision["refutation_performed"] is True
    assert decision["a33_write_performed"] is False


def test_patch_similarity_revision_decision_can_confirm_only_with_scope_validation():
    expanded = _accepted_candidate(
        evidence_summary={
            **_accepted_candidate()["evidence_summary"],
            "distinct_contexts_validated": 2,
        },
        context_replay=["ACTION6", "ACTION3", "ACTION4", "ACTION6"],
    )
    decisions = build_a32_patch_similarity_revision_decisions(_payload(expanded))
    decision = decisions[0].to_dict()

    assert decision["decision"] == CONFIRM_AFTER_SCOPE_LIMITED_REVISION
    assert "scope_limited_confirmation_criteria_satisfied" in decision["reasons"]
    assert decision["decision_record"]["status"] == "confirmed"
    assert decision["confirmation_performed"] is True
    assert decision["a33_ready"] is True
    assert decision["a33_write_performed"] is False
