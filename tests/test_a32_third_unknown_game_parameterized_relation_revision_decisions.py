import copy
import json

import pytest

import theory.a32 as a32
from theory.a32 import (
    third_unknown_game_parameterized_relation_revision_decisions as decisions,
)


@pytest.fixture(scope="module")
def real_payload():
    return decisions.run_a32_third_unknown_game_parameterized_relation_revision()


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        decisions.DEFAULT_SAGE7D_A32_HANDOFF_PATH.read_text(encoding="utf-8")
    )


def test_a32_7_confirms_only_the_scope_limited_parameterized_relation(real_payload):
    summary = real_payload["summary"]

    assert summary["source_handoffs_consumed"] == 1
    assert summary["scientific_revision_decisions"] == 1
    assert summary["scope_limited_parameterized_relations_confirmed"] == 1
    assert summary["parameterized_relations_refuted"] == 0
    assert summary["parameterized_relations_unresolved"] == 0
    assert summary["autonomous_target_effects_confirmed"] == 0
    assert summary["autonomous_target_effects_kept_unresolved"] == 1
    assert summary["scientific_support_counted_by_a32"] == 8
    assert summary["raw_comparison_events_promoted_directly"] == 0
    assert summary["technical_replication_events_counted_as_support"] == 0
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == decisions.A32_7_RELATION_CONFIRMED


def test_a32_7_counts_one_support_per_independent_exact_context(real_payload):
    decision = real_payload["revision_decisions"][0]
    evidence = decision["evidence_summary"]

    assert decision["decision"] == (
        decisions.CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
    )
    assert decision["decision_record"]["status"] == "confirmed"
    assert decision["decision_record"]["support"] == 8
    assert decision["decision_record"]["contradictions"] == 0
    assert decision["decision_record"]["experiments_spent"] == 8
    assert decision["input_record"]["status"] == "unresolved"
    assert decision["input_record"]["support"] == 0
    assert evidence["independent_contexts"] == 8
    assert evidence["cross_budget_replicated_contexts"] == 4
    assert evidence["raw_comparison_events"] == 26
    assert evidence["technical_replication_events"] == 10
    assert evidence["candidate_support_events"] == 8
    assert evidence["scientific_support_after_a32_review"] == 8
    assert evidence["scientific_support_basis"] == ("ONE_PER_INDEPENDENT_EXACT_CONTEXT")
    assert evidence["raw_comparison_events_promoted_directly"] == 0
    assert evidence["technical_replication_events_counted_as_support"] == 0


def test_a32_7_preserves_stable_differentiating_and_equivalent_effects(real_payload):
    evidence = real_payload["revision_decisions"][0]["evidence_summary"]

    assert evidence["differentiating_control_effect_sizes"] == [2.0] * 8
    assert evidence["equivalent_control_effect_sizes"] == [0.0] * 8
    assert evidence["negative_effect_events"] == 0


def test_a32_7_keeps_the_autonomous_target_effect_unresolved(real_payload):
    decision = real_payload["revision_decisions"][0]
    disposition = decision["claim_dispositions"][
        "autonomous_parameterized_target_effect"
    ]

    assert decision["autonomous_target_effect_decision"] == (
        decisions.KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT
    )
    assert disposition["status"] == "unresolved"
    assert disposition["support"] == 0
    assert disposition["confirmation_performed"] is False
    assert disposition["refutation_performed"] is False
    assert decision["autonomous_target_effect_confirmed"] is False
    assert decision["autonomous_target_effect_refuted"] is False
    assert decision["equivalent_control_counted_as_refutation"] is False
    assert "autonomous_target_effect_is_not_identifiable" in decision["reasons"]
    assert "equivalent_control_is_not_refutation" in decision["reasons"]


def test_a32_7_locks_the_exact_game_metric_parameters_and_contexts(real_payload):
    scope = real_payload["revision_decisions"][0]["scope_limits"]

    assert scope["game_id"] == "tn36-ab4f63cc"
    assert scope["metric"] == "local_patch_before_after"
    assert scope["target_action"] == "ACTION6"
    assert scope["target_action_args"] == {"x": 25, "y": 42}
    assert scope["differentiating_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 34, "y": 51}}
    ]
    assert scope["equivalent_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 41, "y": 44}}
    ]
    assert len(scope["context_assessment_ids"]) == 8
    assert len(scope["context_snapshot_hashes"]) == 8
    assert len(set(scope["context_snapshot_hashes"])) == 8
    assert scope["budgets_observed"] == [50, 150, 300]
    assert scope["not_generalized_beyond_game"] is True
    assert scope["not_generalized_beyond_exact_contexts"] is True
    assert scope["not_generalized_beyond_metric"] is True
    assert scope["not_generalized_beyond_parameter_variants"] is True
    assert scope["autonomous_target_effect_excluded_from_confirmation"] is True


def test_a32_7_prepares_a33_4_handoff_without_registry_write(real_payload):
    assert len(real_payload["a33_handoff_candidates"]) == 1
    handoff = real_payload["a33_handoff_candidates"][0]

    assert handoff["registry_entry_type"] == (
        "CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST"
    )
    assert handoff["game_id"] == "tn36-ab4f63cc"
    assert handoff["target_action"] == "ACTION6"
    assert handoff["target_action_args"] == {"x": 25, "y": 42}
    assert handoff["status"] == "confirmed"
    assert handoff["support"] == 8
    assert len(handoff["context_manifest"]) == 8
    assert len(handoff["source_context_snapshot_hashes"]) == 8
    assert handoff["autonomous_target_effect_status"] == "unresolved"
    assert handoff["ready_for_A33_4_registry_review"] is True
    assert handoff["a33_write_performed"] is False


def test_a32_7_top_level_guardrails_are_explicit(real_payload):
    assert real_payload["truth_status"] == decisions.A32_7_TRUTH_STATUS
    assert real_payload["scientific_review_performed"] is True
    assert real_payload["revision_performed"] is True
    assert real_payload["confirmation_performed"] is True
    assert real_payload["refutation_performed"] is False
    assert real_payload["a33_ready"] is True
    assert real_payload["a33_write_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert real_payload["support"] == 8
    assert real_payload["autonomous_target_effect_status"] == "unresolved"
    assert real_payload["equivalent_control_counted_as_refutation"] is False
    assert real_payload["technical_repetitions_counted_as_support"] is False
    assert (
        real_payload["sage_candidate_events_counted_as_support_before_a32_review"]
        is False
    )
    assert real_payload["scope_limited_confirmation_generalized_beyond_game"] is False
    assert all(real_payload["gate"].values())


def test_a32_7_requests_more_tests_for_an_incomplete_dossier(real_source):
    handoff = copy.deepcopy(real_source["a32_review_handoff_items"][0])
    handoff["ready_for_A32_7_review"] = False

    reasons = decisions.parameterized_relation_revision_decision_reasons(
        handoff, handoff["context_manifest"]
    )

    assert "sage7d_parameterized_relation_review_dossier_incomplete" in reasons
    assert decisions.parameterized_relation_revision_decision_label(reasons) == (
        decisions.REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION
    )


def test_a32_7_refutes_only_an_observed_relational_contradiction(real_source):
    handoff = copy.deepcopy(real_source["a32_review_handoff_items"][0])
    differentiating = handoff["relational_contrast"][
        "differentiating_control_variants"
    ][0]
    for control in handoff["context_manifest"][0]["parameterized_controls"]:
        if control["action_args"] == differentiating["action_args"]:
            control["effect_size"] = -1.0

    reasons = decisions.parameterized_relation_revision_decision_reasons(
        handoff, handoff["context_manifest"]
    )

    assert "parameterized_relation_contradiction_observed" in reasons
    assert decisions.parameterized_relation_revision_decision_label(reasons) == (
        decisions.REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION
    )


def test_a32_7_rejects_source_support_or_preselected_verdict(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        decisions.validate_sage7d_parameterized_relation_revision_source(source)

    source = copy.deepcopy(real_source)
    source["a32_review_handoff_items"][0]["a32_decision_preselected"] = True
    with pytest.raises(ValueError, match="review-ready and undecided"):
        decisions.validate_sage7d_parameterized_relation_revision_source(source)


def test_a32_7_rejects_duplicate_context_or_inexact_event_counts(real_source):
    source = copy.deepcopy(real_source)
    contexts = source["a32_review_handoff_items"][0]["context_manifest"]
    contexts[1]["context_snapshot_hash"] = contexts[0]["context_snapshot_hash"]
    with pytest.raises(ValueError, match="independent and identified"):
        decisions.validate_sage7d_parameterized_relation_revision_source(source)

    source = copy.deepcopy(real_source)
    source["a32_review_handoff_items"][0]["raw_comparison_events"] += 1
    with pytest.raises(ValueError, match="counts must be exact"):
        decisions.validate_sage7d_parameterized_relation_revision_source(source)


def test_a32_7_writer_and_package_exports_round_trip(tmp_path, real_payload):
    path = tmp_path / "a32_7.json"

    decisions.write_a32_third_unknown_game_parameterized_relation_revision(
        real_payload, path
    )

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload
    assert a32.A32ThirdUnknownGameParameterizedRelationDecision is (
        decisions.A32ThirdUnknownGameParameterizedRelationDecision
    )
    assert a32.run_a32_third_unknown_game_parameterized_relation_revision is (
        decisions.run_a32_third_unknown_game_parameterized_relation_revision
    )
