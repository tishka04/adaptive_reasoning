import copy
import json

import pytest

from theory.a32 import (
    second_unknown_game_control_dependence_revision_decisions as decisions,
)


@pytest.fixture(scope="module")
def real_payload():
    return decisions.run_a32_second_unknown_game_control_dependence_revision_consumer()


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        decisions.DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_a32_6_confirms_only_the_scope_limited_control_dependent_contrast(
    real_payload,
):
    summary = real_payload["summary"]

    assert summary["source_frontiers_consumed"] == 1
    assert summary["scientific_revision_decisions"] == 1
    assert summary["scope_limited_control_dependent_contrasts_confirmed"] == 1
    assert summary["control_dependent_contrasts_refuted"] == 0
    assert summary["control_dependent_contrasts_unresolved"] == 0
    assert summary["standalone_action2_effects_confirmed"] == 0
    assert summary["standalone_action2_effects_kept_unresolved"] == 1
    assert summary["scientific_support_counted_by_a32"] == 3
    assert summary["raw_support_events_promoted_directly"] == 0
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == (
        decisions.A32_6_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST_CONFIRMED
    )


def test_a32_6_decision_counts_one_support_per_independent_paired_context(
    real_payload,
):
    decision = real_payload["revision_decisions"][0]

    assert decision["decision"] == (
        decisions.CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST
    )
    assert decision["decision_record"]["status"] == "confirmed"
    assert decision["decision_record"]["support"] == 3
    assert decision["decision_record"]["contradictions"] == 0
    assert decision["decision_record"]["experiments_spent"] == 10
    assert decision["input_record"]["status"] == "unresolved"
    assert decision["input_record"]["support"] == 0
    assert decision["evidence_summary"]["raw_support_events"] == 5
    assert decision["evidence_summary"]["raw_support_events_promoted_directly"] == 0
    assert decision["evidence_summary"]["scientific_support_after_a32_review"] == 3
    assert decision["evidence_summary"]["scientific_support_basis"] == (
        "ONE_PER_INDEPENDENT_PAIRED_CONTROL_CONTEXT"
    )


def test_a32_6_preserves_the_paired_control_evidence_and_neutral_exception(
    real_payload,
):
    decision = real_payload["revision_decisions"][0]
    evidence = decision["evidence_summary"]

    assert evidence["independent_paired_control_contexts"] == 3
    assert evidence["paired_control_budgets"] == [50, 150, 300]
    assert evidence["paired_control_effect_gaps"] == [32.0, 32.0, 32.0]
    assert evidence["paired_control_effect_gap_spread"] == 0.0
    assert evidence["action1_controlled_effect_sizes"] == [32.0, 32.0, 32.0]
    assert evidence["action3_controlled_effect_sizes"] == [0.0, 0.0, 0.0]
    assert evidence["target_signals_reproduced_across_control_pairs"] == 3
    assert evidence["replicated_neutral_contexts"] == 1
    assert evidence["unpaired_action1_positive_contexts"] == 2


def test_a32_6_keeps_the_standalone_action2_effect_unresolved(real_payload):
    decision = real_payload["revision_decisions"][0]
    disposition = decision["claim_dispositions"][
        "standalone_unconditional_action2_effect"
    ]

    assert disposition["decision"] == (
        decisions.KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE
    )
    assert disposition["status"] == "unresolved"
    assert disposition["support"] == 0
    assert disposition["confirmation_performed"] is False
    assert disposition["refutation_performed"] is False
    assert decision["standalone_action2_effect_confirmed"] is False
    assert decision["standalone_action2_effect_refuted"] is False
    assert decision["action3_equivalence_counted_as_refutation"] is False
    assert "standalone_action2_effect_is_not_identifiable" in decision["reasons"]
    assert "action3_equivalence_is_not_refutation" in decision["reasons"]


def test_a32_6_limits_confirmation_to_exact_game_contexts_controls_and_metric(
    real_payload,
):
    scope = real_payload["revision_decisions"][0]["scope_limits"]

    assert scope["game_id"] == "wa30-ee6fef47"
    assert scope["metric"] == "local_patch_before_after"
    assert scope["target_action"] == "ACTION2"
    assert scope["control_actions"] == ["ACTION1", "ACTION3"]
    assert scope["budgets"] == [50, 150, 300]
    assert len(scope["reviewed_context_cluster_ids"]) == 6
    assert len(scope["confirmed_paired_context_cluster_ids"]) == 3
    assert len(scope["confirmed_paired_context_snapshot_hashes"]) == 3
    assert scope["neutral_exception_context_cluster_ids"] == [
        "sage6c::context_cluster::001"
    ]
    assert scope["unpaired_action1_positive_context_cluster_ids"] == [
        "sage6c::context_cluster::004",
        "sage6c::context_cluster::006",
    ]
    assert scope["not_generalized_beyond_game"] is True
    assert scope["not_generalized_beyond_exact_paired_contexts"] is True
    assert scope["action1_valid_only_as_recorded_lower_effect_comparator"] is True
    assert scope["standalone_action2_effect_excluded_from_confirmation"] is True


def test_a32_6_prepares_relational_handoff_for_a33_3_without_registry_write(
    real_payload,
):
    assert len(real_payload["a33_handoff_candidates"]) == 1
    handoff = real_payload["a33_handoff_candidates"][0]

    assert handoff["registry_entry_type"] == "CONTROL_DEPENDENT_RELATIONAL_CONTRAST"
    assert handoff["game_id"] == "wa30-ee6fef47"
    assert handoff["target_action"] == "ACTION2"
    assert handoff["control_actions"] == ["ACTION1", "ACTION3"]
    assert handoff["status"] == "confirmed"
    assert handoff["support"] == 3
    assert handoff["budgets"] == [50, 150, 300]
    assert len(handoff["paired_context_snapshot_hashes"]) == 3
    assert [
        row["action2_minus_action1"]
        for row in handoff["controlled_effects_by_pair"]
    ] == [32.0, 32.0, 32.0]
    assert [
        row["action2_minus_action3"]
        for row in handoff["controlled_effects_by_pair"]
    ] == [0.0, 0.0, 0.0]
    assert handoff["standalone_action2_effect_status"] == "unresolved"
    assert handoff["ready_for_A33_registry_review"] is True
    assert handoff["a33_write_performed"] is False


def test_a32_6_top_level_guardrails_are_explicit(real_payload):
    assert real_payload["truth_status"] == decisions.A32_6_TRUTH_STATUS
    assert real_payload["scientific_review_performed"] is True
    assert real_payload["revision_performed"] is True
    assert real_payload["confirmation_performed"] is True
    assert real_payload["refutation_performed"] is False
    assert real_payload["a33_ready"] is True
    assert real_payload["a33_write_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert real_payload["support"] == 3
    assert real_payload["standalone_action2_effect_confirmed"] is False
    assert real_payload["standalone_action2_effect_status"] == "unresolved"
    assert real_payload["action3_equivalence_counted_as_refutation"] is False
    assert real_payload["action1_counted_as_universal_baseline"] is False
    assert real_payload["sage_raw_events_counted_as_support_before_a32_review"] is False
    assert real_payload["scope_limited_confirmation_generalized_beyond_game"] is False
    assert all(real_payload["gate"].values())


def test_a32_6_rejects_source_support_or_prior_verdict(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        decisions.validate_sage6f_control_dependence_revision_source(source)

    source = copy.deepcopy(real_source)
    source["confirmation_performed"] = True
    with pytest.raises(ValueError, match="cannot revise or confirm"):
        decisions.validate_sage6f_control_dependence_revision_source(source)


def test_a32_6_rejects_precounted_verdict_or_context_merge(real_source):
    source = copy.deepcopy(real_source)
    source["paired_control_pattern_counted_as_scientific_verdict"] = True
    with pytest.raises(ValueError, match="cannot pre-count"):
        decisions.validate_sage6f_control_dependence_revision_source(source)

    source = copy.deepcopy(real_source)
    source["context_cluster_manifest"][0]["cross_context_merge_performed"] = True
    with pytest.raises(ValueError, match="boundaries must remain preserved"):
        decisions.validate_sage6f_control_dependence_revision_source(source)


def test_a32_6_rejects_inexact_pair_or_duplicate_paired_context(real_source):
    source = copy.deepcopy(real_source)
    source["paired_control_comparisons"][0][
        "target_signal_reproduced_across_control_pairs"
    ] = False
    with pytest.raises(ValueError, match="must remain exact candidates"):
        decisions.validate_sage6f_control_dependence_revision_source(source)

    source = copy.deepcopy(real_source)
    source["paired_control_comparisons"][1]["context_snapshot_hash"] = source[
        "paired_control_comparisons"
    ][0]["context_snapshot_hash"]
    with pytest.raises(ValueError, match="unique and independent"):
        decisions.validate_sage6f_control_dependence_revision_source(source)


def test_a32_6_requests_more_tests_for_incomplete_dossier(real_source):
    assessment = copy.deepcopy(real_source["a32_review_eligibility_assessment"])
    assessment["ready_for_A32_review"] = False

    reasons = decisions.control_dependence_revision_decision_reasons(
        assessment,
        real_source["paired_control_comparisons"],
        real_source["context_cluster_manifest"],
    )

    assert "sage6f_control_dependence_review_dossier_incomplete" in reasons
    assert decisions.control_dependence_revision_decision_label(reasons) == (
        decisions.REQUEST_MORE_TESTS_FOR_CONTROL_DEPENDENCE
    )


def test_a32_6_refutes_only_an_actual_control_contrast_contradiction(real_source):
    assessment = copy.deepcopy(real_source["a32_review_eligibility_assessment"])
    paired = copy.deepcopy(real_source["paired_control_comparisons"])
    paired[0]["controlled_effect_gap_action1_minus_action3"] = -1.0

    reasons = decisions.control_dependence_revision_decision_reasons(
        assessment,
        paired,
        real_source["context_cluster_manifest"],
    )

    assert "paired_control_contrast_contradiction_observed" in reasons
    assert decisions.control_dependence_revision_decision_label(reasons) == (
        decisions.REFUTE_CONTROL_DEPENDENT_CONTRAST
    )


def test_a32_6_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "a32_6.json"

    decisions.write_a32_second_unknown_game_control_dependence_revision_decisions(
        real_payload, path
    )

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload
