import copy
import json

import pytest

from theory.a32 import unknown_game_parameterized_control_revision_decisions as decisions


@pytest.fixture(scope="module")
def real_payload():
    return decisions.run_a32_unknown_game_parameterized_control_revision_consumer()


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        decisions.DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH.read_text(
            encoding="utf-8"
        )
    )


def _candidate_rows(source, action):
    assessment = next(
        row for row in source["candidate_protocol_assessments"] if row["action"] == action
    )
    experiments = [
        row
        for row in source["executed_parameterized_control_experiments"]
        if row["candidate_id"] == assessment["candidate_id"]
    ]
    return assessment, experiments


def test_a32_5_produces_one_scoped_confirmation_and_one_non_identifiable_result(
    real_payload,
):
    summary = real_payload["summary"]

    assert summary["source_candidates_consumed"] == 2
    assert summary["scientific_revision_decisions"] == 2
    assert summary["scope_limited_confirmations"] == 1
    assert summary["non_identifiable_candidates_kept_unresolved"] == 1
    assert summary["candidates_refuted"] == 0
    assert summary["candidates_requesting_more_tests"] == 0
    assert summary["decision_records_confirmed"] == 1
    assert summary["decision_records_unresolved"] == 1
    assert summary["scientific_support_counted_by_a32"] == 4
    assert summary["a33_ready_candidates"] == 1
    assert summary["status"] == "MIXED_CONFIRMED_AND_UNRESOLVED"
    assert summary["outcome_status"] == (
        decisions.A32_5_MIXED_SCOPE_CONFIRMATION_AND_NON_IDENTIFIABILITY
    )


def test_a32_5_keeps_action6_unresolved_without_false_refutation(real_payload):
    action6 = next(
        row for row in real_payload["revision_decisions"] if row["action"] == "ACTION6"
    )

    assert action6["decision"] == (
        decisions.KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL
    )
    assert action6["decision_record"]["status"] == "unresolved"
    assert action6["decision_record"]["support"] == 0
    assert action6["decision_record"]["experiments_spent"] == 4
    assert action6["confirmation_performed"] is False
    assert action6["refutation_performed"] is False
    assert action6["a33_ready"] is False
    assert action6["evidence_summary"]["target_signals"] == [5.0] * 4
    assert action6["evidence_summary"]["control_signals"] == [5.0] * 4
    assert "target_argument_specific_effect_is_not_identifiable" in action6["reasons"]
    assert (
        "non_discrimination_is_not_refutation_of_generic_action_effect"
        in action6["reasons"]
    )


def test_a32_5_confirms_action5_only_inside_the_observed_scope(real_payload):
    action5 = next(
        row for row in real_payload["revision_decisions"] if row["action"] == "ACTION5"
    )

    assert action5["decision"] == (
        decisions.CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION
    )
    assert action5["decision_record"]["status"] == "confirmed"
    assert action5["decision_record"]["support"] == 4
    assert action5["decision_record"]["contradictions"] == 0
    assert action5["decision_record"]["experiments_spent"] == 4
    assert action5["confirmation_performed"] is True
    assert action5["refutation_performed"] is False
    assert action5["a33_ready"] is True
    assert action5["evidence_summary"]["target_signals"] == [21.0] * 4
    assert action5["evidence_summary"]["control_signals"] == [4.0] * 4
    assert action5["scope_limits"]["not_generalized_beyond_game"] is True
    assert action5["scope_limits"]["not_generalized_to_other_actions"] is True


def test_a32_5_prepares_only_action5_for_a33_review_without_writing_a33(real_payload):
    assert len(real_payload["a33_handoff_candidates"]) == 1
    handoff = real_payload["a33_handoff_candidates"][0]

    assert handoff["game_id"] == "sb26-7fbdac44"
    assert handoff["action"] == "ACTION5"
    assert handoff["action_args"] is None
    assert handoff["status"] == "confirmed"
    assert handoff["support"] == 4
    assert handoff["measurement"] == "local_patch_before_after"
    assert handoff["budgets"] == [50, 300]
    assert len(handoff["context_snapshot_hashes"]) == 4
    assert [row["action_args"] for row in handoff["parameterized_control_variants"]] == [
        {"x": 21, "y": 28},
        {"x": 39, "y": 28},
    ]
    assert handoff["ready_for_A33_registry_review"] is True
    assert handoff["a33_write_performed"] is False


def test_a32_5_top_level_guardrails_are_explicit(real_payload):
    assert real_payload["scientific_review_performed"] is True
    assert real_payload["revision_performed"] is True
    assert real_payload["confirmation_performed"] is True
    assert real_payload["refutation_performed"] is False
    assert real_payload["a33_ready"] is True
    assert real_payload["a33_write_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert real_payload["support"] == 4
    assert real_payload["parameterized_controls_counted_as_distinct_actions"] is False
    assert real_payload["sage_candidate_events_counted_as_support_before_a32_review"] is False
    assert real_payload["neutral_events_counted_as_refutation"] is False
    assert real_payload["non_discrimination_counted_as_refutation"] is False
    assert real_payload["scope_limited_confirmation_generalized_beyond_game"] is False


def test_a32_5_rejects_source_support_or_prior_scientific_decision(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        decisions.validate_sage5j_parameterized_control_revision_source(source)

    source = copy.deepcopy(real_source)
    source["confirmation_performed"] = True
    with pytest.raises(ValueError, match="cannot revise or confirm"):
        decisions.validate_sage5j_parameterized_control_revision_source(source)


def test_a32_5_rejects_precounted_support_or_false_neutral_refutation(real_source):
    source = copy.deepcopy(real_source)
    source["parameterized_control_events_counted_as_scientific_support"] = True
    with pytest.raises(ValueError, match="cannot count candidate events"):
        decisions.validate_sage5j_parameterized_control_revision_source(source)

    source = copy.deepcopy(real_source)
    source["neutral_events_counted_as_refutation"] = True
    with pytest.raises(ValueError, match="cannot count as refutation"):
        decisions.validate_sage5j_parameterized_control_revision_source(source)


def test_a32_5_rejects_protocol_substitution_or_inexact_replay(real_source):
    source = copy.deepcopy(real_source)
    source["summary"]["protocol_substitutions_detected"] = 1
    with pytest.raises(ValueError, match="substituted"):
        decisions.validate_sage5j_parameterized_control_revision_source(source)

    source = copy.deepcopy(real_source)
    source["executed_parameterized_control_experiments"][0][
        "live_prefix_replay_exact"
    ] = False
    with pytest.raises(ValueError, match="replay exactly"):
        decisions.validate_sage5j_parameterized_control_revision_source(source)


def test_a32_5_requests_more_tests_for_an_incomplete_protocol(real_source):
    assessment, experiments = _candidate_rows(real_source, "ACTION5")
    assessment = copy.deepcopy(assessment)
    assessment["protocol_execution_complete"] = False

    reasons = decisions.parameterized_control_revision_decision_reasons(
        assessment, experiments
    )

    assert "pre_registered_protocol_execution_incomplete" in reasons
    assert decisions.parameterized_control_revision_decision_label(reasons) == (
        decisions.REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL
    )


def test_a32_5_refutes_only_an_actual_parameterized_control_contradiction(real_source):
    assessment, experiments = _candidate_rows(real_source, "ACTION5")
    assessment = copy.deepcopy(assessment)
    assessment["raw_support_events"] = 3
    assessment["raw_contradiction_events"] = 1

    reasons = decisions.parameterized_control_revision_decision_reasons(
        assessment, experiments
    )

    assert "parameterized_control_contradiction_observed" in reasons
    assert decisions.parameterized_control_revision_decision_label(reasons) == (
        decisions.REFUTE_AFTER_PARAMETERIZED_CONTROL_CONTRADICTION
    )


def test_a32_5_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "a32_5.json"

    decisions.write_a32_unknown_game_parameterized_control_revision_decisions(
        real_payload, path
    )

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload
