import copy
import json

import pytest

import theory.sage as sage
import theory.sage.third_unknown_game_parameterized_consolidation as consolidation


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        consolidation.DEFAULT_SAGE7B_PARAMETERIZED_EXECUTION_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        consolidation.DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_sage7c_consolidates_all_events_without_recounting_replays(real_payload):
    summary = real_payload["summary"]
    groups = real_payload["consolidated_comparison_groups"]

    assert summary["game_id"] == "tn36-ab4f63cc"
    assert summary["raw_comparison_events"] == 36
    assert summary["consolidated_comparison_groups"] == 20
    assert summary["technical_replication_events"] == 16
    assert sum(row["raw_event_count"] for row in groups) == 36
    assert sum(row["technical_replication_events"] for row in groups) == 16
    assert all(row["independent_context_count"] == 1 for row in groups)
    assert all(row["replication_consistent"] is True for row in groups)
    assert len({row["comparison_group_id"] for row in groups}) == 20


def test_sage7c_builds_ten_exact_multi_control_contexts(real_payload):
    summary = real_payload["summary"]
    contexts = real_payload["parameterized_context_assessments"]

    assert summary["independent_parameterized_contexts"] == 10
    assert summary["cross_budget_replicated_contexts"] == 6
    assert summary["control_dependent_contexts"] == 8
    assert summary["non_discriminating_contexts"] == 2
    assert summary["contradictory_contexts"] == 0
    assert len(contexts) == 10
    assert len({row["context_assessment_id"] for row in contexts}) == 10
    assert all(row["parameterized_controls_count"] == 2 for row in contexts)
    assert all(row["all_repetitions_consistent"] is True for row in contexts)
    assert all(row["independent_context_count"] == 1 for row in contexts)


def test_sage7c_identifies_one_control_dependent_candidate(real_payload):
    candidates = real_payload["parameterized_candidate_dossiers"]
    eligible = real_payload["a32_handoff_candidates"]
    summary = real_payload["summary"]

    assert len(candidates) == 3
    assert len(eligible) == 1
    candidate = eligible[0]
    assert candidate["candidate_id"] == "sage7c::candidate::b98c89c514d9b4ff"
    assert candidate["candidate_type"] == (
        "CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST"
    )
    assert candidate["game_id"] == "tn36-ab4f63cc"
    assert candidate["target_action"] == "ACTION6"
    assert candidate["target_action_args"] == {"x": 25, "y": 42}
    assert candidate["metric"] == "local_patch_before_after"
    assert candidate["independent_contexts"] == 8
    assert candidate["control_dependent_contexts"] == 8
    assert candidate["non_discriminating_contexts"] == 0
    assert candidate["contradictory_contexts"] == 0
    assert candidate["cross_budget_replicated_contexts"] == 4
    assert candidate["raw_comparison_events"] == 26
    assert candidate["technical_replication_events"] == 10
    assert candidate["candidate_support_events"] == 8
    assert candidate["candidate_support_events_counted_as_scientific_support"] is False
    assert candidate["a32_eligibility_status"] == (
        consolidation.A32_ELIGIBLE_CONTROL_DEPENDENT
    )
    assert candidate["a32_handoff_eligible"] is True
    assert summary["a32_handoff_eligible_candidates"] == 1
    assert summary["eligible_candidate_independent_contexts"] == 8


def test_sage7c_preserves_differentiating_and_equivalent_controls(real_payload):
    candidate = real_payload["a32_handoff_candidates"][0]

    assert candidate["differentiating_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 34, "y": 51}}
    ]
    assert candidate["equivalent_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 41, "y": 44}}
    ]
    assert candidate["negative_control_events"] == 0
    assert candidate["minimum_independent_contexts_required"] == 3
    assert candidate["minimum_controls_per_context_required"] == 2
    assert candidate["minimum_cross_budget_replicated_contexts_required"] == 1
    assert candidate["minimum_controls_met"] is True
    assert candidate["all_repetitions_consistent"] is True


def test_sage7c_keeps_every_autonomous_target_effect_unresolved(real_payload):
    candidates = real_payload["parameterized_candidate_dossiers"]
    summary = real_payload["summary"]

    assert summary["autonomous_target_effects_confirmed"] == 0
    assert summary["autonomous_target_effects_unresolved"] == 3
    assert all(
        row["autonomous_target_effect_status"]
        == consolidation.AUTONOMOUS_EFFECT_UNRESOLVED
        for row in candidates
    )
    assert all(row["support"] == 0 for row in candidates)
    assert all(row["a32_decision_performed"] is False for row in candidates)
    assert all(
        row["technical_repetitions_counted_as_independent_contexts"] is False
        for row in candidates
    )
    assert all(
        row["parameterized_controls_counted_as_distinct_actions"] is False
        for row in candidates
    )


def test_sage7c_passes_gate_and_requires_handoff_not_intake(real_payload):
    assert all(real_payload["gate"].values())
    assert real_payload["summary"]["gate_passed"] is True
    assert real_payload["summary"]["ready_for_a32_handoff_compilation"] is True
    assert real_payload["summary"]["required_next_step"] == (
        consolidation.SAGE7C_HANDOFF_REQUIRED
    )
    assert real_payload["outcome_status"] == (
        consolidation.SAGE7C_CONTROL_DEPENDENT_DOSSIER
    )
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == consolidation.SAGE7C_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["raw_events_counted_as_scientific_support"] is False
    assert (
        real_payload["technical_repetitions_counted_as_independent_contexts"] is False
    )
    assert real_payload["candidate_eligibility_counted_as_a32_decision"] is False
    assert real_payload["a32_intake_requested"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage7c_runner_is_pure_and_round_trips(tmp_path):
    output = tmp_path / "sage7c.json"

    payload = consolidation.run_sage7c_parameterized_consolidation(output_path=output)

    assert output.exists()
    assert payload["summary"]["consolidated_comparison_groups"] == 20
    assert payload["summary"]["independent_parameterized_contexts"] == 10
    assert payload["summary"]["a32_handoff_eligible_candidates"] == 1
    assert all(payload["gate"].values())


def test_sage7c_rejects_mutated_source_scientific_state(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        consolidation.validate_sage7c_source(source)

    source = copy.deepcopy(real_source)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        consolidation.validate_sage7c_source(source)

    source = copy.deepcopy(real_source)
    source["positive_deltas_counted_as_support"] = True
    with pytest.raises(ValueError, match="cannot carry a scientific verdict"):
        consolidation.validate_sage7c_source(source)

    source = copy.deepcopy(real_source)
    source["executed_parameterized_experiments"][0][
        "protocol_substitution_detected"
    ] = True
    with pytest.raises(ValueError, match="must be exact and candidate-only"):
        consolidation.validate_sage7c_source(source)


def test_sage7c_rejects_duplicate_or_count_mismatched_comparisons(real_source):
    source = copy.deepcopy(real_source)
    comparison = source["executed_parameterized_experiments"][1][
        "parameterized_comparisons"
    ][0]
    comparison["comparison_id"] = source["executed_parameterized_experiments"][0][
        "parameterized_comparisons"
    ][0]["comparison_id"]
    with pytest.raises(ValueError, match="comparison ids must be unique"):
        consolidation.validate_sage7c_source(source)

    source = copy.deepcopy(real_source)
    source["summary"]["comparison_events"] = 35
    with pytest.raises(ValueError, match="comparison count must be exact"):
        consolidation.validate_sage7c_source(source)


def test_sage7c_writer_round_trips(real_payload, tmp_path):
    output = tmp_path / "round_trip.json"

    consolidation.write_sage7c_parameterized_consolidation(real_payload, output)

    assert json.loads(output.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_sage7c_api():
    assert sage.DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH == (
        consolidation.DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH
    )
    assert (
        sage.run_sage7c_parameterized_consolidation
        is consolidation.run_sage7c_parameterized_consolidation
    )
    assert (
        sage.write_sage7c_parameterized_consolidation
        is consolidation.write_sage7c_parameterized_consolidation
    )
