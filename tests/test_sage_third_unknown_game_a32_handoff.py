import copy
import json

import pytest

import theory.sage as sage
import theory.sage.third_unknown_game_a32_handoff as handoff


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        handoff.DEFAULT_SAGE7C_PARAMETERIZED_CONSOLIDATION_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        handoff.DEFAULT_SAGE7D_A32_HANDOFF_PATH.read_text(encoding="utf-8")
    )


def test_sage7d_compiles_exactly_one_eligible_handoff(real_payload):
    summary = real_payload["summary"]
    items = real_payload["a32_review_handoff_items"]

    assert summary["game_id"] == "tn36-ab4f63cc"
    assert summary["source_candidate_dossiers"] == 3
    assert summary["source_a32_handoff_eligible_candidates"] == 1
    assert summary["handoff_items"] == 1
    assert summary["excluded_candidate_items"] == 2
    assert len(items) == 1
    item = items[0]
    assert item["handoff_id"] == "sage7d::a32_7_handoff::001"
    assert item["source_candidate_id"] == "sage7c::candidate::b98c89c514d9b4ff"
    assert item["candidate_type"] == handoff.RELATIONAL_CANDIDATE_TYPE
    assert item["target_action"] == "ACTION6"
    assert item["target_action_args"] == {"x": 25, "y": 42}
    assert item["metric"] == "local_patch_before_after"


def test_sage7d_preserves_relational_control_roles(real_payload):
    item = real_payload["a32_review_handoff_items"][0]
    relation = item["relational_contrast"]

    assert relation["target_variant"] == {
        "action": "ACTION6",
        "action_args": {"x": 25, "y": 42},
    }
    assert relation["differentiating_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 34, "y": 51}}
    ]
    assert relation["equivalent_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 41, "y": 44}}
    ]
    assert relation["relation_scope"] == (
        "EXACT_TN36_LIVE_PREFIX_CONTEXTS_AND_PARAMETER_VARIANTS"
    )
    assert relation["autonomous_target_effect_status"] == (
        handoff.AUTONOMOUS_EFFECT_UNRESOLVED
    )
    assert item["autonomous_target_effect_confirmed"] is False


def test_sage7d_locks_all_eight_independent_contexts(real_payload):
    item = real_payload["a32_review_handoff_items"][0]
    manifest = item["context_manifest"]

    assert item["independent_contexts"] == 8
    assert item["cross_budget_replicated_contexts"] == 4
    assert item["raw_comparison_events"] == 26
    assert item["technical_replication_events"] == 10
    assert item["candidate_support_events"] == 8
    assert len(manifest) == 8
    assert len({row["context_snapshot_hash"] for row in manifest}) == 8
    assert sum(row["raw_event_count"] for row in manifest) == 26
    assert sum(row["technical_replication_events"] for row in manifest) == 10
    assert sum(row["cross_budget_replicated"] for row in manifest) == 4
    assert all(row["independent_context_count"] == 1 for row in manifest)
    assert all(row["all_repetitions_consistent"] is True for row in manifest)


def test_sage7d_context_manifest_preserves_both_exact_effects(real_payload):
    item = real_payload["a32_review_handoff_items"][0]

    for context in item["context_manifest"]:
        controls = {
            tuple(sorted(row["action_args"].items())): row
            for row in context["parameterized_controls"]
        }
        differentiating = controls[("x", 34), ("y", 51)]
        equivalent = controls[("x", 41), ("y", 44)]
        assert differentiating["target_signal"] == 2.0
        assert differentiating["control_signal"] == 0.0
        assert differentiating["effect_size"] == 2.0
        assert differentiating["discrimination_status"] == (
            "DISCRIMINATING_TARGET_EFFECT_CANDIDATE_ONLY"
        )
        assert equivalent["target_signal"] == 2.0
        assert equivalent["control_signal"] == 2.0
        assert equivalent["effect_size"] == 0.0
        assert equivalent["discrimination_status"] == (
            "NON_DISCRIMINATING_EQUAL_EFFECT_CANDIDATE_ONLY"
        )
        assert len(context["source_comparison_group_ids"]) == 2


def test_sage7d_excludes_both_non_discriminating_candidates(real_payload):
    exclusions = real_payload["excluded_candidate_audit"]

    assert len(exclusions) == 2
    assert {tuple(sorted(row["target_action_args"].items())) for row in exclusions} == {
        (("x", 35), ("y", 42)),
        (("x", 30), ("y", 42)),
    }
    assert all(
        row["a32_eligibility_status"]
        == "A32_REVIEW_INELIGIBLE_NON_DISCRIMINATING_PARAMETERIZED_CANDIDATE_ONLY"
        for row in exclusions
    )
    assert all(row["included_in_handoff"] is False for row in exclusions)
    assert all(row["support"] == 0 for row in exclusions)


def test_sage7d_does_not_preselect_an_a32_decision(real_payload):
    item = real_payload["a32_review_handoff_items"][0]
    summary = real_payload["summary"]

    assert item["allowed_a32_7_decisions"] == list(handoff.ALLOWED_A32_7_DECISIONS)
    assert len(item["scientific_questions_for_a32_7"]) == 4
    assert item["handoff_status"] == handoff.HANDOFF_READY
    assert item["ready_for_A32_7_review"] is True
    assert item["a32_decision_preselected"] is False
    assert item["a32_intake_requested"] is False
    assert summary["allowed_a32_7_decisions"] == list(handoff.ALLOWED_A32_7_DECISIONS)
    assert summary["a32_decision_preselected"] is False


def test_sage7d_passes_gate_and_requests_a32_7_review(real_payload):
    assert all(real_payload["gate"].values())
    assert real_payload["summary"]["gate_passed"] is True
    assert real_payload["summary"]["ready_for_a32_7_scientific_review"] is True
    assert real_payload["summary"]["required_next_step"] == (
        handoff.SAGE7D_A32_REVIEW_REQUIRED
    )
    assert real_payload["outcome_status"] == handoff.SAGE7D_HANDOFF_COMPILED
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == handoff.SAGE7D_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["handoff_counted_as_scientific_support"] is False
    assert real_payload["candidate_eligibility_counted_as_a32_decision"] is False
    assert real_payload["a32_intake_requested"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage7d_runner_is_pure_and_round_trips(tmp_path):
    output = tmp_path / "sage7d.json"

    payload = handoff.run_sage7d_a32_handoff(output_path=output)

    assert output.exists()
    assert payload["summary"]["handoff_items"] == 1
    assert payload["summary"]["handoff_contexts"] == 8
    assert payload["summary"]["excluded_candidate_items"] == 2
    assert all(payload["gate"].values())


def test_sage7d_rejects_mutated_source_contract(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        handoff.validate_sage7d_source(source)

    source = copy.deepcopy(real_source)
    source["a32_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        handoff.validate_sage7d_source(source)

    source = copy.deepcopy(real_source)
    source["candidate_eligibility_counted_as_a32_decision"] = True
    with pytest.raises(ValueError, match="cannot leak support"):
        handoff.validate_sage7d_source(source)

    source = copy.deepcopy(real_source)
    source["a32_handoff_candidates"][0]["autonomous_target_effect_status"] = "CONFIRMED"
    next(
        row
        for row in source["parameterized_candidate_dossiers"]
        if row["a32_handoff_eligible"]
    )["autonomous_target_effect_status"] = "CONFIRMED"
    with pytest.raises(ValueError, match="relational-only"):
        handoff.validate_sage7d_source(source)


def test_sage7d_rejects_control_role_drift(real_source):
    source = copy.deepcopy(real_source)
    candidate = next(
        row
        for row in source["parameterized_candidate_dossiers"]
        if row["a32_handoff_eligible"]
    )
    context_id = candidate["context_assessment_ids"][0]
    context = next(
        row
        for row in source["parameterized_context_assessments"]
        if row["context_assessment_id"] == context_id
    )
    context["parameterized_controls"][0]["effect_size"] = 0.0
    with pytest.raises(ValueError, match="contexts must remain exact"):
        handoff.validate_sage7d_source(source)


def test_sage7d_writer_round_trips(real_payload, tmp_path):
    output = tmp_path / "round_trip.json"

    handoff.write_sage7d_a32_handoff(real_payload, output)

    assert json.loads(output.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_sage7d_api():
    assert sage.DEFAULT_SAGE7D_A32_HANDOFF_PATH == (
        handoff.DEFAULT_SAGE7D_A32_HANDOFF_PATH
    )
    assert sage.run_sage7d_a32_handoff is handoff.run_sage7d_a32_handoff
    assert sage.write_sage7d_a32_handoff is handoff.write_sage7d_a32_handoff
