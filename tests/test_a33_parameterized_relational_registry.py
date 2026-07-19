import copy
import json

import pytest

from theory.a33 import parameterized_relational_registry as registry


@pytest.fixture(scope="module")
def real_payload():
    return registry.run_parameterized_relational_registry_generation()


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        registry.DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_a33_4_registers_one_a32_7_parameterized_relation(real_payload):
    summary = real_payload["summary"]

    assert summary["source_candidates_consumed"] == 1
    assert summary["a32_7_handoff_candidates_consumed"] == 1
    assert summary["parameterized_relational_contrasts_registered"] == 1
    assert summary["autonomous_target_effects_excluded"] == 1
    assert summary["confirmed_support_imported_from_a32_7"] == 8
    assert summary["raw_comparison_events_imported_as_support"] == 0
    assert summary["technical_replication_events_imported_as_support"] == 0
    assert summary["raw_comparison_events_preserved_as_provenance"] == 26
    assert summary["technical_replications_preserved_as_provenance"] == 10
    assert summary["registered_exact_contexts"] == 8
    assert summary["registered_parameterized_control_variants"] == 2
    assert summary["registered_budgets"] == 3
    assert summary["scope_locked_entries"] == 1
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == registry.A33_4_ENTRY_ADDED


def test_a33_4_entry_preserves_the_complete_a32_7_relation(real_payload):
    assert len(real_payload["parameterized_relational_contrasts"]) == 1
    entry = real_payload["parameterized_relational_contrasts"][0]

    assert entry["registry_entry_type"] == (
        registry.CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST
    )
    assert entry["game_id"] == "tn36-ab4f63cc"
    assert entry["target_action"] == "ACTION6"
    assert entry["target_action_args"] == {"x": 25, "y": 42}
    assert entry["differentiating_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 34, "y": 51}}
    ]
    assert entry["equivalent_control_variants"] == [
        {"action": "ACTION6", "action_args": {"x": 41, "y": 44}}
    ]
    assert entry["predicted_metric"] == "local_patch_before_after"
    assert entry["status"] == "confirmed"
    assert entry["confirmed_support"] == 8
    assert entry["contradictions"] == 0
    assert entry["experiments_spent"] == 8
    assert entry["budgets"] == [50, 150, 300]


def test_a33_4_entry_preserves_context_identity_and_provenance(real_payload):
    entry = real_payload["parameterized_relational_contrasts"][0]

    assert len(entry["context_assessment_ids"]) == 8
    assert len(entry["context_snapshot_hashes"]) == 8
    assert len(set(entry["context_snapshot_hashes"])) == 8
    assert len(entry["context_manifest"]) == 8
    assert entry["raw_comparison_events_provenance"] == 26
    assert entry["technical_replication_events_provenance"] == 10
    assert all(context["support"] == 0 for context in entry["context_manifest"])
    assert all(
        len(context["parameterized_controls"]) == 2
        for context in entry["context_manifest"]
    )


def test_a33_4_scope_guards_prevent_parameter_or_context_generalization(
    real_payload,
):
    entry = real_payload["parameterized_relational_contrasts"][0]

    assert entry["known_scope"] == (
        "game_metric_target_parameterized_controls_exact_contexts"
    )
    assert entry["scope_game_locked"] is True
    assert entry["scope_metric_locked"] is True
    assert entry["scope_target_parameter_locked"] is True
    assert entry["scope_control_parameters_locked"] is True
    assert entry["scope_contexts_locked"] is True
    assert entry["scope_budgets_locked"] is True
    assert entry["not_generalized_beyond_game"] is True
    assert entry["not_generalized_beyond_metric"] is True
    assert entry["not_generalized_beyond_parameter_variants"] is True
    assert entry["not_generalized_beyond_exact_contexts"] is True
    assert entry["autonomous_target_effect_excluded"] is True
    assert entry["raw_events_excluded_from_direct_support"] is True
    assert entry["technical_repetitions_excluded_from_support"] is True


def test_a33_4_excludes_the_autonomous_target_effect(real_payload):
    excluded = real_payload["excluded_claim_audit"]

    assert len(excluded) == 1
    assert excluded[0]["claim"] == "AUTONOMOUS_PARAMETERIZED_TARGET_EFFECT"
    assert excluded[0]["claim_status"] == "unresolved"
    assert excluded[0]["claim_decision"] == (
        registry.KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT
    )
    assert excluded[0]["claim_support"] == 0
    assert excluded[0]["registered"] is False
    assert excluded[0]["counted_as_refutation"] is False


def test_a33_4_reuses_a32_truth_without_new_scientific_verdict(real_payload):
    assert real_payload["status"] == "REGISTERED"
    assert real_payload["truth_status"] == registry.A33_4_TRUTH_STATUS
    assert real_payload["registry_validation_performed"] is True
    assert real_payload["registration_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["a33_write_performed"] is True
    assert real_payload["source_a32_7_mutated"] is False
    assert real_payload["legacy_a33_1_registry_mutated"] is False
    assert real_payload["scoped_a33_2_registry_mutated"] is False
    assert real_payload["relational_a33_3_registry_mutated"] is False
    assert real_payload["support"] == 8
    assert real_payload["support_origin"] == "A32.7_DECISION_RECORDS_ONLY"
    assert real_payload["raw_comparison_events_imported_as_support"] == 0
    assert real_payload["technical_replication_events_imported_as_support"] == 0
    assert real_payload["autonomous_target_effect_registered"] is False
    assert real_payload["parameterized_controls_counted_as_distinct_actions"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_a33_4_generation_does_not_modify_existing_a33_registries():
    paths = (
        registry.Path("diagnostics/a33/confirmed_mechanics_registry.json"),
        registry.Path("diagnostics/a33/scoped_unknown_game_registry.json"),
        registry.Path("diagnostics/a33/control_dependent_relational_registry.json"),
    )
    before = tuple(path.read_bytes() for path in paths)

    registry.run_parameterized_relational_registry_generation()

    assert tuple(path.read_bytes() for path in paths) == before


def test_a33_4_rejects_a32_7_prewrite_or_wrong_confirmation(real_source):
    source = copy.deepcopy(real_source)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot pre-write"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["wrong_confirmations"] = 1
    with pytest.raises(ValueError, match="wrong_confirmations must remain 0"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)


def test_a33_4_rejects_autonomous_confirmation_or_scope_unlock(real_source):
    source = copy.deepcopy(real_source)
    source["autonomous_target_effect_confirmed"] = True
    with pytest.raises(ValueError, match="must remain unresolved"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["not_generalized_beyond_parameter_variants"] = (
        False
    )
    with pytest.raises(ValueError, match="scope must remain locked"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)


def test_a33_4_rejects_parameter_or_context_identity_changes(real_source):
    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["target_action_args"]["x"] = 26
    with pytest.raises(ValueError, match="parameter scope changed"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    contexts = source["a33_handoff_candidates"][0]["context_manifest"]
    contexts[1]["context_snapshot_hash"] = contexts[0]["context_snapshot_hash"]
    with pytest.raises(ValueError, match="eight exact unique contexts"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)


def test_a33_4_rejects_support_recount_or_effect_relabelling(real_source):
    source = copy.deepcopy(real_source)
    source["revision_decisions"][0]["evidence_summary"][
        "technical_replication_events_counted_as_support"
    ] = 1
    with pytest.raises(ValueError, match="provenance counts must remain exact"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["revision_decisions"][0]["evidence_summary"][
        "equivalent_control_effect_sizes"
    ][0] = 1.0
    with pytest.raises(ValueError, match="provenance counts must remain exact"):
        registry.validate_a32_7_parameterized_relational_registry_source(source)


def test_a33_4_writer_and_package_exports_round_trip(tmp_path, real_payload):
    path = tmp_path / "a33_4.json"

    registry.write_parameterized_relational_registry(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload

    import theory.a33 as a33

    assert a33.ParameterizedRelationalRegistryEntry is (
        registry.ParameterizedRelationalRegistryEntry
    )
    assert a33.build_parameterized_relational_registry is (
        registry.build_parameterized_relational_registry
    )
