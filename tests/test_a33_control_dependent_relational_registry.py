import copy
import json

import pytest

from theory.a33 import control_dependent_relational_registry as registry


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        registry.DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return registry.run_control_dependent_relational_registry_generation()


def test_a33_3_registers_one_scope_locked_relational_contrast(real_payload):
    summary = real_payload["summary"]

    assert summary["source_candidates_consumed"] == 1
    assert summary["a32_6_handoff_candidates_consumed"] == 1
    assert summary["control_dependent_relational_contrasts_registered"] == 1
    assert summary["unresolved_claims_excluded"] == 2
    assert summary["standalone_action2_effects_excluded"] == 1
    assert summary["action1_universal_baselines_excluded"] == 1
    assert summary["confirmed_support_imported_from_a32_6"] == 3
    assert summary["raw_sage_support_events_imported_directly"] == 0
    assert summary["registered_paired_contexts"] == 3
    assert summary["registered_control_actions"] == 2
    assert summary["registered_budgets"] == 3
    assert summary["scope_locked_entries"] == 1
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == registry.A33_3_ENTRY_ADDED


def test_a33_3_entry_preserves_the_complete_a32_6_relation(real_payload):
    assert len(real_payload["control_dependent_relational_contrasts"]) == 1
    entry = real_payload["control_dependent_relational_contrasts"][0]

    assert entry["registry_entry_type"] == (
        registry.CONTROL_DEPENDENT_RELATIONAL_CONTRAST
    )
    assert entry["game_id"] == "wa30-ee6fef47"
    assert entry["target_action"] == "ACTION2"
    assert entry["control_actions"] == ["ACTION1", "ACTION3"]
    assert entry["mechanic_family"] == (
        "control_dependent_local_patch_change_candidate"
    )
    assert entry["predicted_metric"] == "local_patch_before_after"
    assert entry["status"] == "confirmed"
    assert entry["confirmed_support"] == 3
    assert entry["contradictions"] == 0
    assert entry["experiments_spent"] == 10
    assert entry["budgets"] == [50, 150, 300]
    assert len(entry["paired_context_cluster_ids"]) == 3
    assert len(entry["paired_context_snapshot_hashes"]) == 3
    assert len(entry["source_observation_ids"]) == 10
    assert len(entry["source_paired_comparison_ids"]) == 3


def test_a33_3_entry_preserves_the_32_0_paired_effects(real_payload):
    entry = real_payload["control_dependent_relational_contrasts"][0]
    effects = entry["controlled_effects_by_pair"]

    assert len(effects) == 3
    assert [row["action2_minus_action1"] for row in effects] == [32.0] * 3
    assert [row["action2_minus_action3"] for row in effects] == [0.0] * 3
    assert [row["paired_comparison_id"] for row in effects] == entry[
        "source_paired_comparison_ids"
    ]


def test_a33_3_scope_guards_prevent_silent_generalization(real_payload):
    entry = real_payload["control_dependent_relational_contrasts"][0]

    assert entry["known_scope"] == "game_exact_paired_contexts_target_controls_metric"
    assert entry["scope_game_locked"] is True
    assert entry["scope_contexts_locked"] is True
    assert entry["scope_target_action_locked"] is True
    assert entry["scope_control_actions_locked"] is True
    assert entry["scope_metric_locked"] is True
    assert entry["scope_budgets_locked"] is True
    assert entry["not_generalized_beyond_game"] is True
    assert entry["not_generalized_beyond_exact_paired_contexts"] is True
    assert entry["not_generalized_beyond_recorded_controls"] is True
    assert entry["standalone_action2_effect_excluded"] is True
    assert entry["action1_universal_baseline_excluded"] is True


def test_a33_3_explicitly_excludes_standalone_and_baseline_claims(real_payload):
    excluded = real_payload["excluded_claim_audit"]

    assert len(excluded) == 2
    assert {row["claim"] for row in excluded} == {
        "STANDALONE_UNCONDITIONAL_ACTION2_EFFECT",
        "ACTION1_IS_A_UNIVERSAL_LOWER_EFFECT_BASELINE",
    }
    assert all(row["claim_status"] == "unresolved" for row in excluded)
    assert all(row["claim_support"] == 0 for row in excluded)
    assert all(row["registered"] is False for row in excluded)
    assert all(row["counted_as_refutation"] is False for row in excluded)


def test_a33_3_registers_without_new_scientific_verdict_or_legacy_mutation(
    real_payload,
):
    assert real_payload["status"] == "REGISTERED"
    assert real_payload["truth_status"] == registry.A33_3_TRUTH_STATUS
    assert real_payload["registry_validation_performed"] is True
    assert real_payload["registration_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["a33_write_performed"] is True
    assert real_payload["source_a32_6_mutated"] is False
    assert real_payload["legacy_a33_1_registry_mutated"] is False
    assert real_payload["scoped_a33_2_registry_mutated"] is False
    assert real_payload["support"] == 3
    assert real_payload["support_origin"] == "A32.6_DECISION_RECORDS_ONLY"
    assert real_payload["raw_sage_support_events_imported_directly"] == 0
    assert real_payload["standalone_action2_effect_registered"] is False
    assert real_payload["action1_universal_baseline_registered"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_a33_3_generation_does_not_modify_existing_a33_registries():
    a33_1 = registry.Path("diagnostics/a33/confirmed_mechanics_registry.json")
    a33_2 = registry.Path("diagnostics/a33/scoped_unknown_game_registry.json")
    before = (a33_1.read_bytes(), a33_2.read_bytes())

    registry.run_control_dependent_relational_registry_generation()

    assert (a33_1.read_bytes(), a33_2.read_bytes()) == before


def test_a33_3_rejects_a32_6_prewrite_or_wrong_confirmation(real_source):
    source = copy.deepcopy(real_source)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot pre-write"):
        registry.validate_a32_6_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["wrong_confirmations"] = 1
    with pytest.raises(ValueError, match="wrong_confirmations must remain 0"):
        registry.validate_a32_6_relational_registry_source(source)


def test_a33_3_rejects_standalone_confirmation_or_baseline_generalization(
    real_source,
):
    source = copy.deepcopy(real_source)
    source["standalone_action2_effect_confirmed"] = True
    with pytest.raises(ValueError, match="must remain unresolved"):
        registry.validate_a32_6_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["action1_counted_as_universal_baseline"] = True
    with pytest.raises(ValueError, match="universal baseline"):
        registry.validate_a32_6_relational_registry_source(source)


def test_a33_3_rejects_scope_unlock_or_control_effect_relabelling(real_source):
    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0][
        "not_generalized_beyond_exact_paired_contexts"
    ] = False
    with pytest.raises(ValueError, match="scope must remain locked"):
        registry.validate_a32_6_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["controlled_effects_by_pair"][0][
        "action2_minus_action3"
    ] = 1.0
    with pytest.raises(ValueError, match="32/0 paired contrasts"):
        registry.validate_a32_6_relational_registry_source(source)


def test_a33_3_rejects_context_control_or_budget_scope_changes(real_source):
    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["paired_context_snapshot_hashes"].pop()
    with pytest.raises(ValueError, match="three exact uniques"):
        registry.validate_a32_6_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["control_actions"] = ["ACTION1"]
    with pytest.raises(ValueError, match="controls and metric"):
        registry.validate_a32_6_relational_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["budgets"] = [50, 300]
    with pytest.raises(ValueError, match="three A32.6 budgets"):
        registry.validate_a32_6_relational_registry_source(source)


def test_a33_3_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "a33_3.json"

    registry.write_control_dependent_relational_registry(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload


def test_a33_package_exports_relational_registry_api():
    import theory.a33 as a33

    assert (
        a33.ControlDependentRelationalRegistryEntry
        is registry.ControlDependentRelationalRegistryEntry
    )
    assert (
        a33.DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH
        == registry.DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH
    )
    assert (
        a33.build_control_dependent_relational_registry
        is registry.build_control_dependent_relational_registry
    )
