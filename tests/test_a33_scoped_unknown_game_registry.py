import copy
import json

import pytest

from theory.a33 import scoped_unknown_game_registry as registry


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        registry.DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return registry.run_scoped_unknown_game_registry_generation()


def test_a33_2_registers_only_one_scope_locked_unknown_game_mechanic(real_payload):
    summary = real_payload["summary"]

    assert summary["source_candidates_consumed"] == 2
    assert summary["a32_5_handoff_candidates_consumed"] == 1
    assert summary["scoped_confirmed_mechanics_registered"] == 1
    assert summary["unresolved_candidates_excluded"] == 1
    assert summary["non_identifiable_candidates_excluded"] == 1
    assert summary["confirmed_support_imported_from_a32_5"] == 4
    assert summary["experiments_spent_total"] == 4
    assert summary["registered_contexts"] == 4
    assert summary["registered_parameterized_control_variants"] == 2
    assert summary["scope_locked_entries"] == 1
    assert summary["outcome_status"] == registry.A33_2_ENTRY_ADDED


def test_a33_2_entry_preserves_the_complete_action5_scope(real_payload):
    assert len(real_payload["scoped_confirmed_mechanics"]) == 1
    entry = real_payload["scoped_confirmed_mechanics"][0]

    assert entry["game_id"] == "sb26-7fbdac44"
    assert entry["action"] == "ACTION5"
    assert entry["action_args"] is None
    assert entry["mechanic_family"] == "local_patch_change_candidate"
    assert entry["predicted_metric"] == "local_patch_before_after"
    assert entry["status"] == "confirmed"
    assert entry["confirmed_support"] == 4
    assert entry["contradictions"] == 0
    assert entry["experiments_spent"] == 4
    assert entry["budgets"] == [50, 300]
    assert len(entry["context_snapshot_hashes"]) == 4
    assert len(entry["source_experiment_ids"]) == 4
    assert [row["action_args"] for row in entry["parameterized_control_variants"]] == [
        {"x": 21, "y": 28},
        {"x": 39, "y": 28},
    ]


def test_a33_2_scope_guards_prevent_silent_generalization(real_payload):
    entry = real_payload["scoped_confirmed_mechanics"][0]

    assert entry["known_scope"] == "game_candidate_contexts_measurement"
    assert entry["scope_game_locked"] is True
    assert entry["scope_candidate_locked"] is True
    assert entry["scope_contexts_locked"] is True
    assert entry["scope_measurement_locked"] is True
    assert entry["not_generalized_beyond_game"] is True
    assert entry["not_generalized_beyond_candidate_scope"] is True
    assert entry["not_generalized_to_other_actions"] is True
    assert entry["a32_confirmation_reused_without_reevaluation"] is True
    assert entry["a33_confirmation_performed"] is False


def test_a33_2_explicitly_excludes_non_identifiable_action6(real_payload):
    assert len(real_payload["excluded_candidate_audit"]) == 1
    excluded = real_payload["excluded_candidate_audit"][0]

    assert excluded["action"] == "ACTION6"
    assert excluded["action_args"] == {"x": 26, "y": 57}
    assert excluded["decision_record_status"] == "unresolved"
    assert excluded["decision_record_support"] == 0
    assert excluded["exclusion_reason"] == (
        "NON_IDENTIFIABLE_UNRESOLVED_NOT_REGISTRY_ELIGIBLE"
    )
    assert excluded["registered"] is False
    assert excluded["counted_as_refutation"] is False


def test_a33_2_registers_without_new_scientific_verdict(real_payload):
    assert real_payload["status"] == "REGISTERED"
    assert real_payload["truth_status"] == registry.A33_2_TRUTH_STATUS
    assert real_payload["registry_validation_performed"] is True
    assert real_payload["registration_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["a33_write_performed"] is True
    assert real_payload["source_a32_5_mutated"] is False
    assert real_payload["legacy_a33_1_registry_mutated"] is False
    assert real_payload["support"] == 4
    assert real_payload["support_origin"] == "A32.5_DECISION_RECORDS_ONLY"
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0


def test_a33_2_rejects_a32_5_prewrite_or_wrong_confirmation(real_source):
    source = copy.deepcopy(real_source)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot pre-write"):
        registry.validate_a32_5_scoped_registry_source(source)

    source = copy.deepcopy(real_source)
    source["wrong_confirmations"] = 1
    with pytest.raises(ValueError, match="wrong_confirmations must remain 0"):
        registry.validate_a32_5_scoped_registry_source(source)


def test_a33_2_rejects_unconfirmed_or_non_ready_handoff(real_source):
    source = copy.deepcopy(real_source)
    action5 = next(
        row for row in source["revision_decisions"] if row["action"] == "ACTION5"
    )
    action5["decision_record"]["status"] = "unresolved"
    with pytest.raises(ValueError, match="only confirmed"):
        registry.validate_a32_5_scoped_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["ready_for_A33_registry_review"] = False
    with pytest.raises(ValueError, match="ready for registry review"):
        registry.validate_a32_5_scoped_registry_source(source)


def test_a33_2_rejects_scope_unlock_or_metric_relabelling(real_source):
    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["not_generalized_beyond_game"] = False
    with pytest.raises(ValueError, match="scope must remain locked"):
        registry.validate_a32_5_scoped_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["measurement"] = "changed_pixels"
    with pytest.raises(ValueError, match="measurement must match"):
        registry.validate_a32_5_scoped_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["action_args"] = {"x": 1, "y": 1}
    with pytest.raises(ValueError, match="identity must match"):
        registry.validate_a32_5_scoped_registry_source(source)


def test_a33_2_rejects_missing_contexts_or_control_diversity(real_source):
    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["context_snapshot_hashes"] = []
    with pytest.raises(ValueError, match="contexts must be non-empty"):
        registry.validate_a32_5_scoped_registry_source(source)

    source = copy.deepcopy(real_source)
    source["a33_handoff_candidates"][0]["parameterized_control_variants"] = [
        source["a33_handoff_candidates"][0]["parameterized_control_variants"][0]
    ]
    with pytest.raises(ValueError, match="at least two control variants"):
        registry.validate_a32_5_scoped_registry_source(source)


def test_a33_2_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "a33_2.json"

    registry.write_scoped_unknown_game_registry(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload


def test_a33_package_exports_scoped_registry_api():
    import theory.a33 as a33

    assert a33.ScopedUnknownGameRegistryEntry is registry.ScopedUnknownGameRegistryEntry
    assert (
        a33.DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH
        == registry.DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH
    )
    assert a33.build_scoped_unknown_game_registry is registry.build_scoped_unknown_game_registry
