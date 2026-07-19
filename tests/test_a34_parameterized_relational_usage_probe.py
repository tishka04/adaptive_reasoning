import copy
import json

import pytest

from theory.a34 import parameterized_relational_usage_probe as probe


@pytest.fixture(scope="module")
def real_payload():
    return probe.run_parameterized_relational_usage_probe()


@pytest.fixture(scope="module")
def real_sources():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            probe.DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH,
            probe.DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH,
        )
    )


def test_a34_3_proves_contextual_parameterized_relational_utility(real_payload):
    summary = real_payload["summary"]

    assert summary["registered_relations_probed"] == 1
    assert summary["exact_contexts_probed"] == 8
    assert summary["technical_replay_requests_preserved"] == 5
    assert summary["parameter_choices_changed"] == 8
    assert summary["contextual_relational_utility_events"] == 8
    assert summary["functional_local_progress_events"] == 8
    assert summary["registry_gain_over_baseline"] == [2.0] * 8
    assert summary["registry_gain_over_equivalent"] == [0.0] * 8
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == probe.A34_3_RELATION_USEFUL


def test_a34_3_compares_the_exact_eight_registered_contexts(real_payload):
    entry = real_payload["registry_entry"]
    contexts = real_payload["replay_contexts"]
    results = real_payload["usage_probes"]

    assert [row["context_snapshot_hash"] for row in contexts] == entry[
        "context_snapshot_hashes"
    ]
    assert [row["budget"] for row in contexts] == [50, 50, 150, 50, 50, 150, 50, 150]
    assert [row["source_step"] for row in contexts] == [33, 25, 29, 49, 41, 39, 17, 19]
    assert sum(row["technical_replay_count"] for row in contexts) == 5
    assert all(row["scope_match"] is True for row in results)
    assert all(row["replay_exact_all_arms"] is True for row in results)


def test_a34_3_memory_changes_parameters_without_inventing_autonomous_utility(
    real_payload,
):
    results = real_payload["usage_probes"]

    assert {row["action_family"] for row in results} == {"ACTION6"}
    assert {tuple(row["baseline_action_args"].values()) for row in results} == {
        (34, 51)
    }
    assert {tuple(row["registry_action_args"].values()) for row in results} == {
        (25, 42)
    }
    assert {tuple(row["equivalent_action_args"].values()) for row in results} == {
        (41, 44)
    }
    assert [row["baseline_signal"] for row in results] == [0.0] * 8
    assert [row["registry_signal"] for row in results] == [2.0] * 8
    assert [row["equivalent_signal"] for row in results] == [2.0] * 8
    assert all(row["utility_assessment"] == probe.UTILITY_ASSESSMENT for row in results)
    assert all(row["parameter_choice_changed"] is True for row in results)


def test_a34_3_keeps_parameter_variants_as_one_action_family(real_payload):
    summary = real_payload["summary"]

    assert summary["action_families"] == ["ACTION6"]
    assert summary["distinct_action_families"] == 1
    assert summary["parameterized_variants_counted_as_distinct_actions"] is False
    assert real_payload["parameterized_variants_counted_as_distinct_actions"] is False
    assert all(
        row["parameterized_variants_counted_as_distinct_actions"] is False
        for row in real_payload["usage_probes"]
    )


def test_a34_3_records_that_local_utility_is_not_arc_progress(real_payload):
    summary = real_payload["summary"]

    assert summary["baseline_levels_completed_max"] == 0
    assert summary["registry_levels_completed_max"] == 0
    assert summary["registry_levels_completed_delta_total"] == 0
    assert summary["baseline_wins"] == 0
    assert summary["registry_wins"] == 0
    assert summary["registry_win_rate"] == 0.0
    assert summary["level_or_win_progress_demonstrated"] is False
    assert all(
        row["registry_levels_completed_delta"] == 0 and row["registry_win"] is False
        for row in real_payload["usage_probes"]
    )


def test_a34_3_uses_sage_only_for_replay_not_parameter_choice(real_payload):
    assert all(
        row["canonical_request_selected_without_outcome_read"] is True
        and row["provenance_used_for_parameter_choice"] is False
        and row["registry_used_for_parameter_choice"] is True
        for row in real_payload["replay_contexts"]
    )
    assert real_payload["config"]["inputs_read"] == ["A33.4", "SAGE.7a_REPLAY"]
    assert (
        real_payload["config"]["utility_policy"][
            "sage_source_reconstructs_context_only"
        ]
        is True
    )


def test_a34_3_does_not_reevaluate_truth_or_recount_support(real_payload):
    assert real_payload["truth_status"] == probe.A34_3_TRUTH_STATUS
    assert real_payload["utility_evaluation_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["support"] == 0
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_a34_3_rejects_registry_parameter_or_autonomous_claim_changes(real_sources):
    registry, sage7a = real_sources
    changed = copy.deepcopy(registry)
    changed["parameterized_relational_contrasts"][0]["target_action_args"]["x"] = 26
    with pytest.raises(ValueError, match="identity and scope"):
        probe.validate_a34_3_sources(changed, sage7a)

    changed = copy.deepcopy(registry)
    changed["parameterized_relational_contrasts"][0][
        "autonomous_target_effect_status"
    ] = "confirmed"
    with pytest.raises(ValueError, match="scope and exclusions"):
        probe.validate_a34_3_sources(changed, sage7a)


def test_a34_3_rejects_non_candidate_or_distinct_action_provenance(real_sources):
    registry, sage7a = real_sources
    changed = copy.deepcopy(sage7a)
    changed["support"] = 1
    with pytest.raises(ValueError, match="candidate-only SAGE.7a"):
        probe.validate_a34_3_sources(registry, changed)

    changed = copy.deepcopy(sage7a)
    changed["parameterized_controls_counted_as_distinct_actions"] = True
    with pytest.raises(ValueError, match="candidate-only SAGE.7a"):
        probe.validate_a34_3_sources(registry, changed)


def test_a34_3_rejects_missing_or_drifted_replay(real_sources):
    registry, sage7a = real_sources
    changed = copy.deepcopy(sage7a)
    changed["mini_frontier_m3_requests"] = [
        row
        for row in changed["mini_frontier_m3_requests"]
        if row["context_snapshot_hash"]
        != registry["parameterized_relational_contrasts"][0]["context_snapshot_hashes"][
            0
        ]
    ]
    with pytest.raises(ValueError, match="missing from SAGE.7a"):
        probe.validate_a34_3_sources(registry, changed)

    changed = copy.deepcopy(sage7a)
    request = next(
        row
        for row in changed["mini_frontier_m3_requests"]
        if row["request_id"] == "m2m3::sage7a::live_mini_frontier::050::0033"
    )
    request["context_replay"].pop()
    with pytest.raises(ValueError, match="replay provenance"):
        probe.validate_a34_3_sources(registry, changed)


def test_a34_3_writer_and_package_exports_round_trip(tmp_path, real_payload):
    path = tmp_path / "a34_3.json"

    probe.write_parameterized_relational_usage_probe(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload

    import theory.a34 as a34

    assert a34.ParameterizedRelationalUsageProbeResult is (
        probe.ParameterizedRelationalUsageProbeResult
    )
    assert a34.run_parameterized_relational_usage_probe is (
        probe.run_parameterized_relational_usage_probe
    )
