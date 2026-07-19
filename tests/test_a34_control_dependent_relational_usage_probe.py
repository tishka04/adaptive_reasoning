import copy
import json

import pytest

from theory.a34 import control_dependent_relational_usage_probe as probe


@pytest.fixture(scope="module")
def real_payload():
    return probe.run_control_dependent_relational_usage_probe()


@pytest.fixture(scope="module")
def real_sources():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            probe.DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH,
            probe.DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH,
            probe.DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH,
        )
    )


def test_a34_2_proves_contextual_relational_utility(real_payload):
    summary = real_payload["summary"]

    assert summary["registered_relations_probed"] == 1
    assert summary["exact_contexts_probed"] == 3
    assert summary["action_choices_changed"] == 3
    assert summary["contextual_relational_utility_events"] == 3
    assert summary["functional_local_progress_events"] == 3
    assert summary["registry_gain_over_baseline"] == [32.0] * 3
    assert summary["registry_gain_over_equivalent"] == [0.0] * 3
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == probe.A34_2_RELATION_USEFUL


def test_a34_2_compares_the_exact_three_registered_contexts(real_payload):
    entry = real_payload["registry_entry"]
    contexts = real_payload["replay_contexts"]
    results = real_payload["usage_probes"]

    assert [row["context_snapshot_hash"] for row in contexts] == entry[
        "paired_context_snapshot_hashes"
    ]
    assert [row["context_cluster_id"] for row in contexts] == entry[
        "paired_context_cluster_ids"
    ]
    assert [row["budget"] for row in contexts] == [50, 150, 300]
    assert [row["source_step"] for row in contexts] == [48, 132, 24]
    assert all(row["scope_match"] is True for row in results)
    assert all(row["replay_exact_all_arms"] is True for row in results)


def test_a34_2_memory_changes_action_without_inventing_autonomous_utility(
    real_payload,
):
    results = real_payload["usage_probes"]

    assert {row["baseline_action"] for row in results} == {"ACTION1"}
    assert {row["registry_action"] for row in results} == {"ACTION2"}
    assert {row["equivalent_action"] for row in results} == {"ACTION3"}
    assert [row["baseline_signal"] for row in results] == [1.0, 1.0, 0.0]
    assert [row["registry_signal"] for row in results] == [33.0, 33.0, 32.0]
    assert [row["equivalent_signal"] for row in results] == [33.0, 33.0, 32.0]
    assert all(row["utility_assessment"] == probe.UTILITY_ASSESSMENT for row in results)
    assert all(row["action_choice_changed"] is True for row in results)


def test_a34_2_records_that_local_utility_is_not_arc_progress(real_payload):
    summary = real_payload["summary"]
    results = real_payload["usage_probes"]

    assert summary["baseline_levels_completed_max"] == 0
    assert summary["registry_levels_completed_max"] == 0
    assert summary["registry_levels_completed_delta_total"] == 0
    assert summary["baseline_wins"] == 0
    assert summary["registry_wins"] == 0
    assert summary["registry_win_rate"] == 0.0
    assert summary["level_or_win_progress_demonstrated"] is False
    assert all(row["registry_levels_completed_delta"] == 0 for row in results)
    assert all(row["registry_win"] is False for row in results)


def test_a34_2_uses_sage_only_for_replay_not_action_choice(real_payload):
    assert all(
        row["provenance_used_for_action_choice"] is False
        and row["registry_used_for_action_choice"] is True
        for row in real_payload["replay_contexts"]
    )
    assert real_payload["config"]["inputs_read"] == [
        "A33.3",
        "SAGE.6f_PROVENANCE",
        "SAGE.6a_REPLAY",
    ]
    assert (
        real_payload["config"]["utility_policy"][
            "sage_sources_reconstruct_context_only"
        ]
        is True
    )


def test_a34_2_does_not_reevaluate_truth_or_recount_support(real_payload):
    assert real_payload["truth_status"] == probe.A34_2_TRUTH_STATUS
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


def test_a34_2_rejects_registry_scope_or_autonomous_claim_changes(real_sources):
    registry, sage6f, sage6a = real_sources
    changed = copy.deepcopy(registry)
    changed["control_dependent_relational_contrasts"][0]["game_id"] = "other"
    with pytest.raises(ValueError, match="identity and scope"):
        probe.validate_a34_2_sources(changed, sage6f, sage6a)

    changed = copy.deepcopy(registry)
    changed["control_dependent_relational_contrasts"][0][
        "standalone_action2_effect_status"
    ] = "confirmed"
    with pytest.raises(ValueError, match="scope and exclusions"):
        probe.validate_a34_2_sources(changed, sage6f, sage6a)


def test_a34_2_rejects_non_candidate_provenance_or_missing_replay(real_sources):
    registry, sage6f, sage6a = real_sources
    changed = copy.deepcopy(sage6f)
    changed["support"] = 1
    with pytest.raises(ValueError, match="candidate-only SAGE.6f"):
        probe.validate_a34_2_sources(registry, changed, sage6a)

    changed = copy.deepcopy(sage6a)
    target_id = "m2m3::sage6a::live_mini_frontier::050::0048"
    changed["mini_frontier_m3_requests"] = [
        row
        for row in changed["mini_frontier_m3_requests"]
        if row["request_id"] != target_id
    ]
    with pytest.raises(ValueError, match="replay request is missing"):
        probe.validate_a34_2_sources(registry, sage6f, changed)


def test_a34_2_rejects_replay_context_drift(real_sources):
    registry, sage6f, sage6a = real_sources
    changed = copy.deepcopy(sage6a)
    request = next(
        row
        for row in changed["mini_frontier_m3_requests"]
        if row["request_id"] == "m2m3::sage6a::live_mini_frontier::050::0048"
    )
    request["context_replay"].pop()

    with pytest.raises(ValueError, match="replay provenance"):
        probe.validate_a34_2_sources(registry, sage6f, changed)


def test_a34_2_writer_and_package_exports_round_trip(tmp_path, real_payload):
    path = tmp_path / "a34_2.json"

    probe.write_control_dependent_relational_usage_probe(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload

    import theory.a34 as a34

    assert a34.ControlDependentRelationalUsageProbeResult is (
        probe.ControlDependentRelationalUsageProbeResult
    )
    assert a34.run_control_dependent_relational_usage_probe is (
        probe.run_control_dependent_relational_usage_probe
    )
