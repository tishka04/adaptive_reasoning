import copy
import json

import pytest

import theory.sage as sage
import theory.sage.second_unknown_game_event_consolidation as consolidation


@pytest.fixture(scope="module")
def real_source():
    return json.loads(
        consolidation.DEFAULT_SAGE6B_M3_EXECUTION_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        consolidation.DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_sage6c_builds_six_ordered_event_records(real_payload):
    events = real_payload["event_records"]

    assert len(events) == 6
    assert [row["budget"] for row in events] == [50, 50, 150, 150, 300, 300]
    assert [row["source_step"] for row in events] == [12, 48, 132, 144, 24, 36]
    assert [row["effect_size"] for row in events] == [0.0, 32.0, 32.0, 32.0, 32.0, 32.0]
    assert [row["effect_direction"] for row in events] == [
        "neutral",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
    ]
    assert all(row["live_prefix_replay_exact"] is True for row in events)
    assert all(row["target_context_signature_verified"] is True for row in events)
    assert all(row["control_context_signature_verified"] is True for row in events)


def test_sage6c_preserves_one_cluster_per_live_context(real_payload):
    clusters = real_payload["context_clusters"]

    assert len(clusters) == 6
    assert len({row["context_snapshot_hash"] for row in clusters}) == 6
    assert all(row["events"] == 1 for row in clusters)
    assert all(row["distinct_contexts"] == 1 for row in clusters)
    assert all(row["context_preserved"] is True for row in clusters)
    assert all(row["cross_context_merge_performed"] is False for row in clusters)
    assert clusters[0]["candidate_status"] == (
        consolidation.NEUTRAL_CONTEXT_CANDIDATE_ONLY
    )
    assert all(
        row["candidate_status"] == consolidation.POSITIVE_CONTEXT_CANDIDATE_ONLY
        for row in clusters[1:]
    )


def test_sage6c_builds_comparison_groups_without_merging_evidence(real_payload):
    groups = real_payload["effect_signature_groups"]

    assert len(groups) == 2
    neutral = next(
        row
        for row in groups
        if row["candidate_status"] == consolidation.LOCAL_NEUTRAL_CANDIDATE_ONLY
    )
    stable = next(
        row
        for row in groups
        if row["candidate_status"]
        == consolidation.STABLE_POSITIVE_MULTI_BUDGET_CANDIDATE_ONLY
    )
    assert neutral["events"] == 1
    assert neutral["budgets"] == [50]
    assert neutral["raw_neutral_events"] == 1
    assert stable["events"] == 5
    assert stable["contexts"] == 5
    assert stable["budgets"] == [50, 150, 300]
    assert stable["effect_size_min"] == 32.0
    assert stable["effect_size_max"] == 32.0
    assert stable["effect_size_spread"] == 0.0
    assert stable["stable_positive_multi_budget"] is True
    assert all(row["comparison_index_only"] is True for row in groups)
    assert all(row["cross_context_merge_performed"] is False for row in groups)
    assert all(row["group_counted_as_merged_evidence"] is False for row in groups)


def test_sage6c_detects_stability_and_preserves_neutral_exception(real_payload):
    assessment = real_payload["cross_budget_stability_assessment"]

    assert assessment["candidate_status"] == (
        consolidation.STABLE_POSITIVE_WITH_NEUTRAL_EXCEPTION
    )
    assert assessment["stable_positive_contexts"] == 5
    assert assessment["stable_positive_events"] == 5
    assert assessment["stable_positive_budgets"] == [50, 150, 300]
    assert assessment["stable_positive_effect_size"] == 32.0
    assert assessment["stable_positive_effect_spread"] == 0.0
    assert assessment["stable_positive_across_all_budgets"] is True
    assert assessment["neutral_context_exceptions"] == 1
    assert assessment["negative_context_exceptions"] == 0
    assert assessment["context_sensitive_exception_detected"] is True
    assert assessment["exception_counted_as_refutation"] is False
    assert assessment["exception_context_cluster_ids"] == [
        "sage6c::context_cluster::001"
    ]


def test_sage6c_blocks_a32_review_until_control_diversity(real_payload):
    assessment = real_payload["cross_budget_stability_assessment"]
    frontiers = real_payload["candidate_handoff_frontiers"]

    assert assessment["distinct_control_actions"] == 1
    assert assessment["minimum_distinct_controls_for_a32_review"] == 2
    assert assessment["control_diversity_sufficient_for_a32_review"] is False
    assert assessment["ready_for_A32_handoff_compilation"] is True
    assert assessment["ready_for_A32_review"] is False
    assert assessment["required_followups"] == [
        "ADD_DISTINCT_CONTROL_ACTION_PER_BUDGET",
        "REPLICATE_NEUTRAL_CONTEXT",
        "PRESERVE_CONTEXT_CLUSTER_BOUNDARIES",
    ]
    assert len(frontiers) == 1
    assert frontiers[0]["ready_for_A32_handoff_compilation"] is True
    assert frontiers[0]["ready_for_A32_review"] is False
    assert frontiers[0]["ready_for_A32_review_blocked_reason"] == (
        "INSUFFICIENT_DISTINCT_CONTROL_ACTIONS"
    )


def test_sage6c_passes_all_gates_but_remains_candidate_only(real_payload):
    summary = real_payload["summary"]

    assert all(real_payload["gate"].values())
    assert summary["gate_passed"] is True
    assert summary["event_records"] == 6
    assert summary["context_clusters"] == 6
    assert summary["singleton_context_clusters"] == 6
    assert summary["cross_context_merges_performed"] == 0
    assert summary["all_contexts_preserved_without_merge"] is True
    assert summary["effect_signature_groups"] == 2
    assert summary["stable_positive_multi_budget_groups"] == 1
    assert summary["raw_support_events"] == 5
    assert summary["raw_contradiction_events"] == 0
    assert summary["raw_neutral_events"] == 1
    assert real_payload["outcome_status"] == consolidation.SAGE6C_EVENTS_CONSOLIDATED
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == consolidation.SAGE6C_TRUTH_STATUS
    assert real_payload["support"] == 0
    assert real_payload["stable_pattern_counted_as_scientific_verdict"] is False
    assert real_payload["effect_groups_counted_as_merged_evidence"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_effect_spread_threshold_is_applied_across_positive_contexts():
    events = [
        _synthetic_event("001", 50, "hash-1", 30.0),
        _synthetic_event("002", 150, "hash-2", 32.0),
    ]
    clusters = consolidation.build_context_preserving_clusters(events)

    accepted = consolidation.build_effect_signature_groups(
        events,
        context_clusters=clusters,
        min_stable_contexts=2,
        min_stable_budgets=2,
        max_positive_effect_spread=2.0,
    )
    rejected = consolidation.build_effect_signature_groups(
        events,
        context_clusters=clusters,
        min_stable_contexts=2,
        min_stable_budgets=2,
        max_positive_effect_spread=1.0,
    )

    assert len(accepted) == len(rejected) == 1
    assert accepted[0]["effect_size_spread"] == 2.0
    assert accepted[0]["stable_positive_multi_budget"] is True
    assert rejected[0]["stable_positive_multi_budget"] is False


def test_sage6c_runner_is_pure_and_writes_output(tmp_path):
    output = tmp_path / "sage6c.json"

    payload = consolidation.run_sage6c_second_unknown_game_event_consolidation(
        output_path=output
    )

    assert output.exists()
    assert payload["summary"]["event_records"] == 6
    assert payload["summary"]["stable_positive_contexts"] == 5
    assert payload["summary"]["ready_for_A32_review"] == 0
    assert all(payload["gate"].values())
    assert payload["outcome_status"] == consolidation.SAGE6C_EVENTS_CONSOLIDATED


def test_sage6c_rejects_mutated_scientific_or_replay_state(real_source):
    source = copy.deepcopy(real_source)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        consolidation.validate_sage6c_source(source)

    source = copy.deepcopy(real_source)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        consolidation.validate_sage6c_source(source)

    source = copy.deepcopy(real_source)
    source["scope_generalization_performed"] = True
    with pytest.raises(ValueError, match="cannot generalize"):
        consolidation.validate_sage6c_source(source)

    source = copy.deepcopy(real_source)
    source["support_events_counted_as_support"] = True
    with pytest.raises(ValueError, match="cannot count as evidence"):
        consolidation.validate_sage6c_source(source)

    source = copy.deepcopy(real_source)
    source["controlled_experiments"][0]["live_prefix_replay_exact"] = False
    with pytest.raises(ValueError, match="exact and candidate-only"):
        consolidation.validate_sage6c_source(source)

    source = copy.deepcopy(real_source)
    source["selection_audit"][0]["outcome_metrics_read_for_selection"] = True
    with pytest.raises(ValueError, match="pre-execution"):
        consolidation.validate_sage6c_source(source)


def test_sage6c_writer_round_trips(real_payload, tmp_path):
    output = tmp_path / "round_trip.json"

    consolidation.write_sage6c_second_unknown_game_event_consolidation(
        real_payload, output
    )

    assert json.loads(output.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_sage6c_api():
    assert sage.DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH == (
        consolidation.DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH
    )
    assert (
        sage.run_sage6c_second_unknown_game_event_consolidation
        is consolidation.run_sage6c_second_unknown_game_event_consolidation
    )
    assert (
        sage.write_sage6c_second_unknown_game_event_consolidation
        is consolidation.write_sage6c_second_unknown_game_event_consolidation
    )


def _synthetic_event(event_id, budget, context_hash, effect_size):
    return {
        "event_id": event_id,
        "request_id": f"request-{event_id}",
        "game_id": "fake-wa30",
        "budget": budget,
        "source_step": budget,
        "context_snapshot_hash": context_hash,
        "hypothesis_family": "local_patch_change_candidate",
        "metric": "local_patch_before_after",
        "target_action": "ACTION2",
        "target_action_args": None,
        "control_action": "ACTION1",
        "effect_size": effect_size,
        "effect_direction": "positive",
        "support_events": 1,
        "contradiction_events": 0,
        "neutral_events": 0,
        "live_prefix_replay_exact": True,
        "target_context_signature_verified": True,
        "control_context_signature_verified": True,
    }
