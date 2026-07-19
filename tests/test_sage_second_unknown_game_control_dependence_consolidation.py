import copy
import json

import pytest

import theory.sage as sage
from theory.sage import (
    second_unknown_game_control_dependence_consolidation as consolidation,
)


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        consolidation.DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_sage6f_real_artifact_consolidates_both_sources(real_payload):
    summary = real_payload["summary"]

    assert summary["game_id"] == "wa30-ee6fef47"
    assert summary["budgets"] == [50, 150, 300]
    assert summary["observation_records"] == 10
    assert summary["source_sage6c_observations"] == 6
    assert summary["source_sage6e_observations"] == 4
    assert summary["context_clusters"] == 6
    assert summary["context_clusters_preserved"] == 6
    assert summary["cross_context_merges_performed"] == 0
    assert summary["control_actions"] == ["ACTION1", "ACTION3"]
    assert summary["distinct_control_actions"] == 2
    assert summary["outcome_status"] == consolidation.SAGE6F_A32_REVIEW_ELIGIBLE


def test_sage6f_observations_preserve_source_control_and_context(real_payload):
    observations = real_payload["observation_records"]

    assert len(observations) == 10
    assert len({row["observation_id"] for row in observations}) == 10
    assert [row["source_context_cluster_id"] for row in observations] == [
        "sage6c::context_cluster::001",
        "sage6c::context_cluster::001",
        "sage6c::context_cluster::002",
        "sage6c::context_cluster::002",
        "sage6c::context_cluster::003",
        "sage6c::context_cluster::003",
        "sage6c::context_cluster::004",
        "sage6c::context_cluster::005",
        "sage6c::context_cluster::005",
        "sage6c::context_cluster::006",
    ]
    assert sum(row["source_stage"] == "SAGE.6c" for row in observations) == 6
    assert sum(row["source_stage"] == "SAGE.6e" for row in observations) == 4
    assert sum(row["control_action"] == "ACTION1" for row in observations) == 7
    assert sum(row["control_action"] == "ACTION3" for row in observations) == 3
    assert all(row["support"] == 0 for row in observations)
    assert all(
        row["truth_status"] == consolidation.SAGE6F_TRUTH_STATUS for row in observations
    )
    assert not any(
        row["raw_event_counted_as_scientific_support"] for row in observations
    )


def test_sage6f_keeps_six_context_clusters_and_one_exact_replication(real_payload):
    clusters = real_payload["context_cluster_manifest"]

    assert len(clusters) == 6
    assert all(row["context_preserved"] for row in clusters)
    assert not any(row["cross_context_merge_performed"] for row in clusters)
    assert not any(row["cross_control_effect_merge_performed"] for row in clusters)

    neutral = clusters[0]
    assert neutral["context_cluster_id"] == "sage6c::context_cluster::001"
    assert neutral["control_actions"] == ["ACTION1"]
    assert neutral["effects_by_control"] == {"ACTION1": [0.0, 0.0]}
    assert neutral["observations"] == 2
    assert neutral["independent_contexts"] == 1
    assert neutral["same_context_replications"] == 1
    assert neutral["neutral_context_replication_verified"] is True
    assert neutral["candidate_status"] == (
        consolidation.ACTION1_REPLICATED_NEUTRAL_CONTEXT
    )

    paired = [row for row in clusters if row["paired_control_context"]]
    assert [row["context_cluster_id"] for row in paired] == [
        "sage6c::context_cluster::002",
        "sage6c::context_cluster::003",
        "sage6c::context_cluster::005",
    ]
    assert all(
        row["effects_by_control"] == {"ACTION1": [32.0], "ACTION3": [0.0]}
        for row in paired
    )


def test_sage6f_builds_three_control_conditioned_comparison_indexes(real_payload):
    groups = real_payload["control_conditioned_effect_groups"]

    assert [(row["control_action"], row["effect_direction"]) for row in groups] == [
        ("ACTION1", "positive"),
        ("ACTION1", "neutral"),
        ("ACTION3", "neutral"),
    ]
    assert [row["observations"] for row in groups] == [5, 2, 3]
    assert [row["independent_contexts"] for row in groups] == [5, 1, 3]
    assert [row["budgets"] for row in groups] == [
        [50, 150, 300],
        [50],
        [50, 150, 300],
    ]
    assert groups[0]["effect_sizes"] == [32.0] * 5
    assert groups[1]["effect_sizes"] == [0.0, 0.0]
    assert groups[2]["effect_sizes"] == [0.0, 0.0, 0.0]
    assert all(row["comparison_index_only"] for row in groups)
    assert not any(row["group_counted_as_scientific_evidence"] for row in groups)


def test_sage6f_pairs_controls_once_per_budget(real_payload):
    paired = real_payload["paired_control_comparisons"]

    assert len(paired) == 3
    assert [row["budget"] for row in paired] == [50, 150, 300]
    assert [row["source_step"] for row in paired] == [48, 132, 24]
    assert [row["action1_controlled_effect_size"] for row in paired] == [32.0] * 3
    assert [row["action3_controlled_effect_size"] for row in paired] == [0.0] * 3
    assert [row["controlled_effect_gap_action1_minus_action3"] for row in paired] == [
        32.0
    ] * 3
    assert [row["control_signal_gap_action3_minus_action1"] for row in paired] == [
        32.0
    ] * 3
    assert all(row["target_signal_reproduced_across_control_pairs"] for row in paired)
    assert all(row["context_preserved"] for row in paired)
    assert all(row["control_identities_preserved"] for row in paired)
    assert not any(
        row["paired_comparison_counted_as_scientific_verdict"] for row in paired
    )


def test_sage6f_reformulates_candidate_and_marks_a32_review_eligible(real_payload):
    assessment = real_payload["a32_review_eligibility_assessment"]

    assert assessment["candidate_key"] == (
        "control_dependent_local_patch::wa30-ee6fef47::ACTION2::ACTION1_vs_ACTION3"
    )
    assert assessment["candidate_mechanism_family"] == (
        "control_dependent_local_patch_change_candidate"
    )
    assert (
        assessment["candidate_status"] == consolidation.CONTROL_DEPENDENT_MULTI_BUDGET
    )
    assert assessment["distinct_control_actions"] == 2
    assert assessment["paired_control_contexts"] == 3
    assert assessment["paired_control_budgets"] == [50, 150, 300]
    assert assessment["paired_control_effect_gaps"] == [32.0, 32.0, 32.0]
    assert assessment["paired_control_effect_gap_spread"] == 0.0
    assert assessment["action1_positive_contexts"] == 5
    assert assessment["action1_neutral_observations"] == 2
    assert assessment["action3_neutral_contexts"] == 3
    assert assessment["replicated_neutral_contexts"] == 1
    assert assessment["negative_effect_events"] == 0
    assert all(assessment["eligibility_requirements"].values())
    assert len(assessment["eligibility_requirements"]) == 14
    assert assessment["missing_eligibility_requirements"] == []
    assert assessment["ready_for_A32_review"] is True
    assert assessment["ready_for_A32_review_is_not_verdict"] is True
    assert assessment["ready_for_A32_review_is_not_confirmation"] is True
    assert assessment["a32_review_recommendation"] == (
        consolidation.READY_FOR_A32_CONTROL_DEPENDENCE_REVIEW
    )
    assert assessment["recommended_a32_review_scope"]["must_not_review_as"] == (
        "STANDALONE_UNCONDITIONAL_ACTION2_EFFECT"
    )


def test_sage6f_emits_one_nonverdict_a32_frontier(real_payload):
    frontiers = real_payload["candidate_a32_review_frontiers"]

    assert len(frontiers) == 1
    frontier = frontiers[0]
    assert frontier["frontier_id"] == (
        "sage6f::candidate_a32_control_dependence_frontier::001"
    )
    assert frontier["control_actions"] == ["ACTION1", "ACTION3"]
    assert len(frontier["context_cluster_ids"]) == 6
    assert len(frontier["paired_control_comparison_ids"]) == 3
    assert len(frontier["control_conditioned_effect_group_ids"]) == 3
    assert len(frontier["a32_review_questions"]) == 3
    assert frontier["missing_eligibility_requirements"] == []
    assert frontier["ready_for_A32_review"] is True
    assert frontier["ready_for_A32_review_is_not_verdict"] is True
    assert frontier["ready_for_A32_review_is_not_confirmation"] is True
    assert frontier["a32_intake_requested"] is False
    assert frontier["support"] == 0


def test_sage6f_raw_accounting_and_quarantine_remain_explicit(real_payload):
    summary = real_payload["summary"]

    assert summary["raw_support_events"] == 5
    assert summary["raw_contradiction_events"] == 0
    assert summary["raw_neutral_events"] == 5
    assert summary["eligibility_requirements_passed"] == 14
    assert summary["eligibility_requirements_total"] == 14
    assert summary["missing_eligibility_requirements"] == []
    assert summary["candidate_a32_review_frontiers"] == 1
    assert summary["ready_for_A32_review"] == 1
    assert summary["ready_for_A32_review_is_not_verdict"] is True
    assert summary["a32_intake_requested"] is False

    assert all(real_payload["gate"].values())
    assert summary["gate_passed"] is True
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["truth_status"] == consolidation.SAGE6F_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["execution_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["a32_review_eligibility_counted_as_a32_decision"] is False
    assert real_payload["a32_intake_requested"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False


def test_sage6f_ineligible_assessment_produces_no_frontier():
    source6c = _source6c()
    source6e = _source6e()
    observations = consolidation.build_sage6f_observations(source6c, source6e)
    action3 = next(row for row in observations if row["control_action"] == "ACTION3")
    action3["effect_size"] = -1.0
    action3["effect_direction"] = "negative"
    clusters = consolidation.build_sage6f_context_clusters(
        source_sage6c=source6c,
        observations=observations,
    )
    groups = consolidation.build_control_conditioned_effect_groups(observations)
    paired = consolidation.build_paired_control_comparisons(
        observations=observations,
        context_clusters=clusters,
    )
    assessment = consolidation.assess_a32_review_eligibility(
        source_sage6c=source6c,
        source_sage6e=source6e,
        observations=observations,
        context_clusters=clusters,
        control_groups=groups,
        paired=paired,
    )

    assert assessment["ready_for_A32_review"] is False
    assert "no_negative_effect_events" in assessment["missing_eligibility_requirements"]
    assert (
        consolidation.build_sage6f_a32_review_frontiers(
            assessment=assessment,
            context_clusters=clusters,
            control_groups=groups,
            paired=paired,
        )
        == []
    )


@pytest.mark.parametrize(
    ("source_name", "mutation", "message"),
    [
        ("sage6c", lambda source: source.__setitem__("support", 1), "support 0"),
        (
            "sage6c",
            lambda source: source.__setitem__("a32_write_performed", True),
            "verdict or registry writes",
        ),
        (
            "sage6c",
            lambda source: source["context_clusters"][0].__setitem__(
                "cross_context_merge_performed", True
            ),
            "singleton and unmerged",
        ),
        (
            "sage6c",
            lambda source: source["event_records"][0].__setitem__(
                "control_action", "ACTION3"
            ),
            "ACTION1 observations",
        ),
        (
            "sage6c",
            lambda source: source["summary"].__setitem__("raw_neutral_events", 2),
            "raw event accounting",
        ),
        ("sage6e", lambda source: source.__setitem__("support", 1), "support 0"),
        (
            "sage6e",
            lambda source: source.__setitem__("a33_write_performed", True),
            "verdict or registry writes",
        ),
        (
            "sage6e",
            lambda source: source["controlled_followup_results"][0].__setitem__(
                "control_action_substitution_performed", True
            ),
            "exact and unsubstituted",
        ),
        (
            "sage6e",
            lambda source: source["controlled_followup_results"][0].__setitem__(
                "control_action", "ACTION4"
            ),
            "accounting and control identities",
        ),
    ],
)
def test_sage6f_rejects_invalid_source_state(tmp_path, source_name, mutation, message):
    source6c = _source6c()
    source6e = _source6e()
    source = source6c if source_name == "sage6c" else source6e
    mutation(source)
    source6c_path = tmp_path / "sage6c.json"
    source6e_path = tmp_path / "sage6e.json"
    source6c_path.write_text(json.dumps(source6c), encoding="utf-8")
    source6e_path.write_text(json.dumps(source6e), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        consolidation.run_sage6f_second_unknown_game_control_dependence_consolidation(
            source_sage6c_path=source6c_path,
            source_sage6e_path=source6e_path,
        )


def test_sage6f_rejects_cross_source_context_drift(tmp_path):
    source6e = _source6e()
    source6e["controlled_followup_results"][0]["context_snapshot_hash"] = "drifted"
    source6e_path = tmp_path / "drifted_sage6e.json"
    source6e_path.write_text(json.dumps(source6e), encoding="utf-8")

    with pytest.raises(ValueError, match="contexts must align exactly"):
        consolidation.run_sage6f_second_unknown_game_control_dependence_consolidation(
            source_sage6e_path=source6e_path
        )


def test_sage6f_runner_is_pure_without_output_and_does_not_mutate_sources(tmp_path):
    source6c = _source6c()
    source6e = _source6e()
    before6c = copy.deepcopy(source6c)
    before6e = copy.deepcopy(source6e)
    source6c_path = tmp_path / "sage6c.json"
    source6e_path = tmp_path / "sage6e.json"
    source6c_path.write_text(json.dumps(source6c), encoding="utf-8")
    source6e_path.write_text(json.dumps(source6e), encoding="utf-8")

    consolidation.run_sage6f_second_unknown_game_control_dependence_consolidation(
        source_sage6c_path=source6c_path,
        source_sage6e_path=source6e_path,
    )

    assert json.loads(source6c_path.read_text(encoding="utf-8")) == before6c
    assert json.loads(source6e_path.read_text(encoding="utf-8")) == before6e
    assert set(tmp_path.iterdir()) == {source6c_path, source6e_path}


def test_sage6f_writer_and_package_exports(real_payload, tmp_path):
    output_path = tmp_path / "nested" / "sage6f.json"

    consolidation.write_sage6f_second_unknown_game_control_dependence_consolidation(
        real_payload, output_path
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload
    assert sage.DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH == (
        consolidation.DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH
    )
    assert (
        sage.run_sage6f_second_unknown_game_control_dependence_consolidation
        is consolidation.run_sage6f_second_unknown_game_control_dependence_consolidation
    )
    assert (
        sage.write_sage6f_second_unknown_game_control_dependence_consolidation
        is consolidation.write_sage6f_second_unknown_game_control_dependence_consolidation
    )


def _source6c():
    return json.loads(
        consolidation.DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH.read_text(
            encoding="utf-8"
        )
    )


def _source6e():
    return json.loads(
        consolidation.DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH.read_text(encoding="utf-8")
    )
