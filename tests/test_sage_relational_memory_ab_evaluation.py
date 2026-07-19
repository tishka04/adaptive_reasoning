import copy
import json

import pytest

from theory.sage import relational_memory_ab_evaluation as evaluation


@pytest.fixture(scope="module")
def real_payload():
    return evaluation.run_sage8b_relational_memory_ab_evaluation()


@pytest.fixture(scope="module")
def real_sources():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            evaluation.DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
            evaluation.DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH,
            evaluation.DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
        )
    )


def test_sage8b_evaluates_all_exact_contexts_with_the_integrated_policy(real_payload):
    summary = real_payload["summary"]

    assert summary["paired_episodes_evaluated"] == 11
    assert summary["games_evaluated"] == ["tn36-ab4f63cc", "wa30-ee6fef47"]
    assert summary["memory_policy_applications"] == 11
    assert summary["exact_paired_replays"] == 11
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == evaluation.SAGE8B_LOCAL_ONLY_GAIN


def test_sage8b_uses_levels_completed_as_a_primary_metric(real_payload):
    metric = real_payload["primary_metrics"]["levels_completed"]

    assert metric["no_memory_total_delta"] == 0
    assert metric["with_memory_total_delta"] == 0
    assert metric["absolute_delta_gain"] == 0
    assert metric["no_memory_max_after"] == 0
    assert metric["with_memory_max_after"] == 0
    assert metric["improved"] is False
    assert real_payload["summary"]["levels_completed_improved"] is False


def test_sage8b_uses_win_rate_as_a_primary_metric(real_payload):
    metric = real_payload["primary_metrics"]["win_rate"]

    assert metric["episodes_per_arm"] == 11
    assert metric["no_memory_wins"] == 0
    assert metric["with_memory_wins"] == 0
    assert metric["no_memory"] == 0.0
    assert metric["with_memory"] == 0.0
    assert metric["absolute_gain"] == 0.0
    assert metric["improved"] is False
    assert real_payload["summary"]["win_rate_improved"] is False


def test_sage8b_keeps_local_gain_secondary_and_reports_no_arc_gain(real_payload):
    secondary = real_payload["secondary_metrics"]
    summary = real_payload["summary"]

    assert secondary["classification"] == "SECONDARY_DIAGNOSTIC_ONLY"
    assert secondary["no_memory_total"] == 2.0
    assert secondary["with_memory_total"] == 114.0
    assert secondary["absolute_gain"] == 112.0
    assert secondary["episodes_with_positive_gain"] == 11
    assert secondary["improved"] is True
    assert secondary["counted_as_level_completion"] is False
    assert secondary["counted_as_win"] is False
    assert summary["primary_arc_progress_improved"] is False
    assert summary["local_signal_counted_as_arc_progress"] is False


def test_sage8b_executes_identical_exact_prefixes_in_both_arms(real_payload):
    episodes = real_payload["paired_episodes"]

    assert all(row["same_prefix_between_arms"] is True for row in episodes)
    assert all(row["replay_exact_both_arms"] is True for row in episodes)
    assert all(
        row["no_memory_arm"]["before_signature"]
        == row["with_memory_arm"]["before_signature"]
        == row["context_snapshot_hash"]
        for row in episodes
    )


def test_sage8b_uses_the_policy_decision_for_every_memory_arm(real_payload):
    episodes = real_payload["paired_episodes"]

    assert all(row["relational_memory_consulted"] is True for row in episodes)
    assert all(row["relational_memory_applied"] is True for row in episodes)
    assert all(
        row["policy_target_live_action_selection_passed"] is True for row in episodes
    )
    assert all(
        row["policy_decision"]["decision_reason"]
        == evaluation.LOWER_EFFECT_COMPARATOR_MATCH
        for row in episodes
    )
    assert all(
        row["with_memory_action"] == row["policy_decision"]["selected_action"]
        and row["with_memory_action_args"]
        == row["policy_decision"]["selected_action_args"]
        for row in episodes
    )


def test_sage8b_changes_the_expected_wa30_and_tn36_choices(real_payload):
    episodes = real_payload["paired_episodes"]
    wa30 = [row for row in episodes if row["game_id"] == "wa30-ee6fef47"]
    tn36 = [row for row in episodes if row["game_id"] == "tn36-ab4f63cc"]

    assert len(wa30) == 3
    assert {row["no_memory_action"] for row in wa30} == {"ACTION1"}
    assert {row["with_memory_action"] for row in wa30} == {"ACTION2"}
    assert [row["local_signal_gain"] for row in wa30] == [32.0] * 3

    assert len(tn36) == 8
    assert {row["no_memory_action"] for row in tn36} == {"ACTION6"}
    assert {tuple(row["no_memory_action_args"].values()) for row in tn36} == {(34, 51)}
    assert {tuple(row["with_memory_action_args"].values()) for row in tn36} == {
        (25, 42)
    }
    assert [row["local_signal_gain"] for row in tn36] == [2.0] * 8


def test_sage8b_preserves_technical_replays_without_recounting_them(real_payload):
    specifications = real_payload["evaluation_specifications"]

    assert len(specifications) == 11
    assert sum(row["technical_replay_count"] for row in specifications) == 5
    assert all("request" not in row for row in specifications)
    assert all(
        row["replay_selected_without_outcome_read"] is True for row in specifications
    )


def test_sage8b_does_not_reevaluate_truth_or_count_support(real_payload):
    assert real_payload["truth_status"] == evaluation.SAGE8B_TRUTH_STATUS
    assert real_payload["comparative_evaluation_performed"] is True
    assert real_payload["live_replay_execution_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["support"] == 0
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8b_rejects_incomplete_or_drifted_sources(real_sources):
    policy, a34_2, a34_3 = real_sources
    changed = copy.deepcopy(policy)
    changed["ready_for_comparative_evaluation"] = False
    with pytest.raises(ValueError, match="completed SAGE.8a"):
        evaluation.validate_sage8b_sources(changed, a34_2, a34_3)

    changed = copy.deepcopy(a34_2)
    changed["replay_contexts"].pop()
    with pytest.raises(ValueError, match="scope-locked A34"):
        evaluation.validate_sage8b_sources(policy, changed, a34_3)

    changed = copy.deepcopy(a34_3)
    changed["replay_contexts"][0]["context_snapshot_hash"] = "drifted"
    with pytest.raises(ValueError, match="scopes must match exactly"):
        evaluation.validate_sage8b_sources(policy, a34_2, changed)


def test_sage8b_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8b.json"
    evaluation.write_sage8b_relational_memory_ab_evaluation(real_payload, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8b_relational_memory_ab_evaluation
        is evaluation.run_sage8b_relational_memory_ab_evaluation
    )
    assert (
        sage.write_sage8b_relational_memory_ab_evaluation
        is evaluation.write_sage8b_relational_memory_ab_evaluation
    )
