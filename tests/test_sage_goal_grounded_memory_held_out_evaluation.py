import copy
import json

import pytest

from theory.sage import goal_grounded_memory_held_out_evaluation as evaluation


@pytest.fixture(scope="module")
def real_payload():
    return evaluation.run_sage8i_goal_grounded_memory_held_out_evaluation()


def test_sage8i_evaluates_both_next_levels_without_claiming_a_gain(real_payload):
    summary = real_payload["summary"]

    assert summary["paired_held_out_rollouts_evaluated"] == 2
    assert summary["held_out_levels_evaluated"] == 2
    assert summary["games_evaluated"] == ["tn36-ab4f63cc", "wa30-ee6fef47"]
    assert summary["levels_completed_absolute_gain"] == 0
    assert summary["win_rate_absolute_gain"] == 0.0
    assert summary["held_out_generalization_evaluated"] is True
    assert summary["held_out_generalization_observed"] is False
    assert summary["primary_arc_progress_improved"] is False
    assert summary["primary_arc_progress_regressed"] is False
    assert summary["outcome_status"] == evaluation.SAGE8I_SCOPE_SAFE_NO_GENERALIZATION
    assert summary["gate_passed"] is True


def test_sage8i_builds_bounded_next_level_specifications(real_payload):
    specifications = real_payload["held_out_specifications"]
    by_game = {row["game_id"]: row for row in specifications}

    assert set(by_game) == {"tn36-ab4f63cc", "wa30-ee6fef47"}
    assert by_game["tn36-ab4f63cc"]["setup_action_count"] == 7
    assert by_game["tn36-ab4f63cc"]["held_out_horizon"] == 7
    assert by_game["wa30-ee6fef47"]["setup_action_count"] == 50
    assert by_game["wa30-ee6fef47"]["held_out_horizon"] == 50
    assert all(
        row["expected_setup_level_delta"] == 1
        and row["held_out_next_level"] is True
        and row["setup_excluded_from_primary_metrics"] is True
        and row["structural_generalization_policy_applied"] is False
        for row in specifications
    )


def test_sage8i_setup_reaches_the_same_exact_held_out_start(real_payload):
    for row in real_payload["paired_held_out_rollouts"]:
        no_memory = row["no_memory_arm"]
        with_memory = row["with_memory_arm"]

        assert row["same_held_out_start_between_arms"] is True
        assert row["same_held_out_level_between_arms"] is True
        assert (
            no_memory["held_out_start_visual_digest"]
            == with_memory["held_out_start_visual_digest"]
        )
        assert no_memory["levels_completed_before"] == 1
        assert with_memory["levels_completed_before"] == 1
        assert no_memory["setup_level_delta"] == 1
        assert with_memory["setup_level_delta"] == 1
        assert no_memory["setup_all_actions_legal"] is True
        assert with_memory["setup_all_actions_legal"] is True
        assert row["setup_excluded_from_primary_metrics"] is True


def test_sage8i_uses_the_same_horizon_and_fallback_in_each_pair(real_payload):
    expected_horizons = {"tn36-ab4f63cc": 7, "wa30-ee6fef47": 50}

    for row in real_payload["paired_held_out_rollouts"]:
        horizon = expected_horizons[row["game_id"]]
        assert row["held_out_horizon"] == horizon
        assert row["same_held_out_horizon_between_arms"] is True
        assert row["shared_fallback_planner"] == evaluation.REPLANNING_POLICY
        assert row["no_memory_arm"]["steps_executed"] == horizon
        assert row["with_memory_arm"]["steps_executed"] == horizon


def test_sage8i_exact_memory_is_quarantined_on_every_held_out_state(real_payload):
    for row in real_payload["paired_held_out_rollouts"]:
        arm = row["with_memory_arm"]
        assert row["treatment_exact_memory_quarantined_out_of_scope"] is True
        assert arm["memory_applications"] == 0
        assert arm["memory_scope_misses"] == row["held_out_horizon"]
        assert arm["memory_action_quarantines"] == 0
        assert arm["fallback_applications"] == row["held_out_horizon"]
        assert arm["memory_coverage_rate"] == 0.0
        assert all(
            trace["selection_source"] == "SHARED_STATE_CONDITIONED_FALLBACK"
            and trace["memory_decision"]["memory_applied"] is False
            and trace["memory_decision"]["memory_decision_reason"]
            == "NO_EXACT_GOAL_TRAJECTORY_MEMORY_STATE"
            for trace in arm["trace"]
        )


def test_sage8i_control_never_uses_exact_memory(real_payload):
    for row in real_payload["paired_held_out_rollouts"]:
        arm = row["no_memory_arm"]
        assert row["control_exact_memory_disabled"] is True
        assert arm["memory_enabled"] is False
        assert arm["memory_applications"] == 0
        assert arm["memory_scope_misses"] == 0
        assert arm["fallback_applications"] == row["held_out_horizon"]


def test_sage8i_scope_quarantine_preserves_the_paired_behavior(real_payload):
    for row in real_payload["paired_held_out_rollouts"]:
        comparison = row["comparison"]
        assert comparison["action_trajectories_identical"] is True
        assert comparison["divergent_action_positions"] == 0
        assert comparison["final_visual_digests_identical"] is True
        assert comparison["scope_quarantine_preserved_shared_fallback_behavior"] is True
        assert [
            trace["action_identity"] for trace in row["no_memory_arm"]["trace"]
        ] == [trace["action_identity"] for trace in row["with_memory_arm"]["trace"]]


def test_sage8i_primary_metrics_exclude_setup_progress(real_payload):
    assert real_payload["primary_metrics"] == {
        "primary_metric_order": ["levels_completed", "win_rate"],
        "levels_completed": {
            "no_memory_total_delta": 0,
            "with_memory_total_delta": 0,
            "absolute_delta_gain": 0,
            "no_memory_max_after": 1,
            "with_memory_max_after": 1,
            "improved": False,
        },
        "win_rate": {
            "episodes_per_arm": 2,
            "no_memory_wins": 0,
            "with_memory_wins": 0,
            "no_memory": 0.0,
            "with_memory": 0.0,
            "absolute_gain": 0.0,
            "improved": False,
        },
    }


def test_sage8i_reports_complete_held_out_memory_accounting(real_payload):
    assert real_payload["held_out_metrics"] == {
        "training_memory_entries": 57,
        "held_out_levels_evaluated": 2,
        "held_out_scope_keys_observed": 57,
        "held_out_scope_keys_matching_training_memory": 0,
        "setup_actions_executed_total": 114,
        "no_memory_steps_executed": 57,
        "with_memory_steps_executed": 57,
        "no_memory_applications": 0,
        "with_memory_applications": 0,
        "with_memory_scope_misses": 57,
        "with_memory_action_quarantines": 0,
        "with_memory_fallback_applications": 57,
        "with_memory_exact_coverage_rate": 0.0,
        "episodes_with_identical_action_trajectories": 2,
        "episodes_with_identical_final_states": 2,
    }


def test_sage8i_labels_the_scientific_scope_without_overclaiming(real_payload):
    scope = real_payload["config"]["scientific_scope"]

    assert scope["held_out_generalization_evaluation"] is True
    assert scope["training_trajectory_states_are_excluded_from_treatment_hits"] is True
    assert scope["structural_generalization_policy_applied"] is False
    assert scope["exact_memory_can_only_demonstrate_scope_safe_reuse"] is True
    assert scope["no_gain_means_no_held_out_generalization_observed"] is True
    assert real_payload["held_out_generalization_evaluated"] is True
    assert real_payload["held_out_generalization_observed"] is False
    assert real_payload["exact_memory_quarantined_out_of_scope"] is True
    assert real_payload["structural_generalization_policy_applied"] is False
    assert real_payload["scope_generalization_performed"] is False


def test_sage8i_uses_no_evaluation_outcome_for_learning_or_planning(real_payload):
    design = real_payload["config"]["evaluation_design"]

    assert design["future_outcomes_used_for_action_ranking"] is False
    assert design["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert design["counterfactual_rollouts_performed"] == 0
    assert real_payload["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert real_payload["future_outcomes_used_for_planning"] is False
    assert real_payload["counterfactual_rollouts_performed"] == 0
    assert real_payload["cross_game_transfer_performed"] is False


def test_sage8i_rejects_a_drifted_sage8h_source():
    source = json.loads(
        evaluation.DEFAULT_SAGE8H_GOAL_GROUNDED_RELATIONAL_MEMORY_EVALUATION_PATH.read_text(
            encoding="utf-8"
        )
    )
    changed = copy.deepcopy(source)
    changed["summary"]["held_out_generalization_evaluated"] = True

    with pytest.raises(ValueError, match="exact-replay-gain SAGE.8h"):
        evaluation.validate_sage8i_source(changed)


def test_sage8i_keeps_truth_and_registry_support_untouched(real_payload):
    assert real_payload["truth_status"] == evaluation.SAGE8I_TRUTH_STATUS
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["support"] == 0
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8i_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8i.json"
    evaluation.write_sage8i_goal_grounded_memory_held_out_evaluation(
        real_payload, output_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8i_goal_grounded_memory_held_out_evaluation
        is evaluation.run_sage8i_goal_grounded_memory_held_out_evaluation
    )
    assert (
        sage.write_sage8i_goal_grounded_memory_held_out_evaluation
        is evaluation.write_sage8i_goal_grounded_memory_held_out_evaluation
    )
