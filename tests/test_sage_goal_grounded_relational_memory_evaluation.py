import copy
import json

import pytest

from theory.sage import goal_grounded_relational_memory_evaluation as evaluation


@pytest.fixture(scope="module")
def real_payload():
    return evaluation.run_sage8h_goal_grounded_relational_memory_evaluation()


def test_sage8h_converts_exact_goal_memory_into_arc_level_gain(real_payload):
    summary = real_payload["summary"]

    assert summary["paired_rollouts_evaluated"] == 2
    assert summary["games_evaluated"] == [
        "tn36-ab4f63cc",
        "wa30-ee6fef47",
    ]
    assert summary["no_memory_levels_completed_delta_total"] == 0
    assert summary["with_memory_levels_completed_delta_total"] == 2
    assert summary["levels_completed_absolute_gain"] == 2
    assert summary["primary_arc_progress_improved"] is True
    assert summary["primary_arc_progress_regressed"] is False
    assert summary["outcome_status"] == evaluation.SAGE8H_ARC_GAIN
    assert summary["gate_passed"] is True


def test_sage8h_compiles_every_positive_trajectory_transition(real_payload):
    memory = real_payload["goal_grounded_trajectory_memory"]
    summary = memory["summary"]

    assert summary["source_positive_trajectories"] == 2
    assert summary["source_trajectories_exactly_replayed"] == 2
    assert summary["demonstration_transitions"] == 57
    assert summary["exact_visual_states"] == 57
    assert summary["ambiguous_exact_states"] == 0
    assert summary["games"] == ["tn36-ab4f63cc", "wa30-ee6fef47"]
    assert summary["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert summary["gate_passed"] is True
    assert all(memory["gate"].values())


def test_sage8h_memory_entries_are_exact_and_never_cross_game(real_payload):
    entries = real_payload["goal_grounded_trajectory_memory"]["entries"]

    assert len(entries) == 57
    assert sum(row["game_id"] == "tn36-ab4f63cc" for row in entries) == 7
    assert sum(row["game_id"] == "wa30-ee6fef47" for row in entries) == 50
    assert all(
        row["scope_key"] == f"{row['game_id']}|{row['visual_digest']}"
        and row["memory_id"] == evaluation.MEMORY_ID
        and row["scope"] == evaluation.MEMORY_SCOPE
        and len(row["action_candidates"]) == 1
        and row["exact_match_required"] is True
        and row["fuzzy_match_allowed"] is False
        and row["cross_game_transfer_allowed"] is False
        and row["truth_status"] == evaluation.SAGE8H_TRUTH_STATUS
        and row["support"] == 0
        for row in entries
    )


def test_sage8h_runs_same_reset_horizon_and_fallback_in_each_pair(real_payload):
    expected_horizons = {"tn36-ab4f63cc": 7, "wa30-ee6fef47": 50}

    for row in real_payload["paired_rollouts"]:
        assert row["horizon"] == expected_horizons[row["game_id"]]
        assert row["same_reset_between_arms"] is True
        assert row["same_horizon_between_arms"] is True
        assert row["shared_fallback_planner"] == evaluation.REPLANNING_POLICY
        assert row["control_goal_memory_disabled"] is True
        assert row["relational_goal_memory_applied"] is True
        assert row["no_memory_levels_completed_delta"] == 0
        assert row["with_memory_levels_completed_delta"] == 1


def test_sage8h_treatment_replays_both_source_trajectories_exactly(real_payload):
    for row in real_payload["paired_rollouts"]:
        arm = row["with_memory_arm"]
        assert row["treatment_exact_source_sequence_replayed"] is True
        assert row["treatment_source_positive_final_digest_reproduced"] is True
        assert arm["steps_executed"] == row["horizon"]
        assert arm["memory_applications"] == row["horizon"]
        assert arm["fallback_applications"] == 0
        assert arm["memory_scope_misses"] == 0
        assert arm["memory_action_quarantines"] == 0
        assert arm["memory_coverage_rate"] == 1.0
        assert arm["stopped_on_level_increase"] is True
        assert arm["levels_completed_delta"] == 1
        assert all(
            trace["selection_source"] == "EXACT_GOAL_TRAJECTORY_MEMORY"
            and trace["memory_decision"]["memory_decision_reason"]
            == evaluation.MEMORY_APPLIED
            for trace in arm["trace"]
        )


def test_sage8h_control_uses_only_the_shared_closed_loop_fallback(real_payload):
    for row in real_payload["paired_rollouts"]:
        arm = row["no_memory_arm"]
        assert arm["steps_executed"] == row["horizon"]
        assert arm["memory_applications"] == 0
        assert arm["fallback_applications"] == row["horizon"]
        assert arm["levels_completed_delta"] == 0
        assert arm["exact_source_sequence_replayed"] is False
        assert all(
            trace["selection_source"] == "SHARED_STATE_CONDITIONED_FALLBACK"
            and trace["memory_decision"]["memory_decision_reason"]
            == evaluation.MEMORY_DISABLED
            for trace in arm["trace"]
        )


def test_sage8h_primary_metrics_keep_level_gain_separate_from_wins(real_payload):
    primary = real_payload["primary_metrics"]

    assert primary["primary_metric_order"] == ["levels_completed", "win_rate"]
    assert primary["levels_completed"] == {
        "no_memory_total_delta": 0,
        "with_memory_total_delta": 2,
        "absolute_delta_gain": 2,
        "no_memory_max_after": 0,
        "with_memory_max_after": 1,
        "improved": True,
    }
    assert primary["win_rate"] == {
        "episodes_per_arm": 2,
        "no_memory_wins": 0,
        "with_memory_wins": 0,
        "no_memory": 0.0,
        "with_memory": 0.0,
        "absolute_gain": 0.0,
        "improved": False,
    }


def test_sage8h_reports_full_memory_accounting(real_payload):
    metrics = real_payload["memory_metrics"]
    summary = real_payload["summary"]

    assert metrics["compiled_demonstration_transitions"] == 57
    assert metrics["compiled_exact_visual_states"] == 57
    assert metrics["no_memory_steps_executed"] == 57
    assert metrics["with_memory_steps_executed"] == 57
    assert metrics["no_memory_applications"] == 0
    assert metrics["with_memory_applications"] == 57
    assert metrics["with_memory_fallback_applications"] == 0
    assert metrics["with_memory_exact_coverage_rate"] == 1.0
    assert metrics["exact_source_sequences_replayed"] == 2
    assert metrics["source_positive_final_digests_reproduced"] == 2
    assert summary["episodes_with_divergent_action_trajectories"] == 2


def test_sage8h_does_not_mislabel_replay_conversion_as_generalization(real_payload):
    scope = real_payload["config"]["scientific_scope"]

    assert scope["evaluation_is_exact_training_trajectory_replay"] is True
    assert scope["held_out_generalization_evaluation"] is False
    assert scope["primary_gain_is_replay_conversion_not_generalization"] is True
    assert scope["unseen_level_claim_allowed"] is False
    assert real_payload["exact_replay_conversion_evaluated"] is True
    assert real_payload["held_out_generalization_evaluated"] is False
    assert real_payload["summary"]["held_out_generalization_evaluated"] is False
    assert (
        real_payload["summary"]["primary_gain_is_replay_conversion_not_generalization"]
        is True
    )


def test_sage8h_uses_no_evaluation_outcomes_for_training_tuning_or_planning(
    real_payload,
):
    design = real_payload["config"]["evaluation_design"]
    memory_config = real_payload["goal_grounded_trajectory_memory"]["config"]

    assert design["future_outcomes_used_for_action_ranking"] is False
    assert design["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert memory_config["evaluation_outcome_fields_read"] == []
    assert memory_config["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert real_payload["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert real_payload["future_outcomes_used_for_planning"] is False
    assert real_payload["counterfactual_rollouts_performed"] == 0
    assert real_payload["cross_game_transfer_performed"] is False
    assert real_payload["scope_generalization_performed"] is False


def test_exact_memory_selector_respects_scope_and_live_legality():
    action = FakeAction("ACTION5", {})
    entry = {
        "scope_key": "game-a|digest-a",
        "action_candidates": [
            {
                "action": "ACTION5",
                "action_args": {},
                "action_identity": "ACTION5::{}",
                "minimum_distance_to_observed_positive": 1,
            }
        ],
    }
    index = {"game-a|digest-a": entry}

    selected, decision = evaluation.select_exact_goal_memory_action(
        [action],
        game_id="game-a",
        current_visual_digest="digest-a",
        memory_index=index,
        memory_enabled=True,
    )
    assert selected is action
    assert decision["memory_decision_reason"] == evaluation.MEMORY_APPLIED

    selected, decision = evaluation.select_exact_goal_memory_action(
        [action],
        game_id="game-b",
        current_visual_digest="digest-a",
        memory_index=index,
        memory_enabled=True,
    )
    assert selected is None
    assert decision["memory_decision_reason"] == evaluation.MEMORY_OUT_OF_SCOPE
    assert decision["cross_game_transfer_performed"] is False

    selected, decision = evaluation.select_exact_goal_memory_action(
        [FakeAction("ACTION1", {})],
        game_id="game-a",
        current_visual_digest="digest-a",
        memory_index=index,
        memory_enabled=True,
    )
    assert selected is None
    assert decision["memory_decision_reason"] == evaluation.MEMORY_ACTION_UNAVAILABLE


def test_sage8h_rejects_a_drifted_sage8g_source():
    source = json.loads(
        evaluation.DEFAULT_SAGE8G_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_PATH.read_text(
            encoding="utf-8"
        )
    )
    changed = copy.deepcopy(source)
    changed["held_out_generalization_evaluated"] = True
    changed["planner_activation_authorized"] = False

    with pytest.raises(ValueError, match="exact-target-admitted SAGE.8g"):
        evaluation.validate_sage8h_source(changed)


def test_sage8h_keeps_truth_and_registry_support_untouched(real_payload):
    assert real_payload["truth_status"] == evaluation.SAGE8H_TRUTH_STATUS
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["support"] == 0
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8h_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8h.json"
    evaluation.write_sage8h_goal_grounded_relational_memory_evaluation(
        real_payload, output_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8h_goal_grounded_relational_memory_evaluation
        is evaluation.run_sage8h_goal_grounded_relational_memory_evaluation
    )
    assert (
        sage.write_sage8h_goal_grounded_relational_memory_evaluation
        is evaluation.write_sage8h_goal_grounded_relational_memory_evaluation
    )


class FakeAction:
    def __init__(self, name, action_args):
        self.name = name
        self.action_args = action_args
