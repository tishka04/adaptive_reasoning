import json

import numpy as np
import pytest

from theory.sage import online_causal_exam_learning as learning


@pytest.fixture(scope="module")
def real_payload():
    return learning.run_sage8j_online_causal_exam_learning()


def test_sage8j_runs_online_learning_on_two_fresh_games(real_payload):
    summary = real_payload["summary"]

    assert summary["fresh_games_evaluated"] == 2
    assert summary["games_evaluated"] == ["lf52-271a04aa", "lp85-305b61c3"]
    assert summary["online_learning_during_exam_performed"] is True
    assert summary["learning_algorithm_frozen_before_exam"] is True
    assert summary["agent_policy_state_frozen_during_exam"] is False
    assert summary["observed_outcomes_used_for_next_action_selection"] is True
    assert summary["future_outcomes_used_for_action_selection"] is False
    assert summary["sage8i_action_traces_loaded"] is False
    assert summary["gate_passed"] is True


def test_sage8j_freezes_the_learning_rule_not_the_policy_state(real_payload):
    design = real_payload["config"]["exam_design"]

    assert design["learning_algorithm_frozen_before_exam"] is True
    assert design["action_schedule_frozen_before_exam"] is False
    assert design["beliefs_update_after_each_observed_candidate_effect"] is True
    assert design["learned_state_graph_updates_during_exam"] is True
    assert design["observed_outcomes_become_legal_next_step_evidence"] is True
    assert design["unexecuted_action_outcomes_available"] is False
    assert design["independent_trial_resets_allowed_and_counted"] is True
    assert design["all_replayed_prefix_actions_counted"] is True


def test_sage8j_uses_fresh_targets_without_loading_sage8i_traces(real_payload):
    freshness = real_payload["config"]["freshness_contract"]

    assert freshness["sage8i_games_excluded"] is True
    assert freshness["sage8i_action_traces_loaded"] is False
    assert freshness["prior_target_action_traces_loaded"] == []
    assert freshness["target_games_selected_from_metadata_only"] is True
    assert freshness["game_source_files_inspected_by_agent"] == []
    assert freshness["protocol_frozen_before_first_target_action"] is True
    assert all(
        row["game_id"] not in {"tn36-ab4f63cc", "wa30-ee6fef47"}
        and row["fresh_relative_to_sage8i"] is True
        for row in real_payload["paired_exam_episodes"]
    )


def test_sage8j_pairs_exact_resets_and_predeclared_bounds(real_payload):
    bounds = real_payload["config"]["bounds_per_game_per_arm"]

    assert bounds == {
        "max_action_executions": 512,
        "max_trials": 256,
        "max_sequence_depth": 4,
        "max_discovered_states": 128,
    }
    for row in real_payload["paired_exam_episodes"]:
        assert row["same_reset_between_arms"] is True
        assert row["same_structural_reset_between_arms"] is True
        assert row["same_action_budget_between_arms"] is True
        assert row["same_trial_budget_between_arms"] is True
        for arm_name in ("control_arm", "adaptive_arm"):
            arm = row[arm_name]
            assert arm["action_executions"] <= 512
            assert arm["trials_executed"] <= 256
            assert arm["max_depth"] == 4
            assert arm["all_replayed_actions_counted"] is True
            assert arm["all_selected_actions_legal"] is True


def test_sage8j_performs_real_online_belief_revision(real_payload):
    metrics = real_payload["online_learning_metrics"]

    assert metrics["adaptive_belief_updates"] == 166
    assert metrics["adaptive_hypothesis_revisions"] == 6
    assert metrics["adaptive_discovered_structural_states"] == 26
    assert metrics["adaptive_duplicate_states_pruned"] == 71
    assert metrics["adaptive_no_effect_states_pruned"] == 0
    assert metrics["adaptive_trials_executed"] == 166
    assert metrics["adaptive_action_executions"] == 514


def test_sage8j_learns_action_effect_hypotheses_from_observations(real_payload):
    by_game = {
        row["game_id"]: row["adaptive_arm"]
        for row in real_payload["paired_exam_episodes"]
    }

    assert set(by_game["lf52-271a04aa"]["causal_action_family_beliefs"]) == {
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
        "ACTION6",
    }
    assert set(by_game["lp85-305b61c3"]["causal_action_family_beliefs"]) == {"ACTION6"}
    for arm in by_game.values():
        assert arm["policy_state_frozen_during_exam"] is False
        assert arm["outcomes_used_to_reorder_or_prune"] is True
        assert arm["belief_updates"] == arm["trials_executed"]
        assert arm["nondeterministic_prefixes"] == 0
        assert all(
            belief["trials"] > 0
            and belief["hypothesis_status"] == "OBSERVED_STRUCTURAL_EFFECT"
            and belief["updated_from_observed_outcomes_only"] is True
            for belief in arm["causal_action_family_beliefs"].values()
        )


def test_sage8j_only_updates_after_observing_each_candidate_effect(real_payload):
    for row in real_payload["paired_exam_episodes"]:
        for experiment in row["adaptive_arm"]["experiments"]:
            transition = experiment["last_transition"]
            selection = experiment["online_selection"]
            revision = experiment["belief_revision"]
            assert transition["outcome_observed_only_after_action"] is True
            assert selection["current_observation_used"] is True
            assert selection["future_outcomes_used"] is False
            assert revision["outcome_was_observed_before_update"] is True


def test_sage8j_control_schedule_never_reorders_from_outcomes(real_payload):
    for row in real_payload["paired_exam_episodes"]:
        control = row["control_arm"]
        assert control["schedule_predeclared_before_outcomes"] is True
        assert control["outcomes_used_to_reorder_or_prune"] is False
        assert control["answer_learned_during_exam"] is False
        assert control["belief_updates"] == 0
        assert control["hypothesis_revisions"] == 0
        assert control["discovered_structural_states"] == 0


def test_sage8j_reports_no_arc_gain_without_overclaiming(real_payload):
    summary = real_payload["summary"]

    assert summary["control_levels_completed_delta_total"] == 0
    assert summary["adaptive_levels_completed_delta_total"] == 0
    assert summary["levels_completed_absolute_gain"] == 0
    assert summary["control_wins"] == 0
    assert summary["adaptive_wins"] == 0
    assert summary["adaptive_answers_learned_during_exam"] == 0
    assert summary["adaptive_learned_answers_exactly_replayed"] == 0
    assert summary["primary_arc_progress_improved"] is False
    assert summary["primary_arc_progress_regressed"] is False
    assert summary["outcome_status"] == learning.SAGE8J_ACTIVE_NO_GAIN


def test_structural_state_excludes_a_volatile_bottom_status_row():
    before = np.zeros((8, 8), dtype=np.int32)
    before[1:3, 1:3] = 2
    after = before.copy()
    after[-1, :] = 7

    assert learning.extract_structural_state(
        before
    ) == learning.extract_structural_state(after)


def test_structural_state_tracks_objects_and_pairwise_relations():
    aligned = np.zeros((12, 12), dtype=np.int32)
    aligned[1:3, 1:3] = 2
    aligned[1:3, 8:10] = 4
    shifted = aligned.copy()
    shifted[1:3, 8:10] = 0
    shifted[8:10, 8:10] = 4

    aligned_state = learning.extract_structural_state(aligned)
    shifted_state = learning.extract_structural_state(shifted)
    assert aligned_state["object_count"] == shifted_state["object_count"] == 2
    assert aligned_state["component_signature"] != shifted_state["component_signature"]
    assert aligned_state["relation_signature"] != shifted_state["relation_signature"]
    assert aligned_state["structural_key"] != shifted_state["structural_key"]


def test_online_belief_moves_from_no_effect_to_context_dependent_to_goal_causal():
    beliefs = {}
    no_effect = {
        "structural_state_changed": False,
        "positive_observed": False,
        "effect_signature": "none",
    }
    effect = {
        "structural_state_changed": True,
        "positive_observed": False,
        "effect_signature": "effect",
    }
    goal = {
        "structural_state_changed": True,
        "positive_observed": True,
        "effect_signature": "goal",
    }

    learning.update_online_causal_belief(
        beliefs, action_family="ACTION1", transition=no_effect
    )
    assert beliefs["ACTION1"]["hypothesis_status"] == "NO_STRUCTURAL_EFFECT_OBSERVED"
    learning.update_online_causal_belief(
        beliefs, action_family="ACTION1", transition=effect
    )
    assert beliefs["ACTION1"]["hypothesis_status"] == "CONTEXT_DEPENDENT_EFFECT"
    revision = learning.update_online_causal_belief(
        beliefs, action_family="ACTION1", transition=goal
    )
    assert beliefs["ACTION1"]["hypothesis_status"] == "OBSERVED_GOAL_CAUSAL"
    assert beliefs["ACTION1"]["positive_observations"] == 1
    assert revision["hypothesis_revised"] is True


def test_online_selection_uses_past_effects_but_never_future_outcomes():
    actions = [
        {"action": "ACTION1", "action_args": {}},
        {"action": "ACTION2", "action_args": {}},
    ]
    beliefs = {
        "ACTION1": {"trials": 3, "effect_rate": 1.0},
        "ACTION2": {"trials": 3, "effect_rate": 0.0},
    }

    selected, decision = learning.select_online_experiment_action(actions, beliefs)
    assert selected["action"] == "ACTION1"
    assert decision["past_observed_effects_used"] is True
    assert decision["future_outcomes_used"] is False


def test_static_schedule_is_fully_declared_without_observations():
    commands = [
        {"action": "ACTION1", "action_args": {}},
        {"action": "ACTION2", "action_args": {}},
    ]
    schedule = list(learning.static_sequence_schedule(commands, max_depth=2))

    assert len(schedule) == 6
    assert [[row["action"] for row in sequence] for sequence in schedule] == [
        ["ACTION1"],
        ["ACTION2"],
        ["ACTION1", "ACTION1"],
        ["ACTION1", "ACTION2"],
        ["ACTION2", "ACTION1"],
        ["ACTION2", "ACTION2"],
    ]


def test_sage8j_rejects_reusing_a_sage8i_target():
    with pytest.raises(ValueError, match="exclude SAGE.8i games"):
        learning.run_sage8j_online_causal_exam_learning(target_games=["tn36-ab4f63cc"])


def test_sage8j_keeps_truth_and_registry_support_untouched(real_payload):
    assert real_payload["truth_status"] == learning.SAGE8J_TRUTH_STATUS
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["revision_performed"] is False
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["support"] == 0
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8j_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8j.json"
    learning.write_sage8j_online_causal_exam_learning(real_payload, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8j_online_causal_exam_learning
        is learning.run_sage8j_online_causal_exam_learning
    )
    assert (
        sage.write_sage8j_online_causal_exam_learning
        is learning.write_sage8j_online_causal_exam_learning
    )
