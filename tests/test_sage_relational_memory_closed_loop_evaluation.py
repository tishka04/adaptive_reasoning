import copy
import json
from dataclasses import dataclass, field

import pytest

from theory.sage import relational_memory_closed_loop_evaluation as evaluation


@dataclass(frozen=True)
class Action:
    name: str
    action_args: dict = field(default_factory=dict)


@pytest.fixture(scope="module")
def real_payload():
    return evaluation.run_sage8d_relational_memory_closed_loop_evaluation()


@pytest.fixture(scope="module")
def real_sources():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            evaluation.DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH,
            evaluation.DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
            evaluation.DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
            evaluation.DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH,
            evaluation.DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
        )
    )


def test_sage8d_runs_the_closed_loop_on_all_exact_contexts(real_payload):
    summary = real_payload["summary"]

    assert summary["paired_rollouts_evaluated"] == 11
    assert summary["games_evaluated"] == ["tn36-ab4f63cc", "wa30-ee6fef47"]
    assert summary["continuation_horizon"] == 16
    assert summary["memory_policy_applications"] == 11
    assert summary["exact_paired_replays"] == 11
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == evaluation.SAGE8D_LOCAL_ONLY


def test_sage8d_replans_after_every_live_continuation_step(real_payload):
    metrics = real_payload["planner_metrics"]
    episodes = real_payload["paired_rollouts"]

    assert metrics["no_memory_replanning_decisions"] == 171
    assert metrics["with_memory_replanning_decisions"] == 171
    assert metrics["same_planner_algorithm_all_episodes"] is True
    assert metrics["all_actions_legal"] is True
    for row in episodes:
        for arm_name in ("no_memory_arm", "with_memory_arm"):
            arm = row[arm_name]
            assert arm["replanning_decisions"] == arm["continuation_steps_executed"]
            assert len(arm["trace"]) == arm["continuation_steps_executed"] + 1
            assert all(
                step["phase"] == "state_conditioned_replanning"
                for step in arm["trace"][1:]
            )


def test_sage8d_each_decision_uses_the_immediately_current_visual_state(real_payload):
    for row in real_payload["paired_rollouts"]:
        for arm_name in ("no_memory_arm", "with_memory_arm"):
            trace = row[arm_name]["trace"]
            for index, step in enumerate(trace[1:], start=1):
                previous_state = json.loads(trace[index - 1]["signature"])
                planner = step["planner_decision"]
                assert planner["current_visual_digest"] == previous_state["digest"]
                assert planner["current_state_shape"] == previous_state["shape"]
                assert planner["current_state_observation_used"] is True
                assert planner[
                    "selected_action_identity"
                ] == evaluation.action_identity(step["action"], step["action_args"])


def test_sage8d_closed_loop_trajectories_really_diverge(real_payload):
    metrics = real_payload["planner_metrics"]

    assert metrics["episodes_with_divergent_replanned_trajectories"] == 11
    assert metrics["divergent_replanning_positions"] == 119
    assert all(
        row["trajectory_comparison"]["trajectories_diverged"] is True
        and row["trajectory_comparison"]["same_planner_algorithm_used"] is True
        for row in real_payload["paired_rollouts"]
    )


def test_sage8d_planner_never_reads_future_outcomes_or_simulates(real_payload):
    design = real_payload["config"]["evaluation_design"]
    metrics = real_payload["planner_metrics"]

    assert design["planner_excluded_inputs"] == [
        "future_state",
        "future_local_signal",
        "future_levels_completed",
        "future_game_state",
        "future_win",
        "counterfactual_rollout",
    ]
    assert metrics["future_outcome_fields_read_for_planning"] == []
    assert metrics["counterfactual_rollouts_performed"] == 0
    for row in real_payload["paired_rollouts"]:
        assert row["future_action_schedule_preselected"] is False
        assert row["future_outcome_fields_read_for_planning"] == []
        for arm_name in ("no_memory_arm", "with_memory_arm"):
            for step in row[arm_name]["trace"][1:]:
                planner = step["planner_decision"]
                assert planner["future_outcome_fields_read"] == []
                assert planner["counterfactual_rollouts_performed"] == 0


def test_sage8d_primary_levels_completed_remain_unimproved(real_payload):
    metric = real_payload["primary_metrics"]["levels_completed"]

    assert metric["no_memory_total_delta"] == 0
    assert metric["with_memory_total_delta"] == 0
    assert metric["absolute_delta_gain"] == 0
    assert metric["no_memory_max_after"] == 0
    assert metric["with_memory_max_after"] == 0
    assert metric["improved"] is False
    assert real_payload["summary"]["levels_completed_improved"] is False


def test_sage8d_primary_win_rate_remains_unimproved(real_payload):
    metric = real_payload["primary_metrics"]["win_rate"]

    assert metric["episodes_per_arm"] == 11
    assert metric["no_memory_wins"] == 0
    assert metric["with_memory_wins"] == 0
    assert metric["no_memory"] == 0.0
    assert metric["with_memory"] == 0.0
    assert metric["absolute_gain"] == 0.0
    assert metric["improved"] is False
    assert real_payload["summary"]["primary_arc_progress_improved"] is False
    assert real_payload["summary"]["primary_arc_progress_regressed"] is False


def test_sage8d_keeps_initial_local_gain_secondary(real_payload):
    secondary = real_payload["secondary_metrics"]

    assert secondary["no_memory_total"] == 2.0
    assert secondary["with_memory_total"] == 114.0
    assert secondary["absolute_gain"] == 112.0
    assert secondary["episodes_with_positive_gain"] == 11
    assert secondary["improved"] is True
    assert secondary["counted_as_level_completion"] is False
    assert secondary["counted_as_win"] is False
    assert real_payload["summary"]["local_signal_counted_as_arc_progress"] is False


def test_sage8d_stops_the_same_terminal_pair_early(real_payload):
    terminal = [
        row
        for row in real_payload["paired_rollouts"]
        if row["no_memory_terminal"] or row["with_memory_terminal"]
    ]

    assert len(terminal) == 1
    assert terminal[0]["game_id"] == "tn36-ab4f63cc"
    assert terminal[0]["source_step"] == 49
    assert terminal[0]["no_memory_terminal"] is True
    assert terminal[0]["with_memory_terminal"] is True
    assert terminal[0]["no_memory_win"] is False
    assert terminal[0]["with_memory_win"] is False
    assert terminal[0]["no_memory_continuation_steps_executed"] == 11
    assert terminal[0]["with_memory_continuation_steps_executed"] == 11


def test_state_conditioned_planner_prefers_least_used_legal_action():
    actions = [Action("RESET"), Action("ACTION1"), Action("ACTION2")]
    signature = json.dumps(
        {
            "digest": "visual-state-a",
            "shape": [4, 4],
            "levels_completed": 99,
            "game_state": "WIN",
        }
    )

    selected, decision = evaluation.select_state_conditioned_replanning_action(
        actions,
        current_state_signature=signature,
        family_counts={"ACTION1": 5, "ACTION2": 0},
        concrete_counts={"ACTION1::{}": 5, "ACTION2::{}": 0},
        state_action_visits={},
        previous_action_identity="ACTION1::{}",
    )

    assert selected is actions[2]
    assert decision["legal_candidate_count"] == 2
    assert decision["current_visual_digest"] == "visual-state-a"
    assert decision["selected_action_identity"] == "ACTION2::{}"
    assert decision["future_outcome_fields_read"] == []
    assert decision["counterfactual_rollouts_performed"] == 0


def test_state_conditioned_planner_counts_parameter_variants_as_one_family():
    actions = [
        Action("ACTION6", {"x": 1, "y": 2}),
        Action("ACTION6", {"x": 3, "y": 4}),
    ]
    signature = json.dumps({"digest": "visual-state-b", "shape": [4, 4]})

    selected, decision = evaluation.select_state_conditioned_replanning_action(
        actions,
        current_state_signature=signature,
        family_counts={"ACTION6": 7},
        concrete_counts={
            'ACTION6::{"x":1,"y":2}': 4,
            'ACTION6::{"x":3,"y":4}': 0,
        },
        state_action_visits={},
        previous_action_identity='ACTION6::{"x":1,"y":2}',
    )

    assert selected is actions[1]
    assert decision["selected_score"]["action_family_count"] == 7
    assert decision["selected_score"]["concrete_action_count"] == 0


def test_sage8d_rejects_invalid_config_or_drifted_sources(real_sources):
    sage8c, sage8b, policy, a34_2, a34_3 = real_sources
    with pytest.raises(ValueError, match="greater than one"):
        evaluation._validate_closed_loop_config(1)

    changed = copy.deepcopy(sage8c)
    changed["summary"]["continuation_selected_from_outcomes"] = True
    with pytest.raises(ValueError, match="outcome-blind SAGE.8c"):
        evaluation.validate_sage8d_sources(changed, sage8b, policy, a34_2, a34_3)

    changed = copy.deepcopy(sage8c)
    changed["paired_rollouts"][0]["context_snapshot_hash"] = "drifted"
    with pytest.raises(ValueError, match="scopes must match exactly"):
        evaluation.validate_sage8d_sources(changed, sage8b, policy, a34_2, a34_3)


def test_sage8d_does_not_reevaluate_truth_or_count_support(real_payload):
    assert real_payload["truth_status"] == evaluation.SAGE8D_TRUTH_STATUS
    assert real_payload["comparative_evaluation_performed"] is True
    assert real_payload["closed_loop_live_rollout_performed"] is True
    assert real_payload["state_conditioned_replanning_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["support"] == 0
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8d_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8d.json"
    evaluation.write_sage8d_relational_memory_closed_loop_evaluation(
        real_payload, output_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8d_relational_memory_closed_loop_evaluation
        is evaluation.run_sage8d_relational_memory_closed_loop_evaluation
    )
    assert (
        sage.write_sage8d_relational_memory_closed_loop_evaluation
        is evaluation.write_sage8d_relational_memory_closed_loop_evaluation
    )
