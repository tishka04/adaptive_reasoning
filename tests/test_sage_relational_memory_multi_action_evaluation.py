import copy
import json

import pytest

from theory.sage import relational_memory_multi_action_evaluation as evaluation


@pytest.fixture(scope="module")
def real_payload():
    return evaluation.run_sage8c_relational_memory_multi_action_evaluation()


@pytest.fixture(scope="module")
def real_sources():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            evaluation.DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
            evaluation.DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
            evaluation.DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH,
            evaluation.DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
        )
    )


def test_sage8c_runs_all_exact_contexts_at_fixed_multi_action_horizon(real_payload):
    summary = real_payload["summary"]

    assert summary["paired_rollouts_evaluated"] == 11
    assert summary["games_evaluated"] == ["tn36-ab4f63cc", "wa30-ee6fef47"]
    assert summary["continuation_horizon"] == 16
    assert summary["memory_policy_applications"] == 11
    assert summary["exact_paired_replays"] == 11
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == evaluation.SAGE8C_LOCAL_ONLY


def test_sage8c_primary_levels_completed_remain_unimproved(real_payload):
    metric = real_payload["primary_metrics"]["levels_completed"]

    assert metric["no_memory_total_delta"] == 0
    assert metric["with_memory_total_delta"] == 0
    assert metric["absolute_delta_gain"] == 0
    assert metric["no_memory_max_after"] == 0
    assert metric["with_memory_max_after"] == 0
    assert metric["improved"] is False
    assert real_payload["summary"]["levels_completed_improved"] is False


def test_sage8c_primary_win_rate_remains_unimproved(real_payload):
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


def test_sage8c_executes_the_predeclared_continuation_legally(real_payload):
    metrics = real_payload["rollout_metrics"]

    assert metrics["continuation_steps_requested_per_arm"] == 176
    assert metrics["no_memory_continuation_steps_executed"] == 171
    assert metrics["with_memory_continuation_steps_executed"] == 171
    assert metrics["no_memory_terminal_episodes"] == 1
    assert metrics["with_memory_terminal_episodes"] == 1
    assert metrics["all_actions_legal"] is True


def test_sage8c_keeps_the_initial_local_gain_secondary(real_payload):
    secondary = real_payload["secondary_metrics"]
    summary = real_payload["summary"]

    assert secondary["no_memory_total"] == 2.0
    assert secondary["with_memory_total"] == 114.0
    assert secondary["absolute_gain"] == 112.0
    assert secondary["episodes_with_positive_gain"] == 11
    assert secondary["improved"] is True
    assert secondary["counted_as_level_completion"] is False
    assert secondary["counted_as_win"] is False
    assert summary["local_signal_counted_as_arc_progress"] is False


def test_sage8c_uses_identical_outcome_blind_continuations(real_payload):
    config = real_payload["config"]["evaluation_design"]
    episodes = real_payload["paired_rollouts"]

    assert config["continuation_derived_only_from_pre_decision_action_history"] is True
    assert config["continuation_selected_before_rollout_execution"] is True
    assert config["outcome_fields_read_for_continuation_selection"] == []
    assert all(row["same_prefix_between_arms"] is True for row in episodes)
    assert all(
        row["same_continuation_schedule_between_arms"] is True for row in episodes
    )
    assert all(row["schedule_selected_before_execution"] is True for row in episodes)
    assert all(row["outcome_fields_read_for_schedule"] == [] for row in episodes)
    assert real_payload["summary"]["continuation_selected_from_outcomes"] is False


def test_sage8c_changes_only_the_initial_decision(real_payload):
    episodes = real_payload["paired_rollouts"]

    assert all(row["relational_memory_applied"] is True for row in episodes)
    assert all(
        row["policy_decision"]["decision_reason"]
        == evaluation.LOWER_EFFECT_COMPARATOR_MATCH
        for row in episodes
    )
    for row in episodes:
        no_trace = row["no_memory_arm"]["trace"]
        memory_trace = row["with_memory_arm"]["trace"]
        assert no_trace[0]["phase"] == memory_trace[0]["phase"] == "initial_decision"
        assert no_trace[0]["action"] == row["no_memory_action"]
        assert memory_trace[0]["action"] == row["with_memory_action"]
        assert [step["action"] for step in no_trace[1:]] == [
            step["action"] for step in memory_trace[1:]
        ]
        assert [step["action_args"] for step in no_trace[1:]] == [
            step["action_args"] for step in memory_trace[1:]
        ]


def test_sage8c_stops_only_the_same_terminal_pair_early(real_payload):
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


def test_sage8c_builds_schedule_only_from_recent_history():
    base = {
        "evaluation_id": "sage8b::paired::001",
        "request": {
            "context_replay": ["ACTION1", "ACTION2", "ACTION3"],
            "context_replay_args": [{}, {"x": 2}, {"x": 3}],
        },
    }

    result = evaluation.build_sage8c_rollout_specification(
        base, continuation_horizon=5, continuation_tail_window=2
    )

    assert result["evaluation_id"] == "sage8c::multi_action::001"
    assert result["continuation_schedule"] == [
        {"action": "ACTION2", "action_args": {"x": 2}},
        {"action": "ACTION3", "action_args": {"x": 3}},
        {"action": "ACTION2", "action_args": {"x": 2}},
        {"action": "ACTION3", "action_args": {"x": 3}},
        {"action": "ACTION2", "action_args": {"x": 2}},
    ]
    assert result["outcome_fields_read_for_schedule"] == []


def test_sage8c_rejects_invalid_config_or_drifted_sources(real_sources):
    sage8b, policy, a34_2, a34_3 = real_sources
    with pytest.raises(ValueError, match="greater than one"):
        evaluation._validate_rollout_config(1, 16)
    with pytest.raises(ValueError, match="must be positive"):
        evaluation._validate_rollout_config(16, 0)

    changed = copy.deepcopy(sage8b)
    changed["summary"]["primary_arc_progress_improved"] = True
    with pytest.raises(ValueError, match="local-only SAGE.8b"):
        evaluation.validate_sage8c_sources(changed, policy, a34_2, a34_3)

    changed = copy.deepcopy(sage8b)
    changed["paired_episodes"][0]["context_snapshot_hash"] = "drifted"
    with pytest.raises(ValueError, match="scopes must match exactly"):
        evaluation.validate_sage8c_sources(changed, policy, a34_2, a34_3)


def test_sage8c_does_not_reevaluate_truth_or_count_support(real_payload):
    assert real_payload["truth_status"] == evaluation.SAGE8C_TRUTH_STATUS
    assert real_payload["comparative_evaluation_performed"] is True
    assert real_payload["multi_action_live_rollout_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["support"] == 0
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8c_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8c.json"
    evaluation.write_sage8c_relational_memory_multi_action_evaluation(
        real_payload, output_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8c_relational_memory_multi_action_evaluation
        is evaluation.run_sage8c_relational_memory_multi_action_evaluation
    )
    assert (
        sage.write_sage8c_relational_memory_multi_action_evaluation
        is evaluation.write_sage8c_relational_memory_multi_action_evaluation
    )
