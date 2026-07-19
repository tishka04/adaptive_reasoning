import copy
import json
from dataclasses import dataclass, field

import pytest

from theory.sage import relational_memory_objective_closed_loop_evaluation as evaluation


@dataclass(frozen=True)
class Action:
    name: str
    action_args: dict = field(default_factory=dict)


@pytest.fixture(scope="module")
def real_payload():
    return evaluation.run_sage8e_relational_memory_objective_closed_loop_evaluation()


@pytest.fixture(scope="module")
def real_sources():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            evaluation.DEFAULT_SAGE8D_RELATIONAL_MEMORY_CLOSED_LOOP_EVALUATION_PATH,
            evaluation.DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH,
            evaluation.DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
            evaluation.DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
            evaluation.DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH,
            evaluation.DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
        )
    )


def test_sage8e_runs_the_objective_closed_loop_on_all_exact_contexts(real_payload):
    summary = real_payload["summary"]

    assert summary["paired_rollouts_evaluated"] == 11
    assert summary["games_evaluated"] == ["tn36-ab4f63cc", "wa30-ee6fef47"]
    assert summary["continuation_horizon"] == 16
    assert summary["memory_policy_applications"] == 11
    assert summary["exact_paired_replays"] == 11
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == evaluation.SAGE8E_LOCAL_ONLY


def test_sage8e_compiles_an_outcome_blind_exact_scope_objective(real_payload):
    model = real_payload["objective_model"]
    config = model["config"]
    summary = model["summary"]

    assert summary["source_replays"] == 11
    assert summary["source_replays_exactly_verified"] == 11
    assert summary["demonstration_transitions"] == 456
    assert summary["exact_visual_states"] == 137
    assert summary["ambiguous_visual_states"] == 24
    assert summary["endpoint_visual_states"] == 11
    assert summary["training_outcome_fields_read"] == 0
    assert summary["evaluation_outcomes_used_for_training_or_tuning"] == 0
    assert config["training_outcome_fields_read"] == []
    assert config["evaluation_outcomes_used_for_training_or_tuning"] == []
    assert config["objective_is_arc_progress_truth_claim"] is False
    assert model["scope_generalization_performed"] is False


def test_sage8e_objective_entries_are_exact_game_state_scoped(real_payload):
    entries = real_payload["objective_model"]["state_action_entries"]

    assert len(entries) == 137
    assert all(
        row["scope_key"] == f"{row['game_id']}|{row['visual_digest']}"
        and row["scope"] == evaluation.OBJECTIVE_SCOPE
        and row["exact_match_required"] is True
        and row["fuzzy_match_allowed"] is False
        and row["cross_game_transfer_allowed"] is False
        and row["truth_status"] == evaluation.SAGE8E_TRUTH_STATUS
        and row["support"] == 0
        for row in entries
    )


def test_sage8e_memory_trajectory_stays_in_the_learned_manifold_longer(real_payload):
    metrics = real_payload["objective_metrics"]

    assert metrics["no_memory_replanning_decisions"] == 171
    assert metrics["with_memory_replanning_decisions"] == 171
    assert metrics["no_memory_objective_applications"] == 28
    assert metrics["with_memory_objective_applications"] == 112
    assert metrics["no_memory_objective_quarantines"] == 143
    assert metrics["with_memory_objective_quarantines"] == 59
    assert metrics["no_memory_objective_coverage_rate"] == pytest.approx(28 / 171)
    assert metrics["with_memory_objective_coverage_rate"] == pytest.approx(112 / 171)
    assert metrics["episodes_with_divergent_replanned_trajectories"] == 10
    assert metrics["divergent_replanning_positions"] == 134
    assert metrics["all_actions_legal"] is True


def test_sage8e_each_live_decision_is_exact_scope_or_quarantined(real_payload):
    for row in real_payload["paired_rollouts"]:
        for arm_name in ("no_memory_arm", "with_memory_arm"):
            arm = row[arm_name]
            assert arm["replanning_decisions"] == arm["continuation_steps_executed"]
            assert len(arm["trace"]) == arm["continuation_steps_executed"] + 1
            for index, step in enumerate(arm["trace"][1:], start=1):
                previous_state = json.loads(arm["trace"][index - 1]["signature"])
                planner = step["planner_decision"]
                assert step["phase"] == "scope_safe_objective_replanning"
                assert planner["current_visual_digest"] == previous_state["digest"]
                assert planner["objective_scope_key"] == (
                    f"{row['game_id']}|{previous_state['digest']}"
                )
                assert planner["current_state_observation_used"] is True
                assert planner["future_outcome_fields_read"] == []
                assert planner["evaluation_outcomes_used_for_tuning"] == []
                assert planner["counterfactual_rollouts_performed"] == 0
                assert (
                    planner["learned_objective_applied"]
                    != planner["learned_objective_quarantined"]
                )
                if planner["learned_objective_applied"]:
                    assert planner["objective_scope_match"] is True
                    assert planner["objective_decision_reason"] == (
                        evaluation.OBJECTIVE_APPLIED
                    )
                else:
                    assert planner["objective_decision_reason"] in {
                        evaluation.OBJECTIVE_OUT_OF_SCOPE,
                        evaluation.OBJECTIVE_ACTION_UNAVAILABLE,
                    }


def test_sage8e_primary_arc_metrics_remain_unimproved(real_payload):
    levels = real_payload["primary_metrics"]["levels_completed"]
    wins = real_payload["primary_metrics"]["win_rate"]

    assert levels["no_memory_total_delta"] == 0
    assert levels["with_memory_total_delta"] == 0
    assert levels["absolute_delta_gain"] == 0
    assert levels["improved"] is False
    assert wins["no_memory_wins"] == 0
    assert wins["with_memory_wins"] == 0
    assert wins["no_memory"] == 0.0
    assert wins["with_memory"] == 0.0
    assert wins["absolute_gain"] == 0.0
    assert wins["improved"] is False
    assert real_payload["primary_arc_progress_improved"] is False


def test_sage8e_keeps_the_initial_local_gain_secondary(real_payload):
    secondary = real_payload["secondary_metrics"]

    assert secondary["no_memory_total"] == 2.0
    assert secondary["with_memory_total"] == 114.0
    assert secondary["absolute_gain"] == 112.0
    assert secondary["episodes_with_positive_gain"] == 11
    assert secondary["improved"] is True
    assert secondary["counted_as_level_completion"] is False
    assert secondary["counted_as_win"] is False
    assert real_payload["summary"]["local_signal_counted_as_arc_progress"] is False


def test_exact_scope_selector_prefers_the_demonstrated_action():
    actions = [Action("ACTION1"), Action("ACTION2")]
    signature = json.dumps({"digest": "visual-a", "shape": [4, 4]})
    objective_index = {
        "game-a|visual-a": {
            "action_candidates": [
                {
                    "action_identity": "ACTION2::{}",
                    "demonstration_count": 3,
                }
            ]
        }
    }

    selected, decision = evaluation.select_scope_safe_objective_action(
        actions,
        game_id="game-a",
        current_state_signature=signature,
        objective_index=objective_index,
        family_counts={"ACTION1": 0, "ACTION2": 99},
        concrete_counts={"ACTION1::{}": 0, "ACTION2::{}": 99},
        state_action_visits={},
        previous_action_identity="ACTION2::{}",
    )

    assert selected is actions[1]
    assert decision["learned_objective_applied"] is True
    assert decision["learned_objective_quarantined"] is False
    assert decision["objective_scope_key"] == "game-a|visual-a"
    assert decision["selected_demonstration_count"] == 3


@pytest.mark.parametrize(
    ("game_id", "digest", "reason"),
    [
        ("game-b", "visual-a", evaluation.OBJECTIVE_OUT_OF_SCOPE),
        ("game-a", "visual-b", evaluation.OBJECTIVE_OUT_OF_SCOPE),
        ("game-a", "visual-c", evaluation.OBJECTIVE_ACTION_UNAVAILABLE),
    ],
)
def test_scope_mismatch_or_unavailable_action_is_quarantined(game_id, digest, reason):
    actions = [Action("ACTION1"), Action("ACTION2")]
    objective_index = {
        "game-a|visual-a": {
            "action_candidates": [
                {"action_identity": "ACTION2::{}", "demonstration_count": 2}
            ]
        },
        "game-a|visual-c": {
            "action_candidates": [
                {"action_identity": "ACTION3::{}", "demonstration_count": 2}
            ]
        },
    }

    selected, decision = evaluation.select_scope_safe_objective_action(
        actions,
        game_id=game_id,
        current_state_signature=json.dumps({"digest": digest, "shape": [4, 4]}),
        objective_index=objective_index,
        family_counts={"ACTION1": 0, "ACTION2": 2},
        concrete_counts={"ACTION1::{}": 0, "ACTION2::{}": 2},
        state_action_visits={},
        previous_action_identity="ACTION2::{}",
    )

    assert selected is actions[0]
    assert decision["learned_objective_applied"] is False
    assert decision["learned_objective_quarantined"] is True
    assert decision["objective_decision_reason"] == reason
    assert decision["future_outcome_fields_read"] == []


def test_sage8e_rejects_invalid_config_or_drifted_sources(real_sources):
    sage8d, sage8c, sage8b, policy, a34_2, a34_3 = real_sources
    with pytest.raises(ValueError, match="greater than one"):
        evaluation._validate_objective_closed_loop_config(1)

    changed = copy.deepcopy(sage8d)
    changed["summary"]["primary_arc_progress_improved"] = True
    with pytest.raises(ValueError, match="local-only outcome-blind SAGE.8d"):
        evaluation.validate_sage8e_sources(
            changed, sage8c, sage8b, policy, a34_2, a34_3
        )

    changed = copy.deepcopy(sage8d)
    changed["paired_rollouts"][0]["context_snapshot_hash"] = "drifted"
    with pytest.raises(ValueError, match="context scopes must match exactly"):
        evaluation.validate_sage8e_sources(
            changed, sage8c, sage8b, policy, a34_2, a34_3
        )


def test_sage8e_does_not_reevaluate_truth_or_count_support(real_payload):
    assert real_payload["truth_status"] == evaluation.SAGE8E_TRUTH_STATUS
    assert real_payload["comparative_evaluation_performed"] is True
    assert real_payload["closed_loop_live_rollout_performed"] is True
    assert real_payload["scope_safe_objective_planning_performed"] is True
    assert real_payload["learned_objective_compiled_before_evaluation"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["support"] == 0
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8e_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8e.json"
    evaluation.write_sage8e_relational_memory_objective_closed_loop_evaluation(
        real_payload, output_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8e_relational_memory_objective_closed_loop_evaluation
        is evaluation.run_sage8e_relational_memory_objective_closed_loop_evaluation
    )
    assert (
        sage.write_sage8e_relational_memory_objective_closed_loop_evaluation
        is evaluation.write_sage8e_relational_memory_objective_closed_loop_evaluation
    )
