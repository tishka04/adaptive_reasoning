import copy
import json

import pytest

from theory.sage import goal_grounded_signal_acquisition as acquisition


@pytest.fixture(scope="module")
def real_payload():
    return acquisition.run_sage8f_goal_grounded_signal_acquisition()


def test_sage8f_acquires_real_level_and_win_grounded_signals(real_payload):
    summary = real_payload["summary"]

    assert summary["source_trace_files"] == 18
    assert summary["source_rows_scanned"] == 5711
    assert summary["verified_goal_transitions"] == 74
    assert summary["verified_level_up_transitions"] == 74
    assert summary["verified_win_transitions"] == 5
    assert summary["source_games_count"] == 6
    assert summary["exact_goal_states"] == 67
    assert summary["ambiguous_exact_goal_states"] == 3
    assert summary["frame_continuity_mismatches"] == 0
    assert summary["gate_passed"] is True


def test_sage8f_signal_bank_preserves_observed_source_distribution(real_payload):
    bank_summary = real_payload["goal_signal_bank"]["summary"]

    assert bank_summary["per_game_goal_transitions"] == {
        "ar25-e3c63847": 12,
        "bp35-0a0ad940": 20,
        "cd82-fb555c5d": 8,
        "cn04-65d47d14": 10,
        "dc22-4c9bff3e": 14,
        "ft09-0d8bbf25": 10,
    }
    assert bank_summary["per_action_family_goal_transitions"] == {
        "ACTION1": 8,
        "ACTION2": 17,
        "ACTION3": 16,
        "ACTION4": 8,
        "ACTION5": 4,
        "ACTION6": 21,
    }
    assert bank_summary["frame_continuity_checks"] == 5661
    assert bank_summary["goal_candidates_quarantined"] == 0
    assert bank_summary["reset_rows_excluded"] == 50
    assert bank_summary["non_goal_rows_excluded"] == 5587


def test_sage8f_goal_entries_are_exact_and_never_cross_game(real_payload):
    entries = real_payload["goal_signal_bank"]["entries"]

    assert len(entries) == 67
    assert all(
        row["scope_key"] == f"{row['game_id']}|{row['visual_digest']}"
        and row["scope"] == acquisition.GOAL_SIGNAL_SCOPE
        and row["exact_match_required"] is True
        and row["fuzzy_match_allowed"] is False
        and row["cross_game_transfer_allowed"] is False
        and row["level_up_count"] > 0
        and row["truth_status"] == acquisition.SAGE8F_TRUTH_STATUS
        and row["support"] == 0
        for row in entries
    )
    assert sum(row["demonstration_count"] for row in entries) == 74
    assert sum(row["win_count"] for row in entries) == 5


def test_sage8f_audits_and_blocks_uncovered_target_games(real_payload):
    audit = real_payload["target_coverage_audit"]
    summary = audit["summary"]

    assert summary["target_games_audited"] == 2
    assert summary["target_transitions_scanned"] == 1788
    assert summary["observed_target_goal_transitions"] == 0
    assert summary["exact_target_goal_signal_entries"] == 0
    assert summary["source_goal_signal_demonstrations_quarantined_from_transfer"] == 74
    assert summary["cross_game_transfer_performed"] is False
    assert summary["planner_activation_authorized"] is False
    assert summary["admission_status"] == "TARGET_DOMAIN_BLOCKED"
    assert {row["game_id"] for row in audit["target_games"]} == {
        "tn36-ab4f63cc",
        "wa30-ee6fef47",
    }
    assert all(
        row["target_transitions_scanned"] == 894
        and row["observed_target_goal_transitions"] == 0
        and row["exact_human_goal_signal_entries"] == 0
        and row["source_game_signal_entries_considered_for_transfer"] == 0
        and row["cross_game_signal_entries_quarantined"] == 67
        and row["planner_activation_authorized"] is False
        and row["admission_reason"] == acquisition.TARGET_BLOCK_REASON
        for row in audit["target_games"]
    )


def test_sage8f_does_not_run_an_unadmitted_evaluation(real_payload):
    assert real_payload["outcome_status"] == acquisition.SAGE8F_TARGET_BLOCKED
    assert real_payload["planner_activation_authorized"] is False
    assert real_payload["closed_loop_live_rollout_performed"] is False
    assert real_payload["comparative_evaluation_performed"] is False
    assert real_payload["evaluation_episodes_executed"] == 0
    assert real_payload["summary"]["closed_loop_evaluation_performed"] is False
    assert real_payload["summary"]["evaluation_episodes_executed"] == 0


def test_sage8f_uses_no_sage8e_outcome_for_training_or_tuning(real_payload):
    config = real_payload["config"]["acquisition_design"]
    bank_config = real_payload["goal_signal_bank"]["config"]

    assert config["source_outcomes_are_preexisting_human_demonstrations"] is True
    assert config["sage8e_evaluation_outcomes_used_for_training_or_tuning"] is False
    assert bank_config["evaluation_outcome_fields_read"] == []
    assert real_payload["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert (
        real_payload["summary"][
            "sage8e_evaluation_outcomes_used_for_training_or_tuning"
        ]
        is False
    )


def test_sage8f_extracts_only_continuous_goal_transitions(tmp_path):
    rows = [
        _trace_row(0, "RESET", 0, [[0]], [[0]]),
        _trace_row(1, "ACTION1", 0, [[0]], [[1]]),
        _trace_row(2, "ACTION2", 1, [[1]], [[2]]),
        _trace_row(3, "ACTION3", 1, [[2]], [[3]], state="WIN"),
    ]
    trace_path = tmp_path / "game-a.steps.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    bank = acquisition.acquire_goal_grounded_signal_bank(tmp_path)

    assert bank["summary"]["source_rows_scanned"] == 4
    assert bank["summary"]["reset_rows_excluded"] == 1
    assert bank["summary"]["non_goal_rows_excluded"] == 1
    assert bank["summary"]["verified_goal_transitions"] == 2
    assert bank["summary"]["verified_level_up_transitions"] == 1
    assert bank["summary"]["verified_win_transitions"] == 1
    assert bank["summary"]["exact_goal_states"] == 2


def test_sage8f_rejects_a_goal_candidate_with_broken_continuity(tmp_path):
    rows = [
        _trace_row(0, "RESET", 0, [[0]], [[0]]),
        _trace_row(1, "ACTION2", 1, [[9]], [[2]]),
    ]
    (tmp_path / "game-a.steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="goal signal bank gate"):
        acquisition.acquire_goal_grounded_signal_bank(tmp_path)


def test_target_admission_requires_both_exact_signal_and_observed_positive(tmp_path):
    signal_bank = {
        "entries": [{"game_id": "game-a"}],
        "summary": {"verified_goal_transitions": 1},
    }
    target_path = tmp_path / "game-a.json"
    target_path.write_text(
        json.dumps(
            [
                {
                    "game_id": "game-a",
                    "level_before": 0,
                    "level_after": 1,
                    "level_changed": True,
                    "state_after": "NOT_FINISHED",
                }
            ]
        ),
        encoding="utf-8",
    )

    audit = acquisition.audit_target_goal_signal_coverage(
        signal_bank,
        target_transition_paths=[target_path],
        target_games=["game-a"],
    )

    assert audit["summary"]["observed_target_goal_transitions"] == 1
    assert audit["summary"]["exact_target_goal_signal_entries"] == 1
    assert audit["summary"]["planner_activation_authorized"] is True
    assert audit["target_games"][0]["admission_reason"] == (
        "EXACT_TARGET_GAME_GOAL_SIGNAL_AVAILABLE"
    )


def test_target_admission_blocks_cross_game_signal_even_with_matching_action(tmp_path):
    signal_bank = {
        "entries": [
            {
                "game_id": "source-game",
                "action_candidates": [{"action": "ACTION1"}],
            }
        ],
        "summary": {"verified_goal_transitions": 1},
    }
    target_path = tmp_path / "target-game.json"
    target_path.write_text(
        json.dumps(
            [
                {
                    "game_id": "target-game",
                    "action": "ACTION1",
                    "level_before": 0,
                    "level_after": 0,
                    "level_changed": False,
                    "state_after": "NOT_FINISHED",
                }
            ]
        ),
        encoding="utf-8",
    )

    audit = acquisition.audit_target_goal_signal_coverage(
        signal_bank,
        target_transition_paths=[target_path],
        target_games=["target-game"],
    )

    target = audit["target_games"][0]
    assert target["source_game_signal_entries_considered_for_transfer"] == 0
    assert target["cross_game_signal_entries_quarantined"] == 1
    assert target["planner_activation_authorized"] is False
    assert target["admission_reason"] == acquisition.TARGET_BLOCK_REASON


def test_sage8f_rejects_a_drifted_sage8e_source():
    source = json.loads(
        acquisition.DEFAULT_SAGE8E_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_EVALUATION_PATH.read_text(
            encoding="utf-8"
        )
    )
    changed = copy.deepcopy(source)
    changed["summary"]["primary_arc_progress_improved"] = True

    with pytest.raises(ValueError, match="leak-safe local-only SAGE.8e"):
        acquisition.validate_sage8f_source(changed)


def test_sage8f_does_not_reevaluate_truth_or_count_support(real_payload):
    assert real_payload["truth_status"] == acquisition.SAGE8F_TRUTH_STATUS
    assert real_payload["goal_grounded_signal_acquisition_performed"] is True
    assert real_payload["target_scope_admission_audit_performed"] is True
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["support"] == 0
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8f_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8f.json"
    acquisition.write_sage8f_goal_grounded_signal_acquisition(real_payload, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8f_goal_grounded_signal_acquisition
        is acquisition.run_sage8f_goal_grounded_signal_acquisition
    )
    assert (
        sage.write_sage8f_goal_grounded_signal_acquisition
        is acquisition.write_sage8f_goal_grounded_signal_acquisition
    )


def _trace_row(step, action, levels, before, after, *, state="NOT_FINISHED"):
    return {
        "game_id": "game-a",
        "episode_id": "episode-a",
        "step": step,
        "frame_before": before,
        "available_actions": [1, 2, 3],
        "action": action,
        "action_args": None,
        "frame_after": after,
        "game_state_after": state,
        "levels_completed_after": levels,
        "intent": "none",
    }
