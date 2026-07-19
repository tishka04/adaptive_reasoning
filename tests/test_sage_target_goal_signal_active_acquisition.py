import copy
import json

import pytest

from theory.sage import target_goal_signal_active_acquisition as acquisition


@pytest.fixture(scope="module")
def real_payload():
    return acquisition.run_sage8g_target_goal_signal_active_acquisition()


def test_sage8g_acquires_one_exact_positive_per_target(real_payload):
    summary = real_payload["summary"]

    assert summary["target_games_audited"] == 2
    assert summary["target_games_with_positive_transition"] == 2
    assert summary["target_games_with_exact_positive_replay"] == 2
    assert summary["verified_target_goal_transitions"] == 2
    assert summary["verified_target_level_up_transitions"] == 2
    assert summary["verified_target_win_transitions"] == 0
    assert summary["exact_target_goal_states"] == 2
    assert summary["exact_target_goal_signal_entries"] == 2
    assert summary["planner_activation_authorized"] is True
    assert summary["gate_passed"] is True


def test_sage8g_tn36_protocol_is_exhaustively_bounded_and_finds_mask_858(
    real_payload,
):
    target = _target(real_payload, acquisition.TN36_GAME_ID)
    discovery = target["discovery"]

    assert discovery["protocol"] == acquisition.TN36_PROTOCOL
    assert discovery["protocol_is_bounded"] is True
    assert discovery["toggle_count"] == 10
    assert discovery["configuration_space_size"] == 1024
    assert discovery["configurations_tested"] == 859
    assert discovery["positive_configuration_mask"] == 858
    assert discovery["positive_step"] == 7
    assert discovery["calibration_action_executions"] == 11
    assert discovery["discovery_action_executions"] == 4915
    assert discovery["maximum_action_execution_budget"] == 6155
    assert discovery["submit_action_identity"] == 'ACTION6::{"x":34,"y":51}'
    assert discovery["commands"] == [
        {"action": "ACTION6", "action_args": {"x": 25, "y": 42}},
        {"action": "ACTION6", "action_args": {"x": 35, "y": 42}},
        {"action": "ACTION6", "action_args": {"x": 40, "y": 42}},
        {"action": "ACTION6", "action_args": {"x": 26, "y": 44}},
        {"action": "ACTION6", "action_args": {"x": 36, "y": 44}},
        {"action": "ACTION6", "action_args": {"x": 41, "y": 44}},
        {"action": "ACTION6", "action_args": {"x": 34, "y": 51}},
    ]


def test_sage8g_tn36_submit_is_derived_from_observed_pixel_delta(real_payload):
    discovery = _target(real_payload, acquisition.TN36_GAME_ID)["discovery"]
    calibration = discovery["calibration"]
    minimum = min(row["changed_pixels"] for row in calibration)
    minimum_rows = [row for row in calibration if row["changed_pixels"] == minimum]

    assert len(calibration) == 11
    assert minimum == 1
    assert len(minimum_rows) == 1
    assert minimum_rows[0]["action_identity"] == discovery["submit_action_identity"]
    assert all(
        row["changed_pixels"] == 4 for row in calibration if row is not minimum_rows[0]
    )


def test_sage8g_wa30_replays_the_observation_derived_transport_plan(real_payload):
    target = _target(real_payload, acquisition.WA30_GAME_ID)
    discovery = target["discovery"]

    assert discovery["protocol"] == acquisition.WA30_PROTOCOL
    assert discovery["candidate_origin"] == "BOUNDED_BLACK_BOX_SALIENT_OBJECT_SEARCH"
    assert discovery["protocol_is_bounded"] is True
    assert discovery["candidate_sequences_tested"] == 1
    assert discovery["candidate_action_bound"] == 50
    assert discovery["maximum_action_execution_budget"] == 50
    assert discovery["actions_executed"] == 50
    assert discovery["positive_step"] == 50
    assert [row["action"] for row in discovery["commands"]] == list(
        acquisition.WA30_OBSERVATION_DERIVED_SEQUENCE
    )
    assert discovery["proof"]["positive_command"] == {
        "action": "ACTION5",
        "action_args": {},
    }


def test_sage8g_independent_replays_match_all_exact_state_edges(real_payload):
    expected = {
        acquisition.TN36_GAME_ID: (
            "95cd35ef39b9d52b",
            "1188520fa54c7420",
            "a2f95e098c948b18",
        ),
        acquisition.WA30_GAME_ID: (
            "dae9592860684892",
            "d8e1134efabf42b1",
            "3c30e17de1226e2b",
        ),
    }
    for row in real_payload["target_acquisitions"]:
        proof = row["discovery"]["proof"]
        replay = row["independent_replay"]
        assert (
            proof["reset_visual_digest"],
            proof["before_visual_digest"],
            proof["after_visual_digest"],
        ) == expected[row["game_id"]]
        assert proof["levels_completed_before"] == 0
        assert proof["levels_completed_after"] == 1
        assert proof["level_delta"] == 1
        assert replay["exact_replay_verified"] is True
        assert replay["commands_exactly_replayed"] is True
        assert replay["reset_visual_digest_match"] is True
        assert replay["before_visual_digest_match"] is True
        assert replay["after_visual_digest_match"] is True
        assert replay["outcome_match"] is True


def test_sage8g_signal_bank_is_exact_target_scoped_without_transfer(real_payload):
    bank = real_payload["exact_target_signal_bank"]
    entries = bank["entries"]

    assert bank["summary"]["per_game_goal_transitions"] == {
        acquisition.TN36_GAME_ID: 1,
        acquisition.WA30_GAME_ID: 1,
    }
    assert len(entries) == 2
    assert all(
        row["scope_key"] == f"{row['game_id']}|{row['visual_digest']}"
        and row["scope"] == acquisition.GOAL_SIGNAL_SCOPE
        and row["exact_independent_replay_verified"] is True
        and row["exact_match_required"] is True
        and row["fuzzy_match_allowed"] is False
        and row["cross_game_transfer_allowed"] is False
        and row["level_up_count"] == 1
        and row["win_count"] == 0
        and row["truth_status"] == acquisition.SAGE8G_TRUTH_STATUS
        and row["support"] == 0
        for row in entries
    )


def test_sage8g_exact_target_admission_unblocks_only_the_next_evaluation(
    real_payload,
):
    admission = real_payload["target_admission"]

    assert admission["target_games_audited"] == 2
    assert admission["target_games_admitted"] == 2
    assert admission["all_target_games_admitted"] is True
    assert admission["planner_activation_authorized"] is True
    assert all(
        row["exact_target_goal_signal_entries"] == 1
        and row["positive_transition_acquired"] is True
        and row["exact_independent_replay_verified"] is True
        and row["planner_activation_authorized"] is True
        and row["admission_reason"] == "EXACT_TARGET_GAME_POSITIVE_REPLAY_AVAILABLE"
        for row in admission["target_games"]
    )
    assert real_payload["paired_closed_loop_evaluation_authorized"] is True
    assert real_payload["paired_closed_loop_evaluation_performed"] is False
    assert real_payload["evaluation_episodes_executed"] == 0


def test_sage8g_uses_no_source_code_cross_game_transfer_or_truth_mutation(
    real_payload,
):
    design = real_payload["config"]["acquisition_design"]

    assert design["environment_observation_api_only"] is True
    assert design["game_source_files_opened"] == []
    assert design["future_outcomes_used_for_action_ranking"] is False
    assert design["cross_game_action_transfer_allowed"] is False
    assert real_payload["game_source_files_opened"] == []
    assert real_payload["cross_game_transfer_performed"] is False
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["truth_status"] == acquisition.SAGE8G_TRUTH_STATUS
    assert real_payload["scientific_review_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["registry_support_recounted"] is False
    assert real_payload["a33_mutated"] is False
    assert real_payload["support"] == 0
    assert real_payload["wrong_confirmations"] == 0
    assert all(real_payload["gate"].values())


def test_sage8g_reports_the_real_bounded_action_accounting(real_payload):
    summary = real_payload["summary"]

    assert summary["tn36_toggle_configurations_tested"] == 859
    assert summary["tn36_toggle_configuration_space"] == 1024
    assert summary["discovery_action_executions"] == 4965
    assert summary["independent_replay_action_executions"] == 57
    assert summary["total_live_action_executions"] == 5022
    assert summary["all_protocols_bounded"] is True
    assert summary["game_source_files_opened"] == 0


def test_sage8g_keeps_a_bounded_protocol_valid_when_admission_is_incomplete():
    source = json.loads(
        acquisition.DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH.read_text(
            encoding="utf-8"
        )
    )
    discovery = acquisition.bounded_unsupported_target_protocol("game-a")
    acquisitions = [
        {
            "discovery": discovery,
            "independent_replay": {"exact_replay_verified": False},
            "admission": {
                "planner_activation_authorized": False,
                "exact_target_goal_signal_entries": 0,
            },
        }
    ]
    bank = acquisition.build_exact_target_signal_bank([], ["game-a"])

    gate = acquisition.build_sage8g_gate(
        source,
        acquisitions,
        bank,
        planner_activation_authorized=False,
    )

    assert all(gate.values())


def test_sage8g_rejects_a_drifted_sage8f_source():
    source = json.loads(
        acquisition.DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH.read_text(
            encoding="utf-8"
        )
    )
    changed = copy.deepcopy(source)
    changed["planner_activation_authorized"] = True

    with pytest.raises(ValueError, match="target-blocked SAGE.8f"):
        acquisition.validate_sage8g_source(changed)


def test_sage8g_writer_and_package_exports_round_trip(tmp_path, real_payload):
    output_path = tmp_path / "sage8g.json"
    acquisition.write_sage8g_target_goal_signal_active_acquisition(
        real_payload, output_path
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == real_payload

    import theory.sage as sage

    assert (
        sage.run_sage8g_target_goal_signal_active_acquisition
        is acquisition.run_sage8g_target_goal_signal_active_acquisition
    )
    assert (
        sage.write_sage8g_target_goal_signal_active_acquisition
        is acquisition.write_sage8g_target_goal_signal_active_acquisition
    )


def _target(payload, game_id):
    return next(
        row for row in payload["target_acquisitions"] if row["game_id"] == game_id
    )
