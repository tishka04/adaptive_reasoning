from theory.m1.mechanic_typing import (
    profile_game_mechanics,
    profile_one_game,
    summarize_profiles,
)


def _row(game_id, **overrides):
    row = {
        "game_id": game_id,
        "action": "ACTION1",
        "action_args": None,
        "color_changed": False,
        "position_changed": False,
        "object_motion_vectors": [],
        "created_object_count": 0,
        "removed_object_count": 0,
        "object_count_changed": False,
        "adjacency_changed": False,
        "shape_changed": False,
        "player_moved": False,
        "level_progressed": False,
        "num_cells_changed": 0,
        "changed_cell_ratio": 0.0,
        "color_pairs_changed": [],
    }
    row.update(overrides)
    return row


def test_mechanic_typing_flags_color_changes_without_source_grounding():
    rows = [
        _row(
            "bp35-test",
            color_changed=True,
            position_changed=True,
            object_motion_vectors=[{"dx": 1}],
            created_object_count=1,
            removed_object_count=1,
            object_count_changed=True,
            adjacency_changed=True,
            shape_changed=True,
            player_moved=True,
            num_cells_changed=10,
            changed_cell_ratio=0.2,
            color_pairs_changed=[{"before_color": 1, "after_color": 2}],
        )
        for _ in range(4)
    ]

    profile = profile_one_game(
        "bp35-test",
        rows,
        grounding_fit={
            "source_color_predictiveness": 0.0,
            "source_not_selectable_rate": 1.0,
        },
    )

    assert profile.raw_rates["color_change_rate"] == 1.0
    assert profile.mechanic_scores["position_motion"] == 1.0
    assert profile.mechanic_scores["color_source_grounding"] == 0.0
    assert (
        profile.representation_warning
        == "color_source_schema_misaligned_with_observed_mechanics"
    )
    assert profile.status == "UNRESOLVED"
    assert profile.trace_support_counted_as_proof is False
    assert profile.prior_counted_as_proof is False


def test_mechanic_typing_keeps_dc22_like_source_grounding_as_fit():
    rows = [
        _row(
            "dc22-test",
            color_changed=True,
            position_changed=True,
            object_motion_vectors=[{"dx": 1}],
            adjacency_changed=True,
            num_cells_changed=5,
            color_pairs_changed=[{"before_color": 8, "after_color": 2}],
        )
        for _ in range(3)
    ]

    profile = profile_one_game(
        "dc22-test",
        rows,
        grounding_fit={
            "source_color_predictiveness": 0.25,
            "source_not_selectable_rate": 0.75,
            "pairs_entering_agenda_rate": 0.1,
        },
    )

    assert profile.grounding_fit["source_color_predictiveness"] == 0.25
    assert profile.representation_warning is None
    assert "color_change" in profile.mechanic_tags


def test_profile_game_mechanics_and_summary_are_json_ready():
    profiles = profile_game_mechanics(
        [
            _row("aa-game", color_changed=True, num_cells_changed=1),
            _row("aa-game", action="RESET", color_changed=True, num_cells_changed=1),
            _row("bb-game", position_changed=True, object_motion_vectors=[{}]),
        ],
        grounding_fit={
            "aa-game": {"source_color_predictiveness": 0.5},
            "bb-game": {"source_color_predictiveness": 0.0},
        },
    )
    summary = summarize_profiles(profiles)

    assert [profile.game_id for profile in profiles] == ["aa-game", "bb-game"]
    assert profiles[0].non_reset_observations == 1
    assert summary["game_count"] == 2
    assert summary["source_color_predictiveness_by_game"]["aa-game"] == 0.5
    assert profiles[0].to_dict()["status"] == "UNRESOLVED"
