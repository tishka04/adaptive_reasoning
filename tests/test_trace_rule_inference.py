from human_trace.schema import EpisodeRecord, StepRecord
from human_trace.task_program import TaskProgram
from trace_rule_inference import (
    EpisodeBundle,
    build_level_segments,
    build_task_program,
    infer_cross_level_invariants,
    _select_bundles,
)


def _step(idx, before, action, after, *, level=0, state="NOT_FINISHED", intent="test_move"):
    return StepRecord(
        game_id="ar25-e3c63847",
        episode_id="ep",
        step=idx,
        frame_before=before,
        available_actions=[1, 2, 3, 4, 5, 6, 7],
        action=action,
        action_args=None,
        frame_after=after,
        game_state_after=state,
        levels_completed_after=level,
        intent=intent,
    )


def _bundle(steps):
    return EpisodeBundle(
        episode=EpisodeRecord(
            game_id="ar25-e3c63847",
            episode_id="ep",
            started_at="2026-06-08T00:00:00+00:00",
            ended_at="2026-06-08T00:01:00+00:00",
            n_steps=len(steps),
            final_state="GAME_OVER",
            levels_completed=2,
            objective_guess="must_match_yellow_and_purple_shapes",
            discovered_mechanics=[
                "gray_shape_is_controling_shapes",
                "dotes_lines_act_as_cursors",
                "yellow_shapes_can_go_out_the_grid",
            ],
            discovered_mistakes=["all_shapes_do_not_must_match"],
        ),
        steps=steps,
    )


def test_build_level_segments_splits_successful_level_ups():
    steps = [
        _step(0, [[1]], "ACTION1", [[2]], level=0),
        _step(1, [[2]], "ACTION2", [[3]], level=1),
        _step(2, [[3]], "ACTION5", [[4]], level=1, intent="test_interaction"),
        _step(3, [[4]], "ACTION3", [[5]], level=2),
    ]

    segments = build_level_segments(_bundle(steps), max_level=2)

    assert [segment.level_number for segment in segments] == [1, 2]
    assert segments[0].actions == ["ACTION1", "ACTION2"]
    assert segments[0].level_up_action == "ACTION2"
    assert segments[1].actions == ["ACTION5", "ACTION3"]
    assert segments[1].trace_start_step == 2


def test_select_bundles_ignores_truncated_high_level_episode():
    full = _bundle(
        [
            _step(0, [[1]], "ACTION1", [[2]], level=1),
            _step(1, [[2]], "ACTION2", [[3]], level=2),
        ]
    )
    truncated = EpisodeBundle(
        episode=EpisodeRecord(
            game_id="ar25-e3c63847",
            episode_id="truncated",
            started_at="2026-06-08T00:00:00+00:00",
            ended_at="2026-06-08T00:00:01+00:00",
            n_steps=1,
            final_state="QUIT",
            levels_completed=7,
        ),
        steps=[_step(0, [[9]], "ACTION7", [[9]], level=7)],
    )

    selected = _select_bundles([truncated, full], max_level=2, episode_id=None)

    assert [bundle.episode_id for bundle in selected] == ["ep"]


def test_build_task_program_marks_action3_distinct_as_contradiction():
    bundle = _bundle(
        [
            _step(0, [[1]], "ACTION1", [[2]], level=0),
            _step(1, [[2]], "ACTION2", [[3]], level=1),
            _step(2, [[3]], "ACTION5", [[4]], level=1),
            _step(3, [[4]], "ACTION3", [[5]], level=2),
        ]
    )
    segments = build_level_segments(bundle, max_level=2)
    evidence = {
        "objective_guesses": {"must_match_yellow_and_purple_shapes": 1},
        "discovered_mechanics": {
            "gray_shape_is_controling_shapes": 1,
            "dotes_lines_act_as_cursors": 1,
            "yellow_shapes_can_go_out_the_grid": 1,
        },
        "discovered_mistakes": {"all_shapes_do_not_must_match": 1},
    }
    level7 = {
        "action3_safe_distinct_bbox_count": 0,
        "fatal_distinct_bbox_count": 8,
    }

    invariants = infer_cross_level_invariants(segments, evidence, level7)
    program = build_task_program(
        game_id="ar25-e3c63847",
        selected_bundles=[bundle],
        invariants=invariants,
        level7_diagnostic=level7,
    )

    reloaded = TaskProgram.from_json(program.to_json())
    assert reloaded.goal_family == "correspondence"
    assert any("ACTION3 distinct bbox" in item for item in reloaded.anti_patterns)
    assert any(role.action == "ACTION5" and role.role == "control_switch" for role in reloaded.action_roles)
    assert any(goal.id == "probe_mask_toggle" for goal in reloaded.subgoal_tests)
