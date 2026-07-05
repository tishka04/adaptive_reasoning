import textwrap

import numpy as np

from compare_sampling_setups import SETUPS, parse_runner_output
from human_trace.integration import HumanPriorPack, seed_game_memory
from human_trace.memory import HumanTrace, HumanTraceMemory
from human_trace.schema import EpisodeRecord, StepRecord
from v4_1_reasoning_system.arc_agi.game_memory import GameMemory
from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig


def _simple_pack() -> HumanPriorPack:
    game_id = "ar25-e3c63847"
    steps = [
        StepRecord(
            game_id=game_id,
            episode_id="ep0",
            step=0,
            frame_before=[
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            available_actions=[1, 4, 6],
            action="ACTION4",
            action_args=None,
            frame_after=[
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            game_state_after="NOT_FINISHED",
            levels_completed_after=0,
            intent="test_move",
            hypothesis="player moves right",
        ),
        StepRecord(
            game_id=game_id,
            episode_id="ep0",
            step=1,
            frame_before=[
                [0, 0, 0],
                [0, 2, 1],
                [0, 0, 0],
            ],
            available_actions=[1, 4, 6],
            action="ACTION6",
            action_args={"x": 1, "y": 1},
            frame_after=[
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            game_state_after="WIN",
            levels_completed_after=1,
            intent="test_click",
            hypothesis="click removes target",
        ),
    ]
    episodes = [
        EpisodeRecord(
            game_id=game_id,
            episode_id="ep0",
            started_at="2026-05-30T10:00:00+00:00",
            ended_at="2026-05-30T10:00:10+00:00",
            n_steps=2,
            final_state="WIN",
            levels_completed=1,
            game_type_guess="click_puzzle",
            objective_guess="remove the target",
            discovered_mechanics=["ACTION4 moves player right", "ACTION6 clicks target"],
        )
    ]
    return HumanPriorPack(
        game_id=game_id,
        steps=steps,
        episodes=episodes,
        hypothesis_priors=[("human::player moves right", 0.7)],
    )


def _danger_pack() -> HumanPriorPack:
    game_id = "ar25-e3c63847"
    steps = [
        StepRecord(
            game_id=game_id,
            episode_id="danger_ep",
            step=0,
            frame_before=[[0, 1], [0, 0]],
            available_actions=[1, 2, 3],
            action="ACTION1",
            action_args=None,
            frame_after=[[0, 0], [1, 0]],
            game_state_after="NOT_FINISHED",
            levels_completed_after=0,
            intent="test_move",
        ),
        StepRecord(
            game_id=game_id,
            episode_id="danger_ep",
            step=1,
            frame_before=[[0, 0], [1, 2]],
            available_actions=[1, 2, 3],
            action="ACTION2",
            action_args=None,
            frame_after=[[3, 0], [0, 0]],
            game_state_after="NOT_FINISHED",
            levels_completed_after=1,
            intent="repeat_success",
        ),
        StepRecord(
            game_id=game_id,
            episode_id="danger_ep",
            step=2,
            frame_before=[[3, 0], [0, 4]],
            available_actions=[1, 2, 3],
            action="ACTION3",
            action_args=None,
            frame_after=[[0, 0], [0, 0]],
            game_state_after="GAME_OVER",
            levels_completed_after=1,
            intent="avoid_danger",
        ),
    ]
    episodes = [
        EpisodeRecord(
            game_id=game_id,
            episode_id="danger_ep",
            started_at="2026-05-30T10:00:00+00:00",
            ended_at="2026-05-30T10:00:30+00:00",
            n_steps=3,
            final_state="GAME_OVER",
            levels_completed=1,
            game_type_guess="move",
            objective_guess="reach next level then avoid trap",
        )
    ]
    return HumanPriorPack(game_id=game_id, steps=steps, episodes=episodes)


def test_seed_game_memory_resets_online_counters_but_keeps_priors():
    memory = GameMemory()
    stats = seed_game_memory(_simple_pack(), memory)

    assert stats["steps_replayed"] == 2
    assert stats["seeded_total_actions"] == 2
    assert stats["seeded_total_game_overs"] == 0
    assert stats["seeded_max_level"] == 1

    # Aggregated priors are preserved.
    assert memory.action_profiles["ACTION4"].times_tried == 1
    assert memory.action_profiles["ACTION6"].times_tried == 1
    assert memory.get_hypothesis("human::player moves right") == 0.7
    assert memory.get_effective_click_values() == {2}
    assert memory.level_action_sequences[0] == ["ACTION4"]
    assert memory.level_action_sequences[1] == ["ACTION6"]

    # Online counters/history start clean for the live agent.
    assert memory.total_actions == 0
    assert memory.total_resets == 0
    assert memory.total_game_overs == 0
    assert memory.current_level == 0
    assert memory.max_level_reached == 0
    assert memory.action_history == []
    assert memory.click_history == []
    assert memory._prev_grid is None
    assert memory._prev_grid_hash == 0


def test_parse_runner_output_backfills_agent_only_actions_from_old_logs():
    stdout = textwrap.dedent(
        """
        [game-memory] human replay: 1120 steps, 1 clicks, 21 hypotheses, human_hit_level=7 (counter reset → agent max_level now measures agent only)
        ------------------------------------------------------------------------------------------
          RESULT
        ------------------------------------------------------------------------------------------
          won:           False
          max level:     0
          total actions: 1871
          assoc wins:    0
          elapsed:       64.5s
          final goal:    [human] match yellow and purple shapes
        """
    )

    parsed = parse_runner_output(stdout)

    assert parsed["raw_total_actions"] == 1871
    assert parsed["human_steps_replayed"] == 1120
    assert parsed["agent_only_actions"] == 751


def test_parse_runner_output_preserves_fixed_logs_when_counters_are_reset():
    stdout = textwrap.dedent(
        """
        [game-memory] human replay: 1120 steps, 1 clicks, 21 hypotheses, human_hit_level=7 (counter reset → agent max_level now measures agent only)
        ------------------------------------------------------------------------------------------
          RESULT
        ------------------------------------------------------------------------------------------
          won:           False
          max level:     0
          total actions: 751
          assoc wins:    0
          elapsed:       64.5s
          final goal:    [human] match yellow and purple shapes
        """
    )

    parsed = parse_runner_output(stdout)

    assert parsed["raw_total_actions"] == 751
    assert parsed["human_steps_replayed"] == 1120
    assert parsed["agent_only_actions"] == 751


def test_compare_includes_task_program_hypothesis_setup():
    by_name = {setup.name: setup for setup in SETUPS}

    assert "task_program_hypothesis" in by_name
    setup = by_name["task_program_hypothesis"]
    assert setup.use_task_program is True
    assert setup.use_human_priors is False
    assert setup.sampler_stage == "v1"
    assert setup.planner_mode == "hypothesis"


def test_compare_includes_latent_program_v2_setups():
    by_name = {setup.name: setup for setup in SETUPS}

    assert "latent_program" in by_name
    latent = by_name["latent_program"]
    assert latent.use_task_program is False
    assert latent.use_human_priors is False
    assert latent.sampler_stage == "v2"
    assert latent.planner_mode == "prior"

    assert "task_program_latent" in by_name
    repaired = by_name["task_program_latent"]
    assert repaired.use_task_program is True
    assert repaired.use_human_priors is False
    assert repaired.sampler_stage == "v2"


def test_human_trace_memory_extracts_reusable_schema_from_pack():
    memory = HumanTraceMemory.from_prior_pack(_simple_pack())

    assert len(memory.traces) == 1
    trace = memory.traces[0]
    schema = trace.abstract_schema

    assert trace.goal_family == "click_puzzle"
    assert "move toward target" in schema.abstract_steps
    assert "interact when adjacent" in schema.abstract_steps
    assert schema.preferred_actions[:2] == ["ACTION4", "ACTION6"]


def test_human_trace_memory_aligns_non_reset_frames_with_actions():
    pack = _simple_pack()
    memory = HumanTraceMemory.from_prior_pack(pack)
    trace = memory.traces[0]

    assert trace.actions == ["ACTION4", "ACTION6"]
    assert trace.action_data == [None, {"x": 1, "y": 1}]
    assert trace.frames[0] == pack.steps[0].frame_before
    assert trace.frames[1] == pack.steps[1].frame_before
    assert trace.levels_completed == 1
    assert len(trace.success_prefixes) == 1
    assert trace.success_prefixes[0].actions == ["ACTION4", "ACTION6"]
    assert trace.success_prefixes[0].end_level == 1
    assert trace.danger_suffixes == []

    candidate = memory.aligned_trace_at(
        np.array(pack.steps[0].frame_before, dtype=np.int32),
        available_actions=["ACTION1", "ACTION4", "ACTION6"],
    )

    assert candidate is not None
    aligned_trace, idx = candidate
    assert aligned_trace.episode_id == "ep0"
    assert idx == 0

    assert memory.aligned_success_prefix_at(
        np.array(pack.steps[0].frame_before, dtype=np.int32),
        available_actions=["ACTION1", "ACTION4", "ACTION6"],
        current_level=7,
    ) is None


def test_human_trace_memory_distills_danger_suffix_and_recovery_frontier():
    pack = _danger_pack()
    memory = HumanTraceMemory.from_prior_pack(pack)
    trace = memory.traces[0]

    assert len(trace.success_prefixes) == 1
    assert trace.success_prefixes[0].actions == ["ACTION1", "ACTION2"]
    assert trace.success_prefixes[0].start_level == 0
    assert trace.success_prefixes[0].end_level == 1
    assert len(trace.danger_suffixes) == 1
    assert trace.danger_suffixes[0].actions == ["ACTION3"]

    frontier = memory.recovery_frontier_at(
        np.array(pack.steps[2].frame_before, dtype=np.int32),
    )

    assert frontier is not None
    assert frontier.avoid_actions == ["ACTION3"]
    assert frontier.danger_suffix.actions == ["ACTION3"]


def test_human_trace_memory_tiebreak_prefers_deeper_progress():
    memory = HumanTraceMemory()
    shared_frame = [[0, 1], [0, 0]]
    memory.add_trace(HumanTrace(
        game_id="ar25-e3c63847",
        episode_id="level4",
        frames=[shared_frame],
        actions=["ACTION1"],
        score=0.85,
        final_state="GAME_OVER",
        levels_completed=4,
    ))
    memory.add_trace(HumanTrace(
        game_id="ar25-e3c63847",
        episode_id="level7",
        frames=[shared_frame],
        actions=["ACTION3", "ACTION3"],
        score=0.85,
        final_state="GAME_OVER",
        levels_completed=7,
    ))

    trace, idx = memory.aligned_trace_at(
        np.array(shared_frame, dtype=np.int32),
        available_actions=["ACTION1", "ACTION3"],
    )

    assert trace.episode_id == "level7"
    assert idx == 0


def test_reasoning_loop_bootstraps_aligned_trace_before_explore():
    pack = _simple_pack()
    trace_memory = HumanTraceMemory.from_prior_pack(pack)
    loop = AdaptiveReasoningLoop(
        config=LoopConfig(
            ablation_stage="trajectory_memory",
            explore_budget=99,
            use_llm=False,
        )
    )
    loop.set_human_trace_memory(trace_memory)

    result = loop.step(
        current_grid=np.array(pack.steps[0].frame_before, dtype=np.int32),
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=["ACTION1", "ACTION4", "ACTION6"],
    )

    assert result["action"] == "ACTION4"
    assert result["phase"] == "execute"
    assert result["trajectory"].source == "human_success_prefix"
    assert loop.state.last_trajectory_debug["policy"] == "human_success_prefix"


def test_reasoning_loop_replans_at_recovery_frontier_and_avoids_danger_action():
    pack = _danger_pack()
    trace_memory = HumanTraceMemory.from_prior_pack(pack)
    loop = AdaptiveReasoningLoop(
        config=LoopConfig(
            ablation_stage="trajectory_memory",
            explore_budget=99,
            use_llm=False,
        )
    )
    loop.set_human_trace_memory(trace_memory)

    result = loop.step(
        current_grid=np.array(pack.steps[2].frame_before, dtype=np.int32),
        game_state="NOT_FINISHED",
        levels_completed=1,
        available_actions=["ACTION1", "ACTION2", "ACTION3"],
    )

    assert result["action"] != "ACTION3"
    assert result["trajectory_debug"]["avoid_actions"] == ["ACTION3"]
    assert result["trajectory_debug"]["danger_suffix"] == ["ACTION3"]
