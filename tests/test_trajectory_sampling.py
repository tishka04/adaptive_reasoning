import numpy as np
import pytest

from v4_1_reasoning_system.arc_agi.goal_pursuit import GoalContext
from v4_1_reasoning_system.arc_agi.state_describer import GameObservation
from v4_1_reasoning_system.arc_agi.trajectory_memory import (
    TrajectoryMemory,
    TrajectoryRecord,
)


def _fake_observation() -> GameObservation:
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[2, 2] = 3
    return GameObservation(
        grid_description="test grid",
        objects=[{"value": 3, "size": 1, "center_y": 2, "center_x": 2, "is_player": False}],
        player_info={"y": 1, "x": 1, "value": 2, "confidence": 0.8},
        action_semantics={"ACTION1": "move_up", "ACTION2": "move_down", "ACTION6": "click"},
        memory_summary={"states_visited": 1, "total_actions": 0, "max_level": 0, "total_game_overs": 0},
        raw_grid=grid,
        level=0,
        game_state="NOT_FINISHED",
        action_counter=0,
    )


def test_goal_recognizer_emits_first_class_hypothesis():
    from v4_1_reasoning_system.arc_agi.goal_decomposer import SubGoal
    from v4_1_reasoning_system.arc_agi.goal_recognizer import GoalRecognizer

    observation = _fake_observation()
    subgoal = SubGoal(
        id=1,
        description="switch control and match shapes",
        success_hint="role_switch",
        metadata={
            "task_program_id": "achieve_correspond_shapes",
            "expected_signal": "role_switch",
            "click_targets": [{"x": 2, "y": 2}],
        },
    )

    hypothesis = GoalRecognizer().predict(
        observation,
        current_goal=None,
        current_subgoal=subgoal,
        task_program=type("Prog", (), {"goal_family": "correspondence"})(),
    )

    assert hypothesis.family == "correspondence"
    assert hypothesis.source == "task_program"
    assert hypothesis.confidence > 0.8
    assert hypothesis.possible_player is not None


def test_human_prior_decay_and_growth_schedule():
    memory = TrajectoryMemory()
    cold = memory.human_prior_weight(0)
    late = memory.human_prior_weight(60)

    assert cold > late

    base = memory.human_prior_trust
    memory.update_prior_trust("human", prediction_match=0.1, progress_delta=0.0)
    assert memory.human_prior_trust == pytest.approx(base * 0.5)

    decayed = memory.human_prior_trust
    memory.update_prior_trust("human", prediction_match=0.9, progress_delta=0.2)
    assert memory.human_prior_trust > decayed


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_sampler_respects_horizon_and_uses_v0_sources():
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler

    class StubAssocMemory:
        def __init__(self):
            self._next = 1

        def retrieve_action(self, grid, available_actions, recent_actions=None, temperature=1.0):
            action = available_actions[self._next % len(available_actions)]
            self._next += 1
            return action, None

    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="navigation",
        objective_id="reach_target",
        progress_signals=["player moved"],
        anti_signals=[],
        source_confidence=0.8,
        preferred_actions=["ACTION1", "ACTION2"],
    )

    samples = TrajectorySampler().sample(
        observation=observation,
        goal_context=goal_context,
        memory=None,
        assoc_memory=StubAssocMemory(),
        trajectory_memory=TrajectoryMemory(),
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"],
        action_counter=5,
        k=10,
        horizon=4,
        state_embedding=np.ones(4, dtype=np.float32),
    )

    assert len(samples) == 10
    assert all(len(sample.actions) == 4 for sample in samples)
    assert {sample.source for sample in samples} == {"heuristic", "random"}


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_heuristic_sampler_starts_from_preferred_actions():
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler

    class StubAssocMemory:
        def retrieve_action(self, grid, available_actions, recent_actions=None, temperature=1.0):
            return available_actions[-1], None

    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="navigation",
        objective_id="reach_target",
        progress_signals=["player moved"],
        anti_signals=[],
        source_confidence=0.8,
        preferred_actions=["ACTION2", "ACTION3"],
    )

    samples = TrajectorySampler().sample(
        observation=observation,
        goal_context=goal_context,
        memory=None,
        assoc_memory=StubAssocMemory(),
        trajectory_memory=TrajectoryMemory(),
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        action_counter=5,
        k=8,
        horizon=3,
        state_embedding=np.ones(4, dtype=np.float32),
    )

    heuristic_samples = [sample for sample in samples if sample.source == "heuristic"]
    assert heuristic_samples
    assert all(sample.actions[0] in {"ACTION2", "ACTION3"} for sample in heuristic_samples)


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_v1_sampler_adds_prior_source_from_task_program_and_human_trace():
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler

    class StubAssocMemory:
        def retrieve_action(self, grid, available_actions, recent_actions=None, temperature=1.0):
            return available_actions[-1], None

    class StubSchema:
        preferred_actions = ["ACTION5", "ACTION2"]
        score = 0.9

    class StubTrace:
        actions = ["ACTION5", "ACTION2", "ACTION2"]
        abstract_schema = StubSchema()
        score = 1.0

    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="correspondence",
        objective_id="achieve_correspond_shapes",
        progress_signals=["match_shapes"],
        anti_signals=[],
        source_confidence=0.9,
        human_prior_weight=0.45,
        preferred_actions=["ACTION5", "ACTION2", "ACTION3"],
    )

    samples = TrajectorySampler(stage="v1").sample(
        observation=observation,
        goal_context=goal_context,
        memory=None,
        assoc_memory=StubAssocMemory(),
        trajectory_memory=TrajectoryMemory(),
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION5"],
        human_traces=[StubTrace()],
        action_counter=5,
        k=10,
        horizon=3,
        state_embedding=np.ones(4, dtype=np.float32),
    )

    prior_samples = [sample for sample in samples if sample.source == "prior"]
    assert prior_samples
    assert any(sample.actions[0] == "ACTION5" for sample in prior_samples)
    assert any(sample.metadata.get("prior_kind") == "human_trace" for sample in prior_samples)
    assert any(float(sample.metadata.get("human_compatibility", 0.0)) > 0.0 for sample in prior_samples)


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_v2_sampler_emits_latent_program_macro_sequences():
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler

    class StubAssocMemory:
        def retrieve_action(self, grid, available_actions, recent_actions=None, temperature=1.0):
            return available_actions[-1], None

    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="correspondence",
        objective_id="achieve_level2_correspondence",
        progress_signals=["level_advance"],
        anti_signals=[],
        source_confidence=0.9,
        preferred_actions=["ACTION5"],
        metadata={
            "latent_task_program_confidence": 0.72,
            "latent_preferred_sequences": [
                ["ACTION2", "ACTION2", "ACTION3"],
                ["ACTION3", "ACTION2", "ACTION2"],
            ],
        },
    )

    samples = TrajectorySampler(stage="v2").sample(
        observation=observation,
        goal_context=goal_context,
        memory=None,
        assoc_memory=StubAssocMemory(),
        trajectory_memory=TrajectoryMemory(),
        available_actions=["ACTION2", "ACTION3", "ACTION5"],
        action_counter=20,
        k=8,
        horizon=4,
        state_embedding=np.ones(4, dtype=np.float32),
    )

    latent_samples = [sample for sample in samples if sample.source == "latent_program"]
    assert latent_samples
    assert latent_samples[0].actions[:3] == ["ACTION2", "ACTION2", "ACTION3"]
    assert latent_samples[0].metadata["prior_kind"] == "latent_program"
    assert latent_samples[0].metadata["latent_macro_sequence"] == [
        "ACTION2",
        "ACTION2",
        "ACTION3",
    ]


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_hypothesis_planner_samples_from_induced_action_beliefs():
    from v4_1_reasoning_system.arc_agi.game_memory import ActionProfile, GameMemory
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler

    memory = GameMemory()
    move = ActionProfile(action_name="ACTION1", times_tried=5, times_changed_grid=5, times_moved_player=5)
    move.displacements = [(-1.0, 0.0), (-1.0, 0.0)]
    switch = ActionProfile(action_name="ACTION5", times_tried=4, times_changed_grid=3)
    memory.action_profiles["ACTION1"] = move
    memory.action_profiles["ACTION5"] = switch

    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="correspondence",
        objective_id="test_switch_then_move",
        progress_signals=["role_switch"],
        anti_signals=[],
        source_confidence=0.8,
    )

    samples = TrajectorySampler(planner_mode="hypothesis").sample(
        observation=observation,
        goal_context=goal_context,
        memory=memory,
        assoc_memory=None,
        trajectory_memory=TrajectoryMemory(),
        available_actions=["ACTION1", "ACTION2", "ACTION5"],
        action_counter=5,
        k=8,
        horizon=3,
        state_embedding=np.ones(4, dtype=np.float32),
    )

    hypothesis_samples = [sample for sample in samples if sample.source == "hypothesis"]
    assert hypothesis_samples
    assert any(sample.metadata.get("hypothesis_kind") == "control_switch" for sample in hypothesis_samples)
    assert all("hypothesis_belief_summary" in sample.metadata for sample in hypothesis_samples)


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_hypothesis_planner_rotates_away_from_recently_repeated_leads():
    from v4_1_reasoning_system.arc_agi.game_memory import ActionProfile, GameMemory
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler

    memory = GameMemory()
    repeated_move = ActionProfile(
        action_name="ACTION1",
        times_tried=8,
        times_changed_grid=8,
        times_moved_player=8,
    )
    repeated_move.displacements = [(-1.0, 0.0)] * 4
    switch_probe = ActionProfile(
        action_name="ACTION5",
        times_tried=1,
        times_changed_grid=1,
    )
    memory.action_profiles["ACTION1"] = repeated_move
    memory.action_profiles["ACTION5"] = switch_probe

    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="correspondence",
        objective_id="test_avoid_repeated_probe",
        progress_signals=["role_switch"],
        anti_signals=[],
        source_confidence=0.8,
        metadata={"recent_actions": ["ACTION1", "ACTION1", "ACTION1", "ACTION1"]},
    )

    samples = TrajectorySampler(planner_mode="hypothesis").sample(
        observation=observation,
        goal_context=goal_context,
        memory=memory,
        assoc_memory=None,
        trajectory_memory=TrajectoryMemory(),
        available_actions=["ACTION1", "ACTION2", "ACTION5"],
        action_counter=20,
        k=8,
        horizon=3,
        state_embedding=np.ones(4, dtype=np.float32),
    )

    hypothesis_samples = [sample for sample in samples if sample.source == "hypothesis"]
    assert hypothesis_samples
    assert hypothesis_samples[0].actions[0] != "ACTION1"
    assert any(sample.actions[0] == "ACTION5" for sample in hypothesis_samples)
    repeated_samples = [
        sample for sample in hypothesis_samples
        if sample.actions and sample.actions[0] == "ACTION1"
    ]
    assert repeated_samples
    assert repeated_samples[0].metadata["hypothesis_recent_penalty"] > 0.9
    assert all("hypothesis_experiment_score" in sample.metadata for sample in hypothesis_samples)


def test_build_goal_context_keeps_generic_preferences_without_level_hacks():
    from v4_1_reasoning_system.arc_agi.goal_decomposer import SubGoal
    from v4_1_reasoning_system.arc_agi.goal_recognizer import GoalHypothesis
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig

    loop = AdaptiveReasoningLoop(config=LoopConfig())
    observation = _fake_observation()
    observation.level = 1
    loop.state.current_subgoal = SubGoal(
        id=2,
        description="make progress on the current correspondence objective",
        success_hint="level_advance",
        metadata={
            "task_program_id": "achieve_level2_correspondence",
            "expected_signal": "level_advance",
            "prefer_actions": ["ACTION2", "ACTION3", "ACTION5", "ACTION2"],
            "program_level": 2,
        },
    )

    goal_context = loop._build_goal_context(
        observation,
        GoalHypothesis(
            family="correspondence",
            confidence=0.9,
            source="task_program",
        ),
    )

    assert goal_context.preferred_actions == ["ACTION2", "ACTION3", "ACTION5"]
    assert goal_context.human_prior_weight == 0.0
    assert "force_first_action" not in goal_context.metadata
    assert "suppress_switch_first" not in goal_context.metadata
    assert "confirmed_switch_action" not in goal_context.metadata
    assert "allow_compositional_switch" not in goal_context.metadata
    assert "required_movement_burst" not in goal_context.metadata


def test_build_goal_context_restores_human_prior_weight_in_v1():
    from v4_1_reasoning_system.arc_agi.goal_decomposer import SubGoal
    from v4_1_reasoning_system.arc_agi.goal_recognizer import GoalHypothesis
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig

    loop = AdaptiveReasoningLoop(config=LoopConfig(sampler_stage="v1"))
    observation = _fake_observation()
    loop.state.current_subgoal = SubGoal(
        id=1,
        description="follow the best current task prior",
        success_hint="player moved",
        metadata={
            "task_program_id": "reach_target",
            "prefer_actions": ["ACTION2", "ACTION3"],
        },
    )

    goal_context = loop._build_goal_context(
        observation,
        GoalHypothesis(
            family="navigation",
            confidence=0.8,
            source="task_program",
        ),
    )

    assert goal_context.preferred_actions == ["ACTION2", "ACTION3"]
    assert goal_context.human_prior_weight > 0.0


def test_trajectory_heuristics_penalizes_repeated_control_switch_lead():
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import SampledTrajectory

    class Scalar:
        def __init__(self, value: float):
            self.value = value

        def item(self) -> float:
            return self.value

    class Aux:
        progress_prob = Scalar(0.10)
        risk_prob = Scalar(0.10)
        novelty_score = Scalar(0.10)

    loop = AdaptiveReasoningLoop(config=LoopConfig(sampler_stage="v1"))
    goal_context = GoalContext(
        goal_family="correspondence",
        objective_id="achieve_level2_correspondence",
        progress_signals=["levels_completed increments"],
        anti_signals=[],
        source_confidence=0.85,
        human_prior_weight=0.5,
        preferred_actions=["ACTION2", "ACTION3", "ACTION5"],
        metadata={
            "expected_signal": "level_advance",
            "recent_actions": ["ACTION5"],
            "control_switch_actions": ["ACTION5"],
            "movement_actions": ["ACTION2", "ACTION3"],
        },
    )
    switch_first = SampledTrajectory(
        actions=["ACTION5", "ACTION2", "ACTION2"],
        source="prior",
        metadata={"preferred_fraction": 1.0, "human_compatibility": 0.8},
    )
    movement_first = SampledTrajectory(
        actions=["ACTION2", "ACTION2", "ACTION3"],
        source="prior",
        metadata={"preferred_fraction": 1.0, "human_compatibility": 0.8},
    )

    switch_scores = loop._trajectory_heuristics(
        _fake_observation(), goal_context, switch_first, Aux()
    )
    movement_scores = loop._trajectory_heuristics(
        _fake_observation(), goal_context, movement_first, Aux()
    )

    assert switch_first.metadata["control_switch_repeat_penalty"] > 0.5
    assert movement_first.metadata["control_switch_followup_bonus"] > 0.0
    assert switch_scores["risk"] > movement_scores["risk"]
    assert switch_scores["goal_progress"] < movement_scores["goal_progress"]
    assert switch_scores["human_compatibility"] == pytest.approx(0.25)


def test_latent_task_program_generator_overrides_weak_static_prior():
    from v4_1_reasoning_system.arc_agi.latent_task_programmer import (
        LatentTaskProgramGenerator,
    )
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import SampledTrajectory

    goal_context = GoalContext(
        goal_family="correspondence",
        objective_id="achieve_level2_correspondence",
        progress_signals=["levels_completed increments"],
        anti_signals=[],
        source_confidence=0.8,
        preferred_actions=["ACTION5"],
        metadata={
            "from_task_program": True,
            "expected_signal": "level_advance",
            "prefer_actions": ["ACTION5"],
        },
    )
    weak_prior = SampledTrajectory(
        actions=["ACTION5", "ACTION5", "ACTION2"],
        source="prior",
        risk=0.4,
        energy=0.7,
        score=-0.5,
    )
    scorer_winner = SampledTrajectory(
        actions=["ACTION2", "ACTION1", "ACTION2"],
        source="random",
        risk=0.05,
        energy=0.2,
        score=0.2,
    )

    program = LatentTaskProgramGenerator().build(
        goal_context=goal_context,
        trajectories=[weak_prior, scorer_winner],
        selected=scorer_winner,
        top_indices=[1, 0],
    )

    assert program is not None
    assert program.preferred_actions[:2] == ["ACTION2", "ACTION1"]
    assert program.preferred_sequences[0] == ["ACTION2", "ACTION1", "ACTION2"]
    assert program.metadata["latent_override_static_program"] is True


def test_v2_goal_context_consumes_installed_latent_task_program():
    from v4_1_reasoning_system.arc_agi.goal_decomposer import SubGoal
    from v4_1_reasoning_system.arc_agi.goal_recognizer import GoalHypothesis
    from v4_1_reasoning_system.arc_agi.latent_task_programmer import LatentTaskProgram
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig

    loop = AdaptiveReasoningLoop(config=LoopConfig(sampler_stage="v2"))
    loop.state.current_subgoal = SubGoal(
        id=2,
        description="static program says switch",
        success_hint="levels_completed increments",
        metadata={
            "from_task_program": True,
            "task_program_id": "achieve_level2_correspondence",
            "expected_signal": "level_advance",
            "prefer_actions": ["ACTION5"],
        },
    )
    loop._install_latent_task_program(
        LatentTaskProgram(
            objective_id="latent::achieve_level2_correspondence",
            description="prefer scorer movement",
            preferred_actions=["ACTION2", "ACTION3"],
            preferred_sequences=[["ACTION2", "ACTION2", "ACTION3"]],
            avoid_actions=["ACTION7"],
            confidence=0.7,
            expected_signal="level_advance",
            metadata={"latent_override_static_program": True},
        )
    )

    goal_context = loop._build_goal_context(
        _fake_observation(),
        GoalHypothesis(
            family="correspondence",
            confidence=0.8,
            source="trajectory_scorer",
        ),
    )

    assert goal_context.preferred_actions[:3] == ["ACTION2", "ACTION3", "ACTION5"]
    assert goal_context.metadata["latent_preferred_sequences"][0] == [
        "ACTION2",
        "ACTION2",
        "ACTION3",
    ]
    assert "latent_avoid: ACTION7" in goal_context.anti_signals
    assert loop.state.current_subgoal.metadata["prefer_actions"][:2] == ["ACTION2", "ACTION3"]


def test_runtime_human_trace_sampling_is_disabled_in_v0():
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig

    loop = AdaptiveReasoningLoop(config=LoopConfig())

    assert loop._allow_runtime_human_trace_sampling(0) is False
    assert loop._allow_runtime_human_trace_sampling(1) is False


def test_runtime_human_trace_sampling_is_enabled_in_v1():
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig

    loop = AdaptiveReasoningLoop(config=LoopConfig(sampler_stage="v1"))

    assert loop._allow_runtime_human_trace_sampling(0) is True
    assert loop._allow_runtime_human_trace_sampling(1) is True


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_trajectory_scoring_prefers_goal_progress_over_source_similarity():
    import torch

    from v4_1_reasoning_system.arc_agi.energy_scorer import GameEnergyScorer
    from v4_1_reasoning_system.arc_agi.game_world_model import (
        GameAuxPredictions,
        WorldModelConfig,
    )
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import SampledTrajectory

    cfg = WorldModelConfig(latent_dim=8, strategy_dim=4, hidden_dim=16, device="cpu")
    scorer = GameEnergyScorer(cfg)
    for param in scorer.ebm.parameters():
        torch.nn.init.constant_(param, 0.0)
    with torch.no_grad():
        scorer.ebm.heuristic_weights.zero_()

    z_t = torch.zeros(1, 8)
    strategy_embs = torch.zeros(1, 2, 4)
    z_hats = torch.zeros(1, 2, 8)
    aux_list = [
        GameAuxPredictions(
            progress_prob=torch.tensor([0.1]),
            risk_prob=torch.tensor([0.1]),
            novelty_score=torch.tensor([0.1]),
        ),
        GameAuxPredictions(
            progress_prob=torch.tensor([0.1]),
            risk_prob=torch.tensor([0.1]),
            novelty_score=torch.tensor([0.1]),
        ),
    ]

    low_progress = SampledTrajectory(
        actions=["ACTION1", "ACTION2"],
        source="random",
        metadata={"goal_progress": 0.2, "novelty": 0.1, "risk": 0.1},
    )
    high_progress = SampledTrajectory(
        actions=["ACTION1", "ACTION2"],
        source="heuristic",
        metadata={"goal_progress": 0.8, "novelty": 0.1, "risk": 0.1},
    )

    decision = scorer.score_trajectories(
        z_t,
        [low_progress, high_progress],
        strategy_embs,
        z_hats,
        aux_list,
    )

    assert decision.selected_trajectory is high_progress


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_execute_only_commits_first_action_and_forces_replanning():
    from v4_1_reasoning_system.arc_agi.actioner import ActionResult
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import SampledTrajectory

    loop = AdaptiveReasoningLoop(config=LoopConfig())
    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="navigation",
        objective_id="reach_target",
        progress_signals=["player moved"],
        anti_signals=[],
        source_confidence=0.7,
    )
    trajectory = SampledTrajectory(
        actions=["ACTION1", "ACTION2", "ACTION3"],
        goal_context=goal_context,
        source="heuristic",
    )
    strategy = trajectory.to_strategy()

    def _fake_act(strategy, subgoal, observation, memory, available_actions):
        return ActionResult(action=strategy.action_plan[0], action_data=None, reason="test")

    loop.actioner.act = _fake_act
    loop.state.current_strategy = strategy
    loop.state.current_trajectory = trajectory
    loop.state.current_goal_context = goal_context

    result = loop._phase_execute(observation, ["ACTION1", "ACTION2", "ACTION3"])

    assert result["action"] == "ACTION1"
    assert result["trajectory"] is trajectory
    assert loop.state.current_strategy is None
    assert loop.state.current_trajectory is None


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_actioner_resets_sequence_index_for_each_new_strategy():
    from v4_1_reasoning_system.arc_agi.actioner import Actioner
    from v4_1_reasoning_system.arc_agi.goal_decomposer import SubGoal
    from v4_1_reasoning_system.arc_agi.strategy_generator import GameStrategy, StrategyType

    actioner = Actioner()
    observation = _fake_observation()
    subgoal = SubGoal(id=1, description="test", success_hint="changed")
    strategy_a = GameStrategy(
        strategy_type=StrategyType.SEQUENCE_ACTIONS,
        description="first plan",
        action_plan=["ACTION5", "ACTION1", "ACTION2"],
        rationale="test",
    )
    strategy_b = GameStrategy(
        strategy_type=StrategyType.SEQUENCE_ACTIONS,
        description="second plan",
        action_plan=["ACTION3", "ACTION4", "ACTION5"],
        rationale="test",
    )

    result_a = actioner.act(
        strategy_a,
        subgoal,
        observation,
        memory=None,
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
    )
    result_b = actioner.act(
        strategy_b,
        subgoal,
        observation,
        memory=None,
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
    )

    assert result_a.action == "ACTION5"
    assert result_b.action == "ACTION3"


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_failed_prefixes_are_not_replayed_unchanged():
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler

    class StubAssocMemory:
        def retrieve_action(self, grid, available_actions, recent_actions=None, temperature=1.0):
            return available_actions[0], None

    observation = _fake_observation()
    goal_context = GoalContext(
        goal_family="navigation",
        objective_id="reach_target",
        progress_signals=["player moved"],
        anti_signals=[],
        source_confidence=0.8,
        preferred_actions=["ACTION1", "ACTION2"],
    )
    memory = TrajectoryMemory()
    failed = ["ACTION1", "ACTION2"]
    seed = TrajectoryRecord(
        goal_family="navigation",
        objective_id="reach_target",
        actions=failed,
        source="agent",
        score=0.7,
        progress_delta=0.7,
        prediction_match=0.8,
        success=True,
    )
    memory.records.append(seed)
    memory.fragments.append(seed)
    memory.remember_failed_prefix(failed)

    samples = TrajectorySampler().sample(
        observation=observation,
        goal_context=goal_context,
        memory=None,
        assoc_memory=StubAssocMemory(),
        trajectory_memory=memory,
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        action_counter=5,
        k=8,
        horizon=2,
        state_embedding=np.ones(4, dtype=np.float32),
    )

    assert all(
        tuple(sample.actions) != tuple(failed) or sample.source == "random"
        for sample in samples
    )


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_v0_does_not_start_continuation_even_on_later_levels():
    from v4_1_reasoning_system.arc_agi.actioner import ActionResult
    from v4_1_reasoning_system.arc_agi.goal_recognizer import GoalHypothesis
    from v4_1_reasoning_system.arc_agi.reasoning_loop import AdaptiveReasoningLoop, LoopConfig
    from v4_1_reasoning_system.arc_agi.trajectory_sampler import SampledTrajectory

    loop = AdaptiveReasoningLoop(config=LoopConfig())
    observation = _fake_observation()
    observation.level = 1
    goal_context = GoalContext(
        goal_family="correspondence",
        objective_id="achieve_correspond_shapes",
        progress_signals=["match_shapes"],
        anti_signals=[],
        source_confidence=0.85,
        metadata={"task_program_id": "achieve_correspond_shapes"},
    )
    goal_hypothesis = GoalHypothesis(
        family="correspondence",
        confidence=0.9,
        possible_player={"x": 1, "y": 1},
        source="task_program",
    )
    trajectory = SampledTrajectory(
        actions=["ACTION5", "ACTION1", "ACTION2"],
        goal_context=goal_context,
        source="heuristic",
        metadata={"goal_progress": 0.8},
    )

    loop.actioner.act = lambda strategy, subgoal, observation, memory, available_actions: ActionResult(
        action="ACTION5", action_data=None, reason="test"
    )
    loop.state.current_strategy = trajectory.to_strategy()
    loop.state.current_trajectory = trajectory
    loop.state.current_goal_context = goal_context
    loop.state.current_goal_hypothesis = goal_hypothesis
    loop.state.last_observation = observation

    loop._phase_execute(observation, ["ACTION1", "ACTION2", "ACTION5"])

    assert loop._allow_continuation(1) is False
    assert loop.state.current_continuation is None
    assert loop.state.pending_continuation is None
