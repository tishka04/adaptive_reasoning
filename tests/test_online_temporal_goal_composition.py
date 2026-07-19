"""State-conditioned temporal composition and terminal-credit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation, build_transition_record
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.online_temporal_goal_composition import (
    OnlineTemporalGoalComposer,
    TemporalPlanStatus,
)
from theory.online_terminal_objective import OnlineTerminalObjectiveStore
from theory.unified_cognitive_controller import UnifiedCognitiveController


def _candidate(
    objective_id: str,
    family: str,
    *,
    source: int | None,
    target: int | None,
    predicate: str,
    priority: float = 0.8,
) -> GeneratedGoalHypothesis:
    return GeneratedGoalHypothesis(
        objective_id=objective_id,
        family=family,
        source_color=source,
        target_color=target,
        predicate=predicate,
        supporting_rule_keys=(),
        supporting_actions=("ACTION6",),
        generation_reason="synthetic_temporal_test",
        prior_priority=priority,
    )


def _grid(source_positions=((1, 1), (1, 3), (3, 1), (3, 3))):
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[6, 6] = 4
    for position in source_positions:
        grid[position] = 3
    return grid


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION6"],
        infer_players=False,
    )


def _update(
    before: np.ndarray,
    after: np.ndarray,
    *,
    levels_before: int = 0,
    levels_after: int = 0,
    game_state_after: str = "NOT_FINISHED",
):
    record = build_transition_record(
        action="ACTION6",
        action_args={"x": 1, "y": 1},
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION6"],
        levels_completed_before=levels_before,
        levels_completed_after=levels_after,
        game_state_after=game_state_after,
    )
    return SimpleNamespace(record=record)


def _convert_store() -> tuple[OnlineTerminalObjectiveStore, str]:
    store = OnlineTerminalObjectiveStore()
    objective_id = "goal-convert-3-to-4"
    store.register_generated(_candidate(
        objective_id,
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    ))
    return store, objective_id


def test_composer_turns_a_large_deficit_into_ordered_distance_thresholds():
    store, objective_id = _convert_store()
    composer = OnlineTemporalGoalComposer()

    plans = composer.compose(_observation(_grid()), store)

    plan = next(
        item for item in plans
        if item.generation_reason == "live_distance_threshold_decomposition"
    )
    assert [step.objective_id for step in plan.steps] == [
        objective_id,
        objective_id,
    ]
    assert [step.target_distance for step in plan.steps] == [2.0, 0.0]
    assert all(
        step.guard == "distance_measurable_and_above_target"
        for step in plan.steps
    )


def test_composer_orders_creation_before_a_goal_whose_color_is_missing():
    store = OnlineTerminalObjectiveStore()
    conversion = _candidate(
        "goal-convert-5-to-4",
        "convert",
        source=5,
        target=4,
        predicate="source_target_color_transform",
        priority=0.9,
    )
    appearance = _candidate(
        "goal-appear-3-4",
        "appear",
        source=3,
        target=4,
        predicate="adjacent_to",
        priority=0.8,
    )
    store.register_generated(conversion)
    store.register_generated(appearance)
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[2, 2] = 3
    grid[5, 5] = 5

    plans = OnlineTemporalGoalComposer().compose(_observation(grid), store)

    dependent = next(
        item for item in plans
        if item.generation_reason == "live_objective_dependency_composition"
    )
    assert dependent.target_objective_id == appearance.objective_id
    assert [step.objective_id for step in dependent.steps] == [
        conversion.objective_id,
        appearance.objective_id,
    ]
    assert dependent.steps[0].role == "create_missing_precondition"


def test_executor_reobserves_distance_and_advances_one_threshold_at_a_time():
    store, _ = _convert_store()
    composer = OnlineTemporalGoalComposer()
    before = _grid()
    composer.compose(_observation(before), store)
    first = composer.select_step(_observation(before), store)
    assert first is not None
    assert first.step_index == 0
    assert first.target_distance == 2.0

    after = before.copy()
    after[1, 1] = 4
    after[1, 3] = 4
    outcome = composer.observe_transition(
        _update(before, after),
        store=store,
        plan_id=first.plan_id,
        step_index=first.step_index,
        objective_id=first.objective_id,
        target_distance=first.target_distance,
        intervention_id="ACTION6::first-threshold",
        context_signature="threshold-context",
    )

    assert outcome["distance_reduction"] == 2.0
    assert outcome["step_completed"] is True
    second = composer.select_step(_observation(after), store)
    assert second is not None
    assert second.plan_id == first.plan_id
    assert second.step_index == 1
    assert second.target_distance == 0.0


def test_executor_abandons_and_allows_replanning_after_measured_stall():
    store, _ = _convert_store()
    composer = OnlineTemporalGoalComposer(max_stalls_per_step=2)
    grid = _grid()
    composer.compose(_observation(grid), store)
    selection = composer.select_step(_observation(grid), store)
    assert selection is not None

    first = composer.observe_transition(
        _update(grid, grid.copy()),
        store=store,
        plan_id=selection.plan_id,
        step_index=selection.step_index,
        objective_id=selection.objective_id,
        target_distance=selection.target_distance,
        intervention_id="ACTION6::noop-1",
    )
    assert first["plan_abandoned"] is False
    again = composer.select_step(_observation(grid), store)
    assert again is not None
    second = composer.observe_transition(
        _update(grid, grid.copy()),
        store=store,
        plan_id=again.plan_id,
        step_index=again.step_index,
        objective_id=again.objective_id,
        target_distance=again.target_distance,
        intervention_id="ACTION6::noop-2",
    )

    assert second["plan_abandoned"] is True
    assert composer.active_plan_id == ""
    plan = composer.plan(selection.plan_id)
    assert plan is not None
    assert plan.stalls == 2
    assert plan.abandonment_reasons == {"state_condition_stalled": 1}


def test_game_over_quarantines_the_sequence_not_the_goal_hypothesis():
    store, objective_id = _convert_store()
    composer = OnlineTemporalGoalComposer()
    grid = _grid()
    composer.compose(_observation(grid), store)
    selection = composer.select_step(_observation(grid), store)
    assert selection is not None
    changed = grid.copy()
    changed[1, 1] = 4

    composer.observe_transition(
        _update(grid, changed, game_state_after="GAME_OVER"),
        store=store,
        plan_id=selection.plan_id,
        step_index=selection.step_index,
        objective_id=selection.objective_id,
        target_distance=selection.target_distance,
        intervention_id="ACTION6::dangerous",
    )

    plan = composer.plan(selection.plan_id)
    objective = store.objective(objective_id)
    assert plan is not None and objective is not None
    assert plan.unsafe_failures == 1
    assert plan.dangerous_interventions == {"ACTION6::dangerous": 1}
    assert plan.status == TemporalPlanStatus.CANDIDATE
    assert objective.status.value == "candidate"


def test_local_sequence_completion_gets_no_value_without_terminal_learning():
    store, _ = _convert_store()
    composer = OnlineTemporalGoalComposer()
    before = _grid()
    composer.compose(_observation(before), store)
    first = composer.select_step(_observation(before), store)
    assert first is not None
    middle = before.copy()
    middle[1, 1] = middle[1, 3] = 4
    composer.observe_transition(
        _update(before, middle),
        store=store,
        plan_id=first.plan_id,
        step_index=first.step_index,
        objective_id=first.objective_id,
        target_distance=first.target_distance,
    )
    second = composer.select_step(_observation(middle), store)
    assert second is not None
    after = middle.copy()
    after[3, 1] = after[3, 3] = 4
    outcome = composer.observe_transition(
        _update(middle, after),
        store=store,
        plan_id=second.plan_id,
        step_index=second.step_index,
        objective_id=second.objective_id,
        target_distance=second.target_distance,
        context_signature="local-only",
    )

    plan = composer.plan(first.plan_id)
    assert plan is not None
    assert outcome["plan_completed"] is True
    assert plan.local_completions == 1
    assert plan.terminal_support == 0
    assert plan.status == TemporalPlanStatus.CANDIDATE
    composer.start_branch()
    assert plan.nonterminal_completions == 1
    assert plan.terminal_contradictions == 1


def test_plan_requires_two_independent_terminal_contexts_for_support():
    store, _ = _convert_store()
    composer = OnlineTemporalGoalComposer(minimum_terminal_support=2)
    plan_id = ""
    for branch, context in enumerate(("terminal-a", "terminal-b")):
        before = _grid()
        composer.compose(_observation(before), store)
        first = composer.select_step(_observation(before), store)
        assert first is not None
        plan_id = first.plan_id
        middle = before.copy()
        middle[1, 1] = middle[1, 3] = 4
        composer.observe_transition(
            _update(before, middle),
            store=store,
            plan_id=first.plan_id,
            step_index=first.step_index,
            objective_id=first.objective_id,
            target_distance=first.target_distance,
            context_signature=f"{context}-milestone",
        )
        second = composer.select_step(_observation(middle), store)
        assert second is not None
        after = middle.copy()
        after[3, 1] = after[3, 3] = 4
        outcome = composer.observe_transition(
            _update(middle, after, levels_after=1),
            store=store,
            plan_id=second.plan_id,
            step_index=second.step_index,
            objective_id=second.objective_id,
            target_distance=second.target_distance,
            intervention_id=f"ACTION6::{context}",
            context_signature=context,
        )
        assert outcome["terminal_credited_plans"] == [plan_id]
        plan = composer.plan(plan_id)
        assert plan is not None
        if branch == 0:
            assert plan.status == TemporalPlanStatus.NEEDS_CONTRAST
            composer.start_branch()

    plan = composer.plan(plan_id)
    assert plan is not None
    assert plan.terminal_support == 2
    assert plan.status == TemporalPlanStatus.TERMINAL_SUPPORTED


def test_controller_emits_auditable_temporal_step_then_replans_from_new_state():
    controller = UnifiedCognitiveController(
        "synthetic-temporal-controller",
        available_actions=["ACTION6"],
    )
    store = controller.terminal_objectives
    store.register_generated(_candidate(
        "goal-convert-3-to-4",
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
        priority=2.0,
    ))
    before = _grid()

    first = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )

    assert first.source == "temporal_subgoal_probe"
    assert first.temporal_plan_id
    assert first.temporal_step_index == 0
    assert first.temporal_step_count == 2
    assert first.temporal_step_target_distance == 2.0
    assert first.objective_distance == 4.0
    changed = before.copy()
    changed[1, 1] = 4
    controller.observe_transition(
        action=first.action_name,
        action_data=first.action_data,
        grid_before=before,
        grid_after=changed,
        available_actions=["ACTION6"],
    )
    continued = controller.select_action(
        current_grid=changed,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    assert continued.temporal_plan_id == first.temporal_plan_id
    assert continued.temporal_step_index == 0
    assert continued.objective_distance == 3.0
