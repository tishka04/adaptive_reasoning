"""Active generation, discrimination, ambiguity, and ablation tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from v3.schemas import PlayerHypothesis

from theory.generic_discriminating_experiment_designer import (
    DesignedExperimentAction,
)
from theory.live_transition_loop import build_observation, build_transition_record
from theory.online_goal_hypothesis import (
    GeneratedGoalHypothesis,
    GoalHypothesisGenerator,
    ObjectiveDiscriminatingExperimentDesigner,
    intervention_id,
)
from theory.online_terminal_objective import OnlineTerminalObjectiveStore
from theory.promoted_relational_rule import PromotedRelationalRule
from theory.unified_cognitive_controller import UnifiedCognitiveController


def _observation() -> object:
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[1, 1] = 2
    grid[3, 3] = 3
    grid[3, 8] = 4
    grid[8, 8] = 5
    observation = build_observation(
        grid,
        available_actions=["ACTION1", "ACTION2", "ACTION6"],
    )
    observation.player_candidates = [
        PlayerHypothesis(value=2, position=(1, 1), confidence=1.0)
    ]
    return observation


def _candidate(
    objective_id: str,
    family: str,
    *,
    source: int | None,
    target: int | None,
    predicate: str,
    actions: tuple[str, ...] = ("ACTION6",),
    priority: float = 0.5,
) -> GeneratedGoalHypothesis:
    return GeneratedGoalHypothesis(
        objective_id=objective_id,
        family=family,
        source_color=source,
        target_color=target,
        predicate=predicate,
        supporting_rule_keys=(),
        supporting_actions=actions,
        generation_reason="synthetic_live_counterfactual",
        prior_priority=priority,
    )


def _update(
    before: np.ndarray,
    after: np.ndarray,
    *,
    levels_after: int = 0,
    game_state_after: str = "NOT_FINISHED",
):
    record = build_transition_record(
        action="ACTION6",
        action_args={"x": 2, "y": 2},
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION6"],
        levels_completed_before=0,
        levels_completed_after=levels_after,
        game_state_after=game_state_after,
    )
    return SimpleNamespace(record=record)


def test_generator_builds_all_five_measurable_goal_families_with_bounds():
    observation = _observation()
    preserve = PromotedRelationalRule(
        action="ACTION6",
        family="relation",
        predicate="aligned_with",
        source_color=3,
        target_color=4,
        expected_outcome="preserved",
        source_prediction_key="preserved-mechanic-not-goal",
        support=2,
        independent_contexts={"a", "b"},
    )
    generator = GoalHypothesisGenerator(
        max_candidates_total=10,
        max_candidates_per_family=2,
    )

    candidates = generator.generate(
        observation=observation,
        rules=[preserve],
        available_actions=["ACTION1", "ACTION2", "ACTION6"],
    )

    assert {candidate.family for candidate in candidates} == {
        "appear",
        "break",
        "exhaust",
        "reach",
        "convert",
    }
    assert len(candidates) <= 10
    assert all(candidate.supporting_actions for candidate in candidates)
    assert not any(candidate.family == "preserved" for candidate in candidates)


def test_designer_chooses_an_intervention_that_separates_two_goals():
    observation = _observation()
    store = OnlineTerminalObjectiveStore(max_probe_actions_total=8)
    convert = _candidate(
        "goal-convert-3-4",
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
        priority=0.8,
    )
    exhaust = _candidate(
        "goal-exhaust-5",
        "exhaust",
        source=5,
        target=None,
        predicate="object_count_equals_zero",
        priority=0.7,
    )
    store.register_generated(convert)
    store.register_generated(exhaust)
    designer = ObjectiveDiscriminatingExperimentDesigner()

    choice = designer.design(
        observation=observation,
        store=store,
        safe_actions=["ACTION6"],
        click_actions=[
            DesignedExperimentAction(
                name="ACTION6", raw_action="ACTION6", action_args={"x": 3, "y": 3}
            ),
            DesignedExperimentAction(
                name="ACTION6", raw_action="ACTION6", action_args={"x": 8, "y": 8}
            ),
        ],
    )

    assert choice is not None
    assert choice.expected_divergence == 2.0
    assert len(choice.competing_objective_ids) == 2
    assert len(choice.predicted_reduction_objective_ids) == 1


def test_online_goal_bank_remains_globally_bounded_across_states():
    store = OnlineTerminalObjectiveStore()
    for index in range(20):
        store.register_generated_bounded(
            _candidate(
                f"goal-{index}",
                "exhaust",
                source=index + 1,
                target=None,
                predicate="object_count_equals_zero",
                priority=float(index) / 20.0,
            ),
            max_objectives=6,
        )

    assert len(store.objectives()) == 6
    assert min(item.prior_priority for item in store.objectives()) >= 0.7

    per_family = OnlineTerminalObjectiveStore()
    for index in range(6):
        per_family.register_generated_bounded(
            _candidate(
                f"family-goal-{index}",
                "exhaust",
                source=index + 1,
                target=None,
                predicate="object_count_equals_zero",
                priority=float(index),
            ),
            max_objectives=10,
            max_per_family=2,
        )
    assert len(per_family.objectives()) == 2


def test_terminal_event_with_two_reduced_goals_remains_ambiguous():
    store = OnlineTerminalObjectiveStore()
    convert = _candidate(
        "goal-convert-3-4",
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    )
    exhaust = _candidate(
        "goal-exhaust-3",
        "exhaust",
        source=3,
        target=None,
        predicate="object_count_equals_zero",
    )
    store.register_generated(convert)
    store.register_generated(exhaust)
    before = np.zeros((7, 7), dtype=np.int32)
    before[2, 2] = 3
    after = before.copy()
    after[2, 2] = 4

    outcome = store.observe_transition(
        _update(before, after, levels_after=1),
        objective_id=convert.objective_id,
        intervention_id="ACTION6::convert",
        predicted_objective_ids=(convert.objective_id, exhaust.objective_id),
        context_signature="context-a",
    )

    assert outcome["terminal_credited_objectives"] == []
    assert set(outcome["terminal_ambiguous_objectives"]) == {
        convert.objective_id,
        exhaust.objective_id,
    }
    summary = store.summary()
    assert summary["ambiguous_terminal_events"] == 1
    assert summary["statuses"]["ambiguous_terminal"] == 2


def test_goal_requires_two_independent_terminal_contexts():
    store = OnlineTerminalObjectiveStore(minimum_terminal_support=2)
    goal = _candidate(
        "goal-convert-3-4",
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    )
    store.register_generated(goal)
    for index, position in enumerate(((2, 2), (4, 4))):
        before = np.zeros((7, 7), dtype=np.int32)
        before[position] = 3
        after = before.copy()
        after[position] = 4
        store.observe_transition(
            _update(before, after, levels_after=1),
            objective_id=goal.objective_id,
            intervention_id=f"ACTION6::context-{index}",
            context_signature=f"context-{index}",
        )
        if index == 0:
            assert store.objective(goal.objective_id).status.value == "needs_contrast"
            store.start_branch()

    objective = store.objective(goal.objective_id)
    assert objective is not None
    assert objective.terminal_support == 2
    assert objective.status.value == "terminal_supported"


def test_game_over_marks_intervention_unsafe_without_refuting_goal():
    store = OnlineTerminalObjectiveStore()
    goal = _candidate(
        "goal-convert-3-4",
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    )
    store.register_generated(goal)
    before = np.zeros((7, 7), dtype=np.int32)
    before[2, 2] = 3
    after = before.copy()
    after[2, 2] = 4

    store.observe_transition(
        _update(before, after, game_state_after="GAME_OVER"),
        objective_id=goal.objective_id,
        intervention_id="ACTION6::unsafe",
        context_signature="unsafe-context",
    )

    objective = store.objective(goal.objective_id)
    assert objective is not None
    assert objective.unsafe_plan_failures == 1
    assert objective.terminal_contradictions == 0
    assert objective.status.value == "candidate"
    assert store.intervention_is_unsafe(goal.objective_id, "ACTION6::unsafe")


def test_terminal_ablation_can_disprove_necessity_without_mechanic_refutation():
    store = OnlineTerminalObjectiveStore()
    goal = _candidate(
        "goal-convert-3-4",
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    )
    store.register_generated(goal)
    before = np.zeros((7, 7), dtype=np.int32)
    before[2, 2] = 3
    transformed = before.copy()
    transformed[2, 2] = 4
    store.observe_transition(
        _update(before, transformed, levels_after=1),
        objective_id=goal.objective_id,
        intervention_id="ACTION6::positive",
        context_signature="positive-context",
    )
    store.start_branch()

    store.observe_transition(
        _update(transformed, transformed, levels_after=1),
        intervention_id="ACTION6::ablation",
        ablation_of_objective_id=goal.objective_id,
        context_signature="ablation-context",
    )

    objective = store.objective(goal.objective_id)
    assert objective is not None
    assert objective.ablation_attempts == 1
    assert objective.ablation_contradictions == 1
    assert objective.terminal_contradictions == 0
    assert objective.terminal_support == 1


def test_designer_schedules_selective_ablation_after_first_terminal_support():
    observation = _observation()
    store = OnlineTerminalObjectiveStore()
    goal = _candidate(
        "goal-convert-3-4",
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    )
    alternative = _candidate(
        "goal-exhaust-5",
        "exhaust",
        source=5,
        target=None,
        predicate="object_count_equals_zero",
    )
    store.register_generated(goal)
    store.register_generated(alternative)
    after = observation.raw_grid.copy()
    after[3, 3] = 4
    positive_intervention = intervention_id("ACTION6", {"x": 3, "y": 3})
    store.observe_transition(
        _update(observation.raw_grid, after, levels_after=1),
        objective_id=goal.objective_id,
        intervention_id=positive_intervention,
        predicted_objective_ids=(goal.objective_id,),
        context_signature="positive-context",
    )
    store.start_branch()

    choice = ObjectiveDiscriminatingExperimentDesigner().design(
        observation=observation,
        store=store,
        safe_actions=["ACTION6"],
        click_actions=[
            DesignedExperimentAction(
                name="ACTION6", raw_action="ACTION6", action_args={"x": 3, "y": 3}
            ),
            DesignedExperimentAction(
                name="ACTION6", raw_action="ACTION6", action_args={"x": 8, "y": 8}
            ),
        ],
    )

    assert choice is not None
    assert choice.ablation_of_objective_id == goal.objective_id
    assert choice.intervention_id != positive_intervention
    assert goal.objective_id not in choice.predicted_reduction_objective_ids


def test_unified_controller_prioritizes_goal_discrimination_with_audit_fields():
    controller = UnifiedCognitiveController(
        "synthetic-active-goals",
        available_actions=["ACTION6"],
    )
    observation = _observation()

    decision = controller.select_action(
        current_grid=observation.raw_grid,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )

    assert decision.source in {
        "terminal_objective_discriminator",
        "terminal_objective_probe",
    }
    assert decision.objective_id
    assert decision.objective_status == "candidate"
    assert decision.objective_distance is not None
    assert decision.intervention_id.startswith("ACTION6::")
    assert len(decision.mechanic_hypotheses) == 2
