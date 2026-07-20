"""Online causal-precondition induction and utility arbitration tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation, build_transition_record
from theory.online_causal_subgoal_graph import (
    CausalMechanicEvidence,
    CausalSubgoalEdgeStatus,
    OnlineCausalSubgoalGraph,
    transition_effect_signature,
)
from theory.online_goal_hypothesis import (
    GeneratedGoalHypothesis,
    ObjectiveDiscriminatingExperimentDesigner,
)
from theory.online_temporal_goal_composition import (
    OnlineTemporalGoalComposer,
    TemporalGoalPlan,
    TemporalSubgoalStep,
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
    actions: tuple[str, ...] = ("ACTION6",),
    priority: float = 0.8,
) -> GeneratedGoalHypothesis:
    return GeneratedGoalHypothesis(
        objective_id=objective_id,
        family=family,
        source_color=source,
        target_color=target,
        predicate=predicate,
        supporting_rule_keys=(),
        supporting_actions=actions,
        generation_reason="synthetic_causal_test",
        prior_priority=priority,
    )


def _store() -> tuple[OnlineTerminalObjectiveStore, str, str]:
    store = OnlineTerminalObjectiveStore()
    source_id = "goal-convert-3-to-4"
    target_id = "goal-appear-4-5"
    store.register_generated(_candidate(
        source_id,
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    ))
    store.register_generated(_candidate(
        target_id,
        "appear",
        source=4,
        target=5,
        predicate="adjacent_to",
        actions=("ACTION1",),
        priority=1.2,
    ))
    return store, source_id, target_id


def _grid(
    source_positions=((1, 1), (1, 3)),
    *,
    four_position=(5, 5),
    five_position=None,
) -> np.ndarray:
    grid = np.zeros((8, 8), dtype=np.int32)
    for position in source_positions:
        grid[position] = 3
    grid[four_position] = 4
    if five_position is not None:
        grid[five_position] = 5
    return grid


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION1", "ACTION6"],
        infer_players=False,
    )


def _update(before: np.ndarray, after: np.ndarray, *, game_over: bool = False):
    record = build_transition_record(
        action="ACTION6",
        action_args={"x": 1, "y": 1},
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1", "ACTION6"],
        game_state_after="GAME_OVER" if game_over else "NOT_FINISHED",
    )
    return SimpleNamespace(record=record)


def _edge_for(graph, source_id, target_id):
    return next(
        edge for edge in graph.edges()
        if edge.source_objective_id == source_id
        and edge.target_objective_id == target_id
    )


def test_blocked_target_generates_bounded_structurally_linked_preconditions():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph(max_edges_per_blocked_target=2)
    observation = _observation(_grid())

    edges = graph.note_blocked(target_id, observation, store)

    edge = _edge_for(graph, source_id, target_id)
    assert edge in edges
    assert edge.shared_colors == (4,)
    assert edge.generation_reasons == {"shared_structural_entity_precondition"}
    assert graph.summary()["blocked_target_events"] == 1
    assert len([
        item for item in graph.edges()
        if item.target_objective_id == target_id
    ]) <= 2


def test_causal_edge_compiles_into_a_state_guarded_temporal_plan():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph()
    observation = _observation(_grid())
    graph.note_blocked(target_id, observation, store)
    edge = _edge_for(graph, source_id, target_id)
    composer = OnlineTemporalGoalComposer(max_plans=20)

    plans = composer.compose(
        observation,
        store,
        causal_edges=graph.candidate_edges(observation, store),
    )

    plan = next(item for item in plans if item.causal_edge_key == edge.edge_key)
    assert plan.generation_reason == "online_learned_causal_subgoal_dependency"
    assert [step.objective_id for step in plan.steps] == [source_id, target_id]
    assert plan.steps[0].role == "learned_causal_precondition"
    assert plan.steps[1].role == "causally_enabled_objective"
    initial_utility = plan.causal_edge_utility
    edge.trials = 1
    edge.actions = 10
    edge.unsafe_failures = 2
    composer.compose(observation, store, causal_edges=[edge])
    assert plan.causal_edge_utility < initial_utility


def test_two_independent_availability_recoveries_confirm_an_edge_only():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph(minimum_independent_support=2)
    edge_key = ""
    for branch, positions in enumerate((((1, 1), (1, 3)), ((2, 1), (2, 4)))):
        before = _grid(source_positions=positions)
        observation = _observation(before)
        graph.note_blocked(target_id, observation, store)
        edge = _edge_for(graph, source_id, target_id)
        edge_key = edge.edge_key
        graph.begin_trial(edge.edge_key, context_signature=f"source-{branch}")
        after = before.copy()
        after[positions[0]] = 4
        graph.observe_transition(
            _update(before, after),
            store=store,
            source_objective_id=source_id,
            edge_key=edge.edge_key,
            source_step_completed=True,
            context_signature=f"source-{branch}",
        )
        graph.note_intervention_availability(
            target_id,
            available=True,
            observation=_observation(after),
            store=store,
            context_signature=f"target-{branch}",
        )
        if branch == 0:
            assert edge.status == CausalSubgoalEdgeStatus.CANDIDATE
            graph.start_branch()

    edge = graph.edge(edge_key)
    target = store.objective(target_id)
    assert edge is not None and target is not None
    assert edge.availability_successes == 2
    assert edge.status == CausalSubgoalEdgeStatus.CONFIRMED
    assert graph.summary()["cross_branch_confirmations"] == 1
    assert target.status.value == "candidate"


def test_two_failed_recoveries_refute_the_causal_edge_not_the_goal():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph()
    edge_key = ""
    for branch in range(2):
        before = _grid(source_positions=((branch + 1, 1), (branch + 1, 3)))
        graph.note_blocked(target_id, _observation(before), store)
        edge = _edge_for(graph, source_id, target_id)
        edge_key = edge.edge_key
        graph.begin_trial(edge.edge_key, context_signature=f"failed-{branch}")
        after = before.copy()
        after[branch + 1, 1] = 4
        graph.observe_transition(
            _update(before, after),
            store=store,
            source_objective_id=source_id,
            edge_key=edge.edge_key,
            source_step_completed=True,
            context_signature=f"failed-{branch}",
        )
        graph.note_intervention_availability(
            target_id,
            available=False,
            observation=_observation(after),
            store=store,
            context_signature=f"still-blocked-{branch}",
        )
        if branch == 0:
            graph.start_branch()

    edge = graph.edge(edge_key)
    target = store.objective(target_id)
    assert edge is not None and target is not None
    assert edge.contradictions == 2
    assert edge.status == CausalSubgoalEdgeStatus.REFUTED
    assert target.status.value == "candidate"


def test_goal_cochange_supplies_causal_support_without_terminal_credit():
    store = OnlineTerminalObjectiveStore()
    source_id = "goal-convert-3-to-4"
    target_id = "goal-exhaust-3"
    store.register_generated(_candidate(
        source_id,
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
    ))
    store.register_generated(_candidate(
        target_id,
        "exhaust",
        source=3,
        target=None,
        predicate="object_count_equals_zero",
    ))
    graph = OnlineCausalSubgoalGraph()
    before = _grid()
    graph.note_blocked(target_id, _observation(before), store)
    edge = _edge_for(graph, source_id, target_id)
    after = before.copy()
    after[1, 1] = 4

    outcome = graph.observe_transition(
        _update(before, after),
        store=store,
        source_objective_id=source_id,
        edge_key=edge.edge_key,
        context_signature="cochange-context",
    )

    assert outcome["cochange_supported_edges"] == [edge.edge_key]
    assert edge.cochange_supports == 1
    assert edge.support_events == 1
    assert store.objective(target_id).terminal_support == 0


def test_target_change_without_source_progress_is_not_causal_support():
    store = OnlineTerminalObjectiveStore()
    source_id = "goal-convert-3-to-4"
    target_id = "goal-exhaust-5"
    store.register_generated(_candidate(
        source_id,
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
        priority=2.0,
    ))
    store.register_generated(_candidate(
        target_id,
        "exhaust",
        source=5,
        target=None,
        predicate="object_count_equals_zero",
    ))
    before = _grid(five_position=(6, 1))
    before[6, 4] = 5
    graph = OnlineCausalSubgoalGraph()
    graph.note_blocked(target_id, _observation(before), store)
    edge = _edge_for(graph, source_id, target_id)
    after = before.copy()
    after[6, 1] = 4

    outcome = graph.observe_transition(
        _update(before, after),
        store=store,
        source_objective_id=source_id,
        edge_key=edge.edge_key,
        context_signature="target-only-change",
    )

    assert outcome["cochange_supported_edges"] == []
    assert edge.support_events == 0


def test_blocked_state_signature_transfers_across_object_positions():
    store, _, target_id = _store()
    graph = OnlineCausalSubgoalGraph()
    first = _observation(_grid(four_position=(5, 5)))
    second = _observation(_grid(
        source_positions=((3, 2), (6, 1)),
        four_position=(1, 6),
    ))
    target = store.objective(target_id)
    assert target is not None

    assert graph.state_signature(target, first) == graph.state_signature(
        target, second
    )
    graph.note_blocked(target_id, first, store)
    assert graph.candidate_edges(second, store)


def test_plan_utility_rewards_progress_and_penalizes_cost_risk_and_abandonment():
    steps = (
        TemporalSubgoalStep("source", 0.0),
        TemporalSubgoalStep("target", 0.0),
    )
    useful = TemporalGoalPlan("useful", "target", steps, "test", prior_priority=1.0)
    useful.starts = 2
    useful.actions = 4
    useful.progress_events = 3
    useful.step_completions = 2
    wasteful = TemporalGoalPlan("wasteful", "target", steps, "test", prior_priority=1.0)
    wasteful.starts = 2
    wasteful.actions = 8
    wasteful.abandonments = 2
    wasteful.unsafe_failures = 1

    assert useful.expected_progress_probability > wasteful.expected_progress_probability
    assert useful.expected_cost < wasteful.expected_cost
    assert useful.selection_utility > wasteful.selection_utility


def test_controller_uses_a_discovered_precondition_with_full_audit_fields():
    controller = UnifiedCognitiveController(
        "synthetic-causal-controller",
        available_actions=["ACTION6"],
    )
    source_id = "goal-convert-3-to-4"
    target_id = "goal-appear-4-5"
    controller.terminal_objectives.register_generated(_candidate(
        source_id,
        "convert",
        source=3,
        target=4,
        predicate="source_target_color_transform",
        priority=1.0,
    ))
    controller.terminal_objectives.register_generated(_candidate(
        target_id,
        "appear",
        source=4,
        target=5,
        predicate="adjacent_to",
        actions=("ACTION1",),
        priority=5.0,
    ))
    before = _grid()
    first = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    assert controller.causal_subgoals.summary()["blocked_target_events"] >= 1
    after = before.copy()
    after[1, 1] = 4
    controller.observe_transition(
        action=first.action_name,
        action_data=first.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION6"],
    )

    decision = controller.select_action(
        current_grid=after,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )

    assert decision.source == "temporal_subgoal_probe"
    assert decision.causal_subgoal_edge_key
    assert decision.temporal_step_index == 0
    assert decision.temporal_expected_progress_probability is not None
    assert decision.temporal_expected_cost is not None
    assert decision.temporal_selection_utility is not None
    assert decision.causal_intervention_signature
    assert decision.causal_intervention_utility is not None


def test_effect_signature_transfers_across_absolute_positions():
    first_before = _grid(source_positions=((1, 1), (1, 3)))
    first_after = first_before.copy()
    first_after[1, 1] = 4
    second_before = _grid(source_positions=((3, 2), (6, 1)))
    second_after = second_before.copy()
    second_after[3, 2] = 4

    assert transition_effect_signature(
        _update(first_before, first_after)
    ) == transition_effect_signature(_update(second_before, second_after))


def test_partial_source_progress_receives_delayed_effect_credit_when_target_opens():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph(delayed_credit_window=3)
    before = _grid()
    graph.note_blocked(target_id, _observation(before), store)
    edge = _edge_for(graph, source_id, target_id)
    graph.begin_trial(edge.edge_key, context_signature="before-progress")
    after = before.copy()
    after[1, 1] = 4
    graph.observe_transition(
        _update(before, after),
        store=store,
        source_objective_id=source_id,
        edge_key=edge.edge_key,
        source_step_completed=False,
        intervention_signature="ACTION6::color:3",
        context_signature="progress",
    )
    # Credit may arrive after unrelated observations, not only on the source step.
    graph.observe_transition(
        _update(after, after),
        store=store,
        context_signature="intervening-observation",
    )
    graph.note_intervention_availability(
        target_id,
        available=True,
        observation=_observation(after),
        store=store,
        context_signature="target-open",
    )

    assert edge.support_events == 1
    assert edge.availability_successes == 1
    assert len(edge.effect_evidence) == 1
    evidence = next(iter(edge.effect_evidence.values()))
    assert evidence.enablement_successes == 1
    assert (
        edge.intervention_evidence["ACTION6::color:3"].enablement_successes
        == 1
    )
    assert graph.summary()["delayed_credit_events"] == 1


def test_two_supports_in_one_branch_do_not_confirm_a_causal_mechanic():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph(minimum_independent_support=2)
    edge = None
    for index, position in enumerate(((1, 1), (1, 3))):
        before = _grid()
        graph.note_blocked(target_id, _observation(before), store)
        edge = _edge_for(graph, source_id, target_id)
        graph.begin_trial(edge.edge_key, context_signature=f"same-branch-{index}")
        after = before.copy()
        after[position] = 4
        graph.observe_transition(
            _update(before, after),
            store=store,
            source_objective_id=source_id,
            edge_key=edge.edge_key,
            context_signature=f"progress-{index}",
        )
        graph.note_intervention_availability(
            target_id,
            available=True,
            observation=_observation(after),
            store=store,
            context_signature=f"open-{index}",
        )

    assert edge is not None
    assert edge.support_events == 2
    assert len(edge.support_branches) == 1
    assert edge.status == CausalSubgoalEdgeStatus.CANDIDATE


def test_two_failures_in_one_branch_do_not_refute_a_causal_mechanic():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph(minimum_independent_support=2)
    edge = None
    for index, position in enumerate(((1, 1), (1, 3))):
        before = _grid()
        graph.note_blocked(target_id, _observation(before), store)
        edge = _edge_for(graph, source_id, target_id)
        graph.begin_trial(edge.edge_key, context_signature=f"same-branch-{index}")
        after = before.copy()
        after[position] = 4
        graph.observe_transition(
            _update(before, after),
            store=store,
            source_objective_id=source_id,
            edge_key=edge.edge_key,
        )
        graph.note_intervention_availability(
            target_id,
            available=False,
            observation=_observation(after),
            store=store,
            context_signature=f"closed-{index}",
        )

    assert edge is not None
    assert edge.contradictions == 2
    assert len(edge.contradiction_branches) == 1
    assert edge.status == CausalSubgoalEdgeStatus.CANDIDATE


def test_supported_edge_gets_a_reserved_confirmation_start_on_new_branch():
    store, source_id, target_id = _store()
    graph = OnlineCausalSubgoalGraph(minimum_independent_support=2)
    before = _grid()
    graph.note_blocked(target_id, _observation(before), store)
    edge = _edge_for(graph, source_id, target_id)
    graph.begin_trial(edge.edge_key, context_signature="branch-zero")
    after = before.copy()
    after[1, 1] = 4
    graph.observe_transition(
        _update(before, after),
        store=store,
        source_objective_id=source_id,
        edge_key=edge.edge_key,
    )
    graph.note_intervention_availability(
        target_id,
        available=True,
        observation=_observation(after),
        store=store,
        context_signature="open-zero",
    )
    graph.start_branch()
    graph.note_blocked(target_id, _observation(before), store)
    candidates = graph.candidate_edges(_observation(before), store)
    composer = OnlineTemporalGoalComposer(
        max_plans=20,
        max_plan_starts_total=0,
        max_confirmation_starts_per_branch=1,
    )
    composer.start_branch()
    composer.compose(_observation(before), store, causal_edges=candidates)

    selection = composer.select_step(_observation(before), store)

    assert selection is not None
    assert selection.causal_edge_key == edge.edge_key
    assert selection.causal_confirmation_priority is True
    assert composer.summary()["reserved_confirmation_starts"] == 1


def test_learned_semantic_intervention_utility_guides_same_goal_choice():
    store = OnlineTerminalObjectiveStore()
    objective_id = "goal-appear-3-4"
    store.register_generated(_candidate(
        objective_id,
        "appear",
        source=3,
        target=4,
        predicate="adjacent_to",
        actions=("ACTION1", "ACTION2"),
    ))
    observation = _observation(_grid())
    designer = ObjectiveDiscriminatingExperimentDesigner()
    evidence = CausalMechanicEvidence(
        signature="ACTION2",
        observations=2,
        source_progress_events=2,
        enablement_successes=1,
    )

    choice = designer.design(
        observation=observation,
        store=store,
        safe_actions=("ACTION1", "ACTION2"),
        preferred_objective_id=objective_id,
        intervention_utilities={"ACTION2": evidence.utility},
        require_selectable=False,
    )

    assert choice is not None
    assert choice.action_name == "ACTION2"
    assert choice.reason == "effect-guided causal intervention"
