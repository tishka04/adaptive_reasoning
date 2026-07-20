"""Effect-conditioned downstream goal learning tests for SAGE.8q."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation, build_transition_record
from theory.online_causal_option import OnlineCausalOptionStore
from theory.online_causal_subgoal_graph import (
    CausalMechanicEvidence,
    CausalSubgoalEdge,
)
from theory.online_effect_conditioned_subgoal import (
    EffectConditionedSubgoalStatus,
    OnlineEffectConditionedSubgoalStore,
)
from theory.online_goal_hypothesis import (
    GeneratedGoalHypothesis,
    GoalHypothesisGenerator,
)
from theory.online_terminal_objective import OnlineTerminalObjectiveStore
from theory.unified_cognitive_controller import UnifiedCognitiveController


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION1", "ACTION2", "ACTION4"],
        infer_players=False,
    )


def _grid(*, threes: int = 2, marker: int = 8) -> np.ndarray:
    grid = np.zeros((6, 6), dtype=np.int32)
    for index in range(threes):
        grid[1, 1 + 2 * index] = 3
    grid[4, 4] = marker
    return grid


def _objective_store() -> OnlineTerminalObjectiveStore:
    store = OnlineTerminalObjectiveStore()
    store.register_generated(GeneratedGoalHypothesis(
        objective_id="terminal::exhaust::color3",
        family="exhaust",
        source_color=3,
        target_color=None,
        predicate="object_count_equals_zero",
        supporting_rule_keys=(),
        supporting_actions=("ACTION1", "ACTION2"),
        generation_reason="synthetic_effect_subgoal",
        prior_priority=1.0,
    ))
    return store


def _edge() -> CausalSubgoalEdge:
    edge = CausalSubgoalEdge(
        edge_key="causal::source=>terminal::exhaust::color3",
        source_objective_id="source",
        target_objective_id="terminal::exhaust::color3",
        minimum_independent_support=2,
    )
    edge.support_events = 2
    edge.support_contexts.update({"context-0", "context-1"})
    edge.support_branches.update({0, 1})
    edge.intervention_evidence["ACTION1"] = CausalMechanicEvidence(
        signature="ACTION1",
        observations=2,
        source_progress_events=2,
        enablement_successes=2,
    )
    return edge


def _update(before: np.ndarray, after: np.ndarray, action: str = "ACTION2"):
    record = build_transition_record(
        action=action,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1", "ACTION2", "ACTION4"],
        infer_players=False,
    )
    return SimpleNamespace(record=record, action=action)


def test_real_color_flow_generates_measurable_goals_without_terminal_proof():
    before = _grid(threes=2, marker=8)
    after = _grid(threes=1, marker=9)

    candidates = GoalHypothesisGenerator().generate_from_transition(
        observation_before=_observation(before),
        observation_after=_observation(after),
        action_name="ACTION2",
    )

    ids = {candidate.objective_id for candidate in candidates}
    assert "terminal::exhaust::color3" in ids
    assert "terminal::convert::8_to_9" in ids
    assert all("observed_effect" in candidate.generation_reason for candidate in candidates)


def test_effect_to_objective_link_records_trigger_progress_without_suffix_proof():
    store = _objective_store()
    memory = OnlineEffectConditionedSubgoalStore()
    before = _grid(threes=2)
    after = _grid(threes=1, marker=9)

    outcome = memory.link_effect(
        option_id="option",
        effect_signature="changed:some|colors:3:-1,8:-1,9:+1",
        observation_before=_observation(before),
        observation_after=_observation(after),
        store=store,
        branch_index=0,
        context_signature="effect-0",
        action_signature="ACTION2",
    )

    assert outcome["reduced_objective_ids"] == [
        "terminal::exhaust::color3"
    ]
    subgoal = memory.subgoals()[0]
    assert subgoal.status == EffectConditionedSubgoalStatus.CANDIDATE
    assert subgoal.trigger_progress_events == 1
    assert subgoal.best_progress_sequence == ()
    assert store.objective(subgoal.objective_id).terminal_support == 0


def test_effect_link_is_selected_while_objective_is_incomplete():
    store = _objective_store()
    memory = OnlineEffectConditionedSubgoalStore()
    before = _grid(threes=2)
    after = _grid(threes=1, marker=9)
    effect = "changed:some|colors:3:-1,8:-1,9:+1"
    memory.link_effect(
        option_id="option",
        effect_signature=effect,
        observation_before=_observation(before),
        observation_after=_observation(after),
        store=store,
        branch_index=0,
        context_signature="effect-0",
        action_signature="ACTION2",
    )

    selection = memory.select(
        option_id="option",
        observed_effect_signatures=[effect],
        observation=_observation(after),
        store=store,
    )

    assert selection is not None
    assert selection.objective_id == "terminal::exhaust::color3"
    assert selection.status == EffectConditionedSubgoalStatus.CANDIDATE
    assert selection.best_progress_sequence == ()
    assert memory.select(
        option_id="option",
        observed_effect_signatures=[effect],
        observation=_observation(after),
        store=store,
        excluded_subgoal_ids=[selection.subgoal_id],
    ) is None


def test_reserved_local_measurement_survives_terminal_goal_refutation():
    store = _objective_store()
    memory = OnlineEffectConditionedSubgoalStore()
    before = _grid(threes=2)
    after = _grid(threes=1, marker=9)
    effect = "changed:some|colors:3:-1,8:-1,9:+1"
    link = memory.link_effect(
        option_id="option",
        effect_signature=effect,
        observation_before=_observation(before),
        observation_after=_observation(after),
        store=store,
        branch_index=0,
        context_signature="effect-0",
        action_signature="ACTION2",
    )
    subgoal_id = link["generated_subgoal_ids"][0]
    objective = store.objective("terminal::exhaust::color3")
    assert objective is not None
    objective.terminal_contradictions = 2

    unreserved = memory.select(
        option_id="option",
        observed_effect_signatures=[effect],
        observation=_observation(after),
        store=store,
    )
    reserved = memory.select(
        option_id="option",
        observed_effect_signatures=[],
        observation=_observation(after),
        store=store,
        preferred_subgoal_id=subgoal_id,
        reserve_preferred_context=True,
    )

    assert unreserved is None
    assert reserved is not None
    assert reserved.subgoal_id == subgoal_id


def test_multi_step_pursuit_records_and_replays_complete_progress_sequence():
    store = _objective_store()
    memory = OnlineEffectConditionedSubgoalStore()
    initial = _grid(threes=2, marker=8)
    opened = _grid(threes=2, marker=9)
    effect = "changed:one|colors:8:-1,9:+1"
    link = memory.link_effect(
        option_id="option",
        effect_signature=effect,
        observation_before=_observation(initial),
        observation_after=_observation(opened),
        store=store,
        branch_index=0,
        context_signature="effect-0",
        action_signature="ACTION4",
    )
    subgoal_id = link["generated_subgoal_ids"][0]

    pursuit = memory.observe_pursuit_with_store(
        subgoal_id=subgoal_id,
        observation_before=_observation(opened),
        observation_after=_observation(_grid(threes=1, marker=9)),
        store=store,
        branch_index=0,
        context_signature="pursuit-0",
        action_signature="ACTION2",
        sequence=("ACTION4", "ACTION2"),
    )

    assert pursuit["progress"] is True
    assert (
        memory.subgoal(subgoal_id).status
        == EffectConditionedSubgoalStatus.PROGRESS_SUPPORTED
    )
    assert memory.subgoal(subgoal_id).best_progress_sequence == (
        "ACTION4",
        "ACTION2",
    )


def test_two_failed_branches_refute_progress_link_but_reset_censoring_does_not():
    store = _objective_store()
    memory = OnlineEffectConditionedSubgoalStore()
    initial = _grid(threes=2, marker=8)
    opened = _grid(threes=2, marker=9)
    link = memory.link_effect(
        option_id="option",
        effect_signature="changed:one|colors:8:-1,9:+1",
        observation_before=_observation(initial),
        observation_after=_observation(opened),
        store=store,
        branch_index=0,
        context_signature="effect-0",
        action_signature="ACTION4",
    )
    subgoal_id = link["generated_subgoal_ids"][0]
    memory.close_pursuit(
        subgoal_id,
        branch_index=0,
        sequence=("ACTION1",),
        progressed=False,
        censored=True,
    )
    for branch in (1, 2):
        memory.close_pursuit(
            subgoal_id,
            branch_index=branch,
            sequence=("ACTION1", "ACTION2"),
            progressed=False,
            censored=False,
        )

    assert memory.subgoal(subgoal_id).status == EffectConditionedSubgoalStatus.REFUTED
    assert memory.summary()["censored_pursuits"] == 1


def test_unproductive_option_rollout_is_pruned_to_the_base_budget():
    edge = _edge()
    options = OnlineCausalOptionStore(
        max_downstream_actions=6,
        base_downstream_actions=2,
        progress_extension_actions=2,
    )
    options.sync_confirmed_edges([edge])
    options.note_openings([edge.edge_key], context_signature="opened")
    store = _objective_store()
    grid = _grid()

    for _ in range(2):
        selection = options.select_downstream(
            _observation(grid),
            safe_actions=["ACTION1", "ACTION2", "ACTION4"],
        )
        assert selection is not None
        options.observe_transition(
            _update(grid, grid, action=selection.action_name),
            store=store,
            option_id=selection.option_id,
            causal_edge_key=selection.edge_key,
            intervention_signature=selection.intervention_signature,
        )

    assert options.active_option_id == ""
    assert options.summary()["budget_pruned_rollouts"] == 1
    assert options.summary()["downstream_actions"] == 2


def test_measured_progress_extends_the_option_budget_online():
    edge = _edge()
    options = OnlineCausalOptionStore(
        max_downstream_actions=5,
        base_downstream_actions=2,
        progress_extension_actions=2,
    )
    options.sync_confirmed_edges([edge])
    options.note_openings([edge.edge_key], context_signature="opened")
    store = _objective_store()
    before = _grid(threes=2)
    after = _grid(threes=1, marker=9)
    selection = options.select_downstream(
        _observation(before),
        safe_actions=["ACTION1", "ACTION2", "ACTION4"],
        preferred_intervention_signatures=["ACTION2"],
    )
    assert selection is not None

    outcome = options.observe_transition(
        _update(before, after, action=selection.action_name),
        store=store,
        option_id=selection.option_id,
        causal_edge_key=selection.edge_key,
        intervention_signature=selection.intervention_signature,
    )

    assert outcome["rollout_action_budget"] == 4
    assert options.summary()["dynamic_budget_extensions"] == 1
    assert options.active_option_id


def test_controller_pursues_effect_conditioned_subgoal_on_the_next_step():
    controller = UnifiedCognitiveController(
        "synthetic-effect-conditioned-controller",
        available_actions=["ACTION1", "ACTION2", "ACTION4"],
    )
    controller.terminal_objectives.register_generated(GeneratedGoalHypothesis(
        objective_id="terminal::exhaust::color3",
        family="exhaust",
        source_color=3,
        target_color=None,
        predicate="object_count_equals_zero",
        supporting_rule_keys=(),
        supporting_actions=("ACTION1", "ACTION2"),
        generation_reason="synthetic_controller_target",
        prior_priority=1.0,
    ))
    edge = _edge()
    controller.causal_options.sync_confirmed_edges([edge])
    controller.causal_options.note_openings(
        [edge.edge_key],
        context_signature="observed-opening",
    )
    before = _grid(threes=2, marker=8)
    after = _grid(threes=1, marker=9)
    first = controller.select_action(
        current_grid=before,
        available_actions=["ACTION1", "ACTION2", "ACTION4"],
        legacy_action="ACTION1",
    )
    controller.observe_transition(
        action=first.action_name,
        action_data=first.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1", "ACTION2", "ACTION4"],
    )

    decision = controller.select_action(
        current_grid=after,
        available_actions=["ACTION1", "ACTION2", "ACTION4"],
        legacy_action="ACTION4",
    )

    assert decision.source == "causal_option_effect_subgoal_probe"
    assert decision.objective_id == "terminal::exhaust::color3"
    assert decision.causal_option_phase == "effect_conditioned_subgoal"
    assert decision.causal_option_downstream_subgoal_id
    assert decision.causal_option_downstream_subgoal_status == "candidate"
    assert decision.causal_option_trigger_effect_signature
    assert decision.causal_option_downstream_subgoal_utility is not None
