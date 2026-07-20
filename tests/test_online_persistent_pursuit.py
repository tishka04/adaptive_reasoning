"""Progress-gated persistent downstream pursuit tests for SAGE.8s."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation, build_transition_record
from theory.online_causal_option import OnlineCausalOptionStore
from theory.online_causal_subgoal_graph import (
    CausalMechanicEvidence,
    CausalSubgoalEdge,
)
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.online_persistent_pursuit import OnlinePersistentPursuitPolicy
from theory.online_terminal_objective import OnlineTerminalObjectiveStore


def _grid(threes: int, *, marker: int = 8) -> np.ndarray:
    grid = np.zeros((7, 7), dtype=np.int32)
    for index in range(threes):
        grid[1 + index * 2, 1 + index * 2] = 3
    grid[6, 6] = marker
    return grid


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION1", "ACTION2"],
        infer_players=False,
    )


def _store() -> OnlineTerminalObjectiveStore:
    store = OnlineTerminalObjectiveStore()
    store.register_generated(GeneratedGoalHypothesis(
        objective_id="terminal::exhaust::color3",
        family="exhaust",
        source_color=3,
        target_color=None,
        predicate="object_count_equals_zero",
        supporting_rule_keys=(),
        supporting_actions=("ACTION1", "ACTION2"),
        generation_reason="synthetic_persistent_pursuit",
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


def _update(
    before: np.ndarray,
    after: np.ndarray,
    *,
    action: str = "ACTION2",
):
    record = build_transition_record(
        action=action,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1", "ACTION2"],
        infer_players=False,
    )
    return SimpleNamespace(record=record, action=action)


def _progress_supported_open_option(*, persistent: bool):
    options = OnlineCausalOptionStore(
        enable_persistent_directional_pursuit=persistent,
    )
    edge = _edge()
    options.sync_confirmed_edges([edge])
    option = options.options()[0]
    options.note_openings([edge.edge_key], context_signature="opened")
    active = options._active
    assert active is not None
    store = _store()
    effect = "changed:some|colors:8:-1,9:+1"
    link = options.downstream_subgoals.link_effect(
        option_id=option.option_id,
        effect_signature=effect,
        observation_before=_observation(_grid(3, marker=8)),
        observation_after=_observation(_grid(3, marker=9)),
        store=store,
        branch_index=0,
        context_signature="trigger",
        action_signature="ACTION1",
    )
    subgoal_id = link["generated_subgoal_ids"][0]
    progress = options.downstream_subgoals.observe_pursuit_with_store(
        subgoal_id=subgoal_id,
        observation_before=_observation(_grid(3, marker=9)),
        observation_after=_observation(_grid(2, marker=9)),
        store=store,
        branch_index=0,
        context_signature="first-progress",
        action_signature="ACTION2",
        sequence=("ACTION2",),
        effect_signature="changed:some|colors:3:-1",
    )
    assert progress["progress"] is True
    active.effect_signatures.append(effect)
    active.downstream_subgoal_id = subgoal_id
    active.downstream_subgoal_attempts[subgoal_id] = 2
    active.downstream_objective_attempts[
        "terminal::exhaust::color3"
    ] = 2
    active.effect_conditioned_progress_events = 1
    active.downstream_subgoal_progress_events = 1
    return options, option, store, subgoal_id


def test_budget_expands_only_after_real_pursuit_progress():
    policy = OnlinePersistentPursuitPolicy(
        base_actions_per_subgoal=2,
        actions_per_progress=2,
        max_actions_per_subgoal=6,
        rollout_actions_per_progress=2,
        max_rollout_actions=10,
        credit_steps_per_progress=4,
        max_credit_window=16,
    )

    assert policy.action_limit(0) == 2
    assert policy.rollout_budget(6, 0) == 6
    assert policy.credit_window(8, 0) == 8
    assert policy.action_limit(1) == 4
    assert policy.action_limit(3) == 6
    assert policy.rollout_budget(6, 1) == 8
    assert policy.rollout_budget(6, 3) == 10
    assert policy.credit_window(8, 1) == 12
    assert policy.credit_window(8, 3) == 16


def test_progress_supported_subgoal_keeps_control_past_fixed_attempt_limit():
    options, _option, store, subgoal_id = _progress_supported_open_option(
        persistent=True
    )
    observation = _observation(_grid(2, marker=9))

    subgoal = options.select_effect_conditioned_subgoal(
        observation,
        store=store,
        safe_actions=["ACTION1", "ACTION2"],
    )

    assert subgoal is not None
    assert subgoal.subgoal_id == subgoal_id
    predictions = options.directional_action_predictions(
        observation,
        store=store,
        downstream_subgoal=subgoal,
        safe_actions=["ACTION1", "ACTION2"],
    )
    selection = options.select_downstream(
        observation,
        safe_actions=["ACTION1", "ACTION2"],
        downstream_subgoal=subgoal,
        directional_predictions=predictions,
    )

    assert selection is not None
    assert selection.action_name == "ACTION2"
    assert selection.persistent_pursuit is True
    assert selection.persistent_attempt_index == 3
    assert selection.persistent_action_limit == 4
    assert "continue progress-supported" in selection.reason
    active = options._active
    assert active is not None
    assert options._current_action_budget(active) == 8
    assert options._current_credit_window(active) == 12


def test_persistent_transition_can_repeat_progress_and_extend_rollout():
    options, option, store, _subgoal_id = _progress_supported_open_option(
        persistent=True
    )
    before = _grid(2, marker=9)
    observation = _observation(before)
    subgoal = options.select_effect_conditioned_subgoal(
        observation,
        store=store,
        safe_actions=["ACTION1", "ACTION2"],
    )
    assert subgoal is not None
    predictions = options.directional_action_predictions(
        observation,
        store=store,
        downstream_subgoal=subgoal,
        safe_actions=["ACTION1", "ACTION2"],
    )
    selection = options.select_downstream(
        observation,
        safe_actions=["ACTION1", "ACTION2"],
        downstream_subgoal=subgoal,
        directional_predictions=predictions,
    )
    assert selection is not None

    outcome = options.observe_transition(
        _update(before, _grid(1, marker=9)),
        store=store,
        option_id=option.option_id,
        causal_edge_key=option.edge_key,
        intervention_signature=selection.intervention_signature,
        downstream_subgoal_id=selection.downstream_subgoal_id,
        persistent_pursuit=selection.persistent_pursuit,
    )

    assert outcome["persistent_pursuit"] is True
    assert outcome["persistent_continuation_progress"] is True
    assert outcome["rollout_action_budget"] == 10
    summary = options.summary()["persistent_directional_pursuit"]
    assert summary["continuation_actions"] == 1
    assert summary["continuation_progress_events"] == 1
    assert summary["repeated_progress_events"] == 1
    assert summary["rollout_budget_extensions"] == 2
    assert summary["credit_window_extensions"] == 2


def test_ablation_preserves_the_fixed_two_action_cutoff():
    options, _option, store, _subgoal_id = _progress_supported_open_option(
        persistent=False
    )
    active = options._active
    assert active is not None

    selection = options.select_effect_conditioned_subgoal(
        _observation(_grid(2, marker=9)),
        store=store,
        safe_actions=["ACTION1", "ACTION2"],
    )

    assert selection is None
    assert options._current_action_budget(active) == 6
    assert options._current_credit_window(active) == 8
    summary = options.summary()["persistent_directional_pursuit"]
    assert summary["enabled"] is False
    assert summary["continuation_actions"] == 0


def test_observed_mode_bridge_composes_into_repeated_progress():
    options, option, store, subgoal_id = _progress_supported_open_option(
        persistent=True
    )
    subgoal_memory = options.downstream_subgoals.subgoal(subgoal_id)
    objective = store.objective("terminal::exhaust::color3")
    assert subgoal_memory is not None
    assert objective is not None
    subgoal_memory.successful_sequences.clear()
    current = _observation(_grid(2, marker=9))
    bridge_state = _observation(_grid(2, marker=7))
    progress_state = _observation(_grid(1, marker=7))
    options.downstream_subgoals.directional_model.observe(
        option_id=option.option_id,
        objective=objective,
        observation_before=current,
        observation_after=bridge_state,
        action_signature="ACTION1",
        effect_signature="7:+1,9:-1",
        branch_index=0,
        context_signature="learned-bridge",
        source="pursuit",
    )
    options.downstream_subgoals.directional_model.observe(
        option_id=option.option_id,
        objective=objective,
        observation_before=bridge_state,
        observation_after=progress_state,
        action_signature="ACTION2",
        effect_signature="3:-1",
        branch_index=0,
        context_signature="learned-followup",
        source="pursuit",
    )

    selected_subgoal = options.select_effect_conditioned_subgoal(
        current,
        store=store,
        safe_actions=["ACTION1", "ACTION2"],
    )
    assert selected_subgoal is not None
    predictions = options.directional_action_predictions(
        current,
        store=store,
        downstream_subgoal=selected_subgoal,
        safe_actions=["ACTION1", "ACTION2"],
    )
    bridge = options.select_downstream(
        current,
        safe_actions=["ACTION1", "ACTION2"],
        downstream_subgoal=selected_subgoal,
        directional_predictions=predictions,
    )
    assert bridge is not None
    assert bridge.action_name == "ACTION1"
    assert bridge.directional_effect_status == "bridge"
    assert bridge.directional_bridge_followup_action_signature == "ACTION2"
    options.observe_transition(
        _update(_grid(2, marker=9), _grid(2, marker=7), action="ACTION1"),
        store=store,
        option_id=option.option_id,
        causal_edge_key=option.edge_key,
        intervention_signature=bridge.intervention_signature,
        downstream_subgoal_id=bridge.downstream_subgoal_id,
        persistent_pursuit=bridge.persistent_pursuit,
    )

    selected_subgoal = options.select_effect_conditioned_subgoal(
        bridge_state,
        store=store,
        safe_actions=["ACTION1", "ACTION2"],
    )
    assert selected_subgoal is not None
    predictions = options.directional_action_predictions(
        bridge_state,
        store=store,
        downstream_subgoal=selected_subgoal,
        safe_actions=["ACTION1", "ACTION2"],
    )
    followup = options.select_downstream(
        bridge_state,
        safe_actions=["ACTION1", "ACTION2"],
        downstream_subgoal=selected_subgoal,
        directional_predictions=predictions,
    )
    assert followup is not None
    assert followup.action_name == "ACTION2"
    assert followup.directional_effect_status == "progressive"
    outcome = options.observe_transition(
        _update(_grid(2, marker=7), _grid(1, marker=7)),
        store=store,
        option_id=option.option_id,
        causal_edge_key=option.edge_key,
        intervention_signature=followup.intervention_signature,
        downstream_subgoal_id=followup.downstream_subgoal_id,
        persistent_pursuit=followup.persistent_pursuit,
    )

    assert outcome["persistent_continuation_progress"] is True
    summary = options.summary()["persistent_directional_pursuit"]
    assert summary["continuation_actions"] == 2
    assert summary["bridge_actions"] == 1
    assert summary["directional_policy_actions"] == 2
    assert summary["continuation_progress_events"] == 1
    assert summary["repeated_progress_events"] == 1
