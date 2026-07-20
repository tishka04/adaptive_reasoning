"""Latent-mode directional action learning tests for SAGE.8r."""

from __future__ import annotations

import numpy as np

from theory.live_transition_loop import build_observation
from theory.online_causal_option import OnlineCausalOptionStore
from theory.online_causal_subgoal_graph import (
    CausalMechanicEvidence,
    CausalSubgoalEdge,
)
from theory.online_effect_conditioned_subgoal import (
    OnlineEffectConditionedSubgoalStore,
)
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.online_state_conditioned_effect import (
    DirectionalEffectStatus,
    OnlineStateConditionedEffectModel,
    latent_mode_signature,
)
from theory.online_terminal_objective import OnlineTerminalObjectiveStore


def _grid(threes: int, *, offset: int = 0) -> np.ndarray:
    grid = np.zeros((7, 7), dtype=np.int32)
    for index in range(threes):
        row = 1 + ((index + offset) % 3) * 2
        column = 1 + ((index * 2 + offset) % 3) * 2
        grid[row, column] = 3
    grid[6, 6] = 8
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
        generation_reason="synthetic_directional_test",
        prior_priority=1.0,
    ))
    return store


def _objective(store: OnlineTerminalObjectiveStore):
    objective = store.objective("terminal::exhaust::color3")
    assert objective is not None
    return objective


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


def test_latent_mode_signature_is_position_invariant():
    objective = _objective(_store())

    first = latent_mode_signature(_observation(_grid(2)), objective)
    moved = latent_mode_signature(_observation(_grid(2, offset=1)), objective)

    assert first == moved


def test_same_action_can_be_progressive_then_regressive_across_modes():
    store = _store()
    objective = _objective(store)
    model = OnlineStateConditionedEffectModel()
    high = _observation(_grid(2))
    low = _observation(_grid(1))

    model.observe(
        option_id="option",
        objective=objective,
        observation_before=high,
        observation_after=low,
        action_signature="ACTION2",
        effect_signature="3:-1",
        branch_index=0,
        context_signature="forward",
        source="trigger",
    )
    model.observe(
        option_id="option",
        objective=objective,
        observation_before=low,
        observation_after=high,
        action_signature="ACTION2",
        effect_signature="3:+1",
        branch_index=0,
        context_signature="reverse",
        source="pursuit",
    )

    progressive = model.predict(
        option_id="option",
        objective=objective,
        observation=high,
        action_signature="ACTION2",
    )
    regressive = model.predict(
        option_id="option",
        objective=objective,
        observation=low,
        action_signature="ACTION2",
    )

    assert progressive.status == DirectionalEffectStatus.PROGRESSIVE
    assert progressive.compatible is True
    assert regressive.status == DirectionalEffectStatus.REGRESSIVE
    assert regressive.compatible is False
    assert progressive.reversible_across_modes is True
    assert model.summary()["reversible_action_objectives"] == 1


def test_related_evidence_requests_one_contrast_in_a_new_mode():
    store = _store()
    objective = _objective(store)
    model = OnlineStateConditionedEffectModel()
    model.observe(
        option_id="option",
        objective=objective,
        observation_before=_observation(_grid(2)),
        observation_after=_observation(_grid(1)),
        action_signature="ACTION2",
        effect_signature="3:-1",
        branch_index=0,
        context_signature="observed-mode",
        source="trigger",
    )

    prediction = model.predict(
        option_id="option",
        objective=objective,
        observation=_observation(_grid(3)),
        action_signature="ACTION2",
    )

    assert prediction.status == DirectionalEffectStatus.NEEDS_MODE_CONTRAST
    assert prediction.compatible is True
    assert prediction.exact_mode_evidence is False

    new_mode = _observation(_grid(3))
    model.observe(
        option_id="option",
        objective=objective,
        observation_before=new_mode,
        observation_after=new_mode,
        action_signature="ACTION2",
        effect_signature="changed:zero",
        branch_index=0,
        context_signature="stalled-contrast",
        source="pursuit",
    )
    exhausted_contrast = model.predict(
        option_id="option",
        objective=objective,
        observation=new_mode,
        action_signature="ACTION2",
    )

    assert exhausted_contrast.status == DirectionalEffectStatus.NEUTRAL
    assert exhausted_contrast.compatible is False


def test_repeated_neutral_action_is_blocked_in_the_exact_mode():
    store = _store()
    objective = _objective(store)
    model = OnlineStateConditionedEffectModel()
    observation = _observation(_grid(2))
    for index in range(2):
        model.observe(
            option_id="option",
            objective=objective,
            observation_before=observation,
            observation_after=observation,
            action_signature="ACTION1",
            effect_signature="changed:zero",
            branch_index=0,
            context_signature=f"neutral-{index}",
            source="pursuit",
        )

    prediction = model.predict(
        option_id="option",
        objective=objective,
        observation=observation,
        action_signature="ACTION1",
    )

    assert prediction.status == DirectionalEffectStatus.NEUTRAL
    assert prediction.compatible is False


def test_causal_suffix_prefers_mode_contrast_then_blocks_known_regression():
    store = _store()
    memory = OnlineEffectConditionedSubgoalStore()
    high = _observation(_grid(2))
    low = _observation(_grid(1))
    link = memory.link_effect(
        option_id=(
            "causal-option::causal::source=>terminal::exhaust::color3"
        ),
        effect_signature="changed:some|colors:3:-1",
        observation_before=high,
        observation_after=low,
        store=store,
        branch_index=0,
        context_signature="trigger",
        action_signature="ACTION2",
    )
    subgoal_id = link["generated_subgoal_ids"][0]

    options = OnlineCausalOptionStore()
    edge = _edge()
    options.sync_confirmed_edges([edge])
    option = options.options()[0]
    options.downstream_subgoals = memory
    options.note_openings([edge.edge_key], context_signature="opened")
    subgoal = memory.select(
        option_id=option.option_id,
        observed_effect_signatures=["changed:some|colors:3:-1"],
        observation=low,
        store=store,
    )
    assert subgoal is not None
    predictions = options.directional_action_predictions(
        low,
        store=store,
        downstream_subgoal=subgoal,
        safe_actions=["ACTION1", "ACTION2"],
    )

    contrast = options.select_downstream(
        low,
        safe_actions=["ACTION1", "ACTION2"],
        downstream_subgoal=subgoal,
        directional_predictions=predictions,
    )

    assert contrast is not None
    assert contrast.action_name == "ACTION2"
    assert contrast.directional_mode_contrast is True
    memory.observe_pursuit_with_store(
        subgoal_id=subgoal_id,
        observation_before=low,
        observation_after=high,
        store=store,
        branch_index=0,
        context_signature="regressive-contrast",
        action_signature="ACTION2",
        sequence=("ACTION2",),
        effect_signature="changed:some|colors:3:+1",
    )
    predictions = options.directional_action_predictions(
        low,
        store=store,
        downstream_subgoal=subgoal,
        safe_actions=["ACTION1", "ACTION2"],
    )

    guarded = options.select_downstream(
        low,
        safe_actions=["ACTION1", "ACTION2"],
        downstream_subgoal=subgoal,
        directional_predictions=predictions,
    )

    assert guarded is not None
    assert guarded.action_name == "ACTION1"
    assert memory.summary()["state_conditioned_directional_model"][
        "blocked_regressive_actions"
    ] >= 1
