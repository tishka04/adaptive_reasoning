"""Entity/role anchored intervention tests for SAGE.8t."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation
from theory.online_causal_option import OnlineCausalOptionStore
from theory.online_causal_subgoal_graph import CausalSubgoalEdge
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.online_semantic_intervention import semantic_intervention_anchor
from theory.online_state_conditioned_effect import (
    DirectionalEffectStatus,
    OnlineStateConditionedEffectModel,
)
from theory.online_terminal_objective import OnlineTerminalObjectiveStore


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION6"],
        infer_players=False,
    )


def _click(x: int, y: int) -> SimpleNamespace:
    return SimpleNamespace(action_args={"x": x, "y": y})


def _objective_store() -> OnlineTerminalObjectiveStore:
    store = OnlineTerminalObjectiveStore()
    store.register_generated(GeneratedGoalHypothesis(
        objective_id="terminal::exhaust::color3",
        family="exhaust",
        source_color=3,
        target_color=None,
        predicate="object_count_equals_zero",
        supporting_rule_keys=(),
        supporting_actions=("ACTION6",),
        generation_reason="synthetic_entity_anchor_test",
        prior_priority=1.0,
    ))
    return store


def _objective(store: OnlineTerminalObjectiveStore):
    objective = store.objective("terminal::exhaust::color3")
    assert objective is not None
    return objective


def _distance_grid(threes: int) -> np.ndarray:
    grid = np.zeros((9, 9), dtype=np.int32)
    for index in range(threes):
        grid[1 + index * 2, 7] = 3
    return grid


def _confirmed_edge() -> CausalSubgoalEdge:
    edge = CausalSubgoalEdge(
        edge_key="causal::source=>target",
        source_objective_id="source",
        target_objective_id="target",
        minimum_independent_support=2,
    )
    edge.support_events = 2
    edge.support_contexts.update({"context-a", "context-b"})
    edge.support_branches.update({0, 1})
    return edge


def test_same_color_objects_with_different_roles_do_not_alias():
    grid = np.zeros((9, 9), dtype=np.int32)
    grid[0, 0] = 8
    grid[4, 4] = 8
    observation = _observation(grid)

    corner = semantic_intervention_anchor(
        "ACTION6", {"x": 0, "y": 0}, observation
    )
    interior = semantic_intervention_anchor(
        "ACTION6", {"x": 4, "y": 4}, observation
    )

    assert corner.legacy_signature == interior.legacy_signature == "ACTION6::color:8"
    assert corner.entity_signature == interior.entity_signature
    assert corner.structural_role_signature != interior.structural_role_signature
    assert corner.transfer_signature != interior.transfer_signature
    assert corner.concrete_signature != interior.concrete_signature


def test_equivalent_entity_anchor_transfers_across_absolute_positions():
    first_grid = np.zeros((10, 10), dtype=np.int32)
    first_grid[2:4, 2:4] = 8
    second_grid = np.zeros((10, 10), dtype=np.int32)
    second_grid[5:7, 6:8] = 8

    first = semantic_intervention_anchor(
        "ACTION6", {"x": 2, "y": 2}, _observation(first_grid)
    )
    second = semantic_intervention_anchor(
        "ACTION6", {"x": 6, "y": 5}, _observation(second_grid)
    )

    assert first.transfer_signature == second.transfer_signature
    assert first.concrete_signature == second.concrete_signature
    assert first.instance_signature == second.instance_signature == "slot:1of1"


def test_equivalent_instances_keep_distinct_concrete_slots():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2, 2] = 8
    grid[7, 7] = 8
    observation = _observation(grid)

    first = semantic_intervention_anchor(
        "ACTION6", {"x": 2, "y": 2}, observation
    )
    second = semantic_intervention_anchor(
        "ACTION6", {"x": 7, "y": 7}, observation
    )

    assert first.transfer_signature == second.transfer_signature
    assert first.concrete_signature != second.concrete_signature
    assert {first.instance_signature, second.instance_signature} == {
        "slot:1of2",
        "slot:2of2",
    }


def test_directional_progress_transfers_to_an_equivalent_unseen_instance():
    store = _objective_store()
    objective = _objective(store)
    model = OnlineStateConditionedEffectModel()
    before = _observation(_distance_grid(2))
    after = _observation(_distance_grid(1))
    transfer = "ACTION6::entity:color8:shape1x1-dot:area1::role:interior"
    observed = f"{transfer}::instance:slot:1of2"
    equivalent = f"{transfer}::instance:slot:2of2"

    model.observe(
        option_id="option",
        objective=objective,
        observation_before=before,
        observation_after=after,
        action_signature=observed,
        effect_signature="3:-1",
        branch_index=0,
        context_signature="observed-instance",
        source="pursuit",
    )
    prediction = model.predict(
        option_id="option",
        objective=objective,
        observation=before,
        action_signature=equivalent,
    )

    assert prediction.status == DirectionalEffectStatus.PROGRESSIVE
    assert prediction.compatible is True
    assert prediction.structural_transfer_evidence is True
    assert prediction.exact_mode_evidence is True
    assert model.summary()["structural_transfer_predictions"] == 1


def test_opposing_equivalent_instances_require_entity_contrast():
    store = _objective_store()
    objective = _objective(store)
    model = OnlineStateConditionedEffectModel()
    before = _observation(_distance_grid(2))
    progress = _observation(_distance_grid(1))
    regress = _observation(_distance_grid(3))
    transfer = "ACTION6::entity:color8:shape1x1-dot:area1::role:interior"
    first = f"{transfer}::instance:slot:1of3"
    second = f"{transfer}::instance:slot:2of3"
    unseen = f"{transfer}::instance:slot:3of3"
    for signature, after, effect in (
        (first, progress, "3:-1"),
        (second, regress, "3:+1"),
    ):
        model.observe(
            option_id="option",
            objective=objective,
            observation_before=before,
            observation_after=after,
            action_signature=signature,
            effect_signature=effect,
            branch_index=0,
            context_signature=signature,
            source="pursuit",
        )

    prediction = model.predict(
        option_id="option",
        objective=objective,
        observation=before,
        action_signature=unseen,
    )
    first_prediction = model.predict(
        option_id="option",
        objective=objective,
        observation=before,
        action_signature=first,
    )
    second_prediction = model.predict(
        option_id="option",
        objective=objective,
        observation=before,
        action_signature=second,
    )

    assert prediction.status == DirectionalEffectStatus.NEEDS_ENTITY_CONTRAST
    assert prediction.compatible is True
    assert prediction.entity_alias_conflict is True
    assert prediction.reversible_across_modes is False
    assert first_prediction.status == DirectionalEffectStatus.PROGRESSIVE
    assert second_prediction.status == DirectionalEffectStatus.REGRESSIVE
    assert model.summary()["entity_alias_conflicts"] == 1
    assert model.summary()["entity_contrast_predictions"] == 1


def test_causal_option_trial_limits_are_per_concrete_entity_slot():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2, 2] = 8
    grid[7, 7] = 8
    observation = _observation(grid)
    clicks = (_click(2, 2), _click(7, 7))
    store = OnlineCausalOptionStore(max_trials_per_signature=1)
    edge = _confirmed_edge()
    store.sync_confirmed_edges([edge])
    store.note_openings([edge.edge_key], context_signature="opened")

    first = store.select_downstream(
        observation,
        safe_actions=["ACTION6"],
        click_actions=clicks,
    )
    assert first is not None
    assert first.entity_anchored_intervention is True
    assert store._active is not None
    store._active.signature_attempts[first.intervention_signature] = 1

    second = store.select_downstream(
        observation,
        safe_actions=["ACTION6"],
        click_actions=clicks,
    )

    assert second is not None
    assert second.intervention_signature != first.intervention_signature
    assert (
        second.intervention_transfer_signature
        == first.intervention_transfer_signature
    )


def test_entity_anchor_ablation_restores_color_only_identity():
    grid = np.zeros((7, 7), dtype=np.int32)
    grid[3, 3] = 8
    observation = _observation(grid)

    anchor = semantic_intervention_anchor(
        "ACTION6",
        {"x": 3, "y": 3},
        observation,
        enabled=False,
    )

    assert anchor.anchored is False
    assert anchor.concrete_signature == "ACTION6::color:8"
    assert anchor.transfer_signature == "ACTION6::color:8"
