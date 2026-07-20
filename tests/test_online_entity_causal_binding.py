"""Active online entity/effect binding tests for SAGE.8u."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation
from theory.online_causal_option import OnlineCausalOptionStore
from theory.online_causal_subgoal_graph import CausalSubgoalEdge
from theory.online_effect_conditioned_subgoal import (
    EffectConditionedDownstreamSubgoal,
    EffectConditionedSubgoalSelection,
    EffectConditionedSubgoalStatus,
)
from theory.online_entity_causal_binding import (
    EntityBindingPrediction,
    EntityBindingStatus,
    EntityCorrespondenceStatus,
    OnlineEntityCausalBindingStore,
)
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.online_semantic_intervention import semantic_intervention_anchor
from theory.online_terminal_objective import OnlineTerminalObjectiveStore


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION6"],
        infer_players=False,
    )


def _objective():
    objectives = OnlineTerminalObjectiveStore()
    objectives.register_generated(GeneratedGoalHypothesis(
        objective_id="terminal::exhaust::color3",
        family="exhaust",
        source_color=3,
        target_color=None,
        predicate="object_count_equals_zero",
        supporting_rule_keys=(),
        supporting_actions=("ACTION6",),
        generation_reason="synthetic_entity_binding_test",
        prior_priority=1.0,
    ))
    objective = objectives.objective("terminal::exhaust::color3")
    assert objective is not None
    return objective


def _observe(
    binding: OnlineEntityCausalBindingStore,
    before: np.ndarray,
    after: np.ndarray,
    *,
    x: int,
    y: int,
    context: str,
):
    observation_before = _observation(before)
    observation_after = _observation(after)
    action_data = {"x": x, "y": y}
    anchor = semantic_intervention_anchor(
        "ACTION6",
        action_data,
        observation_before,
    )
    outcome = binding.observe(
        option_id="option",
        objective=_objective(),
        observation_before=observation_before,
        observation_after=observation_after,
        action_name="ACTION6",
        action_data=action_data,
        action_signature=anchor.concrete_signature,
        anchor=anchor,
        branch_index=0,
        context_signature=context,
    )
    return outcome, anchor, observation_before, observation_after


def test_color_transformation_keeps_target_correspondence_and_track():
    before = np.zeros((7, 7), dtype=np.int32)
    before[2, 2] = 8
    transformed = before.copy()
    transformed[2, 2] = 9
    moved = transformed.copy()
    moved[2, 2] = 0
    moved[2, 3] = 9
    binding = OnlineEntityCausalBindingStore()

    first, _anchor, _before, _after = _observe(
        binding,
        before,
        transformed,
        x=2,
        y=2,
        context="transform",
    )
    second, _anchor, _before, _after = _observe(
        binding,
        transformed,
        moved,
        x=2,
        y=2,
        context="move",
    )

    assert first["correspondence"]["status"] == "transformed"
    assert second["correspondence"]["status"] == "moved"
    assert (
        first["correspondence"]["track_id"]
        == second["correspondence"]["track_id"]
    )
    summary = binding.summary()
    assert summary["transformed_entities"] == 1
    assert summary["moved_entities"] == 1
    assert summary["tracks_created"] == 1
    assert summary["tracks_reused"] == 1


def test_removed_target_is_an_observed_entity_change():
    before = np.zeros((7, 7), dtype=np.int32)
    before[2, 2] = 8
    after = np.zeros_like(before)

    outcome, _anchor, _before, _after = _observe(
        OnlineEntityCausalBindingStore(),
        before,
        after,
        x=2,
        y=2,
        context="removed",
    )

    assert outcome["correspondence"]["status"] == "removed"
    assert outcome["carrier"] is True


def test_ambiguous_correspondence_receives_no_causal_credit():
    before = np.zeros((7, 7), dtype=np.int32)
    before[3, 3] = 8
    before[6, 6] = 3
    after = np.zeros_like(before)
    after[3, 2] = 8
    after[3, 4] = 8
    binding = OnlineEntityCausalBindingStore()

    outcome, _anchor, _before, _after = _observe(
        binding,
        before,
        after,
        x=3,
        y=3,
        context="ambiguous",
    )

    assert outcome["correspondence"]["status"] == "ambiguous"
    assert outcome["gain"] > 0.0
    assert outcome["carrier"] is False
    summary = binding.summary()
    assert summary["ambiguous_entities"] == 1
    assert summary["carrier_progress_events"] == 0
    assert summary["noncarrier_progress_events"] == 0


def test_progress_is_bound_only_when_the_target_entity_changes():
    before = np.zeros((7, 7), dtype=np.int32)
    before[3, 3] = 8
    before[6, 6] = 3
    after = before.copy()
    after[6, 6] = 0
    binding = OnlineEntityCausalBindingStore()

    outcome, anchor, observation, _after = _observe(
        binding,
        before,
        after,
        x=3,
        y=3,
        context="noncarrier-progress",
    )
    prediction = binding.predict(
        option_id="option",
        objective=_objective(),
        observation=observation,
        action_signature=anchor.concrete_signature,
    )

    assert outcome["correspondence"]["status"] == "stable"
    assert outcome["gain"] > 0.0
    assert outcome["carrier"] is False
    assert prediction.status == EntityBindingStatus.MISBOUND
    assert prediction.compatible is False


def test_online_binding_conflict_generates_a_controlled_target_contrast():
    corner_before = np.zeros((7, 7), dtype=np.int32)
    corner_before[0, 0] = 8
    corner_before[6, 6] = 3
    corner_after = np.zeros_like(corner_before)
    interior_before = np.zeros((7, 7), dtype=np.int32)
    interior_before[3, 3] = 8
    interior_before[6, 6] = 3
    interior_after = interior_before.copy()
    interior_after[6, 6] = 0
    binding = OnlineEntityCausalBindingStore()

    progressive, progressive_anchor, progressive_observation, _after = _observe(
        binding,
        corner_before,
        corner_after,
        x=0,
        y=0,
        context="carrier-progress",
    )
    noncarrier, control_anchor, control_observation, _after = _observe(
        binding,
        interior_before,
        interior_after,
        x=3,
        y=3,
        context="noncarrier-progress",
    )
    progressive_prediction = binding.predict(
        option_id="option",
        objective=_objective(),
        observation=progressive_observation,
        action_signature=progressive_anchor.concrete_signature,
    )
    control_prediction = binding.predict(
        option_id="option",
        objective=_objective(),
        observation=control_observation,
        action_signature=control_anchor.concrete_signature,
    )

    assert progressive["carrier"] is True
    assert noncarrier["carrier"] is False
    assert progressive_prediction.status == EntityBindingStatus.PROGRESSIVE_CARRIER
    assert control_prediction.status == EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST
    assert control_prediction.controlled_contrast is True
    assert control_prediction.conflict_observed is True
    assert binding.summary()["binding_conflicts"] == 1


def test_binding_ablation_records_no_observation_or_prediction():
    before = np.zeros((5, 5), dtype=np.int32)
    before[2, 2] = 8
    after = np.zeros_like(before)
    binding = OnlineEntityCausalBindingStore(enabled=False)

    outcome, anchor, observation, _after = _observe(
        binding,
        before,
        after,
        x=2,
        y=2,
        context="disabled",
    )
    binding.predict(
        option_id="option",
        objective=_objective(),
        observation=observation,
        action_signature=anchor.concrete_signature,
    )

    assert outcome == {"observed": False}
    summary = binding.summary()
    assert summary["enabled"] is False
    assert summary["observations"] == 0
    assert summary["predictions"] == 0


def test_controlled_target_contrast_guides_only_persistent_pursuit():
    grid = np.zeros((7, 7), dtype=np.int32)
    grid[0, 0] = 8
    grid[3, 3] = 8
    grid[6, 6] = 3
    observation = _observation(grid)
    options = OnlineCausalOptionStore()
    edge = CausalSubgoalEdge(
        edge_key="causal::source=>terminal::exhaust::color3",
        source_objective_id="source",
        target_objective_id="terminal::exhaust::color3",
        minimum_independent_support=2,
    )
    edge.support_events = 2
    edge.support_contexts.update({"a", "b"})
    edge.support_branches.update({0, 1})
    options.sync_confirmed_edges([edge])
    option = options.options()[0]
    options.note_openings([edge.edge_key], context_signature="opened")
    active = options._active
    assert active is not None
    subgoal = EffectConditionedDownstreamSubgoal(
        subgoal_id="subgoal",
        option_id=option.option_id,
        trigger_effect_signature="effect",
        objective_id="terminal::exhaust::color3",
        objective_family="exhaust",
        progress_events=1,
        progress_contexts={"progress"},
    )
    options.downstream_subgoals._subgoals[subgoal.subgoal_id] = subgoal
    active.downstream_subgoal_id = subgoal.subgoal_id
    active.downstream_subgoal_attempts[subgoal.subgoal_id] = 2
    selected_subgoal = EffectConditionedSubgoalSelection(
        subgoal_id=subgoal.subgoal_id,
        option_id=option.option_id,
        objective_id=subgoal.objective_id,
        trigger_effect_signature=subgoal.trigger_effect_signature,
        status=EffectConditionedSubgoalStatus.PROGRESS_SUPPORTED,
        distance=1.0,
        utility=1.0,
        best_progress_sequence=(),
    )
    clicks = (
        SimpleNamespace(action_args={"x": 0, "y": 0}),
        SimpleNamespace(action_args={"x": 3, "y": 3}),
    )
    first = semantic_intervention_anchor(
        "ACTION6", {"x": 0, "y": 0}, observation
    )
    second = semantic_intervention_anchor(
        "ACTION6", {"x": 3, "y": 3}, observation
    )
    base = {
        "option_id": option.option_id,
        "objective_id": subgoal.objective_id,
        "mode_signature": "mode",
        "legacy_signature": "ACTION6::color:8",
        "expected_gain": 0.0,
        "confidence": 0.5,
        "conflict_observed": True,
        "track_id": "",
    }
    predictions = {
        first.concrete_signature: EntityBindingPrediction(
            action_signature=first.concrete_signature,
            action_transfer_signature=first.transfer_signature,
            status=EntityBindingStatus.MISBOUND,
            compatible=False,
            controlled_contrast=False,
            reason="misbound",
            **base,
        ),
        second.concrete_signature: EntityBindingPrediction(
            action_signature=second.concrete_signature,
            action_transfer_signature=second.transfer_signature,
            status=EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST,
            compatible=True,
            controlled_contrast=True,
            reason="controlled contrast",
            **base,
        ),
    }

    selection = options.select_downstream(
        observation,
        safe_actions=["ACTION6"],
        click_actions=clicks,
        downstream_subgoal=selected_subgoal,
        entity_binding_predictions=predictions,
    )

    assert selection is not None
    assert selection.action_data == {"x": 3, "y": 3}
    assert selection.persistent_pursuit is True
    assert selection.entity_binding_controlled_contrast is True
    assert selection.entity_binding_conflict_observed is True
    binding_summary = options.entity_causal_bindings.summary()
    assert binding_summary["blocked_misbound_actions"] == 1
    assert binding_summary["controlled_contrast_selections"] == 1
    pursuit = options.persistent_pursuit.summary()
    assert pursuit["entity_binding_contrast_actions"] == 1


def test_correspondence_status_values_are_auditable():
    assert {status.value for status in EntityCorrespondenceStatus} == {
        "stable",
        "moved",
        "transformed",
        "removed",
        "ambiguous",
    }
