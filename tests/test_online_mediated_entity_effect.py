"""Indirect affected-entity induction tests for SAGE.8v."""

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
)
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.online_mediated_entity_effect import (
    MediatedEffectPrediction,
    MediatedEffectStatus,
    OnlineMediatedEntityEffectStore,
    SceneCorrespondenceStatus,
)
from theory.online_semantic_intervention import semantic_intervention_anchor
from theory.online_terminal_objective import OnlineTerminalObjectiveStore


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION6"],
        infer_players=False,
    )


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
        generation_reason="synthetic_mediated_entity_effect_test",
        prior_priority=1.0,
    ))
    return store


def _objective():
    objective = _objective_store().objective("terminal::exhaust::color3")
    assert objective is not None
    return objective


def _observe(
    memory: OnlineMediatedEntityEffectStore,
    before: np.ndarray,
    after: np.ndarray,
    *,
    x: int,
    y: int,
    context: str,
):
    observation_before = _observation(before)
    action_data = {"x": x, "y": y}
    anchor = semantic_intervention_anchor(
        "ACTION6",
        action_data,
        observation_before,
    )
    outcome = memory.observe(
        option_id="option",
        objective=_objective(),
        observation_before=observation_before,
        observation_after=_observation(after),
        action_data=action_data,
        action_signature=anchor.concrete_signature,
        anchor=anchor,
        branch_index=0,
        context_signature=context,
    )
    return outcome, anchor, observation_before


def _target_and_mediator(*, target=(3, 3), mediator=(3, 5)) -> np.ndarray:
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[target] = 8
    grid[mediator] = 3
    return grid


def test_scene_matching_tracks_a_moved_indirect_entity_across_frames():
    before = _target_and_mediator()
    moved = before.copy()
    moved[3, 5] = 0
    moved[3, 6] = 3
    moved_again = moved.copy()
    moved_again[3, 6] = 0
    moved_again[3, 7] = 3
    memory = OnlineMediatedEntityEffectStore()

    first, _anchor, _observation = _observe(
        memory, before, moved, x=3, y=3, context="move-1"
    )
    second, _anchor, _observation = _observe(
        memory, moved, moved_again, x=3, y=3, context="move-2"
    )
    first_moved = next(
        item for item in first["scene_correspondences"]
        if item["status"] == "moved"
    )
    second_moved = next(
        item for item in second["scene_correspondences"]
        if item["status"] == "moved"
    )

    assert first_moved["track_id"] == second_moved["track_id"]
    summary = memory.summary()
    assert summary["moved_entities"] == 2
    assert summary["tracks_reused"] >= 2


def test_scene_matching_distinguishes_transform_remove_and_appearance():
    memory = OnlineMediatedEntityEffectStore()
    transformed_before = _target_and_mediator()
    transformed_after = transformed_before.copy()
    transformed_after[3, 5] = 4
    transformed, _anchor, _observation = _observe(
        memory,
        transformed_before,
        transformed_after,
        x=3,
        y=3,
        context="transform",
    )
    removed_before = _target_and_mediator()
    removed_after = removed_before.copy()
    removed_after[3, 5] = 0
    removed, _anchor, _observation = _observe(
        memory,
        removed_before,
        removed_after,
        x=3,
        y=3,
        context="remove",
    )
    appeared_before = np.zeros((12, 12), dtype=np.int32)
    appeared_before[3, 3] = 8
    appeared_after = appeared_before.copy()
    appeared_after[7:9, 7:9] = 6
    appeared, _anchor, _observation = _observe(
        memory,
        appeared_before,
        appeared_after,
        x=3,
        y=3,
        context="appear",
    )

    assert any(
        item["status"] == "transformed"
        for item in transformed["scene_correspondences"]
    )
    assert any(
        item["status"] == "removed"
        for item in removed["scene_correspondences"]
    )
    assert any(
        item["status"] == "appeared"
        for item in appeared["scene_correspondences"]
    )


def test_one_progress_event_requests_a_mediator_contrast():
    before = _target_and_mediator()
    after = before.copy()
    after[3, 5] = 0
    memory = OnlineMediatedEntityEffectStore()

    outcome, anchor, observation = _observe(
        memory,
        before,
        after,
        x=3,
        y=3,
        context="first-progress",
    )
    prediction = memory.predict(
        option_id="option",
        objective=_objective(),
        observation=observation,
        action_signature=anchor.concrete_signature,
    )

    assert outcome["gain"] > 0.0
    assert len(outcome["candidate_mediator_signatures"]) == 1
    assert prediction.status == MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST
    assert prediction.controlled_contrast is True
    assert prediction.compatible is True


def test_target_carried_progress_is_not_recredited_to_an_indirect_change():
    before = _target_and_mediator()
    after = before.copy()
    after[3, 3] = 0
    after[3, 5] = 0
    memory = OnlineMediatedEntityEffectStore()

    outcome, anchor, observation = _observe(
        memory,
        before,
        after,
        x=3,
        y=3,
        context="direct-target-progress",
    )
    prediction = memory.predict(
        option_id="option",
        objective=_objective(),
        observation=observation,
        action_signature=anchor.concrete_signature,
    )

    assert outcome["target_stable_for_mediation"] is False
    assert outcome["candidate_mediator_signatures"] == []
    assert prediction.status == MediatedEffectStatus.UNKNOWN
    assert memory.summary()["direct_target_progress_events"] == 1


def test_translated_controlled_repetitions_support_one_indirect_carrier():
    first_before = _target_and_mediator(target=(3, 3), mediator=(3, 5))
    first_after = first_before.copy()
    first_after[3, 5] = 0
    second_before = _target_and_mediator(target=(6, 6), mediator=(6, 8))
    second_after = second_before.copy()
    second_after[6, 8] = 0
    memory = OnlineMediatedEntityEffectStore()

    first, _anchor, _observation = _observe(
        memory,
        first_before,
        first_after,
        x=3,
        y=3,
        context="translated-progress-1",
    )
    second, anchor, observation = _observe(
        memory,
        second_before,
        second_after,
        x=6,
        y=6,
        context="translated-progress-2",
    )
    prediction = memory.predict(
        option_id="option",
        objective=_objective(),
        observation=observation,
        action_signature=anchor.concrete_signature,
    )

    assert (
        first["candidate_mediator_signatures"]
        == second["candidate_mediator_signatures"]
    )
    assert prediction.status == MediatedEffectStatus.SUPPORTED
    assert prediction.supported_mediator_signature
    assert prediction.controlled_contrast is False
    assert memory.summary()["supported_hyperedges"] == 1


def test_candidate_set_intersection_eliminates_a_concomitant_change():
    before = _target_and_mediator()
    before[7:9, 7:9] = 4
    changed_together = before.copy()
    changed_together[3, 5] = 0
    changed_together[7:9, 7:9] = 5
    mediator_only = before.copy()
    mediator_only[3, 5] = 0
    memory = OnlineMediatedEntityEffectStore()

    first, _anchor, _observation = _observe(
        memory,
        before,
        changed_together,
        x=3,
        y=3,
        context="two-candidates",
    )
    second, anchor, observation = _observe(
        memory,
        before,
        mediator_only,
        x=3,
        y=3,
        context="one-candidate",
    )
    prediction = memory.predict(
        option_id="option",
        objective=_objective(),
        observation=observation,
        action_signature=anchor.concrete_signature,
    )

    assert len(first["candidate_mediator_signatures"]) == 2
    assert len(second["candidate_mediator_signatures"]) == 1
    assert prediction.status == MediatedEffectStatus.SUPPORTED
    assert prediction.candidate_mediator_signatures == tuple(
        second["candidate_mediator_signatures"]
    )


def test_ambiguous_scene_matching_never_creates_a_progress_candidate_set():
    before = _target_and_mediator(target=(1, 1), mediator=(5, 5))
    after = before.copy()
    after[5, 5] = 0
    after[5, 4] = 3
    after[5, 6] = 3
    memory = OnlineMediatedEntityEffectStore()

    outcome, _anchor, _observation = _observe(
        memory,
        before,
        after,
        x=1,
        y=1,
        context="ambiguous",
    )

    assert any(
        item["status"] == SceneCorrespondenceStatus.AMBIGUOUS.value
        for item in outcome["scene_correspondences"]
    )
    assert outcome["carrier_candidates"] == []
    assert memory.summary()["ambiguous_entities"] >= 1


def test_mediated_effect_ablation_records_nothing():
    before = _target_and_mediator()
    after = before.copy()
    after[3, 5] = 0
    memory = OnlineMediatedEntityEffectStore(enabled=False)

    outcome, anchor, observation = _observe(
        memory,
        before,
        after,
        x=3,
        y=3,
        context="disabled",
    )
    memory.predict(
        option_id="option",
        objective=_objective(),
        observation=observation,
        action_signature=anchor.concrete_signature,
    )

    assert outcome == {"observed": False}
    summary = memory.summary()
    assert summary["observations"] == 0
    assert summary["predictions"] == 0


def test_mediated_contrast_can_override_a_misbound_target_in_persistence():
    grid = _target_and_mediator()
    grid[8, 8] = 8
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
    selection_goal = EffectConditionedSubgoalSelection(
        subgoal_id=subgoal.subgoal_id,
        option_id=option.option_id,
        objective_id=subgoal.objective_id,
        trigger_effect_signature=subgoal.trigger_effect_signature,
        status=EffectConditionedSubgoalStatus.PROGRESS_SUPPORTED,
        distance=1.0,
        utility=1.0,
        best_progress_sequence=(),
    )
    anchor = semantic_intervention_anchor(
        "ACTION6", {"x": 3, "y": 3}, observation
    )
    alternative_anchor = semantic_intervention_anchor(
        "ACTION6", {"x": 8, "y": 8}, observation
    )
    binding = EntityBindingPrediction(
        option_id=option.option_id,
        objective_id=subgoal.objective_id,
        mode_signature="mode",
        action_signature=anchor.concrete_signature,
        action_transfer_signature=anchor.transfer_signature,
        legacy_signature=anchor.legacy_signature,
        status=EntityBindingStatus.MISBOUND,
        expected_gain=0.0,
        confidence=0.5,
        compatible=False,
        controlled_contrast=False,
        conflict_observed=False,
        track_id="track",
        reason="target stayed stable",
    )
    mediated = MediatedEffectPrediction(
        option_id=option.option_id,
        objective_id=subgoal.objective_id,
        mode_signature="mode",
        action_signature=anchor.concrete_signature,
        action_transfer_signature=anchor.transfer_signature,
        status=MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST,
        expected_gain=1.0,
        confidence=0.5,
        compatible=True,
        controlled_contrast=True,
        supported_mediator_signature="",
        candidate_mediator_signatures=("candidate",),
        reason="repeat to isolate mediator",
    )
    alternative_binding = EntityBindingPrediction(
        option_id=option.option_id,
        objective_id=subgoal.objective_id,
        mode_signature="mode",
        action_signature=alternative_anchor.concrete_signature,
        action_transfer_signature=alternative_anchor.transfer_signature,
        legacy_signature=alternative_anchor.legacy_signature,
        status=EntityBindingStatus.UNKNOWN,
        expected_gain=0.0,
        confidence=0.0,
        compatible=True,
        controlled_contrast=False,
        conflict_observed=False,
        track_id="",
        reason="unknown target",
    )
    alternative_mediated = MediatedEffectPrediction(
        option_id=option.option_id,
        objective_id=subgoal.objective_id,
        mode_signature="mode",
        action_signature=alternative_anchor.concrete_signature,
        action_transfer_signature=alternative_anchor.transfer_signature,
        status=MediatedEffectStatus.UNKNOWN,
        expected_gain=0.0,
        confidence=0.0,
        compatible=True,
        controlled_contrast=False,
        supported_mediator_signature="",
        candidate_mediator_signatures=(),
        reason="unknown mediated effect",
    )

    selection = options.select_downstream(
        observation,
        safe_actions=["ACTION6"],
        click_actions=(
            SimpleNamespace(action_args={"x": 3, "y": 3}),
            SimpleNamespace(action_args={"x": 8, "y": 8}),
        ),
        downstream_subgoal=selection_goal,
        entity_binding_predictions={
            anchor.concrete_signature: binding,
            alternative_anchor.concrete_signature: alternative_binding,
        },
        mediated_effect_predictions={
            anchor.concrete_signature: mediated,
            alternative_anchor.concrete_signature: alternative_mediated,
        },
    )

    assert selection is not None
    assert selection.action_data == {"x": 3, "y": 3}
    assert selection.persistent_pursuit is True
    assert selection.entity_binding_status == "misbound"
    assert selection.mediated_effect_status == "needs_mediator_contrast"
    assert selection.mediated_effect_controlled_contrast is True
    assert (
        options.persistent_pursuit.summary()[
            "mediated_effect_contrast_actions"
        ]
        == 1
    )
