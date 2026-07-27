from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from theory.sage12.action_target_data import (
    EFFECT_LABELS,
    build_action_target_trace,
)
from theory.sage12.mechanic_induction import (
    MechanicQuery,
    MechanicRule,
    MechanicWindowRecord,
    PersistentRoleTracker,
    SemanticTransitionEvent,
    _shuffle_context,
    build_mechanic_windows,
    fit_source_priors,
    predict_mechanic_effects,
    rule_to_semantic_hypothesis,
    score_rule,
    validate_model_view,
)


def _grid(player=(2, 2), target=None):
    grid = np.zeros((12, 16), dtype=np.int32)
    if player is not None:
        grid[player] = 1
    if target is not None:
        grid[target] = 2
    return grid


def _trace(
    *,
    step=0,
    reset=0,
    before=None,
    after=None,
    action="ACTION4",
):
    before = _grid(player=(2, 2 + step)) if before is None else before
    after = _grid(player=(2, 3 + step)) if after is None else after
    return build_action_target_trace(
        game_id="bp35",
        source_split="source_train",
        policy_seed=131,
        reset_index=reset,
        step_index=step,
        collection_phase="test",
        available_action_names=(
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION6",
        ),
        selected_action_name=action,
        selected_action_data={},
        frame_before=before,
        frame_after=after,
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        levels_completed_before=0,
        levels_completed_after=0,
    )


def _event(
    *,
    action="ACTION4",
    family="move",
    anchor="open",
    moved=True,
):
    return SemanticTransitionEvent(
        action_name=action,
        action_family=family,
        anchor_condition=anchor,
        effects={
            "actor_displaced": moved,
            "target_created": False,
            "target_removed": False,
            "target_moved": False,
        },
        applicable={label: True for label in EFFECT_LABELS},
        actor_role_known=True,
    )


def _window(context=None, query=None, moved=True):
    context = tuple(context or [_event() for _ in range(8)])
    query = query or MechanicQuery("ACTION4", "move", "open")
    return MechanicWindowRecord(
        game_id="bp35",
        source_split="source_train",
        policy_seed=131,
        reset_index=0,
        query_step_index=8,
        context=context,
        query=query,
        labels={
            "actor_displaced": moved,
            "target_created": False,
            "target_removed": False,
            "target_moved": False,
        },
        applicable={label: True for label in EFFECT_LABELS},
        actor_role_known=True,
    )


def test_window_builder_requires_contiguous_frames_and_reset():
    traces = [_trace(step=index) for index in range(10)]
    windows = build_mechanic_windows(traces, context_length=8)

    assert len(windows) == 2
    assert all(len(window.context) == 8 for window in windows)

    broken = list(traces)
    broken[9] = replace(broken[9], reset_index=1)
    assert len(build_mechanic_windows(broken, context_length=8)) == 1


def test_model_view_excludes_provenance_and_outcome():
    window = _window()

    validate_model_view(window)
    rendered = str(window.model_view()).lower()

    assert window.game_id not in rendered
    assert "labels" not in rendered
    assert "policy_seed" not in rendered
    assert "frame" not in rendered


def test_mechanic_rule_never_carries_observed_support():
    with pytest.raises(ValueError, match="support=0"):
        MechanicRule(
            rule_id="bad",
            action_scope_kind="family",
            action_scope_value="move",
            anchor_condition="open",
            effect="actor_displaced",
            support=1,
        )


def test_persistent_tracker_emits_bounded_semantic_event():
    event = PersistentRoleTracker().observe(_trace())

    assert event.action_family == "move"
    assert event.anchor_condition in {
        "open",
        "empty",
        "occupied_actor",
        "occupied_object",
        "unknown",
    }
    assert set(event.effects) == set(EFFECT_LABELS)


def test_rule_evidence_updates_posterior_without_mutating_rule():
    context = [_event(moved=True), _event(moved=True), _event(moved=False)]
    rule = MechanicRule(
        rule_id="move_actor",
        action_scope_kind="family",
        action_scope_value="move",
        anchor_condition="open",
        effect="actor_displaced",
    )
    priors = {
        "family|move|open|actor_displaced": {
            "positive": 1,
            "applicable": 2,
        }
    }

    evidence = score_rule(rule, context, priors)

    assert evidence.observed_support == 2
    assert evidence.observed_refutations == 1
    assert evidence.posterior_probability == pytest.approx(0.6)
    assert rule.support == 0


def test_sequence_conditioned_prediction_uses_local_mechanic():
    positive = _window(moved=True)
    negative = _window(context=[_event(moved=False) for _ in range(8)], moved=False)
    priors = fit_source_priors([positive, negative])

    probabilities, evidence = predict_mechanic_effects(
        positive.context,
        positive.query,
        priors,
    )

    assert probabilities["actor_displaced"] > 0.8
    assert evidence
    assert all(item.rule.support == 0 for item in evidence)


def test_outcome_and_binding_shuffles_preserve_query():
    context = tuple(
        _event(
            action="ACTION4" if index % 2 else "ACTION6",
            family="move" if index % 2 else "click",
            anchor="open" if index % 2 else "occupied_object",
            moved=bool(index % 2),
        )
        for index in range(8)
    )
    window = _window(context=context)

    outcome = _shuffle_context([window], binding=False)[0]
    binding = _shuffle_context([window], binding=True)[0]

    assert outcome.query == window.query
    assert binding.query == window.query
    assert [item.action_name for item in outcome.context] == [
        item.action_name for item in window.context
    ]
    assert [item.anchor_condition for item in binding.context] != [
        item.anchor_condition for item in window.context
    ]


def test_rule_compiles_to_existing_zero_support_hypothesis():
    query = MechanicQuery("ACTION4", "move", "open")
    rule = MechanicRule(
        rule_id="move_actor",
        action_scope_kind="family",
        action_scope_value="move",
        anchor_condition="open",
        effect="actor_displaced",
    )

    hypothesis = rule_to_semantic_hypothesis(rule, query, confidence=0.75)

    assert hypothesis.action_name == "ACTION4"
    assert hypothesis.support == 0
    assert hypothesis.effects[0].predicate.name == "moved"


def test_window_round_trip_preserves_model_view():
    window = _window()
    restored = MechanicWindowRecord.from_dict(window.to_dict())

    assert restored.window_digest == window.window_digest
    assert restored.model_view() == window.model_view()
