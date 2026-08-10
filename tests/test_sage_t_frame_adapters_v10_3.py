from __future__ import annotations

import pytest

from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    ObservedTransition,
    PredictionPacket,
)
from theory.sage_t.frame_adapters_v10_3 import (
    assert_safe_binding_payload,
    project_goal_transition,
    resolve_pre_action_root,
)


def _entity(entity_id: str, *roles: str, center=None, attributes=()):
    return AbstractEntity(entity_id, roles, attributes=attributes, center=center)


def test_movement_actor_has_priority_over_ambiguous_targets() -> None:
    state = AbstractState(
        entities=(
            _entity("player-local", "actor", "player"),
            _entity("target-a", "target"),
            _entity("target-b", "target"),
        )
    )
    resolved = resolve_pre_action_root(state, ActionCandidate("ACTION4"))
    assert resolved.entity_id == "player-local"
    assert resolved.method == "movement_actor"
    assert resolved.unique is True


def test_parameterized_action_uses_unique_transient_anchor() -> None:
    state = AbstractState(
        entities=(
            _entity("a", "target", center=(2, 1)),
            _entity("b", "target", center=(7, 7)),
        )
    )
    resolved = resolve_pre_action_root(
        state, ActionCandidate("ACTION6", {"x": 1, "y": 2})
    )
    assert resolved.entity_id == "a"
    assert resolved.method == "transient_action_anchor"


def test_parameterized_anchor_tie_remains_ambiguous() -> None:
    state = AbstractState(
        entities=(
            _entity("a", "target", center=(0, 0)),
            _entity("b", "target", center=(0, 2)),
        )
    )
    resolved = resolve_pre_action_root(
        state, ActionCandidate("ACTION6", {"x": 1, "y": 0})
    )
    assert resolved.entity_id is None
    assert resolved.unique is False
    assert resolved.method == "ambiguous_transient_action_anchor"


def test_after_root_is_not_inferred_from_action_effects() -> None:
    before = AbstractState(entities=(_entity("before-root", "actor", "player"),))
    after = AbstractState(entities=(_entity("changed-object", "target"),))
    evidence = ObservedTransition(
        state_before=before,
        action=ActionCandidate("ACTION1"),
        state_after=after,
        observation=PredictionPacket(),
    )
    projection = project_goal_transition(evidence, event_id="synthetic-event")
    assert projection.binding.unique is True
    assert projection.binding.after_signature_match is False
    assert projection.binding.complete is False
    assert "after_structural_signature_match" in projection.binding.missing


def test_pre_action_binding_can_continue_by_transient_branch_local_id() -> None:
    before = AbstractState(entities=(_entity("local", "actor", "player"),))
    after = AbstractState(entities=(_entity("local", "actor", "player", attributes=(("area", "large"),)),))
    projection = project_goal_transition(
        ObservedTransition(
            state_before=before,
            action=ActionCandidate("ACTION2"),
            state_after=after,
            observation=PredictionPacket(),
        ),
        event_id="same-branch-binding",
    )
    assert projection.binding.complete is True
    assert projection.binding.after_signature_match is True


def test_binding_serialization_rejects_raw_grounding_fields() -> None:
    assert_safe_binding_payload(
        {
            "method": "movement_actor",
            "structural_signature": "a" * 64,
            "unique": True,
        }
    )
    with pytest.raises(ValueError, match="forbidden grounding field"):
        assert_safe_binding_payload({"method": "anchor", "x": 4})
