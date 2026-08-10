from __future__ import annotations

from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    ObservedTransition,
    PredictionPacket,
)
from theory.sage_t.frame_adapters_v10_3_1 import (
    project_goal_transition,
    shared_quotient_transport,
)


def _entity(entity_id: str, *roles: str, center=None):
    return AbstractEntity(entity_id, roles, center=center)


def test_movement_actor_continues_without_post_effect_inference() -> None:
    before = AbstractState(entities=(_entity("actor-before", "actor", "player"),))
    after = AbstractState(entities=(_entity("actor-after", "actor", "player"),))
    projection = project_goal_transition(
        ObservedTransition(
            before,
            ActionCandidate("ACTION1"),
            after,
            PredictionPacket(),
        ),
        event_id="movement-continuation",
    )
    assert projection.binding.pre_action_complete is True
    assert projection.binding.after_root_available is True
    assert projection.binding.after_method == "movement_actor_continuation"
    assert projection.binding.as_dict()["post_action_effect_inference_used"] is False


def test_terminal_root_disappearance_does_not_erase_pre_action_binding() -> None:
    before = AbstractState(
        entities=(
            _entity("target", "target", center=(2, 3)),
            _entity("other", "object", center=(9, 9)),
        )
    )
    after = AbstractState(entities=())
    projection = project_goal_transition(
        ObservedTransition(
            before,
            ActionCandidate("ACTION6", {"x": 3, "y": 2}),
            after,
            PredictionPacket(terminal_probability=1.0, known_channels=frozenset({"terminal"})),
            events=("level_complete",),
        ),
        event_id="terminal-disappearance",
    )
    assert projection.binding.complete is True
    assert projection.binding.after_root_available is False
    assert "after_root_unavailable" in projection.binding.missing


def _endpoint(*facts):
    return {
        "entity_count": 2,
        "regime_index": 0,
        "role_rows": [{"count": 1, "roles": ["action_root", "target"]}],
        "fact_rows": list(facts),
        "counter_rows": [],
        "register_rows": [],
        "topology_rows": [{"amount": 1, "name": "component_count"}],
    }


def test_orientation_erased_common_quotient_is_exact_and_nonidentity() -> None:
    exists = {"arity": 1, "count": 2, "has_literal": False, "predicate": "exists", "truth": True}
    north = {"arity": 2, "count": 1, "has_literal": False, "predicate": "north_of", "truth": True}
    ahead = {"arity": 2, "count": 1, "has_literal": False, "predicate": "ahead_of", "truth": True}
    frames = {
        "root_only": {"complete": True, "before": _endpoint(exists), "after": _endpoint(exists)},
        "allocentric_object_relative": {
            "complete": True,
            "before": _endpoint(exists, north),
            "after": _endpoint(exists),
        },
        "action_aligned_relational": {
            "complete": True,
            "before": _endpoint(exists, ahead),
            "after": _endpoint(exists),
        },
        "action_rooted_topological": {"complete": True, "before": _endpoint(exists), "after": _endpoint(exists)},
    }
    summary, certificates = shared_quotient_transport(frames)
    assert summary["multiframe_exact_nonidentity"] is True
    assert summary["common_quotient_changed"] is False
    assert certificates[0]["source_frame"] != certificates[0]["target_frame"]
    assert certificates[0]["round_trip_exact"] is True
    assert all(item["comparable"] is False for item in certificates[1:])


def test_incomplete_pair_is_visible_but_never_exact() -> None:
    frames = {
        "root_only": {"complete": True},
        "allocentric_object_relative": {"complete": True},
        "action_aligned_relational": {"complete": False},
        "action_rooted_topological": {"complete": False},
    }
    summary, certificates = shared_quotient_transport(frames)
    assert summary["exact_nonidentity_certificate_count"] == 0
    assert certificates[0]["comparable"] is False
    assert certificates[0]["exact"] is False

