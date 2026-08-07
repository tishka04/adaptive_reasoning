from __future__ import annotations

from dataclasses import replace

import pytest

from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
)
from theory.sage_t.frame_adapters_v10_2 import (
    FRAME_PROJECTORS,
    concrete_frame_projectors,
    frame_projector,
    project_action_aligned_relational,
    project_frozen_frame,
    project_transition_with_frozen_frames,
)
from theory.sage_t.observer_frames_v10_2 import (
    ACTION_ALIGNED_RELATIONAL_FRAME,
    ACTION_ROOTED_TOPOLOGICAL_FRAME,
    ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
    OBSERVER_FRAME_SPECS,
    ROOT_ONLY_FRAME,
    FrameProjection,
    ObserverFrameSpec,
    audit_identity_leaks,
    canonical_json,
    project_state,
    state_model_payload,
)


def _scene(
    *,
    names: tuple[str, str, str] = ("actor_local", "target_local", "obstacle_local"),
    offset: tuple[float, float] = (0.0, 0.0),
    order: tuple[int, ...] = (0, 1, 2),
    rotated: bool = False,
    unsafe: bool = False,
) -> tuple[AbstractState, ActionCandidate]:
    actor_id, target_id, obstacle_id = names
    row, col = offset
    centers = (
        ((row, col), (row, col + 2.0), (row, col + 4.0))
        if not rotated
        else ((row, col), (row + 2.0, col), (row + 4.0, col))
    )
    target_attributes = [("area", "medium"), ("aspect", "wide"), ("holes", "1")]
    if unsafe:
        target_attributes.extend((("color", "7"), ("value", "#ff00aa")))
    entities = (
        AbstractEntity(
            actor_id,
            ("actor", "movable", "object"),
            (("area", "small"), ("aspect", "compact")),
            centers[0],
        ),
        AbstractEntity(
            target_id,
            ("object", "target") + (("bp35",) if unsafe else ()),
            tuple(target_attributes),
            centers[1],
        ),
        AbstractEntity(
            obstacle_id,
            ("object", "obstacle"),
            (("area", "one"), ("aspect", "tall")),
            centers[2],
        ),
    )
    facts = {
        GroundFact("exists", (actor_id,)),
        GroundFact("exists", (target_id,)),
        GroundFact("exists", (obstacle_id,)),
        GroundFact("contact", (actor_id, target_id)),
        GroundFact("contact", (target_id, obstacle_id)),
        GroundFact(
            "east_of" if not rotated else "south_of",
            (target_id, actor_id),
        ),
        GroundFact(
            "east_of" if not rotated else "south_of",
            (obstacle_id, target_id),
        ),
    }
    if unsafe:
        facts.add(GroundFact("same_color", (target_id, obstacle_id), value="7"))
    state = AbstractState(
        entities=tuple(entities[index] for index in order),
        true_facts=frozenset(facts),
        registers=(("seed", "bp35-0a0ad940"),),
        topology=(("seed", 2101), ("component_count", 3)),
        regime_index=2101,
    )
    action = ActionCandidate(
        "ACTION2" if rotated else "ACTION4",
        {
            "target_object_id": target_id,
            "x": centers[1][1],
            "y": centers[1][0],
            "game_id": "bp35-0a0ad940",
        },
    )
    return state, action


def _transition() -> ObservedTransition:
    before, action = _scene()
    target_id = "target_local"
    after = replace(
        before,
        true_facts=before.true_facts | {GroundFact("solved", (target_id,))},
    )
    return ObservedTransition(
        state_before=before,
        action=action,
        state_after=after,
        observation=PredictionPacket(
            object_deltas={"solved": 1.0},
            progress_mean=1.0,
            progress_distribution={"increase": 1.0},
            terminal_probability=0.0,
            goal_probability=1.0,
            known_channels=frozenset({"objects", "progress", "terminal", "goal"}),
            state_after=after,
        ),
        events=("progress",),
    )


def test_registry_covers_exactly_the_four_frozen_frames() -> None:
    assert tuple(FRAME_PROJECTORS) == tuple(
        frame.frame_id for frame in OBSERVER_FRAME_SPECS
    )
    assert concrete_frame_projectors() == dict(FRAME_PROJECTORS)
    assert frame_projector(ROOT_ONLY_FRAME) is FRAME_PROJECTORS["root_only"]
    with pytest.raises(TypeError):
        FRAME_PROJECTORS["root_only"] = project_action_aligned_relational  # type: ignore[index]
    with pytest.raises(ValueError, match="no concrete projector"):
        frame_projector("camera_relative")
    with pytest.raises(ValueError, match="not the frozen"):
        frame_projector(ObserverFrameSpec("root_only", "root_only", "test_v1", True))


@pytest.mark.parametrize("frame", OBSERVER_FRAME_SPECS)
def test_projectors_are_deterministic_and_permutation_identity_invariant(
    frame: ObserverFrameSpec,
) -> None:
    first_state, first_action = _scene()
    renamed_state, renamed_action = _scene(
        names=(
            "actor_renamed",
            "target_renamed",
            "obstacle_renamed",
        ),
        offset=(100.0, -75.0),
        order=(2, 0, 1),
    )

    first = project_frozen_frame(frame, first_state, first_action, "before")
    repeated = project_frozen_frame(frame, first_state, first_action, "before")
    renamed = project_frozen_frame(frame, renamed_state, renamed_action, "before")

    assert first == repeated
    assert first.complete is True
    assert "counters" in first.covered_channels
    assert state_model_payload(first.state) == state_model_payload(renamed.state)


@pytest.mark.parametrize("frame", OBSERVER_FRAME_SPECS)
def test_canonical_outputs_strip_colors_coordinates_and_all_source_identity(
    frame: ObserverFrameSpec,
) -> None:
    source, action = _scene(
        names=(
            "bp35-0a0ad940_actor",
            "persistent_target_17",
            "550e8400-e29b-41d4-a716-446655440000",
        ),
        offset=(12.0, 31.0),
        unsafe=True,
    )
    projection = project_state(
        source,
        action,
        frame,
        stage="before",
        projector=frame_projector(frame),
    )
    rendered = canonical_json(projection.canonical_payload).lower()

    assert audit_identity_leaks(projection.canonical_payload) == ()
    assert "bp35" not in rendered
    assert "persistent_target" not in rendered
    assert "550e8400" not in rendered
    assert "color" not in rendered
    assert "#ff00aa" not in rendered
    assert "2101" not in rendered
    assert "12.0" not in rendered
    assert "31.0" not in rendered
    assert "action_data" not in rendered


def test_action_aligned_frame_is_invariant_to_joint_scene_action_rotation() -> None:
    horizontal_state, horizontal_action = _scene()
    vertical_state, vertical_action = _scene(rotated=True)

    horizontal = project_frozen_frame(
        ACTION_ALIGNED_RELATIONAL_FRAME,
        horizontal_state,
        horizontal_action,
        "before",
    )
    vertical = project_frozen_frame(
        ACTION_ALIGNED_RELATIONAL_FRAME,
        vertical_state,
        vertical_action,
        "before",
    )
    horizontal_payload = state_model_payload(horizontal.state)
    rendered = canonical_json(horizontal_payload)

    assert horizontal_payload == state_model_payload(vertical.state)
    assert '"north"' not in rendered
    assert '"south"' not in rendered
    assert '"east"' not in rendered
    assert '"west"' not in rendered
    assert '"ahead"' in rendered
    assert '"behind"' in rendered


def test_allocentric_frame_retains_only_relative_compass_relations() -> None:
    state, action = _scene(offset=(250.0, -400.0))
    projection = project_frozen_frame(
        ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
        state,
        action,
        "before",
    )
    rendered = canonical_json(state_model_payload(projection.state))

    assert '"east"' in rendered
    assert "250" not in rendered
    assert "400" not in rendered
    assert "target_local" not in rendered


def test_allocentric_geometry_can_be_recovered_from_relational_facts() -> None:
    root = AbstractEntity("root_local", ("object", "target"))
    neighbor = AbstractEntity("neighbor_local", ("object",))
    state = AbstractState(
        entities=(neighbor, root),
        true_facts=frozenset(
            {
                GroundFact("north_of", ("neighbor_local", "root_local")),
                GroundFact("adjacent", ("neighbor_local", "root_local")),
            }
        ),
    )
    action = ActionCandidate("ACTION5", {"target_id": "root_local"})

    projection = project_frozen_frame(
        ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
        state,
        action,
        "before",
    )
    rendered = canonical_json(state_model_payload(projection.state))

    assert projection.complete is True
    assert '"north"' in rendered
    assert '"adjacent"' in rendered


def test_topological_frame_exports_invariants_not_the_ephemeral_graph() -> None:
    state, action = _scene()
    projection = project_frozen_frame(
        ACTION_ROOTED_TOPOLOGICAL_FRAME,
        state,
        action,
        "before",
    )
    topology = dict(projection.state.topology)
    rendered = canonical_json(state_model_payload(projection.state))

    assert topology["connected_components"] == 1
    assert topology["structural_edges"] == 2
    assert topology["articulation_points"] == 1
    assert topology["bridges"] == 2
    assert topology["root_is_articulation"] == 1
    assert topology["root_bridge_incidence"] == 2
    assert topology["actor_root_distance"] == 1
    assert "cells" not in rendered
    assert "center" not in rendered
    assert "node_id" not in rendered
    assert "relations" not in rendered


def test_missing_grounding_and_geometry_return_explicit_partial_views() -> None:
    empty = AbstractState()
    action = ActionCandidate("ACTION6")
    for frame in OBSERVER_FRAME_SPECS:
        projection = project_frozen_frame(frame, empty, action, "before")
        assert projection.complete is False
        assert "root_entity" in projection.missing
        assert state_model_payload(projection.state)

    ungrounded = AbstractState(
        entities=(
            AbstractEntity("left", ("object",)),
            AbstractEntity("right", ("object",)),
        )
    )
    aligned = project_frozen_frame(
        ACTION_ALIGNED_RELATIONAL_FRAME,
        ungrounded,
        action,
        "before",
    )
    assert aligned.complete is False
    assert {
        "action_axis",
        "relative_geometry",
        "relative_proximity",
        "root_entity",
    }.issubset(aligned.missing)


def test_neighbor_class_truncation_is_explicitly_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, action = _scene()
    monkeypatch.setattr(
        "theory.sage_t.frame_adapters_v10_2.MAXIMUM_NEIGHBOR_CLASSES",
        1,
    )

    projection = project_frozen_frame(
        ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
        state,
        action,
        "before",
    )

    assert projection.complete is False
    assert "neighbor_class_budget_truncated" in projection.missing
    assert "neighbor_class_budget_truncated" in projection.audit_tags


def test_heuristic_actor_or_target_root_is_explicitly_incomplete() -> None:
    actor = AbstractEntity("actor_local", ("actor", "object"), center=(1.0, 1.0))
    target = AbstractEntity("target_local", ("object", "target"), center=(2.0, 2.0))
    state = AbstractState(entities=(actor, target))
    action = ActionCandidate("ACTION6")

    projection = project_frozen_frame(ROOT_ONLY_FRAME, state, action, "before")

    assert projection.complete is False
    assert projection.missing == ("action_binding",)


@pytest.mark.parametrize(
    ("action", "facts"),
    (
        (
            ActionCandidate(
                "ACTION6",
                {"entity_id": "left", "target_id": "right"},
            ),
            frozenset(),
        ),
        (
            ActionCandidate("ACTION6"),
            frozenset(
                {
                    GroundFact("selected", ("left",)),
                    GroundFact("selected", ("right",)),
                }
            ),
        ),
    ),
)
def test_conflicting_root_evidence_is_explicitly_incomplete(
    action: ActionCandidate,
    facts: frozenset[GroundFact],
) -> None:
    state = AbstractState(
        entities=(
            AbstractEntity("left", ("object", "target")),
            AbstractEntity("right", ("object", "target")),
        ),
        true_facts=facts,
    )

    projection = project_frozen_frame(ROOT_ONLY_FRAME, state, action, "before")

    assert projection.complete is False
    assert "root_entity" in projection.missing


@pytest.mark.parametrize("frame", OBSERVER_FRAME_SPECS)
def test_untrusted_source_topology_values_are_not_forwarded(
    frame: ObserverFrameSpec,
) -> None:
    state, action = _scene()
    poisoned = replace(
        state,
        topology=(
            ("holes", 2101),
            ("actor_root_distance", 2101),
            ("component_count", 2101),
        ),
    )

    projection = project_frozen_frame(frame, poisoned, action, "before")
    rendered = canonical_json(state_model_payload(projection.state))

    assert "2101" not in rendered


def test_full_transition_adapter_emits_one_common_outcome_and_four_views() -> None:
    evidence = _transition()
    first = project_transition_with_frozen_frames(
        evidence,
        event_nonce="episode_1_step_1",
    )
    repeated = project_transition_with_frozen_frames(
        evidence,
        event_nonce="episode_1_step_1",
    )

    assert first.frame_ids == tuple(
        sorted(frame.frame_id for frame in OBSERVER_FRAME_SPECS)
    )
    assert first.common_outcome.state_after is None
    assert first.canonical_checksum == repeated.canonical_checksum
    assert all(projection.complete for projection in first.projections)
    assert all(
        projection.before.provenance and projection.after.provenance
        for projection in first.projections
    )
    for frame_id in first.frame_ids:
        restored = first.observed_transition(frame_id)
        assert (
            first.common_outcome.known_channels <= restored.observation.known_channels
        )
        assert restored.observation.progress_mean == first.common_outcome.progress_mean
        projection = first.projection(frame_id)
        assert {
            "objects",
            "relations",
            "topology",
        } & restored.observation.known_channels == projection.observation.known_channels


def test_wrong_frame_and_invalid_stage_fail_closed() -> None:
    state, action = _scene()
    with pytest.raises(ValueError, match="requires observer frame"):
        project_action_aligned_relational(
            ROOT_ONLY_FRAME,
            state,
            action,
            "before",
        )
    with pytest.raises(ValueError, match="stage must be"):
        project_frozen_frame(ROOT_ONLY_FRAME, state, action, "during")


def test_direct_projection_can_be_wrapped_by_frame_projection_contract() -> None:
    state, action = _scene()
    projected = project_frozen_frame(ROOT_ONLY_FRAME, state, action, "after")
    wrapped = FrameProjection(
        frame=ROOT_ONLY_FRAME,
        state=projected.state,
        action=action,
        stage="after",
        complete=projected.complete,
        missing=projected.missing,
        covered_channels=projected.covered_channels,
        provenance=projected.provenance,
        audit_tags=projected.audit_tags,
    )

    assert wrapped.complete is True
    assert len(wrapped.canonical_checksum) == 64
    assert wrapped.canonical_payload["state"] == state_model_payload(projected.state)
