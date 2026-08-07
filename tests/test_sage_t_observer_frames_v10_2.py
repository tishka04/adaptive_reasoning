from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
)
from theory.sage_t.observer_frames_v10_2 import (
    ACTION_ALIGNED_RELATIONAL_FRAME,
    ACTION_ROOTED_TOPOLOGICAL_FRAME,
    ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
    OBSERVER_FRAME_SPECS,
    PREDICTIVE_PROJECTION_CHANNELS,
    PROJECTION_CHANNEL_VOCABULARY,
    PROJECTION_CHANNEL_VOCABULARY_VERSION,
    PROJECTION_CHANNELS,
    ROOT_ONLY_FRAME,
    STATE_PROJECTION_CHANNELS,
    FrameProjection,
    ObserverFrameSpec,
    PhysicalEventBundle,
    ProjectedState,
    ProjectedTransition,
    audit_identity_leaks,
    canonical_json,
    identity_projector,
    observed_transition_event_id,
    observer_frame_spec,
    project_observed_transition,
    project_state,
    state_model_payload,
    validate_unique_event_ids,
)


def _state(
    *,
    changed: bool = False,
    entity_id: str = "local_target",
    roles: tuple[str, ...] = ("object", "target"),
    attributes: tuple[tuple[str, str], ...] = (("size", "small"),),
    center: tuple[float, float] = (2.0, 3.0),
) -> AbstractState:
    facts = {GroundFact("exists", (entity_id,))}
    if changed:
        facts.add(GroundFact("changed", (entity_id,)))
    return AbstractState(
        entities=(
            AbstractEntity(
                entity_id=entity_id,
                roles=roles,
                attributes=attributes,
                center=center,
            ),
        ),
        true_facts=frozenset(facts),
        counters=(("step_count", float(changed)),),
        topology=(("component_count", 1),),
    )


def _transition(
    *,
    action: ActionCandidate | None = None,
    events: tuple[str, ...] = ("progress",),
) -> ObservedTransition:
    before = _state()
    after = _state(changed=True)
    selected = action or ActionCandidate(
        "ACTION2",
        {
            "x": 7,
            "y": 9,
            "target_object_id": "branch_target_42",
        },
    )
    outcome = PredictionPacket(
        object_deltas={"changed": 1.0},
        relation_deltas={"contact_added": 0.75},
        topology_deltas={"component_count_changed": 1.0},
        progress_mean=1.0,
        progress_distribution={"increase": 1.0},
        terminal_probability=0.0,
        goal_probability=0.25,
        known_channels=frozenset(
            {"objects", "relations", "topology", "progress", "terminal", "goal"}
        ),
        state_after=after,
    )
    return ObservedTransition(
        state_before=before,
        action=selected,
        state_after=after,
        observation=outcome,
        events=events,
    )


def _deterministic_projector(
    frame: ObserverFrameSpec,
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> ProjectedState:
    del action
    frame_rank = tuple(item.frame_id for item in OBSERVER_FRAME_SPECS).index(
        frame.frame_id
    )
    stage_rank = 0 if stage == "before" else 1
    projected = replace(
        state,
        counters=state.counters
        + (
            ("frame_rank", float(frame_rank)),
            ("stage_rank", float(stage_rank)),
        ),
    )
    return ProjectedState(
        state=projected,
        covered_channels=("facts", "topology"),
        provenance=("unit_test_projector",),
    )


def test_four_frame_specs_are_frozen_versioned_and_deterministic() -> None:
    assert OBSERVER_FRAME_SPECS == (
        ROOT_ONLY_FRAME,
        ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
        ACTION_ALIGNED_RELATIONAL_FRAME,
        ACTION_ROOTED_TOPOLOGICAL_FRAME,
    )
    assert tuple(frame.frame_id for frame in OBSERVER_FRAME_SPECS) == (
        "root_only",
        "allocentric_object_relative",
        "action_aligned_relational",
        "action_rooted_topological",
    )
    assert len({frame.canonical_hash for frame in OBSERVER_FRAME_SPECS}) == 4
    assert all(
        frame.canonical_checksum == frame.canonical_hash
        for frame in OBSERVER_FRAME_SPECS
    )
    assert observer_frame_spec("ROOT_ONLY") is ROOT_ONLY_FRAME
    with pytest.raises(ValueError, match="unknown observer frame"):
        observer_frame_spec("camera_relative")
    with pytest.raises(FrozenInstanceError):
        ROOT_ONLY_FRAME.frame_id = "changed"  # type: ignore[misc]


def test_projected_event_is_deterministic_and_has_one_common_outcome() -> None:
    evidence = _transition()
    hooks = {frame.frame_id: _deterministic_projector for frame in OBSERVER_FRAME_SPECS}

    first = project_observed_transition(
        evidence,
        projectors=hooks,
        event_nonce="episode_1_step_1",
    )
    second = project_observed_transition(
        evidence,
        projectors=hooks,
        event_nonce="episode_1_step_1",
    )

    assert first.event_id == second.event_id
    assert first.canonical_checksum == second.canonical_checksum
    assert first.frame_ids == tuple(
        sorted(frame.frame_id for frame in OBSERVER_FRAME_SPECS)
    )
    assert first.common_outcome.state_after is None
    assert first.common_outcome.known_channels == {
        "progress",
        "terminal",
        "goal",
    }
    assert not first.common_outcome.object_deltas
    assert not first.common_outcome.relation_deltas
    assert not first.common_outcome.topology_deltas
    assert all(item.complete for item in first.projections)
    assert all(
        item.covered_channels == ("facts", "topology") for item in first.projections
    )
    for frame_id in first.frame_ids:
        observed = first.observed_transition(frame_id)
        projected = first.projection(frame_id)
        assert observed.state_before == projected.before.state
        assert observed.state_after == projected.after.state
        assert observed.observation.state_after == projected.after.state
        assert observed.observation.progress_mean == first.common_outcome.progress_mean
        assert (
            observed.observation.terminal_probability
            == first.common_outcome.terminal_probability
        )
        assert (
            observed.observation.goal_probability
            == first.common_outcome.goal_probability
        )
        assert (
            first.common_outcome.known_channels <= observed.observation.known_channels
        )
        assert "objects" in observed.observation.known_channels
        assert projected.observation.known_channels <= {
            "objects",
            "relations",
            "topology",
        }
        assert not (
            projected.observation.known_channels & {"progress", "terminal", "goal"}
        )


def test_canonical_projection_omits_grounding_coordinates_and_local_ids() -> None:
    first_state = _state(entity_id="branch_a", center=(12.0, 31.0))
    renamed_state = _state(entity_id="branch_b", center=(99.0, -4.0))
    first_action = ActionCandidate("ACTION2", {"x": 12, "y": 31})
    renamed_action = ActionCandidate("ACTION2", {"x": 99, "y": -4})

    first = FrameProjection(
        frame=ROOT_ONLY_FRAME,
        state=first_state,
        action=first_action,
    )
    renamed = FrameProjection(
        frame=ROOT_ONLY_FRAME,
        state=renamed_state,
        action=renamed_action,
    )

    assert first.canonical_payload == renamed.canonical_payload
    assert first.canonical_checksum == renamed.canonical_checksum
    assert first.source_checksum != renamed.source_checksum
    rendered = str(first.canonical_payload)
    assert "branch_a" not in rendered
    assert "action_data" not in rendered
    assert "center" not in rendered
    assert "12.0" not in rendered
    assert audit_identity_leaks(first.canonical_payload) == ()


@pytest.mark.parametrize(
    "attributes",
    (
        (("cx", "12"),),
        (("cy", "31"),),
        (("rgb", "255_0_0"),),
        (("texture", "smooth"),),
        (("area", "gigantic"),),
    ),
)
def test_closed_structural_attribute_schema_rejects_unknown_keys_and_values(
    attributes: tuple[tuple[str, str], ...],
) -> None:
    state = _state(attributes=attributes)

    with pytest.raises(ValueError, match="closed structural schema"):
        state_model_payload(state)
    with pytest.raises(ValueError, match="closed structural schema|identity leak"):
        FrameProjection(
            frame=ROOT_ONLY_FRAME,
            state=state,
            action=ActionCandidate("ACTION1"),
        )


def test_canonical_json_rejects_python_objects_and_non_string_keys() -> None:
    with pytest.raises(TypeError, match="canonical JSON"):
        canonical_json({"opaque": object()})
    with pytest.raises(TypeError, match="canonical JSON"):
        canonical_json({1: "not_a_json_object_key"})


def test_state_model_payload_is_invariant_to_entity_tuple_order() -> None:
    actor = AbstractEntity("actor_local", ("actor", "object"), center=(1.0, 2.0))
    target = AbstractEntity("target_local", ("object", "target"), center=(3.0, 4.0))
    facts = frozenset(
        {
            GroundFact("exists", ("actor_local",)),
            GroundFact("exists", ("target_local",)),
            GroundFact("near", ("actor_local", "target_local")),
        }
    )
    first = AbstractState(entities=(actor, target), true_facts=facts)
    second = AbstractState(entities=(target, actor), true_facts=facts)

    assert state_model_payload(first) == state_model_payload(second)


def test_symmetric_cycle_canonicalization_is_alpha_invariant() -> None:
    logical_ids = ("local_a", "local_b", "local_c", "local_d")
    renamed = {
        "local_a": "branch_z",
        "local_b": "branch_w",
        "local_c": "branch_y",
        "local_d": "branch_x",
    }

    def cycle_state(
        mapping: dict[str, str], *, reverse_entities: bool
    ) -> AbstractState:
        entities = tuple(
            AbstractEntity(mapping[entity_id], ("object",)) for entity_id in logical_ids
        )
        if reverse_entities:
            entities = tuple(reversed(entities))
        facts = frozenset(
            GroundFact(
                "adjacent",
                (
                    mapping[entity_id],
                    mapping[logical_ids[(index + 1) % len(logical_ids)]],
                ),
            )
            for index, entity_id in enumerate(logical_ids)
        )
        return AbstractState(entities=entities, true_facts=facts)

    identity = {entity_id: entity_id for entity_id in logical_ids}
    first = cycle_state(identity, reverse_entities=False)
    relabelled = cycle_state(renamed, reverse_entities=True)

    assert state_model_payload(first) == state_model_payload(relabelled)


def test_oversized_symmetric_canonicalization_fails_closed() -> None:
    entity_ids = tuple(f"local_{index}" for index in range(9))
    state = AbstractState(
        entities=tuple(
            AbstractEntity(entity_id, ("object",)) for entity_id in entity_ids
        ),
        true_facts=frozenset(
            GroundFact(
                "adjacent",
                (entity_id, entity_ids[(index + 1) % len(entity_ids)]),
            )
            for index, entity_id in enumerate(entity_ids)
        ),
    )
    with pytest.raises(ValueError, match="bounded canonical labeling"):
        state_model_payload(state)


@pytest.mark.parametrize(
    ("payload", "kind"),
    (
        ({"attribute": {"color": 7}}, "forbidden_field"),
        ({"color_7": 1.0}, "raw_value"),
        ({"attribute": {"value": 7}}, "forbidden_field"),
        ({"attribute": "color_7"}, "raw_value"),
        ({"attribute": "red"}, "raw_value"),
        ({"location": "at (12, 31)"}, "absolute_coordinate"),
        ({"identity": "persistent_object_17"}, "persistent_identity"),
        (
            {"identity": "550e8400-e29b-41d4-a716-446655440000"},
            "persistent_identity",
        ),
        ({"source": "bp35-0a0ad940"}, "game_id"),
        ({"source": "mt35-deadbeef"}, "game_id"),
    ),
)
def test_generic_identity_audit_rejects_forbidden_payloads(
    payload: object,
    kind: str,
) -> None:
    assert any(item.startswith(f"{kind}:") for item in audit_identity_leaks(payload))


@pytest.mark.parametrize(
    "state",
    (
        _state(attributes=(("color", "7"),)),
        _state(entity_id="(12,31)"),
        _state(roles=("object", "persistent_object")),
        _state(entity_id="bp35-0a0ad940"),
    ),
)
def test_frame_projection_fails_closed_on_identity_leaks(state: AbstractState) -> None:
    with pytest.raises(ValueError, match="identity leak"):
        FrameProjection(
            frame=ROOT_ONLY_FRAME,
            state=state,
            action=ActionCandidate("ACTION1"),
        )


def test_frame_and_projection_metadata_are_identity_audited() -> None:
    with pytest.raises(ValueError, match="observer frame spec"):
        ObserverFrameSpec("bp35", "root_only", "test_v1", True)
    with pytest.raises(ValueError, match="projected-state metadata"):
        ProjectedState(_state(), provenance=("source_bp35",))


def test_missing_or_incomplete_projector_has_explicit_partial_fallback() -> None:
    state = _state()
    action = ActionCandidate("ACTION1")

    missing_hook = project_state(
        state,
        action,
        ROOT_ONLY_FRAME,
        stage="before",
    )
    incomplete_hook = project_state(
        state,
        action,
        ROOT_ONLY_FRAME,
        stage="after",
        projector=lambda frame, source, selected, stage: None,
    )

    assert missing_hook.state is state
    assert missing_hook.complete is False
    assert missing_hook.covered_channels == ()
    assert missing_hook.missing == ("frame_specific_projection",)
    assert missing_hook.audit_tags == ("abstract_state_fallback",)
    assert incomplete_hook.state is state
    assert incomplete_hook.complete is False
    assert incomplete_hook.covered_channels == ()
    assert incomplete_hook.audit_tags == ("projector_incomplete_fallback",)
    assert ProjectedState(state, complete=False).missing == (
        "unspecified_projection_data",
    )
    with pytest.raises(ValueError, match="complete projected state"):
        ProjectedState(state, complete=True, missing=("topology",))


def test_projection_channel_vocabulary_is_versioned_and_closed() -> None:
    state = _state()

    assert PROJECTION_CHANNEL_VOCABULARY_VERSION.endswith("-v1")
    assert PROJECTION_CHANNELS == STATE_PROJECTION_CHANNELS
    assert "counters" in STATE_PROJECTION_CHANNELS
    assert "objects" in PREDICTIVE_PROJECTION_CHANNELS
    assert set(PROJECTION_CHANNEL_VOCABULARY) == (
        set(STATE_PROJECTION_CHANNELS) | set(PREDICTIVE_PROJECTION_CHANNELS)
    )
    assert ProjectedState(state, covered_channels=("counters",)).covered_channels == (
        "counters",
    )
    with pytest.raises(ValueError, match="unknown covered projection channel"):
        ProjectedState(state, covered_channels=("invented_channel",))


def test_adapter_supports_mixed_hooks_without_live_grid() -> None:
    bundle = project_observed_transition(
        _transition(),
        frames=(ROOT_ONLY_FRAME, ACTION_ALIGNED_RELATIONAL_FRAME),
        projectors={ROOT_ONLY_FRAME.frame_id: identity_projector},
    )

    assert bundle.projection(ROOT_ONLY_FRAME.frame_id).complete is True
    assert bundle.projection(ROOT_ONLY_FRAME.frame_id).covered_channels == tuple(
        sorted(PROJECTION_CHANNELS)
    )
    relational = bundle.projection(ACTION_ALIGNED_RELATIONAL_FRAME.frame_id)
    assert relational.complete is False
    assert relational.missing == ("frame_specific_projection",)


def test_event_ids_are_stable_nonce_sensitive_and_duplicate_checked() -> None:
    evidence = _transition()
    first_id = observed_transition_event_id(evidence, nonce="step_1")
    repeated_id = observed_transition_event_id(evidence, nonce="step_1")
    next_id = observed_transition_event_id(evidence, nonce="step_2")
    assert first_id == repeated_id
    assert first_id != next_id

    first = project_observed_transition(
        evidence,
        frames=(ROOT_ONLY_FRAME,),
        event_id=first_id,
    )
    duplicate = project_observed_transition(
        evidence,
        frames=(ROOT_ONLY_FRAME,),
        event_id=first_id,
    )
    other = project_observed_transition(
        evidence,
        frames=(ROOT_ONLY_FRAME,),
        event_id=next_id,
    )
    assert validate_unique_event_ids((first, other)) == (first, other)
    with pytest.raises(ValueError, match="duplicate physical event ids"):
        validate_unique_event_ids((first, duplicate))


def test_duplicate_frames_and_mismatched_transitions_fail_closed() -> None:
    evidence = _transition()
    with pytest.raises(ValueError, match="frame list contains duplicates"):
        project_observed_transition(
            evidence,
            frames=(ROOT_ONLY_FRAME, ROOT_ONLY_FRAME),
        )

    before = project_state(
        evidence.state_before,
        evidence.action,
        ROOT_ONLY_FRAME,
        stage="before",
        projector=identity_projector,
    )
    after_other_frame = project_state(
        evidence.state_after,
        evidence.action,
        ACTION_ALIGNED_RELATIONAL_FRAME,
        stage="after",
        projector=identity_projector,
    )
    with pytest.raises(ValueError, match="frames do not match"):
        ProjectedTransition("event", before, after_other_frame)

    valid_after = project_state(
        evidence.state_after,
        evidence.action,
        ROOT_ONLY_FRAME,
        stage="after",
        projector=identity_projector,
    )
    transition = ProjectedTransition("event", before, valid_after)
    with pytest.raises(ValueError, match="duplicate frame ids"):
        PhysicalEventBundle(
            event_id="event",
            action=evidence.action,
            common_outcome=evidence.observation,
            projections=(transition, transition),
        )


def test_observer_frame_bank_is_bounded_to_four() -> None:
    evidence = _transition()
    fifth = ObserverFrameSpec(
        "fifth_frame",
        "root_only",
        "test_v1",
        False,
    )
    frames = (*OBSERVER_FRAME_SPECS, fifth)
    with pytest.raises(ValueError, match="exceeds 4 frames"):
        project_observed_transition(evidence, frames=frames)

    projections = []
    for frame in frames:
        before = project_state(
            evidence.state_before,
            evidence.action,
            frame,
            stage="before",
            projector=identity_projector,
        )
        after = project_state(
            evidence.state_after,
            evidence.action,
            frame,
            stage="after",
            projector=identity_projector,
        )
        projections.append(ProjectedTransition("event", before, after))
    with pytest.raises(ValueError, match="exceeds 4 observer frames"):
        PhysicalEventBundle(
            event_id="event",
            action=evidence.action,
            common_outcome=evidence.observation,
            projections=tuple(projections),
        )


def test_bundle_refuses_no_available_frame_projection() -> None:
    evidence = _transition()
    with pytest.raises(ValueError, match="at least one observer projection"):
        PhysicalEventBundle(
            event_id="physical_event",
            action=evidence.action,
            common_outcome=evidence.observation,
            projections=(),
        )


def test_physical_event_metadata_is_identity_audited() -> None:
    with pytest.raises(ValueError, match="identity leak in physical events"):
        project_observed_transition(
            _transition(events=("source_bp35-0a0ad940",)),
            frames=(ROOT_ONLY_FRAME,),
        )
