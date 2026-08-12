"""Compile real SAGE observations into identity-free SAGE.T evidence."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from theory.sage12.mt.transition import compile_mt_transition
from theory.sage12.scene_graph import build_scene_graph
from v3.schemas import GameObservation, TransitionRecord

from .contracts import (
    RELATION_PREDICATES,
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
)

_EVENT_ALIASES = {
    "birth": "created",
    "death": "removed",
    "relative_motion": "moved",
    "persist": "persisted",
    "growth": "morphology_changed",
    "contraction": "morphology_changed",
    "noop": "no_effect",
}


def _predicate_fact(key: str) -> GroundFact | None:
    try:
        return GroundFact.from_key(key)
    except ValueError:
        return None


def compile_observation(
    observation: GameObservation,
    *,
    topology: Mapping[str, int] | None = None,
    registers: Mapping[str, str] | None = None,
    regime_index: int = 0,
) -> AbstractState:
    """Build public facts without exposing raw colors or absolute coordinates."""

    graph = build_scene_graph(observation)
    entities = tuple(
        AbstractEntity(
            entity_id=entity.entity_id,
            roles=entity.roles,
            attributes=(
                ("area", entity.area_bucket),
                ("aspect", entity.aspect_bucket),
            ),
            center=entity.center,
        )
        for entity in graph.entities
    )
    facts = {
        fact
        for fact in (_predicate_fact(key) for key in graph.state_predicates)
        if fact is not None
    }
    for entity in entities:
        facts.add(GroundFact("exists", (entity.entity_id,)))
        for role in entity.roles:
            facts.add(GroundFact("role", (entity.entity_id, role)))
    false_facts: set[GroundFact] = set()
    if str(observation.game_state).upper() != "GAME_OVER":
        false_facts.add(GroundFact("game_over"))
    if str(observation.game_state).upper() not in {"WIN", "WON", "VICTORY"}:
        false_facts.add(GroundFact("level_complete"))
    false_facts.difference_update(facts)
    counters = {
        "levels_completed": float(observation.levels_completed),
    }
    return AbstractState(
        entities=entities,
        true_facts=frozenset(facts),
        false_facts=frozenset(false_facts),
        counters=tuple(counters.items()),
        registers=tuple((registers or {}).items()),
        topology=tuple((topology or {}).items()),
        regime_index=regime_index,
    )


def compile_causal_observation(
    observation: GameObservation,
    *,
    topology: Mapping[str, int] | None = None,
    registers: Mapping[str, str] | None = None,
    regime_index: int = 0,
) -> AbstractState:
    """Build the bounded object-centric state used by causal-program runtime.

    The general SAGE scene graph deliberately contains all pairwise spatial
    relations.  On dense 64x64 worlds this can exceed eighty thousand facts,
    even when a causal particle declares only two variables.  The causal
    runtime instead keeps entities, roles, and player-local relations.  This is
    linear in the number of objects and preserves the intervention-relevant
    state without silently turning every particle into a pixel world model.
    """

    player = observation.best_player
    player_position = tuple(player.position) if player is not None else None
    affordances: dict[int, set[str]] = {}
    for affordance in observation.affordances:
        if isinstance(affordance.target, int):
            affordances.setdefault(int(affordance.target), set()).add(
                str(affordance.kind.value)
            )
    entities = []
    object_pairs = []
    for index, obj in enumerate(
        sorted(observation.objects, key=lambda item: (item.center, item.area))
    ):
        roles = {"object"}
        if player_position is not None and player_position in obj.cells:
            roles.add("player")
        roles.update(affordances.get(int(obj.object_id), ()))
        if not roles.intersection({"player", "hazardous", "collectible"}):
            roles.add("target")
        r0, c0, r1, c1 = obj.bbox
        height = max(1, int(r1) - int(r0) + 1)
        width = max(1, int(c1) - int(c0) + 1)
        aspect = "square"
        if width >= 2 * height:
            aspect = "wide"
        elif height >= 2 * width:
            aspect = "tall"
        area = int(obj.area)
        area_bucket = (
            "one" if area <= 1 else "small" if area <= 4 else "medium" if area <= 16 else "large"
        )
        entity = AbstractEntity(
            entity_id=f"e{index}:{area_bucket}:{aspect}",
            roles=tuple(sorted(roles)),
            attributes=(("area", area_bucket), ("aspect", aspect)),
            center=tuple(float(value) for value in obj.center),
        )
        entities.append(entity)
        object_pairs.append((entity, obj))
    if player_position is not None and not any(
        entity.has_role("player") for entity in entities
    ):
        entities.insert(
            0,
            AbstractEntity(
                "e_player",
                ("object", "player"),
                (("area", "one"), ("aspect", "square")),
                (float(player_position[0]), float(player_position[1])),
            ),
        )

    facts: set[GroundFact] = set()
    for entity in entities:
        facts.add(GroundFact("exists", (entity.entity_id,)))
        for role in entity.roles:
            facts.add(GroundFact("role", (entity.entity_id, role)))
    player_entity = next(
        (entity for entity in entities if entity.has_role("player")),
        None,
    )
    if player_entity is not None and player_entity.center is not None:
        height, width = observation.raw_grid.shape[:2]
        local_scale = max(1.0, math.hypot(height, width))
        for target in entities:
            if target is player_entity or target.center is None:
                continue
            dr = target.center[0] - player_entity.center[0]
            dc = target.center[1] - player_entity.center[1]
            distance = math.hypot(dr, dc)
            if distance / local_scale <= 0.25:
                facts.add(GroundFact("near", (player_entity.entity_id, target.entity_id)))
            if abs(dr) <= 0.5 or abs(dc) <= 0.5:
                facts.add(GroundFact("aligned", (player_entity.entity_id, target.entity_id)))
            if distance <= 1.5:
                facts.add(GroundFact("contact", (player_entity.entity_id, target.entity_id)))
                facts.add(GroundFact("adjacent", (player_entity.entity_id, target.entity_id)))
            if dr < -0.5:
                facts.add(GroundFact("north_of", (target.entity_id, player_entity.entity_id)))
            elif dr > 0.5:
                facts.add(GroundFact("south_of", (target.entity_id, player_entity.entity_id)))
            if dc < -0.5:
                facts.add(GroundFact("west_of", (target.entity_id, player_entity.entity_id)))
            elif dc > 0.5:
                facts.add(GroundFact("east_of", (target.entity_id, player_entity.entity_id)))

    false_facts: set[GroundFact] = set()
    if str(observation.game_state).upper() != "GAME_OVER":
        false_facts.add(GroundFact("game_over"))
    if str(observation.game_state).upper() not in {"WIN", "WON", "VICTORY"}:
        false_facts.add(GroundFact("level_complete"))
    counters = (("levels_completed", float(observation.levels_completed)),)
    return AbstractState(
        entities=tuple(entities),
        true_facts=frozenset(facts),
        false_facts=frozenset(false_facts - facts),
        counters=counters,
        registers=tuple((registers or {}).items()),
        topology=tuple((topology or {}).items()),
        regime_index=regime_index,
    )


def _normalize_mt_event(raw: str) -> str:
    base = str(raw).split("#", 1)[0]
    if base.startswith("relation_added:"):
        return base
    if base.startswith("relation_removed:"):
        return base
    if base.startswith("invariant:"):
        parts = base.split(":")
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
    return _EVENT_ALIASES.get(base, base)


def _relation_events(
    before: AbstractState,
    after: AbstractState,
) -> tuple[str, ...]:
    before_relations = {
        fact for fact in before.true_facts if fact.predicate in RELATION_PREDICATES
    }
    after_relations = {
        fact for fact in after.true_facts if fact.predicate in RELATION_PREDICATES
    }
    output = [
        f"relation_added:{fact.predicate}"
        for fact in after_relations - before_relations
    ]
    output.extend(
        f"relation_removed:{fact.predicate}"
        for fact in before_relations - after_relations
    )
    return tuple(sorted(output))


def _event_channels(
    events: Iterable[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    objects: dict[str, float] = {}
    relations: dict[str, float] = {}
    topology: dict[str, float] = {}
    for raw in events:
        event = _normalize_mt_event(raw)
        if event.startswith("relation_"):
            relations[event] = 1.0
        elif (
            event.startswith(("component_", "hole_", "cycle_", "boundary_"))
            or "topolog" in event
        ):
            topology[event] = 1.0
        elif event not in {"persisted"}:
            objects[event] = 1.0
    return objects, relations, topology


def compile_transition_record(
    record: TransitionRecord,
    *,
    source_game_id: str = "",
    regime_index: int = 0,
    compact_causal_state: bool = False,
) -> ObservedTransition:
    """Compile one observed before/action/after triple into common evidence."""

    if str(record.action.name).strip().upper() == "RESET":
        before = compile_observation(
            record.obs_before,
            regime_index=regime_index,
        )
        after = compile_observation(
            record.obs_after,
            regime_index=regime_index,
        )
        return ObservedTransition(
            state_before=before,
            action=ActionCandidate("RESET"),
            state_after=after,
            observation=PredictionPacket(state_after=after),
            reset=True,
        )

    before_player = (
        tuple(record.obs_before.best_player.position)
        if record.obs_before.best_player is not None
        else None
    )
    after_player = (
        tuple(record.obs_after.best_player.position)
        if record.obs_after.best_player is not None
        else None
    )
    mt = compile_mt_transition(
        record.obs_before.raw_grid,
        record.action.name,
        record.obs_after.raw_grid,
        action_data={
            key: value
            for key, value in {
                "x": record.action.x,
                "y": record.action.y,
            }.items()
            if value is not None
        },
        source_game_id=source_game_id,
        player_position_before=before_player,
        player_position_after=after_player,
        productive=not record.diff.is_noop,
        risk=bool(record.diff.game_over),
    )
    state_compiler = (
        compile_causal_observation if compact_causal_state else compile_observation
    )
    before = state_compiler(
        record.obs_before,
        topology=mt.graph_before.invariants,
        regime_index=regime_index,
    )
    after = state_compiler(
        record.obs_after,
        topology=mt.graph_after.invariants,
        regime_index=regime_index,
    )
    events = [_normalize_mt_event(item) for item in mt.events]
    events.extend(_relation_events(before, after))
    if record.diff.is_noop:
        events.append("no_effect")
    if record.diff.level_complete:
        events.extend(("progress", "level_complete"))
    if record.diff.game_over:
        events.append("game_over")
    events = sorted(set(events))
    object_deltas, relation_deltas, topology_deltas = _event_channels(events)
    progress_delta = max(
        0.0,
        float(record.obs_after.levels_completed)
        - float(record.obs_before.levels_completed),
    )
    level_signal = bool(record.diff.level_complete or progress_delta > 0.0)
    known_channels = {
        "objects",
        "relations",
        "topology",
        "progress",
        "terminal",
    }
    goal_probability = None
    if level_signal or str(record.obs_after.game_state).upper() == "WIN":
        known_channels.add("goal")
        goal_probability = 1.0
    observation = PredictionPacket(
        object_deltas=object_deltas,
        relation_deltas=relation_deltas,
        topology_deltas=topology_deltas,
        progress_mean=progress_delta,
        progress_distribution={
            f"value:{progress_delta:.6g}": 1.0,
        },
        terminal_probability=float(bool(record.diff.game_over)),
        goal_probability=goal_probability,
        known_channels=frozenset(known_channels),
        state_after=after,
    )
    action_data = {
        key: value
        for key, value in {
            "x": record.action.x,
            "y": record.action.y,
        }.items()
        if value is not None
    }
    return ObservedTransition(
        state_before=before,
        action=ActionCandidate(record.action.name, action_data),
        state_after=after,
        observation=observation,
        events=tuple(events),
        reset=str(record.action.name).upper() == "RESET",
    )


def observed_event_counts(
    transitions: Iterable[ObservedTransition],
) -> Mapping[str, int]:
    return Counter(event for transition in transitions for event in transition.events)


__all__ = [
    "compile_causal_observation", "compile_observation",
    "compile_transition_record",
    "observed_event_counts",
]
