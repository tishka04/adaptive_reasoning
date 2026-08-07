"""Compile real SAGE observations into identity-free SAGE.T evidence."""

from __future__ import annotations

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
    before = compile_observation(
        record.obs_before,
        topology=mt.graph_before.invariants,
        regime_index=regime_index,
    )
    after = compile_observation(
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
    "compile_observation",
    "compile_transition_record",
    "observed_event_counts",
]
