"""Concrete, structural-only observer-frame adapters for SAGE.T10.2.

The adapters in this module bridge the frozen :mod:`theory.sage_t.contracts`
state representation to the four observer frames registered for T10.2.  They
do not consume grids.  Absolute centres and grounded action arguments may be
used transiently to establish a local root or relative relation, but they are
never copied to the projected :class:`~theory.sage_t.contracts.AbstractState`.

The representation lineage is intentionally explicit:

* ``root_only`` retains a compact structural root summary;
* ``allocentric_object_relative`` mirrors the V4.9 object-relative view;
* ``action_aligned_relational`` replaces compass directions with the V4.10
  intervention axis;
* ``action_rooted_topological`` constructs an in-memory V4.16 MT graph and
  persists only V4.19 permutation-invariant topology plus structural classes.

Every public projector implements the ``FrameProjector`` hook consumed by
``project_observed_transition``.  Missing grounding or geometry is represented
by ``ProjectedState.complete = False`` and explicit ``missing`` fields; no
other frame is used to fabricate the unavailable information.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from theory.sage12.mt.graph import (
    MorphoTopologicalGraph,
    MTNode,
    MTRelation,
)
from theory.sage12.topological_invariants_v4_19 import topological_invariants

from .contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
)
from .observer_frames_v10_2 import (
    ACTION_ALIGNED_RELATIONAL_FRAME,
    ACTION_ROOTED_TOPOLOGICAL_FRAME,
    ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
    OBSERVER_FRAME_SPECS,
    ROOT_ONLY_FRAME,
    FrameProjector,
    ObserverFrameSpec,
    PhysicalEventBundle,
    ProjectedState,
    canonical_json,
    project_observed_transition,
    state_model_payload,
)

FORMAT_VERSION = "sage_t10_2_frame_adapters_v1"
MAXIMUM_NEIGHBOR_CLASSES = 16
TOPOLOGY_DISTANCE_SENTINEL = 16

_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MOVE_VECTORS = {
    "ACTION1": (-1.0, 0.0),
    "ACTION2": (1.0, 0.0),
    "ACTION3": (0.0, -1.0),
    "ACTION4": (0.0, 1.0),
}
_DIRECTION_VECTORS = {
    "north": (-1.0, 0.0),
    "north_east": (-1.0, 1.0),
    "east": (0.0, 1.0),
    "south_east": (1.0, 1.0),
    "south": (1.0, 0.0),
    "south_west": (1.0, -1.0),
    "west": (0.0, -1.0),
    "north_west": (-1.0, -1.0),
}
_REQUEST_VECTORS = {
    "up": (-1.0, 0.0),
    "down": (1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
    **_DIRECTION_VECTORS,
}
_REQUEST_NAMES = {
    "ACTION1": "up",
    "ACTION2": "down",
    "ACTION3": "left",
    "ACTION4": "right",
}
_OPPOSITE_DIRECTION = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}
_SAFE_ROLES = frozenset(
    {
        "action_root",
        "actor",
        "container",
        "free_region",
        "goal",
        "movable",
        "neighbor",
        "object",
        "obstacle",
        "player",
        "selected",
        "sink",
        "source",
        "space",
        "structural_class",
        "target",
    }
)
_SAFE_RELATIONS = frozenset(
    {
        "adjacent",
        "aligned",
        "attached",
        "contact",
        "detached",
        "encloses",
        "inside",
        "near",
        "reachable",
        "same_attribute",
        "same_shape",
    }
)
_SYMMETRIC_RELATIONS = frozenset(
    {
        "adjacent",
        "aligned",
        "contact",
        "near",
        "same_attribute",
        "same_shape",
    }
)
_UNARY_STATE_PREDICATES = frozenset(
    {
        "changed",
        "created",
        "moved",
        "reachable",
        "removed",
        "selected",
        "solved",
    }
)
_GLOBAL_STATE_PREDICATES = frozenset(
    {
        "game_over",
        "level_complete",
        "no_effect",
    }
)
_MT_RELATIONS = frozenset({"aligned", "contact", "encloses", "near"})
_TARGET_ARGUMENTS = frozenset(
    {
        "entity",
        "entity_id",
        "object_id",
        "selected",
        "target",
        "target_entity",
        "target_id",
        "target_object_id",
    }
)


@dataclass(frozen=True)
class _RootResolution:
    entity_id: str | None
    source: str
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RelativeCompilation:
    root: Mapping[str, Any]
    root_roles: tuple[str, ...]
    root_facts: tuple[str, ...]
    neighbors: tuple[Mapping[str, Any], ...]
    missing: tuple[str, ...]
    audit_tags: tuple[str, ...]


def _require_frame(frame: ObserverFrameSpec, expected: ObserverFrameSpec) -> None:
    if not isinstance(frame, ObserverFrameSpec) or frame != expected:
        raise ValueError(f"projector requires observer frame {expected.frame_id}")


def _require_inputs(
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> str:
    if not isinstance(state, AbstractState):
        raise TypeError("frame projector requires AbstractState")
    if not isinstance(action, ActionCandidate):
        raise TypeError("frame projector requires ActionCandidate")
    normalized = str(stage).strip().lower()
    if normalized not in {"before", "after"}:
        raise ValueError("projection stage must be before or after")
    entity_ids = [entity.entity_id for entity in state.entities]
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("frame adapter requires unique local entity ids")
    return normalized


def _safe_roles(entity: AbstractEntity) -> tuple[str, ...]:
    roles = set(entity.roles) & set(_SAFE_ROLES)
    if "player" in roles:
        roles.add("actor")
    if not roles:
        roles.add("object")
    return tuple(sorted(roles))


def _token(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized if _TOKEN.fullmatch(normalized) else "unknown"


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value == 2:
        return "two"
    if value <= 4:
        return "few"
    return "many"


def _area_bucket(value: Any) -> str:
    token = _token(value)
    aliases = {
        "one": "one",
        "single": "one",
        "small": "small",
        "medium": "medium",
        "large": "large",
        "very_large": "very_large",
    }
    if token in aliases:
        return aliases[token]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(number) or number < 0.0:
        return "unknown"
    if number <= 1.0:
        return "one"
    if number <= 4.0:
        return "small"
    if number <= 16.0:
        return "medium"
    if number <= 64.0:
        return "large"
    return "very_large"


def _bounded_count_bucket(value: Any) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    return _count_bucket(min(max(number, 0), 16))


def _allowed_token(value: Any, allowed: frozenset[str]) -> str:
    token = _token(value)
    return token if token in allowed else "unknown"


def _structural_attributes(entity: AbstractEntity) -> tuple[tuple[str, str], ...]:
    raw = dict(entity.attributes)
    output: dict[str, str] = {}
    for key in ("area_bucket", "area", "size"):
        if key in raw:
            output["area"] = _area_bucket(raw[key])
            break
    for key in ("aspect_bucket", "aspect"):
        if key in raw:
            output["aspect"] = _allowed_token(
                raw[key],
                frozenset({"compact", "square", "tall", "unknown", "wide"}),
            )
            break
    for key in ("compactness_bucket", "compactness"):
        if key in raw:
            output["compactness"] = _allowed_token(
                raw[key],
                frozenset({"compact", "irregular", "round", "sparse", "unknown"}),
            )
            break
    if "holes" in raw:
        output["hole_bucket"] = _bounded_count_bucket(raw["holes"])
    if "boundary_contacts" in raw:
        output["boundary_contact_bucket"] = _bounded_count_bucket(
            raw["boundary_contacts"]
        )
    if "affordance" in raw:
        output["affordance"] = _allowed_token(
            raw["affordance"],
            frozenset(
                {
                    "blocked",
                    "fixed",
                    "interactive",
                    "movable",
                    "passable",
                    "unknown",
                }
            ),
        )
    if "path_status" in raw:
        output["path_status"] = _allowed_token(
            raw["path_status"],
            frozenset({"blocked", "open", "reachable", "unknown", "unreachable"}),
        )
    if "occupied" in raw:
        output["occupied"] = (
            "yes"
            if str(raw["occupied"]).strip().lower() in {"1", "true", "yes"}
            else "no"
        )
    return tuple(sorted(output.items()))


def _entity_kind(entity: AbstractEntity) -> str:
    roles = set(_safe_roles(entity))
    return "free_region" if roles & {"free_region", "space"} else "object"


def _entity_descriptor(entity: AbstractEntity) -> dict[str, Any]:
    return {
        "kind": _entity_kind(entity),
        "roles": list(_safe_roles(entity)),
        "attributes": [list(item) for item in _structural_attributes(entity)],
    }


def _entity_choice_signature(state: AbstractState, entity_id: str) -> str:
    by_id = {entity.entity_id: entity for entity in state.entities}
    entity = by_id[entity_id]
    context = []
    for fact in sorted(state.true_facts, key=lambda item: item.key):
        if entity_id not in fact.terms or fact.value:
            continue
        terms = []
        for term in fact.terms:
            if term == entity_id:
                terms.append("self")
            elif term in by_id:
                terms.append(_entity_descriptor(by_id[term]))
            else:
                terms.append("literal_omitted")
        context.append((fact.predicate, terms))
    return canonical_json(
        {
            "entity": _entity_descriptor(entity),
            "context": sorted(context, key=canonical_json),
        }
    )


def _unique_structural_choice(
    state: AbstractState,
    candidates: Sequence[str],
) -> str | None:
    if not candidates:
        return None
    signatures: dict[str, list[str]] = {}
    for entity_id in candidates:
        signature = _entity_choice_signature(state, entity_id)
        signatures.setdefault(signature, []).append(entity_id)
    minimum = min(signatures)
    matches = signatures[minimum]
    return matches[0] if len(matches) == 1 else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _action_anchor(action: ActionCandidate) -> tuple[float, float] | None:
    row = action.action_data.get("row", action.action_data.get("y"))
    col = action.action_data.get("col", action.action_data.get("x"))
    parsed_row = _finite_number(row)
    parsed_col = _finite_number(col)
    if parsed_row is None or parsed_col is None:
        return None
    return parsed_row, parsed_col


def _resolve_root(state: AbstractState, action: ActionCandidate) -> _RootResolution:
    by_id = {entity.entity_id: entity for entity in state.entities}
    action_bindings = {
        str(value)
        for key in _TARGET_ARGUMENTS
        if (value := action.action_data.get(key)) is not None and str(value) in by_id
    }
    if len(action_bindings) > 1:
        return _RootResolution(
            None,
            "ambiguous_action_binding",
            ("action_binding", "root_entity"),
        )
    if action_bindings:
        return _RootResolution(next(iter(action_bindings)), "action_binding")

    register_bindings = {
        value
        for key, value in state.registers
        if key in {"action_root", "selected", "target"} and value in by_id
    }
    if len(register_bindings) > 1:
        return _RootResolution(
            None,
            "ambiguous_state_register",
            ("action_binding", "root_entity"),
        )
    if register_bindings:
        return _RootResolution(next(iter(register_bindings)), "state_register")

    selected = {
        fact.terms[0]
        for fact in state.true_facts
        if fact.predicate == "selected"
        and len(fact.terms) == 1
        and fact.terms[0] in by_id
    }
    if selected:
        if len(selected) == 1:
            return _RootResolution(next(iter(selected)), "selected_fact")
        return _RootResolution(None, "ambiguous_selected", ("root_entity",))

    for role in ("action_root", "selected", "target"):
        candidates = [
            entity.entity_id for entity in state.entities if role in entity.roles
        ]
        if candidates:
            if len(candidates) == 1:
                missing = ("action_binding",) if role == "target" else ()
                return _RootResolution(candidates[0], f"role_{role}", missing)
            return _RootResolution(
                None,
                f"ambiguous_role_{role}",
                ("root_entity",),
            )

    anchor = _action_anchor(action)
    if anchor is not None:
        located = [entity for entity in state.entities if entity.center is not None]
        if located:
            distances = {
                entity.entity_id: math.dist(anchor, entity.center or anchor)
                for entity in located
            }
            minimum = min(distances.values())
            nearest = [
                entity_id
                for entity_id, distance in distances.items()
                if math.isclose(distance, minimum, abs_tol=1e-9)
            ]
            if len(nearest) == 1:
                missing = () if minimum <= 0.75 else ("exact_action_binding",)
                return _RootResolution(nearest[0], "action_anchor", missing)
            return _RootResolution(
                None,
                "ambiguous_action_anchor",
                ("root_entity",),
            )

    actors = [
        entity.entity_id
        for entity in state.entities
        if set(entity.roles) & {"actor", "player"}
    ]
    if actors:
        if len(actors) == 1:
            missing = () if action.action_name in _MOVE_VECTORS else ("action_binding",)
            return _RootResolution(actors[0], "actor_fallback", missing)
        return _RootResolution(
            None,
            "ambiguous_actor_fallback",
            ("root_entity",),
        )
    if len(state.entities) == 1:
        return _RootResolution(
            state.entities[0].entity_id,
            "single_entity",
            ("action_binding",),
        )
    return _RootResolution(None, "unresolved", ("root_entity",))


def _action_family(action: ActionCandidate) -> str:
    if action.action_name in _MOVE_VECTORS:
        return "move"
    if action.action_name in {"ACTION5", "ACTION6"}:
        return "interact"
    if action.action_data:
        return "parameterized"
    return "other"


def _requested_direction(action: ActionCandidate) -> str:
    if action.action_name in _REQUEST_NAMES:
        return _REQUEST_NAMES[action.action_name]
    for key in ("direction", "requested_direction"):
        token = _token(action.action_data.get(key, "none"))
        if token in _REQUEST_VECTORS:
            return token
    return "none"


def _relation_count(state: AbstractState) -> int:
    entity_ids = {entity.entity_id for entity in state.entities}
    return sum(
        1
        for fact in state.true_facts
        if fact.predicate in _SAFE_RELATIONS
        and fact.terms
        and all(term in entity_ids for term in fact.terms)
        and not fact.value
    )


def _global_facts(state: AbstractState) -> set[GroundFact]:
    return {
        GroundFact(fact.predicate)
        for fact in state.true_facts
        if fact.predicate in _GLOBAL_STATE_PREDICATES
        and not fact.terms
        and not fact.value
    }


def _root_unary_facts(state: AbstractState, root_id: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                fact.predicate
                for fact in state.true_facts
                if fact.predicate in _UNARY_STATE_PREDICATES
                and fact.terms == (root_id,)
                and not fact.value
            }
        )
    )


def _root_kind(entity: AbstractEntity | None) -> str:
    if entity is None:
        return "action_root"
    roles = set(_safe_roles(entity))
    if "actor" in roles:
        return "actor"
    if "target" in roles:
        return "target_object"
    if _entity_kind(entity) == "free_region":
        return "free_region"
    return "occupied_object"


def _root_entity(
    source: AbstractEntity | None,
    action: ActionCandidate,
) -> AbstractEntity:
    roles = set(_safe_roles(source)) if source is not None else {"object"}
    roles.add("action_root")
    attributes = dict(_structural_attributes(source)) if source is not None else {}
    attributes.update(
        {
            "action_family": _action_family(action),
            "root_kind": _root_kind(source),
        }
    )
    return AbstractEntity(
        "e0",
        tuple(sorted(roles)),
        attributes=tuple(sorted(attributes.items())),
    )


def _projection_tags(state: AbstractState) -> tuple[str, ...]:
    tags = {
        "absolute_coordinates_not_emitted",
        "grounded_identity_relabelled",
        "structural_projection_only",
    }
    if any(entity.center is not None for entity in state.entities):
        tags.add("relative_geometry_consumed")
    return tuple(sorted(tags))


def _final_projected_state(
    state: AbstractState,
    *,
    missing: Sequence[str],
    covered_channels: Sequence[str],
    provenance: Sequence[str],
    audit_tags: Sequence[str],
) -> ProjectedState:
    # Reuse the observer firewall before returning a hook result directly.
    state_model_payload(state)
    normalized_missing = tuple(sorted(set(missing)))
    return ProjectedState(
        state=state,
        complete=not normalized_missing,
        missing=normalized_missing,
        covered_channels=tuple(covered_channels),
        provenance=tuple(provenance),
        audit_tags=tuple(audit_tags),
    )


def project_root_only(
    frame: ObserverFrameSpec,
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> ProjectedState:
    """Project one state to a compact, identity-free structural root."""

    _require_frame(frame, ROOT_ONLY_FRAME)
    _require_inputs(state, action, stage)
    resolution = _resolve_root(state, action)
    by_id = {entity.entity_id: entity for entity in state.entities}
    source_root = by_id.get(resolution.entity_id or "")
    root = _root_entity(source_root, action)
    facts = _global_facts(state)
    facts.add(GroundFact("exists", (root.entity_id,)))
    if resolution.entity_id is not None:
        facts.update(
            GroundFact(predicate, (root.entity_id,))
            for predicate in _root_unary_facts(state, resolution.entity_id)
        )
    roles = Counter(role for entity in state.entities for role in _safe_roles(entity))
    counters = {
        "actor_count": float(roles["actor"]),
        "entity_count": float(len(state.entities)),
        "relation_count": float(_relation_count(state)),
        "target_count": float(roles["target"]),
    }
    topology = {
        "free_region_count": sum(
            _entity_kind(entity) == "free_region" for entity in state.entities
        ),
        "node_count": len(state.entities),
        "object_count": sum(
            _entity_kind(entity) == "object" for entity in state.entities
        ),
    }
    projected = AbstractState(
        entities=(root,),
        true_facts=frozenset(facts),
        counters=tuple(counters.items()),
        topology=tuple(topology.items()),
    )
    return _final_projected_state(
        projected,
        missing=resolution.missing,
        covered_channels=("entities", "facts", "counters", "objects", "topology"),
        provenance=("sage12_v4_9_root_summary", "t10_2_abstract_state_adapter"),
        audit_tags=_projection_tags(state),
    )


def _direction(
    origin: tuple[float, float],
    target: tuple[float, float],
) -> str:
    dr = float(target[0] - origin[0])
    dc = float(target[1] - origin[1])
    vertical = "north" if dr < -0.5 else "south" if dr > 0.5 else ""
    horizontal = "west" if dc < -0.5 else "east" if dc > 0.5 else ""
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    return vertical or horizontal or "overlap"


def _direction_from_facts(
    state: AbstractState,
    origin_id: str,
    target_id: str,
) -> str:
    directions: set[str] = set()
    for fact in state.true_facts:
        direction = fact.predicate.removesuffix("_of")
        if direction not in _OPPOSITE_DIRECTION or len(fact.terms) != 2:
            continue
        subject, obj = fact.terms
        if (subject, obj) == (target_id, origin_id):
            directions.add(direction)
        elif (subject, obj) == (origin_id, target_id):
            directions.add(_OPPOSITE_DIRECTION[direction])
    if not directions:
        return "unknown"
    vertical = next((item for item in ("north", "south") if item in directions), "")
    horizontal = next((item for item in ("east", "west") if item in directions), "")
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    return vertical or horizontal or "unknown"


def _relative_direction(
    state: AbstractState,
    origin: AbstractEntity,
    target: AbstractEntity,
) -> str:
    if origin.center is not None and target.center is not None:
        return _direction(origin.center, target.center)
    return _direction_from_facts(state, origin.entity_id, target.entity_id)


def _proximity(
    state: AbstractState,
    left: AbstractEntity,
    right: AbstractEntity,
) -> str:
    pair = {left.entity_id, right.entity_id}
    for predicate in ("contact", "adjacent", "near"):
        if any(
            fact.predicate == predicate
            and len(fact.terms) == 2
            and set(fact.terms) == pair
            for fact in state.true_facts
        ):
            return predicate
    if left.center is None or right.center is None:
        return "unknown"
    distance = math.dist(left.center, right.center)
    if distance <= 0.5:
        return "overlap"
    if distance <= 1.5:
        return "adjacent"
    if distance <= 4.0:
        return "near"
    if distance <= 10.0:
        return "mid"
    return "far"


def _area_rank(entity: AbstractEntity) -> int | None:
    area = dict(_structural_attributes(entity)).get("area", "unknown")
    return {
        "one": 1,
        "small": 2,
        "medium": 3,
        "large": 4,
        "very_large": 5,
    }.get(area)


def _relative_size(root: AbstractEntity, neighbor: AbstractEntity) -> str:
    root_rank = _area_rank(root)
    neighbor_rank = _area_rank(neighbor)
    if root_rank is None or neighbor_rank is None:
        return "unknown"
    if neighbor_rank < root_rank:
        return "smaller"
    if neighbor_rank > root_rank:
        return "larger"
    return "equal"


def _root_relations(
    state: AbstractState,
    root_id: str,
    neighbor_id: str,
) -> tuple[tuple[str, str], ...]:
    output: set[tuple[str, str]] = set()
    for fact in state.true_facts:
        if fact.value or fact.predicate not in _SAFE_RELATIONS:
            continue
        if len(fact.terms) == 1 and fact.terms == (neighbor_id,):
            output.add((fact.predicate, "neighbor_unary"))
        elif len(fact.terms) == 2:
            subject, obj = fact.terms
            if {subject, obj} != {root_id, neighbor_id}:
                continue
            if fact.predicate in _SYMMETRIC_RELATIONS:
                output.add((fact.predicate, "symmetric"))
            elif (subject, obj) == (root_id, neighbor_id):
                output.add((fact.predicate, "root_to_neighbor"))
            else:
                output.add((fact.predicate, "neighbor_to_root"))
    return tuple(sorted(output))


def _actor_entity(state: AbstractState) -> AbstractEntity | None:
    actors = [
        entity for entity in state.entities if set(entity.roles) & {"actor", "player"}
    ]
    if len(actors) == 1:
        return actors[0]
    choice = _unique_structural_choice(
        state,
        [entity.entity_id for entity in actors],
    )
    return next((entity for entity in actors if entity.entity_id == choice), None)


def _compile_relative(
    state: AbstractState,
    action: ActionCandidate,
) -> _RelativeCompilation:
    resolution = _resolve_root(state, action)
    by_id = {entity.entity_id: entity for entity in state.entities}
    root_entity = by_id.get(resolution.entity_id or "")
    missing = set(resolution.missing)
    tags = set(_projection_tags(state))
    if root_entity is None:
        missing.add("root_entity")

    actor = _actor_entity(state)
    actor_direction = (
        _relative_direction(state, root_entity, actor)
        if root_entity is not None and actor is not None
        else "unknown"
    )
    actor_relation = (
        _proximity(state, root_entity, actor)
        if root_entity is not None and actor is not None
        else "unknown"
    )
    root_attributes = dict(_structural_attributes(root_entity)) if root_entity else {}
    root: dict[str, Any] = {
        "action_family": _action_family(action),
        "requested_direction": _requested_direction(action),
        "root_kind": _root_kind(root_entity),
        "root_occupied": int(root_entity is not None),
        "root_area_bucket": root_attributes.get("area", "unknown"),
        "root_aspect_bucket": root_attributes.get("aspect", "unknown"),
        "root_affordance": root_attributes.get("affordance", "unknown"),
        "actor_relation": actor_relation,
        "actor_relative_direction": actor_direction,
        "path_status": root_attributes.get("path_status", "unknown"),
        "player_available": int(actor is not None),
    }
    root_roles = (
        tuple(sorted(set(_safe_roles(root_entity)) | {"action_root"}))
        if root_entity is not None
        else ("action_root", "object")
    )
    root_facts = (
        _root_unary_facts(state, root_entity.entity_id)
        if root_entity is not None
        else ()
    )

    neighbors: list[dict[str, Any]] = []
    for entity in state.entities:
        if root_entity is not None and entity.entity_id == root_entity.entity_id:
            continue
        direction = (
            _relative_direction(state, root_entity, entity)
            if root_entity is not None
            else "unknown"
        )
        proximity = (
            _proximity(state, root_entity, entity)
            if root_entity is not None
            else "unknown"
        )
        if direction == "unknown":
            missing.add("relative_geometry")
        if proximity == "unknown":
            missing.add("relative_proximity")
        aligned_row = int(
            root_entity is not None
            and root_entity.center is not None
            and entity.center is not None
            and abs(root_entity.center[0] - entity.center[0]) <= 0.5
        )
        aligned_col = int(
            root_entity is not None
            and root_entity.center is not None
            and entity.center is not None
            and abs(root_entity.center[1] - entity.center[1]) <= 0.5
        )
        attributes = dict(_structural_attributes(entity))
        neighbors.append(
            {
                "direction": direction,
                "proximity": proximity,
                "relative_size": (
                    _relative_size(root_entity, entity)
                    if root_entity is not None
                    else "unknown"
                ),
                "area_bucket": attributes.get("area", "unknown"),
                "aspect_bucket": attributes.get("aspect", "unknown"),
                "compactness_bucket": attributes.get("compactness", "unknown"),
                "hole_bucket": attributes.get("hole_bucket", "unknown"),
                "is_actor": int("actor" in _safe_roles(entity)),
                "aligned_row": aligned_row,
                "aligned_col": aligned_col,
                "roles": _safe_roles(entity),
                "relations": (
                    _root_relations(state, root_entity.entity_id, entity.entity_id)
                    if root_entity is not None
                    else ()
                ),
            }
        )

    groups: dict[str, tuple[dict[str, Any], int]] = {}
    for neighbor in neighbors:
        signature = canonical_json(neighbor)
        if signature in groups:
            descriptor, count = groups[signature]
            groups[signature] = (descriptor, count + 1)
        else:
            groups[signature] = (neighbor, 1)
    ordered = sorted(groups.values(), key=lambda item: canonical_json(item[0]))
    if len(ordered) > MAXIMUM_NEIGHBOR_CLASSES:
        tags.add("neighbor_class_budget_truncated")
        missing.add("neighbor_class_budget_truncated")
    compiled_neighbors = tuple(
        {**descriptor, "multiplicity": _count_bucket(count)}
        for descriptor, count in ordered[:MAXIMUM_NEIGHBOR_CLASSES]
    )
    return _RelativeCompilation(
        root=root,
        root_roles=root_roles,
        root_facts=root_facts,
        neighbors=compiled_neighbors,
        missing=tuple(sorted(missing)),
        audit_tags=tuple(sorted(tags)),
    )


def _axis_vector(
    action: ActionCandidate,
    state: AbstractState,
    root_id: str | None,
) -> tuple[tuple[float, float] | None, str]:
    if action.action_name in _MOVE_VECTORS:
        return _MOVE_VECTORS[action.action_name], "movement"
    requested = _requested_direction(action)
    if requested in _REQUEST_VECTORS:
        return _REQUEST_VECTORS[requested], "movement"
    by_id = {entity.entity_id: entity for entity in state.entities}
    root = by_id.get(root_id or "")
    actor = _actor_entity(state)
    if (
        root is not None
        and actor is not None
        and root.center is not None
        and actor.center is not None
    ):
        vector = (
            root.center[0] - actor.center[0],
            root.center[1] - actor.center[1],
        )
        if math.hypot(*vector) > 0.0:
            return vector, "actor_to_target"
    return None, "none"


def _axis_relation(
    direction: str,
    axis: tuple[float, float] | None,
) -> str:
    if direction == "overlap":
        return "overlap"
    vector = _DIRECTION_VECTORS.get(direction)
    if vector is None or axis is None:
        return "radial"
    norm = math.hypot(*axis) * math.hypot(*vector)
    if norm <= 0.0:
        return "radial"
    forward = (axis[0] * vector[0] + axis[1] * vector[1]) / norm
    right_axis = (axis[1], -axis[0])
    right = (right_axis[0] * vector[0] + right_axis[1] * vector[1]) / norm
    if abs(forward) >= abs(right):
        return "ahead" if forward >= 0.0 else "behind"
    return "lateral_right" if right >= 0.0 else "lateral_left"


def _materialize_root_relations(
    facts: set[GroundFact],
    alias: str,
    relations: Sequence[Sequence[str]],
) -> None:
    for predicate, orientation in relations:
        if predicate not in _SAFE_RELATIONS or predicate == "reachable":
            continue
        if orientation == "neighbor_to_root":
            terms = (alias, "e0")
        else:
            terms = ("e0", alias)
        facts.add(GroundFact(predicate, terms))
    if any(
        predicate == "reachable" and orientation == "neighbor_unary"
        for predicate, orientation in relations
    ):
        facts.add(GroundFact("reachable", (alias,)))


def _add_allocentric_direction_facts(
    facts: set[GroundFact],
    alias: str,
    direction: str,
) -> None:
    if "north" in direction:
        facts.add(GroundFact("north_of", (alias, "e0")))
    if "south" in direction:
        facts.add(GroundFact("south_of", (alias, "e0")))
    if "east" in direction:
        facts.add(GroundFact("east_of", (alias, "e0")))
    if "west" in direction:
        facts.add(GroundFact("west_of", (alias, "e0")))


def _relative_state(
    compilation: _RelativeCompilation,
    source: AbstractState,
    *,
    aligned: bool,
    axis: tuple[float, float] | None = None,
    axis_source: str = "none",
) -> AbstractState:
    root_attributes = {
        key: str(value).lower()
        for key, value in compilation.root.items()
        if key
        not in (
            {"requested_direction", "actor_relative_direction"} if aligned else set()
        )
    }
    if aligned:
        root_attributes["axis_source"] = axis_source
    entities = [
        AbstractEntity(
            "e0",
            compilation.root_roles,
            attributes=tuple(sorted(root_attributes.items())),
        )
    ]
    facts = _global_facts(source)
    facts.add(GroundFact("exists", ("e0",)))
    facts.update(GroundFact(predicate, ("e0",)) for predicate in compilation.root_facts)
    contact_degree = 0
    adjacent_degree = 0
    for index, raw in enumerate(compilation.neighbors, start=1):
        item = dict(raw)
        alias = f"e{index}"
        direction = str(item.pop("direction", "unknown"))
        relations = tuple(item.pop("relations", ()))
        roles = {str(role) for role in item.pop("roles", ())}
        roles.update(("neighbor", "structural_class"))
        if aligned:
            item.pop("aligned_row", None)
            item.pop("aligned_col", None)
            item["axis_relation"] = _axis_relation(direction, axis)
            proximity = str(item.get("proximity", "unknown"))
            item["topology_relation"] = (
                "root_contact"
                if proximity == "contact"
                else "root_adjacent"
                if proximity == "adjacent"
                else "near"
                if proximity in {"near", "mid"}
                else "distant"
                if proximity == "far"
                else "unknown"
            )
            item.pop("proximity", None)
        else:
            item["direction"] = direction
        contact_degree += int(any(row[0] == "contact" for row in relations))
        adjacent_degree += int(any(row[0] == "adjacent" for row in relations))
        attributes = tuple(
            sorted((str(key), str(value).lower()) for key, value in item.items())
        )
        entities.append(AbstractEntity(alias, tuple(sorted(roles)), attributes))
        facts.add(GroundFact("exists", (alias,)))
        _materialize_root_relations(facts, alias, relations)
        if not aligned:
            _add_allocentric_direction_facts(facts, alias, direction)
    topology = {
        "component_count": len(entities),
        "contact_edges": contact_degree,
        "structural_edges": _relation_count(source),
    }
    counters = (
        ("adjacent_degree", float(adjacent_degree)),
        ("contact_degree", float(contact_degree)),
        ("neighbor_classes", float(len(compilation.neighbors))),
    )
    return AbstractState(
        entities=tuple(entities),
        true_facts=frozenset(facts),
        counters=counters,
        topology=tuple(topology.items()),
    )


def project_allocentric_object_relative(
    frame: ObserverFrameSpec,
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> ProjectedState:
    """Project a V4.9-style object-relative structural neighborhood."""

    _require_frame(frame, ALLOCENTRIC_OBJECT_RELATIVE_FRAME)
    _require_inputs(state, action, stage)
    compilation = _compile_relative(state, action)
    projected = _relative_state(compilation, state, aligned=False)
    return _final_projected_state(
        projected,
        missing=compilation.missing,
        covered_channels=(
            "entities",
            "facts",
            "counters",
            "objects",
            "relations",
            "topology",
        ),
        provenance=("sage12_v4_9_object_relative", "t10_2_abstract_state_adapter"),
        audit_tags=compilation.audit_tags,
    )


def project_action_aligned_relational(
    frame: ObserverFrameSpec,
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> ProjectedState:
    """Project V4.9 relations onto the compass-free V4.10 action axis."""

    _require_frame(frame, ACTION_ALIGNED_RELATIONAL_FRAME)
    _require_inputs(state, action, stage)
    compilation = _compile_relative(state, action)
    resolution = _resolve_root(state, action)
    axis, axis_source = _axis_vector(action, state, resolution.entity_id)
    missing = set(compilation.missing)
    if axis is None and compilation.neighbors:
        missing.add("action_axis")
    projected = _relative_state(
        compilation,
        state,
        aligned=True,
        axis=axis,
        axis_source=axis_source,
    )
    return _final_projected_state(
        projected,
        missing=tuple(sorted(missing)),
        covered_channels=(
            "entities",
            "facts",
            "counters",
            "objects",
            "relations",
            "topology",
        ),
        provenance=(
            "sage12_v4_10_action_aligned",
            "t10_2_abstract_state_adapter",
        ),
        audit_tags=compilation.audit_tags,
    )


def _raw_attribute_int(entity: AbstractEntity, key: str) -> int:
    raw = dict(entity.attributes).get(key)
    try:
        return min(max(int(float(raw)), 0), 16)
    except (TypeError, ValueError):
        return 0


def _mt_action_relation(
    entity: AbstractEntity,
    root: AbstractEntity | None,
    vector: tuple[float, float] | None,
) -> str:
    if root is None:
        return "unanchored"
    if entity.entity_id == root.entity_id:
        return "overlap"
    if entity.center is None or root.center is None:
        return "unanchored"
    dr = entity.center[0] - root.center[0]
    dc = entity.center[1] - root.center[1]
    if vector is None:
        distance = math.hypot(dr, dc)
        if distance <= 1.5:
            return "radial_near"
        if distance <= 5.0:
            return "radial_mid"
        return "radial_far"
    forward = dr * vector[0] + dc * vector[1]
    lateral = dr * -vector[1] + dc * vector[0]
    if abs(forward) >= abs(lateral):
        return "ahead" if forward > 0.0 else "behind"
    return "lateral_right" if lateral > 0.0 else "lateral_left"


def _mt_graph(
    state: AbstractState,
    action: ActionCandidate,
    root_id: str | None,
) -> MorphoTopologicalGraph:
    by_id = {entity.entity_id: entity for entity in state.entities}
    root = by_id.get(root_id or "")
    vector, _ = _axis_vector(action, state, root_id)
    nodes = []
    for entity in state.entities:
        attributes = dict(_structural_attributes(entity))
        roles = set(_safe_roles(entity))
        if entity.entity_id == root_id:
            roles.add("action_root")
        nodes.append(
            MTNode(
                node_id=entity.entity_id,
                kind=_entity_kind(entity),
                roles=tuple(sorted(roles)),
                area_bucket=attributes.get("area", "unknown"),
                aspect_bucket=attributes.get("aspect", "unknown"),
                compactness_bucket=attributes.get("compactness", "unknown"),
                holes=_raw_attribute_int(entity, "holes"),
                boundary_contacts=_raw_attribute_int(entity, "boundary_contacts"),
                action_relation=_mt_action_relation(entity, root, vector),
                cells=frozenset(),
                center=entity.center or (0.0, 0.0),
            )
        )
    relations = []
    entity_ids = set(by_id)
    for fact in state.true_facts:
        if (
            fact.predicate in _MT_RELATIONS
            and len(fact.terms) == 2
            and all(term in entity_ids for term in fact.terms)
            and not fact.value
        ):
            relations.append(MTRelation(fact.predicate, fact.terms[0], fact.terms[1]))
    object_nodes = [node for node in nodes if node.kind == "object"]
    free_nodes = [node for node in nodes if node.kind == "free_region"]
    contacts = {
        tuple(sorted((relation.subject_id, relation.object_id)))
        for relation in relations
        if relation.kind == "contact" and relation.subject_id != relation.object_id
    }
    invariants = {
        "object_components": len(object_nodes),
        "free_regions": len(free_nodes),
        "holes": sum(node.holes for node in object_nodes),
        "euler_characteristic": len(object_nodes)
        - sum(node.holes for node in object_nodes),
        "contact_edges": len(contacts),
        "boundary_connected_free_regions": sum(
            int(node.boundary_contacts > 0) for node in free_nodes
        ),
        "largest_free_region_bucket": max(
            (
                {
                    "one": 1,
                    "small": 2,
                    "medium": 3,
                    "large": 4,
                    "very_large": 5,
                }.get(node.area_bucket, 0)
                for node in free_nodes
            ),
            default=0,
        ),
    }
    return MorphoTopologicalGraph(
        nodes=tuple(nodes),
        relations=tuple(relations),
        invariants=invariants,
        action_name=action.action_name,
        action_family=_action_family(action),
        signature="",
    )


def _topological_state(
    source: AbstractState,
    graph: MorphoTopologicalGraph,
) -> AbstractState:
    groups: dict[str, tuple[dict[str, Any], int]] = {}
    node_signature: dict[str, str] = {}
    for node in graph.nodes:
        descriptor = node.model_view()
        signature = canonical_json(descriptor)
        node_signature[node.node_id] = signature
        if signature in groups:
            stored, count = groups[signature]
            groups[signature] = (stored, count + 1)
        else:
            groups[signature] = (descriptor, 1)
    ordered = sorted(groups.items())
    aliases = {signature: f"mt{index}" for index, (signature, _) in enumerate(ordered)}
    entities = []
    for signature, (descriptor, count) in ordered:
        roles = {str(role) for role in descriptor["roles"]}
        roles.add("structural_class")
        attributes = {
            "action_relation": str(descriptor["action_relation"]),
            "area": str(descriptor["area_bucket"]),
            "aspect": str(descriptor["aspect_bucket"]),
            "boundary_contact_bucket": _count_bucket(
                int(descriptor["boundary_contacts"])
            ),
            "compactness": str(descriptor["compactness_bucket"]),
            "hole_bucket": _count_bucket(int(descriptor["holes"])),
            "kind": str(descriptor["kind"]),
            "multiplicity": _count_bucket(count),
        }
        entities.append(
            AbstractEntity(
                aliases[signature],
                tuple(sorted(roles)),
                tuple(sorted(attributes.items())),
            )
        )
    facts = _global_facts(source)
    facts.update(GroundFact("exists", (entity.entity_id,)) for entity in entities)
    for relation in graph.relations:
        if relation.kind not in _MT_RELATIONS:
            continue
        subject = aliases[node_signature[relation.subject_id]]
        obj = aliases[node_signature[relation.object_id]]
        facts.add(GroundFact(relation.kind, (subject, obj)))
    invariants = topological_invariants(graph)
    return AbstractState(
        entities=tuple(entities),
        true_facts=frozenset(facts),
        counters=(("structural_classes", float(len(entities))),),
        topology=tuple(sorted(invariants.items())),
    )


def project_action_rooted_topological(
    frame: ObserverFrameSpec,
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> ProjectedState:
    """Project an ephemeral V4.16 graph to V4.19 topology invariants."""

    _require_frame(frame, ACTION_ROOTED_TOPOLOGICAL_FRAME)
    _require_inputs(state, action, stage)
    resolution = _resolve_root(state, action)
    graph = _mt_graph(state, action, resolution.entity_id)
    projected = _topological_state(state, graph)
    missing = set(resolution.missing)
    if not state.entities:
        missing.add("structural_entities")
    root = next(
        (
            entity
            for entity in state.entities
            if entity.entity_id == resolution.entity_id
        ),
        None,
    )
    if root is not None and root.center is None and len(state.entities) > 1:
        missing.add("action_relative_geometry")
    return _final_projected_state(
        projected,
        missing=tuple(sorted(missing)),
        covered_channels=(
            "entities",
            "facts",
            "counters",
            "objects",
            "relations",
            "topology",
        ),
        provenance=(
            "sage12_v4_16_morpho_topological",
            "sage12_v4_19_topological_invariants",
            "t10_2_abstract_state_adapter",
        ),
        audit_tags=_projection_tags(state),
    )


FRAME_PROJECTORS: Mapping[str, FrameProjector] = MappingProxyType(
    {
        ROOT_ONLY_FRAME.frame_id: project_root_only,
        ALLOCENTRIC_OBJECT_RELATIVE_FRAME.frame_id: (
            project_allocentric_object_relative
        ),
        ACTION_ALIGNED_RELATIONAL_FRAME.frame_id: project_action_aligned_relational,
        ACTION_ROOTED_TOPOLOGICAL_FRAME.frame_id: project_action_rooted_topological,
    }
)


def frame_projector(frame: ObserverFrameSpec | str) -> FrameProjector:
    """Resolve the concrete projector for one frozen frame."""

    frame_id = frame.frame_id if isinstance(frame, ObserverFrameSpec) else str(frame)
    normalized = frame_id.strip().lower()
    try:
        projector = FRAME_PROJECTORS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"no concrete projector for observer frame: {frame_id}"
        ) from exc
    if isinstance(frame, ObserverFrameSpec):
        expected = next(
            item for item in OBSERVER_FRAME_SPECS if item.frame_id == normalized
        )
        if frame != expected:
            raise ValueError(f"observer frame spec is not the frozen {normalized} spec")
    return projector


def concrete_frame_projectors(
    frames: Sequence[ObserverFrameSpec] = OBSERVER_FRAME_SPECS,
) -> dict[str, FrameProjector]:
    """Return validated projector hooks for a selected frozen frame bank."""

    selected = tuple(frames)
    if len({frame.frame_id for frame in selected}) != len(selected):
        raise ValueError("observer frame list contains duplicates")
    return {frame.frame_id: frame_projector(frame) for frame in selected}


def project_frozen_frame(
    frame: ObserverFrameSpec,
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> ProjectedState:
    """Dispatch one state through its registered concrete frame adapter."""

    return frame_projector(frame)(frame, state, action, stage)


def project_transition_with_frozen_frames(
    evidence: ObservedTransition,
    *,
    frames: Sequence[ObserverFrameSpec] = OBSERVER_FRAME_SPECS,
    event_id: str | None = None,
    event_nonce: str = "",
) -> PhysicalEventBundle:
    """Wrap one transition with every requested concrete structural view."""

    selected = tuple(frames)
    return project_observed_transition(
        evidence,
        frames=selected,
        projectors=concrete_frame_projectors(selected),
        event_id=event_id,
        event_nonce=event_nonce,
    )


__all__ = [
    "FORMAT_VERSION",
    "FRAME_PROJECTORS",
    "MAXIMUM_NEIGHBOR_CLASSES",
    "TOPOLOGY_DISTANCE_SENTINEL",
    "concrete_frame_projectors",
    "frame_projector",
    "project_action_aligned_relational",
    "project_action_rooted_topological",
    "project_allocentric_object_relative",
    "project_frozen_frame",
    "project_root_only",
    "project_transition_with_frozen_frames",
]
