"""Goal-progress frame adapter for SAGE.T10.3.

T10.3 deliberately keeps the frozen T10.2 frame bank, but fixes how an
action root is selected.  Root resolution is action-family aware and happens
from the pre-action state only.  The after root is obtained solely by an
exact, unique structural-signature match; action effects are never inspected
to invent a post-action root.

Raw entity identifiers and spatial anchors are transient.  The only public
binding evidence is a method name, a structural signature digest, uniqueness,
and completeness flags.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import AbstractEntity, AbstractState, ActionCandidate, ObservedTransition
from .frame_adapters_v10_2 import project_transition_with_frozen_frames
from .observer_frames_v10_2 import (
    OBSERVER_FRAME_SPECS,
    ObserverFrameSpec,
    PhysicalEventBundle,
)

FORMAT_VERSION = "sage-t10.3-frame-adapter-v1"
MOVEMENT_ACTIONS = frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4"})
EXPLICIT_BINDING_KEYS = frozenset(
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
SAFE_ROLES = frozenset(
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
FORBIDDEN_STRUCTURAL_ATTRIBUTE_KEYS = frozenset(
    {"center", "col", "color", "entity_id", "grid", "id", "row", "x", "y"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RootBindingEvidence:
    """Identity-free, serializable evidence for one pre-action root."""

    method: str
    structural_signature: str | None
    unique: bool
    complete: bool
    after_signature_match: bool
    missing: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "method": self.method,
            "structural_signature": self.structural_signature,
            "unique": self.unique,
            "complete": self.complete,
            "after_signature_match": self.after_signature_match,
            "missing": list(self.missing),
            "raw_identifier_retained": False,
            "spatial_anchor_retained": False,
        }


@dataclass(frozen=True)
class GoalFrameProjection:
    bundle: PhysicalEventBundle
    binding: RootBindingEvidence


@dataclass(frozen=True)
class _ResolvedRoot:
    entity_id: str | None
    method: str
    unique: bool
    missing: tuple[str, ...] = ()


def _entity_signature_payload(state: AbstractState, entity: AbstractEntity) -> dict[str, Any]:
    """Return a value-free, identity-free local structural descriptor."""

    entity_ids = {item.entity_id for item in state.entities}
    fact_degrees: Counter[str] = Counter()
    for truth_name, facts in (("true", state.true_facts), ("false", state.false_facts)):
        for fact in facts:
            if entity.entity_id not in fact.terms and fact.value != entity.entity_id:
                continue
            peer_count = sum(term in entity_ids and term != entity.entity_id for term in fact.terms)
            fact_degrees[f"{truth_name}:{fact.predicate}:{len(fact.terms)}:{peer_count}"] += 1
    relation_degrees: Counter[str] = Counter()
    for other in state.entities:
        if other.entity_id == entity.entity_id:
            continue
        for key, value in other.attributes:
            if value == entity.entity_id:
                relation_degrees[f"incoming:{key}"] += 1
    for key, value in entity.attributes:
        relation_degrees[f"outgoing:{key}:{'entity' if value in entity_ids else 'literal'}"] += 1
    return {
        "roles": sorted(set(entity.roles) & SAFE_ROLES),
        "attribute_keys": sorted(
            key
            for key, _value in entity.attributes
            if key not in FORBIDDEN_STRUCTURAL_ATTRIBUTE_KEYS
        ),
        "fact_degrees": sorted(fact_degrees.items()),
        "relation_degrees": sorted(relation_degrees.items()),
    }


def structural_signature(state: AbstractState, entity_id: str) -> str:
    by_id = {entity.entity_id: entity for entity in state.entities}
    try:
        entity = by_id[entity_id]
    except KeyError as exc:
        raise ValueError("root entity is absent from the state") from exc
    return _digest(_entity_signature_payload(state, entity))


def _unique(candidates: Sequence[str], *, method: str, missing: str) -> _ResolvedRoot:
    selected = tuple(dict.fromkeys(str(item) for item in candidates))
    if len(selected) == 1:
        return _ResolvedRoot(selected[0], method, True)
    if selected:
        return _ResolvedRoot(None, f"ambiguous_{method}", False, (missing,))
    return _ResolvedRoot(None, f"missing_{method}", False, (missing,))


def _explicit_bindings(state: AbstractState, action: ActionCandidate) -> tuple[str, ...]:
    entity_ids = {entity.entity_id for entity in state.entities}
    return tuple(
        str(value)
        for key, value in action.action_data.items()
        if key in EXPLICIT_BINDING_KEYS and str(value) in entity_ids
    )


def _action_anchor(action: ActionCandidate) -> tuple[float, float] | None:
    row = action.action_data.get("row", action.action_data.get("y"))
    col = action.action_data.get("col", action.action_data.get("x"))
    try:
        parsed = (float(row), float(col))
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(value) for value in parsed) else None


def _anchored_root(state: AbstractState, action: ActionCandidate) -> _ResolvedRoot:
    anchor = _action_anchor(action)
    if anchor is None:
        return _ResolvedRoot(None, "missing_action_anchor", False, ("action_anchor",))
    located = tuple(entity for entity in state.entities if entity.center is not None)
    if not located:
        return _ResolvedRoot(None, "missing_action_anchor", False, ("action_anchor",))
    distances = {
        entity.entity_id: math.dist(anchor, entity.center or anchor) for entity in located
    }
    minimum = min(distances.values())
    nearest = tuple(
        entity_id
        for entity_id, distance in distances.items()
        if math.isclose(distance, minimum, abs_tol=1e-9)
    )
    return _unique(nearest, method="transient_action_anchor", missing="unique_action_anchor")


def resolve_pre_action_root(state: AbstractState, action: ActionCandidate) -> _ResolvedRoot:
    """Resolve a root without consulting the after state or action effects."""

    if action.action_name in MOVEMENT_ACTIONS:
        actors = tuple(
            entity.entity_id
            for entity in state.entities
            if set(entity.roles) & {"actor", "player"}
        )
        result = _unique(actors, method="movement_actor", missing="unique_actor")
        if result.entity_id is not None or actors:
            return result

    explicit = _explicit_bindings(state, action)
    if explicit:
        return _unique(explicit, method="explicit_action_binding", missing="unique_action_binding")

    if action.action_data and action.action_name not in MOVEMENT_ACTIONS:
        return _anchored_root(state, action)

    by_id = {entity.entity_id: entity for entity in state.entities}
    registers = tuple(
        value
        for key, value in state.registers
        if key in {"selected", "action_root"} and value in by_id
    )
    if registers:
        return _unique(registers, method="state_register", missing="unique_state_register")

    selected_facts = tuple(
        fact.terms[0]
        for fact in state.true_facts
        if fact.predicate == "selected"
        and len(fact.terms) == 1
        and fact.terms[0] in by_id
    )
    if selected_facts:
        return _unique(selected_facts, method="selected_fact", missing="unique_selected_fact")

    for role in ("selected", "action_root", "target"):
        candidates = tuple(
            entity.entity_id for entity in state.entities if role in entity.roles
        )
        if candidates:
            return _unique(candidates, method=f"structural_{role}", missing=f"unique_{role}")
    return _ResolvedRoot(None, "unresolved", False, ("root_entity",))


def _after_root_by_signature(
    before: AbstractState,
    after: AbstractState,
    root_id: str,
) -> str | None:
    # A branch-local compiler id that survives the transition is a continuation
    # of the already-resolved pre-action binding, not a post-action inference.
    # It remains transient and is never serialized.
    if any(entity.entity_id == root_id for entity in after.entities):
        return root_id
    signature = structural_signature(before, root_id)
    matches = tuple(
        entity.entity_id
        for entity in after.entities
        if structural_signature(after, entity.entity_id) == signature
    )
    return matches[0] if len(matches) == 1 else None


def _with_root(state: AbstractState, entity_id: str | None) -> AbstractState:
    registers = {key: value for key, value in state.registers if key != "action_root"}
    if entity_id is not None:
        registers["action_root"] = entity_id
    return replace(state, registers=tuple(registers.items()))


def project_goal_transition(
    evidence: ObservedTransition,
    *,
    frames: Sequence[ObserverFrameSpec] = OBSERVER_FRAME_SPECS,
    event_id: str | None = None,
    event_nonce: str = "",
) -> GoalFrameProjection:
    """Project one transition through the frozen frames with corrected rooting."""

    resolved = resolve_pre_action_root(evidence.state_before, evidence.action)
    signature = (
        structural_signature(evidence.state_before, resolved.entity_id)
        if resolved.entity_id is not None
        else None
    )
    after_root = (
        _after_root_by_signature(
            evidence.state_before, evidence.state_after, resolved.entity_id
        )
        if resolved.entity_id is not None
        else None
    )
    missing = list(resolved.missing)
    if resolved.entity_id is not None and after_root is None:
        missing.append("after_structural_signature_match")
    rooted = replace(
        evidence,
        state_before=_with_root(evidence.state_before, resolved.entity_id),
        state_after=_with_root(evidence.state_after, after_root),
    )
    bundle = project_transition_with_frozen_frames(
        rooted,
        frames=frames,
        event_id=event_id,
        event_nonce=event_nonce,
    )
    binding = RootBindingEvidence(
        method=resolved.method,
        structural_signature=signature,
        unique=resolved.unique,
        complete=bool(resolved.entity_id is not None and after_root is not None),
        after_signature_match=after_root is not None,
        missing=tuple(sorted(set(missing))),
    )
    return GoalFrameProjection(bundle=bundle, binding=binding)


def assert_safe_binding_payload(payload: Mapping[str, Any]) -> None:
    """Reject accidental persistence of raw grounding material."""

    forbidden_keys = {
        "center", "color", "col", "column", "entity", "entity_id", "grid",
        "object_id", "row", "target_id", "x", "y",
    }
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise ValueError(f"forbidden grounding field persisted: {key}")
                stack.append(item)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


__all__ = [
    "FORMAT_VERSION",
    "GoalFrameProjection",
    "MOVEMENT_ACTIONS",
    "RootBindingEvidence",
    "assert_safe_binding_payload",
    "project_goal_transition",
    "resolve_pre_action_root",
    "structural_signature",
]
