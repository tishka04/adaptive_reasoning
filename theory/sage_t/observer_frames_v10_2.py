"""Immutable observer-frame projections for the SAGE.T10.2 challenger.

This module is deliberately additive.  The frozen SAGE.T contracts continue
to describe one executable program and one abstract state.  T10.2 wraps those
contracts with several action-conditioned views of the *same* physical event.

The central firewall is structural:

* :class:`PhysicalEventBundle` owns exactly one common outcome packet;
* :class:`ProjectedTransition` owns only frame-specific before/after states;
* canonical projection payloads omit coordinates, raw colours, game ids and
  grounded entity ids;
* missing frame-specific information is explicit and falls back to the source
  :class:`~theory.sage_t.contracts.AbstractState` without claiming completeness.

Concrete grid encoders can be supplied as projector hooks by later T10.2
modules.  The hooks here operate only on existing SAGE.T contracts, which keeps
unit tests independent of the live ARC environment.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from .contracts import (
    OBJECT_EVENT_PREDICATES,
    RELATION_PREDICATES,
    TOPOLOGY_PREDICATES,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
)

FORMAT_VERSION = "sage-t10.2-observer-frames-v1"
MAXIMUM_OBSERVER_FRAMES = 4
MAXIMUM_CANONICAL_LABELINGS = 65_536

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COORDINATE_LITERAL = re.compile(
    r"(?<![a-z0-9])\(?\s*-?\d+\s*,\s*-?\d+\s*\)?(?![a-z0-9])",
    re.IGNORECASE,
)
_SOURCE_GAME_ID = re.compile(
    r"(?<![a-z0-9])[a-z]{2}\d{2}(?:-[0-9a-f]{8})?(?![a-z0-9])",
    re.IGNORECASE,
)
_INTERNAL_MT_ALIAS = re.compile(r"^mt\d+$", re.IGNORECASE)
_PERSISTENT_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:persistent|global|uuid|object_id|target_object_id|game_id)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_UUID_LITERAL = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])",
    re.IGNORECASE,
)
_HEX_COLOR = re.compile(r"^#[0-9a-f]{3}(?:[0-9a-f]{3})?$", re.IGNORECASE)
_RAW_COLOR_WORDS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "cyan",
        "gray",
        "green",
        "grey",
        "magenta",
        "orange",
        "purple",
        "red",
        "white",
        "yellow",
    }
)
_RAW_VALUE_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:color|colour|pixel|palette)[_:=#-]"
    r"(?!(?:changed|count|invariant|relation)(?:$|[^a-z0-9]))",
    re.IGNORECASE,
)

PROJECTION_CHANNEL_VOCABULARY_VERSION = "sage-t10.2-projection-channels-v1"

# State coverage describes which fields of the frame-local AbstractState are
# available.  Predictive coverage describes which deltas can be scored after
# comparing two projected states.  Keep the historical PROJECTION_CHANNELS
# alias state-only so existing identity adapters retain their frozen default.
STATE_PROJECTION_CHANNELS = (
    "entities",
    "facts",
    "counters",
    "registers",
    "topology",
    "regime",
)
PREDICTIVE_PROJECTION_CHANNELS = (
    "objects",
    "relations",
    "topology",
)
PROJECTION_CHANNELS = STATE_PROJECTION_CHANNELS
PROJECTION_CHANNEL_VOCABULARY = (
    *STATE_PROJECTION_CHANNELS,
    *(
        channel
        for channel in PREDICTIVE_PROJECTION_CHANNELS
        if channel not in STATE_PROJECTION_CHANNELS
    ),
)
_PROJECTION_CHANNEL_VOCABULARY = frozenset(PROJECTION_CHANNEL_VOCABULARY)

FORBIDDEN_IDENTITY_FIELDS = frozenset(
    {
        "bbox",
        "cell",
        "cells",
        "center",
        "col",
        "color",
        "colour",
        "coordinate",
        "coordinates",
        "cx",
        "cy",
        "frame_after",
        "frame_before",
        "game_id",
        "global_id",
        "grid_hash",
        "object_id",
        "palette",
        "persistent_id",
        "pixel",
        "position",
        "raw_grid",
        "raw_value",
        "rgb",
        "row",
        "seed",
        "seed_id",
        "source_game_id",
        "target_object_id",
        "trace_digest",
        "uuid",
        "value",
        "value_token",
        "x",
        "y",
    }
)

_COUNT_BUCKETS = frozenset({"zero", "one", "two", "few", "many", "unknown"})
_AREA_BUCKETS = frozenset({"one", "small", "medium", "large", "very_large", "unknown"})
_ASPECT_BUCKETS = frozenset({"compact", "square", "tall", "wide", "unknown"})
_COMPACTNESS_BUCKETS = frozenset({"compact", "irregular", "round", "sparse", "unknown"})
_DIRECTIONS = frozenset(
    {
        "north",
        "north_east",
        "east",
        "south_east",
        "south",
        "south_west",
        "west",
        "north_west",
        "overlap",
        "unknown",
    }
)

# Canonical observer states accept only the finite structural vocabulary emitted
# by the four frozen adapters.  This is deliberately a schema, not a blacklist:
# new attributes require an explicit protocol amendment before they can affect
# frame hashes or posterior evidence.
STRUCTURAL_ATTRIBUTE_SCHEMA: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "action_family": frozenset({"move", "interact", "parameterized", "other"}),
        "action_relation": frozenset(
            {
                "ahead",
                "behind",
                "lateral_left",
                "lateral_right",
                "overlap",
                "radial_far",
                "radial_mid",
                "radial_near",
                "unanchored",
            }
        ),
        "actor_relation": frozenset(
            {"contact", "adjacent", "near", "overlap", "mid", "far", "unknown"}
        ),
        "actor_relative_direction": _DIRECTIONS,
        "affordance": frozenset(
            {"blocked", "fixed", "interactive", "movable", "passable", "unknown"}
        ),
        "aligned_col": frozenset({"0", "1"}),
        "aligned_row": frozenset({"0", "1"}),
        "area": _AREA_BUCKETS,
        "area_bucket": _AREA_BUCKETS,
        "aspect": _ASPECT_BUCKETS,
        "aspect_bucket": _ASPECT_BUCKETS,
        "axis_relation": frozenset(
            {"ahead", "behind", "lateral_left", "lateral_right", "overlap", "radial"}
        ),
        "axis_source": frozenset({"movement", "actor_to_target", "none"}),
        "boundary_contact_bucket": _COUNT_BUCKETS,
        "compactness": _COMPACTNESS_BUCKETS,
        "compactness_bucket": _COMPACTNESS_BUCKETS,
        "direction": _DIRECTIONS,
        "hole_bucket": _COUNT_BUCKETS,
        "is_actor": frozenset({"0", "1"}),
        "kind": frozenset({"object", "free_region"}),
        "multiplicity": _COUNT_BUCKETS,
        "occupied": frozenset({"yes", "no"}),
        "path_status": frozenset(
            {"blocked", "open", "reachable", "unreachable", "unknown"}
        ),
        "player_available": frozenset({"0", "1"}),
        "proximity": frozenset(
            {"contact", "adjacent", "near", "overlap", "mid", "far", "unknown"}
        ),
        "relative_size": frozenset({"smaller", "equal", "larger", "unknown"}),
        "requested_direction": frozenset(
            {
                "up",
                "down",
                "left",
                "right",
                "north",
                "north_east",
                "east",
                "south_east",
                "south",
                "south_west",
                "west",
                "north_west",
                "none",
            }
        ),
        "root_affordance": frozenset(
            {"blocked", "fixed", "interactive", "movable", "passable", "unknown"}
        ),
        "root_area_bucket": _AREA_BUCKETS,
        "root_aspect_bucket": _ASPECT_BUCKETS,
        "root_kind": frozenset(
            {"action_root", "actor", "free_region", "occupied_object", "target_object"}
        ),
        "root_occupied": frozenset({"0", "1"}),
        "size": _AREA_BUCKETS,
        "topology_relation": frozenset(
            {"root_contact", "root_adjacent", "near", "distant", "unknown"}
        ),
    }
)


def _safe_identifier(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{label} must be a bounded snake_case identifier")
    return normalized


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {
            key: _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        rendered = [_json_safe(item) for item in value]
        return sorted(
            rendered,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError("canonical JSON numbers must be finite")
        return value
    raise TypeError(f"{type(value).__name__} is not supported by canonical JSON")


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used by every checksum."""

    try:
        return json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("canonical JSON requires finite JSON-compatible data") from exc


def canonical_sha256(value: Any) -> str:
    """Hash a value after deterministic JSON normalization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else str(key)


def _string_identity_leaks(
    value: str,
    *,
    path: str,
    forbidden_game_ids: frozenset[str],
) -> list[str]:
    text = str(value).strip()
    lowered = text.lower()
    leaks = []
    if lowered in forbidden_game_ids or (
        not _INTERNAL_MT_ALIAS.fullmatch(lowered)
        and any(_SOURCE_GAME_ID.finditer(lowered))
    ):
        leaks.append(f"game_id:{path}")
    if _COORDINATE_LITERAL.search(text):
        leaks.append(f"absolute_coordinate:{path}")
    if _PERSISTENT_TOKEN.search(lowered):
        leaks.append(f"persistent_identity:{path}")
    if _UUID_LITERAL.search(lowered):
        leaks.append(f"persistent_identity:{path}")
    if (
        _RAW_VALUE_TOKEN.search(lowered)
        or lowered in _RAW_COLOR_WORDS
        or _HEX_COLOR.fullmatch(lowered)
    ):
        leaks.append(f"raw_value:{path}")
    return leaks


def audit_identity_leaks(
    payload: Any,
    *,
    forbidden_game_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return deterministic identity/firewall violations in ``payload``.

    The audit is intentionally generic so later collectors can apply it to
    their compact model views before persisting them.  Paths, rather than raw
    values, are returned so the audit itself cannot echo sensitive identity.
    """

    forbidden = frozenset(str(item).strip().lower() for item in forbidden_game_ids)
    leaks: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                key = str(raw_key)
                normalized = key.strip().lower().replace("-", "_")
                child = _normalized_path(path, key)
                if normalized in FORBIDDEN_IDENTITY_FIELDS:
                    leaks.append(f"forbidden_field:{child}")
                leaks.extend(
                    _string_identity_leaks(
                        key,
                        path=child,
                        forbidden_game_ids=forbidden,
                    )
                )
                visit(item, child)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, (set, frozenset)):
            for index, item in enumerate(sorted(value, key=str)):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, str):
            leaks.extend(
                _string_identity_leaks(
                    value,
                    path=path or "$",
                    forbidden_game_ids=forbidden,
                )
            )

    visit(payload, "$")
    return tuple(sorted(set(leaks)))


@dataclass(frozen=True)
class ObserverFrameSpec:
    """Versioned description of one observer frame."""

    frame_id: str
    family: str
    encoder_version: str
    action_conditioned: bool
    complexity_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_id",
            _safe_identifier(self.frame_id, label="frame id"),
        )
        object.__setattr__(
            self,
            "family",
            _safe_identifier(self.family, label="frame family"),
        )
        object.__setattr__(
            self,
            "encoder_version",
            _safe_identifier(self.encoder_version, label="encoder version"),
        )
        cost = float(self.complexity_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("frame complexity cost must be finite and non-negative")
        object.__setattr__(self, "complexity_cost", cost)
        leaks = audit_identity_leaks(
            {
                "frame_id": self.frame_id,
                "family": self.family,
                "encoder_version": self.encoder_version,
            }
        )
        if leaks:
            raise ValueError(f"identity leak in observer frame spec: {leaks[0]}")

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "frame_id": self.frame_id,
            "family": self.family,
            "encoder_version": self.encoder_version,
            "action_conditioned": bool(self.action_conditioned),
            "complexity_cost": self.complexity_cost,
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash


ROOT_ONLY_FRAME = ObserverFrameSpec(
    frame_id="root_only",
    family="root_only",
    encoder_version="sage12_v4_9",
    action_conditioned=True,
)
ALLOCENTRIC_OBJECT_RELATIVE_FRAME = ObserverFrameSpec(
    frame_id="allocentric_object_relative",
    family="object_relative",
    encoder_version="sage12_v4_9",
    action_conditioned=True,
)
ACTION_ALIGNED_RELATIONAL_FRAME = ObserverFrameSpec(
    frame_id="action_aligned_relational",
    family="action_aligned_relational",
    encoder_version="sage12_v4_10",
    action_conditioned=True,
)
ACTION_ROOTED_TOPOLOGICAL_FRAME = ObserverFrameSpec(
    frame_id="action_rooted_topological",
    family="action_rooted_topological",
    encoder_version="sage12_v4_19",
    action_conditioned=True,
)

OBSERVER_FRAME_SPECS = (
    ROOT_ONLY_FRAME,
    ALLOCENTRIC_OBJECT_RELATIVE_FRAME,
    ACTION_ALIGNED_RELATIONAL_FRAME,
    ACTION_ROOTED_TOPOLOGICAL_FRAME,
)

_FRAME_BY_ID = {frame.frame_id: frame for frame in OBSERVER_FRAME_SPECS}


def observer_frame_spec(frame_id: str) -> ObserverFrameSpec:
    """Resolve one of the four frozen T10.2 frame specifications."""

    normalized = str(frame_id).strip().lower()
    try:
        return _FRAME_BY_ID[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown observer frame: {frame_id}") from exc


def _fact_payload(
    fact: GroundFact,
    *,
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "predicate": fact.predicate,
        "terms": [aliases.get(term, term) for term in fact.terms],
        "literal": aliases.get(fact.value, fact.value),
    }


def _assign_refinement_colors(signatures: Mapping[str, str]) -> dict[str, int]:
    palette = {
        signature: index
        for index, signature in enumerate(sorted(set(signatures.values())))
    }
    return {
        entity_id: palette[signature] for entity_id, signature in signatures.items()
    }


def _refined_entity_colors(state: AbstractState) -> dict[str, int]:
    entity_ids = frozenset(entity.entity_id for entity in state.entities)
    signatures = {
        entity.entity_id: canonical_json(
            {
                "roles": sorted(entity.roles),
                "attributes": sorted(
                    (
                        str(key),
                        ["entity"] if value in entity_ids else ["literal", str(value)],
                    )
                    for key, value in entity.attributes
                ),
            }
        )
        for entity in state.entities
    }
    colors = _assign_refinement_colors(signatures)
    for _ in range(len(state.entities) + 1):
        refined: dict[str, str] = {}
        for entity in state.entities:
            entity_id = entity.entity_id

            def term_token(
                value: str,
                *,
                current_entity_id: str = entity_id,
                current_colors: Mapping[str, int] = colors,
            ) -> list[Any]:
                if value == current_entity_id:
                    return ["self"]
                if value in entity_ids:
                    return ["entity", current_colors[value]]
                return ["literal", str(value)]

            fact_context = []
            for truth, facts in ((True, state.true_facts), (False, state.false_facts)):
                for fact in facts:
                    if entity_id not in fact.terms and fact.value != entity_id:
                        continue
                    fact_context.append(
                        {
                            "truth": truth,
                            "predicate": fact.predicate,
                            "terms": [term_token(term) for term in fact.terms],
                            "literal": term_token(fact.value),
                        }
                    )
            refined[entity_id] = canonical_json(
                {
                    "color": colors[entity_id],
                    "roles": sorted(entity.roles),
                    "attributes": sorted(
                        (
                            str(key),
                            term_token(str(value)),
                        )
                        for key, value in entity.attributes
                    ),
                    "facts": sorted(fact_context, key=canonical_json),
                    "registers": sorted(
                        str(key) for key, value in state.registers if value == entity_id
                    ),
                }
            )
        next_colors = _assign_refinement_colors(refined)
        if next_colors == colors:
            return colors
        colors = next_colors
    return colors


def _relationally_referenced_entity_ids(state: AbstractState) -> frozenset[str]:
    entity_ids = frozenset(entity.entity_id for entity in state.entities)
    referenced: set[str] = set()
    for entity in state.entities:
        for _key, value in entity.attributes:
            if value in entity_ids:
                referenced.update((entity.entity_id, value))
    for facts in (state.true_facts, state.false_facts):
        for fact in facts:
            fact_entities = [term for term in fact.terms if term in entity_ids]
            if fact.value in entity_ids:
                fact_entities.append(fact.value)
            if len(fact_entities) >= 2:
                referenced.update(fact_entities)
    referenced.update(value for _key, value in state.registers if value in entity_ids)
    return frozenset(referenced)


def _state_entity_orders(state: AbstractState) -> tuple[tuple[Any, ...], ...]:
    entity_ids = [entity.entity_id for entity in state.entities]
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("observer-frame states require unique local entity ids")
    if not state.entities:
        return ((),)
    colors = _refined_entity_colors(state)
    groups: dict[int, list[Any]] = {}
    for entity in state.entities:
        groups.setdefault(colors[entity.entity_id], []).append(entity)
    referenced = _relationally_referenced_entity_ids(state)
    choices: list[tuple[tuple[Any, ...], ...]] = []
    candidate_count = 1
    for color in sorted(groups):
        group = tuple(groups[color])
        requires_label_search = len(group) > 1 and any(
            entity.entity_id in referenced for entity in group
        )
        if requires_label_search:
            candidate_count *= math.factorial(len(group))
            if candidate_count > MAXIMUM_CANONICAL_LABELINGS:
                raise ValueError(
                    "symmetric observer graph exceeds bounded canonical labeling"
                )
            choices.append(tuple(itertools.permutations(group)))
        else:
            # Members of an unreferenced refined class render identical rows
            # and unary facts, so their local tuple order cannot affect output.
            choices.append((group,))
    return tuple(
        tuple(entity for group in selected for entity in group)
        for selected in itertools.product(*choices)
    )


def _state_payload_for_order(
    state: AbstractState, entities: Sequence[Any]
) -> dict[str, Any]:
    aliases = {entity.entity_id: f"e{index}" for index, entity in enumerate(entities)}
    return {
        "entities": [
            {
                "alias": aliases[entity.entity_id],
                "roles": sorted(entity.roles),
                "attributes": sorted(
                    (
                        [key, aliases.get(value, value)]
                        for key, value in entity.attributes
                    ),
                    key=canonical_json,
                ),
            }
            for entity in entities
        ],
        "true_facts": sorted(
            (_fact_payload(fact, aliases=aliases) for fact in state.true_facts),
            key=canonical_json,
        ),
        "false_facts": sorted(
            (_fact_payload(fact, aliases=aliases) for fact in state.false_facts),
            key=canonical_json,
        ),
        "counters": sorted((list(item) for item in state.counters), key=canonical_json),
        "registers": sorted(
            ([key, aliases.get(value, value)] for key, value in state.registers),
            key=canonical_json,
        ),
        "topology": sorted((list(item) for item in state.topology), key=canonical_json),
        "regime_index": state.regime_index,
    }


def _validate_structural_attributes(state: AbstractState) -> None:
    for entity in state.entities:
        seen: set[str] = set()
        for key, value in entity.attributes:
            if key in seen:
                raise ValueError("duplicate key in closed structural attribute schema")
            seen.add(key)
            allowed = STRUCTURAL_ATTRIBUTE_SCHEMA.get(key)
            if allowed is None:
                raise ValueError(
                    "attribute key is outside the closed structural schema"
                )
            if value not in allowed:
                raise ValueError(
                    "attribute value is outside the closed structural schema"
                )


def state_model_payload(state: AbstractState) -> dict[str, Any]:
    """Render an alpha-invariant, coordinate-free model view of a state."""

    _validate_structural_attributes(state)
    candidates = (
        _state_payload_for_order(state, entities)
        for entities in _state_entity_orders(state)
    )
    payload = min(candidates, key=canonical_json)
    leaks = audit_identity_leaks(payload)
    if leaks:
        raise ValueError(f"identity leak in abstract-state model view: {leaks[0]}")
    return payload


def _state_source_leaks(state: AbstractState) -> tuple[str, ...]:
    """Audit raw symbolic values while allowing branch-local ids and centres."""

    payload = {
        "entities": [
            {
                "local": entity.entity_id,
                "roles": list(entity.roles),
                "attributes": {key: value for key, value in entity.attributes},
            }
            for entity in state.entities
        ],
        "true": [
            {
                "predicate": fact.predicate,
                "terms": list(fact.terms),
                "fact_literal": fact.value,
            }
            for fact in state.true_facts
        ],
        "false": [
            {
                "predicate": fact.predicate,
                "terms": list(fact.terms),
                "fact_literal": fact.value,
            }
            for fact in state.false_facts
        ],
        "counters": {key: value for key, value in state.counters},
        "registers": {key: value for key, value in state.registers},
        "topology": {key: value for key, value in state.topology},
    }
    return audit_identity_leaks(payload)


@dataclass(frozen=True)
class ProjectedState:
    """Optional rich return value for a frame projector hook."""

    state: AbstractState
    complete: bool = True
    missing: tuple[str, ...] = ()
    covered_channels: tuple[str, ...] = PROJECTION_CHANNELS
    provenance: tuple[str, ...] = ()
    audit_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = tuple(
            sorted(
                {
                    _safe_identifier(item, label="missing projection field")
                    for item in self.missing
                }
            )
        )
        tags = tuple(
            sorted(
                {
                    _safe_identifier(item, label="projection audit tag")
                    for item in self.audit_tags
                }
            )
        )
        covered = tuple(
            sorted(
                {
                    _safe_identifier(item, label="covered projection channel")
                    for item in self.covered_channels
                }
            )
        )
        unknown_channels = tuple(sorted(set(covered) - _PROJECTION_CHANNEL_VOCABULARY))
        if unknown_channels:
            raise ValueError(
                f"unknown covered projection channel(s): {', '.join(unknown_channels)}"
            )
        provenance = tuple(
            sorted(
                {
                    _safe_identifier(item, label="projection provenance")
                    for item in self.provenance
                }
            )
        )
        if self.complete and missing:
            raise ValueError("a complete projected state cannot declare missing fields")
        if not self.complete and not missing:
            missing = ("unspecified_projection_data",)
        object.__setattr__(self, "missing", missing)
        object.__setattr__(self, "covered_channels", covered)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "audit_tags", tags)
        leaks = audit_identity_leaks(
            {
                "missing": missing,
                "covered_channels": covered,
                "provenance": provenance,
                "audit_tags": tags,
            }
        )
        if leaks:
            raise ValueError(f"identity leak in projected-state metadata: {leaks[0]}")


@dataclass(frozen=True)
class FrameProjection:
    """One frame-specific state, with a firewalled canonical model view."""

    frame: ObserverFrameSpec
    state: AbstractState
    action: ActionCandidate
    stage: str = "before"
    complete: bool = True
    missing: tuple[str, ...] = ()
    covered_channels: tuple[str, ...] = PROJECTION_CHANNELS
    provenance: tuple[str, ...] = ()
    audit_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stage = str(self.stage).strip().lower()
        if stage not in {"before", "after"}:
            raise ValueError("projection stage must be before or after")
        object.__setattr__(self, "stage", stage)
        normalized = ProjectedState(
            state=self.state,
            complete=bool(self.complete),
            missing=tuple(self.missing),
            covered_channels=tuple(self.covered_channels),
            provenance=tuple(self.provenance),
            audit_tags=tuple(self.audit_tags),
        )
        object.__setattr__(self, "complete", normalized.complete)
        object.__setattr__(self, "missing", normalized.missing)
        object.__setattr__(self, "covered_channels", normalized.covered_channels)
        object.__setattr__(self, "provenance", normalized.provenance)
        object.__setattr__(self, "audit_tags", normalized.audit_tags)
        leaks = _state_source_leaks(self.state)
        if leaks:
            raise ValueError(f"identity leak in frame projection: {leaks[0]}")
        # Constructing the model payload is itself a stricter second firewall.
        state_model_payload(self.state)

    @property
    def frame_id(self) -> str:
        return self.frame.frame_id

    @property
    def canonical_payload(self) -> dict[str, Any]:
        payload = {
            "format_version": FORMAT_VERSION,
            "frame": self.frame.canonical_payload,
            "state": state_model_payload(self.state),
            # Grounded action data are evidence.  The intervention name is the
            # only action field authorized in the transferable projection.
            "action_name": self.action.action_name,
            "stage": self.stage,
            "complete": bool(self.complete),
            "missing": list(self.missing),
            "covered_channels": list(self.covered_channels),
            "provenance": list(self.provenance),
            "audit_tags": list(self.audit_tags),
        }
        leaks = audit_identity_leaks(payload)
        if leaks:
            raise ValueError(f"identity leak in canonical frame projection: {leaks[0]}")
        return payload

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash

    @property
    def source_checksum(self) -> str:
        """Opaque audit checksum including branch-local grounding state."""

        return canonical_sha256(
            {
                "frame_hash": self.frame.canonical_hash,
                "state_execution_signature": self.state.execution_signature,
                "action_key_digest": hashlib.sha256(
                    self.action.key.encode("utf-8")
                ).hexdigest(),
                "stage": self.stage,
            }
        )


ProjectorResult: TypeAlias = AbstractState | ProjectedState | None


class FrameProjector(Protocol):
    """Hook implemented by a concrete frame encoder."""

    def __call__(
        self,
        frame: ObserverFrameSpec,
        state: AbstractState,
        action: ActionCandidate,
        stage: str,
    ) -> ProjectorResult: ...


def identity_projector(
    frame: ObserverFrameSpec,
    state: AbstractState,
    action: ActionCandidate,
    stage: str,
) -> AbstractState:
    """Explicit identity hook useful for controls and contract-only tests."""

    del frame, action, stage
    return state


def project_state(
    state: AbstractState,
    action: ActionCandidate,
    frame: ObserverFrameSpec,
    *,
    stage: str,
    projector: FrameProjector | None = None,
) -> FrameProjection:
    """Project an existing abstract state, with an explicit partial fallback."""

    if projector is None:
        result = ProjectedState(
            state=state,
            complete=False,
            missing=("frame_specific_projection",),
            covered_channels=(),
            provenance=("abstract_state_adapter",),
            audit_tags=("abstract_state_fallback",),
        )
    else:
        raw = projector(frame, state, action, stage)
        if raw is None:
            result = ProjectedState(
                state=state,
                complete=False,
                missing=("frame_specific_projection",),
                covered_channels=(),
                provenance=("projector_hook",),
                audit_tags=("projector_incomplete_fallback",),
            )
        elif isinstance(raw, AbstractState):
            result = ProjectedState(
                state=raw,
                provenance=("projector_hook",),
            )
        elif isinstance(raw, ProjectedState):
            result = raw
        else:
            raise TypeError(
                "frame projector must return AbstractState, ProjectedState or None"
            )
    return FrameProjection(
        frame=frame,
        state=result.state,
        action=action,
        stage=stage,
        complete=result.complete,
        missing=result.missing,
        covered_channels=result.covered_channels,
        provenance=result.provenance,
        audit_tags=result.audit_tags,
    )


def _prediction_packet_payload(packet: PredictionPacket) -> dict[str, Any]:
    return {
        "object_deltas": dict(sorted(packet.object_deltas.items())),
        "relation_deltas": dict(sorted(packet.relation_deltas.items())),
        "topology_deltas": dict(sorted(packet.topology_deltas.items())),
        "progress_mean": packet.progress_mean,
        "progress_distribution": dict(sorted(packet.progress_distribution.items())),
        "terminal_probability": packet.terminal_probability,
        "goal_probability": packet.goal_probability,
        "known_channels": sorted(packet.known_channels),
        "residual": list(packet.residual),
    }


def packet_without_state(packet: PredictionPacket) -> PredictionPacket:
    """Strip only the state carried by a prediction packet."""

    return replace(packet, state_after=None)


def outcome_without_state(packet: PredictionPacket) -> PredictionPacket:
    """Return the one frame-independent physical outcome.

    Objects, relations and topology are observations made through a frame.  A
    :class:`PhysicalEventBundle` therefore stores only progress, terminal and
    goal evidence in its common packet.  Keeping this projection explicit is
    what prevents four observer frames from counting one physical outcome four
    times.
    """

    physical_channels = frozenset(
        set(packet.known_channels) & {"progress", "terminal", "goal"}
    )
    return PredictionPacket(
        progress_mean=(
            packet.progress_mean if "progress" in physical_channels else None
        ),
        progress_distribution=(
            packet.progress_distribution if "progress" in physical_channels else {}
        ),
        terminal_probability=(
            packet.terminal_probability if "terminal" in physical_channels else None
        ),
        goal_probability=(
            packet.goal_probability if "goal" in physical_channels else None
        ),
        known_channels=physical_channels,
        state_after=None,
    )


def merge_projection_and_physical_outcome(
    projection: PredictionPacket,
    common_outcome: PredictionPacket,
    *,
    state_after: AbstractState,
) -> PredictionPacket:
    """Combine one frame's structural evidence with the common outcome."""

    structural = packet_without_state(projection)
    physical = outcome_without_state(common_outcome)
    return PredictionPacket(
        object_deltas=structural.object_deltas,
        relation_deltas=structural.relation_deltas,
        topology_deltas=structural.topology_deltas,
        progress_mean=physical.progress_mean,
        progress_distribution=physical.progress_distribution,
        terminal_probability=physical.terminal_probability,
        goal_probability=physical.goal_probability,
        known_channels=frozenset(
            set(structural.known_channels) | set(physical.known_channels)
        ),
        residual=structural.residual,
        state_after=state_after,
    )


def _entity_descriptor_counts(state: AbstractState) -> Counter[str]:
    return Counter(
        canonical_json(
            {
                "roles": sorted(entity.roles),
                "attributes": sorted(entity.attributes),
            }
        )
        for entity in state.entities
    )


def _projection_outcome(
    before: FrameProjection,
    after: FrameProjection,
) -> PredictionPacket:
    """Derive only frame-relative structural deltas for one projection.

    Progress, terminal state, and goal satisfaction intentionally remain on
    :class:`PhysicalEventBundle`; this packet can therefore be averaged across
    frames without counting the physical outcome more than once.
    """

    covered = set(before.covered_channels) & set(after.covered_channels)
    known: set[str] = set()
    object_deltas: dict[str, float] = {}
    relation_deltas: dict[str, float] = {}
    topology_deltas: dict[str, float] = {}

    if "entities" in covered:
        known.add("objects")
        left = _entity_descriptor_counts(before.state)
        right = _entity_descriptor_counts(after.state)
        created = sum((right - left).values())
        removed = sum((left - right).values())
        if created:
            object_deltas["created"] = 1.0
        if removed:
            object_deltas["removed"] = 1.0

    if "facts" in covered:
        known.add("relations")
        asserted = after.state.true_facts - before.state.true_facts
        retracted = before.state.true_facts - after.state.true_facts
        for fact, operation in (
            *((fact, "added") for fact in asserted),
            *((fact, "removed") for fact in retracted),
        ):
            if fact.predicate in RELATION_PREDICATES:
                relation_deltas[f"relation_{operation}:{fact.predicate}"] = 1.0
            elif fact.predicate in TOPOLOGY_PREDICATES:
                topology_deltas[fact.predicate] = 1.0
            elif fact.predicate in OBJECT_EVENT_PREDICATES:
                object_deltas[fact.predicate] = 1.0
        if object_deltas:
            known.add("objects")

    if "topology" in covered:
        known.add("topology")
        left_topology = dict(before.state.topology)
        right_topology = dict(after.state.topology)
        for key in sorted(set(left_topology) | set(right_topology)):
            if left_topology.get(key) != right_topology.get(key):
                topology_deltas[str(key)] = 1.0

    return PredictionPacket(
        object_deltas=object_deltas,
        relation_deltas=relation_deltas,
        topology_deltas=topology_deltas,
        known_channels=frozenset(known),
        state_after=after.state,
    )


@dataclass(frozen=True)
class ProjectedTransition:
    """Frame-specific states for one event; the outcome lives on its bundle."""

    event_id: str
    before: FrameProjection
    after: FrameProjection

    def __post_init__(self) -> None:
        if not str(self.event_id).strip():
            raise ValueError("projected transition needs an event id")
        object.__setattr__(self, "event_id", str(self.event_id).strip())
        if self.before.stage != "before" or self.after.stage != "after":
            raise ValueError("projected transition requires before and after stages")
        if self.before.frame_id != self.after.frame_id:
            raise ValueError("projected transition frames do not match")
        if self.before.action.key != self.after.action.key:
            raise ValueError("projected transition actions do not match")

    @property
    def frame_id(self) -> str:
        return self.before.frame_id

    @property
    def action(self) -> ActionCandidate:
        return self.before.action

    @property
    def complete(self) -> bool:
        return bool(self.before.complete and self.after.complete)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.before.missing) | set(self.after.missing)))

    @property
    def covered_channels(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self.before.covered_channels) & set(self.after.covered_channels))
        )

    @property
    def provenance(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.before.provenance) | set(self.after.provenance)))

    @property
    def observation(self) -> PredictionPacket:
        """Frame-specific structural evidence, excluding the common outcome."""

        return _projection_outcome(self.before, self.after)

    @property
    def projection_outcome(self) -> PredictionPacket:
        return self.observation

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "event_id_digest": hashlib.sha256(
                self.event_id.encode("utf-8")
            ).hexdigest(),
            "frame_id": self.frame_id,
            "action_name": self.action.action_name,
            "before_hash": self.before.canonical_hash,
            "after_hash": self.after.canonical_hash,
            "complete": self.complete,
            "missing": list(self.missing),
            "covered_channels": list(self.covered_channels),
            "provenance": list(self.provenance),
            "projection_outcome": _prediction_packet_payload(
                packet_without_state(self.observation)
            ),
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash

    def as_observed_transition(
        self,
        common_outcome: PredictionPacket,
        *,
        events: Sequence[str] = (),
        reset: bool = False,
    ) -> ObservedTransition:
        outcome = merge_projection_and_physical_outcome(
            self.observation,
            common_outcome,
            state_after=self.after.state,
        )
        return ObservedTransition(
            state_before=self.before.state,
            action=self.action,
            state_after=self.after.state,
            observation=outcome,
            events=tuple(events),
            reset=bool(reset),
        )


@dataclass(frozen=True)
class PhysicalEventBundle:
    """One physical event, one common outcome, and several observer views."""

    event_id: str
    action: ActionCandidate
    common_outcome: PredictionPacket
    projections: tuple[ProjectedTransition, ...]
    events: tuple[str, ...] = ()
    reset: bool = False

    def __post_init__(self) -> None:
        if not str(self.event_id).strip():
            raise ValueError("physical event bundle needs an event id")
        object.__setattr__(self, "event_id", str(self.event_id).strip())
        projections = tuple(sorted(self.projections, key=lambda item: item.frame_id))
        if not projections:
            raise ValueError(
                "physical event bundle requires at least one observer projection"
            )
        if len(projections) > MAXIMUM_OBSERVER_FRAMES:
            raise ValueError(
                f"physical event bundle exceeds {MAXIMUM_OBSERVER_FRAMES} observer frames"
            )
        frame_ids = [item.frame_id for item in projections]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("physical event bundle contains duplicate frame ids")
        for projection in projections:
            if projection.event_id != self.event_id:
                raise ValueError("projection event id does not match its bundle")
            if projection.action.key != self.action.key:
                raise ValueError("projection action does not match its bundle")
        outcome = outcome_without_state(self.common_outcome)
        outcome_leaks = audit_identity_leaks(_prediction_packet_payload(outcome))
        if outcome_leaks:
            raise ValueError(f"identity leak in common outcome: {outcome_leaks[0]}")
        event_leaks = audit_identity_leaks({"events": tuple(self.events)})
        if event_leaks:
            raise ValueError(f"identity leak in physical events: {event_leaks[0]}")
        object.__setattr__(self, "common_outcome", outcome)
        object.__setattr__(self, "projections", projections)
        object.__setattr__(
            self,
            "events",
            tuple(sorted({str(event) for event in self.events})),
        )
        object.__setattr__(self, "reset", bool(self.reset))

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(item.frame_id for item in self.projections)

    def projection(self, frame_id: str) -> ProjectedTransition:
        normalized = str(frame_id).strip().lower()
        for item in self.projections:
            if item.frame_id == normalized:
                return item
        raise KeyError(f"event has no projection for frame: {frame_id}")

    def observed_transition(self, frame_id: str) -> ObservedTransition:
        return self.projection(frame_id).as_observed_transition(
            self.common_outcome,
            events=self.events,
            reset=self.reset,
        )

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "event_id_digest": hashlib.sha256(
                self.event_id.encode("utf-8")
            ).hexdigest(),
            "action_name": self.action.action_name,
            "common_outcome": _prediction_packet_payload(self.common_outcome),
            "projection_hashes": [item.canonical_hash for item in self.projections],
            "events": list(self.events),
            "reset": bool(self.reset),
        }

    @property
    def canonical_checksum(self) -> str:
        return canonical_sha256(self.canonical_payload)

    @property
    def canonical_hash(self) -> str:
        return self.canonical_checksum


def observed_transition_event_id(
    evidence: ObservedTransition,
    *,
    nonce: str = "",
) -> str:
    """Derive an opaque deterministic id for a contract-only transition."""

    return canonical_sha256(
        {
            "format_version": FORMAT_VERSION,
            "before_execution_signature": evidence.state_before.execution_signature,
            "after_execution_signature": evidence.state_after.execution_signature,
            "action_key_digest": hashlib.sha256(
                evidence.action.key.encode("utf-8")
            ).hexdigest(),
            "outcome": _prediction_packet_payload(
                packet_without_state(evidence.observation)
            ),
            "events": sorted(set(evidence.events)),
            "reset": bool(evidence.reset),
            "nonce_digest": hashlib.sha256(str(nonce).encode("utf-8")).hexdigest(),
        }
    )


def project_observed_transition(
    evidence: ObservedTransition,
    *,
    frames: Sequence[ObserverFrameSpec] = OBSERVER_FRAME_SPECS,
    projectors: Mapping[str, FrameProjector] | None = None,
    event_id: str | None = None,
    event_nonce: str = "",
) -> PhysicalEventBundle:
    """Wrap one observed transition in several frame-specific state views."""

    selected = tuple(frames)
    if not selected:
        raise ValueError("at least one observer frame is required")
    if len(selected) > MAXIMUM_OBSERVER_FRAMES:
        raise ValueError(
            f"observer frame bank exceeds {MAXIMUM_OBSERVER_FRAMES} frames"
        )
    if len({frame.frame_id for frame in selected}) != len(selected):
        raise ValueError("observer frame list contains duplicates")
    hooks = dict(projectors or {})
    unknown_hooks = set(hooks) - {frame.frame_id for frame in selected}
    if unknown_hooks:
        raise ValueError(
            f"projector hooks target unknown frames: {sorted(unknown_hooks)}"
        )
    identifier = event_id or observed_transition_event_id(evidence, nonce=event_nonce)
    projections = []
    for frame in selected:
        hook = hooks.get(frame.frame_id)
        before = project_state(
            evidence.state_before,
            evidence.action,
            frame,
            stage="before",
            projector=hook,
        )
        after = project_state(
            evidence.state_after,
            evidence.action,
            frame,
            stage="after",
            projector=hook,
        )
        projections.append(
            ProjectedTransition(
                event_id=identifier,
                before=before,
                after=after,
            )
        )
    return PhysicalEventBundle(
        event_id=identifier,
        action=evidence.action,
        common_outcome=outcome_without_state(evidence.observation),
        projections=tuple(projections),
        events=evidence.events,
        reset=evidence.reset,
    )


def validate_unique_event_ids(
    bundles: Iterable[PhysicalEventBundle],
) -> tuple[PhysicalEventBundle, ...]:
    """Materialize bundles and fail closed when a physical id is reused."""

    materialized = tuple(bundles)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for bundle in materialized:
        if bundle.event_id in seen:
            duplicates.add(bundle.event_id)
        seen.add(bundle.event_id)
    if duplicates:
        raise ValueError("duplicate physical event ids")
    return materialized


__all__ = [
    "ACTION_ALIGNED_RELATIONAL_FRAME",
    "ACTION_ROOTED_TOPOLOGICAL_FRAME",
    "ALLOCENTRIC_OBJECT_RELATIVE_FRAME",
    "FORBIDDEN_IDENTITY_FIELDS",
    "FORMAT_VERSION",
    "MAXIMUM_CANONICAL_LABELINGS",
    "MAXIMUM_OBSERVER_FRAMES",
    "OBSERVER_FRAME_SPECS",
    "PREDICTIVE_PROJECTION_CHANNELS",
    "PROJECTION_CHANNELS",
    "PROJECTION_CHANNEL_VOCABULARY",
    "PROJECTION_CHANNEL_VOCABULARY_VERSION",
    "ROOT_ONLY_FRAME",
    "STATE_PROJECTION_CHANNELS",
    "STRUCTURAL_ATTRIBUTE_SCHEMA",
    "FrameProjection",
    "FrameProjector",
    "ObserverFrameSpec",
    "PhysicalEventBundle",
    "ProjectedState",
    "ProjectedTransition",
    "ProjectorResult",
    "audit_identity_leaks",
    "canonical_json",
    "canonical_sha256",
    "identity_projector",
    "observed_transition_event_id",
    "observer_frame_spec",
    "outcome_without_state",
    "project_observed_transition",
    "project_state",
    "state_model_payload",
    "validate_unique_event_ids",
]
