"""Typed, non-authoritative semantic hypotheses for SAGE12."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence, Tuple


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ALLOWED_PREDICATES = frozenset(
    {
        "adjacent",
        "aligned",
        "contact",
        "near",
        "north_of",
        "south_of",
        "east_of",
        "west_of",
        "exists",
        "removed",
        "moved",
        "changed",
        "progress",
        "level_complete",
        "game_over",
    }
)
_ALLOWED_OPERATIONS = frozenset({"assert", "retract"})
ALLOWED_PREDICATES = tuple(sorted(_ALLOWED_PREDICATES))


def _identifier(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a safe snake_case identifier")
    return normalized


@dataclass(frozen=True)
class EntityRef:
    """A structural role to ground against the current scene graph."""

    role: str
    selector: str = "any"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identifier(self.role, field_name="role"))
        object.__setattr__(
            self,
            "selector",
            _identifier(self.selector, field_name="selector"),
        )


@dataclass(frozen=True)
class SemanticPredicate:
    """A bounded relation or state predicate over structural entity roles."""

    name: str
    subject: EntityRef | None = None
    object: EntityRef | None = None
    value: str = ""

    def __post_init__(self) -> None:
        normalized = _identifier(self.name, field_name="predicate name")
        if normalized not in _ALLOWED_PREDICATES:
            raise ValueError(f"unsupported semantic predicate: {normalized}")
        object.__setattr__(self, "name", normalized)
        if len(str(self.value)) > 64:
            raise ValueError("predicate value is too long")

    @property
    def roles(self) -> Tuple[EntityRef, ...]:
        return tuple(
            ref for ref in (self.subject, self.object) if ref is not None
        )


@dataclass(frozen=True)
class SemanticEffect:
    """A predicted predicate edit. It is not evidence."""

    predicate: SemanticPredicate
    operation: str = "assert"

    def __post_init__(self) -> None:
        normalized = _identifier(self.operation, field_name="effect operation")
        if normalized not in _ALLOWED_OPERATIONS:
            raise ValueError(f"unsupported semantic operation: {normalized}")
        object.__setattr__(self, "operation", normalized)


@dataclass(frozen=True)
class SemanticHypothesis:
    """A proposal produced by an LLM or deterministic hypothesis source."""

    hypothesis_id: str
    action_name: str
    action_data: Mapping[str, Any] = field(default_factory=dict)
    preconditions: Tuple[SemanticPredicate, ...] = ()
    effects: Tuple[SemanticEffect, ...] = ()
    confidence: float = 0.0
    rationale: str = ""
    source: str = "local_llm"
    support: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hypothesis_id",
            _identifier(self.hypothesis_id, field_name="hypothesis_id"),
        )
        action_name = str(self.action_name).strip().upper()
        if not action_name or len(action_name) > 32:
            raise ValueError("action_name must be a bounded non-empty string")
        object.__setattr__(self, "action_name", action_name)
        object.__setattr__(self, "action_data", dict(self.action_data))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "source", str(self.source)[:32])
        object.__setattr__(self, "rationale", str(self.rationale)[:512])
        if self.support != 0:
            raise ValueError("proposed hypotheses must enter with support=0")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.effects:
            raise ValueError("a semantic hypothesis needs at least one effect")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SemanticHypothesis":
        def predicate(raw: Mapping[str, Any]) -> SemanticPredicate:
            subject = raw.get("subject")
            target = raw.get("object")
            return SemanticPredicate(
                name=str(raw["name"]),
                subject=(
                    EntityRef(**dict(subject))
                    if isinstance(subject, Mapping)
                    else None
                ),
                object=(
                    EntityRef(**dict(target))
                    if isinstance(target, Mapping)
                    else None
                ),
                value=str(raw.get("value", "")),
            )

        return cls(
            hypothesis_id=str(payload["hypothesis_id"]),
            action_name=str(payload["action_name"]),
            action_data=dict(payload.get("action_data", {})),
            preconditions=tuple(
                predicate(item)
                for item in payload.get("preconditions", ())
                if isinstance(item, Mapping)
            ),
            effects=tuple(
                SemanticEffect(
                    predicate=predicate(item["predicate"]),
                    operation=str(item.get("operation", "assert")),
                )
                for item in payload.get("effects", ())
                if isinstance(item, Mapping)
                and isinstance(item.get("predicate"), Mapping)
            ),
            confidence=float(payload.get("confidence", 0.0)),
            rationale=str(payload.get("rationale", "")),
            source=str(payload.get("source", "local_llm")),
            support=int(payload.get("support", 0)),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)


def hypotheses_from_json(
    raw: str,
    *,
    maximum: int,
) -> Tuple[SemanticHypothesis, ...]:
    """Parse a bounded JSON response and reject prose or malformed entries."""
    payload = json.loads(raw)
    items: Sequence[Any]
    if isinstance(payload, Mapping):
        items = payload.get("hypotheses", ())
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        items = payload
    else:
        raise ValueError("hypothesis response must be a JSON array or object")
    parsed = []
    for item in items[: max(0, int(maximum))]:
        if not isinstance(item, Mapping):
            continue
        parsed.append(SemanticHypothesis.from_mapping(item))
    return tuple(parsed)


def predicate_key(
    predicate: SemanticPredicate,
    bindings: Mapping[str, str] | None = None,
) -> str:
    """Return a stable grounded predicate key."""
    bindings = bindings or {}

    def entity(ref: EntityRef | None) -> str:
        if ref is None:
            return "-"
        return bindings.get(ref.role, f"role:{ref.role}:{ref.selector}")

    return "|".join(
        (
            predicate.name,
            entity(predicate.subject),
            entity(predicate.object),
            str(predicate.value),
        )
    )


__all__ = [
    "ALLOWED_PREDICATES",
    "EntityRef",
    "SemanticEffect",
    "SemanticHypothesis",
    "SemanticPredicate",
    "hypotheses_from_json",
    "predicate_key",
]
