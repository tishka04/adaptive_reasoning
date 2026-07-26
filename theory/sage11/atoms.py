"""Typed symbolic atoms shared by FrameDiff, schemas, and neural hypotheses."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple


ATOM_VOCAB_VERSION = "sage11-atoms-v1"
ATOM_KINDS: Tuple[str, ...] = (
    "state",
    "object",
    "relation",
    "action",
    "effect",
    "progress",
    "risk",
    "subgoal",
)


@dataclass(frozen=True, order=True)
class TypedAtom:
    """One falsifiable symbolic proposition with zero initial support."""

    kind: str
    predicate: str
    arguments: Tuple[str, ...] = ()
    support: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ATOM_KINDS:
            raise ValueError(f"unknown SAGE.11 atom kind {self.kind}")
        if self.support != 0:
            raise ValueError(
                "neural candidate atoms must be created with support=0"
            )

    @property
    def key(self) -> str:
        args = ",".join(self.arguments)
        return f"{self.kind}:{self.predicate}({args})"

    @classmethod
    def parse(cls, value: str) -> "TypedAtom":
        kind, remainder = str(value).split(":", 1)
        predicate, raw_arguments = remainder.rsplit("(", 1)
        arguments = raw_arguments[:-1]
        return cls(
            kind=kind,
            predicate=predicate,
            arguments=tuple(
                item for item in arguments.split(",") if item
            ),
        )


class HashAtomVocabulary:
    """Versioned hashing vocabulary with stable ids and no fitted holdout state."""

    def __init__(self, size: int = 4096) -> None:
        self.size = max(64, int(size))

    def id_for(self, atom: TypedAtom | str) -> int:
        key = atom.key if isinstance(atom, TypedAtom) else str(atom)
        digest = hashlib.sha256(
            f"{ATOM_VOCAB_VERSION}|{key}".encode("utf-8")
        ).digest()
        return 1 + int.from_bytes(digest[:8], "big") % (self.size - 1)

    def encode(
        self,
        atoms: Sequence[TypedAtom | str],
        *,
        limit: int = 256,
    ) -> Tuple[int, ...]:
        return tuple(
            self.id_for(atom)
            for atom in tuple(atoms)[: max(1, int(limit))]
        )


def observation_atoms(observation: Any) -> Tuple[TypedAtom, ...]:
    """Abstract a GameObservation without coordinates or game identity."""
    atoms = {
        TypedAtom(
            "state",
            "game_state",
            (str(getattr(observation, "game_state", "UNKNOWN")).lower(),),
        ),
        TypedAtom(
            "progress",
            "levels_completed_bucket",
            (_count_bucket(getattr(observation, "levels_completed", 0)),),
        ),
    }
    for action in tuple(getattr(observation, "available_actions", ()) or ()):
        atoms.add(TypedAtom("action", "available", (str(action),)))
    for obj in tuple(getattr(observation, "objects", ()) or ()):
        atoms.add(TypedAtom(
            "object",
            "role_present",
            (_object_role(obj),),
        ))
    return tuple(sorted(atoms))


def frame_diff_atoms(diff: Any) -> Tuple[TypedAtom, ...]:
    """Translate FrameDiff fields into the shared typed vocabulary."""
    changed = int(getattr(diff, "num_changed", 0) or 0)
    atoms = {
        TypedAtom("effect", "changed_cells", (_count_bucket(changed),)),
        TypedAtom(
            "effect",
            "player_moved",
            (str(getattr(diff, "player_displacement", None) is not None),),
        ),
        TypedAtom(
            "progress",
            "level_complete",
            (str(bool(getattr(diff, "level_complete", False))),),
        ),
        TypedAtom(
            "risk",
            "game_over",
            (str(bool(getattr(diff, "game_over", False))),),
        ),
    }
    before = tuple(getattr(diff, "changed_values_before", ()) or ())
    after = tuple(getattr(diff, "changed_values_after", ()) or ())
    if before or after:
        atoms.add(TypedAtom(
            "effect",
            "value_multiset_delta",
            (
                _count_bucket(len(before)),
                _count_bucket(len(after)),
            ),
        ))
    return tuple(sorted(atoms))


def schema_effect_atoms(
    effects: Iterable[Any],
) -> Tuple[TypedAtom, ...]:
    """Map causal-schema predicates to the same effect atom namespace."""
    atoms = []
    for effect in effects:
        atoms.append(TypedAtom(
            "effect",
            str(getattr(effect, "predicate", "unknown")),
            (
                str(getattr(effect, "family", "")),
                str(getattr(effect, "direction", "")),
            ),
        ))
    return tuple(sorted(set(atoms)))


def action_features(
    action_name: str,
    action_data: Mapping[str, Any] | None,
) -> Tuple[float, ...]:
    """Small fixed action vector shared by pilot and graph model."""
    data = dict(action_data or {})
    action_index = _action_index(action_name)
    return (
        action_index / 6.0,
        float("x" in data and "y" in data),
        _bounded_number(data.get("x", 0)),
        _bounded_number(data.get("y", 0)),
        float(bool(data)),
        1.0,
    )


def _action_index(action_name: str) -> int:
    raw = str(action_name).upper().replace("ACTION", "")
    try:
        return max(0, min(6, int(raw)))
    except ValueError:
        return 0


def _bounded_number(value: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(value) / 64.0))
    except (TypeError, ValueError):
        return 0.0


def _object_role(obj: Any) -> str:
    area = int(getattr(obj, "area", 1) or 1)
    bbox = tuple(getattr(obj, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))
    height = int(bbox[2]) - int(bbox[0]) + 1
    width = int(bbox[3]) - int(bbox[1]) + 1
    size = _count_bucket(area)
    aspect = "square" if height == width else "vertical" if height > width else "horizontal"
    return f"{size}:{aspect}"


def _count_bucket(value: Any) -> str:
    count = int(value or 0)
    if count <= 0:
        return "zero"
    if count == 1:
        return "one"
    if count <= 4:
        return "few"
    if count <= 15:
        return "some"
    return "many"


__all__ = [
    "ATOM_KINDS",
    "ATOM_VOCAB_VERSION",
    "HashAtomVocabulary",
    "TypedAtom",
    "action_features",
    "frame_diff_atoms",
    "observation_atoms",
    "schema_effect_atoms",
]
