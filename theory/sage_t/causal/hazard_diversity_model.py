"""Identity-free terminal-risk abstraction and action-family diversity.

The T12.4a.4d archives showed two distinct failure modes: the treatment
selected one action family exclusively, and the exact-cell terminal shield
did not recognize any target-local hazard.  This module addresses those
mechanisms without adding a neural model or changing the frozen parent run.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from theory.sage_t.contracts import AbstractState

from .archive import SymbolicArchiveCell
from .contracts import GroundedAction
from .shield_model import ProgressProtectedTerminalShield

ABSTRACT_HAZARD_MODEL_FORMAT = "sage-t12.4a.4d.1-abstract-hazard-model-v1"

_UNINFORMATIVE_ROLES = frozenset({"object", "target", "unknown"})


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def local_hazard_descriptor(
    state: AbstractState,
    action: GroundedAction,
    *,
    radius: int = 7,
) -> dict[str, Any]:
    """Return a translation-invariant local action context.

    ARC click coordinates use ``x=column`` and ``y=row`` while component
    centers are stored as ``(row, column)``.  Only relative integer offsets,
    typed attributes and informative roles are retained.  Absolute positions,
    entity ids, game ids and exact state hashes are deliberately excluded.
    """

    local_entities: list[dict[str, Any]] = []
    data = dict(action.action_data)
    coordinate_grounded = "x" in data and "y" in data
    if coordinate_grounded:
        try:
            x = float(data["x"])
            y = float(data["y"])
        except (TypeError, ValueError):
            coordinate_grounded = False
        else:
            for entity in state.entities:
                if entity.center is None:
                    continue
                delta_row = int(round(float(entity.center[0]) - y))
                delta_column = int(round(float(entity.center[1]) - x))
                if abs(delta_row) > int(radius) or abs(delta_column) > int(radius):
                    continue
                local_entities.append(
                    {
                        "attributes": [list(item) for item in entity.attributes],
                        "delta_column": delta_column,
                        "delta_row": delta_row,
                        "roles": sorted(set(entity.roles) - _UNINFORMATIVE_ROLES),
                    }
                )
    local_entities.sort(key=_canonical)
    return {
        "action_name": action.action_name,
        "coordinate_grounded": coordinate_grounded,
        "local_entities": local_entities,
        "radius": int(radius),
    }


def local_hazard_signature(
    state: AbstractState,
    action: GroundedAction,
    *,
    radius: int = 7,
) -> str:
    return "hazard_" + _checksum(
        local_hazard_descriptor(state, action, radius=radius)
    )[:24]


@dataclass(frozen=True)
class HazardObservation:
    search_seed: int
    lineage_seed: int
    source_exact_hash: str
    state: AbstractState
    action: GroundedAction
    terminal: bool

    @property
    def observation_key(self) -> str:
        return _checksum(
            {
                "action": self.action.key,
                "lineage_seed": self.lineage_seed,
                "search_seed": self.search_seed,
                "source_exact_hash": self.source_exact_hash,
            }
        )


@dataclass(frozen=True)
class HazardSupport:
    signature: str
    descriptor: Mapping[str, Any]
    observations: int
    terminal_observations: int
    search_seeds: tuple[int, ...]
    lineage_seeds: tuple[int, ...]

    @property
    def terminal_rate(self) -> float:
        return self.terminal_observations / max(1, self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": dict(self.descriptor),
            "lineage_seeds": list(self.lineage_seeds),
            "observations": self.observations,
            "search_seeds": list(self.search_seeds),
            "signature": self.signature,
            "terminal_observations": self.terminal_observations,
            "terminal_rate": self.terminal_rate,
        }


class AbstractHazardModel:
    """Frozen lookup over target-local, identity-free hazard signatures."""

    def __init__(
        self,
        *,
        radius: int,
        minimum_support: int,
        unsafe_rate_threshold: float,
        support: Mapping[str, HazardSupport],
    ) -> None:
        self.radius = int(radius)
        self.minimum_support = int(minimum_support)
        self.unsafe_rate_threshold = float(unsafe_rate_threshold)
        self.support = dict(support)

    @classmethod
    def fit(
        cls,
        observations: Sequence[HazardObservation],
        *,
        radius: int,
        minimum_support: int,
        unsafe_rate_threshold: float,
    ) -> AbstractHazardModel:
        deduplicated: dict[str, HazardObservation] = {}
        for item in observations:
            previous = deduplicated.get(item.observation_key)
            if previous is not None and previous.terminal != item.terminal:
                raise ValueError("conflicting terminal labels for one intervention")
            deduplicated[item.observation_key] = item
        grouped: dict[str, dict[str, Any]] = {}
        for item in deduplicated.values():
            descriptor = local_hazard_descriptor(
                item.state,
                item.action,
                radius=radius,
            )
            signature = "hazard_" + _checksum(descriptor)[:24]
            row = grouped.setdefault(
                signature,
                {
                    "descriptor": descriptor,
                    "lineages": set(),
                    "observations": 0,
                    "seeds": set(),
                    "terminal": 0,
                },
            )
            row["observations"] += 1
            row["terminal"] += int(item.terminal)
            row["seeds"].add(int(item.search_seed))
            row["lineages"].add(int(item.lineage_seed))
        support = {
            signature: HazardSupport(
                signature=signature,
                descriptor=dict(row["descriptor"]),
                observations=int(row["observations"]),
                terminal_observations=int(row["terminal"]),
                search_seeds=tuple(sorted(row["seeds"])),
                lineage_seeds=tuple(sorted(row["lineages"])),
            )
            for signature, row in grouped.items()
        }
        return cls(
            radius=radius,
            minimum_support=minimum_support,
            unsafe_rate_threshold=unsafe_rate_threshold,
            support=support,
        )

    @property
    def unsafe_signatures(self) -> frozenset[str]:
        return frozenset(
            signature
            for signature, item in self.support.items()
            if item.observations >= self.minimum_support
            and item.terminal_rate >= self.unsafe_rate_threshold
        )

    def risk(self, state: AbstractState, action: GroundedAction) -> float:
        item = self.support.get(
            local_hazard_signature(state, action, radius=self.radius)
        )
        return 0.0 if item is None else item.terminal_rate

    def is_unsafe(self, state: AbstractState, action: GroundedAction) -> bool:
        return (
            local_hazard_signature(state, action, radius=self.radius)
            in self.unsafe_signatures
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format_version": ABSTRACT_HAZARD_MODEL_FORMAT,
            "minimum_support": self.minimum_support,
            "radius": self.radius,
            "support": [self.support[key].to_dict() for key in sorted(self.support)],
            "unsafe_rate_threshold": self.unsafe_rate_threshold,
            "unsafe_signatures": sorted(self.unsafe_signatures),
        }
        return {**payload, "model_checksum": _checksum(payload)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AbstractHazardModel:
        if payload.get("format_version") != ABSTRACT_HAZARD_MODEL_FORMAT:
            raise ValueError("unsupported abstract hazard model")
        unsigned = dict(payload)
        checksum = str(unsigned.pop("model_checksum"))
        if _checksum(unsigned) != checksum:
            raise ValueError("abstract hazard model checksum mismatch")
        support = {
            str(row["signature"]): HazardSupport(
                signature=str(row["signature"]),
                descriptor=dict(row["descriptor"]),
                observations=int(row["observations"]),
                terminal_observations=int(row["terminal_observations"]),
                search_seeds=tuple(int(value) for value in row["search_seeds"]),
                lineage_seeds=tuple(int(value) for value in row["lineage_seeds"]),
            )
            for row in payload.get("support", ())
        }
        model = cls(
            radius=int(payload["radius"]),
            minimum_support=int(payload["minimum_support"]),
            unsafe_rate_threshold=float(payload["unsafe_rate_threshold"]),
            support=support,
        )
        if sorted(model.unsafe_signatures) != sorted(payload["unsafe_signatures"]):
            raise ValueError("abstract hazard unsafe registry mismatch")
        return model


class StructuralActionDiversityPolicy:
    """Balance action schemas before ranking their grounded instances."""

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.family_selections: Counter[str] = Counter()
        self.static_shield_vetoes = 0
        self.abstract_hazard_vetoes = 0

    def _seeded_key(self, value: str) -> str:
        return hashlib.sha256(f"{self.seed}:{value}".encode("utf-8")).hexdigest()

    def choose(
        self,
        cell: SymbolicArchiveCell,
        candidates: Sequence[GroundedAction],
        *,
        static_shield: ProgressProtectedTerminalShield,
        hazard_model: AbstractHazardModel | None,
        novelty_scorer: Any | None,
    ) -> GroundedAction | None:
        allowed = []
        for action in candidates:
            if not static_shield.allows(cell.cell_id, action):
                self.static_shield_vetoes += 1
                continue
            if (
                hazard_model is not None
                and not static_shield.is_protected(cell.cell_id, action)
                and hazard_model.is_unsafe(cell.state, action)
            ):
                self.abstract_hazard_vetoes += 1
                continue
            allowed.append(action)
        if not allowed:
            return None
        by_family: dict[str, list[GroundedAction]] = {}
        for action in allowed:
            by_family.setdefault(action.action_name, []).append(action)
        family = min(
            by_family,
            key=lambda name: (
                sum(
                    cell.action_attempts.get(action.key, 0)
                    for action in by_family[name]
                ),
                self.family_selections[name],
                self._seeded_key(name),
            ),
        )
        selected = min(
            by_family[family],
            key=lambda action: (
                cell.action_attempts.get(action.key, 0),
                -(
                    novelty_scorer.score(cell.state, action)[1]
                    if novelty_scorer is not None
                    else 0.0
                ),
                -(
                    novelty_scorer.score(cell.state, action)[0]
                    if novelty_scorer is not None
                    else 0.0
                ),
                self._seeded_key(action.key),
            ),
        )
        self.family_selections[family] += 1
        return selected

    def metrics(self) -> dict[str, Any]:
        total = sum(self.family_selections.values())
        return {
            "action_family_counts": dict(sorted(self.family_selections.items())),
            "maximum_action_family_share": (
                0.0
                if total == 0
                else max(self.family_selections.values(), default=0) / total
            ),
            "abstract_hazard_vetoes": self.abstract_hazard_vetoes,
            "selected_actions": total,
            "static_shield_vetoes": self.static_shield_vetoes,
        }


__all__ = [
    "ABSTRACT_HAZARD_MODEL_FORMAT",
    "AbstractHazardModel",
    "HazardObservation",
    "HazardSupport",
    "StructuralActionDiversityPolicy",
    "local_hazard_descriptor",
    "local_hazard_signature",
]
