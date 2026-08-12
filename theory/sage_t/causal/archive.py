"""Symbolic Go-Explore archive backed by exact-prefix restoration.

The archive deliberately separates the symbolic transposition key from the
pixel-exact replay hash.  Symbolically equal observations share one cell, but
their exact variants are never interchanged during restoration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from theory.sage_t.contracts import AbstractEntity, AbstractState, GroundFact

from .contracts import GroundedAction

ARCHIVE_FORMAT = "sage-t12.1-symbolic-archive-v1"
ROOT_PREFIX_ID = "prefix_root"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _digest(value: Any, *, length: int = 24) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _action_payload(action: GroundedAction) -> dict[str, Any]:
    return {
        "action_name": action.action_name,
        "action_data": dict(action.action_data),
    }


def _action_from_payload(payload: Mapping[str, Any]) -> GroundedAction:
    return GroundedAction(
        str(payload["action_name"]),
        dict(payload.get("action_data", {}) or {}),
    )


def abstract_state_to_payload(state: AbstractState) -> dict[str, Any]:
    return {
        "entities": [
            {
                "entity_id": entity.entity_id,
                "roles": list(entity.roles),
                "attributes": [list(item) for item in entity.attributes],
                "center": None if entity.center is None else list(entity.center),
            }
            for entity in state.entities
        ],
        "true_facts": [
            {"predicate": fact.predicate, "terms": list(fact.terms), "value": fact.value}
            for fact in sorted(state.true_facts)
        ],
        "false_facts": [
            {"predicate": fact.predicate, "terms": list(fact.terms), "value": fact.value}
            for fact in sorted(state.false_facts)
        ],
        "counters": [list(item) for item in state.counters],
        "registers": [list(item) for item in state.registers],
        "topology": [list(item) for item in state.topology],
        "regime_index": state.regime_index,
    }


def abstract_state_from_payload(payload: Mapping[str, Any]) -> AbstractState:
    return AbstractState(
        entities=tuple(
            AbstractEntity(
                entity_id=str(item["entity_id"]),
                roles=tuple(str(value) for value in item.get("roles", ())),
                attributes=tuple(
                    (str(pair[0]), str(pair[1]))
                    for pair in item.get("attributes", ())
                ),
                center=(
                    None
                    if item.get("center") is None
                    else (
                        float(item["center"][0]),
                        float(item["center"][1]),
                    )
                ),
            )
            for item in payload.get("entities", ())
        ),
        true_facts=frozenset(_fact_from_payload(value) for value in payload.get("true_facts", ())),
        false_facts=frozenset(_fact_from_payload(value) for value in payload.get("false_facts", ())),
        counters=tuple(
            (str(item[0]), float(item[1]))
            for item in payload.get("counters", ())
        ),
        registers=tuple(
            (str(item[0]), str(item[1]))
            for item in payload.get("registers", ())
        ),
        topology=tuple(
            (str(item[0]), int(item[1]))
            for item in payload.get("topology", ())
        ),
        regime_index=int(payload.get("regime_index", 0)),
    )


def _fact_from_payload(payload: Any) -> GroundFact:
    if isinstance(payload, Mapping):
        return GroundFact(
            predicate=str(payload["predicate"]),
            terms=tuple(str(value) for value in payload.get("terms", ())),
            value=str(payload.get("value", "")),
        )
    return GroundFact.from_key(str(payload))


@dataclass(frozen=True)
class PrefixNode:
    prefix_id: str
    parent_id: str | None
    action: GroundedAction | None
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_id": self.prefix_id,
            "parent_id": self.parent_id,
            "action": None if self.action is None else _action_payload(self.action),
            "depth": self.depth,
        }


class PrefixStore:
    """Content-addressed trie used to avoid repeating complete prefixes."""

    def __init__(self) -> None:
        self._nodes: dict[str, PrefixNode] = {
            ROOT_PREFIX_ID: PrefixNode(ROOT_PREFIX_ID, None, None, 0)
        }

    def extend(self, parent_id: str, action: GroundedAction) -> str:
        parent = self._nodes.get(str(parent_id))
        if parent is None:
            raise KeyError(f"unknown prefix parent: {parent_id}")
        prefix_id = "px_" + _digest(
            {"parent": parent.prefix_id, "action": _action_payload(action)}
        )
        self._nodes.setdefault(
            prefix_id,
            PrefixNode(prefix_id, parent.prefix_id, action, parent.depth + 1),
        )
        return prefix_id

    def depth(self, prefix_id: str) -> int:
        return self._nodes[str(prefix_id)].depth

    def actions(self, prefix_id: str) -> tuple[GroundedAction, ...]:
        output: list[GroundedAction] = []
        node = self._nodes[str(prefix_id)]
        while node.parent_id is not None:
            if node.action is None:
                raise RuntimeError("non-root prefix node lacks an action")
            output.append(node.action)
            node = self._nodes[node.parent_id]
        output.reverse()
        return tuple(output)

    def to_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._nodes[key].to_dict()
            for key in sorted(self._nodes, key=lambda item: (self._nodes[item].depth, item))
        )

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> PrefixStore:
        store = cls()
        store._nodes = {}
        for row in rows:
            raw_action = row.get("action")
            node = PrefixNode(
                prefix_id=str(row["prefix_id"]),
                parent_id=(
                    None if row.get("parent_id") is None else str(row["parent_id"])
                ),
                action=(
                    None
                    if raw_action is None
                    else _action_from_payload(dict(raw_action))
                ),
                depth=int(row["depth"]),
            )
            store._nodes[node.prefix_id] = node
        if ROOT_PREFIX_ID not in store._nodes:
            raise ValueError("serialized prefix store lacks its root")
        return store


@dataclass(frozen=True)
class ArchiveStateVariant:
    exact_hash: str
    prefix_id: str
    path_edge_ids: tuple[str, ...] = ()
    replay_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_hash": self.exact_hash,
            "prefix_id": self.prefix_id,
            "path_edge_ids": list(self.path_edge_ids),
            "replay_failures": self.replay_failures,
        }


@dataclass
class SymbolicArchiveCell:
    cell_id: str
    symbolic_signature: str
    level: int
    legal_action_keys: tuple[str, ...]
    state: AbstractState
    variants: dict[str, ArchiveStateVariant] = field(default_factory=dict)
    action_attempts: dict[str, int] = field(default_factory=dict)
    visits: int = 0
    expansions: int = 0
    terminal: bool = False
    blocked: bool = False

    def best_variant(self, prefixes: PrefixStore) -> ArchiveStateVariant:
        if not self.variants:
            raise RuntimeError(f"archive cell has no exact variant: {self.cell_id}")
        return min(
            self.variants.values(),
            key=lambda item: (
                item.replay_failures,
                prefixes.depth(item.prefix_id),
                item.exact_hash,
            ),
        )

    @property
    def untried_action_count(self) -> int:
        return sum(self.action_attempts.get(key, 0) == 0 for key in self.legal_action_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "symbolic_signature": self.symbolic_signature,
            "level": self.level,
            "legal_action_keys": list(self.legal_action_keys),
            "state": abstract_state_to_payload(self.state),
            "variants": [
                self.variants[key].to_dict() for key in sorted(self.variants)
            ],
            "action_attempts": dict(sorted(self.action_attempts.items())),
            "visits": self.visits,
            "expansions": self.expansions,
            "terminal": self.terminal,
            "blocked": self.blocked,
        }


@dataclass(frozen=True)
class ArchiveEdge:
    edge_id: str
    ordinal: int
    source_cell_id: str
    source_exact_hash: str
    action: GroundedAction
    target_cell_id: str
    target_exact_hash: str
    level_delta: int
    terminal: bool
    success: bool
    changed: bool
    novel: bool
    prefix_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "ordinal": self.ordinal,
            "source_cell_id": self.source_cell_id,
            "source_exact_hash": self.source_exact_hash,
            "action": _action_payload(self.action),
            "target_cell_id": self.target_cell_id,
            "target_exact_hash": self.target_exact_hash,
            "level_delta": self.level_delta,
            "terminal": self.terminal,
            "success": self.success,
            "changed": self.changed,
            "novel": self.novel,
            "prefix_id": self.prefix_id,
        }


class ActionShield(Protocol):
    def allows(self, cell_id: str, action: GroundedAction) -> bool:
        ...


class ActionNoveltyScorer(Protocol):
    def score(self, state: AbstractState, action: GroundedAction) -> tuple[float, float]:
        ...


class GoExploreArchive:
    """Persistent symbolic state graph with deterministic best-first expansion."""

    def __init__(self, *, maximum_cells: int = 50_000, seed: int = 0) -> None:
        self.maximum_cells = max(1, int(maximum_cells))
        self.seed = int(seed)
        self.prefixes = PrefixStore()
        self.cells: dict[str, SymbolicArchiveCell] = {}
        self.edges: dict[str, ArchiveEdge] = {}
        self._edge_observations = 0
        self.replay_attempts = 0
        self.replay_successes = 0
        self.sdk_calls = 0

    @staticmethod
    def cell_key(
        state: AbstractState,
        *,
        level: int,
        legal_actions: Sequence[GroundedAction],
    ) -> str:
        payload = {
            "symbolic_signature": state.signature,
            "level": int(level),
            "legal_actions": sorted(action.key for action in legal_actions),
        }
        return "cell_" + _digest(payload)

    def observe_state(
        self,
        *,
        state: AbstractState,
        exact_hash: str,
        level: int,
        legal_actions: Sequence[GroundedAction],
        prefix_id: str = ROOT_PREFIX_ID,
        path_edge_ids: Sequence[str] = (),
        terminal: bool = False,
    ) -> tuple[SymbolicArchiveCell, bool]:
        cell_id = self.cell_key(state, level=level, legal_actions=legal_actions)
        is_new = cell_id not in self.cells
        if is_new:
            if len(self.cells) >= self.maximum_cells:
                raise RuntimeError("symbolic archive cell limit reached")
            self.cells[cell_id] = SymbolicArchiveCell(
                cell_id=cell_id,
                symbolic_signature=state.signature,
                level=int(level),
                legal_action_keys=tuple(sorted(action.key for action in legal_actions)),
                state=state,
                action_attempts={action.key: 0 for action in legal_actions},
                terminal=bool(terminal),
            )
        cell = self.cells[cell_id]
        cell.visits += 1
        cell.terminal = bool(cell.terminal or terminal)
        for action in legal_actions:
            cell.action_attempts.setdefault(action.key, 0)
        candidate = ArchiveStateVariant(
            exact_hash=str(exact_hash),
            prefix_id=str(prefix_id),
            path_edge_ids=tuple(str(value) for value in path_edge_ids),
        )
        previous = cell.variants.get(candidate.exact_hash)
        if previous is None or self.prefixes.depth(candidate.prefix_id) < self.prefixes.depth(
            previous.prefix_id
        ):
            cell.variants[candidate.exact_hash] = candidate
        return cell, is_new

    def select_cell(self, *, remaining_sdk_calls: int) -> SymbolicArchiveCell | None:
        eligible = []
        for cell in self.cells.values():
            if cell.terminal or cell.blocked:
                continue
            variant = cell.best_variant(self.prefixes)
            restoration_cost = 1 + self.prefixes.depth(variant.prefix_id) + 1
            if restoration_cost > int(remaining_sdk_calls):
                continue
            eligible.append(cell)
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda cell: (
                -cell.level,
                -int(cell.untried_action_count > 0),
                cell.expansions,
                -self.prefixes.depth(cell.best_variant(self.prefixes).prefix_id),
                self._seeded_key(cell.cell_id),
            ),
        )

    def choose_action(
        self,
        cell: SymbolicArchiveCell,
        candidates: Sequence[GroundedAction],
        *,
        shield: ActionShield | None = None,
        novelty_scorer: ActionNoveltyScorer | None = None,
    ) -> GroundedAction | None:
        allowed = [
            action
            for action in candidates
            if shield is None or shield.allows(cell.cell_id, action)
        ]
        if not allowed:
            return None
        if novelty_scorer is None:
            return min(
                allowed,
                key=lambda action: (
                    cell.action_attempts.get(action.key, 0),
                    self._seeded_key(action.key),
                ),
            )
        return min(
            allowed,
            key=lambda action: (
                cell.action_attempts.get(action.key, 0) > 0,
                -novelty_scorer.score(cell.state, action)[1],
                -novelty_scorer.score(cell.state, action)[0],
                cell.action_attempts.get(action.key, 0),
                self._seeded_key(action.key),
            ),
        )

    def _seeded_key(self, value: str) -> str:
        return hashlib.sha256(f"{self.seed}:{value}".encode("utf-8")).hexdigest()

    def add_transition(
        self,
        *,
        source_cell_id: str,
        source_exact_hash: str,
        action: GroundedAction,
        target_state: AbstractState,
        target_exact_hash: str,
        target_level: int,
        target_legal_actions: Sequence[GroundedAction],
        terminal: bool,
        success: bool,
        changed: bool,
    ) -> ArchiveEdge:
        source = self.cells[source_cell_id]
        source_variant = source.variants[source_exact_hash]
        prefix_id = self.prefixes.extend(source_variant.prefix_id, action)
        target_cell_id = self.cell_key(
            target_state,
            level=target_level,
            legal_actions=target_legal_actions,
        )
        novel = target_cell_id not in self.cells
        ordinal = self._edge_observations
        edge_id = "edge_" + _digest(
            {
                "source": source_cell_id,
                "source_exact": source_exact_hash,
                "action": _action_payload(action),
                "target_exact": target_exact_hash,
                "ordinal": ordinal,
            }
        )
        self._edge_observations += 1
        edge = ArchiveEdge(
            edge_id=edge_id,
            ordinal=ordinal,
            source_cell_id=source_cell_id,
            source_exact_hash=source_exact_hash,
            action=action,
            target_cell_id=target_cell_id,
            target_exact_hash=str(target_exact_hash),
            level_delta=max(0, int(target_level) - int(source.level)),
            terminal=bool(terminal),
            success=bool(success),
            changed=bool(changed),
            novel=bool(novel),
            prefix_id=prefix_id,
        )
        self.edges[edge.edge_id] = edge
        source.expansions += 1
        source.action_attempts[action.key] = source.action_attempts.get(action.key, 0) + 1
        self.observe_state(
            state=target_state,
            exact_hash=target_exact_hash,
            level=target_level,
            legal_actions=target_legal_actions,
            prefix_id=prefix_id,
            path_edge_ids=source_variant.path_edge_ids + (edge.edge_id,),
            terminal=terminal,
        )
        return edge

    def note_replay(self, *, exact: bool) -> None:
        self.replay_attempts += 1
        self.replay_successes += int(bool(exact))

    def path_edges(self, variant: ArchiveStateVariant) -> tuple[ArchiveEdge, ...]:
        return tuple(self.edges[edge_id] for edge_id in variant.path_edge_ids)

    def metrics(self) -> dict[str, Any]:
        exact_variants = sum(len(cell.variants) for cell in self.cells.values())
        maximum_level = max((cell.level for cell in self.cells.values()), default=0)
        progress_edges = sum(edge.level_delta > 0 or edge.success for edge in self.edges.values())
        return {
            "symbolic_cells": len(self.cells),
            "exact_variants": exact_variants,
            "edges": len(self.edges),
            "maximum_level": maximum_level,
            "progress_edges": progress_edges,
            "terminal_edges": sum(edge.terminal and not edge.success for edge in self.edges.values()),
            "replay_attempts": self.replay_attempts,
            "replay_successes": self.replay_successes,
            "replay_exact_rate": (
                1.0
                if self.replay_attempts == 0
                else self.replay_successes / self.replay_attempts
            ),
            "sdk_calls": self.sdk_calls,
            "symbolic_cells_per_1000_sdk_calls": (
                0.0 if self.sdk_calls <= 0 else 1000.0 * len(self.cells) / self.sdk_calls
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": ARCHIVE_FORMAT,
            "maximum_cells": self.maximum_cells,
            "seed": self.seed,
            "prefixes": list(self.prefixes.to_rows()),
            "cells": [self.cells[key].to_dict() for key in sorted(self.cells)],
            "edges": [self.edges[key].to_dict() for key in sorted(self.edges)],
            "edge_observations": self._edge_observations,
            "replay_attempts": self.replay_attempts,
            "replay_successes": self.replay_successes,
            "sdk_calls": self.sdk_calls,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoExploreArchive:
        if payload.get("format_version") != ARCHIVE_FORMAT:
            raise ValueError("unsupported symbolic archive payload")
        archive = cls(
            maximum_cells=int(payload.get("maximum_cells", 50_000)),
            seed=int(payload.get("seed", 0)),
        )
        archive.prefixes = PrefixStore.from_rows(payload.get("prefixes", ()))
        for row in payload.get("cells", ()):
            state = abstract_state_from_payload(dict(row["state"]))
            cell = SymbolicArchiveCell(
                cell_id=str(row["cell_id"]),
                symbolic_signature=str(row["symbolic_signature"]),
                level=int(row["level"]),
                legal_action_keys=tuple(str(value) for value in row.get("legal_action_keys", ())),
                state=state,
                action_attempts={
                    str(key): int(value)
                    for key, value in dict(row.get("action_attempts", {})).items()
                },
                visits=int(row.get("visits", 0)),
                expansions=int(row.get("expansions", 0)),
                terminal=bool(row.get("terminal", False)),
                blocked=bool(row.get("blocked", False)),
            )
            cell.variants = {
                str(item["exact_hash"]): ArchiveStateVariant(
                    exact_hash=str(item["exact_hash"]),
                    prefix_id=str(item["prefix_id"]),
                    path_edge_ids=tuple(str(value) for value in item.get("path_edge_ids", ())),
                    replay_failures=int(item.get("replay_failures", 0)),
                )
                for item in row.get("variants", ())
            }
            archive.cells[cell.cell_id] = cell
        for row in payload.get("edges", ()):
            edge = ArchiveEdge(
                edge_id=str(row["edge_id"]),
                ordinal=int(row.get("ordinal", 0)),
                source_cell_id=str(row["source_cell_id"]),
                source_exact_hash=str(row["source_exact_hash"]),
                action=_action_from_payload(dict(row["action"])),
                target_cell_id=str(row["target_cell_id"]),
                target_exact_hash=str(row["target_exact_hash"]),
                level_delta=int(row.get("level_delta", 0)),
                terminal=bool(row.get("terminal", False)),
                success=bool(row.get("success", False)),
                changed=bool(row.get("changed", False)),
                novel=bool(row.get("novel", False)),
                prefix_id=str(row["prefix_id"]),
            )
            archive.edges[edge.edge_id] = edge
        archive._edge_observations = int(payload.get("edge_observations", len(archive.edges)))
        archive.replay_attempts = int(payload.get("replay_attempts", 0))
        archive.replay_successes = int(payload.get("replay_successes", 0))
        archive.sdk_calls = int(payload.get("sdk_calls", 0))
        return archive


__all__ = [
    "ARCHIVE_FORMAT",
    "ROOT_PREFIX_ID",
    "ArchiveEdge",
    "ArchiveStateVariant",
    "GoExploreArchive",
    "PrefixNode",
    "PrefixStore",
    "SymbolicArchiveCell",
    "abstract_state_from_payload",
    "abstract_state_to_payload",
]
