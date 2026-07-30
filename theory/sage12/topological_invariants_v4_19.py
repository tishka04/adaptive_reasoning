"""Canonical topological invariants and causal deltas for SAGE12 V4.19.

The learned view is deliberately identity-free. Local node identifiers,
palette values, absolute coordinates, game identifiers, and seeds never enter
the feature vector. Raw cells remain in-memory only long enough to establish
before/after correspondence through the audited V4.16 compiler.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mt.graph import MorphoTopologicalGraph, MTNode
from .mt.transition import MTTransitionRecord

FORMAT_VERSION = "sage12-topological-invariants-v4.19"
FEATURE_WIDTH = 512
CONFIDENT_CORRESPONDENCE = 0.60
DISTANCE_SENTINEL = 16

FACTOR_NAMES = (
    "birth",
    "death",
    "merge",
    "split",
    "relative_motion",
    "morphology_changed",
    "contact_added",
    "contact_removed",
    "free_region_increased",
    "free_region_decreased",
    "articulation_added",
    "articulation_removed",
    "bridge_added",
    "bridge_removed",
    "reachable_increased",
    "reachable_decreased",
    "root_distance_decreased",
    "root_distance_increased",
    "terminal_progress",
    "risk",
)

RELATION_INVARIANTS = frozenset(
    {
        "structural_edges",
        "connected_components",
        "cycle_rank",
        "articulation_points",
        "bridges",
        "actor_component_size",
        "action_root_component_size",
        "actor_root_distance",
        "root_is_articulation",
        "root_bridge_incidence",
        "reachable_free_regions",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _node_descriptor(
    node: MTNode,
    *,
    include_binding: bool = True,
    roles: Sequence[str] | None = None,
) -> tuple[str, ...]:
    selected_roles = tuple(sorted(roles if roles is not None else node.roles))
    if not include_binding:
        selected_roles = tuple(
            role for role in selected_roles if role != "action_root"
        )
    return (
        f"kind={node.kind}",
        f"roles={','.join(selected_roles) or 'none'}",
        f"area={node.area_bucket}",
        f"aspect={node.aspect_bucket}",
        f"compactness={node.compactness_bucket}",
        f"holes={min(max(int(node.holes), 0), 4)}",
        f"boundary={min(max(int(node.boundary_contacts), 0), 4)}",
        f"action_relation={node.action_relation if include_binding else 'masked'}",
    )


def _structural_adjacency(
    graph: MorphoTopologicalGraph,
) -> dict[str, set[str]]:
    adjacency = {node.node_id: set() for node in graph.nodes}
    for relation in graph.relations:
        if relation.kind not in {"contact", "encloses"}:
            continue
        if relation.subject_id == relation.object_id:
            continue
        if (
            relation.subject_id not in adjacency
            or relation.object_id not in adjacency
        ):
            continue
        adjacency[relation.subject_id].add(relation.object_id)
        adjacency[relation.object_id].add(relation.subject_id)
    return adjacency


def _components(adjacency: Mapping[str, set[str]]) -> list[set[str]]:
    remaining = set(adjacency)
    output: list[set[str]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        output.append(component)
    return output


def _articulation_and_bridges(
    adjacency: Mapping[str, set[str]],
) -> tuple[set[str], set[tuple[str, str]]]:
    """Return Tarjan articulation points and undirected bridges."""

    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    articulations: set[str] = set()
    bridges: set[tuple[str, str]] = set()
    clock = 0

    def visit(node: str) -> None:
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        children = 0
        for neighbor in sorted(adjacency[node]):
            if neighbor not in discovery:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent[node] is None and children > 1:
                    articulations.add(node)
                if parent[node] is not None and low[neighbor] >= discovery[node]:
                    articulations.add(node)
                if low[neighbor] > discovery[node]:
                    bridges.add(tuple(sorted((node, neighbor))))
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(adjacency):
        if node in discovery:
            continue
        parent[node] = None
        visit(node)
    return articulations, bridges


def _shortest_distance(
    adjacency: Mapping[str, set[str]],
    sources: Sequence[str],
    targets: Sequence[str],
) -> int:
    target_set = set(targets)
    if not sources or not target_set:
        return DISTANCE_SENTINEL
    queue = deque((source, 0) for source in sorted(set(sources)))
    seen = set(sources)
    while queue:
        node, distance = queue.popleft()
        if node in target_set:
            return min(distance, DISTANCE_SENTINEL)
        for neighbor in sorted(adjacency[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return DISTANCE_SENTINEL


def topological_invariants(
    graph: MorphoTopologicalGraph,
) -> dict[str, int]:
    """Calculate explicit, permutation-invariant graph topology."""

    adjacency = _structural_adjacency(graph)
    components = _components(adjacency)
    articulations, bridges = _articulation_and_bridges(adjacency)
    by_id = {node.node_id: node for node in graph.nodes}
    actor = [
        node.node_id for node in graph.nodes if "actor" in node.roles
    ]
    roots = [
        node.node_id for node in graph.nodes if "action_root" in node.roles
    ]
    actor_component = next(
        (component for component in components if set(actor) & component),
        set(),
    )
    root_component = next(
        (component for component in components if set(roots) & component),
        set(),
    )
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return {
        **{
            str(key): int(value)
            for key, value in sorted(graph.invariants.items())
        },
        "node_count": len(graph.nodes),
        "object_count": sum(node.kind == "object" for node in graph.nodes),
        "free_region_count": sum(
            node.kind == "free_region" for node in graph.nodes
        ),
        "structural_edges": edge_count,
        "connected_components": len(components),
        "cycle_rank": max(0, edge_count - len(graph.nodes) + len(components)),
        "articulation_points": len(articulations),
        "bridges": len(bridges),
        "actor_component_size": len(actor_component),
        "action_root_component_size": len(root_component),
        "actor_root_distance": _shortest_distance(
            adjacency,
            actor,
            roots,
        ),
        "root_is_articulation": int(bool(set(roots) & articulations)),
        "root_bridge_incidence": sum(
            int(left in roots or right in roots) for left, right in bridges
        ),
        "reachable_free_regions": sum(
            by_id[node_id].kind == "free_region"
            for node_id in actor_component
        ),
    }


def invariant_deltas(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def correspondence_quality(
    transition: MTTransitionRecord,
) -> dict[str, float | int | bool]:
    structural = [
        row
        for row in transition.correspondences
        if row.kind not in {"birth", "death"}
    ]
    confidences = [float(row.confidence) for row in structural]
    confident = [
        value >= CONFIDENT_CORRESPONDENCE for value in confidences
    ]
    return {
        "correspondences": len(transition.correspondences),
        "structural_correspondences": len(structural),
        "mean_confidence": float(np.mean(confidences)) if confidences else 1.0,
        "confident_fraction": (
            float(np.mean(confident)) if confident else 1.0
        ),
        "fully_ambiguous": bool(structural and not any(confident)),
    }


def _event_count(events: Sequence[str], prefix: str) -> int:
    total = 0
    for event in events:
        name, _, raw_count = event.partition("#")
        if name == prefix or name.startswith(prefix):
            total += int(raw_count or 1)
    return total


def causal_factors(
    transition: MTTransitionRecord,
    *,
    terminal_progress: bool,
    risk: bool,
) -> dict[str, bool]:
    """Factor an observed transition into explicit causal graph changes."""

    before = topological_invariants(transition.graph_before)
    after = topological_invariants(transition.graph_after)
    delta = invariant_deltas(before, after)
    events = transition.events
    return {
        "birth": _event_count(events, "birth") > 0,
        "death": _event_count(events, "death") > 0,
        "merge": _event_count(events, "merge") > 0,
        "split": _event_count(events, "split") > 0,
        "relative_motion": _event_count(events, "relative_motion") > 0,
        "morphology_changed": (
            _event_count(events, "morphology_changed") > 0
        ),
        "contact_added": _event_count(events, "relation_added:contact") > 0,
        "contact_removed": (
            _event_count(events, "relation_removed:contact") > 0
        ),
        "free_region_increased": delta.get("free_region_count", 0) > 0,
        "free_region_decreased": delta.get("free_region_count", 0) < 0,
        "articulation_added": delta.get("articulation_points", 0) > 0,
        "articulation_removed": delta.get("articulation_points", 0) < 0,
        "bridge_added": delta.get("bridges", 0) > 0,
        "bridge_removed": delta.get("bridges", 0) < 0,
        "reachable_increased": delta.get("reachable_free_regions", 0) > 0,
        "reachable_decreased": delta.get("reachable_free_regions", 0) < 0,
        "root_distance_decreased": (
            after["actor_root_distance"] < before["actor_root_distance"]
        ),
        "root_distance_increased": (
            after["actor_root_distance"] > before["actor_root_distance"]
        ),
        "terminal_progress": bool(terminal_progress),
        "risk": bool(risk),
    }


def local_topological_value(factors: Mapping[str, bool]) -> float:
    value = (
        0.08 * float(factors["relative_motion"])
        + 0.10 * float(factors["morphology_changed"])
        + 0.12 * float(factors["birth"] or factors["death"])
        + 0.18 * float(factors["merge"] or factors["split"])
        + 0.18 * float(factors["contact_added"])
        + 0.30 * float(factors["reachable_increased"])
        + 0.25 * float(factors["articulation_removed"])
        + 0.20 * float(factors["bridge_removed"])
        + 0.25 * float(factors["root_distance_decreased"])
        + 1.00 * float(factors["terminal_progress"])
        - 0.30 * float(factors["reachable_decreased"])
        - 0.25 * float(factors["root_distance_increased"])
        - 1.00 * float(factors["risk"])
    )
    return float(np.clip(value, -1.0, 1.0))


def _bucket_integer(value: int) -> str:
    if value < 0:
        return "negative"
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 3:
        return "few"
    if value <= 8:
        return "several"
    return "many"


def _binding_roles(
    graph: MorphoTopologicalGraph,
    *,
    swap_binding: bool,
) -> dict[str, tuple[str, ...]]:
    original = {
        node.node_id: tuple(node.roles)
        for node in graph.nodes
    }
    if not swap_binding:
        return original
    ordered = sorted(
        graph.nodes,
        key=lambda node: _node_descriptor(node, include_binding=False),
    )
    roots = [node for node in ordered if "action_root" in node.roles]
    alternatives = [node for node in ordered if "action_root" not in node.roles]
    if not roots or not alternatives:
        return original
    replacement = alternatives[0].node_id
    swapped = {}
    for node in graph.nodes:
        roles = set(node.roles)
        roles.discard("action_root")
        if node.node_id == replacement:
            roles.add("action_root")
        swapped[node.node_id] = tuple(sorted(roles))
    return swapped


def graph_tokens(
    graph: MorphoTopologicalGraph,
    *,
    remove_relations: bool = False,
    swap_binding: bool = False,
    static_only: bool = False,
) -> tuple[str, ...]:
    """Return canonical tokens for the authorized V4.19 student view."""

    invariants = topological_invariants(graph)
    tokens = [f"action_family:{graph.action_family}"]
    for key, value in sorted(invariants.items()):
        if remove_relations and key in RELATION_INVARIANTS:
            continue
        tokens.append(f"invariant:{key}:{_bucket_integer(int(value))}")
    if static_only:
        return tuple(sorted(tokens))
    roles = _binding_roles(graph, swap_binding=swap_binding)
    by_id = {node.node_id: node for node in graph.nodes}
    for node in graph.nodes:
        descriptor = _node_descriptor(
            node,
            include_binding=not remove_relations,
            roles=roles[node.node_id],
        )
        tokens.extend(f"node:{item}" for item in descriptor)
        tokens.append("node_joint:" + "|".join(descriptor))
    if not remove_relations:
        for relation in graph.relations:
            if relation.subject_id not in by_id or relation.object_id not in by_id:
                continue
            subject = _node_descriptor(
                by_id[relation.subject_id],
                roles=roles[relation.subject_id],
            )
            obj = _node_descriptor(
                by_id[relation.object_id],
                roles=roles[relation.object_id],
            )
            endpoints = sorted(
                (
                    "|".join(subject[:4]),
                    "|".join(obj[:4]),
                )
            )
            tokens.append(
                f"edge:{relation.kind}:{endpoints[0]}->{endpoints[1]}"
            )
    return tuple(sorted(tokens))


def feature_vector(
    graph: MorphoTopologicalGraph,
    *,
    remove_relations: bool = False,
    swap_binding: bool = False,
    static_only: bool = False,
    width: int = FEATURE_WIDTH,
) -> np.ndarray:
    counts = Counter(
        graph_tokens(
            graph,
            remove_relations=remove_relations,
            swap_binding=swap_binding,
            static_only=static_only,
        )
    )
    vector = np.zeros(width, dtype=np.float32)
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % width
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * math.sqrt(float(count))
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


@dataclass(frozen=True)
class CompiledTopologicalTransition:
    factors: Mapping[str, bool]
    local_value: float
    before_invariants: Mapping[str, int]
    after_invariants: Mapping[str, int]
    deltas: Mapping[str, int]
    correspondence: Mapping[str, float | int | bool]


def compile_topological_transition(
    transition: MTTransitionRecord,
    *,
    terminal_progress: bool,
    risk: bool,
) -> CompiledTopologicalTransition:
    before = topological_invariants(transition.graph_before)
    after = topological_invariants(transition.graph_after)
    factors = causal_factors(
        transition,
        terminal_progress=terminal_progress,
        risk=risk,
    )
    return CompiledTopologicalTransition(
        factors=factors,
        local_value=local_topological_value(factors),
        before_invariants=before,
        after_invariants=after,
        deltas=invariant_deltas(before, after),
        correspondence=correspondence_quality(transition),
    )


def sparse_vector(vector: np.ndarray) -> list[list[float]]:
    return [
        [int(index), float(vector[index])]
        for index in np.flatnonzero(vector)
    ]


def dense_vector(
    sparse: Sequence[Sequence[float]],
    *,
    width: int = FEATURE_WIDTH,
) -> np.ndarray:
    vector = np.zeros(width, dtype=np.float32)
    for index, value in sparse:
        vector[int(index)] = float(value)
    return vector


def permutation_invariant(
    graph: MorphoTopologicalGraph,
) -> bool:
    if len(graph.nodes) < 2:
        return True
    order = tuple(reversed(range(len(graph.nodes))))
    return bool(
        np.array_equal(
            feature_vector(graph),
            feature_vector(graph.permuted(order)),
        )
    )


def forbidden_field_hits(payload: Mapping[str, Any]) -> list[str]:
    encoded = _canonical(payload).lower()
    forbidden = (
        "game_id",
        "source_game_id",
        "node_id",
        "palette",
        "absolute",
        "coordinate",
        "seed",
        "action_name",
    )
    return [
        field
        for field in forbidden
        if f'"{field}"' in encoded
    ]


__all__ = [
    "CONFIDENT_CORRESPONDENCE",
    "FACTOR_NAMES",
    "FEATURE_WIDTH",
    "CompiledTopologicalTransition",
    "causal_factors",
    "compile_topological_transition",
    "correspondence_quality",
    "dense_vector",
    "feature_vector",
    "forbidden_field_hits",
    "graph_tokens",
    "invariant_deltas",
    "local_topological_value",
    "permutation_invariant",
    "sparse_vector",
    "topological_invariants",
]
