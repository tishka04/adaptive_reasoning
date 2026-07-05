"""Episodic graph memory — structured relational memory for analogical retrieval.

Nodes: state motifs, operators, macros, rules, failures, goals.
Edges: precedes, enables, contradicts, similar, dangerous_after.

Retrieval is graph traversal, not flat lookup.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EpisodeNode:
    """A node in the episodic graph."""
    node_id: str
    kind: str           # state_motif / operator / macro / rule / failure / goal
    content: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    access_count: int = 0


@dataclass
class EpisodeEdge:
    """A directed edge in the episodic graph."""
    src: str
    dst: str
    relation: str       # precedes / enables / contradicts / similar / dangerous_after
    weight: float = 1.0


class EpisodicGraph:
    """Graph-structured episodic memory.

    Supports:
      - Adding nodes and edges from game experience
      - Analogical retrieval: given a motif, find related operators/macros
      - Failure suppression: motifs linked to failures get deprioritised
      - Cross-game transfer of subgraphs
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, EpisodeNode] = {}
        self.edges: List[EpisodeEdge] = []
        self._adjacency: Dict[str, List[EpisodeEdge]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        kind: str,
        content: Optional[Dict[str, Any]] = None,
        strength: float = 1.0,
    ) -> EpisodeNode:
        """Add or update a node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.strength = max(node.strength, strength)
            if content:
                node.content.update(content)
            return node

        node = EpisodeNode(
            node_id=node_id,
            kind=kind,
            content=content or {},
            strength=strength,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        weight: float = 1.0,
    ) -> None:
        """Add or strengthen an edge."""
        # Check for existing
        for e in self._adjacency.get(src, []):
            if e.dst == dst and e.relation == relation:
                e.weight = min(e.weight + 0.2, 3.0)  # strengthen
                return

        edge = EpisodeEdge(src=src, dst=dst, relation=relation, weight=weight)
        self.edges.append(edge)
        self._adjacency[src].append(edge)

    def retrieve_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Tuple[EpisodeNode, str, float]]:
        """Retrieve neighbours of a node, optionally filtered by relation.

        Returns: [(neighbor_node, relation, edge_weight), ...]
        """
        if node_id not in self.nodes:
            return []

        self.nodes[node_id].access_count += 1
        results: List[Tuple[EpisodeNode, str, float]] = []

        for edge in self._adjacency.get(node_id, []):
            if relation and edge.relation != relation:
                continue
            dst_node = self.nodes.get(edge.dst)
            if dst_node is not None:
                results.append((dst_node, edge.relation, edge.weight))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    def retrieve_by_motif(
        self,
        motif_content: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Tuple[EpisodeNode, float]]:
        """Find nodes similar to a given state motif.

        Simple similarity: count matching content keys/values.
        """
        scores: List[Tuple[EpisodeNode, float]] = []

        for node in self.nodes.values():
            if node.kind != "state_motif":
                continue
            sim = self._content_similarity(motif_content, node.content)
            if sim > 0.0:
                scores.append((node, sim * node.strength))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max_results]

    def _content_similarity(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> float:
        """Simple key-value overlap similarity."""
        if not a or not b:
            return 0.0
        keys = set(a.keys()) & set(b.keys())
        if not keys:
            return 0.0
        matches = sum(1 for k in keys if a[k] == b[k])
        return matches / max(len(set(a.keys()) | set(b.keys())), 1)

    def record_transition(
        self,
        state_motif_id: str,
        operator_id: str,
        outcome: str,  # "success" / "failure" / "noop"
    ) -> None:
        """Record a state → operator → outcome transition."""
        self.add_node(state_motif_id, "state_motif")
        self.add_node(operator_id, "operator")

        if outcome == "success":
            self.add_edge(state_motif_id, operator_id, "enables", 1.5)
        elif outcome == "failure":
            self.add_edge(state_motif_id, operator_id, "dangerous_after", 1.0)
            self.add_node(f"fail_{state_motif_id}_{operator_id}", "failure")
            self.add_edge(operator_id,
                          f"fail_{state_motif_id}_{operator_id}",
                          "precedes")
        else:
            self.add_edge(state_motif_id, operator_id, "precedes", 0.5)

    def suppress_failure_paths(
        self,
        candidates: List[str],
    ) -> List[str]:
        """Remove operator IDs that are heavily linked to failures."""
        suppressed: Set[str] = set()
        for oid in candidates:
            neighbors = self.retrieve_neighbors(oid, relation="dangerous_after")
            failure_weight = sum(w for _, _, w in neighbors)
            if failure_weight > 2.0:
                suppressed.add(oid)

        return [c for c in candidates if c not in suppressed]

    def decay(self, factor: float = 0.95) -> None:
        """Decay all node strengths and edge weights."""
        for node in self.nodes.values():
            node.strength *= factor
        for edge in self.edges:
            edge.weight *= factor

        # Remove very weak nodes/edges
        weak_nodes = [nid for nid, n in self.nodes.items()
                      if n.strength < 0.1 and n.access_count == 0]
        for nid in weak_nodes:
            del self.nodes[nid]
            if nid in self._adjacency:
                del self._adjacency[nid]

        self.edges = [e for e in self.edges if e.weight >= 0.05]
        for key in self._adjacency:
            self._adjacency[key] = [
                e for e in self._adjacency[key] if e.weight >= 0.05
            ]

    def export_compact(self) -> Dict[str, Any]:
        """Export a compact representation for cross-game transfer."""
        # Only export high-strength nodes and edges
        nodes = {
            nid: {"kind": n.kind, "content": n.content}
            for nid, n in self.nodes.items()
            if n.strength > 0.5
        }
        edges = [
            {"src": e.src, "dst": e.dst, "rel": e.relation}
            for e in self.edges
            if e.weight > 0.5
        ]
        return {"nodes": nodes, "edges": edges}

    def summary(self) -> str:
        return (
            f"EpisodicGraph: {len(self.nodes)} nodes, "
            f"{len(self.edges)} edges"
        )
