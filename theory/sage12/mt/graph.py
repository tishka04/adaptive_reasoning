"""Identity-free morpho-topological scene graphs for SAGE-MT.

Raw cells and absolute centres are retained only inside the in-memory audit
objects used to align two observed states.  ``model_view`` is the sole
authorized representation for learned components.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..action_target_data import infer_background

GRAPH_FORMAT_VERSION = "sage12-mt-graph-v4.16"
MAXIMUM_COMPONENTS = 64

_MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "ACTION1": (-1, 0),
    "ACTION2": (1, 0),
    "ACTION3": (0, -1),
    "ACTION4": (0, 1),
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _bucket(value: float, boundaries: Sequence[tuple[float, str]]) -> str:
    for limit, label in boundaries:
        if value <= limit:
            return label
    return boundaries[-1][1]


def _area_bucket(area: int) -> str:
    return _bucket(
        float(area),
        (
            (1, "one"),
            (4, "small"),
            (16, "medium"),
            (64, "large"),
            (float("inf"), "very_large"),
        ),
    )


def _aspect_bucket(height: int, width: int) -> str:
    ratio = float(width) / max(float(height), 1.0)
    if ratio <= 2.0 / 3.0:
        return "tall"
    if ratio >= 1.5:
        return "wide"
    return "compact"


def _compactness_bucket(area: int, perimeter: int) -> str:
    if perimeter <= 0:
        return "unknown"
    value = 4.0 * math.pi * float(area) / float(perimeter * perimeter)
    return _bucket(
        value,
        ((0.20, "sparse"), (0.45, "irregular"), (0.70, "compact"), (1.0, "round")),
    )


def _signed_bucket(value: int) -> str:
    if value < 0:
        return "decreased"
    if value > 0:
        return "increased"
    return "preserved"


def _neighbors4(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )


def _components(mask: np.ndarray) -> list[frozenset[tuple[int, int]]]:
    height, width = mask.shape
    unseen = set(zip(*np.nonzero(mask)))
    output: list[frozenset[tuple[int, int]]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        cells = {start}
        while queue:
            row, col = queue.popleft()
            for candidate in _neighbors4(int(row), int(col)):
                rr, cc = candidate
                if (
                    0 <= rr < height
                    and 0 <= cc < width
                    and candidate in unseen
                ):
                    unseen.remove(candidate)
                    cells.add(candidate)
                    queue.append(candidate)
        output.append(frozenset(cells))
    return output


def _perimeter(cells: frozenset[tuple[int, int]]) -> int:
    return sum(
        1
        for row, col in cells
        for candidate in _neighbors4(row, col)
        if candidate not in cells
    )


def _holes(cells: frozenset[tuple[int, int]]) -> int:
    if not cells:
        return 0
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for row, col in cells:
        mask[row - r0, col - c0] = True
    empty = _components(~mask)
    return sum(
        1
        for component in empty
        if not any(
            row in {0, mask.shape[0] - 1}
            or col in {0, mask.shape[1] - 1}
            for row, col in component
        )
    )


def _boundary_contacts(
    cells: frozenset[tuple[int, int]],
    shape: tuple[int, int],
) -> int:
    contacts = set()
    for row, col in cells:
        if row == 0:
            contacts.add("north")
        if row == shape[0] - 1:
            contacts.add("south")
        if col == 0:
            contacts.add("west")
        if col == shape[1] - 1:
            contacts.add("east")
    return len(contacts)


def _center(cells: Iterable[tuple[int, int]]) -> tuple[float, float]:
    values = tuple(cells)
    if not values:
        return (0.0, 0.0)
    return (
        sum(row for row, _ in values) / len(values),
        sum(col for _, col in values) / len(values),
    )


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    values = tuple(cells)
    rows = [row for row, _ in values]
    cols = [col for _, col in values]
    return min(rows), min(cols), max(rows), max(cols)


def _action_axis(
    action_name: str,
    action_data: Mapping[str, Any],
    player_position: tuple[int, int] | None,
) -> tuple[tuple[float, float] | None, tuple[int, int] | None]:
    name = str(action_name).strip().upper()
    vector = _MOVE_VECTORS.get(name)
    row = action_data.get("row")
    col = action_data.get("col")
    if row is None:
        row = action_data.get("y")
    if col is None:
        col = action_data.get("x")
    anchor = (
        (float(row), float(col))
        if row is not None and col is not None
        else (
            (float(player_position[0]), float(player_position[1]))
            if player_position is not None
            else None
        )
    )
    return anchor, vector


def _axis_relation(
    center: tuple[float, float],
    anchor: tuple[float, float] | None,
    vector: tuple[int, int] | None,
) -> str:
    if anchor is None:
        return "unanchored"
    dr = center[0] - anchor[0]
    dc = center[1] - anchor[1]
    if abs(dr) <= 0.5 and abs(dc) <= 0.5:
        return "overlap"
    if vector is None:
        return _bucket(
            math.hypot(dr, dc),
            ((1.5, "radial_near"), (5.0, "radial_mid"), (float("inf"), "radial_far")),
        )
    forward = dr * vector[0] + dc * vector[1]
    lateral = dr * -vector[1] + dc * vector[0]
    if abs(forward) >= abs(lateral):
        return "ahead" if forward > 0 else "behind"
    return "lateral_right" if lateral > 0 else "lateral_left"


@dataclass(frozen=True)
class MTNode:
    """One component or free-space region.

    ``cells`` and ``center`` are alignment-only provenance.  They are omitted
    from :meth:`model_view` and from graph signatures.
    """

    node_id: str
    kind: str
    roles: tuple[str, ...]
    area_bucket: str
    aspect_bucket: str
    compactness_bucket: str
    holes: int
    boundary_contacts: int
    action_relation: str
    cells: frozenset[tuple[int, int]] = field(repr=False, compare=False)
    center: tuple[float, float] = field(repr=False, compare=False)

    def model_view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "roles": list(self.roles),
            "area_bucket": self.area_bucket,
            "aspect_bucket": self.aspect_bucket,
            "compactness_bucket": self.compactness_bucket,
            "holes": min(max(int(self.holes), 0), 4),
            "boundary_contacts": min(max(int(self.boundary_contacts), 0), 4),
            "action_relation": self.action_relation,
        }


@dataclass(frozen=True)
class MTRelation:
    kind: str
    subject_id: str
    object_id: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.kind, self.subject_id, self.object_id


@dataclass(frozen=True)
class MorphoTopologicalGraph:
    nodes: tuple[MTNode, ...]
    relations: tuple[MTRelation, ...]
    invariants: Mapping[str, int]
    action_name: str
    action_family: str
    signature: str
    format_version: str = GRAPH_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != GRAPH_FORMAT_VERSION:
            raise ValueError("unsupported SAGE-MT graph version")

    def model_view(self) -> dict[str, Any]:
        by_id = {node.node_id: index for index, node in enumerate(self.nodes)}
        return {
            "nodes": [node.model_view() for node in self.nodes],
            "relations": [
                {
                    "kind": relation.kind,
                    "subject": by_id[relation.subject_id],
                    "object": by_id[relation.object_id],
                }
                for relation in self.relations
            ],
            "invariants": {
                str(key): int(value)
                for key, value in sorted(self.invariants.items())
            },
            "action_name": self.action_name,
            "action_family": self.action_family,
        }

    def permuted(self, order: Sequence[int]) -> MorphoTopologicalGraph:
        if sorted(order) != list(range(len(self.nodes))):
            raise ValueError("node permutation must cover every node exactly")
        nodes = tuple(self.nodes[index] for index in order)
        return _finalize_graph(
            nodes=nodes,
            relations=self.relations,
            invariants=self.invariants,
            action_name=self.action_name,
            action_family=self.action_family,
        )

    def without_relations(self) -> MorphoTopologicalGraph:
        return _finalize_graph(
            nodes=self.nodes,
            relations=(),
            invariants=self.invariants,
            action_name=self.action_name,
            action_family=self.action_family,
        )


def _action_family(action_name: str, action_data: Mapping[str, Any]) -> str:
    name = str(action_name).strip().upper()
    if name in _MOVE_VECTORS:
        return "move"
    if name in {"ACTION5", "ACTION6"}:
        return "interact"
    if action_data:
        return "parameterized"
    return "other"


def _semantic_node_sort_key(node: MTNode) -> tuple[Any, ...]:
    return (
        node.kind,
        node.roles,
        node.area_bucket,
        node.aspect_bucket,
        node.compactness_bucket,
        node.holes,
        node.boundary_contacts,
        node.action_relation,
        node.center,
    )


def _finalize_graph(
    *,
    nodes: Sequence[MTNode],
    relations: Sequence[MTRelation],
    invariants: Mapping[str, int],
    action_name: str,
    action_family: str,
) -> MorphoTopologicalGraph:
    graph = MorphoTopologicalGraph(
        nodes=tuple(nodes),
        relations=tuple(sorted(set(relations), key=lambda item: item.key)),
        invariants=dict(sorted((str(key), int(value)) for key, value in invariants.items())),
        action_name=str(action_name).strip().upper(),
        action_family=str(action_family),
        signature="",
    )
    view = graph.model_view()
    canonical = {
        "node_multiset": sorted(_canonical(node) for node in view["nodes"]),
        "relation_types": dict(
            sorted(Counter(row["kind"] for row in view["relations"]).items())
        ),
        "invariants": view["invariants"],
        "action_family": graph.action_family,
    }
    return MorphoTopologicalGraph(
        nodes=graph.nodes,
        relations=graph.relations,
        invariants=graph.invariants,
        action_name=graph.action_name,
        action_family=graph.action_family,
        signature=hashlib.sha256(_canonical(canonical).encode("utf-8")).hexdigest()[:20],
    )


def build_mt_graph(
    grid: Any,
    *,
    action_name: str = "",
    action_data: Mapping[str, Any] | None = None,
    player_position: tuple[int, int] | None = None,
    maximum_components: int = MAXIMUM_COMPONENTS,
) -> MorphoTopologicalGraph:
    """Build a palette-free component and free-space topology graph."""

    array = np.asarray(grid, dtype=np.int32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("SAGE-MT requires a non-empty two-dimensional grid")
    action_payload = dict(action_data or {})
    background = infer_background(array)
    anchor, vector = _action_axis(action_name, action_payload, player_position)
    raw_components: list[tuple[str, frozenset[tuple[int, int]]]] = []
    for value in sorted(int(item) for item in np.unique(array) if int(item) != background):
        for component in _components(array == value):
            raw_components.append(("object", component))
    for component in _components(array == background):
        raw_components.append(("free_region", component))
    raw_components.sort(
        key=lambda item: (
            item[0],
            -len(item[1]),
            _bbox(item[1]),
        )
    )
    raw_components = raw_components[: max(1, int(maximum_components))]

    nodes: list[MTNode] = []
    for index, (kind, cells) in enumerate(raw_components):
        r0, c0, r1, c1 = _bbox(cells)
        center = _center(cells)
        roles = set()
        if kind == "free_region":
            roles.add("space")
        else:
            roles.add("object")
        if player_position is not None and player_position in cells:
            roles.add("actor")
        anchor_in_component = bool(
            anchor is not None
            and (round(anchor[0]), round(anchor[1])) in cells
        )
        if anchor_in_component:
            roles.add("action_root")
        nodes.append(
            MTNode(
                node_id=f"mt{index}",
                kind=kind,
                roles=tuple(sorted(roles)),
                area_bucket=_area_bucket(len(cells)),
                aspect_bucket=_aspect_bucket(r1 - r0 + 1, c1 - c0 + 1),
                compactness_bucket=_compactness_bucket(
                    len(cells),
                    _perimeter(cells),
                ),
                holes=_holes(cells) if kind == "object" else 0,
                boundary_contacts=_boundary_contacts(cells, array.shape),
                action_relation=(
                    _axis_relation(center, anchor, vector)
                    if kind == "object"
                    else "overlap"
                    if anchor_in_component
                    else "background_region"
                ),
                cells=cells,
                center=center,
            )
        )
    nodes.sort(key=_semantic_node_sort_key)
    # Local ids are regenerated after canonical ordering. They are graph
    # references only and never persistent object identities.
    nodes = [
        MTNode(
            node_id=f"mt{index}",
            kind=node.kind,
            roles=node.roles,
            area_bucket=node.area_bucket,
            aspect_bucket=node.aspect_bucket,
            compactness_bucket=node.compactness_bucket,
            holes=node.holes,
            boundary_contacts=node.boundary_contacts,
            action_relation=node.action_relation,
            cells=node.cells,
            center=node.center,
        )
        for index, node in enumerate(nodes)
    ]

    diagonal = max(1.0, math.hypot(*array.shape))
    relations: list[MTRelation] = []
    bboxes = {node.node_id: _bbox(node.cells) for node in nodes}
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            distance = math.dist(left.center, right.center)
            contact = any(
                candidate in right.cells
                for row, col in left.cells
                for candidate in _neighbors4(row, col)
            )
            if contact:
                relations.extend(
                    (
                        MTRelation("contact", left.node_id, right.node_id),
                        MTRelation("contact", right.node_id, left.node_id),
                    )
                )
            elif (
                left.kind == "object"
                and right.kind == "object"
                and distance / diagonal <= 0.25
            ):
                relations.extend(
                    (
                        MTRelation("near", left.node_id, right.node_id),
                        MTRelation("near", right.node_id, left.node_id),
                    )
                )
            lbox = bboxes[left.node_id]
            rbox = bboxes[right.node_id]
            if lbox[0] <= rbox[0] and lbox[1] <= rbox[1] and lbox[2] >= rbox[2] and lbox[3] >= rbox[3]:
                relations.append(MTRelation("encloses", left.node_id, right.node_id))
            if rbox[0] <= lbox[0] and rbox[1] <= lbox[1] and rbox[2] >= lbox[2] and rbox[3] >= lbox[3]:
                relations.append(MTRelation("encloses", right.node_id, left.node_id))
            if (
                left.kind == "object"
                and right.kind == "object"
                and abs(left.center[0] - right.center[0]) <= 0.5
            ):
                relations.extend(
                    (
                        MTRelation("aligned", left.node_id, right.node_id),
                        MTRelation("aligned", right.node_id, left.node_id),
                    )
                )
            if (
                left.kind == "object"
                and right.kind == "object"
                and abs(left.center[1] - right.center[1]) <= 0.5
            ):
                relations.extend(
                    (
                        MTRelation("aligned", left.node_id, right.node_id),
                        MTRelation("aligned", right.node_id, left.node_id),
                    )
                )

    object_nodes = [node for node in nodes if node.kind == "object"]
    free_nodes = [node for node in nodes if node.kind == "free_region"]
    object_contact_edges = {
        tuple(sorted((relation.subject_id, relation.object_id)))
        for relation in relations
        if relation.kind == "contact"
        and relation.subject_id != relation.object_id
        and next(node for node in nodes if node.node_id == relation.subject_id).kind
        == "object"
        and next(node for node in nodes if node.node_id == relation.object_id).kind
        == "object"
    }
    invariants = {
        "object_components": len(object_nodes),
        "free_regions": len(free_nodes),
        "holes": sum(node.holes for node in object_nodes),
        "euler_characteristic": len(object_nodes) - sum(node.holes for node in object_nodes),
        "contact_edges": len(object_contact_edges),
        "boundary_connected_free_regions": sum(
            int(node.boundary_contacts > 0) for node in free_nodes
        ),
        "largest_free_region_bucket": (
            {"one": 1, "small": 2, "medium": 3, "large": 4, "very_large": 5}.get(
                max(
                    (node.area_bucket for node in free_nodes),
                    key=lambda item: {
                        "one": 1,
                        "small": 2,
                        "medium": 3,
                        "large": 4,
                        "very_large": 5,
                    }.get(item, 0),
                    default="one",
                ),
                1,
            )
        ),
    }
    return _finalize_graph(
        nodes=nodes,
        relations=relations,
        invariants=invariants,
        action_name=action_name,
        action_family=_action_family(action_name, action_payload),
    )


def graph_from_model_view(
    payload: Mapping[str, Any],
) -> MorphoTopologicalGraph:
    """Rehydrate the coordinate-free persisted model projection."""

    nodes = []
    for index, raw in enumerate(payload.get("nodes", ())):
        item = dict(raw)
        nodes.append(
            MTNode(
                node_id=f"mt{index}",
                kind=str(item["kind"]),
                roles=tuple(str(value) for value in item.get("roles", ())),
                area_bucket=str(item["area_bucket"]),
                aspect_bucket=str(item["aspect_bucket"]),
                compactness_bucket=str(item["compactness_bucket"]),
                holes=int(item.get("holes", 0)),
                boundary_contacts=int(item.get("boundary_contacts", 0)),
                action_relation=str(item.get("action_relation", "unanchored")),
                cells=frozenset(),
                center=(0.0, 0.0),
            )
        )
    relations = []
    for raw in payload.get("relations", ()):
        item = dict(raw)
        subject = int(item["subject"])
        obj = int(item["object"])
        if not 0 <= subject < len(nodes) or not 0 <= obj < len(nodes):
            raise ValueError("persisted SAGE-MT relation references unknown node")
        relations.append(
            MTRelation(
                kind=str(item["kind"]),
                subject_id=nodes[subject].node_id,
                object_id=nodes[obj].node_id,
            )
        )
    return _finalize_graph(
        nodes=nodes,
        relations=relations,
        invariants={
            str(key): int(value)
            for key, value in dict(payload.get("invariants", {})).items()
        },
        action_name=str(payload.get("action_name", "")),
        action_family=str(payload.get("action_family", "other")),
    )


def invariant_delta(
    before: MorphoTopologicalGraph,
    after: MorphoTopologicalGraph,
) -> dict[str, int]:
    keys = set(before.invariants) | set(after.invariants)
    return {
        key: int(after.invariants.get(key, 0)) - int(before.invariants.get(key, 0))
        for key in sorted(keys)
    }


def invariant_delta_tokens(deltas: Mapping[str, int]) -> tuple[str, ...]:
    return tuple(
        f"invariant:{key}:{_signed_bucket(int(value))}"
        for key, value in sorted(deltas.items())
    )


__all__ = [
    "GRAPH_FORMAT_VERSION",
    "MAXIMUM_COMPONENTS",
    "MTNode",
    "MTRelation",
    "MorphoTopologicalGraph",
    "build_mt_graph",
    "graph_from_model_view",
    "invariant_delta",
    "invariant_delta_tokens",
]
