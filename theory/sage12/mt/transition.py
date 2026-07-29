"""Observed graph-to-graph transformations for SAGE-MT."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .graph import (
    MorphoTopologicalGraph,
    MTNode,
    build_mt_graph,
    graph_from_model_view,
    invariant_delta,
    invariant_delta_tokens,
)

TRANSITION_FORMAT_VERSION = "sage12-mt-transition-v4.16"
_AREA_RANK = {
    "one": 1,
    "small": 2,
    "medium": 3,
    "large": 4,
    "very_large": 5,
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class EntityCorrespondence:
    before_ids: tuple[str, ...]
    after_ids: tuple[str, ...]
    kind: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.before_ids and not self.after_ids:
            raise ValueError("empty SAGE-MT correspondence")


@dataclass(frozen=True)
class MTTransitionRecord:
    transition_id: str
    source_game_id: str
    action_name: str
    action_data: Mapping[str, Any]
    graph_before: MorphoTopologicalGraph
    graph_after: MorphoTopologicalGraph
    correspondences: tuple[EntityCorrespondence, ...]
    events: tuple[str, ...]
    invariant_deltas: Mapping[str, int]
    delta_signature: str
    productive: bool | None = None
    risk: bool | None = None
    audit: Mapping[str, Any] = field(default_factory=dict)
    format_version: str = TRANSITION_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != TRANSITION_FORMAT_VERSION:
            raise ValueError("unsupported SAGE-MT transition version")

    def student_view(self) -> dict[str, Any]:
        return {
            "graph_before": self.graph_before.model_view(),
            "action_name": self.action_name,
            "action_family": self.graph_before.action_family,
        }

    def teacher_view(self) -> dict[str, Any]:
        return {
            "graph_before": self.graph_before.model_view(),
            "graph_after": self.graph_after.model_view(),
            "events": list(self.events),
            "invariant_deltas": dict(self.invariant_deltas),
            "delta_signature": self.delta_signature,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "transition_id": self.transition_id,
            "student_view": self.student_view(),
            "teacher": {
                **self.teacher_view(),
                "correspondences": [asdict(item) for item in self.correspondences],
                "productive": self.productive,
                "risk": self.risk,
            },
            "audit": {
                "source_game_id": self.source_game_id,
                "action_data": dict(self.action_data),
                **dict(self.audit),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MTTransitionRecord:
        if payload.get("format_version") != TRANSITION_FORMAT_VERSION:
            raise ValueError("unsupported SAGE-MT transition version")
        student = dict(payload["student_view"])
        teacher = dict(payload["teacher"])
        audit = dict(payload.get("audit", {}))
        source_game_id = str(audit.pop("source_game_id", ""))
        action_data = dict(audit.pop("action_data", {}))
        return cls(
            transition_id=str(payload["transition_id"]),
            source_game_id=source_game_id,
            action_name=str(student["action_name"]),
            action_data=action_data,
            graph_before=graph_from_model_view(student["graph_before"]),
            graph_after=graph_from_model_view(teacher["graph_after"]),
            correspondences=tuple(
                EntityCorrespondence(
                    before_ids=tuple(item.get("before_ids", ())),
                    after_ids=tuple(item.get("after_ids", ())),
                    kind=str(item["kind"]),
                    confidence=float(item["confidence"]),
                )
                for item in teacher.get("correspondences", ())
            ),
            events=tuple(str(item) for item in teacher.get("events", ())),
            invariant_deltas={
                str(key): int(value)
                for key, value in dict(
                    teacher.get("invariant_deltas", {})
                ).items()
            },
            delta_signature=str(teacher["delta_signature"]),
            productive=teacher.get("productive"),
            risk=teacher.get("risk"),
            audit=audit,
        )


def _node_similarity(before: MTNode, after: MTNode) -> float:
    if before.kind != after.kind:
        return -1.0
    overlap = len(before.cells & after.cells) / max(
        1,
        len(before.cells | after.cells),
    )
    morphology = sum(
        (
            before.area_bucket == after.area_bucket,
            before.aspect_bucket == after.aspect_bucket,
            before.compactness_bucket == after.compactness_bucket,
            before.holes == after.holes,
            before.boundary_contacts == after.boundary_contacts,
        )
    ) / 5.0
    distance = math.dist(before.center, after.center)
    motion = 1.0 / (1.0 + distance)
    return 0.65 * overlap + 0.25 * morphology + 0.10 * motion


def _overlap_groups(
    before_nodes: Sequence[MTNode],
    after_nodes: Sequence[MTNode],
) -> tuple[list[EntityCorrespondence], set[str], set[str]]:
    edges: dict[tuple[str, str], float] = {}
    for before in before_nodes:
        for after in after_nodes:
            if before.kind != after.kind:
                continue
            overlap = len(before.cells & after.cells)
            if overlap:
                edges[(before.node_id, after.node_id)] = overlap / max(
                    1,
                    min(len(before.cells), len(after.cells)),
                )
    before_to_after: dict[str, set[str]] = defaultdict(set)
    after_to_before: dict[str, set[str]] = defaultdict(set)
    for before_id, after_id in edges:
        before_to_after[before_id].add(after_id)
        after_to_before[after_id].add(before_id)
    seen_before: set[str] = set()
    seen_after: set[str] = set()
    output: list[EntityCorrespondence] = []
    for start in sorted(before_to_after):
        if start in seen_before:
            continue
        queue = deque([("before", start)])
        group_before: set[str] = set()
        group_after: set[str] = set()
        while queue:
            side, identifier = queue.popleft()
            if side == "before":
                if identifier in group_before:
                    continue
                group_before.add(identifier)
                queue.extend(("after", item) for item in before_to_after[identifier])
            else:
                if identifier in group_after:
                    continue
                group_after.add(identifier)
                queue.extend(("before", item) for item in after_to_before[identifier])
        seen_before.update(group_before)
        seen_after.update(group_after)
        if len(group_before) > 1 and len(group_after) == 1:
            kind = "merge"
        elif len(group_before) == 1 and len(group_after) > 1:
            kind = "split"
        else:
            kind = "persist"
        confidence = sum(
            edges[(before_id, after_id)]
            for before_id in group_before
            for after_id in group_after
            if (before_id, after_id) in edges
        ) / max(1, len(group_before) * len(group_after))
        output.append(
            EntityCorrespondence(
                before_ids=tuple(sorted(group_before)),
                after_ids=tuple(sorted(group_after)),
                kind=kind,
                confidence=float(confidence),
            )
        )
    return output, seen_before, seen_after


def align_graphs(
    before: MorphoTopologicalGraph,
    after: MorphoTopologicalGraph,
) -> tuple[EntityCorrespondence, ...]:
    """Create deterministic one-to-one and many-to-many correspondences."""

    before_objects = [node for node in before.nodes if node.kind == "object"]
    after_objects = [node for node in after.nodes if node.kind == "object"]
    groups, used_before, used_after = _overlap_groups(before_objects, after_objects)
    candidates = []
    for left in before_objects:
        if left.node_id in used_before:
            continue
        for right in after_objects:
            if right.node_id in used_after:
                continue
            similarity = _node_similarity(left, right)
            if similarity >= 0.35:
                candidates.append(
                    (
                        -similarity,
                        left.node_id,
                        right.node_id,
                        similarity,
                    )
                )
    for _, before_id, after_id, similarity in sorted(candidates):
        if before_id in used_before or after_id in used_after:
            continue
        used_before.add(before_id)
        used_after.add(after_id)
        groups.append(
            EntityCorrespondence(
                before_ids=(before_id,),
                after_ids=(after_id,),
                kind="persist",
                confidence=float(similarity),
            )
        )
    for node in before_objects:
        if node.node_id not in used_before:
            groups.append(
                EntityCorrespondence(
                    before_ids=(node.node_id,),
                    after_ids=(),
                    kind="death",
                    confidence=1.0,
                )
            )
    for node in after_objects:
        if node.node_id not in used_after:
            groups.append(
                EntityCorrespondence(
                    before_ids=(),
                    after_ids=(node.node_id,),
                    kind="birth",
                    confidence=1.0,
                )
            )
    return tuple(
        sorted(
            groups,
            key=lambda item: (
                item.kind,
                item.before_ids,
                item.after_ids,
            ),
        )
    )


def _event_tokens(
    before: MorphoTopologicalGraph,
    after: MorphoTopologicalGraph,
    correspondences: Sequence[EntityCorrespondence],
    deltas: Mapping[str, int],
) -> tuple[str, ...]:
    before_by_id = {node.node_id: node for node in before.nodes}
    after_by_id = {node.node_id: node for node in after.nodes}
    events: Counter[str] = Counter()
    for correspondence in correspondences:
        events[correspondence.kind] += 1
        if (
            correspondence.kind != "persist"
            or len(correspondence.before_ids) != 1
            or len(correspondence.after_ids) != 1
        ):
            continue
        left = before_by_id[correspondence.before_ids[0]]
        right = after_by_id[correspondence.after_ids[0]]
        if _AREA_RANK[right.area_bucket] > _AREA_RANK[left.area_bucket]:
            events["growth"] += 1
        elif _AREA_RANK[right.area_bucket] < _AREA_RANK[left.area_bucket]:
            events["contraction"] += 1
        if (
            left.aspect_bucket != right.aspect_bucket
            or left.compactness_bucket != right.compactness_bucket
            or left.holes != right.holes
        ):
            events["morphology_changed"] += 1
        if math.dist(left.center, right.center) > 0.75:
            events["relative_motion"] += 1

    one_to_one = {
        row.before_ids[0]: row.after_ids[0]
        for row in correspondences
        if len(row.before_ids) == 1 and len(row.after_ids) == 1
    }
    before_relations = {
        (relation.kind, relation.subject_id, relation.object_id)
        for relation in before.relations
        if relation.subject_id in one_to_one and relation.object_id in one_to_one
    }
    mapped_relations = {
        (kind, one_to_one[subject], one_to_one[obj])
        for kind, subject, obj in before_relations
    }
    after_relations = {
        (relation.kind, relation.subject_id, relation.object_id)
        for relation in after.relations
    }
    for kind, _, _ in mapped_relations - after_relations:
        events[f"relation_removed:{kind}"] += 1
    for kind, _, _ in after_relations - mapped_relations:
        events[f"relation_added:{kind}"] += 1
    events.update(invariant_delta_tokens(deltas))
    if not events:
        events["noop"] = 1
    return tuple(
        f"{event}#{count}"
        for event, count in sorted(events.items())
    )


def _delta_signature(events: Sequence[str], deltas: Mapping[str, int]) -> str:
    payload = {
        "events": sorted(events),
        "invariants": {
            key: -1 if int(value) < 0 else 1 if int(value) > 0 else 0
            for key, value in sorted(deltas.items())
        },
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]
    dominant = next(
        (
            item.split("#", 1)[0]
            for item in events
            if not item.startswith("invariant:")
        ),
        "noop",
    )
    return f"{dominant}:{digest}"


def compile_mt_transition(
    grid_before: Any,
    action_name: str,
    grid_after: Any,
    *,
    action_data: Mapping[str, Any] | None = None,
    source_game_id: str = "",
    player_position_before: tuple[int, int] | None = None,
    player_position_after: tuple[int, int] | None = None,
    productive: bool | None = None,
    risk: bool | None = None,
    audit: Mapping[str, Any] | None = None,
) -> MTTransitionRecord:
    """Compile one observed state-action-state triple."""

    payload = dict(action_data or {})
    before = build_mt_graph(
        grid_before,
        action_name=action_name,
        action_data=payload,
        player_position=player_position_before,
    )
    after = build_mt_graph(
        grid_after,
        action_name=action_name,
        action_data=payload,
        player_position=player_position_after,
    )
    correspondences = align_graphs(before, after)
    deltas = invariant_delta(before, after)
    events = _event_tokens(before, after, correspondences, deltas)
    signature = _delta_signature(events, deltas)
    transition_payload = {
        "source_game_id": str(source_game_id).split("-", 1)[0],
        "before": before.signature,
        "action": str(action_name).strip().upper(),
        "action_data": payload,
        "after": after.signature,
        "delta": signature,
    }
    transition_id = hashlib.sha256(
        _canonical(transition_payload).encode("utf-8")
    ).hexdigest()
    return MTTransitionRecord(
        transition_id=transition_id,
        source_game_id=str(source_game_id).split("-", 1)[0],
        action_name=str(action_name).strip().upper(),
        action_data=payload,
        graph_before=before,
        graph_after=after,
        correspondences=correspondences,
        events=events,
        invariant_deltas=deltas,
        delta_signature=signature,
        productive=productive,
        risk=risk,
        audit=dict(audit or {}),
    )


__all__ = [
    "TRANSITION_FORMAT_VERSION",
    "EntityCorrespondence",
    "MTTransitionRecord",
    "align_graphs",
    "compile_mt_transition",
]
