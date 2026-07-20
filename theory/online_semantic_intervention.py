"""Entity-anchored, position-invariant identities for online interventions.

The generic ``ACTION6::color:N`` identity is useful for broad exploration but
aliases distinct clickable objects.  This module keeps a concrete structural
slot for local credit while exposing a coarser transfer identity for equivalent
entities.  Neither identity contains absolute coordinates or terminal labels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from v3.schemas import GameObservation, ObjectInfo

from .online_goal_hypothesis import semantic_intervention_signature


INSTANCE_SEPARATOR = "::instance:"


@dataclass(frozen=True)
class SemanticInterventionAnchor:
    """Concrete intervention plus its transferable structural equivalence."""

    action_name: str
    concrete_signature: str
    transfer_signature: str
    legacy_signature: str
    entity_signature: str = ""
    structural_role_signature: str = ""
    instance_signature: str = ""
    anchored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "concrete_signature": self.concrete_signature,
            "transfer_signature": self.transfer_signature,
            "legacy_signature": self.legacy_signature,
            "entity_signature": self.entity_signature,
            "structural_role_signature": self.structural_role_signature,
            "instance_signature": self.instance_signature,
            "anchored": self.anchored,
        }


def semantic_intervention_anchor(
    action_name: str,
    action_data: Mapping[str, Any],
    observation: GameObservation,
    *,
    enabled: bool = True,
) -> SemanticInterventionAnchor:
    """Describe one action without using its absolute click coordinates."""
    normalized = str(action_name).upper()
    legacy = semantic_intervention_signature(
        normalized,
        action_data,
        observation,
    )
    if not enabled or normalized != "ACTION6":
        return SemanticInterventionAnchor(
            action_name=normalized,
            concrete_signature=legacy,
            transfer_signature=legacy,
            legacy_signature=legacy,
        )

    target = _clicked_object(observation, action_data)
    if target is None:
        return SemanticInterventionAnchor(
            action_name=normalized,
            concrete_signature=legacy,
            transfer_signature=legacy,
            legacy_signature=legacy,
        )

    entity = _entity_signature(target)
    role = _structural_role_signature(target, observation)
    peers = [
        item
        for item in observation.objects
        if _entity_signature(item) == entity
        and _structural_role_signature(item, observation) == role
    ]
    peers.sort(key=_instance_order_key)
    ordinal = next(
        (
            index
            for index, item in enumerate(peers, start=1)
            if item.object_id == target.object_id
        ),
        1,
    )
    instance = f"slot:{ordinal}of{max(1, len(peers))}"
    transfer = f"ACTION6::entity:{entity}::role:{role}"
    concrete = f"{transfer}{INSTANCE_SEPARATOR}{instance}"
    return SemanticInterventionAnchor(
        action_name=normalized,
        concrete_signature=concrete,
        transfer_signature=transfer,
        legacy_signature=legacy,
        entity_signature=entity,
        structural_role_signature=role,
        instance_signature=instance,
        anchored=True,
    )


def semantic_transfer_signature(signature: str) -> str:
    """Return the structural alias class encoded in a concrete signature."""
    return str(signature).split(INSTANCE_SEPARATOR, 1)[0]


def is_entity_anchored_signature(signature: str) -> bool:
    value = str(signature)
    return value.startswith("ACTION6::entity:") and INSTANCE_SEPARATOR in value


def _clicked_object(
    observation: GameObservation,
    action_data: Mapping[str, Any],
) -> ObjectInfo | None:
    try:
        x = int(action_data["x"])
        y = int(action_data["y"])
    except (KeyError, TypeError, ValueError):
        return None
    objects = list(observation.objects)
    centered = [
        item
        for item in objects
        if int(round(item.center[1])) == x and int(round(item.center[0])) == y
    ]
    if centered:
        return min(centered, key=lambda item: (item.area, item.object_id))
    containing = [item for item in objects if (y, x) in set(item.cells)]
    if containing:
        return min(containing, key=lambda item: (item.area, item.object_id))
    return None


def _entity_signature(obj: ObjectInfo) -> str:
    row0, col0, row1, col1 = obj.bbox
    normalized_cells = tuple(
        sorted((int(row) - int(row0), int(col) - int(col0)) for row, col in obj.cells)
    )
    payload = ";".join(f"{row},{col}" for row, col in normalized_cells)
    shape = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return ":".join((
        f"color{int(obj.value)}",
        f"shape{int(row1 - row0 + 1)}x{int(col1 - col0 + 1)}-{shape}",
        f"area{_count_bucket(obj.area)}",
    ))


def _structural_role_signature(
    obj: ObjectInfo,
    observation: GameObservation,
) -> str:
    height, width = observation.raw_grid.shape
    row0, col0, row1, col1 = obj.bbox
    horizontal_edges = int(row0 <= 0) + int(row1 >= height - 1)
    vertical_edges = int(col0 <= 0) + int(col1 >= width - 1)
    touched_axes = int(horizontal_edges > 0) + int(vertical_edges > 0)
    boundary = (
        "corner" if touched_axes >= 2 else ("edge" if touched_axes == 1 else "interior")
    )

    player = observation.best_player
    if player is None:
        player_relation = "player-unknown"
    elif player.position in set(obj.cells):
        player_relation = "player-self"
    else:
        distance = min(
            abs(int(row) - int(player.position[0]))
            + abs(int(col) - int(player.position[1]))
            for row, col in obj.cells
        )
        aligned = any(
            int(row) == int(player.position[0])
            or int(col) == int(player.position[1])
            for row, col in obj.cells
        )
        proximity = "adjacent" if distance <= 1 else ("near" if distance <= 4 else "far")
        player_relation = f"player-{proximity}-{'aligned' if aligned else 'offset'}"

    other_objects = [
        item for item in observation.objects if item.object_id != obj.object_id
    ]
    adjacent = sum(_bbox_gap(obj, other) <= 1 for other in other_objects)
    aligned = sum(_center_aligned(obj, other) for other in other_objects)
    same_entity = sum(
        _entity_signature(other) == _entity_signature(obj)
        for other in other_objects
    )
    multiplicity = "unique" if same_entity == 0 else "replicated"
    return ":".join((
        boundary,
        player_relation,
        multiplicity,
        f"adj{_count_bucket(adjacent)}",
        f"align{_count_bucket(aligned)}",
    ))


def _bbox_gap(first: ObjectInfo, second: ObjectInfo) -> int:
    first_row0, first_col0, first_row1, first_col1 = first.bbox
    second_row0, second_col0, second_row1, second_col1 = second.bbox
    row_gap = max(0, second_row0 - first_row1 - 1, first_row0 - second_row1 - 1)
    col_gap = max(0, second_col0 - first_col1 - 1, first_col0 - second_col1 - 1)
    return int(row_gap + col_gap)


def _center_aligned(first: ObjectInfo, second: ObjectInfo) -> bool:
    return bool(
        abs(float(first.center[0]) - float(second.center[0])) < 0.5
        or abs(float(first.center[1]) - float(second.center[1])) < 0.5
    )


def _instance_order_key(obj: ObjectInfo) -> tuple[float, float, int]:
    return (float(obj.center[0]), float(obj.center[1]), int(obj.object_id))


def _count_bucket(count: int) -> str:
    value = max(0, int(count))
    if value <= 2:
        return str(value)
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


__all__ = [
    "SemanticInterventionAnchor",
    "is_entity_anchored_signature",
    "semantic_intervention_anchor",
    "semantic_transfer_signature",
]
