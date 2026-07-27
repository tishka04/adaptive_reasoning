"""Grounded scene graphs without an explicit game-identity feature."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple

from v3.schemas import GameObservation, ObjectInfo


@dataclass(frozen=True)
class GroundedEntity:
    entity_id: str
    roles: Tuple[str, ...]
    center: Tuple[float, float]
    area_bucket: str
    aspect_bucket: str
    value_token: str


@dataclass(frozen=True)
class GroundedRelation:
    kind: str
    subject_id: str
    object_id: str

    @property
    def key(self) -> str:
        return f"{self.kind}|{self.subject_id}|{self.object_id}"


@dataclass(frozen=True)
class SceneGraph:
    entities: Tuple[GroundedEntity, ...]
    relations: Tuple[GroundedRelation, ...]
    state_predicates: frozenset[str]
    signature: str

    def entities_for_role(self, role: str) -> Tuple[GroundedEntity, ...]:
        return tuple(entity for entity in self.entities if role in entity.roles)


def build_scene_graph(observation: GameObservation) -> SceneGraph:
    """Build relations that vary within a game without using game identity."""
    height, width = observation.raw_grid.shape[:2]
    diagonal = max(1.0, math.hypot(height, width))
    player = observation.best_player
    player_position = tuple(player.position) if player is not None else None
    entities = []
    for index, obj in enumerate(
        sorted(observation.objects, key=lambda item: (item.center, item.area))
    ):
        roles = {"object"}
        if player_position is not None and _contains(obj, player_position):
            roles.add("player")
        for affordance in observation.affordances:
            if affordance.target == obj.object_id:
                roles.add(str(affordance.kind.value))
        if not roles.intersection({"player", "hazardous", "collectible"}):
            roles.add("target")
        entity_id = f"e{index}:{_shape_token(obj)}"
        entities.append(
            GroundedEntity(
                entity_id=entity_id,
                roles=tuple(sorted(roles)),
                center=tuple(float(value) for value in obj.center),
                area_bucket=_area_bucket(obj.area),
                aspect_bucket=_aspect_bucket(obj),
                value_token=f"v{int(obj.value)}",
            )
        )
    if player_position is not None and not any(
        "player" in entity.roles for entity in entities
    ):
        entities.insert(
            0,
            GroundedEntity(
                entity_id="e_player",
                roles=("object", "player"),
                center=(float(player_position[0]), float(player_position[1])),
                area_bucket="one",
                aspect_bucket="square",
                value_token="player",
            ),
        )

    relations = []
    for subject in entities:
        for target in entities:
            if subject.entity_id == target.entity_id:
                continue
            dr = target.center[0] - subject.center[0]
            dc = target.center[1] - subject.center[1]
            distance = math.hypot(dr, dc)
            if distance / diagonal <= 0.25:
                relations.append(
                    GroundedRelation("near", subject.entity_id, target.entity_id)
                )
            if abs(dr) <= 0.5:
                relations.append(
                    GroundedRelation(
                        "aligned", subject.entity_id, target.entity_id
                    )
                )
            if abs(dc) <= 0.5:
                relations.append(
                    GroundedRelation(
                        "aligned", subject.entity_id, target.entity_id
                    )
                )
            if distance <= 1.5:
                relations.append(
                    GroundedRelation(
                        "contact", subject.entity_id, target.entity_id
                    )
                )
                relations.append(
                    GroundedRelation(
                        "adjacent", subject.entity_id, target.entity_id
                    )
                )
            if dr < -0.5:
                relations.append(
                    GroundedRelation(
                        "north_of", target.entity_id, subject.entity_id
                    )
                )
            elif dr > 0.5:
                relations.append(
                    GroundedRelation(
                        "south_of", target.entity_id, subject.entity_id
                    )
                )
            if dc < -0.5:
                relations.append(
                    GroundedRelation(
                        "west_of", target.entity_id, subject.entity_id
                    )
                )
            elif dc > 0.5:
                relations.append(
                    GroundedRelation(
                        "east_of", target.entity_id, subject.entity_id
                    )
                )

    state = {relation.key for relation in relations}
    state.update(f"exists|{entity.entity_id}|-|" for entity in entities)
    if str(observation.game_state).upper() == "WIN":
        state.add("level_complete|-|-|")
    if str(observation.game_state).upper() == "GAME_OVER":
        state.add("game_over|-|-|")
    canonical = {
        "entities": [
            {
                "id": entity.entity_id,
                "roles": entity.roles,
                "area": entity.area_bucket,
                "aspect": entity.aspect_bucket,
            }
            for entity in entities
        ],
        "relations": sorted(state),
    }
    signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return SceneGraph(
        entities=tuple(entities),
        relations=tuple(
            sorted(relations, key=lambda relation: relation.key)
        ),
        state_predicates=frozenset(state),
        signature=signature,
    )


@dataclass
class SemanticMemory:
    """Observed support/refutation only; proposals never mutate these counts."""

    support: Counter[str] = field(default_factory=Counter)
    refutations: Counter[str] = field(default_factory=Counter)
    observations: int = 0
    branch_index: int = 0

    def observe(
        self,
        predicted: Iterable[str],
        observed: Iterable[str],
    ) -> None:
        predicted_set = set(predicted)
        observed_set = set(observed)
        for key in predicted_set & observed_set:
            self.support[key] += 1
        for key in predicted_set - observed_set:
            self.refutations[key] += 1
        self.observations += 1

    def start_branch(self) -> None:
        self.branch_index += 1

    def probability(self, predicate: str) -> float:
        supported = self.support[predicate]
        refuted = self.refutations[predicate]
        return (supported + 1.0) / (supported + refuted + 2.0)

    def snapshot(self) -> Mapping[str, object]:
        return {
            "observations": self.observations,
            "branch_index": self.branch_index,
            "support": dict(self.support),
            "refutations": dict(self.refutations),
        }


def _contains(obj: ObjectInfo, position: Tuple[int, int]) -> bool:
    return position in set(obj.cells)


def _shape_token(obj: ObjectInfo) -> str:
    return f"{_area_bucket(obj.area)}:{_aspect_bucket(obj)}"


def _area_bucket(area: int) -> str:
    if area <= 1:
        return "one"
    if area <= 4:
        return "small"
    if area <= 16:
        return "medium"
    return "large"


def _aspect_bucket(obj: ObjectInfo) -> str:
    r0, c0, r1, c1 = obj.bbox
    height = max(1, r1 - r0 + 1)
    width = max(1, c1 - c0 + 1)
    ratio = width / height
    if ratio >= 1.5:
        return "wide"
    if ratio <= 2.0 / 3.0:
        return "tall"
    return "square"


__all__ = [
    "GroundedEntity",
    "GroundedRelation",
    "SceneGraph",
    "SemanticMemory",
    "build_scene_graph",
]
