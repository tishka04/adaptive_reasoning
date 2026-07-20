"""Online causal binding between concrete intervention targets and effects.

SAGE.8t identifies clickable structural slots but does not track the targeted
entity through a transition.  This module assigns short-lived online track ids,
matches color/shape transformations, and attributes objective progress only
when the targeted entity itself changed.  Controlled contrasts are proposed
only after incompatible target bindings were actually observed in one mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple

from v3.schemas import GameObservation, ObjectInfo

from .online_semantic_intervention import (
    SemanticInterventionAnchor,
    entity_signature,
    semantic_legacy_signature,
    semantic_transfer_signature,
    structural_role_signature,
    target_object_for_intervention,
)
from .online_state_conditioned_effect import latent_mode_signature
from .online_terminal_objective import TerminalObjectiveHypothesis


class EntityCorrespondenceStatus(str, Enum):
    STABLE = "stable"
    MOVED = "moved"
    TRANSFORMED = "transformed"
    REMOVED = "removed"
    AMBIGUOUS = "ambiguous"


class EntityBindingStatus(str, Enum):
    UNKNOWN = "unknown"
    PROGRESSIVE_CARRIER = "progressive_carrier"
    REGRESSIVE_CARRIER = "regressive_carrier"
    MISBOUND = "misbound"
    NEEDS_CONTROLLED_CONTRAST = "needs_controlled_contrast"


@dataclass(frozen=True)
class EntityCorrespondence:
    track_id: str
    source_object_id: int
    target_object_id: int
    status: EntityCorrespondenceStatus
    confidence: float
    source_entity_signature: str
    target_entity_signature: str = ""
    source_role_signature: str = ""
    target_role_signature: str = ""

    @property
    def target_changed(self) -> bool:
        return self.status in {
            EntityCorrespondenceStatus.MOVED,
            EntityCorrespondenceStatus.TRANSFORMED,
            EntityCorrespondenceStatus.REMOVED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "source_object_id": self.source_object_id,
            "target_object_id": self.target_object_id,
            "status": self.status.value,
            "confidence": round(float(self.confidence), 4),
            "target_changed": self.target_changed,
            "source_entity_signature": self.source_entity_signature,
            "target_entity_signature": self.target_entity_signature,
            "source_role_signature": self.source_role_signature,
            "target_role_signature": self.target_role_signature,
        }


@dataclass
class EntityCausalBindingEvidence:
    option_id: str
    objective_id: str
    mode_signature: str
    action_transfer_signature: str
    legacy_signature: str
    structural_role_signature: str
    attempts: int = 0
    matched_entities: int = 0
    transformed_entities: int = 0
    moved_entities: int = 0
    removed_entities: int = 0
    ambiguous_entities: int = 0
    carrier_progress_events: int = 0
    carrier_regression_events: int = 0
    noncarrier_progress_events: int = 0
    target_stable_events: int = 0
    unsafe_failures: int = 0
    total_carrier_gain: float = 0.0
    total_carrier_regression: float = 0.0
    track_ids: set[str] = field(default_factory=set)
    branches: set[int] = field(default_factory=set)
    contexts: set[str] = field(default_factory=set)

    @property
    def status(self) -> EntityBindingStatus:
        if self.carrier_progress_events and self.carrier_regression_events:
            return EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST
        if self.carrier_progress_events:
            return EntityBindingStatus.PROGRESSIVE_CARRIER
        if self.carrier_regression_events:
            return EntityBindingStatus.REGRESSIVE_CARRIER
        if self.noncarrier_progress_events:
            return EntityBindingStatus.MISBOUND
        return EntityBindingStatus.UNKNOWN

    @property
    def expected_gain(self) -> float:
        return (
            self.total_carrier_gain
            - self.total_carrier_regression
            - 2.0 * self.unsafe_failures
        ) / (self.attempts + 1.0)

    @property
    def confidence(self) -> float:
        directional = self.carrier_progress_events + self.carrier_regression_events
        return directional / (self.attempts + 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "objective_id": self.objective_id,
            "mode_signature": self.mode_signature,
            "action_transfer_signature": self.action_transfer_signature,
            "legacy_signature": self.legacy_signature,
            "structural_role_signature": self.structural_role_signature,
            "status": self.status.value,
            "attempts": self.attempts,
            "matched_entities": self.matched_entities,
            "transformed_entities": self.transformed_entities,
            "moved_entities": self.moved_entities,
            "removed_entities": self.removed_entities,
            "ambiguous_entities": self.ambiguous_entities,
            "carrier_progress_events": self.carrier_progress_events,
            "carrier_regression_events": self.carrier_regression_events,
            "noncarrier_progress_events": self.noncarrier_progress_events,
            "target_stable_events": self.target_stable_events,
            "unsafe_failures": self.unsafe_failures,
            "total_carrier_gain": round(float(self.total_carrier_gain), 4),
            "total_carrier_regression": round(
                float(self.total_carrier_regression),
                4,
            ),
            "expected_gain": round(float(self.expected_gain), 4),
            "confidence": round(float(self.confidence), 4),
            "track_ids": sorted(self.track_ids),
            "branches": sorted(self.branches),
            "contexts": sorted(self.contexts),
        }


@dataclass(frozen=True)
class EntityBindingPrediction:
    option_id: str
    objective_id: str
    mode_signature: str
    action_signature: str
    action_transfer_signature: str
    legacy_signature: str
    status: EntityBindingStatus
    expected_gain: float
    confidence: float
    compatible: bool
    controlled_contrast: bool
    conflict_observed: bool
    track_id: str
    reason: str

    @property
    def selection_rank(self) -> int:
        return {
            EntityBindingStatus.PROGRESSIVE_CARRIER: 6,
            EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST: 4,
            EntityBindingStatus.UNKNOWN: 2,
            EntityBindingStatus.MISBOUND: 0,
            EntityBindingStatus.REGRESSIVE_CARRIER: -1,
        }[self.status]


class OnlineEntityCausalBindingStore:
    """Track targeted entities and learn which structural argument carries gain."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._evidence: Dict[
            Tuple[str, str, str, str],
            EntityCausalBindingEvidence,
        ] = {}
        self._track_by_frame_object: Dict[Tuple[int, int, int], str] = {}
        self._next_track_id = 1
        self._observations = 0
        self._matched_entities = 0
        self._transformed_entities = 0
        self._moved_entities = 0
        self._removed_entities = 0
        self._ambiguous_entities = 0
        self._tracks_created = 0
        self._tracks_reused = 0
        self._carrier_progress_events = 0
        self._carrier_regression_events = 0
        self._noncarrier_progress_events = 0
        self._predictions = 0
        self._contrast_predictions = 0
        self._contrast_selections = 0
        self._progressive_selections = 0
        self._blocked_misbound_actions = 0

    def observe(
        self,
        *,
        option_id: str,
        objective: TerminalObjectiveHypothesis,
        observation_before: GameObservation,
        observation_after: GameObservation,
        action_name: str,
        action_data: Mapping[str, Any],
        action_signature: str,
        anchor: SemanticInterventionAnchor,
        branch_index: int,
        context_signature: str,
        unsafe: bool = False,
    ) -> Dict[str, Any]:
        if not self.enabled or not anchor.anchored:
            return {"observed": False}
        source = target_object_for_intervention(observation_before, action_data)
        if source is None:
            return {"observed": False}
        mode = latent_mode_signature(observation_before, objective)
        transfer = semantic_transfer_signature(str(action_signature))
        legacy = semantic_legacy_signature(str(action_signature))
        track_id, reused = self._source_track(
            observation_before,
            source,
            branch_index,
        )
        correspondence = _match_target_entity(
            source=source,
            observation_before=observation_before,
            observation_after=observation_after,
            track_id=track_id,
        )
        if reused:
            self._tracks_reused += 1
        if correspondence.target_object_id >= 0 and correspondence.status != (
            EntityCorrespondenceStatus.AMBIGUOUS
        ):
            self._track_by_frame_object[(
                int(branch_index),
                int(observation_after.grid_hash),
                int(correspondence.target_object_id),
            )] = track_id

        key = (str(option_id), objective.objective_id, mode, transfer)
        evidence = self._evidence.get(key)
        if evidence is None:
            evidence = EntityCausalBindingEvidence(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                action_transfer_signature=transfer,
                legacy_signature=legacy,
                structural_role_signature=anchor.structural_role_signature,
            )
            self._evidence[key] = evidence
        evidence.attempts += 1
        evidence.track_ids.add(track_id)
        evidence.branches.add(int(branch_index))
        evidence.contexts.add(str(context_signature))
        evidence.unsafe_failures += int(bool(unsafe))
        self._record_correspondence(evidence, correspondence)

        before = objective.distance(observation_before)
        after = objective.distance(observation_after)
        gain = 0.0 if before is None or after is None else float(before) - float(after)
        resolved = correspondence.status != EntityCorrespondenceStatus.AMBIGUOUS
        carrier = resolved and correspondence.target_changed
        if gain > 0.0 and carrier:
            evidence.carrier_progress_events += 1
            evidence.total_carrier_gain += gain
            self._carrier_progress_events += 1
        elif gain > 0.0 and resolved:
            evidence.noncarrier_progress_events += 1
            self._noncarrier_progress_events += 1
        elif gain < 0.0 and carrier:
            evidence.carrier_regression_events += 1
            evidence.total_carrier_regression += abs(gain)
            self._carrier_regression_events += 1
        if resolved and not carrier:
            evidence.target_stable_events += 1
        self._observations += 1
        return {
            "observed": True,
            "action_name": str(action_name),
            "mode_signature": mode,
            "action_transfer_signature": transfer,
            "legacy_signature": legacy,
            "binding_status": evidence.status.value,
            "gain": gain,
            "carrier": carrier,
            "correspondence": correspondence.to_dict(),
            "conflict_observed": self._group_conflict(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                legacy_signature=legacy,
            ),
        }

    def predict(
        self,
        *,
        option_id: str,
        objective: TerminalObjectiveHypothesis,
        observation: GameObservation,
        action_signature: str,
        track_id: str = "",
        record_prediction: bool = True,
    ) -> EntityBindingPrediction:
        mode = latent_mode_signature(observation, objective)
        transfer = semantic_transfer_signature(str(action_signature))
        legacy = semantic_legacy_signature(str(action_signature))
        evidence = self._evidence.get((
            str(option_id),
            objective.objective_id,
            mode,
            transfer,
        ))
        conflict = self._group_conflict(
            option_id=str(option_id),
            objective_id=objective.objective_id,
            mode_signature=mode,
            legacy_signature=legacy,
        )
        if record_prediction and self.enabled:
            self._predictions += 1
        if evidence is None:
            status = (
                EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST
                if conflict else EntityBindingStatus.UNKNOWN
            )
            if record_prediction and self.enabled and status == (
                EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST
            ):
                self._contrast_predictions += 1
            return EntityBindingPrediction(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                action_signature=str(action_signature),
                action_transfer_signature=transfer,
                legacy_signature=legacy,
                status=status,
                expected_gain=0.0,
                confidence=0.0,
                compatible=True,
                controlled_contrast=status == (
                    EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST
                ),
                conflict_observed=conflict,
                track_id=str(track_id),
                reason=(
                    "observed carrier conflict requires this untested structural "
                    "argument as a same-mode control"
                    if conflict
                    else "no causal target-binding evidence for this argument"
                ),
            )

        status = evidence.status
        controlled_contrast = False
        if (
            conflict
            and status == EntityBindingStatus.MISBOUND
            and evidence.attempts < 2
        ):
            status = EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST
            controlled_contrast = True
            if record_prediction and self.enabled:
                self._contrast_predictions += 1
        compatible = status not in {
            EntityBindingStatus.REGRESSIVE_CARRIER,
            EntityBindingStatus.MISBOUND,
        }
        return EntityBindingPrediction(
            option_id=str(option_id),
            objective_id=objective.objective_id,
            mode_signature=mode,
            action_signature=str(action_signature),
            action_transfer_signature=transfer,
            legacy_signature=legacy,
            status=status,
            expected_gain=evidence.expected_gain,
            confidence=evidence.confidence,
            compatible=compatible,
            controlled_contrast=controlled_contrast,
            conflict_observed=conflict,
            track_id=str(track_id),
            reason={
                EntityBindingStatus.PROGRESSIVE_CARRIER: (
                    "this structural argument carried observed objective progress"
                ),
                EntityBindingStatus.REGRESSIVE_CARRIER: (
                    "this structural argument carried observed objective regression"
                ),
                EntityBindingStatus.MISBOUND: (
                    "objective progress occurred while the targeted entity stayed stable"
                ),
                EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST: (
                    "same-mode target bindings conflict; run a bounded control"
                ),
                EntityBindingStatus.UNKNOWN: (
                    "target correspondence observed without directional carrier evidence"
                ),
            }[status],
        )

    def note_selection(self, prediction: EntityBindingPrediction) -> None:
        if prediction.status == EntityBindingStatus.PROGRESSIVE_CARRIER:
            self._progressive_selections += 1
        if prediction.controlled_contrast:
            self._contrast_selections += 1

    def note_blocked(self, prediction: EntityBindingPrediction) -> None:
        if prediction.status in {
            EntityBindingStatus.REGRESSIVE_CARRIER,
            EntityBindingStatus.MISBOUND,
        }:
            self._blocked_misbound_actions += 1

    def track_id_for(
        self,
        observation: GameObservation,
        anchor: SemanticInterventionAnchor,
        branch_index: int,
    ) -> str:
        if not anchor.anchored or anchor.source_object_id < 0:
            return ""
        return self._track_by_frame_object.get((
            int(branch_index),
            int(observation.grid_hash),
            int(anchor.source_object_id),
        ), "")

    def summary(self) -> Dict[str, Any]:
        evidence = sorted(
            self._evidence.values(),
            key=lambda item: (
                item.option_id,
                item.objective_id,
                item.mode_signature,
                item.action_transfer_signature,
            ),
        )
        return {
            "enabled": self.enabled,
            "observations": self._observations,
            "matched_entities": self._matched_entities,
            "transformed_entities": self._transformed_entities,
            "moved_entities": self._moved_entities,
            "removed_entities": self._removed_entities,
            "ambiguous_entities": self._ambiguous_entities,
            "tracks_created": self._tracks_created,
            "tracks_reused": self._tracks_reused,
            "binding_models": len(evidence),
            "carrier_progress_events": self._carrier_progress_events,
            "carrier_regression_events": self._carrier_regression_events,
            "noncarrier_progress_events": self._noncarrier_progress_events,
            "binding_conflicts": self._binding_conflict_count(evidence),
            "predictions": self._predictions,
            "controlled_contrast_predictions": self._contrast_predictions,
            "controlled_contrast_selections": self._contrast_selections,
            "progressive_carrier_selections": self._progressive_selections,
            "blocked_misbound_actions": self._blocked_misbound_actions,
            "hypotheses": [item.to_dict() for item in evidence],
        }

    def _source_track(
        self,
        observation: GameObservation,
        source: ObjectInfo,
        branch_index: int,
    ) -> tuple[str, bool]:
        key = (
            int(branch_index),
            int(observation.grid_hash),
            int(source.object_id),
        )
        existing = self._track_by_frame_object.get(key)
        if existing:
            return existing, True
        track_id = f"entity-track:{self._next_track_id}"
        self._next_track_id += 1
        self._track_by_frame_object[key] = track_id
        self._tracks_created += 1
        return track_id, False

    def _record_correspondence(
        self,
        evidence: EntityCausalBindingEvidence,
        correspondence: EntityCorrespondence,
    ) -> None:
        status = correspondence.status
        if status == EntityCorrespondenceStatus.AMBIGUOUS:
            evidence.ambiguous_entities += 1
            self._ambiguous_entities += 1
            return
        if status == EntityCorrespondenceStatus.REMOVED:
            evidence.removed_entities += 1
            self._removed_entities += 1
            return
        evidence.matched_entities += 1
        self._matched_entities += 1
        if status == EntityCorrespondenceStatus.TRANSFORMED:
            evidence.transformed_entities += 1
            self._transformed_entities += 1
        elif status == EntityCorrespondenceStatus.MOVED:
            evidence.moved_entities += 1
            self._moved_entities += 1

    def _group_conflict(
        self,
        *,
        option_id: str,
        objective_id: str,
        mode_signature: str,
        legacy_signature: str,
    ) -> bool:
        group = [
            item for item in self._evidence.values()
            if item.option_id == str(option_id)
            and item.objective_id == str(objective_id)
            and item.mode_signature == str(mode_signature)
            and item.legacy_signature == str(legacy_signature)
        ]
        progressive = {
            item.action_transfer_signature
            for item in group
            if item.carrier_progress_events > 0
        }
        incompatible = {
            item.action_transfer_signature
            for item in group
            if item.carrier_regression_events > 0
            or item.noncarrier_progress_events > 0
        }
        return any(left != right for left in progressive for right in incompatible)

    @classmethod
    def _binding_conflict_count(
        cls,
        evidence: Sequence[EntityCausalBindingEvidence],
    ) -> int:
        groups: Dict[
            Tuple[str, str, str, str],
            list[EntityCausalBindingEvidence],
        ] = {}
        for item in evidence:
            key = (
                item.option_id,
                item.objective_id,
                item.mode_signature,
                item.legacy_signature,
            )
            groups.setdefault(key, []).append(item)
        conflicts = 0
        for group in groups.values():
            progressive = {
                item.action_transfer_signature
                for item in group
                if item.carrier_progress_events > 0
            }
            incompatible = {
                item.action_transfer_signature
                for item in group
                if item.carrier_regression_events > 0
                or item.noncarrier_progress_events > 0
            }
            conflicts += int(any(
                left != right for left in progressive for right in incompatible
            ))
        return conflicts


def _match_target_entity(
    *,
    source: ObjectInfo,
    observation_before: GameObservation,
    observation_after: GameObservation,
    track_id: str,
) -> EntityCorrespondence:
    candidates = sorted(
        (
            (_correspondence_score(source, target), target)
            for target in observation_after.objects
        ),
        key=lambda item: (item[0], -item[1].object_id),
        reverse=True,
    )
    source_entity = entity_signature(source)
    source_role = structural_role_signature(source, observation_before)
    if not candidates or candidates[0][0] < 1.5:
        return EntityCorrespondence(
            track_id=track_id,
            source_object_id=int(source.object_id),
            target_object_id=-1,
            status=EntityCorrespondenceStatus.REMOVED,
            confidence=1.0,
            source_entity_signature=source_entity,
            source_role_signature=source_role,
        )
    best_score, target = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else float("-inf")
    if second_score >= 1.5 and best_score - second_score < 0.35:
        return EntityCorrespondence(
            track_id=track_id,
            source_object_id=int(source.object_id),
            target_object_id=-1,
            status=EntityCorrespondenceStatus.AMBIGUOUS,
            confidence=0.0,
            source_entity_signature=source_entity,
            source_role_signature=source_role,
        )
    same_shape = _normalized_cells(source) == _normalized_cells(target)
    if int(source.value) != int(target.value) or not same_shape:
        status = EntityCorrespondenceStatus.TRANSFORMED
    elif tuple(source.center) != tuple(target.center):
        status = EntityCorrespondenceStatus.MOVED
    else:
        status = EntityCorrespondenceStatus.STABLE
    margin = best_score if second_score == float("-inf") else best_score - second_score
    confidence = max(0.0, min(1.0, margin / max(1.0, abs(best_score))))
    return EntityCorrespondence(
        track_id=track_id,
        source_object_id=int(source.object_id),
        target_object_id=int(target.object_id),
        status=status,
        confidence=confidence,
        source_entity_signature=source_entity,
        target_entity_signature=entity_signature(target),
        source_role_signature=source_role,
        target_role_signature=structural_role_signature(target, observation_after),
    )


def _correspondence_score(source: ObjectInfo, target: ObjectInfo) -> float:
    source_cells = set(source.cells)
    target_cells = set(target.cells)
    overlap = len(source_cells & target_cells) / max(1, len(source_cells | target_cells))
    same_shape = _normalized_cells(source) == _normalized_cells(target)
    area_similarity = min(source.area, target.area) / max(1, source.area, target.area)
    same_color = int(source.value) == int(target.value)
    distance = math.dist(source.center, target.center)
    return (
        5.0 * overlap
        + 2.0 * int(same_shape)
        + area_similarity
        + 0.5 * int(same_color)
        - 0.1 * min(20.0, distance)
    )


def _normalized_cells(obj: ObjectInfo) -> tuple[tuple[int, int], ...]:
    row0, col0, _, _ = obj.bbox
    return tuple(sorted(
        (int(row) - int(row0), int(col) - int(col0))
        for row, col in obj.cells
    ))


__all__ = [
    "EntityBindingPrediction",
    "EntityBindingStatus",
    "EntityCausalBindingEvidence",
    "EntityCorrespondence",
    "EntityCorrespondenceStatus",
    "OnlineEntityCausalBindingStore",
]
