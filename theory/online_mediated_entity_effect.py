"""Online induction of indirect entity carriers for causal-option effects.

SAGE.8u can reject credit when the clicked entity stayed stable.  SAGE.8v
extends that observation to the whole scene: it tracks all entities, describes
changed non-target entities relative to the acted entity, and intersects
candidate carrier sets across controlled online repetitions.  A mediated
carrier is supported only by concordant progress in independent contexts;
terminal evaluation is never consulted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple

from v3.schemas import GameObservation, ObjectInfo

from .online_mediated_abstraction import (
    MediatedAbstractionHypothesis,
    induce_mediated_abstraction,
)
from .online_semantic_intervention import (
    SemanticInterventionAnchor,
    entity_signature,
    semantic_transfer_signature,
    structural_role_signature,
    target_object_for_intervention,
)
from .online_state_conditioned_effect import latent_mode_signature
from .online_terminal_objective import TerminalObjectiveHypothesis


class SceneCorrespondenceStatus(str, Enum):
    STABLE = "stable"
    MOVED = "moved"
    TRANSFORMED = "transformed"
    REMOVED = "removed"
    APPEARED = "appeared"
    AMBIGUOUS = "ambiguous"


class MediatedEffectStatus(str, Enum):
    UNKNOWN = "unknown"
    NEEDS_MEDIATOR_CONTRAST = "needs_mediator_contrast"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class SceneEntityCorrespondence:
    track_id: str
    source_object_id: int
    target_object_id: int
    status: SceneCorrespondenceStatus
    confidence: float
    source_entity_signature: str = ""
    target_entity_signature: str = ""
    source_role_signature: str = ""
    target_role_signature: str = ""

    @property
    def changed(self) -> bool:
        return self.status in {
            SceneCorrespondenceStatus.MOVED,
            SceneCorrespondenceStatus.TRANSFORMED,
            SceneCorrespondenceStatus.REMOVED,
            SceneCorrespondenceStatus.APPEARED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "source_object_id": self.source_object_id,
            "target_object_id": self.target_object_id,
            "status": self.status.value,
            "confidence": round(float(self.confidence), 4),
            "changed": self.changed,
            "source_entity_signature": self.source_entity_signature,
            "target_entity_signature": self.target_entity_signature,
            "source_role_signature": self.source_role_signature,
            "target_role_signature": self.target_role_signature,
        }


@dataclass(frozen=True)
class MediatedCarrierCandidate:
    track_id: str
    signature: str
    relation_signature: str
    transition_status: SceneCorrespondenceStatus
    source_object_id: int
    target_object_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "signature": self.signature,
            "relation_signature": self.relation_signature,
            "transition_status": self.transition_status.value,
            "source_object_id": self.source_object_id,
            "target_object_id": self.target_object_id,
        }


@dataclass
class MediatedEntityEffectEvidence:
    option_id: str
    objective_id: str
    mode_signature: str
    action_transfer_signature: str
    objective_colors: Tuple[int, ...] = ()
    anti_unification_enabled: bool = True
    attempts: int = 0
    progress_events: int = 0
    regression_events: int = 0
    stall_events: int = 0
    ambiguous_progress_events: int = 0
    no_candidate_progress_events: int = 0
    direct_target_progress_events: int = 0
    total_gain: float = 0.0
    total_regression: float = 0.0
    progress_contexts: set[str] = field(default_factory=set)
    progress_branches: set[int] = field(default_factory=set)
    contexts: set[str] = field(default_factory=set)
    branches: set[int] = field(default_factory=set)
    progress_candidate_sets: list[Tuple[str, ...]] = field(default_factory=list)
    control_candidate_sets: list[Tuple[str, ...]] = field(default_factory=list)
    regression_candidate_sets: list[Tuple[str, ...]] = field(default_factory=list)
    candidate_supports: Dict[str, int] = field(default_factory=dict)
    candidate_controls: Dict[str, int] = field(default_factory=dict)
    candidate_regressions: Dict[str, int] = field(default_factory=dict)
    candidate_examples: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def candidate_intersection(self) -> set[str]:
        sets = [set(items) for items in self.progress_candidate_sets if items]
        if not sets:
            return set()
        intersection = set(sets[0])
        for items in sets[1:]:
            intersection.intersection_update(items)
        return {
            signature
            for signature in intersection
            if self.candidate_supports.get(signature, 0)
            > self.candidate_controls.get(signature, 0)
        }

    @property
    def supported_mediator_signature(self) -> str:
        candidates = self.candidate_intersection
        if (
            len(candidates) == 1
            and self.progress_events >= 2
            and len(self.progress_contexts) >= 2
        ):
            candidate = next(iter(candidates))
            if self.candidate_supports.get(candidate, 0) >= 2:
                return candidate
        abstraction = self.mediator_abstraction
        return (
            abstraction.signature
            if abstraction is not None and abstraction.supported
            else ""
        )

    @property
    def mediator_abstraction(self) -> MediatedAbstractionHypothesis | None:
        if not self.anti_unification_enabled:
            return None
        return induce_mediated_abstraction(
            self.progress_candidate_sets,
            control_candidate_sets=self.control_candidate_sets,
            regression_candidate_sets=self.regression_candidate_sets,
            preferred_colors=self.objective_colors,
        )

    @property
    def supported_mediator_is_abstract(self) -> bool:
        return self.supported_mediator_signature.startswith(
            "mediated-abstract::"
        )

    @property
    def active_candidate_signatures(self) -> Tuple[str, ...]:
        exact = tuple(sorted(self.candidate_intersection))
        if exact:
            return exact
        abstraction = self.mediator_abstraction
        return () if abstraction is None else (abstraction.signature,)

    @property
    def status(self) -> MediatedEffectStatus:
        if self.supported_mediator_signature:
            return MediatedEffectStatus.SUPPORTED
        if self.progress_events > 0:
            if (
                len(self.progress_candidate_sets) >= 2
                and not self.candidate_intersection
                and self.mediator_abstraction is None
            ):
                return MediatedEffectStatus.CONTRADICTED
            return MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST
        if self.regression_events > 0:
            return MediatedEffectStatus.CONTRADICTED
        return MediatedEffectStatus.UNKNOWN

    @property
    def expected_gain(self) -> float:
        return (
            self.total_gain - self.total_regression
        ) / (self.attempts + 1.0)

    @property
    def confidence(self) -> float:
        directional = self.progress_events + self.regression_events
        return directional / (self.attempts + 1.0)

    def to_dict(self) -> Dict[str, Any]:
        candidates = sorted(
            set(self.candidate_supports)
            | set(self.candidate_controls)
            | set(self.candidate_regressions)
        )
        abstraction = self.mediator_abstraction
        return {
            "option_id": self.option_id,
            "objective_id": self.objective_id,
            "mode_signature": self.mode_signature,
            "action_transfer_signature": self.action_transfer_signature,
            "status": self.status.value,
            "attempts": self.attempts,
            "progress_events": self.progress_events,
            "regression_events": self.regression_events,
            "stall_events": self.stall_events,
            "ambiguous_progress_events": self.ambiguous_progress_events,
            "no_candidate_progress_events": self.no_candidate_progress_events,
            "direct_target_progress_events": self.direct_target_progress_events,
            "total_gain": round(float(self.total_gain), 4),
            "total_regression": round(float(self.total_regression), 4),
            "expected_gain": round(float(self.expected_gain), 4),
            "confidence": round(float(self.confidence), 4),
            "progress_contexts": sorted(self.progress_contexts),
            "progress_branches": sorted(self.progress_branches),
            "contexts": sorted(self.contexts),
            "branches": sorted(self.branches),
            "progress_candidate_sets": [
                list(items) for items in self.progress_candidate_sets
            ],
            "control_candidate_sets": [
                list(items) for items in self.control_candidate_sets
            ],
            "regression_candidate_sets": [
                list(items) for items in self.regression_candidate_sets
            ],
            "candidate_intersection": sorted(self.candidate_intersection),
            "active_candidate_signatures": list(
                self.active_candidate_signatures
            ),
            "supported_mediator_signature": (
                self.supported_mediator_signature
            ),
            "supported_mediator_is_abstract": (
                self.supported_mediator_is_abstract
            ),
            "mediator_abstraction": (
                None if abstraction is None else abstraction.to_dict()
            ),
            "candidate_hyperedges": [
                {
                    "signature": signature,
                    "supports": self.candidate_supports.get(signature, 0),
                    "controls": self.candidate_controls.get(signature, 0),
                    "regressions": self.candidate_regressions.get(signature, 0),
                    "example": dict(self.candidate_examples.get(signature, {})),
                }
                for signature in candidates
            ],
        }


@dataclass(frozen=True)
class MediatedEffectPrediction:
    option_id: str
    objective_id: str
    mode_signature: str
    action_signature: str
    action_transfer_signature: str
    status: MediatedEffectStatus
    expected_gain: float
    confidence: float
    compatible: bool
    controlled_contrast: bool
    supported_mediator_signature: str
    candidate_mediator_signatures: Tuple[str, ...]
    reason: str
    mediator_abstraction: bool = False
    mediator_abstraction_specificity: int = 0
    mediator_abstraction_features: Tuple[Tuple[str, str], ...] = ()

    @property
    def selection_rank(self) -> int:
        return {
            MediatedEffectStatus.SUPPORTED: 7,
            MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST: 4,
            MediatedEffectStatus.UNKNOWN: 2,
            MediatedEffectStatus.CONTRADICTED: -1,
        }[self.status]


class OnlineMediatedEntityEffectStore:
    """Induce indirect action-target -> relation -> affected-entity edges."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        enable_anti_unification: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.enable_anti_unification = bool(enable_anti_unification)
        self._evidence: Dict[
            Tuple[str, str, str, str],
            MediatedEntityEffectEvidence,
        ] = {}
        self._track_by_frame_object: Dict[Tuple[int, int, int], str] = {}
        self._next_track_id = 1
        self._observations = 0
        self._scene_correspondences = 0
        self._changed_entities = 0
        self._moved_entities = 0
        self._transformed_entities = 0
        self._removed_entities = 0
        self._appeared_entities = 0
        self._ambiguous_entities = 0
        self._tracks_created = 0
        self._tracks_reused = 0
        self._progress_with_indirect_candidates = 0
        self._ambiguous_progress_candidate_sets = 0
        self._no_candidate_progress_events = 0
        self._direct_target_progress_events = 0
        self._predictions = 0
        self._supported_predictions = 0
        self._contrast_predictions = 0
        self._supported_selections = 0
        self._contrast_selections = 0
        self._blocked_contradicted_actions = 0

    def observe(
        self,
        *,
        option_id: str,
        objective: TerminalObjectiveHypothesis,
        observation_before: GameObservation,
        observation_after: GameObservation,
        action_data: Mapping[str, Any],
        action_signature: str,
        anchor: SemanticInterventionAnchor,
        branch_index: int,
        context_signature: str,
    ) -> Dict[str, Any]:
        if not self.enabled or not anchor.anchored:
            return {"observed": False}
        acted = target_object_for_intervention(observation_before, action_data)
        if acted is None:
            return {"observed": False}
        correspondences = self._match_scene(
            observation_before,
            observation_after,
            branch_index,
        )
        candidates = self._carrier_candidates(
            acted=acted,
            observation_before=observation_before,
            observation_after=observation_after,
            correspondences=correspondences,
        )
        ambiguous = any(
            item.status == SceneCorrespondenceStatus.AMBIGUOUS
            and item.source_object_id != acted.object_id
            for item in correspondences
        )
        acted_correspondence = next(
            (
                item for item in correspondences
                if item.source_object_id == acted.object_id
            ),
            None,
        )
        target_stable = bool(
            acted_correspondence is not None
            and acted_correspondence.status == SceneCorrespondenceStatus.STABLE
        )
        before_distance = objective.distance(observation_before)
        after_distance = objective.distance(observation_after)
        gain = (
            0.0
            if before_distance is None or after_distance is None
            else float(before_distance) - float(after_distance)
        )
        mode = latent_mode_signature(observation_before, objective)
        transfer = semantic_transfer_signature(str(action_signature))
        key = (str(option_id), objective.objective_id, mode, transfer)
        evidence = self._evidence.get(key)
        if evidence is None:
            evidence = MediatedEntityEffectEvidence(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                action_transfer_signature=transfer,
                objective_colors=tuple(sorted({
                    int(color)
                    for color in (objective.source_color, objective.target_color)
                    if color is not None
                })),
                anti_unification_enabled=self.enable_anti_unification,
            )
            self._evidence[key] = evidence
        evidence.attempts += 1
        evidence.contexts.add(str(context_signature))
        evidence.branches.add(int(branch_index))
        signatures = tuple(sorted(
            item.signature for item in candidates
        )) if target_stable else ()
        for candidate in candidates:
            evidence.candidate_examples.setdefault(
                candidate.signature,
                candidate.to_dict(),
            )
        if gain > 0.0 and not target_stable:
            evidence.direct_target_progress_events += 1
            self._direct_target_progress_events += 1
        elif gain > 0.0:
            evidence.progress_events += 1
            evidence.total_gain += gain
            evidence.progress_contexts.add(str(context_signature))
            evidence.progress_branches.add(int(branch_index))
            if ambiguous:
                evidence.ambiguous_progress_events += 1
                self._ambiguous_progress_candidate_sets += 1
            elif signatures:
                evidence.progress_candidate_sets.append(signatures)
                for signature in signatures:
                    evidence.candidate_supports[signature] = (
                        evidence.candidate_supports.get(signature, 0) + 1
                    )
                self._progress_with_indirect_candidates += 1
            else:
                evidence.no_candidate_progress_events += 1
                self._no_candidate_progress_events += 1
        elif gain < 0.0:
            evidence.regression_events += 1
            evidence.total_regression += abs(gain)
            if signatures:
                evidence.regression_candidate_sets.append(signatures)
            for signature in signatures:
                evidence.candidate_regressions[signature] = (
                    evidence.candidate_regressions.get(signature, 0) + 1
                )
        else:
            evidence.stall_events += 1
            if signatures:
                evidence.control_candidate_sets.append(signatures)
            for signature in signatures:
                evidence.candidate_controls[signature] = (
                    evidence.candidate_controls.get(signature, 0) + 1
                )
        self._observations += 1
        self._record_correspondence_metrics(correspondences)
        return {
            "observed": True,
            "mode_signature": mode,
            "action_transfer_signature": transfer,
            "gain": gain,
            "status": evidence.status.value,
            "supported_mediator_signature": (
                evidence.supported_mediator_signature
            ),
            "supported_mediator_is_abstract": (
                evidence.supported_mediator_is_abstract
            ),
            "mediator_abstraction": (
                None
                if evidence.mediator_abstraction is None
                else evidence.mediator_abstraction.to_dict()
            ),
            "candidate_mediator_signatures": list(signatures),
            "ambiguous_scene_correspondence": ambiguous,
            "target_stable_for_mediation": target_stable,
            "scene_correspondences": [
                item.to_dict() for item in correspondences
            ],
            "carrier_candidates": [item.to_dict() for item in candidates],
        }

    def predict(
        self,
        *,
        option_id: str,
        objective: TerminalObjectiveHypothesis,
        observation: GameObservation,
        action_signature: str,
        record_prediction: bool = True,
    ) -> MediatedEffectPrediction:
        mode = latent_mode_signature(observation, objective)
        transfer = semantic_transfer_signature(str(action_signature))
        evidence = self._evidence.get((
            str(option_id),
            objective.objective_id,
            mode,
            transfer,
        ))
        if record_prediction and self.enabled:
            self._predictions += 1
        if evidence is None:
            return MediatedEffectPrediction(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                action_signature=str(action_signature),
                action_transfer_signature=transfer,
                status=MediatedEffectStatus.UNKNOWN,
                expected_gain=0.0,
                confidence=0.0,
                compatible=True,
                controlled_contrast=False,
                supported_mediator_signature="",
                candidate_mediator_signatures=(),
                reason="no observed indirect carrier evidence in this mode",
            )
        status = evidence.status
        abstraction = evidence.mediator_abstraction
        if record_prediction and self.enabled:
            if status == MediatedEffectStatus.SUPPORTED:
                self._supported_predictions += 1
            elif status == MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST:
                self._contrast_predictions += 1
        return MediatedEffectPrediction(
            option_id=str(option_id),
            objective_id=objective.objective_id,
            mode_signature=mode,
            action_signature=str(action_signature),
            action_transfer_signature=transfer,
            status=status,
            expected_gain=evidence.expected_gain,
            confidence=evidence.confidence,
            compatible=status != MediatedEffectStatus.CONTRADICTED,
            controlled_contrast=(
                status == MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST
            ),
            supported_mediator_signature=(
                evidence.supported_mediator_signature
            ),
            candidate_mediator_signatures=tuple(
                evidence.active_candidate_signatures
            ),
            reason={
                MediatedEffectStatus.SUPPORTED: (
                    "independent online contexts isolate one indirect carrier"
                ),
                MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST: (
                    "repeat this intervention in a controlled context to "
                    "separate indirect carrier candidates"
                ),
                MediatedEffectStatus.CONTRADICTED: (
                    "progressive transitions share no stable indirect carrier"
                ),
                MediatedEffectStatus.UNKNOWN: (
                    "scene changes observed without indirect progress evidence"
                ),
            }[status],
            mediator_abstraction=bool(
                evidence.supported_mediator_is_abstract
                or (
                    abstraction is not None
                    and not evidence.candidate_intersection
                )
            ),
            mediator_abstraction_specificity=(
                0 if abstraction is None else abstraction.specificity
            ),
            mediator_abstraction_features=(
                () if abstraction is None else abstraction.features
            ),
        )

    def note_selection(self, prediction: MediatedEffectPrediction) -> None:
        if prediction.status == MediatedEffectStatus.SUPPORTED:
            self._supported_selections += 1
        elif prediction.controlled_contrast:
            self._contrast_selections += 1

    def note_blocked(self, prediction: MediatedEffectPrediction) -> None:
        if prediction.status == MediatedEffectStatus.CONTRADICTED:
            self._blocked_contradicted_actions += 1

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
            "anti_unification_enabled": self.enable_anti_unification,
            "observations": self._observations,
            "scene_correspondences": self._scene_correspondences,
            "changed_entities": self._changed_entities,
            "moved_entities": self._moved_entities,
            "transformed_entities": self._transformed_entities,
            "removed_entities": self._removed_entities,
            "appeared_entities": self._appeared_entities,
            "ambiguous_entities": self._ambiguous_entities,
            "tracks_created": self._tracks_created,
            "tracks_reused": self._tracks_reused,
            "progress_with_indirect_candidates": (
                self._progress_with_indirect_candidates
            ),
            "ambiguous_progress_candidate_sets": (
                self._ambiguous_progress_candidate_sets
            ),
            "no_candidate_progress_events": self._no_candidate_progress_events,
            "direct_target_progress_events": (
                self._direct_target_progress_events
            ),
            "mediated_effect_models": len(evidence),
            "supported_hyperedges": sum(
                bool(item.supported_mediator_signature) for item in evidence
            ),
            "abstract_hyperedge_hypotheses": sum(
                item.mediator_abstraction is not None for item in evidence
            ),
            "supported_abstract_hyperedges": sum(
                item.supported_mediator_is_abstract for item in evidence
            ),
            "abstract_control_contexts": sum(
                0
                if item.mediator_abstraction is None
                else item.mediator_abstraction.control_contexts
                for item in evidence
            ),
            "abstract_regression_contexts": sum(
                0
                if item.mediator_abstraction is None
                else item.mediator_abstraction.regression_contexts
                for item in evidence
            ),
            "predictions": self._predictions,
            "supported_predictions": self._supported_predictions,
            "controlled_contrast_predictions": self._contrast_predictions,
            "supported_selections": self._supported_selections,
            "controlled_contrast_selections": self._contrast_selections,
            "blocked_contradicted_actions": self._blocked_contradicted_actions,
            "hypotheses": [item.to_dict() for item in evidence],
        }

    def _match_scene(
        self,
        before: GameObservation,
        after: GameObservation,
        branch_index: int,
    ) -> list[SceneEntityCorrespondence]:
        sources = list(before.objects)
        targets = list(after.objects)
        scored = sorted(
            (
                (_correspondence_score(source, target), source, target)
                for source in sources
                for target in targets
            ),
            key=lambda item: (
                item[0],
                -item[1].object_id,
                -item[2].object_id,
            ),
            reverse=True,
        )
        source_scores: Dict[int, list[float]] = {}
        target_scores: Dict[int, list[float]] = {}
        for score, source, target in scored:
            if score < 1.5:
                continue
            source_scores.setdefault(source.object_id, []).append(score)
            target_scores.setdefault(target.object_id, []).append(score)
        ambiguous_sources = {
            object_id
            for object_id, scores in source_scores.items()
            if len(scores) > 1 and scores[0] - scores[1] < 0.35
        }
        ambiguous_targets = {
            object_id
            for object_id, scores in target_scores.items()
            if len(scores) > 1 and scores[0] - scores[1] < 0.35
        }
        ambiguity_linked_targets = {
            target.object_id
            for score, source, target in scored
            if score >= 1.5 and source.object_id in ambiguous_sources
        }
        matched_sources: set[int] = set()
        matched_targets: set[int] = set()
        result: list[SceneEntityCorrespondence] = []
        for score, source, target in scored:
            if score < 1.5:
                break
            if (
                source.object_id in matched_sources
                or target.object_id in matched_targets
                or source.object_id in ambiguous_sources
                or target.object_id in ambiguous_targets
            ):
                continue
            track_id, _reused = self._track_for_source(
                before,
                source,
                branch_index,
            )
            self._track_by_frame_object[(
                int(branch_index),
                int(after.grid_hash),
                int(target.object_id),
            )] = track_id
            matched_sources.add(source.object_id)
            matched_targets.add(target.object_id)
            same_shape = _normalized_cells(source) == _normalized_cells(target)
            if int(source.value) != int(target.value) or not same_shape:
                status = SceneCorrespondenceStatus.TRANSFORMED
            elif tuple(source.center) != tuple(target.center):
                status = SceneCorrespondenceStatus.MOVED
            else:
                status = SceneCorrespondenceStatus.STABLE
            alternatives = source_scores.get(source.object_id, [])
            margin = score if len(alternatives) < 2 else score - alternatives[1]
            result.append(SceneEntityCorrespondence(
                track_id=track_id,
                source_object_id=int(source.object_id),
                target_object_id=int(target.object_id),
                status=status,
                confidence=max(0.0, min(1.0, margin / max(1.0, abs(score)))),
                source_entity_signature=entity_signature(source),
                target_entity_signature=entity_signature(target),
                source_role_signature=structural_role_signature(source, before),
                target_role_signature=structural_role_signature(target, after),
            ))
        for source in sources:
            if source.object_id in matched_sources:
                continue
            track_id, _reused = self._track_for_source(
                before,
                source,
                branch_index,
            )
            ambiguous = source.object_id in ambiguous_sources
            result.append(SceneEntityCorrespondence(
                track_id=track_id,
                source_object_id=int(source.object_id),
                target_object_id=-1,
                status=(
                    SceneCorrespondenceStatus.AMBIGUOUS
                    if ambiguous else SceneCorrespondenceStatus.REMOVED
                ),
                confidence=0.0 if ambiguous else 1.0,
                source_entity_signature=entity_signature(source),
                source_role_signature=structural_role_signature(source, before),
            ))
        for target in targets:
            if (
                target.object_id in matched_targets
                or target.object_id in ambiguous_targets
                or target.object_id in ambiguity_linked_targets
            ):
                continue
            track_id = self._track_for_appearance(
                after,
                target,
                branch_index,
            )
            result.append(SceneEntityCorrespondence(
                track_id=track_id,
                source_object_id=-1,
                target_object_id=int(target.object_id),
                status=SceneCorrespondenceStatus.APPEARED,
                confidence=1.0,
                target_entity_signature=entity_signature(target),
                target_role_signature=structural_role_signature(target, after),
            ))
        return sorted(
            result,
            key=lambda item: (
                item.source_object_id < 0,
                item.source_object_id,
                item.target_object_id,
            ),
        )

    def _carrier_candidates(
        self,
        *,
        acted: ObjectInfo,
        observation_before: GameObservation,
        observation_after: GameObservation,
        correspondences: Sequence[SceneEntityCorrespondence],
    ) -> list[MediatedCarrierCandidate]:
        before_objects = {
            item.object_id: item for item in observation_before.objects
        }
        after_objects = {
            item.object_id: item for item in observation_after.objects
        }
        result = []
        for item in correspondences:
            if (
                not item.changed
                or item.status == SceneCorrespondenceStatus.AMBIGUOUS
                or item.source_object_id == acted.object_id
            ):
                continue
            affected = (
                before_objects.get(item.source_object_id)
                or after_objects.get(item.target_object_id)
            )
            if affected is None:
                continue
            relation = mediated_relation_signature(acted, affected)
            entity = (
                item.source_entity_signature
                or item.target_entity_signature
            )
            role = item.source_role_signature or item.target_role_signature
            signature = "::".join((
                f"mediated:{item.status.value}",
                f"entity:{entity}",
                f"role:{role}",
                f"relation:{relation}",
            ))
            result.append(MediatedCarrierCandidate(
                track_id=item.track_id,
                signature=signature,
                relation_signature=relation,
                transition_status=item.status,
                source_object_id=item.source_object_id,
                target_object_id=item.target_object_id,
            ))
        return sorted(result, key=lambda item: item.signature)

    def _track_for_source(
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
            self._tracks_reused += 1
            return existing, True
        track_id = f"scene-track:{self._next_track_id}"
        self._next_track_id += 1
        self._track_by_frame_object[key] = track_id
        self._tracks_created += 1
        return track_id, False

    def _track_for_appearance(
        self,
        observation: GameObservation,
        target: ObjectInfo,
        branch_index: int,
    ) -> str:
        key = (
            int(branch_index),
            int(observation.grid_hash),
            int(target.object_id),
        )
        existing = self._track_by_frame_object.get(key)
        if existing:
            self._tracks_reused += 1
            return existing
        track_id = f"scene-track:{self._next_track_id}"
        self._next_track_id += 1
        self._track_by_frame_object[key] = track_id
        self._tracks_created += 1
        return track_id

    def _record_correspondence_metrics(
        self,
        correspondences: Sequence[SceneEntityCorrespondence],
    ) -> None:
        self._scene_correspondences += len(correspondences)
        for item in correspondences:
            self._changed_entities += int(item.changed)
            if item.status == SceneCorrespondenceStatus.MOVED:
                self._moved_entities += 1
            elif item.status == SceneCorrespondenceStatus.TRANSFORMED:
                self._transformed_entities += 1
            elif item.status == SceneCorrespondenceStatus.REMOVED:
                self._removed_entities += 1
            elif item.status == SceneCorrespondenceStatus.APPEARED:
                self._appeared_entities += 1
            elif item.status == SceneCorrespondenceStatus.AMBIGUOUS:
                self._ambiguous_entities += 1


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


def mediated_relation_signature(
    acted: ObjectInfo,
    affected: ObjectInfo,
) -> str:
    """Describe a possible acted-entity -> carrier relation pre-transition."""
    acted_cells = set(acted.cells)
    affected_cells = set(affected.cells)
    if acted_cells & affected_cells:
        proximity = "overlap"
    else:
        gap = _bbox_gap(acted, affected)
        proximity = "adjacent" if gap <= 1 else ("near" if gap <= 4 else "far")
    row_delta = float(affected.center[0]) - float(acted.center[0])
    col_delta = float(affected.center[1]) - float(acted.center[1])
    vertical = "same-row" if abs(row_delta) < 0.5 else (
        "below" if row_delta > 0 else "above"
    )
    horizontal = "same-col" if abs(col_delta) < 0.5 else (
        "right" if col_delta > 0 else "left"
    )
    aligned = abs(row_delta) < 0.5 or abs(col_delta) < 0.5
    color_relation = (
        "same-color" if int(acted.value) == int(affected.value)
        else "different-color"
    )
    return ":".join((
        proximity,
        vertical,
        horizontal,
        "aligned" if aligned else "offset",
        color_relation,
    ))


def prospective_mediator_signature(
    acted: ObjectInfo,
    candidate: ObjectInfo,
    observation: GameObservation,
    *,
    expected_change: str,
) -> str:
    """Build a counterfactual carrier descriptor from the current scene only."""
    return "::".join((
        f"mediated:{str(expected_change)}",
        f"entity:{entity_signature(candidate)}",
        f"role:{structural_role_signature(candidate, observation)}",
        f"relation:{mediated_relation_signature(acted, candidate)}",
    ))


def _bbox_gap(first: ObjectInfo, second: ObjectInfo) -> int:
    first_row0, first_col0, first_row1, first_col1 = first.bbox
    second_row0, second_col0, second_row1, second_col1 = second.bbox
    row_gap = max(0, second_row0 - first_row1 - 1, first_row0 - second_row1 - 1)
    col_gap = max(0, second_col0 - first_col1 - 1, first_col0 - second_col1 - 1)
    return int(row_gap + col_gap)


__all__ = [
    "MediatedCarrierCandidate",
    "MediatedEffectPrediction",
    "MediatedEffectStatus",
    "MediatedEntityEffectEvidence",
    "OnlineMediatedEntityEffectStore",
    "SceneCorrespondenceStatus",
    "SceneEntityCorrespondence",
    "mediated_relation_signature",
    "prospective_mediator_signature",
]
