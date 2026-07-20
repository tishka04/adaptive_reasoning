"""Active one-feature tests for online mediated carrier abstractions.

SAGE.8x can induce the invariant part of an indirect carrier after multiple
progressive transitions.  A supported invariant is still observational.  This
module turns it into a bounded experiment: reopen the same causal option on a
later branch, keep the semantic action class fixed, and select only a context
whose prospective carrier differs in exactly one learned feature.  Resolution
uses objective-distance gain and the observed carrier set, never a terminal
answer or a level-specific template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple

from v3.schemas import GameObservation

from .online_mediated_abstraction import (
    FEATURE_ORDER,
    parse_mediated_candidate,
)
from .online_mediated_entity_effect import prospective_mediator_signature
from .online_semantic_intervention import SemanticInterventionAnchor
from .online_state_conditioned_effect import LatentModeRestorationPrediction


DISCRIMINATION_FEATURE_ORDER = (
    "boundary",
    "multiplicity",
    "role_adjacency",
    "role_alignment",
    "proximity",
    "relation_alignment",
    "color_relation",
    "area",
    "player_relation",
    "vertical_relation",
    "horizontal_relation",
    "shape",
    "color",
)


class MediatedDiscriminationStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FEATURE_REQUIRED = "feature_required"
    FEATURE_ELIMINATED = "feature_eliminated"
    EXPIRED = "expired"


@dataclass
class MediatedDiscriminationRequest:
    request_id: str
    option_id: str
    edge_key: str
    objective_id: str
    downstream_subgoal_id: str
    mode_signature: str
    action_transfer_signature: str
    abstraction_signature: str
    abstraction_features: Tuple[Tuple[str, str], ...]
    candidate_features: Tuple[str, ...]
    source_branch: int
    source_context: str
    status: MediatedDiscriminationStatus = (
        MediatedDiscriminationStatus.PENDING
    )
    active_branch: int = -1
    active_opening_context: str = ""
    last_attempt_branch: int = -1
    attempts: int = 0
    selected_actions: int = 0
    tested_feature: str = ""
    expected_value: str = ""
    contrast_value: str = ""
    prospective_candidate_signature: str = ""
    observed_candidate_signatures: Tuple[str, ...] = ()
    objective_gain: float = 0.0
    preparation_branches: set[int] = field(default_factory=set)
    restoration_attempts: int = 0
    restoration_actions: Tuple[str, ...] = ()
    restoration_path_modes: Tuple[str, ...] = ()
    restoration_source_mode: str = ""
    restoration_expected_next_mode: str = ""
    restoration_observed_next_mode: str = ""
    restoration_target_reached: bool = False
    restoration_attempts_by_branch: Dict[int, int] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "option_id": self.option_id,
            "edge_key": self.edge_key,
            "objective_id": self.objective_id,
            "downstream_subgoal_id": self.downstream_subgoal_id,
            "mode_signature": self.mode_signature,
            "action_transfer_signature": self.action_transfer_signature,
            "abstraction_signature": self.abstraction_signature,
            "abstraction_features": dict(self.abstraction_features),
            "candidate_features": list(self.candidate_features),
            "source_branch": self.source_branch,
            "source_context": self.source_context,
            "status": self.status.value,
            "active_branch": self.active_branch,
            "active_opening_context": self.active_opening_context,
            "last_attempt_branch": self.last_attempt_branch,
            "attempts": self.attempts,
            "selected_actions": self.selected_actions,
            "tested_feature": self.tested_feature,
            "expected_value": self.expected_value,
            "contrast_value": self.contrast_value,
            "prospective_candidate_signature": (
                self.prospective_candidate_signature
            ),
            "observed_candidate_signatures": list(
                self.observed_candidate_signatures
            ),
            "objective_gain": round(float(self.objective_gain), 4),
            "preparation_branches": sorted(self.preparation_branches),
            "restoration_attempts": self.restoration_attempts,
            "restoration_actions": list(self.restoration_actions),
            "restoration_path_modes": list(self.restoration_path_modes),
            "restoration_source_mode": self.restoration_source_mode,
            "restoration_expected_next_mode": (
                self.restoration_expected_next_mode
            ),
            "restoration_observed_next_mode": (
                self.restoration_observed_next_mode
            ),
            "restoration_target_reached": self.restoration_target_reached,
            "restoration_attempts_by_branch": {
                str(branch): attempts
                for branch, attempts in sorted(
                    self.restoration_attempts_by_branch.items()
                )
            },
        }


@dataclass(frozen=True)
class MediatedDiscriminationPrediction:
    request_id: str
    option_id: str
    objective_id: str
    action_signature: str
    action_transfer_signature: str
    compatible: bool
    cross_branch: bool
    same_latent_mode: bool
    single_feature_contrast: bool
    tested_feature: str = ""
    expected_value: str = ""
    contrast_value: str = ""
    prospective_candidate_signature: str = ""
    reason: str = ""

    @property
    def selection_rank(self) -> int:
        if not self.compatible or not self.single_feature_contrast:
            return -1
        try:
            offset = DISCRIMINATION_FEATURE_ORDER.index(self.tested_feature)
        except ValueError:
            offset = len(DISCRIMINATION_FEATURE_ORDER)
        return 30 - offset


class OnlineMediatedDiscriminationStore:
    """Schedule and resolve matched one-feature abstraction controls."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        enable_mode_restoration: bool = True,
        max_attempts_per_request: int = 2,
        max_requests: int = 32,
        max_restoration_actions_per_branch: int = 3,
    ) -> None:
        self.enabled = bool(enabled)
        self.enable_mode_restoration = bool(
            enabled and enable_mode_restoration
        )
        self.max_attempts_per_request = max(1, int(max_attempts_per_request))
        self.max_requests = max(1, int(max_requests))
        self.max_restoration_actions_per_branch = max(
            1,
            int(max_restoration_actions_per_branch),
        )
        self._requests: Dict[str, MediatedDiscriminationRequest] = {}
        self._active_request_id = ""
        self._branch_index = 0
        self._next_request_id = 1
        self._tested_features: Dict[
            Tuple[str, str, str, str, str], set[str]
        ] = {}
        self._requests_created = 0
        self._cross_branch_activations = 0
        self._same_branch_blocks = 0
        self._predictions = 0
        self._mode_mismatch_blocks = 0
        self._no_single_feature_blocks = 0
        self._selections = 0
        self._preparation_actions = 0
        self._feature_requirements = 0
        self._feature_eliminations = 0
        self._inconclusive_attempts = 0
        self._expirations = 0
        self._restoration_predictions = 0
        self._restoration_selections = 0
        self._restoration_steps_confirmed = 0
        self._restoration_targets_reached = 0
        self._restoration_failures = 0
        self._restoration_unavailable_contexts: set[
            Tuple[str, int, str]
        ] = set()

    @property
    def active_request(self) -> MediatedDiscriminationRequest | None:
        return self._requests.get(self._active_request_id)

    def requests(self) -> list[MediatedDiscriminationRequest]:
        return sorted(self._requests.values(), key=lambda item: item.request_id)

    def start_branch(self, branch_index: int) -> None:
        self._branch_index = int(branch_index)
        active = self.active_request
        if (
            active is not None
            and active.status == MediatedDiscriminationStatus.ACTIVE
        ):
            active.status = MediatedDiscriminationStatus.PENDING
            active.active_branch = -1
            active.active_opening_context = ""
        self._active_request_id = ""

    def note_opening(
        self,
        *,
        option_id: str,
        edge_key: str,
        branch_index: int,
        opening_context: str,
    ) -> str:
        if not self.enabled:
            return ""
        branch = int(branch_index)
        eligible = [
            request for request in self.requests()
            if request.status == MediatedDiscriminationStatus.PENDING
            and request.option_id == str(option_id)
            and (not request.edge_key or request.edge_key == str(edge_key))
            and branch > request.source_branch
            and branch > request.last_attempt_branch
        ]
        if not eligible:
            if any(
                request.status == MediatedDiscriminationStatus.PENDING
                and request.option_id == str(option_id)
                and branch <= request.source_branch
                for request in self.requests()
            ):
                self._same_branch_blocks += 1
            return ""
        request = eligible[0]
        if self.active_request is not None:
            self.active_request.status = MediatedDiscriminationStatus.PENDING
        request.status = MediatedDiscriminationStatus.ACTIVE
        request.active_branch = branch
        request.active_opening_context = str(opening_context)
        self._active_request_id = request.request_id
        self._cross_branch_activations += 1
        return request.request_id

    def observe_hypothesis(
        self,
        *,
        option_id: str,
        edge_key: str,
        objective_id: str,
        downstream_subgoal_id: str,
        anchor: SemanticInterventionAnchor,
        branch_index: int,
        context_signature: str,
        mediated_outcome: Mapping[str, Any],
        selected_request_id: str = "",
    ) -> str:
        """Resolve a selected control, then reserve the next online test."""
        if not self.enabled or not bool(mediated_outcome.get("observed", False)):
            return ""
        if selected_request_id:
            self._resolve_selected(
                selected_request_id,
                branch_index=branch_index,
                mediated_outcome=mediated_outcome,
            )
        abstraction = mediated_outcome.get("mediator_abstraction") or {}
        if (
            not bool(mediated_outcome.get("supported_mediator_is_abstract"))
            or not bool(abstraction.get("supported"))
            or not anchor.anchored
        ):
            return ""
        return self._ensure_request(
            option_id=str(option_id),
            edge_key=str(edge_key),
            objective_id=str(objective_id),
            downstream_subgoal_id=str(downstream_subgoal_id),
            mode_signature=str(mediated_outcome.get("mode_signature", "")),
            action_transfer_signature=str(
                mediated_outcome.get("action_transfer_signature", "")
            ),
            abstraction_signature=str(abstraction.get("signature", "")),
            abstraction_features={
                str(key): str(value)
                for key, value in dict(abstraction.get("features", {})).items()
            },
            branch_index=int(branch_index),
            context_signature=str(context_signature),
        )

    def preferred_subgoal_id(self, option_id: str) -> str:
        active = self.active_request
        if (
            active is None
            or active.status != MediatedDiscriminationStatus.ACTIVE
            or active.option_id != str(option_id)
        ):
            return ""
        return active.downstream_subgoal_id

    def preferred_preparation_edge_key(self) -> str:
        if not self.enabled or self.active_request is not None:
            return ""
        eligible = [
            request for request in self.requests()
            if request.status == MediatedDiscriminationStatus.PENDING
            and self._branch_index > request.source_branch
            and self._branch_index > request.last_attempt_branch
        ]
        return "" if not eligible else eligible[0].edge_key

    def note_preparation_action(self, edge_key: str) -> None:
        for request in self.requests():
            if (
                request.status == MediatedDiscriminationStatus.PENDING
                and request.edge_key == str(edge_key)
                and self._branch_index > request.source_branch
                and self._branch_index > request.last_attempt_branch
            ):
                request.preparation_branches.add(self._branch_index)
                self._preparation_actions += 1
                return

    def restoration_request(
        self,
        option_id: str,
    ) -> MediatedDiscriminationRequest | None:
        active = self.active_request
        if (
            not self.enable_mode_restoration
            or active is None
            or active.status != MediatedDiscriminationStatus.ACTIVE
            or active.option_id != str(option_id)
            or active.restoration_attempts_by_branch.get(
                active.active_branch,
                0,
            ) >= self.max_restoration_actions_per_branch
        ):
            return None
        return active

    def note_restoration_predictions(self, count: int) -> None:
        if self.enable_mode_restoration:
            self._restoration_predictions += max(0, int(count))

    def note_restoration_unavailable(self, current_mode: str) -> None:
        active = self.active_request
        if not self.enable_mode_restoration or active is None:
            return
        self._restoration_unavailable_contexts.add((
            active.request_id,
            active.active_branch,
            str(current_mode),
        ))

    def note_restoration_selection(
        self,
        prediction: LatentModeRestorationPrediction,
    ) -> str:
        active = self.restoration_request(prediction.option_id)
        if active is None or not prediction.compatible:
            return ""
        active.restoration_attempts += 1
        active.restoration_attempts_by_branch[active.active_branch] = (
            active.restoration_attempts_by_branch.get(active.active_branch, 0)
            + 1
        )
        active.restoration_actions = prediction.path_action_signatures
        active.restoration_path_modes = prediction.path_mode_signatures
        active.restoration_source_mode = prediction.current_mode_signature
        active.restoration_expected_next_mode = (
            prediction.expected_next_mode_signature
        )
        active.restoration_observed_next_mode = ""
        active.restoration_target_reached = False
        self._restoration_selections += 1
        return active.request_id

    def observe_restoration(
        self,
        request_id: str,
        *,
        before_mode: str,
        after_mode: str,
    ) -> Dict[str, Any]:
        request = self._requests.get(str(request_id))
        if (
            not self.enable_mode_restoration
            or request is None
            or not request.restoration_expected_next_mode
        ):
            return {
                "observed": False,
                "step_confirmed": False,
                "target_reached": False,
            }
        request.restoration_observed_next_mode = str(after_mode)
        step_confirmed = bool(
            str(before_mode) == request.restoration_source_mode
            and str(after_mode) == request.restoration_expected_next_mode
        )
        target_reached = bool(
            step_confirmed and str(after_mode) == request.mode_signature
        )
        if step_confirmed:
            self._restoration_steps_confirmed += 1
        else:
            self._restoration_failures += 1
        if target_reached:
            request.restoration_target_reached = True
            self._restoration_targets_reached += 1
        request.restoration_expected_next_mode = ""
        return {
            "observed": True,
            "step_confirmed": step_confirmed,
            "target_reached": target_reached,
            "before_mode": str(before_mode),
            "after_mode": str(after_mode),
            "target_mode": request.mode_signature,
        }

    def predict(
        self,
        *,
        option_id: str,
        anchor: SemanticInterventionAnchor,
        observation: GameObservation,
        mode_signature: str,
        record_prediction: bool = True,
    ) -> MediatedDiscriminationPrediction | None:
        active = self.active_request
        if (
            not self.enabled
            or active is None
            or active.status != MediatedDiscriminationStatus.ACTIVE
            or active.option_id != str(option_id)
        ):
            return None
        cross_branch = bool(active.active_branch > active.source_branch)
        same_mode = bool(str(mode_signature) == active.mode_signature)
        exact_action = bool(
            anchor.anchored
            and anchor.transfer_signature == active.action_transfer_signature
        )
        contrast = (
            self._best_single_feature_contrast(active, anchor, observation)
            if exact_action and cross_branch and same_mode else None
        )
        compatible = contrast is not None
        if record_prediction and compatible:
            self._predictions += 1
        elif record_prediction and exact_action and cross_branch:
            if not same_mode:
                self._mode_mismatch_blocks += 1
            else:
                self._no_single_feature_blocks += 1
        return MediatedDiscriminationPrediction(
            request_id=active.request_id,
            option_id=active.option_id,
            objective_id=active.objective_id,
            action_signature=anchor.concrete_signature,
            action_transfer_signature=anchor.transfer_signature,
            compatible=compatible,
            cross_branch=cross_branch,
            same_latent_mode=same_mode,
            single_feature_contrast=compatible,
            tested_feature="" if contrast is None else contrast[0],
            expected_value="" if contrast is None else contrast[1],
            contrast_value="" if contrast is None else contrast[2],
            prospective_candidate_signature=(
                "" if contrast is None else contrast[3]
            ),
            reason=(
                "matched cross-branch carrier differs in exactly one feature"
                if compatible else
                (
                    "latent mode differs from the abstraction source"
                    if not same_mode else
                    "no one-feature carrier contrast for this semantic action"
                )
            ),
        )

    def note_selection(
        self,
        prediction: MediatedDiscriminationPrediction,
    ) -> None:
        if not prediction.compatible:
            return
        request = self._requests.get(prediction.request_id)
        if (
            request is None
            or request.status != MediatedDiscriminationStatus.ACTIVE
        ):
            return
        request.selected_actions += 1
        request.attempts += 1
        request.last_attempt_branch = request.active_branch
        request.tested_feature = prediction.tested_feature
        request.expected_value = prediction.expected_value
        request.contrast_value = prediction.contrast_value
        request.prospective_candidate_signature = (
            prediction.prospective_candidate_signature
        )
        self._selections += 1

    def summary(self) -> Dict[str, Any]:
        requests = self.requests()
        return {
            "enabled": self.enabled,
            "requests": len(requests),
            "requests_created": self._requests_created,
            "pending_requests": sum(
                item.status == MediatedDiscriminationStatus.PENDING
                for item in requests
            ),
            "active_requests": sum(
                item.status == MediatedDiscriminationStatus.ACTIVE
                for item in requests
            ),
            "cross_branch_activations": self._cross_branch_activations,
            "same_branch_blocks": self._same_branch_blocks,
            "predictions": self._predictions,
            "mode_mismatch_blocks": self._mode_mismatch_blocks,
            "no_single_feature_blocks": self._no_single_feature_blocks,
            "selections": self._selections,
            "preparation_actions": self._preparation_actions,
            "feature_requirements": self._feature_requirements,
            "feature_eliminations": self._feature_eliminations,
            "inconclusive_attempts": self._inconclusive_attempts,
            "expirations": self._expirations,
            "mode_restoration_enabled": self.enable_mode_restoration,
            "restoration_predictions": self._restoration_predictions,
            "restoration_selections": self._restoration_selections,
            "restoration_steps_confirmed": (
                self._restoration_steps_confirmed
            ),
            "restoration_targets_reached": self._restoration_targets_reached,
            "restoration_failures": self._restoration_failures,
            "restoration_unavailable_contexts": len(
                self._restoration_unavailable_contexts
            ),
            "active_request_id": self._active_request_id,
            "hypotheses": [item.to_dict() for item in requests],
        }

    def _ensure_request(
        self,
        *,
        option_id: str,
        edge_key: str,
        objective_id: str,
        downstream_subgoal_id: str,
        mode_signature: str,
        action_transfer_signature: str,
        abstraction_signature: str,
        abstraction_features: Mapping[str, str],
        branch_index: int,
        context_signature: str,
    ) -> str:
        if not abstraction_signature or not action_transfer_signature:
            return ""
        key = (
            option_id,
            objective_id,
            mode_signature,
            action_transfer_signature,
            abstraction_signature,
        )
        if any(
            request.status in {
                MediatedDiscriminationStatus.PENDING,
                MediatedDiscriminationStatus.ACTIVE,
            }
            and self._request_key(request) == key
            for request in self.requests()
        ):
            return ""
        tested = self._tested_features.setdefault(key, set())
        candidates = tuple(
            feature for feature in DISCRIMINATION_FEATURE_ORDER
            if feature in abstraction_features and feature not in tested
        )
        if not candidates or len(self._requests) >= self.max_requests:
            return ""
        request_id = f"mediated-discrimination:{self._next_request_id}"
        self._next_request_id += 1
        ordered_features = tuple(
            (feature, str(abstraction_features[feature]))
            for feature in FEATURE_ORDER
            if feature in abstraction_features
        )
        self._requests[request_id] = MediatedDiscriminationRequest(
            request_id=request_id,
            option_id=option_id,
            edge_key=edge_key,
            objective_id=objective_id,
            downstream_subgoal_id=downstream_subgoal_id,
            mode_signature=mode_signature,
            action_transfer_signature=action_transfer_signature,
            abstraction_signature=abstraction_signature,
            abstraction_features=ordered_features,
            candidate_features=candidates,
            source_branch=branch_index,
            source_context=context_signature,
        )
        self._requests_created += 1
        return request_id

    def _best_single_feature_contrast(
        self,
        request: MediatedDiscriminationRequest,
        anchor: SemanticInterventionAnchor,
        observation: GameObservation,
    ) -> Tuple[str, str, str, str] | None:
        acted = next((
            item for item in observation.objects
            if item.object_id == anchor.source_object_id
        ), None)
        if acted is None:
            return None
        pattern = dict(request.abstraction_features)
        expected_change = pattern.get("change", "transformed")
        candidates = []
        for item in observation.objects:
            if item.object_id == acted.object_id:
                continue
            signature = prospective_mediator_signature(
                acted,
                item,
                observation,
                expected_change=expected_change,
            )
            parsed = parse_mediated_candidate(signature)
            mismatches = [
                feature for feature, expected in pattern.items()
                if feature != "change" and parsed.get(feature) != expected
            ]
            if len(mismatches) != 1:
                continue
            feature = mismatches[0]
            if feature not in request.candidate_features:
                continue
            candidates.append((
                _feature_priority(feature),
                feature,
                pattern[feature],
                parsed.get(feature, ""),
                signature,
            ))
        if not candidates:
            return None
        _priority, feature, expected, contrast, signature = max(candidates)
        return feature, expected, contrast, signature

    def _resolve_selected(
        self,
        request_id: str,
        *,
        branch_index: int,
        mediated_outcome: Mapping[str, Any],
    ) -> None:
        request = self._requests.get(str(request_id))
        if (
            request is None
            or int(branch_index) <= request.source_branch
            or not request.tested_feature
        ):
            return
        signatures = tuple(sorted(
            str(item)
            for item in mediated_outcome.get(
                "candidate_mediator_signatures", ()
            )
            if str(item)
        ))
        request.observed_candidate_signatures = signatures
        request.objective_gain = float(mediated_outcome.get("gain", 0.0) or 0.0)
        contrast_observed = any(
            _matches_contrast(
                request.abstraction_features,
                request.tested_feature,
                request.contrast_value,
                signature,
            )
            for signature in signatures
        )
        reliable_control = bool(
            mediated_outcome.get("target_stable_for_mediation", False)
            and not mediated_outcome.get(
                "ambiguous_scene_correspondence", False
            )
        )
        key = self._request_key(request)
        if request.objective_gain > 0.0 and contrast_observed:
            request.status = MediatedDiscriminationStatus.FEATURE_ELIMINATED
            self._tested_features.setdefault(key, set()).add(
                request.tested_feature
            )
            self._feature_eliminations += 1
            self._active_request_id = ""
            return
        if request.objective_gain <= 0.0 and reliable_control:
            request.status = MediatedDiscriminationStatus.FEATURE_REQUIRED
            self._tested_features.setdefault(key, set()).add(
                request.tested_feature
            )
            self._feature_requirements += 1
            self._active_request_id = ""
            return
        self._inconclusive_attempts += 1
        if request.attempts >= self.max_attempts_per_request:
            request.status = MediatedDiscriminationStatus.EXPIRED
            self._expirations += 1
        else:
            request.status = MediatedDiscriminationStatus.PENDING
            request.active_branch = -1
            request.active_opening_context = ""
        self._active_request_id = ""

    @staticmethod
    def _request_key(
        request: MediatedDiscriminationRequest,
    ) -> Tuple[str, str, str, str, str]:
        return (
            request.option_id,
            request.objective_id,
            request.mode_signature,
            request.action_transfer_signature,
            request.abstraction_signature,
        )


def _matches_contrast(
    abstraction_features: Tuple[Tuple[str, str], ...],
    tested_feature: str,
    contrast_value: str,
    candidate_signature: str,
) -> bool:
    candidate = parse_mediated_candidate(candidate_signature)
    if not candidate or candidate.get(tested_feature) != contrast_value:
        return False
    return all(
        feature == tested_feature or candidate.get(feature) == expected
        for feature, expected in abstraction_features
    )


def _feature_priority(feature: str) -> int:
    try:
        return len(DISCRIMINATION_FEATURE_ORDER) - (
            DISCRIMINATION_FEATURE_ORDER.index(feature)
        )
    except ValueError:
        return 0


__all__ = [
    "DISCRIMINATION_FEATURE_ORDER",
    "MediatedDiscriminationPrediction",
    "MediatedDiscriminationRequest",
    "MediatedDiscriminationStatus",
    "OnlineMediatedDiscriminationStore",
]
