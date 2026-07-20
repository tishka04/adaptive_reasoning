"""Active, cross-branch acquisition for indirect entity-effect hypotheses.

The mediated-effect inducer can identify a useful but ambiguous hyperedge after
one transition.  This module turns that epistemic state into a future
experiment: repeat the same semantic intervention only after the same causal
option is observed open in another branch.  Resolution uses the subsequent
online transition status; terminal labels are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple

from .online_mediated_entity_effect import MediatedEffectStatus
from .online_semantic_intervention import SemanticInterventionAnchor


class MediatedReplicationStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    EXPIRED = "expired"


@dataclass
class MediatedReplicationRequest:
    request_id: str
    option_id: str
    edge_key: str
    objective_id: str
    downstream_subgoal_id: str
    mode_signature: str
    action_transfer_signature: str
    action_entity_signature: str
    action_role_signature: str
    candidate_mediator_signatures: Tuple[str, ...]
    source_branch: int
    source_context: str
    status: MediatedReplicationStatus = MediatedReplicationStatus.PENDING
    active_branch: int = -1
    active_opening_context: str = ""
    last_attempt_branch: int = -1
    attempts: int = 0
    selected_actions: int = 0
    observed_candidate_sets: list[Tuple[str, ...]] = field(default_factory=list)
    supported_mediator_signature: str = ""
    preparation_branches: set[int] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "option_id": self.option_id,
            "edge_key": self.edge_key,
            "objective_id": self.objective_id,
            "downstream_subgoal_id": self.downstream_subgoal_id,
            "mode_signature": self.mode_signature,
            "action_transfer_signature": self.action_transfer_signature,
            "action_entity_signature": self.action_entity_signature,
            "action_role_signature": self.action_role_signature,
            "candidate_mediator_signatures": list(
                self.candidate_mediator_signatures
            ),
            "source_branch": self.source_branch,
            "source_context": self.source_context,
            "status": self.status.value,
            "active_branch": self.active_branch,
            "active_opening_context": self.active_opening_context,
            "last_attempt_branch": self.last_attempt_branch,
            "attempts": self.attempts,
            "selected_actions": self.selected_actions,
            "observed_candidate_sets": [
                list(items) for items in self.observed_candidate_sets
            ],
            "supported_mediator_signature": (
                self.supported_mediator_signature
            ),
            "preparation_branches": sorted(self.preparation_branches),
        }


@dataclass(frozen=True)
class MediatedReplicationPrediction:
    request_id: str
    option_id: str
    objective_id: str
    downstream_subgoal_id: str
    action_signature: str
    action_transfer_signature: str
    compatible: bool
    exact_semantic_replication: bool
    cross_branch: bool
    reason: str

    @property
    def selection_rank(self) -> int:
        return 9 if self.compatible and self.cross_branch else -1


class OnlineMediatedReplicationStore:
    """Reserve and resolve bounded mediated contrasts across real branches."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_attempts_per_request: int = 2,
        max_requests: int = 16,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_attempts_per_request = max(1, int(max_attempts_per_request))
        self.max_requests = max(1, int(max_requests))
        self._requests: Dict[str, MediatedReplicationRequest] = {}
        self._request_by_hypothesis: Dict[Tuple[str, str, str, str], str] = {}
        self._active_request_id = ""
        self._branch_index = 0
        self._next_request_id = 1
        self._requests_created = 0
        self._cross_branch_reservations = 0
        self._cross_branch_activations = 0
        self._same_branch_blocks = 0
        self._exact_replication_predictions = 0
        self._incompatible_predictions = 0
        self._selections = 0
        self._preparation_actions = 0
        self._confirmations = 0
        self._refutations = 0
        self._expirations = 0

    @property
    def active_request(self) -> MediatedReplicationRequest | None:
        return self._requests.get(self._active_request_id)

    def requests(self) -> list[MediatedReplicationRequest]:
        return sorted(self._requests.values(), key=lambda item: item.request_id)

    def start_branch(self, branch_index: int) -> None:
        """Carry unresolved requests forward, never an active rollout."""
        self._branch_index = int(branch_index)
        active = self.active_request
        if active is not None and active.status == MediatedReplicationStatus.ACTIVE:
            active.status = MediatedReplicationStatus.PENDING
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
        """Activate one request only on a later observed opening."""
        if not self.enabled:
            return ""
        branch = int(branch_index)
        eligible = [
            request for request in self.requests()
            if request.status == MediatedReplicationStatus.PENDING
            and request.option_id == str(option_id)
            and (not request.edge_key or request.edge_key == str(edge_key))
            and branch > request.source_branch
            and branch > request.last_attempt_branch
        ]
        if not eligible:
            if any(
                request.status == MediatedReplicationStatus.PENDING
                and request.option_id == str(option_id)
                and branch <= request.source_branch
                for request in self.requests()
            ):
                self._same_branch_blocks += 1
            return ""
        request = eligible[0]
        if self.active_request is not None:
            self.active_request.status = MediatedReplicationStatus.PENDING
        request.status = MediatedReplicationStatus.ACTIVE
        request.active_branch = branch
        request.active_opening_context = str(opening_context)
        self._active_request_id = request.request_id
        self._cross_branch_reservations += 1
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
        """Create or resolve a request from non-terminal transition evidence."""
        if not self.enabled or not bool(mediated_outcome.get("observed", False)):
            return ""
        status = str(mediated_outcome.get("status", ""))
        candidates = tuple(sorted(
            str(item)
            for item in mediated_outcome.get("candidate_mediator_signatures", ())
            if str(item)
        ))
        if selected_request_id:
            self._resolve_selected(
                selected_request_id,
                status=status,
                branch_index=branch_index,
                candidates=candidates,
                supported_mediator_signature=str(
                    mediated_outcome.get("supported_mediator_signature", "")
                ),
            )
        if (
            float(mediated_outcome.get("gain", 0.0) or 0.0) <= 0.0
            or status != MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST.value
            or not candidates
            or not anchor.anchored
        ):
            return ""
        key = (
            str(option_id),
            str(objective_id),
            str(mediated_outcome.get("mode_signature", "")),
            str(mediated_outcome.get("action_transfer_signature", "")),
        )
        existing_id = self._request_by_hypothesis.get(key, "")
        if existing_id:
            existing = self._requests.get(existing_id)
            if existing is not None:
                existing.candidate_mediator_signatures = candidates
                return existing.request_id
        if len(self._requests) >= self.max_requests:
            return ""
        request_id = f"mediated-replication:{self._next_request_id}"
        self._next_request_id += 1
        request = MediatedReplicationRequest(
            request_id=request_id,
            option_id=str(option_id),
            edge_key=str(edge_key),
            objective_id=str(objective_id),
            downstream_subgoal_id=str(downstream_subgoal_id),
            mode_signature=key[2],
            action_transfer_signature=key[3],
            action_entity_signature=anchor.entity_signature,
            action_role_signature=anchor.structural_role_signature,
            candidate_mediator_signatures=candidates,
            source_branch=int(branch_index),
            source_context=str(context_signature),
        )
        self._requests[request_id] = request
        self._request_by_hypothesis[key] = request_id
        self._requests_created += 1
        return request_id

    def preferred_subgoal_id(self, option_id: str) -> str:
        active = self.active_request
        if (
            active is None
            or active.status != MediatedReplicationStatus.ACTIVE
            or active.option_id != str(option_id)
        ):
            return ""
        return active.downstream_subgoal_id

    def preferred_preparation_edge_key(self) -> str:
        """Request recreation of an opening on a later branch."""
        if not self.enabled or self.active_request is not None:
            return ""
        eligible = [
            request for request in self.requests()
            if request.status == MediatedReplicationStatus.PENDING
            and self._branch_index > request.source_branch
            and self._branch_index > request.last_attempt_branch
        ]
        return "" if not eligible else eligible[0].edge_key

    def note_preparation_action(self, edge_key: str) -> None:
        for request in self.requests():
            if (
                request.status == MediatedReplicationStatus.PENDING
                and request.edge_key == str(edge_key)
                and self._branch_index > request.source_branch
                and self._branch_index > request.last_attempt_branch
            ):
                request.preparation_branches.add(self._branch_index)
                self._preparation_actions += 1
                return

    def predict(
        self,
        *,
        option_id: str,
        anchor: SemanticInterventionAnchor,
        record_prediction: bool = True,
    ) -> MediatedReplicationPrediction | None:
        active = self.active_request
        if (
            not self.enabled
            or active is None
            or active.status != MediatedReplicationStatus.ACTIVE
            or active.option_id != str(option_id)
        ):
            return None
        exact = bool(
            anchor.anchored
            and anchor.transfer_signature == active.action_transfer_signature
        )
        cross_branch = bool(active.active_branch > active.source_branch)
        compatible = bool(exact and cross_branch)
        if record_prediction:
            if compatible:
                self._exact_replication_predictions += 1
            else:
                self._incompatible_predictions += 1
        return MediatedReplicationPrediction(
            request_id=active.request_id,
            option_id=active.option_id,
            objective_id=active.objective_id,
            downstream_subgoal_id=active.downstream_subgoal_id,
            action_signature=anchor.concrete_signature,
            action_transfer_signature=anchor.transfer_signature,
            compatible=compatible,
            exact_semantic_replication=exact,
            cross_branch=cross_branch,
            reason=(
                "same semantic intervention after an independent option opening"
                if compatible
                else "not the reserved cross-branch semantic intervention"
            ),
        )

    def note_selection(self, prediction: MediatedReplicationPrediction) -> None:
        if not prediction.compatible:
            return
        request = self._requests.get(prediction.request_id)
        if request is None or request.status != MediatedReplicationStatus.ACTIVE:
            return
        request.selected_actions += 1
        request.attempts += 1
        request.last_attempt_branch = request.active_branch
        self._selections += 1

    def summary(self) -> Dict[str, Any]:
        requests = self.requests()
        return {
            "enabled": self.enabled,
            "requests": len(requests),
            "requests_created": self._requests_created,
            "pending_requests": sum(
                item.status == MediatedReplicationStatus.PENDING
                for item in requests
            ),
            "active_requests": sum(
                item.status == MediatedReplicationStatus.ACTIVE
                for item in requests
            ),
            "cross_branch_reservations": self._cross_branch_reservations,
            "cross_branch_activations": self._cross_branch_activations,
            "same_branch_blocks": self._same_branch_blocks,
            "exact_replication_predictions": (
                self._exact_replication_predictions
            ),
            "incompatible_predictions": self._incompatible_predictions,
            "selections": self._selections,
            "preparation_actions": self._preparation_actions,
            "confirmations": self._confirmations,
            "refutations": self._refutations,
            "expirations": self._expirations,
            "active_request_id": self._active_request_id,
            "hypotheses": [item.to_dict() for item in requests],
        }

    def _resolve_selected(
        self,
        request_id: str,
        *,
        status: str,
        branch_index: int,
        candidates: Tuple[str, ...],
        supported_mediator_signature: str,
    ) -> None:
        request = self._requests.get(str(request_id))
        if request is None or int(branch_index) <= request.source_branch:
            return
        if candidates:
            request.observed_candidate_sets.append(candidates)
        if status == MediatedEffectStatus.SUPPORTED.value:
            request.status = MediatedReplicationStatus.CONFIRMED
            request.supported_mediator_signature = supported_mediator_signature
            self._confirmations += 1
            self._active_request_id = ""
            return
        if status == MediatedEffectStatus.CONTRADICTED.value:
            request.status = MediatedReplicationStatus.REFUTED
            self._refutations += 1
            self._active_request_id = ""
            return
        if request.attempts >= self.max_attempts_per_request:
            request.status = MediatedReplicationStatus.EXPIRED
            self._expirations += 1
        else:
            request.status = MediatedReplicationStatus.PENDING
        request.active_branch = -1
        request.active_opening_context = ""
        self._active_request_id = ""


__all__ = [
    "MediatedReplicationPrediction",
    "MediatedReplicationRequest",
    "MediatedReplicationStatus",
    "OnlineMediatedReplicationStore",
]
