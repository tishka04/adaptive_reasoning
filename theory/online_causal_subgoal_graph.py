"""Online induction of causal, enabling relations between live subgoals.

This memory is mechanical rather than terminal: it learns that progress on one
measurable objective made another objective testable or reduced its deficit.
Candidate edges may be probed with strict bounds, but require independent
observed contexts before they are treated as confirmed causal preconditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from v3.schemas import GameObservation

from .online_terminal_objective import (
    OnlineTerminalObjectiveStore,
    TerminalObjectiveHypothesis,
    TerminalObjectiveStatus,
)


class CausalSubgoalEdgeStatus(str, Enum):
    """Mechanic status of one proposed enabling dependency."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"


@dataclass
class CausalMechanicEvidence:
    """Position-invariant evidence for one intervention or observed effect."""

    signature: str
    observations: int = 0
    source_progress_events: int = 0
    enablement_successes: int = 0
    enablement_failures: int = 0
    support_contexts: set[str] = field(default_factory=set)

    @property
    def progress_probability(self) -> float:
        return (self.source_progress_events + 1.0) / (self.observations + 2.0)

    @property
    def enablement_probability(self) -> float:
        return (self.enablement_successes + 1.0) / (
            self.enablement_successes + self.enablement_failures + 2.0
        )

    @property
    def utility(self) -> float:
        return (
            2.0 * self.progress_probability
            + 2.0 * self.enablement_probability
            - 0.5 * float(self.enablement_failures)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "observations": self.observations,
            "source_progress_events": self.source_progress_events,
            "enablement_successes": self.enablement_successes,
            "enablement_failures": self.enablement_failures,
            "independent_support_contexts": len(self.support_contexts),
            "progress_probability": round(self.progress_probability, 4),
            "enablement_probability": round(self.enablement_probability, 4),
            "utility": round(self.utility, 4),
        }


@dataclass
class CausalSubgoalEdge:
    """A falsifiable hypothesis that source progress enables a target goal."""

    edge_key: str
    source_objective_id: str
    target_objective_id: str
    shared_colors: Tuple[int, ...] = ()
    generation_reasons: set[str] = field(default_factory=set)
    structural_prior: float = 0.0
    minimum_independent_support: int = 2
    trials: int = 0
    actions: int = 0
    source_progress_events: int = 0
    source_completions: int = 0
    plan_failures: int = 0
    unsafe_failures: int = 0
    support_events: int = 0
    contradictions: int = 0
    availability_successes: int = 0
    availability_failures: int = 0
    cochange_supports: int = 0
    support_contexts: set[str] = field(default_factory=set)
    support_branches: set[int] = field(default_factory=set)
    contradiction_contexts: set[str] = field(default_factory=set)
    contradiction_branches: set[int] = field(default_factory=set)
    effect_evidence: Dict[str, CausalMechanicEvidence] = field(default_factory=dict)
    intervention_evidence: Dict[str, CausalMechanicEvidence] = field(
        default_factory=dict
    )
    confirmation_priority_bonus: float = 0.0

    @property
    def status(self) -> CausalSubgoalEdgeStatus:
        if (
            len(self.support_branches)
            >= max(1, int(self.minimum_independent_support))
            and self.support_events > self.contradictions
        ):
            return CausalSubgoalEdgeStatus.CONFIRMED
        if (
            len(self.contradiction_branches) >= 2
            and self.contradictions >= 2 * max(1, self.support_events)
        ):
            return CausalSubgoalEdgeStatus.REFUTED
        return CausalSubgoalEdgeStatus.CANDIDATE

    @property
    def enablement_probability(self) -> float:
        return (self.support_events + 1.0) / (
            self.support_events + self.contradictions + 2.0
        )

    @property
    def progress_probability(self) -> float:
        return (self.source_progress_events + 1.0) / (self.actions + 2.0)

    @property
    def expected_cost(self) -> float:
        if self.trials <= 0:
            return 1.0
        return max(1.0, float(self.actions) / float(self.trials))

    @property
    def risk_rate(self) -> float:
        return float(self.unsafe_failures) / max(1.0, float(self.actions))

    @property
    def needs_independent_confirmation(self) -> bool:
        return bool(
            self.support_events > 0
            and len(self.support_branches)
            < max(1, int(self.minimum_independent_support))
            and self.status == CausalSubgoalEdgeStatus.CANDIDATE
        )

    @property
    def utility(self) -> float:
        status_bonus = {
            CausalSubgoalEdgeStatus.CONFIRMED: 2.0,
            CausalSubgoalEdgeStatus.CANDIDATE: 0.0,
            CausalSubgoalEdgeStatus.REFUTED: -10.0,
        }[self.status]
        return (
            status_bonus
            + float(self.structural_prior)
            + float(self.confirmation_priority_bonus)
            + 2.0 * self.enablement_probability
            + self.progress_probability
            - 0.15 * self.expected_cost
            - 2.0 * self.risk_rate
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_key": self.edge_key,
            "source_objective_id": self.source_objective_id,
            "target_objective_id": self.target_objective_id,
            "shared_colors": list(self.shared_colors),
            "generation_reasons": sorted(self.generation_reasons),
            "structural_prior": round(float(self.structural_prior), 4),
            "status": self.status.value,
            "trials": self.trials,
            "actions": self.actions,
            "source_progress_events": self.source_progress_events,
            "source_completions": self.source_completions,
            "plan_failures": self.plan_failures,
            "unsafe_failures": self.unsafe_failures,
            "support_events": self.support_events,
            "contradictions": self.contradictions,
            "availability_successes": self.availability_successes,
            "availability_failures": self.availability_failures,
            "cochange_supports": self.cochange_supports,
            "independent_support_contexts": len(self.support_contexts),
            "independent_support_branches": len(self.support_branches),
            "independent_contradiction_branches": len(
                self.contradiction_branches
            ),
            "needs_independent_confirmation": self.needs_independent_confirmation,
            "confirmation_priority_bonus": round(
                float(self.confirmation_priority_bonus), 4
            ),
            "enablement_probability": round(self.enablement_probability, 4),
            "progress_probability": round(self.progress_probability, 4),
            "expected_cost": round(self.expected_cost, 4),
            "risk_rate": round(self.risk_rate, 4),
            "utility": round(self.utility, 4),
            "effect_evidence": [
                evidence.to_dict()
                for evidence in sorted(
                    self.effect_evidence.values(),
                    key=lambda item: item.signature,
                )
            ],
            "intervention_evidence": [
                evidence.to_dict()
                for evidence in sorted(
                    self.intervention_evidence.values(),
                    key=lambda item: item.signature,
                )
            ],
        }


@dataclass
class _PendingEdgeTrial:
    edge_key: str
    branch_index: int
    source_context: str
    source_completed: bool = False
    source_progressed: bool = False
    progress_transition_index: int = -1
    effect_signatures: set[str] = field(default_factory=set)
    intervention_signatures: set[str] = field(default_factory=set)


class OnlineCausalSubgoalGraph:
    """Discover and revise state-conditioned enabling subgoal dependencies."""

    def __init__(
        self,
        *,
        max_edges: int = 24,
        max_edges_per_blocked_target: int = 3,
        minimum_independent_support: int = 2,
        delayed_credit_window: int = 6,
        enable_effect_credit: bool = True,
    ) -> None:
        self.max_edges = max(1, int(max_edges))
        self.max_edges_per_blocked_target = max(
            1, int(max_edges_per_blocked_target)
        )
        self.minimum_independent_support = max(
            1, int(minimum_independent_support)
        )
        self.delayed_credit_window = max(1, int(delayed_credit_window))
        self.enable_effect_credit = bool(enable_effect_credit)
        self._edges: Dict[str, CausalSubgoalEdge] = {}
        self._blocked_signatures: Dict[str, set[str]] = {}
        self._pending_trials: Dict[str, _PendingEdgeTrial] = {}
        self._branch_index = 0
        self._blocked_events = 0
        self._availability_checks = 0
        self._recovered_availability_events = 0
        self._edges_generated_total = 0
        self._transition_index = 0
        self._effect_observations = 0
        self._effect_guided_actions = 0
        self._delayed_credit_events = 0
        self._expired_credit_windows = 0
        self._cross_branch_confirmations = 0

    def edges(self) -> List[CausalSubgoalEdge]:
        return sorted(self._edges.values(), key=lambda item: item.edge_key)

    def edge(self, edge_key: str) -> CausalSubgoalEdge | None:
        return self._edges.get(str(edge_key))

    def note_blocked(
        self,
        objective_id: str,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
    ) -> List[CausalSubgoalEdge]:
        """Record a state where no safe direct intervention targets a goal."""
        target = store.objective(objective_id)
        if target is None or target.status == TerminalObjectiveStatus.REFUTED:
            return []
        signature = self.state_signature(target, observation)
        signatures = self._blocked_signatures.setdefault(target.objective_id, set())
        if signature not in signatures:
            signatures.add(signature)
            self._blocked_events += 1
        self._propose_preconditions(target, observation, store)
        return self.candidate_edges(observation, store)

    def note_intervention_availability(
        self,
        objective_id: str,
        *,
        available: bool,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        context_signature: str = "",
    ) -> List[str]:
        """Resolve completed preparation trials from actual target testability."""
        self._availability_checks += 1
        objective = store.objective(objective_id)
        if objective is None:
            return []
        signature = self.state_signature(objective, observation)
        if available and signature in self._blocked_signatures.get(objective_id, set()):
            self._recovered_availability_events += 1
        resolved: List[str] = []
        for edge_key, trial in list(self._pending_trials.items()):
            edge = self.edge(edge_key)
            if (
                edge is None
                or edge.target_objective_id != str(objective_id)
                or not (
                    trial.source_progressed
                    if self.enable_effect_credit
                    else trial.source_completed
                )
            ):
                continue
            context = (
                str(context_signature)
                or f"{trial.source_context}=>{signature}"
            )
            if available:
                edge.availability_successes += 1
                previous_status = edge.status
                self._record_support(
                    edge,
                    context,
                    branch_index=trial.branch_index,
                )
                self._credit_trial_effects(
                    edge,
                    trial,
                    available=True,
                    context=context,
                )
                if (
                    previous_status != CausalSubgoalEdgeStatus.CONFIRMED
                    and edge.status == CausalSubgoalEdgeStatus.CONFIRMED
                ):
                    self._cross_branch_confirmations += 1
                if self.enable_effect_credit:
                    self._delayed_credit_events += 1
            else:
                edge.availability_failures += 1
                self._record_contradiction(
                    edge,
                    context,
                    branch_index=trial.branch_index,
                )
                self._credit_trial_effects(
                    edge,
                    trial,
                    available=False,
                    context=context,
                )
            del self._pending_trials[edge_key]
            resolved.append(edge_key)
        return resolved

    def candidate_edges(
        self,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
    ) -> List[CausalSubgoalEdge]:
        """Return bounded dependencies relevant to a currently blocked state."""
        result = []
        for edge in self.edges():
            edge.confirmation_priority_bonus = (
                2.5
                if self.enable_effect_credit
                and edge.needs_independent_confirmation
                and self._branch_index not in edge.support_branches
                else 0.0
            )
            if edge.status == CausalSubgoalEdgeStatus.REFUTED:
                continue
            source = store.objective(edge.source_objective_id)
            target = store.objective(edge.target_objective_id)
            if source is None or target is None:
                continue
            source_distance = source.distance(observation)
            if source_distance is None or source_distance <= 0.0:
                continue
            target_signature = self.state_signature(target, observation)
            currently_blocked = target_signature in self._blocked_signatures.get(
                target.objective_id, set()
            )
            if not currently_blocked and edge.status != CausalSubgoalEdgeStatus.CONFIRMED:
                continue
            result.append(edge)
        return sorted(
            result,
            key=lambda item: (item.utility, item.edge_key),
            reverse=True,
        )

    def intervention_utilities(self, edge_key: str) -> Dict[str, float]:
        """Return learned semantic intervention utilities for one dependency."""
        if not self.enable_effect_credit:
            return {}
        edge = self.edge(edge_key)
        if edge is None:
            return {}
        return {
            signature: evidence.utility
            for signature, evidence in edge.intervention_evidence.items()
            if evidence.source_progress_events > 0
            or evidence.enablement_successes > 0
        }

    def begin_trial(self, edge_key: str, *, context_signature: str) -> None:
        edge = self.edge(edge_key)
        if edge is None or edge.status == CausalSubgoalEdgeStatus.REFUTED:
            return
        existing = self._pending_trials.get(edge.edge_key)
        if existing is not None and existing.branch_index == self._branch_index:
            return
        edge.trials += 1
        self._pending_trials[edge.edge_key] = _PendingEdgeTrial(
            edge_key=edge.edge_key,
            branch_index=self._branch_index,
            source_context=(
                str(context_signature)
                or f"branch:{self._branch_index}:trial:{edge.trials}"
            ),
        )

    def cancel_trial(self, edge_key: str, *, count_failure: bool = True) -> None:
        """Cancel an unexecuted preparation, for example after a safety veto."""
        edge = self.edge(edge_key)
        if edge is not None and count_failure:
            edge.plan_failures += 1
        self._pending_trials.pop(str(edge_key), None)

    def observe_transition(
        self,
        update: Any,
        *,
        store: OnlineTerminalObjectiveStore,
        source_objective_id: str = "",
        edge_key: str = "",
        source_step_completed: bool = False,
        plan_abandoned: bool = False,
        context_signature: str = "",
        intervention_signature: str = "",
    ) -> Dict[str, Any]:
        """Learn causal co-change and execution cost from one real transition."""
        self._transition_index += 1
        self._expire_delayed_credit()
        source_id = str(source_objective_id)
        context = str(context_signature) or (
            f"branch:{self._branch_index}:grid:{update.record.obs_before.grid_hash}"
        )
        relevant = [
            edge for edge in self.edges()
            if edge.source_objective_id == source_id
            and edge.status != CausalSubgoalEdgeStatus.REFUTED
        ]
        source = store.objective(source_id)
        before_source = (
            None if source is None else source.distance(update.record.obs_before)
        )
        after_source = (
            None if source is None else source.distance(update.record.obs_after)
        )
        source_reduced = bool(
            before_source is not None
            and after_source is not None
            and after_source < before_source
        )
        supported: List[str] = []
        effect_signature = (
            transition_effect_signature(update)
            if self.enable_effect_credit else ""
        )
        for edge in relevant:
            if not source_reduced:
                continue
            target = store.objective(edge.target_objective_id)
            if target is None:
                continue
            before_distance = target.distance(update.record.obs_before)
            after_distance = target.distance(update.record.obs_after)
            became_measurable = before_distance is None and after_distance is not None
            reduced = bool(
                before_distance is not None
                and after_distance is not None
                and after_distance < before_distance
            )
            if became_measurable or reduced:
                edge.cochange_supports += 1
                if self._record_support(
                    edge,
                    context,
                    branch_index=self._branch_index,
                ):
                    supported.append(edge.edge_key)

        selected_edge = self.edge(edge_key)
        if selected_edge is not None:
            selected_edge.actions += 1
            effect_evidence = None
            semantic_intervention = str(intervention_signature)
            intervention_evidence = None
            if self.enable_effect_credit:
                effect_evidence = self._mechanic_evidence(
                    selected_edge.effect_evidence,
                    effect_signature,
                )
                effect_evidence.observations += 1
                self._effect_observations += 1
            if self.enable_effect_credit and semantic_intervention:
                intervention_evidence = self._mechanic_evidence(
                    selected_edge.intervention_evidence,
                    semantic_intervention,
                )
                intervention_evidence.observations += 1
                if semantic_intervention in self.intervention_utilities(
                    selected_edge.edge_key
                ):
                    self._effect_guided_actions += 1
            if source_reduced:
                selected_edge.source_progress_events += 1
                if effect_evidence is not None:
                    effect_evidence.source_progress_events += 1
                if intervention_evidence is not None:
                    intervention_evidence.source_progress_events += 1
                trial = self._pending_trials.get(selected_edge.edge_key)
                if trial is not None:
                    if self.enable_effect_credit:
                        trial.source_progressed = True
                        trial.progress_transition_index = self._transition_index
                        trial.effect_signatures.add(effect_signature)
                        if semantic_intervention:
                            trial.intervention_signatures.add(
                                semantic_intervention
                            )
            if source_step_completed:
                selected_edge.source_completions += 1
                trial = self._pending_trials.get(selected_edge.edge_key)
                if trial is not None:
                    trial.source_completed = True
            if plan_abandoned:
                selected_edge.plan_failures += 1
                trial = self._pending_trials.get(selected_edge.edge_key)
                if (
                    trial is not None
                    and not trial.source_completed
                    and not trial.source_progressed
                ):
                    del self._pending_trials[selected_edge.edge_key]
            if update.record.diff.game_over:
                selected_edge.unsafe_failures += 1

        return {
            "source_objective_id": source_id,
            "edge_key": str(edge_key),
            "cochange_supported_edges": supported,
            "source_step_completed": bool(source_step_completed),
            "plan_abandoned": bool(plan_abandoned),
            "effect_signature": effect_signature if selected_edge is not None else "",
            "intervention_signature": str(intervention_signature),
            "delayed_credit_pending": bool(
                selected_edge is not None
                and selected_edge.edge_key in self._pending_trials
                and self._pending_trials[selected_edge.edge_key].source_progressed
            ),
        }

    def start_branch(self) -> None:
        """Censor unresolved trials without manufacturing contradictions."""
        self._pending_trials.clear()
        self._branch_index += 1

    def summary(self) -> Dict[str, Any]:
        edges = self.edges()
        statuses = {
            status.value: sum(edge.status == status for edge in edges)
            for status in CausalSubgoalEdgeStatus
        }
        return {
            "edges": len(edges),
            "edges_generated_total": self._edges_generated_total,
            "statuses": statuses,
            "blocked_target_events": self._blocked_events,
            "blocked_targets": len(self._blocked_signatures),
            "availability_checks": self._availability_checks,
            "recovered_availability_events": self._recovered_availability_events,
            "pending_trials": len(self._pending_trials),
            "trials": sum(edge.trials for edge in edges),
            "actions": sum(edge.actions for edge in edges),
            "source_progress_events": sum(
                edge.source_progress_events for edge in edges
            ),
            "source_completions": sum(edge.source_completions for edge in edges),
            "plan_failures": sum(edge.plan_failures for edge in edges),
            "unsafe_failures": sum(edge.unsafe_failures for edge in edges),
            "support_events": sum(edge.support_events for edge in edges),
            "contradictions": sum(edge.contradictions for edge in edges),
            "availability_successes": sum(
                edge.availability_successes for edge in edges
            ),
            "availability_failures": sum(
                edge.availability_failures for edge in edges
            ),
            "cochange_supports": sum(edge.cochange_supports for edge in edges),
            "effect_observations": self._effect_observations,
            "effect_guided_actions": self._effect_guided_actions,
            "productive_effect_signatures": sum(
                evidence.enablement_successes > 0
                for edge in edges
                for evidence in edge.effect_evidence.values()
            ),
            "productive_intervention_signatures": sum(
                evidence.enablement_successes > 0
                for edge in edges
                for evidence in edge.intervention_evidence.values()
            ),
            "delayed_credit_events": self._delayed_credit_events,
            "expired_credit_windows": self._expired_credit_windows,
            "cross_branch_confirmations": self._cross_branch_confirmations,
            "confirmed_edges": sum(
                edge.status == CausalSubgoalEdgeStatus.CONFIRMED for edge in edges
            ),
            "refuted_edges": sum(
                edge.status == CausalSubgoalEdgeStatus.REFUTED for edge in edges
            ),
            "hypotheses": [edge.to_dict() for edge in edges],
        }

    def _credit_trial_effects(
        self,
        edge: CausalSubgoalEdge,
        trial: _PendingEdgeTrial,
        *,
        available: bool,
        context: str,
    ) -> None:
        if not self.enable_effect_credit:
            return
        for signature in trial.effect_signatures:
            evidence = self._mechanic_evidence(edge.effect_evidence, signature)
            if available:
                evidence.enablement_successes += 1
                evidence.support_contexts.add(context)
            else:
                evidence.enablement_failures += 1
        for signature in trial.intervention_signatures:
            evidence = self._mechanic_evidence(
                edge.intervention_evidence,
                signature,
            )
            if available:
                evidence.enablement_successes += 1
                evidence.support_contexts.add(context)
            else:
                evidence.enablement_failures += 1

    @staticmethod
    def _mechanic_evidence(
        evidence_store: Dict[str, CausalMechanicEvidence],
        signature: str,
    ) -> CausalMechanicEvidence:
        evidence = evidence_store.get(signature)
        if evidence is None:
            evidence = CausalMechanicEvidence(signature=signature)
            evidence_store[signature] = evidence
        return evidence

    def _expire_delayed_credit(self) -> None:
        for edge_key, trial in list(self._pending_trials.items()):
            if trial.progress_transition_index < 0:
                continue
            if (
                self._transition_index - trial.progress_transition_index
                <= self.delayed_credit_window
            ):
                continue
            del self._pending_trials[edge_key]
            self._expired_credit_windows += 1

    def state_signature(
        self,
        objective: TerminalObjectiveHypothesis,
        observation: GameObservation,
    ) -> str:
        """Return a position-invariant blocked-state signature for transfer."""
        present = {int(obj.value) for obj in observation.objects}
        relevant = tuple(
            int(color) for color in (objective.source_color, objective.target_color)
            if color is not None
        )
        distance = objective.distance(observation)
        if distance is None:
            bucket = "unmeasurable"
        elif distance <= 0.0:
            bucket = "satisfied"
        elif distance <= 1.0:
            bucket = "one"
        else:
            bucket = "many"
        presence = "_".join(
            f"{color}:{int(color in present)}" for color in relevant
        )
        return (
            f"{objective.family}|{objective.predicate}|{presence}|distance:{bucket}"
        )

    def _propose_preconditions(
        self,
        target: TerminalObjectiveHypothesis,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
    ) -> None:
        target_colors = _objective_colors(target)
        candidates: List[Tuple[Tuple[float, ...], TerminalObjectiveHypothesis, Tuple[int, ...]]] = []
        for source in store.objectives():
            if source.objective_id == target.objective_id:
                continue
            if source.status == TerminalObjectiveStatus.REFUTED:
                continue
            distance = source.distance(observation)
            if distance is None or distance <= 0.0:
                continue
            shared = tuple(sorted(target_colors & _objective_colors(source)))
            overlap = float(len(shared))
            candidates.append(((
                overlap,
                float(source.prior_priority),
                -float(distance),
            ), source, shared))
        if not candidates:
            return
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[: self.max_edges_per_blocked_target]
        for _, source, shared in selected:
            prior = 0.15 + 0.35 * len(shared)
            if not shared:
                prior = 0.05
            self._register_edge(
                source=source,
                target=target,
                shared_colors=shared,
                generation_reason=(
                    "shared_structural_entity_precondition"
                    if shared else "bounded_unstructured_precondition_probe"
                ),
                structural_prior=prior,
            )

    def _register_edge(
        self,
        *,
        source: TerminalObjectiveHypothesis,
        target: TerminalObjectiveHypothesis,
        shared_colors: Sequence[int],
        generation_reason: str,
        structural_prior: float,
    ) -> CausalSubgoalEdge | None:
        key = f"causal::{source.objective_id}=>{target.objective_id}"
        existing = self._edges.get(key)
        if existing is not None:
            existing.generation_reasons.add(str(generation_reason))
            existing.structural_prior = max(
                existing.structural_prior, float(structural_prior)
            )
            return existing
        if len(self._edges) >= self.max_edges:
            evictable = [
                edge for edge in self._edges.values()
                if edge.trials == 0
                and edge.status == CausalSubgoalEdgeStatus.CANDIDATE
            ]
            if not evictable:
                return None
            weakest = min(evictable, key=lambda item: (item.utility, item.edge_key))
            if float(structural_prior) <= weakest.structural_prior:
                return None
            del self._edges[weakest.edge_key]
        edge = CausalSubgoalEdge(
            edge_key=key,
            source_objective_id=source.objective_id,
            target_objective_id=target.objective_id,
            shared_colors=tuple(sorted(int(color) for color in shared_colors)),
            generation_reasons={str(generation_reason)},
            structural_prior=float(structural_prior),
            minimum_independent_support=self.minimum_independent_support,
        )
        self._edges[key] = edge
        self._edges_generated_total += 1
        return edge

    def _record_support(
        self,
        edge: CausalSubgoalEdge,
        context: str,
        *,
        branch_index: int,
    ) -> bool:
        if context in edge.support_contexts:
            return False
        edge.support_contexts.add(context)
        edge.support_branches.add(int(branch_index))
        edge.support_events += 1
        return True

    def _record_contradiction(
        self,
        edge: CausalSubgoalEdge,
        context: str,
        *,
        branch_index: int,
    ) -> bool:
        if context in edge.contradiction_contexts:
            return False
        edge.contradiction_contexts.add(context)
        edge.contradiction_branches.add(int(branch_index))
        edge.contradictions += 1
        return True


def _objective_colors(objective: TerminalObjectiveHypothesis) -> set[int]:
    return {
        int(color) for color in (objective.source_color, objective.target_color)
        if color is not None
    }


def transition_effect_signature(update: Any) -> str:
    """Abstract a real transition effect without retaining absolute positions."""
    diff = update.record.diff
    before_values = Counter(int(value) for value in diff.changed_values_before)
    after_values = Counter(int(value) for value in diff.changed_values_after)
    deltas = {
        value: after_values[value] - before_values[value]
        for value in sorted(set(before_values) | set(after_values))
        if after_values[value] != before_values[value]
    }
    delta_signature = ",".join(
        f"{value}:{delta:+d}" for value, delta in deltas.items()
    ) or "none"
    changed = int(diff.num_changed)
    if changed == 0:
        changed_bucket = "zero"
    elif changed == 1:
        changed_bucket = "one"
    elif changed <= 4:
        changed_bucket = "few"
    else:
        changed_bucket = "many"
    # Player identity is heuristic.  A color conversion can make that heuristic
    # jump between objects, so displacement is trusted only for color-balanced
    # transitions such as actual movement.
    displacement = (
        "none"
        if diff.player_displacement is None or deltas
        else f"{int(diff.player_displacement[0]):+d},{int(diff.player_displacement[1]):+d}"
    )
    before_shape = tuple(
        int(value) for value in np.asarray(update.record.obs_before.raw_grid).shape
    )
    after_shape = tuple(
        int(value) for value in np.asarray(update.record.obs_after.raw_grid).shape
    )
    shape_effect = "same" if before_shape == after_shape else "resized"
    return "|".join((
        f"changed:{changed_bucket}",
        f"colors:{delta_signature}",
        f"move:{displacement}",
        f"shape:{shape_effect}",
        f"game_over:{int(bool(diff.game_over))}",
        f"level_complete:{int(bool(diff.level_complete))}",
    ))


__all__ = [
    "CausalMechanicEvidence",
    "CausalSubgoalEdge",
    "CausalSubgoalEdgeStatus",
    "OnlineCausalSubgoalGraph",
    "transition_effect_signature",
]
