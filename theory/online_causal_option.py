"""Hierarchical exploitation of confirmed online causal dependencies.

A confirmed causal edge says that a preparation makes another intervention
available; it does not say that either subgoal is terminal.  This module keeps
that distinction explicit.  It turns the mechanic into a bounded option,
searches a short suffix only after the opening has actually been observed, and
assigns terminal value to the complete option only from a level change or WIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from v3.schemas import GameObservation

from .online_causal_subgoal_graph import (
    CausalSubgoalEdge,
    CausalSubgoalEdgeStatus,
    transition_effect_signature,
)
from .online_effect_conditioned_subgoal import (
    EffectConditionedSubgoalSelection,
    OnlineEffectConditionedSubgoalStore,
)
from .online_entity_causal_binding import (
    EntityBindingPrediction,
    EntityBindingStatus,
    OnlineEntityCausalBindingStore,
)
from .online_mediated_entity_effect import (
    MediatedEffectPrediction,
    MediatedEffectStatus,
    OnlineMediatedEntityEffectStore,
)
from .online_mediated_discrimination import (
    MediatedDiscriminationPrediction,
    OnlineMediatedDiscriminationStore,
)
from .online_mediated_exploitation import (
    MediatedExploitationPrediction,
    OnlineMediatedExploitationStore,
)
from .online_mediated_replication import (
    MediatedReplicationPrediction,
    OnlineMediatedReplicationStore,
)
from .online_persistent_pursuit import OnlinePersistentPursuitPolicy
from .online_semantic_intervention import (
    SemanticInterventionAnchor,
    semantic_intervention_anchor,
)
from .online_state_conditioned_effect import (
    DirectionalActionPrediction,
    LatentModeRestorationPrediction,
    latent_mode_signature,
)
from .online_terminal_objective import OnlineTerminalObjectiveStore


WIN_STATES = {"WIN", "WON", "VICTORY"}


class CausalOptionTerminalStatus(str, Enum):
    """Terminal status of an option, separate from its confirmed mechanic."""

    CANDIDATE = "candidate"
    NEEDS_CONTRAST = "needs_contrast"
    TERMINAL_SUPPORTED = "terminal_supported"
    TERMINAL_REFUTED = "terminal_refuted"


@dataclass
class CausalOptionInterventionEvidence:
    """Online productivity of one semantic action in an opened state."""

    signature: str
    attempts: int = 0
    nonnoop_effects: int = 0
    objective_progress_events: int = 0
    target_completions: int = 0
    terminal_successes: int = 0
    unsafe_failures: int = 0
    effect_signatures: set[str] = field(default_factory=set)

    @property
    def productivity(self) -> float:
        return (
            self.nonnoop_effects
            + 2.0 * self.objective_progress_events
            + 4.0 * self.terminal_successes
            + 1.0
        ) / (self.attempts + 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "attempts": self.attempts,
            "nonnoop_effects": self.nonnoop_effects,
            "objective_progress_events": self.objective_progress_events,
            "target_completions": self.target_completions,
            "terminal_successes": self.terminal_successes,
            "unsafe_failures": self.unsafe_failures,
            "effect_signatures": sorted(self.effect_signatures),
            "productivity": round(self.productivity, 4),
        }


@dataclass
class HierarchicalCausalOption:
    """A confirmed preparation plus a learned, terminally tested suffix."""

    option_id: str
    edge_key: str
    source_objective_id: str
    target_objective_id: str
    minimum_terminal_support: int = 2
    mechanic_confirmed: bool = True
    productive_preparation_signatures: set[str] = field(default_factory=set)
    compilations: int = 0
    opening_events: int = 0
    rollouts: int = 0
    downstream_actions: int = 0
    downstream_effects: int = 0
    downstream_progress_events: int = 0
    effect_conditioned_progress_events: int = 0
    effect_conditioned_completions: int = 0
    target_completions: int = 0
    nonterminal_rollouts: int = 0
    unsafe_rollouts: int = 0
    terminal_contradictions: int = 0
    terminal_contradiction_branches: set[int] = field(default_factory=set)
    terminal_contexts: set[str] = field(default_factory=set)
    terminal_sequences: Dict[Tuple[str, ...], set[str]] = field(
        default_factory=dict
    )
    failed_sequences: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    intervention_evidence: Dict[str, CausalOptionInterventionEvidence] = field(
        default_factory=dict
    )

    @property
    def terminal_support(self) -> int:
        return len(self.terminal_contexts)

    @property
    def status(self) -> CausalOptionTerminalStatus:
        if self.terminal_support >= max(1, int(self.minimum_terminal_support)):
            return CausalOptionTerminalStatus.TERMINAL_SUPPORTED
        if self.terminal_support > 0:
            return CausalOptionTerminalStatus.NEEDS_CONTRAST
        if len(self.terminal_contradiction_branches) >= 2:
            return CausalOptionTerminalStatus.TERMINAL_REFUTED
        return CausalOptionTerminalStatus.CANDIDATE

    @property
    def best_terminal_sequence(self) -> Tuple[str, ...]:
        if not self.terminal_sequences:
            return ()
        return max(
            self.terminal_sequences,
            key=lambda sequence: (
                len(self.terminal_sequences[sequence]),
                -len(sequence),
                sequence,
            ),
        )

    @property
    def utility(self) -> float:
        status_bonus = {
            CausalOptionTerminalStatus.TERMINAL_SUPPORTED: 20.0,
            CausalOptionTerminalStatus.NEEDS_CONTRAST: 8.0,
            CausalOptionTerminalStatus.CANDIDATE: 1.0,
            CausalOptionTerminalStatus.TERMINAL_REFUTED: -10.0,
        }[self.status]
        progress_rate = (
            self.downstream_progress_events + 1.0
        ) / (self.downstream_actions + 2.0)
        return status_bonus + progress_rate - 0.5 * self.unsafe_rollouts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "edge_key": self.edge_key,
            "source_objective_id": self.source_objective_id,
            "target_objective_id": self.target_objective_id,
            "mechanic_confirmed": self.mechanic_confirmed,
            "productive_preparation_signatures": sorted(
                self.productive_preparation_signatures
            ),
            "status": self.status.value,
            "terminal_support": self.terminal_support,
            "terminal_contradictions": self.terminal_contradictions,
            "independent_terminal_contradiction_branches": len(
                self.terminal_contradiction_branches
            ),
            "compilations": self.compilations,
            "opening_events": self.opening_events,
            "rollouts": self.rollouts,
            "downstream_actions": self.downstream_actions,
            "downstream_effects": self.downstream_effects,
            "downstream_progress_events": self.downstream_progress_events,
            "effect_conditioned_progress_events": (
                self.effect_conditioned_progress_events
            ),
            "effect_conditioned_completions": (
                self.effect_conditioned_completions
            ),
            "target_completions": self.target_completions,
            "nonterminal_rollouts": self.nonterminal_rollouts,
            "unsafe_rollouts": self.unsafe_rollouts,
            "terminal_contexts": sorted(self.terminal_contexts),
            "best_terminal_sequence": list(self.best_terminal_sequence),
            "failed_sequences": [
                {"sequence": list(sequence), "failures": failures}
                for sequence, failures in sorted(self.failed_sequences.items())
            ],
            "utility": round(self.utility, 4),
            "intervention_evidence": [
                evidence.to_dict()
                for evidence in sorted(
                    self.intervention_evidence.values(),
                    key=lambda item: item.signature,
                )
            ],
        }


@dataclass
class ActiveCausalOptionRollout:
    option_id: str
    edge_key: str
    branch_index: int
    opening_context: str
    opening_transition_index: int
    action_signatures: List[str] = field(default_factory=list)
    effect_signatures: List[str] = field(default_factory=list)
    signature_attempts: Dict[str, int] = field(default_factory=dict)
    action_family_attempts: Dict[str, int] = field(default_factory=dict)
    downstream_progress_events: int = 0
    effect_conditioned_progress_events: int = 0
    downstream_subgoal_progress_events: int = 0
    target_completed: bool = False
    replay_sequence: Tuple[str, ...] = ()
    downstream_subgoal_id: str = ""
    downstream_subgoal_start_index: int = 0
    downstream_subgoal_progress_at_start: int = 0
    downstream_subgoal_attempts: Dict[str, int] = field(default_factory=dict)
    downstream_objective_attempts: Dict[str, int] = field(default_factory=dict)
    persistent_pursuit_subgoal_id: str = ""


@dataclass(frozen=True)
class CausalOptionSelection:
    option_id: str
    edge_key: str
    target_objective_id: str
    terminal_status: CausalOptionTerminalStatus
    action_name: str
    action_data: Dict[str, Any]
    intervention_signature: str
    intervention_transfer_signature: str
    selection_utility: float
    replaying_terminal_sequence: bool
    intervention_entity_signature: str = ""
    intervention_structural_role_signature: str = ""
    intervention_instance_signature: str = ""
    entity_anchored_intervention: bool = False
    downstream_subgoal_id: str = ""
    downstream_objective_id: str = ""
    downstream_subgoal_status: str = ""
    downstream_trigger_effect_signature: str = ""
    replaying_progress_sequence: bool = False
    latent_mode_signature: str = ""
    directional_effect_status: str = ""
    directional_expected_gain: float | None = None
    directional_confidence: float | None = None
    directionally_compatible: bool = True
    reversible_action: bool = False
    directional_mode_contrast: bool = False
    directional_structural_transfer: bool = False
    directional_entity_alias_conflict: bool = False
    directional_bridge_target_mode_signature: str = ""
    directional_bridge_followup_action_signature: str = ""
    persistent_pursuit: bool = False
    persistent_attempt_index: int = 0
    persistent_action_limit: int = 0
    entity_binding_status: str = ""
    entity_binding_expected_gain: float | None = None
    entity_binding_confidence: float | None = None
    entity_binding_compatible: bool = True
    entity_binding_controlled_contrast: bool = False
    entity_binding_conflict_observed: bool = False
    entity_binding_track_id: str = ""
    mediated_effect_status: str = ""
    mediated_effect_expected_gain: float | None = None
    mediated_effect_confidence: float | None = None
    mediated_effect_compatible: bool = True
    mediated_effect_controlled_contrast: bool = False
    supported_mediator_signature: str = ""
    candidate_mediator_signatures: Tuple[str, ...] = ()
    mediator_abstraction: bool = False
    mediator_abstraction_specificity: int = 0
    mediator_abstraction_features: Tuple[Tuple[str, str], ...] = ()
    mediated_discrimination_request_id: str = ""
    mediated_discrimination_feature: str = ""
    mediated_discrimination_expected_value: str = ""
    mediated_discrimination_contrast_value: str = ""
    mediated_single_feature_contrast: bool = False
    mediated_discrimination_same_latent_mode: bool = False
    mediated_exploitation_policy_id: str = ""
    mediated_exploitation_candidate_signature: str = ""
    mediated_exploitation_effective_features: Tuple[
        Tuple[str, str], ...
    ] = ()
    mediated_exploitation_same_latent_mode: bool = False
    mediated_restoration_request_id: str = ""
    mediated_restoration_target_mode_signature: str = ""
    mediated_restoration_expected_next_mode_signature: str = ""
    mediated_restoration_path_length: int = 0
    mediated_restoration_confidence: float | None = None
    mediated_exploitation_restoration_policy_id: str = ""
    mediated_replication_request_id: str = ""
    mediated_cross_branch_replication: bool = False
    mediated_exact_semantic_replication: bool = False
    reason: str = ""


class OnlineCausalOptionStore:
    """Compile confirmed edges and search bounded suffixes after real openings."""

    def __init__(
        self,
        *,
        max_options: int = 8,
        max_downstream_actions: int = 6,
        max_trials_per_signature: int = 3,
        terminal_credit_window: int = 8,
        minimum_terminal_support: int = 2,
        enable_effect_conditioned_subgoals: bool = True,
        max_effect_conditioned_subgoals: int = 24,
        max_subgoals_per_effect: int = 4,
        base_downstream_actions: int = 4,
        progress_extension_actions: int = 2,
        max_actions_per_effect_conditioned_subgoal: int = 2,
        enable_state_conditioned_directional_control: bool = True,
        enable_persistent_directional_pursuit: bool = True,
        enable_entity_anchored_interventions: bool = True,
        enable_active_entity_causal_binding: bool = True,
        enable_mediated_entity_effect_induction: bool = True,
        enable_online_mediated_anti_unification: bool = True,
        enable_active_mediated_discrimination: bool = True,
        enable_active_mode_restoration: bool = True,
        enable_terminal_mediated_exploitation: bool = True,
        enable_active_mediated_replication: bool = True,
        max_mediated_replication_attempts: int = 2,
        persistent_actions_per_progress: int = 2,
        max_persistent_actions_per_subgoal: int = 6,
        persistent_rollout_actions_per_progress: int = 2,
        max_persistent_downstream_actions: int = 10,
        persistent_credit_steps_per_progress: int = 4,
        max_persistent_credit_window: int = 16,
    ) -> None:
        self.max_options = max(1, int(max_options))
        self.max_downstream_actions = max(1, int(max_downstream_actions))
        self.max_trials_per_signature = max(1, int(max_trials_per_signature))
        self.terminal_credit_window = max(1, int(terminal_credit_window))
        self.minimum_terminal_support = max(1, int(minimum_terminal_support))
        self.enable_effect_conditioned_subgoals = bool(
            enable_effect_conditioned_subgoals
        )
        self.enable_entity_anchored_interventions = bool(
            enable_entity_anchored_interventions
        )
        self.entity_causal_bindings = OnlineEntityCausalBindingStore(
            enabled=enable_active_entity_causal_binding,
        )
        self.mediated_entity_effects = OnlineMediatedEntityEffectStore(
            enabled=enable_mediated_entity_effect_induction,
            enable_anti_unification=enable_online_mediated_anti_unification,
        )
        self.mediated_replications = OnlineMediatedReplicationStore(
            enabled=(
                enable_mediated_entity_effect_induction
                and enable_active_mediated_replication
            ),
            max_attempts_per_request=max_mediated_replication_attempts,
        )
        self.mediated_discriminations = OnlineMediatedDiscriminationStore(
            enabled=(
                enable_mediated_entity_effect_induction
                and enable_online_mediated_anti_unification
                and enable_active_mediated_discrimination
            ),
            enable_mode_restoration=enable_active_mode_restoration,
            max_attempts_per_request=max_mediated_replication_attempts,
        )
        self.mediated_exploitation = OnlineMediatedExploitationStore(
            enabled=(
                enable_mediated_entity_effect_induction
                and enable_online_mediated_anti_unification
                and enable_active_mediated_discrimination
                and enable_terminal_mediated_exploitation
            ),
        )
        self.base_downstream_actions = min(
            self.max_downstream_actions,
            max(1, int(base_downstream_actions)),
        )
        self.progress_extension_actions = max(
            0,
            int(progress_extension_actions),
        )
        self.max_actions_per_effect_conditioned_subgoal = max(
            1,
            int(max_actions_per_effect_conditioned_subgoal),
        )
        self.downstream_subgoals = OnlineEffectConditionedSubgoalStore(
            max_subgoals=max_effect_conditioned_subgoals,
            max_subgoals_per_effect=max_subgoals_per_effect,
            enable_state_conditioned_directional_control=(
                enable_state_conditioned_directional_control
            ),
        )
        self.persistent_pursuit = OnlinePersistentPursuitPolicy(
            enabled=enable_persistent_directional_pursuit,
            base_actions_per_subgoal=(
                self.max_actions_per_effect_conditioned_subgoal
            ),
            actions_per_progress=persistent_actions_per_progress,
            max_actions_per_subgoal=max_persistent_actions_per_subgoal,
            rollout_actions_per_progress=(
                persistent_rollout_actions_per_progress
            ),
            max_rollout_actions=max_persistent_downstream_actions,
            credit_steps_per_progress=(
                persistent_credit_steps_per_progress
            ),
            max_credit_window=max_persistent_credit_window,
        )
        self._options: Dict[str, HierarchicalCausalOption] = {}
        self._active: ActiveCausalOptionRollout | None = None
        self._branch_index = 0
        self._transition_index = 0
        self._compiled_total = 0
        self._selection_count = 0
        self._terminal_events = 0
        self._credited_terminal_events = 0
        self._censored_openings = 0
        self._dynamic_budget_extensions = 0
        self._budget_pruned_rollouts = 0
        self._anchored_candidate_signatures: set[str] = set()
        self._anchored_transfer_signatures: set[str] = set()
        self._anchored_selections = 0

    @property
    def active_option_id(self) -> str:
        return "" if self._active is None else self._active.option_id

    def options(self) -> List[HierarchicalCausalOption]:
        return sorted(self._options.values(), key=lambda item: item.option_id)

    def option(self, option_id: str) -> HierarchicalCausalOption | None:
        return self._options.get(str(option_id))

    def sync_confirmed_edges(
        self,
        edges: Sequence[CausalSubgoalEdge],
    ) -> List[HierarchicalCausalOption]:
        """Compile only mechanically confirmed dependencies."""
        for edge in edges:
            option_id = f"causal-option::{edge.edge_key}"
            existing = self._options.get(option_id)
            if edge.status != CausalSubgoalEdgeStatus.CONFIRMED:
                if existing is not None:
                    existing.mechanic_confirmed = False
                continue
            productive = {
                signature
                for signature, evidence in edge.intervention_evidence.items()
                if evidence.enablement_successes > 0
            }
            if existing is not None:
                existing.mechanic_confirmed = True
                existing.productive_preparation_signatures.update(productive)
                continue
            if len(self._options) >= self.max_options:
                evictable = [
                    option for option in self._options.values()
                    if option.rollouts == 0 and option.terminal_support == 0
                ]
                if not evictable:
                    continue
                weakest = min(evictable, key=lambda item: (item.utility, item.option_id))
                del self._options[weakest.option_id]
            option = HierarchicalCausalOption(
                option_id=option_id,
                edge_key=edge.edge_key,
                source_objective_id=edge.source_objective_id,
                target_objective_id=edge.target_objective_id,
                minimum_terminal_support=self.minimum_terminal_support,
                productive_preparation_signatures=set(productive),
                compilations=1,
            )
            self._options[option_id] = option
            self._compiled_total += 1
        return self.options()

    def note_openings(
        self,
        edge_keys: Sequence[str],
        *,
        context_signature: str,
    ) -> List[str]:
        """Activate one confirmed option only after observed target availability."""
        opened = []
        candidates = [
            option for option in self.options()
            if option.edge_key in {str(key) for key in edge_keys}
            and option.mechanic_confirmed
            and option.status != CausalOptionTerminalStatus.TERMINAL_REFUTED
        ]
        for option in candidates:
            option.opening_events += 1
            opened.append(option.option_id)
        if not candidates:
            return opened
        selected = max(candidates, key=lambda item: (item.utility, item.option_id))
        if self._active is not None and self._active.option_id != selected.option_id:
            self._close_active(censored=True)
        if self._active is None:
            selected.rollouts += 1
            opening_context = (
                str(context_signature)
                or f"branch:{self._branch_index}:opening:{selected.opening_events}"
            )
            self._active = ActiveCausalOptionRollout(
                option_id=selected.option_id,
                edge_key=selected.edge_key,
                branch_index=self._branch_index,
                opening_context=opening_context,
                opening_transition_index=self._transition_index,
                replay_sequence=selected.best_terminal_sequence,
            )
            self.mediated_replications.note_opening(
                option_id=selected.option_id,
                edge_key=selected.edge_key,
                branch_index=self._branch_index,
                opening_context=opening_context,
            )
            self.mediated_discriminations.note_opening(
                option_id=selected.option_id,
                edge_key=selected.edge_key,
                branch_index=self._branch_index,
                opening_context=opening_context,
            )
            self.mediated_exploitation.note_opening(
                option_id=selected.option_id,
                edge_key=selected.edge_key,
                branch_index=self._branch_index,
                opening_context=opening_context,
            )
        return opened

    def select_effect_conditioned_subgoal(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        safe_actions: Sequence[str] = (),
        click_actions: Sequence[Any] = (),
    ) -> EffectConditionedSubgoalSelection | None:
        """Choose a measurable downstream goal exposed by an observed effect."""
        if not self.enable_effect_conditioned_subgoals:
            return None
        active = self._active
        if active is None or active.branch_index != self._branch_index:
            return None
        effects = {str(item) for item in active.effect_signatures}
        concrete_actions = _concrete_actions(
            safe_actions or observation.available_actions,
            click_actions,
        )
        anchors = [
            self.intervention_anchor(action_name, action_data, observation)
            for action_name, action_data in concrete_actions
        ]
        action_signatures = tuple(
            anchor.concrete_signature
            for anchor in anchors
            for signature in (anchor.concrete_signature,)
            if active.signature_attempts.get(signature, 0)
            < self.max_trials_per_signature
        )
        previous_subgoal_id = active.downstream_subgoal_id
        previous = self.downstream_subgoals.subgoal(previous_subgoal_id)
        previous_actionable = self._persistent_subgoal_actionable(
            subgoal_id=previous_subgoal_id,
            observation=observation,
            store=store,
            action_signatures=action_signatures,
        )
        replication_subgoal_id = self.mediated_replications.preferred_subgoal_id(
            active.option_id
        )
        discrimination_subgoal_id = (
            self.mediated_discriminations.preferred_subgoal_id(
                active.option_id
            )
        )
        exploitation_subgoal_id = (
            self.mediated_exploitation.preferred_subgoal_id(
                active.option_id
            )
        )
        preferred_subgoal_id = (
            exploitation_subgoal_id
            or discrimination_subgoal_id
            or replication_subgoal_id
            or (
            previous_subgoal_id
            if previous is not None
            and previous.progress_events > 0
            and previous_actionable
            else ""
            )
        )
        mode_restoration_reservation = bool(
            exploitation_subgoal_id
            or (
                discrimination_subgoal_id
                and self.mediated_discriminations.enable_mode_restoration
            )
        )
        reserved_subgoal = self.downstream_subgoals.subgoal(
            preferred_subgoal_id if mode_restoration_reservation else ""
        )
        reserved_objective_id = (
            "" if reserved_subgoal is None else reserved_subgoal.objective_id
        )
        selection = self.downstream_subgoals.select(
            option_id=active.option_id,
            observed_effect_signatures=active.effect_signatures,
            observation=observation,
            store=store,
            excluded_subgoal_ids=[
                subgoal_id
                for subgoal_id, attempts in active.downstream_subgoal_attempts.items()
                if not mode_restoration_reservation
                or subgoal_id != preferred_subgoal_id
                if attempts >= self._pursuit_action_limit(
                    subgoal_id,
                    allow_persistent=self._persistent_subgoal_actionable(
                        subgoal_id=subgoal_id,
                        observation=observation,
                        store=store,
                        action_signatures=action_signatures,
                    ),
                )
            ],
            excluded_objective_ids=[
                objective_id
                for objective_id, attempts in active.downstream_objective_attempts.items()
                if not mode_restoration_reservation
                or objective_id != reserved_objective_id
                if attempts >= self._objective_action_limit(
                    option_id=active.option_id,
                    objective_id=objective_id,
                    observed_effect_signatures=effects,
                    observation=observation,
                    store=store,
                    action_signatures=action_signatures,
                )
            ],
            preferred_subgoal_id=preferred_subgoal_id,
            reserve_preferred_context=mode_restoration_reservation,
        )
        if selection is not None:
            subgoal = self.downstream_subgoals.subgoal(selection.subgoal_id)
            attempts = active.downstream_subgoal_attempts.get(
                selection.subgoal_id,
                0,
            )
            persistent_actionable = self._persistent_subgoal_actionable(
                subgoal_id=selection.subgoal_id,
                observation=observation,
                store=store,
                action_signatures=action_signatures,
            )
            previous_persistent_subgoal_id = (
                active.persistent_pursuit_subgoal_id
            )
            active.persistent_pursuit_subgoal_id = (
                selection.subgoal_id
                if subgoal is not None
                and subgoal.progress_events > 0
                and attempts
                >= self.max_actions_per_effect_conditioned_subgoal
                and persistent_actionable
                else ""
            )
            if (
                active.persistent_pursuit_subgoal_id
                and active.persistent_pursuit_subgoal_id
                != previous_persistent_subgoal_id
            ):
                self.persistent_pursuit.note_rollout_budget_extension()
                self.persistent_pursuit.note_credit_window_extension()
            self.persistent_pursuit.note_commitment_selection(
                subgoal_id=selection.subgoal_id,
                previous_subgoal_id=previous_subgoal_id,
                pursuit_progress_events=(
                    0 if subgoal is None else subgoal.progress_events
                ),
                attempts=attempts,
            )
        else:
            active.persistent_pursuit_subgoal_id = ""
        return selection

    def select_downstream(
        self,
        observation: GameObservation,
        *,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        preferred_intervention_signatures: Sequence[str] = (),
        action_utilities: Mapping[str, float] | None = None,
        downstream_subgoal: EffectConditionedSubgoalSelection | None = None,
        preferred_action_name: str = "",
        preferred_action_data: Mapping[str, Any] | None = None,
        directional_predictions: Mapping[
            str,
            DirectionalActionPrediction,
        ] | None = None,
        entity_binding_predictions: Mapping[
            str,
            EntityBindingPrediction,
        ] | None = None,
        mediated_effect_predictions: Mapping[
            str,
            MediatedEffectPrediction,
        ] | None = None,
        mediated_discrimination_predictions: Mapping[
            str,
            MediatedDiscriminationPrediction,
        ] | None = None,
        mediated_exploitation_predictions: Mapping[
            str,
            MediatedExploitationPrediction,
        ] | None = None,
        mediated_restoration_predictions: Mapping[
            str,
            LatentModeRestorationPrediction,
        ] | None = None,
        mediated_exploitation_restoration_predictions: Mapping[
            str,
            LatentModeRestorationPrediction,
        ] | None = None,
        mediated_replication_predictions: Mapping[
            str,
            MediatedReplicationPrediction,
        ] | None = None,
    ) -> CausalOptionSelection | None:
        """Select one bounded suffix action in the actually opened state."""
        active = self._active
        if active is None or active.branch_index != self._branch_index:
            return None
        option = self.option(active.option_id)
        if option is None or not option.mechanic_confirmed:
            self._close_active(censored=True)
            return None
        if self._rollout_expired(active):
            self._close_active(censored=True)
            return None
        action_budget = self._current_action_budget(active)
        if len(active.action_signatures) >= action_budget:
            if action_budget < self._maximum_action_budget():
                self._budget_pruned_rollouts += 1
            self._close_active(censored=False)
            return None

        if downstream_subgoal is not None:
            if active.downstream_subgoal_id != downstream_subgoal.subgoal_id:
                self._close_downstream_pursuit(active, censored=True)
                active.downstream_subgoal_id = downstream_subgoal.subgoal_id
                active.downstream_subgoal_start_index = len(
                    active.action_signatures
                )
                active.downstream_subgoal_progress_at_start = (
                    active.downstream_subgoal_progress_events
                )
        elif active.downstream_subgoal_id:
            self._close_downstream_pursuit(active, censored=True)

        candidates = _concrete_actions(safe_actions, click_actions)
        ranked = []
        preferred = {
            str(signature) for signature in preferred_intervention_signatures
            if str(signature)
        }
        learned_action_utilities = dict(action_utilities or {})
        replay_signature = (
            active.replay_sequence[len(active.action_signatures)]
            if len(active.action_signatures) < len(active.replay_sequence)
            else ""
        )
        progress_replay_signature = ""
        if downstream_subgoal is not None:
            progress_offset = (
                len(active.action_signatures)
                - active.downstream_subgoal_start_index
            )
            if progress_offset < len(downstream_subgoal.best_progress_sequence):
                progress_replay_signature = (
                    downstream_subgoal.best_progress_sequence[progress_offset]
                )
        directed_name = str(preferred_action_name).upper()
        directed_data = dict(preferred_action_data or {})
        predictions = dict(directional_predictions or {})
        binding_predictions = dict(entity_binding_predictions or {})
        mediated_predictions = dict(mediated_effect_predictions or {})
        discrimination_predictions = dict(
            mediated_discrimination_predictions or {}
        )
        exploitation_predictions = dict(
            mediated_exploitation_predictions or {}
        )
        restoration_predictions = dict(
            mediated_restoration_predictions or {}
        )
        exploitation_restoration_predictions = dict(
            mediated_exploitation_restoration_predictions or {}
        )
        replication_predictions = dict(mediated_replication_predictions or {})
        selected_subgoal = (
            None
            if downstream_subgoal is None
            else self.downstream_subgoals.subgoal(
                downstream_subgoal.subgoal_id
            )
        )
        persistent_attempt_index = (
            0
            if downstream_subgoal is None
            else active.downstream_subgoal_attempts.get(
                downstream_subgoal.subgoal_id,
                0,
            )
            + 1
        )
        persistent_action_limit = (
            0
            if downstream_subgoal is None
            else self._pursuit_action_limit(downstream_subgoal.subgoal_id)
        )
        persistent_pursuit = bool(
            self.persistent_pursuit.enabled
            and selected_subgoal is not None
            and selected_subgoal.progress_events > 0
            and persistent_attempt_index
            > self.max_actions_per_effect_conditioned_subgoal
        )
        for action_name, action_data in candidates:
            anchor = self.intervention_anchor(
                action_name,
                action_data,
                observation,
            )
            signature = anchor.concrete_signature
            branch_attempts = active.signature_attempts.get(signature, 0)
            if branch_attempts >= self.max_trials_per_signature:
                continue
            evidence = option.intervention_evidence.get(signature)
            total_attempts = 0 if evidence is None else evidence.attempts
            productivity = 0.0 if evidence is None else evidence.productivity
            preparation_penalty = int(
                bool({
                    signature,
                    anchor.transfer_signature,
                    anchor.legacy_signature,
                } & option.productive_preparation_signatures)
            )
            replay_match = int(bool(replay_signature) and signature == replay_signature)
            progress_replay_match = int(
                bool(progress_replay_signature)
                and signature == progress_replay_signature
            )
            directional_prediction = predictions.get(signature)
            binding_prediction = binding_predictions.get(signature)
            mediated_prediction = mediated_predictions.get(signature)
            discrimination_prediction = discrimination_predictions.get(
                signature
            )
            exploitation_prediction = exploitation_predictions.get(signature)
            restoration_prediction = restoration_predictions.get(signature)
            exploitation_restoration_prediction = (
                exploitation_restoration_predictions.get(signature)
            )
            replication_prediction = replication_predictions.get(signature)
            exploitation_restoration_actionable = bool(
                exploitation_restoration_prediction is not None
                and exploitation_restoration_prediction.compatible
            )
            restoration_actionable = bool(
                restoration_prediction is not None
                and restoration_prediction.compatible
            )
            discrimination_actionable = bool(
                discrimination_prediction is not None
                and discrimination_prediction.compatible
                and discrimination_prediction.cross_branch
                and discrimination_prediction.single_feature_contrast
            )
            exploitation_actionable = bool(
                exploitation_prediction is not None
                and exploitation_prediction.compatible
                and exploitation_prediction.same_latent_mode
            )
            replication_actionable = bool(
                replication_prediction is not None
                and replication_prediction.compatible
                and replication_prediction.cross_branch
            )
            binding_actionable = bool(
                binding_prediction is not None
                and binding_prediction.status in {
                    EntityBindingStatus.PROGRESSIVE_CARRIER,
                    EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST,
                }
            )
            mediated_actionable = bool(
                mediated_prediction is not None
                and mediated_prediction.status in {
                    MediatedEffectStatus.SUPPORTED,
                    MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST,
                }
            )
            if (
                persistent_pursuit
                and binding_prediction is not None
                and not binding_prediction.compatible
                and not mediated_actionable
                and not discrimination_actionable
                and not exploitation_actionable
                and not restoration_actionable
                and not exploitation_restoration_actionable
                and not replication_actionable
                and not replay_match
                and not progress_replay_match
            ):
                self.entity_causal_bindings.note_blocked(binding_prediction)
                continue
            if (
                persistent_pursuit
                and mediated_prediction is not None
                and not mediated_prediction.compatible
                and not replication_actionable
                and not discrimination_actionable
                and not exploitation_actionable
                and not restoration_actionable
                and not exploitation_restoration_actionable
                and not replay_match
                and not progress_replay_match
            ):
                self.mediated_entity_effects.note_blocked(
                    mediated_prediction
                )
                continue
            if (
                persistent_pursuit
                and (
                    (
                        directional_prediction is None
                        or directional_prediction.status.value
                        not in {
                            "progressive",
                            "bridge",
                            "needs_entity_contrast",
                            "needs_mode_contrast",
                        }
                    )
                    and not binding_actionable
                    and not mediated_actionable
                    and not discrimination_actionable
                    and not exploitation_actionable
                    and not restoration_actionable
                    and not exploitation_restoration_actionable
                    and not replication_actionable
                )
                and not replay_match
                and not progress_replay_match
            ):
                continue
            if (
                directional_prediction is not None
                and not directional_prediction.compatible
                and not replication_actionable
                and not discrimination_actionable
                and not exploitation_actionable
                and not restoration_actionable
                and not exploitation_restoration_actionable
                and not replay_match
                and not progress_replay_match
            ):
                self.downstream_subgoals.note_directional_blocked(
                    directional_prediction
                )
                continue
            directional_rank = (
                2
                if directional_prediction is None
                else directional_prediction.selection_rank
            )
            directional_gain = (
                0.0
                if directional_prediction is None
                else directional_prediction.expected_gain
            )
            binding_rank = (
                2
                if not persistent_pursuit or binding_prediction is None
                else binding_prediction.selection_rank
            )
            binding_gain = (
                0.0
                if not persistent_pursuit or binding_prediction is None
                else binding_prediction.expected_gain
            )
            mediated_rank = (
                2
                if not persistent_pursuit or mediated_prediction is None
                else mediated_prediction.selection_rank
            )
            mediated_gain = (
                0.0
                if not persistent_pursuit or mediated_prediction is None
                else mediated_prediction.expected_gain
            )
            replication_rank = (
                -1
                if replication_prediction is None
                else replication_prediction.selection_rank
            )
            discrimination_rank = (
                -1
                if discrimination_prediction is None
                else discrimination_prediction.selection_rank
            )
            exploitation_rank = (
                -1
                if exploitation_prediction is None
                else exploitation_prediction.selection_rank
            )
            restoration_rank = (
                -1
                if restoration_prediction is None
                else restoration_prediction.selection_rank
            )
            exploitation_restoration_rank = (
                -1
                if exploitation_restoration_prediction is None
                else 45
                if exploitation_restoration_prediction.compatible
                else -1
            )
            directed_action_match = int(
                bool(directed_name)
                and action_name == directed_name
                and dict(action_data) == directed_data
            )
            exact_preference = int(bool({
                signature,
                anchor.transfer_signature,
                anchor.legacy_signature,
            } & preferred))
            subgoal_action_utility = (
                0.0
                if downstream_subgoal is None
                else self.downstream_subgoals.action_utility(
                    downstream_subgoal.subgoal_id,
                    signature,
                )
            )
            family_novelty = int(
                active.action_family_attempts.get(action_name, 0) == 0
            )
            ranked.append((
                (
                    exploitation_restoration_rank,
                    exploitation_rank,
                    restoration_rank,
                    discrimination_rank,
                    replication_rank,
                    replay_match,
                    progress_replay_match,
                    directional_rank,
                    directional_gain,
                    mediated_rank,
                    mediated_gain,
                    binding_rank,
                    binding_gain,
                    directed_action_match,
                    exact_preference,
                    int(evidence is not None and evidence.terminal_successes > 0),
                    int(evidence is not None and evidence.nonnoop_effects > 0),
                    family_novelty,
                    float(learned_action_utilities.get(action_name, 0.0)),
                    subgoal_action_utility,
                    int(total_attempts == 0),
                    productivity,
                    -preparation_penalty,
                    -branch_attempts,
                    anchor.legacy_signature,
                ),
                action_name,
                action_data,
                signature,
                directional_prediction,
                binding_prediction,
                mediated_prediction,
                discrimination_prediction,
                exploitation_prediction,
                restoration_prediction,
                exploitation_restoration_prediction,
                replication_prediction,
                anchor,
            ))
        if not ranked:
            self._close_active(censored=False)
            return None
        (
            _,
            action_name,
            action_data,
            signature,
            directional_prediction,
            binding_prediction,
            mediated_prediction,
            discrimination_prediction,
            exploitation_prediction,
            restoration_prediction,
            exploitation_restoration_prediction,
            replication_prediction,
            anchor,
        ) = max(ranked, key=lambda item: item[0])
        selected_exploitation_restoration = bool(
            exploitation_restoration_prediction is not None
            and exploitation_restoration_prediction.compatible
        )
        selected_exploitation = bool(
            not selected_exploitation_restoration
            and exploitation_prediction is not None
            and exploitation_prediction.compatible
        )
        selected_restoration = bool(
            not selected_exploitation_restoration
            and not selected_exploitation
            and restoration_prediction is not None
            and restoration_prediction.compatible
        )
        selected_discrimination = bool(
            not selected_exploitation_restoration
            and not selected_exploitation
            and not selected_restoration
            and discrimination_prediction is not None
            and discrimination_prediction.compatible
        )
        selected_replication = bool(
            not selected_exploitation_restoration
            and not selected_exploitation
            and not selected_restoration
            and not selected_discrimination
            and replication_prediction is not None
            and replication_prediction.compatible
        )
        terminal_replay_selected = bool(
            replay_signature and signature == replay_signature
        )
        progress_replay_selected = bool(
            progress_replay_signature
            and signature == progress_replay_signature
        )
        self._selection_count += 1
        if anchor.anchored:
            self._anchored_selections += 1
        if directional_prediction is not None:
            self.downstream_subgoals.note_directional_selection(
                directional_prediction
            )
        if persistent_pursuit and binding_prediction is not None:
            self.entity_causal_bindings.note_selection(binding_prediction)
        if persistent_pursuit and mediated_prediction is not None:
            self.mediated_entity_effects.note_selection(mediated_prediction)
        if (
            selected_discrimination
        ):
            self.mediated_discriminations.note_selection(
                discrimination_prediction
            )
        exploitation_policy_id = ""
        if (
            selected_exploitation
        ):
            exploitation_policy_id = self.mediated_exploitation.note_selection(
                exploitation_prediction
            )
        restoration_request_id = ""
        if (
            selected_restoration
        ):
            restoration_request_id = (
                self.mediated_discriminations.note_restoration_selection(
                    restoration_prediction
                )
            )
        exploitation_restoration_policy_id = ""
        if (
            selected_exploitation_restoration
        ):
            exploitation_restoration_policy_id = (
                self.mediated_exploitation.note_restoration_selection(
                    exploitation_restoration_prediction
                )
            )
        if (
            selected_replication
        ):
            self.mediated_replications.note_selection(replication_prediction)
        self.persistent_pursuit.note_action_selection(
            subgoal_id=(
                "" if downstream_subgoal is None
                else downstream_subgoal.subgoal_id
            ),
            persistent=persistent_pursuit,
            directional_status=(
                "" if directional_prediction is None
                else directional_prediction.status.value
            ),
            entity_binding_status=(
                "" if binding_prediction is None
                else binding_prediction.status.value
            ),
            mediated_effect_status=(
                "" if mediated_prediction is None
                else mediated_prediction.status.value
            ),
        )
        return CausalOptionSelection(
            option_id=option.option_id,
            edge_key=option.edge_key,
            target_objective_id=option.target_objective_id,
            terminal_status=option.status,
            action_name=action_name,
            action_data=dict(action_data),
            intervention_signature=signature,
            intervention_transfer_signature=anchor.transfer_signature,
            selection_utility=option.utility,
            replaying_terminal_sequence=terminal_replay_selected,
            intervention_entity_signature=anchor.entity_signature,
            intervention_structural_role_signature=(
                anchor.structural_role_signature
            ),
            intervention_instance_signature=anchor.instance_signature,
            entity_anchored_intervention=anchor.anchored,
            downstream_subgoal_id=(
                "" if downstream_subgoal is None else downstream_subgoal.subgoal_id
            ),
            downstream_objective_id=(
                "" if downstream_subgoal is None else downstream_subgoal.objective_id
            ),
            downstream_subgoal_status=(
                "" if downstream_subgoal is None else downstream_subgoal.status.value
            ),
            downstream_trigger_effect_signature=(
                ""
                if downstream_subgoal is None
                else downstream_subgoal.trigger_effect_signature
            ),
            replaying_progress_sequence=progress_replay_selected,
            latent_mode_signature=(
                ""
                if directional_prediction is None
                else directional_prediction.mode_signature
            ),
            directional_effect_status=(
                ""
                if directional_prediction is None
                else directional_prediction.status.value
            ),
            directional_expected_gain=(
                None
                if directional_prediction is None
                else directional_prediction.expected_gain
            ),
            directional_confidence=(
                None
                if directional_prediction is None
                else directional_prediction.confidence
            ),
            directionally_compatible=(
                True
                if directional_prediction is None
                else directional_prediction.compatible
            ),
            reversible_action=(
                False
                if directional_prediction is None
                else directional_prediction.reversible_across_modes
            ),
            directional_mode_contrast=(
                False
                if directional_prediction is None
                else not directional_prediction.exact_mode_evidence
                and directional_prediction.status.value
                == "needs_mode_contrast"
            ),
            directional_structural_transfer=(
                False
                if directional_prediction is None
                else directional_prediction.structural_transfer_evidence
            ),
            directional_entity_alias_conflict=(
                False
                if directional_prediction is None
                else directional_prediction.entity_alias_conflict
            ),
            directional_bridge_target_mode_signature=(
                ""
                if directional_prediction is None
                else directional_prediction.bridge_target_mode_signature
            ),
            directional_bridge_followup_action_signature=(
                ""
                if directional_prediction is None
                else directional_prediction.bridge_followup_action_signature
            ),
            persistent_pursuit=persistent_pursuit,
            persistent_attempt_index=persistent_attempt_index,
            persistent_action_limit=persistent_action_limit,
            entity_binding_status=(
                "" if binding_prediction is None
                else binding_prediction.status.value
            ),
            entity_binding_expected_gain=(
                None if binding_prediction is None
                else binding_prediction.expected_gain
            ),
            entity_binding_confidence=(
                None if binding_prediction is None
                else binding_prediction.confidence
            ),
            entity_binding_compatible=(
                True if binding_prediction is None
                else binding_prediction.compatible
            ),
            entity_binding_controlled_contrast=(
                False if binding_prediction is None
                else binding_prediction.controlled_contrast
            ),
            entity_binding_conflict_observed=(
                False if binding_prediction is None
                else binding_prediction.conflict_observed
            ),
            entity_binding_track_id=(
                "" if binding_prediction is None
                else binding_prediction.track_id
            ),
            mediated_effect_status=(
                "" if mediated_prediction is None
                else mediated_prediction.status.value
            ),
            mediated_effect_expected_gain=(
                None if mediated_prediction is None
                else mediated_prediction.expected_gain
            ),
            mediated_effect_confidence=(
                None if mediated_prediction is None
                else mediated_prediction.confidence
            ),
            mediated_effect_compatible=(
                True if mediated_prediction is None
                else mediated_prediction.compatible
            ),
            mediated_effect_controlled_contrast=(
                False if mediated_prediction is None
                else mediated_prediction.controlled_contrast
            ),
            supported_mediator_signature=(
                "" if mediated_prediction is None
                else mediated_prediction.supported_mediator_signature
            ),
            candidate_mediator_signatures=(
                () if mediated_prediction is None
                else mediated_prediction.candidate_mediator_signatures
            ),
            mediator_abstraction=(
                False if mediated_prediction is None
                else mediated_prediction.mediator_abstraction
            ),
            mediator_abstraction_specificity=(
                0 if mediated_prediction is None
                else mediated_prediction.mediator_abstraction_specificity
            ),
            mediator_abstraction_features=(
                () if mediated_prediction is None
                else mediated_prediction.mediator_abstraction_features
            ),
            mediated_discrimination_request_id=(
                "" if not selected_discrimination
                else discrimination_prediction.request_id
            ),
            mediated_discrimination_feature=(
                "" if not selected_discrimination
                else discrimination_prediction.tested_feature
            ),
            mediated_discrimination_expected_value=(
                "" if not selected_discrimination
                else discrimination_prediction.expected_value
            ),
            mediated_discrimination_contrast_value=(
                "" if not selected_discrimination
                else discrimination_prediction.contrast_value
            ),
            mediated_single_feature_contrast=(
                False if not selected_discrimination
                else discrimination_prediction.single_feature_contrast
            ),
            mediated_discrimination_same_latent_mode=(
                False if not selected_discrimination
                else discrimination_prediction.same_latent_mode
            ),
            mediated_exploitation_policy_id=exploitation_policy_id,
            mediated_exploitation_candidate_signature=(
                "" if not selected_exploitation
                else exploitation_prediction.prospective_candidate_signature
            ),
            mediated_exploitation_effective_features=(
                () if not selected_exploitation
                else exploitation_prediction.effective_features
            ),
            mediated_exploitation_same_latent_mode=bool(
                selected_exploitation
                and exploitation_prediction.same_latent_mode
            ),
            mediated_restoration_request_id=restoration_request_id,
            mediated_restoration_target_mode_signature=(
                restoration_prediction.target_mode_signature
                if selected_restoration
                else (
                    exploitation_restoration_prediction.target_mode_signature
                    if selected_exploitation_restoration
                    else ""
                )
            ),
            mediated_restoration_expected_next_mode_signature=(
                restoration_prediction.expected_next_mode_signature
                if selected_restoration
                else (
                    exploitation_restoration_prediction
                    .expected_next_mode_signature
                    if selected_exploitation_restoration
                    else ""
                )
            ),
            mediated_restoration_path_length=(
                restoration_prediction.path_length
                if selected_restoration
                else (
                    exploitation_restoration_prediction.path_length
                    if selected_exploitation_restoration
                    else 0
                )
            ),
            mediated_restoration_confidence=(
                restoration_prediction.confidence
                if selected_restoration
                else (
                    exploitation_restoration_prediction.confidence
                    if selected_exploitation_restoration
                    else None
                )
            ),
            mediated_exploitation_restoration_policy_id=(
                exploitation_restoration_policy_id
            ),
            mediated_replication_request_id=(
                "" if not selected_replication
                else replication_prediction.request_id
            ),
            mediated_cross_branch_replication=bool(
                selected_replication
                and replication_prediction.cross_branch
            ),
            mediated_exact_semantic_replication=bool(
                selected_replication
                and replication_prediction.exact_semantic_replication
            ),
            reason=(
                "restore exact latent mode for compiled mediated exploitation"
                if selected_exploitation_restoration
                else (
                    "execute revised mediated lattice toward terminal evidence"
                    if selected_exploitation
                    else (
                        "restore exact latent mode for active causal discrimination"
                        if selected_restoration
                        else (
                            "run one-feature mediated-abstraction discrimination"
                            if selected_discrimination
                            else (
                                "run reserved cross-branch mediated-effect replication"
                                if selected_replication
                                else (
                                    "replay terminally supported causal-option suffix"
                                    if terminal_replay_selected
                                    else (
                                        "continue progress-supported directional pursuit"
                                        if persistent_pursuit
                                        else (
                                            "pursue effect-conditioned measurable downstream subgoal"
                                            if downstream_subgoal is not None
                                            else "bounded downstream search after confirmed causal opening"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        )

    def directional_action_predictions(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        downstream_subgoal: EffectConditionedSubgoalSelection | None,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, DirectionalActionPrediction]:
        """Score all concrete suffix actions against the active subgoal mode."""
        if downstream_subgoal is None:
            return {}
        signatures = [
            self.intervention_anchor(
                action_name,
                action_data,
                observation,
            ).concrete_signature
            for action_name, action_data in _concrete_actions(
                safe_actions,
                click_actions,
            )
        ]
        active = self._active
        subgoal = self.downstream_subgoals.subgoal(
            downstream_subgoal.subgoal_id
        )
        persistent_bridge_composition = bool(
            self.persistent_pursuit.enabled
            and active is not None
            and subgoal is not None
            and subgoal.progress_events > 0
            and active.downstream_subgoal_attempts.get(
                downstream_subgoal.subgoal_id,
                0,
            )
            >= self.max_actions_per_effect_conditioned_subgoal
        )
        return self.downstream_subgoals.directional_predictions(
            subgoal_id=downstream_subgoal.subgoal_id,
            observation=observation,
            store=store,
            action_signatures=signatures,
            record_predictions=record_predictions,
            enable_bridge_composition=persistent_bridge_composition,
        )

    def entity_binding_action_predictions(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        downstream_subgoal: EffectConditionedSubgoalSelection | None,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, EntityBindingPrediction]:
        """Score concrete targets using only observed entity/effect bindings."""
        if not self.entity_causal_bindings.enabled:
            return {}
        active = self._active
        option = None if active is None else self.option(active.option_id)
        if active is None or option is None:
            return {}
        objective_id = (
            downstream_subgoal.objective_id
            if downstream_subgoal is not None
            else option.target_objective_id
        )
        objective = store.objective(objective_id)
        if objective is None:
            return {}
        result: Dict[str, EntityBindingPrediction] = {}
        for action_name, action_data in _concrete_actions(
            safe_actions,
            click_actions,
        ):
            anchor = self.intervention_anchor(
                action_name,
                action_data,
                observation,
            )
            if not anchor.anchored:
                continue
            result[anchor.concrete_signature] = (
                self.entity_causal_bindings.predict(
                    option_id=option.option_id,
                    objective=objective,
                    observation=observation,
                    action_signature=anchor.concrete_signature,
                    track_id=self.entity_causal_bindings.track_id_for(
                        observation,
                        anchor,
                        active.branch_index,
                    ),
                    record_prediction=record_predictions,
                )
            )
        return result

    def mediated_effect_action_predictions(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        downstream_subgoal: EffectConditionedSubgoalSelection | None,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, MediatedEffectPrediction]:
        """Score actions from online action-target -> affected-entity edges."""
        if not self.mediated_entity_effects.enabled:
            return {}
        active = self._active
        option = None if active is None else self.option(active.option_id)
        if active is None or option is None:
            return {}
        objective_id = (
            downstream_subgoal.objective_id
            if downstream_subgoal is not None
            else option.target_objective_id
        )
        objective = store.objective(objective_id)
        if objective is None:
            return {}
        result: Dict[str, MediatedEffectPrediction] = {}
        for action_name, action_data in _concrete_actions(
            safe_actions,
            click_actions,
        ):
            anchor = self.intervention_anchor(
                action_name,
                action_data,
                observation,
            )
            if not anchor.anchored:
                continue
            result[anchor.concrete_signature] = (
                self.mediated_entity_effects.predict(
                    option_id=option.option_id,
                    objective=objective,
                    observation=observation,
                    action_signature=anchor.concrete_signature,
                    record_prediction=record_predictions,
                )
            )
        return result

    def mediated_discrimination_action_predictions(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        downstream_subgoal: EffectConditionedSubgoalSelection | None,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, MediatedDiscriminationPrediction]:
        """Find matched contexts varying one learned carrier feature."""
        active = self._active
        option = None if active is None else self.option(active.option_id)
        if active is None or option is None:
            return {}
        objective_id = (
            downstream_subgoal.objective_id
            if downstream_subgoal is not None
            else option.target_objective_id
        )
        objective = store.objective(objective_id)
        if objective is None:
            return {}
        mode = latent_mode_signature(observation, objective)
        result: Dict[str, MediatedDiscriminationPrediction] = {}
        for action_name, action_data in _concrete_actions(
            safe_actions,
            click_actions,
        ):
            anchor = self.intervention_anchor(
                action_name,
                action_data,
                observation,
            )
            prediction = self.mediated_discriminations.predict(
                option_id=option.option_id,
                anchor=anchor,
                observation=observation,
                mode_signature=mode,
                record_prediction=record_predictions,
            )
            if prediction is not None:
                result[anchor.concrete_signature] = prediction
        return result

    def mediated_restoration_action_predictions(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, LatentModeRestorationPrediction]:
        """Compose observed mode transitions before an active contrast."""
        active = self._active
        option = None if active is None else self.option(active.option_id)
        if active is None or option is None:
            return {}
        request = self.mediated_discriminations.restoration_request(
            option.option_id
        )
        if request is None:
            return {}
        objective = store.objective(request.objective_id)
        if objective is None:
            return {}
        current_mode = latent_mode_signature(observation, objective)
        if current_mode == request.mode_signature:
            return {}
        signatures = [
            self.intervention_anchor(
                action_name,
                action_data,
                observation,
            ).concrete_signature
            for action_name, action_data in _concrete_actions(
                safe_actions,
                click_actions,
            )
        ]
        predictions = (
            self.downstream_subgoals.directional_model
            .restoration_predictions(
                option_id=option.option_id,
                objective=objective,
                observation=observation,
                target_mode_signature=request.mode_signature,
                action_signatures=signatures,
            )
        )
        if record_predictions:
            self.mediated_discriminations.note_restoration_predictions(
                len(predictions)
            )
            if not predictions:
                self.mediated_discriminations.note_restoration_unavailable(
                    current_mode
                )
        return predictions

    def mediated_exploitation_action_predictions(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, MediatedExploitationPrediction]:
        """Find interventions satisfying the actively revised carrier lattice."""
        active = self._active
        option = None if active is None else self.option(active.option_id)
        policy = self.mediated_exploitation.active_policy
        if active is None or option is None or policy is None:
            return {}
        objective = store.objective(policy.objective_id)
        if objective is None:
            return {}
        mode = latent_mode_signature(observation, objective)
        result: Dict[str, MediatedExploitationPrediction] = {}
        for action_name, action_data in _concrete_actions(
            safe_actions,
            click_actions,
        ):
            anchor = self.intervention_anchor(
                action_name,
                action_data,
                observation,
            )
            prediction = self.mediated_exploitation.predict(
                option_id=option.option_id,
                anchor=anchor,
                observation=observation,
                mode_signature=mode,
                record_prediction=record_predictions,
            )
            if prediction is not None:
                result[anchor.concrete_signature] = prediction
        return result

    def mediated_exploitation_restoration_action_predictions(
        self,
        observation: GameObservation,
        *,
        store: OnlineTerminalObjectiveStore,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, LatentModeRestorationPrediction]:
        """Restore the exact mode required by a compiled exploitation policy."""
        active = self._active
        option = None if active is None else self.option(active.option_id)
        if active is None or option is None:
            return {}
        policy = self.mediated_exploitation.restoration_policy(
            option.option_id
        )
        if policy is None:
            return {}
        objective = store.objective(policy.objective_id)
        if objective is None:
            return {}
        current_mode = latent_mode_signature(observation, objective)
        if current_mode == policy.mode_signature:
            return {}
        signatures = [
            self.intervention_anchor(
                action_name,
                action_data,
                observation,
            ).concrete_signature
            for action_name, action_data in _concrete_actions(
                safe_actions,
                click_actions,
            )
        ]
        predictions = (
            self.downstream_subgoals.directional_model
            .restoration_predictions(
                option_id=option.option_id,
                objective=objective,
                observation=observation,
                target_mode_signature=policy.mode_signature,
                action_signatures=signatures,
            )
        )
        if record_predictions:
            self.mediated_exploitation.note_restoration_predictions(
                len(predictions)
            )
            if not predictions:
                self.mediated_exploitation.note_restoration_unavailable(
                    current_mode
                )
        return predictions

    def mediated_replication_action_predictions(
        self,
        observation: GameObservation,
        *,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        record_predictions: bool = True,
    ) -> Dict[str, MediatedReplicationPrediction]:
        """Expose only the intervention reserved for a later real opening."""
        active = self._active
        option = None if active is None else self.option(active.option_id)
        if active is None or option is None:
            return {}
        result: Dict[str, MediatedReplicationPrediction] = {}
        for action_name, action_data in _concrete_actions(
            safe_actions,
            click_actions,
        ):
            anchor = self.intervention_anchor(
                action_name,
                action_data,
                observation,
            )
            prediction = self.mediated_replications.predict(
                option_id=option.option_id,
                anchor=anchor,
                record_prediction=record_predictions,
            )
            if prediction is not None:
                result[anchor.concrete_signature] = prediction
        return result

    def observe_transition(
        self,
        update: Any,
        *,
        store: OnlineTerminalObjectiveStore,
        option_id: str = "",
        causal_edge_key: str = "",
        intervention_signature: str = "",
        intervention_id: str = "",
        context_signature: str = "",
        downstream_subgoal_id: str = "",
        replaying_progress_sequence: bool = False,
        persistent_pursuit: bool = False,
        mediated_discrimination_request_id: str = "",
        mediated_exploitation_policy_id: str = "",
        mediated_restoration_request_id: str = "",
        mediated_exploitation_restoration_policy_id: str = "",
        mediated_replication_request_id: str = "",
    ) -> Dict[str, Any]:
        """Revise one active option from real effects and terminal outcomes."""
        self._transition_index += 1
        active = self._active
        terminal_success = bool(
            update.record.diff.level_complete
            or update.record.obs_after.levels_completed
            > update.record.obs_before.levels_completed
            or str(update.record.obs_after.game_state).upper() in WIN_STATES
        )
        outcome = {
            "option_id": str(option_id),
            "active_option_id": self.active_option_id,
            "participating_action": False,
            "effect_signature": "",
            "target_progress": False,
            "target_completed": False,
            "generated_downstream_subgoals": [],
            "effect_conditioned_reduced_objectives": [],
            "downstream_subgoal_id": str(downstream_subgoal_id),
            "downstream_subgoal_progress": False,
            "downstream_subgoal_completed": False,
            "downstream_subgoal_distance_reduction": 0.0,
            "directional_mode_signature": "",
            "directional_effect_status": "",
            "directional_gain": 0.0,
            "directional_reversible_across_modes": False,
            "entity_binding_status": "",
            "entity_binding_gain": 0.0,
            "entity_binding_carrier": False,
            "entity_correspondence_status": "",
            "entity_binding_conflict_observed": False,
            "mediated_effect_status": "",
            "mediated_effect_gain": 0.0,
            "supported_mediator_signature": "",
            "candidate_mediator_signatures": [],
            "mediated_scene_changes": 0,
            "mediator_abstraction": False,
            "mediator_abstraction_specificity": 0,
            "mediator_abstraction_features": {},
            "mediated_discrimination_request_id": str(
                mediated_discrimination_request_id
            ),
            "mediated_discrimination_resolution": "",
            "mediated_discrimination_feature": "",
            "mediated_discrimination_expected_value": "",
            "mediated_discrimination_contrast_value": "",
            "mediated_single_feature_contrast": False,
            "mediated_discrimination_same_latent_mode": False,
            "mediated_exploitation_policy_id": str(
                mediated_exploitation_policy_id
            ),
            "mediated_exploitation_compiled_policy_id": "",
            "mediated_exploitation_status": "",
            "mediated_exploitation_progress": False,
            "mediated_exploitation_nonprogress": False,
            "mediated_exploitation_terminal_supported": False,
            "mediated_exploitation_carrier_observed": False,
            "mediated_restoration_request_id": str(
                mediated_restoration_request_id
            ),
            "mediated_restoration_step_confirmed": False,
            "mediated_restoration_target_reached": False,
            "mediated_restoration_before_mode": "",
            "mediated_restoration_after_mode": "",
            "mediated_exploitation_restoration_policy_id": str(
                mediated_exploitation_restoration_policy_id
            ),
            "mediated_exploitation_restoration_step_confirmed": False,
            "mediated_exploitation_restoration_target_reached": False,
            "mediated_replication_request_id": str(
                mediated_replication_request_id
            ),
            "mediated_replication_resolution": "",
            "persistent_pursuit": bool(persistent_pursuit),
            "persistent_continuation_progress": False,
            "rollout_action_budget": 0,
            "terminal_success": terminal_success,
            "terminal_credited_option": "",
            "rollout_closed": False,
        }
        if active is None:
            return outcome
        option = self.option(active.option_id)
        if option is None:
            self._active = None
            return outcome
        participates = bool(
            str(option_id) == active.option_id
            or str(causal_edge_key) == active.edge_key
        )
        if participates:
            signature = str(intervention_signature)
            if not signature:
                signature = self.intervention_anchor(
                    update.action,
                    _action_data(update),
                    update.record.obs_before,
                ).concrete_signature
            effect_signature = transition_effect_signature(update)
            active.action_signatures.append(signature)
            active.effect_signatures.append(effect_signature)
            active.signature_attempts[signature] = (
                active.signature_attempts.get(signature, 0) + 1
            )
            active.action_family_attempts[update.action] = (
                active.action_family_attempts.get(update.action, 0) + 1
            )
            option.downstream_actions += 1
            evidence = option.intervention_evidence.get(signature)
            if evidence is None:
                evidence = CausalOptionInterventionEvidence(signature=signature)
                option.intervention_evidence[signature] = evidence
            evidence.attempts += 1
            evidence.effect_signatures.add(effect_signature)
            if not update.record.diff.is_noop:
                evidence.nonnoop_effects += 1
                option.downstream_effects += 1
            target = store.objective(option.target_objective_id)
            before_distance = (
                None if target is None else target.distance(update.record.obs_before)
            )
            after_distance = (
                None if target is None else target.distance(update.record.obs_after)
            )
            target_progress = bool(
                before_distance is not None
                and after_distance is not None
                and after_distance < before_distance
            )
            target_completed = bool(
                target_progress
                and before_distance > 0.0
                and after_distance <= 0.0
            )
            if target_progress:
                active.downstream_progress_events += 1
                option.downstream_progress_events += 1
                evidence.objective_progress_events += 1
            if target_completed:
                active.target_completed = True
                option.target_completions += 1
                evidence.target_completions += 1
            link_outcome = {
                "generated_subgoal_ids": [],
                "reduced_objective_ids": [],
                "progress_events": 0,
                "completion_events": 0,
            }
            effect_context = "|".join((
                active.opening_context,
                f"branch:{active.branch_index}",
                str(context_signature) or f"transition:{self._transition_index}",
                f"suffix-step:{len(active.action_signatures)}",
            ))
            pursued_subgoal = (
                self.downstream_subgoals.subgoal(str(downstream_subgoal_id))
                if downstream_subgoal_id
                else None
            )
            previous_pursuit_progress_events = (
                0 if pursued_subgoal is None
                else pursued_subgoal.progress_events
            )
            anchor = self.intervention_anchor(
                update.action,
                _action_data(update),
                update.record.obs_before,
            )
            binding_objective = (
                store.objective(pursued_subgoal.objective_id)
                if pursued_subgoal is not None
                else target
            )
            binding_outcome: Dict[str, Any] = {"observed": False}
            if binding_objective is not None:
                binding_outcome = self.entity_causal_bindings.observe(
                    option_id=option.option_id,
                    objective=binding_objective,
                    observation_before=update.record.obs_before,
                    observation_after=update.record.obs_after,
                    action_name=update.action,
                    action_data=_action_data(update),
                    action_signature=signature,
                    anchor=anchor,
                    branch_index=active.branch_index,
                    context_signature=effect_context,
                    unsafe=bool(update.record.diff.game_over),
                )
            restoration_outcome: Dict[str, Any] = {"observed": False}
            if mediated_restoration_request_id:
                restoration_request = next((
                    item
                    for item in self.mediated_discriminations.requests()
                    if item.request_id
                    == str(mediated_restoration_request_id)
                ), None)
                restoration_objective = (
                    None
                    if restoration_request is None
                    else store.objective(restoration_request.objective_id)
                )
                if restoration_objective is not None:
                    self.downstream_subgoals.directional_model.observe(
                        option_id=option.option_id,
                        objective=restoration_objective,
                        observation_before=update.record.obs_before,
                        observation_after=update.record.obs_after,
                        action_signature=signature,
                        action_transfer_signature=anchor.transfer_signature,
                        effect_signature=effect_signature,
                        branch_index=active.branch_index,
                        context_signature=effect_context,
                        source="pursuit",
                        unsafe=bool(update.record.diff.game_over),
                    )
                    restoration_outcome = (
                        self.mediated_discriminations.observe_restoration(
                            str(mediated_restoration_request_id),
                            before_mode=latent_mode_signature(
                                update.record.obs_before,
                                restoration_objective,
                            ),
                            after_mode=latent_mode_signature(
                                update.record.obs_after,
                                restoration_objective,
                            ),
                        )
                    )
            exploitation_restoration_outcome: Dict[str, Any] = {
                "observed": False
            }
            if mediated_exploitation_restoration_policy_id:
                exploitation_policy = next((
                    item
                    for item in self.mediated_exploitation.policies()
                    if item.policy_id
                    == str(mediated_exploitation_restoration_policy_id)
                ), None)
                exploitation_objective = (
                    None
                    if exploitation_policy is None
                    else store.objective(exploitation_policy.objective_id)
                )
                if exploitation_objective is not None:
                    self.downstream_subgoals.directional_model.observe(
                        option_id=option.option_id,
                        objective=exploitation_objective,
                        observation_before=update.record.obs_before,
                        observation_after=update.record.obs_after,
                        action_signature=signature,
                        action_transfer_signature=anchor.transfer_signature,
                        effect_signature=effect_signature,
                        branch_index=active.branch_index,
                        context_signature=effect_context,
                        source="pursuit",
                        unsafe=bool(update.record.diff.game_over),
                    )
                    exploitation_restoration_outcome = (
                        self.mediated_exploitation.observe_restoration(
                            str(
                                mediated_exploitation_restoration_policy_id
                            ),
                            before_mode=latent_mode_signature(
                                update.record.obs_before,
                                exploitation_objective,
                            ),
                            after_mode=latent_mode_signature(
                                update.record.obs_after,
                                exploitation_objective,
                            ),
                        )
                    )
            mediated_outcome: Dict[str, Any] = {"observed": False}
            if binding_objective is not None:
                mediated_outcome = self.mediated_entity_effects.observe(
                    option_id=option.option_id,
                    objective=binding_objective,
                    observation_before=update.record.obs_before,
                    observation_after=update.record.obs_after,
                    action_data=_action_data(update),
                    action_signature=signature,
                    anchor=anchor,
                    branch_index=active.branch_index,
                    context_signature=effect_context,
                )
                discrimination_request_id = (
                    self.mediated_discriminations.observe_hypothesis(
                        option_id=option.option_id,
                        edge_key=option.edge_key,
                        objective_id=binding_objective.objective_id,
                        downstream_subgoal_id=str(downstream_subgoal_id),
                        anchor=anchor,
                        branch_index=active.branch_index,
                        context_signature=effect_context,
                        mediated_outcome=mediated_outcome,
                        selected_request_id=str(
                            mediated_discrimination_request_id
                        ),
                    )
                )
                if discrimination_request_id:
                    outcome["mediated_discrimination_request_id"] = (
                        discrimination_request_id
                    )
                if mediated_discrimination_request_id:
                    resolved_discrimination = next((
                        item
                        for item in self.mediated_discriminations.requests()
                        if item.request_id
                        == str(mediated_discrimination_request_id)
                    ), None)
                    if resolved_discrimination is not None:
                        outcome["mediated_discrimination_resolution"] = (
                            resolved_discrimination.status.value
                        )
                        outcome["mediated_discrimination_feature"] = (
                            resolved_discrimination.tested_feature
                        )
                        outcome[
                            "mediated_discrimination_expected_value"
                        ] = resolved_discrimination.expected_value
                        outcome[
                            "mediated_discrimination_contrast_value"
                        ] = resolved_discrimination.contrast_value
                        outcome["mediated_single_feature_contrast"] = True
                        outcome[
                            "mediated_discrimination_same_latent_mode"
                        ] = True
                        compiled_policy_id = (
                            self.mediated_exploitation.compile_resolved(
                                resolved_discrimination,
                                branch_index=active.branch_index,
                                context_signature=effect_context,
                                excluded_action_signature=signature,
                            )
                        )
                        if compiled_policy_id:
                            outcome[
                                "mediated_exploitation_compiled_policy_id"
                            ] = compiled_policy_id
                exploitation_outcome: Dict[str, Any] = {"observed": False}
                if mediated_exploitation_policy_id:
                    exploitation_outcome = (
                        self.mediated_exploitation.observe_policy_action(
                            str(mediated_exploitation_policy_id),
                            mediated_outcome=mediated_outcome,
                            terminal_success=terminal_success,
                            unsafe=bool(update.record.diff.game_over),
                            context_signature=effect_context,
                        )
                    )
                    outcome["mediated_exploitation_status"] = str(
                        exploitation_outcome.get("status", "")
                    )
                    outcome["mediated_exploitation_progress"] = bool(
                        exploitation_outcome.get("progress", False)
                    )
                    outcome["mediated_exploitation_nonprogress"] = bool(
                        exploitation_outcome.get("nonprogress", False)
                    )
                    outcome[
                        "mediated_exploitation_terminal_supported"
                    ] = bool(
                        exploitation_outcome.get(
                            "terminal_supported",
                            False,
                        )
                    )
                    outcome[
                        "mediated_exploitation_carrier_observed"
                    ] = bool(
                        exploitation_outcome.get("carrier_observed", False)
                    )
                replication_request_id = (
                    self.mediated_replications.observe_hypothesis(
                        option_id=option.option_id,
                        edge_key=option.edge_key,
                        objective_id=binding_objective.objective_id,
                        downstream_subgoal_id=str(downstream_subgoal_id),
                        anchor=anchor,
                        branch_index=active.branch_index,
                        context_signature=effect_context,
                        mediated_outcome=mediated_outcome,
                        selected_request_id=str(
                            mediated_replication_request_id
                        ),
                    )
                )
                if replication_request_id:
                    outcome["mediated_replication_request_id"] = (
                        replication_request_id
                    )
                if mediated_replication_request_id:
                    resolved = next((
                        item for item in self.mediated_replications.requests()
                        if item.request_id
                        == str(mediated_replication_request_id)
                    ), None)
                    if resolved is not None:
                        outcome["mediated_replication_resolution"] = (
                            resolved.status.value
                        )
            if self.enable_effect_conditioned_subgoals:
                link_outcome = self.downstream_subgoals.link_effect(
                    option_id=option.option_id,
                    effect_signature=effect_signature,
                    observation_before=update.record.obs_before,
                    observation_after=update.record.obs_after,
                    store=store,
                    branch_index=active.branch_index,
                    context_signature=effect_context,
                    action_signature=signature,
                    preferred_objective_id=option.target_objective_id,
                    pursued_objective_id=(
                        "" if pursued_subgoal is None
                        else pursued_subgoal.objective_id
                    ),
                )
            pursuit_outcome = {
                "progress": False,
                "completed": False,
                "distance_reduction": 0.0,
            }
            if (
                self.enable_effect_conditioned_subgoals
                and str(downstream_subgoal_id)
                and str(downstream_subgoal_id) == active.downstream_subgoal_id
            ):
                pursuit_sequence = tuple(
                    active.action_signatures[
                        active.downstream_subgoal_start_index:
                    ]
                )
                pursuit_outcome = (
                    self.downstream_subgoals.observe_pursuit_with_store(
                        subgoal_id=str(downstream_subgoal_id),
                        observation_before=update.record.obs_before,
                        observation_after=update.record.obs_after,
                        store=store,
                        branch_index=active.branch_index,
                        context_signature=effect_context,
                        action_signature=signature,
                        sequence=pursuit_sequence,
                        effect_signature=effect_signature,
                        unsafe=bool(update.record.diff.game_over),
                        replayed=bool(replaying_progress_sequence),
                    )
                )
                active.downstream_subgoal_attempts[
                    str(downstream_subgoal_id)
                ] = (
                    active.downstream_subgoal_attempts.get(
                        str(downstream_subgoal_id),
                        0,
                    )
                    + 1
                )
                if pursued_subgoal is not None:
                    objective_id = pursued_subgoal.objective_id
                    active.downstream_objective_attempts[objective_id] = (
                        active.downstream_objective_attempts.get(
                            objective_id,
                            0,
                        )
                        + 1
                    )
            effect_conditioned_progress = bool(
                int(link_outcome["progress_events"]) > 0
                or pursuit_outcome["progress"]
            )
            budget_before = self._current_action_budget(active)
            credit_window_before = self._current_credit_window(active)
            if pursuit_outcome["progress"]:
                active.downstream_subgoal_progress_events += 1
            if effect_conditioned_progress:
                active.effect_conditioned_progress_events += 1
                option.effect_conditioned_progress_events += 1
                budget_after = self._current_action_budget(active)
                if budget_after > budget_before:
                    self._dynamic_budget_extensions += 1
                    if budget_after > self.max_downstream_actions:
                        self.persistent_pursuit.note_rollout_budget_extension()
                if self._current_credit_window(active) > credit_window_before:
                    self.persistent_pursuit.note_credit_window_extension()
            self.persistent_pursuit.note_transition(
                persistent=bool(persistent_pursuit),
                progress=bool(pursuit_outcome["progress"]),
                previous_progress_events=previous_pursuit_progress_events,
                completed=bool(pursuit_outcome["completed"]),
            )
            effect_conditioned_completions = (
                int(link_outcome["completion_events"])
                + int(bool(pursuit_outcome["completed"]))
            )
            option.effect_conditioned_completions += (
                effect_conditioned_completions
            )
            outcome.update({
                "participating_action": True,
                "effect_signature": effect_signature,
                "target_progress": target_progress,
                "target_completed": target_completed,
                "generated_downstream_subgoals": list(
                    link_outcome["generated_subgoal_ids"]
                ),
                "effect_conditioned_reduced_objectives": list(
                    link_outcome["reduced_objective_ids"]
                ),
                "downstream_subgoal_progress": bool(
                    pursuit_outcome["progress"]
                ),
                "downstream_subgoal_completed": bool(
                    pursuit_outcome["completed"]
                ),
                "downstream_subgoal_distance_reduction": float(
                    pursuit_outcome["distance_reduction"]
                ),
                "directional_mode_signature": pursuit_outcome.get(
                    "directional_mode_signature",
                    "",
                ),
                "directional_effect_status": pursuit_outcome.get(
                    "directional_effect_status",
                    "",
                ),
                "directional_gain": float(
                    pursuit_outcome.get("directional_gain", 0.0)
                ),
                "directional_reversible_across_modes": bool(
                    pursuit_outcome.get(
                        "directional_reversible_across_modes",
                        False,
                    )
                ),
                "entity_binding_status": str(
                    binding_outcome.get("binding_status", "")
                ),
                "entity_binding_gain": float(
                    binding_outcome.get("gain", 0.0)
                ),
                "entity_binding_carrier": bool(
                    binding_outcome.get("carrier", False)
                ),
                "entity_correspondence_status": str(
                    binding_outcome.get("correspondence", {}).get(
                        "status",
                        "",
                    )
                ),
                "entity_binding_conflict_observed": bool(
                    binding_outcome.get("conflict_observed", False)
                ),
                "mediated_effect_status": str(
                    mediated_outcome.get("status", "")
                ),
                "mediated_effect_gain": float(
                    mediated_outcome.get("gain", 0.0)
                ),
                "supported_mediator_signature": str(
                    mediated_outcome.get(
                        "supported_mediator_signature",
                        "",
                    )
                ),
                "candidate_mediator_signatures": list(
                    mediated_outcome.get(
                        "candidate_mediator_signatures",
                        [],
                    )
                ),
                "mediated_scene_changes": sum(
                    int(bool(item.get("changed", False)))
                    for item in mediated_outcome.get(
                        "scene_correspondences",
                        [],
                    )
                ),
                "mediator_abstraction": bool(
                    mediated_outcome.get(
                        "supported_mediator_is_abstract",
                        False,
                    )
                ),
                "mediator_abstraction_specificity": int(
                    (mediated_outcome.get("mediator_abstraction") or {}).get(
                        "specificity",
                        0,
                    )
                ),
                "mediator_abstraction_features": dict(
                    (mediated_outcome.get("mediator_abstraction") or {}).get(
                        "features",
                        {},
                    )
                ),
                "mediated_restoration_step_confirmed": bool(
                    restoration_outcome.get("step_confirmed", False)
                ),
                "mediated_restoration_target_reached": bool(
                    restoration_outcome.get("target_reached", False)
                ),
                "mediated_restoration_before_mode": str(
                    restoration_outcome.get("before_mode", "")
                ),
                "mediated_restoration_after_mode": str(
                    restoration_outcome.get("after_mode", "")
                ),
                "mediated_exploitation_restoration_step_confirmed": bool(
                    exploitation_restoration_outcome.get(
                        "step_confirmed",
                        False,
                    )
                ),
                "mediated_exploitation_restoration_target_reached": bool(
                    exploitation_restoration_outcome.get(
                        "target_reached",
                        False,
                    )
                ),
                "persistent_pursuit": bool(persistent_pursuit),
                "persistent_continuation_progress": bool(
                    persistent_pursuit and pursuit_outcome["progress"]
                ),
                "rollout_action_budget": self._current_action_budget(active),
            })

        if terminal_success:
            self._terminal_events += 1
            if participates:
                sequence = tuple(active.action_signatures)
                context = "|".join((
                    active.opening_context,
                    str(context_signature) or f"transition:{self._transition_index}",
                    ">".join(sequence),
                ))
                option.terminal_contexts.add(context)
                option.terminal_sequences.setdefault(sequence, set()).add(context)
                if active.action_signatures:
                    evidence = option.intervention_evidence[
                        active.action_signatures[-1]
                    ]
                    evidence.terminal_successes += 1
                self._credited_terminal_events += 1
                outcome["terminal_credited_option"] = option.option_id
            self._close_downstream_pursuit(active, censored=True)
            self._active = None
            outcome["rollout_closed"] = True
            return outcome
        if update.record.diff.game_over:
            option.unsafe_rollouts += 1
            if participates and active.action_signatures:
                option.intervention_evidence[
                    active.action_signatures[-1]
                ].unsafe_failures += 1
            self._close_downstream_pursuit(active, censored=True)
            self._active = None
            outcome["rollout_closed"] = True
            return outcome
        if (
            len(active.action_signatures) >= self._current_action_budget(active)
            or self._rollout_expired(active)
        ):
            if (
                len(active.action_signatures) >= self._current_action_budget(active)
                and self._current_action_budget(active)
                < self._maximum_action_budget()
            ):
                self._budget_pruned_rollouts += 1
            self._close_active(censored=False)
            outcome["rollout_closed"] = True
        return outcome

    def start_branch(self) -> None:
        if self._active is not None:
            self._close_active(censored=True)
        self._branch_index += 1
        self.mediated_replications.start_branch(self._branch_index)
        self.mediated_discriminations.start_branch(self._branch_index)
        self.mediated_exploitation.start_branch(self._branch_index)

    def summary(self) -> Dict[str, Any]:
        options = self.options()
        statuses = {
            status.value: sum(option.status == status for option in options)
            for status in CausalOptionTerminalStatus
        }
        return {
            "options": len(options),
            "compiled_total": self._compiled_total,
            "statuses": statuses,
            "active_option_id": self.active_option_id,
            "selections": self._selection_count,
            "opening_events": sum(option.opening_events for option in options),
            "rollouts": sum(option.rollouts for option in options),
            "downstream_actions": sum(
                option.downstream_actions for option in options
            ),
            "downstream_effects": sum(
                option.downstream_effects for option in options
            ),
            "downstream_progress_events": sum(
                option.downstream_progress_events for option in options
            ),
            "effect_conditioned_progress_events": sum(
                option.effect_conditioned_progress_events for option in options
            ),
            "effect_conditioned_completions": sum(
                option.effect_conditioned_completions for option in options
            ),
            "target_completions": sum(
                option.target_completions for option in options
            ),
            "nonterminal_rollouts": sum(
                option.nonterminal_rollouts for option in options
            ),
            "unsafe_rollouts": sum(option.unsafe_rollouts for option in options),
            "terminal_events": self._terminal_events,
            "credited_terminal_events": self._credited_terminal_events,
            "terminal_supported_options": sum(
                option.status == CausalOptionTerminalStatus.TERMINAL_SUPPORTED
                for option in options
            ),
            "terminal_refuted_options": sum(
                option.status == CausalOptionTerminalStatus.TERMINAL_REFUTED
                for option in options
            ),
            "censored_openings": self._censored_openings,
            "dynamic_budget_extensions": self._dynamic_budget_extensions,
            "budget_pruned_rollouts": self._budget_pruned_rollouts,
            "entity_anchored_interventions_enabled": (
                self.enable_entity_anchored_interventions
            ),
            "entity_anchored_candidate_signatures": len(
                self._anchored_candidate_signatures
            ),
            "entity_anchored_transfer_signatures": len(
                self._anchored_transfer_signatures
            ),
            "entity_anchored_selections": self._anchored_selections,
            "active_entity_causal_binding": (
                self.entity_causal_bindings.summary()
            ),
            "mediated_entity_effect_induction": (
                self.mediated_entity_effects.summary()
            ),
            "active_mediated_replication": (
                self.mediated_replications.summary()
            ),
            "active_mediated_discrimination": (
                self.mediated_discriminations.summary()
            ),
            "terminal_mediated_exploitation": (
                self.mediated_exploitation.summary()
            ),
            "effect_conditioned_subgoals_enabled": (
                self.enable_effect_conditioned_subgoals
            ),
            "effect_conditioned_subgoals": (
                self.downstream_subgoals.summary()
            ),
            "persistent_directional_pursuit": (
                self.persistent_pursuit.summary()
            ),
            "hypotheses": [option.to_dict() for option in options],
        }

    def intervention_anchor(
        self,
        action_name: str,
        action_data: Mapping[str, Any],
        observation: GameObservation,
    ) -> SemanticInterventionAnchor:
        anchor = semantic_intervention_anchor(
            action_name,
            action_data,
            observation,
            enabled=self.enable_entity_anchored_interventions,
        )
        if anchor.anchored:
            self._anchored_candidate_signatures.add(anchor.concrete_signature)
            self._anchored_transfer_signatures.add(anchor.transfer_signature)
        return anchor

    def _close_active(self, *, censored: bool) -> None:
        active = self._active
        if active is None:
            return
        option = self.option(active.option_id)
        if option is not None:
            self._close_downstream_pursuit(active, censored=censored)
            if censored:
                self._censored_openings += 1
            else:
                option.nonterminal_rollouts += 1
                sequence = tuple(active.action_signatures)
                option.failed_sequences[sequence] = (
                    option.failed_sequences.get(sequence, 0) + 1
                )
                if active.target_completed:
                    if (
                        active.branch_index
                        not in option.terminal_contradiction_branches
                    ):
                        option.terminal_contradiction_branches.add(
                            active.branch_index
                        )
                        option.terminal_contradictions += 1
        self._active = None

    def _close_downstream_pursuit(
        self,
        active: ActiveCausalOptionRollout,
        *,
        censored: bool,
    ) -> None:
        subgoal_id = active.downstream_subgoal_id
        if not subgoal_id:
            return
        sequence = tuple(
            active.action_signatures[active.downstream_subgoal_start_index:]
        )
        progressed = bool(
            active.downstream_subgoal_progress_events
            > active.downstream_subgoal_progress_at_start
        )
        self.downstream_subgoals.close_pursuit(
            subgoal_id,
            branch_index=active.branch_index,
            sequence=sequence,
            progressed=progressed,
            censored=censored,
        )
        self.persistent_pursuit.close_commitment(subgoal_id)
        if active.persistent_pursuit_subgoal_id == subgoal_id:
            active.persistent_pursuit_subgoal_id = ""
        active.downstream_subgoal_id = ""
        active.downstream_subgoal_start_index = len(active.action_signatures)
        active.downstream_subgoal_progress_at_start = (
            active.downstream_subgoal_progress_events
        )

    def _current_action_budget(
        self,
        active: ActiveCausalOptionRollout,
    ) -> int:
        exploitation_policy = self.mediated_exploitation.active_policy
        if (
            exploitation_policy is not None
            and exploitation_policy.actionable
            and exploitation_policy.option_id == active.option_id
        ):
            return self._maximum_action_budget()
        if not self.enable_effect_conditioned_subgoals:
            return self.max_downstream_actions
        base_budget = min(
            self.max_downstream_actions,
            self.base_downstream_actions
            + self.progress_extension_actions
            * active.effect_conditioned_progress_events,
        )
        return self.persistent_pursuit.rollout_budget(
            base_budget,
            active.downstream_subgoal_progress_events
            if active.persistent_pursuit_subgoal_id
            else 0,
        )

    def _maximum_action_budget(self) -> int:
        if not self.persistent_pursuit.enabled:
            return self.max_downstream_actions
        return max(
            self.max_downstream_actions,
            self.persistent_pursuit.max_rollout_actions,
        )

    def _current_credit_window(
        self,
        active: ActiveCausalOptionRollout,
    ) -> int:
        exploitation_policy = self.mediated_exploitation.active_policy
        if (
            exploitation_policy is not None
            and exploitation_policy.actionable
            and exploitation_policy.option_id == active.option_id
        ):
            return max(
                self.terminal_credit_window,
                self.persistent_pursuit.max_credit_window,
            )
        return self.persistent_pursuit.credit_window(
            self.terminal_credit_window,
            active.downstream_subgoal_progress_events
            if active.persistent_pursuit_subgoal_id
            else 0,
        )

    def _pursuit_action_limit(
        self,
        subgoal_id: str,
        *,
        allow_persistent: bool = True,
    ) -> int:
        subgoal = self.downstream_subgoals.subgoal(str(subgoal_id))
        return self.persistent_pursuit.action_limit(
            0
            if subgoal is None or not allow_persistent
            else subgoal.progress_events
        )

    def _persistent_subgoal_actionable(
        self,
        *,
        subgoal_id: str,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        action_signatures: Sequence[str],
    ) -> bool:
        if not self.persistent_pursuit.enabled or not action_signatures:
            return False
        subgoal = self.downstream_subgoals.subgoal(str(subgoal_id))
        if subgoal is None or subgoal.progress_events <= 0:
            return False
        predictions = self.downstream_subgoals.directional_predictions(
            subgoal_id=subgoal.subgoal_id,
            observation=observation,
            store=store,
            action_signatures=action_signatures,
            record_predictions=False,
            enable_bridge_composition=self.persistent_pursuit.enabled,
        )
        directional_actionable = any(
            prediction.compatible
            and prediction.status.value
            in {
                "progressive",
                "bridge",
                "needs_entity_contrast",
                "needs_mode_contrast",
            }
            for prediction in predictions.values()
        )
        objective = store.objective(subgoal.objective_id)
        binding_actionable = bool(
            objective is not None
            and any(
                prediction.compatible
                and prediction.status in {
                    EntityBindingStatus.PROGRESSIVE_CARRIER,
                    EntityBindingStatus.NEEDS_CONTROLLED_CONTRAST,
                }
                for prediction in (
                    self.entity_causal_bindings.predict(
                        option_id=subgoal.option_id,
                        objective=objective,
                        observation=observation,
                        action_signature=signature,
                        record_prediction=False,
                    )
                    for signature in action_signatures
                )
            )
        )
        mediated_actionable = bool(
            objective is not None
            and any(
                prediction.compatible
                and prediction.status in {
                    MediatedEffectStatus.SUPPORTED,
                    MediatedEffectStatus.NEEDS_MEDIATOR_CONTRAST,
                }
                for prediction in (
                    self.mediated_entity_effects.predict(
                        option_id=subgoal.option_id,
                        objective=objective,
                        observation=observation,
                        action_signature=signature,
                        record_prediction=False,
                    )
                    for signature in action_signatures
                )
            )
        )
        return (
            directional_actionable
            or binding_actionable
            or mediated_actionable
        )

    def _objective_action_limit(
        self,
        *,
        option_id: str,
        objective_id: str,
        observed_effect_signatures: set[str],
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        action_signatures: Sequence[str],
    ) -> int:
        limits = [
            self._pursuit_action_limit(
                subgoal.subgoal_id,
                allow_persistent=self._persistent_subgoal_actionable(
                    subgoal_id=subgoal.subgoal_id,
                    observation=observation,
                    store=store,
                    action_signatures=action_signatures,
                ),
            )
            for subgoal in self.downstream_subgoals.subgoals()
            if subgoal.option_id == str(option_id)
            and subgoal.objective_id == str(objective_id)
            and subgoal.trigger_effect_signature in observed_effect_signatures
        ]
        return max(
            limits,
            default=self.max_actions_per_effect_conditioned_subgoal,
        )

    def _rollout_expired(self, active: ActiveCausalOptionRollout) -> bool:
        return (
            self._transition_index - active.opening_transition_index
            > self._current_credit_window(active)
        )


def _concrete_actions(
    safe_actions: Sequence[str],
    click_actions: Sequence[Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    result: List[Tuple[str, Dict[str, Any]]] = []
    seen: set[Tuple[str, Tuple[Tuple[str, str], ...]]] = set()
    for raw_name in safe_actions:
        name = str(raw_name).upper()
        if name in {"", "RESET"}:
            continue
        candidates = (
            [dict(getattr(action, "action_args", {}) or {}) for action in click_actions]
            if name == "ACTION6" else [{}]
        )
        for action_data in candidates:
            identity = (
                name,
                tuple(sorted((str(key), str(value)) for key, value in action_data.items())),
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append((name, action_data))
    return result


def _action_data(update: Any) -> Mapping[str, Any]:
    action = update.record.action
    result: Dict[str, Any] = {}
    if getattr(action, "x", None) is not None:
        result["x"] = int(action.x)
    if getattr(action, "y", None) is not None:
        result["y"] = int(action.y)
    return result


__all__ = [
    "ActiveCausalOptionRollout",
    "CausalOptionInterventionEvidence",
    "CausalOptionSelection",
    "CausalOptionTerminalStatus",
    "HierarchicalCausalOption",
    "OnlineCausalOptionStore",
]
