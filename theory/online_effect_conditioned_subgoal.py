"""Online downstream subgoals grounded in effects observed during an option.

The store never promotes a terminal claim.  It only remembers that, after a
particular causal-option effect, a measurable objective became available or
moved closer.  Those links can guide a short multi-step suffix and are revised
exclusively from later transitions in the same online interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Sequence, Tuple

from v3.schemas import GameObservation

from .online_terminal_objective import (
    OnlineTerminalObjectiveStore,
    TerminalObjectiveStatus,
)
from .online_state_conditioned_effect import (
    DirectionalActionPrediction,
    OnlineStateConditionedEffectModel,
)


class EffectConditionedSubgoalStatus(str, Enum):
    """Mechanical progress status, deliberately separate from terminal truth."""

    CANDIDATE = "candidate"
    PROGRESS_SUPPORTED = "progress_supported"
    REFUTED = "refuted"


@dataclass
class DownstreamSubgoalActionEvidence:
    """Observed utility of one semantic action for one downstream subgoal."""

    signature: str
    attempts: int = 0
    progress_events: int = 0
    completions: int = 0
    stalls: int = 0
    unsafe_failures: int = 0
    total_distance_reduction: float = 0.0

    @property
    def utility(self) -> float:
        return (
            1.0
            + 3.0 * self.progress_events
            + 2.0 * self.completions
            + self.total_distance_reduction
            - 2.0 * self.unsafe_failures
        ) / (2.0 + self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "attempts": self.attempts,
            "progress_events": self.progress_events,
            "completions": self.completions,
            "stalls": self.stalls,
            "unsafe_failures": self.unsafe_failures,
            "total_distance_reduction": round(
                float(self.total_distance_reduction), 4
            ),
            "utility": round(float(self.utility), 4),
        }


@dataclass
class EffectConditionedDownstreamSubgoal:
    """A falsifiable effect-to-objective link learned within the examination."""

    subgoal_id: str
    option_id: str
    trigger_effect_signature: str
    objective_id: str
    objective_family: str
    prior_priority: float = 0.0
    generation_contexts: set[str] = field(default_factory=set)
    generation_branches: set[int] = field(default_factory=set)
    pursuit_actions: int = 0
    trigger_progress_events: int = 0
    trigger_completions: int = 0
    trigger_total_distance_reduction: float = 0.0
    progress_events: int = 0
    completions: int = 0
    stalls: int = 0
    unsafe_failures: int = 0
    total_distance_reduction: float = 0.0
    progress_contexts: set[str] = field(default_factory=set)
    progress_branches: set[int] = field(default_factory=set)
    contradiction_branches: set[int] = field(default_factory=set)
    successful_sequences: Dict[Tuple[str, ...], set[str]] = field(
        default_factory=dict
    )
    failed_sequences: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    action_evidence: Dict[str, DownstreamSubgoalActionEvidence] = field(
        default_factory=dict
    )

    @property
    def status(self) -> EffectConditionedSubgoalStatus:
        if self.progress_contexts:
            return EffectConditionedSubgoalStatus.PROGRESS_SUPPORTED
        if len(self.contradiction_branches) >= 2:
            return EffectConditionedSubgoalStatus.REFUTED
        return EffectConditionedSubgoalStatus.CANDIDATE

    @property
    def best_progress_sequence(self) -> Tuple[str, ...]:
        if not self.successful_sequences:
            return ()
        return max(
            self.successful_sequences,
            key=lambda sequence: (
                len(self.successful_sequences[sequence]),
                -len(sequence),
                sequence,
            ),
        )

    @property
    def utility(self) -> float:
        status_bonus = {
            EffectConditionedSubgoalStatus.PROGRESS_SUPPORTED: 6.0,
            EffectConditionedSubgoalStatus.CANDIDATE: 1.0,
            EffectConditionedSubgoalStatus.REFUTED: -8.0,
        }[self.status]
        return (
            status_bonus
            + float(self.prior_priority)
            + 0.25 * self.trigger_progress_events
            + (self.progress_events + 1.0) / (self.pursuit_actions + 2.0)
            + 0.25 * self.total_distance_reduction
            - 1.5 * self.unsafe_failures
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "option_id": self.option_id,
            "trigger_effect_signature": self.trigger_effect_signature,
            "objective_id": self.objective_id,
            "objective_family": self.objective_family,
            "status": self.status.value,
            "prior_priority": round(float(self.prior_priority), 4),
            "generation_contexts": sorted(self.generation_contexts),
            "generation_branches": sorted(self.generation_branches),
            "pursuit_actions": self.pursuit_actions,
            "trigger_progress_events": self.trigger_progress_events,
            "trigger_completions": self.trigger_completions,
            "trigger_total_distance_reduction": round(
                float(self.trigger_total_distance_reduction), 4
            ),
            "progress_events": self.progress_events,
            "completions": self.completions,
            "stalls": self.stalls,
            "unsafe_failures": self.unsafe_failures,
            "total_distance_reduction": round(
                float(self.total_distance_reduction), 4
            ),
            "progress_contexts": sorted(self.progress_contexts),
            "progress_branches": sorted(self.progress_branches),
            "contradiction_branches": sorted(self.contradiction_branches),
            "best_progress_sequence": list(self.best_progress_sequence),
            "failed_sequences": [
                {"sequence": list(sequence), "failures": failures}
                for sequence, failures in sorted(self.failed_sequences.items())
            ],
            "utility": round(float(self.utility), 4),
            "action_evidence": [
                evidence.to_dict()
                for evidence in sorted(
                    self.action_evidence.values(),
                    key=lambda item: item.signature,
                )
            ],
        }


@dataclass(frozen=True)
class EffectConditionedSubgoalSelection:
    subgoal_id: str
    option_id: str
    objective_id: str
    trigger_effect_signature: str
    status: EffectConditionedSubgoalStatus
    distance: float
    utility: float
    best_progress_sequence: Tuple[str, ...]


class OnlineEffectConditionedSubgoalStore:
    """Learn effect-conditioned, multi-step objective pursuits online."""

    def __init__(
        self,
        *,
        max_subgoals: int = 24,
        max_subgoals_per_effect: int = 4,
        enable_state_conditioned_directional_control: bool = True,
    ) -> None:
        self.max_subgoals = max(1, int(max_subgoals))
        self.max_subgoals_per_effect = max(1, int(max_subgoals_per_effect))
        self.enable_state_conditioned_directional_control = bool(
            enable_state_conditioned_directional_control
        )
        self.directional_model = OnlineStateConditionedEffectModel()
        self._subgoals: Dict[str, EffectConditionedDownstreamSubgoal] = {}
        self._generated_total = 0
        self._selection_count = 0
        self._guided_actions = 0
        self._effect_links = 0
        self._productive_effect_links = 0
        self._trigger_progress_events = 0
        self._pursuit_progress_events = 0
        self._progress_events = 0
        self._completion_events = 0
        self._replayed_actions = 0
        self._failed_pursuits = 0
        self._censored_pursuits = 0

    def subgoals(self) -> list[EffectConditionedDownstreamSubgoal]:
        return sorted(self._subgoals.values(), key=lambda item: item.subgoal_id)

    def subgoal(
        self,
        subgoal_id: str,
    ) -> EffectConditionedDownstreamSubgoal | None:
        return self._subgoals.get(str(subgoal_id))

    def link_effect(
        self,
        *,
        option_id: str,
        effect_signature: str,
        observation_before: GameObservation,
        observation_after: GameObservation,
        store: OnlineTerminalObjectiveStore,
        branch_index: int,
        context_signature: str,
        action_signature: str,
        preferred_objective_id: str = "",
        pursued_objective_id: str = "",
    ) -> Dict[str, Any]:
        """Generate bounded links from a real effect to measurable objectives."""
        effect = str(effect_signature)
        if not effect or effect.startswith("changed:zero"):
            return {
                "generated_subgoal_ids": [],
                "reduced_objective_ids": [],
                "progress_events": 0,
                "completion_events": 0,
            }
        ranked = []
        for objective in store.objectives():
            if objective.status == TerminalObjectiveStatus.REFUTED:
                continue
            before = objective.distance(observation_before)
            after = objective.distance(observation_after)
            if after is None:
                continue
            reduction = (
                0.0
                if before is None
                else max(0.0, float(before) - float(after))
            )
            newly_measurable = before is None and after is not None
            if after <= 0.0 and reduction <= 0.0:
                continue
            ranked.append((
                (
                    int(reduction > 0.0),
                    int(objective.objective_id == str(preferred_objective_id)),
                    int(newly_measurable),
                    float(objective.prior_priority),
                    -float(after),
                    objective.objective_id,
                ),
                objective,
                before,
                after,
                reduction,
            ))
        ranked.sort(key=lambda item: item[0], reverse=True)

        generated: list[str] = []
        reduced: list[str] = []
        progress_events = 0
        completion_events = 0
        for _, objective, before, after, reduction in ranked[
            : self.max_subgoals_per_effect
        ]:
            subgoal = self._ensure_subgoal(
                option_id=str(option_id),
                effect_signature=effect,
                objective_id=objective.objective_id,
                objective_family=objective.family,
                prior_priority=objective.prior_priority,
            )
            if subgoal is None:
                continue
            context = str(context_signature) or (
                f"branch:{int(branch_index)}:effect:{effect}"
            )
            is_new_link = context not in subgoal.generation_contexts
            subgoal.generation_contexts.add(context)
            subgoal.generation_branches.add(int(branch_index))
            if is_new_link:
                self._effect_links += 1
            generated.append(subgoal.subgoal_id)
            if (
                self.enable_state_conditioned_directional_control
                and action_signature
                and objective.objective_id != str(pursued_objective_id)
            ):
                self.directional_model.observe(
                    option_id=str(option_id),
                    objective=objective,
                    observation_before=observation_before,
                    observation_after=observation_after,
                    action_signature=str(action_signature),
                    effect_signature=effect,
                    branch_index=branch_index,
                    context_signature=context,
                    source="trigger",
                )
            if reduction <= 0.0:
                continue
            completed = bool(
                before is not None and float(before) > 0.0 and float(after) <= 0.0
            )
            recorded = self._record_trigger_progress(
                subgoal,
                reduction=reduction,
                completed=completed,
                branch_index=branch_index,
                context_signature=context,
            )
            if recorded:
                reduced.append(objective.objective_id)
                progress_events += 1
                completion_events += int(completed)
                self._productive_effect_links += 1
        return {
            "generated_subgoal_ids": list(dict.fromkeys(generated)),
            "reduced_objective_ids": list(dict.fromkeys(reduced)),
            "progress_events": progress_events,
            "completion_events": completion_events,
        }

    def select(
        self,
        *,
        option_id: str,
        observed_effect_signatures: Sequence[str],
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        excluded_subgoal_ids: Sequence[str] = (),
        excluded_objective_ids: Sequence[str] = (),
    ) -> EffectConditionedSubgoalSelection | None:
        """Choose the best still-measurable subgoal exposed by seen effects."""
        effects = {str(signature) for signature in observed_effect_signatures}
        excluded = {str(subgoal_id) for subgoal_id in excluded_subgoal_ids}
        excluded_objectives = {
            str(objective_id) for objective_id in excluded_objective_ids
        }
        ranked = []
        for subgoal in self._subgoals.values():
            if subgoal.subgoal_id in excluded:
                continue
            if subgoal.objective_id in excluded_objectives:
                continue
            if subgoal.option_id != str(option_id):
                continue
            if subgoal.trigger_effect_signature not in effects:
                continue
            if subgoal.status == EffectConditionedSubgoalStatus.REFUTED:
                continue
            objective = store.objective(subgoal.objective_id)
            if objective is None or objective.status == TerminalObjectiveStatus.REFUTED:
                continue
            distance = objective.distance(observation)
            if distance is None or distance <= 0.0:
                continue
            assessment = store.assess_objective(objective, observation)
            ranked.append((
                (
                    int(
                        subgoal.status
                        == EffectConditionedSubgoalStatus.PROGRESS_SUPPORTED
                    ),
                    int(
                        objective.status
                        == TerminalObjectiveStatus.TERMINAL_SUPPORTED
                    ),
                    float(subgoal.utility),
                    float(assessment.priority)
                    if assessment.priority != float("-inf")
                    else float(objective.prior_priority),
                    -float(distance),
                    -subgoal.pursuit_actions,
                    subgoal.subgoal_id,
                ),
                subgoal,
                float(distance),
            ))
        if not ranked:
            return None
        _, subgoal, distance = max(ranked, key=lambda item: item[0])
        self._selection_count += 1
        return EffectConditionedSubgoalSelection(
            subgoal_id=subgoal.subgoal_id,
            option_id=subgoal.option_id,
            objective_id=subgoal.objective_id,
            trigger_effect_signature=subgoal.trigger_effect_signature,
            status=subgoal.status,
            distance=distance,
            utility=subgoal.utility,
            best_progress_sequence=subgoal.best_progress_sequence,
        )

    def action_utility(self, subgoal_id: str, signature: str) -> float:
        subgoal = self.subgoal(subgoal_id)
        if subgoal is None:
            return 0.0
        evidence = subgoal.action_evidence.get(str(signature))
        return 0.0 if evidence is None else float(evidence.utility)

    def directional_predictions(
        self,
        *,
        subgoal_id: str,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        action_signatures: Sequence[str],
    ) -> Dict[str, DirectionalActionPrediction]:
        """Predict signed objective effects for concrete actions in this mode."""
        if not self.enable_state_conditioned_directional_control:
            return {}
        subgoal = self.subgoal(subgoal_id)
        objective = None if subgoal is None else store.objective(
            subgoal.objective_id
        )
        if subgoal is None or objective is None:
            return {}
        return {
            str(signature): self.directional_model.predict(
                option_id=subgoal.option_id,
                objective=objective,
                observation=observation,
                action_signature=str(signature),
            )
            for signature in dict.fromkeys(str(item) for item in action_signatures)
        }

    def note_directional_selection(
        self,
        prediction: DirectionalActionPrediction,
    ) -> None:
        self.directional_model.note_selection(prediction)

    def note_directional_blocked(
        self,
        prediction: DirectionalActionPrediction,
    ) -> None:
        self.directional_model.note_blocked(prediction)

    def observe_pursuit_with_store(
        self,
        *,
        subgoal_id: str,
        observation_before: GameObservation,
        observation_after: GameObservation,
        store: OnlineTerminalObjectiveStore,
        branch_index: int,
        context_signature: str,
        action_signature: str,
        sequence: Sequence[str],
        effect_signature: str = "",
        unsafe: bool = False,
        replayed: bool = False,
    ) -> Dict[str, Any]:
        """Store-aware pursuit revision used by the live controller."""
        subgoal = self.subgoal(subgoal_id)
        objective = None if subgoal is None else store.objective(subgoal.objective_id)
        if subgoal is None or objective is None:
            return {
                "subgoal_id": str(subgoal_id),
                "progress": False,
                "completed": False,
                "distance_reduction": 0.0,
            }
        self._guided_actions += 1
        if replayed:
            self._replayed_actions += 1
        subgoal.pursuit_actions += 1
        evidence = subgoal.action_evidence.setdefault(
            str(action_signature),
            DownstreamSubgoalActionEvidence(signature=str(action_signature)),
        )
        evidence.attempts += 1
        outcome = self._observe_pursuit_with_distances(
            subgoal=subgoal,
            evidence=evidence,
            distance_before=objective.distance(observation_before),
            distance_after=objective.distance(observation_after),
            branch_index=branch_index,
            context_signature=context_signature,
            action_signature=action_signature,
            sequence=sequence,
            unsafe=unsafe,
        )
        if self.enable_state_conditioned_directional_control:
            directional = self.directional_model.observe(
                option_id=subgoal.option_id,
                objective=objective,
                observation_before=observation_before,
                observation_after=observation_after,
                action_signature=str(action_signature),
                effect_signature=str(effect_signature),
                branch_index=branch_index,
                context_signature=context_signature,
                source="pursuit",
                unsafe=unsafe,
            )
            outcome.update({
                "directional_mode_signature": directional["mode_signature"],
                "directional_effect_status": directional["status"],
                "directional_gain": directional["gain"],
                "directional_reversible_across_modes": directional[
                    "reversible_across_modes"
                ],
            })
        return outcome

    def close_pursuit(
        self,
        subgoal_id: str,
        *,
        branch_index: int,
        sequence: Sequence[str],
        progressed: bool,
        censored: bool,
    ) -> None:
        subgoal = self.subgoal(subgoal_id)
        if subgoal is None:
            return
        if censored:
            self._censored_pursuits += 1
            return
        if progressed:
            return
        normalized = tuple(str(item) for item in sequence)
        subgoal.failed_sequences[normalized] = (
            subgoal.failed_sequences.get(normalized, 0) + 1
        )
        subgoal.contradiction_branches.add(int(branch_index))
        self._failed_pursuits += 1

    def summary(self) -> Dict[str, Any]:
        subgoals = self.subgoals()
        return {
            "subgoals": len(subgoals),
            "generated_total": self._generated_total,
            "statuses": {
                status.value: sum(item.status == status for item in subgoals)
                for status in EffectConditionedSubgoalStatus
            },
            "effect_links": self._effect_links,
            "productive_effect_links": self._productive_effect_links,
            "selections": self._selection_count,
            "guided_actions": self._guided_actions,
            "progress_events": self._progress_events,
            "trigger_progress_events": self._trigger_progress_events,
            "pursuit_progress_events": self._pursuit_progress_events,
            "completion_events": self._completion_events,
            "replayed_actions": self._replayed_actions,
            "failed_pursuits": self._failed_pursuits,
            "censored_pursuits": self._censored_pursuits,
            "state_conditioned_directional_control_enabled": (
                self.enable_state_conditioned_directional_control
            ),
            "state_conditioned_directional_model": (
                self.directional_model.summary()
            ),
            "hypotheses": [subgoal.to_dict() for subgoal in subgoals],
        }

    def _ensure_subgoal(
        self,
        *,
        option_id: str,
        effect_signature: str,
        objective_id: str,
        objective_family: str,
        prior_priority: float,
    ) -> EffectConditionedDownstreamSubgoal | None:
        identity = (
            f"effect-subgoal::{option_id}::{effect_signature}=>{objective_id}"
        )
        existing = self._subgoals.get(identity)
        if existing is not None:
            existing.prior_priority = max(
                existing.prior_priority,
                float(prior_priority),
            )
            return existing
        if len(self._subgoals) >= self.max_subgoals:
            evictable = [
                item for item in self._subgoals.values()
                if item.status == EffectConditionedSubgoalStatus.CANDIDATE
                and item.pursuit_actions == 0
            ]
            if not evictable:
                return None
            weakest = min(evictable, key=lambda item: (item.utility, item.subgoal_id))
            if weakest.utility >= 1.0 + float(prior_priority):
                return None
            del self._subgoals[weakest.subgoal_id]
        subgoal = EffectConditionedDownstreamSubgoal(
            subgoal_id=identity,
            option_id=str(option_id),
            trigger_effect_signature=str(effect_signature),
            objective_id=str(objective_id),
            objective_family=str(objective_family),
            prior_priority=float(prior_priority),
        )
        self._subgoals[identity] = subgoal
        self._generated_total += 1
        return subgoal

    def _record_progress(
        self,
        subgoal: EffectConditionedDownstreamSubgoal,
        *,
        reduction: float,
        completed: bool,
        branch_index: int,
        context_signature: str,
        sequence: Sequence[str],
        action_signature: str,
        count_pursuit_action: bool,
    ) -> bool:
        context = str(context_signature) or (
            f"branch:{int(branch_index)}:progress:{subgoal.progress_events}"
        )
        if context in subgoal.progress_contexts:
            return False
        subgoal.progress_contexts.add(context)
        subgoal.progress_branches.add(int(branch_index))
        subgoal.progress_events += 1
        subgoal.total_distance_reduction += float(reduction)
        subgoal.completions += int(completed)
        normalized = tuple(str(item) for item in sequence if str(item))
        if normalized:
            subgoal.successful_sequences.setdefault(normalized, set()).add(context)
        evidence = None
        if action_signature:
            evidence = subgoal.action_evidence.setdefault(
                str(action_signature),
                DownstreamSubgoalActionEvidence(signature=str(action_signature)),
            )
        if evidence is not None:
            if count_pursuit_action and evidence.attempts <= 0:
                evidence.attempts += 1
            evidence.progress_events += 1
            evidence.total_distance_reduction += float(reduction)
            evidence.completions += int(completed)
        self._progress_events += 1
        self._pursuit_progress_events += 1
        self._completion_events += int(completed)
        return True

    def _record_trigger_progress(
        self,
        subgoal: EffectConditionedDownstreamSubgoal,
        *,
        reduction: float,
        completed: bool,
        branch_index: int,
        context_signature: str,
    ) -> bool:
        context = str(context_signature) or (
            f"branch:{int(branch_index)}:trigger:{subgoal.trigger_progress_events}"
        )
        trigger_context = f"trigger::{context}"
        if trigger_context in subgoal.generation_contexts:
            return False
        subgoal.generation_contexts.add(trigger_context)
        subgoal.trigger_progress_events += 1
        subgoal.trigger_completions += int(completed)
        subgoal.trigger_total_distance_reduction += float(reduction)
        self._progress_events += 1
        self._trigger_progress_events += 1
        self._completion_events += int(completed)
        return True

    def _observe_pursuit_with_distances(
        self,
        *,
        subgoal: EffectConditionedDownstreamSubgoal,
        evidence: DownstreamSubgoalActionEvidence,
        distance_before: float | None,
        distance_after: float | None,
        branch_index: int,
        context_signature: str,
        action_signature: str,
        sequence: Sequence[str],
        unsafe: bool,
    ) -> Dict[str, Any]:
        if unsafe:
            subgoal.unsafe_failures += 1
            evidence.unsafe_failures += 1
        reduction = (
            0.0
            if distance_before is None or distance_after is None
            else max(0.0, float(distance_before) - float(distance_after))
        )
        completed = bool(
            distance_before is not None
            and distance_after is not None
            and float(distance_before) > 0.0
            and float(distance_after) <= 0.0
        )
        progress = reduction > 0.0
        if progress:
            self._record_progress(
                subgoal,
                reduction=reduction,
                completed=completed,
                branch_index=branch_index,
                context_signature=context_signature,
                sequence=sequence,
                action_signature=action_signature,
                count_pursuit_action=False,
            )
        else:
            subgoal.stalls += 1
            evidence.stalls += 1
        return {
            "subgoal_id": subgoal.subgoal_id,
            "objective_id": subgoal.objective_id,
            "progress": progress,
            "completed": completed,
            "distance_before": distance_before,
            "distance_after": distance_after,
            "distance_reduction": reduction,
        }

__all__ = [
    "DownstreamSubgoalActionEvidence",
    "EffectConditionedDownstreamSubgoal",
    "EffectConditionedSubgoalSelection",
    "EffectConditionedSubgoalStatus",
    "OnlineEffectConditionedSubgoalStore",
]
