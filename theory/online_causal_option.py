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
from .online_goal_hypothesis import semantic_intervention_signature
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
    target_completed: bool = False
    replay_sequence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CausalOptionSelection:
    option_id: str
    edge_key: str
    target_objective_id: str
    terminal_status: CausalOptionTerminalStatus
    action_name: str
    action_data: Dict[str, Any]
    intervention_signature: str
    selection_utility: float
    replaying_terminal_sequence: bool
    reason: str


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
    ) -> None:
        self.max_options = max(1, int(max_options))
        self.max_downstream_actions = max(1, int(max_downstream_actions))
        self.max_trials_per_signature = max(1, int(max_trials_per_signature))
        self.terminal_credit_window = max(1, int(terminal_credit_window))
        self.minimum_terminal_support = max(1, int(minimum_terminal_support))
        self._options: Dict[str, HierarchicalCausalOption] = {}
        self._active: ActiveCausalOptionRollout | None = None
        self._branch_index = 0
        self._transition_index = 0
        self._compiled_total = 0
        self._selection_count = 0
        self._terminal_events = 0
        self._credited_terminal_events = 0
        self._censored_openings = 0

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
            self._active = ActiveCausalOptionRollout(
                option_id=selected.option_id,
                edge_key=selected.edge_key,
                branch_index=self._branch_index,
                opening_context=(
                    str(context_signature)
                    or f"branch:{self._branch_index}:opening:{selected.opening_events}"
                ),
                opening_transition_index=self._transition_index,
                replay_sequence=selected.best_terminal_sequence,
            )
        return opened

    def select_downstream(
        self,
        observation: GameObservation,
        *,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        preferred_intervention_signatures: Sequence[str] = (),
        action_utilities: Mapping[str, float] | None = None,
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
        if len(active.action_signatures) >= self.max_downstream_actions:
            self._close_active(censored=False)
            return None

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
        for action_name, action_data in candidates:
            signature = semantic_intervention_signature(
                action_name,
                action_data,
                observation,
            )
            branch_attempts = active.signature_attempts.get(signature, 0)
            if branch_attempts >= self.max_trials_per_signature:
                continue
            evidence = option.intervention_evidence.get(signature)
            total_attempts = 0 if evidence is None else evidence.attempts
            productivity = 0.0 if evidence is None else evidence.productivity
            preparation_penalty = int(
                signature in option.productive_preparation_signatures
            )
            replay_match = int(bool(replay_signature) and signature == replay_signature)
            exact_preference = int(signature in preferred)
            family_novelty = int(
                active.action_family_attempts.get(action_name, 0) == 0
            )
            ranked.append((
                (
                    replay_match,
                    exact_preference,
                    int(evidence is not None and evidence.terminal_successes > 0),
                    int(evidence is not None and evidence.nonnoop_effects > 0),
                    family_novelty,
                    float(learned_action_utilities.get(action_name, 0.0)),
                    int(total_attempts == 0),
                    productivity,
                    -preparation_penalty,
                    -branch_attempts,
                    signature,
                ),
                action_name,
                action_data,
                signature,
            ))
        if not ranked:
            self._close_active(censored=False)
            return None
        _, action_name, action_data, signature = max(ranked, key=lambda item: item[0])
        self._selection_count += 1
        return CausalOptionSelection(
            option_id=option.option_id,
            edge_key=option.edge_key,
            target_objective_id=option.target_objective_id,
            terminal_status=option.status,
            action_name=action_name,
            action_data=dict(action_data),
            intervention_signature=signature,
            selection_utility=option.utility,
            replaying_terminal_sequence=bool(replay_signature),
            reason=(
                "replay terminally supported causal-option suffix"
                if replay_signature
                else "bounded downstream search after confirmed causal opening"
            ),
        )

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
                signature = semantic_intervention_signature(
                    update.action,
                    _action_data(update),
                    update.record.obs_before,
                )
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
            outcome.update({
                "participating_action": True,
                "effect_signature": effect_signature,
                "target_progress": target_progress,
                "target_completed": target_completed,
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
            self._active = None
            outcome["rollout_closed"] = True
            return outcome
        if update.record.diff.game_over:
            option.unsafe_rollouts += 1
            if participates and active.action_signatures:
                option.intervention_evidence[
                    active.action_signatures[-1]
                ].unsafe_failures += 1
            self._active = None
            outcome["rollout_closed"] = True
            return outcome
        if (
            len(active.action_signatures) >= self.max_downstream_actions
            or self._rollout_expired(active)
        ):
            self._close_active(censored=False)
            outcome["rollout_closed"] = True
        return outcome

    def start_branch(self) -> None:
        if self._active is not None:
            self._close_active(censored=True)
        self._branch_index += 1

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
            "hypotheses": [option.to_dict() for option in options],
        }

    def _close_active(self, *, censored: bool) -> None:
        active = self._active
        if active is None:
            return
        option = self.option(active.option_id)
        if option is not None:
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

    def _rollout_expired(self, active: ActiveCausalOptionRollout) -> bool:
        return (
            self._transition_index - active.opening_transition_index
            > self.terminal_credit_window
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
