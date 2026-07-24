"""Terminal-only exploration beyond locally completed goal hypotheses.

SAGE can often drive a measurable hypothesis to its postcondition without
finishing the level.  Such a state is useful evidence about *where the current
goal stops being sufficient*, but it is not positive terminal evidence.  This
module keeps those states as negative terminal frontiers and runs a bounded
continuation from them.  Repeated frontiers whose current bound is exhausted
receive a larger continuation horizon, without using intermediate progress as
evidence.  A continuation is credited only when the environment reports a
level change or a win.  When the bounded horizon expires, a dormant lineage
can keep observing the unchanged live policy.  A later terminal event only
nominates that lineage; exact terminal replay is required before credit.

The explorer is deliberately agnostic to game identity and objective family.
It receives stable state signatures, objective identifiers, and concrete legal
actions observed by the live controller.  Failed local progress never promotes
a continuation.  SAGE.9i may also supply a generic structural frontier without
an objective postcondition.  SAGE.9j forces its first terminal to remain
candidate-only until exact same-frontier replay.  SAGE.9k then pre-registers
causal block cuts and credits a shorter sequence only when its own live test
reaches a terminal.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TerminalFrontierAction:
    """One concrete action eligible for a bounded frontier suffix."""

    action_name: str
    action_data: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def from_parts(
        cls,
        action_name: str,
        action_data: Mapping[str, Any] | None = None,
    ) -> "TerminalFrontierAction":
        return cls(
            action_name=str(action_name).upper(),
            action_data=tuple(sorted(dict(action_data or {}).items())),
        )

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self.action_data)

    @property
    def signature(self) -> str:
        return f"{self.action_name}|{self.action_data}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_name,
            "action_data": self.data,
        }


@dataclass(frozen=True)
class TerminalFrontierSelection:
    """One action selected as part of a terminally evaluated suffix."""

    frontier_id: str
    objective_ids: Tuple[str, ...]
    action: TerminalFrontierAction
    step_index: int
    action_limit: int
    replaying_successful_continuation: bool
    replaying_dormant_terminal_candidate: bool
    testing_causal_reduction: bool
    testing_structural_transfer: bool
    reason: str


@dataclass
class SuccessfulContinuation:
    """A suffix whose final observed transition changed level or won."""

    actions: Tuple[TerminalFrontierAction, ...]
    state_signatures: Tuple[str, ...]
    confirmations: int = 1
    causal_reduction: bool = False
    removed_action_indices: Tuple[int, ...] = ()
    structural_transfer: bool = False
    source_frontier_id: str = ""

    @property
    def signature(self) -> Tuple[str, ...]:
        return tuple(action.signature for action in self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "state_signatures": list(self.state_signatures),
            "confirmations": self.confirmations,
            "causal_reduction": self.causal_reduction,
            "removed_action_indices": list(self.removed_action_indices),
            "structural_transfer": self.structural_transfer,
            "source_frontier_id": self.source_frontier_id,
        }


@dataclass
class DormantTerminalContinuation:
    """A delayed-terminal lineage awaiting exact online replay."""

    actions: Tuple[TerminalFrontierAction, ...]
    state_signatures: Tuple[str, ...]
    level_progressed: bool
    won: bool
    terminal_observations: int = 1
    replay_attempts: int = 0
    confirmations: int = 0
    refutations: int = 0
    divergences: int = 0

    @property
    def signature(self) -> Tuple[str, ...]:
        return tuple(action.signature for action in self.actions)

    @property
    def status(self) -> str:
        if self.confirmations:
            return "terminal_confirmed"
        if self.refutations:
            return "refuted"
        if self.divergences:
            return "inconclusive_divergence"
        return "awaiting_replay"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "state_signatures": list(self.state_signatures),
            "level_progressed": self.level_progressed,
            "won": self.won,
            "terminal_observations": self.terminal_observations,
            "replay_attempts": self.replay_attempts,
            "confirmations": self.confirmations,
            "refutations": self.refutations,
            "divergences": self.divergences,
            "status": self.status,
        }


@dataclass
class CausalReductionProbe:
    """A pre-registered action-block cut tested only by terminal outcome."""

    probe_id: str
    source_signature: Tuple[str, ...]
    actions: Tuple[TerminalFrontierAction, ...]
    removed_action_indices: Tuple[int, ...]
    strategy: str
    generation: int = 1
    parent_probe_id: str = ""
    kept_origin_indices: Tuple[int, ...] = ()
    replay_attempts: int = 0
    confirmations: int = 0
    refutations: int = 0
    divergences: int = 0

    @property
    def status(self) -> str:
        if self.confirmations:
            return "terminal_confirmed"
        if self.refutations:
            return "refuted"
        if self.divergences:
            return "inconclusive_divergence"
        return "awaiting_test"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "source_signature": list(self.source_signature),
            "actions": [action.to_dict() for action in self.actions],
            "removed_action_indices": list(self.removed_action_indices),
            "strategy": self.strategy,
            "generation": self.generation,
            "parent_probe_id": self.parent_probe_id,
            "kept_origin_indices": list(self.kept_origin_indices),
            "replay_attempts": self.replay_attempts,
            "confirmations": self.confirmations,
            "refutations": self.refutations,
            "divergences": self.divergences,
            "status": self.status,
        }


@dataclass
class FrontierAcquisitionPath:
    """Exact online path from a branch reset state to a frontier."""

    actions: Tuple[TerminalFrontierAction, ...]
    state_signatures: Tuple[str, ...]
    attempts: int = 0
    confirmations: int = 0
    divergences: int = 0
    censored: int = 0

    @property
    def signature(self) -> Tuple[str, ...]:
        return (
            tuple(action.signature for action in self.actions)
            + tuple(f"state::{state}" for state in self.state_signatures)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "state_signatures": list(self.state_signatures),
            "attempts": self.attempts,
            "confirmations": self.confirmations,
            "divergences": self.divergences,
            "censored": self.censored,
        }


@dataclass
class StructuralTransferProbe:
    """Outcome-blind continuation transfer between equivalent frontiers."""

    probe_id: str
    source_frontier_id: str
    source_continuation_signature: Tuple[str, ...]
    actions: Tuple[TerminalFrontierAction, ...]
    attempts: int = 0
    confirmations: int = 0
    refutations: int = 0
    divergences: int = 0

    @property
    def status(self) -> str:
        if self.confirmations:
            return "terminal_confirmed"
        if self.refutations:
            return "refuted"
        if self.divergences:
            return "inconclusive_divergence"
        return "awaiting_test"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "source_frontier_id": self.source_frontier_id,
            "source_continuation_signature": list(
                self.source_continuation_signature
            ),
            "actions": [action.to_dict() for action in self.actions],
            "attempts": self.attempts,
            "confirmations": self.confirmations,
            "refutations": self.refutations,
            "divergences": self.divergences,
            "status": self.status,
        }


@dataclass
class TerminalNegativeFrontier:
    """A postcondition state observed without terminal success."""

    frontier_id: str
    state_signature: str
    objective_ids: Tuple[str, ...]
    frontier_kind: str = "objective_postcondition"
    structural_equivalence_signature: str = ""
    structural_trigger_signatures: set[str] = field(default_factory=set)
    structural_trigger_families: set[str] = field(default_factory=set)
    context_signatures: set[str] = field(default_factory=set)
    captures: int = 0
    trials: int = 0
    terminal_credits: int = 0
    nonterminal_suffixes: int = 0
    unsafe_suffixes: int = 0
    censored_suffixes: int = 0
    allocated_action_limit: int = 0
    horizon_extensions: int = 0
    horizon_history: list[int] = field(default_factory=list)
    longest_suffix_actions: int = 0
    successful_continuations: Dict[
        Tuple[str, ...], SuccessfulContinuation
    ] = field(default_factory=dict)
    dormant_terminal_candidates: Dict[
        Tuple[str, ...], DormantTerminalContinuation
    ] = field(default_factory=dict)
    causal_reduction_probes: Dict[str, CausalReductionProbe] = field(
        default_factory=dict
    )
    acquisition_paths: Dict[
        Tuple[str, ...], FrontierAcquisitionPath
    ] = field(default_factory=dict)
    structural_transfer_probes: Dict[
        str, StructuralTransferProbe
    ] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "state_signature": self.state_signature,
            "objective_ids": list(self.objective_ids),
            "frontier_kind": self.frontier_kind,
            "structural_equivalence_signature": (
                self.structural_equivalence_signature
            ),
            "structural_trigger_signatures": sorted(
                self.structural_trigger_signatures
            ),
            "structural_trigger_families": sorted(
                self.structural_trigger_families
            ),
            "context_signatures": sorted(self.context_signatures),
            "captures": self.captures,
            "trials": self.trials,
            "terminal_credits": self.terminal_credits,
            "nonterminal_suffixes": self.nonterminal_suffixes,
            "unsafe_suffixes": self.unsafe_suffixes,
            "censored_suffixes": self.censored_suffixes,
            "allocated_action_limit": self.allocated_action_limit,
            "horizon_extensions": self.horizon_extensions,
            "horizon_history": list(self.horizon_history),
            "longest_suffix_actions": self.longest_suffix_actions,
            "successful_continuations": [
                continuation.to_dict()
                for _, continuation in sorted(
                    self.successful_continuations.items()
                )
            ],
            "dormant_terminal_candidates": [
                continuation.to_dict()
                for _, continuation in sorted(
                    self.dormant_terminal_candidates.items()
                )
            ],
            "causal_reduction_probes": [
                probe.to_dict()
                for _, probe in sorted(self.causal_reduction_probes.items())
            ],
            "acquisition_paths": [
                path.to_dict()
                for _, path in sorted(self.acquisition_paths.items())
            ],
            "structural_transfer_probes": [
                probe.to_dict()
                for _, probe in sorted(
                    self.structural_transfer_probes.items()
                )
            ],
        }


@dataclass
class _ActiveSuffix:
    frontier_id: str
    actions: list[TerminalFrontierAction]
    state_signatures: list[str]
    action_limit: int
    replay: SuccessfulContinuation | None = None
    dormant_candidate_replay: DormantTerminalContinuation | None = None
    causal_reduction_probe: CausalReductionProbe | None = None
    structural_transfer_probe: StructuralTransferProbe | None = None
    pending: TerminalFrontierSelection | None = None


@dataclass
class _DormantLineage:
    frontier_id: str
    actions: list[TerminalFrontierAction]
    state_signatures: list[str]


@dataclass
class _ActiveReacquisition:
    frontier_id: str
    path: FrontierAcquisitionPath
    actions: list[TerminalFrontierAction]
    pending: TerminalFrontierAction | None = None


@dataclass(frozen=True)
class TerminalFrontierReacquisitionSelection:
    """One exact prefix action used to return to a confirmed frontier."""

    frontier_id: str
    action: TerminalFrontierAction
    step_index: int
    action_limit: int
    reason: str


class OnlineTerminalFrontierExplorer:
    """Explore and terminally credit bounded suffixes from negative frontiers."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_frontiers: int = 24,
        max_suffix_actions: int = 6,
        max_trials_per_frontier: int = 4,
        max_candidates_per_state: int = 12,
        enable_adaptive_horizon: bool = True,
        max_adaptive_suffix_actions: int = 24,
        adaptive_horizon_increment: int = 6,
        enable_dormant_terminal_lineage: bool = True,
        max_dormant_lineage_actions: int = 80,
        max_dormant_candidates_per_frontier: int = 4,
        max_dormant_candidate_replays: int = 1,
        enable_structural_terminal_attribution: bool = True,
        enable_terminal_causal_reduction: bool = True,
        max_causal_reduction_probes_per_frontier: int = 3,
        max_causal_reduction_replays: int = 1,
        enable_active_frontier_reacquisition: bool = True,
        max_frontier_acquisition_paths: int = 4,
        max_frontier_reacquisition_actions: int = 80,
        max_frontier_reacquisition_attempts: int = 2,
        enable_recursive_terminal_causal_minimization: bool = True,
        max_causal_reduction_generations: int = 3,
        max_causal_reduction_probes_total: int = 12,
        enable_structural_frontier_transfer: bool = True,
        max_structural_transfer_probes_per_frontier: int = 2,
        max_structural_transfer_attempts: int = 1,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_frontiers = max(1, int(max_frontiers))
        self.max_suffix_actions = max(1, int(max_suffix_actions))
        self.max_trials_per_frontier = max(
            1,
            int(max_trials_per_frontier),
        )
        self.max_candidates_per_state = max(
            1,
            int(max_candidates_per_state),
        )
        self.enable_adaptive_horizon = bool(enable_adaptive_horizon)
        self.max_adaptive_suffix_actions = max(
            self.max_suffix_actions,
            int(max_adaptive_suffix_actions),
        )
        self.adaptive_horizon_increment = max(
            1,
            int(adaptive_horizon_increment),
        )
        self.enable_dormant_terminal_lineage = bool(
            enable_dormant_terminal_lineage
        )
        self.max_dormant_lineage_actions = max(
            self.max_adaptive_suffix_actions,
            int(max_dormant_lineage_actions),
        )
        self.max_dormant_candidates_per_frontier = max(
            1,
            int(max_dormant_candidates_per_frontier),
        )
        self.max_dormant_candidate_replays = max(
            1,
            int(max_dormant_candidate_replays),
        )
        self.enable_structural_terminal_attribution = bool(
            enable_structural_terminal_attribution
        )
        self.enable_terminal_causal_reduction = bool(
            enable_terminal_causal_reduction
        )
        self.max_causal_reduction_probes_per_frontier = max(
            1,
            int(max_causal_reduction_probes_per_frontier),
        )
        self.max_causal_reduction_replays = max(
            1,
            int(max_causal_reduction_replays),
        )
        self.enable_active_frontier_reacquisition = bool(
            enable_active_frontier_reacquisition
        )
        self.max_frontier_acquisition_paths = max(
            1,
            int(max_frontier_acquisition_paths),
        )
        self.max_frontier_reacquisition_actions = max(
            1,
            int(max_frontier_reacquisition_actions),
        )
        self.max_frontier_reacquisition_attempts = max(
            1,
            int(max_frontier_reacquisition_attempts),
        )
        self.enable_recursive_terminal_causal_minimization = bool(
            enable_recursive_terminal_causal_minimization
        )
        self.max_causal_reduction_generations = max(
            1,
            int(max_causal_reduction_generations),
        )
        self.max_causal_reduction_probes_total = max(
            self.max_causal_reduction_probes_per_frontier,
            int(max_causal_reduction_probes_total),
        )
        self.enable_structural_frontier_transfer = bool(
            enable_structural_frontier_transfer
        )
        self.max_structural_transfer_probes_per_frontier = max(
            1,
            int(max_structural_transfer_probes_per_frontier),
        )
        self.max_structural_transfer_attempts = max(
            1,
            int(max_structural_transfer_attempts),
        )
        self._frontiers: Dict[str, TerminalNegativeFrontier] = {}
        self._actions_by_state: Dict[
            str,
            Dict[str, TerminalFrontierAction],
        ] = {}
        self._choice_counts: Counter[Tuple[str, Tuple[str, ...], str]] = (
            Counter()
        )
        self._active: _ActiveSuffix | None = None
        self._dormant: _DormantLineage | None = None
        self._active_reacquisition: _ActiveReacquisition | None = None
        self._reacquired_frontier_id = ""
        self._branch_actions: list[TerminalFrontierAction] = []
        self._branch_state_signatures: list[str] = []
        self._reacquisition_start_checked = False
        self._frontiers_captured = 0
        self._duplicate_captures = 0
        self._trials_started = 0
        self._suffix_actions = 0
        self._terminal_credits = 0
        self._level_change_credits = 0
        self._win_credits = 0
        self._successful_replays = 0
        self._replay_divergences = 0
        self._capacity_blocks = 0
        self._branch_trial_blocks = 0
        self._adaptive_horizon_extensions = 0
        self._adaptive_horizon_actions_granted = 0
        self._extended_suffix_actions = 0
        self._dormant_lineages_started = 0
        self._dormant_lineage_actions = 0
        self._dormant_lineage_terminal_candidates = 0
        self._dormant_lineage_level_candidates = 0
        self._dormant_lineage_win_candidates = 0
        self._dormant_lineage_censored = 0
        self._dormant_lineage_expired = 0
        self._dormant_lineage_unsafe = 0
        self._dormant_candidate_capacity_blocks = 0
        self._dormant_candidate_replay_attempts = 0
        self._dormant_candidate_replay_actions = 0
        self._dormant_candidate_confirmations = 0
        self._dormant_candidate_refutations = 0
        self._dormant_candidate_divergences = 0
        self._structural_frontiers_captured = 0
        self._structural_terminal_candidates = 0
        self._structural_candidate_confirmations = 0
        self._structural_terminal_credits = 0
        self._structural_attribution_blocks = 0
        self._causal_reduction_probes_compiled = 0
        self._causal_reduction_attempts = 0
        self._causal_reduction_actions = 0
        self._causal_reduction_confirmations = 0
        self._causal_reduction_refutations = 0
        self._causal_reduction_divergences = 0
        self._causal_reduction_terminal_credits = 0
        self._acquisition_paths_recorded = 0
        self._frontier_reacquisition_attempts = 0
        self._frontier_reacquisition_actions = 0
        self._frontier_reacquisition_confirmations = 0
        self._frontier_reacquisition_divergences = 0
        self._frontier_reacquisition_censored = 0
        self._recursive_reduction_probes_compiled = 0
        self._maximum_reduction_generation = 0
        self._structural_transfer_probes_compiled = 0
        self._structural_transfer_attempts = 0
        self._structural_transfer_actions = 0
        self._structural_transfer_confirmations = 0
        self._structural_transfer_refutations = 0
        self._structural_transfer_divergences = 0
        self._structural_transfer_terminal_credits = 0
        self._trial_started_this_branch = False

    @property
    def active_frontier_id(self) -> str:
        return "" if self._active is None else self._active.frontier_id

    @property
    def active_suffix_started(self) -> bool:
        """Whether at least one action of the active suffix was selected."""
        return bool(
            self._active is not None
            and (self._active.actions or self._active.pending is not None)
        )

    @property
    def active_replay_available(self) -> bool:
        """Whether the active trial has a confirmed or candidate replay."""
        return bool(
            self._active is not None
            and (
                self._active.replay is not None
                or self._active.dormant_candidate_replay is not None
                or self._active.causal_reduction_probe is not None
                or self._active.structural_transfer_probe is not None
            )
        )

    @property
    def active_reacquisition_available(self) -> bool:
        """Whether an exact reset-to-frontier path is active or can start."""
        if not self.enabled or not self.enable_active_frontier_reacquisition:
            return False
        if self._active_reacquisition is not None:
            return True
        if self._reacquisition_start_checked or self._branch_actions:
            return False
        return any(
            frontier.frontier_kind == "structural_change"
            and bool(frontier.successful_continuations)
            and any(
                path.attempts < self.max_frontier_reacquisition_attempts
                for path in frontier.acquisition_paths.values()
            )
            for frontier in self._frontiers.values()
        )

    def remember_action(
        self,
        state_signature: str,
        action_name: str,
        action_data: Mapping[str, Any] | None = None,
    ) -> None:
        """Retain a concrete action only after it was legal and executed."""
        state = str(state_signature)
        action = TerminalFrontierAction.from_parts(action_name, action_data)
        if not state or not action.action_name:
            return
        candidates = self._actions_by_state.setdefault(state, {})
        if action.signature in candidates:
            return
        if len(candidates) >= self.max_candidates_per_state:
            return
        candidates[action.signature] = action

    def capture(
        self,
        *,
        state_signature: str,
        objective_ids: Iterable[str],
        context_signature: str = "",
        frontier_kind: str = "objective_postcondition",
        structural_trigger_signature: str = "",
        structural_trigger_families: Iterable[str] = (),
        structural_equivalence_signature: str = "",
    ) -> str:
        """Capture a nonterminal postcondition and start one bounded trial."""
        if not self.enabled:
            return ""
        objectives = tuple(sorted({str(item) for item in objective_ids if item}))
        state = str(state_signature)
        if not state or not objectives:
            return ""
        kind = str(frontier_kind or "objective_postcondition")
        frontier_id = _frontier_id(state, objectives, kind)
        frontier = self._frontiers.get(frontier_id)
        if self._active is not None or self._trial_started_this_branch:
            self._branch_trial_blocks += 1
            if frontier is None:
                return ""
            self._duplicate_captures += 1
            frontier.captures += 1
            if context_signature:
                frontier.context_signatures.add(str(context_signature))
            if structural_trigger_signature:
                frontier.structural_trigger_signatures.add(
                    str(structural_trigger_signature)
                )
            frontier.structural_trigger_families.update(
                str(item) for item in structural_trigger_families if item
            )
            if structural_equivalence_signature:
                frontier.structural_equivalence_signature = str(
                    structural_equivalence_signature
                )
            self._record_frontier_acquisition_path(frontier)
            self._compile_structural_transfer_probes(frontier)
            return frontier_id
        if frontier is None:
            if len(self._frontiers) >= self.max_frontiers:
                self._capacity_blocks += 1
                return ""
            frontier = TerminalNegativeFrontier(
                frontier_id=frontier_id,
                state_signature=state,
                objective_ids=objectives,
                frontier_kind=kind,
                structural_equivalence_signature=str(
                    structural_equivalence_signature
                ),
                allocated_action_limit=self.max_suffix_actions,
                horizon_history=[self.max_suffix_actions],
            )
            self._frontiers[frontier_id] = frontier
            self._frontiers_captured += 1
            if kind == "structural_change":
                self._structural_frontiers_captured += 1
        else:
            self._duplicate_captures += 1
        frontier.captures += 1
        if context_signature:
            frontier.context_signatures.add(str(context_signature))
        if structural_trigger_signature:
            frontier.structural_trigger_signatures.add(
                str(structural_trigger_signature)
            )
        frontier.structural_trigger_families.update(
            str(item) for item in structural_trigger_families if item
        )
        if structural_equivalence_signature:
            frontier.structural_equivalence_signature = str(
                structural_equivalence_signature
            )
        self._record_frontier_acquisition_path(frontier)
        self._compile_structural_transfer_probes(frontier)
        replay = self._best_successful_continuation(frontier)
        causal_reduction_probe = (
            self._best_causal_reduction_probe(frontier)
            if replay is not None
            and frontier.frontier_kind == "structural_change"
            and self.enable_terminal_causal_reduction
            else None
        )
        if causal_reduction_probe is not None:
            replay = None
        dormant_candidate = (
            None
            if replay is not None or causal_reduction_probe is not None
            else self._best_dormant_terminal_candidate(frontier)
        )
        structural_transfer_probe = (
            self._best_structural_transfer_probe(frontier)
            if replay is None
            and causal_reduction_probe is None
            and dormant_candidate is None
            else None
        )
        if (
            replay is None
            and dormant_candidate is None
            and causal_reduction_probe is None
            and structural_transfer_probe is None
            and frontier.trials >= self.max_trials_per_frontier
        ):
            return frontier_id
        if replay is not None and replay.confirmations >= 2:
            return frontier_id
        if replay is None and dormant_candidate is None:
            if (
                causal_reduction_probe is None
                and structural_transfer_probe is None
            ):
                self._extend_exhausted_frontier_horizon(frontier)
        if dormant_candidate is not None:
            dormant_candidate.replay_attempts += 1
            self._dormant_candidate_replay_attempts += 1
        if causal_reduction_probe is not None:
            causal_reduction_probe.replay_attempts += 1
            self._causal_reduction_attempts += 1
        if structural_transfer_probe is not None:
            structural_transfer_probe.attempts += 1
            self._structural_transfer_attempts += 1
        frontier.trials += 1
        self._trials_started += 1
        self._trial_started_this_branch = True
        self._active = _ActiveSuffix(
            frontier_id=frontier_id,
            actions=[],
            state_signatures=[state],
            action_limit=(
                len(dormant_candidate.actions)
                if dormant_candidate is not None
                else (
                    len(causal_reduction_probe.actions)
                    if causal_reduction_probe is not None
                    else (
                        len(structural_transfer_probe.actions)
                        if structural_transfer_probe is not None
                        else (
                            len(replay.actions)
                            if replay is not None
                            else frontier.allocated_action_limit
                        )
                    )
                )
            ),
            replay=replay,
            dormant_candidate_replay=dormant_candidate,
            causal_reduction_probe=causal_reduction_probe,
            structural_transfer_probe=structural_transfer_probe,
        )
        return frontier_id

    def capture_structural(
        self,
        *,
        state_signature: str,
        trigger_signature: str,
        trigger_families: Iterable[str],
        equivalence_signature: str = "",
        context_signature: str = "",
    ) -> str:
        """Capture a generic structural boundary without objective credit."""
        return self.capture(
            state_signature=state_signature,
            objective_ids=("structural::generic",),
            context_signature=context_signature,
            frontier_kind="structural_change",
            structural_trigger_signature=trigger_signature,
            structural_trigger_families=trigger_families,
            structural_equivalence_signature=equivalence_signature,
        )

    def select_reacquisition(
        self,
        *,
        state_signature: str,
        available_actions: Sequence[str],
    ) -> TerminalFrontierReacquisitionSelection | None:
        """Replay one exact reset-to-frontier action for SAGE.9l."""
        if (
            not self.enabled
            or not self.enable_active_frontier_reacquisition
            or self._active is not None
        ):
            return None
        state = str(state_signature)
        allowed = {str(action).upper() for action in available_actions}
        active = self._active_reacquisition
        if active is None:
            if self._reacquisition_start_checked or self._branch_actions:
                return None
            self._reacquisition_start_checked = True
            candidates: list[
                tuple[
                    TerminalNegativeFrontier,
                    FrontierAcquisitionPath,
                ]
            ] = []
            for frontier in self._frontiers.values():
                if (
                    frontier.frontier_kind != "structural_change"
                    or not frontier.successful_continuations
                ):
                    continue
                for path in frontier.acquisition_paths.values():
                    if (
                        path.actions
                        and path.state_signatures
                        and path.state_signatures[0] == state
                        and path.attempts
                        < self.max_frontier_reacquisition_attempts
                    ):
                        candidates.append((frontier, path))
            if not candidates:
                return None
            frontier, path = min(
                candidates,
                key=lambda item: (
                    item[1].attempts,
                    len(item[1].actions),
                    item[0].frontier_id,
                    item[1].signature,
                ),
            )
            path.attempts += 1
            self._frontier_reacquisition_attempts += 1
            active = _ActiveReacquisition(
                frontier_id=frontier.frontier_id,
                path=path,
                actions=[],
            )
            self._active_reacquisition = active

        if active.pending is not None:
            return None
        step = len(active.actions)
        path = active.path
        if (
            step >= len(path.actions)
            or step >= len(path.state_signatures)
            or state != path.state_signatures[step]
        ):
            path.divergences += 1
            self._frontier_reacquisition_divergences += 1
            self._active_reacquisition = None
            return None
        action = path.actions[step]
        if action.action_name not in allowed:
            path.divergences += 1
            self._frontier_reacquisition_divergences += 1
            self._active_reacquisition = None
            return None
        active.pending = action
        return TerminalFrontierReacquisitionSelection(
            frontier_id=active.frontier_id,
            action=action,
            step_index=step,
            action_limit=len(path.actions),
            reason=(
                "exact online reset-to-frontier reacquisition for a "
                "terminal-confirmed structural continuation"
            ),
        )

    def activate_reacquired_frontier(
        self,
        *,
        state_signature: str,
    ) -> str:
        """Open the existing frontier after an exact SAGE.9l reacquisition."""
        frontier_id = self._reacquired_frontier_id
        self._reacquired_frontier_id = ""
        frontier = self._frontiers.get(frontier_id)
        if (
            frontier is None
            or frontier.state_signature != str(state_signature)
        ):
            return ""
        return self.capture(
            state_signature=frontier.state_signature,
            objective_ids=frontier.objective_ids,
            frontier_kind=frontier.frontier_kind,
            structural_trigger_families=(
                frontier.structural_trigger_families
            ),
            structural_equivalence_signature=(
                frontier.structural_equivalence_signature
            ),
        )

    def select(
        self,
        *,
        state_signature: str,
        available_actions: Sequence[str],
        proposed_actions: Sequence[TerminalFrontierAction] = (),
        restrict_to_proposed: bool = False,
    ) -> TerminalFrontierSelection | None:
        """Select a replay action or the least-tested concrete continuation."""
        active = self._active
        if not self.enabled or active is None or active.pending is not None:
            return None
        frontier = self._frontiers[active.frontier_id]
        allowed = {str(action).upper() for action in available_actions}
        state = str(state_signature)

        replay_sequence = (
            active.replay
            if active.replay is not None
            else active.dormant_candidate_replay
        )
        if replay_sequence is not None:
            step = len(active.actions)
            expected_states = replay_sequence.state_signatures
            if step < len(expected_states) and state != expected_states[step]:
                self._replay_divergences += 1
                if active.dormant_candidate_replay is not None:
                    active.dormant_candidate_replay.divergences += 1
                    self._dormant_candidate_divergences += 1
                self._active = None
                return None
            elif step < len(replay_sequence.actions):
                action = replay_sequence.actions[step]
                if action.action_name in allowed:
                    return self._record_selection(
                        active,
                        frontier,
                        action,
                        replaying=active.replay is not None,
                        replaying_dormant_candidate=(
                            active.dormant_candidate_replay is not None
                        ),
                        testing_causal_reduction=False,
                        testing_structural_transfer=False,
                    )
                self._replay_divergences += 1
                if active.dormant_candidate_replay is not None:
                    active.dormant_candidate_replay.divergences += 1
                    self._dormant_candidate_divergences += 1
                self._active = None
                return None
        causal_probe = active.causal_reduction_probe
        if causal_probe is not None:
            step = len(active.actions)
            if step < len(causal_probe.actions):
                action = causal_probe.actions[step]
                if action.action_name in allowed:
                    return self._record_selection(
                        active,
                        frontier,
                        action,
                        replaying=False,
                        replaying_dormant_candidate=False,
                        testing_causal_reduction=True,
                        testing_structural_transfer=False,
                    )
                causal_probe.divergences += 1
                self._causal_reduction_divergences += 1
                self._active = None
                return None
        transfer_probe = active.structural_transfer_probe
        if transfer_probe is not None:
            step = len(active.actions)
            if step < len(transfer_probe.actions):
                action = transfer_probe.actions[step]
                if action.action_name in allowed:
                    return self._record_selection(
                        active,
                        frontier,
                        action,
                        replaying=False,
                        replaying_dormant_candidate=False,
                        testing_causal_reduction=False,
                        testing_structural_transfer=True,
                    )
                transfer_probe.divergences += 1
                self._structural_transfer_divergences += 1
                self._active = None
                return None

        candidates: Dict[str, TerminalFrontierAction] = {}
        for action in proposed_actions:
            if action.action_name in allowed:
                candidates.setdefault(action.signature, action)
        if not restrict_to_proposed:
            for action in self._actions_by_state.get(state, {}).values():
                if action.action_name in allowed:
                    candidates.setdefault(action.signature, action)
            for action_name in sorted(allowed):
                if action_name == "ACTION6":
                    continue
                action = TerminalFrontierAction.from_parts(action_name)
                candidates.setdefault(action.signature, action)
        if not candidates:
            return None
        prefix = tuple(action.signature for action in active.actions)
        action = min(
            candidates.values(),
            key=lambda candidate: (
                self._choice_counts[
                    (frontier.frontier_id, prefix, candidate.signature)
                ],
                candidate.signature,
            ),
        )
        return self._record_selection(
            active,
            frontier,
            action,
            replaying=False,
            replaying_dormant_candidate=False,
            testing_causal_reduction=False,
            testing_structural_transfer=False,
        )

    def observe_transition(
        self,
        *,
        state_signature_before: str,
        state_signature_after: str,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        level_progressed: bool,
        won: bool,
        game_over: bool,
    ) -> Dict[str, Any]:
        """Revise a suffix using terminal outcomes and nothing weaker."""
        self.remember_action(
            state_signature_before,
            action_name,
            action_data,
        )
        observed = TerminalFrontierAction.from_parts(action_name, action_data)
        self._record_branch_transition(
            state_signature_before=str(state_signature_before),
            state_signature_after=str(state_signature_after),
            observed=observed,
        )
        reacquisition_outcome = self._observe_reacquisition_transition(
            state_signature_after=str(state_signature_after),
            observed=observed,
            level_progressed=bool(level_progressed),
            won=bool(won),
            game_over=bool(game_over),
        )
        if reacquisition_outcome is not None:
            return reacquisition_outcome
        active = self._active
        if active is None or active.pending is None:
            return self._observe_dormant_lineage(
                state_signature_after=str(state_signature_after),
                observed=observed,
                level_progressed=bool(level_progressed),
                won=bool(won),
                game_over=bool(game_over),
            )
        selection = active.pending
        dormant_candidate = active.dormant_candidate_replay
        causal_probe = active.causal_reduction_probe
        transfer_probe = active.structural_transfer_probe
        replaying = bool(
            selection.replaying_successful_continuation
            and observed.signature == selection.action.signature
        )
        replaying_dormant_candidate = bool(
            selection.replaying_dormant_terminal_candidate
            and observed.signature == selection.action.signature
        )
        if selection.replaying_successful_continuation and not replaying:
            self._replay_divergences += 1
            active.replay = None
        if (
            selection.replaying_dormant_terminal_candidate
            and not replaying_dormant_candidate
        ):
            self._replay_divergences += 1
            self._dormant_candidate_divergences += 1
            if dormant_candidate is not None:
                dormant_candidate.divergences += 1
            active.dormant_candidate_replay = None
        testing_causal_reduction = bool(
            selection.testing_causal_reduction
            and causal_probe is not None
            and observed.signature == selection.action.signature
        )
        if selection.testing_causal_reduction and not testing_causal_reduction:
            if causal_probe is not None:
                causal_probe.divergences += 1
            self._causal_reduction_divergences += 1
            active.causal_reduction_probe = None
        testing_structural_transfer = bool(
            selection.testing_structural_transfer
            and transfer_probe is not None
            and observed.signature == selection.action.signature
        )
        if (
            selection.testing_structural_transfer
            and not testing_structural_transfer
        ):
            if transfer_probe is not None:
                transfer_probe.divergences += 1
            self._structural_transfer_divergences += 1
            active.structural_transfer_probe = None
        active.pending = None
        active.actions.append(observed)
        active.state_signatures.append(str(state_signature_after))
        self._suffix_actions += 1
        if selection.replaying_dormant_terminal_candidate:
            self._dormant_candidate_replay_actions += 1
        if selection.testing_causal_reduction:
            self._causal_reduction_actions += 1
        if selection.testing_structural_transfer:
            self._structural_transfer_actions += 1
        if (
            len(active.actions) > self.max_suffix_actions
            and active.replay is None
            and dormant_candidate is None
            and causal_probe is None
            and transfer_probe is None
        ):
            self._extended_suffix_actions += 1
        terminal_success = bool(level_progressed or won)
        frontier = self._frontiers[active.frontier_id]
        frontier.longest_suffix_actions = max(
            frontier.longest_suffix_actions,
            len(active.actions),
        )
        outcome = {
            "frontier_id": frontier.frontier_id,
            "objective_ids": list(frontier.objective_ids),
            "suffix_step": len(active.actions) - 1,
            "action_limit": active.action_limit,
            "adaptive_horizon": bool(
                active.action_limit > self.max_suffix_actions
                and not selection.replaying_successful_continuation
                and not selection.replaying_dormant_terminal_candidate
                and not selection.testing_causal_reduction
                and not selection.testing_structural_transfer
            ),
            "terminal_success": terminal_success,
            "level_progressed": bool(level_progressed),
            "won": bool(won),
            "game_over": bool(game_over),
            "credited": False,
            "replaying_successful_continuation": replaying,
            "replaying_dormant_terminal_candidate": (
                replaying_dormant_candidate
            ),
            "testing_causal_reduction": testing_causal_reduction,
            "testing_structural_transfer": testing_structural_transfer,
            "dormant_terminal_candidate_nominated": False,
            "dormant_terminal_candidate_confirmed": False,
            "causal_reduction_confirmed": False,
            "causal_reduction_refuted": False,
            "structural_transfer_confirmed": False,
            "structural_transfer_refuted": False,
        }
        if terminal_success:
            actions = tuple(active.actions)
            states = tuple(active.state_signatures)
            if selection.testing_causal_reduction:
                if testing_causal_reduction and causal_probe is not None:
                    causal_probe.confirmations += 1
                    self._causal_reduction_confirmations += 1
                    self._credit_continuation(
                        frontier,
                        actions,
                        states,
                        level_progressed=bool(level_progressed),
                        won=bool(won),
                        replaying=True,
                        causal_reduction=True,
                        removed_action_indices=(
                            causal_probe.removed_action_indices
                        ),
                    )
                    self._causal_reduction_terminal_credits += 1
                    outcome["credited"] = True
                    outcome["causal_reduction_confirmed"] = True
                    if (
                        self.enable_recursive_terminal_causal_minimization
                        and causal_probe.generation
                        < self.max_causal_reduction_generations
                    ):
                        self._compile_causal_reduction_probes(
                            frontier,
                            actions,
                            generation=causal_probe.generation + 1,
                            parent_probe_id=causal_probe.probe_id,
                            origin_indices=(
                                causal_probe.kept_origin_indices
                            ),
                            previously_removed=(
                                causal_probe.removed_action_indices
                            ),
                        )
            elif selection.testing_structural_transfer:
                if (
                    testing_structural_transfer
                    and transfer_probe is not None
                ):
                    transfer_probe.confirmations += 1
                    self._structural_transfer_confirmations += 1
                    self._credit_continuation(
                        frontier,
                        actions,
                        states,
                        level_progressed=bool(level_progressed),
                        won=bool(won),
                        replaying=True,
                        structural_transfer=True,
                        source_frontier_id=(
                            transfer_probe.source_frontier_id
                        ),
                    )
                    self._structural_transfer_terminal_credits += 1
                    outcome["credited"] = True
                    outcome["structural_transfer_confirmed"] = True
            elif selection.replaying_dormant_terminal_candidate:
                exact_terminal_replay = bool(
                    replaying_dormant_candidate
                    and dormant_candidate is not None
                    and len(actions) == len(dormant_candidate.actions)
                )
                if exact_terminal_replay:
                    if dormant_candidate is not None:
                        dormant_candidate.confirmations += 1
                    self._dormant_candidate_confirmations += 1
                    self._credit_continuation(
                        frontier,
                        actions,
                        states,
                        level_progressed=bool(level_progressed),
                        won=bool(won),
                        replaying=True,
                    )
                    if frontier.frontier_kind == "structural_change":
                        self._structural_candidate_confirmations += 1
                        self._compile_causal_reduction_probes(
                            frontier,
                            actions,
                        )
                        self._compile_structural_transfer_probes_for_source(
                            frontier
                        )
                    outcome["credited"] = True
                    outcome["dormant_terminal_candidate_confirmed"] = True
                else:
                    if replaying_dormant_candidate:
                        if dormant_candidate is not None:
                            dormant_candidate.divergences += 1
                        self._dormant_candidate_divergences += 1
                    candidate = self._record_dormant_terminal_candidate(
                        frontier,
                        actions,
                        states,
                        level_progressed=bool(level_progressed),
                        won=bool(won),
                    )
                    outcome["dormant_terminal_candidate_nominated"] = bool(
                        candidate is not None
                    )
            elif selection.replaying_successful_continuation and replaying:
                self._credit_continuation(
                    frontier,
                    actions,
                    states,
                    level_progressed=bool(level_progressed),
                    won=bool(won),
                    replaying=True,
                )
                if frontier.frontier_kind == "structural_change":
                    self._compile_causal_reduction_probes(
                        frontier,
                        actions,
                    )
                    self._compile_structural_transfer_probes_for_source(
                        frontier
                    )
                outcome["credited"] = True
            elif frontier.frontier_kind == "structural_change":
                if self.enable_structural_terminal_attribution:
                    candidate = self._record_dormant_terminal_candidate(
                        frontier,
                        actions,
                        states,
                        level_progressed=bool(level_progressed),
                        won=bool(won),
                    )
                    outcome["dormant_terminal_candidate_nominated"] = bool(
                        candidate is not None
                    )
                else:
                    self._structural_attribution_blocks += 1
            else:
                self._credit_continuation(
                    frontier,
                    actions,
                    states,
                    level_progressed=bool(level_progressed),
                    won=bool(won),
                    replaying=replaying,
                )
                outcome["credited"] = True
            self._active = None
        elif selection.testing_structural_transfer:
            if testing_structural_transfer and (
                game_over or len(active.actions) >= active.action_limit
            ):
                if transfer_probe is not None:
                    transfer_probe.refutations += 1
                self._structural_transfer_refutations += 1
                outcome["structural_transfer_refuted"] = True
            if game_over:
                frontier.unsafe_suffixes += 1
            if (
                not testing_structural_transfer
                or game_over
                or len(active.actions) >= active.action_limit
            ):
                self._active = None
        elif selection.testing_causal_reduction:
            if testing_causal_reduction and (
                game_over or len(active.actions) >= active.action_limit
            ):
                if causal_probe is not None:
                    causal_probe.refutations += 1
                self._causal_reduction_refutations += 1
                outcome["causal_reduction_refuted"] = True
            if game_over:
                frontier.unsafe_suffixes += 1
            if (
                not testing_causal_reduction
                or game_over
                or len(active.actions) >= active.action_limit
            ):
                self._active = None
        elif selection.replaying_dormant_terminal_candidate:
            if replaying_dormant_candidate and (
                game_over or len(active.actions) >= active.action_limit
            ):
                if dormant_candidate is not None:
                    dormant_candidate.refutations += 1
                self._dormant_candidate_refutations += 1
            if game_over:
                frontier.unsafe_suffixes += 1
            if (
                not replaying_dormant_candidate
                or game_over
                or len(active.actions) >= active.action_limit
            ):
                self._active = None
        elif game_over:
            frontier.unsafe_suffixes += 1
            self._active = None
        elif len(active.actions) >= active.action_limit:
            frontier.nonterminal_suffixes += 1
            if active.replay is None:
                self._start_dormant_lineage(active)
            self._active = None
        return outcome

    def start_branch(self) -> None:
        """Censor an unfinished suffix without inventing negative credit."""
        if self._active is not None:
            self._frontiers[self._active.frontier_id].censored_suffixes += 1
        if self._dormant is not None:
            self._dormant_lineage_censored += 1
        if self._active_reacquisition is not None:
            self._active_reacquisition.path.censored += 1
            self._frontier_reacquisition_censored += 1
        self._active = None
        self._dormant = None
        self._active_reacquisition = None
        self._reacquired_frontier_id = ""
        self._branch_actions = []
        self._branch_state_signatures = []
        self._reacquisition_start_checked = False
        self._trial_started_this_branch = False

    def frontiers(self) -> Tuple[TerminalNegativeFrontier, ...]:
        return tuple(
            self._frontiers[key]
            for key in sorted(self._frontiers)
        )

    def summary(self) -> Dict[str, Any]:
        """Return auditable attribution counters for SAGE.9f-SAGE.9n."""
        successful = sum(
            len(frontier.successful_continuations)
            for frontier in self._frontiers.values()
        )
        dormant_candidates = [
            candidate
            for frontier in self._frontiers.values()
            for candidate in frontier.dormant_terminal_candidates.values()
        ]
        reduction_probes = [
            probe
            for frontier in self._frontiers.values()
            for probe in frontier.causal_reduction_probes.values()
        ]
        acquisition_paths = [
            path
            for frontier in self._frontiers.values()
            for path in frontier.acquisition_paths.values()
        ]
        transfer_probes = [
            probe
            for frontier in self._frontiers.values()
            for probe in frontier.structural_transfer_probes.values()
        ]
        return {
            "enabled": self.enabled,
            "max_frontiers": self.max_frontiers,
            "max_suffix_actions": self.max_suffix_actions,
            "max_trials_per_frontier": self.max_trials_per_frontier,
            "adaptive_horizon_enabled": self.enable_adaptive_horizon,
            "max_adaptive_suffix_actions": self.max_adaptive_suffix_actions,
            "adaptive_horizon_increment": self.adaptive_horizon_increment,
            "dormant_terminal_lineage_enabled": (
                self.enable_dormant_terminal_lineage
            ),
            "max_dormant_lineage_actions": self.max_dormant_lineage_actions,
            "max_dormant_candidates_per_frontier": (
                self.max_dormant_candidates_per_frontier
            ),
            "max_dormant_candidate_replays": (
                self.max_dormant_candidate_replays
            ),
            "structural_terminal_attribution_enabled": (
                self.enable_structural_terminal_attribution
            ),
            "terminal_causal_reduction_enabled": (
                self.enable_terminal_causal_reduction
            ),
            "max_causal_reduction_probes_per_frontier": (
                self.max_causal_reduction_probes_per_frontier
            ),
            "max_causal_reduction_replays": (
                self.max_causal_reduction_replays
            ),
            "active_frontier_reacquisition_enabled": (
                self.enable_active_frontier_reacquisition
            ),
            "max_frontier_reacquisition_actions": (
                self.max_frontier_reacquisition_actions
            ),
            "max_frontier_reacquisition_attempts": (
                self.max_frontier_reacquisition_attempts
            ),
            "recursive_terminal_causal_minimization_enabled": (
                self.enable_recursive_terminal_causal_minimization
            ),
            "max_causal_reduction_generations": (
                self.max_causal_reduction_generations
            ),
            "structural_frontier_transfer_enabled": (
                self.enable_structural_frontier_transfer
            ),
            "frontiers": len(self._frontiers),
            "frontiers_captured": self._frontiers_captured,
            "duplicate_captures": self._duplicate_captures,
            "capacity_blocks": self._capacity_blocks,
            "branch_trial_blocks": self._branch_trial_blocks,
            "adaptive_horizon_extensions": self._adaptive_horizon_extensions,
            "adaptive_horizon_actions_granted": (
                self._adaptive_horizon_actions_granted
            ),
            "extended_suffix_actions": self._extended_suffix_actions,
            "frontiers_with_extended_horizon": sum(
                int(frontier.horizon_extensions > 0)
                for frontier in self._frontiers.values()
            ),
            "maximum_allocated_horizon": max(
                (
                    frontier.allocated_action_limit
                    for frontier in self._frontiers.values()
                ),
                default=self.max_suffix_actions,
            ),
            "dormant_lineages_started": self._dormant_lineages_started,
            "dormant_lineage_actions": self._dormant_lineage_actions,
            "dormant_lineage_terminal_candidates": (
                self._dormant_lineage_terminal_candidates
            ),
            "dormant_lineage_level_candidates": (
                self._dormant_lineage_level_candidates
            ),
            "dormant_lineage_win_candidates": (
                self._dormant_lineage_win_candidates
            ),
            "dormant_lineage_censored": self._dormant_lineage_censored,
            "dormant_lineage_expired": self._dormant_lineage_expired,
            "dormant_lineage_unsafe": self._dormant_lineage_unsafe,
            "dormant_terminal_candidates": len(dormant_candidates),
            "dormant_candidate_capacity_blocks": (
                self._dormant_candidate_capacity_blocks
            ),
            "dormant_candidate_replay_attempts": (
                self._dormant_candidate_replay_attempts
            ),
            "dormant_candidate_replay_actions": (
                self._dormant_candidate_replay_actions
            ),
            "dormant_candidate_confirmations": (
                self._dormant_candidate_confirmations
            ),
            "dormant_candidate_refutations": (
                self._dormant_candidate_refutations
            ),
            "dormant_candidate_divergences": (
                self._dormant_candidate_divergences
            ),
            "structural_frontiers_captured": (
                self._structural_frontiers_captured
            ),
            "structural_terminal_candidates": (
                self._structural_terminal_candidates
            ),
            "structural_candidate_confirmations": (
                self._structural_candidate_confirmations
            ),
            "structural_terminal_credits": (
                self._structural_terminal_credits
            ),
            "structural_attribution_blocks": (
                self._structural_attribution_blocks
            ),
            "causal_reduction_probes": len(reduction_probes),
            "causal_reduction_probes_compiled": (
                self._causal_reduction_probes_compiled
            ),
            "causal_reduction_attempts": self._causal_reduction_attempts,
            "causal_reduction_actions": self._causal_reduction_actions,
            "causal_reduction_confirmations": (
                self._causal_reduction_confirmations
            ),
            "causal_reduction_refutations": (
                self._causal_reduction_refutations
            ),
            "causal_reduction_divergences": (
                self._causal_reduction_divergences
            ),
            "causal_reduction_terminal_credits": (
                self._causal_reduction_terminal_credits
            ),
            "recursive_reduction_probes_compiled": (
                self._recursive_reduction_probes_compiled
            ),
            "maximum_reduction_generation": (
                self._maximum_reduction_generation
            ),
            "minimum_confirmed_reduction_length": min(
                (
                    len(probe.actions)
                    for probe in reduction_probes
                    if probe.confirmations
                ),
                default=0,
            ),
            "frontier_acquisition_paths": len(acquisition_paths),
            "acquisition_paths_recorded": self._acquisition_paths_recorded,
            "frontier_reacquisition_attempts": (
                self._frontier_reacquisition_attempts
            ),
            "frontier_reacquisition_actions": (
                self._frontier_reacquisition_actions
            ),
            "frontier_reacquisition_confirmations": (
                self._frontier_reacquisition_confirmations
            ),
            "frontier_reacquisition_divergences": (
                self._frontier_reacquisition_divergences
            ),
            "frontier_reacquisition_censored": (
                self._frontier_reacquisition_censored
            ),
            "maximum_frontier_acquisition_path_length": max(
                (len(path.actions) for path in acquisition_paths),
                default=0,
            ),
            "structural_transfer_probes": len(transfer_probes),
            "structural_transfer_probes_compiled": (
                self._structural_transfer_probes_compiled
            ),
            "structural_transfer_attempts": (
                self._structural_transfer_attempts
            ),
            "structural_transfer_actions": self._structural_transfer_actions,
            "structural_transfer_confirmations": (
                self._structural_transfer_confirmations
            ),
            "structural_transfer_refutations": (
                self._structural_transfer_refutations
            ),
            "structural_transfer_divergences": (
                self._structural_transfer_divergences
            ),
            "structural_transfer_terminal_credits": (
                self._structural_transfer_terminal_credits
            ),
            "maximum_dormant_candidate_length": max(
                (len(candidate.actions) for candidate in dormant_candidates),
                default=0,
            ),
            "trials_started": self._trials_started,
            "suffix_actions": self._suffix_actions,
            "terminal_credits": self._terminal_credits,
            "level_change_credits": self._level_change_credits,
            "win_credits": self._win_credits,
            "successful_continuations": successful,
            "successful_replays": self._successful_replays,
            "replay_divergences": self._replay_divergences,
            "nonterminal_suffixes": sum(
                frontier.nonterminal_suffixes
                for frontier in self._frontiers.values()
            ),
            "unsafe_suffixes": sum(
                frontier.unsafe_suffixes
                for frontier in self._frontiers.values()
            ),
            "censored_suffixes": sum(
                frontier.censored_suffixes
                for frontier in self._frontiers.values()
            ),
            "active_frontier_id": self.active_frontier_id,
            "active_dormant_frontier_id": (
                "" if self._dormant is None else self._dormant.frontier_id
            ),
            "active_reacquisition_frontier_id": (
                ""
                if self._active_reacquisition is None
                else self._active_reacquisition.frontier_id
            ),
            "records": [frontier.to_dict() for frontier in self.frontiers()],
        }

    def _record_branch_transition(
        self,
        *,
        state_signature_before: str,
        state_signature_after: str,
        observed: TerminalFrontierAction,
    ) -> None:
        """Retain one bounded exact prefix for future online reacquisition."""
        before = str(state_signature_before)
        after = str(state_signature_after)
        if not self._branch_state_signatures:
            self._branch_state_signatures = [before]
        elif self._branch_state_signatures[-1] != before:
            self._branch_actions = []
            self._branch_state_signatures = [before]
        if len(self._branch_actions) >= self.max_frontier_reacquisition_actions:
            return
        self._branch_actions.append(observed)
        self._branch_state_signatures.append(after)

    def _observe_reacquisition_transition(
        self,
        *,
        state_signature_after: str,
        observed: TerminalFrontierAction,
        level_progressed: bool,
        won: bool,
        game_over: bool,
    ) -> Dict[str, Any] | None:
        active = self._active_reacquisition
        if active is None or active.pending is None:
            return None
        expected_action = active.pending
        active.pending = None
        path = active.path
        step = len(active.actions)
        expected_after = (
            path.state_signatures[step + 1]
            if step + 1 < len(path.state_signatures)
            else ""
        )
        exact = bool(
            observed.signature == expected_action.signature
            and str(state_signature_after) == expected_after
        )
        outcome = _empty_outcome()
        outcome.update(
            {
                "frontier_id": active.frontier_id,
                "terminal_success": bool(level_progressed or won),
                "level_progressed": bool(level_progressed),
                "won": bool(won),
                "game_over": bool(game_over),
                "frontier_reacquisition_observation": True,
                "frontier_reacquisition_confirmed": False,
                "frontier_reacquisition_diverged": False,
            }
        )
        if not exact or level_progressed or won or game_over:
            path.divergences += 1
            self._frontier_reacquisition_divergences += 1
            outcome["frontier_reacquisition_diverged"] = True
            self._active_reacquisition = None
            return outcome
        active.actions.append(observed)
        self._frontier_reacquisition_actions += 1
        if len(active.actions) >= len(path.actions):
            frontier = self._frontiers.get(active.frontier_id)
            if (
                frontier is not None
                and str(state_signature_after) == frontier.state_signature
            ):
                path.confirmations += 1
                self._frontier_reacquisition_confirmations += 1
                self._reacquired_frontier_id = active.frontier_id
                outcome["frontier_reacquisition_confirmed"] = True
            else:
                path.divergences += 1
                self._frontier_reacquisition_divergences += 1
                outcome["frontier_reacquisition_diverged"] = True
            self._active_reacquisition = None
        return outcome

    def _record_frontier_acquisition_path(
        self,
        frontier: TerminalNegativeFrontier,
    ) -> None:
        if (
            not self.enable_active_frontier_reacquisition
            or frontier.frontier_kind != "structural_change"
            or not self._branch_actions
            or not self._branch_state_signatures
            or self._branch_state_signatures[-1] != frontier.state_signature
            or len(self._branch_actions)
            > self.max_frontier_reacquisition_actions
        ):
            return
        actions = tuple(self._branch_actions)
        states = tuple(self._branch_state_signatures)
        signature = (
            tuple(action.signature for action in actions)
            + tuple(f"state::{state}" for state in states)
        )
        if signature in frontier.acquisition_paths:
            return
        if len(frontier.acquisition_paths) >= self.max_frontier_acquisition_paths:
            return
        frontier.acquisition_paths[signature] = FrontierAcquisitionPath(
            actions=actions,
            state_signatures=states,
        )
        self._acquisition_paths_recorded += 1

    def _compile_structural_transfer_probes(
        self,
        target: TerminalNegativeFrontier,
    ) -> None:
        """Nominate confirmed continuations across one structural class."""
        if (
            not self.enable_structural_frontier_transfer
            or target.frontier_kind != "structural_change"
            or not target.structural_equivalence_signature
            or len(target.structural_transfer_probes)
            >= self.max_structural_transfer_probes_per_frontier
        ):
            return
        sources = sorted(
            (
                frontier
                for frontier in self._frontiers.values()
                if frontier.frontier_id != target.frontier_id
                and frontier.frontier_kind == "structural_change"
                and frontier.structural_equivalence_signature
                == target.structural_equivalence_signature
                and frontier.successful_continuations
            ),
            key=lambda item: item.frontier_id,
        )
        for source in sources:
            continuations = sorted(
                source.successful_continuations.values(),
                key=lambda item: (
                    -item.confirmations,
                    len(item.actions),
                    item.signature,
                ),
            )
            for continuation in continuations:
                if (
                    continuation.structural_transfer
                    and continuation.confirmations < 2
                ):
                    continue
                payload = "|".join(
                    (
                        source.frontier_id,
                        target.frontier_id,
                        "|".join(continuation.signature),
                    )
                )
                probe_id = (
                    "structural-transfer::"
                    f"{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"
                )
                if probe_id in target.structural_transfer_probes:
                    continue
                target.structural_transfer_probes[probe_id] = (
                    StructuralTransferProbe(
                        probe_id=probe_id,
                        source_frontier_id=source.frontier_id,
                        source_continuation_signature=(
                            continuation.signature
                        ),
                        actions=continuation.actions,
                    )
                )
                self._structural_transfer_probes_compiled += 1
                if (
                    len(target.structural_transfer_probes)
                    >= self.max_structural_transfer_probes_per_frontier
                ):
                    return

    def _compile_structural_transfer_probes_for_source(
        self,
        source: TerminalNegativeFrontier,
    ) -> None:
        if (
            not self.enable_structural_frontier_transfer
            or not source.structural_equivalence_signature
        ):
            return
        for target in self._frontiers.values():
            if (
                target.frontier_id != source.frontier_id
                and target.structural_equivalence_signature
                == source.structural_equivalence_signature
            ):
                self._compile_structural_transfer_probes(target)

    def _best_structural_transfer_probe(
        self,
        frontier: TerminalNegativeFrontier,
    ) -> StructuralTransferProbe | None:
        if not self.enable_structural_frontier_transfer:
            return None
        candidates = [
            probe
            for probe in frontier.structural_transfer_probes.values()
            if probe.confirmations == 0
            and probe.refutations == 0
            and probe.attempts < self.max_structural_transfer_attempts
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                len(item.actions),
                item.source_frontier_id,
                item.probe_id,
            ),
        )

    def _record_selection(
        self,
        active: _ActiveSuffix,
        frontier: TerminalNegativeFrontier,
        action: TerminalFrontierAction,
        *,
        replaying: bool,
        replaying_dormant_candidate: bool,
        testing_causal_reduction: bool,
        testing_structural_transfer: bool,
    ) -> TerminalFrontierSelection:
        prefix = tuple(item.signature for item in active.actions)
        self._choice_counts[(frontier.frontier_id, prefix, action.signature)] += 1
        selection = TerminalFrontierSelection(
            frontier_id=frontier.frontier_id,
            objective_ids=frontier.objective_ids,
            action=action,
            step_index=len(active.actions),
            action_limit=active.action_limit,
            replaying_successful_continuation=replaying,
            replaying_dormant_terminal_candidate=(
                replaying_dormant_candidate
            ),
            testing_causal_reduction=testing_causal_reduction,
            testing_structural_transfer=testing_structural_transfer,
            reason=(
                "replay terminal-credited continuation from identical frontier"
                if replaying
                else "replay delayed-terminal candidate from identical frontier"
                if replaying_dormant_candidate
                else "controlled causal cut of a terminal-confirmed continuation"
                if testing_causal_reduction
                else (
                    "terminal-only continuation transfer from a structurally "
                    "equivalent confirmed frontier"
                )
                if testing_structural_transfer
                else (
                    "adaptive terminal-only continuation after exhausted "
                    "negative frontier"
                    if active.action_limit > self.max_suffix_actions
                    else (
                        "bounded contrast after nonterminal objective "
                        "postcondition"
                    )
                )
            ),
        )
        active.pending = selection
        return selection

    def _start_dormant_lineage(self, active: _ActiveSuffix) -> None:
        """Keep observing the live policy after its bounded suffix expires."""
        if (
            not self.enable_dormant_terminal_lineage
            or len(active.actions) >= self.max_dormant_lineage_actions
        ):
            return
        frontier = self._frontiers[active.frontier_id]
        if (
            frontier.frontier_kind == "structural_change"
            and not self.enable_structural_terminal_attribution
        ):
            self._structural_attribution_blocks += 1
            return
        self._dormant = _DormantLineage(
            frontier_id=active.frontier_id,
            actions=list(active.actions),
            state_signatures=list(active.state_signatures),
        )
        self._dormant_lineages_started += 1

    def _observe_dormant_lineage(
        self,
        *,
        state_signature_after: str,
        observed: TerminalFrontierAction,
        level_progressed: bool,
        won: bool,
        game_over: bool,
    ) -> Dict[str, Any]:
        lineage = self._dormant
        if not self.enable_dormant_terminal_lineage or lineage is None:
            return _empty_outcome()
        lineage.actions.append(observed)
        lineage.state_signatures.append(str(state_signature_after))
        self._dormant_lineage_actions += 1
        frontier = self._frontiers[lineage.frontier_id]
        terminal_success = bool(level_progressed or won)
        outcome = _empty_outcome()
        outcome.update(
            {
                "frontier_id": frontier.frontier_id,
                "objective_ids": list(frontier.objective_ids),
                "suffix_step": len(lineage.actions) - 1,
                "action_limit": self.max_dormant_lineage_actions,
                "terminal_success": terminal_success,
                "level_progressed": bool(level_progressed),
                "won": bool(won),
                "game_over": bool(game_over),
                "dormant_lineage_observation": True,
            }
        )
        if terminal_success:
            candidate = self._record_dormant_terminal_candidate(
                frontier,
                tuple(lineage.actions),
                tuple(lineage.state_signatures),
                level_progressed=bool(level_progressed),
                won=bool(won),
            )
            outcome["dormant_terminal_candidate_nominated"] = bool(
                candidate is not None
            )
            self._dormant = None
        elif game_over:
            self._dormant_lineage_unsafe += 1
            self._dormant = None
        elif len(lineage.actions) >= self.max_dormant_lineage_actions:
            self._dormant_lineage_expired += 1
            self._dormant = None
        return outcome

    def _record_dormant_terminal_candidate(
        self,
        frontier: TerminalNegativeFrontier,
        actions: Tuple[TerminalFrontierAction, ...],
        state_signatures: Tuple[str, ...],
        *,
        level_progressed: bool,
        won: bool,
    ) -> DormantTerminalContinuation | None:
        """Nominate delayed terminal evidence without granting credit."""
        signature = tuple(action.signature for action in actions)
        candidate = frontier.dormant_terminal_candidates.get(signature)
        if candidate is not None:
            candidate.terminal_observations += 1
            candidate.level_progressed = bool(
                candidate.level_progressed or level_progressed
            )
            candidate.won = bool(candidate.won or won)
            return candidate
        if (
            len(frontier.dormant_terminal_candidates)
            >= self.max_dormant_candidates_per_frontier
        ):
            self._dormant_candidate_capacity_blocks += 1
            return None
        candidate = DormantTerminalContinuation(
            actions=actions,
            state_signatures=state_signatures,
            level_progressed=bool(level_progressed),
            won=bool(won),
        )
        frontier.dormant_terminal_candidates[signature] = candidate
        self._dormant_lineage_terminal_candidates += 1
        self._dormant_lineage_level_candidates += int(bool(level_progressed))
        self._dormant_lineage_win_candidates += int(bool(won))
        if frontier.frontier_kind == "structural_change":
            self._structural_terminal_candidates += 1
        return candidate

    def _credit_continuation(
        self,
        frontier: TerminalNegativeFrontier,
        actions: Tuple[TerminalFrontierAction, ...],
        state_signatures: Tuple[str, ...],
        *,
        level_progressed: bool,
        won: bool,
        replaying: bool,
        causal_reduction: bool = False,
        removed_action_indices: Tuple[int, ...] = (),
        structural_transfer: bool = False,
        source_frontier_id: str = "",
    ) -> None:
        """Credit only an actually observed terminal continuation."""
        signature = tuple(action.signature for action in actions)
        continuation = frontier.successful_continuations.get(signature)
        if continuation is None:
            continuation = SuccessfulContinuation(
                actions,
                state_signatures,
                causal_reduction=bool(causal_reduction),
                removed_action_indices=tuple(removed_action_indices),
                structural_transfer=bool(structural_transfer),
                source_frontier_id=str(source_frontier_id),
            )
            frontier.successful_continuations[signature] = continuation
        else:
            continuation.confirmations += 1
        frontier.terminal_credits += 1
        self._terminal_credits += 1
        self._level_change_credits += int(bool(level_progressed))
        self._win_credits += int(bool(won))
        if replaying:
            self._successful_replays += 1
        if frontier.frontier_kind == "structural_change":
            self._structural_terminal_credits += 1

    def _compile_causal_reduction_probes(
        self,
        frontier: TerminalNegativeFrontier,
        actions: Tuple[TerminalFrontierAction, ...],
        *,
        generation: int = 1,
        parent_probe_id: str = "",
        origin_indices: Tuple[int, ...] = (),
        previously_removed: Tuple[int, ...] = (),
    ) -> None:
        """Pre-register deterministic cuts, recursively after confirmation."""
        if (
            not self.enable_terminal_causal_reduction
            or frontier.frontier_kind != "structural_change"
            or len(actions) < 4
            or generation > self.max_causal_reduction_generations
            or len(frontier.causal_reduction_probes)
            >= self.max_causal_reduction_probes_total
            or (
                generation > 1
                and not self.enable_recursive_terminal_causal_minimization
            )
        ):
            return
        length = len(actions)
        origins = (
            tuple(origin_indices)
            if len(origin_indices) == length
            else tuple(range(length))
        )
        block = max(1, length // 4)
        cuts = (
            ("drop_leading_quarter", tuple(range(0, block))),
            (
                "drop_middle_quarter",
                tuple(
                    range(
                        max(0, (length - block) // 2),
                        max(0, (length - block) // 2) + block,
                    )
                ),
            ),
            (
                "drop_trailing_quarter",
                tuple(range(max(0, length - block), length)),
            ),
        )
        source_signature = tuple(action.signature for action in actions)
        for strategy, removed in cuts[
            : self.max_causal_reduction_probes_per_frontier
        ]:
            if (
                len(frontier.causal_reduction_probes)
                >= self.max_causal_reduction_probes_total
            ):
                break
            removed_set = set(removed)
            reduced = tuple(
                action
                for index, action in enumerate(actions)
                if index not in removed_set
            )
            if not reduced or len(reduced) >= len(actions):
                continue
            kept_origins = tuple(
                origin
                for index, origin in enumerate(origins)
                if index not in removed_set
            )
            removed_origins = tuple(sorted(
                set(previously_removed).union(
                    origins[index] for index in removed_set
                )
            ))
            payload = "|".join(
                (
                    frontier.frontier_id,
                    str(generation),
                    parent_probe_id,
                    strategy,
                    ",".join(map(str, removed_origins)),
                    "|".join(source_signature),
                )
            )
            probe_id = (
                "causal-reduction::"
                f"{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"
            )
            if probe_id in frontier.causal_reduction_probes:
                continue
            frontier.causal_reduction_probes[probe_id] = CausalReductionProbe(
                probe_id=probe_id,
                source_signature=source_signature,
                actions=reduced,
                removed_action_indices=removed_origins,
                strategy=strategy,
                generation=int(generation),
                parent_probe_id=str(parent_probe_id),
                kept_origin_indices=kept_origins,
            )
            self._causal_reduction_probes_compiled += 1
            if generation > 1:
                self._recursive_reduction_probes_compiled += 1
            self._maximum_reduction_generation = max(
                self._maximum_reduction_generation,
                int(generation),
            )

    def _best_causal_reduction_probe(
        self,
        frontier: TerminalNegativeFrontier,
    ) -> CausalReductionProbe | None:
        probes = [
            probe
            for probe in frontier.causal_reduction_probes.values()
            if probe.confirmations == 0
            and probe.refutations == 0
            and probe.replay_attempts < self.max_causal_reduction_replays
        ]
        if not probes:
            return None
        return min(
            probes,
            key=lambda item: (
                len(item.actions),
                item.strategy,
                item.probe_id,
            ),
        )

    def _extend_exhausted_frontier_horizon(
        self,
        frontier: TerminalNegativeFrontier,
    ) -> None:
        """Grant one larger bound only after the previous bound was exhausted."""
        if (
            not self.enable_adaptive_horizon
            or frontier.allocated_action_limit >= self.max_adaptive_suffix_actions
            or frontier.nonterminal_suffixes <= frontier.horizon_extensions
        ):
            return
        previous = frontier.allocated_action_limit
        allocated = min(
            self.max_adaptive_suffix_actions,
            previous + self.adaptive_horizon_increment,
        )
        if allocated <= previous:
            return
        frontier.allocated_action_limit = allocated
        frontier.horizon_extensions += 1
        frontier.horizon_history.append(allocated)
        self._adaptive_horizon_extensions += 1
        self._adaptive_horizon_actions_granted += allocated - previous

    def _best_dormant_terminal_candidate(
        self,
        frontier: TerminalNegativeFrontier,
    ) -> DormantTerminalContinuation | None:
        candidates = [
            candidate
            for candidate in frontier.dormant_terminal_candidates.values()
            if candidate.confirmations == 0
            and candidate.refutations == 0
            and candidate.replay_attempts < self.max_dormant_candidate_replays
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item.terminal_observations,
                -len(item.actions),
                item.signature,
            ),
        )

    @staticmethod
    def _best_successful_continuation(
        frontier: TerminalNegativeFrontier,
    ) -> SuccessfulContinuation | None:
        if not frontier.successful_continuations:
            return None
        return max(
            frontier.successful_continuations.values(),
            key=lambda item: (
                item.confirmations,
                -len(item.actions),
                item.signature,
            ),
        )


def _frontier_id(
    state: str,
    objectives: Sequence[str],
    frontier_kind: str,
) -> str:
    payload = (
        f"{frontier_kind}|{state}|{'|'.join(objectives)}".encode("utf-8")
    )
    return f"terminal-frontier::{hashlib.sha1(payload).hexdigest()[:16]}"


def _empty_outcome() -> Dict[str, Any]:
    return {
        "frontier_id": "",
        "objective_ids": [],
        "suffix_step": None,
        "action_limit": 0,
        "adaptive_horizon": False,
        "terminal_success": False,
        "level_progressed": False,
        "won": False,
        "game_over": False,
        "credited": False,
        "replaying_successful_continuation": False,
        "replaying_dormant_terminal_candidate": False,
        "testing_causal_reduction": False,
        "testing_structural_transfer": False,
        "dormant_lineage_observation": False,
        "dormant_terminal_candidate_nominated": False,
        "dormant_terminal_candidate_confirmed": False,
        "causal_reduction_confirmed": False,
        "causal_reduction_refuted": False,
        "structural_transfer_confirmed": False,
        "structural_transfer_refuted": False,
        "frontier_reacquisition_observation": False,
        "frontier_reacquisition_confirmed": False,
        "frontier_reacquisition_diverged": False,
    }


__all__ = [
    "CausalReductionProbe",
    "DormantTerminalContinuation",
    "FrontierAcquisitionPath",
    "OnlineTerminalFrontierExplorer",
    "SuccessfulContinuation",
    "TerminalFrontierAction",
    "TerminalFrontierReacquisitionSelection",
    "TerminalFrontierSelection",
    "TerminalNegativeFrontier",
    "StructuralTransferProbe",
]
