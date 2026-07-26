"""Online frontier-oriented exploration for stalled ARC episodes.

The explorer has no access to game identifiers, level indices, rewards, or
solutions.  It observes only the live grid, legal actions, branch-progress
diagnostics, and the transition produced by its own interventions.

SAGE.9v turns a sterile branch into a bounded scientific phase:

* detect repeated states or a lack of terminal progress;
* enumerate concrete, parameterized actuators rather than action names only;
* describe clicked objects by palette-invariant structural roles;
* prioritize state/action and object/action pairs that have not been tested;
* continue a productive intervention for a short, bounded burst;
* credit only observed novel effects, states, or terminal outcomes.

SAGE.10a keeps a bounded eligibility trace for productive, safe frontier
effects and attributes a later terminal in the same branch to at most one
intervention per scientific sequence.  SAGE.10b relays that identity through
bounded, structurally linked sub-effects.  SAGE.10c adds complementary stall
signals that do not require exact state recurrence, and SAGE.10d lets a
retired explorer re-arm only after a later level enters a genuine stall.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


ActionData = Tuple[Tuple[str, Any], ...]


@dataclass(frozen=True)
class FrontierExperimentSelection:
    """One concrete intervention selected at a detected frontier."""

    action_name: str
    action_data: Dict[str, Any]
    frontier_id: str
    state_signature: str
    context_signature: str
    actuator_signature: str
    target_role_signature: str
    information_score: float
    sequence_id: str
    sequence_step: int
    sequence_limit: int
    state_action_untested: bool
    actuator_untested: bool
    object_role_untested: bool
    reason: str


@dataclass(frozen=True)
class FrontierEligibilityAssessment:
    """Read-only diagnosis of whether frontier authority is currently eligible."""

    eligible: bool
    state_signature: str = ""
    context_signature: str = ""
    stagnant: bool = False
    stall_reasons: Tuple[str, ...] = ()
    in_active_sequence: bool = False
    untested_actuator_available: bool = False
    candidate_count: int = 0
    blocked_reason: str = ""


@dataclass(frozen=True)
class DelayedFrontierCredit:
    """One earlier information-seeking action credited by a later terminal."""

    eligibility_id: str
    frontier_id: str
    sequence_id: str
    action_name: str
    action_data: Dict[str, Any]
    actuator_signature: str
    target_role_signature: str
    effect_signature: str
    state_signature_before: str
    state_signature_after: str
    delay_actions: int
    novel_effect: bool
    novel_state: bool
    information_gain: float
    relay_hops: int = 0
    relay_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FrontierDelayedCreditUpdate:
    """Resolution of branch-local frontier eligibility traces."""

    credited: Tuple[DelayedFrontierCredit, ...] = ()
    relayed_eligibility_ids: Tuple[str, ...] = ()
    expired_eligibility_ids: Tuple[str, ...] = ()
    discarded_eligibility_ids: Tuple[str, ...] = ()


@dataclass
class _ActuatorEvidence:
    trials: int = 0
    noops: int = 0
    unsafe_outcomes: int = 0
    terminal_outcomes: int = 0
    effect_signatures: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class _FrontierEligibility:
    eligibility_id: str
    frontier_id: str
    sequence_id: str
    action_name: str
    action_data: Dict[str, Any]
    actuator_signature: str
    target_role_signature: str
    effect_signature: str
    state_signature_before: str
    state_signature_after: str
    created_transition_index: int
    transition_index: int
    novel_effect: bool
    novel_state: bool
    information_gain: float
    causal_effect_signature: str = ""
    current_effect_signature: str = ""
    current_causal_effect_signature: str = ""
    current_target_role_signature: str = ""
    component_signatures: Tuple[str, ...] = ()
    relay_depth: int = 0
    relay_reasons: Tuple[str, ...] = ()


class OnlineFrontierExplorer:
    """Select bounded information-seeking actions when progress stalls."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_stagnant_steps: int = 6,
        max_experiments_per_state: int = 8,
        max_sequence_actions: int = 3,
        max_trials_per_actuator: int = 2,
        minimum_failed_branches: int = 0,
        enable_delayed_terminal_credit: bool = True,
        delayed_terminal_credit_window: int = 12,
        max_delayed_credits_per_terminal: int = 3,
        enable_subeffect_eligibility_relay: bool = True,
        max_subeffect_relay_depth: int = 3,
        enable_generalized_stall_detection: bool = True,
        effect_novelty_stall_actions: int = 12,
        zero_terminal_branch_stall_count: int = 3,
        enable_per_level_rearming: bool = True,
        nonprogress_demotion_threshold: int = 2,
    ) -> None:
        self.enabled = bool(enabled)
        self.minimum_stagnant_steps = max(
            1,
            int(minimum_stagnant_steps),
        )
        self.max_experiments_per_state = max(
            1,
            int(max_experiments_per_state),
        )
        self.max_sequence_actions = max(1, int(max_sequence_actions))
        self.max_trials_per_actuator = max(
            1,
            int(max_trials_per_actuator),
        )
        self.minimum_failed_branches = max(
            0,
            int(minimum_failed_branches),
        )
        self.enable_delayed_terminal_credit = bool(
            enable_delayed_terminal_credit
        )
        self.delayed_terminal_credit_window = max(
            1,
            int(delayed_terminal_credit_window),
        )
        self.max_delayed_credits_per_terminal = max(
            1,
            int(max_delayed_credits_per_terminal),
        )
        self.enable_subeffect_eligibility_relay = bool(
            enable_subeffect_eligibility_relay
        )
        self.max_subeffect_relay_depth = max(
            1,
            int(max_subeffect_relay_depth),
        )
        self.enable_generalized_stall_detection = bool(
            enable_generalized_stall_detection
        )
        self.effect_novelty_stall_actions = max(
            1,
            int(effect_novelty_stall_actions),
        )
        self.zero_terminal_branch_stall_count = max(
            1,
            int(zero_terminal_branch_stall_count),
        )
        self.enable_per_level_rearming = bool(enable_per_level_rearming)
        self.nonprogress_demotion_threshold = max(
            1,
            int(nonprogress_demotion_threshold),
        )

        self._state_visits: Counter[str] = Counter()
        self._state_action_trials: Counter[Tuple[str, str]] = Counter()
        self._state_experiments: Counter[str] = Counter()
        self._actuator_evidence: Dict[str, _ActuatorEvidence] = {}
        self._tested_target_roles: set[str] = set()
        self._seen_effects: set[str] = set()
        self._seen_states: set[str] = set()
        self._frontier_states: set[str] = set()
        self._target_role_actions: Dict[str, set[str]] = defaultdict(set)
        self._context_actuator_nonprogress: Counter[
            Tuple[str, str]
        ] = Counter()
        self._demoted_context_actuators: set[Tuple[str, str]] = set()

        self._pending: FrontierExperimentSelection | None = None
        self._active_sequence_id = ""
        self._active_sequence_step = 0
        self._active_sequence_remaining = 0
        self._sequence_serial = 0
        self._branches_started = 0
        self._failed_branches = 0
        self._branch_terminal_progress = False
        self._terminal_progress_observed = False
        self._branch_index = 0
        self._branch_transition_index = 0
        self._eligibility_serial = 0
        self._delayed_eligibilities: list[_FrontierEligibility] = []
        self._seen_transition_effect_classes: set[str] = set()
        self._actions_since_novel_effect = 0
        self._level_rearm_pending = False
        self._last_stall_reasons: Tuple[str, ...] = ()

        self._states_assessed = 0
        self._stagnation_detections = 0
        self._frontier_entries = 0
        self._experiments = 0
        self._sequence_actions = 0
        self._multi_step_sequences = 0
        self._untested_state_actions = 0
        self._untested_actuator_actions = 0
        self._untested_object_actions = 0
        self._productive_experiments = 0
        self._noop_experiments = 0
        self._unsafe_experiments = 0
        self._novel_effects = 0
        self._novel_states = 0
        self._terminal_credits = 0
        self._information_gain = 0.0
        self._delayed_eligibilities_registered = 0
        self._delayed_terminal_events = 0
        self._delayed_terminal_credits = 0
        self._delayed_credit_delay_actions = 0
        self._delayed_credit_max_delay = 0
        self._expired_delayed_eligibilities = 0
        self._discarded_delayed_eligibilities = 0
        self._censored_delayed_eligibilities = 0
        self._unsafe_delayed_eligibilities = 0
        self._subeffect_relays_created = 0
        self._subeffect_relay_depth_histogram: Counter[int] = Counter()
        self._subeffect_relay_reason_counts: Counter[str] = Counter()
        self._effect_novelty_stalls = 0
        self._actuator_coverage_stalls = 0
        self._zero_terminal_branch_stalls = 0
        self._level_changes_observed = 0
        self._per_level_rearms = 0
        self._protected_competence_blocks = 0
        self._nonprogress_outcomes = 0
        self._context_actuator_demotions = 0
        self._context_actuator_demotion_blocks = 0
        self._context_actuator_reactivations = 0

    def assess_eligibility(
        self,
        *,
        current_grid: Any,
        available_actions: Sequence[str],
        available_action_candidates: Sequence[Any] | None,
        branch_diagnostics: Mapping[str, Any],
    ) -> FrontierEligibilityAssessment:
        """Diagnose a frontier without consuming authority or mutating memory."""
        if not self.enabled:
            return FrontierEligibilityAssessment(
                eligible=False,
                blocked_reason="disabled",
            )
        if (
            self._failed_branches < self.minimum_failed_branches
            and not self._level_rearm_pending
        ):
            return FrontierEligibilityAssessment(
                eligible=False,
                blocked_reason="minimum_failed_branches",
            )
        grid = np.asarray(current_grid, dtype=np.int32)
        if grid.ndim != 2 or grid.size == 0:
            return FrontierEligibilityAssessment(
                eligible=False,
                blocked_reason="invalid_grid",
            )
        state_signature = _state_signature(grid)
        context_signature = _context_signature(grid)
        candidates = _concrete_candidates(
            grid,
            available_actions,
            available_action_candidates,
        )
        if not candidates:
            return FrontierEligibilityAssessment(
                eligible=False,
                state_signature=state_signature,
                context_signature=context_signature,
                blocked_reason="no_candidates",
            )
        eligible_candidates = tuple(
            candidate
            for candidate in candidates
            if (
                context_signature,
                candidate[2],
            ) not in self._demoted_context_actuators
        )
        untested_actuator_available = any(
            (
                self._actuator_evidence.get(actuator) is None
                or self._actuator_evidence[actuator].trials
                < self.max_trials_per_actuator
            )
            for _, _, actuator, _ in eligible_candidates
        )
        stagnant, stall_reasons = self._stagnation_assessment(
            state_signature,
            branch_diagnostics,
            untested_actuator_available=untested_actuator_available,
            prospective_state_visits=(
                self._state_visits[state_signature] + 1
            ),
        )
        in_active_sequence = bool(self._active_sequence_remaining > 0)
        blocked_reason = ""
        if not eligible_candidates:
            blocked_reason = "all_context_actuators_demoted"
        elif (
            self._state_experiments[state_signature]
            >= self.max_experiments_per_state
        ):
            blocked_reason = "state_experiment_budget"
        elif self._terminal_progress_observed and not (
            self.enable_per_level_rearming
            and self._level_rearm_pending
            and stagnant
        ):
            blocked_reason = "terminal_retreat"
        elif not stagnant and not in_active_sequence:
            blocked_reason = "not_stagnant"
        return FrontierEligibilityAssessment(
            eligible=not bool(blocked_reason),
            state_signature=state_signature,
            context_signature=context_signature,
            stagnant=bool(stagnant),
            stall_reasons=stall_reasons,
            in_active_sequence=in_active_sequence,
            untested_actuator_available=untested_actuator_available,
            candidate_count=len(eligible_candidates),
            blocked_reason=blocked_reason,
        )

    def select(
        self,
        *,
        current_grid: Any,
        available_actions: Sequence[str],
        available_action_candidates: Sequence[Any] | None,
        branch_diagnostics: Mapping[str, Any],
        protected_competence_available: bool = False,
        assessment: FrontierEligibilityAssessment | None = None,
    ) -> FrontierExperimentSelection | None:
        """Return the most informative safe-looking concrete intervention."""
        if not self.enabled:
            return None
        if protected_competence_available:
            self._protected_competence_blocks += 1
            self._clear_sequence()
            return None
        grid = np.asarray(current_grid, dtype=np.int32)
        if grid.ndim != 2 or grid.size == 0:
            return None
        if assessment is None:
            assessment = self.assess_eligibility(
                current_grid=grid,
                available_actions=available_actions,
                available_action_candidates=available_action_candidates,
                branch_diagnostics=branch_diagnostics,
            )
        if not assessment.eligible:
            if assessment.blocked_reason == "all_context_actuators_demoted":
                self._context_actuator_demotion_blocks += 1
            if assessment.in_active_sequence:
                self._clear_sequence()
            return None

        self._states_assessed += 1
        state_signature = assessment.state_signature
        context_signature = assessment.context_signature
        self._state_visits[state_signature] += 1
        self._seen_states.add(state_signature)
        in_active_sequence = assessment.in_active_sequence
        candidates = _concrete_candidates(
            grid,
            available_actions,
            available_action_candidates,
        )
        if not candidates:
            return None
        stagnant = assessment.stagnant
        stall_reasons = assessment.stall_reasons
        self._last_stall_reasons = stall_reasons
        if self._terminal_progress_observed:
            if not (
                self.enable_per_level_rearming
                and self._level_rearm_pending
                and stagnant
            ):
                return None
            self._terminal_progress_observed = False
            self._level_rearm_pending = False
            self._per_level_rearms += 1
        if not stagnant and not in_active_sequence:
            return None
        if stagnant:
            self._stagnation_detections += 1
            self._effect_novelty_stalls += int(
                "effect_novelty_stall" in stall_reasons
            )
            self._actuator_coverage_stalls += int(
                "actuator_coverage_stall" in stall_reasons
            )
            self._zero_terminal_branch_stalls += int(
                "zero_terminal_branch_stall" in stall_reasons
            )
        if (
            self._state_experiments[state_signature]
            >= self.max_experiments_per_state
        ):
            if in_active_sequence:
                self._clear_sequence()
            return None

        ranked = []
        for action_name, action_data, actuator, target_role in candidates:
            if (
                context_signature,
                actuator,
            ) in self._demoted_context_actuators:
                self._context_actuator_demotion_blocks += 1
                continue
            state_trials = self._state_action_trials[
                (state_signature, actuator)
            ]
            evidence = self._actuator_evidence.get(actuator)
            actuator_trials = 0 if evidence is None else evidence.trials
            role_untested = bool(
                target_role
                and target_role not in self._tested_target_roles
            )
            noop_rate = (
                0.0
                if evidence is None or evidence.trials <= 0
                else evidence.noops / evidence.trials
            )
            unsafe_rate = (
                0.0
                if evidence is None or evidence.trials <= 0
                else evidence.unsafe_outcomes / evidence.trials
            )
            effect_diversity = (
                0 if evidence is None else len(evidence.effect_signatures)
            )
            if actuator_trials >= self.max_trials_per_actuator:
                continue
            score = (
                8.0 * float(state_trials == 0)
                + 5.0 * float(actuator_trials == 0)
                + 3.0 * float(role_untested)
                + 1.5 * float(effect_diversity == 0)
                + 0.5 * float(in_active_sequence)
                - 4.0 * noop_rate
                - 10.0 * unsafe_rate
                - 0.25 * float(state_trials)
            )
            ranked.append((
                score,
                -state_trials,
                -actuator_trials,
                action_name,
                repr(action_data),
                action_data,
                actuator,
                target_role,
                state_trials == 0,
                actuator_trials == 0,
                role_untested,
            ))
        if not ranked:
            self._clear_sequence()
            return None
        ranked.sort(reverse=True)
        (
            score,
            _,
            _,
            action_name,
            _,
            action_data,
            actuator,
            target_role,
            state_untested,
            actuator_untested,
            role_untested,
        ) = ranked[0]

        if not self._active_sequence_id:
            self._sequence_serial += 1
            self._active_sequence_id = (
                f"frontier-sequence-{self._sequence_serial:04d}"
            )
            self._active_sequence_step = 0
        self._active_sequence_step += 1
        frontier_id = f"frontier::{state_signature}"
        selection = FrontierExperimentSelection(
            action_name=action_name,
            action_data=dict(action_data),
            frontier_id=frontier_id,
            state_signature=state_signature,
            context_signature=context_signature,
            actuator_signature=actuator,
            target_role_signature=target_role,
            information_score=float(score),
            sequence_id=self._active_sequence_id,
            sequence_step=self._active_sequence_step,
            sequence_limit=self.max_sequence_actions,
            state_action_untested=bool(state_untested),
            actuator_untested=bool(actuator_untested),
            object_role_untested=bool(role_untested),
            reason=(
                "stagnant frontier: maximize causal information over "
                "untested actuator/object interventions"
            ),
        )
        self._pending = selection
        self._state_action_trials[(state_signature, actuator)] += 1
        self._state_experiments[state_signature] += 1
        self._experiments += 1
        self._sequence_actions += 1
        if state_untested:
            self._untested_state_actions += 1
        if actuator_untested:
            self._untested_actuator_actions += 1
        if role_untested:
            self._untested_object_actions += 1
        if state_signature not in self._frontier_states:
            self._frontier_states.add(state_signature)
            self._frontier_entries += 1
        return selection

    def observe_transition(
        self,
        *,
        grid_before: Any,
        grid_after: Any,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        no_effect: bool,
        game_over: bool,
        terminal_success: bool,
        causal_effect_signature: str = "",
    ) -> Dict[str, Any]:
        """Credit only information physically observed after our action."""
        pending = self._pending
        self._pending = None
        if (
            pending is None
            or pending.action_name != str(action_name)
            or dict(pending.action_data) != dict(action_data or {})
        ):
            return {"observed": False}

        before = np.asarray(grid_before, dtype=np.int32)
        after = np.asarray(grid_after, dtype=np.int32)
        effect_signature = _effect_signature(before, after)
        component_signatures = _effect_component_signatures(before, after)
        after_signature = _state_signature(after)
        evidence = self._actuator_evidence.setdefault(
            pending.actuator_signature,
            _ActuatorEvidence(),
        )
        evidence.trials += 1
        evidence.noops += int(bool(no_effect))
        evidence.unsafe_outcomes += int(bool(game_over))
        evidence.terminal_outcomes += int(bool(terminal_success))
        evidence.effect_signatures[effect_signature] += 1
        if pending.target_role_signature:
            self._tested_target_roles.add(
                pending.target_role_signature
            )
            self._target_role_actions[
                pending.target_role_signature
            ].add(pending.actuator_signature)

        novel_effect = effect_signature not in self._seen_effects
        novel_state = after_signature not in self._seen_states
        self._seen_effects.add(effect_signature)
        self._seen_states.add(after_signature)
        gain = (
            float(novel_effect)
            + float(novel_state)
            + 2.0 * float(terminal_success)
        )
        self._information_gain += gain
        self._novel_effects += int(novel_effect)
        self._novel_states += int(novel_state)
        self._terminal_credits += int(bool(terminal_success))
        self._noop_experiments += int(bool(no_effect))
        self._unsafe_experiments += int(bool(game_over))
        productive = bool(
            terminal_success
            or (not no_effect and not game_over and (novel_effect or novel_state))
        )
        self._productive_experiments += int(productive)
        context_actuator_key = (
            pending.context_signature,
            pending.actuator_signature,
        )
        if productive:
            self._context_actuator_nonprogress[context_actuator_key] = 0
            if context_actuator_key in self._demoted_context_actuators:
                self._demoted_context_actuators.discard(
                    context_actuator_key
                )
                self._context_actuator_reactivations += 1
        else:
            self._nonprogress_outcomes += 1
            self._context_actuator_nonprogress[context_actuator_key] += 1
            if (
                self._context_actuator_nonprogress[context_actuator_key]
                >= self.nonprogress_demotion_threshold
                and context_actuator_key
                not in self._demoted_context_actuators
            ):
                self._demoted_context_actuators.add(context_actuator_key)
                self._context_actuator_demotions += 1
        eligibility_id = ""
        if (
            self.enable_delayed_terminal_credit
            and productive
            and not terminal_success
            and not game_over
            and (novel_effect or novel_state)
        ):
            self._eligibility_serial += 1
            eligibility_id = (
                f"frontier-eligibility-{self._branch_index:04d}-"
                f"{self._eligibility_serial:06d}"
            )
            self._delayed_eligibilities.append(
                _FrontierEligibility(
                    eligibility_id=eligibility_id,
                    frontier_id=pending.frontier_id,
                    sequence_id=pending.sequence_id,
                    action_name=pending.action_name,
                    action_data=dict(pending.action_data),
                    actuator_signature=pending.actuator_signature,
                    target_role_signature=(
                        pending.target_role_signature
                    ),
                    effect_signature=effect_signature,
                    state_signature_before=pending.state_signature,
                    state_signature_after=after_signature,
                    created_transition_index=(
                        self._branch_transition_index + 1
                    ),
                    transition_index=(
                        self._branch_transition_index + 1
                    ),
                    novel_effect=bool(novel_effect),
                    novel_state=bool(novel_state),
                    information_gain=float(gain),
                    causal_effect_signature=str(
                        causal_effect_signature
                    ),
                    current_effect_signature=effect_signature,
                    current_causal_effect_signature=str(
                        causal_effect_signature
                    ),
                    current_target_role_signature=(
                        pending.target_role_signature
                    ),
                    component_signatures=component_signatures,
                )
            )
            self._delayed_eligibilities_registered += 1

        if (
            productive
            and not terminal_success
            and not game_over
            and pending.sequence_step < self.max_sequence_actions
        ):
            if pending.sequence_step == 1:
                self._multi_step_sequences += 1
            self._active_sequence_remaining = (
                self.max_sequence_actions - pending.sequence_step
            )
        elif self._active_sequence_remaining > 0 and not game_over:
            self._active_sequence_remaining -= 1
            if self._active_sequence_remaining <= 0:
                self._clear_sequence()
        else:
            self._clear_sequence()

        return {
            "observed": True,
            "productive": productive,
            "novel_effect": novel_effect,
            "novel_state": novel_state,
            "terminal_credit": bool(terminal_success),
            "information_gain": gain,
            "effect_signature": effect_signature,
            "causal_effect_signature": str(causal_effect_signature),
            "component_signatures": component_signatures,
            "delayed_credit_eligibility_id": eligibility_id,
        }

    def start_branch(self) -> Tuple[str, ...]:
        """End any censored burst while retaining cross-reset evidence."""
        discarded = tuple(
            eligibility.eligibility_id
            for eligibility in self._delayed_eligibilities
        )
        if discarded:
            self._censored_delayed_eligibilities += len(discarded)
            self._discarded_delayed_eligibilities += len(discarded)
        self._delayed_eligibilities = []
        if self._branches_started > 0:
            if self._branch_terminal_progress:
                self._failed_branches = 0
            else:
                self._failed_branches += 1
        self._branches_started += 1
        self._branch_index += 1
        self._branch_transition_index = 0
        self._branch_terminal_progress = False
        self._level_rearm_pending = False
        self._pending = None
        self._clear_sequence()
        return discarded

    def note_level_change(self) -> None:
        """Allow a later level to re-arm only after its own stall is observed."""
        self._level_changes_observed += 1
        if (
            self.enable_per_level_rearming
            and self._terminal_progress_observed
        ):
            self._level_rearm_pending = True

    def pending_causal_effect_signatures(self) -> Tuple[str, ...]:
        """Expose only abstract effect identities needed for graph linkage."""
        return tuple(sorted({
            signature
            for item in self._delayed_eligibilities
            for signature in (
                item.causal_effect_signature,
                item.current_causal_effect_signature,
            )
            if signature
        }))

    def note_transition(
        self,
        *,
        terminal_success: bool,
        game_over: bool = False,
        grid_before: Any | None = None,
        grid_after: Any | None = None,
        action_data: Mapping[str, Any] | None = None,
        causal_effect_signature: str = "",
        causally_linked_effect_signatures: Sequence[str] = (),
    ) -> FrontierDelayedCreditUpdate:
        """Suspend upstream exploration once any existing skill progresses."""
        self._branch_transition_index += 1
        expired = []
        discarded = []
        credited: list[DelayedFrontierCredit] = []
        relayed: list[str] = []
        current_effect_signature = ""
        current_target_role_signature = ""
        current_component_signatures: Tuple[str, ...] = ()
        if grid_before is not None and grid_after is not None:
            before = np.asarray(grid_before, dtype=np.int32)
            after = np.asarray(grid_after, dtype=np.int32)
            if before.ndim == 2 and after.ndim == 2:
                current_effect_signature = _effect_signature(before, after)
                current_component_signatures = (
                    _effect_component_signatures(before, after)
                )
                current_target_role_signature = _target_role_signature(
                    before,
                    action_data or {},
                )
        if current_effect_signature:
            if (
                current_effect_signature
                not in self._seen_transition_effect_classes
                and current_effect_signature != _noop_effect_signature()
            ):
                self._seen_transition_effect_classes.add(
                    current_effect_signature
                )
                self._actions_since_novel_effect = 0
            else:
                self._actions_since_novel_effect += 1
        if (
            self.enable_subeffect_eligibility_relay
            and not game_over
            and current_effect_signature
            and current_effect_signature != _noop_effect_signature()
        ):
            linked = set(str(item) for item in causally_linked_effect_signatures)
            refreshed = []
            for item in self._delayed_eligibilities:
                reason = self._relay_reason(
                    item,
                    effect_signature=current_effect_signature,
                    causal_effect_signature=str(causal_effect_signature),
                    target_role_signature=current_target_role_signature,
                    component_signatures=current_component_signatures,
                    causally_linked_effect_signatures=linked,
                )
                if (
                    not reason
                    or item.transition_index
                    >= self._branch_transition_index
                    or item.relay_depth >= self.max_subeffect_relay_depth
                ):
                    refreshed.append(item)
                    continue
                relayed_item = replace(
                    item,
                    transition_index=self._branch_transition_index,
                    current_effect_signature=current_effect_signature,
                    current_causal_effect_signature=str(
                        causal_effect_signature
                    ),
                    current_target_role_signature=(
                        current_target_role_signature
                    ),
                    component_signatures=(
                        current_component_signatures
                    ),
                    relay_depth=item.relay_depth + 1,
                    relay_reasons=item.relay_reasons + (reason,),
                )
                refreshed.append(relayed_item)
                relayed.append(item.eligibility_id)
                self._subeffect_relays_created += 1
                self._subeffect_relay_depth_histogram[
                    relayed_item.relay_depth
                ] += 1
                self._subeffect_relay_reason_counts[reason] += 1
            self._delayed_eligibilities = refreshed
        if terminal_success and self.enable_delayed_terminal_credit:
            eligible = [
                item
                for item in self._delayed_eligibilities
                if (
                    1
                    <= self._branch_transition_index
                    - item.transition_index
                    <= self.delayed_terminal_credit_window
                )
            ]
            best_by_sequence: Dict[str, _FrontierEligibility] = {}
            for item in eligible:
                current = best_by_sequence.get(item.sequence_id)
                if current is None or self._eligibility_rank(
                    item
                ) > self._eligibility_rank(current):
                    best_by_sequence[item.sequence_id] = item
            selected = sorted(
                best_by_sequence.values(),
                key=self._eligibility_rank,
                reverse=True,
            )[: self.max_delayed_credits_per_terminal]
            selected_ids = {
                item.eligibility_id for item in selected
            }
            for item in selected:
                delay = (
                    self._branch_transition_index
                    - item.created_transition_index
                )
                credited.append(
                    DelayedFrontierCredit(
                        eligibility_id=item.eligibility_id,
                        frontier_id=item.frontier_id,
                        sequence_id=item.sequence_id,
                        action_name=item.action_name,
                        action_data=dict(item.action_data),
                        actuator_signature=item.actuator_signature,
                        target_role_signature=(
                            item.target_role_signature
                        ),
                        effect_signature=item.effect_signature,
                        state_signature_before=(
                            item.state_signature_before
                        ),
                        state_signature_after=(
                            item.state_signature_after
                        ),
                        delay_actions=delay,
                        novel_effect=item.novel_effect,
                        novel_state=item.novel_state,
                        information_gain=item.information_gain,
                        relay_hops=item.relay_depth,
                        relay_reasons=item.relay_reasons,
                    )
                )
                evidence = self._actuator_evidence.get(
                    item.actuator_signature
                )
                if evidence is not None:
                    evidence.terminal_outcomes += 1
                self._delayed_terminal_credits += 1
                self._delayed_credit_delay_actions += delay
                self._delayed_credit_max_delay = max(
                    self._delayed_credit_max_delay,
                    delay,
                )
            discarded = [
                item.eligibility_id
                for item in self._delayed_eligibilities
                if item.eligibility_id not in selected_ids
            ]
            if credited:
                self._delayed_terminal_events += 1
            self._discarded_delayed_eligibilities += len(discarded)
            self._delayed_eligibilities = []
        elif game_over:
            discarded = [
                item.eligibility_id
                for item in self._delayed_eligibilities
            ]
            self._unsafe_delayed_eligibilities += len(discarded)
            self._discarded_delayed_eligibilities += len(discarded)
            self._delayed_eligibilities = []
        else:
            retained = []
            for item in self._delayed_eligibilities:
                age = (
                    self._branch_transition_index
                    - item.transition_index
                )
                if age > self.delayed_terminal_credit_window:
                    expired.append(item.eligibility_id)
                else:
                    retained.append(item)
            self._delayed_eligibilities = retained
            self._expired_delayed_eligibilities += len(expired)
        if terminal_success:
            self._branch_terminal_progress = True
            self._terminal_progress_observed = True
            if self._demoted_context_actuators:
                self._context_actuator_reactivations += len(
                    self._demoted_context_actuators
                )
                self._demoted_context_actuators.clear()
                self._context_actuator_nonprogress.clear()
            self._pending = None
            self._clear_sequence()
        return FrontierDelayedCreditUpdate(
            credited=tuple(credited),
            relayed_eligibility_ids=tuple(relayed),
            expired_eligibility_ids=tuple(expired),
            discarded_eligibility_ids=tuple(discarded),
        )

    @property
    def active_sequence(self) -> bool:
        return bool(self._active_sequence_remaining > 0)

    def cancel_pending(self) -> None:
        """Cancel a selection vetoed before execution without fabricating data."""
        pending = self._pending
        if pending is None:
            return
        key = (pending.state_signature, pending.actuator_signature)
        self._state_action_trials[key] = max(
            0,
            self._state_action_trials[key] - 1,
        )
        self._state_experiments[pending.state_signature] = max(
            0,
            self._state_experiments[pending.state_signature] - 1,
        )
        self._experiments = max(0, self._experiments - 1)
        self._sequence_actions = max(0, self._sequence_actions - 1)
        if pending.state_action_untested:
            self._untested_state_actions = max(
                0,
                self._untested_state_actions - 1,
            )
        if pending.actuator_untested:
            self._untested_actuator_actions = max(
                0,
                self._untested_actuator_actions - 1,
            )
        if pending.object_role_untested:
            self._untested_object_actions = max(
                0,
                self._untested_object_actions - 1,
            )
        self._pending = None
        self._clear_sequence()

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "minimum_stagnant_steps": self.minimum_stagnant_steps,
            "max_experiments_per_state": self.max_experiments_per_state,
            "max_sequence_actions": self.max_sequence_actions,
            "max_trials_per_actuator": self.max_trials_per_actuator,
            "minimum_failed_branches": self.minimum_failed_branches,
            "delayed_terminal_credit_enabled": (
                self.enable_delayed_terminal_credit
            ),
            "delayed_terminal_credit_window": (
                self.delayed_terminal_credit_window
            ),
            "max_delayed_credits_per_terminal": (
                self.max_delayed_credits_per_terminal
            ),
            "subeffect_eligibility_relay_enabled": (
                self.enable_subeffect_eligibility_relay
            ),
            "max_subeffect_relay_depth": self.max_subeffect_relay_depth,
            "generalized_stall_detection_enabled": (
                self.enable_generalized_stall_detection
            ),
            "effect_novelty_stall_actions": (
                self.effect_novelty_stall_actions
            ),
            "zero_terminal_branch_stall_count": (
                self.zero_terminal_branch_stall_count
            ),
            "per_level_rearming_enabled": (
                self.enable_per_level_rearming
            ),
            "nonprogress_demotion_threshold": (
                self.nonprogress_demotion_threshold
            ),
            "branches_started": self._branches_started,
            "failed_branches": self._failed_branches,
            "terminal_progress_observed": (
                self._terminal_progress_observed
            ),
            "states_assessed": self._states_assessed,
            "stagnation_detections": self._stagnation_detections,
            "frontier_entries": self._frontier_entries,
            "frontier_states": len(self._frontier_states),
            "experiments": self._experiments,
            "sequence_actions": self._sequence_actions,
            "multi_step_sequences": self._multi_step_sequences,
            "untested_state_actions": self._untested_state_actions,
            "untested_actuator_actions": self._untested_actuator_actions,
            "untested_object_actions": self._untested_object_actions,
            "productive_experiments": self._productive_experiments,
            "noop_experiments": self._noop_experiments,
            "unsafe_experiments": self._unsafe_experiments,
            "novel_effects": self._novel_effects,
            "novel_states": self._novel_states,
            "terminal_credits": self._terminal_credits,
            "delayed_eligibilities_registered": (
                self._delayed_eligibilities_registered
            ),
            "delayed_eligibilities_pending": len(
                self._delayed_eligibilities
            ),
            "delayed_terminal_events": self._delayed_terminal_events,
            "delayed_terminal_credits": self._delayed_terminal_credits,
            "delayed_credit_delay_actions": (
                self._delayed_credit_delay_actions
            ),
            "delayed_credit_max_delay": (
                self._delayed_credit_max_delay
            ),
            "expired_delayed_eligibilities": (
                self._expired_delayed_eligibilities
            ),
            "discarded_delayed_eligibilities": (
                self._discarded_delayed_eligibilities
            ),
            "censored_delayed_eligibilities": (
                self._censored_delayed_eligibilities
            ),
            "unsafe_delayed_eligibilities": (
                self._unsafe_delayed_eligibilities
            ),
            "subeffect_relays_created": self._subeffect_relays_created,
            "subeffect_relay_depth_histogram": {
                str(depth): count
                for depth, count in sorted(
                    self._subeffect_relay_depth_histogram.items()
                )
            },
            "subeffect_relay_reason_counts": dict(
                self._subeffect_relay_reason_counts
            ),
            "effect_novelty_stalls": self._effect_novelty_stalls,
            "actuator_coverage_stalls": self._actuator_coverage_stalls,
            "zero_terminal_branch_stalls": (
                self._zero_terminal_branch_stalls
            ),
            "actions_since_novel_effect": (
                self._actions_since_novel_effect
            ),
            "level_changes_observed": self._level_changes_observed,
            "level_rearm_pending": self._level_rearm_pending,
            "per_level_rearms": self._per_level_rearms,
            "protected_competence_blocks": (
                self._protected_competence_blocks
            ),
            "nonprogress_outcomes": self._nonprogress_outcomes,
            "context_actuator_demotions": (
                self._context_actuator_demotions
            ),
            "context_actuator_demotion_blocks": (
                self._context_actuator_demotion_blocks
            ),
            "context_actuator_reactivations": (
                self._context_actuator_reactivations
            ),
            "demoted_context_actuators": len(
                self._demoted_context_actuators
            ),
            "last_stall_reasons": list(self._last_stall_reasons),
            "information_gain": round(self._information_gain, 4),
            "actuator_models": len(self._actuator_evidence),
            "tested_object_roles": len(self._tested_target_roles),
            "active_sequence_id": self._active_sequence_id,
            "actuators": {
                actuator: {
                    "trials": evidence.trials,
                    "noops": evidence.noops,
                    "unsafe_outcomes": evidence.unsafe_outcomes,
                    "terminal_outcomes": evidence.terminal_outcomes,
                    "effect_signatures": dict(evidence.effect_signatures),
                }
                for actuator, evidence in self._actuator_evidence.items()
            },
        }

    def _stagnation_assessment(
        self,
        state_signature: str,
        diagnostics: Mapping[str, Any],
        *,
        untested_actuator_available: bool,
        prospective_state_visits: int | None = None,
    ) -> Tuple[bool, Tuple[str, ...]]:
        branch_actions = int(diagnostics.get("branch_actions", 0) or 0)
        if branch_actions < self.minimum_stagnant_steps:
            return False, ()
        terminal_stall = int(
            diagnostics.get("actions_since_terminal_improvement", 0)
            or 0
        )
        max_hash_repeat = int(
            diagnostics.get("max_hash_repeat", 0) or 0
        )
        unique_states = int(
            diagnostics.get("unique_states_in_window", 0) or 0
        )
        window_actions = int(
            diagnostics.get("window_actions", branch_actions) or 0
        )
        repeated_state = bool(
            max_hash_repeat >= max(
                3,
                self.minimum_stagnant_steps // 2,
            )
        )
        low_novelty_cycle = bool(
            window_actions >= self.minimum_stagnant_steps * 2
            and unique_states
            <= max(2, int(window_actions * 0.35))
        )
        recurrent_current_state = bool(
            (
                self._state_visits[state_signature]
                if prospective_state_visits is None
                else prospective_state_visits
            )
            >= 3
            and max_hash_repeat >= 2
        )
        reasons = []
        if (
            terminal_stall >= self.minimum_stagnant_steps
            and (repeated_state or low_novelty_cycle or recurrent_current_state)
        ):
            reasons.append("state_recurrence_stall")
        if self.enable_generalized_stall_detection:
            if (
                terminal_stall >= self.minimum_stagnant_steps
                and self._actions_since_novel_effect
                >= self.effect_novelty_stall_actions
            ):
                reasons.append("effect_novelty_stall")
            if (
                terminal_stall >= self.minimum_stagnant_steps
                and untested_actuator_available
            ):
                reasons.append("actuator_coverage_stall")
            if (
                terminal_stall >= self.minimum_stagnant_steps
                and self._failed_branches
                >= max(
                    self.minimum_failed_branches,
                    self.zero_terminal_branch_stall_count,
                )
            ):
                reasons.append("zero_terminal_branch_stall")
        return bool(reasons), tuple(reasons)

    @staticmethod
    def _relay_reason(
        eligibility: _FrontierEligibility,
        *,
        effect_signature: str,
        causal_effect_signature: str,
        target_role_signature: str,
        component_signatures: Tuple[str, ...],
        causally_linked_effect_signatures: set[str],
    ) -> str:
        if (
            target_role_signature
            and eligibility.current_target_role_signature
            and target_role_signature
            == eligibility.current_target_role_signature
        ):
            return "same_object_role"
        if set(component_signatures).intersection(
            eligibility.component_signatures
        ):
            return "same_component"
        eligibility_causal_signatures = {
            eligibility.causal_effect_signature,
            eligibility.current_causal_effect_signature,
        }
        eligibility_causal_signatures.discard("")
        if eligibility_causal_signatures.intersection(
            causally_linked_effect_signatures
        ):
            return "causal_subgoal_graph"
        if (
            causal_effect_signature
            and causal_effect_signature
            in eligibility_causal_signatures
            and effect_signature != eligibility.current_effect_signature
        ):
            return "same_causal_effect_class"
        return ""

    def _clear_sequence(self) -> None:
        self._active_sequence_id = ""
        self._active_sequence_step = 0
        self._active_sequence_remaining = 0

    @staticmethod
    def _eligibility_rank(
        eligibility: _FrontierEligibility,
    ) -> Tuple[int, int, float, int, str]:
        return (
            int(eligibility.novel_state),
            int(eligibility.novel_effect),
            float(eligibility.information_gain),
            int(eligibility.transition_index),
            eligibility.eligibility_id,
        )


def _concrete_candidates(
    grid: np.ndarray,
    available_actions: Sequence[str],
    raw_candidates: Sequence[Any] | None,
) -> Tuple[Tuple[str, Dict[str, Any], str, str], ...]:
    allowed = {
        str(action)
        for action in available_actions
        if str(action) and str(action) != "RESET"
    }
    concrete: list[Tuple[str, Dict[str, Any]]] = []
    for raw in tuple(raw_candidates or ()):
        name = str(getattr(raw, "name", ""))
        if name not in allowed:
            continue
        data = dict(getattr(raw, "action_args", {}) or {})
        concrete.append((name, data))
    represented = {name for name, _ in concrete}
    for name in sorted(allowed - represented):
        concrete.append((name, {}))

    result = []
    seen = set()
    for name, data in concrete:
        normalized = _normalized_action_data(data)
        identity = (name, normalized)
        if identity in seen:
            continue
        seen.add(identity)
        target_role = _target_role_signature(grid, data)
        actuator = _actuator_signature(name, data, target_role)
        result.append((name, dict(data), actuator, target_role))
    return tuple(result)


def _normalized_action_data(data: Mapping[str, Any]) -> ActionData:
    normalized = []
    for key, value in sorted(data.items(), key=lambda item: str(item[0])):
        if isinstance(value, (int, float, str, bool)) or value is None:
            stable = value
        else:
            stable = repr(value)
        normalized.append((str(key), stable))
    return tuple(normalized)


def _actuator_signature(
    action_name: str,
    action_data: Mapping[str, Any],
    target_role: str,
) -> str:
    argument_schema = tuple(sorted(str(key) for key in action_data))
    payload = (str(action_name), argument_schema, target_role)
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _target_role_signature(
    grid: np.ndarray,
    action_data: Mapping[str, Any],
) -> str:
    if "x" not in action_data or "y" not in action_data:
        return ""
    try:
        x = int(action_data["x"])
        y = int(action_data["y"])
    except (TypeError, ValueError):
        return ""
    height, width = grid.shape
    if not (0 <= x < width and 0 <= y < height):
        return "outside_grid"
    background = _background_value(grid)
    value = int(grid[y, x])
    occupancy = "background" if value == background else "object"
    component = _component(grid, x, y, value)
    xs = [coordinate[0] for coordinate in component]
    ys = [coordinate[1] for coordinate in component]
    component_width = max(xs) - min(xs) + 1
    component_height = max(ys) - min(ys) + 1
    area = len(component)
    area_bucket = (
        "single"
        if area == 1
        else "small"
        if area <= 4
        else "medium"
        if area <= 15
        else "large"
    )
    position = (
        min(2, (3 * x) // max(1, width)),
        min(2, (3 * y) // max(1, height)),
    )
    payload = (
        occupancy,
        area_bucket,
        min(component_width, 5),
        min(component_height, 5),
        position,
    )
    return repr(payload)


def _component(
    grid: np.ndarray,
    start_x: int,
    start_y: int,
    value: int,
) -> Tuple[Tuple[int, int], ...]:
    height, width = grid.shape
    pending = [(start_x, start_y)]
    seen = {(start_x, start_y)}
    while pending:
        x, y = pending.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in seen or int(grid[ny, nx]) != value:
                continue
            seen.add((nx, ny))
            pending.append((nx, ny))
    return tuple(sorted(seen))


def _background_value(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _state_signature(grid: np.ndarray) -> str:
    payload = (
        tuple(int(value) for value in grid.shape),
        grid.astype(np.int32, copy=False).tobytes(),
    )
    return hashlib.sha1(repr(payload).encode("latin1")).hexdigest()[:16]


def _context_signature(grid: np.ndarray) -> str:
    """Return a palette/position-light local context for demotion scope."""
    background = _background_value(grid)
    visited: set[Tuple[int, int]] = set()
    components = []
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if (x, y) in visited or int(grid[y, x]) == background:
                continue
            value = int(grid[y, x])
            component = _component(grid, x, y, value)
            visited.update(component)
            xs = [coordinate[0] for coordinate in component]
            ys = [coordinate[1] for coordinate in component]
            area = len(component)
            components.append((
                (
                    "single"
                    if area == 1
                    else "small"
                    if area <= 4
                    else "medium"
                    if area <= 15
                    else "large"
                ),
                min(5, max(xs) - min(xs) + 1),
                min(5, max(ys) - min(ys) + 1),
            ))
    payload = (
        tuple(int(value) for value in grid.shape),
        tuple(sorted(components)),
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _effect_signature(before: np.ndarray, after: np.ndarray) -> str:
    if before.shape != after.shape:
        payload = ("shape_change", before.shape, after.shape)
    else:
        changed = np.argwhere(before != after)
        if not len(changed):
            payload = ("noop",)
        else:
            y_min, x_min = changed.min(axis=0)
            y_max, x_max = changed.max(axis=0)
            changed_count = len(changed)
            count_bucket = (
                "one"
                if changed_count == 1
                else "few"
                if changed_count <= 4
                else "many"
            )
            before_values = len(set(int(v) for v in before[before != after]))
            after_values = len(set(int(v) for v in after[before != after]))
            payload = (
                "change",
                count_bucket,
                int(x_max - x_min + 1),
                int(y_max - y_min + 1),
                before_values,
                after_values,
            )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _noop_effect_signature() -> str:
    return hashlib.sha1(repr(("noop",)).encode("utf-8")).hexdigest()[:16]


def _effect_component_signatures(
    before: np.ndarray,
    after: np.ndarray,
) -> Tuple[str, ...]:
    """Describe changed components without coordinates or palette identity."""
    if before.shape != after.shape or before.ndim != 2:
        return ("shape_change",)
    changed = np.argwhere(before != after)
    if not len(changed):
        return ()
    signatures = set()
    for grid in (before, after):
        background = _background_value(grid)
        visited: set[Tuple[int, int]] = set()
        for y_raw, x_raw in changed:
            x = int(x_raw)
            y = int(y_raw)
            if (x, y) in visited:
                continue
            value = int(grid[y, x])
            component = _component(grid, x, y, value)
            visited.update(component)
            xs = [coordinate[0] for coordinate in component]
            ys = [coordinate[1] for coordinate in component]
            normalized = tuple(sorted(
                (cx - min(xs), cy - min(ys))
                for cx, cy in component
            ))
            payload = (
                "background" if value == background else "object",
                len(component),
                max(xs) - min(xs) + 1,
                max(ys) - min(ys) + 1,
                normalized,
            )
            signatures.add(
                hashlib.sha1(
                    repr(payload).encode("utf-8")
                ).hexdigest()[:16]
            )
    return tuple(sorted(signatures))


__all__ = [
    "DelayedFrontierCredit",
    "FrontierDelayedCreditUpdate",
    "FrontierEligibilityAssessment",
    "FrontierExperimentSelection",
    "OnlineFrontierExplorer",
]
