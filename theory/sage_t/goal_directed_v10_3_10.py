"""Directional, cycle-resistant source controller for SAGE.T10.3.10.

T10.3.9 was durable but its first real sequence reset exposed a precise
planning defect: the causal-frontier planner selected an action from frozen
counts and then reused that same selection for an entire 16--32 step macro.
A changing timer-like frame could therefore look novel forever even though no
goal-directed progress was being made.  The same run also showed that online
relational-rule verification, which is not needed by the SAGE.T option
posterior, dominated observation time late in a reset.

This continuation keeps the real unified-controller boundary and updates the
same mechanic theory and SAGE.T posterior after every physical transition.  It
changes only the experimental source policy:

* structural effects receive exploration credit once per causal context, not
  once per changing raw frame;
* repeated action/effect contexts abort an exploratory option;
* frontier planning updates simulated usage counts at every planned step and
  caps identical action runs at two;
* online transition history is bounded to eight records and relational-rule
  verification is deferred (mechanic and SAGE.T posterior updates remain
  immediate).

Level increments remain the only success credit.  Structural gains only order
experiments and can never promote a program.  Persisted options contain no
coordinates, colors, entity identities, raw grids, game names, or seeds.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from typing import Any

from v3.schemas import TransitionRecord

from .contracts import AbstractState, ActionCandidate
from .goal_directed_v10_3_2 import GoalDirectedOption, OptionStep
from .goal_directed_v10_3_9 import (
    CAUSAL_PROBE_SOURCE,
    REPRODUCTION_SOURCE,
    CausalSubgoalAutomatonInducer,
    CausalSubgoalSageTController,
    robust_effect_descriptor,
)
from .goal_directed_v10_3_5 import ScheduledUnifiedCognitiveController

FORMAT_VERSION = "sage-t10.3.10-directional-progress-v1"
DIRECTIONAL_FRONTIER_SOURCE = "reset_local_directional_subgoal_frontier"
MAXIMUM_DIRECTIONAL_OPTION_HORIZON = 6
MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN = 2
REPEATED_EFFECT_STALL_LIMIT = 3
BOUNDED_TRANSITION_HISTORY = 8
DEFERRED_RULE_VERIFICATION_INTERVAL = 1_000_000


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sign(value: int) -> int:
    return 0 if value == 0 else 1 if value > 0 else -1


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except (TypeError, ValueError):
        return 0


def directional_milestone_descriptor(record: Any) -> dict[str, Any]:
    """Return a coarse, identity-free candidate-subgoal descriptor.

    The descriptor deliberately drops raw frame hashes and exact cell values.
    Thus a countdown or animation that changes pixels while preserving the
    same structural relation is observed repeatedly rather than rewarded as a
    fresh milestone on every frame.
    """

    effect = robust_effect_descriptor(record)
    before = record.obs_before
    after = record.obs_after
    before_objects = _safe_len(getattr(before, "objects", ()))
    after_objects = _safe_len(getattr(after, "objects", ()))
    before_actions = _safe_len(getattr(before, "available_actions", ()))
    after_actions = _safe_len(getattr(after, "available_actions", ()))
    before_grid = getattr(before, "raw_grid", None)
    after_grid = getattr(after, "raw_grid", None)
    before_shape = tuple(getattr(before_grid, "shape", ()))
    after_shape = tuple(getattr(after_grid, "shape", ()))
    return {
        "mode": str(effect["mode"]),
        "level_progress": bool(effect["level_progress"]),
        "terminal": bool(effect["terminal"]),
        "noop": bool(effect["noop"]),
        "actor_axis": str(effect["actor_axis"]),
        "component_delta_sign": _sign(after_objects - before_objects),
        "action_space_delta_sign": _sign(after_actions - before_actions),
        "shape_changed": before_shape != after_shape,
        "object_set_changed": bool(
            effect["created_bucket"] or effect["removed_bucket"]
        ),
        "actor_or_object_moved": bool(effect["moved_bucket"])
        or str(effect["actor_axis"]) not in {"none", "unknown"},
    }


def directional_milestone_signature(record: Any) -> str:
    return _sha(directional_milestone_descriptor(record))


class DirectionalProgressAutomatonInducer(CausalSubgoalAutomatonInducer):
    """Compose short frontiers and distinguish gain from visual activity."""

    def __init__(self) -> None:
        super().__init__()
        self._causal_contexts: set[tuple[str, str, str]] = set()
        self._action_gains: Counter[str] = Counter()
        self._action_stalls: Counter[str] = Counter()
        self._milestone_visits: Counter[str] = Counter()
        self._recent_pairs: deque[tuple[str, str]] = deque(maxlen=8)
        self._last_milestone = "branch_start"
        self._last_pair: tuple[str, str] | None = None
        self._identical_pair_streak = 0
        self._last_transition_gain = False
        self._last_transition_stalled = False
        self._directional_gain_events = 0
        self._repeated_effect_events = 0
        self._maximum_identical_pair_streak = 0
        self._frontiers_composed = 0
        self._maximum_planned_identical_run = 0

    def start_branch(self) -> None:
        super().start_branch()
        self._recent_pairs.clear()
        self._last_milestone = "branch_start"
        self._last_pair = None
        self._identical_pair_streak = 0
        self._last_transition_gain = False
        self._last_transition_stalled = False

    @property
    def last_transition_gain(self) -> bool:
        return self._last_transition_gain

    @property
    def last_transition_stalled(self) -> bool:
        return self._last_transition_stalled

    def observe(
        self,
        record: TransitionRecord,
        *,
        selected_step: OptionStep | None,
        active_option: GoalDirectedOption | None,
    ) -> GoalDirectedOption | None:
        action_name = str(record.action.name).strip().upper()
        descriptor = directional_milestone_descriptor(record)
        milestone = _sha(descriptor)
        context = (self._last_milestone, action_name, milestone)
        context_is_new = context not in self._causal_contexts
        milestone_is_new = self._milestone_visits[milestone] == 0
        structural = bool(
            descriptor["object_set_changed"]
            or descriptor["actor_or_object_moved"]
            or descriptor["shape_changed"]
            or descriptor["component_delta_sign"]
            or descriptor["action_space_delta_sign"]
        )
        # A raw frame may keep changing while the same causal milestone repeats
        # (for example an animation or countdown).  Context novelty is useful
        # for the local effect graph, but only the first observation of the
        # coarse milestone receives exploration gain.
        gain = bool(descriptor["level_progress"] or (milestone_is_new and structural))

        pair = (action_name, milestone)
        if pair == self._last_pair:
            self._identical_pair_streak += 1
        else:
            self._identical_pair_streak = 1
        self._maximum_identical_pair_streak = max(
            self._maximum_identical_pair_streak,
            self._identical_pair_streak,
        )
        repeated = not context_is_new or self._milestone_visits[milestone] > 0
        stalled = bool(
            not descriptor["level_progress"]
            and (
                descriptor["terminal"]
                or descriptor["noop"]
                or (not gain and self._identical_pair_streak >= REPEATED_EFFECT_STALL_LIMIT)
            )
        )

        self._causal_contexts.add(context)
        self._milestone_visits[milestone] += 1
        self._recent_pairs.append(pair)
        self._last_pair = pair
        self._last_milestone = milestone
        self._last_transition_gain = gain
        self._last_transition_stalled = stalled
        if gain:
            self._action_gains[action_name] += 1
            self._directional_gain_events += 1
        if repeated:
            self._repeated_effect_events += 1
        if stalled:
            self._action_stalls[action_name] += 1

        return super().observe(
            record,
            selected_step=selected_step,
            active_option=active_option,
        )

    def compose_frontier(
        self,
        legal_action_names: Sequence[str],
        *,
        rotation: int,
        horizon: int = MAXIMUM_DIRECTIONAL_OPTION_HORIZON,
    ) -> GoalDirectedOption | None:
        legal = tuple(
            sorted(
                {
                    str(item).strip().upper()
                    for item in legal_action_names
                    if str(item).strip().upper().startswith("ACTION")
                }
            )
        )
        if len(legal) < 2:
            return None
        offset = int(rotation) % len(legal)
        rotated = legal[offset:] + legal[:offset]
        length = max(2, min(MAXIMUM_DIRECTIONAL_OPTION_HORIZON, int(horizon)))
        simulated_uses: Counter[str] = Counter()
        steps: list[OptionStep] = []
        last_action: str | None = None
        run = 0

        for index in range(length):
            ranked = sorted(
                rotated,
                key=lambda action: (
                    int(
                        action == last_action
                        and run >= MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN
                    ),
                    self._action_stalls[action],
                    -self._action_gains[action],
                    simulated_uses[action],
                    self._terminal_uses[action],
                    self._noop_uses[action],
                    self._action_uses[action],
                    (rotated.index(action) - index) % len(rotated),
                ),
            )
            action = ranked[0]
            if action == last_action:
                run += 1
            else:
                last_action = action
                run = 1
            self._maximum_planned_identical_run = max(
                self._maximum_planned_identical_run,
                run,
            )
            simulated_uses[action] += 1
            steps.append(
                OptionStep(
                    action,
                    binding_method="unique_action_schema",
                    expected_effect=self._predicted_effect(action),
                )
            )

        if len({step.action_name for step in steps}) < 2:
            return None
        self._frontiers_composed += 1
        return GoalDirectedOption(
            schema="mixed_automaton",
            steps=tuple(steps),
            source=DIRECTIONAL_FRONTIER_SOURCE,
        )

    def summary(self) -> dict[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "directional_gain_events": self._directional_gain_events,
                "repeated_effect_events": self._repeated_effect_events,
                "action_gains": dict(self._action_gains),
                "action_stalls": dict(self._action_stalls),
                "causal_context_count": len(self._causal_contexts),
                "maximum_identical_action_effect_streak": (
                    self._maximum_identical_pair_streak
                ),
                "frontiers_composed": self._frontiers_composed,
                "maximum_planned_identical_action_run": (
                    self._maximum_planned_identical_run
                ),
                "maximum_directional_option_horizon": (
                    MAXIMUM_DIRECTIONAL_OPTION_HORIZON
                ),
                "last_transition_gain": self._last_transition_gain,
                "last_transition_stalled": self._last_transition_stalled,
            }
        )
        return base


class DirectionalProgressSageTController(CausalSubgoalSageTController):
    """SAGE.T source authority that abandons repeated-effect experiments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.inducer = DirectionalProgressAutomatonInducer()
        self._effect_stall_aborts = 0
        self._directional_frontier_trials = 0

    def _choose_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any] = (),
    ) -> GoalDirectedOption | None:
        if self.phase == "confirmation":
            return super()._choose_option(
                state,
                candidates,
                goal_hypotheses=goal_hypotheses,
            )
        if self.reproduce_mixed_registry:
            reproduced = self._registry_mixed_option()
            if reproduced is not None:
                return reproduced
        probe = self._probe_option(candidates)
        if probe is not None:
            return probe
        names = self._rotated_names(candidates)
        option = self.inducer.compose_frontier(
            names,
            rotation=self._exploration_rotation + self._directional_frontier_trials,
            horizon=MAXIMUM_DIRECTIONAL_OPTION_HORIZON,
        )
        if option is None:
            return None
        rebound: list[OptionStep] = []
        for step in option.steps:
            method = self._binding_method(step.action_name, candidates)
            if method is None:
                return None
            rebound.append(
                OptionStep(
                    step.action_name,
                    binding_method=method,
                    expected_effect=step.expected_effect,
                )
            )
        self._directional_frontier_trials += 1
        return GoalDirectedOption(
            schema="mixed_automaton",
            steps=tuple(rebound),
            source=DIRECTIONAL_FRONTIER_SOURCE,
        )

    def observe_transition(self, record: TransitionRecord) -> None:
        active_before = self._pending_option
        super().observe_transition(record)
        exploratory = bool(
            active_before is not None
            and active_before.source
            in {DIRECTIONAL_FRONTIER_SOURCE, CAUSAL_PROBE_SOURCE}
        )
        if (
            self.phase != "confirmation"
            and exploratory
            and self.inducer.last_transition_stalled
            and self._active_option is not None
        ):
            self._effect_stall_aborts += 1
            self._finish_active_option(
                progressed=False,
                reason="repeated_action_effect_stall",
            )

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "directional_progress_exploration": True,
                "level_progress_is_only_success_credit": True,
                "effect_stall_aborts": self._effect_stall_aborts,
                "directional_frontier_trials": self._directional_frontier_trials,
                "maximum_directional_option_horizon": (
                    MAXIMUM_DIRECTIONAL_OPTION_HORIZON
                ),
                "maximum_planned_identical_action_run": (
                    MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN
                ),
                "raw_state_novelty_rewarded": False,
                "directional_inducer": self.inducer.summary(),
            }
        )
        return base


class DirectionalProgressUnifiedCognitiveController(
    ScheduledUnifiedCognitiveController
):
    """Unified shell with immediate posterior and deferred rule verification."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.belief_loop.verify_every = DEFERRED_RULE_VERIFICATION_INTERVAL
        self._directional_bounded_observations = 0
        self._directional_maximum_retained = 0

    def observe_transition(self, *args: Any, **kwargs: Any):
        transitions = self.belief_loop.profiler.transitions
        if len(transitions) >= BOUNDED_TRANSITION_HISTORY:
            del transitions[: len(transitions) - BOUNDED_TRANSITION_HISTORY + 1]
        update = super().observe_transition(*args, **kwargs)
        transitions = self.belief_loop.profiler.transitions
        if len(transitions) > BOUNDED_TRANSITION_HISTORY:
            del transitions[:-BOUNDED_TRANSITION_HISTORY]
        self._directional_maximum_retained = max(
            self._directional_maximum_retained,
            len(transitions),
        )
        self._directional_bounded_observations += 1
        return update

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "bounded_transition_history": BOUNDED_TRANSITION_HISTORY,
                "maximum_retained_transitions": self._directional_maximum_retained,
                "online_relational_rule_verification": False,
                "relational_rule_verification_deferred": True,
                "mechanic_theory_updated_each_transition": True,
                "sage_t_posterior_updated_each_transition": True,
                "directional_bounded_observations": (
                    self._directional_bounded_observations
                ),
            }
        )
        return base


__all__ = [
    "BOUNDED_TRANSITION_HISTORY",
    "DEFERRED_RULE_VERIFICATION_INTERVAL",
    "DIRECTIONAL_FRONTIER_SOURCE",
    "DirectionalProgressAutomatonInducer",
    "DirectionalProgressSageTController",
    "DirectionalProgressUnifiedCognitiveController",
    "FORMAT_VERSION",
    "MAXIMUM_DIRECTIONAL_OPTION_HORIZON",
    "MAXIMUM_PLANNED_IDENTICAL_ACTION_RUN",
    "REPEATED_EFFECT_STALL_LIMIT",
    "REPRODUCTION_SOURCE",
    "directional_milestone_descriptor",
    "directional_milestone_signature",
]
