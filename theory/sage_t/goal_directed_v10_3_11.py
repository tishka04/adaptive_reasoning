"""Bounded-posterior, goal-conditioned controller for SAGE.T10.3.11.

T10.3.10 bounded the live belief profiler but not the executable-program
posterior owned by :class:`SageTController`.  The latter retained every
transition, replayed them when adding programs, and allowed local mutation on
each surprising observation.  The scheduled unified shell also passed an
empty goal bank to SAGE.T.  This continuation repairs those two integration
defects without changing any frozen predecessor.

Only level increments are success.  Goal-distance reductions are ephemeral
planning evidence and never promote a transferable program on their own.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..live_transition_loop import build_observation
from ..unified_cognitive_controller import _normalize_actions
from .contracts import AbstractState, ActionCandidate, ObservedTransition
from .goal_directed_v10_3_2 import GoalDirectedOption, OptionStep
from .goal_directed_v10_3_5 import scheduled_unified_config
from .goal_directed_v10_3_10 import (
    DirectionalProgressSageTController,
    DirectionalProgressUnifiedCognitiveController,
)
from .posterior import ProgramPosterior

FORMAT_VERSION = "sage-t10.3.11-bounded-goal-conditioned-v1"
POSTERIOR_HISTORY_LIMIT = 16
CONTROLLER_TRANSITION_LIMIT = 16
GOAL_OPTION_HORIZON = 12
GOAL_CONDITIONED_SOURCE = "bounded_terminal_objective_frontier"


def _digest_message(exc: BaseException) -> str:
    """Persist a safe fingerprint, never an environment payload."""

    message = f"{type(exc).__name__}:{str(exc)[:160]}"
    return hashlib.sha256(message.encode("utf-8", errors="replace")).hexdigest()


class BoundedProgramPosterior(ProgramPosterior):
    """Incremental posterior with a strict evidence window and no live repair.

    Program mutation remains available to explicit offline callers through the
    inherited ``repair`` method, but physical observations never trigger it.
    This makes per-event posterior work depend on the frozen particle cap, not
    on the number of actions already taken in the reset.
    """

    def __init__(self, *args: Any, history_limit: int = POSTERIOR_HISTORY_LIMIT, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history_limit = max(1, int(history_limit))
        self.observations = 0
        self.maximum_history = 0
        self.live_repairs_suppressed = 0

    def observe(
        self,
        evidence: ObservedTransition,
        *,
        allow_repair: bool = True,
    ) -> None:
        if allow_repair:
            self.live_repairs_suppressed += 1
        super().observe(evidence, allow_repair=False)
        if not evidence.reset:
            self.observations += 1
        if len(self._history) > self.history_limit:
            del self._history[:-self.history_limit]
        self.maximum_history = max(self.maximum_history, len(self._history))

    def bounded_summary(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "history": len(self._history),
            "maximum_history": self.maximum_history,
            "history_limit": self.history_limit,
            "live_repairs_suppressed": self.live_repairs_suppressed,
            "repairs_attempted": self._repairs_attempted,
            "repairs_admitted": self._repairs_admitted,
        }


class GoalConditionedSageTController(DirectionalProgressSageTController):
    """Choose reset-local automata for measurable terminal objectives."""

    def __init__(
        self,
        *args: Any,
        goal_conditioning_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.posterior = BoundedProgramPosterior(
            executor=self.executor,
            maximum_particles=self.config.maximum_programs,
            history_limit=POSTERIOR_HISTORY_LIMIT,
        )
        self.goal_conditioning_enabled = bool(goal_conditioning_enabled)
        self._live_goal_hypotheses: tuple[Any, ...] = ()
        self._goal_hypotheses_received = 0
        self._goal_conditioned_options = 0
        self._goal_conditioned_actions = 0
        self._goal_action_gains: Counter[str] = Counter()
        self._goal_action_trials: Counter[str] = Counter()
        self._active_goal_objective_ids: tuple[str, ...] = ()
        self._last_goal_objective_ids: tuple[str, ...] = ()
        self._posterior_observation_rejections = 0
        self._observation_errors: Counter[str] = Counter()
        self._last_observation_error_digest = ""
        self._program_reassemblies = 0
        self._maximum_controller_transitions = 0

    def set_live_goal_hypotheses(self, hypotheses: Sequence[Any]) -> None:
        self._live_goal_hypotheses = tuple(hypotheses) if self.goal_conditioning_enabled else ()
        self._goal_hypotheses_received += len(self._live_goal_hypotheses)

    @property
    def last_goal_objective_ids(self) -> tuple[str, ...]:
        return self._last_goal_objective_ids

    def decide(self, *args: Any, goal_hypotheses: Sequence[Any] = (), **kwargs: Any):
        hypotheses = tuple(goal_hypotheses) or self._live_goal_hypotheses
        before_names = self._available_action_names
        before_needs = self._needs_reassembly
        arbitration = super().decide(
            *args,
            goal_hypotheses=hypotheses,
            **kwargs,
        )
        if before_needs or before_names != self._available_action_names:
            self._program_reassemblies += 1
        self._note_goal_action_if_selected(arbitration.applied)
        return arbitration

    def fast_active_decision(self, *args: Any, **kwargs: Any):
        selected = super().fast_active_decision(*args, **kwargs)
        self._note_goal_action_if_selected(selected is not None)
        return selected

    def _note_goal_action_if_selected(self, selected: bool) -> None:
        active = self._pending_option
        if selected and active is not None and active.source == GOAL_CONDITIONED_SOURCE:
            self._goal_conditioned_actions += 1
            self._last_goal_objective_ids = self._active_goal_objective_ids
        else:
            self._last_goal_objective_ids = ()

    def _choose_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any] = (),
    ) -> GoalDirectedOption | None:
        if self.phase == "confirmation" or self.reproduce_mixed_registry:
            return super()._choose_option(
                state,
                candidates,
                goal_hypotheses=goal_hypotheses,
            )
        if not self.goal_conditioning_enabled:
            return super()._choose_option(
                state,
                candidates,
                goal_hypotheses=(),
            )

        legal = tuple(sorted({candidate.action_name for candidate in candidates}))
        votes: Counter[str] = Counter()
        objective_ids: list[str] = []
        for hypothesis in goal_hypotheses:
            supported = {
                str(action).strip().upper()
                for action in getattr(hypothesis, "supporting_actions", ())
            }
            participating = sorted(supported.intersection(legal))
            if not participating:
                continue
            objective_id = str(getattr(hypothesis, "objective_id", ""))
            if objective_id:
                objective_ids.append(objective_id)
            priority = max(1, int(round(10.0 * float(getattr(hypothesis, "prior_priority", 0.0) or 0.0))))
            for action in participating:
                votes[action] += priority
        ranked = sorted(
            votes,
            key=lambda action: (
                -self._goal_action_gains[action],
                -votes[action],
                self._goal_action_trials[action],
                action,
            ),
        )
        if not ranked:
            return super()._choose_option(
                state,
                candidates,
                goal_hypotheses=goal_hypotheses,
            )
        if len(ranked) == 1:
            complements = [action for action in legal if action != ranked[0]]
            if complements:
                ranked.append(min(complements, key=lambda action: (self._goal_action_trials[action], action)))

        steps: list[OptionStep] = []
        for index in range(GOAL_OPTION_HORIZON):
            action = ranked[index % len(ranked)]
            method = self._binding_method(action, candidates)
            if method is None:
                continue
            steps.append(
                OptionStep(
                    action,
                    binding_method=method,
                    expected_effect="terminal_objective_distance_reduction",
                )
            )
            self._goal_action_trials[action] += 1
        if len(steps) < 2 or len({step.action_name for step in steps}) < 2:
            return super()._choose_option(
                state,
                candidates,
                goal_hypotheses=goal_hypotheses,
            )
        self._active_goal_objective_ids = tuple(sorted(set(objective_ids)))
        self._goal_conditioned_options += 1
        return GoalDirectedOption(
            schema="mixed_automaton",
            steps=tuple(steps),
            source=GOAL_CONDITIONED_SOURCE,
        )

    def note_objective_outcome(self, action_name: str, outcome: Mapping[str, Any]) -> None:
        if outcome.get("all_reduced_objectives"):
            self._goal_action_gains[str(action_name).strip().upper()] += 1

    def observe_transition(self, record: Any) -> None:
        before = self.posterior.observations
        try:
            super().observe_transition(record)
        except Exception as exc:
            self._observation_errors[type(exc).__name__] += 1
            self._last_observation_error_digest = _digest_message(exc)
            raise
        finally:
            if len(self._transitions) > CONTROLLER_TRANSITION_LIMIT:
                del self._transitions[:-CONTROLLER_TRANSITION_LIMIT]
            self._maximum_controller_transitions = max(
                self._maximum_controller_transitions,
                len(self._transitions),
            )
        if self.posterior.observations == before:
            self._posterior_observation_rejections += 1
        # Reassembly is unnecessary after an ordinary observation.  The base
        # decision path still notices a changed legal-action signature.
        self._needs_reassembly = False

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self._live_goal_hypotheses = ()
        self._active_goal_objective_ids = ()
        self._last_goal_objective_ids = ()

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "goal_conditioning_enabled": self.goal_conditioning_enabled,
                "goal_hypotheses_received": self._goal_hypotheses_received,
                "goal_conditioned_options": self._goal_conditioned_options,
                "goal_conditioned_actions": self._goal_conditioned_actions,
                "goal_action_gain_events": sum(self._goal_action_gains.values()),
                "posterior_observation_rejections": self._posterior_observation_rejections,
                "observation_errors": dict(self._observation_errors),
                "last_observation_error_digest": self._last_observation_error_digest,
                "program_reassemblies": self._program_reassemblies,
                "maximum_controller_transitions": self._maximum_controller_transitions,
                "bounded_program_posterior": self.posterior.bounded_summary(),
                "objective_ids_persisted": False,
            }
        )
        return base


class GoalConditionedUnifiedCognitiveController(
    DirectionalProgressUnifiedCognitiveController
):
    """Scheduled unified shell that actually supplies and revises goals."""

    def __init__(self, *args: Any, goal_conditioning_enabled: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.goal_conditioning_enabled = bool(goal_conditioning_enabled)
        self._goal_generation_calls = 0
        self._maximum_live_objectives = 0
        self._objective_observations = 0

    def select_action(self, **kwargs: Any):
        actions = _normalize_actions(kwargs.get("available_actions", ()))
        self.theory.seed_actions(actions)
        observation = build_observation(
            kwargs["current_grid"],
            available_actions=actions,
            game_state=str(kwargs.get("game_state", "NOT_FINISHED")),
            levels_completed=int(kwargs.get("levels_completed", 0)),
            infer_players=True,
        )
        safe_actions = self._safe_actions(observation.grid_hash, actions) or list(actions)
        if self.goal_conditioning_enabled:
            self._generate_goal_hypotheses(observation, safe_actions)
            self._goal_generation_calls += 1
        objectives = tuple(self.terminal_objectives.objectives()) if self.goal_conditioning_enabled else ()
        self._maximum_live_objectives = max(self._maximum_live_objectives, len(objectives))
        goal = self.sage_t_controller
        if isinstance(goal, GoalConditionedSageTController):
            goal.set_live_goal_hypotheses(objectives)
        decision = super().select_action(**kwargs)
        objective_ids = () if not isinstance(goal, GoalConditionedSageTController) else goal.last_goal_objective_ids
        if objective_ids and decision.source == "sage_t_joint_program":
            decision = replace(
                decision,
                objective_id=objective_ids[0],
                predicted_goal_reductions=objective_ids,
            )
            self._pending_decision = decision
        return decision

    def observe_transition(self, **kwargs: Any):
        pending = self._pending_decision
        update = super().observe_transition(**kwargs)
        if self.goal_conditioning_enabled:
            outcome = self._observe_pending_terminal_objective(update, pending)
            self._objective_observations += 1
            goal = self.sage_t_controller
            if isinstance(goal, GoalConditionedSageTController):
                goal.note_objective_outcome(str(kwargs.get("action", "")), outcome)
        return update

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        objective_summary = self.terminal_objectives.summary()
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "goal_conditioning_enabled": self.goal_conditioning_enabled,
                "goal_generation_calls": self._goal_generation_calls,
                "maximum_live_objectives": self._maximum_live_objectives,
                "objective_observations": self._objective_observations,
                "objective_distance_reductions": int(objective_summary.get("distance_reductions", 0)),
                "level_progress_is_only_success_credit": True,
                "goal_payloads_persisted": False,
            }
        )
        return base


def goal_conditioned_unified_config(*, sage_t_authority_mode: str):
    """Keep the predecessor's symmetric bounded unified configuration."""

    return scheduled_unified_config(sage_t_authority_mode=sage_t_authority_mode)


__all__ = [
    "CONTROLLER_TRANSITION_LIMIT",
    "FORMAT_VERSION",
    "GOAL_CONDITIONED_SOURCE",
    "GOAL_OPTION_HORIZON",
    "POSTERIOR_HISTORY_LIMIT",
    "BoundedProgramPosterior",
    "GoalConditionedSageTController",
    "GoalConditionedUnifiedCognitiveController",
    "goal_conditioned_unified_config",
]
