"""Functional-first end-to-end controller for SAGE.T10.3.6.

T10.3.5 established that the scheduled integration shell can execute within
the decision budget, but it also exposed a more important semantic defect:
structurally equivalent parameterised actions collapsed to one binding and an
arbitrary visual change could keep an option alive.  This continuation makes
the functional question primary.  It balances ambiguous bindings across fresh
resets, re-plans structural successor paths after every action, and gives
credit only to an observed level increment.  Latency is telemetry only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from v3.schemas import TransitionRecord

from .contracts import AbstractState, ActionCandidate
from .goal_directed_v10_3_2 import (
    MAXIMUM_OPTION_HORIZON,
    GoalDirectedOption,
    OptionStep,
)
from .goal_directed_v10_3_3 import DYNAMIC_SUCCESSOR, _same_action_data
from .goal_directed_v10_3_5 import ScheduledGoalDirectedSageTController
from .progress_witness_v10 import GroundedAction, SearchConfig, chain_successor_macro

FORMAT_VERSION = "sage-t10.3.6-functional-end-to-end-v1"
BALANCED_CAUSAL_BINDING = "balanced_reset_local_causal_binding"
WITNESS_BINDING = "fresh_canonical_witness_binding"
MAXIMUM_REPEAT_HORIZON = 16


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _candidate_order_key(candidate: ActionCandidate) -> str:
    """Return an ephemeral order key; it is never included in a program."""

    payload = {
        "action_name": candidate.action_name,
        "action_data": dict(candidate.action_data),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class FunctionalGoalDirectedSageTController(ScheduledGoalDirectedSageTController):
    """Goal controller that tests causal bindings instead of visual activity.

    ``exploration_offset`` is a reset-local intervention schedule.  It chooses
    different members of an otherwise indistinguishable legal-action class on
    different fresh resets.  The offset and grounded action data never enter a
    persisted option, so a promoted program remains game/seed/coordinate free.
    """

    def __init__(
        self,
        *args: Any,
        exploration_offset: int = 0,
        witness_schema: str | None = None,
        witness_horizon: int | None = None,
        prefer_mixed: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("warmup_actions", 0)
        kwargs.setdefault("exploration_interval", 1)
        super().__init__(*args, **kwargs)
        self.exploration_offset = max(0, int(exploration_offset))
        self.witness_schema = None if witness_schema is None else str(witness_schema)
        self.witness_horizon = (
            None if witness_horizon is None else max(1, int(witness_horizon))
        )
        self.prefer_mixed = bool(prefer_mixed)
        self._seen_state_hashes: set[str] = set()
        self._causal_cycle_aborts = 0
        self._balanced_binding_uses = 0
        self._witness_binding_uses = 0
        self._functional_option_trials = 0
        self._functional_sources: Counter[str] = Counter()
        self._last_successor_action_data: dict[str, Any] | None = None
        self._successor_advances = 0

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self._seen_state_hashes.clear()
        self._last_successor_action_data = None
        # Make the first decision an experiment.  T10.3.6 is intentionally a
        # functional test, not another warm-up/latency experiment.
        self._exploration_since_option = self.exploration_interval

    @staticmethod
    def _dynamic_successor_option(
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        horizon: int | None = None,
        source: str,
    ) -> GoalDirectedOption | None:
        grounded = tuple(
            GroundedAction(
                candidate.action_name,
                tuple(dict(candidate.action_data).items()),
            )
            for candidate in candidates
        )
        chain = chain_successor_macro(
            state,
            grounded,
            config=SearchConfig(maximum_horizon=MAXIMUM_OPTION_HORIZON),
        )
        if chain is None or not chain.actions:
            return None
        length = len(chain.actions) if horizon is None else int(horizon)
        length = max(1, min(MAXIMUM_OPTION_HORIZON, length))
        action_name = chain.actions[0].action_name
        return GoalDirectedOption(
            schema="path_successor",
            steps=tuple(
                OptionStep(
                    action_name,
                    binding_method=DYNAMIC_SUCCESSOR,
                    expected_effect="noncyclic_structural_successor",
                )
                for _ in range(length)
            ),
            source=source,
        )

    @staticmethod
    def _balanced_repeat_option(
        candidates: Sequence[ActionCandidate],
        *,
        horizon: int,
        witness: bool,
        source: str,
    ) -> GoalDirectedOption | None:
        parameterized: dict[str, list[ActionCandidate]] = {}
        for candidate in candidates:
            if candidate.action_data:
                parameterized.setdefault(candidate.action_name, []).append(candidate)
        if not parameterized:
            return None
        # The choice of action schema is structural.  Selection of a concrete
        # member is deferred to the reset-local balanced intervention below.
        action_name = sorted(
            parameterized,
            key=lambda name: (-len(parameterized[name]), name),
        )[0]
        method = WITNESS_BINDING if witness else BALANCED_CAUSAL_BINDING
        return GoalDirectedOption(
            schema="repeat_target",
            steps=tuple(
                OptionStep(
                    action_name,
                    binding_method=method,
                    expected_effect="level_progress_only",
                )
                for _ in range(max(1, min(MAXIMUM_REPEAT_HORIZON, int(horizon))))
            ),
            source=source,
        )

    def _mixed_option(
        self,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any],
    ) -> GoalDirectedOption | None:
        by_name: dict[str, list[ActionCandidate]] = {}
        for candidate in candidates:
            by_name.setdefault(candidate.action_name, []).append(candidate)
        generated = self.inducer.mixed_candidates(
            tuple(by_name),
            subgoal_action_names=tuple(
                action
                for hypothesis in goal_hypotheses
                for action in getattr(hypothesis, "supporting_actions", ())
            ),
        )
        if not generated:
            return None
        rebound: list[OptionStep] = []
        for step in generated[0].steps:
            matches = by_name.get(step.action_name, ())
            if len(matches) == 1:
                method = "unique_action_schema"
            elif matches and all(candidate.action_data for candidate in matches):
                method = BALANCED_CAUSAL_BINDING
            else:
                return None
            rebound.append(
                OptionStep(
                    step.action_name,
                    binding_method=method,
                    expected_effect="level_progress_only",
                )
            )
        return GoalDirectedOption(
            schema="mixed_automaton",
            steps=tuple(rebound),
            source="blank_posterior_mixed_causal_discovery",
        )

    def _choose_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any] = (),
    ) -> GoalDirectedOption | None:
        transferred = [
            option
            for option in self.registry.eligible_transferred_options()
            if option.option_id not in self._tried_option_ids
        ]
        reproductions = [
            option
            for option in self.registry.reproduction_candidates()
            if option.option_id not in self._tried_option_ids
        ]
        if self.phase == "confirmation" and transferred:
            assessed = self.evaluator.assess(
                tuple(transferred),
                registry=self.registry,
                tried_option_ids=self._tried_option_ids,
            )
            if assessed:
                self._functional_sources["compiled_registry"] += 1
                return assessed[0].option
        if self.phase in {"reproduction", "discovery"} and reproductions:
            assessed = self.evaluator.assess(
                tuple(reproductions),
                registry=self.registry,
                tried_option_ids=self._tried_option_ids,
            )
            if assessed:
                self._functional_sources["reproduction_registry"] += 1
                return assessed[0].option

        if self.witness_schema == "path_successor":
            option = self._dynamic_successor_option(
                state,
                candidates,
                horizon=self.witness_horizon,
                source="t10_0b_structure_fresh_regrounding",
            )
        elif self.witness_schema == "repeat_target":
            option = self._balanced_repeat_option(
                candidates,
                horizon=self.witness_horizon or 5,
                witness=True,
                source="t10_0b_structure_fresh_regrounding",
            )
        else:
            # Blank-posterior discovery: structural paths are proposed before
            # a generic balanced causal intervention.  No game id or historic
            # action is consulted.
            option = (
                self._mixed_option(
                    candidates,
                    goal_hypotheses=goal_hypotheses,
                )
                if self.prefer_mixed
                else None
            )
            if option is None:
                option = self._dynamic_successor_option(
                    state,
                    candidates,
                    source="blank_posterior_structural_discovery",
                )
            if option is None:
                option = self._balanced_repeat_option(
                    candidates,
                    horizon=MAXIMUM_REPEAT_HORIZON,
                    witness=False,
                    source="blank_posterior_causal_binding_discovery",
                )
        if option is not None:
            self._functional_option_trials += 1
            self._functional_sources[option.source] += 1
        return option

    def _continue_active_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
    ) -> ActionCandidate | None:
        option = self._active_option
        if option is None or self._active_cursor >= len(option.steps):
            return super()._continue_active_option(state, candidates)
        step = option.steps[self._active_cursor]
        if step.binding_method == DYNAMIC_SUCCESSOR:
            grounded = tuple(
                GroundedAction(
                    candidate.action_name,
                    tuple(dict(candidate.action_data).items()),
                )
                for candidate in candidates
            )
            chain = chain_successor_macro(
                state,
                grounded,
                config=SearchConfig(maximum_horizon=MAXIMUM_OPTION_HORIZON),
            )
            if chain is None or not chain.actions:
                self._last_grounding_failure = "dynamic_successor_miss"
                return None
            next_index = min(self._active_cursor, len(chain.actions) - 1)
            if self._last_successor_action_data is not None:
                for index, item in enumerate(chain.actions[:-1]):
                    if _same_action_data(item.data, self._last_successor_action_data):
                        next_index = index + 1
                        break
            wanted = chain.actions[next_index]
            matches = [
                candidate
                for candidate in candidates
                if candidate.action_name == wanted.action_name
                and _same_action_data(candidate.action_data, wanted.data)
            ]
            if len(matches) != 1:
                self._last_grounding_failure = "dynamic_successor_reacquisition_miss"
                return None
            selected = matches[0]
            if (
                self._last_successor_action_data is not None
                and not _same_action_data(
                    selected.action_data, self._last_successor_action_data
                )
            ):
                self._successor_advances += 1
            self._last_successor_action_data = dict(selected.action_data)
            self._pending_grounded_candidate = selected
            self._binding_method_uses[step.binding_method] += 1
            return selected
        if step.binding_method not in {BALANCED_CAUSAL_BINDING, WITNESS_BINDING}:
            return super()._continue_active_option(state, candidates)
        matches = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.action_name == step.action_name
                and candidate.action_data
            ),
            key=_candidate_order_key,
        )
        if not matches:
            self._last_grounding_failure = "balanced_causal_binding_miss"
            return None
        selected = matches[self.exploration_offset % len(matches)]
        self._pending_grounded_candidate = selected
        self._binding_method_uses[step.binding_method] += 1
        if step.binding_method == WITNESS_BINDING:
            self._witness_binding_uses += 1
        else:
            self._balanced_binding_uses += 1
        return selected

    def observe_transition(self, record: TransitionRecord) -> None:
        """Update the posterior without visually productive auto-extension."""

        # Skip ScheduledGoalDirectedSageTController.observe_transition because
        # its T10.3.5 productive-extension rule intentionally used any non-noop
        # transition.  The next implementation in the MRO retains relational
        # grounding and posterior/registry updates but not that extension.
        super(ScheduledGoalDirectedSageTController, self).observe_transition(record)
        progressed = bool(
            record.diff.level_complete
            or record.obs_after.levels_completed > record.obs_before.levels_completed
        )
        state_hash = str(record.obs_after.grid_hash)
        if progressed:
            self._seen_state_hashes.clear()
            self._last_successor_action_data = None
            return
        if state_hash in self._seen_state_hashes and self._active_option is not None:
            self._causal_cycle_aborts += 1
            self._finish_active_option(progressed=False, reason="causal_state_cycle")
        self._seen_state_hashes.add(state_hash)

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "functional_first": True,
                "latency_is_telemetry_only": True,
                "level_progress_is_only_success_credit": True,
                "visual_productive_extension_enabled": False,
                "exploration_offset_persisted": False,
                "causal_cycle_aborts": self._causal_cycle_aborts,
                "balanced_binding_uses": self._balanced_binding_uses,
                "witness_binding_uses": self._witness_binding_uses,
                "functional_option_trials": self._functional_option_trials,
                "functional_sources": dict(self._functional_sources),
                "mixed_discovery_preferred": self.prefer_mixed,
                "successor_advances": self._successor_advances,
                "successor_action_data_persisted": False,
            }
        )
        return base


__all__ = [
    "BALANCED_CAUSAL_BINDING",
    "FORMAT_VERSION",
    "FunctionalGoalDirectedSageTController",
    "MAXIMUM_REPEAT_HORIZON",
    "WITNESS_BINDING",
]
