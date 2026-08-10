"""Relational-binding recovery controller for the T10.3.3 source pilot.

T10.3.2 proved that a coordinate-free structural digest can still be too
coarse: several legal parameterized actions may address distinct targets with
the same digest.  This continuation keeps coordinates and entity identities
strictly branch-local.  Persisted programs retain only a binding method and a
structural/effect contract; live action data are reacquired on every reset.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from v3.schemas import TransitionRecord

from .contracts import AbstractState, ActionCandidate
from .goal_directed_v10_3_2 import (
    MAXIMUM_OPTION_HORIZON,
    GoalDirectedOption,
    GoalDirectedSageTController,
    OptionStep,
)
from .progress_witness_v10 import GroundedAction, SearchConfig, chain_successor_macro

FORMAT_VERSION = "sage-t10.3.3-relational-binding-recovery-v1"
BRANCH_PRODUCTIVE_ANCHOR = "branch_productive_anchor"
DYNAMIC_SUCCESSOR = "dynamic_successor_replan"
UNIQUE_STRUCTURAL = "unique_structural_candidate"
UNIQUE_ACTION_SCHEMA = "unique_action_schema"
PROTECTED_ROUTE_STERILE_LIMIT = 4


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _action_data(candidate: ActionCandidate) -> dict[str, Any]:
    return dict(candidate.action_data)


def _same_action_data(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _canonical(dict(left)) == _canonical(dict(right))


class RelationalGoalDirectedSageTController(GoalDirectedSageTController):
    """Goal controller with reset-local anchors and explicit binding failures."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._productive_anchor_by_action: dict[str, dict[str, Any]] = {}
        self._proposal_anchor_by_action: dict[str, dict[str, Any]] = {}
        self._pending_grounded_candidate: ActionCandidate | None = None
        self._consecutive_sterile_transitions = 0
        self._binding_rejections: Counter[str] = Counter()
        self._binding_method_uses: Counter[str] = Counter()
        self._structural_collision_count = 0
        self._proposal_reacquisitions = 0
        self._last_grounding_failure = "grounding_miss"

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self._clear_ephemeral_bindings()
        self._consecutive_sterile_transitions = 0
        self._binding_rejections.clear()
        self._binding_method_uses.clear()
        self._structural_collision_count = 0
        self._proposal_reacquisitions = 0
        self._last_grounding_failure = "grounding_miss"

    def _clear_ephemeral_bindings(self) -> None:
        self._productive_anchor_by_action.clear()
        self._proposal_anchor_by_action.clear()
        self._pending_grounded_candidate = None

    def decide(
        self,
        *,
        symbolic_action_name: str,
        symbolic_action_data: Mapping[str, Any] | None,
        observation: Any,
        legal_actions: Sequence[Any],
        mechanic_theory: Any | None = None,
        goal_hypotheses: Sequence[Any] = (),
        route_memory: Any | None = None,
        danger_veto: Callable[[ActionCandidate], bool] | None = None,
        protected_route: bool = False,
    ):
        proposal_name = str(symbolic_action_name).strip().upper()
        proposal_data = dict(symbolic_action_data or {})
        self._proposal_anchor_by_action[proposal_name] = proposal_data

        def symmetric_veto(candidate: ActionCandidate) -> bool:
            if (
                candidate.action_name == proposal_name
                and _same_action_data(candidate.action_data, proposal_data)
            ):
                return False
            productive_anchor = self._productive_anchor_by_action.get(
                candidate.action_name
            )
            if (
                productive_anchor is not None
                and _same_action_data(candidate.action_data, productive_anchor)
            ):
                return False
            return bool(danger_veto is not None and danger_veto(candidate))

        effective_protection = bool(
            protected_route
            and self._consecutive_sterile_transitions
            < PROTECTED_ROUTE_STERILE_LIMIT
        )
        return super().decide(
            symbolic_action_name=symbolic_action_name,
            symbolic_action_data=symbolic_action_data,
            observation=observation,
            legal_actions=legal_actions,
            mechanic_theory=mechanic_theory,
            goal_hypotheses=goal_hypotheses,
            route_memory=route_memory,
            danger_veto=symmetric_veto,
            protected_route=effective_protection,
        )

    def _choose_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any] = (),
    ) -> GoalDirectedOption | None:
        untried_transferred = [
            option
            for option in self.registry.eligible_transferred_options()
            if option.option_id not in self._tried_option_ids
        ]
        if self.phase == "confirmation" and untried_transferred:
            assessments = self.evaluator.assess(
                tuple(untried_transferred),
                registry=self.registry,
                tried_option_ids=self._tried_option_ids,
            )
            return assessments[0].option if assessments else None

        untried_reproductions = [
            option
            for option in self.registry.reproduction_candidates()
            if option.option_id not in self._tried_option_ids
        ]
        if self.phase == "discovery" and untried_reproductions:
            assessments = self.evaluator.assess(
                tuple(untried_reproductions),
                registry=self.registry,
                tried_option_ids=self._tried_option_ids,
            )
            return assessments[0].option if assessments else None

        generated: list[GoalDirectedOption] = list(untried_transferred)
        by_name: dict[str, list[ActionCandidate]] = {}
        signatures: dict[tuple[str, str], list[ActionCandidate]] = {}
        for candidate in candidates:
            by_name.setdefault(candidate.action_name, []).append(candidate)
            signature = self._candidate_signature(state, candidate)
            if signature is not None:
                signatures.setdefault((candidate.action_name, signature), []).append(
                    candidate
                )

        for rows in signatures.values():
            if len(rows) > 1:
                self._structural_collision_count += len(rows)

        for action_name, anchor in sorted(self._productive_anchor_by_action.items()):
            if sum(
                _same_action_data(candidate.action_data, anchor)
                for candidate in by_name.get(action_name, ())
            ) != 1:
                continue
            for length in (2, 4, 8, 16):
                generated.append(
                    GoalDirectedOption(
                        schema="repeat_target",
                        steps=tuple(
                            OptionStep(
                                action_name,
                                binding_method=BRANCH_PRODUCTIVE_ANCHOR,
                                expected_effect="branch_productive_effect",
                            )
                            for _ in range(length)
                        ),
                        source="branch_productive_reacquisition",
                    )
                )

        for (action_name, signature), rows in sorted(signatures.items()):
            if len(rows) != 1 or not rows[0].action_data:
                continue
            step = OptionStep(
                action_name,
                binding_method=UNIQUE_STRUCTURAL,
                structural_signature=signature,
            )
            for length in (2, 4, 8, 16):
                generated.append(
                    GoalDirectedOption(
                        schema="repeat_target",
                        steps=tuple(step for _ in range(length)),
                        source="unique_structural_search",
                    )
                )

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
        if chain is not None and chain.actions:
            generated.append(
                GoalDirectedOption(
                    schema="path_successor",
                    steps=tuple(
                        OptionStep(
                            item.action_name,
                            binding_method=DYNAMIC_SUCCESSOR,
                        )
                        for item in chain.actions[:MAXIMUM_OPTION_HORIZON]
                    ),
                    source="dynamic_successor_replanning",
                )
            )

        mixed = self.inducer.mixed_candidates(
            tuple(candidate.action_name for candidate in candidates),
            subgoal_action_names=tuple(
                action
                for hypothesis in goal_hypotheses
                for action in getattr(hypothesis, "supporting_actions", ())
            ),
        )
        for option in mixed:
            rebound = []
            for step in option.steps:
                rows = by_name.get(step.action_name, ())
                if len(rows) == 1:
                    method = UNIQUE_ACTION_SCHEMA
                elif step.action_name in self._productive_anchor_by_action:
                    method = BRANCH_PRODUCTIVE_ANCHOR
                else:
                    rebound = []
                    break
                rebound.append(
                    OptionStep(
                        step.action_name,
                        binding_method=method,
                        expected_effect=step.expected_effect,
                    )
                )
            if rebound:
                generated.append(
                    GoalDirectedOption(
                        schema="mixed_automaton",
                        steps=tuple(rebound),
                        source="effect_graph_subgoal_composition",
                    )
                )

        unique = {option.option_id: option for option in generated}
        assessments = self.evaluator.assess(
            tuple(unique.values()),
            registry=self.registry,
            tried_option_ids=self._tried_option_ids,
        )
        return assessments[0].option if assessments else None

    def _continue_active_option(
        self, state: AbstractState, candidates: Sequence[ActionCandidate]
    ) -> ActionCandidate | None:
        option = self._active_option
        if option is None:
            return None
        if self._active_cursor >= len(option.steps):
            self._finish_active_option(progressed=False, reason="option_exhausted")
            return None
        step = option.steps[self._active_cursor]
        matches = [
            candidate
            for candidate in candidates
            if candidate.action_name == step.action_name
        ]
        selected: ActionCandidate | None = None
        if step.binding_method == BRANCH_PRODUCTIVE_ANCHOR:
            anchor = self._productive_anchor_by_action.get(step.action_name)
            used_proposal = False
            if anchor is None:
                anchor = self._proposal_anchor_by_action.get(step.action_name)
                used_proposal = anchor is not None
            rebound = [
                candidate
                for candidate in matches
                if anchor is not None
                and _same_action_data(candidate.action_data, anchor)
            ]
            if len(rebound) == 1:
                selected = rebound[0]
                if used_proposal:
                    self._proposal_reacquisitions += 1
            else:
                self._last_grounding_failure = (
                    "missing_branch_productive_anchor"
                    if anchor is None
                    else "ambiguous_branch_productive_anchor"
                )
        elif step.binding_method == DYNAMIC_SUCCESSOR:
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
            if chain is not None and chain.actions:
                next_action = chain.actions[0]
                rebound = [
                    candidate
                    for candidate in matches
                    if _same_action_data(candidate.action_data, next_action.data)
                ]
                if len(rebound) == 1:
                    selected = rebound[0]
            if selected is None:
                self._last_grounding_failure = "dynamic_successor_miss"
        elif step.binding_method == UNIQUE_STRUCTURAL:
            rebound = [
                candidate
                for candidate in matches
                if self._candidate_signature(state, candidate)
                == step.structural_signature
            ]
            if len(rebound) == 1:
                selected = rebound[0]
            else:
                self._last_grounding_failure = "structural_equivalence_collision"
        elif len(matches) == 1:
            selected = matches[0]
        else:
            self._last_grounding_failure = "action_schema_ambiguity"

        if selected is not None:
            self._pending_grounded_candidate = selected
            self._binding_method_uses[step.binding_method] += 1
        return selected

    def _finish_active_option(self, *, progressed: bool, reason: str) -> None:
        normalized_reason = str(reason)
        if not progressed and normalized_reason == "grounding_miss":
            normalized_reason = self._last_grounding_failure
        if not progressed:
            self._binding_rejections[normalized_reason] += 1
        super()._finish_active_option(
            progressed=progressed,
            reason=normalized_reason,
        )
        self._pending_grounded_candidate = None
        self._last_grounding_failure = "grounding_miss"

    def observe_transition(self, record: TransitionRecord) -> None:
        super().observe_transition(record)
        action_name = str(record.action.name).strip().upper()
        action_data = {
            key: value
            for key, value in (("x", record.action.x), ("y", record.action.y))
            if value is not None
        }
        productive = bool(not record.diff.is_noop and not record.diff.game_over)
        if productive and action_data:
            self._productive_anchor_by_action[action_name] = action_data
        if record.diff.is_noop:
            self._consecutive_sterile_transitions += 1
        else:
            self._consecutive_sterile_transitions = 0
        self._pending_grounded_candidate = None

    def note_level_change(self) -> None:
        self._clear_ephemeral_bindings()
        self._consecutive_sterile_transitions = 0
        super().note_level_change()

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "ephemeral_binding_memory": True,
                "ephemeral_anchor_count": len(self._productive_anchor_by_action),
                "ephemeral_action_data_persisted": False,
                "binding_rejections": dict(self._binding_rejections),
                "binding_method_uses": dict(self._binding_method_uses),
                "structural_collision_count": self._structural_collision_count,
                "proposal_reacquisitions": self._proposal_reacquisitions,
                "protected_route_sterile_limit": PROTECTED_ROUTE_STERILE_LIMIT,
            }
        )
        return base


__all__ = [
    "BRANCH_PRODUCTIVE_ANCHOR",
    "DYNAMIC_SUCCESSOR",
    "FORMAT_VERSION",
    "PROTECTED_ROUTE_STERILE_LIMIT",
    "UNIQUE_ACTION_SCHEMA",
    "UNIQUE_STRUCTURAL",
    "RelationalGoalDirectedSageTController",
]
