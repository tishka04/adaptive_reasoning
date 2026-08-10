"""Bounded-compute continuation for the T10.3.3 relational controller.

T10.3.3 produced a genuine level-progress witness, but its full unified
proposal path became progressively more expensive and kept running after the
registered objective had already been reached.  This module preserves the
coordinate-free relational binding recovery while bounding the symbolic
proposal profile and providing a fast path for an already-grounded SAGE.T
option.  Both experimental and comparison arms use the same bounded unified
configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..live_transition_loop import build_observation
from ..unified_cognitive_controller import (
    CognitiveDecision,
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
    _normalize_action,
    _normalize_actions,
)
from .compiler import compile_observation
from .contracts import ActionCandidate, normalized_action_candidates
from .goal_directed_v10_3_3 import (
    PROTECTED_ROUTE_STERILE_LIMIT,
    RelationalGoalDirectedSageTController,
    _same_action_data,
)

FORMAT_VERSION = "sage-t10.3.4-bounded-compute-v1"
DISCOVERY_WARMUP_ACTIONS = 8
EXPLORATION_ACTIONS_BETWEEN_OPTIONS = 8
TRANSITION_HISTORY_LIMIT = 32
OPERATOR_INDUCTION_INTERVAL = 8


def bounded_unified_config(*, sage_t_authority_mode: str) -> UnifiedCognitiveConfig:
    """Return the preregistered symmetric low-growth unified profile."""

    authority = str(sage_t_authority_mode).strip().lower()
    if authority not in {"active", "off"}:
        raise ValueError("bounded unified authority must be active or off")
    return UnifiedCognitiveConfig(
        max_bootstrap_experiments=16,
        reprobe_interval=8,
        max_click_targets=8,
        max_generated_goal_candidates=6,
        max_generated_goals_per_family=1,
        max_temporal_plans=6,
        max_temporal_plan_starts_total=6,
        max_causal_subgoal_edges=12,
        max_causal_hierarchical_options=4,
        operator_induction_interval=OPERATOR_INDUCTION_INTERVAL,
        enable_operator_planning=False,
        enable_theory_planning=False,
        enable_promoted_options=False,
        enable_causal_hierarchical_options=False,
        enable_effect_conditioned_downstream_subgoals=False,
        enable_persistent_directional_pursuit=False,
        enable_mediated_entity_effect_induction=False,
        enable_online_mediated_anti_unification=False,
        enable_active_mediated_discrimination=False,
        enable_active_mode_restoration=False,
        enable_terminal_mediated_exploitation=False,
        enable_successor_policy_chaining=False,
        enable_active_successor_exploration=False,
        enable_successor_structural_transfer=False,
        enable_active_mediated_replication=False,
        enable_horizon_stable_learning_epochs=False,
        enable_online_horizon_learning_arbiter=False,
        enable_terminal_negative_frontier_exploration=False,
        enable_adaptive_terminal_frontier_horizon=False,
        enable_dormant_terminal_lineage=False,
        enable_structural_terminal_frontiers=False,
        enable_terminal_causal_reduction=False,
        enable_active_frontier_reacquisition=False,
        enable_recursive_terminal_causal_minimization=False,
        enable_structural_frontier_transfer=False,
        enable_progressive_terminal_routes=False,
        enable_level_route_memory=False,
        enable_terminal_relational_stencil_induction=False,
        enable_online_structural_break_detection=False,
        enable_active_structural_hypothesis_arbitration=False,
        enable_structural_regime_abstraction=False,
        enable_hierarchical_structural_theory_composition=False,
        enable_frontier_oriented_exploration=False,
        enable_delayed_frontier_terminal_credit=False,
        enable_subeffect_eligibility_relay=False,
        enable_generalized_frontier_stall_detection=False,
        enable_transferable_causal_schema_export=False,
        enable_transferable_causal_schema_priors=False,
        enable_terminal_multiform_relational_induction=False,
        max_level_routes=8,
        max_level_route_actions=64,
        sage_t_authority_mode=authority,
        sage_t_counterfactual_gate_passed=authority == "active",
        sage_t_active_gate_passed=authority == "active",
    )


class BoundedGoalDirectedSageTController(RelationalGoalDirectedSageTController):
    """Relational controller exposing a safe active-option continuation path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("warmup_actions", DISCOVERY_WARMUP_ACTIONS)
        kwargs.setdefault("exploration_interval", EXPLORATION_ACTIONS_BETWEEN_OPTIONS)
        super().__init__(*args, **kwargs)
        self._fast_path_attempts = 0
        self._fast_path_applied = 0
        self._fast_path_fallbacks = 0

    @property
    def fast_path_ready(self) -> bool:
        """Whether a live option can be continued without recomputing a proposal."""

        return self._active_option is not None

    def fast_active_decision(
        self,
        *,
        symbolic_action_name: str,
        symbolic_action_data: Mapping[str, Any] | None,
        observation: Any,
        legal_actions: Sequence[Any],
        danger_veto: Callable[[ActionCandidate], bool] | None = None,
        protected_route: bool = False,
    ) -> ActionCandidate | None:
        """Continue only an existing option; never induce a new one here."""

        if self._active_option is None:
            return None
        self._fast_path_attempts += 1
        proposal_name = str(symbolic_action_name).strip().upper()
        proposal_data = dict(symbolic_action_data or {})
        self._proposal_anchor_by_action[proposal_name] = proposal_data
        effective_protection = bool(
            protected_route
            and self._consecutive_sterile_transitions < PROTECTED_ROUTE_STERILE_LIMIT
        )
        if effective_protection:
            self._fast_path_fallbacks += 1
            return None
        try:
            candidates = normalized_action_candidates(legal_actions)
            state = compile_observation(
                observation,
                regime_index=self._regime_index,
            )
        except (TypeError, ValueError):
            self._finish_active_option(
                progressed=False, reason="uncompilable_fast_path"
            )
            self._fast_path_fallbacks += 1
            return None

        self._last_decision_registry_checksum = None
        self._decision_index += 1
        selected = self._continue_active_option(state, candidates)
        if selected is None:
            self._finish_active_option(progressed=False, reason="grounding_miss")
            self._fast_path_fallbacks += 1
            return None

        productive_anchor = self._productive_anchor_by_action.get(selected.action_name)
        vetoed = bool(
            danger_veto is not None
            and not (
                selected.action_name == proposal_name
                and _same_action_data(selected.action_data, proposal_data)
            )
            and not (
                productive_anchor is not None
                and _same_action_data(selected.action_data, productive_anchor)
            )
            and danger_veto(selected)
        )
        if vetoed:
            self._finish_active_option(progressed=False, reason="danger_veto")
            self._fast_path_fallbacks += 1
            return None

        self._pending_step = self._active_option.steps[self._active_cursor]
        self._pending_option = self._active_option
        self._source_counts["sage_t_goal_option_fast_path"] += 1
        transferred_ids = {
            option.option_id for option in self.registry.eligible_transferred_options()
        }
        if (
            self.registry_checksum is not None
            and self._active_option.option_id in transferred_ids
        ):
            self._registry_used_in_decision = True
            self._last_decision_registry_checksum = self.registry_checksum
        self._fast_path_applied += 1
        return selected

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "discovery_warmup_actions": DISCOVERY_WARMUP_ACTIONS,
                "exploration_actions_between_options": (
                    EXPLORATION_ACTIONS_BETWEEN_OPTIONS
                ),
                "fast_path_attempts": self._fast_path_attempts,
                "fast_path_applied": self._fast_path_applied,
                "fast_path_fallbacks": self._fast_path_fallbacks,
            }
        )
        return base


class BoundedUnifiedCognitiveController(UnifiedCognitiveController):
    """Unified controller with symmetric caps and active-option short circuit."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bounded_fast_path_decisions = 0
        self._bounded_fast_path_fallbacks = 0
        self._maximum_retained_transitions = 0

    def select_action(
        self,
        *,
        current_grid: Any,
        available_actions: Sequence[Any],
        legacy_action: Any,
        legacy_action_data: Mapping[str, Any] | None = None,
        available_action_candidates: Sequence[Any] | None = None,
        game_state: str = "NOT_FINISHED",
        levels_completed: int = 0,
    ) -> CognitiveDecision:
        goal = self.sage_t_controller
        if (
            not isinstance(goal, BoundedGoalDirectedSageTController)
            or not goal.fast_path_ready
        ):
            return super().select_action(
                current_grid=current_grid,
                available_actions=available_actions,
                legacy_action=legacy_action,
                legacy_action_data=legacy_action_data,
                available_action_candidates=available_action_candidates,
                game_state=game_state,
                levels_completed=levels_completed,
            )

        actions = _normalize_actions(available_actions)
        legacy_name = _normalize_action(legacy_action)
        if legacy_name not in actions and actions:
            legacy_name = actions[0]
        self._step += 1
        self._branch_step += 1
        self.theory.seed_actions(actions)
        observation = build_observation(
            current_grid,
            available_actions=actions,
            game_state=game_state,
            levels_completed=levels_completed,
            infer_players=True,
        )
        safe_actions = self._safe_actions(observation.grid_hash, actions) or actions
        protected = self._protected_competence_available(observation, safe_actions)
        legal = tuple(available_action_candidates or safe_actions)
        selected = goal.fast_active_decision(
            symbolic_action_name=legacy_name,
            symbolic_action_data=legacy_action_data,
            observation=observation,
            legal_actions=legal,
            protected_route=protected,
            danger_veto=lambda candidate: self._neural_danger_veto(
                observation.grid_hash,
                candidate.action_name,
                candidate.action_data,
            ),
        )
        if selected is not None and selected.action_name in safe_actions:
            decision = CognitiveDecision(
                action_name=selected.action_name,
                action_data=dict(selected.action_data),
                source="sage_t_joint_program",
                reason="bounded_active_option_fast_path",
                confidence=max(
                    (particle.probability for particle in goal.posterior.particles),
                    default=0.0,
                ),
            )
            self._bounded_fast_path_decisions += 1
        else:
            decision = CognitiveDecision(
                action_name=legacy_name,
                action_data=dict(legacy_action_data or {}),
                source="bounded_legacy_fallback",
                reason="active option fast path became unavailable",
            )
            self._bounded_fast_path_fallbacks += 1
        self._pending_decision = decision
        self._pending_action_candidates = tuple(available_action_candidates or ())
        self._decision_sources[decision.source] += 1
        return decision

    def observe_transition(self, *args: Any, **kwargs: Any):
        update = super().observe_transition(*args, **kwargs)
        transitions = self.belief_loop.profiler.transitions
        if len(transitions) > TRANSITION_HISTORY_LIMIT:
            del transitions[:-TRANSITION_HISTORY_LIMIT]
        self._maximum_retained_transitions = max(
            self._maximum_retained_transitions,
            len(transitions),
        )
        return update

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "bounded_compute_profile": True,
                "bounded_fast_path_decisions": self._bounded_fast_path_decisions,
                "bounded_fast_path_fallbacks": self._bounded_fast_path_fallbacks,
                "transition_history_limit": TRANSITION_HISTORY_LIMIT,
                "maximum_retained_transitions": self._maximum_retained_transitions,
                "operator_induction_interval": OPERATOR_INDUCTION_INTERVAL,
                "operator_planning_enabled": self.config.enable_operator_planning,
            }
        )
        return base


__all__ = [
    "DISCOVERY_WARMUP_ACTIONS",
    "EXPLORATION_ACTIONS_BETWEEN_OPTIONS",
    "FORMAT_VERSION",
    "OPERATOR_INDUCTION_INTERVAL",
    "TRANSITION_HISTORY_LIMIT",
    "BoundedGoalDirectedSageTController",
    "BoundedUnifiedCognitiveController",
    "bounded_unified_config",
]
