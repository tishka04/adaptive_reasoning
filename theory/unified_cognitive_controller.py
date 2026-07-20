"""Single live execution path for the repository's cognitive components.

The repository historically grew several capable but disconnected stacks.
This controller is deliberately thin: it does not implement another agent.
It orchestrates the existing perception, belief revision, experiment design,
operator induction, theory-conditioned planning, safety memory, and legacy
planner behind one decision interface that can be used by the registered ARC
agent.

Every belief update is caused by an observed before/action/after transition.
The controller never steps an environment and never treats a prior or a trace
as proof.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from v3.control.operator_search import OperatorSearcher
from v3.control.progress_tracker import ProgressTracker
from v3.mechanics.operator_inducer import OperatorInducer
from v3.schemas import GameObservation, Operator
from v5.control.anti_attractor import AntiAttractor
from v5.control.danger_memory import DangerMemoryV5, action_key
from v5.schemas import PrimitiveAction

from .epistemic_metrics import HypothesisStatus
from .experiment_designer import DiscriminatingExperimentDesigner
from .generic_discriminating_experiment_designer import (
    DesignedExperimentAction,
    DiscriminatingPrediction,
    GenericDiscriminatingExperimentDesigner,
)
from .live_transition_loop import (
    LiveTransitionBeliefLoop,
    LiveTransitionUpdate,
    build_observation,
)
from .online_relational_option import (
    CompiledRelationalOption,
    OnlineRelationalOptionCompiler,
    OptionAssessment,
    OptionExecutionMemory,
    observe_option_progress,
)
from .online_goal_hypothesis import (
    GoalHypothesisGenerator,
    ObjectiveDiscriminatingExperimentDesigner,
    intervention_id,
    semantic_intervention_signature,
)
from .online_causal_subgoal_graph import OnlineCausalSubgoalGraph
from .online_causal_option import OnlineCausalOptionStore
from .online_terminal_objective import (
    OnlineTerminalObjectiveStore,
    TerminalObjectiveAssessment,
    TerminalObjectiveStatus,
)
from .online_temporal_goal_composition import (
    OnlineTemporalGoalComposer,
    TemporalPlanStatus,
)
from .promoted_relational_rule import PromotedRelationalRule
from .theory_conditioned_planner import TheoryConditionedPlanner


@dataclass(frozen=True)
class UnifiedCognitiveConfig:
    """Bounded runtime policy for the consolidated controller."""

    max_bootstrap_experiments: int = 32
    reprobe_interval: int = 12
    max_click_targets: int = 12
    max_relation_target_colors: int = 4
    generic_confirmation_evidence: int = 2
    generic_refutation_evidence: int = 2
    promotion_min_support: int = 2
    promotion_min_independent_contexts: int = 2
    operator_induction_interval: int = 1
    max_option_attempts_per_context: int = 2
    min_option_executions_before_quarantine: int = 3
    max_terminal_objective_probes_per_objective: int = 2
    max_terminal_objective_probes_total: int = 16
    terminal_objective_credit_window: int = 6
    terminal_objective_min_support: int = 2
    max_terminal_objective_ablations_per_objective: int = 1
    max_generated_goal_candidates: int = 10
    max_generated_goals_per_family: int = 2
    max_temporal_plans: int = 12
    max_temporal_plan_starts_total: int = 12
    max_temporal_starts_per_plan: int = 2
    max_temporal_actions_per_plan: int = 8
    max_temporal_stalls_per_step: int = 2
    max_causal_subgoal_edges: int = 24
    max_causal_edges_per_blocked_target: int = 3
    causal_edge_min_support: int = 2
    causal_effect_credit_window: int = 6
    max_causal_confirmation_starts_per_branch: int = 2
    max_causal_hierarchical_options: int = 8
    max_causal_option_downstream_actions: int = 6
    max_causal_option_trials_per_signature: int = 3
    causal_option_terminal_credit_window: int = 8
    max_effect_conditioned_downstream_subgoals: int = 24
    max_downstream_subgoals_per_effect: int = 4
    causal_option_base_downstream_actions: int = 4
    causal_option_progress_extension_actions: int = 2
    max_actions_per_effect_conditioned_subgoal: int = 2
    persistent_actions_per_progress: int = 2
    max_persistent_actions_per_subgoal: int = 6
    persistent_rollout_actions_per_progress: int = 2
    max_persistent_downstream_actions: int = 10
    persistent_credit_steps_per_progress: int = 4
    max_persistent_credit_window: int = 16
    enable_relational_experiments: bool = True
    enable_operator_planning: bool = True
    enable_theory_planning: bool = True
    enable_promoted_options: bool = True
    enable_active_goal_hypotheses: bool = True
    enable_temporal_goal_composition: bool = True
    enable_causal_subgoal_induction: bool = True
    enable_causal_effect_credit: bool = True
    enable_causal_hierarchical_options: bool = True
    enable_effect_conditioned_downstream_subgoals: bool = True
    enable_state_conditioned_directional_control: bool = True
    enable_persistent_directional_pursuit: bool = True
    enable_entity_anchored_interventions: bool = True


@dataclass(frozen=True)
class CognitiveDecision:
    """One auditable primitive decision emitted by the unified controller."""

    action_name: str
    action_data: Dict[str, Any] = field(default_factory=dict)
    source: str = "legacy_fallback"
    reason: str = ""
    confidence: float = 0.0
    competing_hypotheses: Tuple[str, ...] = ()
    mechanic_hypotheses: Tuple[str, ...] = ()
    operator_id: str = ""
    source_rule_key: str = ""
    option_id: str = ""
    preparation_for_rule_key: str = ""
    objective_id: str = ""
    objective_status: str = ""
    objective_distance: float | None = None
    intervention_id: str = ""
    ablation_of_objective_id: str = ""
    predicted_goal_reductions: Tuple[str, ...] = ()
    temporal_plan_id: str = ""
    temporal_target_objective_id: str = ""
    temporal_plan_status: str = ""
    temporal_step_index: int | None = None
    temporal_step_count: int = 0
    temporal_step_target_distance: float | None = None
    causal_subgoal_edge_key: str = ""
    temporal_expected_progress_probability: float | None = None
    temporal_expected_cost: float | None = None
    temporal_selection_utility: float | None = None
    causal_intervention_signature: str = ""
    causal_intervention_utility: float | None = None
    causal_confirmation_trial: bool = False
    causal_option_id: str = ""
    causal_option_edge_key: str = ""
    causal_option_terminal_status: str = ""
    causal_option_phase: str = ""
    causal_option_intervention_signature: str = ""
    causal_option_intervention_transfer_signature: str = ""
    causal_option_intervention_entity_signature: str = ""
    causal_option_intervention_structural_role_signature: str = ""
    causal_option_intervention_instance_signature: str = ""
    causal_option_entity_anchored_intervention: bool = False
    causal_option_selection_utility: float | None = None
    causal_option_replaying_terminal_sequence: bool = False
    causal_option_downstream_subgoal_id: str = ""
    causal_option_downstream_subgoal_status: str = ""
    causal_option_trigger_effect_signature: str = ""
    causal_option_downstream_subgoal_utility: float | None = None
    causal_option_replaying_progress_sequence: bool = False
    causal_option_latent_mode_signature: str = ""
    causal_option_directional_effect_status: str = ""
    causal_option_directional_expected_gain: float | None = None
    causal_option_directional_confidence: float | None = None
    causal_option_directionally_compatible: bool = True
    causal_option_reversible_action: bool = False
    causal_option_directional_mode_contrast: bool = False
    causal_option_directional_structural_transfer: bool = False
    causal_option_directional_entity_alias_conflict: bool = False
    causal_option_directional_bridge_target_mode_signature: str = ""
    causal_option_directional_bridge_followup_action_signature: str = ""
    causal_option_persistent_pursuit: bool = False
    causal_option_persistent_attempt_index: int = 0
    causal_option_persistent_action_limit: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_name,
            "action_data": dict(self.action_data),
            "source": self.source,
            "reason": self.reason,
            "confidence": round(float(self.confidence), 4),
            "competing_hypotheses": list(self.competing_hypotheses),
            "mechanic_hypotheses": list(self.mechanic_hypotheses),
            "operator_id": self.operator_id,
            "source_rule_key": self.source_rule_key,
            "option_id": self.option_id,
            "preparation_for_rule_key": self.preparation_for_rule_key,
            "objective_id": self.objective_id,
            "objective_status": self.objective_status,
            "objective_distance": self.objective_distance,
            "intervention_id": self.intervention_id,
            "ablation_of_objective_id": self.ablation_of_objective_id,
            "predicted_goal_reductions": list(self.predicted_goal_reductions),
            "temporal_plan_id": self.temporal_plan_id,
            "temporal_target_objective_id": self.temporal_target_objective_id,
            "temporal_plan_status": self.temporal_plan_status,
            "temporal_step_index": self.temporal_step_index,
            "temporal_step_count": self.temporal_step_count,
            "temporal_step_target_distance": (
                self.temporal_step_target_distance
            ),
            "causal_subgoal_edge_key": self.causal_subgoal_edge_key,
            "temporal_expected_progress_probability": (
                self.temporal_expected_progress_probability
            ),
            "temporal_expected_cost": self.temporal_expected_cost,
            "temporal_selection_utility": self.temporal_selection_utility,
            "causal_intervention_signature": self.causal_intervention_signature,
            "causal_intervention_utility": self.causal_intervention_utility,
            "causal_confirmation_trial": self.causal_confirmation_trial,
            "causal_option_id": self.causal_option_id,
            "causal_option_edge_key": self.causal_option_edge_key,
            "causal_option_terminal_status": self.causal_option_terminal_status,
            "causal_option_phase": self.causal_option_phase,
            "causal_option_intervention_signature": (
                self.causal_option_intervention_signature
            ),
            "causal_option_intervention_transfer_signature": (
                self.causal_option_intervention_transfer_signature
            ),
            "causal_option_intervention_entity_signature": (
                self.causal_option_intervention_entity_signature
            ),
            "causal_option_intervention_structural_role_signature": (
                self.causal_option_intervention_structural_role_signature
            ),
            "causal_option_intervention_instance_signature": (
                self.causal_option_intervention_instance_signature
            ),
            "causal_option_entity_anchored_intervention": (
                self.causal_option_entity_anchored_intervention
            ),
            "causal_option_selection_utility": (
                self.causal_option_selection_utility
            ),
            "causal_option_replaying_terminal_sequence": (
                self.causal_option_replaying_terminal_sequence
            ),
            "causal_option_downstream_subgoal_id": (
                self.causal_option_downstream_subgoal_id
            ),
            "causal_option_downstream_subgoal_status": (
                self.causal_option_downstream_subgoal_status
            ),
            "causal_option_trigger_effect_signature": (
                self.causal_option_trigger_effect_signature
            ),
            "causal_option_downstream_subgoal_utility": (
                self.causal_option_downstream_subgoal_utility
            ),
            "causal_option_replaying_progress_sequence": (
                self.causal_option_replaying_progress_sequence
            ),
            "causal_option_latent_mode_signature": (
                self.causal_option_latent_mode_signature
            ),
            "causal_option_directional_effect_status": (
                self.causal_option_directional_effect_status
            ),
            "causal_option_directional_expected_gain": (
                self.causal_option_directional_expected_gain
            ),
            "causal_option_directional_confidence": (
                self.causal_option_directional_confidence
            ),
            "causal_option_directionally_compatible": (
                self.causal_option_directionally_compatible
            ),
            "causal_option_reversible_action": (
                self.causal_option_reversible_action
            ),
            "causal_option_directional_mode_contrast": (
                self.causal_option_directional_mode_contrast
            ),
            "causal_option_directional_structural_transfer": (
                self.causal_option_directional_structural_transfer
            ),
            "causal_option_directional_entity_alias_conflict": (
                self.causal_option_directional_entity_alias_conflict
            ),
            "causal_option_directional_bridge_target_mode_signature": (
                self.causal_option_directional_bridge_target_mode_signature
            ),
            "causal_option_directional_bridge_followup_action_signature": (
                self.causal_option_directional_bridge_followup_action_signature
            ),
            "causal_option_persistent_pursuit": (
                self.causal_option_persistent_pursuit
            ),
            "causal_option_persistent_attempt_index": (
                self.causal_option_persistent_attempt_index
            ),
            "causal_option_persistent_action_limit": (
                self.causal_option_persistent_action_limit
            ),
        }


class UnifiedCognitiveController:
    """Orchestrate existing cognitive modules in one live control loop.

    Decision priority is intentionally causal rather than score-only:

    1. escape an observed lethal/no-op attractor;
    2. execute a plan justified by a confirmed theory rule;
    3. run a bounded discriminating experiment while beliefs are unresolved;
    4. plan with induced, state-conditioned operators;
    5. fall back to the existing v4_1 decision.
    """

    def __init__(
        self,
        game_id: str,
        *,
        available_actions: Sequence[Any] | None = None,
        config: UnifiedCognitiveConfig | None = None,
    ) -> None:
        self.game_id = str(game_id)
        self.config = config or UnifiedCognitiveConfig()
        actions = _normalize_actions(available_actions or [])
        self.belief_loop = LiveTransitionBeliefLoop(
            game_id=self.game_id,
            available_actions=actions,
            infer_players=True,
            verify_every=1,
        )
        self.operator_inducer = OperatorInducer()
        self.experiment_designer = DiscriminatingExperimentDesigner()
        self.generic_designer = GenericDiscriminatingExperimentDesigner(
            max_competing_hypotheses=2,
        )
        self.theory_planner = TheoryConditionedPlanner()
        self.option_compiler = OnlineRelationalOptionCompiler()
        self.option_memory = OptionExecutionMemory()
        self.goal_generator = GoalHypothesisGenerator(
            max_candidates_total=self.config.max_generated_goal_candidates,
            max_candidates_per_family=(
                self.config.max_generated_goals_per_family
            ),
        )
        self.objective_experiment_designer = (
            ObjectiveDiscriminatingExperimentDesigner()
        )
        self.terminal_objectives = OnlineTerminalObjectiveStore(
            max_probe_actions_per_objective=(
                self.config.max_terminal_objective_probes_per_objective
            ),
            max_probe_actions_total=(
                self.config.max_terminal_objective_probes_total
            ),
            terminal_credit_window=(
                self.config.terminal_objective_credit_window
            ),
            minimum_terminal_support=(
                self.config.terminal_objective_min_support
            ),
            max_ablation_actions_per_objective=(
                self.config.max_terminal_objective_ablations_per_objective
            ),
        )
        self.causal_subgoals = OnlineCausalSubgoalGraph(
            max_edges=self.config.max_causal_subgoal_edges,
            max_edges_per_blocked_target=(
                self.config.max_causal_edges_per_blocked_target
            ),
            minimum_independent_support=self.config.causal_edge_min_support,
            delayed_credit_window=self.config.causal_effect_credit_window,
            enable_effect_credit=self.config.enable_causal_effect_credit,
        )
        self.temporal_goals = OnlineTemporalGoalComposer(
            max_plans=self.config.max_temporal_plans,
            max_plan_starts_total=(
                self.config.max_temporal_plan_starts_total
            ),
            max_starts_per_plan=self.config.max_temporal_starts_per_plan,
            max_actions_per_plan=self.config.max_temporal_actions_per_plan,
            max_stalls_per_step=self.config.max_temporal_stalls_per_step,
            terminal_credit_window=(
                self.config.terminal_objective_credit_window
            ),
            minimum_terminal_support=(
                self.config.terminal_objective_min_support
            ),
            max_confirmation_starts_per_branch=(
                self.config.max_causal_confirmation_starts_per_branch
                if self.config.enable_causal_effect_credit else 0
            ),
        )
        self.causal_options = OnlineCausalOptionStore(
            max_options=self.config.max_causal_hierarchical_options,
            max_downstream_actions=(
                self.config.max_causal_option_downstream_actions
            ),
            max_trials_per_signature=(
                self.config.max_causal_option_trials_per_signature
            ),
            terminal_credit_window=(
                self.config.causal_option_terminal_credit_window
            ),
            minimum_terminal_support=(
                self.config.terminal_objective_min_support
            ),
            enable_effect_conditioned_subgoals=(
                self.config.enable_effect_conditioned_downstream_subgoals
            ),
            max_effect_conditioned_subgoals=(
                self.config.max_effect_conditioned_downstream_subgoals
            ),
            max_subgoals_per_effect=(
                self.config.max_downstream_subgoals_per_effect
            ),
            base_downstream_actions=(
                self.config.causal_option_base_downstream_actions
            ),
            progress_extension_actions=(
                self.config.causal_option_progress_extension_actions
            ),
            max_actions_per_effect_conditioned_subgoal=(
                self.config.max_actions_per_effect_conditioned_subgoal
            ),
            enable_state_conditioned_directional_control=(
                self.config.enable_state_conditioned_directional_control
            ),
            enable_persistent_directional_pursuit=(
                self.config.enable_persistent_directional_pursuit
            ),
            enable_entity_anchored_interventions=(
                self.config.enable_entity_anchored_interventions
            ),
            persistent_actions_per_progress=(
                self.config.persistent_actions_per_progress
            ),
            max_persistent_actions_per_subgoal=(
                self.config.max_persistent_actions_per_subgoal
            ),
            persistent_rollout_actions_per_progress=(
                self.config.persistent_rollout_actions_per_progress
            ),
            max_persistent_downstream_actions=(
                self.config.max_persistent_downstream_actions
            ),
            persistent_credit_steps_per_progress=(
                self.config.persistent_credit_steps_per_progress
            ),
            max_persistent_credit_window=(
                self.config.max_persistent_credit_window
            ),
        )
        self.operator_searcher = OperatorSearcher(beam_width=4, max_depth=5)
        self.progress = ProgressTracker()
        self.danger_memory = DangerMemoryV5()
        self.anti_attractor = AntiAttractor()

        self._step = 0
        self._experiment_decisions = 0
        self._generic_experiment_decisions = 0
        self._click_cursor = 0
        self._pending_decision: CognitiveDecision | None = None
        self._predictions: Dict[str, DiscriminatingPrediction] = {}
        self._prediction_evidence: Dict[str, Dict[str, Any]] = {}
        self._promoted_prediction_keys: set[str] = set()
        self._option_context_attempts: Counter[Tuple[str, int]] = Counter()
        self._decision_sources: Counter[str] = Counter()
        self._observed_transitions = 0
        self._operator_plans = 0
        self._theory_plans = 0
        self._safety_vetoes = 0
        self._generic_revisions = 0
        self._rules_promoted = 0
        self._promoted_option_decisions = 0
        self._option_preparation_decisions = 0
        self._option_outcomes: List[Dict[str, Any]] = []
        self._objective_outcomes: List[Dict[str, Any]] = []
        self._generated_goal_candidates = 0
        self._effect_conditioned_goal_candidates_generated = 0
        self._objective_experiment_decisions = 0
        self._objective_discriminator_decisions = 0
        self._objective_ablation_decisions = 0
        self._temporal_plan_decisions = 0
        self._temporal_outcomes: List[Dict[str, Any]] = []
        self._causal_subgoal_outcomes: List[Dict[str, Any]] = []
        self._causal_option_outcomes: List[Dict[str, Any]] = []

    @property
    def theory(self):
        return self.belief_loop.theory

    def seed_task_program(self, path: Any) -> None:
        """Feed the same Task Program to the scientific belief store."""
        self.belief_loop.seed_task_program(path)

    def register_predictions(
        self,
        predictions: Iterable[DiscriminatingPrediction],
    ) -> None:
        """Register candidate-only predictions without counting them as proof."""
        for prediction in predictions:
            existing = self._predictions.get(prediction.key)
            if existing is None:
                self._predictions[prediction.key] = prediction
            elif existing.status == HypothesisStatus.UNRESOLVED:
                # Keep live evidence/status local while allowing a stronger
                # epistemic prior to improve experiment ordering.
                self._predictions[prediction.key] = replace(
                    existing,
                    epistemic_prior=max(
                        existing.epistemic_prior,
                        prediction.epistemic_prior,
                    ),
                )

    def observe_transition(
        self,
        *,
        action: Any,
        grid_before: Any,
        grid_after: Any,
        available_actions: Sequence[Any] | None = None,
        game_state_before: str = "NOT_FINISHED",
        game_state_after: str = "NOT_FINISHED",
        levels_completed_before: int = 0,
        levels_completed_after: int = 0,
        action_data: Mapping[str, Any] | None = None,
    ) -> LiveTransitionUpdate:
        """Ingest one real transition into every consolidated memory layer."""
        action_name = _normalize_action(action)
        actions = _normalize_actions(available_actions or self.theory.actions())
        self.theory.seed_actions(actions)
        pending = self._pending_decision
        was_experiment = bool(
            pending is not None
            and pending.action_name == action_name
            and pending.source in {
                "discriminating_experiment",
                "relational_experiment",
                "terminal_objective_probe",
                "terminal_objective_discriminator",
                "terminal_objective_ablation",
                "temporal_subgoal_probe",
                "causal_option_downstream_probe",
                "causal_option_effect_subgoal_probe",
            }
        )
        aligned_before, aligned_after = _align_grids(grid_before, grid_after)
        update = self.belief_loop.observe_grids(
            action=action_name,
            action_args=dict(action_data or {}),
            grid_before=aligned_before,
            grid_after=aligned_after,
            available_actions=actions,
            game_state_before=game_state_before,
            game_state_after=game_state_after,
            levels_completed_before=levels_completed_before,
            levels_completed_after=levels_completed_after,
            timestamp=self._observed_transitions,
            was_experiment=was_experiment,
        )
        self._observed_transitions += 1

        if (
            self._observed_transitions
            % max(1, int(self.config.operator_induction_interval))
            == 0
        ):
            self.operator_inducer.induce(
                self.belief_loop.profiler,
                self.belief_loop.profiler.transitions,
            )

        before_hash = _grid_hash(grid_before)
        after_hash = _grid_hash(grid_after)
        is_noop = bool(update.record.diff.is_noop)
        if is_noop:
            self.anti_attractor.note_no_effect(before_hash, action_name)
        self.anti_attractor.observe(
            grid_hash=after_hash,
            action_name=action_name,
            is_noop=is_noop,
        )
        if update.record.diff.game_over:
            primitive = PrimitiveAction(
                action_name,
                x=_optional_int((action_data or {}).get("x")),
                y=_optional_int((action_data or {}).get("y")),
            )
            self.danger_memory.record_death(before_hash, action_key(primitive))

        operator_predicted_ok = False
        if pending is not None and pending.operator_id:
            operator = self.operator_inducer.operators.get(pending.operator_id)
            if operator is not None:
                operator_predicted_ok = _operator_prediction_matches(
                    operator,
                    update,
                )
                self.operator_inducer.record_validation(
                    pending.operator_id,
                    predicted_ok=operator_predicted_ok,
                    had_progress=bool(
                        update.record.diff.num_changed
                        or update.record.diff.level_complete
                    ),
                )

        self._revise_pending_predictions(update, pending)
        self._promote_confirmed_predictions()
        self._generate_effect_conditioned_goal_hypotheses(update, pending)
        self._observe_pending_terminal_objective(update, pending)
        self._observe_pending_temporal_plan(update, pending)
        self._observe_pending_causal_option(update, pending)
        self._observe_pending_option(update, pending)
        self.progress.on_action(
            grid_hash=after_hash,
            diff_signature=_diff_signature(update),
            macro_id=pending.operator_id if pending else None,
            is_noop=is_noop,
            game_over=bool(update.record.diff.game_over),
            player_moved=update.record.diff.player_displacement is not None,
            num_changed=int(update.record.diff.num_changed),
            objects=update.record.obs_after.objects,
            current_validated_ops=self.operator_inducer.num_locked(),
            current_validated_rules=self._validated_rule_count(),
            operator_predicted_ok=operator_predicted_ok,
            is_click=action_name == "ACTION6",
            is_transform=int(update.record.diff.num_changed) >= 5,
        )
        self._pending_decision = None
        return update

    def select_action(
        self,
        *,
        current_grid: Any,
        available_actions: Sequence[Any],
        legacy_action: Any,
        legacy_action_data: Mapping[str, Any] | None = None,
        game_state: str = "NOT_FINISHED",
        levels_completed: int = 0,
    ) -> CognitiveDecision:
        """Choose the next action through the consolidated decision path."""
        self._step += 1
        actions = _normalize_actions(available_actions)
        legacy_name = _normalize_action(legacy_action)
        if legacy_name not in actions and actions:
            legacy_name = actions[0]
        self.theory.seed_actions(actions)
        observation = build_observation(
            current_grid,
            available_actions=actions,
            game_state=game_state,
            levels_completed=levels_completed,
            infer_players=True,
        )
        safe_actions = self._safe_actions(observation.grid_hash, actions)
        if not safe_actions:
            safe_actions = list(actions)
        self._generate_goal_hypotheses(observation, safe_actions)
        if self.config.enable_causal_hierarchical_options:
            self.causal_options.sync_confirmed_edges(
                self.causal_subgoals.edges()
            )

        decision = self._select_escape(observation, safe_actions)
        if decision is None:
            decision = self._select_causal_option(observation, safe_actions)
        if decision is None:
            decision = self._select_temporal_plan(observation, safe_actions)
        if decision is None:
            decision = self._select_promoted_option(observation, safe_actions)
        if decision is None:
            decision = self._select_objective_experiment(observation, safe_actions)
        if decision is None:
            decision = self._select_theory_plan(observation, safe_actions)
        if decision is None and self._should_experiment():
            decision = self._select_experiment(observation, safe_actions)
        if decision is None:
            decision = self._select_operator_plan(observation, safe_actions)
        if decision is None and self._should_reprobe():
            decision = self._select_experiment(observation, safe_actions)
        if decision is None:
            decision = CognitiveDecision(
                action_name=legacy_name,
                action_data=dict(legacy_action_data or {}),
                source="legacy_fallback",
                reason="no confirmed theory plan or useful experiment/operator plan",
            )

        decision = self._guard_decision(decision, observation, safe_actions)
        self._pending_decision = decision
        self._decision_sources[decision.source] += 1
        return decision

    def on_reset(self) -> None:
        """Start a fresh behavioral branch while retaining learned theory."""
        self._pending_decision = None
        self.terminal_objectives.start_branch()
        self.temporal_goals.start_branch()
        self.causal_subgoals.start_branch()
        self.causal_options.start_branch()
        self.progress.start_new_branch(
            current_validated_ops=self.operator_inducer.num_locked(),
            current_validated_rules=self._validated_rule_count(),
        )

    def on_level_change(self) -> None:
        """Keep transferable mechanics but reset branch-local control state."""
        self.on_reset()

    def summary(self) -> Dict[str, Any]:
        prediction_statuses = Counter(
            prediction.status.value for prediction in self._predictions.values()
        )
        return {
            "execution_path": "unified_cognitive_controller",
            "transitions_observed": self._observed_transitions,
            "decision_sources": dict(self._decision_sources),
            "experiments_selected": self._experiment_decisions,
            "relational_experiments_selected": self._generic_experiment_decisions,
            "generic_hypothesis_revisions": self._generic_revisions,
            "generic_predictions": len(self._predictions),
            "generic_prediction_statuses": dict(prediction_statuses),
            "promoted_relational_rules": self._rules_promoted,
            "active_promoted_relational_rules": len(
                self.theory.promoted_relational_rules()
            ),
            "promoted_option_decisions": self._promoted_option_decisions,
            "option_preparation_decisions": self._option_preparation_decisions,
            "option_execution": self.option_memory.summary(),
            "recent_option_outcomes": self._option_outcomes[-10:],
            "terminal_objectives": self.terminal_objectives.summary(),
            "recent_objective_outcomes": self._objective_outcomes[-10:],
            "generated_goal_candidates": self._generated_goal_candidates,
            "effect_conditioned_goal_candidates_generated": (
                self._effect_conditioned_goal_candidates_generated
            ),
            "objective_experiment_decisions": self._objective_experiment_decisions,
            "objective_discriminator_decisions": (
                self._objective_discriminator_decisions
            ),
            "objective_ablation_decisions": self._objective_ablation_decisions,
            "temporal_goal_composition": self.temporal_goals.summary(),
            "temporal_plan_decisions": self._temporal_plan_decisions,
            "recent_temporal_outcomes": self._temporal_outcomes[-10:],
            "causal_subgoal_graph": self.causal_subgoals.summary(),
            "recent_causal_subgoal_outcomes": (
                self._causal_subgoal_outcomes[-10:]
            ),
            "causal_hierarchical_options": self.causal_options.summary(),
            "recent_causal_option_outcomes": self._causal_option_outcomes[-10:],
            "operators_induced": len(self.operator_inducer.operators),
            "operators_locked": self.operator_inducer.num_locked(),
            "operator_plans": self._operator_plans,
            "rules": len(self.belief_loop.rule_engine.rules),
            "high_confidence_rules": len(
                self.belief_loop.rule_engine.high_confidence_rules()
            ),
            "theory_plans": self._theory_plans,
            "theory": self.theory.summary(),
            "danger_records": len(self.danger_memory),
            "safety_vetoes": self._safety_vetoes,
            "progress": self.progress.summary(),
        }

    def _should_experiment(self) -> bool:
        return self._experiment_decisions < max(
            0,
            int(self.config.max_bootstrap_experiments),
        )

    def _should_reprobe(self) -> bool:
        interval = max(1, int(self.config.reprobe_interval))
        return self._step % interval == 0

    def _safe_actions(self, grid_hash: int, actions: Sequence[str]) -> List[str]:
        result = []
        for action in actions:
            if action == "RESET":
                continue
            if self.danger_memory.is_lethal(grid_hash, action):
                continue
            if self.anti_attractor.is_banned_noop(grid_hash, action):
                continue
            result.append(action)
        return result

    def _select_escape(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        branch_stalled = self.progress.should_kill_branch()
        if not branch_stalled and not self.anti_attractor.should_escape(self._step):
            return None
        action = self.anti_attractor.pick_escape_action(
            available=safe_actions,
            grid_hash=observation.grid_hash,
            is_lethal=self.danger_memory.is_lethal,
        )
        if action is None:
            return None
        self.anti_attractor.note_escape(self._step)
        if branch_stalled:
            self.progress.start_new_branch(
                current_validated_ops=self.operator_inducer.num_locked(),
                current_validated_rules=self._validated_rule_count(),
            )
        return CognitiveDecision(
            action_name=action,
            action_data=self._default_action_data(action, observation),
            source="anti_attractor_escape",
            reason="observed no-op/repetition/novelty stall",
            confidence=1.0,
        )

    def _generate_goal_hypotheses(
        self,
        observation: GameObservation,
        safe_actions: Sequence[str],
    ) -> None:
        """Create bounded measurable counterfactuals without granting proof."""
        if not self.config.enable_active_goal_hypotheses:
            return
        candidates = self.goal_generator.generate(
            observation=observation,
            rules=self.theory.promoted_relational_rules(),
            available_actions=safe_actions,
        )
        for candidate in candidates:
            self.terminal_objectives.register_generated_bounded(
                candidate,
                max_objectives=self.config.max_generated_goal_candidates,
                max_per_family=self.config.max_generated_goals_per_family,
            )
        self._generated_goal_candidates = len(self.terminal_objectives.objectives())

    def _generate_effect_conditioned_goal_hypotheses(
        self,
        update: LiveTransitionUpdate,
        pending: CognitiveDecision | None,
    ) -> None:
        """Generate downstream goals only from a participating observed effect."""
        if (
            not self.config.enable_effect_conditioned_downstream_subgoals
            or pending is None
            or update.record.diff.is_noop
        ):
            return
        active = self.causal_options.option(
            self.causal_options.active_option_id
        )
        if active is None:
            return
        participates = bool(
            pending.causal_option_id == active.option_id
            or pending.causal_option_edge_key == active.edge_key
            or pending.causal_subgoal_edge_key == active.edge_key
        )
        if not participates:
            return
        candidates = self.goal_generator.generate_from_transition(
            observation_before=update.record.obs_before,
            observation_after=update.record.obs_after,
            action_name=pending.action_name,
            rules=self.theory.promoted_relational_rules(),
        )
        for candidate in candidates:
            registered = self.terminal_objectives.register_generated_bounded(
                candidate,
                max_objectives=self.config.max_generated_goal_candidates,
                max_per_family=self.config.max_generated_goals_per_family,
            )
            if registered is not None:
                self._effect_conditioned_goal_candidates_generated += 1
        self._generated_goal_candidates = len(
            self.terminal_objectives.objectives()
        )

    def _select_causal_option(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        """Search a bounded suffix after an actually observed causal opening."""
        if not self.config.enable_causal_hierarchical_options:
            return None
        active_option = self.causal_options.option(
            self.causal_options.active_option_id
        )
        target_objective = (
            None if active_option is None
            else self.terminal_objectives.objective(
                active_option.target_objective_id
            )
        )
        click_actions = (
            self._click_actions(observation)
            if "ACTION6" in safe_actions else ()
        )
        downstream_subgoal = (
            self.causal_options.select_effect_conditioned_subgoal(
                observation,
                store=self.terminal_objectives,
                safe_actions=safe_actions,
                click_actions=click_actions,
            )
        )
        selected_objective = (
            self.terminal_objectives.objective(
                downstream_subgoal.objective_id
            )
            if downstream_subgoal is not None
            else target_objective
        )
        directed_choice = None
        if downstream_subgoal is not None:
            directed_choice = self.objective_experiment_designer.design(
                observation=observation,
                store=self.terminal_objectives,
                safe_actions=safe_actions,
                click_actions=click_actions,
                operators=self.operator_inducer.operators.values(),
                preferred_objective_id=downstream_subgoal.objective_id,
                allow_ablation=False,
                require_selectable=False,
            )
        preferred_signatures = list(
            self._causal_option_preferences(selected_objective)
        )
        if directed_choice is not None:
            directed_signature = semantic_intervention_signature(
                directed_choice.action_name,
                directed_choice.action_data,
                observation,
            )
            preferred_signatures.insert(0, directed_signature)
        preferred_signatures = list(dict.fromkeys(preferred_signatures))
        action_utilities = {}
        for action in safe_actions:
            stats = self.belief_loop.profiler.get_stats(action)
            if stats is None:
                continue
            action_utilities[action] = (
                float(stats.change_rate) - 2.0 * float(stats.death_rate)
            )
        directional_predictions = (
            self.causal_options.directional_action_predictions(
                observation,
                store=self.terminal_objectives,
                downstream_subgoal=downstream_subgoal,
                safe_actions=safe_actions,
                click_actions=click_actions,
            )
        )
        selection = self.causal_options.select_downstream(
            observation,
            safe_actions=safe_actions,
            click_actions=click_actions,
            preferred_intervention_signatures=preferred_signatures,
            action_utilities=action_utilities,
            downstream_subgoal=downstream_subgoal,
            preferred_action_name=(
                "" if directed_choice is None else directed_choice.action_name
            ),
            preferred_action_data=(
                None if directed_choice is None else directed_choice.action_data
            ),
            directional_predictions=directional_predictions,
        )
        if selection is None:
            return None
        objective = self.terminal_objectives.objective(
            selection.downstream_objective_id
            or selection.target_objective_id
        )
        assessment = (
            None
            if objective is None
            else self.terminal_objectives.assess_objective(
                objective,
                observation,
            )
        )
        action_intervention_id = intervention_id(
            selection.action_name,
            selection.action_data,
        )
        source = (
            "causal_option_terminal_replay"
            if selection.replaying_terminal_sequence
            else (
                "causal_option_effect_subgoal_probe"
                if selection.downstream_subgoal_id
                else "causal_option_downstream_probe"
            )
        )
        directed_action_selected = bool(
            directed_choice is not None
            and selection.action_name == directed_choice.action_name
            and dict(selection.action_data) == dict(directed_choice.action_data)
        )
        return CognitiveDecision(
            action_name=selection.action_name,
            action_data=dict(selection.action_data),
            source=source,
            reason=(
                f"{selection.reason}; confirmed edge {selection.edge_key}; "
                "execute one suffix action then re-observe"
            ),
            confidence=(
                0.9
                if selection.replaying_terminal_sequence
                else (0.65 if selection.downstream_subgoal_id else 0.5)
            ),
            objective_id=(
                selection.downstream_objective_id
                or selection.target_objective_id
            ),
            objective_status=(
                "" if assessment is None else assessment.status.value
            ),
            objective_distance=(
                None if assessment is None else assessment.distance
            ),
            intervention_id=action_intervention_id,
            predicted_goal_reductions=(
                directed_choice.predicted_reduction_objective_ids
                if directed_action_selected and directed_choice is not None
                else ()
            ),
            causal_option_id=selection.option_id,
            causal_option_edge_key=selection.edge_key,
            causal_option_terminal_status=selection.terminal_status.value,
            causal_option_phase=(
                "effect_conditioned_subgoal"
                if selection.downstream_subgoal_id
                else "downstream_search"
            ),
            causal_option_intervention_signature=(
                selection.intervention_signature
            ),
            causal_option_intervention_transfer_signature=(
                selection.intervention_transfer_signature
            ),
            causal_option_intervention_entity_signature=(
                selection.intervention_entity_signature
            ),
            causal_option_intervention_structural_role_signature=(
                selection.intervention_structural_role_signature
            ),
            causal_option_intervention_instance_signature=(
                selection.intervention_instance_signature
            ),
            causal_option_entity_anchored_intervention=(
                selection.entity_anchored_intervention
            ),
            causal_option_selection_utility=selection.selection_utility,
            causal_option_replaying_terminal_sequence=(
                selection.replaying_terminal_sequence
            ),
            causal_option_downstream_subgoal_id=(
                selection.downstream_subgoal_id
            ),
            causal_option_downstream_subgoal_status=(
                selection.downstream_subgoal_status
            ),
            causal_option_trigger_effect_signature=(
                selection.downstream_trigger_effect_signature
            ),
            causal_option_downstream_subgoal_utility=(
                None
                if downstream_subgoal is None
                else downstream_subgoal.utility
            ),
            causal_option_replaying_progress_sequence=(
                selection.replaying_progress_sequence
            ),
            causal_option_latent_mode_signature=(
                selection.latent_mode_signature
            ),
            causal_option_directional_effect_status=(
                selection.directional_effect_status
            ),
            causal_option_directional_expected_gain=(
                selection.directional_expected_gain
            ),
            causal_option_directional_confidence=(
                selection.directional_confidence
            ),
            causal_option_directionally_compatible=(
                selection.directionally_compatible
            ),
            causal_option_reversible_action=selection.reversible_action,
            causal_option_directional_mode_contrast=(
                selection.directional_mode_contrast
            ),
            causal_option_directional_structural_transfer=(
                selection.directional_structural_transfer
            ),
            causal_option_directional_entity_alias_conflict=(
                selection.directional_entity_alias_conflict
            ),
            causal_option_directional_bridge_target_mode_signature=(
                selection.directional_bridge_target_mode_signature
            ),
            causal_option_directional_bridge_followup_action_signature=(
                selection.directional_bridge_followup_action_signature
            ),
            causal_option_persistent_pursuit=(
                selection.persistent_pursuit
            ),
            causal_option_persistent_attempt_index=(
                selection.persistent_attempt_index
            ),
            causal_option_persistent_action_limit=(
                selection.persistent_action_limit
            ),
        )

    def _causal_option_preferences(
        self,
        objective: Any,
    ) -> Tuple[str, ...]:
        """Reuse directed mechanics without treating them as terminal proof."""
        if objective is None:
            return ()
        result = []
        for rule_key in sorted(objective.supporting_rule_keys):
            rule = self.theory.promoted_relational_rule(rule_key)
            if rule is None:
                continue
            if rule.action == "ACTION6":
                signature = f"ACTION6::color:{int(rule.source_color)}"
            else:
                signature = str(rule.action)
            if signature not in result:
                result.append(signature)
        if result:
            return tuple(result)
        for action in sorted(objective.supporting_actions):
            if action == "ACTION6":
                color = (
                    objective.target_color
                    if objective.target_color is not None
                    else objective.source_color
                )
                if color is not None:
                    result.append(f"ACTION6::color:{int(color)}")
            elif action not in result:
                result.append(action)
        return tuple(result)

    def _select_temporal_plan(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        """Execute one guarded subgoal intervention, then force re-observation."""
        if (
            not self.config.enable_active_goal_hypotheses
            or not self.config.enable_temporal_goal_composition
        ):
            return None
        causal_edges = (
            self.causal_subgoals.candidate_edges(
                observation,
                self.terminal_objectives,
            )
            if self.config.enable_causal_subgoal_induction else ()
        )
        self.temporal_goals.compose(
            observation,
            self.terminal_objectives,
            causal_edges=causal_edges,
        )
        selection = self.temporal_goals.select_step(
            observation,
            self.terminal_objectives,
        )
        if selection is None:
            return None
        causal_intervention_utilities = (
            self.causal_subgoals.intervention_utilities(
                selection.causal_edge_key
            )
            if selection.causal_edge_key else {}
        )
        choice = self.objective_experiment_designer.design(
            observation=observation,
            store=self.terminal_objectives,
            safe_actions=safe_actions,
            click_actions=(
                self._click_actions(observation)
                if "ACTION6" in safe_actions else ()
            ),
            operators=self.operator_inducer.operators.values(),
            preferred_objective_id=selection.objective_id,
            intervention_utilities=causal_intervention_utilities,
            allow_ablation=False,
            require_selectable=False,
        )
        if choice is None:
            if self.config.enable_causal_subgoal_induction:
                self.causal_subgoals.note_intervention_availability(
                    selection.objective_id,
                    available=False,
                    observation=observation,
                    store=self.terminal_objectives,
                    context_signature=(
                        f"branch:{self.terminal_objectives.branch_index}:"
                        f"blocked:{observation.grid_hash}"
                    ),
                )
                self.causal_subgoals.note_blocked(
                    selection.objective_id,
                    observation,
                    self.terminal_objectives,
                )
            self.temporal_goals.reject_active_step(
                "no_safe_intervention_for_enabled_guard"
            )
            return None

        if self.config.enable_causal_subgoal_induction:
            causal_context = (
                f"branch:{self.terminal_objectives.branch_index}:"
                f"state:{observation.grid_hash}"
            )
            opened_edge_keys = self.causal_subgoals.note_intervention_availability(
                selection.objective_id,
                available=True,
                observation=observation,
                store=self.terminal_objectives,
                context_signature=causal_context,
            )
            if self.config.enable_causal_hierarchical_options:
                self.causal_options.sync_confirmed_edges(
                    self.causal_subgoals.edges()
                )
                self.causal_options.note_openings(
                    opened_edge_keys,
                    context_signature=causal_context,
                )
            if selection.causal_edge_key and selection.step_index == 0:
                self.causal_subgoals.begin_trial(
                    selection.causal_edge_key,
                    context_signature=causal_context,
                )

        mechanic_hypotheses: Tuple[str, ...] = ()
        if choice.action_name == "ACTION6":
            concrete = DesignedExperimentAction(
                name=choice.action_name,
                raw_action=choice.action_name,
                action_args=dict(choice.action_data),
            )
            self.register_predictions(self._live_predictions(observation, [concrete]))
            mechanic_choice = self.generic_designer.design(
                hypotheses=list(self._predictions.values()),
                live_grid=observation.raw_grid,
                available_actions=[concrete],
            )
            if mechanic_choice is not None:
                mechanic_hypotheses = tuple(mechanic_choice.competing_keys)

        objective = self.terminal_objectives.objective(selection.objective_id)
        assessment = (
            None
            if objective is None
            else self.terminal_objectives.assess_objective(
                objective,
                observation,
            )
        )
        if objective is not None:
            self.terminal_objectives.record_selection(
                objective.objective_id,
                observation,
                is_probe=choice.is_probe,
                intervention_id=choice.intervention_id,
            )
        self._temporal_plan_decisions += 1
        source = (
            "temporal_subgoal_option"
            if selection.plan_status == TemporalPlanStatus.TERMINAL_SUPPORTED
            else "temporal_subgoal_probe"
        )
        causal_intervention = (
            semantic_intervention_signature(
                choice.action_name,
                choice.action_data,
                observation,
            )
            if selection.causal_edge_key else ""
        )
        return CognitiveDecision(
            action_name=choice.action_name,
            action_data=dict(choice.action_data),
            source=source,
            reason=(
                f"{selection.reason}; {choice.reason}; "
                "execute one intervention then re-observe"
            ),
            confidence=(
                0.9
                if selection.plan_status == TemporalPlanStatus.TERMINAL_SUPPORTED
                else 0.45
            ),
            competing_hypotheses=choice.competing_objective_ids,
            mechanic_hypotheses=mechanic_hypotheses,
            objective_id=selection.objective_id,
            objective_status=(
                "" if assessment is None else assessment.status.value
            ),
            objective_distance=selection.current_distance,
            intervention_id=choice.intervention_id,
            predicted_goal_reductions=(
                choice.predicted_reduction_objective_ids
            ),
            temporal_plan_id=selection.plan_id,
            temporal_target_objective_id=selection.target_objective_id,
            temporal_plan_status=selection.plan_status.value,
            temporal_step_index=selection.step_index,
            temporal_step_count=selection.step_count,
            temporal_step_target_distance=selection.target_distance,
            causal_subgoal_edge_key=selection.causal_edge_key,
            temporal_expected_progress_probability=(
                selection.expected_progress_probability
            ),
            temporal_expected_cost=selection.expected_cost,
            temporal_selection_utility=selection.selection_utility,
            causal_intervention_signature=causal_intervention,
            causal_intervention_utility=(
                causal_intervention_utilities.get(causal_intervention, 0.0)
                if causal_intervention else None
            ),
            causal_confirmation_trial=(
                selection.causal_confirmation_priority
            ),
        )

    def _select_objective_experiment(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        """Discriminate generated goals before spending mechanic-only probes."""
        if not self.config.enable_active_goal_hypotheses:
            return None
        choice = self.objective_experiment_designer.design(
            observation=observation,
            store=self.terminal_objectives,
            safe_actions=safe_actions,
            click_actions=(
                self._click_actions(observation)
                if "ACTION6" in safe_actions else ()
            ),
            operators=self.operator_inducer.operators.values(),
        )
        if choice is None:
            return None
        mechanic_hypotheses: Tuple[str, ...] = ()
        if choice.action_name == "ACTION6":
            concrete = DesignedExperimentAction(
                name=choice.action_name,
                raw_action=choice.action_name,
                action_args=dict(choice.action_data),
            )
            self.register_predictions(
                self._live_predictions(observation, [concrete])
            )
            mechanic_choice = self.generic_designer.design(
                hypotheses=list(self._predictions.values()),
                live_grid=observation.raw_grid,
                available_actions=[concrete],
            )
            if mechanic_choice is not None:
                mechanic_hypotheses = tuple(mechanic_choice.competing_keys)
        objective = self.terminal_objectives.objective(choice.objective_id)
        assessment = (
            None
            if objective is None
            else self.terminal_objectives.assess_objective(
                objective,
                observation,
            )
        )
        if objective is not None and assessment is not None:
            self.terminal_objectives.record_selection(
                objective.objective_id,
                observation,
                is_probe=choice.is_probe,
                intervention_id=choice.intervention_id,
            )

        self._objective_experiment_decisions += 1
        if choice.ablation_of_objective_id:
            source = "terminal_objective_ablation"
            self._objective_ablation_decisions += 1
        elif assessment is not None and (
            assessment.status == TerminalObjectiveStatus.TERMINAL_SUPPORTED
        ):
            source = "terminal_objective_option"
        elif len(choice.competing_objective_ids) >= 2:
            source = "terminal_objective_discriminator"
            self._objective_discriminator_decisions += 1
        else:
            source = "terminal_objective_probe"
        return CognitiveDecision(
            action_name=choice.action_name,
            action_data=dict(choice.action_data),
            source=source,
            reason=(
                f"{choice.reason}; predicted reductions="
                f"{choice.predicted_reduction_objective_ids}"
            ),
            confidence=max(
                0.0,
                min(1.0, float(choice.expected_divergence) / 3.0),
            ),
            competing_hypotheses=choice.competing_objective_ids,
            mechanic_hypotheses=mechanic_hypotheses,
            objective_id=choice.objective_id,
            objective_status=(
                "" if assessment is None else assessment.status.value
            ),
            objective_distance=(
                None if assessment is None else assessment.distance
            ),
            intervention_id=choice.intervention_id,
            ablation_of_objective_id=choice.ablation_of_objective_id,
            predicted_goal_reductions=(
                choice.predicted_reduction_objective_ids
            ),
        )

    def _select_promoted_option(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        """Probe or exploit options only through a measurable goal deficit."""
        if not self.config.enable_promoted_options:
            return None
        rules = self.theory.promoted_relational_rules()
        if not rules:
            return None
        options = self.option_compiler.compile(rules)
        if not options:
            return None
        by_key = {option.rule_key: option for option in options}
        self.terminal_objectives.seed_rules(rules)
        candidates: List[
            Tuple[
                PromotedRelationalRule,
                CompiledRelationalOption,
                TerminalObjectiveAssessment,
            ]
        ] = []
        for rule in rules:
            if self.option_memory.is_sterile(
                rule.key,
                min_executions=(
                    self.config.min_option_executions_before_quarantine
                ),
            ):
                continue
            option = by_key.get(rule.key)
            if option is None or option.action not in safe_actions:
                continue
            objective = self.terminal_objectives.assess_rule(rule, observation)
            if (
                objective is None
                or not objective.selectable
                or (
                    self.config.enable_active_goal_hypotheses
                    and objective.status
                    != TerminalObjectiveStatus.TERMINAL_SUPPORTED
                )
            ):
                continue
            candidates.append((rule, option, objective))

        candidates.sort(
            key=lambda item: (
                item[2].priority,
                self.option_memory.value(item[0]),
                item[0].level_successes,
                item[0].confidence,
                item[0].key,
            ),
            reverse=True,
        )
        for rule, option, objective in candidates:
            context_key = (rule.key, observation.grid_hash)
            if self._option_context_attempts[context_key] >= max(
                1,
                int(self.config.max_option_attempts_per_context),
            ):
                continue
            assessment = self.option_compiler.assess(option, observation)
            if assessment.already_satisfied:
                continue
            if assessment.ready:
                return self._option_decision(
                    option,
                    observation,
                    preparation_for_rule_key="",
                    objective=objective,
                )
            chain = self.option_compiler.preparation_chain(
                option,
                options,
                observation,
            )
            if chain:
                preparer = chain[0]
                if preparer.action in safe_actions:
                    return self._option_decision(
                        preparer,
                        observation,
                        preparation_for_rule_key=rule.key,
                        objective=objective,
                    )
            preparer = self._select_induced_precondition_operator(
                option,
                assessment,
                objective,
                observation,
                safe_actions,
            )
            if preparer is not None:
                return preparer
        return None

    def _option_decision(
        self,
        option: CompiledRelationalOption,
        observation: GameObservation,
        *,
        preparation_for_rule_key: str,
        objective: TerminalObjectiveAssessment,
    ) -> CognitiveDecision:
        rule = self.theory.promoted_relational_rule(option.rule_key)
        confidence = 0.0 if rule is None else rule.confidence
        self._option_context_attempts[(option.rule_key, observation.grid_hash)] += 1
        operator = self._best_operator_for_option(option, observation)
        action_data = self._option_action_data(option, observation, operator)
        selected_intervention = intervention_id(option.action, action_data)
        source = "terminal_objective_preparation" if preparation_for_rule_key else (
            "terminal_objective_probe"
            if objective.is_probe
            else "terminal_objective_option"
        )
        self.terminal_objectives.record_selection(
            objective.objective_id,
            observation,
            is_probe=objective.is_probe,
            intervention_id=selected_intervention,
        )
        if preparation_for_rule_key:
            self._option_preparation_decisions += 1
        else:
            self._promoted_option_decisions += 1
        return CognitiveDecision(
            action_name=option.action,
            action_data=action_data,
            source=source,
            reason=(
                f"{objective.reason}: {objective.objective_id} via "
                f"compiled confirmed rule {option.rule_key}"
                + (
                    f" prepares {preparation_for_rule_key}"
                    if preparation_for_rule_key
                    else ""
                )
            ),
            confidence=confidence,
            operator_id="" if operator is None else operator.operator_id,
            source_rule_key=option.rule_key,
            option_id=option.option_id,
            preparation_for_rule_key=preparation_for_rule_key,
            objective_id=objective.objective_id,
            objective_status=objective.status.value,
            objective_distance=objective.distance,
            intervention_id=selected_intervention,
            predicted_goal_reductions=(objective.objective_id,),
        )

    def _best_operator_for_option(
        self,
        option: CompiledRelationalOption,
        observation: GameObservation,
    ) -> Operator | None:
        candidates = [
            operator
            for operator in self.operator_inducer.operators.values()
            if _normalize_action(operator.primitive_action) == option.action
            and operator.preconditions_met(observation)
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda operator: (
                int(
                    operator.parameters.get("target_value")
                    == option.source_color
                ),
                operator.confidence,
                operator.support,
            ),
        )

    def _option_action_data(
        self,
        option: CompiledRelationalOption,
        observation: GameObservation,
        operator: Operator | None,
    ) -> Dict[str, Any]:
        if option.action != "ACTION6":
            return {}
        candidates = [
            obj for obj in observation.objects
            if int(obj.value) == int(option.source_color)
        ]
        if candidates:
            target = min(candidates, key=lambda obj: (obj.area, obj.object_id))
            return {
                "x": int(round(target.center[1])),
                "y": int(round(target.center[0])),
            }
        if operator is not None:
            return _operator_action_data(operator, operator.parameters, observation)
        return self._default_action_data(option.action, observation)

    def _select_induced_precondition_operator(
        self,
        target_option: CompiledRelationalOption,
        assessment: OptionAssessment,
        objective: TerminalObjectiveAssessment,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        missing_colors = {
            int(predicate.split("_")[1])
            for predicate in assessment.missing_predicates
            if predicate.startswith("color_") and predicate.endswith("_present")
            and predicate.split("_")[1].isdigit()
        }
        if not missing_colors:
            return None
        candidates: List[Operator] = []
        for operator in self.operator_inducer.operators.values():
            action = _normalize_action(operator.primitive_action)
            if action not in safe_actions or not operator.preconditions_met(observation):
                continue
            creates_missing = any(
                effect.__class__.__name__ == "CreatesObject"
                and getattr(effect, "value", None) in missing_colors
                for effect in operator.expected_effects
            )
            if creates_missing:
                candidates.append(operator)
        if not candidates:
            return None
        operator = max(
            candidates,
            key=lambda item: (item.confidence, item.support, -item.cost_estimate),
        )
        self._option_preparation_decisions += 1
        action = _normalize_action(operator.primitive_action)
        action_data = _operator_action_data(
            operator,
            operator.parameters,
            observation,
        )
        selected_intervention = intervention_id(action, action_data)
        self.terminal_objectives.record_selection(
            objective.objective_id,
            observation,
            is_probe=objective.is_probe,
            intervention_id=selected_intervention,
        )
        return CognitiveDecision(
            action_name=action,
            action_data=action_data,
            source="terminal_objective_preparation",
            reason=(
                f"induced operator {operator.operator_id} prepares "
                f"{target_option.rule_key}: {assessment.missing_predicates}"
            ),
            confidence=operator.confidence,
            operator_id=operator.operator_id,
            option_id=target_option.option_id,
            preparation_for_rule_key=target_option.rule_key,
            objective_id=objective.objective_id,
            objective_status=objective.status.value,
            objective_distance=objective.distance,
            intervention_id=selected_intervention,
            predicted_goal_reductions=(objective.objective_id,),
        )

    def _select_theory_plan(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        if not self.config.enable_theory_planning:
            return None
        plan = self.theory_planner.plan(self.theory, safe_actions)
        if plan is None or not plan.planned_actions:
            return None
        planned = plan.planned_actions[0]
        action = _normalize_action(planned.action)
        if action not in safe_actions:
            return None
        self._theory_plans += 1
        return CognitiveDecision(
            action_name=action,
            action_data=self._default_action_data(action, observation),
            source="theory_plan",
            reason=planned.reason,
            confidence=1.0,
            source_rule_key=planned.rule_key,
        )

    def _select_experiment(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        choice = self.experiment_designer.design(self.theory, safe_actions)
        if choice is None:
            return None
        self._experiment_decisions += 1
        if (
            choice.action == "ACTION6"
            and self.config.enable_relational_experiments
        ):
            relational = self._select_relational_experiment(
                observation,
                safe_actions,
            )
            if relational is not None:
                return relational
        return CognitiveDecision(
            action_name=choice.action,
            action_data=self._default_action_data(choice.action, observation),
            source="discriminating_experiment",
            reason=choice.rationale,
            confidence=max(0.0, min(1.0, choice.expected_divergence / 2.0)),
            competing_hypotheses=tuple(choice.competing_keys),
        )

    def _select_relational_experiment(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        if "ACTION6" not in safe_actions:
            return None
        concrete = self._click_actions(observation)
        if not concrete:
            return None
        self.register_predictions(
            self._live_predictions(observation, concrete)
        )
        choice = self.generic_designer.design(
            hypotheses=list(self._predictions.values()),
            live_grid=observation.raw_grid,
            available_actions=concrete,
        )
        if choice is None:
            return None
        self._generic_experiment_decisions += 1
        return CognitiveDecision(
            action_name=choice.action.name,
            action_data=dict(choice.action.action_args),
            source="relational_experiment",
            reason=(
                f"{choice.selection_reason}: {choice.divergence_reason}"
            ),
            confidence=max(0.0, min(1.0, choice.expected_divergence / 4.0)),
            competing_hypotheses=tuple(choice.competing_keys),
        )

    def _select_operator_plan(
        self,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision | None:
        if not self.config.enable_operator_planning:
            return None
        if not self.operator_inducer.operators:
            return None
        target = _nearest_non_player_target(observation)
        trace = self.operator_searcher.search(
            observation,
            self.operator_inducer,
            self.belief_loop.rule_engine,
            target=target,
        )
        if not trace:
            return None
        call = trace[0]
        operator = self.operator_inducer.operators.get(call.operator_id)
        if operator is None or not operator.primitive_action:
            return None
        action = _normalize_action(operator.primitive_action)
        if action not in safe_actions:
            return None
        self._operator_plans += 1
        return CognitiveDecision(
            action_name=action,
            action_data=_operator_action_data(operator, call.args, observation),
            source="operator_plan",
            reason=(
                f"induced operator {operator.operator_id} "
                f"(confidence={operator.confidence:.3f})"
            ),
            confidence=float(operator.confidence),
            operator_id=operator.operator_id,
        )

    def _guard_decision(
        self,
        decision: CognitiveDecision,
        observation: GameObservation,
        safe_actions: List[str],
    ) -> CognitiveDecision:
        primitive = PrimitiveAction(
            decision.action_name,
            x=_optional_int(decision.action_data.get("x")),
            y=_optional_int(decision.action_data.get("y")),
        )
        lethal = self.danger_memory.is_lethal(
            observation.grid_hash,
            action_key(primitive),
        )
        banned_noop = self.anti_attractor.is_banned_noop(
            observation.grid_hash,
            decision.action_name,
        )
        if not lethal and not banned_noop:
            return decision
        self._safety_vetoes += 1
        alternatives = [
            action for action in safe_actions
            if action != decision.action_name
        ]
        alternative = self.anti_attractor.pick_escape_action(
            available=alternatives,
            grid_hash=observation.grid_hash,
            is_lethal=self.danger_memory.is_lethal,
        )
        if alternative is None:
            return decision
        if decision.temporal_plan_id:
            self.temporal_goals.reject_active_step("safety_veto")
        if decision.causal_subgoal_edge_key:
            self.causal_subgoals.cancel_trial(
                decision.causal_subgoal_edge_key,
                count_failure=True,
            )
        return CognitiveDecision(
            action_name=alternative,
            action_data=self._default_action_data(alternative, observation),
            source="safety_veto",
            reason=f"vetoed observed unsafe decision from {decision.source}",
            confidence=1.0,
            competing_hypotheses=decision.competing_hypotheses,
        )

    def _default_action_data(
        self,
        action: str,
        observation: GameObservation,
    ) -> Dict[str, Any]:
        if action != "ACTION6":
            return {}
        clicks = self._click_actions(observation)
        if not clicks:
            height, width = observation.raw_grid.shape
            return {"x": int(width // 2), "y": int(height // 2)}
        selected = clicks[self._click_cursor % len(clicks)]
        self._click_cursor += 1
        return dict(selected.action_args)

    def _click_actions(
        self,
        observation: GameObservation,
    ) -> List[DesignedExperimentAction]:
        actions: List[DesignedExperimentAction] = []
        seen: set[Tuple[int, int]] = set()
        objects = sorted(
            observation.objects,
            key=lambda obj: (obj.area, obj.value, obj.object_id),
        )
        for obj in objects:
            x = int(round(obj.center[1]))
            y = int(round(obj.center[0]))
            coord = (x, y)
            if coord in seen:
                continue
            seen.add(coord)
            actions.append(DesignedExperimentAction(
                name="ACTION6",
                raw_action="ACTION6",
                action_args={"x": x, "y": y},
            ))
            if len(actions) >= max(1, int(self.config.max_click_targets)):
                break
        return actions

    def _live_predictions(
        self,
        observation: GameObservation,
        click_actions: Sequence[DesignedExperimentAction],
    ) -> List[DiscriminatingPrediction]:
        grid = observation.raw_grid
        background = _background_value(grid)
        color_counts = Counter(int(value) for value in grid.ravel())
        target_colors = [
            color for color, _ in color_counts.most_common()
            if color != background
        ][: max(1, int(self.config.max_relation_target_colors))]
        predictions: List[DiscriminatingPrediction] = []
        for action in click_actions:
            x = int(action.action_args["x"])
            y = int(action.action_args["y"])
            source = int(grid[y, x])
            for outcome in ("local", "global"):
                predictions.append(DiscriminatingPrediction(
                    key=f"effect_scope::ACTION6::source{source}::{outcome}",
                    action="ACTION6",
                    source_color=source,
                    family="effect_scope",
                    predicate="effect_scope",
                    predicted_outcome=outcome,
                ))
            for outcome in ("changed", "stable"):
                predictions.append(DiscriminatingPrediction(
                    key=f"object_count::ACTION6::source{source}::{outcome}",
                    action="ACTION6",
                    source_color=source,
                    family="object_count",
                    predicate="object_count",
                    predicted_outcome=outcome,
                ))
            for target in target_colors:
                if target == source:
                    continue
                predictions.append(DiscriminatingPrediction(
                    key=f"color_transform::ACTION6::source{source}::{source}_{target}",
                    action="ACTION6",
                    source_color=source,
                    target_color=target,
                    family="color_transform",
                    predicate="source_target_color_transform",
                    predicted_outcome=f"{source}->{target}",
                ))
                for predicate in (
                    "same_shape",
                    "aligned_with",
                    "adjacent_to",
                    "paired_with",
                ):
                    holds = _relation_holds(
                        observation,
                        predicate,
                        source,
                        target,
                    )
                    outcomes = (
                        ("preserved", "broken")
                        if holds
                        else ("appears", "absent")
                    )
                    for outcome in outcomes:
                        predictions.append(DiscriminatingPrediction(
                            key=(
                                f"relation::ACTION6::{predicate}::"
                                f"colors{source}_{target}::{outcome}"
                            ),
                            action="ACTION6",
                            source_color=source,
                            target_color=target,
                            family="relation",
                            predicate=predicate,
                            predicted_outcome=outcome,
                        ))
        return _dedupe_predictions(predictions)

    def _revise_pending_predictions(
        self,
        update: LiveTransitionUpdate,
        pending: CognitiveDecision | None,
    ) -> None:
        if pending is None or pending.source not in {
            "relational_experiment",
            "terminal_objective_probe",
            "terminal_objective_discriminator",
            "terminal_objective_ablation",
            "temporal_subgoal_probe",
            "temporal_subgoal_option",
        }:
            return
        prediction_keys = (
            pending.competing_hypotheses
            if pending.source == "relational_experiment"
            else pending.mechanic_hypotheses
        )
        for key in prediction_keys:
            prediction = self._predictions.get(key)
            if prediction is None:
                continue
            observed = _observe_prediction_outcome(update, prediction)
            if observed == "unobservable":
                continue
            evidence = self._prediction_evidence.setdefault(
                key,
                {
                    "support": 0,
                    "contradictions": 0,
                    "experiments": 0,
                    "contexts": set(),
                },
            )
            evidence["experiments"] += 1
            evidence["contexts"].add(
                _transition_context_signature(update, pending.action_data)
            )
            if observed == prediction.outcome:
                evidence["support"] += 1
            else:
                evidence["contradictions"] += 1
            status = HypothesisStatus.UNRESOLVED
            if (
                evidence["support"]
                >= max(1, int(self.config.generic_confirmation_evidence))
                and evidence["support"] > evidence["contradictions"]
            ):
                status = HypothesisStatus.CONFIRMED
            elif (
                evidence["contradictions"]
                >= max(1, int(self.config.generic_refutation_evidence))
                and evidence["contradictions"] >= evidence["support"]
            ):
                status = HypothesisStatus.REFUTED
            self._predictions[key] = replace(prediction, status=status)
            self._generic_revisions += 1

    def _promote_confirmed_predictions(self) -> None:
        """Move live-confirmed predictions into GameTheory after the gate."""
        for key, prediction in self._predictions.items():
            if key in self._promoted_prediction_keys:
                continue
            if prediction.status != HypothesisStatus.CONFIRMED:
                continue
            evidence = self._prediction_evidence.get(key, {})
            contexts = set(evidence.get("contexts", set()) or set())
            support = int(evidence.get("support", 0) or 0)
            if support < max(1, int(self.config.promotion_min_support)):
                continue
            if len(contexts) < max(
                1,
                int(self.config.promotion_min_independent_contexts),
            ):
                continue
            rule = PromotedRelationalRule.from_prediction(
                prediction,
                support=support,
                contradictions=int(evidence.get("contradictions", 0) or 0),
                experiments_spent=int(evidence.get("experiments", 0) or 0),
                independent_contexts=contexts,
                minimum_support=self.config.promotion_min_support,
                minimum_independent_contexts=(
                    self.config.promotion_min_independent_contexts
                ),
            )
            if rule.status != HypothesisStatus.CONFIRMED:
                continue
            self.theory.add_promoted_relational_rule(rule)
            self._promoted_prediction_keys.add(key)
            self._rules_promoted += 1

    def _observe_pending_terminal_objective(
        self,
        update: LiveTransitionUpdate,
        pending: CognitiveDecision | None,
    ) -> None:
        """Keep mechanic effects separate from observed terminal credit."""
        context = "" if pending is None else _transition_context_signature(
            update,
            pending.action_data,
        )
        outcome = self.terminal_objectives.observe_transition(
            update,
            objective_id="" if pending is None else pending.objective_id,
            rule_key="" if pending is None else pending.source_rule_key,
            intervention_id="" if pending is None else pending.intervention_id,
            ablation_of_objective_id=(
                "" if pending is None else pending.ablation_of_objective_id
            ),
            predicted_objective_ids=(
                () if pending is None else pending.predicted_goal_reductions
            ),
            context_signature=context,
        )
        if (
            outcome["objective_id"]
            or outcome["terminal_success"]
            or outcome["expired_nonterminal_completions"]
        ):
            self._objective_outcomes.append(outcome)

    def _observe_pending_temporal_plan(
        self,
        update: LiveTransitionUpdate,
        pending: CognitiveDecision | None,
    ) -> None:
        """Revise ordered guards while reserving plan value for terminals."""
        context = "" if pending is None else _transition_context_signature(
            update,
            pending.action_data,
        )
        outcome = self.temporal_goals.observe_transition(
            update,
            store=self.terminal_objectives,
            plan_id="" if pending is None else pending.temporal_plan_id,
            step_index=(
                None if pending is None else pending.temporal_step_index
            ),
            objective_id="" if pending is None else pending.objective_id,
            target_distance=(
                None
                if pending is None
                else pending.temporal_step_target_distance
            ),
            intervention_id="" if pending is None else pending.intervention_id,
            context_signature=context,
        )
        causal_outcome = self.causal_subgoals.observe_transition(
            update,
            store=self.terminal_objectives,
            source_objective_id=(
                "" if pending is None else pending.objective_id
            ),
            edge_key=(
                "" if pending is None else pending.causal_subgoal_edge_key
            ),
            source_step_completed=bool(
                pending is not None
                and pending.temporal_step_index == 0
                and outcome["step_completed"]
            ),
            plan_abandoned=bool(outcome["plan_abandoned"]),
            context_signature=context,
            intervention_signature=(
                "" if pending is None
                else pending.causal_intervention_signature
            ),
        )
        if (
            causal_outcome["edge_key"]
            or causal_outcome["cochange_supported_edges"]
        ):
            self._causal_subgoal_outcomes.append(causal_outcome)
        if (
            outcome["plan_id"]
            or outcome["terminal_success"]
            or outcome["expired_nonterminal_completions"]
        ):
            self._temporal_outcomes.append(outcome)

    def _observe_pending_causal_option(
        self,
        update: LiveTransitionUpdate,
        pending: CognitiveDecision | None,
    ) -> None:
        if not self.config.enable_causal_hierarchical_options:
            return
        context = "" if pending is None else _transition_context_signature(
            update,
            pending.action_data,
        )
        option_intervention_signature = ""
        if pending is not None and (
            pending.causal_option_id
            or pending.causal_option_edge_key
            or pending.causal_subgoal_edge_key
        ):
            option_intervention_signature = (
                pending.causal_option_intervention_signature
                or self.causal_options.intervention_anchor(
                    pending.action_name,
                    pending.action_data,
                    update.record.obs_before,
                ).concrete_signature
            )
        outcome = self.causal_options.observe_transition(
            update,
            store=self.terminal_objectives,
            option_id="" if pending is None else pending.causal_option_id,
            causal_edge_key=(
                "" if pending is None else (
                    pending.causal_option_edge_key
                    or pending.causal_subgoal_edge_key
                )
            ),
            intervention_signature=option_intervention_signature,
            intervention_id=(
                "" if pending is None else pending.intervention_id
            ),
            context_signature=context,
            downstream_subgoal_id=(
                ""
                if pending is None
                else pending.causal_option_downstream_subgoal_id
            ),
            replaying_progress_sequence=(
                False
                if pending is None
                else pending.causal_option_replaying_progress_sequence
            ),
            persistent_pursuit=(
                False
                if pending is None
                else pending.causal_option_persistent_pursuit
            ),
        )
        if (
            outcome["active_option_id"]
            or outcome["terminal_credited_option"]
            or outcome["rollout_closed"]
        ):
            self._causal_option_outcomes.append(outcome)

    def _observe_pending_option(
        self,
        update: LiveTransitionUpdate,
        pending: CognitiveDecision | None,
    ) -> None:
        if pending is None or not pending.source_rule_key:
            return
        if pending.source not in {
            "terminal_objective_option",
            "terminal_objective_probe",
            "terminal_objective_preparation",
        }:
            return
        rule = self.theory.promoted_relational_rule(pending.source_rule_key)
        if rule is None:
            return
        progress = observe_option_progress(update, rule)
        used_as_preparation = bool(pending.preparation_for_rule_key)
        self.option_memory.record(
            rule.key,
            progress,
            used_as_preparation=used_as_preparation,
        )
        rule.observe_application(
            expected_outcome_observed=progress.expected_outcome_observed,
            functional_progress=progress.functional_progress,
            level_progress=progress.level_progressed,
            visual_change=progress.visual_change,
            context_signature=_transition_context_signature(
                update,
                pending.action_data,
            ),
        )
        self._option_outcomes.append({
            "rule_key": rule.key,
            "option_id": pending.option_id,
            "used_as_preparation": used_as_preparation,
            "preparation_for_rule_key": pending.preparation_for_rule_key,
            "rule_status_after": rule.status.value,
            **progress.to_dict(),
        })

    def _validated_rule_count(self) -> int:
        return (
            len(self.belief_loop.rule_engine.high_confidence_rules())
            + len(self.theory.promoted_relational_rules())
        )


def _normalize_action(action: Any) -> str:
    if isinstance(action, (int, np.integer)):
        return "RESET" if int(action) == 0 else f"ACTION{int(action)}"
    raw = getattr(action, "name", action)
    text = str(raw or "").strip().upper()
    if "." in text:
        text = text.split(".")[-1]
    if text.isdigit():
        return "RESET" if int(text) == 0 else f"ACTION{int(text)}"
    return text


def _normalize_actions(actions: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for action in actions:
        name = _normalize_action(action)
        if name and name != "RESET" and name not in result:
            result.append(name)
    return result


def _grid_hash(grid: Any) -> int:
    return hash(np.asarray(grid, dtype=np.int32).tobytes())


def _align_grids(grid_before: Any, grid_after: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Pad resized frames to a shared canvas before structured differencing."""
    before = np.asarray(grid_before, dtype=np.int32)
    after = np.asarray(grid_after, dtype=np.int32)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError(
            f"expected two 2D grids, got {before.shape} and {after.shape}"
        )
    if before.shape == after.shape:
        return before, after
    height = max(before.shape[0], after.shape[0])
    width = max(before.shape[1], after.shape[1])
    before_canvas = np.full(
        (height, width),
        _background_value(before),
        dtype=np.int32,
    )
    after_canvas = np.full(
        (height, width),
        _background_value(after),
        dtype=np.int32,
    )
    before_canvas[: before.shape[0], : before.shape[1]] = before
    after_canvas[: after.shape[0], : after.shape[1]] = after
    return before_canvas, after_canvas


def _background_value(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _operator_prediction_matches(
    operator: Operator,
    update: LiveTransitionUpdate,
) -> bool:
    if not operator.expected_effects:
        return update.record.diff.num_changed > 0
    return all(effect.matches(update.record.diff) for effect in operator.expected_effects)


def _diff_signature(update: LiveTransitionUpdate) -> str:
    diff = update.record.diff
    return "|".join((
        update.action,
        str(diff.num_changed),
        str(diff.player_displacement),
        str(len(diff.created_objects)),
        str(len(diff.removed_objects)),
        str(bool(diff.game_over)),
        str(bool(diff.level_complete)),
    ))


def _transition_context_signature(
    update: LiveTransitionUpdate,
    action_data: Mapping[str, Any] | None,
) -> str:
    """Stable independent-context identity for live promotion evidence."""
    before = np.asarray(update.record.obs_before.raw_grid, dtype=np.int32)
    digest = hashlib.sha1(before.tobytes()).hexdigest()[:16]
    args = tuple(
        sorted((str(key), str(value)) for key, value in dict(action_data or {}).items())
    )
    return "|".join((
        str(before.shape),
        digest,
        str(update.record.obs_before.levels_completed),
        str(args),
    ))


def _nearest_non_player_target(
    observation: GameObservation,
) -> Dict[str, Any] | None:
    player = observation.best_player
    if player is None:
        return None
    pr, pc = player.position
    candidates = [
        obj for obj in observation.objects
        if obj.value != player.value and obj.area <= 30
    ]
    if not candidates:
        return None
    target = min(
        candidates,
        key=lambda obj: (
            abs(float(obj.center[0]) - pr) + abs(float(obj.center[1]) - pc),
            obj.area,
        ),
    )
    return {
        "position": (
            int(round(target.center[0])),
            int(round(target.center[1])),
        ),
        "value": int(target.value),
    }


def _operator_action_data(
    operator: Operator,
    call_args: Mapping[str, Any],
    observation: GameObservation,
) -> Dict[str, Any]:
    if operator.primitive_action != "ACTION6":
        return {}
    x = _optional_int(call_args.get("x", operator.primitive_x))
    y = _optional_int(call_args.get("y", operator.primitive_y))
    if x is not None and y is not None:
        return {"x": x, "y": y}
    target_value = call_args.get(
        "target_value",
        operator.parameters.get("target_value"),
    )
    for obj in observation.objects:
        if target_value is None or int(obj.value) == int(target_value):
            return {
                "x": int(round(obj.center[1])),
                "y": int(round(obj.center[0])),
            }
    height, width = observation.raw_grid.shape
    return {"x": int(width // 2), "y": int(height // 2)}


def _observe_prediction_outcome(
    update: LiveTransitionUpdate,
    prediction: DiscriminatingPrediction,
) -> str:
    record = update.record
    before = record.obs_before.raw_grid
    after = record.obs_after.raw_grid
    family = prediction.normalized_family
    if family == "color_transform":
        if prediction.target_color is None:
            return "unobservable"
        mask = (before != after) & (before == int(prediction.source_color))
        if not bool(mask.any()):
            return "none"
        values, counts = np.unique(after[mask], return_counts=True)
        target = int(values[int(np.argmax(counts))])
        return f"{prediction.source_color}->{target}"
    if family == "effect_scope":
        changed = record.diff.changed_cells
        if not changed:
            return "none"
        if record.action.x is None or record.action.y is None:
            return "global" if record.diff.num_changed >= 10 else "local"
        local = sum(
            1 for row, col in changed
            if max(
                abs(int(row) - int(record.action.y)),
                abs(int(col) - int(record.action.x)),
            ) <= 8
        )
        return "local" if local / max(1, len(changed)) >= 0.8 else "global"
    if family == "object_count":
        changed = (
            len(record.obs_before.objects) != len(record.obs_after.objects)
            or bool(record.diff.created_objects)
            or bool(record.diff.removed_objects)
        )
        return "changed" if changed else "stable"
    if family == "relation" and prediction.target_color is not None:
        before_holds = _relation_holds(
            record.obs_before,
            prediction.predicate_name,
            prediction.source_color,
            int(prediction.target_color),
        )
        after_holds = _relation_holds(
            record.obs_after,
            prediction.predicate_name,
            prediction.source_color,
            int(prediction.target_color),
        )
        if not before_holds and after_holds:
            return "appears"
        if before_holds and not after_holds:
            return "broken"
        if before_holds and after_holds:
            return "preserved"
        return "absent"
    return "unobservable"


def _relation_holds(
    observation: GameObservation,
    predicate: str,
    source_color: int,
    target_color: int,
) -> bool:
    source = [obj for obj in observation.objects if obj.value == int(source_color)]
    target = [obj for obj in observation.objects if obj.value == int(target_color)]
    if not source or not target:
        return False
    predicate = str(predicate)
    if predicate == "paired_with":
        return True
    if predicate == "same_shape":
        return any(
            first.shape_signature == second.shape_signature
            and first.area == second.area
            for first in source for second in target
        )
    if predicate == "aligned_with":
        return any(
            int(round(first.center[0])) == int(round(second.center[0]))
            or int(round(first.center[1])) == int(round(second.center[1]))
            for first in source for second in target
        )
    if predicate == "adjacent_to":
        return any(
            _objects_adjacent(first.cells, second.cells)
            for first in source for second in target
        )
    return False


def _objects_adjacent(
    first_cells: Sequence[Tuple[int, int]],
    second_cells: Sequence[Tuple[int, int]],
) -> bool:
    second = set(second_cells)
    for row, col in first_cells:
        if any(
            (row + drow, col + dcol) in second
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            return True
    return False


def _dedupe_predictions(
    predictions: Iterable[DiscriminatingPrediction],
) -> List[DiscriminatingPrediction]:
    result: List[DiscriminatingPrediction] = []
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.key in seen:
            continue
        seen.add(prediction.key)
        result.append(prediction)
    return result


__all__ = [
    "CognitiveDecision",
    "UnifiedCognitiveConfig",
    "UnifiedCognitiveController",
]
