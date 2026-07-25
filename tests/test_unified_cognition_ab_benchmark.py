"""Protocol and metric tests for the paired held-out A/B runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import theory.unified_cognition_ab_benchmark as benchmark
from theory.unified_cognition_ab_benchmark import (
    run_unified_cognition_ab_benchmark,
)


@dataclass
class _FakeAction:
    id: int
    data: dict | None = None


@dataclass
class _FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1, 2)


class _FakeGame:
    def _get_valid_actions(self):
        return [_FakeAction(1), _FakeAction(2)]


class _FakeEnv:
    def __init__(self) -> None:
        self._game = _FakeGame()
        self.levels = 0
        self.grid = np.zeros((7, 7), dtype=np.int32)
        self.grid[3, 3] = 2

    def step(self, action, data=None):
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.levels = 0
            self.grid = np.zeros((7, 7), dtype=np.int32)
            self.grid[3, 3] = 2
            return _FakeFrame(self.grid.copy())
        if value == 1:
            self.levels += 1
            self.grid[3, 3] = 3
            return _FakeFrame(
                self.grid.copy(),
                state="WIN",
                levels_completed=self.levels,
            )
        self.grid[3, 4] = 4
        return _FakeFrame(self.grid.copy(), levels_completed=self.levels)


class _OneActionGame:
    def _get_valid_actions(self):
        return [_FakeAction(1)]


class _FakeTwoLevelEnv:
    def __init__(self) -> None:
        self._game = _OneActionGame()
        self.levels = 0
        self.grid = np.zeros((7, 7), dtype=np.int32)
        self.grid[3, 3] = 2

    def step(self, action, data=None):
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.levels = 0
            self.grid.fill(0)
            self.grid[3, 3] = 2
            return _FakeFrame(
                self.grid.copy(),
                levels_completed=self.levels,
                available_actions=(1,),
            )
        self.levels += 1
        self.grid[3, 3] = 2 + self.levels
        return _FakeFrame(
            self.grid.copy(),
            state="WIN" if self.levels >= 2 else "NOT_FINISHED",
            levels_completed=self.levels,
            available_actions=(1,),
        )


def test_ab_benchmark_pairs_fresh_resets_budgets_seeds_and_reports_failures():
    created = []

    def factory(game_id):
        created.append(game_id)
        return _FakeEnv()

    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-synthetic"],
        seeds=[7, 11],
        action_budget_per_reset=4,
        resets=2,
        env_factory=factory,
    )

    protocol = payload["paired_protocol"]
    assert payload["schema_version"] == "sage.unified_cognition_ab_held_out.v42"
    assert protocol["protocol_gate_passed"] is True
    assert protocol["same_reset_visual_states"] is True
    assert protocol["online_learning_within_arm_only"] is True
    assert protocol["evaluation_outcomes_used_for_training_or_tuning"] is False
    assert protocol["causal_subgoal_induction_enabled_in_unified"] is True
    assert protocol["causal_effect_credit_enabled_in_unified"] is True
    assert protocol["causal_hierarchical_options_enabled_in_unified"] is True
    assert (
        protocol[
            "effect_conditioned_downstream_subgoals_enabled_in_unified"
        ]
        is True
    )
    assert (
        protocol[
            "state_conditioned_directional_control_enabled_in_unified"
        ]
        is True
    )
    assert (
        protocol["persistent_directional_pursuit_enabled_in_unified"]
        is True
    )
    assert protocol["entity_anchored_interventions_enabled_in_unified"] is True
    assert protocol["active_entity_causal_binding_enabled_in_unified"] is True
    assert (
        protocol["mediated_entity_effect_induction_enabled_in_unified"]
        is True
    )
    assert (
        protocol["online_mediated_anti_unification_enabled_in_unified"]
        is True
    )
    assert (
        protocol["active_mediated_discrimination_enabled_in_unified"]
        is True
    )
    assert protocol["active_mode_restoration_enabled_in_unified"] is True
    assert (
        protocol["terminal_mediated_exploitation_enabled_in_unified"]
        is True
    )
    assert protocol["successor_policy_chaining_enabled_in_unified"] is True
    assert (
        protocol["successor_structural_transfer_enabled_in_unified"] is True
    )
    assert protocol["active_mediated_replication_enabled_in_unified"] is True
    assert (
        protocol["horizon_stable_learning_epochs_enabled_in_unified"] is True
    )
    assert (
        protocol["online_horizon_learning_arbiter_enabled_in_unified"] is True
    )
    assert protocol["structural_terminal_frontiers_enabled_in_unified"] is True
    assert (
        protocol["structural_terminal_attribution_enabled_in_unified"] is True
    )
    assert protocol["terminal_causal_reduction_enabled_in_unified"] is True
    assert protocol["active_frontier_reacquisition_enabled_in_unified"] is True
    assert (
        protocol[
            "recursive_terminal_causal_minimization_enabled_in_unified"
        ]
        is True
    )
    assert protocol["structural_frontier_transfer_enabled_in_unified"] is True
    assert protocol["progressive_terminal_routes_enabled_in_unified"] is True
    assert protocol["controller_rebranches_after_level_change"] is True
    assert (
        protocol[
            "terminal_relational_stencil_induction_enabled_in_unified"
        ]
        is True
    )
    assert (
        protocol["online_structural_break_detection_enabled_in_unified"]
        is True
    )
    assert (
        protocol["terminal_relational_stencil_relation_permuted_in_unified"]
        is False
    )
    assert (
        protocol["relational_memory_conditioned_by_regime_in_unified"]
        is True
    )
    assert (
        protocol[
            "active_structural_hypothesis_arbitration_enabled_in_unified"
        ]
        is True
    )
    assert (
        protocol["structural_regime_abstraction_enabled_in_unified"]
        is True
    )
    assert (
        protocol[
            "hierarchical_structural_theory_composition_enabled_in_unified"
        ]
        is True
    )
    assert len(payload["pairs"]) == 2
    assert len(created) == 8  # 2 seeds x 2 arms x 2 fresh resets

    metrics = payload["metrics"]
    assert set(metrics) == {
        "legacy_only",
        "unified",
        "delta_unified_minus_legacy",
    }
    assert "levels_completed" in metrics["unified"]
    assert "wins" in metrics["unified"]
    assert "experiment_actions" in metrics["unified"]
    assert "operator_plan_actions" in metrics["unified"]
    assert "operator_plan_streak_peak" in metrics["unified"]
    assert "operator_plan_budget_blocks" in metrics["unified"]
    assert "operator_plan_progress_resets" in metrics["unified"]
    assert "horizon_arbiter_evaluations" in metrics["unified"]
    assert "horizon_arbiter_reservations" in metrics["unified"]
    assert "horizon_arbiter_releases" in metrics["unified"]
    assert (
        "horizon_arbiter_causal_uncertainty_reservations"
        in metrics["unified"]
    )
    assert "horizon_arbiter_terminal_test_reservations" in metrics["unified"]
    assert "horizon_arbiter_priority_peak" in metrics["unified"]
    assert "terminal_objective_probe_actions" in metrics["unified"]
    assert "structural_frontier_signals_generated" in metrics["unified"]
    assert "structural_frontier_captures" in metrics["unified"]
    assert "structural_terminal_candidates" in metrics["unified"]
    assert "structural_terminal_credits" in metrics["unified"]
    assert "terminal_causal_reduction_probes" in metrics["unified"]
    assert "terminal_causal_reduction_confirmations" in metrics["unified"]
    assert "terminal_recursive_reduction_probes" in metrics["unified"]
    assert "terminal_maximum_reduction_generation" in metrics["unified"]
    assert "terminal_frontier_acquisition_paths" in metrics["unified"]
    assert "terminal_frontier_reacquisition_actions" in metrics["unified"]
    assert "structural_transfer_probes" in metrics["unified"]
    assert "structural_transfer_terminal_credits" in metrics["unified"]
    assert "max_level_reached" in metrics["unified"]
    assert "level_rebranches" in metrics["unified"]
    assert "progressive_terminal_routes" in metrics["unified"]
    assert "progressive_terminal_route_actions" in metrics["unified"]
    assert "progressive_terminal_route_confirmations" in metrics["unified"]
    assert "structural_regimes" in metrics["unified"]
    assert "structural_prediction_residuals" in metrics["unified"]
    assert "structural_terminal_condition_residuals" in metrics["unified"]
    assert "structural_breaks_detected" in metrics["unified"]
    assert "structural_old_theory_suspensions" in metrics["unified"]
    assert "structural_revision_hypotheses_generated" in metrics["unified"]
    assert "structural_revision_actions" in metrics["unified"]
    assert "structural_revision_confirmations" in metrics["unified"]
    assert "structural_revision_refutations" in metrics["unified"]
    assert "structural_arbitration_decisions" in metrics["unified"]
    assert "structural_discriminating_experiments" in metrics["unified"]
    assert (
        "structural_unactionable_hypotheses_refuted"
        in metrics["unified"]
    )
    assert "structural_regime_families" in metrics["unified"]
    assert "structural_family_transfers" in metrics["unified"]
    assert "structural_family_transfer_actions" in metrics["unified"]
    assert "structural_theory_programs" in metrics["unified"]
    assert "structural_theory_switches" in metrics["unified"]
    assert "structural_theory_reactivations" in metrics["unified"]
    assert "terminal_objective_grounded_actions" in metrics["unified"]
    assert "terminal_objective_discriminator_actions" in metrics["unified"]
    assert "terminal_objective_ablation_actions" in metrics["unified"]
    assert "generated_goal_hypotheses" in metrics["unified"]
    assert "objective_distance_reductions" in metrics["unified"]
    assert "objective_ambiguous_terminal_events" in metrics["unified"]
    assert "terminal_supported_objectives" in metrics["unified"]
    assert "temporal_subgoal_probe_actions" in metrics["unified"]
    assert "temporal_subgoal_option_actions" in metrics["unified"]
    assert "temporal_plans_generated" in metrics["unified"]
    assert "temporal_plan_starts" in metrics["unified"]
    assert "temporal_plan_actions" in metrics["unified"]
    assert "temporal_step_completions" in metrics["unified"]
    assert "temporal_plan_abandonments" in metrics["unified"]
    assert "terminal_supported_temporal_plans" in metrics["unified"]
    assert "causal_dependency_plans" in metrics["unified"]
    assert "causal_dependency_plan_actions" in metrics["unified"]
    assert "causal_edges_generated" in metrics["unified"]
    assert "causal_blocked_target_events" in metrics["unified"]
    assert "causal_edge_trials" in metrics["unified"]
    assert "causal_edge_support_events" in metrics["unified"]
    assert "causal_edge_contradictions" in metrics["unified"]
    assert "confirmed_causal_edges" in metrics["unified"]
    assert "refuted_causal_edges" in metrics["unified"]
    assert "causal_effect_observations" in metrics["unified"]
    assert "causal_effect_guided_actions" in metrics["unified"]
    assert "causal_productive_effect_signatures" in metrics["unified"]
    assert "causal_delayed_credit_events" in metrics["unified"]
    assert "causal_cross_branch_confirmations" in metrics["unified"]
    assert "causal_reserved_confirmation_starts" in metrics["unified"]
    assert "causal_options_compiled" in metrics["unified"]
    assert "causal_option_opening_events" in metrics["unified"]
    assert "causal_option_rollouts" in metrics["unified"]
    assert "causal_option_downstream_actions" in metrics["unified"]
    assert "causal_option_terminal_credited_events" in metrics["unified"]
    assert "entity_anchored_candidate_signatures" in metrics["unified"]
    assert "entity_anchored_transfer_signatures" in metrics["unified"]
    assert "entity_anchored_selections" in metrics["unified"]
    assert "entity_binding_observations" in metrics["unified"]
    assert "entity_binding_tracks_created" in metrics["unified"]
    assert "entity_binding_transformed_entities" in metrics["unified"]
    assert "entity_binding_carrier_progress_events" in metrics["unified"]
    assert "entity_binding_noncarrier_progress_events" in metrics["unified"]
    assert "entity_binding_conflicts" in metrics["unified"]
    assert "entity_binding_controlled_contrast_selections" in metrics["unified"]
    assert "mediated_effect_observations" in metrics["unified"]
    assert "mediated_effect_scene_correspondences" in metrics["unified"]
    assert "mediated_effect_changed_entities" in metrics["unified"]
    assert "mediated_effect_tracks_created" in metrics["unified"]
    assert "mediated_effect_models" in metrics["unified"]
    assert "mediated_effect_supported_hyperedges" in metrics["unified"]
    assert "mediated_effect_direct_target_progress_events" in metrics["unified"]
    assert (
        "mediated_effect_progress_with_indirect_candidates"
        in metrics["unified"]
    )
    assert (
        "mediated_effect_controlled_contrast_selections"
        in metrics["unified"]
    )
    assert "mediated_replication_requests_created" in metrics["unified"]
    assert "mediated_replication_cross_branch_activations" in metrics["unified"]
    assert "mediated_replication_selections" in metrics["unified"]
    assert "mediated_replication_preparation_starts" in metrics["unified"]
    assert "mediated_replication_preparation_actions" in metrics["unified"]
    assert "mediated_replication_confirmations" in metrics["unified"]
    assert "mediated_replication_refutations" in metrics["unified"]
    assert "mediated_abstraction_hypotheses" in metrics["unified"]
    assert "mediated_abstraction_supported_hyperedges" in metrics["unified"]
    assert "mediated_abstraction_control_contexts" in metrics["unified"]
    assert "mediated_abstraction_regression_contexts" in metrics["unified"]
    assert "mediated_discrimination_requests_created" in metrics["unified"]
    assert "mediated_discrimination_predictions" in metrics["unified"]
    assert (
        "mediated_discrimination_mode_mismatch_blocks" in metrics["unified"]
    )
    assert "mediated_discrimination_selections" in metrics["unified"]
    assert "mediated_restoration_actions" in metrics["unified"]
    assert "mediated_restoration_predictions" in metrics["unified"]
    assert "mediated_restoration_selections" in metrics["unified"]
    assert "mediated_restoration_steps_confirmed" in metrics["unified"]
    assert "mediated_restoration_targets_reached" in metrics["unified"]
    assert "mediated_restoration_failures" in metrics["unified"]
    assert "mediated_exploitation_policies_compiled" in metrics["unified"]
    assert "mediated_exploitation_actions" in metrics["unified"]
    assert "mediated_exploitation_progress_events" in metrics["unified"]
    assert "mediated_exploitation_terminal_events" in metrics["unified"]
    assert (
        "mediated_discrimination_feature_requirements" in metrics["unified"]
    )
    assert (
        "mediated_discrimination_feature_eliminations" in metrics["unified"]
    )
    assert "terminal_supported_causal_options" in metrics["unified"]
    assert "effect_conditioned_goal_candidates_generated" in metrics["unified"]
    assert "effect_conditioned_subgoals_generated" in metrics["unified"]
    assert "effect_conditioned_subgoal_links" in metrics["unified"]
    assert "productive_effect_subgoal_links" in metrics["unified"]
    assert "effect_conditioned_subgoal_guided_actions" in metrics["unified"]
    assert "effect_conditioned_subgoal_progress_events" in metrics["unified"]
    assert "effect_conditioned_trigger_progress_events" in metrics["unified"]
    assert "effect_conditioned_pursuit_progress_events" in metrics["unified"]
    assert "directional_effect_observations" in metrics["unified"]
    assert "directional_pursuit_observations" in metrics["unified"]
    assert "directional_reversible_action_objectives" in metrics["unified"]
    assert "directional_mode_contrast_selections" in metrics["unified"]
    assert "directional_bridge_predictions" in metrics["unified"]
    assert "directional_bridge_selections" in metrics["unified"]
    assert "directional_entity_anchored_action_models" in metrics["unified"]
    assert "directional_structural_transfer_predictions" in metrics["unified"]
    assert (
        "mediated_successor_structural_transfer_predictions"
        in metrics["unified"]
    )
    assert "directional_entity_alias_conflicts" in metrics["unified"]
    assert "directional_entity_contrast_selections" in metrics["unified"]
    assert "directional_blocked_regressive_actions" in metrics["unified"]
    assert "persistent_pursuit_commitment_selections" in metrics["unified"]
    assert "persistent_pursuit_continuation_actions" in metrics["unified"]
    assert "persistent_pursuit_progress_events" in metrics["unified"]
    assert "persistent_pursuit_bridge_actions" in metrics["unified"]
    assert "persistent_pursuit_entity_contrast_actions" in metrics["unified"]
    assert (
        "persistent_pursuit_entity_binding_contrast_actions"
        in metrics["unified"]
    )
    assert (
        "persistent_pursuit_mediated_effect_policy_actions"
        in metrics["unified"]
    )
    assert (
        "persistent_pursuit_mediated_effect_contrast_actions"
        in metrics["unified"]
    )
    assert "persistent_pursuit_rollout_budget_extensions" in metrics["unified"]
    assert "persistent_pursuit_longest_continuation" in metrics["unified"]
    assert "causal_option_dynamic_budget_extensions" in metrics["unified"]
    assert "causal_option_budget_pruned_rollouts" in metrics["unified"]
    assert "failure_causes" in payload


def test_ab_benchmark_rebranches_controller_and_gates_consecutive_depth_two(
    monkeypatch,
):
    monkeypatch.setattr(benchmark, "_reset_env", lambda env: env.step(0))
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-two-level-synthetic"],
        seeds=[3],
        action_budget_per_reset=4,
        resets=1,
        env_factory=lambda _game_id: _FakeTwoLevelEnv(),
    )

    unified = payload["metrics"]["unified"]
    assert unified["max_level_reached"] == 2
    assert unified["level_rebranches"] == 1
    assert payload["depth_two_gate_passed"] is True


def test_ab_benchmark_exposes_a_reproducible_causal_subgoal_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-causal-ablation"],
        seeds=[3],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_causal_subgoal_induction=False,
    )

    assert (
        payload["paired_protocol"][
            "causal_subgoal_induction_enabled_in_unified"
        ]
        is False
    )
    assert payload["metrics"]["unified"]["causal_edges_generated"] == 0
    assert payload["metrics"]["unified"]["causal_dependency_plans"] == 0


def test_ab_benchmark_exposes_a_reproducible_causal_effect_credit_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-effect-credit-ablation"],
        seeds=[5],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_causal_effect_credit=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["causal_subgoal_induction_enabled_in_unified"] is True
    assert protocol["causal_effect_credit_enabled_in_unified"] is False
    assert payload["metrics"]["unified"]["causal_effect_guided_actions"] == 0
    assert (
        payload["metrics"]["unified"]["causal_reserved_confirmation_starts"]
        == 0
    )


def test_ab_benchmark_exposes_a_reproducible_causal_option_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-causal-option-ablation"],
        seeds=[7],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_causal_hierarchical_options=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["causal_effect_credit_enabled_in_unified"] is True
    assert protocol["causal_hierarchical_options_enabled_in_unified"] is False
    assert payload["metrics"]["unified"]["causal_options_compiled"] == 0
    assert payload["metrics"]["unified"]["causal_option_downstream_actions"] == 0


def test_ab_benchmark_exposes_effect_conditioned_subgoal_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-effect-subgoal-ablation"],
        seeds=[9],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_effect_conditioned_downstream_subgoals=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["causal_hierarchical_options_enabled_in_unified"] is True
    assert (
        protocol[
            "effect_conditioned_downstream_subgoals_enabled_in_unified"
        ]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["effect_conditioned_goal_candidates_generated"] == 0
    assert metrics["effect_conditioned_subgoals_generated"] == 0
    assert metrics["effect_conditioned_subgoal_guided_actions"] == 0


def test_ab_benchmark_exposes_directional_control_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-directional-control-ablation"],
        seeds=[13],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_state_conditioned_directional_control=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "effect_conditioned_downstream_subgoals_enabled_in_unified"
        ]
        is True
    )
    assert (
        protocol[
            "state_conditioned_directional_control_enabled_in_unified"
        ]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["directional_effect_observations"] == 0
    assert metrics["directional_predictions"] == 0


def test_ab_benchmark_exposes_persistent_pursuit_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-persistent-pursuit-ablation"],
        seeds=[17],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_persistent_directional_pursuit=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "state_conditioned_directional_control_enabled_in_unified"
        ]
        is True
    )
    assert (
        protocol["persistent_directional_pursuit_enabled_in_unified"]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["persistent_pursuit_commitment_selections"] == 0
    assert metrics["persistent_pursuit_continuation_actions"] == 0
    assert metrics["persistent_pursuit_progress_events"] == 0


def test_ab_benchmark_exposes_entity_anchor_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-entity-anchor-ablation"],
        seeds=[19],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_entity_anchored_interventions=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["persistent_directional_pursuit_enabled_in_unified"] is True
    assert protocol["entity_anchored_interventions_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["entity_anchored_candidate_signatures"] == 0
    assert metrics["entity_anchored_transfer_signatures"] == 0
    assert metrics["entity_anchored_selections"] == 0


def test_ab_benchmark_exposes_active_entity_binding_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-entity-binding-ablation"],
        seeds=[23],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_active_entity_causal_binding=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["entity_anchored_interventions_enabled_in_unified"] is True
    assert protocol["active_entity_causal_binding_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["entity_binding_observations"] == 0
    assert metrics["entity_binding_predictions"] == 0
    assert metrics["entity_binding_controlled_contrast_selections"] == 0


def test_ab_benchmark_exposes_mediated_entity_effect_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-mediated-effect-ablation"],
        seeds=[29],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_mediated_entity_effect_induction=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["active_entity_causal_binding_enabled_in_unified"] is True
    assert (
        protocol["mediated_entity_effect_induction_enabled_in_unified"]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_effect_observations"] == 0
    assert metrics["mediated_effect_predictions"] == 0
    assert metrics["mediated_effect_supported_hyperedges"] == 0
    assert metrics["mediated_effect_direct_target_progress_events"] == 0
    assert metrics["mediated_effect_controlled_contrast_selections"] == 0


def test_ab_benchmark_exposes_active_mediated_replication_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-mediated-replication-ablation"],
        seeds=[31],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_active_mediated_replication=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["mediated_entity_effect_induction_enabled_in_unified"] is True
    assert protocol["active_mediated_replication_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_replication_requests_created"] == 0
    assert metrics["mediated_replication_cross_branch_activations"] == 0
    assert metrics["mediated_replication_selections"] == 0
    assert metrics["mediated_replication_preparation_starts"] == 0
    assert metrics["mediated_replication_preparation_actions"] == 0
    assert metrics["mediated_replication_confirmations"] == 0
    assert metrics["mediated_replication_refutations"] == 0


def test_ab_benchmark_exposes_online_mediated_anti_unification_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-mediated-abstraction-ablation"],
        seeds=[37],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_online_mediated_anti_unification=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["mediated_entity_effect_induction_enabled_in_unified"] is True
    assert (
        protocol["online_mediated_anti_unification_enabled_in_unified"]
        is False
    )
    assert protocol["active_mediated_replication_enabled_in_unified"] is True
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_abstraction_hypotheses"] == 0
    assert metrics["mediated_abstraction_supported_hyperedges"] == 0
    assert metrics["mediated_abstraction_control_contexts"] == 0
    assert metrics["mediated_abstraction_regression_contexts"] == 0


def test_ab_benchmark_exposes_active_mediated_discrimination_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-mediated-discrimination-ablation"],
        seeds=[41],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_active_mediated_discrimination=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["online_mediated_anti_unification_enabled_in_unified"] is True
    assert (
        protocol["active_mediated_discrimination_enabled_in_unified"]
        is False
    )
    assert protocol["active_mediated_replication_enabled_in_unified"] is True
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_discrimination_requests_created"] == 0
    assert metrics["mediated_discrimination_predictions"] == 0
    assert metrics["mediated_discrimination_mode_mismatch_blocks"] == 0
    assert metrics["mediated_discrimination_selections"] == 0
    assert metrics["mediated_discrimination_feature_requirements"] == 0
    assert metrics["mediated_discrimination_feature_eliminations"] == 0


def test_ab_benchmark_exposes_active_mode_restoration_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-mode-restoration-ablation"],
        seeds=[43],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_active_mode_restoration=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["active_mediated_discrimination_enabled_in_unified"]
        is True
    )
    assert protocol["active_mode_restoration_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_restoration_actions"] == 0
    assert metrics["mediated_restoration_predictions"] == 0
    assert metrics["mediated_restoration_selections"] == 0
    assert metrics["mediated_restoration_targets_reached"] == 0


def test_ab_benchmark_exposes_terminal_mediated_exploitation_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-mediated-exploitation-ablation"],
        seeds=[47],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_terminal_mediated_exploitation=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["active_mode_restoration_enabled_in_unified"] is True
    assert (
        protocol["terminal_mediated_exploitation_enabled_in_unified"]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_exploitation_policies_compiled"] == 0
    assert metrics["mediated_exploitation_predictions"] == 0
    assert metrics["mediated_exploitation_selections"] == 0
    assert metrics["mediated_exploitation_terminal_events"] == 0


def test_ab_benchmark_exposes_successor_policy_chaining_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-successor-chain-ablation"],
        seeds=[53],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_successor_policy_chaining=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["terminal_mediated_exploitation_enabled_in_unified"]
        is True
    )
    assert protocol["successor_policy_chaining_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_successor_states_captured"] == 0
    assert metrics["mediated_successor_action_selections"] == 0
    assert metrics["mediated_successor_progress_events"] == 0


def test_ab_benchmark_exposes_successor_structural_transfer_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-successor-analogy-ablation"],
        seeds=[59],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_successor_structural_transfer=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["successor_policy_chaining_enabled_in_unified"] is True
    assert (
        protocol["successor_structural_transfer_enabled_in_unified"] is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["mediated_successor_structural_policy_classes"] == 0
    assert metrics["mediated_successor_structural_transfer_predictions"] == 0
    assert metrics["mediated_successor_structural_transfer_selections"] == 0


def test_ab_benchmark_exposes_horizon_stable_learning_epoch_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-horizon-stability-ablation"],
        seeds=[61],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_horizon_stable_learning_epochs=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["successor_structural_transfer_enabled_in_unified"] is True
    assert (
        protocol["horizon_stable_learning_epochs_enabled_in_unified"] is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["operator_plan_budget_blocks"] == 0
    assert metrics["operator_plan_progress_resets"] == 0


def test_ab_benchmark_exposes_online_horizon_learning_arbiter_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-online-horizon-arbiter-ablation"],
        seeds=[67],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_online_horizon_learning_arbiter=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["horizon_stable_learning_epochs_enabled_in_unified"] is True
    assert (
        protocol["online_horizon_learning_arbiter_enabled_in_unified"] is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["horizon_arbiter_evaluations"] == 0
    assert metrics["horizon_arbiter_reservations"] == 0
    assert metrics["horizon_arbiter_releases"] == 0


def test_ab_benchmark_exposes_terminal_negative_frontier_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-terminal-frontier-ablation"],
        seeds=[71],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_terminal_negative_frontier_exploration=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["online_horizon_learning_arbiter_enabled_in_unified"] is True
    assert (
        protocol[
            "terminal_negative_frontier_exploration_enabled_in_unified"
        ]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_frontier_suffix_actions"] == 0
    assert metrics["terminal_frontiers_captured"] == 0
    assert metrics["terminal_frontier_terminal_credits"] == 0


def test_ab_benchmark_exposes_adaptive_terminal_frontier_horizon_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-adaptive-frontier-horizon-ablation"],
        seeds=[73],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_adaptive_terminal_frontier_horizon=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "terminal_negative_frontier_exploration_enabled_in_unified"
        ]
        is True
    )
    assert (
        protocol[
            "adaptive_terminal_frontier_horizon_enabled_in_unified"
        ]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_frontier_adaptive_horizon_extensions"] == 0
    assert metrics["terminal_frontier_extended_suffix_actions"] == 0
    assert metrics["terminal_frontier_adaptive_horizon_actions_granted"] == 0


def test_ab_benchmark_exposes_dormant_terminal_lineage_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-dormant-terminal-lineage-ablation"],
        seeds=[79],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_dormant_terminal_lineage=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "adaptive_terminal_frontier_horizon_enabled_in_unified"
        ]
        is True
    )
    assert protocol["dormant_terminal_lineage_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_frontier_dormant_lineages_started"] == 0
    assert metrics["terminal_frontier_dormant_lineage_actions"] == 0
    assert metrics["terminal_frontier_dormant_lineage_censored"] == 0
    assert metrics["terminal_frontier_dormant_lineage_expired"] == 0


def test_ab_benchmark_exposes_structural_frontier_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-structural-frontier-ablation"],
        seeds=[83],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_structural_terminal_frontiers=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["structural_terminal_frontiers_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["structural_frontier_signals_generated"] == 0
    assert metrics["structural_frontier_captures"] == 0


def test_ab_benchmark_exposes_structural_attribution_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-structural-attribution-ablation"],
        seeds=[89],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_structural_terminal_attribution=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["structural_terminal_attribution_enabled_in_unified"]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["structural_terminal_candidates"] == 0
    assert metrics["structural_terminal_credits"] == 0


def test_ab_benchmark_exposes_terminal_causal_reduction_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-terminal-causal-reduction-ablation"],
        seeds=[97],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_terminal_causal_reduction=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["terminal_causal_reduction_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_causal_reduction_probes"] == 0
    assert metrics["terminal_causal_reduction_actions"] == 0
    assert metrics["terminal_causal_reduction_credits"] == 0


def test_ab_benchmark_exposes_active_frontier_reacquisition_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-frontier-reacquisition-ablation"],
        seeds=[101],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_active_frontier_reacquisition=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["active_frontier_reacquisition_enabled_in_unified"]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_frontier_acquisition_paths"] == 0
    assert metrics["terminal_frontier_reacquisition_attempts"] == 0
    assert metrics["terminal_frontier_reacquisition_actions"] == 0


def test_ab_benchmark_exposes_recursive_terminal_minimization_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-recursive-minimization-ablation"],
        seeds=[103],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_recursive_terminal_causal_minimization=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "recursive_terminal_causal_minimization_enabled_in_unified"
        ]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_recursive_reduction_probes"] == 0
    assert metrics["terminal_maximum_reduction_generation"] <= 1


def test_ab_benchmark_exposes_structural_frontier_transfer_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-structural-transfer-ablation"],
        seeds=[107],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_structural_frontier_transfer=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["structural_frontier_transfer_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["structural_transfer_probes"] == 0
    assert metrics["structural_transfer_actions"] == 0
    assert metrics["structural_transfer_terminal_credits"] == 0


def test_ab_benchmark_exposes_progressive_terminal_route_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-progressive-route-ablation"],
        seeds=[109],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_progressive_terminal_routes=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol["progressive_terminal_routes_enabled_in_unified"] is False
    metrics = payload["metrics"]["unified"]
    assert metrics["progressive_terminal_routes"] == 0
    assert metrics["progressive_terminal_route_attempts"] == 0
    assert metrics["progressive_terminal_route_actions"] == 0
    assert payload["depth_two_gate_passed"] is False


def test_ab_benchmark_exposes_terminal_relational_stencil_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-terminal-relational-stencil-ablation"],
        seeds=[113],
        action_budget_per_reset=3,
        resets=2,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_terminal_relational_stencil_induction=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "terminal_relational_stencil_induction_enabled_in_unified"
        ]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_relational_stencil_examples"] == 0
    assert metrics["terminal_relational_stencil_decisions"] == 0
    assert metrics["terminal_relational_stencil_rules"] == 0


def test_ab_benchmark_exposes_online_structural_break_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-structural-break-ablation"],
        seeds=[103],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_online_structural_break_detection=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["online_structural_break_detection_enabled_in_unified"]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["structural_breaks_detected"] == 0
    assert metrics["structural_revision_hypotheses_generated"] == 0
    assert metrics["structural_revision_actions"] == 0


def test_ab_benchmark_exposes_permuted_relation_control():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-permuted-relation-control"],
        seeds=[107],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        permute_terminal_relational_stencil_relation=True,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["terminal_relational_stencil_relation_permuted_in_unified"]
        is True
    )
    assert (
        protocol["relational_memory_conditioned_by_regime_in_unified"]
        is True
    )


def test_ab_benchmark_composes_permutation_with_revision_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-composed-sage9q-controls"],
        seeds=[108],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_online_structural_break_detection=False,
        permute_terminal_relational_stencil_relation=True,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["online_structural_break_detection_enabled_in_unified"]
        is False
    )
    assert (
        protocol["terminal_relational_stencil_relation_permuted_in_unified"]
        is True
    )
    summary = payload["pairs"][0]["unified"]["controller_summary"]
    assert (
        summary["online_structural_break_detection"]["enabled"]
        is False
    )
    assert (
        summary["terminal_relational_stencil_induction"][
            "permute_confirmed_relation"
        ]
        is True
    )


def test_ab_benchmark_exposes_unconditioned_regime_memory_control():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-unconditioned-regime-memory"],
        seeds=[109],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        condition_relational_memory_by_regime=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["relational_memory_conditioned_by_regime_in_unified"]
        is False
    )
    assert (
        protocol["online_structural_break_detection_enabled_in_unified"]
        is True
    )


def test_ab_benchmark_exposes_sage9s_9t_9u_ablations():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-sage9stu-ablations"],
        seeds=[110],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_active_structural_hypothesis_arbitration=False,
        enable_structural_regime_abstraction=False,
        enable_hierarchical_structural_theory_composition=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "active_structural_hypothesis_arbitration_enabled_in_unified"
        ]
        is False
    )
    assert (
        protocol["structural_regime_abstraction_enabled_in_unified"]
        is False
    )
    assert (
        protocol[
            "hierarchical_structural_theory_composition_enabled_in_unified"
        ]
        is False
    )
    summary = payload["pairs"][0]["unified"]["controller_summary"][
        "online_structural_break_detection"
    ]
    assert summary["active_hypothesis_arbitration_enabled"] is False
    assert summary["regime_abstraction_enabled"] is False
    assert summary["hierarchical_theory_composition_enabled"] is False


def test_ab_benchmark_exposes_sage9v_frontier_exploration_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-sage9v-ablation"],
        seeds=[111],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_frontier_oriented_exploration=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol["frontier_oriented_exploration_enabled_in_unified"]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["frontier_stagnation_detections"] == 0
    assert metrics["frontier_experiments"] == 0
    assert metrics["frontier_terminal_credits"] == 0
    summary = payload["pairs"][0]["unified"]["controller_summary"][
        "frontier_oriented_exploration"
    ]
    assert summary["enabled"] is False
    assert payload["frontier_oriented_exploration_gate_passed"] is False


def test_ab_benchmark_exposes_sage9w_multiform_relation_ablation():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-sage9w-ablation"],
        seeds=[112],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_terminal_multiform_relational_induction=False,
    )

    protocol = payload["paired_protocol"]
    assert (
        protocol[
            "terminal_multiform_relational_induction_enabled_in_unified"
        ]
        is False
    )
    metrics = payload["metrics"]["unified"]
    assert metrics["terminal_multiform_observations"] == 0
    assert metrics["terminal_multiform_confirmed_patterns"] == 0
    assert metrics["terminal_multiform_selections"] == 0
    summary = payload["pairs"][0]["unified"]["controller_summary"][
        "terminal_multiform_relational_induction"
    ]
    assert summary["enabled"] is False
    assert payload["multiform_relational_induction_gate_passed"] is False


def test_ab_benchmark_exposes_sage10b_plus_isolated_ablations():
    payload = run_unified_cognition_ab_benchmark(
        game_ids=["held-out-sage10b-plus-ablation"],
        seeds=[113],
        action_budget_per_reset=3,
        resets=1,
        env_factory=lambda _game_id: _FakeEnv(),
        enable_subeffect_eligibility_relay=False,
        enable_generalized_frontier_stall_detection=False,
        enable_per_level_frontier_rearming=False,
        enable_level_route_memory=False,
        enable_level_route_shortening=False,
    )

    protocol = payload["paired_protocol"]
    assert protocol[
        "subeffect_eligibility_relay_enabled_in_unified"
    ] is False
    assert protocol[
        "generalized_frontier_stall_detection_enabled_in_unified"
    ] is False
    assert protocol[
        "per_level_frontier_rearming_enabled_in_unified"
    ] is False
    assert protocol["level_route_memory_enabled_in_unified"] is False
    assert protocol["level_route_shortening_enabled_in_unified"] is False
    controller = payload["pairs"][0]["unified"]["controller_summary"]
    frontier = controller["frontier_oriented_exploration"]
    routes = controller["level_route_memory"]
    assert frontier["subeffect_eligibility_relay_enabled"] is False
    assert frontier["generalized_stall_detection_enabled"] is False
    assert frontier["per_level_rearming_enabled"] is False
    assert routes["enabled"] is False
    assert routes["shortening_enabled"] is False
    assert routes["observed_routes"] == 0
    assert routes["routes"] == 0
