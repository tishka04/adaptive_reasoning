"""Protocol and metric tests for the paired held-out A/B runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    assert payload["schema_version"] == "sage.unified_cognition_ab_held_out.v14"
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
    assert "terminal_objective_probe_actions" in metrics["unified"]
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
