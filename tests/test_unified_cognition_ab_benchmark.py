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
    assert payload["schema_version"] == "sage.unified_cognition_ab_held_out.v4"
    assert protocol["protocol_gate_passed"] is True
    assert protocol["same_reset_visual_states"] is True
    assert protocol["online_learning_within_arm_only"] is True
    assert protocol["evaluation_outcomes_used_for_training_or_tuning"] is False
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
    assert "failure_causes" in payload
