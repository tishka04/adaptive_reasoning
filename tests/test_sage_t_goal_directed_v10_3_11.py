from __future__ import annotations

from theory.sage_t.goal_directed_v10_3_11 import (
    POSTERIOR_HISTORY_LIMIT,
    BoundedProgramPosterior,
    GoalConditionedSageTController,
    GoalConditionedUnifiedCognitiveController,
)
from theory.sage_t.t10_3_11_runtime import _delayed_goal_loop


def test_bounded_posterior_contract_starts_empty_and_disables_live_repair() -> None:
    posterior = BoundedProgramPosterior(history_limit=7)

    summary = posterior.bounded_summary()

    assert summary == {
        "observations": 0,
        "history": 0,
        "maximum_history": 0,
        "history_limit": 7,
        "live_repairs_suppressed": 0,
        "repairs_attempted": 0,
        "repairs_admitted": 0,
    }


def test_delayed_goal_loop_crosses_sixteen_actions_with_constant_memory() -> None:
    result = _delayed_goal_loop(3421)

    assert result["won"] is True
    assert result["actions"] == 18
    assert result["distinct_actions"] == 2
    assert result["alternating_prefix"] is True
    assert result["goal_conditioned_options"] >= 1
    assert result["goal_conditioned_actions"] == result["actions"]
    assert result["posterior_observations"] == result["actions"]
    assert result["posterior_maximum_history"] == POSTERIOR_HISTORY_LIMIT
    assert result["posterior_repairs_attempted"] == 0
    assert result["program_reassemblies"] <= 2


def test_goal_controller_types_remain_inside_unified_boundary() -> None:
    assert issubclass(GoalConditionedSageTController, object)
    assert issubclass(GoalConditionedUnifiedCognitiveController, object)

