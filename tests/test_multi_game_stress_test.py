"""A31 multi-game stress-test tests."""

from __future__ import annotations

from pathlib import Path

from theory.multi_game_evaluation import EvaluationTrace
from theory.multi_game_stress_test import (
    MultiGameStressTestResult,
    StressCurvePoint,
    classify_failure,
    run_multi_game_stress_test,
    select_stress_traces,
)


def test_classify_failure_keeps_scientific_blockage_types():
    assert (
        classify_failure("agenda_failed:not_enough_relation_candidates_for_agenda")
        == "not_enough_relation_candidates"
    )
    assert (
        classify_failure("functional_progress:no_functional_progress_observed")
        == "no_functional_progress"
    )
    assert (
        classify_failure("agenda_failed:local_relation_agenda_not_observed")
        == "relation_agenda_not_observed"
    )
    assert (
        classify_failure("agenda_failed:no_live_compatible_hypothesis")
        == "no_live_compatible_hypothesis"
    )


def test_stress_result_aggregates_budget_curve_points():
    result = MultiGameStressTestResult(
        budgets=[5],
        traces=[
            EvaluationTrace(
                game_id="ft09-0d8bbf25",
                trace_path=Path("human_traces/ft09.steps.jsonl"),
            )
        ],
        curve_points=[
            StressCurvePoint(
                game_id="ft09-0d8bbf25",
                trace_path=Path("human_traces/ft09.steps.jsonl"),
                experiment_budget=5,
                experiments_run=5,
                hypotheses_confirmed=4,
                hypotheses_refuted=6,
                useful_new_states=3,
                negative_memory_contexts_avoided=2,
                wrong_confirmations=0,
            ),
            StressCurvePoint(
                game_id="bp35-0a0ad940",
                trace_path=Path("human_traces/bp35.steps.jsonl"),
                experiment_budget=5,
                skips=1,
                failures=1,
                failure_types={"not_enough_relation_candidates": 1},
            ),
        ],
    )

    summary = result.budget_summaries[0]

    assert result.experiments_run == 5
    assert result.hypotheses_confirmed == 4
    assert result.hypotheses_refuted == 6
    assert result.negative_memory_contexts_avoided == 2
    assert result.wrong_confirmations_zero
    assert result.failures_are_typed
    assert result.failure_types == {"not_enough_relation_candidates": 1}
    assert summary.updates_per_experiment == 2.0
    assert summary.useful_new_states_per_experiment == 0.6


def test_select_stress_traces_can_keep_all_or_latest_per_game():
    paths = [
        Path("human_traces/ft09-0d8bbf25.20260617-142254.steps.jsonl"),
        Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        Path("human_traces/dc22-4c9bff3e.20260616-150906.steps.jsonl"),
    ]

    all_traces = select_stress_traces(trace_paths=paths, latest_per_game=False)
    latest = select_stress_traces(trace_paths=paths, latest_per_game=True)

    assert len(all_traces) == 3
    assert len(latest) == 2
    selected = {trace.game_id: trace.trace_path.name for trace in latest}
    assert selected["ft09-0d8bbf25"] == (
        "ft09-0d8bbf25.20260617-142428.steps.jsonl"
    )


def test_multi_game_stress_smoke_ft09_budget_curve():
    result = run_multi_game_stress_test(
        trace_paths=[
            Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl")
        ],
        budgets=[2],
        include_ar25=False,
        max_candidates=20,
        min_pixel_support=1,
        max_attempts_per_budget=2,
    )

    assert result.trace_count == 1
    assert result.game_count == 1
    assert result.experiments_run >= 2
    assert result.hypotheses_confirmed >= 1
    assert result.hypotheses_refuted >= 1
    assert result.useful_new_states >= 1
    assert result.negative_memory_contexts_avoided >= 1
    assert result.wrong_confirmations == 0
    assert result.wrong_confirmations_zero
    assert result.failures_are_typed
    assert result.curve_points[0].stopped_reason == "budget_reached"
