"""A30 multi-game evaluation tests."""

from __future__ import annotations

from pathlib import Path

from theory.multi_game_evaluation import (
    EvaluationResult,
    MultiGameEvaluationResult,
    run_multi_game_evaluation,
    select_evaluation_traces,
)


def test_multi_game_aggregate_metrics_do_not_need_new_architecture():
    first = EvaluationResult(
        game_id="g1",
        experiments_run=4,
        hypotheses_confirmed=3,
        hypotheses_refuted=1,
        negative_memories_created=2,
        negative_memory_contexts_avoided=1,
        useful_new_states=2,
    )
    second = EvaluationResult(
        game_id="g2",
        experiments_run=2,
        hypotheses_confirmed=0,
        hypotheses_refuted=2,
        wrong_confirmations=0,
        useful_new_states=1,
    )

    result = MultiGameEvaluationResult(game_results=[first, second])

    assert result.games_evaluated == 2
    assert result.experiments_run == 6
    assert result.hypotheses_confirmed == 3
    assert result.hypotheses_refuted == 3
    assert result.experimental_efficiency == 1.0
    assert result.negative_memories_created == 2
    assert result.negative_memory_contexts_avoided == 1
    assert result.functional_progress == 0.5
    assert result.wrong_confirmations == 0


def test_select_evaluation_traces_keeps_latest_per_game():
    traces = select_evaluation_traces(
        trace_paths=[
            Path("human_traces/ft09-0d8bbf25.20260617-142254.steps.jsonl"),
            Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
            Path("human_traces/dc22-4c9bff3e.20260616-150906.steps.jsonl"),
        ],
        max_games=10,
        include_ar25=False,
    )

    selected = {trace.game_id: trace.trace_path.name for trace in traces}

    assert selected["ft09-0d8bbf25"] == (
        "ft09-0d8bbf25.20260617-142428.steps.jsonl"
    )
    assert selected["dc22-4c9bff3e"] == (
        "dc22-4c9bff3e.20260616-150906.steps.jsonl"
    )


def test_multi_game_evaluation_smoke_ft09_without_transfer_checks():
    result = run_multi_game_evaluation(
        trace_paths=[
            Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl")
        ],
        max_games=1,
        include_ar25=False,
        max_candidates=20,
        min_pixel_support=1,
        run_transfer_checks=False,
    )

    assert result.games_evaluated == 1
    assert result.errors == []
    assert result.experiments_run >= 1
    assert result.hypotheses_confirmed >= 1
    assert result.hypotheses_refuted >= 1
    assert result.negative_memories_created >= 1
    assert result.negative_memory_contexts_avoided >= 1
    assert result.useful_new_states >= 1
    assert result.functional_progress > 0.0
    assert result.wrong_confirmations == 0
    assert result.trace_support_counted_as_proof is False
