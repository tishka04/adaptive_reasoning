"""A27 negative memory inside functional non-ar25 agenda tests."""

from __future__ import annotations

from pathlib import Path

from theory.non_ar25_functional_negative_memory import (
    run_functional_negative_memory_agenda,
)


def test_negative_memory_changes_next_functional_non_ar25_agenda():
    result = run_functional_negative_memory_agenda(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
    )

    assert result.error == ""
    assert result.negative_memory_used_in_functional_agenda
    assert result.failed_relation_context_not_selected_again
    assert result.alternative_relation_context_selected
    assert result.functional_progress_non_ar25_remains_measurable
    assert result.wrong_confirmations == 0
    assert result.trace_support_counted_as_proof is False

    assert result.negative_memory.records
    record = result.negative_memory.records[0]
    assert record.failed_relation_context == "ACTION6::same_shape::colors9_8"
    assert record.tested_hypothesis == (
        "relation::ACTION6::same_shape::colors9_8::broken"
    )
    assert record.effect == "avoid_same_functional_agenda_context"

    assert result.first_attempt is not None
    assert result.first_attempt.functional_progress_non_ar25
    assert result.first_selected_relation_contexts == [
        "ACTION6::same_shape::colors9_8"
    ]

    assert result.second_attempt is not None
    assert result.second_attempt.functional_progress_non_ar25
    assert result.second_selected_relation_contexts == [
        "ACTION6::aligned_with::colors9_8"
    ]
    assert result.second_attempt.agenda_result is not None
    assert result.second_attempt.agenda_result.excluded_relation_context_signatures == [
        "ACTION6::same_shape::colors9_8"
    ]
    assert result.second_attempt.agenda_result.selected_option_order == [
        "prepare_relation_aligned_with_colors9_8",
        "avoid_relation_adjacent_to_absent_colors9_8",
    ]
