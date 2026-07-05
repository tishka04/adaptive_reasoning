"""A28 transferability model tests."""

from __future__ import annotations

from pathlib import Path

from theory.non_ar25_functional_negative_memory import FunctionalAgendaNegativeRecord
from theory.non_ar25_transferability_model import (
    build_negative_transferability_model,
    run_negative_transferability_model,
)


def test_negative_transferability_model_groups_repeated_contexts():
    records = [
        FunctionalAgendaNegativeRecord(
            target_game="ft09-0d8bbf25",
            failed_relation_context="ACTION6::same_shape::colors9_8",
            tested_hypothesis="relation::ACTION6::same_shape::colors9_8::broken",
            observed_outcome="preserved",
        ),
        FunctionalAgendaNegativeRecord(
            target_game="ft09-0d8bbf25",
            failed_relation_context="ACTION6::same_shape::colors9_8",
            tested_hypothesis="relation::ACTION6::same_shape::colors9_8::broken",
            observed_outcome="preserved",
        ),
        FunctionalAgendaNegativeRecord(
            target_game="ft09-0d8bbf25",
            failed_relation_context="ACTION6::aligned_with::colors9_8",
            tested_hypothesis="relation::ACTION6::aligned_with::colors9_8::broken",
            observed_outcome="preserved",
        ),
    ]

    model = build_negative_transferability_model(records, min_negative_count=2)

    assert model.downweighted_context_signatures == [
        "ACTION6::same_shape::colors9_8"
    ]
    assert model.priority_adjustment_for("ACTION6::same_shape::colors9_8") == -2.0
    assert model.context_remains_testable("ACTION6::aligned_with::colors9_8")
    assert len(model.groups) == 2
    grouped = [
        group
        for group in model.groups
        if group.context_signature == "ACTION6::same_shape::colors9_8"
    ][0]
    assert grouped.predicate == "same_shape"
    assert grouped.predicted_outcome == "broken"
    assert grouped.count == 2


def test_transferability_model_adapts_functional_non_ar25_agenda():
    result = run_negative_transferability_model(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
        repeated_observations=2,
        min_negative_count=2,
    )

    assert result.error == ""
    assert len(result.negative_records) == 2
    assert result.repeated_negative_contexts_downweighted
    assert result.source_confirmations_unchanged
    assert result.unrelated_contexts_remain_testable
    assert result.functional_progress_still_measurable
    assert result.wrong_confirmations == 0
    assert result.trace_support_counted_as_proof is False

    assert result.transferability_model.downweighted_context_signatures == [
        "ACTION6::same_shape::colors9_8"
    ]
    assert result.source_confirmed_keys == [
        "relation::ACTION6::same_shape::colors9_8::preserved"
    ]
    assert result.adapted_attempt is not None
    assert result.adapted_attempt.functional_progress_non_ar25
    assert result.adapted_attempt.agenda_result is not None
    assert result.adapted_attempt.agenda_result.excluded_relation_context_signatures == [
        "ACTION6::same_shape::colors9_8"
    ]
    assert result.adapted_attempt.agenda_result.selected_option_order == [
        "prepare_relation_aligned_with_colors9_8",
        "avoid_relation_adjacent_to_absent_colors9_8",
    ]
