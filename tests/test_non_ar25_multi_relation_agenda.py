"""A25 non-ar25 multi-relation agenda tests."""

from __future__ import annotations

from pathlib import Path

from theory.epistemic_metrics import HypothesisStatus
from theory.non_ar25_multi_relation_agenda import (
    run_non_ar25_multi_relation_agenda,
)


def test_non_ar25_multi_relation_agenda_gates_active_test_on_local_relations():
    result = run_non_ar25_multi_relation_agenda(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
    )

    assert result.error == ""
    assert result.non_ar25_multi_relation_agenda
    assert result.relation_candidate_count >= 2
    assert result.includes_prepare_and_avoid_options
    assert result.selection_reason == "missing_preconditions"
    assert result.order_chosen_by_missing_preconditions
    assert result.selected_option_order == [
        "prepare_relation_same_shape_colors9_8",
        "prepare_relation_aligned_with_colors9_8",
        "avoid_relation_adjacent_to_absent_colors9_8",
    ]
    assert result.missing_preconditions == [
        "relation_present::same_shape::colors9_8",
        "relation_present::aligned_with::colors9_8",
        "relation_absent::adjacent_to::colors9_8",
    ]
    assert result.local_agenda_observed
    assert result.validation_test_called
    assert result.validation_test_called_only_when_local_agenda_observed
    assert result.active_transition_produced
    assert result.transition_count == 1
    assert result.env_actions == 1
    assert result.hypothesis_remains_unresolved_until_observation
    assert result.trace_support_counted_as_proof is False
    assert result.wrong_confirmations == 0

    assert result.experiment is not None
    assert result.experiment.action.name == "ACTION6"
    assert result.experiment.prediction_families == ("relation", "relation")
    assert result.experiment.has_divergent_predictions

    assert any(
        revision.key == "relation::ACTION6::same_shape::colors9_8::preserved"
        and revision.status_after == HypothesisStatus.CONFIRMED
        for revision in result.revisions
    )
    assert any(
        revision.key == "relation::ACTION6::same_shape::colors9_8::broken"
        and revision.status_after == HypothesisStatus.REFUTED
        for revision in result.revisions
    )
