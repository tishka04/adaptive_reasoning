"""A21 closed-loop negative transfer tests."""

from __future__ import annotations

from theory.closed_loop_negative_transfer import run_closed_loop_negative_transfer
from theory.epistemic_metrics import HypothesisStatus


def test_closed_loop_negative_transfer_changes_second_experiment():
    result = run_closed_loop_negative_transfer()

    assert result.error == ""
    assert result.negative_memory_used
    assert result.repeated_failed_context_not_selected
    assert result.alternative_experiment_selected
    assert result.source_relation_remains_confirmed
    assert result.source_relation_status == HypothesisStatus.CONFIRMED
    assert result.wrong_confirmations == 0

    assert result.first_attempt is not None
    assert result.first_attempt.transferred_prior_used
    assert result.first_attempt.local_revision_after_observation
    assert result.first_attempt.wrong_confirmations == 0

    assert result.negative_memory.records
    record = result.negative_memory.records[0]
    assert record.effect == "downweight_same_transfer_context"
    assert record.source_relation == (
        "relation::ACTION6::same_shape::colors9_8::preserved"
    )
    assert record.target_game == "bp35-0a0ad940"
    assert record.tested_hypothesis == (
        "relation::ACTION3::same_shape::colors10_3::preserved"
    )
    assert record.observed_outcome == "absent"

    assert result.first_attempt.experiment is not None
    assert result.second_experiment is not None
    assert result.first_attempt.experiment.competing_keys != (
        result.second_experiment.competing_keys
    )
    assert result.repeated_context not in result.second_experiment.competing_keys
    assert result.second_experiment.epistemic_prior > 0.0
    assert result.second_experiment.prior_source_keys == (
        "relation::ACTION6::same_shape::colors9_8::preserved",
    )
    assert result.second_experiment.prediction_families == ("relation", "relation")
    assert result.second_experiment.divergence_reason == (
        "same_source_different_same_shape"
    )

    assert result.second_selected_predictions
    assert all(
        prediction.status == HypothesisStatus.UNRESOLVED
        for prediction in result.second_selected_predictions
    )
    assert all(
        prediction.prior_counted_as_proof is False
        for prediction in result.second_selected_predictions
    )
