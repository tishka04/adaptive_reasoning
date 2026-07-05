"""A19 active validation of transferred relation priors tests."""

from __future__ import annotations

from theory.epistemic_metrics import HypothesisStatus
from theory.active_transfer_validation import run_active_transfer_validation


def test_active_transfer_validation_uses_prior_then_revises_from_target_observation():
    result = run_active_transfer_validation()

    assert result.error == ""
    assert result.source_game_id == "ft09-0d8bbf25"
    assert result.target_game_id == "bp35-0a0ad940"

    assert result.transferred_prior_used
    assert result.transfer_changed_experiment_order
    assert result.active_transition_target_game
    assert result.hypothesis_remains_unresolved_until_observation
    assert result.local_revision_after_observation
    assert result.transition_count == 1
    assert result.env_actions == 1
    assert result.wrong_confirmations == 0

    assert result.source_result is not None
    assert any(
        revision.family == "relation"
        and revision.predicate == "same_shape"
        and revision.observed_outcome == "preserved"
        and revision.status_after == HypothesisStatus.CONFIRMED
        for revision in result.source_result.revisions
    )

    assert result.transfer is not None
    assert result.transfer.transferred_but_unconfirmed
    assert result.transfer.confirmed_target_count == 0
    assert result.transfer.prior_counted_as_proof is False
    assert result.transfer.records
    assert all(record.status == HypothesisStatus.UNRESOLVED for record in result.transfer.records)
    assert all(record.support == 0 for record in result.transfer.records)
    assert all(record.experiments_spent == 0 for record in result.transfer.records)

    assert result.baseline_experiment is not None
    assert result.baseline_experiment.epistemic_prior == 0.0
    assert result.baseline_experiment.competing_keys != result.experiment.competing_keys

    assert result.experiment is not None
    assert result.experiment.epistemic_prior > 0.0
    assert result.experiment.prediction_families == ("relation", "relation")
    assert result.experiment.divergence_reason == "same_source_different_same_shape"
    assert result.experiment.prior_source_keys == (
        "relation::ACTION6::same_shape::colors9_8::preserved",
    )
    assert "same_shape" in result.experiment.competing_keys[0]

    assert result.selected_predictions_before_observation
    assert all(
        prediction.status == HypothesisStatus.UNRESOLVED
        for prediction in result.selected_predictions_before_observation
    )
    assert all(
        prediction.prior_counted_as_proof is False
        for prediction in result.selected_predictions_before_observation
    )

    assert result.revisions
    assert all(revision.family == "relation" for revision in result.revisions)
    assert all(revision.predicate == "same_shape" for revision in result.revisions)
    assert all(
        revision.status_after in {HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED}
        for revision in result.revisions
    )
    assert any(
        revision.status_after == HypothesisStatus.REFUTED
        for revision in result.revisions
    )
    assert all(revision.reason.startswith("observed_") for revision in result.revisions)

    assert result.score is not None
    assert result.score.wrong_confirmations == 0
    assert result.transition_update is not None
    assert result.transition_update.record.action.name == result.experiment.action.name
