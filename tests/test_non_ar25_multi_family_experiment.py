"""A16 active non-ar25 multi-family experiment tests."""

from __future__ import annotations

from pathlib import Path

from theory.epistemic_metrics import HypothesisStatus
from theory.non_ar25_multi_family_experiment import (
    run_non_ar25_multi_family_experiment,
)


def test_ft09_multi_family_experiment_discriminates_effect_scope():
    result = run_non_ar25_multi_family_experiment(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
        preferred_family="effect_scope",
    )

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.trace_support_counted_as_proof is False
    assert result.active_transition_non_ar25
    assert result.transition_count == 1
    assert result.env_actions == 1

    assert "color_transform" in result.supported_prediction_families
    assert "effect_scope" in result.supported_prediction_families
    assert len(result.supported_prediction_families) >= 2

    assert result.experiment is not None
    assert result.experiment.action.name == "ACTION6"
    assert result.experiment.prediction_families == (
        "effect_scope",
        "effect_scope",
    )
    assert result.experiment.predicted_outcomes == ("local", "global")
    assert result.experiment.divergence_reason == (
        "same_source_different_effect_scope"
    )
    assert result.selected_experiment_has_divergent_predictions

    revisions = {revision.predicted_outcome: revision for revision in result.revisions}
    assert revisions["local"].status_after == HypothesisStatus.CONFIRMED
    assert revisions["local"].observed_outcome == "local"
    assert revisions["local"].reason == "observed_predicted_effect_scope"
    assert revisions["global"].status_after == HypothesisStatus.REFUTED
    assert revisions["global"].observed_outcome == "local"
    assert revisions["global"].reason == "observed_divergent_effect_scope"

    assert result.revision_differs_across_hypotheses
    assert result.wrong_confirmations == 0
    assert result.score is not None
    assert result.score.wrong_confirmations == 0

    assert result.transition_update is not None
    record = result.transition_update.record
    assert record.action.name == "ACTION6"
    assert record.action.x is not None
    assert record.action.y is not None
    assert record.diff.num_changed > 0


def test_ft09_multi_family_experiment_revises_relation_predicate():
    result = run_non_ar25_multi_family_experiment(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
        preferred_family="relation",
    )

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.trace_support_counted_as_proof is False
    assert result.active_transition_non_ar25
    assert result.transition_count == 1
    assert result.env_actions == 1

    assert "relation" in result.supported_prediction_families
    assert result.experiment is not None
    assert result.experiment.action.name == "ACTION6"
    assert result.experiment.prediction_families == ("relation", "relation")
    assert result.experiment.predicted_outcomes == ("preserved", "broken")
    assert result.experiment.predicted_pairs == ((9, 8), (9, 8))
    assert result.experiment.divergence_reason == "same_source_different_same_shape"
    assert result.selected_experiment_has_divergent_predictions

    revisions = {revision.predicted_outcome: revision for revision in result.revisions}
    assert revisions["preserved"].family == "relation"
    assert revisions["preserved"].predicate == "same_shape"
    assert revisions["preserved"].status_after == HypothesisStatus.CONFIRMED
    assert revisions["preserved"].observed_outcome == "preserved"
    assert revisions["preserved"].reason == "observed_predicted_relation"

    assert revisions["broken"].family == "relation"
    assert revisions["broken"].predicate == "same_shape"
    assert revisions["broken"].status_after == HypothesisStatus.REFUTED
    assert revisions["broken"].observed_outcome == "preserved"
    assert revisions["broken"].reason == "observed_divergent_relation"

    assert result.revision_differs_across_hypotheses
    assert result.wrong_confirmations == 0
    assert result.score is not None
    assert result.score.wrong_confirmations == 0

    assert result.transition_update is not None
    record = result.transition_update.record
    assert record.action.name == "ACTION6"
    assert record.action.x is not None
    assert record.action.y is not None
    assert record.diff.num_changed > 0
