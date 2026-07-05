"""A14 discriminating non-ar25 active experiment tests."""

from __future__ import annotations

from pathlib import Path

from theory.epistemic_metrics import HypothesisStatus
from theory.non_ar25_discriminating_experiment import (
    run_non_ar25_discriminating_experiment,
)


def test_ft09_discriminating_experiment_separates_competing_hypotheses():
    result = run_non_ar25_discriminating_experiment(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
    )

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.trace_support_counted_as_proof is False
    assert result.active_transition_non_ar25
    assert result.transition_count == 1
    assert result.env_actions == 1

    assert result.competing_hypothesis_count >= 2
    assert result.selected_experiment_has_divergent_predictions
    assert result.revision_differs_across_hypotheses
    assert result.wrong_confirmations == 0
    assert result.score is not None
    assert result.score.wrong_confirmations == 0

    assert result.experiment is not None
    assert result.experiment.action.name == "ACTION6"
    assert result.experiment.action.action_args["x"] is not None
    assert result.experiment.action.action_args["y"] is not None
    assert result.experiment.divergence_reason == "same_source_different_target"
    assert result.experiment.competing_keys == (
        "correspondence::ACTION6::modifies::colors9_8",
        "correspondence::ACTION6::modifies::colors9_12",
    )
    assert result.experiment.predicted_pairs == ((9, 8), (9, 12))

    revisions = {revision.key: revision for revision in result.revisions}
    confirmed = revisions["correspondence::ACTION6::modifies::colors9_8"]
    refuted = revisions["correspondence::ACTION6::modifies::colors9_12"]

    assert confirmed.status_before == HypothesisStatus.UNRESOLVED
    assert confirmed.status_after == HypothesisStatus.CONFIRMED
    assert confirmed.reason == "observed_predicted_pair"
    assert confirmed.pair_change_pixels >= 4
    assert "source_target_color_transform" in confirmed.observed_predicates
    assert confirmed.trace_support_counted_as_proof is False

    assert refuted.status_before == HypothesisStatus.UNRESOLVED
    assert refuted.status_after == HypothesisStatus.REFUTED
    assert refuted.reason == "observed_divergent_pair"
    assert refuted.pair_change_pixels == 0
    assert refuted.trace_support_counted_as_proof is False

    assert result.transition_update is not None
    record = result.transition_update.record
    assert record.action.name == "ACTION6"
    assert record.action.x is not None
    assert record.action.y is not None
    assert record.diff.num_changed > 0
