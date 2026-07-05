"""A18 relation transfer tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from theory.cross_game_correspondence_discovery import discover_cross_game_correspondences
from theory.epistemic_metrics import HypothesisStatus
from theory.generic_discriminating_experiment_designer import (
    DiscriminatingPrediction,
    GenericDiscriminatingExperimentDesigner,
)
from theory.non_ar25_multi_family_experiment import MultiFamilyHypothesisRevision
from theory.relation_transfer import (
    apply_relation_transfer_priors,
    extract_relation_transfer_priors,
    run_relation_transfer,
)


@dataclass(frozen=True)
class _FakeAction:
    name: str
    raw_action: str
    action_args: dict[str, int]


def test_relation_transfer_adds_prior_to_target_candidate_without_confirmation():
    source_revision = MultiFamilyHypothesisRevision(
        key="relation::ACTION6::same_shape::colors9_8::preserved",
        family="relation",
        predicate="same_shape",
        predicted_outcome="preserved",
        observed_outcome="preserved",
        status_after=HypothesisStatus.CONFIRMED,
    )
    target = discover_cross_game_correspondences(
        Path("human_traces/bp35-0a0ad940.20260615-174246.steps.jsonl"),
        game_id="bp35-0a0ad940",
        min_pixel_support=1,
        top_k=20,
    )

    transfer = run_relation_transfer(
        source_game_id="ft09-0d8bbf25",
        source_revisions=[source_revision],
        target_game_id=target.game_id,
        target_candidates=target.candidates,
        prior_weight=7.0,
    )

    assert transfer.source_game_id == "ft09-0d8bbf25"
    assert transfer.target_game_id == "bp35-0a0ad940"
    assert transfer.priors
    assert transfer.priors[0].predicate == "same_shape"
    assert transfer.priors[0].outcome == "preserved"
    assert transfer.priors[0].counted_as_proof is False

    assert transfer.transferred_count > 0
    assert transfer.transferred_but_unconfirmed
    assert transfer.confirmed_target_count == 0
    assert transfer.prior_counted_as_proof is False
    assert any(
        prediction.predicate_name == "same_shape"
        and prediction.outcome == "preserved"
        and prediction.status == HypothesisStatus.UNRESOLVED
        and prediction.epistemic_prior == 7.0
        for prediction in transfer.transferred_predictions
    )

    assert transfer.records
    assert all(record.status == HypothesisStatus.UNRESOLVED for record in transfer.records)
    assert all(record.support == 0 for record in transfer.records)
    assert all(record.experiments_spent == 0 for record in transfer.records)


def test_relation_transfer_prior_changes_designer_order_but_not_status():
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[1, 1] = 4
    grid[2, 2] = 9
    actions = [
        _FakeAction("ACTION6", "adjacent_action", {"x": 1, "y": 1}),
        _FakeAction("ACTION6", "same_shape_action", {"x": 2, "y": 2}),
    ]
    predictions = [
        DiscriminatingPrediction(
            "relation::ACTION6::adjacent_to::colors4_5::preserved",
            "ACTION6",
            4,
            5,
            family="relation",
            predicate="adjacent_to",
            predicted_outcome="preserved",
            support=10,
        ),
        DiscriminatingPrediction(
            "relation::ACTION6::adjacent_to::colors4_5::broken",
            "ACTION6",
            4,
            5,
            family="relation",
            predicate="adjacent_to",
            predicted_outcome="broken",
            support=10,
        ),
        DiscriminatingPrediction(
            "relation::ACTION6::same_shape::colors9_8::preserved",
            "ACTION6",
            9,
            8,
            family="relation",
            predicate="same_shape",
            predicted_outcome="preserved",
            support=1,
        ),
        DiscriminatingPrediction(
            "relation::ACTION6::same_shape::colors9_8::broken",
            "ACTION6",
            9,
            8,
            family="relation",
            predicate="same_shape",
            predicted_outcome="broken",
            support=1,
        ),
    ]
    designer = GenericDiscriminatingExperimentDesigner()

    without_transfer = designer.design(
        hypotheses=predictions,
        live_grid=grid,
        available_actions=actions,
        preferred_family="relation",
    )
    assert without_transfer is not None
    assert without_transfer.action.raw_action == "adjacent_action"
    assert without_transfer.epistemic_prior == 0.0

    source_revision = MultiFamilyHypothesisRevision(
        key="relation::ACTION6::same_shape::colors9_8::preserved",
        family="relation",
        predicate="same_shape",
        predicted_outcome="preserved",
        observed_outcome="preserved",
        status_after=HypothesisStatus.CONFIRMED,
    )
    priors = extract_relation_transfer_priors(
        [source_revision],
        source_game_id="ft09-0d8bbf25",
        prior_weight=100.0,
    )
    transferred = apply_relation_transfer_priors(predictions, priors)

    with_transfer = designer.design(
        hypotheses=transferred,
        live_grid=grid,
        available_actions=actions,
        preferred_family="relation",
    )

    assert with_transfer is not None
    assert with_transfer.action.raw_action == "same_shape_action"
    assert with_transfer.epistemic_prior == 100.0
    assert with_transfer.prior_source_keys == (
        "relation::ACTION6::same_shape::colors9_8::preserved",
    )

    transferred_same_shape = [
        prediction
        for prediction in transferred
        if prediction.predicate_name == "same_shape"
    ]
    assert transferred_same_shape
    assert all(
        prediction.status == HypothesisStatus.UNRESOLVED
        for prediction in transferred_same_shape
    )
    assert all(
        prediction.prior_counted_as_proof is False
        for prediction in transferred_same_shape
    )
