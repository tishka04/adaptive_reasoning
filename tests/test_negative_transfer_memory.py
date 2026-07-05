"""A20 negative transfer memory tests."""

from __future__ import annotations

from theory.active_transfer_validation import run_active_transfer_validation
from theory.epistemic_metrics import HypothesisStatus
from theory.generic_discriminating_experiment_designer import DiscriminatingPrediction
from theory.negative_transfer_memory import (
    NegativeTransferRecord,
    NegativeTransferMemory,
    apply_negative_transfer_memory,
    build_negative_transfer_records,
    transfer_context_signature_from_prediction,
)
from theory.relation_transfer import RelationTransferPrior, apply_relation_transfer_priors


def test_negative_transfer_memory_downweights_same_target_context_only():
    source_key = "relation::ACTION6::same_shape::colors9_8::preserved"
    prior = RelationTransferPrior(
        source_game_id="ft09-0d8bbf25",
        source_key=source_key,
        predicate="same_shape",
        outcome="preserved",
        weight=10.0,
    )
    same_context = DiscriminatingPrediction(
        "relation::ACTION3::same_shape::colors10_3::preserved",
        "ACTION3",
        10,
        3,
        family="relation",
        predicate="same_shape",
        predicted_outcome="preserved",
    )
    other_context = DiscriminatingPrediction(
        "relation::ACTION3::same_shape::colors3_10::preserved",
        "ACTION3",
        3,
        10,
        family="relation",
        predicate="same_shape",
        predicted_outcome="preserved",
    )
    transferred = apply_relation_transfer_priors(
        [same_context, other_context],
        [prior],
    )
    memory = NegativeTransferMemory()
    memory.add(
        NegativeTransferRecord(
            source_relation=source_key,
            target_game="bp35-0a0ad940",
            target_context_signature=transfer_context_signature_from_prediction(
                transferred[0],
                target_game="bp35-0a0ad940",
            ),
            tested_hypothesis=transferred[0].key,
            observed_outcome="absent",
            weight_delta=10.0,
        )
    )

    adjusted_same_game = apply_negative_transfer_memory(
        transferred,
        memory,
        target_game="bp35-0a0ad940",
    )
    adjusted_other_game = apply_negative_transfer_memory(
        transferred,
        memory,
        target_game="dc22-4c9bff3e",
    )

    assert adjusted_same_game[0].epistemic_prior == 0.0
    assert adjusted_same_game[0].status == HypothesisStatus.UNRESOLVED
    assert adjusted_same_game[0].prior_counted_as_proof is False
    assert adjusted_same_game[1].epistemic_prior == 10.0

    assert adjusted_other_game[0].epistemic_prior == 10.0
    assert adjusted_other_game[1].epistemic_prior == 10.0


def test_negative_transfer_records_from_a19_keep_source_relation_intact():
    result = run_active_transfer_validation()

    records = build_negative_transfer_records(
        target_game=result.target_game_id,
        selected_predictions=result.selected_predictions_before_observation,
        revisions=result.revisions,
    )

    assert result.error == ""
    assert result.source_result is not None
    assert any(
        revision.key == "relation::ACTION6::same_shape::colors9_8::preserved"
        and revision.status_after == HypothesisStatus.CONFIRMED
        for revision in result.source_result.revisions
    )

    assert records
    record = records[0]
    assert record.source_relation == "relation::ACTION6::same_shape::colors9_8::preserved"
    assert record.target_game == "bp35-0a0ad940"
    assert record.tested_hypothesis == (
        "relation::ACTION3::same_shape::colors10_3::preserved"
    )
    assert record.observed_outcome == "absent"
    assert record.effect == "downweight_same_transfer_context"

    memory = NegativeTransferMemory(records=list(records))
    adjusted = apply_negative_transfer_memory(
        result.selected_predictions_before_observation,
        memory,
        target_game=result.target_game_id,
    )
    original_prior = result.selected_predictions_before_observation[0].epistemic_prior

    assert original_prior > 0.0
    assert adjusted[0].epistemic_prior < original_prior
    assert adjusted[0].status == HypothesisStatus.UNRESOLVED
    assert adjusted[0].prior_counted_as_proof is False
    assert result.wrong_confirmations == 0
